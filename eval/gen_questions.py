"""
LLM-assisted candidate generator for the retrieval eval fixture.

Samples nodes from the SQLite index (stratified: at least one per document,
text length in a sensible answerable range) and asks the LLM to draft one
paraphrased, self-contained question answerable only from each node. Writes
``eval/candidates.jsonl`` for human curation — keep the good ones, drop the
ambiguous ones, and add cross-doc / paraphrase questions by hand before
promoting to ``eval/questions.jsonl`` (the fixture run_eval.py reads).

Why curation matters: generated questions tend to echo passage vocabulary,
which flatters lexical BM25 and understates semantic recall. The "paraphrase,
don't copy exact phrases" instruction reduces that, but a human pass is what
makes the numbers trustworthy.

Requires the index to be built WITH node text (the trees here have it). Build:
    python3 build_index.py --folder ./results

Usage (defaults route to vLLM 'rag-llm' via VLLM_BASE_URL in .env):
    python3 eval/gen_questions.py --folder ./results --num 30
    python3 eval/gen_questions.py --folder ./results --model vllm/rag-llm --num 40
"""
import argparse
import asyncio
import json
import logging
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retrieve_pageindex as rp  # noqa: E402
from pageindex.index_store import DEFAULT_DB_NAME  # noqa: E402
from pageindex.utils import extract_json  # noqa: E402

logger = logging.getLogger("eval.gen_questions")

SYSTEM_PROMPT = (
    "You write evaluation questions for a document-retrieval system. Given one "
    "passage from a document, you write a single specific, factual question that "
    "can be answered ONLY using that passage."
)

USER_TEMPLATE = """Document: {doc_id}
Section: {section}

Passage:
{text}

Write ONE specific factual question answerable only from this passage. Requirements:
- Self-contained: name the specific subject, method, or term so the question makes \
sense on its own (the retrieval system sees only the question, not this passage).
- Paraphrase: do NOT copy long exact phrases from the passage; reword where you can \
so the question is not trivially keyword-matchable.{extra_rules}
- Exactly one question, no preamble or explanation.

Respond with JSON only: {{"question": "..."}}"""

# Appended in --paraphrase mode: pushes overlap down so the question tests
# semantic (not lexical) retrieval. The post-hoc --max-overlap filter is the
# hard guarantee; this just makes more candidates pass it.
HARD_PARAPHRASE_RULE = (
    "\n- AGGRESSIVELY reword: replace the passage's distinctive nouns and verbs with "
    "synonyms or short descriptions, avoid reusing its technical terms verbatim when a "
    "paraphrase exists, and don't quote its numbers or proper names unless essential. "
    "Aim for a question whose words barely overlap the passage while still pointing "
    "unambiguously to its content."
)

_OVERLAP_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been being this that these those "
    "what which who whom how why when where does do did how it its as by with from at into "
    "than then so such can could would should may might will shall has have had not no".split()
)


def word_overlap(question: str, passage: str) -> float:
    """Fraction of the question's distinct content words that also appear in the
    passage (stopwords removed). A weak circularity proxy — a self-contained
    question must still name its subject, so some overlap is unavoidable; this is
    informational, not a hard gate. Prefer BM25-difficulty for fixture selection."""
    q = {w for w in _OVERLAP_RE.findall((question or "").lower()) if w not in _STOPWORDS}
    if not q:
        return 0.0
    p = set(_OVERLAP_RE.findall((passage or "").lower()))
    return len(q & p) / len(q)


def open_db(folder: Path, db_path: str | None) -> sqlite3.Connection:
    db = Path(db_path).expanduser().resolve() if db_path else folder / DEFAULT_DB_NAME
    if not db.is_file():
        raise SystemExit(
            f"Index DB not found: {db}\n"
            f"Build it first: python3 build_index.py --folder {folder}"
        )
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def sample_nodes(conn, num: int, per_doc_min: int, min_chars: int, max_chars: int,
                 rng: Random) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT doc_id, node_id, title, breadcrumb, text, text_len "
        "FROM nodes WHERE node_id IS NOT NULL AND text_len BETWEEN ? AND ? "
        "AND text != ''",
        (min_chars, max_chars),
    ).fetchall()
    if not rows:
        raise SystemExit(
            f"No nodes with text_len in [{min_chars}, {max_chars}] and non-empty text. "
            f"Was the index built with node text?"
        )
    by_doc: dict[str, list] = defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)
    for v in by_doc.values():
        rng.shuffle(v)
    docs = sorted(by_doc)
    rng.shuffle(docs)

    selected: list = []
    used: set[tuple] = set()
    for d in docs:  # ensure per-doc coverage first
        for r in by_doc[d][:per_doc_min]:
            selected.append(r)
            used.add((d, r["node_id"]))
    remaining = [r for d in docs for r in by_doc[d] if (d, r["node_id"]) not in used]
    rng.shuffle(remaining)
    for r in remaining:
        if len(selected) >= num:
            break
        selected.append(r)
    rng.shuffle(selected)
    return selected[:num]


def parse_question(raw: str) -> str | None:
    obj = extract_json(raw)
    if isinstance(obj, dict):
        q = obj.get("question")
        if isinstance(q, str) and q.strip():
            return q.strip()
    # Fallback: first non-empty line, stripped of a "Question:" prefix / quotes.
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("question:"):
            line = line[len("question:"):].strip()
        return line.strip().strip('"').strip()
    return None


async def generate(selected: list, provider: str, model: str, base_url: str | None,
                   api_key: str | None, text_cap: int, max_tokens: int,
                   concurrency: int, tags: list[str], hard_paraphrase: bool,
                   id_prefix: str) -> list[dict]:
    sem = asyncio.Semaphore(max(1, concurrency))
    results: list[dict | None] = [None] * len(selected)
    extra_rules = HARD_PARAPHRASE_RULE if hard_paraphrase else ""

    async def one(i: int, row):
        section = row["breadcrumb"] or row["title"] or ""
        passage = (row["text"] or "")
        user = USER_TEMPLATE.format(
            doc_id=row["doc_id"], section=section, text=passage[:text_cap],
            extra_rules=extra_rules,
        )
        async with sem:
            try:
                raw = await rp._synthesize_answer(
                    provider, model, base_url, api_key, SYSTEM_PROMPT, user, max_tokens
                )
            except Exception:
                logger.exception("generation failed for %s/%s", row["doc_id"], row["node_id"])
                return
        question = parse_question(raw)
        if not question:
            logger.warning("empty question for %s/%s; skipping", row["doc_id"], row["node_id"])
            return
        results[i] = {
            "id": f"{id_prefix}{i:03d}",
            "question": question,
            "relevant": [{"doc_id": row["doc_id"], "node_id": row["node_id"]}],
            "tags": list(tags),
            "overlap": round(word_overlap(question, passage), 3),
            "source_preview": passage[:300],
        }

    await asyncio.gather(*(one(i, r) for i, r in enumerate(selected)))
    return [r for r in results if r is not None]


def main():
    parser = argparse.ArgumentParser(description="Generate retrieval-eval question candidates.")
    parser.add_argument("--folder", default="./results",
                        help="Folder of PageIndex trees (default: ./results).")
    parser.add_argument("--db", default=None, help="Index DB path (default: <folder>/.pageindex_index.db).")
    parser.add_argument("--out", default=str(ROOT / "eval" / "candidates.jsonl"),
                        help="Output JSONL of candidates (default: eval/candidates.jsonl).")
    parser.add_argument("--num", type=int, default=30, help="Number of candidates to generate.")
    parser.add_argument("--per-doc-min", type=int, default=1, help="Minimum nodes sampled per document.")
    parser.add_argument("--min-chars", type=int, default=800, help="Min node text length to sample.")
    parser.add_argument("--max-chars", type=int, default=8000, help="Max node text length to sample.")
    parser.add_argument("--text-cap", type=int, default=4000, help="Chars of node text put in the prompt.")
    parser.add_argument("--max-tokens", type=int, default=256, help="LLM output cap per question.")
    parser.add_argument("--concurrency", type=int, default=8, help="Max in-flight LLM calls.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (reproducible).")
    parser.add_argument("--paraphrase", action="store_true",
                        help="Aggressive-paraphrase prompt + tag 'paraphrase'; defaults --max-overlap "
                             "to 0.35 and id prefix to 'p'. Use to build the low-overlap subset that "
                             "fairly tests semantic retrieval (BM25 can't win on exact tokens).")
    parser.add_argument("--max-overlap", type=float, default=None,
                        help="Drop candidates whose question/source word-overlap exceeds this "
                             "(0-1). Default: off (1.0), or 0.35 in --paraphrase mode.")
    parser.add_argument("--tags", default=None,
                        help="Comma-separated tags for each candidate "
                             "(default 'single-doc,generated', or 'paraphrase,single-doc' in --paraphrase mode).")
    parser.add_argument("--provider", default="auto", help="auto|openai|vllm|ollama|litellm.")
    parser.add_argument("--model", default="vllm/rag-llm", help="Model (prefix routes provider).")
    parser.add_argument("--base-url", default=None, help="Override server base URL.")
    parser.add_argument("--verbose", action="store_true", help="INFO logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    provider, bare_model = rp.resolve_provider(args.model, args.provider)
    # Mirror retrieve_pageindex.main(): pull base_url from <PROVIDER>_BASE_URL when unset.
    base_url = args.base_url or os.getenv(f"{provider.upper()}_BASE_URL")
    if provider == "openai" and not base_url:
        base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv(f"{provider.upper()}_API_KEY")

    conn = open_db(folder, args.db)
    try:
        selected = sample_nodes(conn, args.num, args.per_doc_min,
                                args.min_chars, args.max_chars, Random(args.seed))
    finally:
        conn.close()
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    else:
        tags = ["paraphrase", "single-doc"] if args.paraphrase else ["single-doc", "generated"]
    id_prefix = "p" if args.paraphrase else "g"

    print(f"Sampled {len(selected)} node(s) across "
          f"{len({r['doc_id'] for r in selected})} document(s). "
          f"Generating via provider={provider} model={bare_model} "
          f"base_url={base_url or rp.DEFAULT_BASE_URLS.get(provider, 'default')} "
          f"(paraphrase={args.paraphrase}, max_overlap={args.max_overlap or 'off'}, tags={tags}) ...")

    candidates = asyncio.run(generate(
        selected, provider, bare_model, base_url, api_key,
        args.text_cap, args.max_tokens, args.concurrency,
        tags=tags, hard_paraphrase=args.paraphrase, id_prefix=id_prefix,
    ))

    # Write all candidates (lowest overlap first); only drop if --max-overlap is set
    # explicitly. Overlap is a weak proxy — a self-contained question must name its
    # subject, so some topic-vocabulary overlap is unavoidable. Prefer selecting the
    # final fixture by BM25 difficulty (does BM25 already rank the gold node #1?).
    candidates.sort(key=lambda c: c["overlap"])
    if args.max_overlap is not None:
        before = len(candidates)
        candidates = [c for c in candidates if c["overlap"] <= args.max_overlap]
        print(f"Dropped {before - len(candidates)} candidate(s) over overlap {args.max_overlap}.")
    if candidates:
        ov = [c["overlap"] for c in candidates]
        print(f"Overlap min/mean/max = {min(ov):.2f}/{sum(ov)/len(ov):.2f}/{max(ov):.2f}")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(candidates)} candidate(s) to {out}")
    print("Next: review the file, delete weak/ambiguous questions, then save the "
          "keepers (plus hand-written cross-doc questions) as eval/questions.jsonl.")


if __name__ == "__main__":
    main()
