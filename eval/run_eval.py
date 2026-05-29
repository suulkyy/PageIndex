"""
Retrieval-only evaluation harness for the PageIndex SQLite index.

Measures *retrieval* quality in isolation from synthesis: given a fixture of
question -> relevant node(s), it runs the real retrieval pipeline
(``retrieve_pageindex._hybrid_select`` over an ``IndexStore``) and reports
recall@k, MRR, and document-level recall@k. No LLM call, so the BM25 path runs
fully offline; vector / rerank columns need the :8001 / :8002 servers.

This is the baseline tool for the planned retrieval changes (embed body text,
field-weighted BM25, passage-level granularity): run it before and after each
change to see the delta instead of spot-checking.

Usage:
    # single retriever (bm25 is offline)
    python3 eval/run_eval.py --folder ./results --retriever bm25

    # side-by-side comparison of all retrievers (vector/rerank need the servers)
    python3 eval/run_eval.py --folder ./results --compare

    # custom fixture
    python3 eval/run_eval.py --folder ./results --fixture eval/questions.jsonl

Fixture format (JSONL, one object per line):
    {"id": "q001",
     "question": "...",
     "relevant": [{"doc_id": "2305.13245", "node_id": "0042"}],
     "tags": ["single-doc"]}

  - ``relevant`` is a list -> supports multi-hop / cross-doc gold.
  - ``node_id`` may be unpadded; it is canonicalized against the index.
  - An entry with ``node_id`` null/absent contributes only to doc-level recall.
  - ``tags`` is optional; metrics are also broken down per tag.

Matching is at node granularity and forward-compatible with sub-node passages:
a future passage candidate carries its parent ``node_id``, so a passage hit
still counts for its node here.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retrieve_pageindex as rp  # noqa: E402  (path set above)

logger = logging.getLogger("eval.run_eval")

K_VALUES = (1, 5, 10, 20)

# (label, retriever, rerank_enabled)
COMPARE_CONFIGS = (
    ("bm25", "bm25", False),
    ("vector", "vector", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
)


# ── fixture loading ──────────────────────────────────────────────────────────

def load_fixture(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(
            f"Fixture not found: {path}\n"
            f"Generate candidates with eval/gen_questions.py, curate them into this file."
        )
    questions: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("skipping %s:%d (bad JSON: %s)", path.name, lineno, e)
                continue
            if not obj.get("question") or not obj.get("relevant"):
                logger.warning("skipping %s:%d (missing question/relevant)", path.name, lineno)
                continue
            obj.setdefault("id", f"q{lineno:03d}")
            obj.setdefault("tags", [])
            questions.append(obj)
    if not questions:
        raise SystemExit(f"No usable questions in {path}")
    return questions


def resolve_gold(store, questions: list[dict]) -> tuple[list[dict], int]:
    """Canonicalize each relevant node_id against the index. Drops gold entries
    whose node can't be resolved (stale fixture) and warns. Returns
    (prepared_questions, unresolved_count). Each prepared question gains
    ``gold_nodes`` = set[(doc_id, node_id)] and ``gold_docs`` = set[doc_id]."""
    prepared: list[dict] = []
    unresolved = 0
    for q in questions:
        gold_nodes: set[tuple] = set()
        gold_docs: set[str] = set()
        for rel in q["relevant"]:
            doc_id = rel.get("doc_id")
            if not doc_id:
                continue
            gold_docs.add(doc_id)
            node_id = rel.get("node_id")
            if node_id in (None, ""):
                continue
            canonical = store._resolve_node_id(doc_id, node_id)
            if canonical is None:
                unresolved += 1
                logger.warning("unresolved gold node %s/%s in %s", doc_id, node_id, q["id"])
                continue
            gold_nodes.add((doc_id, canonical))
        if not gold_nodes and not gold_docs:
            logger.warning("question %s has no resolvable gold; skipping", q["id"])
            continue
        prepared.append({**q, "gold_nodes": gold_nodes, "gold_docs": gold_docs})
    return prepared, unresolved


# ── per-question ranking + aggregation ───────────────────────────────────────

def _ranked_keys(candidates: list[dict]) -> list[tuple]:
    """(doc_id, node_id) in rank order, de-duped (first occurrence wins)."""
    seen, out = set(), []
    for c in candidates:
        key = (c.get("doc_id"), c.get("node_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _question_ranks(candidates: list[dict], gold_nodes: set, gold_docs: set) -> dict:
    """Return 1-based ranks: node_ranks[gold_node] and doc_ranks[gold_doc]
    (None when not retrieved)."""
    ranked = _ranked_keys(candidates)
    node_ranks = {g: None for g in gold_nodes}
    for i, key in enumerate(ranked, 1):
        if key in node_ranks and node_ranks[key] is None:
            node_ranks[key] = i
    doc_ranks = {d: None for d in gold_docs}
    for i, (doc_id, _) in enumerate(ranked, 1):
        if doc_id in doc_ranks and doc_ranks[doc_id] is None:
            doc_ranks[doc_id] = i
    return {"node_ranks": node_ranks, "doc_ranks": doc_ranks}


def aggregate(per_question: list[dict]) -> dict:
    """Macro-average metrics over questions. Node metrics (MRR, R@k) average only
    over questions that have node-level gold; doc metrics (DocR@k) over those with
    doc-level gold. This keeps doc-only cross-doc questions (no pinned node) from
    diluting node recall, and vice versa. Metrics with no eligible questions are
    omitted (rendered as '-')."""
    if not per_question:
        return {}
    with_node = [pq for pq in per_question if pq["node_ranks"]]
    with_doc = [pq for pq in per_question if pq["doc_ranks"]]
    metrics: dict[str, float] = {}

    if with_node:
        rr_sum = 0.0
        for pq in with_node:
            found = [r for r in pq["node_ranks"].values() if r is not None]
            rr_sum += 1.0 / min(found) if found else 0.0
        metrics["MRR"] = rr_sum / len(with_node)

    for k in K_VALUES:
        if with_node:
            node_recall = sum(
                sum(1 for r in pq["node_ranks"].values() if r is not None and r <= k)
                / len(pq["node_ranks"])
                for pq in with_node
            )
            metrics[f"R@{k}"] = node_recall / len(with_node)
        if with_doc:
            doc_recall = sum(
                sum(1 for r in pq["doc_ranks"].values() if r is not None and r <= k)
                / len(pq["doc_ranks"])
                for pq in with_doc
            )
            metrics[f"DocR@{k}"] = doc_recall / len(with_doc)
    return metrics


def evaluate_config(store, questions: list[dict], retriever: str, rerank: bool,
                    pool: int) -> tuple[dict, dict, str]:
    """Run one retriever config over all questions. Returns
    (overall_metrics, per_tag_metrics, retriever_used_label)."""
    per_question: list[dict] = []
    used_labels: set[str] = set()
    for q in questions:
        candidates, used = rp._hybrid_select(
            store, q["question"], pool, retriever=retriever,
            rerank_enabled=rerank, verbose=False,
        )
        used_labels.add(used)
        pq = _question_ranks(candidates, q["gold_nodes"], q["gold_docs"])
        pq["tags"] = q.get("tags", [])
        per_question.append(pq)

    overall = aggregate(per_question)
    tags = sorted({t for pq in per_question for t in pq["tags"]})
    per_tag = {
        tag: aggregate([pq for pq in per_question if tag in pq["tags"]])
        for tag in tags
    }
    # If a config silently degraded (e.g. vector with servers down -> bm25),
    # surface the actual retriever(s) that ran.
    used_label = ",".join(sorted(used_labels))
    return overall, per_tag, used_label


# ── reporting ────────────────────────────────────────────────────────────────

METRIC_ORDER = ["MRR"] + [f"R@{k}" for k in K_VALUES] + [f"DocR@{k}" for k in K_VALUES]


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def print_single(label: str, used: str, overall: dict, per_tag: dict, n: int):
    print(f"\n=== retriever: {label}  (ran as: {used})  |  {n} question(s) ===")
    for m in METRIC_ORDER:
        print(f"  {m:<8} {_fmt(overall.get(m))}")
    if per_tag:
        print("  per-tag:")
        for tag, mt in per_tag.items():
            row = "  ".join(f"{m}={_fmt(mt.get(m))}" for m in ("MRR", "R@5", "R@10", "DocR@10"))
            print(f"    [{tag}]  {row}")


def print_compare(results: list[tuple], n: int):
    """results: list of (label, used, overall, per_tag)."""
    labels = [r[0] for r in results]
    col_w = max(13, max(len(l) for l in labels) + 1)
    header = "metric".ljust(10) + "".join(l.rjust(col_w) for l in labels)
    print(f"\n=== retriever comparison  |  {n} question(s) ===")
    print(header)
    print("-" * len(header))
    for m in METRIC_ORDER:
        row = m.ljust(10) + "".join(_fmt(r[2].get(m)).rjust(col_w) for r in results)
        print(row)
    # "ran as" footnote exposes silent degradation (vector -> bm25 when servers down).
    print("\nran as:")
    for label, used, _, _ in results:
        note = "" if used == label or label in used else f"   <- NOTE: differs from requested '{label}'"
        print(f"  {label:<14} {used}{note}")

    all_tags = sorted({t for _, _, _, pt in results for t in pt})
    if all_tags:
        print("\nper-tag R@10:")
        sub = "tag".ljust(16) + "".join(l.rjust(col_w) for l in labels)
        print(sub)
        for tag in all_tags:
            row = tag.ljust(16) + "".join(
                _fmt(r[3].get(tag, {}).get("R@10")).rjust(col_w) for r in results
            )
            print(row)


def main():
    parser = argparse.ArgumentParser(description="Retrieval-only eval over the PageIndex index.")
    parser.add_argument("--folder", default="./results",
                        help="Folder of PageIndex tree JSON files (default: ./results).")
    parser.add_argument("--db", default=None,
                        help="Index DB path (default: <folder>/.pageindex_index.db).")
    parser.add_argument("--fixture", default=str(ROOT / "eval" / "questions.jsonl"),
                        help="Ground-truth JSONL fixture (default: eval/questions.jsonl).")
    parser.add_argument("--retriever", default="bm25",
                        choices=["bm25", "vector", "hybrid"],
                        help="Single-config retriever (ignored with --compare). Default bm25 (offline).")
    parser.add_argument("--rerank", action="store_true",
                        help="Enable cross-encoder rerank for the single-config run (needs :8002).")
    parser.add_argument("--compare", action="store_true",
                        help="Run bm25 / vector / hybrid / hybrid+rerank side by side.")
    parser.add_argument("--pool", type=int, default=50,
                        help="Candidate pool size per query (>= max k). Default 50.")
    parser.add_argument("--verbose", action="store_true", help="INFO logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    pool = max(args.pool, max(K_VALUES))
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    store = rp._open_index_for_folder(folder, args.db)
    if store is None:
        raise SystemExit(
            "Could not open/refresh the index. Build it first: "
            "python3 build_index.py --folder " + str(folder)
        )

    try:
        raw = load_fixture(Path(args.fixture).expanduser())
        questions, unresolved = resolve_gold(store, raw)
        n = len(questions)
        if n == 0:
            raise SystemExit("No questions with resolvable gold nodes — is the fixture stale vs the index?")
        n_node = sum(1 for q in questions if q["gold_nodes"])
        n_doc = sum(1 for q in questions if q["gold_docs"])
        print(f"Loaded {n} question(s) from {args.fixture} "
              f"({n_node} with node gold -> MRR/R@k; {n_doc} with doc gold -> DocR@k)"
              + (f"  [{unresolved} gold node(s) unresolved, excluded]" if unresolved else ""))

        if args.compare:
            results = []
            for label, retriever, rerank in COMPARE_CONFIGS:
                overall, per_tag, used = evaluate_config(store, questions, retriever, rerank, pool)
                results.append((label, used, overall, per_tag))
            print_compare(results, n)
        else:
            label = args.retriever + ("+rerank" if args.rerank else "")
            overall, per_tag, used = evaluate_config(
                store, questions, args.retriever, args.rerank, pool)
            print_single(label, used, overall, per_tag, n)
    finally:
        store.close()


if __name__ == "__main__":
    main()
