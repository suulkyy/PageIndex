"""
Local retrieval over a folder of PageIndex tree JSON files.

Usage:
    # OpenAI / hosted (default)
    python3 retrieve_pageindex.py --folder ./results --question "..."

    # vLLM (OpenAI-compatible server)
    python3 retrieve_pageindex.py --folder ./results --question "..." \\
        --provider vllm --model meta-llama/Llama-3.1-8B-Instruct \\
        --base-url http://localhost:8000/v1

    # Ollama (OpenAI-compatible /v1 endpoint)
    python3 retrieve_pageindex.py --folder ./results --question "..." \\
        --provider ollama --model llama3.1:8b
        # base-url defaults to http://localhost:11434/v1

    # LiteLLM-routed (any provider supported by LiteLLM)
    python3 retrieve_pageindex.py --folder ./results --question "..." \\
        --provider litellm --model anthropic/claude-sonnet-4-6

Provider auto-detection (when --provider auto, default):
    ollama/<name>  → ollama        vllm/<name>    → vllm
    openai/<name>  → openai        litellm/<name> → litellm
    plain name     → openai        provider/<name>→ litellm

Env fallbacks:
    OPENAI_API_KEY / CHATGPT_API_KEY   OpenAI / hosted
    VLLM_BASE_URL, VLLM_API_KEY        vLLM
    OLLAMA_BASE_URL, OLLAMA_API_KEY    Ollama
    OPENAI_BASE_URL                    Override OpenAI base
    PAGEINDEX_RETRIEVE_ENABLE_THINKING vLLM final-answer thinking, default false
    PAGEINDEX_AGENT_MAX_TOKENS         Per-turn output cap. Default 2048, or
                                       8192 when thinking retrieval is enabled.
    PAGEINDEX_RETRIEVE_THINKING_EVIDENCE_MAX_CHARS
                                       Evidence cap for final thinking pass.
    PAGEINDEX_STRUCTURE_SUMMARY_MAX_CHARS
                                       Per-node summary preview in structure tool.
                                       Default 160; 0 removes summaries.
    PAGEINDEX_STRUCTURE_MAX_CHARS      Char budget for get_document_structure
                                       payload. Default 120000 (~30K tokens).
                                       Depth auto-clips until the JSON fits.
    PAGEINDEX_NODE_TEXT_MAX_CHARS      Default window length for get_node_content
                                       when length is unspecified (default 8000).
    PAGEINDEX_NODE_TEXT_WINDOW_MAX_CHARS
                                       Hard cap on a single get_node_content
                                       window even when the agent asks for more.
                                       Default 32000.

Each JSON in --folder must be a tree produced by run_pageindex.py / page_index_main,
i.e. of shape {"doc_name": ..., "doc_description"?: ..., "structure": [...]}.

The retriever runs fully locally:
- No PAGEINDEX_API_KEY (cloud service) required.

The agent is given four tools (plus an explicit answer() tool):
- list_documents(mode='brief'|'full')
- get_document(doc_id)
- get_document_structure(doc_id, max_depth=None, from_node_id=None)
      # depth auto-tuned by node count; auto-clipped to fit a char budget.
      # Older structure outputs are auto-elided from rolling history (F8).
- get_node_content(doc_id, node_id, offset=0, length=None)
      # windowed; response includes total_chars + has_more for pagination.

Generate trees with `--if-add-node-text yes --if-add-node-id yes` for best retrieval.
Without node text, the agent must answer from summaries alone.
"""
import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from pageindex.utils import ConfigLoader

load_dotenv(override=True)

logger = logging.getLogger("retrieve_pageindex")

_TS_RE = re.compile(r"^\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class _Tee:
    """Mirror writes to a stream + log file. Prepends timestamps in the file
    copy for lines that don't already start with one. Terminal copy
    untouched."""
    def __init__(self, stream, file_obj):
        self._stream = stream
        self._file = file_obj
        self._pending = ""
    def write(self, data):
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            buf = self._pending + (data or "")
            while True:
                idx = buf.find("\n")
                if idx == -1:
                    self._pending = buf
                    break
                line = buf[:idx]
                buf = buf[idx + 1:]
                if line and not _TS_RE.match(line):
                    self._file.write(f"[{_ts()}] {line}\n")
                else:
                    self._file.write(line + "\n")
            self._file.flush()
        except Exception:
            pass
        return len(data) if isinstance(data, str) else 0
    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            if self._pending:
                self._file.write(self._pending)
                self._pending = ""
            self._file.flush()
        except Exception:
            pass
    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()
    def fileno(self):
        return self._stream.fileno()
    def __getattr__(self, name):
        return getattr(self._stream, name)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> Path:
    """Configure root logger with timestamped formatter. Tee stdout+stderr
    into the same log file so bare print() and tracebacks land there too.
    Returns resolved log file path."""
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    if log_file:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = logs_dir / f"retrieve_{ts}.log"

    f = open(path, "a", encoding="utf-8", buffering=1)
    f.write(f"\n[{_ts()}] === log start: {path} ===\n")
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)

    fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(fmt, datefmt=datefmt)
    # Single StreamHandler pointed at the (already-teed) sys.stderr —
    # logger lines reach the file via the tee, no separate FileHandler so
    # nothing double-writes.
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Quiet noisy libs unless DEBUG
    if root.level > logging.DEBUG:
        for noisy in ("httpx", "httpcore", "openai", "asyncio", "LiteLLM"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    return path


# ── Tree helpers ──────────────────────────────────────────────────────────────

def _iter_nodes(structure):
    for node in structure or []:
        yield node
        if node.get('nodes'):
            yield from _iter_nodes(node['nodes'])


def _detect_type(structure) -> str:
    for node in _iter_nodes(structure):
        if 'line_num' in node:
            return 'md'
        if 'start_index' in node or 'end_index' in node:
            return 'pdf'
    return 'pdf'


def _max_end_index(structure) -> int:
    return max((n.get('end_index') or 0 for n in _iter_nodes(structure)), default=0)


def _max_line_num(structure) -> int:
    return max((n.get('line_num') or 0 for n in _iter_nodes(structure)), default=0)


def _find_node_by_id(structure, node_id):
    """Look up a node by id, tolerating common LLM mistakes.

    Accepts:
      - the canonical zero-padded string ('0007', '1119')
      - bare ints (7, 1119) — local Qwen3-class models routinely drop the
        quotes when serializing tool-call JSON, so the schema is permissive
        and resolution happens here.
      - unpadded strings ('7') — same root cause as above; we pad to the
        widths actually seen in the tree.
    Returns None on miss; callers should hand back a suggestion list."""
    if node_id is None or node_id == "":
        return None
    target = str(node_id).strip()
    # Strip surrounding quotes weak models occasionally leave in ('"0007"').
    if len(target) >= 2 and target[0] == target[-1] and target[0] in ("'", '"'):
        target = target[1:-1]
    candidates = {target}
    widths = {len(n.get('node_id')) for n in _iter_nodes(structure)
              if isinstance(n.get('node_id'), str)}
    for w in widths:
        if len(target) < w and target.isdigit():
            candidates.add(target.zfill(w))
    for node in _iter_nodes(structure):
        if node.get('node_id') in candidates:
            return node
    return None


def _suggest_node_ids(structure, node_id, limit: int = 5) -> list[str]:
    """Closest existing node_ids by numeric distance, for not-found hints."""
    target = str(node_id).strip().lstrip("\"'").rstrip("\"'")
    if not target.isdigit():
        return [n.get('node_id') for n in _iter_nodes(structure)
                if isinstance(n.get('node_id'), str)][:limit]
    target_int = int(target)
    pairs: list[tuple[int, str]] = []
    for n in _iter_nodes(structure):
        nid = n.get('node_id')
        if isinstance(nid, str) and nid.isdigit():
            pairs.append((abs(int(nid) - target_int), nid))
    pairs.sort()
    return [nid for _, nid in pairs[:limit]]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _truncate_text(value: str | None, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return value
    if max_chars <= 0:
        return None
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def _compact_structure_for_tool(structure, summary_max_chars: int, max_depth: int | None = None, _depth: int = 0):
    """Return a navigation-sized tree. When ``max_depth`` is set, recursion
    stops at that depth and each pruned node carries ``has_children``/
    ``num_descendants`` so the agent knows where to drill via from_node_id."""
    compact = []
    for node in structure or []:
        item = {
            'title': node.get('title', ''),
            'node_id': node.get('node_id'),
        }
        for key in ('start_index', 'end_index', 'line_num'):
            if node.get(key) is not None:
                item[key] = node.get(key)
        summary = _truncate_text(node.get('summary'), summary_max_chars)
        if summary:
            item['summary'] = summary
        children = node.get('nodes') or []
        if children:
            if max_depth is not None and _depth + 1 >= max_depth:
                item['has_children'] = True
                item['num_descendants'] = _count_descendants(children)
            else:
                item['nodes'] = _compact_structure_for_tool(
                    children, summary_max_chars, max_depth, _depth + 1
                )
        compact.append(item)
    return compact


def _count_descendants(structure) -> int:
    total = 0
    for n in structure or []:
        total += 1 + _count_descendants(n.get('nodes') or [])
    return total


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate. We're budgeting, not enforcing a hard limit, so a
    chars/4 heuristic is sufficient and avoids importing a tokenizer."""
    return (len(text) + 3) // 4


def _build_structure_payload(structure, summary_cap: int, max_depth: int | None, top_n: int | None = None):
    nodes = structure[:top_n] if top_n is not None else structure
    return _compact_structure_for_tool(nodes, summary_cap, max_depth)


def _fit_structure_to_budget(structure, summary_cap_env: int, requested_depth: int | None, char_budget: int):
    """Auto-clip depth (and finally summaries / top-level count) until the
    serialized payload fits ``char_budget``. Returns (compact, depth_used,
    summary_cap_used, top_n_used)."""
    if char_budget <= 0:
        return _build_structure_payload(structure, summary_cap_env, requested_depth), requested_depth, summary_cap_env, None

    # Depth ladder: requested → 3 → 2 → 1. Dedup while preserving order.
    ladder: list[int | None] = []
    seen: set[int | None] = set()
    for d in (requested_depth, 3, 2, 1):
        if d not in seen:
            ladder.append(d)
            seen.add(d)

    last_compact = None
    for depth in ladder:
        compact = _build_structure_payload(structure, summary_cap_env, depth)
        last_compact = compact
        if len(json.dumps(compact, ensure_ascii=False)) <= char_budget:
            return compact, depth, summary_cap_env, None

    # Still too big at depth=1: drop summaries.
    if summary_cap_env > 0:
        compact = _build_structure_payload(structure, 0, 1)
        last_compact = compact
        if len(json.dumps(compact, ensure_ascii=False)) <= char_budget:
            return compact, 1, 0, None

    # Last resort: slice top-level nodes.
    top_n = max(1, len(structure))
    while top_n > 1:
        top_n = max(1, top_n // 2)
        compact = _build_structure_payload(structure, 0, 1, top_n=top_n)
        last_compact = compact
        if len(json.dumps(compact, ensure_ascii=False)) <= char_budget:
            return compact, 1, 0, top_n

    return last_compact, 1, 0, 1


# ── Document loading ──────────────────────────────────────────────────────────

def load_documents(folder: Path) -> dict:
    documents = {}
    logger.info("Loading documents from %s", folder)
    for path in sorted(folder.glob("*.json")):
        if path.name == "_meta.json":
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("skipping %s: %s", path.name, e)
            continue

        if isinstance(data, list):
            structure = data
            doc_name = path.stem
            doc_description = ""
        elif isinstance(data, dict):
            structure = data.get('structure') or []
            doc_name = data.get('doc_name') or path.stem
            doc_description = data.get('doc_description') or ""
        else:
            logger.warning("skipping %s: unrecognized JSON shape", path.name)
            continue

        if not structure:
            logger.warning("skipping %s: no structure found", path.name)
            continue

        doc_type = _detect_type(structure)
        doc_id = path.stem
        doc = {
            'id': doc_id,
            'type': doc_type,
            'doc_name': doc_name,
            'doc_description': doc_description,
            'structure': structure,
        }
        if doc_type == 'pdf':
            doc['page_count'] = _max_end_index(structure)
        else:
            doc['line_count'] = _max_line_num(structure)
        documents[doc_id] = doc
        size = doc.get('page_count') if doc_type == 'pdf' else doc.get('line_count')
        unit = "pages" if doc_type == 'pdf' else "lines"
        logger.info("  loaded %s [%s, %s %s] %s", doc_id, doc_type, size, unit, doc_name)

    logger.info("Loaded %d document(s)", len(documents))
    return documents


# ── Local tool implementations ────────────────────────────────────────────────

def tool_list_documents(documents: dict, mode: str = "brief") -> str:
    """Return all indexed docs. mode='brief' (default) omits doc_description so
    the listing stays tiny on multi-doc corpora; the rolling agent history
    re-sends this every turn. mode='full' includes descriptions — use only when
    the agent needs to triage many docs in one shot."""
    logger.info("tool: list_documents(mode=%s)", mode)
    out = []
    include_description = mode == "full"
    for doc_id, doc in documents.items():
        entry = {
            'doc_id': doc_id,
            'doc_name': doc.get('doc_name', ''),
            'type': doc.get('type', ''),
        }
        if include_description:
            entry['doc_description'] = doc.get('doc_description', '')
        if doc.get('type') == 'pdf':
            entry['page_count'] = doc.get('page_count', 0)
        else:
            entry['line_count'] = doc.get('line_count', 0)
        out.append(entry)
    return json.dumps(out, ensure_ascii=False)


def tool_get_document(documents: dict, doc_id: str) -> str:
    logger.info("tool: get_document(doc_id=%s)", doc_id)
    doc = documents.get(doc_id)
    if not doc:
        logger.warning("  doc_id %s not found", doc_id)
        return json.dumps({'error': f'Document {doc_id} not found'})
    result = {
        'doc_id': doc_id,
        'doc_name': doc.get('doc_name', ''),
        'doc_description': doc.get('doc_description', ''),
        'type': doc.get('type', ''),
        'status': 'completed',
    }
    if doc.get('type') == 'pdf':
        result['page_count'] = doc.get('page_count', 0)
    else:
        result['line_count'] = doc.get('line_count', 0)
    return json.dumps(result, ensure_ascii=False)


def _auto_max_depth_for(total_nodes: int) -> int | None:
    """Default depth when the caller didn't specify one. Stays unlimited for
    small docs (where the full tree fits easily) and tightens as size grows."""
    if total_nodes > 500:
        return 1
    if total_nodes > 200:
        return 2
    if total_nodes > 80:
        return 3
    return None


def tool_get_document_structure(
    documents: dict,
    doc_id: str,
    max_depth: int | None = None,
    from_node_id: str | int | None = None,
) -> str:
    """Return a navigation-sized compact tree.

    Auto-clips depth (and finally summaries / top-level count) so the payload
    always fits ``PAGEINDEX_STRUCTURE_MAX_CHARS`` (default 120 000 chars,
    ~30 K tokens). Pass ``from_node_id`` to view just one branch; pass
    ``max_depth`` to override the auto default. Original tree is never lost —
    re-call with deeper ``max_depth`` or with ``from_node_id=<pruned-id>`` to
    keep drilling."""
    logger.info(
        "tool: get_document_structure(doc_id=%s, max_depth=%s, from_node_id=%s)",
        doc_id, max_depth, from_node_id,
    )
    doc = documents.get(doc_id)
    if not doc:
        logger.warning("  doc_id %s not found", doc_id)
        return json.dumps({'error': f'Document {doc_id} not found'})

    full_structure = doc.get('structure', []) or []
    # Treat empty/zero ints as "no scope" so from_node_id=0 means "root tree"
    # rather than "find node_id 0" (which surprises the agent because root is
    # node_id "0000", not "0").
    has_scope = from_node_id not in (None, "", 0, "0")
    if has_scope:
        node = _find_node_by_id(full_structure, from_node_id)
        if not node:
            suggestions = _suggest_node_ids(full_structure, from_node_id)
            return json.dumps({
                'error': f'Node {from_node_id!r} not found in {doc_id}',
                'hint': (
                    "node_id must be the zero-padded string used in the tree "
                    "(e.g. '0007', '1119'). Try one of the nearest existing ids, "
                    "or omit from_node_id to view the full tree."
                ),
                'suggested_node_ids': suggestions,
            })
        view = [node]
        scope = f'subtree(from_node_id={node.get("node_id")!r})'
    else:
        view = full_structure
        scope = 'full'

    total_nodes = _count_descendants(view)
    summary_cap = _env_int("PAGEINDEX_STRUCTURE_SUMMARY_MAX_CHARS", 160)
    char_budget = _env_int("PAGEINDEX_STRUCTURE_MAX_CHARS", 120000)

    # Clamp pathological max_depth values. max_depth=0 would prune everything
    # at the root and return an essentially empty payload — agents occasionally
    # ask for it thinking it means "minimum"; treat as 1 (top-level only).
    if max_depth is not None:
        try:
            max_depth = int(max_depth)
        except (TypeError, ValueError):
            max_depth = None
        if max_depth is not None and max_depth < 1:
            max_depth = 1

    requested_depth = max_depth if max_depth is not None else _auto_max_depth_for(total_nodes)
    compact, depth_used, summary_cap_used, top_n_used = _fit_structure_to_budget(
        view, summary_cap, requested_depth, char_budget,
    )

    hint_parts = []
    if depth_used is not None:
        hint_parts.append(
            f"Tree shown to depth {depth_used}. Nodes with has_children=true are pruned; "
            f"call get_document_structure(doc_id, from_node_id=<id>) or pass a larger max_depth to expand."
        )
    else:
        hint_parts.append("Full depth shown (small doc).")
    if requested_depth is not None and depth_used != requested_depth:
        hint_parts.append(
            f"depth auto-clipped from requested {requested_depth} to {depth_used} to fit char budget."
        )
    if summary_cap_used == 0 and summary_cap > 0:
        hint_parts.append("summaries dropped to fit budget.")
    if top_n_used is not None:
        hint_parts.append(
            f"top-level nodes truncated to first {top_n_used}; use from_node_id to expand the rest."
        )
    hint_parts.append(
        "Use get_node_content(doc_id, node_id) for text. Avoid re-fetching the full structure once you have node_ids."
    )

    payload = {
        'doc_id': doc_id,
        'scope': scope,
        'total_nodes_in_scope': total_nodes,
        'depth_used': depth_used,
        'summary_max_chars': summary_cap_used,
        'char_budget': char_budget,
        'hint': ' '.join(hint_parts),
        'structure': compact,
    }
    return json.dumps(payload, ensure_ascii=False)


def tool_get_node_content(
    documents: dict,
    doc_id: str,
    node_id: str | int,
    offset: int = 0,
    length: int | None = None,
) -> str:
    """Return a node's payload, windowed by (offset, length).

    Default behavior matches the legacy prefix-cut: when offset=0 and length is
    None, returns the first ``PAGEINDEX_NODE_TEXT_MAX_CHARS`` chars (default
    8000). With explicit offset / length the agent can scroll through long
    nodes (book chapters can be 200 K+ chars). Always returns
    ``total_chars``, ``offset``, ``returned_chars`` and ``has_more`` so the
    agent can plan follow-up windows."""
    logger.info(
        "tool: get_node_content(doc_id=%s, node_id=%s, offset=%s, length=%s)",
        doc_id, node_id, offset, length,
    )
    doc = documents.get(doc_id)
    if not doc:
        logger.warning("  doc_id %s not found", doc_id)
        return json.dumps({'error': f'Document {doc_id} not found'})
    node = _find_node_by_id(doc.get('structure', []), node_id)
    if not node:
        logger.warning("  node_id %s not found in %s", node_id, doc_id)
        return json.dumps({
            'error': f'Node {node_id!r} not found in {doc_id}',
            'hint': (
                "node_id must be the zero-padded string used in the tree "
                "(e.g. '0007', '1119')."
            ),
            'suggested_node_ids': _suggest_node_ids(doc.get('structure', []), node_id),
        })

    resolved_node_id = node.get('node_id', str(node_id))
    payload = {
        'doc_id': doc_id,
        'node_id': resolved_node_id,
        'title': node.get('title', ''),
        'start_index': node.get('start_index'),
        'end_index': node.get('end_index'),
        'line_num': node.get('line_num'),
    }

    text = node.get('text')
    summary = node.get('summary')
    if text:
        full = text
        payload['source'] = 'text'
    elif summary:
        full = summary
        payload['source'] = 'summary'
        payload['note'] = 'Node text was not generated; returning summary instead.'
    else:
        full = ''
        payload['source'] = 'none'
        payload['note'] = 'Neither text nor summary available for this node.'

    total = len(full)
    default_cap = _env_int("PAGEINDEX_NODE_TEXT_MAX_CHARS", 8000)
    max_window = _env_int("PAGEINDEX_NODE_TEXT_WINDOW_MAX_CHARS", 32000)

    # Normalize offset / length. offset clamps to [0, total]; length defaults
    # to the legacy cap when unspecified and is capped at max_window so a
    # weak model can't ask for 200 K at once.
    try:
        offset_int = max(0, int(offset))
    except (TypeError, ValueError):
        offset_int = 0
    offset_int = min(offset_int, total)

    if length is None:
        window = default_cap if default_cap > 0 else total
    else:
        try:
            window = int(length)
        except (TypeError, ValueError):
            window = default_cap
        if window <= 0:
            window = total
    if max_window > 0:
        window = min(window, max_window)

    end = min(total, offset_int + window) if window > 0 else total
    sliced = full[offset_int:end] if total else ''

    payload['text'] = sliced
    payload['total_chars'] = total
    payload['offset'] = offset_int
    payload['returned_chars'] = len(sliced)
    payload['has_more'] = end < total
    if payload['has_more']:
        payload.setdefault(
            'note',
            f"Windowed view: chars [{offset_int}, {end}) of {total}. "
            f"Call again with offset={end} to continue."
        )
    return json.dumps(payload, ensure_ascii=False)


# ── History compaction (F8) ───────────────────────────────────────────────────

def _make_structure_history_compactor(target_tool: str = "get_document_structure"):
    """Build a call_model_input_filter that elides stale tool outputs from the
    rolling input items. Only the *most recent* ``target_tool`` output is
    retained verbatim; older ones are replaced with a tiny placeholder so the
    tree dump doesn't keep occupying tens of thousands of tokens turn after
    turn. The tool is idempotent, so the agent can always re-call to refetch.
    """
    from agents.run_config import ModelInputData

    def _filter(data):
        items = data.model_data.input
        if not items:
            return data.model_data

        # Build call_id → tool name from function_call items.
        call_id_to_name: dict[str, str] = {}
        for it in items:
            if isinstance(it, dict) and it.get("type") == "function_call":
                cid = it.get("call_id")
                nm = it.get("name")
                if cid and nm:
                    call_id_to_name[cid] = nm

        # Find indices of all function_call_output items belonging to the
        # target tool. Keep the last; elide the rest.
        target_indices: list[int] = []
        for idx, it in enumerate(items):
            if not (isinstance(it, dict) and it.get("type") == "function_call_output"):
                continue
            if call_id_to_name.get(str(it.get("call_id", ""))) == target_tool:
                target_indices.append(idx)

        if len(target_indices) <= 1:
            return data.model_data

        to_elide = set(target_indices[:-1])
        new_items: list = []
        elided_chars = 0
        for idx, it in enumerate(items):
            if idx in to_elide and isinstance(it, dict):
                original = it.get("output", "")
                original_str = original if isinstance(original, str) else str(original)
                placeholder = (
                    f"[elided earlier {target_tool} output ({len(original_str)} chars); "
                    f"re-call the tool if you need the tree again]"
                )
                if len(placeholder) < len(original_str):
                    elided_chars += len(original_str) - len(placeholder)
                    trimmed = dict(it)
                    trimmed["output"] = placeholder
                    new_items.append(trimmed)
                    continue
            new_items.append(it)

        if elided_chars > 0:
            logger.info(
                "history-compactor: elided %d %s output(s), saved ~%d chars",
                len(to_elide), target_tool, elided_chars,
            )
        return ModelInputData(input=new_items, instructions=data.model_data.instructions)

    return _filter


# ── Agent ─────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """
You are PageIndex, a local document QA assistant.

ID FORMAT (important — weak models get this wrong):
- doc_id is the JSON filename stem, e.g. "book1_structure".
- node_id is a zero-padded quoted string, e.g. "0000", "0441", "1119". Always pass it
  as a JSON string with quotes, NOT as a bare integer.

TOOL USE:
- Call list_documents() first (default mode='brief'). Switch to mode='full' only if
  doc names alone aren't enough to disambiguate.
- Call get_document(doc_id) when you need one doc's full description.
- Call get_document_structure(doc_id) to inspect the compact tree. For large books
  the call auto-clips depth; pruned nodes carry has_children=true and num_descendants.
  Drill into a chapter with get_document_structure(doc_id="book1_structure",
  from_node_id="0441"). Use max_depth=2 on small docs only. Do NOT re-call for the
  full tree once you already have candidate node_ids — older structure outputs are
  auto-elided from the rolling history to save context.
- Call get_node_content(doc_id="book1_structure", node_id="0441") for text. The
  response carries total_chars and has_more. If has_more=true and you need the rest
  of a long node, call again with offset=<previous offset + returned_chars>. Prefer
  leaf nodes; expand to parents only if needed.
- Before each tool call, output one short sentence explaining the reason.
- If a tool returns an error with suggested_node_ids, switch to one of those — do
  NOT retry the same argument value.
Answer based only on tool output. Give a thorough, well-structured answer that explains the key evidence, important caveats, and how the cited evidence supports the conclusion. Cite the doc_id and node_id for each major claim.
When ready to answer, either write normal assistant text or call answer(answer=...).
Do not call answer before using the document tools needed for evidence.
"""


DEFAULT_BASE_URLS = {
    "vllm": "http://localhost:8000/v1",
    "ollama": "http://localhost:11434/v1",
}

PROVIDER_PREFIXES = ("ollama/", "vllm/", "openai/", "litellm/")


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def resolve_provider(model: str, provider: str) -> tuple[str, str]:
    """Return (provider, bare_model_name). Strip recognized provider prefixes from model."""
    if provider == "auto":
        if not model:
            provider = "openai"
        elif model.startswith("ollama/"):
            provider, model = "ollama", model[len("ollama/"):]
        elif model.startswith("vllm/"):
            provider, model = "vllm", model[len("vllm/"):]
        elif model.startswith("openai/"):
            provider, model = "openai", model[len("openai/"):]
        elif model.startswith("litellm/"):
            provider, model = "litellm", model[len("litellm/"):]
        elif "/" in model:
            provider = "litellm"
        else:
            provider = "openai"
    else:
        for p in PROVIDER_PREFIXES:
            if model and model.startswith(p):
                model = model[len(p):]
                break
    return provider, model


def _build_agent_model(provider: str, model: str, base_url: str | None, api_key: str | None):
    """Construct the model object passed to Agent(model=...).

    For openai-compatible local servers (vllm/ollama) we build an AsyncOpenAI client
    pointed at the local base_url and wrap it with OpenAIChatCompletionsModel so the
    Agents SDK uses Chat Completions (which both vLLM and Ollama implement) rather
    than the Responses API (OpenAI-hosted only).
    """
    if provider == "litellm":
        # Agents SDK auto-loads litellm extension when model string starts with "litellm/".
        try:
            import agents.extensions.models.litellm_model  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "Missing dependency for litellm provider. Install with: "
                "pip install 'openai-agents[litellm]'"
            ) from e
        return f"litellm/{model}"

    if provider == "openai":
        # Hosted OpenAI or a custom OPENAI_BASE_URL — let Agents SDK use its default
        # OpenAIResponsesModel via the model name string.
        if base_url:
            from openai import AsyncOpenAI
            from agents import OpenAIChatCompletionsModel
            client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
            )
            return OpenAIChatCompletionsModel(model=model, openai_client=client)
        return model

    # vllm / ollama: OpenAI-compatible Chat Completions
    from openai import AsyncOpenAI
    from agents import OpenAIChatCompletionsModel
    resolved_base = base_url or DEFAULT_BASE_URLS[provider]
    resolved_key = api_key or os.getenv(f"{provider.upper()}_API_KEY") or "EMPTY"
    client = AsyncOpenAI(base_url=resolved_base, api_key=resolved_key)
    return OpenAIChatCompletionsModel(model=model, openai_client=client)


async def _refine_vllm_answer_with_thinking(
    model: str,
    base_url: str,
    api_key: str | None,
    question: str,
    draft: str,
    tool_outputs: list[str],
    max_tokens: int,
) -> str:
    """Run a final non-tool vLLM call with Qwen thinking enabled.

    Tool turns stay non-thinking because qwen3 reasoning-parser can put tool
    intent in `reasoning` without emitting valid OpenAI tool calls. This pass
    gives users thinking during final answer synthesis without breaking tools.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.warning("openai package unavailable; skipping thinking refinement")
        return draft

    evidence_cap = int(os.getenv("PAGEINDEX_RETRIEVE_THINKING_EVIDENCE_MAX_CHARS", "12000"))
    evidence = "\n\n".join(tool_outputs)
    if evidence_cap > 0 and len(evidence) > evidence_cap:
        evidence = evidence[:evidence_cap]

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key or os.getenv("VLLM_API_KEY") or "EMPTY",
    )
    logger.info(
        "running vLLM thinking refinement max_tokens=%s evidence_chars=%d",
        max_tokens,
        len(evidence),
    )
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the retrieval evidence and draft answer. "
                        "Aim for a comprehensive yet accessible answer: preserve the depth, "
                        "structure, specific figures, and supporting points already present in "
                        "the draft, but reorganize for clarity — short paragraphs, headings or "
                        "bullets where they aid scanability, and plain language alongside any "
                        "technical terms. Correct factual errors against the evidence and keep "
                        "all document/node citations. Do not shorten by dropping content; only "
                        "trim genuine redundancy."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Draft answer:\n{draft}\n\n"
                        f"Retrieval evidence:\n{evidence}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
    except Exception:
        logger.exception("vLLM thinking refinement failed; returning draft answer")
        return draft

    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    reasoning = getattr(choice.message, "reasoning", None) or getattr(choice.message, "reasoning_content", None) or ""
    logger.info(
        "vLLM thinking refinement finish_reason=%s content_chars=%d reasoning_chars=%d",
        choice.finish_reason,
        len(content),
        len(reasoning),
    )
    if not content:
        logger.warning("vLLM thinking refinement returned empty content; returning draft answer")
        return draft
    return content


def query_agent(
    documents: dict,
    model: str,
    question: str,
    verbose: bool = False,
    provider: str = "auto",
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    try:
        from agents import Agent, Runner, RunConfig, function_tool, set_tracing_disabled
        from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
        from openai.types.responses import (
            ResponseTextDeltaEvent,
            ResponseReasoningSummaryTextDeltaEvent,
        )
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: openai-agents. Install with: pip install openai-agents"
        ) from e

    set_tracing_disabled(True)

    provider, bare_model = resolve_provider(model, provider)
    agent_model = _build_agent_model(provider, bare_model, base_url, api_key)
    resolved_base = base_url or DEFAULT_BASE_URLS.get(provider, "default")
    logger.info("provider=%s model=%s base_url=%s", provider, bare_model, resolved_base)
    logger.info("question: %s", question)

    @function_tool
    def list_documents(mode: str = "brief") -> str:
        """List indexed documents. mode='brief' (default) returns id/name/type/size;
        mode='full' also returns doc_description. Prefer 'brief' when only navigating —
        fetch a single description via get_document(doc_id) when needed."""
        return tool_list_documents(documents, mode=mode)

    @function_tool
    def get_document(doc_id: str) -> str:
        """Get document metadata: status, page/line count, name, description."""
        return tool_get_document(documents, doc_id)

    @function_tool
    def get_document_structure(
        doc_id: str,
        max_depth: int | None = None,
        from_node_id: str | int | None = None,
    ) -> str:
        """Get a navigation-sized compact tree.

        node_id values in this tree are zero-padded strings like "0007" or
        "1119". Pass from_node_id as a quoted string; bare ints (1049) are also
        accepted and will be padded to match.

        Optional args:
        - max_depth: maximum depth to expand (1 = top-level only, 2 = chapters
          plus sections, etc.). Auto-tuned by node count when None; auto-clipped
          further to fit a char budget. Pruned nodes carry has_children=true so
          you can drill via from_node_id.
        - from_node_id: return only the subtree rooted at this node. Omit to
          view the full tree."""
        return tool_get_document_structure(
            documents, doc_id, max_depth=max_depth, from_node_id=from_node_id,
        )

    @function_tool
    def get_node_content(
        doc_id: str,
        node_id: str | int,
        offset: int = 0,
        length: int | None = None,
    ) -> str:
        """Get the text of a node, windowed by (offset, length).

        node_id is the zero-padded string seen in the tree ("0007", "1119").
        Bare ints (7, 1119) are also accepted and padded to match.

        Default (offset=0, length=None) returns the first ~8000 chars. For long
        nodes the response includes total_chars and has_more — pass
        offset=<previous offset + returned_chars> to continue. Pass an explicit
        length to request a smaller window."""
        return tool_get_node_content(
            documents, doc_id, node_id, offset=offset, length=length,
        )

    @function_tool
    def answer(answer: str) -> str:
        """Return a thorough final answer to the user after document evidence has been gathered."""
        return answer

    answer_thinking = _env_truthy("PAGEINDEX_RETRIEVE_ENABLE_THINKING", False)
    if provider == "vllm" and "PAGEINDEX_AGENT_MAX_TOKENS" not in os.environ:
        default_max_tokens = 8192 if answer_thinking else 2048
    else:
        default_max_tokens = 2048

    # Bound output tokens per turn so vLLM/Ollama don't allocate worst-case
    # KV slots for runaway generations. Thinking retrieval needs more budget
    # because qwen3 reasoning tokens precede final content.
    _max_out = int(os.getenv("PAGEINDEX_AGENT_MAX_TOKENS", str(default_max_tokens)))
    extra_body = None
    if provider == "vllm":
        # Keep the tool loop non-thinking. With vLLM --reasoning-parser qwen3,
        # thinking tool turns can route tool intent into `reasoning` only and
        # leave `content`/`tool_calls` empty. Optional thinking is applied later
        # in a final answer refinement pass.
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        logger.info(
            "vllm tool_loop_thinking=False answer_thinking=%s max_tokens=%s",
            answer_thinking,
            _max_out if _max_out > 0 else "server-default",
        )

    agent_kwargs = dict(
        name="PageIndexLocal",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[list_documents, get_document, get_document_structure, get_node_content, answer],
        tool_use_behavior={"stop_at_tool_names": ["answer"]},
        model=agent_model,
    )
    if _max_out > 0 or extra_body:
        try:
            from agents import ModelSettings
            settings = {}
            if _max_out > 0:
                settings["max_tokens"] = _max_out
            settings["parallel_tool_calls"] = False
            if extra_body:
                settings["extra_body"] = extra_body
            agent_kwargs["model_settings"] = ModelSettings(**settings)
        except ImportError:
            logger.warning("agents.ModelSettings not available; model settings skipped")
    agent = Agent(**agent_kwargs)

    async def _run():
        # Cap agent loop length. Long loops blow up the rolling message
        # history, which is what eventually saturates vLLM's KV cache on
        # local hosts.
        _max_turns = int(os.getenv("PAGEINDEX_AGENT_MAX_TURNS", "30"))
        # F8: elide stale get_document_structure outputs from the rolling
        # history so a 30 K-token tree dump doesn't keep occupying ctx after
        # the agent has already extracted its node_ids.
        run_config = RunConfig(
            call_model_input_filter=_make_structure_history_compactor(),
            tracing_disabled=True,
        )
        streamed_run = Runner.run_streamed(
            agent, question, max_turns=_max_turns, run_config=run_config,
        )
        current_kind = None
        tool_outputs_for_refine = []
        async for event in streamed_run.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                if isinstance(event.data, ResponseReasoningSummaryTextDeltaEvent):
                    if current_kind != "reasoning":
                        if current_kind is not None:
                            print()
                        print("\n[reasoning]: ", end="", flush=True)
                    print(event.data.delta, end="", flush=True)
                    current_kind = "reasoning"
                elif isinstance(event.data, ResponseTextDeltaEvent):
                    if current_kind != "text":
                        if current_kind is not None:
                            print()
                        print("\n[text]: ", end="", flush=True)
                    print(event.data.delta, end="", flush=True)
                    current_kind = "text"
            elif isinstance(event, RunItemStreamEvent):
                item = event.item
                if item.type == "tool_call_item":
                    if current_kind is not None:
                        print()
                    raw = item.raw_item
                    args = getattr(raw, "arguments", "{}")
                    args_str = f"({args})" if verbose else ""
                    print(f"\n[tool call]: {raw.name}{args_str}", flush=True)
                    logger.info("agent tool_call: %s args=%s", raw.name, args)
                    current_kind = None
                elif item.type == "tool_call_output_item":
                    output = str(item.output)
                    tool_outputs_for_refine.append(output)
                    preview = output[:200] + "..." if len(output) > 200 else output
                    # Log full output to file; keep stdout preview short.
                    log_cap = _env_int("PAGEINDEX_TOOL_LOG_MAX_CHARS", 4000)
                    logged_output = output if log_cap <= 0 else output[:log_cap]
                    if log_cap > 0 and len(output) > log_cap:
                        logged_output += f"... [truncated from {len(output)} chars]"
                    logger.info("agent tool_output (%d chars): %s", len(output), logged_output)
                    if verbose:
                        if current_kind is not None:
                            print()
                        print(f"\n[tool call output]: {preview}", flush=True)
                        current_kind = None
        if current_kind is not None:
            print()
        final = "" if not streamed_run.final_output else str(streamed_run.final_output)
        if final and provider == "vllm" and answer_thinking:
            final = await _refine_vllm_answer_with_thinking(
                model=bare_model,
                base_url=resolved_base,
                api_key=api_key,
                question=question,
                draft=final,
                tool_outputs=tool_outputs_for_refine,
                max_tokens=_max_out if _max_out > 0 else 8192,
            )
        if not final and provider == "vllm":
            logger.warning(
                "empty final answer from vLLM; if thinking is enabled, raise "
                "PAGEINDEX_AGENT_MAX_TOKENS or disable thinking with "
                "PAGEINDEX_RETRIEVE_ENABLE_THINKING=false"
            )
        logger.info("agent final answer (%d chars): %s", len(final), final)
        return final

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run()).result()
    except RuntimeError:
        return asyncio.run(_run())


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Local retrieval over a folder of PageIndex tree JSON files."
    )
    parser.add_argument("--folder", required=True,
                        help="Folder containing PageIndex tree JSON files.")
    parser.add_argument("--question", required=True,
                        help="Question to answer.")
    parser.add_argument("--model", default=None,
                        help="LLM model (overrides config.yaml retrieve_model/model). "
                             "Prefix with ollama/, vllm/, openai/, or litellm/ for auto-detect.")
    parser.add_argument("--provider", default="auto",
                        choices=["auto", "openai", "vllm", "ollama", "litellm"],
                        help="LLM provider. 'auto' detects from model prefix.")
    parser.add_argument("--base-url", default=None,
                        help="Override base URL for OpenAI-compatible server "
                             "(vllm: http://localhost:8000/v1, ollama: http://localhost:11434/v1).")
    parser.add_argument("--api-key", default=None,
                        help="API key for chosen provider (env fallbacks: OPENAI_API_KEY, "
                             "VLLM_API_KEY, OLLAMA_API_KEY).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print tool call arguments and output previews.")
    parser.add_argument("--log-file", default=None,
                        help="Path to log file (default: ./logs/retrieve_<timestamp>.log).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level (default: INFO). DEBUG also enables httpx/openai chatter.")
    args = parser.parse_args()

    log_path = setup_logging(level=args.log_level, log_file=args.log_file)
    logger.info("=== retrieve_pageindex run start ===")
    logger.info("log file: %s", log_path)

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    documents = load_documents(folder)
    if not documents:
        raise SystemExit(f"No usable JSON tree files found in {folder}")

    print(f"Loaded {len(documents)} document(s) from {folder}:")
    for doc_id, doc in documents.items():
        size = doc.get('page_count') if doc['type'] == 'pdf' else doc.get('line_count')
        unit = "pages" if doc['type'] == 'pdf' else "lines"
        print(f"  - {doc_id}  [{doc['type']}, {size} {unit}]  {doc.get('doc_name', '')}")
    print()

    # Resolve model: explicit --model wins, then config retrieve_model, then config model.
    if args.model:
        model = args.model
    else:
        opt = ConfigLoader().load(None)
        model = getattr(opt, 'retrieve_model', None) or opt.model

    # Provider-aware preflight: only require OpenAI key when targeting hosted OpenAI.
    provider, _ = resolve_provider(model, args.provider)
    base_url = args.base_url or os.getenv(f"{provider.upper()}_BASE_URL")
    if provider == "openai" and not base_url:
        base_url = os.getenv("OPENAI_BASE_URL")
    if provider == "openai" and not base_url:
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("CHATGPT_API_KEY") and not args.api_key:
            raise SystemExit(
                "Provider 'openai' requires OPENAI_API_KEY (or CHATGPT_API_KEY, or --api-key). "
                "For local models, use --provider vllm|ollama|litellm."
            )

    print(f"Question: {args.question}\n")

    try:
        answer = query_agent(
            documents,
            model,
            args.question,
            verbose=args.verbose,
            provider=args.provider,
            base_url=base_url,
            api_key=args.api_key,
        )
    except Exception:
        logger.exception("query_agent failed")
        raise
    print("\n" + "=" * 60)
    print("Final answer:")
    print("=" * 60)
    print(answer)
    logger.info("=== retrieve_pageindex run end ===")


if __name__ == "__main__":
    main()
