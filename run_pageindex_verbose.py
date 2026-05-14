"""
Verbose version of run_pageindex.py.

Same CLI as run_pageindex.py, plus:
  --verbose / --quiet     toggle live progress stream (default: verbose ON)
  --log-level             python logging level for litellm/asyncio chatter

Delegates the actual pipeline to the upstream `page_index_main` /
`md_to_tree` so behavior stays identical to `run_pageindex.py`. Verbose
overlay is purely observational:
  1. Wraps `pageindex.utils.JsonLogger` so every `.info` / `.error` call
     also emits a single-line summary to stderr (the original JSON file
     under ./logs/ is still written).
  2. Times each top-level phase: PDF/MD pipeline + save.
  3. Forwards the existing scattered `print(...)` statements unchanged
     (they already announce TOC detection, mode selection, accuracy,
     retries, etc.).
  4. Optionally tees the stderr stream to a log file under ./logs/.
"""
import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime

_TS_RE = re.compile(r"^\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")

# ── Force instruct (non-thinking) mode for tree-build ─────────────────────────
# Tree-build issues hundreds of small JSON-only LLM calls. A Qwen3 thinking
# trace per call multiplies wall-clock 2-4×. Even when the user enables
# `--reasoning-parser qwen3` + `VLLM_ENABLE_THINKING=true` server/global-side
# (typically for retrieve_pageindex.py), we hard-pin tree-build to
# instruct/non-thinking mode here BEFORE pageindex.utils reads the env in
# `_provider_kwargs`. Override has to happen pre-import so the default takes
# effect; we also re-assert after import for safety.
os.environ["VLLM_ENABLE_THINKING"] = "false"
os.environ["OLLAMA_THINK"] = "false"

# Wire stderr logging early so any module-level logging calls during import
# are captured.
_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format=_LOG_FORMAT)

# ── Verbose JsonLogger wrapper ────────────────────────────────────────────────

import pageindex.utils as _pi_utils  # noqa: E402

# Re-assert in case caller exported a different value before `python` ran;
# `_provider_kwargs` reads os.getenv at every call, so this sticks for the
# whole tree-build pipeline.
os.environ["VLLM_ENABLE_THINKING"] = "false"
os.environ["OLLAMA_THINK"] = "false"

_OriginalJsonLogger = _pi_utils.JsonLogger
_VERBOSE = True  # toggled by CLI

# ── Phase 0: preflight + checkpoint + heartbeat state ─────────────────────────
# Wrapper-only observability/safety. None of these mutate pipeline output.
_CHECKPOINT_PATH = None
_CHECKPOINT_LOCK = threading.Lock()
_CURRENT_PHASE = None
_CURRENT_PHASE_T0 = None
_LLM_LOCK = threading.Lock()
_LLM_DONE = 0
_LLM_INFLIGHT = 0
_LLM_LATENCY_SUM = 0.0
_LLM_LATENCY_N = 0
_HEARTBEAT_STOP = threading.Event()


def _short(value, limit=400):
    """Render any log payload as a single-line preview."""
    try:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, default=str)
        else:
            text = str(value)
    except Exception:
        text = repr(value)
    text = text.replace("\n", " ")
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class _Tee:
    """Mirror writes to a stream + log file. Prepends timestamps in the file
    copy for any line that doesn't already start with one. Terminal copy
    stays untouched."""
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


def _setup_log_file(path):
    """Tee BOTH stdout and stderr to log file. Reconfigure root logging so
    prior handlers inherit the new sys.stderr. Returns resolved Path."""
    from pathlib import Path as _Path
    p = _Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "a", encoding="utf-8", buffering=1)
    f.write(f"\n[{_ts()}] === log start: {p} ===\n")
    sys.stderr = _Tee(sys.stderr, f)
    sys.stdout = _Tee(sys.stdout, f)
    # Rewire any existing StreamHandler to new sys.stderr.
    for h in list(logging.getLogger().handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.stream = sys.stderr
    return p


class _VerboseJsonLogger(_OriginalJsonLogger):
    """JsonLogger that also streams every log line to stderr with timestamp."""

    def log(self, level, message, **kwargs):
        super().log(level, message, **kwargs)
        if not _VERBOSE:
            return
        try:
            preview = _short(message)
            print(f"[{_ts()}] [{level:5s}] {preview}", file=sys.stderr, flush=True)
        except Exception as e:  # never let logging break the pipeline
            print(f"[{_ts()}] [verbose-log-error] {e}", file=sys.stderr, flush=True)


_pi_utils.JsonLogger = _VerboseJsonLogger
# Re-export under the names the package members already imported.
# NOTE: `import pageindex.page_index as X` would bind X to the FUNCTION
# `page_index` (re-exported by pageindex/__init__.py via `from .page_index
# import *`), not the submodule — `import a.b as c` resolves via
# `getattr(a, 'b')`. Reach into sys.modules to get the actual submodule.
import importlib  # noqa: E402
importlib.import_module("pageindex.page_index")
importlib.import_module("pageindex.page_index_md")
_pi_pdf = sys.modules["pageindex.page_index"]
_pi_md = sys.modules["pageindex.page_index_md"]
_pi_pdf.JsonLogger = _VerboseJsonLogger
_pi_md.JsonLogger = _VerboseJsonLogger


# ── Phase timer helper ────────────────────────────────────────────────────────

@contextmanager
def _phase(name: str):
    global _CURRENT_PHASE, _CURRENT_PHASE_T0
    if _VERBOSE:
        print(f"\n[{_ts()}] === {name} ===", file=sys.stderr, flush=True)
    prev_phase, prev_t0 = _CURRENT_PHASE, _CURRENT_PHASE_T0
    _CURRENT_PHASE = name
    _CURRENT_PHASE_T0 = time.perf_counter()
    t0 = _CURRENT_PHASE_T0
    try:
        yield
    finally:
        if _VERBOSE:
            dt = time.perf_counter() - t0
            print(f"[{_ts()}] === {name} done in {dt:.2f}s ===", file=sys.stderr, flush=True)
        _CURRENT_PHASE, _CURRENT_PHASE_T0 = prev_phase, prev_t0


# ── Phase 0: preflight, checkpoint hooks, LLM counters, heartbeat ─────────────

def _preflight_vllm(model_str, base_url):
    """Probe vLLM /v1/models for the served model's max_model_len; auto-clamp
    the client-side runtime when the env/CLI value exceeds the server's.

    Why: book2.pdf was bitten by env=65536 but server=32768 — every prompt
    silently overflowed, LiteLLM retried 10× per call, then the pipeline
    crashed with `vLLM prompt budget exceeded: headroom=-87029`. A 10s
    HTTP probe at startup eliminates the entire failure class."""
    if not model_str:
        return
    normalized = model_str.removeprefix("litellm/")
    if not normalized.startswith(("vllm/", "hosted_vllm/")):
        return
    served_name = normalized.split("/", 1)[1]

    if not base_url:
        base_url = os.environ.get("VLLM_API_BASE") or os.environ.get("VLLM_BASE_URL")
    if not base_url:
        print(f"[{_ts()}] [preflight] no vLLM base URL; skipping",
              file=sys.stderr, flush=True)
        return
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    url = base + "/models"

    try:
        req = urllib.request.Request(url, headers={"Authorization": "Bearer EMPTY"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"[{_ts()}] [preflight] {url} unreachable ({e}); skipping",
              file=sys.stderr, flush=True)
        return

    models = data.get("data") or []
    match = next((m for m in models if m.get("id") == served_name), None)
    if match is None:
        ids = [m.get("id") for m in models]
        print(f"[{_ts()}] [preflight] '{served_name}' not served; available={ids}",
              file=sys.stderr, flush=True)
        return

    server_max = match.get("max_model_len")
    if not isinstance(server_max, int) or server_max <= 0:
        print(f"[{_ts()}] [preflight] server did not report max_model_len for {served_name}",
              file=sys.stderr, flush=True)
        return

    client_max = _pi_utils.get_llm_runtime_int(
        "vllm_max_model_len", ("VLLM_MAX_MODEL_LEN",), 16384,
    )
    if client_max > server_max:
        print(
            f"[{_ts()}] [preflight] client vllm_max_model_len={client_max} > "
            f"server={server_max} ({served_name}); auto-clamping",
            file=sys.stderr, flush=True,
        )
        _pi_utils.configure_llm_runtime(vllm_max_model_len=server_max)
        os.environ["VLLM_MAX_MODEL_LEN"] = str(server_max)
    else:
        print(
            f"[{_ts()}] [preflight] client={client_max} ≤ server={server_max} "
            f"({served_name}) — OK",
            file=sys.stderr, flush=True,
        )


def _checkpoint(structure, stage_name, extras=None):
    """Atomic JSON snapshot of the partial tree. No-op when path not set."""
    if not _CHECKPOINT_PATH:
        return
    try:
        payload = {"_partial": True, "_stage": stage_name, "_ts": _ts()}
        if isinstance(extras, dict):
            payload.update(extras)
        payload["structure"] = structure
        tmp = _CHECKPOINT_PATH + ".tmp"
        with _CHECKPOINT_LOCK:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp, _CHECKPOINT_PATH)
        print(f"[{_ts()}] [checkpoint] {stage_name} → {_CHECKPOINT_PATH}",
              file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[{_ts()}] [checkpoint-error] {stage_name}: {e}",
              file=sys.stderr, flush=True)


def _install_checkpoint_hooks():
    """Decorate each top-level pipeline stage so a partial tree lands on
    disk after every irreversible step. Originals are called first; only
    the side-effect of writing JSON is added — pipeline output unchanged."""
    # `pageindex.page_index` (attribute on the package) is the FUNCTION
    # exported via `from .page_index import *` in __init__.py — not the
    # submodule. Reach into sys.modules for the actual module object so
    # rebinds take effect.
    _pi = sys.modules["pageindex.page_index"]

    orig_tree_parser = _pi.tree_parser
    orig_write_id = _pi.write_node_id
    orig_add_text = _pi.add_node_text
    orig_gen_summaries = _pi.generate_summaries_for_structure
    orig_gen_desc = _pi.generate_doc_description

    async def _tree_parser_with_ckpt(*a, **kw):
        result = await orig_tree_parser(*a, **kw)
        _checkpoint(result, "tree_parser")
        return result

    def _write_id_with_ckpt(structure, *a, **kw):
        ret = orig_write_id(structure, *a, **kw)
        _checkpoint(structure, "write_node_id")
        return ret

    def _add_text_with_ckpt(structure, *a, **kw):
        ret = orig_add_text(structure, *a, **kw)
        _checkpoint(structure, "add_node_text")
        return ret

    async def _gen_summaries_with_ckpt(structure, *a, **kw):
        ret = await orig_gen_summaries(structure, *a, **kw)
        _checkpoint(structure, "generate_summaries_for_structure")
        return ret

    def _gen_desc_with_ckpt(clean_structure, *a, **kw):
        ret = orig_gen_desc(clean_structure, *a, **kw)
        _checkpoint(clean_structure, "generate_doc_description",
                    extras={"doc_description": ret})
        return ret

    _pi.tree_parser = _tree_parser_with_ckpt
    _pi.write_node_id = _write_id_with_ckpt
    _pi.add_node_text = _add_text_with_ckpt
    _pi.generate_summaries_for_structure = _gen_summaries_with_ckpt
    _pi.generate_doc_description = _gen_desc_with_ckpt


def _install_llm_counters():
    """Instrument llm_completion / llm_acompletion in every module that
    binds them, so the heartbeat can show rolling call counts + latency.
    Both page_index modules star-import from utils, so the symbol must be
    rebound in each (utils, page_index, page_index_md)."""
    # See _install_checkpoint_hooks for why sys.modules is required.
    _pi = sys.modules["pageindex.page_index"]
    _pi_md_mod = sys.modules["pageindex.page_index_md"]

    orig_sync = _pi_utils.llm_completion
    orig_async = _pi_utils.llm_acompletion

    def wrapped_sync(*a, **kw):
        global _LLM_DONE, _LLM_INFLIGHT, _LLM_LATENCY_SUM, _LLM_LATENCY_N
        with _LLM_LOCK:
            _LLM_INFLIGHT += 1
        t0 = time.perf_counter()
        try:
            return orig_sync(*a, **kw)
        finally:
            dt = time.perf_counter() - t0
            with _LLM_LOCK:
                _LLM_INFLIGHT -= 1
                _LLM_DONE += 1
                _LLM_LATENCY_SUM += dt
                _LLM_LATENCY_N += 1

    async def wrapped_async(*a, **kw):
        global _LLM_DONE, _LLM_INFLIGHT, _LLM_LATENCY_SUM, _LLM_LATENCY_N
        with _LLM_LOCK:
            _LLM_INFLIGHT += 1
        t0 = time.perf_counter()
        try:
            return await orig_async(*a, **kw)
        finally:
            dt = time.perf_counter() - t0
            with _LLM_LOCK:
                _LLM_INFLIGHT -= 1
                _LLM_DONE += 1
                _LLM_LATENCY_SUM += dt
                _LLM_LATENCY_N += 1

    for mod in (_pi_utils, _pi, _pi_md_mod):
        if getattr(mod, "llm_completion", None) is not None:
            mod.llm_completion = wrapped_sync
        if getattr(mod, "llm_acompletion", None) is not None:
            mod.llm_acompletion = wrapped_async


def _heartbeat_loop(interval):
    while not _HEARTBEAT_STOP.wait(interval):
        try:
            phase = _CURRENT_PHASE
            elapsed = (time.perf_counter() - _CURRENT_PHASE_T0) if _CURRENT_PHASE_T0 else 0.0
            with _LLM_LOCK:
                done = _LLM_DONE
                inflight = _LLM_INFLIGHT
                avg_ms = (_LLM_LATENCY_SUM / _LLM_LATENCY_N * 1000.0) if _LLM_LATENCY_N else 0.0
            print(
                f"[{_ts()}] [heartbeat] phase={phase or 'idle'} elapsed={elapsed:.1f}s "
                f"llm_done={done} llm_inflight={inflight} avg_ms={avg_ms:.0f}",
                file=sys.stderr, flush=True,
            )
        except Exception as e:
            print(f"[{_ts()}] [heartbeat-error] {e}", file=sys.stderr, flush=True)


def _start_heartbeat(interval=15.0):
    _HEARTBEAT_STOP.clear()
    t = threading.Thread(
        target=_heartbeat_loop, args=(interval,),
        daemon=True, name="pageindex-heartbeat",
    )
    t.start()
    return t


def _stop_heartbeat():
    _HEARTBEAT_STOP.set()


# ── Patched page_index_main / md_to_tree wrappers ─────────────────────────────

from pageindex import *  # noqa: E402,F401,F403
from pageindex.page_index import page_index_main as _page_index_main  # noqa: E402
from pageindex.page_index_md import md_to_tree  # noqa: E402
from pageindex.utils import ConfigLoader, configure_llm_runtime  # noqa: E402


# ── CLI (mirrors run_pageindex.py) ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Process PDF or Markdown document and generate structure (verbose).'
    )
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

    parser.add_argument('--model', type=str, default=None, help='Model to use (overrides config.yaml)')
    parser.add_argument('--base-url', type=str, default=None,
                        help='Base URL for hosted vLLM/Ollama-compatible servers')
    parser.add_argument('--vllm-max-model-len', type=int, default=None,
                        help='Server max model length for vLLM/hosted_vllm')
    parser.add_argument('--vllm-max-tokens', type=int, default=None,
                        help='Per-request output cap for vLLM/hosted_vllm; 0 leaves server default')
    parser.add_argument('--vllm-timeout', type=float, default=None,
                        help='Request timeout for vLLM/hosted_vllm')
    parser.add_argument('--vllm-ctx-margin', type=int, default=None,
                        help='Context safety margin for vLLM max_tokens clamping')
    parser.add_argument('--llm-concurrency', type=int, default=None,
                        help='Max in-flight async LLM calls')
    parser.add_argument('--group-max-tokens', type=int, default=None,
                        help='Max tokens per no-TOC page group')
    parser.add_argument('--group-prompt-overhead', type=int, default=None,
                        help='Prompt reserve used when sizing no-TOC page groups')
    parser.add_argument('--toc-chunk-max-tokens', type=int, default=None,
                        help='Max tokens per chunk for LLM TOC transformation fallback')

    parser.add_argument('--toc-check-pages', type=int, default=None,
                        help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--toc-verify-sample', type=int, default=None,
                        help='Number of TOC entries to verify; 0 checks all (PDF only)')
    parser.add_argument('--max-pages-per-node', type=int, default=None,
                        help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=None,
                        help='Maximum number of tokens per node (PDF only)')

    parser.add_argument('--if-add-node-id', type=str, default=None)
    parser.add_argument('--if-add-node-summary', type=str, default=None)
    parser.add_argument('--if-add-doc-description', type=str, default=None)
    parser.add_argument('--if-add-node-text', type=str, default=None)

    parser.add_argument('--if-thinning', type=str, default='no',
                        help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                        help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                        help='Token threshold for generating summaries (markdown only)')

    parser.add_argument('--verbose', dest='verbose', action='store_true', default=True,
                        help='Stream progress to stderr (default: on)')
    parser.add_argument('--quiet', dest='verbose', action='store_false',
                        help='Disable verbose stderr stream')
    parser.add_argument('--log-level', type=str, default='WARNING',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Python logging level for stderr (litellm, asyncio, etc.)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Tee stderr stream to this file. Default: ./logs/run_<ts>.log. '
                             'Pass "none" to disable.')
    args = parser.parse_args()
    # Seed _LLM_RUNTIME from CLI args before any pipeline call; helpers in
    # pageindex.utils / pageindex.page_index pick up these values via
    # `get_llm_runtime_value` and fall back to env vars when None.
    configure_llm_runtime(
        vllm_base_url=args.base_url,
        ollama_base_url=args.base_url,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_max_tokens=args.vllm_max_tokens,
        vllm_timeout=args.vllm_timeout,
        vllm_ctx_margin=args.vllm_ctx_margin,
        llm_concurrency=args.llm_concurrency,
        pageindex_group_max_tokens=args.group_max_tokens,
        pageindex_group_prompt_overhead=args.group_prompt_overhead,
        toc_chunk_max_tokens=args.toc_chunk_max_tokens,
    )

    global _VERBOSE
    _VERBOSE = args.verbose
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    log_file_path = None
    if args.log_file is None:
        os.makedirs("./logs", exist_ok=True)
        log_file_path = f"./logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    elif args.log_file.lower() != "none":
        log_file_path = args.log_file
    if log_file_path:
        resolved = _setup_log_file(log_file_path)
        print(f"[{_ts()}] [log-file] {resolved}", file=sys.stderr, flush=True)

    if not args.pdf_path and not args.md_path:
        raise SystemExit("Either --pdf_path or --md_path must be specified")
    if args.pdf_path and args.md_path:
        raise SystemExit("Only one of --pdf_path or --md_path can be specified")

    if args.pdf_path:
        if not args.pdf_path.lower().endswith('.pdf'):
            raise SystemExit("PDF file must have .pdf extension")
        if not os.path.isfile(args.pdf_path):
            raise SystemExit(f"PDF file not found: {args.pdf_path}")

        user_opt = {
            'model': args.model,
            'toc_check_page_num': args.toc_check_pages,
            'toc_verify_sample_num': args.toc_verify_sample,
            'max_page_num_each_node': args.max_pages_per_node,
            'max_token_num_each_node': args.max_tokens_per_node,
            'if_add_node_id': args.if_add_node_id,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
        }
        opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

        if _VERBOSE:
            print(f"[{_ts()}] [config] {vars(opt)}", file=sys.stderr, flush=True)

        # Phase 0.1 — preflight the vLLM server (no-op for non-vLLM models).
        _preflight_vllm(opt.model, args.base_url)

        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        output_dir = './results'
        output_file = f'{output_dir}/{pdf_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)

        # Phase 0.2 — wire stage checkpointing before the pipeline runs.
        global _CHECKPOINT_PATH
        _CHECKPOINT_PATH = f'{output_dir}/{pdf_name}_structure.partial.json'
        _install_checkpoint_hooks()

        # Phase 0.3 — instrument LLM calls + start heartbeat thread.
        _install_llm_counters()
        _start_heartbeat(interval=15.0)

        try:
            with _phase(f"PDF pipeline: {args.pdf_path}"):
                result = _page_index_main(args.pdf_path, opt)

            with _phase(f"Save → {output_file}"):
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
        finally:
            _stop_heartbeat()

        # Final artifact landed; the partial checkpoint is now stale.
        try:
            if _CHECKPOINT_PATH and os.path.exists(_CHECKPOINT_PATH):
                os.remove(_CHECKPOINT_PATH)
        except OSError:
            pass

        print(f'Tree structure saved to: {output_file}')

    elif args.md_path:
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise SystemExit("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise SystemExit(f"Markdown file not found: {args.md_path}")

        import asyncio
        config_loader = ConfigLoader()
        user_opt = {
            'model': args.model,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
            'if_add_node_id': args.if_add_node_id,
        }
        opt = config_loader.load({k: v for k, v in user_opt.items() if v is not None})

        if _VERBOSE:
            print(f"[{_ts()}] [config] {vars(opt)}", file=sys.stderr, flush=True)

        # Phase 0.1 — preflight (no-op for non-vLLM models).
        _preflight_vllm(opt.model, args.base_url)
        # Phase 0.3 — counters + heartbeat. Checkpointing hooks live on
        # pageindex.page_index, not the markdown pipeline.
        _install_llm_counters()
        _start_heartbeat(interval=15.0)

        try:
            with _phase(f"Markdown pipeline: {args.md_path}"):
                result = asyncio.run(md_to_tree(
                    md_path=args.md_path,
                    if_thinning=args.if_thinning.lower() == 'yes',
                    min_token_threshold=args.thinning_threshold,
                    if_add_node_summary=opt.if_add_node_summary,
                    summary_token_threshold=args.summary_token_threshold,
                    model=opt.model,
                    if_add_doc_description=opt.if_add_doc_description,
                    if_add_node_text=opt.if_add_node_text,
                    if_add_node_id=opt.if_add_node_id,
                ))

            md_name = os.path.splitext(os.path.basename(args.md_path))[0]
            output_dir = './results'
            output_file = f'{output_dir}/{md_name}_structure.json'
            os.makedirs(output_dir, exist_ok=True)

            with _phase(f"Save → {output_file}"):
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
        finally:
            _stop_heartbeat()

        print(f'Tree structure saved to: {output_file}')


if __name__ == "__main__":
    main()
