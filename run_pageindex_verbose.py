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
import time
from contextlib import contextmanager
from datetime import datetime

_TS_RE = re.compile(r"^\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")

# Wire stderr logging early so any module-level logging calls during import
# are captured.
_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format=_LOG_FORMAT)

# ── Verbose JsonLogger wrapper ────────────────────────────────────────────────

import pageindex.utils as _pi_utils  # noqa: E402

_OriginalJsonLogger = _pi_utils.JsonLogger
_VERBOSE = True  # toggled by CLI


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
import pageindex.page_index as _pi_pdf  # noqa: E402
import pageindex.page_index_md as _pi_md  # noqa: E402
_pi_pdf.JsonLogger = _VerboseJsonLogger
_pi_md.JsonLogger = _VerboseJsonLogger


# ── Phase timer helper ────────────────────────────────────────────────────────

@contextmanager
def _phase(name: str):
    if _VERBOSE:
        print(f"\n[{_ts()}] === {name} ===", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if _VERBOSE:
            dt = time.perf_counter() - t0
            print(f"[{_ts()}] === {name} done in {dt:.2f}s ===", file=sys.stderr, flush=True)


# ── Patched page_index_main / md_to_tree wrappers ─────────────────────────────

from pageindex import *  # noqa: E402,F401,F403
from pageindex.page_index import page_index_main as _page_index_main  # noqa: E402
from pageindex.page_index_md import md_to_tree  # noqa: E402
from pageindex.utils import ConfigLoader  # noqa: E402


# ── CLI (mirrors run_pageindex.py) ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Process PDF or Markdown document and generate structure (verbose).'
    )
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

    parser.add_argument('--model', type=str, default=None, help='Model to use (overrides config.yaml)')

    parser.add_argument('--toc-check-pages', type=int, default=None,
                        help='Number of pages to check for table of contents (PDF only)')
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

        with _phase(f"PDF pipeline: {args.pdf_path}"):
            result = _page_index_main(args.pdf_path, opt)

        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        output_dir = './results'
        output_file = f'{output_dir}/{pdf_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)

        with _phase(f"Save → {output_file}"):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

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

        print(f'Tree structure saved to: {output_file}')


if __name__ == "__main__":
    main()
