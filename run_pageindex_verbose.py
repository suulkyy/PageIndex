"""
Verbose version of run_pageindex.py.

Same CLI as run_pageindex.py, plus:
  --verbose / --quiet     toggle live progress stream (default: verbose ON)
  --log-level             python logging level for litellm/asyncio chatter

Live progress is written to stderr so stdout stays clean (the final save line
still prints to stdout). Internally this:
  1. Wraps `pageindex.utils.JsonLogger` so every `.info` / `.error` call also
     emits a single-line summary to stderr (the original JSON file under
     ./logs/ is still written).
  2. Times each top-level phase: PDF parse, tree_parser, post-processing,
     summary generation, doc description, save.
  3. Forwards the existing scattered `print(...)` statements unchanged (they
     already announce TOC detection, mode selection, accuracy, retries, etc.).
"""
import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime

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
    """Mirror writes to stderr and an open log file."""
    def __init__(self, stream, file_obj):
        self._stream = stream
        self._file = file_obj
    def write(self, data):
        self._stream.write(data)
        try:
            self._file.write(data)
            self._file.flush()
        except Exception:
            pass
        return len(data) if isinstance(data, str) else 0
    def flush(self):
        self._stream.flush()
        try:
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
    """Tee stderr to log file. Reconfigure root logging so prior handlers
    inherit the new sys.stderr. Returns resolved Path."""
    from pathlib import Path as _Path
    p = _Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "a", encoding="utf-8", buffering=1)
    f.write(f"\n[{_ts()}] === log start: {p} ===\n")
    sys.stderr = _Tee(sys.stderr, f)
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
from pageindex.page_index_md import md_to_tree  # noqa: E402
from pageindex.utils import ConfigLoader  # noqa: E402
from pageindex.page_index import tree_parser as _tree_parser  # noqa: E402


def _verbose_page_index_main(doc, opt):
    """Mirror pageindex.page_index.page_index_main with per-phase timing.
    Keep this pipeline aligned with page_index_main when upstream changes.
    """
    import asyncio
    from io import BytesIO
    from pageindex.utils import (
        get_page_tokens, write_node_id, add_node_text,
        generate_summaries_for_structure, remove_structure_text,
        generate_doc_description, format_structure, get_pdf_name,
        create_clean_structure_for_description,
    )

    logger = _VerboseJsonLogger(doc)

    is_valid_pdf = (
        (isinstance(doc, str) and os.path.isfile(doc) and doc.lower().endswith(".pdf"))
        or isinstance(doc, BytesIO)
    )
    if not is_valid_pdf:
        raise ValueError("Unsupported input type. Expected a PDF file path or BytesIO object.")

    with _phase("Parse PDF + tokenize pages"):
        page_list = get_page_tokens(doc, model=opt.model)
        total_tokens = sum(p[1] for p in page_list)
        logger.info({'total_page_number': len(page_list)})
        logger.info({'total_token': total_tokens})
        if _VERBOSE:
            print(f"[{_ts()}]   pages={len(page_list)}  tokens={total_tokens}", file=sys.stderr, flush=True)

    _ORDER = ['title', 'node_id', 'start_index', 'end_index', 'summary', 'text', 'nodes']

    async def _build():
        with _phase("Build tree (tree_parser)"):
            structure = await _tree_parser(page_list, opt, doc=doc, logger=logger)

        if opt.if_add_node_id == 'yes':
            with _phase("Assign node_id"):
                write_node_id(structure)

        if opt.if_add_node_text == 'yes':
            with _phase("Attach node text"):
                add_node_text(structure, page_list)

        doc_description = None
        if opt.if_add_node_summary == 'yes':
            with _phase("Generate node summaries"):
                if opt.if_add_node_text == 'no':
                    add_node_text(structure, page_list)
                await generate_summaries_for_structure(structure, model=opt.model)
                if opt.if_add_node_text == 'no':
                    remove_structure_text(structure)
            if opt.if_add_doc_description == 'yes':
                with _phase("Generate doc description"):
                    clean_structure = create_clean_structure_for_description(structure)
                    doc_description = generate_doc_description(clean_structure, model=opt.model)

        with _phase("Format structure"):
            structure = format_structure(structure, order=_ORDER)

        result = {'doc_name': get_pdf_name(doc), 'structure': structure}
        if doc_description is not None:
            result['doc_description'] = doc_description
            # Match upstream key ordering: doc_name, doc_description, structure
            result = {'doc_name': result['doc_name'],
                      'doc_description': doc_description,
                      'structure': structure}
        return result

    return asyncio.run(_build())


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
            result = _verbose_page_index_main(args.pdf_path, opt)

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
