# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PageIndex builds a hierarchical "table-of-contents" tree from long PDFs (or Markdown) and uses LLM tree search for vectorless, reasoning-based RAG. No vector DB, no chunking. Output is a JSON tree of nodes with `title`, `node_id`, `start_index`/`end_index` (PDF) or `line_num` (MD), optional `summary`/`text`, and nested `nodes`.

## Setup

```bash
pip3 install --upgrade -r requirements.txt          # runtime
pip3 install openai-agents                          # only for retrieval/agent demos
```

`.env` must contain at least `OPENAI_API_KEY` (alias `CHATGPT_API_KEY` accepted). Multi-provider routing goes through LiteLLM, so any `provider/model` string LiteLLM supports works (e.g. `anthropic/claude-sonnet-4-6`).

`pyproject.toml` declares `requires-python = ">=3.12"`. `uv.lock` is checked in — use `uv sync` if working with uv.

## Commands

### Tree generation
```bash
python3 run_pageindex.py --pdf_path path/to/doc.pdf
python3 run_pageindex.py --md_path  path/to/doc.md
```
Output: `./results/<basename>_structure.json`. JSON logger writes per-doc trace to `./logs/`.

Verbose variant streams phase timings + log previews to stderr (final save still on stdout):
```bash
python3 run_pageindex_verbose.py --pdf_path doc.pdf [--quiet] [--log-level DEBUG] [--log-file path]
```
Flags specific to the verbose script:
- `--verbose` / `--quiet` — toggle live stderr progress stream (default ON). `--quiet` keeps phase timing but suppresses per-log-line previews.
- `--log-level` — python `logging` level (default `WARNING`); raise to `INFO`/`DEBUG` for litellm + asyncio chatter on stderr.
- `--log-file` — override tee destination (default `./logs/run_<YYYYMMDD_HHMMSS>.log`). Pass `none` to disable. Both `sys.stdout` and `sys.stderr` are tee'd into this file, so the upstream `print(...)` calls (`Parsing PDF...`, TOC mode/accuracy chatter, final save line) and any tracebacks land in the log alongside the streamed JsonLogger previews. Bare lines get a `[YYYY-MM-DD HH:MM:SS.mmm]` prefix in the file copy; lines that already start with a timestamp (logger format, phase markers) pass through unchanged so timestamps don't double up. Terminal output is not modified.

The verbose script delegates the actual pipeline to upstream `pageindex.page_index.page_index_main` (PDF) and `pageindex.page_index_md.md_to_tree` (MD) — same return shape, key ordering, and branching as `run_pageindex.py`. Verbose adds only: JsonLogger monkey-patch (live stderr previews), `_phase` timer around the whole PDF/MD pipeline + save, and the stdout/stderr tee.
All other flags match `run_pageindex.py`.

Common flags (all override `pageindex/config.yaml`): `--model`, `--toc-check-pages`, `--max-pages-per-node`, `--max-tokens-per-node`, `--if-add-node-id yes|no`, `--if-add-node-summary`, `--if-add-doc-description`, `--if-add-node-text`. Markdown-only: `--if-thinning`, `--thinning-threshold`, `--summary-token-threshold`.

#### Running with custom Ollama / vLLM models

Tree-build LLM calls go through `llm_completion` / `llm_acompletion` (LiteLLM). Prefix `--model` to route by provider; `pageindex/utils.py:_provider_kwargs` auto-injects `api_base` from env vars.

**Ollama** — prefix `ollama/` or `ollama_chat/`:
```bash
OLLAMA_API_BASE=http://localhost:11434 \
python3 run_pageindex_verbose.py --pdf_path doc.pdf \
  --model ollama_chat/qwen3:14b
```
- Base URL env (first wins): `OLLAMA_API_BASE` → `OLLAMA_BASE_URL` → `OLLAMA_HOST`. Defaults to LiteLLM's built-in if unset.
- `OLLAMA_TIMEOUT` — request timeout in seconds (default `1800`).
- `OLLAMA_THINK` — set `true`/`1`/`yes` to keep reasoning output for qwen3, deepseek-r1, etc. Default off (stripped) so JSON parsers don't choke on `<think>` blocks.

**vLLM (HTTP server)** — prefix `hosted_vllm/`, point at OpenAI-compatible endpoint (local or remote host):
```bash
# local
VLLM_API_BASE=http://localhost:8000/v1 \
python3 run_pageindex_verbose.py --pdf_path doc.pdf \
  --model hosted_vllm/meta-llama/Llama-3.1-8B-Instruct

# remote host (e.g. shared GPU box)
VLLM_API_BASE=http://<VLLM_HOST>:8000/v1 \
python3 run_pageindex_verbose.py --pdf_path doc.pdf \
  --model hosted_vllm/Qwen/Qwen3.5-9B
```
- Use `hosted_vllm/` (not bare `vllm/`) for HTTP-served vLLM. LiteLLM's `vllm/` prefix invokes offline batching and tries `import vllm` locally — fails with `VLLMException - No module named 'vllm'` when calling a remote server.
- Base URL env: `VLLM_API_BASE` or `VLLM_BASE_URL`. Required if vLLM not on default LiteLLM target. Set to remote host's `:8000/v1` to use a non-local server.
- Model string after `hosted_vllm/` must match the `--served-model-name` that vLLM serves (verify via `curl $VLLM_API_BASE/models`).
- `pageindex/utils.py:_provider_kwargs` accepts both `vllm/` and `hosted_vllm/` for env-driven `api_base` injection.

Outputs land in `./results/` regardless of provider. Scratch dirs `results_ollama/` and `results_vllm/` in repo root are user-managed copies — not auto-populated.

For best retrieval, pass `--if-add-node-text yes --if-add-node-id yes` (else agent must answer from summaries alone).

### Local retrieval over a folder of trees
```bash
python3 retrieve_pageindex.py --folder ./results --question "..."
# providers: auto (default), openai, vllm, ollama, litellm
# prefix model with ollama/, vllm/, openai/, litellm/ for auto-detect
python3 retrieve_pageindex.py --folder ./results --question "..." --provider ollama --model llama3.1:8b
```
Logs: `./logs/retrieve_<YYYYMMDD_HHMMSS>.log` (`--log-file` to override). Both `sys.stdout` and `sys.stderr` are tee'd into this file, so the agent's streamed reasoning, tool-call/tool-output prints, final answer, and any tracebacks all land in the log. Bare lines get a `[YYYY-MM-DD HH:MM:SS.mmm]` prefix in the file copy; logger lines (already formatted with asctime) pass through unchanged. `--verbose` prints tool args + output previews.

#### Running retrieval with custom Ollama / vLLM models

`retrieve_pageindex.py` uses `openai-agents` (Chat Completions API) for local servers, and routes LiteLLM-supported strings through the LiteLLM extension. `resolve_provider` strips `ollama/`, `vllm/`, `openai/`, `litellm/` prefixes; explicit `--provider` overrides auto-detect.

**Ollama** — OpenAI-compatible endpoint at `/v1`:
```bash
OLLAMA_BASE_URL=http://localhost:11434/v1 \
python3 retrieve_pageindex.py --folder ./results --question "..." \
  --provider ollama --model llama3.1:8b
# or via prefix auto-detect:
python3 retrieve_pageindex.py --folder ./results --question "..." \
  --model ollama/qwen3:14b
```
- Default `base_url` if unset: `http://localhost:11434/v1`. Override with `--base-url` or `OLLAMA_BASE_URL`.
- API key fallback: `OLLAMA_API_KEY` → literal `"EMPTY"` (Ollama ignores it but `AsyncOpenAI` requires non-empty).
- Note: this script reads `OLLAMA_BASE_URL` (with `/v1`), **not** `OLLAMA_API_BASE` used by the tree-build path.

**vLLM** — OpenAI-compatible (local or remote host):
```bash
# local
VLLM_BASE_URL=http://localhost:8000/v1 \
python3 retrieve_pageindex.py --folder ./results --question "..." \
  --provider vllm --model meta-llama/Llama-3.1-8B-Instruct

# remote host
VLLM_BASE_URL=http://<VLLM_HOST>:8000/v1 \
python3 retrieve_pageindex.py --folder ./results --question "..." \
  --provider vllm --model Qwen/Qwen3.5-9B

# or prefix auto-detect (combine with env or --base-url):
python3 retrieve_pageindex.py --folder ./results --question "..." \
  --base-url http://<VLLM_HOST>:8000/v1 \
  --model vllm/Qwen/Qwen3.5-9B
```
- Default `base_url` if unset: `http://localhost:8000/v1`. Override with `--base-url` or `VLLM_BASE_URL` to point at remote vLLM host.
- vLLM server must be launched with `--enable-auto-tool-choice` and `--tool-call-parser <parser>` (e.g. `hermes`, `llama3_json`). Without these, the agent's tool calls return `400 - "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`.
- API key fallback: `VLLM_API_KEY` → `"EMPTY"`.
- Model name must match vLLM's `--served-model-name`.

**LiteLLM passthrough** — for non-OpenAI providers (Anthropic, Bedrock, etc.) use `--provider litellm` with full `provider/model` string; requires `pip install openai-agents-litellm`.

vllm/ollama paths build an `AsyncOpenAI` client + `OpenAIChatCompletionsModel` (Chat Completions, not Responses API) — agent streams reasoning + tool calls live regardless.

### Agentic demo
```bash
python3 examples/agentic_vectorless_rag_demo.py
```

### Cookbook notebooks (`cookbook/`)
- `pageindex_RAG_simple.ipynb` — minimal vectorless RAG walkthrough.
- `vision_RAG_pageindex.ipynb` — page-image (no-OCR) reasoning RAG.
- `agentic_retrieval.ipynb` — agent-driven retrieval over a built tree.
- `pageIndex_chat_quickstart.ipynb` — chat-style quickstart on top of PageIndex output.

### Tests / lint
None configured. No pytest, ruff, or CI test workflow. CI workflows under `.github/workflows/` are for CodeQL, dependency review, and issue dedupe — not project tests.

## Architecture

### Pipeline (`pageindex/page_index.py`, ~1150 lines, the core)
Entrypoint: `page_index_main(doc, opt) → page_index(...)` (sync wrappers around an async builder). Flow:

1. `get_page_tokens` — parse PDF via PyPDF2 (default) or pymupdf, return `[(text, token_count), ...]`.
2. `check_toc` — LLM inspects first `toc_check_page_num` pages, decides one of three modes:
   - `process_toc_with_page_numbers` (TOC found with page refs)
   - `process_toc_no_page_numbers` (TOC found, no refs)
   - `process_no_toc` (synthesize from content)
3. `meta_processor` runs the chosen mode → produces a flat `toc_with_page_number` list of `{title, physical_index, list_index}`. Then `verify_toc` LLM-checks each title against its claimed page; if accuracy 0.6–1.0 it calls `fix_incorrect_toc_with_retries` (up to 3 attempts); below 0.6 it falls back to the next mode.
4. `validate_and_truncate_physical_indices` nulls out indices past EOD.
5. `add_preface_if_needed` + `check_title_appearance_in_start_concurrent` align titles to page boundaries.
6. `post_processing` (in `utils.py`) converts the flat list into a nested tree.
7. `process_large_node_recursively` — for any node with span > `max_page_num_each_node` AND tokens ≥ `max_token_num_each_node`, recurse with `process_no_toc` to subdivide.
8. Optional passes: `write_node_id` (zero-padded `0001`-style ids), `add_node_text` (slices page text into nodes), `generate_summaries_for_structure` (LLM, concurrent), `generate_doc_description` (top-level LLM summary).

Markdown path (`pageindex/page_index_md.py`, `md_to_tree`) is a separate async pipeline keyed off `#`-heading levels — no PDF parsing, no TOC detection, optional thinning to merge sparse subsections.

### Config (`pageindex/utils.py:ConfigLoader`)
Defaults live in `pageindex/config.yaml`; user dict is validated against that key set (unknown keys raise) then merged. Returns a `SimpleNamespace`. CLI scripts collect args, drop `None`, then call `ConfigLoader().load(user_opt)`.

`config.yaml` keys: `model`, `retrieve_model` (falls back to `model`), `toc_check_page_num`, `max_page_num_each_node`, `max_token_num_each_node`, `if_add_node_id`, `if_add_node_summary`, `if_add_doc_description`, `if_add_node_text`. Yes/no flags are strings, not bools.

### LLM layer (`pageindex/utils.py`)
All LLM calls go through `llm_completion` / `llm_acompletion` — both wrap LiteLLM with `temperature=0`, retry up to 10× with 1s backoff, and strip a `litellm/` prefix before dispatch (LiteLLM picks provider from the rest of the string). `litellm.drop_params = True` — unsupported params silently dropped per provider. `extract_json` tolerates code fences and partial JSON when parsing model replies.

### Retrieval (`pageindex/retrieve.py`, `pageindex/client.py`)
`retrieve.py` exposes 3 stateless tools over a `documents` dict: `get_document`, `get_document_structure` (text fields stripped), `get_page_content` (PDF: page nums; MD: line nums via `_get_md_page_content`). `client.py:PageIndexClient` is the user-facing wrapper — `index()` writes trees + cached page text into a workspace dir keyed by uuid; tools resolve through it. `_normalize_retrieve_model` keeps `litellm/` and `openai/` as passthrough but rewrites bare `provider/model` to `litellm/provider/model` so the OpenAI Agents SDK routes via LiteLLM.

### Standalone retriever (`retrieve_pageindex.py`)
Loads `*.json` from a folder, infers PDF vs MD per-file by node fields (`start_index` → pdf, `line_num` → md), exposes `list_documents` / `get_document` / `get_document_structure` / `get_node_content` to an `openai-agents` `Agent`. `resolve_provider` strips known prefixes; `_build_agent_model` constructs an `AsyncOpenAI` client + `OpenAIChatCompletionsModel` for vllm/ollama (Chat Completions, not Responses API). Streams reasoning + tool calls live; falls back to `asyncio.run` in a worker thread when called from inside an existing loop.

### Logging (`pageindex/utils.py:JsonLogger`)
Per-document JSON-line logger writing to `./logs/<sanitized_doc_name>.log`. `run_pageindex_verbose.py` monkey-patches it to also emit single-line stderr previews, then tees both `sys.stdout` and `sys.stderr` into a separate timestamped tee log (`./logs/run_<YYYYMMDD_HHMMSS>.log`) so all upstream prints + tracebacks are captured. The tee adds `[YYYY-MM-DD HH:MM:SS.mmm]` per bare line in the file copy; lines that already begin with a timestamp pattern pass through unchanged. `retrieve_pageindex.py` does the same stdout+stderr tee into `./logs/retrieve_<YYYYMMDD_HHMMSS>.log`, with a single `StreamHandler` pointed at the (already-tee'd) `sys.stderr` — there's no separate `FileHandler`, so logger lines reach the file via the tee and never double-write.

## Conventions worth knowing

- Page indices are **1-based**, inclusive at both ends.
- `physical_index` is internal during construction; the persisted output uses `start_index`/`end_index`.
- `format_structure(..., order=[...])` enforces field ordering before serialization — keep new fields out of the order list if you don't want them surfaced.
- `if_add_node_summary=yes` forces `add_node_text` even when `if_add_node_text=no`, then strips text after summarization (`remove_structure_text`).
- The "results" directory committed in `examples/documents/` was deleted in the working tree (see `git status`); regenerate with `run_pageindex.py` if needed.
