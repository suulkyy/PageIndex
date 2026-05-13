# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PageIndex builds a hierarchical "table-of-contents" tree from long PDFs (or Markdown) and uses LLM tree search for vectorless, reasoning-based RAG. No vector DB, no chunking. Output is a JSON tree of nodes with `title`, `node_id`, `start_index`/`end_index` (PDF) or `line_num` (MD), optional `summary`/`text`, and nested `nodes`.

## Repo layout (tracked files only)

- Top level: `run_pageindex.py`, `run_pageindex_verbose.py`, `retrieve_pageindex.py`, `pyproject.toml`, `requirements.txt`, `LICENSE`, `README.md`, `AGENTS.md`, `CLAUDE.md`.
- `pageindex/` package: `page_index.py` (PDF pipeline), `page_index_md.py` (Markdown pipeline), `utils.py` (LLM layer, config loader, helpers), `retrieve.py` + `client.py` (programmatic retrieval), `config.yaml` (defaults), `__init__.py`.
- `cookbook/` notebooks: `pageindex_RAG_simple.ipynb`, `vision_RAG_pageindex.ipynb`, `agentic_retrieval.ipynb`, `pageIndex_chat_quickstart.ipynb`, `README.md`.
- `examples/` — `agentic_vectorless_rag_demo.py`, `documents/` (sample PDFs + pre-built `_structure.json` outputs under `documents/results/`), `tutorials/doc-search/`, `tutorials/tree-search/`, `workspace/` (sample workspace dump used by `PageIndexClient`).
- `.github/` — workflows for CodeQL, dependency review, and issue dedupe (`autoclose-labeled-issues`, `backfill-dedupe`, `issue-dedupe`, `remove-autoclose-label`); `.claude/commands/dedupe.md` is the dedupe agent prompt.

## Setup

```bash
# uv-managed (preferred — pyproject.toml pins everything)
uv sync

# pip-only fallback
pip3 install --upgrade -r requirements.txt
pip3 install openai-agents       # required for retrieve_pageindex.py + agent demos
pip3 install json-repair          # required by extract_json fallback path
```

`pyproject.toml` already pins `json-repair>=0.59.5` and `openai-agents[litellm]>=0.10.5`, so `uv sync` covers both. The `requirements.txt` path leaves them optional/commented — install manually if you don't use uv.

`pyproject.toml` declares `requires-python = ">=3.12"`. Run scripts via `uv run python <script.py>` (bare `python3` misses uv-managed deps like `python-dotenv`, `litellm`, `json-repair`).

`.env` is gitignored via `.env*`; a tracked template lives at `env.example` (no leading dot, otherwise the glob would catch it) — copy to `.env` and fill in `<VLLM_HOST>`. `.env` must contain at least `OPENAI_API_KEY` (alias `CHATGPT_API_KEY` accepted) when using hosted OpenAI; for local vLLM/Ollama the `VLLM_API_KEY`/`OLLAMA_API_KEY` placeholders in `env.example` are sufficient. Multi-provider routing goes through LiteLLM, so any `provider/model` string LiteLLM supports works (e.g. `anthropic/claude-sonnet-4-6`).

### Sample `.env` for a remote vLLM (Qwen3.6-27B-AWQ-INT4 at `--max-model-len 65536`)

Tuned for the dedicated remote LLM box at `<VLLM_HOST>` (2× Quadro RTX 5000, Turing CC 7.5, 32 GB total):

```bash
vllm serve cyankiwi/Qwen3.6-27B-AWQ-INT4 \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --max-model-len 65536 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.92 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --served-model-name rag-llm
```

Key points vs older Qwen3.5-9B recipe:
- **No YARN.** Qwen3.6 has 262k native context; `--max-model-len 65536` is well inside it. No `--hf-overrides rope_scaling` needed.
- **No `--enable-chunked-prefill`.** vLLM 0.20.x V1 engine forces chunked prefill on; the flag is silently ignored (and `--no-enable-chunked-prefill` errors). Drop it for cleanness.
- **No `--quantization` flag.** This build is packaged as `compressed-tensors` W4A16, not legacy AWQ. vLLM auto-detects from `config.json` and picks the Marlin INT4 kernel on Turing. Passing `--quantization awq_marlin` trips a mismatch check (`Quantization method specified in the model config (compressed-tensors) does not match the quantization method specified in the quantization argument`). Use `--quantization compressed-tensors` explicitly if you want belt-and-suspenders, but auto-detect is the simpler path and survives future model swaps.
- **AWQ-INT4 (~14 GB total weight)** leaves comfortable KV headroom on 2× 16 GB cards even at full `--gpu-memory-utilization 0.92` (LLM is the sole occupant; embedding + reranker live on a separate laptop GPU).
- **`qwen3_coder` is the correct tool parser** for the Qwen3.5/3.6 dense family (per vLLM's Qwen3.5 recipe). Don't swap to `hermes` — that's for Qwen3-Next-80B-A3B MoE.
- Tree-build/tool-loop turns disable thinking via `extra_body={"chat_template_kwargs":{"enable_thinking": false}}` (handled in `pageindex/utils.py:_provider_kwargs` and `retrieve_pageindex.py`). The `--reasoning-parser qwen3` flag is kept because vLLM's recipe recommends it; client-side per-request opt-outs are the intended pattern.

Sample `.env`:

```bash
# ─── Tree-build (run_pageindex.py / run_pageindex_verbose.py) ──────────────
VLLM_API_BASE=http://<VLLM_HOST>:8000/v1
VLLM_TIMEOUT=3600
PAGEINDEX_LLM_CONCURRENCY=8                  # match remote --max-num-seqs 8 to avoid KV preemption
VLLM_MAX_TOKENS=1024
VLLM_MAX_MODEL_LEN=65536
VLLM_CTX_MARGIN=1024
# VLLM_ENABLE_THINKING=true   # leave off — tree-build is JSON-only; thinking inflates wall-clock 2-4x with no quality gain

# ─── Retrieval (retrieve_pageindex.py) ──────────────────────────────────────
VLLM_BASE_URL=http://<VLLM_HOST>:8000/v1     # retrieval reads VLLM_BASE_URL (with /v1), not VLLM_API_BASE
VLLM_API_KEY=EMPTY                           # AsyncOpenAI requires non-empty; vLLM ignores value

PAGEINDEX_AGENT_MAX_TURNS=20                 # caps runaway loops; books rarely need >15 turns

# Bounded tool-output sizes — prevents the rolling agent history from
# overflowing 65k. The earlier 30,721-token retrieval failure on the annual-
# report PDF was caused by uncapped structure summaries; these defaults
# stop it from recurring.
PAGEINDEX_NODE_TEXT_MAX_CHARS=12000
PAGEINDEX_STRUCTURE_SUMMARY_MAX_CHARS=120
PAGEINDEX_RETRIEVE_THINKING_EVIDENCE_MAX_CHARS=20000
PAGEINDEX_TOOL_LOG_MAX_CHARS=4000

# ─── Default: throughput profile, no thinking on final answer ───────────────
# Flip both to enable CoT-refined answers (+5–15 s/query, biggest quality lever):
#   PAGEINDEX_RETRIEVE_ENABLE_THINKING=true
#   PAGEINDEX_AGENT_MAX_TOKENS=8192
PAGEINDEX_RETRIEVE_ENABLE_THINKING=false
PAGEINDEX_AGENT_MAX_TOKENS=2048
```

Pass `--model hosted_vllm/rag-llm` to `run_pageindex.py` and `--provider vllm --model rag-llm` to `retrieve_pageindex.py` — both pick up the URL from `.env` so no `--base-url` is needed.

Two-name gotcha: `VLLM_API_BASE` is read by the **tree-build** path (`pageindex/utils.py:_provider_kwargs`), `VLLM_BASE_URL` (with `/v1`) is read by the **retrieval** path (`retrieve_pageindex.py`). Set both to the same URL so either script works without `--base-url`.

## Commands

### Tree generation
```bash
python3 run_pageindex.py --pdf_path path/to/doc.pdf
python3 run_pageindex.py --md_path  path/to/doc.md
```
Output: `./results/<basename>_structure.json` (`results/` is created on first run; gitignored output dir). JSON logger writes per-doc trace to `./logs/` (also gitignored).

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

Common flags (all override `pageindex/config.yaml`): `--model`, `--toc-check-pages`, `--toc-verify-sample`, `--max-pages-per-node`, `--max-tokens-per-node`, `--if-add-node-id yes|no`, `--if-add-node-summary`, `--if-add-doc-description`, `--if-add-node-text`. Markdown-only: `--if-thinning`, `--thinning-threshold`, `--summary-token-threshold`.

Runtime/provider flags (also in both scripts; preferred over env fallbacks — see `pageindex/utils.py:configure_llm_runtime`):
- `--base-url` — OpenAI-compatible endpoint (vLLM/Ollama). Sets `vllm_base_url` / `ollama_base_url`.
- `--vllm-timeout` — request timeout in seconds (default `1800`).
- `--vllm-max-model-len` — server's `--max-model-len` value (default `16384`). Used by `_resolve_max_tokens` for headroom math.
- `--vllm-max-tokens` — per-request output cap (default `2048`; `0` leaves server default). Dynamically clamped further by `_resolve_max_tokens`.
- `--vllm-ctx-margin` — safety margin in tokens for chat-template overhead (default `256`).
- `--llm-concurrency` — max in-flight async LLM calls (default `8`). Match this to vLLM `--max-num-seqs`.
- `--group-max-tokens` — page grouping target for `process_no_toc` / chunked LLM calls. If unset, scales from `vllm_max_model_len − vllm_max_tokens − margin − group_prompt_overhead`, clamped to `[2000, min(model_len/2, 10000)]` for vLLM; defaults to `20000` for non-vLLM providers.
- `--group-prompt-overhead` — token budget reserved for the prompt template wrapping each grouped page block (default `1600`).
- `--toc-chunk-max-tokens` — max tokens per chunk when LLM-transforming a printed TOC into JSON (default `6000`). Used by `_toc_chunks` in the chunked LLM fallback.

Old env vars still work as fallbacks: `VLLM_API_BASE`/`VLLM_BASE_URL`, `VLLM_TIMEOUT`, `VLLM_MAX_MODEL_LEN`, `VLLM_MAX_TOKENS`, `VLLM_CTX_MARGIN`, `PAGEINDEX_LLM_CONCURRENCY`, `PAGEINDEX_GROUP_MAX_TOKENS`, `PAGEINDEX_GROUP_PROMPT_OVERHEAD`, `PAGEINDEX_TOC_CHUNK_MAX_TOKENS`. CLI args win.

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
- `VLLM_TIMEOUT` — request timeout in seconds (default `1800`). LiteLLM's built-in default for `hosted_vllm` is 600s, which times out on slow remote hosts running big-context models (e.g. `litellm.Timeout: Hosted_vllmException - Connection timed out after 600.0 seconds`). Raise via `VLLM_TIMEOUT=3600` etc.
- `PAGEINDEX_LLM_CONCURRENCY` — max in-flight async LLM calls (default `8`). Tree-build fans out via `asyncio.gather` at `verify_toc`, `check_title_appearance_in_start_concurrent`, `fix_incorrect_toc_with_retries`, `process_large_node_recursively`, and `generate_summaries_for_structure`. Without a cap, a long doc can fire 50+ concurrent requests at vLLM. On a single-GPU vLLM that overflows the GPU KV cache → vLLM preempts running sequences (KV usage observed dropping from ~99% to 0%, `Avg generation throughput` collapses to 0 t/s, requests cycle Running → Waiting). The cap lives in `pageindex/utils.py:_get_llm_sem` and wraps `llm_acompletion`. Raise on multi-GPU/big-VRAM hosts; lower if you still see preemption thrash.
- vLLM `--reasoning-parser` (e.g. `qwen3`) splits model output: `<think>...</think>` goes to `message.reasoning_content`, the rest to `message.content`. If the visible content ends up empty, `extract_json` fails with `Expecting value: line 1 column 1 (char 0)` and `Failed to parse JSON even after cleanup`. Two safeguards: (1) `_extract_message_text` in `pageindex/utils.py` falls back to `reasoning_content` (and LiteLLM's `provider_specific_fields`) when `content` is empty; (2) `extract_json` strips `<think>...</think>` blocks before parsing. Best practice: don't enable `--reasoning-parser` for non-thinking instruct models (Qwen3.5-9B Instruct, etc.) — drop the flag to keep all tokens in `content`.
- Qwen3 chat-template `enable_thinking` defaults to `True`. Even without `--reasoning-parser`, the model emits `<think>...</think>JSON` in raw `content`. If `max_tokens` truncates mid-think (no closing `</think>`), `extract_json`'s open-ended `<think>.*` fallback erases everything → empty string → the same `Expecting value: line 1 column 1` error. Mitigated by passing `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` per request — `pageindex/utils.py:_provider_kwargs` does this by default for `vllm/`/`hosted_vllm/` prefixes. Override:
  - `VLLM_ENABLE_THINKING=true` — opt back into thinking (raises a hard requirement to also raise `VLLM_MAX_TOKENS` so the trace + JSON both fit). Note: `run_pageindex_verbose.py` hard-pins `VLLM_ENABLE_THINKING=false` (and `OLLAMA_THINK=false`) at module import. Tree-build issues hundreds of small JSON-only calls; thinking traces multiply wall-clock 2-4× and add no value. The pin overrides any externally-exported value, so you can keep `VLLM_ENABLE_THINKING=true` exported globally for `retrieve_pageindex.py` and tree-build still runs in instruct mode.
  - `VLLM_MAX_TOKENS` — per-request `max_tokens` user cap (default `2048`, set `0` to leave server default). The actual `max_tokens` sent is the minimum of this cap and the dynamic headroom `max-model-len − prompt_tokens − margin` computed by `_resolve_max_tokens` per call (uses `litellm.token_counter`). PageIndex tree-build prompts can hit 14 k+ tokens (raw page text fed to TOC detectors); without dynamic clamping you get `litellm.ContextWindowExceededError: This model's maximum context length is 16384 tokens. However, you requested 2048 output tokens and your prompt contains at least 14337 input tokens, for a total of at least 16385 tokens.`
  - `VLLM_MAX_MODEL_LEN` — server's `--max-model-len` value (default `16384`). Used by `_resolve_max_tokens` to size headroom. Bump to match if you raise server context (`--max-model-len 32768` → `VLLM_MAX_MODEL_LEN=32768`).
  - `VLLM_CTX_MARGIN` — safety margin in tokens reserved beyond `prompt_tokens + max_tokens` (default `256`). Covers chat-template overhead, role tokens, end-of-turn markers, etc.

Outputs land in `./results/` regardless of provider. For best retrieval, pass `--if-add-node-text yes --if-add-node-id yes` (else agent must answer from summaries alone).

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
- Long-context KV pressure: agent message history grows every turn (tool outputs accumulate). Env caps blunt this:
  - `PAGEINDEX_NODE_TEXT_MAX_CHARS` — truncate `get_node_content` text payload (default `8000`, set `0` to disable). Truncated payload includes `truncated: true`, `original_text_chars`, `truncated_to_chars`, and a `note` pointing at this env.
  - `PAGEINDEX_STRUCTURE_SUMMARY_MAX_CHARS` — per-node summary preview returned by `get_document_structure` (default `160`; `0` removes summaries). The structure tool returns a navigation-sized compact tree (`title`, `node_id`, span fields, truncated `summary`, nested `nodes`) — not the raw full-text-stripped tree — because full summaries across textbook-scale docs can saturate local vLLM context before the agent ever requests node text.
  - `PAGEINDEX_AGENT_MAX_TOKENS` — `ModelSettings(max_tokens=…)` cap on per-turn output (default `2048`, or `8192` when thinking retrieval is enabled and this env is unset; set `0` to leave server default). Bounds vLLM's worst-case KV slot allocation.
  - `PAGEINDEX_AGENT_MAX_TURNS` — `Runner.run_streamed(max_turns=…)` (default `30`, was upstream `100`). Cuts off runaway loops on weak models before history saturates KV.
  - `PAGEINDEX_TOOL_LOG_MAX_CHARS` — log-file cap on per-tool-output bytes (default `4000`; `0` disables). Stdout preview is always 200 chars regardless of this cap.
  - `PAGEINDEX_RETRIEVE_ENABLE_THINKING` — vLLM final-answer thinking switch (default `false`). Tool-loop turns always run with thinking disabled to keep tool-call format reliable; only the final answer-refinement pass uses thinking when this is enabled. For Qwen servers launched with `--reasoning-parser qwen3`, even this final pass stays disabled to avoid vLLM emitting tool-call intent in `reasoning` with `content: null`.
  - `PAGEINDEX_RETRIEVE_THINKING_EVIDENCE_MAX_CHARS` — char budget for retrieved evidence fed into the final thinking pass (default `12000`).
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

### Pipeline (`pageindex/page_index.py`, ~1500 lines, the core)
Entrypoint: `page_index_main(doc, opt) → page_index(...)` (sync wrappers around an async builder). Flow:

1. `get_page_tokens` — parse PDF via PyPDF2 (default) or pymupdf, return `[(text, token_count), ...]`.
2. `check_toc` — LLM inspects first `toc_check_page_num` pages, decides one of three modes:
   - `process_toc_with_page_numbers` (TOC found with page refs)
   - `process_toc_no_page_numbers` (TOC found, no refs)
   - `process_no_toc` (synthesize from content)
3. `meta_processor` runs the chosen mode → produces a flat `toc_with_page_number` list of `{title, physical_index, list_index}`. Then `verify_toc` LLM-checks N entries (default `toc_verify_sample_num=40`; pass `0` to check all); if accuracy 0.6–1.0 it calls `fix_incorrect_toc_with_retries` (up to 3 attempts); below 0.6 it falls back to the next mode.
4. `validate_and_truncate_physical_indices` nulls out indices past EOD.
5. `add_preface_if_needed` + `check_title_appearance_in_start_concurrent` align titles to page boundaries. The concurrent checker first runs a local heuristic `check_title_appearance_in_start_heuristic` (normalized prefix match within first 1500 chars) — only items the heuristic can't classify fall through to an LLM call.
6. `post_processing` (in `utils.py`) converts the flat list into a nested tree.
7. `process_large_node_recursively` — for any node with span > `max_page_num_each_node` AND tokens ≥ `max_token_num_each_node`, recurse with `process_no_toc` to subdivide.
8. Optional passes: `write_node_id` (zero-padded `0001`-style ids), `add_node_text` (slices page text into nodes), `generate_summaries_for_structure` (LLM, concurrent), `generate_doc_description` (top-level LLM summary). For vLLM models, `generate_doc_description` token-budgets its prompt against `vllm_max_model_len − vllm_max_tokens − vllm_ctx_margin − 256` and progressively shrinks the structure (full → summaries truncated to 400 chars → depth ≤2 + 300 chars → depth ≤1 + 200 chars → depth ≤2 titles-only → depth ≤1 titles-only) until it fits. Without this, a 700+-page book like `examples/documents/PRML.pdf` produces an 85k-token clean structure that overflows even a 65k context. `create_clean_structure_for_description` accepts `max_depth` and `summary_max_chars` to drive the trim ladder.

**TOC heuristics that pre-empt LLM calls** (added for textbook-scale inputs like `examples/documents/PRML.pdf`):
- `parse_toc_content_heuristic` — deterministic line-by-line parser that emits `{structure, title, page}` items directly. `toc_transformer` uses it whenever it returns ≥8 items at confidence ≥0.65, skipping the LLM transform entirely.
- `_toc_chunks` + `_transform_toc_chunk_with_llm` — when the heuristic isn't confident enough, the TOC text is split into ≤`toc_chunk_max_tokens` chunks (default 6000) and transformed chunk-by-chunk, then merged via `_merge_toc_items`. Avoids one giant JSON-generation call that would blow `max-model-len`.
- `guess_page_offset_from_toc` — deterministic page-offset guess from the first content-page heuristic. Tried before falling back to `calculate_page_offset` over LLM-matched pairs.
- `_finish_toc_json` — when an LLM TOC call returns `finish_reason=length`, salvages all complete JSON objects from the truncated buffer and re-prompts the model to "continue the structure" (up to 3 attempts) instead of raising.

**Synthesize-from-content (`process_no_toc`) prompt bounds** (added to keep the fallback mode usable on textbook-scale inputs):
- `generate_toc_continue` ships only the *tail* of the accumulated TOC (default `tail_n=15`) instead of the entire tree. Previously it serialized every prior entry via `json.dumps(...indent=2)` on every iteration, so the prompt grew linearly with chunk count — an 800-page book accumulating hundreds of entries pushed prompts past 64 k tokens and tripped vLLM's `ContextWindowExceededError` on the final chunks. The tail gives the model enough context to continue numbering without ballooning the prompt.
- `process_no_toc`'s per-chunk continuation loop wraps each `generate_toc_continue` call in `try/except` and logs `chunk N/M` on failure instead of unwinding the whole pipeline. One bad chunk deep in a 30-minute run no longer torches all prior accumulated TOC work — downstream code receives a partial tree and fills the gap.

**Tree-build hierarchy recovery (`list_to_tree`)** (added so subsections still nest under their chapters when the LLM skips intermediate headings):
- The original `list_to_tree` only attached an item to its **immediate** parent by structure-prefix lookup (`"8.7.3.1"` → `"8.7.3"`). When a long-book LLM extraction emitted `"8 Optimization"` and `"8.7.3.1 E step"` but missed `"8.7"` / `"8.7.3"` (their headings landed on a different `process_no_toc` chunk), the orphan dropped straight to root level — producing the flat-top-level pathology where chapters and 4-deep subsections sit as siblings (observed: 791/860 pages of book1 fell outside any top-level node before the fix).
- New behavior: walk the prefix chain until an existing ancestor is found and **synthesize placeholder nodes** for the gap, marked with title `"Section <structure>"`. After all real children are attached, synthesized ancestors recompute their `start_index`/`end_index` from the descendant min/max so the placeholder span correctly envelopes its real subsections. The fast path (immediate parent exists) is unchanged, so well-formed short-doc trees behave identically. When **no ancestor exists at all** (lone orphan with no chapter context anywhere in the doc), the item still attaches to root — synthesis is skipped to avoid wrapping a single leaf in a chain of empty placeholders.

**Per-node summary prompt bounds** (added so book-scale inputs survive the summarization phase):
- `generate_node_summary` token-budgets the `node['text']` payload against `vllm_max_model_len − vllm_max_tokens − vllm_ctx_margin − 256 − 80` (the 80 is empirical scaffolding overhead for the prompt template). For nodes whose text fits, behavior is unchanged — single LLM call. For nodes that overflow (e.g. a long chapter where `process_large_node_recursively` couldn't subdivide because span > 10 pages but tokens were just below `max_token_num_each_node`, or vice versa), it switches to a map-reduce pass: `_split_text_for_token_budget` slices the text on paragraph/sentence boundaries into budget-fitting chunks, summarizes each, then summarizes the concatenated chunk-summaries. One additional reduce pass handles the rare case where chunk-summaries themselves overflow. Without this, an 860-page book hits `ValueError: vLLM prompt budget exceeded: prompt=74876 tok ... headroom=-10364` from `_resolve_max_tokens` after a 60-min pipeline.
- `generate_summaries_for_structure` uses `asyncio.gather(..., return_exceptions=True)` and falls back to a 1500-char head-only excerpt for any node whose summary call fails, logging `node_id`/`title` for each. Previously a single overlarge or transient-error node would crash the entire `gather` and discard the whole tree at the very last step of the pipeline.

Markdown path (`pageindex/page_index_md.py`, `md_to_tree`) is a separate async pipeline keyed off `#`-heading levels — no PDF parsing, no TOC detection, optional thinning to merge sparse subsections.

### Config (`pageindex/utils.py:ConfigLoader`)
Defaults live in `pageindex/config.yaml`; user dict is validated against that key set (unknown keys raise) then merged. Returns a `SimpleNamespace`. CLI scripts collect args, drop `None`, then call `ConfigLoader().load(user_opt)`.

`config.yaml` keys: `model`, `retrieve_model` (falls back to `model`), `toc_check_page_num`, `toc_verify_sample_num`, `max_page_num_each_node`, `max_token_num_each_node`, `if_add_node_id`, `if_add_node_summary`, `if_add_doc_description`, `if_add_node_text`. Yes/no flags are strings, not bools.

Runtime knobs that change the LLM layer (concurrency, vLLM/Ollama URLs, max-tokens, group sizes) are NOT in `config.yaml` — they live in the per-process `_LLM_RUNTIME` dict managed by `pageindex/utils.py:configure_llm_runtime(**kwargs)`. CLI scripts call this once after `argparse.parse_args()` so subsequent `_provider_kwargs` / `_resolve_max_tokens` / `_get_llm_sem` reads pick up the values; absent CLI args, helpers fall back to the matching env vars (`get_llm_runtime_value(name, env_names, default)`).

### LLM layer (`pageindex/utils.py`)
All LLM calls go through `llm_completion` / `llm_acompletion` — both wrap LiteLLM with `temperature=0`, retry up to 10× with 1s backoff, and strip a `litellm/` prefix before dispatch (LiteLLM picks provider from the rest of the string). `litellm.drop_params = True` — unsupported params silently dropped per provider. `extract_json` tolerates code fences and partial JSON when parsing model replies. Cleanup chain: (1) raw_decode of first valid JSON, (2) `json.loads` after newline/whitespace normalize, (3) trailing-comma strip + `json.loads`, (4) `json_repair.repair_json` final fallback (handles missing commas between objects like `}{`, unescaped quotes inside strings, single-quoted keys, unquoted keys, trailing commas — common Qwen3.5-9B malformations seen as `Expecting ',' delimiter: line 1 column N`).

### Retrieval (`pageindex/retrieve.py`, `pageindex/client.py`)
`retrieve.py` exposes 3 stateless tools over a `documents` dict: `get_document`, `get_document_structure` (text fields stripped), `get_page_content` (PDF: page nums; MD: line nums via `_get_md_page_content`). `client.py:PageIndexClient` is the user-facing wrapper — `index()` writes trees + cached page text into a workspace dir keyed by uuid; tools resolve through it. A sample workspace lives at `examples/workspace/` (`_meta.json` + uuid-keyed JSON) for hands-on exploration. `_normalize_retrieve_model` keeps `litellm/` and `openai/` as passthrough but rewrites bare `provider/model` to `litellm/provider/model` so the OpenAI Agents SDK routes via LiteLLM.

### Standalone retriever (`retrieve_pageindex.py`)
Loads `*.json` from a folder, infers PDF vs MD per-file by node fields (`start_index` → pdf, `line_num` → md). The agent gets **5 tools**:
- `list_documents()`
- `get_document(doc_id)` — metadata only.
- `get_document_structure(doc_id)` — returns a navigation-sized compact tree (`_compact_structure_for_tool`): `title`, `node_id`, span fields, truncated `summary` (≤ `PAGEINDEX_STRUCTURE_SUMMARY_MAX_CHARS=160` chars), nested `nodes`. Full text is never inlined here.
- `get_node_content(doc_id, node_id)` — node payload; text capped at `PAGEINDEX_NODE_TEXT_MAX_CHARS=8000` chars by default with truncation metadata when applied.
- `answer(answer)` — explicit final-answer tool. The agent is configured with `tool_use_behavior={"stop_at_tool_names": ["answer"]}` so calling it terminates the run and emits the argument as the final response. Added because some weak local models hallucinated a fake `answer` tool call mid-stream and crashed the runner.

The agent is constructed with `parallel_tool_calls=False` (serial tool calling) — keeps tool order deterministic and avoids weak models interleaving partial calls. `resolve_provider` strips known prefixes; `_build_agent_model` constructs an `AsyncOpenAI` client + `OpenAIChatCompletionsModel` for vllm/ollama (Chat Completions, not Responses API). Streams reasoning + tool calls live; falls back to `asyncio.run` in a worker thread when called from inside an existing loop.

### Logging (`pageindex/utils.py:JsonLogger`)
Per-document JSON-line logger writing to `./logs/<sanitized_doc_name>.log` (`logs/` is gitignored). `run_pageindex_verbose.py` monkey-patches it to also emit single-line stderr previews, then tees both `sys.stdout` and `sys.stderr` into a separate timestamped tee log (`./logs/run_<YYYYMMDD_HHMMSS>.log`) so all upstream prints + tracebacks are captured. The tee adds `[YYYY-MM-DD HH:MM:SS.mmm]` per bare line in the file copy; lines that already begin with a timestamp pattern pass through unchanged. `retrieve_pageindex.py` does the same stdout+stderr tee into `./logs/retrieve_<YYYYMMDD_HHMMSS>.log`, with a single `StreamHandler` pointed at the (already-tee'd) `sys.stderr` — there's no separate `FileHandler`, so logger lines reach the file via the tee and never double-write.

## Conventions worth knowing

- Page indices are **1-based**, inclusive at both ends.
- `physical_index` is internal during construction; the persisted output uses `start_index`/`end_index`.
- `format_structure(..., order=[...])` enforces field ordering before serialization — keep new fields out of the order list if you don't want them surfaced.
- `if_add_node_summary=yes` forces `add_node_text` even when `if_add_node_text=no`, then strips text after summarization (`remove_structure_text`).
- `examples/documents/` ships sample PDFs and the corresponding pre-built trees under `examples/documents/results/*_structure.json` are checked in — handy fixtures for retrieval demos without re-running the pipeline.
- `examples/tutorials/doc-search/` and `examples/tutorials/tree-search/` hold walkthrough notes (Markdown only) for the two retrieval styles.
- `.gitignore` excludes `.env*`, `.venv/`, `logs/`, `__pycache__`, `.ipynb_checkpoints`, `.DS_Store`. Anything you generate under `./results/` or `./logs/` stays local.
- See `AGENTS.md` for repo-wide contributor guidelines (style, commit format, security tips). It's the human-facing companion to this file; keep the two consistent.
