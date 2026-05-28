# Repository Guidelines

## Project Structure & Module Organization

`pageindex/` contains the library code: PDF indexing in `page_index.py`, Markdown indexing in `page_index_md.py`, retrieval helpers in `retrieve.py`, the persistent SQLite BM25+vector retrieval index in `index_store.py`, embedding/rerank HTTP clients in `semantic.py`, shared utilities in `utils.py`, and defaults in `config.yaml`. Top-level entry points are `run_pageindex.py`, `run_pageindex_verbose.py`, `retrieve_pageindex.py`, and `build_index.py` (builds the SQLite index; `--embeddings` adds vectors). `examples/` holds demos, sample PDFs, tutorials, and reference JSON under `examples/documents/results/`. `cookbook/` contains notebook walkthroughs. No `tests/` directory yet.

## Build, Test, and Development Commands

Use Python 3.12+.

```bash
uv sync
uv run python run_pageindex_verbose.py --pdf_path examples/documents/2023-annual-report-truncated.pdf
uv run python run_pageindex.py --md_path path/to/document.md
uv run python build_index.py --folder ./results
PAGEINDEX_AGENT_MAX_TOKENS=2048 PAGEINDEX_AGENT_MAX_TURNS=10 uv run python retrieve_pageindex.py --folder ./results --question "What is this document about?"
uv run python retrieve_pageindex.py --folder ./results --mode fast --provider vllm --model rag-llm --question "What is this document about?"
python3 -m py_compile pageindex/*.py run_pageindex.py run_pageindex_verbose.py retrieve_pageindex.py build_index.py
pip3 install --upgrade -r requirements.txt
```

`uv sync` installs dependencies from `pyproject.toml` and `uv.lock`. Tree generation writes `./results/<name>_structure.json`. Use `--if-add-node-text yes --if-add-node-id yes` for retrieval-ready JSON. The compile command is the lightweight validation step; pip is the fallback installer.

## Local vLLM & Retrieval Tips

For hosted vLLM, use model strings like `hosted_vllm/Qwen/Qwen3.5-9B` and pass `--base-url http://host:8000/v1`. Match `--llm-concurrency` to server `--max-num-seqs`; for 32k-context 2x RTX 5000 runs, `2` is safer. Retrieval uses compact structure output and an `answer` compatibility tool. Useful knobs: `PAGEINDEX_STRUCTURE_SUMMARY_MAX_CHARS=120`, `PAGEINDEX_NODE_TEXT_MAX_CHARS=10000`, `PAGEINDEX_AGENT_MAX_TOKENS=2048`, `PAGEINDEX_RETRIEVE_ENABLE_THINKING=false`.

`retrieve_pageindex.py --mode` selects the strategy (default `hybrid`): `fast` (one-shot BM25 over the SQLite index + a single synthesis call — fastest, cross-document, skips loading every tree into RAM), `agent` (iterative loop + a `search_all_documents` cross-doc tool), `hybrid` (BM25 candidates seed the agent). All modes auto-refresh the index at `<folder>/.pageindex_index.db` (gitignored); `build_index.py` builds it offline. Fast/hybrid knobs: `PAGEINDEX_FAST_TOP_K=8`, `PAGEINDEX_FAST_POOL=50`, `PAGEINDEX_FAST_MAX_PER_DOC` (set `2`–`3` to diversify cross-doc results on multi-topic questions), `PAGEINDEX_FAST_EVIDENCE_MAX_CHARS=60000`, `PAGEINDEX_HYBRID_SEED_K=12`. Phase-2 semantic layer (reuses LightRAG `:8001`/`:8002`): build vectors with `build_index.py --embeddings`, then `PAGEINDEX_FAST_RETRIEVER=auto` fuses BM25+vector via RRF and `PAGEINDEX_RERANK=true` adds a cross-encoder pass; both degrade gracefully to BM25 if the servers are down. No vector DB / ANN — at this corpus size brute-force cosine is sub-10ms and the bottleneck is LLM round-trips.

## Coding Style & Naming Conventions

Write idiomatic Python with 4-space indentation, `snake_case` functions and variables, and `PascalCase` classes. Preserve existing CLI flag spelling, including mixed styles like `--pdf_path`, `--base-url`, and `--toc-check-pages`. Prefer explicit imports; avoid expanding wildcard imports. No formatter or linter is configured, so match nearby code and avoid style-only rewrites.

## Testing Guidelines

Automated tests are not configured yet. For new behavior, add focused `pytest` tests under `tests/` using `test_*.py` names, and document fixtures or API keys. Until then, run `py_compile` plus a small-PDF smoke test and verify JSON shape. LLM-backed commands require `.env` credentials such as `OPENAI_API_KEY` or provider-specific LiteLLM settings.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `Move environment variables to arguments`. Follow that style: describe the user-visible change first and keep the subject concise. Pull requests should include a clear description, commands run, sample input/output for retrieval or indexing changes, linked issues if applicable, and screenshots only for notebook or documentation rendering changes.

## Security & Configuration Tips

Do not commit `.env`, API keys, generated logs, or large new document assets unless they are intentional examples. Prefer CLI flags or environment variables over hard-coded provider endpoints, model names, or credentials.
