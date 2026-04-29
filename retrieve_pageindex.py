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

Each JSON in --folder must be a tree produced by run_pageindex.py / page_index_main,
i.e. of shape {"doc_name": ..., "doc_description"?: ..., "structure": [...]}.

The retriever runs fully locally:
- No PAGEINDEX_API_KEY (cloud service) required.

The agent is given four tools:
- list_documents()
- get_document(doc_id)
- get_document_structure(doc_id)         # text fields stripped
- get_node_content(doc_id, node_id)      # returns node text, falls back to summary

Generate trees with `--if-add-node-text yes --if-add-node-id yes` for best retrieval.
Without node text, the agent must answer from summaries alone.
"""
import argparse
import asyncio
import concurrent.futures
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from pageindex.utils import ConfigLoader, remove_fields

load_dotenv(override=True)


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


def _find_node_by_id(structure, node_id: str):
    for node in _iter_nodes(structure):
        if node.get('node_id') == node_id:
            return node
    return None


# ── Document loading ──────────────────────────────────────────────────────────

def load_documents(folder: Path) -> dict:
    documents = {}
    for path in sorted(folder.glob("*.json")):
        if path.name == "_meta.json":
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {path.name}: {e}", file=sys.stderr)
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
            print(f"Warning: skipping {path.name}: unrecognized JSON shape", file=sys.stderr)
            continue

        if not structure:
            print(f"Warning: skipping {path.name}: no structure found", file=sys.stderr)
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

    return documents


# ── Local tool implementations ────────────────────────────────────────────────

def tool_list_documents(documents: dict) -> str:
    out = []
    for doc_id, doc in documents.items():
        entry = {
            'doc_id': doc_id,
            'doc_name': doc.get('doc_name', ''),
            'doc_description': doc.get('doc_description', ''),
            'type': doc.get('type', ''),
        }
        if doc.get('type') == 'pdf':
            entry['page_count'] = doc.get('page_count', 0)
        else:
            entry['line_count'] = doc.get('line_count', 0)
        out.append(entry)
    return json.dumps(out, ensure_ascii=False)


def tool_get_document(documents: dict, doc_id: str) -> str:
    doc = documents.get(doc_id)
    if not doc:
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


def tool_get_document_structure(documents: dict, doc_id: str) -> str:
    doc = documents.get(doc_id)
    if not doc:
        return json.dumps({'error': f'Document {doc_id} not found'})
    structure_no_text = remove_fields(doc.get('structure', []), fields=['text'])
    return json.dumps(structure_no_text, ensure_ascii=False)


def tool_get_node_content(documents: dict, doc_id: str, node_id: str) -> str:
    doc = documents.get(doc_id)
    if not doc:
        return json.dumps({'error': f'Document {doc_id} not found'})
    node = _find_node_by_id(doc.get('structure', []), node_id)
    if not node:
        return json.dumps({'error': f'Node {node_id} not found in {doc_id}'})
    payload = {
        'doc_id': doc_id,
        'node_id': node_id,
        'title': node.get('title', ''),
        'start_index': node.get('start_index'),
        'end_index': node.get('end_index'),
        'line_num': node.get('line_num'),
    }
    text = node.get('text')
    summary = node.get('summary')
    if text:
        payload['text'] = text
        payload['source'] = 'text'
    elif summary:
        payload['text'] = summary
        payload['source'] = 'summary'
        payload['note'] = 'Node text was not generated; returning summary instead.'
    else:
        payload['text'] = ''
        payload['source'] = 'none'
        payload['note'] = 'Neither text nor summary available for this node.'
    return json.dumps(payload, ensure_ascii=False)


# ── Agent ─────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """
You are PageIndex, a local document QA assistant.
TOOL USE:
- Call list_documents() first to discover available docs.
- Call get_document(doc_id) to confirm metadata.
- Call get_document_structure(doc_id) to inspect the tree and pick the most relevant node_id(s) (use titles + summaries).
- Call get_node_content(doc_id, node_id) to read a node's full text. Prefer leaf nodes; expand to parents only if needed.
- Before each tool call, output one short sentence explaining the reason.
Answer based only on tool output. Be concise and cite the doc_id and node_id you used.
"""


DEFAULT_BASE_URLS = {
    "vllm": "http://localhost:8000/v1",
    "ollama": "http://localhost:11434/v1",
}

PROVIDER_PREFIXES = ("ollama/", "vllm/", "openai/", "litellm/")


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
        from agents import Agent, Runner, function_tool, set_tracing_disabled
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
    print(f"Provider: {provider}  Model: {bare_model}"
          + (f"  Base URL: {base_url or DEFAULT_BASE_URLS.get(provider, 'default')}"
             if provider in ("vllm", "ollama") or base_url else ""))

    @function_tool
    def list_documents() -> str:
        """List all indexed documents available locally."""
        return tool_list_documents(documents)

    @function_tool
    def get_document(doc_id: str) -> str:
        """Get document metadata: status, page/line count, name, description."""
        return tool_get_document(documents, doc_id)

    @function_tool
    def get_document_structure(doc_id: str) -> str:
        """Get the full tree structure (without raw text) to find relevant nodes."""
        return tool_get_document_structure(documents, doc_id)

    @function_tool
    def get_node_content(doc_id: str, node_id: str) -> str:
        """Get the full text of a specific node by its node_id."""
        return tool_get_node_content(documents, doc_id, node_id)

    agent = Agent(
        name="PageIndexLocal",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[list_documents, get_document, get_document_structure, get_node_content],
        model=agent_model,
    )

    async def _run():
        streamed_run = Runner.run_streamed(agent, question, max_turns=100)
        current_kind = None
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
                    current_kind = None
                elif item.type == "tool_call_output_item" and verbose:
                    if current_kind is not None:
                        print()
                    output = str(item.output)
                    preview = output[:200] + "..." if len(output) > 200 else output
                    print(f"\n[tool call output]: {preview}", flush=True)
                    current_kind = None
        if current_kind is not None:
            print()
        return "" if not streamed_run.final_output else str(streamed_run.final_output)

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
    args = parser.parse_args()

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

    answer = query_agent(
        documents,
        model,
        args.question,
        verbose=args.verbose,
        provider=args.provider,
        base_url=base_url,
        api_key=args.api_key,
    )
    print("\n" + "=" * 60)
    print("Final answer:")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
