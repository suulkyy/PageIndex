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
    PAGEINDEX_NODE_TEXT_MAX_CHARS      Cap text returned by get_node_content().

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


def _find_node_by_id(structure, node_id: str):
    for node in _iter_nodes(structure):
        if node.get('node_id') == node_id:
            return node
    return None


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


def _compact_structure_for_tool(structure, summary_max_chars: int):
    """Return a navigation-sized tree. Full summaries across large books can
    exceed local vLLM context before the agent gets to request node text."""
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
        children = _compact_structure_for_tool(node.get('nodes') or [], summary_max_chars)
        if children:
            item['nodes'] = children
        compact.append(item)
    return compact


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

def tool_list_documents(documents: dict) -> str:
    logger.info("tool: list_documents()")
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


def tool_get_document_structure(documents: dict, doc_id: str) -> str:
    logger.info("tool: get_document_structure(doc_id=%s)", doc_id)
    doc = documents.get(doc_id)
    if not doc:
        logger.warning("  doc_id %s not found", doc_id)
        return json.dumps({'error': f'Document {doc_id} not found'})
    summary_cap = _env_int("PAGEINDEX_STRUCTURE_SUMMARY_MAX_CHARS", 160)
    structure = _compact_structure_for_tool(doc.get('structure', []), summary_cap)
    payload = {
        'doc_id': doc_id,
        'note': (
            "Compact structure: node text omitted. Use get_node_content(doc_id, node_id) "
            "for the selected node."
        ),
        'summary_max_chars': summary_cap,
        'structure': structure,
    }
    return json.dumps(payload, ensure_ascii=False)


def tool_get_node_content(documents: dict, doc_id: str, node_id: str) -> str:
    logger.info("tool: get_node_content(doc_id=%s, node_id=%s)", doc_id, node_id)
    doc = documents.get(doc_id)
    if not doc:
        logger.warning("  doc_id %s not found", doc_id)
        return json.dumps({'error': f'Document {doc_id} not found'})
    node = _find_node_by_id(doc.get('structure', []), node_id)
    if not node:
        logger.warning("  node_id %s not found in %s", node_id, doc_id)
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

    # Cap returned text so the agent's growing message history doesn't push
    # vLLM/Ollama past the KV cache budget on long sessions or huge nodes.
    # Override via PAGEINDEX_NODE_TEXT_MAX_CHARS=0 to disable, or any int
    # to set a custom cap.
    cap = _env_int("PAGEINDEX_NODE_TEXT_MAX_CHARS", 8000)
    if cap > 0 and isinstance(payload.get('text'), str) and len(payload['text']) > cap:
        original = len(payload['text'])
        payload['text'] = payload['text'][:cap]
        payload['truncated'] = True
        payload['original_text_chars'] = original
        payload['truncated_to_chars'] = cap
        payload.setdefault('note',
            f"Text truncated to {cap} chars (original {original}). Adjust PAGEINDEX_NODE_TEXT_MAX_CHARS to change.")
    return json.dumps(payload, ensure_ascii=False)


# ── Agent ─────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """
You are PageIndex, a local document QA assistant.
TOOL USE:
- Call list_documents() first to discover available docs.
- Call get_document(doc_id) to confirm metadata.
- Call get_document_structure(doc_id) to inspect the compact tree and pick the most relevant node_id(s).
- Call get_node_content(doc_id, node_id) to read a node's full text. Prefer leaf nodes; expand to parents only if needed.
- Before each tool call, output one short sentence explaining the reason.
Answer based only on tool output. Be concise and cite the doc_id and node_id you used.
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
                        "Be concise. Preserve document/node citations if present."
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
    resolved_base = base_url or DEFAULT_BASE_URLS.get(provider, "default")
    logger.info("provider=%s model=%s base_url=%s", provider, bare_model, resolved_base)
    logger.info("question: %s", question)

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

    answer_thinking = _env_truthy("PAGEINDEX_RETRIEVE_ENABLE_THINKING", False)
    if provider == "vllm" and "PAGEINDEX_AGENT_MAX_TOKENS" not in os.environ:
        default_max_tokens = 8192 if answer_thinking else 1024
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
        tools=[list_documents, get_document, get_document_structure, get_node_content],
        model=agent_model,
    )
    if _max_out > 0 or extra_body:
        try:
            from agents import ModelSettings
            settings = {}
            if _max_out > 0:
                settings["max_tokens"] = _max_out
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
        streamed_run = Runner.run_streamed(agent, question, max_turns=_max_turns)
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
