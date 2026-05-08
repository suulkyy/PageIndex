import litellm
import logging
import os
import textwrap
from datetime import datetime
import time
import json
import PyPDF2
import copy
import asyncio
import pymupdf
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import logging
import yaml
from pathlib import Path
from types import SimpleNamespace as config

# Backward compatibility: support CHATGPT_API_KEY as alias for OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")

litellm.drop_params = True


# ── Concurrency cap for LLM calls ─────────────────────────────────────────────
# Tree-build fans out LLM calls via asyncio.gather (verify_toc, summary
# generation, title-appearance checks, recursive node processing). Without
# a cap, a long doc can fire 50+ concurrent requests at the LLM. On a
# single-GPU vLLM that overflows the KV cache: vLLM preempts running
# sequences, throughput collapses to 0 t/s, and requests cycle Running →
# Waiting → Running. Bound concurrency to a safe default and let the user
# raise it via env when they have headroom.
_LLM_SEM = None
_LLM_SEM_LOOP = None
# Per-process registry seeded by `configure_llm_runtime(**kwargs)` from CLI
# scripts. Read by `_provider_kwargs`, `_resolve_max_tokens`, `_get_llm_sem`,
# and `page_list_to_group_text` via `get_llm_runtime_value`. CLI args win;
# env vars are fallbacks.
_LLM_RUNTIME = {}


def configure_llm_runtime(**kwargs):
    """Set per-process LLM runtime options from CLI/API callers.

    Environment variables remain as backward-compatible fallbacks, but scripts
    should prefer passing explicit CLI arguments and calling this helper.
    """
    global _LLM_SEM, _LLM_SEM_LOOP
    changed_concurrency = False
    for key, value in kwargs.items():
        if value is None:
            continue
        _LLM_RUNTIME[key] = value
        if key == "llm_concurrency":
            changed_concurrency = True
    if changed_concurrency:
        _LLM_SEM = None
        _LLM_SEM_LOOP = None


def get_llm_runtime_value(name, env_names=(), default=None):
    if name in _LLM_RUNTIME:
        return _LLM_RUNTIME[name]
    for env_name in env_names:
        value = os.getenv(env_name)
        if value is not None:
            return value
    return default


def get_llm_runtime_int(name, env_names=(), default=0):
    value = get_llm_runtime_value(name, env_names, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_llm_runtime_float(name, env_names=(), default=0.0):
    value = get_llm_runtime_value(name, env_names, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_llm_sem():
    """Lazy per-loop asyncio.Semaphore. Re-create when running under a
    different event loop (tests, nested asyncio.run)."""
    global _LLM_SEM, _LLM_SEM_LOOP
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _LLM_SEM is None or _LLM_SEM_LOOP is not loop:
        n = get_llm_runtime_int("llm_concurrency", ("PAGEINDEX_LLM_CONCURRENCY",), 8)
        _LLM_SEM = asyncio.Semaphore(max(1, n))
        _LLM_SEM_LOOP = loop
    return _LLM_SEM


def count_tokens(text, model=None):
    if not text:
        return 0
    return litellm.token_counter(model=model, text=text)


def _provider_kwargs(model):
    kwargs = {}
    if not model:
        return kwargs
    if model.startswith(("ollama/", "ollama_chat/")):
        base = get_llm_runtime_value(
            "ollama_base_url",
            ("OLLAMA_API_BASE", "OLLAMA_BASE_URL", "OLLAMA_HOST"),
        )
        if base:
            kwargs["api_base"] = base
        timeout = get_llm_runtime_float("ollama_timeout", ("OLLAMA_TIMEOUT",), 1800.0)
        kwargs["timeout"] = timeout
        # Disable reasoning/thinking output for qwen3, deepseek-r1, etc.
        # Override with OLLAMA_THINK=true if you want it back on.
        if os.getenv("OLLAMA_THINK", "false").lower() not in ("1", "true", "yes"):
            kwargs["think"] = False
    elif model.startswith(("vllm/", "hosted_vllm/")):
        base = get_llm_runtime_value(
            "vllm_base_url",
            ("VLLM_API_BASE", "VLLM_BASE_URL"),
        )
        if base:
            kwargs["api_base"] = base
        # LiteLLM default for hosted_vllm is 600s, which times out on slow
        # remote vLLM hosts running big-context models. Mirror the ollama
        # pattern: 1800s default, override with VLLM_TIMEOUT.
        timeout = get_llm_runtime_float("vllm_timeout", ("VLLM_TIMEOUT",), 1800.0)
        kwargs["timeout"] = timeout
        # max_tokens injected dynamically per-call via _resolve_max_tokens()
        # below — needs prompt size to clamp under server's max-model-len.
        # Disable Qwen3-family chat-template thinking by default. Qwen3
        # tokenizer's chat template defaults to enable_thinking=True, so
        # the model emits <think>...</think>JSON. If max_tokens truncates
        # mid-think, the open-ended <think>.* fallback in extract_json
        # wipes everything → empty content → "Expecting value: line 1
        # column 1". Forcing enable_thinking=False keeps tokens in JSON.
        # Set VLLM_ENABLE_THINKING=true to opt back in.
        if os.getenv("VLLM_ENABLE_THINKING", "false").lower() not in ("1", "true", "yes"):
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"].setdefault(
                "chat_template_kwargs", {"enable_thinking": False}
            )
    return kwargs


def _resolve_max_tokens(model, messages):
    """Dynamically clamp max_tokens so prompt + output fits the server's
    max-model-len. Only applies to vllm/hosted_vllm. Returns dict that
    callers merge into completion kwargs."""
    if not model or not model.startswith(("vllm/", "hosted_vllm/")):
        return {}
    user_cap_v = get_llm_runtime_int("vllm_max_tokens", ("VLLM_MAX_TOKENS",), 2048)
    if user_cap_v <= 0:
        return {}  # opted out — leave server default
    model_len = get_llm_runtime_int("vllm_max_model_len", ("VLLM_MAX_MODEL_LEN",), 16384)
    margin = get_llm_runtime_int("vllm_ctx_margin", ("VLLM_CTX_MARGIN",), 256)
    try:
        prompt_text = "\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        prompt_tokens = count_tokens(prompt_text, model=model) or 0
    except Exception:
        prompt_tokens = 0
    headroom = model_len - prompt_tokens - margin
    if headroom < 256:
        # Prompt nearly fills (or overflows) the context. Old behavior was
        # to clamp max_tokens to a 64-token floor and let vLLM raise
        # `VLLMValidationError: prompt+output > max-model-len` (often by
        # a single token, surfacing only as opaque server-side error
        # after a real network round-trip). Raise client-side instead so
        # the caller sees an actionable error before paying for the
        # wasted call.
        raise ValueError(
            f"vLLM prompt budget exceeded: prompt={prompt_tokens} tok, "
            f"margin={margin}, max_model_len={model_len}, "
            f"headroom={headroom} (< 256). Reduce input via "
            f"--toc-check-pages, --group-max-tokens, "
            f"--toc-chunk-max-tokens, or raise --vllm-max-model-len."
        )
    return {"max_tokens": min(user_cap_v, headroom)}


def _extract_message_text(message):
    """Return text content. Falls back to reasoning_content when the
    server's reasoning-parser routed all tokens there (vLLM
    --reasoning-parser, deepseek-r1, qwen3-thinking, etc.). Empty
    `content` with non-empty `reasoning_content` would otherwise look
    like an empty completion and trip JSON parsers."""
    content = getattr(message, "content", None) or ""
    if content.strip():
        return content
    reasoning = getattr(message, "reasoning_content", None) or ""
    if reasoning.strip():
        return reasoning
    # LiteLLM also exposes reasoning under provider_specific_fields/_hidden_params
    extras = getattr(message, "provider_specific_fields", None) or {}
    if isinstance(extras, dict):
        for key in ("reasoning_content", "reasoning"):
            v = extras.get(key)
            if isinstance(v, str) and v.strip():
                return v
    return content


def llm_completion(model, prompt, chat_history=None, return_finish_reason=False):
    if model:
        model = model.removeprefix("litellm/")
    extra = _provider_kwargs(model)
    max_retries = 10
    messages = list(chat_history) + [{"role": "user", "content": prompt}] if chat_history else [{"role": "user", "content": prompt}]
    extra.update(_resolve_max_tokens(model, messages))
    for i in range(max_retries):
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=0,
                **extra,
            )
            content = _extract_message_text(response.choices[0].message)
            if return_finish_reason:
                finish_reason = "max_output_reached" if response.choices[0].finish_reason == "length" else "finished"
                return content, finish_reason
            return content
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                if return_finish_reason:
                    return "", "error"
                return ""



async def llm_acompletion(model, prompt):
    if model:
        model = model.removeprefix("litellm/")
    extra = _provider_kwargs(model)
    max_retries = 10
    messages = [{"role": "user", "content": prompt}]
    extra.update(_resolve_max_tokens(model, messages))
    sem = _get_llm_sem()
    async with sem:
        for i in range(max_retries):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0,
                    **extra,
                )
                return _extract_message_text(response.choices[0].message)
            except Exception as e:
                print('************* Retrying *************')
                logging.error(f"Error: {e}")
                if i < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    logging.error('Max retries reached for prompt: ' + prompt)
                    return ""
            
            
def get_json_content(response):
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]
        
    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]
    
    json_content = response.strip()
    return json_content
         

def extract_json(content):
    try:
        if not content:
            return {}
        import re as _re
        # Strip reasoning-model think blocks (qwen3, deepseek-r1, etc.)
        content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL)
        content = _re.sub(r"<think>.*", "", content, flags=_re.DOTALL)
        content = content.strip()

        # First, try to extract JSON enclosed within ```json and ```
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7  # Adjust index to start after the delimiter
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            # Prefer first '[' if present (list responses), else first '{'.
            first_bracket = content.find("[")
            first_brace = content.find("{")
            if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
                start = first_bracket
            elif first_brace != -1:
                start = first_brace
            else:
                start = 0
            json_content = content[start:].strip()

        # Use raw_decode to handle "Extra data" — return only first valid JSON value.
        try:
            decoder = json.JSONDecoder()
            value, _ = decoder.raw_decode(json_content)
            return value
        except json.JSONDecodeError:
            pass

        # Clean up common issues that might cause parsing errors
        json_content = json_content.replace('None', 'null')  # Replace Python None with JSON null
        json_content = json_content.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
        json_content = ' '.join(json_content.split())  # Normalize whitespace

        # Attempt to parse and return the JSON object
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to extract JSON: {e}")
        # Try to clean up the content further if initial parsing fails
        try:
            # Remove any trailing commas before closing brackets/braces
            json_content = json_content.replace(',]', ']').replace(',}', '}')
            return json.loads(json_content)
        except Exception:
            # Last resort: json_repair handles common LLM mistakes —
            # missing commas between objects, unescaped quotes, single
            # quotes, unquoted keys, trailing commas. Catches Qwen3.5-9B
            # outputs that emit `}{` or `} {` between array elements.
            try:
                from json_repair import repair_json
                repaired = repair_json(json_content, return_objects=True)
                if repaired not in ({}, [], "", None):
                    return repaired
            except Exception:
                pass
            logging.error("Failed to parse JSON even after cleanup")
            return {}
    except Exception as e:
        logging.error(f"Unexpected error while extracting JSON: {e}")
        return {}

def write_node_id(data, node_id=0):
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if 'nodes' in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id

def get_nodes(structure):
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop('nodes', None)
        nodes = [structure_node]
        for key in list(structure.keys()):
            if 'nodes' in key:
                nodes.extend(get_nodes(structure[key]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(get_nodes(item))
        return nodes
    
def structure_to_list(structure):
    if isinstance(structure, dict):
        nodes = []
        nodes.append(structure)
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes

    
def get_leaf_nodes(structure):
    if isinstance(structure, dict):
        if not structure['nodes']:
            structure_node = copy.deepcopy(structure)
            structure_node.pop('nodes', None)
            return [structure_node]
        else:
            leaf_nodes = []
            for key in list(structure.keys()):
                if 'nodes' in key:
                    leaf_nodes.extend(get_leaf_nodes(structure[key]))
            return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes

def is_leaf_node(data, node_id):
    # Helper function to find the node by its node_id
    def find_node(data, node_id):
        if isinstance(data, dict):
            if data.get('node_id') == node_id:
                return data
            for key in data.keys():
                if 'nodes' in key:
                    result = find_node(data[key], node_id)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_node(item, node_id)
                if result:
                    return result
        return None

    # Find the node with the given node_id
    node = find_node(data, node_id)

    # Check if the node is a leaf node
    if node and not node.get('nodes'):
        return True
    return False

def get_last_node(structure):
    return structure[-1]


def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    ###return text not list 
    text=""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text+=page.extract_text()
    return text

def get_pdf_title(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else 'Untitled'
    return title

def get_text_of_pages(pdf_path, start_page, end_page, tag=True):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(start_page-1, end_page):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if tag:
            text += f"<start_index_{page_num+1}>\n{page_text}\n<end_index_{page_num+1}>\n"
        else:
            text += page_text
    return text

def get_first_start_page_from_text(text):
    start_page = -1
    start_page_match = re.search(r'<start_index_(\d+)>', text)
    if start_page_match:
        start_page = int(start_page_match.group(1))
    return start_page

def get_last_start_page_from_text(text):
    start_page = -1
    # Find all matches of start_index tags
    start_page_matches = re.finditer(r'<start_index_(\d+)>', text)
    # Convert iterator to list and get the last match if any exist
    matches_list = list(start_page_matches)
    if matches_list:
        start_page = int(matches_list[-1].group(1))
    return start_page


def sanitize_filename(filename, replacement='-'):
    # In Linux, only '/' and '\0' (null) are invalid in filenames.
    # Null can't be represented in strings, so we only handle '/'.
    return filename.replace('/', replacement)

def get_pdf_name(pdf_path):
    # Extract PDF name
    if isinstance(pdf_path, str):
        pdf_name = os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else 'Untitled'
        pdf_name = sanitize_filename(pdf_name)
    return pdf_name


class JsonLogger:
    def __init__(self, file_path):
        # Extract PDF name for logger name
        pdf_name = get_pdf_name(file_path)
            
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        # Initialize empty list to store all messages
        self.log_data = []

    def log(self, level, message, **kwargs):
        if isinstance(message, dict):
            self.log_data.append(message)
        else:
            self.log_data.append({'message': message})
        # Add new message to the log data
        
        # Write entire log data to file
        with open(self._filepath(), "w") as f:
            json.dump(self.log_data, f, indent=2)

    def info(self, message, **kwargs):
        self.log("INFO", message, **kwargs)

    def error(self, message, **kwargs):
        self.log("ERROR", message, **kwargs)

    def debug(self, message, **kwargs):
        self.log("DEBUG", message, **kwargs)

    def exception(self, message, **kwargs):
        kwargs["exception"] = True
        self.log("ERROR", message, **kwargs)

    def _filepath(self):
        return os.path.join("logs", self.filename)
    



def list_to_tree(data):
    def get_parent_structure(structure):
        """Helper function to get the parent structure code"""
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None
    
    # First pass: Create nodes and track parent-child relationships
    nodes = {}
    root_nodes = []
    
    for item in data:
        structure = item.get('structure')
        node = {
            'title': item.get('title'),
            'start_index': item.get('start_index'),
            'end_index': item.get('end_index'),
            'nodes': []
        }
        
        nodes[structure] = node
        
        # Find parent
        parent_structure = get_parent_structure(structure)
        
        if parent_structure:
            # Add as child to parent if parent exists
            if parent_structure in nodes:
                nodes[parent_structure]['nodes'].append(node)
            else:
                root_nodes.append(node)
        else:
            # No parent, this is a root node
            root_nodes.append(node)
    
    # Helper function to clean empty children arrays
    def clean_node(node):
        if not node['nodes']:
            del node['nodes']
        else:
            for child in node['nodes']:
                clean_node(child)
        return node
    
    # Clean and return the tree
    return [clean_node(node) for node in root_nodes]

def add_preface_if_needed(data):
    if not isinstance(data, list) or not data:
        return data

    if data[0]['physical_index'] is not None and data[0]['physical_index'] > 1:
        preface_node = {
            "structure": "0",
            "title": "Preface",
            "physical_index": 1,
        }
        data.insert(0, preface_node)
    return data



def get_page_tokens(pdf_path, model=None, pdf_parser="PyPDF2"):
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = litellm.token_counter(model=model, text=page_text)
            page_list.append((page_text, token_length))
        return page_list
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            pdf_stream = pdf_path
            doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            doc = pymupdf.open(pdf_path)
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = litellm.token_counter(model=model, text=page_text)
            page_list.append((page_text, token_length))
        return page_list
    else:
        raise ValueError(f"Unsupported PDF parser: {pdf_parser}")

        

def get_text_of_pdf_pages(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += pdf_pages[page_num][0]
    return text

def get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += f"<physical_index_{page_num+1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num+1}>\n"
    return text

def get_number_of_pages(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    num = len(pdf_reader.pages)
    return num



def post_processing(structure, end_physical_index):
    # First convert page_number to start_index in flat list
    for i, item in enumerate(structure):
        item['start_index'] = item.get('physical_index')
        if i < len(structure) - 1:
            if structure[i + 1].get('appear_start') == 'yes':
                item['end_index'] = structure[i + 1]['physical_index']-1
            else:
                item['end_index'] = structure[i + 1]['physical_index']
        else:
            item['end_index'] = end_physical_index
        # Clamp inverted spans (end < start). Happens when two consecutive
        # items share the same physical_index and the next has
        # appear_start='yes' — the `next.physical_index - 1` rule would put
        # end one page before start. Treat as a single-page node instead of
        # surfacing pp15-14 / pp20-19 / pp31-30 in the final tree.
        if (
            isinstance(item.get('start_index'), int)
            and isinstance(item.get('end_index'), int)
            and item['end_index'] < item['start_index']
        ):
            item['end_index'] = item['start_index']
    tree = list_to_tree(structure)
    if len(tree)!=0:
        return tree
    else:
        ### remove appear_start 
        for node in structure:
            node.pop('appear_start', None)
            node.pop('physical_index', None)
        return structure

def clean_structure_post(data):
    if isinstance(data, dict):
        data.pop('page_number', None)
        data.pop('start_index', None)
        data.pop('end_index', None)
        if 'nodes' in data:
            clean_structure_post(data['nodes'])
    elif isinstance(data, list):
        for section in data:
            clean_structure_post(section)
    return data

def remove_fields(data, fields=['text']):
    if isinstance(data, dict):
        return {k: remove_fields(v, fields)
            for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data

def print_toc(tree, indent=0):
    for node in tree:
        print('  ' * indent + node['title'])
        if node.get('nodes'):
            print_toc(node['nodes'], indent + 1)

def print_json(data, max_len=40, indent=2):
    def simplify_data(obj):
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + '...'
        else:
            return obj
    
    simplified = simplify_data(data)
    print(json.dumps(simplified, indent=indent, ensure_ascii=False))


def remove_structure_text(data):
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def check_token_limit(structure, limit=110000):
    list = structure_to_list(structure)
    for node in list:
        num_tokens = count_tokens(node['text'], model=None)
        if num_tokens > limit:
            print(f"Node ID: {node['node_id']} has {num_tokens} tokens")
            print("Start Index:", node['start_index'])
            print("End Index:", node['end_index'])
            print("Title:", node['title'])
            print("\n")


def convert_physical_index_to_int(data):
    if isinstance(data, list):
        for i in range(len(data)):
            # Check if item is a dictionary and has 'physical_index' key
            if isinstance(data[i], dict) and 'physical_index' in data[i]:
                if isinstance(data[i]['physical_index'], str):
                    if data[i]['physical_index'].startswith('<physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].rstrip('>').strip())
                    elif data[i]['physical_index'].startswith('physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].strip())
    elif isinstance(data, str):
        if data.startswith('<physical_index_'):
            data = int(data.split('_')[-1].rstrip('>').strip())
        elif data.startswith('physical_index_'):
            data = int(data.split('_')[-1].strip())
        # Check data is int
        if isinstance(data, int):
            return data
        else:
            return None
    return data


def convert_page_to_int(data):
    for item in data:
        if 'page' in item and isinstance(item['page'], str):
            try:
                item['page'] = int(item['page'])
            except ValueError:
                # Keep original value if conversion fails
                pass
    return data


def add_node_text(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text(node[index], pdf_pages)
    return


def add_node_text_with_labels(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text_with_labels(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text_with_labels(node[index], pdf_pages)
    return


async def generate_node_summary(node, model=None):
    prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

    Partial Document Text: {node['text']}
    
    Directly return the description, do not include any other text.
    """
    response = await llm_acompletion(model, prompt)
    return response


async def generate_summaries_for_structure(structure, model=None):
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model=model) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    
    for node, summary in zip(nodes, summaries):
        node['summary'] = summary
    return structure


def create_clean_structure_for_description(structure, max_depth=None, summary_max_chars=None, _depth=0):
    """
    Create a clean structure for document description generation,
    excluding unnecessary fields like 'text'.

    max_depth: drop child `nodes` past this depth (root is depth 0).
    summary_max_chars: truncate `summary`/`prefix_summary` to this many chars.
    """
    if isinstance(structure, dict):
        clean_node = {}
        for key in ['title', 'node_id', 'summary', 'prefix_summary']:
            if key in structure:
                value = structure[key]
                if (
                    summary_max_chars
                    and key in ('summary', 'prefix_summary')
                    and isinstance(value, str)
                    and len(value) > summary_max_chars
                ):
                    value = value[:summary_max_chars].rstrip() + '…'
                clean_node[key] = value

        if 'nodes' in structure and structure['nodes']:
            if max_depth is None or _depth < max_depth:
                clean_node['nodes'] = create_clean_structure_for_description(
                    structure['nodes'], max_depth, summary_max_chars, _depth + 1
                )

        return clean_node
    elif isinstance(structure, list):
        return [
            create_clean_structure_for_description(item, max_depth, summary_max_chars, _depth)
            for item in structure
        ]
    else:
        return structure


def generate_doc_description(structure, model=None):
    """Build a doc description prompt that fits the server's context window.

    For long books the full tree of summaries can exceed max-model-len. Try
    progressively smaller representations until the prompt fits, falling
    back to a titles-only outline as a last resort.
    """
    def build_prompt(s):
        return (
            "Your are an expert in generating descriptions for a document.\n"
            "You are given a structure of a document. Your task is to generate a one-sentence "
            "description for the document, which makes it easy to distinguish the document from "
            "other documents.\n\n"
            f"Document Structure: {s}\n\n"
            "Directly return the description, do not include any other text."
        )

    is_vllm = bool(model) and model.removeprefix("litellm/").startswith(("vllm/", "hosted_vllm/"))
    if is_vllm:
        model_len = get_llm_runtime_int("vllm_max_model_len", ("VLLM_MAX_MODEL_LEN",), 16384)
        output_cap = get_llm_runtime_int("vllm_max_tokens", ("VLLM_MAX_TOKENS",), 2048) or 2048
        margin = get_llm_runtime_int("vllm_ctx_margin", ("VLLM_CTX_MARGIN",), 256)
        budget = model_len - output_cap - margin - 256  # 256 for chat-template framing
    else:
        budget = None  # OpenAI/Anthropic — context big enough, skip the dance

    candidates = [
        structure,
        create_clean_structure_for_description(structure, summary_max_chars=400),
        create_clean_structure_for_description(structure, max_depth=2, summary_max_chars=300),
        create_clean_structure_for_description(structure, max_depth=1, summary_max_chars=200),
        create_clean_structure_for_description(structure, max_depth=2, summary_max_chars=0),
        create_clean_structure_for_description(structure, max_depth=1, summary_max_chars=0),
    ]

    chosen = candidates[-1]
    if budget is not None:
        for cand in candidates:
            prompt = build_prompt(cand)
            if (count_tokens(prompt, model=model) or 0) <= budget:
                chosen = cand
                break

    return llm_completion(model, build_prompt(chosen))


def reorder_dict(data, key_order):
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure, order=None):
    if not order:
        return structure
    if isinstance(structure, dict):
        if 'nodes' in structure:
            structure['nodes'] = format_structure(structure['nodes'], order)
        if not structure.get('nodes'):
            structure.pop('nodes', None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


class ConfigLoader:
    def __init__(self, default_path: str = None):
        if default_path is None:
            default_path = Path(__file__).parent / "config.yaml"
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict):
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        """
        Load the configuration, merging user options with default values.
        """
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")

        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}
        return config(**merged)

def create_node_mapping(tree):
    """Create a flat dict mapping node_id to node for quick lookup."""
    mapping = {}
    def _traverse(nodes):
        for node in nodes:
            if node.get('node_id'):
                mapping[node['node_id']] = node
            if node.get('nodes'):
                _traverse(node['nodes'])
    _traverse(tree)
    return mapping

def print_tree(tree, indent=0):
    for node in tree:
        summary = node.get('summary') or node.get('prefix_summary', '')
        summary_str = f"  —  {summary[:60]}..." if summary else ""
        print('  ' * indent + f"[{node.get('node_id', '?')}] {node.get('title', '')}{summary_str}")
        if node.get('nodes'):
            print_tree(node['nodes'], indent + 1)

def print_wrapped(text, width=100):
    for line in text.splitlines():
        print(textwrap.fill(line, width=width))
