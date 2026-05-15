import os
import json
import copy
import math
import random
import re
from .utils import *
# Star import skips underscore-prefixed names; pull in the semaphore
# explicitly so Phase 3 chunk parallelism can share the project-wide
# in-flight LLM-call cap.
from .utils import _get_llm_sem
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


################### check title in page #########################################################
async def check_title_appearance(item, page_list, start_index=1, model=None):    
    title=item['title']
    if 'physical_index' not in item or item['physical_index'] is None:
        return {'list_index': item.get('list_index'), 'answer': 'no', 'title':title, 'page_number': None}
    
    
    page_number = item['physical_index']
    page_text = page_list[page_number-start_index][0]

    
    prompt = f"""
    Your job is to check if the given section appears or starts in the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.
    
    Reply format:
    {{
        "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not include explanations or any other text."""

    response = await llm_acompletion(model=model, prompt=prompt)
    response = extract_json(response)
    if 'answer' in response:
        answer = response['answer']
    else:
        answer = 'no'
    return {'list_index': item['list_index'], 'answer': answer, 'title': title, 'page_number': page_number}


async def check_title_appearance_in_start(title, page_text, model=None, logger=None):    
    prompt = f"""
    You will be given the current section title and the current page_text.
    Your job is to check if the current section starts in the beginning of the given page_text.
    If there are other contents before the current section title, then the current section does not start in the beginning of the given page_text.
    If the current section title is the first content in the given page_text, then the current section starts in the beginning of the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.
    
    reply format:
    {{
        "start_begin": "yes or no" (yes if the section starts in the beginning of the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not include explanations or any other text."""

    response = await llm_acompletion(model=model, prompt=prompt)
    response = extract_json(response)
    if logger:
        logger.info(f"Response: {response}")
    return response.get("start_begin", "no")


def _normalize_for_title_match(text):
    text = text or ""
    for src, dst in _TOC_NOISE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def check_title_appearance_in_start_heuristic(title, page_text):
    """Cheap local check: does the (normalized) title appear near the top
    of the page? Returns 'yes'/'no' when confident, or None when the
    heuristic can't decide and the LLM should be consulted."""
    title_norm = _normalize_for_title_match(title)
    if not title_norm:
        return None
    prefix_norm = _normalize_for_title_match((page_text or "")[:1500])
    if not prefix_norm:
        return None
    pos = prefix_norm.find(title_norm)
    if pos == -1:
        return None
    return "yes" if pos <= 160 else "no"


async def check_title_appearance_in_start_concurrent(structure, page_list, model=None, logger=None):
    if logger:
        logger.info("Checking title appearance in start concurrently")
    
    # skip items without physical_index
    for item in structure:
        if item.get('physical_index') is None:
            item['appear_start'] = 'no'

    # only for items with valid physical_index
    tasks = []
    valid_items = []
    heuristic_count = 0
    for item in structure:
        if item.get('physical_index') is not None:
            page_text = page_list[item['physical_index'] - 1][0]
            heuristic_answer = check_title_appearance_in_start_heuristic(item['title'], page_text)
            if heuristic_answer is not None:
                item['appear_start'] = heuristic_answer
                heuristic_count += 1
            else:
                tasks.append(check_title_appearance_in_start(item['title'], page_text, model=model, logger=logger))
                valid_items.append(item)
    if logger:
        logger.info(f"title start heuristic resolved {heuristic_count}; llm fallback {len(tasks)}")

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(valid_items, results):
        if isinstance(result, Exception):
            if logger:
                logger.error(f"Error checking start for {item['title']}: {result}")
            item['appear_start'] = 'no'
        else:
            item['appear_start'] = result

    return structure


def toc_detector_single_page(content, model=None):
    prompt = f"""
    Your job is to detect if there is a table of content provided in the given text.

    Given text: {content}

    return the following JSON format:
    {{
        "toc_detected": "<yes or no>"
    }}

    Directly return the final JSON structure. Do not include explanations or any other text.
    Please note: abstract,summary, notation list, figure list, table list, etc. are not table of contents."""

    response = llm_completion(model=model, prompt=prompt)
    # print('response', response)
    json_content = extract_json(response)
    return json_content.get('toc_detected', 'no')


def check_if_toc_extraction_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a partial document  and a  table of contents.
    Your job is to check if the  table of contents is complete, which it contains all the main sections in the partial document.

    Reply format:
    {{
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not include explanations or any other text."""

    prompt = prompt + '\n Document:\n' + content + '\n Table of contents:\n' + toc
    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content.get('completed', 'no')


def check_if_toc_transformation_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a raw table of contents and a  table of contents.
    Your job is to check if the  table of contents is complete.

    Reply format:
    {{
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not include explanations or any other text."""

    prompt = prompt + '\n Raw Table of contents:\n' + content + '\n Cleaned Table of contents:\n' + toc
    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content.get('completed', 'no')

def extract_toc_content(content, model=None):
    prompt = f"""
    Your job is to extract the full table of contents from the given text, replace ... with :

    Given text: {content}

    Directly return the full table of contents content. Do not output anything else."""

    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    
    if_complete = check_if_toc_transformation_is_complete(content, response, model)
    if if_complete == "yes" and finish_reason == "finished":
        return response
    
    chat_history = [
        {"role": "user", "content": prompt}, 
        {"role": "assistant", "content": response},    
    ]
    prompt = f"""please continue the generation of table of contents , directly output the remaining part of the structure"""
    new_response, finish_reason = llm_completion(model=model, prompt=prompt, chat_history=chat_history, return_finish_reason=True)
    response = response + new_response
    if_complete = check_if_toc_transformation_is_complete(content, response, model)
    
    attempt = 0
    max_attempts = 5

    while not (if_complete == "yes" and finish_reason == "finished"):
        attempt += 1
        if attempt > max_attempts:
            raise Exception('Failed to complete table of contents after maximum retries')

        chat_history = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        prompt = f"""please continue the generation of table of contents , directly output the remaining part of the structure"""
        new_response, finish_reason = llm_completion(model=model, prompt=prompt, chat_history=chat_history, return_finish_reason=True)
        response = response + new_response
        if_complete = check_if_toc_transformation_is_complete(content, response, model)
    
    return response

def detect_page_index(toc_content, model=None):
    print('start detect_page_index')
    prompt = f"""
    You will be given a table of contents.

    Your job is to detect if there are page numbers/indices given within the table of contents.

    Given text: {toc_content}

    Reply format:
    {{
        "page_index_given_in_toc": "<yes or no>"
    }}
    Directly return the final JSON structure. Do not include explanations or any other text."""

    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content.get('page_index_given_in_toc', 'no')

def toc_extractor(page_list, toc_page_list, model):
    def transform_dots_to_colon(text):
        text = re.sub(r'\.{5,}', ': ', text)
        # Handle dots separated by spaces
        text = re.sub(r'(?:\. ){5,}\.?', ': ', text)
        return text
    
    toc_content = ""
    for page_index in toc_page_list:
        toc_content += page_list[page_index][0]
    toc_content = transform_dots_to_colon(toc_content)
    has_page_index = detect_page_index(toc_content, model=model)
    
    return {
        "toc_content": toc_content,
        "page_index_given_in_toc": has_page_index
    }




def toc_index_extractor(toc, content, model=None):
    print('start toc_index_extractor')
    toc_extractor_prompt = """
    You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format: 
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        },
        ...
    ]

    Only add the physical_index to the sections that are in the provided pages.
    If the section is not in the provided pages, do not add the physical_index to it.
    Directly return the final JSON structure. Do not output anything else."""

    prompt = toc_extractor_prompt + '\nTable of contents:\n' + str(toc) + '\nDocument pages:\n' + content
    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)    
    return json_content



# Ligature characters that PyPDF2/pymupdf often emit verbatim. Map back to
# ASCII so substring/title matching against the rendered page text works.
_TOC_NOISE_REPLACEMENTS = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def _normalize_toc_text(text):
    for src, dst in _TOC_NOISE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    # Common PDF extraction issue: one TOC entry is glued to the next one.
    text = re.sub(r"(?<=[0-9ivxlcdm])(?=Appendix\s+[A-Z]\b)", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=[0-9ivxlcdm])(?=References\b)", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=[0-9ivxlcdm])(?=Index\b)", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=[0-9ivxlcdm])(?=Contents\s+[ivxlcdm]+\b)", "\n", text, flags=re.IGNORECASE)
    return text


def _strip_toc_line_noise(line):
    line = line.strip()
    line = re.sub(r"(?<=\d)[ivxlcdm]+\s*CONTENTS\b", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\b[ivxlcdm]+\s*CONTENTS\b", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\s*CONTENTS\s+[ivxlcdm]+\b", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def _normalize_toc_title(title):
    title = re.sub(r"(?:\s*\.\s*){2,}", " ", title)
    title = re.sub(r"\s*:\s*$", "", title)
    title = re.sub(r"\s+", " ", title).strip(" .:")
    # PDF extraction often splits a leading capital from the rest: "V ariables".
    title = re.sub(r"\b([A-Z])\s+([a-z][a-z]+)\b", r"\1\2", title)
    return title


def _normalize_toc_page(page_text):
    page_text = page_text.strip().replace(" ", "")
    if page_text.isdigit():
        return int(page_text)
    # Roman front-matter pages do not share the Arabic-page offset.
    if re.fullmatch(r"[ivxlcdm]+", page_text, flags=re.IGNORECASE):
        return None
    return None


def _toc_entry_from_match(match, current_chapter, front_matter_index):
    head = match.group("head").strip()
    title = _normalize_toc_title(match.group("title") or "")
    page = _normalize_toc_page(match.group("page"))
    lower_head = head.lower()

    if re.fullmatch(r"\d+(?:\.\d+)*", head):
        structure = head
        if "." not in head:
            current_chapter = head
        title = title or head
    elif lower_head.startswith("appendix"):
        letter = head.split()[-1]
        structure = letter
        title = _normalize_toc_title(f"{head} {title}")
    elif lower_head == "exercises":
        structure = f"{current_chapter}.exercises" if current_chapter else None
        title = "Exercises"
    elif lower_head in ("references", "index"):
        structure = lower_head
        title = head.title()
    else:
        front_matter_index += 1
        structure = f"0.{front_matter_index}"
        title = _normalize_toc_title(f"{head} {title}")

    return {
        "item": {"structure": structure, "title": title, "page": page},
        "current_chapter": current_chapter,
        "front_matter_index": front_matter_index,
    }


def _parse_toc_line_heuristic(line, current_chapter, front_matter_index):
    entry_pattern = re.compile(
        r"^(?P<head>\d+(?:\.\d+)*|Appendix\s+[A-Z]|Exercises|References|Index|Preface|Mathematical notation)\b"
        r"\s*(?P<title>.*?)\s*(?:[:.]\s*)?(?P<page>(?:\d\s*){1,5}|[ivxlcdm]+)\s*$",
        flags=re.IGNORECASE,
    )
    match = entry_pattern.match(line)
    if not match:
        return None
    parsed = _toc_entry_from_match(match, current_chapter, front_matter_index)
    item = parsed["item"]
    if not item["title"] or item["title"].lower().startswith("contents"):
        return None
    return parsed


def parse_toc_content_heuristic(toc_content):
    """Deterministic line-by-line TOC parser. Returns (items, stats) where
    items is a list of `{structure, title, page}` dicts and stats reports
    confidence = items/candidates. `toc_transformer` skips the LLM step
    when items >= 8 and confidence >= 0.65 — handles textbook-scale TOCs
    (e.g. PRML.pdf) without sending tens of pages to an LLM."""
    text = _normalize_toc_text(toc_content)
    items = []
    current_chapter = None
    front_matter_index = 0
    candidate_count = 0

    for raw_line in text.splitlines():
        line = _strip_toc_line_noise(raw_line)
        if not line or line.lower().startswith("contents"):
            continue
        if re.fullmatch(r"[ivxlcdm]+", line, flags=re.IGNORECASE):
            continue

        if re.search(r"(?:\d\s*){1,5}$|[ivxlcdm]+$", line, flags=re.IGNORECASE):
            candidate_count += 1

        parsed = _parse_toc_line_heuristic(line, current_chapter, front_matter_index)
        if not parsed:
            continue
        current_chapter = parsed["current_chapter"]
        front_matter_index = parsed["front_matter_index"]
        items.append(parsed["item"])

    # Confidence is parsing yield, computed BEFORE dedup so the gating
    # threshold in `toc_transformer` reflects raw parser quality, not
    # how many duplicates a particular layout happened to produce.
    confidence = len(items) / candidate_count if candidate_count else 0
    items = _dedupe_heuristic_items(items)
    stats = {
        "items": len(items),
        "candidates": candidate_count,
        "confidence": round(confidence, 3),
    }
    return items, stats


def _toc_chunks(toc_content, model=None):
    """Split a long TOC into chunks of ≤ `toc_chunk_max_tokens` lines so
    each chunk can be transformed by an LLM call without overflowing
    `max-model-len`. Used as the chunked LLM fallback when
    `parse_toc_content_heuristic` lacks confidence."""
    max_tokens = get_llm_runtime_int(
        "toc_chunk_max_tokens",
        ("PAGEINDEX_TOC_CHUNK_MAX_TOKENS",),
        6000,
    )
    lines = [_strip_toc_line_noise(line) for line in _normalize_toc_text(toc_content).splitlines()]
    lines = [line for line in lines if line]
    chunks = []
    current = []

    for line in lines:
        candidate = "\n".join(current + [line])
        try:
            token_count = count_tokens(candidate, model=model)
        except Exception:
            token_count = len(candidate) // 4
        if current and token_count > max_tokens:
            chunks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("\n".join(current))
    return chunks or [toc_content]


def _transform_toc_chunk_with_llm(toc_content, model=None, logger=None):
    prompt = """
    You are given a table of contents. Transform it into JSON.

    Return only a JSON array. Each item must be:
    {
      "structure": <section index such as "1", "1.2", "A", or null>,
      "title": <section title>,
      "page": <printed page number as integer, or null>
    }

    Preserve all entries in the given table of contents. Do not output prose.
    Table of contents:
    """ + toc_content

    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    items = _finish_toc_json(prompt, response, finish_reason, model=model, logger=logger)
    if isinstance(items, dict) and isinstance(items.get("table_of_contents"), list):
        items = items["table_of_contents"]
    if not isinstance(items, list):
        items = []
    return convert_page_to_int(items)


def toc_transformer(toc_content, model=None, logger=None):
    """Turn raw TOC text into JSON items. Tries the deterministic
    heuristic first; falls back to a chunk-by-chunk LLM transform with
    `_finish_toc_json` continuation when the heuristic isn't confident."""
    print('start toc_transformer')
    heuristic_items, stats = parse_toc_content_heuristic(toc_content)
    if logger:
        logger.info(f'toc heuristic stats: {stats}')
    if stats["items"] >= 8 and stats["confidence"] >= 0.65:
        if logger:
            logger.info(f'toc_transformer using heuristic parser with {stats["items"]} items')
        return convert_page_to_int(heuristic_items)

    chunks = _toc_chunks(toc_content, model=model)
    if logger:
        logger.info(f'toc_transformer using chunked LLM fallback with {len(chunks)} chunks')

    transformed = []
    for chunk in chunks:
        chunk_items = _transform_toc_chunk_with_llm(chunk, model=model, logger=logger)
        transformed = _merge_toc_items(transformed, chunk_items)

    if transformed:
        return convert_page_to_int(transformed)
    raise Exception('Failed to complete toc transformation after maximum retries')
    



def find_toc_pages(start_page_index, page_list, opt, logger=None):
    print('start find_toc_pages')
    last_page_is_yes = False
    toc_page_list = []
    i = start_page_index
    
    while i < len(page_list):
        # Only check beyond max_pages if we're still finding TOC pages
        if i >= opt.toc_check_page_num and not last_page_is_yes:
            break
        detected_result = toc_detector_single_page(page_list[i][0],model=opt.model)
        if detected_result == 'yes':
            if logger:
                logger.info(f'Page {i} has toc')
            toc_page_list.append(i)
            last_page_is_yes = True
        elif detected_result == 'no' and last_page_is_yes:
            if logger:
                logger.info(f'Found the last page with toc: {i-1}')
            break
        i += 1
    
    if not toc_page_list and logger:
        logger.info('No toc found')
        
    return toc_page_list

def remove_page_number(data):
    if isinstance(data, dict):
        data.pop('page_number', None)  
        data.pop('page', None)
        for key in list(data.keys()):
            if 'nodes' in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data

def extract_matching_page_pairs(toc_page, toc_physical_index, start_page_index):
    pairs = []
    for phy_item in toc_physical_index:
        for page_item in toc_page:
            if phy_item.get('title') == page_item.get('title'):
                physical_index = phy_item.get('physical_index')
                if physical_index is not None and int(physical_index) >= start_page_index:
                    pairs.append({
                        'title': phy_item.get('title'),
                        'page': page_item.get('page'),
                        'physical_index': physical_index
                    })
    return pairs


def calculate_page_offset(pairs):
    differences = []
    for pair in pairs:
        try:
            physical_index = pair['physical_index']
            page_number = pair['page']
            difference = physical_index - page_number
            differences.append(difference)
        except (KeyError, TypeError):
            continue
    
    if not differences:
        return None
    
    difference_counts = {}
    for diff in differences:
        difference_counts[diff] = difference_counts.get(diff, 0) + 1
    
    most_common = max(difference_counts.items(), key=lambda x: x[1])[0]
    
    return most_common

def add_page_offset_to_toc_json(data, offset):
    for i in range(len(data)):
        if data[i].get('page') is not None and isinstance(data[i]['page'], int):
            data[i]['physical_index'] = data[i]['page'] + offset
            del data[i]['page']
    
    return data


def guess_page_offset_from_toc(toc_items, toc_page_list):
    """Cheap deterministic offset guess: the first content page is
    typically TOC-end + 2 (one blank page after the TOC). Map that
    physical index back to the printed page number on the first
    body-section TOC item to derive the offset. Skips front-matter
    items (structure starts with '0.') and roman-page entries.
    Returns None when the heuristic can't decide; callers fall back
    to `calculate_page_offset` over LLM-matched pairs."""
    if not toc_items or not toc_page_list:
        return None
    first_content_physical_index = toc_page_list[-1] + 2
    for item in toc_items:
        page = item.get("page")
        structure = item.get("structure")
        if not isinstance(page, int) or page <= 0:
            continue
        if isinstance(structure, str) and structure.startswith("0."):
            continue
        offset = first_content_physical_index - page
        return offset if offset >= 0 else None
    return None


def _default_group_max_tokens(model=None):
    """Compute the page-group token budget based on the active server's
    context window. For vllm/hosted_vllm we subtract the per-call output
    cap, the chat-template safety margin, and a fixed prompt overhead so
    the assembled prompt + JSON output fits in `max-model-len`. Falls
    back to the upstream 20 k default for OpenAI/Anthropic-class models
    where context is large enough to be a non-issue."""
    normalized_model = model.removeprefix("litellm/") if model else ""
    if not normalized_model.startswith(("vllm/", "hosted_vllm/")):
        return 20000

    model_len = get_llm_runtime_int("vllm_max_model_len", ("VLLM_MAX_MODEL_LEN",), 16384)
    output_cap = get_llm_runtime_int("vllm_max_tokens", ("VLLM_MAX_TOKENS",), 2048)
    if output_cap <= 0:
        output_cap = 2048
    margin = get_llm_runtime_int("vllm_ctx_margin", ("VLLM_CTX_MARGIN",), 256)
    prompt_overhead = get_llm_runtime_int(
        "pageindex_group_prompt_overhead",
        ("PAGEINDEX_GROUP_PROMPT_OVERHEAD",),
        1600,
    )
    available = model_len - output_cap - margin - prompt_overhead
    return max(2000, min(available, model_len // 2, 10000))


def page_list_to_group_text(page_contents, token_lengths, max_tokens=None, overlap_page=1, model=None):
    # Upstream default targets GPT-4 (128k context). Local vLLM servers often
    # run at 16k/32k, so use a smaller default unless explicitly overridden.
    if max_tokens is None:
        max_tokens = get_llm_runtime_int(
            "pageindex_group_max_tokens",
            ("PAGEINDEX_GROUP_MAX_TOKENS",),
            0,
        ) or _default_group_max_tokens(model)
    num_tokens = sum(token_lengths)
    
    if num_tokens <= max_tokens:
        # merge all pages into one text
        page_text = "".join(page_contents)
        return [page_text]
    
    subsets = []
    current_subset = []
    current_token_count = 0

    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(((num_tokens / expected_parts_num) + max_tokens) / 2)
    
    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_subset and current_token_count + page_tokens > average_tokens_per_part:

            subsets.append(''.join(current_subset))
            # Start new subset from overlap if specified
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])
        
        # Add current page to the subset
        current_subset.append(page_content)
        current_token_count += page_tokens

    # Add the last subset if it contains any pages
    if current_subset:
        subsets.append(''.join(current_subset))
    
    print('divide page_list to groups', len(subsets))
    return subsets

def _extract_page_range_from_group(group_text):
    """Parse `<physical_index_X>` tags from a page-group text and return
    (min_phys, max_phys). Returns (None, None) when no tags are found.
    Used by `process_toc_no_page_numbers` to tell
    `add_page_number_to_toc` which page range a given group covers so it
    can prune candidates that cannot possibly start there."""
    if not group_text:
        return None, None
    indices = re.findall(r'<physical_index_(\d+)>', group_text)
    if not indices:
        return None, None
    ints = [int(s) for s in indices]
    return min(ints), max(ints)


def _coerce_physical_index_to_int(phys):
    """Best-effort int conversion for a `physical_index` value.

    Accepts either a plain int or the `<physical_index_N>` string format
    items wear while still in flight (before
    `convert_physical_index_to_int` runs). Returns None when neither."""
    if isinstance(phys, int):
        return phys
    if isinstance(phys, str):
        m = re.search(r'physical_index_(\d+)', phys)
        if m:
            return int(m.group(1))
    return None


def _filter_by_feasible_range(structure, unfilled_indices, g_min, g_max):
    """Phase 3.1 chapter-local pruning.

    For each unfilled item, the nearest filled neighbours in TOC order
    bound its physical_index: prev_phys ≤ feasible ≤ next_phys. Drop
    items whose feasible window doesn't overlap the current group's
    page range [g_min, g_max]. As the document fills out, each
    chapter's subsections converge onto the page-group that actually
    covers them — total LLM-call cost falls from O(TOC × groups) to
    roughly O(TOC). Strictly preserves correctness: a dropped item
    cannot start in this group's pages, so it has nothing to contribute
    here and will be reconsidered against later groups."""
    if g_min is None or g_max is None:
        return list(unfilled_indices)

    out = []
    unfilled_set = set(unfilled_indices)
    n = len(structure)
    for i in unfilled_indices:
        prev_phys = 1
        for j in range(i - 1, -1, -1):
            if j in unfilled_set:
                continue
            it = structure[j]
            if not isinstance(it, dict):
                continue
            cand = _coerce_physical_index_to_int(it.get('physical_index'))
            if cand is not None:
                prev_phys = cand
                break
        next_phys = float('inf')
        for j in range(i + 1, n):
            if j in unfilled_set:
                continue
            it = structure[j]
            if not isinstance(it, dict):
                continue
            cand = _coerce_physical_index_to_int(it.get('physical_index'))
            if cand is not None:
                next_phys = cand
                break
        if g_max >= prev_phys and g_min <= next_phys:
            out.append(i)
    return out


_ADD_PAGE_NUMBER_PROMPT = """
    You are given an JSON structure of a document and a partial part of the document. Your task is to check if the title that is described in the structure is started in the partial given document.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    If the full target section starts in the partial given document, insert the given JSON structure with the "start": "yes", and "start_index": "<physical_index_X>".

    If the full target section does not start in the partial given document, insert "start": "no",  "start_index": None.

    The response should be in the following format.
        [
            {
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "start": "<yes or no>",
                "physical_index": "<physical_index_X> (keep the format)" or None
            },
            ...
        ]
    Directly return the final JSON structure. Do not output anything else."""


async def _add_page_number_chunk(part, sub_structure, model=None, logger=None):
    """Single LLM call asking which items in `sub_structure` start
    within `part`. Salvages truncated replies via `_finish_toc_json`
    (recovers the items that did parse and re-prompts for the rest).
    Returns a list of {structure, title, physical_index, ...} dicts.

    Async via `asyncio.to_thread` around the sync llm_completion +
    _finish_toc_json body — keeps the salvage logic in one place while
    letting the outer add_page_number_to_toc run many chunk calls
    concurrently under `_get_llm_sem`."""
    cleaned = []
    for item in sub_structure:
        if not isinstance(item, dict):
            continue
        cleaned.append({
            k: v for k, v in item.items()
            if k not in ('physical_index', 'start', 'start_index')
        })

    prompt = (
        _ADD_PAGE_NUMBER_PROMPT
        + f"\n\nCurrent Partial Document:\n{part}\n\nGiven Structure\n"
        + json.dumps(cleaned, indent=2)
        + "\n"
    )

    def _sync_body():
        response, finish_reason = llm_completion(
            model=model, prompt=prompt, return_finish_reason=True,
        )
        items = _finish_toc_json(prompt, response, finish_reason, model=model, logger=logger)
        if isinstance(items, dict) and isinstance(items.get('table_of_contents'), list):
            items = items['table_of_contents']
        if not isinstance(items, list):
            items = []
        for item in items:
            if isinstance(item, dict) and 'start' in item:
                del item['start']
        return items

    return await asyncio.to_thread(_sync_body)


async def add_page_number_to_toc(part, structure, model=None, chunk_size=None, logger=None, group_page_range=None):
    """Fill `physical_index` for items in `structure` whose section
    starts within `part`.

    Five layered scaling behaviors on top of the upstream single-shot
    prompt. Same correctness, bounded prompt size, parallelism within
    a page-group:

      * **Tail-only replay** (Phase 2.2) — items already carrying a
        non-None `physical_index` from an earlier page-group pass are
        kept in the returned structure but NOT re-sent to the LLM.
      * **Chunked structure** (Phase 2.1) — unfilled items are split
        into `chunk_size`-item windows
        (env `PAGEINDEX_FILL_CHUNK_SIZE`, default 50). Each window
        fires its own LLM call so the per-call prompt is bounded
        irrespective of TOC size.
      * **Truncation salvage** (Phase 2.3) — each chunk call routes
        through `_finish_toc_json` to recover from
        `finish_reason=length`.
      * **Chapter-local pruning** (Phase 3.1) — when the caller
        supplies `group_page_range=(g_min, g_max)`, drop unfilled
        items whose feasible window (bounded by the nearest filled
        TOC neighbours) doesn't overlap. As items fill in across the
        page-group loop, later groups receive a smaller candidate
        list — total LLM-call cost falls from O(TOC × groups) toward
        O(TOC).
      * **Within-group parallelism** (Phase 3.2) — chunk calls fan
        out via `asyncio.gather` under `_get_llm_sem`. Disjoint item
        subsets → no fill conflicts. Wall-clock falls near-linearly
        with `PAGEINDEX_LLM_CONCURRENCY`.

    Mutates `structure` in place when given a list (preserves items
    the LLM omits). For backward compatibility with
    `process_none_page_numbers`'s single-item call site, also accepts
    a dict and returns the chunk reply directly (caller indexes [0])."""
    if isinstance(structure, dict):
        return await _add_page_number_chunk(part, [structure], model=model, logger=logger)
    if not isinstance(structure, list):
        return structure

    unfilled_indices = [
        i for i, it in enumerate(structure)
        if isinstance(it, dict) and it.get('physical_index') is None
    ]
    if not unfilled_indices:
        return structure

    if group_page_range and group_page_range[0] is not None:
        before = len(unfilled_indices)
        unfilled_indices = _filter_by_feasible_range(
            structure, unfilled_indices,
            group_page_range[0], group_page_range[1],
        )
        if logger and before != len(unfilled_indices):
            logger.info(
                f'add_page_number_to_toc: chapter-local pruning '
                f'{before} → {len(unfilled_indices)} candidates '
                f'for group_range=[{group_page_range[0]}, {group_page_range[1]}]'
            )
        if not unfilled_indices:
            return structure

    if chunk_size is None:
        chunk_size = get_llm_runtime_int(
            "pageindex_fill_chunk_size",
            ("PAGEINDEX_FILL_CHUNK_SIZE",),
            50,
        )
    chunk_size = max(1, chunk_size)

    chunks = [
        unfilled_indices[start:start + chunk_size]
        for start in range(0, len(unfilled_indices), chunk_size)
    ]
    if logger:
        logger.info(
            f'add_page_number_to_toc: {len(unfilled_indices)} unfilled items '
            f'split into {len(chunks)} chunk(s) of <= {chunk_size}'
        )

    sem = _get_llm_sem()

    async def _run_chunk(chunk_idx, chunk_idx_list):
        sub_structure = [structure[i] for i in chunk_idx_list]
        async with sem:
            try:
                sub_result = await _add_page_number_chunk(
                    part, sub_structure, model=model, logger=logger,
                )
                return chunk_idx_list, sub_result
            except Exception as e:
                if logger:
                    logger.info(
                        f'add_page_number_to_toc chunk {chunk_idx+1}/{len(chunks)} '
                        f'failed; keeping prior state. err={e}'
                    )
                return chunk_idx_list, []

    results = await asyncio.gather(*(
        _run_chunk(i, chunk) for i, chunk in enumerate(chunks)
    ))

    for chunk_idx_list, sub_result in results:
        result_by_key = {}
        for r in sub_result:
            if not isinstance(r, dict):
                continue
            key = (
                str(r.get('structure', '')) if r.get('structure') is not None else None,
                str(r.get('title', '')) if r.get('title') is not None else None,
            )
            result_by_key[key] = r

        for i in chunk_idx_list:
            orig = structure[i]
            if not isinstance(orig, dict):
                continue
            key = (
                str(orig.get('structure', '')) if orig.get('structure') is not None else None,
                str(orig.get('title', '')) if orig.get('title') is not None else None,
            )
            r = result_by_key.get(key)
            if r is None:
                continue
            phys = r.get('physical_index')
            if phys is None:
                continue
            orig['physical_index'] = phys

    return structure


def remove_first_physical_index_section(text):
    """
    Removes the first section between <physical_index_X> and <physical_index_X> tags,
    and returns the remaining text.
    """
    pattern = r'<physical_index_\d+>.*?<physical_index_\d+>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Remove the first matched section
        return text.replace(match.group(0), '', 1)
    return text


def _extract_complete_toc_items(content):
    """Salvage every fully-formed JSON object from a possibly-truncated
    LLM response. Used when generation hit `max_output_reached` mid-array
    — returns whatever objects parsed cleanly so `_finish_toc_json` can
    re-prompt for the rest instead of throwing the whole call away."""
    if not content:
        return []

    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = get_json_content(cleaned)

    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(cleaned)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("table_of_contents"), list):
            return [item for item in parsed["table_of_contents"] if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    items = []
    idx = 0
    while idx < len(cleaned):
        obj_start = cleaned.find("{", idx)
        if obj_start == -1:
            break
        try:
            value, offset = decoder.raw_decode(cleaned[obj_start:])
        except json.JSONDecodeError:
            idx = obj_start + 1
            continue
        if isinstance(value, dict):
            if isinstance(value.get("table_of_contents"), list):
                items.extend(item for item in value["table_of_contents"] if isinstance(item, dict))
            else:
                items.append(value)
        idx = obj_start + max(offset, 1)
    return items


_CONTINUATION_SUFFIX_RE = re.compile(
    r'\s*[\(\[\-–—]+\s*(?:continued|cont\.?|cont\'d|part\s*\d+)\s*[\)\]]?\s*$',
    re.I,
)


def _strip_continuation_suffix(title):
    """Remove trailing '(Continued)', 'cont.', '- Part 2' etc. so that the
    same logical section emitted across multiple chunks collapses to one
    title."""
    if not isinstance(title, str):
        return title
    cleaned = title
    # Strip repeatedly in case the model stacked suffixes ("X (cont) (cont)").
    for _ in range(3):
        new = _CONTINUATION_SUFFIX_RE.sub('', cleaned).rstrip()
        if new == cleaned:
            break
        cleaned = new
    return cleaned


def _normalize_title_for_match(title):
    if not isinstance(title, str):
        return ''
    return re.sub(r'\s+', ' ', _strip_continuation_suffix(title)).strip().lower()


def _dedupe_heuristic_items(items, logger=None):
    """Collapse entries with identical (structure, title) into a single
    item, preferring the LAST occurrence in document order.

    Why: textbooks frequently ship two contents sections — a "Brief
    Contents" (chapters only) followed by a detailed "Contents"
    (chapters + subsections). `parse_toc_content_heuristic` captures
    both, duplicating every chapter entry. Keeping the LAST occurrence
    selects the detailed Contents pass, preserving chapter→subsection
    adjacency in the list so `post_processing` can build correct
    parent/child relationships and `guess_page_offset_from_toc` anchors
    on the detailed-contents pagination (not the brief-contents page
    numbers that broke the book2.pdf 1370-page run: 3.74% verify
    accuracy from the wrong anchor).

    Items missing both structure and title are passed through unchanged
    (preserves narrative front-matter)."""
    if not isinstance(items, list):
        return items
    last_index = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        struct = item.get('structure')
        title = item.get('title')
        if not struct and not title:
            continue
        key = (
            str(struct) if struct else None,
            str(title).strip().lower() if title else None,
        )
        last_index[key] = i

    dropped = 0
    cleaned = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            cleaned.append(item)
            continue
        struct = item.get('structure')
        title = item.get('title')
        if not struct and not title:
            cleaned.append(item)
            continue
        key = (
            str(struct) if struct else None,
            str(title).strip().lower() if title else None,
        )
        if last_index.get(key) == i:
            cleaned.append(item)
        else:
            dropped += 1

    if dropped and logger:
        logger.info(
            f'_dedupe_heuristic_items dropped {dropped} duplicate entries '
            '(typically Brief Contents vs detailed Contents)'
        )
    return cleaned


def _normalize_toc_items(items, logger=None):
    """Canonical post-mutation cleanup for TOC item lists. Composes
    `_dedupe_heuristic_items` (Brief-vs-detailed Contents collisions)
    with `_collapse_continuation_items` (LLM continuation echoes).
    Idempotent — safe to call at every TOC mutation point."""
    items = _dedupe_heuristic_items(items, logger=logger)
    items = _collapse_continuation_items(items, logger=logger)
    return items


def _collapse_continuation_items(items, logger=None):
    """Drop continuation echoes the model emits when a section spans several
    chunks. Two adjacent items collapse when they share the same `structure`
    key OR the same normalized title (after stripping (Continued)/Part N
    suffixes). The first occurrence wins; later echoes are folded away.
    Single-page span repeats at the same physical_index are also collapsed
    even when not adjacent — keeps process_large_node_recursively from
    fanning a one-page node into N siblings."""
    if not isinstance(items, list):
        return items
    cleaned = []
    seen_struct_phys = set()
    seen_title_phys = set()
    dropped = 0
    for it in items:
        if not isinstance(it, dict):
            cleaned.append(it)
            continue
        title = _strip_continuation_suffix(str(it.get('title', '')))
        it['title'] = title
        struct = str(it.get('structure', '') or '')
        phys = str(it.get('physical_index', '') or '')
        norm_title = _normalize_title_for_match(title)

        # Adjacent collapse against last kept item.
        if cleaned:
            prev = cleaned[-1]
            prev_struct = str(prev.get('structure', '') or '')
            prev_title = _normalize_title_for_match(prev.get('title', ''))
            same_struct = bool(struct) and struct == prev_struct
            same_title = bool(norm_title) and norm_title == prev_title
            if same_struct or same_title:
                dropped += 1
                continue

        # Non-adjacent collapse: same (structure, physical_index) or
        # (title, physical_index) seen earlier in the list.
        if struct and phys and (struct, phys) in seen_struct_phys:
            dropped += 1
            continue
        if norm_title and phys and (norm_title, phys) in seen_title_phys:
            dropped += 1
            continue

        cleaned.append(it)
        if struct and phys:
            seen_struct_phys.add((struct, phys))
        if norm_title and phys:
            seen_title_phys.add((norm_title, phys))

    if dropped and logger:
        logger.info(f'_collapse_continuation_items dropped {dropped} continuation echoes')
    return cleaned


def _merge_toc_items(existing, new_items):
    """Append new TOC entries while deduplicating by
    (structure, title, physical_index, page) — guards against the model
    repeating already-emitted entries during a continuation re-prompt."""
    merged = list(existing)
    seen = {
        (
            str(item.get("structure")),
            str(item.get("title")),
            str(item.get("physical_index")),
            str(item.get("page")),
        )
        for item in merged
        if isinstance(item, dict)
    }
    for item in new_items:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("structure")),
            str(item.get("title")),
            str(item.get("physical_index")),
            str(item.get("page")),
        )
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def _finish_toc_json(prompt, response, finish_reason, model=None, logger=None, max_attempts=3):
    """Recover from `finish_reason=length` on TOC-generation calls. Salvage
    complete objects from the truncated buffer, then re-prompt the model
    (with the prior turn replayed in chat_history) up to `max_attempts`
    times asking it to continue without repeating. Replaces the older
    behavior of raising on truncation and losing all prior work."""
    items = _extract_complete_toc_items(response)
    if finish_reason == 'finished':
        return items if items else extract_json(response)

    if finish_reason != 'max_output_reached':
        raise Exception(f'finish reason: {finish_reason}')

    if logger:
        logger.info(f'TOC generation hit max output; recovered {len(items)} complete items')

    chat_history = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    for _ in range(max_attempts):
        last_item = items[-1] if items else None
        continue_prompt = f"""
        Continue extracting the same table-of-contents JSON.
        Return only a JSON array of remaining complete objects.
        Do not repeat any objects already returned.
        Last complete object already kept:
        {json.dumps(last_item, ensure_ascii=False)}
        """
        response, finish_reason = llm_completion(
            model=model,
            prompt=continue_prompt,
            chat_history=chat_history,
            return_finish_reason=True,
        )
        new_items = _extract_complete_toc_items(response)
        items = _merge_toc_items(items, new_items)
        chat_history.extend([
            {"role": "user", "content": continue_prompt},
            {"role": "assistant", "content": response},
        ])
        if logger:
            logger.info(f'TOC continuation finish_reason={finish_reason}; total recovered items={len(items)}')
        if finish_reason == 'finished':
            break
        if finish_reason != 'max_output_reached' or not response:
            break

    if items:
        return items
    raise Exception(f'finish reason: {finish_reason}')


### add verify completeness
def generate_toc_continue(toc_content, part, model=None, logger=None, tail_n=15):
    print('start generate_toc_continue')
    prompt = """
    You are an expert in extracting hierarchical tree structure.
    You are given a tree structure of the previous part and the text of the current part.
    Your task is to continue the tree structure from the previous part to include the current part.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X. \

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    Strict continuation rules (follow exactly):
    - Only emit a section when its heading actually begins inside the given text.
    - Do NOT repeat any section already present in "Previous tree structure".
    - Do NOT add suffixes like "(Continued)", "(cont.)", "Part 2", or fabricate a new structure index ("F.1", "F.2") for a section that has already been emitted. A section that physically continues from an earlier chunk must be skipped entirely.
    - If the current text contains no new section starts, return [] and nothing else.
    - Never invent headings that are not visibly printed in the text.

    The response should be in the following format.
        [
            {
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            },
            ...
        ]

    Directly return the additional part of the final JSON structure. Do not output anything else."""

    # Only ship the tail of the prior TOC. The model just needs the last few
    # entries to know where to continue numbering. Shipping the entire
    # accumulated tree grows the prompt linearly with iterations and overflows
    # max-model-len on long books (an 800-page book can accumulate hundreds of
    # entries → ~50k tokens of prior-TOC payload alone).
    if isinstance(toc_content, list) and tail_n and len(toc_content) > tail_n:
        prior = toc_content[-tail_n:]
        prior_label = f'Previous tree structure (last {len(prior)} of {len(toc_content)} entries)'
    else:
        prior = toc_content
        prior_label = 'Previous tree structure'
    prompt = prompt + '\nGiven text\n:' + part + f'\n{prior_label}\n:' + json.dumps(prior, indent=2)
    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    return _finish_toc_json(prompt, response, finish_reason, model=model, logger=logger)
    
### add verify completeness
def generate_toc_init(part, model=None, logger=None):
    print('start generate_toc_init')
    prompt = """
    You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X. 

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format. 
        [
            {{
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            }},
            
        ],


    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\nGiven text\n:' + part
    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    return _finish_toc_json(prompt, response, finish_reason, model=model, logger=logger)

def process_no_toc(page_list, start_index=1, model=None, logger=None):
    page_contents=[]
    token_lengths=[]
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    group_texts = page_list_to_group_text(page_contents, token_lengths, model=model)
    logger.info(f'len(group_texts): {len(group_texts)}')

    toc_with_page_number = generate_toc_init(group_texts[0], model, logger=logger)
    if not isinstance(toc_with_page_number, list):
        toc_with_page_number = []
    for chunk_idx, group_text in enumerate(group_texts[1:], start=1):
        try:
            toc_with_page_number_additional = generate_toc_continue(toc_with_page_number, group_text, model, logger=logger)
        except Exception as e:
            # One bad chunk shouldn't throw away tens of minutes of accumulated
            # TOC work. Log and continue with what we have so far; downstream
            # code can still build a partial tree.
            if logger:
                logger.info(
                    f'generate_toc_continue failed on chunk {chunk_idx}/{len(group_texts) - 1}; '
                    f'keeping {len(toc_with_page_number)} items so far. err={e}'
                )
            continue
        if isinstance(toc_with_page_number_additional, list):
            # Dedup-on-extend: prevents the LLM repeating prior entries
            # (structure, title, physical_index, page key match) across
            # continuation chunks.
            toc_with_page_number = _merge_toc_items(
                toc_with_page_number, toc_with_page_number_additional
            )
    logger.info(f'generate_toc: {toc_with_page_number}')

    # Canonical normalize: dedup Brief-vs-detailed Contents collisions +
    # collapse '(Continued)' echoes (different `structure` keys "F.1",
    # "F.2" on identical physical_index get folded back into one section).
    toc_with_page_number = _normalize_toc_items(toc_with_page_number, logger=logger)
    logger.info(f'after _normalize_toc_items: {toc_with_page_number}')

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f'convert_physical_index_to_int: {toc_with_page_number}')

    return toc_with_page_number

async def process_toc_no_page_numbers(toc_content, toc_page_list, page_list,  start_index=1, model=None, logger=None):
    page_contents=[]
    token_lengths=[]
    toc_content = toc_transformer(toc_content, model, logger=logger)
    logger.info(f'toc_transformer: {toc_content}')
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    
    group_texts = page_list_to_group_text(page_contents, token_lengths, model=model)
    logger.info(f'len(group_texts): {len(group_texts)}')

    toc_with_page_number=copy.deepcopy(toc_content)
    # Outer loop stays sequential so tail-only replay + chapter-local
    # pruning compound: each group reads the latest fills and narrows
    # candidates further. Within-group chunk parallelism is what
    # actually speeds this up (see add_page_number_to_toc).
    for group_text in group_texts:
        group_min, group_max = _extract_page_range_from_group(group_text)
        toc_with_page_number = await add_page_number_to_toc(
            group_text, toc_with_page_number, model, logger=logger,
            group_page_range=(group_min, group_max),
        )
    logger.info(f'add_page_number_to_toc: {toc_with_page_number}')

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f'convert_physical_index_to_int: {toc_with_page_number}')

    return toc_with_page_number



def _quick_offset_heuristic_score(toc_items, offset, page_list, sample_size=10):
    """Apply `offset` to `toc_items` and check (purely deterministically)
    whether each item's title appears at the top of its predicted page.

    Returns (score, scored_count) where score = yes_count / scored_count,
    or (0.0, 0) when nothing could be scored. Uses
    `check_title_appearance_in_start_heuristic` (normalized prefix match
    within the first 1500 chars of the predicted page). Pure-Python — no
    LLM calls — fast enough to evaluate both candidate offsets before
    committing.

    Sample is spread evenly across the TOC, not the first N items, since
    front-matter items (Preface, Acknowledgements) have weak
    title-appearance signal and would bias a head-only sample."""
    if offset is None or not toc_items or not page_list:
        return 0.0, 0
    candidates = [
        it for it in toc_items
        if isinstance(it, dict)
        and isinstance(it.get('page'), int)
        and it.get('page') > 0
    ]
    if not candidates:
        return 0.0, 0
    n = len(candidates)
    if n <= sample_size:
        sampled = candidates
    else:
        step = n / sample_size
        sampled = [candidates[int(i * step)] for i in range(sample_size)]

    yes = 0
    scored = 0
    for item in sampled:
        page = item.get('page')
        title = item.get('title') or ''
        phys = page + offset
        if phys < 1 or phys > len(page_list):
            continue
        page_text = page_list[phys - 1][0] if page_list[phys - 1] else ''
        verdict = check_title_appearance_in_start_heuristic(title, page_text)
        if verdict == 'yes':
            yes += 1
            scored += 1
        elif verdict == 'no':
            scored += 1
        # None → ambiguous, don't count toward scored
    if scored == 0:
        return 0.0, 0
    return yes / scored, scored


def _compute_llm_pair_offset(toc_with_page_number, toc_page_list, page_list,
                             toc_check_page_num=None, model=None, logger=None):
    """LLM-paired offset: extract physical_index for items in the first N
    content pages via `toc_index_extractor` (chunked to fit `max-model-len`),
    match titles against `toc_with_page_number`, and `calculate_page_offset`
    from the resulting pairs. Returns None on failure.

    Extracted into a helper so `process_toc_with_page_numbers` can call it
    either as the primary path (when `guess_page_offset_from_toc` returns
    None) or as a cross-validation candidate against the cheap deterministic
    guess (Phase 1.3 of the book-scale fix plan)."""
    toc_no_page_number = remove_page_number(copy.deepcopy(toc_with_page_number))

    start_page_index = toc_page_list[-1] + 1
    page_snippets = []
    page_token_lens = []
    for page_index in range(start_page_index, min(start_page_index + (toc_check_page_num or 0), len(page_list))):
        snippet = f"<physical_index_{page_index+1}>\n{page_list[page_index][0]}\n<physical_index_{page_index+1}>\n\n"
        page_snippets.append(snippet)
        page_token_lens.append(count_tokens(snippet, model=model) or 0)

    budget = _default_group_max_tokens(model)
    toc_json_tokens = count_tokens(str(toc_no_page_number), model=model) or 0
    # Reserve room for the toc JSON (replayed in every prompt) and a
    # ~800-token prompt template overhead. Floor at 2000 so very long
    # TOCs still split into one-page-at-a-time chunks.
    content_budget = max(2000, budget - toc_json_tokens - 800)
    if logger:
        logger.info(
            f'toc_index chunking: budget={budget} toc_json_tokens={toc_json_tokens} '
            f'content_budget={content_budget} pages={len(page_snippets)}'
        )

    chunks = []
    current = []
    current_tokens = 0
    for snippet, snippet_tokens in zip(page_snippets, page_token_lens):
        if current and current_tokens + snippet_tokens > content_budget:
            chunks.append(''.join(current))
            current = [snippet]
            current_tokens = snippet_tokens
        else:
            current.append(snippet)
            current_tokens += snippet_tokens
    if current:
        chunks.append(''.join(current))
    if logger:
        logger.info(f'toc_index_extractor running over {len(chunks)} chunk(s)')

    toc_with_physical_index = []
    for i, chunk in enumerate(chunks):
        chunk_result = toc_index_extractor(toc_no_page_number, chunk, model)
        if isinstance(chunk_result, dict):
            # Defensive: extract_json may wrap list under a key
            if isinstance(chunk_result.get('table_of_contents'), list):
                chunk_result = chunk_result['table_of_contents']
            else:
                chunk_result = []
        if not isinstance(chunk_result, list):
            chunk_result = []
        chunk_result = convert_physical_index_to_int(chunk_result)
        toc_with_physical_index = _merge_toc_items(toc_with_physical_index, chunk_result)
        if logger:
            logger.info(
                f'toc_index_extractor chunk {i+1}/{len(chunks)} added '
                f'{len(chunk_result)} items; total={len(toc_with_physical_index)}'
            )

    matching_pairs = extract_matching_page_pairs(
        toc_with_page_number, toc_with_physical_index, start_page_index,
    )
    if logger:
        logger.info(f'matching_pairs: {matching_pairs}')

    offset = calculate_page_offset(matching_pairs)
    if logger:
        logger.info(f'llm_pair_offset: {offset}')
    return offset


async def process_toc_with_page_numbers(toc_content, toc_page_list, page_list,
                                        toc_check_page_num=None, model=None,
                                        logger=None, force_llm_pair_offset=False):
    """Drive the with-page-numbers TOC alignment path.

    Phase 1.3 cross-validation: when `guess_page_offset_from_toc` returns
    a deterministic cheap guess, run a heuristic 10-item check. If the
    cheap offset scores ≥ 0.7, commit to it; otherwise compute the
    LLM-pair offset and pick whichever scores higher. When
    `force_llm_pair_offset=True`, skip the cheap path entirely (used by
    meta_processor's cascade re-offset on verify accuracy < 0.2)."""
    toc_with_page_number = toc_transformer(toc_content, model, logger=logger)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    cheap_offset = None
    if not force_llm_pair_offset:
        cheap_offset = guess_page_offset_from_toc(toc_with_page_number, toc_page_list)
        logger.info(f'offset guess from toc: {cheap_offset}')
    else:
        logger.info('force_llm_pair_offset=True; skipping cheap deterministic guess')

    offset = None
    if cheap_offset is not None:
        cheap_score, cheap_scored = _quick_offset_heuristic_score(
            toc_with_page_number, cheap_offset, page_list,
        )
        logger.info(
            f'cheap_offset={cheap_offset} heuristic score={cheap_score:.2f} '
            f'({cheap_scored} items scored)'
        )
        if cheap_score >= 0.7 and cheap_scored >= 3:
            offset = cheap_offset
        else:
            # Cheap guess is dubious — compute LLM-pair offset and pick
            # whichever heuristic-scores higher on the same sample.
            llm_offset = _compute_llm_pair_offset(
                toc_with_page_number, toc_page_list, page_list,
                toc_check_page_num=toc_check_page_num, model=model, logger=logger,
            )
            if llm_offset is not None:
                llm_score, llm_scored = _quick_offset_heuristic_score(
                    toc_with_page_number, llm_offset, page_list,
                )
                logger.info(
                    f'llm_pair_offset={llm_offset} heuristic score={llm_score:.2f} '
                    f'({llm_scored} items scored)'
                )
                # Strict > to prefer cheap on ties (deterministic, no LLM cost).
                offset = llm_offset if llm_score > cheap_score else cheap_offset
                logger.info(
                    f'offset cross-validated: cheap={cheap_offset} '
                    f'llm={llm_offset} → chosen={offset}'
                )
            else:
                offset = cheap_offset
                logger.info(f'llm_pair_offset unavailable; falling back to cheap={offset}')
    if offset is None:
        # Either force_llm_pair_offset=True or guess_page_offset_from_toc
        # returned None (rare). Run LLM-pair as the primary path.
        offset = _compute_llm_pair_offset(
            toc_with_page_number, toc_page_list, page_list,
            toc_check_page_num=toc_check_page_num, model=model, logger=logger,
        )
    if offset is None:
        raise Exception('Failed to infer page offset from table of contents')

    toc_with_page_number = add_page_offset_to_toc_json(toc_with_page_number, offset)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_with_page_number = await process_none_page_numbers(toc_with_page_number, page_list, model=model)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    return toc_with_page_number



##check if needed to process none page numbers
async def process_none_page_numbers(toc_items, page_list, start_index=1, model=None):
    for i, item in enumerate(toc_items):
        if "physical_index" not in item:
            if item.get("page") is None:
                item["physical_index"] = None
                continue
            # logger.info(f"fix item: {item}")
            # Find previous physical_index
            prev_physical_index = 0  # Default if no previous item exists
            for j in range(i - 1, -1, -1):
                if toc_items[j].get('physical_index') is not None:
                    prev_physical_index = toc_items[j]['physical_index']
                    break
            
            # Find next physical_index
            next_physical_index = -1  # Default if no next item exists
            for j in range(i + 1, len(toc_items)):
                if toc_items[j].get('physical_index') is not None:
                    next_physical_index = toc_items[j]['physical_index']
                    break

            page_contents = []
            for page_index in range(prev_physical_index, next_physical_index+1):
                # Add bounds checking to prevent IndexError
                list_index = page_index - start_index
                if list_index >= 0 and list_index < len(page_list):
                    page_text = f"<physical_index_{page_index}>\n{page_list[list_index][0]}\n<physical_index_{page_index}>\n\n"
                    page_contents.append(page_text)
                else:
                    continue

            item_copy = copy.deepcopy(item)
            del item_copy['page']
            result = await add_page_number_to_toc(page_contents, item_copy, model)
            if isinstance(result[0]['physical_index'], str) and result[0]['physical_index'].startswith('<physical_index'):
                item['physical_index'] = int(result[0]['physical_index'].split('_')[-1].rstrip('>').strip())
                del item['page']
    
    return toc_items




def check_toc(page_list, opt=None):
    toc_page_list = find_toc_pages(start_page_index=0, page_list=page_list, opt=opt)
    if len(toc_page_list) == 0:
        print('no toc found')
        return {'toc_content': None, 'toc_page_list': [], 'page_index_given_in_toc': 'no'}
    else:
        print('toc found')
        toc_json = toc_extractor(page_list, toc_page_list, opt.model)

        if toc_json['page_index_given_in_toc'] == 'yes':
            print('index found')
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'yes'}
        else:
            current_start_index = toc_page_list[-1] + 1
            
            while (toc_json['page_index_given_in_toc'] == 'no' and 
                   current_start_index < len(page_list) and 
                   current_start_index < opt.toc_check_page_num):
                
                additional_toc_pages = find_toc_pages(
                    start_page_index=current_start_index,
                    page_list=page_list,
                    opt=opt
                )
                
                if len(additional_toc_pages) == 0:
                    break

                additional_toc_json = toc_extractor(page_list, additional_toc_pages, opt.model)
                if additional_toc_json['page_index_given_in_toc'] == 'yes':
                    print('index found')
                    return {'toc_content': additional_toc_json['toc_content'], 'toc_page_list': additional_toc_pages, 'page_index_given_in_toc': 'yes'}

                else:
                    current_start_index = additional_toc_pages[-1] + 1
            print('index not found')
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'no'}






################### fix incorrect toc #########################################################
async def single_toc_item_index_fixer(section_title, content, model=None):
    toc_extractor_prompt = """
    You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    Reply in a JSON format:
    {
        "physical_index": "<physical_index_X>" (keep the format)
    }
    Directly return the final JSON structure. Do not include explanations or any other text."""

    prompt = toc_extractor_prompt + '\nSection Title:\n' + str(section_title) + '\nDocument pages:\n' + content
    response = await llm_acompletion(model=model, prompt=prompt)
    json_content = extract_json(response)    
    return convert_physical_index_to_int(json_content.get('physical_index'))



async def fix_incorrect_toc(toc_with_page_number, page_list, incorrect_results, start_index=1, model=None, logger=None):
    print(f'start fix_incorrect_toc with {len(incorrect_results)} incorrect results')
    incorrect_indices = {result['list_index'] for result in incorrect_results}
    
    end_index = len(page_list) + start_index - 1
    
    incorrect_results_and_range_logs = []
    # Helper function to process and check a single incorrect item
    async def process_and_check_item(incorrect_item):
        list_index = incorrect_item['list_index']
        
        # Check if list_index is valid
        if list_index < 0 or list_index >= len(toc_with_page_number):
            # Return an invalid result for out-of-bounds indices
            return {
                'list_index': list_index,
                'title': incorrect_item['title'],
                'physical_index': incorrect_item.get('physical_index'),
                'is_valid': False
            }
        
        # Find the previous correct item
        prev_correct = None
        for i in range(list_index-1, -1, -1):
            if i not in incorrect_indices and i >= 0 and i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    prev_correct = physical_index
                    break
        # If no previous correct item found, use start_index
        if prev_correct is None:
            prev_correct = start_index - 1
        
        # Find the next correct item
        next_correct = None
        for i in range(list_index+1, len(toc_with_page_number)):
            if i not in incorrect_indices and i >= 0 and i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    next_correct = physical_index
                    break
        # If no next correct item found, use end_index
        if next_correct is None:
            next_correct = end_index
        
        incorrect_results_and_range_logs.append({
            'list_index': list_index,
            'title': incorrect_item['title'],
            'prev_correct': prev_correct,
            'next_correct': next_correct
        })

        page_contents=[]
        for page_index in range(prev_correct, next_correct+1):
            # Add bounds checking to prevent IndexError
            page_list_idx = page_index - start_index
            if page_list_idx >= 0 and page_list_idx < len(page_list):
                page_text = f"<physical_index_{page_index}>\n{page_list[page_list_idx][0]}\n<physical_index_{page_index}>\n\n"
                page_contents.append(page_text)
            else:
                continue
        content_range = ''.join(page_contents)
        
        physical_index_int = await single_toc_item_index_fixer(incorrect_item['title'], content_range, model)
        
        # Check if the result is correct
        check_item = incorrect_item.copy()
        check_item['physical_index'] = physical_index_int
        check_result = await check_title_appearance(check_item, page_list, start_index, model)

        return {
            'list_index': list_index,
            'title': incorrect_item['title'],
            'physical_index': physical_index_int,
            'is_valid': check_result['answer'] == 'yes'
        }

    # Process incorrect items concurrently
    tasks = [
        process_and_check_item(item)
        for item in incorrect_results
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(incorrect_results, results):
        if isinstance(result, Exception):
            print(f"Processing item {item} generated an exception: {result}")
            continue
    results = [result for result in results if not isinstance(result, Exception)]

    # Update the toc_with_page_number with the fixed indices and check for any invalid results
    invalid_results = []
    for result in results:
        if result['is_valid']:
            # Add bounds checking to prevent IndexError
            list_idx = result['list_index']
            if 0 <= list_idx < len(toc_with_page_number):
                toc_with_page_number[list_idx]['physical_index'] = result['physical_index']
            else:
                # Index is out of bounds, treat as invalid
                invalid_results.append({
                    'list_index': result['list_index'],
                    'title': result['title'],
                    'physical_index': result['physical_index'],
                })
        else:
            invalid_results.append({
                'list_index': result['list_index'],
                'title': result['title'],
                'physical_index': result['physical_index'],
            })

    logger.info(f'incorrect_results_and_range_logs: {incorrect_results_and_range_logs}')
    logger.info(f'invalid_results: {invalid_results}')

    return toc_with_page_number, invalid_results



async def fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results, start_index=1, max_attempts=3, model=None, logger=None):
    print('start fix_incorrect_toc')
    fix_attempt = 0
    current_toc = toc_with_page_number
    current_incorrect = incorrect_results

    while current_incorrect:
        print(f"Fixing {len(current_incorrect)} incorrect results")
        
        current_toc, current_incorrect = await fix_incorrect_toc(current_toc, page_list, current_incorrect, start_index, model, logger)
                
        fix_attempt += 1
        if fix_attempt >= max_attempts:
            logger.info("Maximum fix attempts reached")
            break
    
    return current_toc, current_incorrect




################### verify toc #########################################################
async def verify_toc(page_list, list_result, start_index=1, N=None, model=None):
    print('start verify_toc')
    # Find the last non-None physical_index
    last_physical_index = None
    for item in reversed(list_result):
        if item.get('physical_index') is not None:
            last_physical_index = item['physical_index']
            break
    
    # Early return if we don't have valid physical indices
    if last_physical_index is None or last_physical_index < len(page_list)/2:
        return 0, []
    
    # Determine which items to check
    if N is None:
        print('check all items')
        sample_indices = range(0, len(list_result))
    else:
        N = min(N, len(list_result))
        print(f'check {N} items')
        sample_indices = random.sample(range(0, len(list_result)), N)

    # Prepare items with their list indices
    indexed_sample_list = []
    for idx in sample_indices:
        item = list_result[idx]
        # Skip items with None physical_index (these were invalidated by validate_and_truncate_physical_indices)
        if item.get('physical_index') is not None:
            item_with_index = item.copy()
            item_with_index['list_index'] = idx  # Add the original index in list_result
            indexed_sample_list.append(item_with_index)

    # Run checks concurrently
    tasks = [
        check_title_appearance(item, page_list, start_index, model)
        for item in indexed_sample_list
    ]
    results = await asyncio.gather(*tasks)
    
    # Process results
    correct_count = 0
    incorrect_results = []
    for result in results:
        if result['answer'] == 'yes':
            correct_count += 1
        else:
            incorrect_results.append(result)
    
    # Calculate accuracy
    checked_count = len(results)
    accuracy = correct_count / checked_count if checked_count > 0 else 0
    print(f"accuracy: {accuracy*100:.2f}%")
    return accuracy, incorrect_results





################### main process #########################################################
async def meta_processor(page_list, mode=None, toc_content=None, toc_page_list=None,
                         start_index=1, opt=None, logger=None,
                         force_llm_pair_offset=False):
    print(mode)
    print(f'start_index: {start_index}')

    if mode == 'process_toc_with_page_numbers':
        toc_with_page_number = await process_toc_with_page_numbers(
            toc_content, toc_page_list, page_list,
            toc_check_page_num=opt.toc_check_page_num,
            model=opt.model,
            logger=logger,
            force_llm_pair_offset=force_llm_pair_offset,
        )
    elif mode == 'process_toc_no_page_numbers':
        toc_with_page_number = await process_toc_no_page_numbers(toc_content, toc_page_list, page_list, model=opt.model, logger=logger)
    else:
        toc_with_page_number = process_no_toc(page_list, start_index=start_index, model=opt.model, logger=logger)

    toc_with_page_number = [item for item in toc_with_page_number if item.get('physical_index') is not None]

    toc_with_page_number = validate_and_truncate_physical_indices(
        toc_with_page_number,
        len(page_list),
        start_index=start_index,
        logger=logger
    )

    verify_sample_num = getattr(opt, 'toc_verify_sample_num', None)
    if verify_sample_num is not None and verify_sample_num <= 0:
        verify_sample_num = None
    accuracy, incorrect_results = await verify_toc(
        page_list,
        toc_with_page_number,
        start_index=start_index,
        N=verify_sample_num,
        model=opt.model,
    )

    logger.info({
        'mode': mode,
        'accuracy': accuracy,
        'incorrect_results': incorrect_results,
        'force_llm_pair_offset': force_llm_pair_offset,
    })
    if accuracy == 1.0 and len(incorrect_results) == 0:
        return toc_with_page_number
    if accuracy > 0.6 and len(incorrect_results) > 0:
        toc_with_page_number, incorrect_results = await fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results,start_index=start_index, max_attempts=3, model=opt.model, logger=logger)
        return toc_with_page_number
    else:
        # Phase 1.4 cascade: when verify accuracy collapses (< 0.2) for
        # `process_toc_with_page_numbers` and we have NOT yet forced the
        # LLM-pair offset path, retry the same mode with
        # force_llm_pair_offset=True before falling through to the lossier
        # no_page_numbers fallback. This catches wrong-anchor offsets that
        # the heuristic cross-validation in process_toc_with_page_numbers
        # accepted but actual LLM verification rejects.
        if (mode == 'process_toc_with_page_numbers'
                and accuracy < 0.2
                and not force_llm_pair_offset):
            logger.info(
                f'cascade re-offset: accuracy={accuracy:.3f} < 0.2; '
                'retrying process_toc_with_page_numbers with '
                'force_llm_pair_offset=True before falling through'
            )
            return await meta_processor(
                page_list,
                mode='process_toc_with_page_numbers',
                toc_content=toc_content,
                toc_page_list=toc_page_list,
                start_index=start_index,
                opt=opt,
                logger=logger,
                force_llm_pair_offset=True,
            )
        if mode == 'process_toc_with_page_numbers':
            return await meta_processor(page_list, mode='process_toc_no_page_numbers', toc_content=toc_content, toc_page_list=toc_page_list, start_index=start_index, opt=opt, logger=logger)
        elif mode == 'process_toc_no_page_numbers':
            return await meta_processor(page_list, mode='process_no_toc', start_index=start_index, opt=opt, logger=logger)
        else:
            raise Exception('Processing failed')
        
 
async def process_large_node_recursively(node, page_list, opt=None, logger=None):
    # Single-page (or zero-span) nodes can never be subdivided meaningfully —
    # children would all carry the parent's physical_index and collapse to
    # duplicate siblings. Skip outright; also short-circuit obviously
    # inverted spans that may have leaked through.
    span_pages = (node.get('end_index') or 0) - (node.get('start_index') or 0) + 1
    if span_pages <= 1:
        return node

    node_page_list = page_list[node['start_index']-1:node['end_index']]
    token_num = sum([page[1] for page in node_page_list])

    if node['end_index'] - node['start_index'] > opt.max_page_num_each_node and token_num >= opt.max_token_num_each_node:
        print('large node:', node['title'], 'start_index:', node['start_index'], 'end_index:', node['end_index'], 'token_num:', token_num)

        node_toc_tree = await meta_processor(node_page_list, mode='process_no_toc', start_index=node['start_index'], opt=opt, logger=logger)
        # Belt-and-braces: canonical normalize before these items become
        # tree siblings — dedup duplicate (structure, title) + collapse
        # '(Continued)' echoes the model produced inside this recursion.
        node_toc_tree = _normalize_toc_items(node_toc_tree, logger=logger)
        node_toc_tree = await check_title_appearance_in_start_concurrent(node_toc_tree, page_list, model=opt.model, logger=logger)

        # Filter out items with None physical_index before post_processing
        valid_node_toc_items = [item for item in node_toc_tree if item.get('physical_index') is not None]
        
        if valid_node_toc_items and node['title'].strip() == valid_node_toc_items[0]['title'].strip():
            node['nodes'] = post_processing(valid_node_toc_items[1:], node['end_index'])
            node['end_index'] = valid_node_toc_items[1]['start_index'] if len(valid_node_toc_items) > 1 else node['end_index']
        else:
            node['nodes'] = post_processing(valid_node_toc_items, node['end_index'])
            node['end_index'] = valid_node_toc_items[0]['start_index'] if valid_node_toc_items else node['end_index']
        
    if 'nodes' in node and node['nodes']:
        tasks = [
            process_large_node_recursively(child_node, page_list, opt, logger=logger)
            for child_node in node['nodes']
        ]
        await asyncio.gather(*tasks)
    
    return node

async def tree_parser(page_list, opt, doc=None, logger=None):
    check_toc_result = check_toc(page_list, opt)
    logger.info(check_toc_result)

    if check_toc_result.get("toc_content") and check_toc_result["toc_content"].strip() and check_toc_result["page_index_given_in_toc"] == "yes":
        toc_with_page_number = await meta_processor(
            page_list, 
            mode='process_toc_with_page_numbers', 
            start_index=1, 
            toc_content=check_toc_result['toc_content'], 
            toc_page_list=check_toc_result['toc_page_list'], 
            opt=opt,
            logger=logger)
    else:
        toc_with_page_number = await meta_processor(
            page_list, 
            mode='process_no_toc', 
            start_index=1, 
            opt=opt,
            logger=logger)

    toc_with_page_number = add_preface_if_needed(toc_with_page_number)
    toc_with_page_number = await check_title_appearance_in_start_concurrent(toc_with_page_number, page_list, model=opt.model, logger=logger)
    
    # Filter out items with None physical_index before post_processings
    valid_toc_items = [item for item in toc_with_page_number if item.get('physical_index') is not None]
    
    toc_tree = post_processing(valid_toc_items, len(page_list))
    tasks = [
        process_large_node_recursively(node, page_list, opt, logger=logger)
        for node in toc_tree
    ]
    await asyncio.gather(*tasks)
    
    return toc_tree


def page_index_main(doc, opt=None):
    logger = JsonLogger(doc)
    
    is_valid_pdf = (
        (isinstance(doc, str) and os.path.isfile(doc) and doc.lower().endswith(".pdf")) or 
        isinstance(doc, BytesIO)
    )
    if not is_valid_pdf:
        raise ValueError("Unsupported input type. Expected a PDF file path or BytesIO object.")

    print('Parsing PDF...')
    page_list = get_page_tokens(doc, model=opt.model)

    logger.info({'total_page_number': len(page_list)})
    logger.info({'total_token': sum([page[1] for page in page_list])})

    async def page_index_builder():
        structure = await tree_parser(page_list, opt, doc=doc, logger=logger)
        if opt.if_add_node_id == 'yes':
            write_node_id(structure)    
        if opt.if_add_node_text == 'yes':
            add_node_text(structure, page_list)
        if opt.if_add_node_summary == 'yes':
            if opt.if_add_node_text == 'no':
                add_node_text(structure, page_list)
            await generate_summaries_for_structure(structure, model=opt.model)
            if opt.if_add_node_text == 'no':
                remove_structure_text(structure)
            if opt.if_add_doc_description == 'yes':
                # Create a clean structure without unnecessary fields for description generation
                clean_structure = create_clean_structure_for_description(structure)
                doc_description = generate_doc_description(clean_structure, model=opt.model)
                structure = format_structure(structure, order=['title', 'node_id', 'start_index', 'end_index', 'summary', 'text', 'nodes'])
                return {
                    'doc_name': get_pdf_name(doc),
                    'doc_description': doc_description,
                    'structure': structure,
                }
        structure = format_structure(structure, order=['title', 'node_id', 'start_index', 'end_index', 'summary', 'text', 'nodes'])
        return {
            'doc_name': get_pdf_name(doc),
            'structure': structure,
        }

    return asyncio.run(page_index_builder())


def page_index(doc, model=None, toc_check_page_num=None, toc_verify_sample_num=None, max_page_num_each_node=None, max_token_num_each_node=None,
               if_add_node_id=None, if_add_node_summary=None, if_add_doc_description=None, if_add_node_text=None):
    
    user_opt = {
        arg: value for arg, value in locals().items()
        if arg != "doc" and value is not None
    }
    opt = ConfigLoader().load(user_opt)
    return page_index_main(doc, opt)


def validate_and_truncate_physical_indices(toc_with_page_number, page_list_length, start_index=1, logger=None):
    """
    Validates and truncates physical indices that exceed the actual document length.
    This prevents errors when TOC references pages that don't exist in the document (e.g. the file is broken or incomplete).
    """
    if not toc_with_page_number:
        return toc_with_page_number
    
    max_allowed_page = page_list_length + start_index - 1
    truncated_items = []
    
    for i, item in enumerate(toc_with_page_number):
        if item.get('physical_index') is not None:
            original_index = item['physical_index']
            if original_index > max_allowed_page:
                item['physical_index'] = None
                truncated_items.append({
                    'title': item.get('title', 'Unknown'),
                    'original_index': original_index
                })
                if logger:
                    logger.info(f"Removed physical_index for '{item.get('title', 'Unknown')}' (was {original_index}, too far beyond document)")
    
    if truncated_items and logger:
        logger.info(f"Total removed items: {len(truncated_items)}")
        
    print(f"Document validation: {page_list_length} pages, max allowed index: {max_allowed_page}")
    if truncated_items:
        print(f"Truncated {len(truncated_items)} TOC items that exceeded document length")
     
    return toc_with_page_number
