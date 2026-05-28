"""
Optional Phase-2 semantic layer: embedding + rerank HTTP clients.

These target OpenAI-/Cohere-compatible servers — by default the LightRAG
sibling stack (Qwen3-Embedding at ``:8001``, a reranker at ``:8002``). Kept
separate from ``index_store`` so BM25-only retrieval has zero dependency on
these servers running; the index build and query paths degrade to BM25 when
embeddings/rerank are unavailable.

Config is env-driven (CLI/caller args win):
  PAGEINDEX_EMBED_BASE_URL   default http://localhost:8001/v1
  PAGEINDEX_EMBED_MODEL      default rag-embed
  PAGEINDEX_EMBED_API_KEY    default EMPTY  (vLLM/TEI ignore the value)
  PAGEINDEX_EMBED_BATCH      default 16
  PAGEINDEX_EMBED_TIMEOUT    default 120 (seconds)
  PAGEINDEX_RERANK_BASE_URL  default http://localhost:8002/v1/rerank  (full path)
  PAGEINDEX_RERANK_MODEL     default rag-rerank
  PAGEINDEX_RERANK_API_KEY   default EMPTY
  PAGEINDEX_RERANK_TIMEOUT   default 120 (seconds)
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("pageindex.semantic")


def _env(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val not in (None, "") else default


def embed_config(base_url=None, model=None, api_key=None, batch=None, timeout=None) -> dict:
    return {
        "base_url": base_url or _env("PAGEINDEX_EMBED_BASE_URL", "http://localhost:8001/v1"),
        "model": model or _env("PAGEINDEX_EMBED_MODEL", "rag-embed"),
        "api_key": api_key or _env("PAGEINDEX_EMBED_API_KEY", "EMPTY"),
        "batch": int(batch or _env("PAGEINDEX_EMBED_BATCH", "16")),
        "timeout": float(timeout or _env("PAGEINDEX_EMBED_TIMEOUT", "120")),
    }


def rerank_config(base_url=None, model=None, api_key=None, timeout=None) -> dict:
    return {
        "base_url": base_url or _env("PAGEINDEX_RERANK_BASE_URL", "http://localhost:8002/v1/rerank"),
        "model": model or _env("PAGEINDEX_RERANK_MODEL", "rag-rerank"),
        "api_key": api_key or _env("PAGEINDEX_RERANK_API_KEY", "EMPTY"),
        "timeout": float(timeout or _env("PAGEINDEX_RERANK_TIMEOUT", "120")),
    }


def embed_texts(
    texts: list[str],
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    batch: int | None = None,
    timeout: float | None = None,
) -> list[list[float]]:
    """Embed a list of texts via an OpenAI-compatible /v1/embeddings endpoint.

    Returns one vector (list[float]) per input, in order. Raises on failure so
    the index build fails loudly (a half-embedded index would silently degrade
    retrieval). Empty strings are embedded as-is (servers handle them); callers
    should pass a non-empty fallback if that matters."""
    if not texts:
        return []
    from openai import OpenAI

    cfg = embed_config(base_url, model, api_key, batch, timeout)
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=cfg["timeout"])
    out: list[list[float]] = []
    bs = max(1, cfg["batch"])
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        resp = client.embeddings.create(model=cfg["model"], input=chunk)
        # OpenAI returns data possibly out of order; sort by index to be safe.
        ordered = sorted(resp.data, key=lambda d: d.index)
        out.extend(list(d.embedding) for d in ordered)
    if len(out) != len(texts):
        raise RuntimeError(f"embedding count mismatch: got {len(out)} for {len(texts)} inputs")
    return out


def embed_query(question: str, **kwargs) -> list[float]:
    return embed_texts([question], **kwargs)[0]


def rerank(
    query: str,
    documents: list[str],
    top_k: int | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> list[tuple[int, float]]:
    """Rerank documents against a query via a Cohere v2 /v1/rerank endpoint
    (vLLM- or TEI-served). Returns [(original_index, relevance_score), ...]
    sorted best-first, length min(top_k, len(documents)).

    Returns None on transport/parse failure so callers can fall back to the
    pre-rerank ordering instead of erroring out a whole query."""
    if not documents:
        return []
    import httpx

    cfg = rerank_config(base_url, model, api_key, timeout)
    payload = {"model": cfg["model"], "query": query, "documents": documents}
    if top_k:
        payload["top_n"] = int(top_k)
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"] and cfg["api_key"] != "EMPTY":
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    try:
        resp = httpx.post(cfg["base_url"], json=payload, headers=headers, timeout=cfg["timeout"])
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.exception("rerank request failed; caller should fall back to prior order")
        return None
    results = data.get("results")
    if not isinstance(results, list):
        logger.warning("rerank response missing 'results'; falling back")
        return None
    pairs: list[tuple[int, float]] = []
    for r in results:
        idx = r.get("index")
        score = r.get("relevance_score", r.get("score"))
        if isinstance(idx, int) and isinstance(score, (int, float)):
            pairs.append((idx, float(score)))
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs[: top_k] if top_k else pairs
