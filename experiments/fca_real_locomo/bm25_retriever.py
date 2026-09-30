"""Lightweight BM25 retrieval over memory records."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _memory_doc(mem: Any) -> str:
    if isinstance(mem, dict):
        return " ".join([
            mem.get("content", ""),
            mem.get("context", ""),
            " ".join(mem.get("keywords") or []),
            " ".join(mem.get("tags") or []),
            mem.get("timestamp", ""),
        ])
    return " ".join([
        mem.content or "",
        mem.context or "",
        " ".join(mem.keywords or []),
        " ".join(mem.tags or []),
        mem.timestamp or "",
    ])


def bm25_search(
    memories: List[Any],
    query: str,
    top_k: int = 50,
    candidate_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """
    BM25 over memories, optionally restricted to candidate_ids subset.
    Returns [{"memory_id", "bm25_score"}, ...] descending.
    """
    if not memories or not query.strip():
        return []

    id_allow: Optional[Set[str]] = set(candidate_ids) if candidate_ids else None
    filtered: List[Any] = []
    for m in memories:
        mid = m.memory_id if hasattr(m, "memory_id") else m["memory_id"]
        if id_allow is not None and mid not in id_allow:
            continue
        filtered.append(m)

    if not filtered:
        return []

    docs = [_memory_doc(m) for m in filtered]
    tokenized = [_tokenize(d) for d in docs]
    if not any(tokenized):
        return []

    qtoks = _tokenize(query)
    if not qtoks:
        return []

    if HAS_RANK_BM25:
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(qtoks)
    else:
        # simple TF-IDF-like fallback
        scores = _tfidf_scores(tokenized, qtoks)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    out = []
    for idx, sc in ranked:
        if sc <= 0 and HAS_RANK_BM25:
            continue
        mid = filtered[idx].memory_id if hasattr(filtered[idx], "memory_id") else filtered[idx]["memory_id"]
        out.append({"memory_id": mid, "bm25_score": float(sc)})
    return out


def bm25_search_multi(
    memories: List[Any],
    queries: List[str],
    top_k: int = 50,
    candidate_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """Union BM25 scores across queries (max score per memory)."""
    best: Dict[str, float] = {}
    for q in queries:
        for hit in bm25_search(memories, q, top_k=top_k, candidate_ids=candidate_ids):
            mid = hit["memory_id"]
            best[mid] = max(best.get(mid, 0.0), hit["bm25_score"])
    return [{"memory_id": mid, "bm25_score": sc} for mid, sc in sorted(best.items(), key=lambda x: -x[1])[:top_k]]


def _tfidf_scores(docs: List[List[str]], query: List[str]) -> List[float]:
    from collections import Counter
    import math
    N = len(docs)
    df: Dict[str, int] = {}
    for doc in docs:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1
    qset = set(query)
    scores = []
    for doc in docs:
        tf = Counter(doc)
        s = 0.0
        for t in qset:
            if t in tf:
                idf = math.log((N + 1) / (df.get(t, 0) + 1))
                s += tf[t] * idf
        scores.append(s)
    return scores
