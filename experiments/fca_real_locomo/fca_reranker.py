"""Query-conditioned local FCA reranker for A-Mem retrieval candidates."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Set, Tuple, Any

from experiments.fca_real_locomo.extract_attributes import is_generic_attr, structural_overlap

PREFIX_WEIGHTS = {
    "entity:": 3.0,
    "speaker:": 3.0,
    "event:": 2.5,
    "date:": 3.0,
    "month:": 2.5,
    "year:": 2.0,
    "kw:": 1.0,
    "tag:": 0.5,
    "ctx:": 0.8,
    "ask:": 0.0,  # query-only
    "qtype:": 0.0,
}


def _minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _attr_weight(attr: str, is_temporal: bool) -> float:
    for prefix, w in PREFIX_WEIGHTS.items():
        if attr.startswith(prefix):
            if prefix in ("date:", "month:", "year:") and not is_temporal:
                return w * 0.5
            return w
    return 0.3


def compute_fca_scores(
    query_attrs: Set[str],
    candidate_attrs: Dict[str, Set[str]],
    is_temporal: bool = False,
) -> Dict[str, float]:
    """Compute raw FCA scores for each candidate memory_id."""
    N = len(candidate_attrs)
    if N == 0:
        return {}

    # document frequency over candidates
    df: Dict[str, int] = {}
    for attrs in candidate_attrs.values():
        for a in query_attrs & attrs:
            df[a] = df.get(a, 0) + 1

    scores: Dict[str, float] = {}
    for mid, mem_attrs in candidate_attrs.items():
        shared = query_attrs & mem_attrs
        if not shared:
            scores[mid] = 0.0
            continue
        s = 0.0
        generic_count = 0
        for a in shared:
            if is_generic_attr(a):
                generic_count += 1
                s -= 1.0
                continue
            idf = math.log((N + 1) / (df.get(a, 0) + 1))
            s += _attr_weight(a, is_temporal) * idf
        # broadness penalty: many generic overlaps
        s -= 0.3 * generic_count
        scores[mid] = max(s, 0.0)
    return scores


def _has_structural_share(query_attrs: Set[str], mem_attrs: Set[str]) -> bool:
    struct = structural_overlap(query_attrs, mem_attrs)
    if not struct:
        return False
    prefixes = ("entity:", "speaker:", "event:", "kw:", "date:", "month:", "year:")
    return any(any(a.startswith(p) for p in prefixes) for a in struct)


def rerank(
    query_record: Dict,
    candidates: List[Dict],
    memory_attrs: Dict[str, Set[str]],
    query_attrs: Set[str],
    method: str = "soft",
    alpha: float = 0.7,
    top_k: int = 10,
    hard_threshold: float = 0.05,
) -> List[Dict]:
    """
    Rerank candidates (each has memory_id, cosine_score).

    Returns list of dicts with memory_id, cosine_score, fca_score, final_score, rank.
    """
    if not candidates:
        return []

    is_temporal = int(query_record.get("category", 0)) == 2
    is_multihop = int(query_record.get("category", 0)) == 1

    cand_ids = [c["memory_id"] for c in candidates]
    cand_attrs = {mid: memory_attrs.get(mid, set()) for mid in cand_ids}

    cosine = {c["memory_id"]: float(c.get("cosine_score", 0.0)) for c in candidates}
    fca_raw = compute_fca_scores(query_attrs, cand_attrs, is_temporal=is_temporal)
    cosine_n = _minmax_norm(cosine)
    fca_n = _minmax_norm(fca_raw)

    scored: List[Tuple[str, float, float, float]] = []
    for mid in cand_ids:
        cs = cosine_n.get(mid, 0.0)
        fs = fca_n.get(mid, 0.0)
        final = alpha * cs + (1 - alpha) * fs
        scored.append((mid, cs, fs, final))

    # hard gate filter
    if method == "hard" and not is_multihop:
        filtered = []
        for mid, cs, fs, final in scored:
            mem_a = cand_attrs.get(mid, set())
            if not _has_structural_share(query_attrs, mem_a):
                continue
            if fs < hard_threshold and not structural_overlap(query_attrs, mem_a):
                continue
            # reject generic-only overlap
            shared = query_attrs & mem_a
            if shared and all(is_generic_attr(a) for a in shared):
                continue
            filtered.append((mid, cs, fs, final))
        scored = filtered if filtered else scored

    scored.sort(key=lambda x: x[3], reverse=True)

    if method == "diverse" or (method == "hard" and is_multihop):
        scored = _apply_diversity(scored, candidates, top_k, is_multihop=is_multihop)
    else:
        scored = scored[:top_k]

    id_to_cand = {c["memory_id"]: c for c in candidates}
    out = []
    for rank, (mid, cs, fs, final) in enumerate(scored[:top_k]):
        base = id_to_cand.get(mid, {})
        out.append({
            "memory_id": mid,
            "cosine_score": cosine.get(mid, 0.0),
            "fca_score": fca_raw.get(mid, 0.0),
            "fca_score_norm": fs,
            "cosine_score_norm": cs,
            "final_score": final,
            "rank": rank,
            "memory_index": base.get("memory_index"),
            "timestamp": base.get("timestamp", ""),
            "content_snippet": (base.get("content", "") or "")[:120],
        })
    return out


def _content_hash(content: str) -> str:
    return hashlib.md5((content or "").strip().lower().encode()).hexdigest()[:12]


def _apply_diversity(
    scored: List[Tuple[str, float, float, float]],
    candidates: List[Dict],
    top_k: int,
    is_multihop: bool = False,
) -> List[Tuple[str, float, float, float]]:
    id_to = {c["memory_id"]: c for c in candidates}
    selected: List[Tuple[str, float, float, float]] = []
    date_counts: Dict[str, int] = {}
    session_counts: Dict[str, int] = {}
    content_hashes: Set[str] = set()

    max_per_date = 4 if is_multihop else 3
    max_per_session = 5 if is_multihop else 4

    for item in scored:
        mid, cs, fs, final = item
        c = id_to.get(mid, {})
        date = c.get("date", "") or c.get("timestamp", "")[:20]
        session = c.get("session_id", "") or date
        ch = _content_hash(c.get("content", ""))

        if ch in content_hashes:
            continue
        if date_counts.get(date, 0) >= max_per_date:
            continue
        if session_counts.get(session, 0) >= max_per_session:
            continue

        selected.append(item)
        date_counts[date] = date_counts.get(date, 0) + 1
        session_counts[session] = session_counts.get(session, 0) + 1
        content_hashes.add(ch)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        seen = {s[0] for s in selected}
        for item in scored:
            if item[0] not in seen:
                selected.append(item)
                if len(selected) >= top_k:
                    break
    return selected
