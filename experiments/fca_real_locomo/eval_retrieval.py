"""Offline retrieval quality metrics for FCA reranking evaluation."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Set, Any, Optional

from experiments.fca_real_locomo.extract_attributes import (
    structural_overlap,
    is_generic_attr,
    _extract_dates,
)

DATE_RE = re.compile(
    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December),?\s+(\d{4})",
    re.I,
)
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


def _memory_text(mem: Dict) -> str:
    parts = [
        mem.get("content", ""),
        mem.get("context", ""),
        " ".join(mem.get("keywords", []) or []),
        " ".join(mem.get("tags", []) or []),
        mem.get("timestamp", ""),
    ]
    return " ".join(parts).lower()


def _answer_tokens(reference: str) -> List[str]:
    toks = re.findall(r"[a-zA-Z0-9]+", reference.lower())
    return [t for t in toks if len(t) > 1]


def answer_hit_at_k(reference: str, top_memories: List[Dict]) -> float:
    if not reference or not top_memories:
        return 0.0
    ref_lower = reference.lower().strip()
    toks = _answer_tokens(reference)
    for mem in top_memories:
        text = _memory_text(mem)
        if ref_lower and ref_lower in text:
            return 1.0
        if toks:
            hits = sum(1 for t in toks if t in text)
            if hits >= max(1, len(toks) * 0.6):
                return 1.0
    return 0.0


def temporal_hit_at_k(reference: str, top_memories: List[Dict]) -> float:
    if not reference or not top_memories:
        return 0.0
    ref_dates = _extract_dates(reference)
    ref_years = {a for a in ref_dates if a.startswith("year:")}
    ref_months = {a for a in ref_dates if a.startswith("month:")}
    ref_lower = reference.lower()

    for mem in top_memories:
        text = _memory_text(mem)
        mem_dates = _extract_dates(text)
        if ref_dates & mem_dates:
            return 1.0
        for y in ref_years:
            if y.split(":", 1)[1] in text:
                return 1.0
        for m in ref_months:
            parts = m.split(":", 1)[1].split("-")
            if len(parts) == 2 and parts[1].lower() in text:
                return 1.0
        for month in MONTHS:
            if month in ref_lower and month in text:
                return 1.0
    return 0.0


def person_event_match_rate(
    query_attrs: Set[str],
    top_memories: List[Dict],
    memory_attrs: Dict[str, Set[str]],
) -> float:
    if not top_memories:
        return 0.0
    q_entities = {a for a in query_attrs if a.startswith("entity:") or a.startswith("speaker:")}
    q_events = {a for a in query_attrs if a.startswith("event:")}
    if not q_entities and not q_events:
        q_entities = {a for a in query_attrs if a.startswith("kw:")}

    matched = 0
    for mem in top_memories:
        mid = mem.get("memory_id", "")
        ma = memory_attrs.get(mid, set())
        has_entity = bool(q_entities & ma) if q_entities else True
        has_event = bool(q_events & ma) if q_events else True
        if has_entity and has_event:
            matched += 1
    return matched / len(top_memories)


def redundancy_metrics(top_memories: List[Dict]) -> Dict[str, float]:
    if not top_memories:
        return {
            "unique_date_count": 0,
            "unique_session_count": 0,
            "duplicate_content_rate": 0.0,
            "redundancy_rate": 0.0,
        }
    dates = []
    sessions = []
    hashes = []
    for mem in top_memories:
        dates.append(mem.get("date", "") or mem.get("timestamp", "")[:15])
        sessions.append(mem.get("session_id", "") or mem.get("timestamp", ""))
        h = hashlib.md5((mem.get("content", "") or "").strip().lower().encode()).hexdigest()
        hashes.append(h)

    n = len(top_memories)
    unique_dates = len(set(dates))
    unique_sessions = len(set(sessions))
    dup_rate = 1.0 - len(set(hashes)) / n
    redundancy = 1.0 - unique_dates / n if n else 0.0
    return {
        "unique_date_count": float(unique_dates),
        "unique_session_count": float(unique_sessions),
        "duplicate_content_rate": dup_rate,
        "redundancy_rate": redundancy,
    }


def context_length(top_memories: List[Dict]) -> Dict[str, float]:
    total = sum(len(_memory_text(m)) for m in top_memories)
    return {
        "context_char_len": float(total),
        "memory_count": float(len(top_memories)),
    }


def topic_pollution_rate(
    query_attrs: Set[str],
    top_memories: List[Dict],
    memory_attrs: Dict[str, Set[str]],
) -> float:
    if not top_memories:
        return 0.0
    polluted = 0
    for mem in top_memories:
        mid = mem.get("memory_id", "")
        ma = memory_attrs.get(mid, set())
        shared = query_attrs & ma
        if not shared:
            polluted += 1
            continue
        struct = structural_overlap(query_attrs, ma)
        has_entity_event_time = any(
            a.startswith(p) for a in struct
            for p in ("entity:", "speaker:", "event:", "date:", "month:", "year:")
        )
        if not has_entity_event_time:
            # only generic/kw overlap
            if all(is_generic_attr(a) or a.startswith("kw:") for a in shared):
                polluted += 1
    return polluted / len(top_memories)


def evaluate_retrieval_for_question(
    question: Dict,
    top_memories: List[Dict],
    memory_attrs: Dict[str, Set[str]],
    query_attrs: Set[str],
    method: str,
) -> Dict[str, float]:
    cat = int(question.get("category", 0))
    ref = question.get("reference", "")

    metrics = {
        "answer_hit_at_10": answer_hit_at_k(ref, top_memories),
        "person_event_match_rate": person_event_match_rate(query_attrs, top_memories, memory_attrs),
        "topic_pollution_rate": topic_pollution_rate(query_attrs, top_memories, memory_attrs),
    }
    metrics.update(redundancy_metrics(top_memories))
    metrics.update(context_length(top_memories))

    if cat == 2:
        metrics["temporal_hit_at_10"] = temporal_hit_at_k(ref, top_memories)
    else:
        metrics["temporal_hit_at_10"] = float("nan")

    return metrics


def aggregate_metrics(rows: List[Dict[str, Any]], subset: str = "all") -> Dict[str, float]:
    if not rows:
        return {}
    keys = [
        "answer_hit_at_10", "temporal_hit_at_10", "person_event_match_rate",
        "topic_pollution_rate", "redundancy_rate", "duplicate_content_rate",
        "context_char_len", "unique_date_count",
    ]
    out = {"subset": subset, "count": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if k in r and r[k] == r[k]]  # skip nan
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out
