"""FCA-guided memory set selection (FCA-MS v0)."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from experiments.fca_real_locomo.extract_attributes import (
    _add_kw_attrs,
    _extract_dates,
    _extract_entities,
    _extract_events,
    extract_memory_attributes,
    extract_query_attributes,
)
from experiments.fca_real_locomo.encode_query import encode_query as _encode_query

ATTR_WEIGHTS = {
    "entity": 1.0,
    "speaker": 0.8,
    "time": 1.2,
    "date": 1.2,
    "month": 1.2,
    "year": 1.2,
    "event": 1.1,
    "action": 1.1,
    "object": 1.3,
    "answer_candidate": 1.3,
    "keyword": 0.5,
    "kw": 0.5,
    "ctx": 0.5,
    "tag": 0.5,
    "session": 0.4,
}

PREFIX_TO_BUCKET = {
    "entity": "entity",
    "speaker": "speaker",
    "date": "time",
    "month": "time",
    "year": "time",
    "event": "event",
    "kw": "keyword",
    "ctx": "keyword",
    "tag": "keyword",
}


def _get_mem_field(mem: Any, key: str, default: Any = "") -> Any:
    if isinstance(mem, dict):
        return mem.get(key, default)
    return getattr(mem, key, default)


def _mem_text(mem: Any) -> str:
    content = _get_mem_field(mem, "content", "")
    context = _get_mem_field(mem, "context", "")
    return " ".join(filter(None, [content, context])).strip()


def _flat_attrs_to_buckets(flat: Set[str]) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {
        "entity": [], "speaker": [], "time": [], "event": [], "action": [],
        "object": [], "location": [], "keyword": [], "tag": [], "session": [],
        "answer_candidate": [],
    }
    for attr in flat:
        if ":" not in attr:
            continue
        prefix, val = attr.split(":", 1)
        bucket = PREFIX_TO_BUCKET.get(prefix, prefix)
        if bucket not in buckets:
            buckets[bucket] = []
        if val and val not in buckets[bucket]:
            buckets[bucket].append(val)
    return buckets


def extract_memory_concept_attributes(
    memory: Any,
    speakers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    flat = extract_memory_attributes(memory, speakers)
    buckets = _flat_attrs_to_buckets(flat)
    text = _mem_text(memory)
    # action from events
    for ev in buckets.get("event", []):
        if ev not in buckets["action"]:
            buckets["action"].append(ev)
    # answer_candidate: short quoted spans / dates in text
    for m in re.finditer(r'"([^"]{3,80})"', text):
        buckets["answer_candidate"].append(m.group(1)[:80])
    buckets["answer_candidate"].extend(buckets.get("time", [])[:3])
    session = _get_mem_field(memory, "session_id", "")
    if session:
        buckets["session"] = [str(session)]
    return {
        "memory_id": _get_mem_field(memory, "memory_id", ""),
        "attributes": buckets,
        "flat_attributes": sorted(flat),
        "text": text,
    }


def extract_requirement_attributes(requirement: Dict[str, Any]) -> Dict[str, List[str]]:
    expected = requirement.get("expected_attributes") or {}
    if expected:
        return {k: list(v) for k, v in expected.items()}
    desc = requirement.get("description", "")
    flat = extract_query_attributes(desc, category=1, known_speakers=None)
    buckets = _flat_attrs_to_buckets(flat)
    for tok in re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", desc):
        if tok[0].isupper():
            buckets.setdefault("entity", [])
            if tok not in buckets["entity"]:
                buckets["entity"].append(tok)
    return buckets


def _bucket_weight(bucket: str) -> float:
    return ATTR_WEIGHTS.get(bucket, 0.6)


def attribute_overlap(req_attrs: Dict[str, List[str]], mem_attrs: Dict[str, List[str]]) -> float:
    num = 0.0
    den = 0.0
    for bucket, req_vals in req_attrs.items():
        if not req_vals:
            continue
        w = _bucket_weight(bucket)
        mem_vals = {v.lower() for v in mem_attrs.get(bucket, [])}
        for rv in req_vals:
            rv_l = rv.lower()
            den += w
            if any(rv_l in mv or mv in rv_l for mv in mem_vals):
                num += w
            elif bucket == "keyword" and any(rv_l in mv for mv in mem_vals):
                num += w * 0.5
    return num / den if den else 0.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rare_attribute_bonus(
    req_attrs: Dict[str, List[str]],
    mem_attrs: Dict[str, List[str]],
    pool_attrs: List[Dict[str, List[str]]],
) -> float:
    bonus = 0.0
    rare_buckets = ("time", "event", "object", "answer_candidate")
    for bucket in rare_buckets:
        mem_vals = mem_attrs.get(bucket, [])
        if not mem_vals:
            continue
        req_vals = req_attrs.get(bucket, [])
        if not req_vals and bucket not in ("object", "answer_candidate"):
            continue
        freq = 0
        for pa in pool_attrs:
            if pa.get(bucket):
                freq += 1
        rarity = 1.0 - min(freq / max(len(pool_attrs), 1), 1.0)
        if rarity > 0.5:
            bonus += 0.15 * _bucket_weight(bucket)
    return min(bonus, 0.3)


def coverage_score(
    requirement: Dict[str, Any],
    memory: Dict[str, Any],
    req_emb: Optional[np.ndarray] = None,
    mem_emb: Optional[np.ndarray] = None,
    pool_attr_list: Optional[List[Dict[str, List[str]]]] = None,
) -> float:
    req_attrs = extract_requirement_attributes(requirement)
    mem_attrs = memory.get("attributes") or {}
    attr_ov = attribute_overlap(req_attrs, mem_attrs)
    dense_sim = 0.0
    if req_emb is not None and mem_emb is not None:
        dense_sim = max(_cosine(req_emb, mem_emb), 0.0)
    rab = rare_attribute_bonus(req_attrs, mem_attrs, pool_attr_list or [])
    return 0.45 * attr_ov + 0.35 * dense_sim + 0.20 * rab


def build_coverage_matrix(
    requirements: List[Dict[str, Any]],
    candidate_memories: List[Dict[str, Any]],
    embed_model: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    pool_attr_list = [m.get("attributes", {}) for m in candidate_memories]
    req_embs: Dict[str, np.ndarray] = {}
    mem_embs: Dict[str, np.ndarray] = {}
    for req in requirements:
        rid = req.get("requirement_id") or req.get("id", "")
        req_embs[rid] = _encode_query(req.get("description", ""), embed_model)
    for mem in candidate_memories:
        mid = mem["memory_id"]
        mem_embs[mid] = _encode_query(mem.get("text", ""), embed_model)

    matrix: Dict[str, Dict[str, float]] = {}
    for req in requirements:
        rid = req.get("requirement_id") or req.get("id", "")
        matrix[rid] = {}
        for mem in candidate_memories:
            mid = mem["memory_id"]
            matrix[rid][mid] = coverage_score(
                req, mem, req_embs.get(rid), mem_embs.get(mid), pool_attr_list,
            )
    return {"matrix": matrix, "requirements": requirements, "memory_ids": [m["memory_id"] for m in candidate_memories]}


def _redundancy(m: Dict[str, Any], selected: List[Dict[str, Any]], mem_embs: Dict[str, np.ndarray]) -> float:
    if not selected:
        return 0.0
    mid = m["memory_id"]
    max_cos = 0.0
    max_attr = 0.0
    m_attrs = m.get("attributes", {})
    m_flat = set(m.get("flat_attributes") or [])
    me = mem_embs.get(mid)
    for s in selected:
        se = mem_embs.get(s["memory_id"])
        if me is not None and se is not None:
            max_cos = max(max_cos, _cosine(me, se))
        s_flat = set(s.get("flat_attributes") or [])
        if m_flat and s_flat:
            max_attr = max(max_attr, len(m_flat & s_flat) / max(len(m_flat | s_flat), 1))
    return 0.6 * max_cos + 0.4 * max_attr


def _new_requirement_coverage(
    m: Dict[str, Any],
    requirements: List[Dict[str, Any]],
    matrix: Dict[str, Dict[str, float]],
    covered: Set[str],
) -> float:
    gain = 0.0
    mid = m["memory_id"]
    for req in requirements:
        rid = req.get("requirement_id") or req.get("id", "")
        if rid in covered:
            continue
        sc = matrix.get(rid, {}).get(mid, 0.0)
        if sc >= 0.25:
            gain += sc
    return gain


def _new_attribute_coverage(m: Dict[str, Any], selected: List[Dict[str, Any]]) -> float:
    if not selected:
        return 0.1
    covered: Set[str] = set()
    for s in selected:
        for bucket, vals in (s.get("attributes") or {}).items():
            for v in vals:
                covered.add(f"{bucket}:{v.lower()}")
    gain = 0.0
    for bucket, vals in (m.get("attributes") or {}).items():
        for v in vals:
            key = f"{bucket}:{v.lower()}"
            if key not in covered:
                gain += 0.05 * _bucket_weight(bucket)
    return gain


def _answer_specificity(m: Dict[str, Any]) -> float:
    attrs = m.get("attributes") or {}
    score = 0.0
    if attrs.get("answer_candidate"):
        score += 0.25
    if attrs.get("time"):
        score += 0.15
    if attrs.get("event"):
        score += 0.10
    vague_markers = ("discussion", "conversation", "support", "positive")
    text_l = (m.get("text") or "").lower()
    if any(vm in text_l for vm in vague_markers) and not attrs.get("answer_candidate"):
        score -= 0.10
    return score


def fca_greedy_select(
    requirements: List[Dict[str, Any]],
    candidate_memories: List[Dict[str, Any]],
    max_set_size: int = 6,
    min_set_size: int = 1,
    threshold: float = 0.05,
) -> Tuple[List[str], Dict[str, Any]]:
    if not candidate_memories:
        return [], {"matrix": {}, "steps": []}
    cov = build_coverage_matrix(requirements, candidate_memories)
    matrix = cov["matrix"]
    mem_by_id = {m["memory_id"]: m for m in candidate_memories}
    mem_embs = {m["memory_id"]: _encode_query(m.get("text", ""), "all-MiniLM-L6-v2") for m in candidate_memories}

    selected: List[Dict[str, Any]] = []
    selected_ids: List[str] = []
    covered_reqs: Set[str] = set()
    steps: List[Dict[str, Any]] = []

    while len(selected) < max_set_size:
        best_mid = None
        best_gain = -1e9
        for m in candidate_memories:
            mid = m["memory_id"]
            if mid in selected_ids:
                continue
            marginal = (
                _new_requirement_coverage(m, requirements, matrix, covered_reqs)
                + _new_attribute_coverage(m, selected)
                + _answer_specificity(m)
                - _redundancy(m, selected, mem_embs)
                - 0.02 * len(selected)
            )
            if marginal > best_gain:
                best_gain = marginal
                best_mid = mid
        if best_mid is None or best_gain < threshold:
            break
        m = mem_by_id[best_mid]
        selected.append(m)
        selected_ids.append(best_mid)
        for req in requirements:
            rid = req.get("requirement_id") or req.get("id", "")
            if matrix.get(rid, {}).get(best_mid, 0) >= 0.25:
                covered_reqs.add(rid)
        steps.append({"memory_id": best_mid, "marginal_gain": best_gain})

    if len(selected_ids) < min_set_size and candidate_memories:
        top = max(
            candidate_memories,
            key=lambda m: sum(matrix.get(r.get("requirement_id") or r.get("id", ""), {}).get(m["memory_id"], 0) for r in requirements),
        )
        if top["memory_id"] not in selected_ids:
            selected_ids = [top["memory_id"]]

    req_cov = len(covered_reqs) / max(len(requirements), 1)
    redundancy = 0.0
    if len(selected) > 1:
        pairs = 0
        total = 0.0
        for i, a in enumerate(selected):
            for b in selected[i + 1:]:
                total += _redundancy(b, [a], mem_embs)
                pairs += 1
        redundancy = total / pairs if pairs else 0.0

    meta = {
        "matrix": matrix,
        "steps": steps,
        "requirement_coverage_rate": req_cov,
        "redundancy_score": redundancy,
        "covered_requirements": sorted(covered_reqs),
    }
    return selected_ids, meta


def fca_prefilter_candidates(
    requirements: List[Dict[str, Any]],
    candidate_memories: List[Dict[str, Any]],
    keep_k: int = 20,
) -> List[str]:
    if len(candidate_memories) <= keep_k:
        return [m["memory_id"] for m in candidate_memories]
    cov = build_coverage_matrix(requirements, candidate_memories)
    matrix = cov["matrix"]
    scores: Dict[str, float] = {}
    for m in candidate_memories:
        mid = m["memory_id"]
        scores[mid] = sum(matrix.get(r.get("requirement_id") or r.get("id", ""), {}).get(mid, 0) for r in requirements)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    kept: List[str] = []
    kept_mems: List[Dict[str, Any]] = []
    mem_by_id = {m["memory_id"]: m for m in candidate_memories}
    for mid, _ in ranked:
        if len(kept) >= keep_k:
            break
        m = mem_by_id[mid]
        if kept_mems and _redundancy(m, kept_mems, {x["memory_id"]: _encode_query(x.get("text", ""), "all-MiniLM-L6-v2") for x in candidate_memories}) > 0.85:
            continue
        kept.append(mid)
        kept_mems.append(m)
    if len(kept) < keep_k:
        for mid, _ in ranked:
            if mid not in kept:
                kept.append(mid)
            if len(kept) >= keep_k:
                break
    return kept


def build_context(
    selected_ids: List[str],
    candidate_pool: List[Dict[str, Any]],
    mode: str = "selected_plus_fill",
    top_k: int = 10,
) -> List[str]:
    pool_ids = [m["memory_id"] for m in candidate_pool]
    id_set = set(pool_ids)
    selected = [mid for mid in selected_ids if mid in id_set]
    if mode == "selected_only":
        return selected[:top_k]
    out = list(selected)
    for mid in pool_ids:
        if mid not in out:
            out.append(mid)
        if len(out) >= top_k:
            break
    return out[:top_k]


def build_requirement_only_prompt(question: str) -> str:
    return f"""Decompose this question into information requirements needed to answer it.

Question:
{question}

Output JSON only:
{{
  "information_requirements": [
    {{
      "requirement_id": "R1",
      "description": "...",
      "type": "entity_anchor | answer_bearing_fact | entity_bridge | temporal_bridge | speaker_bridge | event_bridge | relation_object | list_item | coreference | event_context | temporal_context"
    }}
  ]
}}"""


def build_fca_seed_refine_prompt(
    question: str,
    seed_subset: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    max_changes: int = 2,
) -> str:
    from experiments.fca_real_locomo.set_sufficiency_prompts import format_memories_for_prompt, allowed_memory_ids_block, STRICT_MEMORY_ID_RULES
    seed_block = format_memories_for_prompt(seed_subset)
    cand_block = format_memories_for_prompt(candidates)
    allowed = allowed_memory_ids_block(candidates)
    return f"""You refine a memory set for answering a question.

Question:
{question}

FCA seed set (greedy selected):
{seed_block}

Candidate memories:
{cand_block}

You may add, remove, or replace at most {max_changes} memories from the seed set.
Prefer keeping the seed unless a candidate clearly fills a missing requirement.

{STRICT_MEMORY_ID_RULES}

{allowed}

Output JSON only:
{{
  "selected_subset": [
    {{"memory_id": "<exact id>", "supports": ["R1"], "reason": "..."}}
  ],
  "changes": ["added m3", "removed m1"]
}}"""
