"""Stage-wise answer recall diagnostics for person-fact retrieval pipeline."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from experiments.fca_real_locomo.eval_retrieval import answer_hit_at_k


def split_reference_parts(reference: str) -> List[str]:
    """Split compound references like 'beach, mountains, forest' into parts."""
    if not reference:
        return []
    ref = reference.strip()
    if not ref:
        return []
    parts = re.split(r"\s*[,;/]\s*|\s+and\s+", ref, flags=re.I)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [ref]


def memory_matches_reference(reference: str, mem: Dict[str, Any]) -> bool:
    """Whether a single memory contains the reference (same logic as answer_hit_at_k)."""
    return answer_hit_at_k(reference, [mem]) >= 1.0


def find_answer_memory_ids(
    reference: str,
    memories: List[Dict[str, Any]],
    limit: int = 20,
) -> List[str]:
    ids: List[str] = []
    for mem in memories:
        if memory_matches_reference(reference, mem):
            mid = mem.get("memory_id")
            if mid and mid not in ids:
                ids.append(mid)
            if len(ids) >= limit:
                break
    return ids


def answer_hit_any(reference: str, memories: List[Dict[str, Any]]) -> bool:
    parts = split_reference_parts(reference)
    if len(parts) <= 1:
        return answer_hit_at_k(reference, memories) >= 1.0
    return any(answer_hit_at_k(part, memories) >= 1.0 for part in parts)


def answer_hit_all(reference: str, memories: List[Dict[str, Any]]) -> bool:
    parts = split_reference_parts(reference)
    if len(parts) <= 1:
        return answer_hit_at_k(reference, memories) >= 1.0
    return all(answer_hit_at_k(part, memories) >= 1.0 for part in parts)


def first_answer_rank(
    reference: str,
    ranked_memories: List[Dict[str, Any]],
) -> Optional[int]:
    """1-based rank of first memory containing reference; None if absent."""
    for i, mem in enumerate(ranked_memories, start=1):
        if memory_matches_reference(reference, mem):
            return i
    return None


def count_answer_memories(
    reference: str,
    memories: List[Dict[str, Any]],
) -> int:
    return sum(1 for mem in memories if memory_matches_reference(reference, mem))


def classify_loss_stage(stage: Dict[str, Any]) -> str:
    """Assign A–E loss bucket based on stage recall flags (any-hit)."""
    full = stage.get("full_memory_answer_hit", False)
    person = stage.get("person_pool_answer_hit", False)
    cand = stage.get("candidate_pool_answer_hit", False)
    verify = stage.get("verify_k_answer_hit", False)
    top10 = stage.get("final_top10_answer_hit", False)

    if not full:
        return "A"
    if not person:
        return "B"
    if not cand:
        return "C"
    if not verify:
        return "D"
    if not top10:
        return "E"
    return "none"


def compute_stage_recall(
    reference: str,
    *,
    all_memories: List[Dict[str, Any]],
    dense_pool: List[Dict[str, Any]],
    person_pool_memories: List[Dict[str, Any]],
    bm25_pool: List[Dict[str, Any]],
    expansion_pool: List[Dict[str, Any]],
    candidate_pool: List[Dict[str, Any]],
    hybrid_ranked: List[Dict[str, Any]],
    verify_k_list: List[Dict[str, Any]],
    final_top10: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build per-question stage_recall diagnostic block."""
    full_ids = find_answer_memory_ids(reference, all_memories, limit=20)

    stage: Dict[str, Any] = {
        "reference_parts": split_reference_parts(reference),
        "full_memory_answer_hit": answer_hit_any(reference, all_memories),
        "full_memory_answer_all_hit": answer_hit_all(reference, all_memories),
        "full_memory_answer_ids": full_ids,
        "dense_pool_answer_hit": answer_hit_any(reference, dense_pool),
        "dense_pool_answer_all_hit": answer_hit_all(reference, dense_pool),
        "dense_pool_answer_rank": first_answer_rank(reference, dense_pool),
        "person_pool_answer_hit": answer_hit_any(reference, person_pool_memories),
        "person_pool_answer_all_hit": answer_hit_all(reference, person_pool_memories),
        "person_pool_answer_count": count_answer_memories(reference, person_pool_memories),
        "bm25_pool_answer_hit": answer_hit_any(reference, bm25_pool),
        "bm25_pool_answer_all_hit": answer_hit_all(reference, bm25_pool),
        "expansion_pool_answer_hit": answer_hit_any(reference, expansion_pool),
        "expansion_pool_answer_all_hit": answer_hit_all(reference, expansion_pool),
        "candidate_pool_answer_hit": answer_hit_any(reference, candidate_pool),
        "candidate_pool_answer_all_hit": answer_hit_all(reference, candidate_pool),
        "candidate_pool_answer_rank_local": first_answer_rank(reference, hybrid_ranked),
        "verify_k_answer_hit": answer_hit_any(reference, verify_k_list),
        "verify_k_answer_all_hit": answer_hit_all(reference, verify_k_list),
        "final_top10_answer_hit": answer_hit_any(reference, final_top10),
        "final_top10_answer_all_hit": answer_hit_all(reference, final_top10),
        "loss_stage": "none",
    }
    stage["loss_stage"] = classify_loss_stage(stage)
    return stage


def aggregate_stage_recall(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate stage coverage and loss breakdown across questions."""
    stages = [r.get("stage_recall") for r in rows if r.get("stage_recall")]
    n = len(stages)
    if n == 0:
        return {}

    def _any_rate(any_key: str) -> float:
        return sum(1 for s in stages if s.get(any_key)) / n

    def _all_rate(all_key: str) -> float:
        return sum(1 for s in stages if s.get(all_key)) / n

    def _coverage(any_key: str, all_key: str) -> Dict[str, float]:
        return {
            "any_hit_rate": _any_rate(any_key),
            "all_hit_rate": _all_rate(all_key),
        }

    loss_counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "none": 0}
    for s in stages:
        bucket = s.get("loss_stage", "none")
        loss_counts[bucket] = loss_counts.get(bucket, 0) + 1

    return {
        "n_questions": n,
        "stage_coverage": {
            "full_memory": _coverage("full_memory_answer_hit", "full_memory_answer_all_hit"),
            "dense_top_k": _coverage("dense_pool_answer_hit", "dense_pool_answer_all_hit"),
            "person_pool": _coverage("person_pool_answer_hit", "person_pool_answer_all_hit"),
            "bm25_pool": _coverage("bm25_pool_answer_hit", "bm25_pool_answer_all_hit"),
            "expansion_pool": _coverage("expansion_pool_answer_hit", "expansion_pool_answer_all_hit"),
            "candidate_pool": _coverage("candidate_pool_answer_hit", "candidate_pool_answer_all_hit"),
            "verify_k": _coverage("verify_k_answer_hit", "verify_k_answer_all_hit"),
            "final_top10": _coverage("final_top10_answer_hit", "final_top10_answer_all_hit"),
        },
        "loss_breakdown": {
            k: {"count": v, "rate": v / n}
            for k, v in loss_counts.items()
        },
        "loss_breakdown_labels": {
            "A": "full memory 中没有答案",
            "B": "full memory 有答案，但 person pool 没有",
            "C": "person pool 有答案，但 candidate pool 没有",
            "D": "candidate pool 有答案，但 verify_k 没有",
            "E": "verify_k 有答案，但 final top10 没有",
            "none": "无丢失（top10 已命中）",
        },
    }
