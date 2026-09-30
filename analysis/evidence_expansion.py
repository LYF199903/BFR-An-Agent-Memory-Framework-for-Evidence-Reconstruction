#!/usr/bin/env python3
"""Evidence expansion agent design and oracle expansion evaluation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from experiments.fca_real_locomo.eval_retrieval import answer_hit_at_k
from experiments.fca_real_locomo.load_amem_cache import MemoryRecord, _parse_date
from experiments.fca_real_locomo.stage_recall import find_answer_memory_ids

EXPANSION_ACTIONS = (
    "expand_retrieval_query",
    "retrieve_requirement_specific",
    "search_temporal_neighbors",
    "search_session_neighbors",
    "stop",
)

REWARD_COMPONENTS = (
    "newly_covered_requirements",
    "evidence_recall_improvement",
    "answer_correctness",
    "retrieval_cost",
    "redundant_retrieval",
)


@dataclass
class ExpansionState:
  question: str
  gold_answer: str
  selected_ids: Set[str]
  uncovered_requirements: List[Dict[str, Any]] = field(default_factory=list)
  expanded_ids: Set[str] = field(default_factory=set)
  actions_taken: List[str] = field(default_factory=list)


def _selected_ids(row: Dict[str, Any]) -> Set[str]:
    ids = row.get("selected_memory_ids")
    if ids:
        return set(ids)
    subset = row.get("selected_subset") or []
    return {s["memory_id"] for s in subset if s.get("memory_id")}


def _top10_ids(row: Dict[str, Any]) -> Set[str]:
    if row.get("context_memory_ids"):
        return set(row["context_memory_ids"])
    if row.get("top10_memory_ids"):
        return set(row["top10_memory_ids"])
    top10 = row.get("context_top10") or row.get("top10_memories") or []
    return {m["memory_id"] for m in top10 if m.get("memory_id")}


def _mem_dicts(mems: List[MemoryRecord]) -> List[Dict[str, Any]]:
    return [
        {
            "memory_id": m.memory_id,
            "content": m.content,
            "context": m.context,
            "keywords": m.keywords,
            "tags": m.tags,
            "timestamp": m.timestamp,
        }
        for m in mems
    ]


def _session_key(mem: MemoryRecord) -> str:
    ts = mem.session_id or mem.timestamp or ""
    if not ts:
        return ""
    return hashlib.md5(ts.strip().lower().encode()).hexdigest()[:12]


def _memory_lookup(mems: List[MemoryRecord]) -> Dict[str, MemoryRecord]:
    return {m.memory_id: m for m in mems}


def gold_memory_ids(reference: str, mem_dicts: List[Dict[str, Any]]) -> Set[str]:
    return set(find_answer_memory_ids(reference, mem_dicts))


def uncovered_requirements(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return requirements not covered by FCA-MS selection."""
    reqs = row.get("information_requirements") or []
    if not reqs:
        return []

    covered: Set[str] = set()
    for rc in row.get("requirement_coverage") or []:
        if rc.get("covered"):
            covered.add(rc.get("requirement_id", ""))

    missing = row.get("missing_requirements") or []
    if missing:
        missing_ids = {
            m.get("requirement_id") if isinstance(m, dict) else str(m)
            for m in missing
        }
        return [r for r in reqs if r.get("requirement_id") in missing_ids]

    return [r for r in reqs if r.get("requirement_id") not in covered]


def requirement_ids_covered_by_memories(
    req_ids: Set[str],
    memory_ids: Set[str],
    requirement_coverage: List[Dict[str, Any]],
) -> Set[str]:
    """Infer which requirements are supported by a memory set."""
    covered: Set[str] = set()
    for rc in requirement_coverage or []:
        rid = rc.get("requirement_id", "")
        mem_ids = set(rc.get("memory_ids") or [])
        if rid and mem_ids & memory_ids:
            covered.add(rid)
    return covered & req_ids if req_ids else covered


def evidence_recall(reference: str, memory_ids: Set[str], mem_dicts: List[Dict[str, Any]]) -> float:
    gold = gold_memory_ids(reference, mem_dicts)
    if not gold:
        return 0.0
    return len(gold & memory_ids) / len(gold)


def answer_hit(reference: str, memory_ids: Set[str], mem_dicts: List[Dict[str, Any]]) -> bool:
    selected = [m for m in mem_dicts if m["memory_id"] in memory_ids]
    return answer_hit_at_k(reference, selected) >= 1.0


def expand_oracle_full(selected_ids: Set[str], gold_ids: Set[str]) -> Set[str]:
    """Oracle: add all gold-bearing memories from full dialogue memory."""
    return selected_ids | gold_ids


def expand_session_neighbors(
    selected_ids: Set[str],
    memories: List[MemoryRecord],
    lookup: Dict[str, MemoryRecord],
) -> Set[str]:
    sessions = {_session_key(lookup[mid]) for mid in selected_ids if mid in lookup}
    sessions.discard("")
    added = {
        m.memory_id for m in memories
        if _session_key(m) in sessions and m.memory_id not in selected_ids
    }
    return selected_ids | added


def expand_temporal_neighbors(
    selected_ids: Set[str],
    memories: List[MemoryRecord],
    lookup: Dict[str, MemoryRecord],
) -> Set[str]:
    dates = {_parse_date(lookup[mid].timestamp) for mid in selected_ids if mid in lookup}
    dates.discard("")
    if not dates:
        return set(selected_ids)

    added: Set[str] = set()
    for mem in memories:
        if mem.memory_id in selected_ids:
            continue
        mem_date = _parse_date(mem.timestamp)
        if not mem_date:
            continue
        for date in dates:
            if mem_date == date or mem_date.split()[-1:] == date.split()[-1:]:
                added.add(mem.memory_id)
                break
    return selected_ids | added


def expand_requirement_specific(
    selected_ids: Set[str],
    uncovered: List[Dict[str, Any]],
    memories: List[MemoryRecord],
    limit_per_req: int = 3,
) -> Set[str]:
    """Retrieve memories matching uncovered requirement descriptions."""
    if not uncovered:
        return set(selected_ids)

    expanded = set(selected_ids)
    for req in uncovered:
        desc = (req.get("description") or "").lower()
        tokens = {t for t in re.findall(r"[a-z0-9]+", desc) if len(t) > 3}
        if not tokens:
            continue
        scored: List[tuple[int, str]] = []
        for mem in memories:
            if mem.memory_id in expanded:
                continue
            text = f"{mem.content} {mem.context}".lower()
            overlap = sum(1 for t in tokens if t in text)
            if overlap:
                scored.append((overlap, mem.memory_id))
        scored.sort(reverse=True)
        for _, mid in scored[:limit_per_req]:
            expanded.add(mid)
    return expanded


def evaluate_expansion(
    baseline_ids: Set[str],
    expanded_ids: Set[str],
    reference: str,
    mem_dicts: List[Dict[str, Any]],
    requirement_ids: Set[str],
    requirement_coverage: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score an expansion relative to baseline selection."""
    gold = gold_memory_ids(reference, mem_dicts)
    base_recall = evidence_recall(reference, baseline_ids, mem_dicts)
    new_recall = evidence_recall(reference, expanded_ids, mem_dicts)

    base_req_cov = requirement_ids_covered_by_memories(requirement_ids, baseline_ids, requirement_coverage)
    new_req_cov = requirement_ids_covered_by_memories(requirement_ids, expanded_ids, requirement_coverage)

    added = expanded_ids - baseline_ids
    redundant = len(added & baseline_ids)  # always 0 by construction
    retrieval_cost = len(added)

    return {
        "baseline_evidence_recall": base_recall,
        "expanded_evidence_recall": new_recall,
        "evidence_recall_gain": new_recall - base_recall,
        "baseline_requirements_covered": len(base_req_cov),
        "expanded_requirements_covered": len(new_req_cov),
        "newly_covered_requirements": len(new_req_cov - base_req_cov),
        "baseline_answer_hit": answer_hit(reference, baseline_ids, mem_dicts),
        "expanded_answer_hit": answer_hit(reference, expanded_ids, mem_dicts),
        "answer_recovered": (not answer_hit(reference, baseline_ids, mem_dicts)
                             and answer_hit(reference, expanded_ids, mem_dicts)),
        "gold_in_full_memory": bool(gold),
        "gold_ids": sorted(gold),
        "num_added_memories": retrieval_cost,
        "redundant_retrieval": redundant,
        "full_recall_achieved": new_recall >= 1.0 if gold else False,
    }


def run_oracle_expansion_eval(
    row: Dict[str, Any],
    memories: List[MemoryRecord],
) -> Dict[str, Any]:
    """Evaluate all expansion strategies on one insufficient_evidence case."""
    lookup = _memory_lookup(memories)
    mem_dicts = _mem_dicts(memories)
    reference = row.get("gold_answer") or ""
    baseline = _selected_ids(row)
    if not baseline:
        baseline = _top10_ids(row)

    gold = gold_memory_ids(reference, mem_dicts)
    uncovered = uncovered_requirements(row)
    req_ids = {r.get("requirement_id", "") for r in row.get("information_requirements") or []}
    req_cov = row.get("requirement_coverage") or []

    strategies = {
        "oracle_full_memory": expand_oracle_full(baseline, gold),
        "session_neighbors": expand_session_neighbors(baseline, memories, lookup),
        "temporal_neighbors": expand_temporal_neighbors(baseline, memories, lookup),
        "requirement_specific": expand_requirement_specific(baseline, uncovered, memories),
    }

    strategy_results = {
        name: evaluate_expansion(baseline, expanded, reference, mem_dicts, req_ids, req_cov)
        for name, expanded in strategies.items()
    }

    return {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "gold_answer": reference,
        "method": row.get("method"),
        "baseline_size": len(baseline),
        "uncovered_requirement_count": len(uncovered),
        "uncovered_requirements": uncovered,
        "gold_in_full_memory": bool(gold),
        "gold_in_top10": row.get("gold_in_top10"),
        "baseline_evidence_recall": row.get("evidence_recall"),
        "strategies": strategy_results,
        "best_strategy": max(
            strategy_results,
            key=lambda k: (
                strategy_results[k]["expanded_answer_hit"],
                strategy_results[k]["expanded_evidence_recall"],
                -strategy_results[k]["num_added_memories"],
            ),
        ),
    }
