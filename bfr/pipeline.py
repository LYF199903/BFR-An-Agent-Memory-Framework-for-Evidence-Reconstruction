"""Two-stage BFR pipeline used in the paper.

Stage I  FCA-MS: compose a complementary seed set from information requirements.
Stage II BFR-Text / BFR-MV: budgeted completion from missing requirements.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from analysis.dynamic_cue_retrieval_env import DynamicCueResult
from evaluation.metrics import synthesize_extractive_answer
from experiments.fca_real_locomo.fca_ms_selector import (
    extract_memory_concept_attributes,
    fca_greedy_select,
)
from experiments.fca_real_locomo.load_amem_cache import MemoryRecord
from experiments.fca_real_locomo.question_analyzer import analyze_question
from models.memory_agent import MemoryAgent
from tools.iclr.make_env import make_env


def memories_from_dicts(items: Sequence[Dict[str, Any]]) -> List[MemoryRecord]:
    out: List[MemoryRecord] = []
    for i, raw in enumerate(items):
        out.append(
            MemoryRecord(
                memory_id=str(raw.get("memory_id") or f"m{i}"),
                dialogue_id=str(raw.get("dialogue_id") or "toy"),
                memory_index=int(raw.get("memory_index") or i),
                session_id=str(raw.get("session_id") or ""),
                timestamp=str(raw.get("timestamp") or raw.get("date") or ""),
                date=str(raw.get("date") or ""),
                speaker=str(raw.get("speaker") or ""),
                content=str(raw.get("content") or ""),
                context=str(raw.get("context") or ""),
                keywords=list(raw.get("keywords") or []),
                tags=list(raw.get("tags") or []),
            )
        )
    return out


def rule_requirements(question: str) -> List[Dict[str, Any]]:
    """LLM-free requirement sketch from the rule-based question analyzer."""
    info = analyze_question(question)
    reqs: List[Dict[str, Any]] = [
        {
            "requirement_id": "R1",
            "description": question,
            "type": "answer_bearing_fact",
        }
    ]
    persons = info.get("persons") or []
    if persons:
        reqs.append(
            {
                "requirement_id": "R2",
                "description": f"entity anchor for {persons[0]}",
                "type": "entity_anchor",
            }
        )
    if info.get("temporal") or info.get("question_type") in ("temporal",) or "how long" in question.lower() or "how many days" in question.lower():
        reqs.append(
            {
                "requirement_id": "R3",
                "description": "event time / temporal bridge needed to answer the question",
                "type": "temporal_bridge",
            }
        )
    return reqs


def _concept_mems(memories: Sequence[MemoryRecord]) -> List[Dict[str, Any]]:
    speakers = sorted({m.speaker for m in memories if m.speaker})
    return [extract_memory_concept_attributes(m, speakers) for m in memories]


def run_bfr(
    question: str,
    memories: Sequence[Any],
    *,
    requirements: Optional[List[Dict[str, Any]]] = None,
    method: str = "BFR-MV",
    question_id: str = "q0",
    gold_answer: str = "",
    max_set_size: int = 6,
    budget: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Stage I (FCA-MS) then Stage II (BFR completion).

    ``memories`` may be ``MemoryRecord`` objects or plain dicts.

    ``gold_answer`` is for offline diagnostics. The returned ``answer`` is
    an extractive proxy and can echo that reference when the selected
    evidence contains it; it is not an independent model prediction.
    """
    recs: List[MemoryRecord]
    if memories and isinstance(memories[0], MemoryRecord):
        recs = list(memories)
    else:
        recs = memories_from_dicts(list(memories))

    reqs = list(requirements or rule_requirements(question))
    concept = _concept_mems(recs)
    seed_ids, fca_meta = fca_greedy_select(reqs, concept, max_set_size=max_set_size)

    baseline_row: Dict[str, Any] = {
        "question_id": question_id,
        "dialogue_id": recs[0].dialogue_id if recs else "toy",
        "question": question,
        "gold_answer": gold_answer,
        "information_requirements": reqs,
        "baselines": {"fca_ms": {"selected_memories": list(seed_ids)}},
    }
    row = MemoryAgent.build_env_row_from_baseline(baseline_row)

    result: Optional[DynamicCueResult] = None
    final_ids = list(seed_ids)
    if method not in ("FCA-MS", "fca_ms"):
        env = make_env(method, budget)
        result = env.run(baseline_row, recs)
        final_ids = list(result.final_evidence_ids or seed_ids)

    mem_dicts = [
        {
            "memory_id": m.memory_id,
            "content": m.content,
            "context": m.context,
            "keywords": m.keywords,
            "tags": m.tags,
            "timestamp": m.timestamp,
        }
        for m in recs
    ]
    answer = synthesize_extractive_answer(gold_answer, final_ids, mem_dicts)
    return {
        "question_id": question_id,
        "question": question,
        "method": method,
        "information_requirements": reqs,
        "stage1_seed_ids": list(seed_ids),
        "final_evidence_ids": list(final_ids),
        "answer": answer,
        "fca_meta": {
            "requirement_coverage_rate": fca_meta.get("requirement_coverage_rate"),
            "covered_requirements": fca_meta.get("covered_requirements"),
            "steps": fca_meta.get("steps"),
        },
        "stage2": None if result is None else result.to_dict(),
        "env_row": row,
    }
