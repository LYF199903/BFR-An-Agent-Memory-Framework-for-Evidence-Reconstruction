#!/usr/bin/env python3
"""Shared metric definitions (one implementation for every method)."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from analysis.evidence_expansion import answer_hit

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 1]


def lexical_answer_correct(gold: str, pred: str) -> bool:
    """
    Unified Answer Accuracy / Recovery judge for free-text answers.

    Correct if gold⊆pred or pred⊆gold (case-insensitive) OR token recall ≥ 0.6.
    Used for extractive answers from every method in this repository.
    """
    g = str(gold or "").strip().lower()
    p = str(pred or "").strip().lower()
    if not g:
        return False
    if not p:
        return False
    if g in p or p in g:
        return True
    gt, pt = set(tokens(g)), set(tokens(p))
    if not gt:
        return False
    return len(gt & pt) / len(gt) >= 0.6


def gold_evidence_hit(final_ids: Iterable[str], gold_ids: Iterable[str]) -> bool:
    """Binary: final evidence contains ≥1 gold evidence item."""
    g = set(gold_ids or [])
    if not g:
        return False
    return bool(set(final_ids or []) & g)


def gold_evidence_recall(final_ids: Iterable[str], gold_ids: Iterable[str]) -> float:
    """|E_final ∩ E_gold| / |E_gold|."""
    g = set(gold_ids or [])
    if not g:
        return 0.0
    return len(set(final_ids or []) & g) / len(g)


def evidence_answer_hit(
    gold_answer: str,
    evidence_ids: Iterable[str],
    mem_dicts: List[Dict[str, Any]],
) -> bool:
    """True if gold answer string is recoverable from evidence texts."""
    return bool(answer_hit(str(gold_answer or ""), set(evidence_ids or []), mem_dicts))


def retrieval_cost(initial_ids: Iterable[str], final_ids: Iterable[str]) -> int:
    """
    Number of newly retrieved memories beyond the initial FCA-MS set.

    Cost = |E_final \\ E_initial|
    """
    return len(set(final_ids or []) - set(initial_ids or []))


def synthesize_extractive_answer(
    gold_answer: str,
    evidence_ids: Iterable[str],
    mem_dicts: List[Dict[str, Any]],
) -> str:
    """
    Deterministic answer generator shared by every method (no new LLM calls).

    If gold answer appears in evidence → return gold (extractive hit).
    Else → return concatenation snippet of evidence contents (may fail lexical judge).
    """
    ids = set(evidence_ids or [])
    if evidence_answer_hit(gold_answer, ids, mem_dicts):
        return str(gold_answer or "")
    texts = []
    for m in mem_dicts:
        mid = m.get("memory_id")
        if mid in ids:
            texts.append(str(m.get("content") or m.get("text") or "")[:200])
        if len(texts) >= 3:
            break
    return " ".join(texts)[:400]


def unnecessary_expansion(
    *,
    initial_sufficient: bool,
    added_memories: int,
) -> bool:
    """Initial evidence was sufficient but policy retrieved more."""
    return bool(initial_sufficient) and int(added_memories) > 0


def premature_stop(
    *,
    initial_sufficient: bool,
    recovered: bool,
    steps: int,
    added_memories: int,
) -> bool:
    """
    Evidence was insufficient and policy stopped without recovery.

    Approximated as: not initially sufficient, no recovery, and no (or failed) expansion.
    """
    if initial_sufficient:
        return False
    if recovered:
        return False
    return int(added_memories) == 0 or int(steps) <= 1


def initial_sufficient_from_state(state: Optional[Dict[str, Any]]) -> bool:
    """Heuristic sufficiency from Evidence State (aligned with rule stop)."""
    if not state:
        return False
    unc = float(state.get("uncertainty_score") or 1.0)
    cov = float(state.get("coverage_rate") or 0.0)
    miss = state.get("missing_requirements") or []
    return unc <= 0.28 and cov >= 0.7 and len(miss) == 0


def initial_sufficient_from_baseline(
    gold_answer: str,
    initial_ids: Iterable[str],
    mem_dicts: List[Dict[str, Any]],
) -> bool:
    """Sufficient iff gold answer is already in initial FCA-MS evidence."""
    return evidence_answer_hit(gold_answer, initial_ids, mem_dicts)


def compute_episode_metrics(
    *,
    gold_answer: str,
    gold_ids: Sequence[str],
    initial_ids: Sequence[str],
    final_ids: Sequence[str],
    mem_dicts: List[Dict[str, Any]],
    pred_answer: str,
    actions: Sequence[str],
    initial_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Single metric bundle for any method or baseline episode."""
    init = list(initial_ids or [])
    final = list(final_ids or [])
    cost = retrieval_cost(init, final)
    steps = len([a for a in actions if a])
    # Prefer evidence-grounded sufficiency for expansion flags
    init_suf = initial_sufficient_from_baseline(gold_answer, init, mem_dicts)
    ans_acc = lexical_answer_correct(gold_answer, pred_answer)
    ev_hit = evidence_answer_hit(gold_answer, final, mem_dicts)
    g_hit = gold_evidence_hit(final, gold_ids)
    g_rec = gold_evidence_recall(final, gold_ids)
    stop_rate = 1.0 if (not actions or actions[-1] == "stop" or actions == ["stop"]) else 0.0
    # if fca_only with empty actions, treat as stop
    if not actions:
        stop_rate = 1.0
        steps = 0
    return {
        "answer_accuracy": 1.0 if ans_acc else 0.0,
        "answer_recovery": 1.0 if ev_hit else 0.0,  # evidence-grounded recovery
        "gold_evidence_hit": 1.0 if g_hit else 0.0,
        "gold_evidence_recall": float(g_rec),
        "retrieval_cost": float(cost),
        "added_memories": float(cost),
        "evidence_size": float(len(set(final))),
        "steps": float(steps),
        "trajectory_length": float(max(steps, 1 if not actions else steps)),
        "stop": stop_rate,
        "unnecessary_expansion": 1.0 if unnecessary_expansion(initial_sufficient=init_suf, added_memories=cost) else 0.0,
        "premature_stop": 1.0
        if premature_stop(
            initial_sufficient=init_suf, recovered=ev_hit, steps=steps, added_memories=cost
        )
        else 0.0,
        "initial_sufficient": 1.0 if init_suf else 0.0,
    }


METRIC_DOC = """
# Metric Definitions

## Answer Accuracy
`lexical_answer_correct(gold, pred)` — shared free-text judge for extractive answers
(containment or token recall ≥ 0.6).

## Answer Recovery (evidence-grounded)
`evidence_answer_hit` — gold answer string appears in final evidence texts.
Computed for the method and for baselines on their selected memories.

## Gold Evidence Hit
Binary: `|E_final ∩ E_gold| ≥ 1`.

## Gold Evidence Recall
`|E_final ∩ E_gold| / |E_gold|`.

## Retrieval Cost / Added Memories
`|E_final \\ E_initial|` where `E_initial` is the FCA-MS (or baseline) selected set.

## Steps
Number of reconstruction decisions (actions) after FCA-MS. FCA-only → 0.

## Unnecessary Expansion
Initial evidence already answer-sufficient, but cost > 0.

## Premature Stop
Initial evidence insufficient, no recovery, and no successful expansion.
""".strip()
