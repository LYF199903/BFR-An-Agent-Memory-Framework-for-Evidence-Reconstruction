#!/usr/bin/env python3
"""Baseline adapters over the shared flat memory store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

# Reuse only; do not change algorithm defaults here.


ONLINE_FORBIDDEN = (
    "gold_answer",
    "gold_evidence",
    "gold_memory_ids",
    "gold_ids",
    "has_answer",
    "answer_session_ids",
    "human_sufficiency",
    "human_gap",
)


def assert_online_clean(payload: Dict[str, Any], where: str) -> None:
    for k in ONLINE_FORBIDDEN:
        if k in payload:
            raise ValueError(f"gold/offline field {k!r} leaked into {where}")


@dataclass
class MemoryBackend:
    retriever: Any

    def search(
        self,
        query: str,
        view: str = "text",
        filters: Optional[Dict[str, Any]] = None,
        budget: Optional[Dict[str, Any]] = None,
        *,
        exclude: Optional[Set[str]] = None,
        action: Optional[Any] = None,
    ) -> Dict[str, Any]:
        filters = filters or {}
        budget = budget or {}
        assert_online_clean({"query": query, **filters}, "MemoryBackend.search")
        top_k = int(budget.get("top_k") or 8)
        exclude = exclude or set()
        if action is not None and hasattr(self.retriever, "execute_action"):
            ids = self.retriever.execute_action(action, exclude=exclude)
        elif view in ("hybrid", "text_time") and hasattr(self.retriever, "search_hybrid"):
            ids = self.retriever.search_hybrid(query, top_k=top_k, exclude=exclude)
        elif view == "entity" and hasattr(self.retriever, "search_entity"):
            ids = self.retriever.search_entity(query, top_k=top_k, exclude=exclude)
        elif view in ("time", "temporal") and hasattr(self.retriever, "search_temporal"):
            ids = self.retriever.search_temporal(query, top_k=top_k, exclude=exclude)
        elif view == "session" and hasattr(self.retriever, "search_session"):
            ids = self.retriever.search_session(query, top_k=top_k, exclude=exclude)
        else:
            ids = self.retriever.search_text(query, top_k=top_k, exclude=exclude)
        return {"candidates": list(ids), "actual_cost": {"retrieval_calls": 1, "returned": len(ids)}}


@dataclass
class StateBuilder:
    builder: Any

    def build(
        self,
        question: str,
        visible_evidence: Sequence[str],
        history: Sequence[str],
        budget: Optional[Dict[str, Any]] = None,
        *,
        row: Optional[Dict[str, Any]] = None,
        memories: Optional[list] = None,
    ) -> Dict[str, Any]:
        budget = budget or {}
        assert_online_clean({"question": question}, "StateBuilder.build")
        if row is None or memories is None:
            return {"predicted_state": {"question": question, "evidence_ids": list(visible_evidence)}}
        state = self.builder.rebuild(
            row,
            memories,
            set(visible_evidence),
            retrieval_history=list(history),
            step=int(budget.get("step") or 0),
        )
        d = state.to_dict() if hasattr(state, "to_dict") else {"evidence_ids": list(visible_evidence)}
        for k in ONLINE_FORBIDDEN:
            d.pop(k, None)
        return {"predicted_state": d}


@dataclass
class Controller:
    kind: str = "dynamic_cue"
    env: Any = None

    def decide(
        self,
        observation: Dict[str, Any],
        allowed_actions: Sequence[str],
        budget: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        assert_online_clean(observation, "Controller.decide")
        if self.kind == "stop":
            return {"action": "stop"}
        return {"action": allowed_actions[0] if allowed_actions else "stop"}


@dataclass
class EvidenceRouter:
    router: Any

    def select(
        self,
        candidates: Sequence[str],
        visible_state: Dict[str, Any],
        budget: Optional[Dict[str, Any]] = None,
        *,
        lookup: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        budget = budget or {}
        assert_online_clean(visible_state, "EvidenceRouter.select")
        if lookup is None:
            return list(candidates)[: int(budget.get("keep_top") or 5)]
        out = self.router.route(
            candidates,
            lookup=lookup,
            question=str(visible_state.get("question") or ""),
            missing_requirements=list(visible_state.get("missing_requirements") or []),
            current_evidence_ids=list(visible_state.get("evidence_ids") or []),
            keep_top=int(budget.get("keep_top") or 5),
        )
        return list(out.get("kept_ids") or [])


@dataclass
class Answerer:
    def answer(self, question: str, evidence_or_native_representation: Any) -> str:
        assert_online_clean({"question": question}, "Answerer.answer")
        from evaluation.metrics import synthesize_extractive_answer

        gold_for_extractive = ""  # never pass gold; extractive falls back to snippets
        ids = []
        mem_dicts = []
        if isinstance(evidence_or_native_representation, dict):
            ids = list(evidence_or_native_representation.get("evidence_ids") or [])
            mem_dicts = list(evidence_or_native_representation.get("mem_dicts") or [])
        return synthesize_extractive_answer(gold_for_extractive, ids, mem_dicts)


@dataclass
class Evaluator:
    def score(self, question_id: str, prediction: Dict[str, Any], offline_labels: Dict[str, Any]) -> Dict[str, Any]:
        from evaluation.metrics import gold_evidence_hit, gold_evidence_recall, lexical_answer_correct

        gold_ids = offline_labels.get("gold_memory_ids") or []
        gold_answer = offline_labels.get("gold_answer") or ""
        pred_ids = prediction.get("selected_source_ids") or []
        pred_ans = prediction.get("answer") or ""
        hit = gold_evidence_hit(pred_ids, gold_ids) if gold_ids else None
        rec = gold_evidence_recall(pred_ids, gold_ids) if gold_ids else None
        lex = lexical_answer_correct(gold_answer, pred_ans) if gold_answer else None
        return {
            "question_id": question_id,
            "gold_hit": hit,
            "gold_recall": rec,
            "lexical": lex,
        }


METHOD_ADAPTERS = {
    "B00": {"retrieval": "single_view", "control": "question_only_fixed_queries", "backend": "shared_flat"},
    "B01": {"retrieval": "multiview", "control": "question_only_rerank", "backend": "multiview"},
    "B02": {"retrieval": "multiview", "control": "binary_sufficiency_gate", "backend": "multiview"},
    "B03": {"retrieval": "multiview", "control": "s2g_inspired", "backend": "multiview", "official": False},
    "B04": {"retrieval": "multiview", "control": "generic_llm_tools", "backend": "multiview"},
    "B05": {"retrieval": "multiview", "control": "plain_history_summary", "backend": "multiview"},
    "O00": {"retrieval": "single_view", "control": "state_rule_router", "backend": "shared_flat"},
    "O01": {"retrieval": "multiview", "control": "state_rule_router", "backend": "dynamic_cue"},
    "O02": {"retrieval": "multiview", "control": "state_rule_router_no_bridge", "backend": "dynamic_cue"},
}
