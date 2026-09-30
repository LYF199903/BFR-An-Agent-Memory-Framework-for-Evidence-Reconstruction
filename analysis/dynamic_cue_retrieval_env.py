#!/usr/bin/env python3
"""
Dynamic Cue Retrieval environment / loop.

FCA-MS → Evidence State → cue hypotheses → parameterized retrieval
→ candidate routing → update state → extract new cues → repeat

Deterministic, score-based controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from analysis.evidence_state_v2 import EvidenceStateV2, EvidenceStateV2Builder
from analysis.parameterized_evidence_action import ParameterizedEvidenceAction
from experiments.fca_real_locomo.question_analyzer import extract_persons
from models.evidence_candidate_router import EvidenceCandidateRouter
from models.evidence_cue_generator import EvidenceCueGenerator
from models.memory_agent import MemoryAgent
from retrieval.multiview_flat_retriever import MultiViewFlatRetriever
from retrieval.shared_flat_retriever import _mid


@dataclass
class DynamicCueStepTrace:
    step: int
    target_requirement: str
    hypotheses: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    retrieved_ids: List[str]
    kept_ids: List[str]
    router_diagnostics: List[Dict[str, Any]]
    new_anchors: List[Dict[str, Any]]
    gold_hit_after: Optional[bool] = None  # filled only in eval if gold provided externally


@dataclass
class DynamicCueResult:
    question_id: str
    initial_evidence_ids: List[str]
    final_evidence_ids: List[str]
    steps: List[DynamicCueStepTrace] = field(default_factory=list)
    retrieval_calls: int = 0
    unique_accessed: int = 0
    retained_count: int = 0
    cue_source: str = ""
    stop_reason: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    # optional component flags
    use_cue: bool = True
    use_parameterized: bool = True
    use_router: bool = True
    use_multiview: bool = True
    trajectory_bridges: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "initial_evidence_ids": self.initial_evidence_ids,
            "final_evidence_ids": self.final_evidence_ids,
            "retrieval_calls": self.retrieval_calls,
            "unique_accessed": self.unique_accessed,
            "retained_count": self.retained_count,
            "cue_source": self.cue_source,
            "stop_reason": self.stop_reason,
            "config": self.config,
            "use_cue": self.use_cue,
            "use_parameterized": self.use_parameterized,
            "use_router": self.use_router,
            "use_multiview": self.use_multiview,
            "trajectory_bridges": self.trajectory_bridges,
            "steps": [
                {
                    "step": s.step,
                    "target_requirement": s.target_requirement,
                    "hypotheses": s.hypotheses,
                    "actions": s.actions,
                    "retrieved_ids": s.retrieved_ids,
                    "kept_ids": s.kept_ids,
                    "router_diagnostics": s.router_diagnostics,
                    "new_anchors": s.new_anchors,
                    "gold_hit_after": s.gold_hit_after,
                }
                for s in self.steps
            ],
        }


class DynamicCueRetrievalEnv:
    """
    Deterministic Dynamic Cue loop with shared retrieval budgets.
    """

    def __init__(
        self,
        *,
        max_steps: int = 3,
        max_retrieval_calls: int = 6,
        max_new_memories_total: int = 30,
        max_retained_per_step: int = 8,
        hypotheses_per_step: int = 2,
        top_k: int = 12,
        cue_source: str = "question_requirement_evidence",
        use_cue: bool = True,
        use_parameterized: bool = True,
        use_router: bool = True,
        use_multiview: bool = True,
        router_delta: Optional[float] = None,
        router_beta: Optional[float] = None,
        use_fixed_schedule: bool = False,
        use_state_stop: bool = True,
        use_state_router: bool = True,
        use_state_cue: bool = True,
        extra_multiview_passes: Optional[bool] = None,
        controller_mode: str = "legacy",
    ):
        self.max_steps = max_steps
        self.max_retrieval_calls = max_retrieval_calls
        self.max_new_memories_total = max_new_memories_total
        self.max_retained_per_step = max_retained_per_step
        self.hypotheses_per_step = hypotheses_per_step
        self.top_k = top_k
        self.cue_source = cue_source
        self.use_cue = use_cue
        self.use_parameterized = use_parameterized
        self.use_router = use_router
        self.use_multiview = use_multiview
        self.use_fixed_schedule = bool(use_fixed_schedule)
        self.use_state_stop = bool(use_state_stop)
        self.use_state_router = bool(use_state_router)
        self.use_state_cue = bool(use_state_cue)
        self.controller_mode = str(controller_mode or "legacy")
        if extra_multiview_passes is None:
            self.extra_multiview_passes = not self.use_fixed_schedule
        else:
            self.extra_multiview_passes = bool(extra_multiview_passes)

        cue_src = cue_source if use_cue else "question_only"
        if not self.use_state_cue and cue_src != "question_only":
            cue_src = "question_only"
        self.builder = EvidenceStateV2Builder()
        self.cue_gen = EvidenceCueGenerator(
            cue_source=cue_src,
            max_hypotheses=max(hypotheses_per_step, 3),
            top_k=top_k,
        )
        router_kw: Dict[str, Any] = dict(
            keep_top=max_retained_per_step,
            retrieve_cap=40,
            max_keep_hard=20,
            force_relevance=0.10,
        )
        if router_delta is not None:
            router_kw["delta"] = float(router_delta)
        if router_beta is not None:
            router_kw["beta"] = float(router_beta)
        if not self.use_state_router:
            router_kw["beta"] = 0.0
            router_kw["delta"] = 0.0
        self.router = EvidenceCandidateRouter(**router_kw)

    def run(
        self,
        baseline_row: Dict[str, Any],
        memories: list,
    ) -> DynamicCueResult:
        row = MemoryAgent.build_env_row_from_baseline(baseline_row)
        state = self.builder.build_from_row(row, memories)
        initial_ids = list(state.evidence_ids)
        return self._run_initialized(
            baseline_row,
            memories,
            initial_ids,
            baseline_ids=set(initial_ids),
            start_step=0,
            state=state,
        )

    def run_from_evidence(
        self,
        baseline_row: Dict[str, Any],
        memories: list,
        evidence_ids: Sequence[str],
        *,
        remaining_steps: int,
        remaining_calls: int,
        remaining_new: int,
        start_step: int = 0,
    ) -> DynamicCueResult:
        """Continue from a reconstructed evidence set (P04 interventions). Rebuilds state from ids only."""
        saved = (self.max_steps, self.max_retrieval_calls, self.max_new_memories_total)
        self.max_steps = int(remaining_steps)
        self.max_retrieval_calls = int(remaining_calls)
        self.max_new_memories_total = int(remaining_new)
        start = list(dict.fromkeys(str(x) for x in evidence_ids if x))
        try:
            return self._run_initialized(
                baseline_row,
                memories,
                start,
                baseline_ids=set(start),
                start_step=int(start_step),
            )
        finally:
            self.max_steps, self.max_retrieval_calls, self.max_new_memories_total = saved

    def _fixed_schedule_actions(
        self, state: Any, evidence_order: Optional[List[str]] = None
    ) -> List[ParameterizedEvidenceAction]:
        """Question-driven multi-view schedule. Same views every step; query may include a missing requirement."""
        q = state.question or ""
        query = q
        if self.use_state_cue:
            missing = list(state.missing_requirements or [])
            if missing:
                desc = str(missing[0].get("description") or missing[0].get("requirement_id") or "")
                if desc:
                    query = (q + " " + desc).strip()[:180]
        persons = extract_persons(q)
        actions: List[ParameterizedEvidenceAction] = [
            ParameterizedEvidenceAction(
                type="query_rewrite",
                query=query,
                retrieval_view="text",
                top_k=self.top_k,
                confidence=0.5,
            )
        ]
        if persons:
            actions.append(
                ParameterizedEvidenceAction(
                    type="entity_expand",
                    query=persons[0],
                    retrieval_view="entity",
                    anchors=[persons[0]],
                    top_k=self.top_k,
                    confidence=0.5,
                )
            )
        ordered = [x for x in (evidence_order or list(state.evidence_ids or [])) if x]
        if ordered:
            actions.append(
                ParameterizedEvidenceAction(
                    type="session_expand",
                    query=query,
                    retrieval_view="session",
                    source_memory_ids=[ordered[-1]],
                    top_k=self.top_k,
                    confidence=0.5,
                )
            )
        return actions

    def _run_initialized(
        self,
        baseline_row: Dict[str, Any],
        memories: list,
        initial_ids: List[str],
        *,
        baseline_ids: Set[str],
        start_step: int,
        state: Any = None,
    ) -> DynamicCueResult:
        row = MemoryAgent.build_env_row_from_baseline(baseline_row)
        lookup = {_mid(m): m for m in memories}
        qid = str(baseline_row.get("question_id") or row.get("question_id") or "")
        evidence_order: List[str] = list(dict.fromkeys(str(x) for x in initial_ids if x))
        evidence: Set[str] = set(evidence_order)
        accessed: Set[str] = set(initial_ids)
        retrieval_history: List[str] = []
        bridge_all: List[Dict[str, Any]] = []
        steps: List[DynamicCueStepTrace] = []
        retrieval_calls = 0
        trajectory_bridges: List[Dict[str, Any]] = []
        stop_reason = "budget_steps"
        if state is None:
            state = self.builder.rebuild(
                row,
                memories,
                evidence,
                baseline_ids=baseline_ids,
                retrieval_history=retrieval_history,
                step=start_step,
            )

        retriever = MultiViewFlatRetriever(memories, top_k_default=self.top_k)
        retriever.set_context(qid, "dynamic_cue")

        # Ablation: without multiview, only use search_text / entity via same object but force text
        for step_i in range(self.max_steps):
            if retrieval_calls >= self.max_retrieval_calls:
                stop_reason = "budget_calls"
                break
            if len(evidence) - len(baseline_ids) >= self.max_new_memories_total:
                stop_reason = "budget_new"
                break

            # Adaptive stop is Evidence-State-conditioned; Q-Fixed / ES-Fixed disable it.
            if (
                self.use_state_stop
                and step_i > 0
                and not state.missing_requirements
                and state.uncertainty_score < 0.20
            ):
                stop_reason = "state_stop"
                break

            # Generate hypotheses
            if self.use_fixed_schedule:
                actions = self._fixed_schedule_actions(state, evidence_order)
                gen = {
                    "active_anchors": [],
                    "search_hypotheses": [],
                    "actions": actions,
                    "target_requirement": (
                        str((state.missing_requirements or [{}])[0].get("description") or "")
                        if self.use_state_cue and state.missing_requirements
                        else ""
                    ),
                }
            elif self.use_cue:
                gen = self.cue_gen.generate(state, lookup, k=self.hypotheses_per_step)
            else:
                # current-flat-like: single question rewrite
                gen = {
                    "active_anchors": [],
                    "search_hypotheses": [
                        {
                            "target_requirement": "",
                            "operation": "query_rewrite",
                            "anchors": [],
                            "query": state.question,
                            "source_memory_ids": [],
                            "confidence": 0.5,
                            "retrieval_view": "text",
                        }
                    ],
                    "actions": [
                        ParameterizedEvidenceAction(
                            type="query_rewrite",
                            query=state.question,
                            retrieval_view="text",
                            top_k=self.top_k,
                            confidence=0.5,
                        )
                    ],
                    "target_requirement": "",
                }

            state.active_anchors = list(gen.get("active_anchors") or state.active_anchors)
            state.search_hypotheses = list(gen.get("search_hypotheses") or [])
            actions: List[ParameterizedEvidenceAction] = list(gen.get("actions") or [])

            if not self.use_parameterized:
                # collapse to plain text query_rewrite only
                actions = [
                    ParameterizedEvidenceAction(
                        type="query_rewrite",
                        target_requirement=gen.get("target_requirement") or "",
                        query=(actions[0].query if actions else state.question),
                        retrieval_view="text",
                        top_k=self.top_k,
                        confidence=0.5,
                    )
                ]

            retrieved_step: List[str] = []
            for action in actions:
                if retrieval_calls >= self.max_retrieval_calls:
                    break
                if not self.use_multiview:
                    # text-only backend
                    ids = retriever.search_text(action.query or state.question, top_k=self.top_k, exclude=evidence)
                else:
                    ids = retriever.execute_action(action, exclude=evidence)
                retrieval_calls += 1
                retrieval_history.append(action.to_key())
                for mid in ids:
                    accessed.add(mid)
                retrieved_step.extend(ids)

            # Mandatory entity pass for question persons (helps never-retrieved)
            if self.use_multiview and self.extra_multiview_passes and retrieval_calls < self.max_retrieval_calls:
                ents = [a.get("value") for a in (state.active_anchors or []) if a.get("type") == "entity"]
                ents = [e for e in ents if e][:1]
                if not ents:
                    ents = extract_persons(state.question)[:1]
                if ents:
                    ids = retriever.search_entity(ents[0], top_k=self.top_k, exclude=evidence)
                    retrieval_calls += 1
                    retrieval_history.append(f"entity_expand|{ents[0]}")
                    for x in ids:
                        accessed.add(x)
                    retrieved_step.extend(ids)

            # Session-expand: prefer evidence sharing question entities / newest
            if (
                self.use_multiview
                and self.extra_multiview_passes
                and retrieval_calls < self.max_retrieval_calls
                and state.evidence_ids
            ):
                mid = evidence_order[-1] if evidence_order else ""
                # prefer initial evidence in same dialogue session as question entities if possible
                q_ents = {a.get("value", "").lower() for a in (state.active_anchors or []) if a.get("type") == "entity"}
                for cand in list(state.evidence_ids)[:8]:
                    m = lookup.get(cand)
                    if not m:
                        continue
                    blob = f"{getattr(m,'content','')} {getattr(m,'speaker','')}".lower()
                    if any(e and e in blob for e in q_ents):
                        mid = cand
                        break
                ids = retriever.search_session("", top_k=self.top_k, exclude=evidence, anchor_memory_id=mid)
                retrieval_calls += 1
                retrieval_history.append(f"session_expand|{mid}")
                for x in ids:
                    accessed.add(x)
                retrieved_step.extend(ids)

            # unique preserve order
            seen_r = set()
            retrieved_uniq = []
            for mid in retrieved_step:
                if mid not in seen_r:
                    seen_r.add(mid)
                    retrieved_uniq.append(mid)

            if self.use_router:
                miss = list(state.missing_requirements or []) if self.use_state_router else []
                routed = self.router.route(
                    retrieved_uniq,
                    lookup=lookup,
                    question=state.question,
                    missing_requirements=miss,
                    current_evidence_ids=sorted(evidence),
                    keep_top=self.max_retained_per_step,
                )
                kept = list(routed["kept_ids"])
                diag = list(routed["diagnostics"])
                bridges = list(routed["bridge_evidence"])
            else:
                # keep first N retrieved
                kept = [m for m in retrieved_uniq if m not in evidence][: self.max_retained_per_step]
                diag = [{"memory_id": m, "keep": True, "reason": "no_router"} for m in kept]
                bridges = []

            # Track bridge hops for analysis
            for b in bridges:
                trajectory_bridges.append(
                    {
                        "step": step_i + 1,
                        "memory_id": b.get("memory_id"),
                        "reason": b.get("reason"),
                        "generated_cues": b.get("generated_cues"),
                    }
                )
            bridge_all.extend(bridges)

            # Update evidence
            before = set(evidence)
            for mid in kept:
                if len(evidence) - len(baseline_ids) >= self.max_new_memories_total:
                    break
                if mid not in evidence:
                    evidence.add(mid)
                    evidence_order.append(mid)

            state = self.builder.rebuild(
                row,
                memories,
                evidence,
                baseline_ids=baseline_ids,
                retrieval_history=retrieval_history,
                step=step_i + 1,
                bridge_evidence=bridge_all,
            )
            new_anchors = [a for a in state.active_anchors if a.get("source") == "evidence"]

            steps.append(
                DynamicCueStepTrace(
                    step=step_i + 1,
                    target_requirement=str(gen.get("target_requirement") or ""),
                    hypotheses=list(gen.get("search_hypotheses") or []),
                    actions=[a.to_dict() for a in actions],
                    retrieved_ids=retrieved_uniq,
                    kept_ids=kept,
                    router_diagnostics=diag,
                    new_anchors=new_anchors,
                )
            )

            if not kept:
                # no progress
                continue

        return DynamicCueResult(
            question_id=qid,
            initial_evidence_ids=initial_ids,
            final_evidence_ids=sorted(evidence),
            steps=steps,
            retrieval_calls=retrieval_calls,
            unique_accessed=len(accessed),
            retained_count=len(evidence),
            cue_source=self.cue_source,
            stop_reason=stop_reason,
            config={
                "max_steps": self.max_steps,
                "max_retrieval_calls": self.max_retrieval_calls,
                "max_new_memories_total": self.max_new_memories_total,
                "max_retained_per_step": self.max_retained_per_step,
                "hypotheses_per_step": self.hypotheses_per_step,
                "top_k": self.top_k,
                "controller_mode": self.controller_mode,
                "use_fixed_schedule": self.use_fixed_schedule,
                "use_state_stop": self.use_state_stop,
                "use_state_router": self.use_state_router,
                "use_state_cue": self.use_state_cue,
                "extra_multiview_passes": self.extra_multiview_passes,
            },
            use_cue=self.use_cue,
            use_parameterized=self.use_parameterized,
            use_router=self.use_router,
            use_multiview=self.use_multiview,
            trajectory_bridges=trajectory_bridges,
        )
