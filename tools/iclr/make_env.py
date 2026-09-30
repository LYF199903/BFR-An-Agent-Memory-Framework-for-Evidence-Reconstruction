"""Paper method constructors over DynamicCueRetrievalEnv.

Names match the ICLR manuscript:
  FCA-MS      Stage I only (no env loop)
  OnePass-MV  one multi-view pass
  BFR-Text    budgeted completion, text view
  BFR-MV      budgeted completion, multi-view
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from analysis.dynamic_cue_retrieval_env import DynamicCueRetrievalEnv

DEFAULT_BUDGET = {
    "max_reasoning_steps": 3,
    "max_retrieval_calls": 6,
    "max_new_memories": 30,
}


def make_env(method_id: str, budget: Optional[Dict[str, Any]] = None) -> DynamicCueRetrievalEnv:
    budget = {**DEFAULT_BUDGET, **(budget or {})}
    steps = int(budget["max_reasoning_steps"])
    calls = int(budget["max_retrieval_calls"])
    newm = int(budget["max_new_memories"])
    common = dict(
        max_steps=steps,
        max_retrieval_calls=calls,
        max_new_memories_total=newm,
        max_retained_per_step=min(8, newm),
        hypotheses_per_step=2,
        top_k=12,
    )
    key = str(method_id).strip()
    if key in ("OnePass-MV", "OneShot-MV"):
        oneshot = dict(common)
        oneshot["max_steps"] = 1
        oneshot["max_retrieval_calls"] = min(3, calls)
        return DynamicCueRetrievalEnv(
            **oneshot,
            cue_source="question_only",
            use_cue=True,
            use_parameterized=True,
            use_router=True,
            use_multiview=True,
            use_fixed_schedule=True,
            use_state_stop=False,
            use_state_router=False,
            use_state_cue=False,
            extra_multiview_passes=False,
            router_beta=0.0,
            router_delta=0.0,
            controller_mode="oneshot_mv",
        )
    if key in ("BFR-Text", "O00"):
        return DynamicCueRetrievalEnv(
            **common,
            cue_source="question_requirement_evidence",
            use_cue=True,
            use_parameterized=True,
            use_router=True,
            use_multiview=False,
        )
    if key in ("BFR-MV", "O01"):
        return DynamicCueRetrievalEnv(
            **common,
            cue_source="question_requirement_evidence",
            use_cue=True,
            use_parameterized=True,
            use_router=True,
            use_multiview=True,
        )
    raise ValueError(f"Unknown method_id={method_id!r}; expected OnePass-MV, BFR-Text, or BFR-MV")
