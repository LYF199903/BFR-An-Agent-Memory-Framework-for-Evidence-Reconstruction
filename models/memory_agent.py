"""Row adapter used by the Stage II retrieval environment.

Learned policy checkpoints are not included.
"""

from __future__ import annotations

from typing import Any, Dict


class MemoryAgent:
    @staticmethod
    def build_env_row_from_baseline(
        baseline_row: Dict[str, Any],
        *,
        method: str = "fca_ms",
    ) -> Dict[str, Any]:
        """Build DecisionEnv row from an FCA-MS (or equivalent) init set."""
        b = (baseline_row.get("baselines") or {}).get(method) or {}
        selected = list(b.get("selected_memories") or [])
        return {
            "question_id": baseline_row.get("question_id"),
            "dialogue_id": baseline_row.get("dialogue_id"),
            "question": baseline_row.get("question"),
            "gold_answer": str(baseline_row.get("gold_answer") or ""),
            "selected_memory_ids": selected,
            "top10_memory_ids": selected,
            "information_requirements": list(baseline_row.get("information_requirements") or []),
            "gold_memory_ids": list(baseline_row.get("gold_memory_ids") or []),
        }
