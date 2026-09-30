"""Prompt templates for MS-Select memory set selection and sufficiency diagnosis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from experiments.fca_real_locomo.set_sufficiency_prompts import (
    STRICT_MEMORY_ID_RULES,
    allowed_memory_ids_block,
    build_missing_gap_prompt,
    build_setr_selection_prompt,
    build_sufficiency_judge_prompt,
    format_memories_for_prompt,
)


def build_setr_selection_prompt_pool(
    question: str,
    memories: List[Dict[str, Any]],
    pool_label: str = "retrieved",
) -> str:
    mem_block = format_memories_for_prompt(memories)
    n = len(memories)
    allowed_ids = allowed_memory_ids_block(memories)
    return f"""You are given a question and {n} {pool_label} memories.

Your task is to determine what information is required to answer the question,
and select the smallest subset of memories that together provide that information.

Question:
{question}

Retrieved memories:
{mem_block}

Step 1: Decompose the question into information requirements.
Each requirement should be specific and answer-oriented.

Step 2: For each requirement, identify which memory or memories support it.
Use memory ids only.

Step 3: Select the smallest memory subset S that can answer the question.
Prefer multiple memories when the question requires multi-hop reasoning.

Rules:
- Do not select memories just because they mention the same person.
- Select a memory only if it contributes a necessary fact, bridge, temporal clue, speaker clue, or answer-bearing information.
- If no memory supports a requirement, mark it as missing.
- For multi-hop questions, select ALL memories needed to connect the reasoning chain.
- For list questions, select memories that cover different required list items.
- For temporal questions, select memories that bind the correct event to the correct date.
{STRICT_MEMORY_ID_RULES}

{allowed_ids}

Output JSON only:

{{
  "information_requirements": [
    {{
      "requirement_id": "R1",
      "description": "...",
      "type": "entity_anchor | answer_bearing_fact | entity_bridge | temporal_bridge | speaker_bridge | event_bridge | relation_object | list_item | coreference"
    }}
  ],
  "requirement_coverage": [
    {{
      "requirement_id": "R1",
      "covered": true,
      "memory_ids": ["..."],
      "explanation": "..."
    }}
  ],
  "selected_subset": [
    {{
      "memory_id": "...",
      "supports": ["R1"],
      "reason": "..."
    }}
  ],
  "missing_requirements": [
    {{
      "requirement_id": "R2",
      "description": "...",
      "type": "..."
    }}
  ]
}}"""


def build_sufficiency_judge_prompt_strict(
    question: str,
    subset_memories: List[Dict[str, Any]],
    context_memories: List[Dict[str, Any]],
    context_label: str = "Full top-10 memories",
) -> str:
    subset_block = format_memories_for_prompt(subset_memories)
    ctx_block = format_memories_for_prompt(context_memories)
    return f"""You are given a question and retrieved memories. Apply STRICT sufficiency criteria.

Question:
{question}

Candidate memory subset:
{subset_block}

{context_label}:
{ctx_block}

Your task:
1. Decide whether the selected subset is sufficient to answer the question.
2. Decide whether the full context memory set is sufficient to answer the question.
3. Explain what exact information is present or missing.

STRICT rules for multi-hop questions:
- A single memory is sufficient ONLY if it explicitly contains the complete reasoning chain AND final answer.
- If a memory contains the answer but NOT the bridge connecting it to the question, mark INSUFFICIENT.
- If the question asks for multiple facts or list items, partial coverage is INSUFFICIENT.
- Do NOT mark sufficient just because a plausible answer appears somewhere.
- "Sufficient" means the answer can be derived without guessing.

Output JSON only:

{{
  "subset_sufficient": false,
  "top10_sufficient": false,
  "answerable_from_subset": "...",
  "answerable_from_top10": "...",
  "supporting_memory_ids": [],
  "missing_information": [],
  "confidence": "high | medium | low"
}}"""


def build_gap_retrieval_expand_prompt(
    question: str,
    retrieval_cues: List[str],
    existing_ids: List[str],
    candidate_memories: List[Dict[str, Any]],
) -> str:
    cues = "\n".join(f"- {c}" for c in retrieval_cues)
    mem_block = format_memories_for_prompt(candidate_memories[:30])
    return f"""You are helping retrieve missing memories for a question.

Question:
{question}

Missing retrieval cues:
{cues}

Already retrieved memory ids (do not re-select these unless necessary):
{existing_ids}

Candidate memories from dialogue (not yet in context):
{mem_block}

Select up to 3 additional memory ids that best fill the missing information gap.
Output JSON only:

{{
  "added_memory_ids": ["..."],
  "reason": "..."
}}"""


BRIDGE_GAP_TYPES = {
    "entity_bridge",
    "temporal_bridge",
    "speaker_bridge",
    "event_bridge",
    "relation_object",
    "coreference",
}


PROMPT_VERSIONS = {
    "v1": {
        "setr": lambda q, mems, **_: build_setr_selection_prompt(q, mems[:10]),
        "setr_pool": build_setr_selection_prompt_pool,
        "judge": lambda q, sub, ctx, **kw: build_sufficiency_judge_prompt(q, sub, ctx),
        "gap": build_missing_gap_prompt,
        "strict": False,
    },
    "v2_strict": {
        "setr": lambda q, mems, **_: build_setr_selection_prompt(q, mems[:10]),
        "setr_pool": build_setr_selection_prompt_pool,
        "judge": lambda q, sub, ctx, **kw: build_sufficiency_judge_prompt_strict(
            q, sub, ctx, context_label=kw.get("context_label", "Full top-10 memories")
        ),
        "gap": build_missing_gap_prompt,
        "strict": True,
    },
}
