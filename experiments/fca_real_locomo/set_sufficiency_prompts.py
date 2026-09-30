"""Prompt templates for set-level memory sufficiency diagnosis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def format_memories_for_prompt(memories: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for mem in memories:
        parts = [
            f"[memory_id: {mem['memory_id']} | rank: {mem.get('rank', '?')}]",
        ]
        for key in ("speaker", "date", "session", "text"):
            val = mem.get(key)
            if val:
                parts.append(f"{key}: {val}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def allowed_memory_ids_block(memories: List[Dict[str, Any]], limit: int = 60) -> str:
    ids = [str(m["memory_id"]) for m in memories[:limit]]
    return (
        "ALLOWED memory_id values — you MUST copy these EXACT strings "
        "(UUID format, do NOT invent ids, ranks, or indices):\n"
        + "\n".join(f"- {mid}" for mid in ids)
    )


STRICT_MEMORY_ID_RULES = """Memory ID rules (CRITICAL):
- Every memory_id MUST be copied exactly from the [memory_id: ...] header above.
- Do NOT use rank numbers, indices, placeholders, or invented ids (e.g. "1", "36", "m1").
- If a requirement is unsupported, leave it in missing_requirements instead of guessing an id.
- requirement_coverage.memory_ids and selected_subset.memory_id must all be from ALLOWED ids."""


def build_setr_selection_prompt(
    question: str,
    memories: List[Dict[str, Any]],
    gold_answer: Optional[str] = None,
    include_gold_reference: bool = False,
) -> str:
    mem_block = format_memories_for_prompt(memories)
    gold_block = ""
    if include_gold_reference and gold_answer:
        gold_block = f"\nGold answer, for evaluation reference:\n{gold_answer}\n"
    allowed_ids = allowed_memory_ids_block(memories)

    return f"""You are given a question and 10 retrieved memories.

Your task is to determine what information is required to answer the question,
and select the smallest subset of memories that together provide that information.

Question:
{question}
{gold_block}
Retrieved memories:
{mem_block}

Step 1: Decompose the question into information requirements.
Each requirement should be specific and answer-oriented.

Step 2: For each requirement, identify which memory or memories support it.
Use memory ids only.

Step 3: Select the smallest memory subset S that can answer the question.

Rules:
- Do not select memories just because they mention the same person.
- Select a memory only if it contributes a necessary fact, bridge, temporal clue, speaker clue, or answer-bearing information.
- If no memory supports a requirement, mark it as missing.
- For multi-hop questions, select all memories needed to connect the reasoning chain.
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
      "memory_ids": ["<exact memory_id from ALLOWED list>"],
      "explanation": "..."
    }}
  ],
  "selected_subset": [
    {{
      "memory_id": "<exact memory_id from ALLOWED list>",
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


def build_sufficiency_judge_prompt(
    question: str,
    subset_memories: List[Dict[str, Any]],
    top10_memories: List[Dict[str, Any]],
) -> str:
    subset_block = format_memories_for_prompt(subset_memories)
    top10_block = format_memories_for_prompt(top10_memories)
    return f"""You are given a question and a set of retrieved memories.

Question:
{question}

Candidate memory subset:
{subset_block}

Full top-10 memories:
{top10_block}

Your task:
1. Decide whether the selected subset is sufficient to answer the question.
2. Decide whether the full top-10 memory context is sufficient to answer the question.
3. Explain what exact information is present or missing.

Important:
- "Sufficient" means the answer can be derived from the memories without guessing.
- A memory set is insufficient if it only mentions the correct person but not the required fact.
- A memory set is insufficient if it has the answer but lacks the bridge needed to connect it to the question.
- For temporal questions, the event and date must be correctly linked.
- For list questions, the context must cover all or most required items.

Output JSON only:

{{
  "subset_sufficient": true,
  "top10_sufficient": true,
  "answerable_from_subset": "...",
  "answerable_from_top10": "...",
  "supporting_memory_ids": ["m1", "m3"],
  "missing_information": [],
  "confidence": "high | medium | low"
}}"""


def build_missing_gap_prompt(
    question: str,
    top10_memories: List[Dict[str, Any]],
    missing_requirements: List[Dict[str, Any]],
) -> str:
    top10_block = format_memories_for_prompt(top10_memories)
    missing_block = "\n".join(
        f"- {r.get('requirement_id', '?')}: {r.get('description', '')} (type: {r.get('type', '')})"
        for r in missing_requirements
    ) or "(none listed)"
    return f"""The current top-10 memories are insufficient to answer the question.

Question:
{question}

Top-10 memories:
{top10_block}

Missing requirements from previous step:
{missing_block}

Your task:
1. Identify the missing information gap.
2. Classify the gap type.
3. Verify that the missing information is not present in the top-10 memories.
4. Convert the gap into retrieval cues that could be used to search memory again.

Gap types:
- answer_bearing_fact
- entity_bridge
- temporal_bridge
- speaker_bridge
- event_bridge
- relation_object
- list_item
- coreference
- session_context
- ambiguous_question
- gold_annotation_issue

Output JSON only:

{{
  "missing_gaps": [
    {{
      "gap": "...",
      "gap_type": "...",
      "verified_missing": true,
      "evidence_for_missing": "Explain why top-10 does not contain it.",
      "retrieval_cues": [
        "...",
        "..."
      ]
    }}
  ]
}}"""


PROMPT_VERSIONS = {
    "v1": {
        "setr": build_setr_selection_prompt,
        "judge": build_sufficiency_judge_prompt,
        "gap": build_missing_gap_prompt,
    },
}
