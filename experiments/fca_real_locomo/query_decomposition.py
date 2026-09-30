"""Lightweight query decomposition for FGW evidence matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.fca_real_locomo.question_analyzer import (
    analyze_question,
    generate_typed_queries,
)

@dataclass
class QueryNode:
    id: str
    text: str
    embedding: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text}


def encode_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    from experiments.fca_real_locomo.encode_query import encode_query

    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return np.stack([encode_query(t, model_name) for t in texts])


def decompose_question_rule(
    question: str,
    known_speakers: Optional[List[str]] = None,
    min_nodes: int = 2,
    max_nodes: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> List[QueryNode]:
    """Rule-based decomposition: original Q + typed sub-queries (2-5 nodes)."""
    info = analyze_question(question, known_speakers)
    texts: List[str] = [question.strip()]

    typed = generate_typed_queries(info)
    for t in typed:
        t = t.strip()
        if t and t.lower() not in {x.lower() for x in texts}:
            texts.append(t)
        if len(texts) >= max_nodes:
            break

    persons = info.get("persons") or []
    slot = info.get("fact_slot", "general")
    if len(texts) < min_nodes and persons:
        p = persons[0]
        texts.append(f"{p} {slot.replace('_', ' ')}")
    if len(texts) < min_nodes:
        texts.append(f"{question} background context")
    if len(texts) < min_nodes:
        texts.append(f"{question} related events")

    texts = texts[:max_nodes]
    embs = encode_texts(texts, embedding_model)
    return [
        QueryNode(id=f"q{i + 1}", text=texts[i], embedding=embs[i])
        for i in range(len(texts))
    ]


def decompose_question_llm(
    question: str,
    llm_client,
    cache_key: str,
    known_speakers: Optional[List[str]] = None,
    min_nodes: int = 2,
    max_nodes: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> List[QueryNode]:
    """Optional LLM decomposition; falls back to rule-based on failure."""
    system = "You decompose questions into 2-5 short sub-questions for memory retrieval."
    user = f"""Split this question into 2-5 atomic sub-questions that together cover the logic needed to answer it.

Question: {question}

Return JSON: {{"subquestions": ["...", "..."]}}"""
    try:
        out = llm_client.complete_json(system, user, cache_key)
        subs = out.get("parsed", {}).get("subquestions") or []
        texts = [question.strip()]
        for s in subs:
            s = str(s).strip()
            if s and s not in texts:
                texts.append(s)
            if len(texts) >= max_nodes:
                break
        if len(texts) >= min_nodes:
            embs = encode_texts(texts[:max_nodes], embedding_model)
            return [
                QueryNode(id=f"q{i + 1}", text=texts[i], embedding=embs[i])
                for i in range(len(texts[:max_nodes]))
            ]
    except Exception:
        pass
    return decompose_question_rule(
        question, known_speakers, min_nodes, max_nodes, embedding_model,
    )
