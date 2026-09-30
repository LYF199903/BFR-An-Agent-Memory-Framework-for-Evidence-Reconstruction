"""Load cached A-Mem memories and QA results into unified records."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

CATEGORY_NAMES = {
    1: "multihop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


@dataclass
class MemoryRecord:
    memory_id: str
    dialogue_id: str
    memory_index: int
    session_id: str = ""
    timestamp: str = ""
    date: str = ""
    speaker: str = ""
    content: str = ""
    context: str = ""
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    links: List = field(default_factory=list)
    embedding: Optional[List[float]] = None
    raw: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("embedding") is not None:
            d["embedding"] = None  # omit large vectors in jsonl
        return d


@dataclass
class QuestionRecord:
    question_id: str
    dialogue_id: str
    category: int
    category_name: str
    question: str
    reference: str
    prediction: str
    metrics: Dict[str, float] = field(default_factory=dict)
    retrieved_memory_ids: List[str] = field(default_factory=list)
    raw: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_speaker(content: str) -> str:
    if content.startswith("Speaker ") and "says" in content:
        try:
            part = content.split("says", 1)[0]
            return part.replace("Speaker ", "").strip()
        except Exception:
            pass
    return ""


def _parse_date(timestamp: str) -> str:
    if not timestamp:
        return ""
    # e.g. "8:29 pm on 13 June, 2023"
    if " on " in timestamp:
        return timestamp.split(" on ", 1)[-1].strip()
    return timestamp.strip()


def load_memories_from_cache(cache_dir: str) -> Dict[str, List[MemoryRecord]]:
    """Load all memories keyed by dialogue_id (sample index string)."""
    by_dialogue: Dict[str, List[MemoryRecord]] = {}
    if not os.path.isdir(cache_dir):
        return by_dialogue

    for fname in sorted(os.listdir(cache_dir)):
        if not fname.startswith("memory_cache_sample_") or not fname.endswith(".pkl"):
            continue
        sample_idx = fname.replace("memory_cache_sample_", "").replace(".pkl", "")
        mem_path = os.path.join(cache_dir, fname)
        emb_path = os.path.join(cache_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy")

        with open(mem_path, "rb") as f:
            mem_dict = pickle.load(f)

        embeddings = None
        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)

        records: List[MemoryRecord] = []
        for idx, (mid, note) in enumerate(mem_dict.items()):
            content = getattr(note, "content", "") or ""
            ts = getattr(note, "timestamp", "") or ""
            emb = None
            if embeddings is not None and idx < len(embeddings):
                emb = embeddings[idx].tolist()

            records.append(MemoryRecord(
                memory_id=str(mid),
                dialogue_id=str(sample_idx),
                memory_index=idx,
                session_id=ts,
                timestamp=ts,
                date=_parse_date(ts),
                speaker=_parse_speaker(content),
                content=content,
                context=getattr(note, "context", "") or "",
                keywords=list(getattr(note, "keywords", []) or []),
                tags=list(getattr(note, "tags", []) or []),
                links=list(getattr(note, "links", []) or []),
                embedding=emb,
                raw=repr(note)[:500],
            ))
        by_dialogue[str(sample_idx)] = records

    return by_dialogue


def load_questions_from_results(
    results_path: str,
    dataset_path: Optional[str] = None,
) -> List[QuestionRecord]:
    """Load questions from results JSON."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: List[QuestionRecord] = []
    for i, item in enumerate(data.get("individual_results", [])):
        cat = int(item.get("category", 0))
        records.append(QuestionRecord(
            question_id=f"q_{i:05d}",
            dialogue_id=str(item.get("sample_id", "")),
            category=cat,
            category_name=CATEGORY_NAMES.get(cat, f"cat{cat}"),
            question=item.get("question", ""),
            reference=item.get("reference", "") or "",
            prediction=item.get("prediction", "") or "",
            metrics=item.get("metrics", {}) or {},
            raw=item,
        ))
    return records


def build_dialogue_speakers(dataset_path: str) -> Dict[str, List[str]]:
    """Map dialogue_id -> [speaker_a, speaker_b] from locomo dataset."""
    if not dataset_path or not os.path.exists(dataset_path):
        return {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for i, sample in enumerate(data):
        conv = sample.get("conversation", {})
        out[str(i)] = [conv.get("speaker_a", ""), conv.get("speaker_b", "")]
    return out


def get_memory_index_map(records: List[MemoryRecord]) -> Dict[str, MemoryRecord]:
    return {r.memory_id: r for r in records}


def embedding_matrix(records: List[MemoryRecord]) -> Tuple[np.ndarray, List[str]]:
    """Return (N, D) matrix and parallel memory_ids."""
    ids = []
    rows = []
    for r in records:
        if r.embedding is not None:
            ids.append(r.memory_id)
            rows.append(r.embedding)
    if not rows:
        return np.zeros((0, 384)), []
    return np.array(rows, dtype=np.float32), ids
