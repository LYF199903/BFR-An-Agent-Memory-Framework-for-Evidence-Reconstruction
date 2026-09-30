#!/usr/bin/env python3
"""
Evidence State v2 — actionable extensions for Dynamic Cue Retrieval.

Old EvidenceState fields remain. New optional fields:
  active_anchors, enriched search_hypotheses, bridge_evidence
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from analysis.evidence_state import EvidenceState, EvidenceStateBuilder
from experiments.fca_real_locomo.load_amem_cache import MemoryRecord
from experiments.fca_real_locomo.question_analyzer import extract_persons

MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
EVENT_VERBS = {
    "stop", "stopped", "quit", "leave", "left", "start", "started",
    "join", "joined", "move", "moved", "lose", "lost", "win", "won",
    "buy", "bought", "sell", "sold", "visit", "visited", "travel",
    "travelling", "traveling", "work", "worked", "meet", "met",
    "open", "opened", "launch", "launched", "graduate", "graduated",
    "hurt", "injured", "injury", "illness", "sick", "health",
}


def _mem_blob(m: Any) -> str:
    if isinstance(m, dict):
        return f"{m.get('content') or ''} {m.get('context') or ''} {m.get('speaker') or ''}"
    return f"{getattr(m, 'content', '')} {getattr(m, 'context', '')} {getattr(m, 'speaker', '')}"


def _mid(m: Any) -> str:
    return str(getattr(m, "memory_id", None) or (m.get("memory_id") if isinstance(m, dict) else "") or "")


def extract_time_phrases(text: str) -> List[str]:
    t = text or ""
    out: List[str] = []
    for m in re.finditer(
        r"\b(?:\d{1,2}\s+)?(?:" + "|".join(MONTHS) + r")[a-z]*,?\s*\d{4}\b",
        t,
        flags=re.I,
    ):
        out.append(m.group(0).strip())
    for m in re.finditer(r"\b(?:yesterday|today|last week|next month|this month)\b", t, re.I):
        out.append(m.group(0).strip())
    for mon in MONTHS:
        if re.search(rf"\b{mon}\b", t, re.I):
            out.append(mon.capitalize())
    seen, uniq = set(), []
    for x in out:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


def extract_event_phrases(text: str, max_n: int = 6) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z'-]+", text or "")
    phrases = []
    lower = [t.lower() for t in toks]
    for i, w in enumerate(lower):
        if w in EVENT_VERBS and i + 1 < len(toks):
            phrases.append(" ".join(toks[i : i + 3])[:60])
        elif w.endswith("ing") and len(w) > 5:
            phrases.append(toks[i])
    for t in toks:
        if t[0].isupper() and len(t) > 2:
            phrases.append(t)
    seen, out = set(), []
    for p in phrases:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
        if len(out) >= max_n:
            break
    return out


def extract_topic_phrases(text: str, max_n: int = 5) -> List[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "have", "been",
        "were", "what", "when", "where", "which", "about", "into", "your",
        "their", "they", "them", "said", "says", "just", "like", "also",
        "does", "will", "would", "could", "should", "really",
    }
    toks = [t.lower() for t in re.findall(r"[a-zA-Z]{4,}", text or "") if t.lower() not in stop]
    return [w for w, _ in Counter(toks).most_common(max_n)]


def build_active_anchors(
    question: str,
    evidence_ids: Sequence[str],
    lookup: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Question- + evidence-derived anchors. Never uses gold."""
    anchors: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def add(atype: str, value: str, source: str, mid: str = "", conf: float = 0.5):
        v = str(value or "").strip()
        if not v or len(v) < 2:
            return
        key = f"{atype}:{v.lower()}"
        if key in seen:
            return
        seen.add(key)
        anchors.append(
            {
                "type": atype,
                "value": v,
                "source_memory_id": mid,
                "source": source,
                "confidence": conf,
            }
        )

    for p in extract_persons(question):
        add("entity", p, "question", conf=0.9)
    for t in extract_time_phrases(question):
        add("time", t, "question", conf=0.7)
    for e in extract_event_phrases(question, max_n=4):
        add("event", e, "question", conf=0.5)
    for t in extract_topic_phrases(question, max_n=4):
        add("topic", t, "question", conf=0.4)
    for tok in re.findall(r"[A-Za-z]{4,}", question or "")[:8]:
        if tok.lower() not in {"when", "what", "where", "which", "does", "have", "both"}:
            add("phrase", tok, "question", conf=0.35)

    for mid in evidence_ids:
        m = lookup.get(mid)
        if not m:
            continue
        text = _mem_blob(m)
        speaker = getattr(m, "speaker", None) or (m.get("speaker") if isinstance(m, dict) else "") or ""
        if speaker:
            sp = str(speaker).split(":")[0].strip()
            if sp and len(sp.split()) <= 3 and len(sp) < 40:
                add("entity", sp, "evidence", mid, 0.8)
        for p in extract_persons(text):
            add("entity", p, "evidence", mid, 0.75)
        for t in extract_time_phrases(text)[:3]:
            add("time", t, "evidence", mid, 0.7)
        ts = getattr(m, "timestamp", None) or (m.get("timestamp") if isinstance(m, dict) else "") or ""
        for t in extract_time_phrases(str(ts))[:2]:
            add("time", t, "evidence", mid, 0.85)
        for e in extract_event_phrases(text, max_n=3):
            add("event", e, "evidence", mid, 0.55)
        for ph in extract_topic_phrases(text, max_n=3):
            add("phrase", ph, "evidence", mid, 0.4)
        kws = getattr(m, "keywords", None) or (m.get("keywords") if isinstance(m, dict) else []) or []
        for kw in list(kws)[:4]:
            add("phrase", str(kw), "evidence", mid, 0.45)

    return anchors


@dataclass
class EvidenceStateV2:
    """Actionable Evidence State with Dynamic Cue fields."""

    base: EvidenceState = field(default_factory=EvidenceState)
    active_anchors: List[Dict[str, Any]] = field(default_factory=list)
    bridge_evidence: List[Dict[str, Any]] = field(default_factory=list)
    search_hypotheses: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def question(self) -> str:
        return self.base.question

    @property
    def question_id(self) -> str:
        return self.base.question_id

    @property
    def evidence_ids(self) -> List[str]:
        return self.base.evidence_ids

    @property
    def missing_requirements(self) -> List[Dict[str, Any]]:
        return self.base.missing_requirements

    @property
    def covered_requirements(self) -> List[Dict[str, Any]]:
        return self.base.covered_requirements

    @property
    def step(self) -> int:
        return self.base.step

    @property
    def uncertainty_score(self) -> float:
        return self.base.uncertainty_score

    def to_dict(self) -> Dict[str, Any]:
        d = self.base.to_dict()
        d["active_anchors"] = list(self.active_anchors)
        d["bridge_evidence"] = list(self.bridge_evidence)
        if self.search_hypotheses:
            d["search_hypotheses"] = list(self.search_hypotheses)
        return d

    def refresh_anchors(self, lookup: Dict[str, Any]) -> None:
        self.active_anchors = build_active_anchors(self.question, self.evidence_ids, lookup)
        # sync onto base for serialization compatibility
        self.base.active_anchors = list(self.active_anchors)
        self.base.bridge_evidence = list(self.bridge_evidence)

    @classmethod
    def from_base(cls, state: EvidenceState, lookup: Optional[Dict[str, Any]] = None) -> "EvidenceStateV2":
        v2 = cls(base=state)
        if getattr(state, "active_anchors", None):
            v2.active_anchors = list(state.active_anchors)
        if getattr(state, "bridge_evidence", None):
            v2.bridge_evidence = list(state.bridge_evidence)
        if lookup is not None:
            v2.refresh_anchors(lookup)
        return v2


class EvidenceStateV2Builder:
    def __init__(self):
        self.inner = EvidenceStateBuilder()

    def build_from_row(self, row: Dict[str, Any], memories: List[MemoryRecord]) -> EvidenceStateV2:
        base = self.inner.build_from_row(row, memories)
        lookup = {_mid(m): m for m in memories}
        return EvidenceStateV2.from_base(base, lookup=lookup)

    def rebuild(
        self,
        row: Dict[str, Any],
        memories: List[MemoryRecord],
        evidence_ids: Set[str],
        *,
        baseline_ids: Optional[Set[str]] = None,
        retrieval_history: Optional[List[str]] = None,
        prior_transitions: Optional[List[Dict[str, Any]]] = None,
        step: int = 0,
        bridge_evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> EvidenceStateV2:
        base = self.inner.rebuild_from_evidence_ids(
            row,
            memories,
            evidence_ids,
            baseline_ids=baseline_ids,
            retrieval_history=retrieval_history,
            prior_transitions=prior_transitions,
            step=step,
        )
        lookup = {_mid(m): m for m in memories}
        v2 = EvidenceStateV2.from_base(base, lookup=lookup)
        if bridge_evidence is not None:
            v2.bridge_evidence = list(bridge_evidence)
            v2.base.bridge_evidence = list(bridge_evidence)
        return v2
