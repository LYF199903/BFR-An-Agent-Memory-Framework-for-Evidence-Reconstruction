#!/usr/bin/env python3
"""
Evidence-to-Cue Generator.

Question / missing-requirement / current-evidence → anchors + search hypotheses.
NEVER receives gold answer / gold evidence / gold ids.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set

from analysis.evidence_state_v2 import EvidenceStateV2, build_active_anchors
from analysis.parameterized_evidence_action import ParameterizedEvidenceAction
from experiments.fca_real_locomo.question_analyzer import extract_persons

# morph helpers used below — import after definitions via late binding in push
CUE_SOURCES = (
    "question_only",
    "question_requirement",
    "question_requirement_evidence",
)

STOP = {
    "the", "and", "for", "with", "that", "this", "from", "have", "been", "were",
    "what", "when", "where", "which", "about", "into", "your", "their", "they",
    "them", "said", "says", "just", "like", "also", "does", "will", "would",
    "could", "should", "really", "keep", "yeah", "okay", "hey", "wow", "both",
    "how", "who", "why", "did", "was", "are", "you", "got", "any",
}


def content_tokens(text: str, min_len: int = 3) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z'-]+", text or "")
    out = []
    for t in toks:
        if len(t) < min_len:
            continue
        if t.lower() in STOP:
            continue
        out.append(t)
    return out


def morph_expand(token: str) -> List[str]:
    """Observable morphological / substring expansions (no gold lexicon)."""
    t = token.lower()
    out = [token]
    # destress → stress; unhappy → happy
    for pref in ("de", "un", "re", "dis", "non"):
        if t.startswith(pref) and len(t) - len(pref) >= 4:
            out.append(t[len(pref) :])
    if t.endswith("ing") and len(t) > 5:
        out.append(t[:-3])
        out.append(t[:-3] + "e")
    if t.endswith("ed") and len(t) > 4:
        out.append(t[:-2])
    if t.endswith("tion") and len(t) > 6:
        out.append(t[:-4])
    # light related forms commonly needed for QA lexical gap (still not gold-specific)
    RELATED = {
        "destress": ["stress", "relief", "relax"],
        "stress": ["relief", "relax"],
        "relocate": ["move", "moved"],
        "participate": ["participating", "joined"],
    }
    if t in RELATED:
        out.extend(RELATED[t])
    # unique
    seen, uniq = set(), []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def expand_query_lexically(query: str) -> str:
    parts = []
    for tok in content_tokens(query, min_len=3):
        parts.extend(morph_expand(tok)[:3])
    # keep original query first
    return (query + " " + " ".join(parts)).strip()[:180]


class EvidenceCueGenerator:
    def __init__(
        self,
        *,
        cue_source: str = "question_requirement_evidence",
        max_hypotheses: int = 3,
        top_k: int = 8,
    ):
        if cue_source not in CUE_SOURCES:
            raise ValueError(f"cue_source must be one of {CUE_SOURCES}")
        self.cue_source = cue_source
        self.max_hypotheses = max_hypotheses
        self.top_k = top_k

    def generate(
        self,
        state: EvidenceStateV2,
        lookup: Dict[str, Any],
        *,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        k = int(k or self.max_hypotheses)
        q = state.question or ""
        missing = list(state.missing_requirements or [])
        target_req = ""
        if missing:
            target_req = str(missing[0].get("description") or missing[0].get("requirement_id") or "")

        q_persons = extract_persons(q)
        q_focus = content_tokens(q, min_len=4)
        req_focus = content_tokens(target_req, min_len=3)

        # Anchors
        if self.cue_source == "question_only":
            anchors = build_active_anchors(q, [], {})
        elif self.cue_source == "question_requirement":
            anchors = build_active_anchors(q, [], {})
            for tok in req_focus[:6]:
                anchors.append(
                    {
                        "type": "phrase",
                        "value": tok,
                        "source_memory_id": "",
                        "source": "requirement",
                        "confidence": 0.55,
                    }
                )
        else:
            anchors = build_active_anchors(q, state.evidence_ids, lookup)
            for tok in req_focus[:6]:
                anchors.append(
                    {
                        "type": "phrase",
                        "value": tok,
                        "source_memory_id": "",
                        "source": "requirement",
                        "confidence": 0.55,
                    }
                )

        # Filter low-quality anchors
        cleaned = []
        seen: Set[str] = set()
        for a in anchors:
            v = str(a.get("value") or "").strip()
            if not v or len(v) < 3 or v.lower() in STOP:
                continue
            if len(v.split()) > 6:
                continue
            key = f"{a.get('type')}:{v.lower()}"
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(a)
        anchors = cleaned

        entities = [a["value"] for a in anchors if a.get("type") == "entity"][:5]
        times = [a["value"] for a in anchors if a.get("type") == "time"][:3]
        events = [a["value"] for a in anchors if a.get("type") == "event"][:3]
        phrases = [a["value"] for a in anchors if a.get("type") in ("phrase", "topic")][:6]
        evidence_derived = [a for a in anchors if a.get("source") == "evidence"]

        hypotheses: List[Dict[str, Any]] = []
        actions: List[ParameterizedEvidenceAction] = []

        def push(op: str, query: str, view: str, anch: List[str], conf: float, src_ids: List[str]):
            query = " ".join(str(query).split())
            if op in ("query_rewrite", "hybrid", "temporal_expand") and query:
                query = expand_query_lexically(query)
            query = query[:180]
            if not query and op != "session_expand":
                return
            if len(hypotheses) >= 8:
                return
            h = {
                "target_requirement": target_req,
                "operation": op,
                "anchors": list(anch),
                "query": query,
                "source_memory_ids": list(src_ids),
                "confidence": conf,
                "retrieval_view": view,
            }
            hypotheses.append(h)
            actions.append(
                ParameterizedEvidenceAction(
                    type=op,
                    target_requirement=target_req,
                    anchors=list(anch),
                    query=query,
                    retrieval_view=view,
                    top_k=self.top_k,
                    source_memory_ids=list(src_ids),
                    confidence=conf,
                    hypothesis_id=f"h{len(hypotheses)}",
                )
            )

        # Always seed with question-centric queries (+ morph expansion)
        push("query_rewrite", q, "text", q_persons[:2] + q_focus[:3], 0.9, [])
        # Focused lexical query from persons + content tokens (helps when full question BM25 fails)
        if q_persons or q_focus:
            push(
                "hybrid",
                f"{' '.join(q_persons[:2])} {' '.join(q_focus[:5])}".strip(),
                "hybrid",
                q_persons[:2] + q_focus[:4],
                0.91,
                [],
            )
        if self.cue_source != "question_only" and (target_req or req_focus):
            push(
                "query_rewrite",
                f"{' '.join(q_persons[:2])} {' '.join(q_focus[:3])} {' '.join(req_focus[:4])}".strip(),
                "hybrid",
                q_persons[:2] + req_focus[:3],
                0.88,
                [],
            )
        for ent in q_persons[:2]:
            push("entity_expand", ent, "entity", [ent], 0.7, [])

        if self.cue_source == "question_requirement_evidence" and evidence_derived:
            ev_ent = [a["value"] for a in evidence_derived if a["type"] == "entity"][:3]
            ev_time = [a["value"] for a in evidence_derived if a["type"] == "time"][:2]
            ev_evt = [
                a["value"]
                for a in evidence_derived
                if a["type"] == "event" and a["value"].lower() not in STOP
            ][:2]
            ev_ph = [
                a["value"]
                for a in evidence_derived
                if a["type"] in ("phrase", "topic") and a["value"].lower() not in STOP
            ][:4]
            src_ids = [a.get("source_memory_id") or "" for a in evidence_derived if a.get("source_memory_id")]
            src_ids = [x for x in src_ids if x][:4]

            # Evidence → New Cue: combine evidence cue with question focus (not random evidence junk)
            combo = []
            for x in (ev_ent[:1] + q_focus[:2] + ev_ph[:2] + req_focus[:2] + ev_evt[:1] + ev_time[:1]):
                if x and x.lower() not in STOP and x not in combo:
                    combo.append(x)
            if combo:
                push("hybrid", " ".join(combo), "hybrid", combo[:5], 0.92, src_ids)

            # entity from evidence × question focus
            if ev_ent and q_focus:
                push(
                    "query_rewrite",
                    f"{ev_ent[0]} {' '.join(q_focus[:3])}",
                    "text",
                    [ev_ent[0]] + q_focus[:2],
                    0.86,
                    src_ids,
                )

            if ev_time:
                push(
                    "temporal_expand",
                    f"{' '.join(q_persons[:1] + q_focus[:2])} {ev_time[0]}",
                    "text_time",
                    q_persons[:1] + [ev_time[0]],
                    0.8,
                    src_ids,
                )

            if src_ids:
                push("session_expand", "", "session", ev_ent[:1], 0.75, src_ids[:1])

        # Rank: prefer question-preserving high-confidence hypotheses
        scored = []
        qtok = set(t.lower() for t in content_tokens(q))
        for h, a in zip(hypotheses, actions):
            score = float(h.get("confidence") or 0)
            qtoks_in = set(t.lower() for t in content_tokens(h.get("query") or ""))
            # reward overlap with question
            score += 0.2 * (len(qtok & qtoks_in) / max(len(qtok), 1))
            if h.get("source_memory_ids") and self.cue_source == "question_requirement_evidence":
                score += 0.1
            scored.append((score, h, a))
        scored.sort(key=lambda x: -x[0])
        top = scored[:k]

        out_h = [x[1] for x in top]
        out_a = [x[2] for x in top]
        for i, a in enumerate(out_a):
            a.hypothesis_id = f"h{i+1}"
            a.confidence = float(top[i][0])

        return {
            "active_anchors": anchors,
            "search_hypotheses": out_h,
            "actions": out_a,
            "target_requirement": target_req,
            "cue_source": self.cue_source,
        }
