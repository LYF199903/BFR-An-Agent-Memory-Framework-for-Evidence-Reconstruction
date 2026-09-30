#!/usr/bin/env python3
"""
Requirement-aware candidate router.

retrieved candidates → score → keep useful / bridge evidence → Evidence State update
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set


def _tokens(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


def _stems(toks: Set[str]) -> Set[str]:
    """Lightweight stemming / prefix expansion for destress↔stress, dancing↔dance."""
    out = set(toks)
    for t in list(toks):
        if t.endswith("ing") and len(t) > 5:
            out.add(t[:-3])
            out.add(t[:-3] + "e")
        if t.endswith("ed") and len(t) > 4:
            out.add(t[:-2])
            out.add(t[:-1])
        if t.endswith("s") and len(t) > 4:
            out.add(t[:-1])
        # substring bridges: destress contains stress
        if len(t) >= 6:
            out.add(t[2:])
            out.add(t[3:])
    return out


def _blob(m: Any) -> str:
    if isinstance(m, dict):
        return f"{m.get('content') or ''} {m.get('context') or ''} {m.get('speaker') or ''}"
    return f"{getattr(m, 'content', '')} {getattr(m, 'context', '')} {getattr(m, 'speaker', '')}"


def _overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), 1)


@dataclass
class EvidenceCandidateRouter:
    alpha: float = 0.40
    beta: float = 0.25
    gamma: float = 0.10
    delta: float = 0.25
    keep_top: int = 8
    retrieve_cap: int = 40
    min_score: float = 0.05
    # Always keep high-relevance hits even if not in top combined-score slice
    force_relevance: float = 0.10
    max_keep_hard: int = 20

    def score_candidate(
        self,
        memory: Any,
        *,
        question: str,
        missing_requirements: Sequence[Dict[str, Any]],
        current_evidence_ids: Set[str],
        current_evidence_text: str,
        memory_id: str,
    ) -> Dict[str, Any]:
        text = _blob(memory)
        qtok = _stems(_tokens(question))
        mtok = _stems(_tokens(text))
        etok = _stems(_tokens(current_evidence_text))

        relevance = _overlap(qtok, mtok)
        # substring soft match
        soft = 0.0
        for qt in list(qtok)[:12]:
            if len(qt) < 4:
                continue
            if any(qt in mt or mt in qt for mt in mtok if len(mt) >= 4):
                soft += 0.08
        relevance = min(1.0, relevance + soft)

        miss_desc = " ".join(
            str(r.get("description") or "") for r in (missing_requirements or [])[:4]
        )
        rtok = _stems(_tokens(miss_desc))
        coverage = _overlap(rtok, mtok) if rtok else 0.0

        novelty = 1.0 - _overlap(mtok, etok)
        if memory_id in current_evidence_ids:
            novelty = 0.0

        new_caps = set(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
        old_caps = set(re.findall(r"\b[A-Z][a-z]{2,}\b", current_evidence_text))
        months = {
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        }
        new_months = {w for w in re.findall(r"[A-Za-z]+", text) if w in months} - {
            w for w in re.findall(r"[A-Za-z]+", current_evidence_text) if w in months
        }
        bridge_ents = len(new_caps - old_caps)
        bridge_value = min(1.0, 0.2 * bridge_ents + 0.3 * len(new_months))
        bridge_kw = {
            "because", "after", "before", "since", "when", "why", "stopped",
            "started", "due", "stress", "relief", "passion", "escape",
        }
        if _tokens(text) & bridge_kw and novelty > 0.15:
            bridge_value = min(1.0, bridge_value + 0.15)

        score = (
            self.alpha * relevance
            + self.beta * coverage
            + self.gamma * novelty
            + self.delta * bridge_value
        )
        # Strong relevance should always be keep-eligible
        if relevance >= 0.25:
            score = max(score, 0.35)

        reason_parts = []
        if relevance >= 0.12:
            reason_parts.append("relevant")
        if coverage >= 0.12:
            reason_parts.append("covers_missing")
        if bridge_value >= 0.2:
            reason_parts.append("bridge")
        if novelty >= 0.35 and relevance >= 0.08:
            reason_parts.append("novel")
        if not reason_parts and score >= self.min_score:
            reason_parts.append("weak_match")

        return {
            "memory_id": memory_id,
            "relevance": round(relevance, 4),
            "coverage": round(coverage, 4),
            "novelty": round(novelty, 4),
            "bridge_value": round(bridge_value, 4),
            "score": round(score, 4),
            "keep": False,
            "reason": "|".join(reason_parts) or "low_score",
            "generated_cues": sorted(list((new_caps - old_caps)))[:6]
            + sorted(list(new_months))[:3],
        }

    def route(
        self,
        candidate_ids: Sequence[str],
        *,
        lookup: Dict[str, Any],
        question: str,
        missing_requirements: Sequence[Dict[str, Any]],
        current_evidence_ids: Sequence[str],
        keep_top: Optional[int] = None,
    ) -> Dict[str, Any]:
        keep_n = int(keep_top or self.keep_top)
        cur = set(current_evidence_ids)
        texts = []
        for mid in list(cur)[:20]:
            m = lookup.get(mid)
            if m:
                texts.append(_blob(m)[:300])
        cur_text = " ".join(texts)

        diagnostics = []
        for mid in list(candidate_ids)[: self.retrieve_cap]:
            if mid in cur:
                continue
            m = lookup.get(mid)
            if not m:
                continue
            diagnostics.append(
                self.score_candidate(
                    m,
                    question=question,
                    missing_requirements=missing_requirements,
                    current_evidence_ids=cur,
                    current_evidence_text=cur_text,
                    memory_id=mid,
                )
            )
        diagnostics.sort(key=lambda d: (-float(d["score"]), str(d["memory_id"])))
        kept = []
        kept_set = set()
        hard = int(self.max_keep_hard)

        # Tier 1: top by combined score
        for i, d in enumerate(diagnostics):
            if i < keep_n and float(d["score"]) >= self.min_score:
                d["keep"] = True
                kept.append(d["memory_id"])
                kept_set.add(d["memory_id"])

        # Tier 2: top by relevance (independent quota — avoids score-distractors crowding out)
        by_rel = sorted(diagnostics, key=lambda d: (-float(d["relevance"]), str(d["memory_id"])))
        rel_added = 0
        for d in by_rel:
            if len(kept) >= hard:
                break
            if d["memory_id"] in kept_set:
                continue
            if float(d["relevance"]) >= self.force_relevance:
                d["keep"] = True
                kept.append(d["memory_id"])
                kept_set.add(d["memory_id"])
                rel_added += 1
                if rel_added >= keep_n:
                    break

        # Tier 3: force any remaining with strong score/bridge until hard cap
        for d in sorted(diagnostics, key=lambda x: (-float(x["score"]), str(x["memory_id"]))):
            if len(kept) >= hard:
                break
            if d["memory_id"] in kept_set:
                continue
            if float(d["score"]) >= 0.22 or (
                float(d["bridge_value"]) >= 0.30 and float(d["relevance"]) >= 0.10
            ):
                d["keep"] = True
                kept.append(d["memory_id"])
                kept_set.add(d["memory_id"])

        # Tier 4: reachability-first — keep next-best by relevance down to 0.08
        for d in by_rel:
            if len(kept) >= hard:
                break
            if d["memory_id"] in kept_set:
                continue
            if float(d["relevance"]) >= 0.08:
                d["keep"] = True
                kept.append(d["memory_id"])
                kept_set.add(d["memory_id"])

        for d in diagnostics:
            if d["memory_id"] not in kept_set:
                d["keep"] = False

        # Guarantee: always keep top-1 by relevance if empty
        if not kept and diagnostics:
            best_rel = sorted(
                diagnostics, key=lambda d: (-float(d["relevance"]), str(d["memory_id"]))
            )[0]
            if float(best_rel["relevance"]) >= 0.1:
                best_rel["keep"] = True
                kept.append(best_rel["memory_id"])

        bridge = [
            {
                "memory_id": d["memory_id"],
                "reason": d["reason"],
                "generated_cues": d.get("generated_cues") or [],
            }
            for d in diagnostics
            if d["keep"] and "bridge" in (d.get("reason") or "")
        ]
        return {
            "kept_ids": kept,
            "diagnostics": diagnostics,
            "bridge_evidence": bridge,
        }
