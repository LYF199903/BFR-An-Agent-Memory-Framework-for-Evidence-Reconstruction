#!/usr/bin/env python3
"""Evidence state reconstruction: structured tracking of what we know / don't know."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from experiments.fca_real_locomo.load_amem_cache import MemoryRecord
from experiments.fca_real_locomo.question_analyzer import extract_persons

from analysis.evidence_expansion import (
    _memory_lookup,
    _selected_ids,
    _top10_ids,
    uncovered_requirements,
)


def _tokens(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


def _overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), 1)


def _mem_text(m: MemoryRecord) -> str:
    return f"{m.content} {m.context} {' '.join(sorted(str(k) for k in (m.keywords or [])))}"


def _short(text: str, n: int = 120) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


@dataclass
class EvidenceState:
    """Structured evidence state for a QA episode (static + dynamic fields)."""

    question_id: str = ""
    question: str = ""
    covered_requirements: List[Dict[str, Any]] = field(default_factory=list)
    missing_requirements: List[Dict[str, Any]] = field(default_factory=list)
    evidence_confidence: float = 0.0
    supporting_memories: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    search_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_history: List[str] = field(default_factory=list)

    # Dynamic / agent fields (v6)
    belief_distribution: Dict[str, float] = field(default_factory=dict)
    state_transition_history: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_score: float = 1.0
    evidence_ids: List[str] = field(default_factory=list)
    baseline_evidence_ids: List[str] = field(default_factory=list)
    step: int = 0

    # Extra diagnostics
    coverage_rate: float = 0.0
    num_supporting_memories: int = 0
    entities: List[str] = field(default_factory=list)

    # Optional Dynamic Cue fields (v2; default empty — safe for old checkpoints)
    active_anchors: List[Dict[str, Any]] = field(default_factory=list)
    bridge_evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def snapshot_metrics(self) -> Dict[str, float]:
        return {
            "evidence_confidence": self.evidence_confidence,
            "uncertainty_score": self.uncertainty_score,
            "coverage_rate": self.coverage_rate,
            "num_missing": float(len(self.missing_requirements)),
            "num_support": float(len(self.supporting_memories)),
            "belief_entropy": _belief_entropy(self.belief_distribution),
        }

    def to_prompt_block(self, max_items: int = 4) -> str:
        """Compact textual rendering for state-aware query generation."""
        lines = [f"Question: {self.question}"]
        if self.covered_requirements:
            cov = "; ".join(
                f"{r.get('requirement_id')}:{_short(r.get('description', ''), 60)}"
                for r in self.covered_requirements[:max_items]
            )
            lines.append(f"Covered: {cov}")
        if self.missing_requirements:
            miss = "; ".join(
                f"{r.get('requirement_id')}:{_short(r.get('description', ''), 60)}"
                for r in self.missing_requirements[:max_items]
            )
            lines.append(f"Missing: {miss}")
        lines.append(f"EvidenceConfidence: {self.evidence_confidence:.2f}")
        lines.append(f"UncertaintyScore: {self.uncertainty_score:.2f}")
        if self.belief_distribution:
            top_b = sorted(self.belief_distribution.items(), key=lambda x: -x[1])[:max_items]
            lines.append("Beliefs: " + "; ".join(f"{k}:{v:.2f}" for k, v in top_b))
        if self.uncertainty:
            unc = "; ".join(
                f"{u.get('type')}:{_short(str(u.get('detail', '')), 50)}"
                for u in self.uncertainty[:max_items]
            )
            lines.append(f"Uncertainty: {unc}")
        if self.contradictions:
            conj = "; ".join(
                _short(str(c.get('detail', '')), 50) for c in self.contradictions[:max_items]
            )
            lines.append(f"Contradictions: {conj}")
        if self.search_hypotheses:
            hyp = "; ".join(
                _short(h.get("hypothesis", ""), 70) for h in self.search_hypotheses[:max_items]
            )
            lines.append(f"Hypotheses: {hyp}")
        if self.supporting_memories:
            sm = "; ".join(
                f"{m.get('memory_id')}:{_short(m.get('snippet', ''), 40)}"
                for m in self.supporting_memories[:3]
            )
            lines.append(f"Support: {sm}")
        return "\n".join(lines)


def _belief_entropy(dist: Dict[str, float]) -> float:
    if not dist:
        return 0.0
    vals = [max(v, 1e-12) for v in dist.values()]
    s = sum(vals)
    probs = [v / s for v in vals]
    import math
    return float(-sum(p * math.log(p) for p in probs))


def uncertainty_score_from_parts(
    confidence: float,
    missing: List[Dict[str, Any]],
    requirements: List[Dict[str, Any]],
    uncertainty_items: List[Dict[str, Any]],
    contradictions: List[Dict[str, Any]],
    belief_dist: Dict[str, float],
) -> float:
    """Scalar uncertainty in [0, 1]; higher = less resolved."""
    n_req = len(requirements) or 1
    miss_rate = len(missing) / n_req
    item_pen = min(len(uncertainty_items) / 6.0, 1.0)
    conj_pen = min(len(contradictions) / 3.0, 1.0)
    ent = _belief_entropy(belief_dist)
    # normalize entropy roughly by log(|beliefs|+1)
    import math
    ent_n = ent / max(math.log(len(belief_dist) + 1), 1e-6) if belief_dist else 0.5
    score = (
        0.35 * miss_rate
        + 0.25 * (1.0 - confidence)
        + 0.20 * item_pen
        + 0.10 * conj_pen
        + 0.10 * min(ent_n, 1.0)
    )
    return float(max(0.0, min(1.0, score)))


def build_belief_distribution(
    question: str,
    missing: List[Dict[str, Any]],
    selected: List[MemoryRecord],
    entities: List[str],
) -> Dict[str, float]:
    """
    Soft belief mass over missing requirement slots and entity/fact hypotheses.
    Belief rises when current evidence overlaps the slot weakly (candidate but unresolved).
    """
    beliefs: Dict[str, float] = {}
    q_toks = _tokens(question)

    for r in missing:
        rid = r.get("requirement_id") or "R?"
        desc = r.get("description", "")
        key = f"req:{rid}"
        toks = _tokens(desc)
        best = 0.0
        for m in selected:
            best = max(best, _overlap(toks, _tokens(_mem_text(m))))
        # Low overlap → high unresolved belief mass on this slot
        beliefs[key] = float(max(0.05, 1.0 - best))

    for ent in entities:
        key = f"entity:{ent}"
        hits = sum(1 for m in selected if ent.lower() in _mem_text(m).lower())
        # Missing entity evidence → higher belief that we still need facts about it
        beliefs[key] = float(1.0 / (1.0 + hits))

    # Fact hypotheses from question content words not well covered
    for tok in list(q_toks)[:8]:
        if tok in {"what", "which", "when", "where", "does", "have", "been"}:
            continue
        key = f"fact:{tok}"
        hits = sum(1 for m in selected if tok in _mem_text(m).lower())
        beliefs[key] = float(1.0 / (1.0 + hits))

    # Normalize
    s = sum(beliefs.values()) or 1.0
    return {k: v / s for k, v in beliefs.items()}


class EvidenceStateBuilder:
    """Build EvidenceState from question + FCA requirements + selected memories."""

    def __init__(self, max_support: int = 8, max_hypotheses: int = 6):
        self.max_support = max_support
        self.max_hypotheses = max_hypotheses

    def build(
        self,
        question: str,
        requirements: List[Dict[str, Any]],
        selected_memories: List[MemoryRecord],
        requirement_coverage: Optional[List[Dict[str, Any]]] = None,
        question_id: str = "",
        retrieval_history: Optional[List[str]] = None,
        row: Optional[Dict[str, Any]] = None,
    ) -> EvidenceState:
        requirements = requirements or []
        requirement_coverage = requirement_coverage or []
        retrieval_history = list(retrieval_history or [])
        row = row or {}

        covered_ids = {
            rc.get("requirement_id")
            for rc in requirement_coverage
            if rc.get("covered")
        }
        # Prefer explicit missing_requirements when present
        if row.get("missing_requirements"):
            missing_ids = {
                m.get("requirement_id") if isinstance(m, dict) else str(m)
                for m in row["missing_requirements"]
            }
            missing = [r for r in requirements if r.get("requirement_id") in missing_ids]
            covered = [r for r in requirements if r.get("requirement_id") in covered_ids]
        else:
            covered = [r for r in requirements if r.get("requirement_id") in covered_ids]
            missing = [r for r in requirements if r.get("requirement_id") not in covered_ids]

        # If no FCA coverage annotations, infer by lexical overlap with requirements
        if requirements and not requirement_coverage and not row.get("missing_requirements"):
            covered, missing = self._infer_coverage(requirements, selected_memories)

        support = self._supporting_memories(selected_memories, question, missing + covered)
        confidence = self._evidence_confidence(
            question, requirements, covered, missing, selected_memories, support,
        )
        uncertainty = self._uncertainty(question, missing, covered, selected_memories, confidence)
        contradictions = self._contradictions(selected_memories)
        hypotheses = self._search_hypotheses(
            question, missing, uncertainty, contradictions, selected_memories,
        )
        entities = list(extract_persons(question))
        beliefs = build_belief_distribution(question, missing, selected_memories, entities)
        unc_score = uncertainty_score_from_parts(
            confidence, missing, requirements, uncertainty, contradictions, beliefs,
        )

        evidence_ids = [m.memory_id for m in selected_memories]
        n_req = len(requirements) or 1
        return EvidenceState(
            question_id=question_id or row.get("question_id", ""),
            question=question,
            covered_requirements=[
                {
                    "requirement_id": r.get("requirement_id"),
                    "description": r.get("description", ""),
                    "type": r.get("type"),
                }
                for r in covered
            ],
            missing_requirements=[
                {
                    "requirement_id": r.get("requirement_id"),
                    "description": r.get("description", ""),
                    "type": r.get("type"),
                }
                for r in missing
            ],
            evidence_confidence=confidence,
            supporting_memories=support,
            uncertainty=uncertainty,
            contradictions=contradictions,
            search_hypotheses=hypotheses,
            retrieval_history=retrieval_history,
            belief_distribution=beliefs,
            state_transition_history=[],
            uncertainty_score=unc_score,
            evidence_ids=evidence_ids,
            baseline_evidence_ids=list(evidence_ids),
            step=0,
            coverage_rate=len(covered) / n_req,
            num_supporting_memories=len(support),
            entities=entities,
        )

    def rebuild_from_evidence_ids(
        self,
        row: Dict[str, Any],
        memories: List[MemoryRecord],
        evidence_ids: Set[str],
        baseline_ids: Optional[Set[str]] = None,
        retrieval_history: Optional[List[str]] = None,
        prior_transitions: Optional[List[Dict[str, Any]]] = None,
        step: int = 0,
    ) -> EvidenceState:
        """Rebuild EvidenceState after new evidence is added (state transition)."""
        lookup = _memory_lookup(memories)
        selected = [lookup[mid] for mid in sorted(evidence_ids) if mid in lookup]
        # Re-infer coverage from current evidence (dynamic), keeping FCA annotations as soft prior
        requirements = row.get("information_requirements") or []
        requirement_coverage = row.get("requirement_coverage") or []

        # Prefer dynamic inference from current evidence set
        if requirements:
            covered, missing = self._infer_coverage(requirements, selected)
        else:
            covered, missing = [], []

        # If FCA said covered and memory still present, keep covered
        fca_covered = {
            rc.get("requirement_id")
            for rc in requirement_coverage
            if rc.get("covered") and set(rc.get("memory_ids") or []) & evidence_ids
        }
        covered_ids = {r.get("requirement_id") for r in covered} | fca_covered
        covered = [r for r in requirements if r.get("requirement_id") in covered_ids]
        missing = [r for r in requirements if r.get("requirement_id") not in covered_ids]

        support = self._supporting_memories(selected, row.get("question", ""), missing + covered)
        confidence = self._evidence_confidence(
            row.get("question", ""), requirements, covered, missing, selected, support,
        )
        uncertainty = self._uncertainty(
            row.get("question", ""), missing, covered, selected, confidence,
        )
        contradictions = self._contradictions(selected)
        hypotheses = self._search_hypotheses(
            row.get("question", ""), missing, uncertainty, contradictions, selected,
        )
        entities = list(extract_persons(row.get("question", "")))
        beliefs = build_belief_distribution(row.get("question", ""), missing, selected, entities)
        unc_score = uncertainty_score_from_parts(
            confidence, missing, requirements, uncertainty, contradictions, beliefs,
        )
        base = list(baseline_ids) if baseline_ids is not None else list(evidence_ids)
        n_req = len(requirements) or 1
        return EvidenceState(
            question_id=row.get("question_id", ""),
            question=row.get("question", ""),
            covered_requirements=[
                {"requirement_id": r.get("requirement_id"), "description": r.get("description", ""), "type": r.get("type")}
                for r in covered
            ],
            missing_requirements=[
                {"requirement_id": r.get("requirement_id"), "description": r.get("description", ""), "type": r.get("type")}
                for r in missing
            ],
            evidence_confidence=confidence,
            supporting_memories=support,
            uncertainty=uncertainty,
            contradictions=contradictions,
            search_hypotheses=hypotheses,
            retrieval_history=list(retrieval_history or []),
            belief_distribution=beliefs,
            state_transition_history=list(prior_transitions or []),
            uncertainty_score=unc_score,
            evidence_ids=sorted(evidence_ids),
            baseline_evidence_ids=sorted(base),
            step=step,
            coverage_rate=len(covered) / n_req,
            num_supporting_memories=len(support),
            entities=entities,
        )

    def build_from_row(
        self,
        row: Dict[str, Any],
        memories: List[MemoryRecord],
        retrieval_history: Optional[List[str]] = None,
    ) -> EvidenceState:
        selected_ids = set(_selected_ids(row)) or set(_top10_ids(row))
        lookup = _memory_lookup(memories)
        selected = [lookup[mid] for mid in selected_ids if mid in lookup]
        # Preserve order roughly by selected_ids listing when available
        ordered_ids = list(_selected_ids(row)) or list(_top10_ids(row))
        selected_ordered = []
        seen = set()
        for mid in ordered_ids:
            if mid in lookup and mid not in seen:
                selected_ordered.append(lookup[mid])
                seen.add(mid)
        for m in selected:
            if m.memory_id not in seen:
                selected_ordered.append(m)

        return self.build(
            question=row.get("question", ""),
            requirements=row.get("information_requirements") or [],
            selected_memories=selected_ordered,
            requirement_coverage=row.get("requirement_coverage") or [],
            question_id=row.get("question_id", ""),
            retrieval_history=retrieval_history,
            row=row,
        )

    def _infer_coverage(
        self,
        requirements: List[Dict[str, Any]],
        selected: List[MemoryRecord],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        covered, missing = [], []
        for r in requirements:
            desc = r.get("description", "")
            toks = _tokens(desc)
            best = 0.0
            for m in selected:
                best = max(best, _overlap(toks, _tokens(_mem_text(m))))
            if best >= 0.25:
                covered.append(r)
            else:
                missing.append(r)
        return covered, missing

    def _supporting_memories(
        self,
        selected: List[MemoryRecord],
        question: str,
        requirements: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        q_toks = _tokens(question)
        req_toks = _tokens(" ".join(r.get("description", "") for r in requirements))
        scored = []
        for m in selected:
            text = _mem_text(m)
            toks = _tokens(text)
            score = 0.6 * _overlap(q_toks, toks) + 0.4 * _overlap(req_toks, toks)
            scored.append((score, m))
        scored.sort(key=lambda t: (-float(t[0]), str(getattr(t[1], "memory_id", ""))))
        out = []
        for score, m in scored[: self.max_support]:
            out.append({
                "memory_id": m.memory_id,
                "speaker": m.speaker or "",
                "timestamp": m.timestamp or "",
                "keywords": list(m.keywords or [])[:8],
                "snippet": _short(m.content, 140),
                "relevance": round(float(score), 3),
            })
        return out

    def _evidence_confidence(
        self,
        question: str,
        requirements: List[Dict[str, Any]],
        covered: List[Dict[str, Any]],
        missing: List[Dict[str, Any]],
        selected: List[MemoryRecord],
        support: List[Dict[str, Any]],
    ) -> float:
        n_req = len(requirements) or 1
        coverage = len(covered) / n_req
        avg_rel = (
            sum(s.get("relevance", 0) for s in support) / len(support) if support else 0.0
        )
        density = min(len(selected) / 10.0, 1.0)
        # Penalize many missing requirements
        miss_pen = len(missing) / n_req
        conf = 0.45 * coverage + 0.35 * avg_rel + 0.20 * density - 0.25 * miss_pen
        return float(max(0.0, min(1.0, conf)))

    def _uncertainty(
        self,
        question: str,
        missing: List[Dict[str, Any]],
        covered: List[Dict[str, Any]],
        selected: List[MemoryRecord],
        confidence: float,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for r in missing:
            items.append({
                "type": "missing_requirement",
                "requirement_id": r.get("requirement_id"),
                "detail": r.get("description", ""),
            })
        if confidence < 0.45:
            items.append({
                "type": "low_confidence",
                "detail": f"overall evidence confidence={confidence:.2f}",
            })
        if not selected:
            items.append({"type": "empty_evidence", "detail": "no selected memories"})
        # Entity mentioned in question but weakly present in evidence
        for ent in extract_persons(question):
            hits = sum(1 for m in selected if ent.lower() in _mem_text(m).lower())
            if hits == 0:
                items.append({
                    "type": "entity_gap",
                    "detail": f"entity '{ent}' not found in selected evidence",
                })
        if covered and missing:
            items.append({
                "type": "partial_coverage",
                "detail": f"{len(covered)} covered / {len(missing)} missing",
            })
        return items[:8]

    def _contradictions(self, selected: List[MemoryRecord]) -> List[Dict[str, Any]]:
        """Lightweight contradiction heuristics over selected evidence."""
        items: List[Dict[str, Any]] = []
        # Number conflicts: same nearby context words with different numbers
        num_patterns: Dict[str, List[Tuple[str, str]]] = {}
        for m in selected:
            content = m.content or ""
            for match in re.finditer(r"(\b[a-zA-Z]{3,}\b(?:\s+\b[a-zA-Z]{3,}\b){0,2}).{0,20}?(\d{1,4})", content):
                key = match.group(1).lower()
                num_patterns.setdefault(key, []).append((match.group(2), m.memory_id))
        for key, vals in num_patterns.items():
            nums = {v[0] for v in vals}
            if len(nums) >= 2:
                items.append({
                    "type": "numeric_conflict",
                    "detail": f"conflicting numbers for '{key}': {sorted(nums)}",
                    "memory_ids": sorted({v[1] for v in vals}),
                })

        # Negation vs affirmative keyword clash on shared keywords
        pos, neg = [], []
        for m in selected:
            text = (m.content or "").lower()
            if any(n in text for n in (" not ", " never ", " no ", " don't ", " doesn't ")):
                neg.append(m.memory_id)
            else:
                pos.append(m.memory_id)
        # Only flag if shared keywords between a negated and non-negated memory
        lookup = {m.memory_id: m for m in selected}
        for nid in neg[:3]:
            nk = set(lookup[nid].keywords or [])
            for pid in pos[:5]:
                pk = set(lookup[pid].keywords or [])
                shared = nk & pk
                if len(shared) >= 2:
                    items.append({
                        "type": "polarity_conflict",
                        "detail": f"shared keywords {sorted(shared)[:4]} across negated/affirmative memories",
                        "memory_ids": [nid, pid],
                    })
                    break
        return items[:5]

    def _search_hypotheses(
        self,
        question: str,
        missing: List[Dict[str, Any]],
        uncertainty: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]],
        selected: List[MemoryRecord],
    ) -> List[Dict[str, Any]]:
        hyps: List[Dict[str, Any]] = []
        entities = list(extract_persons(question))

        for r in missing:
            desc = r.get("description", "")
            hyps.append({
                "hypothesis": f"Retrieve facts for missing requirement: {desc}",
                "query_seed": f"{question} {desc}",
                "source": "missing_requirement",
                "requirement_id": r.get("requirement_id"),
            })
            for e in entities[:2]:
                hyps.append({
                    "hypothesis": f"Entity-focused search for {e} regarding: {_short(desc, 50)}",
                    "query_seed": f"{e} {desc}",
                    "source": "entity_missing_requirement",
                    "requirement_id": r.get("requirement_id"),
                })

        for u in uncertainty:
            if u.get("type") == "entity_gap":
                detail = str(u.get("detail", ""))
                m = re.search(r"'([^']+)'", detail)
                ent = m.group(1) if m else ""
                if ent:
                    hyps.append({
                        "hypothesis": f"Locate memories mentioning entity {ent}",
                        "query_seed": ent,
                        "source": "entity_gap",
                    })

        for c in contradictions:
            hyps.append({
                "hypothesis": f"Resolve contradiction: {c.get('detail', '')}",
                "query_seed": f"{question} {c.get('detail', '')}",
                "source": "contradiction",
            })

        # Bridge from current evidence keywords
        for m in selected[:4]:
            for kw in (m.keywords or [])[:2]:
                if kw and kw.lower() not in _tokens(question):
                    hyps.append({
                        "hypothesis": f"Expand via evidence keyword '{kw}'",
                        "query_seed": f"{question} {kw}",
                        "source": "evidence_keyword",
                    })

        # Deduplicate by query_seed
        seen = set()
        uniq = []
        for h in hyps:
            key = (h.get("query_seed") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            uniq.append(h)
            if len(uniq) >= self.max_hypotheses:
                break
        return uniq
