#!/usr/bin/env python3
"""
Multi-view flat retriever over raw LoCoMo turns.

NO graph edges / neighbor traversal / associative index.
Each memory is an independent searchable record with multiple fields.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from retrieval.shared_flat_retriever import (
    SharedFlatRetriever,
    _mid,
    _parse_date,
    _session_key,
    _text,
    _timestamp,
    _tokens,
)

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)


def _normalize_text(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _speaker(m: Any) -> str:
    sp = getattr(m, "speaker", None) or (m.get("speaker") if isinstance(m, dict) else "") or ""
    sp = str(sp)
    if ":" in sp:
        sp = sp.split(":", 1)[0].strip()
    # also parse "Name: ..." from content
    if not sp:
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "") or ""
        m2 = re.match(r"^([A-Z][a-zA-Z]+)\s*:", str(content))
        if m2:
            sp = m2.group(1)
    return sp


def _entities_from_memory(m: Any) -> List[str]:
    text = _text(m)
    ents = []
    sp = _speaker(m)
    if sp:
        ents.append(sp)
    for name in re.findall(r"\b([A-Z][a-z]{2,})\b", text):
        if name.lower() not in {"the", "and", "for", "with", "this", "that", "hey"}:
            ents.append(name)
    kws = getattr(m, "keywords", None) or (m.get("keywords") if isinstance(m, dict) else []) or []
    ents.extend(str(k) for k in kws[:5])
    # dedupe
    seen, out = set(), []
    for e in ents:
        k = e.lower()
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def _event_phrase(m: Any) -> str:
    text = _text(m)
    verbs = {
        "stop", "stopped", "start", "started", "lost", "won", "joined", "moved",
        "visited", "opened", "launched", "graduated", "bought", "sold", "met",
    }
    toks = re.findall(r"[A-Za-z']+", text)
    for i, t in enumerate(toks):
        if t.lower() in verbs:
            return " ".join(toks[i : i + 4])[:80]
    return ""


@dataclass
class MemoryView:
    memory_id: str
    raw_text: str
    normalized_text: str
    speaker: str
    entities: List[str]
    time: str
    event: str
    session_id: str
    descriptor: str = ""


@dataclass
class MultiViewFlatRetriever:
    """Flat multi-field retrieval; wraps SharedFlatRetriever for base searches."""

    memories: Sequence[Any]
    top_k_default: int = 8
    logs: List[Dict[str, Any]] = field(default_factory=list)
    question_id: str = ""
    method: str = "dynamic_cue"
    _step: int = 0
    _views: Dict[str, MemoryView] = field(default_factory=dict)
    _base: Optional[SharedFlatRetriever] = None

    def __post_init__(self):
        self._base = SharedFlatRetriever(self.memories, top_k_default=self.top_k_default)
        self._views = {}
        for m in self.memories:
            mid = _mid(m)
            if not mid:
                continue
            raw = _text(m)
            ts = _timestamp(m)
            self._views[mid] = MemoryView(
                memory_id=mid,
                raw_text=raw,
                normalized_text=_normalize_text(raw),
                speaker=_speaker(m),
                entities=_entities_from_memory(m),
                time=_parse_date(ts) or ts,
                event=_event_phrase(m),
                session_id=_session_key(m),
                descriptor=" ".join(
                    [
                        _speaker(m),
                        _parse_date(ts),
                        " ".join(_entities_from_memory(m)[:3]),
                        (_event_phrase(m) or "")[:40],
                    ]
                ).strip(),
            )

    def set_context(self, question_id: str, method: str = "dynamic_cue") -> None:
        self.question_id = str(question_id)
        self.method = str(method)
        self._step = 0
        if self._base:
            self._base.set_context(question_id, method)

    def _log(self, action: str, query: str, ids: List[str], scores: List[float]) -> None:
        self.logs.append(
            {
                "question_id": self.question_id,
                "step": self._step,
                "action": action,
                "query": query,
                "returned_memory_ids": list(ids),
                "scores": list(scores),
                "method": self.method,
            }
        )
        self._step += 1
        if self._base:
            # keep shared log in sync for dump compatibility
            pass

    def search_text(self, query: str, top_k: Optional[int] = None, exclude: Optional[Set[str]] = None) -> List[str]:
        assert self._base is not None
        ids = self._base.search_text(query, top_k=top_k, exclude=exclude)
        # also match descriptor lightly
        k = int(top_k or self.top_k_default)
        exclude = exclude or set()
        qtoks = set(_tokens(query))
        extra = []
        for mid, v in self._views.items():
            if mid in exclude or mid in ids:
                continue
            dtoks = set(_tokens(v.descriptor + " " + v.normalized_text))
            ov = len(qtoks & dtoks)
            if ov >= 2:
                extra.append((ov, mid))
        extra.sort(key=lambda x: (-float(x[0]), str(x[1])))
        for _, mid in extra:
            if mid not in ids:
                ids.append(mid)
            if len(ids) >= k:
                break
        ids = ids[:k]
        self._log("search_text", query, ids, [1.0] * len(ids))
        return ids

    def search_entity(self, entity: str, top_k: Optional[int] = None, exclude: Optional[Set[str]] = None) -> List[str]:
        k = int(top_k or self.top_k_default)
        exclude = exclude or set()
        ent = str(entity or "").lower().strip()
        if not ent:
            self._log("search_entity", entity, [], [])
            return []
        scored = []
        for mid, v in self._views.items():
            if mid in exclude:
                continue
            blob = (v.raw_text + " " + " ".join(v.entities) + " " + v.speaker).lower()
            if ent in blob:
                scored.append((blob.count(ent) + (2 if ent in v.speaker.lower() else 0), mid))
        scored.sort(key=lambda x: (-float(x[0]), str(x[1])))
        ids = [m for _, m in scored[:k]]
        self._log("search_entity", entity, ids, [float(s) for s, _ in scored[:k]])
        return ids

    def search_temporal(
        self,
        time_constraint: str,
        top_k: Optional[int] = None,
        exclude: Optional[Set[str]] = None,
        *,
        anchor_memory_id: str = "",
        window: str = "same_day",
    ) -> List[str]:
        assert self._base is not None
        ids = self._base.search_temporal(
            time_constraint,
            top_k=top_k,
            exclude=exclude,
            anchor_memory_id=anchor_memory_id,
            window=window,
        )
        # also match month name in view.time / text
        k = int(top_k or self.top_k_default)
        exclude = exclude or set()
        constraint = str(time_constraint or "").lower()
        months = [m for m in MONTHS if m in constraint]
        extra = []
        for mid, v in self._views.items():
            if mid in exclude or mid in ids:
                continue
            vt = (v.time + " " + v.raw_text).lower()
            ok = False
            if constraint and constraint in vt:
                ok = True
            for mon in months:
                if mon in vt:
                    ok = True
            if ok:
                extra.append(mid)
            if len(ids) + len(extra) >= k:
                break
        for mid in extra:
            if mid not in ids:
                ids.append(mid)
        ids = ids[:k]
        self._log("search_temporal", f"{time_constraint}|{anchor_memory_id}|{window}", ids, [1.0] * len(ids))
        return ids

    def search_session(
        self,
        session_id: str = "",
        top_k: Optional[int] = None,
        exclude: Optional[Set[str]] = None,
        *,
        anchor_memory_id: str = "",
    ) -> List[str]:
        assert self._base is not None
        ids = self._base.search_session(
            session_id, top_k=top_k, exclude=exclude, anchor_memory_id=anchor_memory_id
        )
        self._log("search_session", session_id or anchor_memory_id, ids, [1.0] * len(ids))
        return ids

    def search_hybrid(
        self,
        query: str,
        *,
        entities: Optional[Sequence[str]] = None,
        time_constraint: str = "",
        top_k: Optional[int] = None,
        exclude: Optional[Set[str]] = None,
        anchor_memory_id: str = "",
    ) -> List[str]:
        """Combine text + entity + temporal signals (still flat; no edges)."""
        k = int(top_k or self.top_k_default)
        exclude = exclude or set()
        scores: Counter = Counter()
        for mid in self.search_text(query, top_k=k * 2, exclude=exclude):
            scores[mid] += 3.0
        for ent in list(entities or [])[:3]:
            for mid in self.search_entity(ent, top_k=k, exclude=exclude):
                scores[mid] += 2.0
        if time_constraint or anchor_memory_id:
            for mid in self.search_temporal(
                time_constraint,
                top_k=k,
                exclude=exclude,
                anchor_memory_id=anchor_memory_id,
                window="same_month" if not anchor_memory_id else "same_day",
            ):
                scores[mid] += 2.5
        # descriptor overlap
        qtoks = set(_tokens(query))
        for mid, v in self._views.items():
            if mid in exclude:
                continue
            ov = len(qtoks & set(_tokens(v.descriptor)))
            if ov:
                scores[mid] += 0.5 * ov
        ranked = [
            mid
            for mid, _sc in sorted(scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
            if mid not in exclude
        ][:k]
        self._log("search_hybrid", query, ranked, [float(scores[m]) for m in ranked])
        return ranked

    def execute_action(
        self,
        action: Any,
        exclude: Optional[Set[str]] = None,
    ) -> List[str]:
        """Execute a ParameterizedEvidenceAction or dict."""
        if hasattr(action, "to_dict"):
            d = action.to_dict()
        else:
            d = dict(action)
        view = str(d.get("retrieval_view") or "text")
        q = str(d.get("query") or "")
        k = int(d.get("top_k") or self.top_k_default)
        anchors = [str(a) for a in (d.get("anchors") or [])]
        src = [str(x) for x in (d.get("source_memory_ids") or [])]
        atype = str(d.get("type") or "query_rewrite")
        exclude = exclude or set()

        if atype == "session_expand" or view == "session":
            return self.search_session("", top_k=k, exclude=exclude, anchor_memory_id=src[0] if src else "")
        if atype == "entity_expand" or view == "entity":
            ent = anchors[0] if anchors else q
            return self.search_entity(ent, top_k=k, exclude=exclude)
        if atype == "temporal_expand" or view in ("time", "text_time"):
            tc = " ".join(a for a in anchors if any(m in a.lower() for m in MONTHS)) or q
            return self.search_temporal(
                tc,
                top_k=k,
                exclude=exclude,
                anchor_memory_id=src[0] if src else "",
                window="same_month",
            )
        if atype == "hybrid" or view == "hybrid":
            times = [a for a in anchors if any(m in a.lower() for m in MONTHS)]
            ents = [a for a in anchors if a not in times]
            return self.search_hybrid(
                q or " ".join(anchors),
                entities=ents,
                time_constraint=times[0] if times else "",
                top_k=k,
                exclude=exclude,
                anchor_memory_id=src[0] if src else "",
            )
        return self.search_text(q or " ".join(anchors), top_k=k, exclude=exclude)

    def dump_logs(self) -> List[Dict[str, Any]]:
        return list(self.logs)
