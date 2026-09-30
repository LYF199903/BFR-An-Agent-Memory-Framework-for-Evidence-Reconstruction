#!/usr/bin/env python3
"""Shared flat retrieval over LoCoMo turn pool (no graph indexing)."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 1]


def _session_key(m: Any) -> str:
    sid = getattr(m, "session_id", None) or (m.get("session_id") if isinstance(m, dict) else "") or ""
    ts = getattr(m, "timestamp", None) or (m.get("timestamp") if isinstance(m, dict) else "") or ""
    raw = str(sid or ts).strip().lower()
    if not raw:
        return ""
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _mid(m: Any) -> str:
    return str(getattr(m, "memory_id", None) or (m.get("memory_id") if isinstance(m, dict) else "") or "")


def _text(m: Any) -> str:
    if isinstance(m, dict):
        return f"{m.get('content') or ''} {m.get('context') or ''} {m.get('speaker') or ''}"
    return f"{getattr(m, 'content', '')} {getattr(m, 'context', '')} {getattr(m, 'speaker', '')}"


def _timestamp(m: Any) -> str:
    return str(getattr(m, "timestamp", None) or (m.get("timestamp") if isinstance(m, dict) else "") or "")


def _parse_date(ts: str) -> str:
    # keep trailing "13 June, 2023" style fragment if present
    s = str(ts or "")
    if " on " in s:
        return s.split(" on ", 1)[-1].strip()
    return s.strip()


@dataclass
class RetrievalLog:
    question_id: str
    step: int
    action: str
    query: str
    returned_memory_ids: List[str]
    scores: List[float]
    method: str = ""


@dataclass
class SharedFlatRetriever:
    """Common retrieval backend for Ours-Flat and MRAgent-Flat."""

    memories: Sequence[Any]
    top_k_default: int = 5
    logs: List[RetrievalLog] = field(default_factory=list)
    question_id: str = ""
    method: str = ""
    _step: int = 0

    def set_context(self, question_id: str, method: str) -> None:
        self.question_id = str(question_id)
        self.method = str(method)
        self._step = 0

    def _log(self, action: str, query: str, ids: List[str], scores: List[float]) -> None:
        self.logs.append(
            RetrievalLog(
                question_id=self.question_id,
                step=self._step,
                action=action,
                query=query,
                returned_memory_ids=list(ids),
                scores=list(scores),
                method=self.method,
            )
        )
        self._step += 1

    def search_text(self, query: str, top_k: Optional[int] = None, exclude: Optional[Set[str]] = None) -> List[str]:
        k = int(top_k or self.top_k_default)
        exclude = exclude or set()
        q = _tokens(query)
        if not q:
            self._log("search_text", query, [], [])
            return []
        df: Counter = Counter()
        docs = []
        for m in self.memories:
            toks = _tokens(_text(m))
            docs.append((_mid(m), toks))
            for t in set(toks):
                df[t] += 1
        n = max(len(docs), 1)
        scored = []
        for mid, toks in docs:
            if not mid or mid in exclude or not toks:
                continue
            tf = Counter(toks)
            score = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1.0 + n / (1.0 + df[t]))
                score += (1.0 + math.log(1.0 + tf[t])) * idf
            if score > 0:
                scored.append((score, mid))
        scored.sort(key=lambda x: (-float(x[0]), str(x[1])))
        ids = [mid for _, mid in scored[:k]]
        scores = [float(s) for s, _ in scored[:k]]
        self._log("search_text", query, ids, scores)
        return ids

    def search_entity(self, entity: str, top_k: Optional[int] = None, exclude: Optional[Set[str]] = None) -> List[str]:
        k = int(top_k or self.top_k_default)
        exclude = exclude or set()
        ent = str(entity or "").lower().strip()
        if not ent:
            self._log("search_entity", entity, [], [])
            return []
        scored = []
        for m in self.memories:
            mid = _mid(m)
            if not mid or mid in exclude:
                continue
            text = _text(m).lower()
            if ent in text:
                scored.append((text.count(ent), mid))
        scored.sort(key=lambda x: (-float(x[0]), str(x[1])))
        ids = [mid for _, mid in scored[:k]]
        scores = [float(s) for s, _ in scored[:k]]
        self._log("search_entity", entity, ids, scores)
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
        k = int(top_k or self.top_k_default)
        exclude = exclude or set()
        # Prefer anchor memory date; else match constraint substring in timestamps
        anchor_date = ""
        if anchor_memory_id:
            for m in self.memories:
                if _mid(m) == anchor_memory_id:
                    anchor_date = _parse_date(_timestamp(m))
                    break
        constraint = str(time_constraint or "").lower()
        scored = []
        for m in self.memories:
            mid = _mid(m)
            if not mid or mid in exclude:
                continue
            ts = _timestamp(m)
            md = _parse_date(ts)
            ok = False
            if anchor_date and md:
                if window == "same_day" and md == anchor_date:
                    ok = True
                elif window == "same_month" and md.split()[-1:] == anchor_date.split()[-1:]:
                    ok = True
            if constraint and constraint in ts.lower():
                ok = True
            if ok:
                scored.append((1.0, mid))
        scored.sort(key=lambda x: (-float(x[0]), str(x[1])))
        ids = [mid for _, mid in scored[:k]]
        scores = [1.0] * len(ids)
        q = f"{time_constraint}|{anchor_memory_id}|{window}"
        self._log("search_temporal", q, ids, scores)
        return ids

    def search_session(
        self,
        session_id: str,
        top_k: Optional[int] = None,
        exclude: Optional[Set[str]] = None,
        *,
        anchor_memory_id: str = "",
    ) -> List[str]:
        k = int(top_k or self.top_k_default)
        exclude = exclude or set()
        sid = str(session_id or "")
        if not sid and anchor_memory_id:
            for m in self.memories:
                if _mid(m) == anchor_memory_id:
                    sid = _session_key(m)
                    break
        ids = []
        for m in self.memories:
            mid = _mid(m)
            if not mid or mid in exclude:
                continue
            if _session_key(m) == sid or (sid and sid == str(getattr(m, "session_id", "") or "")):
                ids.append(mid)
            if len(ids) >= k:
                break
        self._log("search_session", sid, ids, [1.0] * len(ids))
        return ids

    def dump_logs(self) -> List[Dict[str, Any]]:
        return [
            {
                "question_id": x.question_id,
                "step": x.step,
                "action": x.action,
                "query": x.query,
                "returned_memory_ids": x.returned_memory_ids,
                "scores": x.scores,
                "method": x.method,
            }
            for x in self.logs
        ]
