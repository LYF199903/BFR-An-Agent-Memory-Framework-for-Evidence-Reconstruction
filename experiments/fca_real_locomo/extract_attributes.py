"""Rule-based attribute extraction for FCA reranking on real LoCoMo memories."""

from __future__ import annotations

import re
from typing import Dict, List, Set, Optional

GENERIC_TERMS = {
    "conversation", "discussion", "experience", "support", "personal",
    "topic", "memory", "interaction", "general", "activity",
    "event", "information", "context", "talk", "friend", "communication",
    "social", "dialogue", "greeting", "updates", "enthusiasm", "passion",
    "inspiration", "positivity", "projects", "informal", "career", "employment",
    "entrepreneurship", "transition", "relationship", "engagement",
}

EVENT_WHITELIST = {
    "develop", "developed", "create", "created", "make", "made", "build", "built",
    "paint", "painted", "go", "went", "visit", "visited", "attend", "attended",
    "start", "started", "open", "opened", "lose", "lost", "get", "got", "join",
    "joined", "move", "moved", "travel", "traveled", "buy", "bought", "sell",
    "sold", "learn", "learned", "teach", "taught", "perform", "performed",
    "host", "hosted", "expand", "expanded", "recolor", "keep", "present", "presentation",
    "video", "show", "demonstrate", "graduate", "marry", "divorce", "celebrate",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "what", "when", "where", "who", "how", "which", "why",
    "that", "this", "these", "those", "it", "its", "they", "them", "their", "she",
    "he", "her", "his", "my", "your", "our", "me", "you", "we", "i", "about",
    "any", "all", "some", "than", "then", "into", "over", "after", "before",
}

MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

DATE_RE = re.compile(
    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December),?\s+(\d{4})",
    re.I,
)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
NAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'-]{1,}", text.lower())


def _add_kw_attrs(attrs: Set[str], text: str, prefix: str = "kw") -> None:
    for tok in _tokenize(text):
        if tok in STOPWORDS or tok in GENERIC_TERMS or len(tok) < 3:
            continue
        attrs.add(f"{prefix}:{tok}")


def _extract_dates(text: str) -> Set[str]:
    found = set()
    for m in DATE_RE.finditer(text):
        day, month, year = m.group(1), m.group(2), m.group(3)
        mon = MONTH_MAP.get(month.lower(), "")
        if mon:
            found.add(f"date:{year}-{mon}-{int(day):02d}")
            found.add(f"month:{year}-{mon}")
            found.add(f"year:{year}")
    for y in YEAR_RE.findall(text):
        pass
    for m in re.finditer(r"\b(19|20)\d{2}\b", text):
        found.add(f"year:{m.group(0)}")
    return found


def _extract_entities(text: str, known_speakers: Optional[List[str]] = None) -> Set[str]:
    ents = set()
    if known_speakers:
        for sp in known_speakers:
            if sp and sp.lower() in text.lower():
                ents.add(f"entity:{sp}")
                ents.add(f"speaker:{sp}")
    for m in NAME_RE.finditer(text):
        name = m.group(1)
        if name.lower() not in {"speaker", "image", "may", "when", "what", "where"}:
            ents.add(f"entity:{name}")
    return ents


def _extract_events(text: str) -> Set[str]:
    ev = set()
    for tok in _tokenize(text):
        if tok in EVENT_WHITELIST:
            ev.add(f"event:{tok}")
    return ev


def extract_memory_attributes(
    memory: Dict,
    known_speakers: Optional[List[str]] = None,
) -> Set[str]:
    """Extract attribute set from a MemoryRecord dict or object."""
    attrs: Set[str] = set()

    def _get(k, default=""):
        if isinstance(memory, dict):
            return memory.get(k, default)
        return getattr(memory, k, default)

    speaker = _get("speaker", "")
    if speaker:
        attrs.add(f"speaker:{speaker}")
        attrs.add(f"entity:{speaker}")

    ts = _get("timestamp", "") or _get("date", "")
    attrs.update(_extract_dates(ts))

    for field, prefix in [("content", "kw"), ("context", "ctx")]:
        text = _get(field, "") or ""
        attrs.update(_extract_dates(text))
        attrs.update(_extract_entities(text, known_speakers))
        attrs.update(_extract_events(text))
        _add_kw_attrs(attrs, text, prefix=prefix)

    for kw in _get("keywords", []) or []:
        k = str(kw).lower().strip()
        if k and k not in GENERIC_TERMS:
            attrs.add(f"kw:{k.replace(' ', '_')}")

    for tag in _get("tags", []) or []:
        t = str(tag).lower().strip()
        if t and t not in GENERIC_TERMS:
            attrs.add(f"tag:{t.replace(' ', '_')}")

    return attrs


def extract_query_attributes(
    question: str,
    category: int,
    known_speakers: Optional[List[str]] = None,
) -> Set[str]:
    attrs: Set[str] = set()
    q_lower = question.lower()

    cat_map = {1: "multihop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
    attrs.add(f"qtype:{cat_map.get(category, 'unknown')}")

    if category == 2 or any(w in q_lower for w in ["when", "what year", "what date", "how long ago"]):
        attrs.add("ask:time")
    if category == 1:
        attrs.add("ask:multihop")
    if any(w in q_lower for w in ["who", "whose"]):
        attrs.add("ask:person")
    if any(w in q_lower for w in ["where", "location"]):
        attrs.add("ask:place")
    if any(w in q_lower for w in ["what", "which"]):
        attrs.add("ask:what")

    attrs.update(_extract_dates(question))
    attrs.update(_extract_entities(question, known_speakers))
    attrs.update(_extract_events(question))
    _add_kw_attrs(attrs, question, prefix="kw")

    return attrs


def is_generic_attr(attr: str) -> bool:
    if attr.startswith("kw:"):
        return attr.split(":", 1)[1] in GENERIC_TERMS
    if attr.startswith("tag:"):
        return attr.split(":", 1)[1].replace("_", " ") in GENERIC_TERMS
    if attr.startswith("ctx:"):
        return attr.split(":", 1)[1] in GENERIC_TERMS
    return False


def structural_overlap(query_attrs: Set[str], mem_attrs: Set[str]) -> Set[str]:
    """Shared non-generic structural attrs."""
    shared = query_attrs & mem_attrs
    return {a for a in shared if not is_generic_attr(a)}
