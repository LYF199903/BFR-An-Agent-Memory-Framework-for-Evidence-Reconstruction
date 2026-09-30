"""Rule-based question analyzer and person-fact pool helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

NAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

SKIP_NAMES = {
    "What", "Which", "When", "Where", "Who", "How", "Why", "Does", "Did",
    "Has", "Have", "Are", "Was", "Were", "The", "And", "But", "For",
    "Speaker", "Image", "May", "June", "July", "August",
}

SLOT_PATTERNS = [
    (r"\bidentity\b",
     "identity", "topic", "what"),
    (r"relationship status|single|married|dating",
     "relationship", "text", "what"),
    (r"moved from|come from|came from|from .+ ago|previously lived|relocated from|hometown|origin|搬来|从哪里来|搬到这里之前",
     "moved_from", "location", "location"),
    (r"research|researched|studied|study|研究什么|学什么",
     "research_topic", "topic", "topic"),
    (r"visited|traveled|travelled|went to|been to|去过哪里",
     "visited_location", "location", "location"),
    (r"participated|attended|took part|做过什么|参加活动|partake|activities",
     "attended_event", "activity", "activity"),
    (r"volunteer|charity|mentoring",
     "volunteer_activity", "activity", "activity"),
    (r"\bcamped\b|camping|camp at",
     "camped_location", "location", "list"),
    (r"background|grew up|childhood|upbringing",
     "background", "text", "attribute"),
    (r"\bidentity\b|身份",
     "identity", "attribute", "what"),
    (r"how many times|how often|几次|多少次|number of times",
     "count_event", "number", "counting"),
    (r"\bboth\b|\bcommon\b|\bshare\b|\bshared\b|共同|都做过",
     "intersection", "activity", "intersection"),
    (r"\bwhen\b|what year|what date|什么时候",
     "event_time", "time", "temporal"),
    (r"\bbefore\b|\bafter\b|之前|之后",
     "temporal_order", "event", "temporal"),
]


def extract_persons(question: str, known_speakers: Optional[List[str]] = None) -> List[str]:
    found: List[str] = []
    if known_speakers:
        for sp in known_speakers:
            if sp and sp.lower() in question.lower():
                found.append(sp)
    for m in NAME_RE.finditer(question):
        name = m.group(1)
        if name in SKIP_NAMES:
            continue
        if name not in found:
            found.append(name)
    return found


def analyze_question(
    question: str,
    known_speakers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    ql = question.lower()
    persons = extract_persons(question, known_speakers)

    fact_slot = "general"
    answer_type = "text"
    question_type = "what"
    temporal = None

    for pattern, slot, atype, qtype in SLOT_PATTERNS:
        if re.search(pattern, ql, re.I):
            fact_slot = slot
            answer_type = atype
            question_type = qtype
            break

    if re.search(r"\bwhere\b", ql):
        question_type = "location"
        if fact_slot == "general":
            fact_slot = "location"
            answer_type = "location"
    elif re.search(r"\bwho\b", ql):
        question_type = "who"
    elif re.search(r"\bwhat\b", ql):
        question_type = "what"
    elif re.search(r"\bwhich\b", ql):
        question_type = "which"

    years = YEAR_RE.findall(question)
    if years:
        temporal = years[0] if isinstance(years[0], str) else str(years[0])
    else:
        ym = re.search(r"\b(19|20)\d{2}\b", question)
        if ym:
            temporal = ym.group(0)

    return {
        "persons": persons,
        "fact_slot": fact_slot,
        "answer_type": answer_type,
        "temporal": temporal,
        "question_type": question_type,
        "raw_question": question,
    }


def generate_typed_queries(question_info: Dict[str, Any]) -> List[str]:
    persons = question_info.get("persons") or []
    slot = question_info.get("fact_slot", "general")
    temporal = question_info.get("temporal")
    question = question_info.get("raw_question", "")
    queries: List[str] = []

    def _p(prefix_templates: List[str]) -> None:
        for person in persons:
            for tpl in prefix_templates:
                queries.append(tpl.format(person=person, year=temporal or ""))

    if slot == "moved_from":
        _p([
            "{person} moved from",
            "{person} came from",
            "{person} previously lived in",
            "{person} relocated from",
            "{person} origin",
            "{person} hometown",
        ])
    elif slot == "research_topic":
        _p([
            "{person} studied",
            "{person} researched",
            "{person} research topic",
            "{person} project about",
            "{person} interested in",
        ])
    elif slot == "identity":
        _p([
            "{person} identity",
            "{person} transgender",
            "{person} identifies as",
            "{person} gender",
        ])
    elif slot == "relationship":
        _p([
            "{person} relationship",
            "{person} single",
            "{person} dating",
            "{person} partner",
        ])
    elif slot == "visited_location":
        _p([
            "{person} visited",
            "{person} traveled to",
            "{person} went to",
            "{person} trip to",
        ])
    elif slot == "activity":
        _p([
            "{person} participated",
            "{person} attended",
            "{person} activity",
            "{person} event",
        ])
    elif slot == "count_event":
        event_kw = _extract_event_keyword(question)
        for person in persons:
            if event_kw:
                queries.extend([
                    f"{person} {event_kw} {temporal or ''}".strip(),
                    f"{person} went to {event_kw}",
                    f"{person} visited {event_kw}",
                    f"{person} in {temporal or ''}".strip(),
                ])
            else:
                queries.append(f"{person} how many times")
    elif slot == "intersection" and len(persons) >= 2:
        for person in persons[:3]:
            queries.extend([
                f"{person} volunteered",
                f"{person} charity",
                f"{person} activity",
                f"{person} both",
            ])
    elif slot in ("event_time", "temporal_order"):
        _p([
            "{person} when",
            "{person} date",
            "{person} before",
            "{person} after",
        ])
    else:
        for person in persons:
            queries.append(f"{person} {question}")

    # dedupe preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for q in queries:
        qn = q.strip()
        if qn and qn.lower() not in seen:
            seen.add(qn.lower())
            out.append(qn)
    return out


def _extract_event_keyword(question: str) -> str:
    ql = question.lower()
    for kw in ["beach", "concert", "trip", "hike", "visit", "game", "party", "wedding"]:
        if kw in ql:
            return kw
    return ""


def memory_mentions_person(mem: Any, person: str) -> bool:
    pl = person.lower()
    speaker = (getattr(mem, "speaker", None) or mem.get("speaker", "") if isinstance(mem, dict) else getattr(mem, "speaker", "")) or ""
    if speaker.lower() == pl:
        return True
    parts = []
    if isinstance(mem, dict):
        parts = [mem.get("content", ""), mem.get("context", ""), " ".join(mem.get("keywords") or []), " ".join(mem.get("tags") or [])]
    else:
        parts = [mem.content, mem.context, " ".join(mem.keywords or []), " ".join(mem.tags or [])]
    return pl in " ".join(parts).lower()


def build_person_pool(memories: List[Any], persons: List[str]) -> List[str]:
    if not persons:
        return []
    ids: List[str] = []
    seen: Set[str] = set()
    for mem in memories:
        mid = mem.memory_id if hasattr(mem, "memory_id") else mem["memory_id"]
        if any(memory_mentions_person(mem, p) for p in persons):
            if mid not in seen:
                seen.add(mid)
                ids.append(mid)
    return ids


def expand_by_session_and_links(
    seed_ids: List[str],
    memories: List[Any],
    window: int = 3,
) -> List[str]:
    id_to = {m.memory_id if hasattr(m, "memory_id") else m["memory_id"]: m for m in memories}
    idx_to_id = {}
    for i, m in enumerate(memories):
        mid = m.memory_id if hasattr(m, "memory_id") else m["memory_id"]
        idx_to_id[i] = mid

    session_groups: Dict[str, List[int]] = {}
    for i, m in enumerate(memories):
        sid = (m.session_id if hasattr(m, "session_id") else m.get("session_id", "")) or (m.timestamp if hasattr(m, "timestamp") else m.get("timestamp", ""))
        session_groups.setdefault(sid, []).append(i)

    expanded: Set[str] = set(seed_ids)
    for sid in seed_ids:
        if sid not in id_to:
            continue
        mem = id_to[sid]
        midx = mem.memory_index if hasattr(mem, "memory_index") else None

        # session ±window by memory_index proximity within same session bucket
        sess = (mem.session_id if hasattr(mem, "session_id") else mem.get("session_id", "")) or mem.timestamp
        indices = session_groups.get(sess, [])
        if midx is not None:
            for j in indices:
                if abs(j - midx) <= window:
                    expanded.add(idx_to_id[j])
        else:
            for j in indices:
                expanded.add(idx_to_id[j])

        # linked memories (may be indices)
        links = mem.links if hasattr(mem, "links") else mem.get("links", []) or []
        for link in links:
            try:
                li = int(link)
                if 0 <= li < len(memories):
                    expanded.add(idx_to_id[li])
            except (TypeError, ValueError):
                if isinstance(link, str) and link in id_to:
                    expanded.add(link)

    return list(expanded)


def _extract_event_cues(question: str) -> List[str]:
    ql = question.lower()
    cues: List[str] = []
    verb_map = {
        "camped": ["camped", "camping", "camp"],
        "camp": ["camp", "camping", "camped"],
        "moved": ["moved", "relocated", "move"],
        "visited": ["visited", "visit", "went"],
        "attended": ["attended", "attend", "participated"],
        "volunteer": ["volunteer", "volunteering", "charity"],
        "painted": ["painted", "paint", "painting"],
        "read": ["read", "reading", "book"],
        "traveled": ["traveled", "travelled", "travel", "trip"],
    }
    for key, variants in verb_map.items():
        if key in ql or any(v in ql for v in variants):
            cues.extend(variants[:2])
    for kw in ["beach", "mountain", "forest", "identity", "single", "married",
               "pottery", "swimming", "hiking", "concert", "wedding"]:
        if kw in ql:
            cues.append(kw)
    # dedupe
    seen: Set[str] = set()
    out: List[str] = []
    for c in cues:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            out.append(cl)
    return out


def _infer_requires_multiple(question: str, fact_slots: List[str]) -> bool:
    ql = question.lower()
    if any(w in ql for w in ["both", "common", "share", "all ", "activities", "events", "books", "countries"]):
        return True
    if "list" in fact_slots or any(s in fact_slots for s in ("camped_location", "visited_location", "count_event", "intersection")):
        return True
    if "," in question:
        return True
    return False


def extract_query_cues(
    question: str,
    known_speakers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extended query cue extraction for Chain-FCA."""
    base = analyze_question(question, known_speakers)
    ql = question.lower()

    fact_slot = base.get("fact_slot", "general")
    fact_slots = [fact_slot] if fact_slot != "general" else []
    if fact_slot == "location" and "camp" in ql:
        fact_slots = ["camped_location"]
    if "volunteer" in ql or "charity" in ql:
        if "volunteer_activity" not in fact_slots:
            fact_slots.append("volunteer_activity")
    if re.search(r"participated|attended", ql) and "attended_event" not in fact_slots:
        fact_slots.append("attended_event")

    entities = base.get("persons") or []
    event_cues = _extract_event_cues(question)
    temporal_cues: List[str] = []
    if base.get("temporal"):
        temporal_cues.append(str(base["temporal"]))
    for m in re.finditer(r"\b(19|20)\d{2}\b", question):
        y = m.group(0)
        if y not in temporal_cues:
            temporal_cues.append(y)

    answer_type = base.get("answer_type", "text")
    if fact_slot == "identity":
        answer_type = "identity"
    elif fact_slot in ("moved_from", "visited_location", "camped_location", "location"):
        answer_type = "location"
    elif fact_slot == "count_event":
        answer_type = "number"

    question_type = base.get("question_type", "what")
    if _infer_requires_multiple(question, fact_slots):
        question_type = "list" if question_type not in ("count", "counting") else question_type

    if not fact_slots:
        fact_slots = ["general"]

    return {
        "entities": entities,
        "event_cues": event_cues,
        "fact_slots": fact_slots,
        "answer_type": answer_type,
        "question_type": question_type,
        "temporal_cues": temporal_cues,
        "requires_multiple_evidence": _infer_requires_multiple(question, fact_slots),
        "raw_question": question,
        # backward compat
        "persons": entities,
        "fact_slot": fact_slots[0],
        "temporal": base.get("temporal"),
    }
