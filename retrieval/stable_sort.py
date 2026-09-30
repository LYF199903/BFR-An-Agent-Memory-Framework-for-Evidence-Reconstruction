"""Deterministic ranking helpers: higher score first, then memory_id."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple


def scored_key(score: Any, memory_id: Any) -> Tuple[float, str]:
    return (-float(score), str(memory_id))


def sort_scored_pairs(pairs: Iterable[Sequence[Any]]) -> List[Tuple[Any, Any]]:
    """pairs of (score, memory_id) → stable descending score, ascending id."""
    items = [(p[0], p[1]) for p in pairs]
    items.sort(key=lambda x: scored_key(x[0], x[1]))
    return items


def rank_ids(pairs: Iterable[Sequence[Any]], *, k: int | None = None) -> List[str]:
    ranked = [str(mid) for _, mid in sort_scored_pairs(pairs)]
    if k is None:
        return ranked
    return ranked[: int(k)]
