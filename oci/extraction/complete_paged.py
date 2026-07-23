"""Lossless complete-note paging and citation/reconciliation validation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

COMPLETE_PAGED_VERSION = "complete_paged_v1"


@dataclass(frozen=True)
class CompleteNotePage:
    page_index: int
    core_start: int
    core_end: int
    context_start: int
    context_end: int
    text_sha256: str
    core_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plan_complete_note_pages(
    text: str, *, core_chars: int = 13_488, context_chars: int = 256,
    max_page_chars: int = 14_000,
) -> tuple[CompleteNotePage, ...]:
    if core_chars < 1 or context_chars < 0 or core_chars + 2 * context_chars > max_page_chars:
        raise ValueError("invalid complete-note page geometry")
    pages: list[CompleteNotePage] = []
    for index, start in enumerate(range(0, len(text), core_chars)):
        end = min(len(text), start + core_chars)
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        rendered = text[context_start:context_end]
        if len(rendered) > max_page_chars:
            raise RuntimeError("complete-note page exceeds its hard character bound")
        pages.append(CompleteNotePage(index, start, end, context_start, context_end, _sha(rendered), _sha(text[start:end])))
    validate_complete_note_page_plan(text, pages, max_page_chars=max_page_chars)
    return tuple(pages)


def validate_complete_note_page_plan(
    text: str, pages: Sequence[CompleteNotePage], *, max_page_chars: int = 14_000,
) -> None:
    expected = 0
    for index, page in enumerate(pages):
        if page.page_index != index or page.core_start != expected:
            raise ValueError("page cores are missing, duplicated, or reordered")
        if not (page.context_start <= page.core_start < page.core_end <= page.context_end):
            raise ValueError("invalid core/context page bounds")
        rendered = text[page.context_start:page.context_end]
        if len(rendered) > max_page_chars or _sha(rendered) != page.text_sha256:
            raise ValueError("page text changed or exceeds its bound")
        if _sha(text[page.core_start:page.core_end]) != page.core_sha256:
            raise ValueError("page core changed")
        expected = page.core_end
    if expected != len(text) or (text and not pages):
        raise ValueError("page cores do not cover the complete note exactly once")


def validate_absolute_citations(
    text: str, citations: Sequence[Mapping[str, Any]], *, page: CompleteNotePage | None = None,
) -> tuple[dict[str, Any], ...]:
    unique: dict[tuple[int, int], dict[str, Any]] = {}
    for citation in citations:
        start, end = int(citation.get("start", -1)), int(citation.get("end", -1))
        quote = citation.get("text")
        if not isinstance(quote, str) or not (0 <= start < end <= len(text)) or text[start:end] != quote:
            raise ValueError("citation cannot be located at its exact absolute offset")
        if page is not None and not (page.context_start <= start and end <= page.context_end):
            raise ValueError("citation lies outside the page supplied to the model")
        unique[(start, end)] = {"start": start, "end": end, "text": quote, "sha256": _sha(quote)}
    return tuple(unique[key] for key in sorted(unique))


def bounded_recursive_reconcile(
    results: Sequence[Any], reducer: Callable[[Sequence[Any]], Any], *, fan_in: int = 16,
) -> tuple[Any, Mapping[str, Any]]:
    if fan_in < 2 or not results:
        raise ValueError("reconciliation needs at least one child and fan_in >= 2")
    level = list(results)
    ledger: list[dict[str, Any]] = []
    depth = 0
    while len(level) > 1:
        next_level = []
        for start in range(0, len(level), fan_in):
            children = level[start:start + fan_in]
            reduced = reducer(children)
            ledger.append({"depth": depth, "child_start": start, "child_count": len(children)})
            next_level.append(reduced)
        level = next_level
        depth += 1
    return level[0], {
        "schema_version": COMPLETE_PAGED_VERSION,
        "leaf_count": len(results), "fan_in": fan_in, "reductions": ledger,
        "every_child_accounted_for_exactly_once": True,
    }


__all__ = ["COMPLETE_PAGED_VERSION", "CompleteNotePage", "bounded_recursive_reconcile", "plan_complete_note_pages", "validate_absolute_citations", "validate_complete_note_page_plan"]
