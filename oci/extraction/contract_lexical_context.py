"""Deterministic, contract-only lexical context selection for extraction.

The selector is deliberately label-free and model-free.  Its query vocabulary
comes exclusively from the explicit feature contracts supplied for the current
request; it never uses a clinical ontology or dataset-specific concept list.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from ..config import ExplicitFeatureSpec

CONTRACT_LEXICAL_CONTEXT_VERSION = "contract_lexical_rag_v1"
EXTRACTION_GROUPING_VERSION = "explicit_feature_request_grouping_v2"
CONTRACT_LEXICAL_CHUNK_CHARS = 1200
CONTRACT_LEXICAL_CHUNK_OVERLAP_CHARS = 120

_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_GENERIC_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "category",
        "clinical",
        "continuous",
        "did",
        "do",
        "does",
        "domain",
        "feature",
        "field",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "may",
        "not",
        "object",
        "of",
        "on",
        "or",
        "parent",
        "recorded",
        "reported",
        "the",
        "this",
        "to",
        "type",
        "unknown",
        "value",
        "was",
        "were",
        "with",
    }
)


@dataclass(frozen=True)
class RetrievedContractExcerpt:
    """One verbatim span selected from the source document."""

    start: int
    end: int
    score: float


@dataclass(frozen=True)
class ContractLexicalContext:
    """Versioned compact context and its deterministic audit metadata."""

    text: str
    version: str
    original_char_count: int
    max_chars: int
    query_sha256: str
    query_tokens: tuple[str, ...]
    query_phrases: tuple[str, ...]
    selected_excerpts: tuple[RetrievedContractExcerpt, ...]
    fallback_tail_used: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Chunk:
    start: int
    end: int
    text: str
    tokens: tuple[str, ...]
    normalized_text: str
    score: float = 0.0


def _tokens(value: Any) -> tuple[str, ...]:
    text = str(value or "").replace("_", " ").lower()
    return tuple(token for token in _TOKEN.findall(text) if token not in _GENERIC_STOPWORDS)


def _contract_strings(specs: Sequence[ExplicitFeatureSpec]) -> list[str]:
    values: list[str] = []
    for spec in specs:
        values.append(str(spec.name))
        if spec.description:
            values.append(str(spec.description))
        values.extend(str(value) for value in (spec.categories or []))
        aliases = spec.value_aliases or {}
        if isinstance(aliases, Mapping):
            for category in sorted(aliases, key=str):
                values.append(str(category))
                raw_values = aliases[category]
                alias_values = raw_values if isinstance(raw_values, (list, tuple)) else [raw_values]
                values.extend(sorted((str(value) for value in alias_values), key=str))
    return values


def _contract_query(
    specs: Sequence[ExplicitFeatureSpec],
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    token_set: set[str] = set()
    phrase_set: set[str] = set()
    contract_values = _contract_strings(specs)
    for value in contract_values:
        tokens = _tokens(value)
        token_set.update(tokens)
        maximum = min(5, len(tokens))
        for width in range(2, maximum + 1):
            for start in range(0, len(tokens) - width + 1):
                phrase_set.add(" ".join(tokens[start : start + width]))
    tokens = tuple(sorted(token_set))
    phrases = tuple(sorted(phrase_set, key=lambda value: (-len(value.split()), value)))
    query_body = {
        "version": CONTRACT_LEXICAL_CONTEXT_VERSION,
        "contract_values": contract_values,
        "tokens": tokens,
        "phrases": phrases,
    }
    query_sha256 = hashlib.sha256(
        json.dumps(
            query_body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return tokens, phrases, query_sha256


def _chunk_spans(text: str) -> list[tuple[int, int]]:
    if not text:
        return [(0, 0)]
    spans: list[tuple[int, int]] = []
    start = 0
    n_chars = len(text)
    while start < n_chars:
        proposed_end = min(n_chars, start + CONTRACT_LEXICAL_CHUNK_CHARS)
        end = proposed_end
        if proposed_end < n_chars:
            minimum_boundary = start + int(CONTRACT_LEXICAL_CHUNK_CHARS * 0.65)
            newline = text.rfind("\n", minimum_boundary, proposed_end)
            whitespace = text.rfind(" ", minimum_boundary, proposed_end)
            boundary = max(newline, whitespace)
            if boundary > start:
                end = boundary + 1
        if end <= start:
            end = min(n_chars, start + CONTRACT_LEXICAL_CHUNK_CHARS)
        spans.append((start, end))
        if end >= n_chars:
            break
        start = max(start + 1, end - CONTRACT_LEXICAL_CHUNK_OVERLAP_CHARS)
    return spans


def _scored_chunks(
    text: str,
    query_tokens: Sequence[str],
    query_phrases: Sequence[str],
) -> list[_Chunk]:
    chunks = []
    for start, end in _chunk_spans(text):
        excerpt = text[start:end]
        # Use the same generic-stopword normalization on both sides so an
        # exact source phrase such as ``ribbon of quartz`` matches the
        # contract phrase ``ribbon quartz`` after normalization.
        tokens = _tokens(excerpt)
        chunks.append(
            _Chunk(
                start=start,
                end=end,
                text=excerpt,
                tokens=tokens,
                normalized_text=" ".join(tokens),
            )
        )
    document_frequency = {
        token: sum(token in set(chunk.tokens) for chunk in chunks) for token in query_tokens
    }
    chunk_count = max(1, len(chunks))
    scored: list[_Chunk] = []
    for chunk in chunks:
        counts = Counter(chunk.tokens)
        score = 0.0
        for token in query_tokens:
            count = counts.get(token, 0)
            if not count:
                continue
            inverse_frequency = (
                math.log((1.0 + chunk_count) / (1.0 + document_frequency[token])) + 1.0
            )
            score += inverse_frequency * (1.0 + math.log(float(count)))
        padded = f" {chunk.normalized_text} "
        for phrase in query_phrases:
            if f" {phrase} " in padded:
                score += 1.5 + 0.5 * len(phrase.split())
        scored.append(
            _Chunk(
                start=chunk.start,
                end=chunk.end,
                text=chunk.text,
                tokens=chunk.tokens,
                normalized_text=chunk.normalized_text,
                score=float(score),
            )
        )
    return scored


def _overlaps(left: RetrievedContractExcerpt, right: RetrievedContractExcerpt) -> bool:
    return left.start < right.end and right.start < left.end


def _render_context(
    source: str,
    excerpts: Sequence[RetrievedContractExcerpt],
    *,
    fallback_tail_used: bool,
) -> str:
    lines = [
        f"[{CONTRACT_LEXICAL_CONTEXT_VERSION}]",
        "Contract-guided lexical excerpts; text is verbatim and kept in source order.",
    ]
    for index, excerpt in enumerate(sorted(excerpts, key=lambda item: item.start), start=1):
        kind = "Fallback tail excerpt" if fallback_tail_used else "Retrieved excerpt"
        lines.extend(
            [
                f"[{kind} {index} | source chars {excerpt.start}:{excerpt.end}]",
                source[excerpt.start : excerpt.end],
            ]
        )
    return "\n".join(lines)


def _focus_position(
    excerpt: str,
    query_tokens: Sequence[str],
    query_phrases: Sequence[str],
) -> int:
    lowered = excerpt.lower().replace("_", " ")
    positions = [
        position
        for value in [*query_phrases, *query_tokens]
        if (position := lowered.find(value)) >= 0
    ]
    return min(positions) if positions else len(excerpt) // 2


def compact_contract_lexical_context(
    clinical_text: Any,
    specs: Sequence[ExplicitFeatureSpec],
    *,
    max_chars: int,
) -> ContractLexicalContext:
    """Return labeled, verbatim contract-relevant excerpts under ``max_chars``.

    Ranking uses only contract-derived query tokens/phrases, within-document IDF,
    and deterministic lexical overlap.  Treatment, outcome, fold, and dataset
    labels are not accepted by this API.
    """
    budget = int(max_chars)
    if budget < 256:
        raise ValueError("contract lexical context max_chars must be at least 256")
    source = str(clinical_text or "")
    query_tokens, query_phrases, query_sha256 = _contract_query(specs)
    chunks = _scored_chunks(source, query_tokens, query_phrases)
    ranked = sorted(chunks, key=lambda chunk: (-chunk.score, chunk.start, chunk.end))
    positive = [chunk for chunk in ranked if chunk.score > 0.0]
    fallback = not positive
    if fallback:
        tail_size = min(len(source), max(0, budget - 180))
        start = len(source) - tail_size
        selected = [RetrievedContractExcerpt(start, len(source), 0.0)]
    else:
        selected: list[RetrievedContractExcerpt] = []
        selected_chunks: set[tuple[int, int]] = set()
        for chunk in positive:
            candidate = RetrievedContractExcerpt(chunk.start, chunk.end, chunk.score)
            if any(_overlaps(candidate, existing) for existing in selected):
                continue
            rendered = _render_context(source, [*selected, candidate], fallback_tail_used=False)
            if len(rendered) <= budget:
                selected.append(candidate)
                selected_chunks.add((chunk.start, chunk.end))

        # When no complete chunk fits, or a useful residual budget remains,
        # retain one match-centered verbatim slice rather than dropping it.
        for chunk in positive:
            if (chunk.start, chunk.end) in selected_chunks:
                continue
            if any(
                _overlaps(
                    RetrievedContractExcerpt(chunk.start, chunk.end, chunk.score),
                    existing,
                )
                for existing in selected
            ):
                continue
            empty = RetrievedContractExcerpt(chunk.start, chunk.start, chunk.score)
            available = budget - len(
                _render_context(source, [*selected, empty], fallback_tail_used=False)
            )
            if available < 160:
                continue
            focus = _focus_position(chunk.text, query_tokens, query_phrases)
            local_start = max(0, min(len(chunk.text) - available, focus - available // 2))
            local_end = min(len(chunk.text), local_start + available)
            candidate = RetrievedContractExcerpt(
                chunk.start + local_start,
                chunk.start + local_end,
                chunk.score,
            )
            if any(_overlaps(candidate, existing) for existing in selected):
                continue
            selected.append(candidate)
            break

        if not selected:
            # The fixed labels themselves consume part of the budget.  Center a
            # final slice in the best chunk and shrink until the hard bound holds.
            best = positive[0]
            empty = RetrievedContractExcerpt(best.start, best.start, best.score)
            available = max(
                0,
                budget - len(_render_context(source, [empty], fallback_tail_used=False)),
            )
            focus = _focus_position(best.text, query_tokens, query_phrases)
            local_start = max(0, min(len(best.text) - available, focus - available // 2))
            selected = [
                RetrievedContractExcerpt(
                    best.start + local_start,
                    best.start + local_start + available,
                    best.score,
                )
            ]

    selected = sorted(selected, key=lambda item: item.start)
    rendered = _render_context(source, selected, fallback_tail_used=fallback)
    if len(rendered) > budget:
        overflow = len(rendered) - budget
        last = selected[-1]
        selected[-1] = RetrievedContractExcerpt(
            last.start,
            max(last.start, last.end - overflow),
            last.score,
        )
        rendered = _render_context(source, selected, fallback_tail_used=fallback)
    if len(rendered) > budget:  # pragma: no cover - guards future label changes
        raise RuntimeError("contract lexical context exceeded its hard character budget")
    return ContractLexicalContext(
        text=rendered,
        version=CONTRACT_LEXICAL_CONTEXT_VERSION,
        original_char_count=len(source),
        max_chars=budget,
        query_sha256=query_sha256,
        query_tokens=query_tokens,
        query_phrases=query_phrases,
        selected_excerpts=tuple(selected),
        fallback_tail_used=fallback,
    )


__all__ = [
    "CONTRACT_LEXICAL_CHUNK_CHARS",
    "CONTRACT_LEXICAL_CHUNK_OVERLAP_CHARS",
    "CONTRACT_LEXICAL_CONTEXT_VERSION",
    "EXTRACTION_GROUPING_VERSION",
    "ContractLexicalContext",
    "RetrievedContractExcerpt",
    "compact_contract_lexical_context",
]
