"""Sanitized note-grounding diagnostics for extracted causal variables.

The checker never emits note text, row identifiers, or row-level extracted
values. It looks for contract anchors and already-extracted values in bounded
note windows and returns aggregate counts/rates only. For categorical contracts
it also counts declared-category evidence near the anchor, revealing missed
extraction opportunities, category-mapping errors, and ambiguous ontologies.

Input text is temporally valid by design. This module deliberately performs no
treatment-boundary detection, temporal classification, or timing-based
eligibility check. Semantic timepoint wording can still be part of a contract's
meaning and therefore remains available to ordinary lexical grounding.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .all_evidence_fusion import CandidateContract, source_text_temporal_policy_audit
from .frozen_extraction_cache_overlay import expected_extraction_columns
from .post_extraction_scientific_policy import ExtractionGroundingPolicy

EXTRACTION_GROUNDING_DIAGNOSTIC_VERSION = "extraction_grounding_diagnostic_v6"

_NOTE_BREAK = re.compile(r"\s*<new_note>\s*", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
_RAW_WORD = re.compile(r"[A-Za-z0-9]+")
_EXPLICIT_ACRONYM = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,11}(?![A-Za-z0-9])")
_CLAUSE_BREAK = re.compile(r"[.!?;]")
_UNASSERTED_CATEGORY_PREFIX = re.compile(
    r"(?:\b(?:no|not|never|without|denies|denied|deny)\b"
    r"(?:\s+[a-z0-9-]+){0,4}\s*|\bnon[- ]*|"
    r"\b(?:if|whether|possible|possibly|suspected|pending|indeterminate)\b"
    r"[^.!?;]{0,48})$",
    re.IGNORECASE,
)
# Units are recognized only when the contract declares one with neutral syntax.
# The expression intentionally contains no vocabulary of known units.  A bare
# ``in`` declaration is kept strict (one atom or an explicitly separated
# compound) so ordinary prose is not silently reinterpreted as a unit.
_EXPLICIT_UNIT_DECLARATION = re.compile(
    r"(?:\bunits?\s*(?::|=|\bis\b|\bare\b)\s*|\bin\s+)"
    r"(?P<unit>(?:[a-z0-9\u00b5\u03bc\u00b0%]+)"
    r"(?:\s*[/_.\u00b7*-]\s*[a-z0-9\u00b5\u03bc\u00b0%]+)*)"
    r"(?=\s*[.!?,;)]|\s*$)",
    re.IGNORECASE,
)


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def _eligible_anchor_token(token: str) -> bool:
    """Apply only shape-based, domain-neutral lexical eligibility rules."""

    return bool(len(token) >= 3 or any(character.isdigit() for character in token))


def _token_pattern(token: str) -> re.Pattern[str]:
    """Match a contract token without concept-specific spelling overrides.

    Digit-bearing identifiers may be written with neutral identifier separators
    (``qx7``, ``qx-7``, or ``q_x_7``).  Letter-only words remain exact lexical
    tokens, including short words and acronyms.
    """

    exact = _normalize(token)
    if any(character.isdigit() for character in exact):
        body = r"[\s_-]*".join(re.escape(character) for character in exact)
    else:
        body = re.escape(exact)
    return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])", re.IGNORECASE)


def _token_is_source_attested(token: str, texts: Sequence[str]) -> bool:
    pattern = _token_pattern(token)
    return any(pattern.search(_normalize(text)) is not None for text in texts)


def _syntax_identifier_tokens(surface: str) -> tuple[str, ...]:
    """Return only syntax-signalled identifiers from contract text.

    Prose words do not become independent anchors.  Upper-case acronyms and
    digit-bearing identifiers do because their surface syntax marks them as
    identifiers without requiring a curated concept list.
    """

    raw_acronyms = (_normalize(match.group(0)) for match in _EXPLICIT_ACRONYM.finditer(surface))
    digit_identifiers = (
        _normalize(token)
        for token in _RAW_WORD.findall(surface)
        if any(character.isdigit() for character in token)
    )
    return tuple(dict.fromkeys((*raw_acronyms, *digit_identifiers)))


def _contract_anchor_groups(
    spec: Mapping[str, Any],
    texts: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    """Build a small set of conjunctive, contract-derived anchor signatures.

    Eligible name components are conjunctive, including ordinary short name
    tokens.  Only syntax-signalled acronyms and digit-bearing identifiers in
    the name or description may act as separate strong signatures.  Every
    emitted token must be attested in the supplied source text; contract text
    alone can never manufacture evidence.
    """

    raw_name = _WORD.findall(_normalize(str(spec["name"]).replace("_", " ")))
    name_tokens = tuple(
        dict.fromkeys(token for token in raw_name if _eligible_anchor_token(token))
    )
    syntax_identifiers = tuple(
        dict.fromkeys(
            (
                *_syntax_identifier_tokens(str(spec["name"]).replace("_", " ")),
                *_syntax_identifier_tokens(str(spec.get("description") or "")),
            )
        )
    )
    attested_name_tokens = tuple(
        token for token in name_tokens if _token_is_source_attested(token, texts)
    )
    attested_syntax_identifiers = tuple(
        token
        for token in syntax_identifiers
        if _token_is_source_attested(token, texts)
    )
    groups: list[tuple[str, ...]] = []
    if name_tokens and len(attested_name_tokens) == len(name_tokens):
        groups.append(name_tokens)
    for token in attested_name_tokens:
        if any(character.isdigit() for character in token):
            singleton = (token,)
            if singleton not in groups:
                groups.append(singleton)
    for token in attested_syntax_identifiers:
        singleton = (token,)
        if singleton not in groups:
            groups.append(singleton)
    # Preserve every source-attested signature. A fixed top-k selection would
    # silently omit valid dataset-specific identifiers.
    return tuple(groups)


def _contract_anchor_tokens(groups: Sequence[Sequence[str]]) -> tuple[str, ...]:
    """Return the unique tokens represented in the conjunctive signatures."""

    return tuple(dict.fromkeys(token for group in groups for token in group))


def _expected_unit(spec: Mapping[str, Any]) -> tuple[str | None, re.Pattern[str] | None]:
    if str(spec.get("type")) != "continuous":
        return None, None
    contract = _normalize(str(spec.get("description") or ""))
    declarations = list(_EXPLICIT_UNIT_DECLARATION.finditer(contract))
    if declarations:
        label = _normalize(declarations[-1].group("unit"))
        return label, _literal_surface_pattern(label, flexible_separators=True)
    return None, None


def _literal_surface_pattern(
    surface: str,
    *,
    flexible_separators: bool,
) -> re.Pattern[str]:
    """Compile an exact contract surface with neutral separator tolerance."""

    normalized = _normalize(surface)
    parts = tuple(part for part in re.split(r"[\s_./\u00b7*-]+", normalized) if part)
    if not parts:
        body = re.escape(normalized)
    else:
        separator = r"[\s_./\u00b7*-]+" if flexible_separators else r"\s+"
        body = separator.join(re.escape(part) for part in parts)
    left_boundary = r"(?<![a-z0-9])" if normalized[:1].isalnum() else ""
    right_boundary = r"(?![a-z0-9])" if normalized[-1:].isalnum() else ""
    return re.compile(rf"{left_boundary}(?:{body}){right_boundary}", re.IGNORECASE)


def _numeric_patterns(value: Any) -> tuple[re.Pattern[str], ...]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ()
    if not math.isfinite(numeric):
        return ()
    renderings = {format(numeric, ".12g")}
    if float(numeric).is_integer():
        renderings.add(str(int(numeric)))
    else:
        renderings.update({f"{numeric:.1f}", f"{numeric:.2f}", f"{numeric:.3f}"})
    patterns: list[re.Pattern[str]] = []
    for rendering in sorted(renderings, key=len, reverse=True):
        integer, dot, fraction = rendering.partition(".")
        if dot:
            body = rf"{re.escape(integer)}[.,]{re.escape(fraction.rstrip('0'))}0*"
        else:
            body = re.escape(integer)
        patterns.append(re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE))
    return tuple(patterns)


def _category_patterns(
    value: Any,
    spec: Mapping[str, Any],
) -> tuple[tuple[re.Pattern[str], bool], ...]:
    canonical = _normalize(value)
    if not canonical:
        return ()
    candidates: list[tuple[str, bool]] = [(canonical, False)]
    aliases = spec.get("value_aliases") or {}
    if isinstance(aliases, Mapping):
        for raw_category, raw_aliases in aliases.items():
            if _normalize(raw_category) != canonical:
                continue
            values = raw_aliases if isinstance(raw_aliases, (list, tuple)) else [raw_aliases]
            candidates.extend((_normalize(alias), True) for alias in values if _normalize(alias))
    result: list[tuple[re.Pattern[str], bool]] = []
    for candidate, is_alias in candidates:
        if not _WORD.search(candidate):
            continue
        result.append(
            (
                _literal_surface_pattern(candidate, flexible_separators=True),
                is_alias,
            )
        )
    return tuple(result)


def _declared_category_index(value: Any, spec: Mapping[str, Any]) -> int | None:
    """Map one extracted categorical value to one unambiguous declared category.

    This identity is used only while aggregating a row and is never returned.
    Ambiguous aliases deliberately map to ``None`` rather than making a strong
    category-mismatch claim.
    """

    normalized = _normalize(value)
    if not normalized:
        return None
    aliases = spec.get("value_aliases") or {}
    matches: list[int] = []
    for category_index, raw_category in enumerate(spec.get("categories") or ()):
        category = _normalize(raw_category)
        variants = {category}
        if isinstance(aliases, Mapping):
            for alias_category, raw_aliases in aliases.items():
                if _normalize(alias_category) != category:
                    continue
                exact_aliases = (
                    raw_aliases if isinstance(raw_aliases, (list, tuple)) else [raw_aliases]
                )
                variants.update(_normalize(alias) for alias in exact_aliases if _normalize(alias))
        if normalized in variants:
            matches.append(category_index)
    return matches[0] if len(matches) == 1 else None


def _category_match_is_asserted(
    segment: str,
    match: _ValueMatch,
    *,
    prefix_chars: int,
) -> bool:
    """Conservatively reject negated or hypothetical category mentions."""

    prefix = segment[max(0, match.start - int(prefix_chars)) : match.start]
    return _UNASSERTED_CATEGORY_PREFIX.search(prefix) is None


def _anchor_spans(
    text: str,
    groups: Sequence[Sequence[str]],
    *,
    maximum_group_span: int = 96,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for group in groups:
        exact_group = tuple(dict.fromkeys(map(str, group)))
        if not exact_group:
            continue
        events: list[tuple[int, int, int]] = []
        for token_index, token in enumerate(exact_group):
            events.extend(
                (match.start(), match.end(), token_index)
                for match in _token_pattern(token).finditer(text)
            )
        if len({event[2] for event in events}) != len(exact_group):
            continue
        events.sort()
        counts = [0] * len(exact_group)
        covered = 0
        left = 0
        for right, event in enumerate(events):
            token_index = event[2]
            if counts[token_index] == 0:
                covered += 1
            counts[token_index] += 1
            while covered == len(exact_group):
                window = events[left : right + 1]
                start = window[0][0]
                end = max(item[1] for item in window)
                if end - start <= int(maximum_group_span):
                    spans.append((start, end))
                left_token = events[left][2]
                counts[left_token] -= 1
                if counts[left_token] == 0:
                    covered -= 1
                left += 1
    return sorted(set(spans))


@dataclass(frozen=True)
class _ValueMatch:
    start: int
    end: int
    alias: bool = False
    unit_supported: bool | None = None


def _value_matches(
    segment: str,
    *,
    value: Any,
    spec: Mapping[str, Any],
    unit_pattern: re.Pattern[str] | None,
    unit_window_chars: int,
) -> tuple[_ValueMatch, ...]:
    if str(spec["type"]) == "continuous":
        result: dict[tuple[int, int], _ValueMatch] = {}
        for pattern in _numeric_patterns(value):
            for match in pattern.finditer(segment):
                left = max(0, match.start() - int(unit_window_chars))
                right = min(len(segment), match.end() + int(unit_window_chars))
                result[(match.start(), match.end())] = _ValueMatch(
                    start=match.start(),
                    end=match.end(),
                    unit_supported=(
                        None
                        if unit_pattern is None
                        else bool(unit_pattern.search(segment[left:right]))
                    ),
                )
        return tuple(result[key] for key in sorted(result))

    result: dict[tuple[int, int], _ValueMatch] = {}
    for pattern, is_alias in _category_patterns(value, spec):
        for match in pattern.finditer(segment):
            result[(match.start(), match.end())] = _ValueMatch(
                match.start(), match.end(), alias=is_alias
            )
    return tuple(result[key] for key in sorted(result))


def _span_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] < right[0]:
        return right[0] - left[1]
    if right[1] < left[0]:
        return left[0] - right[1]
    return 0


def _same_local_clause(left: tuple[int, int], right: tuple[int, int], segment: str) -> bool:
    between_start = min(left[1], right[1])
    between_end = max(left[0], right[0])
    if between_end <= between_start:
        return True
    return _CLAUSE_BREAK.search(segment[between_start:between_end]) is None


def _row_support(
    text: str,
    *,
    value: Any,
    spec: Mapping[str, Any],
    anchor_groups: Sequence[Sequence[str]],
    unit_pattern: re.Pattern[str] | None,
    policy: ExtractionGroundingPolicy,
) -> Mapping[str, Any]:
    segments = [segment for segment in _NOTE_BREAK.split(_normalize(text)) if segment]
    if not segments:
        segments = [""]
    value_grounded = False
    alias = False
    unit_support: list[bool] = []
    anchor_found = False
    supported_categories: set[int] = set()
    for segment in segments:
        anchors = _anchor_spans(
            segment,
            anchor_groups,
            maximum_group_span=policy.maximum_group_span_chars,
        )
        anchor_found = anchor_found or bool(anchors)
        unit_window_chars = max(
            int(policy.unit_window_min_chars),
            min(
                int(policy.unit_window_max_chars),
                int(policy.anchor_value_window_chars)
                // int(policy.unit_window_divisor),
            ),
        )
        matches = _value_matches(
            segment,
            value=value,
            spec=spec,
            unit_pattern=unit_pattern,
            unit_window_chars=unit_window_chars,
        )
        for value_match in matches:
            if (
                str(spec["type"]) == "categorical"
                and not _category_match_is_asserted(
                    segment,
                    value_match,
                    prefix_chars=policy.category_assertion_prefix_chars,
                )
            ):
                continue
            value_span = (value_match.start, value_match.end)
            if not any(
                _span_distance(anchor, value_span)
                <= int(policy.anchor_value_window_chars)
                and _same_local_clause(anchor, value_span, segment)
                for anchor in anchors
            ):
                continue
            value_grounded = True
            alias = alias or value_match.alias
            if value_match.unit_supported is not None:
                unit_support.append(bool(value_match.unit_supported))
        if str(spec["type"]) == "categorical" and anchors:
            for category_index, category in enumerate(spec.get("categories") or ()):
                category_matches = _value_matches(
                    segment,
                    value=category,
                    spec=spec,
                    unit_pattern=None,
                    unit_window_chars=unit_window_chars,
                )
                for category_match in category_matches:
                    if not _category_match_is_asserted(
                        segment,
                        category_match,
                        prefix_chars=policy.category_assertion_prefix_chars,
                    ):
                        continue
                    category_span = (category_match.start, category_match.end)
                    if not any(
                        _span_distance(anchor, category_span)
                        <= int(policy.anchor_value_window_chars)
                        and _same_local_clause(anchor, category_span, segment)
                        for anchor in anchors
                    ):
                        continue
                    supported_categories.add(category_index)
    supported_category_indices = tuple(sorted(supported_categories))
    return {
        "anchor_found": anchor_found,
        "value_grounded": value_grounded,
        "alias_support": alias,
        "unit_supported": (None if not unit_support else any(unit_support)),
        # These opaque positions exist only long enough to update aggregate
        # counters in the caller.  Neither positions nor row-level support are
        # included in the returned diagnostic artifact.
        "declared_category_supported_indices": supported_category_indices,
    }


def _missing_mask(frame: pd.DataFrame, spec: Mapping[str, Any]) -> np.ndarray:
    value_column, missing_column = expected_extraction_columns(spec)
    if value_column not in frame or missing_column not in frame:
        raise ValueError(f"extracted frame is missing columns for {spec['name']!r}")
    declared = frame[missing_column].fillna(True).astype(bool).to_numpy()
    return declared | frame[value_column].isna().to_numpy()


def build_extraction_grounding_diagnostics(
    frame: pd.DataFrame,
    texts: Sequence[str],
    specs: Sequence[Mapping[str, Any]],
    *,
    diagnostic_start: int = 1,
    policy: ExtractionGroundingPolicy | None = None,
    window_chars: int = 96,
    minimum_evaluable_rows: int = 3,
    maximum_alternative_category_only_rate: float = 0.50,
) -> list[dict[str, Any]]:
    """Return aggregate ontology/value grounding diagnostics.

    Only aggregate counts and rates are returned.  Note text, evidence spans,
    row identifiers, and extracted row values are intentionally absent.
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("grounding diagnostics require a non-empty extracted frame")
    exact_texts = tuple(texts)
    if len(exact_texts) != len(frame) or not all(isinstance(value, str) for value in exact_texts):
        raise ValueError("texts must contain one exact string per extracted row")
    if isinstance(diagnostic_start, bool) or int(diagnostic_start) < 1:
        raise ValueError("diagnostic_start must be a positive integer")
    if policy is None:
        # Compatibility-only behavior. The typed portable workflow supplies
        # the complete closed policy and never enters this branch.
        policy = ExtractionGroundingPolicy(
            anchor_group_selection="all_source_attested_unbounded",
            maximum_group_span_chars=96,
            anchor_value_window_chars=int(window_chars),
            category_assertion_prefix_chars=64,
            unit_window_min_chars=12,
            unit_window_max_chars=32,
            unit_window_divisor=3,
            minimum_evaluable_rows=int(minimum_evaluable_rows),
            maximum_alternative_category_only_rate=float(
                maximum_alternative_category_only_rate
            ),
            unsupported_value_warning_rate=0.25,
            minimum_unit_support_rate=0.50,
        )
    elif not isinstance(policy, ExtractionGroundingPolicy):
        raise TypeError("policy must be ExtractionGroundingPolicy")
    if int(policy.anchor_value_window_chars) < 40:
        raise ValueError(
            "extraction grounding anchor_value_window_chars must be at least 40"
        )

    canonical = [CandidateContract(spec).extraction_spec for spec in specs]
    diagnostics: list[dict[str, Any]] = []
    for offset, spec in enumerate(canonical):
        anchor_groups = _contract_anchor_groups(spec, exact_texts)
        anchor_tokens = _contract_anchor_tokens(anchor_groups)
        expected_unit, unit_pattern = _expected_unit(spec)
        value_column, _ = expected_extraction_columns(spec)
        missing = _missing_mask(frame, spec)
        categorical = str(spec["type"]) == "categorical"
        counters = {
            "observed": 0,
            "anchor_found": 0,
            "grounded": 0,
            "unsupported": 0,
            "alias": 0,
            "unit_evaluable": 0,
            "unit_supported": 0,
            "missing": 0,
            "missing_anchor": 0,
            "missing_single_category": 0,
            "observed_single_category": 0,
            "alternative_category_only": 0,
            "locally_grounded_category_evaluable": 0,
            "locally_grounded_alternative_category_only": 0,
            "category_conflict": 0,
            "observed_category_conflict": 0,
            "missing_category_conflict": 0,
        }
        for position, text in enumerate(exact_texts):
            row_missing = bool(missing[position])
            if row_missing and not categorical:
                continue
            support = _row_support(
                text,
                value=(None if row_missing else frame.iloc[position][value_column]),
                spec=spec,
                anchor_groups=anchor_groups,
                unit_pattern=unit_pattern,
                policy=policy,
            )
            supported_categories = tuple(support["declared_category_supported_indices"])
            category_conflict = len(supported_categories) > 1
            if categorical:
                counters["category_conflict"] += int(category_conflict)
            if row_missing:
                counters["missing"] += 1
                counters["missing_anchor"] += int(bool(support["anchor_found"]))
                counters["missing_single_category"] += int(len(supported_categories) == 1)
                counters["missing_category_conflict"] += int(category_conflict)
                continue

            counters["observed"] += 1
            counters["anchor_found"] += int(bool(support["anchor_found"]))
            counters["grounded"] += int(bool(support["value_grounded"]))
            counters["unsupported"] += int(not bool(support["value_grounded"]))
            counters["alias"] += int(bool(support["alias_support"]))
            if support["unit_supported"] is not None:
                counters["unit_evaluable"] += 1
                counters["unit_supported"] += int(bool(support["unit_supported"]))
            if categorical:
                counters["observed_category_conflict"] += int(category_conflict)
                if len(supported_categories) == 1:
                    counters["observed_single_category"] += 1
                    sole_category = supported_categories[0]
                    extracted_category = _declared_category_index(
                        frame.iloc[position][value_column], spec
                    )
                    extracted_supported = bool(support["value_grounded"])
                    alternative_only = (
                        not extracted_supported and extracted_category != sole_category
                    )
                    counters["alternative_category_only"] += int(alternative_only)
                    counters["locally_grounded_category_evaluable"] += 1
                    counters["locally_grounded_alternative_category_only"] += int(alternative_only)

        observed = counters["observed"]
        grounded_rate = counters["grounded"] / max(observed, 1)
        unsupported_rate = counters["unsupported"] / max(observed, 1)
        category_evaluable = counters["locally_grounded_category_evaluable"]
        locally_grounded_alternative_count = counters["locally_grounded_alternative_category_only"]
        locally_grounded_alternative_rate = locally_grounded_alternative_count / max(
            category_evaluable, 1
        )
        hard_failures: list[str] = []
        warnings: list[str] = []
        if (
            observed
            and unsupported_rate > float(policy.unsupported_value_warning_rate)
        ):
            warnings.append("many_values_not_lexically_grounded_to_contract")
        if counters["missing_single_category"]:
            warnings.append("missing_rows_have_single_declared_category_support")
        if counters["alternative_category_only"]:
            warnings.append("alternative_category_only_support_observed")
        if counters["category_conflict"]:
            warnings.append("multiple_declared_categories_supported_near_anchor")
        if (
            locally_grounded_alternative_count
            >= int(policy.minimum_evaluable_rows)
            and category_evaluable >= int(policy.minimum_evaluable_rows)
            and locally_grounded_alternative_rate
            > float(policy.maximum_alternative_category_only_rate)
        ):
            hard_failures.append("alternative_category_only_value_support")
        if not anchor_groups:
            warnings.append("contract_anchor_not_discriminative")
        unit_rate = (
            None
            if not counters["unit_evaluable"]
            else counters["unit_supported"] / counters["unit_evaluable"]
        )
        if (
            unit_rate is not None
            and counters["unit_evaluable"] >= int(policy.minimum_evaluable_rows)
            and unit_rate < float(policy.minimum_unit_support_rate)
        ):
            warnings.append("expected_unit_not_consistently_supported")

        revision_guidance: list[str] = []
        if counters["missing_single_category"]:
            revision_guidance.append("review_missingness_logic_and_declared_alias_coverage")
        if counters["alternative_category_only"]:
            revision_guidance.append("review_canonical_category_mapping_and_extraction_prompt")
        if counters["category_conflict"]:
            revision_guidance.append("review_category_mutual_exclusivity_and_aliases")

        diagnostics.append(
            {
                "diagnostic_id": f"diagnostic_{int(diagnostic_start) + offset:04d}",
                "kind": "extraction_text_grounding",
                "diagnostic_version": EXTRACTION_GROUNDING_DIAGNOSTIC_VERSION,
                "feature_name": str(spec["name"]),
                "contract_anchor_group_count": len(anchor_groups),
                "contract_anchor_token_count": len(anchor_tokens),
                "observed_row_count": observed,
                "anchor_detected_row_count": counters["anchor_found"],
                "value_grounding": {
                    "supported_row_count": counters["grounded"],
                    "unsupported_row_count": counters["unsupported"],
                    "alias_supported_row_count": counters["alias"],
                    "supported_rate": grounded_rate,
                    "unsupported_rate": unsupported_rate,
                },
                "source_text_temporal_policy": source_text_temporal_policy_audit(),
                "unit_alignment": {
                    "expected_unit": expected_unit,
                    "evaluable_row_count": counters["unit_evaluable"],
                    "supported_rate": unit_rate,
                    "unit_bound_to_matched_value_window": True,
                },
                "categorical_ontology_alignment": {
                    "applicable": categorical,
                    "declared_category_count": len(spec.get("categories") or ()),
                    "missing_row_count": counters["missing"],
                    "missing_anchor_detected_row_count": counters["missing_anchor"],
                    "missing_single_declared_category_supported_row_count": counters[
                        "missing_single_category"
                    ],
                    "observed_single_declared_category_supported_row_count": counters[
                        "observed_single_category"
                    ],
                    "alternative_category_only_supported_row_count": counters[
                        "alternative_category_only"
                    ],
                    "locally_grounded_evaluable_row_count": category_evaluable,
                    "locally_grounded_alternative_category_only_supported_row_count": (
                        locally_grounded_alternative_count
                    ),
                    "locally_grounded_alternative_category_only_support_rate": (
                        locally_grounded_alternative_rate
                    ),
                    "conflicting_multiple_categories_supported_row_count": counters[
                        "category_conflict"
                    ],
                    "observed_conflicting_multiple_categories_supported_row_count": counters[
                        "observed_category_conflict"
                    ],
                    "missing_conflicting_multiple_categories_supported_row_count": counters[
                        "missing_category_conflict"
                    ],
                    "support_requires_contract_anchor_and_local_clause": True,
                    "hard_failure_requires_locally_grounded_alternative": True,
                    "hard_failure_minimum_evaluable_rows": int(minimum_evaluable_rows),
                    "hard_failure_maximum_alternative_category_only_rate": float(
                        maximum_alternative_category_only_rate
                    ),
                    "row_level_category_evidence_exposed": False,
                },
                "configuration": {
                    "anchor_value_maximum_gap_chars": int(window_chars),
                    "minimum_evaluable_rows": int(minimum_evaluable_rows),
                    "maximum_alternative_category_only_rate": float(
                        maximum_alternative_category_only_rate
                    ),
                },
                "hard_failures": hard_failures,
                "warnings": warnings,
                "revision_guidance": revision_guidance,
                "passed": not hard_failures,
                "raw_note_text_exposed": False,
                "row_identifiers_exposed": False,
                "row_level_extracted_values_exposed": False,
            }
        )
    return diagnostics


__all__ = [
    "EXTRACTION_GROUNDING_DIAGNOSTIC_VERSION",
    "build_extraction_grounding_diagnostics",
]
