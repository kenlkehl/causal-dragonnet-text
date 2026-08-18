"""Patient-level extraction, empirical review, and causal estimation for Stage 2.

The implementation is deliberately file-oriented.  Every expensive extraction
batch and every review round writes an ordinary result followed by
``complete.json``.  A repeated invocation reads those files and continues at
the first unfinished operation.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

EXTRACTION_CHECKPOINT_SCHEMA_VERSION = (
    "stage2_single_patient_extraction_v4_continuous_category_fallback"
)
EXTRACTION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION = (
    "stage2_single_patient_feature_batch_extraction_v3_continuous_category_fallback"
)
PAGE_EXTRACTION_CHECKPOINT_SCHEMA_VERSION = (
    "stage2_single_patient_page_extraction_v3_continuous_category_fallback"
)
PAGE_RECONCILIATION_CHECKPOINT_SCHEMA_VERSION = (
    "stage2_lossless_feature_partition_reconciliation_v4_continuous_category_fallback"
)
REVIEW_CHECKPOINT_SCHEMA_VERSION = "stage2_feature_partition_review_v3_signal_pruning"
ESTIMATION_CHECKPOINT_SCHEMA_VERSION = "stage2_outer_estimation_v2_modeling_strategy"
EXTRACTION_ISSUE_SCHEMA_VERSION = "stage2_extraction_issues_v1"
ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION = (
    "stage2_training_failure_ontology_refinement_v2_request_policy"
)
# Compatibility defaults for Stage 2 config objects created before ontology
# refinement was added.  Keeping this boundary tolerant also protects a
# long-running workflow if an older caller passes a config object directly.
DEFAULT_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS = 3
DEFAULT_MAX_ONTOLOGY_REFINEMENT_ROUNDS = 2
DEFAULT_EXTRACTION_FEATURE_BATCH_SIZE = 10


class RequestJSON(Protocol):
    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        validate: Callable[[Mapping[str, Any]], dict[str, Any]],
        *,
        request_kind: str = "interpretation",
    ) -> dict[str, Any]: ...

_SCALAR_EXTRACTION_RULES = (
    "Return one scalar value or null per feature; never return an object or array.",
    "For a continuous feature, return one JSON number whenever the record supplies the "
    "requested numeric measurement. If the record supplies only a documented categorical "
    "or threshold representation of that same measurement, return that one concise string "
    "instead of discarding it or inventing a number. From a composite such as 147/93, use "
    "only a component explicitly named by the feature; if the definition requests multiple "
    "components, return null rather than a ratio string or aggregate.",
)

CONTINUOUS_MODELING_STRATEGIES = frozenset(
    {
        "continuous",
        "categorical",
        "continuous_with_categorical_fallback",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _ontology_refinement_limits(config: Any) -> tuple[int, int]:
    """Read refinement limits from current or pre-refinement config objects."""

    missing = [
        name
        for name in (
            "ontology_refinement_min_failure_patients",
            "max_ontology_refinement_rounds",
        )
        if not hasattr(config, name)
    ]
    if missing:
        LOGGER.warning(
            "Stage 2 received a pre-ontology-refinement config without %s; "
            "using compatibility defaults",
            ", ".join(missing),
        )
    return (
        int(
            getattr(
                config,
                "ontology_refinement_min_failure_patients",
                DEFAULT_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS,
            )
        ),
        int(
            getattr(
                config,
                "max_ontology_refinement_rounds",
                DEFAULT_MAX_ONTOLOGY_REFINEMENT_ROUNDS,
            )
        ),
    )


def _configured_extraction_feature_batch_size(config: Any) -> int:
    """Read the feature prompt cap from current or pre-batching configs."""

    if not hasattr(config, "extraction_feature_batch_size"):
        LOGGER.warning(
            "Stage 2 received a pre-feature-batching config; using the default "
            "extraction feature batch size of %s",
            DEFAULT_EXTRACTION_FEATURE_BATCH_SIZE,
        )
    return int(
        getattr(
            config,
            "extraction_feature_batch_size",
            DEFAULT_EXTRACTION_FEATURE_BATCH_SIZE,
        )
    )


def _value_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint ordered modeling values for checkpoint compatibility."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": list(map(str, frame.columns)),
                "dtypes": [str(dtype) for dtype in frame.dtypes],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _clean_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    raise ValueError("extracted feature values must be scalar JSON values or null")


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    key = re.sub(r"[^a-z0-9]+", "", value.lower())
    return key in {"", "unknown", "notdocumented", "missing", "nan", "na", "null", "none"}


def _normalized_category_values(*, value_type: Any, values: Sequence[Any]) -> list[str]:
    normalized_type = str(value_type or "ambiguous").strip().lower()
    normalized = [str(item).strip() for item in values if str(item).strip()]
    if normalized_type in {"binary", "categorical", "ordinal"} and len(normalized) == 1:
        separated = [
            part.strip() for part in re.split(r"\s*[,;|]\s*", normalized[0]) if part.strip()
        ]
        if len(separated) > 1:
            normalized = separated
    if normalized_type == "binary" and len(normalized) == 1:
        slash_separated = [part.strip() for part in normalized[0].split("/") if part.strip()]
        if len(slash_separated) == 2:
            normalized = slash_separated

    expand_integer_ranges = normalized_type in {"binary", "ordinal"} or (
        normalized_type == "categorical" and len(normalized) == 1
    )
    if expand_integer_ranges:
        expanded: list[str] = []
        for value in normalized:
            match = re.fullmatch(
                r"([+-]?\d+)\s*(?:-|–|—|to)\s*([+-]?\d+)",
                value,
                flags=re.IGNORECASE,
            )
            if match is None:
                expanded.append(value)
                continue
            start, stop = (int(token) for token in match.groups())
            if stop < start or stop - start > 100:
                expanded.append(value)
                continue
            expanded.extend(str(category) for category in range(start, stop + 1))
        normalized = expanded
    return list(dict.fromkeys(normalized))


_CATEGORY_SCHEMA_LABEL = re.compile(
    r"^(?:binary|boolean|categorical|category|categories|ordinal|class|classes|"
    r"level|levels|allowed\s+(?:category|categories|value|values))$",
    flags=re.IGNORECASE,
)


def _validated_closed_category_values(
    *,
    value_type: Any,
    values: Sequence[Any],
    source: str,
) -> list[str]:
    """Normalize and validate one extraction-ready closed ontology.

    This is deliberately domain-agnostic.  It validates the shape of the
    ontology, not which clinical labels should appear in it.
    """

    normalized_type = str(value_type or "ambiguous").strip().lower()
    if normalized_type not in {"binary", "categorical", "ordinal"}:
        raise ValueError(f"{source} is not a closed-ontology value type")
    categories = _normalized_category_values(
        value_type=normalized_type,
        values=values,
    )
    identity_keys = [
        re.sub(r"[\W_]+", " ", category, flags=re.UNICODE).strip().casefold()
        for category in categories
    ]
    if len(identity_keys) != len(set(identity_keys)):
        duplicate_keys = sorted(
            {
                identity_key
                for identity_key in identity_keys
                if identity_keys.count(identity_key) > 1
            }
        )
        duplicate_values = [
            category
            for category, identity_key in zip(categories, identity_keys)
            if identity_key in duplicate_keys
        ]
        raise ValueError(
            f"{source} categories_or_unit must contain categories that are distinct "
            "after case and spacing normalization; return each category once and remove "
            f"these normalization-equivalent values: {duplicate_values[:12]!r}"
        )
    if normalized_type == "binary" and len(categories) != 2:
        raise ValueError(
            f"{source} binary categories_or_unit must contain exactly two distinct "
            "scalar categories as separate array items; "
            f"received {len(categories)}: {categories!r}"
        )
    if normalized_type in {"categorical", "ordinal"} and len(categories) < 2:
        raise ValueError(
            f"{source} {normalized_type} categories_or_unit must contain at least two "
            "distinct scalar categories as separate array items; "
            f"received {len(categories)}: {categories!r}"
        )
    placeholders = [
        category for category in categories if _CATEGORY_SCHEMA_LABEL.fullmatch(category.strip())
    ]
    if placeholders:
        raise ValueError(
            f"{source} categories_or_unit contains schema label(s) rather than "
            f"extractable values: {placeholders!r}"
        )
    return categories


def _declared_categories(definition: Mapping[str, Any]) -> list[str]:
    return _normalized_category_values(
        value_type=definition.get("value_type"),
        values=definition.get("categories_or_unit") or [],
    )


def _prompt_feature_definitions(
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project frozen definitions to fields needed for patient measurement."""

    output: list[dict[str, Any]] = []
    for definition in definitions:
        row = {
            key: definition.get(key)
            for key in (
                "name",
                "description",
                "value_type",
                "categories_or_unit",
                "measurement_definition",
                "missing_value_rule",
            )
        }
        if str(row.get("value_type")) in {"binary", "categorical", "ordinal"}:
            row["categories_or_unit"] = _declared_categories(row)
        if str(row.get("value_type")) == "continuous":
            row["accepted_representations"] = (
                "one JSON number, or one documented categorical/threshold string when "
                "the numeric measurement is unavailable"
            )
        output.append(row)
    return output


def _stale_category_ontology_audit(path: Path) -> dict[str, Any] | None:
    """Return an audit whose old closed ontology now expands more precisely."""

    if not path.is_file():
        return None
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(audit, Mapping):
        return None
    for item in audit.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        previous = [str(value) for value in item.get("allowed_categories") or []]
        current = _normalized_category_values(
            value_type=item.get("value_type"),
            values=previous,
        )
        if current != previous:
            return dict(audit)
    return None


def _supersede_stale_category_ontology_audit(
    path: Path,
    *,
    previous: Mapping[str, Any] | None,
) -> None:
    if previous is None or _stale_category_ontology_audit(path) is None:
        return
    _write_json(
        path,
        {
            "schema_version": "stage2_category_ontology_repair_v1",
            "resolution": "superseded_by_expanded_category_ontology",
            "superseded_at": _now(),
            "previous_audit": dict(previous),
        },
    )


def _feature_name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _aligned_extraction_values(
    values: Mapping[Any, Any],
    *,
    feature_names: Sequence[str],
    row_id: int,
) -> dict[str, Any]:
    """Recover harmless key drift and conservatively fill omitted features."""

    raw = {str(key): value for key, value in values.items()}
    expected = set(feature_names)
    aligned = {name: raw[name] for name in feature_names if name in raw}
    used = set(aligned)
    aliases: dict[str, str] = {}
    missing: list[str] = []
    for name in feature_names:
        if name in aligned:
            continue
        candidates = [
            candidate
            for candidate in raw
            if candidate not in used and _feature_name_key(candidate) == _feature_name_key(name)
        ]
        if len(candidates) == 1:
            candidate = candidates[0]
            aligned[name] = raw[candidate]
            used.add(candidate)
            aliases[candidate] = name
        else:
            aligned[name] = None
            missing.append(name)
    extra = sorted(set(raw) - used)
    if aliases or missing or extra:
        LOGGER.warning(
            "Stage 2 extraction normalized feature keys row_id=%s aliases=%s "
            "missing_as_null=%s extras_dropped=%s",
            row_id,
            aliases,
            missing,
            extra,
        )
    if set(aligned) != expected:  # pragma: no cover - defensive invariant
        raise RuntimeError("Stage 2 feature-key normalization changed the expected schema")
    return aligned


def _canonical_category(value: Any, declared: Sequence[str]) -> str | None:
    text = str(value).strip()
    if text in declared:
        return text
    key = re.sub(r"[^a-z0-9]+", "", text.lower())
    matches = [
        category for category in declared if re.sub(r"[^a-z0-9]+", "", category.lower()) == key
    ]
    if len(matches) == 1:
        return matches[0]
    numeric_tokens = re.findall(r"(?<!\d)-?\d+(?:\.\d+)?(?!\d)", text)
    if len(numeric_tokens) == 1:
        numeric_matches = [
            category
            for category in declared
            if re.findall(r"(?<!\d)-?\d+(?:\.\d+)?(?!\d)", category) == numeric_tokens
        ]
        if len(numeric_matches) == 1:
            return numeric_matches[0]
    return None


class _ExtractionCategoryError(ValueError):
    """Closed-vocabulary extraction failures with enough state for safe recovery."""

    def __init__(
        self,
        *,
        issues: Sequence[Mapping[str, Any]],
        response: Mapping[str, Any],
    ) -> None:
        self.issues = tuple(dict(issue) for issue in issues)
        self.response = copy.deepcopy(dict(response))
        first = self.issues[0]
        allowed = json.dumps(
            first["allowed_categories"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        suffix = (
            f"; {len(self.issues) - 1} additional invalid categorical value(s)"
            if len(self.issues) > 1
            else ""
        )
        super().__init__(
            f"feature {first['feature_name']!r} value "
            f"{first['prior_extracted_value']!r} is invalid; allowed values are "
            f"{allowed} or null{suffix}"
        )


class _ExtractionValueError(ValueError):
    """Scalar/type extraction failures that can be repaired feature by feature."""

    def __init__(
        self,
        *,
        issues: Sequence[Mapping[str, Any]],
        response: Mapping[str, Any],
    ) -> None:
        self.issues = tuple(dict(issue) for issue in issues)
        self.response = copy.deepcopy(dict(response))
        first = self.issues[0]
        suffix = (
            f"; {len(self.issues) - 1} additional invalid value(s)" if len(self.issues) > 1 else ""
        )
        super().__init__(f"feature {first['feature_name']!r} {first['reason']}" f"{suffix}")


def _extraction_error_from_exception(
    exc: BaseException,
    error_type: type[_ExtractionCategoryError] | type[_ExtractionValueError],
) -> _ExtractionCategoryError | _ExtractionValueError | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, error_type):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _category_error_from_exception(exc: BaseException) -> _ExtractionCategoryError | None:
    result = _extraction_error_from_exception(exc, _ExtractionCategoryError)
    return result if isinstance(result, _ExtractionCategoryError) else None


def _value_error_from_exception(exc: BaseException) -> _ExtractionValueError | None:
    result = _extraction_error_from_exception(exc, _ExtractionValueError)
    return result if isinstance(result, _ExtractionValueError) else None


def _null_invalid_extraction_values(
    error: _ExtractionValueError,
) -> dict[str, Any]:
    patched = copy.deepcopy(error.response)
    rows_by_id = {int(row["row_id"]): row for row in patched["rows"]}
    for issue in error.issues:
        rows_by_id[int(issue["row_id"])]["values"][str(issue["feature_name"])] = None
    return patched


def _category_ontology_plan(
    error: _ExtractionCategoryError,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Deduplicate equal invalid values while preserving their response targets."""

    item_by_key: dict[str, dict[str, Any]] = {}
    targets_by_key: dict[str, list[dict[str, Any]]] = {}
    for issue in error.issues:
        key = json.dumps(
            {
                "feature_name": issue["feature_name"],
                "prior_extracted_value": issue["prior_extracted_value"],
                "allowed_categories": issue["allowed_categories"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if key not in item_by_key:
            definition = issue["definition"]
            item_by_key[key] = {
                "mapping_id": "",
                "feature_name": str(issue["feature_name"]),
                "value_type": str(definition.get("value_type") or "categorical"),
                "description": str(definition.get("description") or ""),
                "measurement_definition": str(definition.get("measurement_definition") or ""),
                "missing_value_rule": str(definition.get("missing_value_rule") or ""),
                "allowed_categories": list(issue["allowed_categories"]),
                "prior_extracted_value": issue["prior_extracted_value"],
                "occurrence_count": 0,
            }
            targets_by_key[key] = []
        item_by_key[key]["occurrence_count"] += 1
        targets_by_key[key].append(
            {
                "row_id": int(issue["row_id"]),
                "feature_name": str(issue["feature_name"]),
            }
        )

    items: list[dict[str, Any]] = []
    targets: dict[str, list[dict[str, Any]]] = {}
    for index, key in enumerate(sorted(item_by_key), start=1):
        mapping_id = f"category_mapping_{index:04d}"
        item = dict(item_by_key[key])
        item["mapping_id"] = mapping_id
        items.append(item)
        targets[mapping_id] = list(targets_by_key[key])
    return items, targets


def _category_ontology_prompt(items: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Ask the configured Stage 2 LLM to normalize values without patient text."""

    body = {
        "job": "map_extracted_values_to_declared_category_ontology",
        "rules": [
            "Use only the feature definition, allowed categories, and prior extracted value.",
            "Do not perform clinical extraction and do not infer any new patient information.",
            "Map by semantic equivalence to exactly one allowed category.",
            "Return null when the prior value does not map unambiguously.",
            "Return every mapping_id exactly once and no additional mapping IDs.",
        ],
        "items": [dict(item) for item in items],
        "response": {
            "corrections": [
                {
                    "mapping_id": "one supplied mapping_id",
                    "value": "one exact allowed category or null",
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You normalize previously extracted categorical values to a closed ontology. "
                "Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _validate_category_ontology(
    value: Mapping[str, Any],
    *,
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    raw_corrections = value.get("corrections")
    if not isinstance(raw_corrections, list):
        raise ValueError("category ontology response requires a corrections array")
    item_by_id = {str(item["mapping_id"]): item for item in items}
    corrections: dict[str, Any] = {}
    for raw in raw_corrections:
        if not isinstance(raw, Mapping):
            raise ValueError("each category ontology correction must be an object")
        mapping_id = str(raw.get("mapping_id") or "")
        if mapping_id not in item_by_id or mapping_id in corrections:
            raise ValueError("category ontology returned an unknown or duplicate mapping_id")
        if "value" not in raw:
            raise ValueError(f"category mapping {mapping_id!r} omitted its value")
        extracted = _clean_scalar(raw.get("value"))
        declared = list(item_by_id[mapping_id]["allowed_categories"])
        if extracted is None:
            corrections[mapping_id] = None
            continue
        if _is_missing_scalar(extracted) and _canonical_category(extracted, declared) is None:
            corrections[mapping_id] = None
            continue
        canonical = _canonical_category(extracted, declared)
        if canonical is None:
            allowed = json.dumps(declared, ensure_ascii=False, separators=(",", ":"))
            raise ValueError(
                f"category mapping {mapping_id!r} returned {extracted!r}; "
                f"allowed values are {allowed} or null"
            )
        corrections[mapping_id] = canonical
    if set(corrections) != set(item_by_id):
        raise ValueError("category ontology response omitted one or more mapping IDs")
    return {
        "corrections": [
            {"mapping_id": mapping_id, "value": corrections[mapping_id]}
            for mapping_id in item_by_id
        ]
    }


def _apply_category_corrections(
    response: Mapping[str, Any],
    *,
    corrections: Mapping[str, Any],
    targets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    patched = copy.deepcopy(dict(response))
    rows_by_id = {int(row["row_id"]): row for row in patched["rows"]}
    values_by_mapping = {
        str(row["mapping_id"]): row.get("value") for row in corrections["corrections"]
    }
    for mapping_id, mapping_targets in targets.items():
        for target in mapping_targets:
            rows_by_id[int(target["row_id"])]["values"][str(target["feature_name"])] = (
                values_by_mapping[mapping_id]
            )
    return patched


def _request_validated_extraction(
    *,
    messages: Sequence[Mapping[str, str]],
    row_ids: Sequence[int],
    definitions: Sequence[Mapping[str, Any]],
    request_json: RequestJSON,
    ontology_audit_path: Path,
) -> dict[str, Any]:
    """Extract rows, then recover closed-category failures without resending notes."""

    issue_audit_path = ontology_audit_path.with_name("extraction_issues.json")
    issue_events: list[dict[str, Any]] = []
    try:
        validated = request_json(
            messages,
            lambda value: _validate_extraction(
                value,
                row_ids=row_ids,
                definitions=definitions,
            ),
            request_kind="extraction",
        )
        _write_json(
            issue_audit_path,
            {
                "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                "completed_at": _now(),
                "events": [],
            },
        )
        return validated
    except ValueError as exc:
        value_error = _value_error_from_exception(exc)
        value_repair_audit: dict[str, Any] | None = None
        category_error = _category_error_from_exception(exc)
        if value_error is not None:
            issue_events.extend(
                {
                    "failure_kind": "invalid_scalar_or_value_type",
                    "row_id": int(issue["row_id"]),
                    "feature_name": str(issue["feature_name"]),
                    "reason": str(issue["reason"]),
                    "prior_extracted_value": copy.deepcopy(issue.get("prior_extracted_value")),
                }
                for issue in value_error.issues
            )
            patched = _null_invalid_extraction_values(value_error)
            value_repair_audit = {
                "schema_version": "stage2_invalid_feature_value_repair_v1",
                "resolution": "conservative_invalid_features_null",
                "original_validation_error": str(exc),
                "issues": [dict(issue) for issue in value_error.issues],
            }
            try:
                validated = _validate_extraction(
                    patched,
                    row_ids=row_ids,
                    definitions=definitions,
                )
            except _ExtractionCategoryError as patched_category_error:
                category_error = patched_category_error
            else:
                _write_json(
                    ontology_audit_path.with_name("invalid_feature_value_repair.json"),
                    value_repair_audit,
                )
                LOGGER.warning(
                    "Stage 2 extraction retained the valid fields and replaced %s "
                    "invalid feature value(s) with null",
                    len(value_error.issues),
                )
                _write_json(
                    issue_audit_path,
                    {
                        "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                        "completed_at": _now(),
                        "events": issue_events,
                    },
                )
                return validated
        if category_error is None:
            feature_names = [str(definition["name"]) for definition in definitions]
            conservative = {
                "rows": [
                    {
                        "row_id": int(row_id),
                        "values": {name: None for name in feature_names},
                    }
                    for row_id in row_ids
                ]
            }
            validated = _validate_extraction(
                conservative,
                row_ids=row_ids,
                definitions=definitions,
            )
            failure_path = ontology_audit_path.with_name("extraction_failure.json")
            _write_json(
                failure_path,
                {
                    "schema_version": "stage2_extraction_failure_v1",
                    "resolution": "conservative_all_null",
                    "failed_at": _now(),
                    "row_ids": [int(row_id) for row_id in row_ids],
                    "feature_names": feature_names,
                    "validation_error": f"{type(exc).__name__}: {exc}",
                },
            )
            issue_events.extend(
                {
                    "failure_kind": "structural_response_failure",
                    "row_id": int(row_id),
                    "feature_name": None,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                for row_id in row_ids
            )
            _write_json(
                issue_audit_path,
                {
                    "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                    "completed_at": _now(),
                    "events": issue_events,
                },
            )
            LOGGER.warning(
                "Stage 2 extraction remained structurally invalid after repairs; "
                "replacing %s extracted value(s) for %s patient(s) with null (%s: %s)",
                len(feature_names) * len(row_ids),
                len(row_ids),
                type(exc).__name__,
                exc,
            )
            return validated

    issue_events.extend(
        {
            "failure_kind": "out_of_ontology_category",
            "row_id": int(issue["row_id"]),
            "feature_name": str(issue["feature_name"]),
            "reason": "extracted value is outside the declared closed ontology",
            "prior_extracted_value": copy.deepcopy(issue.get("prior_extracted_value")),
            "allowed_categories": list(issue.get("allowed_categories") or []),
        }
        for issue in category_error.issues
    )
    items, targets = _category_ontology_plan(category_error)
    LOGGER.warning(
        "Stage 2 extraction exhausted ordinary repairs for %s invalid categorical "
        "value(s); requesting note-free ontology mapping",
        len(category_error.issues),
    )
    ontology_error: str | None = None
    try:
        corrections = request_json(
            _category_ontology_prompt(items),
            lambda value: _validate_category_ontology(value, items=items),
            request_kind="extraction",
        )
        resolution = "llm_category_ontology"
    except ValueError as exc:
        ontology_error = f"{type(exc).__name__}: {exc}"
        resolution = "conservative_null"
        corrections = {
            "corrections": [
                {"mapping_id": str(item["mapping_id"]), "value": None} for item in items
            ]
        }
        LOGGER.warning(
            "Stage 2 category ontology mapping remained invalid; replacing %s "
            "unmappable categorical value(s) with null (%s)",
            len(category_error.issues),
            ontology_error,
        )

    patched = _apply_category_corrections(
        category_error.response,
        corrections=corrections,
        targets=targets,
    )
    validated = _validate_extraction(
        patched,
        row_ids=row_ids,
        definitions=definitions,
    )
    if value_repair_audit is not None:
        _write_json(
            ontology_audit_path.with_name("invalid_feature_value_repair.json"),
            value_repair_audit,
        )
    _write_json(
        ontology_audit_path,
        {
            "schema_version": "stage2_category_ontology_repair_v1",
            "resolution": resolution,
            "original_validation_error": str(category_error),
            "ontology_validation_error": ontology_error,
            "items": items,
            "targets": targets,
            "corrections": corrections["corrections"],
        },
    )
    _write_json(
        issue_audit_path,
        {
            "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
            "completed_at": _now(),
            "events": issue_events,
        },
    )
    return validated


def _legacy_extraction_issue_audit(directory: Path) -> dict[str, Any]:
    """Reconstruct the issue ledger from pre-ledger extraction audit files."""

    events: list[dict[str, Any]] = []

    def read_mapping(filename: str) -> Mapping[str, Any] | None:
        path = directory / filename
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, Mapping) else None

    value_audit = read_mapping("invalid_feature_value_repair.json")
    if value_audit is not None:
        for issue in value_audit.get("issues") or []:
            if not isinstance(issue, Mapping):
                continue
            try:
                row_id = int(issue["row_id"])
            except (KeyError, TypeError, ValueError):
                continue
            events.append(
                {
                    "failure_kind": "invalid_scalar_or_value_type",
                    "row_id": row_id,
                    "feature_name": str(issue.get("feature_name") or ""),
                    "reason": str(issue.get("reason") or ""),
                    "prior_extracted_value": copy.deepcopy(issue.get("prior_extracted_value")),
                }
            )

    category_audit = read_mapping("category_ontology_repair.json")
    if category_audit is not None:
        items_by_id = {
            str(item.get("mapping_id") or ""): item
            for item in category_audit.get("items") or []
            if isinstance(item, Mapping) and str(item.get("mapping_id") or "")
        }
        targets_by_id = category_audit.get("targets") or {}
        if isinstance(targets_by_id, Mapping):
            for mapping_id, raw_targets in targets_by_id.items():
                item = items_by_id.get(str(mapping_id))
                if item is None or not isinstance(raw_targets, list):
                    continue
                for target in raw_targets:
                    if not isinstance(target, Mapping):
                        continue
                    try:
                        row_id = int(target["row_id"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    events.append(
                        {
                            "failure_kind": "out_of_ontology_category",
                            "row_id": row_id,
                            "feature_name": str(
                                target.get("feature_name") or item.get("feature_name") or ""
                            ),
                            "reason": ("extracted value is outside the declared closed ontology"),
                            "prior_extracted_value": copy.deepcopy(
                                item.get("prior_extracted_value")
                            ),
                            "allowed_categories": list(item.get("allowed_categories") or []),
                        }
                    )

    structural_audit = read_mapping("extraction_failure.json")
    if structural_audit is not None:
        for raw_row_id in structural_audit.get("row_ids") or []:
            try:
                row_id = int(raw_row_id)
            except (TypeError, ValueError):
                continue
            events.append(
                {
                    "failure_kind": "structural_response_failure",
                    "row_id": row_id,
                    "feature_name": None,
                    "reason": str(structural_audit.get("validation_error") or ""),
                }
            )

    return {
        "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
        "completed_at": _now(),
        "reconstructed_from_legacy_audits": True,
        "events": events,
    }


def _ensure_extraction_issue_audit(directory: Path) -> dict[str, Any]:
    """Return an issue ledger, reconstructing it for a compatible old checkpoint."""

    path = directory / "extraction_issues.json"
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping) and isinstance(value.get("events"), list):
                return dict(value)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    reconstructed = _legacy_extraction_issue_audit(directory)
    _write_json(path, reconstructed)
    return reconstructed


def _extraction_prompt(
    *,
    definitions: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if len(rows) != 1:
        raise ValueError("Stage 2 extraction prompts must contain exactly one patient's record")
    body = {
        "job": "extract_stage2_patient_variables",
        "rules": [
            "Use only the supplied clinical text for the patient in that row.",
            "Apply the measurement definition and missing-value rule literally.",
            "For a binary, categorical, or ordinal feature, return one declared category exactly.",
            "Do not substitute 0/1 or true/false for a declared category unless that "
            "exact value is declared.",
            *_SCALAR_EXTRACTION_RULES,
            "Return null when the record does not support a value.",
            "Return every row and every feature exactly once.",
        ],
        "features": _prompt_feature_definitions(definitions),
        "patients": list(rows),
        "response": {
            "rows": [
                {
                    "row_id": "one supplied integer row_id",
                    "values": {"every supplied feature name": "scalar value or null"},
                }
            ]
        },
    }
    # Keep clinical text as Unicode. ASCII escaping can expand multilingual
    # notes several-fold and consumes model tokens on literal ``\\uXXXX``
    # sequences without adding information.
    return [
        {
            "role": "system",
            "content": "You extract prespecified variables from supplied clinical text. Return JSON only.",
        },
        {
            "role": "user",
            "content": json.dumps(body, sort_keys=True, ensure_ascii=False),
        },
    ]


def _prompt_chars(messages: Sequence[Mapping[str, str]]) -> int:
    """Return the exact rendered content characters sent to the endpoint."""

    return sum(len(str(message.get("content") or "")) for message in messages)


def _page_reconciliation_prompt(
    *,
    definitions: Sequence[Mapping[str, Any]],
    row_id: int,
    page_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    body = {
        "job": "reconcile_stage2_patient_variable_pages",
        "rules": [
            "Every supplied page was extracted from a lossless contiguous span of one note.",
            "Review every page result and apply each feature's measurement and missing-value rules.",
            "A null page does not override a supported value on another page.",
            "Resolve multiple supported values using document order and the specified temporal or aggregation rule.",
            "Do not invent evidence that is absent from all page results.",
            "For a binary, categorical, or ordinal feature, return one declared category exactly.",
            "Do not substitute 0/1 or true/false for a declared category unless that "
            "exact value is declared.",
            *_SCALAR_EXTRACTION_RULES,
            "Return every feature exactly once for the original row_id.",
        ],
        "features": _prompt_feature_definitions(definitions),
        "row_id": int(row_id),
        "page_results": list(page_results),
        "response": {
            "rows": [
                {
                    "row_id": int(row_id),
                    "values": {"every supplied feature name": "scalar value or null"},
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You reconcile complete-note page extractions without dropping any page. "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(body, sort_keys=True, ensure_ascii=False),
        },
    ]


def _page_results_for_definitions(
    page_results: Sequence[Mapping[str, Any]],
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain every page while projecting values to one feature subset."""

    feature_names = [str(definition["name"]) for definition in definitions]
    return [
        {
            **{key: value for key, value in page.items() if key != "values"},
            "values": {name: dict(page.get("values") or {}).get(name) for name in feature_names},
        }
        for page in page_results
    ]


def _partition_feature_definitions(
    definitions: Sequence[Mapping[str, Any]],
    *,
    feature_batch_size: int,
) -> list[list[Mapping[str, Any]]]:
    """Return stable consecutive feature slices for extraction prompts."""

    if (
        isinstance(feature_batch_size, bool)
        or not isinstance(feature_batch_size, int)
        or feature_batch_size < 1
    ):
        raise ValueError("feature_batch_size must be a positive integer")
    return [
        list(definitions[start : start + feature_batch_size])
        for start in range(0, len(definitions), feature_batch_size)
    ]


def _partition_page_reconciliation_definitions(
    *,
    definitions: Sequence[Mapping[str, Any]],
    row_id: int,
    page_results: Sequence[Mapping[str, Any]],
    max_prompt_chars: int,
    feature_batch_size: int,
) -> list[list[Mapping[str, Any]]]:
    """Partition only features; every batch continues to see every note page."""

    batches: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for definition in definitions:
        singleton_results = _page_results_for_definitions(page_results, [definition])
        singleton_messages = _page_reconciliation_prompt(
            definitions=[definition],
            row_id=row_id,
            page_results=singleton_results,
        )
        singleton_prompt_chars = _prompt_chars(singleton_messages)
        if singleton_prompt_chars > int(max_prompt_chars):
            raise ValueError(
                "Stage 2 cannot reconcile every lossless note page for one feature "
                f"within max_prompt_chars ({singleton_prompt_chars} > {max_prompt_chars}); "
                "increase the prompt budget"
            )
        proposed = [*current, definition]
        proposed_results = _page_results_for_definitions(page_results, proposed)
        messages = _page_reconciliation_prompt(
            definitions=proposed,
            row_id=row_id,
            page_results=proposed_results,
        )
        prompt_chars = _prompt_chars(messages)
        if current and (
            len(proposed) > int(feature_batch_size)
            or prompt_chars > int(max_prompt_chars)
        ):
            batches.append(current)
            current = [definition]
        else:
            current = proposed
    if current:
        batches.append(current)
    return batches


def _validate_extraction(
    value: Mapping[str, Any],
    *,
    row_ids: Sequence[int],
    definitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError("extraction response requires a rows array")
    expected_rows = {int(row_id) for row_id in row_ids}
    feature_names = [str(feature["name"]) for feature in definitions]
    by_row: dict[int, dict[str, Any]] = {}
    definitions_by_name = {str(feature["name"]): feature for feature in definitions}
    category_issues: list[dict[str, Any]] = []
    value_issues: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("each extraction row must be an object")
        row_id = int(raw.get("row_id"))
        if row_id not in expected_rows or row_id in by_row:
            raise ValueError("extraction returned an unknown or duplicate row_id")
        values = raw.get("values")
        if not isinstance(values, Mapping):
            raise ValueError("each extraction row requires a values object")
        aligned_values = _aligned_extraction_values(
            values,
            feature_names=feature_names,
            row_id=row_id,
        )
        clean_values: dict[str, Any] = {}
        for name in feature_names:
            definition = definitions_by_name[name]
            value_type = str(definition.get("value_type") or "ambiguous")
            declared = _declared_categories(definition)
            try:
                extracted = _clean_scalar(aligned_values[name])
            except ValueError:
                extracted = aligned_values[name]
                value_issues.append(
                    {
                        "row_id": row_id,
                        "feature_name": name,
                        "value_type": value_type,
                        "reason": (
                            "requires one scalar JSON value or null, but the model "
                            f"returned {type(extracted).__name__}"
                        ),
                        "prior_extracted_value": extracted,
                    }
                )
                clean_values[name] = extracted
                continue
            if _is_missing_scalar(extracted) and _canonical_category(extracted, declared) is None:
                extracted = None
            if extracted is not None and value_type == "continuous":
                if isinstance(extracted, bool) or not isinstance(extracted, (int, float, str)):
                    value_issues.append(
                        {
                            "row_id": row_id,
                            "feature_name": name,
                            "value_type": value_type,
                            "reason": (
                                "requires one JSON number, one documented categorical or "
                                "threshold string, or null"
                            ),
                            "prior_extracted_value": extracted,
                        }
                    )
                    clean_values[name] = extracted
                    continue
                if isinstance(extracted, (int, float)):
                    extracted = float(extracted)
                else:
                    extracted = extracted.strip()
            elif extracted is not None and value_type in {"binary", "categorical", "ordinal"}:
                canonical = _canonical_category(extracted, declared) if declared else str(extracted)
                if canonical is None:
                    category_issues.append(
                        {
                            "row_id": row_id,
                            "feature_name": name,
                            "prior_extracted_value": extracted,
                            "allowed_categories": list(declared),
                            "definition": dict(definition),
                        }
                    )
                else:
                    extracted = canonical
            clean_values[name] = extracted
        by_row[row_id] = {"row_id": row_id, "values": clean_values}
    if set(by_row) != expected_rows:
        raise ValueError("extraction response omitted one or more supplied rows")
    normalized_response = {"rows": [by_row[int(row_id)] for row_id in row_ids]}
    if value_issues:
        raise _ExtractionValueError(
            issues=value_issues,
            response=normalized_response,
        )
    if category_issues:
        raise _ExtractionCategoryError(issues=category_issues, response=normalized_response)
    return normalized_response


def _partition_rows_for_prompt(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_prompt_chars: int,
    definition_batches: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[list[list[Mapping[str, Any]]], list[Mapping[str, Any]]]:
    """Create only singleton extraction requests, measured at exact prompt size.

    Rows that cannot fit even by themselves are returned separately for
    lossless page planning.  This avoids the old oversized-singleton hole in
    the approximate JSON-size partitioner.  Multi-patient prompts are forbidden
    by construction as well as by ``_extraction_prompt``.
    """

    batches: list[list[Mapping[str, Any]]] = []
    oversized: list[Mapping[str, Any]] = []
    for row in rows:
        prompt_sizes = (
            _prompt_chars(
                _extraction_prompt(
                    definitions=batch_definitions,
                    rows=[row],
                )
            )
            for batch_definitions in definition_batches
        )
        if any(size > int(max_prompt_chars) for size in prompt_sizes):
            oversized.append(row)
        else:
            batches.append([row])
    return batches, oversized


def _lossless_extraction_pages(
    row: Mapping[str, Any],
    *,
    definition_batches: Sequence[Sequence[Mapping[str, Any]]],
    max_prompt_chars: int,
) -> list[dict[str, Any]]:
    """Split one note into the largest exact prompt-sized contiguous pages."""

    source = str(row.get("text") or "")
    row_id = int(row["row_id"])
    if not source:
        raise ValueError(
            "an empty Stage 2 row exceeded the prompt budget before note text was added; "
            "increase stage2.extraction_max_prompt_chars or shorten the feature definitions"
        )
    pages: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(source):
        low = cursor + 1
        high = len(source)
        best: dict[str, Any] | None = None
        while low <= high:
            end = (low + high) // 2
            candidate = {
                "row_id": row_id,
                "text": source[cursor:end],
                "page": {
                    "page_index": len(pages) + 1,
                    "char_start": cursor,
                    "char_end": end,
                    "document_chars": len(source),
                },
            }
            prompt_sizes = (
                _prompt_chars(
                    _extraction_prompt(
                        definitions=batch_definitions,
                        rows=[candidate],
                    )
                )
                for batch_definitions in definition_batches
            )
            if all(size <= int(max_prompt_chars) for size in prompt_sizes):
                best = candidate
                low = end + 1
            else:
                high = end - 1
        if best is None:
            raise ValueError(
                "Stage 2 feature definitions and prompt envelope leave no room for even "
                "one source character; increase stage2.extraction_max_prompt_chars or "
                "shorten the feature definitions"
            )
        pages.append(best)
        cursor = int(best["page"]["char_end"])
    if "".join(str(page["text"]) for page in pages) != source:
        raise RuntimeError("Stage 2 lossless page planner changed patient text")
    return pages


def _summarize_extraction_failures(
    *,
    output_dir: Path,
    definitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate feature-attributable failures across distinct patients."""

    definition_names = {str(definition["name"]) for definition in definitions}
    patterns: dict[tuple[str, str, str], dict[str, Any]] = {}
    structural_rows: set[int] = set()
    issue_files = sorted(output_dir.rglob("extraction_issues.json"))
    for path in issue_files:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            audit = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(audit, Mapping):
            continue
        for raw_event in audit.get("events") or []:
            if not isinstance(raw_event, Mapping):
                continue
            try:
                row_id = int(raw_event["row_id"])
            except (KeyError, TypeError, ValueError):
                continue
            feature_name = str(raw_event.get("feature_name") or "")
            failure_kind = str(raw_event.get("failure_kind") or "unknown")
            reason = str(raw_event.get("reason") or "")
            if not feature_name:
                structural_rows.add(row_id)
                continue
            if feature_name not in definition_names:
                continue
            signature_reason = reason if failure_kind == "invalid_scalar_or_value_type" else ""
            key = (feature_name, failure_kind, signature_reason)
            pattern = patterns.setdefault(
                key,
                {
                    "feature_name": feature_name,
                    "failure_kind": failure_kind,
                    "reason": reason,
                    "patient_row_ids": set(),
                    "example_values": [],
                    "allowed_categories": list(raw_event.get("allowed_categories") or []),
                },
            )
            pattern["patient_row_ids"].add(row_id)
            if "prior_extracted_value" in raw_event:
                example = copy.deepcopy(raw_event.get("prior_extracted_value"))
                example_key = json.dumps(
                    example,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                existing_keys = {
                    json.dumps(
                        value,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    for value in pattern["example_values"]
                }
                if example_key not in existing_keys and len(pattern["example_values"]) < 12:
                    pattern["example_values"].append(example)

    rendered_patterns = []
    for pattern in patterns.values():
        row_ids = sorted(pattern.pop("patient_row_ids"))
        rendered_patterns.append(
            {
                **pattern,
                "patient_count": len(row_ids),
                "patient_row_ids": row_ids,
            }
        )
    rendered_patterns.sort(
        key=lambda pattern: (
            -int(pattern["patient_count"]),
            str(pattern["feature_name"]),
            str(pattern["failure_kind"]),
            str(pattern["reason"]),
        )
    )
    summary = {
        "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
        "completed_at": _now(),
        "issue_files": len(issue_files),
        "feature_failure_patterns": rendered_patterns,
        "structural_failure_patient_count": len(structural_rows),
        "structural_failure_patient_row_ids": sorted(structural_rows),
    }
    _write_json(output_dir / "failure_summary.json", summary)
    return summary


def extract_rows(
    *,
    dataset: pd.DataFrame,
    row_ids: Sequence[int],
    text_column: str,
    definitions: Sequence[Mapping[str, Any]],
    output_dir: Path,
    request_json: RequestJSON,
    workers: int,
    max_prompt_chars: int,
    feature_batch_size: int = DEFAULT_EXTRACTION_FEATURE_BATCH_SIZE,
) -> pd.DataFrame:
    """Extract one patient per prompt in bounded feature slices, then merge them."""

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = [str(feature["name"]) for feature in definitions]
    definition_batches = _partition_feature_definitions(
        definitions,
        feature_batch_size=feature_batch_size,
    )
    if not definitions:
        frame = pd.DataFrame({"_oci_row_id": [int(value) for value in row_ids]})
        _write_frame(output_dir / "extracted.csv", frame)
        failure_summary = _summarize_extraction_failures(
            output_dir=output_dir,
            definitions=definitions,
        )
        _write_json(
            output_dir / "complete.json",
            {
                "status": "complete",
                "rows": len(frame),
                "feature_failure_patterns": len(failure_summary["feature_failure_patterns"]),
                "structural_failure_patients": failure_summary["structural_failure_patient_count"],
            },
        )
        return frame

    request_rows = [
        {
            "row_id": int(row_id),
            "text": (
                ""
                if pd.isna(dataset.iloc[int(row_id)][text_column])
                else str(dataset.iloc[int(row_id)][text_column])
            ),
        }
        for row_id in row_ids
    ]
    extraction_definitions = _prompt_feature_definitions(definitions)
    batches, oversized_rows = _partition_rows_for_prompt(
        request_rows,
        max_prompt_chars=int(max_prompt_chars),
        definition_batches=definition_batches,
    )

    page_requests: list[dict[str, Any]] = []
    for row in oversized_rows:
        page_requests.extend(
            _lossless_extraction_pages(
                row,
                definition_batches=definition_batches,
                max_prompt_chars=int(max_prompt_chars),
            )
        )

    if len(definition_batches) > 1:
        LOGGER.info(
            "Stage 2 extraction features=%s feature_batch_size=%s "
            "feature_batches_per_patient=%s patients=%s pages=%s",
            len(definitions),
            feature_batch_size,
            len(definition_batches),
            len(batches),
            len(page_requests),
        )

    # Snapshot every saved singleton result before concurrent workers begin
    # rewriting the newly numbered batch directories.  Removing an old
    # multi-patient batch shifts every later singleton's batch number, but its
    # row_id remains a stable identity that lets us safely relocate it.
    singleton_checkpoints: dict[int, list[dict[str, Any]]] = {}
    for saved_dir in sorted((output_dir / "batches").glob("batch_*")):
        if not saved_dir.is_dir() or saved_dir.is_symlink():
            continue
        saved_complete_path = saved_dir / "complete.json"
        saved_result_path = saved_dir / "result.json"
        saved_manifest_path = saved_dir / "row_ids.json"
        saved_audit_path = saved_dir / "category_ontology_repair.json"
        saved_issue_path = saved_dir / "extraction_issues.json"
        if not (
            saved_complete_path.is_file()
            and saved_result_path.is_file()
            and saved_manifest_path.is_file()
        ):
            continue
        if _stale_category_ontology_audit(saved_audit_path) is not None:
            continue
        try:
            saved_completion = json.loads(saved_complete_path.read_text(encoding="utf-8"))
            saved_result = json.loads(saved_result_path.read_text(encoding="utf-8"))
            saved_row_ids = json.loads(saved_manifest_path.read_text(encoding="utf-8"))
            saved_audit = (
                json.loads(saved_audit_path.read_text(encoding="utf-8"))
                if saved_audit_path.is_file()
                else None
            )
            saved_issues = (
                json.loads(saved_issue_path.read_text(encoding="utf-8"))
                if saved_issue_path.is_file()
                else _legacy_extraction_issue_audit(saved_dir)
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            not isinstance(saved_completion, Mapping)
            or saved_completion.get("status") != "complete"
            or not isinstance(saved_result, Mapping)
            or not isinstance(saved_row_ids, list)
            or len(saved_row_ids) != 1
        ):
            continue
        try:
            saved_row_id = int(saved_row_ids[0])
        except (TypeError, ValueError):
            continue
        singleton_checkpoints.setdefault(saved_row_id, []).append(
            {
                "source_dir": saved_dir,
                "completion": dict(saved_completion),
                "result": dict(saved_result),
                "category_ontology_audit": saved_audit,
                "extraction_issues": saved_issues,
            }
        )

    def request_feature_batches(
        *,
        parent_dir: Path,
        row: Mapping[str, Any],
        parent_schema_version: str,
    ) -> dict[str, Any]:
        """Extract and checkpoint each feature slice for one patient or page."""

        if len(definition_batches) < 2:  # pragma: no cover - caller invariant
            raise RuntimeError("feature-batched extraction requires multiple feature batches")
        row_id = int(row["row_id"])
        merged_values: dict[str, Any] = {}
        for feature_batch_index, batch_definitions in enumerate(
            definition_batches,
            start=1,
        ):
            feature_dir = (
                parent_dir
                / "feature_batches"
                / f"batch_{feature_batch_index:05d}"
            )
            result_path = feature_dir / "result.json"
            complete_path = feature_dir / "complete.json"
            ontology_audit_path = feature_dir / "category_ontology_repair.json"
            batch_input = {
                "schema_version": EXTRACTION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION,
                "parent_schema_version": parent_schema_version,
                "definitions": _prompt_feature_definitions(batch_definitions),
                "row": dict(row),
            }
            input_fingerprint = _value_fingerprint(batch_input)
            stale_audit = _stale_category_ontology_audit(ontology_audit_path)
            result: dict[str, Any] | None = None
            if complete_path.is_file() and result_path.is_file() and stale_audit is None:
                try:
                    completion = json.loads(complete_path.read_text(encoding="utf-8"))
                    cached = json.loads(result_path.read_text(encoding="utf-8"))
                    if (
                        isinstance(completion, Mapping)
                        and completion.get("schema_version")
                        == EXTRACTION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION
                        and completion.get("input_fingerprint") == input_fingerprint
                        and isinstance(cached, Mapping)
                    ):
                        result = _validate_extraction(
                            cached,
                            row_ids=[row_id],
                            definitions=batch_definitions,
                        )
                        _ensure_extraction_issue_audit(feature_dir)
                except (
                    KeyError,
                    OSError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    result = None
                if result is None:
                    LOGGER.info(
                        "rerun incompatible Stage 2 extraction feature batch: %s",
                        feature_dir,
                    )
            if result is None:
                if stale_audit is not None:
                    LOGGER.info(
                        "retry stale Stage 2 category ontology feature batch: %s",
                        feature_dir,
                    )
                feature_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    feature_dir / "input.json",
                    {**batch_input, "input_fingerprint": input_fingerprint},
                )
                messages = _extraction_prompt(
                    definitions=batch_definitions,
                    rows=[row],
                )
                if _prompt_chars(messages) > int(max_prompt_chars):  # pragma: no cover
                    raise RuntimeError(
                        "Stage 2 extraction planner emitted an oversized feature batch"
                    )
                result = _request_validated_extraction(
                    messages=messages,
                    row_ids=[row_id],
                    definitions=batch_definitions,
                    request_json=request_json,
                    ontology_audit_path=ontology_audit_path,
                )
                _write_json(result_path, result)
                _supersede_stale_category_ontology_audit(
                    ontology_audit_path,
                    previous=stale_audit,
                )
                _write_json(
                    complete_path,
                    {
                        "status": "complete",
                        "schema_version": (
                            EXTRACTION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION
                        ),
                        "input_fingerprint": input_fingerprint,
                        "completed_at": _now(),
                        "row_id": row_id,
                        "features": len(batch_definitions),
                        "feature_batch": feature_batch_index,
                    },
                )
            merged_values.update(dict(result["rows"][0]["values"]))

        merged = _validate_extraction(
            {"rows": [{"row_id": row_id, "values": merged_values}]},
            row_ids=[row_id],
            definitions=definitions,
        )
        # Any parent-level issue ledger belongs to an older all-feature request.
        # Active issue ledgers now live beside the feature-slice checkpoints.
        _write_json(
            parent_dir / "extraction_issues.json",
            {
                "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                "completed_at": _now(),
                "events": [],
                "delegated_to_feature_batches": True,
            },
        )
        return merged

    def run_batch(index: int, batch: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if len(batch) != 1:  # pragma: no cover - enforced by the planner
            raise RuntimeError("Stage 2 extraction planner created a multi-patient batch")
        batch_dir = output_dir / "batches" / f"batch_{index:05d}"
        result_path = batch_dir / "result.json"
        complete_path = batch_dir / "complete.json"
        ontology_audit_path = batch_dir / "category_ontology_repair.json"
        row_ids = [int(row["row_id"]) for row in batch]
        input_fingerprint = _value_fingerprint(
            {
                "schema_version": EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
                "definitions": extraction_definitions,
                "rows": list(batch),
            }
        )
        stale_audit = _stale_category_ontology_audit(ontology_audit_path)
        candidates = list(singleton_checkpoints.get(row_ids[0], []))
        candidates.sort(
            key=lambda candidate: (
                (
                    0
                    if (
                        candidate["completion"].get("schema_version")
                        == EXTRACTION_CHECKPOINT_SCHEMA_VERSION
                        and candidate["completion"].get("input_fingerprint") == input_fingerprint
                    )
                    else 1
                ),
                str(candidate["source_dir"]),
            )
        )
        for candidate in candidates:
            completion = candidate["completion"]
            schema_version = completion.get("schema_version")
            if schema_version == EXTRACTION_CHECKPOINT_SCHEMA_VERSION:
                if completion.get("input_fingerprint") != input_fingerprint:
                    continue
            elif schema_version is not None:
                continue
            try:
                validated = _validate_extraction(
                    candidate["result"],
                    row_ids=row_ids,
                    definitions=definitions,
                )
            except (KeyError, TypeError, ValueError):
                continue
            source_dir = Path(candidate["source_dir"])
            legacy = schema_version is None
            relocated = source_dir != batch_dir
            if not legacy and not relocated and stale_audit is None:
                if isinstance(candidate.get("extraction_issues"), Mapping):
                    _write_json(
                        batch_dir / "extraction_issues.json",
                        candidate["extraction_issues"],
                    )
                return list(validated["rows"])

            batch_dir.mkdir(parents=True, exist_ok=True)
            _write_json(batch_dir / "row_ids.json", row_ids)
            _write_json(result_path, validated)
            source_audit = candidate.get("category_ontology_audit")
            if isinstance(source_audit, Mapping):
                _write_json(ontology_audit_path, source_audit)
            elif ontology_audit_path.is_file():
                try:
                    previous_audit = json.loads(ontology_audit_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    previous_audit = None
                _write_json(
                    ontology_audit_path,
                    {
                        "schema_version": "stage2_category_ontology_repair_v1",
                        "resolution": "superseded_by_single_patient_checkpoint_reindex",
                        "superseded_at": _now(),
                        "previous_audit": previous_audit,
                    },
                )
            source_issues = candidate.get("extraction_issues")
            if isinstance(source_issues, Mapping):
                _write_json(batch_dir / "extraction_issues.json", source_issues)
            _write_json(
                complete_path,
                {
                    "status": "complete",
                    "schema_version": EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
                    "input_fingerprint": input_fingerprint,
                    "completed_at": completion.get("completed_at") or _now(),
                    "rows": 1,
                    "adopted_legacy_single_patient_checkpoint": legacy,
                    "relocated_single_patient_checkpoint": relocated,
                    "checkpoint_source": str(source_dir),
                    "adopted_at": _now(),
                },
            )
            LOGGER.info(
                "%s single-patient Stage 2 extraction checkpoint: %s -> %s",
                "relocate" if relocated else "adopt legacy",
                source_dir,
                batch_dir,
            )
            return list(validated["rows"])

        if complete_path.is_file() or result_path.is_file():
            LOGGER.info("rerun incompatible Stage 2 extraction checkpoint: %s", batch_dir)
        if stale_audit is not None:
            LOGGER.info("retry stale Stage 2 category ontology batch: %s", batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        _write_json(batch_dir / "row_ids.json", row_ids)
        if len(definition_batches) == 1:
            messages = _extraction_prompt(
                definitions=definitions,
                rows=batch,
            )
            result = _request_validated_extraction(
                messages=messages,
                row_ids=row_ids,
                definitions=definitions,
                request_json=request_json,
                ontology_audit_path=ontology_audit_path,
            )
            if _prompt_chars(messages) > int(max_prompt_chars):  # pragma: no cover
                raise RuntimeError("Stage 2 extraction planner emitted an oversized batch")
        else:
            result = request_feature_batches(
                parent_dir=batch_dir,
                row=batch[0],
                parent_schema_version=EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
            )
        _write_json(result_path, result)
        _supersede_stale_category_ontology_audit(
            ontology_audit_path,
            previous=stale_audit,
        )
        _write_json(
            complete_path,
            {
                "status": "complete",
                "schema_version": EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
                "input_fingerprint": input_fingerprint,
                "completed_at": _now(),
                "rows": 1,
                "features": len(definitions),
                "feature_batches": len(definition_batches),
                "feature_batch_size": feature_batch_size,
            },
        )
        return list(result["rows"])

    def run_page(page: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        page_meta = dict(page["page"])
        row_id = int(page["row_id"])
        page_index = int(page_meta["page_index"])
        page_dir = output_dir / "pages" / f"row_{row_id:08d}" / f"page_{page_index:05d}"
        result_path = page_dir / "result.json"
        complete_path = page_dir / "complete.json"
        ontology_audit_path = page_dir / "category_ontology_repair.json"
        input_fingerprint = _value_fingerprint(
            {
                "schema_version": PAGE_EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
                "definitions": extraction_definitions,
                "row": dict(page),
            }
        )
        stale_audit = _stale_category_ontology_audit(ontology_audit_path)
        if complete_path.is_file() and result_path.is_file() and stale_audit is None:
            try:
                completion = json.loads(complete_path.read_text(encoding="utf-8"))
                stored = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    completion.get("schema_version") == PAGE_EXTRACTION_CHECKPOINT_SCHEMA_VERSION
                    and completion.get("input_fingerprint") == input_fingerprint
                ):
                    validated = _validate_extraction(
                        stored,
                        row_ids=[row_id],
                        definitions=definitions,
                    )
                    _ensure_extraction_issue_audit(page_dir)
                    return page_meta, dict(validated["rows"][0])
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            LOGGER.info("rerun incompatible Stage 2 extraction page: %s", page_dir)
        if stale_audit is not None:
            LOGGER.info("retry stale Stage 2 category ontology page: %s", page_dir)
        page_dir.mkdir(parents=True, exist_ok=True)
        _write_json(page_dir / "page.json", page_meta)
        if len(definition_batches) == 1:
            messages = _extraction_prompt(
                definitions=definitions,
                rows=[page],
            )
            if _prompt_chars(messages) > int(max_prompt_chars):  # pragma: no cover
                raise RuntimeError("Stage 2 extraction planner emitted an oversized page")
            result = _request_validated_extraction(
                messages=messages,
                row_ids=[row_id],
                definitions=definitions,
                request_json=request_json,
                ontology_audit_path=ontology_audit_path,
            )
        else:
            result = request_feature_batches(
                parent_dir=page_dir,
                row=page,
                parent_schema_version=PAGE_EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
            )
        _write_json(result_path, result)
        _supersede_stale_category_ontology_audit(
            ontology_audit_path,
            previous=stale_audit,
        )
        _write_json(
            complete_path,
            {
                "status": "complete",
                "schema_version": PAGE_EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
                "input_fingerprint": input_fingerprint,
                "completed_at": _now(),
                "features": len(definitions),
                "feature_batches": len(definition_batches),
                "feature_batch_size": feature_batch_size,
                **page_meta,
            },
        )
        return page_meta, dict(result["rows"][0])

    completed: list[tuple[int, list[dict[str, Any]]]] = []
    completed_pages: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    task_count = len(batches) + len(page_requests)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(int(workers), max(1, task_count)))
    ) as executor:
        batch_futures = {
            executor.submit(run_batch, index, batch): index
            for index, batch in enumerate(batches, start=1)
        }
        page_futures = {
            executor.submit(run_page, page): int(page["row_id"]) for page in page_requests
        }
        for future in concurrent.futures.as_completed([*batch_futures, *page_futures]):
            if future in batch_futures:
                completed.append((batch_futures[future], future.result()))
            else:
                row_id = page_futures[future]
                completed_pages.setdefault(row_id, []).append(future.result())
    values_by_row = {
        int(row["row_id"]): dict(row["values"])
        for _index, rows in sorted(completed)
        for row in rows
    }

    def reconcile_row(
        row_id: int,
        page_values: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    ) -> dict[str, Any]:
        reconciliation_dir = output_dir / "pages" / f"row_{row_id:08d}" / "reconciliation"
        result_path = reconciliation_dir / "result.json"
        complete_path = reconciliation_dir / "complete.json"
        ontology_audit_path = reconciliation_dir / "category_ontology_repair.json"
        stale_audit = _stale_category_ontology_audit(ontology_audit_path)
        ordered = sorted(page_values, key=lambda item: int(item[0]["page_index"]))
        page_results = [
            {**dict(meta), "values": dict(result["values"])} for meta, result in ordered
        ]
        reconciliation_fingerprint = _value_fingerprint(
            {
                "schema_version": PAGE_RECONCILIATION_CHECKPOINT_SCHEMA_VERSION,
                "row_id": int(row_id),
                "definitions": extraction_definitions,
                "page_results": page_results,
            }
        )
        if complete_path.is_file() and result_path.is_file() and stale_audit is None:
            try:
                completion = json.loads(complete_path.read_text(encoding="utf-8"))
                stored = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    completion.get("schema_version")
                    == PAGE_RECONCILIATION_CHECKPOINT_SCHEMA_VERSION
                    and completion.get("input_fingerprint") == reconciliation_fingerprint
                ):
                    validated = _validate_extraction(
                        stored,
                        row_ids=[row_id],
                        definitions=definitions,
                    )
                    _ensure_extraction_issue_audit(reconciliation_dir)
                    return dict(validated["rows"][0]["values"])
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            LOGGER.info(
                "rerun incompatible Stage 2 page reconciliation: %s",
                reconciliation_dir,
            )
        if stale_audit is not None:
            LOGGER.info(
                "retry stale Stage 2 category ontology reconciliation: %s",
                reconciliation_dir,
            )
        definition_batches = _partition_page_reconciliation_definitions(
            definitions=definitions,
            row_id=row_id,
            page_results=page_results,
            max_prompt_chars=int(max_prompt_chars),
            feature_batch_size=feature_batch_size,
        )
        if len(definition_batches) > 1:
            LOGGER.info(
                "Stage 2 lossless page reconciliation row_id=%s pages=%s features=%s "
                "feature_batches=%s",
                row_id,
                len(page_results),
                len(definitions),
                len(definition_batches),
            )

        merged_values: dict[str, Any] = {}
        for batch_index, batch_definitions in enumerate(definition_batches, start=1):
            batch_page_results = _page_results_for_definitions(
                page_results,
                batch_definitions,
            )
            messages = _page_reconciliation_prompt(
                definitions=batch_definitions,
                row_id=row_id,
                page_results=batch_page_results,
            )
            prompt_chars = _prompt_chars(messages)
            if prompt_chars > int(max_prompt_chars):  # pragma: no cover - planner invariant
                raise RuntimeError(
                    "Stage 2 feature-partitioned page reconciliation exceeded " "max_prompt_chars"
                )
            if len(definition_batches) == 1:
                batch_dir = reconciliation_dir
                batch_ontology_audit_path = ontology_audit_path
            else:
                batch_dir = reconciliation_dir / "feature_batches" / f"batch_{batch_index:05d}"
                batch_ontology_audit_path = batch_dir / "category_ontology_repair.json"
            batch_result_path = batch_dir / "result.json"
            batch_complete_path = batch_dir / "complete.json"
            batch_stale_audit = _stale_category_ontology_audit(batch_ontology_audit_path)
            batch_input = {
                "schema_version": PAGE_RECONCILIATION_CHECKPOINT_SCHEMA_VERSION,
                "row_id": int(row_id),
                "definitions": _prompt_feature_definitions(batch_definitions),
                "page_results": batch_page_results,
            }
            batch_fingerprint = _value_fingerprint(batch_input)
            batch_result: dict[str, Any] | None = None
            if (
                batch_complete_path.is_file()
                and batch_result_path.is_file()
                and batch_stale_audit is None
            ):
                try:
                    completion = json.loads(batch_complete_path.read_text(encoding="utf-8"))
                    cached = json.loads(batch_result_path.read_text(encoding="utf-8"))
                    if completion.get("input_fingerprint") == batch_fingerprint:
                        batch_result = _validate_extraction(
                            cached,
                            row_ids=[row_id],
                            definitions=batch_definitions,
                        )
                        _ensure_extraction_issue_audit(batch_dir)
                except (
                    KeyError,
                    OSError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    batch_result = None
            if batch_result is None:
                if batch_stale_audit is not None:
                    LOGGER.info(
                        "retry stale Stage 2 category ontology reconciliation batch: %s",
                        batch_dir,
                    )
                batch_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    batch_dir / "input.json",
                    {**batch_input, "input_fingerprint": batch_fingerprint},
                )
                batch_result = _request_validated_extraction(
                    messages=messages,
                    row_ids=[row_id],
                    definitions=batch_definitions,
                    request_json=request_json,
                    ontology_audit_path=batch_ontology_audit_path,
                )
                _write_json(batch_result_path, batch_result)
                _supersede_stale_category_ontology_audit(
                    batch_ontology_audit_path,
                    previous=batch_stale_audit,
                )
                _write_json(
                    batch_complete_path,
                    {
                        "status": "complete",
                        "schema_version": PAGE_RECONCILIATION_CHECKPOINT_SCHEMA_VERSION,
                        "input_fingerprint": batch_fingerprint,
                        "completed_at": _now(),
                        "pages": len(batch_page_results),
                        "features": len(batch_definitions),
                    },
                )
            merged_values.update(dict(batch_result["rows"][0]["values"]))

        result = _validate_extraction(
            {"rows": [{"row_id": int(row_id), "values": merged_values}]},
            row_ids=[row_id],
            definitions=definitions,
        )
        reconciliation_dir.mkdir(parents=True, exist_ok=True)
        _write_json(reconciliation_dir / "page_manifest.json", page_results)
        _write_json(result_path, result)
        _supersede_stale_category_ontology_audit(
            ontology_audit_path,
            previous=stale_audit,
        )
        _write_json(
            complete_path,
            {
                "status": "complete",
                "schema_version": PAGE_RECONCILIATION_CHECKPOINT_SCHEMA_VERSION,
                "input_fingerprint": reconciliation_fingerprint,
                "completed_at": _now(),
                "pages": len(page_results),
                "feature_batches": len(definition_batches),
                "feature_batch_size": feature_batch_size,
            },
        )
        return dict(result["rows"][0]["values"])

    if completed_pages:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(int(workers), len(completed_pages)))
        ) as executor:
            reconciliation_futures = {
                executor.submit(reconcile_row, row_id, page_values): row_id
                for row_id, page_values in completed_pages.items()
            }
            for future in concurrent.futures.as_completed(reconciliation_futures):
                row_id = reconciliation_futures[future]
                values_by_row[row_id] = future.result()
    records = []
    for row_id in row_ids:
        record: dict[str, Any] = {"_oci_row_id": int(row_id)}
        record.update({name: values_by_row[int(row_id)].get(name) for name in feature_names})
        records.append(record)
    frame = pd.DataFrame(records, columns=["_oci_row_id", *feature_names])
    _write_frame(output_dir / "extracted.csv", frame)
    failure_summary = _summarize_extraction_failures(
        output_dir=output_dir,
        definitions=definitions,
    )
    _write_json(
        output_dir / "complete.json",
        {
            "status": "complete",
            "completed_at": _now(),
            "rows": len(frame),
            "features": len(feature_names),
            "feature_batch_size": feature_batch_size,
            "feature_batches_per_patient": len(definition_batches),
            "batches": len(batches),
            "paged_rows": len(oversized_rows),
            "pages": len(page_requests),
            "feature_failure_patterns": len(failure_summary["feature_failure_patterns"]),
            "structural_failure_patients": failure_summary["structural_failure_patient_count"],
        },
    )
    return frame


def _feature_modeling_strategy(feature: Mapping[str, Any]) -> str:
    value_type = str(feature.get("value_type") or "ambiguous").strip().lower()
    if value_type != "continuous":
        return "categorical"
    strategy = str(feature.get("modeling_strategy") or "continuous").strip().lower()
    if strategy not in CONTINUOUS_MODELING_STRATEGIES:
        raise ValueError(
            f"continuous feature {feature.get('name')!r} has unsupported "
            f"modeling_strategy {strategy!r}"
        )
    return strategy


def _normalized_feature_modeling_definition(
    feature: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(feature)
    if str(normalized.get("value_type") or "ambiguous").strip().lower() == "continuous":
        normalized["modeling_strategy"] = _feature_modeling_strategy(normalized)
    else:
        normalized.pop("modeling_strategy", None)
    return normalized


def feature_summaries(
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    row_count = max(1, len(frame))
    for feature in definitions:
        name = str(feature["name"])
        series = frame[name] if name in frame else pd.Series([None] * len(frame))
        nonmissing = series.dropna()
        counts = nonmissing.astype(str).value_counts()
        dominant = float(counts.iloc[0] / len(nonmissing)) if len(nonmissing) else 1.0
        summary: dict[str, Any] = {
            "feature_id": str(feature["feature_id"]),
            "name": name,
            "rows": len(frame),
            "nonmissing": int(len(nonmissing)),
            "nonmissing_fraction": float(len(nonmissing) / row_count),
            "unique_nonmissing": int(nonmissing.nunique()),
            "dominant_value_fraction": dominant,
            "most_common_values": {str(key): int(count) for key, count in counts.head(8).items()},
        }
        if str(feature.get("value_type")) == "continuous":
            numeric_values = pd.to_numeric(nonmissing, errors="coerce")
            numeric = numeric_values.dropna()
            categorical = nonmissing.loc[numeric_values.isna()].astype(str)
            categorical_counts = categorical.value_counts()
            summary.update(
                {
                    "numeric_nonmissing": int(len(numeric)),
                    "numeric_nonmissing_fraction": float(len(numeric) / row_count),
                    "categorical_fallback_nonmissing": int(len(categorical)),
                    "categorical_fallback_nonmissing_fraction": float(
                        len(categorical) / row_count
                    ),
                    "categorical_fallback_values": {
                        str(key): int(count)
                        for key, count in categorical_counts.head(12).items()
                    },
                    "numeric_mean": float(numeric.mean()) if len(numeric) else None,
                    "numeric_sd": float(numeric.std(ddof=0)) if len(numeric) else None,
                    "recommended_modeling_strategy": (
                        "continuous_with_categorical_fallback"
                        if len(numeric) and len(categorical)
                        else "categorical"
                        if len(categorical)
                        else "continuous"
                    ),
                }
            )
        summaries.append(summary)
    return summaries


def _assert_extraction_health(
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    minimum_row_nonmissing_fraction: float,
    audit_path: Path,
) -> dict[str, Any]:
    """Reject final extraction matrices that are effectively all missing."""

    feature_names = [str(feature["name"]) for feature in definitions]
    if not feature_names:
        audit = {
            "schema_version": "stage2_final_extraction_health_v1",
            "status": "not_applicable",
            "scope": scope,
            "rows": int(len(frame)),
            "features": 0,
        }
        _write_json(audit_path, audit)
        return audit
    missing_columns = sorted(set(feature_names) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Stage 2 final {scope} extraction is missing feature columns: " f"{missing_columns}"
        )
    values = frame[feature_names]
    rows_with_any = values.notna().any(axis=1)
    row_fraction = float(rows_with_any.mean()) if len(values) else 0.0
    nonmissing_cells = int(values.notna().to_numpy(dtype=bool).sum())
    total_cells = int(values.shape[0] * values.shape[1])
    audit = {
        "schema_version": "stage2_final_extraction_health_v1",
        "status": ("ok" if row_fraction >= float(minimum_row_nonmissing_fraction) else "failed"),
        "scope": scope,
        "rows": int(len(values)),
        "features": int(len(feature_names)),
        "rows_with_any_nonmissing": int(rows_with_any.sum()),
        "all_null_rows": int((~rows_with_any).sum()),
        "row_nonmissing_fraction": row_fraction,
        "minimum_row_nonmissing_fraction": float(minimum_row_nonmissing_fraction),
        "nonmissing_cells": nonmissing_cells,
        "nonmissing_cell_fraction": (float(nonmissing_cells / total_cells) if total_cells else 0.0),
        "definitions_fingerprint": _value_fingerprint(list(definitions)),
    }
    _write_json(audit_path, audit)
    if audit["status"] != "ok":
        raise ValueError(
            f"Stage 2 final {scope} extraction is catastrophically sparse: "
            f"only {audit['rows_with_any_nonmissing']}/{audit['rows']} rows contain "
            "any retained feature value "
            f"({row_fraction:.3f} < {minimum_row_nonmissing_fraction:.3f})"
        )
    return audit


class _FeatureEncoder:
    def __init__(self, definitions: Sequence[Mapping[str, Any]]) -> None:
        self.definitions = list(definitions)
        self.encodings: list[tuple[str, str, Any]] = []

    def fit(self, frame: pd.DataFrame) -> "_FeatureEncoder":
        self.encodings = []
        for feature in self.definitions:
            name = str(feature["name"])
            modeling_strategy = _feature_modeling_strategy(feature)
            series = frame[name] if name in frame else pd.Series([None] * len(frame))
            if modeling_strategy in {
                "continuous",
                "continuous_with_categorical_fallback",
            }:
                numeric = pd.to_numeric(series, errors="coerce")
                median = float(numeric.median()) if numeric.notna().any() else 0.0
                scale = float(numeric.fillna(median).std(ddof=0))
                if modeling_strategy == "continuous":
                    parameters: Any = (median, scale or 1.0)
                else:
                    fallback_mask = series.notna() & numeric.isna()
                    fallback_categories = sorted(
                        str(item) for item in series.loc[fallback_mask].astype(str).unique()
                    )
                    parameters = (median, scale or 1.0, fallback_categories)
                self.encodings.append((name, modeling_strategy, parameters))
            else:
                closed_ontology = str(
                    feature.get("value_type") or "ambiguous"
                ) in {"binary", "categorical", "ordinal"}
                declared = _declared_categories(feature) if closed_ontology else []
                observed = [str(item) for item in series.dropna().astype(str).unique()]
                categories = list(dict.fromkeys([*declared, *sorted(observed), "__missing__"]))
                encoding = "categorical" if closed_ontology else "categorical_with_other"
                self.encodings.append((name, encoding, categories))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for name, value_type, parameters in self.encodings:
            series = frame[name] if name in frame else pd.Series([None] * len(frame))
            if value_type == "continuous":
                median, scale = parameters
                numeric = pd.to_numeric(series, errors="coerce")
                missing = numeric.isna().to_numpy(dtype=float)
                values = (numeric.fillna(median).to_numpy(dtype=float) - median) / scale
                columns.extend([values, missing])
            elif value_type == "continuous_with_categorical_fallback":
                median, scale, categories = parameters
                numeric = pd.to_numeric(series, errors="coerce")
                raw_missing = series.isna()
                categorical_mask = series.notna() & numeric.isna()
                normalized = series.where(categorical_mask, "").astype(str)
                values = (numeric.fillna(median).to_numpy(dtype=float) - median) / scale
                columns.extend(
                    [
                        values,
                        numeric.notna().to_numpy(dtype=float),
                        raw_missing.to_numpy(dtype=float),
                    ]
                )
                for category in categories:
                    columns.append(
                        (categorical_mask & (normalized == category)).to_numpy(dtype=float)
                    )
                columns.append(
                    (
                        categorical_mask
                        & ~normalized.isin(categories)
                    ).to_numpy(dtype=float)
                )
            else:
                normalized = series.where(series.notna(), "__missing__").astype(str)
                for category in parameters:
                    columns.append((normalized == category).to_numpy(dtype=float))
                if value_type == "categorical_with_other":
                    columns.append((~normalized.isin(parameters)).to_numpy(dtype=float))
        if not columns:
            return np.empty((len(frame), 0), dtype=float)
        return np.column_stack(columns).astype(float, copy=False)


def _definitions_for_roles(
    definitions: Sequence[Mapping[str, Any]], roles: set[str]
) -> list[Mapping[str, Any]]:
    return [
        feature
        for feature in definitions
        if set(map(str, feature.get("roles") or [])).intersection(roles)
    ]


class _ConstantClassifier:
    classes_ = np.asarray([0, 1], dtype=int)

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probability = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - probability, probability])


class _ConstantRegressor:
    def __init__(self, mean: float) -> None:
        self.mean = float(mean)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.mean, dtype=float)


def _fit_classifier(x: np.ndarray, y: np.ndarray, *, seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y)) < 2 or x.shape[1] == 0:
        return _ConstantClassifier(float(np.mean(y)))
    model = LogisticRegression(max_iter=2_000, C=1.0, random_state=seed)
    model.fit(x, y.astype(int))
    return model


def _predict_probability(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x), dtype=float)
    return probabilities[:, classes.index(1)].astype(float)


def _fit_regressor(x: np.ndarray, y: np.ndarray) -> Any:
    from sklearn.linear_model import Ridge

    if x.shape[1] == 0:
        return _ConstantRegressor(float(np.mean(y)))
    model = Ridge(alpha=1.0)
    model.fit(x, y)
    return model


@dataclass
class _OutcomeModels:
    control: Any
    treated: Any
    binary: bool


def _fit_outcome_models(
    x: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    binary: bool,
    seed: int,
) -> _OutcomeModels:
    models = []
    for arm in (0, 1):
        mask = treatment.astype(int) == arm
        if not mask.any():
            mask = np.ones(len(treatment), dtype=bool)
        if binary:
            models.append(_fit_classifier(x[mask], outcome[mask], seed=seed + arm))
        else:
            models.append(_fit_regressor(x[mask], outcome[mask]))
    return _OutcomeModels(control=models[0], treated=models[1], binary=binary)


def _predict_outcomes(models: _OutcomeModels, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if models.binary:
        return _predict_probability(models.control, x), _predict_probability(models.treated, x)
    return (
        np.asarray(models.control.predict(x), dtype=float),
        np.asarray(models.treated.predict(x), dtype=float),
    )


@dataclass
class _EffectModel:
    constant: float
    model: Any | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(x), self.constant, dtype=float)
        return np.asarray(self.model.predict(x), dtype=float)


def _fit_effect_model(
    x: np.ndarray,
    pseudo_outcome: np.ndarray,
    *,
    seed: int,
    trees: int | None,
) -> _EffectModel:
    finite = np.isfinite(pseudo_outcome)
    if not finite.any():
        return _EffectModel(constant=0.0)
    target = pseudo_outcome[finite]
    if len(target) >= 20:
        lower, upper = np.quantile(target, [0.01, 0.99])
        target = np.clip(target, lower, upper)
    constant = float(np.mean(target))
    if x.shape[1] == 0 or len(target) < 12:
        return _EffectModel(constant=constant)
    if trees is None:
        from sklearn.linear_model import Ridge

        model: Any = Ridge(alpha=2.0)
    else:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=int(trees),
            min_samples_leaf=max(2, min(20, len(target) // 10)),
            max_features="sqrt",
            n_jobs=1,
            random_state=seed,
        )
    model.fit(x[finite], target)
    return _EffectModel(constant=constant, model=model)


def _dr_score(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: np.ndarray,
    *,
    clip: float,
) -> np.ndarray:
    e = np.clip(propensity, clip, 1.0 - clip)
    score = (
        mu1
        - mu0
        + treatment * (outcome - mu1) / e
        - (1.0 - treatment) * (outcome - mu0) / (1.0 - e)
    )
    return np.where(np.isfinite(score), score, np.nan)


def _safe_auc(y: np.ndarray, probability: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, probability))


def _prediction_metrics(
    *,
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    observed_outcome: np.ndarray,
    binary_outcome: bool,
    r_loss: float,
) -> dict[str, Any]:
    from sklearn.metrics import log_loss, mean_squared_error

    e = np.clip(propensity, 1e-6, 1.0 - 1e-6)
    metrics: dict[str, Any] = {
        "treatment_log_loss": float(log_loss(treatment.astype(int), e, labels=[0, 1])),
        "treatment_brier": float(np.mean((treatment - e) ** 2)),
        "treatment_auc": _safe_auc(treatment, e),
        "r_loss": float(r_loss),
    }
    if binary_outcome:
        probability = np.clip(observed_outcome, 1e-6, 1.0 - 1e-6)
        metrics.update(
            {
                "outcome_log_loss": float(
                    log_loss(outcome.astype(int), probability, labels=[0, 1])
                ),
                "outcome_brier": float(np.mean((outcome - probability) ** 2)),
                "outcome_auc": _safe_auc(outcome, probability),
            }
        )
    else:
        rmse = float(math.sqrt(mean_squared_error(outcome, observed_outcome)))
        variance = float(np.var(outcome))
        metrics.update(
            {
                "outcome_rmse": rmse,
                "outcome_r2": (
                    float(1.0 - np.mean((outcome - observed_outcome) ** 2) / variance)
                    if variance > 0
                    else None
                ),
            }
        )
    return metrics


def _metric_improvements(
    baseline: Mapping[str, Any],
    enhanced: Mapping[str, Any],
) -> dict[str, Any]:
    """Orient every validation metric so positive values mean improvement."""

    improvements: dict[str, Any] = {}
    for key in sorted(set(baseline).intersection(enhanced)):
        if baseline[key] is None or enhanced[key] is None:
            improvements[key] = None
        elif key.endswith("auc") or key.endswith("r2"):
            improvements[key] = float(enhanced[key] - baseline[key])
        else:
            improvements[key] = float(baseline[key] - enhanced[key])
    return improvements


def _heldout_signal_diagnostic(
    performance: Mapping[str, Any],
    *,
    metric: str,
    minimum_positive_fold_fraction: float = 0.5,
) -> dict[str, Any]:
    fold_values = [
        row.get("improvement_positive_is_better", {}).get(metric)
        for row in performance.get("inner_fold_performance") or []
    ]
    finite = [
        float(value)
        for value in fold_values
        if value is not None and math.isfinite(value)
    ]
    aggregate = performance.get("improvement_positive_is_better", {}).get(metric)
    aggregate = (
        float(aggregate)
        if aggregate is not None and math.isfinite(float(aggregate))
        else None
    )
    positive_folds = sum(value > 0.0 for value in finite)
    positive_fraction = float(positive_folds / len(finite)) if finite else 0.0
    return {
        "metric": metric,
        "aggregate_improvement": aggregate,
        "fold_improvements": finite,
        "positive_folds": int(positive_folds),
        "evaluated_folds": int(len(finite)),
        "positive_fold_fraction": positive_fraction,
        "minimum_positive_fold_fraction": float(minimum_positive_fold_fraction),
        "supported": bool(
            aggregate is not None
            and aggregate > 0.0
            and positive_fraction >= float(minimum_positive_fold_fraction)
        ),
    }


def _feature_role_signal_diagnostics(
    feature: Mapping[str, Any],
    performance: Mapping[str, Any],
    *,
    binary_outcome: bool,
) -> dict[str, Any]:
    """Evaluate each claimed causal role using predictions on inner held-out rows."""

    outcome_metric = "outcome_brier" if binary_outcome else "outcome_rmse"
    treatment = _heldout_signal_diagnostic(performance, metric="treatment_brier")
    outcome = _heldout_signal_diagnostic(performance, metric=outcome_metric)
    effect = _heldout_signal_diagnostic(performance, metric="effect_model_r_loss")
    claimed_roles = list(map(str, feature.get("roles") or []))
    role_signals: dict[str, Any] = {}
    if "confounder" in claimed_roles:
        role_signals["confounder"] = {
            "supported": bool(treatment["supported"] and outcome["supported"]),
            "requires_treatment_and_outcome_signal": True,
            "treatment_signal": treatment,
            "outcome_signal": outcome,
        }
    if "prognostic" in claimed_roles:
        role_signals["prognostic"] = {
            "supported": bool(outcome["supported"]),
            "outcome_signal": outcome,
        }
    if "effect_modifier" in claimed_roles:
        role_signals["effect_modifier"] = {
            "supported": bool(effect["supported"]),
            "residual_effect_signal": effect,
        }
    return {
        "claimed_roles": claimed_roles,
        "role_signals": role_signals,
        "has_any_claimed_role_signal": any(
            bool(signal.get("supported")) for signal in role_signals.values()
        ),
    }


def _fallback_inner_splits(
    row_ids: Sequence[int], *, folds: int, seed: int
) -> list[dict[str, Any]]:
    from sklearn.model_selection import KFold

    row_ids = np.asarray(row_ids, dtype=int)
    count = min(max(2, int(folds)), len(row_ids))
    if count < 2:
        return []
    splitter = KFold(n_splits=count, shuffle=True, random_state=seed)
    return [
        {
            "inner_fold": index,
            "fit_row_ids": row_ids[fit].tolist(),
            "heldout_row_ids": row_ids[heldout].tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(row_ids), start=1)
    ]


def evaluate_definitions(
    *,
    dataset: pd.DataFrame,
    extracted: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    inner_folds: int,
    seed: int,
    propensity_clip: float,
    include_ablation: bool = True,
) -> dict[str, Any]:
    fit_ids = [int(value) for value in split["fit_row_ids"]]
    extraction_by_id = extracted.set_index("_oci_row_id", drop=False)
    supplied = list(split.get("inner_splits") or [])
    inner = supplied or _fallback_inner_splits(
        fit_ids,
        folds=inner_folds,
        seed=seed,
    )
    predictions: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "t",
            "y",
            "base_e",
            "feature_e",
            "base_y",
            "feature_y",
            "base_r_residual",
            "feature_null_effect_r_residual",
            "feature_r_residual",
        )
    }
    all_defs = list(definitions)
    propensity_defs = _definitions_for_roles(all_defs, {"confounder"})
    effect_defs = _definitions_for_roles(all_defs, {"effect_modifier"})
    binary = str(outcome_type) == "binary"
    fold_performance: list[dict[str, Any]] = []

    for fold_index, fold in enumerate(inner, start=1):
        train_ids = [int(value) for value in fold["fit_row_ids"] if int(value) in fit_ids]
        valid_ids = [int(value) for value in fold["heldout_row_ids"] if int(value) in fit_ids]
        if not train_ids or not valid_ids:
            continue
        train_features = extraction_by_id.loc[train_ids].reset_index(drop=True)
        valid_features = extraction_by_id.loc[valid_ids].reset_index(drop=True)
        train_data = dataset.iloc[train_ids]
        valid_data = dataset.iloc[valid_ids]
        t_train = train_data[treatment_column].to_numpy(dtype=float)
        y_train = train_data[outcome_column].to_numpy(dtype=float)
        t_valid = valid_data[treatment_column].to_numpy(dtype=float)
        y_valid = valid_data[outcome_column].to_numpy(dtype=float)

        base_x_train = np.empty((len(train_ids), 0), dtype=float)
        base_x_valid = np.empty((len(valid_ids), 0), dtype=float)
        base_t_model = _fit_classifier(base_x_train, t_train, seed=seed + fold_index)
        base_outcome = _fit_outcome_models(
            base_x_train, t_train, y_train, binary=binary, seed=seed + fold_index
        )
        base_e_train = _predict_probability(base_t_model, base_x_train)
        base_e_valid = _predict_probability(base_t_model, base_x_valid)
        base_mu0_train, base_mu1_train = _predict_outcomes(base_outcome, base_x_train)
        base_mu0_valid, base_mu1_valid = _predict_outcomes(base_outcome, base_x_valid)
        base_m_train = base_e_train * base_mu1_train + (1 - base_e_train) * base_mu0_train
        base_m_valid = base_e_valid * base_mu1_valid + (1 - base_e_valid) * base_mu0_valid
        base_pseudo = (y_train - base_m_train) / np.where(
            np.abs(t_train - base_e_train) < propensity_clip,
            np.where(t_train - base_e_train < 0, -propensity_clip, propensity_clip),
            t_train - base_e_train,
        )
        base_effect = _fit_effect_model(
            np.empty((len(train_ids), 0)), base_pseudo, seed=seed + fold_index, trees=None
        )
        base_tau = base_effect.predict(np.empty((len(valid_ids), 0)))

        t_encoder = _FeatureEncoder(propensity_defs).fit(train_features)
        x_t_train = t_encoder.transform(train_features)
        x_t_valid = t_encoder.transform(valid_features)
        y_encoder = _FeatureEncoder(all_defs).fit(train_features)
        x_y_train = y_encoder.transform(train_features)
        x_y_valid = y_encoder.transform(valid_features)
        effect_encoder = _FeatureEncoder(effect_defs).fit(train_features)
        x_effect_train = effect_encoder.transform(train_features)
        x_effect_valid = effect_encoder.transform(valid_features)
        feature_t_model = _fit_classifier(x_t_train, t_train, seed=seed + 100 + fold_index)
        feature_outcome = _fit_outcome_models(
            x_y_train, t_train, y_train, binary=binary, seed=seed + 100 + fold_index
        )
        feature_e_train = _predict_probability(feature_t_model, x_t_train)
        feature_e_valid = _predict_probability(feature_t_model, x_t_valid)
        feature_mu0_train, feature_mu1_train = _predict_outcomes(feature_outcome, x_y_train)
        feature_mu0_valid, feature_mu1_valid = _predict_outcomes(feature_outcome, x_y_valid)
        feature_m_train = (
            feature_e_train * feature_mu1_train + (1 - feature_e_train) * feature_mu0_train
        )
        feature_m_valid = (
            feature_e_valid * feature_mu1_valid + (1 - feature_e_valid) * feature_mu0_valid
        )
        feature_pseudo = (y_train - feature_m_train) / np.where(
            np.abs(t_train - feature_e_train) < propensity_clip,
            np.where(t_train - feature_e_train < 0, -propensity_clip, propensity_clip),
            t_train - feature_e_train,
        )
        feature_effect = _fit_effect_model(
            x_effect_train, feature_pseudo, seed=seed + 200 + fold_index, trees=None
        )
        feature_null_effect = _fit_effect_model(
            np.empty((len(train_ids), 0)),
            feature_pseudo,
            seed=seed + 200 + fold_index,
            trees=None,
        )
        feature_tau = feature_effect.predict(x_effect_valid)
        feature_null_tau = feature_null_effect.predict(np.empty((len(valid_ids), 0)))

        predictions["t"].append(t_valid)
        predictions["y"].append(y_valid)
        predictions["base_e"].append(base_e_valid)
        predictions["feature_e"].append(feature_e_valid)
        predictions["base_y"].append(np.where(t_valid == 1, base_mu1_valid, base_mu0_valid))
        predictions["feature_y"].append(
            np.where(t_valid == 1, feature_mu1_valid, feature_mu0_valid)
        )
        base_r_residual = (y_valid - base_m_valid) - (t_valid - base_e_valid) * base_tau
        feature_r_residual = (
            (y_valid - feature_m_valid) - (t_valid - feature_e_valid) * feature_tau
        )
        feature_null_effect_r_residual = (
            (y_valid - feature_m_valid)
            - (t_valid - feature_e_valid) * feature_null_tau
        )
        predictions["base_r_residual"].append(base_r_residual)
        predictions["feature_null_effect_r_residual"].append(
            feature_null_effect_r_residual
        )
        predictions["feature_r_residual"].append(feature_r_residual)
        fold_base = _prediction_metrics(
            treatment=t_valid,
            outcome=y_valid,
            propensity=base_e_valid,
            observed_outcome=np.where(t_valid == 1, base_mu1_valid, base_mu0_valid),
            binary_outcome=binary,
            r_loss=float(np.mean(base_r_residual**2)),
        )
        fold_enhanced = _prediction_metrics(
            treatment=t_valid,
            outcome=y_valid,
            propensity=feature_e_valid,
            observed_outcome=np.where(t_valid == 1, feature_mu1_valid, feature_mu0_valid),
            binary_outcome=binary,
            r_loss=float(np.mean(feature_r_residual**2)),
        )
        fold_base["effect_model_r_loss"] = float(
            np.mean(feature_null_effect_r_residual**2)
        )
        fold_enhanced["effect_model_r_loss"] = float(np.mean(feature_r_residual**2))
        fold_performance.append(
            {
                "inner_fold": int(fold.get("inner_fold") or fold_index),
                "evaluation_rows": int(len(valid_ids)),
                "baseline": fold_base,
                "with_extracted_features": fold_enhanced,
                "improvement_positive_is_better": _metric_improvements(
                    fold_base,
                    fold_enhanced,
                ),
            }
        )
    if not predictions["t"]:
        raise ValueError("Stage 2 empirical review has no usable inner validation folds")
    joined = {key: np.concatenate(value) for key, value in predictions.items()}
    base = _prediction_metrics(
        treatment=joined["t"],
        outcome=joined["y"],
        propensity=joined["base_e"],
        observed_outcome=joined["base_y"],
        binary_outcome=binary,
        r_loss=float(np.mean(joined["base_r_residual"] ** 2)),
    )
    enhanced = _prediction_metrics(
        treatment=joined["t"],
        outcome=joined["y"],
        propensity=joined["feature_e"],
        observed_outcome=joined["feature_y"],
        binary_outcome=binary,
        r_loss=float(np.mean(joined["feature_r_residual"] ** 2)),
    )
    base["effect_model_r_loss"] = float(
        np.mean(joined["feature_null_effect_r_residual"] ** 2)
    )
    enhanced["effect_model_r_loss"] = float(
        np.mean(joined["feature_r_residual"] ** 2)
    )
    improvements = _metric_improvements(base, enhanced)
    result: dict[str, Any] = {
        "evaluation_rows": int(len(joined["t"])),
        "inner_folds": int(len(fold_performance)),
        "inner_fold_performance": fold_performance,
        "baseline": base,
        "with_extracted_features": enhanced,
        "improvement_positive_is_better": improvements,
    }
    if include_ablation and definitions:
        ablations = []
        individual_signals = []
        for feature in definitions:
            singleton_result = evaluate_definitions(
                dataset=dataset,
                extracted=extracted,
                definitions=[feature],
                split=split,
                treatment_column=treatment_column,
                outcome_column=outcome_column,
                outcome_type=outcome_type,
                inner_folds=inner_folds,
                seed=seed,
                propensity_clip=propensity_clip,
                include_ablation=False,
            )
            individual_signals.append(
                {
                    "feature_id": str(feature["feature_id"]),
                    "name": str(feature["name"]),
                    **_feature_role_signal_diagnostics(
                        feature,
                        singleton_result,
                        binary_outcome=binary,
                    ),
                    "baseline": singleton_result["baseline"],
                    "with_feature": singleton_result["with_extracted_features"],
                    "improvement_positive_is_better": singleton_result[
                        "improvement_positive_is_better"
                    ],
                    "inner_fold_performance": singleton_result["inner_fold_performance"],
                }
            )
            without = [
                candidate
                for candidate in definitions
                if str(candidate["feature_id"]) != str(feature["feature_id"])
            ]
            without_result = evaluate_definitions(
                dataset=dataset,
                extracted=extracted,
                definitions=without,
                split=split,
                treatment_column=treatment_column,
                outcome_column=outcome_column,
                outcome_type=outcome_type,
                inner_folds=inner_folds,
                seed=seed,
                propensity_clip=propensity_clip,
                include_ablation=False,
            )
            without_metrics = without_result["with_extracted_features"]
            contribution = _metric_improvements(without_metrics, enhanced)
            ablations.append(
                {
                    "feature_id": str(feature["feature_id"]),
                    "name": str(feature["name"]),
                    "metrics_without_feature": without_metrics,
                    "feature_contribution_positive_is_better": contribution,
                }
            )
        result["individual_feature_signal"] = individual_signals
        result["leave_one_feature_out"] = ablations
    return result


def _review_prompt(
    *,
    clinical_question: str,
    definitions: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
    allow_measurement_revision: bool,
    min_nonmissing_fraction: float,
    max_dominant_fraction: float,
    feature_set_index: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    detailed_feature_ids = [str(feature["feature_id"]) for feature in definitions]
    configured_feature_ids = [
        str(feature["feature_id"])
        for feature in definitions
        if feature.get("configured_explicit_feature") is True
    ]
    body = {
        "job": "review_stage2_variables_against_training_fold_performance",
        "clinical_question": clinical_question,
        "information_boundary": (
            "All extraction summaries and performance metrics come only from the outer training fold. "
            "No outer-heldout outcomes are included."
        ),
        "allow_measurement_revision": allow_measurement_revision,
        "quality_guides": {
            "minimum_nonmissing_fraction": min_nonmissing_fraction,
            "maximum_dominant_value_fraction": max_dominant_fraction,
        },
        "review_scope": {
            "detailed_feature_ids": detailed_feature_ids,
            "configured_explicit_feature_ids": configured_feature_ids,
            "feature_count_in_entire_set": len(feature_set_index or definitions),
        },
        "rules": [
            "Give every detailed feature exactly one decision.",
            "Do not return decisions for index-only features; they are reviewed in other requests.",
            "Use the feature-set index to recognize related or potentially redundant variables.",
            "Keep a feature when extraction is usable and its scientific role remains plausible.",
            "Drop a feature when it is essentially unmeasured, invariant, or unsupported after extraction.",
            "The individual_feature_signal diagnostics come from models fit on each inner "
            "training split and scored only on that split's held-out rows. A deterministic "
            "gate will remove unsupported causal roles and will drop a feature with no "
            "supported role; do not claim signal that those diagnostics do not show.",
            "Use leave-one-feature-out metrics to distinguish a feature's contribution from overall model performance.",
            "For every retained continuous feature, inspect numeric_nonmissing and "
            "categorical_fallback_nonmissing in its extraction summary and choose a "
            "modeling_strategy. Choose continuous for usable numeric measurements, "
            "categorical when only stable categories are available, or "
            "continuous_with_categorical_fallback when both representations carry "
            "information. Never invent a numeric value for a category or threshold.",
            "Use revise only to clarify how the same evidence-supported measurement is extracted.",
            "For a revised binary variable, provide exactly two distinct scalar ontology values as separate categories_or_unit array items.",
            "For a revised categorical or ordinal variable, provide at least two distinct scalar ontology values as separate categories_or_unit array items.",
            "Do not add a new feature, change a causal role, or change supporting evidence.",
            "A feature listed in configured_explicit_feature_ids was required by the investigator with a supplied ontology; return keep and do not revise or drop it.",
            "Predictive performance is diagnostic evidence, not permission to use a post-treatment variable.",
            (
                "Measurement revision is permitted and will be evaluated in another round."
                if allow_measurement_revision
                else "This is the final review round; choose only keep or drop."
            ),
        ],
        "feature_set_index": list(feature_set_index or []),
        "features": list(definitions),
        "extraction_summaries": list(summaries),
        "inner_validation_performance": dict(performance),
        "response": {
            "feature_decisions": [
                {
                    "feature_id": "one supplied feature_id",
                    "action": "keep|drop|revise",
                    "reason": "scientific and empirical reason",
                    "modeling_strategy": (
                        "required for every kept or revised continuous feature: "
                        "continuous|categorical|continuous_with_categorical_fallback"
                    ),
                    "value_type": "required only for revise",
                    "categories_or_unit": ["required only for revise"],
                    "measurement_definition": "required only for revise",
                    "missing_value_rule": "required only for revise",
                }
            ],
            "overall_assessment": "brief assessment of the feature set",
        },
    }
    return [
        {
            "role": "system",
            "content": "You review prespecified variables using training-fold evidence only. Return JSON only.",
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _review_feature_set_index(
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain compact whole-set context in every partitioned review request."""

    return [
        {
            "feature_id": str(feature["feature_id"]),
            "name": str(feature["name"]),
            "description": str(feature.get("description") or ""),
            "roles": list(feature.get("roles") or []),
            "value_type": str(feature.get("value_type") or ""),
            "modeling_strategy": str(feature.get("modeling_strategy") or ""),
        }
        for feature in definitions
    ]


def _review_performance_for_features(
    performance: Mapping[str, Any],
    *,
    feature_ids: set[str],
) -> dict[str, Any]:
    """Return global metrics plus each detailed feature's own ablation metrics."""

    per_feature_keys = {"individual_feature_signal", "leave_one_feature_out"}
    selected = {
        key: copy.deepcopy(value)
        for key, value in performance.items()
        if key not in per_feature_keys
    }
    selected["individual_feature_signal"] = [
        copy.deepcopy(row)
        for row in performance.get("individual_feature_signal") or []
        if str(row.get("feature_id") or "") in feature_ids
    ]
    selected["leave_one_feature_out"] = [
        copy.deepcopy(row)
        for row in performance.get("leave_one_feature_out") or []
        if str(row.get("feature_id") or "") in feature_ids
    ]
    return selected


def _review_prompt_for_features(
    *,
    clinical_question: str,
    definitions: Sequence[Mapping[str, Any]],
    summaries_by_id: Mapping[str, Mapping[str, Any]],
    performance: Mapping[str, Any],
    feature_set_index: Sequence[Mapping[str, Any]],
    allow_measurement_revision: bool,
    min_nonmissing_fraction: float,
    max_dominant_fraction: float,
) -> list[dict[str, str]]:
    feature_ids = {str(feature["feature_id"]) for feature in definitions}
    return _review_prompt(
        clinical_question=clinical_question,
        definitions=definitions,
        summaries=[summaries_by_id[str(feature["feature_id"])] for feature in definitions],
        performance=_review_performance_for_features(
            performance,
            feature_ids=feature_ids,
        ),
        allow_measurement_revision=allow_measurement_revision,
        min_nonmissing_fraction=min_nonmissing_fraction,
        max_dominant_fraction=max_dominant_fraction,
        feature_set_index=feature_set_index,
    )


def _partition_review_features(
    *,
    clinical_question: str,
    definitions: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
    allow_measurement_revision: bool,
    min_nonmissing_fraction: float,
    max_dominant_fraction: float,
    max_prompt_chars: int,
) -> list[list[dict[str, Any]]]:
    """Partition detailed review inputs while preserving every per-feature diagnostic.

    The soft limit leaves space for validation-repair turns. A single unusually
    verbose feature may exceed that soft limit, but it must still fit the configured
    hard transport limit.
    """

    hard_limit = int(max_prompt_chars)
    if hard_limit < 1:
        raise ValueError("Stage 2 max_prompt_chars must be positive")
    soft_limit = min(hard_limit, max(20_000, int(hard_limit * 0.6)))
    summaries_by_id = {str(row["feature_id"]): row for row in summaries}
    expected_ids = {str(feature["feature_id"]) for feature in definitions}
    missing_summaries = sorted(expected_ids - set(summaries_by_id))
    if missing_summaries:
        raise ValueError(
            "Stage 2 review is missing extraction summaries for feature ID(s): "
            f"{missing_summaries[:8]}"
        )
    feature_set_index = _review_feature_set_index(definitions)
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for raw_feature in definitions:
        feature = dict(raw_feature)
        proposed = [*current, feature]
        messages = _review_prompt_for_features(
            clinical_question=clinical_question,
            definitions=proposed,
            summaries_by_id=summaries_by_id,
            performance=performance,
            feature_set_index=feature_set_index,
            allow_measurement_revision=allow_measurement_revision,
            min_nonmissing_fraction=min_nonmissing_fraction,
            max_dominant_fraction=max_dominant_fraction,
        )
        prompt_chars = _prompt_chars(messages)
        if current and prompt_chars > soft_limit:
            batches.append(current)
            current = [feature]
            singleton_messages = _review_prompt_for_features(
                clinical_question=clinical_question,
                definitions=current,
                summaries_by_id=summaries_by_id,
                performance=performance,
                feature_set_index=feature_set_index,
                allow_measurement_revision=allow_measurement_revision,
                min_nonmissing_fraction=min_nonmissing_fraction,
                max_dominant_fraction=max_dominant_fraction,
            )
            prompt_chars = _prompt_chars(singleton_messages)
        else:
            current = proposed
        if prompt_chars > hard_limit:
            raise ValueError(
                "Stage 2 cannot review one complete feature within max_prompt_chars "
                f"({prompt_chars} > {hard_limit}); shorten that feature's metadata or "
                "increase the prompt budget"
            )
    if current:
        batches.append(current)
    return batches


def _validate_review(
    value: Mapping[str, Any],
    *,
    definitions: Sequence[Mapping[str, Any]],
    allow_measurement_revision: bool,
    summaries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    decisions = value.get("feature_decisions")
    if not isinstance(decisions, list):
        raise ValueError("review response requires feature_decisions")
    by_id = {str(feature["feature_id"]): dict(feature) for feature in definitions}
    clean: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("each feature decision must be an object")
        feature_id = str(decision.get("feature_id") or "")
        if feature_id not in by_id or feature_id in clean:
            raise ValueError("review named an unknown or duplicate feature_id")
        action = str(decision.get("action") or "")
        if action not in {"keep", "drop", "revise"}:
            raise ValueError("review action must be keep, drop, or revise")
        if by_id[feature_id].get("configured_explicit_feature") is True and action != "keep":
            raise ValueError(
                f"investigator-configured feature {feature_id!r} must be kept without revision"
            )
        if action == "revise" and not allow_measurement_revision:
            raise ValueError("measurement revision is not permitted in the final review round")
        row: dict[str, Any] = {
            "feature_id": feature_id,
            "action": action,
            "reason": str(decision.get("reason") or ""),
        }
        if action == "revise":
            value_type = str(decision.get("value_type") or "")
            categories = decision.get("categories_or_unit")
            if value_type not in {"binary", "categorical", "continuous", "ordinal"}:
                raise ValueError("a revised variable requires an operational value_type")
            if not isinstance(categories, list) or not categories:
                raise ValueError("a revised variable requires categories_or_unit")
            if value_type in {"binary", "categorical", "ordinal"}:
                categories = _validated_closed_category_values(
                    value_type=value_type,
                    values=categories,
                    source=f"revised feature {feature_id!r}",
                )
            for key in ("measurement_definition", "missing_value_rule"):
                if not str(decision.get(key) or "").strip():
                    raise ValueError(f"a revised variable requires {key}")
            row.update(
                {
                    "value_type": value_type,
                    "categories_or_unit": [str(item) for item in categories],
                    "measurement_definition": str(decision["measurement_definition"]),
                    "missing_value_rule": str(decision["missing_value_rule"]),
                }
            )
        resulting_value_type = str(
            row.get("value_type") or by_id[feature_id].get("value_type") or "ambiguous"
        )
        if action != "drop" and resulting_value_type == "continuous":
            modeling_strategy = str(decision.get("modeling_strategy") or "").strip().lower()
            if not modeling_strategy and summaries is None:
                modeling_strategy = _feature_modeling_strategy(by_id[feature_id])
            if modeling_strategy not in CONTINUOUS_MODELING_STRATEGIES:
                raise ValueError(
                    "a retained continuous variable requires modeling_strategy to be "
                    "continuous, categorical, or continuous_with_categorical_fallback"
                )
            row["modeling_strategy"] = modeling_strategy
        clean[feature_id] = row
    if set(clean) != set(by_id):
        raise ValueError("review must decide every supplied feature")
    return {
        "feature_decisions": [clean[feature_id] for feature_id in by_id],
        "overall_assessment": str(value.get("overall_assessment") or ""),
    }


def _request_partitioned_review(
    *,
    clinical_question: str,
    definitions: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
    allow_measurement_revision: bool,
    min_nonmissing_fraction: float,
    max_dominant_fraction: float,
    max_prompt_chars: int,
    output_dir: Path,
    request_json: RequestJSON,
) -> dict[str, Any]:
    """Review every feature in resumable prompt-sized groups."""

    batches = _partition_review_features(
        clinical_question=clinical_question,
        definitions=definitions,
        summaries=summaries,
        performance=performance,
        allow_measurement_revision=allow_measurement_revision,
        min_nonmissing_fraction=min_nonmissing_fraction,
        max_dominant_fraction=max_dominant_fraction,
        max_prompt_chars=max_prompt_chars,
    )
    summaries_by_id = {str(row["feature_id"]): row for row in summaries}
    feature_set_index = _review_feature_set_index(definitions)
    prompt_sizes: list[int] = []
    decisions: list[dict[str, Any]] = []
    assessments: list[str] = []
    for batch_index, batch_definitions in enumerate(batches, start=1):
        messages = _review_prompt_for_features(
            clinical_question=clinical_question,
            definitions=batch_definitions,
            summaries_by_id=summaries_by_id,
            performance=performance,
            feature_set_index=feature_set_index,
            allow_measurement_revision=allow_measurement_revision,
            min_nonmissing_fraction=min_nonmissing_fraction,
            max_dominant_fraction=max_dominant_fraction,
        )
        prompt_chars = _prompt_chars(messages)
        prompt_sizes.append(prompt_chars)
        if prompt_chars > int(max_prompt_chars):  # pragma: no cover - planner invariant
            raise RuntimeError("Stage 2 review partition exceeded max_prompt_chars")
        batch_dir = output_dir / f"batch_{batch_index:05d}"
        input_value = {
            "schema_version": REVIEW_CHECKPOINT_SCHEMA_VERSION,
            "clinical_question": clinical_question,
            "allow_measurement_revision": allow_measurement_revision,
            "feature_set_index": feature_set_index,
            "definitions": list(batch_definitions),
            "summaries": [
                summaries_by_id[str(feature["feature_id"])] for feature in batch_definitions
            ],
            "performance": _review_performance_for_features(
                performance,
                feature_ids={str(feature["feature_id"]) for feature in batch_definitions},
            ),
            "quality_guides": {
                "minimum_nonmissing_fraction": min_nonmissing_fraction,
                "maximum_dominant_value_fraction": max_dominant_fraction,
            },
        }
        input_fingerprint = _value_fingerprint(input_value)
        result_path = batch_dir / "result.json"
        complete_path = batch_dir / "complete.json"
        batch_review: dict[str, Any] | None = None
        if result_path.is_file() and complete_path.is_file():
            try:
                completion = json.loads(complete_path.read_text(encoding="utf-8"))
                cached = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    completion.get("schema_version") == REVIEW_CHECKPOINT_SCHEMA_VERSION
                    and completion.get("input_fingerprint") == input_fingerprint
                ):
                    batch_review = _validate_review(
                        cached,
                        definitions=batch_definitions,
                        allow_measurement_revision=allow_measurement_revision,
                        summaries=[
                            summaries_by_id[str(feature["feature_id"])]
                            for feature in batch_definitions
                        ],
                    )
            except (
                KeyError,
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                batch_review = None
        if batch_review is None:
            batch_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                batch_dir / "input.json",
                {**input_value, "input_fingerprint": input_fingerprint},
            )
            batch_review = request_json(
                messages,
                lambda value, batch_definitions=batch_definitions: _validate_review(
                    value,
                    definitions=batch_definitions,
                    allow_measurement_revision=allow_measurement_revision,
                    summaries=[
                        summaries_by_id[str(feature["feature_id"])]
                        for feature in batch_definitions
                    ],
                ),
            )
            _write_json(result_path, batch_review)
            _write_json(
                complete_path,
                {
                    "status": "complete",
                    "schema_version": REVIEW_CHECKPOINT_SCHEMA_VERSION,
                    "input_fingerprint": input_fingerprint,
                    "completed_at": _now(),
                    "batch_index": batch_index,
                    "batches": len(batches),
                    "features": len(batch_definitions),
                    "prompt_chars": prompt_chars,
                },
            )
        decisions.extend(batch_review["feature_decisions"])
        assessment = str(batch_review.get("overall_assessment") or "").strip()
        if assessment:
            assessments.append(f"Review group {batch_index}/{len(batches)}: {assessment}")
    LOGGER.info(
        "Stage 2 feature review groups=%s features=%s prompt_chars=%s",
        len(batches),
        len(definitions),
        prompt_sizes,
    )
    return _validate_review(
        {
            "feature_decisions": decisions,
            "overall_assessment": "\n".join(assessments),
        },
        definitions=definitions,
        allow_measurement_revision=allow_measurement_revision,
        summaries=summaries,
    )


def _apply_review(
    definitions: Sequence[Mapping[str, Any]],
    review: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    decisions = {str(row["feature_id"]): row for row in review["feature_decisions"]}
    revised: list[dict[str, Any]] = []
    measurement_changed = False
    for feature in definitions:
        decision = decisions[str(feature["feature_id"])]
        if decision["action"] == "drop":
            continue
        updated = dict(feature)
        if decision["action"] == "revise":
            measurement_changed = True
            for key in (
                "value_type",
                "categories_or_unit",
                "measurement_definition",
                "missing_value_rule",
            ):
                updated[key] = decision[key]
        if "modeling_strategy" in decision:
            updated["modeling_strategy"] = decision["modeling_strategy"]
        revised.append(_normalized_feature_modeling_definition(updated))
    return revised, measurement_changed


def _changed_feature_representation_ids(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> set[str]:
    """Identify retained features that need evaluation under a changed representation."""

    before_by_id = {str(feature["feature_id"]): feature for feature in before}
    fields = (
        "value_type",
        "categories_or_unit",
        "measurement_definition",
        "missing_value_rule",
        "modeling_strategy",
    )
    changed: set[str] = set()
    for feature in after:
        feature_id = str(feature["feature_id"])
        previous = before_by_id.get(feature_id)
        if previous is None:
            changed.add(feature_id)
            continue
        prior_view = {key: copy.deepcopy(previous.get(key)) for key in fields}
        current_view = {key: copy.deepcopy(feature.get(key)) for key in fields}
        if _value_fingerprint(prior_view) != _value_fingerprint(current_view):
            changed.add(feature_id)
    return changed


def _apply_empirical_signal_pruning(
    definitions: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
    *,
    defer_feature_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove unsupported causal roles using inner-held-out singleton models.

    A dual-role feature may remain under the role that validates even when its
    other proposed role does not. Investigator-configured features are audited
    but remain immutable.
    """

    deferred = set(defer_feature_ids or set())
    signal_by_id = {
        str(row["feature_id"]): row
        for row in performance.get("individual_feature_signal") or []
    }
    retained: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for raw_feature in definitions:
        feature = dict(raw_feature)
        feature_id = str(feature["feature_id"])
        roles = list(map(str, feature.get("roles") or []))
        signal = signal_by_id.get(feature_id)
        if signal is None:
            raise ValueError(
                f"Stage 2 signal pruning is missing inner-held-out diagnostics for {feature_id!r}"
            )
        role_signals = signal.get("role_signals") or {}
        supported_roles = [
            role
            for role in roles
            if bool((role_signals.get(role) or {}).get("supported"))
        ]
        if feature.get("configured_explicit_feature") is True:
            retained.append(feature)
            action = "keep_configured"
            retained_roles = roles
        elif feature_id in deferred:
            retained.append(feature)
            action = "defer_until_re_evaluated"
            retained_roles = roles
        elif supported_roles:
            feature["roles"] = supported_roles
            retained.append(feature)
            retained_roles = supported_roles
            action = "keep" if supported_roles == roles else "prune_unsupported_roles"
        else:
            retained_roles = []
            action = "drop_no_heldout_role_signal"
        decisions.append(
            {
                "feature_id": feature_id,
                "name": str(feature["name"]),
                "action": action,
                "claimed_roles": roles,
                "retained_roles": retained_roles,
                "role_signals": copy.deepcopy(role_signals),
            }
        )
    report = {
        "schema_version": "stage2_inner_heldout_signal_pruning_v1",
        "features_evaluated": len(definitions),
        "features_retained": len(retained),
        "features_dropped": sum(
            decision["action"] == "drop_no_heldout_role_signal" for decision in decisions
        ),
        "features_with_roles_pruned": sum(
            decision["action"] == "prune_unsupported_roles" for decision in decisions
        ),
        "features_deferred_for_re_evaluation": sum(
            decision["action"] == "defer_until_re_evaluated" for decision in decisions
        ),
        "decisions": decisions,
    }
    return retained, report


def _ontology_refinement_prompt(
    *,
    feature: Mapping[str, Any],
    failure_patterns: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Request a same-feature ontology repair from repeated training failures."""

    body = {
        "job": "refine_stage2_feature_ontology_from_repeated_extraction_failures",
        "information_boundary": (
            "These aggregate diagnostics come only from outer-training patients. "
            "No held-out patient text, treatment, or outcome is supplied."
        ),
        "feature": {
            key: copy.deepcopy(feature.get(key))
            for key in (
                "feature_id",
                "name",
                "description",
                "value_type",
                "categories_or_unit",
                "measurement_definition",
                "missing_value_rule",
                "roles",
            )
        },
        "repeated_failure_patterns": [
            {
                key: copy.deepcopy(pattern.get(key))
                for key in (
                    "failure_kind",
                    "reason",
                    "patient_count",
                    "example_values",
                    "allowed_categories",
                )
            }
            for pattern in failure_patterns
        ],
        "rules": [
            "Refine only the supplied feature's extraction ontology; do not rename, merge, split, add, or drop a feature and do not change its causal roles.",
            "The example values are prior model outputs that failed validation, not verified patient facts.",
            "Use revise only when the repeated failures identify a correctable mismatch in value type, closed categories or unit, measurement definition, or missing-value rule.",
            "Use keep when the current ontology is already appropriate and the failures do not justify a change.",
            "A revised ontology must still define exactly one reusable patient-level scalar measurement.",
            "Prefer a numeric continuous ontology when the named measurement is realistically extractable as one number; include one unit when applicable.",
            "For binary variables return exactly two distinct extractable scalar categories; for categorical or ordinal variables return at least two.",
            "Do not blindly add every failed output as a category; choose a stable, reproducible ontology and clarify how source documentation maps to it.",
            "Return JSON only.",
        ],
        "response": {
            "action": "keep|revise",
            "reason": "why the ontology is retained or changed",
            "description": "required for revise",
            "value_type": "binary|categorical|continuous|ordinal; required for revise",
            "categories_or_unit": ["required for revise; empty only for unitless continuous"],
            "measurement_definition": "required for revise",
            "missing_value_rule": "required for revise",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You refine one clinical extraction ontology from repeated validation "
                "failures on training patients. Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True, ensure_ascii=False)},
    ]


def _validate_ontology_refinement(
    value: Mapping[str, Any],
    *,
    feature: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a bounded same-feature ontology decision."""

    if feature.get("configured_explicit_feature") is True:
        raise ValueError("investigator-configured feature ontology cannot be revised")
    action = str(value.get("action") or "").strip().lower()
    if action not in {"keep", "revise"}:
        raise ValueError("ontology refinement action must be keep or revise")
    decision: dict[str, Any] = {
        "feature_id": str(feature["feature_id"]),
        "feature_name": str(feature["name"]),
        "action": action,
        "reason": str(value.get("reason") or "").strip(),
    }
    if action == "keep":
        return decision

    description = str(value.get("description") or "").strip()
    value_type = str(value.get("value_type") or "").strip().lower()
    categories = value.get("categories_or_unit")
    measurement_definition = str(value.get("measurement_definition") or "").strip()
    missing_value_rule = str(value.get("missing_value_rule") or "").strip()
    if not description:
        raise ValueError("a revised ontology requires description")
    if value_type not in {"binary", "categorical", "continuous", "ordinal"}:
        raise ValueError("a revised ontology requires an operational value_type")
    if not isinstance(categories, list):
        raise ValueError("a revised ontology requires a categories_or_unit array")
    clean_categories = [str(item).strip() for item in categories if str(item).strip()]
    if value_type in {"binary", "categorical", "ordinal"}:
        clean_categories = _validated_closed_category_values(
            value_type=value_type,
            values=clean_categories,
            source=f"refined feature {feature['feature_id']!r}",
        )
    elif len(clean_categories) > 1:
        raise ValueError("a revised continuous ontology may name at most one unit")
    if not measurement_definition:
        raise ValueError("a revised ontology requires measurement_definition")
    if not missing_value_rule:
        raise ValueError("a revised ontology requires missing_value_rule")
    decision.update(
        {
            "description": description,
            "value_type": value_type,
            "categories_or_unit": clean_categories,
            "measurement_definition": measurement_definition,
            "missing_value_rule": missing_value_rule,
        }
    )
    return decision


def _repeated_ontology_failure_patterns(
    summary: Mapping[str, Any],
    *,
    minimum_patients: int,
) -> dict[str, list[dict[str, Any]]]:
    """Select feature-specific patterns repeated across enough distinct patients."""

    selected: dict[str, list[dict[str, Any]]] = {}
    for raw_pattern in summary.get("feature_failure_patterns") or []:
        if not isinstance(raw_pattern, Mapping):
            continue
        if int(raw_pattern.get("patient_count") or 0) < int(minimum_patients):
            continue
        feature_name = str(raw_pattern.get("feature_name") or "")
        if feature_name:
            selected.setdefault(feature_name, []).append(dict(raw_pattern))
    return selected


def _request_ontology_refinements(
    *,
    definitions: Sequence[Mapping[str, Any]],
    repeated_patterns: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    request_json: RequestJSON,
    workers: int,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    """Checkpoint independent ontology decisions and preserve explicit definitions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    features_by_name = {str(feature["name"]): dict(feature) for feature in definitions}
    unknown_names = sorted(set(repeated_patterns) - set(features_by_name))
    if unknown_names:
        raise ValueError(
            "ontology refinement received failure patterns for unknown feature names: "
            f"{unknown_names}"
        )
    input_value = {
        "schema_version": ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION,
        "definitions": list(definitions),
        "repeated_failure_patterns": {
            name: list(patterns) for name, patterns in repeated_patterns.items()
        },
    }
    input_fingerprint = _value_fingerprint(input_value)
    _write_json(
        output_dir / "input.json",
        {**input_value, "input_fingerprint": input_fingerprint},
    )

    decisions_by_name: dict[str, dict[str, Any]] = {}
    jobs: list[tuple[int, str, dict[str, Any], list[dict[str, str]]]] = []
    for index, feature in enumerate(definitions, start=1):
        name = str(feature["name"])
        if name not in repeated_patterns:
            continue
        if feature.get("configured_explicit_feature") is True:
            decisions_by_name[name] = {
                "feature_id": str(feature["feature_id"]),
                "feature_name": name,
                "action": "keep",
                "reason": (
                    "Repeated failures were recorded, but the investigator-configured "
                    "ontology is immutable."
                ),
                "configured_explicit_feature": True,
            }
            continue
        jobs.append(
            (
                index,
                name,
                dict(feature),
                _ontology_refinement_prompt(
                    feature=feature,
                    failure_patterns=repeated_patterns[name],
                ),
            )
        )

    def request_one(
        job: tuple[int, str, dict[str, Any], list[dict[str, str]]],
    ) -> tuple[str, dict[str, Any]]:
        index, name, feature, messages = job
        feature_dir = output_dir / f"feature_{index:03d}"
        feature_input = {
            "schema_version": ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION,
            "feature": feature,
            "failure_patterns": list(repeated_patterns[name]),
        }
        feature_fingerprint = _value_fingerprint(feature_input)
        result_path = feature_dir / "result.json"
        complete_path = feature_dir / "complete.json"
        if result_path.is_file() and complete_path.is_file():
            try:
                completion = json.loads(complete_path.read_text(encoding="utf-8"))
                cached = json.loads(result_path.read_text(encoding="utf-8"))
                if completion.get("input_fingerprint") == feature_fingerprint:
                    return name, _validate_ontology_refinement(cached, feature=feature)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        feature_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            feature_dir / "input.json",
            {**feature_input, "input_fingerprint": feature_fingerprint},
        )
        try:
            decision = request_json(
                messages,
                lambda value: _validate_ontology_refinement(value, feature=feature),
            )
        except ValueError as exc:
            decision = {
                "feature_id": str(feature["feature_id"]),
                "feature_name": name,
                "action": "keep",
                "reason": (
                    "Ontology refinement response remained invalid; retained the prior "
                    f"ontology: {type(exc).__name__}: {exc}"
                ),
                "validation_fallback": True,
            }
            _write_json(
                feature_dir / "fallback.json",
                {
                    "status": "conservative_keep",
                    "completed_at": _now(),
                    "validation_error": f"{type(exc).__name__}: {exc}",
                },
            )
        _write_json(result_path, decision)
        _write_json(
            complete_path,
            {
                "status": "complete",
                "schema_version": ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION,
                "input_fingerprint": feature_fingerprint,
                "completed_at": _now(),
                "action": decision["action"],
            },
        )
        return name, decision

    if jobs:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(int(workers), len(jobs)))
        ) as executor:
            futures = {executor.submit(request_one, job): job[1] for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                name, decision = future.result()
                decisions_by_name[name] = decision

    updated: list[dict[str, Any]] = []
    changed_names: list[str] = []
    for raw_feature in definitions:
        feature = dict(raw_feature)
        name = str(feature["name"])
        decision = decisions_by_name.get(name)
        if decision is not None and decision["action"] == "revise":
            before = {
                key: copy.deepcopy(feature.get(key))
                for key in (
                    "description",
                    "value_type",
                    "categories_or_unit",
                    "measurement_definition",
                    "missing_value_rule",
                )
            }
            for key in (
                "description",
                "value_type",
                "categories_or_unit",
                "measurement_definition",
                "missing_value_rule",
            ):
                feature[key] = copy.deepcopy(decision[key])
            after = {key: copy.deepcopy(feature.get(key)) for key in before}
            if _value_fingerprint(before) != _value_fingerprint(after):
                changed_names.append(name)
        updated.append(feature)

    ordered_decisions = [
        decisions_by_name[str(feature["name"])]
        for feature in definitions
        if str(feature["name"]) in decisions_by_name
    ]
    report = {
        "schema_version": ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION,
        "input_fingerprint": input_fingerprint,
        "completed_at": _now(),
        "triggered_features": len(repeated_patterns),
        "model_requested_features": len(jobs),
        "immutable_explicit_features": sum(
            decision.get("configured_explicit_feature") is True for decision in ordered_decisions
        ),
        "changed_feature_names": changed_names,
        "decisions": ordered_decisions,
    }
    _write_json(output_dir / "result.json", {"definitions": updated, **report})
    _write_json(output_dir / "complete.json", {"status": "complete", **report})
    return updated, bool(changed_names), report


def _extract_training_with_ontology_feedback(
    *,
    dataset: pd.DataFrame,
    row_ids: Sequence[int],
    text_column: str,
    definitions: Sequence[Mapping[str, Any]],
    output_dir: Path,
    feedback_dir: Path,
    request_json: RequestJSON,
    workers: int,
    max_prompt_chars: int,
    feature_batch_size: int,
    minimum_failure_patients: int,
    max_refinement_rounds: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    """Extract training rows and re-extract after repeated-failure ontology repairs."""

    current = [dict(feature) for feature in definitions]
    extraction_dir = output_dir
    rounds: list[dict[str, Any]] = []
    stopped_reason = "maximum_refinement_rounds_reached"
    extracted: pd.DataFrame | None = None
    for pass_index in range(0, int(max_refinement_rounds) + 1):
        extracted = extract_rows(
            dataset=dataset,
            row_ids=row_ids,
            text_column=text_column,
            definitions=current,
            output_dir=extraction_dir,
            request_json=request_json,
            workers=workers,
            max_prompt_chars=max_prompt_chars,
            feature_batch_size=feature_batch_size,
        )
        summary = json.loads((extraction_dir / "failure_summary.json").read_text(encoding="utf-8"))
        repeated = _repeated_ontology_failure_patterns(
            summary,
            minimum_patients=minimum_failure_patients,
        )
        if not repeated:
            stopped_reason = "no_repeated_feature_failures"
            break
        if pass_index >= int(max_refinement_rounds):
            break
        round_number = pass_index + 1
        round_dir = feedback_dir / f"round_{round_number:03d}"
        updated, changed, report = _request_ontology_refinements(
            definitions=current,
            repeated_patterns=repeated,
            output_dir=round_dir,
            request_json=request_json,
            workers=workers,
        )
        rounds.append(
            {
                "round": round_number,
                "source_extraction_dir": str(extraction_dir),
                "repeated_failure_features": sorted(repeated),
                "changed_feature_names": list(report["changed_feature_names"]),
                "immutable_explicit_features": int(report["immutable_explicit_features"]),
            }
        )
        if not changed:
            stopped_reason = "no_ontology_changes"
            break
        current = updated
        extraction_dir = round_dir / "extraction"

    if extracted is None:  # pragma: no cover - loop always runs once
        raise RuntimeError("ontology feedback loop did not perform extraction")
    feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_result = {
        "schema_version": ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION,
        "completed_at": _now(),
        "minimum_failure_patients": int(minimum_failure_patients),
        "maximum_refinement_rounds": int(max_refinement_rounds),
        "rounds_executed": len(rounds),
        "stopped_reason": stopped_reason,
        "rounds": rounds,
        "definitions": current,
    }
    _write_json(feedback_dir / "result.json", feedback_result)
    _write_json(feedback_dir / "complete.json", {"status": "complete", **feedback_result})
    return extracted, current, len(rounds)


def _cross_fitted_nuisance(
    *,
    dataset: pd.DataFrame,
    extracted: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    fit_ids: Sequence[int],
    inner_splits: Sequence[Mapping[str, Any]],
    treatment_column: str,
    outcome_column: str,
    binary: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_ids = [int(value) for value in fit_ids]
    position = {row_id: index for index, row_id in enumerate(fit_ids)}
    e = np.full(len(fit_ids), np.nan)
    mu0 = np.full(len(fit_ids), np.nan)
    mu1 = np.full(len(fit_ids), np.nan)
    extracted_by_id = extracted.set_index("_oci_row_id", drop=False)
    propensity_defs = _definitions_for_roles(definitions, {"confounder"})
    for fold_index, fold in enumerate(inner_splits, start=1):
        train_ids = [int(value) for value in fold["fit_row_ids"] if int(value) in position]
        valid_ids = [int(value) for value in fold["heldout_row_ids"] if int(value) in position]
        if not train_ids or not valid_ids:
            continue
        train_features = extracted_by_id.loc[train_ids].reset_index(drop=True)
        valid_features = extracted_by_id.loc[valid_ids].reset_index(drop=True)
        train_data = dataset.iloc[train_ids]
        t_train = train_data[treatment_column].to_numpy(dtype=float)
        y_train = train_data[outcome_column].to_numpy(dtype=float)
        t_encoder = _FeatureEncoder(propensity_defs).fit(train_features)
        y_encoder = _FeatureEncoder(definitions).fit(train_features)
        x_t_train = t_encoder.transform(train_features)
        x_t_valid = t_encoder.transform(valid_features)
        x_y_train = y_encoder.transform(train_features)
        x_y_valid = y_encoder.transform(valid_features)
        treatment_model = _fit_classifier(x_t_train, t_train, seed=seed + fold_index)
        outcome_models = _fit_outcome_models(
            x_y_train,
            t_train,
            y_train,
            binary=binary,
            seed=seed + fold_index,
        )
        fold_mu0, fold_mu1 = _predict_outcomes(outcome_models, x_y_valid)
        valid_positions = [position[row_id] for row_id in valid_ids]
        e[valid_positions] = _predict_probability(treatment_model, x_t_valid)
        mu0[valid_positions] = fold_mu0
        mu1[valid_positions] = fold_mu1
    if np.isnan(e).any() or np.isnan(mu0).any() or np.isnan(mu1).any():
        raise ValueError(
            "inner splits do not provide one nuisance prediction for every outer-fit row"
        )
    return e, mu0, mu1


def estimate_outer_fold(
    *,
    dataset: pd.DataFrame,
    extracted_fit: pd.DataFrame,
    extracted_heldout: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    unit_id_column: str,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    inner_folds: int,
    seed: int,
    propensity_clip: float,
    estimation_trees: int,
    output_dir: Path,
) -> dict[str, Any]:
    complete_path = output_dir / "complete.json"
    diagnostics_path = output_dir / "diagnostics.json"
    estimation_input_fingerprint = _value_fingerprint(
        {
            "schema_version": ESTIMATION_CHECKPOINT_SCHEMA_VERSION,
            "definitions": list(definitions),
            "split": dict(split),
            "unit_id_column": unit_id_column,
            "treatment_column": treatment_column,
            "outcome_column": outcome_column,
            "outcome_type": outcome_type,
            "inner_folds": inner_folds,
            "seed": seed,
            "propensity_clip": propensity_clip,
            "estimation_trees": estimation_trees,
            "dataset_modeling_fingerprint": _frame_fingerprint(
                dataset[[unit_id_column, treatment_column, outcome_column]]
            ),
            "extracted_fit_fingerprint": _frame_fingerprint(extracted_fit),
            "extracted_heldout_fingerprint": _frame_fingerprint(extracted_heldout),
        }
    )
    if complete_path.is_file() and diagnostics_path.is_file():
        try:
            completion = json.loads(complete_path.read_text(encoding="utf-8"))
            if (
                completion.get("schema_version") == ESTIMATION_CHECKPOINT_SCHEMA_VERSION
                and completion.get("input_fingerprint") == estimation_input_fingerprint
            ):
                return json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except (
            AttributeError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            pass
        LOGGER.info("rerun incompatible Stage 2 outer-fold estimation: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_ids = [int(value) for value in split["fit_row_ids"]]
    heldout_ids = [int(value) for value in split["heldout_row_ids"]]
    inner = list(split.get("inner_splits") or []) or _fallback_inner_splits(
        fit_ids, folds=inner_folds, seed=seed
    )
    binary = str(outcome_type) == "binary"
    e_oof, mu0_oof, mu1_oof = _cross_fitted_nuisance(
        dataset=dataset,
        extracted=extracted_fit,
        definitions=definitions,
        fit_ids=fit_ids,
        inner_splits=inner,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        binary=binary,
        seed=seed,
    )
    fit_data = dataset.iloc[fit_ids]
    heldout_data = dataset.iloc[heldout_ids]
    t_fit = fit_data[treatment_column].to_numpy(dtype=float)
    y_fit = fit_data[outcome_column].to_numpy(dtype=float)
    t_heldout = heldout_data[treatment_column].to_numpy(dtype=float)
    y_heldout = heldout_data[outcome_column].to_numpy(dtype=float)

    propensity_defs = _definitions_for_roles(definitions, {"confounder"})
    effect_defs = _definitions_for_roles(definitions, {"effect_modifier"})
    t_encoder = _FeatureEncoder(propensity_defs).fit(extracted_fit)
    y_encoder = _FeatureEncoder(definitions).fit(extracted_fit)
    effect_encoder = _FeatureEncoder(effect_defs).fit(extracted_fit)
    x_t_fit = t_encoder.transform(extracted_fit)
    x_t_heldout = t_encoder.transform(extracted_heldout)
    x_y_fit = y_encoder.transform(extracted_fit)
    x_y_heldout = y_encoder.transform(extracted_heldout)
    x_effect_fit = effect_encoder.transform(extracted_fit)
    x_effect_heldout = effect_encoder.transform(extracted_heldout)
    treatment_model = _fit_classifier(x_t_fit, t_fit, seed=seed + 10_000)
    outcome_models = _fit_outcome_models(x_y_fit, t_fit, y_fit, binary=binary, seed=seed + 10_000)
    propensity = _predict_probability(treatment_model, x_t_heldout)
    mu0, mu1 = _predict_outcomes(outcome_models, x_y_heldout)
    pseudo_fit = _dr_score(
        y_fit,
        t_fit,
        mu0_oof,
        mu1_oof,
        e_oof,
        clip=propensity_clip,
    )
    effect_model = _fit_effect_model(
        x_effect_fit,
        pseudo_fit,
        seed=seed + 20_000,
        trees=estimation_trees,
    )
    cate = effect_model.predict(x_effect_heldout)
    aipw = _dr_score(
        y_heldout,
        t_heldout,
        mu0,
        mu1,
        propensity,
        clip=propensity_clip,
    )
    predictions = pd.DataFrame(
        {
            "_oci_row_id": heldout_ids,
            unit_id_column: heldout_data[unit_id_column].tolist(),
            "treatment": t_heldout,
            "outcome": y_heldout,
            "propensity": propensity,
            "mu0": mu0,
            "mu1": mu1,
            "aipw_score": aipw,
            "estimated_cate": cate,
        }
    )
    _write_frame(output_dir / "predictions.csv", predictions)
    finite = aipw[np.isfinite(aipw)]
    if not len(finite):
        raise ValueError("outer-fold estimation produced no finite AIPW scores")
    ate = float(np.mean(finite))
    standard_error = (
        float(np.std(finite, ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else None
    )
    diagnostics = {
        "rows": len(heldout_ids),
        "fit_rows": len(fit_ids),
        "features": len(definitions),
        "confounders": len(propensity_defs),
        "effect_modifiers": len(effect_defs),
        "ate_aipw": ate,
        "standard_error": standard_error,
        "confidence_interval_95": (
            [ate - 1.96 * standard_error, ate + 1.96 * standard_error]
            if standard_error is not None
            else None
        ),
        "mean_estimated_cate": float(np.mean(cate)) if len(cate) else None,
        "propensity_min": float(np.min(propensity)) if len(propensity) else None,
        "propensity_max": float(np.max(propensity)) if len(propensity) else None,
        "propensity_clip": propensity_clip,
        "clipped_low_rows": int(np.sum(propensity < propensity_clip)),
        "clipped_high_rows": int(np.sum(propensity > 1.0 - propensity_clip)),
        "predictions_path": str(output_dir / "predictions.csv"),
    }
    _write_json(diagnostics_path, diagnostics)
    _write_json(
        complete_path,
        {
            "status": "complete",
            "schema_version": ESTIMATION_CHECKPOINT_SCHEMA_VERSION,
            "input_fingerprint": estimation_input_fingerprint,
            "completed_at": _now(),
            "rows": len(heldout_ids),
        },
    )
    return diagnostics


def run_fold_analysis(
    *,
    dataset: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    clinical_question: str,
    unit_id_column: str,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    inner_folds: int,
    seed: int,
    output_dir: Path,
    request_json: RequestJSON,
    config: Any,
) -> dict[str, Any]:
    """Run bounded training-fold review and held-out causal estimation."""

    (
        ontology_refinement_min_failure_patients,
        max_ontology_refinement_rounds,
    ) = _ontology_refinement_limits(config)
    extraction_feature_batch_size = _configured_extraction_feature_batch_size(config)
    fit_ids = [int(value) for value in split["fit_row_ids"]]
    heldout_ids = [int(value) for value in split["heldout_row_ids"]]
    current: list[dict[str, Any]] = []
    for raw_feature in definitions:
        feature = dict(raw_feature)
        value_type = str(feature.get("value_type") or "ambiguous").strip().lower()
        if value_type in {"binary", "categorical", "ordinal"}:
            feature["categories_or_unit"] = _validated_closed_category_values(
                value_type=value_type,
                values=feature.get("categories_or_unit") or [],
                source=f"feature {feature.get('name')!r}",
            )
        current.append(_normalized_feature_modeling_definition(feature))
    final_fit_extraction: pd.DataFrame | None = None
    final_fit_definitions: list[dict[str, Any]] | None = None
    agent_review_rounds = 0
    evaluation_rounds = 0
    ontology_refinement_rounds = 0
    maximum_evaluation_rounds = int(config.max_review_rounds) + 4 * len(current) + 4
    for round_index in range(1, maximum_evaluation_rounds + 1):
        evaluation_rounds = round_index
        round_dir = output_dir / "review" / f"round_{round_index:03d}"
        _write_json(round_dir / "definitions.json", {"features": current})
        extracted, current, feedback_rounds = _extract_training_with_ontology_feedback(
            dataset=dataset,
            row_ids=fit_ids,
            text_column=text_column,
            definitions=current,
            output_dir=round_dir / "extraction",
            feedback_dir=round_dir / "ontology_refinement",
            request_json=request_json,
            workers=config.workers,
            max_prompt_chars=config.extraction_max_prompt_chars,
            feature_batch_size=extraction_feature_batch_size,
            minimum_failure_patients=ontology_refinement_min_failure_patients,
            max_refinement_rounds=max_ontology_refinement_rounds,
        )
        ontology_refinement_rounds += feedback_rounds
        current = [
            _normalized_feature_modeling_definition(feature) for feature in current
        ]
        extraction_definitions = [dict(feature) for feature in current]
        _write_json(
            round_dir / "definitions_after_ontology_refinement.json",
            {"features": current, "ontology_refinement_rounds": feedback_rounds},
        )
        summaries = feature_summaries(extracted, current)
        performance = evaluate_definitions(
            dataset=dataset,
            extracted=extracted,
            definitions=current,
            split=split,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            outcome_type=outcome_type,
            inner_folds=inner_folds,
            seed=seed + 1_000 * round_index,
            propensity_clip=config.propensity_clip,
        )
        _write_json(round_dir / "extraction_summary.json", summaries)
        _write_json(round_dir / "performance.json", performance)
        review_path = round_dir / "review.json"
        complete_path = round_dir / "complete.json"
        review_performed = bool(current) and agent_review_rounds < int(
            config.max_review_rounds
        )
        review_input_fingerprint: str | None = None
        if review_performed:
            agent_review_rounds += 1
            allow_revision = agent_review_rounds < int(config.max_review_rounds)
            review_input_fingerprint = _value_fingerprint(
                {
                    "schema_version": REVIEW_CHECKPOINT_SCHEMA_VERSION,
                    "clinical_question": clinical_question,
                    "definitions": current,
                    "summaries": summaries,
                    "performance": performance,
                    "allow_measurement_revision": allow_revision,
                    "minimum_nonmissing_fraction": config.min_nonmissing_fraction,
                    "maximum_dominant_value_fraction": config.max_dominant_fraction,
                }
            )
            review: dict[str, Any] | None = None
            if complete_path.is_file() and review_path.is_file():
                try:
                    completion = json.loads(complete_path.read_text(encoding="utf-8"))
                    if (
                        completion.get("review_schema_version")
                        == REVIEW_CHECKPOINT_SCHEMA_VERSION
                        and completion.get("review_input_fingerprint")
                        == review_input_fingerprint
                    ):
                        review = _validate_review(
                            json.loads(review_path.read_text(encoding="utf-8")),
                            definitions=current,
                            allow_measurement_revision=allow_revision,
                            summaries=summaries,
                        )
                except (
                    AttributeError,
                    OSError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    review = None
            if review is None:
                review = _request_partitioned_review(
                    clinical_question=clinical_question,
                    definitions=current,
                    summaries=summaries,
                    performance=performance,
                    allow_measurement_revision=allow_revision,
                    min_nonmissing_fraction=config.min_nonmissing_fraction,
                    max_dominant_fraction=config.max_dominant_fraction,
                    max_prompt_chars=config.max_prompt_chars,
                    output_dir=round_dir / "review_batches",
                    request_json=request_json,
                )
                _write_json(review_path, review)
            reviewed, _measurement_changed = _apply_review(current, review)
        else:
            review = {
                "feature_decisions": [],
                "overall_assessment": (
                    "Evaluation-only convergence round after the final allowed agent review."
                    if current
                    else "No retained features to review."
                ),
                "evaluation_only": True,
            }
            _write_json(review_path, review)
            reviewed = [dict(feature) for feature in current]
        representation_changed_ids = _changed_feature_representation_ids(current, reviewed)
        updated, signal_pruning = _apply_empirical_signal_pruning(
            reviewed,
            performance,
            defer_feature_ids=representation_changed_ids,
        )
        _write_json(round_dir / "signal_pruning.json", signal_pruning)
        definitions_changed = _value_fingerprint(current) != _value_fingerprint(updated)
        final_fit_extraction = extracted
        final_fit_definitions = extraction_definitions
        current = updated
        _write_json(
            complete_path,
            {
                "status": "complete",
                "completed_at": _now(),
                "evaluation_round": round_index,
                "agent_review_performed": review_performed,
                "agent_review_rounds": agent_review_rounds,
                "review_schema_version": (
                    REVIEW_CHECKPOINT_SCHEMA_VERSION if review_performed else None
                ),
                "review_input_fingerprint": review_input_fingerprint,
                "definitions_changed": definitions_changed,
                "features_retained": len(current),
            },
        )
        if not definitions_changed:
            break
    else:  # pragma: no cover - finite feature/role changes should converge first
        raise RuntimeError("Stage 2 empirical feature pruning did not converge")

    if final_fit_extraction is None or final_fit_definitions is None:
        raise RuntimeError("Stage 2 review did not produce a training-fold extraction")
    names = [str(feature["name"]) for feature in current]
    if _value_fingerprint(final_fit_definitions) == _value_fingerprint(current):
        fit_selected = final_fit_extraction[["_oci_row_id", *names]].copy()
        _write_frame(output_dir / "extraction" / "fit" / "extracted.csv", fit_selected)
        _write_json(
            output_dir / "extraction" / "fit" / "complete.json",
            {
                "status": "complete",
                "completed_at": _now(),
                "rows": len(fit_selected),
                "features": len(current),
                "reused_from_evaluation_round": evaluation_rounds,
            },
        )
    else:
        LOGGER.info(
            "Stage 2 final feature set changed during review; re-extracting %s "
            "training rows against the %s retained definition(s)",
            len(fit_ids),
            len(current),
        )
        fit_selected, current, feedback_rounds = _extract_training_with_ontology_feedback(
            dataset=dataset,
            row_ids=fit_ids,
            text_column=text_column,
            definitions=current,
            output_dir=output_dir / "extraction" / "fit",
            feedback_dir=output_dir / "extraction" / "fit_ontology_refinement",
            request_json=request_json,
            workers=config.workers,
            max_prompt_chars=config.extraction_max_prompt_chars,
            feature_batch_size=extraction_feature_batch_size,
            minimum_failure_patients=ontology_refinement_min_failure_patients,
            max_refinement_rounds=max_ontology_refinement_rounds,
        )
        ontology_refinement_rounds += feedback_rounds
        names = [str(feature["name"]) for feature in current]
    _write_json(
        output_dir / "final_definitions.json",
        {
            "features": current,
            "review_rounds": agent_review_rounds,
            "evaluation_rounds": evaluation_rounds,
            "ontology_refinement_rounds": ontology_refinement_rounds,
        },
    )
    _assert_extraction_health(
        fit_selected,
        current,
        scope="training",
        minimum_row_nonmissing_fraction=config.min_nonmissing_fraction,
        audit_path=output_dir / "extraction" / "fit_health.json",
    )
    heldout_extraction = extract_rows(
        dataset=dataset,
        row_ids=heldout_ids,
        text_column=text_column,
        definitions=current,
        output_dir=output_dir / "extraction" / "heldout",
        request_json=request_json,
        workers=config.workers,
        max_prompt_chars=config.extraction_max_prompt_chars,
        feature_batch_size=extraction_feature_batch_size,
    )
    _assert_extraction_health(
        heldout_extraction,
        current,
        scope="heldout",
        minimum_row_nonmissing_fraction=config.min_nonmissing_fraction,
        audit_path=output_dir / "extraction" / "heldout_health.json",
    )
    combined = pd.concat([fit_selected, heldout_extraction], ignore_index=True).sort_values(
        "_oci_row_id"
    )
    _write_frame(output_dir / "extraction" / "extracted_features.csv", combined)
    diagnostics = estimate_outer_fold(
        dataset=dataset,
        extracted_fit=fit_selected,
        extracted_heldout=heldout_extraction,
        definitions=current,
        split=split,
        unit_id_column=unit_id_column,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        outcome_type=outcome_type,
        inner_folds=inner_folds,
        seed=seed,
        propensity_clip=config.propensity_clip,
        estimation_trees=config.estimation_trees,
        output_dir=output_dir / "estimation",
    )
    return {
        "features": current,
        "review_rounds": agent_review_rounds,
        "evaluation_rounds": evaluation_rounds,
        "ontology_refinement_rounds": ontology_refinement_rounds,
        "estimation": diagnostics,
    }


__all__ = ["run_fold_analysis"]
