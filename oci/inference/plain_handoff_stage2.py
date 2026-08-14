"""Plain, resumable Stage 2 analysis of the researcher Stage 1 handoff.

This module intentionally treats a directory as the checkpoint.  It reads the
ordinary JSONL handoff, defines and extracts patient-level variables, reviews
them using training-fold performance, and produces cross-fitted causal
estimates.  It has no bundle format, artifact authentication, immutable
request, content hashes, or checkpoint adoption.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .plain_handoff_stage2_evidence import (
    EVIDENCE_COMPILER_VERSION,
    SUPPORTED_STAGE2_ARCHITECTURES,
    compile_stage2_handoff_evidence,
)

LOGGER = logging.getLogger(__name__)

ALLOWED_VALUE_TYPES = {"binary", "categorical", "continuous", "ordinal", "ambiguous"}
ALLOWED_EVIDENCE_AXES = {
    "treatment",
    "outcome",
    "residual_effect",
    "matched_pair",
    "semantic",
    "unclear",
}
ALLOWED_ROLES = {"confounder", "prognostic", "effect_modifier"}
MAX_RESPONSE_REPAIRS = 5
DEFAULT_EXTRACTION_MAX_PROMPT_CHARS = 640_000
DEFAULT_CONSOLIDATION_MAX_PROMPT_CHARS = 640_000
DEFAULT_OPERATIONALIZATION_MAX_PROMPT_CHARS = 640_000
DEFAULT_CONSOLIDATION_BATCH_SIZE = 20
DEFAULT_CONSOLIDATION_ALPHABETICAL_ROUNDS = 5
DEFAULT_CONSOLIDATION_SHUFFLE_ROUNDS = 50
DEFAULT_CONSOLIDATION_MAX_ROUNDS = (
    DEFAULT_CONSOLIDATION_ALPHABETICAL_ROUNDS + DEFAULT_CONSOLIDATION_SHUFFLE_ROUNDS
)
DEFAULT_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS = 3
DEFAULT_MAX_ONTOLOGY_REFINEMENT_ROUNDS = 2
CONSOLIDATION_SCHEMA_VERSION = "global_candidate_pool_v13_merge_only_then_role_filter"
GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION = (
    "alphabetical_then_seeded_shuffle_candidate_batches_v8_merge_only"
)
EXTRACTION_ONTOLOGY_FEEDBACK_SCHEMA_VERSION = (
    "training_failure_ontology_refinement_v1_explicit_feature_invariants"
)
OPERATIONALIZATION_SCHEMA_VERSION = "feature_name_bounded_supporting_text_v4"
INTERPRETATION_SCHEMA_VERSION = "clinical_feature_text_only_ordinals_v7"
INTERPRETATION_AUDIT_SCHEMA_VERSION = "rejected_packet_text_only_ordinals_v4"
CONFIGURED_EXPLICIT_FEATURE_ARCHITECTURE = "configured_explicit_feature"

_CONCEPT_IDENTITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "assessment",
    "at",
    "baseline",
    "be",
    "biomarker",
    "by",
    "clinical",
    "confounder",
    "diagnosis",
    "disease",
    "documentation",
    "effect",
    "evidence",
    "expression",
    "feature",
    "for",
    "from",
    "has",
    "have",
    "history",
    "in",
    "information",
    "is",
    "level",
    "measurement",
    "metastasis",
    "modifier",
    "mutation",
    "outcome",
    "of",
    "on",
    "or",
    "patient",
    "presence",
    "pretreatment",
    "prognostic",
    "record",
    "residual",
    "score",
    "specific",
    "status",
    "that",
    "the",
    "this",
    "to",
    "treatment",
    "use",
    "value",
    "variable",
    "was",
    "were",
    "with",
    "without",
}


def _concept_identity_tokens(*values: Any) -> set[str]:
    """Return conservative lexical anchors for one patient-level measurement."""

    text = " ".join(str(value) for value in values if value is not None).lower()
    tokens: set[str] = set()
    for raw_token in re.findall(r"[a-z]+[a-z0-9]*", text):
        if raw_token in _CONCEPT_IDENTITY_STOPWORDS:
            continue
        token = raw_token
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        if len(token) > 1 and token not in _CONCEPT_IDENTITY_STOPWORDS:
            tokens.add(token)

    # Add compact variants for separator-delimited identifier fragments without
    # knowing anything about the clinical domain. This makes forms such as
    # ``ab_c1`` and ``ab-c1`` comparable while remaining conservative for words.
    for compound in re.findall(r"[a-z0-9]+(?:[_-][a-z0-9]+)+", text):
        parts = re.split(r"[_-]", compound)
        for start in range(len(parts) - 1):
            for stop in range(start + 2, len(parts) + 1):
                segment = parts[start:stop]
                has_digit = any(any(char.isdigit() for char in part) for part in segment)
                all_short = all(len(part) <= 3 for part in segment)
                if not (has_digit or all_short):
                    continue
                compact = "".join(segment)
                if len(compact) > 2:
                    tokens.add(compact)
    return tokens


def _consolidation_route_is_semantically_compatible(
    candidate: Mapping[str, Any],
    feature: Mapping[str, Any],
) -> bool:
    """Require a shared measurement anchor before two concepts may be routed together."""

    candidate_tokens = _concept_identity_tokens(
        candidate.get("name"),
        candidate.get("description"),
    )
    feature_tokens = _concept_identity_tokens(
        feature.get("name"),
        feature.get("description"),
        feature.get("measurement_definition"),
    )
    # Old/custom candidate producers may not supply concept text. In that case
    # packet grounding remains the only available check.
    if not candidate_tokens or not feature_tokens:
        return True
    return bool(candidate_tokens.intersection(feature_tokens))


def _canonical_evidence_axes(value: Any) -> list[str]:
    raw_axes = [value] if isinstance(value, str) else list(value or [])
    aliases = {
        "assignment": ("treatment",),
        "confounder": ("treatment", "outcome"),
        "effect": ("residual_effect",),
        "effect_modifier": ("residual_effect",),
        "heterogeneity": ("residual_effect",),
        "interaction": ("residual_effect",),
        "matched": ("matched_pair",),
        "propensity": ("treatment",),
        "prognostic": ("outcome",),
        "r_loss": ("residual_effect",),
        "unknown": ("unclear",),
    }
    canonical: set[str] = set()
    for raw_axis in raw_axes:
        tokens = re.split(r"[,;|/]", str(raw_axis))
        for token in tokens:
            normalized = re.sub(r"[^a-z0-9]+", "_", token.strip().lower()).strip("_")
            if normalized in ALLOWED_EVIDENCE_AXES:
                canonical.add(normalized)
            else:
                canonical.update(aliases.get(normalized, ()))
    return sorted(canonical or {"unclear"})


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


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _value_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_ite_correlation(
    truth: Sequence[float],
    estimate: Sequence[float],
    *,
    rank: bool,
) -> float | None:
    left = np.asarray(truth, dtype=float)
    right = np.asarray(estimate, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 2:
        return None
    left = left[finite]
    right = right[finite]
    if rank:
        left = pd.Series(left).rank(method="average").to_numpy(dtype=float)
        right = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    if float(np.std(left)) <= 0.0 or float(np.std(right)) <= 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _evaluate_stage2_oracle_ite(
    *,
    prediction_path: Path,
    dataset: pd.DataFrame,
    output_dir: Path,
    oracle_ite_column: str = "true_ite_prob",
) -> dict[str, Any]:
    """Evaluate frozen cross-fitted ITEs without exposing truth to modeling."""

    prediction_path = Path(prediction_path)
    output_dir = Path(output_dir)
    metrics_path = output_dir / "posthoc_oracle_ite_metrics.json"
    frozen_sha256 = _file_sha256(prediction_path)
    base: dict[str, Any] = {
        "schema_version": "stage2_posthoc_oracle_ite_v1",
        "available": False,
        "evaluation_is_post_hoc": True,
        "all_outer_predictions_frozen_before_oracle_join": True,
        "oracle_columns_consumed_by_modeling": False,
        "oracle_ite_column": oracle_ite_column,
        "estimated_ite_column": "estimated_cate",
        "frozen_prediction_path": str(prediction_path),
        "frozen_prediction_sha256": frozen_sha256,
        "metrics_path": str(metrics_path),
    }
    if oracle_ite_column not in dataset.columns:
        payload = {
            **base,
            "reason": f"dataset does not contain {oracle_ite_column!r}",
        }
        _write_json(metrics_path, payload)
        return payload

    predictions = pd.read_csv(prediction_path)
    if any(str(column).startswith("true_") for column in predictions.columns):
        raise RuntimeError("frozen Stage 2 predictions contain an oracle column")
    required = {"_oci_row_id", "outer_fold", "estimated_cate"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"frozen Stage 2 predictions lack required columns: {missing}")

    oracle_values = pd.to_numeric(dataset[oracle_ite_column], errors="coerce")
    if oracle_values.isna().any() or not np.isfinite(oracle_values.to_numpy(dtype=float)).all():
        payload = {
            **base,
            "reason": f"dataset column {oracle_ite_column!r} is not complete and finite",
        }
        _write_json(metrics_path, payload)
        return payload
    oracle = pd.DataFrame(
        {
            "_oci_row_id": np.arange(len(dataset), dtype=int),
            oracle_ite_column: oracle_values.to_numpy(dtype=float),
        }
    )
    evaluated = predictions.merge(
        oracle,
        on="_oci_row_id",
        how="left",
        validate="one_to_one",
    )
    if len(evaluated) != len(dataset) or evaluated[oracle_ite_column].isna().any():
        raise ValueError("oracle ITE join did not cover every frozen Stage 2 prediction")

    def metrics_for(frame: pd.DataFrame) -> dict[str, Any]:
        truth = frame[oracle_ite_column].to_numpy(dtype=float)
        estimate = pd.to_numeric(frame["estimated_cate"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(truth) & np.isfinite(estimate)
        truth = truth[finite]
        estimate = estimate[finite]
        if not len(truth):
            return {
                "n": int(len(frame)),
                "finite_pairs": 0,
                "pearson_correlation": None,
                "spearman_correlation": None,
                "mae": None,
                "rmse": None,
                "mean_error": None,
                "mean_estimated_ite": None,
                "oracle_ate": None,
                "ate_bias": None,
                "estimated_ite_standard_deviation": None,
                "oracle_ite_standard_deviation": None,
            }
        error = estimate - truth
        return {
            "n": int(len(frame)),
            "finite_pairs": int(len(truth)),
            "pearson_correlation": _safe_ite_correlation(truth, estimate, rank=False),
            "spearman_correlation": _safe_ite_correlation(truth, estimate, rank=True),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mean_error": float(np.mean(error)),
            "mean_estimated_ite": float(np.mean(estimate)),
            "oracle_ate": float(np.mean(truth)),
            "ate_bias": float(np.mean(estimate) - np.mean(truth)),
            "estimated_ite_standard_deviation": float(np.std(estimate)),
            "oracle_ite_standard_deviation": float(np.std(truth)),
        }

    evaluated_path = output_dir / "posthoc_predictions_with_oracle_ite.csv"
    temporary = evaluated_path.with_name(f".{evaluated_path.name}.{os.getpid()}.tmp")
    evaluated.to_csv(temporary, index=False)
    os.replace(temporary, evaluated_path)
    payload = {
        **base,
        "available": True,
        "predictions_with_oracle_path": str(evaluated_path),
        "overall": metrics_for(evaluated),
        "per_fold": [
            {"outer_fold": int(fold), **metrics_for(frame)}
            for fold, frame in evaluated.groupby("outer_fold", sort=True)
        ],
    }
    _write_json(metrics_path, payload)
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path} line {line_number} is not a JSON object")
            yield dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


@dataclass(frozen=True)
class Stage2ExplicitFeature:
    """Investigator-specified feature and complete extraction ontology."""

    name: str
    description: str
    value_type: str
    categories_or_unit: tuple[str, ...]
    measurement_definition: str
    missing_value_rule: str
    roles: tuple[str, ...]
    stability_summary: str = ""
    caveats: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("stage2 explicit feature name must be a string")
        name = re.sub(r"[^a-z0-9]+", "_", str(self.name).strip().lower()).strip("_")
        if not name:
            raise ValueError("stage2 explicit feature name must contain letters or numbers")
        description = str(self.description or "").strip()
        if not description:
            raise ValueError(f"stage2 explicit feature {name!r} requires a nonempty description")
        value_type = str(self.value_type or "").strip().lower()
        value_type = {
            "bool": "binary",
            "boolean": "binary",
            "category": "categorical",
            "numeric": "continuous",
            "number": "continuous",
            "unknown": "ambiguous",
        }.get(value_type, value_type)
        if value_type not in ALLOWED_VALUE_TYPES:
            raise ValueError(
                f"stage2 explicit feature {name!r} value_type must be one of "
                f"{sorted(ALLOWED_VALUE_TYPES)}"
            )
        raw_categories: Any = self.categories_or_unit
        if isinstance(raw_categories, str):
            raw_categories = [raw_categories]
        if not isinstance(raw_categories, Sequence) or isinstance(
            raw_categories, (bytes, bytearray)
        ):
            raise ValueError(f"stage2 explicit feature {name!r} categories_or_unit must be a list")
        categories = [str(item).strip() for item in raw_categories if str(item).strip()]
        if value_type in {"binary", "categorical", "ordinal"}:
            from .plain_handoff_stage2_analysis import _validated_closed_category_values

            categories = _validated_closed_category_values(
                value_type=value_type,
                values=categories,
                source=f"stage2 explicit feature {name!r}",
            )
        measurement_definition = str(self.measurement_definition or "").strip()
        if not measurement_definition:
            raise ValueError(f"stage2 explicit feature {name!r} requires measurement_definition")
        missing_value_rule = str(self.missing_value_rule or "").strip()
        if not missing_value_rule:
            raise ValueError(f"stage2 explicit feature {name!r} requires missing_value_rule")
        raw_roles: Any = self.roles
        if isinstance(raw_roles, str):
            raw_roles = [raw_roles]
        if not isinstance(raw_roles, Sequence) or isinstance(raw_roles, (bytes, bytearray)):
            raise ValueError(f"stage2 explicit feature {name!r} roles must be a list")
        roles = list(
            dict.fromkeys(str(role).strip().lower() for role in raw_roles if str(role).strip())
        )
        if not roles:
            raise ValueError(f"stage2 explicit feature {name!r} requires at least one causal role")
        unsupported_roles = sorted(set(roles) - ALLOWED_ROLES)
        if unsupported_roles:
            raise ValueError(
                f"stage2 explicit feature {name!r} contains unsupported roles: "
                f"{unsupported_roles}; allowed roles are {sorted(ALLOWED_ROLES)}"
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "categories_or_unit", tuple(categories))
        object.__setattr__(self, "measurement_definition", measurement_definition)
        object.__setattr__(self, "missing_value_rule", missing_value_rule)
        object.__setattr__(self, "roles", tuple(roles))
        object.__setattr__(self, "stability_summary", str(self.stability_summary or "").strip())
        object.__setattr__(self, "caveats", str(self.caveats or "").strip())

    def as_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "value_type": self.value_type,
            "categories_or_unit": list(self.categories_or_unit),
            "measurement_definition": self.measurement_definition,
            "missing_value_rule": self.missing_value_rule,
            "roles": list(self.roles),
            "stability_summary": self.stability_summary,
            "caveats": self.caveats,
        }


def _stage2_explicit_feature_from_mapping(
    raw: Any,
    *,
    source: str,
) -> Stage2ExplicitFeature:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source} must be an object containing a feature ontology")
    entry = dict(raw)
    nested = entry.pop("ontology", None)
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError(f"{source}.ontology must be an object")
        combined = dict(nested)
        combined.update(entry)
    else:
        combined = entry

    def required_value(key: str, *aliases: str) -> Any:
        for candidate in (key, *aliases):
            if candidate in combined:
                return combined[candidate]
        alias_text = f" (or {', '.join(aliases)})" if aliases else ""
        raise ValueError(f"{source} requires ontology field {key}{alias_text}")

    raw_categories = required_value("categories_or_unit", "categories", "unit")
    if raw_categories is None:
        raw_categories = []
    elif isinstance(raw_categories, (str, int, float, bool)):
        raw_categories = [raw_categories]
    elif not isinstance(raw_categories, Sequence):
        raise ValueError(f"{source}.categories_or_unit must be a list or unit string")
    raw_roles = required_value("roles")
    if isinstance(raw_roles, str):
        raw_roles = [raw_roles]
    elif not isinstance(raw_roles, Sequence):
        raise ValueError(f"{source}.roles must be a list")

    return Stage2ExplicitFeature(
        name=required_value("name"),
        description=required_value("description"),
        value_type=required_value("value_type", "type"),
        categories_or_unit=tuple(item for item in raw_categories if item is not None),
        measurement_definition=required_value("measurement_definition"),
        missing_value_rule=required_value("missing_value_rule"),
        roles=tuple(role for role in raw_roles if role is not None),
        stability_summary=str(combined.get("stability_summary") or ""),
        caveats=str(combined.get("caveats") or ""),
    )


def _stage2_explicit_features_from_value(raw: Any) -> tuple[Stage2ExplicitFeature, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("stage2.explicit_features.enabled must be true or false")
        entries = raw.get("features")
        if entries is None:
            raise ValueError("stage2.explicit_features must contain a features list")
        if not enabled:
            if entries:
                raise ValueError(
                    "stage2.explicit_features cannot contain features when enabled=false"
                )
            return ()
    else:
        entries = raw
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        raise ValueError("stage2.explicit_features must be a list of feature ontologies")
    features = tuple(
        _stage2_explicit_feature_from_mapping(
            entry,
            source=f"stage2.explicit_features[{index}]",
        )
        for index, entry in enumerate(entries)
    )
    names = [feature.name for feature in features]
    duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicate_names:
        raise ValueError(
            "stage2.explicit_features contains duplicate normalized feature names: "
            f"{duplicate_names}"
        )
    return features


@dataclass(frozen=True)
class PlainHandoffStage2Config:
    endpoint: str
    model: str = ""
    api_key: str = "EMPTY"
    request_timeout: float = 7_200.0
    transport_max_attempts: int = 3
    transport_retry_backoff: float = 2.0
    max_prompt_chars: int = 100_000
    # Candidate consolidation has its own prompt allowance even though each
    # request sees only one bounded deterministic batch.
    consolidation_max_prompt_chars: int = DEFAULT_CONSOLIDATION_MAX_PROMPT_CHARS
    # A merged alias family can cite substantially more evidence than one
    # interpretation batch. Pack that evidence under an independent allowance.
    operationalization_max_prompt_chars: int = DEFAULT_OPERATIONALIZATION_MAX_PROMPT_CHARS
    consolidation_batch_size: int = DEFAULT_CONSOLIDATION_BATCH_SIZE
    consolidation_alphabetical_rounds: int = DEFAULT_CONSOLIDATION_ALPHABETICAL_ROUNDS
    consolidation_max_rounds: int = DEFAULT_CONSOLIDATION_MAX_ROUNDS
    # Extraction repeats a complete frozen feature ontology for one patient.
    # Keep its larger context allowance separate so discovery batching and its
    # evidence-compilation fingerprints remain stable.
    extraction_max_prompt_chars: int = DEFAULT_EXTRACTION_MAX_PROMPT_CHARS
    evidence_compiler: str = EVIDENCE_COMPILER_VERSION
    required_architectures: tuple[str, ...] = SUPPORTED_STAGE2_ARCHITECTURES
    included_architectures: tuple[str, ...] | None = None
    evidence_max_cards_per_fold: int = 400
    evidence_max_exemplars_per_card: int = 4
    evidence_max_exemplar_chars: int = 2_400
    # Accepted for compatibility with existing run files. Consolidation no
    # longer caps or ranks semantic groups before extraction.
    max_candidates_per_fold: int = 50
    # Accepted for compatibility with existing run files. Consolidation no
    # longer uses a progressive or oversampled feature beam.
    consolidation_oversample_factor: int = 4
    workers: int = 4
    max_review_rounds: int = 2
    ontology_refinement_min_failure_patients: int = DEFAULT_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS
    max_ontology_refinement_rounds: int = DEFAULT_MAX_ONTOLOGY_REFINEMENT_ROUNDS
    estimation_trees: int = 200
    propensity_clip: float = 0.02
    min_nonmissing_fraction: float = 0.05
    max_dominant_fraction: float = 0.98
    temperature: float = 0.0
    enable_thinking: bool = False
    explicit_features: tuple[Stage2ExplicitFeature, ...] = ()

    def validate(self, *, require_model: bool = True) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("stage2.endpoint must be one HTTP(S) OpenAI-compatible base URL")
        if require_model and not self.model.strip():
            raise ValueError("stage2.model must be nonempty")
        if self.request_timeout <= 0:
            raise ValueError("stage2.request_timeout must be positive")
        if self.transport_max_attempts < 1:
            raise ValueError("stage2.transport_max_attempts must be positive")
        if self.transport_retry_backoff < 0:
            raise ValueError("stage2.transport_retry_backoff must be nonnegative")
        if self.max_prompt_chars < 4_000:
            raise ValueError("stage2.max_prompt_chars must be at least 4000")
        if self.consolidation_max_prompt_chars < 4_000:
            raise ValueError("stage2.consolidation_max_prompt_chars must be at least 4000")
        if self.operationalization_max_prompt_chars < 4_000:
            raise ValueError("stage2.operationalization_max_prompt_chars must be at least 4000")
        if self.consolidation_batch_size < 2:
            raise ValueError("stage2.consolidation_batch_size must be at least 2")
        if self.consolidation_alphabetical_rounds < 0:
            raise ValueError("stage2.consolidation_alphabetical_rounds must be nonnegative")
        if self.consolidation_max_rounds < 1:
            raise ValueError("stage2.consolidation_max_rounds must be positive")
        if self.extraction_max_prompt_chars < 4_000:
            raise ValueError("stage2.extraction_max_prompt_chars must be at least 4000")
        if self.evidence_compiler != EVIDENCE_COMPILER_VERSION:
            raise ValueError(
                f"stage2.evidence_compiler must be {EVIDENCE_COMPILER_VERSION}; "
                "raw_packets_v1 was retired because it merged distinct scientific "
                "architectures"
            )
        required = tuple(self.required_architectures)
        if len(required) != len(set(required)):
            raise ValueError("stage2.required_architectures must not contain duplicates")
        unsupported = sorted(set(required) - set(SUPPORTED_STAGE2_ARCHITECTURES))
        if unsupported:
            raise ValueError(
                f"stage2.required_architectures contains unsupported values: {unsupported}"
            )
        included = (
            tuple(self.included_architectures) if self.included_architectures is not None else None
        )
        if included is not None:
            if len(included) != len(set(included)):
                raise ValueError("stage2.included_architectures must not contain duplicates")
            unsupported_included = sorted(set(included) - set(SUPPORTED_STAGE2_ARCHITECTURES))
            if unsupported_included:
                raise ValueError(
                    "stage2.included_architectures contains unsupported values: "
                    f"{unsupported_included}"
                )
            if not set(required).issubset(included):
                raise ValueError(
                    "stage2.required_architectures must be a subset of "
                    "stage2.included_architectures"
                )
        if self.evidence_max_cards_per_fold < 16:
            raise ValueError("stage2.evidence_max_cards_per_fold must be at least 16")
        if self.evidence_max_exemplars_per_card < 1:
            raise ValueError("stage2.evidence_max_exemplars_per_card must be positive")
        if self.evidence_max_exemplar_chars < 256:
            raise ValueError("stage2.evidence_max_exemplar_chars must be at least 256")
        if self.max_candidates_per_fold < 1:
            raise ValueError("stage2.max_candidates_per_fold must be positive")
        if self.consolidation_oversample_factor < 1:
            raise ValueError("stage2.consolidation_oversample_factor must be positive")
        if self.workers < 1:
            raise ValueError("stage2.workers must be positive")
        if self.max_review_rounds < 1:
            raise ValueError("stage2.max_review_rounds must be positive")
        if self.ontology_refinement_min_failure_patients < 2:
            raise ValueError("stage2.ontology_refinement_min_failure_patients must be at least 2")
        if self.max_ontology_refinement_rounds < 0:
            raise ValueError("stage2.max_ontology_refinement_rounds must be nonnegative")
        if self.estimation_trees < 10:
            raise ValueError("stage2.estimation_trees must be at least 10")
        if not 0.0 < self.propensity_clip < 0.5:
            raise ValueError("stage2.propensity_clip must be between 0 and 0.5")
        if not 0.0 <= self.min_nonmissing_fraction <= 1.0:
            raise ValueError("stage2.min_nonmissing_fraction must be between 0 and 1")
        if not 0.0 <= self.max_dominant_fraction <= 1.0:
            raise ValueError("stage2.max_dominant_fraction must be between 0 and 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("stage2.temperature must be between 0 and 2")
        names: list[str] = []
        for index, feature in enumerate(self.explicit_features):
            if not isinstance(feature, Stage2ExplicitFeature):
                raise ValueError(
                    "stage2.explicit_features entries must be Stage2ExplicitFeature "
                    f"objects; invalid entry at index {index}"
                )
            names.append(feature.name)
        duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicate_names:
            raise ValueError(
                "stage2.explicit_features contains duplicate normalized feature names: "
                f"{duplicate_names}"
            )

    def public_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["api_key"] = "<redacted>"
        values["explicit_features"] = [
            feature.as_definition() for feature in self.explicit_features
        ]
        return values


def plain_stage2_config_from_mapping(
    raw: Mapping[str, Any],
    *,
    default_workers: int,
) -> PlainHandoffStage2Config | None:
    if raw.get("command"):
        raise ValueError(
            "stage2.command is not used by the plain workflow; configure stage2.endpoint"
        )
    endpoint = str(raw.get("endpoint") or "").strip()
    model = str(raw.get("model") or "").strip()
    explicit_features = _stage2_explicit_features_from_value(raw.get("explicit_features"))
    if not endpoint and not model:
        return None
    if not endpoint:
        raise ValueError("stage2.endpoint is required when stage2.model is specified")
    api_key = str(raw.get("api_key") or os.environ.get("OCI_STAGE2_API_KEY") or "EMPTY")
    legacy_extraction_batch_size = raw.get("extraction_batch_size")
    if legacy_extraction_batch_size is not None and int(legacy_extraction_batch_size) != 1:
        LOGGER.warning(
            "stage2.extraction_batch_size=%s is ignored; Stage 2 extraction is permanently "
            "isolated to one patient per prompt",
            legacy_extraction_batch_size,
        )
    if raw.get("max_tokens") is not None:
        LOGGER.warning("stage2.max_tokens is ignored; Stage 2 does not send an output-token limit")

    def architecture_names(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            if value.strip().lower() == "all":
                return SUPPORTED_STAGE2_ARCHITECTURES
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return tuple(value)

    config = PlainHandoffStage2Config(
        endpoint=endpoint.rstrip("/"),
        model=model,
        api_key=api_key,
        request_timeout=float(raw.get("request_timeout", 7_200.0)),
        transport_max_attempts=int(raw.get("transport_max_attempts", 3)),
        transport_retry_backoff=float(raw.get("transport_retry_backoff", 2.0)),
        max_prompt_chars=int(raw.get("max_prompt_chars", 100_000)),
        consolidation_max_prompt_chars=int(
            raw.get(
                "consolidation_max_prompt_chars",
                DEFAULT_CONSOLIDATION_MAX_PROMPT_CHARS,
            )
        ),
        operationalization_max_prompt_chars=int(
            raw.get(
                "operationalization_max_prompt_chars",
                DEFAULT_OPERATIONALIZATION_MAX_PROMPT_CHARS,
            )
        ),
        consolidation_batch_size=int(
            raw.get("consolidation_batch_size", DEFAULT_CONSOLIDATION_BATCH_SIZE)
        ),
        consolidation_alphabetical_rounds=int(
            raw.get(
                "consolidation_alphabetical_rounds",
                DEFAULT_CONSOLIDATION_ALPHABETICAL_ROUNDS,
            )
        ),
        consolidation_max_rounds=int(
            raw.get("consolidation_max_rounds", DEFAULT_CONSOLIDATION_MAX_ROUNDS)
        ),
        extraction_max_prompt_chars=int(
            raw.get(
                "extraction_max_prompt_chars",
                DEFAULT_EXTRACTION_MAX_PROMPT_CHARS,
            )
        ),
        evidence_compiler=str(raw.get("evidence_compiler", EVIDENCE_COMPILER_VERSION)).strip(),
        required_architectures=architecture_names(
            raw.get("required_architectures", SUPPORTED_STAGE2_ARCHITECTURES)
        ),
        included_architectures=(
            architecture_names(raw["included_architectures"])
            if raw.get("included_architectures") is not None
            else None
        ),
        evidence_max_cards_per_fold=int(raw.get("evidence_max_cards_per_fold", 400)),
        evidence_max_exemplars_per_card=int(raw.get("evidence_max_exemplars_per_card", 4)),
        evidence_max_exemplar_chars=int(raw.get("evidence_max_exemplar_chars", 2_400)),
        max_candidates_per_fold=int(raw.get("max_candidates_per_fold", 50)),
        consolidation_oversample_factor=int(raw.get("consolidation_oversample_factor", 4)),
        workers=max(1, int(raw.get("workers", min(4, max(1, default_workers))))),
        max_review_rounds=int(raw.get("max_review_rounds", 2)),
        ontology_refinement_min_failure_patients=int(
            raw.get(
                "ontology_refinement_min_failure_patients",
                DEFAULT_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS,
            )
        ),
        max_ontology_refinement_rounds=int(
            raw.get(
                "max_ontology_refinement_rounds",
                DEFAULT_MAX_ONTOLOGY_REFINEMENT_ROUNDS,
            )
        ),
        estimation_trees=int(raw.get("estimation_trees", 200)),
        propensity_clip=float(raw.get("propensity_clip", 0.02)),
        min_nonmissing_fraction=float(raw.get("min_nonmissing_fraction", 0.05)),
        max_dominant_fraction=float(raw.get("max_dominant_fraction", 0.98)),
        temperature=float(raw.get("temperature", 0.0)),
        enable_thinking=bool(raw.get("enable_thinking", False)),
        explicit_features=explicit_features,
    )
    config.validate(require_model=False)
    return config


def _served_model_ids(config: PlainHandoffStage2Config) -> list[str]:
    """Return the distinct model IDs advertised by an OpenAI-compatible server."""

    from openai import OpenAI

    client = OpenAI(
        base_url=config.endpoint,
        api_key=config.api_key,
        timeout=config.request_timeout,
        max_retries=2,
    )
    try:
        response = client.models.list()
    except Exception as exc:
        raise RuntimeError(
            "Stage 2 could not auto-discover a model from "
            f"{config.endpoint}/models: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        client.close()
    model_ids = {
        str(getattr(model, "id", "")).strip()
        for model in response.data
        if str(getattr(model, "id", "")).strip()
    }
    return sorted(model_ids)


def _resolve_stage2_model(config: PlainHandoffStage2Config) -> PlainHandoffStage2Config:
    """Use the sole model advertised by the endpoint when none was configured."""

    if config.model.strip():
        return config
    model_ids = _served_model_ids(config)
    if not model_ids:
        models_url = f"{config.endpoint}/models"
        raise RuntimeError(f"Stage 2 model auto-discovery found no models at {models_url}")
    if len(model_ids) != 1:
        raise RuntimeError(
            "Stage 2 model auto-discovery requires exactly one served model; "
            f"{config.endpoint}/models advertised {model_ids}. Set stage2.model explicitly."
        )
    resolved = replace(config, model=model_ids[0])
    LOGGER.info(
        "auto-discovered Stage 2 model=%s from %s/models",
        resolved.model,
        config.endpoint,
    )
    return resolved


_DROP_KEYS = {
    "artifacts",
    "artifact_inventory",
    "common_vocabulary",
    "config",
    "fit_row_ids",
    "heldout_row_ids",
    "metrics",
    "model_diagnostics",
    "predictions",
    "run_config",
    "schema_version",
    "train_activations",
}


def _is_operational_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in _DROP_KEYS
        or lowered.endswith(("_sha256", "_hash", "_fingerprint", "_path"))
        or lowered.startswith(("authenticated_", "attestation_", "immutable_"))
    )


def _scientific_projection(value: Any) -> Any:
    """Remove old control-plane fields while retaining readable evidence."""

    if isinstance(value, Mapping):
        return {
            str(key): _scientific_projection(child)
            for key, child in value.items()
            if not _is_operational_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_scientific_projection(child) for child in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _infer_evidence_axes(value: Any) -> list[str]:
    """Infer statistical axes from ordinary Stage 1 field names and labels."""

    tokens: list[str] = []
    axis_value_keys = {
        "architecture",
        "axis",
        "axes",
        "bank",
        "contrast_family",
        "evidence_type",
        "json_path",
        "meaning",
        "mechanical_role",
        "objective",
        "observable_axes",
        "role",
        "signal",
        "target",
        "target_source",
    }

    def visit(child: Any) -> None:
        if isinstance(child, Mapping):
            for key, nested in child.items():
                lowered = str(key).lower()
                tokens.append(lowered)
                if lowered in axis_value_keys and isinstance(nested, (str, list, tuple)):
                    if isinstance(nested, str):
                        tokens.append(nested.lower())
                    else:
                        tokens.extend(str(item).lower() for item in nested)
                visit(nested)
        elif isinstance(child, (list, tuple)):
            for nested in child:
                visit(nested)

    visit(value)
    joined = " ".join(tokens)
    axes: set[str] = {"semantic"}
    if "treatment" in joined or "propensity" in joined:
        axes.add("treatment")
    if "outcome" in joined or "prognostic" in joined:
        axes.add("outcome")
    if any(
        token in joined
        for token in (
            "residual_effect",
            "residual effect",
            "pseudo_target",
            "r_loss",
            "r-loss",
            "heterogeneity",
            "interaction",
            "uplift",
            "effect_modifier",
            "effect modifier",
        )
    ) or any(token in {"effect", "effect_bank", "r"} for token in tokens):
        axes.add("residual_effect")
    if "matched_pair" in joined or "matched pair" in joined:
        axes.add("matched_pair")
    return sorted(axes)


def _row_sections(row: Mapping[str, Any]) -> list[tuple[str, Any, str]]:
    """Expose natural scientific sections without requiring one input schema."""

    source = str(row.get("source") or "unknown")
    payload = row.get("evidence")
    if not isinstance(payload, Mapping):
        return [(source, payload, "evidence")]
    architecture = payload.get("architecture")
    if isinstance(architecture, str) and architecture.strip():
        return [(architecture.strip(), payload, "evidence")]

    sections: list[tuple[str, Any, str]] = []
    if source == "text_models":
        importance = payload.get("importance")
        if importance:
            sections.append(("sparse_and_matched_pair_models", importance, "importance"))
        embedding = payload.get("embedding_contrast_evidence")
        if isinstance(embedding, Mapping) and embedding.get("contrasts"):
            for index, contrast in enumerate(embedding["contrasts"], start=1):
                sections.append(
                    (
                        "embedding_contrasts_and_retrieval_terms",
                        contrast,
                        f"embedding_contrast_evidence.contrasts[{index - 1}]",
                    )
                )
        elif embedding:
            sections.append(
                (
                    "embedding_contrasts_and_retrieval_terms",
                    embedding,
                    "embedding_contrast_evidence",
                )
            )
        htr = payload.get("htr_evidence")
        if isinstance(htr, Mapping):
            for key, value in htr.items():
                if value:
                    sections.append(("hierarchical_neural_text", value, f"htr_evidence.{key}"))
        elif htr:
            sections.append(("hierarchical_neural_text", htr, "htr_evidence"))
    elif source == "tfidf":
        discovery = payload.get("discovery")
        if isinstance(discovery, Mapping):
            topic_banks = discovery.get("topic_banks")
            if isinstance(topic_banks, Mapping):
                for bank, value in topic_banks.items():
                    if value:
                        sections.append(("tfidf_topics", value, f"discovery.topic_banks.{bank}"))
            score_tests = discovery.get("topic_score_tests")
            if isinstance(score_tests, Mapping) and score_tests.get("effect_orphan_ngram_branch"):
                sections.append(
                    (
                        "tfidf_orphan_ngrams",
                        score_tests["effect_orphan_ngram_branch"],
                        "discovery.topic_score_tests.effect_orphan_ngram_branch",
                    )
                )
            elif score_tests:
                sections.append(("tfidf_orphan_ngrams", score_tests, "discovery.topic_score_tests"))
    elif source == "neural_queries":
        query_evidence = payload.get("evidence")
        if isinstance(query_evidence, list):
            by_bank: dict[str, list[Any]] = defaultdict(list)
            for row_value in query_evidence:
                bank = (
                    str(row_value.get("bank") or "unspecified")
                    if isinstance(row_value, Mapping)
                    else "unspecified"
                )
                by_bank[bank].append(row_value)
            for bank, values in by_bank.items():
                sections.append(("neural_query_moments", values, f"evidence.{bank}"))
        elif query_evidence:
            sections.append(("neural_query_moments", query_evidence, "evidence"))
    return sections or [(source, payload, "evidence")]


def _json_chars(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), sort_keys=True))


def _split_value(
    value: Any,
    *,
    max_chars: int,
    path: str,
    path_budget: Callable[[str], int] | None = None,
) -> list[tuple[str, Any]]:
    def fragment_budget(fragment_path: str) -> int:
        if path_budget is None:
            return int(max_chars)
        return min(int(max_chars), int(path_budget(fragment_path)))

    def fragment_fits(fragment_path: str, fragment: Any) -> bool:
        return _json_chars(fragment) <= fragment_budget(fragment_path)

    if fragment_fits(path, value):
        return [(path, value)]
    if isinstance(value, Mapping):
        fragments: list[tuple[str, Any]] = []
        scalars: dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if isinstance(child, (Mapping, list, tuple)):
                fragments.extend(
                    _split_value(
                        child,
                        max_chars=max_chars,
                        path=child_path,
                        path_budget=path_budget,
                    )
                )
            else:
                scalars[str(key)] = child
        if scalars:
            if fragment_fits(path, scalars):
                fragments.insert(0, (path, scalars))
            else:
                for key, child in scalars.items():
                    fragments.extend(
                        _split_value(
                            child,
                            max_chars=max_chars,
                            path=f"{path}.{key}",
                            path_budget=path_budget,
                        )
                    )
        return fragments
    if isinstance(value, (list, tuple)):
        output: list[tuple[str, Any]] = []
        batch: list[Any] = []
        batch_chars = 2  # Opening and closing brackets in compact JSON.
        batch_start = 0
        for index, child in enumerate(value):
            child_chars = _json_chars(child)
            if batch:
                candidate_path = f"{path}[{batch_start}:{index + 1}]"
                candidate_chars = batch_chars + 1 + child_chars
                if candidate_chars <= fragment_budget(candidate_path):
                    batch.append(child)
                    batch_chars = candidate_chars
                    continue
                output.append((f"{path}[{batch_start}:{index}]", batch))
                batch = []
                batch_chars = 2
            singleton_path = f"{path}[{index}:{index + 1}]"
            singleton_chars = 2 + child_chars
            if singleton_chars <= fragment_budget(singleton_path):
                batch = [child]
                batch_chars = singleton_chars
                batch_start = index
            else:
                output.extend(
                    _split_value(
                        child,
                        max_chars=max_chars,
                        path=f"{path}[{index}]",
                        path_budget=path_budget,
                    )
                )
                batch_start = index + 1
        if batch:
            output.append((f"{path}[{batch_start}:{len(value)}]", batch))
        return output
    text = str(value)
    segments: list[tuple[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        segment_path = f"{path}.text_segment_{len(segments) + 1:03d}"
        low, high = cursor + 1, len(text)
        best = cursor
        while low <= high:
            end = (low + high) // 2
            if fragment_fits(segment_path, text[cursor:end]):
                best = end
                low = end + 1
            else:
                high = end - 1
        if best == cursor:
            raise ValueError(
                f"max_packet_chars cannot encode one source character at {segment_path}"
            )
        segments.append((segment_path, text[cursor:best]))
        cursor = best
    if "".join(str(segment) for _path, segment in segments) != text:
        raise RuntimeError("Stage 2 scalar packetization changed source text")
    return segments


def packetize_handoff(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_packet_chars: int,
) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        try:
            outer_fold = int(row["outer_fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"handoff row {row_index} has no integer outer_fold") from exc
        for section_index, (architecture, raw_section, section_path) in enumerate(
            _row_sections(row), start=1
        ):
            section = _scientific_projection(raw_section)
            prototype = {
                "packet_id": (
                    f"outer_{outer_fold:03d}_row_{row_index:04d}_"
                    f"section_{section_index:02d}_part_000000000000"
                ),
                "source": str(row.get("source") or "unknown"),
                "architecture": str(architecture),
                "outer_fold": outer_fold,
                "inner_fold": row.get("inner_fold"),
                "scope": str(row.get("scope") or "unspecified"),
                "json_path": section_path,
                "observable_axes": [
                    "matched_pair",
                    "outcome",
                    "residual_effect",
                    "semantic",
                    "treatment",
                    "unclear",
                ],
                "content": "",
            }

            def packet_content_budget(json_path: str) -> int:
                envelope = {**prototype, "json_path": json_path}
                envelope_chars = _json_chars(envelope) - _json_chars("")
                return int(max_packet_chars) - envelope_chars

            if packet_content_budget(section_path) < 1:
                raise ValueError("max_packet_chars is too small for the Stage 2 packet envelope")
            fragments = _split_value(
                section,
                max_chars=int(max_packet_chars),
                path=section_path,
                path_budget=packet_content_budget,
            )
            for fragment_index, (json_path, content) in enumerate(fragments, start=1):
                packet = {
                    "packet_id": (
                        f"outer_{outer_fold:03d}_row_{row_index:04d}_"
                        f"section_{section_index:02d}_part_{fragment_index:03d}"
                    ),
                    "source": str(row.get("source") or "unknown"),
                    "architecture": str(architecture),
                    "outer_fold": outer_fold,
                    "inner_fold": row.get("inner_fold"),
                    "scope": str(row.get("scope") or "unspecified"),
                    "json_path": json_path,
                    "observable_axes": _infer_evidence_axes(
                        {
                            "architecture": architecture,
                            "json_path": json_path,
                            "content": content,
                        }
                    ),
                    "content": content,
                }
                if _json_chars(packet) > int(max_packet_chars):
                    raise RuntimeError("Stage 2 packet planner emitted an oversized packet")
                packets.append(packet)
    if not packets:
        raise ValueError("the Stage 1 handoff contains no evidence packets")
    return packets


def _partition_packets(
    packets: Sequence[Mapping[str, Any]],
    *,
    max_chars: int,
) -> list[list[Mapping[str, Any]]]:
    batches: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for packet in packets:
        candidate = [*current, packet]
        if current and _json_chars(candidate) > max_chars:
            batches.append(current)
            current = []
        current.append(packet)
    if current:
        batches.append(current)
    return batches


CompletionFunction = Callable[[Sequence[Mapping[str, str]], PlainHandoffStage2Config], str]


class _RetryableStage2ResponseError(RuntimeError):
    """A completed transport that did not yield any response content."""


class _Stage2OutputLengthError(ValueError):
    """The server exhausted the available completion length."""


class _Stage2ResponseValidationError(ValueError):
    """A response remained structurally invalid after bounded repairs."""


def _openai_completion(
    messages: Sequence[Mapping[str, str]],
    config: PlainHandoffStage2Config,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        base_url=config.endpoint,
        api_key=config.api_key,
        timeout=config.request_timeout,
        # Stage 2 owns completion retries so they are logged, bounded, and do
        # not multiply invisibly with SDK-level retries.
        max_retries=0,
    )
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": list(messages),
        "temperature": config.temperature,
        "response_format": {"type": "json_object"},
    }
    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": config.enable_thinking}}
    prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
    if prompt_chars > int(config.max_prompt_chars):
        raise ValueError(
            "Stage 2 rendered prompt exceeds max_prompt_chars; the caller must "
            f"partition it losslessly before transport ({prompt_chars} > "
            f"{config.max_prompt_chars})"
        )
    LOGGER.info(
        "Stage 2 request endpoint=%s model=%s prompt_chars=%s",
        config.endpoint,
        config.model,
        prompt_chars,
    )
    try:
        response = client.chat.completions.create(**kwargs)
    finally:
        client.close()
    choice = response.choices[0]
    finish_reason = str(getattr(choice, "finish_reason", "") or "")
    if finish_reason == "length":
        raise _Stage2OutputLengthError(
            "Stage 2 server stopped the response with finish_reason=length"
        )
    content = choice.message.content
    if not content:
        raise _RetryableStage2ResponseError("Stage 2 model returned an empty response")
    return str(content)


def _is_retryable_transport_error(exc: Exception) -> bool:
    """Return whether a failed OpenAI-compatible request is safe to retry."""

    if isinstance(exc, _RetryableStage2ResponseError):
        return True
    try:
        from openai import APIConnectionError, APIStatusError
    except ImportError:  # pragma: no cover - OpenAI is required for live requests
        return False
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        status_code = int(exc.status_code)
        return status_code in {408, 409, 429} or status_code >= 500
    return False


def _completion_with_transport_retries(
    messages: Sequence[Mapping[str, str]],
    config: PlainHandoffStage2Config,
    completion: CompletionFunction,
) -> str:
    max_attempts = max(1, int(config.transport_max_attempts))
    for attempt in range(1, max_attempts + 1):
        try:
            return completion(messages, config)
        except Exception as exc:
            if not _is_retryable_transport_error(exc) or attempt == max_attempts:
                raise
            delay = float(config.transport_retry_backoff) * (2 ** (attempt - 1))
            LOGGER.warning(
                "Stage 2 transport failed; retrying request attempt %s/%s " "after %.1fs (%s: %s)",
                attempt + 1,
                max_attempts,
                delay,
                type(exc).__name__,
                exc,
            )
            if delay > 0:
                time.sleep(delay)
    raise RuntimeError("unreachable Stage 2 transport retry state")


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Stage 2 response must be one JSON object")
    return value


def _compact_json_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Losslessly reclaim prompt space from JSON message bodies for repairs."""

    compacted: list[dict[str, Any]] = []
    for message in messages:
        row = dict(message)
        content = row.get("content")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                rendered = json.dumps(
                    parsed,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            else:
                if len(rendered) < len(content):
                    row["content"] = rendered
        compacted.append(row)
    return compacted


def _repair_message(exc: Exception) -> dict[str, str]:
    if isinstance(exc, _Stage2OutputLengthError):
        content = (
            "The previous JSON exceeded the available response length. Return one materially "
            "shorter corrected JSON object using the same required schema. Remove redundancy, "
            "merge duplicate entries, and keep descriptions and rationales concise. Do not omit "
            "required records or fields. Return JSON only."
        )
    else:
        content = (
            "The previous JSON failed validation. Correct this exact error: "
            f"{type(exc).__name__}: {exc}. Return one corrected JSON object only."
        )
    return {
        "role": "user",
        "content": content,
    }


def _bounded_repair_directive(exc: Exception, *, max_chars: int) -> str:
    """Keep the validation failure visible even in a packed repair prompt."""

    if max_chars < 16:
        raise ValueError("Stage 2 repair prompt has no room for its validation error") from exc
    message = f"Fix this validation error: {type(exc).__name__}: {exc}. JSON only."
    if len(message) <= max_chars:
        return message
    detail = f"{type(exc).__name__}: {exc}"
    if len(detail) <= max_chars:
        return detail
    return detail[: max_chars - 3].rstrip() + "..."


def _request_json(
    *,
    messages: Sequence[Mapping[str, str]],
    config: PlainHandoffStage2Config,
    completion: CompletionFunction,
    validate: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    base_conversation = [dict(message) for message in messages]
    conversation = [dict(message) for message in base_conversation]
    first_error: Exception | None = None
    max_attempts = 1 + MAX_RESPONSE_REPAIRS
    for attempt in range(max_attempts):
        response: str | None = None
        prompt_chars = sum(len(str(message.get("content") or "")) for message in conversation)
        if prompt_chars > int(config.max_prompt_chars):
            raise ValueError(
                "Stage 2 rendered prompt exceeds max_prompt_chars before transport "
                f"({prompt_chars} > {config.max_prompt_chars})"
            )
        try:
            response = _completion_with_transport_retries(
                conversation,
                config,
                completion,
            )
            return validate(_parse_json_object(response))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if attempt == max_attempts - 1:
                raise _Stage2ResponseValidationError(
                    f"Stage 2 response remained invalid after {max_attempts - 1} repairs: {exc}"
                ) from exc
            first_error = exc
            LOGGER.warning(
                "Stage 2 response failed validation; repair attempt %s/%s (%s: %s)",
                attempt + 1,
                max_attempts - 1,
                type(exc).__name__,
                exc,
            )
            repair_message = _repair_message(exc)
            repair_context = [dict(message) for message in base_conversation]
            if response is not None:
                repair_context.append({"role": "assistant", "content": str(response)})
            for candidate_context in (repair_context, base_conversation):
                repaired = [*candidate_context, repair_message]
                repaired_chars = sum(len(str(message.get("content") or "")) for message in repaired)
                if repaired_chars <= int(config.max_prompt_chars):
                    conversation = repaired
                    break
            else:
                repaired = []
            if repaired:
                continue

            # A fully packed initial prompt may leave no room for another turn.
            # Minify JSON bodies without changing their content, then retry with
            # the same explicit validation error.
            compact_context = _compact_json_messages(repair_context)
            compact_repaired = [*compact_context, repair_message]
            if sum(len(str(message.get("content") or "")) for message in compact_repaired) <= int(
                config.max_prompt_chars
            ):
                conversation = compact_repaired
                continue
            compact_base = _compact_json_messages(base_conversation)
            compact_repaired = [*compact_base, repair_message]
            if sum(len(str(message.get("content") or "")) for message in compact_repaired) <= int(
                config.max_prompt_chars
            ):
                conversation = compact_repaired
                continue

            # If lossless compaction is still insufficient, spend the system
            # message's full character budget on the concrete validation error.
            # The complete original user payload remains present.
            system_index = next(
                (
                    index
                    for index, message in enumerate(compact_base)
                    if str(message.get("role") or "") == "system"
                ),
                None,
            )
            if system_index is None:
                raise ValueError(
                    "Stage 2 repair prompt cannot fit max_prompt_chars and has no "
                    "system instruction available to replace"
                ) from exc
            conversation = [dict(message) for message in compact_base]
            non_system_chars = sum(
                len(str(message.get("content") or ""))
                for index, message in enumerate(conversation)
                if index != system_index
            )
            available = int(config.max_prompt_chars) - non_system_chars
            conversation[system_index]["content"] = _bounded_repair_directive(
                exc,
                max_chars=available,
            )
    raise RuntimeError(f"unreachable Stage 2 response state: {first_error}")


def _checkpointed_request_json(
    *,
    output_dir: Path | None,
    input_value: Mapping[str, Any],
    messages: Sequence[Mapping[str, str]],
    config: PlainHandoffStage2Config,
    completion: CompletionFunction,
    validate: Callable[[Mapping[str, Any]], dict[str, Any]],
    validation_fallback: (
        Callable[[_Stage2ResponseValidationError], Mapping[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
    """Cache one validated LLM leaf by its complete deterministic input."""

    def request_with_fallback() -> tuple[dict[str, Any], _Stage2ResponseValidationError | None]:
        try:
            return (
                _request_json(
                    messages=messages,
                    config=config,
                    completion=completion,
                    validate=validate,
                ),
                None,
            )
        except _Stage2ResponseValidationError as exc:
            if validation_fallback is None:
                raise
            result = validate(validation_fallback(exc))
            LOGGER.warning(
                "Stage 2 response remained invalid; using conservative validated fallback (%s)",
                exc,
            )
            return result, exc

    if output_dir is None:
        result, _fallback_error = request_with_fallback()
        return result
    output_dir = Path(output_dir)
    checkpoint_input = {
        "consolidation_schema": CONSOLIDATION_SCHEMA_VERSION,
        **dict(input_value),
    }
    input_fingerprint = _value_fingerprint(checkpoint_input)
    input_path = output_dir / "input.json"
    result_path = output_dir / "result.json"
    complete_path = output_dir / "complete.json"
    if input_path.is_file() and result_path.is_file() and complete_path.is_file():
        try:
            previous_input = json.loads(input_path.read_text(encoding="utf-8"))
            completion_state = json.loads(complete_path.read_text(encoding="utf-8"))
            cached_result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                previous_input.get("input_fingerprint") == input_fingerprint
                and completion_state.get("input_fingerprint") == input_fingerprint
            ):
                validated = validate(cached_result)
                LOGGER.info("skip completed Stage 2 consolidation request: %s", output_dir)
                return validated
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        LOGGER.info("rerun stale or inconsistent Stage 2 consolidation request: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        input_path,
        {
            **checkpoint_input,
            "input_fingerprint": input_fingerprint,
        },
    )
    result, fallback_error = request_with_fallback()
    _write_json(result_path, result)
    if fallback_error is not None:
        _write_json(
            output_dir / "fallback.json",
            {
                "status": "conservative_validation_fallback",
                "completed_at": _now(),
                "validation_error": str(fallback_error),
            },
        )
    _write_json(
        complete_path,
        {
            "status": (
                "complete_with_validation_fallback"
                if fallback_error is not None
                else "complete"
            ),
            "completed_at": _now(),
            "input_fingerprint": input_fingerprint,
            "validation_fallback": fallback_error is not None,
        },
    )
    return result


def _interpretation_response_contract() -> dict[str, Any]:
    """Return the shared response contract for interpretation passes."""

    return {
        "candidates": [
            {
                "name": "snake_case_clinical_feature_name",
                "description": (
                    "exactly one underlying or explicitly documented patient-level clinical "
                    "feature that could produce the observed text evidence pattern"
                ),
                "supporting_items": [1],
                "evidence_rationale": (
                    "how the cited words, phrases, or clinical context could arise from this "
                    "feature, including whether the feature is explicit or inferred"
                ),
                "caveats": "limitations, ambiguity, or competing clinical explanations",
            }
        ],
    }


def _interpretation_evidence_items(
    packets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose only readable text under prompt-local ordinal labels."""

    items: list[dict[str, Any]] = []
    for item_number, packet in enumerate(packets, start=1):
        texts = _readable_supporting_text([packet])
        if not texts:
            raise ValueError(f"interpretation evidence item {item_number} has no readable text")
        items.append({"item": item_number, "text": texts})
    return items


def _interpretation_prompt(
    *,
    architecture: str,
    packets: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    del architecture
    body = {
        "job": "infer_clinical_features_from_text_evidence",
        "task": (
            "Identify patient-level clinical features supported by the supplied text. Some "
            "items may support multiple features or no valid feature."
        ),
        "rules": [
            "Each candidate must represent one patient-level clinical variable with one value per patient. It must be assignable by examining one patient's record without comparing or aggregating across patients.",
            "Return explicitly documented clinical features and narrower latent clinical features reasonably implied by the text.",
            "Return distinct measurements or attributes as separate candidates. Do not combine them into a profile, burden, syndrome, or general patient state.",
            "Use longitudinal information as clinical context when it appears in the evidence. Do not perform temporal eligibility filtering.",
            "Do not return patient names, administrative identifiers, documentation artifacts, descriptions of the input collection, multiple-patient heterogeneity, grouping methods, or analysis methods as clinical features.",
            "If no valid feature is supported, return an empty candidates list. Never turn the absence of a common feature into a candidate.",
            "Every returned candidate must have a nonempty snake_case name. Omit any candidate you cannot name; never return a blank or null name.",
            "For each candidate, cite one or more supplied item numbers in supporting_items and explain how its text supports the feature.",
            "Do not choose a value type, unit, categories, or extraction ontology in this step.",
        ],
        "evidence_items": _interpretation_evidence_items(packets),
        "response": _interpretation_response_contract(),
    }
    return [
        {
            "role": "system",
            "content": (
                "Identify patient-level clinical features supported by the supplied text. "
                "Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _rejected_packet_audit_prompt(
    *,
    architecture: str,
    packets: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build a recall-oriented second pass over initially rejected packets."""

    del architecture
    body = {
        "job": "audit_unmapped_text_evidence_for_missed_clinical_features",
        "task": (
            "The supplied text was not cited by an initial review. Re-examine every item for "
            "patient-level clinical features that may have been missed."
        ),
        "rules": [
            "Review every evidence item independently. One clear item is sufficient to support a candidate; a clue does not need to recur.",
            "Each candidate must represent one patient-level clinical variable with one value per patient. It must be assignable by examining one patient's record without comparing or aggregating across patients.",
            "Return explicitly documented clinical features and narrower latent clinical features reasonably implied by the text.",
            "Return distinct measurements or attributes as separate candidates. Do not combine them into a profile, burden, syndrome, or general patient state.",
            "Use longitudinal information as clinical context when it appears in the evidence. Do not perform temporal eligibility filtering.",
            "Do not return patient names, administrative identifiers, documentation artifacts, descriptions of the input collection, multiple-patient heterogeneity, grouping methods, or analysis methods as clinical features.",
            "If no valid feature is supported, return an empty candidates list. Never turn the absence of a common feature into a candidate.",
            "Every returned candidate must have a nonempty snake_case name. Omit any candidate you cannot name; never return a blank or null name.",
            "For each candidate, cite one or more supplied item numbers in supporting_items and explain how its text supports the feature.",
            "Do not choose a value type, unit, categories, or extraction ontology in this step.",
        ],
        "evidence_items": _interpretation_evidence_items(packets),
        "response": _interpretation_response_contract(),
    }
    return [
        {
            "role": "system",
            "content": (
                "Re-examine the supplied text for missed patient-level clinical features. "
                "Favor recall among valid individual-patient variables, but return no "
                "candidate for input or analysis artifacts. Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _validate_interpretation(
    value: Mapping[str, Any],
    *,
    packet_ids: Sequence[str],
    packet_evidence_axes: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    ordered_packet_ids = list(map(str, packet_ids))
    if len(ordered_packet_ids) != len(set(ordered_packet_ids)):
        raise ValueError("interpretation input contains duplicate packet IDs")
    packet_id_by_item = {
        item_number: packet_id for item_number, packet_id in enumerate(ordered_packet_ids, start=1)
    }
    payload = value
    if not isinstance(payload.get("concepts"), list):
        for key in ("result", "response", "interpretation"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    concepts = next(
        (
            payload.get(key)
            for key in ("concepts", "features", "variables", "candidates")
            if isinstance(payload.get(key), list)
        ),
        None,
    )
    if not isinstance(concepts, list):
        raise ValueError("interpretation requires a concepts list")
    clean_concepts: list[dict[str, Any]] = []
    for concept_index, concept in enumerate(concepts, start=1):
        if not isinstance(concept, Mapping):
            raise ValueError("each interpreted concept must be an object")
        name = str(concept.get("name") or concept.get("feature_name") or "").strip()
        if not name:
            LOGGER.warning(
                "Stage 2 interpretation ignored unnamed candidate at position=%s; "
                "its citations were not used",
                concept_index,
            )
            continue
        raw_supports = concept.get("supporting_items") or []
        if isinstance(raw_supports, (str, int)):
            raw_supports = [raw_supports]
        elif not isinstance(raw_supports, Sequence):
            raw_supports = []
        cited_items: list[int] = []
        invalid_items: list[Any] = []
        for raw_item in raw_supports:
            if isinstance(raw_item, bool):
                invalid_items.append(raw_item)
                continue
            try:
                item_number = int(raw_item)
            except (TypeError, ValueError):
                invalid_items.append(raw_item)
                continue
            if str(raw_item).strip() not in {str(item_number), f"{item_number}.0"}:
                invalid_items.append(raw_item)
                continue
            if item_number not in packet_id_by_item:
                invalid_items.append(raw_item)
                continue
            cited_items.append(item_number)
        cited_items = list(dict.fromkeys(cited_items))
        supports = [packet_id_by_item[item_number] for item_number in cited_items]
        if invalid_items:
            LOGGER.warning(
                "Stage 2 interpretation concept=%s ignored %s invalid supporting item(s): %s",
                name,
                len(invalid_items),
                invalid_items[:8],
            )
        if not supports:
            LOGGER.warning(
                "Stage 2 interpretation dropped ungrounded concept=%s; no supplied "
                "evidence item cited it",
                name,
            )
            continue
        if packet_evidence_axes is not None:
            axes = sorted(
                {
                    axis
                    for packet_id in supports
                    for axis in _canonical_evidence_axes(packet_evidence_axes.get(packet_id))
                }
            )
        else:
            axes = []
        evidence_rationale = str(
            concept.get("evidence_rationale")
            or concept.get("pattern_rationale")
            or concept.get("rationale")
            or ""
        ).strip()
        if not evidence_rationale:
            raise ValueError(
                f"interpreted candidate {name!r} has no evidence_rationale explaining "
                "how the cited text evidence could arise from it"
            )
        clean_concepts.append(
            {
                "name": name,
                "description": str(concept.get("description") or name),
                "supporting_packet_ids": supports,
                "evidence_axes": axes,
                "evidence_rationale": evidence_rationale,
                "caveats": str(concept.get("caveats") or ""),
            }
        )
    clean_dispositions: dict[str, Any] = {}
    for packet_id in ordered_packet_ids:
        names = sorted(
            concept["name"]
            for concept in clean_concepts
            if packet_id in concept["supporting_packet_ids"]
        )
        clean_dispositions[packet_id] = {
            "status": "supports_concept" if names else "reviewed_no_specific_concept",
            "concept_names": names,
            "reason": (
                "Derived from the candidates' evidence-item citations."
                if names
                else "No returned candidate cited this evidence item."
            ),
        }
    return {"concepts": clean_concepts, "packet_dispositions": clean_dispositions}


def _cached_interpretation_matches_packets(
    value: Any,
    *,
    packet_ids: set[str],
) -> bool:
    """Return whether a normalized checkpoint cites exactly its current inputs."""

    if not isinstance(value, Mapping):
        return False
    concepts = value.get("concepts")
    dispositions = value.get("packet_dispositions")
    if not isinstance(concepts, list) or not isinstance(dispositions, Mapping):
        return False
    if {str(packet_id) for packet_id in dispositions} != packet_ids:
        return False
    for concept in concepts:
        if not isinstance(concept, Mapping):
            return False
        supports = concept.get("supporting_packet_ids")
        if not isinstance(supports, list) or not supports:
            return False
        if not {str(packet_id) for packet_id in supports} <= packet_ids:
            return False
        if not str(concept.get("evidence_rationale") or "").strip():
            return False
    return True


def _partition_packets_for_prompt(
    packets: Sequence[Mapping[str, Any]],
    *,
    render_prompt: Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, str]]],
    max_prompt_chars: int,
) -> list[list[Mapping[str, Any]]]:
    """Pack evidence using the exact rendered prompt supplied by the caller."""

    batches: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for packet in packets:
        candidate = [*current, packet]
        messages = render_prompt(candidate)
        prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
        if not current and prompt_chars > int(max_prompt_chars):
            raise ValueError("one Stage 2 evidence packet cannot fit the rendered prompt budget")
        if current and prompt_chars > int(max_prompt_chars):
            batches.append(current)
            current = [packet]
            singleton = render_prompt(current)
            if sum(len(message["content"]) for message in singleton) > int(max_prompt_chars):
                raise ValueError(
                    "one Stage 2 evidence packet cannot fit the rendered prompt budget"
                )
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _partition_interpretation_packets(
    packets: Sequence[Mapping[str, Any]],
    *,
    architecture: str,
    max_prompt_chars: int,
) -> list[list[Mapping[str, Any]]]:
    """Pack evidence using the exact fully rendered interpretation prompt."""

    return _partition_packets_for_prompt(
        packets,
        render_prompt=lambda batch: _interpretation_prompt(
            architecture=architecture,
            packets=batch,
        ),
        max_prompt_chars=max_prompt_chars,
    )


def _partition_rejected_packet_audit(
    packets: Sequence[Mapping[str, Any]],
    *,
    architecture: str,
    max_prompt_chars: int,
) -> list[list[Mapping[str, Any]]]:
    """Pack rejected packets using the exact rendered audit prompt."""

    return _partition_packets_for_prompt(
        packets,
        render_prompt=lambda batch: _rejected_packet_audit_prompt(
            architecture=architecture,
            packets=batch,
        ),
        max_prompt_chars=max_prompt_chars,
    )


def _merge_interpretation_audit(
    *,
    packet_ids: set[str],
    initial: Mapping[str, Any],
    audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge grounded audit recoveries into an initial interpretation result."""

    concepts = [
        dict(concept)
        for result in [initial, *audits]
        for concept in list(result.get("concepts") or [])
    ]
    initial_dispositions = dict(initial.get("packet_dispositions") or {})
    latest_dispositions = dict(initial_dispositions)
    for audit in audits:
        latest_dispositions.update(dict(audit.get("packet_dispositions") or {}))

    dispositions: dict[str, Any] = {}
    for packet_id in sorted(packet_ids):
        names = sorted(
            str(concept["name"])
            for concept in concepts
            if packet_id in set(map(str, concept.get("supporting_packet_ids") or []))
        )
        source = latest_dispositions.get(packet_id)
        reason = str(source.get("reason") or "") if isinstance(source, Mapping) else ""
        dispositions[packet_id] = {
            "status": "supports_concept" if names else "reviewed_no_specific_concept",
            "concept_names": names,
            "reason": reason
            or (
                "Recovered by the rejected-packet audit."
                if names
                else "No interpretation pass recovered a defensible clinical feature."
            ),
        }

    initially_rejected = {
        str(packet_id)
        for packet_id, disposition in initial_dispositions.items()
        if isinstance(disposition, Mapping)
        and disposition.get("status") == "reviewed_no_specific_concept"
    }
    recovered = sorted(
        packet_id
        for packet_id in initially_rejected
        if dispositions.get(packet_id, {}).get("status") == "supports_concept"
    )
    remaining = sorted(initially_rejected - set(recovered))
    return {
        "concepts": concepts,
        "packet_dispositions": dispositions,
        "rejected_packet_audit": {
            "schema_version": INTERPRETATION_AUDIT_SCHEMA_VERSION,
            "initially_rejected_packet_ids": sorted(initially_rejected),
            "recovered_packet_ids": recovered,
            "remaining_rejected_packet_ids": remaining,
            "audit_batches": len(audits),
        },
    }


def _consolidation_prompt(
    *,
    clinical_question: str,
    outer_fold: int,
    candidates: Sequence[Mapping[str, Any]],
    max_candidates: int,
) -> list[dict[str, str]]:
    del clinical_question, outer_fold, candidates, max_candidates
    raise RuntimeError(
        "the monolithic Stage 2 consolidation prompt is retired; use candidate-ID "
        "alias grouping and per-group operationalization"
    )


def _validate_consolidation(
    value: Mapping[str, Any],
    *,
    candidates: Sequence[Mapping[str, Any]],
    max_candidates: int,
) -> dict[str, Any]:
    # Retained only for validating historical monolithic-consolidation
    # responses. The former feature-count limit is intentionally ignored.
    del max_candidates
    features = value.get("features")
    dispositions = value.get("candidate_dispositions")
    if not isinstance(features, list):
        raise ValueError("consolidation requires a features list")
    if not isinstance(dispositions, Mapping):
        raise ValueError(
            "consolidation requires candidate_dispositions for every supplied candidate"
        )
    dispositions = {str(candidate_id): row for candidate_id, row in dispositions.items()}

    def feature_name_key(name: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

    def string_list(raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, (str, int, float, bool)):
            return [str(raw)]
        if not isinstance(raw, Sequence):
            return []
        return list(dict.fromkeys(str(item) for item in raw if item is not None))

    candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    candidate_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    missing_disposition_ids = sorted(candidate_ids - set(dispositions))
    if missing_disposition_ids:
        raise ValueError(
            "consolidation omitted candidate disposition(s): " f"{missing_disposition_ids[:8]}"
        )
    extra_disposition_ids = sorted(set(dispositions) - candidate_ids)
    if extra_disposition_ids:
        LOGGER.warning(
            "Stage 2 consolidation ignored %s unknown candidate disposition(s): %s",
            len(extra_disposition_ids),
            extra_disposition_ids[:8],
        )
    status_aliases = {
        "keep": "retained",
        "kept": "retained",
        "retain": "retained",
        "combine": "merged",
        "combined": "merged",
        "merge": "merged",
        "drop": "excluded",
        "dropped": "excluded",
        "exclude": "excluded",
    }
    normalized_dispositions: dict[str, dict[str, str]] = {}
    for candidate_id in sorted(candidate_ids):
        raw_disposition = dispositions[candidate_id]
        if not isinstance(raw_disposition, Mapping):
            raise ValueError(f"candidate disposition {candidate_id!r} must be an object")
        status = str(raw_disposition.get("status") or "").strip().lower()
        status = status_aliases.get(status, status)
        if status not in {"retained", "merged", "excluded"}:
            raise ValueError(
                f"candidate disposition {candidate_id!r} has unsupported status {status!r}"
            )
        feature_name = str(raw_disposition.get("feature_name") or "").strip()
        if status != "excluded" and not feature_name:
            raise ValueError(
                f"candidate disposition {candidate_id!r} with status={status!r} "
                "must name a returned feature"
            )
        normalized_dispositions[candidate_id] = {
            "status": status,
            "feature_name": feature_name,
            "reason": str(raw_disposition.get("reason") or "").strip(),
        }
    allowed_packets = {
        str(packet_id)
        for candidate in candidates
        for packet_id in string_list(candidate.get("supporting_packet_ids"))
    }
    allowed_architectures = {
        str(architecture)
        for candidate in candidates
        for architecture in [
            candidate["architecture"],
            *string_list(candidate.get("supporting_architectures")),
        ]
    }
    packet_axes: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        for packet_id in string_list(candidate.get("supporting_packet_ids")):
            per_packet = candidate.get("packet_evidence_axes") or {}
            packet_axes[str(packet_id)].update(
                str(axis)
                for axis in per_packet.get(
                    str(packet_id),
                    candidate["evidence_axes"],
                )
            )
    clean_features: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(features, start=1):
        if not isinstance(feature, Mapping):
            raise ValueError(f"consolidation feature at position={feature_index} must be an object")
        name = str(feature.get("name") or feature.get("feature_name") or "").strip()
        if not name:
            raise ValueError(f"consolidation feature at position={feature_index} has no name")
        name_key = feature_name_key(name)
        matched_candidate_ids = {
            candidate_id
            for candidate_id, disposition in normalized_dispositions.items()
            if disposition["status"] != "excluded"
            and feature_name_key(disposition["feature_name"]) == name_key
        }
        if not matched_candidate_ids:
            raise ValueError(
                f"returned feature {name!r} is not referenced by any retained or merged "
                "candidate disposition"
            )
        incompatible_candidate_ids = [
            candidate_id
            for candidate_id in sorted(matched_candidate_ids)
            if not _consolidation_route_is_semantically_compatible(
                candidate_by_id[candidate_id], feature
            )
        ]
        if incompatible_candidate_ids:
            incompatible_names = [
                str(candidate_by_id[candidate_id].get("name") or candidate_id)
                for candidate_id in incompatible_candidate_ids
            ]
            raise ValueError(
                f"returned feature {name!r} has semantically incompatible candidate "
                f"route(s): {incompatible_names[:8]}. Distinct measurements must not "
                "be merged merely because packet evidence overlaps"
            )

        cited_packets = string_list(
            feature.get("supporting_packet_ids") or feature.get("packet_ids")
        )
        unknown_packets = [
            packet_id for packet_id in cited_packets if packet_id not in allowed_packets
        ]
        if unknown_packets:
            raise ValueError(
                f"returned feature {name!r} cites unknown packet ID(s): " f"{unknown_packets[:8]}"
            )
        routed_packets = {
            str(packet_id)
            for candidate_id in matched_candidate_ids
            for packet_id in string_list(candidate_by_id[candidate_id].get("supporting_packet_ids"))
        }
        unrelated_packets = sorted(set(cited_packets) - routed_packets)
        if unrelated_packets:
            LOGGER.warning(
                "Stage 2 consolidation feature=%s discarded %s known packet citation(s) "
                "not carried by candidates routed to that feature: %s",
                name,
                len(unrelated_packets),
                unrelated_packets[:8],
            )
        packets = [packet_id for packet_id in cited_packets if packet_id in routed_packets]
        for candidate_id in sorted(matched_candidate_ids):
            packets.extend(string_list(candidate_by_id[candidate_id].get("supporting_packet_ids")))
        packets = list(
            dict.fromkeys(packet_id for packet_id in packets if packet_id in allowed_packets)
        )
        if not packets:
            raise ValueError(f"returned feature {name!r} has no supplied candidate evidence")

        raw_categories = feature.get("categories_or_unit")
        if isinstance(raw_categories, Mapping):
            raw_categories = (
                raw_categories.get("categories")
                or raw_categories.get("values")
                or raw_categories.get("unit")
            )
        if raw_categories is None:
            raw_categories = feature.get("categories") or feature.get("unit")
        categories = string_list(raw_categories)

        value_type = str(feature.get("value_type") or "ambiguous").strip().lower()
        value_type = {
            "bool": "binary",
            "boolean": "binary",
            "category": "categorical",
            "numeric": "continuous",
            "number": "continuous",
            "unknown": "ambiguous",
        }.get(value_type, value_type)
        if value_type not in ALLOWED_VALUE_TYPES:
            value_type = "ambiguous"
        if value_type in {"binary", "categorical", "ordinal"}:
            # Models sometimes serialize an enumeration as one delimited string
            # or an ordinal ontology as an integer range such as ``0-4``.
            from .plain_handoff_stage2_analysis import _validated_closed_category_values

            categories = _validated_closed_category_values(
                value_type=value_type,
                values=categories,
                source=f"returned feature {name!r}",
            )

        architectures = [
            architecture
            for architecture in string_list(feature.get("supporting_architectures"))
            if architecture in allowed_architectures
        ]
        if not architectures:
            architectures = list(
                dict.fromkeys(
                    str(architecture)
                    for candidate_id, candidate in candidate_by_id.items()
                    if candidate_id in matched_candidate_ids
                    or set(string_list(candidate.get("supporting_packet_ids"))).intersection(
                        packets
                    )
                    for architecture in [
                        candidate["architecture"],
                        *string_list(candidate.get("supporting_architectures")),
                    ]
                    if str(architecture) in allowed_architectures
                )
            )

        description = str(feature.get("description") or name).strip()
        clean_features.append(
            {
                "name": name,
                "description": description,
                "value_type": value_type,
                "categories_or_unit": categories,
                "roles": [
                    role for role in string_list(feature.get("roles")) if role in ALLOWED_ROLES
                ],
                "measurement_definition": str(
                    feature.get("measurement_definition") or description
                ).strip(),
                "missing_value_rule": str(
                    feature.get("missing_value_rule")
                    or "Return null when not documented in the pretreatment record."
                ).strip(),
                "supporting_packet_ids": packets,
                "supporting_architectures": architectures,
                "stability_summary": str(feature.get("stability_summary") or ""),
                "caveats": str(feature.get("caveats") or ""),
            }
        )
    deduplicated_features: dict[str, dict[str, Any]] = {}
    for feature in clean_features:
        key = feature_name_key(feature["name"])
        existing = deduplicated_features.get(key)
        if existing is None:
            deduplicated_features[key] = feature
            continue
        raise ValueError(f"consolidation returned duplicate feature name {feature['name']!r}")
    clean_features = list(deduplicated_features.values())
    clean_dispositions: dict[str, dict[str, str]] = {}
    features_by_name = {feature["name"]: feature for feature in clean_features}
    features_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in clean_features:
        key = feature_name_key(feature["name"])
        features_by_key[key].append(feature)
    for candidate_id in sorted(candidate_ids):
        raw_disposition = normalized_dispositions[candidate_id]
        candidate_packets = {
            str(packet_id)
            for packet_id in string_list(candidate_by_id[candidate_id].get("supporting_packet_ids"))
        }
        status = raw_disposition["status"]
        feature_name = raw_disposition["feature_name"]
        reason = raw_disposition["reason"]
        if status == "excluded":
            clean_dispositions[candidate_id] = {
                "status": "excluded",
                "feature_name": "",
                "reason": reason or "Candidate was excluded by consolidation.",
            }
            continue
        feature = features_by_name.get(feature_name)
        if feature is None and feature_name:
            key = feature_name_key(feature_name)
            matches = features_by_key.get(key, [])
            if len(matches) == 1:
                feature = matches[0]
        if feature is None:
            raise ValueError(
                f"candidate disposition {candidate_id!r} references missing returned "
                f"feature {feature_name!r}"
            )
        if not _consolidation_route_is_semantically_compatible(
            candidate_by_id[candidate_id], feature
        ):
            raise ValueError(
                f"candidate {candidate_id!r} "
                f"({candidate_by_id[candidate_id].get('name')!r}) cannot be merged "
                f"into semantically incompatible feature {feature['name']!r}"
            )
        if not candidate_packets <= set(feature["supporting_packet_ids"]):
            raise ValueError(
                f"returned feature {feature['name']!r} did not preserve all packet "
                f"evidence for candidate {candidate_id!r}"
            )
        clean_dispositions[candidate_id] = {
            "status": status,
            "feature_name": str(feature["name"]),
            "reason": reason or "Candidate was reconciled to the returned grounded feature.",
        }

    routed_features: list[dict[str, Any]] = []
    for feature in clean_features:
        axes = {
            axis
            for packet_id in feature["supporting_packet_ids"]
            for axis in packet_axes.get(packet_id, set())
        }
        derived_roles: list[str] = []
        if {"treatment", "outcome"} <= axes:
            derived_roles.append("confounder")
        elif "outcome" in axes:
            derived_roles.append("prognostic")
        if axes.intersection({"residual_effect", "matched_pair"}):
            derived_roles.append("effect_modifier")
        if not derived_roles:
            raise ValueError(
                f"returned feature {feature['name']!r} has no supported Stage 2 causal role"
            )
        feature["roles"] = derived_roles
        routed_features.append(feature)

    routed_names = {feature["name"] for feature in routed_features}
    for disposition in clean_dispositions.values():
        if disposition["status"] != "excluded" and disposition["feature_name"] not in routed_names:
            raise ValueError(
                "candidate disposition references unrouted feature "
                f"{disposition['feature_name']!r}"
            )
    used_names = {
        disposition["feature_name"]
        for disposition in clean_dispositions.values()
        if disposition["status"] != "excluded"
    }
    unused_features = [
        feature["name"] for feature in routed_features if feature["name"] not in used_names
    ]
    if unused_features:
        raise ValueError(
            "returned feature(s) have no retained or merged candidate route: "
            f"{unused_features[:8]}"
        )
    return {"features": routed_features, "candidate_dispositions": clean_dispositions}


def _string_values(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, int, float, bool)):
        return [str(raw)]
    if not isinstance(raw, Sequence):
        return []
    return list(dict.fromkeys(str(item) for item in raw if item is not None))


def _short_text(value: Any, *, max_chars: int) -> str:
    rendered = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 3].rstrip() + "..."


def _snake_case_name(value: Any, *, fallback: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return name or fallback


def _candidate_architectures(candidate: Mapping[str, Any]) -> list[str]:
    primary = str(candidate.get("architecture") or "").strip()
    inherited = _string_values(candidate.get("supporting_architectures"))
    if primary in {
        "deterministic_candidate_group",
        "bounded_multi_architecture_consolidation",
    }:
        primary = ""
    if primary == CONFIGURED_EXPLICIT_FEATURE_ARCHITECTURE:
        primary = ""
    return list(
        dict.fromkeys(
            architecture
            for architecture in [
                primary,
                *inherited,
            ]
            if architecture
        )
    )


def _configured_feature_definitions(
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw = candidate.get("configured_feature_definitions") or []
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    definitions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, Mapping):
            continue
        definition = dict(value)
        name = _snake_case_name(definition.get("name"), fallback="")
        if not name or name in seen:
            continue
        definition["name"] = name
        definitions.append(definition)
        seen.add(name)
    return definitions


def _group_roles(group: Mapping[str, Any]) -> list[str]:
    configured = _configured_feature_definitions(group)
    if configured:
        if len(configured) != 1:
            raise ValueError(
                "Stage 2 consolidation attempted to merge multiple investigator-configured "
                f"features: {[feature['name'] for feature in configured]}"
            )
        return [
            role for role in _string_values(configured[0].get("roles")) if role in ALLOWED_ROLES
        ]
    return _derive_roles(group.get("evidence_axes") or [])


def _filter_candidate_groups_by_causal_role(
    groups: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Apply the auditable causal-role filter after lossless alias consolidation."""

    retained: list[dict[str, Any]] = []
    exclusions: dict[str, str] = {}
    decisions: list[dict[str, Any]] = []
    exclusion_reason = (
        "Excluded because its Stage 1 evidence does not support a Stage 2 "
        "confounder, prognostic, or effect-modifier role."
    )
    for group in groups:
        roles = _group_roles(group)
        origin_candidate_ids = _string_values(group.get("origin_candidate_ids"))
        if roles:
            retained.append(dict(group))
            decisions.append(
                {
                    "name": str(group["name"]),
                    "status": "retained",
                    "roles": roles,
                    "origin_candidate_ids": origin_candidate_ids,
                }
            )
            continue
        for origin in origin_candidate_ids:
            exclusions[origin] = exclusion_reason
        decisions.append(
            {
                "name": str(group["name"]),
                "status": "excluded",
                "roles": [],
                "origin_candidate_ids": origin_candidate_ids,
                "reason": exclusion_reason,
            }
        )
    return retained, exclusions, decisions


def _flatten_member_measurements(
    members: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Preserve original candidate views through repeated consolidation rounds."""

    flattened: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for member in members:
        inherited = member.get("member_measurements")
        raw_views = (
            list(inherited)
            if isinstance(inherited, Sequence)
            and not isinstance(inherited, (str, bytes, bytearray))
            and inherited
            else [member]
        )
        for raw_view in raw_views:
            if not isinstance(raw_view, Mapping):
                continue
            view = {
                "name": str(raw_view.get("name") or "").strip(),
                "description": _short_text(raw_view.get("description"), max_chars=500),
                "evidence_rationale": _short_text(
                    raw_view.get("evidence_rationale"),
                    max_chars=700,
                ),
                "value_type": str(raw_view.get("value_type") or "ambiguous"),
            }
            identity = (
                view["name"],
                view["description"],
                view["evidence_rationale"],
                view["value_type"],
            )
            if identity in seen:
                continue
            flattened.append(view)
            seen.add(identity)
    return flattened


def _materialize_candidate_group(
    *,
    candidate_id: str,
    members: Sequence[Mapping[str, Any]],
    canonical_name: str,
    canonical_description: str,
    ontology_packet_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    packets = list(
        dict.fromkeys(
            packet_id
            for member in members
            for packet_id in _string_values(member.get("supporting_packet_ids"))
        )
    )
    architectures = list(
        dict.fromkeys(
            architecture for member in members for architecture in _candidate_architectures(member)
        )
    )
    evidence_axes = sorted(
        {
            axis
            for member in members
            for axis in _canonical_evidence_axes(member.get("evidence_axes"))
        }
    )
    packet_evidence_axes: dict[str, list[str]] = {}
    for member in members:
        inherited = member.get("packet_evidence_axes") or {}
        member_axes = _canonical_evidence_axes(member.get("evidence_axes"))
        for packet_id in _string_values(member.get("supporting_packet_ids")):
            packet_evidence_axes[packet_id] = sorted(
                {
                    *packet_evidence_axes.get(packet_id, []),
                    *_canonical_evidence_axes(inherited.get(packet_id) or member_axes),
                }
            )
    origins = list(
        dict.fromkeys(
            origin
            for member in members
            for origin in (
                _string_values(member.get("origin_candidate_ids")) or [str(member["candidate_id"])]
            )
        )
    )
    value_types = list(
        dict.fromkeys(str(member.get("value_type") or "ambiguous") for member in members)
    )
    descriptions = [
        str(member.get("description") or "").strip()
        for member in members
        if str(member.get("description") or "").strip()
    ]
    caveats = list(
        dict.fromkeys(
            str(member.get("caveats") or "").strip()
            for member in members
            if str(member.get("caveats") or "").strip()
        )
    )
    configured_definitions: list[dict[str, Any]] = []
    configured_names: set[str] = set()
    for member in members:
        for definition in _configured_feature_definitions(member):
            configured_name = str(definition["name"])
            if configured_name in configured_names:
                continue
            configured_definitions.append(definition)
            configured_names.add(configured_name)
    description = canonical_description or (descriptions[0] if descriptions else canonical_name)
    if ontology_packet_ids is None:
        ontology_packet_ids = [
            packet_id
            for member in members
            for packet_id in (
                _string_values(member.get("ontology_packet_ids"))
                or _string_values(member.get("supporting_packet_ids"))[:1]
            )
        ]
    return {
        "candidate_id": candidate_id,
        "architecture": "deterministic_candidate_group",
        "supporting_architectures": architectures,
        "name": canonical_name,
        "description": description,
        "value_type": value_types[0] if len(value_types) == 1 else "ambiguous",
        "supporting_packet_ids": packets,
        "evidence_axes": evidence_axes,
        "packet_evidence_axes": packet_evidence_axes,
        "caveats": " ".join(caveats),
        "origin_candidate_ids": origins,
        "configured_feature_definitions": configured_definitions,
        # Internal routing only. Python resolves these to readable supporting
        # text; the ontology model never receives packet structure or IDs.
        "ontology_packet_ids": list(dict.fromkeys(map(str, ontology_packet_ids))),
        "member_measurements": _flatten_member_measurements(members),
    }


def _canonical_cluster_member(
    members: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    name_counts = Counter(_snake_case_name(member.get("name"), fallback="") for member in members)
    return max(
        enumerate(members),
        key=lambda item: (
            bool(_configured_feature_definitions(item[1])),
            name_counts[_snake_case_name(item[1].get("name"), fallback="")],
            len(_string_values(item[1].get("origin_candidate_ids"))),
            len(_string_values(item[1].get("supporting_packet_ids"))),
            bool(str(item[1].get("description") or "").strip()),
            -item[0],
        ),
    )[1]


def _materialize_exact_name_groups(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Coalesce exact normalized names before iterative semantic consolidation.

    This is identity bookkeeping, not semantic consolidation: distinct names
    are never compared or merged here. Combining exact names keeps every batch
    response contract unambiguous while preserving candidate evidence and
    provenance.
    """

    members_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for index, candidate in enumerate(candidates, start=1):
        name = _snake_case_name(candidate.get("name"), fallback=f"measurement_{index:03d}")
        members_by_name.setdefault(name, []).append(candidate)

    materialized: list[dict[str, Any]] = []
    for group_index, (name, members) in enumerate(members_by_name.items(), start=1):
        canonical = _canonical_cluster_member(members)
        materialized.append(
            _materialize_candidate_group(
                candidate_id=f"candidate_pool_group_{group_index:04d}",
                members=members,
                canonical_name=name,
                canonical_description=_short_text(
                    canonical.get("description") or name,
                    max_chars=2_000,
                ),
                ontology_packet_ids=_string_values(canonical.get("supporting_packet_ids"))[:1],
            )
        )
    return materialized


def _candidate_group_sort_key(group: Mapping[str, Any]) -> tuple[str, str, str]:
    name = str(group.get("name") or "")
    return (
        _snake_case_name(name, fallback=""),
        name.casefold(),
        str(group.get("candidate_id") or ""),
    )


def _alphabetical_candidate_batches(
    groups: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    round_number: int,
) -> tuple[int, list[list[dict[str, Any]]]]:
    """Sort groups and shift nonoverlapping batch boundaries between rounds."""

    if batch_size < 2:
        raise ValueError("candidate consolidation batch_size must be at least 2")
    if round_number < 1:
        raise ValueError("candidate consolidation round_number must be positive")
    ordered = [dict(group) for group in sorted(groups, key=_candidate_group_sort_key)]
    if not ordered:
        return 0, []
    if len(ordered) <= batch_size:
        return 0, [ordered]

    shift_step = max(1, batch_size // 2)
    while math.gcd(shift_step, batch_size) != 1:
        shift_step += 1
    boundary_offset = ((round_number - 1) * shift_step) % batch_size
    batches: list[list[dict[str, Any]]] = []
    cursor = 0
    if boundary_offset:
        batches.append(ordered[:boundary_offset])
        cursor = boundary_offset
    while cursor < len(ordered):
        batches.append(ordered[cursor : cursor + batch_size])
        cursor += batch_size
    return boundary_offset, batches


def _seeded_shuffle_candidate_batches(
    groups: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    seed: int,
    shuffle_round: int,
) -> list[list[dict[str, Any]]]:
    """Deterministically shuffle a sorted pool before forming bounded batches."""

    if batch_size < 2:
        raise ValueError("candidate consolidation batch_size must be at least 2")
    if shuffle_round < 1:
        raise ValueError("candidate consolidation shuffle_round must be positive")
    ordered = [dict(group) for group in sorted(groups, key=_candidate_group_sort_key)]

    def shuffled_key(group: Mapping[str, Any]) -> tuple[str, tuple[str, str, str]]:
        identity = "\0".join(
            (
                str(int(seed)),
                str(int(shuffle_round)),
                str(group.get("name") or ""),
                str(group.get("candidate_id") or ""),
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest(), _candidate_group_sort_key(
            group
        )

    shuffled = sorted(ordered, key=shuffled_key)
    return [shuffled[start : start + batch_size] for start in range(0, len(shuffled), batch_size)]


def _candidate_consolidation_batches(
    groups: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    round_number: int,
    alphabetical_rounds: int,
    seed: int,
) -> tuple[str, int | None, int | None, list[list[dict[str, Any]]]]:
    """Use shifted alphabetical partitions, then seeded shuffled partitions."""

    if alphabetical_rounds < 0:
        raise ValueError("candidate consolidation alphabetical_rounds must be nonnegative")
    if round_number <= alphabetical_rounds:
        boundary_offset, batches = _alphabetical_candidate_batches(
            groups,
            batch_size=batch_size,
            round_number=round_number,
        )
        return "alphabetical_shift", boundary_offset, None, batches
    shuffle_round = round_number - alphabetical_rounds
    return (
        "seeded_shuffle",
        None,
        shuffle_round,
        _seeded_shuffle_candidate_batches(
            groups,
            batch_size=batch_size,
            seed=seed,
            shuffle_round=shuffle_round,
        ),
    )


def _coalesce_exact_candidate_group_names(
    groups: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Coalesce identical canonical outputs produced by independent batches."""

    members_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for index, group in enumerate(groups, start=1):
        name = _snake_case_name(group.get("name"), fallback=f"measurement_{index:03d}")
        members_by_name.setdefault(name, []).append(group)

    coalesced: list[dict[str, Any]] = []
    exact_merges = 0
    for name, members in members_by_name.items():
        if len(members) == 1:
            coalesced.append(dict(members[0]))
            continue
        exact_merges += len(members) - 1
        canonical = _canonical_cluster_member(members)
        coalesced.append(
            _materialize_candidate_group(
                candidate_id=str(members[0]["candidate_id"]),
                members=members,
                canonical_name=name,
                canonical_description=_short_text(
                    canonical.get("description") or name,
                    max_chars=2_000,
                ),
            )
        )
    return sorted(coalesced, key=_candidate_group_sort_key), exact_merges


def _candidate_pool_feature_view(group: Mapping[str, Any]) -> dict[str, Any]:
    """Expose all distinct candidate descriptions without internal provenance."""

    descriptions = list(
        dict.fromkeys(
            description
            for description in [
                _short_text(group.get("description"), max_chars=400),
                *(
                    _short_text(member.get("description"), max_chars=400)
                    for member in list(group.get("member_measurements") or [])
                ),
            ]
            if description
        )
    )
    return {
        "name": str(group.get("name") or ""),
        "descriptions": descriptions,
    }


def _derive_roles(evidence_axes: Sequence[str]) -> list[str]:
    axes = set(_canonical_evidence_axes(evidence_axes))
    roles: list[str] = []
    if {"treatment", "outcome"} <= axes:
        roles.append("confounder")
    elif "outcome" in axes:
        roles.append("prognostic")
    if axes.intersection({"residual_effect", "matched_pair"}):
        roles.append("effect_modifier")
    return roles


def _global_candidate_pool_prompt(
    *,
    groups: Sequence[Mapping[str, Any]],
    configured_feature_names: Sequence[str] = (),
    batch_ordering: str = "alphabetical_shift",
) -> list[dict[str, str]]:
    """Consolidate aliases in one bounded candidate-pool batch without filtering."""

    features = [
        _candidate_pool_feature_view(group)
        for group in sorted(groups, key=_candidate_group_sort_key)
    ]

    body: dict[str, Any] = {
        "job": "consolidate_stage2_candidate_pool",
        "task": (
            "Review this "
            + (
                "alphabetically adjacent"
                if batch_ordering == "alphabetical_shift"
                else "deterministically shuffled"
            )
            + " batch of interpreted candidate features. "
            "Partition semantic aliases and equivalent representations of each underlying "
            "patient-level measurement within the supplied batch. Every supplied feature "
            "must survive this pass either unchanged or as an input to exactly one merge. "
            "Later rounds will use new deterministic partitions of the consolidated candidates."
        ),
        "features": features,
        "rules": [
            "Every name absent from merge_directives will be retained unchanged; do not restate unchanged features.",
            "This is merge-only ontology consolidation, not feature filtering or quality review. Never exclude or drop a supplied feature.",
            "Each merge directive must contain at least two exact names from features.",
            "Treat merge_directives as a disjoint partition of alias families within this batch, not as sequential rename operations: return exactly one directive for each complete supplied alias family and never chain or split one family across directives.",
            "Each directive's inputs must list every exact supplied feature name in that alias family within this batch, including the selected canonical name when output reuses a supplied feature name.",
            "An output that equals a supplied feature name is valid only when that exact name appears in the same directive's inputs; it must not be an input of another directive or an unchanged feature.",
            "Use each feature name at most once across all merge inputs.",
            "Merge spelling variants, abbreviations, synonymous clinical names, and all clearly equivalent representations of the same underlying measurement.",
            "A general measurement name, its quantitative score, a thresholded or coarsened status, a named category, and a name containing one observed value belong together when they can all be represented by one underlying patient variable.",
            "Prefer an information-preserving underlying measurement name over a threshold, category, or observed value encoded in one candidate name.",
            "When a value-encoded or awkward alias has a clear underlying measurement in this batch, merge it into that measurement; otherwise retain it unchanged.",
            "Judge alias families jointly across this entire batch; do not require a direct lexical match between every pair of members in one family.",
            "Do not merge merely related but independently varying variables, a diagnosis with a related laboratory value, a broad concept with one independently varying component, different anatomical sites, different biomarkers, or different timepoints.",
            "The output must be one concise snake_case canonical name for the consolidated measurement. It may reuse the best input name or provide a clearer equivalent name.",
            "When semantic equivalence is uncertain, do not merge the features.",
            "Return only exact supplied feature names in merge inputs. Return no internal IDs, provenance, definitions, explanations, unchanged feature names, or exclusion list.",
        ],
        "response": {
            "merge_directives": [
                {
                    "inputs": [
                        "all exact supplied names in one alias family, including a reused output name"
                    ],
                    "output": "one snake_case canonical feature name",
                }
            ]
        },
    }
    configured_names = list(map(str, configured_feature_names))
    if configured_names:
        body["configured_feature_names"] = configured_names
        body["rules"].extend(
            [
                "Never merge two names listed in configured_feature_names; the investigator specified them as distinct features.",
                "When one merge input is listed in configured_feature_names, output that exact configured name so its investigator-supplied ontology remains authoritative.",
            ]
        )
    return [
        {
            "role": "system",
            "content": "Consolidate aliases without filtering any features. Return JSON only.",
        },
        {
            "role": "user",
            "content": json.dumps(body, sort_keys=True, ensure_ascii=False),
        },
    ]


def _validate_global_candidate_pool_directives(
    value: Mapping[str, Any],
    *,
    group_names: Sequence[str],
    configured_feature_names: Sequence[str] = (),
    group_descriptions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Validate directives using only names and descriptions supplied in the batch."""

    available = [str(name) for name in group_names]
    if len(available) != len(set(available)):
        raise ValueError("global candidate pool requires unique supplied feature names")
    available_set = set(available)
    exact_alias_owners: dict[str, set[str]] = defaultdict(set)
    normalized_alias_owners: dict[str, set[str]] = defaultdict(set)

    def register_alias(alias: Any, owner: str) -> None:
        rendered = str(alias or "").strip()
        if not rendered:
            return
        exact_alias_owners[rendered.casefold()].add(owner)
        normalized = _snake_case_name(rendered, fallback="")
        if normalized:
            normalized_alias_owners[normalized].add(owner)

    supplied_descriptions = group_descriptions or {}
    unknown_description_names = sorted(set(map(str, supplied_descriptions)) - available_set)
    if unknown_description_names:
        raise ValueError(
            "global candidate pool received descriptions for unknown feature names: "
            f"{unknown_description_names}"
        )
    for name in available:
        register_alias(name, name)
        for description in supplied_descriptions.get(name, ()):
            register_alias(description, name)
    configured_names = set(map(str, configured_feature_names))
    unknown_configured_names = sorted(configured_names - available_set)
    if unknown_configured_names:
        raise ValueError(
            "global candidate pool received unknown configured feature names: "
            f"{unknown_configured_names}"
        )

    def resolve_input_name(raw_name: Any) -> str:
        rendered = str(raw_name or "").strip()
        if rendered in available_set:
            return rendered
        matches = exact_alias_owners.get(rendered.casefold(), set())
        if len(matches) == 1:
            return next(iter(matches))
        normalized = _snake_case_name(rendered, fallback="")
        matches = normalized_alias_owners.get(normalized, set())
        if len(matches) == 1:
            return next(iter(matches))
        raise ValueError(f"global candidate pool named unknown or ambiguous feature {rendered!r}")

    def resolve_output_name(raw_name: Any) -> tuple[str, str | None]:
        rendered = str(raw_name or "").strip()
        if not rendered:
            return "", None
        try:
            known_name = resolve_input_name(rendered)
        except ValueError:
            return _snake_case_name(rendered, fallback=""), None
        return known_name, known_name

    payload = value
    if not isinstance(payload.get("merge_directives"), list):
        for key in ("result", "response", "consolidation"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if "exclude_feature_names" in payload:
        raise ValueError(
            "iterative candidate consolidation is merge-only; omit exclude_feature_names"
        )
    raw_directives = payload.get("merge_directives")
    if raw_directives is None:
        raw_directives = payload.get("merges")
    if not isinstance(raw_directives, list):
        raise ValueError("global candidate pool requires a merge_directives array")

    directives: list[dict[str, Any]] = []
    used_inputs: set[str] = set()
    input_directive_by_name: dict[str, int] = {}
    for index, raw_directive in enumerate(raw_directives, start=1):
        if not isinstance(raw_directive, Mapping):
            raise ValueError(f"global merge directive {index} must be an object")
        raw_inputs = raw_directive.get("inputs")
        if not isinstance(raw_inputs, list):
            raise ValueError(f"global merge directive {index} requires an inputs array")
        inputs = list(dict.fromkeys(resolve_input_name(name) for name in raw_inputs))
        output, known_output_name = resolve_output_name(raw_directive.get("output"))
        if not output:
            raise ValueError(f"global merge directive {index} requires an output name")
        # Models sometimes omit a reused canonical feature from ``inputs`` even
        # though they select it as ``output``. The intended complete family is
        # unambiguous when that output resolves to one supplied feature.
        if known_output_name is not None and known_output_name not in inputs:
            inputs.append(known_output_name)
        if len(inputs) < 2:
            # A name and its own supplied prose description can be emitted as
            # two apparent aliases. Once both resolve to the same feature this
            # is a harmless no-op, not a reason to reject the whole batch.
            continue
        repeated = sorted(set(inputs).intersection(used_inputs))
        if repeated:
            owners = {name: input_directive_by_name[name] for name in repeated}
            raise ValueError(
                "global merge input names may appear in only one directive; "
                f"directive {index} repeats inputs already used by earlier directives: "
                f"{dict(list(owners.items())[:8])}. Combine the complete alias family "
                "into one directive instead of chaining or splitting directives"
            )
        configured_inputs = [name for name in inputs if name in configured_names]
        if len(configured_inputs) > 1:
            raise ValueError(
                "global merge directives must not combine distinct investigator-configured "
                f"features: {configured_inputs}"
            )
        if configured_inputs and output != configured_inputs[0]:
            raise ValueError(
                "a global merge containing an investigator-configured feature must use "
                f"that exact configured name as output: {configured_inputs[0]!r}"
            )
        for name in inputs:
            input_directive_by_name[name] = index
        used_inputs.update(inputs)
        directives.append({"inputs": inputs, "output": output})

    output_directive_by_name: dict[str, int] = {}
    pass_through_names = set(available) - used_inputs
    for index, directive in enumerate(directives, start=1):
        output = str(directive["output"])
        previous_output_index = output_directive_by_name.get(output)
        if previous_output_index is not None:
            raise ValueError(
                f"global merge directive {index} duplicates output name {output!r} from "
                f"directive {previous_output_index}; each directive requires a unique output"
            )
        own_inputs = set(map(str, directive["inputs"]))
        if output in available and output not in own_inputs:
            input_owner = input_directive_by_name.get(output)
            if input_owner is not None:
                raise ValueError(
                    f"global merge directive {index} output name {output!r} is an input "
                    f"of global merge directive {input_owner}; do not chain directives. "
                    "Combine the complete alias family into one directive, or choose an "
                    "output name that is not a supplied feature"
                )
            if output in pass_through_names:
                raise ValueError(
                    f"global merge directive {index} output name {output!r} names an "
                    "unchanged supplied feature; include it in this directive's inputs, "
                    "or choose an output name that is not a supplied feature"
                )
            raise RuntimeError(  # pragma: no cover - exhaustive partition invariant
                f"unclassified global merge output collision for {output!r}"
            )
        output_directive_by_name[output] = index
    return {"merge_directives": directives}


def _apply_global_candidate_pool_directives(
    groups: Sequence[Mapping[str, Any]],
    directives: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply name directives while passing every unmentioned group through unchanged."""

    group_by_name = {str(group["name"]): group for group in groups}
    position_by_name = {str(group["name"]): index for index, group in enumerate(groups)}
    directive_by_first_position: dict[int, Mapping[str, Any]] = {}
    consumed: set[str] = set()
    for directive in directives:
        inputs = [str(name) for name in directive["inputs"]]
        missing = [name for name in inputs if name not in group_by_name]
        if missing:  # pragma: no cover - validator invariant
            raise ValueError(f"global merge directive contains unknown input(s): {missing[:8]}")
        first_position = min(position_by_name[name] for name in inputs)
        directive_by_first_position[first_position] = directive
        consumed.update(inputs)

    merged: list[dict[str, Any]] = []
    for position, raw_group in enumerate(groups):
        group_name = str(raw_group["name"])
        position_directive = directive_by_first_position.get(position)
        if position_directive is not None:
            inputs = [str(name) for name in position_directive["inputs"]]
            members = [group_by_name[name] for name in inputs]
            output_name = str(position_directive["output"])
            canonical = next(
                (member for member in members if str(member["name"]) == output_name),
                _canonical_cluster_member(members),
            )
            first = groups[position]
            merged.append(
                _materialize_candidate_group(
                    candidate_id=str(first["candidate_id"]),
                    members=members,
                    canonical_name=output_name,
                    canonical_description=_short_text(
                        canonical.get("description") or output_name,
                        max_chars=2_000,
                    ),
                )
            )
            continue
        if group_name not in consumed:
            merged.append(dict(raw_group))
    return merged


def _operationalization_prompt(
    *,
    feature_name: str,
    supporting_evidence: Sequence[str],
) -> list[dict[str, str]]:
    evidence = list(
        dict.fromkeys(str(item).strip() for item in supporting_evidence if str(item).strip())
    )
    if not evidence:
        raise ValueError("operationalization requires readable supporting evidence")
    instructions = {
        "task": (
            "Define the extraction ontology for the named candidate clinical feature. "
            "Decide its value type, allowed values or unit, and measurement rule from the "
            "candidate name and readable supporting clinical evidence."
        ),
        "rules": [
            "Define exactly the named scalar pretreatment measurement; do not rename, merge, or split it in this step.",
            "Determine value_type yourself from what the named feature means and how it is represented in the supplied evidence. No value type from an earlier discovery step is being provided.",
            "The evidence may contain unrelated clues; use only text that actually bears on the named feature.",
            "Specify a reproducible extraction target from a complete patient record.",
            "Do not invent an ad hoc score, formula, or index to force multiple distinct measurements into one scalar.",
            "Prefer value_type continuous, with a clinically meaningful unit when applicable, when the named feature can realistically be extracted as a numeric measurement. Use categorical or ordinal only when continuous measurement is infeasible or would misrepresent the feature.",
            "For categorical or ordinal variables, enumerate the extraction ontology; for continuous variables, provide the unit when applicable.",
            "For a binary variable, categories_or_unit must contain exactly two distinct extractable scalar values as separate array items.",
            "For a categorical or ordinal variable, categories_or_unit must contain at least two distinct extractable scalar values as separate array items.",
            "List each category exactly once. Categories that differ only by capitalization, punctuation, underscores, or spacing are duplicates and must not both appear.",
            "Never use a type label such as binary or categorical, or a combined phrase such as present-or-absent, as one ontology value.",
            "Define how absent, ambiguous, and conflicting documentation is represented.",
            "Return one flat JSON object with every response field shown below; measurement_definition and missing_value_rule are required nonempty strings.",
        ],
        "response": {
            "description": "one patient-level scalar measurement",
            "value_type": "binary|categorical|continuous|ordinal|ambiguous",
            "categories_or_unit": ["categories or one unit string"],
            "measurement_definition": "what to extract from the pretreatment record",
            "missing_value_rule": "how missing or ambiguous documentation is represented",
            "stability_summary": "scientific support summary without provenance identifiers",
            "caveats": "remaining scientific limitations",
        },
    }
    body = {
        "candidate_feature_name": str(feature_name),
        "supporting_evidence": evidence,
    }
    return [
        {
            "role": "system",
            "content": json.dumps(
                {
                    "instruction": (
                        "Define one clinical feature ontology from its name and supporting "
                        "clinical text. Return JSON only."
                    ),
                    **instructions,
                },
                sort_keys=True,
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _readable_supporting_text(
    packets: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Project compiled evidence packets to unique readable text only."""

    texts: list[str] = []
    for packet in packets:
        content = packet.get("content")
        if not isinstance(content, Mapping):
            continue
        representative_evidence = content.get("representative_evidence") or []
        if isinstance(representative_evidence, (str, Mapping)):
            representative_evidence = [representative_evidence]
        if not isinstance(representative_evidence, Sequence):
            continue
        for item in representative_evidence:
            raw_text = item.get("text") if isinstance(item, Mapping) else item
            text = str(raw_text or "").strip()
            if text:
                texts.append(text)
    return list(dict.fromkeys(texts))


def _pack_operationalization_supporting_evidence(
    *,
    feature_name: str,
    supporting_evidence: Sequence[str],
    max_prompt_chars: int,
) -> tuple[list[str], dict[str, Any]]:
    """Pack whole evidence excerpts under a prompt limit with repair headroom."""

    evidence = list(
        dict.fromkeys(str(item).strip() for item in supporting_evidence if str(item).strip())
    )
    if not evidence:
        raise ValueError("operationalization requires readable supporting evidence")
    prompt_limit = int(max_prompt_chars)
    repair_headroom = min(16_000, max(512, prompt_limit // 20))
    initial_prompt_budget = prompt_limit - repair_headroom

    # The system message is independent of the evidence values. Compute the
    # exact JSON-list contribution without repeatedly rendering a growing body.
    template = _operationalization_prompt(
        feature_name=feature_name,
        supporting_evidence=[evidence[0]],
    )
    system_chars = len(template[0]["content"])
    empty_body_chars = len(
        json.dumps(
            {
                "candidate_feature_name": str(feature_name),
                "supporting_evidence": [],
            },
            sort_keys=True,
        )
    )
    fixed_chars_without_list = system_chars + empty_body_chars - 2
    list_chars = 2
    packed: list[str] = []
    for text_value in evidence:
        separator_chars = 2 if packed else 0
        candidate_list_chars = (
            list_chars + separator_chars + len(json.dumps(text_value))
        )
        if fixed_chars_without_list + candidate_list_chars > initial_prompt_budget:
            continue
        packed.append(text_value)
        list_chars = candidate_list_chars

    truncated_items = 0
    if not packed:
        available_encoded_chars = initial_prompt_budget - fixed_chars_without_list - 2
        first = evidence[0]
        best = ""
        low, high = 1, len(first)
        while low <= high:
            midpoint = (low + high) // 2
            candidate = first[:midpoint].rstrip()
            if midpoint < len(first):
                candidate = candidate.rstrip(" .") + "..."
            if candidate and len(json.dumps(candidate)) <= available_encoded_chars:
                best = candidate
                low = midpoint + 1
            else:
                high = midpoint - 1
        if not best:
            raise ValueError(
                "stage2.operationalization_max_prompt_chars is too small for the "
                "operationalization instructions and one evidence excerpt"
            )
        packed = [best]
        truncated_items = 1

    messages = _operationalization_prompt(
        feature_name=feature_name,
        supporting_evidence=packed,
    )
    prompt_chars = sum(len(message["content"]) for message in messages)
    if prompt_chars > initial_prompt_budget:  # pragma: no cover - exact accounting invariant
        raise RuntimeError(
            "operationalization evidence packing exceeded its calculated prompt budget"
        )
    metadata = {
        "available_evidence_items": len(evidence),
        "included_evidence_items": len(packed),
        "omitted_evidence_items": len(evidence) - len(packed),
        "truncated_evidence_items": truncated_items,
        "available_evidence_chars": sum(len(item) for item in evidence),
        "included_evidence_chars": sum(len(item) for item in packed),
        "available_evidence_fingerprint": _value_fingerprint(evidence),
        "prompt_chars": prompt_chars,
        "initial_prompt_budget_chars": initial_prompt_budget,
        "repair_headroom_chars": repair_headroom,
        "request_prompt_limit_chars": prompt_limit,
    }
    return packed, metadata


def _ambiguous_operationalization_fallback(
    *,
    group: Mapping[str, Any],
    validation_error: _Stage2ResponseValidationError,
) -> dict[str, Any]:
    """Return a conservative extraction-ready ontology after exhausted repairs."""

    name = str(group.get("name") or "measurement")
    readable_name = name.replace("_", " ")
    description = str(group.get("description") or readable_name).strip()
    existing_caveats = str(group.get("caveats") or "").strip()
    fallback_caveat = (
        "The ontology response remained structurally invalid after bounded repairs; "
        "the value type is conservatively marked ambiguous for training-fold extraction "
        "and review."
    )
    return {
        "description": description,
        "value_type": "ambiguous",
        "categories_or_unit": [],
        "measurement_definition": (
            f"Extract one explicitly documented pretreatment scalar for {readable_name}; "
            "preserve the documented scalar representation without inventing categories."
        ),
        "missing_value_rule": (
            "Return null when the pretreatment record does not explicitly document one "
            "unambiguous scalar value."
        ),
        "stability_summary": "Model-authored ontology required a conservative fallback.",
        "caveats": " ".join(value for value in (existing_caveats, fallback_caveat) if value),
        "validation_fallback_error": str(validation_error),
    }


def _validate_operationalization(
    value: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
) -> dict[str, Any]:
    # Some instruction-following models wrap the requested fields in a named
    # object or helpfully repeat the feature name/provenance. Those extras are
    # harmless here: Python never reads them when assembling the final feature.
    # Prefer the first recognized nested object while preserving usable scalar
    # fields returned at the top level.
    normalized = dict(value)
    for key in ("operationalization", "feature", "definition", "variable", "result"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            normalized.update(nested)
            break
    raw_features = value.get("features")
    if isinstance(raw_features, Sequence) and not isinstance(raw_features, str):
        if len(raw_features) == 1 and isinstance(raw_features[0], Mapping):
            normalized.update(raw_features[0])

    raw_categories = normalized.get("categories_or_unit")
    if isinstance(raw_categories, Mapping):
        raw_categories = (
            raw_categories.get("categories")
            or raw_categories.get("values")
            or raw_categories.get("unit")
        )
    if raw_categories is None:
        raw_categories = (
            normalized.get("categories")
            or normalized.get("allowed_values")
            or normalized.get("levels")
            or normalized.get("unit")
        )
    categories = _string_values(raw_categories)
    raw_value_type = (
        normalized.get("value_type") or normalized.get("data_type") or normalized.get("type")
    )
    if not str(raw_value_type or "").strip():
        raise ValueError(
            "operationalization requires the model to choose value_type from "
            "binary, categorical, continuous, ordinal, or ambiguous"
        )
    value_type = str(raw_value_type)
    value_type = value_type.strip().lower()
    value_type = {
        "bool": "binary",
        "boolean": "binary",
        "category": "categorical",
        "numeric": "continuous",
        "number": "continuous",
        "unknown": "ambiguous",
    }.get(value_type, value_type)
    if value_type not in ALLOWED_VALUE_TYPES:
        raise ValueError(
            "operationalization value_type must be binary, categorical, continuous, "
            f"ordinal, or ambiguous; received {value_type!r}"
        )
    if value_type in {"binary", "categorical", "ordinal"}:
        from .plain_handoff_stage2_analysis import _validated_closed_category_values

        categories = _validated_closed_category_values(
            value_type=value_type,
            values=categories,
            source="operationalization",
        )
    description = str(
        normalized.get("description")
        or normalized.get("clinical_definition")
        or normalized.get("summary")
        or group.get("description")
        or ""
    ).strip()
    measurement_definition = str(
        normalized.get("measurement_definition")
        or normalized.get("operational_definition")
        or normalized.get("extraction_definition")
        or normalized.get("extraction_instruction")
        or normalized.get("measurement_rule")
        or normalized.get("how_to_measure")
        or (
            normalized.get("definition")
            if not isinstance(normalized.get("definition"), Mapping)
            else ""
        )
        or ""
    ).strip()
    if not measurement_definition:
        canonical_name = str(group.get("name") or "measurement").replace("_", " ")
        canonical_description = description or canonical_name
        measurement_definition = (
            f"Extract one pretreatment scalar for {canonical_name} according to this "
            f"canonical definition: {canonical_description}"
        )
    missing_value_rule = str(
        normalized.get("missing_value_rule")
        or normalized.get("missingness_rule")
        or normalized.get("missing_data_rule")
        or normalized.get("missing_value_handling")
        or ""
    ).strip()
    if not missing_value_rule:
        missing_value_rule = (
            "Return null when the pretreatment record does not explicitly document "
            "a single unambiguous value."
        )
    architecture_count = len(_candidate_architectures(group))
    packet_count = len(_string_values(group.get("supporting_packet_ids")))
    stability_summary = str(
        normalized.get("stability_summary")
        or normalized.get("support_summary")
        or normalized.get("evidence_summary")
        or ""
    ).strip()
    if not stability_summary:
        stability_summary = (
            f"Supported by {packet_count} evidence packet(s) across "
            f"{architecture_count} Stage 1 architecture(s)."
        )
    return {
        "description": description or str(group.get("name") or ""),
        "value_type": value_type,
        "categories_or_unit": categories,
        "measurement_definition": measurement_definition,
        "missing_value_rule": missing_value_rule,
        "stability_summary": stability_summary,
        "caveats": str(
            normalized.get("caveats") or normalized.get("limitations") or group.get("caveats") or ""
        ).strip(),
    }


def _load_stage2_splits(
    *,
    provenance_path: Path | None,
    dataset_rows: int,
    outer_fold_ids: Sequence[int],
    inner_folds: int,
    seed: int,
) -> dict[int, dict[str, Any]]:
    """Read the ordinary Stage 1 split file, with a deterministic fallback."""

    rows: list[dict[str, Any]] = []
    if provenance_path is not None and Path(provenance_path).is_file():
        rows = [
            json.loads(line)
            for line in Path(provenance_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not rows:
        from sklearn.model_selection import KFold

        fold_ids = sorted(set(map(int, outer_fold_ids)))
        if len(fold_ids) < 2:
            raise FileNotFoundError(
                "full Stage 2 needs components/tfidf/split_provenance.jsonl when "
                "the handoff contains fewer than two outer folds"
            )
        splitter = KFold(n_splits=len(fold_ids), shuffle=True, random_state=int(seed))
        all_rows = np.arange(dataset_rows, dtype=int)
        for outer_fold, (fit, heldout) in zip(fold_ids, splitter.split(all_rows)):
            fit_ids = all_rows[fit]
            inner_count = min(max(2, int(inner_folds)), len(fit_ids))
            inner_rows: list[dict[str, Any]] = []
            if inner_count >= 2:
                inner_splitter = KFold(
                    n_splits=inner_count,
                    shuffle=True,
                    random_state=int(seed) + 51_000 + int(outer_fold),
                )
                for inner_fold, (inner_fit, inner_heldout) in enumerate(
                    inner_splitter.split(fit_ids), start=1
                ):
                    inner_rows.append(
                        {
                            "inner_fold": inner_fold,
                            "fit_row_ids": fit_ids[inner_fit].tolist(),
                            "heldout_row_ids": fit_ids[inner_heldout].tolist(),
                        }
                    )
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "fit_row_ids": fit_ids.tolist(),
                    "heldout_row_ids": all_rows[heldout].tolist(),
                    "inner_splits": inner_rows,
                }
            )
    by_fold: dict[int, dict[str, Any]] = {}
    for raw in rows:
        outer_fold = int(raw["outer_fold"])
        fit_ids = [int(value) for value in raw["fit_row_ids"]]
        heldout_ids = [int(value) for value in raw["heldout_row_ids"]]
        if outer_fold in by_fold:
            raise ValueError(f"duplicate split provenance for outer fold {outer_fold}")
        if not fit_ids or not heldout_ids:
            raise ValueError(f"outer fold {outer_fold} requires nonempty fit and heldout rows")
        if set(fit_ids).intersection(heldout_ids):
            raise ValueError(f"outer fold {outer_fold} has overlapping fit and heldout rows")
        if any(value < 0 or value >= dataset_rows for value in [*fit_ids, *heldout_ids]):
            raise ValueError(f"outer fold {outer_fold} contains an out-of-range row id")
        inner_splits = []
        for inner_index, inner in enumerate(raw.get("inner_splits") or [], start=1):
            inner_splits.append(
                {
                    "inner_fold": int(inner.get("inner_fold", inner_index)),
                    "fit_row_ids": [int(value) for value in inner["fit_row_ids"]],
                    "heldout_row_ids": [int(value) for value in inner["heldout_row_ids"]],
                }
            )
        by_fold[outer_fold] = {
            "outer_fold": outer_fold,
            "fit_row_ids": fit_ids,
            "heldout_row_ids": heldout_ids,
            "inner_splits": inner_splits,
        }
    missing = sorted(set(map(int, outer_fold_ids)) - set(by_fold))
    if missing:
        raise ValueError(f"split provenance is missing Stage 2 outer folds: {missing}")
    return by_fold


class PlainHandoffStage2:
    def __init__(
        self,
        *,
        config: PlainHandoffStage2Config,
        clinical_question: str,
        completion: CompletionFunction | None = None,
    ) -> None:
        config = _resolve_stage2_model(config)
        config.validate()
        self.config = config
        self.clinical_question = str(clinical_question)
        self.completion = completion or _openai_completion

    def _load_or_compile_evidence(
        self,
        *,
        handoff_path: Path,
        output_dir: Path,
        seed: int,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        """Load a valid compiled plan or build it once from the raw handoff."""

        compilation_dir = output_dir / "evidence_compilation"
        packets_path = compilation_dir / "packets.jsonl"
        summary_path = compilation_dir / "summary.json"
        complete_path = compilation_dir / "compile_complete.json"
        max_packet_chars = max(2_000, self.config.max_prompt_chars // 4)
        signature = {
            "compiler": self.config.evidence_compiler,
            "compiler_version": EVIDENCE_COMPILER_VERSION,
            "required_architectures": list(self.config.required_architectures),
            "included_architectures": (
                None
                if self.config.included_architectures is None
                else list(self.config.included_architectures)
            ),
            "max_cards_per_outer_fold": self.config.evidence_max_cards_per_fold,
            "max_exemplars_per_card": self.config.evidence_max_exemplars_per_card,
            "max_exemplar_chars": self.config.evidence_max_exemplar_chars,
            "max_packet_chars": max_packet_chars,
            "seed": int(seed),
        }
        signature_fingerprint = _value_fingerprint(signature)
        handoff_size = handoff_path.stat().st_size
        hash_started = time.monotonic()
        handoff_sha256 = _file_sha256(handoff_path)
        LOGGER.info(
            "fingerprinted Stage 1 handoff bytes=%s seconds=%.2f path=%s",
            handoff_size,
            time.monotonic() - hash_started,
            handoff_path,
        )
        if complete_path.is_file() and packets_path.is_file() and summary_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if (
                complete.get("handoff_sha256") == handoff_sha256
                and complete.get("compiler_signature_sha256") == signature_fingerprint
            ):
                packets = _read_jsonl(packets_path)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                LOGGER.info(
                    "loaded cached Stage 2 evidence compilation packets=%s path=%s",
                    len(packets),
                    compilation_dir,
                )
                return packets, summary

        compile_started = time.monotonic()
        compiled = compile_stage2_handoff_evidence(
            _iter_jsonl(handoff_path),
            handoff_path=handoff_path,
            max_cards_per_outer_fold=self.config.evidence_max_cards_per_fold,
            max_exemplars_per_card=self.config.evidence_max_exemplars_per_card,
            max_exemplar_chars=self.config.evidence_max_exemplar_chars,
            max_packet_chars=max_packet_chars,
            seed=seed,
            required_architectures=self.config.required_architectures,
            included_architectures=self.config.included_architectures,
        )
        packets = [dict(packet) for packet in compiled.packets]
        summary = dict(compiled.summary)
        for outer_fold, cards in compiled.cards_by_outer_fold.items():
            fold_dir = compilation_dir / f"outer_{int(outer_fold):03d}"
            _write_jsonl(fold_dir / "cards.jsonl", cards)
            _write_jsonl(
                fold_dir / "members.jsonl",
                compiled.members_by_outer_fold[outer_fold],
            )
            _write_jsonl(
                fold_dir / "lineage.jsonl",
                compiled.lineage_by_outer_fold[outer_fold],
            )
        elapsed = time.monotonic() - compile_started
        summary = {
            **summary,
            "handoff_path": str(handoff_path),
            "handoff_bytes": handoff_size,
            "handoff_sha256": handoff_sha256,
            "compiler_signature": signature,
            "compiler_signature_sha256": signature_fingerprint,
            "compilation_seconds": elapsed,
        }
        _write_jsonl(packets_path, packets)
        _write_json(summary_path, summary)
        _write_json(
            complete_path,
            {
                "status": "complete",
                "completed_at": _now(),
                "handoff_sha256": handoff_sha256,
                "compiler_signature_sha256": signature_fingerprint,
                "packets": len(packets),
            },
        )
        LOGGER.info(
            "compiled Stage 1 evidence compiler=%s packets=%s seconds=%.2f path=%s",
            self.config.evidence_compiler,
            len(packets),
            elapsed,
            compilation_dir,
        )
        return packets, summary

    def _interpret_batch(
        self,
        *,
        architecture: str,
        packets: Sequence[Mapping[str, Any]],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        input_value = {
            "interpretation_schema": INTERPRETATION_SCHEMA_VERSION,
            "architecture": architecture,
            "packets": list(packets),
        }
        input_fingerprint = _value_fingerprint(input_value)
        packet_ids = [str(packet["packet_id"]) for packet in packets]
        packet_id_set = set(packet_ids)
        if len(packet_ids) != len(packet_id_set):
            raise ValueError("Stage 2 interpretation batch contains duplicate packet IDs")
        complete_path = output_dir / "complete.json"
        result_path = output_dir / "result.json"
        input_path = output_dir / "input.json"
        if complete_path.is_file() and result_path.is_file() and input_path.is_file():
            previous = json.loads(input_path.read_text(encoding="utf-8"))
            completion_state = json.loads(complete_path.read_text(encoding="utf-8"))
            previous_fingerprint = previous.get("input_fingerprint")
            if previous_fingerprint is None:
                previous_fingerprint = _value_fingerprint(
                    {
                        "interpretation_schema": previous.get("interpretation_schema"),
                        "architecture": previous.get("architecture"),
                        "packets": previous.get("packets") or [],
                    }
                )
            if (
                previous_fingerprint == input_fingerprint
                and completion_state.get("input_fingerprint") == input_fingerprint
            ):
                try:
                    cached_result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cached_result = None
                if _cached_interpretation_matches_packets(
                    cached_result,
                    packet_ids=packet_id_set,
                ) and (
                    cached_result.get("rejected_packet_audit", {}).get("schema_version")
                    == INTERPRETATION_AUDIT_SCHEMA_VERSION
                ):
                    LOGGER.info("skip completed Stage 2 interpretation: %s", output_dir)
                    return cached_result
            LOGGER.info("rerun stale or inconsistent Stage 2 interpretation: %s", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(input_path, {**input_value, "input_fingerprint": input_fingerprint})
        packet_evidence_axes: dict[str, list[str]] = {}
        for packet in packets:
            packet_id = str(packet["packet_id"])
            raw_axes = packet.get("observable_axes") or packet.get("evidence_axes")
            content = packet.get("content")
            if not raw_axes and isinstance(content, Mapping):
                raw_axes = content.get("evidence_axes") or content.get("observable_axes")
            packet_evidence_axes[packet_id] = _canonical_evidence_axes(raw_axes)
        initial = _checkpointed_request_json(
            output_dir=output_dir / "initial",
            input_value={
                "phase": "initial_interpretation",
                **input_value,
            },
            messages=_interpretation_prompt(
                architecture=architecture,
                packets=packets,
            ),
            config=self.config,
            completion=self.completion,
            validate=lambda value: _validate_interpretation(
                value,
                packet_ids=packet_ids,
                packet_evidence_axes=packet_evidence_axes,
            ),
        )

        packet_by_id = {str(packet["packet_id"]): packet for packet in packets}
        rejected_packets = [
            packet_by_id[packet_id]
            for packet_id, disposition in initial["packet_dispositions"].items()
            if disposition.get("status") == "reviewed_no_specific_concept"
        ]
        audit_results: list[Mapping[str, Any]] = []
        if rejected_packets:
            audit_batches = _partition_rejected_packet_audit(
                rejected_packets,
                architecture=architecture,
                max_prompt_chars=self.config.max_prompt_chars,
            )
            LOGGER.info(
                "Stage 2 rejected-packet audit architecture=%s rejected_packets=%s " "batches=%s",
                architecture,
                len(rejected_packets),
                len(audit_batches),
            )
            for batch_index, batch in enumerate(audit_batches, start=1):
                batch_ids = [str(packet["packet_id"]) for packet in batch]
                audit_results.append(
                    _checkpointed_request_json(
                        output_dir=(
                            output_dir / "rejected_packet_audit" / f"batch_{batch_index:03d}"
                        ),
                        input_value={
                            "phase": "rejected_packet_audit",
                            "audit_schema": INTERPRETATION_AUDIT_SCHEMA_VERSION,
                            "interpretation_schema": INTERPRETATION_SCHEMA_VERSION,
                            "architecture": architecture,
                            "packets": list(batch),
                        },
                        messages=_rejected_packet_audit_prompt(
                            architecture=architecture,
                            packets=batch,
                        ),
                        config=self.config,
                        completion=self.completion,
                        validate=lambda value, batch_ids=batch_ids: _validate_interpretation(
                            value,
                            packet_ids=batch_ids,
                            packet_evidence_axes=packet_evidence_axes,
                        ),
                    )
                )

        result = _merge_interpretation_audit(
            packet_ids=packet_id_set,
            initial=initial,
            audits=audit_results,
        )
        _write_json(result_path, result)
        _write_json(
            complete_path,
            {
                "status": "complete",
                "completed_at": _now(),
                "input_fingerprint": input_fingerprint,
                "initially_rejected_packets": len(rejected_packets),
                "recovered_packets": len(result["rejected_packet_audit"]["recovered_packet_ids"]),
                "audit_batches": len(audit_results),
            },
        )
        return result

    def _operationalize_candidate_group(
        self,
        *,
        group: Mapping[str, Any],
        packet_by_id: Mapping[str, Mapping[str, Any]],
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        configured = _configured_feature_definitions(group)
        if configured:
            if len(configured) != 1:
                raise ValueError(
                    "Stage 2 consolidation attempted to assign more than one supplied "
                    "ontology to one feature group"
                )
            definition = dict(configured[0])
            configured_name = str(definition.pop("name"))
            group_name = _snake_case_name(group.get("name"), fallback="")
            if group_name != configured_name:
                raise ValueError(
                    "Stage 2 consolidation renamed an investigator-configured feature; "
                    f"expected {configured_name!r}, received {group_name!r}"
                )
            roles = [
                role
                for role in _string_values(definition.pop("roles", []))
                if role in ALLOWED_ROLES
            ]
            if not roles:  # pragma: no cover - config validation invariant
                raise ValueError(
                    f"Stage 2 configured feature {configured_name!r} has no causal role"
                )
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    output_dir / "provided_ontology.json",
                    {
                        "status": "used_without_model_operationalization",
                        "source": "stage2.explicit_features",
                        "feature": {
                            "name": configured_name,
                            **definition,
                            "roles": roles,
                        },
                    },
                )
            return {
                "name": configured_name,
                **definition,
                "roles": roles,
                "supporting_packet_ids": _string_values(group.get("supporting_packet_ids")),
                "supporting_architectures": _string_values(group.get("supporting_architectures")),
                "configured_explicit_feature": True,
            }

        ontology_packet_ids = _string_values(group.get("ontology_packet_ids"))
        if not ontology_packet_ids:
            raise ValueError(
                f"Stage 2 group {group.get('name')!r} has no original evidence packet "
                "for ontology definition"
            )
        missing_packet_ids = [
            packet_id for packet_id in ontology_packet_ids if packet_id not in packet_by_id
        ]
        if missing_packet_ids:
            raise ValueError(
                f"Stage 2 group {group.get('name')!r} cites unknown ontology evidence "
                f"packet(s): {missing_packet_ids[:8]}"
            )
        evidence_packets = [packet_by_id[packet_id] for packet_id in ontology_packet_ids]
        available_supporting_evidence = _readable_supporting_text(evidence_packets)
        if not available_supporting_evidence:
            raise ValueError(
                f"Stage 2 group {group.get('name')!r} has no readable supporting "
                "evidence for ontology definition"
            )
        feature_name = str(group.get("name") or "")
        operationalization_prompt_limit = int(
            self.config.operationalization_max_prompt_chars
        )
        supporting_evidence, evidence_packing = (
            _pack_operationalization_supporting_evidence(
                feature_name=feature_name,
                supporting_evidence=available_supporting_evidence,
                max_prompt_chars=operationalization_prompt_limit,
            )
        )
        if evidence_packing["omitted_evidence_items"] or evidence_packing[
            "truncated_evidence_items"
        ]:
            LOGGER.warning(
                "Stage 2 operationalization packed supporting evidence feature=%s "
                "included=%s available=%s prompt_chars=%s prompt_limit=%s",
                feature_name,
                evidence_packing["included_evidence_items"],
                evidence_packing["available_evidence_items"],
                evidence_packing["prompt_chars"],
                operationalization_prompt_limit,
            )
        messages = _operationalization_prompt(
            feature_name=feature_name,
            supporting_evidence=supporting_evidence,
        )
        request_config = replace(
            self.config,
            max_prompt_chars=operationalization_prompt_limit,
        )
        operational = _checkpointed_request_json(
            output_dir=output_dir,
            input_value={
                "phase": "group_operationalization",
                "operationalization_schema": OPERATIONALIZATION_SCHEMA_VERSION,
                "candidate_feature_name": feature_name,
                "ontology_packet_ids": ontology_packet_ids,
                "supporting_evidence": supporting_evidence,
                "evidence_packing": evidence_packing,
            },
            messages=messages,
            config=request_config,
            completion=self.completion,
            validate=lambda value: _validate_operationalization(value, group=group),
            validation_fallback=lambda exc: _ambiguous_operationalization_fallback(
                group=group,
                validation_error=exc,
            ),
        )
        roles = _group_roles(group)
        if not roles:
            raise ValueError(
                f"Stage 2 group {group.get('name')!r} has no evidence-supported causal role"
            )
        return {
            "name": str(group["name"]),
            **operational,
            "roles": roles,
            "supporting_packet_ids": _string_values(group.get("supporting_packet_ids")),
            "supporting_architectures": _string_values(group.get("supporting_architectures")),
        }

    def _consolidate_candidate_pool(
        self,
        *,
        outer_fold: int,
        groups: Sequence[Mapping[str, Any]],
        output_dir: Path | None = None,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Losslessly merge aliases across shifted and seeded-shuffled batches."""

        current = [dict(group) for group in sorted(groups, key=_candidate_group_sort_key)]
        if not current or all(_configured_feature_definitions(group) for group in current):
            return current
        prompt_limit = int(self.config.consolidation_max_prompt_chars)
        request_config = replace(self.config, max_prompt_chars=prompt_limit)
        no_change_partitions: set[tuple[tuple[str, ...], ...]] = set()
        round_summaries: list[dict[str, Any]] = []
        stopped_reason = "maximum_rounds_reached"
        process_input = {
            "phase": "iterative_candidate_pool_merge_only_consolidation",
            "consolidation_schema": CONSOLIDATION_SCHEMA_VERSION,
            "global_candidate_pool_schema": GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION,
            "outer_fold": int(outer_fold),
            "consolidation_batch_size": int(self.config.consolidation_batch_size),
            "consolidation_alphabetical_rounds": int(self.config.consolidation_alphabetical_rounds),
            "consolidation_max_rounds": int(self.config.consolidation_max_rounds),
            "consolidation_seed": int(seed),
            "features": [_candidate_pool_feature_view(group) for group in current],
            "configured_feature_names": [
                str(group["name"]) for group in current if _configured_feature_definitions(group)
            ],
        }
        process_fingerprint = _value_fingerprint(process_input)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                output_dir / "input.json",
                {**process_input, "input_fingerprint": process_fingerprint},
            )

        for round_number in range(1, int(self.config.consolidation_max_rounds) + 1):
            if not current:
                stopped_reason = "candidate_pool_empty"
                break
            if all(_configured_feature_definitions(group) for group in current):
                stopped_reason = "only_explicit_features_remain"
                break
            ordering, boundary_offset, shuffle_round, batches = _candidate_consolidation_batches(
                current,
                batch_size=int(self.config.consolidation_batch_size),
                round_number=round_number,
                alphabetical_rounds=int(self.config.consolidation_alphabetical_rounds),
                seed=int(seed) + 1_000_003 * int(outer_fold),
            )
            partition_signature = tuple(
                tuple(str(group["name"]) for group in batch) for batch in batches
            )
            if partition_signature in no_change_partitions:
                stopped_reason = "repeated_no_change_partition"
                LOGGER.info(
                    "Stage 2 iterative consolidation converged before round=%s; "
                    "the unchanged candidate pool already used this partition",
                    round_number,
                )
                break

            round_dir = output_dir / f"round_{round_number:03d}" if output_dir is not None else None
            batch_responses: dict[int, dict[str, Any]] = {}
            jobs: list[dict[str, Any]] = []
            for batch_number, batch in enumerate(batches, start=1):
                group_names = [str(group["name"]) for group in batch]
                configured_feature_names = [
                    str(group["name"]) for group in batch if _configured_feature_definitions(group)
                ]
                if len(configured_feature_names) == len(batch):
                    batch_responses[batch_number] = {"merge_directives": []}
                    continue
                messages = _global_candidate_pool_prompt(
                    groups=batch,
                    configured_feature_names=configured_feature_names,
                    batch_ordering=ordering,
                )
                prompt_chars = sum(len(message["content"]) for message in messages)
                if prompt_chars > prompt_limit:
                    raise ValueError(
                        "one Stage 2 candidate consolidation batch cannot fit the prompt "
                        f"budget in round {round_number}, batch {batch_number} "
                        f"({prompt_chars} > {prompt_limit}); reduce "
                        "stage2.consolidation_batch_size or increase "
                        "stage2.consolidation_max_prompt_chars"
                    )
                feature_views = [_candidate_pool_feature_view(group) for group in batch]
                group_descriptions = {
                    str(feature["name"]): tuple(_string_values(feature.get("descriptions")))
                    for feature in feature_views
                }
                jobs.append(
                    {
                        "batch_number": batch_number,
                        "output_dir": (
                            round_dir / f"batch_{batch_number:03d}"
                            if round_dir is not None
                            else None
                        ),
                        "input_value": {
                            "phase": "iterative_candidate_pool_batch_consolidation",
                            "global_candidate_pool_schema": (GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION),
                            "outer_fold": int(outer_fold),
                            "round": round_number,
                            "batch": batch_number,
                            "ordering": ordering,
                            "boundary_offset": boundary_offset,
                            "shuffle_round": shuffle_round,
                            "consolidation_batch_size": int(self.config.consolidation_batch_size),
                            "consolidation_alphabetical_rounds": int(
                                self.config.consolidation_alphabetical_rounds
                            ),
                            "consolidation_max_rounds": int(self.config.consolidation_max_rounds),
                            "consolidation_seed": int(seed),
                            "features": feature_views,
                            "configured_feature_names": configured_feature_names,
                        },
                        "messages": messages,
                        "validate": (
                            lambda value, names=tuple(group_names), configured=tuple(
                                configured_feature_names
                            ), descriptions=group_descriptions: (
                                _validate_global_candidate_pool_directives(
                                    value,
                                    group_names=names,
                                    configured_feature_names=configured,
                                    group_descriptions=descriptions,
                                )
                            )
                        ),
                    }
                )

            validation_fallbacks: dict[int, str] = {}
            if jobs:
                job_by_batch = {int(job["batch_number"]): job for job in jobs}
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(1, min(self.config.workers, len(jobs)))
                ) as executor:
                    futures = {
                        executor.submit(
                            _checkpointed_request_json,
                            output_dir=job["output_dir"],
                            input_value=job["input_value"],
                            messages=job["messages"],
                            config=request_config,
                            completion=self.completion,
                            validate=job["validate"],
                        ): int(job["batch_number"])
                        for job in jobs
                    }
                    for future in concurrent.futures.as_completed(futures):
                        batch_number = futures[future]
                        try:
                            batch_responses[batch_number] = future.result()
                        except _Stage2ResponseValidationError as exc:
                            # Consolidation is an optional semantic reduction.
                            # Retaining every member of an invalid batch is the
                            # conservative, lossless fallback and necessarily
                            # preserves investigator-configured features.
                            validation_fallbacks[batch_number] = str(exc)
                            batch_responses[batch_number] = {"merge_directives": []}
                            job = job_by_batch[batch_number]
                            batch_output_dir = job["output_dir"]
                            if batch_output_dir is not None:
                                _write_json(
                                    Path(batch_output_dir) / "fallback.json",
                                    {
                                        "status": "conservative_passthrough",
                                        "completed_at": _now(),
                                        "round": round_number,
                                        "batch": batch_number,
                                        "retained_feature_names": [
                                            str(feature["name"])
                                            for feature in job["input_value"]["features"]
                                        ],
                                        "validation_error": str(exc),
                                    },
                                )
                            LOGGER.warning(
                                "Stage 2 consolidation round=%s batch=%s remained invalid; "
                                "retaining all %s supplied features unchanged (%s)",
                                round_number,
                                batch_number,
                                len(job["input_value"]["features"]),
                                exc,
                            )

            next_groups: list[dict[str, Any]] = []
            directive_count = 0
            merged_input_count = 0
            for batch_number, batch in enumerate(batches, start=1):
                response = batch_responses[batch_number]
                directives = list(response["merge_directives"])
                directive_count += len(directives)
                merged_input_count += sum(
                    max(0, len(_string_values(directive.get("inputs"))) - 1)
                    for directive in directives
                )
                next_groups.extend(
                    _apply_global_candidate_pool_directives(
                        batch,
                        directives,
                    )
                )

            next_groups, cross_batch_exact_merges = _coalesce_exact_candidate_group_names(
                next_groups
            )
            for group in next_groups:
                configured = _configured_feature_definitions(group)
                if len(configured) > 1:
                    raise ValueError(
                        "iterative Stage 2 consolidation attempted to merge distinct "
                        "investigator-configured features: "
                        f"{[feature['name'] for feature in configured]}"
                    )
                if configured and str(group["name"]) != str(configured[0]["name"]):
                    raise ValueError(
                        "iterative Stage 2 consolidation renamed an investigator-configured "
                        f"feature; expected {configured[0]['name']!r}, received "
                        f"{group['name']!r}"
                    )

            changed = bool(merged_input_count or cross_batch_exact_merges)
            round_summary = {
                "round": round_number,
                "ordering": ordering,
                "boundary_offset": boundary_offset,
                "shuffle_round": shuffle_round,
                "input_groups": len(current),
                "batches": len(batches),
                "model_requested_batches": len(jobs),
                "explicit_only_batches": len(batches) - len(jobs),
                "merge_directives": directive_count,
                "merged_inputs_removed": merged_input_count,
                "cross_batch_exact_name_merges": cross_batch_exact_merges,
                "validation_fallback_batches": len(validation_fallbacks),
                "validation_fallback_batch_numbers": sorted(validation_fallbacks),
                "output_groups": len(next_groups),
                "changed": changed,
            }
            round_summaries.append(round_summary)
            if round_dir is not None:
                _write_json(
                    round_dir / "complete.json",
                    {
                        "status": "complete",
                        "completed_at": _now(),
                        "global_candidate_pool_schema": (GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION),
                        **round_summary,
                    },
                )
            LOGGER.info(
                "Stage 2 iterative consolidation round=%s offset=%s input_groups=%s "
                "ordering=%s batches=%s merge_directives=%s "
                "cross_batch_exact_merges=%s output_groups=%s changed=%s",
                round_number,
                boundary_offset,
                len(current),
                ordering,
                len(batches),
                directive_count,
                cross_batch_exact_merges,
                len(next_groups),
                changed,
            )
            current = next_groups
            if not current:
                stopped_reason = "candidate_pool_empty"
                break
            if all(_configured_feature_definitions(group) for group in current):
                stopped_reason = "only_explicit_features_remain"
                break
            if changed:
                no_change_partitions.clear()
            elif not validation_fallbacks:
                no_change_partitions.add(partition_signature)
                if len(batches) == 1:
                    stopped_reason = "single_batch_no_change"
                    break
            elif len(batches) == 1:
                stopped_reason = "single_batch_validation_fallback"
                break

        if output_dir is not None:
            _write_json(
                output_dir / "result.json",
                {
                    "groups": current,
                    "rounds": round_summaries,
                },
            )
            _write_json(
                output_dir / "complete.json",
                {
                    "status": "complete",
                    "completed_at": _now(),
                    "input_fingerprint": process_fingerprint,
                    "global_candidate_pool_schema": GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION,
                    "rounds_executed": len(round_summaries),
                    "stopped_reason": stopped_reason,
                    "output_groups": len(current),
                    "validation_fallback_batches": sum(
                        int(summary["validation_fallback_batches"]) for summary in round_summaries
                    ),
                },
            )
        return current

    def _consolidate_candidates(
        self,
        *,
        outer_fold: int,
        candidates: Sequence[Mapping[str, Any]],
        evidence_packets: Sequence[Mapping[str, Any]],
        output_dir: Path | None = None,
        seed: int = 42,
    ) -> Mapping[str, Any]:
        """Iteratively consolidate candidate batches, then operationalize routed groups."""

        configured_candidates = [
            {
                "candidate_id": f"configured_explicit_feature_{index:04d}",
                "architecture": CONFIGURED_EXPLICIT_FEATURE_ARCHITECTURE,
                "name": feature.name,
                "description": feature.description,
                "value_type": feature.value_type,
                "supporting_packet_ids": [],
                "evidence_axes": [],
                "evidence_rationale": "Investigator-specified feature for this analysis.",
                "caveats": feature.caveats,
                "configured_feature_definitions": [feature.as_definition()],
            }
            for index, feature in enumerate(self.config.explicit_features, start=1)
        ]
        all_candidates = [*configured_candidates, *(dict(candidate) for candidate in candidates)]
        original_ids = [str(candidate["candidate_id"]) for candidate in all_candidates]
        if len(original_ids) != len(set(original_ids)):
            raise ValueError("Stage 2 consolidation received duplicate candidate IDs")
        packet_by_id = {str(packet["packet_id"]): packet for packet in evidence_packets}
        if len(packet_by_id) != len(evidence_packets):
            raise ValueError("Stage 2 ontology evidence contains duplicate packet IDs")
        cited_packet_ids = {
            packet_id
            for candidate in all_candidates
            for packet_id in _string_values(candidate.get("supporting_packet_ids"))
        }
        unknown_packet_ids = sorted(cited_packet_ids - set(packet_by_id))
        if unknown_packet_ids:
            raise ValueError(
                "Stage 2 candidates cite ontology evidence outside the supplied packet "
                f"set: {unknown_packet_ids[:8]}"
            )
        groups = _materialize_exact_name_groups(all_candidates)
        LOGGER.info(
            "Stage 2 candidate pool candidates=%s distinct_names=%s",
            len(all_candidates),
            len(groups),
        )
        retained_before_role_filter = self._consolidate_candidate_pool(
            outer_fold=outer_fold,
            groups=groups,
            output_dir=(
                output_dir / "candidate_pool_consolidation" if output_dir is not None else None
            ),
            seed=seed,
        )
        retained_groups, exclusions, role_filter_decisions = (
            _filter_candidate_groups_by_causal_role(retained_before_role_filter)
        )
        if output_dir is not None:
            _write_json(
                output_dir / "causal_role_filter.json",
                {
                    "phase": "post_consolidation_causal_role_filter",
                    "input_groups": len(retained_before_role_filter),
                    "retained_groups": len(retained_groups),
                    "excluded_groups": len(retained_before_role_filter) - len(retained_groups),
                    "decisions": role_filter_decisions,
                },
            )

        features_by_group_id: dict[str, dict[str, Any]] = {}
        if retained_groups:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(self.config.workers, len(retained_groups)))
            ) as executor:
                futures = {
                    executor.submit(
                        self._operationalize_candidate_group,
                        group=group,
                        packet_by_id=packet_by_id,
                        output_dir=(
                            output_dir / "operationalization" / f"group_{group_index:03d}"
                            if output_dir is not None
                            else None
                        ),
                    ): str(group["candidate_id"])
                    for group_index, group in enumerate(retained_groups, start=1)
                }
                for future in concurrent.futures.as_completed(futures):
                    features_by_group_id[futures[future]] = future.result()
        features = [features_by_group_id[str(group["candidate_id"])] for group in retained_groups]

        dispositions: dict[str, dict[str, str]] = {
            candidate_id: {
                "status": "excluded",
                "feature_name": "",
                "reason": exclusions.get(
                    candidate_id,
                    "Excluded before final Stage 2 operationalization.",
                ),
            }
            for candidate_id in original_ids
        }
        for group, feature in zip(retained_groups, features):
            origins = [
                origin
                for origin in _string_values(group.get("origin_candidate_ids"))
                if origin in dispositions
            ]
            for origin_index, origin in enumerate(origins):
                dispositions[origin] = {
                    "status": "retained" if origin_index == 0 else "merged",
                    "feature_name": str(feature["name"]),
                    "reason": (
                        "Retained as the canonical candidate for this scalar measurement."
                        if origin_index == 0
                        else "Merged with candidates describing the same scalar measurement."
                    ),
                }
        return {"features": features, "candidate_dispositions": dispositions}

    def _run_outer_fold(
        self,
        *,
        outer_fold: int,
        packets: Sequence[Mapping[str, Any]],
        output_dir: Path,
        dataset: pd.DataFrame | None = None,
        split: Mapping[str, Any] | None = None,
        unit_id_column: str = "patient_id",
        text_column: str = "clinical_text",
        treatment_column: str = "treatment_indicator",
        outcome_column: str = "outcome_indicator",
        outcome_type: str = "binary",
        inner_folds: int = 5,
        seed: int = 42,
    ) -> Mapping[str, Any]:
        complete_path = output_dir / "complete.json"
        features_path = output_dir / "feature_definitions.json"
        final_features_path = output_dir / "final_definitions.json"
        definitions_complete_path = output_dir / "definitions_complete.json"
        interpreted_candidates_path = output_dir / "interpreted_candidates.json"
        definition_inputs = {
            "outer_fold": int(outer_fold),
            "compiler": self.config.evidence_compiler,
            "interpretation_schema": INTERPRETATION_SCHEMA_VERSION,
            "consolidation_schema": CONSOLIDATION_SCHEMA_VERSION,
            "consolidation_batch_size": int(self.config.consolidation_batch_size),
            "consolidation_alphabetical_rounds": int(self.config.consolidation_alphabetical_rounds),
            "consolidation_max_rounds": int(self.config.consolidation_max_rounds),
            "consolidation_seed": int(seed),
            "extraction_ontology_feedback_schema": (EXTRACTION_ONTOLOGY_FEEDBACK_SCHEMA_VERSION),
            "ontology_refinement_min_failure_patients": int(
                self.config.ontology_refinement_min_failure_patients
            ),
            "max_ontology_refinement_rounds": int(self.config.max_ontology_refinement_rounds),
            "clinical_question": self.clinical_question,
            "explicit_features": [
                feature.as_definition() for feature in self.config.explicit_features
            ],
            "packets": list(packets),
        }
        evidence_input_fingerprint = _value_fingerprint(
            {
                **definition_inputs,
                "global_candidate_pool_schema": GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION,
            }
        )
        definitions_state = (
            json.loads(definitions_complete_path.read_text(encoding="utf-8"))
            if definitions_complete_path.is_file()
            else {}
        )
        completion = (
            json.loads(complete_path.read_text(encoding="utf-8")) if complete_path.is_file() else {}
        )
        if (
            completion.get("phase") == "causal_estimation"
            and final_features_path.is_file()
            and (output_dir / "estimation" / "complete.json").is_file()
        ):
            if definitions_state.get("evidence_input_fingerprint") != evidence_input_fingerprint:
                raise RuntimeError(
                    f"Stage 2 outer fold {outer_fold} was completed from a different "
                    "evidence plan, feature-definition policy, or explicit-feature "
                    "configuration. Preserve it for audit and use a fresh Stage 2 output "
                    "directory before rerunning."
                )
            LOGGER.info("skip completed Stage 2 outer fold=%s", outer_fold)
            final = json.loads(final_features_path.read_text(encoding="utf-8"))
            return {
                "outer_fold": outer_fold,
                "features": list(final.get("features") or []),
                "candidate_dispositions": (
                    json.loads(features_path.read_text(encoding="utf-8")).get(
                        "candidate_dispositions", {}
                    )
                    if features_path.is_file()
                    else {}
                ),
                "review_rounds": int(final.get("review_rounds") or 0),
                "ontology_refinement_rounds": int(final.get("ontology_refinement_rounds") or 0),
                "estimation": json.loads(
                    (output_dir / "estimation" / "diagnostics.json").read_text(encoding="utf-8")
                ),
            }
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "input_packets.jsonl", packets)

        by_architecture: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for packet in packets:
            by_architecture[str(packet["architecture"])].append(packet)
        candidates: list[dict[str, Any]] | None = None
        if features_path.is_file():
            if definitions_state.get("evidence_input_fingerprint") == evidence_input_fingerprint:
                final = json.loads(features_path.read_text(encoding="utf-8"))
            else:
                raise RuntimeError(
                    f"Stage 2 outer fold {outer_fold} has feature definitions from a "
                    "different evidence plan, feature-definition policy, or explicit-feature "
                    "configuration. Preserve the old output for audit and use a fresh "
                    "Stage 2 output directory."
                )
        else:
            jobs: list[tuple[str, int, list[Mapping[str, Any]], Path]] = []
            for architecture_index, architecture in enumerate(sorted(by_architecture), start=1):
                batches = _partition_interpretation_packets(
                    by_architecture[architecture],
                    architecture=architecture,
                    max_prompt_chars=self.config.max_prompt_chars,
                )
                for batch_index, batch in enumerate(batches, start=1):
                    jobs.append(
                        (
                            architecture,
                            batch_index,
                            batch,
                            output_dir
                            / "interpretations"
                            / f"architecture_{architecture_index:02d}"
                            / f"batch_{batch_index:03d}",
                        )
                    )
            LOGGER.info(
                "Stage 2 outer_fold=%s architectures=%s interpretation_batches=%s workers=%s",
                outer_fold,
                len(by_architecture),
                len(jobs),
                min(self.config.workers, len(jobs)),
            )
            results: list[tuple[str, Mapping[str, Any]]] = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(self.config.workers, len(jobs)))
            ) as executor:
                futures = {
                    executor.submit(
                        self._interpret_batch,
                        architecture=architecture,
                        packets=batch,
                        output_dir=job_dir,
                    ): architecture
                    for architecture, _batch_index, batch, job_dir in jobs
                }
                for future in concurrent.futures.as_completed(futures):
                    results.append((futures[future], future.result()))

            packet_by_id = {str(packet["packet_id"]): packet for packet in packets}
            candidates = []
            for architecture, result in sorted(results, key=lambda item: item[0]):
                for concept in result["concepts"]:
                    supporting_packet_ids = [
                        str(packet_id) for packet_id in concept["supporting_packet_ids"]
                    ]
                    unknown_packet_ids = sorted(set(supporting_packet_ids) - set(packet_by_id))
                    if unknown_packet_ids:
                        raise RuntimeError(
                            "Stage 2 interpretation checkpoint cites packets outside "
                            f"the current evidence plan: {unknown_packet_ids[:8]}"
                        )
                    evidence_axes = sorted(
                        {
                            str(axis)
                            for packet_id in supporting_packet_ids
                            for axis in packet_by_id[packet_id]["observable_axes"]
                        }
                    )
                    candidates.append(
                        {
                            "candidate_id": f"candidate_{len(candidates) + 1:04d}",
                            "architecture": architecture,
                            **concept,
                            "supporting_packet_ids": supporting_packet_ids,
                            "evidence_axes": evidence_axes,
                        }
                    )
            _write_json(interpreted_candidates_path, candidates)

        if candidates is not None:
            if not candidates and not self.config.explicit_features:
                final = {
                    "outer_fold": outer_fold,
                    "features": [],
                    "candidate_dispositions": {},
                }
            else:
                consolidated = self._consolidate_candidates(
                    outer_fold=outer_fold,
                    candidates=candidates,
                    evidence_packets=packets,
                    output_dir=output_dir / "consolidation",
                    seed=seed,
                )
                features = []
                for index, feature in enumerate(consolidated["features"], start=1):
                    features.append(
                        {
                            "feature_id": f"outer_{outer_fold:03d}_feature_{index:03d}",
                            **feature,
                        }
                    )
                final = {
                    "outer_fold": outer_fold,
                    "features": features,
                    "candidate_dispositions": consolidated["candidate_dispositions"],
                }
            _write_json(features_path, final)
            _write_json(
                definitions_complete_path,
                {
                    "status": "complete",
                    "completed_at": _now(),
                    "evidence_input_fingerprint": evidence_input_fingerprint,
                    "consolidation_schema": CONSOLIDATION_SCHEMA_VERSION,
                    "global_candidate_pool_schema": GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION,
                    "consolidation_batch_size": int(self.config.consolidation_batch_size),
                    "consolidation_alphabetical_rounds": int(
                        self.config.consolidation_alphabetical_rounds
                    ),
                    "consolidation_max_rounds": int(self.config.consolidation_max_rounds),
                    "extraction_ontology_feedback_schema": (
                        EXTRACTION_ONTOLOGY_FEEDBACK_SCHEMA_VERSION
                    ),
                    "ontology_refinement_min_failure_patients": int(
                        self.config.ontology_refinement_min_failure_patients
                    ),
                    "max_ontology_refinement_rounds": int(
                        self.config.max_ontology_refinement_rounds
                    ),
                    "architectures": len(by_architecture),
                    "packets": len(packets),
                    "features": len(final["features"]),
                },
            )

        if dataset is None:
            _write_json(
                complete_path,
                {
                    "status": "complete",
                    "phase": "feature_definitions",
                    "completed_at": _now(),
                    "evidence_input_fingerprint": evidence_input_fingerprint,
                    "features": len(final["features"]),
                },
            )
            return final
        if split is None:
            raise ValueError(f"Stage 2 outer fold {outer_fold} has no row split")

        from .plain_handoff_stage2_analysis import run_fold_analysis

        # Analysis contains the high-context, one-patient extraction calls. Its
        # review planner still uses max_prompt_chars, but transport must accept
        # extraction prompts up to their independent limit.
        analysis_request_config = replace(
            self.config,
            max_prompt_chars=max(
                self.config.max_prompt_chars,
                self.config.extraction_max_prompt_chars,
            ),
        )
        analysis = run_fold_analysis(
            dataset=dataset,
            definitions=final["features"],
            split=split,
            clinical_question=self.clinical_question,
            unit_id_column=unit_id_column,
            text_column=text_column,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            outcome_type=outcome_type,
            inner_folds=inner_folds,
            seed=seed + 100_000 * outer_fold,
            output_dir=output_dir,
            request_json=lambda messages, validate: _request_json(
                messages=messages,
                config=analysis_request_config,
                completion=self.completion,
                validate=validate,
            ),
            config=self.config,
        )
        completed = {
            "outer_fold": outer_fold,
            "features": analysis["features"],
            "candidate_dispositions": final.get("candidate_dispositions", {}),
            "review_rounds": analysis["review_rounds"],
            "ontology_refinement_rounds": analysis["ontology_refinement_rounds"],
            "estimation": analysis["estimation"],
        }
        _write_json(
            complete_path,
            {
                "status": "complete",
                "phase": "causal_estimation",
                "completed_at": _now(),
                "evidence_input_fingerprint": evidence_input_fingerprint,
                "features": len(completed["features"]),
                "review_rounds": completed["review_rounds"],
                "ontology_refinement_rounds": completed["ontology_refinement_rounds"],
                "estimation": completed["estimation"],
            },
        )
        return completed

    def run(
        self,
        *,
        handoff_path: Path,
        output_dir: Path,
        dataset: pd.DataFrame | None = None,
        split_provenance_path: Path | None = None,
        unit_id_column: str = "patient_id",
        text_column: str = "clinical_text",
        treatment_column: str = "treatment_indicator",
        outcome_column: str = "outcome_indicator",
        outcome_type: str = "binary",
        inner_folds: int = 5,
        seed: int = 42,
    ) -> Mapping[str, Any]:
        handoff_path = Path(handoff_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "config.json", self.config.public_dict())
        packets, compilation_summary = self._load_or_compile_evidence(
            handoff_path=handoff_path,
            output_dir=output_dir,
            seed=seed,
        )

        packets_by_outer: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for packet in packets:
            packets_by_outer[int(packet["outer_fold"])].append(packet)
        splits: dict[int, dict[str, Any]] = {}
        if dataset is not None:
            dataset = dataset.reset_index(drop=True)
            required_columns = {
                unit_id_column,
                text_column,
                treatment_column,
                outcome_column,
            }
            missing = sorted(required_columns - set(dataset.columns))
            if missing:
                raise ValueError(f"Stage 2 dataset is missing configured columns: {missing}")
            treatment_numeric = pd.to_numeric(dataset[treatment_column], errors="coerce")
            treatment_values = set(treatment_numeric.dropna().unique())
            if treatment_numeric.isna().any() or not treatment_values <= {0.0, 1.0}:
                raise ValueError("Stage 2 treatment must be a complete binary 0/1 column")
            if outcome_type not in {"binary", "continuous"}:
                raise ValueError("Stage 2 outcome_type must be binary or continuous")
            outcome_numeric = pd.to_numeric(dataset[outcome_column], errors="coerce")
            if outcome_numeric.isna().any():
                raise ValueError("Stage 2 outcome column must be complete and numeric")
            if outcome_type == "binary" and not set(outcome_numeric.unique()) <= {0.0, 1.0}:
                raise ValueError("binary Stage 2 outcome must contain only 0 and 1")
            splits = _load_stage2_splits(
                provenance_path=split_provenance_path,
                dataset_rows=len(dataset),
                outer_fold_ids=sorted(packets_by_outer),
                inner_folds=inner_folds,
                seed=seed,
            )
        fold_results = [
            self._run_outer_fold(
                outer_fold=outer_fold,
                packets=packets_by_outer[outer_fold],
                output_dir=output_dir / f"outer_{outer_fold:03d}",
                dataset=dataset,
                split=splits.get(outer_fold),
                unit_id_column=unit_id_column,
                text_column=text_column,
                treatment_column=treatment_column,
                outcome_column=outcome_column,
                outcome_type=outcome_type,
                inner_folds=inner_folds,
                seed=seed,
            )
            for outer_fold in sorted(packets_by_outer)
        ]
        _write_jsonl(output_dir / "features_by_outer_fold.jsonl", fold_results)
        name_counts: Counter[str] = Counter()
        for result in fold_results:
            names_in_fold = {
                re.sub(r"[^a-z0-9]+", "_", str(feature["name"]).lower()).strip("_")
                for feature in result["features"]
            }
            name_counts.update(names_in_fold)
        summary = {
            "outer_folds": len(fold_results),
            "evidence_packets": len(packets),
            "evidence_compiler": self.config.evidence_compiler,
            "evidence_compilation_path": str(output_dir / "evidence_compilation"),
            "evidence_compilation": compilation_summary,
            "features_by_fold": {
                str(result["outer_fold"]): len(result["features"]) for result in fold_results
            },
            "feature_name_fold_counts": dict(sorted(name_counts.items())),
            "features_path": str(output_dir / "features_by_outer_fold.jsonl"),
        }
        artifacts = [
            str(output_dir / "features_by_outer_fold.jsonl"),
            str(output_dir / "summary.json"),
        ]
        if dataset is not None:
            prediction_frames = []
            for result in fold_results:
                outer_fold = int(result["outer_fold"])
                frame = pd.read_csv(
                    output_dir / f"outer_{outer_fold:03d}" / "estimation" / "predictions.csv"
                )
                frame.insert(1, "outer_fold", outer_fold)
                prediction_frames.append(frame)
            predictions = pd.concat(prediction_frames, ignore_index=True)
            row_ids = predictions["_oci_row_id"].astype(int).tolist()
            if len(row_ids) != len(dataset) or set(row_ids) != set(range(len(dataset))):
                raise ValueError(
                    "outer heldout predictions must cover every dataset row exactly once"
                )
            if len(set(row_ids)) != len(row_ids):
                raise ValueError("a dataset row appears in more than one outer heldout fold")
            predictions = predictions.sort_values("_oci_row_id").reset_index(drop=True)
            predictions_path = output_dir / "cross_fitted_predictions.csv"
            temporary = predictions_path.with_name(f".{predictions_path.name}.{os.getpid()}.tmp")
            predictions.to_csv(temporary, index=False)
            os.replace(temporary, predictions_path)
            oracle_ite_evaluation = _evaluate_stage2_oracle_ite(
                prediction_path=predictions_path,
                dataset=dataset,
                output_dir=output_dir,
            )
            oracle_overall = dict(oracle_ite_evaluation.get("overall") or {})
            scores = predictions["aipw_score"].to_numpy(dtype=float)
            scores = scores[np.isfinite(scores)]
            if not len(scores):
                raise ValueError("cross-fitted Stage 2 estimation produced no finite AIPW scores")
            ate = float(np.mean(scores))
            standard_error = (
                float(np.std(scores, ddof=1) / math.sqrt(len(scores))) if len(scores) > 1 else None
            )
            causal_estimate = {
                "estimator": "cross-fitted_aipw_with_fold_trained_nuisance_models",
                "estimand": "average_treatment_effect",
                "rows": len(predictions),
                "ate": ate,
                "standard_error": standard_error,
                "confidence_interval_95": (
                    [ate - 1.96 * standard_error, ate + 1.96 * standard_error]
                    if standard_error is not None
                    else None
                ),
                "mean_estimated_cate": float(predictions["estimated_cate"].mean()),
                "oracle_ite_pearson_correlation": oracle_overall.get("pearson_correlation"),
                "oracle_ite_spearman_correlation": oracle_overall.get("spearman_correlation"),
                "oracle_ite_evaluation": oracle_ite_evaluation,
                "predictions_path": str(predictions_path),
            }
            causal_path = output_dir / "causal_estimate.json"
            _write_json(causal_path, causal_estimate)
            summary.update(
                {
                    "phase": "causal_estimation",
                    "causal_estimate": causal_estimate,
                    "cross_fitted_predictions_path": str(predictions_path),
                }
            )
            artifacts.extend(
                [
                    str(predictions_path),
                    str(causal_path),
                    str(output_dir / "posthoc_oracle_ite_metrics.json"),
                ]
            )
            if oracle_ite_evaluation.get("available"):
                artifacts.append(str(output_dir / "posthoc_predictions_with_oracle_ite.csv"))
        else:
            summary["phase"] = "feature_definitions"
        _write_json(output_dir / "summary.json", summary)
        return {
            "artifacts": artifacts,
            **summary,
        }


def run_plain_handoff_stage2(
    *,
    handoff_path: Path,
    output_dir: Path,
    clinical_question: str,
    config: PlainHandoffStage2Config,
    completion: CompletionFunction | None = None,
    dataset: pd.DataFrame | None = None,
    split_provenance_path: Path | None = None,
    unit_id_column: str = "patient_id",
    text_column: str = "clinical_text",
    treatment_column: str = "treatment_indicator",
    outcome_column: str = "outcome_indicator",
    outcome_type: str = "binary",
    inner_folds: int = 5,
    seed: int = 42,
) -> Mapping[str, Any]:
    return PlainHandoffStage2(
        config=config,
        clinical_question=clinical_question,
        completion=completion,
    ).run(
        handoff_path=handoff_path,
        output_dir=output_dir,
        dataset=dataset,
        split_provenance_path=split_provenance_path,
        unit_id_column=unit_id_column,
        text_column=text_column,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        outcome_type=outcome_type,
        inner_folds=inner_folds,
        seed=seed,
    )


__all__ = [
    "PlainHandoffStage2",
    "PlainHandoffStage2Config",
    "Stage2ExplicitFeature",
    "packetize_handoff",
    "plain_stage2_config_from_mapping",
    "run_plain_handoff_stage2",
]
