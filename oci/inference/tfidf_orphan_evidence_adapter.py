"""Authenticate and compact full-outer TF-IDF orphan n-gram evidence.

This adapter is deliberately mechanical.  It reads an effect n-gram score
table that was already fitted on one outer-training partition, removes terms
already summarized by the fitted topic banks, applies generic safety filters,
and greedily groups the strongest remaining terms by lexical token overlap.
It never reads patient text, treatment/outcome labels, or a held-out score
artifact, and it does not invoke a language model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .all_evidence_fusion import source_text_temporal_policy_audit

ORPHAN_NGRAM_EVIDENCE_ADAPTER_SCHEMA_VERSION = "tfidf_full_outer_orphan_ngram_evidence_v2"

_SHA256 = re.compile(r"^[0-9a-f]{64}$", flags=re.IGNORECASE)
_FORBIDDEN_FIELD = re.compile(
    r"(?:^|_)(?:oracle|true|ground_truth|groundtruth)(?:_|$)",
    flags=re.IGNORECASE,
)
_FORBIDDEN_SCORE_FIELD = re.compile(
    r"(?:^|_)(?:heldout|holdout|test|testing|oracle|true|ground_truth|groundtruth)" r"(?:_|$)",
    flags=re.IGNORECASE,
)
_ROW_LEVEL_FIELD = re.compile(
    r"(?:^|_)(?:row_id|patient_id|record_id|treatment|outcome)(?:_|$)|^[ty]$",
    flags=re.IGNORECASE,
)
_UNSAFE_SCORE_FILENAME = re.compile(
    r"(?:^|[_-])(?:heldout|holdout|test|testing)(?:[_-]|$)",
    flags=re.IGNORECASE,
)
_TOKEN = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)

# These filters describe data classes, not any disease, treatment, or benchmark.
_IDENTIFIER_NOISE = re.compile(
    r"\b(?:patient\s+id|medical\s+record|record\s+number|mrn|account\s+number|"
    r"accession\s+number|claim\s+number|document\s+id|social\s+security|ssn|"
    r"date\s+of\s+birth|dob|patient\s+name|email\s+address|phone\s+number|"
    r"street\s+address|postal\s+code|zip\s+code)\b",
    flags=re.IGNORECASE,
)
_ADMIN_NOISE = re.compile(
    r"\b(?:administrative|appointment|scheduling|scheduled\s+visit|rescheduled|"
    r"billing|insurance|prior\s+authorization|claim\s+status|faxed|phone\s+call|"
    r"encounter\s+date|note\s+date|dictated\s+by|electronically\s+signed|"
    r"signed\s+by|copied\s+to|document\s+control|routing\s+message)\b",
    flags=re.IGNORECASE,
)
_EMAIL_OR_URL = re.compile(
    r"(?:\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b|https?://|www\.)",
    flags=re.IGNORECASE,
)
_PHONE_OR_SSN = re.compile(r"(?:\b\d{3}[-.)\s]+\d{2,3}[-.\s]+\d{4}\b|\b\d{3}-\d{2}-\d{4}\b)")
_UUID_OR_LONG_ID = re.compile(
    r"(?:\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|" r"\b\d{7,}\b)",
    flags=re.IGNORECASE,
)

_LEXICAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
        "patient",
        "patients",
        "documented",
        "finding",
        "findings",
        "history",
        "level",
        "noted",
        "report",
        "reports",
        "result",
        "status",
        "value",
    }
)

_PRESERVED_NUMERIC_FIELDS = (
    "moment",
    "robust_se",
    "signed_score",
    "unsigned_score",
    "fit_signed_score",
    "fit_unsigned_score",
    "combined_importance",
    "constant_residual_effect",
    "nuisance_source_agreement",
    "subsample_selection_stability",
    "subsample_sign_agreement",
    "tail_contrast_sign_agreement",
)
_PRESERVED_INTEGER_FIELDS = (
    "support_control",
    "support_treated",
    "rank",
    "fit_rank",
    "screen_rank",
)
_ALLOWED_SCORE_SCOPES = frozenset(
    {"fit", "training", "outer_fit", "outer_train", "full_outer_train", "fit_oof"}
)


@dataclass(frozen=True)
class OrphanNgramEvidenceAdapterConfig:
    """Deterministic safety and size bounds for the residual branch."""

    min_abs_fit_score: float = 2.0
    lexical_overlap_threshold: float = 0.5
    max_candidates: int = 96
    max_clusters: int = 12
    max_terms_per_cluster: int = 8
    max_term_chars: int = 160
    max_ngram_tokens: int = 6

    def validate(self) -> None:
        if not math.isfinite(float(self.min_abs_fit_score)) or float(self.min_abs_fit_score) < 0.0:
            raise ValueError("min_abs_fit_score must be finite and non-negative")
        if not 0.0 < float(self.lexical_overlap_threshold) <= 1.0:
            raise ValueError("lexical_overlap_threshold must be in (0, 1]")
        if not 1 <= int(self.max_candidates) <= 1000:
            raise ValueError("max_candidates must be in [1, 1000]")
        if not 1 <= int(self.max_clusters) <= 64:
            raise ValueError("max_clusters must be in [1, 64]")
        if not 1 <= int(self.max_terms_per_cluster) <= 15:
            raise ValueError("max_terms_per_cluster must be in [1, 15]")
        if not 8 <= int(self.max_term_chars) <= 500:
            raise ValueError("max_term_chars must be in [8, 500]")
        if not 1 <= int(self.max_ngram_tokens) <= 10:
            raise ValueError("max_ngram_tokens must be in [1, 10]")


@dataclass(frozen=True)
class AdaptedOrphanNgramEvidence:
    """Detached fusion payload and its non-prompt audit record."""

    outer_fold: int
    _topic_banks_json: str
    _branch_json: str
    _audit_json: str

    @classmethod
    def create(
        cls,
        *,
        outer_fold: int,
        topic_banks: Mapping[str, Any],
        branch: Mapping[str, Any],
        audit: Mapping[str, Any],
    ) -> "AdaptedOrphanNgramEvidence":
        return cls(
            outer_fold=int(outer_fold),
            _topic_banks_json=_canonical_json(topic_banks),
            _branch_json=_canonical_json(branch),
            _audit_json=_canonical_json(audit),
        )

    @property
    def branch(self) -> dict[str, Any]:
        return json.loads(self._branch_json)

    @property
    def audit(self) -> dict[str, Any]:
        return json.loads(self._audit_json)

    @property
    def discovery_patch(self) -> dict[str, Any]:
        """Return the exact mapping recognized by ``_find_orphan_branch``."""

        return {"effect_orphan_ngram_branch": self.branch}

    @property
    def fusion_payload(self) -> dict[str, Any]:
        """Return a minimal TF-IDF payload accepted by all-evidence fusion."""

        branch = self.branch
        branch.pop("source_artifact_audit", None)
        return {
            "outer_fold": int(self.outer_fold),
            "scope": "outer_train",
            "discovery": {
                "topic_banks": json.loads(self._topic_banks_json),
                "effect_orphan_ngram_branch": branch,
            },
        }


def adapt_full_outer_orphan_ngram_evidence(
    row: Mapping[str, Any],
    effect_score_path: Path | str | None = None,
    *,
    artifact_base_dir: Path | str | None = None,
    expected_sha256: str | None = None,
    config: OrphanNgramEvidenceAdapterConfig = OrphanNgramEvidenceAdapterConfig(),
) -> AdaptedOrphanNgramEvidence:
    """Build an authenticated, fold-local orphan n-gram evidence branch.

    ``row`` must be one validated ``full_outer_train`` TF-IDF discovery row.
    When supplied, ``effect_score_path`` is checked against the effect artifact
    reference embedded in that row.  When omitted, the registered reference is
    resolved relative to ``artifact_base_dir`` (and then the current working
    directory for legacy references).  A declared or caller-supplied SHA-256
    is also enforced when present.
    """

    config.validate()
    if not isinstance(row, Mapping):
        raise TypeError("row must be a mapping")
    _reject_forbidden_fields(row, path="row")
    outer_fold, discovery, fit_ids, heldout_ids = _validate_full_outer_row(row)
    topic_banks = discovery.get("topic_banks")
    if not isinstance(topic_banks, Mapping):
        raise ValueError("full_outer_train discovery is missing topic_banks")

    artifacts = discovery.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("full_outer_train discovery is missing artifacts")
    score_tests_path = artifacts.get("topic_score_tests")
    if score_tests_path not in (None, ""):
        raise ValueError("full_outer_train row references a heldout/test score artifact")
    raw_scores = artifacts.get("ngram_scores")
    if not isinstance(raw_scores, Mapping) or "effect" not in raw_scores:
        raise ValueError("full_outer_train row is missing the effect n-gram artifact")

    reference_path, inline_hash = _parse_effect_reference(raw_scores["effect"])
    referenced = _resolve_registered_reference(
        reference_path,
        artifact_base_dir=None if artifact_base_dir is None else Path(artifact_base_dir),
    )
    requested = (
        referenced
        if effect_score_path is None
        else Path(effect_score_path).expanduser().resolve(strict=True)
    )
    if not requested.is_file() or requested.suffix.lower() != ".parquet":
        raise ValueError("effect_score_path must be an existing Parquet file")
    if _UNSAFE_SCORE_FILENAME.search(requested.stem):
        raise ValueError("heldout/test effect score artifacts are not allowed")
    if referenced != requested:
        raise ValueError("effect score artifact path does not match the discovery row")

    declared_hashes = _collect_declared_hashes(
        row=row,
        discovery=discovery,
        artifacts=artifacts,
        reference_path=reference_path,
        inline_hash=inline_hash,
    )
    if expected_sha256 is not None:
        declared_hashes.append(_validate_sha256(expected_sha256, source="expected_sha256"))
    if len(set(declared_hashes)) > 1:
        raise ValueError("conflicting SHA-256 registrations for effect score artifact")
    digest = _sha256_file(requested)
    if declared_hashes and digest != declared_hashes[0]:
        raise ValueError("effect score artifact SHA-256 does not match registration")

    frame = pd.read_parquet(requested)
    _validate_score_frame(frame)
    represented_terms = _represented_topic_terms(topic_banks)
    records, counts = _eligible_residual_records(
        frame,
        represented_terms=represented_terms,
        config=config,
    )
    clusters = _cluster_records(records, outer_fold=outer_fold, config=config)
    selected_ids = [str(cluster["cluster_id"]) for cluster in clusters]

    artifact_audit = {
        "registered_reference": reference_path,
        "resolved_path": str(requested),
        "sha256": digest,
        "declared_sha256_verified": bool(declared_hashes),
        "byte_size": int(requested.stat().st_size),
        "parquet_row_count": int(len(frame)),
        "parquet_columns": sorted(map(str, frame.columns)),
    }
    audit = {
        "schema_version": ORPHAN_NGRAM_EVIDENCE_ADAPTER_SCHEMA_VERSION,
        "outer_fold": outer_fold,
        "scope": "full_outer_train",
        "fit_row_fingerprint": _row_set_fingerprint(fit_ids),
        "heldout_row_fingerprint": _row_set_fingerprint(heldout_ids),
        "artifact": artifact_audit,
        "outer_fit_scope_authenticated": True,
        "heldout_scored_artifact_used": False,
        "heldout_text_or_labels_accessed": False,
        "model_inference_performed": False,
        "source_text_temporal_policy": source_text_temporal_policy_audit(),
        "represented_topic_term_count": len(represented_terms),
        **counts,
        "selected_cluster_count": len(clusters),
        "selected_term_count": sum(len(cluster["terms"]) for cluster in clusters),
    }
    branch = {
        "schema_version": ORPHAN_NGRAM_EVIDENCE_ADAPTER_SCHEMA_VERSION,
        "status": "completed" if clusters else "no_eligible_residual_ngrams",
        "candidate_definition": (
            "eligible full-outer-fit effect n-grams absent from every fitted topic "
            "bank after generic identifier and administrative filters"
        ),
        "uses_outer_heldout_labels": False,
        "uses_heldout_treatment_and_outcome": False,
        "fits_patient_level_cate_model": False,
        "topic_term_exclusion_is_fit_side": True,
        "cluster_construction_uses_heldout_rows_or_labels": False,
        "selection_rule": (
            "descending outer-fit importance with deterministic greedy lexical-token "
            "overlap clustering"
        ),
        "lexical_overlap_threshold": float(config.lexical_overlap_threshold),
        "selected_cluster_ids": selected_ids,
        "selected_clusters": clusters,
        "selection_count": len(clusters),
        "source_artifact_audit": artifact_audit,
    }
    return AdaptedOrphanNgramEvidence.create(
        outer_fold=outer_fold,
        topic_banks=topic_banks,
        branch=branch,
        audit=audit,
    )


def _validate_full_outer_row(
    row: Mapping[str, Any],
) -> tuple[int, Mapping[str, Any], tuple[Any, ...], tuple[Any, ...]]:
    scope = str(row.get("scope") or "").strip().lower()
    if scope != "full_outer_train":
        raise ValueError("orphan n-gram evidence requires a full_outer_train row")
    if row.get("inner_fold") is not None:
        raise ValueError("full_outer_train row cannot declare an inner_fold")
    try:
        outer_fold = int(row["outer_fold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("full_outer_train row requires a positive outer_fold") from exc
    if outer_fold < 1:
        raise ValueError("full_outer_train row requires a positive outer_fold")
    discovery = row.get("discovery")
    if not isinstance(discovery, Mapping):
        raise ValueError("full_outer_train row is missing discovery")
    if discovery.get("heldout_score_tests_enabled") not in (None, False):
        raise ValueError("full_outer_train discovery enables heldout score tests")
    score_tests = discovery.get("topic_score_tests")
    if score_tests is not None:
        if not isinstance(score_tests, Mapping):
            raise ValueError("topic_score_tests must be a mapping when present")
        if bool(score_tests.get("uses_heldout_treatment_and_outcome")):
            raise ValueError("full_outer_train discovery contains heldout-scored tests")
        status = str(score_tests.get("status") or "").strip().lower()
        if status in {"completed", "passed", "selected"}:
            raise ValueError("full_outer_train discovery contains a completed score test")

    fit_ids = _row_id_tuple(row.get("fit_row_ids"), field="fit_row_ids")
    heldout_ids = _row_id_tuple(row.get("heldout_row_ids"), field="heldout_row_ids")
    if set(fit_ids).intersection(heldout_ids):
        raise ValueError("full_outer_train fit and heldout row IDs overlap")
    discovery_fit = _row_id_tuple(discovery.get("fit_row_ids"), field="discovery.fit_row_ids")
    discovery_heldout = _row_id_tuple(
        discovery.get("heldout_row_ids"), field="discovery.heldout_row_ids"
    )
    if set(map(str, discovery_fit)) != set(map(str, fit_ids)):
        raise ValueError("discovery fit rows do not match full_outer_train row")
    if set(map(str, discovery_heldout)) != set(map(str, heldout_ids)):
        raise ValueError("discovery heldout rows do not match full_outer_train row")
    for container_name, container in (("row", row), ("discovery", discovery)):
        for key, ids in (
            ("fit_row_fingerprint", fit_ids),
            ("heldout_row_fingerprint", heldout_ids),
        ):
            declared = container.get(key)
            if declared is not None and str(declared) != _row_set_fingerprint(ids):
                raise ValueError(f"{container_name}.{key} does not match row IDs")
    return outer_fold, discovery, fit_ids, heldout_ids


def _row_id_tuple(value: Any, *, field: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result = tuple(value)
    if len(set(map(str, result))) != len(result):
        raise ValueError(f"{field} must contain unique row IDs")
    return result


def _parse_effect_reference(value: Any) -> tuple[str, str | None]:
    inline_hash: str | None = None
    if isinstance(value, Mapping):
        path_value = next(
            (
                value.get(key)
                for key in ("path", "artifact_path", "file", "uri")
                if value.get(key) not in (None, "")
            ),
            None,
        )
        hash_value = next(
            (
                value.get(key)
                for key in ("sha256", "artifact_sha256", "content_sha256")
                if value.get(key) not in (None, "")
            ),
            None,
        )
        if hash_value is not None:
            inline_hash = _validate_sha256(hash_value, source="effect artifact")
    else:
        path_value = value
    reference = str(path_value or "").strip()
    if not reference:
        raise ValueError("effect n-gram artifact registration has no path")
    if _UNSAFE_SCORE_FILENAME.search(Path(reference).stem):
        raise ValueError("heldout/test effect score artifacts are not allowed")
    return reference, inline_hash


def _resolve_registered_reference(
    reference: str,
    *,
    artifact_base_dir: Path | None,
) -> Path:
    path = Path(reference).expanduser()
    candidates: list[Path]
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = []
        if artifact_base_dir is not None:
            candidates.append(artifact_base_dir.expanduser() / path)
        candidates.append(Path.cwd() / path)
    resolved: list[Path] = []
    for candidate in candidates:
        try:
            value = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if value.is_file():
            resolved.append(value)
    if not resolved:
        raise ValueError("registered effect n-gram artifact does not exist")
    unique = list(dict.fromkeys(resolved))
    if len(unique) > 1:
        raise ValueError("relative effect artifact reference resolves ambiguously")
    return unique[0]


def _collect_declared_hashes(
    *,
    row: Mapping[str, Any],
    discovery: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    reference_path: str,
    inline_hash: str | None,
) -> list[str]:
    hashes = [inline_hash] if inline_hash is not None else []
    reference_name = Path(reference_path).name
    for container in (artifacts, discovery, row):
        for key in (
            "ngram_score_hashes",
            "ngram_scores_sha256",
            "ngram_score_sha256",
            "artifact_hashes",
            "artifact_sha256",
        ):
            values = container.get(key)
            if not isinstance(values, Mapping):
                continue
            candidate = None
            for lookup in ("effect", reference_path, reference_name):
                if values.get(lookup) not in (None, ""):
                    candidate = values[lookup]
                    break
            if isinstance(candidate, Mapping):
                candidate = candidate.get("sha256") or candidate.get("content_sha256")
            if candidate not in (None, ""):
                hashes.append(_validate_sha256(candidate, source=f"{key}.effect"))
    return hashes


def _validate_sha256(value: Any, *, source: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{source} must be a 64-character SHA-256 digest")
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_score_frame(frame: pd.DataFrame) -> None:
    columns = [str(column) for column in frame.columns]
    forbidden = [column for column in columns if _FORBIDDEN_SCORE_FIELD.search(column)]
    if forbidden:
        raise ValueError(
            f"effect score artifact contains forbidden heldout/test fields: {forbidden}"
        )
    row_level = [column for column in columns if _ROW_LEVEL_FIELD.search(column)]
    if row_level:
        raise ValueError(f"effect score artifact contains patient-level fields: {row_level}")
    if "feature" not in frame.columns or "eligible" not in frame.columns:
        raise ValueError("effect score artifact requires feature and eligible columns")
    if not ({"signed_score", "fit_signed_score"} & set(frame.columns)):
        raise ValueError("effect score artifact requires a signed outer-fit score")
    for scope_column in ("scope", "score_scope", "prediction_scope"):
        if scope_column not in frame.columns:
            continue
        values = {
            str(value).strip().lower() for value in frame[scope_column].dropna().unique().tolist()
        }
        if not values or not values <= _ALLOWED_SCORE_SCOPES:
            raise ValueError("effect score artifact contains non-outer-fit score rows")


def _represented_topic_terms(topic_banks: Mapping[str, Any]) -> set[str]:
    represented: set[str] = set()
    for bank in topic_banks.values():
        if not isinstance(bank, Mapping):
            continue
        topics = bank.get("topics")
        if not isinstance(topics, (list, tuple)):
            continue
        for topic in topics:
            if not isinstance(topic, Mapping):
                continue
            terms = topic.get("terms")
            if not isinstance(terms, (list, tuple)):
                continue
            for raw in terms:
                if isinstance(raw, Mapping):
                    raw = raw.get("term") or raw.get("feature") or raw.get("ngram")
                normalized = _normalize_term(raw)
                if normalized:
                    represented.add(normalized)
    return represented


def _eligible_residual_records(
    frame: pd.DataFrame,
    *,
    represented_terms: set[str],
    config: OrphanNgramEvidenceAdapterConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = {
        "source_row_count": int(len(frame)),
        "ineligible_row_count": 0,
        "invalid_term_or_score_count": 0,
        "represented_topic_term_exclusion_count": 0,
        "identifier_noise_exclusion_count": 0,
        "administrative_noise_exclusion_count": 0,
        "other_generic_noise_exclusion_count": 0,
        "below_min_abs_fit_score_count": 0,
        "duplicate_term_exclusion_count": 0,
        "eligible_residual_count_before_bound": 0,
        "bounded_candidate_count": 0,
    }
    records: list[dict[str, Any]] = []
    for source_index, raw in enumerate(frame.to_dict(orient="records"), start=1):
        if not _eligible_bool(raw.get("eligible")):
            counts["ineligible_row_count"] += 1
            continue
        term = str(raw.get("feature") or "").strip()
        normalized = _normalize_term(term)
        signed = _finite_number(raw.get("fit_signed_score"))
        if signed is None:
            signed = _finite_number(raw.get("signed_score"))
        if (
            not normalized
            or signed is None
            or len(term) > int(config.max_term_chars)
            or len(_TOKEN.findall(term)) > int(config.max_ngram_tokens)
        ):
            counts["invalid_term_or_score_count"] += 1
            continue
        if normalized in represented_terms:
            counts["represented_topic_term_exclusion_count"] += 1
            continue
        noise_reason = _generic_noise_reason(term, normalized)
        if noise_reason is not None:
            counts[f"{noise_reason}_exclusion_count"] += 1
            continue
        if abs(float(signed)) < float(config.min_abs_fit_score):
            counts["below_min_abs_fit_score_count"] += 1
            continue
        record = _term_record(raw, term=term, source_rank=source_index)
        record["_normalized"] = normalized
        record["_tokens"] = _lexical_tokens(term)
        record["_strength"] = _record_strength(record)
        records.append(record)

    records.sort(
        key=lambda item: (
            -float(item["_strength"]),
            int(item["fit_rank"]),
            str(item["_normalized"]),
            str(item["term"]),
        )
    )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        normalized = str(record["_normalized"])
        if normalized in seen:
            counts["duplicate_term_exclusion_count"] += 1
            continue
        seen.add(normalized)
        unique.append(record)
    counts["eligible_residual_count_before_bound"] = len(unique)
    bounded = unique[: int(config.max_candidates)]
    counts["bounded_candidate_count"] = len(bounded)
    return bounded, counts


def _eligible_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", "", "none", "nan"}:
        return False
    raise ValueError(f"eligible contains a non-boolean value: {value!r}")


def _term_record(raw: Mapping[str, Any], *, term: str, source_rank: int) -> dict[str, Any]:
    output: dict[str, Any] = {"term": term}
    for field in _PRESERVED_NUMERIC_FIELDS:
        value = _finite_number(raw.get(field))
        if value is not None:
            output[field] = value
    for field in _PRESERVED_INTEGER_FIELDS:
        value = _finite_number(raw.get(field))
        if value is not None:
            output[field] = int(value)
    output["fit_rank"] = int(output.get("fit_rank", source_rank))
    signed = output.get("signed_score", output.get("fit_signed_score"))
    fit_signed = output.get("fit_signed_score", signed)
    if signed is not None:
        output["signed_score"] = float(signed)
        output.setdefault("unsigned_score", abs(float(signed)))
    if fit_signed is not None:
        output["fit_signed_score"] = float(fit_signed)
        output.setdefault("fit_unsigned_score", abs(float(fit_signed)))
    output.setdefault(
        "combined_importance",
        float(output.get("fit_unsigned_score", output.get("unsigned_score", 0.0))),
    )
    return output


def _record_strength(record: Mapping[str, Any]) -> float:
    for field in (
        "combined_importance",
        "fit_unsigned_score",
        "unsigned_score",
        "fit_signed_score",
        "signed_score",
    ):
        value = _finite_number(record.get(field))
        if value is not None:
            return abs(value)
    return 0.0


def _cluster_records(
    records: Sequence[Mapping[str, Any]],
    *,
    outer_fold: int,
    config: OrphanNgramEvidenceAdapterConfig,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    threshold = float(config.lexical_overlap_threshold)
    for raw in records:
        tokens = frozenset(raw.get("_tokens") or ())
        best_index: int | None = None
        best_similarity = -1.0
        for index, group in enumerate(groups):
            if len(group["members"]) >= int(config.max_terms_per_cluster):
                continue
            similarity = _overlap_coefficient(tokens, group["seed_tokens"])
            if similarity >= threshold and similarity > best_similarity:
                best_index = index
                best_similarity = similarity
        if best_index is None:
            if len(groups) >= int(config.max_clusters):
                continue
            groups.append(
                {
                    "seed_tokens": tokens,
                    "members": [(raw, 1.0)],
                }
            )
        else:
            groups[best_index]["members"].append((raw, best_similarity))

    output: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        members: list[dict[str, Any]] = []
        for raw, similarity in group["members"]:
            member = {str(key): value for key, value in raw.items() if not str(key).startswith("_")}
            member["lexical_similarity_to_seed"] = float(similarity)
            members.append(member)
        output.append(
            {
                "cluster_id": f"effect_orphan_outer_{outer_fold:03d}_{index:03d}",
                "evidence_kind": "orphan_raw_ngram_cluster",
                "terms": members,
                "seed_term": members[0]["term"],
                "fit_rank": min(int(member["fit_rank"]) for member in members),
                "maximum_abs_fit_signed_score": max(
                    abs(float(member["fit_signed_score"])) for member in members
                ),
                "grouping_method": "greedy_seed_lexical_token_overlap_coefficient",
            }
        )
    return output


def _generic_noise_reason(term: str, normalized: str) -> str | None:
    if _IDENTIFIER_NOISE.search(term) or _EMAIL_OR_URL.search(term):
        return "identifier_noise"
    if _PHONE_OR_SSN.search(term) or _UUID_OR_LONG_ID.search(term):
        return "identifier_noise"
    digits = "".join(character for character in term if character.isdigit())
    tokens = _TOKEN.findall(normalized)
    if tokens and all(token.isdigit() for token in tokens) and len(digits) >= 6:
        return "identifier_noise"
    if _ADMIN_NOISE.search(term):
        return "administrative_noise"
    if not _lexical_tokens(term):
        return "other_generic_noise"
    return None


def _normalize_term(value: Any) -> str:
    return " ".join(_TOKEN.findall(str(value or "").lower()))


def _lexical_tokens(value: Any) -> tuple[str, ...]:
    tokens = [
        token
        for token in _TOKEN.findall(str(value or "").lower())
        if token not in _LEXICAL_STOPWORDS
    ]
    return tuple(dict.fromkeys(tokens))


def _overlap_coefficient(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left.intersection(right)) / min(len(left), len(right)))


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _reject_forbidden_fields(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            if _FORBIDDEN_FIELD.search(name):
                raise ValueError(f"forbidden true/oracle field at {path}.{name}")
            _reject_forbidden_fields(child, path=f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, path=f"{path}[{index}]")


def _row_set_fingerprint(row_ids: Sequence[Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            sorted(str(value) for value in row_ids),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
