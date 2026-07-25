"""Fold-honest adapters for cohort query-moment evidence.

The adapter has two deliberately small modes:

* authenticate and sanitize a previously fitted ``query_evidence.json`` file;
* derive sparse query activations from terms already present in a fold-local
  TF-IDF topic/orphan handoff.

Neither mode imports, loads, or calls a language model.  Query definitions in
the fallback are entirely mechanical: supplied terms become a fixed sparse
vocabulary, and all treatment, outcome, and effect moments are evaluated only
on the explicitly registered outer-training rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .all_evidence_fusion import (
    FoldEvidenceInput,
    FoldEvidenceProvenance,
    NEURAL_QUERY_MOMENTS,
    NEURAL_QUERY_SOURCE,
    SPARSE_QUERY_MOMENTS,
    SPARSE_QUERY_SOURCE,
)
from .neural_cohort_witness import (
    fit_constant_residual_effect,
    standardized_cohort_moments,
    standardized_direct_target_contrasts,
)
from .tfidf_topic_discovery import row_set_fingerprint

QUERY_MOMENT_ADAPTER_SCHEMA_VERSION = "fold_honest_sparse_query_moments_v1"
NEURAL_QUERY_EVIDENCE_BUNDLE_SCHEMA_VERSION = "neural_query_moment_evidence_v1"

_BANKS = ("treatment", "outcome", "effect")
_ROLE_BY_BANK = {
    "treatment": "confounder",
    "outcome": "confounder",
    "effect": "effect_modifier",
}
_FORBIDDEN_KEY = re.compile(
    r"(?:^|_)(?:oracle|true|ground_truth)(?:_|$)|(?:oracle|ground_truth)",
    flags=re.IGNORECASE,
)
_FORBIDDEN_STRING = re.compile(
    r"\boracle\b|\bground[\s_-]*truth\b|\btrue\b|\btrue[_-][a-z0-9_]+\b",
    flags=re.IGNORECASE,
)
_TOKEN = re.compile(r"[a-z0-9%<>+=-]+", flags=re.IGNORECASE)
_LEGACY_ROW_IN_EVIDENCE_ID = re.compile(r"__row_(\d+)__", flags=re.IGNORECASE)
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_QUERY_FIELDS = frozenset(
    {
        "query_id",
        "bank",
        "mechanical_role",
        "statistical_gate_applied",
        "member_count",
        "member_subfolds",
        "fit_standardized_score",
        "top_chunks",
        "top_contrastive_ngrams",
    }
)
_CHUNK_FIELDS = frozenset(
    {
        "evidence_id",
        "_oci_row_id",
        "row_id",
        "chunk_index",
        "similarity",
        "text",
        "chunk_text",
    }
)
_TERM_FIELDS = frozenset(
    {
        "term",
        "tfidf_contrast",
        "loading",
        "signed_score",
        "fit_signed_score",
        "standardized_score",
        "rank",
        "fit_rank",
    }
)


@dataclass(frozen=True)
class QueryMomentEvidenceAdapterConfig:
    """Strict size limits shared by artifact and sparse-fallback modes."""

    max_queries: int = 24
    # Historical neural-query artifacts retained 20 contrastive terms.  Fusion
    # applies its own smaller prompt cap after this adapter authenticates them.
    max_terms_per_query: int = 32
    max_chunks_per_query: int = 16
    fallback_chunks_per_query: int = 8
    max_excerpt_chars: int = 1200
    max_term_chars: int = 160
    max_ngram_tokens: int = 6

    def validate(self) -> None:
        if int(self.max_queries) < 1:
            raise ValueError("max_queries must be positive")
        if int(self.max_terms_per_query) < 1:
            raise ValueError("max_terms_per_query must be positive")
        if int(self.max_chunks_per_query) < 1:
            raise ValueError("max_chunks_per_query must be positive")
        if not 1 <= int(self.fallback_chunks_per_query) <= int(self.max_chunks_per_query):
            raise ValueError(
                "fallback_chunks_per_query must be positive and no larger than "
                "max_chunks_per_query"
            )
        if int(self.max_excerpt_chars) < 1:
            raise ValueError("max_excerpt_chars must be positive")
        if int(self.max_term_chars) < 1:
            raise ValueError("max_term_chars must be positive")
        if int(self.max_ngram_tokens) < 1:
            raise ValueError("max_ngram_tokens must be positive")


@dataclass(frozen=True)
class AdaptedQueryMomentEvidence:
    """Detached evidence payload plus a non-prompt audit record."""

    provenance: FoldEvidenceProvenance
    source_kind: str
    _payload_json: str
    _audit_json: str

    @classmethod
    def create(
        cls,
        *,
        provenance: FoldEvidenceProvenance,
        source_kind: str,
        payload: Mapping[str, Any],
        audit: Mapping[str, Any],
    ) -> "AdaptedQueryMomentEvidence":
        if source_kind not in {NEURAL_QUERY_SOURCE, SPARSE_QUERY_SOURCE}:
            raise ValueError("query-moment evidence has an invalid source kind")
        return cls(
            provenance=provenance,
            source_kind=source_kind,
            _payload_json=_canonical_json(payload),
            _audit_json=_canonical_json(audit),
        )

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._payload_json)

    @property
    def audit(self) -> dict[str, Any]:
        return json.loads(self._audit_json)

    def as_fold_evidence_input(self) -> FoldEvidenceInput:
        """Return the exact input type accepted by all-evidence fusion."""

        return FoldEvidenceInput(
            source_kind=self.source_kind,
            payload=self.payload,
            provenance=self.provenance,
        )


@dataclass(frozen=True)
class _SparseQueryDefinition:
    bank: str
    terms: tuple[tuple[str, float], ...]


def load_query_moment_evidence_artifact(
    path: Path | str,
    *,
    provenance: FoldEvidenceProvenance,
    expected_sha256: str | None = None,
    registered_fit_row_ids: Sequence[Hashable] | None = None,
    registered_heldout_row_ids: Sequence[Hashable] | None = None,
    config: QueryMomentEvidenceAdapterConfig = QueryMomentEvidenceAdapterConfig(),
) -> AdaptedQueryMomentEvidence:
    """Load a legacy or wrapped query-evidence artifact and verify its split.

    Bare legacy lists do not contain a full fit-row registry.  For those files,
    ``provenance`` is the mandatory registry and every retrieved row is checked
    against it.  Callers with a separately sealed artifact registry can pass
    ``registered_*_row_ids`` for an additional exact membership check.
    """

    config.validate()
    train_ids, heldout_ids = _validate_outer_provenance(provenance)
    _validate_registered_partition(
        registered_fit_row_ids,
        registered_heldout_row_ids,
        train_ids=train_ids,
        heldout_ids=heldout_ids,
    )
    requested = Path(path).resolve()
    content = requested.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if not _HEX_SHA256.fullmatch(expected):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        if digest != expected:
            raise ValueError("query-evidence artifact SHA-256 does not match registration")
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid query-evidence JSON: {requested}") from exc
    _reject_forbidden_content(raw, path="artifact")

    declared_partition = False
    if isinstance(raw, list):
        query_rows = raw
    elif isinstance(raw, Mapping):
        declared_source_kind = str(raw.get("source_kind") or "").strip().lower()
        if declared_source_kind and declared_source_kind != NEURAL_QUERY_SOURCE:
            raise ValueError("registered neural query artifact declares a different source kind")
        declared_source_family = str(raw.get("source_family") or "").strip().lower()
        if declared_source_family and declared_source_family != NEURAL_QUERY_MOMENTS:
            raise ValueError(
                "registered neural query artifact declares a different source family"
            )
        adapter_mode = str(raw.get("adapter_mode") or "").strip().lower()
        if adapter_mode and adapter_mode != "authenticated_neural_query_artifact":
            raise ValueError("registered neural query artifact is not learned neural evidence")
        if raw.get("outer_fold") is not None and int(raw["outer_fold"]) != int(
            provenance.outer_fold
        ):
            raise ValueError("query-evidence outer_fold does not match provenance")
        scope = str(raw.get("scope") or "").strip().lower()
        if scope and scope not in {"outer_train", "full_outer_train"}:
            raise ValueError("query-evidence artifact is not an outer-training artifact")
        declared_fit = raw.get("fit_row_ids", raw.get("train_row_ids"))
        declared_heldout = raw.get("heldout_row_ids")
        if declared_fit is not None or declared_heldout is not None:
            if declared_fit is None or declared_heldout is None:
                raise ValueError("artifact must declare both fit and heldout row IDs")
            _validate_registered_partition(
                declared_fit,
                declared_heldout,
                train_ids=train_ids,
                heldout_ids=heldout_ids,
            )
            declared_partition = True
        declared_fit_fingerprint = raw.get("fit_row_fingerprint")
        declared_heldout_fingerprint = raw.get("heldout_row_fingerprint")
        if declared_fit_fingerprint is not None and str(
            declared_fit_fingerprint
        ) != row_set_fingerprint(train_ids):
            raise ValueError("artifact fit_row_fingerprint does not match provenance")
        if declared_heldout_fingerprint is not None and str(
            declared_heldout_fingerprint
        ) != row_set_fingerprint(heldout_ids):
            raise ValueError("artifact heldout_row_fingerprint does not match provenance")
        query_rows = raw.get("query_evidence")
    else:
        query_rows = None
    if not isinstance(query_rows, list) or not query_rows:
        raise ValueError("query-evidence artifact must contain a non-empty query list")
    if all(
        isinstance(row, Mapping)
        and str(row.get("query_id") or "").startswith("sparse_")
        for row in query_rows
    ):
        raise ValueError("sparse fallback evidence cannot be registered as neural query evidence")
    sanitized, cited_rows = _sanitize_query_records(
        query_rows,
        train_ids=train_ids,
        heldout_ids=heldout_ids,
        config=config,
    )
    payload = _build_payload(
        provenance,
        sanitized,
        mode="authenticated_neural_query_artifact",
    )
    audit = {
        "schema_version": QUERY_MOMENT_ADAPTER_SCHEMA_VERSION,
        "mode": "authenticated_neural_query_artifact",
        "source_kind": NEURAL_QUERY_SOURCE,
        "source_family": NEURAL_QUERY_MOMENTS,
        "query_definition_kind": "learned_neural_cohort_queries",
        "artifact_path": str(requested),
        "artifact_sha256": digest,
        "outer_fold": int(provenance.outer_fold),
        "fit_row_fingerprint": row_set_fingerprint(train_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
        "artifact_declared_full_partition": bool(declared_partition),
        "artifact_declared_neural_source_kind": bool(
            isinstance(raw, Mapping) and raw.get("source_kind") == NEURAL_QUERY_SOURCE
        ),
        "separate_partition_registration_checked": bool(
            registered_fit_row_ids is not None or registered_heldout_row_ids is not None
        ),
        "query_count": len(sanitized),
        "retrieved_fit_row_count": len(cited_rows),
        "retrieved_rows_are_outer_train_only": True,
        "model_inference_performed": False,
    }
    return AdaptedQueryMomentEvidence.create(
        provenance=provenance,
        source_kind=NEURAL_QUERY_SOURCE,
        payload=payload,
        audit=audit,
    )


def derive_sparse_query_moment_evidence(
    *,
    provenance: FoldEvidenceProvenance,
    outer_train_row_ids: Sequence[Hashable],
    outer_train_texts: Sequence[str],
    treatment: Sequence[float],
    outcome: Sequence[float],
    tfidf_topic_evidence: Mapping[str, Any],
    config: QueryMomentEvidenceAdapterConfig = QueryMomentEvidenceAdapterConfig(),
) -> AdaptedQueryMomentEvidence:
    """Build deterministic sparse query moments from supplied TF-IDF terms.

    The function requires the complete outer-training partition.  It has no
    heldout text or heldout label parameter, making accidental heldout moment
    evaluation impossible once the row-membership check passes.
    """

    config.validate()
    train_ids, heldout_ids = _validate_outer_provenance(provenance)
    ordered_ids = tuple(_normalize_row_id(value) for value in outer_train_row_ids)
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("outer_train_row_ids contains duplicates")
    if set(ordered_ids) != train_ids:
        unexpected = set(ordered_ids) - train_ids
        if unexpected & heldout_ids:
            raise ValueError("outer_train_row_ids contains an outer-heldout row")
        raise ValueError("outer_train_row_ids does not exactly match fold provenance")
    texts = tuple(str(value or "") for value in outer_train_texts)
    treatment_values = _finite_vector(treatment, name="treatment")
    outcome_values = _finite_vector(outcome, name="outcome")
    n_rows = len(ordered_ids)
    if not (len(texts) == len(treatment_values) == len(outcome_values) == n_rows):
        raise ValueError("texts, labels, and outer-training row IDs must have equal lengths")
    if n_rows < 3:
        raise ValueError("at least three outer-training rows are required")
    if set(np.unique(treatment_values).tolist()) != {0.0, 1.0}:
        raise ValueError("outer-training treatment must contain binary values 0 and 1")
    _reject_forbidden_content(tfidf_topic_evidence, path="tfidf_topic_evidence")
    definitions, definition_audit = _extract_sparse_query_definitions(
        tfidf_topic_evidence,
        config=config,
    )
    if not definitions:
        raise ValueError("supplied TF-IDF evidence contains no usable topic/orphan terms")

    vocabulary_terms = sorted(
        {term for definition in definitions for term, _weight in definition.terms}
    )
    maximum_ngram = max(len(term.split()) for term in vocabulary_terms)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)[a-z0-9%<>+=-]+",
        ngram_range=(1, maximum_ngram),
        vocabulary={term: index for index, term in enumerate(vocabulary_terms)},
        sublinear_tf=True,
        smooth_idf=True,
        norm="l2",
        dtype=np.float64,
    )
    sparse_terms = vectorizer.fit_transform(texts)

    treatment_scores = None
    outcome_scores = None
    treatment_residual = treatment_values - float(np.mean(treatment_values))
    outcome_residual = outcome_values - float(np.mean(outcome_values))
    constant_effect = fit_constant_residual_effect(
        treatment_residual,
        outcome_residual,
    )
    query_rows: list[dict[str, Any]] = []
    bank_counts = {bank: 0 for bank in _BANKS}
    zero_activation_count = 0
    skipped_excerpt_count = 0
    for definition in definitions:
        indices = np.asarray(
            [vectorizer.vocabulary_[term] for term, _weight in definition.terms],
            dtype=int,
        )
        weights = np.asarray(
            [weight for _term, weight in definition.terms],
            dtype=float,
        )
        weights /= max(float(np.sum(weights)), 1e-12)
        activations = np.asarray(sparse_terms[:, indices] @ weights).reshape(-1)
        nonzero = np.flatnonzero(activations > 0.0)
        if not len(nonzero):
            zero_activation_count += 1
            continue
        if definition.bank == "treatment":
            if treatment_scores is None:
                treatment_scores = standardized_direct_target_contrasts(
                    activations[:, None],
                    treatment_values,
                    binary=True,
                )
            scores = treatment_scores
        elif definition.bank == "outcome":
            if outcome_scores is None:
                outcome_scores = standardized_direct_target_contrasts(
                    activations[:, None],
                    outcome_values,
                    binary=False,
                )
            scores = outcome_scores
        else:
            scores = standardized_cohort_moments(
                activations[:, None],
                treatment_residual,
                outcome_residual,
                constant_effect=constant_effect,
                center_with_evaluation_treatment=True,
            )
        # The cached direct-target dictionaries above are valid only for their
        # corresponding activation.  Reset them before the next definition.
        if definition.bank == "treatment":
            treatment_scores = None
        elif definition.bank == "outcome":
            outcome_scores = None

        bank_counts[definition.bank] += 1
        query_id = f"sparse_{definition.bank}_query_{bank_counts[definition.bank]:03d}"
        ranked_positions = sorted(
            nonzero.tolist(),
            key=lambda position: (-float(activations[position]), int(position)),
        )
        chunks: list[dict[str, Any]] = []
        source_terms = [term for term, _weight in definition.terms]
        for position in ranked_positions:
            excerpt = _term_centered_excerpt(
                texts[position],
                source_terms,
                max_chars=int(config.max_excerpt_chars),
            )
            if not excerpt or _FORBIDDEN_STRING.search(excerpt):
                skipped_excerpt_count += 1
                continue
            chunks.append(
                {
                    "evidence_id": f"{query_id}__evidence_{len(chunks) + 1:03d}",
                    "_oci_row_id": ordered_ids[position],
                    "chunk_index": 0,
                    "similarity": float(activations[position]),
                    "text": excerpt,
                }
            )
            if len(chunks) >= int(config.fallback_chunks_per_query):
                break
        top_terms = [
            {
                "term": term,
                "loading": float(weight / max(float(np.sum(weights)), 1e-12)),
            }
            for (term, _raw_weight), weight in zip(definition.terms, weights)
        ]
        query_rows.append(
            {
                "query_id": query_id,
                "bank": definition.bank,
                "mechanical_role": _ROLE_BY_BANK[definition.bank],
                "statistical_gate_applied": False,
                "member_count": len(definition.terms),
                "member_subfolds": [],
                "fit_standardized_score": float(scores["standardized_scores"][0]),
                "top_chunks": chunks,
                "top_contrastive_ngrams": top_terms,
            }
        )
    if not query_rows:
        raise ValueError("no supplied TF-IDF query term occurs in outer-training text")
    sanitized, cited_rows = _sanitize_query_records(
        query_rows,
        train_ids=train_ids,
        heldout_ids=heldout_ids,
        config=config,
    )
    payload = _build_payload(provenance, sanitized, mode="deterministic_sparse_fallback")
    audit = {
        "schema_version": QUERY_MOMENT_ADAPTER_SCHEMA_VERSION,
        "mode": "deterministic_sparse_fallback",
        "source_kind": SPARSE_QUERY_SOURCE,
        "source_family": SPARSE_QUERY_MOMENTS,
        "query_definition_kind": "fixed_sparse_tfidf_term_queries",
        "outer_fold": int(provenance.outer_fold),
        "fit_row_fingerprint": row_set_fingerprint(train_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
        "query_definition_source": "supplied_tfidf_topic_and_orphan_terms_only",
        "query_count": len(sanitized),
        "query_count_by_bank": {
            bank: sum(row["bank"] == bank for row in sanitized) for bank in _BANKS
        },
        "retrieved_fit_row_count": len(cited_rows),
        "moment_row_count": n_rows,
        "moment_rows_are_outer_train_only": True,
        "heldout_text_or_labels_accessed": False,
        "zero_activation_definition_count": int(zero_activation_count),
        "forbidden_excerpt_skip_count": int(skipped_excerpt_count),
        "definition_audit": definition_audit,
        "model_inference_performed": False,
    }
    return AdaptedQueryMomentEvidence.create(
        provenance=provenance,
        source_kind=SPARSE_QUERY_SOURCE,
        payload=payload,
        audit=audit,
    )


def adapt_query_moment_evidence(
    *,
    provenance: FoldEvidenceProvenance,
    artifact_path: Path | str | None = None,
    expected_artifact_sha256: str | None = None,
    registered_fit_row_ids: Sequence[Hashable] | None = None,
    registered_heldout_row_ids: Sequence[Hashable] | None = None,
    outer_train_row_ids: Sequence[Hashable] | None = None,
    outer_train_texts: Sequence[str] | None = None,
    treatment: Sequence[float] | None = None,
    outcome: Sequence[float] | None = None,
    tfidf_topic_evidence: Mapping[str, Any] | None = None,
    config: QueryMomentEvidenceAdapterConfig = QueryMomentEvidenceAdapterConfig(),
) -> AdaptedQueryMomentEvidence:
    """Use an existing artifact when present, otherwise use the sparse fallback."""

    if artifact_path is not None and Path(artifact_path).exists():
        return load_query_moment_evidence_artifact(
            artifact_path,
            provenance=provenance,
            expected_sha256=expected_artifact_sha256,
            registered_fit_row_ids=registered_fit_row_ids,
            registered_heldout_row_ids=registered_heldout_row_ids,
            config=config,
        )
    if expected_artifact_sha256 is not None:
        raise FileNotFoundError("registered query-evidence artifact is missing")
    missing = [
        name
        for name, value in (
            ("outer_train_row_ids", outer_train_row_ids),
            ("outer_train_texts", outer_train_texts),
            ("treatment", treatment),
            ("outcome", outcome),
            ("tfidf_topic_evidence", tfidf_topic_evidence),
        )
        if value is None
    ]
    if missing:
        raise ValueError("sparse query-moment fallback is missing inputs: " + ", ".join(missing))
    return derive_sparse_query_moment_evidence(
        provenance=provenance,
        outer_train_row_ids=outer_train_row_ids,
        outer_train_texts=outer_train_texts,
        treatment=treatment,
        outcome=outcome,
        tfidf_topic_evidence=tfidf_topic_evidence,
        config=config,
    )


def reseal_legacy_neural_query_moment_evidence(
    *,
    query_evidence_path: Path | str,
    query_subfold_audit_path: Path | str,
    summary_path: Path | str,
    provenance: FoldEvidenceProvenance,
    config: QueryMomentEvidenceAdapterConfig = QueryMomentEvidenceAdapterConfig(),
) -> dict[str, Any]:
    """Create a self-declaring bundle from the July 14 bare artifact trio.

    The rich historical summary and subfold audit contain fields outside this
    migration's safety boundary.  ``ijson`` projects only the summary scope and
    exact subfold row partitions; no diagnostic, label, prediction, or post-hoc
    evaluation values are materialized.
    """

    train_ids, heldout_ids = _validate_outer_provenance(provenance)
    evidence_path = Path(query_evidence_path).resolve()
    audit_path = Path(query_subfold_audit_path).resolve()
    source_summary_path = Path(summary_path).resolve()
    for path, label in (
        (evidence_path, "query evidence"),
        (audit_path, "query subfold audit"),
        (source_summary_path, "query summary"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    initial_hashes = {
        "raw_query_evidence_sha256": _sha256_path(evidence_path),
        "query_subfold_audit_sha256": _sha256_path(audit_path),
        "summary_sha256": _sha256_path(source_summary_path),
    }
    adapted = load_query_moment_evidence_artifact(
        evidence_path,
        provenance=provenance,
        expected_sha256=initial_hashes["raw_query_evidence_sha256"],
        registered_fit_row_ids=provenance.train_row_ids,
        registered_heldout_row_ids=provenance.heldout_row_ids,
        config=config,
    )
    if adapted.audit["artifact_declared_full_partition"]:
        raise ValueError("legacy resealing expects a bare query-evidence artifact")

    summary_scope = _project_single_json_value(
        source_summary_path,
        "scope",
        label="query summary scope",
    )
    if not isinstance(summary_scope, Mapping):
        raise ValueError("query summary scope must be an object")
    if int(summary_scope.get("outer_fold", 0)) != int(provenance.outer_fold):
        raise ValueError("query summary outer fold does not match provenance")
    if str(summary_scope.get("fit_row_fingerprint") or "") != row_set_fingerprint(
        train_ids
    ):
        raise ValueError("query summary fit fingerprint does not match provenance")
    if str(summary_scope.get("heldout_row_fingerprint") or "") != row_set_fingerprint(
        heldout_ids
    ):
        raise ValueError("query summary heldout fingerprint does not match provenance")

    subfold_numbers = _project_json_values(audit_path, "item.fold")
    subfold_train = _project_json_values(
        audit_path,
        "item.identity_payload.train_row_ids",
    )
    subfold_validation = _project_json_values(
        audit_path,
        "item.identity_payload.validation_row_ids",
    )
    if not subfold_numbers or not (
        len(subfold_numbers) == len(subfold_train) == len(subfold_validation)
    ):
        raise ValueError("query subfold audit has incomplete row partitions")
    normalized_folds = [_positive_int(value, name="query subfold fold") for value in subfold_numbers]
    if len(normalized_folds) != len(set(normalized_folds)):
        raise ValueError("query subfold audit contains duplicate folds")
    validation_counts: dict[Hashable, int] = {}
    for fold, raw_fit, raw_validation in zip(
        normalized_folds,
        subfold_train,
        subfold_validation,
    ):
        if not isinstance(raw_fit, list) or not isinstance(raw_validation, list):
            raise ValueError(f"query subfold {fold} row partitions must be lists")
        fit = [_normalize_row_id(value) for value in raw_fit]
        validation = [_normalize_row_id(value) for value in raw_validation]
        if len(fit) != len(set(fit)) or len(validation) != len(set(validation)):
            raise ValueError(f"query subfold {fold} contains duplicate row IDs")
        fit_set = set(fit)
        validation_set = set(validation)
        if fit_set & validation_set or fit_set | validation_set != train_ids:
            raise ValueError(
                f"query subfold {fold} does not exactly partition outer training"
            )
        if (fit_set | validation_set) & heldout_ids:
            raise ValueError(f"query subfold {fold} contains an outer-heldout row")
        for row_id in validation:
            validation_counts[row_id] = validation_counts.get(row_id, 0) + 1
    if set(validation_counts) != train_ids or set(validation_counts.values()) != {1}:
        raise ValueError("query subfold validation rows do not partition outer training once")

    final_hashes = {
        "raw_query_evidence_sha256": _sha256_path(evidence_path),
        "query_subfold_audit_sha256": _sha256_path(audit_path),
        "summary_sha256": _sha256_path(source_summary_path),
    }
    if final_hashes != initial_hashes:
        raise RuntimeError("a neural query migration source changed during validation")
    ordered_fit = [_normalize_row_id(value) for value in provenance.train_row_ids]
    ordered_heldout = [_normalize_row_id(value) for value in provenance.heldout_row_ids]
    return {
        "schema_version": NEURAL_QUERY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "source_kind": NEURAL_QUERY_SOURCE,
        "source_family": NEURAL_QUERY_MOMENTS,
        "outer_fold": int(provenance.outer_fold),
        "scope": "outer_train",
        "fit_row_ids": ordered_fit,
        "heldout_row_ids": ordered_heldout,
        "fit_row_fingerprint": row_set_fingerprint(ordered_fit),
        "heldout_row_fingerprint": row_set_fingerprint(ordered_heldout),
        "query_evidence": adapted.payload["query_evidence"],
        "source_provenance": {
            **initial_hashes,
            "summary_scope_fingerprints_verified": True,
            "exact_subfold_partitions_verified": True,
            "retrieved_evidence_rows_verified_outer_train_only": True,
        },
    }


def _build_payload(
    provenance: FoldEvidenceProvenance,
    query_rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    sparse = mode == "deterministic_sparse_fallback"
    return {
        "schema_version": QUERY_MOMENT_ADAPTER_SCHEMA_VERSION,
        "source_kind": SPARSE_QUERY_SOURCE if sparse else NEURAL_QUERY_SOURCE,
        "source_family": SPARSE_QUERY_MOMENTS if sparse else NEURAL_QUERY_MOMENTS,
        "outer_fold": int(provenance.outer_fold),
        "scope": "outer_train",
        "adapter_mode": mode,
        "query_evidence": list(query_rows),
    }


def _validate_outer_provenance(
    provenance: FoldEvidenceProvenance,
) -> tuple[set[Hashable], set[Hashable]]:
    if not isinstance(provenance, FoldEvidenceProvenance):
        raise TypeError("provenance must be FoldEvidenceProvenance")
    if provenance.scope != "outer_train" or provenance.inner_fold is not None:
        raise ValueError("query moments require full outer-training provenance")
    train_ids = {_normalize_row_id(value) for value in provenance.train_row_ids}
    heldout_ids = {_normalize_row_id(value) for value in provenance.heldout_row_ids}
    return train_ids, heldout_ids


def _validate_registered_partition(
    registered_fit_row_ids: Sequence[Hashable] | None,
    registered_heldout_row_ids: Sequence[Hashable] | None,
    *,
    train_ids: set[Hashable],
    heldout_ids: set[Hashable],
) -> None:
    if registered_fit_row_ids is None and registered_heldout_row_ids is None:
        return
    if registered_fit_row_ids is None or registered_heldout_row_ids is None:
        raise ValueError("both registered fit and heldout row IDs are required")
    fit = tuple(_normalize_row_id(value) for value in registered_fit_row_ids)
    heldout = tuple(_normalize_row_id(value) for value in registered_heldout_row_ids)
    if len(fit) != len(set(fit)) or len(heldout) != len(set(heldout)):
        raise ValueError("registered partition contains duplicate row IDs")
    if set(fit) != train_ids or set(heldout) != heldout_ids:
        raise ValueError("registered artifact partition does not match fold provenance")


def _sanitize_query_records(
    query_rows: Sequence[Any],
    *,
    train_ids: set[Hashable],
    heldout_ids: set[Hashable],
    config: QueryMomentEvidenceAdapterConfig,
) -> tuple[list[dict[str, Any]], set[Hashable]]:
    if len(query_rows) > int(config.max_queries):
        raise ValueError("query-evidence artifact exceeds max_queries")
    output: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    evidence_ids: set[str] = set()
    cited_rows: set[Hashable] = set()
    for query_index, raw_query in enumerate(query_rows):
        path = f"query_evidence[{query_index}]"
        if not isinstance(raw_query, Mapping):
            raise TypeError(f"{path} must be an object")
        unexpected = set(raw_query) - _QUERY_FIELDS
        if unexpected:
            raise ValueError(f"{path} contains unsupported fields: {sorted(unexpected)}")
        query_id = str(raw_query.get("query_id") or "").strip()
        if not query_id or len(query_id) > 100:
            raise ValueError(f"{path}.query_id must contain 1-100 characters")
        if query_id in query_ids:
            raise ValueError("query IDs must be unique")
        query_ids.add(query_id)
        bank = str(raw_query.get("bank") or "").strip().lower()
        if bank not in _BANKS:
            raise ValueError(f"{path}.bank must be one of {_BANKS}")
        expected_role = _ROLE_BY_BANK[bank]
        supplied_role = raw_query.get("mechanical_role")
        if supplied_role is not None and str(supplied_role) != expected_role:
            raise ValueError(f"{path}.mechanical_role does not match its bank")
        if raw_query.get("statistical_gate_applied") not in (None, False):
            raise ValueError(f"{path} cannot contain statistically gated query evidence")
        member_count = _nonnegative_int(
            raw_query.get("member_count", 0),
            name=f"{path}.member_count",
        )
        subfolds = raw_query.get("member_subfolds") or []
        if not isinstance(subfolds, list):
            raise TypeError(f"{path}.member_subfolds must be a list")
        normalized_subfolds = [
            _positive_int(value, name=f"{path}.member_subfolds") for value in subfolds
        ]
        fit_score = _finite_scalar(
            raw_query.get("fit_standardized_score"),
            name=f"{path}.fit_standardized_score",
            allow_none=True,
        )
        raw_chunks = raw_query.get("top_chunks") or []
        if not isinstance(raw_chunks, list):
            raise TypeError(f"{path}.top_chunks must be a list")
        if len(raw_chunks) > int(config.max_chunks_per_query):
            raise ValueError(f"{path}.top_chunks exceeds the configured bound")
        chunks: list[dict[str, Any]] = []
        for chunk_index, raw_chunk in enumerate(raw_chunks):
            chunk_path = f"{path}.top_chunks[{chunk_index}]"
            if not isinstance(raw_chunk, Mapping):
                raise TypeError(f"{chunk_path} must be an object")
            unexpected_chunk = set(raw_chunk) - _CHUNK_FIELDS
            if unexpected_chunk:
                raise ValueError(
                    f"{chunk_path} contains unsupported fields: {sorted(unexpected_chunk)}"
                )
            if "_oci_row_id" in raw_chunk and "row_id" in raw_chunk:
                raise ValueError(f"{chunk_path} contains two row-ID fields")
            if "_oci_row_id" not in raw_chunk and "row_id" not in raw_chunk:
                raise ValueError(f"{chunk_path} lacks an explicit row ID")
            row_id = _normalize_row_id(raw_chunk.get("_oci_row_id", raw_chunk.get("row_id")))
            if row_id in heldout_ids:
                raise ValueError(f"{chunk_path} contains an outer-heldout row")
            if row_id not in train_ids:
                raise ValueError(f"{chunk_path} contains a row outside outer training")
            evidence_id = str(raw_chunk.get("evidence_id") or "").strip()
            if not evidence_id or len(evidence_id) > 180:
                raise ValueError(f"{chunk_path}.evidence_id is missing or too long")
            if evidence_id in evidence_ids:
                raise ValueError("retrieved evidence IDs must be globally unique")
            evidence_ids.add(evidence_id)
            encoded_row = _LEGACY_ROW_IN_EVIDENCE_ID.search(evidence_id)
            if encoded_row is not None and isinstance(row_id, int):
                if int(encoded_row.group(1)) != row_id:
                    raise ValueError(f"{chunk_path}.evidence_id encodes a different row")
            text = str(raw_chunk.get("text", raw_chunk.get("chunk_text", ""))).strip()
            # The historical retrieval builder could emit an empty excerpt
            # when a cached chunk index no longer resolved.  Its row lineage is
            # checked above, but an empty excerpt is not evidence and is safely
            # omitted.  Nonempty oversize excerpts remain a hard failure.
            if not text:
                continue
            if len(text) > int(config.max_excerpt_chars):
                raise ValueError(f"{chunk_path}.text exceeds the configured bound")
            _reject_forbidden_content(text, path=f"{chunk_path}.text")
            similarity = _finite_scalar(
                raw_chunk.get("similarity"),
                name=f"{chunk_path}.similarity",
                allow_none=True,
            )
            artifact_chunk_index = _nonnegative_int(
                raw_chunk.get("chunk_index", 0),
                name=f"{chunk_path}.chunk_index",
            )
            chunk = {
                "evidence_id": evidence_id,
                "_oci_row_id": row_id,
                "chunk_index": artifact_chunk_index,
                "text": text,
            }
            if similarity is not None:
                chunk["similarity"] = similarity
            chunks.append(chunk)
            cited_rows.add(row_id)
        raw_terms = raw_query.get("top_contrastive_ngrams") or []
        if not isinstance(raw_terms, list):
            raise TypeError(f"{path}.top_contrastive_ngrams must be a list")
        if len(raw_terms) > int(config.max_terms_per_query):
            raise ValueError(f"{path}.top_contrastive_ngrams exceeds the configured bound")
        terms: list[dict[str, Any]] = []
        for term_index, raw_term in enumerate(raw_terms):
            term_path = f"{path}.top_contrastive_ngrams[{term_index}]"
            row = raw_term if isinstance(raw_term, Mapping) else {"term": raw_term}
            unexpected_term = set(row) - _TERM_FIELDS
            if unexpected_term:
                raise ValueError(
                    f"{term_path} contains unsupported fields: {sorted(unexpected_term)}"
                )
            term = str(row.get("term") or "").strip()
            if not term or len(term) > int(config.max_term_chars):
                raise ValueError(f"{term_path}.term is empty or exceeds the bound")
            _reject_forbidden_content(term, path=f"{term_path}.term")
            term_record: dict[str, Any] = {"term": term}
            for field_name in _TERM_FIELDS - {"term"}:
                if row.get(field_name) is not None:
                    term_record[field_name] = _finite_scalar(
                        row[field_name],
                        name=f"{term_path}.{field_name}",
                    )
            terms.append(term_record)
        if not chunks and not terms:
            raise ValueError(f"{path} contains neither retrieved chunks nor terms")
        query_record: dict[str, Any] = {
            "query_id": query_id,
            "bank": bank,
            "mechanical_role": expected_role,
            "statistical_gate_applied": False,
            "member_count": member_count,
            "member_subfolds": normalized_subfolds,
            "top_chunks": chunks,
            "top_contrastive_ngrams": terms,
        }
        if fit_score is not None:
            query_record["fit_standardized_score"] = fit_score
        output.append(query_record)
    return output, cited_rows


def _extract_sparse_query_definitions(
    payload: Mapping[str, Any],
    *,
    config: QueryMomentEvidenceAdapterConfig,
) -> tuple[list[_SparseQueryDefinition], dict[str, Any]]:
    discovery = (
        payload.get("discovery") if isinstance(payload.get("discovery"), Mapping) else payload
    )
    banks = discovery.get("topic_banks") if isinstance(discovery, Mapping) else None
    if not isinstance(banks, Mapping):
        raise ValueError("TF-IDF evidence is missing discovery.topic_banks")
    definitions: list[_SparseQueryDefinition] = []
    seen: set[tuple[str, tuple[tuple[str, float], ...]]] = set()
    skipped_terms = 0
    rejected_over_capacity_definitions = 0

    def add(bank: str, raw_terms: Any) -> None:
        nonlocal skipped_terms, rejected_over_capacity_definitions
        terms = _definition_terms(raw_terms, config=config)
        if terms is None:
            skipped_terms += 1
            return
        key = (bank, terms)
        if key in seen:
            return
        if len(definitions) >= int(config.max_queries):
            rejected_over_capacity_definitions += 1
            raise ValueError(
                "sparse query definitions exceed configured max_queries; "
                "refusing silent definition omission"
            )
        seen.add(key)
        definitions.append(_SparseQueryDefinition(bank=bank, terms=terms))

    for bank in _BANKS:
        bank_payload = banks.get(bank)
        topics = bank_payload.get("topics") if isinstance(bank_payload, Mapping) else None
        if not isinstance(topics, (list, tuple)):
            continue
        for topic in topics:
            if not isinstance(topic, Mapping):
                skipped_terms += 1
                continue
            add(bank, topic.get("terms"))

    orphan = _find_orphan_branch(discovery)
    if isinstance(orphan, Mapping):
        selected_ids = {str(value) for value in orphan.get("selected_cluster_ids") or []}
        clusters = orphan.get("selected_clusters") or orphan.get("clusters") or []
        if isinstance(clusters, (list, tuple)):
            for cluster in clusters:
                if not isinstance(cluster, Mapping):
                    skipped_terms += 1
                    continue
                cluster_id = str(cluster.get("cluster_id") or cluster.get("topic_id") or "")
                if selected_ids and cluster_id not in selected_ids:
                    continue
                add(
                    "effect",
                    cluster.get("terms")
                    or cluster.get("member_terms")
                    or cluster.get("supporting_terms"),
                )
    return definitions, {
        "usable_definition_count": len(definitions),
        "skipped_or_empty_definition_count": int(skipped_terms),
        "definition_limit_reached_count": int(
            rejected_over_capacity_definitions
        ),
        "definitions_truncated": False,
        "uses_supplied_terms_only": True,
        "uses_fixed_domain_vocabulary": False,
        "uses_label_scores_as_query_weights": False,
    }


def _definition_terms(
    raw_terms: Any,
    *,
    config: QueryMomentEvidenceAdapterConfig,
) -> tuple[tuple[str, float], ...] | None:
    if not isinstance(raw_terms, (list, tuple)):
        return None
    if len(raw_terms) > int(config.max_terms_per_query):
        raise ValueError(
            "sparse query definition exceeds configured max_terms_per_query; "
            "refusing silent term omission"
        )
    weights: dict[str, float] = {}
    for raw_term in raw_terms:
        row = raw_term if isinstance(raw_term, Mapping) else {"term": raw_term}
        term = _canonical_term(row.get("term") or row.get("feature") or row.get("ngram"))
        if (
            not term
            or len(term) > int(config.max_term_chars)
            or len(term.split()) > int(config.max_ngram_tokens)
            or _FORBIDDEN_STRING.search(term)
        ):
            continue
        loading = row.get("loading")
        try:
            weight = abs(float(loading)) if loading is not None else 1.0
        except (TypeError, ValueError):
            continue
        if not math.isfinite(weight):
            continue
        weights[term] = max(weights.get(term, 0.0), weight)
    if not weights:
        return None
    if not any(weight > 0.0 for weight in weights.values()):
        weights = {term: 1.0 for term in weights}
    return tuple(sorted(weights.items(), key=lambda item: (-item[1], item[0])))


def _find_orphan_branch(discovery: Mapping[str, Any]) -> Any:
    direct = discovery.get("effect_orphan_ngram_branch")
    if isinstance(direct, Mapping):
        return direct
    for key in ("topic_score_tests", "topic_score_selection", "score_tests"):
        nested = discovery.get(key)
        if isinstance(nested, Mapping) and isinstance(
            nested.get("effect_orphan_ngram_branch"), Mapping
        ):
            return nested["effect_orphan_ngram_branch"]
    return None


def _term_centered_excerpt(text: str, terms: Sequence[str], *, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", str(text)).strip()
    if len(value) <= max_chars:
        return value
    match_start = None
    lowered = value.lower()
    for term in terms:
        tokens = term.split()
        pattern = r"(?<![a-z0-9])" + r"\W+".join(map(re.escape, tokens)) + r"(?![a-z0-9])"
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match is not None and (match_start is None or match.start() < match_start):
            match_start = match.start()
    if match_start is None:
        return value[:max_chars]
    start = max(0, int(match_start) - max_chars // 3)
    stop = min(len(value), start + max_chars)
    start = max(0, stop - max_chars)
    return value[start:stop]


def _canonical_term(value: Any) -> str:
    return " ".join(token.lower() for token in _TOKEN.findall(str(value or "")))


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _normalize_row_id(value: Any) -> Hashable:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError("query-evidence row IDs must be integer or string JSON scalars")
    if isinstance(value, str) and not value:
        raise ValueError("query-evidence row IDs cannot be empty strings")
    return value


def _finite_scalar(value: Any, *, name: str, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _positive_int(value: Any, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result < 1:
        raise ValueError(f"{name} must contain positive integers")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if result < 0 or float(value) != result:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _reject_forbidden_content(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if _FORBIDDEN_KEY.search(key):
                raise ValueError(f"forbidden target/oracle field at {path}.{key}")
            _reject_forbidden_content(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_content(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and _FORBIDDEN_STRING.search(value):
        raise ValueError(f"forbidden target/oracle string at {path}")


def _project_json_values(path: Path, prefix: str) -> list[Any]:
    """Materialize only one allowlisted JSON prefix from a rich artifact."""

    try:
        import ijson
    except ImportError as exc:  # pragma: no cover - available in the project env
        raise RuntimeError("ijson is required for safe neural-query artifact resealing") from exc
    with path.open("rb") as handle:
        return list(ijson.items(handle, prefix))


def _project_single_json_value(path: Path, prefix: str, *, label: str) -> Any:
    values = _project_json_values(path, prefix)
    if len(values) != 1:
        raise ValueError(f"{label} must occur exactly once")
    return values[0]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


__all__ = [
    "NEURAL_QUERY_EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "QUERY_MOMENT_ADAPTER_SCHEMA_VERSION",
    "AdaptedQueryMomentEvidence",
    "QueryMomentEvidenceAdapterConfig",
    "adapt_query_moment_evidence",
    "derive_sparse_query_moment_evidence",
    "load_query_moment_evidence_artifact",
    "reseal_legacy_neural_query_moment_evidence",
]
