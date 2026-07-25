"""Two-phase role-neutral TF-IDF execution for one physical scope group.

This module is intentionally not wired into the all-ten-family worker yet.
It establishes a genuine fit/view boundary for the two native TF-IDF concept
families:

* ``tfidf_topics``; and
* the configured ``residual_tfidf_ngrams`` profile, represented by the
  existing ``tfidf_orphan_ngrams`` catalog family.

The physical owner is fit once from its complete text/treatment/outcome rows.
Fit state and both family payloads are sealed before the callable capable of
opening exact-inner held-out text is invoked.  Cumulative-review logical views
are immutable references to that fit-only state and never receive sealed text.
The exact-inner view is transformed only after the loader boundary.  No
held-out label argument exists.

All executable model state uses the closed JSON/per-array-NPY TF-IDF format.
Text is passed to vectorizers as complete strings; this module has no slicing,
token budget, excerpt count, or implicit feature cap.  Every scientific limit
comes from the supplied typed configuration.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import AppliedInferenceConfig
from .all_evidence_discovery_interfaces import (
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPICS,
)
from .bow_native_proof_capture import _text_sha256
from .lossless_stage1_evidence_catalog import (
    NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
)
from .production_stage1_legacy_scope_fragments import (
    LEGACY_STAGE1_FIT_ONLY_FAMILY_SEAL_SCHEMA,
    build_role_neutral_fit_only_family_seal,
)
from .production_stage1_scope_scheduler import Stage1ScopePlan, Stage1ScopeSpec
from .role_neutral_bow_group_execution import (
    _canonical_json,
    _float_hex_sha256,
    _require_sha256,
    _row_order_fingerprint,
    _sha256_file,
    _sha256_json,
    _tree_sha256,
    _write_new_bytes,
    _write_new_json,
    _write_new_npy,
)
from .tfidf_safe_artifacts import (
    INDEX_FILENAME,
    load_fitted_topic_context,
    safe_artifact_content_sha256,
)
from .tfidf_topic_discovery import stable_hash
from .tfidf_topic_stage1 import _fit_tfidf_topic_context_nested_calibration


ROLE_NEUTRAL_TFIDF_GROUP_REQUEST_SCHEMA = (
    "production_role_neutral_tfidf_physical_group_request_v1"
)
ROLE_NEUTRAL_TFIDF_FIT_STATE_SCHEMA = (
    "production_role_neutral_tfidf_fit_state_v1"
)
ROLE_NEUTRAL_TFIDF_LOGICAL_VIEW_SCHEMA = (
    "production_role_neutral_tfidf_logical_view_v1"
)
ROLE_NEUTRAL_TFIDF_GROUP_EXECUTION_SCHEMA = (
    "production_role_neutral_tfidf_group_execution_v1"
)
ROLE_NEUTRAL_TFIDF_SCIENTIFIC_PLAN_SCHEMA = (
    "production_role_neutral_tfidf_scientific_fit_plan_v1"
)

_FIT_STATE_DIRECTORY = "fit_state"
_FIT_STATE_METADATA = "metadata.json"
_FITTED_CONTEXT_DIRECTORY = "fitted_context"
_FIT_TOPIC_VALUES = "fit_topic_values.npy"
_FIT_RESIDUAL_VALUES = "fit_residual_tfidf_values.npy"
_LOGICAL_VIEW_DIRECTORY = "logical_views"
_TERMINAL_FILE = "execution_manifest.json"
_FAMILY_ORDER = (TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS)
_FAMILY_PROFILE = {
    TFIDF_TOPICS: "tfidf_topics",
    TFIDF_ORPHAN_NGRAMS: "residual_tfidf_ngrams",
}
_FIT_SEAL_FILES = {
    TFIDF_TOPICS: "fit_only_tfidf_topics_family_seal.json",
    TFIDF_ORPHAN_NGRAMS: "fit_only_residual_tfidf_ngrams_family_seal.json",
}


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    target = Path(path)
    digest, size = _sha256_file(target)
    del digest, size
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _binary_vector(values: Sequence[Any], *, label: str, length: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (int(length),) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be one finite vector aligned to fit rows")
    if not set(np.unique(result)).issubset({0.0, 1.0}):
        raise ValueError(f"{label} must be binary")
    return result


def _closed_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(key): _closed_json_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_closed_json_value(child) for child in value]
    raise TypeError(f"unsupported TF-IDF evidence value: {type(value).__name__}")


@dataclass(frozen=True)
class RoleNeutralTfidfPhysicalGroupRequest:
    """Path/device-neutral authority for one TF-IDF physical fit."""

    plan_scientific_content_sha256: str
    physical_owner: Stage1ScopeSpec
    logical_members: tuple[Stage1ScopeSpec, ...]
    content_sha256: str
    authority_plan: Stage1ScopePlan = field(repr=False, compare=False)

    @classmethod
    def from_plan(
        cls,
        *,
        plan: Stage1ScopePlan,
        physical_owner_scope_id: str,
    ) -> "RoleNeutralTfidfPhysicalGroupRequest":
        if not isinstance(plan, Stage1ScopePlan):
            raise TypeError("role-neutral TF-IDF request requires a Stage1ScopePlan")
        owner = plan.scope(str(physical_owner_scope_id))
        if plan.physical_owner(owner.scope_id).scope_id != owner.scope_id:
            raise ValueError("role-neutral TF-IDF request must name a physical owner")
        matches = [
            members
            for candidate, members in plan.physical_scope_groups
            if candidate.scope_id == owner.scope_id
        ]
        if len(matches) != 1:
            raise RuntimeError("TF-IDF physical owner has no unique logical group")
        members = matches[0]
        if members[0].scope_id != owner.scope_id:
            raise RuntimeError("TF-IDF physical group changed canonical owner order")
        if any(
            tuple(member.fit_row_ids) != tuple(owner.fit_row_ids)
            or int(member.scope_seed) != int(owner.scope_seed)
            for member in members
        ):
            raise ValueError(
                "TF-IDF physical reuse changed ordered fit rows or group seed"
            )
        aliases = members[1:]
        if aliases and (
            owner.scope_kind != "exact_inner"
            or any(member.scope_kind != "cumulative_spent" for member in aliases)
        ):
            raise ValueError("TF-IDF reuse supports exact-inner/cumulative groups only")
        body = cls._body(
            plan_scientific_content_sha256=plan.scientific_content_sha256,
            owner=owner,
            members=members,
        )
        return cls(
            plan_scientific_content_sha256=plan.scientific_content_sha256,
            physical_owner=owner,
            logical_members=members,
            content_sha256=_sha256_json(body),
            authority_plan=plan,
        )

    @staticmethod
    def _body(
        *,
        plan_scientific_content_sha256: str,
        owner: Stage1ScopeSpec,
        members: Sequence[Stage1ScopeSpec],
    ) -> dict[str, Any]:
        return {
            "schema_version": ROLE_NEUTRAL_TFIDF_GROUP_REQUEST_SCHEMA,
            "plan_scientific_content_sha256": plan_scientific_content_sha256,
            "physical_owner": owner.as_dict(),
            "logical_members": [member.as_dict() for member in members],
            "logical_scope_count": len(members),
            "fit_row_ids": list(owner.fit_row_ids),
            "fit_row_order_fingerprint": _row_order_fingerprint(owner.fit_row_ids),
            "canonical_group_seed": int(owner.scope_seed),
            "families": list(_FAMILY_ORDER),
            "profile_by_family": copy.deepcopy(_FAMILY_PROFILE),
            "heldout_labels_supplied": False,
            "peer_group_definitions_supplied": False,
        }

    def as_dict(self) -> dict[str, Any]:
        _require_sha256(
            self.plan_scientific_content_sha256,
            label="role-neutral TF-IDF scientific plan identity",
        )
        if (
            not self.logical_members
            or self.logical_members[0].scope_id != self.physical_owner.scope_id
            or len({member.scope_id for member in self.logical_members})
            != len(self.logical_members)
            or any(
                tuple(member.fit_row_ids)
                != tuple(self.physical_owner.fit_row_ids)
                or int(member.scope_seed) != int(self.physical_owner.scope_seed)
                for member in self.logical_members
            )
        ):
            raise ValueError("role-neutral TF-IDF logical authority changed")
        body = self._body(
            plan_scientific_content_sha256=self.plan_scientific_content_sha256,
            owner=self.physical_owner,
            members=self.logical_members,
        )
        if _sha256_json(body) != self.content_sha256:
            raise RuntimeError("role-neutral TF-IDF request content changed")
        return {**body, "content_sha256": self.content_sha256}


def _configuration(
    config: AppliedInferenceConfig,
    *,
    request: RoleNeutralTfidfPhysicalGroupRequest,
) -> dict[str, Any]:
    if not isinstance(config, AppliedInferenceConfig):
        raise TypeError("TF-IDF physical fit requires AppliedInferenceConfig")
    if str(config.outcome_type).strip().lower() != "binary":
        raise ValueError("role-neutral TF-IDF v1 supports only a binary outcome")
    if not hasattr(config, "seed"):
        raise ValueError("role-neutral TF-IDF requires an explicit workflow seed")
    forest = config.architecture.multi_model_forest
    topic = forest.tfidf_topic
    if (
        str(topic.score_selection_label_policy) != "nested_fit_calibration"
        or not bool(topic.score_test_enabled)
    ):
        raise ValueError(
            "role-neutral TF-IDF requires configured nested_fit_calibration score testing"
        )
    return {
        "outcome_type": "binary",
        "configured_columns": {
            "text": str(config.text_column),
            "treatment": str(config.treatment_column),
            "outcome": str(config.outcome_column),
        },
        "workflow_seed": int(config.seed),
        "canonical_group_seed": int(request.physical_owner.scope_seed),
        "scope_seed_application": (
            "nested_split_and_tfidf_random_state_use_canonical_group_seed_v1"
        ),
        "bow_views": [asdict(view) for view in forest.bow_views],
        "nuisance_folds": int(forest.nuisance_folds),
        "tfidf_nested_calibration_folds": int(
            forest.tfidf_nested_calibration_folds
        ),
        "tfidf_topic": asdict(topic),
        "text_input_policy": "complete_strings_no_slicing_or_truncation_v1",
        "text_truncation_applied": False,
        "implicit_feature_or_topic_caps_added_by_executor": False,
    }


def _producer_identity() -> str:
    import oci.config as config_module
    import oci.inference.production_stage1_legacy_scope_fragments as seal_module
    import oci.inference.role_neutral_bow_group_execution as layout_module
    import oci.inference.tfidf_safe_artifacts as safe_module
    import oci.inference.tfidf_topic_discovery as discovery_module
    import oci.inference.tfidf_topic_score_selection as score_module
    import oci.inference.tfidf_topic_stage1 as stage1_module

    functions = (
        _configuration,
        _topic_payload,
        _residual_payload_and_terms,
        _transform_families,
        execute_role_neutral_tfidf_physical_group,
        replay_role_neutral_tfidf_exact_transform,
    )
    modules = (
        config_module,
        seal_module,
        layout_module,
        safe_module,
        discovery_module,
        score_module,
        stage1_module,
    )
    module_hashes = {
        str(module.__name__): hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
        for module in modules
    }
    return _sha256_json(
        {
            "schema_version": "production_role_neutral_tfidf_producer_identity_v1",
            "producer_module_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "function_sources": [inspect.getsource(function) for function in functions],
            "dependency_module_sha256": module_hashes,
        }
    )


def _scientific_fit_plan(
    *,
    request: RoleNeutralTfidfPhysicalGroupRequest,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": ROLE_NEUTRAL_TFIDF_SCIENTIFIC_PLAN_SCHEMA,
        "plan_scientific_content_sha256": request.plan_scientific_content_sha256,
        "group_request_content_sha256": request.content_sha256,
        "physical_owner_scope_id": request.physical_owner.scope_id,
        "physical_owner_scope_sha256": request.physical_owner.as_dict()["scope_sha256"],
        "fit_row_ids": list(request.physical_owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(
            request.physical_owner.fit_row_ids
        ),
        "canonical_group_seed": int(request.physical_owner.scope_seed),
        "configuration_identity_sha256": _sha256_json(configuration),
        "producer_identity_sha256": _producer_identity(),
        "families": list(_FAMILY_ORDER),
        "physical_fit_count": 1,
        "text_truncation_applied": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _copy_safe_context(source_index: Path, destination: Path) -> Path:
    safe_artifact_content_sha256(source_index)
    source_root = source_index.parent
    destination.mkdir(parents=True, exist_ok=False)
    for source in sorted(source_root.iterdir(), key=lambda path: path.name):
        if source.is_symlink() or not source.is_file():
            raise ValueError("fitted TF-IDF safe state contains a linked/non-file entry")
        _write_new_bytes(destination / source.name, source.read_bytes())
    index = destination / INDEX_FILENAME
    safe_artifact_content_sha256(index)
    return index


def _topic_payload(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    evidence: list[dict[str, Any]] = []
    columns: list[str] = []
    banks = metadata.get("topic_banks")
    if not isinstance(banks, Mapping):
        raise ValueError("TF-IDF fit has no topic-bank metadata")
    for bank_name in ("treatment", "outcome", "effect"):
        bank = banks.get(bank_name) or {}
        topics = bank.get("topics") or []
        if not isinstance(topics, list):
            raise ValueError("TF-IDF topic metadata is malformed")
        for position, topic in enumerate(topics):
            topic_id = str(
                topic.get("topic_id")
                or f"{bank_name}_topic_{position + 1:03d}"
            )
            columns.append(f"{bank_name}::{topic_id}")
            terms = topic.get("terms") or []
            if not terms:
                evidence.append(
                    {
                        "bank": bank_name,
                        "topic_id": topic_id,
                        "witness_kind": "fitted_topic_without_rendered_terms",
                    }
                )
            for term_position, term in enumerate(terms):
                evidence.append(
                    {
                        "bank": bank_name,
                        "topic_id": topic_id,
                        "topic_position": int(position),
                        "term_position": int(term_position),
                        "witness_kind": "fitted_consensus_nmf_topic_term",
                        **_closed_json_value(dict(term)),
                    }
                )
    if not evidence:
        evidence.append(
            {
                "witness_kind": "no_feasible_fitted_topic",
                "reason": "fit_side_topic_model_returned_zero_components",
            }
        )
    return (
        {
            "schema_version": NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
            "family": TFIDF_TOPICS,
            "architecture_evidence": evidence,
        },
        columns,
    )


def _residual_payload_and_terms(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("TF-IDF fit has no artifact registry")
    scores_path = Path(str((artifacts.get("ngram_scores") or {}).get("effect") or ""))
    scores = pd.read_parquet(scores_path)
    if "feature" not in scores or "eligible" not in scores:
        raise ValueError("TF-IDF effect score artifact lacks residual candidate fields")
    represented = {
        str(term.get("term"))
        for topic in (
            ((metadata.get("topic_banks") or {}).get("effect") or {}).get("topics")
            or []
        )
        for term in (topic.get("terms") or [])
        if str(term.get("term") or "").strip()
    }
    candidates = scores[
        scores["eligible"].fillna(False).astype(bool)
        & ~scores["feature"].astype(str).isin(represented)
    ].copy()
    terms: list[str] = []
    evidence: list[dict[str, Any]] = []
    evidence_fields = (
        "feature",
        "moment",
        "robust_se",
        "signed_score",
        "unsigned_score",
        "support_control",
        "support_treated",
        "nuisance_source_agreement",
        "subsample_selection_stability",
        "subsample_sign_agreement",
        "tail_contrast_sign_agreement",
        "combined_importance",
        "eligible",
    )
    for fit_rank, (_index, row) in enumerate(candidates.iterrows(), start=1):
        term = str(row["feature"])
        if not term or term in terms:
            continue
        terms.append(term)
        evidence.append(
            {
                "witness_kind": "fit_side_residual_tfidf_ngram",
                "fit_rank": int(fit_rank),
                "represented_in_effect_topic": False,
                **{
                    field: _closed_json_value(row[field])
                    for field in evidence_fields
                    if field in row
                },
            }
        )
    if not evidence:
        evidence.append(
            {
                "witness_kind": "no_eligible_residual_tfidf_ngram",
                "reason": (
                    "all_fit_side_effect_ngrams_were_ineligible_or_represented_by_topics"
                ),
            }
        )
    return (
        {
            "schema_version": NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
            "family": TFIDF_ORPHAN_NGRAMS,
            "architecture_evidence": evidence,
        },
        terms,
    )


def _topic_matrix(
    *,
    fitted: Any,
    texts: Sequence[str],
    expected_columns: Sequence[str],
) -> np.ndarray:
    transformed = fitted.transform_topics(tuple(texts))
    arrays: list[np.ndarray] = []
    columns: list[str] = []
    for bank_name in ("treatment", "outcome", "effect"):
        values = np.asarray(
            transformed.get(bank_name, np.zeros((len(texts), 0))),
            dtype=np.float64,
        )
        if values.ndim != 2 or values.shape[0] != len(texts):
            raise ValueError("TF-IDF topic transform returned an invalid row shape")
        bank = fitted.topic_banks.get(bank_name)
        topic_terms = [] if bank is None else bank.topic_terms
        for position in range(values.shape[1]):
            topic_id = f"{bank_name}_topic_{position + 1:03d}"
            columns.append(f"{bank_name}::{topic_id}")
            arrays.append(values[:, position])
        if bank is not None and len(topic_terms) != values.shape[1]:
            raise ValueError("TF-IDF live topic model and rendered metadata disagree")
    # Metadata topic IDs currently use the same deterministic IDs. Keeping the
    # equality explicit prevents a future naming change from silently
    # reordering numerical columns.
    if columns != list(expected_columns):
        raise ValueError("TF-IDF topic transform column contract changed")
    return (
        np.column_stack(arrays).astype(np.float64, copy=False)
        if arrays
        else np.empty((len(texts), 0), dtype=np.float64)
    )


def _residual_matrix(
    *,
    fitted: Any,
    texts: Sequence[str],
    terms: Sequence[str],
) -> np.ndarray:
    vocabulary = fitted.common_vectorizer.vocabulary_
    missing = [term for term in terms if term not in vocabulary]
    if missing:
        raise ValueError(f"sealed residual TF-IDF terms left the vocabulary: {missing[:3]}")
    if not terms:
        return np.empty((len(texts), 0), dtype=np.float64)
    columns = [int(vocabulary[term]) for term in terms]
    values = fitted.common_vectorizer.transform(tuple(texts))[:, columns]
    result = np.asarray(values.toarray(), dtype=np.float64)
    if result.shape != (len(texts), len(terms)) or not np.isfinite(result).all():
        raise ValueError("residual TF-IDF transform returned invalid values")
    return result


def _transform_families(
    *,
    fitted: Any,
    texts: Sequence[str],
    topic_columns: Sequence[str],
    residual_terms: Sequence[str],
) -> dict[str, tuple[list[str], np.ndarray]]:
    return {
        TFIDF_TOPICS: (
            list(topic_columns),
            _topic_matrix(
                fitted=fitted,
                texts=texts,
                expected_columns=topic_columns,
            ),
        ),
        TFIDF_ORPHAN_NGRAMS: (
            [f"residual_tfidf::{term}" for term in residual_terms],
            _residual_matrix(
                fitted=fitted,
                texts=texts,
                terms=residual_terms,
            ),
        ),
    }


def _array_registration(path: Path, *, relative_to: Path, columns: Sequence[str]) -> dict[str, Any]:
    with path.open("rb") as handle:
        values = np.load(handle, allow_pickle=False, mmap_mode=None)
    if values.dtype.hasobject or values.ndim != 2:
        raise ValueError("TF-IDF family array must be one non-object matrix")
    digest, size = _sha256_file(path)
    return {
        "relative_path": path.relative_to(relative_to).as_posix(),
        "sha256": digest,
        "size_bytes": size,
        "dtype": values.dtype.str,
        "shape": [int(item) for item in values.shape],
        "columns": list(columns),
    }


def _fit_seal(
    *,
    request: RoleNeutralTfidfPhysicalGroupRequest,
    family: str,
    payload: Mapping[str, Any],
    fit_state_sha256: str,
    scientific_plan: Mapping[str, Any],
) -> dict[str, Any]:
    if family not in _FAMILY_ORDER:
        raise ValueError("TF-IDF fit seal names another family")
    if (
        request.authority_plan.scientific_content_sha256
        != request.plan_scientific_content_sha256
    ):
        raise RuntimeError("TF-IDF request plan authority changed")
    return build_role_neutral_fit_only_family_seal(
        plan=request.authority_plan,
        physical_owner_scope_id=request.physical_owner.scope_id,
        family=family,
        evidence_payload=payload,
        producer_identity_sha256=str(scientific_plan["producer_identity_sha256"]),
        configuration_identity_sha256=str(
            scientific_plan["configuration_identity_sha256"]
        ),
        fit_state_artifact_sha256=fit_state_sha256,
    )


def _expected_execution_events(
    *,
    request: RoleNeutralTfidfPhysicalGroupRequest,
    scientific_plan_sha256: str,
    fit_state_sha256: str,
) -> list[dict[str, Any]]:
    """Derive the only valid fit/seal/view event sequence."""

    owner = request.physical_owner
    events: list[dict[str, Any]] = [
        {
            "sequence": 1,
            "event": "fit_completed",
            "families": list(_FAMILY_ORDER),
            "scientific_fit_plan_sha256": scientific_plan_sha256,
            "fit_state_artifact_sha256": fit_state_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        }
    ]
    for family in _FAMILY_ORDER:
        events.append(
            {
                "sequence": len(events) + 1,
                "event": "fit_family_artifact_sealed",
                "family": family,
                "fit_state_artifact_sha256": fit_state_sha256,
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
            }
        )
    for member in request.logical_members[1:]:
        for family in _FAMILY_ORDER:
            events.append(
                {
                    "sequence": len(events) + 1,
                    "event": "cumulative_fit_only_view_published",
                    "logical_scope_id": member.scope_id,
                    "family": family,
                    "registered_heldout_text_accessed": False,
                    "registered_heldout_labels_accessed": False,
                }
            )
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "exact_heldout_text_opened",
            "logical_scope_id": owner.scope_id,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
        }
    )
    for family in _FAMILY_ORDER:
        events.extend(
            [
                {
                    "sequence": len(events) + 1,
                    "event": "exact_heldout_transform_completed",
                    "logical_scope_id": owner.scope_id,
                    "family": family,
                    "registered_heldout_text_accessed": True,
                    "registered_heldout_labels_accessed": False,
                },
                {
                    "sequence": len(events) + 2,
                    "event": "exact_logical_view_published",
                    "logical_scope_id": owner.scope_id,
                    "family": family,
                    "registered_heldout_text_accessed": True,
                    "registered_heldout_labels_accessed": False,
                },
            ]
        )
    return events


def _validate_array(
    *,
    root: Path,
    registration: Mapping[str, Any],
    expected_rows: int,
    expected_relative_path: str,
) -> np.ndarray:
    path = root / str(registration.get("relative_path"))
    digest, size = _sha256_file(path)
    values = np.load(path, allow_pickle=False, mmap_mode="r")
    if (
        str(registration.get("relative_path")) != str(expected_relative_path)
        or digest != registration.get("sha256")
        or size != registration.get("size_bytes")
        or values.dtype.str != registration.get("dtype")
        or list(values.shape) != registration.get("shape")
        or values.ndim != 2
        or values.shape[0] != int(expected_rows)
        or values.shape[1] != len(registration.get("columns") or [])
        or values.dtype.hasobject
    ):
        raise ValueError("TF-IDF registered array failed validation")
    return values


def _validate_fit_side(
    *,
    root: Path,
    request: RoleNeutralTfidfPhysicalGroupRequest,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Any]:
    fit_root = root / _FIT_STATE_DIRECTORY
    metadata = _read_json(
        fit_root / _FIT_STATE_METADATA,
        label="role-neutral TF-IDF fit metadata",
    )
    body = {key: copy.deepcopy(value) for key, value in metadata.items() if key != "content_sha256"}
    required = {
        "schema_version",
        "group_request_content_sha256",
        "scientific_fit_plan",
        "fit_row_ids",
        "fit_row_order_fingerprint",
        "fit_text_sha256",
        "fit_treatment_sha256",
        "fit_outcome_sha256",
        "configuration",
        "configuration_identity_sha256",
        "producer_identity_sha256",
        "fitted_context",
        "fit_family_arrays",
        "topic_columns",
        "residual_terms",
        "array_layout",
        "model_state_layout",
        "registered_heldout_text_accessed",
        "registered_heldout_labels_accessed",
        "oracle_fields_accessed",
        "text_truncation_applied",
        "content_sha256",
    }
    if (
        set(metadata) != required
        or metadata.get("schema_version") != ROLE_NEUTRAL_TFIDF_FIT_STATE_SCHEMA
        or metadata.get("group_request_content_sha256") != request.content_sha256
        or metadata.get("fit_row_ids") != list(request.physical_owner.fit_row_ids)
        or metadata.get("fit_row_order_fingerprint")
        != _row_order_fingerprint(request.physical_owner.fit_row_ids)
        or metadata.get("registered_heldout_text_accessed") is not False
        or metadata.get("registered_heldout_labels_accessed") is not False
        or metadata.get("oracle_fields_accessed") is not False
        or metadata.get("text_truncation_applied") is not False
        or metadata.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("role-neutral TF-IDF fit metadata is invalid")
    plan = metadata["scientific_fit_plan"]
    configuration = metadata["configuration"]
    if (
        not isinstance(configuration, Mapping)
        or metadata["configuration_identity_sha256"]
        != _sha256_json(configuration)
    ):
        raise ValueError("role-neutral TF-IDF configuration identity changed")
    expected_plan = _scientific_fit_plan(
        request=request,
        configuration=configuration,
    )
    if (
        plan != expected_plan
        or plan.get("schema_version")
        != ROLE_NEUTRAL_TFIDF_SCIENTIFIC_PLAN_SCHEMA
        or plan.get("producer_identity_sha256")
        != metadata["producer_identity_sha256"]
    ):
        raise ValueError("role-neutral TF-IDF scientific fit plan is invalid")
    fitted_index = fit_root / str(metadata["fitted_context"]["relative_path"])
    index_digest, index_size = _sha256_file(fitted_index)
    if (
        metadata["fitted_context"].get("relative_path")
        != f"{_FITTED_CONTEXT_DIRECTORY}/{INDEX_FILENAME}"
        or fitted_index.name != INDEX_FILENAME
        or index_digest != metadata["fitted_context"]["sha256"]
        or index_size != metadata["fitted_context"]["size_bytes"]
        or safe_artifact_content_sha256(fitted_index)
        != metadata["fitted_context"]["transitive_content_sha256"]
    ):
        raise ValueError("role-neutral TF-IDF fitted context registration changed")
    fitted = load_fitted_topic_context(fitted_index)
    arrays = metadata["fit_family_arrays"]
    if not isinstance(arrays, Mapping) or set(arrays) != set(_FAMILY_ORDER):
        raise ValueError("role-neutral TF-IDF fit arrays are incomplete")
    _validate_array(
        root=fit_root,
        registration=arrays[TFIDF_TOPICS],
        expected_rows=len(request.physical_owner.fit_row_ids),
        expected_relative_path=_FIT_TOPIC_VALUES,
    )
    _validate_array(
        root=fit_root,
        registration=arrays[TFIDF_ORPHAN_NGRAMS],
        expected_rows=len(request.physical_owner.fit_row_ids),
        expected_relative_path=_FIT_RESIDUAL_VALUES,
    )
    expected_fit_entries = {
        _FIT_STATE_METADATA,
        _FITTED_CONTEXT_DIRECTORY,
        _FIT_TOPIC_VALUES,
        _FIT_RESIDUAL_VALUES,
    }
    if {path.name for path in fit_root.iterdir()} != expected_fit_entries:
        raise ValueError("role-neutral TF-IDF fit state contains an extra or missing entry")
    seals: dict[str, dict[str, Any]] = {}
    fit_state_sha = _tree_sha256(fit_root)
    for family in _FAMILY_ORDER:
        seal = _read_json(
            root / _FIT_SEAL_FILES[family],
            label=f"role-neutral TF-IDF {family} seal",
        )
        seal_body = {
            key: copy.deepcopy(value)
            for key, value in seal.items()
            if key != "content_sha256"
        }
        payload = seal.get("evidence_payload")
        expected_seal = (
            build_role_neutral_fit_only_family_seal(
                plan=request.authority_plan,
                physical_owner_scope_id=request.physical_owner.scope_id,
                family=family,
                evidence_payload=payload or {},
                producer_identity_sha256=seal.get(
                    "producer_identity_sha256"
                ),
                configuration_identity_sha256=seal.get(
                    "configuration_identity_sha256"
                ),
                fit_state_artifact_sha256=seal.get(
                    "fit_state_artifact_sha256"
                ),
            )
            if isinstance(payload, Mapping)
            else None
        )
        if (
            seal.get("schema_version") != LEGACY_STAGE1_FIT_ONLY_FAMILY_SEAL_SCHEMA
            or seal.get("family") != family
            or seal.get("physical_owner_scope_id")
            != request.physical_owner.scope_id
            or seal.get("fit_state_artifact_sha256") != fit_state_sha
            or not isinstance(payload, Mapping)
            or payload.get("family") != family
            or payload.get("schema_version")
            != NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION
            or not isinstance(payload.get("architecture_evidence"), list)
            or not payload["architecture_evidence"]
            or seal.get("evidence_payload_sha256") != _sha256_json(payload)
            or seal.get("registered_heldout_text_accessed") is not False
            or seal.get("registered_heldout_labels_accessed") is not False
            or seal.get("content_sha256") != _sha256_json(seal_body)
            or seal != expected_seal
        ):
            raise ValueError(f"role-neutral TF-IDF {family} fit seal is invalid")
        seals[family] = seal
    return metadata, seals, fitted


def _write_prediction(
    *,
    root: Path,
    owner: Stage1ScopeSpec,
    family: str,
    columns: Sequence[str],
    values: np.ndarray,
) -> dict[str, Any]:
    logical_root = root / _LOGICAL_VIEW_DIRECTORY
    path = logical_root / f"{owner.scope_id}.{family}.predictions.npy"
    _write_new_npy(path, values)
    return _array_registration(path, relative_to=root, columns=columns)


def execute_role_neutral_tfidf_physical_group(
    *,
    request: RoleNeutralTfidfPhysicalGroupRequest,
    output_root: Path | str,
    fit_texts: Sequence[str],
    fit_treatment: Sequence[Any],
    fit_outcome: Sequence[Any],
    config: AppliedInferenceConfig,
    exact_heldout_text_loader: Callable[[tuple[int, ...]], Sequence[str]],
) -> Mapping[str, Any]:
    """Fit once, seal both TF-IDF families, then open exact held-out text."""

    if not isinstance(request, RoleNeutralTfidfPhysicalGroupRequest):
        raise TypeError("TF-IDF physical execution requires its typed request")
    request.as_dict()
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("TF-IDF physical output root must be absolute")
    if root.exists() or root.is_symlink():
        raise FileExistsError("TF-IDF physical output root must be fresh")
    if not callable(exact_heldout_text_loader):
        raise TypeError("TF-IDF exact held-out text loader must be callable")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    owner = request.physical_owner
    texts = tuple(fit_texts)
    if len(texts) != len(owner.fit_row_ids) or any(not isinstance(text, str) for text in texts):
        raise ValueError("TF-IDF fit texts must align exactly to owner rows")
    treatment = _binary_vector(
        fit_treatment,
        label="fit treatment",
        length=len(texts),
    )
    outcome = _binary_vector(
        fit_outcome,
        label="fit outcome",
        length=len(texts),
    )
    configuration = _configuration(config, request=request)
    configuration_identity = _sha256_json(configuration)
    scientific_plan = _scientific_fit_plan(
        request=request,
        configuration=configuration,
    )

    fit_config = copy.deepcopy(config)
    fit_config.text_column = "_role_neutral_text"
    fit_config.treatment_column = "_role_neutral_treatment"
    fit_config.outcome_column = "_role_neutral_outcome"
    fit_config.seed = int(owner.scope_seed)
    fit_config.architecture.multi_model_forest.tfidf_topic.random_state = int(
        owner.scope_seed
    )
    fit_df = pd.DataFrame(
        {
            "_oci_row_id": list(owner.fit_row_ids),
            fit_config.text_column: list(texts),
            fit_config.treatment_column: treatment,
            fit_config.outcome_column: outcome,
        }
    )
    # The existing nested fitter validates its transform path and currently
    # requires at least one label-free transform row. Use a deterministic
    # internal probe that is neither a cohort row nor registered held-out text.
    # Its outputs remain in ephemeral scratch and are never published.
    fit_only_transform_probe = pd.DataFrame(
        {
            "_oci_row_id": [-1],
            fit_config.text_column: [""],
        }
    )
    with tempfile.TemporaryDirectory(
        prefix="oci-role-neutral-tfidf-fit-",
        dir=root.parent,
    ) as temporary:
        scratch = Path(temporary) / "context"
        scratch.mkdir()
        metadata = _fit_tfidf_topic_context_nested_calibration(
            spec={
                "outer_fold": int(owner.outer_fold),
                "inner_fold": owner.inner_fold,
                "scope": owner.scope_kind,
                "scope_id": owner.scope_id,
                "fit_df": fit_df,
                "heldout_df": fit_only_transform_probe,
            },
            config=fit_config,
            artifact_dir=scratch,
        )
        source_index = Path(metadata["artifacts"]["fitted_context"])
        fitted = load_fitted_topic_context(source_index)
        topic_payload, topic_columns = _topic_payload(metadata)
        residual_payload, residual_terms = _residual_payload_and_terms(metadata)
        fit_values = _transform_families(
            fitted=fitted,
            texts=texts,
            topic_columns=topic_columns,
            residual_terms=residual_terms,
        )
        fit_root = root / _FIT_STATE_DIRECTORY
        fit_root.mkdir(exist_ok=False)
        fitted_index = _copy_safe_context(
            source_index,
            fit_root / _FITTED_CONTEXT_DIRECTORY,
        )

    topic_path = fit_root / _FIT_TOPIC_VALUES
    residual_path = fit_root / _FIT_RESIDUAL_VALUES
    _write_new_npy(topic_path, fit_values[TFIDF_TOPICS][1])
    _write_new_npy(residual_path, fit_values[TFIDF_ORPHAN_NGRAMS][1])
    fitted_digest, fitted_size = _sha256_file(fitted_index)
    fitted_registration = {
        "relative_path": fitted_index.relative_to(fit_root).as_posix(),
        "sha256": fitted_digest,
        "size_bytes": fitted_size,
        "transitive_content_sha256": safe_artifact_content_sha256(fitted_index),
    }
    fit_arrays = {
        TFIDF_TOPICS: _array_registration(
            topic_path,
            relative_to=fit_root,
            columns=topic_columns,
        ),
        TFIDF_ORPHAN_NGRAMS: _array_registration(
            residual_path,
            relative_to=fit_root,
            columns=[f"residual_tfidf::{term}" for term in residual_terms],
        ),
    }
    fit_body = {
        "schema_version": ROLE_NEUTRAL_TFIDF_FIT_STATE_SCHEMA,
        "group_request_content_sha256": request.content_sha256,
        "scientific_fit_plan": scientific_plan,
        "fit_row_ids": list(owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(owner.fit_row_ids),
        "fit_text_sha256": _text_sha256(owner.fit_row_ids, texts),
        "fit_treatment_sha256": _float_hex_sha256(treatment),
        "fit_outcome_sha256": _float_hex_sha256(outcome),
        "configuration": configuration,
        "configuration_identity_sha256": configuration_identity,
        "producer_identity_sha256": scientific_plan["producer_identity_sha256"],
        "fitted_context": fitted_registration,
        "fit_family_arrays": fit_arrays,
        "topic_columns": topic_columns,
        "residual_terms": residual_terms,
        "array_layout": "one_nonobject_npy_per_family_matrix_v1",
        "model_state_layout": "closed_json_and_per_array_npy_v2",
        "registered_heldout_text_accessed": False,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
    }
    fit_metadata = {**fit_body, "content_sha256": _sha256_json(fit_body)}
    _write_new_json(fit_root / _FIT_STATE_METADATA, fit_metadata)
    fit_state_sha = _tree_sha256(fit_root)
    payloads = {
        TFIDF_TOPICS: topic_payload,
        TFIDF_ORPHAN_NGRAMS: residual_payload,
    }
    seals = {
        family: _fit_seal(
            request=request,
            family=family,
            payload=payloads[family],
            fit_state_sha256=fit_state_sha,
            scientific_plan=scientific_plan,
        )
        for family in _FAMILY_ORDER
    }
    seal_registrations: dict[str, dict[str, Any]] = {}
    for family in _FAMILY_ORDER:
        path = root / _FIT_SEAL_FILES[family]
        _write_new_json(path, seals[family])
        digest, size = _sha256_file(path)
        seal_registrations[family] = {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "size_bytes": size,
            "content_sha256": seals[family]["content_sha256"],
        }
    _validate_fit_side(root=root, request=request)

    logical_root = root / _LOGICAL_VIEW_DIRECTORY
    logical_root.mkdir(exist_ok=False)
    events: list[dict[str, Any]] = [
        {
            "sequence": 1,
            "event": "fit_completed",
            "families": list(_FAMILY_ORDER),
            "scientific_fit_plan_sha256": scientific_plan["content_sha256"],
            "fit_state_artifact_sha256": fit_state_sha,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        }
    ]
    for family in _FAMILY_ORDER:
        events.append(
            {
                "sequence": len(events) + 1,
                "event": "fit_family_artifact_sealed",
                "family": family,
                "fit_state_artifact_sha256": fit_state_sha,
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
            }
        )
    logical_registrations: list[dict[str, Any]] = []
    for member in request.logical_members[1:]:
        if member.scope_kind != "cumulative_spent":
            raise RuntimeError("TF-IDF physical alias changed logical purpose")
        for family in _FAMILY_ORDER:
            view_body = {
                "schema_version": ROLE_NEUTRAL_TFIDF_LOGICAL_VIEW_SCHEMA,
                "group_request_content_sha256": request.content_sha256,
                "scientific_fit_plan_sha256": scientific_plan["content_sha256"],
                "logical_scope_id": member.scope_id,
                "logical_scope_sha256": member.as_dict()["scope_sha256"],
                "logical_purpose": member.scope_kind,
                "physical_owner_scope_id": owner.scope_id,
                "family": family,
                "scientific_profile": _FAMILY_PROFILE[family],
                "fit_only_family_seal_sha256": seal_registrations[family]["sha256"],
                "fit_only_family_seal_content_sha256": seals[family][
                    "content_sha256"
                ],
                "view_input_policy": "sealed_row_ids_only_no_text_or_labels_v1",
                "logical_heldout_row_ids": list(member.heldout_row_ids),
                "logical_transform_performed": False,
                "prediction_artifact": None,
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
                "reuses_physical_fit_by_reference": True,
            }
            view = {**view_body, "content_sha256": _sha256_json(view_body)}
            path = logical_root / f"{member.scope_id}.{family}.json"
            _write_new_json(path, view)
            digest, size = _sha256_file(path)
            logical_registrations.append(
                {
                    "logical_scope_id": member.scope_id,
                    "family": family,
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": digest,
                    "size_bytes": size,
                    "content_sha256": view["content_sha256"],
                }
            )
            events.append(
                {
                    "sequence": len(events) + 1,
                    "event": "cumulative_fit_only_view_published",
                    "logical_scope_id": member.scope_id,
                    "family": family,
                    "registered_heldout_text_accessed": False,
                    "registered_heldout_labels_accessed": False,
                }
            )

    loaded = exact_heldout_text_loader(tuple(owner.heldout_row_ids))
    heldout_texts = tuple(loaded)
    if len(heldout_texts) != len(owner.heldout_row_ids) or any(
        not isinstance(text, str) for text in heldout_texts
    ):
        raise ValueError("TF-IDF exact held-out loader returned another row/text shape")
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "exact_heldout_text_opened",
            "logical_scope_id": owner.scope_id,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
        }
    )
    replay_fitted = load_fitted_topic_context(fitted_index)
    live_transforms = _transform_families(
        fitted=fitted,
        texts=heldout_texts,
        topic_columns=topic_columns,
        residual_terms=residual_terms,
    )
    replay_transforms = _transform_families(
        fitted=replay_fitted,
        texts=heldout_texts,
        topic_columns=topic_columns,
        residual_terms=residual_terms,
    )
    for family in _FAMILY_ORDER:
        columns, values = live_transforms[family]
        replay_columns, replay_values = replay_transforms[family]
        if columns != replay_columns or not np.allclose(
            values,
            replay_values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RuntimeError("live TF-IDF transform differs from sealed safe state")
        prediction = _write_prediction(
            root=root,
            owner=owner,
            family=family,
            columns=columns,
            values=values,
        )
        events.append(
            {
                "sequence": len(events) + 1,
                "event": "exact_heldout_transform_completed",
                "logical_scope_id": owner.scope_id,
                "family": family,
                "registered_heldout_text_accessed": True,
                "registered_heldout_labels_accessed": False,
            }
        )
        view_body = {
            "schema_version": ROLE_NEUTRAL_TFIDF_LOGICAL_VIEW_SCHEMA,
            "group_request_content_sha256": request.content_sha256,
            "scientific_fit_plan_sha256": scientific_plan["content_sha256"],
            "logical_scope_id": owner.scope_id,
            "logical_scope_sha256": owner.as_dict()["scope_sha256"],
            "logical_purpose": owner.scope_kind,
            "physical_owner_scope_id": owner.scope_id,
            "family": family,
            "scientific_profile": _FAMILY_PROFILE[family],
            "fit_only_family_seal_sha256": seal_registrations[family]["sha256"],
            "fit_only_family_seal_content_sha256": seals[family]["content_sha256"],
            "view_input_policy": "heldout_row_ids_and_complete_text_no_labels_v1",
            "logical_heldout_row_ids": list(owner.heldout_row_ids),
            "logical_heldout_text_sha256": _text_sha256(
                owner.heldout_row_ids,
                heldout_texts,
            ),
            "logical_transform_performed": True,
            "prediction_artifact": prediction,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
            "reuses_live_physical_fit": True,
            "model_state_reloaded_for_primary_transform": False,
            "sealed_state_replay_checked": True,
        }
        view = {**view_body, "content_sha256": _sha256_json(view_body)}
        path = logical_root / f"{owner.scope_id}.{family}.json"
        _write_new_json(path, view)
        digest, size = _sha256_file(path)
        logical_registrations.append(
            {
                "logical_scope_id": owner.scope_id,
                "family": family,
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "size_bytes": size,
                "content_sha256": view["content_sha256"],
            }
        )
        events.append(
            {
                "sequence": len(events) + 1,
                "event": "exact_logical_view_published",
                "logical_scope_id": owner.scope_id,
                "family": family,
                "registered_heldout_text_accessed": True,
                "registered_heldout_labels_accessed": False,
            }
        )
    logical_registrations.sort(
        key=lambda row: (
            next(
                position
                for position, member in enumerate(request.logical_members)
                if member.scope_id == row["logical_scope_id"]
            ),
            _FAMILY_ORDER.index(row["family"]),
        )
    )
    terminal_body = {
        "schema_version": ROLE_NEUTRAL_TFIDF_GROUP_EXECUTION_SCHEMA,
        "status": "complete",
        "group_request": request.as_dict(),
        "scientific_fit_plan_sha256": scientific_plan["content_sha256"],
        "families": list(_FAMILY_ORDER),
        "profile_by_family": copy.deepcopy(_FAMILY_PROFILE),
        "physical_fit_count": 1,
        "fit_state_artifact_sha256": fit_state_sha,
        "fit_only_family_seals": seal_registrations,
        "logical_views": logical_registrations,
        "event_order": events,
        "fit_completed_before_registered_heldout_text_access": True,
        "fit_sealed_before_registered_heldout_text_access": True,
        "cumulative_views_published_by_reference_without_sealed_text": True,
        "live_model_objects_reused_for_exact_transform": True,
        "model_state_reloaded_for_primary_transform": False,
        "sealed_safe_state_replay_checked": True,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
        "pickle_or_joblib_loaded": False,
        "compressed_npz_loaded_or_written": False,
        "all_ten_family_adapter_enabled": False,
    }
    terminal = {**terminal_body, "content_sha256": _sha256_json(terminal_body)}
    _write_new_json(root / _TERMINAL_FILE, terminal)
    return validate_role_neutral_tfidf_group_execution(
        root=root,
        request=request,
    )


def replay_role_neutral_tfidf_exact_transform(
    *,
    root: Path | str,
    request: RoleNeutralTfidfPhysicalGroupRequest,
    exact_heldout_texts: Sequence[str],
) -> Mapping[str, Any]:
    """Freshly replay both exact transforms from safe JSON/NPY state."""

    source = Path(root)
    if source.is_symlink():
        raise ValueError("TF-IDF replay root cannot be a symlink")
    tree = source.resolve(strict=True)
    metadata, _seals, fitted = _validate_fit_side(root=tree, request=request)
    texts = tuple(exact_heldout_texts)
    if len(texts) != len(request.physical_owner.heldout_row_ids) or any(
        not isinstance(text, str) for text in texts
    ):
        raise ValueError("TF-IDF replay text does not align to exact held-out rows")
    transformed = _transform_families(
        fitted=fitted,
        texts=texts,
        topic_columns=metadata["topic_columns"],
        residual_terms=metadata["residual_terms"],
    )
    return {
        "family_predictions": {
            family: {
                "columns": columns,
                "predictions": values,
            }
            for family, (columns, values) in transformed.items()
        },
        "scientific_fit_plan_sha256": metadata["scientific_fit_plan"][
            "content_sha256"
        ],
        "fit_state_artifact_sha256": _tree_sha256(tree / _FIT_STATE_DIRECTORY),
        "state_source": "authenticated_json_and_npy_only",
        "live_model_objects_available": False,
        "pickle_or_joblib_loaded": False,
        "compressed_npz_loaded": False,
        "text_truncation_applied": False,
    }


def validate_role_neutral_tfidf_group_execution(
    *,
    root: Path | str,
    request: RoleNeutralTfidfPhysicalGroupRequest,
) -> Mapping[str, Any]:
    """Fresh path-only validation of one completed TF-IDF physical group."""

    source = Path(root)
    if source.is_symlink():
        raise ValueError("TF-IDF execution root cannot be a symlink")
    tree = source.resolve(strict=True)
    if not tree.is_dir():
        raise ValueError("TF-IDF execution root must be a directory")
    metadata, seals, _fitted = _validate_fit_side(root=tree, request=request)
    terminal = _read_json(
        tree / _TERMINAL_FILE,
        label="role-neutral TF-IDF terminal manifest",
    )
    body = {key: copy.deepcopy(value) for key, value in terminal.items() if key != "content_sha256"}
    required = {
        "schema_version",
        "status",
        "group_request",
        "scientific_fit_plan_sha256",
        "families",
        "profile_by_family",
        "physical_fit_count",
        "fit_state_artifact_sha256",
        "fit_only_family_seals",
        "logical_views",
        "event_order",
        "fit_completed_before_registered_heldout_text_access",
        "fit_sealed_before_registered_heldout_text_access",
        "cumulative_views_published_by_reference_without_sealed_text",
        "live_model_objects_reused_for_exact_transform",
        "model_state_reloaded_for_primary_transform",
        "sealed_safe_state_replay_checked",
        "registered_heldout_labels_accessed",
        "oracle_fields_accessed",
        "text_truncation_applied",
        "pickle_or_joblib_loaded",
        "compressed_npz_loaded_or_written",
        "all_ten_family_adapter_enabled",
        "content_sha256",
    }
    events = terminal.get("event_order")
    logical_rows = terminal.get("logical_views")
    fit_state_sha = _tree_sha256(tree / _FIT_STATE_DIRECTORY)
    expected_events = _expected_execution_events(
        request=request,
        scientific_plan_sha256=metadata["scientific_fit_plan"][
            "content_sha256"
        ],
        fit_state_sha256=fit_state_sha,
    )
    if (
        set(terminal) != required
        or terminal.get("schema_version") != ROLE_NEUTRAL_TFIDF_GROUP_EXECUTION_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("group_request") != request.as_dict()
        or terminal.get("scientific_fit_plan_sha256")
        != metadata["scientific_fit_plan"]["content_sha256"]
        or terminal.get("families") != list(_FAMILY_ORDER)
        or terminal.get("profile_by_family") != _FAMILY_PROFILE
        or terminal.get("physical_fit_count") != 1
        or terminal.get("fit_state_artifact_sha256") != fit_state_sha
        or any(seal["fit_state_artifact_sha256"] != fit_state_sha for seal in seals.values())
        or not isinstance(logical_rows, list)
        or len(logical_rows) != len(request.logical_members) * len(_FAMILY_ORDER)
        or events != expected_events
        or terminal.get("fit_completed_before_registered_heldout_text_access") is not True
        or terminal.get("fit_sealed_before_registered_heldout_text_access") is not True
        or terminal.get(
            "cumulative_views_published_by_reference_without_sealed_text"
        )
        is not True
        or terminal.get("live_model_objects_reused_for_exact_transform") is not True
        or terminal.get("model_state_reloaded_for_primary_transform") is not False
        or terminal.get("sealed_safe_state_replay_checked") is not True
        or terminal.get("registered_heldout_labels_accessed") is not False
        or terminal.get("oracle_fields_accessed") is not False
        or terminal.get("text_truncation_applied") is not False
        or terminal.get("pickle_or_joblib_loaded") is not False
        or terminal.get("compressed_npz_loaded_or_written") is not False
        or terminal.get("all_ten_family_adapter_enabled") is not False
        or terminal.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("role-neutral TF-IDF terminal manifest is invalid")
    seal_regs = terminal.get("fit_only_family_seals")
    if not isinstance(seal_regs, Mapping) or set(seal_regs) != set(_FAMILY_ORDER):
        raise ValueError("role-neutral TF-IDF seal registry is invalid")
    for family in _FAMILY_ORDER:
        path = tree / _FIT_SEAL_FILES[family]
        digest, size = _sha256_file(path)
        registration = seal_regs[family]
        if (
            not isinstance(registration, Mapping)
            or set(registration)
            != {
                "relative_path",
                "sha256",
                "size_bytes",
                "content_sha256",
            }
            or registration.get("relative_path") != _FIT_SEAL_FILES[family]
            or registration.get("sha256") != digest
            or registration.get("size_bytes") != size
            or registration.get("content_sha256") != seals[family]["content_sha256"]
        ):
            raise ValueError("role-neutral TF-IDF seal registration changed")
    expected_logical_keys = {
        (member.scope_id, family)
        for member in request.logical_members
        for family in _FAMILY_ORDER
    }
    observed_logical_keys = {
        (str(row.get("logical_scope_id")), str(row.get("family")))
        for row in logical_rows
        if isinstance(row, Mapping)
    }
    if observed_logical_keys != expected_logical_keys:
        raise ValueError("role-neutral TF-IDF logical registry is incomplete")
    expected_logical_files: set[str] = set()
    for registration in logical_rows:
        if (
            not isinstance(registration, Mapping)
            or set(registration)
            != {
                "logical_scope_id",
                "family",
                "relative_path",
                "sha256",
                "size_bytes",
                "content_sha256",
            }
        ):
            raise ValueError("role-neutral TF-IDF logical registration is invalid")
        path = tree / str(registration["relative_path"])
        digest, size = _sha256_file(path)
        view = _read_json(path, label="role-neutral TF-IDF logical view")
        view_body = {
            key: copy.deepcopy(value)
            for key, value in view.items()
            if key != "content_sha256"
        }
        family = str(registration["family"])
        scope_id = str(registration["logical_scope_id"])
        member = next(member for member in request.logical_members if member.scope_id == scope_id)
        transformed = scope_id == request.physical_owner.scope_id
        expected_relative_path = (
            f"{_LOGICAL_VIEW_DIRECTORY}/{scope_id}.{family}.json"
        )
        common_view_fields = {
            "schema_version",
            "group_request_content_sha256",
            "scientific_fit_plan_sha256",
            "logical_scope_id",
            "logical_scope_sha256",
            "logical_purpose",
            "physical_owner_scope_id",
            "family",
            "scientific_profile",
            "fit_only_family_seal_sha256",
            "fit_only_family_seal_content_sha256",
            "view_input_policy",
            "logical_heldout_row_ids",
            "logical_transform_performed",
            "prediction_artifact",
            "registered_heldout_text_accessed",
            "registered_heldout_labels_accessed",
            "content_sha256",
        }
        expected_view_fields = common_view_fields | (
            {
                "logical_heldout_text_sha256",
                "reuses_live_physical_fit",
                "model_state_reloaded_for_primary_transform",
                "sealed_state_replay_checked",
            }
            if transformed
            else {"reuses_physical_fit_by_reference"}
        )
        if (
            set(view) != expected_view_fields
            or registration.get("relative_path") != expected_relative_path
            or registration.get("sha256") != digest
            or registration.get("size_bytes") != size
            or registration.get("content_sha256") != view.get("content_sha256")
            or view.get("content_sha256") != _sha256_json(view_body)
            or view.get("family") != family
            or view.get("logical_scope_id") != scope_id
            or view.get("logical_scope_sha256") != member.as_dict()["scope_sha256"]
            or view.get("schema_version")
            != ROLE_NEUTRAL_TFIDF_LOGICAL_VIEW_SCHEMA
            or view.get("group_request_content_sha256")
            != request.content_sha256
            or view.get("scientific_fit_plan_sha256")
            != metadata["scientific_fit_plan"]["content_sha256"]
            or view.get("logical_purpose") != member.scope_kind
            or view.get("physical_owner_scope_id")
            != request.physical_owner.scope_id
            or view.get("scientific_profile") != _FAMILY_PROFILE[family]
            or view.get("fit_only_family_seal_sha256")
            != seal_regs[family]["sha256"]
            or view.get("fit_only_family_seal_content_sha256")
            != seals[family]["content_sha256"]
            or view.get("logical_heldout_row_ids")
            != list(member.heldout_row_ids)
            or view.get("logical_transform_performed") is not transformed
            or view.get("registered_heldout_text_accessed") is not transformed
            or view.get("registered_heldout_labels_accessed") is not False
            or (not transformed and view.get("prediction_artifact") is not None)
            or (
                transformed
                and (
                    view.get("view_input_policy")
                    != "heldout_row_ids_and_complete_text_no_labels_v1"
                    or view.get("reuses_live_physical_fit") is not True
                    or view.get("model_state_reloaded_for_primary_transform")
                    is not False
                    or view.get("sealed_state_replay_checked") is not True
                )
            )
            or (
                not transformed
                and (
                    view.get("view_input_policy")
                    != "sealed_row_ids_only_no_text_or_labels_v1"
                    or view.get("reuses_physical_fit_by_reference") is not True
                )
            )
        ):
            raise ValueError("role-neutral TF-IDF logical view is invalid")
        expected_logical_files.add(path.name)
        if transformed:
            prediction = view.get("prediction_artifact")
            values = _validate_array(
                root=tree,
                registration=prediction,
                expected_rows=len(request.physical_owner.heldout_row_ids),
                expected_relative_path=(
                    f"{_LOGICAL_VIEW_DIRECTORY}/{scope_id}.{family}."
                    "predictions.npy"
                ),
            )
            del values
            expected_logical_files.add(
                Path(str(prediction["relative_path"])).name
            )
    logical_root = tree / _LOGICAL_VIEW_DIRECTORY
    if logical_root.is_symlink() or not logical_root.is_dir():
        raise ValueError("role-neutral TF-IDF logical directory is linked or missing")
    if {path.name for path in logical_root.iterdir()} != expected_logical_files:
        raise ValueError("role-neutral TF-IDF logical directory is not closed")
    expected_root = {
        _FIT_STATE_DIRECTORY,
        _LOGICAL_VIEW_DIRECTORY,
        _TERMINAL_FILE,
        *set(_FIT_SEAL_FILES.values()),
    }
    if {path.name for path in tree.iterdir()} != expected_root:
        raise ValueError("role-neutral TF-IDF execution tree is not closed")
    return terminal
