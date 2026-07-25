from __future__ import annotations

import inspect
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    OUTCOME_AXIS,
    TFIDF_SEMANTIC_RETRIEVAL,
    canonical_json,
    content_sha256,
)
from oci.inference.all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from oci.inference.context_fit_upstream_gate_provider import (
    ContextFitUpstreamGateProvider,
    ContextFitUpstreamPrediction,
)
from oci.inference.context_fit_upstream_cache_overlay import (
    CONTEXT_FIT_CACHE_INDEX_SCHEMA_VERSION,
    AuthenticatedContextFitGateCacheOverlay,
    authenticate_context_fit_cache_index_registrations,
)
from oci.inference.coordinate_preserving_context_fit_upstream_backend import (
    CoordinatePreservingContextFitUpstreamBackend,
)
from oci.inference.direct_upstream_numerical_manifest import (
    SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON,
)
from oci.inference.first_untouched_gate_direct_numerical_preparation import (
    FirstUntouchedGatePreparationBounds,
    prepare_first_untouched_gate_direct_numerical,
)
from oci.inference.final_context_fit_upstream_bank import (
    FinalContextFitUpstreamProducer,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
    SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION,
    RoleNeutralEvidenceCatalog,
    Stage1EvidenceAtom,
    validate_role_neutral_catalog,
)
from oci.inference.production_coordinate_preserving_upstream_schema import (
    build_production_coordinate_preserving_schema,
)

_VIEWS = (
    "linear_unigram_c0p5",
    "linear_1_2",
    "linear_1_3",
    "linear_2_4_min_df3",
    "extratrees_1_3",
    "random_forest_1_2",
)
_SEMANTIC_MEMBER_BATCH_SIZE = 3


def _catalog(*, outer_fold: int = 1, scope: str = "inner_train"):
    split_fingerprint = "1" * 64
    atoms = []
    for ordinal, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES, start=1):
        member_id = f"member_{ordinal:03d}"
        origin = {"source": f"closed_{ordinal:03d}"}
        content = {
            "terms": [
                {
                    "member_id": member_id,
                    "term": f"documented clinical clue {ordinal:03d}",
                }
            ]
        }
        origin_sha = content_sha256(origin)
        content_sha = content_sha256(content)
        identity = {
            "atom_kind": "test_term_atom",
            "source_kind": f"closed_source_{ordinal:03d}",
            "source_family": family,
            "observable_axes": (OUTCOME_AXIS,),
            "member_ids": (member_id,),
            "split_fingerprint": split_fingerprint,
            "origin_sha256": origin_sha,
            "content_sha256": content_sha,
        }
        atoms.append(
            Stage1EvidenceAtom(
                evidence_id=f"evidence_{content_sha256(identity)}",
                atom_kind="test_term_atom",
                source_kind=f"closed_source_{ordinal:03d}",
                source_family=family,
                observable_axes=(OUTCOME_AXIS,),
                member_ids=(member_id,),
                split_fingerprint=split_fingerprint,
                origin_sha256=origin_sha,
                content_sha256=content_sha,
                _origin_json=canonical_json(origin),
                _content_json=canonical_json(content),
            )
        )
    inner_fold = 1 if scope == "inner_train" else None
    semantic_member_batching = {
        "schema_version": SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION,
        "semantic_member_batch_size": _SEMANTIC_MEMBER_BATCH_SIZE,
        "selection_or_truncation_authorized": False,
        "complete_member_coverage_required": True,
    }
    identity = {
        "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
        "semantic_member_batching": semantic_member_batching,
        "outer_fold": outer_fold,
        "scope": scope,
        "inner_fold": inner_fold,
        "split_fingerprint": split_fingerprint,
        "atoms": [atom.as_dict() for atom in atoms],
        "non_grounding_numerical_summaries": [],
    }
    result = RoleNeutralEvidenceCatalog(
        outer_fold=outer_fold,
        scope=scope,
        inner_fold=inner_fold,
        split_fingerprint=split_fingerprint,
        atoms=tuple(atoms),
        non_grounding_numerical_summaries=(),
        catalog_sha256=content_sha256(identity),
        _audit_json=canonical_json(
            {
                "semantic_member_batching": semantic_member_batching,
                "semantic_member_batch_size": _SEMANTIC_MEMBER_BATCH_SIZE,
                "semantic_member_batches_truncated": False,
            }
        ),
    )
    validate_role_neutral_catalog(result)
    return result


def _schema():
    return build_production_coordinate_preserving_schema(
        namespace="first_gate_test",
        bow_view_names=_VIEWS,
        source_config_sha256="f" * 64,
        cluster_max_components=1,
        tfidf_topic_count=1,
        max_orphan_features=1,
        neural_query_counts={"treatment": 1, "outcome": 1, "effect": 1},
    )


class _CompleteChildBackend:
    def __init__(self, config) -> None:
        self.config = config
        self.calls: list[dict[str, object]] = []

    def identity(self):
        return {"backend": "complete_first_gate_test_child_v1"}

    def fit_predict(self, **kwargs):
        self.calls.append(kwargs)
        gate_ids = tuple(kwargs["gate_row_ids"])
        rows = len(gate_ids)
        source_names = tuple(row.child_name for row in self.config.calibrated_sources)
        source_kinds = tuple(row.source_kind for row in self.config.calibrated_sources)

        named = tuple(row for row in self.config.named_raw_coordinates if row.required)
        feature_names = [row.child_name for row in named]
        feature_kinds = [row.source_kind for row in named]
        feature_roles = [row.consumer_role for row in named]
        volatile = (
            (
                "stage1_raw__embedding__cluster_confounder_treatment_pc1__mean_cosine__as_propensity",
                "embedding_clustered",
                PROPENSITY_NUISANCE_FEATURE_ROLE,
            ),
            (
                "stage1_raw__embedding__cluster_effect_residualized_interaction_pc1__mean_cosine",
                "embedding_clustered",
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            ),
            (
                "tfidf_treatment_topic_001",
                "tfidf_topics",
                PROPENSITY_NUISANCE_FEATURE_ROLE,
            ),
            (
                "tfidf_outcome_topic_001",
                "tfidf_topics",
                OUTCOME_NUISANCE_FEATURE_ROLE,
            ),
            (
                "tfidf_effect_topic_001",
                "tfidf_topic_contrast",
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            ),
            (
                "tfidf_orphan_001_aaaaaaaaaaaa",
                "tfidf_orphan_ngrams",
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            ),
        )
        feature_names.extend(row[0] for row in volatile)
        feature_kinds.extend(row[1] for row in volatile)
        feature_roles.extend(row[2] for row in volatile)
        source_values = np.arange(rows * len(source_names), dtype=float).reshape(
            rows, len(source_names)
        )
        feature_values = np.arange(rows * len(feature_names), dtype=float).reshape(
            rows, len(feature_names)
        )
        return ContextFitUpstreamPrediction(
            gate_row_ids=gate_ids,
            calibrated_source_names=source_names,
            calibrated_source_kinds=source_kinds,
            calibrated_source_values=source_values,
            feature_names=tuple(feature_names),
            feature_kinds=tuple(feature_kinds),
            feature_roles=tuple(feature_roles),
            feature_values=feature_values,
        )


class _CountingProvider(ContextFitUpstreamGateProvider):
    def __init__(self, cache_dir: Path, *, backend) -> None:
        super().__init__(cache_dir, backend=backend)
        self.bind_calls: list[dict[str, object]] = []

    def bind_fold(self, **kwargs):
        self.bind_calls.append(kwargs)
        return super().bind_fold(**kwargs)


class _IdentityDriftingProvider(_CountingProvider):
    def __init__(self, cache_dir: Path, *, backend) -> None:
        self.identity_epoch = 1
        super().__init__(cache_dir, backend=backend)

    def identity(self):
        return {**super().identity(), "test_identity_epoch": self.identity_epoch}

    def bind_fold(self, **kwargs):
        bound = super().bind_fold(**kwargs)
        self.identity_epoch += 1
        return bound


def _provider(tmp_path: Path, *, drifting: bool = False):
    config = _schema()
    child = _CompleteChildBackend(config)
    backend = CoordinatePreservingContextFitUpstreamBackend(child, config=config)
    provider_type = _IdentityDriftingProvider if drifting else _CountingProvider
    return provider_type(tmp_path / "gate_cache", backend=backend), child


def _prepare(tmp_path: Path, provider, **overrides):
    kwargs = {
        "outer_fold": 1,
        "initial_spent_row_ids": (1, 2, 3, 4),
        "initial_spent_texts": ("spent a", "spent b", "spent c", "spent d"),
        "initial_spent_treatment": (0.0, 1.0, 0.0, 1.0),
        "initial_spent_outcome": (0.1, 0.4, 0.2, 0.8),
        "initial_spent_inner_fold_ids": (1, 2, 1, 2),
        "first_gate_row_ids": (8, 9),
        "first_gate_texts": ("gate a", "gate b"),
        "catalog": _catalog(),
        "provider": provider,
        "destination": tmp_path / "direct" / "direct_upstream_numerical_manifest.json",
    }
    kwargs.update(overrides)
    return prepare_first_untouched_gate_direct_numerical(**kwargs)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_record(identity):
    detached = json.loads(canonical_json(identity))
    return {
        "identity": detached,
        "identity_sha256": content_sha256(detached),
    }


def _authenticated_gate_source(
    tmp_path: Path,
    *,
    provider: ContextFitUpstreamGateProvider,
    final_producer: FinalContextFitUpstreamProducer,
    cache_manifest_path: Path,
):
    companion_body = {
        "runner_schema_version": "all_evidence_fusion_outer_runner_v20",
        "post_extraction_review_providers": {
            "calibrated_gate_sources": _identity_record(provider.identity()),
            "role_aware_gate_feature_banks": _identity_record(provider.identity()),
        },
        "final_upstream_model_inputs": {
            "producer": _identity_record(final_producer.identity()),
        },
    }
    companion = tmp_path / "immutable_input_manifest.json"
    companion_payload = {
        "schema_version": companion_body["runner_schema_version"],
        "body": companion_body,
        "content_sha256": content_sha256(companion_body),
    }
    companion.write_text(
        json.dumps(companion_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cache = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    content = {
        "schema_version": CONTEXT_FIT_CACHE_INDEX_SCHEMA_VERSION,
        "entries": [
            {
                "kind": "review_gate",
                "cache_manifest_path": str(cache_manifest_path),
                "cache_manifest_sha256": _file_sha256(cache_manifest_path),
                "cache_files": {
                    cache["source_values_file"]: cache["source_values_sha256"],
                    cache["feature_values_file"]: cache["feature_values_sha256"],
                    cache["source_context_values_file"]: cache["source_context_values_sha256"],
                    cache["feature_context_values_file"]: cache["feature_context_values_sha256"],
                },
                "run_manifest_path": str(companion),
                "run_manifest_sha256": _file_sha256(companion),
            }
        ],
    }
    index = tmp_path / "context_fit_cache_index.json"
    index_payload = {**content, "content_sha256": content_sha256(content)}
    index.write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return authenticate_context_fit_cache_index_registrations([f"{index}::{_file_sha256(index)}"])


def test_prepares_exact_v3_manifest_once_without_gate_labels_and_returns_bound_view(
    tmp_path: Path,
) -> None:
    provider, child = _provider(tmp_path)
    result = _prepare(tmp_path, provider)

    assert len(provider.bind_calls) == 1
    bind = provider.bind_calls[0]
    assert set(bind) == {
        "outer_fold",
        "context",
        "context_texts",
        "gate_texts",
        "exact_gate_row_ids",
    }
    assert tuple(bind["context"].extracted.columns) == ("_oci_row_id",)
    assert bind["context"].extracted["_oci_row_id"].tolist() == [1, 2, 3, 4]
    assert all("gate_treatment" not in call and "gate_outcome" not in call for call in child.calls)
    assert len(child.calls) == 3  # two context OOF fits and one label-free gate fit
    gate_call = child.calls[-1]
    assert tuple(gate_call["gate_row_ids"]) == (8, 9)
    assert tuple(gate_call["gate_texts"]) == ("gate a", "gate b")

    manifest = result.persisted_manifest.manifest
    assert manifest.source_cache_schema == "context_fit_upstream_gate_cache_v6"
    assert manifest.semantic_catalog_sha256 == _catalog().catalog_sha256
    assert tuple(row.source_family for row in manifest.family_coverage) == (
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        assert manifest.family(family).semantic_atom_ids == tuple(
            atom.evidence_id for atom in _catalog().family_atoms(family)
        )
    assert {row.source_family for row in manifest.family_coverage if row.numerical_zero_reason} == {
        TFIDF_SEMANTIC_RETRIEVAL
    }
    assert manifest.family(TFIDF_SEMANTIC_RETRIEVAL).numerical_zero_reason == (
        SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON
    )

    feature_view = result.bound_provider.get_gate_feature_bank_view(
        outer_fold=1, exact_gate_row_ids=(8, 9)
    )
    assert feature_view.row_ids == (8, 9)
    assert feature_view.values.shape[0] == 2
    result.verify()
    assert result.audit["upstream_cache_binding"]["bind_fold_invocation_count"] == 1
    assert result.audit["first_untouched_gate_binding"]["treatment_accepted"] is False
    assert result.audit["first_untouched_gate_binding"]["outcome_accepted"] is False
    assert result.audit["assurances"]["raw_matrix_values_exposed_to_discovery"] is False


def test_public_api_has_no_gate_label_parameters() -> None:
    parameters = inspect.signature(prepare_first_untouched_gate_direct_numerical).parameters
    assert "first_gate_treatment" not in parameters
    assert "first_gate_outcome" not in parameters
    assert "gate_treatment" not in parameters
    assert "gate_outcome" not in parameters


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"catalog": _catalog(outer_fold=2)}, "catalog outer fold"),
        ({"catalog": _catalog(scope="outer_train")}, "spent-only catalog"),
        ({"first_gate_row_ids": (4, 9)}, "must be disjoint"),
        ({"initial_spent_outcome": (0.1, 0.2)}, "one-dimensional with length"),
        (
            {"bounds": replace(FirstUntouchedGatePreparationBounds(), max_initial_spent_rows=3)},
            "row count exceeds",
        ),
        (
            {
                "bounds": replace(
                    FirstUntouchedGatePreparationBounds(),
                    max_total_text_utf8_bytes=1,
                )
            },
            "text exceeds",
        ),
    ),
)
def test_prebind_mismatches_and_bounds_fail_before_provider_call(
    tmp_path: Path, overrides, message: str
) -> None:
    provider, _child = _provider(tmp_path)
    with pytest.raises((TypeError, ValueError), match=message):
        _prepare(tmp_path, provider, **overrides)
    assert provider.bind_calls == []


def test_provider_identity_drift_fails_after_the_single_bind_without_sidecar(
    tmp_path: Path,
) -> None:
    provider, _child = _provider(tmp_path, drifting=True)
    destination = tmp_path / "direct" / "direct_upstream_numerical_manifest.json"

    with pytest.raises(ValueError, match="identity changed during first-gate bind"):
        _prepare(tmp_path, provider, destination=destination)

    assert len(provider.bind_calls) == 1
    assert not destination.exists()


def test_bound_cache_tampering_invalidates_returned_preparation(tmp_path: Path) -> None:
    provider, _child = _provider(tmp_path)
    result = _prepare(tmp_path, provider)
    source_manifest = result.bound_provider.authenticated_cache_manifest_path
    source_manifest.write_bytes(source_manifest.read_bytes() + b" ")

    with pytest.raises(ValueError, match="manifest changed"):
        result.verify()


def test_manifest_replay_is_immutable_and_identical(tmp_path: Path) -> None:
    first_provider, _first_child = _provider(tmp_path)
    first = _prepare(tmp_path, first_provider)
    payload = json.loads(first.persisted_manifest.path.read_text(encoding="utf-8"))

    second_provider, second_child = _provider(tmp_path)
    second = _prepare(tmp_path, second_provider)

    assert second_child.calls == []  # exact authenticated upstream cache replay
    assert json.loads(second.persisted_manifest.path.read_text(encoding="utf-8")) == payload
    assert second.persisted_manifest.file_sha256 == first.persisted_manifest.file_sha256
    assert second.persisted_manifest.manifest.content_sha256 == (
        first.persisted_manifest.manifest.content_sha256
    )


def test_authenticated_overlay_exact_hit_binds_once_without_backend_call(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_provider, _source_child = _provider(source_root)
    source_final = FinalContextFitUpstreamProducer(
        source_root / "final_cache",
        backend=source_provider.backend,
    )
    source = _prepare(source_root, source_provider)
    registrations = _authenticated_gate_source(
        source_root,
        provider=source_provider,
        final_producer=source_final,
        cache_manifest_path=source.bound_provider.authenticated_cache_manifest_path,
    )

    fresh_root = tmp_path / "fresh"
    fresh_root.mkdir()
    config = _schema()
    fresh_child = _CompleteChildBackend(config)
    fresh_backend = CoordinatePreservingContextFitUpstreamBackend(
        fresh_child,
        config=config,
    )
    raw_provider = ContextFitUpstreamGateProvider(
        fresh_root / "gate_cache",
        backend=fresh_backend,
    )
    fresh_final = FinalContextFitUpstreamProducer(
        fresh_root / "final_cache",
        backend=fresh_backend,
    )
    overlay = AuthenticatedContextFitGateCacheOverlay(
        provider=raw_provider,
        runtime_producer=fresh_final,
        sources=registrations,
        output_root=fresh_root,
        hierarchical_first_gate_preparation=True,
    )

    replay = _prepare(fresh_root, overlay)

    assert fresh_child.calls == []
    upstream = replay.audit["upstream_cache_binding"]
    assert upstream["bind_fold_invocation_count"] == 1
    assert upstream["authenticated_overlay_used"] is True
    assert upstream["bind_provider_kind"] == ("authenticated_context_fit_gate_cache_overlay")
    assert upstream["bind_provider_identity"] == overlay.identity()
    assert upstream["raw_delegate_provider_identity"] == raw_provider.identity()
    assert upstream["bind_provider_identity"]["delegate_provider_identity"] == (
        upstream["raw_delegate_provider_identity"]
    )
    assert replay.persisted_manifest.manifest.content_sha256 == (
        source.persisted_manifest.manifest.content_sha256
    )
