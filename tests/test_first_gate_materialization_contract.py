from __future__ import annotations

import copy
import inspect
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
from oci.inference.coordinate_preserving_context_fit_upstream_backend import (
    CoordinatePreservingContextFitUpstreamBackend,
)
from oci.inference.direct_upstream_numerical_manifest import (
    SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON,
)
from oci.inference.first_gate_materialization_contract import (
    FIRST_GATE_DEFERRED_MATERIALIZATION_BOUNDARY,
    FirstGateMaterializationIntent,
    FirstGateMaterializationRealizationAttestation,
    prepare_first_gate_materialization_intent,
)
from oci.inference.first_untouched_gate_direct_numerical_preparation import (
    prepare_first_untouched_gate_direct_numerical,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
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
    identity = {
        "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
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
        _audit_json="{}",
    )
    validate_role_neutral_catalog(result)
    return result


def _schema():
    return build_production_coordinate_preserving_schema(
        namespace="first_gate_intent_test",
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
        return {"backend": "complete_first_gate_intent_test_child_v1"}

    def fit_predict(self, **kwargs):
        self.calls.append(kwargs)
        rows = len(tuple(kwargs["gate_row_ids"]))
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
        return ContextFitUpstreamPrediction(
            gate_row_ids=tuple(kwargs["gate_row_ids"]),
            calibrated_source_names=source_names,
            calibrated_source_kinds=source_kinds,
            calibrated_source_values=np.arange(rows * len(source_names), dtype=float).reshape(
                rows, len(source_names)
            ),
            feature_names=tuple(feature_names),
            feature_kinds=tuple(feature_kinds),
            feature_roles=tuple(feature_roles),
            feature_values=np.arange(rows * len(feature_names), dtype=float).reshape(
                rows, len(feature_names)
            ),
        )


def _provider(tmp_path: Path):
    config = _schema()
    child = _CompleteChildBackend(config)
    backend = CoordinatePreservingContextFitUpstreamBackend(child, config=config)
    return ContextFitUpstreamGateProvider(tmp_path / "gate_cache", backend=backend), child


def _intent(tmp_path: Path, provider, **overrides):
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
    }
    kwargs.update(overrides)
    return prepare_first_gate_materialization_intent(**kwargs)


def _materialize(tmp_path: Path, provider):
    return prepare_first_untouched_gate_direct_numerical(
        outer_fold=1,
        initial_spent_row_ids=(1, 2, 3, 4),
        initial_spent_texts=("spent a", "spent b", "spent c", "spent d"),
        initial_spent_treatment=(0.0, 1.0, 0.0, 1.0),
        initial_spent_outcome=(0.1, 0.4, 0.2, 0.8),
        initial_spent_inner_fold_ids=(1, 2, 1, 2),
        first_gate_row_ids=(8, 9),
        first_gate_texts=("gate a", "gate b"),
        catalog=_catalog(),
        provider=provider,
        destination=tmp_path / "direct" / "direct_upstream_numerical_manifest.json",
    )


def test_intent_is_pure_exact_and_covers_all_architectures(tmp_path: Path) -> None:
    provider, child = _provider(tmp_path)
    assert not provider.cache_dir.exists()

    intent = _intent(tmp_path, provider)

    assert child.calls == []
    assert not provider.cache_dir.exists()
    assert provider._prepared == {}
    intent.verify()
    round_trip = FirstGateMaterializationIntent.from_dict(intent.as_dict())
    assert round_trip == intent
    body = intent.body
    assert body["materialization_boundary"]["boundary"] == (
        FIRST_GATE_DEFERRED_MATERIALIZATION_BOUNDARY
    )
    assert body["source_cache_key"] == content_sha256(body["exact_cache_binding"])
    assert body["exact_cache_binding"]["gate_labels_in_binding"] is False
    assert body["exact_cache_binding"]["gate_labels_exposed_to_backend"] is False
    assert [row["source_family"] for row in body["coordinate_schema"]["family_coverage"]] == list(
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    zero_rows = [
        row for row in body["coordinate_schema"]["family_coverage"] if row["numerical_zero_reason"]
    ]
    assert zero_rows == [
        {
            **next(
                row
                for row in body["coordinate_schema"]["family_coverage"]
                if row["source_family"] == TFIDF_SEMANTIC_RETRIEVAL
            ),
            "numerical_zero_reason": SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON,
        }
    ]
    assert all(
        row["coordinate_ids"]
        for row in body["coordinate_schema"]["family_coverage"]
        if row["source_family"] != TFIDF_SEMANTIC_RETRIEVAL
    )
    assert body["assurances"]["placeholder_matrix_or_value_hashes_used"] is False
    assert "source_manifest_sha256" not in body


def test_realized_manifest_audit_and_bound_provider_match_intent(tmp_path: Path) -> None:
    provider, child = _provider(tmp_path)
    intent = _intent(tmp_path, provider)
    prepared = _materialize(tmp_path, provider)

    assert len(child.calls) == 3
    structural_only = intent.verify_realization(prepared.persisted_manifest.manifest)
    assert structural_only.body["exact_intent_match"] is True
    assert structural_only.body["bound_source_manifest_bytes_verified"] is False
    assert structural_only.body["source_matrix_and_column_values_reauthenticated"] is False
    attestation = intent.verify_realization(
        prepared.persisted_manifest.manifest,
        preparation_audit=prepared.audit,
        bound_provider=prepared.bound_provider,
    )
    assert isinstance(attestation, FirstGateMaterializationRealizationAttestation)
    assert attestation.body["intent_content_sha256"] == intent.content_sha256
    assert attestation.body["exact_intent_match"] is True
    assert attestation.body["unknown_pre_fit_matrix_and_value_hashes_now_authenticated"] is True
    assert (
        FirstGateMaterializationRealizationAttestation.from_dict(attestation.as_dict())
        == attestation
    )

    changed_schema = replace(
        prepared.persisted_manifest.manifest,
        stable_output_schema_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="stable_output_schema_sha256"):
        intent.verify_realization(changed_schema)
    changed_audit = copy.deepcopy(prepared.audit)
    changed_audit["initial_spent_binding"]["outcome_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="initial-spent outcome_sha256"):
        intent.verify_realization(
            prepared.persisted_manifest.manifest,
            preparation_audit=changed_audit,
        )


def test_api_has_no_gate_labels_and_exact_hashes_detect_input_or_packet_changes(
    tmp_path: Path,
) -> None:
    parameters = inspect.signature(prepare_first_gate_materialization_intent).parameters
    assert not {
        "first_gate_treatment",
        "first_gate_outcome",
        "gate_treatment",
        "gate_outcome",
    } & set(parameters)
    provider, child = _provider(tmp_path)
    first = _intent(tmp_path, provider)
    second = _intent(
        tmp_path,
        provider,
        initial_spent_outcome=(0.1, 0.4, 0.2, 0.81),
    )
    assert first.content_sha256 != second.content_sha256
    assert first.body["source_cache_key"] != second.body["source_cache_key"]
    assert child.calls == []

    changed = first.as_dict()
    changed["body"]["input_bindings"]["first_untouched_gate"]["text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content SHA-256 mismatch"):
        FirstGateMaterializationIntent.from_dict(changed)

    with pytest.raises(ValueError, match="must be disjoint"):
        _intent(tmp_path, provider, first_gate_row_ids=(4, 9))
