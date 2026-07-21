from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.inference.all_evidence_fusion import (
    LEGACY_ALL_SOURCE,
    NEURAL_QUERY_SOURCE,
    TFIDF_TOPIC_SOURCE,
)
from oci.inference.all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    ObservableCausalRows,
)
from oci.inference.authenticated_semantic_retrieval_compatibility import (
    current_spent_projection_compatibility_identity,
    restore_current_spent_projection_semantic_retrieval_view,
)
from oci.inference.context_fit_upstream_cache_overlay import (
    AuthenticatedContextFitGateCacheOverlay,
    authenticate_context_fit_cache_index_registrations,
)
from oci.inference.context_fit_upstream_gate_provider import (
    ContextFitUpstreamGateProvider,
    ContextFitUpstreamPrediction,
)
from oci.inference.final_context_fit_upstream_bank import FinalContextFitUpstreamProducer
from oci.inference.fold_honest_r_stack import FitRowProvenance
from oci.inference.fold_honest_signal_fusion import row_set_fingerprint
from oci.inference.hierarchical_preparation_prefix_salvage import (
    HIERARCHICAL_PREPARATION_FOLD_SCHEMA_VERSION,
    HIERARCHICAL_PREPARATION_INPUT_SCHEMA_VERSION,
    HierarchicalPreparationPrefixSalvageError,
    export_completed_hierarchical_preparation_prefix,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    build_role_neutral_evidence_catalog,
)
from oci.inference.review_spent_evidence_cache_overlay import (
    AuthenticatedReviewSpentEvidenceCacheOverlay,
    authenticate_review_spent_cache_registrations,
)
from oci.inference.review_spent_evidence_provider import (
    ContextFitReviewSpentEvidenceProvider,
    SpentDiscoveryEvidence,
)

_RAW_SPENT_REQUIRED_FAMILIES = tuple(
    family
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    if family != "tfidf_semantic_retrieval_contrasts"
)


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_sha(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wrapper(path: Path, *, schema: str, body: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content_sha = _json_sha(body)
    path.write_text(
        _canonical_json(
            {
                "schema_version": schema,
                "body": body,
                "content_sha256": content_sha,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return content_sha


def _identity_record(identity: dict) -> dict:
    return {"identity": identity, "identity_sha256": _json_sha(identity)}


def _float_hex_sha(values) -> str:
    return _json_sha([float(value).hex() for value in np.asarray(values, dtype=float)])


def _legacy_payload(outer_fold: int, review_round: int) -> dict:
    return {
        "outer_fold": outer_fold,
        "scope": "inner_train",
        "inner_fold": review_round + 1,
        "context": {
            "evidence_digest": {
                "confounders": {
                    "bow_blurbs": [
                        {
                            "source": "linear_1_2.confounder_overlap",
                            "view_name": "linear_1_2",
                            "bow_model": "linear",
                            "evidence_type": "confounder_overlap",
                            "meaning": "Terms shared by treatment and outcome models.",
                            "rows": [{"feature": "baseline wheel alignment", "score": 2.0}],
                        }
                    ],
                    "embedding_chunks": [
                        {
                            "name": "treatment",
                            "contrast_family": "marginal",
                            "direction_source": "mean_difference",
                            "concept_probe_scores": [
                                {"concept": "baseline actuator status", "score": 0.7}
                            ],
                        }
                    ],
                    "htr_blurbs": [
                        {
                            "stage": "nuisance",
                            "meaning": "Attention for treatment and outcome nuisance models.",
                            "rows": [{"top_token_spans": [{"token": "baseline vibration burden"}]}],
                        }
                    ],
                },
                "effect_modifiers": {
                    "bow_blurbs": [
                        {
                            "source": "linear_1_2.pseudo_target_positive",
                            "view_name": "linear_1_2",
                            "bow_model": "linear",
                            "evidence_type": "pseudo_target_positive",
                            "meaning": "Terms associated with an R-stage pseudo-target.",
                            "rows": [{"feature": "baseline coating status", "score": 3.0}],
                        },
                        {
                            "source": (
                                "matched_pair_uplift.pair_uplift__linear_1_2."
                                "uplift_pair_features"
                            ),
                            "view_name": "pair_uplift__linear_1_2",
                            "bow_model": "linear",
                            "evidence_type": "uplift_pair_features",
                            "meaning": "Matched-pair treated versus control outcome terms.",
                            "rows": [{"feature": "prior load burden", "score": 1.5}],
                        },
                    ],
                    "embedding_chunks": [
                        {
                            "name": "cluster_residualized_interaction_pc1",
                            "contrast_family": (
                                "cluster_local_residualized_interaction_contrast_basis"
                            ),
                            "direction_source": "mean_difference",
                            "concept_probe_scores": [
                                {"concept": "baseline calibration pattern", "score": -0.6}
                            ],
                        }
                    ],
                    "htr_blurbs": [
                        {
                            "stage": "effect",
                            "meaning": "Attention for residual-effect heterogeneity.",
                            "rows": [{"top_token_spans": [{"token": "baseline material result"}]}],
                        }
                    ],
                },
            }
        },
    }


def _tfidf_payload(outer_fold: int, review_round: int) -> dict:
    def topic(bank: str, phrase: str) -> dict:
        return {
            "topic_id": f"{bank}_topic_001",
            "bank": bank,
            "terms": [
                {
                    "term": phrase,
                    "loading": 0.8,
                    "screen_rank": 1,
                    "signed_score": 0.6,
                }
            ],
        }

    return {
        "outer_fold": outer_fold,
        "scope": "inner_train",
        "inner_fold": review_round + 1,
        "discovery": {
            "topic_banks": {
                "treatment": {"topics": [topic("treatment", "baseline routing pattern")]},
                "outcome": {"topics": [topic("outcome", "baseline failure pattern")]},
                "effect": {"topics": [topic("effect", "baseline sensor phrase")]},
            },
            "effect_orphan_ngram_branch": {
                "status": "completed",
                "selected_cluster_ids": ["cluster_001"],
                "selected_clusters": [
                    {
                        "cluster_id": "cluster_001",
                        "terms": [
                            {
                                "term": "unmodeled baseline phrase",
                                "combined_importance": 2.2,
                                "fit_rank": 1,
                                "fit_signed_score": 2.1,
                                "lexical_similarity_to_seed": 1.0,
                                "signed_score": 2.1,
                                "support_control": 20,
                                "support_treated": 21,
                            }
                        ],
                    }
                ],
            },
        },
    }


def _query_payload(outer_fold: int, review_round: int) -> dict:
    rows = []
    for bank, term in (
        ("treatment", "baseline age"),
        ("outcome", "performance status"),
        ("effect", "egfr mutation"),
    ):
        rows.append(
            {
                "query_id": f"{bank}_query_001",
                "bank": bank,
                "mechanical_role": "effect_modifier" if bank == "effect" else "confounder",
                "statistical_gate_applied": False,
                "member_count": 4,
                "fit_standardized_score": 3.2,
                "top_chunks": [],
                "top_contrastive_ngrams": [
                    {"term": term, "tfidf_contrast": 0.4},
                    {"term": f"{term} secondary", "tfidf_contrast": -0.2},
                ],
            }
        )
    return {
        "outer_fold": outer_fold,
        "scope": "inner_train",
        "inner_fold": review_round + 1,
        "query_evidence": rows,
    }


class _SpentBackend:
    def __init__(self, source_kind: str) -> None:
        self.source_kind = source_kind
        self.calls = 0

    def identity(self):
        if self.source_kind == LEGACY_ALL_SOURCE:
            return {
                "backend": "historical_stage1_spent_discovery_v5",
                "code_sha256": ("de11740a862c13d59d340e1dba26fb1202820dec4a0055c49819b7e01eccc1f1"),
                "concept_projection": (
                    "short_bow_terms_htr_tokens_or_per_row_chunk_attention_contrast_"
                    "embedding_tail_ngrams_v2"
                ),
                "raw_attention_or_embedding_excerpts_retained": False,
                "embedding_language_model_launch_allowed": False,
                "future_row_text_decoded_or_materialized": False,
            }
        return {"backend": f"prefix_test_{self.source_kind}_v1"}

    def fit_discovery(
        self,
        *,
        outer_fold,
        review_round,
        exact_spent_row_ids,
        spent_texts,
        spent_treatment,
        spent_outcome,
        work_dir,
    ):
        del spent_texts, spent_treatment, spent_outcome, work_dir
        self.calls += 1
        if self.source_kind == LEGACY_ALL_SOURCE:
            payload = _legacy_payload(outer_fold, review_round)
        elif self.source_kind == TFIDF_TOPIC_SOURCE:
            payload = _tfidf_payload(outer_fold, review_round)
        else:
            payload = _query_payload(outer_fold, review_round)
        return SpentDiscoveryEvidence.create(
            source_kind=self.source_kind,
            payload=payload,
            fit_row_provenance=FitRowProvenance(fit_row_ids=frozenset(exact_spent_row_ids)),
        )


def _spent_backends() -> tuple[_SpentBackend, ...]:
    return (
        _SpentBackend(LEGACY_ALL_SOURCE),
        _SpentBackend(TFIDF_TOPIC_SOURCE),
        _SpentBackend(NEURAL_QUERY_SOURCE),
    )


class _GateBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def identity(self):
        return {"backend": "prefix_test_safe_gate_v1"}

    def fit_predict(self, **kwargs):
        self.calls.append(kwargs)
        gate_ids = tuple(kwargs["gate_row_ids"])
        rows = len(gate_ids)
        return ContextFitUpstreamPrediction(
            gate_row_ids=gate_ids,
            calibrated_source_names=("bow_r",),
            calibrated_source_kinds=("nested_calibrated_bow_r",),
            calibrated_source_values=np.linspace(-0.2, 0.2, rows).reshape(-1, 1),
            feature_names=("bow_propensity", "htr_outcome", "pair_uplift"),
            feature_kinds=("bow_nuisance", "htr_neural", "matched_pair_uplift"),
            feature_roles=(
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                OUTCOME_NUISANCE_FEATURE_ROLE,
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            ),
            feature_values=np.column_stack(
                (
                    np.linspace(0.2, 0.8, rows),
                    np.linspace(0.7, 0.4, rows),
                    np.linspace(-1.0, 1.0, rows),
                )
            ),
        )


def _completed_prefix_fixture(tmp_path: Path) -> SimpleNamespace:
    preparation = tmp_path / "source_preparation"
    scratch = tmp_path / "source_scratch"
    preparation.mkdir()
    scratch.mkdir()
    spent_ids = (0, 1, 2, 3, 4, 5)
    sealed_ids = (6, 7, 8, 9)
    gate_ids = (6, 7)
    spent_texts = tuple(f"spent row {row_id}" for row_id in spent_ids)
    gate_texts = tuple(f"gate row {row_id}" for row_id in gate_ids)
    treatment = np.asarray((0, 1, 0, 1, 0, 1), dtype=float)
    outcome = np.asarray((0, 0, 1, 1, 0, 1), dtype=float)
    inner_fold_ids = (1, 1, 2, 2, 3, 3)

    spent_provider = ContextFitReviewSpentEvidenceProvider(
        backends=_spent_backends(),
        cache_dir=scratch / "post_extraction_review_spent_evidence_cache",
        required_source_families=_RAW_SPENT_REQUIRED_FAMILIES,
    )
    evidence_inputs = tuple(
        spent_provider.get_spent_evidence_inputs(
            outer_fold=1,
            review_round=0,
            exact_spent_row_ids=spent_ids,
            exact_sealed_row_ids=sealed_ids,
            spent_texts=spent_texts,
            spent_treatment=treatment,
            spent_outcome=outcome,
        )
    )
    compatibility = restore_current_spent_projection_semantic_retrieval_view(
        evidence_inputs,
        spent_evidence_provider=spent_provider,
        outer_fold=1,
        review_round=0,
        exact_spent_row_ids=spent_ids,
        exact_sealed_row_ids=sealed_ids,
        spent_texts=spent_texts,
        spent_treatment=treatment,
        spent_outcome=outcome,
    )
    compatibility_audit = {
        **compatibility.audit,
        "ledger_content_sha256": compatibility.ledger_content_sha256,
        "restored_object_count": compatibility.restored_object_count,
    }
    catalog = build_role_neutral_evidence_catalog(compatibility.evidence_inputs)
    assert all(catalog.family_atoms(family) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES)

    gate_backend = _GateBackend()
    gate_provider = ContextFitUpstreamGateProvider(
        scratch / "post_extraction_review_gate_cache",
        backend=gate_backend,
    )
    context = ObservableCausalRows(
        row_ids=spent_ids,
        extracted=pd.DataFrame({"_oci_row_id": spent_ids}),
        treatment=treatment,
        outcome=outcome,
        inner_fold_ids=inner_fold_ids,
    )
    bound = gate_provider.bind_fold(
        outer_fold=1,
        context=context,
        context_texts=spent_texts,
        gate_texts=gate_texts,
        exact_gate_row_ids=gate_ids,
    )
    gate_manifest = bound.authenticated_cache_manifest_path
    final_producer = FinalContextFitUpstreamProducer(
        scratch / "final_context_fit_upstream_cache",
        backend=_GateBackend(),
    )
    spent_record = _identity_record(dict(spent_provider.identity()))
    gate_record = _identity_record(dict(gate_provider.identity()))
    final_record = _identity_record(dict(final_producer.identity()))

    runner_schema = "prefix_salvage_test_runner_v1"
    companion_body = {
        "runner_schema_version": runner_schema,
        "post_extraction_review_providers": {
            "calibrated_gate_sources": gate_record,
            "role_aware_gate_feature_banks": gate_record,
        },
        "final_upstream_model_inputs": {"producer": final_record},
    }
    companion = preparation / "context_fit_overlay_companions" / "companion.json"
    _write_wrapper(companion, schema=runner_schema, body=companion_body)
    companion_sha = _file_sha(companion)

    input_body = {
        "runner_schema_version": runner_schema,
        "preparation_schema_version": HIERARCHICAL_PREPARATION_INPUT_SCHEMA_VERSION,
        "dataset": {"sha256": _json_sha({"dataset": "prefix-test"})},
        "outer_folds": [{"outer_fold": 1}, {"outer_fold": 2}],
        "semantic_retrieval_compatibility": (
            current_spent_projection_compatibility_identity()
        ),
        "spent_evidence_provider": spent_record,
        "shared_first_gate_provider": gate_record,
        "final_upstream_producer": final_record,
        "raw_final_upstream_producer": final_record,
        "context_fit_overlay_companion": {
            "path": str(companion.resolve()),
            "sha256": companion_sha,
            "overlay_compatible_closed_run_attestation": True,
        },
    }
    input_manifest = preparation / "immutable_hierarchical_input_manifest.json"
    _write_wrapper(
        input_manifest,
        schema=HIERARCHICAL_PREPARATION_INPUT_SCHEMA_VERSION,
        body=input_body,
    )

    fold_dir = preparation / "outer_fold_001"
    catalog_path = fold_dir / "role_neutral_evidence_catalog.json"
    catalog_envelope_sha = _write_wrapper(
        catalog_path,
        schema="role_neutral_evidence_catalog_preparation_envelope_v1",
        body=catalog.as_dict(),
    )
    chunk_plan_sha = _json_sha({"plan": "complete-all-ten"})
    chunk_path = fold_dir / "architecture_chunk_plan.json"
    chunk_envelope_sha = _write_wrapper(
        chunk_path,
        schema="architecture_chunk_plan_preparation_envelope_v1",
        body={"plan_sha256": chunk_plan_sha},
    )
    packet = {"outer_fold": 1, "all_active_architectures": True}
    approval_sha = _json_sha(packet)
    wrapper_path = fold_dir / "approved_hierarchical_wrapper_precommit.json"
    wrapper_envelope_sha = _write_wrapper(
        wrapper_path,
        schema="hierarchical_all_evidence_runner_batch_packet_v1",
        body={"approval_sha256": approval_sha, "packet": packet},
    )
    direct_content = {
        "schema_version": "direct_upstream_numerical_manifest_v3",
        "channel": "direct_upstream_numerical",
        "source_cache_key": gate_manifest.parent.name,
        "source_manifest_sha256": _file_sha(gate_manifest),
        "semantic_catalog_sha256": catalog.catalog_sha256,
        "family_coverage": [{"source_family": family} for family in ACTIVE_STAGE1_CONCEPT_FAMILIES],
        "all_active_stage1_architectures_covered": True,
        "concept_grounding_allowed": False,
    }
    direct = {**direct_content, "content_sha256": _json_sha(direct_content)}
    direct_path = fold_dir / "direct_upstream_numerical_manifest.json"
    direct_path.write_text(_canonical_json(direct) + "\n", encoding="utf-8")

    schedule = {
        "outer_fold": 1,
        "initial_spent_fold_ids": [1, 2, 3],
        "gate_fold_ids": [4, 5],
        "partitions": [
            {"fold_id": 1, "row_ids": [0, 1]},
            {"fold_id": 2, "row_ids": [2, 3]},
            {"fold_id": 3, "row_ids": [4, 5]},
            {"fold_id": 4, "row_ids": [6, 7]},
            {"fold_id": 5, "row_ids": [8, 9]},
        ],
    }
    initial_audit = {
        "review_round": 0,
        "consumer_review_round": 0,
        "spent_evidence_context_epoch": 0,
        "provider_review_round_argument": 0,
        "spent_row_count": len(spent_ids),
        "sealed_row_count": len(sealed_ids),
        "spent_row_fingerprint": row_set_fingerprint(spent_ids),
        "sealed_row_fingerprint": row_set_fingerprint(sealed_ids),
        "provider_identity_sha256": spent_record["identity_sha256"],
        "semantic_retrieval_compatibility": compatibility_audit,
    }
    bound_identity = dict(bound.identity())
    first_gate_audit = {
        "outer_fold": 1,
        "initial_spent_binding": {
            "row_count": len(spent_ids),
            "row_ids_sha256": _json_sha(list(spent_ids)),
            "text_sha256": _json_sha(list(spent_texts)),
            "treatment_sha256": _float_hex_sha(treatment),
            "outcome_sha256": _float_hex_sha(outcome),
            "inner_fold_assignment_sha256": _json_sha(
                {"row_ids": list(spent_ids), "inner_fold_ids": list(inner_fold_ids)}
            ),
        },
        "first_untouched_gate_binding": {
            "row_count": len(gate_ids),
            "row_ids_sha256": _json_sha(list(gate_ids)),
            "text_sha256": _json_sha(list(gate_texts)),
            "treatment_accepted": False,
            "outcome_accepted": False,
        },
        "upstream_cache_binding": {
            "bound_provider_identity_sha256": _json_sha(bound_identity),
            "source_cache_key": gate_manifest.parent.name,
            "source_manifest_sha256": _file_sha(gate_manifest),
        },
    }
    fold_body = {
        "outer_fold": 1,
        "schedule_audit": schedule,
        "initial_spent_evidence_audit": initial_audit,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_envelope_content_sha256": catalog_envelope_sha,
        "catalog_sha256": catalog.catalog_sha256,
        "chunk_plan_path": str(chunk_path.resolve()),
        "chunk_plan_envelope_content_sha256": chunk_envelope_sha,
        "chunk_plan_sha256": chunk_plan_sha,
        "direct_manifest_path": str(direct_path.resolve()),
        "direct_manifest_file_sha256": _file_sha(direct_path),
        "direct_manifest_content_sha256": direct["content_sha256"],
        "authenticated_first_gate_cache_manifest_sha256": _file_sha(gate_manifest),
        "first_gate_preparation_audit": first_gate_audit,
        "first_gate_preparation_audit_sha256": _json_sha(first_gate_audit),
        "first_gate_provider_identity": bound_identity,
        "first_gate_cache_materialized_before_discovery": True,
        "first_gate_labels_supplied_to_provider": False,
        "first_gate_views_exposed_to_discovery": False,
        "wrapper_precommit_path": str(wrapper_path.resolve()),
        "wrapper_precommit_envelope_content_sha256": wrapper_envelope_sha,
        "wrapper_approval_sha256": approval_sha,
        "hierarchy_runner_calls_during_preparation": 0,
    }
    fold_manifest = fold_dir / "immutable_fold_preparation.json"
    _write_wrapper(
        fold_manifest,
        schema=HIERARCHICAL_PREPARATION_FOLD_SCHEMA_VERSION,
        body=fold_body,
    )

    temporary_spent = (
        scratch
        / "post_extraction_review_spent_evidence_cache"
        / "review_spent_001_00_incomplete"
        / "backend_01"
    )
    temporary_spent.mkdir(parents=True)
    (temporary_spent / "fitted_context.joblib").write_bytes(b"must-not-be-indexed")
    backend_work = scratch / "post_extraction_review_gate_cache" / "backend_work"
    backend_work.mkdir(parents=True)
    (backend_work / "partial.joblib").write_bytes(b"must-not-be-indexed")

    return SimpleNamespace(
        preparation=preparation,
        scratch=scratch,
        spent_ids=spent_ids,
        sealed_ids=sealed_ids,
        gate_ids=gate_ids,
        spent_texts=spent_texts,
        gate_texts=gate_texts,
        treatment=treatment,
        outcome=outcome,
        inner_fold_ids=inner_fold_ids,
        context=context,
        gate_manifest=gate_manifest,
        fold_manifest=fold_manifest,
        spent_provider=spent_provider,
        spent_request={
            "outer_fold": 1,
            "review_round": 0,
            "exact_spent_row_ids": spent_ids,
            "exact_sealed_row_ids": sealed_ids,
            "spent_texts": spent_texts,
            "spent_treatment": treatment,
            "spent_outcome": outcome,
        },
    )


def test_completed_prefix_exports_exact_overlay_sources_and_ignores_executable_state(
    tmp_path: Path,
) -> None:
    fixture = _completed_prefix_fixture(tmp_path)
    exported = export_completed_hierarchical_preparation_prefix(
        preparation_dir=fixture.preparation,
        scratch_output_dir=fixture.scratch,
        spent_evidence_provider=fixture.spent_provider,
        spent_requests_by_outer_fold={1: fixture.spent_request},
        destination=tmp_path / "salvaged_prefix",
    )

    assert exported.completed_outer_folds == (1,)
    assert len(exported.review_spent_registrations) == 1
    assert (
        len(authenticate_review_spent_cache_registrations(exported.review_spent_registrations)) == 1
    )
    gate_sources = authenticate_context_fit_cache_index_registrations(
        (exported.context_fit_index_registration,)
    )
    assert len(gate_sources) == 1
    assert gate_sources[0].kind == "review_gate"
    exported.validate_authentication()
    emitted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in exported.destination.iterdir()
        if path.suffix == ".json"
    )
    assert "fitted_context.joblib" not in emitted
    assert "backend_work/" not in emitted
    assert "_fit_call_checkpoints/" not in emitted

    fresh = tmp_path / "fresh_overlay"
    fresh.mkdir()
    fresh_backends = _spent_backends()
    fresh_spent_provider = ContextFitReviewSpentEvidenceProvider(
        backends=fresh_backends,
        cache_dir=fresh / "spent",
        required_source_families=_RAW_SPENT_REQUIRED_FAMILIES,
    )
    spent_overlay = AuthenticatedReviewSpentEvidenceCacheOverlay(
        provider=fresh_spent_provider,
        sources=authenticate_review_spent_cache_registrations(exported.review_spent_registrations),
        output_root=fresh,
    )
    spent_overlay.get_spent_evidence_inputs(
        outer_fold=1,
        review_round=0,
        exact_spent_row_ids=fixture.spent_ids,
        exact_sealed_row_ids=fixture.sealed_ids,
        spent_texts=fixture.spent_texts,
        spent_treatment=fixture.treatment,
        spent_outcome=fixture.outcome,
    )
    assert all(backend.calls == 0 for backend in fresh_backends)

    fresh_gate_backend = _GateBackend()
    fresh_gate_provider = ContextFitUpstreamGateProvider(fresh / "gate", backend=fresh_gate_backend)
    fresh_final_producer = FinalContextFitUpstreamProducer(fresh / "final", backend=_GateBackend())
    gate_overlay = AuthenticatedContextFitGateCacheOverlay(
        provider=fresh_gate_provider,
        runtime_producer=fresh_final_producer,
        sources=gate_sources,
        output_root=fresh,
        hierarchical_first_gate_preparation=True,
    )
    gate_overlay.bind_fold(
        outer_fold=1,
        context=fixture.context,
        context_texts=fixture.spent_texts,
        gate_texts=fixture.gate_texts,
        exact_gate_row_ids=fixture.gate_ids,
    )
    assert fresh_gate_backend.calls == []


def test_completed_prefix_rejects_checkpoint_only_gate_state(tmp_path: Path) -> None:
    fixture = _completed_prefix_fixture(tmp_path)
    fixture.gate_manifest.unlink()
    assert tuple(
        fixture.scratch.glob(
            "post_extraction_review_gate_cache/_fit_call_checkpoints/*/manifest.json"
        )
    )

    with pytest.raises(
        HierarchicalPreparationPrefixSalvageError,
        match="complete top-level gate manifest",
    ):
        export_completed_hierarchical_preparation_prefix(
            preparation_dir=fixture.preparation,
            scratch_output_dir=fixture.scratch,
            spent_evidence_provider=fixture.spent_provider,
            spent_requests_by_outer_fold={1: fixture.spent_request},
            destination=tmp_path / "rejected_checkpoint_only",
        )


def test_completed_prefix_rejects_tampered_immutable_fold_manifest(tmp_path: Path) -> None:
    fixture = _completed_prefix_fixture(tmp_path)
    raw = json.loads(fixture.fold_manifest.read_text(encoding="utf-8"))
    raw["body"]["outer_fold"] = 99
    fixture.fold_manifest.write_text(_canonical_json(raw) + "\n", encoding="utf-8")

    with pytest.raises(
        HierarchicalPreparationPrefixSalvageError,
        match="content hash mismatch",
    ):
        export_completed_hierarchical_preparation_prefix(
            preparation_dir=fixture.preparation,
            scratch_output_dir=fixture.scratch,
            spent_evidence_provider=fixture.spent_provider,
            spent_requests_by_outer_fold={1: fixture.spent_request},
            destination=tmp_path / "rejected_tamper",
        )


def test_completed_prefix_rejects_noncurrent_semantic_compatibility_identity(
    tmp_path: Path,
) -> None:
    fixture = _completed_prefix_fixture(tmp_path)
    input_manifest = fixture.preparation / "immutable_hierarchical_input_manifest.json"
    raw = json.loads(input_manifest.read_text(encoding="utf-8"))
    identity = raw["body"]["semantic_retrieval_compatibility"]
    identity["body"]["implementation_file_sha256"] = "0" * 64
    identity["content_sha256"] = _json_sha(identity["body"])
    raw["content_sha256"] = _json_sha(raw["body"])
    input_manifest.write_text(_canonical_json(raw) + "\n", encoding="utf-8")

    with pytest.raises(
        HierarchicalPreparationPrefixSalvageError,
        match="semantic compatibility identity is not current",
    ):
        export_completed_hierarchical_preparation_prefix(
            preparation_dir=fixture.preparation,
            scratch_output_dir=fixture.scratch,
            spent_evidence_provider=fixture.spent_provider,
            spent_requests_by_outer_fold={1: fixture.spent_request},
            destination=tmp_path / "rejected_compatibility_identity",
        )


def test_completed_prefix_rejects_resealed_false_semantic_compatibility_audit(
    tmp_path: Path,
) -> None:
    fixture = _completed_prefix_fixture(tmp_path)
    raw = json.loads(fixture.fold_manifest.read_text(encoding="utf-8"))
    compatibility = raw["body"]["initial_spent_evidence_audit"]["semantic_retrieval_compatibility"]
    compatibility["restored_object_count"] += 1
    raw["content_sha256"] = _json_sha(raw["body"])
    fixture.fold_manifest.write_text(_canonical_json(raw) + "\n", encoding="utf-8")

    with pytest.raises(
        HierarchicalPreparationPrefixSalvageError,
        match="semantic-retrieval compatibility audit changed",
    ):
        export_completed_hierarchical_preparation_prefix(
            preparation_dir=fixture.preparation,
            scratch_output_dir=fixture.scratch,
            spent_evidence_provider=fixture.spent_provider,
            spent_requests_by_outer_fold={1: fixture.spent_request},
            destination=tmp_path / "rejected_false_compatibility",
        )
