from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import oci.inference.all_evidence_fusion_runner as fusion_runner_module
from oci.inference.all_evidence_fusion import (
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPIC_SOURCE,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
)
from oci.inference.all_evidence_fusion_runner import (
    AllEvidenceFusionRunner,
    AllEvidenceFusionRunnerConfig,
    FoldTrainExplicitEncoder,
    QueryEvidenceArtifact,
    TfidfOrphanNgramArtifact,
    _build_injected_review_partition_schedule,
    _build_final_upstream_meta_inner_fold_ids,
    _build_review_partition_schedule,
    _prepare_final_upstream_head_inputs,
    _reconstruct_forest_potential_outcomes,
    _validate_closed_staged_fusion_audit,
    _sanitize_retained_legacy_digest,
    evaluate_frozen_all_evidence_predictions,
    load_candidate_pool,
    load_legacy_full_outer_evidence,
    load_outer_splits_from_primary_predictions,
    load_sanitized_dataset,
)
from oci.inference.all_evidence_post_extraction_review import (
    GateAcceptanceDecision,
    GateFeatureBankView,
    GateSourceSignalView,
    ObservableCausalRows,
    OUTCOME_NUISANCE_FEATURE_ROLE,
    POST_EXTRACTION_REVIEW_FRESH_NORMALIZATION_VERSION,
    POST_EXTRACTION_REVIEW_PROMPT_VERSION,
    POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from oci.inference.context_fit_upstream_gate_provider import ContextFitUpstreamPrediction
from oci.inference.final_context_fit_upstream_bank import FinalContextFitUpstreamProducer
from oci.inference.final_context_fit_causal_forest_adapter import (
    FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID,
    FixedCausalForestHeadBackend,
)
from oci.inference.final_context_fit_r_stack_adapter import (
    EXACT_OUTCOME_PREDICTION,
    EXACT_PROPENSITY_PREDICTION,
    SealedExactNuisanceBankExtension,
)
from oci.inference.fold_honest_r_stack import FitRowProvenance
from oci.inference.frozen_extraction_cache_overlay import (
    LEGACY_EXTRACTION_CACHE_INDEX_SCHEMA_VERSION,
    FrozenExtractionCacheOverlay,
    extraction_contract_sha256,
    ordered_dataset_text_fingerprint,
    sha256_file,
)
from oci.inference.staged_all_evidence_fusion_agent import StagedAllEvidenceFusionAgent
from oci.inference.tfidf_topic_discovery import HANDOFF_SCHEMA_VERSION, row_set_fingerprint


def test_runner_default_grid_explores_stronger_logistic_regularization():
    assert AllEvidenceFusionRunnerConfig().regularization_grid == (
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
    )


def test_runner_defaults_to_strict_two_round_review_and_manifest_identity():
    config = AllEvidenceFusionRunnerConfig()

    assert config.post_extraction_review_rounds == 2
    assert config.require_review_source_signals is True
    assert config.require_review_feature_banks is True
    assert config.require_final_upstream_inputs is True
    assert config.require_final_upstream_neural_query_inputs is True
    assert config.require_final_causal_forest is True
    assert config.allow_degraded_review_without_all_upstream is False

    strict_manifest_identity = fusion_runner_module._content_sha256(
        fusion_runner_module.asdict(config)
    )
    explicitly_nonadaptive = AllEvidenceFusionRunnerConfig(
        post_extraction_review_rounds=0
    )
    nonadaptive_manifest_identity = fusion_runner_module._content_sha256(
        fusion_runner_module.asdict(explicitly_nonadaptive)
    )
    assert strict_manifest_identity != nonadaptive_manifest_identity


def test_runner_default_fails_closed_without_exact_causal_forest_runtime(tmp_path):
    with pytest.raises(ValueError, match="required final causal forest.*exact raw"):
        AllEvidenceFusionRunner(
            dataset_path=tmp_path / "dataset.parquet",
            legacy_handoff_path=tmp_path / "legacy.jsonl",
            tfidf_handoff_path=tmp_path / "tfidf.jsonl",
            output_dir=tmp_path / "output",
            fusion_agent=_FusionAgent(),
            extraction_provider=_Extractor(),
        )


def test_required_neural_query_moments_cannot_enable_sparse_fallback():
    with pytest.raises(ValueError, match="cannot enable the sparse query fallback"):
        AllEvidenceFusionRunnerConfig(
            require_neural_query_moments=True,
            derive_sparse_query_moments_when_missing=True,
        )


@pytest.mark.parametrize("value", [-1, 9, True, 1.5])
def test_runner_config_rejects_invalid_review_quality_retry_bound(value):
    with pytest.raises(ValueError, match="post_extraction_review_max_quality_retries"):
        AllEvidenceFusionRunnerConfig(post_extraction_review_max_quality_retries=value)


@pytest.mark.parametrize(
    "required_flag",
    ["require_final_upstream_inputs", "require_final_upstream_neural_query_inputs"],
)
def test_runner_fails_closed_when_required_final_upstream_producer_is_missing(
    tmp_path,
    required_flag,
):
    with pytest.raises(ValueError, match="no post-registry producer"):
        AllEvidenceFusionRunner(
            dataset_path=tmp_path / "dataset.parquet",
            legacy_handoff_path=tmp_path / "legacy.jsonl",
            tfidf_handoff_path=tmp_path / "tfidf.jsonl",
            output_dir=tmp_path / "output",
            fusion_agent=_FusionAgent(),
            extraction_provider=_Extractor(),
            config=AllEvidenceFusionRunnerConfig(
                post_extraction_review_rounds=0,
                **{required_flag: True},
            ),
        )


def test_required_final_forest_needs_exact_raw_runtime_and_defaults_to_fixed_backend(
    tmp_path,
):
    with pytest.raises(ValueError, match="exact raw FinalContextFitUpstreamProducer"):
        AllEvidenceFusionRunner(
            dataset_path=tmp_path / "dataset.parquet",
            legacy_handoff_path=tmp_path / "legacy.jsonl",
            tfidf_handoff_path=tmp_path / "tfidf.jsonl",
            output_dir=tmp_path / "missing_raw",
            fusion_agent=_FusionAgent(),
            extraction_provider=_Extractor(),
            config=AllEvidenceFusionRunnerConfig(
                post_extraction_review_rounds=0,
                require_final_causal_forest=True,
            ),
        )

    raw = FinalContextFitUpstreamProducer(
        tmp_path / "final_cache",
        backend=_FinalRunnerSignalBackend(),
    )
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "fixed",
        fusion_agent=_FusionAgent(),
        extraction_provider=_Extractor(),
        final_upstream_producer=raw,
        raw_final_upstream_producer=raw,
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            require_final_causal_forest=True,
        ),
    )

    assert type(runner.final_causal_forest_backend) is FixedCausalForestHeadBackend
    assert runner.final_causal_forest_backend.identity() == (
        FixedCausalForestHeadBackend().identity()
    )


@pytest.mark.parametrize("value", [0, 1, True, 2.5])
def test_runner_config_rejects_invalid_final_upstream_meta_inner_folds(value):
    with pytest.raises(ValueError, match="final_upstream_meta_inner_folds"):
        AllEvidenceFusionRunnerConfig(final_upstream_meta_inner_folds=value)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan"), True])
def test_runner_config_rejects_invalid_final_upstream_head_regularization(value):
    with pytest.raises(ValueError, match="final_upstream_head_regularization"):
        AllEvidenceFusionRunnerConfig(final_upstream_head_regularization=value)


def test_final_upstream_meta_inner_partition_is_deterministic_and_stratified():
    rows = pd.DataFrame(
        {
            "_oci_row_id": np.arange(100, 124),
            "treatment": np.tile([0, 1], 12),
            "outcome": np.tile([0, 0, 1, 1], 6),
        }
    )
    kwargs = {
        "n_splits": 3,
        "random_state": 19,
        "outer_fold": 2,
        "treatment_column": "treatment",
        "outcome_column": "outcome",
        "outcome_type": "binary",
    }
    first, first_audit = _build_final_upstream_meta_inner_fold_ids(rows, **kwargs)
    second, second_audit = _build_final_upstream_meta_inner_fold_ids(rows, **kwargs)

    assert first == second
    assert first_audit == second_audit
    assert set(first) == {1, 2, 3}
    assert first_audit["strategy"] == "joint_treatment_outcome"
    assert first_audit["row_level_assignments_persisted_in_runner_audit"] is False
    assert all(set(record["treatment_counts"].values()) == {4} for record in first_audit["folds"])


def test_review_partition_schedule_is_deterministic_outer_train_only_and_balanced():
    rows = pd.DataFrame(
        {
            "_oci_row_id": np.arange(80, 160),
            "treatment": np.tile([0, 0, 1, 1], 20),
            "outcome": np.tile([0.0, 1.0, 0.0, 1.0], 20),
        }
    )
    kwargs = {
        "outer_fold": 2,
        "review_rounds": 2,
        "minimum_partition_rows": 8,
        "random_state": 17,
        "treatment_column": "treatment",
        "outcome_column": "outcome",
        "outcome_type": "binary",
    }

    first = _build_review_partition_schedule(rows, **kwargs)
    second = _build_review_partition_schedule(rows, **kwargs)

    assert first.audit == second.audit
    assert first.initial_spent_fold_ids == (1, 2, 3)
    assert first.gate_fold_ids == (4, 5)
    assigned = [row_id for partition in first.row_ids_by_fold.values() for row_id in partition]
    assert set(assigned) == set(rows["_oci_row_id"])
    assert len(assigned) == len(set(assigned))
    assert all(row_id >= 80 for row_id in assigned)
    assert all(row["treatment_counts"]["0"] > 0 for row in first.audit["partitions"])
    assert all(row["treatment_counts"]["1"] > 0 for row in first.audit["partitions"])
    assert first.audit["outer_heldout_rows_used"] is False


def test_injected_five_fold_schedule_uses_four_spent_folds_for_one_final_gate():
    rows = pd.DataFrame(
        {
            "_oci_row_id": np.arange(100),
            "treatment": np.tile([0, 0, 1, 1], 25),
            "outcome": np.tile([0.0, 1.0, 0.0, 1.0], 25),
        }
    )

    class Provider:
        def get_review_partition_assignments(self, **kwargs):
            assert kwargs["exact_outer_train_row_ids"] == tuple(range(100))
            return {fold: tuple(range((fold - 1) * 20, fold * 20)) for fold in range(1, 6)}

    schedule = _build_injected_review_partition_schedule(
        rows,
        outer_fold=1,
        review_rounds=1,
        minimum_partition_rows=8,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        provider=Provider(),
        provider_identity={"identity_sha256": "a" * 64},
    )

    assert schedule.initial_spent_fold_ids == (1, 2, 3, 4)
    assert schedule.gate_fold_ids == (5,)
    assert schedule.audit["initial_spent_partition_count"] == 4


def test_review_config_requires_each_gate_provider_independently(tmp_path):
    with pytest.raises(ValueError, match="spent-only evidence provider"):
        AllEvidenceFusionRunner(
            dataset_path=tmp_path / "data.parquet",
            legacy_handoff_path=tmp_path / "legacy.jsonl",
            tfidf_handoff_path=tmp_path / "tfidf.jsonl",
            output_dir=tmp_path / "out",
            fusion_agent=_FusionAgent(),
            extraction_provider=_Extractor(),
            review_agent=lambda context: {},
            config=AllEvidenceFusionRunnerConfig(
                post_extraction_review_rounds=1,
                allow_degraded_review_without_all_upstream=True,
            ),
        )

    config = AllEvidenceFusionRunnerConfig(
        post_extraction_review_rounds=1,
        require_review_source_signals=True,
        allow_degraded_review_without_all_upstream=True,
    )
    with pytest.raises(ValueError, match="no gate-local provider"):
        AllEvidenceFusionRunner(
            dataset_path=tmp_path / "data.parquet",
            legacy_handoff_path=tmp_path / "legacy.jsonl",
            tfidf_handoff_path=tmp_path / "tfidf.jsonl",
            output_dir=tmp_path / "out",
            fusion_agent=_FusionAgent(),
            extraction_provider=_Extractor(),
            review_agent=lambda context: {},
            review_spent_evidence_provider=_SpentEvidenceProvider(),
            config=config,
        )

    class PrecomputedFeatureBank:
        def identity(self):
            return {"provider": "test_precomputed_feature_bank_v1"}

        def get_review_partition_assignments(self, **kwargs):  # pragma: no cover
            raise AssertionError("startup validation should run first")

        def get_gate_feature_bank_view(self, **kwargs):  # pragma: no cover
            raise AssertionError("startup validation should run first")

    precomputed = PrecomputedFeatureBank()
    with pytest.raises(ValueError, match="sequential context-fit feature banks"):
        AllEvidenceFusionRunner(
            dataset_path=tmp_path / "data.parquet",
            legacy_handoff_path=tmp_path / "legacy.jsonl",
            tfidf_handoff_path=tmp_path / "tfidf.jsonl",
            output_dir=tmp_path / "out",
            fusion_agent=_FusionAgent(),
            extraction_provider=_Extractor(),
            review_agent=lambda context: {},
            review_spent_evidence_provider=_SpentEvidenceProvider(),
            review_partition_provider=precomputed,
            review_gate_feature_bank_provider=precomputed,
            config=AllEvidenceFusionRunnerConfig(
                post_extraction_review_rounds=2,
                require_review_feature_banks=True,
                allow_degraded_review_without_all_upstream=True,
            ),
        )

    with pytest.raises(ValueError, match="authenticated exact review partitions"):
        AllEvidenceFusionRunner(
            dataset_path=tmp_path / "data.parquet",
            legacy_handoff_path=tmp_path / "legacy.jsonl",
            tfidf_handoff_path=tmp_path / "tfidf.jsonl",
            output_dir=tmp_path / "out",
            fusion_agent=_FusionAgent(),
            extraction_provider=_Extractor(),
            review_agent=lambda context: {},
            review_spent_evidence_provider=_SpentEvidenceProvider(),
            review_gate_feature_bank_provider=object(),
            config=AllEvidenceFusionRunnerConfig(
                post_extraction_review_rounds=1,
                require_review_feature_banks=True,
                allow_degraded_review_without_all_upstream=True,
            ),
        )


def test_adaptive_required_neural_queries_reject_spent_provider_without_selector_leg(
    tmp_path,
):
    class BindableGateProvider:
        def identity(self):
            return {"provider": "test_bindable_gate_provider_v1"}

        def bind_fold(self, **_kwargs):  # pragma: no cover - startup validation only
            raise AssertionError("startup validation should run first")

    class FinalProducer:
        def identity(self):
            return {"producer": "test_final_upstream_producer_v1"}

        def produce(self, **_kwargs):  # pragma: no cover - startup validation only
            raise AssertionError("startup validation should run first")

    gate_provider = BindableGateProvider()
    with pytest.raises(ValueError, match="spent discovery provider identity.*neural_query"):
        AllEvidenceFusionRunner(
            dataset_path=tmp_path / "data.parquet",
            legacy_handoff_path=tmp_path / "legacy.jsonl",
            tfidf_handoff_path=tmp_path / "tfidf.jsonl",
            output_dir=tmp_path / "out",
            fusion_agent=_FusionAgent(),
            extraction_provider=_Extractor(),
            review_agent=_FusionAgent(),
            review_spent_evidence_provider=_SpentEvidenceProvider(),
            review_gate_source_provider=gate_provider,
            review_gate_feature_bank_provider=gate_provider,
            final_upstream_producer=FinalProducer(),
            config=AllEvidenceFusionRunnerConfig(
                post_extraction_review_rounds=1,
                require_review_source_signals=True,
                require_review_feature_banks=True,
                require_final_upstream_inputs=True,
                require_final_upstream_neural_query_inputs=True,
                require_neural_query_moments=True,
                allow_degraded_review_without_all_upstream=True,
            ),
        )


def test_runner_gate_bind_boundary_exposes_only_exact_ids_text_and_spent_labels(tmp_path):
    context = ObservableCausalRows(
        row_ids=(1, 2, 3, 4),
        extracted=pd.DataFrame({"baseline": [0.0, 1.0, 0.0, 1.0]}),
        treatment=np.asarray([0.0, 1.0, 0.0, 1.0]),
        outcome=np.asarray([0.2, 0.8, 0.3, 0.7]),
        inner_fold_ids=(1, 1, 2, 2),
    )
    exact_gate_ids = (9, 8)
    gate_texts = ("exact note nine", "exact note eight")
    context_texts = tuple(f"spent note {row_id}" for row_id in context.row_ids)
    calls: list[dict[str, object]] = []

    class Bound:
        def __init__(self, *, reverse: bool = False):
            self.reverse = reverse

        def _rows(self):
            return tuple(reversed(exact_gate_ids)) if self.reverse else exact_gate_ids

        @staticmethod
        def _context_lineage():
            return tuple(
                FitRowProvenance(
                    fit_row_ids=frozenset(
                        candidate
                        for candidate, candidate_fold in zip(
                            context.row_ids, context.inner_fold_ids
                        )
                        if candidate_fold != fold_id
                    )
                )
                for fold_id in context.inner_fold_ids
            )

        def get_gate_source_view(self, *, outer_fold, exact_gate_row_ids):
            assert outer_fold == 1
            assert exact_gate_row_ids == exact_gate_ids
            lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
            return GateSourceSignalView(
                row_ids=self._rows(),
                source_names=("opaque_source",),
                source_kinds=("nested_calibrated_effect",),
                values=np.arange(2, dtype=float).reshape(-1, 1),
                fit_row_provenance=(lineage,),
                context_row_ids=context.row_ids,
                context_inner_fold_ids=context.inner_fold_ids,
                context_values=np.arange(len(context.row_ids), dtype=float).reshape(-1, 1),
                context_fit_row_provenance=(self._context_lineage(),),
            )

        def get_gate_feature_bank_view(self, *, outer_fold, exact_gate_row_ids):
            assert outer_fold == 1
            assert exact_gate_row_ids == exact_gate_ids
            lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
            return GateFeatureBankView(
                row_ids=self._rows(),
                feature_names=("opaque_feature",),
                source_kinds=("whole_embedding_contrast",),
                consumer_roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,),
                values=np.arange(2, dtype=float).reshape(-1, 1),
                fit_row_provenance=(lineage,),
                context_row_ids=context.row_ids,
                context_inner_fold_ids=context.inner_fold_ids,
                context_values=np.arange(len(context.row_ids), dtype=float).reshape(-1, 1),
                context_fit_row_provenance=(self._context_lineage(),),
            )

    class Bindable:
        reverse = False

        def identity(self):
            return {"provider": "adversarial_gate_boundary_probe_v1"}

        # Any legacy ``gate`` argument makes this explicit signature fail.
        def bind_fold(
            self,
            *,
            outer_fold,
            context,
            context_texts,
            gate_texts,
            exact_gate_row_ids,
        ):
            calls.append(
                {
                    "outer_fold": outer_fold,
                    "context": context,
                    "context_texts": context_texts,
                    "gate_texts": gate_texts,
                    "exact_gate_row_ids": exact_gate_row_ids,
                }
            )
            return Bound(reverse=self.reverse)

    provider = Bindable()
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=_Extractor(),
        review_gate_source_provider=provider,
        review_gate_feature_bank_provider=provider,
        config=AllEvidenceFusionRunnerConfig(post_extraction_review_rounds=0),
    )
    source = runner._gate_source_view(
        outer_fold=1,
        gate_row_ids=exact_gate_ids,
        context=context,
        context_texts=context_texts,
        gate_texts=gate_texts,
    )
    features = runner._gate_feature_bank_view(
        outer_fold=1,
        gate_row_ids=exact_gate_ids,
        context=context,
        context_texts=context_texts,
        gate_texts=gate_texts,
    )

    assert source is not None and source.row_ids == exact_gate_ids
    assert features is not None and features.row_ids == exact_gate_ids
    assert len(calls) == 2
    assert all(
        set(call)
        == {
            "outer_fold",
            "context",
            "context_texts",
            "gate_texts",
            "exact_gate_row_ids",
        }
        for call in calls
    )
    assert all(call["context"] is context for call in calls)
    assert all(call["exact_gate_row_ids"] == exact_gate_ids for call in calls)
    assert all(call["gate_texts"] == gate_texts for call in calls)

    provider.reverse = True
    with pytest.raises(ValueError, match="changed the exact gate row order/set"):
        runner._gate_source_view(
            outer_fold=1,
            gate_row_ids=exact_gate_ids,
            context=context,
            context_texts=context_texts,
            gate_texts=gate_texts,
        )
    with pytest.raises(ValueError, match="changed the exact gate row order/set"):
        runner._gate_feature_bank_view(
            outer_fold=1,
            gate_row_ids=exact_gate_ids,
            context=context,
            context_texts=context_texts,
            gate_texts=gate_texts,
        )


def test_explicit_encoder_uses_fixed_imputation_and_defers_scaling_to_head():
    spec = {
        "name": "sensor_value",
        "type": "continuous",
        "roles": ["effect_modifier"],
    }
    value = "explicit_feat_sensor_value"
    missing = f"{value}_missing"
    train = pd.DataFrame({value: [10.0, np.nan], missing: [False, True]})
    heldout = pd.DataFrame({value: [25.0, np.nan], missing: [False, True]})

    encoder = FoldTrainExplicitEncoder().fit(train, [spec])
    encoded = encoder.transform(heldout)

    assert encoded.tolist() == [[25.0, 0.0], [0.0, 1.0]]
    assert encoder.state_dict()["train_summaries_used_by_model"] is False


def _topic_banks() -> dict:
    return {
        "treatment": {
            "topics": [
                {
                    "topic_id": "treatment_1",
                    "terms": [{"term": "baseline", "loading": 0.8}],
                }
            ]
        },
        "outcome": {
            "topics": [
                {
                    "topic_id": "outcome_1",
                    "terms": [{"term": "inlet load", "loading": 0.7}],
                }
            ]
        },
        "effect": {
            "topics": [
                {
                    "topic_id": "effect_1",
                    "terms": [{"term": "prerun alloy phase", "loading": 0.9}],
                }
            ]
        },
    }


def _tfidf_row(
    *,
    outer_fold,
    scope,
    fit_ids,
    heldout_ids,
    inner_fold=None,
    effect_ngram_registration=None,
):
    fit_ids = list(fit_ids)
    heldout_ids = list(heldout_ids)
    identities = {
        "dataset_content_fingerprint": "1" * 64,
        "dataset_ordered_row_fingerprint": "2" * 64,
        "split_semantics_hash": "3" * 64,
    }
    discovery = {
        "fit_row_ids": fit_ids,
        "heldout_row_ids": heldout_ids,
        "fit_row_fingerprint": row_set_fingerprint(fit_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
        "topic_banks": _topic_banks(),
        "topic_score_tests": (
            {"status": "not_run", "uses_heldout_treatment_and_outcome": False}
            if scope == "full_outer_train"
            else {"status": "completed", "uses_heldout_treatment_and_outcome": True}
        ),
        "artifacts": {
            "topic_score_tests": None,
            **(
                {"ngram_scores": {"effect": effect_ngram_registration}}
                if effect_ngram_registration is not None
                else {}
            ),
        },
        **identities,
    }
    return {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "fold_key": outer_fold if inner_fold is None else outer_fold * 1000 + inner_fold,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "scope": scope,
        "fit_row_ids": fit_ids,
        "heldout_row_ids": heldout_ids,
        "fit_row_fingerprint": row_set_fingerprint(fit_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
        "split_registry_content_hash": "4" * 64,
        "discovery": discovery,
        **identities,
    }


def _write_handoffs(
    tmp_path: Path,
    *,
    oracle_in_legacy=False,
    full_effect_ngram_registrations=None,
):
    legacy_rows = []
    tfidf_rows = []
    all_ids = set(range(12))
    heldouts = {1: set(range(6)), 2: set(range(6, 12))}
    for fold in (1, 2):
        fit = sorted(all_ids - heldouts[fold])
        heldout = sorted(heldouts[fold])
        feature_row = {"feature": "baseline status", "score": 2.0}
        if oracle_in_legacy and fold == 1:
            feature_row["feature"] = "oracle-selected baseline status"
        importance = {
            "views": [
                {
                    "view_name": "linear_unigram",
                    "confounder_overlap": [feature_row],
                    "treatment_positive": [feature_row],
                    "outcome_positive": [feature_row],
                    "pseudo_target_positive": [
                        {"feature": "prerun inlet valve status", "score": 1.5}
                    ],
                }
            ]
        }
        legacy_rows.append(
            {
                "schema_version": "multi_model_agentic_discovery_handoff_v1",
                "fold_key": fold,
                "outer_fold": fold,
                "scope": "full_outer_train",
                "n_rows": len(fit),
                "metrics": {"oracle_true_ite_corr": 0.99},
                "importance": importance,
                "embedding_contrast_evidence": {},
                "htr_evidence": {},
                "context": {"synthetic_examples": ["must not be read"]},
            }
        )
        # Exact-inner records are accepted but intentionally never consumed.
        for inner_fold in (1, 2):
            legacy_rows.append(
                {
                    "schema_version": "multi_model_agentic_discovery_handoff_v1",
                    "fold_key": fold * 1000 + inner_fold,
                    "outer_fold": fold,
                    "inner_fold": inner_fold,
                    "scope": "candidate_selection_inner_fit",
                    "n_rows": 3,
                    "heldout_rows": 3,
                    "importance": importance,
                    "embedding_contrast_evidence": {},
                    "htr_evidence": {},
                    "context": {"synthetic_examples": ["must not be read"]},
                }
            )
        tfidf_rows.append(
            _tfidf_row(
                outer_fold=fold,
                scope="full_outer_train",
                fit_ids=fit,
                heldout_ids=heldout,
                effect_ngram_registration=((full_effect_ngram_registrations or {}).get(fold)),
            )
        )
        first = fit[:3]
        second = fit[3:]
        tfidf_rows.extend(
            [
                _tfidf_row(
                    outer_fold=fold,
                    inner_fold=1,
                    scope="candidate_selection_inner_fit",
                    fit_ids=second,
                    heldout_ids=first,
                ),
                _tfidf_row(
                    outer_fold=fold,
                    inner_fold=2,
                    scope="candidate_selection_inner_fit",
                    fit_ids=first,
                    heldout_ids=second,
                ),
            ]
        )
    legacy_path = tmp_path / "legacy.jsonl"
    legacy_path.write_text("".join(json.dumps(row) + "\n" for row in legacy_rows))
    tfidf_path = tmp_path / "tfidf.jsonl"
    tfidf_path.write_text("".join(json.dumps(row) + "\n" for row in tfidf_rows))
    return legacy_path, tfidf_path


def _write_candidate_pool(path: Path, outer_fold: int):
    payload = {
        "outer_fold": outer_fold,
        "valid_proposals": [
            {
                "action": "add",
                "name": "inlet_valve_status",
                "type": "categorical",
                "categories": ["absent", "present"],
                "roles": ["confounder", "effect_modifier"],
                "description": "Status documented before treatment.",
                "rationale": "multiple discovery signals",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class _FusionAgent:
    def __init__(self):
        self.calls = 0
        self.request_contexts = []

    def __call__(self, request):
        self.calls += 1
        self.request_contexts.append(request.context())
        evidence = next(
            block
            for block in request.evidence_blocks
            if "valve" in json.dumps(block.content).lower()
        )
        return {
            "selected_candidate_ids": ["candidate_0001"],
            "selection_notes": [
                {
                    "candidate_id": "candidate_0001",
                    "supporting_evidence_ids": [evidence.evidence_id],
                    "supporting_source_families": [evidence.source_families[0]],
                    "reason": "supported by fold-local evidence",
                }
            ],
        }


class _SpentEvidenceProvider:
    def __init__(self):
        self.calls = []

    def identity(self):
        return {"provider": "test_context_fit_spent_evidence_v1"}

    def get_spent_evidence_inputs(
        self,
        *,
        outer_fold,
        review_round,
        exact_spent_row_ids,
        exact_sealed_row_ids,
        spent_texts,
        spent_treatment,
        spent_outcome,
    ):
        self.calls.append(
            {
                "outer_fold": outer_fold,
                "review_round": review_round,
                "spent": exact_spent_row_ids,
                "sealed": exact_sealed_row_ids,
                "texts": spent_texts,
                "treatment": np.asarray(spent_treatment).copy(),
                "outcome": np.asarray(spent_outcome).copy(),
            }
        )
        provenance = FoldEvidenceProvenance(
            outer_fold=outer_fold,
            train_row_ids=exact_spent_row_ids,
            heldout_row_ids=exact_sealed_row_ids,
            scope="inner_train",
            inner_fold=review_round + 1,
            artifact_id=f"test-spent-{outer_fold}-{review_round}",
        )
        return [
            FoldEvidenceInput(
                TFIDF_TOPIC_SOURCE,
                {
                    "outer_fold": outer_fold,
                    "scope": "inner_train",
                    "inner_fold": review_round + 1,
                    "discovery": {
                        "topic_banks": {
                            "effect": {
                                "topics": [
                                    {
                                        "topic_id": "private_query_topic",
                                        "terms": [
                                            {
                                                "term": "inlet valve",
                                                "loading": 0.8,
                                            },
                                            {
                                                "term": "baseline status",
                                                "loading": 0.7,
                                            },
                                            {
                                                "term": "rotor grade",
                                                "loading": 0.6,
                                            },
                                        ],
                                    }
                                ]
                            }
                        }
                    },
                },
                provenance,
            )
        ]


def _json_content_sha256(value) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class _ProposalBaseAgent:
    def __init__(self):
        self.calls = 0

    def propose(self, context):
        self.calls += 1
        evidence = next(
            (row for row in context["evidence"] if "valve" in json.dumps(row["content"]).lower()),
            None,
        )
        if context["mode"] == "propose":
            if evidence is None:
                return {"proposals": []}
            return {
                "proposals": [
                    {
                        "name": "inlet_valve_status",
                        "type": "categorical",
                        "categories": ["absent", "present"],
                        "roles": ["confounder", "effect_modifier"],
                        "description": "Inlet valve status documented before treatment.",
                        "supporting_evidence_ids": [evidence["evidence_id"]],
                        "supporting_source_families": [evidence["source_families"][0]],
                        "rationale": "Supported by fold-local evidence.",
                    }
                ]
            }
        if evidence is None:
            return {"selected_candidate_ids": [], "selection_notes": []}
        candidate = context["candidates"][0]
        return {
            "selected_candidate_ids": [candidate["candidate_id"]],
            "selection_notes": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "supporting_evidence_ids": [evidence["evidence_id"]],
                    "supporting_source_families": [evidence["source_families"][0]],
                    "reason": "Supported by fold-local evidence.",
                }
            ],
        }


class _LeakyStagedAuditAgent:
    private_text = "private selector reasoning must never be persisted"

    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.proposal_agent = StagedAllEvidenceFusionAgent(
            base_agent,
            final_max_candidates=1,
        )

    def propose(self, context):
        return self.proposal_agent.propose(context)

    @property
    def last_stage_audit(self):
        audit = self.proposal_agent.last_stage_audit
        if audit is None:
            return None
        # The closed schema must reject raw text even under a benign-looking
        # key that a blacklist cannot classify.
        audit["trace"] = self.private_text
        return audit


class _SearchConfiguredProposalAgent:
    def __init__(
        self,
        *,
        enable_thinking,
        thinking_token_budget=None,
        max_tokens=25000,
    ):
        self.search_config = type(
            "SearchConfig",
            (),
            {
                "agent_enable_thinking": enable_thinking,
                "agent_thinking_token_budget": thinking_token_budget,
                "agent_max_tokens": max_tokens,
            },
        )()

    def propose(self, context):  # pragma: no cover - startup validation runs first
        raise AssertionError("proposal must not run during startup validation")


@pytest.mark.parametrize("staged", [False, True])
def test_runner_rejects_declared_and_effective_fusion_reasoning_mismatch(
    tmp_path,
    staged,
):
    base = _SearchConfiguredProposalAgent(enable_thinking=False)
    fusion_agent = StagedAllEvidenceFusionAgent(base, final_max_candidates=1) if staged else base

    with pytest.raises(
        ValueError,
        match=(
            "fusion reasoning configuration mismatch: .*"
            "fusion_enable_thinking=True.*agent_enable_thinking=False"
        ),
    ):
        AllEvidenceFusionRunner(
            dataset_path=tmp_path / "dataset.parquet",
            legacy_handoff_path=tmp_path / "legacy.jsonl",
            tfidf_handoff_path=tmp_path / "tfidf.jsonl",
            output_dir=tmp_path / "output",
            fusion_agent=fusion_agent,
            extraction_provider=_Extractor(),
            config=AllEvidenceFusionRunnerConfig(
                post_extraction_review_rounds=0,
                fusion_enable_thinking=True,
            ),
        )


def test_runner_accepts_matching_staged_fusion_reasoning_configuration(tmp_path):
    base = _SearchConfiguredProposalAgent(
        enable_thinking=True,
        thinking_token_budget=4096,
    )
    fusion_agent = StagedAllEvidenceFusionAgent(base, final_max_candidates=1)

    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=fusion_agent,
        extraction_provider=_Extractor(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            fusion_enable_thinking=True,
            fusion_max_tokens=25000,
            fusion_thinking_token_budget=4096,
        ),
    )

    assert runner.fusion_agent is fusion_agent


@pytest.mark.parametrize("staged", [False, True])
def test_runner_rejects_declared_and_effective_fusion_thinking_budget_mismatch(
    tmp_path,
    staged,
):
    base = _SearchConfiguredProposalAgent(
        enable_thinking=True,
        thinking_token_budget=2048,
    )
    fusion_agent = StagedAllEvidenceFusionAgent(base, final_max_candidates=1) if staged else base

    with pytest.raises(
        ValueError,
        match=(
            "fusion thinking token budget configuration mismatch: .*"
            "fusion_thinking_token_budget=4096.*agent_thinking_token_budget=2048"
        ),
    ):
        AllEvidenceFusionRunner(
            dataset_path=tmp_path / "dataset.parquet",
            legacy_handoff_path=tmp_path / "legacy.jsonl",
            tfidf_handoff_path=tmp_path / "tfidf.jsonl",
            output_dir=tmp_path / "output",
            fusion_agent=fusion_agent,
            extraction_provider=_Extractor(),
            config=AllEvidenceFusionRunnerConfig(
                post_extraction_review_rounds=0,
                fusion_enable_thinking=True,
                fusion_thinking_token_budget=4096,
            ),
        )


@pytest.mark.parametrize("invalid_budget", [0, -1, True, 1.5])
def test_runner_config_rejects_invalid_fusion_thinking_budget(invalid_budget):
    with pytest.raises(ValueError, match="fusion_thinking_token_budget"):
        AllEvidenceFusionRunnerConfig(fusion_thinking_token_budget=invalid_budget)


@pytest.mark.parametrize("invalid_enable", [None, 0, 1, "false"])
def test_runner_config_rejects_non_boolean_fusion_thinking_switch(invalid_enable):
    with pytest.raises(ValueError, match="fusion_enable_thinking"):
        AllEvidenceFusionRunnerConfig(fusion_enable_thinking=invalid_enable)


@pytest.mark.parametrize("invalid_max_tokens", [0, -1, True, 1.5, "25000", None])
def test_runner_config_rejects_invalid_fusion_max_tokens(invalid_max_tokens):
    with pytest.raises(ValueError, match="fusion_max_tokens"):
        AllEvidenceFusionRunnerConfig(fusion_max_tokens=invalid_max_tokens)


def test_runner_config_reserves_answer_tokens_beyond_fusion_thinking_budget():
    with pytest.raises(ValueError, match="strictly less than fusion_max_tokens"):
        AllEvidenceFusionRunnerConfig(
            fusion_max_tokens=4096,
            fusion_thinking_token_budget=4096,
        )


@pytest.mark.parametrize("staged", [False, True])
def test_runner_rejects_declared_and_effective_fusion_max_token_mismatch(
    tmp_path,
    staged,
):
    base = _SearchConfiguredProposalAgent(
        enable_thinking=True,
        thinking_token_budget=4096,
        max_tokens=24000,
    )
    fusion_agent = StagedAllEvidenceFusionAgent(base, final_max_candidates=1) if staged else base

    with pytest.raises(
        ValueError,
        match=(
            "fusion max token configuration mismatch: .*"
            "fusion_max_tokens=25000.*agent_max_tokens=24000"
        ),
    ):
        AllEvidenceFusionRunner(
            dataset_path=tmp_path / "dataset.parquet",
            legacy_handoff_path=tmp_path / "legacy.jsonl",
            tfidf_handoff_path=tmp_path / "tfidf.jsonl",
            output_dir=tmp_path / "output",
            fusion_agent=fusion_agent,
            extraction_provider=_Extractor(),
            config=AllEvidenceFusionRunnerConfig(
                post_extraction_review_rounds=0,
                fusion_enable_thinking=True,
                fusion_max_tokens=25000,
                fusion_thinking_token_budget=4096,
            ),
        )


def _write_effect_ngram_scores(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "feature": [
                "copper meadow",
                "copper meadow texture",
                "prerun alloy phase",
                "batch scheduling",
            ],
            "signed_score": [4.2, 3.8, 9.0, 8.0],
            "unsigned_score": [4.2, 3.8, 9.0, 8.0],
            "combined_importance": [4.1, 3.7, 9.0, 8.0],
            "eligible": [True, True, True, True],
        }
    ).to_parquet(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Extractor:
    @staticmethod
    def adaptive_review_contract_local_extraction():
        return True

    def ensure_features(self, dataset, specs):
        output = dataset.copy()
        for spec in specs:
            output[f"explicit_feat_{spec.name}"] = np.where(
                output["_oci_row_id"] < 6, "present", "absent"
            )
            output[f"explicit_feat_{spec.name}_missing"] = False
        return output


class _FinalRunnerSignalBackend:
    """Deterministic label-free transform used to exercise the runner boundary."""

    def __init__(self, mode="signal"):
        if mode not in {"signal", "constant"}:
            raise ValueError("unsupported test mode")
        self.mode = mode

    def identity(self):
        return {"backend": "final_runner_signal_v1", "mode": self.mode}

    def fit_predict(
        self,
        *,
        outer_fold,
        context_row_ids,
        context_texts,
        context_treatment,
        context_outcome,
        gate_row_ids,
        gate_texts,
        work_dir,
    ):
        del (
            outer_fold,
            context_row_ids,
            context_texts,
            context_treatment,
            context_outcome,
            gate_texts,
            work_dir,
        )
        row_ids = np.asarray(gate_row_ids, dtype=int)
        signal = ((row_ids // 2) % 2).astype(float)
        if self.mode == "constant":
            signal = np.zeros_like(signal)
        return ContextFitUpstreamPrediction(
            gate_row_ids=tuple(int(value) for value in row_ids),
            calibrated_source_names=("direct_bow_calibrated_tau",),
            calibrated_source_kinds=("bow_r_loss",),
            calibrated_source_values=signal[:, None],
            feature_names=(
                "query_treatment_basis",
                "query_outcome_basis",
                "query_effect_basis",
            ),
            feature_kinds=(
                "neural_query_treatment_moments",
                "neural_query_outcome_moments",
                "neural_query_effect_moments",
            ),
            feature_roles=(
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                OUTCOME_NUISANCE_FEATURE_ROLE,
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            ),
            feature_values=np.column_stack((signal, 1.0 - signal, 2.0 * signal)),
        )


class _NeuralRequirementVariantBackend(_FinalRunnerSignalBackend):
    def __init__(self, variant):
        super().__init__("signal")
        self.variant = variant

    def identity(self):
        return {
            "backend": "neural_requirement_variant_v1",
            "variant": self.variant,
        }

    def fit_predict(self, **kwargs):
        prediction = super().fit_predict(**kwargs)
        names = prediction.feature_names
        kinds = prediction.feature_kinds
        roles = prediction.feature_roles
        if self.variant == "name_spoof":
            names = (
                "neural_query_treatment_spoof",
                "neural_query_outcome_spoof",
                "neural_query_effect_spoof",
            )
            kinds = ("unrelated_moments",) * 3
        elif self.variant == "missing_effect":
            kinds = (*kinds[:2], "embedding_clustered")
        elif self.variant == "wrong_treatment_role":
            roles = (
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                *roles[1:],
            )
        else:
            raise RuntimeError("unsupported neural-requirement test variant")
        return ContextFitUpstreamPrediction(
            gate_row_ids=prediction.gate_row_ids,
            calibrated_source_names=prediction.calibrated_source_names,
            calibrated_source_kinds=prediction.calibrated_source_kinds,
            calibrated_source_values=prediction.calibrated_source_values,
            feature_names=names,
            feature_kinds=kinds,
            feature_roles=roles,
            feature_values=prediction.feature_values,
        )


class _RecordingFinalUpstreamProducer:
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = []

    def identity(self):
        return self.delegate.identity()

    def produce(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return self.delegate.produce(**kwargs)


def _dataset(tmp_path):
    frame = pd.DataFrame(
        {
            "text": [f"baseline note {index}" for index in range(12)],
            "treatment": [index % 2 for index in range(12)],
            "outcome": [(index // 2) % 2 for index in range(12)],
            "hidden_prompt": ["withheld generator material"] * 12,
            "event_timeline": ["post treatment"] * 12,
            "true_ite_prob": np.linspace(-0.1, 0.2, 12),
        }
    )
    path = tmp_path / "dataset.parquet"
    frame.to_parquet(path, index=False)
    return frame, path


class _PostExtractionReviewAgent:
    def __init__(self, response, events):
        self.response = response
        self.events = events
        self.calls = 0
        self.contexts = []
        self.last_response_trace = "private selector reasoning must never be persisted"

    def propose(self, context):
        self.calls += 1
        self.events.append("proposal")
        self.contexts.append(copy.deepcopy(context))
        return copy.deepcopy(self.response)


class _SelectiveReviewExtractor:
    def __init__(self, events):
        self.events = events
        self.calls = []
        self.row_id_calls = []
        self.description_calls = []

    @staticmethod
    def adaptive_review_contract_local_extraction():
        return True

    def ensure_features(self, dataset, specs):
        self.events.append("selective_extraction")
        self.calls.append([spec.name for spec in specs])
        self.row_id_calls.append(tuple(map(int, dataset["_oci_row_id"].tolist())))
        self.description_calls.append([str(spec.description or "") for spec in specs])
        output = dataset.copy()
        for spec in specs:
            revised = "revised" in str(spec.description or "").lower()
            output[f"explicit_feat_{spec.name}"] = np.where(
                output["_oci_row_id"] % 2 == 0,
                "present" if revised else "absent",
                "absent" if revised else "present",
            )
            output[f"explicit_feat_{spec.name}_missing"] = False
        return output


class _ConstantSelectiveReviewExtractor(_SelectiveReviewExtractor):
    def ensure_features(self, dataset, specs):
        self.events.append("selective_extraction")
        self.calls.append([spec.name for spec in specs])
        self.row_id_calls.append(tuple(map(int, dataset["_oci_row_id"].tolist())))
        self.description_calls.append([str(spec.description or "") for spec in specs])
        output = dataset.copy()
        for spec in specs:
            revised = "revised" in str(spec.description or "").lower()
            output[f"explicit_feat_{spec.name}"] = (
                "present"
                if revised
                else np.where(output["_oci_row_id"] % 2 == 0, "absent", "present")
            )
            output[f"explicit_feat_{spec.name}_missing"] = False
        return output


class _BoundReviewFeatureBank:
    def __init__(self, outer_fold, gate_ids, view):
        self.outer_fold = outer_fold
        self.gate_ids = gate_ids
        self.view = view

    def get_gate_feature_bank_view(self, *, outer_fold, exact_gate_row_ids):
        assert outer_fold == self.outer_fold
        assert exact_gate_row_ids == self.gate_ids
        return self.view


class _BindableReviewFeatureBank:
    def __init__(self, events):
        self.events = events

    def identity(self):
        return {"provider": "test_context_fit_feature_bank_v1"}

    def bind_fold(
        self,
        *,
        outer_fold,
        context,
        context_texts,
        gate_texts,
        exact_gate_row_ids,
    ):
        self.events.append("feature_bank_bind")
        assert len(context_texts) == len(context.row_ids)
        assert len(gate_texts) == len(exact_gate_row_ids)
        lineage = FitRowProvenance(fit_row_ids=frozenset(context.row_ids))
        context_lineage = tuple(
            FitRowProvenance(
                fit_row_ids=frozenset(
                    candidate
                    for candidate, candidate_fold in zip(context.row_ids, context.inner_fold_ids)
                    if candidate_fold != fold_id
                )
            )
            for fold_id in context.inner_fold_ids
        )
        view = GateFeatureBankView(
            row_ids=exact_gate_row_ids,
            feature_names=("opaque_raw_basis",),
            source_kinds=("whole_embedding_contrast",),
            consumer_roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,),
            values=np.arange(len(exact_gate_row_ids), dtype=float).reshape(-1, 1),
            fit_row_provenance=(lineage,),
            context_row_ids=context.row_ids,
            context_inner_fold_ids=context.inner_fold_ids,
            context_values=np.arange(len(context.row_ids), dtype=float).reshape(-1, 1),
            context_fit_row_provenance=(context_lineage,),
        )
        return _BoundReviewFeatureBank(outer_fold, exact_gate_row_ids, view)


class _FutureFitReviewFeatureBank(_BindableReviewFeatureBank):
    def __init__(self, events, *, future_row_id, response_cache_path):
        super().__init__(events)
        self.future_row_id = future_row_id
        self.response_cache_path = Path(response_cache_path)

    def bind_fold(
        self,
        *,
        outer_fold,
        context,
        context_texts,
        gate_texts,
        exact_gate_row_ids,
    ):
        self.events.append("feature_bank_bind")
        assert self.response_cache_path.exists()
        lineage = FitRowProvenance(fit_row_ids=frozenset((*context.row_ids, self.future_row_id)))
        view = GateFeatureBankView(
            row_ids=exact_gate_row_ids,
            feature_names=("opaque_future_fit_basis",),
            source_kinds=("whole_embedding_contrast",),
            consumer_roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,),
            values=np.arange(len(exact_gate_row_ids), dtype=float).reshape(-1, 1),
            fit_row_provenance=(lineage,),
        )
        return _BoundReviewFeatureBank(outer_fold, exact_gate_row_ids, view)


def _review_loop_rows():
    data = pd.DataFrame(
        {
            "_oci_row_id": np.arange(80),
            "text": [f"inlet valve note {index}" for index in range(80)],
            "treatment": np.tile([0, 0, 1, 1], 20),
            "outcome": np.tile([0.0, 1.0, 0.0, 1.0], 20),
        }
    )
    label_free = data[["_oci_row_id", "text"]].copy()
    extracted = label_free.copy()
    extracted["explicit_feat_inlet_valve_status"] = np.where(
        extracted["_oci_row_id"] % 2 == 0,
        "absent",
        "present",
    )
    extracted["explicit_feat_inlet_valve_status_missing"] = False
    return data, label_free, extracted


def _initial_spent_review_inputs(
    runner,
    data,
    extracted,
    *,
    outer_fold=1,
    train_ids=None,
    schedule=None,
):
    exact_train_ids = tuple(
        map(int, data["_oci_row_id"].tolist()) if train_ids is None else map(int, train_ids)
    )
    if schedule is None:
        outer_train = (
            data.set_index("_oci_row_id", drop=False)
            .loc[list(exact_train_ids)]
            .reset_index(drop=True)
        )
        schedule = runner._review_schedule(outer_train=outer_train, outer_fold=outer_fold)
    initial_spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
    initial_spent_extracted = (
        extracted.set_index("_oci_row_id", drop=False)
        .loc[list(initial_spent_ids)]
        .reset_index(drop=True)
    )
    return schedule, initial_spent_extracted


@pytest.mark.parametrize("invalid_scope", ["full", "spent_plus_one"])
def test_adaptive_review_rejects_initial_extraction_outside_exact_initial_spent_scope(
    tmp_path,
    invalid_scope,
):
    events = []
    initial_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Status documented at baseline before treatment.",
    }
    stop_response = {
        "schema_version": "all_evidence_post_extraction_review_response_v1",
        "operations": [
            {
                "action": "stop",
                "target_names": [],
                "contract": None,
                "supporting_diagnostic_ids": [],
                "supporting_evidence_ids": [],
                "reason": "No revision needed.",
            }
        ],
    }
    extractor = _SelectiveReviewExtractor(events)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=_PostExtractionReviewAgent(stop_response, events),
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_max_quality_retries=0,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _review_loop_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    if invalid_scope == "full":
        invalid_initial = extracted
    else:
        extra_id = schedule.row_ids(schedule.gate_fold_ids)[0]
        extra = extracted.loc[extracted["_oci_row_id"] == extra_id]
        invalid_initial = pd.concat([initial_spent_extracted, extra], ignore_index=True)

    with pytest.raises(
        ValueError,
        match="must contain exactly the ordered initial-spent rows",
    ):
        runner._run_post_extraction_review(
            data=data,
            label_free=label_free,
            outer_fold=1,
            train_ids=tuple(data["_oci_row_id"]),
            initial_specs=[initial_spec],
            initial_extracted=invalid_initial,
            fold_dir=tmp_path / "output" / "outer_fold_001",
            review_schedule=schedule,
        )

    assert events == []
    assert extractor.row_id_calls == []


def test_adaptive_extraction_scopes_spent_then_gate_then_post_freeze_outer_heldout(
    tmp_path,
    monkeypatch,
):
    events = []
    initial_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Status documented at baseline before treatment.",
    }
    revised_spec = {
        **initial_spec,
        "description": "Revised status documented at baseline before treatment.",
    }
    response = {
        "schema_version": "all_evidence_post_extraction_review_response_v1",
        "operations": [
            {
                "action": "revise",
                "target_names": ["inlet_valve_status"],
                "contract": revised_spec,
                "supporting_diagnostic_ids": ["diagnostic_0001"],
                "supporting_evidence_ids": ["evidence_0001"],
                "reason": "Exercise strict staged extraction scopes.",
            }
        ],
    }
    extractor = _SelectiveReviewExtractor(events)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=_PostExtractionReviewAgent(response, events),
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _review_loop_rows()
    train_ids = tuple(range(64))
    heldout_ids = tuple(range(64, 80))
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
        train_ids=train_ids,
    )
    spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
    gate_ids = schedule.row_ids(schedule.gate_fold_ids)
    real_quality_guard = runner._candidate_post_extraction_quality_guard

    def record_quality_guard(*args, **kwargs):
        events.append("quality_guard")
        return real_quality_guard(*args, **kwargs)

    def accept(*args, **kwargs):
        events.append("gate_acceptance")
        return GateAcceptanceDecision(
            accepted=True,
            reasons=(),
            current={},
            candidate={},
            guards={},
            decision_sha256="e" * 64,
        )

    monkeypatch.setattr(runner, "_candidate_post_extraction_quality_guard", record_quality_guard)
    monkeypatch.setattr(fusion_runner_module, "evaluate_untouched_gate_acceptance", accept)

    final_specs, final_extracted, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=train_ids,
        initial_specs=[initial_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )

    assert final_specs == [revised_spec]
    assert tuple(final_extracted["_oci_row_id"]) == tuple(label_free["_oci_row_id"])
    assert extractor.row_id_calls == [spent_ids, gate_ids, gate_ids, heldout_ids]
    assert events == [
        "proposal",
        "selective_extraction",
        "quality_guard",
        "selective_extraction",
        "selective_extraction",
        "gate_acceptance",
        "selective_extraction",
    ]
    assert not set(gate_ids).intersection(extractor.row_id_calls[0])
    assert not set(heldout_ids).intersection(extractor.row_id_calls[0])
    assert audit["initial_extraction_saw_only_initial_spent_rows"] is True
    assert audit["candidate_quality_extraction_saw_only_spent_rows"] is True
    assert audit["gate_extraction_started_only_after_candidate_quality_passed"] is True
    assert (
        audit["unconsumed_and_outer_heldout_text_extraction_started_after_registry_freeze"] is True
    )
    assert audit["post_freeze_extraction_completion"]["mode"] == (
        "remaining_rows_only_after_registry_freeze"
    )


def _rotor_grounding_review_rows(*, include_treatment_boundary: bool):
    data, label_free, extracted = _review_loop_rows()
    timing = "Process cycle started. " if include_treatment_boundary else ""
    data["text"] = [
        (
            f"RAW_NOTE_SPAN_{index:03d}. {timing}"
            f"Rotor grade {'present' if index % 2 == 0 else 'absent'}."
        )
        for index in range(len(data))
    ]
    label_free["text"] = data["text"]
    extracted = extracted.drop(
        columns=[
            "explicit_feat_inlet_valve_status",
            "explicit_feat_inlet_valve_status_missing",
        ]
    )
    extracted["explicit_feat_rotor_grade"] = np.where(
        extracted["_oci_row_id"] % 2 == 0,
        "absent",
        "present",
    )
    extracted["explicit_feat_rotor_grade_missing"] = False
    return data, label_free, extracted


def _unsafe_rotor_and_safe_baseline_review_rows():
    data, label_free, extracted = _rotor_grounding_review_rows(include_treatment_boundary=True)
    extracted["explicit_feat_inlet_valve_status"] = np.where(
        extracted["_oci_row_id"] % 2 == 0,
        "absent",
        "present",
    )
    extracted["explicit_feat_inlet_valve_status_missing"] = False
    return data, label_free, extracted


def _review_diagnostic_id(context, *, feature_name, kind="extraction_text_grounding"):
    return next(
        str(row["diagnostic_id"])
        for row in context["diagnostics"]
        if row.get("kind") == kind and row.get("feature_name") == feature_name
    )


class _TemporalSafetyRepairAgent:
    def __init__(self, *, first_action):
        self.first_action = first_action
        self.contexts = []

    def propose(self, context):
        self.contexts.append(copy.deepcopy(context))
        diagnostic_id = _review_diagnostic_id(context, feature_name="rotor_grade")
        if len(self.contexts) == 1 and self.first_action == "stop":
            operation = {
                "action": "stop",
                "target_names": [],
                "contract": None,
                "supporting_diagnostic_ids": [],
                "supporting_evidence_ids": [],
                "reason": "Attempt convergence before repairing the ontology mismatch.",
            }
        elif len(self.contexts) == 1 and self.first_action == "re_role":
            operation = {
                "action": "re_role",
                "target_names": ["rotor_grade"],
                "contract": {
                    "name": "rotor_grade",
                    "type": "categorical",
                    "categories": ["absent", "present"],
                    "roles": ["effect_modifier"],
                    "description": "Rotor grade documented before treatment.",
                },
                "supporting_diagnostic_ids": [diagnostic_id],
                "supporting_evidence_ids": [],
                "reason": "Attempt a role-only change that retains the unsafe values.",
            }
        else:
            operation = {
                "action": "drop",
                "target_names": ["rotor_grade"],
                "contract": None,
                "supporting_diagnostic_ids": [diagnostic_id],
                "supporting_evidence_ids": [],
                "reason": "Remove the unresolved post-treatment-only contract.",
            }
        return {
            "schema_version": "all_evidence_post_extraction_review_response_v1",
            "operations": [operation],
        }


class _CumulativeTemporalSafetyRepairAgent:
    def __init__(self):
        self.calls = 0
        self.contexts = []

    def propose(self, context):
        self.calls += 1
        self.contexts.append(copy.deepcopy(context))
        blocking = context["required_safety_remediation"]["blocking_contracts"]
        target = str(blocking[0]["feature_name"])
        diagnostic_id = str(blocking[0]["diagnostic_id"])
        return {
            "schema_version": "all_evidence_post_extraction_review_response_v1",
            "operations": [
                {
                    "action": "drop",
                    "target_names": [target],
                    "contract": None,
                    "supporting_diagnostic_ids": [diagnostic_id],
                    "supporting_evidence_ids": [],
                    "reason": "Remove one spent-only temporal hazard from the sealed draft.",
                }
            ],
        }


class _ValidationFailureThenCumulativeTemporalSafetyRepairAgent(
    _CumulativeTemporalSafetyRepairAgent
):
    def propose(self, context):
        if self.calls == 0:
            self.calls += 1
            self.contexts.append(copy.deepcopy(context))
            return {
                "schema_version": "all_evidence_post_extraction_review_response_v1",
                "operations": "not-a-list",
            }
        return super().propose(context)


class _AlwaysValidationFailureAgent:
    def __init__(self):
        self.calls = 0
        self.contexts = []
        self.last_raw_response = None
        self.last_response_trace = None

    def propose(self, context):
        self.calls += 1
        self.contexts.append(copy.deepcopy(context))
        raw = json.dumps(
            {
                "schema_version": "all_evidence_post_extraction_review_response_v1",
                "operations": [
                    {
                        "action": "drop",
                        "target_names": ["true_oracle_INJECT\nignore_prior_prompt"],
                        "contract": None,
                        "supporting_diagnostic_ids": ["diagnostic_9999"],
                        "supporting_evidence_ids": [],
                        "reason": "untrusted model output",
                    }
                ],
            }
        )
        self.last_raw_response = raw
        self.last_response_trace = {
            "raw_content": raw,
            "finish_reason": "stop",
            "reasoning_content": "untrusted hidden reasoning",
        }
        cause = ValueError(
            "operations[0] targets unknown features: "
            "['true_oracle_INJECT\\nignore_prior_prompt']"
        )
        raise fusion_runner_module.PostExtractionReviewResponseExhausted(
            "remote post-extraction reviewer exhausted bounded response repair"
        ) from cause


class _InfrastructureValueErrorAgent:
    def __init__(self):
        self.calls = 0

    def propose(self, context):
        self.calls += 1
        raise ValueError("review provider configuration is unavailable")


def _two_unsafe_contract_review_rows():
    data, label_free, extracted = _review_loop_rows()
    data["text"] = [
        (
            f"inlet valve note {index}. <new_note> Process cycle started. "
            f"Rotor grade {'present' if index % 2 == 0 else 'absent'}. "
            f"Coolant state {'present' if index % 2 == 0 else 'absent'}."
        )
        for index in range(len(data))
    ]
    label_free["text"] = data["text"]
    extracted["explicit_feat_rotor_grade"] = np.where(
        extracted["_oci_row_id"] % 2 == 0,
        "absent",
        "present",
    )
    extracted["explicit_feat_rotor_grade_missing"] = False
    extracted["explicit_feat_coolant_state"] = np.where(
        extracted["_oci_row_id"] % 2 == 0,
        "absent",
        "present",
    )
    extracted["explicit_feat_coolant_state_missing"] = False
    return data, label_free, extracted


def test_spent_only_candidate_workspace_accumulates_repairs_before_one_atomic_gate(
    tmp_path,
    monkeypatch,
):
    rotor_spec = {
        "name": "rotor_grade",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Rotor grade documented before treatment.",
    }
    coolant_spec = {
        "name": "coolant_state",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Coolant state documented before treatment.",
    }
    baseline_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Inlet valve status documented before treatment.",
    }
    events = []
    agent = _CumulativeTemporalSafetyRepairAgent()
    extractor = _SelectiveReviewExtractor(events)
    gate_calls = []

    def accept(current_context, current_gate, current_specs, candidate_specs, **kwargs):
        gate_calls.append(
            {
                "current": [str(spec["name"]) for spec in current_specs],
                "candidate": [str(spec["name"]) for spec in candidate_specs],
                "context_rows": tuple(current_context.row_ids),
                "gate_rows": tuple(current_gate.row_ids),
            }
        )
        events.append("gate_acceptance")
        return GateAcceptanceDecision(
            accepted=True,
            reasons=(),
            current={},
            candidate={},
            guards={},
            decision_sha256="8" * 64,
        )

    monkeypatch.setattr(fusion_runner_module, "evaluate_untouched_gate_acceptance", accept)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_max_operations=1,
            post_extraction_review_max_quality_retries=2,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _two_unsafe_contract_review_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
    gate_ids = schedule.row_ids(schedule.gate_fold_ids)

    specs, final_extracted, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[rotor_spec, coolant_spec, baseline_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )

    assert specs == [baseline_spec]
    assert tuple(final_extracted["_oci_row_id"]) == tuple(label_free["_oci_row_id"])
    assert agent.calls == 2
    assert [
        context["candidate_workspace"]["staged_attempt_count"] for context in agent.contexts
    ] == [
        0,
        1,
    ]
    assert [
        row["feature_name"]
        for row in agent.contexts[0]["required_safety_remediation"]["blocking_contracts"]
    ] == ["rotor_grade", "coolant_state"]
    assert [
        row["feature_name"]
        for row in agent.contexts[1]["required_safety_remediation"]["blocking_contracts"]
    ] == ["coolant_state"]
    assert [str(spec["name"]) for spec in agent.contexts[1]["current_contracts"]] == [
        "coolant_state",
        "inlet_valve_status",
    ]
    assert gate_calls == [
        {
            "current": ["rotor_grade", "coolant_state", "inlet_valve_status"],
            "candidate": ["inlet_valve_status"],
            "context_rows": spent_ids,
            "gate_rows": gate_ids,
        }
    ]
    assert extractor.row_id_calls == [gate_ids]
    assert events == ["selective_extraction", "gate_acceptance"]
    assert audit["candidate_workspace_stage_count"] == 1
    assert audit["gate_evaluated_proposal_count"] == 1
    assert audit["consumed_gate_count"] == 1
    attempt_audits = audit["round_audits"][0]["attempt_audits"]
    assert [row["status"] for row in attempt_audits] == [
        "candidate_workspace_advanced_pre_gate_retrying",
        "accepted",
    ]
    staged_body = json.loads(Path(attempt_audits[0]["path"]).read_text())["body"]
    assert staged_body["workspace_advanced"] is True
    assert staged_body["workspace_accepted"] is False
    assert staged_body["gate_accessed"] is False
    assert staged_body["gate_consumed"] is False
    assert staged_body["workspace_stage"]["hard_failure_count_before"] == 2
    assert staged_body["workspace_stage"]["hard_failure_count_after"] == 1
    assert len(staged_body["workspace_extraction_before_attempt_sha256"]) == 64
    assert len(staged_body["workspace_extraction_after_attempt_sha256"]) == 64
    assert (
        staged_body["workspace_extraction_before_attempt_sha256"]
        != staged_body["workspace_extraction_after_attempt_sha256"]
    )
    terminal_body = json.loads(Path(attempt_audits[1]["path"]).read_text())["body"]
    projection = terminal_body["gate_extraction"]["candidate_registry_projection_audit"]
    assert projection["removed_names"] == ["rotor_grade", "coolant_state"]
    assert projection["selective_reextraction_names"] == []
    assert projection["reused_extraction_names"] == ["inlet_valve_status"]
    assert len(terminal_body["candidate_spent_extraction_sha256"]) == 64
    assert len(terminal_body["gate_extraction"]["current_registry_extraction_sha256"]) == 64
    assert len(terminal_body["gate_extraction"]["candidate_registry_extraction_sha256"]) == 64

    # Request-bound caches deterministically reconstruct the same staged workspace.
    rerun = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[rotor_spec, coolant_spec, baseline_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )
    assert agent.calls == 2
    assert rerun[0] == specs
    assert rerun[2]["round_audits"] == audit["round_audits"]


def test_review_response_failure_is_sealed_retried_and_replayed_pre_gate(
    tmp_path,
    monkeypatch,
):
    rotor_spec = {
        "name": "rotor_grade",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Rotor grade documented before treatment.",
    }
    coolant_spec = {
        "name": "coolant_state",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Coolant state documented before treatment.",
    }
    baseline_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Inlet valve status documented before treatment.",
    }
    events = []
    agent = _ValidationFailureThenCumulativeTemporalSafetyRepairAgent()
    extractor = _SelectiveReviewExtractor(events)
    gate_calls = []

    def accept(*args, **kwargs):
        gate_calls.append("gate")
        return GateAcceptanceDecision(
            accepted=True,
            reasons=(),
            current={},
            candidate={},
            guards={},
            decision_sha256="7" * 64,
        )

    monkeypatch.setattr(fusion_runner_module, "evaluate_untouched_gate_acceptance", accept)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_max_operations=1,
            post_extraction_review_max_quality_retries=3,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _two_unsafe_contract_review_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    fold_dir = tmp_path / "output" / "outer_fold_001"

    specs, _, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[rotor_spec, coolant_spec, baseline_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=fold_dir,
        review_schedule=schedule,
    )

    assert specs == [baseline_spec]
    assert agent.calls == 3
    assert [
        context["candidate_workspace"]["staged_attempt_count"] for context in agent.contexts
    ] == [0, 0, 1]
    failure_feedback = [
        row
        for row in agent.contexts[1]["diagnostics"]
        if row.get("kind") == "review_response_validation_retry_feedback"
    ]
    assert len(failure_feedback) == 1
    assert failure_feedback[0]["same_gate_remains_sealed"] is True
    attempts = audit["round_audits"][0]["attempt_audits"]
    assert [row["status"] for row in attempts] == [
        "review_response_validation_failed_pre_gate_retrying",
        "candidate_workspace_advanced_pre_gate_retrying",
        "accepted",
    ]
    assert attempts[0]["gate_accessed"] is False
    assert attempts[0]["gate_consumed"] is False
    failure_round = json.loads(Path(attempts[0]["path"]).read_text())["body"]
    assert (
        failure_round["workspace_specs_before_attempt_sha256"]
        == failure_round["workspace_specs_after_attempt_sha256"]
    )
    assert (
        failure_round["workspace_extraction_before_attempt_sha256"]
        == failure_round["workspace_extraction_after_attempt_sha256"]
    )
    failure_path = (
        fold_dir / "post_extraction_review/round_001/attempt_001/immutable_review_failure.json"
    )
    failure_payload = json.loads(failure_path.read_text())
    assert failure_payload["schema_version"] == (
        fusion_runner_module.POST_EXTRACTION_REVIEW_FAILURE_SCHEMA_VERSION
    )
    assert failure_payload["content_sha256"] == _json_content_sha256(failure_payload["body"])
    assert failure_payload["body"]["raw_response_persisted"] is False
    assert failure_payload["body"]["raw_reasoning_persisted"] is False
    assert "raw_content" not in json.dumps(failure_payload)
    assert "reasoning_content" not in json.dumps(failure_payload)
    assert audit["response_validation_rejection_count"] == 1
    assert audit["response_validation_retry_count"] == 1
    assert audit["response_validation_retry_exhausted"] is False
    assert audit["valid_operation_proposal_count"] == 2
    assert gate_calls == ["gate"]

    rerun = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[rotor_spec, coolant_spec, baseline_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=fold_dir,
        review_schedule=schedule,
    )
    assert agent.calls == 3
    assert rerun[0] == specs
    assert rerun[2]["round_audits"] == audit["round_audits"]

    # A recomputed second authority for the same attempt is rejected even when
    # both individual wrappers are otherwise request-bound and valid.
    attempt_one = fold_dir / "post_extraction_review/round_001/attempt_001"
    attempt_two = fold_dir / "post_extraction_review/round_001/attempt_002"
    dual_response = json.loads((attempt_two / "immutable_review_response.json").read_text())
    attempt_one_request = json.loads((attempt_one / "immutable_review_request.json").read_text())
    dual_response["body"]["review_attempt"] = 1
    dual_response["body"]["request_sha256"] = attempt_one_request["body"]["request_sha256"]
    dual_response["content_sha256"] = _json_content_sha256(dual_response["body"])
    (attempt_one / "immutable_review_response.json").write_text(json.dumps(dual_response))
    with pytest.raises(RuntimeError, match="both a valid response and a failure audit"):
        runner._run_post_extraction_review(
            data=data,
            label_free=label_free,
            outer_fold=1,
            train_ids=tuple(data["_oci_row_id"]),
            initial_specs=[rotor_spec, coolant_spec, baseline_spec],
            initial_extracted=initial_spent_extracted,
            fold_dir=fold_dir,
            review_schedule=schedule,
        )


def test_review_response_exhaustion_raises_before_freezing_safe_baseline(
    tmp_path,
    monkeypatch,
):
    baseline_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Inlet valve status documented before treatment.",
    }
    events = []
    agent = _AlwaysValidationFailureAgent()
    extractor = _SelectiveReviewExtractor(events)
    monkeypatch.setattr(
        fusion_runner_module,
        "evaluate_untouched_gate_acceptance",
        lambda *args, **kwargs: pytest.fail("invalid reviewer response accessed the gate"),
    )
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_max_quality_retries=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _review_loop_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    fold_dir = tmp_path / "output" / "outer_fold_001"

    with pytest.raises(RuntimeError, match="exhausted bounded response validation"):
        runner._run_post_extraction_review(
            data=data,
            label_free=label_free,
            outer_fold=1,
            train_ids=tuple(data["_oci_row_id"]),
            initial_specs=[baseline_spec],
            initial_extracted=initial_spent_extracted,
            fold_dir=fold_dir,
            review_schedule=schedule,
        )

    assert agent.calls == 2
    assert extractor.row_id_calls == []
    assert "true_oracle" not in json.dumps(agent.contexts[1])
    assert "ignore_prior_prompt" not in json.dumps(agent.contexts[1])
    feedback = next(
        row
        for row in agent.contexts[1]["diagnostics"]
        if row.get("kind") == "review_response_validation_retry_feedback"
    )
    assert feedback["failure_code"] == "invalid_operation_target"
    assert feedback["failed_contract_names"] == []
    assert len(feedback["failure_issue_sha256"]) == 64
    round_dir = fold_dir / "post_extraction_review/round_001"
    statuses = []
    for attempt in (1, 2):
        failure_payload = json.loads(
            (round_dir / f"attempt_{attempt:03d}/immutable_review_failure.json").read_text()
        )
        serialized = json.dumps(failure_payload)
        assert "true_oracle" not in serialized
        assert "ignore_prior_prompt" not in serialized
        assert "untrusted hidden reasoning" not in serialized
        assert failure_payload["body"]["failed_contract_names"] == []
        assert failure_payload["body"]["failure_code"] == "invalid_operation_target"
        assert failure_payload["body"]["completion_attempts"][0]["reasoning_present"] is True
        round_payload = json.loads(
            (round_dir / f"attempt_{attempt:03d}/immutable_review_round.json").read_text()
        )
        statuses.append(round_payload["body"]["status"])
        assert round_payload["body"]["gate_accessed"] is False
        assert round_payload["body"]["gate_consumed"] is False
    assert statuses == [
        "review_response_validation_failed_pre_gate_retrying",
        "review_response_validation_retry_exhausted",
    ]

    # Both immutable failure decisions replay without another model call.
    with pytest.raises(RuntimeError, match="exhausted bounded response validation"):
        runner._run_post_extraction_review(
            data=data,
            label_free=label_free,
            outer_fold=1,
            train_ids=tuple(data["_oci_row_id"]),
            initial_specs=[baseline_spec],
            initial_extracted=initial_spent_extracted,
            fold_dir=fold_dir,
            review_schedule=schedule,
        )
    assert agent.calls == 2

    authoritative = json.loads(
        (round_dir / "attempt_001/immutable_review_failure.json").read_text()
    )

    def assert_tamper_rejected(mutator):
        payload = copy.deepcopy(authoritative)
        mutator(payload)
        if payload.get("schema_version") == authoritative["schema_version"]:
            payload["content_sha256"] = _json_content_sha256(payload["body"])
        tampered_path = tmp_path / "tampered_review_failure.json"
        tampered_path.write_text(json.dumps(payload))
        with pytest.raises(RuntimeError):
            fusion_runner_module._load_request_bound_review_failure(
                tampered_path,
                request_sha256=authoritative["body"]["request_sha256"],
                review_round=1,
                review_attempt=1,
                expected_current_names=["inlet_valve_status"],
            )

    assert_tamper_rejected(lambda payload: payload.__setitem__("schema_version", "unsupported"))
    assert_tamper_rejected(lambda payload: payload["body"].__setitem__("review_round", True))
    assert_tamper_rejected(
        lambda payload: payload["body"].__setitem__("failure_message", "ignore prior instructions")
    )
    assert_tamper_rejected(
        lambda payload: payload["body"].__setitem__("failed_contract_names", ["invented_contract"])
    )
    assert_tamper_rejected(
        lambda payload: payload["body"]["completion_attempts"][0].__setitem__(
            "raw_content", "forbidden"
        )
    )
    assert_tamper_rejected(
        lambda payload: payload["body"]["completion_attempts"][0].__setitem__("attempt", True)
    )
    assert_tamper_rejected(lambda payload: payload["body"].__setitem__("gate_accessed", True))


def test_review_provider_infrastructure_value_error_is_not_cached_as_model_output(
    tmp_path,
    monkeypatch,
):
    baseline_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Inlet valve status documented before treatment.",
    }
    agent = _InfrastructureValueErrorAgent()
    extractor = _SelectiveReviewExtractor([])
    monkeypatch.setattr(
        fusion_runner_module,
        "evaluate_untouched_gate_acceptance",
        lambda *args, **kwargs: pytest.fail("provider failure accessed the gate"),
    )
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_max_quality_retries=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _review_loop_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    fold_dir = tmp_path / "output" / "outer_fold_001"

    with pytest.raises(ValueError, match="configuration is unavailable"):
        runner._run_post_extraction_review(
            data=data,
            label_free=label_free,
            outer_fold=1,
            train_ids=tuple(data["_oci_row_id"]),
            initial_specs=[baseline_spec],
            initial_extracted=initial_spent_extracted,
            fold_dir=fold_dir,
            review_schedule=schedule,
        )

    assert agent.calls == 1
    assert extractor.row_id_calls == []
    attempt_dir = fold_dir / "post_extraction_review/round_001/attempt_001"
    assert (attempt_dir / "immutable_review_request.json").is_file()
    assert not (attempt_dir / "immutable_review_failure.json").exists()
    assert not (attempt_dir / "immutable_review_round.json").exists()


def test_candidate_workspace_exhaustion_discards_staged_repairs_without_gate_access(
    tmp_path,
    monkeypatch,
):
    specs = [
        {
            "name": name,
            "type": "categorical",
            "categories": ["absent", "present"],
            "roles": ["confounder"],
            "description": f"{label} documented before treatment.",
        }
        for name, label in (
            ("rotor_grade", "Rotor grade"),
            ("coolant_state", "Coolant state"),
            ("pressure_state", "Pressure state"),
        )
    ]
    baseline_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Inlet valve status documented before treatment.",
    }
    data, label_free, extracted = _two_unsafe_contract_review_rows()
    data["text"] = [
        f"{text} Pressure state {'present' if index % 2 == 0 else 'absent'}."
        for index, text in enumerate(data["text"])
    ]
    label_free["text"] = data["text"]
    extracted["text"] = data["text"]
    extracted["explicit_feat_pressure_state"] = np.where(
        extracted["_oci_row_id"] % 2 == 0,
        "absent",
        "present",
    )
    extracted["explicit_feat_pressure_state_missing"] = False
    agent = _CumulativeTemporalSafetyRepairAgent()
    events = []
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=_SelectiveReviewExtractor(events),
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_max_operations=1,
            post_extraction_review_max_quality_retries=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    monkeypatch.setattr(
        fusion_runner_module,
        "evaluate_untouched_gate_acceptance",
        lambda *args, **kwargs: pytest.fail("workspace exhaustion accessed the sealed gate"),
    )

    with pytest.raises(RuntimeError, match="cannot freeze an unresolved retained registry"):
        runner._run_post_extraction_review(
            data=data,
            label_free=label_free,
            outer_fold=1,
            train_ids=tuple(data["_oci_row_id"]),
            initial_specs=[*specs, baseline_spec],
            initial_extracted=initial_spent_extracted,
            fold_dir=tmp_path / "output" / "outer_fold_001",
            review_schedule=schedule,
        )

    assert agent.calls == 2
    assert [
        context["candidate_workspace"]["staged_attempt_count"] for context in agent.contexts
    ] == [
        0,
        1,
    ]
    assert events == []
    attempt_dir = tmp_path / "output" / "outer_fold_001" / "post_extraction_review" / "round_001"
    first = json.loads((attempt_dir / "attempt_001" / "immutable_review_round.json").read_text())[
        "body"
    ]
    second = json.loads((attempt_dir / "attempt_002" / "immutable_review_round.json").read_text())[
        "body"
    ]
    assert first["status"] == "candidate_workspace_advanced_pre_gate_retrying"
    assert first["workspace_advanced"] is True
    assert first["workspace_accepted"] is False
    assert second["status"] == "retained_registry_ontology_retry_exhausted"
    assert second["workspace_advanced"] is False
    assert second["workspace_accepted"] is False
    assert second["gate_accessed"] is False
    assert second["gate_consumed"] is False
    unresolved = json.loads(
        (
            tmp_path
            / "output"
            / "outer_fold_001"
            / "post_extraction_review"
            / "unresolved_retained_registry_ontology.json"
        ).read_text()
    )["body"]
    assert unresolved["unconsumed_gate_or_outer_heldout_text_extracted"] is False
    assert unresolved["retained_registry_ontology_guard"]["failed_names"] == sorted(
        ["rotor_grade", "coolant_state", "pressure_state"]
    )


class _RevisionWorkspaceAgent:
    def __init__(self, revised_baseline_spec):
        self.revised_baseline_spec = revised_baseline_spec
        self.contexts = []

    def propose(self, context):
        self.contexts.append(copy.deepcopy(context))
        attempt = len(self.contexts)
        if attempt == 1:
            diagnostic_id = _review_diagnostic_id(
                context,
                feature_name="valve_status",
            )
            evidence_id = next(
                str(row["evidence_id"])
                for row in context["sanitized_evidence_catalog"]
                if "inlet valve" in json.dumps(row).lower()
            )
            operation = {
                "action": "revise",
                "target_names": ["valve_status"],
                "contract": self.revised_baseline_spec,
                "supporting_diagnostic_ids": [diagnostic_id],
                "supporting_evidence_ids": [evidence_id],
                "reason": "Bind extraction to the explicitly documented baseline value.",
            }
        elif attempt == 2:
            diagnostic_id = _review_diagnostic_id(context, feature_name="coolant_state")
            coolant = next(
                spec for spec in context["current_contracts"] if spec["name"] == "coolant_state"
            )
            operation = {
                "action": "re_role",
                "target_names": ["coolant_state"],
                "contract": {**coolant, "roles": ["effect_modifier"]},
                "supporting_diagnostic_ids": [diagnostic_id],
                "supporting_evidence_ids": [],
                "reason": "Deliberately test a non-progressing role-only proposal.",
            }
        else:
            diagnostic_id = _review_diagnostic_id(context, feature_name="coolant_state")
            operation = {
                "action": "drop",
                "target_names": ["coolant_state"],
                "contract": None,
                "supporting_diagnostic_ids": [diagnostic_id],
                "supporting_evidence_ids": [],
                "reason": "Remove the remaining post-treatment-only contract.",
            }
        return {
            "schema_version": "all_evidence_post_extraction_review_response_v1",
            "operations": [operation],
        }


def test_workspace_keeps_staged_revision_across_bad_retry_and_reextracts_it_on_gate(
    tmp_path,
    monkeypatch,
):
    baseline_spec = {
        "name": "valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Valve status before treatment.",
    }
    revised_baseline_spec = {
        **baseline_spec,
        "description": "Revised valve status at intake before assignment.",
    }
    coolant_spec = {
        "name": "coolant_state",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Coolant state before treatment.",
    }
    data, label_free, extracted = _review_loop_rows()
    baseline_values = np.where(data["_oci_row_id"] % 2 == 0, "present", "absent")
    post_values = np.where(data["_oci_row_id"] % 2 == 0, "absent", "present")
    data["text"] = [
        (
            f"At intake valve status {baseline}; coolant state {baseline}. "
            "<new_note> Process cycle started."
        )
        for baseline in baseline_values
    ]
    label_free["text"] = data["text"]
    extracted["text"] = data["text"]
    extracted = extracted.drop(
        columns=[
            "explicit_feat_inlet_valve_status",
            "explicit_feat_inlet_valve_status_missing",
        ]
    )
    extracted["explicit_feat_valve_status"] = post_values
    extracted["explicit_feat_valve_status_missing"] = False
    extracted["explicit_feat_coolant_state"] = post_values
    extracted["explicit_feat_coolant_state_missing"] = False
    events = []
    extractor = _SelectiveReviewExtractor(events)
    agent = _RevisionWorkspaceAgent(revised_baseline_spec)

    def accept(*args, **kwargs):
        events.append("gate_acceptance")
        return GateAcceptanceDecision(
            accepted=True,
            reasons=(),
            current={},
            candidate={},
            guards={},
            decision_sha256="7" * 64,
        )

    monkeypatch.setattr(fusion_runner_module, "evaluate_untouched_gate_acceptance", accept)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_max_operations=1,
            post_extraction_review_max_quality_retries=3,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
    gate_ids = schedule.row_ids(schedule.gate_fold_ids)

    final_specs, _final_extracted, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[baseline_spec, coolant_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )

    assert final_specs == [revised_baseline_spec]
    assert [
        context["candidate_workspace"]["staged_attempt_count"] for context in agent.contexts
    ] == [
        0,
        1,
        1,
    ]
    assert [spec["name"] for spec in agent.contexts[2]["current_contracts"]] == [
        "valve_status",
        "coolant_state",
    ]
    assert agent.contexts[2]["current_contracts"][0] == revised_baseline_spec
    assert agent.contexts[2]["current_contracts"][1]["roles"] == ["confounder"]
    statuses = [row["status"] for row in audit["round_audits"][0]["attempt_audits"]]
    assert statuses == [
        "candidate_workspace_advanced_pre_gate_retrying",
        "retained_registry_ontology_rejected_pre_gate_retrying",
        "accepted",
    ]
    assert audit["candidate_workspace_stage_count"] == 1
    assert extractor.row_id_calls == [spent_ids, gate_ids, gate_ids]
    assert extractor.calls == [
        ["valve_status"],
        ["valve_status", "coolant_state"],
        ["valve_status"],
    ]
    terminal = json.loads(Path(audit["round_audits"][0]["attempt_audits"][-1]["path"]).read_text())[
        "body"
    ]
    projection = terminal["gate_extraction"]["candidate_registry_projection_audit"]
    assert projection["selective_reextraction_names"] == ["valve_status"]
    assert projection["removed_names"] == ["coolant_state"]
    assert terminal["workspace_stage_history_after_attempt"][0]["hard_failure_count_before"] == 2
    assert terminal["workspace_stage_history_after_attempt"][0]["hard_failure_count_after"] == 1
    staged_hash = terminal["workspace_stage_history_after_attempt"][0][
        "workspace_extraction_after_sha256"
    ]
    assert len(staged_hash) == 64
    assert len(terminal["candidate_spent_extraction_sha256"]) == 64
    assert (
        terminal["candidate_spent_extraction_sha256"]
        != terminal["accepted_round_baseline_spent_extraction_sha256"]
    )

    # Exact extraction hashes are part of each subsequent request, so replay
    # reconstructs the same revised values before cached responses are accepted.
    rerun = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[baseline_spec, coolant_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )
    assert len(agent.contexts) == 3
    assert rerun[0] == final_specs
    assert rerun[2]["round_audits"] == audit["round_audits"]


def test_adaptive_review_rejects_request_group_dependent_nonlocal_extractor(tmp_path):
    class GroupDependentExtractor(_SelectiveReviewExtractor):
        extraction_request_group_dependent = True

        @staticmethod
        def adaptive_review_contract_local_extraction():
            return False

    with pytest.raises(ValueError, match="contract-local extraction semantics"):
        AllEvidenceFusionRunner(
            dataset_path=tmp_path / "dataset.parquet",
            legacy_handoff_path=tmp_path / "legacy.jsonl",
            tfidf_handoff_path=tmp_path / "tfidf.jsonl",
            output_dir=tmp_path / "output",
            fusion_agent=_FusionAgent(),
            extraction_provider=GroupDependentExtractor([]),
            review_agent=_PostExtractionReviewAgent({}, []),
            review_spent_evidence_provider=_SpentEvidenceProvider(),
            config=AllEvidenceFusionRunnerConfig(
                post_extraction_review_rounds=1,
                allow_degraded_review_without_all_upstream=True,
            ),
        )

    class MissingCapabilityExtractor:
        def ensure_features(self, dataset, specs):  # pragma: no cover - startup only
            raise AssertionError("startup validation should fail first")

    with pytest.raises(ValueError, match="declare adaptive_review_contract_local_extraction"):
        AllEvidenceFusionRunner(
            dataset_path=tmp_path / "dataset.parquet",
            legacy_handoff_path=tmp_path / "legacy.jsonl",
            tfidf_handoff_path=tmp_path / "tfidf.jsonl",
            output_dir=tmp_path / "output",
            fusion_agent=_FusionAgent(),
            extraction_provider=MissingCapabilityExtractor(),
            review_agent=_PostExtractionReviewAgent({}, []),
            review_spent_evidence_provider=_SpentEvidenceProvider(),
            config=AllEvidenceFusionRunnerConfig(
                post_extraction_review_rounds=1,
                allow_degraded_review_without_all_upstream=True,
            ),
        )


def test_cumulative_workspace_stages_merge_then_applies_full_projection_on_gate(
    tmp_path,
    monkeypatch,
):
    left = {
        "name": "valve_left_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Left valve status before treatment.",
    }
    right = {
        "name": "valve_right_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Right valve status before treatment.",
    }
    coolant = {
        "name": "coolant_state",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Coolant state before treatment.",
    }
    merged = {
        "name": "valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Revised valve status at intake before assignment.",
    }

    class MergeAgent:
        def __init__(self):
            self.contexts = []

        def propose(self, context):
            self.contexts.append(copy.deepcopy(context))
            if len(self.contexts) == 1:
                evidence_id = next(
                    str(row["evidence_id"])
                    for row in context["sanitized_evidence_catalog"]
                    if "inlet valve" in json.dumps(row).lower()
                )
                operation = {
                    "action": "merge",
                    "target_names": ["valve_left_status", "valve_right_status"],
                    "contract": merged,
                    "supporting_diagnostic_ids": [
                        _review_diagnostic_id(context, feature_name="valve_left_status"),
                        _review_diagnostic_id(context, feature_name="valve_right_status"),
                    ],
                    "supporting_evidence_ids": [evidence_id],
                    "reason": "Unify redundant valve contracts at the baseline timepoint.",
                }
            else:
                operation = {
                    "action": "drop",
                    "target_names": ["coolant_state"],
                    "contract": None,
                    "supporting_diagnostic_ids": [
                        _review_diagnostic_id(context, feature_name="coolant_state")
                    ],
                    "supporting_evidence_ids": [],
                    "reason": "Remove the remaining post-treatment-only extraction.",
                }
            return {
                "schema_version": "all_evidence_post_extraction_review_response_v1",
                "operations": [operation],
            }

    data, label_free, extracted = _review_loop_rows()
    baseline_values = np.where(data["_oci_row_id"] % 2 == 0, "present", "absent")
    post_values = np.where(data["_oci_row_id"] % 2 == 0, "absent", "present")
    data["text"] = [
        (
            f"At intake valve left status {baseline}; valve right status {baseline}; "
            f"coolant state {baseline}. <new_note> Process cycle started. "
        )
        for baseline in baseline_values
    ]
    label_free["text"] = data["text"]
    extracted = label_free.copy()
    for name in ("valve_left_status", "valve_right_status", "coolant_state"):
        extracted[f"explicit_feat_{name}"] = post_values
        extracted[f"explicit_feat_{name}_missing"] = False
    events = []
    extractor = _SelectiveReviewExtractor(events)
    agent = MergeAgent()

    def accept(*args, **kwargs):
        events.append("gate_acceptance")
        return GateAcceptanceDecision(
            accepted=True,
            reasons=(),
            current={},
            candidate={},
            guards={},
            decision_sha256="6" * 64,
        )

    monkeypatch.setattr(fusion_runner_module, "evaluate_untouched_gate_acceptance", accept)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_max_operations=1,
            post_extraction_review_max_quality_retries=2,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
    gate_ids = schedule.row_ids(schedule.gate_fold_ids)

    final_specs, _final_extracted, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[left, right, coolant],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )

    assert final_specs == [merged]
    assert [row["status"] for row in audit["round_audits"][0]["attempt_audits"]] == [
        "candidate_workspace_advanced_pre_gate_retrying",
        "accepted",
    ]
    assert extractor.row_id_calls == [spent_ids, gate_ids, gate_ids]
    assert extractor.calls == [
        ["valve_status"],
        ["valve_left_status", "valve_right_status", "coolant_state"],
        ["valve_status"],
    ]
    terminal = json.loads(Path(audit["round_audits"][0]["attempt_audits"][-1]["path"]).read_text())[
        "body"
    ]
    assert [row["action"] for row in terminal["cumulative_operation_audit"]] == [
        "merge",
        "drop",
    ]
    projection = terminal["gate_extraction"]["candidate_registry_projection_audit"]
    assert projection["removed_names"] == [
        "valve_left_status",
        "valve_right_status",
        "coolant_state",
    ]
    assert projection["added_names"] == ["valve_status"]
    assert projection["selective_reextraction_names"] == ["valve_status"]


def test_gate_rejection_discards_cumulative_workspace_before_next_round(
    tmp_path,
    monkeypatch,
):
    rotor = {
        "name": "rotor_grade",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Rotor grade before treatment.",
    }
    coolant = {
        "name": "coolant_state",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Coolant state before treatment.",
    }
    baseline = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Inlet valve status before treatment.",
    }

    class RejectThenStopAgent:
        def __init__(self):
            self.contexts = []

        def propose(self, context):
            self.contexts.append(copy.deepcopy(context))
            if int(context["review_round"]) == 1:
                target = context["required_safety_remediation"]["blocking_contracts"][0]
                operation = {
                    "action": "drop",
                    "target_names": [target["feature_name"]],
                    "contract": None,
                    "supporting_diagnostic_ids": [target["diagnostic_id"]],
                    "supporting_evidence_ids": [],
                    "reason": "Remove one unsafe draft contract.",
                }
            else:
                operation = {
                    "action": "stop",
                    "target_names": [],
                    "contract": None,
                    "supporting_diagnostic_ids": [],
                    "supporting_evidence_ids": [],
                    "reason": "Test that rejected staged edits were discarded.",
                }
            return {
                "schema_version": "all_evidence_post_extraction_review_response_v1",
                "operations": [operation],
            }

    data, label_free, extracted = _two_unsafe_contract_review_rows()
    events = []
    extractor = _SelectiveReviewExtractor(events)
    agent = RejectThenStopAgent()
    gate_calls = []

    def reject(*args, **kwargs):
        gate_calls.append(True)
        events.append("gate_acceptance")
        return GateAcceptanceDecision(
            accepted=False,
            reasons=("test_gate_rejection",),
            current={},
            candidate={},
            guards={},
            decision_sha256="5" * 64,
        )

    monkeypatch.setattr(fusion_runner_module, "evaluate_untouched_gate_acceptance", reject)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=2,
            post_extraction_review_max_operations=1,
            post_extraction_review_max_quality_retries=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    first_gate_ids = schedule.row_ids((schedule.gate_fold_ids[0],))

    with pytest.raises(RuntimeError, match="cannot freeze an unresolved retained registry"):
        runner._run_post_extraction_review(
            data=data,
            label_free=label_free,
            outer_fold=1,
            train_ids=tuple(data["_oci_row_id"]),
            initial_specs=[rotor, coolant, baseline],
            initial_extracted=initial_spent_extracted,
            fold_dir=tmp_path / "output" / "outer_fold_001",
            review_schedule=schedule,
        )

    assert gate_calls == [True]
    assert extractor.row_id_calls == [first_gate_ids]
    round_two_contexts = [
        context for context in agent.contexts if int(context["review_round"]) == 2
    ]
    assert len(round_two_contexts) == 2
    assert round_two_contexts[0]["candidate_workspace"]["staged_attempt_count"] == 0
    assert [spec["name"] for spec in round_two_contexts[0]["current_contracts"]] == [
        "rotor_grade",
        "coolant_state",
        "inlet_valve_status",
    ]
    first_round_terminal = json.loads(
        (
            tmp_path
            / "output"
            / "outer_fold_001"
            / "post_extraction_review"
            / "round_001"
            / "attempt_002"
            / "immutable_review_round.json"
        ).read_text()
    )["body"]
    assert first_round_terminal["status"] == "rejected"
    assert first_round_terminal["workspace_accepted"] is False
    assert first_round_terminal["gate_accessed"] is True
    assert first_round_terminal["gate_consumed"] is True
    second_round_terminal = json.loads(
        (
            tmp_path
            / "output"
            / "outer_fold_001"
            / "post_extraction_review"
            / "round_002"
            / "attempt_002"
            / "immutable_review_round.json"
        ).read_text()
    )["body"]
    assert second_round_terminal["status"] == "unresolved_ontology_convergence_retry_exhausted"
    assert second_round_terminal["gate_accessed"] is False
    assert second_round_terminal["gate_consumed"] is False


@pytest.mark.parametrize("first_action", ["stop", "re_role"])
def test_unsafe_retained_registry_requires_extraction_repair_before_gate(
    tmp_path,
    monkeypatch,
    first_action,
):
    rotor_spec = {
        "name": "rotor_grade",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Rotor grade documented before treatment.",
    }
    baseline_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Baseline status documented before treatment.",
    }
    events = []
    agent = _TemporalSafetyRepairAgent(first_action=first_action)
    extractor = _SelectiveReviewExtractor(events)

    def accept(*args, **kwargs):
        events.append("gate_acceptance")
        return GateAcceptanceDecision(
            accepted=True,
            reasons=(),
            current={},
            candidate={},
            guards={},
            decision_sha256="9" * 64,
        )

    monkeypatch.setattr(fusion_runner_module, "evaluate_untouched_gate_acceptance", accept)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_max_quality_retries=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _unsafe_rotor_and_safe_baseline_review_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
    gate_ids = schedule.row_ids(schedule.gate_fold_ids)

    specs, final_extracted, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[rotor_spec, baseline_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )

    assert specs == [baseline_spec]
    assert tuple(final_extracted["_oci_row_id"]) == tuple(label_free["_oci_row_id"])
    assert extractor.row_id_calls == [gate_ids]
    assert not set(spent_ids).intersection(extractor.row_id_calls[0])
    assert events == ["selective_extraction", "gate_acceptance"]
    assert len(agent.contexts) == 2
    feedback = [
        row
        for row in agent.contexts[1]["diagnostics"]
        if row.get("kind") == "retained_registry_ontology_retry_feedback"
    ]
    assert len(feedback) == 1
    assert feedback[0]["ontology_mismatched_contract_names"] == ["rotor_grade"]
    assert audit["retained_ontology_rejection_count"] == 1
    assert audit["retained_ontology_retry_count"] == 1
    assert audit["unresolved_ontology_convergence_rejection_count"] == int(first_action == "stop")
    assert audit["gate_evaluated_proposal_count"] == 1
    first_attempt = audit["round_audits"][0]["attempt_audits"][0]
    assert first_attempt["gate_accessed"] is False
    assert first_attempt["gate_consumed"] is False
    first_body = json.loads(Path(first_attempt["path"]).read_text())["body"]
    guard = first_body["retained_registry_ontology_guard"]
    assert guard["failed_names"] == ["rotor_grade"]
    if first_action == "re_role":
        assert first_body["candidate_post_extraction_quality_guard"]["applicable"] is False
        assert (
            first_body["selective_extraction"]["role_only_columns_reused_without_remote_extraction"]
            is True
        )


def test_unresolved_unsafe_stop_exhaustion_fails_before_any_sealed_extraction(
    tmp_path,
    monkeypatch,
):
    rotor_spec = {
        "name": "rotor_grade",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Rotor grade documented before treatment.",
    }
    stop_response = {
        "schema_version": "all_evidence_post_extraction_review_response_v1",
        "operations": [
            {
                "action": "stop",
                "target_names": [],
                "contract": None,
                "supporting_diagnostic_ids": [],
                "supporting_evidence_ids": [],
                "reason": "Incorrectly request convergence with unresolved ontology.",
            }
        ],
    }
    events = []
    extractor = _SelectiveReviewExtractor(events)
    agent = _PostExtractionReviewAgent(stop_response, events)
    monkeypatch.setattr(
        fusion_runner_module,
        "evaluate_untouched_gate_acceptance",
        lambda *args, **kwargs: pytest.fail("unsafe convergence accessed a sealed gate"),
    )
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_max_quality_retries=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _unsafe_rotor_and_safe_baseline_review_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )

    with pytest.raises(
        RuntimeError,
        match="cannot freeze an unresolved retained registry",
    ):
        runner._run_post_extraction_review(
            data=data,
            label_free=label_free,
            outer_fold=1,
            train_ids=tuple(data["_oci_row_id"]),
            initial_specs=[rotor_spec],
            initial_extracted=initial_spent_extracted[
                [
                    "_oci_row_id",
                    "text",
                    "explicit_feat_rotor_grade",
                    "explicit_feat_rotor_grade_missing",
                ]
            ],
            fold_dir=tmp_path / "output" / "outer_fold_001",
            review_schedule=schedule,
        )

    assert events == ["proposal", "proposal"]
    assert extractor.row_id_calls == []
    review_dir = tmp_path / "output" / "outer_fold_001" / "post_extraction_review"
    first = json.loads(
        (review_dir / "round_001" / "attempt_001" / "immutable_review_round.json").read_text()
    )["body"]
    second = json.loads(
        (review_dir / "round_001" / "attempt_002" / "immutable_review_round.json").read_text()
    )["body"]
    assert first["status"] == "unresolved_ontology_convergence_rejected_retrying"
    assert second["status"] == "unresolved_ontology_convergence_retry_exhausted"
    assert first["gate_accessed"] is False
    assert second["gate_accessed"] is False
    failure = json.loads((review_dir / "unresolved_retained_registry_ontology.json").read_text())[
        "body"
    ]
    assert failure["retained_registry_ontology_guard"]["failed_names"] == ["rotor_grade"]
    assert failure["unconsumed_gate_or_outer_heldout_text_extracted"] is False


def test_safe_stop_ignores_ordinary_unchanged_contract_missingness(tmp_path):
    baseline_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Status documented at baseline before treatment.",
    }
    stop_response = {
        "schema_version": "all_evidence_post_extraction_review_response_v1",
        "operations": [
            {
                "action": "stop",
                "target_names": [],
                "contract": None,
                "supporting_diagnostic_ids": [],
                "supporting_evidence_ids": [],
                "reason": "No ontology revision is needed.",
            }
        ],
    }
    events = []
    extractor = _SelectiveReviewExtractor(events)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=_PostExtractionReviewAgent(stop_response, events),
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _review_loop_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    initial_spent_extracted["explicit_feat_inlet_valve_status"] = None
    initial_spent_extracted["explicit_feat_inlet_valve_status_missing"] = True
    gate_ids = schedule.row_ids(schedule.gate_fold_ids)

    specs, final_extracted, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[baseline_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )

    assert specs == [baseline_spec]
    assert tuple(final_extracted["_oci_row_id"]) == tuple(label_free["_oci_row_id"])
    assert extractor.row_id_calls == [gate_ids]
    assert events == ["proposal", "selective_extraction"]
    assert audit["stopped_by_agent_or_no_change"] is True
    assert audit["unresolved_ontology_convergence_rejection_count"] == 0
    assert audit["final_retained_registry_ontology_guard"]["passed"] is True
    body = json.loads(Path(audit["round_audits"][0]["path"]).read_text())["body"]
    assert body["status"] == "agent_stop"
    assert body["retained_registry_ontology_guard"]["passed"] is True


def test_retained_registry_safety_includes_high_confidence_category_mismatch():
    spec = {
        "name": "inlet_valve",
        "type": "categorical",
        "categories": ["negative", "positive"],
        "roles": ["effect_modifier"],
        "description": "Valve status documented before treatment.",
    }
    grounding = {
        "feature_name": "inlet_valve",
        "hard_failures": ["alternative_category_only_value_support"],
    }

    guard = AllEvidenceFusionRunner._retained_registry_ontology_from_grounding(
        [spec],
        [grounding],
    )

    assert guard["passed"] is False
    assert guard["failed_names"] == ["inlet_valve"]
    assert guard["failed_names_by_reason"]["alternative_category_only_value_support"] == [
        "inlet_valve"
    ]
    assert "categorical_ontology_alignment" in guard["safety_dimensions"]


def test_category_mismatch_safety_feedback_reaches_the_next_reasoning_attempt():
    safety = {
        "failed_names": ["inlet_valve"],
        "hard_failure_policy": [
            "alternative_category_only_value_support",
        ],
        "diagnostics": [
            {
                "diagnostic_id": "diagnostic_0001",
                "kind": "extraction_text_grounding",
                "feature_name": "inlet_valve",
                "hard_failures": ["alternative_category_only_value_support"],
            }
        ],
    }

    feedback = AllEvidenceFusionRunner._ontology_retry_feedback_diagnostic(
        safety,
        review_round=1,
        failed_attempt=1,
        response_sha256="a" * 64,
        operation_audit=[],
        proposal_kind="stop",
    )

    assert feedback["ontology_mismatched_contract_names"] == ["inlet_valve"]
    assert feedback["failed_retained_registry_ontology"][0]["hard_failures"] == [
        "alternative_category_only_value_support"
    ]
    assert "different declared category" in feedback["non_repeat_guidance"]


def test_review_loop_freezes_proposal_selectively_reextracts_and_uses_separate_frames(
    tmp_path,
    monkeypatch,
):
    events = []
    initial_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Status documented at baseline before treatment.",
    }
    revised_spec = {
        **initial_spec,
        "description": "Revised status measured at baseline before treatment.",
    }
    response = {
        "schema_version": "all_evidence_post_extraction_review_response_v1",
        "operations": [
            {
                "action": "revise",
                "target_names": ["inlet_valve_status"],
                "contract": revised_spec,
                "supporting_diagnostic_ids": ["diagnostic_0001"],
                "supporting_evidence_ids": ["evidence_0001"],
                "reason": "Clarify the baseline measurement contract.",
            }
        ],
    }
    agent = _PostExtractionReviewAgent(response, events)
    extractor = _SelectiveReviewExtractor(events)
    observed = {}

    def accept(context, gate, current_specs, candidate_specs, **kwargs):
        events.append("gate_acceptance")
        assert isinstance(kwargs["feature_bank_view"], GateFeatureBankView)
        candidate_context = kwargs["candidate_context"]
        current = context.extracted["explicit_feat_inlet_valve_status"].tolist()
        candidate = candidate_context.extracted["explicit_feat_inlet_valve_status"].tolist()
        assert current != candidate
        assert gate.row_ids == kwargs["candidate_gate"].row_ids
        observed["current"] = current
        observed["candidate"] = candidate
        return GateAcceptanceDecision(
            accepted=True,
            reasons=(),
            current={},
            candidate={},
            guards={},
            decision_sha256="a" * 64,
        )

    monkeypatch.setattr(fusion_runner_module, "evaluate_untouched_gate_acceptance", accept)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        review_gate_feature_bank_provider=_BindableReviewFeatureBank(events),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_min_partition_rows=8,
            require_review_feature_banks=True,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _review_loop_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
    gate_ids = schedule.row_ids(schedule.gate_fold_ids)

    final_specs, final_extracted, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[initial_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )

    assert events == [
        "proposal",
        "selective_extraction",
        "selective_extraction",
        "selective_extraction",
        "feature_bank_bind",
        "gate_acceptance",
    ]
    assert extractor.calls == [["inlet_valve_status"]] * 3
    assert extractor.row_id_calls == [spent_ids, gate_ids, gate_ids]
    assert final_specs == [revised_spec]
    assert (
        final_extracted["explicit_feat_inlet_valve_status"].tolist()
        != extracted["explicit_feat_inlet_valve_status"].tolist()
    )
    assert audit["round_audits"][0]["status"] == "accepted"
    assert audit["gate_provider_bind_input_contract"] == (
        "spent_observable_rows_plus_exact_gate_ids_and_text_only_v1"
    )
    assert audit["gate_treatment_or_outcome_supplied_to_providers"] is False
    context = agent.contexts[0]
    assert context["sealed_gate"] == {
        "aggregates_exposed": False,
        "feature_bank_values_exposed": False,
        "outcome_exposed": False,
        "row_ids_exposed": False,
        "source_values_exposed": False,
        "text_exposed": False,
        "treatment_exposed": False,
    }
    assert '"row_ids":' not in json.dumps(context)
    assert "inlet valve note 79" not in json.dumps(context)
    assert context["evidence_sanitization"]["spent_only_source_blocks_retained"] >= 1
    assert context["evidence_sanitization"]["full_outer_discovery_blocks_available"] is False
    assert "inlet valve" in json.dumps(context)
    assert "private_query_topic" not in json.dumps(context)
    serialized_audits = "".join(Path(row["path"]).read_text() for row in audit["round_audits"])
    assert agent.last_response_trace not in serialized_audits

    # A retry reconstructs and authenticates the same context/response cache;
    # it does not ask the remote review agent a second time.
    rerun = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[initial_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )
    assert agent.calls == 1
    assert rerun[0] == final_specs


def test_review_role_only_revision_reuses_extracted_columns(tmp_path, monkeypatch):
    events = []
    initial_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Status documented at baseline before treatment.",
    }
    rerolled = {**initial_spec, "roles": ["effect_modifier"]}
    response = {
        "schema_version": "all_evidence_post_extraction_review_response_v1",
        "operations": [
            {
                "action": "re_role",
                "target_names": ["inlet_valve_status"],
                "contract": rerolled,
                "supporting_diagnostic_ids": ["diagnostic_0001"],
                "supporting_evidence_ids": [],
                "reason": "Retain values but narrow the causal role.",
            }
        ],
    }
    agent = _PostExtractionReviewAgent(response, events)
    extractor = _SelectiveReviewExtractor(events)

    def reject(*args, **kwargs):
        events.append("gate_acceptance")
        return GateAcceptanceDecision(
            accepted=False,
            reasons=("penalized_relative_r_loss_not_improved",),
            current={},
            candidate={},
            guards={},
            decision_sha256="b" * 64,
        )

    monkeypatch.setattr(fusion_runner_module, "evaluate_untouched_gate_acceptance", reject)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _review_loop_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    gate_ids = schedule.row_ids(schedule.gate_fold_ids)
    final_specs, final_extracted, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[initial_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )

    assert extractor.calls == [["inlet_valve_status"]]
    assert extractor.row_id_calls == [gate_ids]
    assert events == ["proposal", "selective_extraction", "gate_acceptance"]
    assert final_specs == [initial_spec]
    assert final_extracted.equals(extracted)
    round_body = json.loads(Path(audit["round_audits"][0]["path"]).read_text())["body"]
    assert round_body["selective_extraction"]["selective_reextraction_spec_count"] == 0
    assert (
        round_body["selective_extraction"]["role_only_columns_reused_without_remote_extraction"]
        is True
    )
    assert round_body["gate"]["consumed_after_gate_evaluation"] is True


def test_spent_only_initial_request_is_invariant_to_sealed_gate_values(tmp_path):
    data, _label_free, _extracted = _review_loop_rows()
    provider = _SpentEvidenceProvider()
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=_Extractor(),
        review_agent=lambda context: {},
        review_spent_evidence_provider=provider,
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    schedule = runner._review_schedule(outer_train=data, outer_fold=1)
    first, first_audit = runner._spent_fusion_request(
        data=data,
        schedule=schedule,
        spent_fold_ids=schedule.initial_spent_fold_ids,
        outer_fold=1,
        review_round=0,
    )
    perturbed = data.copy()
    sealed = schedule.row_ids(schedule.gate_fold_ids)
    mask = perturbed["_oci_row_id"].isin(sealed)
    perturbed.loc[mask, "text"] = "future gate changed completely"
    perturbed.loc[mask, "treatment"] = 1 - perturbed.loc[mask, "treatment"]
    perturbed.loc[mask, "outcome"] = 1.0 - perturbed.loc[mask, "outcome"]
    second, second_audit = runner._spent_fusion_request(
        data=perturbed,
        schedule=schedule,
        spent_fold_ids=schedule.initial_spent_fold_ids,
        outer_fold=1,
        review_round=0,
    )

    assert first.context() == second.context()
    assert first_audit == second_audit
    assert all("future gate changed" not in text for text in provider.calls[-1]["texts"])
    assert first_audit["future_gate_text_or_labels_supplied_to_provider"] is False


def test_spent_context_epoch_rejects_nonprefix_or_consumer_round_drift(tmp_path):
    data, _label_free, _extracted = _review_loop_rows()
    provider = _SpentEvidenceProvider()
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=_Extractor(),
        review_agent=lambda context: {},
        review_spent_evidence_provider=provider,
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=2,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    schedule = runner._review_schedule(outer_train=data, outer_fold=2)

    with pytest.raises(ValueError, match="exact consumed review-gate prefix"):
        runner._spent_fusion_request(
            data=data,
            schedule=schedule,
            spent_fold_ids=(
                *schedule.initial_spent_fold_ids,
                schedule.gate_fold_ids[1],
            ),
            outer_fold=2,
            review_round=2,
        )
    with pytest.raises(ValueError, match="consumer review_round - 1"):
        runner._spent_fusion_request(
            data=data,
            schedule=schedule,
            spent_fold_ids=(
                *schedule.initial_spent_fold_ids,
                schedule.gate_fold_ids[0],
            ),
            outer_fold=2,
            review_round=1,
        )

    assert provider.calls == []


def test_spent_evidence_sanitizer_keeps_concepts_and_aggregates_only():
    sanitized = AllEvidenceFusionRunner._sanitize_spent_evidence_catalog(
        [
            {
                "evidence_id": "evidence_0001",
                "source_families": ["neural_query_moments"],
                "role_hint": "effect_modifier",
                "content": {
                    "kind": "query_moment",
                    "query_id": "private_query_name",
                    "term": "inlet valve",
                    "fit_standardized_score": 1.25,
                    "summaries": [
                        "short attention term",
                        "record 123456",
                    ],
                    "concept_scores": [
                        {"concept": "quartz load index", "score": 0.7},
                        {"concept": "record id 998877", "score": 0.9},
                    ],
                    "summary": "raw record-level sentence",
                    "retrieved_training_excerpts": ["identifiable note text"],
                    "values": [0.2, 0.8],
                },
            }
        ]
    )

    serialized = json.dumps(sanitized)
    assert "inlet valve" in serialized
    assert "fit_standardized_score" in serialized
    assert "short attention term" in serialized
    assert "quartz load index" in serialized
    assert "record 123456" not in serialized
    assert "record id 998877" not in serialized
    assert "query_id_sha256" in serialized
    assert "private_query_name" not in serialized
    assert "raw record-level sentence" not in serialized
    assert "identifiable note text" not in serialized
    assert '"values"' not in serialized


def test_candidate_quality_failure_exhausts_bounded_retries_before_gate_access(
    tmp_path,
    monkeypatch,
):
    events = []
    initial_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Status documented at baseline before treatment.",
    }
    revised_spec = {
        **initial_spec,
        "description": "Revised status documented at baseline before treatment.",
    }
    response = {
        "schema_version": "all_evidence_post_extraction_review_response_v1",
        "operations": [
            {
                "action": "revise",
                "target_names": ["inlet_valve_status"],
                "contract": revised_spec,
                "supporting_diagnostic_ids": ["diagnostic_0001"],
                "supporting_evidence_ids": ["evidence_0001"],
                "reason": "Clarify the extraction contract.",
            }
        ],
    }
    extractor = _ConstantSelectiveReviewExtractor(events)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=_PostExtractionReviewAgent(response, events),
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    monkeypatch.setattr(
        fusion_runner_module,
        "evaluate_untouched_gate_acceptance",
        lambda *args, **kwargs: pytest.fail("quality failure accessed the gate"),
    )
    data, label_free, extracted = _review_loop_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
    gate_ids = schedule.row_ids(schedule.gate_fold_ids)
    specs, final_extracted, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[initial_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )

    assert specs == [initial_spec]
    assert final_extracted.equals(extracted)
    assert events == ["proposal", "selective_extraction"] * 3 + ["selective_extraction"]
    assert extractor.row_id_calls == [spent_ids, spent_ids, spent_ids, gate_ids]
    assert audit["candidate_quality_rejection_count"] == 3
    assert audit["candidate_quality_retry_count"] == 2
    assert audit["review_attempt_count"] == 3
    assert audit["quality_retry_exhausted"] is True
    assert audit["stopped_by_agent_or_no_change"] is False
    assert audit["consumed_gate_count"] == 0
    assert audit["round_audits"][0]["status"] == "quality_retry_exhausted"
    assert [row["status"] for row in audit["round_audits"][0]["attempt_audits"]] == [
        "candidate_quality_rejected_pre_gate_retrying",
        "candidate_quality_rejected_pre_gate_retrying",
        "quality_retry_exhausted",
    ]
    body = json.loads(Path(audit["round_audits"][0]["path"]).read_text())["body"]
    assert body["status"] == "quality_retry_exhausted"
    assert body["candidate_post_extraction_quality_guard"]["failed_names"] == [
        "inlet_valve_status"
    ]
    assert body["gate_accessed"] is False
    assert body["gate_consumed"] is False
    assert body["same_gate_remains_sealed"] is True
    retry_diagnostics = [
        row
        for row in body["sanitized_context"]["diagnostics"]
        if row.get("kind") == "candidate_quality_retry_feedback"
    ]
    assert len(retry_diagnostics) == 2
    assert all(row["same_gate_remains_sealed"] for row in retry_diagnostics)


def test_candidate_quality_retry_revises_again_before_spending_same_gate(
    tmp_path,
    monkeypatch,
):
    events = []
    initial_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Status documented at baseline before treatment.",
    }
    first_revision = {
        **initial_spec,
        "description": "First revised status documented at baseline before treatment.",
    }
    repaired_revision = {
        **initial_spec,
        "description": "Repaired status documented at baseline before treatment.",
    }

    class RetryAgent:
        def __init__(self):
            self.contexts = []

        def propose(self, context):
            self.contexts.append(copy.deepcopy(context))
            events.append("proposal")
            if len(self.contexts) == 1:
                contract = first_revision
                diagnostic_id = "diagnostic_0001"
            else:
                retry_rows = [
                    row
                    for row in context["diagnostics"]
                    if row.get("kind") == "candidate_quality_retry_feedback"
                ]
                assert len(retry_rows) == 1
                assert retry_rows[0]["same_gate_remains_sealed"] is True
                assert retry_rows[0]["failed_contract_names"] == ["inlet_valve_status"]
                contract = repaired_revision
                diagnostic_id = retry_rows[0]["diagnostic_id"]
            return {
                "schema_version": "all_evidence_post_extraction_review_response_v1",
                "operations": [
                    {
                        "action": "revise",
                        "target_names": ["inlet_valve_status"],
                        "contract": contract,
                        "supporting_diagnostic_ids": [diagnostic_id],
                        "supporting_evidence_ids": ["evidence_0001"],
                        "reason": "Repair the spent-only extraction quality failure.",
                    }
                ],
            }

    class RetryExtractor:
        def __init__(self):
            self.calls = []
            self.row_id_calls = []

        @staticmethod
        def adaptive_review_contract_local_extraction():
            return True

        def ensure_features(self, dataset, specs):
            events.append("selective_extraction")
            self.calls.append([spec.name for spec in specs])
            self.row_id_calls.append(tuple(map(int, dataset["_oci_row_id"].tolist())))
            output = dataset.copy()
            for spec in specs:
                description = str(spec.description or "")
                if description.startswith("First revised"):
                    values = "present"
                elif description.startswith("Repaired"):
                    values = np.where(output["_oci_row_id"] % 2 == 0, "present", "absent")
                else:
                    values = np.where(output["_oci_row_id"] % 2 == 0, "absent", "present")
                output[f"explicit_feat_{spec.name}"] = values
                output[f"explicit_feat_{spec.name}_missing"] = False
            return output

    def accept(*args, **kwargs):
        events.append("gate_acceptance")
        return GateAcceptanceDecision(
            accepted=True,
            reasons=(),
            current={},
            candidate={},
            guards={},
            decision_sha256="c" * 64,
        )

    monkeypatch.setattr(fusion_runner_module, "evaluate_untouched_gate_acceptance", accept)
    agent = RetryAgent()
    extractor = RetryExtractor()
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_max_quality_retries=2,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _review_loop_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
    gate_ids = schedule.row_ids(schedule.gate_fold_ids)
    specs, _final_extracted, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[initial_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )

    assert specs == [repaired_revision]
    assert events == [
        "proposal",
        "selective_extraction",
        "proposal",
        "selective_extraction",
        "selective_extraction",
        "selective_extraction",
        "gate_acceptance",
    ]
    assert extractor.calls == [["inlet_valve_status"]] * 4
    assert extractor.row_id_calls == [spent_ids, spent_ids, gate_ids, gate_ids]
    assert [context["review_attempt"] for context in agent.contexts] == [1, 2]
    assert audit["candidate_quality_rejection_count"] == 1
    assert audit["candidate_quality_retry_count"] == 1
    assert audit["gate_evaluated_proposal_count"] == 1
    assert audit["consumed_gate_count"] == 1
    assert audit["quality_retry_exhausted"] is False
    round_audit = audit["round_audits"][0]
    assert round_audit["status"] == "accepted"
    assert round_audit["attempt_count"] == 2
    assert [row["status"] for row in round_audit["attempt_audits"]] == [
        "candidate_quality_rejected_pre_gate_retrying",
        "accepted",
    ]
    for attempt in (1, 2):
        attempt_dir = (
            tmp_path
            / "output"
            / "outer_fold_001"
            / "post_extraction_review"
            / "round_001"
            / f"attempt_{attempt:03d}"
        )
        assert (attempt_dir / "immutable_review_request.json").is_file()
        assert (attempt_dir / "immutable_review_response.json").is_file()
        assert (attempt_dir / "immutable_review_round.json").is_file()

    # Process retry authenticates both attempt-bound responses and does not ask
    # the remote reasoning agent again.
    rerun = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[initial_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )
    assert len(agent.contexts) == 2
    assert rerun[0] == specs
    assert rerun[2]["round_audits"] == audit["round_audits"]


def test_temporal_wording_does_not_reject_before_gate_and_uses_exact_spent_texts(
    tmp_path,
    monkeypatch,
):
    events = []
    initial_spec = {
        "name": "rotor_grade",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Rotor grade documented before treatment.",
    }
    revised_spec = {
        **initial_spec,
        "description": "Revised Rotor grade documented before treatment.",
    }
    response = {
        "schema_version": "all_evidence_post_extraction_review_response_v1",
        "operations": [
            {
                "action": "revise",
                "target_names": ["rotor_grade"],
                "contract": revised_spec,
                "supporting_diagnostic_ids": ["diagnostic_0001"],
                "supporting_evidence_ids": ["evidence_0001"],
                "reason": "Clarify the pretreatment extraction contract.",
            }
        ],
    }
    provider = _SpentEvidenceProvider()
    agent = _PostExtractionReviewAgent(response, events)
    extractor = _SelectiveReviewExtractor(events)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=provider,
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_max_quality_retries=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _rotor_grounding_review_rows(include_treatment_boundary=True)
    extracted["explicit_feat_rotor_grade"] = np.where(
        extracted["_oci_row_id"] % 2 == 0, "present", "absent"
    )
    schedule = runner._review_schedule(outer_train=data, outer_fold=1)
    spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
    sealed_ids = schedule.row_ids(schedule.gate_fold_ids)
    text_by_id = label_free.set_index("_oci_row_id")["text"]
    exact_spent_texts = tuple(text_by_id.loc[list(spent_ids)].astype(str))
    sealed_texts = tuple(text_by_id.loc[list(sealed_ids)].astype(str))
    _, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
        schedule=schedule,
    )

    grounding_calls = []
    real_grounding = fusion_runner_module.build_extraction_grounding_diagnostics

    def record_grounding(frame, texts, specs, **kwargs):
        grounding_calls.append(tuple(texts))
        return real_grounding(frame, texts, specs, **kwargs)

    monkeypatch.setattr(
        fusion_runner_module,
        "build_extraction_grounding_diagnostics",
        record_grounding,
    )

    def accept(*args, **kwargs):
        events.append("gate_acceptance")
        return GateAcceptanceDecision(
            accepted=True,
            reasons=(),
            current={},
            candidate={},
            guards={},
            decision_sha256="3" * 64,
        )

    monkeypatch.setattr(fusion_runner_module, "evaluate_untouched_gate_acceptance", accept)

    specs, final_extracted, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[initial_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )

    assert specs == [revised_spec]
    assert tuple(final_extracted["_oci_row_id"]) == tuple(label_free["_oci_row_id"])
    assert events == [
        "proposal",
        "selective_extraction",
        "selective_extraction",
        "selective_extraction",
        "gate_acceptance",
    ]
    assert extractor.row_id_calls == [spent_ids, sealed_ids, sealed_ids]
    assert audit["candidate_quality_rejection_count"] == 0
    assert audit["candidate_quality_retry_count"] == 0
    assert audit["gate_evaluated_proposal_count"] == 1
    assert audit["consumed_gate_count"] == 1
    assert len(grounding_calls) == 3
    assert grounding_calls[0] == exact_spent_texts
    assert grounding_calls[1] == exact_spent_texts
    assert grounding_calls[2] == tuple(text_by_id.loc[list((*spent_ids, *sealed_ids))].astype(str))
    assert all(call["texts"] == exact_spent_texts for call in provider.calls)

    for context in agent.contexts:
        serialized_context = json.dumps(context)
        remediation = context["required_safety_remediation"]
        assert remediation["computed_from_exact_spent_rows_only"] is True
        assert remediation["sealed_gate_used"] is False
        assert remediation["all_listed_contracts_must_be_resolved_before_gate"] is True
        assert remediation["hard_failure_policy"] == ["alternative_category_only_value_support"]
        assert context["source_text_temporal_policy"]["temporal_boundary_enforced"] is False
        assert remediation["blocking_contract_count"] == len(remediation["blocking_contracts"])
        available_evidence_ids = {
            row["evidence_id"] for row in context["sanitized_evidence_catalog"]
        }
        for blocking_contract in remediation["blocking_contracts"]:
            assert blocking_contract["safe_fallback_action"] == "drop"
            assert blocking_contract["same_name_grounded_evidence_ids"]
            assert set(blocking_contract["same_name_grounded_evidence_ids"]).issubset(
                available_evidence_ids
            )
        assert context["candidate_workspace"]["workspace_accepted"] is False
        assert context["candidate_workspace"]["same_gate_remains_sealed"] is True
        assert "Rotor grade present" not in serialized_context
        assert all(note not in serialized_context for note in (*exact_spent_texts, *sealed_texts))
    body = json.loads(Path(audit["round_audits"][0]["path"]).read_text())["body"]
    guard = body["candidate_post_extraction_quality_guard"]
    grounding = next(
        row for row in guard["diagnostics"] if row["kind"] == "extraction_text_grounding"
    )
    assert grounding["hard_failures"] == []
    assert grounding["passed"] is True
    assert "temporal_correctness" not in grounding
    assert grounding["source_text_temporal_policy"]["temporal_boundary_enforced"] is False
    assert guard["raw_note_text_persisted"] is False


def test_blocking_review_contract_with_no_grounded_source_exposes_drop_fallback(tmp_path):
    spec = {
        "name": "rotor_grade",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Rotor grade documented before treatment.",
    }
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=_SelectiveReviewExtractor([]),
        review_agent=_TemporalSafetyRepairAgent(first_action="drop"),
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _unsafe_rotor_and_safe_baseline_review_rows()
    schedule, spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
    spent_label_free = (
        label_free.set_index("_oci_row_id", drop=False).loc[list(spent_ids)].reset_index(drop=True)
    )
    fold_by_row = {
        row_id: fold_id
        for fold_id, row_ids in schedule.row_ids_by_fold.items()
        for row_id in row_ids
    }
    spent = runner._observable_review_rows(
        row_ids=spent_ids,
        extracted=spent_extracted,
        data=data,
        fold_by_row=fold_by_row,
    )
    context = runner._build_sanitized_review_context(
        review_round=1,
        review_attempt=1,
        spent=spent,
        spent_texts=tuple(spent_label_free["text"].astype(str)),
        specs=[spec],
        evidence_catalog=[
            {
                "source_families": ["tfidf_topics"],
                "role_hint": "confounder",
                "content": {"concept": "unrelated inlet valve"},
            }
        ],
        spent_evidence_audit={
            "provider_identity_sha256": "a" * 64,
            "review_round": 1,
            "consumer_review_round": 1,
            "spent_evidence_context_epoch": 0,
            "provider_review_round_argument": 0,
            "consumed_gate_count_before_context_fit": 0,
            "context_epoch_policy_version": (
                fusion_runner_module.SPENT_EVIDENCE_CONTEXT_EPOCH_POLICY_VERSION
            ),
            "spent_row_count": len(spent_ids),
            "sealed_row_count": len(schedule.row_ids(schedule.gate_fold_ids)),
            "source_kinds": ["tfidf_topics"],
        },
        accepted_round_baseline_specs=[spec],
        workspace_extraction_sha256=runner._extraction_projection_sha256(
            spent_extracted,
            [spec],
        ),
    )

    remediation = context["required_safety_remediation"]
    assert remediation["blocking_contract_count"] == 1
    blocking = remediation["blocking_contracts"][0]
    assert blocking["feature_name"] == "rotor_grade"
    assert blocking["hard_failures"] == ["alternative_category_only_value_support"]
    assert blocking["same_name_grounded_evidence_ids"] == []
    assert blocking["safe_fallback_action"] == "drop"


def test_unqualified_timepoint_grounding_is_accepted_and_candidate_reaches_gate(
    tmp_path, monkeypatch
):
    events = []
    initial_spec = {
        "name": "rotor_grade",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Rotor grade documented before treatment.",
    }
    revised_spec = {
        **initial_spec,
        "description": "Revised Rotor grade documented before treatment.",
    }
    response = {
        "schema_version": "all_evidence_post_extraction_review_response_v1",
        "operations": [
            {
                "action": "revise",
                "target_names": ["rotor_grade"],
                "contract": revised_spec,
                "supporting_diagnostic_ids": ["diagnostic_0001"],
                "supporting_evidence_ids": ["evidence_0001"],
                "reason": "Clarify the pretreatment extraction contract.",
            }
        ],
    }

    def accept(*args, **kwargs):
        events.append("gate_acceptance")
        return GateAcceptanceDecision(
            accepted=True,
            reasons=(),
            current={},
            candidate={},
            guards={},
            decision_sha256="d" * 64,
        )

    monkeypatch.setattr(fusion_runner_module, "evaluate_untouched_gate_acceptance", accept)
    agent = _PostExtractionReviewAgent(response, events)
    extractor = _SelectiveReviewExtractor(events)
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _rotor_grounding_review_rows(include_treatment_boundary=False)
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    spent_ids = schedule.row_ids(schedule.initial_spent_fold_ids)
    gate_ids = schedule.row_ids(schedule.gate_fold_ids)
    specs, _final_extracted, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=1,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[initial_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_001",
        review_schedule=schedule,
    )

    assert specs == [revised_spec]
    assert events == [
        "proposal",
        "selective_extraction",
        "selective_extraction",
        "selective_extraction",
        "gate_acceptance",
    ]
    assert extractor.row_id_calls == [spent_ids, gate_ids, gate_ids]
    assert audit["candidate_quality_rejection_count"] == 0
    assert audit["gate_evaluated_proposal_count"] == 1
    assert audit["consumed_gate_count"] == 1
    body = json.loads(Path(audit["round_audits"][0]["path"]).read_text())["body"]
    guard = body["candidate_post_extraction_quality_guard"]
    grounding = next(
        row for row in guard["diagnostics"] if row["kind"] == "extraction_text_grounding"
    )
    assert guard["passed"] is True
    assert grounding["passed"] is True
    assert grounding["hard_failures"] == []
    assert "temporal_correctness" not in grounding
    assert grounding["source_text_temporal_policy"]["temporal_boundary_enforced"] is False
    assert not any("timepoint" in warning for warning in grounding["warnings"])
    serialized_context = json.dumps(agent.contexts[0])
    serialized_guard = json.dumps(guard)
    assert "Rotor grade present" not in serialized_context
    assert "Rotor grade present" not in serialized_guard
    assert all(note not in serialized_context for note in label_free["text"])
    assert all(note not in serialized_guard for note in label_free["text"])


def test_consumed_gate_feedback_is_sanitized_and_visible_next_round(tmp_path, monkeypatch):
    initial_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Status documented at baseline before treatment.",
    }
    rerolled = {**initial_spec, "roles": ["effect_modifier"]}

    class FeedbackAgent:
        def __init__(self):
            self.contexts = []

        def propose(self, context):
            self.contexts.append(copy.deepcopy(context))
            if len(self.contexts) == 1:
                return {
                    "schema_version": "all_evidence_post_extraction_review_response_v1",
                    "operations": [
                        {
                            "action": "re_role",
                            "target_names": ["inlet_valve_status"],
                            "contract": rerolled,
                            "supporting_diagnostic_ids": ["diagnostic_0001"],
                            "supporting_evidence_ids": [],
                            "reason": "Test one bounded role revision.",
                        }
                    ],
                }
            feedback = [
                row for row in context["diagnostics"] if row.get("kind") == "prior_gate_feedback"
            ]
            assert len(feedback) == 1
            return {
                "schema_version": "all_evidence_post_extraction_review_response_v1",
                "operations": [
                    {
                        "action": "stop",
                        "target_names": [],
                        "contract": None,
                        "supporting_diagnostic_ids": [],
                        "supporting_evidence_ids": [],
                        "reason": "Stop after inspecting prior observable gate feedback.",
                    }
                ],
            }

    source_name = "secret_direct_signal_name"
    feature_name = "secret_neural_query_name"
    source_kind = "nested_calibrated_bow_weighted_r"

    def side(*, ratio, weighted_r, score, contracts, columns, correlation, context_ratio, role):
        return {
            "metrics": {
                "effect": {
                    "r_loss_ratio": ratio,
                    "weighted_r_loss": weighted_r,
                }
            },
            "complexity": {
                "contract_count": contracts,
                "encoded_column_count": columns,
            },
            "penalized_relative_r_loss_score": score,
            "source_signal_evaluation": {
                "sources": [
                    {
                        "source_name": source_name,
                        "source_kind": source_kind,
                        "tau_correlation": correlation,
                        "contextual_r_loss_ratio": context_ratio,
                    }
                ]
            },
            "feature_bank_evaluation": {
                "features": [
                    {
                        "feature_name": feature_name,
                        "source_kind": "neural_query_effect_moments",
                        "consumer_role": UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                        "role_matched_prediction_correlation": role,
                    }
                ],
                "preservation_score_by_consumer_role": {
                    UNCALIBRATED_EFFECT_MODIFIER_ROLE: role,
                },
                "preservation_by_source_kind_and_consumer_role": [
                    {
                        "source_kind": "neural_query_effect_moments",
                        "consumer_role": UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                        "feature_count": 1,
                        "finite_correlation_count": 1,
                        "mean_absolute_role_matched_prediction_correlation": role,
                        "aggregate_absolute_role_matched_prediction_correlation": role,
                        "aggregate_absolute_correlation_share": 1.0,
                        "leave_family_out_feature_mean_absolute_correlation": None,
                        "feature_mean_absolute_correlation_delta_when_family_removed": None,
                    }
                ],
            },
        }

    decision = GateAcceptanceDecision(
        accepted=False,
        reasons=("penalized_relative_r_loss_not_improved", "source_direction_guard_failed"),
        current=side(
            ratio=0.8,
            weighted_r=0.4,
            score=0.81,
            contracts=1,
            columns=3,
            correlation=0.7,
            context_ratio=0.9,
            role=0.6,
        ),
        candidate=side(
            ratio=1.0,
            weighted_r=0.5,
            score=1.01,
            contracts=1,
            columns=2,
            correlation=0.3,
            context_ratio=1.1,
            role=0.2,
        ),
        guards={
            "penalized_relative_r_loss": {"passed": False},
            "source_direction_preservation": {
                "passed": False,
                "by_source": {
                    f"{source_kind}::{source_name}": {
                        "passed": False,
                        "same_direction": True,
                    }
                },
            },
            "feature_bank_preservation": {
                "passed": False,
                "by_consumer_role": {
                    UNCALIBRATED_EFFECT_MODIFIER_ROLE: {"passed": False},
                },
                "by_source_kind_and_consumer_role": [
                    {
                        "source_kind": "neural_query_effect_moments",
                        "consumer_role": UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                        "passed": False,
                        "feature_count_matches": True,
                        "minimum_candidate_score": 0.55,
                    }
                ],
            },
        },
        decision_sha256="d" * 64,
    )
    monkeypatch.setattr(
        fusion_runner_module,
        "evaluate_untouched_gate_acceptance",
        lambda *args, **kwargs: decision,
    )
    agent = FeedbackAgent()
    extractor = _SelectiveReviewExtractor([])
    spent_provider = _SpentEvidenceProvider()
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=extractor,
        review_agent=agent,
        review_spent_evidence_provider=spent_provider,
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=2,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _review_loop_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
        outer_fold=2,
    )
    _initial_request, initial_selector_evidence_audit = runner._spent_fusion_request(
        data=data,
        schedule=schedule,
        spent_fold_ids=schedule.initial_spent_fold_ids,
        outer_fold=2,
        review_round=0,
    )
    first_gate_ids = schedule.row_ids((schedule.gate_fold_ids[0],))
    second_gate_ids = schedule.row_ids((schedule.gate_fold_ids[1],))
    specs, final_extracted, audit = runner._run_post_extraction_review(
        data=data,
        label_free=label_free,
        outer_fold=2,
        train_ids=tuple(data["_oci_row_id"]),
        initial_specs=[initial_spec],
        initial_extracted=initial_spent_extracted,
        fold_dir=tmp_path / "output" / "outer_fold_002",
        review_schedule=schedule,
        initial_selector_evidence_audit=initial_selector_evidence_audit,
    )

    assert specs == [initial_spec]
    assert final_extracted.equals(extracted)
    assert extractor.row_id_calls == [first_gate_ids, second_gate_ids]
    assert len(agent.contexts) == 2
    assert [context["review_round"] for context in agent.contexts] == [1, 2]
    assert [call["review_round"] for call in spent_provider.calls] == [0, 0, 1]
    assert [call["outer_fold"] for call in spent_provider.calls] == [2, 2, 2]
    assert spent_provider.calls[0]["spent"] == spent_provider.calls[1]["spent"]
    assert spent_provider.calls[0]["sealed"] == spent_provider.calls[1]["sealed"]
    assert spent_provider.calls[2]["spent"] == schedule.row_ids(
        (*schedule.initial_spent_fold_ids, schedule.gate_fold_ids[0])
    )
    assert spent_provider.calls[2]["sealed"] == second_gate_ids
    assert initial_selector_evidence_audit["consumer_review_round"] == 0
    assert initial_selector_evidence_audit["spent_evidence_context_epoch"] == 0
    assert [
        context["spent_evidence_provenance"]["spent_evidence_context_epoch"]
        for context in agent.contexts
    ] == [0, 1]
    assert [
        context["spent_evidence_provenance"]["consumer_review_round"]
        for context in agent.contexts
    ] == [1, 2]
    feedback_rows = [
        row for row in agent.contexts[1]["diagnostics"] if row.get("kind") == "prior_gate_feedback"
    ]
    assert len(feedback_rows) == 1
    feedback = feedback_rows[0]
    assert feedback["proposal_status"] == "rejected"
    assert feedback["objective"]["r_loss_ratio_delta"] == pytest.approx(0.2)
    assert feedback["objective"]["penalized_score_delta"] == pytest.approx(0.2)
    assert feedback["complexity"]["encoded_column_count"]["candidate_minus_current"] == -1
    assert feedback["opaque_calibrated_source_preservation"][0]["source_id"] == ("gate_source_0001")
    assert feedback["feature_bank_preservation_by_consumer_role"][0]["guard_passed"] is False
    family_feedback = feedback["feature_bank_preservation_by_source_kind_and_consumer_role"][0]
    assert family_feedback["source_kind"] == "neural_query_effect_moments"
    assert family_feedback["consumer_role"] == UNCALIBRATED_EFFECT_MODIFIER_ROLE
    assert family_feedback["guard_passed"] is False
    assert family_feedback["feature_count_matches"] is True
    assert family_feedback["metrics"]["mean_absolute_role_matched_prediction_correlation"][
        "candidate_minus_current"
    ] == pytest.approx(-0.4)
    assert "Do not repeat" in feedback["non_repeat_guidance"]
    serialized_context = json.dumps(agent.contexts[1])
    assert source_name not in serialized_context
    assert feature_name not in serialized_context
    assert "gate_source_0001" in serialized_context
    first_audit_text = Path(audit["round_audits"][0]["path"]).read_text()
    assert source_name not in first_audit_text
    assert feature_name not in first_audit_text
    assert audit["consumed_gate_count"] == 1
    assert audit["prior_gate_feedback_diagnostic_count"] == 1
    assert [row["status"] for row in audit["round_audits"]] == ["rejected", "agent_stop"]
    assert audit["spent_evidence_context_epoch_policy"]["policy_version"] == (
        fusion_runner_module.SPENT_EVIDENCE_CONTEXT_EPOCH_POLICY_VERSION
    )
    round_bodies = [
        json.loads(Path(row["path"]).read_text())["body"] for row in audit["round_audits"]
    ]
    assert [body["review_round"] for body in round_bodies] == [1, 2]
    assert [
        body["spent_evidence_context_audit"]["spent_evidence_context_epoch"]
        for body in round_bodies
    ] == [0, 1]


def test_operationally_invalid_response_is_not_cached(tmp_path):
    events = []
    initial_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Status documented at baseline before treatment.",
    }
    response = {
        "schema_version": "all_evidence_post_extraction_review_response_v1",
        "operations": [
            {
                "action": "drop",
                "target_names": ["inlet_valve_status"],
                "contract": None,
                "supporting_diagnostic_ids": ["diagnostic_0001"],
                "supporting_evidence_ids": [],
                "reason": "Remove the only contract.",
            }
        ],
    }
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=_Extractor(),
        review_agent=_PostExtractionReviewAgent(response, events),
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=1,
            post_extraction_review_max_quality_retries=0,
            post_extraction_review_min_partition_rows=8,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    data, label_free, extracted = _review_loop_rows()
    schedule, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
    )
    with pytest.raises(RuntimeError, match="exhausted bounded response validation"):
        runner._run_post_extraction_review(
            data=data,
            label_free=label_free,
            outer_fold=1,
            train_ids=tuple(data["_oci_row_id"]),
            initial_specs=[initial_spec],
            initial_extracted=initial_spent_extracted,
            fold_dir=tmp_path / "output" / "outer_fold_001",
            review_schedule=schedule,
        )
    cache = (
        tmp_path
        / "output"
        / "outer_fold_001"
        / "post_extraction_review"
        / "round_001"
        / "attempt_001"
        / "immutable_review_response.json"
    )
    assert not cache.exists()
    attempt_dir = cache.parent
    failure_payload = json.loads((attempt_dir / "immutable_review_failure.json").read_text())
    assert failure_payload["body"]["failure_type"] == "runner_boundary_validation"
    assert failure_payload["body"]["failure_code"] == "runner_boundary_response_invalid"
    round_payload = json.loads((attempt_dir / "immutable_review_round.json").read_text())
    assert round_payload["body"]["status"] == "review_response_validation_retry_exhausted"
    assert round_payload["body"]["gate_accessed"] is False
    assert round_payload["body"]["gate_consumed"] is False


def test_feature_bank_lineage_cannot_include_a_later_unspent_gate(tmp_path):
    events = []
    initial_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Status documented at baseline before treatment.",
    }
    revised_spec = {
        **initial_spec,
        "description": "Revised status documented at baseline before treatment.",
    }
    response = {
        "schema_version": "all_evidence_post_extraction_review_response_v1",
        "operations": [
            {
                "action": "revise",
                "target_names": ["inlet_valve_status"],
                "contract": revised_spec,
                "supporting_diagnostic_ids": ["diagnostic_0001"],
                "supporting_evidence_ids": ["evidence_0001"],
                "reason": "Clarify the extraction contract.",
            }
        ],
    }
    data, label_free, extracted = _review_loop_rows()
    schedule = _build_review_partition_schedule(
        data,
        outer_fold=1,
        review_rounds=2,
        minimum_partition_rows=8,
        random_state=42,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
    )
    future_row = schedule.row_ids((schedule.gate_fold_ids[1],))[0]
    cache_path = (
        tmp_path
        / "output"
        / "outer_fold_001"
        / "post_extraction_review"
        / "round_001"
        / "attempt_001"
        / "immutable_review_response.json"
    )
    runner = AllEvidenceFusionRunner(
        dataset_path=tmp_path / "dataset.parquet",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=_SelectiveReviewExtractor(events),
        review_agent=_PostExtractionReviewAgent(response, events),
        review_spent_evidence_provider=_SpentEvidenceProvider(),
        review_gate_feature_bank_provider=_FutureFitReviewFeatureBank(
            events,
            future_row_id=future_row,
            response_cache_path=cache_path,
        ),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=2,
            post_extraction_review_min_partition_rows=8,
            require_review_feature_banks=True,
            allow_degraded_review_without_all_upstream=True,
        ),
    )
    _, initial_spent_extracted = _initial_spent_review_inputs(
        runner,
        data,
        extracted,
        schedule=schedule,
    )
    with pytest.raises(ValueError, match="unspent future partition"):
        runner._run_post_extraction_review(
            data=data,
            label_free=label_free,
            outer_fold=1,
            train_ids=tuple(data["_oci_row_id"]),
            initial_specs=[initial_spec],
            initial_extracted=initial_spent_extracted,
            fold_dir=tmp_path / "output" / "outer_fold_001",
            review_schedule=schedule,
        )
    assert cache_path.exists()
    assert events == [
        "proposal",
        "selective_extraction",
        "selective_extraction",
        "selective_extraction",
        "feature_bank_bind",
    ]


def _write_query_artifact(
    path: Path,
    *,
    fold: int,
    fit_ids: list[int],
    heldout_ids: list[int],
    wrapped: bool,
) -> QueryEvidenceArtifact:
    row_id = fit_ids[0]
    evidence = [
        {
            "query_id": f"effect_query_{fold:03d}",
            "bank": "effect",
            "mechanical_role": "effect_modifier",
            "statistical_gate_applied": False,
            "member_count": 3,
            "member_subfolds": [1, 2],
            "fit_standardized_score": 1.2,
            "top_chunks": [
                {
                    "evidence_id": (f"effect_query_{fold:03d}__row_{row_id:05d}__chunk_000"),
                    "_oci_row_id": row_id,
                    "chunk_index": 0,
                    "similarity": 0.8,
                    "text": "baseline amber lattice",
                }
            ],
            "top_contrastive_ngrams": [{"term": "amber lattice", "tfidf_contrast": 0.2}],
        }
    ]
    body = (
        {
            "source_kind": "neural_query_moments",
            "source_family": "neural_query_moments",
            "outer_fold": fold,
            "scope": "outer_train",
            "fit_row_ids": fit_ids,
            "heldout_row_ids": heldout_ids,
            "fit_row_fingerprint": row_set_fingerprint(fit_ids),
            "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
            "query_evidence": evidence,
        }
        if wrapped
        else evidence
    )
    path.write_text(json.dumps(body), encoding="utf-8")
    return QueryEvidenceArtifact(
        path=path,
        outer_fold=fold,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        fit_row_fingerprint=row_set_fingerprint(fit_ids),
        heldout_row_fingerprint=row_set_fingerprint(heldout_ids),
    )


def test_direct_runner_required_neural_queries_rejects_bare_artifact(tmp_path):
    _, dataset_path = _dataset(tmp_path)
    legacy_path, tfidf_path = _write_handoffs(tmp_path)
    artifacts = {
        1: _write_query_artifact(
            tmp_path / "fold_1_bare_query_evidence.json",
            fold=1,
            fit_ids=list(range(6, 12)),
            heldout_ids=list(range(6)),
            wrapped=False,
        ),
        2: _write_query_artifact(
            tmp_path / "fold_2_scoped_query_evidence.json",
            fold=2,
            fit_ids=list(range(6)),
            heldout_ids=list(range(6, 12)),
            wrapped=True,
        ),
    }
    agent = _FusionAgent()
    runner = AllEvidenceFusionRunner(
        dataset_path=dataset_path,
        legacy_handoff_path=legacy_path,
        tfidf_handoff_path=tfidf_path,
        output_dir=tmp_path / "required_neural_output",
        fusion_agent=agent,
        extraction_provider=_Extractor(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            interaction_inner_folds=2,
            regularization_grid=(0.1, 1.0),
            require_neural_query_moments=True,
            neural_query_moment_artifacts_by_fold=artifacts,
        ),
    )

    with pytest.raises(ValueError, match="declare exact fit and heldout row IDs"):
        runner.run()
    assert agent.calls == 0


def test_dataset_loader_projects_exact_safe_columns_at_read_time(tmp_path, monkeypatch):
    _, path = _dataset(tmp_path)
    original = pd.read_parquet
    calls = []

    def spy(requested, *args, **kwargs):
        calls.append(kwargs.get("columns"))
        return original(requested, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    sanitized = load_sanitized_dataset(
        path,
        text_column="text",
        treatment_column="treatment",
        outcome_column="outcome",
    )
    assert calls == [["text", "treatment", "outcome"]]
    assert sanitized.columns.tolist() == ["_oci_row_id", "text", "treatment", "outcome"]


def test_dataset_loader_projects_the_same_whole_file_snapshot_it_hashes(
    tmp_path,
    monkeypatch,
):
    source, path = _dataset(tmp_path)
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    replacement_path = tmp_path / "replacement_dataset.parquet"
    replacement = source.copy()
    replacement["text"] = "replacement text"
    replacement.to_parquet(replacement_path, index=False)
    replacement_bytes = replacement_path.read_bytes()
    original_read_parquet = pd.read_parquet
    parsed_sources = []

    def replace_path_before_parse(source, *args, **kwargs):
        parsed_sources.append(source)
        path.write_bytes(replacement_bytes)
        return original_read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(fusion_runner_module.pd, "read_parquet", replace_path_before_parse)
    sanitized, artifact_sha256 = fusion_runner_module._load_sanitized_dataset_snapshot(
        path,
        text_column="text",
        treatment_column="treatment",
        outcome_column="outcome",
    )

    assert len(parsed_sources) == 1
    assert isinstance(parsed_sources[0], io.BytesIO)
    assert artifact_sha256 == expected_sha256
    assert sanitized["text"].tolist() == source["text"].tolist()
    assert hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256


def test_primary_prediction_split_loader_reads_only_id_and_fold_columns(tmp_path, monkeypatch):
    path = tmp_path / "primary_predictions.parquet"
    pd.DataFrame(
        {
            "_oci_row_id": range(4),
            "outer_fold": [1, 2, 1, 2],
            "cv_fold": [1, 2, 1, 2],
            "true_ite_prob": [0.1, 0.2, 0.3, 0.4],
            "hidden_prompt": ["forbidden"] * 4,
        }
    ).to_parquet(path, index=False)
    original = pd.read_parquet
    calls = []

    def spy(requested, *args, **kwargs):
        calls.append(kwargs.get("columns"))
        return original(requested, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    splits = load_outer_splits_from_primary_predictions(path, dataset_row_count=4)
    assert calls == [["_oci_row_id", "outer_fold", "cv_fold"]]
    assert splits == {1: (0, 2), 2: (1, 3)}


def test_primary_split_loader_parses_the_same_whole_file_snapshot_it_hashes(
    tmp_path,
    monkeypatch,
):
    import pyarrow.parquet as pq

    path = tmp_path / "primary_predictions.parquet"
    pd.DataFrame(
        {
            "_oci_row_id": range(4),
            "outer_fold": [1, 2, 1, 2],
            "cv_fold": [1, 2, 1, 2],
            "true_ite_prob": [0.1, 0.2, 0.3, 0.4],
        }
    ).to_parquet(path, index=False)
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    replacement_path = tmp_path / "replacement_primary_predictions.parquet"
    pd.DataFrame(
        {
            "_oci_row_id": range(4),
            "outer_fold": [2, 1, 2, 1],
            "cv_fold": [2, 1, 2, 1],
        }
    ).to_parquet(replacement_path, index=False)
    original_parquet_file = pq.ParquetFile
    original_read_parquet = pd.read_parquet
    inspected_sources = []
    parsed_sources = []

    def replace_path_before_schema_inspection(source, *args, **kwargs):
        inspected_sources.append(source)
        replacement_path.replace(path)
        return original_parquet_file(source, *args, **kwargs)

    def record_projection_source(source, *args, **kwargs):
        parsed_sources.append(source)
        return original_read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", replace_path_before_schema_inspection)
    monkeypatch.setattr(fusion_runner_module.pd, "read_parquet", record_projection_source)
    splits, artifact_sha256 = (
        fusion_runner_module._load_outer_splits_from_primary_predictions_snapshot(
            path,
            dataset_row_count=4,
        )
    )

    assert isinstance(inspected_sources[0], io.BytesIO)
    assert isinstance(parsed_sources[0], io.BytesIO)
    assert splits == {1: (0, 2), 2: (1, 3)}
    assert artifact_sha256 == expected_sha256
    assert hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256


def test_legacy_handoff_hash_identifies_the_exact_snapshot_streamed(tmp_path, monkeypatch):
    legacy_path, _ = _write_handoffs(tmp_path)
    expected_sha256 = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    replacement_path = tmp_path / "replacement_legacy.jsonl"
    replacement_path.write_text("{}\n", encoding="utf-8")
    original_iterator = fusion_runner_module._iter_allowlisted_legacy_records
    parsed_snapshots = []

    def replace_path_before_parse(snapshot):
        parsed_snapshots.append(snapshot)
        replacement_path.replace(legacy_path)
        yield from original_iterator(snapshot)

    monkeypatch.setattr(
        fusion_runner_module,
        "_iter_allowlisted_legacy_records",
        replace_path_before_parse,
    )
    loaded = load_legacy_full_outer_evidence(legacy_path)

    assert isinstance(parsed_snapshots[0], bytes)
    assert loaded.artifact_sha256 == expected_sha256
    assert set(loaded.rows_by_outer_fold) == {1, 2}
    assert hashlib.sha256(legacy_path.read_bytes()).hexdigest() != expected_sha256


def test_tfidf_handoff_hash_identifies_the_exact_snapshot_parsed(tmp_path, monkeypatch):
    _, tfidf_path = _write_handoffs(tmp_path)
    expected_sha256 = hashlib.sha256(tfidf_path.read_bytes()).hexdigest()
    replacement_path = tmp_path / "replacement_tfidf.jsonl"
    replacement_path.write_text("{}\n", encoding="utf-8")
    original_reader = fusion_runner_module._read_jsonl_snapshot
    parsed_snapshots = []

    def replace_path_before_parse(snapshot, **kwargs):
        parsed_snapshots.append(snapshot)
        replacement_path.replace(tfidf_path)
        return original_reader(snapshot, **kwargs)

    monkeypatch.setattr(
        fusion_runner_module,
        "_read_jsonl_snapshot",
        replace_path_before_parse,
    )
    loaded = fusion_runner_module.load_resealed_tfidf_handoff(
        tfidf_path,
        dataset_row_count=12,
    )

    assert isinstance(parsed_snapshots[0], bytes)
    assert loaded.artifact_sha256 == expected_sha256
    assert set(loaded.full_rows_by_outer_fold) == {1, 2}
    assert hashlib.sha256(tfidf_path.read_bytes()).hexdigest() != expected_sha256


def test_candidate_pool_hash_identifies_the_exact_snapshot_parsed(tmp_path, monkeypatch):
    path = tmp_path / "candidate_pool.json"
    _write_candidate_pool(path, outer_fold=1)
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    replacement_path = tmp_path / "replacement_candidate_pool.json"
    replacement_path.write_text("{}", encoding="utf-8")
    original_loads = json.loads
    parsed_values = []

    def replace_path_before_parse(value, *args, **kwargs):
        if not parsed_values:
            replacement_path.replace(path)
        parsed_values.append(value)
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(fusion_runner_module.json, "loads", replace_path_before_parse)
    contracts, audit = load_candidate_pool(path, expected_outer_fold=1)

    assert isinstance(parsed_values[0], str)
    assert [contract.extraction_spec["name"] for contract in contracts] == [
        "inlet_valve_status"
    ]
    assert audit["sha256"] == expected_sha256
    assert hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256


def test_posthoc_evaluator_parses_the_same_immutable_bytes_it_authenticates(
    tmp_path,
    monkeypatch,
):
    prediction_path = tmp_path / "frozen_predictions.parquet"
    pd.DataFrame(
        {
            "_oci_row_id": [0, 1],
            "outer_fold": [1, 1],
            "pred_ite_prob": [0.1, 0.9],
        }
    ).to_parquet(prediction_path, index=False)
    expected_sha = hashlib.sha256(prediction_path.read_bytes()).hexdigest()

    replacement_path = tmp_path / "replacement_predictions.parquet"
    pd.DataFrame(
        {
            "_oci_row_id": [0, 1],
            "outer_fold": [1, 1],
            "pred_ite_prob": [0.9, 0.1],
        }
    ).to_parquet(replacement_path, index=False)
    replacement_bytes = replacement_path.read_bytes()
    original_read_parquet = pd.read_parquet
    parsed_sources = []

    def replace_path_before_parse(source, *args, **kwargs):
        parsed_sources.append(source)
        prediction_path.write_bytes(replacement_bytes)
        return original_read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(fusion_runner_module.pd, "read_parquet", replace_path_before_parse)
    metrics = evaluate_frozen_all_evidence_predictions(
        prediction_path=prediction_path,
        expected_prediction_sha256=expected_sha,
        oracle_frame=pd.DataFrame({"_oci_row_id": [0, 1], "true_ite_prob": [0.1, 0.9]}),
        output_dir=tmp_path / "posthoc",
        oracle_ite_column="true_ite_prob",
    )

    assert len(parsed_sources) == 1
    assert isinstance(parsed_sources[0], io.BytesIO)
    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() != expected_sha
    assert metrics["overall"]["pearson_correlation"] == pytest.approx(1.0)
    assert metrics["overall"]["mae"] == pytest.approx(0.0)


def test_runner_freezes_oracle_free_predictions_and_train_only_encoder_state(tmp_path):
    source, dataset_path = _dataset(tmp_path)
    legacy_path, tfidf_path = _write_handoffs(tmp_path)
    pools = {}
    for fold in (1, 2):
        path = tmp_path / f"pool_{fold}.json"
        _write_candidate_pool(path, fold)
        pools[fold] = path

    overlay_spec = {
        "name": "inlet_valve_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Status documented before treatment.",
        "value_aliases": None,
    }
    overlay_dataset = pd.DataFrame(
        {
            "_oci_row_id": np.arange(len(source)),
            "text": source["text"].tolist(),
        }
    )
    overlay_artifact_path = tmp_path / "historical_extraction.parquet"
    pd.DataFrame(
        {
            "__oci_cache_row_index": np.arange(len(source)),
            "explicit_feat_inlet_valve_status": np.where(
                np.arange(len(source)) < 6,
                "present",
                "absent",
            ),
            "explicit_feat_inlet_valve_status_missing": False,
        }
    ).to_parquet(overlay_artifact_path, index=False)
    overlay_index_path = tmp_path / "extraction_cache_index.json"
    overlay_index_path.write_text(
        json.dumps(
            {
                "schema_version": LEGACY_EXTRACTION_CACHE_INDEX_SCHEMA_VERSION,
                "entries": [
                    {
                        "contract": overlay_spec,
                        "contract_sha256": extraction_contract_sha256(overlay_spec),
                        "model_identity": "unspecified_remote_model",
                        "prompt_template_version": (
                            "explicit_features_v5+source_text_temporally_valid_by_design_v1"
                        ),
                        "dataset_text_fingerprint": ordered_dataset_text_fingerprint(
                            overlay_dataset
                        ),
                        "expected_row_count": len(source),
                        "artifact_path": overlay_artifact_path.name,
                        "artifact_sha256": sha256_file(overlay_artifact_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cache_overlay = FrozenExtractionCacheOverlay(
        [overlay_index_path],
        expected_row_count=len(source),
    )

    agent = _FusionAgent()
    runner = AllEvidenceFusionRunner(
        dataset_path=dataset_path,
        legacy_handoff_path=legacy_path,
        tfidf_handoff_path=tfidf_path,
        output_dir=tmp_path / "output",
        fusion_agent=agent,
        extraction_provider=_Extractor(),
        candidate_pool_paths=pools,
        cache_overlay=cache_overlay,
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            interaction_inner_folds=2,
            regularization_grid=(0.1, 1.0),
            derive_sparse_query_moments_when_missing=True,
            fusion_thinking_token_budget=4096,
        ),
    )
    result = runner.run()
    assert agent.calls == 2
    predictions = pd.read_parquet(result.prediction_path)
    assert len(predictions) == 12
    assert predictions["_oci_row_id"].is_unique
    assert not any(
        token in column.lower()
        for column in predictions.columns
        for token in ("true", "oracle", "ground_truth")
    )

    fold_one = json.loads(result.fold_manifest_paths[0].read_text())["body"]
    assert fold_one["source_text_temporal_policy"]["temporal_boundary_enforced"] is False
    present_index = fold_one["encoder"]["feature_names"].index(
        "inlet_valve_status__category__present"
    )
    # Fold 1 trains only on rows 6..11 (all absent). Heldout rows 0..5 are all
    # present, so a leaked whole-dataset normalizer would have mean 0.5.
    assert fold_one["encoder"]["train_means"][present_index] == 0.0
    assert fold_one["encoder"]["model_input_preprocessing"] == "fixed_zero_imputation_only"
    assert fold_one["encoder"]["train_summaries_used_by_model"] is False
    assert fold_one["outer_heldout_outcomes_used"] is False
    assert fold_one["extraction"]["authenticated_artifact_sha256s"] == [
        sha256_file(overlay_artifact_path)
    ]
    assert fold_one["extraction"]["authenticated_cache_hits"][0][
        "cache_index_sha256"
    ] == sha256_file(overlay_index_path)
    assert fold_one["staged_fusion_audit"] == {
        "status": "unavailable_not_exposed_by_agent",
        "persisted_with_request_bound_response_cache": True,
        "audit": None,
    }
    assert fold_one["query_evidence"]["mode"] == "deterministic_sparse_fallback"
    assert fold_one["query_evidence"]["source_kind"] == "sparse_query_moments"
    assert fold_one["query_evidence"]["source_family"] == "sparse_query_moments"
    assert fold_one["query_evidence"]["model_inference_performed"] is False
    assert (
        fold_one["tfidf_orphan_ngram_evidence"]["status"]
        == "not_available_no_effect_ngram_registration"
    )

    frozen_sha = result.prediction_sha256
    oracle = pd.DataFrame({"_oci_row_id": range(12), "true_ite_prob": source["true_ite_prob"]})
    metrics = evaluate_frozen_all_evidence_predictions(
        prediction_path=result.prediction_path,
        expected_prediction_sha256=frozen_sha,
        oracle_frame=oracle,
        output_dir=tmp_path / "posthoc",
        oracle_ite_column="true_ite_prob",
    )
    assert metrics["oracle_join_performed_posthoc"] is True
    assert result.prediction_sha256 == frozen_sha
    assert "true_ite_prob" not in pd.read_parquet(result.prediction_path).columns

    # A complete rerun authenticates the request-bound response caches and does
    # not call the injected remote selector again.
    rerun = runner.run()
    assert rerun.prediction_sha256 == result.prediction_sha256
    assert agent.calls == 2

    input_manifest = json.loads(
        (tmp_path / "output" / "immutable_input_manifest.json").read_text()
    )["body"]
    expected_epoch_policy = {
        "policy_version": (
            fusion_runner_module.SPENT_EVIDENCE_CONTEXT_EPOCH_POLICY_VERSION
        ),
        "epoch_definition": "number_of_review_gates_consumed_before_context_fit",
        "provider_review_round_argument_is_context_epoch": True,
        "consumer_review_round_is_separate": True,
        "initial_selector_context_epoch": 0,
        "first_review_reuses_initial_selector_context_epoch": True,
    }
    assert fusion_runner_module.RUNNER_SCHEMA_VERSION == (
        "all_evidence_fusion_outer_runner_v20"
    )
    assert input_manifest["runner_schema_version"] == fusion_runner_module.RUNNER_SCHEMA_VERSION
    assert input_manifest["spent_evidence_context_epoch_policy"] == expected_epoch_policy
    assert fold_one["spent_evidence_context_epoch_policy"] == expected_epoch_policy
    assert input_manifest["source_text_temporal_policy"]["temporal_boundary_enforced"] is False
    run_manifest = json.loads(result.run_manifest_path.read_text())["body"]
    assert run_manifest["source_text_temporal_policy"]["temporal_boundary_enforced"] is False
    assert run_manifest["spent_evidence_context_epoch_policy"] == expected_epoch_policy
    assert input_manifest["effective_runner_config"]["random_state"] == 42
    assert input_manifest["effective_runner_config"]["extraction_model_identity"] == (
        "unspecified_remote_model"
    )
    assert input_manifest["effective_runner_config"]["fusion_model_identity"] == (
        "unspecified_remote_model"
    )
    assert input_manifest["effective_runner_config"]["fusion_enable_thinking"] is True
    assert input_manifest["effective_runner_config"]["fusion_max_tokens"] == 25000
    assert input_manifest["effective_runner_config"]["fusion_thinking_token_budget"] == 4096
    assert input_manifest["effective_runner_config"]["extraction_enable_thinking"] is False
    assert input_manifest["effective_runner_config"]["remote_endpoint_pool_identity"] == (
        "unspecified_remote_endpoint_pool"
    )
    assert input_manifest["post_extraction_review_response_boundary"] == {
        "prompt_version": POST_EXTRACTION_REVIEW_PROMPT_VERSION,
        "response_schema_version": POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
        "request_schema_version": fusion_runner_module.POST_EXTRACTION_REVIEW_REQUEST_SCHEMA_VERSION,
        "response_cache_schema_version": (
            fusion_runner_module.POST_EXTRACTION_REVIEW_RESPONSE_CACHE_SCHEMA_VERSION
        ),
        "failure_cache_schema_version": (
            fusion_runner_module.POST_EXTRACTION_REVIEW_FAILURE_SCHEMA_VERSION
        ),
        "round_audit_schema_version": (
            fusion_runner_module.POST_EXTRACTION_REVIEW_ROUND_SCHEMA_VERSION
        ),
        "operation_apply_policy_version": (
            fusion_runner_module.POST_EXTRACTION_REVIEW_OPERATION_APPLY_POLICY_VERSION
        ),
        "ordered_extraction_projection_sha256_version": (
            fusion_runner_module.ORDERED_EXTRACTION_PROJECTION_SHA256_VERSION
        ),
        "fresh_response_normalization_version": (
            POST_EXTRACTION_REVIEW_FRESH_NORMALIZATION_VERSION
        ),
        "grounding_repair_version": (
            fusion_runner_module.POST_EXTRACTION_REVIEW_GROUNDING_REPAIR_VERSION
        ),
        "response_validation_retry_policy_version": (
            fusion_runner_module.POST_EXTRACTION_REVIEW_RESPONSE_VALIDATION_RETRY_POLICY_VERSION
        ),
        "cached_failure_replay_enabled": True,
        "invalid_raw_response_persisted": False,
        "invalid_raw_reasoning_persisted": False,
        "cached_response_normalization_enabled": False,
        "candidate_workspace_policy_version": (
            fusion_runner_module.POST_EXTRACTION_REVIEW_CANDIDATE_WORKSPACE_POLICY_VERSION
        ),
    }
    assert POST_EXTRACTION_REVIEW_PROMPT_VERSION == "all_evidence_post_extraction_review_v12"
    assert input_manifest["post_extraction_review_extraction_semantics"] == {
        "adaptive_review_enabled": False,
        "provider_declares_request_group_dependency": False,
        "contract_local_request_semantics_verified": True,
        "required_for_selective_review": True,
        "enforcement_version": (
            fusion_runner_module.ADAPTIVE_REVIEW_CONTRACT_LOCAL_EXTRACTION_VERSION
        ),
    }
    overlay_manifest = input_manifest["conditional_extraction_cache_overlay"]
    assert overlay_manifest["active"] is True
    assert overlay_manifest["overlay_identity"]["identity"] == cache_overlay.identity()
    assert overlay_manifest["overlay_identity"]["identity_sha256"] == _json_content_sha256(
        cache_overlay.identity()
    )

    # Changing only the conditional overlay must fail at the immutable input
    # boundary before a cached selector response or extractor can be used.
    empty_index_path = tmp_path / "empty_extraction_cache_index.json"
    empty_index_path.write_text(
        json.dumps(
            {
                "schema_version": LEGACY_EXTRACTION_CACHE_INDEX_SCHEMA_VERSION,
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    changed_overlay_agent = _FusionAgent()
    changed_overlay_runner = AllEvidenceFusionRunner(
        dataset_path=dataset_path,
        legacy_handoff_path=legacy_path,
        tfidf_handoff_path=tfidf_path,
        output_dir=tmp_path / "output",
        fusion_agent=changed_overlay_agent,
        extraction_provider=_Extractor(),
        candidate_pool_paths=pools,
        cache_overlay=FrozenExtractionCacheOverlay(
            [empty_index_path],
            expected_row_count=len(source),
        ),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            interaction_inner_folds=2,
            regularization_grid=(0.1, 1.0),
            derive_sparse_query_moments_when_missing=True,
            fusion_thinking_token_budget=4096,
        ),
    )
    with pytest.raises(RuntimeError, match="immutable_input_manifest.json"):
        changed_overlay_runner.run()
    assert changed_overlay_agent.calls == 0

    # A changed execution configuration must fail at the first immutable
    # boundary, before a stale response can be reused or extraction can run.
    changed_agent = _FusionAgent()
    changed_runner = AllEvidenceFusionRunner(
        dataset_path=dataset_path,
        legacy_handoff_path=legacy_path,
        tfidf_handoff_path=tfidf_path,
        output_dir=tmp_path / "output",
        fusion_agent=changed_agent,
        extraction_provider=_Extractor(),
        candidate_pool_paths=pools,
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            interaction_inner_folds=2,
            regularization_grid=(0.1, 1.0),
            derive_sparse_query_moments_when_missing=True,
            fusion_enable_thinking=False,
        ),
    )
    with pytest.raises(RuntimeError, match="immutable_input_manifest.json"):
        changed_runner.run()
    assert changed_agent.calls == 0

    changed_cap_agent = _FusionAgent()
    changed_cap_runner = AllEvidenceFusionRunner(
        dataset_path=dataset_path,
        legacy_handoff_path=legacy_path,
        tfidf_handoff_path=tfidf_path,
        output_dir=tmp_path / "output",
        fusion_agent=changed_cap_agent,
        extraction_provider=_Extractor(),
        candidate_pool_paths=pools,
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            interaction_inner_folds=2,
            regularization_grid=(0.1, 1.0),
            derive_sparse_query_moments_when_missing=True,
            fusion_max_tokens=24000,
            fusion_thinking_token_budget=4096,
        ),
    )
    with pytest.raises(RuntimeError, match="immutable_input_manifest.json"):
        changed_cap_runner.run()
    assert changed_cap_agent.calls == 0


def test_runner_persists_request_bound_staged_agent_audit(tmp_path):
    _, dataset_path = _dataset(tmp_path)
    legacy_path, tfidf_path = _write_handoffs(tmp_path)
    base_agent = _ProposalBaseAgent()
    staged_agent = StagedAllEvidenceFusionAgent(base_agent, final_max_candidates=1)
    runner = AllEvidenceFusionRunner(
        dataset_path=dataset_path,
        legacy_handoff_path=legacy_path,
        tfidf_handoff_path=tfidf_path,
        output_dir=tmp_path / "staged_output",
        fusion_agent=staged_agent,
        extraction_provider=_Extractor(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            max_candidates=1,
            interaction_inner_folds=2,
            regularization_grid=(0.1, 1.0),
        ),
    )

    result = runner.run()
    assert base_agent.calls == 8
    for manifest_path in result.fold_manifest_paths:
        body = json.loads(manifest_path.read_text())["body"]
        staged = body["staged_fusion_audit"]
        assert staged["status"] == "captured_and_request_bound"
        assert staged["persisted_with_request_bound_response_cache"] is True
        assert staged["audit"]["outer_fold"] == body["outer_fold"]
        assert staged["audit"]["original_request_sha256"] == body["fusion_request_sha256"]
        assert staged["audit"]["returned_response_sha256"] == body["fusion_response_sha256"]

        response_cache = json.loads(Path(body["fusion_response_cache_path"]).read_text())["body"]
        assert response_cache["staged_fusion_audit_status"] == ("captured_and_request_bound")
        assert response_cache["staged_fusion_audit"] == staged["audit"]

    valid_audit = response_cache["staged_fusion_audit"]
    unknown_field_paths = (
        (),
        ("role_specific_proposal_policy",),
        ("stages", 0),
        ("stages", 3),
        ("stages", 3, "selection_postprocessor"),
        ("proposal_union",),
        ("proposal_union", "same_name_merge"),
        ("proposal_union", "safe_union"),
        ("proposal_union", "safe_union", "identity"),
        ("proposal_union", "safe_union", "dispositions", 0),
        ("stages", 0, "reasoning_trace_presence"),
    )
    for nested_path in unknown_field_paths:
        malformed = copy.deepcopy(valid_audit)
        target = malformed
        for component in nested_path:
            target = target[component]
        target["details"] = "private selector reasoning"
        with pytest.raises(RuntimeError, match=r"closed .* schema.*unknown=\['details'\]"):
            _validate_closed_staged_fusion_audit(malformed)

    malformed_type = copy.deepcopy(valid_audit)
    malformed_type["stages"][0]["reasoning_trace_presence"]["completion_attempt_count"] = True
    with pytest.raises(RuntimeError, match="expected integer"):
        _validate_closed_staged_fusion_audit(malformed_type)

    # A cache-only rerun authenticates and reuses the persisted audit without
    # inventing a new staged trace or changing the immutable fold manifest.
    rerun = runner.run()
    assert rerun.prediction_sha256 == result.prediction_sha256
    assert base_agent.calls == 8

    # A content-hash-valid cache still fails closed when its staged audit has
    # any field outside the exact v3 schema. This exercises the cache path,
    # rather than only validating an agent's fresh in-memory audit.
    first_manifest = json.loads(result.fold_manifest_paths[0].read_text())["body"]
    cache_path = Path(first_manifest["fusion_response_cache_path"])
    cached = json.loads(cache_path.read_text())
    cached["body"]["staged_fusion_audit"]["details"] = "cached private selector reasoning"
    cached["content_sha256"] = _json_content_sha256(cached["body"])
    cache_path.write_text(json.dumps(cached, indent=2, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match=r"closed .* schema.*unknown=\['details'\]"):
        runner.run()


def test_runner_rejects_unknown_raw_reasoning_from_fresh_staged_audit(tmp_path):
    _, dataset_path = _dataset(tmp_path)
    legacy_path, tfidf_path = _write_handoffs(tmp_path)
    base_agent = _ProposalBaseAgent()
    agent = _LeakyStagedAuditAgent(base_agent)
    output_dir = tmp_path / "private_audit_output"
    runner = AllEvidenceFusionRunner(
        dataset_path=dataset_path,
        legacy_handoff_path=legacy_path,
        tfidf_handoff_path=tfidf_path,
        output_dir=output_dir,
        fusion_agent=agent,
        extraction_provider=_Extractor(),
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            max_candidates=1,
            interaction_inner_folds=2,
            regularization_grid=(0.1, 1.0),
        ),
    )

    with pytest.raises(RuntimeError, match=r"closed .* schema.*unknown=\['trace'\]"):
        runner.run()

    assert base_agent.calls == 4
    assert not (output_dir / "outer_fold_001" / "immutable_fusion_response.json").exists()
    persisted_json = "\n".join(
        path.read_text(encoding="utf-8") for path in output_dir.rglob("*.json")
    )
    assert agent.private_text not in persisted_json


def test_runner_adds_default_full_outer_orphan_source_and_manifest_audit(tmp_path):
    _, dataset_path = _dataset(tmp_path)
    registrations = {}
    for fold in (1, 2):
        relative = Path("tfidf_artifacts") / f"fold_{fold}_effect_ngram_scores.parquet"
        digest = _write_effect_ngram_scores(tmp_path / relative)
        registrations[fold] = {"path": str(relative), "sha256": digest}
    legacy_path, tfidf_path = _write_handoffs(
        tmp_path,
        full_effect_ngram_registrations=registrations,
    )
    pools = {}
    for fold in (1, 2):
        path = tmp_path / f"orphan_pool_{fold}.json"
        _write_candidate_pool(path, fold)
        pools[fold] = path

    agent = _FusionAgent()
    result = AllEvidenceFusionRunner(
        dataset_path=dataset_path,
        legacy_handoff_path=legacy_path,
        tfidf_handoff_path=tfidf_path,
        output_dir=tmp_path / "orphan_output",
        fusion_agent=agent,
        extraction_provider=_Extractor(),
        candidate_pool_paths=pools,
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            interaction_inner_folds=2,
            regularization_grid=(0.1, 1.0),
        ),
    ).run()

    assert agent.calls == 2
    for context in agent.request_contexts:
        assert TFIDF_ORPHAN_NGRAMS in context["source_family_coverage"]["present_source_families"]
        serialized = json.dumps(context).lower()
        assert "copper meadow" in serialized
        assert "_oci_row_id" not in serialized
        assert "oracle" not in serialized
        assert "effect_ngram_scores.parquet" not in serialized
    for manifest_path in result.fold_manifest_paths:
        audit = json.loads(manifest_path.read_text())["body"]["tfidf_orphan_ngram_evidence"]
        assert audit["scope"] == "full_outer_train"
        assert audit["artifact"]["declared_sha256_verified"] is True
        assert audit["artifact_resolution"]["mode"] == "resealed_handoff_reference"
        assert audit["heldout_scored_artifact_used"] is False
        assert audit["model_inference_performed"] is False
        assert audit["source_artifact_audit_removed_before_fusion"] is True


def test_runner_registry_repairs_nonportable_per_fold_orphan_paths(tmp_path):
    _, dataset_path = _dataset(tmp_path)
    stale_registrations = {}
    registry = {}
    for fold in (1, 2):
        relocated = tmp_path / "relocated" / f"fold_{fold}_effect_ngram_scores.parquet"
        digest = _write_effect_ngram_scores(relocated)
        stale_registrations[fold] = {
            "path": f"/retired/nonportable/fold_{fold}_effect_ngram_scores.parquet",
            "sha256": digest,
        }
        registry[fold] = TfidfOrphanNgramArtifact(
            path=relocated,
            artifact_sha256=digest,
        )
    legacy_path, tfidf_path = _write_handoffs(
        tmp_path,
        full_effect_ngram_registrations=stale_registrations,
    )
    pools = {}
    for fold in (1, 2):
        path = tmp_path / f"registry_pool_{fold}.json"
        _write_candidate_pool(path, fold)
        pools[fold] = path

    agent = _FusionAgent()
    result = AllEvidenceFusionRunner(
        dataset_path=dataset_path,
        legacy_handoff_path=legacy_path,
        tfidf_handoff_path=tfidf_path,
        output_dir=tmp_path / "registry_output",
        fusion_agent=agent,
        extraction_provider=_Extractor(),
        candidate_pool_paths=pools,
        tfidf_orphan_artifacts_by_fold=registry,
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            interaction_inner_folds=2,
            regularization_grid=(0.1, 1.0),
            require_tfidf_orphan_ngrams=True,
        ),
    ).run()

    assert agent.calls == 2
    for manifest_path in result.fold_manifest_paths:
        audit = json.loads(manifest_path.read_text())["body"]["tfidf_orphan_ngram_evidence"]
        resolution = audit["artifact_resolution"]
        assert resolution["mode"] == "explicit_per_fold_registry_override"
        assert resolution["explicit_registry_used"] is True
        assert audit["artifact"]["declared_sha256_verified"] is True
        assert audit["selected_cluster_count"] == 1


def test_exact_inner_legacy_handoff_is_loaded_full_outer_only(tmp_path):
    legacy_path, _ = _write_handoffs(tmp_path)
    loaded = load_legacy_full_outer_evidence(legacy_path)
    assert set(loaded.rows_by_outer_fold) == {1, 2}
    assert loaded.ignored_non_full_context_count == 4
    assert set(loaded.rows_by_outer_fold[1]) == {
        "outer_fold",
        "scope",
        "n_rows",
        "context",
    }


def test_legacy_digest_drops_unused_metrics_and_oracle_keys_recursively():
    cleaned, count = _sanitize_retained_legacy_digest(
        {
            "effect_modifiers": {
                "htr_blurbs": [
                    {
                        "rows": [{"text": "pretreatment pattern"}],
                        "metrics": {"delta_logit_true_ite_corr": 0.9},
                        "true_ite_rank": 1,
                    }
                ]
            }
        }
    )
    row = cleaned["effect_modifiers"]["htr_blurbs"][0]
    assert row == {"rows": [{"text": "pretreatment pattern"}]}
    assert count == 2


def test_cross_fold_candidate_pool_is_rejected(tmp_path):
    path = tmp_path / "pool.json"
    _write_candidate_pool(path, outer_fold=2)
    with pytest.raises(ValueError, match="outer fold"):
        load_candidate_pool(path, expected_outer_fold=1)


def test_oracle_content_in_legacy_evidence_is_rejected_before_agent_call(tmp_path):
    _, dataset_path = _dataset(tmp_path)
    legacy_path, tfidf_path = _write_handoffs(tmp_path, oracle_in_legacy=True)
    pools = {}
    for fold in (1, 2):
        path = tmp_path / f"pool_{fold}.json"
        _write_candidate_pool(path, fold)
        pools[fold] = path
    runner = AllEvidenceFusionRunner(
        dataset_path=dataset_path,
        legacy_handoff_path=legacy_path,
        tfidf_handoff_path=tfidf_path,
        output_dir=tmp_path / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=_Extractor(),
        candidate_pool_paths=pools,
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            interaction_inner_folds=2,
        ),
    )
    with pytest.raises(ValueError, match="forbidden oracle/true"):
        runner.run()


def _runner_final_package(tmp_path, *, mode="signal", backend=None):
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(12),
            "text": [f"baseline note {index}" for index in range(12)],
            "treatment": np.arange(12) % 2,
            "outcome": (np.arange(12) // 2) % 2,
        }
    )
    train_ids = tuple(range(6, 12))
    heldout_ids = tuple(range(6))
    train = frame.set_index("_oci_row_id", drop=False).loc[list(train_ids)]
    meta_ids, _ = _build_final_upstream_meta_inner_fold_ids(
        train,
        n_splits=3,
        random_state=42,
        outer_fold=1,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
    )
    producer = FinalContextFitUpstreamProducer(
        tmp_path / f"final_cache_{mode}",
        backend=_FinalRunnerSignalBackend(mode) if backend is None else backend,
    )
    package = producer.produce(
        outer_fold=1,
        outer_train_row_ids=train_ids,
        outer_train_texts=tuple(train["text"]),
        outer_train_treatment=train["treatment"].to_numpy(dtype=float),
        outer_train_outcome=train["outcome"].to_numpy(dtype=float),
        outer_heldout_row_ids=heldout_ids,
        outer_heldout_texts=tuple(
            frame.set_index("_oci_row_id").loc[list(heldout_ids), "text"].tolist()
        ),
        meta_inner_fold_ids=meta_ids,
    )
    return producer, package, train_ids, heldout_ids, meta_ids


def test_final_upstream_boundary_rejects_wrong_rows_and_tampering_and_routes_modifiers(
    tmp_path,
):
    producer, package, train_ids, heldout_ids, meta_ids = _runner_final_package(tmp_path)
    producer_identity_sha256 = _json_content_sha256(producer.identity())
    prepared = _prepare_final_upstream_head_inputs(
        package,
        outer_fold=1,
        expected_train_row_ids=train_ids,
        expected_heldout_row_ids=heldout_ids,
        expected_meta_inner_fold_ids=meta_ids,
        expected_producer_identity_sha256=producer_identity_sha256,
        require_neural_query_inputs=True,
    )

    assert prepared.train_values.shape == (6, 4)
    # The calibrated tau source is always a modifier; of the three raw query
    # families, only the effect-moment role is a modifier in modifier-only mode.
    assert prepared.modifier_indices == (0, 3)
    assert prepared.audit["neural_query_inputs"]["recognized_raw_kinds"] == [
        "neural_query_effect_moments",
        "neural_query_outcome_moments",
        "neural_query_treatment_moments",
    ]
    assert prepared.audit["row_level_numerical_vectors_persisted_in_runner_audit"] is False

    with pytest.raises(ValueError, match="train row identity or order changed"):
        _prepare_final_upstream_head_inputs(
            package,
            outer_fold=1,
            expected_train_row_ids=tuple(reversed(train_ids)),
            expected_heldout_row_ids=heldout_ids,
            expected_meta_inner_fold_ids=meta_ids,
            expected_producer_identity_sha256=producer_identity_sha256,
            require_neural_query_inputs=True,
        )

    matrix_path = package.manifest_path.parent / "raw_feature_train_oof.npy"
    matrix = np.load(matrix_path, allow_pickle=False)
    matrix[0, 0] += 0.25
    with matrix_path.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    with pytest.raises(ValueError, match="failed SHA-256 authentication"):
        _prepare_final_upstream_head_inputs(
            package,
            outer_fold=1,
            expected_train_row_ids=train_ids,
            expected_heldout_row_ids=heldout_ids,
            expected_meta_inner_fold_ids=meta_ids,
            expected_producer_identity_sha256=producer_identity_sha256,
            require_neural_query_inputs=True,
        )


@pytest.mark.parametrize(
    "variant",
    ["name_spoof", "missing_effect", "wrong_treatment_role"],
)
def test_required_final_neural_query_inputs_need_all_exact_raw_kind_role_pairs(
    tmp_path,
    variant,
):
    producer, package, train_ids, heldout_ids, meta_ids = _runner_final_package(
        tmp_path,
        backend=_NeuralRequirementVariantBackend(variant),
    )
    with pytest.raises(ValueError, match="absent or have the wrong consumer role"):
        _prepare_final_upstream_head_inputs(
            package,
            outer_fold=1,
            expected_train_row_ids=train_ids,
            expected_heldout_row_ids=heldout_ids,
            expected_meta_inner_fold_ids=meta_ids,
            expected_producer_identity_sha256=_json_content_sha256(producer.identity()),
            require_neural_query_inputs=True,
        )


def _write_heterogeneous_runner_dataset(path: Path) -> None:
    row_ids = np.arange(12)
    treatment = row_ids % 2
    modifier = (row_ids // 2) % 2
    outcome = (treatment == modifier).astype(int)
    pd.DataFrame(
        {
            "text": [f"baseline note {index}" for index in row_ids],
            "treatment": treatment,
            "outcome": outcome,
        }
    ).to_parquet(path, index=False)


def _run_with_final_upstream_mode(tmp_path, *, mode):
    run_dir = tmp_path / mode
    run_dir.mkdir()
    dataset_path = run_dir / "dataset.parquet"
    _write_heterogeneous_runner_dataset(dataset_path)
    legacy_path, tfidf_path = _write_handoffs(run_dir)
    pools = {}
    for fold in (1, 2):
        path = run_dir / f"pool_{fold}.json"
        _write_candidate_pool(path, fold)
        pools[fold] = path
    delegate = FinalContextFitUpstreamProducer(
        run_dir / "final_upstream_cache",
        backend=_FinalRunnerSignalBackend(mode),
    )
    producer = _RecordingFinalUpstreamProducer(delegate)
    result = AllEvidenceFusionRunner(
        dataset_path=dataset_path,
        legacy_handoff_path=legacy_path,
        tfidf_handoff_path=tfidf_path,
        output_dir=run_dir / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=_Extractor(),
        final_upstream_producer=producer,
        candidate_pool_paths=pools,
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            interaction_inner_folds=2,
            interact_all_features=False,
            regularization_grid=(0.003, 10.0),
            require_final_upstream_inputs=True,
            require_final_upstream_neural_query_inputs=True,
            final_upstream_meta_inner_folds=3,
            final_upstream_head_regularization=10.0,
        ),
    ).run()
    return producer, result


def test_runner_uses_label_free_final_upstream_api_singleton_head_and_predictions_are_sensitive(
    tmp_path,
):
    signal_producer, signal_result = _run_with_final_upstream_mode(tmp_path, mode="signal")
    _, constant_result = _run_with_final_upstream_mode(tmp_path, mode="constant")

    expected_api = {
        "outer_fold",
        "outer_train_row_ids",
        "outer_train_texts",
        "outer_train_treatment",
        "outer_train_outcome",
        "outer_heldout_row_ids",
        "outer_heldout_texts",
        "meta_inner_fold_ids",
    }
    assert len(signal_producer.calls) == 2
    assert all(set(call) == expected_api for call in signal_producer.calls)
    assert all("outer_heldout_outcome" not in call for call in signal_producer.calls)
    assert all("outer_heldout_treatment" not in call for call in signal_producer.calls)

    for manifest_path in signal_result.fold_manifest_paths:
        body = json.loads(manifest_path.read_text())["body"]
        upstream = body["final_upstream_model_inputs"]
        policy = upstream["head_regularization_policy"]
        assert policy["grid"] == [10.0]
        assert policy["singleton_grid_precommitted_in_runner_config"] is True
        assert policy["adaptive_regularization_choice_performed"] is False
        assert policy["selection_equals_precommitted_singleton"] is True
        assert upstream["outer_heldout_labels_passed_to_producer"] is False
        assert upstream["post_extraction_registry_frozen_before_production"] is True
        assert upstream["direct_upstream_numerical_signals_used_as_final_model_inputs"] is True
        assert upstream["neural_query_inputs"]["used_as_final_model_inputs"] is True
        assert upstream["raw_features"]["modifier_only_routing"]["interaction_column_count"] == 1
        assert body["head_tuning"]["selected_regularization"] == 10.0
        assert len(body["head_tuning"]["mean_validation_loss"]) == 1
        assert (
            body["observed_label_use"]["precommitted_singleton_final_head_regularization"] is True
        )

    signal = pd.read_parquet(signal_result.prediction_path).sort_values("_oci_row_id")
    constant = pd.read_parquet(constant_result.prediction_path).sort_values("_oci_row_id")
    assert not np.allclose(signal["pred_ite_prob"], constant["pred_ite_prob"])


class _RunnerFirstEffectForestBackend:
    def __init__(self):
        self.calls = []

    def identity(self):
        return {
            "backend": "runner_first_effect_forest_test_v1",
            "honest": True,
            "inference": False,
            "tune_model": False,
        }

    def fit_predict(
        self,
        *,
        effect_train,
        control_train,
        treatment,
        outcome,
        effect_heldout,
        control_heldout,
    ):
        self.calls.append(
            {
                "effect_train_shape": tuple(np.asarray(effect_train).shape),
                "control_train_shape": tuple(np.asarray(control_train).shape),
                "treatment": np.asarray(treatment).copy(),
                "outcome": np.asarray(outcome).copy(),
                "effect_heldout": np.asarray(effect_heldout).copy(),
                "control_heldout": np.asarray(control_heldout).copy(),
            }
        )
        return np.asarray(effect_heldout, dtype=float)[:, 0]


class _FakeRunnerNuisanceDerivation:
    def __init__(self, package, nuisance):
        self.package = package
        self.nuisance = nuisance

    def verify_authenticated_content(self, package, *, runtime_producer):
        assert package is self.package
        assert type(runtime_producer) is FinalContextFitUpstreamProducer
        self.nuisance.validate_parent(package)

    def audit_record(self):
        return {
            "schema_version": "test_authenticated_nuisance_derivation_v1",
            "package_cache_key": self.package.cache_key,
            "nuisance_content_sha256": self.nuisance.content_sha256,
            "semantic_inference_from_feature_names": False,
            "row_level_values_persisted": False,
        }


def _test_exact_nuisance_derivation(package, *, runtime_producer):
    assert type(runtime_producer) is FinalContextFitUpstreamProducer
    source = package.calibrated_sources
    names = ("bow_e", "htr_e", "bow_m", "htr_m")
    train_values = np.full((len(source.train_row_ids), 4), 0.5, dtype=float)
    heldout_values = np.column_stack(
        (
            np.full(len(source.heldout_row_ids), 0.45),
            np.full(len(source.heldout_row_ids), 0.55),
            np.full(len(source.heldout_row_ids), 0.50),
            np.full(len(source.heldout_row_ids), 0.60),
        )
    )
    train_lineage = tuple(
        tuple(row[0] for _ in names) for row in source.train_oof_fit_row_provenance
    )
    heldout_lineage = tuple(
        tuple(row[0] for _ in names) for row in source.outer_heldout_fit_row_provenance
    )
    nuisance = SealedExactNuisanceBankExtension.seal_for_package(
        package,
        prediction_names=names,
        prediction_kinds=(
            "bow_nuisance",
            "htr_nuisance",
            "bow_nuisance",
            "htr_nuisance",
        ),
        prediction_semantics=(
            EXACT_PROPENSITY_PREDICTION,
            EXACT_PROPENSITY_PREDICTION,
            EXACT_OUTCOME_PREDICTION,
            EXACT_OUTCOME_PREDICTION,
        ),
        train_oof_values=train_values,
        outer_heldout_values=heldout_values,
        train_oof_fit_row_provenance=train_lineage,
        outer_heldout_fit_row_provenance=heldout_lineage,
    )
    return _FakeRunnerNuisanceDerivation(package, nuisance)


def _run_with_exact_runtime_forest(tmp_path, monkeypatch, *, mode):
    run_dir = tmp_path / f"forest_{mode}"
    run_dir.mkdir()
    dataset_path = run_dir / "dataset.parquet"
    _write_heterogeneous_runner_dataset(dataset_path)
    legacy_path, tfidf_path = _write_handoffs(run_dir)
    pools = {}
    for fold in (1, 2):
        path = run_dir / f"pool_{fold}.json"
        _write_candidate_pool(path, fold)
        pools[fold] = path
    raw_producer = FinalContextFitUpstreamProducer(
        run_dir / "final_upstream_cache",
        backend=_FinalRunnerSignalBackend(mode),
    )
    backend = _RunnerFirstEffectForestBackend()
    monkeypatch.setattr(
        fusion_runner_module,
        "derive_exact_nuisance_from_runtime_stable_stage1",
        _test_exact_nuisance_derivation,
    )
    result = AllEvidenceFusionRunner(
        dataset_path=dataset_path,
        legacy_handoff_path=legacy_path,
        tfidf_handoff_path=tfidf_path,
        output_dir=run_dir / "output",
        fusion_agent=_FusionAgent(),
        extraction_provider=_Extractor(),
        final_upstream_producer=raw_producer,
        raw_final_upstream_producer=raw_producer,
        final_causal_forest_backend=backend,
        candidate_pool_paths=pools,
        config=AllEvidenceFusionRunnerConfig(
            post_extraction_review_rounds=0,
            interaction_inner_folds=2,
            require_final_upstream_inputs=True,
            require_final_upstream_neural_query_inputs=True,
            require_final_causal_forest=True,
            final_upstream_meta_inner_folds=3,
        ),
    ).run()
    return backend, result


def test_exact_raw_runtime_uses_causal_forest_role_routing_and_never_s_head(
    tmp_path,
    monkeypatch,
):
    signal_backend, signal_result = _run_with_exact_runtime_forest(
        tmp_path,
        monkeypatch,
        mode="signal",
    )
    _, constant_result = _run_with_exact_runtime_forest(
        tmp_path,
        monkeypatch,
        mode="constant",
    )

    assert len(signal_backend.calls) == 2
    assert all(call["effect_train_shape"][1] >= 2 for call in signal_backend.calls)
    assert all(call["control_train_shape"][1] >= 6 for call in signal_backend.calls)
    for manifest_path in signal_result.fold_manifest_paths:
        body = json.loads(manifest_path.read_text())["body"]
        estimator = body["final_ite_estimator"]
        assert estimator["mode"] == FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID
        assert estimator["structured_interaction_head_constructed"] is False
        assert estimator["outer_heldout_labels_used"] is False
        assert (
            estimator["explicit_feature_role_routing"]["dual_role_columns_copied_to_both_x_and_w"]
            is True
        )
        assert estimator["potential_outcome_reconstruction"]["s_learner_fit"] is False
        assert body["head_tuning"] is None
        assert body["observed_label_use"]["complete_outer_train_final_causal_forest_fit"] is True
        prediction = pd.read_parquet(body["prediction_path"])
        np.testing.assert_allclose(
            prediction["pred_y1_prob"] - prediction["pred_y0_prob"],
            prediction["pred_ite_prob"],
        )

    signal = pd.read_parquet(signal_result.prediction_path).sort_values("_oci_row_id")
    constant = pd.read_parquet(constant_result.prediction_path).sort_values("_oci_row_id")
    assert not np.allclose(signal["pred_ite_prob"], constant["pred_ite_prob"])


def test_binary_forest_reconstruction_audits_clipped_final_estimand(tmp_path):
    _producer, package, _train_ids, heldout_ids, _meta_ids = _runner_final_package(tmp_path)
    nuisance = _test_exact_nuisance_derivation(
        package,
        runtime_producer=FinalContextFitUpstreamProducer(
            tmp_path / "unused_identity",
            backend=_FinalRunnerSignalBackend(),
        ),
    ).nuisance
    raw_tau = np.asarray([1.4, -1.3, 0.2, -0.4, 0.0, 0.8], dtype=float)
    assert len(raw_tau) == len(heldout_ids)

    p0, p1, final_tau, audit = _reconstruct_forest_potential_outcomes(
        raw_tau,
        exact_nuisance=nuisance,
        outcome_type="binary",
    )

    np.testing.assert_array_equal(final_tau, np.clip(raw_tau, -1.0, 1.0))
    np.testing.assert_allclose(p1 - p0, final_tau)
    assert audit["forest_tau_clip_count"] == 2
    assert audit["forest_tau_clipping_changed_estimand"] is True
    assert audit["final_prediction_estimand"] == "minus_one_to_one_clipped_forest_tau"
    assert audit["final_estimand_equals_unmodified_sealed_forest_tau"] is False
    assert audit["raw_sealed_forest_tau_values_sha256"] != (
        audit["final_prediction_estimand_values_sha256"]
    )
