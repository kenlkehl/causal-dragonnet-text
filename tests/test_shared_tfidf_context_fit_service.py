from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from oci.config import AppliedInferenceConfig
from oci.inference.context_fit_upstream_gate_provider import ContextFitUpstreamGateProvider
from oci.inference.final_context_fit_upstream_bank import FinalContextFitUpstreamProducer
from oci.inference.review_spent_evidence_provider import (
    TfidfTopicOrphanSpentDiscoveryBackend,
)
from oci.inference.shared_tfidf_context_fit_service import (
    InMemorySharedTfidfContextFitService,
    SharedTfidfContextBackend,
    SharedTfidfSpentDiscoveryBackend,
    build_shared_tfidf_context_fit_backends,
)
from oci.inference.tfidf_topic_discovery import FittedTopicContext
from oci.inference.tfidf_upstream_gate_backend import TfidfTopicOrphanContextBackend
import oci.inference.review_spent_evidence_provider as spent_module
import oci.inference.shared_tfidf_context_fit_service as shared_module
import oci.inference.tfidf_upstream_gate_backend as context_module


@dataclass
class _TestNuisanceStack:
    offset: float

    def predict(self, texts):
        values = np.asarray(
            [self.offset + 0.01 * len(str(text).split()) for text in texts],
            dtype=float,
        )
        return values, {"test_view": values.copy()}


@dataclass
class _TestTopicBank:
    column: int
    scale: float

    def transform(self, matrix):
        values = sparse.csr_matrix(matrix)[:, [int(self.column)]].toarray()
        return np.asarray(values, dtype=float) * float(self.scale)


def _topic_metadata(bank: str, term: str) -> dict:
    return {
        "bank": bank,
        "topics": [
            {
                "topic_id": f"{bank}_topic_001",
                "bank": bank,
                "terms": [{"term": term, "loading": 1.0}],
            }
        ],
    }


@pytest.fixture
def shared_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "stage1.json"
    config_path.write_text("{}", encoding="utf-8")
    config = AppliedInferenceConfig()
    forest = config.architecture.multi_model_forest
    forest.bow_views = forest.bow_views[:1]
    forest.tfidf_topic.topic_count = 1
    snapshot = SimpleNamespace(
        source_path=config_path.resolve(),
        sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        applied_config=lambda: copy.deepcopy(config),
        verify_source=lambda: None,
    )
    monkeypatch.setattr(
        context_module,
        "_historical_stage1_config_snapshot",
        lambda _path, _snapshot=None: snapshot,
    )

    calls: list[dict] = []

    def fake_fit_tfidf_topic_context(**kwargs):
        fit_df = kwargs["fit_df"]
        heldout_df = kwargs["heldout_df"]
        text_column = kwargs["text_column"]
        output = Path(kwargs["artifact_dir"])
        output.mkdir(parents=True, exist_ok=True)
        fit_texts = fit_df[text_column].fillna("").astype(str).tolist()
        heldout_texts = heldout_df[text_column].fillna("").astype(str).tolist()
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit(fit_texts)
        vocabulary = vectorizer.vocabulary_
        required_terms = ("baseline", "risk", "stable disease")
        if any(term not in vocabulary for term in required_terms):
            raise AssertionError("test fit context omitted a required deterministic term")
        fitted = FittedTopicContext(
            common_vectorizer=vectorizer,
            treatment_stack=_TestNuisanceStack(0.20),
            outcome_stack=_TestNuisanceStack(0.40),
            topic_banks={
                "treatment": _TestTopicBank(vocabulary["baseline"], 1.0),
                "outcome": _TestTopicBank(vocabulary["risk"], 2.0),
                "effect": _TestTopicBank(vocabulary["stable disease"], 3.0),
            },
            config_hash="test-fit-config-v1",
        )
        fitted_path = output / "fitted_context.joblib"
        joblib.dump(fitted, fitted_path)
        topic_metadata = {
            "treatment": _topic_metadata("treatment", "baseline"),
            "outcome": _topic_metadata("outcome", "risk"),
            "effect": _topic_metadata("effect", "stable disease"),
        }
        scores_path = output / "effect_ngram_scores.parquet"
        pd.DataFrame(
            {
                "feature": ["brain metastases", "stable disease"],
                "eligible": [True, True],
                "fit_signed_score": [4.0, 2.0],
                "combined_importance": [4.0, 2.0],
                "support_control": [5, 5],
                "support_treated": [5, 5],
            }
        ).to_parquet(scores_path, index=False)
        transformed = fitted.transform_topics(heldout_texts)
        heldout_topic_path = output / "heldout_topic_values.npz"
        np.savez(heldout_topic_path, **transformed)
        treatment_values, treatment_views = fitted.treatment_stack.predict(heldout_texts)
        outcome_values, outcome_views = fitted.outcome_stack.predict(heldout_texts)
        nuisance_path = output / "nuisance_predictions.parquet"
        nuisance_rows = []
        for position, row_id in enumerate(heldout_df["_oci_row_id"].astype(int)):
            nuisance_rows.append(
                {
                    "_oci_row_id": int(row_id),
                    "prediction_scope": "external_heldout",
                    "treatment_stacked": float(treatment_values[position]),
                    "outcome_stacked": float(outcome_values[position]),
                    "treatment_view__test_view": float(treatment_views["test_view"][position]),
                    "outcome_view__test_view": float(outcome_views["test_view"][position]),
                }
            )
        pd.DataFrame(nuisance_rows).to_parquet(nuisance_path, index=False)
        metadata = {
            "schema_version": "test-context-v1",
            "scope_id": kwargs["scope_id"],
            "fit_row_ids": fit_df["_oci_row_id"].astype(int).tolist(),
            "heldout_row_ids": heldout_df["_oci_row_id"].astype(int).tolist(),
            "config_hash": fitted.config_hash,
            "topic_banks": topic_metadata,
            "artifacts": {
                "fitted_context": str(fitted_path.resolve()),
                "ngram_scores": {"effect": str(scores_path.resolve())},
                "heldout_topic_values": str(heldout_topic_path.resolve()),
                "nuisance_predictions": str(nuisance_path.resolve()),
            },
        }
        (output / "context_metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        calls.append(
            {
                "fit_ids": tuple(fit_df["_oci_row_id"].astype(int)),
                "heldout_ids": tuple(heldout_df["_oci_row_id"].astype(int)),
                "heldout_columns": tuple(heldout_df.columns),
                "artifact_dir": str(output),
            }
        )
        return metadata

    monkeypatch.setattr(spent_module, "fit_tfidf_topic_context", fake_fit_tfidf_topic_context)
    monkeypatch.setattr(context_module, "fit_tfidf_topic_context", fake_fit_tfidf_topic_context)

    spent_delegate = TfidfTopicOrphanSpentDiscoveryBackend(
        stage1_config_path=config_path,
    )
    context_delegate = TfidfTopicOrphanContextBackend(
        stage1_config_path=config_path,
        max_orphan_features=2,
        minimum_orphan_arm_support=2,
    )
    shared = build_shared_tfidf_context_fit_backends(
        spent_discovery_backend=spent_delegate,
        context_backend=context_delegate,
    )
    return SimpleNamespace(
        calls=calls,
        config_path=config_path,
        spent=shared.spent_discovery_backend,
        context=shared.context_backend,
        context_delegate=context_delegate,
        service=shared.service,
        shared=shared,
    )


def _fit_inputs():
    return {
        "outer_fold": 2,
        "exact_spent_row_ids": (1, 2, 3, 4, 5, 6),
        "spent_texts": (
            "baseline risk brain metastases stable disease",
            "baseline low risk stable disease",
            "baseline risk brain metastases",
            "baseline low risk stable disease",
            "baseline risk brain metastases",
            "baseline low risk stable disease",
        ),
        "spent_treatment": np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        "spent_outcome": np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
    }


def _prediction_inputs(fit: dict, *, work_dir: Path):
    return {
        "outer_fold": fit["outer_fold"],
        "context_row_ids": fit["exact_spent_row_ids"],
        "context_texts": fit["spent_texts"],
        "context_treatment": fit["spent_treatment"],
        "context_outcome": fit["spent_outcome"],
        "gate_row_ids": (10, 11),
        "gate_texts": (
            "baseline risk brain metastases",
            "baseline stable disease",
        ),
        "work_dir": work_dir,
    }


def test_exact_spent_fit_is_transform_only_and_numerically_equivalent(
    tmp_path: Path,
    shared_fixture,
) -> None:
    fit = _fit_inputs()
    shared_fixture.spent.fit_discovery(
        **fit,
        review_round=1,
        work_dir=tmp_path / "spent",
    )
    assert len(shared_fixture.calls) == 1
    assert shared_fixture.calls[0]["heldout_columns"] == ("_oci_row_id", "clinical_text")
    shared = shared_fixture.context.fit_predict(
        **_prediction_inputs(fit, work_dir=tmp_path / "shared_should_remain_empty")
    )
    assert len(shared_fixture.calls) == 1
    assert shared_fixture.service.reuse_count == 1

    direct = shared_fixture.context_delegate.fit_predict(
        **_prediction_inputs(fit, work_dir=tmp_path / "direct")
    )
    assert len(shared_fixture.calls) == 2
    assert shared.gate_row_ids == direct.gate_row_ids
    assert shared.calibrated_source_names == direct.calibrated_source_names
    assert shared.feature_names == direct.feature_names
    assert shared.feature_kinds == direct.feature_kinds
    assert shared.feature_roles == direct.feature_roles
    np.testing.assert_allclose(shared.feature_values, direct.feature_values, rtol=0, atol=0)
    assert not (tmp_path / "shared_should_remain_empty").exists()


def test_factory_binds_both_wrappers_to_one_authenticated_service(shared_fixture) -> None:
    shared = shared_fixture.shared
    assert shared.spent_discovery_backend.service is shared.service
    assert shared.context_backend.service is shared.service
    assert shared.context_backend.backend is shared_fixture.context_delegate
    spent_identity = shared.spent_discovery_backend.identity()
    context_identity = shared.context_backend.identity()
    assert spent_identity["service"] == context_identity["service"]
    transform_only = {"max_orphan_features", "minimum_orphan_arm_support"}
    assert {
        key: value
        for key, value in spent_identity["fit_source"].items()
        if key not in transform_only
    } == {
        key: value
        for key, value in context_identity["delegate"].items()
        if key not in transform_only
    }
    assert len(spent_identity["wrapper_code_sha256"]) == 64
    assert len(context_identity["delegate_module_sha256"]) == 64


def test_gate_and_final_provider_identities_bind_shared_service_and_code(
    tmp_path: Path,
    shared_fixture,
) -> None:
    context = shared_fixture.shared.context_backend
    gate = ContextFitUpstreamGateProvider(tmp_path / "gate", backend=context)
    final = FinalContextFitUpstreamProducer(tmp_path / "final", backend=context)

    context_identity = context.identity()
    assert gate.identity()["backend"] == context_identity
    final_identity = final.identity()
    assert final_identity["backend_identity"] == context_identity
    runtime = final_identity["backend_runtime_attestation"]
    assert runtime["class_qualname"] == "SharedTfidfContextBackend"
    assert len(runtime["module_file_sha256"]) == 64
    assert runtime["members"][0]["class_qualname"] == "TfidfTopicOrphanContextBackend"
    assert final_identity["backend_identity"]["service"]["fit_binding"] == (
        "exact_ordered_rows_text_treatment_outcome_and_outer_fold_v1"
    )


def test_factory_fails_closed_when_spent_and_context_fit_configs_differ(
    shared_fixture,
) -> None:
    incompatible_context = TfidfTopicOrphanContextBackend(
        stage1_config_path=shared_fixture.config_path,
        outcome_type="continuous",
    )
    with pytest.raises(ValueError, match="identity is incompatible"):
        build_shared_tfidf_context_fit_backends(
            spent_discovery_backend=shared_fixture.shared.spent_discovery_backend.backend,
            context_backend=incompatible_context,
        )


def test_subset_or_mismatched_context_delegates_and_latest_spent_context_is_active(
    tmp_path: Path,
    shared_fixture,
) -> None:
    fit = _fit_inputs()
    shared_fixture.spent.fit_discovery(
        **fit,
        review_round=1,
        work_dir=tmp_path / "spent_first",
    )
    calls_after_registration = len(shared_fixture.calls)
    subset = _prediction_inputs(fit, work_dir=tmp_path / "subset")
    for field in (
        "context_row_ids",
        "context_texts",
        "context_treatment",
        "context_outcome",
    ):
        subset[field] = subset[field][:-1]
    subset["gate_row_ids"] = (6,)
    subset["gate_texts"] = (fit["spent_texts"][-1],)
    shared_fixture.context.fit_predict(**subset)
    assert len(shared_fixture.calls) == calls_after_registration + 1
    assert shared_fixture.service.reuse_count == 0

    mismatched = _prediction_inputs(fit, work_dir=tmp_path / "mismatched")
    mismatched["context_outcome"] = np.asarray(fit["spent_outcome"], dtype=float).copy()
    mismatched["context_outcome"][0] = 1.0
    shared_fixture.context.fit_predict(**mismatched)
    assert len(shared_fixture.calls) == calls_after_registration + 2
    assert shared_fixture.service.reuse_count == 0

    second = copy.deepcopy(fit)
    second["exact_spent_row_ids"] = (*fit["exact_spent_row_ids"], 7)
    second["spent_texts"] = (*fit["spent_texts"], "baseline risk stable disease")
    second["spent_treatment"] = np.append(fit["spent_treatment"], 1.0)
    second["spent_outcome"] = np.append(fit["spent_outcome"], 0.0)
    shared_fixture.spent.fit_discovery(
        **second,
        review_round=2,
        work_dir=tmp_path / "spent_second",
    )
    calls_after_second = len(shared_fixture.calls)
    shared_fixture.context.fit_predict(
        **_prediction_inputs(fit, work_dir=tmp_path / "old_no_longer_active")
    )
    assert len(shared_fixture.calls) == calls_after_second + 1
    assert shared_fixture.service.reuse_count == 0
    shared_fixture.context.fit_predict(
        **_prediction_inputs(second, work_dir=tmp_path / "second_exact_hit")
    )
    assert len(shared_fixture.calls) == calls_after_second + 1
    assert shared_fixture.service.reuse_count == 1


def test_service_identity_is_immutable_and_a_new_process_analogue_cannot_import_fit(
    tmp_path: Path,
    shared_fixture,
) -> None:
    fit = _fit_inputs()
    shared_fixture.spent.fit_discovery(
        **fit,
        review_round=1,
        work_dir=tmp_path / "spent",
    )
    identity = shared_fixture.context.identity()
    assert identity["reuse_condition"] == "most_recent_exact_spent_fit_key_match_only_v1"
    assert identity["non_exact_and_subset_calls"] == "delegate_unchanged_v1"
    assert identity["gate_labels_accepted"] is False
    assert identity["cross_run_artifact_acceptance"] is False
    service_identity = shared_fixture.service.identity()
    assert service_identity["storage"] == "private_current_process_memory_only"
    assert service_identity["cross_run_artifact_or_joblib_acceptance"] is False
    assert service_identity["cache_path_or_import_api_exposed"] is False
    assert len(service_identity["service_code_sha256"]) == 64

    fresh_service = InMemorySharedTfidfContextFitService(
        source_backend_identity=shared_fixture.context_delegate.identity(),
    )
    fresh_context = SharedTfidfContextBackend(
        backend=shared_fixture.context_delegate,
        service=fresh_service,
    )
    before = len(shared_fixture.calls)
    fresh_context.fit_predict(
        **_prediction_inputs(fit, work_dir=tmp_path / "fresh_service_delegate")
    )
    assert len(shared_fixture.calls) == before + 1
    assert fresh_service.registered_fit_count == 0
    assert fresh_service.reuse_count == 0


def test_service_rejects_incompatible_fit_source_identity(shared_fixture) -> None:
    incompatible = copy.deepcopy(shared_fixture.context_delegate.identity())
    incompatible["outcome_type"] = "continuous"
    with pytest.raises(ValueError, match="identity is incompatible"):
        shared_fixture.service.assert_source_identity(incompatible)


def test_service_rejects_nonbinary_context_treatment(
    tmp_path: Path,
    shared_fixture,
) -> None:
    fit = _fit_inputs()
    fit["spent_treatment"] = np.asarray(fit["spent_treatment"], dtype=float).copy()
    fit["spent_treatment"][0] = 0.25
    with pytest.raises(ValueError, match="must be binary"):
        shared_fixture.spent.fit_discovery(
            **fit,
            review_round=1,
            work_dir=tmp_path / "invalid",
        )


def test_identity_rechecks_wrapper_delegate_and_module_runtime_attestations(
    shared_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_module_digest = shared_module._module_sha256()
    monkeypatch.setattr(shared_module, "_module_sha256", lambda: "0" * 64)
    with pytest.raises(RuntimeError, match="code changed"):
        shared_fixture.context.identity()
    monkeypatch.setattr(shared_module, "_module_sha256", lambda: original_module_digest)

    original_module_attestor = shared_module._module_for_object_sha256
    monkeypatch.setattr(shared_module, "_module_for_object_sha256", lambda _value: "1" * 64)
    with pytest.raises(RuntimeError, match="runtime module changed|backend module changed"):
        shared_fixture.context.identity()
    monkeypatch.setattr(
        shared_module,
        "_module_for_object_sha256",
        original_module_attestor,
    )

    shared_fixture.context_delegate._identity["max_orphan_features"] += 1
    with pytest.raises(RuntimeError, match="identity changed"):
        shared_fixture.context.identity()


@pytest.mark.parametrize("method_name", ["identity", "fit_predict"])
def test_context_wrapper_rejects_delegate_instance_method_overrides(
    shared_fixture,
    method_name: str,
) -> None:
    setattr(shared_fixture.context_delegate, method_name, lambda *args, **kwargs: {})
    with pytest.raises(RuntimeError, match="per-instance method overrides"):
        shared_fixture.context.identity()


def test_spent_wrapper_rejects_delegate_instance_fit_override(shared_fixture) -> None:
    shared_fixture.spent.backend.fit_discovery = lambda **_kwargs: None
    with pytest.raises(RuntimeError, match="per-instance method overrides"):
        shared_fixture.spent.identity()


@pytest.mark.parametrize("method_name", ["assert_source_identity", "transform_active_exact"])
def test_context_wrapper_rejects_service_instance_method_overrides(
    shared_fixture,
    method_name: str,
) -> None:
    object.__setattr__(shared_fixture.service, method_name, lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="per-instance method overrides"):
        shared_fixture.context.identity()


def test_wrapper_methods_cannot_be_replaced_per_instance(shared_fixture) -> None:
    with pytest.raises(AttributeError, match="cannot be overridden"):
        shared_fixture.context.fit_predict = lambda **_kwargs: None

    object.__setattr__(shared_fixture.context, "fit_predict", lambda **_kwargs: None)
    with pytest.raises(RuntimeError, match="instance override|per-instance method overrides"):
        shared_fixture.context.identity()


def test_service_methods_cannot_be_replaced_per_instance(shared_fixture) -> None:
    with pytest.raises(AttributeError, match="cannot be overridden"):
        shared_fixture.service.transform_active_exact = lambda **_kwargs: None

    object.__setattr__(
        shared_fixture.service,
        "transform_active_exact",
        lambda **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="instance override"):
        shared_fixture.service.transform_active_exact


def test_factory_rejects_preexisting_delegate_instance_override(shared_fixture) -> None:
    context_delegate = TfidfTopicOrphanContextBackend(
        stage1_config_path=shared_fixture.config_path,
    )
    context_delegate.fit_predict = lambda **_kwargs: None
    with pytest.raises(RuntimeError, match="per-instance method overrides"):
        build_shared_tfidf_context_fit_backends(
            spent_discovery_backend=shared_fixture.spent.backend,
            context_backend=context_delegate,
        )
