from __future__ import annotations

import copy
import hashlib
import importlib.abc
import importlib.util
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import oci.inference.neural_query_context_backend as query_context_module

from oci.inference.all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from oci.inference.all_evidence_fusion import prepare_all_evidence_fusion
from oci.inference.all_evidence_fusion_runner import AllEvidenceFusionRunner
from oci.inference.neural_query_agentic_forest import NeuralQueryAgenticForestConfig
from oci.inference.neural_query_context_backend import (
    ContextFitNeuralQueryService,
    NeuralQueryContextBackend,
    NeuralQuerySpentDiscoveryBackend,
    NeuralQuerySpentEvidenceProvider,
)
from oci.inference.neural_query_signal_artifact import query_signal_columns
from oci.inference.stable_context_fit_upstream_backend import (
    CrossFitStableUpstreamBackend,
    CrossFitStableUpstreamSchemaConfig,
    PrecommittedRawFeatureFamily,
)
from oci.inference.review_spent_evidence_provider import (
    ContextFitReviewSpentEvidenceProvider,
)

_TEST_TEXTS = (
    "smoking status current",
    "stage iv adenocarcinoma",
    "performance status two",
    "liver metastasis present",
    "future sealed wording",
)


class _FakeBoundFrozenEmbeddings:
    def __init__(self, cache: "_FakeFrozenEmbeddings", row_ids: tuple[int, ...]) -> None:
        self.cache = cache
        self.row_ids = row_ids

    def identity(self):
        return {"cache": self.cache.identity(), "row_ids": list(self.row_ids)}

    def chunk_matrices(self, row_ids):
        requested = tuple(map(int, row_ids))
        if not set(requested) <= set(self.row_ids):
            raise ValueError("embedding provider refuses a non-spent row")
        self.cache.matrix_requests.append(requested)
        return [np.asarray([[float(row_id + 1), 1.0]], dtype=np.float32) for row_id in requested]

    def chunk_texts(self, row_ids):
        requested = tuple(map(int, row_ids))
        if not set(requested) <= set(self.row_ids):
            raise ValueError("embedding provider refuses a non-spent row")
        self.cache.text_requests.append(requested)
        return [[self.cache.texts[row_id]] for row_id in requested]


class _FakeFrozenEmbeddings:
    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts
        self.matrix_requests: list[tuple[int, ...]] = []
        self.text_requests: list[tuple[int, ...]] = []

    @property
    def row_count(self):
        return len(self.texts)

    def identity(self):
        return {"provider": "fake_frozen_embeddings", "row_count": len(self.texts)}

    def bind_spent(self, row_ids, texts):
        rows = tuple(map(int, row_ids))
        exact = tuple(texts)
        if len(rows) != len(exact) or any(
            self.texts[row_id] != text for row_id, text in zip(rows, exact)
        ):
            raise ValueError("spent text does not match its frozen embedding cache row")
        return _FakeBoundFrozenEmbeddings(self, rows)


def _query_config() -> NeuralQueryAgenticForestConfig:
    return NeuralQueryAgenticForestConfig(
        treatment_query_count=1,
        outcome_query_count=1,
        effect_query_count=1,
        query_inner_folds=2,
        initial_pool_size=1,
        query_epochs=1,
        final_refit_epochs=1,
        evidence_top_patients=1,
        evidence_background_patients=1,
        evidence_top_ngrams=2,
        max_features_per_query=1,
        max_raw_feature_candidates=3,
        max_canonical_features=3,
    )


def _service(
    tmp_path: Path,
    *,
    query_config: NeuralQueryAgenticForestConfig | None = None,
) -> ContextFitNeuralQueryService:
    # Unit tests exercise the closed cache/service boundary without constructing
    # a real 4096-dimensional embedding cache or fitting a nuisance stack.
    tmp_path.mkdir(parents=True, exist_ok=True)
    service = object.__new__(ContextFitNeuralQueryService)
    service.cache_dir = tmp_path / "query-cache"
    service._owned_discoveries = {}
    service.dataset_path = tmp_path / "dataset.parquet"
    service.stage1_config_path = tmp_path / "stage1.json"
    service.stage1_config_path.write_text("{}", encoding="utf-8")
    service._stage1_config_snapshot = SimpleNamespace(
        sha256=hashlib.sha256(service.stage1_config_path.read_bytes()).hexdigest(),
        verify_source=lambda: None,
    )
    service.text_column = "text"
    service.embedding_cache = _FakeFrozenEmbeddings(_TEST_TEXTS)
    service._dataset_row_count = service.embedding_cache.row_count
    service._nuisance_views = ({"name": "test_unigram_view"},)
    service.query_config = query_config or _query_config()
    service.nuisance_folds = 2
    service.devices = ("cpu",)
    service.seed = 13
    service.outcome_type = "binary"
    service._identity = service._identity_payload()
    return service


def _discovery():
    return {
        "banks": {
            bank: {
                "queries": np.asarray([[1.0, 0.0]], dtype=np.float32),
                "records": [
                    {
                        "query_id": f"{bank}_context_query_001",
                        "member_count": 2,
                        "member_subfolds": [1, 2],
                        "fit_standardized_score": 0.4,
                    }
                ],
            }
            for bank in ("treatment", "outcome", "effect")
        },
        "subfold_audit": [],
    }


def _patch_discovery(monkeypatch, calls: list[tuple[int, ...]]) -> None:
    def fake_fit(**kwargs):
        calls.append(tuple(kwargs["row_ids"]))
        return _discovery()

    def fake_evidence(*, bank, row_ids, chunk_matrices, **_kwargs):
        assert tuple(row_ids) == (0, 1, 2, 3)
        assert len(chunk_matrices) == len(row_ids)
        return [
            {
                "query_id": f"{bank}_context_query_001",
                "bank": bank,
                "mechanical_role": ("effect_modifier" if bank == "effect" else "confounder"),
                "member_count": 2,
                "fit_standardized_score": 0.4,
                "top_chunks": [
                    {
                        "_oci_row_id": 0,
                        "chunk_index": 0,
                        "text": "must not cross provider boundary",
                    }
                ],
                "top_contrastive_ngrams": [
                    {"term": f"{bank} concept", "tfidf_contrast": 0.7},
                    {"term": "patient id 12345678", "tfidf_contrast": 9.0},
                    {"term": "response after treatment", "tfidf_contrast": 8.0},
                ],
            }
        ]

    def fake_activations(chunks, queries, **_kwargs):
        assert len(queries) == 1
        return np.asarray([[float(chunk[0, 0])] for chunk in chunks], dtype=float)

    monkeypatch.setattr(
        "oci.inference.neural_query_context_backend._fit_context_query_discovery",
        fake_fit,
    )
    monkeypatch.setattr(
        "oci.inference.neural_query_context_backend.build_query_evidence",
        fake_evidence,
    )
    monkeypatch.setattr(
        "oci.inference.neural_query_context_backend.soft_retrieval_activations",
        fake_activations,
    )


def test_spent_evidence_and_gate_features_share_one_exact_context_fit(tmp_path, monkeypatch):
    calls: list[tuple[int, ...]] = []
    _patch_discovery(monkeypatch, calls)
    service = _service(tmp_path)
    spent_provider = NeuralQuerySpentEvidenceProvider(service)
    backend = NeuralQueryContextBackend(service)
    spent_rows = (0, 1, 2, 3)
    spent_texts = tuple(_TEST_TEXTS[row_id] for row_id in spent_rows)
    treatment = np.asarray([0.0, 1.0, 0.0, 1.0])
    outcome = np.asarray([0.0, 1.0, 1.0, 0.0])

    evidence_inputs = spent_provider.get_spent_evidence_inputs(
        outer_fold=2,
        review_round=0,
        exact_spent_row_ids=spent_rows,
        exact_sealed_row_ids=(4,),
        spent_texts=spent_texts,
        spent_treatment=treatment,
        spent_outcome=outcome,
    )
    assert calls == [spent_rows]
    item = evidence_inputs[0]
    assert item.provenance.train_row_ids == spent_rows
    assert item.provenance.heldout_row_ids == (4,)
    assert item.provenance.scope == "inner_train"
    assert item.provenance.inner_fold == 1
    serialized = repr(item.payload)
    assert "must not cross provider boundary" not in serialized
    assert "_oci_row_id" not in serialized
    assert [row["bank"] for row in item.payload["query_evidence"]] == [
        "treatment",
        "outcome",
        "effect",
    ]

    prediction = backend.fit_predict(
        outer_fold=2,
        context_row_ids=spent_rows,
        context_texts=spent_texts,
        context_treatment=treatment,
        context_outcome=outcome,
        gate_row_ids=(4,),
        gate_texts=(_TEST_TEXTS[4],),
        work_dir=tmp_path / "unused",
    )
    assert calls == [spent_rows], "gate transformation must reuse the spent-context fit"
    assert prediction.feature_values.shape == (1, 9)
    assert prediction.feature_names == (
        "neural_query_treatment_signed_mean",
        "neural_query_treatment_absolute_max",
        "neural_query_treatment_signed_order_01",
        "neural_query_outcome_signed_mean",
        "neural_query_outcome_absolute_max",
        "neural_query_outcome_signed_order_01",
        "neural_query_effect_signed_mean",
        "neural_query_effect_absolute_max",
        "neural_query_effect_signed_order_01",
    )
    np.testing.assert_allclose(prediction.feature_values, np.full((1, 9), 5.0))
    assert prediction.feature_roles == (
        *(PROPENSITY_NUISANCE_FEATURE_ROLE for _ in range(3)),
        *(OUTCOME_NUISANCE_FEATURE_ROLE for _ in range(3)),
        *(UNCALIBRATED_EFFECT_MODIFIER_ROLE for _ in range(3)),
    )
    assert prediction.calibrated_source_names == ()
    assert backend.identity()["fit_local_query_indices_exposed"] is False


def _unequal_query_config() -> NeuralQueryAgenticForestConfig:
    return replace(
        _query_config(),
        treatment_query_count=2,
        outcome_query_count=3,
        effect_query_count=4,
        initial_pool_size=4,
        max_raw_feature_candidates=9,
        max_canonical_features=9,
    )


def _moment_schema(config: NeuralQueryAgenticForestConfig):
    roles = {
        "treatment": PROPENSITY_NUISANCE_FEATURE_ROLE,
        "outcome": OUTCOME_NUISANCE_FEATURE_ROLE,
        "effect": UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    }
    artifact_names = query_signal_columns(
        {bank: config.query_count(bank) for bank in ("treatment", "outcome", "effect")}
    )
    return CrossFitStableUpstreamSchemaConfig(
        namespace="neural_moments",
        raw_families=tuple(
            PrecommittedRawFeatureFamily(
                source_kind=f"neural_query_{bank}_moments",
                consumer_role=roles[bank],
                signed_order_width=config.query_count(bank),
                exact_passthrough_feature_names=tuple(
                    name for name in artifact_names if name.startswith(f"neural_query_{bank}_")
                ),
            )
            for bank in ("treatment", "outcome", "effect")
        ),
    )


def _permutable_discovery(
    config: NeuralQueryAgenticForestConfig,
    *,
    reverse: bool = False,
    flip_treatment_second: bool = False,
):
    markers = {
        "treatment": (9.0, 2.0),
        # Deliberately cancellation-sensitive: unsorted floating reduction can
        # differ after a fit-local query permutation, while v3 sorts first.
        "outcome": (1.0e16, 1.0e16, 1.0),
        "effect": (8.0, 4.0, 7.0, 5.0),
    }
    scores = {
        "treatment": (0.0, -1.0 if flip_treatment_second else 1.0),
        "outcome": (1.0, -1.0, 1.0),
        "effect": (-1.0, 1.0, 0.0, 1.0),
    }
    banks = {}
    for bank in ("treatment", "outcome", "effect"):
        count = config.query_count(bank)
        order = tuple(reversed(range(count))) if reverse else tuple(range(count))
        banks[bank] = {
            "queries": np.asarray(
                [[markers[bank][index], 1.0] for index in order],
                dtype=np.float32,
            ),
            "records": [
                {
                    "query_id": f"{bank}_fit_local_{index + 1:03d}",
                    "member_count": 2,
                    "member_subfolds": [1, 2],
                    "fit_standardized_score": scores[bank][index],
                }
                for index in order
            ],
        }
    return {"banks": banks, "subfold_audit": []}


def _backend_with_discovery(tmp_path: Path, config, discovery):
    service = _service(tmp_path, query_config=config)
    service.discovery_for_context = lambda **_kwargs: (copy.deepcopy(discovery), "test-cache")
    return NeuralQueryContextBackend(service)


def _moment_prediction(backend, tmp_path: Path):
    return backend.fit_predict(
        outer_fold=2,
        context_row_ids=(0, 1, 2, 3),
        context_texts=_TEST_TEXTS[:4],
        context_treatment=np.asarray([0.0, 1.0, 0.0, 1.0]),
        context_outcome=np.asarray([0.0, 1.0, 1.0, 0.0]),
        gate_row_ids=(4,),
        gate_texts=(_TEST_TEXTS[4],),
        work_dir=tmp_path / "unused",
    )


def test_exact_neural_moments_are_permutation_invariant_and_not_double_summarized(
    tmp_path,
    monkeypatch,
):
    config = _unequal_query_config()

    def fake_activations(chunks, queries, **_kwargs):
        return np.asarray(
            [
                [float(query[0]) + float(chunk[0, 0]) / 10.0 for query in queries]
                for chunk in chunks
            ],
            dtype=float,
        )

    monkeypatch.setattr(query_context_module, "soft_retrieval_activations", fake_activations)
    first_child = _backend_with_discovery(
        tmp_path / "first",
        config,
        _permutable_discovery(config),
    )
    permuted_child = _backend_with_discovery(
        tmp_path / "permuted",
        config,
        _permutable_discovery(config, reverse=True),
    )
    flipped_child = _backend_with_discovery(
        tmp_path / "flipped",
        config,
        _permutable_discovery(config, flip_treatment_second=True),
    )

    direct = _moment_prediction(first_child, tmp_path)
    permuted = _moment_prediction(permuted_child, tmp_path)
    np.testing.assert_array_equal(direct.feature_values, permuted.feature_values)
    assert direct.feature_names == permuted.feature_names
    assert direct.feature_names == query_signal_columns(
        {bank: config.query_count(bank) for bank in ("treatment", "outcome", "effect")}
    )
    assert direct.feature_kinds == (
        *("neural_query_treatment_moments" for _ in range(4)),
        *("neural_query_outcome_moments" for _ in range(5)),
        *("neural_query_effect_moments" for _ in range(6)),
    )
    assert set(zip(direct.feature_kinds, direct.feature_roles)) == {
        ("neural_query_treatment_moments", PROPENSITY_NUISANCE_FEATURE_ROLE),
        ("neural_query_outcome_moments", OUTCOME_NUISANCE_FEATURE_ROLE),
        ("neural_query_effect_moments", UNCALIBRATED_EFFECT_MODIFIER_ROLE),
    }

    # The treatment query with zero fit score has the largest raw activation
    # (9.5). It contributes zero to signed mean/order but must still determine
    # the v3 absolute-max moment.
    np.testing.assert_allclose(direct.feature_values[0, :4], [1.25, 9.5, 2.5, 0.0])

    stable = CrossFitStableUpstreamBackend(first_child, config=_moment_schema(config))
    stable_prediction = _moment_prediction(stable, tmp_path)
    assert stable_prediction.feature_names == direct.feature_names
    assert stable_prediction.feature_kinds == direct.feature_kinds
    assert stable_prediction.feature_roles == direct.feature_roles
    assert stable_prediction.feature_values.shape == direct.feature_values.shape == (1, 15)
    np.testing.assert_array_equal(stable_prediction.feature_values, direct.feature_values)
    assert stable.identity()["exact_preaggregated_features_reduced_again"] is False

    flipped = _moment_prediction(flipped_child, tmp_path)
    assert flipped.feature_values[0, 1] == direct.feature_values[0, 1]
    assert not np.array_equal(flipped.feature_values[0, :4], direct.feature_values[0, :4])


@pytest.mark.parametrize("bad_score", [None, np.nan, np.inf, "not-a-score", True])
def test_neural_context_discovery_rejects_missing_or_nonfinite_fit_scores(
    tmp_path,
    bad_score,
):
    service = _service(tmp_path)
    discovery = _discovery()
    discovery["banks"]["effect"]["records"][0]["fit_standardized_score"] = bad_score

    with pytest.raises(ValueError, match="fit standardized score must be finite"):
        service._validate_discovery(discovery)


def test_spent_evidence_never_requests_sealed_chunks(tmp_path, monkeypatch):
    calls: list[tuple[int, ...]] = []
    _patch_discovery(monkeypatch, calls)
    service = _service(tmp_path)
    provider = NeuralQuerySpentEvidenceProvider(service)
    kwargs = {
        "outer_fold": 1,
        "review_round": 1,
        "exact_spent_row_ids": (0, 1, 2, 3),
        "exact_sealed_row_ids": (4,),
        "spent_texts": _TEST_TEXTS[:4],
        "spent_treatment": np.asarray([0.0, 1.0, 0.0, 1.0]),
        "spent_outcome": np.asarray([0.0, 1.0, 1.0, 0.0]),
    }
    first = provider.get_spent_evidence_inputs(**kwargs)[0]
    second = provider.get_spent_evidence_inputs(**kwargs)[0]
    assert first.payload == second.payload
    assert first.provenance == second.provenance
    assert calls == [(0, 1, 2, 3)]
    assert service.embedding_cache.matrix_requests == [
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
    ]
    assert service.embedding_cache.text_requests == [(0, 1, 2, 3), (0, 1, 2, 3)]
    assert all(4 not in rows for rows in service.embedding_cache.matrix_requests)
    assert all(4 not in rows for rows in service.embedding_cache.text_requests)


def test_live_service_state_mutation_fails_closed(tmp_path):
    service = _service(tmp_path)
    service.devices = ("cpu", "cpu")
    with pytest.raises(RuntimeError, match="state changed after binding"):
        service.identity()


def test_executable_cache_root_accepts_only_nonexistent_or_empty_real_directory(tmp_path):
    nonexistent = tmp_path / "new-cache"
    assert (
        query_context_module._validated_fresh_executable_cache_root(nonexistent)
        == nonexistent.resolve()
    )

    empty = tmp_path / "empty-cache"
    empty.mkdir()
    assert query_context_module._validated_fresh_executable_cache_root(empty) == empty.resolve()

    populated = tmp_path / "populated-cache"
    populated.mkdir()
    (populated / "query_discovery.joblib").write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="pre-existing checkpoints are forbidden"):
        query_context_module._validated_fresh_executable_cache_root(populated)

    symlink = tmp_path / "linked-cache"
    symlink.symlink_to(empty, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        query_context_module._validated_fresh_executable_cache_root(symlink)


def test_dependency_code_change_invalidates_service_identity(tmp_path, monkeypatch):
    service = _service(tmp_path)
    changed = dict(query_context_module._dependency_code_sha256s())
    changed["neural_cohort_witness"] = "f" * 64
    monkeypatch.setattr(
        query_context_module,
        "_dependency_code_sha256s",
        lambda: changed,
    )
    with pytest.raises(RuntimeError, match="state changed after binding"):
        service.identity()


def test_discovery_runtime_code_change_invalidates_service_identity(tmp_path, monkeypatch):
    service = _service(tmp_path)
    monkeypatch.setattr(
        query_context_module,
        "_query_discovery_runtime_code_sha256",
        lambda: "f" * 64,
    )
    with pytest.raises(RuntimeError, match="state changed after binding"):
        service.identity()


def test_context_discovery_import_and_use_forbid_oracle_experiment_package(monkeypatch):
    for name in tuple(sys.modules):
        if name == "oracle_experiment_scripts" or name.startswith("oracle_experiment_scripts."):
            sys.modules.pop(name, None)

    class _ForbidOracleExperimentImport(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "oracle_experiment_scripts" or fullname.startswith(
                "oracle_experiment_scripts."
            ):
                raise ImportError("oracle experiment import is forbidden")
            return None

    blocker = _ForbidOracleExperimentImport()
    sys.meta_path.insert(0, blocker)
    try:
        # Execute both source files under isolated package-qualified module
        # names while the experiment package is fatal.  This covers their full
        # import surfaces without replacing the canonical classes used by the
        # rest of this test process.
        from oci.inference import neural_query_discovery_runtime as discovery_runtime

        def isolated_module(source_module, name):
            spec = importlib.util.spec_from_file_location(name, source_module.__file__)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        isolated_module(
            discovery_runtime,
            "oci.inference._neural_query_discovery_runtime_import_test",
        )
        backend = isolated_module(
            query_context_module,
            "oci.inference._neural_query_context_backend_import_test",
        )
        nuisance = {
            "treatment": {"stacked_oof": np.asarray([0.2, 0.8, 0.3, 0.7])},
            "outcome": {"stacked_oof": np.asarray([0.3, 0.7, 0.6, 0.4])},
        }
        monkeypatch.setattr(
            backend,
            "fit_joint_cross_fitted_nuisance_stacks",
            lambda **_kwargs: nuisance,
        )
        captured = {}

        def fake_runtime(**kwargs):
            captured.update(kwargs)
            return {"banks": {"treatment": {}, "outcome": {}, "effect": {}}}

        monkeypatch.setattr(backend, "fit_in_memory_query_discovery", fake_runtime)
        result = backend._fit_context_query_discovery(
            row_ids=(0, 1, 2, 3),
            chunks=tuple(np.ones((1, 2), dtype=np.float32) for _ in range(4)),
            texts=("row zero", "row one", "row two", "row three"),
            treatment=np.asarray([0.0, 1.0, 0.0, 1.0]),
            outcome=np.asarray([0.0, 1.0, 1.0, 0.0]),
            outcome_binary=True,
            nuisance_views=({"name": "test"},),
            query_config=backend.NeuralQueryAgenticForestConfig(),
            nuisance_folds=2,
            devices=("cpu",),
            seed=13,
        )
    finally:
        sys.meta_path.remove(blocker)

    assert set(result["banks"]) == {"treatment", "outcome", "effect"}
    assert captured["fit_ids"] == (0, 1, 2, 3)
    assert not any(
        name == "oracle_experiment_scripts" or name.startswith("oracle_experiment_scripts.")
        for name in sys.modules
    )


def test_context_discovery_calls_production_in_memory_runtime(monkeypatch):
    nuisance = {
        "treatment": {"stacked_oof": np.asarray([0.2, 0.8, 0.3, 0.7])},
        "outcome": {"stacked_oof": np.asarray([0.3, 0.7, 0.6, 0.4])},
    }
    monkeypatch.setattr(
        query_context_module,
        "fit_joint_cross_fitted_nuisance_stacks",
        lambda **_kwargs: nuisance,
    )
    captured = {}

    def fake_fit_query_discovery(**kwargs):
        captured.update(kwargs)
        return _discovery()

    monkeypatch.setattr(
        query_context_module,
        "fit_in_memory_query_discovery",
        fake_fit_query_discovery,
    )
    result = query_context_module._fit_context_query_discovery(
        row_ids=(0, 1, 2, 3),
        chunks=tuple(np.ones((1, 2), dtype=np.float32) for _ in range(4)),
        texts=_TEST_TEXTS[:4],
        treatment=np.asarray([0.0, 1.0, 0.0, 1.0]),
        outcome=np.asarray([0.0, 1.0, 1.0, 0.0]),
        outcome_binary=True,
        nuisance_views=({"name": "test"},),
        query_config=_query_config(),
        nuisance_folds=2,
        devices=("cpu",),
        seed=13,
    )

    assert result["banks"].keys() == {"treatment", "outcome", "effect"}
    assert captured["fit_ids"] == (0, 1, 2, 3)
    assert "checkpoint_dir" not in captured
    assert "use_executable_checkpoints" not in captured


def test_production_runtime_structurally_has_no_executable_checkpoint_api(monkeypatch):
    from oci.inference import neural_query_discovery_runtime as runtime

    public_parameters = inspect.signature(runtime.fit_in_memory_query_discovery).parameters
    subfold_parameters = inspect.signature(runtime._fit_subfold).parameters
    assert "checkpoint_dir" not in public_parameters
    assert "checkpoint_path" not in public_parameters
    assert "use_executable_checkpoints" not in public_parameters
    assert "checkpoint_path" not in subfold_parameters
    assert "use_executable_checkpoints" not in subfold_parameters
    assert "joblib" not in runtime.__dict__

    class _Predictor:
        def predict(self, texts):
            return np.full(len(texts), 0.5, dtype=float), {}

    def fake_nuisance(**kwargs):
        size = len(kwargs["treatment"])
        return {
            "treatment": {
                "stacked_oof": np.full(size, 0.5, dtype=float),
                "fitted": _Predictor(),
                "metrics": {},
            },
            "outcome": {
                "stacked_oof": np.full(size, 0.5, dtype=float),
                "fitted": _Predictor(),
                "metrics": {},
            },
        }

    def fake_query_fit(*_args, **_kwargs):
        return {
            "queries": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "train_standardized_scores": np.asarray([0.4]),
            "query_drift": np.asarray([0.0]),
            "loss_history": [],
            "objective": "test",
        }

    monkeypatch.setattr(runtime, "fit_joint_cross_fitted_nuisance_stacks", fake_nuisance)
    monkeypatch.setattr(runtime, "fit_soft_target_queries", fake_query_fit)
    monkeypatch.setattr(runtime, "fit_soft_contrast_queries", fake_query_fit)
    monkeypatch.setattr(
        runtime,
        "soft_retrieval_activations",
        lambda chunks, queries, **_kwargs: np.ones((len(chunks), len(queries))),
    )
    monkeypatch.setattr(
        runtime,
        "standardized_direct_target_contrasts",
        lambda *_args, **_kwargs: {"standardized_scores": np.asarray([0.3])},
    )
    monkeypatch.setattr(
        runtime,
        "standardized_cohort_moments",
        lambda *_args, **_kwargs: {"standardized_scores": np.asarray([0.3])},
    )
    monkeypatch.setattr(
        runtime,
        "cohort_contribution",
        lambda u, v: (np.asarray(u) * np.asarray(v), 0.0),
    )

    result = runtime._fit_subfold(
        fold=1,
        train_indices=np.asarray([0, 1]),
        validation_indices=np.asarray([2, 3]),
        row_ids=(0, 1, 2, 3),
        chunks=tuple(np.ones((1, 2), dtype=np.float32) for _ in range(4)),
        texts=_TEST_TEXTS[:4],
        treatment=np.asarray([0.0, 1.0, 0.0, 1.0]),
        outcome=np.asarray([0.0, 1.0, 1.0, 0.0]),
        outcome_binary=True,
        nuisance_views=({"name": "test"},),
        nuisance_folds=2,
        config=_query_config(),
        seed=13,
        device="cpu",
        parent_input_binding_sha256="a" * 64,
    )

    assert result["fold"] == 1
    assert set(result["banks"]) == {"treatment", "outcome", "effect"}
    assert result["identity_payload"]["executable_checkpoint_io"] is False


def test_production_runtime_matches_historical_in_memory_discovery_algorithm(
    tmp_path,
    monkeypatch,
):
    from oci.inference import neural_query_discovery_runtime as runtime
    from oracle_experiment_scripts import run_neural_query_agentic_forest as historical

    config = _query_config()
    row_ids = tuple(range(8))
    texts = tuple(f"synthetic row {row_id}" for row_id in row_ids)
    chunks = tuple(np.asarray([[float(row_id + 1), 1.0]], dtype=np.float32) for row_id in row_ids)
    treatment = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    outcome = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    fit_e = np.linspace(0.2, 0.6, len(row_ids))
    fit_m = np.linspace(0.3, 0.7, len(row_ids))
    nuisance_views = ({"name": "synthetic_view"},)

    def run_implementation(module, fit_discovery, *, historical_runtime):
        trace = []

        class _Predictor:
            def __init__(self, name, offset):
                self.name = name
                self.offset = float(offset)

            def predict(self, prediction_texts):
                trace.append(
                    {
                        "call": "nuisance_predict",
                        "name": self.name,
                        "texts": list(prediction_texts),
                    }
                )
                values = self.offset + 0.01 * np.arange(len(prediction_texts))
                return values, {"source": self.name}

        def fake_nuisance(**kwargs):
            current_treatment = np.asarray(kwargs["treatment"], dtype=float)
            current_outcome = np.asarray(kwargs["outcome"], dtype=float)
            trace.append(
                {
                    "call": "nuisance_fit",
                    "texts": list(kwargs["texts"]),
                    "treatment": current_treatment.tolist(),
                    "outcome": current_outcome.tolist(),
                    "outcome_binary": bool(kwargs["outcome_binary"]),
                    "folds": int(kwargs["folds"]),
                    "random_state": int(kwargs["random_state"]),
                }
            )
            return {
                "treatment": {
                    "stacked_oof": 0.2 + 0.1 * current_treatment,
                    "fitted": _Predictor("treatment", 0.25),
                    "metrics": {"kind": "treatment", "rows": len(current_treatment)},
                },
                "outcome": {
                    "stacked_oof": 0.4 + 0.1 * current_outcome,
                    "fitted": _Predictor("outcome", 0.45),
                    "metrics": {"kind": "outcome", "rows": len(current_outcome)},
                },
            }

        def fake_activations(current_chunks, queries, *, temperature, device):
            query_values = np.asarray(queries, dtype=float)
            row_markers = np.asarray(
                [float(np.asarray(chunk)[0, 0]) for chunk in current_chunks],
                dtype=float,
            )
            values = row_markers[:, None] + query_values[None, :, 0]
            trace.append(
                {
                    "call": "activations",
                    "rows": row_markers.tolist(),
                    "queries": query_values.tolist(),
                    "temperature": float(temperature),
                    "device": str(device),
                }
            )
            return values

        def fake_query_result(
            current_chunks,
            *,
            bank,
            config,
            seed,
            device,
            initial_queries,
            objective,
        ):
            initial = (
                None if initial_queries is None else np.asarray(initial_queries, dtype=np.float32)
            )
            if initial is None:
                base = {"treatment": 1.0, "outcome": 2.0, "effect": 3.0}[bank]
                queries = np.asarray(
                    [
                        [base + float(seed) / 10_000.0 + index, 1.0]
                        for index in range(int(config.n_prototypes))
                    ],
                    dtype=np.float32,
                )
            else:
                queries = initial.copy()
                queries[:, 0] += np.float32(0.01)
            trace.append(
                {
                    "call": "query_fit",
                    "bank": bank,
                    "seed": int(seed),
                    "device": str(device),
                    "initial_queries": None if initial is None else initial.tolist(),
                    "objective": str(objective),
                }
            )
            train_activations = fake_activations(
                current_chunks,
                queries,
                temperature=float(config.temperature),
                device=device,
            )
            return {
                "queries": queries,
                "train_activations": train_activations,
                "train_standardized_scores": np.linspace(
                    0.4,
                    0.4 + 0.1 * (len(queries) - 1),
                    len(queries),
                ),
                "query_drift": np.linspace(0.01, 0.01 * len(queries), len(queries)),
                "loss_history": [float(seed)],
                "objective": str(objective),
            }

        def fake_target_queries(
            current_chunks,
            _target,
            *,
            binary,
            config,
            seed,
            device,
            initial_queries=None,
            target_name="target",
        ):
            return fake_query_result(
                current_chunks,
                bank=str(target_name),
                config=config,
                seed=seed,
                device=device,
                initial_queries=initial_queries,
                objective=f"direct_{target_name}_binary_{bool(binary)}",
            )

        def fake_contrast_queries(
            current_chunks,
            _contribution,
            *,
            center_weights,
            config,
            seed,
            device,
            initial_queries=None,
            objective_name="effect",
        ):
            trace.append(
                {
                    "call": "effect_weights",
                    "weights": np.asarray(center_weights, dtype=float).tolist(),
                }
            )
            return fake_query_result(
                current_chunks,
                bank="effect",
                config=config,
                seed=seed,
                device=device,
                initial_queries=initial_queries,
                objective=objective_name,
            )

        def fake_direct_audit(activations, target, *, binary):
            values = np.asarray(activations, dtype=float)
            target_values = np.asarray(target, dtype=float)
            scores = np.mean(
                (target_values - np.mean(target_values))[:, None] * values,
                axis=0,
            )
            trace.append(
                {
                    "call": "direct_audit",
                    "binary": bool(binary),
                    "target": target_values.tolist(),
                    "scores": scores.tolist(),
                }
            )
            return {"standardized_scores": scores}

        def fake_cohort_contribution(u, v):
            u_values = np.asarray(u, dtype=float)
            v_values = np.asarray(v, dtype=float)
            trace.append(
                {
                    "call": "cohort_contribution",
                    "u": u_values.tolist(),
                    "v": v_values.tolist(),
                }
            )
            return u_values * v_values, 0.125

        def fake_cohort_audit(
            activations,
            u,
            v,
            *,
            constant_effect,
        ):
            values = np.asarray(activations, dtype=float)
            residual_product = np.asarray(u, dtype=float) * np.asarray(v, dtype=float)
            scores = np.mean(residual_product[:, None] * values, axis=0)
            trace.append(
                {
                    "call": "cohort_audit",
                    "constant_effect": float(constant_effect),
                    "scores": scores.tolist(),
                }
            )
            return {"standardized_scores": scores}

        original_consensus = module.build_ungated_consensus_query_bank

        def traced_consensus(
            candidates,
            *,
            candidate_activations,
            n_queries,
            bank,
            seed,
        ):
            trace.append(
                {
                    "call": "consensus",
                    "bank": str(bank),
                    "seed": int(seed),
                    "n_queries": int(n_queries),
                    "candidate_ids": [row["candidate_id"] for row in candidates],
                    "candidate_train_scores": [
                        float(row["train_standardized_score"]) for row in candidates
                    ],
                    "candidate_validation_audits": [
                        float(row["validation_audit_standardized_score"]) for row in candidates
                    ],
                }
            )
            return original_consensus(
                candidates,
                candidate_activations=np.asarray(candidate_activations, dtype=float),
                n_queries=n_queries,
                bank=bank,
                seed=seed,
            )

        with monkeypatch.context() as scoped:
            scoped.setattr(module, "fit_joint_cross_fitted_nuisance_stacks", fake_nuisance)
            scoped.setattr(module, "fit_soft_target_queries", fake_target_queries)
            scoped.setattr(module, "fit_soft_contrast_queries", fake_contrast_queries)
            scoped.setattr(module, "soft_retrieval_activations", fake_activations)
            scoped.setattr(module, "standardized_direct_target_contrasts", fake_direct_audit)
            scoped.setattr(module, "cohort_contribution", fake_cohort_contribution)
            scoped.setattr(module, "standardized_cohort_moments", fake_cohort_audit)
            scoped.setattr(module, "build_ungated_consensus_query_bank", traced_consensus)
            kwargs = {
                "fit_ids": row_ids,
                "fit_chunks": chunks,
                "fit_texts": texts,
                "treatment": treatment,
                "outcome": outcome,
                "outcome_binary": True,
                "fit_e": fit_e,
                "fit_m": fit_m,
                "nuisance_views": nuisance_views,
                "config": config,
                "nuisance_folds": 2,
                "devices": ("cpu",),
                "seed": 31,
            }
            if historical_runtime:
                kwargs.update(
                    {
                        "checkpoint_dir": tmp_path / "unused_historical_checkpoints",
                        "use_executable_checkpoints": False,
                    }
                )
            result = fit_discovery(**kwargs)
        return result, trace

    production_result, production_trace = run_implementation(
        runtime,
        runtime.fit_in_memory_query_discovery,
        historical_runtime=False,
    )
    historical_result, historical_trace = run_implementation(
        historical,
        historical._fit_query_discovery,
        historical_runtime=True,
    )

    assert production_trace == historical_trace
    assert [row["random_state"] for row in production_trace if row["call"] == "nuisance_fit"] == [
        10032,
        10033,
    ]
    assert [
        (row["bank"], row["seed"]) for row in production_trace if row["call"] == "consensus"
    ] == [("treatment", 1031), ("outcome", 1032), ("effect", 1033)]
    assert [
        (row["bank"], row["seed"])
        for row in production_trace
        if row["call"] == "query_fit" and row["initial_queries"] is not None
    ] == [("treatment", 2031), ("outcome", 2032), ("effect", 2033)]

    assert (
        set(production_result["banks"])
        == set(historical_result["banks"])
        == {
            "treatment",
            "outcome",
            "effect",
        }
    )
    for bank in ("treatment", "outcome", "effect"):
        production_bank = production_result["banks"][bank]
        historical_bank = historical_result["banks"][bank]
        np.testing.assert_allclose(production_bank["queries"], historical_bank["queries"])
        np.testing.assert_allclose(
            production_bank["train_activations"],
            historical_bank["train_activations"],
        )
        assert production_bank["records"] == historical_bank["records"]
        assert production_bank["consensus"] == historical_bank["consensus"]
        assert production_bank["objective"] == historical_bank["objective"]
        assert production_bank["all_queries_retained"] is True
        assert production_bank["statistical_gate_applied"] is False

    def comparable_subfold_audits(result):
        return [
            {
                key: value
                for key, value in audit.items()
                if key not in {"identity", "identity_payload"}
            }
            for audit in result["subfold_audit"]
        ]

    assert comparable_subfold_audits(production_result) == comparable_subfold_audits(
        historical_result
    )
    assert production_result["all_queries_retained"] is True
    assert production_result["validation_audits_used_for_selection"] is False
    assert production_result["executable_checkpoint_io"] is False


def test_neural_query_discovery_backend_composes_with_spent_provider(tmp_path, monkeypatch):
    calls: list[tuple[int, ...]] = []
    _patch_discovery(monkeypatch, calls)
    service = _service(tmp_path)
    provider = ContextFitReviewSpentEvidenceProvider(
        backends=(NeuralQuerySpentDiscoveryBackend(service),),
        cache_dir=tmp_path / "spent-provider-cache",
        required_source_families=("neural_query_moments",),
    )
    inputs = provider.get_spent_evidence_inputs(
        outer_fold=1,
        review_round=0,
        exact_spent_row_ids=(0, 1, 2, 3),
        exact_sealed_row_ids=(4,),
        spent_texts=_TEST_TEXTS[:4],
        spent_treatment=np.asarray([0.0, 1.0, 0.0, 1.0]),
        spent_outcome=np.asarray([0.0, 1.0, 1.0, 0.0]),
    )
    assert len(inputs) == 1
    assert inputs[0].source_kind == "neural_query_moments"
    assert inputs[0].provenance.train_row_ids == (0, 1, 2, 3)
    assert inputs[0].provenance.heldout_row_ids == (4,)
    assert calls == [(0, 1, 2, 3)]
    request = prepare_all_evidence_fusion(inputs)
    sanitized = AllEvidenceFusionRunner._sanitize_spent_evidence_catalog(
        request.context()["evidence"]
    )
    ngrams = [ngram for row in sanitized for ngram in row["content"].get("contrastive_ngrams", [])]
    assert {row["term"] for row in ngrams} == {
        "treatment concept",
        "outcome concept",
        "effect concept",
        "response after treatment",
    }
    contrast_by_term = {row["term"]: row["tfidf_contrast"] for row in ngrams}
    assert contrast_by_term["treatment concept"] == 0.7
    assert contrast_by_term["outcome concept"] == 0.7
    assert contrast_by_term["effect concept"] == 0.7
    assert contrast_by_term["response after treatment"] == 8.0


def test_context_cache_reuse_never_executes_self_consistently_replaced_checkpoint(
    tmp_path,
    monkeypatch,
):
    calls: list[tuple[int, ...]] = []
    _patch_discovery(monkeypatch, calls)
    service = _service(tmp_path)
    rows = (0, 1, 2, 3)
    kwargs = {
        "outer_fold": 1,
        "context_row_ids": rows,
        "context_texts": _TEST_TEXTS[:4],
        "context_treatment": np.asarray([0.0, 1.0, 0.0, 1.0]),
        "context_outcome": np.asarray([0.0, 1.0, 1.0, 0.0]),
    }
    original, cache_key = service.discovery_for_context(**kwargs)
    checkpoint = service.cache_dir / cache_key / "query_discovery.joblib"
    manifest_path = service.cache_dir / cache_key / "manifest.json"

    forged = _discovery()
    forged["banks"]["treatment"]["queries"] = np.asarray([[0.0, 1.0]], dtype=np.float32)
    query_context_module.joblib.dump(forged, checkpoint)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"] = query_context_module._sha256_file(checkpoint)
    content = {key: value for key, value in manifest.items() if key != "content_sha256"}
    manifest["content_sha256"] = query_context_module._sha256_json(content)
    manifest_path.write_text(
        query_context_module._canonical_json(manifest) + "\n",
        encoding="utf-8",
    )

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("a mutable executable checkpoint reached joblib.load")

    monkeypatch.setattr(query_context_module.joblib, "load", forbidden_load)
    reused, reused_key = service.discovery_for_context(**kwargs)

    assert reused_key == cache_key
    assert calls == [rows]
    assert np.array_equal(
        reused["banks"]["treatment"]["queries"],
        original["banks"]["treatment"]["queries"],
    )
    assert not np.array_equal(
        reused["banks"]["treatment"]["queries"],
        forged["banks"]["treatment"]["queries"],
    )


def test_context_cache_return_value_cannot_mutate_trusted_in_memory_discovery(
    tmp_path,
    monkeypatch,
):
    calls: list[tuple[int, ...]] = []
    _patch_discovery(monkeypatch, calls)
    service = _service(tmp_path)
    rows = (0, 1, 2, 3)
    kwargs = {
        "outer_fold": 1,
        "context_row_ids": rows,
        "context_texts": _TEST_TEXTS[:4],
        "context_treatment": np.asarray([0.0, 1.0, 0.0, 1.0]),
        "context_outcome": np.asarray([0.0, 1.0, 1.0, 0.0]),
    }
    returned, cache_key = service.discovery_for_context(**kwargs)
    returned["banks"]["treatment"]["queries"][0, 0] = -99.0
    returned["banks"]["treatment"]["records"][0]["fit_standardized_score"] = -99.0

    reused, reused_key = service.discovery_for_context(**kwargs)

    assert reused_key == cache_key
    assert calls == [rows]
    assert reused["banks"]["treatment"]["queries"].tolist() == [[1.0, 0.0]]
    assert reused["banks"]["treatment"]["records"][0]["fit_standardized_score"] == 0.4


def test_context_cache_never_loads_entry_created_by_another_service_instance(
    tmp_path,
    monkeypatch,
):
    calls: list[tuple[int, ...]] = []
    _patch_discovery(monkeypatch, calls)
    rows = (0, 1, 2, 3)
    kwargs = {
        "outer_fold": 1,
        "context_row_ids": rows,
        "context_texts": _TEST_TEXTS[:4],
        "context_treatment": np.asarray([0.0, 1.0, 0.0, 1.0]),
        "context_outcome": np.asarray([0.0, 1.0, 1.0, 0.0]),
    }
    first = _service(tmp_path)
    first.discovery_for_context(**kwargs)
    second = _service(tmp_path)

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("an unowned executable checkpoint reached joblib.load")

    monkeypatch.setattr(query_context_module.joblib, "load", forbidden_load)
    with pytest.raises(ValueError, match="not held in trusted service memory"):
        second.discovery_for_context(**kwargs)


def test_gate_text_must_match_authenticated_projection(tmp_path, monkeypatch):
    calls: list[tuple[int, ...]] = []
    _patch_discovery(monkeypatch, calls)
    service = _service(tmp_path)
    backend = NeuralQueryContextBackend(service)
    with pytest.raises(ValueError, match="frozen embedding cache row"):
        backend.fit_predict(
            outer_fold=1,
            context_row_ids=(0, 1, 2, 3),
            context_texts=_TEST_TEXTS[:4],
            context_treatment=np.asarray([0.0, 1.0, 0.0, 1.0]),
            context_outcome=np.asarray([0.0, 1.0, 1.0, 0.0]),
            gate_row_ids=(4,),
            gate_texts=("different gate text",),
            work_dir=tmp_path,
        )


def test_review_sanitizer_retains_context_query_contrastive_ngrams():
    sanitized = AllEvidenceFusionRunner._sanitize_spent_evidence_catalog(
        [
            {
                "source_families": ["neural_query_moments"],
                "role_hint": "effect_modifier",
                "content": {
                    "kind": "neural_query_moment",
                    "bank": "effect",
                    "query_id": "effect_context_query_001",
                    "contrastive_ngrams": [{"term": "liver metastasis", "tfidf_contrast": 0.71}],
                    "fit_standardized_score": 0.42,
                },
            }
        ]
    )
    assert sanitized[0]["content"]["contrastive_ngrams"] == [
        {"term": "liver metastasis", "tfidf_contrast": 0.71}
    ]
    assert "query_id" not in sanitized[0]["content"]
    assert "query_id_sha256" in sanitized[0]["content"]
