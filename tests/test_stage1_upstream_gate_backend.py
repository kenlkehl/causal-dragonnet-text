from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.config import AppliedInferenceConfig
from oci.inference.all_evidence_post_extraction_review import ObservableCausalRows
from oci.inference.context_fit_upstream_gate_provider import (
    CompositeContextFitUpstreamBackend,
    ContextFitUpstreamGateProvider,
)
from oci.models.concept_embedding_utils import chunk_text_words
import oci.inference.stage1_upstream_gate_backend as module
from oci.inference.review_spent_evidence_provider import (
    SpentOnlyFrozenChunkEmbeddingCache,
)
from oci.inference.stage1_upstream_gate_backend import (
    ExactFrozenChunkEmbeddingProvider,
)


def _write_cache(path: Path, texts: tuple[str, ...]) -> tuple[str, ...]:
    path.mkdir(parents=True, exist_ok=True)
    rows = [tuple(chunk_text_words(text, 3, 1, 8, "last")) for text in texts]
    flattened = tuple(chunk for row in rows for chunk in row)
    embeddings = np.arange(len(flattened) * 4, dtype=np.float16).reshape(len(flattened), 4)
    offsets = np.asarray([0, *np.cumsum([len(row) for row in rows]).tolist()], dtype=np.int64)
    np.save(path / "chunk_embeddings.npy", embeddings)
    np.save(path / "offsets.npy", offsets)
    metadata = {
        "num_samples": len(texts),
        "hidden_size": 4,
        "chunk_size_words": 3,
        "chunk_overlap_words": 1,
        "max_chunks": 8,
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with (path / "chunk_texts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"chunks": list(row)}) + "\n")
    return flattened


def _fake_config_snapshot(path: Path, config: AppliedInferenceConfig) -> SimpleNamespace:
    forest = config.architecture.multi_model_forest
    forest.htr_evidence_enabled = True
    forest.matched_pair_uplift_enabled = True
    forest.matched_pair_htr_enabled = True
    return SimpleNamespace(
        source_path=path.resolve(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        applied_config=lambda: copy.deepcopy(config),
        verify_source=lambda: None,
    )


def _write_historical_config(path: Path, config: AppliedInferenceConfig) -> None:
    path.write_text(json.dumps({"config": asdict(config)}), encoding="utf-8")


class _FakeContextPredictionHTRProvider:
    def __init__(self, *, config, device, **_kwargs):
        self.config = config
        self.device = device
        self.seal_calls = 0

    def identity(self):
        return module.context_prediction_htr_provider_identity(
            self.config,
            device=self.device,
        )

    def seal_prediction_only_bundle(self, bundle):
        self.seal_calls += 1
        assert np.all(np.isfinite(bundle.x_test))
        assert np.all(np.isfinite(bundle.w_test))
        return SimpleNamespace(
            x_test=np.asarray(bundle.x_test, dtype=np.float32),
            w_test=np.asarray(bundle.w_test, dtype=np.float32),
            x_names=tuple(bundle.x_names),
            w_names=tuple(bundle.w_names),
            feature_rows=tuple(bundle.feature_rows),
        )


def test_historical_config_snapshot_hashes_and_parses_the_same_immutable_bytes(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "stage1.json"
    config = AppliedInferenceConfig()
    config.text_column = "snapshot_text"
    _write_historical_config(config_path, config)
    snapshot = module.HistoricalStage1ConfigSnapshot.from_path(config_path)

    assert snapshot.sha256 == hashlib.sha256(config_path.read_bytes()).hexdigest()
    first = snapshot.applied_config()
    first.text_column = "caller_mutation"
    assert snapshot.applied_config().text_column == "snapshot_text"

    config.text_column = "canonical_path_mutation"
    _write_historical_config(config_path, config)
    with pytest.raises(RuntimeError, match="config path changed"):
        snapshot.verify_source()


def test_historical_config_snapshot_rejects_post_read_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "stage1.json"
    config = AppliedInferenceConfig()
    _write_historical_config(config_path, config)
    replacement = AppliedInferenceConfig()
    replacement.text_column = "replacement_text"
    replacement_payload = json.dumps({"config": asdict(replacement)}).encode("utf-8")
    real_read_bytes = Path.read_bytes
    swapped = False

    def swapping_read_bytes(path: Path) -> bytes:
        nonlocal swapped
        payload = real_read_bytes(path)
        if path == config_path and not swapped:
            config_path.write_bytes(replacement_payload)
            swapped = True
        return payload

    monkeypatch.setattr(Path, "read_bytes", swapping_read_bytes)
    with pytest.raises(RuntimeError, match="changed"):
        module.HistoricalStage1ConfigSnapshot.from_path(config_path)
    assert swapped is True


def test_effective_stage1_config_digest_covers_all_fields_but_normalizes_private_path() -> None:
    left = AppliedInferenceConfig()
    right = copy.deepcopy(left)
    left.architecture.htr_sentence_model = "/tmp/private-snapshot-one"
    right.architecture.htr_sentence_model = "/tmp/private-snapshot-two"
    assert module._effective_applied_config_sha256(left) == (
        module._effective_applied_config_sha256(right)
    )

    right.training.learning_rate *= 2.0
    assert module._effective_applied_config_sha256(left) != (
        module._effective_applied_config_sha256(right)
    )


def test_private_htr_snapshot_isolated_from_source_and_detects_private_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "htr"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (source / "config.json").write_bytes(b'{"model":"fixed"}')
    (nested / "weights.bin").write_bytes(b"original weights")
    snapshot = module.PrivateHTRModelTreeSnapshot(source)

    (nested / "weights.bin").write_bytes(b"mutated source weights")
    snapshot.verify()
    private_weights = snapshot.path / "nested" / "weights.bin"
    assert private_weights.read_bytes() == b"original weights"

    private_weights.chmod(0o600)
    private_weights.write_bytes(b"mutated private weights")
    with pytest.raises(RuntimeError, match="private HTR model snapshot changed"):
        snapshot.verify()


def test_exact_frozen_chunk_provider_never_encodes_novel_text(tmp_path: Path) -> None:
    texts = ("one two three four five", "six seven eight nine")
    flattened = _write_cache(tmp_path, texts)
    provider = ExactFrozenChunkEmbeddingProvider(tmp_path, dataset_texts=texts)

    values = provider.encode_chunks(flattened)
    assert values.shape == (len(flattened), 4)
    assert provider.identity()["novel_text_encoding_allowed"] is False
    with pytest.raises(ValueError, match="refuses novel"):
        provider.encode_chunks((*flattened, "new text"))
    with pytest.raises(ValueError, match="refuses novel"):
        provider.encode_chunks(tuple(reversed(flattened)))


def test_exact_frozen_chunk_provider_binds_dataset_text_projection(tmp_path: Path) -> None:
    texts = ("one two three four five", "six seven eight nine")
    _write_cache(tmp_path, texts)
    with pytest.raises(ValueError, match="exact configured dataset projection"):
        ExactFrozenChunkEmbeddingProvider(
            tmp_path,
            dataset_texts=("one two three four changed", texts[1]),
        )


def test_exact_frozen_chunk_provider_rejects_embedding_shape_tamper(
    tmp_path: Path,
) -> None:
    texts = ("one two three four five", "six seven eight nine")
    _write_cache(tmp_path, texts)
    np.save(tmp_path / "offsets.npy", np.asarray([0, 1, 99], dtype=np.int64))
    with pytest.raises(ValueError, match="offsets do not span"):
        ExactFrozenChunkEmbeddingProvider(tmp_path, dataset_texts=texts)


def test_exact_frozen_chunk_provider_rejects_path_swap_during_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = ("one two three four five", "six seven eight nine")
    _write_cache(tmp_path, texts)
    target = tmp_path / "chunk_embeddings.npy"
    replacement = b"replacement after the authenticated read"
    real_read_bytes = Path.read_bytes
    swapped = False

    def swapping_read_bytes(path: Path) -> bytes:
        nonlocal swapped
        payload = real_read_bytes(path)
        if path == target and not swapped:
            target.write_bytes(replacement)
            swapped = True
        return payload

    monkeypatch.setattr(Path, "read_bytes", swapping_read_bytes)
    with pytest.raises(RuntimeError, match="changed while it was being authenticated"):
        ExactFrozenChunkEmbeddingProvider(tmp_path, dataset_texts=texts)
    assert swapped is True


@pytest.mark.parametrize(
    "filename",
    ["metadata.json", "chunk_embeddings.npy", "offsets.npy", "chunk_texts.jsonl"],
)
def test_exact_frozen_chunk_provider_detaches_every_authenticated_representation(
    tmp_path: Path,
    filename: str,
) -> None:
    texts = ("one two three four five", "six seven eight nine")
    flattened = _write_cache(tmp_path, texts)
    provider = ExactFrozenChunkEmbeddingProvider(tmp_path, dataset_texts=texts)
    identity = provider.identity()
    digest_fields = {
        "metadata.json": "metadata_sha256",
        "chunk_embeddings.npy": "embeddings_sha256",
        "offsets.npy": "offsets_sha256",
        "chunk_texts.jsonl": "chunk_texts_sha256",
    }
    assert (
        identity[digest_fields[filename]]
        == hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
    )
    assert identity["provider"] == "exact_frozen_chunk_embedding_provider_v2"
    assert identity["embeddings_path_backed"] is False
    assert not isinstance(provider._embeddings, np.memmap)
    assert provider._embeddings.flags.writeable is False
    assert provider._offsets.flags.writeable is False

    before_embeddings = provider.encode_chunks(flattened)
    before_matrices = provider.chunk_matrices((0, 1))
    before_chunks = provider.chunk_texts((0, 1))
    before_metadata = dict(provider.metadata)
    target = tmp_path / filename
    if filename == "metadata.json":
        target.write_text(json.dumps({"replaced": True}), encoding="utf-8")
    elif filename == "chunk_embeddings.npy":
        np.save(target, np.full_like(before_embeddings, 99.0, dtype=np.float16))
    elif filename == "offsets.npy":
        np.save(target, np.asarray([0, 1, len(flattened)], dtype=np.int64))
    else:
        target.write_text('{"chunks":["replacement"]}\n', encoding="utf-8")

    np.testing.assert_array_equal(provider.encode_chunks(flattened), before_embeddings)
    for actual, expected in zip(provider.chunk_matrices((0, 1)), before_matrices):
        np.testing.assert_array_equal(actual, expected)
    assert provider.chunk_texts((0, 1)) == before_chunks
    assert dict(provider.metadata) == before_metadata
    with pytest.raises(RuntimeError, match="path changed after authentication"):
        provider.identity()


def test_exact_frozen_chunk_provider_metadata_and_outputs_cannot_mutate_snapshot(
    tmp_path: Path,
) -> None:
    texts = ("one two three four five", "six seven eight nine")
    flattened = _write_cache(tmp_path, texts)
    provider = ExactFrozenChunkEmbeddingProvider(tmp_path, dataset_texts=texts)
    metadata = provider.metadata
    metadata["chunk_size_words"] = 999
    output = provider.encode_chunks(flattened)
    output[:] = -1
    assert provider.metadata["chunk_size_words"] == 3
    assert np.all(provider.encode_chunks(flattened) >= 0)


def test_historical_context_backend_defers_all_semantic_cache_text_to_fit_predict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = (
        "context zero alpha beta",
        "context one gamma delta",
        "sealed future must stay closed",
        "gate three epsilon zeta",
    )
    cache_dir = tmp_path / "embedding_cache"
    _write_cache(cache_dir, texts)
    # Deliberately not a parquet file.  Construction must authenticate from
    # cache metadata/bytes without opening the dataset text projection.
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"not parquet; future text must not be materialized")
    config_path = tmp_path / "stage1_config.json"
    config_path.write_text("{}", encoding="utf-8")
    htr_dir = tmp_path / "htr"
    htr_dir.mkdir()
    (htr_dir / "weights.bin").write_bytes(b"fixed-test-weights")

    config = AppliedInferenceConfig()
    snapshot = _fake_config_snapshot(config_path, config)
    monkeypatch.setattr(
        module,
        "_historical_stage1_config_snapshot",
        lambda _path, _snapshot=None: snapshot,
    )
    monkeypatch.setattr(module, "_resolve_htr_model_path", lambda _config: htr_dir)

    decoded_rows: list[int] = []
    original_cached_chunks = SpentOnlyFrozenChunkEmbeddingCache._cached_chunks

    def tracked_cached_chunks(self, row_id: int):
        decoded_rows.append(int(row_id))
        return original_cached_chunks(self, row_id)

    monkeypatch.setattr(
        SpentOnlyFrozenChunkEmbeddingCache,
        "_cached_chunks",
        tracked_cached_chunks,
    )
    seen: dict[str, object] = {}

    class FakeStage1Runner:
        def __init__(self, *, dataset, config, **_kwargs):
            assert dataset[config.text_column].tolist() == [""] * len(texts)
            self.dataset = dataset
            self.embedding_evidence_generator = None
            seen["runner"] = self
            seen["htr_provider"] = _kwargs["htr_evidence_provider"]

        def _build_feature_bundle(self, *, train_df, test_df, outer_fold):
            assert outer_fold == 7
            assert list(test_df.columns) == ["_oci_row_id", config.text_column]
            generator = self.embedding_evidence_generator
            assert generator is not None
            generator.prepare(self.dataset)
            train_rows = tuple(map(int, train_df["_oci_row_id"]))
            test_rows = tuple(map(int, test_df["_oci_row_id"]))
            assert generator._patient_embeddings(train_rows).shape == (len(train_rows), 4)
            assert generator._patient_embeddings(test_rows).shape == (len(test_rows), 4)
            with pytest.raises(ValueError, match="non-spent row"):
                generator.provider.chunk_matrix(2)
            seen["bound_provider"] = generator.provider
            n_train = len(train_rows)
            n_test = len(test_rows)
            return SimpleNamespace(
                x_train=np.column_stack(
                    [np.linspace(0.1, 0.2, n_train), np.linspace(1.1, 1.2, n_train)]
                ),
                x_test=np.column_stack(
                    [np.linspace(0.3, 0.4, n_test), np.linspace(1.3, 1.4, n_test)]
                ),
                w_train=np.empty((n_train, 0), dtype=float),
                w_test=np.empty((n_test, 0), dtype=float),
                x_names=("bow__weighted_r", "bow__modifier_basis"),
                w_names=(),
                feature_rows=(
                    {
                        "feature_name": "bow__weighted_r",
                        "source_family": "bow",
                        "objective": "direct_weighted_r",
                    },
                    {
                        "feature_name": "bow__modifier_basis",
                        "source_family": "bow",
                        "objective": "r_pseudo_outcome",
                    },
                ),
            )

    monkeypatch.setattr(module, "MultiModelForestStage1Runner", FakeStage1Runner)
    monkeypatch.setattr(
        module,
        "HistoricalStage1ContextPredictionHTRProvider",
        _FakeContextPredictionHTRProvider,
    )
    backend = module.HistoricalStage1ContextBackend(
        dataset_path=dataset_path,
        stage1_config_path=config_path,
        embedding_cache_dir=cache_dir,
        device="cpu",
        bow_fold_parallelism=3,
        required_families=("bow_weighted_r",),
    )

    assert decoded_rows == []
    assert not hasattr(backend, "_dataset_texts")
    identity = backend.identity()
    assert identity["dataset_text_read_or_hashed_at_construction"] is False
    assert identity["future_row_text_decoded_or_materialized"] is False
    assert identity["effective_config_schema_version"] == module.EFFECTIVE_STAGE1_CONFIG_ID
    assert identity["effective_config_sha256"] == backend.effective_config_sha256()
    assert backend.config.architecture.htr_require_live_unfrozen_encoder_attestation is True
    assert backend.config.architecture.multi_model_forest.bow_fold_parallelism == "3"
    assert backend.config.architecture.multi_model_forest.bow_parallel_backend == "threads"
    assert backend.config.architecture.multi_model_forest.htr_fold_parallelism == "1"
    assert backend.config.architecture.multi_model_forest.cpus_total == 3
    assert identity["bow_fold_parallelism"] == 3
    assert identity["bow_parallel_backend"] == "threads"
    assert identity["htr_fold_parallelism"] == 1
    assert identity["htr_runtime_source_attestation"] == (backend.htr_runtime_source_attestation())
    assert identity["context_prediction_htr_provider_required"] is True
    assert identity["context_prediction_htr_provider"]["configured_legacy_model_attempts"] == 20
    assert (
        identity["context_prediction_htr_provider"]["configured_context_prediction_model_attempts"]
        == 8
    )
    assert identity["context_train_pair_or_effect_predictions_consumed"] is False

    # Composite/provider construction calls member identities but must still
    # not decode any context, gate, or sealed semantic text.
    composite = CompositeContextFitUpstreamBackend((backend,))
    provider = ContextFitUpstreamGateProvider(tmp_path / "gate_cache", backend=composite)
    assert decoded_rows == []
    context = ObservableCausalRows(
        row_ids=(0, 1),
        extracted=pd.DataFrame({"placeholder": [0.0, 1.0]}),
        treatment=np.asarray([0.0, 1.0]),
        outcome=np.asarray([1.0, 0.0]),
        inner_fold_ids=(1, 2),
    )
    bound = provider.bind_fold(
        outer_fold=7,
        context=context,
        context_texts=(texts[0], texts[1]),
        gate_texts=(texts[3],),
        exact_gate_row_ids=(3,),
    )

    assert set(decoded_rows) == {0, 1, 3}
    assert 2 not in decoded_rows
    source = bound.get_gate_source_view(outer_fold=7, exact_gate_row_ids=(3,))
    features = bound.get_gate_feature_bank_view(outer_fold=7, exact_gate_row_ids=(3,))
    assert source.values.shape == (1, 1)
    assert features.values.shape == (1, 1)
    assert source.context_values.shape == (2, 1)
    assert features.context_values.shape == (2, 1)
    assert "bound_provider" in seen
    assert isinstance(seen["htr_provider"], _FakeContextPredictionHTRProvider)
    assert seen["htr_provider"].seal_calls == 1

    backend.config.training.learning_rate *= 2.0
    with pytest.raises(RuntimeError, match="effective Stage-1 runtime config changed"):
        backend.identity()


def test_historical_context_backend_fails_closed_on_wrong_explicit_gate_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = ("context exact words", "sealed exact words", "gate exact words")
    cache_dir = tmp_path / "embedding_cache"
    _write_cache(cache_dir, texts)
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"never parsed")
    config_path = tmp_path / "stage1_config.json"
    config_path.write_text("{}", encoding="utf-8")
    htr_dir = tmp_path / "htr"
    htr_dir.mkdir()
    (htr_dir / "weights.bin").write_bytes(b"weights")
    snapshot = _fake_config_snapshot(config_path, AppliedInferenceConfig())
    monkeypatch.setattr(
        module,
        "_historical_stage1_config_snapshot",
        lambda _path, _snapshot=None: snapshot,
    )
    monkeypatch.setattr(module, "_resolve_htr_model_path", lambda _config: htr_dir)

    class MustNotRun:
        def __init__(self, **_kwargs):
            raise AssertionError("Stage-1 runner must not start before text binding succeeds")

    monkeypatch.setattr(module, "MultiModelForestStage1Runner", MustNotRun)
    backend = module.HistoricalStage1ContextBackend(
        dataset_path=dataset_path,
        stage1_config_path=config_path,
        embedding_cache_dir=cache_dir,
        device="cpu",
        required_families=("bow_weighted_r",),
    )

    with pytest.raises(ValueError, match="does not match"):
        backend.fit_predict(
            outer_fold=1,
            context_row_ids=(0,),
            context_texts=(texts[0],),
            context_treatment=np.asarray([0.0]),
            context_outcome=np.asarray([1.0]),
            gate_row_ids=(2,),
            gate_texts=("gate altered beyond recognition",),
            work_dir=tmp_path / "work",
        )


def _fit_default_required_family_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_htr_neural: bool,
):
    texts = ("context alpha words", "context beta words", "gate gamma words")
    cache_dir = tmp_path / "embedding_cache"
    _write_cache(cache_dir, texts)
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"never parsed")
    config_path = tmp_path / "stage1_config.json"
    config_path.write_text("{}", encoding="utf-8")
    htr_dir = tmp_path / "htr"
    htr_dir.mkdir()
    (htr_dir / "weights.bin").write_bytes(b"fixed-test-weights")
    snapshot = _fake_config_snapshot(config_path, AppliedInferenceConfig())
    monkeypatch.setattr(
        module,
        "_historical_stage1_config_snapshot",
        lambda _path, _snapshot=None: snapshot,
    )
    monkeypatch.setattr(module, "_resolve_htr_model_path", lambda _config: htr_dir)

    class FakeStage1Runner:
        def __init__(self, **_kwargs):
            self.embedding_evidence_generator = None

        def _build_feature_bundle(self, *, train_df, test_df, outer_fold):
            assert outer_fold == 3
            x_specs = [
                ("bow__weighted_r", "bow", "direct_weighted_r", ""),
                ("htr__weighted_r", "htr", "direct_weighted_r", ""),
                ("pair__uplift", "bow_pair_uplift", "pair_uplift", ""),
                (
                    "embedding__whole",
                    "embedding_contrast",
                    "whole_cohort",
                    "whole_cohort",
                ),
                (
                    "embedding__cluster",
                    "embedding_contrast",
                    "cluster_contrast",
                    "clustered",
                ),
            ]
            if include_htr_neural:
                x_specs.append(("htr__effect_basis", "htr", "r_pseudo_outcome", ""))
            w_specs = [
                ("bow__treatment_nuisance", "bow", "treatment_nuisance", ""),
                ("htr__outcome_nuisance", "htr", "outcome_nuisance", ""),
            ]
            n_train = len(train_df)
            n_test = len(test_df)
            return SimpleNamespace(
                x_train=np.arange(n_train * len(x_specs), dtype=float).reshape(
                    n_train, len(x_specs)
                ),
                x_test=np.arange(n_test * len(x_specs), dtype=float).reshape(n_test, len(x_specs)),
                w_train=np.arange(n_train * len(w_specs), dtype=float).reshape(
                    n_train, len(w_specs)
                ),
                w_test=np.arange(n_test * len(w_specs), dtype=float).reshape(n_test, len(w_specs)),
                x_names=tuple(spec[0] for spec in x_specs),
                w_names=tuple(spec[0] for spec in w_specs),
                feature_rows=tuple(
                    {
                        "feature_name": name,
                        "source_family": family,
                        "objective": objective,
                        "contrast_family": contrast_family,
                    }
                    for name, family, objective, contrast_family in (*x_specs, *w_specs)
                ),
            )

    monkeypatch.setattr(module, "MultiModelForestStage1Runner", FakeStage1Runner)
    monkeypatch.setattr(
        module,
        "HistoricalStage1ContextPredictionHTRProvider",
        _FakeContextPredictionHTRProvider,
    )
    backend = module.HistoricalStage1ContextBackend(
        dataset_path=dataset_path,
        stage1_config_path=config_path,
        embedding_cache_dir=cache_dir,
        device="cpu",
    )
    prediction = backend.fit_predict(
        outer_fold=3,
        context_row_ids=(0, 1),
        context_texts=texts[:2],
        context_treatment=np.asarray([0.0, 1.0]),
        context_outcome=np.asarray([1.0, 0.0]),
        gate_row_ids=(2,),
        gate_texts=(texts[2],),
        work_dir=tmp_path / "work",
    )
    return backend, prediction


def test_historical_context_backend_default_requires_htr_neural_effect_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match=r"missing required upstream families: htr_neural"):
        _fit_default_required_family_bundle(
            tmp_path,
            monkeypatch,
            include_htr_neural=False,
        )


def test_historical_context_backend_default_accepts_present_htr_neural_effect_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, prediction = _fit_default_required_family_bundle(
        tmp_path,
        monkeypatch,
        include_htr_neural=True,
    )

    assert "htr_neural" in backend.identity()["required_families"]
    assert "htr_neural" in prediction.feature_kinds
