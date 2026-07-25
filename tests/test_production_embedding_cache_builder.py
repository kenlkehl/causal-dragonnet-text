from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import oci.inference.production_embedding_cache_builder as builder_module
from oci.inference.production_embedding_cache_builder import (
    PRODUCTION_EMBEDDING_CACHE_METADATA_SCHEMA,
    PRODUCTION_EMBEDDING_CACHE_PROVENANCE_SCHEMA,
    PRODUCTION_EMBEDDING_CACHE_RESULT_SCHEMA,
    build_production_embedding_cache,
    validate_published_production_embedding_cache,
)
from oci.inference.review_spent_evidence_provider import (
    SpentOnlyFrozenChunkEmbeddingCache,
)
from oci.inference.production_stage1_bundle import _validate_cache_configuration

_SENTENCE_MODEL_NAME = "sentence-transformers/test-logical-model"


class _FakeTokenizer:
    def __init__(self) -> None:
        self.audit_inputs: list[str] = []

    def __call__(
        self,
        inputs,
        *,
        add_special_tokens,
        truncation,
        padding,
        return_length,
    ):
        assert add_special_tokens is True
        assert truncation is False
        assert padding is False
        assert return_length is True
        self.audit_inputs.extend(inputs)
        return {"length": [len(value.split()) + 2 for value in inputs]}


class _FakeEncoder:
    def __init__(
        self,
        *,
        dimension: int = 5,
        max_seq_length: int = 64,
        expected_prompt_name: str | None = None,
        expected_prompt: str | None = "",
    ) -> None:
        self.dimension = dimension
        self.max_seq_length = max_seq_length
        self.tokenizer = _FakeTokenizer()
        self.default_prompt_name = None
        self.prompts = {}
        self.encoded_chunks: list[str] = []
        self.expected_prompt_name = expected_prompt_name
        self.expected_prompt = expected_prompt

    def encode(
        self,
        chunks,
        *,
        prompt_name,
        prompt,
        batch_size,
        output_value,
        precision,
        convert_to_numpy,
        convert_to_tensor,
        normalize_embeddings,
        truncate_dim,
        show_progress_bar,
        pool,
        chunk_size,
    ):
        assert prompt_name == self.expected_prompt_name
        assert prompt == self.expected_prompt
        assert batch_size == len(chunks)
        assert output_value == "sentence_embedding"
        assert precision == "float32"
        assert convert_to_numpy is True
        assert convert_to_tensor is False
        assert truncate_dim is None
        assert show_progress_bar is False
        assert pool is None
        assert chunk_size is None
        output = []
        for chunk in chunks:
            self.encoded_chunks.append(chunk)
            digest = hashlib.sha256(chunk.encode("utf-8")).digest()
            vector = np.asarray(
                [float(digest[index] + 1) for index in range(self.dimension)],
                dtype=np.float32,
            )
            if normalize_embeddings:
                vector /= np.linalg.norm(vector)
            output.append(vector)
        return np.asarray(output, dtype=np.float32)


def _write_inputs(tmp_path: Path):
    dataset = tmp_path / "cohort.parquet"
    texts = (
        "zero one two three four five",
        "alpha beta gamma delta epsilon zeta",
        "short clinical row",
    )
    pd.DataFrame(
        {
            "clinical_text": texts,
            "treatment": [0, 1, 0],
            "outcome": [1, 0, 1],
        }
    ).to_parquet(dataset, index=False)
    model = tmp_path / "local-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"safe-test"}\n', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"safe-local-weights")
    return dataset, model, texts


def _chunk_configuration() -> dict[str, object]:
    return {
        "chunk_size_words": 3,
        "chunk_overlap_words": 1,
        "max_chunks": 3,
        "chunk_selection": "last",
        "normalize_embeddings": True,
        "max_seq_length": 16,
        "prompt_policy": "disabled",
        "prompt_name": None,
        "output_value": "sentence_embedding",
        "precision": "float32",
        "convert_to_numpy": True,
        "convert_to_tensor": False,
        "truncate_dim": None,
        "pooling_output_policy": "single_process_sentence_embedding_v1",
        "model_dtype": "float32",
        "stored_array_dtype": "float32",
        "zero_vector_policy": "reject",
    }


def _install_fake_encoder(monkeypatch, encoder=None, *, callback=None):
    encoder = _FakeEncoder() if encoder is None else encoder
    observed = {}

    def load(*, model_path, device, max_seq_length, model_dtype):
        observed.update(
            {
                "model_path": model_path,
                "device": device,
                "max_seq_length": max_seq_length,
                "model_dtype": model_dtype,
                "offline_environment": {
                    key: os.environ.get(key) for key in builder_module._OFFLINE_ENVIRONMENT
                },
            }
        )
        with pytest.raises(RuntimeError, match="network access is forbidden"):
            socket.getaddrinfo("example.com", 443)
        if callback is not None:
            callback()
        return encoder

    monkeypatch.setattr(builder_module, "_load_local_sentence_encoder", load)
    return encoder, observed


def _assert_no_partial_build(tmp_path: Path, target: Path) -> None:
    assert not target.exists()
    assert not target.is_symlink()
    assert not list(tmp_path.glob(f".{target.name}.building-*"))


def _build_cache_for_validation(tmp_path: Path, monkeypatch):
    dataset, model, texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    target = tmp_path / "published-cache"
    result = build_production_embedding_cache(
        dataset_path=dataset,
        text_column="clinical_text",
        local_model_path=model,
        sentence_model_name=_SENTENCE_MODEL_NAME,
        chunk_configuration=_chunk_configuration(),
        target_dir=target,
        device="cpu",
        batch_size=2,
    )
    return dataset, model, texts, target, result.identity()


def _rewrite_cache_metadata(target: Path, mutate) -> None:
    path = target / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    mutate(metadata)
    metadata["production_provenance_sha256"] = builder_module._sha256_json(
        metadata["production_provenance"]
    )
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_builds_exact_offline_atomic_cache_and_returns_closed_identity(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, texts = _write_inputs(tmp_path)
    encoder, observed = _install_fake_encoder(monkeypatch)
    target = tmp_path / "published-cache"
    previous_offline = os.environ.get("HF_HUB_OFFLINE")

    result = build_production_embedding_cache(
        dataset_path=dataset,
        text_column="clinical_text",
        local_model_path=model,
        sentence_model_name=_SENTENCE_MODEL_NAME,
        chunk_configuration=_chunk_configuration(),
        target_dir=target,
        device="cpu",
        batch_size=2,
    )

    assert result.cache_path == target.resolve()
    assert set(path.name for path in target.iterdir()) == {
        "metadata.json",
        "chunk_embeddings.npy",
        "offsets.npy",
        "chunk_texts.jsonl",
    }
    assert not list(tmp_path.glob(".published-cache.building-*"))
    assert observed["model_path"] == model.resolve()
    assert observed["device"] == "cpu"
    assert observed["max_seq_length"] == 16
    assert observed["model_dtype"] == "float32"
    assert set(observed["offline_environment"].values()) == {"1"}
    assert os.environ.get("HF_HUB_OFFLINE") == previous_offline
    assert encoder.encoded_chunks[:2] == ["zero one two", "two three four"]

    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == PRODUCTION_EMBEDDING_CACHE_METADATA_SCHEMA
    assert metadata["sentence_model_name"] == _SENTENCE_MODEL_NAME
    assert metadata["uncapped_total_chunks"] == metadata["total_chunks"]
    assert metadata["uncapped_chunk_counts_sha256"] == builder_module._sha256_json(
        metadata["chunk_counts"]
    )
    assert metadata["chunk_cap_nonbinding"] is True
    assert metadata["semantic_truncation_allowed"] is False
    ordered_token_counts = tuple(len(value.split()) + 2 for value in encoder.tokenizer.audit_inputs)
    assert metadata["max_observed_token_count"] == max(ordered_token_counts)
    assert metadata["ordered_token_counts_sha256"] == builder_module._sha256_json(
        ordered_token_counts
    )
    assert metadata["tokenizer_truncation_allowed"] is False
    assert metadata["chunking_mode"] == (
        "whitespace_word_chunks_tokenizer_verified_nontruncating_v3"
    )
    provenance = metadata["production_provenance"]
    assert provenance["schema_version"] == PRODUCTION_EMBEDDING_CACHE_PROVENANCE_SCHEMA
    assert len(provenance["builder_code_sha256"]) == 64
    assert provenance["sentence_model_name"] == _SENTENCE_MODEL_NAME
    assert provenance["local_model"]["path"] == str(model.resolve())
    assert provenance["local_model"]["path"] != provenance["sentence_model_name"]
    assert provenance["dataset"]["row_count"] == len(texts)
    assert provenance["chunk_configuration"] == _chunk_configuration()
    assert provenance["network_access_allowed"] is False
    assert provenance["partial_cache_reuse_allowed"] is False
    assert provenance["uncapped_total_chunks"] == metadata["total_chunks"]
    assert provenance["uncapped_chunk_counts_sha256"] == metadata["uncapped_chunk_counts_sha256"]
    assert provenance["chunk_cap_nonbinding"] is True
    assert provenance["semantic_truncation_allowed"] is False
    assert provenance["max_observed_token_count"] == metadata["max_observed_token_count"]
    assert provenance["ordered_token_counts_sha256"] == metadata["ordered_token_counts_sha256"]
    assert provenance["tokenizer_truncation_allowed"] is False
    for name, registration in provenance["companion_cache_files"].items():
        assert registration["sha256"] == hashlib.sha256((target / name).read_bytes()).hexdigest()

    identity = result.identity()
    assert identity["schema_version"] == PRODUCTION_EMBEDDING_CACHE_RESULT_SCHEMA
    assert identity["builder_code_sha256"] == provenance["builder_code_sha256"]
    assert identity["sentence_model_name"] == _SENTENCE_MODEL_NAME
    assert identity["cache_configuration_sha256"] == provenance["cache_configuration_sha256"]
    assert set(identity["cache_files"]) == {
        "metadata.json",
        "chunk_embeddings.npy",
        "offsets.npy",
        "chunk_texts.jsonl",
    }
    for name, registration in identity["cache_files"].items():
        assert registration["sha256"] == hashlib.sha256((target / name).read_bytes()).hexdigest()
    cache = SpentOnlyFrozenChunkEmbeddingCache(result.cache_path)
    bound = cache.bind_spent(tuple(range(len(texts))), texts)
    assert tuple(bound.row_ids) == tuple(range(len(texts)))
    assert identity["provider_identity"] == cache.identity()
    wrapper_config = SimpleNamespace(
        architecture=SimpleNamespace(
            multi_model_forest=SimpleNamespace(
                embedding_contrast=SimpleNamespace(
                    model_name=_SENTENCE_MODEL_NAME,
                    chunk_size_words=3,
                    chunk_overlap_words=1,
                    max_chunks=3,
                    chunk_selection="last",
                    normalize_embeddings=True,
                    max_seq_length=16,
                )
            )
        )
    )
    _validate_cache_configuration(cache, wrapper_config)


def test_disabled_prompt_policy_overrides_hidden_model_default_prompt(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    encoder = _FakeEncoder()
    encoder.default_prompt_name = "hidden-default"
    encoder.prompts = {"hidden-default": "HIDDEN PREFIX "}
    _install_fake_encoder(monkeypatch, encoder=encoder)

    target = tmp_path / "cache"
    build_production_embedding_cache(
        dataset_path=dataset,
        text_column="clinical_text",
        local_model_path=model,
        sentence_model_name=_SENTENCE_MODEL_NAME,
        chunk_configuration=_chunk_configuration(),
        target_dir=target,
        batch_size=2,
    )

    assert encoder.tokenizer.audit_inputs[0] == "zero one two"
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["prompt_policy"] == "disabled"
    assert metadata["prompt_name"] is None
    assert metadata["resolved_prompt_length"] == 0
    assert metadata["resolved_prompt_sha256"] == hashlib.sha256(b"").hexdigest()


def test_authenticated_named_prompt_is_exactly_audited_encoded_and_sealed(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    encoder = _FakeEncoder(
        expected_prompt_name="query",
        expected_prompt=None,
    )
    encoder.prompts = {"query": "Represent this clinical note: "}
    _install_fake_encoder(monkeypatch, encoder=encoder)
    configuration = _chunk_configuration()
    configuration.update(
        {
            "prompt_policy": "authenticated_model_prompt_name",
            "prompt_name": "query",
        }
    )

    target = tmp_path / "cache"
    build_production_embedding_cache(
        dataset_path=dataset,
        text_column="clinical_text",
        local_model_path=model,
        sentence_model_name=_SENTENCE_MODEL_NAME,
        chunk_configuration=configuration,
        target_dir=target,
        batch_size=2,
    )

    prefix = encoder.prompts["query"]
    assert encoder.tokenizer.audit_inputs[0] == prefix + "zero one two"
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["resolved_prompt_length"] == len(prefix)
    assert metadata["resolved_prompt_sha256"] == hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def test_device_and_batch_are_operational_not_cache_scientific_identity(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    configuration = _chunk_configuration()

    first = build_production_embedding_cache(
        dataset_path=dataset,
        text_column="clinical_text",
        local_model_path=model,
        sentence_model_name=_SENTENCE_MODEL_NAME,
        chunk_configuration=configuration,
        target_dir=tmp_path / "cache-a",
        device=None,
        batch_size=1,
    ).identity()
    second = build_production_embedding_cache(
        dataset_path=dataset,
        text_column="clinical_text",
        local_model_path=model,
        sentence_model_name=_SENTENCE_MODEL_NAME,
        chunk_configuration=configuration,
        target_dir=tmp_path / "cache-b",
        device="cpu",
        batch_size=3,
    ).identity()

    assert first["cache_configuration_sha256"] == second["cache_configuration_sha256"]
    assert (
        first["cache_files"]["chunk_embeddings.npy"]["sha256"]
        == second["cache_files"]["chunk_embeddings.npy"]["sha256"]
    )
    first_metadata = json.loads(
        (tmp_path / "cache-a" / "metadata.json").read_text(encoding="utf-8")
    )
    second_metadata = json.loads(
        (tmp_path / "cache-b" / "metadata.json").read_text(encoding="utf-8")
    )
    assert first_metadata["production_provenance"]["encoder_execution"]["batch_size"] == 1
    assert second_metadata["production_provenance"]["encoder_execution"]["batch_size"] == 3


@pytest.mark.parametrize("field_name", sorted(builder_module._CHUNK_CONFIG_FIELDS))
def test_cache_configuration_requires_every_encoder_and_chunk_setting(
    tmp_path: Path,
    monkeypatch,
    field_name: str,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    configuration = _chunk_configuration()
    configuration.pop(field_name)
    with pytest.raises(ValueError, match="closed|chunk_configuration"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=configuration,
            target_dir=tmp_path / "cache",
        )


def test_cache_configuration_rejects_extra_encoder_setting(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    configuration = {**_chunk_configuration(), "implicit_prompt_fallback": True}
    with pytest.raises(ValueError, match="closed|chunk_configuration"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=configuration,
            target_dir=tmp_path / "cache",
        )


def test_model_declared_hidden_truncate_dimension_is_rejected(
    tmp_path: Path,
    monkeypatch,
):
    import sentence_transformers

    model = tmp_path / "model"
    model.mkdir()
    observed = {}

    class HiddenTruncationEncoder:
        truncate_dim = 64

    def construct(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return HiddenTruncationEncoder()

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", construct)
    with pytest.raises(ValueError, match="hidden truncate_dim"):
        builder_module._load_local_sentence_encoder(
            model_path=model,
            device="cpu",
            max_seq_length=128,
            model_dtype="float32",
        )
    assert observed["kwargs"]["truncate_dim"] is None


def test_validates_published_cache_without_network_or_model_load_and_matches_fresh_identity(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, _texts, target, fresh_identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("published-cache validation attempted network or model loading")

    monkeypatch.setattr(builder_module, "_load_local_sentence_encoder", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)

    validated = validate_published_production_embedding_cache(
        cache_dir=target,
        dataset_path=dataset,
        text_column="clinical_text",
        sentence_model_name=_SENTENCE_MODEL_NAME,
        chunk_configuration=_chunk_configuration(),
        expected_local_model_path=model,
    )
    assert validated == fresh_identity


def test_rejects_semantically_truncating_chunk_cap_before_model_authentication(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    configuration = _chunk_configuration()
    configuration["max_chunks"] = 2
    target = tmp_path / "published-cache"
    model_authenticated = False

    def forbidden_model_snapshot(_path):
        nonlocal model_authenticated
        model_authenticated = True
        raise AssertionError("model authentication must follow chunk-loss preflight")

    monkeypatch.setattr(builder_module, "_model_tree_snapshot", forbidden_model_snapshot)
    with pytest.raises(ValueError, match="semantic truncation"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=configuration,
            target_dir=target,
            device="cpu",
            batch_size=1,
        )
    assert model_authenticated is False
    _assert_no_partial_build(tmp_path, target)


def test_rejects_tokenizer_level_semantic_truncation_before_first_embedding(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, _texts = _write_inputs(tmp_path)

    class OverflowTokenizer(_FakeTokenizer):
        def __call__(self, inputs, **kwargs):
            super().__call__(inputs, **kwargs)
            return {"length": [17 for _value in inputs]}

    encoder = _FakeEncoder(max_seq_length=16)
    encoder.tokenizer = OverflowTokenizer()
    _install_fake_encoder(monkeypatch, encoder=encoder)
    target = tmp_path / "published-cache"

    with pytest.raises(ValueError, match="tokenizer would cause semantic truncation"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
            device="cpu",
            batch_size=1,
        )
    assert encoder.encoded_chunks == []
    _assert_no_partial_build(tmp_path, target)


@pytest.mark.parametrize(
    ("policy", "succeeds"),
    (("reject", False), ("preserve", True)),
)
def test_zero_vector_policy_is_explicit_and_sealed(
    tmp_path: Path,
    monkeypatch,
    policy: str,
    succeeds: bool,
):
    dataset, model, _texts = _write_inputs(tmp_path)

    class ZeroEncoder(_FakeEncoder):
        def encode(self, chunks, **kwargs):
            values = super().encode(chunks, **kwargs)
            values[0] = 0.0
            return values

    _install_fake_encoder(monkeypatch, encoder=ZeroEncoder())
    configuration = _chunk_configuration()
    configuration["zero_vector_policy"] = policy
    target = tmp_path / f"cache-{policy}"
    if not succeeds:
        with pytest.raises(ValueError, match="zero vector"):
            build_production_embedding_cache(
                dataset_path=dataset,
                text_column="clinical_text",
                local_model_path=model,
                sentence_model_name=_SENTENCE_MODEL_NAME,
                chunk_configuration=configuration,
                target_dir=target,
            )
        _assert_no_partial_build(tmp_path, target)
        return

    build_production_embedding_cache(
        dataset_path=dataset,
        text_column="clinical_text",
        local_model_path=model,
        sentence_model_name=_SENTENCE_MODEL_NAME,
        chunk_configuration=configuration,
        target_dir=target,
    )
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["zero_vector_policy"] == "preserve"
    assert metadata["zero_vector_count"] > 0


def test_read_only_validator_rejects_a_cap_that_would_truncate_supplied_source(
    tmp_path: Path,
    monkeypatch,
):
    dataset, _model, _texts, target, fresh_identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )
    longer_dataset = tmp_path / "longer-cohort.parquet"
    pd.DataFrame(
        {
            "clinical_text": [" ".join(f"word-{index}" for index in range(9))],
        }
    ).to_parquet(longer_dataset, index=False)
    with pytest.raises(ValueError, match="semantic truncation"):
        validate_published_production_embedding_cache(
            cache_dir=target,
            dataset_path=longer_dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
        )
    validated = validate_published_production_embedding_cache(
        cache_dir=target,
        dataset_path=dataset,
        text_column="clinical_text",
        sentence_model_name=_SENTENCE_MODEL_NAME,
        chunk_configuration=_chunk_configuration(),
    )
    assert validated == fresh_identity
    hash_fields = {
        "metadata.json": "metadata_sha256",
        "chunk_embeddings.npy": "embeddings_sha256",
        "offsets.npy": "offsets_sha256",
        "chunk_texts.jsonl": "chunk_texts_sha256",
    }
    for name, hash_field in hash_fields.items():
        registration = validated["cache_files"][name]
        assert registration["size_bytes"] == (target / name).stat().st_size
        assert registration["sha256"] == validated["provider_identity"][hash_field]

    validated["cache_files"].clear()
    assert (
        validate_published_production_embedding_cache(
            cache_dir=target,
            dataset_path=dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
        )
        == fresh_identity
    )


def test_published_validator_rejects_rehashed_forged_provenance(
    tmp_path: Path,
    monkeypatch,
):
    dataset, _model, _texts, target, _identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )

    def forge(metadata):
        metadata["production_provenance"]["dataset"]["path"] = str(
            tmp_path / "forged-cohort.parquet"
        )

    _rewrite_cache_metadata(target, forge)
    with pytest.raises(ValueError, match="supplied cohort"):
        validate_published_production_embedding_cache(
            cache_dir=target,
            dataset_path=dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
        )


@pytest.mark.parametrize("drift_kind", ("dataset_bytes", "ordered_text"))
def test_published_validator_rejects_dataset_and_text_drift(
    tmp_path: Path,
    monkeypatch,
    drift_kind: str,
):
    dataset, _model, texts, target, _identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )
    changed_texts = list(texts)
    if drift_kind == "ordered_text":
        changed_texts[0] = "entirely different clinical evidence"
    pd.DataFrame(
        {
            "clinical_text": changed_texts,
            "treatment": [1, 0, 1],
            "outcome": [0, 1, 0],
        }
    ).to_parquet(dataset, index=False)
    expected_message = (
        "exact uncapped source projection" if drift_kind == "ordered_text" else "supplied cohort"
    )
    with pytest.raises(ValueError, match=expected_message):
        validate_published_production_embedding_cache(
            cache_dir=target,
            dataset_path=dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
        )


def test_published_validator_rejects_chunk_configuration_drift(
    tmp_path: Path,
    monkeypatch,
):
    dataset, _model, _texts, target, _identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )
    changed = _chunk_configuration()
    changed["max_chunks"] = 4
    with pytest.raises(ValueError, match="authenticated policy"):
        validate_published_production_embedding_cache(
            cache_dir=target,
            dataset_path=dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=changed,
        )


def test_published_validator_rejects_encoder_output_configuration_drift(
    tmp_path: Path,
    monkeypatch,
):
    dataset, _model, _texts, target, _identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )
    changed = _chunk_configuration()
    changed["model_dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="authenticated policy"):
        validate_published_production_embedding_cache(
            cache_dir=target,
            dataset_path=dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=changed,
        )


def test_published_validator_rejects_cache_file_drift(
    tmp_path: Path,
    monkeypatch,
):
    dataset, _model, _texts, target, _identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )
    embeddings = target / "chunk_embeddings.npy"
    embeddings.write_bytes(embeddings.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="authenticated policy"):
        validate_published_production_embedding_cache(
            cache_dir=target,
            dataset_path=dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
        )


def test_published_validator_rejects_rehashed_nonbuilder_embedding_dtype(
    tmp_path: Path,
    monkeypatch,
):
    dataset, _model, _texts, target, _identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )
    embeddings_path = target / "chunk_embeddings.npy"
    embeddings = np.load(embeddings_path, allow_pickle=False)
    np.save(embeddings_path, embeddings.astype(np.float64), allow_pickle=False)

    def rebind_companion(metadata):
        payload = embeddings_path.read_bytes()
        metadata["production_provenance"]["companion_cache_files"]["chunk_embeddings.npy"] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    _rewrite_cache_metadata(target, rebind_companion)
    with pytest.raises(ValueError, match="arrays do not match"):
        validate_published_production_embedding_cache(
            cache_dir=target,
            dataset_path=dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
        )


@pytest.mark.parametrize("link_kind", ("root", "ancestor"))
def test_published_validator_rejects_symlinked_cache_paths(
    tmp_path: Path,
    monkeypatch,
    link_kind: str,
):
    dataset, _model, _texts, target, _identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )
    if link_kind == "root":
        supplied_cache = tmp_path / "published-cache-link"
        supplied_cache.symlink_to(target, target_is_directory=True)
    else:
        ancestor = tmp_path / "cache-parent-link"
        ancestor.symlink_to(tmp_path, target_is_directory=True)
        supplied_cache = ancestor / target.name
    with pytest.raises(ValueError, match="real directory|symlinked"):
        validate_published_production_embedding_cache(
            cache_dir=supplied_cache,
            dataset_path=dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
        )


def test_published_validator_rejects_exact_byte_cache_root_replacement(
    tmp_path: Path,
    monkeypatch,
):
    dataset, _model, _texts, target, _identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )
    original = builder_module._validate_cache_content

    def replace_root(root, **kwargs):
        validated = original(root, **kwargs)
        displaced = tmp_path / "displaced-cache-root"
        root.rename(displaced)
        root.mkdir()
        for source in displaced.iterdir():
            (root / source.name).write_bytes(source.read_bytes())
        return validated

    monkeypatch.setattr(builder_module, "_validate_cache_content", replace_root)
    with pytest.raises(RuntimeError, match="cache root changed"):
        validate_published_production_embedding_cache(
            cache_dir=target,
            dataset_path=dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
        )


def test_published_validator_rejects_current_builder_code_drift(
    tmp_path: Path,
    monkeypatch,
):
    dataset, _model, _texts, target, _identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(builder_module, "_builder_code_sha256", lambda: "0" * 64)
    with pytest.raises(ValueError, match="authenticated policy"):
        validate_published_production_embedding_cache(
            cache_dir=target,
            dataset_path=dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
        )


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    (
        ("storage_format", "forged_storage"),
        ("dtype", "float64"),
        ("chunking_mode", "forged_chunking"),
        ("effective_max_seq_length", 0),
        ("max_observed_token_count", 17),
        ("ordered_token_counts_sha256", "0" * 64),
        ("tokenizer_truncation_allowed", True),
        ("actual_max_len", 1),
        ("hidden_size", "5"),
        ("num_samples", 3.0),
        ("total_chunks", 5.0),
        ("chunk_counts", [2.0, 2, 1]),
        ("normalize_embeddings", 1),
    ),
)
def test_published_validator_rejects_rehashed_metadata_policy_drift(
    tmp_path: Path,
    monkeypatch,
    field_name: str,
    forged_value,
):
    dataset, _model, _texts, target, _identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )
    _rewrite_cache_metadata(
        target,
        lambda metadata: metadata.__setitem__(field_name, forged_value),
    )
    with pytest.raises(ValueError, match="authenticated policy|invalid dimensions"):
        validate_published_production_embedding_cache(
            cache_dir=target,
            dataset_path=dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
        )


@pytest.mark.parametrize("drift_kind", ("path", "tree_bytes"))
def test_published_validator_rejects_optional_local_model_drift(
    tmp_path: Path,
    monkeypatch,
    drift_kind: str,
):
    dataset, model, _texts, target, _identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )
    expected_model = model
    if drift_kind == "path":
        expected_model = tmp_path / "same-bytes-different-model-path"
        expected_model.mkdir()
        for source in model.iterdir():
            (expected_model / source.name).write_bytes(source.read_bytes())
    else:
        (model / "config.json").write_text('{"model_type":"drifted"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="supplied local model"):
        validate_published_production_embedding_cache(
            cache_dir=target,
            dataset_path=dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            expected_local_model_path=expected_model,
        )


@pytest.mark.parametrize("drift_kind", ("hash", "open_schema", "policy"))
def test_published_validator_cross_checks_closed_provider_identity(
    tmp_path: Path,
    monkeypatch,
    drift_kind: str,
):
    dataset, _model, _texts, target, _identity = _build_cache_for_validation(
        tmp_path,
        monkeypatch,
    )
    original = SpentOnlyFrozenChunkEmbeddingCache.identity

    def forged_provider_identity(cache):
        identity = original(cache)
        if drift_kind == "hash":
            identity["metadata_sha256"] = "0" * 64
        elif drift_kind == "open_schema":
            identity["untrusted_extra_field"] = True
        else:
            identity["novel_text_encoding_allowed"] = True
        return identity

    monkeypatch.setattr(
        SpentOnlyFrozenChunkEmbeddingCache,
        "identity",
        forged_provider_identity,
    )
    with pytest.raises(ValueError, match="provider"):
        validate_published_production_embedding_cache(
            cache_dir=target,
            dataset_path=dataset,
            text_column="clinical_text",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
        )


@pytest.mark.parametrize(
    "invalid_name",
    (None, "", "   ", " leading-space", "trailing-space ", "line\nbreak", 17),
)
def test_requires_nonempty_exact_logical_sentence_model_name(
    tmp_path: Path,
    monkeypatch,
    invalid_name,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    target = tmp_path / "cache"
    with pytest.raises(ValueError, match="sentence_model_name"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=invalid_name,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
        )
    _assert_no_partial_build(tmp_path, target)


@pytest.mark.parametrize("mismatch_location", ("metadata", "provenance"))
def test_rejects_logical_sentence_model_name_mismatch_and_cleans_target(
    tmp_path: Path,
    monkeypatch,
    mismatch_location: str,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    original = builder_module._write_json_new

    def write_mismatch(path, value):
        changed = copy.deepcopy(dict(value))
        if path.name == "metadata.json":
            if mismatch_location == "metadata":
                changed["sentence_model_name"] = "sentence-transformers/another-model"
            else:
                provenance = changed["production_provenance"]
                provenance["sentence_model_name"] = "sentence-transformers/another-model"
                provenance["cache_configuration_sha256"] = (
                    builder_module._cache_configuration_sha256(
                        sentence_model_name=provenance["sentence_model_name"],
                        chunk_configuration=provenance["chunk_configuration"],
                    )
                )
                changed["production_provenance_sha256"] = builder_module._sha256_json(provenance)
        original(path, changed)

    monkeypatch.setattr(builder_module, "_write_json_new", write_mismatch)
    target = tmp_path / "cache"
    with pytest.raises(ValueError, match="authenticated policy"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
        )
    _assert_no_partial_build(tmp_path, target)


def test_dataset_drift_fails_and_cleans_atomic_target(tmp_path: Path, monkeypatch):
    dataset, model, _texts = _write_inputs(tmp_path)

    def drift():
        dataset.write_bytes(dataset.read_bytes() + b"drift")

    _install_fake_encoder(monkeypatch, callback=drift)
    target = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="dataset changed"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
        )
    _assert_no_partial_build(tmp_path, target)


def test_local_model_drift_fails_and_cleans_atomic_target(tmp_path: Path, monkeypatch):
    dataset, model, _texts = _write_inputs(tmp_path)

    def drift():
        (model / "config.json").write_text('{"model_type":"changed"}\n', encoding="utf-8")

    _install_fake_encoder(monkeypatch, callback=drift)
    target = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="model tree changed"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
        )
    _assert_no_partial_build(tmp_path, target)


def test_chunk_configuration_drift_fails_and_cleans_target(tmp_path: Path, monkeypatch):
    dataset, model, _texts = _write_inputs(tmp_path)
    configuration = _chunk_configuration()

    def drift():
        configuration["max_chunks"] = 4

    _install_fake_encoder(monkeypatch, callback=drift)
    target = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="configuration changed"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=configuration,
            target_dir=target,
        )
    _assert_no_partial_build(tmp_path, target)


@pytest.mark.parametrize("selection", (None, "middle"))
def test_requires_explicit_supported_chunk_selection(
    tmp_path: Path,
    monkeypatch,
    selection: str | None,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    configuration = _chunk_configuration()
    if selection is None:
        del configuration["chunk_selection"]
    else:
        configuration["chunk_selection"] = selection
    with pytest.raises(ValueError, match="chunk_selection"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=configuration,
            target_dir=tmp_path / "cache",
        )


def test_configured_first_chunk_selection_is_supported_when_cap_is_nonbinding(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    configuration = _chunk_configuration()
    configuration["chunk_selection"] = "first"
    target = tmp_path / "cache"

    build_production_embedding_cache(
        dataset_path=dataset,
        text_column="clinical_text",
        local_model_path=model,
        sentence_model_name=_SENTENCE_MODEL_NAME,
        chunk_configuration=configuration,
        target_dir=target,
    )

    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["chunk_selection"] == "first"
    assert metadata["chunk_cap_nonbinding"] is True
    assert metadata["semantic_truncation_allowed"] is False


def test_rejects_missing_local_model(tmp_path: Path, monkeypatch):
    dataset, _model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    with pytest.raises(ValueError, match="local model path"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=tmp_path / "missing-model",
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=tmp_path / "cache",
        )


def test_rejects_model_tree_symlink(tmp_path: Path, monkeypatch):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    (model / "linked-config.json").symlink_to(model / "config.json")
    with pytest.raises(ValueError, match="real regular file|linked"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=tmp_path / "cache",
        )


@pytest.mark.parametrize("executable_kind", ("suffix", "mode"))
def test_rejects_executable_model_artifacts(
    tmp_path: Path,
    monkeypatch,
    executable_kind: str,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    if executable_kind == "suffix":
        (model / "remote_model.py").write_text("raise RuntimeError\n", encoding="utf-8")
    else:
        executable = model / "launch.txt"
        executable.write_text("unsafe\n", encoding="utf-8")
        executable.chmod(0o700)
    with pytest.raises(ValueError, match="executable"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=tmp_path / "cache",
        )


def test_rejects_fresh_target_symlink_without_touching_referent(tmp_path: Path, monkeypatch):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    referent = tmp_path / "referent"
    referent.mkdir()
    target = tmp_path / "cache"
    target.symlink_to(referent, target_is_directory=True)
    with pytest.raises(FileExistsError, match="fresh"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
        )
    assert referent.is_dir()
    assert target.is_symlink()


def test_rejects_and_preserves_partial_existing_target(tmp_path: Path, monkeypatch):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    target = tmp_path / "cache"
    target.mkdir()
    marker = target / "partial.bin"
    marker.write_bytes(b"must remain")
    with pytest.raises(FileExistsError, match="fresh"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
        )
    assert marker.read_bytes() == b"must remain"


def test_rejects_extra_generated_cache_file_and_cleans_target(tmp_path: Path, monkeypatch):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    original = builder_module._encode_chunks

    def with_extra(**kwargs):
        result = original(**kwargs)
        (kwargs["output_path"].parent / "unregistered.txt").write_text("extra", encoding="utf-8")
        return result

    monkeypatch.setattr(builder_module, "_encode_chunks", with_extra)
    target = tmp_path / "cache"
    with pytest.raises(ValueError, match="exactly its four"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
        )
    _assert_no_partial_build(tmp_path, target)


def test_rejects_executable_generated_cache_file_and_cleans_target(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    original = builder_module._write_chunk_registry

    def executable_registry(path, sample_chunks):
        original(path, sample_chunks)
        path.chmod(0o700)

    monkeypatch.setattr(builder_module, "_write_chunk_registry", executable_registry)
    target = tmp_path / "cache"
    with pytest.raises(ValueError, match="executable"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
        )
    _assert_no_partial_build(tmp_path, target)


def test_all_row_reload_rejects_row_order_text_mismatch_and_cleans_target(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    original = builder_module._write_chunk_registry

    def reversed_registry(path, sample_chunks):
        original(path, tuple(reversed(sample_chunks)))

    monkeypatch.setattr(builder_module, "_write_chunk_registry", reversed_registry)
    target = tmp_path / "cache"
    with pytest.raises(ValueError, match="does not align with embeddings"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
        )
    _assert_no_partial_build(tmp_path, target)


def test_encoder_failure_restores_offline_guards_and_cleans_atomic_temp(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, _texts = _write_inputs(tmp_path)

    class FailingEncoder(_FakeEncoder):
        def encode(self, *_args, **_kwargs):
            raise RuntimeError("injected encoding failure")

    _install_fake_encoder(monkeypatch, encoder=FailingEncoder())
    target = tmp_path / "cache"
    before_getaddrinfo = socket.getaddrinfo
    before_offline = os.environ.get("HF_HUB_OFFLINE")
    with pytest.raises(RuntimeError, match="injected encoding failure"):
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
        )
    _assert_no_partial_build(tmp_path, target)
    assert socket.getaddrinfo is before_getaddrinfo
    assert os.environ.get("HF_HUB_OFFLINE") == before_offline


@pytest.mark.parametrize(
    ("exception_type", "payload"),
    ((KeyboardInterrupt, "injected encoding interrupt"), (SystemExit, 73)),
    ids=("keyboard-interrupt", "system-exit"),
)
def test_base_exception_during_encode_restores_guards_cleans_and_preserves_original(
    tmp_path: Path,
    monkeypatch,
    exception_type,
    payload,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    interruption = exception_type(payload)

    class InterruptingEncoder(_FakeEncoder):
        def encode(self, *_args, **_kwargs):
            raise interruption

    _install_fake_encoder(monkeypatch, encoder=InterruptingEncoder())
    target = tmp_path / "cache"
    before_getaddrinfo = socket.getaddrinfo
    before_offline = os.environ.get("HF_HUB_OFFLINE")

    with pytest.raises(exception_type) as caught:
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
        )

    assert caught.value is interruption
    _assert_no_partial_build(tmp_path, target)
    assert socket.getaddrinfo is before_getaddrinfo
    assert os.environ.get("HF_HUB_OFFLINE") == before_offline


def test_interrupt_immediately_after_atomic_rename_cleans_published_target(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    _install_fake_encoder(monkeypatch)
    target = tmp_path / "cache"
    interruption = KeyboardInterrupt("injected post-rename interrupt")
    original_rename = os.rename

    def interrupt_after_publish(source, destination):
        original_rename(source, destination)
        if Path(destination) == target:
            raise interruption

    monkeypatch.setattr(builder_module.os, "rename", interrupt_after_publish)

    with pytest.raises(KeyboardInterrupt) as caught:
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
        )

    assert caught.value is interruption
    _assert_no_partial_build(tmp_path, target)


def test_interrupt_during_encode_preserves_unowned_target_created_mid_build(
    tmp_path: Path,
    monkeypatch,
):
    dataset, model, _texts = _write_inputs(tmp_path)
    target = tmp_path / "cache"
    marker = target / "external.bin"
    interruption = KeyboardInterrupt("injected interrupt after external target creation")

    class InterruptingEncoder(_FakeEncoder):
        def encode(self, *_args, **_kwargs):
            target.mkdir()
            marker.write_bytes(b"must remain")
            raise interruption

    _install_fake_encoder(monkeypatch, encoder=InterruptingEncoder())

    with pytest.raises(KeyboardInterrupt) as caught:
        build_production_embedding_cache(
            dataset_path=dataset,
            text_column="clinical_text",
            local_model_path=model,
            sentence_model_name=_SENTENCE_MODEL_NAME,
            chunk_configuration=_chunk_configuration(),
            target_dir=target,
        )

    assert caught.value is interruption
    assert marker.read_bytes() == b"must remain"
    assert not list(tmp_path.glob(f".{target.name}.building-*"))
