from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import oci.inference.production_authenticated_tree_cache as tree_cache_module
import oci.inference.production_embedding_cache_builder as builder_module
import oci.inference.production_embedding_cache_relocation as relocation_module
from oci.inference.production_embedding_cache_builder import (
    build_production_embedding_cache,
)
from oci.inference.production_embedding_cache_relocation import (
    PRODUCTION_EMBEDDING_CACHE_RELOCATION_ATTESTATION_SCHEMA,
    PRODUCTION_EMBEDDING_CACHE_RELOCATION_RESULT_SCHEMA,
    PRODUCTION_EMBEDDING_CACHE_RELOCATION_TERMINAL_SCHEMA,
    PRODUCTION_EMBEDDING_CACHE_RELOCATOR_VERSION,
    ProductionEmbeddingCacheRelocationOptions,
    relocate_authenticated_production_embedding_cache,
    validate_relocated_production_embedding_cache,
)
from oci.inference.production_text_preparation import (
    TextPreparationOptions,
    prepare_modeling_cohort,
)

_MODEL_NAME = "Qwen/test-production-embedding"


class _Tokenizer:
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
        return {"length": [len(value.split()) + 2 for value in inputs]}


class _Encoder:
    max_seq_length = 32
    tokenizer = _Tokenizer()
    default_prompt_name = None
    prompts: dict[str, str] = {}

    def float(self):
        return self

    def eval(self):
        return self

    def encode(
        self,
        chunks,
        *,
        batch_size,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar,
    ):
        assert batch_size == len(chunks)
        assert convert_to_numpy is True
        assert show_progress_bar is False
        output = []
        for chunk in chunks:
            digest = hashlib.sha256(chunk.encode("utf-8")).digest()
            vector = np.asarray([float(value + 1) for value in digest[:5]], dtype=np.float32)
            if normalize_embeddings:
                vector /= np.linalg.norm(vector)
            output.append(vector)
        return np.asarray(output, dtype=np.float32)


def _chunk_configuration() -> dict[str, object]:
    return {
        "chunk_size_words": 4,
        "chunk_overlap_words": 1,
        "max_chunks": 10,
        "chunk_selection": "last",
        "normalize_embeddings": True,
        "max_seq_length": 32,
    }


def _prepare(dataset: Path, output: Path) -> tuple[Path, Path]:
    prepare_modeling_cohort(
        TextPreparationOptions(
            dataset_path=dataset,
            output_dir=output,
            unit_id_column="person",
            text_column="note",
            treatment_column="therapy",
            outcome_column="response",
            repeated_character_threshold=4,
        )
    )
    return output / "modeling_cohort.parquet", output / "preparation_manifest.json"


def _case(tmp_path: Path, monkeypatch) -> ProductionEmbeddingCacheRelocationOptions:
    dataset = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "person": ["p2", "p0", "p1", "p3"],
            "note": [
                "alpha beta gamma delta epsilon",
                "one two three four five six",
                "clinical" + "\u2015" * 7 + "text",
                "",
            ],
            "therapy": [0, 1, 0, 1],
            "response": [1, 0, 0, 1],
        }
    ).to_parquet(dataset, index=False)
    source_prepared, source_manifest = _prepare(dataset, tmp_path / "source-prepared")
    fresh_prepared, fresh_manifest = _prepare(dataset, tmp_path / "fresh-prepared")

    model = tmp_path / "local-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"safe-test"}\n', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"safe-weights")
    monkeypatch.setattr(
        builder_module,
        "_load_local_sentence_encoder",
        lambda **_kwargs: _Encoder(),
    )
    source_cache = tmp_path / "source-cache"
    build_production_embedding_cache(
        dataset_path=source_prepared,
        text_column="note",
        local_model_path=model,
        sentence_model_name=_MODEL_NAME,
        chunk_configuration=_chunk_configuration(),
        target_dir=source_cache,
        device="cpu",
        batch_size=2,
    )
    return ProductionEmbeddingCacheRelocationOptions(
        source_cache_dir=source_cache,
        source_prepared_cohort_path=source_prepared,
        source_preparation_manifest_path=source_manifest,
        fresh_prepared_cohort_path=fresh_prepared,
        fresh_preparation_manifest_path=fresh_manifest,
        local_model_path=model,
        target_dir=tmp_path / "relocated",
        unit_id_column="person",
        text_column="note",
        treatment_column="therapy",
        outcome_column="response",
        sentence_model_name=_MODEL_NAME,
        chunk_configuration=_chunk_configuration(),
    )


def test_relocates_cache_and_prepared_bytes_with_closed_attestation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    options = _case(tmp_path, monkeypatch)

    result = relocate_authenticated_production_embedding_cache(options)

    assert result.root == options.target_dir
    assert result.cache_dir == options.target_dir / "embedding_cache"
    assert result.prepared_cohort_path == (
        options.target_dir / "prepared" / "modeling_cohort.parquet"
    )
    assert set(path.name for path in result.root.iterdir()) == {
        "prepared",
        "embedding_cache",
        "relocation_attestation.json",
        "complete_manifest.json",
    }
    assert set(path.name for path in result.cache_dir.iterdir()) == {
        "metadata.json",
        "chunk_embeddings.npy",
        "offsets.npy",
        "chunk_texts.jsonl",
    }
    for name in (
        "metadata.json",
        "chunk_embeddings.npy",
        "offsets.npy",
        "chunk_texts.jsonl",
    ):
        source = options.source_cache_dir / name
        copied = result.cache_dir / name
        assert copied.read_bytes() == source.read_bytes()
        assert copied.stat().st_ino != source.stat().st_ino
    assert result.prepared_cohort_path.read_bytes() == (
        options.source_prepared_cohort_path.read_bytes()
    )
    assert (
        result.prepared_cohort_path.stat().st_ino
        != options.source_prepared_cohort_path.stat().st_ino
    )

    attestation = json.loads(result.attestation_path.read_text(encoding="utf-8"))
    terminal = json.loads(result.terminal_manifest_path.read_text(encoding="utf-8"))
    assert attestation["schema_version"] == PRODUCTION_EMBEDDING_CACHE_RELOCATION_ATTESTATION_SCHEMA
    assert terminal["schema_version"] == PRODUCTION_EMBEDDING_CACHE_RELOCATION_TERMINAL_SCHEMA
    assert attestation["proofs"]["source_and_fresh_rows_equal"] is True
    assert attestation["proofs"]["hardlinks_allowed"] is False
    assert attestation["proofs"]["local_model_revalidation_policy"] == (
        "single_full_hash_process_local_inventory_guard_v1"
    )
    assert terminal["status"] == "complete"
    assert result.identity()["schema_version"] == (
        PRODUCTION_EMBEDDING_CACHE_RELOCATION_RESULT_SCHEMA
    )
    assert (
        result.identity()["authenticated_tree_code_sha256"]
        == hashlib.sha256(Path(tree_cache_module.__file__).read_bytes()).hexdigest()
    )

    reopened = validate_relocated_production_embedding_cache(options)
    assert reopened.identity() == result.identity()
    assert reopened.cache_build_identity == result.cache_build_identity


def test_historical_cache_provenance_uses_one_shared_model_authentication_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert PRODUCTION_EMBEDDING_CACHE_RELOCATOR_VERSION == (
        "production_embedding_cache_relocator_v2"
    )
    assert hashlib.sha256(Path(builder_module.__file__).read_bytes()).hexdigest() == (
        "9af77ce3cc47ea77c819974f4b55885ddeb279f758bbac6ca5b987ac9d61aabd"
    )
    options = _case(tmp_path, monkeypatch)
    source_metadata = json.loads(
        (options.source_cache_dir / "metadata.json").read_text(encoding="utf-8")
    )
    source_model_provenance = source_metadata["production_provenance"]["local_model"]
    validator_calls: list[object] = []
    authenticated_model_files: list[str] = []
    historical_validator = relocation_module.validate_published_production_embedding_cache
    full_file_authentication = tree_cache_module._stable_file_authentication

    def validate_without_rehashing_model(**kwargs):
        validator_calls.append(kwargs["expected_local_model_path"])
        return historical_validator(**kwargs)

    def count_full_file_authentication(root, relative_path):
        authenticated_model_files.append(relative_path)
        return full_file_authentication(root, relative_path)

    monkeypatch.setattr(
        relocation_module,
        "validate_published_production_embedding_cache",
        validate_without_rehashing_model,
    )
    monkeypatch.setattr(
        tree_cache_module,
        "_stable_file_authentication",
        count_full_file_authentication,
    )
    monkeypatch.setattr(
        builder_module,
        "_model_tree_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "the historical validator must not rehash the local model"
        ),
    )

    result = relocate_authenticated_production_embedding_cache(options)
    reopened = validate_relocated_production_embedding_cache(options)

    # Source, temporary copy, terminal reopening, and the explicit second
    # reopening all retain the historical full cache validator.
    assert len(validator_calls) == 6
    assert all(value is None for value in validator_calls)
    assert sorted(authenticated_model_files) == [
        "config.json",
        "model.safetensors",
    ]
    assert (
        result.cache_build_identity["local_model_tree_sha256"]
        == source_model_provenance["tree_sha256"]
    )
    assert reopened.cache_build_identity == result.cache_build_identity
    copied_metadata = json.loads((result.cache_dir / "metadata.json").read_text(encoding="utf-8"))
    assert copied_metadata["production_provenance"]["local_model"] == source_model_provenance


@pytest.mark.parametrize(
    "field",
    [
        "path",
        "tree_sha256",
        "file_count",
        "directory_count",
        "total_file_bytes",
    ],
)
def test_shared_model_capability_compares_every_metadata_provenance_field(
    tmp_path: Path,
    monkeypatch,
    field: str,
) -> None:
    options = _case(tmp_path, monkeypatch)
    identity = copy.deepcopy(
        dict(
            relocation_module.validate_published_production_embedding_cache(
                cache_dir=options.source_cache_dir,
                dataset_path=options.source_prepared_cohort_path,
                text_column=options.text_column,
                sentence_model_name=options.sentence_model_name,
                chunk_configuration=options.chunk_configuration,
                expected_local_model_path=None,
            )
        )
    )
    metadata_path = options.source_cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = metadata["production_provenance"]["local_model"]
    if field == "path":
        model[field] = str((tmp_path / "substituted-model").resolve())
    elif field == "tree_sha256":
        model[field] = "0" * 64
        identity["local_model_tree_sha256"] = model[field]
    else:
        model[field] = int(model[field]) + 1
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    metadata_snapshot = relocation_module._stable_file_snapshot(
        metadata_path,
        label="tampered metadata fixture",
    )[1]
    identity["cache_files"]["metadata.json"] = metadata_snapshot.registration()

    with pytest.raises(ValueError, match="provenance differs"):
        relocation_module._authenticate_local_model_against_builder_cache(
            local_model_path=options.local_model_path,
            cache_root=options.source_cache_dir,
            cache_identity=identity,
        )


def test_rejects_a_different_fresh_preparation_before_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    options = _case(tmp_path, monkeypatch)
    different_dataset = tmp_path / "different.parquet"
    frame = pd.read_parquet(tmp_path / "source.parquet")
    frame.loc[0, "note"] = "different prepared narrative"
    frame.to_parquet(different_dataset, index=False)
    different_cohort, different_manifest = _prepare(
        different_dataset,
        tmp_path / "different-prepared",
    )
    mismatched = ProductionEmbeddingCacheRelocationOptions(
        **{
            **options.__dict__,
            "fresh_prepared_cohort_path": different_cohort,
            "fresh_preparation_manifest_path": different_manifest,
        }
    )

    with pytest.raises(ValueError, match="fresh preparation identity differs"):
        relocate_authenticated_production_embedding_cache(mismatched)
    assert not options.target_dir.exists()
    assert not list(tmp_path.glob(".relocated.relocating-*"))


@pytest.mark.parametrize(
    ("relative_path", "expected_message"),
    [
        (
            Path("embedding_cache") / "chunk_texts.jsonl",
            "artifact bytes differ|failed authentication|cache",
        ),
        (
            Path("relocation_attestation.json"),
            "artifact bytes differ|invalid JSON|attestation",
        ),
        (
            Path("complete_manifest.json"),
            "invalid JSON|terminal",
        ),
    ],
)
def test_read_only_validator_rejects_destination_tampering(
    tmp_path: Path,
    monkeypatch,
    relative_path: Path,
    expected_message: str,
) -> None:
    options = _case(tmp_path, monkeypatch)
    relocate_authenticated_production_embedding_cache(options)
    with (options.target_dir / relative_path).open("ab") as handle:
        handle.write(b"\nTAMPER")

    with pytest.raises((ValueError, RuntimeError), match=expected_message):
        validate_relocated_production_embedding_cache(options)


def test_read_only_validator_reauthenticates_source_and_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    options = _case(tmp_path, monkeypatch)
    relocate_authenticated_production_embedding_cache(options)
    (options.local_model_path / "model.safetensors").write_bytes(b"changed-weights")

    with pytest.raises(
        (ValueError, RuntimeError),
        match="local model|inventory|changed",
    ):
        validate_relocated_production_embedding_cache(options)


def test_read_only_validator_rejects_same_byte_model_inode_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    options = _case(tmp_path, monkeypatch)
    relocate_authenticated_production_embedding_cache(options)
    artifact = options.local_model_path / "model.safetensors"
    original_bytes = artifact.read_bytes()
    original_inode = artifact.stat().st_ino
    replacement = options.local_model_path / "replacement.safetensors"
    replacement.write_bytes(original_bytes)
    os.replace(replacement, artifact)
    assert artifact.stat().st_ino != original_inode

    with pytest.raises(
        (ValueError, RuntimeError),
        match="local model|inventory|changed",
    ):
        validate_relocated_production_embedding_cache(options)


def test_read_only_validator_rejects_a_substituted_hard_link(
    tmp_path: Path,
    monkeypatch,
) -> None:
    options = _case(tmp_path, monkeypatch)
    result = relocate_authenticated_production_embedding_cache(options)
    copied_offsets = result.cache_dir / "offsets.npy"
    copied_offsets.unlink()
    os.link(options.source_cache_dir / "offsets.npy", copied_offsets)

    with pytest.raises(ValueError, match="hard link|non-linked"):
        validate_relocated_production_embedding_cache(options)


def test_read_only_validator_rejects_an_arbitrary_hard_link(
    tmp_path: Path,
    monkeypatch,
) -> None:
    options = _case(tmp_path, monkeypatch)
    result = relocate_authenticated_production_embedding_cache(options)
    copied_offsets = result.cache_dir / "offsets.npy"
    unrelated_copy = tmp_path / "same-offsets.npy"
    shutil.copyfile(copied_offsets, unrelated_copy)
    copied_offsets.unlink()
    os.link(unrelated_copy, copied_offsets)

    with pytest.raises(ValueError, match="non-linked"):
        validate_relocated_production_embedding_cache(options)


def test_read_only_validator_rejects_root_inode_substitution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    options = _case(tmp_path, monkeypatch)
    relocate_authenticated_production_embedding_cache(options)
    original_validate_inputs = relocation_module._validate_inputs
    substituted = False

    def validate_then_substitute(*args, **kwargs):
        nonlocal substituted
        result = original_validate_inputs(*args, **kwargs)
        if not substituted:
            substituted = True
            backup = tmp_path / "relocated-original"
            options.target_dir.rename(backup)
            shutil.copytree(backup, options.target_dir)
        return result

    monkeypatch.setattr(
        relocation_module,
        "_validate_inputs",
        validate_then_substitute,
    )
    with pytest.raises(RuntimeError, match="root or artifacts changed"):
        validate_relocated_production_embedding_cache(options)


def test_source_tampering_and_changed_configuration_are_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    options = _case(tmp_path, monkeypatch)
    relocate_authenticated_production_embedding_cache(options)
    with (options.source_cache_dir / "chunk_texts.jsonl").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises((ValueError, RuntimeError), match="cache"):
        validate_relocated_production_embedding_cache(options)

    changed = ProductionEmbeddingCacheRelocationOptions(
        **{
            **options.__dict__,
            "chunk_configuration": {
                **dict(options.chunk_configuration),
                "max_chunks": 11,
            },
        }
    )
    with pytest.raises((ValueError, RuntimeError), match="cache|configuration|policy"):
        validate_relocated_production_embedding_cache(changed)
