from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import oci.inference.portable_artifacts as portable_artifacts_module
import oci.inference.operator_trusted_embedding_cache_reader as trusted_reader_module
import oci.inference.review_spent_evidence_provider as spent_module
from oci.inference.operator_trusted_checkpoint_adoption import (
    validate_operator_trusted_portable_artifact,
)
from oci.inference.operator_trusted_embedding_cache_reader import (
    OperatorTrustedSpentOnlyFrozenChunkEmbeddingCache,
    build_operator_trusted_cache_read_proof,
    validate_operator_trusted_cache_read_proof,
)
from oci.inference.portable_artifacts import (
    ArtifactCompatibility,
    adopt_checkpoint,
    publish_portable_artifact,
)
from oci.inference.portable_identity import identity_sha256


def _digest(label: str) -> str:
    return identity_sha256({"label": label})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compatibility() -> ArtifactCompatibility:
    return ArtifactCompatibility(
        dataset_identity=_digest("dataset"),
        split_identity=_digest("split"),
        row_order_identity=_digest("rows"),
        model_identities={"embedding": _digest("model")},
        prompt_identities={},
        configuration_identity=_digest("config"),
        seed_identity=_digest("seed"),
        producer_code_identity=_digest("producer"),
        runtime_compatibility_class="python-posix-test-v1",
    )


def _migration(*, stored_array_dtype: str = "float32") -> dict:
    body = {
        "schema_version": "legacy_terminal_typed_request_migration_identity_v1",
        "phase": "embedding_cache",
        "typed_expectation": {
            "chunk_configuration": {
                "stored_array_dtype": stored_array_dtype,
            },
        },
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _case(
    tmp_path: Path,
    *,
    stored_array_dtype: str = "float32",
):
    root = tmp_path / "artifact"
    cache_dir = root / "embedding_cache"
    cache_dir.mkdir(parents=True)
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    np.save(cache_dir / "chunk_embeddings.npy", embeddings)
    np.save(
        cache_dir / "offsets.npy",
        np.asarray([0, 1, 2], dtype=np.int64),
    )
    (cache_dir / "chunk_texts.jsonl").write_text(
        "\n".join(
            (
                json.dumps({"chunks": ["alpha"]}),
                json.dumps({"chunks": ["beta"]}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "fixture_embedding_cache_v1",
        "num_samples": 2,
        "total_chunks": 2,
        "hidden_size": 2,
        "sentence_model_name": "fixture/model",
        "chunk_size_words": 32,
        "chunk_overlap_words": 0,
        "max_chunks": 4,
        "chunk_selection": "last",
        "normalize_embeddings": True,
        "max_seq_length": 64,
        "production_provenance": {
            "chunk_configuration": {
                "chunk_size_words": 32,
                "chunk_overlap_words": 0,
                "max_chunks": 4,
                "chunk_selection": "last",
                "normalize_embeddings": True,
                "max_seq_length": 64,
            },
        },
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    files = {
        name: {
            "sha256": _sha256(cache_dir / name),
            "size_bytes": (cache_dir / name).stat().st_size,
        }
        for name in (
            "metadata.json",
            "chunk_embeddings.npy",
            "offsets.npy",
            "chunk_texts.jsonl",
        )
    }
    provider = {
        "provider": "spent_only_frozen_chunk_embedding_cache_v2",
        "metadata_sha256": files["metadata.json"]["sha256"],
        "embeddings_sha256": files["chunk_embeddings.npy"]["sha256"],
        "offsets_sha256": files["offsets.npy"]["sha256"],
        "chunk_texts_sha256": files["chunk_texts.jsonl"]["sha256"],
        "row_count": 2,
        "chunk_count": 2,
        "cache_snapshot_authentication": "streamed_private_fd_sha256_v1",
        "chunk_text_storage": "private_fd_pread_lazy_row_decode_v1",
        "embeddings_path_backed": False,
        "private_snapshot_embedding_mmap": True,
        "future_row_text_decoded": False,
        "novel_text_encoding_allowed": False,
    }
    build_identity = {
        "schema_version": "fixture_cache_build_v1",
        "row_count": 2,
        "chunk_count": 2,
        "hidden_size": 2,
        "cache_files": files,
        "provider_identity": provider,
    }
    artifact = publish_portable_artifact(
        root=root,
        artifact_kind="embedding_cache",
        artifact_schema="fixture_embedding_cache_checkpoint_v1",
        compatibility=_compatibility(),
        upstream_artifact_ids=(_digest("prepared"),),
        payload_paths=tuple(
            f"embedding_cache/{name}"
            for name in (
                "metadata.json",
                "chunk_embeddings.npy",
                "offsets.npy",
                "chunk_texts.jsonl",
            )
        ),
    )
    prior_root = tmp_path / "prior"
    adopt_checkpoint(
        source=artifact.root,
        attestation_root=prior_root,
        consumer_request_sha256=_digest("prior request"),
        validated_artifact=artifact,
    )
    prior_path = prior_root / f"{artifact.artifact_id}.adoption.json"
    trusted = validate_operator_trusted_portable_artifact(
        source=artifact.root,
        prior_attestation_path=prior_path,
        expected_kind="embedding_cache",
        expected_compatibility_key=artifact.compatibility_key,
        expected_upstream_artifact_ids=(_digest("prepared"),),
    )
    migration = _migration(stored_array_dtype=stored_array_dtype)
    proof = build_operator_trusted_cache_read_proof(
        trusted,
        cache_dir=cache_dir.resolve(),
        cache_build_identity=build_identity,
        provider_identity=provider,
        migration_identity=migration,
    )
    return cache_dir.resolve(), proof, provider


def test_operator_trusted_reader_uses_path_mmaps_without_rehash_or_snapshot_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir, proof, provider = _case(tmp_path)
    cache_paths = {
        (cache_dir / name).resolve()
        for name in (
            "metadata.json",
            "chunk_embeddings.npy",
            "offsets.npy",
            "chunk_texts.jsonl",
        )
    }
    original_hash = portable_artifacts_module._safe_file_hash_with_identity

    def reject_cache_hash(path: Path, *, label: str):
        if Path(path).resolve() in cache_paths:
            raise AssertionError(f"cache payload was rehashed: {label}")
        return original_hash(path, label=label)

    def reject_snapshot(*_args, **_kwargs):
        raise AssertionError("cache payload was privately copied")

    monkeypatch.setattr(
        portable_artifacts_module,
        "_safe_file_hash_with_identity",
        reject_cache_hash,
    )
    monkeypatch.setattr(spent_module, "_snapshot_cache_file", reject_snapshot)
    monkeypatch.setattr(tempfile, "TemporaryFile", reject_snapshot)

    cache = OperatorTrustedSpentOnlyFrozenChunkEmbeddingCache(
        cache_dir,
        proof=proof,
    )
    assert isinstance(cache._embeddings, np.memmap)
    assert isinstance(cache._offsets, np.memmap)
    assert cache.identity() == provider
    assert cache.authenticated_snapshot_identity() == provider
    bound = cache.bind_spent((0, 1), ("alpha", "beta"))
    assert np.array_equal(
        bound.chunk_matrix(0),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    assert proof["payload_bytes_reauthenticated"] is False
    assert proof["fresh_full_byte_validation_achieved"] is False
    assert proof["global_release_certified"] is False


def test_operator_trusted_reader_rejects_stat_discontinuity(
    tmp_path: Path,
) -> None:
    cache_dir, proof, _provider = _case(tmp_path)
    metadata_path = cache_dir / "metadata.json"
    original = metadata_path.read_bytes()
    metadata_path.write_bytes(original)

    with pytest.raises(ValueError, match="stat identity changed"):
        validate_operator_trusted_cache_read_proof(
            proof,
            cache_dir=cache_dir,
        )


def test_operator_trusted_reader_rejects_migration_dtype_mismatch(
    tmp_path: Path,
) -> None:
    cache_dir, proof, _provider = _case(
        tmp_path,
        stored_array_dtype="float16",
    )

    with pytest.raises(ValueError, match="dtype differs"):
        OperatorTrustedSpentOnlyFrozenChunkEmbeddingCache(
            cache_dir,
            proof=proof,
        )


def test_operator_trusted_reader_reuses_only_closed_authenticated_line_spans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir, proof, _provider = _case(tmp_path)
    first_stop = len(
        (json.dumps({"chunks": ["alpha"]}) + "\n").encode("utf-8")
    )
    file_size = (cache_dir / "chunk_texts.jsonl").stat().st_size
    spans = ((0, first_stop), (first_stop, file_size))

    def reject_scan(*_args, **_kwargs):
        raise AssertionError("authenticated line spans were rescanned")

    monkeypatch.setattr(
        trusted_reader_module,
        "_snapshot_line_spans",
        reject_scan,
    )
    cache = OperatorTrustedSpentOnlyFrozenChunkEmbeddingCache(
        cache_dir,
        proof=proof,
        authenticated_line_spans=spans,
    )
    bound = cache.bind_spent((0, 1), ("alpha", "beta"))
    assert bound.chunk_texts((0, 1)) == (("alpha",), ("beta",))

    with pytest.raises(ValueError, match="contiguous nonempty coverage"):
        OperatorTrustedSpentOnlyFrozenChunkEmbeddingCache(
            cache_dir,
            proof=proof,
            authenticated_line_spans=(
                (0, first_stop),
                (first_stop + 1, file_size),
            ),
        )
