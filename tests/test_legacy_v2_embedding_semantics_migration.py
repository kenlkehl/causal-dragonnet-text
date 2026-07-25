from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import oci.inference.legacy_checkpoint_migration as migration
from oci.inference.portable_workflow_spec import identity_sha256


def _sha256(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _encoder_configuration() -> dict[str, object]:
    return {
        "chunk_size_words": 256,
        "chunk_overlap_words": 64,
        "max_chunks": 128,
        "chunk_selection": "last",
        "normalize_embeddings": True,
        "max_seq_length": 1024,
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


def test_exact_v2_producer_derives_closed_encoder_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "model"
    pooling_root = model_root / "1_Pooling"
    pooling_root.mkdir(parents=True)
    sentence_configuration = model_root / "config_sentence_transformers.json"
    pooling_configuration = pooling_root / "config.json"
    modules = model_root / "modules.json"
    sentence_configuration.write_text(
        json.dumps(
            {
                "prompts": {"query": "unused", "document": ""},
                "default_prompt_name": None,
                "similarity_fn_name": "cosine",
            }
        ),
        encoding="utf-8",
    )
    pooling_configuration.write_text(
        json.dumps(
            {
                "word_embedding_dimension": 4,
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": False,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False,
                "pooling_mode_weightedmean_tokens": False,
                "pooling_mode_lasttoken": True,
                "include_prompt": True,
            }
        ),
        encoding="utf-8",
    )
    modules.write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.models.Pooling",
                },
                {
                    "idx": 2,
                    "name": "2",
                    "path": "2_Normalize",
                    "type": "sentence_transformers.models.Normalize",
                },
            ]
        ),
        encoding="utf-8",
    )
    model_files = {
        path.relative_to(model_root).as_posix(): {
            "relative_path": path.relative_to(model_root).as_posix(),
            "sha256": _sha256(path)[0],
            "size_bytes": _sha256(path)[1],
        }
        for path in (sentence_configuration, pooling_configuration, modules)
    }

    snapshot = tmp_path / "snapshot"
    builder = snapshot / "oci" / "inference" / "production_embedding_cache_builder.py"
    lock = snapshot / "uv.lock"
    builder.parent.mkdir(parents=True)
    builder.write_text("# exact frozen builder fixture\n", encoding="utf-8")
    lock.write_text("# exact frozen dependency lock fixture\n", encoding="utf-8")
    snapshot_files = []
    for path in (builder, lock):
        digest, size = _sha256(path)
        snapshot_files.append(
            {
                "relative_path": path.relative_to(snapshot).as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
        )
        path.chmod(0o444)
    snapshot_files.sort(key=lambda row: str(row["relative_path"]))
    snapshot_body = {
        "schema_version": "production_source_snapshot_v1",
        "source_repository": str((tmp_path / "source").resolve()),
        "files": snapshot_files,
        "file_count": len(snapshot_files),
        "python_bytecode_writes_allowed": False,
    }
    snapshot_manifest = snapshot / "source_snapshot_manifest.json"
    snapshot_manifest.write_text(
        json.dumps(
            {
                **snapshot_body,
                "content_sha256": identity_sha256(snapshot_body),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    snapshot_manifest.chmod(0o444)
    builder.parent.chmod(0o555)
    builder.parent.parent.chmod(0o555)
    snapshot.chmod(0o555)

    model_tree_sha256 = identity_sha256({"fixture": "model tree"})
    producer = {
        "source_snapshot_content_sha256": identity_sha256(snapshot_body),
        "builder_relative_path": builder.relative_to(snapshot).as_posix(),
        "builder_code_sha256": _sha256(builder)[0],
        "dependency_lock_relative_path": lock.relative_to(snapshot).as_posix(),
        "dependency_lock_sha256": _sha256(lock)[0],
        "dependency_versions": {
            "sentence-transformers": "fixture",
            "torch": "fixture",
            "transformers": "fixture",
        },
        "model_tree_sha256": model_tree_sha256,
        "model_evidence_files": {relative: row["sha256"] for relative, row in model_files.items()},
    }
    monkeypatch.setattr(
        migration,
        "_ALLOWLISTED_V2_ENCODER_PRODUCER",
        producer,
    )

    workflow_root = tmp_path / "legacy_workflow"
    phase_manifest = workflow_root / "phases" / "embedding_cache" / "complete_manifest.json"
    phase_manifest.parent.mkdir(parents=True)
    request_body = {
        "schema_version": "production_all_evidence_workflow_v3",
        "source_snapshot_root": str(snapshot.resolve()),
        "source_snapshot": {
            "root": str(snapshot.resolve()),
            "manifest_path": str(snapshot_manifest.resolve()),
            "content_sha256": identity_sha256(snapshot_body),
            "file_count": len(snapshot_files),
        },
        "embedding_local_model_path": str(model_root.resolve()),
        "embedding_model_tree": {
            "files": sorted(
                model_files.values(),
                key=lambda row: str(row["relative_path"]),
            )
        },
    }
    request_sha256 = identity_sha256(request_body)
    (workflow_root / "immutable_run_request.json").write_text(
        json.dumps(
            {**request_body, "request_sha256": request_sha256},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": migration._EMBEDDING_V2_METADATA_SCHEMA,
        "hidden_size": 4,
        "production_provenance": {
            "schema_version": migration._EMBEDDING_V2_PROVENANCE_SCHEMA,
            "builder_version": migration._EMBEDDING_V2_BUILDER_VERSION,
            "builder_code_sha256": _sha256(builder)[0],
            "local_model": {
                "path": str(model_root.resolve()),
                "tree_sha256": model_tree_sha256,
            },
        },
    }
    validated = {
        "manifest_path": str(phase_manifest.resolve()),
        "manifest": {"request_sha256": request_sha256},
    }
    proof = migration._derive_allowlisted_v2_encoder_semantics(
        validated=validated,
        metadata=metadata,
        requested_configuration=_encoder_configuration(),
        embedding_model_tree_sha256=model_tree_sha256,
    )
    assert proof["status"] == "accepted_exact_frozen_v5_v2_producer"
    assert proof["default_prompt_name"] is None
    assert proof["sentence_pooling"] == "last_token_then_normalize_v1"
    assert proof["derived_encoder_configuration"] == (
        migration._allowlisted_v2_encoder_configuration()
    )
    assert proof["legacy_runtime_package_versions_separately_recorded"] is False

    changed = _encoder_configuration()
    changed["truncate_dim"] = 2
    with pytest.raises(ValueError, match="closed v3 encoder/output policy"):
        migration._derive_allowlisted_v2_encoder_semantics(
            validated=validated,
            metadata=metadata,
            requested_configuration=changed,
            embedding_model_tree_sha256=model_tree_sha256,
        )


def test_embedding_semantics_scan_proves_zero_and_normalization_policy() -> None:
    values = np.asarray(
        [
            [1.0, 0.0],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )
    proof = migration._scan_embedding_semantics(
        values,
        normalize_embeddings=True,
        zero_vector_policy="reject",
    )
    assert proof["zero_vector_count"] == 0
    assert proof["nonzero_vectors_within_unit_norm_tolerance"] is True

    with pytest.raises(ValueError, match="zero_vector_policy"):
        migration._scan_embedding_semantics(
            np.vstack([values, np.zeros((1, 2), dtype=np.float32)]),
            normalize_embeddings=True,
            zero_vector_policy="reject",
        )
