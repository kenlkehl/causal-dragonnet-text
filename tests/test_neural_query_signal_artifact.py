from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from oracle_experiment_scripts import run_neural_query_agentic_forest as producer
from oci.inference.neural_query_signal_artifact import (
    QUERY_BANKS,
    build_fold_honest_query_signals,
    sha256_file,
    write_fold_honest_query_signal_artifact,
)

_PARENT_INPUT_BINDING_SHA256 = hashlib.sha256(b"test-parent-input-binding").hexdigest()
_QUERY_DISCOVERY_IDENTITY = hashlib.sha256(b"test-query-discovery-identity").hexdigest()


def test_query_evidence_stage_projects_only_exact_model_columns(tmp_path, monkeypatch):
    calls = []

    def projected_read(source, *, columns):
        assert isinstance(source, io.BytesIO)
        calls.append((source.getvalue(), columns))
        import pandas as pd

        return pd.DataFrame({column: [0] for column in columns})

    monkeypatch.setattr(producer.pd, "read_parquet", projected_read)
    path = tmp_path / "dataset.parquet"
    path.write_bytes(b"exact model projection bytes")
    frame = producer._load_exact_model_projection(
        path,
        patient_id_column="patient_id",
        text_column="clinical_text",
        treatment_column="treatment",
        outcome_column="outcome",
    )

    assert calls == [
        (
            b"exact model projection bytes",
            ["patient_id", "clinical_text", "treatment", "outcome"],
        )
    ]
    assert (
        frame.attrs["source_snapshot_sha256"]
        == hashlib.sha256(b"exact model projection bytes").hexdigest()
    )
    assert frame.columns.tolist() == [
        "patient_id",
        "clinical_text",
        "treatment",
        "outcome",
    ]
    with pytest.raises(ValueError, match="must be distinct"):
        producer._load_exact_model_projection(
            path,
            patient_id_column="patient_id",
            text_column="clinical_text",
            treatment_column="treatment",
            outcome_column="treatment",
        )


def test_standalone_joblib_loader_deserializes_the_hashed_snapshot_after_path_swap(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.joblib"
    replacement = tmp_path / "replacement.joblib"
    joblib.dump({"identity": "authenticated", "value": [1, 2, 3]}, checkpoint)
    joblib.dump({"identity": "replacement", "value": [999]}, replacement)
    original_bytes = checkpoint.read_bytes()
    original_read_bytes = Path.read_bytes

    def swap_after_snapshot(path):
        snapshot = original_read_bytes(path)
        if path == checkpoint:
            replacement.replace(checkpoint)
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", swap_after_snapshot)
    loaded, digest = producer._load_joblib_exact_bytes(checkpoint)

    assert loaded == {"identity": "authenticated", "value": [1, 2, 3]}
    assert digest == hashlib.sha256(original_bytes).hexdigest()
    assert original_read_bytes(checkpoint) != original_bytes


def test_model_projection_parses_the_hashed_snapshot_after_path_swap(tmp_path, monkeypatch):
    import pandas as pd

    columns = ["patient_id", "clinical_text", "treatment", "outcome"]
    original = pd.DataFrame(
        [["p1", "baseline original", 0, 1]],
        columns=columns,
    )
    replacement = pd.DataFrame(
        [["p9", "replacement", 1, 0]],
        columns=columns,
    )
    path = tmp_path / "dataset.parquet"
    replacement_path = tmp_path / "replacement.parquet"
    original.to_parquet(path, index=False)
    replacement.to_parquet(replacement_path, index=False)
    original_bytes = path.read_bytes()
    original_read_bytes = Path.read_bytes

    def swap_after_snapshot(candidate):
        snapshot = original_read_bytes(candidate)
        if candidate == path:
            replacement_path.replace(path)
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", swap_after_snapshot)
    loaded = producer._load_exact_model_projection(
        path,
        patient_id_column="patient_id",
        text_column="clinical_text",
        treatment_column="treatment",
        outcome_column="outcome",
    )

    assert loaded.to_dict("records") == original.to_dict("records")
    assert loaded.attrs["source_snapshot_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert original_read_bytes(path) != original_bytes


def test_executable_query_reuse_requires_parent_input_binding_before_deserialization(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "unbound_subfold.joblib"
    checkpoint.write_bytes(b"must never be deserialized")
    monkeypatch.setattr(
        producer,
        "_load_joblib_exact_bytes",
        lambda _path: (_ for _ in ()).throw(AssertionError("checkpoint was loaded")),
    )
    with pytest.raises(ValueError, match="parent input binding"):
        producer._fit_subfold(
            fold=1,
            train_indices=np.asarray([0]),
            validation_indices=np.asarray([1]),
            row_ids=(0, 1),
            chunks=(np.ones((1, 2)), np.ones((1, 2))),
            texts=("baseline zero", "baseline one"),
            treatment=np.asarray([0.0, 1.0]),
            outcome=np.asarray([0.0, 1.0]),
            outcome_binary=True,
            nuisance_views=({"name": "view", "ngram_range": [1, 2]},),
            nuisance_folds=2,
            config=producer.NeuralQueryAgenticForestConfig(
                treatment_query_count=1,
                outcome_query_count=1,
                effect_query_count=1,
                query_inner_folds=2,
                initial_pool_size=2,
                query_epochs=1,
                final_refit_epochs=1,
            ),
            seed=7,
            device="cpu",
            checkpoint_path=checkpoint,
            use_executable_checkpoints=True,
        )


def _candidate(bank: str, fold: int, index: int, query: list[float], score: float) -> dict:
    return {
        "candidate_id": f"{bank}_fold_{fold:02d}_query_{index:03d}",
        "query": np.asarray(query, dtype=np.float32),
        "train_standardized_score": score,
        # A consumer must never use this validation-label diagnostic as signal.
        "validation_audit_standardized_score": 9999.0,
    }


def _checkpoint(path: Path, *, fold: int, train: list[int], validation: list[int]) -> Path:
    banks = {}
    for bank_index, bank in enumerate(QUERY_BANKS):
        banks[bank] = {
            "candidates": [
                _candidate(bank, fold, 1, [1.0, 0.0], 1.0 + bank_index),
                _candidate(bank, fold, 2, [0.0, 1.0], -1.0 - bank_index),
            ]
        }
    identity_payload = {
        "train_row_ids": train,
        "validation_row_ids": validation,
        "unused_train_labels": [0.0] * len(train),
        "parent_input_binding_sha256": _PARENT_INPUT_BINDING_SHA256,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    joblib.dump(
        {
            "identity": identity,
            "fold": fold,
            "identity_payload": identity_payload,
            "banks": banks,
        },
        path,
    )
    return path


def _discovery() -> dict:
    banks = {}
    for bank_index, bank in enumerate(QUERY_BANKS):
        banks[bank] = {
            "queries": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "records": [
                {
                    "query_id": f"{bank}_query_001",
                    "fit_standardized_score": 1.5 + bank_index,
                },
                {
                    "query_id": f"{bank}_query_002",
                    "fit_standardized_score": -1.5 - bank_index,
                },
            ],
            # Deliberately unusable: full-refit train activations must be ignored.
            "train_activations": np.full((4, 2), np.nan),
        }
    return {
        "identity": _QUERY_DISCOVERY_IDENTITY,
        "parent_input_binding_sha256": _PARENT_INPUT_BINDING_SHA256,
        "banks": banks,
    }


def _discovery_checkpoint(path: Path) -> Path:
    joblib.dump(_discovery(), path)
    return path


def _chunks() -> list[np.ndarray]:
    return [
        np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
        np.asarray([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32),
        np.asarray([[0.7, 0.3]], dtype=np.float32),
        np.asarray([[0.2, 0.8]], dtype=np.float32),
        np.asarray([[0.9, 0.1]], dtype=np.float32),
        np.asarray([[0.3, 0.7]], dtype=np.float32),
    ]


def test_builds_inner_oof_and_outer_heldout_query_signal_artifacts(tmp_path):
    checkpoints = [
        _checkpoint(tmp_path / "subfold_01.joblib", fold=1, train=[2, 3], validation=[0, 1]),
        _checkpoint(tmp_path / "subfold_02.joblib", fold=2, train=[0, 1], validation=[2, 3]),
    ]
    chunks = _chunks()

    result = build_fold_honest_query_signals(
        outer_fold=1,
        fit_row_ids=[0, 1, 2, 3],
        heldout_row_ids=[4, 5],
        fit_chunk_matrices=chunks[:4],
        heldout_chunk_matrices=chunks[4:],
        query_discovery_checkpoint_path=_discovery_checkpoint(
            tmp_path / "final_query_discovery.joblib"
        ),
        subfold_checkpoint_paths=checkpoints,
        temperature=0.1,
        devices_by_bank={bank: "cpu" for bank in QUERY_BANKS},
        expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
        expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
    )

    train = result.activations.loc[result.activations["row_scope"] == "outer_train_inner_oof"]
    heldout = result.activations.loc[result.activations["row_scope"] == "outer_heldout_final_refit"]
    assert train.groupby("_oci_row_id").size().to_dict() == {0: 6, 1: 6, 2: 6, 3: 6}
    assert heldout.groupby("_oci_row_id").size().to_dict() == {4: 6, 5: 6}
    assert set(train["inner_fold"].dropna().astype(int)) == {1, 2}
    assert set(heldout["inner_fold"].dropna()) == set()
    assert 9999.0 not in set(result.activations["fit_standardized_score"])
    assert result.audit["full_refit_train_activations_used"] is False
    assert result.audit["validation_audit_scores_used_as_signal"] is False
    assert result.audit["outer_heldout_labels_accessed"] is False
    assert result.audit["posthoc_targets_consumed"] is False
    assert result.audit["dataset_specific_truth_consumed"] is False
    assert result.audit["final_refit_fit_row_ids"] == [0, 1, 2, 3]
    assert result.audit["outer_heldout_row_ids"] == [4, 5]
    assert result.audit["subfold_checkpoints"][0]["fit_row_ids"] == [2, 3]
    assert result.audit["subfold_checkpoints"][0]["validation_row_ids"] == [0, 1]
    assert len(result.audit["subfold_checkpoints"][0]["split_fingerprint"]) == 64
    assert result.audit["final_refit_checkpoint"]["sha256"] == sha256_file(
        tmp_path / "final_query_discovery.joblib"
    )
    assert len(result.signals) == 6
    assert "neural_query_effect_signed_order_02" in result.signals.columns

    written = write_fold_honest_query_signal_artifact(
        tmp_path / "written_query_signals",
        bundle=result,
    )
    assert written.manifest_path.is_file()
    assert written.signal_parquet_path.is_file()
    assert written.signal_parquet_sha256 == sha256_file(written.signal_parquet_path)


def test_final_checkpoint_path_replacement_cannot_change_loaded_snapshot(
    tmp_path,
    monkeypatch,
):
    checkpoints = [
        _checkpoint(tmp_path / "subfold_01.joblib", fold=1, train=[2, 3], validation=[0, 1]),
        _checkpoint(tmp_path / "subfold_02.joblib", fold=2, train=[0, 1], validation=[2, 3]),
    ]
    final_checkpoint = _discovery_checkpoint(tmp_path / "final_query_discovery.joblib")
    replacement = tmp_path / "replacement_query_discovery.joblib"
    joblib.dump({"banks": {}}, replacement)
    original_sha256 = sha256_file(final_checkpoint)
    original_load = joblib.load
    loaded_sources = []

    def replace_path_before_deserialization(source, *args, **kwargs):
        loaded_sources.append(source)
        if len(loaded_sources) == 1:
            replacement.replace(final_checkpoint)
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(joblib, "load", replace_path_before_deserialization)
    chunks = _chunks()
    result = build_fold_honest_query_signals(
        outer_fold=1,
        fit_row_ids=[0, 1, 2, 3],
        heldout_row_ids=[4, 5],
        fit_chunk_matrices=chunks[:4],
        heldout_chunk_matrices=chunks[4:],
        query_discovery_checkpoint_path=final_checkpoint,
        subfold_checkpoint_paths=checkpoints,
        temperature=0.1,
        devices_by_bank={bank: "cpu" for bank in QUERY_BANKS},
        expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
        expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
    )

    assert isinstance(loaded_sources[0], io.BytesIO)
    assert result.audit["final_refit_checkpoint"]["sha256"] == original_sha256
    assert sha256_file(final_checkpoint) != original_sha256


def test_subfold_checkpoint_mutation_cannot_change_loaded_snapshot(tmp_path, monkeypatch):
    checkpoints = [
        _checkpoint(tmp_path / "subfold_01.joblib", fold=1, train=[2, 3], validation=[0, 1]),
        _checkpoint(tmp_path / "subfold_02.joblib", fold=2, train=[0, 1], validation=[2, 3]),
    ]
    final_checkpoint = _discovery_checkpoint(tmp_path / "final_query_discovery.joblib")
    original_sha256 = sha256_file(checkpoints[0])
    original_load = joblib.load
    loaded_sources = []

    def mutate_path_before_deserialization(source, *args, **kwargs):
        loaded_sources.append(source)
        if len(loaded_sources) == 2:
            checkpoints[0].write_bytes(b"mutated-after-checkpoint-snapshot")
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(joblib, "load", mutate_path_before_deserialization)
    chunks = _chunks()
    result = build_fold_honest_query_signals(
        outer_fold=1,
        fit_row_ids=[0, 1, 2, 3],
        heldout_row_ids=[4, 5],
        fit_chunk_matrices=chunks[:4],
        heldout_chunk_matrices=chunks[4:],
        query_discovery_checkpoint_path=final_checkpoint,
        subfold_checkpoint_paths=checkpoints,
        temperature=0.1,
        devices_by_bank={bank: "cpu" for bank in QUERY_BANKS},
        expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
        expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
    )

    assert isinstance(loaded_sources[1], io.BytesIO)
    assert result.audit["subfold_checkpoints"][0]["sha256"] == original_sha256
    assert sha256_file(checkpoints[0]) != original_sha256


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("identity", "f" * 64, "identity does not match"),
        ("parent_input_binding_sha256", "e" * 64, "wrong parent input binding"),
    ],
)
def test_final_query_checkpoint_must_match_current_run_identity(
    tmp_path,
    field,
    replacement,
    message,
):
    checkpoints = [
        _checkpoint(tmp_path / "subfold_01.joblib", fold=1, train=[2, 3], validation=[0, 1]),
        _checkpoint(tmp_path / "subfold_02.joblib", fold=2, train=[0, 1], validation=[2, 3]),
    ]
    discovery = _discovery()
    discovery[field] = replacement
    final_checkpoint = tmp_path / "final_query_discovery.joblib"
    joblib.dump(discovery, final_checkpoint)

    with pytest.raises(ValueError, match=message):
        build_fold_honest_query_signals(
            outer_fold=1,
            fit_row_ids=[0, 1, 2, 3],
            heldout_row_ids=[4, 5],
            fit_chunk_matrices=_chunks()[:4],
            heldout_chunk_matrices=_chunks()[4:],
            query_discovery_checkpoint_path=final_checkpoint,
            subfold_checkpoint_paths=checkpoints,
            temperature=0.1,
            devices_by_bank={bank: "cpu" for bank in QUERY_BANKS},
            expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
            expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
        )


def test_subfold_with_valid_partition_but_wrong_parent_binding_is_rejected(tmp_path):
    bad = _checkpoint(
        tmp_path / "subfold_wrong_parent.joblib",
        fold=1,
        train=[2, 3],
        validation=[0, 1],
    )
    checkpoint = joblib.load(bad)
    checkpoint["identity_payload"]["parent_input_binding_sha256"] = "d" * 64
    checkpoint["identity"] = hashlib.sha256(
        json.dumps(checkpoint["identity_payload"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    joblib.dump(checkpoint, bad)

    with pytest.raises(ValueError, match="wrong parent input binding"):
        build_fold_honest_query_signals(
            outer_fold=1,
            fit_row_ids=[0, 1, 2, 3],
            heldout_row_ids=[4, 5],
            fit_chunk_matrices=_chunks()[:4],
            heldout_chunk_matrices=_chunks()[4:],
            query_discovery_checkpoint_path=_discovery_checkpoint(
                tmp_path / "final_query_discovery.joblib"
            ),
            subfold_checkpoint_paths=[bad],
            temperature=0.1,
            devices_by_bank={bank: "cpu" for bank in QUERY_BANKS},
            expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
            expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
        )


def test_subfold_identity_payload_mutation_with_stale_identity_is_rejected(tmp_path):
    stale = _checkpoint(
        tmp_path / "subfold_stale_identity.joblib",
        fold=1,
        train=[2, 3],
        validation=[0, 1],
    )
    checkpoint = joblib.load(stale)
    checkpoint["identity_payload"]["unused_train_labels"] = [123.0, 456.0]
    joblib.dump(checkpoint, stale)

    with pytest.raises(ValueError, match="identity payload was changed"):
        build_fold_honest_query_signals(
            outer_fold=1,
            fit_row_ids=[0, 1, 2, 3],
            heldout_row_ids=[4, 5],
            fit_chunk_matrices=_chunks()[:4],
            heldout_chunk_matrices=_chunks()[4:],
            query_discovery_checkpoint_path=_discovery_checkpoint(
                tmp_path / "final_query_discovery.joblib"
            ),
            subfold_checkpoint_paths=[stale],
            temperature=0.1,
            devices_by_bank={bank: "cpu" for bank in QUERY_BANKS},
            expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
            expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
        )


def test_query_producer_seals_saved_checkpoints_without_in_memory_discovery_argument(
    tmp_path,
):
    checkpoints = [
        _checkpoint(tmp_path / "saved_subfold_01.joblib", fold=1, train=[2, 3], validation=[0, 1]),
        _checkpoint(tmp_path / "saved_subfold_02.joblib", fold=2, train=[0, 1], validation=[2, 3]),
    ]
    discovery_checkpoint = _discovery_checkpoint(tmp_path / "saved_discovery.joblib")
    chunks = _chunks()

    written = producer._write_authenticated_query_signals_from_checkpoints(
        tmp_path / "authenticated",
        outer_fold=2,
        fit_row_ids=[0, 1, 2, 3],
        heldout_row_ids=[4, 5],
        fit_chunk_matrices=chunks[:4],
        heldout_chunk_matrices=chunks[4:],
        query_discovery_checkpoint_path=discovery_checkpoint,
        subfold_checkpoint_paths=checkpoints,
        temperature=0.1,
        devices_by_bank={bank: "cpu" for bank in QUERY_BANKS},
        expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
        expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
    )

    assert written.manifest_path.is_file()
    assert written.signal_parquet_path.is_file()
    assert written.manifest_sha256 == sha256_file(written.manifest_path)


def test_missing_subfold_query_checkpoint_fails_instead_of_using_full_fit_train_values(
    tmp_path,
):
    with pytest.raises(FileNotFoundError, match="per-subfold query checkpoint"):
        build_fold_honest_query_signals(
            outer_fold=1,
            fit_row_ids=[0, 1],
            heldout_row_ids=[2],
            fit_chunk_matrices=_chunks()[:2],
            heldout_chunk_matrices=_chunks()[2:3],
            query_discovery_checkpoint_path=_discovery_checkpoint(
                tmp_path / "final_query_discovery.joblib"
            ),
            subfold_checkpoint_paths=[tmp_path / "missing.joblib"],
            temperature=0.1,
            devices_by_bank={bank: "cpu" for bank in QUERY_BANKS},
            expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
            expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
        )


def test_subfold_checkpoint_cannot_include_outer_heldout_row(tmp_path):
    checkpoint = _checkpoint(
        tmp_path / "bad_subfold.joblib",
        fold=1,
        train=[1, 2, 3],
        validation=[0, 4],
    )
    with pytest.raises(ValueError, match="exactly partition outer training"):
        build_fold_honest_query_signals(
            outer_fold=1,
            fit_row_ids=[0, 1, 2, 3],
            heldout_row_ids=[4, 5],
            fit_chunk_matrices=_chunks()[:4],
            heldout_chunk_matrices=_chunks()[4:],
            query_discovery_checkpoint_path=_discovery_checkpoint(
                tmp_path / "final_query_discovery.joblib"
            ),
            subfold_checkpoint_paths=[checkpoint],
            temperature=0.1,
            devices_by_bank={bank: "cpu" for bank in QUERY_BANKS},
            expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
            expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
        )


def test_query_script_reuses_dedicated_authenticated_artifact_without_touching_legacy(
    tmp_path,
):
    output_dir = tmp_path / "query_run"
    output_dir.mkdir()
    legacy_manifest = output_dir / "query_signal_manifest.json"
    legacy_manifest.write_text("legacy-running-job-output", encoding="utf-8")
    checkpoints = [
        _checkpoint(
            tmp_path / "resume_subfold_01.joblib",
            fold=1,
            train=[2, 3],
            validation=[0, 1],
        ),
        _checkpoint(
            tmp_path / "resume_subfold_02.joblib",
            fold=2,
            train=[0, 1],
            validation=[2, 3],
        ),
    ]
    chunks = _chunks()
    bundle = build_fold_honest_query_signals(
        outer_fold=1,
        fit_row_ids=[0, 1, 2, 3],
        heldout_row_ids=[4, 5],
        fit_chunk_matrices=chunks[:4],
        heldout_chunk_matrices=chunks[4:],
        query_discovery_checkpoint_path=_discovery_checkpoint(
            tmp_path / "resume_final_query_discovery.joblib"
        ),
        subfold_checkpoint_paths=checkpoints,
        temperature=0.1,
        devices_by_bank={bank: "cpu" for bank in QUERY_BANKS},
        expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
        expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
    )
    artifact = write_fold_honest_query_signal_artifact(
        producer._authenticated_query_signal_directory(output_dir),
        bundle=bundle,
    )
    summary = {
        "schema_version": "test-stage-summary-v1",
        **producer._authenticated_query_signal_summary_fields(artifact),
    }
    producer._write_json(output_dir / "query_evidence_stage_summary.json", summary)

    reused = producer._reuse_authenticated_query_signal_artifact(
        output_dir,
        expected_outer_fold=1,
        expected_split_fingerprint=bundle.audit["split_fingerprint"],
        expected_outer_train_row_ids=[0, 1, 2, 3],
        expected_outer_heldout_row_ids=[4, 5],
        expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
        expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
    )

    assert reused is not None
    assert reused.manifest_sha256 == artifact.manifest_sha256
    assert reused.signal_parquet_sha256 == artifact.signal_parquet_sha256
    assert reused.manifest_path.parent.name == "authenticated_query_signals"
    assert legacy_manifest.read_text(encoding="utf-8") == "legacy-running-job-output"
