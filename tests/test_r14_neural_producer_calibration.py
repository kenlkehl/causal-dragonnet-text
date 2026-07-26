from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_r14_neural_producer_calibration as calibration_module
from scripts.run_r14_neural_producer_calibration import (
    _EXPECTED_CANDIDATE_PARALLELISM,
    _EXPECTED_TASK_COUNTS,
    _ObserveAdamWPrefix,
    _PrefixFinished,
    _PreparedNeuralInnerTask,
    _adamw_state_observation,
    _array_sha256,
    _barrier_wrapped_tasks,
    _compare_prefix_outputs,
    _gpu_summary,
    _maximum_overlap,
    _interval_union_seconds,
    _phase_summary,
    _publish_parameter_bundle,
    _throughput_ratios,
    _validated_optimizer_state_proof,
    _validate_main_args,
    build_parser,
)


def _required_cli(tmp_path: Path) -> list[str]:
    return [
        "--prepared-context-manifest",
        str((tmp_path / "prepared_stage1_context_manifest.json").resolve()),
        "--source-snapshot-root",
        str((tmp_path / "snapshot").resolve()),
        "--output-root",
        str((tmp_path / "fresh").resolve()),
        "--htr-model-path",
        str((tmp_path / "htr").resolve()),
        "--ordinary-full-byte-cache-fallback",
        "true",
        "--candidate-device",
        "cuda:0",
        "--candidate-device",
        "cuda:1",
        "--cpu-budget",
        "64",
        "--warmup-optimizer-steps",
        "1",
        "--measured-optimizer-steps",
        "32",
        "--prefix-relative-tolerance",
        "3e-5",
        "--prefix-absolute-tolerance",
        "3e-6",
        "--gpu-max-allocation-fraction",
        "0.85",
        "--gpu-minimum-headroom-bytes",
        str(6 * 1024**3),
        "--gpu-sample-interval-seconds",
        "0.001",
        "--minimum-throughput-ratio",
        "1.5",
        "--candidate-slot-cap-per-device",
        "2",
        "--htr-training-batch-size",
        "8",
        "--htr-sentence-encoder-batch-size",
        "16",
        "--htr-data-loader-workers",
        "8",
        "--htr-candidate-fold-parallelism",
        "4",
        "--htr-fold-parallel-backend",
        "processes",
        "--htr-reuse-tokenizer-and-chunk-plans",
        "true",
        "--htr-chunk-plan-cache-max-entries",
        "1000",
        "--htr-tokenized-chunk-cache-max-entries",
        "40000",
        "--neural-candidate-inner-fold-parallelism",
        "4",
        "--neural-fold-parallel-backend",
        "processes",
        "--neural-candidate-bank-parallelism",
        "3",
        "--neural-worker-cpu-threads",
        "1",
    ]


def test_parser_closes_bounded_prefix_and_resource_contract(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    args = parser.parse_args(_required_cli(tmp_path))
    assert _validate_main_args(args) == ("cuda:0", "cuda:1")
    assert args.warmup_optimizer_steps == 1
    assert args.measured_optimizer_steps == 32
    assert args.cpu_budget == 64
    assert _EXPECTED_TASK_COUNTS == {
        "htr_nuisance": 5,
        "htr_effect": 5,
        "matched_pair_htr": 5,
        "neural_inner_folds": 5,
        "neural_final_banks": 3,
    }
    assert _EXPECTED_CANDIDATE_PARALLELISM == {
        "htr_nuisance": 4,
        "htr_effect": 4,
        "matched_pair_htr": 4,
        "neural_inner_folds": 4,
        "neural_final_banks": 3,
    }

    invalid = _required_cli(tmp_path)
    invalid[invalid.index("0.001")] = "0.2"
    with pytest.raises(ValueError, match="sampling interval"):
        _validate_main_args(parser.parse_args(invalid))

    invalid = _required_cli(tmp_path)
    invalid[invalid.index("32")] = "3"
    with pytest.raises(ValueError, match="32 measured"):
        _validate_main_args(parser.parse_args(invalid))

    invalid = _required_cli(tmp_path)
    position = invalid.index("--candidate-slot-cap-per-device")
    invalid[position + 1] = "3"
    with pytest.raises(ValueError, match="exactly two safe slots"):
        _validate_main_args(parser.parse_args(invalid))


def test_main_validation_refuses_old_complete_owner_shape(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    values = _required_cli(tmp_path)
    position = values.index("--warmup-optimizer-steps")
    del values[position : position + 2]
    with pytest.raises(SystemExit):
        parser.parse_args(values)

    values = _required_cli(tmp_path)
    values[values.index("64")] = "65"
    with pytest.raises(ValueError, match="cannot exceed 64"):
        _validate_main_args(parser.parse_args(values))


def test_neural_nuisance_preparation_uses_five_overlapping_spawn_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = tuple(object() for _index in range(5))
    outcomes = tuple(
        (
            _PreparedNeuralInnerTask(
                canonical_task=task,
                train_e=np.asarray([index], dtype=np.float64),
                train_m=np.asarray([index + 1], dtype=np.float64),
                validation_e=np.asarray([index + 2], dtype=np.float64),
                validation_m=np.asarray([index + 3], dtype=np.float64),
                nuisance_identity={"task": index},
                complete_input_plan_content_sha256="a" * 64,
            ),
            {
                "worker_pid": 10_000 + index,
                "started_monotonic_ns": 100 + index,
                "finished_monotonic_ns": 200 + index,
            },
        )
        for index, task in enumerate(tasks)
    )
    observed: dict[str, object] = {}

    class FakeExecutor:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def __enter__(self):
            return self

        def map(self, function, values, *, chunksize):
            observed["function"] = function
            observed["values"] = tuple(values)
            observed["chunksize"] = chunksize
            return outcomes

        def __exit__(self, *_exc):
            return None

    ticks = iter((1, 300))
    monkeypatch.setattr(
        calibration_module.concurrent.futures,
        "ProcessPoolExecutor",
        FakeExecutor,
    )
    monkeypatch.setattr(
        calibration_module.time,
        "monotonic_ns",
        lambda: next(ticks),
    )
    prepared, attestation = (
        calibration_module._prepare_neural_inner_nuisance(
            tasks,
            cpu_budget=64,
        )
    )
    assert [value.canonical_task for value in prepared] == list(tasks)
    assert observed["max_workers"] == 5
    assert observed["chunksize"] == 1
    assert observed["values"] == tasks
    assert attestation["backend"] == "spawn_processes"
    assert attestation["configured_worker_count"] == 5
    assert attestation["maximum_concurrent_workers"] == 5


def test_five_canonical_tasks_use_four_party_and_one_party_waves(
    tmp_path: Path,
) -> None:
    class Barrier:
        def __init__(self, parties: int) -> None:
            self.parties = parties

    class Context:
        def Barrier(self, parties: int) -> Barrier:
            return Barrier(parties)

    wrapped = _barrier_wrapped_tasks(
        context=Context(),
        run_root=tmp_path,
        phase="neural_inner_folds",
        tasks=tuple(range(5)),
        identities=tuple({"index": index} for index in range(5)),
        parallelism=4,
        warmup_steps=1,
        measured_steps=32,
    )
    assert len({id(value.ready_barrier) for value in wrapped[:4]}) == 1
    assert wrapped[0].ready_barrier.parties == 4
    assert wrapped[4].ready_barrier.parties == 1
    assert wrapped[4].ready_barrier is not wrapped[0].ready_barrier


def test_primary_htr_calibration_covers_barrier_bound_effect_tasks() -> None:
    from oci.inference.role_neutral_htr_group_execution import (
        _EffectFoldTask,
    )

    plan_sha256 = "9" * 64
    config_payload = {
        "effect_objectives": ["squared_r_loss"],
        "effect_folds": 5,
        "effect_epochs": 120,
    }

    class Config:
        def as_dict(self) -> dict[str, object]:
            return copy.deepcopy(config_payload)

    fit_positions = np.asarray([0, 1, 2], dtype=np.int64)
    eligible_fit_positions = np.asarray([0, 2], dtype=np.int64)
    validation_positions = np.asarray([3], dtype=np.int64)
    y_residual = np.asarray([0.2, -0.4, 0.6, -0.8], dtype=np.float64)
    t_residual = np.asarray([0.5, -0.5, 0.25, -0.25], dtype=np.float64)
    pseudo_outcome = y_residual / t_residual
    task = _EffectFoldTask(
        objective="squared_r_loss",
        fold=1,
        split_seed=101,
        model_seed=202,
        fit_positions=fit_positions,
        eligible_fit_positions=eligible_fit_positions,
        validation_positions=validation_positions,
        y_residual=y_residual,
        t_residual=t_residual,
        pseudo_outcome=pseudo_outcome,
        config=Config(),
        model_marker="/authenticated/htr",
        operational_controls=None,
        text_authority=SimpleNamespace(
            materialized_plan=SimpleNamespace(
                content_sha256=plan_sha256,
            )
        ),
    )

    assert calibration_module._PHASES[:2] == (
        "htr_nuisance",
        "htr_effect",
    )
    assert _EXPECTED_TASK_COUNTS["htr_nuisance"] == 5
    assert _EXPECTED_TASK_COUNTS["htr_effect"] == 5
    assert _EXPECTED_CANDIDATE_PARALLELISM["htr_effect"] == 4
    assert callable(calibration_module._WORKER_BY_PHASE["htr_effect"])

    source_e_sha256 = "7" * 64
    source_m_sha256 = "8" * 64
    prepared = calibration_module._PreparedHTREffectTask(
        canonical_task=task,
        source_nuisance_oof_e_sha256=source_e_sha256,
        source_nuisance_oof_m_sha256=source_m_sha256,
    )
    identity = calibration_module._htr_effect_task_identity(prepared)
    assert {
        "objective": "squared_r_loss",
        "fold": 1,
        "split_seed": 101,
        "model_seed": 202,
        "fit_positions_sha256": _array_sha256(fit_positions),
        "eligible_fit_positions_sha256": _array_sha256(
            eligible_fit_positions
        ),
        "validation_positions_sha256": _array_sha256(
            validation_positions
        ),
        "y_residual_sha256": _array_sha256(y_residual),
        "t_residual_sha256": _array_sha256(t_residual),
        "pseudo_outcome_sha256": _array_sha256(pseudo_outcome),
        "config": config_payload,
        "complete_plan_content_sha256": plan_sha256,
    }.items() <= identity.items()
    assert identity["source_nuisance_oof_e_sha256"] == source_e_sha256
    assert identity["source_nuisance_oof_m_sha256"] == source_m_sha256
    assert identity["prefix_conditioned_on_bounded_nuisance_oof"] is True


def _production_scored_candidate(
    query: np.ndarray | None = None,
) -> dict[str, object]:
    value = np.asarray(
        [0.25, -0.5, 1.0] if query is None else query,
        dtype=np.float32,
    )
    return {
        "candidate_id": "treatment_fold_01_query_001",
        "bank": "treatment",
        "subfold": 1,
        "query": value.tolist(),
        "query_dtype": value.dtype.str,
        "query_shape": [int(size) for size in value.shape],
        "query_sha256": _array_sha256(value),
        "train_standardized_score": 0.75,
        "validation_audit_standardized_score": -0.25,
        "validation_audit_only_not_used_for_gating": True,
        "query_drift": 0.125,
        "calibration_prefix_derived_with_production_scoring": True,
    }


def test_candidate_query_evidence_requires_real_production_scoring(
    tmp_path: Path,
) -> None:
    query = np.asarray([0.25, -0.5, 1.0], dtype=np.float32)
    candidate = _production_scored_candidate(query)
    matrix = (
        calibration_module._validated_calibration_candidate_queries(
            [candidate]
        )
    )
    assert matrix.dtype == np.dtype(np.float32)
    assert matrix.shape == (1, 3)
    published = calibration_module._write_self_hashed_json(
        tmp_path / "candidate.json",
        {"candidates": [candidate]},
    )
    assert published["candidates"][0]["query"] == [0.25, -0.5, 1.0]

    forged = copy.deepcopy(candidate)
    forged["query_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="failed authentication"):
        calibration_module._validated_calibration_candidate_queries(
            [forged]
        )

    placeholder = copy.deepcopy(candidate)
    placeholder.pop(
        "calibration_prefix_derived_with_production_scoring"
    )
    placeholder["train_standardized_score"] = 0.0
    placeholder["calibration_prefix_score_placeholder"] = True
    with pytest.raises(RuntimeError, match="schema|authentication|scoring"):
        calibration_module._validated_calibration_candidate_queries(
            [placeholder]
        )

    unproved = copy.deepcopy(candidate)
    unproved[
        "calibration_prefix_derived_with_production_scoring"
    ] = False
    with pytest.raises(RuntimeError, match="authentication|scoring"):
        calibration_module._validated_calibration_candidate_queries(
            [unproved]
        )

    with pytest.raises(TypeError, match="ndarray"):
        calibration_module._write_self_hashed_json(
            tmp_path / "raw-array.json",
            {"query": query},
        )


def test_prefix_candidate_scores_call_production_scoring_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from oci.inference import neural_cohort_witness as witness
    from oci.inference import neural_query_discovery_runtime as runtime

    activation_calls: list[tuple[int, np.ndarray]] = []
    audit_calls: list[np.ndarray] = []
    moment_calls: list[tuple[object, ...]] = []
    vector_norm_calls: list[tuple[object, ...]] = []

    def fake_activations(
        chunks,
        queries,
        *,
        temperature,
        device,
        patient_batch_size,
    ):
        assert temperature == 0.25
        assert device == "cpu"
        assert patient_batch_size == 2
        activation_calls.append(
            (len(chunks), np.asarray(queries, dtype=np.float32).copy())
        )
        offset = 10.0 if len(chunks) == 2 else 0.0
        return (
            np.arange(len(chunks) * len(queries), dtype=np.float32)
            .reshape(len(chunks), len(queries))
            + offset
        )

    def fake_validation_audit(activations, target, *, binary):
        assert binary is True
        assert np.array_equal(target, np.asarray([0.0, 1.0]))
        audit_calls.append(np.asarray(activations).copy())
        return {
            "standardized_scores": np.asarray(
                [-2.0, 3.0],
                dtype=np.float64,
            )
        }

    def fake_train_moments(*args, **kwargs):
        assert kwargs == {"epsilon": 1e-6}
        moment_calls.append(args)
        return torch.as_tensor([4.5, -1.5], dtype=torch.float32)

    original_vector_norm = torch.linalg.vector_norm

    def observed_vector_norm(*args, **kwargs):
        vector_norm_calls.append(args)
        return original_vector_norm(*args, **kwargs)

    monkeypatch.setattr(
        witness,
        "soft_retrieval_activations",
        fake_activations,
    )
    monkeypatch.setattr(
        witness,
        "standardized_direct_target_contrasts",
        fake_validation_audit,
    )
    monkeypatch.setattr(
        witness,
        "_torch_standardized_moments",
        fake_train_moments,
    )
    monkeypatch.setattr(
        runtime,
        "_witness_config",
        lambda *_args, **_kwargs: SimpleNamespace(epsilon=1e-6),
    )
    monkeypatch.setattr(
        torch.linalg,
        "vector_norm",
        observed_vector_norm,
    )

    initial = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    terminal = np.asarray([[0.8, 0.6], [0.6, 0.8]], dtype=np.float32)
    natural_train_activations = np.arange(
        8,
        dtype=np.float32,
    ).reshape(4, 2)
    natural_train_scores = np.asarray([4.5, -1.5], dtype=np.float64)
    natural_query_drift = np.linalg.norm(
        terminal - initial,
        axis=1,
    ).astype(np.float64)
    candidate_queries, train_scores, validation_scores, drift, proof = (
        calibration_module._production_prefix_candidate_scores(
            bank="treatment",
            train_chunks=tuple(
                np.ones((1, 2), dtype=np.float32) for _index in range(4)
            ),
            validation_chunks=tuple(
                np.ones((1, 2), dtype=np.float32) for _index in range(2)
            ),
            train_treatment=np.asarray(
                [0.0, 1.0, 0.0, 1.0],
                dtype=np.float64,
            ),
            train_outcome=np.asarray(
                [0.0, 0.0, 1.0, 1.0],
                dtype=np.float64,
            ),
            validation_treatment=np.asarray(
                [0.0, 1.0],
                dtype=np.float64,
            ),
            validation_outcome=np.asarray(
                [1.0, 0.0],
                dtype=np.float64,
            ),
            train_e=np.full(4, 0.5, dtype=np.float64),
            train_m=np.full(4, 0.5, dtype=np.float64),
            validation_e=np.full(2, 0.5, dtype=np.float64),
            validation_m=np.full(2, 0.5, dtype=np.float64),
            outcome_binary=True,
            query_config=SimpleNamespace(
                temperature=0.25,
                retrieval_patient_batch_size=2,
            ),
            initial_queries=initial,
            terminal_queries=terminal,
            production_queries=terminal.copy(),
            production_train_activations=natural_train_activations,
            production_train_standardized_scores=natural_train_scores,
            production_query_drift=natural_query_drift,
            production_constant_effect=None,
            device="cpu",
        )
    )

    assert np.array_equal(candidate_queries, terminal)
    assert np.array_equal(train_scores, natural_train_scores)
    assert np.array_equal(validation_scores, np.asarray([-2.0, 3.0]))
    assert np.array_equal(drift, natural_query_drift)
    assert [count for count, _queries in activation_calls] == [4, 2]
    assert len(audit_calls) == len(moment_calls) == len(vector_norm_calls) == 1
    assert proof[
        "production_scoring_recomputed_after_prefix_outside_measured_window"
    ] is True
    assert proof["training_score_policy"] == (
        "torch_population_std_weighted_center_production_v1"
    )
    assert proof["validation_score_policy"] == (
        "numpy_sample_std_production_audit_v1"
    )
    proof_body = {
        key: value for key, value in proof.items() if key != "content_sha256"
    }
    assert proof["content_sha256"] == calibration_module._sha256_json(
        proof_body
    )


def _self_hashed_complete_input_plan(producer: str) -> tuple[dict, str]:
    body = {
        "schema_version": "production_r14_complete_input_plan_v1",
        "producer": producer,
        "complete_row_and_chunk_coverage": True,
        "semantic_truncation_applied": False,
    }
    digest = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {**body, "content_sha256": digest}, digest


def test_matched_and_neural_identities_bind_independent_complete_plans() -> None:
    matched_plan, matched_digest = _self_hashed_complete_input_plan(
        "matched_pair_htr"
    )
    neural_plan, neural_digest = _self_hashed_complete_input_plan(
        "neural_queries"
    )
    assert matched_plan["content_sha256"] != neural_plan["content_sha256"]

    matched_canonical = SimpleNamespace(
        objective="matched_pair_uplift",
        fold=1,
        split_seed=11,
        htr_seed=12,
        owner_scope_seed=13,
        owner_fit_row_ids=(0, 1, 2, 3),
        fit_texts=("zero", "one", "two", "three"),
        treatment=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        outcome=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        propensity_probability=np.asarray(
            [0.2, 0.7, 0.3, 0.8],
            dtype=np.float64,
        ),
        outcome_nuisance_probability=np.asarray(
            [0.1, 0.4, 0.6, 0.9],
            dtype=np.float64,
        ),
        fit_positions=np.asarray([0, 1, 2], dtype=np.int64),
        validation_positions=np.asarray([3], dtype=np.int64),
        config={"htr_batch_size": 8},
    )
    matched = calibration_module._PreparedMatchedTask(
        canonical_task=matched_canonical,
        complete_input_plan_content_sha256=matched_digest,
    )
    matched_identity = calibration_module._matched_task_identity(matched)
    assert (
        matched_identity["complete_input_plan_content_sha256"]
        == matched_digest
    )

    class QueryConfig:
        def to_dict(self) -> dict[str, object]:
            return {"query_epochs": 120, "final_refit_epochs": 120}

    base_arguments = {
        "fold": 1,
        "seed": 23,
        "train_indices": np.asarray([0, 1, 2], dtype=np.int64),
        "validation_indices": np.asarray([3], dtype=np.int64),
        "row_ids": (0, 1, 2, 3),
        "texts": ("zero", "one", "two", "three"),
        "treatment": np.asarray(
            [0.0, 1.0, 0.0, 1.0],
            dtype=np.float64,
        ),
        "outcome": np.asarray(
            [0.0, 0.0, 1.0, 1.0],
            dtype=np.float64,
        ),
        "outcome_binary": True,
        "parent_input_binding_sha256": "7" * 64,
        "nuisance_views": ({"name": "word"},),
        "nuisance_folds": 5,
        "nuisance_stack_config": {"folds": 5},
        "config": QueryConfig(),
    }
    neural_inner = _PreparedNeuralInnerTask(
        canonical_task=SimpleNamespace(arguments=base_arguments),
        train_e=np.asarray([0.2, 0.7, 0.3], dtype=np.float64),
        train_m=np.asarray([0.1, 0.4, 0.6], dtype=np.float64),
        validation_e=np.asarray([0.8], dtype=np.float64),
        validation_m=np.asarray([0.9], dtype=np.float64),
        nuisance_identity={"content_sha256": "8" * 64},
        complete_input_plan_content_sha256=neural_digest,
    )
    inner_identity = calibration_module._neural_inner_task_identity(
        neural_inner
    )
    assert (
        inner_identity["complete_input_plan_content_sha256"]
        == neural_digest
    )
    assert inner_identity["outcome_binary"] is True
    assert inner_identity["parent_input_binding_sha256"] == "7" * 64
    assert inner_identity["nuisance_folds"] == 5
    assert inner_identity["nuisance_views_sha256"] == (
        calibration_module._sha256_json([{"name": "word"}])
    )
    assert inner_identity["nuisance_stack_config_sha256"] == (
        calibration_module._sha256_json({"folds": 5})
    )

    final_arguments = {
        "bank": "treatment",
        "bank_index": 0,
        "seed": 23,
        "row_ids": base_arguments["row_ids"],
        "texts": base_arguments["texts"],
        "treatment": base_arguments["treatment"],
        "outcome": base_arguments["outcome"],
        "outcome_binary": True,
        "fit_e": np.asarray([0.2, 0.7, 0.3, 0.8], dtype=np.float64),
        "fit_m": np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64),
        "candidates": [_production_scored_candidate()],
        "config": base_arguments["config"],
    }
    neural_final = calibration_module._PreparedNeuralFinalTask(
        canonical_task=SimpleNamespace(arguments=final_arguments),
        complete_input_plan_content_sha256=neural_digest,
    )
    final_identity = calibration_module._neural_final_task_identity(
        neural_final
    )
    assert (
        final_identity["complete_input_plan_content_sha256"]
        == neural_digest
    )
    assert final_identity["outcome_binary"] is True
    assert (
        matched_identity["complete_input_plan_content_sha256"]
        != inner_identity["complete_input_plan_content_sha256"]
    )

    with pytest.raises(ValueError, match="input-plan identity is invalid"):
        calibration_module._matched_task_identity(
            calibration_module._PreparedMatchedTask(
                canonical_task=matched_canonical,
                complete_input_plan_content_sha256="not-a-sha256",
            )
        )
    with pytest.raises(ValueError, match="input-plan identity is invalid"):
        calibration_module._neural_inner_task_identity(
            _PreparedNeuralInnerTask(
                canonical_task=neural_inner.canonical_task,
                train_e=neural_inner.train_e,
                train_m=neural_inner.train_m,
                validation_e=neural_inner.validation_e,
                validation_m=neural_inner.validation_m,
                nuisance_identity=neural_inner.nuisance_identity,
                complete_input_plan_content_sha256="not-a-sha256",
            )
        )
    with pytest.raises(ValueError, match="input-plan identity is invalid"):
        calibration_module._neural_final_task_identity(
            calibration_module._PreparedNeuralFinalTask(
                canonical_task=neural_final.canonical_task,
                complete_input_plan_content_sha256="not-a-sha256",
            )
        )


def _stateful_prefix(start: int, finish: int, *, measured: int = 3) -> dict:
    layout = '[{"parameter_index":0}]'
    layout_sha256 = hashlib.sha256(layout.encode("utf-8")).hexdigest()

    def boundary(expected_step: int) -> dict:
        return {
            "schema_version": (
                "adamw_optimizer_state_boundary_observation_v2"
            ),
            "expected_optimizer_step": expected_step,
            "optimizer_parameter_count": 1,
            "state_parameter_count": 1,
            "stateless_parameter_count": 0,
            "state_object_count": 3,
            "state_tensor_count": 3,
            "state_tensor_bytes": 12,
            "all_optimizer_parameters_classified": True,
            "all_stateless_parameters_have_no_gradient": True,
            "all_stateless_parameters_have_no_optimizer_state": True,
            "all_stateful_parameters_have_finite_gradients": True,
            "all_required_state_keys_observed": True,
            "all_state_tensors_finite": True,
            "object_layout_canonical_json": layout,
            "object_layout_sha256": layout_sha256,
        }

    return {
        "warmup_optimizer_steps": 1,
        "measured_optimizer_steps": measured,
        "completed_warmup_optimizer_steps_at_interval_start": 1,
        "optimizer_state_verified_monotonic_ns": start - 3,
        "ready_wait_started_monotonic_ns": start - 2,
        "ready_wait_finished_monotonic_ns": start - 1,
        "measured_started_monotonic_ns": start,
        "measured_finished_monotonic_ns": finish,
        "optimizer_state_finish_verified_monotonic_ns": finish + 1,
        "optimizer_state_at_interval_start": boundary(1),
        "optimizer_state_at_interval_finish": boundary(1 + measured),
        "optimizer_state_persistence_observed": True,
    }


def test_measured_overlap_and_phase_throughput_use_optimizer_intervals() -> None:
    assert _maximum_overlap([(10, 30), (20, 40), (25, 35)]) == 3
    assert _interval_union_seconds([(10, 20), (30, 50)]) == pytest.approx(
        30e-9
    )
    results = [
        {
            "prefix_output": _stateful_prefix(10, 40)
        },
        {
            "prefix_output": _stateful_prefix(10, 40)
        },
    ]
    execution = {
        "maximum_concurrent_leases": 2,
        "task_intervals": [
            {
                "device": "cuda:0",
                "gpu_peak_allocated_bytes": 10,
                "gpu_peak_reserved_bytes": 12,
            },
            {
                "device": "cuda:1",
                "gpu_peak_allocated_bytes": 11,
                "gpu_peak_reserved_bytes": 13,
            },
        ],
    }
    summary = _phase_summary(
        results,
        execution_attestation=execution,
        expected_parallelism=2,
    )
    assert summary["measured_optimizer_steps"] == 6
    assert summary["measured_optimizer_maximum_concurrency"] == 2
    assert summary["aggregate_measured_optimizer_steps_per_second"] == pytest.approx(
        2e8
    )

    serialized = [dict(results[0]), dict(results[1])]
    serialized[1] = {
        "prefix_output": _stateful_prefix(41, 60)
    }
    with pytest.raises(RuntimeError, match="did not overlap"):
        _phase_summary(
            serialized,
            execution_attestation=execution,
            expected_parallelism=2,
        )


def _complete_parameter_prefix(
    root: Path,
    *,
    phase: str,
    index: int,
    parameter_delta: float,
) -> dict:
    values = np.linspace(-0.5, 0.5, num=65, dtype=np.float32)
    values[33] += np.float32(parameter_delta)
    relative = (
        Path("parameter_bundles") / phase / f"task_{index:03d}.bin"
    ).as_posix()
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = _publish_parameter_bundle(
        path,
        relative_path=relative,
        arrays=(values,),
        group_indices=(0,),
        parameter_indices_within_group=(0,),
    )
    return {
        "terminal_parameter_count": 1,
        "terminal_parameter_sha256s": [_array_sha256(values)],
        "terminal_parameter_element_counts": [65],
        "terminal_parameter_all_finite": [True],
        "terminal_all_parameters_finite": True,
        "terminal_parameter_group_indices": [0],
        "terminal_parameter_indices_within_group": [0],
        "terminal_parameter_bundle": bundle,
    }


def _fixture_run(
    rate: float,
    *,
    root: Path | None = None,
    delta: float = 0.0,
    parameter_delta: float = 0.0,
) -> dict:
    if root is not None:
        root.mkdir(parents=True, exist_ok=False)
        root = root.resolve(strict=True)
    phases = {}
    for phase, count in _EXPECTED_TASK_COUNTS.items():
        exact_terminal = phase in {
            "htr_nuisance",
            "htr_effect",
            "matched_pair_htr",
        }
        phases[phase] = {
            "aggregate_measured_optimizer_steps_per_second": rate,
            "prefix_results": [
                {
                    "canonical_index": index,
                    "canonical_identity": {
                        "task": index,
                        "phase": phase,
                    },
                    "prefix_output": {
                        "measured_started_monotonic_ns": 10,
                        "measured_finished_monotonic_ns": 20,
                        "measured_optimizer_steps": 3,
                        "loss_prefix": [0.5 + delta, 0.25],
                        "terminal_parameter_shapes": [
                            [65] if exact_terminal else [2]
                        ],
                        "terminal_parameter_dtypes": ["<f4"],
                        "terminal_parameter_samples": [
                            0.1 + (0.0 if exact_terminal else delta),
                            0.2,
                        ],
                        **(
                            {
                                **_complete_parameter_prefix(
                                    root,
                                    phase=phase,
                                    index=index,
                                    parameter_delta=parameter_delta,
                                )
                            }
                            if exact_terminal and root is not None
                            else {
                                "terminal_parameter_count": 1,
                                "terminal_parameter_sha256s": ["a" * 64],
                                "terminal_parameter_element_counts": [65],
                                "terminal_parameter_all_finite": [True],
                                "terminal_all_parameters_finite": True,
                                "terminal_parameter_group_indices": [0],
                                "terminal_parameter_indices_within_group": [0],
                                "terminal_parameter_bundle": {},
                            }
                            if exact_terminal
                            else {}
                        ),
                        **(
                            {
                                "fit_pair_count": 23,
                                "fit_pair_text_sha256": "b" * 64,
                            }
                            if phase == "matched_pair_htr"
                            else {}
                        ),
                    },
                }
                for index in range(count)
            ],
        }
    return {
        "root": str(root) if root is not None else "/not-used",
        "phases": phases,
    }


def test_prefix_equality_checks_all_complete_values_and_all_run_pairs(
    tmp_path: Path,
) -> None:
    baseline = _fixture_run(10.0, root=tmp_path / "baseline")
    candidate_a = _fixture_run(
        20.0,
        root=tmp_path / "candidate-a",
        delta=1e-7,
        parameter_delta=1e-7,
    )
    candidate_b = _fixture_run(
        18.0,
        root=tmp_path / "candidate-b",
        delta=2e-7,
        parameter_delta=2e-7,
    )
    for left, right in (
        (baseline, candidate_a),
        (baseline, candidate_b),
        (candidate_a, candidate_b),
    ):
        summary = _compare_prefix_outputs(
            left,
            right,
            relative_tolerance=3e-5,
            absolute_tolerance=3e-6,
        )
        assert summary["floating_values_compared"] > 0
        assert summary["complete_parameter_tensors_compared"] == 15
        assert (
            summary[
                "cross_run_parameter_sha256_differences_tolerance_accepted"
            ]
            == 15
        )

    changed = _fixture_run(20.0, root=tmp_path / "changed")
    changed["phases"]["htr_nuisance"]["prefix_results"][0][
        "canonical_identity"
    ]["task"] = 999
    with pytest.raises(ValueError, match="canonical identity"):
        _compare_prefix_outputs(
            baseline,
            changed,
            relative_tolerance=3e-5,
            absolute_tolerance=3e-6,
        )

    unsampled_divergence = _fixture_run(
        20.0,
        root=tmp_path / "unsampled-divergence",
        parameter_delta=1e-2,
    )
    with pytest.raises(ValueError, match="complete parameter tolerance"):
        _compare_prefix_outputs(
            baseline,
            unsampled_divergence,
            relative_tolerance=3e-5,
            absolute_tolerance=3e-6,
        )

    tampered_sha = _fixture_run(
        20.0,
        root=tmp_path / "tampered-sha",
    )
    tampered_sha["phases"]["htr_nuisance"]["prefix_results"][0][
        "prefix_output"
    ]["terminal_parameter_sha256s"][0] = "c" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        _compare_prefix_outputs(
            baseline,
            tampered_sha,
            relative_tolerance=3e-5,
            absolute_tolerance=3e-6,
        )

    corrupt_bytes = _fixture_run(
        20.0,
        root=tmp_path / "corrupt-bytes",
    )
    corrupt_prefix = corrupt_bytes["phases"]["htr_nuisance"][
        "prefix_results"
    ][0]["prefix_output"]
    corrupt_path = (
        Path(corrupt_bytes["root"])
        / corrupt_prefix["terminal_parameter_bundle"]["relative_path"]
    )
    payload = bytearray(corrupt_path.read_bytes())
    payload[0] ^= 1
    os.chmod(corrupt_path, 0o644)
    corrupt_path.write_bytes(payload)
    os.chmod(corrupt_path, 0o444)
    with pytest.raises(ValueError, match="bundle failed authentication"):
        _compare_prefix_outputs(
            baseline,
            corrupt_bytes,
            relative_tolerance=3e-5,
            absolute_tolerance=3e-6,
        )

    changed_pairs = _fixture_run(
        20.0,
        root=tmp_path / "changed-pairs",
    )
    changed_pairs["phases"]["matched_pair_htr"]["prefix_results"][0][
        "prefix_output"
    ]["fit_pair_count"] += 1
    with pytest.raises(ValueError, match="changed discrete state"):
        _compare_prefix_outputs(
            baseline,
            changed_pairs,
            relative_tolerance=3e-5,
            absolute_tolerance=3e-6,
        )


def test_phase_throughput_uses_conservative_candidate_repetition() -> None:
    baseline = _fixture_run(10.0)
    candidate_a = _fixture_run(20.0)
    candidate_b = _fixture_run(16.0)
    ratios = _throughput_ratios(
        baseline,
        (candidate_a, candidate_b),
        minimum_ratio=1.5,
    )
    assert set(ratios) == set(_EXPECTED_TASK_COUNTS)
    assert all(
        row["single_gpu_baseline_throughput_ratio"] == pytest.approx(1.6)
        and row["throughput_threshold_met"] is True
        for row in ratios.values()
    )


def test_terminal_acceptance_is_the_conjunction_of_every_required_gate() -> None:
    gate_names = (
        "equality_accepted",
        "memory_accepted",
        "all_phase_throughput_thresholds_met",
        "required_kernel_coverage_accepted",
        "complete_input_plan_bindings_accepted",
        "neural_candidate_provenance_accepted",
    )
    decide = calibration_module._calibration_acceptance
    accepted = decide(**dict.fromkeys(gate_names, True))
    assert accepted == {
        **dict.fromkeys(gate_names, True),
        "policy": "all_required_gates_conjunction_v1",
        "calibration_valid": True,
        "multi_gpu_step_throughput_acceleration_claimed": True,
        "deployment_recommendation": (
            "proceed_to_first_complete_owner_validation_gate"
        ),
        "process_exit_code": 0,
    }

    for rejected_gate in gate_names:
        gates = dict.fromkeys(gate_names, True)
        gates[rejected_gate] = False
        rejected = decide(**gates)
        assert rejected == {
            **gates,
            "policy": "all_required_gates_conjunction_v1",
            "calibration_valid": False,
            "multi_gpu_step_throughput_acceleration_claimed": False,
            "deployment_recommendation": "do_not_adopt_kernel_calibration",
            "process_exit_code": 2,
        }


def test_memory_acceptance_maxes_host_and_concurrent_child_bounds() -> None:
    phase_rows = {
        phase: {
            "task_execution_attestation": {
                "task_intervals": [],
            }
        }
        for phase in _EXPECTED_TASK_COUNTS
    }
    phase_rows["htr_nuisance"]["task_execution_attestation"][
        "task_intervals"
    ] = [
        {
            "device": "cuda:0",
            "started_monotonic_ns": 100,
            "finished_monotonic_ns": 300,
            "gpu_peak_allocated_bytes": 100,
            "gpu_peak_reserved_bytes": 250,
        },
        {
            "device": "cuda:0",
            "started_monotonic_ns": 150,
            "finished_monotonic_ns": 250,
            "gpu_peak_allocated_bytes": 200,
            "gpu_peak_reserved_bytes": 300,
        },
        {
            "device": "cuda:1",
            "started_monotonic_ns": 100,
            "finished_monotonic_ns": 300,
            "gpu_peak_allocated_bytes": 50,
            "gpu_peak_reserved_bytes": 75,
        },
    ]
    phase_rows["matched_pair_htr"]["task_execution_attestation"][
        "task_intervals"
    ] = [
        {
            "device": "cuda:0",
            "started_monotonic_ns": 400,
            "finished_monotonic_ns": 500,
            "gpu_peak_allocated_bytes": 400,
            "gpu_peak_reserved_bytes": 450,
        }
    ]
    samples = [
        {
            "device": device,
            "uuid": f"uuid-{device}",
            "gpu_sample_acquisition_started_monotonic_ns": started,
            "gpu_sample_acquisition_finished_monotonic_ns": finished,
            "sample_monotonic_seconds": finished / 1e9,
            "memory_total_bytes": 10_000,
            "memory_used_bytes": used,
        }
        for device, baseline, host_peak in (
            ("cuda:0", 1_000, 1_300),
            ("cuda:1", 2_000, 2_600),
        )
        for started, finished, used in (
            (40, 50, baseline),
            (150, 160, host_peak),
            (510, 520, baseline),
        )
    ]
    sampling_proof = [
        {
            "measured_started_monotonic_ns": 120,
            "measured_finished_monotonic_ns": 200,
            "phase": "htr_nuisance",
            "canonical_task_index": 0,
            "atomic_prefix_index": 0,
            "device": "cuda:0",
            "host_gpu_acquisition_windows_wholly_inside_interval": 1,
            "optimizer_state_proof": {
                "schema_version": (
                    "adamw_optimizer_state_persistence_proof_v2"
                ),
                "required_warmup_optimizer_steps": 1,
                "completed_warmup_optimizer_steps": 1,
                "measured_optimizer_steps": 3,
                "start_expected_optimizer_step": 1,
                "finish_expected_optimizer_step": 4,
                "optimizer_parameter_count": 2,
                "state_parameter_count": 2,
                "stateless_parameter_count": 0,
                "state_tensor_count": 6,
                "state_tensor_bytes": 128,
                "all_optimizer_parameters_classified": True,
                "all_stateless_parameters_have_no_gradient": True,
                "all_stateless_parameters_have_no_optimizer_state": True,
                "all_stateful_parameters_have_finite_gradients": True,
                "object_layout_sha256": "d" * 64,
                "state_verified_monotonic_ns": 90,
                "ready_wait_started_monotonic_ns": 100,
                "ready_wait_finished_monotonic_ns": 110,
                "measured_started_monotonic_ns": 120,
                "measured_finished_monotonic_ns": 200,
                "finish_state_verified_monotonic_ns": 210,
                "accepted": True,
            },
            "accepted": True,
        }
    ]
    summary = _gpu_summary(
        samples,
        phase_rows=phase_rows,
        post_warmup_sampling_rows=sampling_proof,
        sampler_completed_monotonic_ns=530,
        run_finished_monotonic_ns=540,
        devices=("cuda:0", "cuda:1"),
        maximum_fraction=0.5,
        minimum_headroom_bytes=1_000,
    )
    by_device = {row["device"]: row for row in summary["devices"]}
    gpu0 = by_device["cuda:0"]
    assert (
        gpu0["conservative_child_incremental_bound_bytes"] == 550
    )
    assert gpu0["conservative_child_allocated_peak_sum_bytes"] == 400
    assert gpu0["conservative_child_reserved_peak_sum_bytes"] == 550
    assert gpu0["pre_task_external_plus_child_bound_bytes"] == 1_550
    assert gpu0["host_peak_memory_used_bytes"] == 1_300
    assert gpu0["memory_acceptance_peak_bytes"] == 1_550
    assert gpu0["memory_acceptance_peak_source"] == (
        "external_baseline_plus_child_peak_bound"
    )
    gpu1 = by_device["cuda:1"]
    assert gpu1["pre_task_external_plus_child_bound_bytes"] == 2_075
    assert gpu1["host_peak_memory_used_bytes"] == 2_600
    assert gpu1["memory_acceptance_peak_bytes"] == 2_600
    assert gpu1["memory_acceptance_peak_source"] == "host_peak"
    assert summary["memory_safety_accepted"] is True
    assert summary[
        "host_sample_inside_every_post_warmup_optimizer_interval"
    ] is True

    forged_count = copy.deepcopy(sampling_proof)
    forged_count[0][
        "host_gpu_acquisition_windows_wholly_inside_interval"
    ] = 2
    with pytest.raises(RuntimeError, match="inside every post-warmup"):
        _gpu_summary(
            samples,
            phase_rows=phase_rows,
            post_warmup_sampling_rows=forged_count,
            sampler_completed_monotonic_ns=530,
            run_finished_monotonic_ns=540,
            devices=("cuda:0", "cuda:1"),
            maximum_fraction=0.5,
            minimum_headroom_bytes=1_000,
        )

    outside_window_samples = copy.deepcopy(samples)
    for row in outside_window_samples:
        if (
            row["device"] == "cuda:0"
            and row["gpu_sample_acquisition_started_monotonic_ns"] == 150
        ):
            row["gpu_sample_acquisition_finished_monotonic_ns"] = 205
            row["sample_monotonic_seconds"] = 205 / 1e9
    with pytest.raises(RuntimeError, match="inside every post-warmup"):
        _gpu_summary(
            outside_window_samples,
            phase_rows=phase_rows,
            post_warmup_sampling_rows=sampling_proof,
            sampler_completed_monotonic_ns=530,
            run_finished_monotonic_ns=540,
            devices=("cuda:0", "cuda:1"),
            maximum_fraction=0.5,
            minimum_headroom_bytes=1_000,
        )

    missing_state = copy.deepcopy(sampling_proof)
    del missing_state[0]["optimizer_state_proof"]
    with pytest.raises(RuntimeError, match="inside every post-warmup"):
        _gpu_summary(
            samples,
            phase_rows=phase_rows,
            post_warmup_sampling_rows=missing_state,
            sampler_completed_monotonic_ns=530,
            run_finished_monotonic_ns=540,
            devices=("cuda:0", "cuda:1"),
            maximum_fraction=0.5,
            minimum_headroom_bytes=1_000,
        )

    missing_reserved = copy.deepcopy(phase_rows)
    del missing_reserved["htr_nuisance"][
        "task_execution_attestation"
    ]["task_intervals"][0]["gpu_peak_reserved_bytes"]
    with pytest.raises(RuntimeError, match="positive Torch peaks"):
        _gpu_summary(
            samples,
            phase_rows=missing_reserved,
            post_warmup_sampling_rows=sampling_proof,
            sampler_completed_monotonic_ns=530,
            run_finished_monotonic_ns=540,
            devices=("cuda:0", "cuda:1"),
            maximum_fraction=0.5,
            minimum_headroom_bytes=1_000,
        )


def test_exact_terminal_hashes_cover_every_optimizer_parameter(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    class _Ready:
        def wait(self, timeout: float) -> None:
            assert timeout == 900.0

    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([[3.0], [4.0]]))
    optimizer = torch.optim.AdamW([first, second], lr=0.01)
    observer = _ObserveAdamWPrefix(
        device="cpu",
        ready_barrier=_Ready(),
        warmup_steps=1,
        measured_steps=1,
        stop_on_next_step=False,
        capture_full_single_tensor=False,
        capture_exact_parameter_hashes=True,
        parameter_bundle_path=str(tmp_path / "exact-parameters.bin"),
        parameter_bundle_relative_path="exact-parameters.bin",
    )
    with pytest.raises(_PrefixFinished):
        with observer:
            for _step in range(2):
                optimizer.zero_grad(set_to_none=True)
                (first.square().sum() + second.square().sum()).backward()
                optimizer.step()
    result = observer.result()
    assert result["terminal_parameter_count"] == 2
    assert result["terminal_parameter_shapes"] == [[2], [2, 1]]
    assert result["terminal_parameter_element_counts"] == [2, 2]
    assert result["terminal_parameter_all_finite"] == [True, True]
    assert result["terminal_all_parameters_finite"] is True
    assert len(result["terminal_parameter_sha256s"]) == 2
    assert all(
        len(value) == 64 for value in result["terminal_parameter_sha256s"]
    )
    assert result["terminal_parameter_bundle"]["parameter_count"] == 2
    assert (
        result["completed_warmup_optimizer_steps_at_interval_start"] == 1
    )
    assert result["optimizer_state_at_interval_start"][
        "expected_optimizer_step"
    ] == 1
    assert result["optimizer_state_at_interval_finish"][
        "expected_optimizer_step"
    ] == 2
    assert result["optimizer_state_at_interval_start"][
        "optimizer_parameter_count"
    ] == 2
    assert result["optimizer_state_at_interval_start"][
        "state_parameter_count"
    ] == 2
    assert result["optimizer_state_at_interval_start"][
        "stateless_parameter_count"
    ] == 0
    assert result["optimizer_state_persistence_observed"] is True
    assert (
        result["optimizer_state_verified_monotonic_ns"]
        <= result["ready_wait_started_monotonic_ns"]
        <= result["ready_wait_finished_monotonic_ns"]
        <= result["measured_started_monotonic_ns"]
        < result["measured_finished_monotonic_ns"]
        <= result["optimizer_state_finish_verified_monotonic_ns"]
    )


def test_optimizer_state_proof_classifies_never_used_parameter(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    class _Ready:
        def wait(self, timeout: float) -> None:
            assert timeout == 900.0

    active = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    inactive = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    optimizer = torch.optim.AdamW([active, inactive], lr=0.01)
    observer = _ObserveAdamWPrefix(
        device="cpu",
        ready_barrier=_Ready(),
        warmup_steps=1,
        measured_steps=2,
        stop_on_next_step=False,
        capture_full_single_tensor=False,
        capture_exact_parameter_hashes=True,
        parameter_bundle_path=str(tmp_path / "inactive-parameter.bin"),
        parameter_bundle_relative_path="inactive-parameter.bin",
    )
    with pytest.raises(_PrefixFinished):
        with observer:
            for _step in range(3):
                optimizer.zero_grad(set_to_none=True)
                active.square().sum().backward()
                optimizer.step()
    result = observer.result()
    start = result["optimizer_state_at_interval_start"]
    finish = result["optimizer_state_at_interval_finish"]
    for boundary, expected_step in ((start, 1), (finish, 3)):
        assert boundary["schema_version"] == (
            "adamw_optimizer_state_boundary_observation_v2"
        )
        assert boundary["expected_optimizer_step"] == expected_step
        assert boundary["optimizer_parameter_count"] == 2
        assert boundary["state_parameter_count"] == 1
        assert boundary["stateless_parameter_count"] == 1
        assert boundary["all_optimizer_parameters_classified"] is True
        assert (
            boundary["all_stateless_parameters_have_no_gradient"] is True
        )
        assert (
            boundary[
                "all_stateless_parameters_have_no_optimizer_state"
            ]
            is True
        )
        assert (
            boundary["all_stateful_parameters_have_finite_gradients"] is True
        )
    proof = _validated_optimizer_state_proof(result)
    assert proof["schema_version"] == (
        "adamw_optimizer_state_persistence_proof_v2"
    )
    assert proof["optimizer_parameter_count"] == 2
    assert proof["state_parameter_count"] == 1
    assert proof["stateless_parameter_count"] == 1
    assert result["terminal_parameter_count"] == 2
    assert result["terminal_parameter_bundle"]["parameter_count"] == 2


def test_optimizer_state_proof_rejects_late_parameter_activation(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    class _Ready:
        def wait(self, timeout: float) -> None:
            assert timeout == 900.0

    active = torch.nn.Parameter(torch.tensor([1.0]))
    activated_after_warmup = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW(
        [active, activated_after_warmup],
        lr=0.01,
    )
    observer = _ObserveAdamWPrefix(
        device="cpu",
        ready_barrier=_Ready(),
        warmup_steps=1,
        measured_steps=2,
        stop_on_next_step=False,
        capture_full_single_tensor=False,
        capture_exact_parameter_hashes=True,
        parameter_bundle_path=str(tmp_path / "late-activation.bin"),
        parameter_bundle_relative_path="late-activation.bin",
    )
    with pytest.raises(
        RuntimeError,
        match="state step does not match the prefix boundary",
    ):
        with observer:
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = active.square().sum()
                if step > 0:
                    loss = loss + activated_after_warmup.square().sum()
                loss.backward()
                optimizer.step()


def test_adamw_state_observation_rejects_gradient_without_state() -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    parameter.square().sum().backward()
    optimizer.step()
    del optimizer.state[parameter]
    with pytest.raises(RuntimeError, match="gradient but no optimizer state"):
        _adamw_state_observation(optimizer, expected_step=1)


@pytest.mark.parametrize("malformed_state", ({}, []))
def test_adamw_state_observation_rejects_present_malformed_state(
    malformed_state: object,
) -> None:
    torch = pytest.importorskip("torch")
    active = torch.nn.Parameter(torch.tensor([1.0]))
    inactive = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW([active, inactive], lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    active.square().sum().backward()
    optimizer.step()
    optimizer.state[inactive] = malformed_state
    with pytest.raises(RuntimeError, match="empty or malformed state object"):
        _adamw_state_observation(optimizer, expected_step=1)


def test_adamw_state_observation_rejects_state_without_boundary_gradient() -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    parameter.square().sum().backward()
    optimizer.step()
    parameter.grad = None
    with pytest.raises(RuntimeError, match="finite boundary gradient"):
        _adamw_state_observation(optimizer, expected_step=1)


@pytest.mark.parametrize(
    ("corruption", "expected_step", "message"),
    (
        ("missing_key", 1, "object layout changed"),
        ("nonfinite", 1, "non-finite tensor"),
        ("wrong_step", 2, "step does not match"),
    ),
)
def test_adamw_state_observation_rejects_corrupt_state(
    corruption: str,
    expected_step: int,
    message: str,
) -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    parameter.square().sum().backward()
    optimizer.step()
    if corruption == "missing_key":
        del optimizer.state[parameter]["exp_avg"]
    elif corruption == "nonfinite":
        optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
    with pytest.raises(RuntimeError, match=message):
        _adamw_state_observation(
            optimizer,
            expected_step=expected_step,
        )


def test_adamw_state_observation_rejects_vacuous_all_stateless_state() -> None:
    torch = pytest.importorskip("torch")
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW([first, second], lr=0.01)
    with pytest.raises(RuntimeError, match="no tensor storage"):
        _adamw_state_observation(optimizer, expected_step=1)


@pytest.mark.parametrize(
    "forgery",
    ("partition", "stateless_state_boolean", "layout"),
)
def test_optimizer_state_proof_rejects_forged_partition(
    forgery: str,
) -> None:
    prefix = _stateful_prefix(100, 200)
    start = prefix["optimizer_state_at_interval_start"]
    finish = prefix["optimizer_state_at_interval_finish"]
    if forgery == "partition":
        start["optimizer_parameter_count"] = 2
        finish["optimizer_parameter_count"] = 2
    elif forgery == "stateless_state_boolean":
        start[
            "all_stateless_parameters_have_no_optimizer_state"
        ] = False
        finish[
            "all_stateless_parameters_have_no_optimizer_state"
        ] = False
    else:
        start["object_layout_sha256"] = "0" * 64
        finish["object_layout_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="boundary is not authenticated"):
        _validated_optimizer_state_proof(prefix)


def test_neural_terminal_time_and_tensor_are_after_both_projection_copies() -> None:
    torch = pytest.importorskip("torch")

    class _Ready:
        def wait(self, timeout: float) -> None:
            assert timeout == 900.0

    query = torch.nn.Parameter(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
    optimizer = torch.optim.AdamW([query], lr=0.1)
    observer = _ObserveAdamWPrefix(
        device="cpu",
        ready_barrier=_Ready(),
        warmup_steps=1,
        measured_steps=1,
        stop_on_next_step=True,
        capture_full_single_tensor=True,
        capture_exact_parameter_hashes=False,
        parameter_bundle_path=None,
        parameter_bundle_relative_path=None,
    )
    measured_boundary_seen_before_forward = False
    with pytest.raises(_PrefixFinished):
        with observer:
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                if step == 1:
                    measured_boundary_seen_before_forward = (
                        observer.observation.measured_started_monotonic_ns
                        is not None
                    )
                loss = query.square().sum()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    query.copy_(query + 1.0)
                    query.copy_(query + float(step + 1))
    result = observer.result()
    assert measured_boundary_seen_before_forward is True
    assert result["measured_finished_monotonic_ns"] >= (
        result["measured_started_monotonic_ns"]
    )
    assert np.allclose(
        np.asarray(result["terminal_query_tensor"], dtype=np.float32),
        query.detach().numpy(),
    )
