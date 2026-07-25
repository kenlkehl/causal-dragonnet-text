from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oci.config import TfidfNuisanceStackScientificConfig
from oci.inference import neural_query_context_backend as context_module
from oci.inference.all_evidence_discovery_interfaces import NEURAL_QUERY_MOMENTS
from oci.inference.neural_query_agentic_forest import (
    NeuralQueryAgenticForestConfig,
    NeuralQueryEvidenceCapacityOverflowError,
    build_query_evidence,
)
from oci.inference.neural_query_context_backend import ContextFitNeuralQueryService
from oci.inference.production_stage1_scope_scheduler import (
    build_canonical_stage1_scope_plan,
)
from oci.inference.role_neutral_neural_query_group_execution import (
    COMPLETE_EMBEDDING_TEXT_POLICY,
    EXACT_INNER_TRANSFORM_POLICY,
    FAIL_CLOSED_EVIDENCE_CAPACITY_POLICY,
    REGISTERED_HELDOUT_TRANSFORM_POLICY,
    RoleNeutralNeuralQueryPhysicalGroupRequest,
    execute_role_neutral_neural_query_physical_group,
    replay_role_neutral_neural_query_exact_transform,
    replay_role_neutral_neural_query_heldout_transform,
    validate_role_neutral_neural_query_group_execution,
)
from oci.inference.role_neutral_all_ten_binding import (
    authenticate_role_neutral_neural_query_component,
)


def _registry() -> dict:
    row_count = 30
    rows = tuple(range(row_count))
    outer_folds = []
    for outer_fold in range(1, 3):
        start = (outer_fold - 1) * (row_count // 2)
        heldout = tuple(range(start, start + row_count // 2))
        fit = tuple(row for row in rows if row not in set(heldout))
        partitions = tuple(fit[index::5] for index in range(5))
        outer_folds.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": list(fit),
                "heldout_row_ids": list(heldout),
                "inner_folds": [
                    {
                        "inner_fold": inner_fold,
                        "fit_row_ids": [
                            row for row in fit if row not in set(inner_heldout)
                        ],
                        "heldout_row_ids": list(inner_heldout),
                    }
                    for inner_fold, inner_heldout in enumerate(partitions, start=1)
                ],
            }
        )
    return {"dataset_row_count": row_count, "outer_folds": outer_folds}


def _plan(*, gpu_ids: tuple[int, ...] = ()):
    return build_canonical_stage1_scope_plan(
        registry=_registry(),
        registry_content_sha256="a" * 64,
        global_seed=42,
        gpu_ids=gpu_ids,
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=2,
        expected_inner_fold_count=5,
    )


def _query_config(**changes) -> NeuralQueryAgenticForestConfig:
    config = NeuralQueryAgenticForestConfig(
        treatment_query_count=1,
        outcome_query_count=1,
        effect_query_count=1,
        query_inner_folds=2,
        initial_pool_size=1,
        query_epochs=1,
        final_refit_epochs=1,
        learning_rate=0.01,
        temperature=0.1,
        max_query_drift=0.2,
        final_refit_max_query_drift=0.1,
        kmeans_iterations=1,
        kmeans_sample_chunks=100,
        evidence_top_patients=50,
        evidence_background_patients=0,
        evidence_top_ngrams=100,
        evidence_excerpt_chars=None,
        evidence_chunks_per_patient_per_query=None,
        evidence_ngram_analyzer="word",
        evidence_ngram_range_min=1,
        evidence_ngram_range_max=1,
        evidence_ngram_min_df=1,
        evidence_ngram_max_df=1.0,
        evidence_ngram_vocabulary_max_features=None,
        evidence_ngram_strip_accents=None,
        evidence_ngram_lowercase=True,
        evidence_ngram_stop_words=None,
        evidence_ngram_token_pattern=r"(?u)\b\w\w+\b",
        evidence_ngram_binary=False,
        evidence_ngram_norm="l2",
        evidence_ngram_use_idf=True,
        evidence_ngram_smooth_idf=True,
        evidence_ngram_sublinear_tf=True,
        evidence_safe_term_max_tokens=6,
        evidence_safe_term_max_chars=160,
        rag_chunks_per_query=1,
        rag_max_chunks_per_patient=50,
        rag_excerpt_chars=50_000,
        max_features_per_query=1,
        max_raw_feature_candidates=3,
        max_canonical_features=3,
        max_review_rounds=2,
        max_review_additions_per_round=2,
        max_variables_per_extraction_request=10,
    )
    return replace(config, **changes)


class _FakeBoundCache:
    def __init__(self, cache, row_ids):
        self.cache = cache
        self.row_ids = tuple(row_ids)
        self.token_bounded_row_ids = ()

    def identity(self):
        return {
            "cache": self.cache.identity(),
            "spent_row_ids_sha256": hashlib.sha256(
                json.dumps(list(self.row_ids), separators=(",", ":")).encode()
            ).hexdigest(),
            "token_bounded_row_ids_sha256": hashlib.sha256(b"[]").hexdigest(),
        }

    def chunk_matrices(self, row_ids):
        rows = tuple(map(int, row_ids))
        if any(row not in self.row_ids for row in rows):
            raise ValueError("fake scoped embedding view refused a peer row")
        return tuple(self.cache.matrices[row] for row in rows)

    def chunk_texts(self, row_ids):
        rows = tuple(map(int, row_ids))
        if any(row not in self.row_ids for row in rows):
            raise ValueError("fake scoped embedding view refused a peer row")
        return tuple(self.cache.chunks[row] for row in rows)


class _FakeCache:
    def __init__(self, texts: tuple[str, ...], *, two_chunks: bool = False):
        self.texts = texts
        self.chunks = []
        self.matrices = []
        for row_id, text in enumerate(texts):
            if two_chunks:
                self.chunks.append((text, f"second chunk row {row_id}"))
                self.matrices.append(
                    np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
                )
            else:
                self.chunks.append((text,))
                self.matrices.append(
                    np.asarray(
                        [[1.0, 0.0] if row_id % 2 == 0 else [0.0, 1.0]],
                        dtype=np.float32,
                    )
                )
        self.chunks = tuple(self.chunks)
        self.matrices = tuple(self.matrices)
        self.row_count = len(texts)
        self.metadata = {
            "num_samples": len(texts),
            "hidden_size": 2,
            "chunk_cap_nonbinding": True,
            "semantic_truncation_allowed": False,
            "tokenizer_truncation_allowed": False,
            "chunking_mode": "test_exact_nontruncating_chunks_v1",
        }

    def identity(self):
        return {
            "provider": "spent_only_frozen_chunk_embedding_cache_v2",
            "metadata_sha256": "1" * 64,
            "embeddings_sha256": "2" * 64,
            "offsets_sha256": "3" * 64,
            "chunk_texts_sha256": hashlib.sha256(
                json.dumps(self.chunks, separators=(",", ":")).encode()
            ).hexdigest(),
            "row_count": self.row_count,
            "chunk_count": sum(map(len, self.chunks)),
            "cache_snapshot_authentication": "test_closed_bytes",
            "chunk_text_storage": "test_exact_rows",
            "embeddings_path_backed": False,
            "private_snapshot_embedding_mmap": True,
            "future_row_text_decoded": False,
            "novel_text_encoding_allowed": False,
        }

    def bind_spent(self, row_ids, texts):
        rows = tuple(map(int, row_ids))
        exact = tuple(texts)
        if len(rows) != len(exact) or any(
            self.texts[row_id] != text
            for row_id, text in zip(rows, exact, strict=True)
        ):
            raise ValueError("fake cache text does not match its authenticated row")
        return _FakeBoundCache(self, rows)


def _texts() -> tuple[str, ...]:
    rows = [
        f"patient row {row_id} baseline marker_{row_id % 3} therapy_{row_id % 2}"
        for row_id in range(30)
    ]
    rows[16] = ("paddingword " * 1400) + " sentinelafterfourteenthousand"
    return tuple(rows)


def _service(
    root: Path,
    *,
    config: NeuralQueryAgenticForestConfig,
    devices: tuple[str, ...] = ("cpu",),
    two_chunks: bool = False,
) -> ContextFitNeuralQueryService:
    root.mkdir(parents=True, exist_ok=True)
    service = object.__new__(ContextFitNeuralQueryService)
    service.cache_dir = root / "executable-query-scratch"
    service._owned_discoveries = {}
    service._owned_discovery_bindings = {}
    service._owned_discovery_content_sha256s = {}
    service.dataset_path = root / "dataset.parquet"
    service.stage1_config_path = root / "stage1.json"
    service.stage1_config_path.write_text("{}", encoding="utf-8")
    service._stage1_config_snapshot = SimpleNamespace(
        sha256=hashlib.sha256(b"{}").hexdigest(),
        verify_source=lambda: None,
    )
    service.text_column = "configured_note"
    service.embedding_cache = _FakeCache(_texts(), two_chunks=two_chunks)
    service._dataset_row_count = service.embedding_cache.row_count
    service._nuisance_views = ({"name": "configured_nuisance_view"},)
    service._nuisance_stack_config = TfidfNuisanceStackScientificConfig()
    service.query_config = config
    service.query_config.validate()
    service.nuisance_folds = 2
    service.devices = devices
    service.seed = 17
    service.outcome_type = "binary"
    service._identity = service._identity_payload()
    return service


def _request(
    service: ContextFitNeuralQueryService,
    *,
    plan=None,
    physical_owner_scope_id: str | None = None,
    query_config: dict | None = None,
) -> RoleNeutralNeuralQueryPhysicalGroupRequest:
    selected_plan = plan or _plan()
    owner_scope_id = physical_owner_scope_id or next(
        owner.scope_id
        for owner, members in selected_plan.physical_scope_groups
        if len(members) > 1
    )
    return RoleNeutralNeuralQueryPhysicalGroupRequest.from_plan(
        plan=selected_plan,
        physical_owner_scope_id=owner_scope_id,
        query_config=(
            asdict(service.query_config)
            if query_config is None
            else query_config
        ),
        nuisance_folds=service.nuisance_folds,
        seed=service.seed,
        outcome_type=service.outcome_type,
        service_scientific_identity=service.identity(),
        evidence_capacity_policy=FAIL_CLOSED_EVIDENCE_CAPACITY_POLICY,
        embedding_text_coverage_policy=COMPLETE_EMBEDDING_TEXT_POLICY,
        heldout_transform_policy=REGISTERED_HELDOUT_TRANSFORM_POLICY,
    )


def _fake_discovery(row_ids: tuple[int, ...]):
    return {
        "banks": {
            bank: {
                "queries": np.asarray([[1.0, 0.0]], dtype=np.float32),
                "train_activations": np.linspace(
                    0.1,
                    0.9,
                    len(row_ids),
                    dtype=np.float32,
                )[:, None],
                "records": [
                    {
                        "query_id": f"{bank}_context_query_001",
                        "member_count": 2,
                        "member_subfolds": [1, 2],
                        "fit_standardized_score": 0.4,
                    }
                ],
                "consensus": {"method": "test_ungated_consensus"},
                "objective": f"test_{bank}_objective",
                "all_queries_retained": True,
                "statistical_gate_applied": False,
            }
            for bank in ("treatment", "outcome", "effect")
        },
        "runtime": context_module.NEURAL_QUERY_DISCOVERY_RUNTIME_ID,
        "fit_input_binding_sha256": "a" * 64,
        "fit_nuisance_output_binding": {
            "schema_version": (
                context_module.NEURAL_QUERY_NUISANCE_OUTPUT_BINDING_SCHEMA
            ),
            "fit_row_ids": list(row_ids),
            "fit_e_sha256": "b" * 64,
            "fit_m_sha256": "c" * 64,
            "heldout_labels_accessed": False,
        },
        "subfold_audit": [],
        "all_queries_retained": True,
        "validation_audits_used_for_selection": False,
        "executable_checkpoint_io": False,
    }


def _patch_fit(monkeypatch, calls: list[tuple[int, ...]]):
    def fake_fit(**kwargs):
        rows = tuple(kwargs["row_ids"])
        calls.append(rows)
        return _fake_discovery(rows)

    monkeypatch.setattr(context_module, "_fit_context_query_discovery", fake_fit)


def _execute(
    tmp_path: Path,
    monkeypatch,
    *,
    physical_owner_scope_id: str | None = None,
    config: NeuralQueryAgenticForestConfig | None = None,
):
    selected_config = config or _query_config()
    service = _service(tmp_path / "service", config=selected_config)
    request = _request(
        service,
        physical_owner_scope_id=physical_owner_scope_id,
    )
    owner = request.physical_owner
    fit_texts = tuple(_texts()[row] for row in owner.fit_row_ids)
    treatment = np.asarray(
        [position % 2 for position in range(len(fit_texts))],
        dtype=float,
    )
    outcome = 1.0 - treatment
    heldout_texts = tuple(_texts()[row] for row in owner.heldout_row_ids)
    calls: list[tuple[int, ...]] = []
    _patch_fit(monkeypatch, calls)
    root = (tmp_path / "artifact").resolve()
    loader_calls: list[tuple[int, ...]] = []

    def loader(row_ids):
        assert (root / "fit_only_family_seal.json").is_file()
        aliases = request.logical_members[1:]
        if aliases:
            cumulative = json.loads(
                (root / "logical_views" / "001_cumulative.json").read_text(
                    encoding="utf-8"
                )
            )
            assert cumulative["logical_heldout_row_ids"] is None
            assert cumulative["logical_heldout_text_sha256"] is None
            assert cumulative["prediction_artifact"] is None
        else:
            assert not tuple((root / "logical_views").iterdir())
        assert not list((root / "fit_state").rglob("*.joblib"))
        assert not list(service.cache_dir.rglob("*.joblib"))
        loader_calls.append(tuple(row_ids))
        return heldout_texts

    terminal = execute_role_neutral_neural_query_physical_group(
        request=request,
        output_root=root,
        service=service,
        fit_texts=fit_texts,
        fit_treatment=treatment,
        fit_outcome=outcome,
        heldout_text_loader=loader,
    )
    return {
        "root": root,
        "service": service,
        "request": request,
        "heldout_texts": heldout_texts,
        "terminal": terminal,
        "fit_calls": calls,
        "loader_calls": loader_calls,
    }


def test_request_is_device_neutral_and_requires_every_query_setting(tmp_path: Path):
    config = _query_config()
    cpu = _service(tmp_path / "cpu", config=config, devices=("cpu",))
    gpu = _service(
        tmp_path / "gpu",
        config=config,
        devices=("cuda:7", "cuda:2"),
    )
    assert cpu.identity() == gpu.identity()
    assert EXACT_INNER_TRANSFORM_POLICY == REGISTERED_HELDOUT_TRANSFORM_POLICY
    cpu_request = _request(cpu, plan=_plan())
    gpu_request = _request(gpu, plan=_plan(gpu_ids=(7, 2)))
    assert cpu_request.as_dict() == gpu_request.as_dict()

    incomplete = asdict(config)
    incomplete.pop("evidence_excerpt_chars")
    with pytest.raises(ValueError, match="explicitly contain every closed setting"):
        _request(cpu, query_config=incomplete)


def test_fit_once_seals_before_text_loader_and_replays_safe_state(
    tmp_path: Path,
    monkeypatch,
):
    result = _execute(tmp_path, monkeypatch)
    root = result["root"]
    request = result["request"]
    terminal = result["terminal"]

    assert result["fit_calls"] == [request.physical_owner.fit_row_ids]
    assert result["loader_calls"] == [request.physical_owner.heldout_row_ids]
    assert terminal == validate_role_neutral_neural_query_group_execution(
        root=root,
        request=request,
    )
    assert [row["event"] for row in terminal["event_order"]] == [
        "fit_completed",
        "owned_executable_checkpoint_removed",
        "fit_family_artifact_sealed",
        "cumulative_fit_only_view_published",
        "primary_heldout_text_opened",
        "primary_heldout_transform_completed",
        "primary_logical_view_published",
    ]
    seal = json.loads((root / "fit_only_family_seal.json").read_text())
    assert seal["family"] == NEURAL_QUERY_MOMENTS
    terms = {
        term["term"]
        for query in seal["evidence_payload"]["architecture_evidence"]
        for term in query["top_contrastive_ngrams"]
    }
    assert "sentinelafterfourteenthousand" in terms
    assert not list(root.rglob("*.joblib"))
    assert not list(root.rglob("*.npz"))
    receipt = authenticate_role_neutral_neural_query_component(
        root=root,
        plan=request.authority_plan,
        request=request,
    )
    assert tuple(receipt.family_fit_seals) == (NEURAL_QUERY_MOMENTS,)
    assert receipt.lossy_evidence_selection_applied is False

    fresh_service = _service(
        tmp_path / "fresh-service",
        config=_query_config(),
    )
    replay = replay_role_neutral_neural_query_exact_transform(
        root=root,
        request=request,
        service=fresh_service,
        exact_heldout_texts=result["heldout_texts"],
    )
    assert replay["gate_row_ids"] == request.physical_owner.heldout_row_ids
    assert replay["registered_heldout_labels_accessed"] is False
    assert replay["executable_checkpoint_loaded"] is False


def test_full_outer_owner_transforms_outer_heldout_text_without_labels(
    tmp_path: Path,
    monkeypatch,
):
    plan = _plan()
    owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if owner.scope_kind == "full_outer"
    )
    assert members == (owner,)

    result = _execute(
        tmp_path,
        monkeypatch,
        physical_owner_scope_id=owner.scope_id,
    )
    request = result["request"]
    terminal = result["terminal"]
    assert request.physical_owner == owner
    assert request.logical_members == (owner,)
    assert result["fit_calls"] == [owner.fit_row_ids]
    assert result["loader_calls"] == [owner.heldout_row_ids]
    assert [row["event"] for row in terminal["event_order"]] == [
        "fit_completed",
        "owned_executable_checkpoint_removed",
        "fit_family_artifact_sealed",
        "primary_heldout_text_opened",
        "primary_heldout_transform_completed",
        "primary_logical_view_published",
    ]
    assert all(
        event["registered_heldout_labels_accessed"] is False
        for event in terminal["event_order"]
    )
    primary = json.loads(
        (
            result["root"]
            / terminal["logical_views"][0]["relative_path"]
        ).read_text(encoding="utf-8")
    )
    assert primary["logical_purpose"] == "full_outer"
    assert primary["logical_heldout_row_ids"] == list(owner.heldout_row_ids)
    assert (
        primary["view_input_policy"]
        == "registered_heldout_row_ids_and_text_no_labels_v1"
    )
    assert primary["registered_heldout_labels_accessed"] is False
    assert terminal == validate_role_neutral_neural_query_group_execution(
        root=result["root"],
        request=request,
    )
    receipt = authenticate_role_neutral_neural_query_component(
        root=result["root"],
        plan=plan,
        request=request,
    )
    assert receipt.logical_scope_ids == (owner.scope_id,)
    assert receipt.registered_heldout_labels_accessed is False

    replay = replay_role_neutral_neural_query_heldout_transform(
        root=result["root"],
        request=request,
        service=_service(
            tmp_path / "fresh-full-outer-service",
            config=_query_config(),
        ),
        heldout_texts=result["heldout_texts"],
    )
    assert replay["gate_row_ids"] == owner.heldout_row_ids
    assert replay["registered_heldout_labels_accessed"] is False


def test_singleton_cumulative_owner_seals_fit_before_label_free_transform(
    tmp_path: Path,
    monkeypatch,
):
    plan = _plan()
    owner, members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if owner.scope_kind == "cumulative_spent"
    )
    assert members == (owner,)

    result = _execute(
        tmp_path,
        monkeypatch,
        physical_owner_scope_id=owner.scope_id,
    )
    terminal = result["terminal"]
    assert result["request"].logical_members == (owner,)
    assert result["fit_calls"] == [owner.fit_row_ids]
    assert result["loader_calls"] == [owner.heldout_row_ids]
    assert [row["event"] for row in terminal["event_order"]] == [
        "fit_completed",
        "owned_executable_checkpoint_removed",
        "fit_family_artifact_sealed",
        "primary_heldout_text_opened",
        "primary_heldout_transform_completed",
        "primary_logical_view_published",
    ]
    first_text_access = next(
        index
        for index, event in enumerate(terminal["event_order"])
        if event["registered_heldout_text_accessed"]
    )
    assert [
        event["event"] for event in terminal["event_order"][:first_text_access]
    ] == [
        "fit_completed",
        "owned_executable_checkpoint_removed",
        "fit_family_artifact_sealed",
    ]
    assert all(
        event["registered_heldout_labels_accessed"] is False
        for event in terminal["event_order"]
    )
    primary = json.loads(
        (
            result["root"]
            / terminal["logical_views"][0]["relative_path"]
        ).read_text(encoding="utf-8")
    )
    assert primary["logical_purpose"] == "cumulative_spent"
    assert primary["logical_heldout_row_ids"] == list(owner.heldout_row_ids)
    assert primary["registered_heldout_text_accessed"] is True
    assert primary["registered_heldout_labels_accessed"] is False
    assert terminal == validate_role_neutral_neural_query_group_execution(
        root=result["root"],
        request=result["request"],
    )
    receipt = authenticate_role_neutral_neural_query_component(
        root=result["root"],
        plan=plan,
        request=result["request"],
    )
    assert receipt.logical_scope_ids == (owner.scope_id,)
    assert receipt.registered_heldout_labels_accessed is False


def test_null_allocations_cover_all_fit_patients_and_terms(
    tmp_path: Path,
    monkeypatch,
):
    result = _execute(
        tmp_path,
        monkeypatch,
        config=_query_config(
            evidence_top_patients=1,
            evidence_background_patients=None,
            evidence_top_ngrams=None,
            rag_max_chunks_per_patient=None,
            rag_excerpt_chars=None,
        ),
    )
    metadata = json.loads(
        (
            result["root"] / "fit_state" / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    chunk_coverage = metadata["chunk_coverage"]
    evidence_coverage = metadata["evidence_coverage"]
    assert chunk_coverage["configured_patient_evidence_capacity"] is None
    assert chunk_coverage["complete_background_patient_allocation"] is True
    assert chunk_coverage["patient_evidence_capacity_nonbinding"] is True
    assert chunk_coverage["configured_term_count_capacity"] is None
    assert evidence_coverage["term_count_capacity_nonbinding"] is True


def test_capacity_bindings_fail_instead_of_omitting_evidence(
    tmp_path: Path,
    monkeypatch,
):
    patient_config = _query_config(
        evidence_top_patients=1,
        evidence_background_patients=0,
    )
    service = _service(tmp_path / "patients", config=patient_config)
    request = _request(service)
    owner = request.physical_owner
    texts = tuple(_texts()[row] for row in owner.fit_row_ids)
    treatment = np.asarray([index % 2 for index in range(len(texts))])
    with pytest.raises(ValueError, match="patient evidence allocation would omit"):
        execute_role_neutral_neural_query_physical_group(
            request=request,
            output_root=(tmp_path / "patient-artifact").resolve(),
            service=service,
            fit_texts=texts,
            fit_treatment=treatment,
            fit_outcome=1 - treatment,
            exact_heldout_text_loader=lambda _rows: pytest.fail(
                "held-out loader must remain sealed"
            ),
        )

    chunk_config = _query_config(
        evidence_chunks_per_patient_per_query=1,
    )
    chunk_service = _service(
        tmp_path / "chunks",
        config=chunk_config,
        two_chunks=True,
    )
    chunk_request = _request(chunk_service)
    chunk_owner = chunk_request.physical_owner
    chunk_texts = tuple(_texts()[row] for row in chunk_owner.fit_row_ids)
    chunk_treatment = np.asarray(
        [index % 2 for index in range(len(chunk_texts))]
    )
    with pytest.raises(ValueError, match="chunk allocation would omit"):
        execute_role_neutral_neural_query_physical_group(
            request=chunk_request,
            output_root=(tmp_path / "chunk-artifact").resolve(),
            service=chunk_service,
            fit_texts=chunk_texts,
            fit_treatment=chunk_treatment,
            fit_outcome=1 - chunk_treatment,
            exact_heldout_text_loader=lambda _rows: pytest.fail(
                "held-out loader must remain sealed"
            ),
        )


def test_shared_query_evidence_term_and_chunk_capacities_fail_closed():
    common = {
        "bank": "effect",
        "queries": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "query_records": [
            {
                "query_id": "effect_query_001",
                "member_count": 1,
                "fit_standardized_score": 0.2,
            }
        ],
        "row_ids": [0, 1],
        "chunk_matrices": [
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 1.0]], dtype=np.float32),
        ],
        "all_chunk_texts": [["alpha beta gamma"], ["background"]],
        "device": "cpu",
        "seed": 3,
    }
    with pytest.raises(
        NeuralQueryEvidenceCapacityOverflowError,
        match="no terms were silently discarded",
    ):
        build_query_evidence(
            **common,
            config=_query_config(
                evidence_top_patients=1,
                evidence_background_patients=1,
                evidence_top_ngrams=1,
            ),
        )

    with pytest.raises(
        NeuralQueryEvidenceCapacityOverflowError,
        match="no chunks were silently discarded",
    ):
        build_query_evidence(
            **{
                **common,
                "chunk_matrices": [
                    np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
                    np.asarray([[0.0, 1.0], [0.2, 0.8]], dtype=np.float32),
                ],
                "all_chunk_texts": [
                    ["alpha", "second alpha"],
                    ["background", "second background"],
                ],
            },
            config=_query_config(
                evidence_top_patients=1,
                evidence_background_patients=1,
                evidence_chunks_per_patient_per_query=1,
            ),
        )


def test_fresh_validation_rejects_binary_tampering_and_extra_files(
    tmp_path: Path,
    monkeypatch,
):
    result = _execute(tmp_path / "base", monkeypatch)
    source = result["root"]
    request = result["request"]

    tampered = tmp_path / "tampered"
    shutil.copytree(source, tampered)
    array_path = next((tampered / "logical_views" / "primary_predictions").glob(
        "*feature_values.npy"
    ))
    payload = bytearray(array_path.read_bytes())
    payload[-1] ^= 1
    array_path.write_bytes(bytes(payload))
    with pytest.raises((RuntimeError, ValueError)):
        validate_role_neutral_neural_query_group_execution(
            root=tampered,
            request=request,
        )

    extra = tmp_path / "extra"
    shutil.copytree(source, extra)
    (extra / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="extra or missing root member"):
        validate_role_neutral_neural_query_group_execution(
            root=extra,
            request=request,
        )

    linked = tmp_path / "hardlinked"
    shutil.copytree(source, linked)
    evidence = linked / "fit_state" / "evidence.json"
    external = tmp_path / "external-evidence.json"
    shutil.copy2(evidence, external)
    evidence.unlink()
    evidence.hardlink_to(external)
    with pytest.raises(ValueError, match="non-hard-linked"):
        validate_role_neutral_neural_query_group_execution(
            root=linked,
            request=request,
        )
