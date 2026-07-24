from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import oci.inference.neural_query_context_backend as query_context_module

from oci.config import AppliedInferenceConfig, ExperimentConfig
from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    BOW_NUISANCE,
    BOW_R_LOSS,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPICS,
)
from oci.inference.all_evidence_fusion import (
    NEURAL_QUERY_MOMENTS,
    NEURAL_QUERY_SOURCE,
    TFIDF_TOPIC_SOURCE,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    build_role_neutral_evidence_catalog,
)
from oci.inference.neural_query_agentic_forest import NeuralQueryAgenticForestConfig
from oci.inference.production_stage1_bundle import (
    PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    PRODUCTION_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    STAGE1_BEHAVIOR_IDENTITY_SCHEMA,
    STAGE1_EMBEDDING_CLUSTER_FEASIBILITY_AUDIT_SCHEMA,
    STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
    STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
    STAGE1_QUERY_ARTIFACT_SCHEMA,
    STAGE1_SCOPE_INDEX_SCHEMA,
    ProductionStage1BundleBuilder,
    Stage1BundleBuildOptions,
    _catalog_ready_legacy_digest,
    _build_htr_input_nontruncation_audit,
    _component_file_registration,
    _component_native_artifact_registration,
    _matched_pair_subproducer_proofs,
    _read_stable_sha256,
    _registry_scopes,
    _scientific_query_config_identity,
    _seal_component,
    _sha256_file,
    _sha256_json,
    _source_identity,
    _register_neural_query_native_family_proof,
    _validate_neural_query_native_family_proof_index,
    _validate_cache_configuration,
    _validate_effective_config,
    _write_neural_query_moment_artifact,
    _write_immutable_json,
    _write_raw_evidence_sidecar,
    build_embedding_cluster_feasibility_audit,
    build_canonical_split_registry,
    build_parser,
    exact_inner_family_adapter_gate,
    main,
    options_from_args,
    validate_embedding_cluster_feasibility_audit,
    validate_htr_input_nontruncation_audit,
)
from oci.inference.review_spent_evidence_provider import (
    SpentOnlyFrozenChunkEmbeddingCache,
)
from oci.inference.tfidf_topic_discovery import row_set_fingerprint
from oci.inference.stage1_exact_inner_evidence import CanonicalStage1SplitRegistry
from oci.inference.tfidf_topic_split_registry import (
    TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
)


def _valid_config(tmp_path: Path) -> tuple[AppliedInferenceConfig, Path]:
    config = AppliedInferenceConfig(cv_folds=3)
    architecture = config.architecture
    architecture.model_type = "multi_model_forest"
    architecture.htr_freeze_sentence_encoder = False
    model_dir = tmp_path / "htr_model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"test model")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "max_position_embeddings": 512,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "vocab.txt").write_text(
        "\n".join(
            (
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "safe",
                "baseline",
                "text",
                "note",
                "zero",
                "one",
                "two",
                "three",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    architecture.htr_sentence_model = str(model_dir)
    forest = architecture.multi_model_forest
    forest.set_feature_discovery_methods(
        ["bow", "htr", "embedding_contrast"], source="production-wrapper-test"
    )
    forest.matched_pair_uplift_enabled = True
    forest.matched_pair_bow_enabled = True
    forest.matched_pair_htr_enabled = True
    forest.require_honest_outer_split = True
    forest.candidate_consistency_enabled = True
    forest.candidate_consistency_inner_folds = 4
    forest.embedding_contrast.enabled = True
    forest.embedding_contrast.include_cluster_contrast_vectors = True
    return config, model_dir


def _write_cluster_preflight_cache(
    path: Path,
    *,
    texts: tuple[str, ...],
    embeddings: np.ndarray,
) -> SpentOnlyFrozenChunkEmbeddingCache:
    path.mkdir(parents=True)
    if embeddings.shape[0] != len(texts):
        raise AssertionError("cluster-preflight cache needs one embedding per row")
    np.save(path / "chunk_embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    np.save(path / "offsets.npy", np.arange(len(texts) + 1, dtype=np.int64))
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "num_samples": len(texts),
                "hidden_size": int(embeddings.shape[1]),
                "chunk_size_words": 64,
                "chunk_overlap_words": 0,
                "max_chunks": 1,
                "chunk_selection": "last",
            }
        ),
        encoding="utf-8",
    )
    with (path / "chunk_texts.jsonl").open("w", encoding="utf-8") as handle:
        for value in texts:
            handle.write(json.dumps({"chunks": [value]}) + "\n")
    return SpentOnlyFrozenChunkEmbeddingCache(path)


def _cluster_preflight_case(tmp_path: Path) -> dict[str, object]:
    config = AppliedInferenceConfig(cv_folds=2)
    forest = config.architecture.multi_model_forest
    forest.candidate_consistency_inner_folds = 4
    embedding = forest.embedding_contrast
    embedding.enabled = True
    embedding.disable_reason = None
    embedding.chunk_size_words = 64
    embedding.chunk_overlap_words = 0
    embedding.max_chunks = 1
    embedding.chunk_selection = "last"
    embedding.max_chunks_per_patient = 1
    embedding.top_k_chunks_per_tail = 20
    embedding.include_bow_phrases_as_concepts = False
    embedding.concept_phrases = []
    embedding.external_corpus_cache_dirs = []
    embedding.cluster_contrast_n_clusters = 8
    embedding.cluster_contrast_min_cluster_size = 10
    embedding.cluster_contrast_min_group_size = 5
    embedding.cluster_contrast_min_cell_size = 2
    embedding.cluster_contrast_max_components = 3
    embedding.cluster_contrast_top_loadings = 4
    embedding.cluster_contrast_random_state = 42
    embedding.cluster_contrast_kmeans_n_init = 3

    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    for repetition in range(16):
        for cluster in range(8):
            for treatment, outcome in ((0, 0), (0, 1), (1, 0), (1, 1)):
                concordant = treatment == outcome
                text = " ".join(
                    (
                        f"clusterword{cluster}",
                        "treatedword" if treatment else "untreatedword",
                        "positiveword" if outcome else "negativeword",
                        "concordantword" if concordant else "discordantword",
                        f"repeatword{repetition}",
                    )
                )
                rows.append(
                    {
                        config.text_column: text,
                        config.treatment_column: treatment,
                        config.outcome_column: outcome,
                    }
                )
                vector = np.zeros(32, dtype=np.float32)
                vector[cluster] = 20.0
                vector[8 + cluster] = 1.5 if treatment else -1.5
                vector[16 + cluster] = 1.25 if outcome else -1.25
                vector[24 + cluster] = 1.0 if concordant else -1.0
                vectors.append(vector)
    modeling_data = pd.DataFrame(rows)
    texts = tuple(modeling_data[config.text_column].astype(str))
    cache = _write_cluster_preflight_cache(
        tmp_path / "cluster_preflight_cache",
        texts=texts,
        embeddings=np.asarray(vectors, dtype=np.float32),
    )
    registry = build_canonical_split_registry(data=modeling_data, config=config, seed=42)
    return {
        "config": config,
        "modeling_data": modeling_data,
        "cache": cache,
        "cache_identity": cache.identity(),
        "registry": registry,
        "registry_sha256": _sha256_json(registry),
    }


def _registry() -> dict:
    return {
        "schema_version": TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
        "dataset_row_count": 6,
        "outer_folds": [
            {
                "outer_fold": 1,
                "fit_row_ids": [0, 1, 2, 3],
                "heldout_row_ids": [4, 5],
                "inner_folds": [
                    {
                        "inner_fold": 1,
                        "fit_row_ids": [0, 1],
                        "heldout_row_ids": [2, 3],
                    },
                    {
                        "inner_fold": 2,
                        "fit_row_ids": [2, 3],
                        "heldout_row_ids": [0, 1],
                    },
                ],
            }
        ],
    }


def _legacy_rows(registry: dict, registry_sha: str) -> list[dict]:
    rows = []
    for scope in _registry_scopes(registry):
        inner = scope["inner_fold"]
        row = {
            "schema_version": "multi_model_agentic_discovery_handoff_v1",
            "fold_key": (
                scope["outer_fold"] * 1000 + inner if inner is not None else scope["outer_fold"]
            ),
            "outer_fold": scope["outer_fold"],
            "scope": (
                "candidate_consistency_inner_train" if inner is not None else "full_outer_train"
            ),
            "n_rows": len(scope["fit_row_ids"]),
            "importance": {},
            "embedding_contrast_evidence": {},
            "htr_evidence": {},
            "context": {},
            "metrics": {},
            "fit_row_ids": scope["fit_row_ids"],
            "heldout_row_ids": scope["heldout_row_ids"],
            "fit_row_fingerprint": row_set_fingerprint(scope["fit_row_ids"]),
            "heldout_row_fingerprint": row_set_fingerprint(scope["heldout_row_ids"]),
            "split_registry_content_sha256": registry_sha,
            "evidence_scope_fit_was_executed": True,
            "evidence_reused_from_fold_key": None,
            "heldout_labels_supplied_to_evidence_builder": False,
            "lossless_concept_catalog_projection": True,
            "prompt_compactor_used": False,
        }
        if inner is not None:
            row["inner_fold"] = inner
            row["heldout_rows"] = len(scope["heldout_row_ids"])
        rows.append(row)
    return rows


def _pair_proofs(scope_id: str) -> dict:
    subproducers = {
        name: {
            "schema_version": "production_stage1_matched_pair_subproducer_proof_v1",
            "subproducer": name,
            "success": True,
            "output_columns": [f"{name}_output"],
            "model_artifact_sha256": ("a" if name == "bow" else "b") * 64,
            "fit_execution_sha256": ("c" if name == "bow" else "d") * 64,
            "artifact_semantics": "sealed_model_outputs_and_concept_evidence",
        }
        for name in ("bow", "htr")
    }
    return {
        "schema_version": "production_stage1_matched_pair_subproducer_proof_v1",
        "scope_id": scope_id,
        "all_required_subproducers_succeeded": True,
        "subproducers": subproducers,
        "content_sha256": _sha256_json(subproducers),
    }


def _write_legacy_component(tmp_path: Path, registry: dict, registry_sha: str):
    component = tmp_path / "legacy"
    handoff_dir = component / "handoff"
    sidecar_dir = component / "raw_evidence_sidecars"
    handoff_dir.mkdir(parents=True)
    sidecar_dir.mkdir()
    rows = _legacy_rows(registry, registry_sha)
    index_rows = []
    for scope, row in zip(_registry_scopes(registry), rows):
        proofs = _pair_proofs(scope["scope_id"])
        registration = _write_raw_evidence_sidecar(
            sidecar_dir / f"{scope['scope_id']}.json",
            component_root=component,
            scope=scope,
            split_registry_content_sha256=registry_sha,
            raw_evidence={"importance": {"safe": True}},
            matched_pair_proofs=proofs,
        )
        row["raw_evidence_sidecar_sha256"] = registration["sha256"]
        index_rows.append(
            {
                "scope_id": scope["scope_id"],
                "raw_evidence_sidecar": registration,
                "matched_pair_subproducer_proofs_sha256": proofs["content_sha256"],
            }
        )
    handoff = handoff_dir / "discovery_contexts.jsonl"
    handoff.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (component / "exact_scope_index.json").write_text(
        json.dumps(
            {
                "schema_version": STAGE1_SCOPE_INDEX_SCHEMA,
                "split_registry_content_sha256": registry_sha,
                "scopes": index_rows,
            }
        ),
        encoding="utf-8",
    )
    return handoff, rows


def test_cli_is_one_command_and_has_no_digest_approval_option():
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "approval_sha256" not in destinations
    assert not any("approve" in value or "digest" in value for value in destinations)
    args = parser.parse_args(
        [
            "--dataset",
            "cohort.parquet",
            "--stage1-config",
            "stage1.json",
            "--embedding-cache-dir",
            "cache",
            "--output-dir",
            "bundle",
            "--unit-id-column",
            "person_key",
            "--dry-run",
        ]
    )
    assert args.dry_run is True


def test_cli_supports_one_command_fresh_cache_build_and_rejects_write_on_dry_run():
    parser = build_parser()
    values = [
        "--dataset",
        "cohort.parquet",
        "--stage1-config",
        "stage1.json",
        "--embedding-cache-output-dir",
        "/tmp/fresh-stage1-cache",
        "--embedding-local-model-path",
        "/tmp/local-embedding-model",
        "--output-dir",
        "bundle",
        "--unit-id-column",
        "person_key",
    ]
    options = options_from_args(parser.parse_args(values))
    assert options.embedding_cache_dir is None
    assert options.embedding_cache_output_dir == Path("/tmp/fresh-stage1-cache")
    assert options.embedding_local_model_path == Path("/tmp/local-embedding-model")
    with pytest.raises(ValueError, match="dry-run cannot publish"):
        options_from_args(parser.parse_args([*values, "--dry-run"]))


def test_cli_rejects_incomplete_or_resume_fresh_cache_modes():
    parser = build_parser()
    base = [
        "--dataset",
        "cohort.parquet",
        "--stage1-config",
        "stage1.json",
        "--embedding-cache-output-dir",
        "/tmp/fresh-stage1-cache",
        "--output-dir",
        "bundle",
        "--unit-id-column",
        "person_key",
    ]
    with pytest.raises(ValueError, match="local-model-path is required"):
        options_from_args(parser.parse_args(base))
    with pytest.raises(ValueError, match="resume requires"):
        options_from_args(
            parser.parse_args([*base, "--embedding-local-model-path", "/tmp/model", "--resume"])
        )


def test_prepare_rejects_output_root_symlink_before_resolution(tmp_path: Path):
    dataset = tmp_path / "cohort.parquet"
    dataset.write_bytes(b"placeholder")
    config = tmp_path / "stage1.json"
    config.write_text("{}", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(real_output, target_is_directory=True)
    builder = ProductionStage1BundleBuilder(
        Stage1BundleBuildOptions(
            dataset_path=dataset,
            config_path=config,
            embedding_cache_dir=cache,
            output_dir=linked_output,
            unit_id_column="person_key",
        )
    )
    with pytest.raises(ValueError, match="output directory cannot be a symlink"):
        builder.prepare()


@pytest.mark.parametrize("output_inside_cache", (True, False))
def test_prepare_rejects_overlapping_cache_and_output_trees(
    tmp_path: Path, output_inside_cache: bool
):
    dataset = tmp_path / "cohort.parquet"
    dataset.write_bytes(b"placeholder")
    config = tmp_path / "stage1.json"
    config.write_text("{}", encoding="utf-8")
    if output_inside_cache:
        cache = tmp_path / "cache"
        cache.mkdir()
        output = cache / "bundle"
    else:
        output = tmp_path / "bundle"
        output.mkdir()
        cache = output / "cache"
        cache.mkdir()
    builder = ProductionStage1BundleBuilder(
        Stage1BundleBuildOptions(
            dataset_path=dataset,
            config_path=config,
            embedding_cache_dir=cache,
            output_dir=output,
            unit_id_column="person_key",
        )
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        builder.prepare()


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"])
def test_production_identity_rejects_non_utf8_cohort_values(surrogate):
    with pytest.raises(ValueError, match="valid UTF-8"):
        _sha256_json({"clinical_text": surrogate})


def test_cli_returns_nonzero_when_readiness_preflight_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        ProductionStage1BundleBuilder,
        "build",
        lambda _self: {
            "status": "blocked_pending_cumulative_hierarchy_emission_and_e2e_validation"
        },
    )
    assert (
        main(
            [
                "--dataset",
                "cohort.parquet",
                "--stage1-config",
                "stage1.json",
                "--embedding-cache-dir",
                "cache",
                "--output-dir",
                "bundle",
                "--unit-id-column",
                "person_key",
                "--dry-run",
            ]
        )
        == 2
    )


def test_query_config_scientific_identity_is_content_addressed(tmp_path):
    path = tmp_path / "query.json"
    payload = json.dumps(
        asdict(NeuralQueryAgenticForestConfig()),
        sort_keys=True,
    )
    path.write_text(payload, encoding="utf-8")
    _config, first = ProductionStage1BundleBuilder._load_query_config(path)

    replacement = tmp_path / "query.replacement.json"
    replacement.write_text(payload, encoding="utf-8")
    replacement.replace(path)
    _config, second = ProductionStage1BundleBuilder._load_query_config(path)

    assert first["stat_identity"] != second["stat_identity"]
    assert _scientific_query_config_identity(first) == (
        _scientific_query_config_identity(second)
    )


def test_candidate_bundle_build_is_enabled_without_claiming_e2e_certification():
    gate = exact_inner_family_adapter_gate()
    assert gate["production_execution_ready"] is True
    assert gate["candidate_bundle_build_ready"] is True
    assert gate["genuine_one_shot_e2e_certified"] is False
    assert tuple(gate["registered_component_proof_families"]) == (ACTIVE_STAGE1_CONCEPT_FAMILIES)
    expected_missing = tuple(
        family
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        if family not in PRODUCTION_REGISTERED_NATIVE_FAMILY_ADAPTERS
    )
    assert tuple(gate["missing_registered_family_producers"]) == expected_missing
    assert tuple(gate["unregistered_component_proof_families"]) == expected_missing
    assert expected_missing == ()
    assert gate["native_exact_inner_registration_complete"] is True
    assert gate["registered_component_proof_family_count"] == 10
    assert gate["integration_substrate_blockers"] == []
    assert gate["certification_blockers"]


def test_exact_inner_gate_declares_label_access_policy_for_every_family():
    gate = exact_inner_family_adapter_gate()
    policy = gate["family_label_access_policy"]
    assert tuple(policy) == ACTIVE_STAGE1_CONCEPT_FAMILIES
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        assert policy[family] == {
            "fit_text_available": True,
            "fit_treatment_available": True,
            "fit_outcome_available": True,
            "heldout_text_available": True,
            "heldout_treatment_available": False,
            "heldout_outcome_available": False,
            "oracle_fields_available": False,
            "secrets_available": False,
        }
    implementation = gate["native_adapter_implementation_by_family"]
    assert tuple(implementation) == ACTIVE_STAGE1_CONCEPT_FAMILIES
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        assert implementation[family]["implementation_available"] is True
        assert implementation[family]["production_wrapper_registered"] is (
            family in PRODUCTION_REGISTERED_NATIVE_FAMILY_ADAPTERS
        )
        assert implementation[family]["fit_apis"]
        blockers = gate["family_adapter_blockers"][family]
        if family in PRODUCTION_REGISTERED_NATIVE_FAMILY_ADAPTERS:
            assert blockers == []
        else:
            assert any("not registered by the wrapper" in blocker for blocker in blockers)
    resolved = " ".join(gate["resolved_integration_hardening"])
    assert "nested fit/calibration label selection" in resolved
    assert "semantic-retrieval TF-IDF projection is label-free" in resolved


def _write_genuine_neural_query_native_scope(root: Path) -> dict[str, object]:
    root.mkdir(parents=True)
    (root / "artifacts").mkdir()
    model_root = root / "native_models" / "outer_001_inner_001"
    model_root.mkdir(parents=True)
    fit_rows = (0, 1, 2, 3)
    heldout_rows = (4, 5)
    fit_treatment = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=float)
    fit_outcome = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=float)
    service_identity = {"service": "unit_test_owned_neural_query_service"}
    service_identity_sha256 = query_context_module._sha256_json(service_identity)
    binding = {
        "service_identity_sha256": service_identity_sha256,
        "outer_fold": 1,
        "row_ids": list(fit_rows),
        "text_sha256": "1" * 64,
        "treatment_sha256": query_context_module._float_hex_sha256(fit_treatment),
        "outcome_sha256": query_context_module._float_hex_sha256(fit_outcome),
        "row_count": len(fit_rows),
        "embedding_row_binding_sha256": "4" * 64,
    }
    cache_key = query_context_module._sha256_json(binding)
    discovery = {
        "runtime": query_context_module.NEURAL_QUERY_DISCOVERY_RUNTIME_ID,
        "fit_input_binding_sha256": "5" * 64,
        "fit_nuisance_output_binding": {
            "schema_version": query_context_module.NEURAL_QUERY_NUISANCE_OUTPUT_BINDING_SCHEMA,
            "fit_row_ids": list(fit_rows),
            "fit_e_sha256": "6" * 64,
            "fit_m_sha256": "7" * 64,
            "heldout_labels_accessed": False,
        },
        "banks": {
            bank: {
                "queries": np.asarray([[1.0, 0.0]], dtype=np.float32),
                "train_activations": np.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32),
                "records": [
                    {
                        "query_id": f"{bank}_context_query_001",
                        "fit_standardized_score": 0.5,
                        "member_count": 2,
                    }
                ],
                "consensus": {"method": "ungated_test_consensus"},
                "objective": f"test_{bank}_objective",
                "all_queries_retained": True,
                "statistical_gate_applied": False,
            }
            for bank in ("treatment", "outcome", "effect")
        },
        "subfold_audit": [],
        "all_queries_retained": True,
        "validation_audits_used_for_selection": False,
        "executable_checkpoint_io": False,
    }
    service = object.__new__(query_context_module.ContextFitNeuralQueryService)
    service._identity = service_identity
    service.identity = lambda: copy.deepcopy(service_identity)
    service._owned_discoveries = {cache_key: copy.deepcopy(discovery)}
    service._owned_discovery_bindings = {cache_key: copy.deepcopy(binding)}
    service._owned_discovery_content_sha256s = {
        cache_key: query_context_module._owned_discovery_memory_sha256(discovery)
    }
    owned_snapshot = service.write_owned_discovery_snapshot(
        cache_key=cache_key,
        output_dir=model_root / "owned_snapshot",
    )
    split_scope_fingerprint = _sha256_json({"scope": "outer_001_inner_001"})
    data_projection_sha256 = _sha256_json({"projection": "fit-labels-heldout-id-text"})
    prediction = SimpleNamespace(
        gate_row_ids=heldout_rows,
        feature_values=np.asarray([[0.2, 0.6, 0.3], [0.4, 0.7, 0.1]], dtype=np.float32),
        feature_names=(
            "neural_query_treatment_signed_mean",
            "neural_query_outcome_signed_mean",
            "neural_query_effect_signed_mean",
        ),
        feature_kinds=(
            "neural_query_treatment_moments",
            "neural_query_outcome_moments",
            "neural_query_effect_moments",
        ),
        feature_roles=("propensity", "outcome", "effect_modifier"),
    )
    moment_metadata = _write_neural_query_moment_artifact(
        model_root,
        scope_id="outer_001_inner_001",
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        split_scope_fingerprint=split_scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        query_cache_key=cache_key,
        owned_snapshot_metadata=owned_snapshot,
        text_column="clinical_text",
        prediction=prediction,
    )
    model_registration = _component_native_artifact_registration(
        model_root,
        component_root=root,
    )
    query_evidence = [
        {
            "query_id": f"{bank}_context_query_001",
            "bank": bank,
            "mechanical_role": "effect_modifier" if bank == "effect" else "confounder",
            "statistical_gate_applied": False,
            "member_count": 2,
            "fit_standardized_score": 0.5,
            "top_chunks": [],
            "top_contrastive_ngrams": [{"term": f"{bank} baseline marker", "tfidf_contrast": 0.8}],
        }
        for bank in ("treatment", "outcome", "effect")
    ]
    source_payload = {
        "schema_version": STAGE1_QUERY_ARTIFACT_SCHEMA,
        "source_kind": NEURAL_QUERY_SOURCE,
        "source_family": NEURAL_QUERY_MOMENTS,
        "adapter_mode": "authenticated_neural_query_artifact",
        "scope_id": "outer_001_inner_001",
        "outer_fold": 1,
        "inner_fold": 1,
        "scope": "inner_train",
        "fit_row_ids": list(fit_rows),
        "heldout_row_ids": list(heldout_rows),
        "fit_row_fingerprint": row_set_fingerprint(fit_rows),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_rows),
        "split_registry_content_sha256": "8" * 64,
        "query_cache_key": cache_key,
        "heldout_labels_supplied": False,
        "heldout_columns_read": ["_oci_row_id", "clinical_text"],
        "native_model_artifact": {
            "relative_path": model_registration["relative_path"],
            "sha256": model_registration["sha256"],
        },
        "heldout_moment_artifact": {
            "relative_path": (model_root / "heldout_moments.npz").relative_to(root).as_posix(),
            "sha256": moment_metadata["arrays_sha256"],
            "content_sha256": moment_metadata["content_sha256"],
        },
        "query_evidence": query_evidence,
    }
    source_path = root / "artifacts" / "outer_001_inner_001.json"
    _write_immutable_json(source_path, source_payload)
    provenance = FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        scope="inner_train",
        inner_fold=1,
        artifact_id="genuine-neural-query-proof-test",
    )
    catalog = build_role_neutral_evidence_catalog(
        (FoldEvidenceInput(NEURAL_QUERY_SOURCE, source_payload, provenance),),
        require_all_source_kinds=False,
        require_all_architecture_families=False,
        require_upstream_completeness=False,
    )
    configuration = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_REGISTRATION_SCHEMA,
        "scope_id": "outer_001_inner_001",
        "text_column": "clinical_text",
        "query_config": {"treatment_query_count": 1},
        "query_nuisance_folds": 2,
        "query_devices": ["cpu"],
        "seed": 13,
        "outcome_type": "binary",
        "split_registry_content_sha256": "8" * 64,
        "service_identity_sha256": service_identity_sha256,
        "heldout_label_policy": "id_and_text_only",
    }
    registration = _register_neural_query_native_family_proof(
        component_root=root,
        proof_directory=Path("native_family_proofs") / "outer_001_inner_001",
        scope_id="outer_001_inner_001",
        catalog=catalog,
        query_artifact_path=source_path,
        model_artifact_path=model_root,
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=fit_rows,
        heldout_row_ids=heldout_rows,
        fit_treatment=fit_treatment,
        fit_outcome=fit_outcome,
        split_scope_fingerprint=split_scope_fingerprint,
        data_projection_sha256=data_projection_sha256,
        configuration=configuration,
    )
    index_body = {
        "schema_version": STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA,
        "split_registry_content_sha256": "8" * 64,
        "registered_families": [NEURAL_QUERY_MOMENTS],
        "exact_inner_scope_count": 1,
        "scopes": [
            {
                "scope_id": "outer_001_inner_001",
                "outer_fold": 1,
                "inner_fold": 1,
                "registered_families": [NEURAL_QUERY_MOMENTS],
                "content_sha256": registration["content_sha256"],
                "registration": registration["registration"],
            }
        ],
        "executable_checkpoint_files_retained": False,
    }
    index = {**index_body, "content_sha256": _sha256_json(index_body)}
    index_path = root / "native_family_proof_index.json"
    _write_immutable_json(index_path, index)
    index_registration = _component_file_registration(index_path, component_root=root)
    expected_scopes = {
        "outer_001_inner_001": {
            "scope_id": "outer_001_inner_001",
            "outer_fold": 1,
            "inner_fold": 1,
            "fit_row_ids": list(fit_rows),
            "heldout_row_ids": list(heldout_rows),
        }
    }
    return {
        "root": root,
        "model_root": model_root,
        "source_path": source_path,
        "registration": registration,
        "index_registration": index_registration,
        "expected_scopes": expected_scopes,
        "fit_rows": fit_rows,
        "heldout_rows": heldout_rows,
        "fit_treatment": fit_treatment,
        "fit_outcome": fit_outcome,
        "catalog": catalog,
        "configuration": configuration,
        "split_scope_fingerprint": split_scope_fingerprint,
        "data_projection_sha256": data_projection_sha256,
        "modeling_data": pd.DataFrame(
            {
                "treatment_indicator": [*fit_treatment, 0.0, 1.0],
                "outcome_indicator": [*fit_outcome, 1.0, 0.0],
            }
        ),
    }


def test_neural_query_native_registration_binds_real_arrays_moments_and_index(tmp_path):
    built = _write_genuine_neural_query_native_scope(tmp_path / "query")
    validated = _validate_neural_query_native_family_proof_index(
        component_root=built["root"],
        index_registration=built["index_registration"],
        expected_inner_scopes=built["expected_scopes"],
        split_registry_content_sha256="8" * 64,
        modeling_data=built["modeling_data"],
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
    )
    registration = built["registration"]
    [family_row] = registration["family_proofs"]
    assert validated["registered_families"] == [NEURAL_QUERY_MOMENTS]
    assert family_row["proof"]["heldout_labels_accessed"] is False
    assert family_row["proof"]["model_artifact_sha256"] == (family_row["model_artifact"]["sha256"])
    assert family_row["proof"]["source_artifact_sha256"] == _sha256_file(built["source_path"])
    assert not list(Path(built["root"]).rglob("*.joblib"))
    with np.load(Path(built["model_root"]) / "heldout_moments.npz", allow_pickle=False) as data:
        assert tuple(map(int, data["heldout_row_ids"].tolist())) == built["heldout_rows"]
        assert data["feature_values"].shape == (2, 3)


@pytest.mark.parametrize("tamper_target", ["owned_queries", "heldout_moments", "source"])
def test_neural_query_native_registration_rejects_tamper(tmp_path, tamper_target):
    built = _write_genuine_neural_query_native_scope(tmp_path / tamper_target / "query")
    if tamper_target == "owned_queries":
        target = Path(built["model_root"]) / "owned_snapshot" / "arrays.npz"
    elif tamper_target == "heldout_moments":
        target = Path(built["model_root"]) / "heldout_moments.npz"
    else:
        target = Path(built["source_path"])
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="changed"):
        _validate_neural_query_native_family_proof_index(
            component_root=built["root"],
            index_registration=built["index_registration"],
            expected_inner_scopes=built["expected_scopes"],
            split_registry_content_sha256="8" * 64,
            modeling_data=built["modeling_data"],
            treatment_column="treatment_indicator",
            outcome_column="outcome_indicator",
        )


@pytest.mark.parametrize("column", ["treatment_indicator", "outcome_indicator"])
def test_neural_query_native_registration_rejects_canonical_label_drift(tmp_path, column):
    built = _write_genuine_neural_query_native_scope(tmp_path / column / "query")
    drifted = built["modeling_data"].copy()
    drifted.loc[0, column] = 1.0 - float(drifted.loc[0, column])
    with pytest.raises(ValueError, match="canonical fit labels"):
        _validate_neural_query_native_family_proof_index(
            component_root=built["root"],
            index_registration=built["index_registration"],
            expected_inner_scopes=built["expected_scopes"],
            split_registry_content_sha256="8" * 64,
            modeling_data=drifted,
            treatment_column="treatment_indicator",
            outcome_column="outcome_indicator",
        )


def test_query_component_emits_owned_snapshots_exact_moments_and_native_proofs(
    tmp_path,
    monkeypatch,
):
    class FakeService:
        write_owned_discovery_snapshot = (
            query_context_module.ContextFitNeuralQueryService.write_owned_discovery_snapshot
        )

        def __init__(self, **_kwargs):
            self._identity = {"service": "fake_component_owned_query_service"}
            self._owned_discoveries = {}
            self._owned_discovery_bindings = {}
            self._owned_discovery_content_sha256s = {}

        def identity(self):
            return copy.deepcopy(self._identity)

        def discovery_for_context(
            self,
            *,
            outer_fold,
            context_row_ids,
            context_texts,
            context_treatment,
            context_outcome,
        ):
            fit_rows = tuple(map(int, context_row_ids))
            assert len(context_texts) == len(context_treatment) == len(context_outcome)
            binding = {
                "service_identity_sha256": query_context_module._sha256_json(self._identity),
                "outer_fold": int(outer_fold),
                "row_ids": list(fit_rows),
                "text_sha256": query_context_module._sha256_json(list(context_texts)),
                "treatment_sha256": query_context_module._float_hex_sha256(context_treatment),
                "outcome_sha256": query_context_module._float_hex_sha256(context_outcome),
                "row_count": len(fit_rows),
                "embedding_row_binding_sha256": "3" * 64,
            }
            cache_key = query_context_module._sha256_json(binding)
            discovery = {
                "runtime": query_context_module.NEURAL_QUERY_DISCOVERY_RUNTIME_ID,
                "fit_input_binding_sha256": "4" * 64,
                "fit_nuisance_output_binding": {
                    "schema_version": (
                        query_context_module.NEURAL_QUERY_NUISANCE_OUTPUT_BINDING_SCHEMA
                    ),
                    "fit_row_ids": list(fit_rows),
                    "fit_e_sha256": "5" * 64,
                    "fit_m_sha256": "6" * 64,
                    "heldout_labels_accessed": False,
                },
                "banks": {
                    bank: {
                        "queries": np.asarray([[1.0, 0.0]], dtype=np.float32),
                        "train_activations": np.linspace(
                            0.1,
                            0.9,
                            num=len(fit_rows),
                            dtype=np.float32,
                        )[:, None],
                        "records": [
                            {
                                "query_id": f"{bank}_context_query_001",
                                "fit_standardized_score": 0.5,
                                "member_count": 2,
                            }
                        ],
                        "consensus": {"method": "ungated_component_test"},
                        "objective": f"test_{bank}_objective",
                        "all_queries_retained": True,
                        "statistical_gate_applied": False,
                    }
                    for bank in ("treatment", "outcome", "effect")
                },
                "subfold_audit": [],
                "all_queries_retained": True,
                "validation_audits_used_for_selection": False,
                "executable_checkpoint_io": False,
            }
            self._owned_discoveries[cache_key] = copy.deepcopy(discovery)
            self._owned_discovery_bindings[cache_key] = copy.deepcopy(binding)
            self._owned_discovery_content_sha256s[cache_key] = (
                query_context_module._owned_discovery_memory_sha256(discovery)
            )
            return copy.deepcopy(discovery), cache_key

        def safe_evidence(self, *, discovery, **_kwargs):
            assert discovery["executable_checkpoint_io"] is False
            return [
                {
                    "query_id": f"{bank}_context_query_001",
                    "bank": bank,
                    "mechanical_role": ("effect_modifier" if bank == "effect" else "confounder"),
                    "statistical_gate_applied": False,
                    "member_count": 2,
                    "fit_standardized_score": 0.5,
                    "top_chunks": [],
                    "top_contrastive_ngrams": [
                        {"term": f"{bank} component marker", "tfidf_contrast": 0.7}
                    ],
                }
                for bank in ("treatment", "outcome", "effect")
            ]

    class FakeBackend:
        def __init__(self, service):
            self.service = service

        def identity(self):
            return {"backend": "fake_exact_label_free_query_backend"}

        def fit_predict(self, **kwargs):
            assert set(kwargs) == {
                "outer_fold",
                "context_row_ids",
                "context_texts",
                "context_treatment",
                "context_outcome",
                "gate_row_ids",
                "gate_texts",
                "work_dir",
            }
            gate_rows = tuple(map(int, kwargs["gate_row_ids"]))
            assert len(gate_rows) == len(kwargs["gate_texts"])
            values = np.column_stack(
                [
                    np.asarray(gate_rows, dtype=float) / 10.0,
                    np.full(len(gate_rows), 0.5),
                    np.full(len(gate_rows), -0.25),
                ]
            )
            return SimpleNamespace(
                gate_row_ids=gate_rows,
                feature_values=values,
                feature_names=(
                    "neural_query_treatment_signed_mean",
                    "neural_query_outcome_signed_mean",
                    "neural_query_effect_signed_mean",
                ),
                feature_kinds=(
                    "neural_query_treatment_moments",
                    "neural_query_outcome_moments",
                    "neural_query_effect_moments",
                ),
                feature_roles=("propensity", "outcome", "effect_modifier"),
            )

    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle.ContextFitNeuralQueryService",
        FakeService,
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle.NeuralQueryContextBackend",
        FakeBackend,
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle._canonical_cumulative_spent_schedule",
        lambda _registry: SimpleNamespace(scopes=(), schedule_sha256="7" * 64),
    )
    exact = CanonicalStage1SplitRegistry.build(
        dataset_row_ids=tuple(range(12)),
        outer_heldout_row_ids={
            1: (0, 1, 2, 3),
            2: (4, 5, 6, 7),
            3: (8, 9, 10, 11),
        },
        inner_fold_count=2,
        inner_seed_base=51_000,
    )
    registry = {
        "dataset_row_count": 12,
        "inner_seed_base": 51_000,
        "outer_folds": [
            {
                "outer_fold": outer.outer_fold,
                "fit_row_ids": list(outer.train_row_ids),
                "heldout_row_ids": list(outer.heldout_row_ids),
                "inner_folds": [
                    {
                        "inner_fold": inner.inner_fold,
                        "fit_row_ids": list(inner.fit_row_ids),
                        "heldout_row_ids": list(inner.heldout_row_ids),
                    }
                    for inner in outer.inner_splits
                ],
            }
            for outer in exact.outer_splits
        ],
    }
    modeling_data = pd.DataFrame(
        {
            "clinical_text": [f"patient baseline text {index}" for index in range(12)],
            "treatment_indicator": [index % 2 for index in range(12)],
            "outcome_indicator": [(index // 2) % 2 for index in range(12)],
        }
    )
    query_config = NeuralQueryAgenticForestConfig(
        treatment_query_count=1,
        outcome_query_count=1,
        effect_query_count=1,
        query_inner_folds=2,
        initial_pool_size=1,
        query_epochs=1,
        final_refit_epochs=1,
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "stage1_config.json").write_text("{}\n", encoding="utf-8")
    prepared = SimpleNamespace(
        modeling_data=modeling_data,
        registry=registry,
        registry_content_sha256=_sha256_json(registry),
        request_sha256="a" * 64,
        config=SimpleNamespace(
            text_column="clinical_text",
            treatment_column="treatment_indicator",
            outcome_column="outcome_indicator",
            outcome_type="binary",
        ),
        options=SimpleNamespace(
            dataset_path=tmp_path / "cohort.parquet",
            query_nuisance_folds=2,
            query_devices=("cpu",),
            seed=23,
        ),
        embedding_cache=object(),
        query_config=query_config,
    )
    root = tmp_path / "query_component"
    builder = ProductionStage1BundleBuilder(
        Stage1BundleBuildOptions(
            dataset_path=tmp_path / "cohort.parquet",
            config_path=tmp_path / "config.json",
            embedding_cache_dir=tmp_path / "embedding_cache",
            output_dir=output,
            unit_id_column="unit_id",
        )
    )
    builder._run_query_component(root, output, prepared)

    query_index = json.loads((root / "query_artifact_index.json").read_text(encoding="utf-8"))
    proof_index = json.loads((root / "native_family_proof_index.json").read_text(encoding="utf-8"))
    assert len(query_index["scopes"]) == 9
    assert proof_index["exact_inner_scope_count"] == 6
    assert proof_index["registered_families"] == [NEURAL_QUERY_MOMENTS]
    assert query_index["heldout_labels_supplied"] is False
    assert query_index["executable_checkpoint_files_retained"] is False
    assert not list(root.rglob("*.joblib"))
    assert all(
        (root / row["heldout_moment_arrays"]["relative_path"]).is_file()
        for row in query_index["scopes"]
    )


def test_effective_config_requires_every_legacy_architecture(tmp_path: Path):
    config, model_dir = _valid_config(tmp_path)
    validated, htr_path = _validate_effective_config(
        config,
        dataset_path=tmp_path / "cohort.parquet",
        embedding_cache_dir=tmp_path / "cache",
        config_dir=tmp_path,
        seed=19,
    )
    assert htr_path == model_dir
    assert validated.architecture.htr_require_live_unfrozen_encoder_attestation is True
    assert validated.architecture.multi_model_forest.outer_parallelism == "1"

    missing_htr = copy.deepcopy(config)
    missing_htr.architecture.multi_model_forest.htr_evidence_enabled = False
    with pytest.raises(ValueError, match="requires HTR evidence"):
        _validate_effective_config(
            missing_htr,
            dataset_path=tmp_path / "cohort.parquet",
            embedding_cache_dir=tmp_path / "cache",
            config_dir=tmp_path,
            seed=19,
        )

    too_few_hierarchy_partitions = copy.deepcopy(config)
    (
        too_few_hierarchy_partitions.architecture.multi_model_forest.candidate_consistency_inner_folds
    ) = 3
    with pytest.raises(ValueError, match="at least four"):
        _validate_effective_config(
            too_few_hierarchy_partitions,
            dataset_path=tmp_path / "cohort.parquet",
            embedding_cache_dir=tmp_path / "cache",
            config_dir=tmp_path,
            seed=19,
        )

    one_cluster_component = copy.deepcopy(config)
    one_cluster_component.architecture.multi_model_forest.embedding_contrast.cluster_contrast_max_components = (
        1
    )
    with pytest.raises(ValueError, match="at least two emitted components"):
        _validate_effective_config(
            one_cluster_component,
            dataset_path=tmp_path / "cohort.parquet",
            embedding_cache_dir=tmp_path / "cache",
            config_dir=tmp_path,
            seed=19,
        )


def test_htr_input_audit_is_lossless_and_closed(tmp_path: Path):
    config, model_dir = _valid_config(tmp_path)
    model_sha = "a" * 64
    audit = _build_htr_input_nontruncation_audit(
        texts=("safe baseline text", "safe note one two three"),
        config=config,
        htr_model_path=model_dir,
        htr_model_tree_sha256=model_sha,
    )

    assert audit["chunk_cap_nonbinding"] is True
    assert audit["tokenizer_truncation_allowed"] is False
    assert audit["semantic_truncation_allowed"] is False
    assert audit["max_observed_token_count"] <= audit["effective_max_chunk_length"]
    assert audit["applies_to_families"] == [HTR_NEURAL, MATCHED_PAIR_UPLIFT]
    assert (
        validate_htr_input_nontruncation_audit(
            audit,
            config=asdict(config),
            expected_rows=2,
            expected_htr_model_tree_sha256=model_sha,
        )
        == audit
    )


def test_htr_input_audit_rejects_word_cap_before_tokenizer_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config, model_dir = _valid_config(tmp_path)
    config.architecture.htr_chunk_size_words = 2
    config.architecture.htr_chunk_overlap_words = 0
    config.architecture.htr_max_chunks = 1

    def forbidden(_path):
        raise AssertionError("tokenizer must not load before the HTR word-cap preflight")

    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle._load_local_htr_tokenizer",
        forbidden,
    )
    with pytest.raises(ValueError, match="HTR max_chunks would cause semantic truncation"):
        _build_htr_input_nontruncation_audit(
            texts=("one two three four",),
            config=config,
            htr_model_path=model_dir,
            htr_model_tree_sha256="a" * 64,
        )


def test_htr_input_audit_rejects_tokenizer_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config, model_dir = _valid_config(tmp_path)
    config.architecture.htr_chunk_size_words = 3
    config.architecture.htr_chunk_overlap_words = 0
    config.architecture.htr_max_chunks = 2
    config.architecture.htr_max_chunk_length = 4

    class OverflowTokenizer:
        vocab_size = 10
        model_max_length = 512

        def __call__(
            self,
            chunks,
            *,
            add_special_tokens,
            padding,
            truncation,
            return_length,
        ):
            assert add_special_tokens is True
            assert padding is False
            assert truncation is False
            assert return_length is True
            return {"length": [5 for _chunk in chunks]}

    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle._load_local_htr_tokenizer",
        lambda _path: OverflowTokenizer(),
    )
    with pytest.raises(ValueError, match="HTR tokenizer would cause semantic truncation"):
        _build_htr_input_nontruncation_audit(
            texts=("one two three",),
            config=config,
            htr_model_path=model_dir,
            htr_model_tree_sha256="a" * 64,
        )


@pytest.mark.parametrize("chunk_selection", [None, "first"])
def test_cache_configuration_requires_explicit_last_chunk_selection(
    tmp_path: Path,
    chunk_selection: str | None,
):
    config, _model_dir = _valid_config(tmp_path)
    embedding = config.architecture.multi_model_forest.embedding_contrast
    metadata = {
        "sentence_model_name": str(embedding.model_name),
        "chunk_size_words": int(embedding.chunk_size_words),
        "chunk_overlap_words": int(embedding.chunk_overlap_words),
        "max_chunks": int(embedding.max_chunks),
        "normalize_embeddings": bool(embedding.normalize_embeddings),
        "max_seq_length": embedding.max_seq_length,
    }
    if chunk_selection is not None:
        metadata["chunk_selection"] = chunk_selection

    with pytest.raises(ValueError, match="chunk_selection"):
        _validate_cache_configuration(SimpleNamespace(metadata=metadata), config)

    metadata["chunk_selection"] = "last"
    _validate_cache_configuration(SimpleNamespace(metadata=metadata), config)


def test_embedding_cluster_preflight_enumerates_all_native_scopes_with_real_catalogs(
    tmp_path: Path,
):
    case = _cluster_preflight_case(tmp_path)
    audit = build_embedding_cluster_feasibility_audit(
        modeling_data=case["modeling_data"],
        config=case["config"],
        embedding_cache=case["cache"],
        embedding_cache_identity=case["cache_identity"],
        registry=case["registry"],
        registry_content_sha256=case["registry_sha256"],
    )
    repeated_audit = build_embedding_cluster_feasibility_audit(
        modeling_data=case["modeling_data"],
        config=case["config"],
        embedding_cache=case["cache"],
        embedding_cache_identity=case["cache_identity"],
        registry=case["registry"],
        registry_content_sha256=case["registry_sha256"],
        preflight_workers=2,
    )
    assert repeated_audit == audit
    assert repeated_audit["content_sha256"] == audit["content_sha256"]

    assert audit["schema_version"] == STAGE1_EMBEDDING_CLUSTER_FEASIBILITY_AUDIT_SCHEMA
    assert audit["scope_count"] == 12
    assert audit["full_outer_scope_count"] == 2
    assert audit["exact_inner_scope_count"] == 8
    assert audit["cumulative_spent_scope_count"] == 2
    assert audit["scope_order"] == [row["scope_id"] for row in audit["scopes"]]
    assert audit["token_bounded_row_count"] == 0
    assert audit["token_bounded_row_ids_sha256"] == _sha256_json([])
    assert all(row["catalog_atom_count"] > 0 for row in audit["scopes"])
    assert all(row["catalog_member_count"] > 0 for row in audit["scopes"])
    assert all(
        all(value >= 2 for value in row["raw_contrast_count_by_family"].values())
        and all(value >= 2 for value in row["semantic_contrast_count_by_family"].values())
        and all(value >= 2 for value in row["catalog_grounded_component_count_by_family"].values())
        for row in audit["scopes"]
    )
    assert all(
        row["cluster_support_contract"]["kmeans_cluster_count"] == audit["configured_cluster_count"]
        for row in audit["scopes"]
    )
    assert all(
        row["token_bounded_row_count"] == 0
        and row["token_bounded_row_ids_sha256"] == _sha256_json([])
        and row["cluster_support_contract"]["kmeans_parameters"]
        == {
            "n_clusters": audit["configured_cluster_count"],
            "random_state": 42,
            "batch_size": max(128, min(1024, row["fit_row_count"])),
            "n_init": 3,
            "max_iter": 300,
        }
        for row in audit["scopes"]
    )
    for scope in audit["scopes"]:
        assert scope["semantic_mirror_catalog_atom_count"] > 0
        assert scope["semantic_mirror_catalog_member_count"] > 0
        for coverage in scope["component_coverage_by_family"]:
            component_ids = coverage["raw_component_ids"]
            assert component_ids == coverage["semantic_component_ids"]
            assert component_ids == coverage["embedding_clustered_component_ids"]
            assert component_ids == coverage["tfidf_semantic_retrieval_component_ids"]
            assert all(
                coverage[field] == len(component_ids)
                for field in (
                    "raw_component_count",
                    "semantic_component_count",
                    "embedding_clustered_component_count",
                    "tfidf_semantic_retrieval_component_count",
                )
            )
            assert all(
                all(count > 0 for count in coverage[field])
                for field in (
                    "raw_positive_member_counts",
                    "raw_negative_member_counts",
                    "semantic_member_counts",
                    "embedding_clustered_member_counts",
                    "tfidf_semantic_retrieval_member_counts",
                )
            )
            assert (
                coverage["semantic_member_counts"] == coverage["embedding_clustered_member_counts"]
            )
            assert (
                coverage["semantic_member_counts"]
                == coverage["tfidf_semantic_retrieval_member_counts"]
            )
            assert (
                coverage["embedding_clustered_parent_collection_sha256"]
                == coverage["tfidf_semantic_retrieval_parent_collection_sha256"]
            )
            assert coverage["tfidf_semantic_retrieval_parent_family"] == ("embedding_clustered")
    assert all("kmeans_inertia" not in row["cluster_support_contract"] for row in audit["scopes"])
    assert all(
        {family["family_key"] for family in row["cluster_support_contract"]["svd_families"]}
        == {"treatment", "residualized_interaction"}
        for row in audit["scopes"]
    )
    assert all(
        family["local_contrast_count"] >= 2
        and family["numerical_rank"] >= 2
        and family["second_singular_value"] > 0.0
        for row in audit["scopes"]
        for family in row["cluster_support_contract"]["svd_families"]
    )
    assert (
        validate_embedding_cluster_feasibility_audit(
            audit,
            config=case["config"],
            registry=case["registry"],
            registry_content_sha256=case["registry_sha256"],
            embedding_cache_identity=case["cache_identity"],
        )
        == audit
    )

    truncated = copy.deepcopy(audit)
    truncated["scopes"].pop()
    truncated["content_sha256"] = _sha256_json(
        {key: value for key, value in truncated.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="invalid closed envelope"):
        validate_embedding_cluster_feasibility_audit(
            truncated,
            config=case["config"],
            registry=case["registry"],
            registry_content_sha256=case["registry_sha256"],
            embedding_cache_identity=case["cache_identity"],
        )

    wrong_effective_k = copy.deepcopy(audit)
    support = wrong_effective_k["scopes"][0]["cluster_support_contract"]
    support["kmeans_cluster_count"] -= 1
    support["kmeans_cluster_counts"][0] += support["kmeans_cluster_counts"].pop()
    wrong_effective_k["content_sha256"] = _sha256_json(
        {key: value for key, value in wrong_effective_k.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="infeasible in outer_001_full"):
        validate_embedding_cluster_feasibility_audit(
            wrong_effective_k,
            config=case["config"],
            registry=case["registry"],
            registry_content_sha256=case["registry_sha256"],
            embedding_cache_identity=case["cache_identity"],
        )

    tamper_cases = []
    missing_semantic = copy.deepcopy(audit)
    missing_semantic["scopes"][0]["component_coverage_by_family"][0]["semantic_component_ids"][
        0
    ] = "missing-semantic-component"
    tamper_cases.append(missing_semantic)
    wrong_mirror_parent = copy.deepcopy(audit)
    wrong_mirror_parent["scopes"][0]["component_coverage_by_family"][0][
        "tfidf_semantic_retrieval_parent_collection_sha256"
    ][0] = ("f" * 64)
    tamper_cases.append(wrong_mirror_parent)
    empty_member = copy.deepcopy(audit)
    empty_member["scopes"][0]["component_coverage_by_family"][0][
        "tfidf_semantic_retrieval_member_counts"
    ][0] = 0
    tamper_cases.append(empty_member)
    wrong_parameters = copy.deepcopy(audit)
    wrong_parameters["scopes"][0]["cluster_support_contract"]["kmeans_parameters"]["n_init"] += 1
    tamper_cases.append(wrong_parameters)
    for tampered in tamper_cases:
        tampered["content_sha256"] = _sha256_json(
            {key: value for key, value in tampered.items() if key != "content_sha256"}
        )
        with pytest.raises(ValueError, match="infeasible in outer_001_full"):
            validate_embedding_cluster_feasibility_audit(
                tampered,
                config=case["config"],
                registry=case["registry"],
                registry_content_sha256=case["registry_sha256"],
                embedding_cache_identity=case["cache_identity"],
            )


def test_embedding_cluster_preflight_rejects_token_bounded_cache_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    case = _cluster_preflight_case(tmp_path)
    original_bind = case["cache"].bind_spent

    def token_bounded_bind(row_ids, texts):
        provider = original_bind(row_ids, texts)
        provider.token_bounded_row_ids = (int(tuple(row_ids)[0]),)
        return provider

    monkeypatch.setattr(case["cache"], "bind_spent", token_bounded_bind)
    with pytest.raises(
        ValueError,
        match="token-bounded text reconciliation in outer_001_full",
    ):
        build_embedding_cluster_feasibility_audit(
            modeling_data=case["modeling_data"],
            config=case["config"],
            embedding_cache=case["cache"],
            embedding_cache_identity=case["cache_identity"],
            registry=case["registry"],
            registry_content_sha256=case["registry_sha256"],
        )


def test_embedding_cluster_preflight_reports_first_infeasible_ordered_scope(
    tmp_path: Path,
):
    case = _cluster_preflight_case(tmp_path)
    failing_config = copy.deepcopy(case["config"])
    failing_config.architecture.multi_model_forest.embedding_contrast.cluster_contrast_min_cluster_size = (
        200
    )
    with pytest.raises(
        ValueError,
        match=r"infeasible in outer_001_full; observed_cluster_summary=",
    ):
        build_embedding_cluster_feasibility_audit(
            modeling_data=case["modeling_data"],
            config=failing_config,
            embedding_cache=case["cache"],
            embedding_cache_identity=case["cache_identity"],
            registry=case["registry"],
            registry_content_sha256=case["registry_sha256"],
        )


def test_embedding_cluster_preflight_rejects_later_exact_scope_with_kmeans_but_no_svds(
    tmp_path: Path,
):
    case = _cluster_preflight_case(tmp_path)
    config = copy.deepcopy(case["config"])
    embedding = config.architecture.multi_model_forest.embedding_contrast
    embedding.cluster_contrast_min_group_size = 2
    embedding.cluster_contrast_min_cell_size = 1
    modeling_data = case["modeling_data"].copy(deep=True)
    outer = case["registry"]["outer_folds"][1]
    target_partition = set(outer["inner_folds"][3]["heldout_row_ids"])
    for row_id in outer["fit_row_ids"]:
        cluster = (int(row_id) // 4) % 8
        if cluster == 0 or (cluster == 1 and int(row_id) in target_partition):
            continue
        modeling_data.loc[int(row_id), config.treatment_column] = 0
        modeling_data.loc[int(row_id), config.outcome_column] = 0
    target_cluster_rows = sorted(
        row_id for row_id in target_partition if (int(row_id) // 4) % 8 == 1
    )
    assert len(target_cluster_rows) >= 4
    balanced_cells = ((0, 0), (0, 1), (1, 0), (1, 1))
    for index, row_id in enumerate(target_cluster_rows):
        treatment, outcome = balanced_cells[index % len(balanced_cells)]
        modeling_data.loc[int(row_id), config.treatment_column] = treatment
        modeling_data.loc[int(row_id), config.outcome_column] = outcome

    with pytest.raises(ValueError) as captured:
        build_embedding_cluster_feasibility_audit(
            modeling_data=modeling_data,
            config=config,
            embedding_cache=case["cache"],
            embedding_cache_identity=case["cache_identity"],
            registry=case["registry"],
            registry_content_sha256=case["registry_sha256"],
        )
    message = str(captured.value)
    assert "infeasible in outer_002_inner_004" in message
    assert '"n_clusters":8' in message
    assert '"usable_treatment_local_contrasts":1' in message
    assert '"usable_residualized_interaction_local_contrasts":1' in message


def test_builder_does_not_initialize_output_when_preflight_prepare_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_dir = tmp_path / "stage1_bundle"
    builder = ProductionStage1BundleBuilder(
        Stage1BundleBuildOptions(
            dataset_path=tmp_path / "unused.parquet",
            config_path=tmp_path / "unused.json",
            embedding_cache_dir=tmp_path / "unused_cache",
            output_dir=output_dir,
            unit_id_column="person_key",
        )
    )
    initialized = False

    def fail_preflight_prepare():
        raise ValueError("embedding clustered architecture is infeasible in outer_001_full")

    def record_initialize(*_args, **_kwargs):
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(builder, "prepare", fail_preflight_prepare)
    monkeypatch.setattr(builder, "_initialize_output", record_initialize)
    with pytest.raises(ValueError, match="infeasible in outer_001_full"):
        builder.build()
    assert initialized is False
    assert not output_dir.exists()


@pytest.mark.parametrize("token_bounded_row_ids", ((), (0,)))
def test_prepare_projects_only_id_text_treatment_and_observed_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token_bounded_row_ids: tuple[int, ...],
):
    config, _model_dir = _valid_config(tmp_path)
    config.architecture.multi_model_forest.bow_fold_parallelism = "3"
    config.architecture.multi_model_forest.htr_fold_parallelism = "2"
    dataset_path = tmp_path / "cohort.parquet"
    dataset_path.write_bytes(b"authenticated parquet container placeholder")
    config_path = tmp_path / "stage1.json"
    config_path.write_text(json.dumps({"applied_inference": asdict(config)}), encoding="utf-8")
    cache_dir = tmp_path / "embedding_cache"
    cache_dir.mkdir()
    rows = []
    for repetition in range(12):
        for treatment, outcome in ((0, 0), (0, 1), (1, 0), (1, 1)):
            rows.append(
                {
                    "person_key": f"person-{len(rows)}",
                    "clinical_text": f"safe baseline text {repetition}",
                    "treatment_indicator": treatment,
                    "outcome_indicator": outcome,
                }
            )
    projected = pd.DataFrame(rows)
    observed_columns = []

    def fake_read_parquet(_path, *, columns):
        observed_columns.extend(columns)
        return projected.loc[:, columns].copy()

    class FakeCache:
        row_count = len(projected)
        metadata = {
            "sentence_model_name": config.architecture.multi_model_forest.embedding_contrast.model_name,
            "chunk_size_words": config.architecture.multi_model_forest.embedding_contrast.chunk_size_words,
            "chunk_overlap_words": config.architecture.multi_model_forest.embedding_contrast.chunk_overlap_words,
            "max_chunks": config.architecture.multi_model_forest.embedding_contrast.max_chunks,
            "normalize_embeddings": config.architecture.multi_model_forest.embedding_contrast.normalize_embeddings,
            "chunk_selection": "last",
            "max_seq_length": config.architecture.multi_model_forest.embedding_contrast.max_seq_length,
        }

        def __init__(self, path):
            assert Path(path) == cache_dir

        def bind_spent(self, row_ids, texts):
            assert tuple(row_ids) == tuple(range(len(projected)))
            assert len(texts) == len(projected)
            return SimpleNamespace(token_bounded_row_ids=token_bounded_row_ids)

        def identity(self):
            return {"cache_sha256": "d" * 64, "row_count": self.row_count}

    monkeypatch.setattr("oci.inference.production_stage1_bundle.pd.read_parquet", fake_read_parquet)
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle.SpentOnlyFrozenChunkEmbeddingCache",
        FakeCache,
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle.validate_published_production_embedding_cache",
        lambda **_kwargs: {
            "schema_version": "production_arbitrary_cohort_embedding_cache_result_v2",
            "provider_identity": FakeCache(cache_dir).identity(),
        },
    )
    cluster_audit = {"test_cluster_preflight": "closed"}
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle.build_embedding_cluster_feasibility_audit",
        lambda **_kwargs: cluster_audit,
    )
    builder = ProductionStage1BundleBuilder(
        Stage1BundleBuildOptions(
            dataset_path=dataset_path,
            config_path=config_path,
            embedding_cache_dir=cache_dir,
            output_dir=tmp_path / "output",
            unit_id_column="person_key",
            dry_run=True,
        )
    )
    if token_bounded_row_ids:
        with pytest.raises(ValueError, match="token-bounded text reconciliation"):
            builder.prepare()
        return
    prepared = builder.prepare()
    effective = prepared.request["effective_stage1_config"]
    legacy_effective = effective["architecture"]["multi_model_agentic_forest"]
    integrated_effective = effective["architecture"]["multi_model_forest"]
    assert "bow_fold_parallelism" not in legacy_effective
    assert "htr_fold_parallelism" not in legacy_effective
    assert legacy_effective["fold_parallelism"] == "auto"
    assert integrated_effective["bow_fold_parallelism"] == "3"
    assert integrated_effective["htr_fold_parallelism"] == "2"
    parsed_effective = ExperimentConfig.from_dict(
        {"applied_inference": effective}
    ).applied_inference
    assert asdict(parsed_effective) == effective
    assert (
        prepared.config.architecture.multi_model_agentic_forest
        is not prepared.config.architecture.multi_model_forest
    )
    assert observed_columns == [
        "person_key",
        "clinical_text",
        "treatment_indicator",
        "outcome_indicator",
    ]
    assert list(prepared.modeling_data) == [
        "clinical_text",
        "treatment_indicator",
        "outcome_indicator",
    ]
    assert prepared.request["security"]["oracle_columns_decoded_or_materialized"] is False
    assert prepared.request["security"]["whole_parquet_container_authenticated"] is True
    assert prepared.request["security"]["htr_source_word_truncation_allowed"] is False
    assert prepared.request["security"]["htr_tokenizer_truncation_allowed"] is False
    assert prepared.request["htr_input_nontruncation_audit"]["chunk_cap_nonbinding"] is True
    assert prepared.request["embedding_cluster_feasibility_audit"] == cluster_audit
    assert (
        prepared.request["htr_input_nontruncation_audit"]["tokenizer_truncation_allowed"] is False
    )
    hierarchy_identity = prepared.request["hierarchical_discovery_contract_identity"]
    assert (
        hierarchy_identity["semantic_versions"]["interfaces"]["DISCOVERY_INTERFACE_SCHEMA_VERSION"]
        == "all_evidence_discovery_interfaces_v10"
    )
    architecture_contract = prepared.request["architecture_contract"]
    assert (
        architecture_contract["hierarchical_discovery_contract_identity_sha256"]
        == hierarchy_identity["content_sha256"]
    )
    assert (
        architecture_contract["all_ten_architectures_interpreted_separately_before_integration"]
        is True
    )
    assert architecture_contract["within_architecture_consolidation_and_coverage_required"] is True
    assert (
        architecture_contract["hierarchy_typed_cumulative_spent_all_ten_producer_boundary_required"]
        is True
    )
    assert architecture_contract["hierarchy_cumulative_spent_sealed_rows_are_id_only"] is True
    assert (
        architecture_contract["lossless_exact_id_raw_evidence_pages_and_recursive_folds_required"]
        is True
    )
    assert architecture_contract["raw_all_architecture_prompt_allowed"] is False
    assert architecture_contract["legacy_exact_coverage_array_allowed"] is False
    assert (
        architecture_contract["tfidf_truthful_training_scope_policy_required_for_all_three_paths"]
        is True
    )
    assert (
        architecture_contract["tfidf_nested_label_based_selection_required_for_topic_and_orphan"]
        is True
    )
    assert (
        architecture_contract["semantic_retrieval_deterministic_exhaustive_no_selection_required"]
        is True
    )
    assert architecture_contract["manual_digest_approval_required"] is False
    assert architecture_contract["registered_json_parsed_from_authenticated_byte_snapshot"] is True
    assert architecture_contract["strict_duplicate_json_keys_rejected"] is True
    assert (
        architecture_contract["bundle_paths_descriptor_anchored_without_symlink_following"] is True
    )
    assert architecture_contract["loader_to_handoff_manifest_reopen_allowed"] is False
    assert architecture_contract["preparation_wrapper_schema_versions_pinned"] is True
    assert architecture_contract["expected_digests_from_in_memory_prepared_batch_required"] is True
    assert architecture_contract["exact_one_shot_prepared_batch_capability_required"] is True
    assert architecture_contract["exact_one_shot_execution_authorization_required"] is True
    assert architecture_contract["exact_one_shot_same_process_runner_binding_required"] is True
    assert architecture_contract["production_caller_replay_registrations_allowed"] is False
    assert (
        architecture_contract[
            "runtime_provider_identities_and_scientific_file_hashes_reauthenticated"
        ]
        is True
    )
    assert architecture_contract["exact_coordinator_precommit_and_result_types_required"] is True


def test_prepare_builds_fresh_arbitrary_cohort_cache_before_provider_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config, _htr_model = _valid_config(tmp_path)
    dataset_path = tmp_path / "cohort.parquet"
    dataset_path.write_bytes(b"authenticated parquet placeholder")
    config_path = tmp_path / "stage1.json"
    config_path.write_text(json.dumps({"applied_inference": asdict(config)}), encoding="utf-8")
    embedding_model = tmp_path / "embedding-model"
    embedding_model.mkdir()
    (embedding_model / "weights.safetensors").write_bytes(b"local weights")
    cache_target = tmp_path / "fresh-cache"
    rows = pd.DataFrame(
        {
            "person_key": ["p0", "p1", "p2", "p3"],
            "clinical_text": ["note zero", "note one", "note two", "note three"],
            "treatment_indicator": [0, 0, 1, 1],
            "outcome_indicator": [0, 1, 0, 1],
        }
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle.pd.read_parquet",
        lambda _path, *, columns: rows.loc[:, columns].copy(),
    )
    observed = {}

    def fake_build_cache(**kwargs):
        observed.update(kwargs)
        cache_target.mkdir()
        return SimpleNamespace(
            cache_path=cache_target.resolve(),
            identity=lambda: {"published": True},
        )

    class ProviderReached(RuntimeError):
        pass

    def stop_at_provider(path):
        assert Path(path) == cache_target
        raise ProviderReached("fresh cache reached provider boundary")

    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle.build_production_embedding_cache",
        fake_build_cache,
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle.SpentOnlyFrozenChunkEmbeddingCache",
        stop_at_provider,
    )
    builder = ProductionStage1BundleBuilder(
        Stage1BundleBuildOptions(
            dataset_path=dataset_path,
            config_path=config_path,
            embedding_cache_dir=None,
            embedding_cache_output_dir=cache_target,
            embedding_local_model_path=embedding_model,
            output_dir=tmp_path / "bundle",
            unit_id_column="person_key",
        )
    )
    with pytest.raises(ProviderReached, match="provider boundary"):
        builder.prepare()
    embedding = config.architecture.multi_model_forest.embedding_contrast
    assert observed == {
        "dataset_path": dataset_path,
        "text_column": "clinical_text",
        "local_model_path": embedding_model,
        "sentence_model_name": embedding.model_name,
        "chunk_configuration": {
            "chunk_size_words": embedding.chunk_size_words,
            "chunk_overlap_words": embedding.chunk_overlap_words,
            "max_chunks": embedding.max_chunks,
            "chunk_selection": "last",
            "normalize_embeddings": embedding.normalize_embeddings,
            "max_seq_length": embedding.max_seq_length,
        },
        "target_dir": cache_target,
        "device": "cpu",
        "batch_size": embedding.batch_size,
    }


def test_canonical_registry_is_deterministic_and_exact(tmp_path: Path):
    config, _model_dir = _valid_config(tmp_path)
    config.architecture.multi_model_forest.candidate_consistency_inner_folds = 2
    rows = []
    for repetition in range(12):
        for treatment, outcome in ((0, 0), (0, 1), (1, 0), (1, 1)):
            rows.append(
                {
                    "clinical_text": f"row {repetition} {treatment} {outcome}",
                    "treatment_indicator": treatment,
                    "outcome_indicator": outcome,
                }
            )
    data = pd.DataFrame(rows)
    first = build_canonical_split_registry(data=data, config=config, seed=73)
    second = build_canonical_split_registry(data=data, config=config, seed=73)
    assert first == second
    assert first["exact_inner_contract_registry_content_sha256"]
    assert len(_registry_scopes(first)) == 9
    heldout = [row_id for fold in first["outer_folds"] for row_id in fold["heldout_row_ids"]]
    assert sorted(heldout) == list(range(len(data)))


def test_canonical_registry_rejects_scopes_too_small_for_nested_tfidf(tmp_path: Path):
    config, _model_dir = _valid_config(tmp_path)
    config.architecture.multi_model_forest.candidate_consistency_inner_folds = 2
    rows = []
    for repetition in range(3):
        for treatment, outcome in ((0, 0), (0, 1), (1, 0), (1, 1)):
            rows.append(
                {
                    "clinical_text": f"row {repetition} {treatment} {outcome}",
                    "treatment_indicator": treatment,
                    "outcome_indicator": outcome,
                }
            )
    with pytest.raises(ValueError, match="infeasible for production nested TF-IDF"):
        build_canonical_split_registry(
            data=pd.DataFrame(rows),
            config=config,
            seed=73,
        )


def test_immutable_json_and_component_seals_fail_closed_on_tamper(tmp_path: Path):
    immutable = tmp_path / "immutable.json"
    _write_immutable_json(immutable, {"value": 1})
    _write_immutable_json(immutable, {"value": 1})
    with pytest.raises(RuntimeError, match="refusing to mutate"):
        _write_immutable_json(immutable, {"value": 2})

    component = tmp_path / "component"
    component.mkdir()
    evidence = component / "evidence.json"
    evidence.write_text('{"safe": true}\n', encoding="utf-8")
    _seal_component(component, request_sha256="a" * 64, component="legacy")
    evidence.write_text('{"safe": false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="authenticated component file changed"):
        _seal_component(component, request_sha256="a" * 64, component="legacy")


def test_component_manifest_content_hash_is_verified(tmp_path: Path):
    component = tmp_path / "component"
    component.mkdir()
    (component / "evidence.json").write_text("{}\n", encoding="utf-8")
    _seal_component(component, request_sha256="a" * 64, component="query")
    manifest_path = component / "component_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["content_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest content hash is invalid"):
        _seal_component(component, request_sha256="a" * 64, component="query")


def test_legacy_scope_validator_rejects_claimed_full_outer_reuse(tmp_path: Path):
    registry = _registry()
    registry_sha = _sha256_json(registry)
    prepared = SimpleNamespace(
        registry=registry,
        registry_content_sha256=registry_sha,
        modeling_data=pd.DataFrame(index=range(registry["dataset_row_count"])),
    )
    handoff, rows = _write_legacy_component(tmp_path, registry, registry_sha)
    validated = ProductionStage1BundleBuilder._validate_legacy_scope_lineage(handoff, prepared)
    assert set(validated) == {scope["scope_id"] for scope in _registry_scopes(registry)}

    rows[1]["evidence_reused_from_fold_key"] = 1
    handoff.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reused instead of refit"):
        ProductionStage1BundleBuilder._validate_legacy_scope_lineage(handoff, prepared)


def test_legacy_scope_validator_rejects_tampered_raw_sidecar(tmp_path: Path):
    registry = _registry()
    registry_sha = _sha256_json(registry)
    prepared = SimpleNamespace(
        registry=registry,
        registry_content_sha256=registry_sha,
        modeling_data=pd.DataFrame(index=range(registry["dataset_row_count"])),
    )
    handoff, _rows = _write_legacy_component(tmp_path, registry, registry_sha)
    sidecar = next((tmp_path / "legacy" / "raw_evidence_sidecars").glob("*.json"))
    sidecar.write_bytes(sidecar.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="raw evidence sidecar changed"):
        ProductionStage1BundleBuilder._validate_legacy_scope_lineage(handoff, prepared)


def test_catalog_projection_removes_embedding_excerpts_and_attests_derivation():
    digest = _catalog_ready_legacy_digest(
        importance={},
        embedding_evidence={
            "enabled": True,
            "contrasts": [
                {
                    "name": "treatment_mean_difference",
                    "contrast_family": "marginal",
                    "direction_source": "mean_difference",
                    "role_hint": "confounder",
                    "concept_probe_scores": [{"concept": "baseline status", "score": 0.75}],
                }
            ],
        },
        htr_evidence={},
    )
    contrast = digest["confounders"]["embedding_chunks"][0]
    assert contrast["raw_retrieved_excerpts_retained"] is False
    assert contrast["concept_derivation"].startswith("tfidf_ngrams_contrasting")
    assert not any(key.endswith("_chunks") for key in contrast)


def test_catalog_projection_preserves_every_emitted_row_without_prompt_compaction():
    rows = [
        {"feature": "baseline performance status", "score": index / 100.0} for index in range(75)
    ]
    digest = _catalog_ready_legacy_digest(
        importance={
            "views": [
                {
                    "view_name": "linear_1_2",
                    "view_config": {"bow_model": "linear"},
                    "treatment_positive": rows,
                }
            ]
        },
        embedding_evidence={},
        htr_evidence={},
    )
    [group] = digest["confounders"]["bow_blurbs"]
    assert len(group["rows"]) == len(rows)
    assert "prompt_compaction" not in json.dumps(digest)


def _matched_pair_bundle(*, include_htr: bool = True):
    names = [
        "bow__linear_1_2__matched_pair_uplift_delta_logit",
        "bow__linear_1_2__matched_pair_treated_outcome_prob",
        "htr__matched_pair_uplift_delta_logit",
        "htr__matched_pair_treated_outcome_prob",
    ]
    evidence = {
        "importance": {
            "matched_pair_uplift": {"views": [{"view_name": "pair_uplift__linear_1_2"}]}
        },
        "htr_evidence": ({"pair_uplift": {"metrics": {"pairs": 2}}} if include_htr else {}),
    }
    return SimpleNamespace(
        x_names=names,
        x_train=np.arange(16, dtype=float).reshape(4, 4),
        x_test=np.arange(8, dtype=float).reshape(2, 4),
        handoff_evidence=evidence,
        inner_model_rows=[],
    )


def test_bow_and_htr_matched_pair_subproducers_require_separate_model_proofs():
    proofs = _matched_pair_subproducer_proofs(
        bundle=_matched_pair_bundle(),
        expected_bow_views=("linear_1_2",),
        scope_id="outer_001_inner_001",
        fit_row_ids=(0, 1, 2, 3),
        heldout_row_ids=(4, 5),
    )
    assert set(proofs["subproducers"]) == {"bow", "htr"}
    assert all(row["success"] for row in proofs["subproducers"].values())
    assert (
        proofs["subproducers"]["bow"]["model_artifact_sha256"]
        != proofs["subproducers"]["htr"]["model_artifact_sha256"]
    )

    with pytest.raises(RuntimeError, match="HTR matched-pair"):
        _matched_pair_subproducer_proofs(
            bundle=_matched_pair_bundle(include_htr=False),
            expected_bow_views=("linear_1_2",),
            scope_id="outer_001_inner_001",
            fit_row_ids=(0, 1, 2, 3),
            heldout_row_ids=(4, 5),
        )


def test_input_identity_recheck_detects_post_parse_mutation(tmp_path: Path):
    source = tmp_path / "cohort.parquet"
    source.write_bytes(b"first immutable snapshot")
    digest, stat_identity = _read_stable_sha256(source)
    identities = {
        "dataset": {
            "path": str(source),
            "sha256": digest,
            "stat_identity": list(stat_identity),
        }
    }
    ProductionStage1BundleBuilder._revalidate_input_files(identities)
    source.write_bytes(b"mutated after parse")
    with pytest.raises(RuntimeError, match="dataset changed after"):
        ProductionStage1BundleBuilder._revalidate_input_files(identities)


def test_behavior_identity_covers_exact_contract_tree_lock_and_packages():
    identity = _source_identity()
    assert identity["schema_version"] == STAGE1_BEHAVIOR_IDENTITY_SCHEMA
    paths = {row["relative_path"] for row in identity["source_files"]}
    assert "oci/inference/stage1_exact_inner_evidence.py" in paths
    assert {"pyproject.toml", "uv.lock"} <= paths
    assert identity["source_file_count"] == len(identity["source_files"])
    assert identity["installed_distributions"]
    body = dict(identity)
    declared = body.pop("content_sha256")
    assert _sha256_json(body) == declared


def test_partial_tfidf_component_reuse_is_disabled(tmp_path: Path):
    root = tmp_path / "partial_tfidf"
    root.mkdir()
    (root / "unsealed_checkpoint.json").write_text("{}", encoding="utf-8")
    builder = ProductionStage1BundleBuilder(
        Stage1BundleBuildOptions(
            dataset_path=Path("dataset"),
            config_path=Path("config"),
            embedding_cache_dir=Path("cache"),
            output_dir=Path("output"),
            unit_id_column="unit_id",
        )
    )
    with pytest.raises(RuntimeError, match="partial checkpoint reuse is disabled"):
        builder._run_tfidf_component(root, tmp_path, None)


def test_every_scope_must_have_all_ten_catalog_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registry = _registry()
    registry_sha = _sha256_json(registry)
    prepared = SimpleNamespace(
        registry=registry,
        registry_content_sha256=registry_sha,
        modeling_data=pd.DataFrame(index=range(registry["dataset_row_count"])),
        config=SimpleNamespace(
            text_column="clinical_text",
            treatment_column="treatment_indicator",
            outcome_column="outcome_indicator",
        ),
        htr_model_path=tmp_path / "unused-htr-model",
        htr_model_sha256="h" * 64,
        embedding_cache=SimpleNamespace(),
        options=SimpleNamespace(device="cpu"),
    )
    scopes = _registry_scopes(registry)
    legacy_by_scope = {
        scope["scope_id"]: row for scope, row in zip(scopes, _legacy_rows(registry, registry_sha))
    }
    tfidf_by_outer_fold = {}
    enriched_full_by_outer_fold = {}
    query_root = tmp_path / "query"
    (query_root / "artifacts").mkdir(parents=True)
    query_index_rows = []
    for scope in scopes:
        tfidf_row = {
            "outer_fold": scope["outer_fold"],
            "inner_fold": scope["inner_fold"],
            "fit_row_ids": scope["fit_row_ids"],
            "heldout_row_ids": scope["heldout_row_ids"],
            "fit_row_fingerprint": row_set_fingerprint(scope["fit_row_ids"]),
            "heldout_row_fingerprint": row_set_fingerprint(scope["heldout_row_ids"]),
            "discovery": {
                "sentinel_raw_full": scope["inner_fold"] is None,
                "sentinel_resealed_full": False,
            },
        }
        tfidf_by_outer_fold.setdefault(scope["outer_fold"], []).append(tfidf_row)
        if scope["inner_fold"] is None:
            enriched = copy.deepcopy(tfidf_row)
            enriched["discovery"] = {
                "sentinel_raw_full": False,
                "sentinel_resealed_full": True,
                "exact_inner_recurrence": {"groups": []},
            }
            enriched_full_by_outer_fold[scope["outer_fold"]] = enriched
        payload = {
            "scope_id": scope["scope_id"],
            "outer_fold": scope["outer_fold"],
            "scope": "outer_train" if scope["inner_fold"] is None else "inner_train",
            "fit_row_ids": scope["fit_row_ids"],
            "heldout_row_ids": scope["heldout_row_ids"],
            "fit_row_fingerprint": row_set_fingerprint(scope["fit_row_ids"]),
            "heldout_row_fingerprint": row_set_fingerprint(scope["heldout_row_ids"]),
            "split_registry_content_sha256": registry_sha,
            "query_cache_key": "q" * 64,
            "heldout_labels_supplied": False,
            "native_model_artifact": {
                "relative_path": f"native_models/{scope['scope_id']}",
                "sha256": "n" * 64,
            },
            "heldout_moment_artifact": {
                "relative_path": (f"native_models/{scope['scope_id']}/heldout_moments.npz"),
                "sha256": "m" * 64,
            },
            "query_evidence": [],
        }
        if scope["inner_fold"] is not None:
            payload["inner_fold"] = scope["inner_fold"]
        path = query_root / "artifacts" / f"{scope['scope_id']}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        query_index_rows.append(
            {
                "scope_id": scope["scope_id"],
                "path": path.relative_to(query_root).as_posix(),
                "sha256": _sha256_file(path),
                "heldout_labels_supplied": False,
                "heldout_moment_feature_count": 1,
                "native_model_artifact": {
                    "relative_path": f"native_models/{scope['scope_id']}",
                    "kind": "directory",
                    "file_count": 4,
                    "size": 1,
                    "sha256": "n" * 64,
                },
                "owned_snapshot_metadata": {
                    "relative_path": (
                        f"native_models/{scope['scope_id']}/owned_snapshot/metadata.json"
                    )
                },
                "owned_snapshot_arrays": {
                    "relative_path": (
                        f"native_models/{scope['scope_id']}/owned_snapshot/arrays.npz"
                    )
                },
                "heldout_moment_metadata": {
                    "relative_path": (
                        f"native_models/{scope['scope_id']}/heldout_moments.metadata.json"
                    )
                },
                "heldout_moment_arrays": {
                    "relative_path": (f"native_models/{scope['scope_id']}/heldout_moments.npz")
                },
                "native_family_proof_registration": (
                    None if scope["inner_fold"] is None else {"relative_path": "proof.json"}
                ),
            }
        )
    (query_root / "query_artifact_index.json").write_text(
        json.dumps(
            {
                "schema_version": STAGE1_SCOPE_INDEX_SCHEMA,
                "split_registry_content_sha256": registry_sha,
                "registered_native_families": [NEURAL_QUERY_MOMENTS],
                "native_family_proof_index": {"relative_path": "proof-index.json"},
                "heldout_labels_supplied": False,
                "executable_checkpoint_files_retained": False,
                "scopes": query_index_rows,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ProductionStage1BundleBuilder,
        "_validate_legacy_scope_lineage",
        staticmethod(lambda _path, _prepared: legacy_by_scope),
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle.load_resealed_tfidf_handoff",
        lambda *_args, **_kwargs: SimpleNamespace(
            rows_by_outer_fold={
                outer_fold: tuple(rows) for outer_fold, rows in tfidf_by_outer_fold.items()
            },
            full_rows_by_outer_fold=enriched_full_by_outer_fold,
            split_registry_content_hash=registry_sha,
        ),
    )
    counts = {family: 1 for family in ACTIVE_STAGE1_CONCEPT_FAMILIES}

    def fake_catalog(inputs):
        tfidf_input = next(
            evidence for evidence in inputs if evidence.source_kind == TFIDF_TOPIC_SOURCE
        )
        if tfidf_input.provenance.inner_fold is None:
            assert tfidf_input.payload["discovery"]["sentinel_resealed_full"] is True
            assert tfidf_input.payload["discovery"]["sentinel_raw_full"] is False
        return SimpleNamespace(
            audit={
                "atom_count_by_family": counts,
                "semantic_member_count_by_family": counts,
            },
            catalog_sha256="c" * 64,
        )

    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle.build_role_neutral_evidence_catalog",
        fake_catalog,
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle._validate_neural_query_native_family_proof_index",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle._validate_bow_native_family_proof_index",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle._validate_htr_native_family_proof_index",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle._validate_matched_pair_native_family_proof_index",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle._validate_embedding_native_family_proof_index",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle._validate_component_native_registration",
        lambda component_root, registration: Path(component_root)
        / str(registration["relative_path"]),
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle.validate_owned_discovery_snapshot",
        lambda *_args, **_kwargs: {"content_sha256": "s" * 64},
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_bundle._validate_neural_query_moment_artifact",
        lambda *_args, **_kwargs: {
            "arrays_sha256": "m" * 64,
            "feature_count": 1,
        },
    )
    builder = ProductionStage1BundleBuilder(
        Stage1BundleBuildOptions(
            dataset_path=Path("dataset"),
            config_path=Path("config"),
            embedding_cache_dir=Path("cache"),
            output_dir=Path("output"),
            unit_id_column="unit_id",
        )
    )
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    (legacy_root / "exact_scope_index.json").write_text(
        json.dumps(
            {
                "schema_version": STAGE1_SCOPE_INDEX_SCHEMA,
                "split_registry_content_sha256": registry_sha,
                "registered_native_families": [
                    *PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                    *PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                    *PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                    *PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                ],
                "native_bow_family_proof_index": {"relative_path": "bow-proof-index.json"},
                "native_htr_family_proof_index": {"relative_path": "htr-proof-index.json"},
                "native_matched_pair_family_proof_index": {
                    "relative_path": "matched-pair-proof-index.json"
                },
                "native_embedding_family_proof_index": {
                    "relative_path": "embedding-proof-index.json"
                },
                "scopes": [],
            }
        ),
        encoding="utf-8",
    )
    coverage = builder._validate_all_scope_coverage(
        legacy_root=legacy_root,
        tfidf_root=tmp_path / "tfidf",
        query_root=query_root,
        prepared=prepared,
    )
    assert coverage["all_ten_families_nonzero_in_every_scope"] is True

    counts[ACTIVE_STAGE1_CONCEPT_FAMILIES[-1]] = 0
    with pytest.raises(RuntimeError, match="zero concept evidence"):
        builder._validate_all_scope_coverage(
            legacy_root=tmp_path / "legacy",
            tfidf_root=tmp_path / "tfidf",
            query_root=query_root,
            prepared=prepared,
        )
