from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from oci.inference import all_evidence_fusion_cli as cli
from oci.inference.neural_query_signal_artifact import query_signal_columns


def _args(tmp_path: Path, *extra: str):
    paths = {}
    for name in ("dataset", "legacy", "tfidf", "primary"):
        path = tmp_path / f"{name}.artifact"
        path.write_bytes(b"placeholder")
        paths[name] = path
    values = [
        "--benchmark-name",
        "synthetic-five-by-five",
        "--dataset",
        str(paths["dataset"]),
        "--legacy-handoff",
        str(paths["legacy"]),
        "--resealed-tfidf-handoff",
        str(paths["tfidf"]),
        "--primary-splits",
        str(paths["primary"]),
        "--output-dir",
        str(tmp_path / "output"),
        "--endpoint",
        "http://camus:8010/v1",
        "--model",
        "remote/model",
        "--expected-outer-folds",
        "2",
        *extra,
    ]
    return cli.build_parser().parse_args(values)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:8010/v1",
        "HTTP://LOCALHOST.:8010/v1",
        "http://worker.localhost:8010/v1",
        "http://127.0.0.1:8010/v1",
        "http://127.24.9.1:8010/v1",
        "http://127.1:8010/v1",
        "http://2130706433:8010/v1",
        "http://0x7f000001:8010/v1",
        "http://0177.0.0.1:8010/v1",
        "http://0.0.0.0:8010/v1",
        "http://[::1]:8010/v1",
        "http://[::ffff:127.0.0.1]:8010/v1",
        "http://[::]:8010/v1",
    ],
)
def test_remote_endpoint_boundary_rejects_local_and_listener_hosts(endpoint, monkeypatch):
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "current-worker.example.org")
    monkeypatch.setattr(cli.socket, "getfqdn", lambda: "current-worker.example.org")
    monkeypatch.setattr(cli, "_current_host_interface_addresses", lambda names: frozenset())
    monkeypatch.setattr(cli, "_resolve_host_addresses", lambda hostname: frozenset())
    with pytest.raises(ValueError, match="must be remote"):
        cli.validate_remote_endpoint_pool(endpoint)


def test_remote_endpoint_boundary_accepts_cluster_pool_and_deduplicates(monkeypatch):
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "current-worker.example.org")
    monkeypatch.setattr(cli.socket, "getfqdn", lambda: "current-worker.example.org")
    monkeypatch.setattr(
        cli,
        "_current_host_interface_addresses",
        lambda names: frozenset({"10.20.30.40"}),
    )
    monkeypatch.setattr(
        cli,
        "_resolve_host_addresses",
        lambda hostname: frozenset({"10.99.0.8"}) if hostname == "camus" else frozenset(),
    )
    assert (
        cli.validate_remote_endpoint_pool(
            "http://camus:8010/v1/,http://camus:8020/v1,http://camus:8010/v1/"
        )
        == "http://camus:8010/v1,http://camus:8020/v1"
    )


def test_remote_endpoint_boundary_rejects_current_machine_hostname(monkeypatch):
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "current-worker.example.org")
    monkeypatch.setattr(cli.socket, "getfqdn", lambda: "current-worker.example.org")
    monkeypatch.setattr(cli, "_current_host_interface_addresses", lambda names: frozenset())
    monkeypatch.setattr(cli, "_resolve_host_addresses", lambda hostname: frozenset())
    for endpoint in (
        "http://current-worker:8010/v1",
        "http://current-worker.example.org:8010/v1",
    ):
        with pytest.raises(ValueError, match="current machine"):
            cli.validate_remote_endpoint_pool(endpoint)


def test_remote_endpoint_boundary_rejects_dns_alias_resolving_to_loopback(monkeypatch):
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "current-worker.example.org")
    monkeypatch.setattr(cli.socket, "getfqdn", lambda: "current-worker.example.org")
    monkeypatch.setattr(cli, "_current_host_interface_addresses", lambda names: frozenset())
    monkeypatch.setattr(
        cli,
        "_resolve_host_addresses",
        lambda hostname: (
            frozenset({"127.0.0.1"}) if hostname == "loopback-alias.example.org" else frozenset()
        ),
    )

    with pytest.raises(ValueError, match="must be remote"):
        cli.validate_remote_endpoint_pool("http://loopback-alias.example.org:8010/v1")


def test_remote_endpoint_boundary_rejects_alias_to_current_host_interface(monkeypatch):
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "current-worker.example.org")
    monkeypatch.setattr(cli.socket, "getfqdn", lambda: "current-worker.example.org")
    monkeypatch.setattr(
        cli,
        "_current_host_interface_addresses",
        lambda names: frozenset({"10.20.30.40", "2001:db8::40"}),
    )
    monkeypatch.setattr(
        cli,
        "_resolve_host_addresses",
        lambda hostname: (
            frozenset({"10.20.30.40"})
            if hostname == "current-worker-alias.example.org"
            else frozenset()
        ),
    )

    for endpoint in (
        "http://current-worker-alias.example.org:8010/v1",
        "http://10.20.30.40:8010/v1",
        "http://[2001:db8::40]:8010/v1",
    ):
        with pytest.raises(ValueError, match="current machine"):
            cli.validate_remote_endpoint_pool(endpoint)


def test_remote_endpoint_interface_facts_use_assignments_and_resolved_names(monkeypatch):
    monkeypatch.setattr(
        cli.psutil,
        "net_if_addrs",
        lambda: {
            "eth0": [
                SimpleNamespace(family=cli.socket.AF_INET, address="10.20.30.40"),
                SimpleNamespace(family=cli.socket.AF_INET6, address="2001:db8::40%eth0"),
            ]
        },
    )
    monkeypatch.setattr(
        cli,
        "_resolve_host_addresses",
        lambda hostname: frozenset({"10.20.30.41"}),
    )

    assert cli._current_host_interface_addresses(("current-worker",)) == frozenset(
        {"10.20.30.40", "10.20.30.41", "2001:db8::40"}
    )


def test_remote_endpoint_interface_facts_fall_back_when_enumeration_is_forbidden(
    monkeypatch,
):
    def forbidden():
        raise PermissionError("netlink denied")

    monkeypatch.setattr(cli.psutil, "net_if_addrs", forbidden)
    monkeypatch.setattr(
        cli,
        "_resolve_host_addresses",
        lambda hostname: frozenset({"10.20.30.41"}),
    )

    assert cli._current_host_interface_addresses(("current-worker",)) == frozenset({"10.20.30.41"})


def test_extraction_configuration_cannot_enter_non_server_mode(tmp_path):
    args = _args(tmp_path)
    with pytest.raises(ValueError, match="vllm_mode='server'"):
        cli.build_applied_inference_config(args, vllm_mode="python_api")
    with pytest.raises(ValueError, match="vllm_mode='server'"):
        cli.build_applied_inference_config(args, vllm_mode="start_server")

    config = cli.build_applied_inference_config(args)
    assert config.explicit_features.vllm_mode == "server"
    assert config.explicit_features.extraction_provider == "openai"
    assert config.explicit_features.vllm_server_url == "http://camus:8010/v1"
    assert config.explicit_features.vllm_enable_thinking is False
    assert config.explicit_features.source_text_temporally_valid_by_design is True


def test_fusion_reasoning_is_on_while_extraction_reasoning_is_off(tmp_path):
    args = _args(tmp_path)

    fusion = cli.build_agent_config(args)
    extraction = cli.build_applied_inference_config(args)

    assert fusion.agent_enable_thinking is True
    assert fusion.agent_max_tokens == 25000
    assert fusion.agent_thinking_token_budget == 4096
    assert extraction.explicit_features.vllm_enable_thinking is False


@pytest.mark.parametrize("proposal_max_tokens", [4096, 2048])
def test_fixed_fusion_thinking_budget_reserves_answer_tokens(
    tmp_path,
    proposal_max_tokens,
):
    args = _args(
        tmp_path,
        "--proposal-max-tokens",
        str(proposal_max_tokens),
    )

    with pytest.raises(ValueError, match="strictly less than agent_max_tokens"):
        cli.build_agent_config(args)


def test_extraction_prompt_cache_identity_binds_thinking_setting(tmp_path, monkeypatch):
    args = _args(tmp_path)
    thinking_disabled = cli.extraction_prompt_cache_identity(args)

    monkeypatch.setattr(cli, "_EXTRACTION_ENABLE_THINKING", True)
    thinking_enabled = cli.extraction_prompt_cache_identity(args)

    assert thinking_disabled != thinking_enabled


def test_per_fold_orphan_override_is_file_and_sha_bound(tmp_path):
    artifact = tmp_path / "fold_1_effect_scores.parquet"
    artifact.write_bytes(b"immutable orphan scores")
    registration = cli.parse_orphan_ngram_artifact_registrations([f"1={artifact}"])
    assert registration[1].path == artifact.resolve()
    assert len(registration[1].artifact_sha256) == 64

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        cli.parse_orphan_ngram_artifact_registrations([f"1={artifact}::{'0' * 64}"])


def _query_record(row_id: int = 0) -> dict:
    return {
        "query_id": "effect_query_001",
        "bank": "effect",
        "mechanical_role": "effect_modifier",
        "statistical_gate_applied": False,
        "member_count": 3,
        "member_subfolds": [1, 2],
        "fit_standardized_score": 1.5,
        "top_chunks": [
            {
                "evidence_id": f"effect_query_001__row_{row_id:05d}__chunk_000",
                "_oci_row_id": row_id,
                "chunk_index": 0,
                "similarity": 0.8,
                "text": "baseline amber lattice",
            }
        ],
        "top_contrastive_ngrams": [{"term": "amber lattice", "tfidf_contrast": 0.2}],
    }


def test_neural_query_registration_is_sha_and_self_declared_scope_bound(tmp_path):
    artifact = tmp_path / "query_evidence.fold_scoped.json"
    artifact.write_text(
        json.dumps(
            {
                "source_kind": "neural_query_moments",
                "source_family": "neural_query_moments",
                "outer_fold": 1,
                "scope": "outer_train",
                "fit_row_ids": [0, 1],
                "heldout_row_ids": [2, 3],
                "query_evidence": [_query_record(0)],
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    full = {1: {"fit_row_ids": [0, 1], "heldout_row_ids": [2, 3]}}

    registrations = cli.parse_neural_query_moment_artifact_registrations(
        [f"1={artifact}::{digest}"],
        full_outer_rows_by_fold=full,
        require_declared_partition=True,
    )

    assert registrations[1].artifact_sha256 == digest
    assert registrations[1].outer_fold == 1
    assert registrations[1].scope == "outer_train"


def test_required_neural_query_registration_rejects_bare_legacy_artifact(tmp_path):
    artifact = tmp_path / "query_evidence.json"
    artifact.write_text(json.dumps([_query_record(0)]), encoding="utf-8")
    full = {1: {"fit_row_ids": [0, 1], "heldout_row_ids": [2, 3]}}

    with pytest.raises(ValueError, match="hashed fold-scoped bundle"):
        cli.parse_neural_query_moment_artifact_registrations(
            [f"1={artifact}"],
            full_outer_rows_by_fold=full,
            require_declared_partition=True,
        )


def test_precomputed_recursive_review_feature_bank_options_are_removed(tmp_path):
    with pytest.raises(SystemExit):
        _args(tmp_path, "--neural-query-feature-bank-manifest", "1=unsafe.json")
    with pytest.raises(SystemExit):
        _args(tmp_path, "--require-review-feature-banks")


def test_cli_fails_closed_when_adaptive_review_lacks_context_fit_inputs(tmp_path):
    args = _args(
        tmp_path,
        "--post-extraction-review-rounds",
        "1",
        "--max-variables-per-extraction-request",
        "1",
    )
    with pytest.raises(ValueError, match="--review-stage1-config is required"):
        cli.validate_benchmark_inputs(args)


def test_cli_rejects_unbounded_review_quality_retries(tmp_path):
    args = _args(tmp_path, "--post-extraction-review-max-quality-retries", "9")
    with pytest.raises(ValueError, match="max-quality-retries"):
        cli.build_agent_config(args)


def _mock_valid_input_loaders(monkeypatch):
    data = pd.DataFrame(
        {
            "_oci_row_id": range(4),
            "clinical_text": ["a", "b", "c", "d"],
            "treatment_indicator": [0, 1, 0, 1],
            "outcome_indicator": [0.0, 1.0, 1.0, 0.0],
        }
    )
    monkeypatch.setattr(cli, "load_sanitized_dataset", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        cli,
        "load_legacy_full_outer_evidence",
        lambda *args, **kwargs: SimpleNamespace(rows_by_outer_fold={1: {}, 2: {}}),
    )
    monkeypatch.setattr(
        cli,
        "load_resealed_tfidf_handoff",
        lambda *args, **kwargs: SimpleNamespace(
            full_rows_by_outer_fold={
                1: {"fit_row_ids": [2, 3], "heldout_row_ids": [0, 1]},
                2: {"fit_row_ids": [0, 1], "heldout_row_ids": [2, 3]},
            }
        ),
    )
    monkeypatch.setattr(
        cli,
        "load_outer_splits_from_primary_predictions",
        lambda *args, **kwargs: {1: (0, 1), 2: (2, 3)},
    )


def test_validate_benchmark_inputs_requires_fresh_or_empty_output_dir(tmp_path, monkeypatch):
    args = _args(tmp_path, "--dry-run")
    _mock_valid_input_loaders(monkeypatch)
    monkeypatch.setattr(
        cli,
        "validate_remote_endpoint_pool",
        lambda value: "http://camus:8010/v1",
    )
    output_dir = Path(args.output_dir)

    assert cli.validate_benchmark_inputs(args).output_dir == output_dir.resolve()
    assert not output_dir.exists()

    output_dir.mkdir()
    assert cli.validate_benchmark_inputs(args).output_dir == output_dir.resolve()

    (output_dir / "unauthenticated-cache-entry.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="--output-dir must be nonexistent or empty"):
        cli.validate_benchmark_inputs(args)


def _review_dependency_args(
    tmp_path: Path,
    *,
    rows: int = 4,
    modifier_only: bool = True,
) -> list[str]:
    stage1_config = tmp_path / "historical_stage1.json"
    stage1_config.write_text(
        json.dumps({"config": asdict(cli.AppliedInferenceConfig())}),
        encoding="utf-8",
    )
    embedding_cache = tmp_path / "frozen_embeddings"
    embedding_cache.mkdir()
    (embedding_cache / "metadata.json").write_text(
        json.dumps({"num_samples": rows}),
        encoding="utf-8",
    )
    for filename in ("chunk_embeddings.npy", "offsets.npy", "chunk_texts.jsonl"):
        (embedding_cache / filename).write_bytes(b"placeholder")
    values = [
        "--post-extraction-review-rounds",
        "2",
        "--max-variables-per-extraction-request",
        "1",
        "--review-stage1-config",
        str(stage1_config),
        "--review-embedding-cache-dir",
        str(embedding_cache),
        "--review-stage1-device",
        "cuda:0",
        "--review-neural-query-device",
        "cuda:0",
        "--review-neural-query-device",
        "cuda:1",
    ]
    if modifier_only:
        values.append("--modifier-interactions-only")
    return values


def test_adaptive_review_dry_run_rejects_group_dependent_extraction_requests(tmp_path):
    args = _args(
        tmp_path,
        "--dry-run",
        "--post-extraction-review-rounds",
        "1",
        "--max-variables-per-extraction-request",
        "2",
    )
    with pytest.raises(ValueError, match="max-variables-per-extraction-request 1"):
        cli.validate_benchmark_inputs(args)


def _prepare_review_cache_validation(tmp_path, monkeypatch, *extra):
    args = _args(
        tmp_path,
        *_review_dependency_args(tmp_path),
        *extra,
    )
    _mock_valid_input_loaders(monkeypatch)
    monkeypatch.setattr(
        cli,
        "validate_remote_endpoint_pool",
        lambda value: "http://camus:8010/v1",
    )
    return args


def test_review_neural_query_cache_defaults_under_fresh_output(tmp_path, monkeypatch):
    args = _prepare_review_cache_validation(tmp_path, monkeypatch, "--dry-run")

    validated = cli.validate_benchmark_inputs(args)

    assert validated.review_neural_query_cache_dir == (
        Path(args.output_dir).resolve() / "post_extraction_review_neural_query_cache"
    )
    assert not validated.review_neural_query_cache_dir.exists()


def test_review_neural_query_cache_accepts_fresh_direct_child(tmp_path, monkeypatch):
    output = tmp_path / "output"
    cache = output / "fresh-query-cache"
    args = _prepare_review_cache_validation(
        tmp_path,
        monkeypatch,
        "--dry-run",
        "--review-neural-query-cache-dir",
        str(cache),
    )

    assert cli.validate_benchmark_inputs(args).review_neural_query_cache_dir == cache.resolve()

    output.mkdir()
    cache.mkdir()
    assert cli.validate_benchmark_inputs(args).review_neural_query_cache_dir == cache.resolve()


def test_review_neural_query_cache_rejects_prepopulated_checkpoint_before_clients(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    cache = output / "query-cache"
    cache.mkdir(parents=True)
    (cache / "query_discovery.joblib").write_bytes(b"untrusted executable payload")
    args = _prepare_review_cache_validation(
        tmp_path,
        monkeypatch,
        "--review-neural-query-cache-dir",
        str(cache),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("live dependency constructed before cache rejection")

    monkeypatch.setattr(cli, "OpenAICompatibleFeatureSearchAgent", forbidden)
    monkeypatch.setattr(cli, "VLLMExplicitFeatureExtractionProvider", forbidden)
    monkeypatch.setattr(cli, "ContextFitNeuralQueryService", forbidden)

    with pytest.raises(ValueError, match="pre-populated executable checkpoints"):
        cli.run_benchmark(args)


@pytest.mark.parametrize("kind", ["outside", "ancestor", "nested"])
def test_review_neural_query_cache_rejects_output_path_escape(tmp_path, monkeypatch, kind):
    output = tmp_path / "output"
    requested = {
        "outside": tmp_path / "outside-query-cache",
        "ancestor": output,
        "nested": output / "nested" / "query-cache",
    }[kind]
    args = _prepare_review_cache_validation(
        tmp_path,
        monkeypatch,
        "--dry-run",
        "--review-neural-query-cache-dir",
        str(requested),
    )

    with pytest.raises(ValueError, match="direct child of the fresh --output-dir"):
        cli.validate_benchmark_inputs(args)


def test_review_neural_query_cache_rejects_symlink_leaf_and_ancestor(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    leaf = output / "query-cache"
    leaf.symlink_to(outside, target_is_directory=True)
    args = _prepare_review_cache_validation(
        tmp_path,
        monkeypatch,
        "--dry-run",
        "--review-neural-query-cache-dir",
        str(leaf),
    )
    with pytest.raises(ValueError, match="symlink component"):
        cli.validate_benchmark_inputs(args)

    leaf.unlink()
    linked_ancestor = output / "linked-output"
    linked_ancestor.symlink_to(output, target_is_directory=True)
    args.review_neural_query_cache_dir = linked_ancestor / "query-cache"
    with pytest.raises(ValueError, match="symlink component"):
        cli.validate_benchmark_inputs(args)


def test_dry_run_validates_without_constructing_remote_dependencies(tmp_path, monkeypatch):
    args = _args(tmp_path, "--dry-run", "--evaluate-oracle-posthoc")
    _mock_valid_input_loaders(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run constructed a live dependency")

    monkeypatch.setattr(cli, "OpenAICompatibleFeatureSearchAgent", forbidden)
    monkeypatch.setattr(cli, "StagedAllEvidenceFusionAgent", forbidden)
    monkeypatch.setattr(cli, "VLLMExplicitFeatureExtractionProvider", forbidden)
    monkeypatch.setattr(cli, "AllEvidenceFusionRunner", forbidden)
    monkeypatch.setattr(cli, "load_posthoc_oracle_projection", forbidden)

    result = cli.run_benchmark(args)

    assert result["status"] == "validated_dry_run"
    assert result["source_text_temporal_policy"]["temporal_boundary_enforced"] is False
    assert result["clients_constructed"] is False
    assert result["oracle_columns_read"] is False
    assert result["query_moment_fallback_enabled"] is True
    assert result["sparse_query_moment_fallback_enabled"] is True
    assert result["authenticated_neural_query_moment_folds"] == []
    assert result["neural_query_moments_required"] is False
    assert result["tfidf_orphan_adapter_enabled"] is True
    assert result["fusion_enable_thinking"] is True
    assert result["fusion_max_tokens"] == 25000
    assert result["fusion_thinking_token_budget"] == 4096
    assert result["extraction_enable_thinking"] is False
    assert result["extraction_source_text_temporally_valid_by_design"] is True
    assert result["post_extraction_review_rounds"] == 0
    assert result["post_extraction_review_max_quality_retries"] == 2
    assert result["post_extraction_review_agent_is_base_reasoning_agent"] is False
    assert result["post_extraction_review_source_signals_required"] is False
    assert result["post_extraction_review_feature_banks_required"] is False
    assert result["precomputed_recursive_review_feature_banks_enabled"] is False
    assert result["post_extraction_review_spent_discovery_families"] == []
    assert result["read_only_review_spent_cache_source_count"] == 0
    assert result["read_only_review_spent_cache_sources"] == []
    assert result["read_only_context_fit_cache_source_count"] == 0
    assert result["read_only_context_fit_cache_sources"] == []


def test_review_spent_cache_cli_registration_is_repeatable(tmp_path):
    first = f"{tmp_path / 'first.json'}::{'a' * 64}"
    second = f"{tmp_path / 'second.json'}::{'b' * 64}"
    args = _args(
        tmp_path,
        "--read-only-review-spent-evidence-cache",
        first,
        "--read-only-review-spent-evidence-cache",
        second,
    )

    assert args.read_only_review_spent_evidence_cache == [first, second]


def test_context_fit_cache_index_cli_registration_is_repeatable(tmp_path):
    first = f"{tmp_path / 'first-index.json'}::{'a' * 64}"
    second = f"{tmp_path / 'second-index.json'}::{'b' * 64}"
    args = _args(
        tmp_path,
        "--read-only-context-fit-cache-index",
        first,
        "--read-only-context-fit-cache-index",
        second,
    )

    assert args.read_only_context_fit_cache_index == [first, second]


def test_review_query_config_is_closed_and_validated(tmp_path):
    config_path = tmp_path / "query_config.json"
    config_path.write_text(
        json.dumps({"treatment_query_count": 6, "query_epochs": 7}),
        encoding="utf-8",
    )
    args = _args(tmp_path, "--review-neural-query-config", str(config_path))
    config = cli.build_review_neural_query_config(args)
    assert config.treatment_query_count == 6
    assert config.query_epochs == 7

    config_path.write_text(json.dumps({"oracle_query_count": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        cli.build_review_neural_query_config(args)


def test_final_upstream_schema_is_exact_and_precommitted_from_stage1_config(tmp_path):
    args = _args(tmp_path, *_review_dependency_args(tmp_path))
    schema = cli.build_final_upstream_schema_config(args.review_stage1_config)
    applied = cli._minimal_historical_applied_config(Path(args.review_stage1_config).resolve())
    view_names = tuple(view.name for view in applied.architecture.multi_model_forest.bow_views)

    assert schema.namespace == "all_evidence_upstream"
    assert tuple(
        (source.child_name, source.source_kind) for source in schema.calibrated_sources
    ) == tuple(
        (
            f"stage1_calibrated__bow__{name}__effect_weighted_r_tau_pred",
            "nested_calibrated_bow_weighted_r",
        )
        for name in view_names
    ) + (
        (
            "stage1_calibrated__htr__effect_weighted_r_tau_pred",
            "nested_calibrated_htr_weighted_r",
        ),
    )
    assert tuple(family.key for family in schema.raw_families) == (
        cli._FINAL_UPSTREAM_RAW_FAMILY_ROLE_KEYS
    )
    assert len(schema.raw_families) == 19
    assert all(family.required for family in schema.raw_families)
    widths = {family.source_kind: family.signed_order_width for family in schema.raw_families}
    assert widths["neural_query_treatment_moments"] == 5
    assert widths["neural_query_outcome_moments"] == 5
    assert widths["neural_query_effect_moments"] == 5
    assert {
        family.signed_order_width
        for family in schema.raw_families
        if not family.source_kind.startswith("neural_query_")
    } == {16}
    assert len(schema.raw_output_schema()) == 309
    assert schema.reject_unconfigured_calibrated_sources is True
    assert schema.reject_unconfigured_raw_families is True


def test_final_upstream_neural_widths_follow_unequal_query_bank_counts(tmp_path):
    args = _args(tmp_path, *_review_dependency_args(tmp_path))
    query_config = replace(
        cli.NeuralQueryAgenticForestConfig(),
        treatment_query_count=2,
        outcome_query_count=3,
        effect_query_count=4,
        max_raw_feature_candidates=27,
    )
    schema = cli.build_final_upstream_schema_config(
        args.review_stage1_config,
        neural_query_config=query_config,
    )
    widths = {family.source_kind: family.signed_order_width for family in schema.raw_families}

    assert widths["neural_query_treatment_moments"] == 2
    assert widths["neural_query_outcome_moments"] == 3
    assert widths["neural_query_effect_moments"] == 4
    assert tuple(
        name for family in schema.raw_families for name in family.exact_passthrough_feature_names
    ) == query_signal_columns({"treatment": 2, "outcome": 3, "effect": 4})
    assert len(schema.raw_output_schema()) == 303


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--final-upstream-meta-inner-folds", "1", "meta-inner-folds"),
        ("--final-upstream-head-regularization", "0", "head-regularization"),
        ("--final-upstream-head-regularization", "nan", "head-regularization"),
        ("--final-upstream-head-regularization", "inf", "head-regularization"),
        ("--review-stage1-bow-fold-parallelism", "0", "bow-fold-parallelism"),
    ],
)
def test_final_upstream_numeric_configuration_fails_closed(
    tmp_path,
    option,
    value,
    message,
):
    args = _args(tmp_path, option, value)
    with pytest.raises(ValueError, match=message):
        cli.build_agent_config(args)


def test_adaptive_final_upstream_rejects_continuous_outcome(tmp_path):
    args = _args(
        tmp_path,
        *_review_dependency_args(tmp_path),
        "--outcome-type",
        "continuous",
    )
    with pytest.raises(ValueError, match="binary outcome.*matched-pair"):
        cli.build_agent_config(args)


def test_adaptive_final_upstream_requires_explicit_modifier_only_interactions(tmp_path):
    args = _args(tmp_path, *_review_dependency_args(tmp_path, modifier_only=False))
    with pytest.raises(ValueError, match="--modifier-interactions-only"):
        cli.build_agent_config(args)


def test_adaptive_review_dry_run_validates_context_fit_dependencies_only(
    tmp_path,
    monkeypatch,
):
    args = _args(tmp_path, "--dry-run", *_review_dependency_args(tmp_path))
    _mock_valid_input_loaders(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run constructed an adaptive review dependency")

    for name in (
        "ContextFitNeuralQueryService",
        "HistoricalStage1SpentDiscoveryBackend",
        "TfidfTopicOrphanSpentDiscoveryBackend",
        "NeuralQuerySpentDiscoveryBackend",
        "ContextFitReviewSpentEvidenceProvider",
        "HistoricalStage1ContextBackend",
        "TfidfTopicOrphanContextBackend",
        "build_shared_tfidf_context_fit_backends",
        "NeuralQueryContextBackend",
        "CompositeContextFitUpstreamBackend",
        "ContextFitUpstreamGateProvider",
        "CrossFitStableUpstreamBackend",
        "FinalContextFitUpstreamProducer",
    ):
        monkeypatch.setattr(cli, name, forbidden)

    result = cli.run_benchmark(args)

    assert result["status"] == "validated_dry_run"
    assert result["source_text_temporal_policy"]["temporal_boundary_enforced"] is False
    assert result["post_extraction_review_rounds"] == 2
    assert result["max_variables_per_extraction_request"] == 1
    assert result["adaptive_review_contract_local_extraction_verified"] is True
    assert result["post_extraction_review_max_quality_retries"] == 2
    assert result["post_extraction_review_agent_is_base_reasoning_agent"] is True
    assert result["post_extraction_review_source_signals_required"] is True
    assert result["post_extraction_review_feature_banks_required"] is True
    assert result["shared_tfidf_context_fit_service_enabled"] is True
    assert result["shared_tfidf_context_backend_graph"] == cli.SHARED_TFIDF_RUNTIME_GRAPH_ID
    assert result["shared_tfidf_context_backend_graph_selection"] == (
        cli._SHARED_TFIDF_GRAPH_DEFAULT_SELECTION
    )
    assert result["shared_tfidf_disabled_to_preserve_authenticated_cache_identity"] is False
    assert result["shared_tfidf_disabled_to_preserve_authenticated_spent_cache_identity"] is False
    assert result["post_extraction_review_spent_discovery_families"] == sorted(
        cli._REQUIRED_REVIEW_DISCOVERY_FAMILIES
    )
    assert "neural_query_moments" in result["post_extraction_review_spent_discovery_families"]
    assert result["post_extraction_review_gate_provider"] == ("shared_context_fit_all_upstream")
    assert result["review_neural_query_devices"] == ["cuda:0", "cuda:1"]
    assert result["final_upstream_inputs_required"] is True
    assert result["final_upstream_neural_query_inputs_required"] is True
    assert result["final_upstream_producer_constructed"] is False
    assert result["final_causal_forest_required"] is True
    assert result["final_causal_forest_active"] is True
    assert result["final_ite_estimator"] == (cli.FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID)
    forest_backend = dict(result["final_causal_forest_backend"])
    runtime = forest_backend.pop("repository_runtime")
    assert forest_backend == {
        "backend": "repository_causal_forest_prior_working_path_v2",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "honest": True,
        "inference": True,
        "tune_model": True,
        "random_state": 42,
        "exact_nuisance_used_as_fixed_internal_predictions": False,
        "tuning_labels": "outer_train_only",
        "outer_heldout_labels_accepted": False,
    }
    assert runtime["causal_forest_head_module_sha256"]
    assert runtime["econml_distribution_version"] != "not_installed"
    assert result["raw_final_upstream_runtime_retained_separately_from_cache_overlay"] is True
    assert result["final_upstream_meta_inner_folds"] == 3
    assert result["final_upstream_head_regularization"] == 1.0
    assert result["final_upstream_schema_namespace"] == "all_evidence_upstream"
    assert result["final_upstream_signed_order_width"] == 16
    assert result["final_upstream_neural_query_signed_order_widths"] == {
        "treatment": 5,
        "outcome": 5,
        "effect": 5,
    }
    assert result["final_upstream_raw_family_role_count"] == 19
    assert result["final_upstream_raw_column_count"] == 309
    assert result["neural_query_moments_required"] is True
    assert result["neural_query_moment_requirement_flag_set"] is False
    assert result["neural_query_moment_requirement_mode"] == "adaptive_context_fit"
    assert result["modifier_interactions_only_required_for_final_upstream"] is True
    assert result["modifier_interactions_only"] is True
    assert result["clients_constructed"] is False


def test_require_neural_query_moments_fails_when_any_fold_is_unregistered(
    tmp_path,
    monkeypatch,
):
    args = _args(tmp_path, "--dry-run", "--require-neural-query-moments")
    _mock_valid_input_loaders(monkeypatch)

    with pytest.raises(ValueError, match=r"outer folds \[1, 2\]"):
        cli.run_benchmark(args)


def test_adaptive_require_neural_query_uses_context_fit_and_static_artifact_is_audit_only(
    tmp_path,
    monkeypatch,
):
    audit_only_artifact = tmp_path / "legacy_query_evidence.json"
    audit_only_artifact.write_text(json.dumps([_query_record(2)]), encoding="utf-8")
    args = _args(
        tmp_path,
        "--dry-run",
        "--require-neural-query-moments",
        "--neural-query-moment-artifact",
        f"1={audit_only_artifact}",
        *_review_dependency_args(tmp_path),
    )
    _mock_valid_input_loaders(monkeypatch)

    result = cli.run_benchmark(args)

    # This legacy static artifact is optional/audit-only in adaptive mode. It
    # need not declare the fold partition, and the missing second fold cannot
    # disable or replace the required context-fit neural path.
    assert result["authenticated_neural_query_moment_folds"] == [1]
    assert result["neural_query_moment_requirement_mode"] == "adaptive_context_fit"
    assert result["neural_query_moments_required"] is True
    assert result["neural_query_moment_requirement_flag_set"] is True
    assert result["adaptive_context_fit_neural_query_path_required"] is True
    assert result["authenticated_neural_query_artifacts_required"] is False
    assert result["registered_neural_query_artifact_usage"] == (
        "adaptive_audit_only_excluded_from_selector_and_model_inputs"
    )
    assert result["sparse_query_moment_fallback_enabled"] is False
    assert result["sparse_query_moment_fallback_folds"] == []


def test_posthoc_loader_projects_only_oracle_and_returns_exact_join_frame(tmp_path, monkeypatch):
    path = tmp_path / "dataset.parquet"
    path.write_bytes(b"placeholder")
    calls = []

    def projected_read(requested, *, columns):
        calls.append((Path(requested), columns))
        return pd.DataFrame({"true_ite_prob": [0.2, -0.1, 0.4]})

    monkeypatch.setattr(cli.pd, "read_parquet", projected_read)
    frame = cli.load_posthoc_oracle_projection(path)

    assert calls == [(path, ["true_ite_prob"])]
    assert frame.columns.tolist() == ["_oci_row_id", "true_ite_prob"]
    assert frame["_oci_row_id"].tolist() == [0, 1, 2]


def _attested_context_fit_source(kind: str, *, wrapped: bool):
    tfidf_member = (
        {
            "backend": cli.SHARED_TFIDF_CONTEXT_BACKEND_ID,
            "delegate": {"backend": cli.TFIDF_CONTEXT_BACKEND_ID},
            "service": {"service": "in_memory_shared_tfidf_context_fit_service_v1"},
        }
        if wrapped
        else {"backend": cli.TFIDF_CONTEXT_BACKEND_ID}
    )
    backend_identity = {
        "child": {
            "members": [
                {"backend": "test_stage1"},
                tfidf_member,
                {"backend": "test_query"},
            ]
        }
    }
    return SimpleNamespace(
        kind=kind,
        run_attestation=SimpleNamespace(
            final_producer_identity={"backend_identity": backend_identity}
        ),
    )


@pytest.mark.parametrize(
    ("wrapped", "expected"),
    [
        (True, cli.SHARED_TFIDF_RUNTIME_GRAPH_ID),
        (False, cli.UNWRAPPED_TFIDF_RUNTIME_GRAPH_ID),
    ],
)
def test_context_fit_sources_select_their_attested_tfidf_graph(wrapped, expected):
    sources = (
        _attested_context_fit_source("review_gate", wrapped=wrapped),
        _attested_context_fit_source("final_upstream", wrapped=wrapped),
    )
    graph, selection = cli._select_tfidf_context_backend_graph(sources)
    assert graph == expected
    assert selection == cli._SHARED_TFIDF_GRAPH_ATTESTED_SELECTION


def test_context_fit_sources_reject_mixed_tfidf_graphs():
    sources = (
        _attested_context_fit_source("review_gate", wrapped=True),
        _attested_context_fit_source("final_upstream", wrapped=False),
    )
    with pytest.raises(ValueError, match="mix wrapped and unwrapped"):
        cli._select_tfidf_context_backend_graph(sources)


def test_no_context_fit_source_defaults_to_wrapped_even_with_spent_overlay():
    graph, selection = cli._select_tfidf_context_backend_graph(())
    audit = cli._shared_tfidf_runtime_audit(
        review_enabled=True,
        graph=graph,
        selection=selection,
    )
    assert graph == cli.SHARED_TFIDF_RUNTIME_GRAPH_ID
    assert selection == cli._SHARED_TFIDF_GRAPH_DEFAULT_SELECTION
    assert audit["shared_tfidf_context_fit_service_enabled"] is True
    assert audit["authenticated_spent_cache_influences_tfidf_graph_selection"] is False


@pytest.mark.parametrize("with_spent_cache_overlay", [False, True])
@pytest.mark.parametrize("with_context_fit_cache_overlay", [False, True])
@pytest.mark.parametrize("context_fit_wrapped", [False, True])
def test_adaptive_review_wires_shared_query_service_and_gate_provider(
    tmp_path,
    monkeypatch,
    with_spent_cache_overlay,
    with_context_fit_cache_overlay,
    context_fit_wrapped,
):
    review_args = _review_dependency_args(tmp_path)
    args = _args(
        tmp_path,
        *review_args,
        "--review-stage1-bow-fold-parallelism",
        "3",
    )
    authenticated_spent_source = object()
    authenticated_gate_source = _attested_context_fit_source(
        "review_gate",
        wrapped=context_fit_wrapped,
    )
    authenticated_final_source = _attested_context_fit_source(
        "final_upstream",
        wrapped=context_fit_wrapped,
    )
    validated = cli.ValidatedBenchmarkInputs(
        dataset_path=Path(args.dataset),
        legacy_handoff_path=Path(args.legacy_handoff),
        tfidf_handoff_path=Path(args.resealed_tfidf_handoff),
        primary_splits_path=Path(args.primary_splits),
        output_dir=Path(args.output_dir),
        cache_index_paths=(),
        orphan_ngram_artifacts_by_fold={},
        row_count=4,
        outer_folds=(1, 2),
        review_stage1_config_path=Path(args.review_stage1_config).resolve(),
        review_embedding_cache_dir=Path(args.review_embedding_cache_dir).resolve(),
        review_neural_query_cache_dir=tmp_path / "query_context_cache",
        authenticated_review_spent_cache_sources=(
            (authenticated_spent_source,) if with_spent_cache_overlay else ()
        ),
        authenticated_context_fit_cache_sources=(
            (authenticated_gate_source, authenticated_final_source)
            if with_context_fit_cache_overlay
            else ()
        ),
    )
    monkeypatch.setattr(cli, "validate_benchmark_inputs", lambda ignored: validated)
    calls = {}

    class BaseAgent:
        def __init__(self, config):
            assert config.agent_enable_thinking is True
            self.config = config

    class StagedAgent:
        def __init__(self, base, *, final_max_candidates):
            self.base = base
            self.final_max_candidates = final_max_candidates

    class Extractor:
        def __init__(self, config, output_dir):
            assert config.explicit_features.vllm_enable_thinking is False
            assert config.explicit_features.source_text_temporally_valid_by_design is True

    query_service = object()
    stage1_spent = object()
    tfidf_spent = object()
    shared_tfidf_spent = object()
    query_spent = object()
    stage1_gate = object()
    tfidf_gate = object()
    shared_tfidf_gate = object()
    shared_tfidf_service = object()
    query_gate = object()
    spent_provider = object()
    spent_cache_overlay = object()
    composite_backend = object()
    shared_gate_provider = object()
    shared_gate_cache_overlay = object()
    shared_stable_backend = object()
    final_upstream_producer = object()
    shared_final_cache_overlay = object()
    shared_embedding_cache = SimpleNamespace(row_count=validated.row_count)
    shared_htr_snapshot = object()

    def embedding_cache_ctor(path):
        calls.setdefault("embedding_cache_paths", []).append(Path(path))
        return shared_embedding_cache

    def query_service_ctor(**kwargs):
        calls["query_service"] = kwargs
        return query_service

    def stage1_spent_ctor(**kwargs):
        calls["stage1_spent"] = kwargs
        return stage1_spent

    def tfidf_spent_ctor(**kwargs):
        calls["tfidf_spent"] = kwargs
        return tfidf_spent

    def query_spent_ctor(service):
        assert service is query_service
        return query_spent

    def spent_provider_ctor(**kwargs):
        calls["spent_provider"] = kwargs
        return spent_provider

    def spent_cache_overlay_ctor(*, provider, sources, output_root):
        assert provider is spent_provider
        assert sources == (authenticated_spent_source,)
        assert Path(output_root) == validated.output_dir
        calls["spent_cache_overlay"] = True
        return spent_cache_overlay

    def stage1_gate_ctor(**kwargs):
        calls["stage1_gate"] = kwargs
        return stage1_gate

    def tfidf_gate_ctor(**kwargs):
        calls["tfidf_gate"] = kwargs
        return tfidf_gate

    def shared_tfidf_ctor(*, spent_discovery_backend, context_backend):
        assert spent_discovery_backend is tfidf_spent
        assert context_backend is tfidf_gate
        calls["shared_tfidf"] = (spent_discovery_backend, context_backend)
        return SimpleNamespace(
            service=shared_tfidf_service,
            spent_discovery_backend=shared_tfidf_spent,
            context_backend=shared_tfidf_gate,
        )

    def query_gate_ctor(service):
        assert service is query_service
        return query_gate

    def composite_ctor(backends):
        calls["composite"] = tuple(backends)
        return composite_backend

    def gate_provider_ctor(cache_dir, *, backend):
        assert backend is shared_stable_backend
        calls["gate_provider_backend"] = backend
        calls["gate_provider_cache"] = Path(cache_dir)
        return shared_gate_provider

    def stable_backend_ctor(backend, *, config):
        assert backend is composite_backend
        calls["stable_backend_child"] = backend
        calls["stable_schema"] = config
        return shared_stable_backend

    def final_producer_ctor(cache_dir, *, backend):
        assert backend is shared_stable_backend
        calls["final_producer_backend"] = backend
        calls["final_producer_cache"] = Path(cache_dir)
        return final_upstream_producer

    def gate_cache_overlay_ctor(*, provider, runtime_producer, sources, output_root):
        assert provider is shared_gate_provider
        assert runtime_producer is final_upstream_producer
        assert sources == (authenticated_gate_source, authenticated_final_source)
        assert Path(output_root) == validated.output_dir
        calls["gate_cache_overlay"] = True
        return shared_gate_cache_overlay

    def final_cache_overlay_ctor(*, producer, sources, output_root):
        assert producer is final_upstream_producer
        assert sources == (authenticated_gate_source, authenticated_final_source)
        assert Path(output_root) == validated.output_dir
        calls["final_cache_overlay"] = True
        return shared_final_cache_overlay

    prediction_path = tmp_path / "adaptive_predictions.parquet"

    class Runner:
        def __init__(self, **kwargs):
            calls["runner"] = kwargs
            assert kwargs["review_agent"] is kwargs["fusion_agent"].base
            assert kwargs["review_spent_evidence_provider"] is (
                spent_cache_overlay if with_spent_cache_overlay else spent_provider
            )
            assert kwargs["review_partition_provider"] is None
            expected_gate = (
                shared_gate_cache_overlay
                if with_context_fit_cache_overlay
                else shared_gate_provider
            )
            expected_final = (
                shared_final_cache_overlay
                if with_context_fit_cache_overlay
                else final_upstream_producer
            )
            assert kwargs["review_gate_source_provider"] is expected_gate
            assert kwargs["review_gate_feature_bank_provider"] is expected_gate
            assert kwargs["final_upstream_producer"] is expected_final
            assert kwargs["raw_final_upstream_producer"] is final_upstream_producer
            assert kwargs["config"].post_extraction_review_rounds == 2
            assert kwargs["config"].post_extraction_review_max_quality_retries == 2
            assert kwargs["config"].require_review_source_signals is True
            assert kwargs["config"].require_review_feature_banks is True
            assert kwargs["config"].require_final_upstream_inputs is True
            assert kwargs["config"].require_final_upstream_neural_query_inputs is True
            assert kwargs["config"].require_final_causal_forest is True
            assert kwargs["config"].final_upstream_meta_inner_folds == 3
            assert kwargs["config"].final_upstream_head_regularization == 1.0

        def run(self):
            return SimpleNamespace(
                prediction_path=prediction_path,
                prediction_sha256="d" * 64,
                run_manifest_path=tmp_path / "adaptive_run.json",
            )

    monkeypatch.setattr(cli, "OpenAICompatibleFeatureSearchAgent", BaseAgent)
    monkeypatch.setattr(cli, "StagedAllEvidenceFusionAgent", StagedAgent)
    monkeypatch.setattr(cli, "VLLMExplicitFeatureExtractionProvider", Extractor)
    monkeypatch.setattr(cli, "SpentOnlyFrozenChunkEmbeddingCache", embedding_cache_ctor)
    monkeypatch.setattr(cli, "_resolve_htr_model_path", lambda _config: tmp_path / "htr")
    monkeypatch.setattr(
        cli,
        "PrivateHTRModelTreeSnapshot",
        lambda _path: shared_htr_snapshot,
    )
    monkeypatch.setattr(cli, "ContextFitNeuralQueryService", query_service_ctor)
    monkeypatch.setattr(cli, "HistoricalStage1SpentDiscoveryBackend", stage1_spent_ctor)
    monkeypatch.setattr(cli, "TfidfTopicOrphanSpentDiscoveryBackend", tfidf_spent_ctor)
    monkeypatch.setattr(cli, "NeuralQuerySpentDiscoveryBackend", query_spent_ctor)
    monkeypatch.setattr(cli, "ContextFitReviewSpentEvidenceProvider", spent_provider_ctor)
    monkeypatch.setattr(
        cli,
        "AuthenticatedReviewSpentEvidenceCacheOverlay",
        spent_cache_overlay_ctor,
    )
    monkeypatch.setattr(cli, "HistoricalStage1ContextBackend", stage1_gate_ctor)
    monkeypatch.setattr(cli, "TfidfTopicOrphanContextBackend", tfidf_gate_ctor)
    monkeypatch.setattr(
        cli,
        "build_shared_tfidf_context_fit_backends",
        shared_tfidf_ctor,
    )
    monkeypatch.setattr(cli, "NeuralQueryContextBackend", query_gate_ctor)
    monkeypatch.setattr(cli, "CompositeContextFitUpstreamBackend", composite_ctor)
    monkeypatch.setattr(cli, "ContextFitUpstreamGateProvider", gate_provider_ctor)
    monkeypatch.setattr(cli, "CrossFitStableUpstreamBackend", stable_backend_ctor)
    monkeypatch.setattr(cli, "FinalContextFitUpstreamProducer", final_producer_ctor)
    monkeypatch.setattr(
        cli,
        "AuthenticatedContextFitGateCacheOverlay",
        gate_cache_overlay_ctor,
    )
    monkeypatch.setattr(
        cli,
        "AuthenticatedFinalContextFitCacheOverlay",
        final_cache_overlay_ctor,
    )
    monkeypatch.setattr(cli, "AllEvidenceFusionRunner", Runner)

    result = cli.run_benchmark(args)

    assert result["status"] == "completed"
    assert result["final_ite_estimator"] == (cli.FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID)
    assert result["final_causal_forest_active"] is True
    assert result["raw_final_upstream_runtime_retained_separately_from_cache_overlay"] is True
    assert result["source_text_temporal_policy"]["temporal_boundary_enforced"] is False
    assert calls["query_service"]["devices"] == ("cuda:0", "cuda:1")
    assert calls["query_service"]["stage1_config_path"] == (validated.review_stage1_config_path)
    assert calls["embedding_cache_paths"] == [validated.review_embedding_cache_dir]
    assert calls["query_service"]["embedding_cache"] is shared_embedding_cache
    assert calls["stage1_spent"]["embedding_cache"] is shared_embedding_cache
    assert calls["stage1_gate"]["embedding_cache"] is shared_embedding_cache
    assert calls["stage1_spent"]["htr_model_snapshot"] is shared_htr_snapshot
    assert calls["stage1_spent"]["bow_fold_parallelism"] == 3
    assert calls["stage1_spent"]["bow_parallel_backend"] == "threads"
    assert calls.get("spent_cache_overlay", False) is with_spent_cache_overlay
    assert calls["stage1_gate"]["htr_model_snapshot"] is shared_htr_snapshot
    assert calls["stage1_gate"]["bow_fold_parallelism"] == 3
    assert calls["stage1_gate"]["bow_parallel_backend"] == "threads"
    assert (
        calls["query_service"]["stage1_config_snapshot"]
        is calls["stage1_spent"]["stage1_config_snapshot"]
        is calls["tfidf_spent"]["stage1_config_snapshot"]
        is calls["stage1_gate"]["stage1_config_snapshot"]
        is calls["tfidf_gate"]["stage1_config_snapshot"]
    )
    assert (
        calls["stable_schema"].source_config_sha256
        == calls["query_service"]["stage1_config_snapshot"].sha256
    )
    preserve_authenticated_identity = bool(
        with_context_fit_cache_overlay and not context_fit_wrapped
    )
    expected_tfidf_spent = tfidf_spent if preserve_authenticated_identity else shared_tfidf_spent
    expected_tfidf_gate = tfidf_gate if preserve_authenticated_identity else shared_tfidf_gate
    assert calls["spent_provider"]["backends"] == (
        stage1_spent,
        expected_tfidf_spent,
        query_spent,
    )
    assert set(calls["spent_provider"]["required_source_families"]) == set(
        cli._REQUIRED_REVIEW_DISCOVERY_FAMILIES
    )
    assert "neural_query_moments" in calls["spent_provider"]["required_source_families"]
    if preserve_authenticated_identity:
        assert "shared_tfidf" not in calls
    else:
        assert calls["shared_tfidf"] == (tfidf_spent, tfidf_gate)
    assert calls["composite"] == (stage1_gate, expected_tfidf_gate, query_gate)
    assert calls["stable_backend_child"] is composite_backend
    assert calls["gate_provider_backend"] is calls["final_producer_backend"]
    assert calls.get("gate_cache_overlay", False) is with_context_fit_cache_overlay
    assert calls.get("final_cache_overlay", False) is with_context_fit_cache_overlay
    assert calls["stable_schema"].namespace == "all_evidence_upstream"
    assert len(calls["stable_schema"].raw_families) == 19
    assert len(calls["stable_schema"].raw_output_schema()) == 309
    assert result["shared_tfidf_context_fit_service_enabled"] is (
        not preserve_authenticated_identity
    )
    assert result["shared_tfidf_disabled_to_preserve_authenticated_cache_identity"] is (
        with_context_fit_cache_overlay and not context_fit_wrapped
    )
    assert result["shared_tfidf_disabled_to_preserve_authenticated_spent_cache_identity"] is False
    assert result["authenticated_spent_cache_influences_tfidf_graph_selection"] is False
    assert result["shared_tfidf_context_backend_graph"] == (
        cli.UNWRAPPED_TFIDF_RUNTIME_GRAPH_ID
        if preserve_authenticated_identity
        else cli.SHARED_TFIDF_RUNTIME_GRAPH_ID
    )
    assert result["shared_tfidf_context_backend_graph_selection"] == (
        cli._SHARED_TFIDF_GRAPH_ATTESTED_SELECTION
        if with_context_fit_cache_overlay
        else cli._SHARED_TFIDF_GRAPH_DEFAULT_SELECTION
    )
    assert {
        family.source_kind: family.signed_order_width
        for family in calls["stable_schema"].raw_families
        if family.source_kind.startswith("neural_query_")
    } == {
        "neural_query_treatment_moments": 5,
        "neural_query_outcome_moments": 5,
        "neural_query_effect_moments": 5,
    }
    assert calls["stage1_spent"]["device"] == "cuda:0"
    assert calls["stage1_gate"]["device"] == "cuda:0"


def test_adaptive_review_rejects_authenticated_cache_row_count_change(
    tmp_path,
    monkeypatch,
):
    review_args = _review_dependency_args(tmp_path)
    args = _args(tmp_path, *review_args)
    validated = cli.ValidatedBenchmarkInputs(
        dataset_path=Path(args.dataset),
        legacy_handoff_path=Path(args.legacy_handoff),
        tfidf_handoff_path=Path(args.resealed_tfidf_handoff),
        primary_splits_path=Path(args.primary_splits),
        output_dir=Path(args.output_dir),
        cache_index_paths=(),
        orphan_ngram_artifacts_by_fold={},
        row_count=4,
        outer_folds=(1, 2),
        review_stage1_config_path=Path(args.review_stage1_config).resolve(),
        review_embedding_cache_dir=Path(args.review_embedding_cache_dir).resolve(),
        review_neural_query_cache_dir=tmp_path / "query_context_cache",
    )

    class BaseAgent:
        def __init__(self, config):
            self.config = config

    class StagedAgent:
        def __init__(self, base, *, final_max_candidates):
            self.base = base

    class Extractor:
        def __init__(self, config, output_dir):
            pass

    def forbidden(**_kwargs):
        raise AssertionError("review consumers must not start after row-count mismatch")

    monkeypatch.setattr(cli, "validate_benchmark_inputs", lambda _args: validated)
    monkeypatch.setattr(cli, "OpenAICompatibleFeatureSearchAgent", BaseAgent)
    monkeypatch.setattr(cli, "StagedAllEvidenceFusionAgent", StagedAgent)
    monkeypatch.setattr(cli, "VLLMExplicitFeatureExtractionProvider", Extractor)
    monkeypatch.setattr(
        cli,
        "SpentOnlyFrozenChunkEmbeddingCache",
        lambda _path: SimpleNamespace(row_count=3),
    )
    monkeypatch.setattr(cli, "_resolve_htr_model_path", lambda _config: tmp_path / "htr")
    monkeypatch.setattr(cli, "PrivateHTRModelTreeSnapshot", lambda _path: object())
    monkeypatch.setattr(cli, "ContextFitNeuralQueryService", forbidden)

    with pytest.raises(RuntimeError, match="row count changed after validation"):
        cli.run_benchmark(args)


def test_oracle_projection_occurs_only_after_runner_freezes_predictions(tmp_path, monkeypatch):
    args = _args(tmp_path, "--evaluate-oracle-posthoc")
    orphan_overrides = {
        1: cli.TfidfOrphanNgramArtifact(
            path=tmp_path / "fold_1_effect_scores.parquet",
            artifact_sha256="c" * 64,
        )
    }
    validated = cli.ValidatedBenchmarkInputs(
        dataset_path=Path(args.dataset),
        legacy_handoff_path=Path(args.legacy_handoff),
        tfidf_handoff_path=Path(args.resealed_tfidf_handoff),
        primary_splits_path=Path(args.primary_splits),
        output_dir=Path(args.output_dir),
        cache_index_paths=(),
        orphan_ngram_artifacts_by_fold=orphan_overrides,
        row_count=4,
        outer_folds=(1, 2),
    )
    monkeypatch.setattr(cli, "validate_benchmark_inputs", lambda ignored: validated)
    events = []

    class BaseAgent:
        def __init__(self, config):
            self.config = config

    class StagedAgent:
        def __init__(self, base, *, final_max_candidates):
            self.base = base

    class Extractor:
        def __init__(self, config, output_dir):
            assert config.explicit_features.vllm_mode == "server"

    prediction_path = tmp_path / "frozen_predictions.parquet"
    prediction_path.write_bytes(b"immutable frozen prediction test payload")
    prediction_sha256 = hashlib.sha256(prediction_path.read_bytes()).hexdigest()

    class Runner:
        def __init__(self, **kwargs):
            assert kwargs["review_agent"] is kwargs["fusion_agent"].base
            assert kwargs["final_upstream_producer"] is None
            assert kwargs["raw_final_upstream_producer"] is None
            assert kwargs["review_gate_source_provider"] is None
            assert kwargs["review_partition_provider"] is None
            assert kwargs["review_gate_feature_bank_provider"] is None
            assert kwargs["legacy_primary_predictions_path"] == validated.primary_splits_path
            assert kwargs["tfidf_orphan_artifacts_by_fold"] is orphan_overrides
            assert kwargs["config"].include_tfidf_orphan_ngrams is True
            assert kwargs["config"].derive_sparse_query_moments_when_missing is True
            assert kwargs["config"].fusion_model_identity == "remote/model"
            assert kwargs["config"].extraction_model_identity == "remote/model"
            assert kwargs["config"].remote_endpoint_pool_identity == ("http://camus:8010/v1")
            assert kwargs["config"].fusion_enable_thinking is True
            assert kwargs["config"].fusion_max_tokens == 25000
            assert kwargs["config"].fusion_thinking_token_budget == 4096
            assert kwargs["config"].extraction_enable_thinking is False
            assert kwargs["config"].post_extraction_review_rounds == 0
            assert kwargs["config"].post_extraction_review_max_operations == 4
            assert kwargs["config"].post_extraction_review_max_quality_retries == 2
            assert kwargs["config"].require_review_source_signals is False
            assert kwargs["config"].require_review_feature_banks is False
            assert kwargs["config"].require_final_upstream_inputs is False
            assert kwargs["config"].require_final_upstream_neural_query_inputs is False
            assert kwargs["config"].require_final_causal_forest is False
            assert kwargs["config"].final_upstream_meta_inner_folds == 3
            assert kwargs["config"].final_upstream_head_regularization == 1.0

        def run(self):
            events.append("runner_froze_predictions")
            return SimpleNamespace(
                prediction_path=prediction_path,
                prediction_sha256=prediction_sha256,
                run_manifest_path=tmp_path / "run.json",
            )

    def oracle_loader(path):
        events.append("oracle_projection")
        return pd.DataFrame({"_oci_row_id": range(4), "true_ite_prob": [0.1, 0.2, 0.3, 0.4]})

    def evaluator(**kwargs):
        events.append("posthoc_evaluation")
        assert kwargs["prediction_path"] == prediction_path
        assert kwargs["expected_prediction_sha256"] == prediction_sha256
        return {"overall": {"pearson_correlation": 0.7}}

    monkeypatch.setattr(cli, "OpenAICompatibleFeatureSearchAgent", BaseAgent)
    monkeypatch.setattr(cli, "StagedAllEvidenceFusionAgent", StagedAgent)
    monkeypatch.setattr(cli, "VLLMExplicitFeatureExtractionProvider", Extractor)
    monkeypatch.setattr(cli, "AllEvidenceFusionRunner", Runner)
    monkeypatch.setattr(cli, "load_posthoc_oracle_projection", oracle_loader)
    monkeypatch.setattr(cli, "evaluate_frozen_all_evidence_predictions", evaluator)

    result = cli.run_benchmark(args)

    assert events == [
        "runner_froze_predictions",
        "oracle_projection",
        "posthoc_evaluation",
    ]
    assert result["posthoc_oracle_evaluation_performed"] is True


def test_prediction_hash_mismatch_blocks_oracle_projection(tmp_path, monkeypatch):
    args = _args(tmp_path, "--evaluate-oracle-posthoc")
    validated = cli.ValidatedBenchmarkInputs(
        dataset_path=Path(args.dataset),
        legacy_handoff_path=Path(args.legacy_handoff),
        tfidf_handoff_path=Path(args.resealed_tfidf_handoff),
        primary_splits_path=Path(args.primary_splits),
        output_dir=Path(args.output_dir),
        cache_index_paths=(),
        orphan_ngram_artifacts_by_fold={},
        row_count=4,
        outer_folds=(1, 2),
    )
    prediction_path = tmp_path / "mutated_predictions.parquet"
    prediction_path.write_bytes(b"bytes that do not match the declared runner hash")
    monkeypatch.setattr(cli, "validate_benchmark_inputs", lambda ignored: validated)
    monkeypatch.setattr(
        cli,
        "validate_remote_endpoint_pool",
        lambda value: "http://camus:8010/v1",
    )
    monkeypatch.setattr(cli, "OpenAICompatibleFeatureSearchAgent", lambda config: object())
    monkeypatch.setattr(
        cli,
        "StagedAllEvidenceFusionAgent",
        lambda base, *, final_max_candidates: object(),
    )
    monkeypatch.setattr(
        cli,
        "VLLMExplicitFeatureExtractionProvider",
        lambda config, output_dir: object(),
    )
    monkeypatch.setattr(
        cli,
        "AllEvidenceFusionRunner",
        lambda **kwargs: SimpleNamespace(
            run=lambda: SimpleNamespace(
                prediction_path=prediction_path,
                prediction_sha256="0" * 64,
                run_manifest_path=tmp_path / "run.json",
            )
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("oracle loader/evaluator ran after a prediction hash mismatch")

    monkeypatch.setattr(cli, "load_posthoc_oracle_projection", forbidden)
    monkeypatch.setattr(cli, "evaluate_frozen_all_evidence_predictions", forbidden)

    with pytest.raises(ValueError, match="frozen prediction SHA-256 does not match"):
        cli.run_benchmark(args)


def test_no_posthoc_flag_never_reads_oracle(tmp_path, monkeypatch):
    args = _args(tmp_path)
    validated = cli.ValidatedBenchmarkInputs(
        dataset_path=Path(args.dataset),
        legacy_handoff_path=Path(args.legacy_handoff),
        tfidf_handoff_path=Path(args.resealed_tfidf_handoff),
        primary_splits_path=Path(args.primary_splits),
        output_dir=Path(args.output_dir),
        cache_index_paths=(),
        orphan_ngram_artifacts_by_fold={},
        row_count=4,
        outer_folds=(1, 2),
    )
    monkeypatch.setattr(cli, "validate_benchmark_inputs", lambda ignored: validated)
    monkeypatch.setattr(cli, "OpenAICompatibleFeatureSearchAgent", lambda config: object())
    monkeypatch.setattr(
        cli,
        "StagedAllEvidenceFusionAgent",
        lambda base, final_max_candidates: object(),
    )
    monkeypatch.setattr(
        cli,
        "VLLMExplicitFeatureExtractionProvider",
        lambda config, output: object(),
    )
    monkeypatch.setattr(
        cli,
        "AllEvidenceFusionRunner",
        lambda **kwargs: SimpleNamespace(
            run=lambda: SimpleNamespace(
                prediction_path=tmp_path / "predictions.parquet",
                prediction_sha256="b" * 64,
                run_manifest_path=tmp_path / "run.json",
            )
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("oracle was read without explicit post-hoc request")

    monkeypatch.setattr(cli, "load_posthoc_oracle_projection", forbidden)
    result = cli.run_benchmark(args)
    assert result["posthoc_oracle_evaluation_performed"] is False
