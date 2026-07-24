from __future__ import annotations

from pathlib import Path

import pytest

from oci.config import (
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureSpec,
    ModelArchitectureConfig,
)
from oci.extraction import (
    CONTRACT_LEXICAL_CONTEXT_VERSION,
    EXTRACTION_GROUPING_VERSION,
    build_extraction_prompt,
    compact_contract_lexical_context,
)
from oci.extraction.cache import _compute_config_hash
from oci.inference.agentic_explicit_feature_forest import (
    EXTRACTION_PROMPT_VERSION,
    VLLMExplicitFeatureExtractionProvider,
    _spec_extraction_contract_dict,
)
from oci.inference import all_evidence_fusion_cli as fusion_cli


def _abstract_specs() -> list[ExplicitFeatureSpec]:
    return [
        ExplicitFeatureSpec(
            name="aurora_dispatch_interval",
            type="continuous",
            roles=["confounder"],
            description="scheduled aurora dispatch interval in cycles",
        ),
        ExplicitFeatureSpec(
            name="surface_finish",
            type="categorical",
            categories=["cobalt sheen", "umber matte"],
            roles=["effect_modifier"],
            description="declared surface finish of the assembled object",
            value_aliases={"cobalt sheen": ["blue gloss"]},
        ),
    ]


def test_contract_lexical_context_is_deterministic_verbatim_and_budgeted():
    early = "EARLY_SIGNAL The scheduled aurora dispatch interval was seven cycles.\n"
    filler = "Inventory ledger entry: ordinary crates were counted and shelved.\n" * 120
    late = "LATE_SIGNAL The assembled object's surface finish was cobalt sheen.\n"
    source = early + filler + late

    first = compact_contract_lexical_context(source, _abstract_specs(), max_chars=3000)
    second = compact_contract_lexical_context(source, _abstract_specs(), max_chars=3000)

    assert first == second
    assert first.version == CONTRACT_LEXICAL_CONTEXT_VERSION
    assert len(first.text) <= 3000
    assert first.fallback_tail_used is False
    assert "EARLY_SIGNAL" in first.text
    assert "LATE_SIGNAL" in first.text
    assert f"[{CONTRACT_LEXICAL_CONTEXT_VERSION}]" in first.text
    assert "[Retrieved excerpt 1 | source chars " in first.text
    assert tuple(excerpt.start for excerpt in first.selected_excerpts) == tuple(
        sorted(excerpt.start for excerpt in first.selected_excerpts)
    )
    for excerpt in first.selected_excerpts:
        assert source[excerpt.start : excerpt.end] in first.text

    # Causal roles are required by ExplicitFeatureSpec, but are deliberately
    # excluded from the lexical query. Only extraction-contract text is used.
    assert "confounder" not in first.query_tokens
    assert "effect" not in first.query_tokens
    assert "modifier" not in first.query_tokens


def test_contract_lexical_prompt_labels_retrieved_text():
    source = (
        "EARLY_SIGNAL The aurora dispatch interval was four cycles.\n"
        + "Neutral packing manifest.\n" * 100
        + "LATE_SIGNAL The surface finish was umber matte."
    )

    prompt = build_extraction_prompt(
        source,
        _abstract_specs(),
        max_text_length=2600,
        context_strategy="contract_lexical_rag",
    )

    assert "Contract-guided retrieved excerpts:" in prompt
    assert f"[{CONTRACT_LEXICAL_CONTEXT_VERSION}]" in prompt
    assert "EARLY_SIGNAL" in prompt
    assert "LATE_SIGNAL" in prompt
    assert "Neutral packing manifest.\n" * 100 not in prompt


def test_tail_context_does_not_treat_a_literal_rag_marker_as_retrieval_metadata():
    prompt = build_extraction_prompt(
        f"A normal source line contains [{CONTRACT_LEXICAL_CONTEXT_VERSION}] literally.",
        _abstract_specs(),
        max_text_length=400000,
        context_strategy="tail",
    )

    assert "Read this complete clinical note" in prompt
    assert "Clinical Note:" in prompt
    assert "Contract-guided retrieved excerpts:" not in prompt


def test_stopword_spanning_contract_phrase_wins_lexical_ranking():
    spec = ExplicitFeatureSpec(
        name="ribbon_quartz",
        type="continuous",
        roles=["confounder"],
        description=("clinical_domain=abstract; parent_object=assembly: ribbon of quartz width"),
    )
    early = "EARLY_SEPARATE ribbon appeared far away from quartz.\n"
    filler = "Neutral crate manifest without matching material terms.\n" * 70
    late = "LATE_EXACT The ribbon of quartz width was eleven units.\n"

    context = compact_contract_lexical_context(
        early + filler + late,
        [spec],
        max_chars=1500,
    )

    assert "LATE_EXACT" in context.text
    late_score = max(
        excerpt.score
        for excerpt in context.selected_excerpts
        if "LATE_EXACT" in (early + filler + late)[excerpt.start : excerpt.end]
    )
    early_score = max(
        excerpt.score
        for excerpt in context.selected_excerpts
        if "EARLY_SEPARATE" in (early + filler + late)[excerpt.start : excerpt.end]
    )
    assert late_score > early_score
    assert "ribbon quartz" in context.query_phrases
    assert {"clinical", "domain", "parent", "object"}.isdisjoint(context.query_tokens)


def _provider(
    tmp_path: Path,
    *,
    grouping: str,
    context: str = "tail",
    cap: int = 3,
    max_chars: int = 400000,
) -> VLLMExplicitFeatureExtractionProvider:
    config = AppliedInferenceConfig(
        dataset_path="in_memory",
        text_column="text",
        architecture=ModelArchitectureConfig(),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            vllm_mode="server",
            vllm_server_url="http://remote-worker:8010/v1",
            vllm_model_name="fixed-model",
            cache_dir=str(tmp_path),
            max_variables_per_extraction_request=cap,
            extraction_grouping_strategy=grouping,
            extraction_context_strategy=context,
            extraction_max_text_length=max_chars,
        ),
    )
    provider = VLLMExplicitFeatureExtractionProvider(config, tmp_path)
    provider._active_text_hash = "fixed-text-hash"
    return provider


def test_packed_grouping_is_stable_and_respects_configured_cap(tmp_path):
    specs = [
        ExplicitFeatureSpec(
            name=f"abstract_field_{index}",
            type="continuous",
            roles=["confounder"],
            description=f"abstract family {index % 2}: quantity {index}",
        )
        for index in range(8)
    ]
    provider = _provider(tmp_path, grouping="packed", cap=3)

    groups = provider._extraction_spec_groups(specs)

    assert [[spec.name for spec in group] for group in groups] == [
        ["abstract_field_0", "abstract_field_1", "abstract_field_2"],
        ["abstract_field_3", "abstract_field_4", "abstract_field_5"],
        ["abstract_field_6", "abstract_field_7"],
    ]
    assert max(map(len, groups)) == 3


def test_adaptive_review_contract_local_capability_requires_single_target_groups(tmp_path):
    grouped = _provider(tmp_path / "grouped", grouping="packed", cap=3)
    contract_local = _provider(tmp_path / "single", grouping="packed", cap=1)

    assert grouped.extraction_request_group_dependent is True
    assert grouped.adaptive_review_contract_local_extraction() is False
    assert contract_local.adaptive_review_contract_local_extraction() is True


def test_extraction_cache_identity_separates_context_grouping_budget_and_versions(
    tmp_path,
):
    spec = _abstract_specs()[0]
    historical = _provider(tmp_path / "historical", grouping="clinical_domain")
    packed_rag = _provider(
        tmp_path / "packed_rag",
        grouping="packed",
        context="contract_lexical_rag",
        cap=2,
        max_chars=2400,
    )
    historical_config = historical._cache_config([spec])
    packed_rag_config = packed_rag._cache_config([spec])

    assert EXTRACTION_PROMPT_VERSION == "explicit_features_v5"
    assert packed_rag_config["prompt_template_version"] == EXTRACTION_PROMPT_VERSION
    assert packed_rag_config["extraction_grouping_version"] == EXTRACTION_GROUPING_VERSION
    assert (
        packed_rag_config["extraction_context_compactor_version"]
        == CONTRACT_LEXICAL_CONTEXT_VERSION
    )
    assert _compute_config_hash(historical_config) != _compute_config_hash(packed_rag_config)

    base = {
        "prompt_template_version": EXTRACTION_PROMPT_VERSION,
        "extraction_grouping_strategy": "packed",
        "extraction_grouping_version": "grouping_v1",
        "max_variables_per_extraction_request": 2,
        "extraction_context_strategy": "contract_lexical_rag",
        "extraction_context_compactor_version": "context_v1",
        "extraction_max_text_length": 2400,
    }
    variants = [
        base,
        {**base, "extraction_grouping_strategy": "clinical_domain"},
        {**base, "extraction_grouping_version": "grouping_v2"},
        {**base, "max_variables_per_extraction_request": 3},
        {**base, "extraction_context_strategy": "tail"},
        {**base, "extraction_context_compactor_version": "context_v2"},
        {**base, "extraction_max_text_length": 2600},
    ]
    assert len({_compute_config_hash(config) for config in variants}) == len(variants)


def test_per_spec_cache_identity_separates_companion_contracts_and_order(tmp_path):
    target, first_companion = _abstract_specs()
    second_companion = ExplicitFeatureSpec(
        name="crate_span",
        type="continuous",
        roles=["confounder"],
        description="declared span of the packed crate",
    )
    provider = _provider(
        tmp_path,
        grouping="packed",
        context="contract_lexical_rag",
        cap=3,
        max_chars=2400,
    )

    def target_hash(group):
        provider._active_request_group_contracts_by_spec[target.name] = [
            _spec_extraction_contract_dict(spec) for spec in group
        ]
        return _compute_config_hash(provider._cache_config([target]))

    identities = {
        target_hash([target, first_companion]),
        target_hash([target, second_companion]),
        target_hash([first_companion, target]),
    }

    assert len(identities) == 3


def _cli_args(tmp_path: Path, *extra: str):
    return fusion_cli.build_parser().parse_args(
        [
            "--benchmark-name",
            "abstract-benchmark",
            "--dataset",
            str(tmp_path / "dataset.parquet"),
            "--legacy-handoff",
            str(tmp_path / "legacy.json"),
            "--resealed-tfidf-handoff",
            str(tmp_path / "tfidf.json"),
            "--primary-splits",
            str(tmp_path / "primary.parquet"),
            "--output-dir",
            str(tmp_path / "output"),
            "--endpoint",
            "http://remote-worker:8010/v1",
            "--model",
            "remote/model",
            *extra,
        ]
    )


def test_fusion_cli_wires_packed_contract_rag_and_composite_cache_identity(tmp_path):
    default_args = _cli_args(tmp_path)
    invalid_grouped_review_args = _cli_args(
        tmp_path,
        "--extraction-grouping-strategy",
        "packed",
        "--max-variables-per-extraction-request",
        "4",
        "--extraction-context-strategy",
        "contract_lexical_rag",
        "--extraction-max-text-length",
        "12000",
    )
    with pytest.raises(
        ValueError,
        match="max-variables-per-extraction-request 1",
    ):
        fusion_cli.build_applied_inference_config(invalid_grouped_review_args)

    # Adaptive review is mandatory in the current production contract and
    # changed-only extraction must remain contract-local. "packed" therefore
    # has a cap of one here; the context strategy and grouping identity are
    # still independently wired and cache-bound.
    rag_args = _cli_args(
        tmp_path,
        "--extraction-grouping-strategy",
        "packed",
        "--max-variables-per-extraction-request",
        "1",
        "--extraction-context-strategy",
        "contract_lexical_rag",
        "--extraction-max-text-length",
        "12000",
        "--review-stage1-config",
        str(tmp_path / "review-stage1.json"),
        "--review-embedding-cache-dir",
        str(tmp_path / "review-embedding-cache"),
    )

    config = fusion_cli.build_applied_inference_config(rag_args)

    assert config.explicit_features.extraction_grouping_strategy == "packed"
    assert config.explicit_features.max_variables_per_extraction_request == 1
    assert config.explicit_features.extraction_context_strategy == "contract_lexical_rag"
    assert config.explicit_features.extraction_max_text_length == 12000
    assert fusion_cli.extraction_prompt_cache_identity(default_args).startswith(
        f"{EXTRACTION_PROMPT_VERSION}+extraction_semantics:"
    )
    assert fusion_cli.extraction_prompt_cache_identity(rag_args).startswith(
        f"{EXTRACTION_PROMPT_VERSION}+extraction_semantics:"
    )
    assert fusion_cli.extraction_prompt_cache_identity(
        default_args
    ) != fusion_cli.extraction_prompt_cache_identity(rag_args)
