from __future__ import annotations

import ast
import copy
import json
from dataclasses import MISSING, fields, replace
from pathlib import Path

import numpy as np
import pytest

from oci.extraction.complete_paged import (
    CompletePagingGeometry,
    plan_complete_note_pages,
)
from oci.extraction.explicit_features import build_extraction_prompt
from oci.inference.neural_query_agentic_forest import (
    NeuralQueryAgenticForestConfig,
    NeuralQueryEvidenceCapacityOverflowError,
    _contrastive_ngrams,
    build_query_rag_documents,
)
from oci.inference.portable_workflow_spec import (
    LosslessTextWindowSpec,
    ScientificWorkflowSpec,
    SentenceEmbeddingEncoderSpec,
)
from oci.inference.production_role_neutral_producer_factories import (
    missing_role_neutral_architecture_profile_fields,
)
from oci.inference.review_spent_evidence_provider import (
    SemanticWitnessScientificConfig,
    SemanticWitnessTfidfVectorizerConfig,
)
from oci.inference.role_neutral_embedding_group_execution import (
    RoleNeutralEmbeddingScientificConfig,
)
from oci.inference.role_neutral_htr_group_execution import RoleNeutralHTRConfig
from oci.inference.role_neutral_matched_pair_group_execution import (
    RoleNeutralMatchedPairConfig,
)
from oci.models.concept_embedding_utils import chunk_text_words
from oci.models.hierarchical_transformer_extractor import (
    split_text_into_word_chunks,
)
from oci.models.lossless_tokenization import SemanticTruncationError

_REPOSITORY = Path(__file__).resolve().parents[1]
_BENCHMARK_SPEC = _REPOSITORY / "example_configs" / "portable_all_evidence_scientific_nsclc.json"


def _benchmark_mapping() -> dict[str, object]:
    value = json.loads(_BENCHMARK_SPEC.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    "field_name",
    (
        "complete_page_core_chars",
        "complete_page_context_chars",
        "complete_page_max_chars",
        "reconciliation_fan_in",
        "embedding_chunk_size_words",
        "embedding_chunk_overlap_words",
        "embedding_max_chunks",
        "embedding_chunk_selection",
        "embedding_max_seq_length",
        "embedding_normalize",
        "embedding_encoder",
    ),
)
def test_typed_scientific_spec_rejects_every_missing_text_capacity(
    field_name: str,
) -> None:
    payload = _benchmark_mapping()
    text_windows = payload["text_windows"]
    assert isinstance(text_windows, dict)
    text_windows.pop(field_name)

    with pytest.raises(ValueError, match=r"text_windows.*missing"):
        ScientificWorkflowSpec.from_mapping(payload)


@pytest.mark.parametrize(
    "field_name",
    tuple(field.name for field in fields(SentenceEmbeddingEncoderSpec)),
)
def test_typed_scientific_spec_rejects_every_missing_encoder_output_control(
    field_name: str,
) -> None:
    payload = _benchmark_mapping()
    encoder = payload["text_windows"]["embedding_encoder"]
    assert isinstance(encoder, dict)
    encoder.pop(field_name)

    with pytest.raises(ValueError, match=r"embedding_encoder.*missing"):
        ScientificWorkflowSpec.from_mapping(payload)


def test_typed_scientific_spec_rejects_extra_encoder_output_control() -> None:
    payload = _benchmark_mapping()
    encoder = payload["text_windows"]["embedding_encoder"]
    assert isinstance(encoder, dict)
    encoder["hidden_truncation"] = 64

    with pytest.raises(ValueError, match=r"embedding_encoder.*extra"):
        ScientificWorkflowSpec.from_mapping(payload)


def test_typed_capacity_dataclasses_have_no_scientific_fallbacks() -> None:
    for specification in (
        LosslessTextWindowSpec,
        SentenceEmbeddingEncoderSpec,
        CompletePagingGeometry,
        RoleNeutralHTRConfig,
        RoleNeutralMatchedPairConfig,
        RoleNeutralEmbeddingScientificConfig,
        SemanticWitnessScientificConfig,
        SemanticWitnessTfidfVectorizerConfig,
    ):
        defaulted = [
            field.name
            for field in fields(specification)
            if field.default is not MISSING or field.default_factory is not MISSING
        ]
        assert defaulted == []


def test_lexical_semantic_profile_audit_is_closed_at_every_leaf() -> None:
    profiles = _benchmark_mapping()["architecture_profiles"]
    assert isinstance(profiles, dict)
    configuration = profiles["lexical_semantic_retrieval"]["producer_configuration"]
    assert isinstance(configuration, dict)
    base = "architecture_profiles.lexical_semantic_retrieval." "producer_configuration"

    for key in configuration:
        changed = copy.deepcopy(profiles)
        changed["lexical_semantic_retrieval"]["producer_configuration"].pop(key)
        assert f"{base}.{key}" in (missing_role_neutral_architecture_profile_fields(changed))

    for vectorizer_name in ("retrieval_vectorizer", "htr_vectorizer"):
        vectorizer = configuration[vectorizer_name]
        assert isinstance(vectorizer, dict)
        for key in vectorizer:
            changed = copy.deepcopy(profiles)
            changed["lexical_semantic_retrieval"]["producer_configuration"][vectorizer_name].pop(
                key
            )
            assert f"{base}.{vectorizer_name}.{key}" in (
                missing_role_neutral_architecture_profile_fields(changed)
            )

    extra_top = copy.deepcopy(profiles)
    extra_top["lexical_semantic_retrieval"]["producer_configuration"]["silent_term_limit"] = 12
    assert f"{base}.silent_term_limit" in (
        missing_role_neutral_architecture_profile_fields(extra_top)
    )

    extra_nested = copy.deepcopy(profiles)
    extra_nested["lexical_semantic_retrieval"]["producer_configuration"]["retrieval_vectorizer"][
        "silent_text_limit"
    ] = 4096
    assert f"{base}.retrieval_vectorizer.silent_text_limit" in (
        missing_role_neutral_architecture_profile_fields(extra_nested)
    )


@pytest.mark.parametrize(
    ("profile_name", "nested_path"),
    (
        ("hierarchical_transformer", ("chunk_size_words",)),
        ("hierarchical_transformer", ("max_chunks",)),
        ("hierarchical_transformer", ("max_chunk_length",)),
        ("matched_patient_uplift", ("htr_extractor", "chunk_size_words")),
        ("matched_patient_uplift", ("htr_extractor", "max_chunks")),
        ("matched_patient_uplift", ("htr_extractor", "max_chunk_length")),
        ("whole_cohort_embeddings", ("maximum_source_chunks_per_row",)),
        ("whole_cohort_embeddings", ("maximum_retrieval_chunks_per_side",)),
        ("whole_cohort_embeddings", ("maximum_semantic_terms",)),
        (
            "learned_neural_queries",
            ("query_config", "evidence_chunks_per_patient_per_query"),
        ),
        ("learned_neural_queries", ("query_config", "evidence_excerpt_chars")),
        ("learned_neural_queries", ("query_config", "evidence_top_ngrams")),
        ("learned_neural_queries", ("query_config", "rag_max_chunks_per_patient")),
        ("learned_neural_queries", ("query_config", "rag_excerpt_chars")),
    ),
)
def test_six_producer_factory_audit_exposes_missing_capacity_leaf(
    profile_name: str,
    nested_path: tuple[str, ...],
) -> None:
    profiles = copy.deepcopy(_benchmark_mapping()["architecture_profiles"])
    assert isinstance(profiles, dict)
    configuration = profiles[profile_name]["producer_configuration"]
    cursor = configuration
    for name in nested_path[:-1]:
        cursor = cursor[name]
    cursor.pop(nested_path[-1])

    missing = missing_role_neutral_architecture_profile_fields(profiles)
    expected_suffix = ".".join(
        (
            f"architecture_profiles.{profile_name}.producer_configuration",
            *nested_path,
        )
    )
    assert expected_suffix in missing


def test_binding_word_chunk_capacities_abort_instead_of_selecting_text() -> None:
    text = "zero one two three four five six seven"

    with pytest.raises(SemanticTruncationError, match="semantic truncation is forbidden"):
        chunk_text_words(
            text,
            chunk_size_words=3,
            chunk_overlap_words=1,
            max_chunks=2,
            chunk_selection="last",
        )
    with pytest.raises(SemanticTruncationError, match="semantic truncation is forbidden"):
        split_text_into_word_chunks(
            text,
            chunk_size_words=3,
            chunk_overlap_words=1,
            max_chunks=2,
        )


def test_neural_query_term_capacity_aborts_instead_of_top_k_selection() -> None:
    query_profile = _benchmark_mapping()["architecture_profiles"]["learned_neural_queries"][
        "producer_configuration"
    ]["query_config"]
    config = replace(
        NeuralQueryAgenticForestConfig(**query_profile),
        evidence_top_ngrams=1,
    )

    with pytest.raises(
        NeuralQueryEvidenceCapacityOverflowError,
        match="no terms were silently discarded",
    ):
        _contrastive_ngrams(
            ["alpha beta gamma delta epsilon"],
            ["zeta theta"],
            limit=config.evidence_top_ngrams,
            config=config,
        )


@pytest.mark.parametrize(
    ("configuration_update", "match"),
    (
        ({"rag_max_chunks_per_patient": 1}, "no retrieved chunks were silently discarded"),
        ({"rag_excerpt_chars": 4}, "text truncation is forbidden"),
    ),
)
def test_query_rag_capacities_abort_instead_of_omitting_text(
    configuration_update: dict[str, int],
    match: str,
) -> None:
    query_profile = _benchmark_mapping()["architecture_profiles"]["learned_neural_queries"][
        "producer_configuration"
    ]["query_config"]
    config = replace(
        NeuralQueryAgenticForestConfig(**query_profile),
        rag_chunks_per_query=1,
        **configuration_update,
    )

    with pytest.raises(NeuralQueryEvidenceCapacityOverflowError, match=match):
        build_query_rag_documents(
            row_ids=[0],
            chunk_matrices=[
                np.asarray(
                    [[1.0, 0.0], [0.0, 1.0]],
                    dtype=np.float32,
                )
            ],
            all_chunk_texts=[["alpha text", "beta text"]],
            queries=np.asarray(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            ),
            query_ids=["treatment_query_001", "effect_query_001"],
            query_banks=["treatment", "effect"],
            config=config,
            device="cpu",
        )


def test_complete_paging_covers_arbitrarily_long_note_and_unpaged_overflow_fails() -> None:
    geometry = CompletePagingGeometry(
        core_chars=17,
        context_chars=4,
        max_page_chars=25,
    )
    text = "".join(str(index % 10) for index in range(10_003))
    pages = plan_complete_note_pages(text, geometry=geometry)

    assert "".join(text[page.core_start : page.core_end] for page in pages) == text
    assert all(
        len(text[page.context_start : page.context_end]) <= geometry.max_page_chars
        for page in pages
    )
    with pytest.raises(ValueError, match="oversized unpaged input"):
        build_extraction_prompt(
            text,
            [],
            max_text_length=geometry.max_page_chars,
            context_strategy="complete_paged_v1",
            source_text_temporally_valid_by_design=True,
        )


def _semantic_256_geometry_defaults(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: list[str] = []
    semantic_names = {
        "complete_page_context_chars",
        "complete-page-context-chars",
        "context_chars",
    }

    def is_256(node: ast.AST | None) -> bool:
        return isinstance(node, ast.Constant) and node.value == 256

    def target_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in semantic_names and is_256(node.value):
            findings.append(f"line {node.lineno}: keyword {node.arg}=256")
        elif isinstance(node, ast.AnnAssign):
            name = target_name(node.target)
            if name in semantic_names and is_256(node.value):
                findings.append(f"line {node.lineno}: field {name}=256")
        elif isinstance(node, ast.Assign) and is_256(node.value):
            if any(target_name(target) in semantic_names for target in node.targets):
                findings.append(f"line {node.lineno}: assignment=256")
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value in semantic_names and is_256(value):
                    findings.append(f"line {node.lineno}: mapping {key.value}=256")
        elif isinstance(node, ast.Call):
            option_names = {
                argument.value
                for argument in node.args
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
            }
            if "--complete-page-context-chars" in option_names:
                if any(
                    keyword.arg == "default" and is_256(keyword.value) for keyword in node.keywords
                ):
                    findings.append(f"line {node.lineno}: CLI complete-page context default=256")
    return tuple(findings)


def test_nsclc_page_geometry_is_only_benchmark_configuration_not_production_code() -> None:
    benchmark_windows = _benchmark_mapping()["text_windows"]
    assert (
        benchmark_windows["complete_page_core_chars"],
        benchmark_windows["complete_page_context_chars"],
        benchmark_windows["complete_page_max_chars"],
    ) == (13_488, 256, 14_000)

    sources = tuple((_REPOSITORY / "oci").rglob("*.py")) + tuple(
        (_REPOSITORY / "scripts").rglob("*.py")
    )
    exact_benchmark_literals = ("13488", "13_488", "14000", "14_000")
    exact_offenders: list[str] = []
    context_offenders: list[str] = []
    for path in sources:
        source = path.read_text(encoding="utf-8")
        if any(value in source for value in exact_benchmark_literals):
            exact_offenders.append(path.relative_to(_REPOSITORY).as_posix())
        for finding in _semantic_256_geometry_defaults(path):
            context_offenders.append(f"{path.relative_to(_REPOSITORY).as_posix()}:{finding}")

    assert exact_offenders == []
    assert context_offenders == []
