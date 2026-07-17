# oci/extraction/__init__.py
"""Extraction module for CDT.

This module provides LLM-based extraction of explicit features from clinical text.
"""

from .explicit_features import (
    ExplicitFeatureValue,
    VLLMFeatureExtractor,
    build_extraction_prompt,
    build_extraction_repair_prompt,
    infer_vllm_reasoning_parser,
    parse_extraction_response,
    resolve_vllm_reasoning_parser,
    strip_reasoning_trace,
    extract_explicit_features,
)
from .cache import ExtractionCache
from .contract_lexical_context import (
    CONTRACT_LEXICAL_CONTEXT_VERSION,
    EXTRACTION_GROUPING_VERSION,
    ContractLexicalContext,
    RetrievedContractExcerpt,
    compact_contract_lexical_context,
)
from .numeric_inventory import (
    AgenticNumericInventoryExtractor,
    CompletionResult,
    NumericInventoryConfig,
    NumericInventoryLLMClient,
    NumericTextChunk,
    build_chunk_extraction_prompt,
    build_ontology_harmonization_prompt,
    build_patient_reconciliation_prompt,
    chunk_dataset_documents,
    parse_numeric_values_response,
    parse_ontology_mapping_response,
    split_text_into_all_word_chunks,
)
from .llm_routing import (
    OpenAIClientPool,
    call_with_exponential_backoff,
    parse_server_urls,
)

# Backward-compatible import aliases. Old config keys are still rejected.
ExplicitConfounderValue = ExplicitFeatureValue
VLLMConfounderExtractor = VLLMFeatureExtractor
extract_explicit_confounders = extract_explicit_features

__all__ = [
    "ExplicitFeatureValue",
    "VLLMFeatureExtractor",
    "build_extraction_prompt",
    "build_extraction_repair_prompt",
    "infer_vllm_reasoning_parser",
    "parse_extraction_response",
    "resolve_vllm_reasoning_parser",
    "strip_reasoning_trace",
    "extract_explicit_features",
    "ExtractionCache",
    "CONTRACT_LEXICAL_CONTEXT_VERSION",
    "EXTRACTION_GROUPING_VERSION",
    "ContractLexicalContext",
    "RetrievedContractExcerpt",
    "compact_contract_lexical_context",
    "AgenticNumericInventoryExtractor",
    "CompletionResult",
    "NumericInventoryConfig",
    "NumericInventoryLLMClient",
    "NumericTextChunk",
    "build_chunk_extraction_prompt",
    "build_ontology_harmonization_prompt",
    "build_patient_reconciliation_prompt",
    "chunk_dataset_documents",
    "parse_numeric_values_response",
    "parse_ontology_mapping_response",
    "split_text_into_all_word_chunks",
    "OpenAIClientPool",
    "call_with_exponential_backoff",
    "parse_server_urls",
    "ExplicitConfounderValue",
    "VLLMConfounderExtractor",
    "extract_explicit_confounders",
]
