"""Versioned schemas for native hierarchical token/chunk attention evidence."""

ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA = (
    "production_role_neutral_htr_native_attention_evidence_v2"
)
ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_PACKAGE_SCHEMA = (
    "production_role_neutral_htr_token_attention_package_v1"
)
ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_BATCH_SCHEMA = (
    "production_role_neutral_htr_token_attention_batch_v1"
)
ROLE_NEUTRAL_HTR_CHUNK_EVIDENCE_SCHEMA = (
    "production_role_neutral_htr_chunk_attention_atom_v2"
)
ROLE_NEUTRAL_HTR_READABLE_SPAN_SCHEMA = (
    "production_role_neutral_htr_readable_token_span_v1"
)

__all__ = [
    "ROLE_NEUTRAL_HTR_CHUNK_EVIDENCE_SCHEMA",
    "ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA",
    "ROLE_NEUTRAL_HTR_READABLE_SPAN_SCHEMA",
    "ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_BATCH_SCHEMA",
    "ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_PACKAGE_SCHEMA",
]
