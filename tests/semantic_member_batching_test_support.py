"""Explicit semantic-member batching identities for authenticated test catalogs."""

from __future__ import annotations

from typing import Any

from oci.inference.lossless_stage1_evidence_catalog import (
    SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION,
)


def semantic_member_batching_identity(
    *,
    semantic_member_batch_size: int,
) -> dict[str, Any]:
    """Return the closed batching identity with no implicit test default."""

    if (
        isinstance(semantic_member_batch_size, bool)
        or not isinstance(semantic_member_batch_size, int)
        or semantic_member_batch_size < 1
    ):
        raise ValueError("semantic_member_batch_size must be a positive integer")
    return {
        "schema_version": SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION,
        "semantic_member_batch_size": semantic_member_batch_size,
        "selection_or_truncation_authorized": False,
        "complete_member_coverage_required": True,
    }


def semantic_member_batching_audit(
    *,
    semantic_member_batch_size: int,
) -> dict[str, Any]:
    """Return the matching catalog audit projection."""

    identity = semantic_member_batching_identity(
        semantic_member_batch_size=semantic_member_batch_size,
    )
    return {
        "semantic_member_batching": identity,
        "semantic_member_batch_size": semantic_member_batch_size,
        "semantic_member_batches_truncated": False,
    }
