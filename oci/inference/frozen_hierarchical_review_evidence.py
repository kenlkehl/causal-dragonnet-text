"""Freeze accepted Stage-1 support for the first adaptive review round.

The hierarchical discovery path deliberately separates broad feature discovery
from post-extraction repair.  Discovery may inspect every architecture in a
bounded hierarchy, while the first repair round receives only the original raw
atoms cited by features that survived integration. Later rounds use a fresh
exact accumulated-spent hierarchy. This module authenticates the first boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from oci.inference.all_evidence_discovery_interfaces import canonical_json, content_sha256
from oci.inference.hierarchical_all_architecture_discovery import (
    CompletedHierarchicalDiscovery,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    RoleNeutralEvidenceCatalog,
    validate_role_neutral_catalog,
)

FROZEN_HIERARCHICAL_REVIEW_EVIDENCE_SCHEMA_VERSION = "frozen_hierarchical_review_evidence_v2"
FROZEN_HIERARCHICAL_REVIEW_EVIDENCE_POLICY_VERSION = (
    "round_1_accepted_routed_feature_support_only_v3"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _implementation_file_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def frozen_hierarchical_review_evidence_identity() -> dict[str, Any]:
    """Return the closed implementation identity committed before execution."""

    return {
        "schema_version": FROZEN_HIERARCHICAL_REVIEW_EVIDENCE_SCHEMA_VERSION,
        "policy_version": FROZEN_HIERARCHICAL_REVIEW_EVIDENCE_POLICY_VERSION,
        "implementation_file_sha256": _implementation_file_sha256(),
        "accepted_routed_support_only": True,
        "original_content_addressed_ids_preserved": True,
        "rejected_only_excluded": True,
        "planner_lookback_only_excluded": True,
        "bounds_fail_closed_without_truncation": True,
    }


@dataclass(frozen=True)
class FrozenHierarchicalReviewEvidenceConfig:
    """Hard limits for the accepted-support review catalog."""

    max_evidence_ids: int = 512
    max_evidence_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        for name in ("max_evidence_ids", "max_evidence_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")

    def as_dict(self) -> dict[str, int]:
        return {
            "max_evidence_ids": self.max_evidence_ids,
            "max_evidence_bytes": self.max_evidence_bytes,
        }


@dataclass(frozen=True)
class FrozenHierarchicalReviewEvidence:
    """Immutable accepted-support catalog for the first adaptive review round."""

    catalog_sha256: str
    completion_sha256: str
    precommit_sha256: str
    ordered_evidence_ids: tuple[str, ...]
    evidence_count: int
    evidence_bytes: int
    review_evidence_sha256: str
    binding_sha256: str
    _review_rows_json: str = field(repr=False)
    _audit_json: str = field(repr=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("catalog_sha256", self.catalog_sha256),
            ("completion_sha256", self.completion_sha256),
            ("precommit_sha256", self.precommit_sha256),
            ("review_evidence_sha256", self.review_evidence_sha256),
            ("binding_sha256", self.binding_sha256),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256")
        if len(self.ordered_evidence_ids) != len(set(self.ordered_evidence_ids)):
            raise ValueError("ordered_evidence_ids cannot contain duplicates")
        rows = json.loads(self._review_rows_json)
        audit = json.loads(self._audit_json)
        if not isinstance(rows, list) or not isinstance(audit, Mapping):
            raise TypeError("frozen review rows and audit must be valid JSON containers")
        row_ids = tuple(str(row.get("evidence_id") or "") for row in rows)
        if row_ids != self.ordered_evidence_ids:
            raise ValueError("review rows differ from ordered_evidence_ids")
        if any(
            set(row) != {"evidence_id", "source_families", "role_hint", "content"} for row in rows
        ):
            raise ValueError("frozen review rows do not have the closed legacy-review shape")
        if self.evidence_count != len(rows) or self.evidence_count != len(
            self.ordered_evidence_ids
        ):
            raise ValueError("evidence_count differs from frozen review rows")
        canonical_rows = canonical_json(rows)
        if self.evidence_bytes != len(canonical_rows.encode("utf-8")):
            raise ValueError("evidence_bytes differs from frozen review rows")
        if self.review_evidence_sha256 != content_sha256(rows):
            raise ValueError("review_evidence_sha256 does not authenticate review rows")
        identity = {
            "schema_version": FROZEN_HIERARCHICAL_REVIEW_EVIDENCE_SCHEMA_VERSION,
            "materializer_identity": frozen_hierarchical_review_evidence_identity(),
            "catalog_sha256": self.catalog_sha256,
            "completion_sha256": self.completion_sha256,
            "precommit_sha256": self.precommit_sha256,
            "ordered_evidence_ids": list(self.ordered_evidence_ids),
            "evidence_count": self.evidence_count,
            "evidence_bytes": self.evidence_bytes,
            "review_evidence_sha256": self.review_evidence_sha256,
            "audit": audit,
        }
        if self.binding_sha256 != content_sha256(identity):
            raise ValueError("binding_sha256 does not authenticate frozen review evidence")

    @property
    def review_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(self._review_rows_json))

    @property
    def audit(self) -> dict[str, Any]:
        return json.loads(self._audit_json)

    def as_binding_dict(self) -> dict[str, Any]:
        """Return provenance and bounds without copying raw evidence content."""

        return {
            "schema_version": FROZEN_HIERARCHICAL_REVIEW_EVIDENCE_SCHEMA_VERSION,
            "materializer_identity": frozen_hierarchical_review_evidence_identity(),
            "catalog_sha256": self.catalog_sha256,
            "completion_sha256": self.completion_sha256,
            "precommit_sha256": self.precommit_sha256,
            "ordered_evidence_ids": list(self.ordered_evidence_ids),
            "evidence_count": self.evidence_count,
            "evidence_bytes": self.evidence_bytes,
            "review_evidence_sha256": self.review_evidence_sha256,
            "audit": self.audit,
            "binding_sha256": self.binding_sha256,
        }


def freeze_hierarchical_review_evidence(
    *,
    catalog: RoleNeutralEvidenceCatalog,
    completed: CompletedHierarchicalDiscovery,
    config: FrozenHierarchicalReviewEvidenceConfig,
) -> FrozenHierarchicalReviewEvidence:
    """Freeze exactly the raw evidence supporting accepted routed features.

    Limits are checked against the complete accepted-support union.  Nothing is
    truncated to fit a prompt or silently replaced with architecture summaries.
    """

    if not isinstance(catalog, RoleNeutralEvidenceCatalog):
        raise TypeError("catalog must be a RoleNeutralEvidenceCatalog")
    if not isinstance(completed, CompletedHierarchicalDiscovery):
        raise TypeError("completed must be a CompletedHierarchicalDiscovery")
    if not isinstance(config, FrozenHierarchicalReviewEvidenceConfig):
        raise TypeError("config must be a FrozenHierarchicalReviewEvidenceConfig")
    validate_role_neutral_catalog(catalog)
    completed.__post_init__()
    if {dossier.catalog_sha256 for dossier in completed.dossiers} != {catalog.catalog_sha256}:
        raise ValueError("completed hierarchy is bound to a different evidence catalog")

    feature_support = [
        {
            "canonical_name": routed.feature.canonical_name,
            "supporting_evidence_ids": list(routed.feature.supporting_evidence_ids),
        }
        for routed in completed.routed_features
    ]
    if not feature_support:
        raise ValueError("completed hierarchy has no accepted routed features to review")
    accepted_support = {
        evidence_id
        for routed in completed.routed_features
        for evidence_id in routed.feature.supporting_evidence_ids
    }
    if not accepted_support:
        raise ValueError("accepted routed features have no supporting evidence")

    atom_by_id = {atom.evidence_id: atom for atom in catalog.atoms}
    missing = accepted_support - set(atom_by_id)
    if missing:
        raise ValueError(f"accepted features cite evidence outside the catalog: {sorted(missing)}")
    ordered_ids = tuple(
        atom.evidence_id for atom in catalog.atoms if atom.evidence_id in accepted_support
    )
    if set(ordered_ids) != accepted_support:
        raise RuntimeError("accepted-support ordering lost evidence IDs")

    rows: list[dict[str, Any]] = []
    for evidence_id in ordered_ids:
        item = atom_by_id[evidence_id].as_discovery_item()
        rows.append(
            {
                "evidence_id": item.evidence_id,
                "source_families": [item.source_family],
                "role_hint": "",
                "content": dict(item.content),
            }
        )
    evidence_bytes = len(canonical_json(rows).encode("utf-8"))
    if len(rows) > config.max_evidence_ids:
        raise ValueError(
            "accepted-support review evidence exceeds max_evidence_ids; refusing to truncate"
        )
    if evidence_bytes > config.max_evidence_bytes:
        raise ValueError(
            "accepted-support review evidence exceeds max_evidence_bytes; refusing to truncate"
        )

    candidate_by_id = {
        candidate.candidate_id: candidate
        for dossier in completed.dossiers
        for candidate in dossier.architecture_candidates
    }
    rejected_support = {
        evidence_id
        for candidate_id in completed.rejected_candidate_ids
        for evidence_id in candidate_by_id[candidate_id].supporting_evidence_ids
    }
    requested_lookback = set(completed.requested_lookback_evidence_ids)
    rejected_only = rejected_support - accepted_support
    planner_only = requested_lookback - accepted_support - rejected_support
    classified_source_universe = accepted_support | rejected_only | planner_only
    source_universe = accepted_support | rejected_support | requested_lookback
    if classified_source_universe != source_universe:
        raise RuntimeError("frozen-review audit classifications lost source evidence")
    if (
        accepted_support.intersection(rejected_only)
        or accepted_support.intersection(planner_only)
        or rejected_only.intersection(planner_only)
    ):
        raise RuntimeError("frozen-review audit classifications must be disjoint")
    if rejected_only.intersection(ordered_ids) or planner_only.intersection(ordered_ids):
        raise RuntimeError("review evidence contains rejected-only or planner-only evidence")

    audit = {
        "policy_version": FROZEN_HIERARCHICAL_REVIEW_EVIDENCE_POLICY_VERSION,
        "config": config.as_dict(),
        "accepted_feature_support": feature_support,
        "accepted_feature_support_sha256": content_sha256(feature_support),
        "ordered_evidence_ids_sha256": content_sha256(list(ordered_ids)),
        "rejected_only_evidence_id_count_excluded": len(rejected_only),
        "rejected_only_evidence_ids_sha256": content_sha256(sorted(rejected_only)),
        "planner_lookback_only_evidence_id_count_excluded": len(planner_only),
        "planner_lookback_only_evidence_ids_sha256": content_sha256(sorted(planner_only)),
        "architecture_wide_evidence_dumped_to_review": False,
        "dynamic_semantic_refit_for_review": False,
        "round_1_initial_frozen_catalog_only": True,
        "same_frozen_catalog_used_for_later_review_rounds": False,
        "row_level_values_in_review_evidence": False,
    }
    review_evidence_sha256 = content_sha256(rows)
    identity = {
        "schema_version": FROZEN_HIERARCHICAL_REVIEW_EVIDENCE_SCHEMA_VERSION,
        "materializer_identity": frozen_hierarchical_review_evidence_identity(),
        "catalog_sha256": catalog.catalog_sha256,
        "completion_sha256": completed.completion_sha256,
        "precommit_sha256": completed.precommit_sha256,
        "ordered_evidence_ids": list(ordered_ids),
        "evidence_count": len(rows),
        "evidence_bytes": evidence_bytes,
        "review_evidence_sha256": review_evidence_sha256,
        "audit": audit,
    }
    return FrozenHierarchicalReviewEvidence(
        catalog_sha256=catalog.catalog_sha256,
        completion_sha256=completed.completion_sha256,
        precommit_sha256=completed.precommit_sha256,
        ordered_evidence_ids=ordered_ids,
        evidence_count=len(rows),
        evidence_bytes=evidence_bytes,
        review_evidence_sha256=review_evidence_sha256,
        binding_sha256=content_sha256(identity),
        _review_rows_json=canonical_json(rows),
        _audit_json=canonical_json(audit),
    )
