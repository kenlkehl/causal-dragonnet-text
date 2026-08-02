"""Oracle-free post-extraction contract review for all-evidence fusion.

The all-evidence selector operates on model-derived semantic evidence.  This
module implements the separate, later review of the *real extracted values*.
It deliberately has no interface for an oracle treatment effect.  Agent
responses are closed-schema operations over the currently frozen contracts,
and deterministic helpers identify exactly which operations require new
extraction.

The causal acceptance evaluator is intentionally kept separate from the
response grammar below.  A caller can therefore validate and cache a remote
response before any observed outcome is used to score the proposed revision.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from itertools import combinations
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)

from .all_evidence_fusion import (
    CandidateContract,
    ground_evidence_to_extraction_contract,
    source_text_temporal_policy_audit,
)
from .fold_honest_r_stack import FitRowProvenance
from .post_extraction_scientific_policy import (
    ExtractionQualityPolicy,
    ExtractionRedundancyPolicy,
    ReviewEstimatorPolicy,
)

POST_EXTRACTION_REVIEW_PROMPT_VERSION = "all_evidence_post_extraction_review_v12"
POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION = "all_evidence_post_extraction_review_response_v1"
POST_EXTRACTION_REVIEW_FRESH_NORMALIZATION_VERSION = (
    "all_evidence_post_extraction_review_fresh_normalization_v1"
)
POST_EXTRACTION_REVIEW_GROUNDING_REPAIR_VERSION = "exact_sanitized_catalog_grounding_repair_v1"
POST_EXTRACTION_QUALITY_SCHEMA_VERSION = "all_evidence_extraction_quality_v2"
POST_EXTRACTION_CAUSAL_DIAGNOSTIC_SCHEMA_VERSION = (
    "all_evidence_post_extraction_causal_diagnostic_v1"
)
POST_EXTRACTION_GATE_DECISION_SCHEMA_VERSION = "all_evidence_post_extraction_gate_decision_v4"

CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY = "conditional_context_and_gate_v1"
GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY = "gate_only_reference_preservation_v1"
_UPSTREAM_REVIEW_POLICIES = frozenset(
    {
        CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY,
        GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
    }
)

PROPENSITY_NUISANCE_FEATURE_ROLE = "propensity_nuisance_features"
OUTCOME_NUISANCE_FEATURE_ROLE = "outcome_nuisance_features"
UNCALIBRATED_EFFECT_MODIFIER_ROLE = "uncalibrated_effect_modifier_basis"
_GATE_FEATURE_BANK_ROLES = frozenset(
    {
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    }
)
RAW_UNCALIBRATED_FEATURE_SOURCE_KINDS = frozenset(
    {
        "neural_query_moments",
        "neural_query_treatment_moments",
        "neural_query_outcome_moments",
        "neural_query_effect_moments",
        "sparse_query_moments",
        "matched_pair_uplift",
        "whole_embedding_contrast",
        "cluster_embedding_contrast",
        "tfidf_topic_contrast",
        "embedding_whole_cohort",
        "embedding_clustered",
        "tfidf_topics",
        "tfidf_orphan_ngrams",
    }
)

_VALID_ACTIONS = frozenset({"drop", "merge", "re_role", "replace", "revise", "stop"})
_VALID_ROLES = frozenset({"confounder", "effect_modifier"})
_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


def expected_extraction_columns(
    spec: Mapping[str, Any] | Any,
) -> tuple[str, str]:
    """Return the ordinary value and missingness columns for one feature."""

    if is_dataclass(spec) and not isinstance(spec, type):
        spec = asdict(spec)
    if not isinstance(spec, Mapping):
        raise TypeError("an extraction contract must be a mapping or dataclass")
    name = str(spec.get("name") or "").strip()
    if not name:
        raise ValueError("extraction contract name must be non-empty")
    return f"explicit_feat_{name}", f"explicit_feat_{name}_missing"
_OPAQUE_DIAGNOSTIC_ID = re.compile(r"^diagnostic_[0-9]{4}$")
_OPAQUE_EVIDENCE_ID = re.compile(r"^evidence_(?:[0-9]+|[0-9a-f]{64})$")


class PostExtractionReviewResponseExhausted(ValueError):
    """The remote reviewer exhausted its bounded JSON/semantic repair attempts."""


_FORBIDDEN = re.compile(
    r"(?:^|_)(?:true|oracle|ground_truth)(?:_|$)",
    flags=re.IGNORECASE,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _detached(value: Any) -> Any:
    return json.loads(_canonical_json(value))


_MODEL_FACING_MACHINE_ONLY_EXACT_KEYS = frozenset(
    {
        "prompt_version",
        "schema_version",
        "source_text_temporal_policy",
    }
)
_MODEL_FACING_MACHINE_ONLY_KEY_SUFFIXES = (
    "_policy_version",
    "_sha256",
    "_diagnostic_version",
)
_MODEL_FACING_TEMPORAL_POLICY_MARKERS = frozenset(
    {
        "source_text_temporal_policy",
        "source_text_temporally_valid_by_design_v1",
        "source_text_temporally_valid_by_design",
        "temporal_boundary_enforced",
        "post_treatment_semantic_filtering_enabled",
        "temporal_eligibility_affects_selection_or_acceptance",
        "semantic_timepoint_fields_allowed_as_extraction_meaning",
    }
)


def _is_machine_only_review_prompt_key(value: Any) -> bool:
    normalized = str(value).casefold()
    return normalized in _MODEL_FACING_MACHINE_ONLY_EXACT_KEYS or normalized.endswith(
        _MODEL_FACING_MACHINE_ONLY_KEY_SUFFIXES
    )


def _without_machine_only_review_metadata(value: Any) -> Any:
    """Copy prompt data while removing internal machine metadata at any depth."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_machine_only_review_metadata(child)
            for key, child in value.items()
            if not _is_machine_only_review_prompt_key(key)
        }
    if isinstance(value, list):
        return [_without_machine_only_review_metadata(child) for child in value]
    return value


def _assert_no_model_facing_machine_metadata(value: Any) -> None:
    """Fail closed if machine-only keys or timing-policy values escaped projection."""

    def visit(item: Any, *, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if _is_machine_only_review_prompt_key(key):
                    raise ValueError(
                        "post-extraction review prompt contains internal machine metadata "
                        f"key: {path}.{key}"
                    )
                visit(child, path=f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, path=f"{path}[{index}]")

    visit(value, path="context")

    serialized = _canonical_json(value).casefold()
    leaked = sorted(
        marker for marker in _MODEL_FACING_TEMPORAL_POLICY_MARKERS if marker in serialized
    )
    if leaked:
        raise ValueError(
            "post-extraction review prompt contains internal temporal-policy audit markers: "
            f"{leaked}"
        )


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _semantic_extraction_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return extraction-affecting fields, deliberately excluding causal roles."""

    contract = CandidateContract(spec).extraction_spec
    return {
        key: contract.get(key)
        for key in ("name", "type", "categories", "description", "value_aliases")
    }


def extraction_semantics_sha256(spec: Mapping[str, Any]) -> str:
    """Hash only fields that can change extracted values.

    Roles affect the causal estimator but are not instructions to the value
    extractor.  Keeping this identity separate prevents a role-only revision
    from needlessly issuing another remote extraction request.
    """

    return _content_sha256(_semantic_extraction_contract(spec))


@dataclass(frozen=True)
class ReviewOperation:
    action: str
    target_names: tuple[str, ...]
    contract: Mapping[str, Any] | None
    supporting_diagnostic_ids: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    reason: str
    evidence_contract_grounding: tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple,
        repr=False,
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target_names": list(self.target_names),
            "contract": None if self.contract is None else _detached(self.contract),
            "supporting_diagnostic_ids": list(self.supporting_diagnostic_ids),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "reason": self.reason,
        }

    def audit_dict(self) -> dict[str, Any]:
        result = self.as_dict()
        result["evidence_contract_grounding"] = [
            _detached(row) for row in self.evidence_contract_grounding
        ]
        return result


@dataclass(frozen=True)
class ValidatedReviewResponse:
    operations: tuple[ReviewOperation, ...]
    response_sha256: str

    @property
    def stops(self) -> bool:
        return len(self.operations) == 1 and self.operations[0].action == "stop"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
            "operations": [operation.as_dict() for operation in self.operations],
        }


@dataclass(frozen=True)
class AppliedReviewOperations:
    specs: tuple[dict[str, Any], ...]
    reextract_specs: tuple[dict[str, Any], ...]
    removed_names: tuple[str, ...]
    added_names: tuple[str, ...]
    extraction_changed_names: tuple[str, ...]
    role_only_changed_names: tuple[str, ...]
    operation_audit: tuple[dict[str, Any], ...]


def _string_list(
    value: Any,
    *,
    name: str,
    pattern: re.Pattern[str] | None = None,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a string list")
    result = tuple(item.strip() for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{name} must be non-empty")
    if any(not item for item in result):
        raise ValueError(f"{name} cannot contain empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} cannot contain duplicates")
    if pattern is not None and any(pattern.fullmatch(item) is None for item in result):
        raise ValueError(f"{name} contains a malformed opaque ID")
    return result


def collect_post_extraction_diagnostic_ids(value: Any) -> tuple[str, ...]:
    """Collect every nested review diagnostic ID in deterministic JSON order.

    Contract ablations live inside the aggregate causal diagnostic, while gate
    and retry feedback are top-level diagnostic rows.  Treating only top-level
    IDs as citable made the nested ablation IDs visible but unusable.  This
    recursive collector is shared by prompt repair and runner validation so the
    two paths cannot disagree.
    """

    found: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            if "diagnostic_id" in item:
                diagnostic_id = str(item["diagnostic_id"] or "").strip()
                if _OPAQUE_DIAGNOSTIC_ID.fullmatch(diagnostic_id) is None:
                    raise ValueError("review context contains a malformed diagnostic ID")
                found.append(diagnostic_id)
            for key, child in item.items():
                if key != "diagnostic_id":
                    visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    duplicates = sorted({item for item in found if found.count(item) > 1})
    if duplicates:
        raise ValueError(f"review context contains duplicate diagnostic IDs: {duplicates}")
    return tuple(found)


def collect_post_extraction_diagnostic_targets(
    value: Any,
) -> dict[str, tuple[str, ...]]:
    """Map every opaque diagnostic ID to the contracts it directly diagnoses.

    A parent aggregate does not inherit the targets of nested diagnostics.  For
    example, the overall causal-quality row is targetless while each nested
    contract ablation names exactly one contract.  This prevents an agent from
    citing a generic aggregate to justify an unrelated edit.
    """

    diagnostic_ids = collect_post_extraction_diagnostic_ids(value)
    targets: dict[str, tuple[str, ...]] = {}

    def direct_targets(item: Mapping[str, Any]) -> tuple[str, ...]:
        names: list[str] = []
        for target_field in ("feature_name", "contract_name"):
            raw = item.get(target_field)
            if isinstance(raw, str) and raw.strip():
                names.append(raw.strip())
        for target_field in ("feature_names", "failed_contract_names"):
            raw = item.get(target_field)
            if isinstance(raw, (list, tuple)):
                names.extend(str(name).strip() for name in raw if str(name).strip())
        if str(item.get("kind") or "") == "prior_gate_feedback":
            for operation in item.get("prior_operations") or ():
                if isinstance(operation, Mapping):
                    raw_targets = operation.get("target_names")
                    if isinstance(raw_targets, (list, tuple)):
                        names.extend(str(name).strip() for name in raw_targets if str(name).strip())
        return tuple(dict.fromkeys(names))

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            if "diagnostic_id" in item:
                diagnostic_id = str(item["diagnostic_id"]).strip()
                targets[diagnostic_id] = direct_targets(item)
            for key, child in item.items():
                if key != "diagnostic_id":
                    visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    if tuple(targets) != diagnostic_ids:
        raise RuntimeError("diagnostic target mapping disagrees with diagnostic ID traversal")
    return targets


def validate_post_extraction_review_response(
    response: Mapping[str, Any],
    *,
    current_specs: Sequence[Mapping[str, Any]],
    available_diagnostic_ids: Sequence[str],
    available_diagnostic_targets: Mapping[str, Sequence[str]] | None = None,
    available_evidence_ids: Sequence[str] = (),
    available_evidence_catalog: Sequence[Mapping[str, Any]] | None = None,
    max_operations: int = 8,
) -> ValidatedReviewResponse:
    """Validate one closed-schema agent response without scoring it.

    Every non-stop operation must cite at least one supplied diagnostic.  Any
    source-evidence citations are also checked against the sanitized context.
    The returned contracts have already passed the same identifier, schema, and
    ontology validator used by initial all-evidence fusion.
    """

    if not isinstance(response, Mapping):
        raise TypeError("post-extraction review response must be one JSON object")
    allowed_response_fields = {"schema_version", "operations"}
    if set(response) != allowed_response_fields:
        missing = sorted(allowed_response_fields - set(response))
        unexpected = sorted(set(response) - allowed_response_fields)
        raise ValueError(
            "post-extraction review response must contain exactly schema_version and "
            f"operations; missing fields={missing}; unexpected fields={unexpected}"
        )
    if response.get("schema_version") != POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION:
        raise ValueError("unsupported post-extraction review response schema")
    raw_operations = response.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        raise ValueError("operations must be a non-empty list")
    if isinstance(max_operations, bool) or not 1 <= int(max_operations) <= 32:
        raise ValueError("max_operations must be in [1, 32]")
    if len(raw_operations) > int(max_operations):
        raise ValueError("review response exceeds max_operations")

    canonical_current = [CandidateContract(spec).extraction_spec for spec in current_specs]
    current_by_name = {str(spec["name"]): spec for spec in canonical_current}
    if len(current_by_name) != len(canonical_current):
        raise ValueError("current_specs contains duplicate names")
    diagnostic_ids = set(
        _string_list(
            list(available_diagnostic_ids),
            name="available_diagnostic_ids",
            pattern=_OPAQUE_DIAGNOSTIC_ID,
            allow_empty=True,
        )
    )
    evidence_ids = set(
        _string_list(
            list(available_evidence_ids),
            name="available_evidence_ids",
            pattern=_OPAQUE_EVIDENCE_ID,
            allow_empty=True,
        )
    )
    diagnostic_targets: dict[str, frozenset[str]] | None = None
    if available_diagnostic_targets is not None:
        if not isinstance(available_diagnostic_targets, Mapping):
            raise TypeError("available_diagnostic_targets must be a mapping")
        if set(map(str, available_diagnostic_targets)) != diagnostic_ids:
            raise ValueError(
                "available_diagnostic_targets must map every available diagnostic ID exactly"
            )
        diagnostic_targets = {}
        for diagnostic_id, raw_targets in available_diagnostic_targets.items():
            normalized = _string_list(
                list(raw_targets),
                name=f"available_diagnostic_targets[{diagnostic_id}]",
                allow_empty=True,
            )
            diagnostic_targets[str(diagnostic_id)] = frozenset(normalized)
    evidence_catalog_by_id: dict[str, Mapping[str, Any]] | None = None
    if available_evidence_catalog is not None:
        if isinstance(available_evidence_catalog, (str, bytes, Mapping)):
            raise TypeError("available_evidence_catalog must be a sequence of evidence rows")
        evidence_catalog_by_id = {}
        for position, raw_evidence in enumerate(available_evidence_catalog):
            if not isinstance(raw_evidence, Mapping):
                raise TypeError(
                    f"available_evidence_catalog[{position}] must be an evidence mapping"
                )
            evidence_id = str(raw_evidence.get("evidence_id") or "").strip()
            if _OPAQUE_EVIDENCE_ID.fullmatch(evidence_id) is None:
                raise ValueError("available_evidence_catalog contains a malformed evidence ID")
            if evidence_id in evidence_catalog_by_id:
                raise ValueError("available_evidence_catalog contains duplicate evidence IDs")
            evidence_catalog_by_id[evidence_id] = raw_evidence
        if set(evidence_catalog_by_id) != evidence_ids:
            raise ValueError(
                "available_evidence_catalog must contain every available evidence ID exactly"
            )

    allowed_fields = {
        "action",
        "target_names",
        "contract",
        "supporting_diagnostic_ids",
        "supporting_evidence_ids",
        "reason",
    }
    operations: list[ReviewOperation] = []
    used_targets: set[str] = set()
    for index, raw in enumerate(raw_operations):
        path = f"operations[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must be an object with the exact operation schema")
        if set(raw) != allowed_fields:
            missing = sorted(allowed_fields - set(raw))
            unexpected = sorted(set(raw) - allowed_fields)
            raise ValueError(
                f"{path} must contain exactly {sorted(allowed_fields)}; "
                f"missing fields={missing}; unexpected fields={unexpected}"
            )
        action = str(raw.get("action") or "").strip().lower()
        if action not in _VALID_ACTIONS:
            raise ValueError(f"{path}.action must be one of {sorted(_VALID_ACTIONS)}")
        targets = _string_list(
            raw.get("target_names"),
            name=f"{path}.target_names",
            allow_empty=action == "stop",
        )
        if any(_SNAKE_CASE.fullmatch(name) is None or _FORBIDDEN.search(name) for name in targets):
            raise ValueError(f"{path}.target_names contains an invalid feature name")
        if action == "stop":
            if len(raw_operations) != 1:
                raise ValueError("stop must be the only review operation")
            if targets or raw.get("contract") is not None:
                raise ValueError("stop cannot declare targets or a contract")
        else:
            unknown = set(targets) - set(current_by_name)
            if unknown:
                raise ValueError(f"{path} targets unknown features: {sorted(unknown)}")
            overlap = set(targets) & used_targets
            if overlap:
                raise ValueError(f"review operations target features twice: {sorted(overlap)}")
            used_targets.update(targets)

        if action in {"drop", "re_role", "replace", "revise"} and len(targets) != 1:
            raise ValueError(f"{path}.{action} requires exactly one target")
        if action == "merge" and len(targets) < 2:
            raise ValueError(f"{path}.merge requires at least two targets")

        raw_contract = raw.get("contract")
        contract: dict[str, Any] | None = None
        if action in {"merge", "re_role", "replace", "revise"}:
            if not isinstance(raw_contract, Mapping):
                raise ValueError(f"{path}.{action} requires one replacement contract")
            contract = CandidateContract(raw_contract).extraction_spec
        elif raw_contract is not None:
            raise ValueError(f"{path}.{action} cannot include a contract")

        if action in {"re_role", "revise"} and contract is not None:
            if contract["name"] != targets[0]:
                raise ValueError(f"{path}.{action} must preserve the target name")
        if action == "re_role" and contract is not None:
            before = current_by_name[targets[0]]
            if _semantic_extraction_contract(before) != _semantic_extraction_contract(contract):
                raise ValueError("re_role may change roles only")
            if tuple(before.get("roles") or ()) == tuple(contract.get("roles") or ()):
                raise ValueError("re_role must actually change a causal role")
        if action == "revise" and contract is not None:
            before = current_by_name[targets[0]]
            if before == contract:
                raise ValueError("revise must change the contract")
        if action in {"replace", "merge"} and contract is not None:
            untouched_names = set(current_by_name) - set(targets)
            if contract["name"] in untouched_names:
                raise ValueError("replacement contract collides with an untouched feature")

        cited_diagnostics = _string_list(
            raw.get("supporting_diagnostic_ids"),
            name=f"{path}.supporting_diagnostic_ids",
            pattern=_OPAQUE_DIAGNOSTIC_ID,
            allow_empty=action == "stop",
        )
        unknown_diagnostics = set(cited_diagnostics) - diagnostic_ids
        if unknown_diagnostics:
            raise ValueError(f"{path} cites unknown diagnostic IDs: {sorted(unknown_diagnostics)}")
        if action != "stop" and diagnostic_targets is not None:
            diagnosed_targets = set().union(
                *(diagnostic_targets[diagnostic_id] for diagnostic_id in cited_diagnostics)
            )
            unsupported_targets = set(targets) - diagnosed_targets
            if unsupported_targets:
                raise ValueError(
                    f"{path} must cite a diagnostic that directly names every target: "
                    f"{sorted(unsupported_targets)}"
                )
        cited_evidence = _string_list(
            raw.get("supporting_evidence_ids"),
            name=f"{path}.supporting_evidence_ids",
            pattern=_OPAQUE_EVIDENCE_ID,
            allow_empty=True,
        )
        unknown_evidence = set(cited_evidence) - evidence_ids
        if unknown_evidence:
            raise ValueError(f"{path} cites unknown evidence IDs: {sorted(unknown_evidence)}")
        if action in {"merge", "replace", "revise"} and not cited_evidence:
            raise ValueError(f"{path}.{action} must cite source evidence for its contract change")
        grounding_rows: list[dict[str, Any]] = []
        if action in {"merge", "replace", "revise"} and contract is not None:
            if evidence_catalog_by_id is None:
                raise ValueError(
                    f"{path}.{action} requires available_evidence_catalog for semantic "
                    "source-evidence validation"
                )
            unrelated: list[str] = []
            for evidence_id in cited_evidence:
                grounding = ground_evidence_to_extraction_contract(
                    evidence_catalog_by_id[evidence_id],
                    contract,
                )
                grounding_rows.append({"evidence_id": evidence_id, **grounding.as_dict()})
                if not grounding.supported:
                    unrelated.append(evidence_id)
            if unrelated:
                raise ValueError(
                    f"{path}.{action} cites evidence unrelated to the proposed contract: "
                    f"{unrelated}"
                )
        reason = str(raw.get("reason") or "").strip()
        if not reason or len(reason) > 1200:
            raise ValueError(f"{path}.reason must contain 1-1200 characters")
        operations.append(
            ReviewOperation(
                action=action,
                target_names=targets,
                contract=None if contract is None else _detached(contract),
                supporting_diagnostic_ids=cited_diagnostics,
                supporting_evidence_ids=cited_evidence,
                reason=reason,
                evidence_contract_grounding=tuple(grounding_rows),
            )
        )

    normalized = {
        "schema_version": POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION,
        "operations": [operation.as_dict() for operation in operations],
    }
    return ValidatedReviewResponse(
        operations=tuple(operations),
        response_sha256=_content_sha256(normalized),
    )


def render_post_extraction_review_prompt(context: Mapping[str, Any]) -> str:
    """Render the bounded reasoning-agent task from a sanitized context."""

    if not isinstance(context, Mapping):
        raise TypeError("post-extraction review context must be an object")
    if context.get("prompt_version") != POST_EXTRACTION_REVIEW_PROMPT_VERSION:
        raise ValueError("unsupported post-extraction review prompt version")
    if context.get("source_text_temporal_policy") != source_text_temporal_policy_audit():
        raise ValueError("unsupported source-text temporal policy")
    maximum = context.get("max_operations")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 32:
        raise ValueError("post-extraction review max_operations must be in [1, 32]")
    current = context.get("current_contracts")
    diagnostics = context.get("diagnostics")
    evidence = context.get("sanitized_evidence_catalog")
    if not isinstance(current, list) or not current:
        raise ValueError("post-extraction review requires current_contracts")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise ValueError("post-extraction review requires diagnostics")
    if not isinstance(evidence, list):
        raise ValueError("sanitized_evidence_catalog must be a list")
    # Validate the supplied contracts before embedding them in a remote prompt.
    for spec in current:
        CandidateContract(spec)
    # Machine provenance remains available in the internal context/audit, but
    # is recursively removed from this detached copy before any remote request.
    prompt_context = _without_machine_only_review_metadata(_detached(context))
    _assert_no_model_facing_machine_metadata(prompt_context)
    context_json = json.dumps(
        prompt_context,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"""You are performing a bounded post-extraction review for a causal text model.

All information below comes from the current outer-training fold. The next
acceptance gate is intentionally absent: you cannot see its rows, outcomes, or
metrics. You also have no true treatment effects, oracle variables, patient
identifiers, or post-hoc benchmark results.

Review the *observed extracted representation*, not merely the original idea.
Use the supplied diagnostic IDs to address extraction coverage and missingness,
value plausibility, stability across already-spent inner
folds, cross-fitted R-loss, preservation of authenticated upstream numerical
and neural-query signals, redundancy, and feature/source ablations. Raw
upstream features are evaluated only against their declared consumer role, and
each exact (source kind, consumer role) family is guarded independently so one
family cannot compensate for degradation in another. The untouched-gate R-loss
is conditional on the same frozen upstream numerical base used by the final
head: treatment and outcome raw features enter only their matched nuisance
models, while raw effect features and calibrated effect signals enter the
effect regression. Family ablations delete a family and refit; a raw feature is
never interpreted directly as a treatment effect.
Text-grounding diagnostics are aggregate lexical audits over already-spent
notes. Treat repeated locally grounded support for a different declared
category as an unsafe ontology mismatch. Use missing-value opportunities,
category conflicts, weak grounding, alias, or unit warnings as reasons to
clarify the ontology/contract rather than as proof that a row-level value is wrong.
Diagnostics may also contain sanitized prior-gate or extraction-quality retry
feedback. Respect its non-repeat guidance: address the reported failure or make
a materially different proposal while the next gate remains sealed.
The context includes a spent-only ``required_safety_remediation`` summary. Every
contract listed there must be dropped, replaced, revised, or included in a
defensible merge before the cumulative candidate can reach the untouched gate.
Each listed contract also includes ``same_name_grounded_evidence_ids`` computed
by the exact validator grounder. Use one of those IDs for an in-place revision.
If that list is empty and no supplied evidence literally grounds every anchor in
a replacement name, use the listed safe fallback action (drop); do not invent a
broader concept or an unrelated citation merely to preserve the feature.
Partial repairs may be staged in the sealed candidate workspace across bounded
attempts, but they are not accepted changes and reveal no gate information.

You may:
- drop one weak or unsafe contract;
- merge two or more redundant contracts into one operational contract;
- re_role a correctly extracted contract without changing its value contract;
- replace one contract with a clearer evidence-grounded target;
- revise a contract in place, including its categories, measurement definition,
  semantic measurement/timepoint definition, or causal roles; or
- stop when no defensible improvement remains.

Return exactly one JSON object with this schema:
{{
  "operations": [
    {{
      "action": "drop|merge|re_role|replace|revise|stop",
      "target_names": ["existing_contract_name"],
      "contract": {{
        "name": "snake_case_name",
        "type": "categorical|continuous",
        "categories": ["absent", "present"],
        "value_aliases": {{"absent": ["negative"], "present": ["positive"]}},
        "roles": ["confounder", "effect_modifier"],
        "description": "precise extraction target for the supported construct"
      }},
      "supporting_diagnostic_ids": ["diagnostic_0001"],
      "supporting_evidence_ids": ["evidence_0001"],
      "reason": "brief evidence-grounded reason"
    }}
  ]
}}

Rules:
- Return at most {maximum} operations.
- Address as many entries in required_safety_remediation as the operation budget
  permits. Do not spend operations on role-only or cosmetic edits while a listed
  category-ontology hard failure remains unresolved.
- Every operation object always contains exactly these six keys, even when a
  value is null or an empty list: action, target_names, contract,
  supporting_diagnostic_ids, supporting_evidence_ids, reason.
- Every non-stop operation must cite at least one supplied diagnostic ID.
- Each cited diagnostic must directly name the operation target; a generic
  aggregate cannot justify an unrelated edit.
- merge, replace, and revise must cite supplied source evidence with one
  sanitized concept entry containing all normalized exact lexical identity
  anchors in the new or revised contract name. Short acronyms count only when
  that exact token occurs; never infer synonyms or acronym expansions.
  Descriptions, categories, aliases, numeric codes, and structural name words
  cannot establish that match. Never invent an evidence ID or cite an unrelated
  block merely because its ID exists.
- A target may appear in only one operation in this response.
- re_role and revise preserve the target name. re_role changes roles only.
- merge targets at least two existing names and supplies one replacement contract.
- replace targets exactly one existing name and supplies one replacement contract.
- drop uses "contract": null and "supporting_evidence_ids": [] when it cites no
  source evidence; it still cites a target and diagnostic.
- stop is the sole operation and uses "target_names": [], "contract": null,
  "supporting_diagnostic_ids": [], and "supporting_evidence_ids": [].
- re_role always includes supporting_evidence_ids, using [] when no source
  evidence is needed.
- Categorical contracts require 2-8 variable-specific, mutually exclusive values.
- Optional value_aliases must use exact declared categories as keys and map each
  normalized alias to only one category. Aliases must be genuine variants: never
  repeat a canonical category as its own alias, and never repeat a normalized
  alias. Omit aliases for continuous contracts.
- Do not invent a concept from general medical knowledge; ground replacements
  and revisions in the supplied evidence and diagnostics.
- Complexity is penalized by the deterministic acceptance evaluator. Prefer a
  simpler contract set when observable signal is genuinely preserved.
- Do not predict whether the hidden acceptance gate will pass. Propose the most
  defensible revision, or stop.
- Never infer row-level values or source identities from opaque gate/source IDs.

Sanitized review context:
{context_json}
"""


def _normalized_review_alias(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", value).strip().casefold()


def _normalize_fresh_contract_aliases(contract: dict[str, Any]) -> dict[str, int]:
    """Remove only tautological or same-owner duplicate aliases.

    Cross-category collisions are intentionally retained so strict validation
    sends them back to the reasoning agent instead of choosing an ontology.
    """

    result = {
        "self_alias_removed_count": 0,
        "same_owner_duplicate_removed_count": 0,
    }
    categories = contract.get("categories")
    aliases = contract.get("value_aliases")
    if (
        not isinstance(categories, list)
        or not all(isinstance(value, str) and value.strip() for value in categories)
        or not isinstance(aliases, Mapping)
        or not aliases
        or not all(isinstance(key, str) for key in aliases)
        or not set(aliases).issubset(set(categories))
        or not all(
            isinstance(values, list)
            and all(isinstance(value, str) and value.strip() for value in values)
            for values in aliases.values()
        )
    ):
        return result

    normalized: dict[str, list[str]] = {}
    for owner, values in aliases.items():
        owner_normalized = _normalized_review_alias(owner)
        seen_for_owner: set[str] = set()
        retained: list[str] = []
        for alias in values:
            alias_normalized = _normalized_review_alias(alias)
            if alias_normalized == owner_normalized:
                result["self_alias_removed_count"] += 1
                continue
            if alias_normalized in seen_for_owner:
                result["same_owner_duplicate_removed_count"] += 1
                continue
            seen_for_owner.add(alias_normalized)
            retained.append(alias)
        if retained:
            normalized[owner] = retained
    if normalized:
        contract["value_aliases"] = normalized
    else:
        contract.pop("value_aliases", None)
    return result


def _normalize_fresh_post_extraction_review_response(
    response: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonicalize only nonsemantic omissions in a fresh remote response.

    This helper is deliberately absent from standalone and cached validation.
    It never invents an action, target, contract, diagnostic/evidence citation,
    reason, or category. Unknown fields and cross-category alias collisions are
    left untouched for the ordinary remote-repair path.
    """

    if not isinstance(response, Mapping):
        raise TypeError("post-extraction review response must be one JSON object")
    if not isinstance(context, Mapping):
        raise TypeError("post-extraction review context must be an object")
    if context.get("prompt_version") != POST_EXTRACTION_REVIEW_PROMPT_VERSION:
        raise ValueError("unsupported post-extraction review prompt version")
    if context.get("source_text_temporal_policy") != source_text_temporal_policy_audit():
        raise ValueError("unsupported source-text temporal policy")

    normalized = copy.deepcopy(dict(response))
    original_schema_present = "schema_version" in normalized
    original_schema_matches = (
        normalized.get("schema_version") == POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION
    )
    if not original_schema_matches:
        normalized["schema_version"] = POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION
    raw_operations = normalized.get("operations")
    operation_audits: list[dict[str, Any]] = []
    if isinstance(raw_operations, list):
        for index, raw_operation in enumerate(raw_operations):
            if not isinstance(raw_operation, Mapping):
                operation_audits.append(
                    {
                        "operation_index": index,
                        "recognized_action": None,
                        "original_field_count": None,
                        "missing_allowed_fields": [],
                        "unexpected_field_count": None,
                        "inserted_fields": [],
                        "self_alias_removed_count": 0,
                        "same_owner_duplicate_removed_count": 0,
                        "changed": False,
                    }
                )
                continue
            operation = copy.deepcopy(dict(raw_operation))
            original = copy.deepcopy(operation)
            action_value = operation.get("action")
            action = str(action_value).strip().lower() if isinstance(action_value, str) else ""
            inserted_fields: list[str] = []

            defaults: dict[str, Any] = {}
            if action == "drop":
                defaults = {"contract": None, "supporting_evidence_ids": []}
            elif action == "stop":
                defaults = {
                    "target_names": [],
                    "contract": None,
                    "supporting_diagnostic_ids": [],
                    "supporting_evidence_ids": [],
                }
            elif action == "re_role":
                defaults = {"supporting_evidence_ids": []}
            for field_name, default in defaults.items():
                if field_name not in operation:
                    operation[field_name] = copy.deepcopy(default)
                    inserted_fields.append(field_name)

            alias_audit = {
                "self_alias_removed_count": 0,
                "same_owner_duplicate_removed_count": 0,
            }
            contract = operation.get("contract")
            if isinstance(contract, Mapping):
                detached_contract = copy.deepcopy(dict(contract))
                alias_audit = _normalize_fresh_contract_aliases(detached_contract)
                operation["contract"] = detached_contract

            raw_operations[index] = operation
            allowed_fields = {
                "action",
                "target_names",
                "contract",
                "supporting_diagnostic_ids",
                "supporting_evidence_ids",
                "reason",
            }
            operation_audits.append(
                {
                    "operation_index": index,
                    "recognized_action": action if action in _VALID_ACTIONS else None,
                    "original_field_count": len(original),
                    "missing_allowed_fields": sorted(allowed_fields - set(original)),
                    "unexpected_field_count": len(set(original) - allowed_fields),
                    "inserted_fields": sorted(inserted_fields),
                    **alias_audit,
                    "changed": operation != original,
                }
            )

    audit = {
        "schema_version": POST_EXTRACTION_REVIEW_FRESH_NORMALIZATION_VERSION,
        "authoritative_source": "action_grammar_and_contract_categories",
        "original_response_schema_present": original_schema_present,
        "original_response_schema_matched": original_schema_matches,
        "response_schema_normalized": not original_schema_matches,
        "operation_count": len(raw_operations) if isinstance(raw_operations, list) else None,
        "changed_operation_count": sum(bool(row["changed"]) for row in operation_audits),
        "inserted_field_count": sum(len(row["inserted_fields"]) for row in operation_audits),
        "self_alias_removed_count": sum(
            int(row["self_alias_removed_count"]) for row in operation_audits
        ),
        "same_owner_duplicate_removed_count": sum(
            int(row["same_owner_duplicate_removed_count"]) for row in operation_audits
        ),
        "operations": operation_audits,
    }
    return normalized, audit


def post_extraction_review_response_issues(
    response: Any,
    context: Mapping[str, Any],
) -> list[str]:
    """Return one concise schema issue for the generic remote-agent repair loop."""

    try:
        diagnostics = context.get("diagnostics") if isinstance(context, Mapping) else None
        evidence = (
            context.get("sanitized_evidence_catalog") if isinstance(context, Mapping) else None
        )
        diagnostic_ids = collect_post_extraction_diagnostic_ids(diagnostics or [])
        diagnostic_targets = collect_post_extraction_diagnostic_targets(diagnostics or [])
        evidence_ids = [
            str(row.get("evidence_id"))
            for row in (evidence or [])
            if isinstance(row, Mapping) and row.get("evidence_id")
        ]
        validate_post_extraction_review_response(
            response,
            current_specs=context.get("current_contracts") or [],
            available_diagnostic_ids=diagnostic_ids,
            available_diagnostic_targets=diagnostic_targets,
            available_evidence_ids=evidence_ids,
            available_evidence_catalog=evidence or [],
            max_operations=context.get("max_operations", 8),
        )
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def _post_extraction_grounding_repair_hints(
    issues: Sequence[str],
    *,
    context: Mapping[str, Any] | None,
    failed_response: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Describe valid citation choices after a fresh response grounding failure.

    The hints are derived exclusively from the sanitized catalog already present
    in the remote prompt.  They do not relax validation or invent a citation;
    they make the deterministic name-anchor rule actionable on the next repair
    turn.  Standalone/cached response validation remains unchanged.
    """

    if not any("cites evidence unrelated" in str(issue) for issue in issues):
        return []
    if not isinstance(context, Mapping) or not isinstance(failed_response, Mapping):
        return []
    evidence = context.get("sanitized_evidence_catalog")
    operations = failed_response.get("operations")
    if not isinstance(evidence, list) or not isinstance(operations, list):
        return []

    catalog = [row for row in evidence if isinstance(row, Mapping)]
    hints: list[dict[str, Any]] = []
    for index, raw_operation in enumerate(operations):
        if not isinstance(raw_operation, Mapping):
            continue
        action = str(raw_operation.get("action") or "").strip().lower()
        contract = raw_operation.get("contract")
        if action not in {"merge", "replace", "revise"} or not isinstance(contract, Mapping):
            continue
        try:
            canonical_contract = CandidateContract(contract).extraction_spec
        except (TypeError, ValueError):
            continue

        eligible_ids: list[str] = []
        cited_grounding: list[dict[str, Any]] = []
        grounding_by_id: dict[str, Any] = {}
        for evidence_row in catalog:
            evidence_id = str(evidence_row.get("evidence_id") or "").strip()
            if not evidence_id:
                continue
            grounding = ground_evidence_to_extraction_contract(
                evidence_row,
                canonical_contract,
            )
            grounding_by_id[evidence_id] = grounding
            if grounding.supported:
                eligible_ids.append(evidence_id)

        cited_ids = raw_operation.get("supporting_evidence_ids")
        if isinstance(cited_ids, list):
            for raw_id in cited_ids:
                evidence_id = str(raw_id).strip()
                grounding = grounding_by_id.get(evidence_id)
                if grounding is None or grounding.supported:
                    continue
                cited_grounding.append(
                    {
                        "evidence_id": evidence_id,
                        "required_name_anchors": list(grounding.required_name_anchors),
                        "matched_evidence_anchors": list(grounding.matched_evidence_anchors),
                    }
                )
        if not cited_grounding:
            continue
        targets = raw_operation.get("target_names")
        hints.append(
            {
                "operation_index": index,
                "action": action,
                "target_names": (
                    [str(value) for value in targets] if isinstance(targets, list) else []
                ),
                "proposed_contract_name": str(canonical_contract["name"]),
                "failed_citations": cited_grounding,
                "eligible_evidence_ids_for_exact_proposed_name": sorted(eligible_ids),
                "fallback_if_no_eligible_evidence": (
                    "replace this contract-changing operation with one "
                    "diagnostic-grounded drop per unsafe target, within the operation budget"
                ),
            }
        )
    return hints


def build_post_extraction_review_repair_prompt(
    issues: Sequence[str],
    *,
    context: Mapping[str, Any] | None = None,
    failed_response: Mapping[str, Any] | None = None,
) -> str:
    grounding_hints = _post_extraction_grounding_repair_hints(
        issues,
        context=context,
        failed_response=failed_response,
    )
    grounding_instruction = ""
    if grounding_hints:
        grounding_instruction = (
            " Deterministic citation-grounding repair hints derived only from the "
            "sanitized evidence catalog: "
            + json.dumps(grounding_hints, separators=(",", ":"), ensure_ascii=False)
            + ". Do not repeat any failed contract/evidence pairing. Cite only an "
            "eligible evidence ID listed for that exact proposed name. If its eligible "
            "list is empty, do not rename the concept to evade grounding: replace that "
            "operation with one diagnostic-grounded drop per unsafe target, up to the "
            "operation budget. While required safety remediation remains, do not stop."
        )
    return (
        "Repair the post-extraction review response as exactly one JSON object with "
        "an operations array. Every operation "
        "must contain exactly these six keys: action, target_names, contract, "
        "supporting_diagnostic_ids, supporting_evidence_ids, reason. For drop, use "
        "contract:null and supporting_evidence_ids:[] when no source evidence is cited. "
        "For stop, use target_names:[], contract:null, "
        "supporting_diagnostic_ids:[], and supporting_evidence_ids:[]. For re_role, "
        "include supporting_evidence_ids:[] when empty. Do not omit neutral fields or "
        "add any other operation fields. Use only supplied "
        "contract names, diagnostic IDs, and evidence IDs; for merge, replace, or "
        "revise, cite evidence with one concept entry containing all normalized exact "
        "lexical identity anchors in the proposed contract name. Short acronyms count "
        "only when that exact token occurs; never infer synonyms or acronym expansions. "
        "Preserve the closed fields and "
        "operation limits. Value aliases must be genuine variants, never repeat their "
        "canonical category, and must be globally unique after case/spacing "
        "normalization."
        + grounding_instruction
        + " Problems: "
        + "; ".join(str(issue) for issue in issues)
    )


def apply_post_extraction_review_operations(
    current_specs: Sequence[Mapping[str, Any]],
    response: ValidatedReviewResponse,
    *,
    max_contracts: int = 64,
) -> AppliedReviewOperations:
    """Apply one validated atomic revision and report selective extraction work."""

    if not isinstance(response, ValidatedReviewResponse):
        raise TypeError("response must be a ValidatedReviewResponse")
    before = [CandidateContract(spec).extraction_spec for spec in current_specs]
    if response.stops:
        return AppliedReviewOperations(
            specs=tuple(_detached(before)),
            reextract_specs=(),
            removed_names=(),
            added_names=(),
            extraction_changed_names=(),
            role_only_changed_names=(),
            operation_audit=tuple(operation.audit_dict() for operation in response.operations),
        )

    replacement_by_first_position: dict[int, dict[str, Any] | None] = {}
    removed: set[str] = set()
    positions = {str(spec["name"]): index for index, spec in enumerate(before)}
    for operation in response.operations:
        target_positions = [positions[name] for name in operation.target_names]
        first = min(target_positions)
        removed.update(operation.target_names)
        replacement_by_first_position[first] = (
            None if operation.contract is None else _detached(operation.contract)
        )

    after: list[dict[str, Any]] = []
    for index, spec in enumerate(before):
        if index in replacement_by_first_position:
            replacement = replacement_by_first_position[index]
            if replacement is not None:
                after.append(replacement)
        if str(spec["name"]) not in removed:
            after.append(_detached(spec))
    if not after:
        raise ValueError("post-extraction review cannot remove every contract")
    if len(after) > int(max_contracts):
        raise ValueError("post-extraction review exceeds max_contracts")
    names = [str(spec["name"]) for spec in after]
    if len(names) != len(set(names)):
        raise ValueError("post-extraction review produced duplicate contract names")

    before_by_name = {str(spec["name"]): spec for spec in before}
    after_by_name = {str(spec["name"]): spec for spec in after}
    removed_names = tuple(name for name in before_by_name if name not in after_by_name)
    added_names = tuple(name for name in after_by_name if name not in before_by_name)
    extraction_changed: list[str] = []
    role_only: list[str] = []
    for name, spec in after_by_name.items():
        prior = before_by_name.get(name)
        if prior is None or extraction_semantics_sha256(prior) != extraction_semantics_sha256(spec):
            extraction_changed.append(name)
        elif tuple(prior.get("roles") or ()) != tuple(spec.get("roles") or ()):
            role_only.append(name)
    reextract_specs = tuple(_detached(after_by_name[name]) for name in extraction_changed)
    return AppliedReviewOperations(
        specs=tuple(_detached(after)),
        reextract_specs=reextract_specs,
        removed_names=removed_names,
        added_names=added_names,
        extraction_changed_names=tuple(extraction_changed),
        role_only_changed_names=tuple(role_only),
        operation_audit=tuple(operation.audit_dict() for operation in response.operations),
    )


def _missing_mask(frame: pd.DataFrame, spec: Mapping[str, Any]) -> pd.Series:
    value_column, missing_column = expected_extraction_columns(spec)
    missing = {value_column, missing_column} - set(frame.columns)
    if missing:
        raise ValueError(f"extracted frame is missing columns: {sorted(missing)}")
    declared = frame[missing_column].fillna(True).astype(bool)
    if str(spec["type"]) == "continuous":
        numeric = pd.to_numeric(frame[value_column], errors="coerce")
        return declared | numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float))
    return declared | frame[value_column].isna()


def _continuous_outlier_rate(
    values: np.ndarray,
    *,
    policy: ExtractionQualityPolicy,
) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < int(policy.continuous_outlier_minimum_rows):
        return 0.0
    q1, q3 = np.quantile(finite, [0.25, 0.75])
    iqr = float(q3 - q1)
    if iqr <= 0.0:
        return 0.0
    lower = q1 - float(policy.continuous_outlier_iqr_multiplier) * iqr
    upper = q3 + float(policy.continuous_outlier_iqr_multiplier) * iqr
    return float(np.mean((finite < lower) | (finite > upper)))


def _fold_stability(
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
    missing: pd.Series,
    fold_ids: np.ndarray,
    *,
    policy: ExtractionQualityPolicy,
) -> dict[str, Any]:
    value_column, _missing_column = expected_extraction_columns(spec)
    unique_folds = list(dict.fromkeys(fold_ids.tolist()))
    coverage_by_fold: list[dict[str, Any]] = []
    distribution_vectors: list[np.ndarray] = []
    categories = [str(value) for value in spec.get("categories") or []]
    for fold_id in unique_folds:
        mask = fold_ids == fold_id
        fold_missing = missing.to_numpy(dtype=bool)[mask]
        coverage = float(1.0 - np.mean(fold_missing)) if np.any(mask) else 0.0
        coverage_by_fold.append(
            {"fold_id": str(fold_id), "n_rows": int(np.sum(mask)), "coverage": coverage}
        )
        observed_mask = mask & ~missing.to_numpy(dtype=bool)
        if str(spec["type"]) == "categorical":
            values = frame.loc[observed_mask, value_column].fillna("").astype(str)
            counts = values.value_counts(normalize=True)
            distribution_vectors.append(
                np.asarray([float(counts.get(category, 0.0)) for category in categories])
            )
        else:
            values = pd.to_numeric(frame.loc[observed_mask, value_column], errors="coerce")
            finite = values.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            if len(finite):
                distribution_vectors.append(
                    np.asarray(
                        [
                            float(np.median(finite)),
                            float(np.quantile(finite, 0.25)),
                            float(np.quantile(finite, 0.75)),
                        ]
                    )
                )
            else:
                distribution_vectors.append(np.asarray([np.nan, np.nan, np.nan]))
    coverages = np.asarray([row["coverage"] for row in coverage_by_fold], dtype=float)
    distribution_shift = None
    if len(distribution_vectors) >= 2 and str(spec["type"]) == "categorical":
        distribution_shift = float(
            max(
                np.sum(np.abs(left - right)) / 2.0
                for left, right in combinations(distribution_vectors, 2)
            )
        )
    elif len(distribution_vectors) >= 2:
        medians = np.asarray([value[0] for value in distribution_vectors], dtype=float)
        finite = medians[np.isfinite(medians)]
        if len(finite) >= 2:
            pooled = pd.to_numeric(frame.loc[~missing, value_column], errors="coerce").to_numpy(
                dtype=float
            )
            pooled = pooled[np.isfinite(pooled)]
            scale = float(np.subtract(*np.quantile(pooled, [0.75, 0.25]))) if len(pooled) else 0.0
            distribution_shift = float(
                (np.max(finite) - np.min(finite))
                / max(scale, float(policy.fold_continuous_scale_epsilon))
            )
    return {
        "coverage_by_fold": coverage_by_fold,
        "coverage_range": float(np.max(coverages) - np.min(coverages)),
        "coverage_std": float(np.std(coverages)),
        "distribution_shift": _finite_or_none(distribution_shift),
    }


def build_extraction_quality_diagnostics(
    frame: pd.DataFrame,
    specs: Sequence[Mapping[str, Any]],
    *,
    fold_ids: Sequence[Any],
    policy: ExtractionQualityPolicy | None = None,
    minimum_coverage: float = 0.05,
    maximum_unknown_category_rate: float = 0.05,
) -> dict[str, Any]:
    """Summarize coverage, missingness, plausibility, timing, and stability.

    The function consumes extracted values and fold identifiers only.  It does
    not accept treatment, outcome, patient text, row identifiers, or an oracle
    target, which keeps these diagnostics safe to include in an agent prompt.
    """

    if len(frame) == 0:
        raise ValueError("quality diagnostics require at least one row")
    if policy is None:
        # Compatibility-only behavior. Typed portable production supplies the
        # complete policy and never inherits these legacy values.
        policy = ExtractionQualityPolicy(
            minimum_coverage=float(minimum_coverage),
            maximum_unknown_category_rate=float(
                maximum_unknown_category_rate
            ),
            continuous_outlier_minimum_rows=8,
            continuous_outlier_iqr_multiplier=6.0,
            continuous_outlier_warning_rate=0.10,
            fold_coverage_range_warning=0.35,
            fold_continuous_scale_epsilon=1e-8,
        )
    elif not isinstance(policy, ExtractionQualityPolicy):
        raise TypeError("policy must be ExtractionQualityPolicy")
    folds = np.asarray(list(fold_ids), dtype=object)
    if folds.ndim != 1 or len(folds) != len(frame) or len(set(folds.tolist())) < 2:
        raise ValueError("fold_ids must assign every row to at least two folds")
    canonical_specs = [CandidateContract(spec).extraction_spec for spec in specs]
    rows: list[dict[str, Any]] = []
    for position, spec in enumerate(canonical_specs, start=1):
        name = str(spec["name"])
        value_column, _missing_column = expected_extraction_columns(spec)
        missing = _missing_mask(frame, spec)
        observed = frame.loc[~missing, value_column]
        coverage = float(1.0 - missing.mean())
        unique_observed = int(observed.nunique(dropna=True))
        plausibility: dict[str, Any]
        hard_failures: list[str] = []
        warnings: list[str] = []
        if coverage < float(policy.minimum_coverage):
            hard_failures.append("coverage_below_minimum")
        if unique_observed <= 1:
            hard_failures.append("constant_or_unobserved")
        if str(spec["type"]) == "categorical":
            categories = [str(value) for value in spec.get("categories") or []]
            observed_text = observed.fillna("").astype(str)
            unknown_rate = float(np.mean(~observed_text.isin(categories))) if len(observed) else 0.0
            category_counts = {
                category: int(np.sum(observed_text == category)) for category in categories
            }
            if unknown_rate > float(policy.maximum_unknown_category_rate):
                hard_failures.append("out_of_contract_category_values")
            empty_declared = [category for category, count in category_counts.items() if count == 0]
            if empty_declared:
                warnings.append("one_or_more_declared_categories_unobserved")
            plausibility = {
                "unknown_category_rate": unknown_rate,
                "declared_category_counts": category_counts,
                "unobserved_declared_categories": empty_declared,
            }
        else:
            numeric_all = pd.to_numeric(frame[value_column], errors="coerce")
            declared_observed = (
                ~frame[expected_extraction_columns(spec)[1]].fillna(True).astype(bool)
            )
            parse_failure_rate = (
                float(np.mean(numeric_all[declared_observed].isna()))
                if declared_observed.any()
                else 0.0
            )
            finite = numeric_all[~missing].to_numpy(dtype=float)
            outlier_rate = _continuous_outlier_rate(
                finite,
                policy=policy,
            )
            if parse_failure_rate > 0.0:
                hard_failures.append("declared_numeric_value_not_parseable")
            if outlier_rate > float(policy.continuous_outlier_warning_rate):
                warnings.append("high_robust_outlier_rate")
            plausibility = {
                "numeric_parse_failure_rate": parse_failure_rate,
                "robust_outlier_rate": outlier_rate,
                "observed_min": _finite_or_none(np.min(finite)) if len(finite) else None,
                "observed_median": _finite_or_none(np.median(finite)) if len(finite) else None,
                "observed_max": _finite_or_none(np.max(finite)) if len(finite) else None,
            }
        stability = _fold_stability(
            frame,
            spec,
            missing,
            folds,
            policy=policy,
        )
        if (
            stability["coverage_range"]
            > float(policy.fold_coverage_range_warning)
        ):
            warnings.append("coverage_unstable_across_inner_folds")
        row = {
            "diagnostic_id": f"diagnostic_{position:04d}",
            "kind": "feature_quality",
            "feature_name": name,
            "roles": list(spec.get("roles") or []),
            "type": spec["type"],
            "coverage": coverage,
            "missingness": float(missing.mean()),
            "n_unique_observed": unique_observed,
            "value_plausibility": plausibility,
            "source_text_temporal_policy": source_text_temporal_policy_audit(),
            "inner_fold_stability": stability,
            "hard_failures": hard_failures,
            "warnings": warnings,
            "passed": not hard_failures,
        }
        rows.append(row)
    return {
        "schema_version": POST_EXTRACTION_QUALITY_SCHEMA_VERSION,
        "row_count": int(len(frame)),
        "inner_fold_count": int(len(set(folds.tolist()))),
        "source_text_temporal_policy": source_text_temporal_policy_audit(),
        "features": rows,
        "summary": {
            "feature_count": len(rows),
            "passed_count": sum(bool(row["passed"]) for row in rows),
            "failed_count": sum(not bool(row["passed"]) for row in rows),
            "warning_count": sum(len(row["warnings"]) for row in rows),
        },
    }


def _cramers_v(left: pd.Series, right: pd.Series) -> float | None:
    table = pd.crosstab(left, right, dropna=False).to_numpy(dtype=float)
    total = float(table.sum())
    if total <= 0.0 or min(table.shape) <= 1:
        return None
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    mask = expected > 0.0
    chi2 = float(np.sum(np.square(table[mask] - expected[mask]) / expected[mask]))
    denominator = total * float(min(table.shape[0] - 1, table.shape[1] - 1))
    return _finite_or_none(math.sqrt(max(chi2, 0.0) / denominator))


def build_redundancy_diagnostics(
    frame: pd.DataFrame,
    specs: Sequence[Mapping[str, Any]],
    *,
    policy: ExtractionRedundancyPolicy | None = None,
    association_threshold: float = 0.80,
    missingness_agreement_threshold: float = 0.90,
    diagnostic_start: int = 1,
) -> list[dict[str, Any]]:
    """Return pairs with high value association or informative shared missingness.

    Missingness redundancy uses Jaccard overlap conditional on at least one
    missing value. Raw observed/missing agreement is reported for auditing but
    cannot by itself make two fully observed variables redundant.
    """

    if policy is None:
        # Compatibility-only behavior. Typed portable production supplies the
        # closed policy.
        policy = ExtractionRedundancyPolicy(
            association_threshold=float(association_threshold),
            missingness_jaccard_threshold=float(
                missingness_agreement_threshold
            ),
            minimum_pairwise_complete_rows=3,
        )
    elif not isinstance(policy, ExtractionRedundancyPolicy):
        raise TypeError("policy must be ExtractionRedundancyPolicy")
    canonical_specs = [CandidateContract(spec).extraction_spec for spec in specs]
    rows: list[dict[str, Any]] = []
    next_id = int(diagnostic_start)
    for left, right in combinations(canonical_specs, 2):
        left_value, _ = expected_extraction_columns(left)
        right_value, _ = expected_extraction_columns(right)
        left_missing = _missing_mask(frame, left).to_numpy(dtype=bool)
        right_missing = _missing_mask(frame, right).to_numpy(dtype=bool)
        missing_agreement = float(np.mean(left_missing == right_missing))
        missing_union = int(np.sum(left_missing | right_missing))
        missing_intersection = int(np.sum(left_missing & right_missing))
        # Raw agreement is 1.0 for any two fully observed variables and is
        # therefore not evidence of redundancy.  Jaccard similarity conditions
        # on at least one value being missing and retains the useful signal from
        # genuinely shared extraction failures.
        missing_jaccard = (
            None if missing_union == 0 else float(missing_intersection / missing_union)
        )
        complete = ~left_missing & ~right_missing
        association = None
        association_kind = None
        if (
            int(np.sum(complete))
            >= int(policy.minimum_pairwise_complete_rows)
        ):
            if left["type"] == "continuous" and right["type"] == "continuous":
                x = pd.to_numeric(frame.loc[complete, left_value], errors="coerce").to_numpy(
                    dtype=float
                )
                y = pd.to_numeric(frame.loc[complete, right_value], errors="coerce").to_numpy(
                    dtype=float
                )
                if np.std(x) > 0.0 and np.std(y) > 0.0:
                    association = _finite_or_none(abs(np.corrcoef(x, y)[0, 1]))
                    association_kind = "absolute_pearson"
            elif left["type"] == "categorical" and right["type"] == "categorical":
                association = _cramers_v(
                    frame.loc[complete, left_value].astype(str),
                    frame.loc[complete, right_value].astype(str),
                )
                association_kind = "cramers_v"
        value_redundant = bool(
            association is not None
            and association >= float(policy.association_threshold)
        )
        missingness_redundant = bool(
            missing_jaccard is not None
            and missing_jaccard
            >= float(policy.missingness_jaccard_threshold)
        )
        if not value_redundant and not missingness_redundant:
            continue
        rows.append(
            {
                "diagnostic_id": f"diagnostic_{next_id:04d}",
                "kind": "redundancy",
                "feature_names": [left["name"], right["name"]],
                "association_kind": association_kind,
                "association": association,
                "missingness_agreement": missing_agreement,
                "missingness_jaccard": missing_jaccard,
                "missingness_pattern_informative": missing_union > 0,
                "redundancy_reasons": [
                    reason
                    for reason, active in (
                        ("high_value_association", value_redundant),
                        ("shared_missingness_pattern", missingness_redundant),
                    )
                    if active
                ],
                "pairwise_complete_count": int(np.sum(complete)),
            }
        )
        next_id += 1
    return rows


# ---------------------------------------------------------------------------
# Observable-only causal diagnostics and the untouched sequential gate.
# ---------------------------------------------------------------------------


def _review_key(value: Any, *, field_name: str) -> Hashable:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        raise ValueError(f"{field_name} cannot contain missing values")
    try:
        is_missing = bool(value != value)
    except (TypeError, ValueError):
        is_missing = False
    if is_missing:
        raise ValueError(f"{field_name} cannot contain missing values")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} values must be hashable") from exc
    return value


def _review_keys(values: Sequence[Any], *, field_name: str) -> tuple[Hashable, ...]:
    result = tuple(_review_key(value, field_name=field_name) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must be unique")
    return result


def _finite_vector(values: Sequence[float], *, field_name: str, length: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) != int(length):
        raise ValueError(f"{field_name} must be one-dimensional with length {length}")
    if not np.isfinite(result).all():
        raise ValueError(f"{field_name} must contain only finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ObservableCausalRows:
    """Exact observed rows used either for spent diagnostics or one hidden gate.

    ``inner_fold_ids`` are required for model fitting and pre-proposal
    diagnostics.  They may be omitted for a gate because gate rows are never
    used to fit a nuisance or effect model.
    """

    row_ids: tuple[Hashable, ...]
    extracted: pd.DataFrame = field(repr=False)
    treatment: np.ndarray = field(repr=False)
    outcome: np.ndarray = field(repr=False)
    inner_fold_ids: tuple[Hashable, ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        row_ids = _review_keys(self.row_ids, field_name="row_ids")
        if not isinstance(self.extracted, pd.DataFrame):
            raise TypeError("extracted must be a pandas DataFrame")
        if len(self.extracted) != len(row_ids):
            raise ValueError("extracted must have one row per row_id")
        treatment = _finite_vector(
            self.treatment,
            field_name="treatment",
            length=len(row_ids),
        )
        if not set(np.unique(treatment)).issubset({0.0, 1.0}):
            raise ValueError("treatment must be binary")
        outcome = _finite_vector(
            self.outcome,
            field_name="outcome",
            length=len(row_ids),
        )
        folds = self.inner_fold_ids
        if folds is not None:
            folds = tuple(_review_key(value, field_name="inner_fold_ids") for value in folds)
            if len(folds) != len(row_ids):
                raise ValueError("inner_fold_ids must have one value per row")
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "extracted", self.extracted.reset_index(drop=True).copy())
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "inner_fold_ids", folds)


def _validated_context_upstream_bank(
    *,
    context_row_ids: Sequence[Hashable],
    context_inner_fold_ids: Sequence[Hashable],
    context_values: Any,
    context_fit_row_provenance: Sequence[FitRowProvenance | Sequence[FitRowProvenance]],
    column_count: int,
    gate_row_ids: Sequence[Hashable],
    bank_name: str,
) -> tuple[
    tuple[Hashable, ...],
    tuple[Hashable, ...],
    np.ndarray,
    tuple[tuple[FitRowProvenance, ...], ...],
]:
    """Validate the optional context half of one gate-local upstream bank."""

    raw_rows = tuple(context_row_ids)
    raw_folds = tuple(context_inner_fold_ids)
    raw_lineage = tuple(context_fit_row_provenance)
    matrix = np.asarray(context_values, dtype=float)
    absent = not raw_rows and not raw_folds and not raw_lineage and matrix.size == 0
    if absent:
        empty = np.empty((0, int(column_count)), dtype=float)
        empty.setflags(write=False)
        return (), (), empty, ()
    rows = _review_keys(raw_rows, field_name=f"{bank_name}.context_row_ids")
    if frozenset(rows) & frozenset(gate_row_ids):
        raise ValueError(f"{bank_name} context and gate row IDs must be disjoint")
    folds = tuple(
        _review_key(value, field_name=f"{bank_name}.context_inner_fold_ids") for value in raw_folds
    )
    if len(folds) != len(rows) or len(set(folds)) < 2:
        raise ValueError(
            f"{bank_name}.context_inner_fold_ids must define at least two aligned folds"
        )
    if matrix.ndim != 2 or matrix.shape != (len(rows), int(column_count)):
        raise ValueError(
            f"{bank_name}.context_values must have shape {(len(rows), int(column_count))}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{bank_name}.context_values must be finite")
    if len(raw_lineage) != int(column_count):
        raise ValueError(
            f"{bank_name}.context_fit_row_provenance must contain one entry per column"
        )
    normalized: list[tuple[FitRowProvenance, ...]] = []
    gate_set = frozenset(gate_row_ids)
    for item in raw_lineage:
        if isinstance(item, FitRowProvenance):
            per_row = (item,) * len(rows)
        else:
            per_row = tuple(item)
        if len(per_row) != len(rows) or not all(
            isinstance(value, FitRowProvenance) for value in per_row
        ):
            raise TypeError(
                f"each {bank_name} context provenance entry must align one lineage per row"
            )
        for row_id, lineage in zip(rows, per_row):
            fitted = lineage.recursive_fit_row_ids()
            if row_id in fitted:
                raise ValueError(
                    f"{bank_name} context upstream value is not out-of-fold for its row"
                )
            if fitted & gate_set:
                raise ValueError(
                    f"{bank_name} context upstream lineage includes an untouched gate row"
                )
        normalized.append(per_row)
    frozen = matrix.copy()
    frozen.setflags(write=False)
    return rows, folds, frozen, tuple(normalized)


def _aligned_conditional_upstream_values(
    *,
    context_row_ids: Sequence[Hashable],
    context_inner_fold_ids: Sequence[Hashable],
    context_values: np.ndarray,
    context_fit_row_provenance: Sequence[Sequence[FitRowProvenance]],
    gate_row_ids: Sequence[Hashable],
    gate_values: np.ndarray,
    gate_fit_row_provenance: Sequence[Sequence[FitRowProvenance]],
    exact_context_row_ids: Sequence[Hashable],
    exact_context_inner_fold_ids: Sequence[Hashable],
    exact_gate_row_ids: Sequence[Hashable],
    bank_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align and prove exact fold-complement/context-fit upstream lineage."""

    requested_context = _review_keys(exact_context_row_ids, field_name="exact_context_row_ids")
    requested_gate = _review_keys(exact_gate_row_ids, field_name="exact_gate_row_ids")
    if not context_row_ids:
        raise ValueError(
            f"{bank_name} bank lacks required cross-fitted context-side upstream values"
        )
    stored_context = tuple(context_row_ids)
    stored_gate = tuple(gate_row_ids)
    if frozenset(requested_context) != frozenset(stored_context):
        raise ValueError(f"{bank_name} context row IDs must exactly equal the spent context")
    if frozenset(requested_gate) != frozenset(stored_gate):
        raise ValueError(f"{bank_name} gate row IDs must exactly equal the untouched gate")
    requested_folds = tuple(
        _review_key(value, field_name="exact_context_inner_fold_ids")
        for value in exact_context_inner_fold_ids
    )
    if len(requested_folds) != len(requested_context):
        raise ValueError("exact_context_inner_fold_ids must align with exact_context_row_ids")
    fold_by_row = dict(zip(requested_context, requested_folds))
    if tuple(fold_by_row[row_id] for row_id in stored_context) != tuple(context_inner_fold_ids):
        raise ValueError(f"{bank_name} context inner-fold assignments changed")

    context_set = frozenset(requested_context)
    gate_set = frozenset(requested_gate)
    rows_by_fold: dict[Hashable, frozenset[Hashable]] = {}
    for fold_id in set(requested_folds):
        rows_by_fold[fold_id] = frozenset(
            row_id for row_id, value in fold_by_row.items() if value == fold_id
        )
    for per_column in context_fit_row_provenance:
        for row_id, lineage in zip(stored_context, per_column):
            expected = context_set - rows_by_fold[fold_by_row[row_id]]
            if lineage.recursive_fit_row_ids() != expected:
                raise ValueError(
                    f"{bank_name} context lineage must equal the exact complementary inner fold"
                )
    for per_column in gate_fit_row_provenance:
        for lineage in per_column:
            fitted = lineage.recursive_fit_row_ids()
            if fitted != context_set or fitted & gate_set:
                raise ValueError(
                    f"{bank_name} gate lineage must equal the exact complete spent context"
                )

    context_positions = {row_id: position for position, row_id in enumerate(stored_context)}
    gate_positions = {row_id: position for position, row_id in enumerate(stored_gate)}
    aligned_context = np.asarray(
        [context_values[context_positions[row_id]] for row_id in requested_context],
        dtype=float,
    )
    aligned_gate = np.asarray(
        [gate_values[gate_positions[row_id]] for row_id in requested_gate], dtype=float
    )
    return aligned_context, aligned_gate


@dataclass(frozen=True)
class GateSourceSignalView:
    """Authenticated upstream effect signals for a conditional gate fit.

    Provenance is source-major.  Each source entry can be one lineage shared by
    all rows or one lineage per row.  A view is rejected unless *every*
    recursive lineage is disjoint from the complete gate, which is stronger
    than merely excluding the prediction's own row.

    ``context_values`` are cross-fitted on the already-spent context and
    ``values`` are predictions from a fit on that complete context for the
    untouched gate.  The calibrated signals may therefore be used as fixed
    effect-model covariates, just as they are used by the final head.  They are
    never treated as observed effects or oracle targets.
    """

    row_ids: tuple[Hashable, ...]
    source_names: tuple[str, ...]
    source_kinds: tuple[str, ...]
    values: np.ndarray = field(repr=False)
    fit_row_provenance: tuple[FitRowProvenance | tuple[FitRowProvenance, ...], ...] = field(
        repr=False
    )
    context_row_ids: tuple[Hashable, ...] = ()
    context_inner_fold_ids: tuple[Hashable, ...] = field(default=(), repr=False)
    context_values: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float), repr=False
    )
    context_fit_row_provenance: tuple[FitRowProvenance | tuple[FitRowProvenance, ...], ...] = field(
        default=(), repr=False
    )

    def __post_init__(self) -> None:
        row_ids = _review_keys(self.row_ids, field_name="row_ids")
        names = tuple(str(value).strip() for value in self.source_names)
        kinds = tuple(str(value).strip() for value in self.source_kinds)
        if not names or len(names) != len(set(names)) or any(not value for value in names):
            raise ValueError("source_names must be non-empty and unique")
        if len(kinds) != len(names) or any(not value for value in kinds):
            raise ValueError("source_kinds must contain one non-empty kind per source")
        if set(kinds) & RAW_UNCALIBRATED_FEATURE_SOURCE_KINDS:
            raise ValueError(
                "uncalibrated feature bases must use GateFeatureBankView, not the "
                "treatment-effect source view"
            )
        if any(_FORBIDDEN.search(value) for value in (*names, *kinds)):
            raise ValueError("source metadata contains a forbidden benchmark field name")
        matrix = np.asarray(self.values, dtype=float)
        if matrix.ndim != 2 or matrix.shape != (len(row_ids), len(names)):
            raise ValueError("values must have shape (number of gate rows, number of sources)")
        if not np.isfinite(matrix).all():
            raise ValueError("source values must be finite")
        raw_lineage = tuple(self.fit_row_provenance)
        if len(raw_lineage) != len(names):
            raise ValueError("fit_row_provenance must contain one entry per source")
        normalized_lineage: list[tuple[FitRowProvenance, ...]] = []
        gate_set = frozenset(row_ids)
        for source_name, item in zip(names, raw_lineage):
            if isinstance(item, FitRowProvenance):
                per_row = (item,) * len(row_ids)
            else:
                per_row = tuple(item)
            if len(per_row) != len(row_ids) or not all(
                isinstance(value, FitRowProvenance) for value in per_row
            ):
                raise TypeError(
                    "each source provenance entry must be FitRowProvenance or one "
                    "FitRowProvenance per gate row"
                )
            for lineage in per_row:
                overlap = lineage.recursive_fit_row_ids() & gate_set
                if overlap:
                    raise ValueError(
                        "untouched-gate provenance violation for source "
                        f"{source_name!r}: lineage includes gate rows"
                    )
            normalized_lineage.append(per_row)
        context_rows, context_folds, context_matrix, context_lineage = (
            _validated_context_upstream_bank(
                context_row_ids=self.context_row_ids,
                context_inner_fold_ids=self.context_inner_fold_ids,
                context_values=self.context_values,
                context_fit_row_provenance=self.context_fit_row_provenance,
                column_count=len(names),
                gate_row_ids=row_ids,
                bank_name="source",
            )
        )
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "source_names", names)
        object.__setattr__(self, "source_kinds", kinds)
        object.__setattr__(self, "values", matrix)
        object.__setattr__(self, "fit_row_provenance", tuple(normalized_lineage))
        object.__setattr__(self, "context_row_ids", context_rows)
        object.__setattr__(self, "context_inner_fold_ids", context_folds)
        object.__setattr__(self, "context_values", context_matrix)
        object.__setattr__(self, "context_fit_row_provenance", context_lineage)

    def aligned_values(self, exact_row_ids: Sequence[Hashable]) -> np.ndarray:
        """Align values while requiring the caller to name the complete gate."""

        requested = _review_keys(exact_row_ids, field_name="exact_row_ids")
        if frozenset(requested) != frozenset(self.row_ids):
            raise ValueError("source view row IDs must exactly equal the untouched gate")
        positions = {row_id: position for position, row_id in enumerate(self.row_ids)}
        return np.asarray([self.values[positions[row_id]] for row_id in requested], dtype=float)

    def aligned_conditional_values(
        self,
        *,
        exact_context_row_ids: Sequence[Hashable],
        exact_context_inner_fold_ids: Sequence[Hashable],
        exact_gate_row_ids: Sequence[Hashable],
    ) -> tuple[np.ndarray, np.ndarray]:
        return _aligned_conditional_upstream_values(
            context_row_ids=self.context_row_ids,
            context_inner_fold_ids=self.context_inner_fold_ids,
            context_values=self.context_values,
            context_fit_row_provenance=self.context_fit_row_provenance,
            gate_row_ids=self.row_ids,
            gate_values=self.values,
            gate_fit_row_provenance=self.fit_row_provenance,
            exact_context_row_ids=exact_context_row_ids,
            exact_context_inner_fold_ids=exact_context_inner_fold_ids,
            exact_gate_row_ids=exact_gate_row_ids,
            bank_name="source",
        )


@dataclass(frozen=True)
class GateFeatureBankView:
    """Role-aware raw feature bases for a conditional untouched-gate fit.

    Unlike :class:`GateSourceSignalView`, these columns are not calibrated
    treatment effects.  Propensity, outcome, and modifier columns are compared
    only with the matching fitted prediction.  Their raw scales are never
    interpreted as treatment effects.  Cross-fitted context values can enter
    only their declared nuisance/effect design, while complete-context fits
    produce the corresponding untouched-gate values.
    """

    row_ids: tuple[Hashable, ...]
    feature_names: tuple[str, ...]
    source_kinds: tuple[str, ...]
    consumer_roles: tuple[str, ...]
    values: np.ndarray = field(repr=False)
    fit_row_provenance: tuple[FitRowProvenance | tuple[FitRowProvenance, ...], ...] = field(
        repr=False
    )
    context_row_ids: tuple[Hashable, ...] = ()
    context_inner_fold_ids: tuple[Hashable, ...] = field(default=(), repr=False)
    context_values: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float), repr=False
    )
    context_fit_row_provenance: tuple[FitRowProvenance | tuple[FitRowProvenance, ...], ...] = field(
        default=(), repr=False
    )

    def __post_init__(self) -> None:
        row_ids = _review_keys(self.row_ids, field_name="row_ids")
        names = tuple(str(value).strip() for value in self.feature_names)
        kinds = tuple(str(value).strip() for value in self.source_kinds)
        roles = tuple(str(value).strip() for value in self.consumer_roles)
        if not names or len(names) != len(set(names)) or any(not value for value in names):
            raise ValueError("feature_names must be non-empty and unique")
        if len(kinds) != len(names) or any(not value for value in kinds):
            raise ValueError("source_kinds must contain one non-empty kind per feature")
        if len(roles) != len(names) or set(roles) - _GATE_FEATURE_BANK_ROLES:
            raise ValueError(
                "consumer_roles must use the role-correct propensity, outcome, or "
                "uncalibrated modifier feature role"
            )
        if any(_FORBIDDEN.search(value) for value in (*names, *kinds, *roles)):
            raise ValueError("feature-bank metadata contains a forbidden benchmark field name")
        matrix = np.asarray(self.values, dtype=float)
        if matrix.ndim != 2 or matrix.shape != (len(row_ids), len(names)):
            raise ValueError(
                "values must have shape (number of gate rows, number of feature columns)"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("feature-bank values must be finite")
        raw_lineage = tuple(self.fit_row_provenance)
        if len(raw_lineage) != len(names):
            raise ValueError("fit_row_provenance must contain one entry per feature column")
        normalized_lineage: list[tuple[FitRowProvenance, ...]] = []
        gate_set = frozenset(row_ids)
        for feature_name, item in zip(names, raw_lineage):
            if isinstance(item, FitRowProvenance):
                per_row = (item,) * len(row_ids)
            else:
                per_row = tuple(item)
            if len(per_row) != len(row_ids) or not all(
                isinstance(value, FitRowProvenance) for value in per_row
            ):
                raise TypeError(
                    "each feature provenance entry must be FitRowProvenance or one "
                    "FitRowProvenance per gate row"
                )
            for lineage in per_row:
                if lineage.recursive_fit_row_ids() & gate_set:
                    raise ValueError(
                        "untouched-gate provenance violation for feature "
                        f"{feature_name!r}: lineage includes gate rows"
                    )
            normalized_lineage.append(per_row)
        context_rows, context_folds, context_matrix, context_lineage = (
            _validated_context_upstream_bank(
                context_row_ids=self.context_row_ids,
                context_inner_fold_ids=self.context_inner_fold_ids,
                context_values=self.context_values,
                context_fit_row_provenance=self.context_fit_row_provenance,
                column_count=len(names),
                gate_row_ids=row_ids,
                bank_name="feature",
            )
        )
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "source_kinds", kinds)
        object.__setattr__(self, "consumer_roles", roles)
        object.__setattr__(self, "values", matrix)
        object.__setattr__(self, "fit_row_provenance", tuple(normalized_lineage))
        object.__setattr__(self, "context_row_ids", context_rows)
        object.__setattr__(self, "context_inner_fold_ids", context_folds)
        object.__setattr__(self, "context_values", context_matrix)
        object.__setattr__(self, "context_fit_row_provenance", context_lineage)

    def aligned_values(self, exact_row_ids: Sequence[Hashable]) -> np.ndarray:
        requested = _review_keys(exact_row_ids, field_name="exact_row_ids")
        if frozenset(requested) != frozenset(self.row_ids):
            raise ValueError("feature-bank row IDs must exactly equal the untouched gate")
        positions = {row_id: position for position, row_id in enumerate(self.row_ids)}
        return np.asarray([self.values[positions[row_id]] for row_id in requested], dtype=float)

    def aligned_conditional_values(
        self,
        *,
        exact_context_row_ids: Sequence[Hashable],
        exact_context_inner_fold_ids: Sequence[Hashable],
        exact_gate_row_ids: Sequence[Hashable],
    ) -> tuple[np.ndarray, np.ndarray]:
        return _aligned_conditional_upstream_values(
            context_row_ids=self.context_row_ids,
            context_inner_fold_ids=self.context_inner_fold_ids,
            context_values=self.context_values,
            context_fit_row_provenance=self.context_fit_row_provenance,
            gate_row_ids=self.row_ids,
            gate_values=self.values,
            gate_fit_row_provenance=self.fit_row_provenance,
            exact_context_row_ids=exact_context_row_ids,
            exact_context_inner_fold_ids=exact_context_inner_fold_ids,
            exact_gate_row_ids=exact_gate_row_ids,
            bank_name="feature",
        )


_CALIBRATED_EFFECT_INPUT_ROLE = "calibrated_effect_modifier"


@dataclass(frozen=True)
class _UpstreamFamilyColumn:
    input_kind: str
    source_kind: str
    consumer_role: str
    feature_name: str

    @property
    def family_key(self) -> tuple[str, str, str]:
        return self.input_kind, self.source_kind, self.consumer_role


@dataclass(frozen=True)
class _ConditionalUpstreamDesign:
    """One standardized immutable context/gate design shared by both registries."""

    context_by_role: Mapping[str, np.ndarray] = field(repr=False)
    gate_by_role: Mapping[str, np.ndarray] = field(repr=False)
    columns_by_role: Mapping[str, tuple[_UpstreamFamilyColumn, ...]]
    content_sha256: str

    def values(self, role: str, *, scope: str) -> np.ndarray:
        source = self.context_by_role if scope == "context" else self.gate_by_role
        return source[role]

    def verify_content(self) -> None:
        if (
            _upstream_design_sha256(
                self.context_by_role,
                self.gate_by_role,
                self.columns_by_role,
            )
            != self.content_sha256
        ):
            raise RuntimeError("immutable conditional upstream design content changed")

    def without_family(self, key: tuple[str, str, str]) -> "_ConditionalUpstreamDesign":
        context: dict[str, np.ndarray] = {}
        gate: dict[str, np.ndarray] = {}
        columns: dict[str, tuple[_UpstreamFamilyColumn, ...]] = {}
        for role in (
            PROPENSITY_NUISANCE_FEATURE_ROLE,
            OUTCOME_NUISANCE_FEATURE_ROLE,
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        ):
            keep = np.asarray(
                [column.family_key != key for column in self.columns_by_role[role]],
                dtype=bool,
            )
            context[role] = _frozen_matrix(self.context_by_role[role][:, keep])
            gate[role] = _frozen_matrix(self.gate_by_role[role][:, keep])
            columns[role] = tuple(
                column
                for column, retained in zip(self.columns_by_role[role], keep)
                if bool(retained)
            )
        return _make_conditional_upstream_design(context, gate, columns)


def _frozen_matrix(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("upstream design matrices must be finite and two-dimensional")
    result.setflags(write=False)
    return result


def _upstream_design_sha256(
    context: Mapping[str, np.ndarray],
    gate: Mapping[str, np.ndarray],
    columns: Mapping[str, tuple[_UpstreamFamilyColumn, ...]],
) -> str:
    digest = hashlib.sha256()
    for role in (
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    ):
        digest.update(role.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            _canonical_json(
                [
                    {
                        "input_kind": value.input_kind,
                        "source_kind": value.source_kind,
                        "consumer_role": value.consumer_role,
                        "feature_name": value.feature_name,
                    }
                    for value in columns[role]
                ]
            ).encode("utf-8")
        )
        for matrix in (context[role], gate[role]):
            canonical = np.ascontiguousarray(np.asarray(matrix, dtype="<f8"))
            digest.update(_canonical_json(list(canonical.shape)).encode("utf-8"))
            digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _make_conditional_upstream_design(
    context: Mapping[str, np.ndarray],
    gate: Mapping[str, np.ndarray],
    columns: Mapping[str, tuple[_UpstreamFamilyColumn, ...]],
) -> _ConditionalUpstreamDesign:
    required = set(_GATE_FEATURE_BANK_ROLES)
    if set(context) != required or set(gate) != required or set(columns) != required:
        raise ValueError("conditional upstream design must define every exact consumer role")
    frozen_context = {role: _frozen_matrix(values) for role, values in context.items()}
    frozen_gate = {role: _frozen_matrix(values) for role, values in gate.items()}
    normalized_columns = {role: tuple(values) for role, values in columns.items()}
    context_rows = {matrix.shape[0] for matrix in frozen_context.values()}
    gate_rows = {matrix.shape[0] for matrix in frozen_gate.values()}
    if len(context_rows) != 1 or len(gate_rows) != 1:
        raise ValueError("conditional upstream role matrices must have aligned row counts")
    for role in required:
        if frozen_context[role].shape[1] != len(normalized_columns[role]) or frozen_gate[
            role
        ].shape[1] != len(normalized_columns[role]):
            raise ValueError("conditional upstream role metadata must align with columns")
    return _ConditionalUpstreamDesign(
        context_by_role=frozen_context,
        gate_by_role=frozen_gate,
        columns_by_role=normalized_columns,
        content_sha256=_upstream_design_sha256(frozen_context, frozen_gate, normalized_columns),
    )


def _standardize_upstream_pair(
    context: np.ndarray,
    gate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if context.shape[1] != gate.shape[1]:
        raise ValueError("context/gate upstream schemas must be identical")
    if context.shape[1] == 0:
        return _frozen_matrix(context), _frozen_matrix(gate)
    means = np.mean(context, axis=0)
    scales = np.std(context, axis=0, ddof=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    return _frozen_matrix((context - means) / scales), _frozen_matrix((gate - means) / scales)


def _build_conditional_upstream_design(
    context: ObservableCausalRows,
    gate: ObservableCausalRows,
    *,
    source_view: GateSourceSignalView | None,
    feature_bank_view: GateFeatureBankView | None,
) -> _ConditionalUpstreamDesign:
    if context.inner_fold_ids is None:
        raise ValueError("context inner_fold_ids are required for conditional upstream fitting")
    context_columns: dict[str, list[np.ndarray]] = {role: [] for role in _GATE_FEATURE_BANK_ROLES}
    gate_columns: dict[str, list[np.ndarray]] = {role: [] for role in _GATE_FEATURE_BANK_ROLES}
    metadata: dict[str, list[_UpstreamFamilyColumn]] = {
        role: [] for role in _GATE_FEATURE_BANK_ROLES
    }

    if source_view is not None:
        source_context, source_gate = source_view.aligned_conditional_values(
            exact_context_row_ids=context.row_ids,
            exact_context_inner_fold_ids=context.inner_fold_ids,
            exact_gate_row_ids=gate.row_ids,
        )
        role = UNCALIBRATED_EFFECT_MODIFIER_ROLE
        for index, (name, kind) in enumerate(
            zip(source_view.source_names, source_view.source_kinds)
        ):
            context_columns[role].append(source_context[:, index])
            gate_columns[role].append(source_gate[:, index])
            metadata[role].append(
                _UpstreamFamilyColumn(
                    input_kind="calibrated_effect_source",
                    source_kind=kind,
                    consumer_role=_CALIBRATED_EFFECT_INPUT_ROLE,
                    feature_name=name,
                )
            )

    if feature_bank_view is not None:
        feature_context, feature_gate = feature_bank_view.aligned_conditional_values(
            exact_context_row_ids=context.row_ids,
            exact_context_inner_fold_ids=context.inner_fold_ids,
            exact_gate_row_ids=gate.row_ids,
        )
        for index, (name, kind, role) in enumerate(
            zip(
                feature_bank_view.feature_names,
                feature_bank_view.source_kinds,
                feature_bank_view.consumer_roles,
            )
        ):
            context_columns[role].append(feature_context[:, index])
            gate_columns[role].append(feature_gate[:, index])
            metadata[role].append(
                _UpstreamFamilyColumn(
                    input_kind="role_aware_raw_feature",
                    source_kind=kind,
                    consumer_role=role,
                    feature_name=name,
                )
            )

    standardized_context: dict[str, np.ndarray] = {}
    standardized_gate: dict[str, np.ndarray] = {}
    for role in _GATE_FEATURE_BANK_ROLES:
        raw_context = (
            np.column_stack(context_columns[role])
            if context_columns[role]
            else np.empty((len(context.row_ids), 0), dtype=float)
        )
        raw_gate = (
            np.column_stack(gate_columns[role])
            if gate_columns[role]
            else np.empty((len(gate.row_ids), 0), dtype=float)
        )
        standardized_context[role], standardized_gate[role] = _standardize_upstream_pair(
            raw_context, raw_gate
        )
    return _make_conditional_upstream_design(
        standardized_context,
        standardized_gate,
        {role: tuple(metadata[role]) for role in _GATE_FEATURE_BANK_ROLES},
    )


def _validate_gate_only_reference_view(
    view: GateSourceSignalView | GateFeatureBankView,
    *,
    exact_context_row_ids: Sequence[Hashable],
    exact_gate_row_ids: Sequence[Hashable],
    bank_name: str,
) -> None:
    """Prove a cumulative fit is a gate-only reference, never a fit covariate.

    The canonical 40-context workflow has complete-spent-context predictions
    for each untouched review gate, but it deliberately has no nested
    spent-context OOF bank.  A gate-only view is therefore accepted only when
    its context half is completely absent and every gate value has recursive
    lineage equal to the exact spent context.
    """

    context = frozenset(
        _review_keys(exact_context_row_ids, field_name="exact_context_row_ids")
    )
    gate = frozenset(_review_keys(exact_gate_row_ids, field_name="exact_gate_row_ids"))
    if not context or not gate or context & gate:
        raise ValueError("gate-only reference requires non-empty disjoint context and gate rows")
    if (
        view.context_row_ids
        or view.context_inner_fold_ids
        or view.context_fit_row_provenance
        or np.asarray(view.context_values).size
    ):
        raise ValueError(
            f"{bank_name} gate-only reference must not supply context-side upstream values"
        )
    # Aligning the gate half proves exact gate identity without touching the
    # conditional-context API.
    view.aligned_values(tuple(exact_gate_row_ids))
    for per_column in view.fit_row_provenance:
        for lineage in per_column:
            fitted = lineage.recursive_fit_row_ids()
            if fitted != context or fitted & gate:
                raise ValueError(
                    f"{bank_name} gate-only lineage must equal exactly all spent rows "
                    "and contain no gate or future rows"
                )


@dataclass(frozen=True)
class CausalReviewConfig:
    """Deterministic fitting, guard, and complexity settings for one review."""

    e_clip: float = 0.05
    nuisance_ridge_alpha: float = 1.0
    effect_ridge_alpha: float = 1.0
    contract_complexity_penalty: float = 0.002
    encoded_column_complexity_penalty: float = 0.0002
    minimum_score_improvement: float = 0.0
    nuisance_relative_tolerance: float = 0.05
    source_preservation_tolerance: float = 0.05
    source_context_r_loss_relative_tolerance: float = 0.05
    feature_bank_preservation_tolerance: float = 0.05
    estimator_policy: ReviewEstimatorPolicy | None = None

    def __post_init__(self) -> None:
        numeric = {
            "e_clip": self.e_clip,
            "nuisance_ridge_alpha": self.nuisance_ridge_alpha,
            "effect_ridge_alpha": self.effect_ridge_alpha,
            "contract_complexity_penalty": self.contract_complexity_penalty,
            "encoded_column_complexity_penalty": self.encoded_column_complexity_penalty,
            "minimum_score_improvement": self.minimum_score_improvement,
            "nuisance_relative_tolerance": self.nuisance_relative_tolerance,
            "source_preservation_tolerance": self.source_preservation_tolerance,
            "source_context_r_loss_relative_tolerance": (
                self.source_context_r_loss_relative_tolerance
            ),
            "feature_bank_preservation_tolerance": self.feature_bank_preservation_tolerance,
        }
        for name, value in numeric.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if not 0.0 < float(self.e_clip) < 0.5:
            raise ValueError("e_clip must be in (0, 0.5)")
        if float(self.nuisance_ridge_alpha) <= 0.0:
            raise ValueError("nuisance_ridge_alpha must be positive")
        if float(self.effect_ridge_alpha) < 0.0:
            raise ValueError("effect_ridge_alpha must be non-negative")
        for name in (
            "contract_complexity_penalty",
            "encoded_column_complexity_penalty",
            "minimum_score_improvement",
            "nuisance_relative_tolerance",
            "source_preservation_tolerance",
            "source_context_r_loss_relative_tolerance",
            "feature_bank_preservation_tolerance",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.estimator_policy is not None and not isinstance(
            self.estimator_policy,
            ReviewEstimatorPolicy,
        ):
            raise TypeError("estimator_policy must be ReviewEstimatorPolicy")


def _legacy_review_estimator_policy() -> ReviewEstimatorPolicy:
    """Compatibility settings matching the pre-portable estimator behavior."""

    return ReviewEstimatorPolicy(
        standardization_scale_epsilon=1e-8,
        logistic_alpha_floor=1e-12,
        logistic_solver="liblinear",
        logistic_max_iter=1000,
        logistic_random_seed=0,
        logistic_fit_intercept=True,
        logistic_class_weight=None,
        binary_no_features_fallback="prevalence",
        binary_single_class_fallback="prevalence",
        binary_fit_failure_policy="prevalence",
        continuous_minimum_fit_rows=2,
        continuous_degenerate_fallback="mean",
        effect_minimum_usable_rows=2,
        effect_no_usable_fallback="zero",
        effect_degenerate_fallback="weighted_mean",
        ridge_solver="auto",
        ridge_fit_intercept=True,
        ridge_tolerance=1e-4,
        ridge_max_iter=None,
        ridge_positive=False,
        ridge_random_seed=None,
    )


def _review_estimator_policy(
    config: CausalReviewConfig,
) -> ReviewEstimatorPolicy:
    return config.estimator_policy or _legacy_review_estimator_policy()


def _causal_review_config_payload(config: CausalReviewConfig) -> dict[str, Any]:
    return {
        "e_clip": config.e_clip,
        "nuisance_ridge_alpha": config.nuisance_ridge_alpha,
        "effect_ridge_alpha": config.effect_ridge_alpha,
        "contract_complexity_penalty": config.contract_complexity_penalty,
        "encoded_column_complexity_penalty": config.encoded_column_complexity_penalty,
        "minimum_score_improvement": config.minimum_score_improvement,
        "nuisance_relative_tolerance": config.nuisance_relative_tolerance,
        "source_preservation_tolerance": config.source_preservation_tolerance,
        "source_context_r_loss_relative_tolerance": (
            config.source_context_r_loss_relative_tolerance
        ),
        "feature_bank_preservation_tolerance": (config.feature_bank_preservation_tolerance),
        "estimator_policy": _review_estimator_policy(config).__dict__.copy(),
    }


@dataclass(frozen=True)
class GateAcceptanceDecision:
    accepted: bool
    reasons: tuple[str, ...]
    current: Mapping[str, Any] = field(repr=False)
    candidate: Mapping[str, Any] = field(repr=False)
    guards: Mapping[str, Any] = field(repr=False)
    decision_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POST_EXTRACTION_GATE_DECISION_SCHEMA_VERSION,
            "accepted": bool(self.accepted),
            "reasons": list(self.reasons),
            "current": _detached(self.current),
            "candidate": _detached(self.candidate),
            "guards": _detached(self.guards),
            "decision_sha256": self.decision_sha256,
        }


class _RoleEncoder:
    def __init__(
        self,
        specs: Sequence[Mapping[str, Any]],
        role: str,
        *,
        estimator_policy: ReviewEstimatorPolicy,
    ) -> None:
        canonical = [CandidateContract(spec).extraction_spec for spec in specs]
        self.specs = [spec for spec in canonical if role in set(spec.get("roles") or [])]
        self.role = role
        self.estimator_policy = estimator_policy
        self._continuous: dict[str, tuple[float, float]] = {}
        self.column_names: list[str] = []

    @staticmethod
    def _alias_map(spec: Mapping[str, Any]) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for category in spec.get("categories") or []:
            aliases[str(category).strip().casefold()] = str(category)
        raw = spec.get("value_aliases") or {}
        if isinstance(raw, Mapping):
            for canonical, values in raw.items():
                canonical_text = str(canonical)
                aliases[canonical_text.strip().casefold()] = canonical_text
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    for value in values:
                        aliases[str(value).strip().casefold()] = canonical_text
        return aliases

    def fit(self, frame: pd.DataFrame) -> "_RoleEncoder":
        columns: list[str] = []
        for spec in self.specs:
            name = str(spec["name"])
            value_column, _ = expected_extraction_columns(spec)
            missing = _missing_mask(frame, spec).to_numpy(dtype=bool)
            if spec["type"] == "continuous":
                values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
                observed = values[~missing & np.isfinite(values)]
                mean = float(np.mean(observed)) if len(observed) else 0.0
                scale = float(np.std(observed)) if len(observed) else 1.0
                if (
                    not math.isfinite(scale)
                    or scale
                    < float(
                        self.estimator_policy.standardization_scale_epsilon
                    )
                ):
                    scale = 1.0
                self._continuous[name] = (mean, scale)
                columns.extend([f"{name}__value", f"{name}__missing"])
            else:
                columns.extend([f"{name}__{category}" for category in spec.get("categories") or []])
                columns.append(f"{name}__missing")
        self.column_names = columns
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for spec in self.specs:
            name = str(spec["name"])
            value_column, _ = expected_extraction_columns(spec)
            missing = _missing_mask(frame, spec).to_numpy(dtype=bool)
            if spec["type"] == "continuous":
                values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
                mean, scale = self._continuous[name]
                invalid = missing | ~np.isfinite(values)
                standardized = np.where(invalid, mean, values)
                standardized = (standardized - mean) / scale
                blocks.extend([standardized[:, None], invalid.astype(float)[:, None]])
            else:
                alias_map = self._alias_map(spec)
                raw = frame[value_column].fillna("").astype(str).to_numpy()
                canonical = np.asarray(
                    [alias_map.get(value.strip().casefold(), "") for value in raw],
                    dtype=object,
                )
                declared = np.asarray(
                    [str(category) for category in spec.get("categories") or []],
                    dtype=object,
                )
                invalid = missing | ~np.isin(canonical, declared)
                for category in spec.get("categories") or []:
                    blocks.append(((canonical == str(category)) & ~invalid).astype(float)[:, None])
                blocks.append(invalid.astype(float)[:, None])
        if not blocks:
            return np.zeros((len(frame), 0), dtype=float)
        return np.concatenate(blocks, axis=1).astype(float, copy=False)


def _canonical_specs(specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [CandidateContract(spec).extraction_spec for spec in specs]
    names = [str(spec["name"]) for spec in result]
    if len(names) != len(set(names)):
        raise ValueError("contract names must be unique")
    return result


def _encoded_complexity(
    frame: pd.DataFrame,
    specs: Sequence[Mapping[str, Any]],
    *,
    estimator_policy: ReviewEstimatorPolicy,
) -> dict[str, int]:
    w_encoder = _RoleEncoder(
        specs,
        "confounder",
        estimator_policy=estimator_policy,
    ).fit(frame)
    x_encoder = _RoleEncoder(
        specs,
        "effect_modifier",
        estimator_policy=estimator_policy,
    ).fit(frame)
    return {
        "contract_count": int(len(specs)),
        "confounder_encoded_columns": int(len(w_encoder.column_names)),
        "effect_modifier_encoded_columns": int(len(x_encoder.column_names)),
        "encoded_column_count": int(len(w_encoder.column_names) + len(x_encoder.column_names)),
    }


def _binary_outcome(outcome: np.ndarray) -> bool:
    return bool(set(np.unique(np.asarray(outcome, dtype=float))).issubset({0.0, 1.0}))


def _fit_predict_binary(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_predict: np.ndarray,
    *,
    alpha: float,
    policy: ReviewEstimatorPolicy,
) -> np.ndarray:
    prevalence = float(np.mean(y_fit)) if len(y_fit) else 0.5
    if x_fit.shape[1] == 0:
        if policy.binary_no_features_fallback != "prevalence":
            raise RuntimeError("unsupported binary no-features fallback")
        return np.full(len(x_predict), prevalence, dtype=float)
    if len(np.unique(y_fit)) < 2:
        if policy.binary_single_class_fallback != "prevalence":
            raise RuntimeError("unsupported binary single-class fallback")
        return np.full(len(x_predict), prevalence, dtype=float)
    model = LogisticRegression(
        C=1.0 / max(float(alpha), float(policy.logistic_alpha_floor)),
        solver=policy.logistic_solver,
        max_iter=int(policy.logistic_max_iter),
        random_state=int(policy.logistic_random_seed),
        fit_intercept=bool(policy.logistic_fit_intercept),
        class_weight=policy.logistic_class_weight,
    )
    try:
        model.fit(x_fit, y_fit.astype(int))
        return np.asarray(model.predict_proba(x_predict)[:, 1], dtype=float)
    except ValueError:
        if policy.binary_fit_failure_policy == "abort":
            raise
        if policy.binary_fit_failure_policy != "prevalence":
            raise RuntimeError("unsupported binary fit-failure policy")
        return np.full(len(x_predict), prevalence, dtype=float)


def _fit_predict_continuous(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_predict: np.ndarray,
    *,
    alpha: float,
    policy: ReviewEstimatorPolicy,
) -> np.ndarray:
    fallback = float(np.mean(y_fit)) if len(y_fit) else 0.0
    if (
        x_fit.shape[1] == 0
        or len(y_fit) < int(policy.continuous_minimum_fit_rows)
    ):
        if policy.continuous_degenerate_fallback != "mean":
            raise RuntimeError("unsupported continuous degenerate fallback")
        return np.full(len(x_predict), fallback, dtype=float)
    model = Ridge(
        alpha=float(alpha),
        fit_intercept=bool(policy.ridge_fit_intercept),
        solver=policy.ridge_solver,
        tol=float(policy.ridge_tolerance),
        max_iter=policy.ridge_max_iter,
        positive=bool(policy.ridge_positive),
        random_state=policy.ridge_random_seed,
    )
    model.fit(x_fit, y_fit)
    return np.asarray(model.predict(x_predict), dtype=float)


def _fit_predict_nuisance(
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    fit_treatment: np.ndarray,
    fit_outcome: np.ndarray,
    specs: Sequence[Mapping[str, Any]],
    *,
    outcome_is_binary: bool,
    config: CausalReviewConfig,
    fit_propensity_upstream: np.ndarray | None = None,
    predict_propensity_upstream: np.ndarray | None = None,
    fit_outcome_upstream: np.ndarray | None = None,
    predict_outcome_upstream: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    estimator_policy = _review_estimator_policy(config)
    encoder = _RoleEncoder(
        specs,
        "confounder",
        estimator_policy=estimator_policy,
    ).fit(fit_frame)
    x_fit = encoder.transform(fit_frame)
    x_predict = encoder.transform(predict_frame)
    propensity_fit = _append_fixed_upstream(
        x_fit,
        fit_propensity_upstream,
        expected_rows=len(fit_frame),
        name="fit_propensity_upstream",
    )
    propensity_predict = _append_fixed_upstream(
        x_predict,
        predict_propensity_upstream,
        expected_rows=len(predict_frame),
        name="predict_propensity_upstream",
    )
    outcome_fit = _append_fixed_upstream(
        x_fit,
        fit_outcome_upstream,
        expected_rows=len(fit_frame),
        name="fit_outcome_upstream",
    )
    outcome_predict = _append_fixed_upstream(
        x_predict,
        predict_outcome_upstream,
        expected_rows=len(predict_frame),
        name="predict_outcome_upstream",
    )
    propensity = _fit_predict_binary(
        propensity_fit,
        fit_treatment,
        propensity_predict,
        alpha=config.nuisance_ridge_alpha,
        policy=estimator_policy,
    )
    if outcome_is_binary:
        outcome_prediction = _fit_predict_binary(
            outcome_fit,
            fit_outcome,
            outcome_predict,
            alpha=config.nuisance_ridge_alpha,
            policy=estimator_policy,
        )
    else:
        outcome_prediction = _fit_predict_continuous(
            outcome_fit,
            fit_outcome,
            outcome_predict,
            alpha=config.nuisance_ridge_alpha,
            policy=estimator_policy,
        )
    return np.asarray(propensity, dtype=float), np.asarray(outcome_prediction, dtype=float)


def _fit_predict_effect(
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    pseudo_target: np.ndarray,
    pseudo_weight: np.ndarray,
    specs: Sequence[Mapping[str, Any]],
    *,
    config: CausalReviewConfig,
    fit_effect_upstream: np.ndarray | None = None,
    predict_effect_upstream: np.ndarray | None = None,
) -> np.ndarray:
    estimator_policy = _review_estimator_policy(config)
    encoder = _RoleEncoder(
        specs,
        "effect_modifier",
        estimator_policy=estimator_policy,
    ).fit(fit_frame)
    x_fit = encoder.transform(fit_frame)
    x_predict = encoder.transform(predict_frame)
    x_fit = _append_fixed_upstream(
        x_fit,
        fit_effect_upstream,
        expected_rows=len(fit_frame),
        name="fit_effect_upstream",
    )
    x_predict = _append_fixed_upstream(
        x_predict,
        predict_effect_upstream,
        expected_rows=len(predict_frame),
        name="predict_effect_upstream",
    )
    weights = np.asarray(pseudo_weight, dtype=float)
    values = np.asarray(pseudo_target, dtype=float)
    usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(usable):
        if estimator_policy.effect_no_usable_fallback != "zero":
            raise RuntimeError("unsupported effect no-usable fallback")
        return np.zeros(len(predict_frame), dtype=float)
    fallback = float(np.average(values[usable], weights=weights[usable]))
    if (
        x_fit.shape[1] == 0
        or int(np.sum(usable))
        < int(estimator_policy.effect_minimum_usable_rows)
    ):
        if estimator_policy.effect_degenerate_fallback != "weighted_mean":
            raise RuntimeError("unsupported effect degenerate fallback")
        return np.full(len(predict_frame), fallback, dtype=float)
    model = Ridge(
        alpha=float(config.effect_ridge_alpha),
        fit_intercept=bool(estimator_policy.ridge_fit_intercept),
        solver=estimator_policy.ridge_solver,
        tol=float(estimator_policy.ridge_tolerance),
        max_iter=estimator_policy.ridge_max_iter,
        positive=bool(estimator_policy.ridge_positive),
        random_state=estimator_policy.ridge_random_seed,
    )
    model.fit(x_fit[usable], values[usable], sample_weight=weights[usable])
    return np.asarray(model.predict(x_predict), dtype=float)


def _append_fixed_upstream(
    extracted: np.ndarray,
    upstream: np.ndarray | None,
    *,
    expected_rows: int,
    name: str,
) -> np.ndarray:
    base = np.asarray(extracted, dtype=float)
    if upstream is None:
        return base
    fixed = np.asarray(upstream, dtype=float)
    if fixed.ndim != 2 or fixed.shape[0] != int(expected_rows):
        raise ValueError(f"{name} must be a two-dimensional row-aligned matrix")
    if not np.isfinite(fixed).all():
        raise ValueError(f"{name} must contain only finite values")
    if fixed.shape[1] == 0:
        return base
    return np.column_stack((base, fixed))


def _ordered_folds(folds: Sequence[Hashable]) -> tuple[Hashable, ...]:
    # Fold labels are operational only. Canonical JSON avoids dependence on row
    # ordering while supporting heterogeneous but hashable labels.
    return tuple(sorted(set(folds), key=lambda value: _canonical_json(value)))


def _nuisance_oof(
    data: ObservableCausalRows,
    specs: Sequence[Mapping[str, Any]],
    *,
    config: CausalReviewConfig,
    upstream_design: _ConditionalUpstreamDesign | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if data.inner_fold_ids is None:
        raise ValueError("inner_fold_ids are required for nuisance cross-fitting")
    folds = np.asarray(data.inner_fold_ids, dtype=object)
    unique_folds = _ordered_folds(data.inner_fold_ids)
    if len(unique_folds) < 2:
        raise ValueError("nuisance cross-fitting requires at least two fixed inner folds")
    e_hat = np.full(len(data.row_ids), np.nan, dtype=float)
    m_hat = np.full(len(data.row_ids), np.nan, dtype=float)
    outcome_is_binary = _binary_outcome(data.outcome)
    for fold_id in unique_folds:
        heldout = folds == fold_id
        fit = ~heldout
        if not np.any(heldout) or not np.any(fit):
            raise ValueError("every fixed inner fold must have fit and heldout rows")
        e_fold, m_fold = _fit_predict_nuisance(
            data.extracted.loc[fit].reset_index(drop=True),
            data.extracted.loc[heldout].reset_index(drop=True),
            np.asarray(data.treatment[fit], dtype=float),
            np.asarray(data.outcome[fit], dtype=float),
            specs,
            outcome_is_binary=outcome_is_binary,
            config=config,
            fit_propensity_upstream=(
                None
                if upstream_design is None
                else upstream_design.values(PROPENSITY_NUISANCE_FEATURE_ROLE, scope="context")[fit]
            ),
            predict_propensity_upstream=(
                None
                if upstream_design is None
                else upstream_design.values(PROPENSITY_NUISANCE_FEATURE_ROLE, scope="context")[
                    heldout
                ]
            ),
            fit_outcome_upstream=(
                None
                if upstream_design is None
                else upstream_design.values(OUTCOME_NUISANCE_FEATURE_ROLE, scope="context")[fit]
            ),
            predict_outcome_upstream=(
                None
                if upstream_design is None
                else upstream_design.values(OUTCOME_NUISANCE_FEATURE_ROLE, scope="context")[heldout]
            ),
        )
        e_hat[heldout] = e_fold
        m_hat[heldout] = m_fold
    if not np.isfinite(e_hat).all() or not np.isfinite(m_hat).all():
        raise RuntimeError("nuisance cross-fitting did not cover every row")
    return e_hat, m_hat


def _nested_oof_predictions(
    data: ObservableCausalRows,
    specs: Sequence[Mapping[str, Any]],
    *,
    config: CausalReviewConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nested OOF nuisances and weighted-R predictions on fixed folds."""

    if data.inner_fold_ids is None:
        raise ValueError("inner_fold_ids are required for causal diagnostics")
    folds = np.asarray(data.inner_fold_ids, dtype=object)
    unique_folds = _ordered_folds(data.inner_fold_ids)
    if len(unique_folds) < 3:
        raise ValueError("nested diagnostics require at least three fixed inner folds")
    e_hat = np.full(len(data.row_ids), np.nan, dtype=float)
    m_hat = np.full(len(data.row_ids), np.nan, dtype=float)
    tau_hat = np.full(len(data.row_ids), np.nan, dtype=float)
    opaque_fold_index = np.full(len(data.row_ids), -1, dtype=int)
    outcome_is_binary = _binary_outcome(data.outcome)

    for display_index, outer_fold in enumerate(unique_folds, start=1):
        evaluation = folds == outer_fold
        fit = ~evaluation
        fit_positions = np.flatnonzero(fit)
        evaluation_positions = np.flatnonzero(evaluation)
        if not len(evaluation_positions) or not len(fit_positions):
            raise ValueError("every fixed inner fold must have fit and heldout rows")

        e_eval, m_eval = _fit_predict_nuisance(
            data.extracted.iloc[fit_positions].reset_index(drop=True),
            data.extracted.iloc[evaluation_positions].reset_index(drop=True),
            np.asarray(data.treatment[fit_positions], dtype=float),
            np.asarray(data.outcome[fit_positions], dtype=float),
            specs,
            outcome_is_binary=outcome_is_binary,
            config=config,
        )
        e_hat[evaluation_positions] = e_eval
        m_hat[evaluation_positions] = m_eval

        # Inner-inner nuisances ensure that no effect-fit pseudo target uses a
        # nuisance prediction trained on that same effect-fit row.
        fit_data = ObservableCausalRows(
            row_ids=tuple(data.row_ids[position] for position in fit_positions),
            extracted=data.extracted.iloc[fit_positions].reset_index(drop=True),
            treatment=np.asarray(data.treatment[fit_positions], dtype=float),
            outcome=np.asarray(data.outcome[fit_positions], dtype=float),
            inner_fold_ids=tuple(data.inner_fold_ids[position] for position in fit_positions),
        )
        e_inner, m_inner = _nuisance_oof(fit_data, specs, config=config)
        e_inner = np.clip(e_inner, config.e_clip, 1.0 - config.e_clip)
        t_residual = fit_data.treatment - e_inner
        y_residual = fit_data.outcome - m_inner
        pseudo_target = y_residual / t_residual
        pseudo_weight = np.square(t_residual)
        tau_hat[evaluation_positions] = _fit_predict_effect(
            fit_data.extracted,
            data.extracted.iloc[evaluation_positions].reset_index(drop=True),
            pseudo_target,
            pseudo_weight,
            specs,
            config=config,
        )
        opaque_fold_index[evaluation_positions] = display_index
    if not (
        np.isfinite(e_hat).all()
        and np.isfinite(m_hat).all()
        and np.isfinite(tau_hat).all()
        and np.all(opaque_fold_index > 0)
    ):
        raise RuntimeError("nested diagnostics did not cover every row")
    return e_hat, m_hat, tau_hat, opaque_fold_index


def _safe_auc(observed: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(np.unique(observed)) < 2:
        return None
    try:
        return _finite_or_none(roc_auc_score(observed.astype(int), prediction))
    except ValueError:
        return None


def _safe_binary_loss(observed: np.ndarray, prediction: np.ndarray) -> float | None:
    try:
        return _finite_or_none(
            log_loss(
                observed.astype(int),
                np.clip(prediction, 1e-6, 1.0 - 1e-6),
                labels=[0, 1],
            )
        )
    except ValueError:
        return None


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return _finite_or_none(np.corrcoef(left, right)[0, 1])


def _observable_metrics(
    treatment: np.ndarray,
    outcome: np.ndarray,
    e_hat: np.ndarray,
    m_hat: np.ndarray,
    tau_hat: np.ndarray,
    *,
    config: CausalReviewConfig,
) -> dict[str, Any]:
    e_clipped = np.clip(e_hat, config.e_clip, 1.0 - config.e_clip)
    t_residual = treatment - e_clipped
    y_residual = outcome - m_hat
    weighted_r = np.square(y_residual - tau_hat * t_residual)
    zero_r = np.square(y_residual)
    r_loss = float(np.mean(weighted_r))
    zero_loss = float(np.mean(zero_r))
    ratio = r_loss / max(zero_loss, 1e-12)
    treatment_metrics = {
        "auroc": _safe_auc(treatment, e_hat),
        "brier": _finite_or_none(brier_score_loss(treatment.astype(int), e_hat)),
        "log_loss": _safe_binary_loss(treatment, e_hat),
    }
    outcome_is_binary = _binary_outcome(outcome)
    if outcome_is_binary:
        outcome_metrics = {
            "kind": "binary",
            "auroc": _safe_auc(outcome, m_hat),
            "brier": _finite_or_none(brier_score_loss(outcome.astype(int), m_hat)),
            "log_loss": _safe_binary_loss(outcome, m_hat),
            "rmse": None,
            "loss": _safe_binary_loss(outcome, m_hat),
        }
    else:
        rmse = _finite_or_none(math.sqrt(mean_squared_error(outcome, m_hat)))
        outcome_metrics = {
            "kind": "continuous",
            "auroc": None,
            "brier": None,
            "log_loss": None,
            "rmse": rmse,
            "loss": _finite_or_none(mean_squared_error(outcome, m_hat)),
        }
    return {
        "nuisance": {
            "treatment": treatment_metrics,
            "outcome": outcome_metrics,
        },
        "effect": {
            "weighted_r_loss": r_loss,
            "zero_effect_r_loss": zero_loss,
            "r_loss_ratio": ratio,
            "relative_r_loss_improvement": float(1.0 - ratio),
            "tau_mean": float(np.mean(tau_hat)),
            "tau_std": float(np.std(tau_hat)),
            "tau_pseudo_target_corr": _safe_corr(
                tau_hat,
                y_residual / t_residual,
            ),
        },
    }


def _fold_stability_metrics(
    data: ObservableCausalRows,
    e_hat: np.ndarray,
    m_hat: np.ndarray,
    tau_hat: np.ndarray,
    opaque_fold_index: np.ndarray,
    *,
    config: CausalReviewConfig,
) -> dict[str, Any]:
    per_fold: list[dict[str, Any]] = []
    for fold_index in sorted(set(opaque_fold_index.tolist())):
        mask = opaque_fold_index == fold_index
        metrics = _observable_metrics(
            np.asarray(data.treatment[mask]),
            np.asarray(data.outcome[mask]),
            np.asarray(e_hat[mask]),
            np.asarray(m_hat[mask]),
            np.asarray(tau_hat[mask]),
            config=config,
        )
        per_fold.append(
            {
                "fold_id": f"fold_{fold_index:04d}",
                "n_rows": int(np.sum(mask)),
                "weighted_r_loss": metrics["effect"]["weighted_r_loss"],
                "r_loss_ratio": metrics["effect"]["r_loss_ratio"],
                "tau_std": metrics["effect"]["tau_std"],
                "treatment_log_loss": metrics["nuisance"]["treatment"]["log_loss"],
                "outcome_loss": metrics["nuisance"]["outcome"]["loss"],
            }
        )
    summary: dict[str, Any] = {}
    for key in (
        "weighted_r_loss",
        "r_loss_ratio",
        "tau_std",
        "treatment_log_loss",
        "outcome_loss",
    ):
        values = np.asarray(
            [row[key] for row in per_fold if row[key] is not None],
            dtype=float,
        )
        summary[key] = {
            "mean": _finite_or_none(np.mean(values)) if len(values) else None,
            "std": _finite_or_none(np.std(values)) if len(values) else None,
            "range": _finite_or_none(np.max(values) - np.min(values)) if len(values) else None,
        }
    return {"per_fold": per_fold, "summary": summary}


def _metric_delta(without: Any, full: Any) -> float | None:
    without_value = _finite_or_none(without)
    full_value = _finite_or_none(full)
    if without_value is None or full_value is None:
        return None
    return float(without_value - full_value)


def build_causal_review_diagnostics(
    data: ObservableCausalRows,
    specs: Sequence[Mapping[str, Any]],
    *,
    config: CausalReviewConfig | None = None,
    diagnostic_start: int = 1,
) -> dict[str, Any]:
    """Build sanitized pre-proposal diagnostics from already-spent folds only.

    Nuisance targets are cross-fitted inside each effect-model fit partition,
    making the reported weighted R-loss genuinely nested.  No untouched-gate
    object is accepted by this function, so changing future gate outcomes
    cannot change an agent's proposal context.
    """

    if not isinstance(data, ObservableCausalRows):
        raise TypeError("data must be ObservableCausalRows")
    config = config or CausalReviewConfig()
    if not isinstance(config, CausalReviewConfig):
        raise TypeError("config must be CausalReviewConfig")
    if (
        isinstance(diagnostic_start, bool)
        or not isinstance(diagnostic_start, (int, np.integer))
        or int(diagnostic_start) < 1
        or int(diagnostic_start) + len(specs) > 9999
    ):
        raise ValueError("diagnostic_start must be a positive integer")
    canonical = _canonical_specs(specs)
    e_hat, m_hat, tau_hat, fold_index = _nested_oof_predictions(
        data,
        canonical,
        config=config,
    )
    full_metrics = _observable_metrics(
        data.treatment,
        data.outcome,
        e_hat,
        m_hat,
        tau_hat,
        config=config,
    )
    estimator_policy = _review_estimator_policy(config)
    full_complexity = _encoded_complexity(
        data.extracted,
        canonical,
        estimator_policy=estimator_policy,
    )
    ablations: list[dict[str, Any]] = []
    for offset, removed in enumerate(canonical, start=1):
        reduced = [spec for spec in canonical if spec["name"] != removed["name"]]
        a_e, a_m, a_tau, _ = _nested_oof_predictions(data, reduced, config=config)
        metrics = _observable_metrics(
            data.treatment,
            data.outcome,
            a_e,
            a_m,
            a_tau,
            config=config,
        )
        complexity = _encoded_complexity(
            data.extracted,
            reduced,
            estimator_policy=estimator_policy,
        )
        ablations.append(
            {
                "diagnostic_id": f"diagnostic_{int(diagnostic_start) + offset:04d}",
                "kind": "contract_ablation",
                "contract_name": removed["name"],
                "roles": list(removed.get("roles") or []),
                "weighted_r_loss_delta_when_removed": _metric_delta(
                    metrics["effect"]["weighted_r_loss"],
                    full_metrics["effect"]["weighted_r_loss"],
                ),
                "r_loss_ratio_delta_when_removed": _metric_delta(
                    metrics["effect"]["r_loss_ratio"],
                    full_metrics["effect"]["r_loss_ratio"],
                ),
                "treatment_log_loss_delta_when_removed": _metric_delta(
                    metrics["nuisance"]["treatment"]["log_loss"],
                    full_metrics["nuisance"]["treatment"]["log_loss"],
                ),
                "outcome_loss_delta_when_removed": _metric_delta(
                    metrics["nuisance"]["outcome"]["loss"],
                    full_metrics["nuisance"]["outcome"]["loss"],
                ),
                "tau_std_delta_when_removed": _metric_delta(
                    metrics["effect"]["tau_std"],
                    full_metrics["effect"]["tau_std"],
                ),
                "encoded_columns_removed": int(
                    full_complexity["encoded_column_count"] - complexity["encoded_column_count"]
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": POST_EXTRACTION_CAUSAL_DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_id": f"diagnostic_{int(diagnostic_start):04d}",
        "kind": "nested_observable_causal_quality",
        "row_count": int(len(data.row_ids)),
        "fixed_inner_fold_count": int(len(set(data.inner_fold_ids or ()))),
        "evaluation_configuration": _causal_review_config_payload(config),
        "metrics": full_metrics,
        "inner_fold_stability": _fold_stability_metrics(
            data,
            e_hat,
            m_hat,
            tau_hat,
            fold_index,
            config=config,
        ),
        "complexity": full_complexity,
        "contract_ablations": ablations,
    }
    payload["diagnostic_sha256"] = _content_sha256(payload)
    return payload


def _fit_context_predict_gate(
    context: ObservableCausalRows,
    gate: ObservableCausalRows,
    specs: Sequence[Mapping[str, Any]],
    *,
    config: CausalReviewConfig,
    upstream_design: _ConditionalUpstreamDesign | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    if context.inner_fold_ids is None:
        raise ValueError("context inner_fold_ids are required for untouched-gate fitting")
    e_context, m_context = _nuisance_oof(
        context,
        specs,
        config=config,
        upstream_design=upstream_design,
    )
    e_context = np.clip(e_context, config.e_clip, 1.0 - config.e_clip)
    t_residual = context.treatment - e_context
    y_residual = context.outcome - m_context
    pseudo_target = y_residual / t_residual
    pseudo_weight = np.square(t_residual)
    tau_gate = _fit_predict_effect(
        context.extracted,
        gate.extracted,
        pseudo_target,
        pseudo_weight,
        specs,
        config=config,
        fit_effect_upstream=(
            None
            if upstream_design is None
            else upstream_design.values(UNCALIBRATED_EFFECT_MODIFIER_ROLE, scope="context")
        ),
        predict_effect_upstream=(
            None
            if upstream_design is None
            else upstream_design.values(UNCALIBRATED_EFFECT_MODIFIER_ROLE, scope="gate")
        ),
    )
    e_gate, m_gate = _fit_predict_nuisance(
        context.extracted,
        gate.extracted,
        context.treatment,
        context.outcome,
        specs,
        outcome_is_binary=_binary_outcome(context.outcome),
        config=config,
        fit_propensity_upstream=(
            None
            if upstream_design is None
            else upstream_design.values(PROPENSITY_NUISANCE_FEATURE_ROLE, scope="context")
        ),
        predict_propensity_upstream=(
            None
            if upstream_design is None
            else upstream_design.values(PROPENSITY_NUISANCE_FEATURE_ROLE, scope="gate")
        ),
        fit_outcome_upstream=(
            None
            if upstream_design is None
            else upstream_design.values(OUTCOME_NUISANCE_FEATURE_ROLE, scope="context")
        ),
        predict_outcome_upstream=(
            None
            if upstream_design is None
            else upstream_design.values(OUTCOME_NUISANCE_FEATURE_ROLE, scope="gate")
        ),
    )
    gate_metrics = _observable_metrics(
        gate.treatment,
        gate.outcome,
        e_gate,
        m_gate,
        tau_gate,
        config=config,
    )
    return gate_metrics, e_gate, m_gate, tau_gate


def _gate_source_metrics(
    gate: ObservableCausalRows,
    e_hat: np.ndarray,
    m_hat: np.ndarray,
    tau_hat: np.ndarray,
    source_view: GateSourceSignalView,
    *,
    config: CausalReviewConfig,
    gate_reference_only: bool = False,
) -> dict[str, Any]:
    # These are post-fit preservation diagnostics.  Under the legacy
    # conditional policy the authenticated values also enter the effect
    # regression.  Under the gate-only policy they are reference diagnostics
    # only and never enter a fitted design.  In neither policy are they
    # outcomes, pseudo-targets, or observed treatment effects.
    values = source_view.aligned_values(gate.row_ids)
    e_clipped = np.clip(e_hat, config.e_clip, 1.0 - config.e_clip)
    t_residual = gate.treatment - e_clipped
    y_residual = gate.outcome - m_hat
    zero_loss = float(np.mean(np.square(y_residual)))
    rows: list[dict[str, Any]] = []
    signed_correlations: list[float] = []
    absolute_correlations: list[float] = []
    contextual_ratios: list[float] = []
    for column, (name, kind) in enumerate(zip(source_view.source_names, source_view.source_kinds)):
        source = values[:, column]
        correlation = _safe_corr(tau_hat, source)
        contextual_loss = float(np.mean(np.square(y_residual - source * t_residual)))
        contextual_ratio = contextual_loss / max(zero_loss, 1e-12)
        if correlation is not None:
            signed_correlations.append(float(correlation))
            absolute_correlations.append(abs(float(correlation)))
        contextual_ratios.append(contextual_ratio)
        rows.append(
            {
                "source_name": name,
                "source_kind": kind,
                "tau_correlation": correlation,
                "absolute_tau_correlation": (
                    None if correlation is None else abs(float(correlation))
                ),
                "contextual_weighted_r_loss": contextual_loss,
                "contextual_r_loss_ratio": contextual_ratio,
                "zero_effect_weighted_r_loss": zero_loss,
                "contextual_weighted_r_loss_delta_vs_zero_effect": (contextual_loss - zero_loss),
                "contextual_r_loss_ratio_improvement_vs_zero_effect": (1.0 - contextual_ratio),
            }
        )
    result = {
        "sources": rows,
        "source_preservation_score": (
            float(np.mean(absolute_correlations)) if absolute_correlations else None
        ),
        "mean_signed_source_correlation": (
            float(np.mean(signed_correlations)) if signed_correlations else None
        ),
        "mean_absolute_source_correlation": (
            float(np.mean(absolute_correlations)) if absolute_correlations else None
        ),
        "calibrated_sources_used_as_effect_regression_covariates": not gate_reference_only,
        "calibrated_sources_used_as_observed_effect_targets": False,
        "gate_reference_only": bool(gate_reference_only),
        "diagnostic_scope": "untouched_gate_post_fit_reference",
    }
    mean_ratio = float(np.mean(contextual_ratios)) if contextual_ratios else None
    if gate_reference_only:
        result["mean_source_context_r_loss_ratio"] = None
        result["mean_source_gate_reference_r_loss_ratio"] = mean_ratio
    else:
        result["mean_source_context_r_loss_ratio"] = mean_ratio
        result["mean_source_gate_reference_r_loss_ratio"] = None
    return result


def _gate_feature_bank_metrics(
    gate: ObservableCausalRows,
    e_hat: np.ndarray,
    m_hat: np.ndarray,
    tau_hat: np.ndarray,
    feature_bank_view: GateFeatureBankView,
    *,
    gate_reference_only: bool = False,
) -> dict[str, Any]:
    """Measure only role-matched preservation of uncalibrated feature bases."""

    values = feature_bank_view.aligned_values(gate.row_ids)
    prediction_by_role = {
        PROPENSITY_NUISANCE_FEATURE_ROLE: np.asarray(e_hat, dtype=float),
        OUTCOME_NUISANCE_FEATURE_ROLE: np.asarray(m_hat, dtype=float),
        UNCALIBRATED_EFFECT_MODIFIER_ROLE: np.asarray(tau_hat, dtype=float),
    }
    rows: list[dict[str, Any]] = []
    scores_by_role: dict[str, list[float]] = {role: [] for role in _GATE_FEATURE_BANK_ROLES}
    scores_by_family: dict[tuple[str, str], list[float]] = {}
    feature_count_by_family: dict[tuple[str, str], int] = {}
    for column, (name, kind, role) in enumerate(
        zip(
            feature_bank_view.feature_names,
            feature_bank_view.source_kinds,
            feature_bank_view.consumer_roles,
        )
    ):
        family_key = (kind, role)
        scores_by_family.setdefault(family_key, [])
        feature_count_by_family[family_key] = feature_count_by_family.get(family_key, 0) + 1
        correlation = _safe_corr(prediction_by_role[role], values[:, column])
        if correlation is not None:
            absolute_correlation = abs(float(correlation))
            scores_by_role[role].append(absolute_correlation)
            scores_by_family[family_key].append(absolute_correlation)
        rows.append(
            {
                "feature_name": name,
                "source_kind": kind,
                "consumer_role": role,
                "role_matched_prediction_correlation": correlation,
                "absolute_role_matched_prediction_correlation": (
                    None if correlation is None else abs(float(correlation))
                ),
            }
        )
    role_scores = {
        role: (float(np.mean(values)) if values else None)
        for role, values in sorted(scores_by_role.items())
    }
    available = [value for value in role_scores.values() if value is not None]
    all_feature_scores = [
        value for family_scores in scores_by_family.values() for value in family_scores
    ]
    feature_level_mean = float(np.mean(all_feature_scores)) if all_feature_scores else None
    aggregate_correlation = float(np.sum(all_feature_scores)) if all_feature_scores else 0.0
    family_rows: list[dict[str, Any]] = []
    for (kind, role), family_scores in sorted(scores_by_family.items()):
        other_scores = [
            value
            for other_key, other_family_scores in scores_by_family.items()
            if other_key != (kind, role)
            for value in other_family_scores
        ]
        family_aggregate = float(np.sum(family_scores)) if family_scores else 0.0
        family_mean = float(np.mean(family_scores)) if family_scores else None
        leave_family_out_mean = float(np.mean(other_scores)) if other_scores else None
        delta_when_removed = (
            None
            if feature_level_mean is None or leave_family_out_mean is None
            else leave_family_out_mean - feature_level_mean
        )
        family_rows.append(
            {
                "source_kind": kind,
                "consumer_role": role,
                "feature_count": int(feature_count_by_family[(kind, role)]),
                "finite_correlation_count": int(len(family_scores)),
                "mean_absolute_role_matched_prediction_correlation": family_mean,
                "aggregate_absolute_role_matched_prediction_correlation": (family_aggregate),
                "aggregate_absolute_correlation_share": (
                    None
                    if aggregate_correlation <= 0.0
                    else family_aggregate / aggregate_correlation
                ),
                "leave_family_out_feature_mean_absolute_correlation": (leave_family_out_mean),
                "feature_mean_absolute_correlation_delta_when_family_removed": (delta_when_removed),
            }
        )
    return {
        "features": rows,
        "preservation_score_by_consumer_role": role_scores,
        "preservation_by_source_kind_and_consumer_role": family_rows,
        "feature_bank_preservation_score": (float(np.mean(available)) if available else None),
        "feature_level_mean_absolute_role_matched_prediction_correlation": (feature_level_mean),
        "correlation_leave_family_out_sensitivity_definition": (
            "Descriptive change in mean absolute role-matched correlation after "
            "removing one exact source-kind/consumer-role family. Predictive "
            "family ablations are reported separately and always refit the model."
        ),
        "raw_feature_values_used_as_treatment_effects": False,
        "raw_feature_values_used_as_model_inputs": not gate_reference_only,
        "raw_feature_model_input_routing": (
            None
            if gate_reference_only
            else {
                PROPENSITY_NUISANCE_FEATURE_ROLE: "propensity_nuisance_only",
                OUTCOME_NUISANCE_FEATURE_ROLE: "outcome_nuisance_only",
                UNCALIBRATED_EFFECT_MODIFIER_ROLE: "effect_regression_covariate_only",
            }
        ),
        "gate_reference_only": bool(gate_reference_only),
        "diagnostic_scope": "untouched_gate_post_fit_reference",
    }


def _predictive_upstream_family_ablations(
    context: ObservableCausalRows,
    gate: ObservableCausalRows,
    current_specs: Sequence[Mapping[str, Any]],
    candidate_specs: Sequence[Mapping[str, Any]],
    *,
    candidate_context: ObservableCausalRows,
    candidate_gate: ObservableCausalRows,
    design: _ConditionalUpstreamDesign,
    current_full_metrics: Mapping[str, Any],
    candidate_full_metrics: Mapping[str, Any],
    config: CausalReviewConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Refit current/candidate models after deleting each exact upstream family."""

    family_keys = sorted(
        {
            column.family_key
            for role in _GATE_FEATURE_BANK_ROLES
            for column in design.columns_by_role[role]
        }
    )

    def one(
        key: tuple[str, str, str],
        *,
        rows: ObservableCausalRows,
        heldout: ObservableCausalRows,
        specs: Sequence[Mapping[str, Any]],
        full_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        ablated = design.without_family(key)
        metrics, _e_hat, _m_hat, _tau_hat = _fit_context_predict_gate(
            rows,
            heldout,
            specs,
            config=config,
            upstream_design=ablated,
        )
        columns_removed = sum(
            column.family_key == key
            for role in _GATE_FEATURE_BANK_ROLES
            for column in design.columns_by_role[role]
        )
        return {
            "input_kind": key[0],
            "source_kind": key[1],
            "consumer_role": key[2],
            "upstream_columns_removed": int(columns_removed),
            "weighted_r_loss_delta_when_removed": _metric_delta(
                metrics["effect"]["weighted_r_loss"],
                full_metrics["effect"]["weighted_r_loss"],
            ),
            "treatment_log_loss_delta_when_removed": _metric_delta(
                metrics["nuisance"]["treatment"]["log_loss"],
                full_metrics["nuisance"]["treatment"]["log_loss"],
            ),
            "outcome_loss_delta_when_removed": _metric_delta(
                metrics["nuisance"]["outcome"]["loss"],
                full_metrics["nuisance"]["outcome"]["loss"],
            ),
            "ablation_refit_performed": True,
            "raw_feature_value_used_directly_as_tau": False,
        }

    current_rows = [
        one(
            key,
            rows=context,
            heldout=gate,
            specs=current_specs,
            full_metrics=current_full_metrics,
        )
        for key in family_keys
    ]
    candidate_rows = [
        one(
            key,
            rows=candidate_context,
            heldout=candidate_gate,
            specs=candidate_specs,
            full_metrics=candidate_full_metrics,
        )
        for key in family_keys
    ]
    return current_rows, candidate_rows


def _relative_upper_guard(
    candidate: Any,
    current: Any,
    tolerance: float,
) -> tuple[bool, float | None]:
    current_value = _finite_or_none(current)
    candidate_value = _finite_or_none(candidate)
    if current_value is None:
        return True, None
    if candidate_value is None:
        return False, current_value * (1.0 + tolerance)
    maximum = current_value * (1.0 + tolerance) + 1e-12
    return bool(candidate_value <= maximum), maximum


def evaluate_untouched_gate_acceptance(
    context: ObservableCausalRows,
    gate: ObservableCausalRows,
    current_specs: Sequence[Mapping[str, Any]],
    candidate_specs: Sequence[Mapping[str, Any]],
    *,
    source_view: GateSourceSignalView | None = None,
    feature_bank_view: GateFeatureBankView | None = None,
    candidate_context: ObservableCausalRows | None = None,
    candidate_gate: ObservableCausalRows | None = None,
    config: CausalReviewConfig | None = None,
    upstream_review_policy: str = CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY,
) -> GateAcceptanceDecision:
    """Compare exactly one proposed revision on an untouched observable gate.

    The API accepts one candidate, preventing batch selection on the same hidden
    gate.  Callers are responsible for consuming a gate once in their bounded
    sequential loop.  Exact row-set separation and source lineage are enforced
    here before any fit occurs.
    """

    if not isinstance(context, ObservableCausalRows) or not isinstance(gate, ObservableCausalRows):
        raise TypeError("context and gate must be ObservableCausalRows")
    overlap = frozenset(context.row_ids) & frozenset(gate.row_ids)
    if overlap:
        raise ValueError("context and untouched gate row IDs must be disjoint")
    if not len(context.row_ids) or not len(gate.row_ids):
        raise ValueError("context and untouched gate must both be non-empty")
    if (candidate_context is None) != (candidate_gate is None):
        raise ValueError("candidate_context and candidate_gate must be supplied together")
    if candidate_context is not None and candidate_gate is not None:
        if not isinstance(candidate_context, ObservableCausalRows) or not isinstance(
            candidate_gate, ObservableCausalRows
        ):
            raise TypeError("candidate_context and candidate_gate must be ObservableCausalRows")

        def require_same_observed_rows(
            current_rows: ObservableCausalRows,
            proposed_rows: ObservableCausalRows,
            *,
            name: str,
        ) -> None:
            if current_rows.row_ids != proposed_rows.row_ids:
                raise ValueError(f"{name} row IDs/order must match the current representation")
            if not np.array_equal(current_rows.treatment, proposed_rows.treatment):
                raise ValueError(f"{name} treatment must match the current representation")
            if not np.array_equal(current_rows.outcome, proposed_rows.outcome):
                raise ValueError(f"{name} outcome must match the current representation")
            if current_rows.inner_fold_ids != proposed_rows.inner_fold_ids:
                raise ValueError(f"{name} inner folds must match the current representation")

        require_same_observed_rows(context, candidate_context, name="candidate_context")
        require_same_observed_rows(gate, candidate_gate, name="candidate_gate")
    else:
        candidate_context = context
        candidate_gate = gate
    config = config or CausalReviewConfig()
    if not isinstance(config, CausalReviewConfig):
        raise TypeError("config must be CausalReviewConfig")
    if not isinstance(upstream_review_policy, str):
        raise TypeError("upstream_review_policy must be a string")
    upstream_review_policy = upstream_review_policy.strip()
    if upstream_review_policy not in _UPSTREAM_REVIEW_POLICIES:
        raise ValueError(
            "upstream_review_policy must be one of "
            f"{sorted(_UPSTREAM_REVIEW_POLICIES)}"
        )
    gate_reference_only = (
        upstream_review_policy == GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY
    )
    current_specs = _canonical_specs(current_specs)
    candidate_specs = _canonical_specs(candidate_specs)
    if not candidate_specs:
        raise ValueError("candidate_specs cannot remove every contract")
    if source_view is not None:
        if not isinstance(source_view, GateSourceSignalView):
            raise TypeError("source_view must be GateSourceSignalView")
        # Force exact gate identity before fitting either representation.
        source_view.aligned_values(gate.row_ids)
    if feature_bank_view is not None:
        if not isinstance(feature_bank_view, GateFeatureBankView):
            raise TypeError("feature_bank_view must be GateFeatureBankView")
        feature_bank_view.aligned_values(gate.row_ids)

    if gate_reference_only:
        if source_view is not None:
            _validate_gate_only_reference_view(
                source_view,
                exact_context_row_ids=context.row_ids,
                exact_gate_row_ids=gate.row_ids,
                bank_name="source",
            )
        if feature_bank_view is not None:
            _validate_gate_only_reference_view(
                feature_bank_view,
                exact_context_row_ids=context.row_ids,
                exact_gate_row_ids=gate.row_ids,
                bank_name="feature",
            )
        # The empty design is authenticated for the decision record, but None
        # is passed to the fit so no upstream gate value can become a training
        # or prediction covariate.
        upstream_design = _build_conditional_upstream_design(
            context,
            gate,
            source_view=None,
            feature_bank_view=None,
        )
        fit_upstream_design: _ConditionalUpstreamDesign | None = None
    else:
        upstream_design = _build_conditional_upstream_design(
            context,
            gate,
            source_view=source_view,
            feature_bank_view=feature_bank_view,
        )
        fit_upstream_design = upstream_design
    upstream_design_sha256 = upstream_design.content_sha256
    upstream_design.verify_content()

    current_metrics, current_e, current_m, current_tau = _fit_context_predict_gate(
        context,
        gate,
        current_specs,
        config=config,
        upstream_design=fit_upstream_design,
    )
    candidate_metrics, candidate_e, candidate_m, candidate_tau = _fit_context_predict_gate(
        candidate_context,
        candidate_gate,
        candidate_specs,
        config=config,
        upstream_design=fit_upstream_design,
    )
    upstream_design.verify_content()
    if upstream_design.content_sha256 != upstream_design_sha256:  # pragma: no cover
        raise RuntimeError("conditional upstream design identity changed during gate fitting")
    estimator_policy = _review_estimator_policy(config)
    current_complexity = _encoded_complexity(
        context.extracted,
        current_specs,
        estimator_policy=estimator_policy,
    )
    candidate_complexity = _encoded_complexity(
        candidate_context.extracted,
        candidate_specs,
        estimator_policy=estimator_policy,
    )
    current_source: dict[str, Any] | None = None
    candidate_source: dict[str, Any] | None = None
    current_feature_bank: dict[str, Any] | None = None
    candidate_feature_bank: dict[str, Any] | None = None
    if source_view is not None:
        current_source = _gate_source_metrics(
            gate,
            current_e,
            current_m,
            current_tau,
            source_view,
            config=config,
            gate_reference_only=gate_reference_only,
        )
        candidate_source = _gate_source_metrics(
            candidate_gate,
            candidate_e,
            candidate_m,
            candidate_tau,
            source_view,
            config=config,
            gate_reference_only=gate_reference_only,
        )
    if feature_bank_view is not None:
        current_feature_bank = _gate_feature_bank_metrics(
            gate,
            current_e,
            current_m,
            current_tau,
            feature_bank_view,
            gate_reference_only=gate_reference_only,
        )
        candidate_feature_bank = _gate_feature_bank_metrics(
            candidate_gate,
            candidate_e,
            candidate_m,
            candidate_tau,
            feature_bank_view,
            gate_reference_only=gate_reference_only,
        )

    if gate_reference_only:
        current_family_ablations: list[dict[str, Any]] = []
        candidate_family_ablations: list[dict[str, Any]] = []
    else:
        current_family_ablations, candidate_family_ablations = (
            _predictive_upstream_family_ablations(
                context,
                gate,
                current_specs,
                candidate_specs,
                candidate_context=candidate_context,
                candidate_gate=candidate_gate,
                design=upstream_design,
                current_full_metrics=current_metrics,
                candidate_full_metrics=candidate_metrics,
                config=config,
            )
        )
    upstream_design.verify_content()
    if upstream_design.content_sha256 != upstream_design_sha256:  # pragma: no cover
        raise RuntimeError("conditional upstream design identity changed during ablations")

    # Compare both representations on one gate-local scale.  Normalizing each
    # representation by its own zero-effect loss would let a candidate improve
    # the ratio merely by worsening its outcome nuisance denominator.  The
    # current representation's zero-effect loss is fixed before the candidate
    # is scored and is therefore the shared observable reference.
    reference_zero_effect_r_loss = max(
        float(current_metrics["effect"]["zero_effect_r_loss"]),
        1e-12,
    )

    def score(metrics: Mapping[str, Any], complexity: Mapping[str, int]) -> float:
        return float(
            metrics["effect"]["weighted_r_loss"] / reference_zero_effect_r_loss
            + config.contract_complexity_penalty * complexity["contract_count"]
            + config.encoded_column_complexity_penalty * complexity["encoded_column_count"]
        )

    current_score = score(current_metrics, current_complexity)
    candidate_score = score(candidate_metrics, candidate_complexity)
    reasons: list[str] = []
    guards: dict[str, Any] = {
        "evaluation_configuration": _causal_review_config_payload(config),
        "conditional_upstream_design": {
            "content_sha256": upstream_design_sha256,
            "shared_identically_by_current_and_candidate": True,
            "upstream_review_policy": upstream_review_policy,
            "context_numeric_standardization_fit_on_spent_context_only": (
                not gate_reference_only
            ),
            "calibrated_sources_routed_to_effect_regression": (
                source_view is not None and not gate_reference_only
            ),
            "role_aware_raw_features_routed_to_matching_regressions": (
                feature_bank_view is not None and not gate_reference_only
            ),
            "upstream_gate_values_used_as_training_or_prediction_covariates": False
            if gate_reference_only
            else bool(source_view is not None or feature_bank_view is not None),
            "gate_views_used_only_as_post_fit_reference_diagnostics": gate_reference_only,
            "raw_feature_values_used_directly_as_treatment_effects": False,
        },
    }
    maximum_candidate_score = current_score - config.minimum_score_improvement
    observed_score_improvement = current_score - candidate_score
    # A same-complexity, same-loss rewrite is not evidence of improvement.
    # Simpler candidates can still win at identical raw R-loss through the
    # precommitted complexity penalty.
    objective_passed = bool(observed_score_improvement > config.minimum_score_improvement + 1e-12)
    guards["penalized_relative_r_loss"] = {
        "passed": objective_passed,
        "maximum_candidate_score": maximum_candidate_score,
        "observed_score_improvement": observed_score_improvement,
        "minimum_required_score_improvement": config.minimum_score_improvement,
        "reference_zero_effect_r_loss": reference_zero_effect_r_loss,
        "normalization_policy": "shared_current_representation_zero_effect_r_loss",
        "candidate_specific_denominator_used": False,
        "strictly_positive_improvement_required": True,
    }
    if not objective_passed:
        reasons.append("penalized_relative_r_loss_not_improved")

    nuisance_pairs = {
        "treatment_log_loss": (
            candidate_metrics["nuisance"]["treatment"]["log_loss"],
            current_metrics["nuisance"]["treatment"]["log_loss"],
        ),
        "outcome_loss": (
            candidate_metrics["nuisance"]["outcome"]["loss"],
            current_metrics["nuisance"]["outcome"]["loss"],
        ),
    }
    for name, (candidate_value, current_value) in nuisance_pairs.items():
        passed, maximum = _relative_upper_guard(
            candidate_value,
            current_value,
            config.nuisance_relative_tolerance,
        )
        guards[name] = {"passed": passed, "maximum_candidate_value": maximum}
        if not passed:
            reasons.append(f"{name}_guard_failed")

    if current_source is not None and candidate_source is not None:
        current_preservation = _finite_or_none(current_source.get("source_preservation_score"))
        candidate_preservation = _finite_or_none(candidate_source.get("source_preservation_score"))
        minimum_preservation = (
            None
            if current_preservation is None
            else max(
                0.0,
                current_preservation - config.source_preservation_tolerance,
            )
        )
        preservation_passed = bool(
            minimum_preservation is None
            or (
                candidate_preservation is not None
                and candidate_preservation + 1e-12 >= minimum_preservation
            )
        )
        guards["source_preservation"] = {
            "passed": preservation_passed,
            "correlation_measure": "mean_absolute_source_correlation",
            "current_score": current_preservation,
            "candidate_score": candidate_preservation,
            "minimum_candidate_score": minimum_preservation,
        }
        if not preservation_passed:
            reasons.append("source_preservation_guard_failed")

        current_source_rows = {
            (str(row.get("source_name")), str(row.get("source_kind"))): row
            for row in current_source.get("sources", [])
            if isinstance(row, Mapping)
        }
        candidate_source_rows = {
            (str(row.get("source_name")), str(row.get("source_kind"))): row
            for row in candidate_source.get("sources", [])
            if isinstance(row, Mapping)
        }
        directional_rows: dict[str, Any] = {}
        source_identities_match = set(current_source_rows) == set(candidate_source_rows)
        for source_key in sorted(set(current_source_rows) | set(candidate_source_rows)):
            current_row = current_source_rows.get(source_key)
            candidate_row = candidate_source_rows.get(source_key)
            current_correlation = _finite_or_none(
                None if current_row is None else current_row.get("tau_correlation")
            )
            candidate_correlation = _finite_or_none(
                None if candidate_row is None else candidate_row.get("tau_correlation")
            )
            current_absolute_correlation = _finite_or_none(
                None if current_row is None else current_row.get("absolute_tau_correlation")
            )
            candidate_absolute_correlation = _finite_or_none(
                None if candidate_row is None else candidate_row.get("absolute_tau_correlation")
            )
            minimum_absolute_correlation = (
                None
                if current_absolute_correlation is None
                else max(
                    0.0,
                    current_absolute_correlation - config.source_preservation_tolerance,
                )
            )
            same_direction = bool(
                current_correlation is None
                or (
                    candidate_correlation is not None
                    and (
                        current_correlation == 0.0
                        or candidate_correlation == 0.0
                        or np.sign(candidate_correlation) == np.sign(current_correlation)
                    )
                )
            )
            magnitude_preserved = bool(
                minimum_absolute_correlation is None
                or (
                    candidate_absolute_correlation is not None
                    and candidate_absolute_correlation + 1e-12 >= minimum_absolute_correlation
                )
            )
            passed = bool(
                current_row is not None
                and candidate_row is not None
                and same_direction
                and magnitude_preserved
            )
            directional_rows[f"{source_key[1]}::{source_key[0]}"] = {
                "passed": passed,
                "current_signed_correlation": current_correlation,
                "candidate_signed_correlation": candidate_correlation,
                "current_absolute_correlation": current_absolute_correlation,
                "candidate_absolute_correlation": candidate_absolute_correlation,
                "minimum_candidate_absolute_correlation": minimum_absolute_correlation,
                "same_direction": same_direction,
                "magnitude_preserved": magnitude_preserved,
            }
        direction_passed = bool(
            source_identities_match
            and all(bool(row["passed"]) for row in directional_rows.values())
        )
        guards["source_direction_preservation"] = {
            "passed": direction_passed,
            "correlation_measure": "absolute_magnitude_with_same_direction",
            "source_identities_match": source_identities_match,
            "by_source": directional_rows,
        }
        if not direction_passed:
            reasons.append("source_direction_guard_failed")

        source_r_loss_metric = (
            "mean_source_gate_reference_r_loss_ratio"
            if gate_reference_only
            else "mean_source_context_r_loss_ratio"
        )
        context_passed, maximum_context_ratio = _relative_upper_guard(
            candidate_source.get(source_r_loss_metric),
            current_source.get(source_r_loss_metric),
            config.source_context_r_loss_relative_tolerance,
        )
        source_r_loss_guard_name = (
            "source_gate_reference_r_loss"
            if gate_reference_only
            else "source_context_r_loss"
        )
        guards[source_r_loss_guard_name] = {
            "passed": context_passed,
            "maximum_candidate_value": maximum_context_ratio,
            "diagnostic_scope": (
                "untouched_gate_reference"
                if gate_reference_only
                else "conditional_context_and_gate"
            ),
        }
        if not context_passed:
            reasons.append(f"{source_r_loss_guard_name}_guard_failed")

    if current_feature_bank is not None and candidate_feature_bank is not None:
        current_role_scores = current_feature_bank["preservation_score_by_consumer_role"]
        candidate_role_scores = candidate_feature_bank["preservation_score_by_consumer_role"]
        role_guards: dict[str, Any] = {}
        for role in sorted(_GATE_FEATURE_BANK_ROLES):
            current_value = _finite_or_none(current_role_scores.get(role))
            candidate_value = _finite_or_none(candidate_role_scores.get(role))
            minimum = (
                None
                if current_value is None
                else max(
                    0.0,
                    current_value - config.feature_bank_preservation_tolerance,
                )
            )
            passed = bool(
                minimum is None
                or (candidate_value is not None and candidate_value + 1e-12 >= minimum)
            )
            role_guards[role] = {
                "passed": passed,
                "minimum_candidate_score": minimum,
            }
            if not passed:
                reasons.append(f"feature_bank_{role}_preservation_guard_failed")

        def family_rows_by_identity(
            evaluation: Mapping[str, Any],
        ) -> tuple[dict[tuple[str, str], Mapping[str, Any]], bool]:
            raw_rows = evaluation.get(
                "preservation_by_source_kind_and_consumer_role",
                [],
            )
            if not isinstance(raw_rows, list):
                return {}, False
            indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
            valid = True
            for raw_row in raw_rows:
                if not isinstance(raw_row, Mapping):
                    valid = False
                    continue
                kind = str(raw_row.get("source_kind") or "").strip()
                role = str(raw_row.get("consumer_role") or "").strip()
                identity = (kind, role)
                if not kind or role not in _GATE_FEATURE_BANK_ROLES or identity in indexed:
                    valid = False
                    continue
                indexed[identity] = raw_row
            return indexed, bool(valid and len(indexed) == len(raw_rows))

        current_families, current_families_valid = family_rows_by_identity(current_feature_bank)
        candidate_families, candidate_families_valid = family_rows_by_identity(
            candidate_feature_bank
        )
        family_identities_match = bool(
            current_families_valid
            and candidate_families_valid
            and set(current_families) == set(candidate_families)
        )
        family_guards: list[dict[str, Any]] = []
        for kind, role in sorted(set(current_families) | set(candidate_families)):
            current_row = current_families.get((kind, role))
            candidate_row = candidate_families.get((kind, role))
            current_value = _finite_or_none(
                None
                if current_row is None
                else current_row.get("mean_absolute_role_matched_prediction_correlation")
            )
            candidate_value = _finite_or_none(
                None
                if candidate_row is None
                else candidate_row.get("mean_absolute_role_matched_prediction_correlation")
            )
            current_feature_count = (
                None if current_row is None else current_row.get("feature_count")
            )
            candidate_feature_count = (
                None if candidate_row is None else candidate_row.get("feature_count")
            )
            feature_count_matches = bool(
                isinstance(current_feature_count, int)
                and not isinstance(current_feature_count, bool)
                and current_feature_count > 0
                and candidate_feature_count == current_feature_count
            )
            minimum = (
                None
                if current_value is None
                else max(
                    0.0,
                    current_value - config.feature_bank_preservation_tolerance,
                )
            )
            passed = bool(
                family_identities_match
                and current_row is not None
                and candidate_row is not None
                and feature_count_matches
                and (
                    minimum is None
                    or (candidate_value is not None and candidate_value + 1e-12 >= minimum)
                )
            )
            family_guards.append(
                {
                    "source_kind": kind,
                    "consumer_role": role,
                    "passed": passed,
                    "feature_count_matches": feature_count_matches,
                    "current_feature_count": current_feature_count,
                    "candidate_feature_count": candidate_feature_count,
                    "current_preservation_score": current_value,
                    "candidate_preservation_score": candidate_value,
                    "minimum_candidate_score": minimum,
                }
            )
        family_preservation_passed = bool(
            family_identities_match
            and family_guards
            and all(bool(row["passed"]) for row in family_guards)
        )
        if not family_preservation_passed:
            reasons.append("feature_bank_family_preservation_guard_failed")
        role_preservation_passed = all(bool(row["passed"]) for row in role_guards.values())
        guards["feature_bank_preservation"] = {
            "passed": role_preservation_passed and family_preservation_passed,
            "by_consumer_role": role_guards,
            "family_identities_match": family_identities_match,
            "by_source_kind_and_consumer_role": family_guards,
        }

    if gate_reference_only:
        predictive_ablation_status = "unavailable_by_design"
        guards["upstream_predictive_family_ablations"] = {
            "status": predictive_ablation_status,
            "passed": None,
            "by_family": [],
            "predictive_refit_performed": False,
            "gate_decision_constraint_applied": False,
            "reason": (
                "gate-only references have no honest spent-context nested OOF "
                "matrix and therefore cannot be training covariates or deleted "
                "from a predictive fit"
            ),
            "correlation_deletion_used_as_predictive_ablation": False,
        }
    else:
        predictive_ablation_status = "available"
        current_ablation_by_key = {
            (
                str(row["input_kind"]),
                str(row["source_kind"]),
                str(row["consumer_role"]),
            ): row
            for row in current_family_ablations
        }
        candidate_ablation_by_key = {
            (
                str(row["input_kind"]),
                str(row["source_kind"]),
                str(row["consumer_role"]),
            ): row
            for row in candidate_family_ablations
        }
        ablation_identities_match = set(current_ablation_by_key) == set(
            candidate_ablation_by_key
        )
        predictive_ablation_guards: list[dict[str, Any]] = []
        for key in sorted(set(current_ablation_by_key) | set(candidate_ablation_by_key)):
            current_row = current_ablation_by_key.get(key)
            candidate_row = candidate_ablation_by_key.get(key)
            current_delta = _finite_or_none(
                None
                if current_row is None
                else current_row.get("weighted_r_loss_delta_when_removed")
            )
            candidate_delta = _finite_or_none(
                None
                if candidate_row is None
                else candidate_row.get("weighted_r_loss_delta_when_removed")
            )
            current_importance = (
                None
                if current_delta is None
                else max(0.0, current_delta / reference_zero_effect_r_loss)
            )
            candidate_importance = (
                None
                if candidate_delta is None
                else max(0.0, candidate_delta / reference_zero_effect_r_loss)
            )
            tolerance = (
                config.source_preservation_tolerance
                if key[0] == "calibrated_effect_source"
                else config.feature_bank_preservation_tolerance
            )
            minimum = (
                None
                if current_importance is None
                else max(0.0, current_importance - tolerance)
            )
            passed = bool(
                ablation_identities_match
                and current_row is not None
                and candidate_row is not None
                and current_row.get("upstream_columns_removed")
                == candidate_row.get("upstream_columns_removed")
                and minimum is not None
                and candidate_importance is not None
                and candidate_importance + 1e-12 >= minimum
            )
            predictive_ablation_guards.append(
                {
                    "input_kind": key[0],
                    "source_kind": key[1],
                    "consumer_role": key[2],
                    "passed": passed,
                    "current_normalized_predictive_importance": current_importance,
                    "candidate_normalized_predictive_importance": candidate_importance,
                    "minimum_candidate_normalized_predictive_importance": minimum,
                    "tolerance": tolerance,
                    "predictive_refit_performed": True,
                }
            )
        predictive_ablation_passed = bool(
            ablation_identities_match
            and all(bool(row["passed"]) for row in predictive_ablation_guards)
        )
        guards["upstream_predictive_family_ablations"] = {
            "status": predictive_ablation_status,
            "passed": predictive_ablation_passed,
            "family_identities_match": ablation_identities_match,
            "by_family": predictive_ablation_guards,
            "ablation_method": "delete_exact_family_then_refit_all_matched_models",
            "shared_reference_zero_effect_r_loss": reference_zero_effect_r_loss,
            "correlation_deletion_used_as_predictive_ablation": False,
        }
        if not predictive_ablation_passed:
            reasons.append("upstream_predictive_family_ablation_guard_failed")

    current_payload = {
        "metrics": current_metrics,
        "complexity": current_complexity,
        "penalized_relative_r_loss_score": current_score,
        "source_signal_evaluation": current_source,
        "feature_bank_evaluation": current_feature_bank,
        "upstream_predictive_family_ablations": current_family_ablations,
        "upstream_predictive_family_ablation_status": predictive_ablation_status,
        "conditional_upstream_design_sha256": upstream_design_sha256,
    }
    candidate_payload = {
        "metrics": candidate_metrics,
        "complexity": candidate_complexity,
        "penalized_relative_r_loss_score": candidate_score,
        "source_signal_evaluation": candidate_source,
        "feature_bank_evaluation": candidate_feature_bank,
        "upstream_predictive_family_ablations": candidate_family_ablations,
        "upstream_predictive_family_ablation_status": predictive_ablation_status,
        "conditional_upstream_design_sha256": upstream_design_sha256,
    }
    decision_content = {
        "schema_version": POST_EXTRACTION_GATE_DECISION_SCHEMA_VERSION,
        "accepted": not reasons,
        "reasons": reasons,
        "current": current_payload,
        "candidate": candidate_payload,
        "guards": guards,
    }
    return GateAcceptanceDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        current=_detached(current_payload),
        candidate=_detached(candidate_payload),
        guards=_detached(guards),
        decision_sha256=_content_sha256(decision_content),
    )


__all__ = [
    "AppliedReviewOperations",
    "CausalReviewConfig",
    "CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY",
    "GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY",
    "GateAcceptanceDecision",
    "GateFeatureBankView",
    "GateSourceSignalView",
    "ObservableCausalRows",
    "OUTCOME_NUISANCE_FEATURE_ROLE",
    "POST_EXTRACTION_CAUSAL_DIAGNOSTIC_SCHEMA_VERSION",
    "POST_EXTRACTION_GATE_DECISION_SCHEMA_VERSION",
    "POST_EXTRACTION_QUALITY_SCHEMA_VERSION",
    "POST_EXTRACTION_REVIEW_PROMPT_VERSION",
    "POST_EXTRACTION_REVIEW_RESPONSE_SCHEMA_VERSION",
    "PROPENSITY_NUISANCE_FEATURE_ROLE",
    "RAW_UNCALIBRATED_FEATURE_SOURCE_KINDS",
    "ReviewOperation",
    "UNCALIBRATED_EFFECT_MODIFIER_ROLE",
    "ValidatedReviewResponse",
    "apply_post_extraction_review_operations",
    "build_extraction_quality_diagnostics",
    "build_causal_review_diagnostics",
    "build_post_extraction_review_repair_prompt",
    "build_redundancy_diagnostics",
    "collect_post_extraction_diagnostic_ids",
    "collect_post_extraction_diagnostic_targets",
    "extraction_semantics_sha256",
    "evaluate_untouched_gate_acceptance",
    "post_extraction_review_response_issues",
    "render_post_extraction_review_prompt",
    "validate_post_extraction_review_response",
]
