"""Compile authenticated hierarchical discovery into legacy feature contracts.

The hierarchical discovery workflow deliberately uses richer, observational role
routing than the legacy extraction/modeling stack.  This module is the narrow,
deterministic compatibility boundary between those systems.  In particular,
mapping an adjustment role to the legacy ``confounder`` slot means only that the
feature is supplied to the downstream ``W`` adjustment matrix; it is not a claim
that the feature is a causal confounder.

No accepted feature disappears at this boundary.  Features that have no
adjustment or effect-modifier role receive an authenticated non-model
disposition instead of being silently forced into a computational role.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
    AS_DOCUMENTED_UNIT,
    MECHANICAL_MENTION_CATEGORIES,
    canonical_json,
    content_sha256,
    extraction_vocabulary_grounding_policy,
)
from .all_evidence_fusion import CandidateContract
from .hierarchical_all_architecture_discovery import (
    COMPLETED_HIERARCHICAL_DISCOVERY_VERSION,
    CompletedHierarchicalDiscovery,
    RoutedIntegratedFeature,
)

HIERARCHICAL_DISCOVERY_COMPILER_VERSION = "hierarchical_discovery_compiler_v3"
COMPILED_HIERARCHICAL_FEATURE_REGISTRY_VERSION = "compiled_hierarchical_feature_registry_v2"
COMPILED_FEATURE_DISPOSITION_VERSION = "compiled_hierarchical_feature_disposition_v1"
AUTHENTICATED_MODELED_CONTRACT_VERSION = "authenticated_hierarchical_contract_v1"
DEFAULT_MAX_CANDIDATES = 256

MODELED_DISPOSITION = "modeled_in_legacy_causal_forest"
NON_MODEL_TREATMENT_ONLY = "non_model_treatment_prediction_support_only"
NON_MODEL_EXTRACTION_ONLY = "non_model_extraction_definition_support_only"
NON_MODEL_TREATMENT_AND_EXTRACTION_ONLY = "non_model_treatment_and_extraction_support_only"
NON_MODEL_ROLELESS = "non_model_no_adjustment_or_effect_modifier_role"

DOWNSTREAM_ADJUSTMENT_SLOT_AUDIT = (
    "Legacy role 'confounder' is a downstream computational W adjustment-slot "
    "mapping only; it is not a causal-confounder claim."
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SNAKE_CASE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
_DEFINITION_KEYS = frozenset(
    {
        "feature_name",
        "measurement",
        "representation",
        "aliases",
        "distinguish_from",
        "missing_or_ambiguous",
        "supporting_evidence_ids",
    }
)
_DEFINITION_OPTIONAL_KEYS = frozenset({"vocabulary_normalization_audit"})
_REPRESENTATION_KEYS = frozenset({"kind", "unit", "categories"})


def _sha(value: Any) -> str:
    return content_sha256(value)


def _fresh(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _implementation_file_sha256() -> str:
    """Bind compiler outputs to the exact implementation reviewed offline."""

    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _require_sha256(value: str, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _require_string(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{label} cannot be empty")
    return value


def _string_list(
    value: Any,
    *,
    label: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    if not allow_empty and not value:
        raise ValueError(f"{label} cannot be empty")
    result = tuple(
        _require_string(item, label=f"{label}[{index}]") for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{label} cannot contain exact duplicates")
    return result


def _compiler_identity(*, max_candidates: int) -> dict[str, Any]:
    return {
        "schema_version": HIERARCHICAL_DISCOVERY_COMPILER_VERSION,
        "implementation": {
            "module": "oci.inference.hierarchical_discovery_compiler",
            "file_sha256": _implementation_file_sha256(),
        },
        "source_schema_version": COMPLETED_HIERARCHICAL_DISCOVERY_VERSION,
        "output_schema_version": COMPILED_HIERARCHICAL_FEATURE_REGISTRY_VERSION,
        "max_candidates": max_candidates,
        "role_mapping": {
            "effect_modifier": "legacy effect_modifier role",
            "any_adjustment_role": "legacy confounder computational W slot",
            "adjustment_mapping_disclaimer": DOWNSTREAM_ADJUSTMENT_SLOT_AUDIT,
            "roleless": "authenticated explicit non-model disposition",
        },
        "representation_policy": {
            "unresolved": "fail_closed",
            "continuous_categories": "must_be_empty",
            "categorical_categories": (
                "CandidateContract concrete distinct nonempty validation with at least "
                "two values and no compiler category-count cap"
            ),
            "extraction_vocabulary_grounding": extraction_vocabulary_grounding_policy(),
        },
        "candidate_overflow_policy": "fail_closed_without_truncation",
    }


def _contract_identity(
    *, extraction_spec: Mapping[str, Any], source_families: Sequence[str]
) -> dict[str, Any]:
    return {
        "schema_version": AUTHENTICATED_MODELED_CONTRACT_VERSION,
        "extraction_spec": _fresh(extraction_spec),
        "source_families": list(source_families),
    }


@dataclass(frozen=True)
class AuthenticatedModeledCandidate:
    """One immutable CandidateContract record with independent hashes."""

    canonical_name: str
    source_families: tuple[str, ...]
    extraction_spec_sha256: str
    candidate_contract_sha256: str
    record_sha256: str
    _extraction_spec_json: str = field(repr=False)

    def __post_init__(self) -> None:
        self.validate_authentication()

    @classmethod
    def create(
        cls,
        *,
        extraction_spec: Mapping[str, Any],
        source_families: Sequence[str],
    ) -> "AuthenticatedModeledCandidate":
        contract = CandidateContract(
            extraction_spec,
            source_families=source_families,
        )
        spec = contract.extraction_spec
        families = tuple(contract.source_families)
        contract_identity = _contract_identity(
            extraction_spec=spec,
            source_families=families,
        )
        record_without_sha = {
            **contract_identity,
            "canonical_name": spec["name"],
            "extraction_spec_sha256": _sha(spec),
            "candidate_contract_sha256": _sha(contract_identity),
        }
        return cls(
            canonical_name=str(spec["name"]),
            source_families=families,
            extraction_spec_sha256=_sha(spec),
            candidate_contract_sha256=_sha(contract_identity),
            record_sha256=_sha(record_without_sha),
            _extraction_spec_json=canonical_json(spec),
        )

    @property
    def extraction_spec(self) -> dict[str, Any]:
        self.validate_authentication()
        return json.loads(self._extraction_spec_json)

    @property
    def candidate_contract(self) -> CandidateContract:
        self.validate_authentication()
        return CandidateContract(
            json.loads(self._extraction_spec_json),
            source_families=self.source_families,
        )

    def _identity_without_record_sha(self) -> dict[str, Any]:
        spec = json.loads(self._extraction_spec_json)
        return {
            **_contract_identity(
                extraction_spec=spec,
                source_families=self.source_families,
            ),
            "canonical_name": self.canonical_name,
            "extraction_spec_sha256": self.extraction_spec_sha256,
            "candidate_contract_sha256": self.candidate_contract_sha256,
        }

    def validate_authentication(self) -> None:
        if _SNAKE_CASE.fullmatch(self.canonical_name) is None:
            raise ValueError("modeled candidate canonical_name must be lower snake_case")
        if not isinstance(self.source_families, tuple):
            raise TypeError("modeled candidate source_families must be a tuple")
        if not self.source_families or len(self.source_families) != len(set(self.source_families)):
            raise ValueError("modeled candidate source_families must be non-empty and unique")
        if not set(self.source_families) <= ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("modeled candidate cites an inactive source family")
        for label, value in (
            ("extraction_spec_sha256", self.extraction_spec_sha256),
            ("candidate_contract_sha256", self.candidate_contract_sha256),
            ("record_sha256", self.record_sha256),
        ):
            _require_sha256(value, label=label)
        try:
            raw_spec = json.loads(self._extraction_spec_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("modeled candidate extraction spec is not authenticated JSON") from exc
        contract = CandidateContract(raw_spec, source_families=self.source_families)
        spec = contract.extraction_spec
        if spec["name"] != self.canonical_name:
            raise ValueError("modeled candidate name differs from its extraction spec")
        if self.extraction_spec_sha256 != _sha(spec):
            raise ValueError("extraction_spec_sha256 does not authenticate the spec")
        contract_identity = _contract_identity(
            extraction_spec=spec,
            source_families=self.source_families,
        )
        if self.candidate_contract_sha256 != _sha(contract_identity):
            raise ValueError("candidate_contract_sha256 does not authenticate the contract")
        if self.record_sha256 != _sha(self._identity_without_record_sha()):
            raise ValueError("record_sha256 does not authenticate the modeled candidate")

    def as_dict(self) -> dict[str, Any]:
        self.validate_authentication()
        return {
            **self._identity_without_record_sha(),
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True)
class CompiledFeatureDisposition:
    """Authenticated modeled or non-model outcome for one accepted feature."""

    canonical_name: str
    source_families: tuple[str, ...]
    extraction_definition_sha256: str
    adjustment_roles: tuple[str, ...]
    effect_modifier: bool
    treatment_prediction_support: bool
    extraction_definition_support: bool
    disposition: str
    legacy_roles: tuple[str, ...]
    adjustment_slot_audit: str
    modeled_contract_record_sha256: str
    reason: str
    disposition_sha256: str

    def __post_init__(self) -> None:
        self.validate_authentication()

    @classmethod
    def create(
        cls,
        *,
        routed: RoutedIntegratedFeature,
        extraction_definition_sha256: str,
        disposition: str,
        legacy_roles: Sequence[str],
        modeled_contract_record_sha256: str,
        reason: str,
    ) -> "CompiledFeatureDisposition":
        routing = routed.role_routing
        roles = tuple(legacy_roles)
        adjustment_audit = DOWNSTREAM_ADJUSTMENT_SLOT_AUDIT if routing.adjustment_roles else ""
        fields = {
            "schema_version": COMPILED_FEATURE_DISPOSITION_VERSION,
            "canonical_name": routed.feature.canonical_name,
            "source_families": list(routed.feature.source_families),
            "extraction_definition_sha256": extraction_definition_sha256,
            "adjustment_roles": list(routing.adjustment_roles),
            "effect_modifier": routing.effect_modifier,
            "treatment_prediction_support": routing.treatment_prediction_support,
            "extraction_definition_support": routing.extraction_definition_support,
            "disposition": disposition,
            "legacy_roles": list(roles),
            "adjustment_slot_audit": adjustment_audit,
            "modeled_contract_record_sha256": modeled_contract_record_sha256,
            "reason": reason,
        }
        return cls(
            canonical_name=routed.feature.canonical_name,
            source_families=routed.feature.source_families,
            extraction_definition_sha256=extraction_definition_sha256,
            adjustment_roles=routing.adjustment_roles,
            effect_modifier=routing.effect_modifier,
            treatment_prediction_support=routing.treatment_prediction_support,
            extraction_definition_support=routing.extraction_definition_support,
            disposition=disposition,
            legacy_roles=roles,
            adjustment_slot_audit=adjustment_audit,
            modeled_contract_record_sha256=modeled_contract_record_sha256,
            reason=reason,
            disposition_sha256=_sha(fields),
        )

    @property
    def modeled(self) -> bool:
        return self.disposition == MODELED_DISPOSITION

    def _identity_without_sha(self) -> dict[str, Any]:
        return {
            "schema_version": COMPILED_FEATURE_DISPOSITION_VERSION,
            "canonical_name": self.canonical_name,
            "source_families": list(self.source_families),
            "extraction_definition_sha256": self.extraction_definition_sha256,
            "adjustment_roles": list(self.adjustment_roles),
            "effect_modifier": self.effect_modifier,
            "treatment_prediction_support": self.treatment_prediction_support,
            "extraction_definition_support": self.extraction_definition_support,
            "disposition": self.disposition,
            "legacy_roles": list(self.legacy_roles),
            "adjustment_slot_audit": self.adjustment_slot_audit,
            "modeled_contract_record_sha256": self.modeled_contract_record_sha256,
            "reason": self.reason,
        }

    def validate_authentication(self) -> None:
        if _SNAKE_CASE.fullmatch(self.canonical_name) is None:
            raise ValueError("disposition canonical_name must be lower snake_case")
        if not isinstance(self.source_families, tuple):
            raise TypeError("disposition source_families must be a tuple")
        if not self.source_families or len(self.source_families) != len(set(self.source_families)):
            raise ValueError("disposition source_families must be non-empty and unique")
        if not set(self.source_families) <= ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("disposition cites an inactive source family")
        if not isinstance(self.adjustment_roles, tuple):
            raise TypeError("adjustment_roles must be a tuple")
        if len(self.adjustment_roles) != len(set(self.adjustment_roles)) or any(
            not isinstance(value, str) or not value.strip() for value in self.adjustment_roles
        ):
            raise ValueError("adjustment_roles must contain unique non-empty strings")
        for label, value in (
            ("effect_modifier", self.effect_modifier),
            ("treatment_prediction_support", self.treatment_prediction_support),
            ("extraction_definition_support", self.extraction_definition_support),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{label} must be boolean")
        if not isinstance(self.legacy_roles, tuple):
            raise TypeError("legacy_roles must be a tuple")
        _require_sha256(
            self.extraction_definition_sha256,
            label="extraction_definition_sha256",
        )
        if self.modeled:
            if not self.legacy_roles:
                raise ValueError("modeled disposition requires at least one legacy role")
            if self.disposition != MODELED_DISPOSITION:
                raise ValueError("modeled disposition identifier is invalid")
            _require_sha256(
                self.modeled_contract_record_sha256,
                label="modeled_contract_record_sha256",
            )
        else:
            expected_non_model = {
                (True, False): NON_MODEL_TREATMENT_ONLY,
                (False, True): NON_MODEL_EXTRACTION_ONLY,
                (True, True): NON_MODEL_TREATMENT_AND_EXTRACTION_ONLY,
                (False, False): NON_MODEL_ROLELESS,
            }[
                (
                    self.treatment_prediction_support,
                    self.extraction_definition_support,
                )
            ]
            if self.disposition != expected_non_model:
                raise ValueError("non-model disposition differs from deterministic routing")
            if self.legacy_roles or self.modeled_contract_record_sha256:
                raise ValueError("non-model dispositions cannot carry a modeled contract or role")
        expected_roles = tuple(
            role
            for role, enabled in (
                ("confounder", bool(self.adjustment_roles)),
                ("effect_modifier", self.effect_modifier),
            )
            if enabled
        )
        if self.legacy_roles != expected_roles:
            raise ValueError("legacy roles differ from deterministic role routing")
        expected_audit = DOWNSTREAM_ADJUSTMENT_SLOT_AUDIT if self.adjustment_roles else ""
        if self.adjustment_slot_audit != expected_audit:
            raise ValueError("adjustment-slot audit differs from the closed mapping policy")
        _require_string(self.reason, label="disposition reason")
        _require_sha256(self.disposition_sha256, label="disposition_sha256")
        if self.disposition_sha256 != _sha(self._identity_without_sha()):
            raise ValueError("disposition_sha256 does not authenticate the disposition")

    def audit(self) -> dict[str, Any]:
        self.validate_authentication()
        return {
            **self._identity_without_sha(),
            "modeled": self.modeled,
            "disposition_sha256": self.disposition_sha256,
        }


@dataclass(frozen=True)
class CompiledHierarchicalFeatureRegistry:
    """Immutable, content-addressed output consumed by legacy modeling code."""

    source_completion_sha256: str
    compiler_sha256: str
    max_candidates: int
    modeled_candidates: tuple[AuthenticatedModeledCandidate, ...]
    dispositions: tuple[CompiledFeatureDisposition, ...]
    registry_sha256: str
    _compiler_identity_json: str = field(repr=False)

    def __post_init__(self) -> None:
        self.validate_authentication()

    @property
    def compiler_identity(self) -> dict[str, Any]:
        self.validate_authentication()
        return json.loads(self._compiler_identity_json)

    @property
    def specs(self) -> list[dict[str, Any]]:
        self.validate_authentication()
        return [candidate.extraction_spec for candidate in self.modeled_candidates]

    @property
    def contracts(self) -> tuple[CandidateContract, ...]:
        self.validate_authentication()
        return tuple(candidate.candidate_contract for candidate in self.modeled_candidates)

    @property
    def disposition_audit(self) -> list[dict[str, Any]]:
        self.validate_authentication()
        return [disposition.audit() for disposition in self.dispositions]

    def _identity_without_registry_sha(self) -> dict[str, Any]:
        return {
            "schema_version": COMPILED_HIERARCHICAL_FEATURE_REGISTRY_VERSION,
            "source_completion_sha256": self.source_completion_sha256,
            "compiler_identity": json.loads(self._compiler_identity_json),
            "compiler_sha256": self.compiler_sha256,
            "max_candidates": self.max_candidates,
            "modeled_candidates": [candidate.as_dict() for candidate in self.modeled_candidates],
            "dispositions": [disposition.audit() for disposition in self.dispositions],
        }

    def validate_authentication(self) -> None:
        _require_sha256(
            self.source_completion_sha256,
            label="source_completion_sha256",
        )
        _require_sha256(self.compiler_sha256, label="compiler_sha256")
        _require_sha256(self.registry_sha256, label="registry_sha256")
        if isinstance(self.max_candidates, bool) or not isinstance(self.max_candidates, int):
            raise TypeError("max_candidates must be an integer")
        if self.max_candidates < 0:
            raise ValueError("max_candidates cannot be negative")
        if not isinstance(self.modeled_candidates, tuple):
            raise TypeError("modeled_candidates must be an immutable tuple")
        if not isinstance(self.dispositions, tuple):
            raise TypeError("dispositions must be an immutable tuple")
        try:
            observed_identity = json.loads(self._compiler_identity_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("compiler identity is not authenticated JSON") from exc
        expected_identity = _compiler_identity(max_candidates=self.max_candidates)
        if observed_identity != expected_identity:
            raise ValueError("compiler identity differs from the closed compiler policy")
        if self.compiler_sha256 != _sha(observed_identity):
            raise ValueError("compiler_sha256 does not authenticate compiler identity")
        if len(self.modeled_candidates) > self.max_candidates:
            raise ValueError("modeled candidate count exceeds max_candidates")
        names = tuple(candidate.canonical_name for candidate in self.modeled_candidates)
        if len(names) != len(set(names)):
            raise ValueError("modeled candidate names cannot contain duplicates")
        for candidate in self.modeled_candidates:
            if not isinstance(candidate, AuthenticatedModeledCandidate):
                raise TypeError("modeled_candidates contains a non-candidate record")
            candidate.validate_authentication()
        disposition_names = tuple(row.canonical_name for row in self.dispositions)
        if len(disposition_names) != len(set(disposition_names)):
            raise ValueError("disposition names cannot contain duplicates")
        for disposition in self.dispositions:
            if not isinstance(disposition, CompiledFeatureDisposition):
                raise TypeError("dispositions contains a non-disposition record")
            disposition.validate_authentication()
        record_by_name = {row.canonical_name: row for row in self.modeled_candidates}
        modeled_dispositions = {row.canonical_name: row for row in self.dispositions if row.modeled}
        if set(record_by_name) != set(modeled_dispositions):
            raise ValueError("modeled records and modeled dispositions differ")
        for name, disposition in modeled_dispositions.items():
            record = record_by_name[name]
            if disposition.modeled_contract_record_sha256 != record.record_sha256:
                raise ValueError("modeled disposition cites the wrong contract record")
            if disposition.source_families != record.source_families:
                raise ValueError("modeled disposition and contract source families differ")
            if record.candidate_contract.extraction_spec["roles"] != list(disposition.legacy_roles):
                raise ValueError("modeled disposition and candidate contract roles differ")
        if self.registry_sha256 != _sha(self._identity_without_registry_sha()):
            raise ValueError("registry_sha256 does not authenticate the compiled registry")

    def as_dict(self) -> dict[str, Any]:
        self.validate_authentication()
        return {
            **self._identity_without_registry_sha(),
            "registry_sha256": self.registry_sha256,
        }


def _validate_definition(*, routed: RoutedIntegratedFeature, definition: Any) -> dict[str, Any]:
    if not isinstance(definition, Mapping):
        raise TypeError("authenticated extraction definition must be one JSON object")
    if not _DEFINITION_KEYS <= set(definition) <= (_DEFINITION_KEYS | _DEFINITION_OPTIONAL_KEYS):
        raise ValueError("authenticated extraction definition has an unexpected shape")
    detached = _fresh(definition)
    name = _require_string(detached["feature_name"], label="feature_name")
    if name != routed.feature.canonical_name:
        raise ValueError("extraction definition does not preserve the canonical name")
    if _SNAKE_CASE.fullmatch(name) is None:
        raise ValueError("canonical feature name must be lower snake_case")
    _require_string(detached["measurement"], label="measurement")
    representation = detached["representation"]
    if not isinstance(representation, Mapping) or set(representation) != _REPRESENTATION_KEYS:
        raise ValueError("extraction definition representation has an unexpected shape")
    kind = representation["kind"]
    if kind not in {"continuous", "categorical", "unresolved"}:
        raise ValueError("extraction definition representation.kind is invalid")
    unit = _require_string(representation["unit"], label="representation.unit", allow_empty=True)
    categories = _string_list(
        representation["categories"],
        label="representation.categories",
    )
    if kind == "unresolved":
        raise ValueError(f"feature {name!r} has unresolved extraction representation")
    if kind == "continuous":
        if categories:
            raise ValueError("continuous extraction representation cannot define categories")
        if not unit:
            raise ValueError("continuous extraction representation requires a unit statement")
    elif unit:
        raise ValueError("categorical extraction representation cannot define a unit")
    _string_list(detached["aliases"], label="aliases")
    _string_list(detached["distinguish_from"], label="distinguish_from")
    _require_string(detached["missing_or_ambiguous"], label="missing_or_ambiguous")
    if "vocabulary_normalization_audit" in detached:
        audit = detached["vocabulary_normalization_audit"]
        if not isinstance(audit, Mapping):
            raise TypeError("vocabulary_normalization_audit must be one JSON object")
        if audit.get("audit_version") != ("grounded_extraction_vocabulary_normalization_audit_v1"):
            raise ValueError("vocabulary_normalization_audit version is invalid")
    support = _string_list(
        detached["supporting_evidence_ids"],
        label="supporting_evidence_ids",
        allow_empty=False,
    )
    if set(support) != set(routed.feature.supporting_evidence_ids):
        raise ValueError("extraction definition does not preserve complete feature support")
    return detached


def _legacy_roles(routed: RoutedIntegratedFeature) -> tuple[str, ...]:
    roles: list[str] = []
    if routed.role_routing.adjustment_roles:
        roles.append("confounder")
    if routed.role_routing.effect_modifier:
        roles.append("effect_modifier")
    return tuple(roles)


def _legacy_description(definition: Mapping[str, Any]) -> str:
    representation = definition["representation"]
    parts = [str(definition["measurement"]).strip()]
    if representation["kind"] == "continuous":
        if representation["unit"] == AS_DOCUMENTED_UNIT:
            parts.append(
                "Extract a continuous value using the source-documented scale; "
                "as_documented is an extraction mechanic, not a clinical unit assertion."
            )
        else:
            parts.append(f"Extract a continuous value in {representation['unit']}.")
    elif tuple(representation["categories"]) == MECHANICAL_MENTION_CATEGORIES:
        parts.append(
            "Use exactly not_mentioned and mentioned as a document-observation encoding; "
            "these are extraction mechanics, not a clinical status ontology."
        )
    else:
        parts.append("Use exactly the declared categorical values.")
    aliases = definition["aliases"]
    if aliases:
        parts.append(f"Feature-name aliases: {', '.join(aliases)}.")
    distinctions = definition["distinguish_from"]
    if distinctions:
        parts.append(f"Distinguish this feature from: {', '.join(distinctions)}.")
    parts.append(str(definition["missing_or_ambiguous"]).strip())
    return " ".join(parts)


def _legacy_spec(
    *,
    routed: RoutedIntegratedFeature,
    definition: Mapping[str, Any],
    roles: Sequence[str],
) -> dict[str, Any]:
    representation = definition["representation"]
    spec: dict[str, Any] = {
        "name": routed.feature.canonical_name,
        "type": representation["kind"],
        "roles": list(roles),
        "description": _legacy_description(definition),
    }
    if representation["kind"] == "categorical":
        spec["categories"] = list(representation["categories"])
    return spec


def _non_model_disposition(routed: RoutedIntegratedFeature) -> tuple[str, str]:
    routing = routed.role_routing
    if routing.treatment_prediction_support and routing.extraction_definition_support:
        return (
            NON_MODEL_TREATMENT_AND_EXTRACTION_ONLY,
            "Retained for treatment-prediction and extraction-definition support, but it has "
            "no adjustment or effect-modifier role and therefore does not enter the causal "
            "forest feature slots.",
        )
    if routing.treatment_prediction_support:
        return (
            NON_MODEL_TREATMENT_ONLY,
            "Retained as treatment-prediction support only; it has no adjustment or "
            "effect-modifier role and therefore does not enter the causal forest feature slots.",
        )
    if routing.extraction_definition_support:
        return (
            NON_MODEL_EXTRACTION_ONLY,
            "Retained as extraction-definition support only; it has no adjustment or "
            "effect-modifier role and therefore does not enter the causal forest feature slots.",
        )
    return (
        NON_MODEL_ROLELESS,
        "Retained in the discovery audit, but deterministic routing assigned no adjustment or "
        "effect-modifier role, so it does not enter the causal forest feature slots.",
    )


@dataclass(frozen=True)
class HierarchicalDiscoveryCompiler:
    """Deterministic compiler with a content-addressed closed policy identity."""

    max_candidates: int = DEFAULT_MAX_CANDIDATES

    def __post_init__(self) -> None:
        if isinstance(self.max_candidates, bool) or not isinstance(self.max_candidates, int):
            raise TypeError("max_candidates must be an integer")
        if self.max_candidates < 0:
            raise ValueError("max_candidates cannot be negative")

    @property
    def identity(self) -> dict[str, Any]:
        return _compiler_identity(max_candidates=self.max_candidates)

    @property
    def identity_sha256(self) -> str:
        return _sha(self.identity)

    def compile(
        self, completed: CompletedHierarchicalDiscovery
    ) -> CompiledHierarchicalFeatureRegistry:
        if not isinstance(completed, CompletedHierarchicalDiscovery):
            raise TypeError("completed must be CompletedHierarchicalDiscovery")
        # Re-run the source object's complete authentication boundary.  This
        # catches in-memory mutation after its original construction.
        completed.__post_init__()
        routed_names = tuple(row.feature.canonical_name for row in completed.routed_features)
        if len(routed_names) != len(set(routed_names)):
            raise ValueError("completed discovery contains duplicate routed feature names")
        modeled_count = sum(bool(_legacy_roles(row)) for row in completed.routed_features)
        if modeled_count > self.max_candidates:
            raise ValueError(
                "modeled candidate count exceeds max_candidates; refusing to truncate "
                f"{modeled_count} candidates to {self.max_candidates}"
            )

        definitions = completed.extraction_definitions
        modeled: list[AuthenticatedModeledCandidate] = []
        dispositions: list[CompiledFeatureDisposition] = []
        for routed in completed.routed_features:
            name = routed.feature.canonical_name
            definition = _validate_definition(
                routed=routed,
                definition=definitions[name],
            )
            definition_sha256 = _sha(definition)
            roles = _legacy_roles(routed)
            # CandidateContract is also used as the categorical concrete-value
            # validator for non-modeled features.  Its temporary validation role
            # is never emitted or represented as a routing decision.
            validation_roles = roles or ("effect_modifier",)
            spec = _legacy_spec(
                routed=routed,
                definition=definition,
                roles=validation_roles,
            )
            validated = CandidateContract(
                spec,
                source_families=routed.feature.source_families,
            )
            if roles:
                record = AuthenticatedModeledCandidate.create(
                    extraction_spec={
                        **validated.extraction_spec,
                        "roles": list(roles),
                    },
                    source_families=routed.feature.source_families,
                )
                modeled.append(record)
                dispositions.append(
                    CompiledFeatureDisposition.create(
                        routed=routed,
                        extraction_definition_sha256=definition_sha256,
                        disposition=MODELED_DISPOSITION,
                        legacy_roles=roles,
                        modeled_contract_record_sha256=record.record_sha256,
                        reason=(
                            "Deterministic role routing assigned at least one downstream "
                            "adjustment or effect-modifier modeling slot."
                        ),
                    )
                )
            else:
                disposition, reason = _non_model_disposition(routed)
                dispositions.append(
                    CompiledFeatureDisposition.create(
                        routed=routed,
                        extraction_definition_sha256=definition_sha256,
                        disposition=disposition,
                        legacy_roles=(),
                        modeled_contract_record_sha256="",
                        reason=reason,
                    )
                )

        compiler_identity = self.identity
        identity = {
            "schema_version": COMPILED_HIERARCHICAL_FEATURE_REGISTRY_VERSION,
            "source_completion_sha256": completed.completion_sha256,
            "compiler_identity": compiler_identity,
            "compiler_sha256": self.identity_sha256,
            "max_candidates": self.max_candidates,
            "modeled_candidates": [candidate.as_dict() for candidate in modeled],
            "dispositions": [disposition.audit() for disposition in dispositions],
        }
        return CompiledHierarchicalFeatureRegistry(
            source_completion_sha256=completed.completion_sha256,
            compiler_sha256=self.identity_sha256,
            max_candidates=self.max_candidates,
            modeled_candidates=tuple(modeled),
            dispositions=tuple(dispositions),
            registry_sha256=_sha(identity),
            _compiler_identity_json=canonical_json(compiler_identity),
        )


def compile_hierarchical_discovery(
    completed: CompletedHierarchicalDiscovery,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> CompiledHierarchicalFeatureRegistry:
    """Compile one authenticated completion without truncating accepted features."""

    return HierarchicalDiscoveryCompiler(max_candidates=max_candidates).compile(completed)


__all__ = [
    "AUTHENTICATED_MODELED_CONTRACT_VERSION",
    "AuthenticatedModeledCandidate",
    "COMPILED_FEATURE_DISPOSITION_VERSION",
    "COMPILED_HIERARCHICAL_FEATURE_REGISTRY_VERSION",
    "CompiledFeatureDisposition",
    "CompiledHierarchicalFeatureRegistry",
    "DEFAULT_MAX_CANDIDATES",
    "DOWNSTREAM_ADJUSTMENT_SLOT_AUDIT",
    "HIERARCHICAL_DISCOVERY_COMPILER_VERSION",
    "HierarchicalDiscoveryCompiler",
    "MODELED_DISPOSITION",
    "NON_MODEL_EXTRACTION_ONLY",
    "NON_MODEL_ROLELESS",
    "NON_MODEL_TREATMENT_AND_EXTRACTION_ONLY",
    "NON_MODEL_TREATMENT_ONLY",
    "compile_hierarchical_discovery",
]
