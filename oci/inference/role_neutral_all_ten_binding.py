"""All-ten binding for role-neutral Stage 1 physical-fit artifacts.

The architecture-specific executors fit one canonical physical owner and
publish distinct purpose-specific logical views.  This module is the only
place where those independently authenticated component results are joined.
It requires exactly the ten native evidence families, derives one path- and
device-neutral logical source identity per context, and delegates persistence
to the closed 35-physical/40-logical binding format.

No component path is part of scientific identity.  A caller must first invoke
the component's fresh path-only validator and then construct a receipt from
the validated terminal data and fit-only family seals.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)
from .portable_workflow_spec import EVIDENCE_FAMILIES
from .production_stage1_legacy_scope_fragments import (
    build_role_neutral_fit_only_family_seal,
    persist_role_neutral_logical_evidence_bindings,
    validate_persisted_role_neutral_logical_evidence_bindings,
)
from .production_stage1_scope_scheduler import Stage1ScopePlan


ROLE_NEUTRAL_COMPONENT_RECEIPT_SCHEMA = (
    "production_role_neutral_component_receipt_v1"
)
ROLE_NEUTRAL_ALL_TEN_OWNER_SCHEMA = (
    "production_role_neutral_all_ten_physical_owner_v1"
)
ROLE_NEUTRAL_LOGICAL_SOURCE_IDENTITY_SCHEMA = (
    "production_role_neutral_all_ten_logical_source_identity_v1"
)

PORTABLE_TO_NATIVE_FAMILY = {
    "word_treatment_outcome": BOW_NUISANCE,
    "word_residual_effect": BOW_R_LOSS,
    "hierarchical_transformer": HTR_NEURAL,
    "matched_patient_uplift": MATCHED_PAIR_UPLIFT,
    "whole_cohort_embeddings": EMBEDDING_WHOLE_COHORT,
    "cluster_local_embeddings": EMBEDDING_CLUSTERED,
    "lexical_semantic_retrieval": TFIDF_SEMANTIC_RETRIEVAL,
    "tfidf_topics": TFIDF_TOPICS,
    "residual_tfidf_ngrams": TFIDF_ORPHAN_NGRAMS,
    "learned_neural_queries": NEURAL_QUERY_MOMENTS,
}
NATIVE_TO_PORTABLE_FAMILY = {
    native: portable for portable, native in PORTABLE_TO_NATIVE_FAMILY.items()
}
EXPECTED_COMPONENT_FAMILIES = MappingProxyType(
    {
        "bow": (BOW_NUISANCE, BOW_R_LOSS),
        "htr": (HTR_NEURAL,),
        "matched_pair": (MATCHED_PAIR_UPLIFT,),
        "embeddings": (
            EMBEDDING_WHOLE_COHORT,
            EMBEDDING_CLUSTERED,
            TFIDF_SEMANTIC_RETRIEVAL,
        ),
        "tfidf": (TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS),
        "neural_query": (NEURAL_QUERY_MOMENTS,),
    }
)

_HEX = frozenset("0123456789abcdef")
_COMPONENT_TERMINAL_FILE = "execution_manifest.json"


def _validate_static_family_contract() -> None:
    if (
        tuple(PORTABLE_TO_NATIVE_FAMILY) != tuple(EVIDENCE_FAMILIES)
        or tuple(PORTABLE_TO_NATIVE_FAMILY.values())
        != tuple(ACTIVE_STAGE1_CONCEPT_FAMILIES)
        or set(NATIVE_TO_PORTABLE_FAMILY)
        != ACTIVE_STAGE1_CONCEPT_FAMILY_SET
    ):
        raise RuntimeError(
            "portable and native Stage 1 evidence-family contracts diverged"
        )
    component_families = tuple(
        family
        for families in EXPECTED_COMPONENT_FAMILIES.values()
        for family in families
    )
    if component_families != tuple(ACTIVE_STAGE1_CONCEPT_FAMILIES):
        raise RuntimeError(
            "role-neutral producer partition no longer covers the canonical "
            "ten evidence families in order"
        )


_validate_static_family_contract()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _safe_component_tree_sha256(root: Path | str) -> str:
    """Authenticate an already validated component tree without locators."""

    source = Path(root)
    if source.is_symlink():
        raise ValueError("component artifact root cannot be a symbolic link")
    tree = source.resolve(strict=True)
    if tree != source.absolute() or not tree.is_dir():
        raise ValueError("component artifact root must be one canonical directory")
    inventory: list[dict[str, Any]] = []
    seen_inodes: set[tuple[int, int]] = set()
    for path in sorted(tree.rglob("*"), key=lambda value: value.relative_to(tree).as_posix()):
        relative = path.relative_to(tree).as_posix()
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"component artifact contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_nlink) != 1:
            raise ValueError(
                f"component artifact file is not private regular data: {relative}"
            )
        inode = (int(metadata.st_dev), int(metadata.st_ino))
        if inode in seen_inodes:
            raise ValueError("component artifact contains a hard-linked alias")
        seen_inodes.add(inode)
        digest = hashlib.sha256()
        size = 0
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            while block := os.read(descriptor, 1024 * 1024):
                digest.update(block)
                size += len(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        stable = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_mode),
            int(before.st_nlink),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
        )
        if stable != (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_mode),
            int(after.st_nlink),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
        ) or size != int(after.st_size):
            raise RuntimeError(f"component artifact changed while hashing: {relative}")
        inventory.append(
            {
                "relative_path": relative,
                "size_bytes": size,
                "sha256": digest.hexdigest(),
            }
        )
    if not inventory:
        raise ValueError("component artifact tree is empty")
    return _sha256_json(
        {
            "schema_version": "production_role_neutral_component_tree_v1",
            "files": inventory,
        }
    )


def _read_component_json(
    root: Path,
    relative_path: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], str, int]:
    relative = PurePosixPath(str(relative_path))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} has an unsafe relative path")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a safe regular file")
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_nlink) != 1:
        raise ValueError(f"{label} must be private regular data")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} could not be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or (int(before.st_dev), int(before.st_ino))
            != (int(metadata.st_dev), int(metadata.st_ino))
        ):
            raise ValueError(f"{label} changed while it was opened")
        digest = hashlib.sha256()
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            blocks.append(block)
        after = os.fstat(descriptor)
        stable_before = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_mode),
            int(before.st_nlink),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
        )
        stable_after = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_mode),
            int(after.st_nlink),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
        )
        payload = b"".join(blocks)
        if stable_before != stable_after or len(payload) != int(after.st_size):
            raise RuntimeError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=lambda pairs: _duplicate_rejecting_object(
                pairs, label=label
            ),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not closed UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value, digest.hexdigest(), len(payload)


def _read_registered_component_json(
    root: Path,
    registration: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(registration, Mapping):
        raise ValueError(f"{label} registration is malformed")
    required = {"relative_path", "sha256", "size_bytes", "content_sha256"}
    if not required <= set(registration):
        raise ValueError(f"{label} registration is incomplete")
    value, digest, size = _read_component_json(
        root,
        registration.get("relative_path"),
        label=label,
    )
    if (
        digest != _require_sha256(
            registration.get("sha256"),
            label=f"{label} byte identity",
        )
        or size != registration.get("size_bytes")
        or value.get("content_sha256")
        != _require_sha256(
            registration.get("content_sha256"),
            label=f"{label} content identity",
        )
    ):
        raise ValueError(f"{label} differs from its terminal registration")
    return value


def _require_terminal_false(
    terminal: Mapping[str, Any],
    key: str,
    *,
    component: str,
) -> bool:
    if key not in terminal or terminal.get(key) is not False:
        raise ValueError(
            f"{component} terminal must explicitly attest {key}=false"
        )
    return False


def _reopen_validated_component_terminal(
    root: Path,
    terminal: Mapping[str, Any],
    *,
    component: str,
) -> dict[str, Any]:
    if not isinstance(terminal, Mapping):
        raise ValueError(f"{component} validator returned a malformed terminal")
    reopened, _digest, _size = _read_component_json(
        root,
        _COMPONENT_TERMINAL_FILE,
        label=f"{component} execution terminal",
    )
    body = {
        key: copy.deepcopy(value)
        for key, value in reopened.items()
        if key != "content_sha256"
    }
    if (
        reopened != dict(terminal)
        or reopened.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError(
            f"{component} execution terminal changed after fresh validation"
        )
    return reopened


def _duplicate_rejecting_object(
    pairs: Sequence[tuple[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"{label} contains duplicate key {key!r}")
        value[key] = child
    return value


def _owner_group(plan: Stage1ScopePlan, owner_scope_id: str):
    owner = plan.scope(str(owner_scope_id))
    if plan.physical_owner(owner.scope_id).scope_id != owner.scope_id:
        raise ValueError("all-ten component receipt must name a physical owner")
    matches = [
        members
        for candidate, members in plan.physical_scope_groups
        if candidate.scope_id == owner.scope_id
    ]
    if len(matches) != 1:
        raise RuntimeError("physical owner has no unique logical group")
    members = matches[0]
    if (
        not members
        or members[0].scope_id != owner.scope_id
        or any(
            tuple(member.fit_row_ids)
            != tuple(owner.fit_row_ids)
            or int(member.scope_seed) != int(owner.scope_seed)
            for member in members
        )
    ):
        raise ValueError("all-ten logical group changed fit rows or seed")
    return owner, members


def _validate_standard_seal(
    *,
    plan: Stage1ScopePlan,
    owner_scope_id: str,
    family: str,
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(seal, Mapping):
        raise TypeError(f"{family} fit-only seal must be one mapping")
    closed = copy.deepcopy(dict(seal))
    expected = build_role_neutral_fit_only_family_seal(
        plan=plan,
        physical_owner_scope_id=owner_scope_id,
        family=family,
        evidence_payload=closed.get("evidence_payload") or {},
        producer_identity_sha256=closed.get("producer_identity_sha256"),
        configuration_identity_sha256=closed.get(
            "configuration_identity_sha256"
        ),
        fit_state_artifact_sha256=closed.get("fit_state_artifact_sha256"),
    )
    if closed != expected:
        raise ValueError(
            f"{owner_scope_id}/{family} is not the closed role-neutral "
            "fit-only family seal"
        )
    return closed


@dataclass(frozen=True)
class AuthenticatedRoleNeutralComponentReceipt:
    """Path-neutral capability produced after one component validates fresh."""

    component: str
    plan_scientific_content_sha256: str
    physical_owner_scope_id: str
    logical_scope_ids: tuple[str, ...]
    family_fit_seals: Mapping[str, Mapping[str, Any]]
    family_logical_view_content_sha256: Mapping[str, Mapping[str, str]]
    source_terminal_content_sha256: str
    source_tree_sha256: str
    registered_heldout_labels_accessed: bool
    oracle_fields_accessed: bool
    text_truncation_applied: bool
    lossy_evidence_selection_applied: bool
    authentication_content_sha256: str

    @classmethod
    def create(
        cls,
        *,
        plan: Stage1ScopePlan,
        physical_owner_scope_id: str,
        component: str,
        family_fit_seals: Mapping[str, Mapping[str, Any]],
        family_logical_view_content_sha256: Mapping[
            str, Mapping[str, str]
        ],
        source_terminal_content_sha256: str,
        source_tree_sha256: str,
        registered_heldout_labels_accessed: bool = False,
        oracle_fields_accessed: bool = False,
        text_truncation_applied: bool = False,
        lossy_evidence_selection_applied: bool = False,
    ) -> "AuthenticatedRoleNeutralComponentReceipt":
        if not isinstance(plan, Stage1ScopePlan):
            raise TypeError("component receipt requires a Stage1ScopePlan")
        owner, members = _owner_group(plan, physical_owner_scope_id)
        component_name = str(component).strip()
        if component_name not in EXPECTED_COMPONENT_FAMILIES:
            raise ValueError("component receipt names an unknown Stage 1 producer")
        expected_families = EXPECTED_COMPONENT_FAMILIES[component_name]
        if (
            not isinstance(family_fit_seals, Mapping)
            or set(family_fit_seals) != set(expected_families)
        ):
            raise ValueError(
                f"{component_name} receipt must contain exactly its canonical "
                "native family partition"
            )
        if set(family_logical_view_content_sha256) != set(
            family_fit_seals
        ):
            raise ValueError(
                "component fit seals and logical-view family coverage differ"
            )
        scope_ids = tuple(member.scope_id for member in members)
        normalized_seals = {
            family: _validate_standard_seal(
                plan=plan,
                owner_scope_id=owner.scope_id,
                family=family,
                seal=family_fit_seals[family],
            )
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            if family in family_fit_seals
        }
        normalized_views: dict[str, dict[str, str]] = {}
        for family in normalized_seals:
            raw_views = family_logical_view_content_sha256[family]
            if not isinstance(raw_views, Mapping) or set(raw_views) != set(
                scope_ids
            ):
                raise ValueError(
                    f"{family} logical views do not cover exactly its "
                    "physical equivalence group"
                )
            normalized_views[family] = {
                scope_id: _require_sha256(
                    raw_views[scope_id],
                    label=f"{family}/{scope_id} logical-view identity",
                )
                for scope_id in scope_ids
            }
            if len(set(normalized_views[family].values())) != len(scope_ids):
                raise ValueError(
                    f"{family} reused one logical-view identity across "
                    "purpose-specific scopes"
                )
        safety = (
            registered_heldout_labels_accessed,
            oracle_fields_accessed,
            text_truncation_applied,
            lossy_evidence_selection_applied,
        )
        if any(type(value) is not bool for value in safety):
            raise TypeError("component safety statements must be booleans")
        if any(safety):
            raise ValueError(
                "all-ten publication rejects held-out labels, oracle access, "
                "text truncation, and lossy evidence selection"
            )
        terminal_id = _require_sha256(
            source_terminal_content_sha256,
            label=f"{component_name} terminal identity",
        )
        tree_id = _require_sha256(
            source_tree_sha256,
            label=f"{component_name} authenticated tree identity",
        )
        authentication_body = {
            "schema_version": (
                "production_role_neutral_component_authenticated_handle_v1"
            ),
            "component": component_name,
            "plan_scientific_content_sha256": plan.scientific_content_sha256,
            "physical_owner_scope_id": owner.scope_id,
            "logical_scope_ids": list(scope_ids),
            "family_fit_seals": normalized_seals,
            "family_logical_view_content_sha256": normalized_views,
            "source_terminal_content_sha256": terminal_id,
            "source_tree_sha256": tree_id,
            "registered_heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
            "text_truncation_applied": False,
            "lossy_evidence_selection_applied": False,
        }
        return cls(
            component=component_name,
            plan_scientific_content_sha256=plan.scientific_content_sha256,
            physical_owner_scope_id=owner.scope_id,
            logical_scope_ids=scope_ids,
            family_fit_seals=normalized_seals,
            family_logical_view_content_sha256=normalized_views,
            source_terminal_content_sha256=terminal_id,
            source_tree_sha256=tree_id,
            registered_heldout_labels_accessed=False,
            oracle_fields_accessed=False,
            text_truncation_applied=False,
            lossy_evidence_selection_applied=False,
            authentication_content_sha256=_sha256_json(
                authentication_body
            ),
        )

    def _assert_intact(self) -> None:
        authentication_body = {
            "schema_version": (
                "production_role_neutral_component_authenticated_handle_v1"
            ),
            "component": self.component,
            "plan_scientific_content_sha256": (
                self.plan_scientific_content_sha256
            ),
            "physical_owner_scope_id": self.physical_owner_scope_id,
            "logical_scope_ids": list(self.logical_scope_ids),
            "family_fit_seals": self.family_fit_seals,
            "family_logical_view_content_sha256": (
                self.family_logical_view_content_sha256
            ),
            "source_terminal_content_sha256": (
                self.source_terminal_content_sha256
            ),
            "source_tree_sha256": self.source_tree_sha256,
            "registered_heldout_labels_accessed": (
                self.registered_heldout_labels_accessed
            ),
            "oracle_fields_accessed": self.oracle_fields_accessed,
            "text_truncation_applied": self.text_truncation_applied,
            "lossy_evidence_selection_applied": (
                self.lossy_evidence_selection_applied
            ),
        }
        if (
            _sha256_json(authentication_body)
            != self.authentication_content_sha256
        ):
            raise RuntimeError(
                "authenticated role-neutral component receipt was mutated"
            )

    def scientific_dict(self) -> dict[str, Any]:
        """Return scientific content with execution-tree identity excluded."""

        self._assert_intact()
        body = {
            "schema_version": ROLE_NEUTRAL_COMPONENT_RECEIPT_SCHEMA,
            "component": self.component,
            "plan_scientific_content_sha256": (
                self.plan_scientific_content_sha256
            ),
            "physical_owner_scope_id": self.physical_owner_scope_id,
            "logical_scope_ids": list(self.logical_scope_ids),
            "family_fit_artifact_sha256": {
                family: self.family_fit_seals[family]["content_sha256"]
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
                if family in self.family_fit_seals
            },
            "family_logical_view_content_sha256": {
                family: dict(
                    self.family_logical_view_content_sha256[family]
                )
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
                if family in self.family_logical_view_content_sha256
            },
            "source_terminal_content_sha256": (
                self.source_terminal_content_sha256
            ),
            "registered_heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
            "text_truncation_applied": False,
            "lossy_evidence_selection_applied": False,
            "execution_locator_metadata_in_scientific_identity": False,
        }
        return {**body, "content_sha256": _sha256_json(body)}

    def execution_attestation(self) -> dict[str, Any]:
        self._assert_intact()
        body = {
            "schema_version": (
                "production_role_neutral_component_execution_attestation_v1"
            ),
            "component_scientific_content_sha256": self.scientific_dict()[
                "content_sha256"
            ],
            "source_tree_sha256": self.source_tree_sha256,
        }
        return {**body, "content_sha256": _sha256_json(body)}


def validate_authenticated_role_neutral_component_receipt(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
    physical_owner_scope_id: str,
    receipt: AuthenticatedRoleNeutralComponentReceipt,
    expected_component: str | None = None,
) -> AuthenticatedRoleNeutralComponentReceipt:
    """Rebind an in-process receipt to an unchanged canonical source tree."""

    if not isinstance(receipt, AuthenticatedRoleNeutralComponentReceipt):
        raise TypeError("component receipt revalidation requires its typed handle")
    receipt._assert_intact()
    owner, members = _owner_group(plan, physical_owner_scope_id)
    component = (
        receipt.component
        if expected_component is None
        else str(expected_component)
    )
    if (
        component not in EXPECTED_COMPONENT_FAMILIES
        or receipt.component != component
        or receipt.plan_scientific_content_sha256
        != plan.scientific_content_sha256
        or receipt.physical_owner_scope_id != owner.scope_id
        or receipt.logical_scope_ids
        != tuple(member.scope_id for member in members)
        or set(receipt.family_fit_seals)
        != set(EXPECTED_COMPONENT_FAMILIES[component])
    ):
        raise ValueError(
            "authenticated component receipt belongs to another plan, "
            "owner, or producer partition"
        )
    if _safe_component_tree_sha256(root) != receipt.source_tree_sha256:
        raise ValueError("authenticated component source tree changed")
    terminal, _digest, _size = _read_component_json(
        Path(root).resolve(strict=True),
        _COMPONENT_TERMINAL_FILE,
        label=f"{component} execution terminal",
    )
    if (
        terminal.get("content_sha256")
        != receipt.source_terminal_content_sha256
    ):
        raise ValueError("authenticated component terminal identity changed")
    return receipt


def merge_all_ten_components_for_owner(
    *,
    plan: Stage1ScopePlan,
    physical_owner_scope_id: str,
    components: Sequence[AuthenticatedRoleNeutralComponentReceipt],
) -> dict[str, Any]:
    """Require all ten independent families for one physical-fit owner."""

    owner, members = _owner_group(plan, physical_owner_scope_id)
    receipts = tuple(components)
    if not receipts:
        raise ValueError("all-ten owner merge has no component receipts")
    if (
        len(receipts) != len(EXPECTED_COMPONENT_FAMILIES)
        or {receipt.component for receipt in receipts}
        != set(EXPECTED_COMPONENT_FAMILIES)
    ):
        raise ValueError(
            "all-ten owner merge has incomplete family coverage unless it "
            "contains exactly the canonical six role-neutral producer "
            "components"
        )
    expected_scopes = tuple(member.scope_id for member in members)
    seals: dict[str, Mapping[str, Any]] = {}
    views: dict[str, Mapping[str, str]] = {}
    for receipt in receipts:
        if not isinstance(
            receipt, AuthenticatedRoleNeutralComponentReceipt
        ):
            raise TypeError("all-ten merge received an unauthenticated receipt")
        receipt._assert_intact()
        if (
            receipt.plan_scientific_content_sha256
            != plan.scientific_content_sha256
            or receipt.physical_owner_scope_id != owner.scope_id
            or receipt.logical_scope_ids != expected_scopes
        ):
            raise ValueError(
                "component receipt belongs to another scientific plan or group"
            )
        if set(receipt.family_fit_seals) != set(
            EXPECTED_COMPONENT_FAMILIES[receipt.component]
        ):
            raise ValueError(
                f"{receipt.component} receipt changed its canonical family "
                "partition"
            )
        overlap = set(seals) & set(receipt.family_fit_seals)
        if overlap:
            raise ValueError(
                "native family was supplied by more than one component: "
                + ", ".join(sorted(overlap))
            )
        seals.update(receipt.family_fit_seals)
        views.update(receipt.family_logical_view_content_sha256)
    if set(seals) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        missing = sorted(ACTIVE_STAGE1_CONCEPT_FAMILY_SET - set(seals))
        extra = sorted(set(seals) - ACTIVE_STAGE1_CONCEPT_FAMILY_SET)
        raise ValueError(
            "all-ten owner merge has incomplete family coverage; "
            f"missing={missing}, extra={extra}"
        )
    if set(views) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise RuntimeError("all-ten logical-view coverage differs from fit seals")

    logical_source_identities: dict[str, dict[str, Any]] = {}
    for logical in members:
        body = {
            "schema_version": ROLE_NEUTRAL_LOGICAL_SOURCE_IDENTITY_SCHEMA,
            "plan_scientific_content_sha256": (
                plan.scientific_content_sha256
            ),
            "logical_scope_id": logical.scope_id,
            "logical_scope_sha256": logical.as_dict()["scope_sha256"],
            "logical_purpose": logical.scope_kind,
            "physical_owner_scope_id": owner.scope_id,
            "family_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "family_fit_artifact_sha256": {
                family: seals[family]["content_sha256"]
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            },
            "family_logical_view_content_sha256": {
                family: views[family][logical.scope_id]
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            },
            "heldout_labels_supplied": False,
            "oracle_fields_accessed": False,
            "text_truncation_applied": False,
            "lossy_evidence_selection_applied": False,
        }
        logical_source_identities[logical.scope_id] = {
            **body,
            "content_sha256": _sha256_json(body),
        }
    component_scientific_receipts = sorted(
        (receipt.scientific_dict() for receipt in receipts),
        key=lambda row: row["component"],
    )
    body = {
        "schema_version": ROLE_NEUTRAL_ALL_TEN_OWNER_SCHEMA,
        "plan_scientific_content_sha256": plan.scientific_content_sha256,
        "physical_owner_scope_id": owner.scope_id,
        "physical_owner_scope_sha256": owner.as_dict()["scope_sha256"],
        "logical_scope_ids": list(expected_scopes),
        "portable_family_order": list(EVIDENCE_FAMILIES),
        "native_family_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
        "portable_to_native_family": dict(PORTABLE_TO_NATIVE_FAMILY),
        "family_fit_seals": copy.deepcopy(seals),
        "logical_source_identities": logical_source_identities,
        "component_scientific_receipts": component_scientific_receipts,
        "all_ten_nonempty_fit_families_present": True,
        "fit_side_family_ids_shared_by_every_logical_member": True,
        "logical_views_are_purpose_specific": True,
        "heldout_labels_supplied": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
        "lossy_evidence_selection_applied": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def persist_complete_role_neutral_stage1_bindings(
    *,
    root: Any,
    plan: Stage1ScopePlan,
    components_by_physical_owner: Mapping[
        str, Sequence[AuthenticatedRoleNeutralComponentReceipt]
    ],
) -> dict[str, Any]:
    """Publish the complete physical/logical binding after every owner seals."""

    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError("all-ten persistence requires a Stage1ScopePlan")
    owner_ids = tuple(owner.scope_id for owner in plan.physical_scopes)
    if (
        not isinstance(components_by_physical_owner, Mapping)
        or set(components_by_physical_owner) != set(owner_ids)
    ):
        raise ValueError(
            "all-ten component receipts do not cover every physical owner"
        )
    seals_by_owner: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    logical_sources: dict[str, str] = {}
    for owner_id in owner_ids:
        merged = merge_all_ten_components_for_owner(
            plan=plan,
            physical_owner_scope_id=owner_id,
            components=components_by_physical_owner[owner_id],
        )
        seals_by_owner[owner_id] = merged["family_fit_seals"]
        for scope_id, identity in merged[
            "logical_source_identities"
        ].items():
            if scope_id in logical_sources:
                raise RuntimeError("logical scope was bound by two physical owners")
            logical_sources[scope_id] = identity["content_sha256"]
    if set(logical_sources) != {scope.scope_id for scope in plan.scopes}:
        raise RuntimeError("all-ten logical source coverage is incomplete")
    return persist_role_neutral_logical_evidence_bindings(
        root=root,
        plan=plan,
        family_fit_seal_by_physical_owner=seals_by_owner,
        logical_source_artifact_sha256_by_scope=logical_sources,
    )


def validate_complete_role_neutral_stage1_bindings(
    *,
    root: Any,
    plan: Stage1ScopePlan,
) -> dict[str, Any]:
    """Freshly reopen the persisted 35-physical/40-logical binding tree."""

    manifest = validate_persisted_role_neutral_logical_evidence_bindings(
        root=root,
        plan=plan,
    )
    if (
        manifest.get("physical_fit_count") != len(plan.physical_scopes)
        or manifest.get("logical_scope_count") != len(plan.scopes)
        or manifest.get("deduplicated_fit_count")
        != len(plan.scopes) - len(plan.physical_scopes)
    ):
        raise ValueError("complete role-neutral binding counts changed")
    return manifest


def _logical_view_identities(
    *,
    root: Path,
    terminal: Mapping[str, Any],
    families: Sequence[str],
    logical_scope_ids: Sequence[str],
) -> dict[str, dict[str, str]]:
    rows = terminal.get("logical_views")
    if not isinstance(rows, list):
        raise ValueError("component terminal lacks logical-view registrations")
    expected_families = tuple(families)
    expected_scopes = tuple(logical_scope_ids)
    output = {family: {} for family in expected_families}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("component logical-view registration is malformed")
        view = _read_registered_component_json(
            root,
            row,
            label="component logical-view artifact",
        )
        family = (
            str(row.get("family"))
            if row.get("family") is not None
            else str(view.get("family"))
        )
        scope_id = str(row.get("logical_scope_id"))
        if (
            family not in output
            or scope_id not in expected_scopes
            or scope_id in output[family]
            or view.get("family") != family
            or view.get("logical_scope_id") != scope_id
            or view.get("content_sha256")
            != row.get("content_sha256")
            or view.get("registered_heldout_labels_accessed") is not False
        ):
            raise ValueError(
                "component logical-view family/scope registration is duplicated "
                "or inconsistent with its authenticated artifact"
            )
        output[family][scope_id] = _require_sha256(
            row.get("content_sha256"),
            label=f"{family}/{scope_id} registered logical-view content",
        )
    if any(set(rows_by_scope) != set(expected_scopes) for rows_by_scope in output.values()):
        raise ValueError("component logical-view registrations are incomplete")
    if any(
        len(set(rows_by_scope.values())) != len(expected_scopes)
        for rows_by_scope in output.values()
    ):
        raise ValueError(
            "component reused one logical-view artifact across distinct purposes"
        )
    return output


def _standard_component_receipt(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
    physical_owner_scope_id: str,
    component: str,
    families: Sequence[str],
    terminal: Mapping[str, Any],
    seal_registrations: Mapping[str, Mapping[str, Any]],
    lossy_evidence_selection_applied: bool,
) -> AuthenticatedRoleNeutralComponentReceipt:
    source_tree_sha256 = _safe_component_tree_sha256(root)
    tree = Path(root).resolve(strict=True)
    terminal = _reopen_validated_component_terminal(
        tree,
        terminal,
        component=component,
    )
    owner, members = _owner_group(plan, physical_owner_scope_id)
    expected_families = tuple(families)
    if (
        set(expected_families) != set(seal_registrations)
        or len(expected_families) != len(set(expected_families))
    ):
        raise ValueError("component fit-only seal registration coverage changed")
    seals: dict[str, Mapping[str, Any]] = {}
    for family in expected_families:
        registration = seal_registrations[family]
        seal = _read_registered_component_json(
            tree,
            registration,
            label=f"{family} fit-only seal",
        )
        seals[family] = seal
    return AuthenticatedRoleNeutralComponentReceipt.create(
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
        component=component,
        family_fit_seals=seals,
        family_logical_view_content_sha256=_logical_view_identities(
            root=tree,
            terminal=terminal,
            families=expected_families,
            logical_scope_ids=tuple(member.scope_id for member in members),
        ),
        source_terminal_content_sha256=_require_sha256(
            terminal.get("content_sha256"),
            label=f"{component} terminal content",
        ),
        source_tree_sha256=source_tree_sha256,
        registered_heldout_labels_accessed=_require_terminal_false(
            terminal,
            "registered_heldout_labels_accessed",
            component=component,
        ),
        oracle_fields_accessed=_require_terminal_false(
            terminal,
            "oracle_fields_accessed",
            component=component,
        ),
        text_truncation_applied=_require_terminal_false(
            terminal,
            "text_truncation_applied",
            component=component,
        ),
        lossy_evidence_selection_applied=lossy_evidence_selection_applied,
    )


def authenticate_role_neutral_bow_component(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
    physical_owner_scope_id: str,
) -> AuthenticatedRoleNeutralComponentReceipt:
    """Freshly validate and adapt the two complete BoW families."""

    from .role_neutral_bow_group_execution import (
        RoleNeutralBoWPhysicalGroupRequest,
        validate_role_neutral_bow_group_execution,
    )

    request = RoleNeutralBoWPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=physical_owner_scope_id,
    )
    terminal = validate_role_neutral_bow_group_execution(
        root=root,
        request=request,
    )
    registrations = terminal.get("fit_only_family_seals")
    if not isinstance(registrations, Mapping):
        raise ValueError("BoW terminal lacks fit-only family registrations")
    return _standard_component_receipt(
        root=root,
        plan=plan,
        physical_owner_scope_id=physical_owner_scope_id,
        component="bow",
        families=(BOW_NUISANCE, BOW_R_LOSS),
        terminal=terminal,
        seal_registrations=registrations,
        lossy_evidence_selection_applied=False,
    )


def authenticate_role_neutral_htr_component(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
    physical_owner_scope_id: str,
    htr_model_path: Path | str | None = None,
    device: Any = "cpu",
) -> AuthenticatedRoleNeutralComponentReceipt:
    """Freshly replay, validate, and adapt the complete HTR family."""

    from .role_neutral_htr_group_execution import (
        RoleNeutralHTRPhysicalGroupRequest,
        validate_role_neutral_htr_group_execution,
    )

    request = RoleNeutralHTRPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=physical_owner_scope_id,
    )
    terminal = validate_role_neutral_htr_group_execution(
        root=root,
        request=request,
        htr_model_path=htr_model_path,
        device=device,
    )
    registration = terminal.get("fit_only_family_seal")
    if not isinstance(registration, Mapping):
        raise ValueError("HTR terminal lacks its fit-only family registration")
    return _standard_component_receipt(
        root=root,
        plan=plan,
        physical_owner_scope_id=physical_owner_scope_id,
        component="htr",
        families=(HTR_NEURAL,),
        terminal=terminal,
        seal_registrations={HTR_NEURAL: registration},
        lossy_evidence_selection_applied=False,
    )


def authenticate_role_neutral_matched_pair_component(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
    physical_owner_scope_id: str,
    htr_model_identity_sha256: str,
    nuisance_artifact_identity_sha256: str,
    runtime_compatibility_class: str,
) -> AuthenticatedRoleNeutralComponentReceipt:
    """Validate both matched-pair subproducers and derive one common seal."""

    from .lossless_stage1_evidence_catalog import (
        NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
    )
    from .role_neutral_matched_pair_group_execution import (
        RoleNeutralMatchedPairPhysicalGroupRequest,
        validate_role_neutral_matched_pair_group_execution,
    )

    request = RoleNeutralMatchedPairPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=physical_owner_scope_id,
        htr_model_identity_sha256=htr_model_identity_sha256,
        nuisance_artifact_identity_sha256=nuisance_artifact_identity_sha256,
        runtime_compatibility_class=runtime_compatibility_class,
    )
    terminal = validate_role_neutral_matched_pair_group_execution(
        root=root,
        request=request,
    )
    registration = terminal.get("fit_only_family_seal")
    if not isinstance(registration, Mapping):
        raise ValueError("matched-pair terminal lacks its fit-only family seal")
    source_tree_sha256 = _safe_component_tree_sha256(root)
    tree = Path(root).resolve(strict=True)
    terminal = _reopen_validated_component_terminal(
        tree,
        terminal,
        component="matched_pair",
    )
    source_seal = _read_registered_component_json(
        tree,
        registration,
        label="matched-pair fit-only family seal",
    )
    proofs = source_seal.get("subproducer_proofs")
    if (
        not isinstance(proofs, list)
        or not proofs
        or source_seal.get("content_sha256")
        != registration.get("content_sha256")
    ):
        raise ValueError("matched-pair subproducer proof coverage changed")
    normalized_payload = {
        "schema_version": NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
        "family": MATCHED_PAIR_UPLIFT,
        "architecture_evidence": [
            {
                "source_family_seal_content_sha256": source_seal[
                    "content_sha256"
                ],
                "subproducer": proof["subproducer"],
                "evidence_payload_sha256": proof[
                    "evidence_payload_sha256"
                ],
                "evidence_payload": copy.deepcopy(
                    proof["evidence_payload"]
                ),
            }
            for proof in proofs
        ],
    }
    common_seal = build_role_neutral_fit_only_family_seal(
        plan=plan,
        physical_owner_scope_id=physical_owner_scope_id,
        family=MATCHED_PAIR_UPLIFT,
        evidence_payload=normalized_payload,
        producer_identity_sha256=source_seal.get(
            "producer_identity_sha256"
        ),
        configuration_identity_sha256=source_seal.get(
            "configuration_identity_sha256"
        ),
        fit_state_artifact_sha256=source_seal.get(
            "fit_state_artifact_sha256"
        ),
    )
    return AuthenticatedRoleNeutralComponentReceipt.create(
        plan=plan,
        physical_owner_scope_id=physical_owner_scope_id,
        component="matched_pair",
        family_fit_seals={MATCHED_PAIR_UPLIFT: common_seal},
        family_logical_view_content_sha256=_logical_view_identities(
            root=tree,
            terminal=terminal,
            families=(MATCHED_PAIR_UPLIFT,),
            logical_scope_ids=tuple(
                member.scope_id for member in request.logical_members
            ),
        ),
        source_terminal_content_sha256=terminal["content_sha256"],
        source_tree_sha256=source_tree_sha256,
        registered_heldout_labels_accessed=_require_terminal_false(
            terminal,
            "registered_heldout_labels_accessed",
            component="matched_pair",
        ),
        oracle_fields_accessed=_require_terminal_false(
            terminal,
            "oracle_fields_accessed",
            component="matched_pair",
        ),
        text_truncation_applied=_require_terminal_false(
            terminal,
            "text_truncation_applied",
            component="matched_pair",
        ),
        lossy_evidence_selection_applied=_require_terminal_false(
            terminal,
            "top_k_evidence_applied",
            component="matched_pair",
        ),
    )


def authenticate_role_neutral_tfidf_component(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
    physical_owner_scope_id: str,
) -> AuthenticatedRoleNeutralComponentReceipt:
    """Freshly validate and adapt topics plus residual TF-IDF n-grams."""

    from .role_neutral_tfidf_group_execution import (
        RoleNeutralTfidfPhysicalGroupRequest,
        validate_role_neutral_tfidf_group_execution,
    )

    request = RoleNeutralTfidfPhysicalGroupRequest.from_plan(
        plan=plan,
        physical_owner_scope_id=physical_owner_scope_id,
    )
    terminal = validate_role_neutral_tfidf_group_execution(
        root=root,
        request=request,
    )
    registrations = terminal.get("fit_only_family_seals")
    if not isinstance(registrations, Mapping):
        raise ValueError("TF-IDF terminal lacks fit-only family registrations")
    return _standard_component_receipt(
        root=root,
        plan=plan,
        physical_owner_scope_id=physical_owner_scope_id,
        component="tfidf",
        families=(TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS),
        terminal=terminal,
        seal_registrations=registrations,
        lossy_evidence_selection_applied=False,
    )


def authenticate_role_neutral_neural_query_component(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
    request: Any,
) -> AuthenticatedRoleNeutralComponentReceipt:
    """Freshly validate and adapt one learned-neural-query fit artifact."""

    from .role_neutral_neural_query_group_execution import (
        RoleNeutralNeuralQueryPhysicalGroupRequest,
        validate_role_neutral_neural_query_group_execution,
    )

    if not isinstance(
        request, RoleNeutralNeuralQueryPhysicalGroupRequest
    ):
        raise TypeError("neural-query adapter requires its typed request")
    owner, members = _owner_group(
        plan, request.physical_owner.scope_id
    )
    if (
        request.plan_scientific_content_sha256
        != plan.scientific_content_sha256
        or request.physical_owner != owner
        or request.logical_members != members
    ):
        raise ValueError("neural-query request belongs to another plan/group")
    terminal = validate_role_neutral_neural_query_group_execution(
        root=root,
        request=request,
    )
    registration = terminal.get("fit_only_family_seal")
    if not isinstance(registration, Mapping):
        raise ValueError(
            "neural-query terminal lacks its fit-only family registration"
        )
    return _standard_component_receipt(
        root=root,
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
        component="neural_query",
        families=(NEURAL_QUERY_MOMENTS,),
        terminal=terminal,
        seal_registrations={NEURAL_QUERY_MOMENTS: registration},
        lossy_evidence_selection_applied=False,
    )


def authenticate_role_neutral_embedding_component(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
    request: Any,
    clustered_preflight: Any,
    clustered_preflight_state_manifest: Path | str,
    expected_scientific_config: Any = None,
    expected_fit_texts: Sequence[str] | None = None,
    expected_fit_targets: Mapping[str, Sequence[float]] | None = None,
    expected_exact_batch: Any = None,
) -> AuthenticatedRoleNeutralComponentReceipt:
    """Validate canonical preflight reuse and adapt three embedding families."""

    from .role_neutral_embedding_group_execution import (
        RoleNeutralEmbeddingPhysicalGroupRequest,
        validate_role_neutral_embedding_group_execution,
    )

    if not isinstance(request, RoleNeutralEmbeddingPhysicalGroupRequest):
        raise TypeError("embedding adapter requires its typed request")
    owner, members = _owner_group(plan, request.physical_owner.scope_id)
    if (
        request.plan_scientific_content_sha256
        != plan.scientific_content_sha256
        or request.physical_owner != owner
        or request.logical_members != members
    ):
        raise ValueError("embedding request belongs to another plan/group")
    terminal = validate_role_neutral_embedding_group_execution(
        root=root,
        request=request,
        clustered_preflight=clustered_preflight,
        clustered_preflight_state_manifest=(
            clustered_preflight_state_manifest
        ),
        expected_scientific_config=expected_scientific_config,
        expected_fit_texts=expected_fit_texts,
        expected_fit_targets=expected_fit_targets,
        expected_exact_batch=expected_exact_batch,
    )
    registrations = terminal.get("fit_only_family_seals")
    if not isinstance(registrations, Mapping):
        raise ValueError(
            "embedding terminal lacks fit-only family registrations"
        )
    return _standard_component_receipt(
        root=root,
        plan=plan,
        physical_owner_scope_id=owner.scope_id,
        component="embeddings",
        families=(
            EMBEDDING_WHOLE_COHORT,
            EMBEDDING_CLUSTERED,
            TFIDF_SEMANTIC_RETRIEVAL,
        ),
        terminal=terminal,
        seal_registrations=registrations,
        lossy_evidence_selection_applied=_require_terminal_false(
            terminal,
            "semantic_term_truncation_applied",
            component="embeddings",
        ),
    )


__all__ = [
    "AuthenticatedRoleNeutralComponentReceipt",
    "EXPECTED_COMPONENT_FAMILIES",
    "NATIVE_TO_PORTABLE_FAMILY",
    "PORTABLE_TO_NATIVE_FAMILY",
    "ROLE_NEUTRAL_ALL_TEN_OWNER_SCHEMA",
    "ROLE_NEUTRAL_COMPONENT_RECEIPT_SCHEMA",
    "ROLE_NEUTRAL_LOGICAL_SOURCE_IDENTITY_SCHEMA",
    "merge_all_ten_components_for_owner",
    "persist_complete_role_neutral_stage1_bindings",
    "validate_authenticated_role_neutral_component_receipt",
    "validate_complete_role_neutral_stage1_bindings",
    "authenticate_role_neutral_bow_component",
    "authenticate_role_neutral_htr_component",
    "authenticate_role_neutral_matched_pair_component",
    "authenticate_role_neutral_tfidf_component",
    "authenticate_role_neutral_neural_query_component",
    "authenticate_role_neutral_embedding_component",
]
