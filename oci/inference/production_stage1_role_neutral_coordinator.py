"""Production publication gate for role-neutral Stage 1 artifacts.

The six role-neutral producers publish one authenticated component tree per
physical-fit owner.  This module joins those trees only after:

* every owner has exactly the canonical six producer receipts;
* every receipt is rebound to an unchanged, distinct component root;
* the fixed producer-to-family partition covers all ten evidence families;
* the scientific 35-physical/40-logical binding tree validates from disk; and
* operational root locators are recorded outside scientific identity.

The locator attestation is intentionally retained.  The compact scientific
binding copies fit-side evidence seals, but numerical banks and
purpose-specific component views remain in their authenticated producer
trees.  A downstream reader therefore has both the path-neutral scientific
binding and the separately authenticated roots needed to reopen those bytes.

Nothing in this module accepts a legacy role-specific scope fragment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .production_stage1_legacy_scope_fragments import (
    durably_sync_legacy_stage1_tree,
)
from .production_stage1_scope_scheduler import Stage1ScopePlan
from .role_neutral_all_ten_binding import (
    AuthenticatedRoleNeutralComponentReceipt,
    EXPECTED_COMPONENT_FAMILIES,
    ROLE_NEUTRAL_COMPONENT_RECEIPT_SCHEMA,
    persist_complete_role_neutral_stage1_bindings,
    validate_authenticated_role_neutral_component_receipt,
    validate_complete_role_neutral_stage1_bindings,
)

ROLE_NEUTRAL_COORDINATION_GATE_SCHEMA = "production_role_neutral_stage1_coordination_gate_v1"
ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION_SCHEMA = (
    "production_role_neutral_component_locator_attestation_v1"
)
ROLE_NEUTRAL_COORDINATION_SCIENTIFIC_IDENTITY_SCHEMA = (
    "production_role_neutral_stage1_coordination_scientific_identity_v1"
)

ROLE_NEUTRAL_COORDINATION_MANIFEST = "coordination_manifest.json"
ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION = "component_locator_attestation.json"
ROLE_NEUTRAL_SCIENTIFIC_BINDING_DIRECTORY = "scientific_bindings"

_COMPONENT_TERMINAL_FILE = "execution_manifest.json"
_HEX = frozenset("0123456789abcdef")


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


def _duplicate_rejecting_object(
    pairs: Sequence[tuple[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{label} contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one regular JSON file")
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_nlink) != 1:
        raise ValueError(f"{label} must be private regular data")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=lambda pairs: _duplicate_rejecting_object(
                pairs,
                label=label,
            ),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains {constant}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not closed UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_directory(value: Path | str, *, label: str) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if supplied.is_symlink():
        raise ValueError(f"{label} cannot be a symbolic link")
    resolved = supplied.resolve(strict=True)
    if resolved != supplied or not resolved.is_dir():
        raise ValueError(f"{label} must be one canonical directory")
    return resolved


def _stable_private_file_sha256(path: Path) -> tuple[str, int]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ValueError("role-neutral component files must be private regular data")
        digest = hashlib.sha256()
        size = 0
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_mode),
        int(before.st_nlink),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    if identity != (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_mode),
        int(after.st_nlink),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    ) or size != int(after.st_size):
        raise RuntimeError(f"artifact changed while hashing: {path}")
    named = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(named.st_mode)
        or int(named.st_nlink) != 1
        or (int(named.st_dev), int(named.st_ino)) != (int(after.st_dev), int(after.st_ino))
    ):
        raise RuntimeError(f"artifact path was substituted while hashing: {path}")
    return digest.hexdigest(), size


def _component_tree_sha256(root: Path) -> str:
    inventory: list[dict[str, Any]] = []
    seen_inodes: set[tuple[int, int]] = set()
    for path in sorted(
        root.rglob("*"),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"role-neutral component contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_nlink) != 1:
            raise ValueError("role-neutral component contains non-private data: " f"{relative}")
        inode = (int(metadata.st_dev), int(metadata.st_ino))
        if inode in seen_inodes:
            raise ValueError("role-neutral component contains a hard-linked alias")
        seen_inodes.add(inode)
        digest, size = _stable_private_file_sha256(path)
        inventory.append(
            {
                "relative_path": relative,
                "size_bytes": size,
                "sha256": digest,
            }
        )
    if not inventory:
        raise ValueError("role-neutral component tree is empty")
    return _sha256_json(
        {
            "schema_version": "production_role_neutral_component_tree_v1",
            "files": inventory,
        }
    )


def _directory_tree_registration(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        metadata = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"published gate contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_nlink) != 1:
            raise ValueError(f"published gate contains non-private data: {relative}")
        digest, size = _stable_private_file_sha256(path)
        files.append(
            {
                "relative_path": relative,
                "size_bytes": size,
                "sha256": digest,
            }
        )
    body = {
        "schema_version": ("production_role_neutral_scientific_binding_tree_v1"),
        "files": files,
    }
    return {**body, "content_sha256": _sha256_json(body)}


@dataclass(frozen=True)
class RoleNeutralComponentArtifactSource:
    """One receipt paired with the exact component root it authenticated."""

    root: Path
    receipt: AuthenticatedRoleNeutralComponentReceipt


def _normalize_component_sources(
    *,
    plan: Stage1ScopePlan,
    sources_by_physical_owner: Mapping[str, Sequence[RoleNeutralComponentArtifactSource]],
) -> tuple[
    dict[str, tuple[AuthenticatedRoleNeutralComponentReceipt, ...]],
    list[dict[str, Any]],
]:
    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError("role-neutral coordination requires a Stage1ScopePlan")
    owner_ids = tuple(owner.scope_id for owner in plan.physical_scopes)
    if not isinstance(sources_by_physical_owner, Mapping) or set(sources_by_physical_owner) != set(
        owner_ids
    ):
        raise ValueError("role-neutral component sources must cover every physical owner")

    normalized: dict[str, tuple[AuthenticatedRoleNeutralComponentReceipt, ...]] = {}
    registrations: list[dict[str, Any]] = []
    seen_roots: list[Path] = []
    seen_root_inodes: set[tuple[int, int]] = set()
    component_order = tuple(EXPECTED_COMPONENT_FAMILIES)
    for owner in plan.physical_scopes:
        raw_sources = tuple(sources_by_physical_owner[owner.scope_id])
        if any(
            not isinstance(source, RoleNeutralComponentArtifactSource) for source in raw_sources
        ):
            raise TypeError("role-neutral component sources require typed receipt/root pairs")
        if any(
            not isinstance(
                source.receipt,
                AuthenticatedRoleNeutralComponentReceipt,
            )
            for source in raw_sources
        ):
            raise TypeError(
                "role-neutral component sources require authenticated " "component receipts"
            )
        by_component = {source.receipt.component: source for source in raw_sources}
        if (
            len(raw_sources) != len(component_order)
            or len(by_component) != len(raw_sources)
            or set(by_component) != set(component_order)
        ):
            raise ValueError(
                f"{owner.scope_id} must supply exactly the canonical six "
                "role-neutral producer components"
            )
        owner_receipts: list[AuthenticatedRoleNeutralComponentReceipt] = []
        for component in component_order:
            source = by_component[component]
            root = _canonical_directory(
                source.root,
                label=f"{owner.scope_id}/{component} component root",
            )
            root_metadata = os.stat(root, follow_symlinks=False)
            root_inode = (
                int(root_metadata.st_dev),
                int(root_metadata.st_ino),
            )
            if root_inode in seen_root_inodes or any(
                root == previous or root in previous.parents or previous in root.parents
                for previous in seen_roots
            ):
                raise ValueError(
                    "each physical owner/component requires one distinct, " "nonnested source root"
                )
            receipt = validate_authenticated_role_neutral_component_receipt(
                root=root,
                plan=plan,
                physical_owner_scope_id=owner.scope_id,
                receipt=source.receipt,
                expected_component=component,
            )
            seen_roots.append(root)
            seen_root_inodes.add(root_inode)
            owner_receipts.append(receipt)
            registrations.append(
                {
                    "physical_owner_scope_id": owner.scope_id,
                    "component": component,
                    "families": list(EXPECTED_COMPONENT_FAMILIES[component]),
                    "logical_scope_ids": list(receipt.logical_scope_ids),
                    "component_scientific_receipt": (receipt.scientific_dict()),
                    "component_authentication_content_sha256": (
                        receipt.authentication_content_sha256
                    ),
                    "source_terminal_content_sha256": (receipt.source_terminal_content_sha256),
                    "source_tree_sha256": receipt.source_tree_sha256,
                    "absolute_root_locator": str(root),
                    "registered_heldout_labels_accessed": False,
                    "oracle_fields_accessed": False,
                    "text_truncation_applied": False,
                    "lossy_evidence_selection_applied": False,
                }
            )
        normalized[owner.scope_id] = tuple(owner_receipts)
    return normalized, registrations


def _scientific_identity(
    *,
    plan: Stage1ScopePlan,
    binding_terminal: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": (ROLE_NEUTRAL_COORDINATION_SCIENTIFIC_IDENTITY_SCHEMA),
        "plan_scientific_content_sha256": plan.scientific_content_sha256,
        "physical_owner_scope_order": [owner.scope_id for owner in plan.physical_scopes],
        "logical_scope_order": [scope.scope_id for scope in plan.scopes],
        "canonical_component_order": list(EXPECTED_COMPONENT_FAMILIES),
        "canonical_component_family_partition": {
            component: list(families) for component, families in EXPECTED_COMPONENT_FAMILIES.items()
        },
        "binding_terminal_content_sha256": _require_sha256(
            binding_terminal.get("content_sha256"),
            label="role-neutral binding terminal content",
        ),
        "all_ten_fit_side_families_authenticated": True,
        "component_root_locators_included": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _locator_attestation(
    *,
    plan: Stage1ScopePlan,
    registrations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema_version": (ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION_SCHEMA),
        "plan_scientific_content_sha256": plan.scientific_content_sha256,
        "physical_owner_scope_order": [owner.scope_id for owner in plan.physical_scopes],
        "canonical_component_order": list(EXPECTED_COMPONENT_FAMILIES),
        "registration_count": len(registrations),
        "registrations": copy.deepcopy(list(registrations)),
        "component_roots_distinct_and_nonnested": True,
        "all_registered_bytes_reopened": True,
        "locator_metadata_excluded_from_scientific_identity": True,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def publish_role_neutral_stage1_coordination_gate(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
    sources_by_physical_owner: Mapping[str, Sequence[RoleNeutralComponentArtifactSource]],
) -> dict[str, Any]:
    """Publish one explicit all-ten production gate from authenticated roots."""

    destination = Path(root)
    if not destination.is_absolute():
        raise ValueError("role-neutral coordination root must be absolute")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("role-neutral coordination root must be fresh")
    parent = _canonical_directory(
        destination.parent,
        label="role-neutral coordination parent",
    )
    if parent != destination.parent:
        raise ValueError("role-neutral coordination parent must be canonical")
    receipts_by_owner, registrations = _normalize_component_sources(
        plan=plan,
        sources_by_physical_owner=sources_by_physical_owner,
    )
    for registration in registrations:
        source = Path(registration["absolute_root_locator"])
        if destination == source or destination in source.parents or source in destination.parents:
            raise ValueError("component roots and coordination output must be disjoint")

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.role-neutral-gate-",
            dir=parent,
        )
    )
    try:
        binding_root = temporary / ROLE_NEUTRAL_SCIENTIFIC_BINDING_DIRECTORY
        binding_terminal = persist_complete_role_neutral_stage1_bindings(
            root=binding_root,
            plan=plan,
            components_by_physical_owner=receipts_by_owner,
        )
        if (
            validate_complete_role_neutral_stage1_bindings(
                root=binding_root,
                plan=plan,
            )
            != binding_terminal
        ):
            raise RuntimeError("role-neutral scientific binding changed after publication")
        binding_tree = _directory_tree_registration(binding_root)
        locator_attestation = _locator_attestation(
            plan=plan,
            registrations=registrations,
        )
        attestation_path = temporary / ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION
        _write_new_json(attestation_path, locator_attestation)
        attestation_sha256, attestation_size = _stable_private_file_sha256(attestation_path)
        scientific_identity = _scientific_identity(
            plan=plan,
            binding_terminal=binding_terminal,
        )
        manifest_body = {
            "schema_version": ROLE_NEUTRAL_COORDINATION_GATE_SCHEMA,
            "status": "complete",
            "plan_scientific_content_sha256": (plan.scientific_content_sha256),
            "scientific_identity": scientific_identity,
            "scientific_binding_directory": (ROLE_NEUTRAL_SCIENTIFIC_BINDING_DIRECTORY),
            "scientific_binding_tree": binding_tree,
            "component_locator_attestation": {
                "relative_path": (ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION),
                "sha256": attestation_sha256,
                "size_bytes": attestation_size,
                "content_sha256": locator_attestation["content_sha256"],
            },
            "physical_fit_count": len(plan.physical_scopes),
            "logical_scope_count": len(plan.scopes),
            "deduplicated_fit_count": (len(plan.scopes) - len(plan.physical_scopes)),
            "producer_component_count_per_physical_owner": len(EXPECTED_COMPONENT_FAMILIES),
            "all_ten_fit_side_families_authenticated": True,
            "component_logical_and_numerical_roots_retained": True,
            "legacy_role_specific_fragments_adopted": False,
            "component_root_locators_in_scientific_identity": False,
            "event_order": [
                "all_component_roots_authenticated",
                "scientific_bindings_published",
                "component_locator_attestation_published",
                "coordination_gate_sealed",
            ],
        }
        manifest = {
            **manifest_body,
            "content_sha256": _sha256_json(manifest_body),
        }
        _write_new_json(
            temporary / ROLE_NEUTRAL_COORDINATION_MANIFEST,
            manifest,
        )
        durably_sync_legacy_stage1_tree(temporary)
        os.replace(temporary, destination)
        descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_role_neutral_stage1_coordination_gate(
        root=destination,
        plan=plan,
    )


def _validate_scientific_receipt_registration(
    *,
    registration: Mapping[str, Any],
    plan: Stage1ScopePlan,
    owner_scope_id: str,
    component: str,
    expected_logical_scope_ids: Sequence[str],
    expected_family_fit_ids: Mapping[str, str],
    expected_family_fit_seals: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_fields = {
        "physical_owner_scope_id",
        "component",
        "families",
        "logical_scope_ids",
        "component_scientific_receipt",
        "component_authentication_content_sha256",
        "source_terminal_content_sha256",
        "source_tree_sha256",
        "absolute_root_locator",
        "registered_heldout_labels_accessed",
        "oracle_fields_accessed",
        "text_truncation_applied",
        "lossy_evidence_selection_applied",
    }
    scientific = registration.get("component_scientific_receipt")
    expected_families = tuple(EXPECTED_COMPONENT_FAMILIES[component])
    expected_scientific_fields = {
        "schema_version",
        "component",
        "plan_scientific_content_sha256",
        "physical_owner_scope_id",
        "logical_scope_ids",
        "family_fit_artifact_sha256",
        "family_logical_view_content_sha256",
        "source_terminal_content_sha256",
        "registered_heldout_labels_accessed",
        "oracle_fields_accessed",
        "text_truncation_applied",
        "lossy_evidence_selection_applied",
        "execution_locator_metadata_in_scientific_identity",
        "content_sha256",
    }
    if (
        set(registration) != expected_fields
        or registration.get("physical_owner_scope_id") != owner_scope_id
        or registration.get("component") != component
        or registration.get("families") != list(expected_families)
        or registration.get("logical_scope_ids") != list(expected_logical_scope_ids)
        or not isinstance(scientific, Mapping)
        or set(scientific) != expected_scientific_fields
        or scientific.get("schema_version") != ROLE_NEUTRAL_COMPONENT_RECEIPT_SCHEMA
        or scientific.get("component") != component
        or scientific.get("plan_scientific_content_sha256") != plan.scientific_content_sha256
        or scientific.get("physical_owner_scope_id") != owner_scope_id
        or scientific.get("logical_scope_ids") != list(expected_logical_scope_ids)
        or scientific.get("source_terminal_content_sha256")
        != registration.get("source_terminal_content_sha256")
        or set(scientific.get("family_fit_artifact_sha256") or {}) != set(expected_families)
        or scientific.get("family_fit_artifact_sha256")
        != {family: expected_family_fit_ids[family] for family in expected_families}
        or set(scientific.get("family_logical_view_content_sha256") or {}) != set(expected_families)
        or scientific.get("execution_locator_metadata_in_scientific_identity") is not False
        or scientific.get("registered_heldout_labels_accessed") is not False
        or scientific.get("oracle_fields_accessed") is not False
        or scientific.get("text_truncation_applied") is not False
        or scientific.get("lossy_evidence_selection_applied") is not False
        or registration.get("registered_heldout_labels_accessed") is not False
        or registration.get("oracle_fields_accessed") is not False
        or registration.get("text_truncation_applied") is not False
        or registration.get("lossy_evidence_selection_applied") is not False
    ):
        raise ValueError(f"{owner_scope_id}/{component} locator registration changed")
    scientific_body = {
        key: copy.deepcopy(value) for key, value in scientific.items() if key != "content_sha256"
    }
    if scientific.get("content_sha256") != _sha256_json(scientific_body):
        raise ValueError(f"{owner_scope_id}/{component} scientific receipt is invalid")
    for family in expected_families:
        views = scientific["family_logical_view_content_sha256"][family]
        if (
            not isinstance(views, Mapping)
            or set(views) != set(expected_logical_scope_ids)
            or len(set(views.values())) != len(expected_logical_scope_ids)
        ):
            raise ValueError(f"{owner_scope_id}/{component}/{family} logical views changed")
        for scope_id in expected_logical_scope_ids:
            _require_sha256(
                views[scope_id],
                label=(f"{owner_scope_id}/{component}/{family}/{scope_id} " "logical view"),
            )
    authentication_body = {
        "schema_version": ("production_role_neutral_component_authenticated_handle_v1"),
        "component": component,
        "plan_scientific_content_sha256": plan.scientific_content_sha256,
        "physical_owner_scope_id": owner_scope_id,
        "logical_scope_ids": list(expected_logical_scope_ids),
        "family_fit_seals": {
            family: expected_family_fit_seals[family] for family in expected_families
        },
        "family_logical_view_content_sha256": {
            family: scientific["family_logical_view_content_sha256"][family]
            for family in expected_families
        },
        "source_terminal_content_sha256": registration["source_terminal_content_sha256"],
        "source_tree_sha256": registration["source_tree_sha256"],
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
        "lossy_evidence_selection_applied": False,
    }
    if registration.get("component_authentication_content_sha256") != _sha256_json(
        authentication_body
    ):
        raise ValueError(f"{owner_scope_id}/{component} authentication handle changed")

    root = _canonical_directory(
        registration.get("absolute_root_locator"),
        label=f"{owner_scope_id}/{component} registered component root",
    )
    if _component_tree_sha256(root) != _require_sha256(
        registration.get("source_tree_sha256"),
        label=f"{owner_scope_id}/{component} source tree",
    ):
        raise ValueError(f"{owner_scope_id}/{component} registered component tree changed")
    terminal = _read_json(
        root / _COMPONENT_TERMINAL_FILE,
        label=f"{owner_scope_id}/{component} execution terminal",
    )
    if terminal.get("content_sha256") != _require_sha256(
        registration.get("source_terminal_content_sha256"),
        label=f"{owner_scope_id}/{component} source terminal",
    ):
        raise ValueError(f"{owner_scope_id}/{component} execution terminal changed")


def validate_role_neutral_stage1_coordination_gate(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
) -> dict[str, Any]:
    """Fresh path-only validation of bindings and every source component."""

    tree = _canonical_directory(
        root,
        label="role-neutral coordination root",
    )
    top_level = {path.name for path in tree.iterdir()}
    if top_level != {
        ROLE_NEUTRAL_COORDINATION_MANIFEST,
        ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION,
        ROLE_NEUTRAL_SCIENTIFIC_BINDING_DIRECTORY,
    }:
        raise ValueError("role-neutral coordination root has extra/missing data")
    manifest = _read_json(
        tree / ROLE_NEUTRAL_COORDINATION_MANIFEST,
        label="role-neutral coordination manifest",
    )
    manifest_body = {
        key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"
    }
    expected_manifest_fields = {
        "schema_version",
        "status",
        "plan_scientific_content_sha256",
        "scientific_identity",
        "scientific_binding_directory",
        "scientific_binding_tree",
        "component_locator_attestation",
        "physical_fit_count",
        "logical_scope_count",
        "deduplicated_fit_count",
        "producer_component_count_per_physical_owner",
        "all_ten_fit_side_families_authenticated",
        "component_logical_and_numerical_roots_retained",
        "legacy_role_specific_fragments_adopted",
        "component_root_locators_in_scientific_identity",
        "event_order",
        "content_sha256",
    }
    if (
        not isinstance(plan, Stage1ScopePlan)
        or set(manifest) != expected_manifest_fields
        or manifest.get("schema_version") != ROLE_NEUTRAL_COORDINATION_GATE_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("plan_scientific_content_sha256") != plan.scientific_content_sha256
        or manifest.get("scientific_binding_directory") != ROLE_NEUTRAL_SCIENTIFIC_BINDING_DIRECTORY
        or manifest.get("physical_fit_count") != len(plan.physical_scopes)
        or manifest.get("logical_scope_count") != len(plan.scopes)
        or manifest.get("deduplicated_fit_count") != len(plan.scopes) - len(plan.physical_scopes)
        or manifest.get("producer_component_count_per_physical_owner")
        != len(EXPECTED_COMPONENT_FAMILIES)
        or manifest.get("all_ten_fit_side_families_authenticated") is not True
        or manifest.get("component_logical_and_numerical_roots_retained") is not True
        or manifest.get("legacy_role_specific_fragments_adopted") is not False
        or manifest.get("component_root_locators_in_scientific_identity") is not False
        or manifest.get("event_order")
        != [
            "all_component_roots_authenticated",
            "scientific_bindings_published",
            "component_locator_attestation_published",
            "coordination_gate_sealed",
        ]
        or manifest.get("content_sha256") != _sha256_json(manifest_body)
    ):
        raise ValueError("role-neutral coordination manifest is invalid")

    binding_root = tree / ROLE_NEUTRAL_SCIENTIFIC_BINDING_DIRECTORY
    binding_terminal = validate_complete_role_neutral_stage1_bindings(
        root=binding_root,
        plan=plan,
    )
    if manifest.get("scientific_binding_tree") != _directory_tree_registration(
        binding_root
    ) or manifest.get("scientific_identity") != _scientific_identity(
        plan=plan,
        binding_terminal=binding_terminal,
    ):
        raise ValueError("role-neutral scientific binding registration changed")

    attestation_registration = manifest.get("component_locator_attestation")
    if not isinstance(attestation_registration, Mapping) or set(attestation_registration) != {
        "relative_path",
        "sha256",
        "size_bytes",
        "content_sha256",
    }:
        raise ValueError("component locator attestation registration is invalid")
    relative = PurePosixPath(str(attestation_registration.get("relative_path")))
    if relative.parts != (ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION,) or relative.is_absolute():
        raise ValueError("component locator attestation path is noncanonical")
    attestation_path = tree / relative.as_posix()
    attestation_sha, attestation_size = _stable_private_file_sha256(attestation_path)
    attestation = _read_json(
        attestation_path,
        label="component locator attestation",
    )
    attestation_body = {
        key: copy.deepcopy(value) for key, value in attestation.items() if key != "content_sha256"
    }
    expected_attestation_fields = {
        "schema_version",
        "plan_scientific_content_sha256",
        "physical_owner_scope_order",
        "canonical_component_order",
        "registration_count",
        "registrations",
        "component_roots_distinct_and_nonnested",
        "all_registered_bytes_reopened",
        "locator_metadata_excluded_from_scientific_identity",
        "content_sha256",
    }
    expected_registration_count = len(plan.physical_scopes) * len(EXPECTED_COMPONENT_FAMILIES)
    registrations = attestation.get("registrations")
    if (
        dict(attestation_registration)
        != {
            "relative_path": (ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION),
            "sha256": attestation_sha,
            "size_bytes": attestation_size,
            "content_sha256": attestation.get("content_sha256"),
        }
        or set(attestation) != expected_attestation_fields
        or attestation.get("schema_version") != ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION_SCHEMA
        or attestation.get("plan_scientific_content_sha256") != plan.scientific_content_sha256
        or attestation.get("physical_owner_scope_order")
        != [owner.scope_id for owner in plan.physical_scopes]
        or attestation.get("canonical_component_order") != list(EXPECTED_COMPONENT_FAMILIES)
        or attestation.get("registration_count") != expected_registration_count
        or not isinstance(registrations, list)
        or len(registrations) != expected_registration_count
        or attestation.get("component_roots_distinct_and_nonnested") is not True
        or attestation.get("all_registered_bytes_reopened") is not True
        or attestation.get("locator_metadata_excluded_from_scientific_identity") is not True
        or attestation.get("content_sha256") != _sha256_json(attestation_body)
    ):
        raise ValueError("component locator attestation is invalid")

    binding_physical_rows = binding_terminal["physical_payloads"]
    binding_by_owner: dict[str, Mapping[str, Any]] = {}
    for row in binding_physical_rows:
        payload = _read_json(
            binding_root / str(row["relative_path"]),
            label=f"{row['physical_owner_scope_id']} scientific payload",
        )
        binding_by_owner[str(row["physical_owner_scope_id"])] = payload
    seen_paths: list[Path] = []
    seen_inodes: set[tuple[int, int]] = set()
    registration_index = 0
    for owner, members in plan.physical_scope_groups:
        expected_logical_scope_ids = tuple(member.scope_id for member in members)
        for component in EXPECTED_COMPONENT_FAMILIES:
            registration = registrations[registration_index]
            registration_index += 1
            if not isinstance(registration, Mapping):
                raise ValueError("component locator registration is malformed")
            _validate_scientific_receipt_registration(
                registration=registration,
                plan=plan,
                owner_scope_id=owner.scope_id,
                component=component,
                expected_logical_scope_ids=expected_logical_scope_ids,
                expected_family_fit_ids=binding_by_owner[owner.scope_id][
                    "family_fit_artifact_sha256"
                ],
                expected_family_fit_seals=binding_by_owner[owner.scope_id]["family_fit_seals"],
            )
            source = Path(str(registration["absolute_root_locator"]))
            source_metadata = os.stat(source, follow_symlinks=False)
            inode = (
                int(source_metadata.st_dev),
                int(source_metadata.st_ino),
            )
            if inode in seen_inodes or any(
                source == prior or source in prior.parents or prior in source.parents
                for prior in seen_paths
            ):
                raise ValueError("component locator roots are duplicated or nested")
            seen_paths.append(source)
            seen_inodes.add(inode)
    if registration_index != len(registrations):
        raise ValueError("component locator registration order changed")
    return manifest


__all__ = [
    "ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION_SCHEMA",
    "ROLE_NEUTRAL_COORDINATION_GATE_SCHEMA",
    "ROLE_NEUTRAL_COORDINATION_SCIENTIFIC_IDENTITY_SCHEMA",
    "RoleNeutralComponentArtifactSource",
    "publish_role_neutral_stage1_coordination_gate",
    "validate_role_neutral_stage1_coordination_gate",
]
