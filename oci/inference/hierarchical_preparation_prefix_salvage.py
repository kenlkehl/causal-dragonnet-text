"""Authenticate and export a completed prefix of hierarchical preparation folds.

This module is deliberately separate from the live preparation runner.  It reads
only immutable, high-level completion artifacts and emits registrations that the
existing read-only cache overlays can authenticate in a later fresh process.

Executable neural-query joblib files, backend work directories, and intermediate
fit-call checkpoints are never eligible sources.  A fold is salvageable only when
its immutable fold-preparation manifest exists, its exact initial-spent JSON cache
reconstructs the catalog named by that manifest, and its complete top-level gate
bundle authenticates through the ordinary context-fit overlay boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

from .all_evidence_fusion import FoldEvidenceInput, FoldEvidenceProvenance
from .authenticated_semantic_retrieval_compatibility import (
    current_spent_projection_compatibility_identity,
    restore_current_spent_projection_semantic_retrieval_view,
)
from .context_fit_upstream_cache_overlay import (
    CONTEXT_FIT_CACHE_INDEX_SCHEMA_VERSION,
    AuthenticatedContextFitCacheSource,
    authenticate_context_fit_cache_index_registrations,
)
from .fold_honest_signal_fusion import row_set_fingerprint
from .lossless_stage1_evidence_catalog import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    build_role_neutral_evidence_catalog,
)
from .review_spent_evidence_cache_overlay import (
    AuthenticatedReviewSpentCacheSource,
    authenticate_review_spent_cache_registrations,
)

HIERARCHICAL_PREPARATION_PREFIX_SALVAGE_SCHEMA_VERSION = (
    "hierarchical_preparation_completed_prefix_salvage_v1"
)
HIERARCHICAL_PREPARATION_INPUT_SCHEMA_VERSION = (
    "hierarchical_all_evidence_runner_preparation_input_v2"
)
HIERARCHICAL_PREPARATION_FOLD_SCHEMA_VERSION = (
    "hierarchical_all_evidence_runner_fold_preparation_v2"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HASH_WRAPPER_FIELDS = frozenset({"schema_version", "body", "content_sha256"})
_REPLAY_FIELDS = frozenset(
    {
        "schema_version",
        "source_input_manifest",
        "completed_outer_folds",
        "completed_fold_manifests",
        "review_spent_evidence_cache",
        "context_fit_cache_index",
        "assurances",
        "content_sha256",
    }
)
_CANONICAL_GATE_FILES = (
    "calibrated_sources.npy",
    "features.npy",
    "calibrated_sources_context_oof.npy",
    "features_context_oof.npy",
)
_FORBIDDEN_SOURCE_SUFFIXES = frozenset({".joblib", ".pkl", ".pickle", ".pt", ".pth", ".ckpt"})


class HierarchicalPreparationPrefixSalvageError(RuntimeError):
    """A purported completed preparation prefix failed authentication."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _required_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip()
    if _SHA256.fullmatch(digest) is None:
        raise HierarchicalPreparationPrefixSalvageError(f"{label} must be one lowercase SHA-256")
    return digest


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON field {key!r}")
        output[key] = value
    return output


def _parse_json(snapshot: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_closed_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HierarchicalPreparationPrefixSalvageError(
            f"{label} is not closed finite UTF-8 JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise HierarchicalPreparationPrefixSalvageError(f"{label} root must be an object")
    return value


def _read_snapshot(path: Path, *, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise HierarchicalPreparationPrefixSalvageError(f"{label} is unreadable: {path}") from exc


def _read_hash_wrapper(
    path: Path,
    *,
    label: str,
    expected_schema: str | None = None,
) -> tuple[Mapping[str, Any], bytes]:
    snapshot = _read_snapshot(path, label=label)
    wrapper = _parse_json(snapshot, label=label)
    if set(wrapper) != _HASH_WRAPPER_FIELDS:
        raise HierarchicalPreparationPrefixSalvageError(
            f"{label} has an unsupported closed wrapper schema"
        )
    if expected_schema is not None and wrapper["schema_version"] != expected_schema:
        raise HierarchicalPreparationPrefixSalvageError(f"{label} has an unexpected schema version")
    body = wrapper["body"]
    if not isinstance(body, Mapping) or wrapper["content_sha256"] != _sha256_json(body):
        raise HierarchicalPreparationPrefixSalvageError(f"{label} content hash mismatch")
    return wrapper, snapshot


def _identity_record(value: Any, *, label: str) -> tuple[Mapping[str, Any], str]:
    if not isinstance(value, Mapping) or set(value) != {"identity", "identity_sha256"}:
        raise HierarchicalPreparationPrefixSalvageError(f"{label} is malformed")
    identity = value["identity"]
    digest = _required_sha256(value["identity_sha256"], label=f"{label}.identity_sha256")
    if not isinstance(identity, Mapping) or _sha256_json(identity) != digest:
        raise HierarchicalPreparationPrefixSalvageError(f"{label} identity hash mismatch")
    return identity, digest


def _validated_root(value: Path | str, *, label: str) -> Path:
    raw = Path(value).expanduser().absolute()
    if not raw.is_dir() or raw.is_symlink():
        raise HierarchicalPreparationPrefixSalvageError(
            f"{label} must be an existing non-symlink directory"
        )
    return raw.resolve(strict=True)


def _validated_regular_under(
    value: Path | str,
    *,
    root: Path,
    label: str,
) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    candidate = candidate.absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise HierarchicalPreparationPrefixSalvageError(
            f"{label} escaped its authenticated root"
        ) from exc
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise HierarchicalPreparationPrefixSalvageError(f"{label} contains a symlink component")
    if not candidate.is_file():
        raise HierarchicalPreparationPrefixSalvageError(
            f"{label} is not a complete regular file: {candidate}"
        )
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise HierarchicalPreparationPrefixSalvageError(
            f"{label} resolved outside its authenticated root"
        )
    return resolved


def _path_from_body(
    body: Mapping[str, Any],
    field: str,
    *,
    root: Path,
    label: str,
) -> Path:
    value = body.get(field)
    if not isinstance(value, str) or not value.strip():
        raise HierarchicalPreparationPrefixSalvageError(f"{label} lacks {field}")
    return _validated_regular_under(value, root=root, label=f"{label}.{field}")


def _integer_rows(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise HierarchicalPreparationPrefixSalvageError(
            f"{label} must be a sequence of integer row IDs"
        )
    output: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise HierarchicalPreparationPrefixSalvageError(
                f"{label} contains a non-canonical row ID"
            )
        output.append(int(item))
    if not output or len(output) != len(set(output)):
        raise HierarchicalPreparationPrefixSalvageError(f"{label} must be non-empty and unique")
    return tuple(output)


def _schedule_rows(
    schedule: Any,
    *,
    outer_fold: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if not isinstance(schedule, Mapping) or schedule.get("outer_fold") != outer_fold:
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} schedule audit is malformed"
        )
    partitions = schedule.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} schedule has no partitions"
        )
    rows_by_fold: dict[int, tuple[int, ...]] = {}
    all_rows: set[int] = set()
    for row in partitions:
        if not isinstance(row, Mapping):
            raise HierarchicalPreparationPrefixSalvageError(
                f"fold {outer_fold} schedule partition is malformed"
            )
        fold_id = row.get("fold_id")
        if isinstance(fold_id, bool) or not isinstance(fold_id, int) or fold_id < 1:
            raise HierarchicalPreparationPrefixSalvageError(
                f"fold {outer_fold} schedule partition ID is malformed"
            )
        if fold_id in rows_by_fold:
            raise HierarchicalPreparationPrefixSalvageError(
                f"fold {outer_fold} schedule repeats a partition"
            )
        partition_rows = _integer_rows(
            row.get("row_ids"), label=f"fold {outer_fold} partition {fold_id} rows"
        )
        if all_rows & set(partition_rows):
            raise HierarchicalPreparationPrefixSalvageError(
                f"fold {outer_fold} schedule partitions overlap"
            )
        all_rows.update(partition_rows)
        rows_by_fold[int(fold_id)] = partition_rows
    if tuple(sorted(rows_by_fold)) != tuple(range(1, len(rows_by_fold) + 1)):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} schedule partitions are not contiguous"
        )
    initial = tuple(schedule.get("initial_spent_fold_ids") or ())
    gates = tuple(schedule.get("gate_fold_ids") or ())
    if (
        not initial
        or not gates
        or any(isinstance(item, bool) or not isinstance(item, int) for item in (*initial, *gates))
        or tuple(initial) + tuple(gates) != tuple(sorted(rows_by_fold))
    ):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} spent/gate partition order is malformed"
        )
    initial_set = set(initial)
    spent = tuple(
        row_id
        for fold_id in sorted(rows_by_fold)
        if fold_id in initial_set
        for row_id in rows_by_fold[fold_id]
    )
    sealed = tuple(
        row_id
        for fold_id in sorted(rows_by_fold)
        if fold_id not in initial_set
        for row_id in rows_by_fold[fold_id]
    )
    first_gate = rows_by_fold[int(gates[0])]
    inner_fold_ids = tuple(
        fold_id
        for fold_id in sorted(rows_by_fold)
        if fold_id in initial_set
        for _row_id in rows_by_fold[fold_id]
    )
    return spent, sealed, first_gate, inner_fold_ids


def _validate_referenced_fold_artifacts(
    *,
    fold_body: Mapping[str, Any],
    fold_dir: Path,
    outer_fold: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[Path, str]]:
    catalog_path = _path_from_body(
        fold_body,
        "catalog_path",
        root=fold_dir,
        label=f"fold {outer_fold} preparation",
    )
    catalog_wrapper, catalog_snapshot = _read_hash_wrapper(
        catalog_path, label=f"fold {outer_fold} role-neutral catalog"
    )
    if catalog_wrapper["content_sha256"] != fold_body.get("catalog_envelope_content_sha256"):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} catalog envelope hash mismatch"
        )
    catalog = catalog_wrapper["body"]
    if (
        catalog.get("outer_fold") != outer_fold
        or catalog.get("scope") != "inner_train"
        or catalog.get("inner_fold") != 1
        or catalog.get("catalog_sha256") != fold_body.get("catalog_sha256")
    ):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} catalog binding is malformed"
        )

    chunk_path = _path_from_body(
        fold_body,
        "chunk_plan_path",
        root=fold_dir,
        label=f"fold {outer_fold} preparation",
    )
    chunk_wrapper, chunk_snapshot = _read_hash_wrapper(
        chunk_path, label=f"fold {outer_fold} architecture chunk plan"
    )
    if chunk_wrapper["content_sha256"] != fold_body.get(
        "chunk_plan_envelope_content_sha256"
    ) or chunk_wrapper["body"].get("plan_sha256") != fold_body.get("chunk_plan_sha256"):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} architecture chunk-plan binding changed"
        )

    wrapper_path = _path_from_body(
        fold_body,
        "wrapper_precommit_path",
        root=fold_dir,
        label=f"fold {outer_fold} preparation",
    )
    wrapper, wrapper_snapshot = _read_hash_wrapper(
        wrapper_path, label=f"fold {outer_fold} wrapper precommit"
    )
    wrapper_body = wrapper["body"]
    if (
        wrapper["content_sha256"] != fold_body.get("wrapper_precommit_envelope_content_sha256")
        or wrapper_body.get("approval_sha256") != fold_body.get("wrapper_approval_sha256")
        or _sha256_json(wrapper_body.get("packet")) != wrapper_body.get("approval_sha256")
    ):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} wrapper precommit binding changed"
        )

    direct_path = _path_from_body(
        fold_body,
        "direct_manifest_path",
        root=fold_dir,
        label=f"fold {outer_fold} preparation",
    )
    direct_snapshot = _read_snapshot(
        direct_path, label=f"fold {outer_fold} direct numerical manifest"
    )
    direct = _parse_json(direct_snapshot, label=f"fold {outer_fold} direct numerical manifest")
    if _sha256_bytes(direct_snapshot) != fold_body.get("direct_manifest_file_sha256") or direct.get(
        "content_sha256"
    ) != fold_body.get("direct_manifest_content_sha256"):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} direct numerical manifest file binding changed"
        )
    direct_content = {key: value for key, value in direct.items() if key != "content_sha256"}
    if direct.get("content_sha256") != _sha256_json(direct_content):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} direct numerical manifest content hash mismatch"
        )
    families = tuple(
        row.get("source_family")
        for row in direct.get("family_coverage", ())
        if isinstance(row, Mapping)
    )
    if (
        direct.get("all_active_stage1_architectures_covered") is not True
        or families != tuple(ACTIVE_STAGE1_CONCEPT_FAMILIES)
        or direct.get("semantic_catalog_sha256") != catalog.get("catalog_sha256")
    ):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} direct numerical manifest omits an active architecture"
        )
    return (
        catalog,
        direct,
        {
            catalog_path: _sha256_bytes(catalog_snapshot),
            chunk_path: _sha256_bytes(chunk_snapshot),
            wrapper_path: _sha256_bytes(wrapper_snapshot),
            direct_path: _sha256_bytes(direct_snapshot),
        },
    )


def _validate_spent_source(
    *,
    source: AuthenticatedReviewSpentCacheSource,
    outer_fold: int,
    spent_ids: tuple[int, ...],
    sealed_ids: tuple[int, ...],
    initial_binding: Mapping[str, Any],
    provider_identity: Mapping[str, Any],
    provider_identity_sha256: str,
    spent_evidence_provider: object,
    spent_request: Mapping[str, Any],
    semantic_retrieval_compatibility_audit: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> None:
    binding = source.binding
    backends = provider_identity.get("backends")
    if not isinstance(backends, list) or not backends:
        raise HierarchicalPreparationPrefixSalvageError(
            "spent provider identity has no backend identities"
        )
    expected = {
        "outer_fold": outer_fold,
        "review_round": 0,
        "spent_row_ids_sha256": _sha256_json(list(spent_ids)),
        "sealed_row_ids_sha256": _sha256_json(list(sealed_ids)),
        "ordered_spent_text_sha256": initial_binding.get("text_sha256"),
        "provider_identity_sha256": provider_identity_sha256,
        "backend_identities_sha256": _sha256_json(backends),
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} spent cache differs from the immutable fold binding"
        )
    if source.result_count != len(backends):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} spent cache backend result count changed"
        )
    parsed = _parse_json(source.snapshot, label=f"fold {outer_fold} spent cache")
    results = parsed.get("results")
    if not isinstance(results, list):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} spent cache results are malformed"
        )
    provenance = FoldEvidenceProvenance(
        outer_fold=outer_fold,
        train_row_ids=spent_ids,
        heldout_row_ids=sealed_ids,
        scope="inner_train",
        inner_fold=1,
        artifact_id=f"review-spent-{source.cache_key}",
    )
    inputs: list[FoldEvidenceInput] = []
    for row in results:
        if not isinstance(row, Mapping):
            raise HierarchicalPreparationPrefixSalvageError(
                f"fold {outer_fold} spent cache result is malformed"
            )
        payload = row.get("payload")
        if (
            not isinstance(payload, Mapping)
            or payload.get("outer_fold") != outer_fold
            or payload.get("scope") != "inner_train"
            or payload.get("inner_fold") != 1
        ):
            raise HierarchicalPreparationPrefixSalvageError(
                f"fold {outer_fold} spent cache result provenance changed"
            )
        inputs.append(
            FoldEvidenceInput(
                source_kind=str(row.get("source_kind") or ""),
                payload=payload,
                provenance=provenance,
            )
        )
    try:
        compatibility = restore_current_spent_projection_semantic_retrieval_view(
            tuple(inputs),
            spent_evidence_provider=spent_evidence_provider,
            outer_fold=spent_request.get("outer_fold"),
            review_round=spent_request.get("review_round"),
            exact_spent_row_ids=spent_request.get("exact_spent_row_ids"),
            exact_sealed_row_ids=spent_request.get("exact_sealed_row_ids"),
            spent_texts=spent_request.get("spent_texts"),
            spent_treatment=spent_request.get("spent_treatment"),
            spent_outcome=spent_request.get("spent_outcome"),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} raw spent cache failed semantic-retrieval "
            "compatibility authentication"
        ) from exc
    detached_audit = {
        **compatibility.audit,
        "ledger_content_sha256": compatibility.ledger_content_sha256,
        "restored_object_count": compatibility.restored_object_count,
    }
    if detached_audit != semantic_retrieval_compatibility_audit:
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} semantic-retrieval compatibility audit changed"
        )
    try:
        rebuilt = build_role_neutral_evidence_catalog(compatibility.evidence_inputs)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} compatibility view did not form a closed catalog"
        ) from exc
    if rebuilt.as_dict() != catalog:
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} spent cache does not reconstruct its immutable catalog"
        )
    family_counts = {
        family: len(rebuilt.family_atoms(family)) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    if any(count < 1 for count in family_counts.values()):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} spent cache omits an active Stage-1 architecture"
        )


def _spent_source_for_fold(
    *,
    sources: Sequence[AuthenticatedReviewSpentCacheSource],
    outer_fold: int,
    spent_ids: tuple[int, ...],
    sealed_ids: tuple[int, ...],
    initial_binding: Mapping[str, Any],
    provider_identity: Mapping[str, Any],
    provider_identity_sha256: str,
    spent_evidence_provider: object,
    spent_request: Mapping[str, Any],
    semantic_retrieval_compatibility_audit: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> AuthenticatedReviewSpentCacheSource:
    candidates = [
        source for source in sources if source.outer_fold == outer_fold and source.review_round == 0
    ]
    accepted: list[AuthenticatedReviewSpentCacheSource] = []
    failures: list[HierarchicalPreparationPrefixSalvageError] = []
    for source in candidates:
        try:
            _validate_spent_source(
                source=source,
                outer_fold=outer_fold,
                spent_ids=spent_ids,
                sealed_ids=sealed_ids,
                initial_binding=initial_binding,
                provider_identity=provider_identity,
                provider_identity_sha256=provider_identity_sha256,
                spent_evidence_provider=spent_evidence_provider,
                spent_request=spent_request,
                semantic_retrieval_compatibility_audit=(semantic_retrieval_compatibility_audit),
                catalog=catalog,
            )
        except HierarchicalPreparationPrefixSalvageError as exc:
            failures.append(exc)
            continue
        accepted.append(source)
    if not accepted and len(candidates) == 1 and len(failures) == 1:
        raise failures[0]
    if len(accepted) != 1:
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} must have exactly one complete catalog-bound top-level "
            "spent JSON cache"
        )
    return accepted[0]


def _gate_index_entry(
    *,
    gate_manifest: Path,
    companion_path: Path,
    companion_sha256: str,
) -> dict[str, Any]:
    manifest = _parse_json(
        _read_snapshot(gate_manifest, label="complete top-level gate manifest"),
        label="complete top-level gate manifest",
    )
    file_fields = {
        str(manifest.get("source_values_file")): manifest.get("source_values_sha256"),
        str(manifest.get("feature_values_file")): manifest.get("feature_values_sha256"),
        str(manifest.get("source_context_values_file")): manifest.get(
            "source_context_values_sha256"
        ),
        str(manifest.get("feature_context_values_file")): manifest.get(
            "feature_context_values_sha256"
        ),
    }
    if set(file_fields) != set(_CANONICAL_GATE_FILES):
        raise HierarchicalPreparationPrefixSalvageError(
            "complete top-level gate manifest does not name the canonical matrix set"
        )
    return {
        "kind": "review_gate",
        "cache_manifest_path": str(gate_manifest),
        "cache_manifest_sha256": _sha256_bytes(gate_manifest.read_bytes()),
        "cache_files": {
            filename: _required_sha256(file_fields[filename], label=f"gate {filename} SHA-256")
            for filename in _CANONICAL_GATE_FILES
        },
        "run_manifest_path": str(companion_path),
        "run_manifest_sha256": companion_sha256,
    }


def _validate_gate_source(
    *,
    source: AuthenticatedContextFitCacheSource,
    outer_fold: int,
    spent_ids: tuple[int, ...],
    first_gate_ids: tuple[int, ...],
    inner_fold_ids: tuple[int, ...],
    initial_binding: Mapping[str, Any],
    gate_binding: Mapping[str, Any],
    gate_provider_identity: Mapping[str, Any],
    gate_provider_identity_sha256: str,
    direct: Mapping[str, Any],
    fold_body: Mapping[str, Any],
) -> None:
    if source.kind != "review_gate":
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} indexed cache is not a review-gate bundle"
        )
    binding = source.binding
    expected_binding = {
        "outer_fold": outer_fold,
        "provider_identity": gate_provider_identity,
        "context_row_ids_sha256": _sha256_json(list(spent_ids)),
        "context_text_sha256": initial_binding.get("text_sha256"),
        "context_treatment_sha256": initial_binding.get("treatment_sha256"),
        "context_outcome_sha256": initial_binding.get("outcome_sha256"),
        "context_inner_fold_assignment_sha256": initial_binding.get("inner_fold_assignment_sha256"),
        "gate_row_ids_sha256": _sha256_json(list(first_gate_ids)),
        "gate_text_sha256": gate_binding.get("text_sha256"),
        "context_row_count": len(spent_ids),
        "gate_row_count": len(first_gate_ids),
        "gate_labels_in_binding": False,
        "gate_labels_exposed_to_backend": False,
    }
    if any(binding.get(key) != value for key, value in expected_binding.items()):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} top-level gate binding differs from its immutable audit"
        )
    manifest = _parse_json(source.manifest_snapshot, label=f"fold {outer_fold} gate manifest")
    if (
        tuple(manifest.get("context_row_ids") or ()) != spent_ids
        or tuple(manifest.get("context_inner_fold_ids") or ()) != inner_fold_ids
        or tuple(manifest.get("gate_row_ids") or ()) != first_gate_ids
    ):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} top-level gate manifest changed exact row order"
        )
    audit = fold_body.get("first_gate_preparation_audit")
    upstream = audit.get("upstream_cache_binding") if isinstance(audit, Mapping) else None
    bound_identity = fold_body.get("first_gate_provider_identity")
    if not isinstance(upstream, Mapping) or not isinstance(bound_identity, Mapping):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} first-gate provider audit is malformed"
        )
    expected_bound_identity = {
        "outer_fold": outer_fold,
        "gate_row_ids_sha256": _sha256_json(list(first_gate_ids)),
        "parent_identity_sha256": gate_provider_identity_sha256,
        "cache_manifest_sha256": source.cache_manifest_sha256,
    }
    if any(bound_identity.get(key) != value for key, value in expected_bound_identity.items()):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} bound first-gate provider identity changed"
        )
    if (
        _sha256_json(bound_identity) != upstream.get("bound_provider_identity_sha256")
        or upstream.get("source_cache_key") != source.cache_key
        or upstream.get("source_manifest_sha256") != source.cache_manifest_sha256
        or fold_body.get("authenticated_first_gate_cache_manifest_sha256")
        != source.cache_manifest_sha256
        or direct.get("source_cache_key") != source.cache_key
        or direct.get("source_manifest_sha256") != source.cache_manifest_sha256
    ):
        raise HierarchicalPreparationPrefixSalvageError(
            f"fold {outer_fold} direct/gate cache authentication binding changed"
        )


def _write_plain_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(encoded)


@dataclass(frozen=True)
class CompletedHierarchicalPreparationPrefixSalvage:
    """Authenticated registrations exported from a completed fold prefix."""

    destination: Path
    completed_outer_folds: tuple[int, ...]
    review_spent_registrations: tuple[str, ...]
    context_fit_index_registration: str
    replay_manifest_registration: str

    @property
    def execution_arguments(self) -> tuple[str, ...]:
        return (
            *(
                "--read-only-review-spent-evidence-cache=" + registration
                for registration in self.review_spent_registrations
            ),
            "--read-only-context-fit-cache-index=" + self.context_fit_index_registration,
        )

    def validate_authentication(self) -> None:
        replay_path_raw, separator, replay_sha = self.replay_manifest_registration.rpartition("::")
        if not separator:
            raise HierarchicalPreparationPrefixSalvageError(
                "replay manifest registration is malformed"
            )
        replay_path = Path(replay_path_raw).resolve(strict=True)
        snapshot = _read_snapshot(replay_path, label="completed-prefix replay manifest")
        if _sha256_bytes(snapshot) != _required_sha256(replay_sha, label="replay manifest SHA-256"):
            raise HierarchicalPreparationPrefixSalvageError(
                "completed-prefix replay manifest SHA-256 mismatch"
            )
        payload = _parse_json(snapshot, label="completed-prefix replay manifest")
        if (
            set(payload) != _REPLAY_FIELDS
            or payload["schema_version"] != HIERARCHICAL_PREPARATION_PREFIX_SALVAGE_SCHEMA_VERSION
        ):
            raise HierarchicalPreparationPrefixSalvageError(
                "completed-prefix replay manifest has a wrong closed schema"
            )
        content = {key: value for key, value in payload.items() if key != "content_sha256"}
        if payload["content_sha256"] != _sha256_json(content):
            raise HierarchicalPreparationPrefixSalvageError(
                "completed-prefix replay manifest content hash mismatch"
            )
        folds = tuple(payload["completed_outer_folds"])
        if folds != self.completed_outer_folds or folds != tuple(range(1, len(folds) + 1)):
            raise HierarchicalPreparationPrefixSalvageError(
                "completed-prefix replay fold order changed"
            )
        spent = authenticate_review_spent_cache_registrations(
            tuple(payload["review_spent_evidence_cache"]["registrations"])
        )
        if (
            tuple(source.outer_fold for source in spent) != folds
            or tuple(payload["review_spent_evidence_cache"]["registrations"])
            != self.review_spent_registrations
        ):
            raise HierarchicalPreparationPrefixSalvageError(
                "completed-prefix spent registrations changed"
            )
        index_registration = str(payload["context_fit_cache_index"]["registration"])
        if index_registration != self.context_fit_index_registration:
            raise HierarchicalPreparationPrefixSalvageError(
                "completed-prefix context-fit registration changed"
            )
        gate_sources = authenticate_context_fit_cache_index_registrations((index_registration,))
        gate_folds = tuple(int(source.binding.get("outer_fold", 0)) for source in gate_sources)
        if gate_folds != folds or any(source.kind != "review_gate" for source in gate_sources):
            raise HierarchicalPreparationPrefixSalvageError(
                "completed-prefix context index changed fold coverage"
            )
        for source in gate_sources:
            if (
                source.cache_manifest_path.name != "manifest.json"
                or source.cache_manifest_path.parent.name != source.cache_key
                or any(
                    Path(file.filename).suffix.lower() in _FORBIDDEN_SOURCE_SUFFIXES
                    for file in source.files
                )
            ):
                raise HierarchicalPreparationPrefixSalvageError(
                    "completed-prefix context index contains an executable checkpoint"
                )
        for record_name in ("source_input_manifest",):
            record = payload[record_name]
            path = Path(record["path"]).resolve(strict=True)
            if _sha256_bytes(path.read_bytes()) != record["file_sha256"]:
                raise HierarchicalPreparationPrefixSalvageError(
                    f"{record_name} changed after prefix export"
                )
        for record in payload["completed_fold_manifests"]:
            path = Path(record["path"]).resolve(strict=True)
            if _sha256_bytes(path.read_bytes()) != record["file_sha256"]:
                raise HierarchicalPreparationPrefixSalvageError(
                    "a completed fold manifest changed after prefix export"
                )
        assurances = payload["assurances"]
        if not isinstance(assurances, Mapping) or any(
            assurances.get(key) is not expected
            for key, expected in {
                "source_paths_read_only": True,
                "remote_clients_constructed": False,
                "remote_calls_made": False,
                "oracle_columns_read": False,
                "all_active_stage1_architectures_authenticated": True,
                "executable_checkpoint_indexed": False,
                "joblib_indexed": False,
                "backend_work_indexed": False,
                "checkpoint_only_fold_accepted": False,
            }.items()
        ):
            raise HierarchicalPreparationPrefixSalvageError(
                "completed-prefix replay assurances changed"
            )


def export_completed_hierarchical_preparation_prefix(
    *,
    preparation_dir: Path | str,
    scratch_output_dir: Path | str,
    spent_evidence_provider: object,
    spent_requests_by_outer_fold: Mapping[int, Mapping[str, Any]],
    destination: Path | str,
) -> CompletedHierarchicalPreparationPrefixSalvage:
    """Export the contiguous completed fold prefix through read-only overlays.

    The source preparation and scratch trees are never written.  ``destination``
    must be absent and disjoint from both source trees.
    """

    preparation = _validated_root(preparation_dir, label="preparation_dir")
    scratch = _validated_root(scratch_output_dir, label="scratch_output_dir")
    target = Path(destination).expanduser().absolute()
    resolved_target = target.resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise HierarchicalPreparationPrefixSalvageError("destination must be absent")
    if (
        resolved_target == preparation
        or resolved_target.is_relative_to(preparation)
        or preparation.is_relative_to(resolved_target)
        or resolved_target == scratch
        or resolved_target.is_relative_to(scratch)
        or scratch.is_relative_to(resolved_target)
    ):
        raise HierarchicalPreparationPrefixSalvageError(
            "destination must be disjoint from both read-only source trees"
        )

    input_path = _validated_regular_under(
        preparation / "immutable_hierarchical_input_manifest.json",
        root=preparation,
        label="immutable hierarchical input manifest",
    )
    input_wrapper, input_snapshot = _read_hash_wrapper(
        input_path,
        label="immutable hierarchical input manifest",
        expected_schema=HIERARCHICAL_PREPARATION_INPUT_SCHEMA_VERSION,
    )
    input_body = input_wrapper["body"]
    if input_body.get("preparation_schema_version") != (
        HIERARCHICAL_PREPARATION_INPUT_SCHEMA_VERSION
    ):
        raise HierarchicalPreparationPrefixSalvageError(
            "input manifest preparation schema binding changed"
        )
    if input_body.get("semantic_retrieval_compatibility") != (
        current_spent_projection_compatibility_identity()
    ):
        raise HierarchicalPreparationPrefixSalvageError(
            "input manifest semantic compatibility identity is not current"
        )
    outer_rows = input_body.get("outer_folds")
    if not isinstance(outer_rows, list) or not outer_rows:
        raise HierarchicalPreparationPrefixSalvageError("input manifest has no outer-fold registry")
    expected_folds = tuple(
        int(row.get("outer_fold", 0)) if isinstance(row, Mapping) else 0 for row in outer_rows
    )
    if expected_folds != tuple(range(1, len(expected_folds) + 1)):
        raise HierarchicalPreparationPrefixSalvageError(
            "input manifest outer folds are not complete one-based contiguous folds"
        )
    spent_identity, spent_identity_sha = _identity_record(
        input_body.get("spent_evidence_provider"), label="input spent provider"
    )
    gate_identity, gate_identity_sha = _identity_record(
        input_body.get("shared_first_gate_provider"), label="input gate provider"
    )
    _final_identity, _final_identity_sha = _identity_record(
        input_body.get("final_upstream_producer"), label="input final producer"
    )

    companion_record = input_body.get("context_fit_overlay_companion")
    if not isinstance(companion_record, Mapping):
        raise HierarchicalPreparationPrefixSalvageError(
            "input manifest lacks the context-fit overlay companion"
        )
    companion_path = _validated_regular_under(
        str(companion_record.get("path") or ""),
        root=preparation,
        label="context-fit overlay companion",
    )
    companion_sha = _required_sha256(
        companion_record.get("sha256"), label="context-fit overlay companion SHA-256"
    )
    companion_wrapper, companion_snapshot = _read_hash_wrapper(
        companion_path, label="context-fit overlay companion"
    )
    if _sha256_bytes(companion_snapshot) != companion_sha:
        raise HierarchicalPreparationPrefixSalvageError(
            "context-fit overlay companion file hash changed"
        )
    if companion_wrapper["schema_version"] != input_body.get("runner_schema_version"):
        raise HierarchicalPreparationPrefixSalvageError(
            "context-fit overlay companion runner schema changed"
        )

    completed_folds: list[int] = []
    fold_wrappers: dict[int, tuple[Path, Mapping[str, Any], bytes]] = {}
    gap_seen = False
    for outer_fold in expected_folds:
        candidate = (
            preparation / f"outer_fold_{outer_fold:03d}" / ("immutable_fold_preparation.json")
        )
        if candidate.is_file() and not candidate.is_symlink():
            if gap_seen:
                raise HierarchicalPreparationPrefixSalvageError(
                    "completed fold manifests are not a contiguous prefix"
                )
            fold_dir = candidate.parent.resolve(strict=True)
            fold_path = _validated_regular_under(
                candidate,
                root=fold_dir,
                label=f"fold {outer_fold} immutable preparation manifest",
            )
            wrapper, snapshot = _read_hash_wrapper(
                fold_path,
                label=f"fold {outer_fold} immutable preparation manifest",
                expected_schema=HIERARCHICAL_PREPARATION_FOLD_SCHEMA_VERSION,
            )
            if wrapper["body"].get("outer_fold") != outer_fold:
                raise HierarchicalPreparationPrefixSalvageError(
                    f"fold {outer_fold} completion marker cites another fold"
                )
            completed_folds.append(outer_fold)
            fold_wrappers[outer_fold] = (fold_path, wrapper, snapshot)
        else:
            gap_seen = True
    if not completed_folds:
        raise HierarchicalPreparationPrefixSalvageError(
            "no immutable completed preparation fold is available to salvage"
        )
    if not isinstance(spent_requests_by_outer_fold, Mapping) or set(
        spent_requests_by_outer_fold
    ) != set(completed_folds):
        raise HierarchicalPreparationPrefixSalvageError(
            "exact spent requests must cover the completed fold prefix"
        )

    spent_root = _validated_root(
        scratch / "post_extraction_review_spent_evidence_cache",
        label="top-level spent-evidence cache root",
    )
    spent_sources: list[AuthenticatedReviewSpentCacheSource] = []
    for source_path in sorted(spent_root.iterdir(), key=lambda path: path.name):
        if source_path.is_symlink():
            raise HierarchicalPreparationPrefixSalvageError(
                "top-level spent-evidence cache contains a symlink"
            )
        if not source_path.is_file() or source_path.suffix != ".json":
            continue
        source_sha = _sha256_bytes(source_path.read_bytes())
        authenticated = authenticate_review_spent_cache_registrations(
            (f"{source_path.resolve()}::{source_sha}",)
        )
        if len(authenticated) != 1:
            raise HierarchicalPreparationPrefixSalvageError(
                "a top-level spent JSON did not authenticate exactly once"
            )
        spent_sources.append(authenticated[0])

    gate_root = _validated_root(
        scratch / "post_extraction_review_gate_cache",
        label="top-level gate cache root",
    )
    selected_spent: list[AuthenticatedReviewSpentCacheSource] = []
    gate_entries: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    expected_source_digests: dict[Path, str] = {
        input_path: _sha256_bytes(input_snapshot),
        companion_path: companion_sha,
    }
    for outer_fold in completed_folds:
        fold_path, fold_wrapper, fold_snapshot = fold_wrappers[outer_fold]
        fold_body = fold_wrapper["body"]
        fold_dir = fold_path.parent
        catalog, direct, fold_artifact_digests = _validate_referenced_fold_artifacts(
            fold_body=fold_body,
            fold_dir=fold_dir,
            outer_fold=outer_fold,
        )
        expected_source_digests.update(fold_artifact_digests)
        spent_ids, sealed_ids, first_gate_ids, inner_fold_ids = _schedule_rows(
            fold_body.get("schedule_audit"), outer_fold=outer_fold
        )
        initial_audit = fold_body.get("initial_spent_evidence_audit")
        first_gate_audit = fold_body.get("first_gate_preparation_audit")
        if not isinstance(initial_audit, Mapping) or not isinstance(first_gate_audit, Mapping):
            raise HierarchicalPreparationPrefixSalvageError(
                f"fold {outer_fold} lacks immutable spent/first-gate audits"
            )
        semantic_retrieval_audit = initial_audit.get("semantic_retrieval_compatibility")
        if not isinstance(semantic_retrieval_audit, Mapping):
            raise HierarchicalPreparationPrefixSalvageError(
                f"fold {outer_fold} lacks its detached semantic-retrieval " "compatibility audit"
            )
        initial_binding = first_gate_audit.get("initial_spent_binding")
        gate_binding = first_gate_audit.get("first_untouched_gate_binding")
        if not isinstance(initial_binding, Mapping) or not isinstance(gate_binding, Mapping):
            raise HierarchicalPreparationPrefixSalvageError(
                f"fold {outer_fold} first-gate row bindings are malformed"
            )
        if (
            initial_audit.get("review_round") != 0
            or initial_audit.get("consumer_review_round") != 0
            or initial_audit.get("spent_evidence_context_epoch") != 0
            or initial_audit.get("provider_review_round_argument") != 0
            or initial_audit.get("provider_identity_sha256") != spent_identity_sha
            or initial_audit.get("spent_row_count") != len(spent_ids)
            or initial_audit.get("sealed_row_count") != len(sealed_ids)
            or initial_audit.get("spent_row_fingerprint") != row_set_fingerprint(spent_ids)
            or initial_audit.get("sealed_row_fingerprint") != row_set_fingerprint(sealed_ids)
            or initial_binding.get("row_ids_sha256") != _sha256_json(list(spent_ids))
            or initial_binding.get("row_count") != len(spent_ids)
            or gate_binding.get("row_ids_sha256") != _sha256_json(list(first_gate_ids))
            or gate_binding.get("row_count") != len(first_gate_ids)
        ):
            raise HierarchicalPreparationPrefixSalvageError(
                f"fold {outer_fold} spent/gate audit differs from its schedule"
            )
        if (
            _sha256_json(first_gate_audit) != fold_body.get("first_gate_preparation_audit_sha256")
            or fold_body.get("first_gate_cache_materialized_before_discovery") is not True
            or fold_body.get("first_gate_labels_supplied_to_provider") is not False
            or fold_body.get("first_gate_views_exposed_to_discovery") is not False
            or fold_body.get("hierarchy_runner_calls_during_preparation") != 0
        ):
            raise HierarchicalPreparationPrefixSalvageError(
                f"fold {outer_fold} immutable completion assurances changed"
            )
        spent_source = _spent_source_for_fold(
            sources=spent_sources,
            outer_fold=outer_fold,
            spent_ids=spent_ids,
            sealed_ids=sealed_ids,
            initial_binding=initial_binding,
            provider_identity=spent_identity,
            provider_identity_sha256=spent_identity_sha,
            spent_evidence_provider=spent_evidence_provider,
            spent_request=spent_requests_by_outer_fold[outer_fold],
            semantic_retrieval_compatibility_audit=semantic_retrieval_audit,
            catalog=catalog,
        )
        selected_spent.append(spent_source)

        gate_cache_key = _required_sha256(
            direct.get("source_cache_key"), label=f"fold {outer_fold} gate cache key"
        )
        gate_manifest = _validated_regular_under(
            gate_root / gate_cache_key / "manifest.json",
            root=gate_root,
            label=f"fold {outer_fold} complete top-level gate manifest",
        )
        if gate_manifest.parent.parent != gate_root or gate_manifest.parent.name != gate_cache_key:
            raise HierarchicalPreparationPrefixSalvageError(
                f"fold {outer_fold} gate source is checkpoint-only or non-canonical"
            )
        gate_entry = _gate_index_entry(
            gate_manifest=gate_manifest,
            companion_path=companion_path,
            companion_sha256=companion_sha,
        )
        gate_entries.append(gate_entry)
        expected_source_digests[spent_source.source_path] = spent_source.snapshot_sha256
        expected_source_digests[gate_manifest] = gate_entry["cache_manifest_sha256"]
        for filename, digest in gate_entry["cache_files"].items():
            matrix = _validated_regular_under(
                gate_manifest.parent / filename,
                root=gate_root,
                label=f"fold {outer_fold} complete top-level gate matrix {filename}",
            )
            expected_source_digests[matrix] = digest
        expected_source_digests[fold_path] = _sha256_bytes(fold_snapshot)
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "path": str(fold_path),
                "file_sha256": _sha256_bytes(fold_snapshot),
                "content_sha256": fold_wrapper["content_sha256"],
                "spent_cache_key": spent_source.cache_key,
                "gate_cache_key": gate_cache_key,
            }
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        index_content = {
            "schema_version": CONTEXT_FIT_CACHE_INDEX_SCHEMA_VERSION,
            "entries": gate_entries,
        }
        index_payload = {**index_content, "content_sha256": _sha256_json(index_content)}
        temporary_index = temporary / "completed_prefix_context_fit_cache_index.json"
        index_sha = _write_plain_json(temporary_index, index_payload)
        authenticated_gate_sources = authenticate_context_fit_cache_index_registrations(
            (f"{temporary_index}::{index_sha}",)
        )
        if len(authenticated_gate_sources) != len(completed_folds) or tuple(
            int(source.binding.get("outer_fold", 0)) for source in authenticated_gate_sources
        ) != tuple(completed_folds):
            raise HierarchicalPreparationPrefixSalvageError(
                "combined completed-prefix gate index has wrong fold coverage"
            )
        for outer_fold, source, fold_record in zip(
            completed_folds, authenticated_gate_sources, fold_records
        ):
            fold_body = fold_wrappers[outer_fold][1]["body"]
            first_gate_audit = fold_body["first_gate_preparation_audit"]
            initial_binding = first_gate_audit["initial_spent_binding"]
            gate_binding = first_gate_audit["first_untouched_gate_binding"]
            spent_ids, _sealed, first_gate_ids, inner_fold_ids = _schedule_rows(
                fold_body["schedule_audit"], outer_fold=outer_fold
            )
            direct_path = Path(fold_body["direct_manifest_path"])
            direct = _parse_json(
                direct_path.read_bytes(), label=f"fold {outer_fold} direct numerical manifest"
            )
            _validate_gate_source(
                source=source,
                outer_fold=outer_fold,
                spent_ids=spent_ids,
                first_gate_ids=first_gate_ids,
                inner_fold_ids=inner_fold_ids,
                initial_binding=initial_binding,
                gate_binding=gate_binding,
                gate_provider_identity=gate_identity,
                gate_provider_identity_sha256=gate_identity_sha,
                direct=direct,
                fold_body=fold_body,
            )
            if source.cache_key != fold_record["gate_cache_key"]:
                raise HierarchicalPreparationPrefixSalvageError(
                    f"fold {outer_fold} gate index order changed"
                )

        final_index = target / temporary_index.name
        spent_registrations = tuple(
            f"{source.source_path}::{source.registered_sha256}" for source in selected_spent
        )
        final_index_registration = f"{final_index}::{index_sha}"
        replay_content = {
            "schema_version": HIERARCHICAL_PREPARATION_PREFIX_SALVAGE_SCHEMA_VERSION,
            "source_input_manifest": {
                "path": str(input_path),
                "file_sha256": _sha256_bytes(input_snapshot),
                "content_sha256": input_wrapper["content_sha256"],
            },
            "completed_outer_folds": completed_folds,
            "completed_fold_manifests": fold_records,
            "review_spent_evidence_cache": {
                "registrations": list(spent_registrations),
                "source_count": len(spent_registrations),
            },
            "context_fit_cache_index": {
                "registration": final_index_registration,
                "source_count": len(gate_entries),
            },
            "assurances": {
                "source_paths_read_only": True,
                "remote_clients_constructed": False,
                "remote_calls_made": False,
                "oracle_columns_read": False,
                "all_active_stage1_architectures_authenticated": True,
                "active_stage1_architectures": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
                "executable_checkpoint_indexed": False,
                "joblib_indexed": False,
                "backend_work_indexed": False,
                "checkpoint_only_fold_accepted": False,
                "top_level_complete_spent_json_required": True,
                "top_level_complete_gate_bundle_required": True,
            },
        }
        replay_payload = {
            **replay_content,
            "content_sha256": _sha256_json(replay_content),
        }
        temporary_replay = temporary / "completed_prefix_salvage_manifest.json"
        replay_sha = _write_plain_json(temporary_replay, replay_payload)

        for source_path, expected_digest in expected_source_digests.items():
            if _sha256_bytes(source_path.read_bytes()) != expected_digest:
                raise HierarchicalPreparationPrefixSalvageError(
                    f"read-only source changed during salvage export: {source_path}"
                )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    result = CompletedHierarchicalPreparationPrefixSalvage(
        destination=target.resolve(strict=True),
        completed_outer_folds=tuple(completed_folds),
        review_spent_registrations=spent_registrations,
        context_fit_index_registration=final_index_registration,
        replay_manifest_registration=(
            f"{target / 'completed_prefix_salvage_manifest.json'}::{replay_sha}"
        ),
    )
    result.validate_authentication()
    return result


__all__ = [
    "CompletedHierarchicalPreparationPrefixSalvage",
    "HIERARCHICAL_PREPARATION_PREFIX_SALVAGE_SCHEMA_VERSION",
    "HierarchicalPreparationPrefixSalvageError",
    "export_completed_hierarchical_preparation_prefix",
]
