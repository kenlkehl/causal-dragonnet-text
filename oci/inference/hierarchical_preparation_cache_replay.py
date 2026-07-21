"""Authenticate legacy hierarchical-preparation replay registrations.

Version-1 preparation used to materialize two kinds of local results before a
human could approve its all-fold packet:

* one context-epoch-zero spent-discovery JSON cache entry per outer fold; and
* one complete, label-free first-gate context-fit bundle per outer fold.

Current preparation deliberately does *not* create the second artifact.  It
freezes a :class:`FirstGateMaterializationIntent` and materializes numerical
gate inputs only after exact approval and proposal freeze.  Consequently the
reader remains available for authenticating already-published legacy manifests,
while the exporter fails closed for every current prepared batch.  Current
cross-process execution reuses only the authoritative spent-cache registrations.

The legacy reader does not copy or traverse backend work trees,
load joblib checkpoints, decode NumPy matrices, instantiate a model service,
or call a discovery runner.  The context-fit companion attestation and index
are authored by ``prepare_hierarchical_discovery_batch`` and remain the single
authoritative copies; this exporter independently authenticates and binds
them into one closed replay manifest in a distinct fresh subdirectory.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence, TYPE_CHECKING

from .all_evidence_discovery_interfaces import canonical_json, content_sha256
from .context_fit_upstream_cache_overlay import (
    AuthenticatedContextFitCacheSource,
    authenticate_context_fit_cache_index_registrations,
)
from .context_fit_upstream_gate_provider import (
    BoundContextFitUpstreamGateProvider,
    ContextFitUpstreamGateProvider,
)
from .final_context_fit_upstream_bank import FinalContextFitUpstreamProducer
from .review_spent_evidence_cache_overlay import (
    authenticate_review_spent_cache_registrations,
)
from .review_spent_evidence_provider import ContextFitReviewSpentEvidenceProvider

if TYPE_CHECKING:  # pragma: no cover - imported lazily at runtime to avoid a cycle
    from .all_evidence_fusion_runner import PreparedHierarchicalDiscoveryBatch


HIERARCHICAL_PREPARATION_CACHE_REPLAY_SCHEMA_VERSION = "hierarchical_preparation_cache_replay_v1"
HIERARCHICAL_PREPARATION_CACHE_REPLAY_EXPORTER_VERSION = (
    "authenticated_non_executable_hierarchical_preparation_replay_exporter_v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPENT_ARTIFACT_ID = re.compile(r"^review-spent-([0-9a-f]{64})$")
_WRAPPER_FIELDS = frozenset({"schema_version", "body", "content_sha256"})
_REPLAY_BODY_FIELDS = frozenset(
    {
        "exporter_version",
        "preparation_root",
        "prepared_batch_input_manifest",
        "approved_batch",
        "dataset_sha256",
        "ordered_outer_folds",
        "fold_manifests",
        "raw_provider_identities",
        "code_identities",
        "spent_cache_sources",
        "context_fit_companion_attestation",
        "context_fit_cache_index",
        "registrations",
        "assurances",
    }
)
_REGISTRATION_FIELDS = frozenset({"review_spent_evidence_cache", "context_fit_cache_index"})
_RAW_PROVIDER_FIELDS = frozenset({"review_spent_evidence", "review_gate", "final_upstream"})
_ASSURANCES = {
    "initial_spent_context_epoch_only": True,
    "exactly_one_spent_json_per_outer_fold": True,
    "exactly_one_complete_first_gate_bundle_per_outer_fold": True,
    "first_gate_bundle_index_reauthenticated_without_matrix_decode": True,
    "companion_attests_exact_raw_gate_and_final_provider_identities": True,
    "query_discovery_joblib_indexed": False,
    "query_discovery_joblib_loaded": False,
    "executable_checkpoint_indexed": False,
    "executable_checkpoint_loaded": False,
    "backend_work_tree_traversed": False,
    "model_service_instantiated": False,
    "hierarchy_runner_called": False,
    "source_paths_are_read_only_registrations": True,
    "fresh_output_root_not_identity_bearing": True,
}
_CODE_MODULES = (
    "oci.inference.hierarchical_preparation_cache_replay",
    "oci.inference.review_spent_evidence_cache_overlay",
    "oci.inference.context_fit_upstream_cache_overlay",
    "oci.inference.review_spent_evidence_provider",
    "oci.inference.context_fit_upstream_gate_provider",
    "oci.inference.final_context_fit_upstream_bank",
    "oci.inference.approved_hierarchical_discovery_batch",
    "oci.inference.all_evidence_fusion_runner",
)
_FORBIDDEN_INDEXED_SUFFIXES = frozenset(
    {".joblib", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".onnx"}
)


class HierarchicalPreparationCacheReplayAuthenticationError(RuntimeError):
    """A preparation replay source or binding failed closed authentication."""


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
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} is not closed finite UTF-8 JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} root must be one object"
        )
    return value


def _sha256_bytes(snapshot: bytes) -> str:
    return hashlib.sha256(snapshot).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"authenticated file is unreadable: {path}"
        ) from exc


def _required_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} must be one lowercase SHA-256 digest"
        )
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} must be a positive integer"
        )
    return value


def _identity_record(identity: Mapping[str, Any]) -> dict[str, Any]:
    detached = json.loads(canonical_json(identity))
    return {"identity": detached, "identity_sha256": content_sha256(detached)}


def _validate_identity_record(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"identity", "identity_sha256"}:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} identity record is malformed"
        )
    identity = value["identity"]
    if not isinstance(identity, Mapping):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} identity must be one object"
        )
    digest = _required_sha256(value["identity_sha256"], label=f"{label}.identity_sha256")
    if content_sha256(identity) != digest:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} identity hash mismatch"
        )
    return identity


def _read_hash_wrapper(path: Path, *, label: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        snapshot = path.read_bytes()
    except OSError as exc:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} is unreadable: {path}"
        ) from exc
    raw = _parse_json(snapshot, label=label)
    if set(raw) != _WRAPPER_FIELDS or not isinstance(raw["body"], Mapping):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} has an unsupported closed wrapper schema"
        )
    digest = _required_sha256(raw["content_sha256"], label=f"{label}.content_sha256")
    if content_sha256(raw["body"]) != digest:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} content hash mismatch"
        )
    if not isinstance(raw["schema_version"], str) or not raw["schema_version"]:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} schema_version must be non-empty"
        )
    return raw, snapshot


def _absolute_without_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _require_no_symlink_path(path: Path, *, root: Path, label: str) -> Path:
    raw_root = _absolute_without_resolution(root)
    raw_path = _absolute_without_resolution(path)
    try:
        relative = raw_path.relative_to(raw_root)
    except ValueError as exc:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} is outside its authenticated root"
        ) from exc
    current = raw_root
    if current.is_symlink():
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} root may not be a symlink"
        )
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                f"{label} may not traverse a symlink: {current}"
            )
    resolved_root = raw_root.resolve(strict=True)
    try:
        resolved = raw_path.resolve(strict=True)
    except OSError as exc:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} does not exist: {raw_path}"
        ) from exc
    if resolved != raw_path or resolved_root != raw_root:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} must use a canonical non-symlink path"
        )
    return resolved


def _validate_input_manifest_record(value: Any, *, root: Path, label: str) -> Path:
    expected_fields = {
        "path",
        "sha256",
        "byte_count",
        "file_sha256",
        "content_sha256",
        "schema_version",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise HierarchicalPreparationCacheReplayAuthenticationError(f"{label} record is malformed")
    path = _require_no_symlink_path(Path(str(value["path"])), root=root, label=label)
    if not path.is_file():
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} is not a regular file"
        )
    digest = _required_sha256(value["sha256"], label=f"{label}.sha256")
    snapshot = path.read_bytes()
    if (
        _sha256_bytes(snapshot) != digest
        or value["file_sha256"] != digest
        or len(snapshot) != value["byte_count"]
    ):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} bytes changed after replay export"
        )
    return path


def _module_file(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    raw = getattr(module, "__file__", None)
    if not raw:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"code module has no source file: {module_name}"
        )
    path = Path(raw).resolve(strict=True)
    if path.suffix == ".pyc" and path.with_suffix(".py").is_file():
        path = path.with_suffix(".py").resolve(strict=True)
    return path


def _current_code_identities() -> dict[str, Any]:
    return {
        name: {
            "path": str(_module_file(name)),
            "sha256": _sha256_file(_module_file(name)),
        }
        for name in _CODE_MODULES
    }


def _validate_code_identities(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_CODE_MODULES):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "replay code identities have a wrong closed schema"
        )
    current = _current_code_identities()
    if canonical_json(value) != canonical_json(current):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "one or more replay authentication implementations changed"
        )


def _registration(raw_path: Path, digest: str) -> str:
    if "::" in str(raw_path):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "registered cache path may not contain the registration delimiter"
        )
    return f"{raw_path}::{_required_sha256(digest, label='registration SHA-256')}"


def _split_registration(value: str, *, label: str) -> tuple[Path, str]:
    raw_path, separator, raw_digest = str(value).rpartition("::")
    if not separator or not raw_path or not raw_digest:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            f"{label} must use PATH::SHA256"
        )
    return Path(raw_path), _required_sha256(raw_digest, label=f"{label} SHA-256")


def _prepared_class() -> type[Any]:
    # Importing lazily prevents all_evidence_fusion_runner -> this module from
    # becoming a circular import when the CLI later wires the exporter.
    from .all_evidence_fusion_runner import PreparedHierarchicalDiscoveryBatch

    return PreparedHierarchicalDiscoveryBatch


def _prepared_input_manifest(prepared: Any) -> tuple[Mapping[str, Any], bytes, Path]:
    path = Path(prepared.input_manifest_path)
    root = _absolute_without_resolution(path).parent
    path = _require_no_symlink_path(path, root=root, label="prepared input manifest")
    raw, snapshot = _read_hash_wrapper(path, label="prepared input manifest")
    expected = _required_sha256(
        prepared.input_manifest_sha256,
        label="prepared batch input_manifest_sha256",
    )
    if raw["content_sha256"] != expected:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "prepared input manifest bytes differ from the batch binding"
        )
    body = raw["body"]
    dataset = body.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("sha256") != prepared.dataset_sha256:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "prepared dataset SHA-256 differs from its input manifest"
        )
    return raw, snapshot, path


def _validate_raw_providers(
    *,
    prepared: Any,
    input_body: Mapping[str, Any],
    review_spent_evidence_provider: ContextFitReviewSpentEvidenceProvider,
    review_gate_provider: ContextFitUpstreamGateProvider,
    final_upstream_producer: FinalContextFitUpstreamProducer,
) -> dict[str, Any]:
    if type(review_spent_evidence_provider) is not ContextFitReviewSpentEvidenceProvider:
        raise TypeError(
            "replay export requires the exact raw ContextFitReviewSpentEvidenceProvider"
        )
    if type(review_gate_provider) is not ContextFitUpstreamGateProvider:
        raise TypeError("replay export requires the exact raw ContextFitUpstreamGateProvider")
    if type(final_upstream_producer) is not FinalContextFitUpstreamProducer:
        raise TypeError("replay export requires the exact raw FinalContextFitUpstreamProducer")
    records = {
        "review_spent_evidence": _identity_record(review_spent_evidence_provider.identity()),
        "review_gate": _identity_record(review_gate_provider.identity()),
        "final_upstream": _identity_record(final_upstream_producer.identity()),
    }
    expected = {
        "review_spent_evidence": input_body.get("spent_evidence_provider"),
        "review_gate": input_body.get("shared_first_gate_provider"),
        "final_upstream": input_body.get("final_upstream_producer"),
    }
    if canonical_json(records) != canonical_json(expected):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "raw replay providers differ from the prepared input manifest"
        )
    raw_final = input_body.get("raw_final_upstream_producer")
    if raw_final is not None and canonical_json(raw_final) != canonical_json(
        records["final_upstream"]
    ):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "prepared raw final producer differs from the supplied exact producer"
        )
    for label, record in records.items():
        _validate_identity_record(record, label=label)
    # PreparedHierarchicalDiscoveryBatch.__post_init__ checks these basic batch
    # bindings without consulting any model/cache runner.
    prepared.__post_init__()
    return records


def _prepared_fold_numbers(prepared: Any, input_body: Mapping[str, Any]) -> tuple[int, ...]:
    folds = tuple(
        _positive_int(row.outer_fold, label="prepared fold outer_fold") for row in prepared.folds
    )
    expected = tuple(range(1, len(folds) + 1))
    if not folds or folds != expected:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "prepared replay folds are missing, duplicated, extra, or out of order"
        )
    input_rows = input_body.get("outer_folds")
    if not isinstance(input_rows, list):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "prepared input manifest lacks its closed outer-fold list"
        )
    manifest_folds = tuple(
        (
            _positive_int(row.get("outer_fold"), label="input outer_fold")
            if isinstance(row, Mapping)
            else 0
        )
        for row in input_rows
    )
    if manifest_folds != folds:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "prepared object fold set differs from the input manifest"
        )
    return folds


def _spent_source_rows(
    *,
    prepared: Any,
    folds: Sequence[int],
    provider: ContextFitReviewSpentEvidenceProvider,
    provider_record: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    cache_root = _require_no_symlink_path(
        Path(provider.cache_dir),
        root=_absolute_without_resolution(Path(provider.cache_dir)),
        label="spent cache root",
    )
    expected_provider_sha = provider_record["identity_sha256"]
    rows: list[dict[str, Any]] = []
    registrations: list[str] = []
    cache_keys: set[str] = set()
    for outer_fold, prepared_fold in zip(folds, prepared.folds):
        audit = prepared_fold.initial_spent_evidence_audit
        if not isinstance(audit, Mapping):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "prepared fold spent evidence audit is malformed"
            )
        for label in (
            "review_round",
            "consumer_review_round",
            "spent_evidence_context_epoch",
            "provider_review_round_argument",
            "consumed_gate_count_before_context_fit",
        ):
            if audit.get(label) != 0:
                raise HierarchicalPreparationCacheReplayAuthenticationError(
                    f"prepared fold {outer_fold} is not initial context epoch zero"
                )
        if audit.get("provider_identity_sha256") != expected_provider_sha:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "spent evidence audit cites a different raw provider identity"
            )
        evidence_inputs = tuple(prepared_fold.evidence_inputs)
        if not evidence_inputs:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "prepared fold has no initial spent evidence inputs"
            )
        artifact_ids = {item.provenance.artifact_id for item in evidence_inputs}
        if len(artifact_ids) != 1:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "prepared fold evidence does not cite exactly one spent cache key"
            )
        match = _SPENT_ARTIFACT_ID.fullmatch(next(iter(artifact_ids)))
        if match is None:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "prepared fold evidence artifact ID is not a canonical spent cache key"
            )
        cache_key = match.group(1)
        if cache_key in cache_keys:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "prepared folds contain a duplicate spent cache key"
            )
        cache_keys.add(cache_key)
        source_path = _require_no_symlink_path(
            cache_root / f"{cache_key}.json",
            root=cache_root,
            label=f"fold {outer_fold} spent cache source",
        )
        if source_path.parent != cache_root:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "spent cache source is not directly nested under its raw provider cache"
            )
        source_sha = _sha256_file(source_path)
        registration = _registration(source_path, source_sha)
        authenticated = authenticate_review_spent_cache_registrations((registration,))
        if len(authenticated) != 1:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "spent cache registration did not authenticate exactly one source"
            )
        source = authenticated[0]
        if (
            source.outer_fold != outer_fold
            or source.review_round != 0
            or source.cache_key != cache_key
            or source.provider_identity_sha256 != expected_provider_sha
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "spent cache binding differs from its prepared fold provenance"
            )
        parsed = _parse_json(source.snapshot, label=f"fold {outer_fold} spent cache")
        expected_results = [
            {"source_kind": item.source_kind, "payload": item.payload} for item in evidence_inputs
        ]
        if canonical_json(parsed.get("results")) != canonical_json(expected_results):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "spent cache results differ from the prepared evidence inputs"
            )
        provenance_rows: list[dict[str, Any]] = []
        for item in evidence_inputs:
            provenance = item.provenance
            if (
                provenance.outer_fold != outer_fold
                or provenance.scope != "inner_train"
                or provenance.inner_fold != 1
                or provenance.artifact_id != f"review-spent-{cache_key}"
            ):
                raise HierarchicalPreparationCacheReplayAuthenticationError(
                    "spent evidence provenance is not the initial inner-train epoch"
                )
            provenance_rows.append(
                {
                    "source_kind": item.source_kind,
                    "split_fingerprint": provenance.split_fingerprint,
                    "artifact_id": provenance.artifact_id,
                }
            )
        rows.append(
            {
                "outer_fold": outer_fold,
                "registration": registration,
                "authenticated_source_identity": source.identity(),
                "prepared_evidence_inputs_sha256": content_sha256(expected_results),
                "prepared_evidence_provenance": provenance_rows,
            }
        )
        registrations.append(registration)
    # Reauthenticate the whole registration vector to make duplicate path/key
    # checks apply across folds, not just within each fold.
    combined = authenticate_review_spent_cache_registrations(tuple(registrations))
    if tuple(source.outer_fold for source in combined) != tuple(folds):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "spent cache registrations do not cover the complete ordered fold set"
        )
    return rows, tuple(registrations)


def _context_fit_records(
    *,
    prepared: Any,
    folds: Sequence[int],
    preparation_root: Path,
    gate_provider: ContextFitUpstreamGateProvider,
    raw_provider_records: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    companion_path = _require_no_symlink_path(
        Path(prepared.context_fit_overlay_companion_path),
        root=preparation_root,
        label="context-fit companion attestation",
    )
    companion_sha = _required_sha256(
        prepared.context_fit_overlay_companion_sha256,
        label="prepared companion SHA-256",
    )
    if _sha256_file(companion_path) != companion_sha:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "prepared context-fit companion bytes changed"
        )
    companion, companion_snapshot = _read_hash_wrapper(
        companion_path, label="context-fit companion attestation"
    )
    companion_body = companion["body"]
    if companion["schema_version"] != companion_body.get("runner_schema_version"):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "context-fit companion outer schema differs from body.runner_schema_version"
        )
    providers = companion_body.get("post_extraction_review_providers")
    final_inputs = companion_body.get("final_upstream_model_inputs")
    expected_gate = raw_provider_records["review_gate"]
    expected_final = raw_provider_records["final_upstream"]
    if (
        not isinstance(providers, Mapping)
        or canonical_json(providers.get("calibrated_gate_sources")) != canonical_json(expected_gate)
        or canonical_json(providers.get("role_aware_gate_feature_banks"))
        != canonical_json(expected_gate)
        or not isinstance(final_inputs, Mapping)
        or canonical_json(final_inputs.get("producer")) != canonical_json(expected_final)
    ):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "context-fit companion does not attest the exact raw gate/final providers"
        )

    index_path = _require_no_symlink_path(
        Path(prepared.first_gate_context_fit_cache_index_path),
        root=preparation_root,
        label="first-gate context-fit cache index",
    )
    index_sha = _required_sha256(
        prepared.first_gate_context_fit_cache_index_sha256,
        label="prepared first-gate index SHA-256",
    )
    if _sha256_file(index_path) != index_sha:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "prepared first-gate context-fit cache index bytes changed"
        )
    registration = _registration(index_path, index_sha)
    sources = authenticate_context_fit_cache_index_registrations((registration,))
    if len(sources) != len(folds) or any(source.kind != "review_gate" for source in sources):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "first-gate index must contain exactly one review-gate bundle per outer fold"
        )
    by_fold: dict[int, AuthenticatedContextFitCacheSource] = {}
    gate_cache_root = _require_no_symlink_path(
        Path(gate_provider.cache_dir),
        root=_absolute_without_resolution(Path(gate_provider.cache_dir)),
        label="raw first-gate cache root",
    )
    expected_gate_identity = raw_provider_records["review_gate"]["identity"]
    expected_final_identity = raw_provider_records["final_upstream"]["identity"]
    for source in sources:
        outer_fold = _positive_int(
            source.binding.get("outer_fold"), label="indexed gate outer_fold"
        )
        if outer_fold in by_fold:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "first-gate index contains a duplicate outer fold"
            )
        if outer_fold not in folds:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "first-gate index contains an extra outer fold"
            )
        if canonical_json(source.binding.get("provider_identity")) != canonical_json(
            expected_gate_identity
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "indexed gate bundle differs from the exact raw gate provider"
            )
        if canonical_json(source.run_attestation.gate_provider_identity) != canonical_json(
            expected_gate_identity
        ) or canonical_json(source.run_attestation.final_producer_identity) != canonical_json(
            expected_final_identity
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "indexed gate run attestation differs from exact raw providers"
            )
        manifest = _require_no_symlink_path(
            source.cache_manifest_path,
            root=gate_cache_root,
            label=f"fold {outer_fold} first-gate cache manifest",
        )
        if (
            manifest.name != "manifest.json"
            or manifest.parent.name != source.cache_key
            or manifest.parent.parent != gate_cache_root
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "first-gate cache bundle is not canonically nested under its raw provider"
            )
        for file_snapshot in source.files:
            matrix = _require_no_symlink_path(
                manifest.parent / file_snapshot.filename,
                root=gate_cache_root,
                label=f"fold {outer_fold} indexed first-gate matrix",
            )
            if matrix.parent != manifest.parent:
                raise HierarchicalPreparationCacheReplayAuthenticationError(
                    "indexed first-gate matrix escaped its cache-key directory"
                )
            if matrix.suffix.lower() in _FORBIDDEN_INDEXED_SUFFIXES:
                raise HierarchicalPreparationCacheReplayAuthenticationError(
                    "first-gate index contains an executable checkpoint"
                )
        by_fold[outer_fold] = source
    if tuple(sorted(by_fold)) != tuple(folds):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "first-gate index is missing one or more prepared folds"
        )

    source_rows: list[dict[str, Any]] = []
    for outer_fold, prepared_fold in zip(folds, prepared.folds):
        bound = prepared_fold.first_gate_provider
        if type(bound) is not BoundContextFitUpstreamGateProvider:
            raise TypeError("prepared first gate must use the exact bound context-fit provider")
        bound_identity = bound.identity()
        source = by_fold[outer_fold]
        if (
            bound.outer_fold != outer_fold
            or bound.authenticated_cache_manifest_path != source.cache_manifest_path
            or bound_identity.get("cache_manifest_sha256") != source.cache_manifest_sha256
            or bound_identity.get("parent_identity_sha256")
            != raw_provider_records["review_gate"]["identity_sha256"]
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "prepared bound first gate differs from the indexed complete bundle"
            )
        source_rows.append(
            {
                "outer_fold": outer_fold,
                "bound_provider_identity": json.loads(canonical_json(bound_identity)),
                "authenticated_source_identity": source.identity(),
            }
        )
    companion_record = {
        "path": str(companion_path),
        "sha256": companion_sha,
        "byte_count": len(companion_snapshot),
        "runner_schema_version": companion["schema_version"],
        "raw_gate_provider_identity_sha256": raw_provider_records["review_gate"]["identity_sha256"],
        "raw_final_producer_identity_sha256": raw_provider_records["final_upstream"][
            "identity_sha256"
        ],
    }
    index_record = {
        "registration": registration,
        "path": str(index_path),
        "sha256": index_sha,
        "byte_count": index_path.stat().st_size,
        "sources": source_rows,
    }
    return companion_record, index_record, registration


def _fold_manifest_records(
    *, prepared: Any, folds: Sequence[int], preparation_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths: set[Path] = set()
    for outer_fold, prepared_fold in zip(folds, prepared.folds):
        path = _require_no_symlink_path(
            Path(prepared_fold.preparation_manifest_path),
            root=preparation_root,
            label=f"fold {outer_fold} preparation manifest",
        )
        if path in paths:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "prepared folds share one preparation manifest path"
            )
        paths.add(path)
        wrapper, snapshot = _read_hash_wrapper(
            path, label=f"fold {outer_fold} preparation manifest"
        )
        if wrapper["body"].get("outer_fold") != outer_fold:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "fold preparation manifest has a wrong outer fold"
            )
        rows.append(
            {
                "outer_fold": outer_fold,
                "path": str(path),
                "file_sha256": _sha256_bytes(snapshot),
                "content_sha256": wrapper["content_sha256"],
                "schema_version": wrapper["schema_version"],
            }
        )
    return rows


def _approved_batch_record(
    *, prepared: Any, preparation_root: Path, folds: Sequence[int]
) -> dict[str, Any]:
    approval_sha = _required_sha256(
        prepared.approval_sha256, label="prepared batch approval SHA-256"
    )
    precommit = prepared.coordinator.precommit
    precommit.__post_init__()
    if (
        precommit.approval_sha256 != approval_sha
        or content_sha256(precommit.packet) != approval_sha
    ):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "prepared coordinator precommit differs from its approval SHA-256"
        )
    packet_folds = tuple(precommit.packet.get("ordered_outer_folds", ()))
    if packet_folds != tuple(folds):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "batch precommit fold set differs from the prepared folds"
        )
    packet_path = _require_no_symlink_path(
        Path(prepared.batch_packet_path),
        root=preparation_root,
        label="approved hierarchical batch packet",
    )
    wrapper, snapshot = _read_hash_wrapper(packet_path, label="approved hierarchical batch packet")
    body = wrapper["body"]
    if body.get("approval_sha256") != approval_sha or canonical_json(
        body.get("packet")
    ) != canonical_json(precommit.packet):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "persisted batch packet differs from the prepared coordinator"
        )
    return {
        "approval_sha256": approval_sha,
        "packet_path": str(packet_path),
        "packet_file_sha256": _sha256_bytes(snapshot),
        "packet_content_sha256": wrapper["content_sha256"],
    }


def _serialized_replay(body: Mapping[str, Any]) -> bytes:
    payload = {
        "schema_version": HIERARCHICAL_PREPARATION_CACHE_REPLAY_SCHEMA_VERSION,
        "body": json.loads(canonical_json(body)),
        "content_sha256": content_sha256(body),
    }
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class AuthenticatedHierarchicalPreparationCacheReplay:
    """One closed replay manifest and its repeatable overlay registrations."""

    replay_manifest_path: Path
    replay_manifest_sha256: str
    preparation_root: Path
    input_manifest_sha256: str
    batch_approval_sha256: str
    dataset_sha256: str
    ordered_outer_folds: tuple[int, ...]
    spent_cache_registrations: tuple[str, ...]
    context_fit_cache_index_registration: str
    _manifest_snapshot: bytes = field(repr=False)

    @property
    def review_spent_registrations(self) -> tuple[str, ...]:
        """Compatibility name for repeatable spent-cache registrations."""

        return self.spent_cache_registrations

    @property
    def context_fit_index_registration(self) -> str:
        """Compatibility name for the one context-fit index registration."""

        return self.context_fit_cache_index_registration

    @property
    def context_fit_cache_index_registrations(self) -> tuple[str, ...]:
        """Return the one repeatable ``INDEX_PATH::SHA256`` registration."""

        return (self.context_fit_cache_index_registration,)

    @property
    def replay_manifest_registration(self) -> str:
        return _registration(self.replay_manifest_path, self.replay_manifest_sha256)

    def validate_authentication(self) -> None:
        path = _require_no_symlink_path(
            Path(self.replay_manifest_path),
            root=Path(self.preparation_root),
            label="hierarchical replay manifest",
        )
        if path.parent.parent != Path(self.preparation_root):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "replay manifest must live in a distinct direct child of preparation root"
            )
        inventory = tuple(path.parent.iterdir())
        if inventory != (path,) or path.is_symlink():
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "replay export directory must contain exactly its closed manifest"
            )
        snapshot = path.read_bytes()
        digest = _required_sha256(self.replay_manifest_sha256, label="replay manifest SHA-256")
        if _sha256_bytes(snapshot) != digest or snapshot != self._manifest_snapshot:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "replay manifest mutated after authentication"
            )
        raw = _parse_json(snapshot, label="hierarchical replay manifest")
        if (
            set(raw) != _WRAPPER_FIELDS
            or raw["schema_version"] != HIERARCHICAL_PREPARATION_CACHE_REPLAY_SCHEMA_VERSION
            or not isinstance(raw["body"], Mapping)
            or set(raw["body"]) != _REPLAY_BODY_FIELDS
            or raw["content_sha256"] != content_sha256(raw["body"])
            or snapshot != _serialized_replay(raw["body"])
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "replay manifest has a noncanonical or hash-invalid closed schema"
            )
        body = raw["body"]
        if body["exporter_version"] != HIERARCHICAL_PREPARATION_CACHE_REPLAY_EXPORTER_VERSION:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "replay manifest exporter version is unsupported"
            )
        preparation_root = Path(str(body["preparation_root"]))
        if preparation_root != Path(self.preparation_root):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "replay preparation root differs from its authenticated object"
            )
        input_record = body["prepared_batch_input_manifest"]
        input_path = _validate_input_manifest_record(
            input_record, root=preparation_root, label="prepared batch input manifest"
        )
        input_wrapper, input_snapshot = _read_hash_wrapper(
            input_path, label="prepared batch input manifest"
        )
        if (
            input_record.get("file_sha256") != _sha256_bytes(input_snapshot)
            or input_wrapper["content_sha256"] != self.input_manifest_sha256
            or input_record.get("content_sha256") != self.input_manifest_sha256
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "prepared input manifest binding changed"
            )
        if body["dataset_sha256"] != self.dataset_sha256:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "replay dataset SHA-256 changed"
            )
        folds = tuple(body["ordered_outer_folds"])
        if folds != self.ordered_outer_folds or folds != tuple(range(1, len(folds) + 1)):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "replay outer folds are missing, duplicated, extra, or out of order"
            )
        batch = body["approved_batch"]
        if batch.get("approval_sha256") != self.batch_approval_sha256:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "replay batch approval SHA-256 changed"
            )
        packet_path = _require_no_symlink_path(
            Path(str(batch.get("packet_path"))),
            root=preparation_root,
            label="approved batch packet",
        )
        packet_snapshot = packet_path.read_bytes()
        if _sha256_bytes(packet_snapshot) != batch.get("packet_file_sha256"):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "approved batch packet bytes changed"
            )
        packet_wrapper, _ = _read_hash_wrapper(packet_path, label="approved batch packet")
        if (
            packet_wrapper["content_sha256"] != batch.get("packet_content_sha256")
            or packet_wrapper["body"].get("approval_sha256") != self.batch_approval_sha256
            or content_sha256(packet_wrapper["body"].get("packet")) != self.batch_approval_sha256
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "approved batch packet binding changed"
            )
        fold_records = body["fold_manifests"]
        if (
            not isinstance(fold_records, list)
            or tuple(
                row.get("outer_fold") if isinstance(row, Mapping) else None for row in fold_records
            )
            != folds
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "replay fold manifest records are incomplete"
            )
        for row in fold_records:
            fold_path = _require_no_symlink_path(
                Path(str(row.get("path"))),
                root=preparation_root,
                label="fold preparation manifest",
            )
            fold_wrapper, fold_snapshot = _read_hash_wrapper(
                fold_path, label="fold preparation manifest"
            )
            if (
                _sha256_bytes(fold_snapshot) != row.get("file_sha256")
                or fold_wrapper["content_sha256"] != row.get("content_sha256")
                or fold_wrapper["schema_version"] != row.get("schema_version")
                or fold_wrapper["body"].get("outer_fold") != row.get("outer_fold")
            ):
                raise HierarchicalPreparationCacheReplayAuthenticationError(
                    "fold preparation manifest binding changed"
                )
        raw_providers = body["raw_provider_identities"]
        if not isinstance(raw_providers, Mapping) or set(raw_providers) != _RAW_PROVIDER_FIELDS:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "raw replay provider identity schema changed"
            )
        for label, record in raw_providers.items():
            _validate_identity_record(record, label=label)
        _validate_code_identities(body["code_identities"])
        registrations = body["registrations"]
        if not isinstance(registrations, Mapping) or set(registrations) != _REGISTRATION_FIELDS:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "replay registrations have a wrong closed schema"
            )
        spent_registrations = tuple(registrations["review_spent_evidence_cache"])
        context_registrations = tuple(registrations["context_fit_cache_index"])
        if (
            spent_registrations != self.spent_cache_registrations
            or context_registrations != self.context_fit_cache_index_registrations
            or len(spent_registrations) != len(folds)
            or len(context_registrations) != 1
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "replay registration vector changed"
            )
        spent_sources = authenticate_review_spent_cache_registrations(spent_registrations)
        context_sources = authenticate_context_fit_cache_index_registrations(context_registrations)
        spent_rows = body["spent_cache_sources"]
        if (
            not isinstance(spent_rows, list)
            or tuple(
                row.get("outer_fold") if isinstance(row, Mapping) else None for row in spent_rows
            )
            != folds
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "spent replay sources are incomplete"
            )
        if tuple(source.outer_fold for source in spent_sources) != folds or any(
            source.review_round != 0 for source in spent_sources
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "spent replay sources are not exactly context epoch zero"
            )
        for row, source, registration in zip(spent_rows, spent_sources, spent_registrations):
            if (
                row.get("registration") != registration
                or canonical_json(row.get("authenticated_source_identity"))
                != canonical_json(source.identity())
                or source.provider_identity_sha256
                != raw_providers["review_spent_evidence"]["identity_sha256"]
            ):
                raise HierarchicalPreparationCacheReplayAuthenticationError(
                    "spent replay source identity changed"
                )
        companion = body["context_fit_companion_attestation"]
        companion_path = _require_no_symlink_path(
            Path(str(companion.get("path"))),
            root=preparation_root,
            label="context-fit companion attestation",
        )
        companion_snapshot = companion_path.read_bytes()
        companion_wrapper, _ = _read_hash_wrapper(
            companion_path, label="context-fit companion attestation"
        )
        if (
            _sha256_bytes(companion_snapshot) != companion.get("sha256")
            or len(companion_snapshot) != companion.get("byte_count")
            or companion_wrapper["schema_version"]
            != companion_wrapper["body"].get("runner_schema_version")
            or companion_wrapper["schema_version"] != companion.get("runner_schema_version")
            or companion.get("raw_gate_provider_identity_sha256")
            != raw_providers["review_gate"]["identity_sha256"]
            or companion.get("raw_final_producer_identity_sha256")
            != raw_providers["final_upstream"]["identity_sha256"]
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "context-fit companion attestation changed"
            )
        index = body["context_fit_cache_index"]
        if (
            index.get("registration") != self.context_fit_cache_index_registration
            or len(context_sources) != len(folds)
            or any(source.kind != "review_gate" for source in context_sources)
            or tuple(source.binding.get("outer_fold") for source in context_sources) != folds
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "context-fit replay index does not cover exactly the prepared folds"
            )
        index_path, index_sha = _split_registration(
            self.context_fit_cache_index_registration,
            label="context-fit cache index registration",
        )
        if (
            str(index_path) != index.get("path")
            or index_sha != index.get("sha256")
            or _sha256_file(index_path) != index_sha
            or index_path.stat().st_size != index.get("byte_count")
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "context-fit cache index registration bytes changed"
            )
        source_rows = index.get("sources")
        if (
            not isinstance(source_rows, list)
            or tuple(
                row.get("outer_fold") if isinstance(row, Mapping) else None for row in source_rows
            )
            != folds
        ):
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "context-fit index source audit is incomplete"
            )
        for row, source in zip(source_rows, context_sources):
            if canonical_json(row.get("authenticated_source_identity")) != canonical_json(
                source.identity()
            ):
                raise HierarchicalPreparationCacheReplayAuthenticationError(
                    "indexed first-gate source identity changed"
                )
            if canonical_json(source.run_attestation.gate_provider_identity) != canonical_json(
                raw_providers["review_gate"]["identity"]
            ) or canonical_json(source.run_attestation.final_producer_identity) != canonical_json(
                raw_providers["final_upstream"]["identity"]
            ):
                raise HierarchicalPreparationCacheReplayAuthenticationError(
                    "indexed first-gate source provider attestation changed"
                )
            for file_snapshot in source.files:
                if Path(file_snapshot.filename).suffix.lower() in _FORBIDDEN_INDEXED_SUFFIXES:
                    raise HierarchicalPreparationCacheReplayAuthenticationError(
                        "context-fit index names an executable checkpoint"
                    )
        if body["assurances"] != _ASSURANCES:
            raise HierarchicalPreparationCacheReplayAuthenticationError(
                "replay non-executable assurances changed"
            )


def authenticate_hierarchical_preparation_cache_replay(
    registration: str,
) -> AuthenticatedHierarchicalPreparationCacheReplay:
    """Load one mandatory ``REPLAY_MANIFEST_PATH::SHA256`` registration."""

    path, digest = _split_registration(registration, label="replay manifest registration")
    path = path.expanduser().resolve(strict=True)
    snapshot = path.read_bytes()
    if _sha256_bytes(snapshot) != digest:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "registered replay manifest SHA-256 mismatch"
        )
    raw = _parse_json(snapshot, label="hierarchical replay manifest")
    if set(raw) != _WRAPPER_FIELDS or not isinstance(raw.get("body"), Mapping):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "registered replay manifest has a wrong closed schema"
        )
    body = raw["body"]
    registrations = body.get("registrations")
    if not isinstance(registrations, Mapping):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "registered replay manifest lacks registrations"
        )
    replay = AuthenticatedHierarchicalPreparationCacheReplay(
        replay_manifest_path=path,
        replay_manifest_sha256=digest,
        preparation_root=Path(str(body.get("preparation_root"))),
        input_manifest_sha256=str(
            body.get("prepared_batch_input_manifest", {}).get("content_sha256", "")
        ),
        batch_approval_sha256=str(body.get("approved_batch", {}).get("approval_sha256", "")),
        dataset_sha256=str(body.get("dataset_sha256", "")),
        ordered_outer_folds=tuple(body.get("ordered_outer_folds", ())),
        spent_cache_registrations=tuple(registrations.get("review_spent_evidence_cache", ())),
        context_fit_cache_index_registration=str(
            tuple(registrations.get("context_fit_cache_index", ("",)))[0]
        ),
        _manifest_snapshot=snapshot,
    )
    replay.validate_authentication()
    return replay


def export_hierarchical_preparation_cache_replay(
    *,
    prepared_batch: "PreparedHierarchicalDiscoveryBatch",
    review_spent_evidence_provider: ContextFitReviewSpentEvidenceProvider,
    review_gate_provider: ContextFitUpstreamGateProvider,
    final_upstream_producer: FinalContextFitUpstreamProducer,
    destination: Path | str,
) -> AuthenticatedHierarchicalPreparationCacheReplay:
    """Export one legacy registration manifest without model execution.

    ``destination`` must be a nonexistent direct child of the hierarchical
    preparation directory.  Existing files are never overwritten, even when
    their bytes would happen to match.  Current prepared batches are rejected:
    their first-gate numerical materialization is intentionally deferred.
    """

    if type(prepared_batch) is not _prepared_class():
        raise TypeError("prepared_batch must be the exact PreparedHierarchicalDiscoveryBatch type")
    if hasattr(prepared_batch, "first_gate_materialization_intent_index_path"):
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "current hierarchical preparation defers first-gate numerical "
            "materialization; export spent-cache registrations directly"
        )
    input_wrapper, input_snapshot, input_path = _prepared_input_manifest(prepared_batch)
    preparation_root = input_path.parent
    raw_destination = _absolute_without_resolution(Path(destination))
    if raw_destination.parent != preparation_root:
        raise HierarchicalPreparationCacheReplayAuthenticationError(
            "replay destination must be a direct child of hierarchical preparation root"
        )
    if raw_destination.exists() or raw_destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite replay destination: {raw_destination}")
    input_body = input_wrapper["body"]
    raw_provider_records = _validate_raw_providers(
        prepared=prepared_batch,
        input_body=input_body,
        review_spent_evidence_provider=review_spent_evidence_provider,
        review_gate_provider=review_gate_provider,
        final_upstream_producer=final_upstream_producer,
    )
    folds = _prepared_fold_numbers(prepared_batch, input_body)
    spent_rows, spent_registrations = _spent_source_rows(
        prepared=prepared_batch,
        folds=folds,
        provider=review_spent_evidence_provider,
        provider_record=raw_provider_records["review_spent_evidence"],
    )
    companion_record, index_record, context_registration = _context_fit_records(
        prepared=prepared_batch,
        folds=folds,
        preparation_root=preparation_root,
        gate_provider=review_gate_provider,
        raw_provider_records=raw_provider_records,
    )
    fold_manifest_records = _fold_manifest_records(
        prepared=prepared_batch,
        folds=folds,
        preparation_root=preparation_root,
    )
    batch_record = _approved_batch_record(
        prepared=prepared_batch,
        preparation_root=preparation_root,
        folds=folds,
    )
    input_record = {
        "path": str(input_path),
        "sha256": _sha256_bytes(input_snapshot),
        "byte_count": len(input_snapshot),
        "file_sha256": _sha256_bytes(input_snapshot),
        "content_sha256": prepared_batch.input_manifest_sha256,
        "schema_version": input_wrapper["schema_version"],
    }
    # The generic file-record validator has a deliberately smaller schema;
    # retain its three canonical fields and add immutable manifest bindings.
    body = {
        "exporter_version": HIERARCHICAL_PREPARATION_CACHE_REPLAY_EXPORTER_VERSION,
        "preparation_root": str(preparation_root),
        "prepared_batch_input_manifest": input_record,
        "approved_batch": batch_record,
        "dataset_sha256": prepared_batch.dataset_sha256,
        "ordered_outer_folds": list(folds),
        "fold_manifests": fold_manifest_records,
        "raw_provider_identities": raw_provider_records,
        "code_identities": _current_code_identities(),
        "spent_cache_sources": spent_rows,
        "context_fit_companion_attestation": companion_record,
        "context_fit_cache_index": index_record,
        "registrations": {
            "review_spent_evidence_cache": list(spent_registrations),
            "context_fit_cache_index": [context_registration],
        },
        "assurances": dict(_ASSURANCES),
    }
    serialized = _serialized_replay(body)
    raw_destination.mkdir(mode=0o700)
    manifest_path = raw_destination / "hierarchical_preparation_cache_replay.json"
    try:
        with manifest_path.open("xb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:  # defensive; the directory was created exclusively
        raise FileExistsError(f"refusing to overwrite replay manifest: {manifest_path}") from exc
    replay = AuthenticatedHierarchicalPreparationCacheReplay(
        replay_manifest_path=manifest_path,
        replay_manifest_sha256=_sha256_bytes(serialized),
        preparation_root=preparation_root,
        input_manifest_sha256=prepared_batch.input_manifest_sha256,
        batch_approval_sha256=prepared_batch.approval_sha256,
        dataset_sha256=prepared_batch.dataset_sha256,
        ordered_outer_folds=tuple(folds),
        spent_cache_registrations=spent_registrations,
        context_fit_cache_index_registration=context_registration,
        _manifest_snapshot=serialized,
    )
    replay.validate_authentication()
    return replay


__all__ = [
    "HIERARCHICAL_PREPARATION_CACHE_REPLAY_SCHEMA_VERSION",
    "HIERARCHICAL_PREPARATION_CACHE_REPLAY_EXPORTER_VERSION",
    "HierarchicalPreparationCacheReplayAuthenticationError",
    "AuthenticatedHierarchicalPreparationCacheReplay",
    "authenticate_hierarchical_preparation_cache_replay",
    "export_hierarchical_preparation_cache_replay",
]
