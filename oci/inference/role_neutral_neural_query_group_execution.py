"""Fit-once, role-neutral learned-neural-query physical-group artifacts.

The executable neural-query discovery checkpoint exists only in the worker's
fresh scratch cache.  This executor fits the canonical physical owner once,
copies only the service-owned closed JSON and per-array NPY state into the
sealed artifact, removes the executable checkpoint, publishes any cumulative
fit-only aliases, and only then admits the primary owner's held-out text.

No public API in this module accepts held-out treatment or outcome values.
Configured capacities are acceptance allocations: a capacity that would omit
queries, patients, chunks, excerpts, terms, or fitted state aborts the attempt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .all_evidence_discovery_interfaces import NEURAL_QUERY_MOMENTS
from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .lossless_stage1_evidence_catalog import (
    NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
)
from .neural_cohort_witness import soft_retrieval_activations
from .neural_numerical_replay import (
    neural_float_arrays_within_tolerance,
    validate_neural_replay_settings,
)
from .neural_query_agentic_forest import NeuralQueryAgenticForestConfig
from .neural_query_context_backend import (
    NEURAL_QUERY_CONTEXT_SERVICE_ID,
    ContextFitNeuralQueryService,
    NeuralQueryContextBackend,
    validate_owned_discovery_snapshot,
)
from .neural_query_execution_topology import (
    NeuralQueryExecutionTopology,
)
from .production_neural_query_binary_layout import (
    validate_npy_array_set,
    write_npy_array_set,
)
from .production_stage1_legacy_scope_fragments import (
    build_role_neutral_fit_only_family_seal,
)
from .production_stage1_scope_scheduler import Stage1ScopePlan, Stage1ScopeSpec


ROLE_NEUTRAL_NEURAL_QUERY_GROUP_REQUEST_SCHEMA = (
    "production_role_neutral_neural_query_physical_group_request_v3"
)
ROLE_NEUTRAL_NEURAL_QUERY_FIT_STATE_SCHEMA = (
    "production_role_neutral_neural_query_fit_state_v2"
)
ROLE_NEUTRAL_NEURAL_QUERY_LOGICAL_VIEW_SCHEMA = (
    "production_role_neutral_neural_query_logical_view_v2"
)
ROLE_NEUTRAL_NEURAL_QUERY_GROUP_EXECUTION_SCHEMA = (
    "production_role_neutral_neural_query_group_execution_v2"
)
ROLE_NEUTRAL_NEURAL_QUERY_COVERAGE_SCHEMA = (
    "production_role_neutral_neural_query_complete_coverage_v1"
)

FAIL_CLOSED_EVIDENCE_CAPACITY_POLICY = "fail_closed_complete_evidence_v1"
COMPLETE_EMBEDDING_TEXT_POLICY = (
    "authenticated_nonbinding_chunk_and_token_limits_v1"
)
REGISTERED_HELDOUT_TRANSFORM_POLICY = (
    "owned_fit_snapshot_registered_heldout_text_only_no_labels_v1"
)
# Compatibility alias for callers compiled against the original exact-inner
# API.  The value is intentionally the generalized scientific policy so a
# full-outer request cannot acquire an exact-inner-only identity.
EXACT_INNER_TRANSFORM_POLICY = REGISTERED_HELDOUT_TRANSFORM_POLICY

_FIT_STATE_DIRECTORY = "fit_state"
_OWNED_SNAPSHOT_DIRECTORY = "owned_discovery"
_FIT_METADATA_FILE = "metadata.json"
_FIT_EVIDENCE_FILE = "evidence.json"
_FIT_SEAL_FILE = "fit_only_family_seal.json"
_LOGICAL_VIEW_DIRECTORY = "logical_views"
_TERMINAL_FILE = "execution_manifest.json"
_BANKS = ("treatment", "outcome", "effect")
_ROLE_BY_BANK = {
    "treatment": PROPENSITY_NUISANCE_FEATURE_ROLE,
    "outcome": OUTCOME_NUISANCE_FEATURE_ROLE,
    "effect": UNCALIBRATED_EFFECT_MODIFIER_ROLE,
}
_QUERY_CONFIG_FIELDS = tuple(field.name for field in fields(NeuralQueryAgenticForestConfig))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_EXECUTION_KEYS = frozenset(
    {
        "path",
        "dataset_path",
        "model_path",
        "cache_dir",
        "work_root",
        "scratch_root",
        "hostname",
        "host",
        "pid",
        "worker_pid",
        "devices",
        "device_ids",
        "gpu_ids",
        "gpu_count",
        "worker_count",
        "completion_order",
    }
)
_EXECUTABLE_SUFFIXES = frozenset(
    {".joblib", ".pkl", ".pickle", ".npz", ".pt", ".pth"}
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _closed_json(value: Any) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise TypeError("neural-query scientific state must be closed JSON") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _float_hex_sha256(values: Sequence[Any], *, label: str) -> str:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be one finite vector")
    return _sha256_json([float(value).hex() for value in array])


def _binary_vector(values: Sequence[Any], *, label: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.shape != (int(length),)
        or not np.isfinite(array).all()
        or set(np.unique(array).tolist()) != {0.0, 1.0}
    ):
        raise ValueError(
            f"{label} must align to fit rows and contain both binary values"
        )
    return array


def _text_sha256(row_ids: Sequence[int], texts: Sequence[str]) -> str:
    return _sha256_json(
        [
            {"row_id": int(row_id), "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
            for row_id, text in zip(row_ids, texts, strict=True)
        ]
    )


def _row_order_fingerprint(row_ids: Sequence[int]) -> str:
    return _sha256_json([int(row_id) for row_id in row_ids])


def _write_new_bytes(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace immutable artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_bytes(
        Path(path),
        (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _stable_file_sha256(path: Path, *, label: str) -> tuple[str, int]:
    target = Path(path)
    if target.is_symlink():
        raise ValueError(f"{label} cannot be a symbolic link")
    before = target.lstat()
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise ValueError(f"{label} must be one non-hard-linked regular file")
    digest = hashlib.sha256()
    size = 0
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    after = target.lstat()
    signature = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        tuple(getattr(before, field) for field in signature)
        != tuple(getattr(after, field) for field in signature)
        or size != int(after.st_size)
    ):
        raise RuntimeError(f"{label} changed while being authenticated")
    return digest.hexdigest(), size


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    target = Path(path)
    before_digest, before_size = _stable_file_sha256(target, label=label)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    if _stable_file_sha256(target, label=label) != (before_digest, before_size):
        raise RuntimeError(f"{label} changed while being decoded")
    return value


def _tree_descriptor(root: Path) -> dict[str, Any]:
    tree = Path(root)
    if tree.is_symlink() or not tree.is_dir():
        raise ValueError("neural-query artifact tree must be one real directory")
    inventory: list[dict[str, Any]] = []
    for path in sorted(tree.rglob("*"), key=lambda item: item.relative_to(tree).as_posix()):
        relative = path.relative_to(tree).as_posix()
        if path.is_symlink():
            raise ValueError("neural-query artifact tree cannot contain symbolic links")
        if path.is_dir():
            inventory.append({"relative_path": relative, "kind": "directory"})
            continue
        digest, size = _stable_file_sha256(path, label=f"artifact {relative}")
        inventory.append(
            {
                "relative_path": relative,
                "kind": "file",
                "sha256": digest,
                "size_bytes": size,
            }
        )
    if not inventory:
        raise ValueError("neural-query artifact tree is empty")
    body = {
        "schema_version": "production_role_neutral_neural_query_tree_v1",
        "inventory": inventory,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _validate_no_execution_locators(value: Any, *, path: str = "identity") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.casefold() in _FORBIDDEN_EXECUTION_KEYS:
                raise ValueError(
                    f"service scientific identity contains execution locator {path}.{key}"
                )
            _validate_no_execution_locators(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_no_execution_locators(child, path=f"{path}[{index}]")


def _validated_query_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("query_config must be one explicit mapping")
    raw = copy.deepcopy(dict(value))
    if tuple(raw) != _QUERY_CONFIG_FIELDS and set(raw) != set(_QUERY_CONFIG_FIELDS):
        missing = sorted(set(_QUERY_CONFIG_FIELDS) - set(raw))
        extra = sorted(set(raw) - set(_QUERY_CONFIG_FIELDS))
        raise ValueError(
            "query_config must explicitly contain every closed setting; "
            f"missing={missing}, extra={extra}"
        )
    ordered = {name: raw[name] for name in _QUERY_CONFIG_FIELDS}
    try:
        config = NeuralQueryAgenticForestConfig(**ordered)
    except TypeError as exc:
        raise ValueError("query_config cannot be constructed exactly") from exc
    config.validate()
    normalized = _closed_json(asdict(config))
    if _canonical_json(normalized) != _canonical_json(_closed_json(ordered)):
        raise ValueError("query_config changed while being normalized")
    return normalized


def _validated_scientific_configuration(
    *,
    query_config: Mapping[str, Any],
    nuisance_folds: int,
    seed: int,
    outcome_type: str,
    service_scientific_identity: Mapping[str, Any],
    evidence_capacity_policy: str,
    embedding_text_coverage_policy: str,
    heldout_transform_policy: str,
    replay_comparison_policy: str,
    replay_relative_tolerance: float,
    replay_absolute_tolerance: float,
) -> dict[str, Any]:
    query = _validated_query_configuration(query_config)
    if isinstance(nuisance_folds, bool) or not isinstance(nuisance_folds, int):
        raise TypeError("nuisance_folds must be an explicit integer")
    if int(nuisance_folds) < 2:
        raise ValueError("nuisance_folds must be at least two")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an explicit integer")
    outcome = str(outcome_type).strip().lower()
    if outcome != "binary":
        raise ValueError("version 1 role-neutral neural queries require binary outcome")
    if evidence_capacity_policy != FAIL_CLOSED_EVIDENCE_CAPACITY_POLICY:
        raise ValueError("evidence capacity policy must explicitly fail closed")
    if embedding_text_coverage_policy != COMPLETE_EMBEDDING_TEXT_POLICY:
        raise ValueError("embedding text coverage policy must explicitly be complete")
    if heldout_transform_policy != REGISTERED_HELDOUT_TRANSFORM_POLICY:
        raise ValueError("held-out transform policy must explicitly forbid labels")
    replay_policy, replay_rtol, replay_atol = validate_neural_replay_settings(
        policy=replay_comparison_policy,
        relative_tolerance=replay_relative_tolerance,
        absolute_tolerance=replay_absolute_tolerance,
    )
    service_identity = _closed_json(service_scientific_identity)
    if (
        not isinstance(service_identity, dict)
        or service_identity.get("service") != NEURAL_QUERY_CONTEXT_SERVICE_ID
    ):
        raise ValueError("service_scientific_identity is not a context-fit query service")
    _validate_no_execution_locators(service_identity)
    if (
        _canonical_json(service_identity.get("query_config"))
        != _canonical_json(query)
        or service_identity.get("nuisance_folds") != int(nuisance_folds)
        or service_identity.get("seed") != int(seed)
        or service_identity.get("outcome_type") != outcome
        or service_identity.get("gate_labels_accepted") is not False
        or service_identity.get("novel_semantic_encoding_allowed") is not False
    ):
        raise ValueError(
            "explicit neural-query settings differ from the service scientific identity"
        )
    body = {
        "query_config": query,
        "nuisance_folds": int(nuisance_folds),
        "seed": int(seed),
        "outcome_type": outcome,
        "service_scientific_identity": service_identity,
        "evidence_capacity_policy": evidence_capacity_policy,
        "embedding_text_coverage_policy": embedding_text_coverage_policy,
        "heldout_transform_policy": heldout_transform_policy,
        "replay_comparison_policy": replay_policy,
        "replay_relative_tolerance": replay_rtol,
        "replay_absolute_tolerance": replay_atol,
    }
    return {**body, "content_sha256": _sha256_json(body)}


@dataclass(frozen=True)
class RoleNeutralNeuralQueryPhysicalGroupRequest:
    """Closed, device-neutral authority for one learned-query physical fit."""

    plan_scientific_content_sha256: str
    physical_owner: Stage1ScopeSpec
    logical_members: tuple[Stage1ScopeSpec, ...]
    scientific_configuration: Mapping[str, Any]
    content_sha256: str
    authority_plan: Stage1ScopePlan = field(repr=False, compare=False)

    @classmethod
    def from_plan(
        cls,
        *,
        plan: Stage1ScopePlan,
        physical_owner_scope_id: str,
        query_config: Mapping[str, Any],
        nuisance_folds: int,
        seed: int,
        outcome_type: str,
        service_scientific_identity: Mapping[str, Any],
        evidence_capacity_policy: str,
        embedding_text_coverage_policy: str,
        replay_comparison_policy: str,
        replay_relative_tolerance: float,
        replay_absolute_tolerance: float,
        heldout_transform_policy: str | None = None,
        exact_transform_policy: str | None = None,
    ) -> "RoleNeutralNeuralQueryPhysicalGroupRequest":
        if not isinstance(plan, Stage1ScopePlan):
            raise TypeError("role-neutral neural-query request requires a Stage1ScopePlan")
        owner = plan.scope(str(physical_owner_scope_id))
        if plan.physical_owner(owner.scope_id).scope_id != owner.scope_id:
            raise ValueError("role-neutral neural-query request must name a physical owner")
        groups = [
            members
            for candidate, members in plan.physical_scope_groups
            if candidate.scope_id == owner.scope_id
        ]
        if len(groups) != 1:
            raise RuntimeError("physical neural-query owner has no unique logical group")
        members = groups[0]
        if not members or members[0].scope_id != owner.scope_id:
            raise ValueError("physical neural-query logical group changed owner order")
        if any(
            member.scope_seed != owner.scope_seed
            or tuple(member.fit_row_ids) != tuple(owner.fit_row_ids)
            for member in members
        ):
            raise ValueError(
                "neural-query physical reuse requires identical ordered fit rows and seed"
            )
        aliases = members[1:]
        if aliases and (
            owner.scope_kind != "exact_inner"
            or any(member.scope_kind != "cumulative_spent" for member in aliases)
        ):
            raise ValueError(
                "neural-query reuse supports exact-inner/cumulative groups only"
            )
        if (
            heldout_transform_policy is not None
            and exact_transform_policy is not None
            and heldout_transform_policy != exact_transform_policy
        ):
            raise ValueError("held-out transform policy aliases disagree")
        selected_transform_policy = (
            heldout_transform_policy
            if heldout_transform_policy is not None
            else exact_transform_policy
        )
        configuration = _validated_scientific_configuration(
            query_config=query_config,
            nuisance_folds=nuisance_folds,
            seed=seed,
            outcome_type=outcome_type,
            service_scientific_identity=service_scientific_identity,
            evidence_capacity_policy=evidence_capacity_policy,
            embedding_text_coverage_policy=embedding_text_coverage_policy,
            heldout_transform_policy=selected_transform_policy,
            replay_comparison_policy=replay_comparison_policy,
            replay_relative_tolerance=replay_relative_tolerance,
            replay_absolute_tolerance=replay_absolute_tolerance,
        )
        body = {
            "schema_version": ROLE_NEUTRAL_NEURAL_QUERY_GROUP_REQUEST_SCHEMA,
            "plan_scientific_content_sha256": plan.scientific_content_sha256,
            "physical_owner": owner.as_dict(),
            "logical_members": [member.as_dict() for member in members],
            "logical_scope_count": len(members),
            "fit_row_ids": list(owner.fit_row_ids),
            "fit_row_order_fingerprint": _row_order_fingerprint(owner.fit_row_ids),
            "canonical_group_seed": int(owner.scope_seed),
            "scientific_configuration": configuration,
            "heldout_labels_supplied": False,
            "cumulative_heldout_rows_supplied_to_fit": False,
            "execution_devices_bound_to_scientific_identity": False,
        }
        return cls(
            plan_scientific_content_sha256=plan.scientific_content_sha256,
            physical_owner=owner,
            logical_members=members,
            scientific_configuration=configuration,
            content_sha256=_sha256_json(body),
            authority_plan=plan,
        )

    def as_dict(self) -> dict[str, Any]:
        _require_sha256(
            self.plan_scientific_content_sha256,
            label="neural-query scientific plan identity",
        )
        if (
            not isinstance(self.authority_plan, Stage1ScopePlan)
            or self.authority_plan.scientific_content_sha256
            != self.plan_scientific_content_sha256
        ):
            raise RuntimeError("neural-query request plan authority changed")
        configuration = _validated_scientific_configuration(
            query_config=self.scientific_configuration.get("query_config"),
            nuisance_folds=self.scientific_configuration.get("nuisance_folds"),
            seed=self.scientific_configuration.get("seed"),
            outcome_type=self.scientific_configuration.get("outcome_type"),
            service_scientific_identity=self.scientific_configuration.get(
                "service_scientific_identity"
            ),
            evidence_capacity_policy=self.scientific_configuration.get(
                "evidence_capacity_policy"
            ),
            embedding_text_coverage_policy=self.scientific_configuration.get(
                "embedding_text_coverage_policy"
            ),
            heldout_transform_policy=self.scientific_configuration.get(
                "heldout_transform_policy"
            ),
            replay_comparison_policy=self.scientific_configuration.get(
                "replay_comparison_policy"
            ),
            replay_relative_tolerance=self.scientific_configuration.get(
                "replay_relative_tolerance"
            ),
            replay_absolute_tolerance=self.scientific_configuration.get(
                "replay_absolute_tolerance"
            ),
        )
        if configuration != dict(self.scientific_configuration):
            raise RuntimeError("neural-query scientific configuration changed")
        owner = self.physical_owner
        if (
            not self.logical_members
            or self.logical_members[0].scope_id != owner.scope_id
            or len({member.scope_id for member in self.logical_members})
            != len(self.logical_members)
            or any(
                member.scope_seed != owner.scope_seed
                or tuple(member.fit_row_ids) != tuple(owner.fit_row_ids)
                for member in self.logical_members
            )
            or (
                len(self.logical_members) > 1
                and (
                    owner.scope_kind != "exact_inner"
                    or any(
                        member.scope_kind != "cumulative_spent"
                        for member in self.logical_members[1:]
                    )
                )
            )
        ):
            raise ValueError("neural-query logical group authority is invalid")
        body = {
            "schema_version": ROLE_NEUTRAL_NEURAL_QUERY_GROUP_REQUEST_SCHEMA,
            "plan_scientific_content_sha256": self.plan_scientific_content_sha256,
            "physical_owner": owner.as_dict(),
            "logical_members": [member.as_dict() for member in self.logical_members],
            "logical_scope_count": len(self.logical_members),
            "fit_row_ids": list(owner.fit_row_ids),
            "fit_row_order_fingerprint": _row_order_fingerprint(owner.fit_row_ids),
            "canonical_group_seed": int(owner.scope_seed),
            "scientific_configuration": configuration,
            "heldout_labels_supplied": False,
            "cumulative_heldout_rows_supplied_to_fit": False,
            "execution_devices_bound_to_scientific_identity": False,
        }
        if _sha256_json(body) != self.content_sha256:
            raise RuntimeError("role-neutral neural-query group request changed")
        return {**body, "content_sha256": self.content_sha256}


def _validate_service_against_request(
    service: ContextFitNeuralQueryService,
    request: RoleNeutralNeuralQueryPhysicalGroupRequest,
) -> dict[str, Any]:
    if not isinstance(service, ContextFitNeuralQueryService):
        raise TypeError("service must be a ContextFitNeuralQueryService")
    service_identity = _closed_json(service.identity())
    expected = request.scientific_configuration["service_scientific_identity"]
    if service_identity != expected:
        raise ValueError("live neural-query service differs from the scientific request")
    if (
        _canonical_json(asdict(service.query_config))
        != _canonical_json(request.scientific_configuration["query_config"])
        or int(service.nuisance_folds)
        != int(request.scientific_configuration["nuisance_folds"])
        or int(service.seed) != int(request.scientific_configuration["seed"])
        or str(service.outcome_type) != request.scientific_configuration["outcome_type"]
    ):
        raise ValueError("live neural-query service settings changed")
    return service_identity


def _fit_chunk_coverage(
    *,
    service: ContextFitNeuralQueryService,
    request: RoleNeutralNeuralQueryPhysicalGroupRequest,
    fit_texts: tuple[str, ...],
) -> tuple[dict[str, Any], Any]:
    owner = request.physical_owner
    rows, texts, bound = service._bind_rows_and_texts(
        owner.fit_row_ids,
        fit_texts,
        row_name="fit_row_ids",
        text_name="fit_texts",
    )
    if rows != owner.fit_row_ids or texts != fit_texts:
        raise RuntimeError("neural-query fit binding changed owner row order or text")
    token_bounded = tuple(getattr(bound, "token_bounded_row_ids", ()))
    if token_bounded:
        raise ValueError(
            "neural-query fit cache admits token-bounded source rows; complete-text "
            "policy requires exact nontruncated cache rows"
        )
    matrices = tuple(bound.chunk_matrices(rows))
    chunk_texts = tuple(bound.chunk_texts(rows))
    if len(matrices) != len(rows) or len(chunk_texts) != len(rows):
        raise RuntimeError("neural-query cache omitted one or more fit rows")
    counts = tuple(len(chunks) for chunks in chunk_texts)
    if any(
        count < 1
        or np.asarray(matrix).ndim != 2
        or int(np.asarray(matrix).shape[0]) != count
        for count, matrix in zip(counts, matrices, strict=True)
    ):
        raise ValueError("neural-query cache chunk matrices and text are incomplete")
    metadata = _closed_json(service.embedding_cache.metadata)
    if (
        metadata.get("chunk_cap_nonbinding") is not True
        or metadata.get("semantic_truncation_allowed") is not False
        or metadata.get("tokenizer_truncation_allowed") is not False
        or "nontruncating" not in str(metadata.get("chunking_mode") or "")
    ):
        raise ValueError(
            "embedding cache lacks authenticated nonbinding chunk/token coverage"
        )
    query = request.scientific_configuration["query_config"]
    chunk_capacity = query["evidence_chunks_per_patient_per_query"]
    maximum_chunks = max(counts)
    if chunk_capacity is not None and maximum_chunks > int(chunk_capacity):
        raise ValueError(
            "configured neural-query evidence chunk allocation would omit fit "
            f"chunks: required={maximum_chunks}, configured={int(chunk_capacity)}"
        )
    configured_background_capacity = query["evidence_background_patients"]
    patient_capacity = (
        None
        if configured_background_capacity is None
        else int(query["evidence_top_patients"])
        + int(configured_background_capacity)
    )
    if patient_capacity is not None and patient_capacity < len(rows):
        raise ValueError(
            "configured neural-query patient evidence allocation would omit fit "
            f"patients: required={len(rows)}, configured={patient_capacity}"
        )
    excerpt_capacity = query["evidence_excerpt_chars"]
    maximum_excerpt_chars = max(
        len(str(chunk))
        for row_chunks in chunk_texts
        for chunk in row_chunks
    )
    if excerpt_capacity is not None and maximum_excerpt_chars > int(excerpt_capacity):
        raise ValueError(
            "configured neural-query evidence excerpt allocation would truncate "
            f"a chunk: required={maximum_excerpt_chars}, "
            f"configured={int(excerpt_capacity)}"
        )
    chunk_inventory = [
        {
            "row_id": int(row_id),
            "chunk_count": len(row_chunks),
            "ordered_chunk_text_sha256": _sha256_json(
                [
                    hashlib.sha256(str(chunk).encode("utf-8")).hexdigest()
                    for chunk in row_chunks
                ]
            ),
        }
        for row_id, row_chunks in zip(rows, chunk_texts, strict=True)
    ]
    body = {
        "schema_version": ROLE_NEUTRAL_NEURAL_QUERY_COVERAGE_SCHEMA,
        "fit_row_count": len(rows),
        "fit_row_order_fingerprint": _row_order_fingerprint(rows),
        "fit_text_sha256": _text_sha256(rows, texts),
        "chunk_count_by_fit_row": list(counts),
        "total_fit_chunk_count": sum(counts),
        "maximum_fit_chunks_per_row": maximum_chunks,
        "ordered_chunk_inventory_sha256": _sha256_json(chunk_inventory),
        "embedding_chunk_cap_nonbinding": True,
        "embedding_tokenizer_truncation_allowed": False,
        "configured_evidence_chunk_capacity": chunk_capacity,
        "evidence_chunk_capacity_nonbinding": True,
        "configured_patient_evidence_capacity": patient_capacity,
        "complete_background_patient_allocation": (
            configured_background_capacity is None
        ),
        "patient_evidence_capacity_nonbinding": True,
        "maximum_fit_chunk_chars": maximum_excerpt_chars,
        "configured_excerpt_char_capacity": excerpt_capacity,
        "excerpt_capacity_nonbinding": True,
        "configured_term_count_capacity": query["evidence_top_ngrams"],
        "configured_safe_term_token_capacity": int(
            query["evidence_safe_term_max_tokens"]
        ),
        "configured_safe_term_char_capacity": int(
            query["evidence_safe_term_max_chars"]
        ),
        "capacity_policy": FAIL_CLOSED_EVIDENCE_CAPACITY_POLICY,
        "text_truncation_applied": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}, bound


def _validate_safe_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    request: RoleNeutralNeuralQueryPhysicalGroupRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _closed_json(list(evidence))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("safe neural-query evidence must be a list of objects")
    query = request.scientific_configuration["query_config"]
    expected_by_bank = {
        "treatment": int(query["treatment_query_count"]),
        "outcome": int(query["outcome_query_count"]),
        "effect": int(query["effect_query_count"]),
    }
    observed_by_bank = {bank: 0 for bank in _BANKS}
    query_ids: list[str] = []
    term_count = 0
    maximum_term_chars = 0
    maximum_term_tokens = 0
    for row in rows:
        bank = str(row.get("bank") or "")
        if bank not in observed_by_bank:
            raise ValueError("safe neural-query evidence has an unknown bank")
        observed_by_bank[bank] += 1
        query_id = str(row.get("query_id") or "")
        if not query_id:
            raise ValueError("safe neural-query evidence has an empty query ID")
        query_ids.append(query_id)
        if (
            row.get("statistical_gate_applied") is not False
            or row.get("top_chunks") != []
        ):
            raise ValueError("safe query evidence exposed row-level text or a selection gate")
        terms = row.get("top_contrastive_ngrams")
        if not isinstance(terms, list):
            raise ValueError("safe query evidence has no closed term list")
        term_capacity = query["evidence_top_ngrams"]
        if term_capacity is not None and len(terms) > int(term_capacity):
            raise ValueError("safe query evidence exceeded its configured term allocation")
        for term_row in terms:
            if not isinstance(term_row, dict):
                raise ValueError("safe query evidence contains a malformed term")
            term = str(term_row.get("term") or "")
            if not term:
                raise ValueError("safe query evidence contains an empty term")
            maximum_term_chars = max(maximum_term_chars, len(term))
            maximum_term_tokens = max(maximum_term_tokens, len(term.split()))
            term_count += 1
    if observed_by_bank != expected_by_bank:
        raise ValueError(
            "safe neural-query evidence omitted or added configured queries: "
            f"expected={expected_by_bank}, observed={observed_by_bank}"
        )
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("safe neural-query evidence query IDs are not unique")
    if maximum_term_chars > int(query["evidence_safe_term_max_chars"]) or (
        maximum_term_tokens > int(query["evidence_safe_term_max_tokens"])
    ):
        raise ValueError("safe neural-query evidence exceeded its declared term transport")
    body = {
        "expected_query_count_by_bank": expected_by_bank,
        "observed_query_count_by_bank": observed_by_bank,
        "ordered_query_ids_sha256": _sha256_json(query_ids),
        "all_configured_queries_retained": True,
        "term_count": term_count,
        "maximum_term_chars": maximum_term_chars,
        "maximum_term_tokens": maximum_term_tokens,
        "term_count_capacity_nonbinding": True,
        "term_transport_capacity_nonbinding": True,
        "row_level_chunks_removed_after_complete_fit_side_contrast": True,
    }
    return rows, {**body, "content_sha256": _sha256_json(body)}


def _purge_owned_executable_checkpoint(
    service: ContextFitNeuralQueryService,
    *,
    cache_key: str,
) -> dict[str, Any]:
    key = _require_sha256(cache_key, label="owned neural-query cache key")
    base = Path(service.cache_dir)
    target = base / key
    removed: list[str] = []
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise ValueError("owned neural-query executable cache entry is not a directory")
        members = sorted(target.iterdir(), key=lambda path: path.name)
        expected = {"manifest.json", "query_discovery.joblib"}
        if {member.name for member in members} != expected:
            raise ValueError("owned neural-query executable cache entry has extra or missing files")
        for member in members:
            if member.is_symlink():
                raise ValueError("owned executable cache file cannot be a symbolic link")
            before = member.lstat()
            if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
                raise ValueError("owned executable cache file must be non-hard-linked")
        for member in members:
            removed.append(member.name)
            member.unlink()
        target.rmdir()
    if target.exists() or target.is_symlink():
        raise RuntimeError("owned neural-query executable cache entry was not removed")
    remaining_executable = [
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file() and path.suffix.casefold() in _EXECUTABLE_SUFFIXES
    ] if base.exists() else []
    if remaining_executable:
        raise ValueError(
            "neural-query scratch contains another executable checkpoint; "
            "use one fresh short-lived service cache per physical fit"
        )
    body = {
        "schema_version": "production_neural_query_executable_scratch_cleanup_v1",
        "cache_key": key,
        "removed_files": removed,
        "owned_cache_entry_absent": True,
        "remaining_executable_checkpoint_count": 0,
        "sealed_artifact_received_executable_state": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _snapshot_arrays(
    snapshot_root: Path,
    *,
    expected_service_identity_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = validate_owned_discovery_snapshot(
        snapshot_root,
        expected_service_identity_sha256=expected_service_identity_sha256,
    )
    order = tuple(metadata["array_order"])
    descriptor, arrays = validate_npy_array_set(
        snapshot_root / metadata["arrays_directory"],
        expected_order=order,
        expected_inventory=metadata["array_inventory"],
    )
    if (
        descriptor["index_sha256"] != metadata["arrays_index_sha256"]
        or descriptor["content_sha256"] != metadata["arrays_content_sha256"]
    ):
        raise RuntimeError("owned neural-query snapshot arrays changed")
    return dict(metadata), dict(arrays)


def _transform_owned_snapshot(
    *,
    service: ContextFitNeuralQueryService,
    snapshot_root: Path,
    heldout_row_ids: tuple[int, ...],
    heldout_texts: tuple[str, ...],
    expected_service_identity_sha256: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], np.ndarray]:
    metadata, arrays = _snapshot_arrays(
        snapshot_root,
        expected_service_identity_sha256=expected_service_identity_sha256,
    )
    rows, texts, bound = service._bind_rows_and_texts(
        heldout_row_ids,
        heldout_texts,
        row_name="registered_heldout_row_ids",
        text_name="registered_heldout_texts",
    )
    if rows != heldout_row_ids or texts != heldout_texts:
        raise RuntimeError("held-out neural-query binding changed row order")
    chunks = bound.chunk_matrices(rows)
    discovery_banks = metadata["discovery_metadata"]["banks"]
    names: list[str] = []
    kinds: list[str] = []
    roles: list[str] = []
    columns: list[np.ndarray] = []
    for bank_index, bank in enumerate(_BANKS):
        # ``soft_retrieval_activations`` normalizes query rows in place.  The
        # authenticated mmap is deliberately read-only, so transform a private
        # numerical copy while preserving the sealed bytes.
        queries = np.array(
            arrays[f"{bank}_queries"],
            dtype=np.float32,
            copy=True,
            order="C",
        )
        records = discovery_banks[bank]["records"]
        expected = int(metadata["query_count_by_bank"][bank])
        if len(records) != expected or queries.shape[0] != expected:
            raise ValueError("owned neural-query snapshot record count changed")
        activations = soft_retrieval_activations(
            chunks,
            queries,
            temperature=float(service.query_config.temperature),
            device=service.devices[bank_index % len(service.devices)],
            patient_batch_size=int(
                service.query_config.retrieval_patient_batch_size
            ),
        )
        if (
            activations.shape != (len(rows), expected)
            or not np.isfinite(activations).all()
        ):
            raise ValueError("owned neural-query held-out transform is invalid")
        fit_scores = np.asarray(
            [float(record["fit_standardized_score"]) for record in records],
            dtype=np.float64,
        )
        if fit_scores.shape != (expected,) or not np.isfinite(fit_scores).all():
            raise ValueError("owned neural-query fit-score state is invalid")
        signed = np.asarray(activations, dtype=np.float64) * np.sign(fit_scores)[None, :]
        descending = np.sort(signed, axis=1, kind="stable")[:, ::-1]
        bank_names = (
            f"neural_query_{bank}_signed_mean",
            f"neural_query_{bank}_absolute_max",
            *(
                f"neural_query_{bank}_signed_order_{rank:02d}"
                for rank in range(1, expected + 1)
            ),
        )
        bank_values = np.column_stack(
            (
                np.mean(descending, axis=1),
                np.max(np.abs(activations), axis=1),
                descending,
            )
        )
        for column, name in enumerate(bank_names):
            names.append(name)
            kinds.append(f"neural_query_{bank}_moments")
            roles.append(_ROLE_BY_BANK[bank])
            columns.append(np.asarray(bank_values[:, column], dtype=np.float64))
    values = np.column_stack(columns)
    if values.shape != (len(rows), len(names)) or not np.isfinite(values).all():
        raise RuntimeError("owned neural-query held-out feature matrix is incomplete")
    return tuple(names), tuple(kinds), tuple(roles), values


def _producer_identity() -> str:
    from . import neural_query_agentic_forest as evidence_module
    from . import neural_query_context_backend as service_module
    from . import production_neural_query_binary_layout as binary_module

    files = (
        Path(__file__).resolve(),
        Path(evidence_module.__file__).resolve(),
        Path(service_module.__file__).resolve(),
        Path(binary_module.__file__).resolve(),
    )
    return _sha256_json(
        {
            "schema_version": "production_role_neutral_neural_query_producer_identity_v1",
            "transitive_source_sha256s": [
                _stable_file_sha256(path, label=f"producer source {path.name}")[0]
                for path in files
            ],
        }
    )


def _registration(path: Path, *, relative_to: Path) -> dict[str, Any]:
    digest, size = _stable_file_sha256(path, label=f"artifact {path.name}")
    return {
        "relative_path": path.relative_to(relative_to).as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def _fit_state_metadata(
    *,
    request: RoleNeutralNeuralQueryPhysicalGroupRequest,
    service_identity: Mapping[str, Any],
    fit_texts: tuple[str, ...],
    treatment: np.ndarray,
    outcome: np.ndarray,
    snapshot_metadata: Mapping[str, Any],
    snapshot_tree: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    evidence_registration: Mapping[str, Any],
    chunk_coverage: Mapping[str, Any],
    evidence_coverage: Mapping[str, Any],
    scratch_cleanup: Mapping[str, Any],
    producer_identity_sha256: str,
) -> dict[str, Any]:
    owner = request.physical_owner
    body = {
        "schema_version": ROLE_NEUTRAL_NEURAL_QUERY_FIT_STATE_SCHEMA,
        "group_request_content_sha256": request.content_sha256,
        "plan_scientific_content_sha256": request.plan_scientific_content_sha256,
        "physical_owner_scope_id": owner.scope_id,
        "physical_owner_scope_sha256": owner.as_dict()["scope_sha256"],
        "fit_row_ids": list(owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(owner.fit_row_ids),
        "fit_text_sha256": _text_sha256(owner.fit_row_ids, fit_texts),
        "fit_treatment_sha256": _float_hex_sha256(
            treatment,
            label="fit treatment",
        ),
        "fit_outcome_sha256": _float_hex_sha256(
            outcome,
            label="fit outcome",
        ),
        "scientific_configuration": copy.deepcopy(
            dict(request.scientific_configuration)
        ),
        "configuration_identity_sha256": request.scientific_configuration[
            "content_sha256"
        ],
        "service_scientific_identity_sha256": _sha256_json(service_identity),
        "producer_identity_sha256": producer_identity_sha256,
        "owned_discovery": {
            "relative_path": _OWNED_SNAPSHOT_DIRECTORY,
            "cache_key": snapshot_metadata["cache_key"],
            "snapshot_content_sha256": snapshot_metadata["content_sha256"],
            "snapshot_tree_sha256": snapshot_tree["content_sha256"],
            "owned_discovery_content_sha256": snapshot_metadata[
                "owned_discovery_content_sha256"
            ],
            "array_order": copy.deepcopy(snapshot_metadata["array_order"]),
            "array_inventory": copy.deepcopy(snapshot_metadata["array_inventory"]),
            "query_count_by_bank": copy.deepcopy(
                snapshot_metadata["query_count_by_bank"]
            ),
            "fit_row_count": int(snapshot_metadata["fit_row_count"]),
            "executable_serialization_present": False,
        },
        "evidence_artifact": {
            **copy.deepcopy(dict(evidence_registration)),
            "content_sha256": _sha256_json(evidence_payload),
        },
        "chunk_coverage": copy.deepcopy(dict(chunk_coverage)),
        "evidence_coverage": copy.deepcopy(dict(evidence_coverage)),
        "executable_scratch_cleanup": copy.deepcopy(dict(scratch_cleanup)),
        "fit_completed_before_registered_heldout_text_access": True,
        "registered_heldout_text_accessed": False,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
        "storage_format": "closed_json_and_per_array_npy_only",
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _logical_view(
    *,
    request: RoleNeutralNeuralQueryPhysicalGroupRequest,
    member: Stage1ScopeSpec,
    fit_seal_registration: Mapping[str, Any],
    fit_seal: Mapping[str, Any],
    primary: bool,
    heldout_text_sha256: str | None = None,
    prediction_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if primary:
        if (
            member.scope_id != request.physical_owner.scope_id
            or heldout_text_sha256 is None
            or prediction_artifact is None
        ):
            raise ValueError("primary neural-query logical view is incomplete")
        input_policy = "registered_heldout_row_ids_and_text_no_labels_v1"
    else:
        if (
            member.scope_id == request.physical_owner.scope_id
            or member.scope_kind != "cumulative_spent"
            or heldout_text_sha256 is not None
            or prediction_artifact is not None
        ):
            raise ValueError("cumulative neural-query view must be fit-only")
        input_policy = "fit_only_reference_no_heldout_rows_text_or_labels_v1"
    body = {
        "schema_version": ROLE_NEUTRAL_NEURAL_QUERY_LOGICAL_VIEW_SCHEMA,
        "group_request_content_sha256": request.content_sha256,
        "logical_scope_id": member.scope_id,
        "logical_scope_sha256": member.as_dict()["scope_sha256"],
        "logical_purpose": member.scope_kind,
        "physical_owner_scope_id": request.physical_owner.scope_id,
        "family": NEURAL_QUERY_MOMENTS,
        "fit_only_family_seal_sha256": fit_seal_registration["sha256"],
        "fit_only_family_seal_content_sha256": fit_seal["content_sha256"],
        "view_input_policy": input_policy,
        "logical_transform_performed": bool(primary),
        "logical_heldout_row_ids": (
            list(member.heldout_row_ids) if primary else None
        ),
        "logical_heldout_text_sha256": heldout_text_sha256,
        "prediction_artifact": (
            copy.deepcopy(dict(prediction_artifact))
            if prediction_artifact is not None
            else None
        ),
        "registered_heldout_text_accessed": bool(primary),
        "registered_heldout_labels_accessed": False,
        "reuses_live_physical_fit": True,
        "model_state_reloaded_for_primary_transform": False,
        "owned_safe_snapshot_replay_checked": bool(primary),
    }
    return {**body, "content_sha256": _sha256_json(body)}


def execute_role_neutral_neural_query_physical_group(
    *,
    request: RoleNeutralNeuralQueryPhysicalGroupRequest,
    output_root: Path | str,
    service: ContextFitNeuralQueryService,
    fit_texts: Sequence[str],
    fit_treatment: Sequence[Any],
    fit_outcome: Sequence[Any],
    execution_topology: NeuralQueryExecutionTopology | None = None,
    heldout_text_loader: Callable[[tuple[int, ...]], Sequence[str]] | None = None,
    exact_heldout_text_loader: Callable[
        [tuple[int, ...]], Sequence[str]
    ] | None = None,
) -> Mapping[str, Any]:
    """Fit and seal one learned-query owner before opening held-out text."""

    if not isinstance(request, RoleNeutralNeuralQueryPhysicalGroupRequest):
        raise TypeError("execution requires its typed neural-query group request")
    request.as_dict()
    topology = (
        NeuralQueryExecutionTopology(devices=tuple(service.devices))
        if execution_topology is None
        else execution_topology
    )
    if not isinstance(topology, NeuralQueryExecutionTopology):
        raise TypeError(
            "neural-query execution requires a typed deployment topology"
        )
    if tuple(service.devices) != topology.devices:
        raise ValueError(
            "live neural-query service devices differ from the reserved "
            "execution topology"
        )
    service_identity = _validate_service_against_request(service, request)
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("role-neutral neural-query output root must be absolute")
    if root.exists() or root.is_symlink():
        raise FileExistsError("role-neutral neural-query output root must be fresh")
    if heldout_text_loader is not None and exact_heldout_text_loader is not None:
        raise ValueError("supply only one held-out text loader")
    loader = (
        heldout_text_loader
        if heldout_text_loader is not None
        else exact_heldout_text_loader
    )
    if not callable(loader):
        raise TypeError("held-out text loader must be callable")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)

    owner = request.physical_owner
    texts = tuple(fit_texts)
    if (
        len(texts) != len(owner.fit_row_ids)
        or any(not isinstance(text, str) for text in texts)
    ):
        raise ValueError("fit texts must align exactly to physical owner rows")
    treatment = _binary_vector(
        fit_treatment,
        label="fit treatment",
        length=len(texts),
    )
    outcome = _binary_vector(
        fit_outcome,
        label="fit outcome",
        length=len(texts),
    )
    chunk_coverage, _bound = _fit_chunk_coverage(
        service=service,
        request=request,
        fit_texts=texts,
    )

    discovery, cache_key = service.discovery_for_context(
        outer_fold=owner.outer_fold,
        context_row_ids=owner.fit_row_ids,
        context_texts=texts,
        context_treatment=treatment,
        context_outcome=outcome,
    )
    service._validate_discovery(discovery)
    safe_evidence, evidence_coverage = _validate_safe_evidence(
        service.safe_evidence(
            discovery=discovery,
            context_row_ids=owner.fit_row_ids,
            context_texts=texts,
            device_offset=0,
        ),
        request=request,
    )
    evidence_payload = {
        "schema_version": NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
        "family": NEURAL_QUERY_MOMENTS,
        "architecture_evidence": safe_evidence,
    }

    fit_root = root / _FIT_STATE_DIRECTORY
    fit_root.mkdir(exist_ok=False)
    snapshot_root = fit_root / _OWNED_SNAPSHOT_DIRECTORY
    snapshot_metadata = service.write_owned_discovery_snapshot(
        cache_key=cache_key,
        output_dir=snapshot_root,
    )
    service_identity_sha256 = _sha256_json(service_identity)
    snapshot_metadata = validate_owned_discovery_snapshot(
        snapshot_root,
        expected_cache_key=cache_key,
        expected_service_identity_sha256=service_identity_sha256,
    )
    if (
        tuple(snapshot_metadata["binding"]["row_ids"]) != owner.fit_row_ids
        or int(snapshot_metadata["fit_row_count"]) != len(owner.fit_row_ids)
        or snapshot_metadata["query_count_by_bank"]
        != evidence_coverage["observed_query_count_by_bank"]
    ):
        raise RuntimeError("owned neural-query snapshot omitted fit rows or queries")
    snapshot_tree = _tree_descriptor(snapshot_root)

    evidence_path = fit_root / _FIT_EVIDENCE_FILE
    _write_new_json(evidence_path, evidence_payload)
    evidence_registration = _registration(evidence_path, relative_to=fit_root)
    scratch_cleanup = _purge_owned_executable_checkpoint(
        service,
        cache_key=cache_key,
    )
    producer_identity_sha256 = _producer_identity()
    fit_metadata = _fit_state_metadata(
        request=request,
        service_identity=service_identity,
        fit_texts=texts,
        treatment=treatment,
        outcome=outcome,
        snapshot_metadata=snapshot_metadata,
        snapshot_tree=snapshot_tree,
        evidence_payload=evidence_payload,
        evidence_registration=evidence_registration,
        chunk_coverage=chunk_coverage,
        evidence_coverage=evidence_coverage,
        scratch_cleanup=scratch_cleanup,
        producer_identity_sha256=producer_identity_sha256,
    )
    _write_new_json(fit_root / _FIT_METADATA_FILE, fit_metadata)
    fit_state_tree = _tree_descriptor(fit_root)
    fit_state_sha256 = fit_state_tree["content_sha256"]

    fit_seal = build_role_neutral_fit_only_family_seal(
        plan=request.authority_plan,
        physical_owner_scope_id=owner.scope_id,
        family=NEURAL_QUERY_MOMENTS,
        evidence_payload=evidence_payload,
        producer_identity_sha256=producer_identity_sha256,
        configuration_identity_sha256=request.scientific_configuration[
            "content_sha256"
        ],
        fit_state_artifact_sha256=fit_state_sha256,
    )
    seal_path = root / _FIT_SEAL_FILE
    _write_new_json(seal_path, fit_seal)
    seal_registration = {
        **_registration(seal_path, relative_to=root),
        "content_sha256": fit_seal["content_sha256"],
    }

    # Reopen every fit-side byte before the callback capable of admitting
    # registered held-out text becomes reachable.
    _validate_fit_side(
        root=root,
        request=request,
        expected_fit_texts=texts,
        expected_treatment=treatment,
        expected_outcome=outcome,
    )

    logical_root = root / _LOGICAL_VIEW_DIRECTORY
    logical_root.mkdir(exist_ok=False)
    events: list[dict[str, Any]] = [
        {
            "sequence": 1,
            "event": "fit_completed",
            "fit_state_artifact_sha256": fit_state_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        },
        {
            "sequence": 2,
            "event": "owned_executable_checkpoint_removed",
            "cache_key": cache_key,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        },
        {
            "sequence": 3,
            "event": "fit_family_artifact_sealed",
            "family": NEURAL_QUERY_MOMENTS,
            "fit_only_family_seal_sha256": seal_registration["sha256"],
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        },
    ]
    logical_registrations: list[dict[str, Any]] = []
    for alias_index, member in enumerate(request.logical_members[1:], start=1):
        view = _logical_view(
            request=request,
            member=member,
            fit_seal_registration=seal_registration,
            fit_seal=fit_seal,
            primary=False,
        )
        path = logical_root / f"{alias_index:03d}_cumulative.json"
        _write_new_json(path, view)
        logical_registrations.append(
            {
                "logical_scope_id": member.scope_id,
                "logical_purpose": member.scope_kind,
                **_registration(path, relative_to=root),
                "content_sha256": view["content_sha256"],
            }
        )
        events.append(
            {
                "sequence": len(events) + 1,
                "event": "cumulative_fit_only_view_published",
                "logical_scope_id": member.scope_id,
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
            }
        )

    heldout_texts = tuple(
        loader(tuple(owner.heldout_row_ids))
    )
    if (
        len(heldout_texts) != len(owner.heldout_row_ids)
        or any(not isinstance(text, str) for text in heldout_texts)
    ):
        raise ValueError("held-out text loader returned another row/text shape")
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "primary_heldout_text_opened",
            "logical_scope_id": owner.scope_id,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
        }
    )

    live_prediction = NeuralQueryContextBackend(service).fit_predict(
        outer_fold=owner.outer_fold,
        context_row_ids=owner.fit_row_ids,
        context_texts=texts,
        context_treatment=treatment,
        context_outcome=outcome,
        gate_row_ids=owner.heldout_row_ids,
        gate_texts=heldout_texts,
        work_dir=Path(service.cache_dir),
    )
    replay_names, replay_kinds, replay_roles, replay_values = _transform_owned_snapshot(
        service=service,
        snapshot_root=snapshot_root,
        heldout_row_ids=owner.heldout_row_ids,
        heldout_texts=heldout_texts,
        expected_service_identity_sha256=service_identity_sha256,
    )
    if (
        tuple(live_prediction.gate_row_ids) != owner.heldout_row_ids
        or tuple(live_prediction.feature_names) != replay_names
        or tuple(live_prediction.feature_kinds) != replay_kinds
        or tuple(live_prediction.feature_roles) != replay_roles
        or not neural_float_arrays_within_tolerance(
            np.asarray(live_prediction.feature_values, dtype=np.float64),
            replay_values,
            policy=request.scientific_configuration[
                "replay_comparison_policy"
            ],
            relative_tolerance=request.scientific_configuration[
                "replay_relative_tolerance"
            ],
            absolute_tolerance=request.scientific_configuration[
                "replay_absolute_tolerance"
            ],
        )
    ):
        raise RuntimeError(
            "live owned neural-query transform differs from safe snapshot "
            "replay beyond its declared tolerance"
        )
    if (
        live_prediction.calibrated_source_names
        or live_prediction.calibrated_source_kinds
        or np.asarray(live_prediction.calibrated_source_values).shape
        != (len(owner.heldout_row_ids), 0)
    ):
        raise RuntimeError("neural-query held-out transform exposed calibrated effects")

    prediction_root = logical_root / "primary_predictions"
    prediction_descriptor = write_npy_array_set(
        prediction_root,
        {
            "gate_row_ids": np.asarray(owner.heldout_row_ids, dtype=np.int64),
            "feature_values": replay_values.astype(np.float64, copy=False),
        },
        ordered_names=("gate_row_ids", "feature_values"),
    )
    prediction_artifact = {
        "relative_path": prediction_root.relative_to(root).as_posix(),
        "array_order": ["gate_row_ids", "feature_values"],
        "array_inventory": prediction_descriptor["array_inventory"],
        "index_sha256": prediction_descriptor["index_sha256"],
        "arrays_content_sha256": prediction_descriptor["content_sha256"],
        "feature_names": list(replay_names),
        "feature_kinds": list(replay_kinds),
        "feature_roles": list(replay_roles),
        "feature_count": len(replay_names),
        "row_count": len(owner.heldout_row_ids),
        "heldout_labels_present": False,
    }
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "primary_heldout_transform_completed",
            "logical_scope_id": owner.scope_id,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
        }
    )
    primary_view = _logical_view(
        request=request,
        member=owner,
        fit_seal_registration=seal_registration,
        fit_seal=fit_seal,
        primary=True,
        heldout_text_sha256=_text_sha256(owner.heldout_row_ids, heldout_texts),
        prediction_artifact=prediction_artifact,
    )
    primary_path = logical_root / "000_primary.json"
    _write_new_json(primary_path, primary_view)
    logical_registrations.insert(
        0,
        {
            "logical_scope_id": owner.scope_id,
            "logical_purpose": owner.scope_kind,
            **_registration(primary_path, relative_to=root),
            "content_sha256": primary_view["content_sha256"],
        },
    )
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "primary_logical_view_published",
            "logical_scope_id": owner.scope_id,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
        }
    )
    if (Path(service.cache_dir) / cache_key).exists():
        raise RuntimeError(
            "held-out transform recreated an executable neural-query checkpoint"
        )
    if service.identity() != service_identity:
        raise RuntimeError("neural-query service scientific state changed during execution")

    terminal_body = {
        "schema_version": ROLE_NEUTRAL_NEURAL_QUERY_GROUP_EXECUTION_SCHEMA,
        "status": "complete",
        "group_request": request.as_dict(),
        "family": NEURAL_QUERY_MOMENTS,
        "fit_state_artifact_sha256": fit_state_sha256,
        "fit_only_family_seal": seal_registration,
        "logical_views": logical_registrations,
        "event_order": events,
        "fit_completed_before_registered_heldout_text_access": True,
        "fit_sealed_before_registered_heldout_text_access": True,
        "cumulative_views_published_without_heldout_rows_text_or_labels": True,
        "only_primary_view_admitted_heldout_text": True,
        "live_owned_fit_reused_for_primary_transform": True,
        "owned_safe_snapshot_replay_checked": True,
        "executable_checkpoint_absent_from_sealed_artifact": True,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
    }
    terminal = {
        **terminal_body,
        "content_sha256": _sha256_json(terminal_body),
    }
    _write_new_json(root / _TERMINAL_FILE, terminal)
    return validate_role_neutral_neural_query_group_execution(
        root=root,
        request=request,
    )


def _validate_fit_side(
    *,
    root: Path,
    request: RoleNeutralNeuralQueryPhysicalGroupRequest,
    expected_fit_texts: Sequence[str] | None = None,
    expected_treatment: Sequence[Any] | None = None,
    expected_outcome: Sequence[Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    fit_root = Path(root) / _FIT_STATE_DIRECTORY
    if fit_root.is_symlink() or not fit_root.is_dir():
        raise ValueError("neural-query fit state must be one real directory")
    if {path.name for path in fit_root.iterdir()} != {
        _OWNED_SNAPSHOT_DIRECTORY,
        _FIT_METADATA_FILE,
        _FIT_EVIDENCE_FILE,
    }:
        raise ValueError("neural-query fit state has an extra or missing member")
    metadata = _read_json(
        fit_root / _FIT_METADATA_FILE,
        label="neural-query fit metadata",
    )
    metadata_fields = {
        "schema_version",
        "group_request_content_sha256",
        "plan_scientific_content_sha256",
        "physical_owner_scope_id",
        "physical_owner_scope_sha256",
        "fit_row_ids",
        "fit_row_order_fingerprint",
        "fit_text_sha256",
        "fit_treatment_sha256",
        "fit_outcome_sha256",
        "scientific_configuration",
        "configuration_identity_sha256",
        "service_scientific_identity_sha256",
        "producer_identity_sha256",
        "owned_discovery",
        "evidence_artifact",
        "chunk_coverage",
        "evidence_coverage",
        "executable_scratch_cleanup",
        "fit_completed_before_registered_heldout_text_access",
        "registered_heldout_text_accessed",
        "registered_heldout_labels_accessed",
        "oracle_fields_accessed",
        "text_truncation_applied",
        "storage_format",
        "content_sha256",
    }
    metadata_body = {
        key: copy.deepcopy(value)
        for key, value in metadata.items()
        if key != "content_sha256"
    }
    owner = request.physical_owner
    if (
        set(metadata) != metadata_fields
        or metadata.get("schema_version")
        != ROLE_NEUTRAL_NEURAL_QUERY_FIT_STATE_SCHEMA
        or metadata.get("content_sha256") != _sha256_json(metadata_body)
        or metadata.get("group_request_content_sha256") != request.content_sha256
        or metadata.get("plan_scientific_content_sha256")
        != request.plan_scientific_content_sha256
        or metadata.get("physical_owner_scope_id") != owner.scope_id
        or metadata.get("physical_owner_scope_sha256")
        != owner.as_dict()["scope_sha256"]
        or tuple(metadata.get("fit_row_ids") or ()) != owner.fit_row_ids
        or metadata.get("fit_row_order_fingerprint")
        != _row_order_fingerprint(owner.fit_row_ids)
        or metadata.get("scientific_configuration")
        != request.scientific_configuration
        or metadata.get("configuration_identity_sha256")
        != request.scientific_configuration["content_sha256"]
        or metadata.get("service_scientific_identity_sha256")
        != _sha256_json(
            request.scientific_configuration["service_scientific_identity"]
        )
        or metadata.get("producer_identity_sha256") != _producer_identity()
        or metadata.get("fit_completed_before_registered_heldout_text_access")
        is not True
        or metadata.get("registered_heldout_text_accessed") is not False
        or metadata.get("registered_heldout_labels_accessed") is not False
        or metadata.get("oracle_fields_accessed") is not False
        or metadata.get("text_truncation_applied") is not False
        or metadata.get("storage_format")
        != "closed_json_and_per_array_npy_only"
    ):
        raise ValueError("neural-query fit metadata is open, stale, or inconsistent")
    for key in (
        "fit_text_sha256",
        "fit_treatment_sha256",
        "fit_outcome_sha256",
        "configuration_identity_sha256",
        "service_scientific_identity_sha256",
        "producer_identity_sha256",
    ):
        _require_sha256(metadata.get(key), label=f"fit metadata {key}")
    if expected_fit_texts is not None:
        texts = tuple(expected_fit_texts)
        if (
            len(texts) != len(owner.fit_row_ids)
            or any(not isinstance(text, str) for text in texts)
            or metadata["fit_text_sha256"]
            != _text_sha256(owner.fit_row_ids, texts)
        ):
            raise ValueError("neural-query fit text binding differs")
    if expected_treatment is not None and metadata["fit_treatment_sha256"] != (
        _float_hex_sha256(expected_treatment, label="expected fit treatment")
    ):
        raise ValueError("neural-query fit treatment binding differs")
    if expected_outcome is not None and metadata["fit_outcome_sha256"] != (
        _float_hex_sha256(expected_outcome, label="expected fit outcome")
    ):
        raise ValueError("neural-query fit outcome binding differs")

    owned = metadata.get("owned_discovery")
    if not isinstance(owned, dict) or set(owned) != {
        "relative_path",
        "cache_key",
        "snapshot_content_sha256",
        "snapshot_tree_sha256",
        "owned_discovery_content_sha256",
        "array_order",
        "array_inventory",
        "query_count_by_bank",
        "fit_row_count",
        "executable_serialization_present",
    }:
        raise ValueError("neural-query owned snapshot registration is not closed")
    if (
        owned.get("relative_path") != _OWNED_SNAPSHOT_DIRECTORY
        or owned.get("fit_row_count") != len(owner.fit_row_ids)
        or owned.get("executable_serialization_present") is not False
    ):
        raise ValueError("neural-query owned snapshot registration is invalid")
    snapshot_root = fit_root / _OWNED_SNAPSHOT_DIRECTORY
    snapshot = validate_owned_discovery_snapshot(
        snapshot_root,
        expected_cache_key=owned.get("cache_key"),
        expected_service_identity_sha256=metadata[
            "service_scientific_identity_sha256"
        ],
    )
    if (
        snapshot.get("content_sha256") != owned.get("snapshot_content_sha256")
        or snapshot.get("owned_discovery_content_sha256")
        != owned.get("owned_discovery_content_sha256")
        or snapshot.get("array_order") != owned.get("array_order")
        or snapshot.get("array_inventory") != owned.get("array_inventory")
        or snapshot.get("query_count_by_bank") != owned.get("query_count_by_bank")
        or _tree_descriptor(snapshot_root)["content_sha256"]
        != owned.get("snapshot_tree_sha256")
    ):
        raise RuntimeError("neural-query owned snapshot differs from fit metadata")

    evidence = _read_json(
        fit_root / _FIT_EVIDENCE_FILE,
        label="neural-query fit evidence",
    )
    if (
        set(evidence)
        != {"schema_version", "family", "architecture_evidence"}
        or evidence.get("schema_version")
        != NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION
        or evidence.get("family") != NEURAL_QUERY_MOMENTS
        or not isinstance(evidence.get("architecture_evidence"), list)
        or not evidence["architecture_evidence"]
    ):
        raise ValueError("neural-query fit evidence is empty or malformed")
    evidence_registration = metadata.get("evidence_artifact")
    expected_evidence_registration = {
        **_registration(fit_root / _FIT_EVIDENCE_FILE, relative_to=fit_root),
        "content_sha256": _sha256_json(evidence),
    }
    if evidence_registration != expected_evidence_registration:
        raise RuntimeError("neural-query evidence registration changed")

    chunk_coverage = metadata.get("chunk_coverage")
    evidence_coverage = metadata.get("evidence_coverage")
    cleanup = metadata.get("executable_scratch_cleanup")
    if not all(
        isinstance(value, dict)
        and value.get("content_sha256")
        == _sha256_json(
            {
                key: child
                for key, child in value.items()
                if key != "content_sha256"
            }
        )
        for value in (chunk_coverage, evidence_coverage, cleanup)
    ):
        raise ValueError("neural-query coverage or scratch proof is not self-authenticating")
    if (
        chunk_coverage.get("schema_version")
        != ROLE_NEUTRAL_NEURAL_QUERY_COVERAGE_SCHEMA
        or chunk_coverage.get("fit_row_count") != len(owner.fit_row_ids)
        or chunk_coverage.get("fit_row_order_fingerprint")
        != _row_order_fingerprint(owner.fit_row_ids)
        or chunk_coverage.get("embedding_chunk_cap_nonbinding") is not True
        or chunk_coverage.get("embedding_tokenizer_truncation_allowed")
        is not False
        or chunk_coverage.get("evidence_chunk_capacity_nonbinding") is not True
        or chunk_coverage.get("patient_evidence_capacity_nonbinding") is not True
        or chunk_coverage.get("excerpt_capacity_nonbinding") is not True
        or chunk_coverage.get("capacity_policy")
        != FAIL_CLOSED_EVIDENCE_CAPACITY_POLICY
        or chunk_coverage.get("text_truncation_applied") is not False
        or evidence_coverage.get("expected_query_count_by_bank")
        != owned["query_count_by_bank"]
        or evidence_coverage.get("observed_query_count_by_bank")
        != owned["query_count_by_bank"]
        or evidence_coverage.get("all_configured_queries_retained") is not True
        or evidence_coverage.get("term_count_capacity_nonbinding") is not True
        or evidence_coverage.get("term_transport_capacity_nonbinding") is not True
        or evidence_coverage.get(
            "row_level_chunks_removed_after_complete_fit_side_contrast"
        )
        is not True
        or cleanup.get("cache_key") != owned["cache_key"]
        or cleanup.get("owned_cache_entry_absent") is not True
        or cleanup.get("remaining_executable_checkpoint_count") != 0
        or cleanup.get("sealed_artifact_received_executable_state") is not False
    ):
        raise ValueError("neural-query complete-coverage or scratch proof is invalid")
    query_rows = evidence["architecture_evidence"]
    validated_rows, expected_evidence_coverage = _validate_safe_evidence(
        query_rows,
        request=request,
    )
    if validated_rows != query_rows or expected_evidence_coverage != evidence_coverage:
        raise RuntimeError("neural-query evidence coverage replay changed")

    for path in fit_root.rglob("*"):
        if path.is_file() and path.suffix.casefold() in _EXECUTABLE_SUFFIXES:
            raise ValueError("sealed neural-query fit state contains executable serialization")
    fit_state_sha256 = _tree_descriptor(fit_root)["content_sha256"]

    seal_path = Path(root) / _FIT_SEAL_FILE
    seal = _read_json(seal_path, label="neural-query fit-only seal")
    expected_seal = build_role_neutral_fit_only_family_seal(
        plan=request.authority_plan,
        physical_owner_scope_id=owner.scope_id,
        family=NEURAL_QUERY_MOMENTS,
        evidence_payload=evidence,
        producer_identity_sha256=metadata["producer_identity_sha256"],
        configuration_identity_sha256=metadata[
            "configuration_identity_sha256"
        ],
        fit_state_artifact_sha256=fit_state_sha256,
    )
    if seal != expected_seal:
        raise ValueError("neural-query fit-only family seal changed")
    return metadata, seal, fit_state_sha256


def _expected_feature_metadata(
    request: RoleNeutralNeuralQueryPhysicalGroupRequest,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    query = request.scientific_configuration["query_config"]
    names: list[str] = []
    kinds: list[str] = []
    roles: list[str] = []
    for bank in _BANKS:
        count = int(query[f"{bank}_query_count"])
        bank_names = (
            f"neural_query_{bank}_signed_mean",
            f"neural_query_{bank}_absolute_max",
            *(
                f"neural_query_{bank}_signed_order_{rank:02d}"
                for rank in range(1, count + 1)
            ),
        )
        names.extend(bank_names)
        kinds.extend([f"neural_query_{bank}_moments"] * len(bank_names))
        roles.extend([_ROLE_BY_BANK[bank]] * len(bank_names))
    return tuple(names), tuple(kinds), tuple(roles)


def validate_role_neutral_neural_query_group_execution(
    *,
    root: Path | str,
    request: RoleNeutralNeuralQueryPhysicalGroupRequest,
) -> Mapping[str, Any]:
    """Freshly authenticate the complete safe neural-query group artifact."""

    if not isinstance(request, RoleNeutralNeuralQueryPhysicalGroupRequest):
        raise TypeError("validation requires its typed neural-query request")
    request.as_dict()
    artifact = Path(root)
    if artifact.is_symlink() or not artifact.is_dir():
        raise ValueError("neural-query group artifact must be one real directory")
    if {path.name for path in artifact.iterdir()} != {
        _FIT_STATE_DIRECTORY,
        _FIT_SEAL_FILE,
        _LOGICAL_VIEW_DIRECTORY,
        _TERMINAL_FILE,
    }:
        raise ValueError("neural-query group artifact has an extra or missing root member")
    terminal = _read_json(
        artifact / _TERMINAL_FILE,
        label="neural-query execution manifest",
    )
    terminal_body = {
        key: copy.deepcopy(value)
        for key, value in terminal.items()
        if key != "content_sha256"
    }
    terminal_fields = {
        "schema_version",
        "status",
        "group_request",
        "family",
        "fit_state_artifact_sha256",
        "fit_only_family_seal",
        "logical_views",
        "event_order",
        "fit_completed_before_registered_heldout_text_access",
        "fit_sealed_before_registered_heldout_text_access",
        "cumulative_views_published_without_heldout_rows_text_or_labels",
        "only_primary_view_admitted_heldout_text",
        "live_owned_fit_reused_for_primary_transform",
        "owned_safe_snapshot_replay_checked",
        "executable_checkpoint_absent_from_sealed_artifact",
        "registered_heldout_labels_accessed",
        "oracle_fields_accessed",
        "text_truncation_applied",
        "content_sha256",
    }
    if (
        set(terminal) != terminal_fields
        or terminal.get("schema_version")
        != ROLE_NEUTRAL_NEURAL_QUERY_GROUP_EXECUTION_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("content_sha256") != _sha256_json(terminal_body)
        or terminal.get("group_request") != request.as_dict()
        or terminal.get("family") != NEURAL_QUERY_MOMENTS
        or terminal.get("fit_completed_before_registered_heldout_text_access")
        is not True
        or terminal.get("fit_sealed_before_registered_heldout_text_access")
        is not True
        or terminal.get(
            "cumulative_views_published_without_heldout_rows_text_or_labels"
        )
        is not True
        or terminal.get("only_primary_view_admitted_heldout_text") is not True
        or terminal.get("live_owned_fit_reused_for_primary_transform") is not True
        or terminal.get("owned_safe_snapshot_replay_checked") is not True
        or terminal.get("executable_checkpoint_absent_from_sealed_artifact")
        is not True
        or terminal.get("registered_heldout_labels_accessed") is not False
        or terminal.get("oracle_fields_accessed") is not False
        or terminal.get("text_truncation_applied") is not False
    ):
        raise ValueError("neural-query execution manifest is open or inconsistent")

    metadata, seal, fit_state_sha256 = _validate_fit_side(
        root=artifact,
        request=request,
    )
    if terminal.get("fit_state_artifact_sha256") != fit_state_sha256:
        raise RuntimeError("neural-query terminal fit-state hash changed")
    seal_registration = {
        **_registration(artifact / _FIT_SEAL_FILE, relative_to=artifact),
        "content_sha256": seal["content_sha256"],
    }
    if terminal.get("fit_only_family_seal") != seal_registration:
        raise RuntimeError("neural-query terminal seal registration changed")

    logical_root = artifact / _LOGICAL_VIEW_DIRECTORY
    if logical_root.is_symlink() or not logical_root.is_dir():
        raise ValueError("neural-query logical views must be one real directory")
    registrations = terminal.get("logical_views")
    if (
        not isinstance(registrations, list)
        or len(registrations) != len(request.logical_members)
        or [row.get("logical_scope_id") for row in registrations]
        != [member.scope_id for member in request.logical_members]
    ):
        raise ValueError("neural-query terminal logical-view registration is incomplete")
    expected_logical_entries = {
        "000_primary.json",
        "primary_predictions",
        *(
            f"{index:03d}_cumulative.json"
            for index in range(1, len(request.logical_members))
        ),
    }
    if {path.name for path in logical_root.iterdir()} != expected_logical_entries:
        raise ValueError("neural-query logical view tree has an extra or missing entry")

    prediction_registration: dict[str, Any] | None = None
    primary_view: dict[str, Any] | None = None
    for index, (member, registration) in enumerate(
        zip(request.logical_members, registrations, strict=True)
    ):
        if not isinstance(registration, dict):
            raise ValueError("neural-query logical registration must be one object")
        filename = (
            "000_primary.json"
            if index == 0
            else f"{index:03d}_cumulative.json"
        )
        path = logical_root / filename
        expected_registration = {
            "logical_scope_id": member.scope_id,
            "logical_purpose": member.scope_kind,
            **_registration(path, relative_to=artifact),
        }
        view = _read_json(path, label=f"neural-query logical view {index}")
        expected_registration["content_sha256"] = view.get("content_sha256")
        if registration != expected_registration:
            raise RuntimeError("neural-query logical-view registration changed")
        view_body = {
            key: copy.deepcopy(value)
            for key, value in view.items()
            if key != "content_sha256"
        }
        if (
            set(view)
            != {
                "schema_version",
                "group_request_content_sha256",
                "logical_scope_id",
                "logical_scope_sha256",
                "logical_purpose",
                "physical_owner_scope_id",
                "family",
                "fit_only_family_seal_sha256",
                "fit_only_family_seal_content_sha256",
                "view_input_policy",
                "logical_transform_performed",
                "logical_heldout_row_ids",
                "logical_heldout_text_sha256",
                "prediction_artifact",
                "registered_heldout_text_accessed",
                "registered_heldout_labels_accessed",
                "reuses_live_physical_fit",
                "model_state_reloaded_for_primary_transform",
                "owned_safe_snapshot_replay_checked",
                "content_sha256",
            }
            or view.get("content_sha256") != _sha256_json(view_body)
            or view.get("schema_version")
            != ROLE_NEUTRAL_NEURAL_QUERY_LOGICAL_VIEW_SCHEMA
            or view.get("group_request_content_sha256") != request.content_sha256
            or view.get("logical_scope_id") != member.scope_id
            or view.get("logical_scope_sha256")
            != member.as_dict()["scope_sha256"]
            or view.get("logical_purpose") != member.scope_kind
            or view.get("physical_owner_scope_id")
            != request.physical_owner.scope_id
            or view.get("family") != NEURAL_QUERY_MOMENTS
            or view.get("fit_only_family_seal_sha256")
            != seal_registration["sha256"]
            or view.get("fit_only_family_seal_content_sha256")
            != seal["content_sha256"]
            or view.get("registered_heldout_labels_accessed") is not False
            or view.get("reuses_live_physical_fit") is not True
            or view.get("model_state_reloaded_for_primary_transform") is not False
        ):
            raise ValueError("neural-query logical view is open or inconsistent")
        if index == 0:
            if (
                view.get("view_input_policy")
                != "registered_heldout_row_ids_and_text_no_labels_v1"
                or view.get("logical_transform_performed") is not True
                or tuple(view.get("logical_heldout_row_ids") or ())
                != member.heldout_row_ids
                or _SHA256.fullmatch(
                    str(view.get("logical_heldout_text_sha256") or "")
                )
                is None
                or not isinstance(view.get("prediction_artifact"), dict)
                or view.get("registered_heldout_text_accessed") is not True
                or view.get("owned_safe_snapshot_replay_checked") is not True
            ):
                raise ValueError("primary neural-query logical view is incomplete")
            prediction_registration = view["prediction_artifact"]
            primary_view = view
        elif (
            view.get("view_input_policy")
            != "fit_only_reference_no_heldout_rows_text_or_labels_v1"
            or view.get("logical_transform_performed") is not False
            or view.get("logical_heldout_row_ids") is not None
            or view.get("logical_heldout_text_sha256") is not None
            or view.get("prediction_artifact") is not None
            or view.get("registered_heldout_text_accessed") is not False
            or view.get("owned_safe_snapshot_replay_checked") is not False
        ):
            raise ValueError("cumulative neural-query view admitted held-out state")

    if prediction_registration is None or primary_view is None:
        raise RuntimeError("neural-query primary prediction registration is absent")
    expected_prediction_fields = {
        "relative_path",
        "array_order",
        "array_inventory",
        "index_sha256",
        "arrays_content_sha256",
        "feature_names",
        "feature_kinds",
        "feature_roles",
        "feature_count",
        "row_count",
        "heldout_labels_present",
    }
    prediction_root = artifact / str(prediction_registration.get("relative_path"))
    if (
        set(prediction_registration) != expected_prediction_fields
        or prediction_root != logical_root / "primary_predictions"
        or prediction_registration.get("array_order")
        != ["gate_row_ids", "feature_values"]
        or prediction_registration.get("heldout_labels_present") is not False
    ):
        raise ValueError("neural-query prediction registration is open or invalid")
    descriptor, arrays = validate_npy_array_set(
        prediction_root,
        expected_order=("gate_row_ids", "feature_values"),
        expected_inventory=prediction_registration["array_inventory"],
    )
    if (
        descriptor["index_sha256"] != prediction_registration["index_sha256"]
        or descriptor["content_sha256"]
        != prediction_registration["arrays_content_sha256"]
    ):
        raise RuntimeError("neural-query prediction array set changed")
    names, kinds, roles = _expected_feature_metadata(request)
    gate_rows = np.asarray(arrays["gate_row_ids"])
    feature_values = np.asarray(arrays["feature_values"])
    if (
        gate_rows.dtype != np.dtype(np.int64)
        or gate_rows.ndim != 1
        or tuple(map(int, gate_rows.tolist()))
        != request.physical_owner.heldout_row_ids
        or feature_values.dtype != np.dtype(np.float64)
        or feature_values.shape != (len(gate_rows), len(names))
        or not np.isfinite(feature_values).all()
        or prediction_registration.get("feature_names") != list(names)
        or prediction_registration.get("feature_kinds") != list(kinds)
        or prediction_registration.get("feature_roles") != list(roles)
        or prediction_registration.get("feature_count") != len(names)
        or prediction_registration.get("row_count") != len(gate_rows)
    ):
        raise ValueError("neural-query prediction values or feature schema changed")

    events = terminal.get("event_order")
    expected_event_names = [
        "fit_completed",
        "owned_executable_checkpoint_removed",
        "fit_family_artifact_sealed",
        *(
            "cumulative_fit_only_view_published"
            for _member in request.logical_members[1:]
        ),
        "primary_heldout_text_opened",
        "primary_heldout_transform_completed",
        "primary_logical_view_published",
    ]
    if (
        not isinstance(events, list)
        or [event.get("event") for event in events] != expected_event_names
        or [event.get("sequence") for event in events]
        != list(range(1, len(events) + 1))
        or any(
            event.get("registered_heldout_labels_accessed") is not False
            for event in events
        )
    ):
        raise ValueError("neural-query event order is incomplete or reordered")
    open_index = expected_event_names.index("primary_heldout_text_opened")
    if any(
        event.get("registered_heldout_text_accessed") is not False
        for event in events[:open_index]
    ) or any(
        event.get("registered_heldout_text_accessed") is not True
        for event in events[open_index:]
    ):
        raise ValueError("neural-query held-out text access preceded fit-side sealing")
    for path in artifact.rglob("*"):
        if path.is_file() and path.suffix.casefold() in _EXECUTABLE_SUFFIXES:
            raise ValueError("sealed neural-query artifact contains executable serialization")
    # Reopen the terminal once more after all descendants were authenticated.
    if _read_json(
        artifact / _TERMINAL_FILE,
        label="neural-query execution manifest",
    ) != terminal:
        raise RuntimeError("neural-query terminal changed during validation")
    del metadata, primary_view
    return copy.deepcopy(terminal)


def replay_role_neutral_neural_query_heldout_transform(
    *,
    root: Path | str,
    request: RoleNeutralNeuralQueryPhysicalGroupRequest,
    service: ContextFitNeuralQueryService,
    heldout_texts: Sequence[str],
) -> Mapping[str, Any]:
    """Replay primary held-out moments from safe state without executable bytes."""

    terminal = validate_role_neutral_neural_query_group_execution(
        root=root,
        request=request,
    )
    service_identity = _validate_service_against_request(service, request)
    owner = request.physical_owner
    texts = tuple(heldout_texts)
    if (
        len(texts) != len(owner.heldout_row_ids)
        or any(not isinstance(text, str) for text in texts)
    ):
        raise ValueError("held-out replay texts do not align to owner rows")
    primary_registration = terminal["logical_views"][0]
    primary_view = _read_json(
        Path(root) / primary_registration["relative_path"],
        label="neural-query primary replay view",
    )
    if primary_view["logical_heldout_text_sha256"] != _text_sha256(
        owner.heldout_row_ids,
        texts,
    ):
        raise ValueError("held-out replay text differs from the sealed transform")
    names, kinds, roles, values = _transform_owned_snapshot(
        service=service,
        snapshot_root=Path(root) / _FIT_STATE_DIRECTORY / _OWNED_SNAPSHOT_DIRECTORY,
        heldout_row_ids=owner.heldout_row_ids,
        heldout_texts=texts,
        expected_service_identity_sha256=_sha256_json(service_identity),
    )
    prediction = primary_view["prediction_artifact"]
    _descriptor, arrays = validate_npy_array_set(
        Path(root) / prediction["relative_path"],
        expected_order=("gate_row_ids", "feature_values"),
        expected_inventory=prediction["array_inventory"],
    )
    sealed_values = np.asarray(arrays["feature_values"], dtype=np.float64)
    if (
        list(names) != prediction["feature_names"]
        or list(kinds) != prediction["feature_kinds"]
        or list(roles) != prediction["feature_roles"]
        or not neural_float_arrays_within_tolerance(
            values,
            sealed_values,
            policy=request.scientific_configuration[
                "replay_comparison_policy"
            ],
            relative_tolerance=request.scientific_configuration[
                "replay_relative_tolerance"
            ],
            absolute_tolerance=request.scientific_configuration[
                "replay_absolute_tolerance"
            ],
        )
    ):
        raise RuntimeError(
            "fresh neural-query safe-state replay differs from sealed output "
            "beyond its declared tolerance"
        )
    return {
        "gate_row_ids": owner.heldout_row_ids,
        "feature_names": names,
        "feature_kinds": kinds,
        "feature_roles": roles,
        "feature_values": np.array(values, copy=True),
        "registered_heldout_labels_accessed": False,
        "executable_checkpoint_loaded": False,
    }


def replay_role_neutral_neural_query_exact_transform(
    *,
    root: Path | str,
    request: RoleNeutralNeuralQueryPhysicalGroupRequest,
    service: ContextFitNeuralQueryService,
    exact_heldout_texts: Sequence[str],
) -> Mapping[str, Any]:
    """Compatibility wrapper for the generalized held-out replay API."""

    return replay_role_neutral_neural_query_heldout_transform(
        root=root,
        request=request,
        service=service,
        heldout_texts=exact_heldout_texts,
    )


__all__ = [
    "COMPLETE_EMBEDDING_TEXT_POLICY",
    "EXACT_INNER_TRANSFORM_POLICY",
    "FAIL_CLOSED_EVIDENCE_CAPACITY_POLICY",
    "REGISTERED_HELDOUT_TRANSFORM_POLICY",
    "ROLE_NEUTRAL_NEURAL_QUERY_COVERAGE_SCHEMA",
    "ROLE_NEUTRAL_NEURAL_QUERY_FIT_STATE_SCHEMA",
    "ROLE_NEUTRAL_NEURAL_QUERY_GROUP_EXECUTION_SCHEMA",
    "ROLE_NEUTRAL_NEURAL_QUERY_GROUP_REQUEST_SCHEMA",
    "ROLE_NEUTRAL_NEURAL_QUERY_LOGICAL_VIEW_SCHEMA",
    "RoleNeutralNeuralQueryPhysicalGroupRequest",
    "execute_role_neutral_neural_query_physical_group",
    "replay_role_neutral_neural_query_exact_transform",
    "replay_role_neutral_neural_query_heldout_transform",
    "validate_role_neutral_neural_query_group_execution",
]
