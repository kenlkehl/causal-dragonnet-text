"""Typed execution seam for the role-neutral Stage 1 production path.

The six role-neutral architecture producers intentionally have different
scientific inputs.  This module does not hide those inputs in a weak
``**kwargs`` interface and does not synthesize model outputs.  Instead,
deployment code supplies one explicit factory for each producer.  A factory
binds the real producer request, cohort views, model/configuration objects,
and the matching fresh validator to the invocation it receives.

This coordinator owns the invariants shared by all deployments:

* physical owners are derived from :class:`Stage1ScopePlan` by content;
* each physical owner is claimed exactly once by the configured executor;
* each of the canonical six producers executes and authenticates exactly once
  per owner;
* resources, paths, completion order, and worker count are execution-only;
* every returned receipt is freshly rebound to the exact output tree; and
* publication occurs only after complete physical and logical coverage.

``ProductionStage1BundleBuilder.build()`` is deliberately not changed here.
The public function in this module is the integration point for a later
opt-in bundle phase once deployment-specific producer factories are wired.
"""

from __future__ import annotations

import copy
import concurrent.futures
import hashlib
import json
import math
import os
import stat
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import performance_telemetry as telemetry_module
from .portable_resource_scheduler import (
    ResourcePlan,
    assign_physical_fits,
)
from .production_stage1_legacy_scope_fragments import (
    durably_sync_legacy_stage1_tree,
)
from .production_stage1_role_neutral_coordinator import (
    ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION,
    ROLE_NEUTRAL_COORDINATION_MANIFEST,
    RoleNeutralComponentArtifactSource,
    publish_role_neutral_stage1_coordination_gate,
    validate_role_neutral_stage1_coordination_gate,
)
from .production_stage1_scope_scheduler import (
    Stage1ScopePlan,
    Stage1ScopeSpec,
)
from .neural_query_execution_topology import (
    NeuralQueryExecutionTopology,
)
from .neural_query_operational_controls import (
    RoleNeutralNeuralQueryOperationalControls,
)
from .role_neutral_htr_group_execution import (
    ROLE_NEUTRAL_HTR_OPERATIONAL_ATTESTATION_SCHEMA,
    RoleNeutralHTROperationalControls,
)
from .role_neutral_all_ten_binding import (
    AuthenticatedRoleNeutralComponentReceipt,
    EXPECTED_COMPONENT_FAMILIES,
    NATIVE_TO_PORTABLE_FAMILY,
    validate_authenticated_role_neutral_component_receipt,
)

ROLE_NEUTRAL_STAGE1_EXECUTION_SCHEMA = "production_role_neutral_stage1_execution_v1"
ROLE_NEUTRAL_STAGE1_EXECUTION_ATTESTATION_SCHEMA = (
    "production_role_neutral_stage1_execution_attestation_v3"
)

ROLE_NEUTRAL_COMPONENT_DIRECTORY = "components"
ROLE_NEUTRAL_COORDINATION_DIRECTORY = "coordination_gate"
ROLE_NEUTRAL_EXECUTION_ATTESTATION = "execution_attestation.json"
ROLE_NEUTRAL_EXECUTION_MANIFEST = "execution_manifest.json"
ROLE_NEUTRAL_COMPUTE_CANARY_ATTESTATION = "compute_canary_attestation.json"
ROLE_NEUTRAL_COMPUTE_CANARY_SCHEMA = (
    "production_role_neutral_stage1_compute_canary_v1"
)
ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_GATE_SCHEMA = (
    "production_role_neutral_first_owner_validation_gate_v1"
)
ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_POLICY_SCHEMA = (
    "production_role_neutral_first_owner_validation_policy_v1"
)
ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_GATE_SUFFIX = (
    "first_owner_validation_gate.json"
)
_MATCHED_PAIR_OPERATIONAL_ATTESTATION_SCHEMA = (
    "production_role_neutral_matched_pair_operational_attestation_v1"
)
_TFIDF_NUISANCE_EXECUTION_ATTESTATION_SCHEMA = (
    "tfidf_joint_nuisance_fold_execution_attestation_v1"
)

EARLIEST_CANONICAL_OWNER_CANARY_SELECTION = (
    "earliest_canonical_physical_owner_v1"
)
DISTINCT_RESOURCE_CANARY_REPLICA_POLICY = (
    "distinct_compatible_resource_when_available_else_same_v1"
)

_HEX = frozenset("0123456789abcdef")

ROLE_NEUTRAL_COMPONENT_EXECUTION_INTERVAL_SCHEMA = (
    "production_role_neutral_component_execution_interval_v1"
)
ROLE_NEUTRAL_COMPONENT_EXECUTION_CLOCK_DOMAIN = (
    "python_monotonic_ns_systemwide_v1"
)

# These labels describe directly timed architecture-phase execution envelopes,
# not CUDA-kernel occupancy.  The accelerator-associated components contain
# host-side orchestration as part of their measured envelope, so downstream
# analysis is intentionally descriptive and may not claim a throughput
# speedup from interval overlap.
_ACCELERATOR_ASSOCIATED_COMPONENTS = frozenset(
    {"htr", "matched_pair", "neural_query"}
)
_ROLE_NEUTRAL_COMPONENT_EXECUTION_INTERVAL_FIELDS = frozenset(
    {
        "schema_version",
        "physical_owner_scope_id",
        "component",
        "lane_kind",
        "resource_ids",
        "clock_domain_id",
        "started_monotonic_ns",
        "finished_monotonic_ns",
        "status",
        "timestamps_measured_directly",
        "interval_semantics",
    }
)
_ROLE_NEUTRAL_COMPONENT_EXECUTION_INTERVAL_SEMANTICS = (
    "architecture_phase_execution_envelope_not_kernel_occupancy_v1"
)
_ROLE_NEUTRAL_COMPONENT_EXECUTION_REPORT_SEMANTICS = (
    "direct_monotonic_architecture_phase_envelopes_not_kernel_occupancy_v1"
)


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
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _private_file_identity(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"registered file is not regular data: {path}")
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_nlink) != 1:
        raise ValueError(f"registered file is not private data: {path}")
    payload = path.read_bytes()
    after = os.lstat(path)
    if (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    ) != (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    ) or len(
        payload
    ) != int(
        after.st_size
    ):
        raise RuntimeError(f"registered file changed while reading: {path}")
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _canonical_fresh_root(value: Path | str) -> Path:
    root = Path(value)
    if not root.is_absolute():
        raise ValueError("role-neutral execution root must be absolute")
    if root.exists() or root.is_symlink():
        raise FileExistsError("role-neutral execution root must be fresh")
    parent = root.parent
    if parent.is_symlink() or parent.resolve(strict=True) != parent or not parent.is_dir():
        raise ValueError("role-neutral execution parent must be canonical")
    return root


def _owner_group(
    plan: Stage1ScopePlan,
    owner_scope_id: str,
) -> tuple[Stage1ScopeSpec, tuple[Stage1ScopeSpec, ...]]:
    matches = tuple(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if owner.scope_id == owner_scope_id
    )
    if len(matches) != 1:
        raise ValueError("execution task does not name one physical owner")
    return matches[0]


@dataclass(frozen=True)
class RoleNeutralComponentInvocation:
    """One deployment-bound producer call.

    ``output_root``, ``resource``, and ``owner_cpu_budget`` are operational
    capabilities.  They are intentionally absent from
    :meth:`scientific_payload`.
    """

    plan: Stage1ScopePlan
    physical_owner: Stage1ScopeSpec
    logical_members: tuple[Stage1ScopeSpec, ...]
    component: str
    output_root: Path
    resource: str
    neural_query_execution_topology: (
        NeuralQueryExecutionTopology | None
    ) = None
    htr_operational_controls: RoleNeutralHTROperationalControls | None = None
    neural_query_operational_controls: (
        RoleNeutralNeuralQueryOperationalControls | None
    ) = None
    htr_fold_devices: tuple[str, ...] = ()
    owner_cpu_budget: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, Stage1ScopePlan):
            raise TypeError("component invocation requires a Stage1ScopePlan")
        owner, members = _owner_group(
            self.plan,
            self.physical_owner.scope_id,
        )
        if self.physical_owner != owner or self.logical_members != members:
            raise ValueError("component invocation changed its physical group")
        if self.component not in EXPECTED_COMPONENT_FAMILIES:
            raise ValueError("component invocation names an unknown producer")
        if not self.output_root.is_absolute() or not str(self.resource).strip():
            raise ValueError("component invocation lacks operational capabilities")
        topology = self.neural_query_execution_topology
        if topology is None:
            topology = NeuralQueryExecutionTopology.single(self.resource)
            object.__setattr__(
                self,
                "neural_query_execution_topology",
                topology,
            )
        if not isinstance(topology, NeuralQueryExecutionTopology):
            raise TypeError(
                "component invocation requires a typed neural-query "
                "execution topology"
            )
        if (
            topology.primary_device
            != str(self.resource)
        ):
            raise ValueError(
                "component invocation topology must begin with its assigned "
                "primary resource"
            )
        if self.htr_operational_controls is not None and not isinstance(
            self.htr_operational_controls,
            RoleNeutralHTROperationalControls,
        ):
            raise TypeError(
                "component invocation HTR controls must use the typed "
                "deployment-only contract"
            )
        if (
            self.neural_query_operational_controls is not None
            and not isinstance(
                self.neural_query_operational_controls,
                RoleNeutralNeuralQueryOperationalControls,
            )
        ):
            raise TypeError(
                "component invocation neural-query controls must use the "
                "typed deployment-only contract"
            )
        htr_fold_devices = tuple(str(value) for value in self.htr_fold_devices)
        if not htr_fold_devices:
            htr_fold_devices = (str(self.resource),)
            object.__setattr__(self, "htr_fold_devices", htr_fold_devices)
        htr_topology = NeuralQueryExecutionTopology(
            devices=htr_fold_devices,
        )
        if self.resource not in htr_topology.devices:
            raise ValueError(
                "component invocation HTR fold resources omit its primary "
                "owner resource"
            )
        if (
            self.owner_cpu_budget is not None
            and (
                isinstance(self.owner_cpu_budget, bool)
                or int(self.owner_cpu_budget) < 1
            )
        ):
            raise ValueError(
                "component invocation owner CPU budget must be positive"
            )
        if self.neural_query_operational_controls is not None:
            if self.owner_cpu_budget is None:
                raise ValueError(
                    "component invocation neural-query controls require an "
                    "owner CPU budget"
                )
            self.neural_query_operational_controls.bind_task_resources(
                devices=topology.devices,
                owner_cpu_budget=self.owner_cpu_budget,
            )

    def scientific_payload(self) -> dict[str, Any]:
        body = {
            "schema_version": ("production_role_neutral_component_invocation_scientific_v1"),
            "plan_scientific_content_sha256": (self.plan.scientific_content_sha256),
            "physical_owner_scope_id": self.physical_owner.scope_id,
            "logical_scope_ids": [member.scope_id for member in self.logical_members],
            "canonical_fit_row_ids": list(self.physical_owner.fit_row_ids),
            "canonical_group_seed": int(self.physical_owner.scope_seed),
            "component": self.component,
            "native_families": list(EXPECTED_COMPONENT_FAMILIES[self.component]),
            "output_locator_included": False,
            "resource_assignment_included": False,
            "neural_query_device_topology_included": False,
            "htr_operational_controls_included": False,
            "neural_query_operational_controls_included": False,
            "htr_fold_devices_included": False,
        }
        return {**body, "content_sha256": _sha256_json(body)}


@dataclass(frozen=True)
class BoundRoleNeutralComponentProducer:
    """A real producer call and the exact fresh validator paired with it."""

    execute: Callable[[], Any]
    authenticate: Callable[[], AuthenticatedRoleNeutralComponentReceipt]

    def __post_init__(self) -> None:
        if not callable(self.execute) or not callable(self.authenticate):
            raise TypeError("bound role-neutral producer requires execute/authenticate callables")


@dataclass(frozen=True)
class RoleNeutralOperationalComponentReport:
    """Execution-only component report excluded from scientific receipts."""

    component: str
    attestation: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.component not in EXPECTED_COMPONENT_FAMILIES:
            raise ValueError("operational component report names another producer")
        value = copy.deepcopy(dict(self.attestation))
        content_sha256 = value.get("content_sha256")
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            not isinstance(content_sha256, str)
            or _sha256_json(body) != content_sha256
        ):
            raise ValueError(
                "operational component report has an invalid content identity"
            )
        object.__setattr__(self, "attestation", value)


RoleNeutralProducerFactory = Callable[
    [RoleNeutralComponentInvocation],
    BoundRoleNeutralComponentProducer,
]


@dataclass(frozen=True)
class RoleNeutralProducerFactories:
    """Closed, explicit six-producer deployment interface."""

    bow: RoleNeutralProducerFactory
    htr: RoleNeutralProducerFactory
    matched_pair: RoleNeutralProducerFactory
    embeddings: RoleNeutralProducerFactory
    tfidf: RoleNeutralProducerFactory
    neural_query: RoleNeutralProducerFactory

    def as_mapping(self) -> Mapping[str, RoleNeutralProducerFactory]:
        values = {
            "bow": self.bow,
            "htr": self.htr,
            "matched_pair": self.matched_pair,
            "embeddings": self.embeddings,
            "tfidf": self.tfidf,
            "neural_query": self.neural_query,
        }
        if tuple(values) != tuple(EXPECTED_COMPONENT_FAMILIES):
            raise RuntimeError("producer factory order differs from all-ten contract")
        if any(not callable(value) for value in values.values()):
            raise TypeError("every role-neutral producer factory must be callable")
        return values


@dataclass(frozen=True)
class RoleNeutralPhysicalOwnerTask:
    plan: Stage1ScopePlan
    physical_owner: Stage1ScopeSpec
    logical_members: tuple[Stage1ScopeSpec, ...]
    component_parent: Path
    resource: str
    neural_query_execution_topology: (
        NeuralQueryExecutionTopology | None
    ) = None
    htr_operational_controls: RoleNeutralHTROperationalControls | None = None
    neural_query_operational_controls: (
        RoleNeutralNeuralQueryOperationalControls | None
    ) = None
    htr_fold_devices: tuple[str, ...] = ()
    owner_cpu_budget: int | None = None

    def __post_init__(self) -> None:
        owner, members = _owner_group(
            self.plan,
            self.physical_owner.scope_id,
        )
        if owner != self.physical_owner or members != self.logical_members:
            raise ValueError("physical-owner task changed its equivalence group")
        if not self.component_parent.is_absolute():
            raise ValueError("physical-owner task root must be absolute")
        if not str(self.resource).strip():
            raise ValueError("physical-owner task resource cannot be empty")
        topology = self.neural_query_execution_topology
        if topology is None:
            topology = NeuralQueryExecutionTopology.single(self.resource)
            object.__setattr__(
                self,
                "neural_query_execution_topology",
                topology,
            )
        if not isinstance(topology, NeuralQueryExecutionTopology):
            raise TypeError(
                "physical-owner task requires a typed neural-query "
                "execution topology"
            )
        if topology.primary_device != str(self.resource):
            raise ValueError(
                "physical-owner neural-query topology must begin with its "
                "assigned primary resource"
            )
        if self.htr_operational_controls is not None and not isinstance(
            self.htr_operational_controls,
            RoleNeutralHTROperationalControls,
        ):
            raise TypeError(
                "physical-owner task HTR controls must use the typed "
                "deployment-only contract"
            )
        if (
            self.neural_query_operational_controls is not None
            and not isinstance(
                self.neural_query_operational_controls,
                RoleNeutralNeuralQueryOperationalControls,
            )
        ):
            raise TypeError(
                "physical-owner task neural-query controls must use the "
                "typed deployment-only contract"
            )
        htr_fold_devices = tuple(str(value) for value in self.htr_fold_devices)
        if not htr_fold_devices:
            htr_fold_devices = (str(self.resource),)
            object.__setattr__(self, "htr_fold_devices", htr_fold_devices)
        htr_topology = NeuralQueryExecutionTopology(
            devices=htr_fold_devices,
        )
        if self.resource not in htr_topology.devices:
            raise ValueError(
                "physical-owner HTR fold resources omit its primary resource"
            )
        if (
            self.owner_cpu_budget is not None
            and (
                isinstance(self.owner_cpu_budget, bool)
                or int(self.owner_cpu_budget) < 1
            )
        ):
            raise ValueError(
                "physical-owner task CPU budget must be positive"
            )
        if self.neural_query_operational_controls is not None:
            if self.owner_cpu_budget is None:
                raise ValueError(
                    "physical-owner neural-query controls require an owner "
                    "CPU budget"
                )
            self.neural_query_operational_controls.bind_task_resources(
                devices=topology.devices,
                owner_cpu_budget=self.owner_cpu_budget,
            )


@dataclass(frozen=True)
class RoleNeutralPhysicalOwnerResult:
    physical_owner_scope_id: str
    sources: tuple[RoleNeutralComponentArtifactSource, ...]
    component_execution_order: tuple[str, ...]
    resource: str
    execution_telemetry: Mapping[str, Any] | None = None


class RoleNeutralPhysicalOwnerExecutor(Protocol):
    """Deployment executor boundary.

    Implementations may use processes, persistent workers, or a serial lane.
    They must return only after every submitted task has reached a terminal
    result or one task has raised.
    """

    def execute(
        self,
        *,
        tasks: Sequence[RoleNeutralPhysicalOwnerTask],
        worker: Callable[
            [RoleNeutralPhysicalOwnerTask],
            RoleNeutralPhysicalOwnerResult,
        ],
        max_workers: int,
        cpu_budget: int,
    ) -> Sequence[RoleNeutralPhysicalOwnerResult]: ...


@dataclass(frozen=True)
class LocalThreadRoleNeutralPhysicalOwnerExecutor:
    """Bounded single-node executor with completion-order-neutral results.

    Concurrency is supplied by :class:`RoleNeutralStage1ExecutionPolicy`; this
    class has no hidden worker or device defaults.  Producer factories remain
    responsible for binding each task's assigned CPU/CUDA resource.
    """

    thread_name_prefix: str = "oci-stage1-owner"

    def __post_init__(self) -> None:
        if not isinstance(self.thread_name_prefix, str) or not self.thread_name_prefix.strip():
            raise ValueError("thread_name_prefix must be nonempty")

    def execute(
        self,
        *,
        tasks: Sequence[RoleNeutralPhysicalOwnerTask],
        worker: Callable[
            [RoleNeutralPhysicalOwnerTask],
            RoleNeutralPhysicalOwnerResult,
        ],
        max_workers: int,
        cpu_budget: int,
    ) -> Sequence[RoleNeutralPhysicalOwnerResult]:
        rows = tuple(tasks)
        workers = int(max_workers)
        budget = int(cpu_budget)
        if not rows:
            return ()
        if workers < 1 or budget < 1 or workers > budget:
            raise ValueError(
                "local executor requires 1 <= max_workers <= cpu_budget"
            )
        if len({task.physical_owner.scope_id for task in rows}) != len(rows):
            raise ValueError("local executor tasks contain duplicate physical owners")
        if any(
            (
                task.neural_query_execution_topology is not None
                and task.neural_query_execution_topology.spans_multiple_devices
            )
            or len(task.htr_fold_devices) > 1
            for task in rows
        ):
            raise RuntimeError(
                "the in-process thread executor cannot atomically reserve a "
                "multi-device neural execution context"
            )
        results: list[RoleNeutralPhysicalOwnerResult] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, len(rows)),
            thread_name_prefix=self.thread_name_prefix,
        ) as pool:
            futures = {
                pool.submit(worker, task): task.physical_owner.scope_id
                for task in rows
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        return tuple(results)


@dataclass(frozen=True)
class RoleNeutralFirstOwnerValidationPolicy:
    """Deployment-only hard gate before the second physical owner starts."""

    devices: tuple[str, ...]
    gpu_max_allocation_fraction: float
    gpu_minimum_headroom_bytes: int
    gpu_sample_interval_seconds: float
    required_tfidf_parallel_backend: str
    schema_version: str

    def __post_init__(self) -> None:
        devices = tuple(str(value) for value in self.devices)
        if (
            not devices
            or len(devices) != len(set(devices))
            or any(
                not value.startswith("cuda:")
                or not value.split(":", 1)[1].isdigit()
                for value in devices
            )
        ):
            raise ValueError(
                "first-owner validation devices must be distinct CUDA devices"
            )
        fraction = float(self.gpu_max_allocation_fraction)
        if (
            isinstance(self.gpu_max_allocation_fraction, bool)
            or not math.isfinite(fraction)
            or not 0.0 < fraction < 1.0
        ):
            raise ValueError(
                "first-owner GPU maximum allocation fraction must be in (0, 1)"
            )
        if (
            isinstance(self.gpu_minimum_headroom_bytes, bool)
            or not isinstance(self.gpu_minimum_headroom_bytes, int)
            or self.gpu_minimum_headroom_bytes < 1
        ):
            raise ValueError(
                "first-owner GPU minimum headroom must be a positive integer"
            )
        interval = float(self.gpu_sample_interval_seconds)
        if (
            isinstance(self.gpu_sample_interval_seconds, bool)
            or not math.isfinite(interval)
            or interval <= 0.0
        ):
            raise ValueError(
                "first-owner GPU sample interval must be finite and positive"
            )
        backend = str(self.required_tfidf_parallel_backend).strip().lower()
        if backend == "loky":
            backend = "processes"
        if backend not in {"threads", "processes"}:
            raise ValueError(
                "first-owner TF-IDF backend must be threads or processes"
            )
        if (
            self.schema_version
            != ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_POLICY_SCHEMA
        ):
            raise ValueError(
                "unsupported first-owner validation policy schema"
            )
        object.__setattr__(self, "devices", devices)
        object.__setattr__(
            self,
            "gpu_max_allocation_fraction",
            fraction,
        )
        object.__setattr__(
            self,
            "gpu_sample_interval_seconds",
            interval,
        )
        object.__setattr__(
            self,
            "required_tfidf_parallel_backend",
            backend,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "devices": list(self.devices),
            "gpu_max_allocation_fraction": (
                self.gpu_max_allocation_fraction
            ),
            "gpu_minimum_headroom_bytes": (
                self.gpu_minimum_headroom_bytes
            ),
            "gpu_sample_interval_seconds": (
                self.gpu_sample_interval_seconds
            ),
            "required_tfidf_parallel_backend": (
                self.required_tfidf_parallel_backend
            ),
        }


@dataclass(frozen=True)
class RoleNeutralStage1ExecutionPolicy:
    resource_plan: ResourcePlan
    max_parallel_owners: int
    neural_query_execution_topologies: Mapping[
        str,
        NeuralQueryExecutionTopology,
    ] = field(default_factory=dict)
    htr_operational_controls: RoleNeutralHTROperationalControls | None = None
    neural_query_operational_controls: (
        RoleNeutralNeuralQueryOperationalControls | None
    ) = None
    first_owner_validation: (
        RoleNeutralFirstOwnerValidationPolicy | None
    ) = None

    def __post_init__(self) -> None:
        if not isinstance(self.resource_plan, ResourcePlan):
            raise TypeError("execution policy requires a portable ResourcePlan")
        workers = int(self.max_parallel_owners)
        if workers < 1:
            raise ValueError("max_parallel_owners must be positive")
        if workers > int(self.resource_plan.cpu_budget):
            raise ValueError("max_parallel_owners cannot exceed the configured CPU budget")
        gate = self.first_owner_validation
        if gate is not None and not isinstance(
            gate,
            RoleNeutralFirstOwnerValidationPolicy,
        ):
            raise TypeError(
                "first_owner_validation must use its typed deployment policy"
            )
        if gate is not None:
            if tuple(self.resource_plan.devices) != gate.devices:
                raise ValueError(
                    "first-owner validation devices differ from the resource plan"
                )
            if (
                self.htr_operational_controls is None
                or self.neural_query_operational_controls is None
            ):
                raise ValueError(
                    "first-owner validation requires HTR and neural operational "
                    "controls"
                )
        if self.htr_operational_controls is not None and not isinstance(
            self.htr_operational_controls,
            RoleNeutralHTROperationalControls,
        ):
            raise TypeError(
                "execution policy HTR controls must use the typed "
                "deployment-only contract"
            )
        if self.htr_operational_controls is not None:
            selected_devices = tuple(self.resource_plan.devices)
            self.htr_operational_controls.bind_fold_resources(
                devices=selected_devices,
                owner_cpu_budget=int(self.resource_plan.cpu_budget),
            )
        if (
            self.neural_query_operational_controls is not None
            and not isinstance(
                self.neural_query_operational_controls,
                RoleNeutralNeuralQueryOperationalControls,
            )
        ):
            raise TypeError(
                "execution policy neural-query controls must use the typed "
                "deployment-only contract"
            )
        topologies = dict(self.neural_query_execution_topologies)
        selected = set(self.resource_plan.devices)
        if any(
            not isinstance(key, str)
            or not isinstance(value, NeuralQueryExecutionTopology)
            or key != value.primary_device
            for key, value in topologies.items()
        ):
            raise TypeError(
                "neural-query execution topologies must map each primary "
                "device to one typed topology"
            )
        if not set(topologies).issubset(selected) or any(
            not set(value.devices).issubset(selected)
            for value in topologies.values()
        ):
            raise ValueError(
                "neural-query execution topology requests an unavailable "
                "resource"
            )
        object.__setattr__(
            self,
            "neural_query_execution_topologies",
            topologies,
        )
        if self.neural_query_operational_controls is not None:
            effective_owners = (
                max(1, workers // len(self.resource_plan.devices))
                if self.htr_operational_controls is not None
                else workers
            )
            owner_cpu_budget = max(
                1,
                int(self.resource_plan.cpu_budget) // effective_owners,
            )
            runtime_topologies = tuple(topologies.values()) or tuple(
                NeuralQueryExecutionTopology.single(device)
                for device in self.resource_plan.devices
            )
            for topology in runtime_topologies:
                self.neural_query_operational_controls.bind_task_resources(
                    devices=topology.devices,
                    owner_cpu_budget=owner_cpu_budget,
                )

    def neural_query_topology_for(
        self,
        primary_resource: str,
    ) -> NeuralQueryExecutionTopology:
        resource = str(primary_resource)
        if resource not in self.resource_plan.devices:
            raise ValueError(
                "neural-query topology requested an unselected primary resource"
            )
        return self.neural_query_execution_topologies.get(
            resource,
            NeuralQueryExecutionTopology.single(resource),
        )


def _execute_one_owner(
    *,
    task: RoleNeutralPhysicalOwnerTask,
    factories: Mapping[str, RoleNeutralProducerFactory],
) -> RoleNeutralPhysicalOwnerResult:
    task.component_parent.mkdir(parents=True, exist_ok=False)
    sources: list[RoleNeutralComponentArtifactSource] = []
    component_order: list[str] = []
    operational_reports: dict[str, Mapping[str, Any]] = {}
    component_execution_intervals: list[dict[str, Any]] = []
    for component in EXPECTED_COMPONENT_FAMILIES:
        component_root = task.component_parent / component
        invocation = RoleNeutralComponentInvocation(
            plan=task.plan,
            physical_owner=task.physical_owner,
            logical_members=task.logical_members,
            component=component,
            output_root=component_root,
            resource=task.resource,
            neural_query_execution_topology=(
                task.neural_query_execution_topology
            ),
            htr_operational_controls=task.htr_operational_controls,
            neural_query_operational_controls=(
                task.neural_query_operational_controls
            ),
            htr_fold_devices=task.htr_fold_devices,
            owner_cpu_budget=task.owner_cpu_budget,
        )
        bound = factories[component](invocation)
        if not isinstance(bound, BoundRoleNeutralComponentProducer):
            raise TypeError(
                f"{task.physical_owner.scope_id}/{component} factory "
                "did not return a typed bound producer"
            )
        if component_root.exists() or component_root.is_symlink():
            raise FileExistsError(
                f"{task.physical_owner.scope_id}/{component} output "
                "existed before producer execution"
            )
        interval_started_monotonic_ns = time.monotonic_ns()
        execution_result = bound.execute()
        interval_finished_monotonic_ns = time.monotonic_ns()
        if interval_finished_monotonic_ns <= interval_started_monotonic_ns:
            raise RuntimeError(
                "role-neutral component execution interval did not advance"
            )
        accelerator_associated = (
            task.resource != "cpu"
            and component in _ACCELERATOR_ASSOCIATED_COMPONENTS
        )
        if component == "neural_query" and accelerator_associated:
            resource_ids = tuple(
                task.neural_query_execution_topology.devices
            )
        elif (
            component in {"htr", "matched_pair"}
            and accelerator_associated
        ):
            resource_ids = tuple(task.htr_fold_devices)
        elif accelerator_associated:
            resource_ids = (task.resource,)
        else:
            resource_ids = ("host_cpu",)
        component_execution_intervals.append(
            {
                "schema_version": (
                    ROLE_NEUTRAL_COMPONENT_EXECUTION_INTERVAL_SCHEMA
                ),
                "physical_owner_scope_id": (
                    task.physical_owner.scope_id
                ),
                "component": component,
                "lane_kind": (
                    "gpu" if accelerator_associated else "cpu"
                ),
                "resource_ids": list(resource_ids),
                "clock_domain_id": (
                    ROLE_NEUTRAL_COMPONENT_EXECUTION_CLOCK_DOMAIN
                ),
                "started_monotonic_ns": (
                    interval_started_monotonic_ns
                ),
                "finished_monotonic_ns": (
                    interval_finished_monotonic_ns
                ),
                "status": "completed",
                "timestamps_measured_directly": True,
                "interval_semantics": (
                    _ROLE_NEUTRAL_COMPONENT_EXECUTION_INTERVAL_SEMANTICS
                ),
            }
        )
        if isinstance(
            execution_result,
            RoleNeutralOperationalComponentReport,
        ):
            if (
                execution_result.component != component
                or component in operational_reports
            ):
                raise ValueError(
                    "component returned a substituted or duplicate "
                    "operational report"
                )
            operational_reports[component] = copy.deepcopy(
                dict(execution_result.attestation)
            )
        elif (
            component in {"htr", "matched_pair"}
            and task.htr_operational_controls is not None
        ):
            raise RuntimeError(
                f"typed {component} deployment controls were not "
                "operationally attested"
            )
        elif (
            component == "neural_query"
            and task.neural_query_operational_controls is not None
        ):
            raise RuntimeError(
                "typed neural-query deployment controls were not "
                "operationally attested"
            )
        if (
            component_root.is_symlink()
            or component_root.resolve(strict=True) != component_root
            or not component_root.is_dir()
        ):
            raise ValueError(
                f"{task.physical_owner.scope_id}/{component} producer "
                "did not publish its requested canonical root"
            )
        receipt = bound.authenticate()
        receipt = validate_authenticated_role_neutral_component_receipt(
            root=component_root,
            plan=task.plan,
            physical_owner_scope_id=task.physical_owner.scope_id,
            receipt=receipt,
            expected_component=component,
        )
        sources.append(
            RoleNeutralComponentArtifactSource(
                root=component_root,
                receipt=receipt,
            )
        )
        component_order.append(component)
    if tuple(component_order) != tuple(EXPECTED_COMPONENT_FAMILIES):
        raise RuntimeError("physical owner did not execute canonical six producers")
    return RoleNeutralPhysicalOwnerResult(
        physical_owner_scope_id=task.physical_owner.scope_id,
        sources=tuple(sources),
        component_execution_order=tuple(component_order),
        resource=task.resource,
        execution_telemetry={
            "schema_version": (
                "production_role_neutral_component_operational_reports_v2"
            ),
            "component_reports": {
                name: operational_reports[name]
                for name in sorted(operational_reports)
            },
            "component_execution_interval_semantics": (
                _ROLE_NEUTRAL_COMPONENT_EXECUTION_REPORT_SEMANTICS
            ),
            "component_execution_intervals": (
                component_execution_intervals
            ),
        },
    )


def validate_role_neutral_component_execution_intervals(
    *,
    execution_telemetry: Mapping[str, Any] | None,
    expected_physical_owner_scope_id: str,
    expected_primary_resource: str,
    expected_neural_query_resources: Sequence[str],
    expected_htr_resources: Sequence[str] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Close one owner's six directly measured architecture envelopes.

    Process and persistent executors wrap the architecture report in their
    own execution telemetry.  This reader deliberately accepts exactly that
    one wrapper boundary, then validates the same interval contract used by
    an in-process result.  The returned records remain operational metadata;
    device locators and timestamps never enter scientific identity.
    """

    if not isinstance(execution_telemetry, Mapping):
        raise ValueError(
            "role-neutral owner omitted component execution telemetry"
        )
    report: Mapping[str, Any] = execution_telemetry
    if report.get("schema_version") != (
        "production_role_neutral_component_operational_reports_v2"
    ):
        wrapped = report.get("worker_report")
        if not isinstance(wrapped, Mapping):
            raise ValueError(
                "role-neutral executor omitted its component worker report"
            )
        report = wrapped
    intervals = report.get("component_execution_intervals")
    if (
        report.get("schema_version")
        != "production_role_neutral_component_operational_reports_v2"
        or report.get("component_execution_interval_semantics")
        != _ROLE_NEUTRAL_COMPONENT_EXECUTION_REPORT_SEMANTICS
        or not isinstance(report.get("component_reports"), Mapping)
        or not isinstance(intervals, list)
        or len(intervals) != len(EXPECTED_COMPONENT_FAMILIES)
    ):
        raise ValueError(
            "role-neutral component execution report is incomplete"
        )
    owner_scope_id = str(expected_physical_owner_scope_id)
    primary_resource = str(expected_primary_resource)
    neural_resources = tuple(
        str(value) for value in expected_neural_query_resources
    )
    htr_resources = (
        (primary_resource,)
        if expected_htr_resources is None
        else tuple(str(value) for value in expected_htr_resources)
    )
    if (
        not owner_scope_id
        or not primary_resource
        or not neural_resources
        or neural_resources[0] != primary_resource
        or len(neural_resources) != len(set(neural_resources))
        or not htr_resources
        or primary_resource not in htr_resources
        or len(htr_resources) != len(set(htr_resources))
    ):
        raise ValueError(
            "expected role-neutral interval capabilities are invalid"
        )
    closed: list[Mapping[str, Any]] = []
    previous_finish: int | None = None
    for component, row in zip(
        EXPECTED_COMPONENT_FAMILIES,
        intervals,
        strict=True,
    ):
        accelerator_associated = (
            primary_resource != "cpu"
            and component in _ACCELERATOR_ASSOCIATED_COMPONENTS
        )
        expected_lane = "gpu" if accelerator_associated else "cpu"
        expected_resources = (
            neural_resources
            if component == "neural_query" and accelerator_associated
            else (
                htr_resources
                if (
                    component in {"htr", "matched_pair"}
                    and accelerator_associated
                )
                else (
                    (primary_resource,)
                    if accelerator_associated
                    else ("host_cpu",)
                )
            )
        )
        if (
            not isinstance(row, Mapping)
            or set(row)
            != _ROLE_NEUTRAL_COMPONENT_EXECUTION_INTERVAL_FIELDS
            or row.get("schema_version")
            != ROLE_NEUTRAL_COMPONENT_EXECUTION_INTERVAL_SCHEMA
            or row.get("physical_owner_scope_id") != owner_scope_id
            or row.get("component") != component
            or row.get("lane_kind") != expected_lane
            or row.get("resource_ids") != list(expected_resources)
            or row.get("clock_domain_id")
            != ROLE_NEUTRAL_COMPONENT_EXECUTION_CLOCK_DOMAIN
            or isinstance(row.get("started_monotonic_ns"), bool)
            or not isinstance(row.get("started_monotonic_ns"), int)
            or isinstance(row.get("finished_monotonic_ns"), bool)
            or not isinstance(row.get("finished_monotonic_ns"), int)
            or int(row["started_monotonic_ns"]) < 0
            or int(row["finished_monotonic_ns"])
            <= int(row["started_monotonic_ns"])
            or (
                previous_finish is not None
                and int(row["started_monotonic_ns"]) < previous_finish
            )
            or row.get("status") != "completed"
            or row.get("timestamps_measured_directly") is not True
            or row.get("interval_semantics")
            != _ROLE_NEUTRAL_COMPONENT_EXECUTION_INTERVAL_SEMANTICS
        ):
            raise ValueError(
                "role-neutral component execution interval changed, "
                "overlapped its serial peer, or is incomplete"
            )
        previous_finish = int(row["finished_monotonic_ns"])
        closed.append(copy.deepcopy(dict(row)))
    return tuple(closed)


def _compute_canary_scientific_replica(
    result: RoleNeutralPhysicalOwnerResult,
) -> dict[str, Any]:
    """Close one complete six-producer/all-ten scientific replica."""

    if not isinstance(result, RoleNeutralPhysicalOwnerResult):
        raise TypeError("compute-canary replica must be a typed owner result")
    if (
        result.component_execution_order != tuple(EXPECTED_COMPONENT_FAMILIES)
        or len(result.sources) != len(EXPECTED_COMPONENT_FAMILIES)
    ):
        raise ValueError("compute-canary replica lacks the canonical six producers")
    receipts = tuple(source.receipt for source in result.sources)
    if tuple(receipt.component for receipt in receipts) != tuple(
        EXPECTED_COMPONENT_FAMILIES
    ):
        raise ValueError("compute-canary component order changed")
    family_artifact_ids: dict[str, str] = {}
    for receipt in receipts:
        scientific = receipt.scientific_dict()
        for native_family, artifact_id in (
            scientific.get("family_fit_artifact_sha256") or {}
        ).items():
            family = NATIVE_TO_PORTABLE_FAMILY.get(str(native_family))
            if family is None:
                raise ValueError(
                    "compute-canary replica contains an unknown native family"
                )
            if family in family_artifact_ids:
                raise ValueError(
                    "compute-canary replica duplicated an evidence family"
                )
            family_artifact_ids[str(family)] = _require_sha256(
                artifact_id,
                label=f"compute-canary {family} artifact",
            )
    from .portable_workflow_spec import EVIDENCE_FAMILIES

    if set(family_artifact_ids) != set(EVIDENCE_FAMILIES):
        raise ValueError(
            "compute-canary replica does not contain exactly all ten families"
        )
    body = {
        "schema_version": (
            "production_role_neutral_compute_canary_scientific_replica_v1"
        ),
        "physical_owner_scope_id": result.physical_owner_scope_id,
        "canonical_component_order": list(EXPECTED_COMPONENT_FAMILIES),
        "component_scientific_receipts": [
            receipt.scientific_dict() for receipt in receipts
        ],
        "family_artifact_ids": {
            family: family_artifact_ids[family] for family in EVIDENCE_FAMILIES
        },
        "all_ten_scientific_families_present": True,
        "resource_locator_included": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


class _FirstOwnerGpuSampler:
    """Continuously observe total device allocation during the hard gate."""

    def __init__(
        self,
        *,
        devices: tuple[str, ...],
        interval_seconds: float,
    ) -> None:
        self.devices = tuple(devices)
        self.interval_seconds = float(interval_seconds)
        self._samples: list[dict[str, Any]] = []
        self._errors: list[dict[str, str]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        acquisition_started = time.monotonic()
        try:
            rows = telemetry_module.sample_nvidia_gpus(self.devices)
        except BaseException as exc:
            with self._lock:
                self._errors.append(
                    {
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            return
        acquisition_finished = time.monotonic()
        if acquisition_finished < acquisition_started:
            with self._lock:
                self._errors.append(
                    {
                        "exception_type": "RuntimeError",
                        "message": (
                            "first-owner GPU sample acquisition clock "
                            "moved backwards"
                        ),
                    }
                )
            return
        with self._lock:
            self._samples.extend(
                {
                    **copy.deepcopy(dict(row)),
                    "sample_acquisition_started_monotonic_seconds": (
                        acquisition_started
                    ),
                    "sample_acquisition_finished_monotonic_seconds": (
                        acquisition_finished
                    ),
                    # A point sample is observed only when the host NVML
                    # acquisition returns, so its timestamp is completion.
                    "sample_monotonic_seconds": acquisition_finished,
                }
                for row in rows
            )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> "_FirstOwnerGpuSampler":
        self._sample()
        self._thread = threading.Thread(
            target=self._run,
            name="oci-first-owner-gpu-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
        self._sample()

    @property
    def samples(self) -> tuple[Mapping[str, Any], ...]:
        with self._lock:
            return tuple(copy.deepcopy(self._samples))

    @property
    def errors(self) -> tuple[Mapping[str, str], ...]:
        with self._lock:
            return tuple(copy.deepcopy(self._errors))


def _first_owner_gate_path(execution_root: Path) -> Path:
    return (
        execution_root
        / ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_GATE_SUFFIX
    )


def _durably_publish_first_owner_gate(
    *,
    path: Path,
    value: Mapping[str, Any],
) -> None:
    _write_new_json(path, value)
    descriptor = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(
                "first-owner validation parent is not a directory"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validated_operational_attestation(
    value: Any,
    *,
    label: str,
    schema: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} operational attestation is missing")
    result = copy.deepcopy(dict(value))
    body = {
        key: copy.deepcopy(child)
        for key, child in result.items()
        if key != "content_sha256"
    }
    if (
        result.get("schema_version") != schema
        or result.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError(
            f"{label} operational attestation is not self-authenticated"
        )
    return result


def _first_owner_receipt_reauthentication(
    result: RoleNeutralPhysicalOwnerResult,
) -> dict[str, Any]:
    """Bind the freshly reopened component receipts behind coverage claims."""

    if not isinstance(result, RoleNeutralPhysicalOwnerResult):
        raise TypeError(
            "first-owner receipt reauthentication requires a typed result"
        )
    if (
        result.component_execution_order
        != tuple(EXPECTED_COMPONENT_FAMILIES)
        or len(result.sources) != len(EXPECTED_COMPONENT_FAMILIES)
    ):
        raise ValueError(
            "first-owner receipt reauthentication lacks every component"
        )
    components: list[dict[str, Any]] = []
    for component, source in zip(
        EXPECTED_COMPONENT_FAMILIES,
        result.sources,
        strict=True,
    ):
        receipt = source.receipt
        if (
            not isinstance(
                receipt,
                AuthenticatedRoleNeutralComponentReceipt,
            )
            or receipt.component != component
        ):
            raise ValueError(
                "first-owner receipt reauthentication changed component order"
            )
        scientific = receipt.scientific_dict()
        execution = receipt.execution_attestation()
        components.append(
            {
                "component": component,
                "component_authentication_content_sha256": (
                    receipt.authentication_content_sha256
                ),
                "component_scientific_content_sha256": (
                    scientific["content_sha256"]
                ),
                "component_execution_attestation_content_sha256": (
                    execution["content_sha256"]
                ),
                "source_terminal_content_sha256": (
                    receipt.source_terminal_content_sha256
                ),
                "source_tree_sha256": receipt.source_tree_sha256,
                "family_fit_artifact_sha256": copy.deepcopy(
                    scientific["family_fit_artifact_sha256"]
                ),
                "registered_heldout_labels_accessed": False,
                "oracle_fields_accessed": False,
                "text_truncation_applied": False,
                "lossy_evidence_selection_applied": False,
            }
        )
    body = {
        "schema_version": (
            "production_role_neutral_first_owner_receipt_"
            "reauthentication_v1"
        ),
        "physical_owner_scope_id": result.physical_owner_scope_id,
        "canonical_component_order": list(EXPECTED_COMPONENT_FAMILIES),
        "components": components,
        "component_root_count": len(components),
        "every_component_root_reopened_and_tree_rehashed": True,
        "every_component_terminal_reopened_and_content_hash_matched": True,
        "every_component_receipt_self_authenticated": True,
        "complete_text_and_chunk_coverage_reauthenticated": True,
        "coverage_reauthentication_basis": (
            "fresh_parent_component_tree_terminal_and_receipt_validation_v1"
        ),
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _maximum_interval_overlap(
    rows: Sequence[Mapping[str, Any]],
) -> int:
    events: list[tuple[int, int]] = []
    for row in rows:
        started = row.get("started_monotonic_ns")
        finished = row.get("finished_monotonic_ns")
        if (
            isinstance(started, bool)
            or not isinstance(started, int)
            or isinstance(finished, bool)
            or not isinstance(finished, int)
            or finished <= started
        ):
            raise ValueError("first-owner task interval is invalid")
        events.extend(((started, 1), (finished, -1)))
    active = 0
    maximum = 0
    for _timestamp, delta in sorted(
        events,
        key=lambda event: (event[0], event[1]),
    ):
        active += delta
        if active < 0:
            raise ValueError(
                "first-owner task interval released an inactive lease"
            )
        maximum = max(maximum, active)
    if active != 0:
        raise ValueError("first-owner task intervals are not closed")
    return maximum


def _validate_first_owner_task_phase(
    value: Any,
    *,
    phase: str,
    devices: tuple[str, ...],
) -> dict[str, Any]:
    report = _validated_operational_attestation(
        value,
        label=f"neural-query {phase}",
        schema="production_neural_query_task_phase_execution_attestation_v1",
    )
    intervals = report.get("task_intervals")
    per_device = report.get("per_device")
    configured = report.get("configured_parallelism")
    actual_count = report.get("actual_task_count")
    maximum = report.get("maximum_concurrent_leases")
    if (
        report.get("phase") != phase
        or not isinstance(intervals, list)
        or not intervals
        or {
            str(row.get("device"))
            for row in intervals
            if isinstance(row, Mapping)
        }
        != set(devices)
        or not isinstance(per_device, Mapping)
        or set(per_device) != set(devices)
        or isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured < 1
        or isinstance(actual_count, bool)
        or not isinstance(actual_count, int)
        or actual_count != len(intervals)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum != _maximum_interval_overlap(intervals)
        or maximum > configured
        or report.get("configured_total_parallelism_respected") is not True
        or report.get("configured_per_device_slots_respected") is not True
        or report.get("waiting_tasks_hold_no_lease") is not True
        or report.get("canonical_result_order_restored") is not True
        or (
            configured > 1
            and actual_count > 1
            and (
                maximum < 2
                or report.get("process_isolated") is not True
            )
        )
        or any(
            not isinstance(per_device[device], Mapping)
            or int(per_device[device].get("task_count", 0)) < 1
            for device in devices
        )
    ):
        raise ValueError(
            f"neural-query {phase} execution serialized or changed resources"
        )
    return {
        "content_sha256": report["content_sha256"],
        "configured_parallelism": configured,
        "actual_task_count": actual_count,
        "maximum_concurrent_leases": maximum,
        "devices": list(devices),
    }


def _validate_first_owner_component_reports(
    *,
    result: RoleNeutralPhysicalOwnerResult | None = None,
    execution_telemetry: Mapping[str, Any] | None = None,
    policy: RoleNeutralStage1ExecutionPolicy | None = None,
    gate: RoleNeutralFirstOwnerValidationPolicy,
) -> dict[str, Any]:
    del policy
    if (result is None) is (execution_telemetry is None):
        raise ValueError(
            "first-owner component validation requires exactly one "
            "telemetry source"
        )
    telemetry = (
        result.execution_telemetry
        if result is not None
        else execution_telemetry
    )
    if not isinstance(telemetry, Mapping):
        raise ValueError("first owner omitted execution telemetry")
    worker_report: Any = telemetry
    if worker_report.get("schema_version") != (
        "production_role_neutral_component_operational_reports_v2"
    ):
        worker_report = telemetry.get("worker_report")
    if (
        not isinstance(worker_report, Mapping)
        or worker_report.get("schema_version")
        != "production_role_neutral_component_operational_reports_v2"
    ):
        raise ValueError(
            "first owner omitted its authenticated component reports"
        )
    reports = worker_report.get("component_reports")
    expected_reports = {
        "htr",
        "matched_pair",
        "tfidf",
        "neural_query",
    }
    if not isinstance(reports, Mapping) or set(reports) != expected_reports:
        raise ValueError(
            "first owner did not attest every parallel producer"
        )

    htr = _validated_operational_attestation(
        reports["htr"],
        label="HTR",
        schema=ROLE_NEUTRAL_HTR_OPERATIONAL_ATTESTATION_SCHEMA,
    )
    htr_plan = htr.get("fold_resource_plan")
    htr_execution = htr.get("fold_execution")
    if (
        not isinstance(htr_plan, Mapping)
        or htr_plan.get("devices") != list(gate.devices)
        or not isinstance(htr_execution, Mapping)
        or htr_execution.get("resource_plan") != htr_plan
        or htr_execution.get("nuisance_barrier_enforced") is not True
        or htr_execution.get(
            "effect_submitted_only_after_nuisance_oof_and_residuals"
        )
        is not True
        or htr_execution.get(
            "every_selected_device_used_by_each_stage"
        )
        is not True
    ):
        raise ValueError("first-owner HTR execution changed its resource plan")
    htr_intervals = htr_execution.get("fold_intervals")
    if not isinstance(htr_intervals, list) or not htr_intervals:
        raise ValueError("first-owner HTR fold intervals are missing")
    htr_nuisance = [
        row for row in htr_intervals if row.get("stage") == "nuisance"
    ]
    htr_effect = [
        row for row in htr_intervals if row.get("stage") == "effect"
    ]
    htr_parallelism = int(htr_plan.get("fold_parallelism", 0))
    htr_nuisance_overlap = _maximum_interval_overlap(htr_nuisance)
    htr_effect_overlap = _maximum_interval_overlap(htr_effect)
    if (
        htr_parallelism < 1
        or int(htr_execution.get("maximum_concurrent_fold_leases", 0))
        > htr_parallelism
        or (
            htr_parallelism > 1
            and (
                htr_nuisance_overlap < 2
                or htr_effect_overlap < 2
                or htr_execution.get("process_isolated_rng") is not True
            )
        )
    ):
        raise ValueError("first-owner HTR folds serialized")

    matched = _validated_operational_attestation(
        reports["matched_pair"],
        label="matched-pair",
        schema=_MATCHED_PAIR_OPERATIONAL_ATTESTATION_SCHEMA,
    )
    matched_plan = matched.get("fold_resource_plan")
    matched_execution = matched.get("fold_execution")
    if (
        not isinstance(matched_plan, Mapping)
        or matched_plan.get("devices") != list(gate.devices)
        or not isinstance(matched_execution, Mapping)
        or matched_execution.get("resource_plan") != matched_plan
        or matched_execution.get("every_selected_device_used") is not True
    ):
        raise ValueError(
            "first-owner matched-pair execution changed its resource plan"
        )
    matched_intervals = matched_execution.get("fold_intervals")
    if not isinstance(matched_intervals, list) or not matched_intervals:
        raise ValueError("first-owner matched-pair intervals are missing")
    matched_parallelism = int(
        matched_plan.get("fold_parallelism", 0)
    )
    matched_overlap = _maximum_interval_overlap(matched_intervals)
    if (
        matched_parallelism < 1
        or matched_overlap
        != int(
            matched_execution.get(
                "maximum_concurrent_fold_leases",
                -1,
            )
        )
        or matched_overlap > matched_parallelism
        or (
            matched_parallelism > 1
            and (
                matched_overlap < 2
                or matched_execution.get(
                    "process_isolated_rng_and_torch_determinism"
                )
                is not True
            )
        )
    ):
        raise ValueError("first-owner matched-pair folds serialized")

    tfidf = _validated_operational_attestation(
        reports["tfidf"],
        label="TF-IDF",
        schema=_TFIDF_NUISANCE_EXECUTION_ATTESTATION_SCHEMA,
    )
    tfidf_effective = int(tfidf.get("effective_workers", 0))
    tfidf_overlap = int(
        tfidf.get("actual_peak_concurrent_fold_workers", 0)
    )
    tfidf_pids = tfidf.get("worker_pids")
    if (
        tfidf.get("configured_backend")
        != gate.required_tfidf_parallel_backend
        or tfidf_effective < 1
        or tfidf_overlap < 1
        or tfidf_overlap > tfidf_effective
        or tfidf.get("subfold_parallelism") != 1
        or tfidf.get("subfold_joblib_pools_created") is not False
        or tfidf.get("full_data_base_fits_after_fold_barrier") is not True
        or tfidf.get("final_stack_fits_after_fold_barrier") is not True
        or (
            tfidf_effective > 1
            and (
                tfidf_overlap < 2
                or tfidf.get("fold_overlap_observed") is not True
                or not isinstance(tfidf_pids, list)
                or (
                    gate.required_tfidf_parallel_backend == "processes"
                    and len(set(tfidf_pids)) < 2
                )
            )
        )
    ):
        raise ValueError("first-owner TF-IDF folds serialized")

    neural = _validated_operational_attestation(
        reports["neural_query"],
        label="neural-query",
        schema=(
            "production_role_neutral_neural_query_operational_attestation_v1"
        ),
    )
    neural_plan = neural.get("resource_plan")
    phases = neural.get("phases")
    expected_phase_order = [
        "inner_folds_then_consensus_final_refits",
        "safe_evidence_banks",
        "heldout_moment_banks",
    ]
    if (
        not isinstance(neural_plan, Mapping)
        or neural_plan.get("devices") != list(gate.devices)
        or neural.get("phase_order") != expected_phase_order
        or neural.get("phase_count") != len(expected_phase_order)
        or not isinstance(phases, list)
        or len(phases) != len(expected_phase_order)
        or neural.get("all_phase_attestations_self_authenticated")
        is not True
        or neural.get("canonical_execution_order_preserved") is not True
    ):
        raise ValueError(
            "first-owner neural-query operational coverage changed"
        )
    discovery_wrapper = phases[0]
    if (
        not isinstance(discovery_wrapper, Mapping)
        or discovery_wrapper.get("phase_index") != 0
        or discovery_wrapper.get("phase")
        != "inner_folds_then_consensus_final_refits"
    ):
        raise ValueError("first-owner neural discovery phase changed")
    discovery = _validated_operational_attestation(
        discovery_wrapper.get("attestation"),
        label="neural-query discovery",
        schema="production_neural_query_discovery_execution_attestation_v1",
    )
    if (
        discovery.get("resource_plan") != neural_plan
        or discovery.get("inner_fold_barrier_enforced") is not True
        or discovery.get(
            "all_inner_results_verified_before_final_task_construction"
        )
        is not True
    ):
        raise ValueError("first-owner neural-query barrier changed")
    neural_summaries = {
        "inner_folds": _validate_first_owner_task_phase(
            discovery.get("inner_fold_phase"),
            phase="inner_folds",
            devices=gate.devices,
        ),
        "consensus_and_final_refit_banks": (
            _validate_first_owner_task_phase(
                discovery.get("final_bank_phase"),
                phase="consensus_and_final_refit_banks",
                devices=gate.devices,
            )
        ),
    }
    for index, phase in enumerate(
        ("safe_evidence_banks", "heldout_moment_banks"),
        start=1,
    ):
        wrapper = phases[index]
        if (
            not isinstance(wrapper, Mapping)
            or wrapper.get("phase_index") != index
            or wrapper.get("phase") != phase
        ):
            raise ValueError(
                f"first-owner neural-query {phase} wrapper changed"
            )
        neural_summaries[phase] = _validate_first_owner_task_phase(
            wrapper.get("attestation"),
            phase=phase,
            devices=gate.devices,
        )

    return {
        "htr": {
            "content_sha256": htr["content_sha256"],
            "nuisance_maximum_concurrent_leases": (
                htr_nuisance_overlap
            ),
            "effect_maximum_concurrent_leases": htr_effect_overlap,
            "devices": list(gate.devices),
        },
        "matched_pair": {
            "content_sha256": matched["content_sha256"],
            "maximum_concurrent_leases": matched_overlap,
            "devices": list(gate.devices),
        },
        "tfidf": {
            "content_sha256": tfidf["content_sha256"],
            "effective_workers": tfidf_effective,
            "maximum_concurrent_leases": tfidf_overlap,
            "backend": gate.required_tfidf_parallel_backend,
        },
        "neural_query": {
            "content_sha256": neural["content_sha256"],
            "phases": neural_summaries,
            "devices": list(gate.devices),
        },
        "every_parallel_component_report_self_authenticated": True,
        "configured_parallel_work_did_not_serialize": True,
    }


def _first_owner_memory_observation(
    *,
    samples: Sequence[Mapping[str, Any]],
    sampling_errors: Sequence[Mapping[str, str]],
    policy: RoleNeutralFirstOwnerValidationPolicy,
) -> dict[str, Any]:
    if sampling_errors:
        raise RuntimeError(
            "first-owner GPU sampler recorded one or more errors"
        )
    by_device = {
        device: [
            copy.deepcopy(dict(row))
            for row in samples
            if str(row.get("device")) == device
        ]
        for device in policy.devices
    }
    result: dict[str, Any] = {}
    for device in policy.devices:
        rows = by_device[device]
        uuids = {
            str(row.get("uuid"))
            for row in rows
            if isinstance(row.get("uuid"), str)
            and str(row.get("uuid")).strip()
        }
        totals = {
            int(row.get("memory_total_bytes", 0))
            for row in rows
            if isinstance(row.get("memory_total_bytes"), int)
            and not isinstance(row.get("memory_total_bytes"), bool)
        }
        used = [
            int(row.get("memory_used_bytes", -1))
            for row in rows
            if isinstance(row.get("memory_used_bytes"), int)
            and not isinstance(row.get("memory_used_bytes"), bool)
        ]
        acquisition_brackets: list[tuple[float, float, float]] = []
        for row in rows:
            started = row.get(
                "sample_acquisition_started_monotonic_seconds"
            )
            finished = row.get(
                "sample_acquisition_finished_monotonic_seconds"
            )
            sampled = row.get("sample_monotonic_seconds")
            if (
                isinstance(started, bool)
                or not isinstance(started, (int, float))
                or not math.isfinite(float(started))
                or isinstance(finished, bool)
                or not isinstance(finished, (int, float))
                or not math.isfinite(float(finished))
                or float(finished) < float(started)
                or isinstance(sampled, bool)
                or not isinstance(sampled, (int, float))
                or not math.isfinite(float(sampled))
                or float(sampled) != float(finished)
            ):
                raise RuntimeError(
                    "first-owner GPU sample lacks its acquisition bracket"
                )
            acquisition_brackets.append(
                (
                    float(started),
                    float(finished),
                    float(sampled),
                )
            )
        if (
            len(rows) < 2
            or len(uuids) != 1
            or len(totals) != 1
            or next(iter(totals)) <= 0
            or len(used) != len(rows)
            or any(value < 0 for value in used)
            or [
                bracket[2] for bracket in acquisition_brackets
            ]
            != sorted(
                bracket[2] for bracket in acquisition_brackets
            )
        ):
            raise RuntimeError(
                f"first-owner GPU telemetry is incomplete for {device}"
            )
        total = next(iter(totals))
        peak = max(used)
        if peak > total:
            raise RuntimeError(
                f"first-owner GPU allocation exceeds capacity on {device}"
            )
        fraction = peak / total
        headroom = total - peak
        result[device] = {
            "uuid": next(iter(uuids)),
            "sample_count": len(rows),
            "memory_total_bytes": total,
            "host_peak_memory_used_bytes": peak,
            "peak_memory_used_bytes": peak,
            "memory_acceptance_peak_bytes": peak,
            "memory_acceptance_peak_source": (
                "host_nvml_absolute_peak"
            ),
            "peak_allocation_fraction": fraction,
            "minimum_headroom_bytes": headroom,
            "maximum_allocation_fraction_threshold": (
                policy.gpu_max_allocation_fraction
            ),
            "minimum_headroom_threshold_bytes": (
                policy.gpu_minimum_headroom_bytes
            ),
            "allocation_fraction_accepted": (
                fraction <= policy.gpu_max_allocation_fraction
            ),
            "headroom_accepted": (
                headroom >= policy.gpu_minimum_headroom_bytes
            ),
            "host_nvml_absolute_peak_used_for_acceptance": True,
            "sample_timestamp_is_acquisition_completion": True,
            "sample_acquisition_brackets_retained": True,
        }
    accepted = not any(
        row["allocation_fraction_accepted"] is not True
        or row["headroom_accepted"] is not True
        for row in result.values()
    )
    return {
        "devices": result,
        "all_selected_devices_sampled": True,
        "continuous_host_level_sampling": True,
        "host_nvml_absolute_peak_used_for_acceptance": True,
        "memory_acceptance_checks_absolute_peak_fraction_and_headroom": True,
        "sample_timestamp_is_acquisition_completion": True,
        "sample_acquisition_brackets_retained": True,
        "maximum_allocation_fraction_respected": all(
            row["allocation_fraction_accepted"] is True
            for row in result.values()
        ),
        "minimum_headroom_respected": all(
            row["headroom_accepted"] is True
            for row in result.values()
        ),
        "accepted": accepted,
    }


def _normalize_executor_results(
    *,
    plan: Stage1ScopePlan,
    results: Sequence[RoleNeutralPhysicalOwnerResult],
    assigned_resources: Mapping[str, str],
) -> tuple[
    dict[str, tuple[RoleNeutralComponentArtifactSource, ...]],
    tuple[str, ...],
]:
    expected_ids = tuple(owner.scope_id for owner in plan.physical_scopes)
    if isinstance(results, (str, bytes, Mapping)):
        raise TypeError("role-neutral executor results must be one sequence")
    rows = tuple(results)
    if any(not isinstance(row, RoleNeutralPhysicalOwnerResult) for row in rows):
        raise TypeError("role-neutral executor returned an untyped result")
    completed_ids = tuple(row.physical_owner_scope_id for row in rows)
    if len(completed_ids) != len(set(completed_ids)) or set(completed_ids) != set(expected_ids):
        raise ValueError("role-neutral executor results have missing, duplicate, or extra owners")
    by_owner: dict[str, tuple[RoleNeutralComponentArtifactSource, ...]] = {}
    for row in rows:
        if (
            row.component_execution_order != tuple(EXPECTED_COMPONENT_FAMILIES)
            or row.resource != assigned_resources[row.physical_owner_scope_id]
            or len(row.sources) != len(EXPECTED_COMPONENT_FAMILIES)
            or tuple(source.receipt.component for source in row.sources)
            != tuple(EXPECTED_COMPONENT_FAMILIES)
        ):
            raise ValueError(
                f"{row.physical_owner_scope_id} returned an incomplete "
                "or substituted producer group"
            )
        by_owner[row.physical_owner_scope_id] = row.sources
    return by_owner, completed_ids


def _freshly_reauthenticate_owner_result(
    *,
    task: RoleNeutralPhysicalOwnerTask,
    result: RoleNeutralPhysicalOwnerResult,
) -> RoleNeutralPhysicalOwnerResult:
    """Reopen every child-produced byte at the parent trust boundary."""

    if not isinstance(result, RoleNeutralPhysicalOwnerResult):
        raise TypeError("physical-owner result must be typed before reauthentication")
    if (
        result.physical_owner_scope_id != task.physical_owner.scope_id
        or result.resource != task.resource
        or result.component_execution_order != tuple(EXPECTED_COMPONENT_FAMILIES)
        or len(result.sources) != len(EXPECTED_COMPONENT_FAMILIES)
    ):
        raise ValueError("physical-owner result changed its task capability")
    validate_role_neutral_component_execution_intervals(
        execution_telemetry=result.execution_telemetry,
        expected_physical_owner_scope_id=task.physical_owner.scope_id,
        expected_primary_resource=task.resource,
        expected_neural_query_resources=(
            task.neural_query_execution_topology.devices
        ),
        expected_htr_resources=task.htr_fold_devices,
    )
    rebound: list[RoleNeutralComponentArtifactSource] = []
    for component, source in zip(
        EXPECTED_COMPONENT_FAMILIES,
        result.sources,
        strict=True,
    ):
        expected_root = task.component_parent / component
        if (
            not isinstance(source, RoleNeutralComponentArtifactSource)
            or source.root != expected_root
        ):
            raise ValueError(
                "physical-owner result points outside its assigned component root"
            )
        receipt = validate_authenticated_role_neutral_component_receipt(
            root=expected_root,
            plan=task.plan,
            physical_owner_scope_id=task.physical_owner.scope_id,
            receipt=source.receipt,
            expected_component=component,
        )
        rebound.append(
            RoleNeutralComponentArtifactSource(
                root=expected_root,
                receipt=receipt,
            )
        )
    return replace(result, sources=tuple(rebound))


def _execution_telemetry_summary(
    *,
    plan: Stage1ScopePlan,
    owner_results: Sequence[RoleNeutralPhysicalOwnerResult],
    canary_replicas: Sequence[RoleNeutralPhysicalOwnerResult],
    process_isolated: bool,
) -> dict[str, Any]:
    by_owner = {
        result.physical_owner_scope_id: result for result in owner_results
    }
    expected = tuple(owner.scope_id for owner in plan.physical_scopes)
    if len(by_owner) != len(owner_results) or set(by_owner) != set(expected):
        raise ValueError("execution telemetry owner coverage is incomplete")

    def closed(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("owner execution telemetry must be a mapping or null")
        try:
            return json.loads(_canonical_json(value))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TypeError("owner execution telemetry is not closed JSON") from exc

    canary_rows = tuple(canary_replicas)
    if len(canary_rows) not in {0, 2}:
        raise ValueError("compute-canary telemetry must contain zero or two replicas")
    body = {
        "schema_version": (
            "production_role_neutral_owner_execution_telemetry_v1"
        ),
        "physical_owner_order": list(expected),
        "physical_owners": [
            {
                "physical_owner_scope_id": owner_scope_id,
                "resource": by_owner[owner_scope_id].resource,
                "telemetry": closed(
                    by_owner[owner_scope_id].execution_telemetry
                ),
            }
            for owner_scope_id in expected
        ],
        "compute_canary_replicas": [
            {
                "replica": replica,
                "physical_owner_scope_id": result.physical_owner_scope_id,
                "resource": result.resource,
                "telemetry": closed(result.execution_telemetry),
            }
            for replica, result in zip(
                ("replica_a", "replica_b")[: len(canary_rows)],
                canary_rows,
                strict=True,
            )
        ],
        "process_isolated_physical_owners": bool(process_isolated),
        "parent_process_counters_included_in_child_counters": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _validate_execution_telemetry_summary(
    value: Any,
    *,
    plan: Stage1ScopePlan,
    canary_completed: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("owner execution telemetry summary is missing")
    expected_fields = {
        "schema_version",
        "physical_owner_order",
        "physical_owners",
        "compute_canary_replicas",
        "process_isolated_physical_owners",
        "parent_process_counters_included_in_child_counters",
        "content_sha256",
    }
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    owner_ids = tuple(owner.scope_id for owner in plan.physical_scopes)
    owners = value.get("physical_owners")
    replicas = value.get("compute_canary_replicas")
    if (
        set(value) != expected_fields
        or value.get("schema_version")
        != "production_role_neutral_owner_execution_telemetry_v1"
        or value.get("physical_owner_order") != list(owner_ids)
        or not isinstance(owners, list)
        or len(owners) != len(owner_ids)
        or not isinstance(replicas, list)
        or len(replicas) != (2 if canary_completed else 0)
        or type(value.get("process_isolated_physical_owners")) is not bool
        or value.get("parent_process_counters_included_in_child_counters")
        is not False
        or value.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("owner execution telemetry summary is invalid")
    for expected_owner, row in zip(owner_ids, owners, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"physical_owner_scope_id", "resource", "telemetry"}
            or row.get("physical_owner_scope_id") != expected_owner
            or not isinstance(row.get("resource"), str)
            or not row["resource"]
            or (
                row.get("telemetry") is not None
                and not isinstance(row.get("telemetry"), Mapping)
            )
        ):
            raise ValueError("physical-owner execution telemetry row is invalid")
    if canary_completed:
        earliest = min(
            plan.physical_scopes,
            key=lambda owner: int(owner.canonical_index),
        )
        for expected_replica, row in zip(
            ("replica_a", "replica_b"),
            replicas,
            strict=True,
        ):
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "replica",
                    "physical_owner_scope_id",
                    "resource",
                    "telemetry",
                }
                or row.get("replica") != expected_replica
                or row.get("physical_owner_scope_id") != earliest.scope_id
                or not isinstance(row.get("resource"), str)
                or not row["resource"]
                or (
                    row.get("telemetry") is not None
                    and not isinstance(row.get("telemetry"), Mapping)
                )
            ):
                raise ValueError("compute-canary execution telemetry row is invalid")
    return copy.deepcopy(dict(value))


def execute_and_publish_role_neutral_stage1(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
    producer_factories: RoleNeutralProducerFactories,
    policy: RoleNeutralStage1ExecutionPolicy,
    executor: RoleNeutralPhysicalOwnerExecutor,
) -> dict[str, Any]:
    """Execute canonical owners and publish the authenticated all-ten gate."""

    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError("role-neutral execution requires a Stage1ScopePlan")
    if not isinstance(producer_factories, RoleNeutralProducerFactories):
        raise TypeError("role-neutral execution requires typed producer factories")
    if not isinstance(policy, RoleNeutralStage1ExecutionPolicy):
        raise TypeError("role-neutral execution requires a typed resource policy")
    execute = getattr(executor, "execute", None)
    if not callable(execute):
        raise TypeError("role-neutral execution requires a configured executor")
    factories = producer_factories.as_mapping()
    destination = _canonical_fresh_root(root)

    physical_order = tuple(plan.physical_execution_order)
    expected_physical_ids = tuple(owner.scope_id for owner in plan.physical_scopes)
    if len(physical_order) != len(set(physical_order)) or set(physical_order) != set(
        expected_physical_ids
    ):
        raise ValueError("scope plan physical execution order is incomplete")
    assigned_resources = assign_physical_fits(
        physical_order,
        policy.resource_plan,
    )
    groups = {owner.scope_id: (owner, members) for owner, members in plan.physical_scope_groups}
    effective_owner_concurrency = (
        1
        if policy.htr_operational_controls is not None
        else int(policy.max_parallel_owners)
    )
    owner_cpu_budget = max(
        1,
        int(policy.resource_plan.cpu_budget)
        // effective_owner_concurrency,
    )

    destination.mkdir(exist_ok=False)
    component_root = destination / ROLE_NEUTRAL_COMPONENT_DIRECTORY
    component_root.mkdir(exist_ok=False)
    tasks = tuple(
        RoleNeutralPhysicalOwnerTask(
            plan=plan,
            physical_owner=groups[owner_scope_id][0],
            logical_members=groups[owner_scope_id][1],
            component_parent=component_root / owner_scope_id,
            resource=assigned_resources[owner_scope_id],
            neural_query_execution_topology=(
                policy.neural_query_topology_for(
                    assigned_resources[owner_scope_id]
                )
            ),
            htr_operational_controls=policy.htr_operational_controls,
            neural_query_operational_controls=(
                policy.neural_query_operational_controls
            ),
            htr_fold_devices=(
                tuple(policy.resource_plan.devices)
                if policy.htr_operational_controls is not None
                else (assigned_resources[owner_scope_id],)
            ),
            owner_cpu_budget=owner_cpu_budget,
        )
        for owner_scope_id in physical_order
    )
    task_by_owner = {task.physical_owner.scope_id: task for task in tasks}

    claim_lock = threading.Lock()
    claimed: set[str] = set()
    canary_result: dict[str, Any] | None = None
    first_owner_validation_result: dict[str, Any] | None = None
    preexecuted_results: list[RoleNeutralPhysicalOwnerResult] = []
    canary_replica_results: tuple[
        RoleNeutralPhysicalOwnerResult,
        RoleNeutralPhysicalOwnerResult,
    ] | tuple[()] = ()
    process_isolated = (
        getattr(executor, "process_isolated_physical_owners", False) is True
    )

    def guarded_worker(
        task: RoleNeutralPhysicalOwnerTask,
    ) -> RoleNeutralPhysicalOwnerResult:
        if not isinstance(task, RoleNeutralPhysicalOwnerTask):
            raise TypeError("executor submitted an untyped physical-owner task")
        owner_scope_id = task.physical_owner.scope_id
        if owner_scope_id not in task_by_owner or task != task_by_owner[owner_scope_id]:
            raise ValueError("executor substituted a physical-owner task or capability")
        with claim_lock:
            if owner_scope_id in claimed:
                raise RuntimeError(f"physical owner was executed more than once: {owner_scope_id}")
            claimed.add(owner_scope_id)
        return _execute_one_owner(task=task, factories=factories)

    def parent_execution_forbidden(
        _task: RoleNeutralPhysicalOwnerTask,
    ) -> RoleNeutralPhysicalOwnerResult:
        raise RuntimeError(
            "process-isolated executor attempted to run its owner in the "
            "parent RNG process"
        )

    def execute_isolated_tasks(
        selected_tasks: Sequence[RoleNeutralPhysicalOwnerTask],
        *,
        max_workers: int,
    ) -> tuple[RoleNeutralPhysicalOwnerResult, ...]:
        rows = tuple(selected_tasks)
        raw = execute(
            tasks=rows,
            worker=parent_execution_forbidden,
            max_workers=int(max_workers),
            cpu_budget=int(policy.resource_plan.cpu_budget),
        )
        if isinstance(raw, (str, bytes, Mapping)):
            raise TypeError("process-isolated executor returned a non-sequence")
        results = tuple(raw)
        if len(results) != len(rows):
            raise ValueError(
                "process-isolated executor omitted or added a physical owner"
            )
        by_id = {
            result.physical_owner_scope_id: result for result in results
        }
        if len(by_id) != len(results) or set(by_id) != {
            task.physical_owner.scope_id for task in rows
        }:
            raise ValueError(
                "process-isolated executor substituted or duplicated an owner"
            )
        task_by_id = {
            task.physical_owner.scope_id: task for task in rows
        }
        return tuple(
            _freshly_reauthenticate_owner_result(
                task=task_by_id[result.physical_owner_scope_id],
                result=result,
            )
            for result in results
        )

    def run_productive_compute() -> Sequence[RoleNeutralPhysicalOwnerResult]:
        nonlocal canary_result, canary_replica_results
        nonlocal first_owner_validation_result

        executor_tasks = tasks
        gate_policy = policy.first_owner_validation
        if gate_policy is not None:
            selected_task = tasks[0]
            gate_path = _first_owner_gate_path(destination)
            gate_started_monotonic_ns = time.monotonic_ns()
            sampler = _FirstOwnerGpuSampler(
                devices=gate_policy.devices,
                interval_seconds=(
                    gate_policy.gpu_sample_interval_seconds
                ),
            )
            selected_result: RoleNeutralPhysicalOwnerResult | None = None
            scientific_replica: Mapping[str, Any] | None = None
            receipt_reauthentication: Mapping[str, Any] | None = None
            component_summary: Mapping[str, Any] | None = None
            memory_observation: Mapping[str, Any] | None = None
            failures: list[dict[str, str]] = []

            def record_failure(stage: str, exc: BaseException) -> None:
                failures.append(
                    {
                        "stage": stage,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

            try:
                with sampler:
                    if process_isolated:
                        selected_result = execute_isolated_tasks(
                            (selected_task,),
                            max_workers=1,
                        )[0]
                    else:
                        raw = execute(
                            tasks=(selected_task,),
                            worker=guarded_worker,
                            max_workers=1,
                            cpu_budget=int(
                                policy.resource_plan.cpu_budget
                            ),
                        )
                        first_rows = (
                            ()
                            if isinstance(
                                raw,
                                (str, bytes, Mapping),
                            )
                            else tuple(raw)
                        )
                        if (
                            isinstance(raw, (str, bytes, Mapping))
                            or len(first_rows) != 1
                        ):
                            raise ValueError(
                                "first-owner executor returned another result "
                                "cardinality"
                            )
                        only = first_rows[0]
                        selected_result = (
                            _freshly_reauthenticate_owner_result(
                                task=selected_task,
                                result=only,
                            )
                        )
            except BaseException as exc:
                record_failure("owner_execution", exc)

            if selected_result is not None:
                try:
                    receipt_reauthentication = (
                        _first_owner_receipt_reauthentication(
                            selected_result
                        )
                    )
                except BaseException as exc:
                    record_failure(
                        "component_receipt_reauthentication",
                        exc,
                    )
                try:
                    scientific_replica = (
                        _compute_canary_scientific_replica(
                            selected_result
                        )
                    )
                except BaseException as exc:
                    record_failure(
                        "scientific_and_text_coverage",
                        exc,
                    )
                try:
                    component_summary = (
                        _validate_first_owner_component_reports(
                            result=selected_result,
                            policy=policy,
                            gate=gate_policy,
                        )
                    )
                except BaseException as exc:
                    record_failure("component_parallelism", exc)
            try:
                memory_observation = _first_owner_memory_observation(
                    samples=sampler.samples,
                    sampling_errors=sampler.errors,
                    policy=gate_policy,
                )
                if memory_observation.get("accepted") is not True:
                    raise RuntimeError(
                        "first-owner GPU memory exceeded the deployment "
                        "safety policy"
                    )
            except BaseException as exc:
                record_failure("gpu_memory", exc)

            gate_finished_monotonic_ns = time.monotonic_ns()
            passed = not failures
            diagnostic_body = {
                "schema_version": (
                    ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_GATE_SCHEMA
                ),
                "status": "passed" if passed else "failed",
                "plan_scientific_content_sha256": (
                    plan.scientific_content_sha256
                ),
                "physical_owner_scope_id": (
                    selected_task.physical_owner.scope_id
                ),
                "physical_owner_canonical_index": int(
                    selected_task.physical_owner.canonical_index
                ),
                "canonical_fit_row_ids": list(
                    selected_task.physical_owner.fit_row_ids
                ),
                "canonical_heldout_row_ids": list(
                    selected_task.physical_owner.heldout_row_ids
                ),
                "canonical_group_seed": int(
                    selected_task.physical_owner.scope_seed
                ),
                "policy": gate_policy.as_dict(),
                "started_monotonic_ns": gate_started_monotonic_ns,
                "finished_monotonic_ns": gate_finished_monotonic_ns,
                "gpu_samples": [
                    copy.deepcopy(dict(row))
                    for row in sampler.samples
                ],
                "gpu_sampling_errors": [
                    copy.deepcopy(dict(row))
                    for row in sampler.errors
                ],
                "gpu_memory_observation": (
                    None
                    if memory_observation is None
                    else copy.deepcopy(dict(memory_observation))
                ),
                "component_execution_validation": (
                    None
                    if component_summary is None
                    else copy.deepcopy(dict(component_summary))
                ),
                "complete_scientific_owner_replica": (
                    None
                    if scientific_replica is None
                    else copy.deepcopy(dict(scientific_replica))
                ),
                "fresh_component_receipt_reauthentication": (
                    None
                    if receipt_reauthentication is None
                    else copy.deepcopy(
                        dict(receipt_reauthentication)
                    )
                ),
                "fresh_parent_reauthentication_completed": (
                    receipt_reauthentication is not None
                ),
                "complete_text_and_chunk_coverage_reauthenticated": (
                    receipt_reauthentication is not None
                    and receipt_reauthentication.get(
                        "complete_text_and_chunk_coverage_reauthenticated"
                    )
                    is True
                ),
                "complete_text_and_chunk_coverage_basis": (
                    None
                    if receipt_reauthentication is None
                    else receipt_reauthentication.get(
                        "coverage_reauthentication_basis"
                    )
                ),
                "owner_two_submitted_before_gate": False,
                "selected_owner_adopted_as_production_result": passed,
                "replica_b_executed": False,
                "failures": failures,
                "operational_gate_in_scientific_identity": False,
                "external_processes_killed": False,
            }
            diagnostic = {
                **diagnostic_body,
                "content_sha256": _sha256_json(diagnostic_body),
            }
            first_owner_validation_result = diagnostic
            _durably_publish_first_owner_gate(
                path=gate_path,
                value=diagnostic,
            )
            if not passed or selected_result is None:
                raise RuntimeError(
                    "first complete Stage 1 owner failed its hard "
                    f"validation gate before owner two: {gate_path}"
                )
            if process_isolated:
                with claim_lock:
                    claimed.add(
                        selected_task.physical_owner.scope_id
                    )
            preexecuted_results.append(selected_result)
            executor_tasks = tasks[1:]

        if process_isolated:
            results = execute_isolated_tasks(
                executor_tasks,
                max_workers=effective_owner_concurrency,
            )
            with claim_lock:
                claimed.update(
                    task.physical_owner.scope_id for task in executor_tasks
                )
            return results
        return execute(
            tasks=executor_tasks,
            worker=guarded_worker,
            max_workers=effective_owner_concurrency,
            cpu_budget=int(policy.resource_plan.cpu_budget),
        )

    open_session = getattr(executor, "open_session", None)
    if process_isolated and callable(open_session):
        session_marker_root = (
            destination / ".persistent-owner-execution-session"
        )
        with open_session(
            resources=tuple(policy.resource_plan.devices),
            max_workers=effective_owner_concurrency,
            cpu_budget=int(policy.resource_plan.cpu_budget),
            marker_root=session_marker_root,
        ) as persistent_session:
            execute = persistent_session.execute
            executed_results = run_productive_compute()
    else:
        executed_results = run_productive_compute()
    raw_results = tuple(preexecuted_results) + tuple(executed_results)
    if claimed != set(expected_physical_ids):
        raise ValueError("configured executor did not execute every physical owner")
    if process_isolated:
        # ``execute_isolated_tasks`` is the parent trust-boundary reader.  Its
        # results have already had every component tree reopened and
        # authenticated exactly once; replaying them here would multiply
        # ordinary reads without adding another trust boundary.
        authenticated_results = raw_results
    else:
        authenticated_results = tuple(
            _freshly_reauthenticate_owner_result(
                task=task_by_owner[result.physical_owner_scope_id],
                result=result,
            )
            for result in raw_results
        )
    sources_by_owner, completion_order = _normalize_executor_results(
        plan=plan,
        results=authenticated_results,
        assigned_resources=assigned_resources,
    )
    execution_telemetry = _execution_telemetry_summary(
        plan=plan,
        owner_results=authenticated_results,
        canary_replicas=canary_replica_results,
        process_isolated=process_isolated,
    )

    coordination_root = destination / ROLE_NEUTRAL_COORDINATION_DIRECTORY
    gate = publish_role_neutral_stage1_coordination_gate(
        root=coordination_root,
        plan=plan,
        sources_by_physical_owner=sources_by_owner,
    )
    if (
        validate_role_neutral_stage1_coordination_gate(
            root=coordination_root,
            plan=plan,
        )
        != gate
    ):
        raise RuntimeError("coordination gate changed immediately after publication")

    resource_attestation = copy.deepcopy(dict(policy.resource_plan.execution_attestation()))
    resource_body = {
        key: copy.deepcopy(value)
        for key, value in resource_attestation.items()
        if key != "content_sha256"
    }
    if resource_attestation.get("content_sha256") != _sha256_json(resource_body):
        raise ValueError("portable resource execution attestation is invalid")
    canary_registration: dict[str, Any] | None = None
    if canary_result is not None:
        canary_path = destination / ROLE_NEUTRAL_COMPUTE_CANARY_ATTESTATION
        canary_sha256, canary_size = _private_file_identity(canary_path)
        canary_registration = {
            "relative_path": ROLE_NEUTRAL_COMPUTE_CANARY_ATTESTATION,
            "sha256": canary_sha256,
            "size_bytes": canary_size,
            "content_sha256": canary_result["content_sha256"],
        }
    first_owner_validation_registration: dict[str, Any] | None = None
    if first_owner_validation_result is not None:
        first_owner_path = _first_owner_gate_path(destination)
        first_owner_sha256, first_owner_size = _private_file_identity(
            first_owner_path
        )
        first_owner_validation_registration = {
            "relative_path": (
                ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_GATE_SUFFIX
            ),
            "sha256": first_owner_sha256,
            "size_bytes": first_owner_size,
            "content_sha256": (
                first_owner_validation_result["content_sha256"]
            ),
        }
    attestation_body = {
        "schema_version": ROLE_NEUTRAL_STAGE1_EXECUTION_ATTESTATION_SCHEMA,
        "plan_scientific_content_sha256": plan.scientific_content_sha256,
        "resource_attestation": resource_attestation,
        "max_parallel_owners": int(policy.max_parallel_owners),
        "effective_max_parallel_owners": effective_owner_concurrency,
        "owner_cpu_budget": owner_cpu_budget,
        "effective_owner_concurrency_policy": (
            "htr_union_device_owner_serialization_v1"
            if policy.htr_operational_controls is not None
            else "configured_topology_capacity_v1"
        ),
        "cpu_budget": int(policy.resource_plan.cpu_budget),
        "submitted_owner_order": list(physical_order),
        "completed_owner_order": list(completion_order),
        "assigned_resources": {
            owner_scope_id: assigned_resources[owner_scope_id] for owner_scope_id in physical_order
        },
        "producer_component_order": list(EXPECTED_COMPONENT_FAMILIES),
        "physical_owner_execution_count": len(claimed),
        "producer_execution_count": (len(claimed) * len(EXPECTED_COMPONENT_FAMILIES)),
        "compute_canary": copy.deepcopy(canary_registration),
        "first_owner_validation": copy.deepcopy(
            first_owner_validation_registration
        ),
        "compute_canary_replica_execution_count": (
            2 if canary_result is not None else 0
        ),
        "compute_canary_additional_physical_execution_count": (
            1 if canary_result is not None else 0
        ),
        "total_producer_execution_count": (
            (len(claimed) + (1 if canary_result is not None else 0))
            * len(EXPECTED_COMPONENT_FAMILIES)
        ),
        "owner_execution_telemetry": execution_telemetry,
        "paths_devices_worker_count_in_scientific_identity": False,
        "external_processes_killed": False,
    }
    attestation = {
        **attestation_body,
        "content_sha256": _sha256_json(attestation_body),
    }
    attestation_path = destination / ROLE_NEUTRAL_EXECUTION_ATTESTATION
    _write_new_json(attestation_path, attestation)
    attestation_sha256, attestation_size = _private_file_identity(attestation_path)

    manifest_body = {
        "schema_version": ROLE_NEUTRAL_STAGE1_EXECUTION_SCHEMA,
        "status": "complete",
        "plan_scientific_content_sha256": plan.scientific_content_sha256,
        "scientific_identity": copy.deepcopy(gate["scientific_identity"]),
        "coordination_gate": {
            "relative_path": ROLE_NEUTRAL_COORDINATION_DIRECTORY,
            "manifest_content_sha256": gate["content_sha256"],
        },
        "execution_attestation": {
            "relative_path": ROLE_NEUTRAL_EXECUTION_ATTESTATION,
            "sha256": attestation_sha256,
            "size_bytes": attestation_size,
            "content_sha256": attestation["content_sha256"],
        },
        "physical_fit_count": len(plan.physical_scopes),
        "logical_scope_count": len(plan.scopes),
        "deduplicated_fit_count": len(plan.scopes) - len(plan.physical_scopes),
        "producer_component_count_per_physical_owner": len(EXPECTED_COMPONENT_FAMILIES),
        "every_physical_owner_executed_once": True,
        "every_component_executed_and_authenticated_once_per_owner": True,
        "productive_compute_canary_completed": canary_result is not None,
        "selected_canary_replica_adopted_as_production": (
            canary_result is not None
        ),
        "compute_canary_scientific_equality": (
            None
            if canary_result is None
            else canary_result["complete_scientific_artifacts_exactly_equal"]
        ),
        "owner_execution_telemetry": execution_telemetry,
        "coordination_gate_published_after_complete_execution": True,
        "legacy_bundle_build_invoked": False,
        "operational_metadata_excluded_from_scientific_identity": True,
    }
    manifest = {
        **manifest_body,
        "content_sha256": _sha256_json(manifest_body),
    }
    _write_new_json(
        destination / ROLE_NEUTRAL_EXECUTION_MANIFEST,
        manifest,
    )
    durably_sync_legacy_stage1_tree(destination)
    return validate_role_neutral_stage1_execution(
        root=destination,
        plan=plan,
    )


def _first_owner_policy_from_dict(
    value: Any,
) -> RoleNeutralFirstOwnerValidationPolicy:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "devices",
        "gpu_max_allocation_fraction",
        "gpu_minimum_headroom_bytes",
        "gpu_sample_interval_seconds",
        "required_tfidf_parallel_backend",
    }:
        raise ValueError("first-owner validation policy is malformed")
    policy = RoleNeutralFirstOwnerValidationPolicy(
        devices=tuple(value["devices"]),
        gpu_max_allocation_fraction=value[
            "gpu_max_allocation_fraction"
        ],
        gpu_minimum_headroom_bytes=value[
            "gpu_minimum_headroom_bytes"
        ],
        gpu_sample_interval_seconds=value[
            "gpu_sample_interval_seconds"
        ],
        required_tfidf_parallel_backend=value[
            "required_tfidf_parallel_backend"
        ],
        schema_version=value["schema_version"],
    )
    if policy.as_dict() != dict(value):
        raise ValueError("first-owner validation policy changed on reload")
    return policy


def _validate_first_owner_receipt_reauthentication(
    value: Any,
    *,
    physical_owner_scope_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "first-owner receipt reauthentication is missing"
        )
    result = copy.deepcopy(dict(value))
    body = {
        key: copy.deepcopy(child)
        for key, child in result.items()
        if key != "content_sha256"
    }
    expected_fields = {
        "schema_version",
        "physical_owner_scope_id",
        "canonical_component_order",
        "components",
        "component_root_count",
        "every_component_root_reopened_and_tree_rehashed",
        "every_component_terminal_reopened_and_content_hash_matched",
        "every_component_receipt_self_authenticated",
        "complete_text_and_chunk_coverage_reauthenticated",
        "coverage_reauthentication_basis",
        "content_sha256",
    }
    components = result.get("components")
    if (
        set(result) != expected_fields
        or result.get("schema_version")
        != (
            "production_role_neutral_first_owner_receipt_"
            "reauthentication_v1"
        )
        or result.get("physical_owner_scope_id")
        != physical_owner_scope_id
        or result.get("canonical_component_order")
        != list(EXPECTED_COMPONENT_FAMILIES)
        or not isinstance(components, list)
        or len(components) != len(EXPECTED_COMPONENT_FAMILIES)
        or result.get("component_root_count") != len(components)
        or result.get(
            "every_component_root_reopened_and_tree_rehashed"
        )
        is not True
        or result.get(
            "every_component_terminal_reopened_and_content_hash_matched"
        )
        is not True
        or result.get("every_component_receipt_self_authenticated")
        is not True
        or result.get(
            "complete_text_and_chunk_coverage_reauthenticated"
        )
        is not True
        or result.get("coverage_reauthentication_basis")
        != (
            "fresh_parent_component_tree_terminal_and_receipt_"
            "validation_v1"
        )
        or result.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError(
            "first-owner receipt reauthentication is invalid"
        )
    expected_component_fields = {
        "component",
        "component_authentication_content_sha256",
        "component_scientific_content_sha256",
        "component_execution_attestation_content_sha256",
        "source_terminal_content_sha256",
        "source_tree_sha256",
        "family_fit_artifact_sha256",
        "registered_heldout_labels_accessed",
        "oracle_fields_accessed",
        "text_truncation_applied",
        "lossy_evidence_selection_applied",
    }
    for component, row in zip(
        EXPECTED_COMPONENT_FAMILIES,
        components,
        strict=True,
    ):
        if not isinstance(row, Mapping):
            raise ValueError(
                "first-owner receipt component row is malformed"
            )
        scientific_sha256 = _require_sha256(
            row.get("component_scientific_content_sha256"),
            label=f"first-owner {component} scientific receipt",
        )
        tree_sha256 = _require_sha256(
            row.get("source_tree_sha256"),
            label=f"first-owner {component} source tree",
        )
        family_ids = row.get("family_fit_artifact_sha256")
        execution_body = {
            "schema_version": (
                "production_role_neutral_component_execution_"
                "attestation_v1"
            ),
            "component_scientific_content_sha256": scientific_sha256,
            "source_tree_sha256": tree_sha256,
        }
        if (
            set(row) != expected_component_fields
            or row.get("component") != component
            or not isinstance(family_ids, Mapping)
            or set(family_ids)
            != set(EXPECTED_COMPONENT_FAMILIES[component])
            or any(
                _require_sha256(
                    digest,
                    label=f"first-owner {component} family fit",
                )
                != digest
                for digest in family_ids.values()
            )
            or _require_sha256(
                row.get("component_authentication_content_sha256"),
                label=f"first-owner {component} authentication receipt",
            )
            != row.get("component_authentication_content_sha256")
            or _require_sha256(
                row.get("source_terminal_content_sha256"),
                label=f"first-owner {component} source terminal",
            )
            != row.get("source_terminal_content_sha256")
            or row.get(
                "component_execution_attestation_content_sha256"
            )
            != _sha256_json(execution_body)
            or row.get("registered_heldout_labels_accessed")
            is not False
            or row.get("oracle_fields_accessed") is not False
            or row.get("text_truncation_applied") is not False
            or row.get("lossy_evidence_selection_applied") is not False
        ):
            raise ValueError(
                f"first-owner {component} receipt binding is invalid"
            )
    return result


def _validate_first_owner_gate_attestation(
    *,
    path: Path,
    plan: Stage1ScopePlan,
    registration: Mapping[str, Any],
    selected_devices: Sequence[str],
) -> dict[str, Any]:
    expected_registration = {
        "relative_path",
        "sha256",
        "size_bytes",
        "content_sha256",
    }
    if (
        set(registration) != expected_registration
        or registration.get("relative_path")
        != ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_GATE_SUFFIX
    ):
        raise ValueError(
            "first-owner validation registration is malformed"
        )
    digest, size = _private_file_identity(path)
    value = _read_json(
        path,
        label="first-owner validation gate",
    )
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    expected_fields = {
        "schema_version",
        "status",
        "plan_scientific_content_sha256",
        "physical_owner_scope_id",
        "physical_owner_canonical_index",
        "canonical_fit_row_ids",
        "canonical_heldout_row_ids",
        "canonical_group_seed",
        "policy",
        "started_monotonic_ns",
        "finished_monotonic_ns",
        "gpu_samples",
        "gpu_sampling_errors",
        "gpu_memory_observation",
        "component_execution_validation",
        "complete_scientific_owner_replica",
        "fresh_component_receipt_reauthentication",
        "fresh_parent_reauthentication_completed",
        "complete_text_and_chunk_coverage_reauthenticated",
        "complete_text_and_chunk_coverage_basis",
        "owner_two_submitted_before_gate",
        "selected_owner_adopted_as_production_result",
        "replica_b_executed",
        "failures",
        "operational_gate_in_scientific_identity",
        "external_processes_killed",
        "content_sha256",
    }
    first_owner_id = plan.physical_execution_order[0]
    owners = {
        owner.scope_id: owner for owner in plan.physical_scopes
    }
    first_owner = owners[first_owner_id]
    policy = _first_owner_policy_from_dict(value.get("policy"))
    samples = value.get("gpu_samples")
    sampling_errors = value.get("gpu_sampling_errors")
    started = value.get("started_monotonic_ns")
    finished = value.get("finished_monotonic_ns")
    if (
        set(value) != expected_fields
        or value.get("schema_version")
        != ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_GATE_SCHEMA
        or value.get("status") != "passed"
        or value.get("plan_scientific_content_sha256")
        != plan.scientific_content_sha256
        or value.get("physical_owner_scope_id") != first_owner_id
        or value.get("physical_owner_canonical_index")
        != int(first_owner.canonical_index)
        or value.get("canonical_fit_row_ids")
        != list(first_owner.fit_row_ids)
        or value.get("canonical_heldout_row_ids")
        != list(first_owner.heldout_row_ids)
        or value.get("canonical_group_seed")
        != int(first_owner.scope_seed)
        or policy.devices
        != tuple(str(device) for device in selected_devices)
        or isinstance(started, bool)
        or not isinstance(started, int)
        or isinstance(finished, bool)
        or not isinstance(finished, int)
        or finished <= started
        or not isinstance(samples, list)
        or not samples
        or sampling_errors != []
        or value.get("fresh_parent_reauthentication_completed")
        is not True
        or value.get(
            "complete_text_and_chunk_coverage_reauthenticated"
        )
        is not True
        or value.get("complete_text_and_chunk_coverage_basis")
        != (
            "fresh_parent_component_tree_terminal_and_receipt_"
            "validation_v1"
        )
        or value.get("owner_two_submitted_before_gate") is not False
        or value.get(
            "selected_owner_adopted_as_production_result"
        )
        is not True
        or value.get("replica_b_executed") is not False
        or value.get("failures") != []
        or value.get("operational_gate_in_scientific_identity")
        is not False
        or value.get("external_processes_killed") is not False
        or value.get("content_sha256") != _sha256_json(body)
        or digest
        != _require_sha256(
            registration.get("sha256"),
            label="first-owner validation bytes",
        )
        or size != registration.get("size_bytes")
        or value.get("content_sha256")
        != _require_sha256(
            registration.get("content_sha256"),
            label="first-owner validation content",
        )
    ):
        raise ValueError("first-owner validation gate is invalid")
    memory = _first_owner_memory_observation(
        samples=samples,
        sampling_errors=sampling_errors,
        policy=policy,
    )
    if (
        memory.get("accepted") is not True
        or value.get("gpu_memory_observation") != memory
    ):
        raise ValueError(
            "first-owner validation memory evidence changed"
        )
    receipts = _validate_first_owner_receipt_reauthentication(
        value.get("fresh_component_receipt_reauthentication"),
        physical_owner_scope_id=first_owner_id,
    )
    replica = value.get("complete_scientific_owner_replica")
    if not isinstance(replica, Mapping):
        raise ValueError(
            "first-owner validation scientific replica is missing"
        )
    replica_body = {
        key: copy.deepcopy(child)
        for key, child in replica.items()
        if key != "content_sha256"
    }
    scientific_receipts = replica.get("component_scientific_receipts")
    receipt_rows = receipts["components"]
    from .portable_workflow_spec import EVIDENCE_FAMILIES

    portable_family_ids = {
        NATIVE_TO_PORTABLE_FAMILY[native_family]: digest
        for row in receipt_rows
        for native_family, digest in row[
            "family_fit_artifact_sha256"
        ].items()
    }
    if (
        set(replica)
        != {
            "schema_version",
            "physical_owner_scope_id",
            "canonical_component_order",
            "component_scientific_receipts",
            "family_artifact_ids",
            "all_ten_scientific_families_present",
            "resource_locator_included",
            "content_sha256",
        }
        or replica.get("schema_version")
        != "production_role_neutral_compute_canary_scientific_replica_v1"
        or replica.get("physical_owner_scope_id") != first_owner_id
        or replica.get("canonical_component_order")
        != list(EXPECTED_COMPONENT_FAMILIES)
        or not isinstance(scientific_receipts, list)
        or len(scientific_receipts) != len(receipt_rows)
        or set(portable_family_ids) != set(EVIDENCE_FAMILIES)
        or replica.get("family_artifact_ids")
        != {
            family: portable_family_ids[family]
            for family in EVIDENCE_FAMILIES
        }
        or replica.get("all_ten_scientific_families_present")
        is not True
        or replica.get("resource_locator_included") is not False
        or replica.get("content_sha256")
        != _sha256_json(replica_body)
        or any(
            not isinstance(scientific, Mapping)
            or scientific.get("component") != row["component"]
            or scientific.get("content_sha256")
            != row["component_scientific_content_sha256"]
            or scientific.get("source_terminal_content_sha256")
            != row["source_terminal_content_sha256"]
            or scientific.get("family_fit_artifact_sha256")
            != row["family_fit_artifact_sha256"]
            for scientific, row in zip(
                scientific_receipts,
                receipt_rows,
                strict=True,
            )
        )
    ):
        raise ValueError(
            "first-owner validation scientific receipt binding changed"
        )
    component_summary = value.get("component_execution_validation")
    if (
        not isinstance(component_summary, Mapping)
        or component_summary.get(
            "every_parallel_component_report_self_authenticated"
        )
        is not True
        or component_summary.get(
            "configured_parallel_work_did_not_serialize"
        )
        is not True
    ):
        raise ValueError(
            "first-owner validation component evidence is incomplete"
        )
    return value


def _validate_compute_canary_attestation(
    *,
    path: Path,
    plan: Stage1ScopePlan,
    registration: Mapping[str, Any],
    selected_devices: Sequence[str],
) -> dict[str, Any]:
    if set(registration) != {
        "relative_path",
        "sha256",
        "size_bytes",
        "content_sha256",
    } or registration.get("relative_path") != ROLE_NEUTRAL_COMPUTE_CANARY_ATTESTATION:
        raise ValueError("compute-canary registration is malformed")
    digest, size = _private_file_identity(path)
    value = _read_json(path, label="role-neutral compute-canary attestation")
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    expected_fields = {
        "schema_version",
        "plan_scientific_content_sha256",
        "physical_owner_scope_id",
        "canonical_scope_selection",
        "replica_resource_selection",
        "replica_resources",
        "replica_a_scientific_artifact",
        "replica_b_scientific_content_sha256",
        "complete_scientific_artifacts_exactly_equal",
        "selected_replica",
        "selected_replica_adopted_as_production_result",
        "third_fit_executed",
        "replica_b_model_tree_published_to_durable_storage",
        "resource_paths_and_devices_in_scientific_identity",
        "content_sha256",
    }
    replica = value.get("replica_a_scientific_artifact")
    if not isinstance(replica, Mapping):
        raise ValueError("compute-canary selected scientific replica is malformed")
    replica_body = {
        key: copy.deepcopy(child)
        for key, child in replica.items()
        if key != "content_sha256"
    }
    from .portable_workflow_spec import EVIDENCE_FAMILIES

    earliest_owner = min(
        plan.physical_scopes,
        key=lambda owner: int(owner.canonical_index),
    )
    resources = value.get("replica_resources")
    if (
        set(value) != expected_fields
        or value.get("schema_version") != ROLE_NEUTRAL_COMPUTE_CANARY_SCHEMA
        or value.get("plan_scientific_content_sha256")
        != plan.scientific_content_sha256
        or value.get("physical_owner_scope_id") != earliest_owner.scope_id
        or value.get("canonical_scope_selection")
        != EARLIEST_CANONICAL_OWNER_CANARY_SELECTION
        or value.get("replica_resource_selection")
        != DISTINCT_RESOURCE_CANARY_REPLICA_POLICY
        or not isinstance(resources, list)
        or len(resources) != 2
        or any(resource not in selected_devices for resource in resources)
        or (
            len(set(selected_devices)) > 1
            and resources[0] == resources[1]
        )
        or replica.get("physical_owner_scope_id") != earliest_owner.scope_id
        or set(replica.get("family_artifact_ids") or {})
        != set(EVIDENCE_FAMILIES)
        or replica.get("all_ten_scientific_families_present") is not True
        or replica.get("resource_locator_included") is not False
        or replica.get("content_sha256") != _sha256_json(replica_body)
        or value.get("replica_b_scientific_content_sha256")
        != replica.get("content_sha256")
        or value.get("complete_scientific_artifacts_exactly_equal") is not True
        or value.get("selected_replica") != "replica_a"
        or value.get("selected_replica_adopted_as_production_result") is not True
        or value.get("third_fit_executed") is not False
        or value.get("replica_b_model_tree_published_to_durable_storage")
        is not False
        or value.get("resource_paths_and_devices_in_scientific_identity")
        is not False
        or value.get("content_sha256") != _sha256_json(body)
        or digest
        != _require_sha256(
            registration.get("sha256"),
            label="compute-canary attestation bytes",
        )
        or size != registration.get("size_bytes")
        or value.get("content_sha256")
        != _require_sha256(
            registration.get("content_sha256"),
            label="compute-canary attestation content",
        )
    ):
        raise ValueError("role-neutral compute-canary attestation is invalid")
    return value


def validate_role_neutral_stage1_execution(
    *,
    root: Path | str,
    plan: Stage1ScopePlan,
) -> dict[str, Any]:
    """Freshly validate a complete execution and all retained component bytes."""

    tree = Path(root)
    if (
        not tree.is_absolute()
        or tree.is_symlink()
        or tree.resolve(strict=True) != tree
        or not tree.is_dir()
    ):
        raise ValueError("role-neutral execution root must be canonical")
    top_level = {path.name for path in tree.iterdir()}
    required_top_level = {
        ROLE_NEUTRAL_COMPONENT_DIRECTORY,
        ROLE_NEUTRAL_COORDINATION_DIRECTORY,
        ROLE_NEUTRAL_EXECUTION_ATTESTATION,
        ROLE_NEUTRAL_EXECUTION_MANIFEST,
    }
    optional_top_level = {
        ROLE_NEUTRAL_COMPUTE_CANARY_ATTESTATION,
        ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_GATE_SUFFIX,
    }
    if (
        not required_top_level.issubset(top_level)
        or not (top_level - required_top_level).issubset(
            optional_top_level
        )
    ):
        raise ValueError("role-neutral execution root has extra/missing data")
    manifest = _read_json(
        tree / ROLE_NEUTRAL_EXECUTION_MANIFEST,
        label="role-neutral execution manifest",
    )
    manifest_body = {
        key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"
    }
    expected_manifest_fields = {
        "schema_version",
        "status",
        "plan_scientific_content_sha256",
        "scientific_identity",
        "coordination_gate",
        "execution_attestation",
        "physical_fit_count",
        "logical_scope_count",
        "deduplicated_fit_count",
        "producer_component_count_per_physical_owner",
        "every_physical_owner_executed_once",
        "every_component_executed_and_authenticated_once_per_owner",
        "productive_compute_canary_completed",
        "selected_canary_replica_adopted_as_production",
        "compute_canary_scientific_equality",
        "owner_execution_telemetry",
        "coordination_gate_published_after_complete_execution",
        "legacy_bundle_build_invoked",
        "operational_metadata_excluded_from_scientific_identity",
        "content_sha256",
    }
    canary_completed = manifest.get("productive_compute_canary_completed")
    if (
        not isinstance(plan, Stage1ScopePlan)
        or set(manifest) != expected_manifest_fields
        or manifest.get("schema_version") != ROLE_NEUTRAL_STAGE1_EXECUTION_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("plan_scientific_content_sha256") != plan.scientific_content_sha256
        or manifest.get("physical_fit_count") != len(plan.physical_scopes)
        or manifest.get("logical_scope_count") != len(plan.scopes)
        or manifest.get("deduplicated_fit_count") != len(plan.scopes) - len(plan.physical_scopes)
        or manifest.get("producer_component_count_per_physical_owner")
        != len(EXPECTED_COMPONENT_FAMILIES)
        or manifest.get("every_physical_owner_executed_once") is not True
        or manifest.get("every_component_executed_and_authenticated_once_per_owner") is not True
        or type(canary_completed) is not bool
        or manifest.get("selected_canary_replica_adopted_as_production")
        is not canary_completed
        or manifest.get("compute_canary_scientific_equality")
        is not (True if canary_completed else None)
        or not isinstance(manifest.get("owner_execution_telemetry"), Mapping)
        or (
            (ROLE_NEUTRAL_COMPUTE_CANARY_ATTESTATION in top_level)
            is not canary_completed
        )
        or manifest.get("coordination_gate_published_after_complete_execution") is not True
        or manifest.get("legacy_bundle_build_invoked") is not False
        or manifest.get("operational_metadata_excluded_from_scientific_identity") is not True
        or manifest.get("content_sha256") != _sha256_json(manifest_body)
    ):
        raise ValueError("role-neutral execution manifest is invalid")

    gate_registration = manifest.get("coordination_gate")
    if (
        not isinstance(gate_registration, Mapping)
        or set(gate_registration) != {"relative_path", "manifest_content_sha256"}
        or gate_registration.get("relative_path") != ROLE_NEUTRAL_COORDINATION_DIRECTORY
    ):
        raise ValueError("coordination gate registration is malformed")
    gate = validate_role_neutral_stage1_coordination_gate(
        root=tree / ROLE_NEUTRAL_COORDINATION_DIRECTORY,
        plan=plan,
    )
    if gate.get("content_sha256") != _require_sha256(
        gate_registration.get("manifest_content_sha256"),
        label="coordination gate manifest",
    ) or manifest.get("scientific_identity") != gate.get("scientific_identity"):
        raise ValueError("registered coordination gate changed")

    attestation_registration = manifest.get("execution_attestation")
    if not isinstance(attestation_registration, Mapping) or set(attestation_registration) != {
        "relative_path",
        "sha256",
        "size_bytes",
        "content_sha256",
    }:
        raise ValueError("execution attestation registration is malformed")
    relative = PurePosixPath(str(attestation_registration.get("relative_path")))
    if relative.is_absolute() or relative.parts != (ROLE_NEUTRAL_EXECUTION_ATTESTATION,):
        raise ValueError("execution attestation path is noncanonical")
    attestation_path = tree / relative.as_posix()
    digest, size = _private_file_identity(attestation_path)
    attestation = _read_json(
        attestation_path,
        label="role-neutral execution attestation",
    )
    attestation_body = {
        key: copy.deepcopy(value) for key, value in attestation.items() if key != "content_sha256"
    }
    owner_ids = tuple(owner.scope_id for owner in plan.physical_scopes)
    expected_attestation_fields = {
        "schema_version",
        "plan_scientific_content_sha256",
        "resource_attestation",
        "max_parallel_owners",
        "effective_max_parallel_owners",
        "owner_cpu_budget",
        "effective_owner_concurrency_policy",
        "cpu_budget",
        "submitted_owner_order",
        "completed_owner_order",
        "assigned_resources",
        "producer_component_order",
        "physical_owner_execution_count",
        "producer_execution_count",
        "compute_canary",
        "first_owner_validation",
        "compute_canary_replica_execution_count",
        "compute_canary_additional_physical_execution_count",
        "total_producer_execution_count",
        "owner_execution_telemetry",
        "paths_devices_worker_count_in_scientific_identity",
        "external_processes_killed",
        "content_sha256",
    }
    if (
        set(attestation) != expected_attestation_fields
        or digest
        != _require_sha256(
            attestation_registration.get("sha256"),
            label="execution attestation bytes",
        )
        or size != attestation_registration.get("size_bytes")
        or attestation.get("content_sha256")
        != _require_sha256(
            attestation_registration.get("content_sha256"),
            label="execution attestation content",
        )
        or attestation.get("content_sha256") != _sha256_json(attestation_body)
        or attestation.get("schema_version") != ROLE_NEUTRAL_STAGE1_EXECUTION_ATTESTATION_SCHEMA
        or attestation.get("plan_scientific_content_sha256") != plan.scientific_content_sha256
        or attestation.get("submitted_owner_order") != list(plan.physical_execution_order)
        or set(attestation.get("completed_owner_order") or ()) != set(owner_ids)
        or len(attestation.get("completed_owner_order") or ()) != len(owner_ids)
        or set(attestation.get("assigned_resources") or {}) != set(owner_ids)
        or not isinstance(attestation.get("max_parallel_owners"), int)
        or not isinstance(
            attestation.get("effective_max_parallel_owners"),
            int,
        )
        or not isinstance(attestation.get("owner_cpu_budget"), int)
        or not isinstance(attestation.get("cpu_budget"), int)
        or int(attestation["max_parallel_owners"]) < 1
        or int(attestation["effective_max_parallel_owners"]) < 1
        or int(attestation["effective_max_parallel_owners"])
        > int(attestation["max_parallel_owners"])
        or int(attestation["cpu_budget"]) < int(attestation["max_parallel_owners"])
        or int(attestation["owner_cpu_budget"])
        != int(attestation["cpu_budget"])
        // int(attestation["effective_max_parallel_owners"])
        or attestation.get("effective_owner_concurrency_policy")
        not in {
            "configured_topology_capacity_v1",
            "htr_union_device_owner_serialization_v1",
        }
        or (
            attestation.get("effective_owner_concurrency_policy")
            == "configured_topology_capacity_v1"
            and int(attestation["effective_max_parallel_owners"])
            != int(attestation["max_parallel_owners"])
        )
        or (
            attestation.get("effective_owner_concurrency_policy")
            == "htr_union_device_owner_serialization_v1"
            and int(attestation["effective_max_parallel_owners"]) != 1
        )
        or attestation.get("producer_component_order") != list(EXPECTED_COMPONENT_FAMILIES)
        or attestation.get("physical_owner_execution_count") != len(owner_ids)
        or attestation.get("producer_execution_count")
        != len(owner_ids) * len(EXPECTED_COMPONENT_FAMILIES)
        or attestation.get("compute_canary_replica_execution_count")
        != (2 if canary_completed else 0)
        or attestation.get("compute_canary_additional_physical_execution_count")
        != (1 if canary_completed else 0)
        or attestation.get("total_producer_execution_count")
        != (
            len(owner_ids) + (1 if canary_completed else 0)
        )
        * len(EXPECTED_COMPONENT_FAMILIES)
        or (
            isinstance(attestation.get("compute_canary"), Mapping)
            is not canary_completed
        )
        or (
            isinstance(
                attestation.get("first_owner_validation"),
                Mapping,
            )
            is not (
                ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_GATE_SUFFIX
                in top_level
            )
        )
        or (
            canary_completed
            and attestation.get("first_owner_validation") is not None
        )
        or not isinstance(attestation.get("owner_execution_telemetry"), Mapping)
        or attestation.get("owner_execution_telemetry")
        != manifest.get("owner_execution_telemetry")
        or attestation.get("paths_devices_worker_count_in_scientific_identity") is not False
        or attestation.get("external_processes_killed") is not False
    ):
        raise ValueError("role-neutral execution attestation is invalid")
    _validate_execution_telemetry_summary(
        attestation["owner_execution_telemetry"],
        plan=plan,
        canary_completed=bool(canary_completed),
    )
    resource_attestation = attestation.get("resource_attestation")
    if not isinstance(resource_attestation, Mapping):
        raise ValueError("execution attestation lacks its resource report")
    resource_body = {
        key: copy.deepcopy(value)
        for key, value in resource_attestation.items()
        if key != "content_sha256"
    }
    selected_devices = resource_attestation.get("selected_devices")
    assigned_resources = attestation.get("assigned_resources")
    if (
        resource_attestation.get("content_sha256") != _sha256_json(resource_body)
        or resource_attestation.get("cpu_budget") != attestation.get("cpu_budget")
        or not isinstance(selected_devices, list)
        or not selected_devices
        or not isinstance(assigned_resources, Mapping)
        or any(
            assigned_resources[owner_scope_id] not in selected_devices
            for owner_scope_id in owner_ids
        )
    ):
        raise ValueError("resource execution attestation is invalid")
    if canary_completed:
        _validate_compute_canary_attestation(
            path=tree / ROLE_NEUTRAL_COMPUTE_CANARY_ATTESTATION,
            plan=plan,
            registration=attestation["compute_canary"],
            selected_devices=tuple(str(value) for value in selected_devices),
        )
    elif attestation.get("compute_canary") is not None:
        raise ValueError("disabled compute canary has a registration")
    validated_first_owner_gate: Mapping[str, Any] | None = None
    first_owner_registration = attestation.get(
        "first_owner_validation"
    )
    if isinstance(first_owner_registration, Mapping):
        validated_first_owner_gate = (
            _validate_first_owner_gate_attestation(
                path=(
                    tree
                    / ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_GATE_SUFFIX
                ),
                plan=plan,
                registration=first_owner_registration,
                selected_devices=tuple(
                    str(value) for value in selected_devices
                ),
            )
        )
    elif first_owner_registration is not None:
        raise ValueError(
            "first-owner validation registration is malformed"
        )

    locator = _read_json(
        tree / ROLE_NEUTRAL_COORDINATION_DIRECTORY / ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION,
        label="role-neutral component locator attestation",
    )
    expected_roots = {
        (
            owner.scope_id,
            component,
        ): (tree / ROLE_NEUTRAL_COMPONENT_DIRECTORY / owner.scope_id / component)
        for owner in plan.physical_scopes
        for component in EXPECTED_COMPONENT_FAMILIES
    }
    registrations = locator.get("registrations")
    if not isinstance(registrations, list):
        raise ValueError("component locator attestation is malformed")
    seen: set[tuple[str, str]] = set()
    first_owner_component_registrations: dict[
        str,
        Mapping[str, Any],
    ] = {}
    first_owner_id = plan.physical_execution_order[0]
    for registration in registrations:
        if not isinstance(registration, Mapping):
            raise ValueError("component locator registration is malformed")
        key = (
            str(registration.get("physical_owner_scope_id")),
            str(registration.get("component")),
        )
        if key not in expected_roots or key in seen:
            raise ValueError("component locator coverage changed")
        registered_root = Path(str(registration.get("absolute_root_locator")))
        if registered_root != expected_roots[key]:
            raise ValueError("component locator points outside the execution-owned tree")
        if key[0] == first_owner_id:
            first_owner_component_registrations[key[1]] = (
                copy.deepcopy(dict(registration))
            )
        seen.add(key)
    if seen != set(expected_roots):
        raise ValueError("component locator coverage is incomplete")
    if validated_first_owner_gate is not None:
        if set(first_owner_component_registrations) != set(
            EXPECTED_COMPONENT_FAMILIES
        ):
            raise ValueError(
                "first-owner validation lost a component registration"
            )
        receipt_summary = validated_first_owner_gate[
            "fresh_component_receipt_reauthentication"
        ]
        gate_rows = receipt_summary["components"]
        scientific_rows = validated_first_owner_gate[
            "complete_scientific_owner_replica"
        ]["component_scientific_receipts"]
        for component, gate_row, scientific_row in zip(
            EXPECTED_COMPONENT_FAMILIES,
            gate_rows,
            scientific_rows,
            strict=True,
        ):
            registration = first_owner_component_registrations[
                component
            ]
            registered_scientific = registration[
                "component_scientific_receipt"
            ]
            if (
                gate_row.get(
                    "component_authentication_content_sha256"
                )
                != registration.get(
                    "component_authentication_content_sha256"
                )
                or gate_row.get(
                    "component_scientific_content_sha256"
                )
                != registered_scientific.get("content_sha256")
                or gate_row.get("source_terminal_content_sha256")
                != registration.get("source_terminal_content_sha256")
                or gate_row.get("source_tree_sha256")
                != registration.get("source_tree_sha256")
                or gate_row.get("family_fit_artifact_sha256")
                != registered_scientific.get(
                    "family_fit_artifact_sha256"
                )
                or scientific_row != registered_scientific
            ):
                raise ValueError(
                    f"first-owner {component} receipt changed after its gate"
                )
        owner_telemetry_rows = attestation[
            "owner_execution_telemetry"
        ]["physical_owners"]
        first_owner_telemetry = next(
            (
                row.get("telemetry")
                for row in owner_telemetry_rows
                if row.get("physical_owner_scope_id")
                == first_owner_id
            ),
            None,
        )
        gate_policy = _first_owner_policy_from_dict(
            validated_first_owner_gate["policy"]
        )
        recomputed_component_summary = (
            _validate_first_owner_component_reports(
                execution_telemetry=first_owner_telemetry,
                gate=gate_policy,
            )
        )
        if recomputed_component_summary != validated_first_owner_gate.get(
            "component_execution_validation"
        ):
            raise ValueError(
                "first-owner component execution evidence changed "
                "after its gate"
            )
    return manifest


__all__ = [
    "BoundRoleNeutralComponentProducer",
    "DISTINCT_RESOURCE_CANARY_REPLICA_POLICY",
    "EARLIEST_CANONICAL_OWNER_CANARY_SELECTION",
    "ROLE_NEUTRAL_COMPONENT_DIRECTORY",
    "ROLE_NEUTRAL_COMPONENT_EXECUTION_CLOCK_DOMAIN",
    "ROLE_NEUTRAL_COMPONENT_EXECUTION_INTERVAL_SCHEMA",
    "ROLE_NEUTRAL_COMPUTE_CANARY_ATTESTATION",
    "ROLE_NEUTRAL_COMPUTE_CANARY_SCHEMA",
    "ROLE_NEUTRAL_COORDINATION_DIRECTORY",
    "ROLE_NEUTRAL_EXECUTION_ATTESTATION",
    "ROLE_NEUTRAL_EXECUTION_MANIFEST",
    "ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_GATE_SCHEMA",
    "ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_GATE_SUFFIX",
    "ROLE_NEUTRAL_FIRST_OWNER_VALIDATION_POLICY_SCHEMA",
    "ROLE_NEUTRAL_STAGE1_EXECUTION_ATTESTATION_SCHEMA",
    "ROLE_NEUTRAL_STAGE1_EXECUTION_SCHEMA",
    "RoleNeutralComponentInvocation",
    "RoleNeutralFirstOwnerValidationPolicy",
    "LocalThreadRoleNeutralPhysicalOwnerExecutor",
    "NeuralQueryExecutionTopology",
    "RoleNeutralPhysicalOwnerExecutor",
    "RoleNeutralPhysicalOwnerResult",
    "RoleNeutralPhysicalOwnerTask",
    "RoleNeutralOperationalComponentReport",
    "RoleNeutralProducerFactories",
    "RoleNeutralProducerFactory",
    "RoleNeutralStage1ExecutionPolicy",
    "execute_and_publish_role_neutral_stage1",
    "validate_role_neutral_component_execution_intervals",
    "validate_role_neutral_stage1_execution",
]
