"""Deployment-only controls for bounded learned-neural-query task execution.

The learned queries, folds, bank order, and seed formulas are scientific
configuration.  This module contains only the runtime resources used to
execute that fixed work.  Neither controls nor bound resource plans may be
embedded in a scientific request or artifact payload.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA = (
    "production_role_neutral_neural_query_operational_controls_v1"
)
ROLE_NEUTRAL_NEURAL_QUERY_TASK_RESOURCE_PLAN_SCHEMA = (
    "production_role_neutral_neural_query_task_resource_plan_v1"
)
_CUDA_DEVICE = re.compile(r"^cuda:[0-9]+$")
_PARALLEL_BACKENDS = frozenset({"threads", "processes"})


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _execution_devices(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("neural-query task devices must be one sequence")
    devices = tuple(str(value).strip().lower() for value in values)
    if (
        not devices
        or len(devices) != len(set(devices))
        or ("cpu" in devices and devices != ("cpu",))
        or any(
            value != "cpu" and _CUDA_DEVICE.fullmatch(value) is None
            for value in devices
        )
    ):
        raise ValueError(
            "neural-query task devices must be one CPU device or unique "
            "explicit cuda:N devices"
        )
    return devices


@dataclass(frozen=True)
class RoleNeutralNeuralQueryTaskResourcePlan:
    """Runtime-only leases for inner-fold and independent-bank task phases."""

    devices: tuple[str, ...]
    inner_fold_parallelism: int
    fold_parallel_backend: str
    fold_slots_per_device: int
    bank_parallelism: int
    worker_cpu_threads: int
    owner_cpu_budget: int
    schema_version: str = ROLE_NEUTRAL_NEURAL_QUERY_TASK_RESOURCE_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != ROLE_NEUTRAL_NEURAL_QUERY_TASK_RESOURCE_PLAN_SCHEMA
        ):
            raise ValueError(
                "unsupported role-neutral neural-query task resource plan"
            )
        object.__setattr__(self, "devices", _execution_devices(self.devices))
        for name in (
            "inner_fold_parallelism",
            "fold_slots_per_device",
            "bank_parallelism",
            "worker_cpu_threads",
            "owner_cpu_budget",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(
                    getattr(self, name),
                    label=f"neural-query task resource {name}",
                ),
            )
        backend = str(self.fold_parallel_backend).strip().lower()
        if backend not in _PARALLEL_BACKENDS:
            raise ValueError(
                "neural-query fold parallel backend must be 'threads' or "
                "'processes'"
            )
        object.__setattr__(self, "fold_parallel_backend", backend)
        if self.worker_cpu_threads != 1:
            raise ValueError(
                "neural-query workers require one native CPU thread to "
                "prevent nested Torch/BLAS/tokenizer oversubscription"
            )
        slot_capacity = len(self.devices) * self.fold_slots_per_device
        if self.inner_fold_parallelism > slot_capacity:
            raise ValueError(
                "neural-query inner-fold parallelism exceeds configured "
                "per-device slots"
            )
        if self.bank_parallelism > slot_capacity:
            raise ValueError(
                "neural-query bank parallelism exceeds configured "
                "per-device slots"
            )
        maximum_workers = max(
            self.inner_fold_parallelism,
            self.bank_parallelism,
        )
        if maximum_workers * self.worker_cpu_threads > self.owner_cpu_budget:
            raise ValueError(
                "neural-query task CPU threads exceed the owner's global "
                "CPU lease"
            )
        if len(self.devices) > 1 and (
            self.inner_fold_parallelism < len(self.devices)
            or self.bank_parallelism < len(self.devices)
        ):
            raise ValueError(
                "neural-query task parallelism must make every selected "
                "device schedulable in both task phases"
            )
        if (
            self.devices != ("cpu",)
            and maximum_workers > 1
            and backend != "processes"
        ):
            raise ValueError(
                "parallel CUDA neural-query tasks require spawned processes"
            )

    @property
    def slot_devices(self) -> tuple[str, ...]:
        """Canonical slot cycle, allowing multiple slots on each device."""

        return tuple(
            device
            for _slot in range(self.fold_slots_per_device)
            for device in self.devices
        )

    def devices_for_parallelism(self, parallelism: int) -> tuple[str, ...]:
        count = _positive_integer(
            parallelism,
            label="neural-query task parallelism",
        )
        if count > len(self.slot_devices):
            raise ValueError(
                "neural-query task parallelism exceeds available slots"
            )
        return self.slot_devices[:count]

    @property
    def inner_fold_devices(self) -> tuple[str, ...]:
        return self.devices_for_parallelism(self.inner_fold_parallelism)

    @property
    def bank_devices(self) -> tuple[str, ...]:
        return self.devices_for_parallelism(self.bank_parallelism)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "devices": list(self.devices),
            "inner_fold_parallelism": self.inner_fold_parallelism,
            "fold_parallel_backend": self.fold_parallel_backend,
            "fold_slots_per_device": self.fold_slots_per_device,
            "bank_parallelism": self.bank_parallelism,
            "worker_cpu_threads": self.worker_cpu_threads,
            "owner_cpu_budget": self.owner_cpu_budget,
            "inner_fold_devices": list(self.inner_fold_devices),
            "bank_devices": list(self.bank_devices),
            "scientific_identity_includes_resources": False,
        }


@dataclass(frozen=True)
class RoleNeutralNeuralQueryOperationalControls:
    """Required deployment controls for one learned-query physical owner."""

    inner_fold_parallelism: int
    fold_parallel_backend: str
    fold_slots_per_device: int
    bank_parallelism: int
    worker_cpu_threads: int
    schema_version: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA
        ):
            raise ValueError(
                "unsupported role-neutral neural-query operational controls"
            )
        for name in (
            "inner_fold_parallelism",
            "fold_slots_per_device",
            "bank_parallelism",
            "worker_cpu_threads",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(
                    getattr(self, name),
                    label=f"neural-query operational {name}",
                ),
            )
        backend = str(self.fold_parallel_backend).strip().lower()
        if backend not in _PARALLEL_BACKENDS:
            raise ValueError(
                "neural-query operational fold_parallel_backend must be "
                "'threads' or 'processes'"
            )
        object.__setattr__(self, "fold_parallel_backend", backend)
        if self.worker_cpu_threads != 1:
            raise ValueError(
                "neural-query operational worker_cpu_threads must be one"
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "RoleNeutralNeuralQueryOperationalControls":
        if not isinstance(value, Mapping):
            raise TypeError(
                "neural-query operational controls must be one mapping"
            )
        required = set(cls.__dataclass_fields__)
        if set(value) != required:
            raise ValueError(
                "neural-query operational controls must configure every "
                "field exactly; "
                f"missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        return cls(**dict(value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "inner_fold_parallelism": self.inner_fold_parallelism,
            "fold_parallel_backend": self.fold_parallel_backend,
            "fold_slots_per_device": self.fold_slots_per_device,
            "bank_parallelism": self.bank_parallelism,
            "worker_cpu_threads": self.worker_cpu_threads,
        }

    def bind_task_resources(
        self,
        *,
        devices: Sequence[str],
        owner_cpu_budget: int,
    ) -> RoleNeutralNeuralQueryTaskResourcePlan:
        """Bind selected devices and the owner's already-reserved CPU lease."""

        return RoleNeutralNeuralQueryTaskResourcePlan(
            devices=_execution_devices(devices),
            inner_fold_parallelism=self.inner_fold_parallelism,
            fold_parallel_backend=self.fold_parallel_backend,
            fold_slots_per_device=self.fold_slots_per_device,
            bank_parallelism=self.bank_parallelism,
            worker_cpu_threads=self.worker_cpu_threads,
            owner_cpu_budget=_positive_integer(
                owner_cpu_budget,
                label="neural-query owner CPU budget",
            ),
        )


__all__ = [
    "ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA",
    "ROLE_NEUTRAL_NEURAL_QUERY_TASK_RESOURCE_PLAN_SCHEMA",
    "RoleNeutralNeuralQueryOperationalControls",
    "RoleNeutralNeuralQueryTaskResourcePlan",
]
