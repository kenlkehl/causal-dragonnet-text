"""Lightweight deployment-only controls for role-neutral HTR execution.

This contract lives outside the HTR implementation so typed deployment
profiles can parse and authenticate it without importing Torch or model code.
The optimizer training batch is repeated only as a fail-closed binding to the
scientific HTR configuration; it is not an operational tuning parameter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .role_neutral_htr_group_execution import RoleNeutralHTRConfig


ROLE_NEUTRAL_HTR_OPERATIONAL_CONTROLS_SCHEMA = (
    "production_role_neutral_htr_operational_controls_v2"
)
ROLE_NEUTRAL_HTR_FOLD_RESOURCE_PLAN_SCHEMA = (
    "production_role_neutral_htr_fold_resource_plan_v1"
)
_CUDA_DEVICE = re.compile(r"^cuda:[0-9]+$")
_FOLD_PARALLEL_BACKENDS = frozenset({"threads", "processes"})


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _execution_devices(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("HTR fold devices must be one sequence")
    devices = tuple(str(value).strip().lower() for value in values)
    if (
        not devices
        or any(not value for value in devices)
        or len(devices) != len(set(devices))
        or ("cpu" in devices and devices != ("cpu",))
        or any(
            value != "cpu" and _CUDA_DEVICE.fullmatch(value) is None
            for value in devices
        )
    ):
        raise ValueError(
            "HTR fold devices must be one CPU device or unique explicit "
            "cuda:N devices"
        )
    return devices


@dataclass(frozen=True)
class RoleNeutralHTRFoldResourcePlan:
    """Runtime-only fold leases derived from deployment controls.

    ``fold_devices`` is a deterministic lease cycle, not scientific input.
    Each element represents one simultaneous fold slot. Later fold indices
    reuse that cycle only after the bounded executor has released a slot.
    """

    devices: tuple[str, ...]
    fold_parallelism: int
    fold_slots_per_device: int
    owner_cpu_budget: int
    fold_parallel_backend: str
    worker_cpu_threads: int = 1
    schema_version: str = ROLE_NEUTRAL_HTR_FOLD_RESOURCE_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ROLE_NEUTRAL_HTR_FOLD_RESOURCE_PLAN_SCHEMA:
            raise ValueError("unsupported role-neutral HTR fold resource plan")
        object.__setattr__(self, "devices", _execution_devices(self.devices))
        for name in (
            "fold_parallelism",
            "fold_slots_per_device",
            "owner_cpu_budget",
            "worker_cpu_threads",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(
                    getattr(self, name),
                    label=f"HTR fold resource {name}",
                ),
            )
        backend = str(self.fold_parallel_backend).strip().lower()
        if backend not in _FOLD_PARALLEL_BACKENDS:
            raise ValueError(
                "HTR fold parallel backend must be 'threads' or 'processes'"
            )
        object.__setattr__(self, "fold_parallel_backend", backend)
        if self.worker_cpu_threads != 1:
            raise ValueError(
                "HTR fold workers require one native CPU thread to prevent "
                "nested BLAS/OpenMP/tokenizer oversubscription"
            )
        slot_capacity = len(self.devices) * self.fold_slots_per_device
        if self.fold_parallelism > slot_capacity:
            raise ValueError(
                "HTR fold parallelism exceeds configured per-device slots"
            )
        if self.fold_parallelism > self.owner_cpu_budget:
            raise ValueError(
                "HTR fold parallelism exceeds the owner's global CPU lease"
            )
        if len(self.devices) > 1 and self.fold_parallelism < len(self.devices):
            raise ValueError(
                "HTR fold parallelism must schedule at least one fold on "
                "every selected device"
            )

    @property
    def fold_devices(self) -> tuple[str, ...]:
        ordered_slots = tuple(
            device
            for _slot in range(self.fold_slots_per_device)
            for device in self.devices
        )
        return ordered_slots[: self.fold_parallelism]

    def device_for_task(self, index: int) -> str:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("HTR fold task index must be nonnegative")
        leases = self.fold_devices
        return leases[index % len(leases)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "devices": list(self.devices),
            "fold_parallelism": self.fold_parallelism,
            "fold_slots_per_device": self.fold_slots_per_device,
            "owner_cpu_budget": self.owner_cpu_budget,
            "fold_parallel_backend": self.fold_parallel_backend,
            "worker_cpu_threads": self.worker_cpu_threads,
            "fold_devices": list(self.fold_devices),
            "scientific_identity_includes_resources": False,
        }


@dataclass(frozen=True)
class RoleNeutralHTROperationalControls:
    """Explicit non-scientific HTR execution controls.

    Cache capacities are allocation bounds, never selection limits. When
    reuse is enabled, the HTR executor proves every unique note and chunk fits
    before fitting; it never evicts evidence to satisfy the bound.
    """

    training_batch_size: int
    sentence_encoder_batch_size: int
    data_loader_workers: int
    fold_parallelism: int
    fold_parallel_backend: str
    fold_slots_per_device: int
    reuse_tokenizer_and_chunk_plans: bool
    chunk_plan_cache_max_entries: int
    tokenized_chunk_cache_max_entries: int
    schema_version: str = ROLE_NEUTRAL_HTR_OPERATIONAL_CONTROLS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ROLE_NEUTRAL_HTR_OPERATIONAL_CONTROLS_SCHEMA:
            raise ValueError("unsupported role-neutral HTR operational controls")
        for name in (
            "training_batch_size",
            "sentence_encoder_batch_size",
            "fold_parallelism",
            "fold_slots_per_device",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"HTR operational {name} must be positive")
        backend = str(self.fold_parallel_backend).strip().lower()
        if backend not in _FOLD_PARALLEL_BACKENDS:
            raise ValueError(
                "HTR operational fold_parallel_backend must be 'threads' "
                "or 'processes'"
            )
        object.__setattr__(self, "fold_parallel_backend", backend)
        if (
            isinstance(self.data_loader_workers, bool)
            or not isinstance(self.data_loader_workers, int)
            or self.data_loader_workers < 0
        ):
            raise ValueError(
                "HTR operational data_loader_workers must be nonnegative"
            )
        if not isinstance(self.reuse_tokenizer_and_chunk_plans, bool):
            raise TypeError(
                "HTR operational reuse_tokenizer_and_chunk_plans must be boolean"
            )
        for name in (
            "chunk_plan_cache_max_entries",
            "tokenized_chunk_cache_max_entries",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"HTR operational {name} must be nonnegative")
        capacities = (
            self.chunk_plan_cache_max_entries,
            self.tokenized_chunk_cache_max_entries,
        )
        if self.reuse_tokenizer_and_chunk_plans:
            if any(value < 1 for value in capacities):
                raise ValueError(
                    "reusable HTR plans require explicit positive cache capacities"
                )
        else:
            if any(value != 0 for value in capacities):
                raise ValueError(
                    "disabled reusable HTR plans require zero cache capacities"
                )
            if self.data_loader_workers != 0:
                raise ValueError(
                    "HTR data-loader workers execute reusable complete-text "
                    "plan work and therefore require reusable plans"
                )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "RoleNeutralHTROperationalControls":
        if not isinstance(value, Mapping):
            raise TypeError("HTR operational controls must be one mapping")
        required = set(cls.__dataclass_fields__)
        if set(value) != required:
            raise ValueError(
                "HTR operational controls must configure every field exactly; "
                f"missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        return cls(**dict(value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "training_batch_size": self.training_batch_size,
            "sentence_encoder_batch_size": self.sentence_encoder_batch_size,
            "data_loader_workers": self.data_loader_workers,
            "fold_parallelism": self.fold_parallelism,
            "fold_parallel_backend": self.fold_parallel_backend,
            "fold_slots_per_device": self.fold_slots_per_device,
            "reuse_tokenizer_and_chunk_plans": (
                self.reuse_tokenizer_and_chunk_plans
            ),
            "chunk_plan_cache_max_entries": (
                self.chunk_plan_cache_max_entries
            ),
            "tokenized_chunk_cache_max_entries": (
                self.tokenized_chunk_cache_max_entries
            ),
        }

    def validate_for(
        self,
        config: "RoleNeutralHTRConfig",
    ) -> "RoleNeutralHTROperationalControls":
        # Local import keeps portable deployment parsing free of model imports.
        from .role_neutral_htr_group_execution import RoleNeutralHTRConfig

        if not isinstance(config, RoleNeutralHTRConfig):
            raise TypeError("HTR operational controls require typed science")
        config.validated()
        if self.training_batch_size != config.batch_size:
            raise ValueError(
                "HTR optimizer training_batch_size is scientific and cannot "
                "be overridden by deployment controls"
            )
        return self

    def bind_fold_resources(
        self,
        *,
        devices: Sequence[str],
        owner_cpu_budget: int,
    ) -> RoleNeutralHTRFoldResourcePlan:
        """Bind resolved execution devices without changing science."""

        return RoleNeutralHTRFoldResourcePlan(
            devices=_execution_devices(devices),
            fold_parallelism=self.fold_parallelism,
            fold_slots_per_device=self.fold_slots_per_device,
            owner_cpu_budget=_positive_integer(
                owner_cpu_budget,
                label="HTR owner CPU budget",
            ),
            fold_parallel_backend=self.fold_parallel_backend,
        )


__all__ = [
    "ROLE_NEUTRAL_HTR_OPERATIONAL_CONTROLS_SCHEMA",
    "ROLE_NEUTRAL_HTR_FOLD_RESOURCE_PLAN_SCHEMA",
    "RoleNeutralHTRFoldResourcePlan",
    "RoleNeutralHTROperationalControls",
]
