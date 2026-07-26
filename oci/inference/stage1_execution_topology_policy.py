"""Deployment-only learned-query topology policy for Stage 1 execution.

The scientific learned-query request is device neutral.  This module maps a
closed deployment choice onto the explicit device tuples reserved by the
role-neutral executor.  It also computes the number of complete physical
owners that can actually run at once; a context spanning every selected
device must not be counted once per device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .neural_query_execution_topology import (
    NeuralQueryExecutionTopology,
)


STAGE1_EXECUTION_TOPOLOGY_POLICY_SCHEMA = (
    "portable_stage1_execution_topology_policy_v1"
)
ONE_CONTEXT_PER_SELECTED_DEVICE = "one_context_per_selected_device"
ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES = (
    "one_context_spanning_all_selected_devices"
)
SUPPORTED_STAGE1_EXECUTION_TOPOLOGY_MODES = frozenset(
    {
        ONE_CONTEXT_PER_SELECTED_DEVICE,
        ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES,
    }
)


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _selected_devices(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("selected execution devices must be one sequence")
    devices = tuple(str(value).strip() for value in values)
    if (
        not devices
        or any(not value for value in devices)
        or len(devices) != len(set(devices))
        or ("cpu" in devices and devices != ("cpu",))
    ):
        raise ValueError("selected execution devices are invalid")
    # The typed runtime topology performs the closed cpu/cuda:N validation.
    for device in devices:
        NeuralQueryExecutionTopology.single(device)
    return devices


@dataclass(frozen=True)
class Stage1ExecutionTopologyPolicy:
    """One explicit, deployment-only learned-query topology choice."""

    mode: str
    schema_version: str = STAGE1_EXECUTION_TOPOLOGY_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STAGE1_EXECUTION_TOPOLOGY_POLICY_SCHEMA:
            raise ValueError("unsupported Stage 1 execution topology policy")
        normalized = str(self.mode).strip()
        if normalized not in SUPPORTED_STAGE1_EXECUTION_TOPOLOGY_MODES:
            raise ValueError("unsupported Stage 1 execution topology mode")
        object.__setattr__(self, "mode", normalized)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "Stage1ExecutionTopologyPolicy":
        if not isinstance(value, Mapping):
            raise TypeError("Stage 1 execution topology policy must be one mapping")
        required = {"schema_version", "mode"}
        if set(value) != required:
            raise ValueError(
                "Stage 1 execution topology policy must configure every field "
                f"exactly; missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        return cls(
            schema_version=str(value["schema_version"]),
            mode=str(value["mode"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
        }

    def validate_selected_devices(
        self,
        devices: Sequence[str],
    ) -> tuple[str, ...]:
        selected = _selected_devices(devices)
        if (
            self.mode == ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES
            and (selected == ("cpu",) or len(selected) < 2)
        ):
            raise ValueError(
                "a spanning learned-query context requires at least two "
                "selected accelerator devices"
            )
        return selected

    def effective_parallel_owners_for_shape(
        self,
        *,
        resource_kind: str,
        device_count: int,
        workers_per_device: int,
    ) -> int:
        kind = str(resource_kind).strip()
        if kind not in {"cpu", "accelerator"}:
            raise ValueError(
                "Stage 1 topology resource_kind must be cpu or accelerator"
            )
        count = _positive_integer(
            device_count,
            label="Stage 1 topology device_count",
        )
        workers = _positive_integer(
            workers_per_device,
            label="Stage 1 workers_per_device",
        )
        if kind == "cpu" and count != 1:
            raise ValueError(
                "CPU Stage 1 topology requires exactly one device"
            )
        if self.mode == ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES:
            if kind != "accelerator" or count < 2:
                raise ValueError(
                    "a spanning learned-query context requires at least two "
                    "selected accelerator devices"
                )
            return workers
        return count * workers

    def effective_parallel_owners(
        self,
        *,
        devices: Sequence[str],
        workers_per_device: int,
    ) -> int:
        selected = self.validate_selected_devices(devices)
        return self.effective_parallel_owners_for_shape(
            resource_kind=(
                "cpu" if selected == ("cpu",) else "accelerator"
            ),
            device_count=len(selected),
            workers_per_device=workers_per_device,
        )

    def runtime_topologies(
        self,
        devices: Sequence[str],
    ) -> dict[str, NeuralQueryExecutionTopology]:
        selected = self.validate_selected_devices(devices)
        if self.mode == ONE_CONTEXT_PER_SELECTED_DEVICE:
            return {
                device: NeuralQueryExecutionTopology.single(device)
                for device in selected
            }
        return {
            primary: NeuralQueryExecutionTopology(
                devices=(
                    primary,
                    *(
                        device
                        for device in selected
                        if device != primary
                    ),
                )
            )
            for primary in selected
        }

    def scientific_payload(self) -> dict[str, bool]:
        """Constant proof that the topology does not affect scientific IDs."""

        return {
            "execution_topology_included_in_scientific_identity": False,
        }


__all__ = [
    "ONE_CONTEXT_PER_SELECTED_DEVICE",
    "ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES",
    "STAGE1_EXECUTION_TOPOLOGY_POLICY_SCHEMA",
    "SUPPORTED_STAGE1_EXECUTION_TOPOLOGY_MODES",
    "Stage1ExecutionTopologyPolicy",
]
