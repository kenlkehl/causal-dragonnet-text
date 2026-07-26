"""Deployment-only device topology for one learned-neural-query context.

The learned-query scientific request is device neutral.  This module carries
only the operational capability that tells an executor whether one context is
bound to one device or may distribute its three query banks over a compatible
device tuple.  Device names and topology never enter a scientific identity.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


NEURAL_QUERY_EXECUTION_TOPOLOGY_SCHEMA = (
    "deployment_neural_query_execution_topology_v1"
)
_DEVICE = re.compile(r"^(?:cpu|cuda:[0-9]+)$")


@dataclass(frozen=True)
class NeuralQueryExecutionTopology:
    """Exact operational devices reserved for one neural-query context."""

    devices: tuple[str, ...]
    schema_version: str = NEURAL_QUERY_EXECUTION_TOPOLOGY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != NEURAL_QUERY_EXECUTION_TOPOLOGY_SCHEMA:
            raise ValueError("unsupported neural-query execution topology schema")
        if isinstance(self.devices, (str, bytes)) or not isinstance(
            self.devices,
            Sequence,
        ):
            raise TypeError(
                "neural-query execution topology devices must be one sequence"
            )
        devices = tuple(str(value).strip() for value in self.devices)
        if not devices or any(_DEVICE.fullmatch(value) is None for value in devices):
            raise ValueError(
                "neural-query execution topology requires explicit cpu/cuda:N "
                "devices"
            )
        if len(devices) != len(set(devices)):
            raise ValueError(
                "neural-query execution topology devices cannot be duplicated"
            )
        if "cpu" in devices and devices != ("cpu",):
            raise ValueError(
                "CPU cannot be mixed with accelerator devices in one "
                "neural-query context"
            )
        object.__setattr__(self, "devices", devices)

    @classmethod
    def single(cls, device: str) -> "NeuralQueryExecutionTopology":
        return cls(devices=(str(device),))

    @property
    def primary_device(self) -> str:
        return self.devices[0]

    @property
    def spans_multiple_devices(self) -> bool:
        return len(self.devices) > 1

    def execution_payload(self) -> dict[str, Any]:
        """Operational payload; callers must not embed it in scientific IDs."""

        return {
            "schema_version": self.schema_version,
            "devices": list(self.devices),
            "primary_device": self.primary_device,
            "device_count": len(self.devices),
            "scientific_identity_includes_topology": False,
        }

    def scientific_payload(self) -> Mapping[str, Any]:
        """Constant proof that paths and device topology are excluded."""

        return copy.deepcopy(
            {
                "execution_device_topology_included": False,
                "device_assignment_policy": (
                    "round_robin_over_compatible_execution_devices_v1"
                ),
            }
        )


__all__ = [
    "NEURAL_QUERY_EXECUTION_TOPOLOGY_SCHEMA",
    "NeuralQueryExecutionTopology",
]
