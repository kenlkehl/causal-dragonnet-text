"""Lightweight deployment-only controls for role-neutral HTR execution.

This contract lives outside the HTR implementation so typed deployment
profiles can parse and authenticate it without importing Torch or model code.
The optimizer training batch is repeated only as a fail-closed binding to the
scientific HTR configuration; it is not an operational tuning parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .role_neutral_htr_group_execution import RoleNeutralHTRConfig


ROLE_NEUTRAL_HTR_OPERATIONAL_CONTROLS_SCHEMA = (
    "production_role_neutral_htr_operational_controls_v1"
)


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
    reuse_tokenizer_and_chunk_plans: bool
    chunk_plan_cache_max_entries: int
    tokenized_chunk_cache_max_entries: int
    schema_version: str = ROLE_NEUTRAL_HTR_OPERATIONAL_CONTROLS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ROLE_NEUTRAL_HTR_OPERATIONAL_CONTROLS_SCHEMA:
            raise ValueError("unsupported role-neutral HTR operational controls")
        for name in ("training_batch_size", "sentence_encoder_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"HTR operational {name} must be positive")
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


__all__ = [
    "ROLE_NEUTRAL_HTR_OPERATIONAL_CONTROLS_SCHEMA",
    "RoleNeutralHTROperationalControls",
]
