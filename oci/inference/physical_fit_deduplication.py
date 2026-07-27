"""Content-derived Stage 1 logical-context grouping and physical-fit keys."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .portable_workflow_spec import EVIDENCE_FAMILIES, identity_sha256


PHYSICAL_FIT_KEY_SCHEMA = "portable_stage1_physical_fit_key_v2"
LOGICAL_CONTEXT_PLAN_SCHEMA = "portable_stage1_logical_context_plan_v2"
LOGICAL_BINDING_SCHEMA = "portable_stage1_logical_scope_binding_v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
RowId = str | int


def _validated_row_id_keys(
    row_ids: Sequence[Any],
    *,
    label: str,
    require_nonempty: bool,
) -> tuple[tuple[str, RowId], ...]:
    normalized = tuple(row_ids)
    if require_nonempty and not normalized:
        raise ValueError(f"{label} must be nonempty")
    tagged: list[tuple[str, RowId]] = []
    for value in normalized:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TypeError(
                f"{label} must contain only exact string or integer IDs"
            )
        tagged.append(
            (
                "integer" if isinstance(value, int) else "string",
                value,
            )
        )
    if len(tagged) != len(set(tagged)):
        raise ValueError(f"{label} must contain unique IDs")
    return tuple(tagged)


def ordered_row_identity(row_ids: Sequence[Any]) -> str:
    normalized = tuple(row_ids)
    _validated_row_id_keys(
        normalized,
        label="ordered fit rows",
        require_nonempty=True,
    )
    # Hash the original JSON scalars rather than their string projections.
    # In particular, integer row 1 and string row "1" are distinct inputs.
    return identity_sha256(list(normalized))


@dataclass(frozen=True)
class LogicalContext:
    canonical_index: int
    scope_id: str
    purpose: str
    outer_fold: int
    fit_row_ids: tuple[RowId, ...]
    heldout_row_ids: tuple[RowId, ...]
    architecture_identity: str
    target: str
    scientific_configuration_identity: str
    scope_seed: int
    producer_identity: str
    runtime_compatibility_class: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.canonical_index, bool)
            or int(self.canonical_index) != self.canonical_index
            or int(self.canonical_index) < 0
            or isinstance(self.outer_fold, bool)
            or int(self.outer_fold) != self.outer_fold
            or int(self.outer_fold) < 1
        ):
            raise ValueError("logical context indices must be valid")
        if not self.scope_id or not self.purpose or not self.target:
            raise ValueError("logical context labels are required")
        fit_keys = _validated_row_id_keys(
            self.fit_row_ids,
            label="logical fit rows",
            require_nonempty=True,
        )
        heldout_keys = _validated_row_id_keys(
            self.heldout_row_ids,
            label="logical held-out rows",
            require_nonempty=False,
        )
        if set(fit_keys) & set(heldout_keys):
            raise ValueError("logical fit and held-out rows overlap")
        if (
            isinstance(self.scope_seed, bool)
            or int(self.scope_seed) != self.scope_seed
            or int(self.scope_seed) < 0
        ):
            raise ValueError("logical scope seed must be nonnegative")
        for name in (
            "architecture_identity",
            "scientific_configuration_identity",
            "producer_identity",
        ):
            if _SHA256.fullmatch(str(getattr(self, name))) is None:
                raise ValueError(f"{name} must be one SHA-256")
        if not str(self.runtime_compatibility_class).strip():
            raise ValueError("runtime compatibility class must be nonempty")

    @property
    def equivalence_payload(self) -> Mapping[str, Any]:
        # MiniBatchKMeans, stochastic optimizers, and batched neural training
        # consume rows in order.  Set equality is therefore insufficient:
        # physical equivalence requires the exact canonical order.
        return {
            "architecture_identity": self.architecture_identity,
            "target": self.target,
            "fit_row_order_identity": ordered_row_identity(self.fit_row_ids),
            "fit_row_count": len(self.fit_row_ids),
            "scientific_configuration_identity": self.scientific_configuration_identity,
            "producer_identity": self.producer_identity,
            "runtime_compatibility_class": self.runtime_compatibility_class,
        }

    @property
    def equivalence_key(self) -> str:
        return identity_sha256(self.equivalence_payload)


@dataclass(frozen=True)
class PhysicalFitKey:
    architecture_identity: str
    target: str
    fit_row_order_identity: str
    scientific_configuration_identity: str
    canonical_group_seed: int
    producer_identity: str
    runtime_compatibility_class: str
    schema_version: str = PHYSICAL_FIT_KEY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PHYSICAL_FIT_KEY_SCHEMA:
            raise ValueError("unsupported physical-fit key schema")
        if not self.target or not self.runtime_compatibility_class:
            raise ValueError("physical-fit target and runtime class are required")
        if (
            isinstance(self.canonical_group_seed, bool)
            or int(self.canonical_group_seed) != self.canonical_group_seed
            or int(self.canonical_group_seed) < 0
        ):
            raise ValueError("canonical group seed must be nonnegative")
        for name in (
            "architecture_identity",
            "fit_row_order_identity",
            "scientific_configuration_identity",
            "producer_identity",
        ):
            if _SHA256.fullmatch(str(getattr(self, name))) is None:
                raise ValueError(f"{name} must be one SHA-256")

    @property
    def key(self) -> str:
        return identity_sha256(asdict(self))

    def as_dict(self) -> dict[str, Any]:
        """Return the closed key record with its content-derived identity."""

        body = asdict(self)
        return {**body, "content_sha256": self.key}


@dataclass(frozen=True)
class PhysicalFitGroup:
    key: PhysicalFitKey
    canonical_owner: LogicalContext
    logical_contexts: tuple[LogicalContext, ...]


def group_equivalent_contexts(
    contexts: Sequence[LogicalContext],
) -> tuple[PhysicalFitGroup, ...]:
    """Discover physical equivalence from content, never scope-name aliases."""

    if not contexts:
        raise ValueError("logical context plan cannot be empty")
    canonical_indices = [int(value.canonical_index) for value in contexts]
    scope_ids = [value.scope_id for value in contexts]
    if len(canonical_indices) != len(set(canonical_indices)):
        raise ValueError("logical context canonical indices are duplicated")
    if len(scope_ids) != len(set(scope_ids)):
        raise ValueError("logical context IDs are duplicated")
    grouped: dict[str, list[LogicalContext]] = {}
    for context in sorted(contexts, key=lambda value: value.canonical_index):
        grouped.setdefault(context.equivalence_key, []).append(context)
    output: list[PhysicalFitGroup] = []
    for members in grouped.values():
        owner = min(members, key=lambda value: value.canonical_index)
        key = PhysicalFitKey(
            architecture_identity=owner.architecture_identity,
            target=owner.target,
            fit_row_order_identity=ordered_row_identity(owner.fit_row_ids),
            scientific_configuration_identity=owner.scientific_configuration_identity,
            canonical_group_seed=int(owner.scope_seed),
            producer_identity=owner.producer_identity,
            runtime_compatibility_class=owner.runtime_compatibility_class,
        )
        for member in members:
            if member.equivalence_payload != owner.equivalence_payload:
                raise RuntimeError("physical-fit equivalence grouping changed")
            if int(member.scope_seed) != int(owner.scope_seed):
                raise ValueError(
                    "equivalent logical contexts must share one canonical group seed"
                )
        output.append(
            PhysicalFitGroup(
                key=key,
                canonical_owner=owner,
                logical_contexts=tuple(members),
            )
        )
    return tuple(
        sorted(output, key=lambda value: value.canonical_owner.canonical_index)
    )


def derive_logical_context_plan(
    *,
    outer_training_partitions: Mapping[int, Sequence[Sequence[Any]]],
    outer_heldout_rows: Mapping[int, Sequence[Any]],
    architecture_identity: str,
    target: str,
    scientific_configuration_identity: str,
    global_seed: int,
    producer_identity: str,
    runtime_compatibility_class: str,
    review_rounds: int,
) -> tuple[LogicalContext, ...]:
    """Derive full, exact-inner, and cumulative-review scopes from partitions."""

    contexts: list[LogicalContext] = []
    canonical_index = 0
    for outer_fold in sorted(outer_training_partitions):
        partitions = tuple(
            tuple(value for value in partition)
            for partition in outer_training_partitions[outer_fold]
        )
        if len(partitions) < 2 or any(not value for value in partitions):
            raise ValueError("each outer fold needs at least two nonempty partitions")
        flattened = tuple(value for partition in partitions for value in partition)
        flattened_keys = _validated_row_id_keys(
            flattened,
            label="outer training partitions",
            require_nonempty=True,
        )
        outer_heldout = tuple(outer_heldout_rows[outer_fold])
        outer_heldout_keys = _validated_row_id_keys(
            outer_heldout,
            label="outer held-out rows",
            require_nonempty=False,
        )
        if set(flattened_keys) & set(outer_heldout_keys):
            raise ValueError("outer training and held-out rows overlap")

        def append_context(
            *,
            scope_id: str,
            purpose: str,
            fit_rows: tuple[RowId, ...],
            heldout_rows: tuple[RowId, ...],
        ) -> None:
            nonlocal canonical_index
            seed_payload = {
                "schema_version": "portable_canonical_group_seed_v2",
                "global_seed": int(global_seed),
                "outer_fold": int(outer_fold),
                "ordered_fit_rows": list(fit_rows),
                "architecture_identity": architecture_identity,
                "target": target,
            }
            scope_seed = (
                int(identity_sha256(seed_payload)[:16], 16) % (2**31 - 1)
            ) or 1
            contexts.append(
                LogicalContext(
                    canonical_index=canonical_index,
                    scope_id=scope_id,
                    purpose=purpose,
                    outer_fold=int(outer_fold),
                    fit_row_ids=fit_rows,
                    heldout_row_ids=heldout_rows,
                    architecture_identity=architecture_identity,
                    target=target,
                    scientific_configuration_identity=(
                        scientific_configuration_identity
                    ),
                    scope_seed=scope_seed,
                    producer_identity=producer_identity,
                    runtime_compatibility_class=runtime_compatibility_class,
                )
            )
            canonical_index += 1

        prefix = f"outer_{int(outer_fold):03d}"
        append_context(
            scope_id=f"{prefix}_full",
            purpose="full_outer",
            fit_rows=flattened,
            heldout_rows=outer_heldout,
        )
        for inner_index, heldout_partition in enumerate(partitions, start=1):
            fit_rows = tuple(
                value
                for index, partition in enumerate(partitions, start=1)
                if index != inner_index
                for value in partition
            )
            append_context(
                scope_id=f"{prefix}_inner_{inner_index:03d}",
                purpose="exact_inner",
                fit_rows=fit_rows,
                heldout_rows=heldout_partition,
            )
        if int(review_rounds) >= len(partitions):
            raise ValueError("review rounds must leave at least one initial partition")
        initial_count = len(partitions) - int(review_rounds)
        for epoch in range(int(review_rounds)):
            fit_partition_count = initial_count + epoch
            fit_rows = tuple(
                value
                for partition in partitions[:fit_partition_count]
                for value in partition
            )
            heldout_rows = tuple(
                value
                for partition in partitions[fit_partition_count:]
                for value in partition
            )
            append_context(
                scope_id=f"{prefix}_hierarchy_epoch_{epoch:03d}",
                purpose=f"cumulative_review_epoch_{epoch}",
                fit_rows=fit_rows,
                heldout_rows=heldout_rows,
            )
    return tuple(contexts)


def build_logical_binding_records(
    *,
    groups: Sequence[PhysicalFitGroup],
    physical_artifact_ids: Mapping[str, str],
    physical_family_artifact_ids: Mapping[str, Mapping[str, str]],
) -> Mapping[str, Any]:
    """Bind all logical purposes to one physical all-ten result per group."""

    bindings: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []
    for group in groups:
        physical_key = group.key.key
        physical_artifact = physical_artifact_ids.get(physical_key)
        family_ids = physical_family_artifact_ids.get(physical_key)
        if not isinstance(physical_artifact, str) or len(physical_artifact) != 64:
            raise ValueError("physical fit lacks its sealed artifact identity")
        if not isinstance(family_ids, Mapping) or set(family_ids) != set(EVIDENCE_FAMILIES):
            raise ValueError("physical fit lacks exactly all ten evidence-family artifacts")
        if any(len(str(family_ids[name])) != 64 for name in EVIDENCE_FAMILIES):
            raise ValueError("physical evidence-family artifact identity is invalid")
        group_records.append(
            {
                "physical_fit_key": physical_key,
                "canonical_owner_scope_id": group.canonical_owner.scope_id,
                "physical_artifact_id": physical_artifact,
                "fit_row_order_identity": group.key.fit_row_order_identity,
                "canonical_group_seed": group.key.canonical_group_seed,
                "family_artifact_ids": {
                    name: family_ids[name] for name in EVIDENCE_FAMILIES
                },
                "logical_binding_count": len(group.logical_contexts),
            }
        )
        for context in group.logical_contexts:
            body = {
                "schema_version": LOGICAL_BINDING_SCHEMA,
                "scope_id": context.scope_id,
                "purpose": context.purpose,
                "outer_fold": context.outer_fold,
                "physical_fit_key": physical_key,
                "physical_artifact_id": physical_artifact,
                "canonical_owner_scope_id": group.canonical_owner.scope_id,
                "logical_fit_row_order_identity": ordered_row_identity(
                    context.fit_row_ids
                ),
                "logical_heldout_row_order_identity": identity_sha256(
                    list(context.heldout_row_ids)
                ),
                "family_artifact_ids": {
                    name: family_ids[name] for name in EVIDENCE_FAMILIES
                },
            }
            bindings.append({**body, "content_sha256": identity_sha256(body)})
    body = {
        "schema_version": LOGICAL_CONTEXT_PLAN_SCHEMA,
        "logical_context_count": len(bindings),
        "physical_fit_count": len(groups),
        "deduplicated_fit_count": len(bindings) - len(groups),
        "physical_groups": group_records,
        "logical_bindings": bindings,
        "all_ten_families_proven_equal_within_each_group": True,
    }
    return {**body, "content_sha256": identity_sha256(body)}


__all__ = [
    "LOGICAL_BINDING_SCHEMA",
    "LOGICAL_CONTEXT_PLAN_SCHEMA",
    "PHYSICAL_FIT_KEY_SCHEMA",
    "LogicalContext",
    "PhysicalFitGroup",
    "PhysicalFitKey",
    "build_logical_binding_records",
    "derive_logical_context_plan",
    "group_equivalent_contexts",
    "ordered_row_identity",
]
