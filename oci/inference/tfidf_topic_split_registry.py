"""Audited explicit fold registries for exact-context TF-IDF workflows."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION = "tfidf_topic_explicit_split_registry_v1"


class SplitRegistryError(ValueError):
    """Raised when an explicit split registry fails its integrity contract."""


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_set_fingerprint(row_ids: Sequence[int]) -> str:
    return _stable_hash(sorted(str(value) for value in row_ids))


def _integer(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SplitRegistryError(f"{location} must be a JSON integer")
    return int(value)


def _row_ids(
    value: Any,
    *,
    location: str,
    dataset_row_count: int,
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise SplitRegistryError(f"{location} must be a non-empty JSON list")
    result = [
        _integer(item, location=f"{location}[{position}]") for position, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise SplitRegistryError(f"{location} contains duplicate row ids")
    invalid = [row_id for row_id in result if not 0 <= row_id < dataset_row_count]
    if invalid:
        raise SplitRegistryError(f"{location} contains out-of-range row ids: {invalid[:5]}")
    return result


def _partition(
    raw: Mapping[str, Any],
    *,
    location: str,
    universe: set[int],
    dataset_row_count: int,
) -> tuple[list[int], list[int]]:
    fit = _row_ids(
        raw.get("fit_row_ids"),
        location=f"{location}.fit_row_ids",
        dataset_row_count=dataset_row_count,
    )
    heldout = _row_ids(
        raw.get("heldout_row_ids"),
        location=f"{location}.heldout_row_ids",
        dataset_row_count=dataset_row_count,
    )
    fit_set = set(fit)
    heldout_set = set(heldout)
    if fit_set & heldout_set:
        raise SplitRegistryError(f"{location} fit and held-out rows overlap")
    if fit_set | heldout_set != universe:
        missing = sorted(universe - (fit_set | heldout_set))
        extra = sorted((fit_set | heldout_set) - universe)
        raise SplitRegistryError(
            f"{location} does not completely partition its parent rows; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return fit, heldout


def load_tfidf_topic_split_registry(
    path: Path | str,
    *,
    dataset_row_count: int,
    outer_fold_count: int,
    inner_fold_count: int,
) -> Dict[str, Any]:
    """Load, canonicalize, and fully validate an explicit split registry.

    Row order is retained because fitted topic-value matrices use the handoff's
    row order.  The returned content hash therefore changes if either membership
    or ordering changes, but not when the same registry is moved elsewhere.
    """
    requested_path = Path(path).expanduser()
    try:
        raw = json.loads(requested_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SplitRegistryError(f"TF-IDF split registry does not exist: {requested_path}") from exc
    except json.JSONDecodeError as exc:
        raise SplitRegistryError(
            f"TF-IDF split registry is not valid JSON: {requested_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise SplitRegistryError("TF-IDF split registry root must be a JSON object")
    if raw.get("schema_version") != TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION:
        raise SplitRegistryError(
            "Unsupported TF-IDF split registry schema: " f"{raw.get('schema_version')!r}"
        )

    n_rows = _integer(raw.get("dataset_row_count"), location="dataset_row_count")
    if n_rows != int(dataset_row_count):
        raise SplitRegistryError(
            "Split registry dataset_row_count does not match the current dataset: "
            f"{n_rows} != {int(dataset_row_count)}"
        )
    requested_outer = int(outer_fold_count)
    requested_inner = int(inner_fold_count)
    if requested_outer < 2:
        raise SplitRegistryError("An explicit split registry requires configured cv_folds >= 2")
    if requested_inner < 2:
        raise SplitRegistryError(
            "An explicit split registry requires at least two configured inner folds"
        )

    raw_outer = raw.get("outer_folds")
    if not isinstance(raw_outer, list) or len(raw_outer) != requested_outer:
        actual = len(raw_outer) if isinstance(raw_outer, list) else 0
        raise SplitRegistryError(
            "Split registry outer fold count does not match configured cv_folds: "
            f"{actual} != {requested_outer}"
        )
    all_rows = set(range(n_rows))
    outer_by_id: Dict[int, Dict[str, Any]] = {}
    outer_heldout_occurrences: Counter[int] = Counter()
    for position, raw_fold in enumerate(raw_outer):
        location = f"outer_folds[{position}]"
        if not isinstance(raw_fold, dict):
            raise SplitRegistryError(f"{location} must be a JSON object")
        outer_fold = _integer(raw_fold.get("outer_fold"), location=f"{location}.outer_fold")
        if outer_fold in outer_by_id:
            raise SplitRegistryError(f"Duplicate outer_fold={outer_fold}")
        fit, heldout = _partition(
            raw_fold,
            location=location,
            universe=all_rows,
            dataset_row_count=n_rows,
        )
        outer_heldout_occurrences.update(heldout)

        raw_inner = raw_fold.get("inner_folds")
        if not isinstance(raw_inner, list) or len(raw_inner) != requested_inner:
            actual = len(raw_inner) if isinstance(raw_inner, list) else 0
            raise SplitRegistryError(
                f"{location} inner fold count does not match configuration: "
                f"{actual} != {requested_inner}"
            )
        inner_by_id: Dict[int, Dict[str, Any]] = {}
        inner_heldout_occurrences: Counter[int] = Counter()
        outer_fit = set(fit)
        for inner_position, raw_inner_fold in enumerate(raw_inner):
            inner_location = f"{location}.inner_folds[{inner_position}]"
            if not isinstance(raw_inner_fold, dict):
                raise SplitRegistryError(f"{inner_location} must be a JSON object")
            inner_fold = _integer(
                raw_inner_fold.get("inner_fold"),
                location=f"{inner_location}.inner_fold",
            )
            if inner_fold in inner_by_id:
                raise SplitRegistryError(
                    f"Duplicate inner_fold={inner_fold} in outer_fold={outer_fold}"
                )
            inner_fit, inner_heldout = _partition(
                raw_inner_fold,
                location=inner_location,
                universe=outer_fit,
                dataset_row_count=n_rows,
            )
            inner_heldout_occurrences.update(inner_heldout)
            inner_by_id[inner_fold] = {
                "inner_fold": inner_fold,
                "fit_row_ids": inner_fit,
                "heldout_row_ids": inner_heldout,
            }
        expected_inner_ids = set(range(1, requested_inner + 1))
        if set(inner_by_id) != expected_inner_ids:
            raise SplitRegistryError(
                f"outer_fold={outer_fold} inner fold ids must be " f"1..{requested_inner}"
            )
        if set(inner_heldout_occurrences) != outer_fit or any(
            count != 1 for count in inner_heldout_occurrences.values()
        ):
            raise SplitRegistryError(
                f"outer_fold={outer_fold} inner held-out rows must cover each "
                "outer-fit row exactly once"
            )
        outer_by_id[outer_fold] = {
            "outer_fold": outer_fold,
            "fit_row_ids": fit,
            "heldout_row_ids": heldout,
            "inner_folds": [inner_by_id[index] for index in sorted(inner_by_id)],
        }

    expected_outer_ids = set(range(1, requested_outer + 1))
    if set(outer_by_id) != expected_outer_ids:
        raise SplitRegistryError(f"Outer fold ids must be exactly 1..{requested_outer}")
    if set(outer_heldout_occurrences) != all_rows or any(
        count != 1 for count in outer_heldout_occurrences.values()
    ):
        raise SplitRegistryError("Outer held-out rows must cover every dataset row exactly once")

    canonical = {
        "schema_version": TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
        "dataset_row_count": n_rows,
        "outer_folds": [outer_by_id[index] for index in sorted(outer_by_id)],
    }
    return {
        **canonical,
        "content_hash": _stable_hash(canonical),
        "source_path": str(requested_path.resolve()),
    }


def registry_outer_splits(registry: Mapping[str, Any]):
    """Return registry outer splits as row-position arrays."""
    return [
        (
            np.asarray(fold["fit_row_ids"], dtype=int),
            np.asarray(fold["heldout_row_ids"], dtype=int),
        )
        for fold in registry["outer_folds"]
    ]


def registry_inner_splits(
    registry: Mapping[str, Any],
    *,
    outer_fold: int,
    outer_fit_row_ids: Sequence[int],
):
    """Return registry inner splits as positions within an outer-fit frame."""
    fold = registry["outer_folds"][int(outer_fold) - 1]
    expected_outer_fit = list(map(int, fold["fit_row_ids"]))
    actual_outer_fit = list(map(int, outer_fit_row_ids))
    if actual_outer_fit != expected_outer_fit:
        raise SplitRegistryError(f"outer_fold={outer_fold} fit row order differs from the registry")
    local_position = {row_id: position for position, row_id in enumerate(actual_outer_fit)}
    return [
        (
            np.asarray([local_position[int(value)] for value in item["fit_row_ids"]], dtype=int),
            np.asarray(
                [local_position[int(value)] for value in item["heldout_row_ids"]],
                dtype=int,
            ),
        )
        for item in fold["inner_folds"]
    ]


def validate_handoff_rows_against_split_registry(
    rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> None:
    """Require every exact-context handoff row to match the registry exactly."""
    grouped: Dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    seen_fold_keys: set[int] = set()
    for row in rows:
        outer_fold = int(row.get("outer_fold"))
        fold_key = int(row.get("fold_key"))
        if fold_key in seen_fold_keys:
            raise SplitRegistryError(f"Duplicate handoff fold_key={fold_key}")
        seen_fold_keys.add(fold_key)
        grouped[outer_fold].append(row)

    expected_outer_ids = {int(item["outer_fold"]) for item in registry["outer_folds"]}
    if set(grouped) != expected_outer_ids:
        raise SplitRegistryError("Handoff outer folds do not match the explicit split registry")
    for fold in registry["outer_folds"]:
        outer_fold = int(fold["outer_fold"])
        fold_rows = grouped[outer_fold]
        full_rows = [row for row in fold_rows if row.get("scope") == "full_outer_train"]
        inner_rows = {
            int(row.get("inner_fold")): row
            for row in fold_rows
            if row.get("scope") == "candidate_selection_inner_fit"
        }
        if (
            len(full_rows) != 1
            or set(inner_rows) != {int(item["inner_fold"]) for item in fold["inner_folds"]}
            or len(fold_rows) != 1 + len(inner_rows)
        ):
            raise SplitRegistryError(
                f"Handoff exact-context set is incomplete for outer_fold={outer_fold}"
            )
        expected = [(full_rows[0], fold)] + [
            (inner_rows[int(item["inner_fold"])], item) for item in fold["inner_folds"]
        ]
        for row, split in expected:
            label = f"handoff fold_key={row.get('fold_key')}"
            for side in ("fit", "heldout"):
                expected_ids = list(map(int, split[f"{side}_row_ids"]))
                actual_ids = list(map(int, row.get(f"{side}_row_ids") or []))
                if actual_ids != expected_ids:
                    raise SplitRegistryError(
                        f"{label} {side} row ids/order differ from the registry"
                    )
                expected_fingerprint = _row_set_fingerprint(expected_ids)
                if row.get(f"{side}_row_fingerprint") != expected_fingerprint:
                    raise SplitRegistryError(
                        f"{label} {side} row fingerprint differs from the registry"
                    )
                discovery = row.get("discovery") or {}
                discovery_ids = list(map(int, discovery.get(f"{side}_row_ids") or []))
                if discovery_ids != expected_ids:
                    raise SplitRegistryError(
                        f"{label} discovery {side} row ids/order differ from the registry"
                    )
                if discovery.get(f"{side}_row_fingerprint") != expected_fingerprint:
                    raise SplitRegistryError(
                        f"{label} discovery {side} fingerprint differs from the registry"
                    )
