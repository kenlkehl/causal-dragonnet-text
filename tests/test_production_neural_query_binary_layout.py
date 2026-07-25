from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from oci.inference.production_neural_query_binary_layout import (
    validate_npy_array_set,
    write_npy_array_set,
)


_ORDER = ("row_ids", "values", "names")


def _arrays() -> dict[str, np.ndarray]:
    return {
        "row_ids": np.asarray([4, 9], dtype=np.int64),
        "values": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "names": np.asarray(["first", "second"], dtype=str),
    }


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rewrite_authenticated_index(root: Path, mutate: object) -> None:
    path = root / "index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    mutate(index)
    body = {key: value for key, value in index.items() if key != "content_sha256"}
    index["content_sha256"] = _canonical_sha(body)
    path.write_text(
        json.dumps(
            index,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_per_array_layout_is_mmap_safe_ordered_and_immutable(tmp_path: Path) -> None:
    root = tmp_path / "arrays"
    descriptor = write_npy_array_set(root, _arrays(), ordered_names=_ORDER)
    validated, loaded = validate_npy_array_set(
        root,
        expected_order=_ORDER,
        expected_inventory=descriptor["array_inventory"],
    )

    assert validated == descriptor
    assert validated["array_order"] == list(_ORDER)
    assert all(isinstance(array, np.memmap) for array in loaded.values())
    np.testing.assert_array_equal(loaded["values"], _arrays()["values"])
    assert sorted(path.suffix for path in root.iterdir()) == [
        ".json",
        ".npy",
        ".npy",
        ".npy",
    ]
    with pytest.raises(FileExistsError, match="must not already exist"):
        write_npy_array_set(root, _arrays(), ordered_names=_ORDER)


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("missing", "missing, extra, or linked"),
        ("extra", "missing, extra, or linked"),
        ("tampered", "changed after emission"),
        ("symlink", "missing, extra, or linked"),
        ("hardlink", "hard-linked"),
    ),
)
def test_array_layout_rejects_member_substitution(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = tmp_path / mutation
    write_npy_array_set(root, _arrays(), ordered_names=_ORDER)
    first = root / "000_row_ids.npy"
    second = root / "001_values.npy"
    if mutation == "missing":
        first.unlink()
    elif mutation == "extra":
        np.save(root / "extra.npy", np.asarray([1]), allow_pickle=False)
    elif mutation == "tampered":
        first.write_bytes(first.read_bytes() + b"tamper")
    elif mutation == "symlink":
        first.unlink()
        first.symlink_to(second.name)
    elif mutation == "hardlink":
        first.unlink()
        os.link(second, first)
    else:  # pragma: no cover - parameter list is closed above
        raise AssertionError(mutation)

    with pytest.raises((ValueError, RuntimeError), match=match):
        validate_npy_array_set(root, expected_order=_ORDER)


def test_array_layout_rejects_reordered_index_even_when_reauthenticated(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reordered"
    write_npy_array_set(root, _arrays(), ordered_names=_ORDER)

    def reorder(index: dict[str, object]) -> None:
        index["array_order"] = list(reversed(index["array_order"]))
        index["arrays"] = list(reversed(index["arrays"]))

    _rewrite_authenticated_index(root, reorder)
    with pytest.raises(ValueError, match="reordered"):
        validate_npy_array_set(root, expected_order=_ORDER)


@pytest.mark.parametrize("field,replacement", (("dtype", "<f8"), ("shape", [99])))
def test_array_layout_rejects_reauthenticated_dtype_or_shape_drift(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    root = tmp_path / field
    write_npy_array_set(root, _arrays(), ordered_names=_ORDER)

    def drift(index: dict[str, object]) -> None:
        index["arrays"][0][field] = replacement

    _rewrite_authenticated_index(root, drift)
    with pytest.raises(RuntimeError, match="dtype, shape, or content"):
        validate_npy_array_set(root, expected_order=_ORDER)
