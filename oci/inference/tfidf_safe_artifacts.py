"""Closed, pickle-free artifacts for fitted TF-IDF topic contexts.

The production TF-IDF path needs to replay fitted scikit-learn objects, but a
generic object serializer is both unsafe and scientifically opaque.  This
module supports only the estimator classes constructed by
``tfidf_topic_discovery`` and writes every numerical value as an individual
``.npy`` payload.  A canonical JSON index records an ordered, hashed inventory
and the explicit reconstruction recipe.

Readers authenticate the complete directory before constructing an estimator.
Unknown estimators, unknown fields, missing/extra/reordered payloads, linked
files, dtype/shape changes, and incompatible numerical runtimes fail closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import NMF
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
    _tree,
)

from ..config import BoWViewConfig


FITTED_CONTEXT_SCHEMA_VERSION = "tfidf_fitted_context_safe_arrays_v3"
ARRAY_BANK_SCHEMA_VERSION = "tfidf_named_array_bank_v1"
INDEX_FILENAME = "index.json"

_FOREST_CLASSES = {
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
}
_TREE_CLASSES = {
    "ExtraTreeClassifier": ExtraTreeClassifier,
    "ExtraTreeRegressor": ExtraTreeRegressor,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "DecisionTreeRegressor": DecisionTreeRegressor,
}
_LINEAR_CLASSES = {
    "LogisticRegression": LogisticRegression,
    "Ridge": Ridge,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _runtime_compatibility() -> dict[str, str]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
        "scikit_learn": str(sklearn.__version__),
    }


def _json_scalar(value: Any, *, label: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{label} is not finite")
        return value
    if isinstance(value, tuple):
        return [_json_scalar(item, label=f"{label}[]") for item in value]
    if isinstance(value, list):
        return [_json_scalar(item, label=f"{label}[]") for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{label} contains a non-string mapping key")
        return {
            key: _json_scalar(item, label=f"{label}.{key}")
            for key, item in sorted(value.items())
        }
    raise TypeError(f"{label} contains unsupported value type {type(value).__name__}")


def _closed_mapping(
    value: Any,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    keys = set(value)
    missing = sorted(required - keys)
    extra = sorted(keys - required - optional)
    if missing or extra:
        raise ValueError(f"{label} has missing={missing} extra={extra}")
    return value


def _dtype_descriptor(dtype: np.dtype[Any]) -> Any:
    return _json_scalar(np.lib.format.dtype_to_descr(np.dtype(dtype)), label="dtype")


def _safe_artifact_root(index_path: Path) -> Path:
    index_path = Path(os.path.abspath(Path(index_path)))
    if index_path.name != INDEX_FILENAME:
        raise ValueError(f"TF-IDF safe artifact must reference {INDEX_FILENAME}")
    try:
        resolved = index_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("TF-IDF safe artifact index is missing") from exc
    if resolved != index_path:
        raise ValueError("TF-IDF safe artifact cannot traverse a symlink component")
    root = index_path.parent
    if root.is_symlink() or index_path.is_symlink():
        raise ValueError("TF-IDF safe artifact cannot contain symlink components")
    if not root.is_dir() or not index_path.is_file():
        raise ValueError("TF-IDF safe artifact index is missing")
    return root


class _ArrayWriter:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if self.root.exists():
            if self.root.is_symlink() or not self.root.is_dir():
                raise ValueError("TF-IDF artifact root must be a new directory")
            if any(self.root.iterdir()):
                raise FileExistsError(f"TF-IDF artifact root is not empty: {self.root}")
        else:
            self.root.mkdir(parents=True)
        self.inventory: list[dict[str, Any]] = []
        self._counter = 0

    def add(self, label: str, value: Any) -> dict[str, Any]:
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"{label} cannot use object dtype")
        if array.dtype.kind not in "biufcSUV":
            raise TypeError(f"{label} has unsupported dtype {array.dtype}")
        self._counter += 1
        safe_label = "".join(
            character if character.isalnum() or character in "_-" else "_"
            for character in str(label)
        ).strip("_")
        if not safe_label:
            safe_label = "array"
        relative = f"{self._counter:05d}_{safe_label}.npy"
        path = self.root / relative
        with path.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        digest, size = _sha256_file(path)
        entry = {
            "relative_path": relative,
            "size_bytes": int(size),
            "sha256": digest,
            "dtype": _dtype_descriptor(array.dtype),
            "shape": [int(item) for item in array.shape],
        }
        self.inventory.append(entry)
        return {"array": relative}

    def finish(self, *, artifact_kind: str, schema_version: str, state: Any) -> Path:
        body = {
            "artifact_kind": str(artifact_kind),
            "schema_version": str(schema_version),
            "runtime_compatibility": _runtime_compatibility(),
            "state": state,
            "payload_inventory": self.inventory,
        }
        manifest = {**body, "content_sha256": _sha256_bytes(_canonical_bytes(body))}
        index_path = self.root / INDEX_FILENAME
        payload = _canonical_bytes(manifest)
        with index_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return index_path


class _ArrayReader:
    def __init__(
        self,
        index_path: Path | str,
        *,
        artifact_kind: str,
        schema_version: str,
    ) -> None:
        requested_index = Path(index_path)
        self.root = _safe_artifact_root(requested_index)
        self.index_path = self.root / INDEX_FILENAME
        index_stat = self.index_path.lstat()
        if not stat.S_ISREG(index_stat.st_mode) or index_stat.st_nlink != 1:
            raise ValueError("TF-IDF safe artifact index must be one regular, unlinked file")
        try:
            manifest = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("TF-IDF safe artifact index is not valid UTF-8 JSON") from exc
        manifest = _closed_mapping(
            manifest,
            required={
                "artifact_kind",
                "schema_version",
                "runtime_compatibility",
                "state",
                "payload_inventory",
                "content_sha256",
            },
            label="TF-IDF safe artifact index",
        )
        body = {
            key: manifest[key]
            for key in (
                "artifact_kind",
                "schema_version",
                "runtime_compatibility",
                "state",
                "payload_inventory",
            )
        }
        if manifest["content_sha256"] != _sha256_bytes(_canonical_bytes(body)):
            raise ValueError("TF-IDF safe artifact index content hash mismatch")
        if manifest["artifact_kind"] != artifact_kind:
            raise ValueError("TF-IDF safe artifact kind mismatch")
        if manifest["schema_version"] != schema_version:
            raise ValueError("TF-IDF safe artifact schema mismatch")
        if manifest["runtime_compatibility"] != _runtime_compatibility():
            raise ValueError("TF-IDF safe artifact runtime compatibility mismatch")
        raw_inventory = manifest["payload_inventory"]
        if not isinstance(raw_inventory, list):
            raise ValueError("TF-IDF safe artifact inventory must be a list")
        self.inventory: list[Mapping[str, Any]] = []
        self.by_path: dict[str, Mapping[str, Any]] = {}
        prior = ""
        for position, raw in enumerate(raw_inventory):
            entry = _closed_mapping(
                raw,
                required={"relative_path", "size_bytes", "sha256", "dtype", "shape"},
                label=f"TF-IDF payload_inventory[{position}]",
            )
            relative = str(entry["relative_path"])
            if (
                not relative
                or relative != Path(relative).name
                or not relative.endswith(".npy")
                or relative <= prior
            ):
                raise ValueError("TF-IDF payload inventory is reordered or has an invalid path")
            prior = relative
            if relative in self.by_path:
                raise ValueError("TF-IDF payload inventory contains duplicate paths")
            path = self.root / relative
            if path.is_symlink():
                raise ValueError("TF-IDF payload inventory contains a symlink")
            try:
                path_stat = path.lstat()
            except FileNotFoundError as exc:
                raise ValueError("TF-IDF payload inventory is missing a file") from exc
            if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
                raise ValueError("TF-IDF payload inventory contains a hard-linked/non-file payload")
            digest, size = _sha256_file(path)
            if digest != entry["sha256"] or size != int(entry["size_bytes"]):
                raise ValueError("TF-IDF payload bytes were tampered")
            self.inventory.append(entry)
            self.by_path[relative] = entry
        actual_names: list[str] = []
        for child in self.root.iterdir():
            if child.is_symlink() or not child.is_file():
                raise ValueError("TF-IDF safe artifact contains an extra or linked entry")
            actual_names.append(child.name)
        expected_names = [entry["relative_path"] for entry in self.inventory] + [INDEX_FILENAME]
        if sorted(actual_names) != sorted(expected_names):
            raise ValueError("TF-IDF safe artifact contains missing or extra payloads")
        self.state = manifest["state"]
        self.content_sha256 = str(manifest["content_sha256"])
        self._consumed: list[str] = []

    def array(self, reference: Any, *, label: str, mmap: bool = True) -> np.ndarray:
        reference = _closed_mapping(
            reference,
            required={"array"},
            label=f"{label} array reference",
        )
        relative = str(reference["array"])
        entry = self.by_path.get(relative)
        if entry is None:
            raise ValueError(f"{label} references an unregistered array")
        if relative in self._consumed:
            raise ValueError(f"{label} reuses an array payload")
        self._consumed.append(relative)
        try:
            value = np.load(
                self.root / relative,
                allow_pickle=False,
                mmap_mode="r" if mmap else None,
            )
        except Exception as exc:
            raise ValueError(f"{label} array is not a valid pickle-free NPY payload") from exc
        if _dtype_descriptor(value.dtype) != entry["dtype"]:
            raise ValueError(f"{label} array dtype mismatch")
        if list(value.shape) != entry["shape"]:
            raise ValueError(f"{label} array shape mismatch")
        if value.dtype.hasobject:
            raise ValueError(f"{label} array has forbidden object dtype")
        return value

    def finish(self) -> None:
        expected = [entry["relative_path"] for entry in self.inventory]
        if self._consumed != expected:
            raise ValueError("TF-IDF payload arrays were omitted, duplicated, or reordered")


def _estimator_params(estimator: Any, *, label: str) -> dict[str, Any]:
    params = estimator.get_params(deep=False)
    return {
        key: _json_scalar(value, label=f"{label}.params.{key}")
        for key, value in sorted(params.items())
    }


def _serialize_vectorizer(
    value: TfidfVectorizer,
    writer: _ArrayWriter,
    *,
    label: str,
) -> dict[str, Any]:
    if type(value) is not TfidfVectorizer:
        raise TypeError(f"{label} must be the exact TfidfVectorizer class")
    params = value.get_params(deep=False)
    for forbidden in ("preprocessor", "tokenizer", "stop_words"):
        if params[forbidden] is not None:
            raise TypeError(f"{label} uses unsupported callable/list parameter {forbidden}")
    if params["analyzer"] != "word" or params["input"] != "content":
        raise TypeError(f"{label} uses unsupported analyzer/input")
    dtype = np.dtype(params["dtype"])
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError(f"{label} uses unsupported matrix dtype {dtype}")
    vocabulary = getattr(value, "vocabulary_", None)
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise TypeError(f"{label} has no fitted vocabulary")
    ordered: list[str | None] = [None] * len(vocabulary)
    for term, raw_index in vocabulary.items():
        index = int(raw_index)
        if not isinstance(term, str) or index < 0 or index >= len(ordered) or ordered[index] is not None:
            raise ValueError(f"{label} vocabulary is not a bijective dense index")
        ordered[index] = term
    if any(term is None for term in ordered):
        raise ValueError(f"{label} vocabulary indices are incomplete")
    idf = np.asarray(value.idf_)
    if idf.shape != (len(ordered),):
        raise ValueError(f"{label} IDF shape disagrees with vocabulary")
    serial_params = {
        key: _json_scalar(raw, label=f"{label}.params.{key}")
        for key, raw in sorted(params.items())
        if key not in {"dtype", "vocabulary"}
    }
    serial_params["dtype"] = dtype.name
    return {
        "kind": "TfidfVectorizer",
        "params": serial_params,
        "terms": writer.add(f"{label}_terms", np.asarray(ordered, dtype=str)),
        "idf": writer.add(f"{label}_idf", idf),
    }


def _deserialize_vectorizer(
    state: Any,
    reader: _ArrayReader,
    *,
    label: str,
) -> TfidfVectorizer:
    state = _closed_mapping(
        state,
        required={"kind", "params", "terms", "idf"},
        label=label,
    )
    if state["kind"] != "TfidfVectorizer":
        raise ValueError(f"{label} kind mismatch")
    params = dict(state["params"])
    dtype_name = params.pop("dtype", None)
    if dtype_name not in {"float32", "float64"}:
        raise ValueError(f"{label} has unsupported dtype parameter")
    terms = reader.array(state["terms"], label=f"{label}.terms")
    if terms.ndim != 1 or terms.dtype.kind != "U":
        raise ValueError(f"{label} terms must be one-dimensional Unicode")
    term_list = [str(term) for term in terms.tolist()]
    if len(term_list) == 0 or len(set(term_list)) != len(term_list):
        raise ValueError(f"{label} terms must be nonempty and unique")
    idf = reader.array(state["idf"], label=f"{label}.idf")
    if idf.shape != (len(term_list),) or idf.dtype != np.dtype(dtype_name):
        raise ValueError(f"{label} IDF shape/dtype mismatch")
    vocabulary = {term: index for index, term in enumerate(term_list)}
    vectorizer = TfidfVectorizer(
        **params,
        dtype=np.dtype(dtype_name).type,
        vocabulary=vocabulary,
    )
    vectorizer.vocabulary_ = vocabulary
    vectorizer.fixed_vocabulary_ = True
    vectorizer._tfidf = TfidfTransformer(
        norm=params["norm"],
        use_idf=params["use_idf"],
        smooth_idf=params["smooth_idf"],
        sublinear_tf=params["sublinear_tf"],
    )
    vectorizer._tfidf.idf_ = np.asarray(idf).copy()
    vectorizer._tfidf.n_features_in_ = len(term_list)
    return vectorizer


def _serialize_linear(
    value: Any,
    writer: _ArrayWriter,
    *,
    label: str,
) -> dict[str, Any]:
    name = type(value).__name__
    if name not in _LINEAR_CLASSES or type(value) is not _LINEAR_CLASSES[name]:
        raise TypeError(f"{label} uses unsupported fitted estimator {type(value).__name__}")
    if name == "LogisticRegression":
        return {
            "kind": name,
            "params": _estimator_params(value, label=label),
            "n_features_in": int(value.n_features_in_),
            "classes": writer.add(f"{label}_classes", value.classes_),
            "coef": writer.add(f"{label}_coef", value.coef_),
            "intercept": writer.add(f"{label}_intercept", value.intercept_),
            "n_iter": writer.add(f"{label}_n_iter", value.n_iter_),
        }
    n_iter = None if value.n_iter_ is None else writer.add(f"{label}_n_iter", value.n_iter_)
    return {
        "kind": name,
        "params": _estimator_params(value, label=label),
        "n_features_in": int(value.n_features_in_),
        "coef": writer.add(f"{label}_coef", value.coef_),
        "intercept": writer.add(f"{label}_intercept", np.asarray(value.intercept_)),
        "n_iter": n_iter,
        "solver_fitted": str(value.solver_),
    }


def _deserialize_linear(state: Any, reader: _ArrayReader, *, label: str) -> Any:
    state = _closed_mapping(
        state,
        required=(
            {"kind", "params", "n_features_in", "classes", "coef", "intercept", "n_iter"}
            if isinstance(state, Mapping) and state.get("kind") == "LogisticRegression"
            else {
                "kind",
                "params",
                "n_features_in",
                "coef",
                "intercept",
                "n_iter",
                "solver_fitted",
            }
        ),
        label=label,
    )
    name = str(state["kind"])
    cls = _LINEAR_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"{label} has unsupported linear estimator kind")
    model = cls(**dict(state["params"]))
    model.n_features_in_ = int(state["n_features_in"])
    if name == "LogisticRegression":
        model.classes_ = np.asarray(reader.array(state["classes"], label=f"{label}.classes")).copy()
        model.coef_ = np.asarray(reader.array(state["coef"], label=f"{label}.coef")).copy()
        model.intercept_ = np.asarray(
            reader.array(state["intercept"], label=f"{label}.intercept")
        ).copy()
        model.n_iter_ = np.asarray(
            reader.array(state["n_iter"], label=f"{label}.n_iter")
        ).copy()
    else:
        model.coef_ = np.asarray(reader.array(state["coef"], label=f"{label}.coef")).copy()
        intercept = np.asarray(
            reader.array(state["intercept"], label=f"{label}.intercept")
        ).copy()
        model.intercept_ = intercept.item() if intercept.ndim == 0 else intercept
        model.n_iter_ = (
            None
            if state["n_iter"] is None
            else np.asarray(reader.array(state["n_iter"], label=f"{label}.n_iter")).copy()
        )
        model.solver_ = str(state["solver_fitted"])
    return model


def _serialize_tree(
    value: Any,
    writer: _ArrayWriter,
    *,
    label: str,
) -> dict[str, Any]:
    name = type(value).__name__
    if name not in _TREE_CLASSES or type(value) is not _TREE_CLASSES[name]:
        raise TypeError(f"{label} uses unsupported tree class {type(value).__name__}")
    classifier = name.endswith("Classifier")
    raw_tree = value.tree_.__getstate__()
    tree_state = _closed_mapping(
        raw_tree,
        required={"max_depth", "node_count", "nodes", "values"},
        label=f"{label}.tree_state",
    )
    state: dict[str, Any] = {
        "kind": name,
        "params": _estimator_params(value, label=label),
        "n_features_in": int(value.n_features_in_),
        "n_outputs": int(value.n_outputs_),
        "max_features_fitted": int(value.max_features_),
        "tree_max_depth": int(tree_state["max_depth"]),
        "tree_node_count": int(tree_state["node_count"]),
        "tree_nodes": writer.add(f"{label}_nodes", tree_state["nodes"]),
        "tree_values": writer.add(f"{label}_values", tree_state["values"]),
    }
    if classifier:
        state["classes"] = writer.add(f"{label}_classes", value.classes_)
        state["n_classes"] = int(value.n_classes_)
    return state


def _deserialize_tree(state: Any, reader: _ArrayReader, *, label: str) -> Any:
    raw_kind = state.get("kind") if isinstance(state, Mapping) else None
    classifier = str(raw_kind).endswith("Classifier")
    required = {
        "kind",
        "params",
        "n_features_in",
        "n_outputs",
        "max_features_fitted",
        "tree_max_depth",
        "tree_node_count",
        "tree_nodes",
        "tree_values",
    }
    if classifier:
        required |= {"classes", "n_classes"}
    state = _closed_mapping(state, required=required, label=label)
    name = str(state["kind"])
    cls = _TREE_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"{label} has unsupported tree kind")
    estimator = cls(**dict(state["params"]))
    estimator.n_features_in_ = int(state["n_features_in"])
    estimator.n_outputs_ = int(state["n_outputs"])
    estimator.max_features_ = int(state["max_features_fitted"])
    nodes = np.asarray(reader.array(state["tree_nodes"], label=f"{label}.tree_nodes")).copy()
    values = np.asarray(
        reader.array(state["tree_values"], label=f"{label}.tree_values")
    ).copy()
    if classifier:
        estimator.classes_ = np.asarray(
            reader.array(state["classes"], label=f"{label}.classes")
        ).copy()
        estimator.n_classes_ = int(state["n_classes"])
        n_classes = np.asarray([estimator.n_classes_], dtype=np.intp)
    else:
        n_classes = np.ones(estimator.n_outputs_, dtype=np.intp)
    if len(nodes) != int(state["tree_node_count"]) or len(values) != len(nodes):
        raise ValueError(f"{label} tree arrays disagree with node count")
    tree = _tree.Tree(estimator.n_features_in_, n_classes, estimator.n_outputs_)
    tree.__setstate__(
        {
            "max_depth": int(state["tree_max_depth"]),
            "node_count": int(state["tree_node_count"]),
            "nodes": nodes,
            "values": values,
        }
    )
    estimator.tree_ = tree
    return estimator


def _serialize_forest(
    value: Any,
    writer: _ArrayWriter,
    *,
    label: str,
) -> dict[str, Any]:
    name = type(value).__name__
    if name not in _FOREST_CLASSES or type(value) is not _FOREST_CLASSES[name]:
        raise TypeError(f"{label} uses unsupported forest class {type(value).__name__}")
    classifier = name.endswith("Classifier")
    state: dict[str, Any] = {
        "kind": name,
        "params": _estimator_params(value, label=label),
        "n_features_in": int(value.n_features_in_),
        "n_outputs": int(value.n_outputs_),
        "n_samples": int(value._n_samples),
        "n_samples_bootstrap": (
            None if value._n_samples_bootstrap is None else int(value._n_samples_bootstrap)
        ),
        "estimators": [
            _serialize_tree(tree, writer, label=f"{label}_tree_{index:04d}")
            for index, tree in enumerate(value.estimators_)
        ],
    }
    if classifier:
        state["classes"] = writer.add(f"{label}_classes", value.classes_)
        state["n_classes"] = int(value.n_classes_)
    return state


def _deserialize_forest(state: Any, reader: _ArrayReader, *, label: str) -> Any:
    raw_kind = state.get("kind") if isinstance(state, Mapping) else None
    classifier = str(raw_kind).endswith("Classifier")
    required = {
        "kind",
        "params",
        "n_features_in",
        "n_outputs",
        "n_samples",
        "n_samples_bootstrap",
        "estimators",
    }
    if classifier:
        required |= {"classes", "n_classes"}
    state = _closed_mapping(state, required=required, label=label)
    name = str(state["kind"])
    cls = _FOREST_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"{label} has unsupported forest kind")
    model = cls(**dict(state["params"]))
    model.n_features_in_ = int(state["n_features_in"])
    model.n_outputs_ = int(state["n_outputs"])
    model._n_samples = int(state["n_samples"])
    model._n_samples_bootstrap = (
        None
        if state["n_samples_bootstrap"] is None
        else int(state["n_samples_bootstrap"])
    )
    estimators = state["estimators"]
    if not isinstance(estimators, list) or len(estimators) != int(model.n_estimators):
        raise ValueError(f"{label} tree count disagrees with configured n_estimators")
    model.estimators_ = [
        _deserialize_tree(tree, reader, label=f"{label}.estimators[{index}]")
        for index, tree in enumerate(estimators)
    ]
    model.estimator_ = type(model.estimators_[0])() if model.estimators_ else None
    if classifier:
        model.classes_ = np.asarray(
            reader.array(state["classes"], label=f"{label}.classes")
        ).copy()
        model.n_classes_ = int(state["n_classes"])
    return model


def _serialize_estimator(
    value: Any,
    writer: _ArrayWriter,
    *,
    label: str,
) -> dict[str, Any]:
    if type(value).__name__ in _LINEAR_CLASSES:
        return _serialize_linear(value, writer, label=label)
    if type(value).__name__ in _FOREST_CLASSES:
        return _serialize_forest(value, writer, label=label)
    raise TypeError(f"{label} uses unsupported estimator class {type(value).__name__}")


def _deserialize_estimator(state: Any, reader: _ArrayReader, *, label: str) -> Any:
    kind = state.get("kind") if isinstance(state, Mapping) else None
    if kind in _LINEAR_CLASSES:
        return _deserialize_linear(state, reader, label=label)
    if kind in _FOREST_CLASSES:
        return _deserialize_forest(state, reader, label=label)
    raise ValueError(f"{label} has unsupported estimator kind")


def _serialize_stack(value: Any, writer: _ArrayWriter, *, label: str) -> dict[str, Any]:
    bases = []
    if len(value.views) != len(value.base_models):
        raise ValueError(f"{label} view/model counts disagree")
    for index, (vectorizer, estimator, constant) in enumerate(value.base_models):
        if (vectorizer is None) != (estimator is None):
            raise ValueError(f"{label} base model must omit vectorizer and estimator together")
        bases.append(
            {
                "constant": float(constant),
                "vectorizer": (
                    None
                    if vectorizer is None
                    else _serialize_vectorizer(
                        vectorizer,
                        writer,
                        label=f"{label}_base_{index:03d}_vectorizer",
                    )
                ),
                "estimator": (
                    None
                    if estimator is None
                    else _serialize_estimator(
                        estimator,
                        writer,
                        label=f"{label}_base_{index:03d}_estimator",
                    )
                ),
            }
        )
    return {
        "views": [
            _json_scalar(asdict(view), label=f"{label}.views")
            for view in value.views
        ],
        "binary": bool(value.binary),
        "base_models": bases,
        "stack_model": (
            None
            if value.stack_model is None
            else _serialize_estimator(
                value.stack_model,
                writer,
                label=f"{label}_stack_estimator",
            )
        ),
        "stack_constant": float(value.stack_constant),
        "config_hash": str(value.config_hash),
    }


def _deserialize_stack(state: Any, reader: _ArrayReader, *, label: str) -> Any:
    from .tfidf_topic_discovery import CrossFittedStack

    state = _closed_mapping(
        state,
        required={
            "views",
            "binary",
            "base_models",
            "stack_model",
            "stack_constant",
            "config_hash",
        },
        label=label,
    )
    if not isinstance(state["views"], list) or not isinstance(state["base_models"], list):
        raise ValueError(f"{label} views/base_models must be lists")
    views = [BoWViewConfig(**dict(raw)) for raw in state["views"]]
    if len(views) != len(state["base_models"]):
        raise ValueError(f"{label} view/model counts disagree")
    bases = []
    for index, raw in enumerate(state["base_models"]):
        raw = _closed_mapping(
            raw,
            required={"constant", "vectorizer", "estimator"},
            label=f"{label}.base_models[{index}]",
        )
        if (raw["vectorizer"] is None) != (raw["estimator"] is None):
            raise ValueError(f"{label} base model is partially present")
        bases.append(
            (
                None
                if raw["vectorizer"] is None
                else _deserialize_vectorizer(
                    raw["vectorizer"],
                    reader,
                    label=f"{label}.base_models[{index}].vectorizer",
                ),
                None
                if raw["estimator"] is None
                else _deserialize_estimator(
                    raw["estimator"],
                    reader,
                    label=f"{label}.base_models[{index}].estimator",
                ),
                float(raw["constant"]),
            )
        )
    stack_model = (
        None
        if state["stack_model"] is None
        else _deserialize_estimator(
            state["stack_model"],
            reader,
            label=f"{label}.stack_model",
        )
    )
    return CrossFittedStack(
        views=views,
        binary=bool(state["binary"]),
        base_models=bases,
        stack_model=stack_model,
        stack_constant=float(state["stack_constant"]),
        config_hash=str(state["config_hash"]),
    )


def _serialize_nmf(value: NMF, writer: _ArrayWriter, *, label: str) -> dict[str, Any]:
    if type(value) is not NMF:
        raise TypeError(f"{label} must be exact sklearn.decomposition.NMF")
    return {
        "params": _estimator_params(value, label=label),
        "n_features_in": int(value.n_features_in_),
        "n_components_fitted": int(value.n_components_),
        "private_n_components": int(value._n_components),
        "private_beta_loss": float(value._beta_loss),
        "reconstruction_error": writer.add(
            f"{label}_reconstruction_error",
            np.asarray(value.reconstruction_err_),
        ),
        "n_iter": int(value.n_iter_),
        "components": writer.add(f"{label}_components", value.components_),
    }


def _deserialize_nmf(state: Any, reader: _ArrayReader, *, label: str) -> NMF:
    state = _closed_mapping(
        state,
        required={
            "params",
            "n_features_in",
            "n_components_fitted",
            "private_n_components",
            "private_beta_loss",
            "reconstruction_error",
            "n_iter",
            "components",
        },
        label=label,
    )
    model = NMF(**dict(state["params"]))
    model.n_features_in_ = int(state["n_features_in"])
    model.n_components_ = int(state["n_components_fitted"])
    model._n_components = int(state["private_n_components"])
    model._beta_loss = float(state["private_beta_loss"])
    reconstruction_error = np.asarray(
        reader.array(
            state["reconstruction_error"],
            label=f"{label}.reconstruction_error",
        )
    )
    if reconstruction_error.ndim != 0:
        raise ValueError(f"{label} reconstruction error must be scalar")
    model.reconstruction_err_ = reconstruction_error.item()
    model.n_iter_ = int(state["n_iter"])
    model.components_ = np.asarray(
        reader.array(state["components"], label=f"{label}.components")
    ).copy()
    if model.components_.shape != (model.n_components_, model.n_features_in_):
        raise ValueError(f"{label} components shape mismatch")
    return model


def _serialize_topic_bank(value: Any, writer: _ArrayWriter, *, label: str) -> dict[str, Any]:
    if not (
        len(value.models) == len(value.component_norms) == len(value.alignments)
        == len(value.seeds)
    ):
        raise ValueError(f"{label} seed/model/alignment counts disagree")
    return {
        "bank_name": str(value.bank_name),
        "feature_names": [str(item) for item in value.feature_names],
        "selected_indices": writer.add(f"{label}_selected_indices", value.selected_indices),
        "feature_weights": writer.add(f"{label}_feature_weights", value.feature_weights),
        "models": [
            _serialize_nmf(model, writer, label=f"{label}_nmf_{index:03d}")
            for index, model in enumerate(value.models)
        ],
        "component_norms": [
            writer.add(f"{label}_norms_{index:03d}", array)
            for index, array in enumerate(value.component_norms)
        ],
        "alignments": [
            writer.add(f"{label}_alignment_{index:03d}", array)
            for index, array in enumerate(value.alignments)
        ],
        "consensus_loadings": writer.add(
            f"{label}_consensus_loadings", value.consensus_loadings
        ),
        "topic_terms": _json_scalar(value.topic_terms, label=f"{label}.topic_terms"),
        "requested_components": int(value.requested_components),
        "actual_components": int(value.actual_components),
        "terms_per_topic": int(value.terms_per_topic),
        "seeds": [int(seed) for seed in value.seeds],
        "reduction_reason": (
            None if value.reduction_reason is None else str(value.reduction_reason)
        ),
    }


def _deserialize_topic_bank(state: Any, reader: _ArrayReader, *, label: str) -> Any:
    from .tfidf_topic_discovery import ConsensusNMFTopicBank

    state = _closed_mapping(
        state,
        required={
            "bank_name",
            "feature_names",
            "selected_indices",
            "feature_weights",
            "models",
            "component_norms",
            "alignments",
            "consensus_loadings",
            "topic_terms",
            "requested_components",
            "actual_components",
            "terms_per_topic",
            "seeds",
            "reduction_reason",
        },
        label=label,
    )
    models = state["models"]
    norms = state["component_norms"]
    alignments = state["alignments"]
    seeds = state["seeds"]
    if not all(isinstance(value, list) for value in (models, norms, alignments, seeds)):
        raise ValueError(f"{label} model/norm/alignment/seed fields must be lists")
    if not (len(models) == len(norms) == len(alignments) == len(seeds)):
        raise ValueError(f"{label} model/norm/alignment/seed counts disagree")
    selected = np.asarray(
        reader.array(state["selected_indices"], label=f"{label}.selected_indices")
    )
    weights = np.asarray(
        reader.array(state["feature_weights"], label=f"{label}.feature_weights")
    )
    if selected.ndim != 1 or selected.dtype.kind not in "iu" or weights.shape != selected.shape:
        raise ValueError(f"{label} selected indices/weights are invalid")
    actual = int(state["actual_components"])
    terms_per_topic = int(state["terms_per_topic"])
    if terms_per_topic < 1:
        raise ValueError(f"{label} terms_per_topic must be positive")
    loaded_models = [
        _deserialize_nmf(item, reader, label=f"{label}.models[{index}]")
        for index, item in enumerate(models)
    ]
    loaded_norms = [
        np.asarray(reader.array(item, label=f"{label}.component_norms[{index}]")).copy()
        for index, item in enumerate(norms)
    ]
    loaded_alignments = [
        np.asarray(reader.array(item, label=f"{label}.alignments[{index}]")).copy()
        for index, item in enumerate(alignments)
    ]
    if any(array.shape != (actual,) for array in [*loaded_norms, *loaded_alignments]):
        raise ValueError(f"{label} norm/alignment shapes disagree with component count")
    consensus = np.asarray(
        reader.array(state["consensus_loadings"], label=f"{label}.consensus_loadings")
    ).copy()
    if consensus.shape != (actual, len(selected)):
        raise ValueError(f"{label} consensus loading shape mismatch")
    topic_terms = list(state["topic_terms"])
    if len(topic_terms) != actual or any(
        not isinstance(topic, list) or len(topic) != terms_per_topic
        for topic in topic_terms
    ):
        raise ValueError(
            f"{label} topic term evidence does not match configured capacity"
        )
    feature_names = [str(item) for item in state["feature_names"]]
    if len(feature_names) == 0 or len(set(feature_names)) != len(feature_names):
        raise ValueError(f"{label} feature names must be nonempty and unique")
    if len(selected) and (int(selected.min()) < 0 or int(selected.max()) >= len(feature_names)):
        raise ValueError(f"{label} selected index is outside the feature vocabulary")
    return ConsensusNMFTopicBank(
        bank_name=str(state["bank_name"]),
        feature_names=feature_names,
        selected_indices=selected.copy(),
        feature_weights=weights.copy(),
        models=loaded_models,
        component_norms=loaded_norms,
        alignments=loaded_alignments,
        consensus_loadings=consensus,
        topic_terms=topic_terms,
        requested_components=int(state["requested_components"]),
        actual_components=actual,
        terms_per_topic=terms_per_topic,
        seeds=[int(seed) for seed in seeds],
        reduction_reason=(
            None if state["reduction_reason"] is None else str(state["reduction_reason"])
        ),
    )


def write_fitted_topic_context(value: Any, root: Path | str) -> Path:
    """Write one explicit fitted context and return its canonical index path."""

    writer = _ArrayWriter(Path(root))
    state = {
        "config_hash": str(value.config_hash),
        "common_vectorizer": _serialize_vectorizer(
            value.common_vectorizer,
            writer,
            label="common_vectorizer",
        ),
        "treatment_stack": _serialize_stack(
            value.treatment_stack,
            writer,
            label="treatment_stack",
        ),
        "outcome_stack": _serialize_stack(
            value.outcome_stack,
            writer,
            label="outcome_stack",
        ),
        "topic_banks": [
            _serialize_topic_bank(value.topic_banks[name], writer, label=f"topic_bank_{name}")
            for name in sorted(value.topic_banks)
        ],
    }
    return writer.finish(
        artifact_kind="tfidf_fitted_topic_context",
        schema_version=FITTED_CONTEXT_SCHEMA_VERSION,
        state=state,
    )


def load_fitted_topic_context(index_path: Path | str) -> Any:
    """Authenticate and reconstruct one fitted context without pickle/joblib."""

    from .tfidf_topic_discovery import FittedTopicContext

    reader = _ArrayReader(
        index_path,
        artifact_kind="tfidf_fitted_topic_context",
        schema_version=FITTED_CONTEXT_SCHEMA_VERSION,
    )
    state = _closed_mapping(
        reader.state,
        required={
            "config_hash",
            "common_vectorizer",
            "treatment_stack",
            "outcome_stack",
            "topic_banks",
        },
        label="fitted TF-IDF context state",
    )
    common_vectorizer = _deserialize_vectorizer(
        state["common_vectorizer"],
        reader,
        label="common_vectorizer",
    )
    treatment_stack = _deserialize_stack(
        state["treatment_stack"],
        reader,
        label="treatment_stack",
    )
    outcome_stack = _deserialize_stack(
        state["outcome_stack"],
        reader,
        label="outcome_stack",
    )
    banks = state["topic_banks"]
    if not isinstance(banks, list):
        raise ValueError("fitted TF-IDF topic_banks must be a list")
    loaded_banks: dict[str, Any] = {}
    for index, raw in enumerate(banks):
        bank = _deserialize_topic_bank(raw, reader, label=f"topic_banks[{index}]")
        if bank.bank_name in loaded_banks:
            raise ValueError("fitted TF-IDF topic bank names are duplicated")
        loaded_banks[bank.bank_name] = bank
    result = FittedTopicContext(
        common_vectorizer=common_vectorizer,
        treatment_stack=treatment_stack,
        outcome_stack=outcome_stack,
        topic_banks=loaded_banks,
        config_hash=str(state["config_hash"]),
    )
    reader.finish()
    return result


def write_named_array_bank(
    values: Mapping[str, Any],
    root: Path | str,
    *,
    row_count: int,
) -> Path:
    """Write an ordered bank of dense topic arrays as independent NPY files."""

    if int(row_count) < 0:
        raise ValueError("TF-IDF named array bank row_count cannot be negative")
    writer = _ArrayWriter(Path(root))
    if any(not isinstance(name, str) for name in values):
        raise ValueError("TF-IDF named array bank names must be strings")
    names = sorted(values)
    if len(names) != len(set(names)):
        raise ValueError("TF-IDF named array bank names must be unique strings")
    entries = []
    for name in names:
        array = np.asarray(values[name])
        if array.ndim != 2 or array.shape[0] != int(row_count):
            raise ValueError(f"TF-IDF array bank {name!r} has an invalid row shape")
        entries.append({"name": name, "values": writer.add(f"bank_{name}", array)})
    return writer.finish(
        artifact_kind="tfidf_named_array_bank",
        schema_version=ARRAY_BANK_SCHEMA_VERSION,
        state={"row_count": int(row_count), "entries": entries},
    )


def load_named_array_bank(
    index_path: Path | str,
    *,
    expected_row_count: int | None = None,
) -> dict[str, np.ndarray]:
    """Authenticate an array bank and return read-only mmap-backed arrays."""

    reader = _ArrayReader(
        index_path,
        artifact_kind="tfidf_named_array_bank",
        schema_version=ARRAY_BANK_SCHEMA_VERSION,
    )
    state = _closed_mapping(
        reader.state,
        required={"row_count", "entries"},
        label="TF-IDF named array bank state",
    )
    row_count = int(state["row_count"])
    if row_count < 0 or (
        expected_row_count is not None and row_count != int(expected_row_count)
    ):
        raise ValueError("TF-IDF named array bank row count mismatch")
    entries = state["entries"]
    if not isinstance(entries, list):
        raise ValueError("TF-IDF named array bank entries must be a list")
    output: dict[str, np.ndarray] = {}
    prior = ""
    for position, raw in enumerate(entries):
        raw = _closed_mapping(
            raw,
            required={"name", "values"},
            label=f"TF-IDF named array bank entries[{position}]",
        )
        name = str(raw["name"])
        if not name or name <= prior or name in output:
            raise ValueError("TF-IDF named array bank entries are duplicated or reordered")
        prior = name
        values = reader.array(raw["values"], label=f"array bank {name}")
        if values.ndim != 2 or values.shape[0] != row_count:
            raise ValueError(f"TF-IDF named array bank {name!r} shape mismatch")
        output[name] = values
    reader.finish()
    return output


def safe_artifact_content_sha256(index_path: Path | str) -> str:
    """Return the authenticated transitive content root for a safe artifact."""

    index = Path(index_path)
    root = _safe_artifact_root(index)
    try:
        value = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("TF-IDF safe artifact index is unreadable") from exc
    digest = value.get("content_sha256") if isinstance(value, Mapping) else None
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("TF-IDF safe artifact index has no content root")
    # Dispatch through the full validator based on its declared kind.
    kind = value.get("artifact_kind")
    if kind == "tfidf_fitted_topic_context":
        reader = _ArrayReader(
            index,
            artifact_kind=kind,
            schema_version=FITTED_CONTEXT_SCHEMA_VERSION,
        )
    elif kind == "tfidf_named_array_bank":
        reader = _ArrayReader(
            index,
            artifact_kind=kind,
            schema_version=ARRAY_BANK_SCHEMA_VERSION,
        )
    else:
        raise ValueError("TF-IDF safe artifact kind is unsupported")
    for entry in reader.inventory:
        reader.array(
            {"array": entry["relative_path"]},
            label=f"authenticated payload {entry['relative_path']}",
        )
    reader.finish()
    del reader, root
    return digest
