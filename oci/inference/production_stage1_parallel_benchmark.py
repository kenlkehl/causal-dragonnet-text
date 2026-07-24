"""Authenticated, bounded benchmarks for Stage 1 CPU parallelism.

The benchmark has two deliberately narrow input modes:

* a closed numeric fixture plan exercises cluster-like and TF-IDF-like CPU
  work without opening a cohort; and
* a published clustered-preflight scope-input set exercises the real
  one-scope production worker against already authenticated private inputs.

Every selected job is executed at worker counts 1, 4, and 8.  Operational
timings and native-thread observations are kept separate from the canonical
scientific result.  A run is accepted only when the complete ordered
scientific result is byte-canonically identical at all three worker counts.
The terminal manifest is written only after every other artifact is durable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_info, threadpool_limits

BENCHMARK_REQUEST_SCHEMA = "production_stage1_parallel_benchmark_request_v1"
BENCHMARK_RESULT_SCHEMA = "production_stage1_parallel_benchmark_result_v1"
BENCHMARK_TERMINAL_SCHEMA = "production_stage1_parallel_benchmark_terminal_v1"
BENCHMARK_FIXTURE_PLAN_SCHEMA = "production_stage1_parallel_fixture_plan_v1"
BENCHMARK_TERMINAL_NAME = "terminal_manifest.json"
REQUIRED_WORKER_COUNTS = (1, 4, 8)

_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_INPUT_KEY_PARTS = (
    "oracle",
    "true_",
    "potential_outcome",
    "patient_prompt",
    "event_timeline",
)
_THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("benchmark payload contains a non-finite number")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"benchmark payload contains unsupported type {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_closed_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
    ):
        raise ValueError(f"{label} path is not a regular absolute file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a JSON object")
    return dict(value)


def _registration(path: Path, *, root: Path) -> Mapping[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("benchmark artifact is not a regular file")
    relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    return {
        "relative_path": relative,
        "size": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _validate_registration(
    root: Path,
    registration: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    if not isinstance(registration, Mapping) or set(registration) != {
        "relative_path",
        "size",
        "sha256",
    }:
        raise ValueError(f"{label} registration is not closed")
    relative = Path(str(registration["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} registration escapes its root")
    path = (root / relative).absolute()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
        or int(path.stat().st_size) != int(registration["size"])
        or _sha256_file(path)
        != _require_sha256(
            registration["sha256"],
            label=f"{label} SHA",
        )
    ):
        raise ValueError(f"{label} artifact changed")
    return path


def _reject_forbidden_fixture_fields(
    value: Any,
    *,
    path: str = "fixture_plan",
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if any(part in normalized for part in _FORBIDDEN_INPUT_KEY_PARTS):
                raise ValueError(f"{path}.{key} is a forbidden benchmark input field")
            _reject_forbidden_fixture_fields(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_fixture_fields(child, path=f"{path}[{index}]")


def _array_identity(values: np.ndarray) -> Mapping[str, Any]:
    array = np.ascontiguousarray(values)
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError("benchmark numerical output is non-finite")
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _package_versions() -> Mapping[str, str]:
    versions: dict[str, str] = {}
    for name in ("joblib", "numpy", "scikit-learn", "scipy", "threadpoolctl"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


def _source_identity(*, real_preflight: bool) -> Mapping[str, Any]:
    paths = [Path(__file__).resolve(strict=True)]
    if real_preflight:
        paths.extend(
            [
                Path(__file__).with_name("production_stage1_bundle.py").resolve(strict=True),
                Path(__file__)
                .with_name("production_stage1_preflight_scope_inputs.py")
                .resolve(strict=True),
            ]
        )
    rows = [
        {
            "path": str(path),
            "size": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    body = {
        "files": rows,
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _validate_fixture_job(
    value: Mapping[str, Any],
    *,
    kind: str,
    index: int,
) -> Mapping[str, Any]:
    common = {"job_id", "seed", "repetitions"}
    if kind == "cluster_preflight_fixture":
        expected = common | {
            "sample_count",
            "feature_count",
            "cluster_count",
        }
    elif kind == "tfidf_fixture":
        expected = common | {
            "document_count",
            "vocabulary_size",
            "topic_count",
        }
    else:
        raise ValueError("unknown fixture kind")
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{kind} job {index} is not a closed object")
    job_id = str(value["job_id"])
    if job_id != f"{kind}_{index:03d}":
        raise ValueError(f"{kind} job IDs are not canonical")
    integers = {key: int(value[key]) for key in expected if key != "job_id"}
    if any(isinstance(value[key], bool) or integers[key] != value[key] for key in integers):
        raise ValueError(f"{kind} job parameters must be integers")
    if integers["seed"] < 0 or integers["repetitions"] < 1:
        raise ValueError(f"{kind} job seed/repetitions are invalid")
    if kind == "cluster_preflight_fixture":
        if (
            integers["sample_count"] < 24
            or integers["feature_count"] < 4
            or integers["cluster_count"] < 2
            or integers["cluster_count"] >= integers["sample_count"]
        ):
            raise ValueError("cluster fixture dimensions are invalid")
    else:
        if (
            integers["document_count"] < 24
            or integers["vocabulary_size"] < 24
            or integers["topic_count"] < 2
            or integers["topic_count"] >= integers["vocabulary_size"]
        ):
            raise ValueError("TF-IDF fixture dimensions are invalid")
    return {
        "job_id": job_id,
        **{key: integers[key] for key in sorted(integers)},
    }


def load_fixture_plan(path: Path | str) -> Mapping[str, Any]:
    plan_path = Path(path).absolute()
    plan = _read_closed_json(plan_path, label="parallel benchmark fixture plan")
    _reject_forbidden_fixture_fields(plan)
    required = {
        "schema_version",
        "seed",
        "cluster_preflight_jobs",
        "tfidf_jobs",
        "content_sha256",
    }
    body = {key: copy.deepcopy(value) for key, value in plan.items() if key != "content_sha256"}
    if (
        set(plan) != required
        or plan.get("schema_version") != BENCHMARK_FIXTURE_PLAN_SCHEMA
        or plan.get("content_sha256") != _sha256_json(body)
        or isinstance(plan.get("seed"), bool)
        or not isinstance(plan.get("seed"), int)
        or int(plan["seed"]) < 0
    ):
        raise ValueError("parallel benchmark fixture plan has an invalid binding")
    normalized: dict[str, list[Mapping[str, Any]]] = {}
    for key, kind in (
        ("cluster_preflight_jobs", "cluster_preflight_fixture"),
        ("tfidf_jobs", "tfidf_fixture"),
    ):
        rows = plan.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"{key} must be a list")
        normalized[key] = [
            _validate_fixture_job(row, kind=kind, index=index) for index, row in enumerate(rows)
        ]
    if not normalized["cluster_preflight_jobs"] and not normalized["tfidf_jobs"]:
        raise ValueError("fixture plan contains no work")
    if (
        normalized["cluster_preflight_jobs"] != plan["cluster_preflight_jobs"]
        or normalized["tfidf_jobs"] != plan["tfidf_jobs"]
    ):
        raise ValueError("fixture plan is not canonically ordered")
    return copy.deepcopy(plan)


def make_fixture_plan(
    *,
    job_count: int = 8,
    workload_scale: int = 1,
    seed: int = 42,
) -> Mapping[str, Any]:
    if int(job_count) < 8 or int(workload_scale) < 1 or int(seed) < 0:
        raise ValueError("fixture plan requires >=8 jobs, positive scale, and seed")
    count = int(job_count)
    scale = int(workload_scale)
    body = {
        "schema_version": BENCHMARK_FIXTURE_PLAN_SCHEMA,
        "seed": int(seed),
        "cluster_preflight_jobs": [
            {
                "job_id": f"cluster_preflight_fixture_{index:03d}",
                "seed": int(seed) + 1000 + index,
                "repetitions": scale,
                "sample_count": 180 * scale,
                "feature_count": 24,
                "cluster_count": 6,
            }
            for index in range(count)
        ],
        "tfidf_jobs": [
            {
                "job_id": f"tfidf_fixture_{index:03d}",
                "seed": int(seed) + 2000 + index,
                "repetitions": scale,
                "document_count": 220 * scale,
                "vocabulary_size": 180,
                "topic_count": 6,
            }
            for index in range(count)
        ],
    }
    return {**body, "content_sha256": _sha256_json(body)}


def write_fixture_plan(
    path: Path | str,
    *,
    job_count: int = 8,
    workload_scale: int = 1,
    seed: int = 42,
) -> Path:
    target = Path(path).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(
        target,
        make_fixture_plan(
            job_count=job_count,
            workload_scale=workload_scale,
            seed=seed,
        ),
    )
    _fsync_directory(target.parent)
    return target.resolve(strict=True)


def _load_preflight_payloads(
    path: Path | str,
    *,
    selected_scope_ids: Sequence[str] = (),
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    manifest_path = Path(path).absolute()
    manifest = _read_closed_json(
        manifest_path,
        label="preflight scope-input set manifest",
    )
    required = {
        "schema_version",
        "registry_content_sha256",
        "scope_order",
        "scope_count",
        "scopes",
        "one_scope_per_worker_payload",
        "content_sha256",
    }
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    scope_order = manifest.get("scope_order")
    rows = manifest.get("scopes")
    if (
        set(manifest) != required
        or manifest.get("schema_version") != "production_stage1_preflight_scope_input_set_v2"
        or manifest.get("content_sha256") != _sha256_json(body)
        or manifest.get("one_scope_per_worker_payload") is not True
        or not isinstance(scope_order, list)
        or not isinstance(rows, list)
        or len(rows) != len(scope_order)
        or manifest.get("scope_count") != len(scope_order)
        or len(scope_order) != len(set(map(str, scope_order)))
    ):
        raise ValueError("preflight scope-input set manifest is invalid")
    root = manifest_path.parent
    requested = tuple(map(str, selected_scope_ids))
    if len(requested) != len(set(requested)):
        raise ValueError("selected preflight scope IDs are duplicated")
    selected = set(requested or map(str, scope_order))
    if not selected.issubset(set(map(str, scope_order))):
        raise ValueError("selected preflight scope ID is absent")
    payloads: list[Mapping[str, Any]] = []
    child_identities: list[Mapping[str, Any]] = []
    for scope_id, row in zip(map(str, scope_order), rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"scope_id", "manifest"}
            or row.get("scope_id") != scope_id
        ):
            raise ValueError("preflight scope-input row changed")
        child_path = _validate_registration(
            root,
            row["manifest"],
            label=f"{scope_id} preflight manifest",
        )
        child = _read_closed_json(
            child_path,
            label=f"{scope_id} preflight child manifest",
        )
        child_body = {
            key: copy.deepcopy(value) for key, value in child.items() if key != "content_sha256"
        }
        if (
            child.get("schema_version") != "production_stage1_preflight_scope_input_v2"
            or child.get("content_sha256") != _sha256_json(child_body)
            or not isinstance(child.get("scope"), Mapping)
            or child["scope"].get("scope_id") != scope_id
        ):
            raise ValueError("preflight child manifest has an invalid binding")
        if scope_id in selected:
            payloads.append(
                {
                    "schema_version": "production_stage1_preflight_worker_payload_v1",
                    "scope_id": scope_id,
                    "manifest_path": str(child_path),
                    "manifest_content_sha256": str(child["content_sha256"]),
                }
            )
            child_identities.append(
                {
                    "scope_id": scope_id,
                    "manifest_relative_path": child_path.relative_to(root).as_posix(),
                    "manifest_file_sha256": _sha256_file(child_path),
                    "manifest_content_sha256": str(child["content_sha256"]),
                }
            )
    canonical_selected = [scope_id for scope_id in map(str, scope_order) if scope_id in selected]
    if [row["scope_id"] for row in payloads] != canonical_selected:
        raise RuntimeError("preflight benchmark payload order changed")
    identity_body = {
        "set_manifest_path": str(manifest_path),
        "set_manifest_file_sha256": _sha256_file(manifest_path),
        "set_manifest_content_sha256": str(manifest["content_sha256"]),
        "registry_content_sha256": _require_sha256(
            manifest["registry_content_sha256"],
            label="preflight registry SHA",
        ),
        "selected_scope_order": canonical_selected,
        "children": child_identities,
    }
    return tuple(payloads), {
        **identity_body,
        "content_sha256": _sha256_json(identity_body),
    }


def _cluster_fixture(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD

    seed = int(payload["seed"])
    identities: list[Mapping[str, Any]] = []
    for repetition in range(int(payload["repetitions"])):
        generator = np.random.default_rng(seed + repetition)
        samples = generator.normal(
            size=(
                int(payload["sample_count"]),
                int(payload["feature_count"]),
            )
        )
        samples += generator.integers(
            0,
            int(payload["cluster_count"]),
            size=(len(samples), 1),
        )
        fitted = KMeans(
            n_clusters=int(payload["cluster_count"]),
            n_init=5,
            max_iter=80,
            algorithm="lloyd",
            random_state=seed + repetition,
        ).fit(samples)
        residual = samples - fitted.cluster_centers_[fitted.labels_]
        svd = TruncatedSVD(
            n_components=min(5, int(payload["feature_count"]) - 1),
            algorithm="randomized",
            random_state=seed + repetition,
        ).fit(residual)
        identities.append(
            {
                "labels": _array_identity(fitted.labels_.astype("<i8")),
                "centers": _array_identity(np.asarray(fitted.cluster_centers_, dtype="<f8")),
                "singular_values": _array_identity(np.asarray(svd.singular_values_, dtype="<f8")),
                "components": _array_identity(np.asarray(svd.components_, dtype="<f8")),
            }
        )
    return {
        "schema_version": "production_stage1_cluster_fixture_output_v1",
        "job_id": str(payload["job_id"]),
        "repetitions": identities,
    }


def _synthetic_documents(
    *,
    seed: int,
    document_count: int,
    vocabulary_size: int,
) -> tuple[list[str], np.ndarray]:
    generator = np.random.default_rng(seed)
    labels = np.arange(document_count, dtype=np.int64) % 2
    generator.shuffle(labels)
    documents: list[str] = []
    for label in labels:
        common = generator.integers(4, vocabulary_size, size=48)
        anchors = generator.integers(0, 2, size=12) + (0 if label == 0 else 2)
        tokens = np.concatenate((common, anchors))
        generator.shuffle(tokens)
        documents.append(" ".join(f"term{int(token):04d}" for token in tokens))
    return documents, labels


def _tfidf_fixture(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    seed = int(payload["seed"])
    identities: list[Mapping[str, Any]] = []
    for repetition in range(int(payload["repetitions"])):
        documents, labels = _synthetic_documents(
            seed=seed + repetition,
            document_count=int(payload["document_count"]),
            vocabulary_size=int(payload["vocabulary_size"]),
        )
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=int(payload["vocabulary_size"]) * 3,
            dtype=np.float64,
        )
        matrix = vectorizer.fit_transform(documents)
        topics = NMF(
            n_components=int(payload["topic_count"]),
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            max_iter=120,
            random_state=seed + repetition,
        ).fit_transform(matrix)
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=300,
            random_state=seed + repetition,
        ).fit(topics, labels)
        identities.append(
            {
                "vocabulary_sha256": _sha256_json(
                    sorted(
                        (str(term), int(index)) for term, index in vectorizer.vocabulary_.items()
                    )
                ),
                "matrix_indptr": _array_identity(matrix.indptr.astype("<i8")),
                "matrix_indices": _array_identity(matrix.indices.astype("<i8")),
                "matrix_data": _array_identity(np.asarray(matrix.data, dtype="<f8")),
                "topics": _array_identity(np.asarray(topics, dtype="<f8")),
                "coefficients": _array_identity(np.asarray(model.coef_, dtype="<f8")),
                "probabilities": _array_identity(
                    np.asarray(model.predict_proba(topics), dtype="<f8")
                ),
            }
        )
    return {
        "schema_version": "production_stage1_tfidf_fixture_output_v1",
        "job_id": str(payload["job_id"]),
        "repetitions": identities,
    }


def _real_preflight(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    from .production_stage1_bundle import (
        _embedding_cluster_preflight_loky_scope,
    )

    return _embedding_cluster_preflight_loky_scope(payload)


def _execute_job(
    *,
    kind: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Execute one job with a native numerical-thread limit of one."""

    with threadpool_limits(limits=1):
        if kind == "cluster_preflight_fixture":
            scientific = _cluster_fixture(payload)
        elif kind == "tfidf_fixture":
            scientific = _tfidf_fixture(payload)
        elif kind == "cluster_preflight_authenticated":
            scientific = _real_preflight(payload)
        else:
            raise ValueError(f"unsupported benchmark kind {kind!r}")
        pool_rows = [
            {
                "user_api": str(row.get("user_api") or ""),
                "internal_api": str(row.get("internal_api") or ""),
                "prefix": str(row.get("prefix") or ""),
                "num_threads": int(row.get("num_threads", -1)),
            }
            for row in threadpool_info()
        ]
    if any(row["num_threads"] != 1 for row in pool_rows):
        raise RuntimeError("native numerical thread cap was not effective")
    job_id = str(payload.get("job_id") or payload.get("scope_id") or "")
    body = {
        "job_id": job_id,
        "scientific_output": _jsonable(scientific),
    }
    return {
        **body,
        "scientific_output_sha256": _sha256_json(body),
        "native_thread_cap": {
            "requested_threads": 1,
            "effective_pools": pool_rows,
            "all_observed_pools_limited_to_one": True,
            "environment": {key: os.environ.get(key) for key in _THREAD_ENVIRONMENT_KEYS},
        },
    }


def _warm_benchmark_worker(*, kind: str) -> Mapping[str, Any]:
    """Import the measured stack before steady-state wall timing."""

    with threadpool_limits(limits=1):
        if kind == "cluster_preflight_fixture":
            from sklearn.cluster import KMeans  # noqa: F401
            from sklearn.decomposition import TruncatedSVD  # noqa: F401
        elif kind == "tfidf_fixture":
            from sklearn.decomposition import NMF  # noqa: F401
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
            from sklearn.linear_model import LogisticRegression  # noqa: F401
        elif kind == "cluster_preflight_authenticated":
            from .production_stage1_bundle import (  # noqa: F401
                _embedding_cluster_preflight_loky_scope,
            )
        else:
            raise ValueError(f"unsupported benchmark kind {kind!r}")
        pools = [int(row.get("num_threads", -1)) for row in threadpool_info()]
    if any(value != 1 for value in pools):
        raise RuntimeError("benchmark warmup did not enforce native thread caps")
    return {"pid": os.getpid(), "native_pool_threads": pools}


def _shutdown_reusable_loky_executor(*, workers: int) -> None:
    """Prevent one worker-count trial from warming the next trial's pool."""

    from joblib.externals.loky import get_reusable_executor

    executor = get_reusable_executor(max_workers=int(workers), reuse=True)
    executor.shutdown(wait=True, kill_workers=True)


def _run_family(
    *,
    kind: str,
    payloads: Sequence[Mapping[str, Any]],
    worker_counts: Sequence[int],
) -> Mapping[str, Any]:
    if not payloads:
        raise ValueError("parallel benchmark family has no jobs")
    job_order = tuple(
        str(payload.get("job_id") or payload.get("scope_id") or "") for payload in payloads
    )
    if any(not value for value in job_order) or len(job_order) != len(set(job_order)):
        raise ValueError("parallel benchmark job IDs are empty or duplicated")
    input_rows = [
        {
            "job_id": job_id,
            "payload_sha256": _sha256_json(payload),
        }
        for job_id, payload in zip(job_order, payloads)
    ]
    runs: list[Mapping[str, Any]] = []
    baseline_identity: str | None = None
    baseline_seconds: float | None = None
    for requested_workers in worker_counts:
        effective_workers = min(int(requested_workers), len(payloads))
        if effective_workers == 1:
            _warm_benchmark_worker(kind=kind)
            started = time.perf_counter_ns()
            raw_results = [_execute_job(kind=kind, payload=payload) for payload in payloads]
            elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000.0
        else:
            with parallel_config(
                backend="loky",
                n_jobs=effective_workers,
                inner_max_num_threads=1,
            ):
                with Parallel(batch_size=1, pre_dispatch="all") as parallel:
                    warm_rows = parallel(
                        delayed(_warm_benchmark_worker)(kind=kind)
                        for _index in range(effective_workers)
                    )
                    if len(warm_rows) != effective_workers:
                        raise RuntimeError("parallel benchmark warmup was incomplete")
                    started = time.perf_counter_ns()
                    raw_results = parallel(
                        delayed(_execute_job)(kind=kind, payload=payload) for payload in payloads
                    )
                    elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000.0
            _shutdown_reusable_loky_executor(workers=effective_workers)
        if len(raw_results) != len(payloads):
            raise RuntimeError("parallel benchmark returned incomplete work")
        by_job: dict[str, Mapping[str, Any]] = {}
        for result in raw_results:
            job_id = str(result.get("job_id") or "")
            if job_id in by_job:
                raise RuntimeError("parallel benchmark returned a duplicate job")
            by_job[job_id] = result
        if set(by_job) != set(job_order):
            raise RuntimeError("parallel benchmark substituted a job")
        ordered = [by_job[job_id] for job_id in job_order]
        scientific_body = {
            "kind": kind,
            "job_order": list(job_order),
            "outputs": [
                {
                    "job_id": row["job_id"],
                    "scientific_output": row["scientific_output"],
                    "scientific_output_sha256": row["scientific_output_sha256"],
                }
                for row in ordered
            ],
        }
        scientific_identity = _sha256_json(scientific_body)
        if baseline_identity is None:
            baseline_identity = scientific_identity
            baseline_seconds = elapsed_seconds
        elif scientific_identity != baseline_identity:
            raise RuntimeError(f"{kind} scientific output changed at {requested_workers} workers")
        assert baseline_seconds is not None
        runs.append(
            {
                "requested_workers": int(requested_workers),
                "effective_workers": effective_workers,
                "wall_seconds": elapsed_seconds,
                "speedup_vs_one_worker": baseline_seconds / elapsed_seconds,
                "scientific_identity_sha256": scientific_identity,
                "native_thread_caps": [
                    {
                        "job_id": row["job_id"],
                        **row["native_thread_cap"],
                    }
                    for row in ordered
                ],
            }
        )
    assert baseline_identity is not None
    body = {
        "schema_version": BENCHMARK_RESULT_SCHEMA,
        "kind": kind,
        "job_count": len(payloads),
        "job_order": list(job_order),
        "input_jobs": input_rows,
        "required_worker_counts": list(REQUIRED_WORKER_COUNTS),
        "runs": runs,
        "canonical_scientific_identity_sha256": baseline_identity,
        "exact_scientific_equality_across_worker_counts": True,
        "native_thread_limit": 1,
        "wall_time_kind": "steady_state_after_worker_import_warmup",
        "worker_import_warmup_excluded_from_wall_time": True,
        "loky_executor_restarted_between_parallel_worker_counts": True,
    }
    return {**body, "content_sha256": _sha256_json(body)}


@dataclass(frozen=True)
class Stage1ParallelBenchmarkOptions:
    output_root: Path
    fixture_plan_path: Path | None = None
    preflight_scope_input_set_manifest: Path | None = None
    preflight_scope_ids: tuple[str, ...] = ()
    worker_counts: tuple[int, ...] = REQUIRED_WORKER_COUNTS

    def __post_init__(self) -> None:
        root = Path(self.output_root).absolute()
        if root.exists() or root.is_symlink():
            raise ValueError("parallel benchmark output root must be fresh")
        if tuple(map(int, self.worker_counts)) != REQUIRED_WORKER_COUNTS:
            raise ValueError("production benchmark worker counts are fixed at 1, 4, and 8")
        fixture = (
            None if self.fixture_plan_path is None else Path(self.fixture_plan_path).absolute()
        )
        preflight = (
            None
            if self.preflight_scope_input_set_manifest is None
            else Path(self.preflight_scope_input_set_manifest).absolute()
        )
        if fixture is None and preflight is None:
            raise ValueError("parallel benchmark requires a fixture plan or preflight inputs")
        scope_ids = tuple(map(str, self.preflight_scope_ids))
        if scope_ids and preflight is None:
            raise ValueError("preflight scope selection requires a preflight input set")
        object.__setattr__(self, "output_root", root)
        object.__setattr__(self, "fixture_plan_path", fixture)
        object.__setattr__(
            self,
            "preflight_scope_input_set_manifest",
            preflight,
        )
        object.__setattr__(self, "preflight_scope_ids", scope_ids)
        object.__setattr__(self, "worker_counts", REQUIRED_WORKER_COUNTS)


def run_stage1_parallel_benchmark(
    options: Stage1ParallelBenchmarkOptions,
) -> Mapping[str, Any]:
    """Run and seal one fresh benchmark."""

    root = options.output_root
    root.mkdir(parents=True, exist_ok=False)
    fixture: Mapping[str, Any] | None = None
    fixture_identity: Mapping[str, Any] | None = None
    families: list[tuple[str, tuple[Mapping[str, Any], ...]]] = []
    if options.fixture_plan_path is not None:
        fixture = load_fixture_plan(options.fixture_plan_path)
        fixture_identity = {
            "path": str(options.fixture_plan_path.resolve(strict=True)),
            "file_sha256": _sha256_file(options.fixture_plan_path),
            "content_sha256": str(fixture["content_sha256"]),
        }
        if fixture["cluster_preflight_jobs"]:
            families.append(
                (
                    "cluster_preflight_fixture",
                    tuple(fixture["cluster_preflight_jobs"]),
                )
            )
        if fixture["tfidf_jobs"]:
            families.append(("tfidf_fixture", tuple(fixture["tfidf_jobs"])))
    preflight_identity: Mapping[str, Any] | None = None
    if options.preflight_scope_input_set_manifest is not None:
        payloads, preflight_identity = _load_preflight_payloads(
            options.preflight_scope_input_set_manifest,
            selected_scope_ids=options.preflight_scope_ids,
        )
        families.append(("cluster_preflight_authenticated", payloads))
    if len({name for name, _payloads in families}) != len(families):
        raise RuntimeError("parallel benchmark family plan is duplicated")
    real_preflight = preflight_identity is not None
    code_identity = _source_identity(real_preflight=real_preflight)
    request_body = {
        "schema_version": BENCHMARK_REQUEST_SCHEMA,
        "worker_counts": list(REQUIRED_WORKER_COUNTS),
        "native_thread_limit": 1,
        "family_order": [name for name, _payloads in families],
        "fixture_input_identity": fixture_identity,
        "preflight_input_identity": preflight_identity,
        "source_code_identity": code_identity,
        "cohort_dataset_path_supplied": False,
        "oracle_input_supplied": False,
    }
    request = {**request_body, "content_sha256": _sha256_json(request_body)}
    request_path = root / "benchmark_request.json"
    _write_json_exclusive(request_path, request)

    result_paths: dict[str, Path] = {}
    result_values: dict[str, Mapping[str, Any]] = {}
    for family, payloads in families:
        value = _run_family(
            kind=family,
            payloads=payloads,
            worker_counts=REQUIRED_WORKER_COUNTS,
        )
        path = root / f"{family}.json"
        _write_json_exclusive(path, value)
        result_paths[family] = path
        result_values[family] = value
    if _source_identity(real_preflight=real_preflight) != code_identity:
        raise RuntimeError("benchmark source code changed during execution")
    summary_body = {
        "schema_version": "production_stage1_parallel_benchmark_summary_v1",
        "status": "accepted",
        "request_content_sha256": request["content_sha256"],
        "family_order": list(result_values),
        "families": {
            name: {
                "job_count": value["job_count"],
                "canonical_scientific_identity_sha256": value[
                    "canonical_scientific_identity_sha256"
                ],
                "exact_scientific_equality_across_worker_counts": True,
                "wall_seconds_by_workers": {
                    str(row["requested_workers"]): row["wall_seconds"] for row in value["runs"]
                },
                "speedup_vs_one_worker": {
                    str(row["requested_workers"]): row["speedup_vs_one_worker"]
                    for row in value["runs"]
                },
            }
            for name, value in result_values.items()
        },
    }
    summary = {**summary_body, "content_sha256": _sha256_json(summary_body)}
    summary_path = root / "benchmark_summary.json"
    _write_json_exclusive(summary_path, summary)
    _fsync_directory(root)

    files = {
        "request": _registration(request_path, root=root),
        "summary": _registration(summary_path, root=root),
        "results": {name: _registration(path, root=root) for name, path in result_paths.items()},
    }
    terminal_body = {
        "schema_version": BENCHMARK_TERMINAL_SCHEMA,
        "status": "accepted",
        "request_content_sha256": request["content_sha256"],
        "summary_content_sha256": summary["content_sha256"],
        "family_order": list(result_values),
        "files": files,
        "terminal_manifest_written_last": True,
    }
    terminal = {
        **terminal_body,
        "content_sha256": _sha256_json(terminal_body),
    }
    _write_json_exclusive(root / BENCHMARK_TERMINAL_NAME, terminal)
    _fsync_directory(root)
    return validate_stage1_parallel_benchmark(root)


def validate_stage1_parallel_benchmark(
    output_root: Path | str,
) -> Mapping[str, Any]:
    """Fresh path-only validation of one terminal benchmark."""

    root = Path(output_root).absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("parallel benchmark root is invalid")
    terminal_path = root / BENCHMARK_TERMINAL_NAME
    terminal = _read_closed_json(
        terminal_path,
        label="parallel benchmark terminal manifest",
    )
    terminal_body = {
        key: copy.deepcopy(value) for key, value in terminal.items() if key != "content_sha256"
    }
    required = {
        "schema_version",
        "status",
        "request_content_sha256",
        "summary_content_sha256",
        "family_order",
        "files",
        "terminal_manifest_written_last",
        "content_sha256",
    }
    if (
        set(terminal) != required
        or terminal.get("schema_version") != BENCHMARK_TERMINAL_SCHEMA
        or terminal.get("status") != "accepted"
        or terminal.get("terminal_manifest_written_last") is not True
        or terminal.get("content_sha256") != _sha256_json(terminal_body)
    ):
        raise ValueError("parallel benchmark terminal manifest is invalid")
    files = terminal.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "request",
        "summary",
        "results",
    }:
        raise ValueError("parallel benchmark file registry is invalid")
    request_path = _validate_registration(root, files["request"], label="request")
    summary_path = _validate_registration(root, files["summary"], label="summary")
    results = files["results"]
    family_order = terminal.get("family_order")
    if (
        not isinstance(results, Mapping)
        or not isinstance(family_order, list)
        or set(results) != set(map(str, family_order))
        or len(family_order) != len(set(map(str, family_order)))
    ):
        raise ValueError("parallel benchmark result registry changed")
    request = _read_closed_json(request_path, label="benchmark request")
    request_body = {
        key: copy.deepcopy(value) for key, value in request.items() if key != "content_sha256"
    }
    if (
        request.get("schema_version") != BENCHMARK_REQUEST_SCHEMA
        or request.get("content_sha256") != _sha256_json(request_body)
        or request.get("content_sha256") != terminal.get("request_content_sha256")
        or request.get("worker_counts") != list(REQUIRED_WORKER_COUNTS)
        or request.get("oracle_input_supplied") is not False
        or request.get("cohort_dataset_path_supplied") is not False
    ):
        raise ValueError("parallel benchmark request changed")
    summary = _read_closed_json(summary_path, label="benchmark summary")
    summary_body = {
        key: copy.deepcopy(value) for key, value in summary.items() if key != "content_sha256"
    }
    if (
        summary.get("content_sha256") != _sha256_json(summary_body)
        or summary.get("content_sha256") != terminal.get("summary_content_sha256")
        or summary.get("status") != "accepted"
        or summary.get("family_order") != family_order
    ):
        raise ValueError("parallel benchmark summary changed")
    expected_files = {
        BENCHMARK_TERMINAL_NAME,
        str(files["request"]["relative_path"]),
        str(files["summary"]["relative_path"]),
    }
    for family in family_order:
        registration = results.get(family)
        result_path = _validate_registration(
            root,
            registration,
            label=f"{family} result",
        )
        expected_files.add(str(registration["relative_path"]))
        result = _read_closed_json(result_path, label=f"{family} result")
        result_body = {
            key: copy.deepcopy(value) for key, value in result.items() if key != "content_sha256"
        }
        runs = result.get("runs")
        if (
            result.get("schema_version") != BENCHMARK_RESULT_SCHEMA
            or result.get("kind") != family
            or result.get("content_sha256") != _sha256_json(result_body)
            or result.get("required_worker_counts") != list(REQUIRED_WORKER_COUNTS)
            or result.get("exact_scientific_equality_across_worker_counts") is not True
            or not isinstance(runs, list)
            or [row.get("requested_workers") for row in runs] != list(REQUIRED_WORKER_COUNTS)
            or len({row.get("scientific_identity_sha256") for row in runs}) != 1
            or any(
                not math.isfinite(float(row.get("wall_seconds", float("nan"))))
                or float(row["wall_seconds"]) <= 0.0
                or not math.isfinite(float(row.get("speedup_vs_one_worker", float("nan"))))
                or any(
                    cap.get("all_observed_pools_limited_to_one") is not True
                    or any(
                        int(pool.get("num_threads", -1)) != 1
                        for pool in cap.get("effective_pools") or ()
                    )
                    for cap in row.get("native_thread_caps") or ()
                )
                for row in runs
            )
        ):
            raise ValueError(f"{family} benchmark result is invalid")
    observed_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("parallel benchmark tree contains a symlink")
    if observed_files != expected_files:
        raise ValueError("parallel benchmark tree contains unregistered files")
    if terminal_path.stat().st_mtime_ns < max(
        path.stat().st_mtime_ns
        for path in root.rglob("*")
        if path.is_file() and path != terminal_path
    ):
        raise ValueError("parallel benchmark terminal manifest was not written last")
    return copy.deepcopy(summary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark identical Stage 1 CPU work at 1, 4, and 8 loky workers "
            "and require exact canonical scientific equality."
        )
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--fixture-plan", type=Path)
    parser.add_argument("--preflight-scope-input-set-manifest", type=Path)
    parser.add_argument("--preflight-scope-id", action="append", default=[])
    parser.add_argument("--write-fixture-plan", type=Path)
    parser.add_argument("--fixture-job-count", type=int, default=8)
    parser.add_argument("--fixture-workload-scale", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.write_fixture_plan is not None:
        if (
            args.output_root is not None
            or args.fixture_plan is not None
            or args.preflight_scope_input_set_manifest is not None
            or args.preflight_scope_id
        ):
            raise ValueError("--write-fixture-plan cannot be combined with a benchmark run")
        written = write_fixture_plan(
            args.write_fixture_plan,
            job_count=int(args.fixture_job_count),
            workload_scale=int(args.fixture_workload_scale),
            seed=int(args.seed),
        )
        print(str(written))
        return 0
    if args.output_root is None:
        raise ValueError("--output-root is required for a benchmark run")
    result = run_stage1_parallel_benchmark(
        Stage1ParallelBenchmarkOptions(
            output_root=args.output_root,
            fixture_plan_path=args.fixture_plan,
            preflight_scope_input_set_manifest=(args.preflight_scope_input_set_manifest),
            preflight_scope_ids=tuple(args.preflight_scope_id),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BENCHMARK_FIXTURE_PLAN_SCHEMA",
    "BENCHMARK_TERMINAL_NAME",
    "REQUIRED_WORKER_COUNTS",
    "Stage1ParallelBenchmarkOptions",
    "load_fixture_plan",
    "main",
    "make_fixture_plan",
    "run_stage1_parallel_benchmark",
    "validate_stage1_parallel_benchmark",
    "write_fixture_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
