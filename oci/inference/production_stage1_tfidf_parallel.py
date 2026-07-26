"""Process-isolated cumulative-spent TF-IDF scheduling.

Each task contains one canonical cumulative review scope only: its spent
text/labels, sealed row IDs, replay canary, and immutable scientific config.
No cohort dataframe or source-dataset path is accepted.  The parent restores
canonical order after loky execution and rejects missing, duplicate, reordered,
or substituted results.
"""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
import math
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from ..config import (
    AppliedInferenceConfig,
    ModelArchitectureConfig,
)
from .stage1_cumulative_spent_evidence import (
    CumulativeSpentStage1FamilyRequest,
)
from .stage1_cumulative_spent_native_adapters import (
    CumulativeSpentReplayCanary,
)
from .stage1_cumulative_spent_remaining_adapters import (
    emit_cumulative_spent_tfidf_capture,
)

TFIDF_CUMULATIVE_TASK_SCHEMA = "production_cumulative_tfidf_scope_task_v1"
TFIDF_CUMULATIVE_RESULT_SCHEMA = "production_cumulative_tfidf_scope_result_v1"
TFIDF_SPENT_ONLY_DATASET_MARKER = "__spent_only_in_memory__"
_SCOPE_ID = re.compile(r"^outer_[0-9]{3}_hierarchy_epoch_[0-9]{3}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_DIRECT_CALL_SEED = 42


def _configured_tfidf_seed(
    config: AppliedInferenceConfig,
    *,
    override: int | None = None,
) -> int:
    """Resolve the compatibility config's nullable seed without changing it."""

    value = config.seed if override is None else override
    if value is None:
        value = _LEGACY_DIRECT_CALL_SEED
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("cumulative TF-IDF seed must be a nonnegative integer")
    return int(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _serialized_loky_import_bootstrap(
    *,
    module_name: str,
    lock_identity: str,
) -> Mapping[str, Any]:
    """Serialize heavy worker imports across loky children on FUSE/SSHFS."""

    lock_digest = hashlib.sha256(
        str(lock_identity).encode("utf-8")
    ).hexdigest()
    lock_path = (
        Path(tempfile.gettempdir())
        / f"oci-stage1-loky-import-{lock_digest}.lock"
    )
    code = (
        "import fcntl, importlib\n"
        f"with open({str(lock_path)!r}, 'a+b') as _oci_import_lock:\n"
        "    fcntl.flock(_oci_import_lock.fileno(), fcntl.LOCK_EX)\n"
        f"    importlib.import_module({str(module_name)!r})\n"
        "    fcntl.flock(_oci_import_lock.fileno(), fcntl.LOCK_UN)\n"
    )
    return {
        "initializer": builtins.exec,
        "initargs": (code,),
    }


def derive_cumulative_tfidf_scope_seed(
    *,
    global_seed: int,
    scope_id: str,
) -> int:
    digest = hashlib.sha256(
        f"{int(global_seed)}\0{str(scope_id)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def project_tfidf_worker_config(
    source: AppliedInferenceConfig,
    *,
    seed: int | None = None,
) -> AppliedInferenceConfig:
    """Copy only TF-IDF scientific settings and remove external locators."""

    if not isinstance(source, AppliedInferenceConfig):
        raise TypeError("TF-IDF config projection requires AppliedInferenceConfig")
    nn_config = copy.deepcopy(source.architecture.multi_model_forest)
    nn_config.prespecified_features_json = None
    nn_config.prespecified_features = []
    nn_config.prespecified_confounders = []
    nn_config.prespecified_effect_modifiers = []
    nn_config.embedding_contrast.cache_dir = None
    nn_config.embedding_contrast.external_corpus_cache_dirs = []
    projected = AppliedInferenceConfig(
        clinical_question=None,
        outcome_type=source.outcome_type,
        dataset_path=TFIDF_SPENT_ONLY_DATASET_MARKER,
        text_column=source.text_column,
        outcome_column=source.outcome_column,
        treatment_column=source.treatment_column,
        split_column=source.split_column,
        cv_folds=int(source.cv_folds),
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=nn_config,
        ),
    )
    projected.seed = _configured_tfidf_seed(source, override=seed)
    return projected


@dataclass(frozen=True)
class CumulativeTfidfScopeTask:
    """One pickle-safe, spent-only cumulative TF-IDF fit request."""

    canonical_index: int
    scope_id: str
    family_order: tuple[str, ...]
    requests: Mapping[str, CumulativeSpentStage1FamilyRequest]
    replay_canary: CumulativeSpentReplayCanary
    config: AppliedInferenceConfig
    component_root: Path
    artifact_dir: Path
    execution_record_dir: Path
    proof_dir: Path

    def __post_init__(self) -> None:
        if int(self.canonical_index) < 0:
            raise ValueError("cumulative TF-IDF canonical index must be nonnegative")
        if _SCOPE_ID.fullmatch(str(self.scope_id)) is None:
            raise ValueError("cumulative TF-IDF scope ID is not canonical")
        if not isinstance(self.config, AppliedInferenceConfig):
            raise TypeError("cumulative TF-IDF task requires AppliedInferenceConfig")
        if str(self.config.dataset_path) != TFIDF_SPENT_ONLY_DATASET_MARKER:
            raise ValueError(
                "cumulative TF-IDF task config cannot expose a cohort dataset path"
            )
        nn_config = self.config.architecture.multi_model_forest
        if (
            nn_config.prespecified_features_json is not None
            or nn_config.embedding_contrast.cache_dir is not None
            or nn_config.embedding_contrast.external_corpus_cache_dirs
        ):
            raise ValueError(
                "cumulative TF-IDF task config exposes an external locator"
            )
        families = tuple(map(str, self.family_order))
        if not families or len(families) != len(set(families)):
            raise ValueError("cumulative TF-IDF family order is invalid")
        if set(self.requests) != set(families):
            raise ValueError("cumulative TF-IDF task has incomplete family requests")
        reference: CumulativeSpentStage1FamilyRequest | None = None
        for family in families:
            request = self.requests[family]
            if not isinstance(request, CumulativeSpentStage1FamilyRequest):
                raise TypeError("cumulative TF-IDF request is not typed")
            if request.family != family or request.scope_id != self.scope_id:
                raise ValueError("cumulative TF-IDF request changed scope or family")
            if reference is None:
                reference = request
            elif (
                request.data_projection_sha256
                != reference.data_projection_sha256
                or request.spent_rows != reference.spent_rows
                or request.sealed_row_ids != reference.sealed_row_ids
            ):
                raise ValueError(
                    "cumulative TF-IDF family requests do not share one spent-only projection"
                )
        assert reference is not None
        self.replay_canary.assert_matches(reference)
        root = Path(self.component_root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("cumulative TF-IDF component root must already exist")
        resolved_root = root.resolve(strict=True)
        output_paths = (
            Path(self.artifact_dir),
            Path(self.execution_record_dir),
            Path(self.proof_dir),
        )
        if len(set(output_paths)) != len(output_paths):
            raise ValueError("cumulative TF-IDF task output directories must be unique")
        for path in output_paths:
            if path.exists() or path.is_symlink():
                raise ValueError(
                    "cumulative TF-IDF scope output directories must be fresh"
                )
            try:
                path.parent.resolve(strict=True).relative_to(resolved_root)
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(
                    "cumulative TF-IDF scope output escapes its component"
                ) from exc
        object.__setattr__(self, "canonical_index", int(self.canonical_index))
        object.__setattr__(self, "scope_id", str(self.scope_id))
        object.__setattr__(self, "family_order", families)
        object.__setattr__(self, "component_root", resolved_root)
        object.__setattr__(self, "artifact_dir", output_paths[0])
        object.__setattr__(self, "execution_record_dir", output_paths[1])
        object.__setattr__(self, "proof_dir", output_paths[2])

    @property
    def input_identity(self) -> Mapping[str, Any]:
        reference = self.requests[self.family_order[0]]
        scope_seed = derive_cumulative_tfidf_scope_seed(
            global_seed=_configured_tfidf_seed(self.config),
            scope_id=self.scope_id,
        )
        body = {
            "schema_version": TFIDF_CUMULATIVE_TASK_SCHEMA,
            "canonical_index": self.canonical_index,
            "scope_id": self.scope_id,
            "family_order": list(self.family_order),
            "request_binding_sha256_by_family": {
                family: self.requests[family].binding_sha256
                for family in self.family_order
            },
            "data_projection_sha256": reference.data_projection_sha256,
            "spent_row_ids": list(reference.spent_row_ids),
            "sealed_row_ids": list(reference.sealed_row_ids),
            "replay_canary": dict(self.replay_canary.binding),
            "global_seed": _configured_tfidf_seed(self.config),
            "scope_seed": scope_seed,
            "scientific_tfidf_config_sha256": _sha256_json(
                asdict(self.config)
            ),
            "cohort_frame_supplied": False,
            "source_cohort_locator_supplied": False,
            "embedding_cache_locator_supplied": False,
            "external_corpus_locator_supplied": False,
            "split_registry_row_ids_only": True,
        }
        return {**body, "content_sha256": _sha256_json(body)}


def execute_cumulative_tfidf_scope_task(
    task: CumulativeTfidfScopeTask,
) -> Mapping[str, Any]:
    """Fit and register one scope inside the same live loky worker."""

    if not isinstance(task, CumulativeTfidfScopeTask):
        raise TypeError("cumulative TF-IDF worker requires one typed task")
    with threadpool_limits(limits=1):
        emissions = emit_cumulative_spent_tfidf_capture(
            requests=task.requests,
            replay_canary=task.replay_canary,
            config=task.config,
            artifact_dir=task.artifact_dir,
            execution_record_dir=task.execution_record_dir,
        )
        # Import lazily: the registration helper currently lives in the
        # production bundle, while the scheduler itself remains independently
        # importable and testable.
        from .production_stage1_bundle import (
            _register_cumulative_spent_remaining_scope,
        )

        registration = _register_cumulative_spent_remaining_scope(
            component_root=task.component_root,
            proof_directory=task.proof_dir,
            requests=task.requests,
            replay_canary=task.replay_canary,
            emissions=emissions,
            families=task.family_order,
        )
    body = {
        "schema_version": TFIDF_CUMULATIVE_RESULT_SCHEMA,
        "canonical_index": task.canonical_index,
        "scope_id": task.scope_id,
        "task_input_identity_sha256": task.input_identity["content_sha256"],
        "registration": registration,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _execute_bounded(
    task: CumulativeTfidfScopeTask,
    executor: Callable[[CumulativeTfidfScopeTask], Mapping[str, Any]],
) -> Mapping[str, Any]:
    from .production_stage1_scope_scheduler import (
        _enforce_stage1_torch_determinism,
        seed_stage1_scope_rngs,
    )

    _enforce_stage1_torch_determinism()
    seed_stage1_scope_rngs(
        derive_cumulative_tfidf_scope_seed(
            global_seed=int(task.config.seed),
            scope_id=task.scope_id,
        ),
        gpu_id=None,
    )
    with threadpool_limits(limits=1):
        return executor(task)


def _validate_results(
    tasks: Sequence[CumulativeTfidfScopeTask],
    results: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    expected_order = tuple(task.scope_id for task in tasks)
    if len(results) != len(tasks):
        raise RuntimeError("cumulative TF-IDF execution returned missing results")
    by_scope: dict[str, Mapping[str, Any]] = {}
    for raw in results:
        if not isinstance(raw, Mapping):
            raise TypeError("cumulative TF-IDF worker result is not a mapping")
        result = dict(raw)
        body = dict(result)
        declared = body.pop("content_sha256", None)
        required = {
            "schema_version",
            "canonical_index",
            "scope_id",
            "task_input_identity_sha256",
            "registration",
            "content_sha256",
        }
        scope_id = str(result.get("scope_id") or "")
        if (
            set(result) != required
            or result.get("schema_version") != TFIDF_CUMULATIVE_RESULT_SCHEMA
            or _SHA256.fullmatch(str(declared or "")) is None
            or _sha256_json(body) != declared
            or scope_id in by_scope
        ):
            raise ValueError(
                "cumulative TF-IDF execution returned a duplicate or invalid result"
            )
        by_scope[scope_id] = result
    if set(by_scope) != set(expected_order):
        raise RuntimeError("cumulative TF-IDF execution returned incomplete scope coverage")
    ordered: list[Mapping[str, Any]] = []
    for canonical_index, task in enumerate(tasks):
        result = by_scope[task.scope_id]
        if (
            task.canonical_index != canonical_index
            or result.get("canonical_index") != canonical_index
            or result.get("task_input_identity_sha256")
            != task.input_identity["content_sha256"]
            or not isinstance(result.get("registration"), Mapping)
        ):
            raise ValueError(
                "cumulative TF-IDF result was reordered or substituted"
            )
        ordered.append(result)
    return tuple(ordered)


def run_cumulative_tfidf_scope_tasks(
    *,
    tasks: Sequence[CumulativeTfidfScopeTask],
    workers: int,
    executor: Callable[
        [CumulativeTfidfScopeTask], Mapping[str, Any]
    ] = execute_cumulative_tfidf_scope_task,
) -> tuple[Mapping[str, Any], ...]:
    """Run canonical tasks serially or with loky and return canonical results."""

    ordered_tasks = tuple(tasks)
    if not ordered_tasks:
        raise ValueError("cumulative TF-IDF task plan cannot be empty")
    resolved_workers = int(workers)
    if resolved_workers < 1 or not math.isfinite(float(workers)):
        raise ValueError("cumulative TF-IDF worker count must be positive")
    scope_ids = tuple(task.scope_id for task in ordered_tasks)
    if (
        len(scope_ids) != len(set(scope_ids))
        or any(
            task.canonical_index != index
            for index, task in enumerate(ordered_tasks)
        )
    ):
        raise ValueError(
            "cumulative TF-IDF task plan is duplicated, missing, or reordered"
        )
    output_directories = [
        str(path)
        for task in ordered_tasks
        for path in (
            task.artifact_dir,
            task.execution_record_dir,
            task.proof_dir,
        )
    ]
    if len(output_directories) != len(set(output_directories)):
        raise ValueError("cumulative TF-IDF tasks alias an output directory")
    job_count = min(len(ordered_tasks), resolved_workers)
    if job_count == 1:
        results = tuple(_execute_bounded(task, executor) for task in ordered_tasks)
    else:
        bootstrap = _serialized_loky_import_bootstrap(
            module_name=__name__,
            lock_identity=str(ordered_tasks[0].component_root),
        )
        with parallel_config(
            backend="loky",
            n_jobs=job_count,
            inner_max_num_threads=1,
        ):
            results = tuple(
                Parallel(
                    batch_size=1,
                    pre_dispatch="all",
                    **bootstrap,
                )(
                    delayed(_execute_bounded)(task, executor)
                    for task in ordered_tasks
                )
            )
    return _validate_results(ordered_tasks, results)


__all__ = [
    "CumulativeTfidfScopeTask",
    "TFIDF_CUMULATIVE_RESULT_SCHEMA",
    "TFIDF_SPENT_ONLY_DATASET_MARKER",
    "TFIDF_CUMULATIVE_TASK_SCHEMA",
    "execute_cumulative_tfidf_scope_task",
    "project_tfidf_worker_config",
    "run_cumulative_tfidf_scope_tasks",
]
