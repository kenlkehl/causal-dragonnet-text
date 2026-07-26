"""Two-phase role-neutral BoW execution for one physical Stage 1 group.

This module is the first native execution pattern for physical-fit reuse.  It
does not replace the all-ten-family production worker yet.  It establishes the
ordering that worker must follow:

1. receive only one physical owner and its complete logical group;
2. fit and authenticate BoW nuisance and residual-effect state from the
   owner's fit rows;
3. publish a fit-only seal for each completed family;
4. publish cumulative-review views as references without opening sealed text;
5. only then request exact-inner held-out text and transform it with the live
   fitted objects for both families.

No held-out treatment or outcome argument exists.  Text is never sliced or
truncated here.  Vocabulary size, n-gram range, fold count, learner settings,
clipping, and seeds all come from the typed request/configuration.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from ..config import BoWViewConfig
from .all_evidence_discovery_interfaces import BOW_NUISANCE, BOW_R_LOSS
from .bow_native_proof_capture import (
    _ArrayStore,
    _array_sha256,
    _capture_learner,
    _capture_vectorizer,
    _predict_learner,
    _restore_vectorizer,
    _text_sha256,
)
from .lossless_stage1_evidence_catalog import (
    NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
)
from .multi_model_forest_stage1 import (
    _binary_split_items,
    _bounded_fold_count,
    _bow_view_to_dict,
    _fit_regressor,
    _make_bow_classifier,
    _make_bow_regressor,
    _make_bow_vectorizer,
    _model_params,
    _vectorizer_params,
)
from .production_stage1_legacy_scope_fragments import (
    LEGACY_STAGE1_FIT_ONLY_FAMILY_SEAL_SCHEMA,
)
from .production_role_neutral_stage2_handoff import (
    ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD,
    build_role_neutral_stage2_fit_projection_proof,
    validate_role_neutral_stage2_fit_projection_proof,
)
from .production_stage1_scope_scheduler import Stage1ScopePlan, Stage1ScopeSpec

ROLE_NEUTRAL_BOW_GROUP_REQUEST_SCHEMA = "production_role_neutral_bow_physical_group_request_v1"
ROLE_NEUTRAL_BOW_FIT_STATE_SCHEMA = "production_role_neutral_bow_fit_state_v2"
ROLE_NEUTRAL_BOW_LOGICAL_VIEW_SCHEMA = "production_role_neutral_bow_logical_view_v2"
ROLE_NEUTRAL_BOW_GROUP_EXECUTION_SCHEMA = "production_role_neutral_bow_group_execution_v2"

_FIT_STATE_DIRECTORY = "fit_state"
_FIT_STATE_METADATA = "metadata.json"
_FIT_SEAL_FILES = {
    BOW_NUISANCE: "fit_only_family_seal.json",
    BOW_R_LOSS: "fit_only_bow_r_loss_family_seal.json",
}
_LOGICAL_VIEW_DIRECTORY = "logical_views"
_TERMINAL_FILE = "execution_manifest.json"
_HEX = frozenset("0123456789abcdef")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _sha256_file(path: Path) -> tuple[str, int]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"artifact is not one regular file: {target}")
    before = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise ValueError(f"artifact file is linked or nonregular: {target}")
    digest = hashlib.sha256()
    size = 0
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    after = target.stat(follow_symlinks=False)
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if tuple(getattr(before, field) for field in identity) != tuple(
        getattr(after, field) for field in identity
    ) or size != int(after.st_size):
        raise RuntimeError(f"artifact changed while hashing: {target}")
    return digest.hexdigest(), size


def _tree_sha256(root: Path) -> str:
    tree = Path(root)
    if tree.is_symlink() or not tree.is_dir():
        raise ValueError("fit-state tree must be one real directory")
    rows: list[dict[str, Any]] = []
    for path in sorted(tree.rglob("*"), key=lambda item: item.relative_to(tree).as_posix()):
        relative = path.relative_to(tree).as_posix()
        if path.is_symlink():
            raise ValueError("fit-state tree cannot contain symbolic links")
        if path.is_dir():
            rows.append({"path": relative, "kind": "directory"})
            continue
        digest, size = _sha256_file(path)
        rows.append(
            {
                "path": relative,
                "kind": "file",
                "sha256": digest,
                "size_bytes": size,
            }
        )
    if not rows:
        raise ValueError("fit-state tree is empty")
    return _sha256_json(
        {
            "schema_version": "production_role_neutral_bow_tree_v1",
            "inventory": rows,
        }
    )


def _write_new_bytes(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace immutable artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_bytes(
        path,
        (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_new_npy(path: Path, value: np.ndarray) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace immutable array: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("role-neutral BoW arrays cannot use object dtype")
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = Path(path).read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _row_order_fingerprint(row_ids: Sequence[int]) -> str:
    return _sha256_json([int(row_id) for row_id in row_ids])


def _binary_vector(values: Sequence[Any], *, label: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (int(length),) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be one finite vector aligned to fit rows")
    if not set(np.unique(array)).issubset({0.0, 1.0}):
        raise ValueError(f"{label} must be binary")
    return array.astype(np.int64)


def _float_hex_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return _sha256_json([float(value).hex() for value in array])


def _derived_seed(
    group_seed: int,
    *,
    view_name: str,
    objective: str,
    purpose: str,
    fold: int = 0,
) -> int:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": "production_role_neutral_bow_seed_v1",
                "group_seed": int(group_seed),
                "view_name": str(view_name),
                "objective": str(objective),
                "purpose": str(purpose),
                "fold": int(fold),
            }
        ).encode("utf-8")
    ).digest()
    result = int.from_bytes(digest[:8], "big") % (2**31 - 1)
    return result or 1


@dataclass(frozen=True)
class RoleNeutralBoWPhysicalGroupRequest:
    """Closed authority passed to one physical BoW worker."""

    plan_scientific_content_sha256: str
    physical_owner: Stage1ScopeSpec
    logical_members: tuple[Stage1ScopeSpec, ...]
    content_sha256: str

    @classmethod
    def from_plan(
        cls,
        *,
        plan: Stage1ScopePlan,
        physical_owner_scope_id: str,
    ) -> "RoleNeutralBoWPhysicalGroupRequest":
        if not isinstance(plan, Stage1ScopePlan):
            raise TypeError("role-neutral BoW request requires a Stage1ScopePlan")
        owner = plan.scope(str(physical_owner_scope_id))
        if plan.physical_owner(owner.scope_id).scope_id != owner.scope_id:
            raise ValueError("role-neutral BoW request must name a physical owner")
        matches = [
            members
            for candidate, members in plan.physical_scope_groups
            if candidate.scope_id == owner.scope_id
        ]
        if len(matches) != 1:
            raise RuntimeError("physical owner has no unique logical group")
        members = matches[0]
        if members[0].scope_id != owner.scope_id:
            raise RuntimeError("physical logical group changed owner order")
        if any(
            tuple(member.fit_row_ids) != tuple(owner.fit_row_ids)
            or member.scope_seed != owner.scope_seed
            for member in members
        ):
            raise ValueError(
                "role-neutral BoW reuse requires identical ordered fit rows and seed"
            )
        aliases = members[1:]
        if aliases and (
            owner.scope_kind != "exact_inner"
            or any(member.scope_kind != "cumulative_spent" for member in aliases)
        ):
            raise ValueError("role-neutral BoW reuse supports exact-inner/cumulative groups only")
        body = {
            "schema_version": ROLE_NEUTRAL_BOW_GROUP_REQUEST_SCHEMA,
            "plan_scientific_content_sha256": (plan.scientific_content_sha256),
            "physical_owner": owner.as_dict(),
            "logical_members": [member.as_dict() for member in members],
            "logical_scope_count": len(members),
            "fit_row_ids": list(owner.fit_row_ids),
            "fit_row_order_fingerprint": _row_order_fingerprint(owner.fit_row_ids),
            "canonical_group_seed": int(owner.scope_seed),
            "heldout_labels_supplied": False,
            "peer_group_definitions_supplied": False,
        }
        return cls(
            plan_scientific_content_sha256=(plan.scientific_content_sha256),
            physical_owner=owner,
            logical_members=members,
            content_sha256=_sha256_json(body),
        )

    def as_dict(self) -> dict[str, Any]:
        _require_sha256(
            self.plan_scientific_content_sha256,
            label="role-neutral BoW plan identity",
        )
        if (
            not self.logical_members
            or self.logical_members[0].scope_id != self.physical_owner.scope_id
            or len({member.scope_id for member in self.logical_members})
            != len(self.logical_members)
            or any(
                tuple(member.fit_row_ids)
                != tuple(self.physical_owner.fit_row_ids)
                or member.scope_seed != self.physical_owner.scope_seed
                for member in self.logical_members
            )
            or (
                len(self.logical_members) > 1
                and (
                    self.physical_owner.scope_kind != "exact_inner"
                    or any(
                        member.scope_kind != "cumulative_spent"
                        for member in self.logical_members[1:]
                    )
                )
            )
        ):
            raise ValueError("role-neutral BoW logical group authority is invalid")
        body = {
            "schema_version": ROLE_NEUTRAL_BOW_GROUP_REQUEST_SCHEMA,
            "plan_scientific_content_sha256": (self.plan_scientific_content_sha256),
            "physical_owner": self.physical_owner.as_dict(),
            "logical_members": [member.as_dict() for member in self.logical_members],
            "logical_scope_count": len(self.logical_members),
            "fit_row_ids": list(self.physical_owner.fit_row_ids),
            "fit_row_order_fingerprint": _row_order_fingerprint(self.physical_owner.fit_row_ids),
            "canonical_group_seed": int(self.physical_owner.scope_seed),
            "heldout_labels_supplied": False,
            "peer_group_definitions_supplied": False,
        }
        if _sha256_json(body) != self.content_sha256:
            raise RuntimeError("role-neutral BoW group request changed")
        return {**body, "content_sha256": self.content_sha256}


@dataclass(frozen=True)
class AuthenticatedRoleNeutralBoWNuisanceBank:
    """Path-neutral nuisance probabilities authenticated from one BoW fit.

    The bank deliberately contains no treatment or outcome field.  Fit-row
    values are the cross-fitted ensemble nuisance probabilities sealed before
    registered held-out text is opened.  Held-out values are reconstructed
    from the exact logical transform and therefore remain label-free.
    """

    plan_scientific_content_sha256: str
    physical_owner_scope_id: str
    fit_row_ids: tuple[int, ...]
    heldout_row_ids: tuple[int, ...]
    fit_propensity_probability: tuple[float, ...]
    fit_outcome_nuisance_probability: tuple[float, ...]
    heldout_propensity_probability: tuple[float, ...]
    heldout_outcome_nuisance_probability: tuple[float, ...]
    source_terminal_content_sha256: str
    fit_state_artifact_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        fit_rows = tuple(map(int, self.fit_row_ids))
        heldout_rows = tuple(map(int, self.heldout_row_ids))
        if (
            not fit_rows
            or not heldout_rows
            or len(fit_rows) != len(set(fit_rows))
            or len(heldout_rows) != len(set(heldout_rows))
            or set(fit_rows).intersection(heldout_rows)
        ):
            raise ValueError("BoW nuisance bank row partitions are invalid")
        object.__setattr__(self, "fit_row_ids", fit_rows)
        object.__setattr__(self, "heldout_row_ids", heldout_rows)
        probability_fields = (
            ("fit_propensity_probability", len(fit_rows)),
            ("fit_outcome_nuisance_probability", len(fit_rows)),
            ("heldout_propensity_probability", len(heldout_rows)),
            ("heldout_outcome_nuisance_probability", len(heldout_rows)),
        )
        for name, expected_length in probability_fields:
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if (
                values.shape != (expected_length,)
                or not np.isfinite(values).all()
                or np.any(values < 0.0)
                or np.any(values > 1.0)
            ):
                raise ValueError(f"BoW nuisance bank {name} is not a probability vector")
            object.__setattr__(
                self,
                name,
                tuple(float(value) for value in values),
            )
        _require_sha256(
            self.plan_scientific_content_sha256,
            label="BoW nuisance bank plan identity",
        )
        _require_sha256(
            self.source_terminal_content_sha256,
            label="BoW nuisance bank terminal identity",
        )
        _require_sha256(
            self.fit_state_artifact_sha256,
            label="BoW nuisance bank fit-state identity",
        )
        if not str(self.physical_owner_scope_id).strip():
            raise ValueError("BoW nuisance bank physical owner is empty")
        if self.content_sha256 != _sha256_json(self.identity_payload()):
            raise ValueError("BoW nuisance bank content identity changed")

    def identity_payload(self) -> dict[str, Any]:
        """Return the compact scientific identity without materializing labels."""

        return {
            "schema_version": "production_role_neutral_bow_nuisance_bank_v1",
            "plan_scientific_content_sha256": (self.plan_scientific_content_sha256),
            "physical_owner_scope_id": self.physical_owner_scope_id,
            "fit_row_ids": list(self.fit_row_ids),
            "heldout_row_ids": list(self.heldout_row_ids),
            "fit_propensity_probability_sha256": _float_hex_sha256(
                np.asarray(self.fit_propensity_probability, dtype=np.float64)
            ),
            "fit_outcome_nuisance_probability_sha256": _float_hex_sha256(
                np.asarray(
                    self.fit_outcome_nuisance_probability,
                    dtype=np.float64,
                )
            ),
            "heldout_propensity_probability_sha256": _float_hex_sha256(
                np.asarray(
                    self.heldout_propensity_probability,
                    dtype=np.float64,
                )
            ),
            "heldout_outcome_nuisance_probability_sha256": _float_hex_sha256(
                np.asarray(
                    self.heldout_outcome_nuisance_probability,
                    dtype=np.float64,
                )
            ),
            "source_terminal_content_sha256": (self.source_terminal_content_sha256),
            "fit_state_artifact_sha256": self.fit_state_artifact_sha256,
            "heldout_treatment_field_present": False,
            "heldout_outcome_field_present": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "content_sha256": self.content_sha256,
        }


@dataclass
class _LiveFold:
    family: str
    objective: str
    view_name: str
    fold: int
    classification: bool
    vectorizer: TfidfVectorizer | None
    learner: Any | None
    constant_prediction: float | None
    captured_vectorizer: Mapping[str, Any] | None
    captured_learner: Mapping[str, Any]


@dataclass
class _UncapturedFold:
    """One fitted fold before deterministic parent-side proof capture."""

    fold: int
    fit_pos: np.ndarray
    validation_pos: np.ndarray
    fold_seed: int
    vectorizer: TfidfVectorizer | None
    learner: Any | None
    constant_prediction: float | None
    validation_prediction: np.ndarray
    fit_sample_weight: np.ndarray | None


def _resolve_fold_parallelism(
    *,
    setting: int | str,
    task_count: int,
    owner_cpu_budget: int,
) -> int:
    if isinstance(setting, bool):
        raise TypeError("bow_fold_parallelism must be an integer or 'auto'")
    budget = int(owner_cpu_budget)
    if isinstance(owner_cpu_budget, bool) or budget < 1:
        raise ValueError("BoW owner CPU budget must be a positive integer")
    if int(task_count) < 1:
        raise ValueError("BoW fold task count must be positive")
    text = str(setting).strip().lower()
    if text == "auto":
        requested = budget
    else:
        try:
            requested = int(text)
        except ValueError as exc:
            raise ValueError(
                "bow_fold_parallelism must be a positive integer or 'auto'"
            ) from exc
        if requested < 1 or str(requested) != text:
            raise ValueError(
                "bow_fold_parallelism must be a positive integer or 'auto'"
            )
    if requested > budget:
        raise ValueError(
            "bow_fold_parallelism cannot exceed the physical owner's CPU budget"
        )
    return min(requested, int(task_count))


def _parallel_backend_name(value: str) -> str:
    backend = str(value).strip().lower()
    if backend == "loky":
        backend = "processes"
    if backend not in {"threads", "processes"}:
        raise ValueError(
            "bow_parallel_backend must be 'threads', 'processes', or 'loky'"
        )
    return "threading" if backend == "threads" else "loky"


def _run_fold_tasks(
    *,
    run_fold: Callable[[int, np.ndarray, np.ndarray], _UncapturedFold],
    split_items: Sequence[tuple[int, tuple[Any, Any]]],
    bow_fold_parallelism: int | str,
    bow_parallel_backend: str,
    owner_cpu_budget: int,
) -> list[_UncapturedFold]:
    """Fit independent folds concurrently without sharing proof state."""

    n_jobs = _resolve_fold_parallelism(
        setting=bow_fold_parallelism,
        task_count=len(split_items),
        owner_cpu_budget=owner_cpu_budget,
    )
    backend = _parallel_backend_name(bow_parallel_backend)

    def invoke(
        fold: int,
        raw_fit_pos: Any,
        raw_validation_pos: Any,
    ) -> _UncapturedFold:
        # Every estimator constructed by this producer is already configured
        # for one learner job.  This process-wide/native-library limit also
        # prevents BLAS/OpenMP nesting from multiplying the configured fold
        # concurrency inside one physical-owner CPU allocation.
        with threadpool_limits(limits=1):
            return run_fold(
                int(fold),
                np.asarray(raw_fit_pos, dtype=int),
                np.asarray(raw_validation_pos, dtype=int),
            )

    # The outer limit keeps the process-wide native-library setting at one
    # until every threaded task has returned.  The per-task limit above also
    # applies the same rule inside a process-backend child.
    with threadpool_limits(limits=1):
        if n_jobs == 1:
            results = [
                invoke(fold, raw_fit_pos, raw_validation_pos)
                for fold, (raw_fit_pos, raw_validation_pos) in split_items
            ]
        else:
            results = Parallel(
                n_jobs=n_jobs,
                backend=backend,
                batch_size=1,
                pre_dispatch="all",
            )(
                delayed(invoke)(fold, raw_fit_pos, raw_validation_pos)
                for fold, (raw_fit_pos, raw_validation_pos) in split_items
            )
    ordered = sorted(results, key=lambda result: int(result.fold))
    expected = [int(fold) for fold, _split in split_items]
    if [int(result.fold) for result in ordered] != expected:
        raise RuntimeError("parallel BoW fold execution changed canonical fold coverage")
    return ordered


def _fit_one_binary_fold(
    *,
    request: RoleNeutralBoWPhysicalGroupRequest,
    fit_texts: tuple[str, ...],
    labels: np.ndarray,
    objective: str,
    view: BoWViewConfig,
    fold: int,
    fit_pos: np.ndarray,
    validation_pos: np.ndarray,
    vectorizer_params: Mapping[str, Any],
    model_params: Mapping[str, Any],
    e_clip: float,
) -> _UncapturedFold:
    fold_seed = _derived_seed(
        request.physical_owner.scope_seed,
        view_name=view.name,
        objective=objective,
        purpose="fold_model",
        fold=fold,
    )
    vectorizer: TfidfVectorizer | None = None
    learner: Any | None = None
    constant: float | None = None
    if len(np.unique(labels[fit_pos])) < 2:
        constant = float(np.mean(labels[fit_pos]))
        validation_prediction = np.full(
            len(validation_pos),
            constant,
            dtype=np.float64,
        )
    else:
        vectorizer = _make_bow_vectorizer(dict(vectorizer_params))
        x_fit = vectorizer.fit_transform(
            [fit_texts[int(position)] for position in fit_pos]
        )
        x_validation = vectorizer.transform(
            [fit_texts[int(position)] for position in validation_pos]
        )
        learner = _make_bow_classifier(
            dict(model_params),
            random_state=fold_seed,
        )
        learner.fit(x_fit, labels[fit_pos])
        validation_prediction = learner.predict_proba(x_validation)[:, 1]
    validation_prediction = np.clip(
        np.asarray(validation_prediction, dtype=np.float64),
        e_clip,
        1.0 - e_clip,
    )
    return _UncapturedFold(
        fold=int(fold),
        fit_pos=np.asarray(fit_pos, dtype=int),
        validation_pos=np.asarray(validation_pos, dtype=int),
        fold_seed=int(fold_seed),
        vectorizer=vectorizer,
        learner=learner,
        constant_prediction=constant,
        validation_prediction=validation_prediction,
        fit_sample_weight=None,
    )


def _fit_binary_objective(
    *,
    request: RoleNeutralBoWPhysicalGroupRequest,
    fit_texts: tuple[str, ...],
    labels: np.ndarray,
    objective: str,
    view: BoWViewConfig,
    view_index: int,
    nuisance_folds: int,
    e_clip: float,
    store: _ArrayStore,
    bow_fold_parallelism: int | str,
    bow_parallel_backend: str,
    owner_cpu_budget: int,
) -> tuple[list[_LiveFold], list[dict[str, Any]], np.ndarray]:
    split_seed = _derived_seed(
        request.physical_owner.scope_seed,
        view_name=view.name,
        objective=objective,
        purpose="cross_fit_split",
    )
    split_items = list(
        enumerate(
            _binary_split_items(
                labels,
                requested_folds=int(nuisance_folds),
                random_state=split_seed,
            ),
            start=1,
        )
    )
    oof = np.full(len(labels), np.nan, dtype=np.float64)
    live: list[_LiveFold] = []
    records: list[dict[str, Any]] = []
    vectorizer_params = _vectorizer_params(view)
    model_params = _model_params(view)
    fitted_folds = _run_fold_tasks(
        run_fold=lambda fold, fit_pos, validation_pos: _fit_one_binary_fold(
            request=request,
            fit_texts=fit_texts,
            labels=labels,
            objective=objective,
            view=view,
            fold=fold,
            fit_pos=fit_pos,
            validation_pos=validation_pos,
            vectorizer_params=vectorizer_params,
            model_params=model_params,
            e_clip=e_clip,
        ),
        split_items=split_items,
        bow_fold_parallelism=bow_fold_parallelism,
        bow_parallel_backend=bow_parallel_backend,
        owner_cpu_budget=owner_cpu_budget,
    )
    for fitted in fitted_folds:
        fold = int(fitted.fold)
        fit_pos = fitted.fit_pos
        validation_pos = fitted.validation_pos
        vectorizer = fitted.vectorizer
        learner = fitted.learner
        constant = fitted.constant_prediction
        validation_prediction = fitted.validation_prediction
        oof[validation_pos] = validation_prediction
        prefix = f"{objective}_{int(view_index):04d}_{int(fold):04d}"
        captured_vectorizer = (
            None
            if vectorizer is None
            else _capture_vectorizer(
                vectorizer,
                store,
                f"{prefix}_vectorizer",
                vectorizer_params=vectorizer_params,
            )
        )
        captured_learner = _capture_learner(
            learner,
            store,
            f"{prefix}_learner",
            classification=True,
            constant_prediction=constant,
        )
        record = {
            "family": BOW_NUISANCE,
            "objective": objective,
            "view_name": view.name,
            "view_config": _bow_view_to_dict(view),
            "fold": int(fold),
            "seed": int(fitted.fold_seed),
            "fit_row_ids": [
                int(request.physical_owner.fit_row_ids[int(position)]) for position in fit_pos
            ],
            "validation_row_ids": [
                int(request.physical_owner.fit_row_ids[int(position)])
                for position in validation_pos
            ],
            "fit_target": store.add(
                f"{prefix}_fit_target",
                labels[fit_pos],
            ),
            "validation_target": store.add(
                f"{prefix}_validation_target",
                labels[validation_pos],
            ),
            "validation_prediction": store.add(
                f"{prefix}_validation_prediction",
                validation_prediction,
            ),
            "vectorizer": captured_vectorizer,
            "learner": captured_learner,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        }
        records.append(record)
        live.append(
            _LiveFold(
                family=BOW_NUISANCE,
                objective=objective,
                view_name=view.name,
                fold=int(fold),
                classification=True,
                vectorizer=vectorizer,
                learner=learner,
                constant_prediction=constant,
                captured_vectorizer=captured_vectorizer,
                captured_learner=captured_learner,
            )
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("role-neutral BoW cross-fit left an unpredicted fit row")
    return live, records, oof


def _fit_one_regression_fold(
    *,
    request: RoleNeutralBoWPhysicalGroupRequest,
    fit_texts: tuple[str, ...],
    values: np.ndarray,
    weights: np.ndarray | None,
    objective: str,
    view: BoWViewConfig,
    fold: int,
    fit_pos: np.ndarray,
    validation_pos: np.ndarray,
    vectorizer_params: Mapping[str, Any],
    model_params: Mapping[str, Any],
) -> _UncapturedFold:
    fold_seed = _derived_seed(
        request.physical_owner.scope_seed,
        view_name=view.name,
        objective=objective,
        purpose="fold_model",
        fold=fold,
    )
    vectorizer = _make_bow_vectorizer(dict(vectorizer_params))
    x_fit = vectorizer.fit_transform(
        [fit_texts[int(position)] for position in fit_pos]
    )
    x_validation = vectorizer.transform(
        [fit_texts[int(position)] for position in validation_pos]
    )
    learner = _make_bow_regressor(
        dict(model_params),
        random_state=fold_seed,
    )
    fold_weight = None if weights is None else weights[fit_pos]
    _fit_regressor(
        learner,
        x_fit,
        values[fit_pos],
        sample_weight=fold_weight,
        unsupported_sample_weight_policy=str(
            model_params["unsupported_sample_weight_policy"]
        ),
    )
    validation_prediction = np.asarray(
        learner.predict(x_validation),
        dtype=np.float64,
    )
    if (
        validation_prediction.shape != (len(validation_pos),)
        or not np.isfinite(validation_prediction).all()
    ):
        raise RuntimeError("BoW residual-effect fold emitted invalid predictions")
    return _UncapturedFold(
        fold=int(fold),
        fit_pos=np.asarray(fit_pos, dtype=int),
        validation_pos=np.asarray(validation_pos, dtype=int),
        fold_seed=int(fold_seed),
        vectorizer=vectorizer,
        learner=learner,
        constant_prediction=None,
        validation_prediction=validation_prediction,
        fit_sample_weight=(
            None
            if fold_weight is None
            else np.asarray(fold_weight, dtype=np.float64)
        ),
    )


def _fit_regression_objective(
    *,
    request: RoleNeutralBoWPhysicalGroupRequest,
    fit_texts: tuple[str, ...],
    target: np.ndarray,
    sample_weight: np.ndarray | None,
    objective: str,
    view: BoWViewConfig,
    view_index: int,
    effect_folds: int,
    store: _ArrayStore,
    bow_fold_parallelism: int | str,
    bow_parallel_backend: str,
    owner_cpu_budget: int,
) -> tuple[list[_LiveFold], list[dict[str, Any]], np.ndarray]:
    values = np.asarray(target, dtype=np.float64)
    if values.shape != (len(fit_texts),) or not np.isfinite(values).all():
        raise ValueError("BoW residual-effect target must be finite and fit-row aligned")
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if (
            weights.shape != values.shape
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError("BoW residual-effect weights are invalid")
    fold_count = _bounded_fold_count(int(effect_folds), len(values))
    split_seed = _derived_seed(
        request.physical_owner.scope_seed,
        view_name=view.name,
        objective=objective,
        purpose="cross_fit_split",
    )
    splitter = KFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=split_seed,
    )
    vectorizer_params = _vectorizer_params(view)
    model_params = _model_params(view)
    oof = np.full(len(values), np.nan, dtype=np.float64)
    live: list[_LiveFold] = []
    records: list[dict[str, Any]] = []
    split_items = list(enumerate(splitter.split(fit_texts), start=1))
    fitted_folds = _run_fold_tasks(
        run_fold=lambda fold, fit_pos, validation_pos: _fit_one_regression_fold(
            request=request,
            fit_texts=fit_texts,
            values=values,
            weights=weights,
            objective=objective,
            view=view,
            fold=fold,
            fit_pos=fit_pos,
            validation_pos=validation_pos,
            vectorizer_params=vectorizer_params,
            model_params=model_params,
        ),
        split_items=split_items,
        bow_fold_parallelism=bow_fold_parallelism,
        bow_parallel_backend=bow_parallel_backend,
        owner_cpu_budget=owner_cpu_budget,
    )
    for fitted in fitted_folds:
        fold = int(fitted.fold)
        fit_pos = fitted.fit_pos
        validation_pos = fitted.validation_pos
        vectorizer = fitted.vectorizer
        learner = fitted.learner
        fold_weight = fitted.fit_sample_weight
        validation_prediction = fitted.validation_prediction
        if vectorizer is None or learner is None:
            raise RuntimeError("fitted BoW regression fold lost live model state")
        oof[validation_pos] = validation_prediction
        prefix = f"{objective}_{int(view_index):04d}_{int(fold):04d}"
        captured_vectorizer = _capture_vectorizer(
            vectorizer,
            store,
            f"{prefix}_vectorizer",
            vectorizer_params=vectorizer_params,
        )
        captured_learner = _capture_learner(
            learner,
            store,
            f"{prefix}_learner",
            classification=False,
            constant_prediction=None,
        )
        record = {
            "family": BOW_R_LOSS,
            "objective": objective,
            "view_name": view.name,
            "view_config": _bow_view_to_dict(view),
            "fold": int(fold),
            "seed": int(fitted.fold_seed),
            "fit_row_ids": [
                int(request.physical_owner.fit_row_ids[int(position)]) for position in fit_pos
            ],
            "validation_row_ids": [
                int(request.physical_owner.fit_row_ids[int(position)])
                for position in validation_pos
            ],
            "fit_target": store.add(
                f"{prefix}_fit_target",
                values[fit_pos],
            ),
            "validation_target": store.add(
                f"{prefix}_validation_target",
                values[validation_pos],
            ),
            "fit_sample_weight": (
                None
                if fold_weight is None
                else store.add(
                    f"{prefix}_fit_sample_weight",
                    fold_weight,
                )
            ),
            "validation_prediction": store.add(
                f"{prefix}_validation_prediction",
                validation_prediction,
            ),
            "vectorizer": captured_vectorizer,
            "learner": captured_learner,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        }
        records.append(record)
        live.append(
            _LiveFold(
                family=BOW_R_LOSS,
                objective=objective,
                view_name=view.name,
                fold=int(fold),
                classification=False,
                vectorizer=vectorizer,
                learner=learner,
                constant_prediction=None,
                captured_vectorizer=captured_vectorizer,
                captured_learner=captured_learner,
            )
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("BoW residual-effect cross-fit left an unpredicted fit row")
    return live, records, oof


def _evidence_payload(
    *,
    records: Sequence[Mapping[str, Any]],
    store: _ArrayStore,
    family: str,
) -> dict[str, Any]:
    if family not in {BOW_NUISANCE, BOW_R_LOSS}:
        raise ValueError("role-neutral BoW evidence requested another family")
    evidence: list[dict[str, Any]] = []
    for record in records:
        if record.get("family") != family:
            continue
        vectorizer = record.get("vectorizer")
        if vectorizer is None:
            evidence.append(
                {
                    "objective": record["objective"],
                    "view_name": record["view_name"],
                    "fold": int(record["fold"]),
                    "witness_kind": "constant_fit",
                    "constant_prediction": float(record["learner"]["constant_prediction"]),
                }
            )
            continue
        names = vectorizer.get("feature_names")
        idf_key = vectorizer.get("idf_array")
        if not isinstance(names, list) or not names or not isinstance(idf_key, str):
            raise RuntimeError("captured BoW vectorizer has no complete vocabulary")
        idf = np.asarray(store.arrays[idf_key], dtype=np.float64)
        if idf.shape != (len(names),):
            raise RuntimeError("captured BoW vocabulary and IDF state differ")
        for feature_index, (term, weight) in enumerate(zip(names, idf, strict=True)):
            evidence.append(
                {
                    "objective": record["objective"],
                    "view_name": record["view_name"],
                    "fold": int(record["fold"]),
                    "witness_kind": "fitted_tfidf_term",
                    "feature_index": int(feature_index),
                    "term": str(term),
                    "idf": float(weight),
                }
            )
    if not evidence:
        raise RuntimeError("role-neutral BoW fit produced no evidence atoms")
    return {
        "schema_version": NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
        "family": family,
        "architecture_evidence": evidence,
    }


def _producer_identity() -> str:
    functions = (
        _resolve_fold_parallelism,
        _parallel_backend_name,
        _run_fold_tasks,
        _fit_one_binary_fold,
        _fit_binary_objective,
        _fit_one_regression_fold,
        _fit_regression_objective,
        _evidence_payload,
        execute_role_neutral_bow_physical_group,
        _binary_split_items,
        _bounded_fold_count,
        _fit_regressor,
        _make_bow_vectorizer,
        _make_bow_classifier,
        _make_bow_regressor,
        _capture_vectorizer,
        _capture_learner,
        _restore_vectorizer,
        _predict_learner,
        build_role_neutral_stage2_fit_projection_proof,
        validate_role_neutral_stage2_fit_projection_proof,
    )
    return _sha256_json(
        {
            "schema_version": "production_role_neutral_bow_producer_identity_v1",
            "sources": [inspect.getsource(function) for function in functions],
        }
    )


def _write_fit_state(
    *,
    root: Path,
    request: RoleNeutralBoWPhysicalGroupRequest,
    fit_texts: tuple[str, ...],
    treatment: np.ndarray,
    outcome: np.ndarray,
    view_configs: tuple[dict[str, Any], ...],
    nuisance_folds: int,
    effect_folds: int,
    e_clip: float,
    records: Sequence[Mapping[str, Any]],
    oof_by_objective_view: Mapping[str, np.ndarray],
    derived_fit_quantities: Mapping[str, np.ndarray],
    store: _ArrayStore,
) -> tuple[dict[str, Any], str, str]:
    fit_root = root / _FIT_STATE_DIRECTORY
    arrays_root = fit_root / "arrays"
    arrays_root.mkdir(parents=True, exist_ok=False)
    oof_references = {
        name: store.add(f"oof_{index:04d}", values)
        for index, (name, values) in enumerate(sorted(oof_by_objective_view.items()))
    }
    derived_references = {
        name: store.add(f"derived_{index:04d}", values)
        for index, (name, values) in enumerate(sorted(derived_fit_quantities.items()))
    }
    array_inventory: dict[str, dict[str, Any]] = {}
    for key in sorted(store.arrays):
        path = arrays_root / f"{key}.npy"
        _write_new_npy(path, store.arrays[key])
        digest, size = _sha256_file(path)
        record = store.inventory[key]
        array_inventory[key] = {
            "relative_path": path.relative_to(fit_root).as_posix(),
            "dtype": record["dtype"],
            "shape": record["shape"],
            "content_sha256": record["content_sha256"],
            "file_sha256": digest,
            "size_bytes": size,
        }
    configuration = {
        "view_configs": [copy.deepcopy(value) for value in view_configs],
        "nuisance_folds": int(nuisance_folds),
        "effect_folds": int(effect_folds),
        "e_clip": float(e_clip),
        "outcome_type": "binary",
        "nuisance_source_policy": "mean_of_fit_row_oof_bow_views_v1",
        "pseudo_target_formula": "(outcome-m_hat)/(treatment-clipped_e_hat)",
        "r_weight_formula": "(treatment-clipped_e_hat)^2",
        "residual_effect_objectives": [
            "effect_pseudo_target",
            "effect_weighted_r",
        ],
        "text_truncation_applied": False,
    }
    configuration_identity = _sha256_json(configuration)
    body = {
        "schema_version": ROLE_NEUTRAL_BOW_FIT_STATE_SCHEMA,
        "group_request_content_sha256": request.content_sha256,
        "plan_scientific_content_sha256": (request.plan_scientific_content_sha256),
        "physical_owner_scope_id": request.physical_owner.scope_id,
        "physical_owner_scope_sha256": request.physical_owner.as_dict()["scope_sha256"],
        "fit_row_ids": list(request.physical_owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(request.physical_owner.fit_row_ids),
        "canonical_group_seed": int(request.physical_owner.scope_seed),
        "fit_text_sha256": _text_sha256(
            request.physical_owner.fit_row_ids,
            fit_texts,
        ),
        "fit_treatment_sha256": _float_hex_sha256(treatment),
        "fit_outcome_sha256": _float_hex_sha256(outcome),
        "configuration": configuration,
        "configuration_identity_sha256": configuration_identity,
        "producer_identity_sha256": _producer_identity(),
        "fold_records": copy.deepcopy(list(records)),
        "oof_predictions": oof_references,
        "derived_fit_quantities": derived_references,
        "array_inventory": array_inventory,
        "array_layout": "one_npy_per_array_mmap_safe_v1",
        "model_objects_retained_in_worker_memory": True,
        "registered_heldout_text_accessed": False,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
    }
    metadata = {**body, "content_sha256": _sha256_json(body)}
    _write_new_json(fit_root / _FIT_STATE_METADATA, metadata)
    fit_state_sha256 = _tree_sha256(fit_root)
    return metadata, fit_state_sha256, configuration_identity


def _fit_seal(
    *,
    request: RoleNeutralBoWPhysicalGroupRequest,
    family: str,
    evidence_payload: Mapping[str, Any],
    fit_state_sha256: str,
    configuration_identity_sha256: str,
) -> dict[str, Any]:
    if family not in {BOW_NUISANCE, BOW_R_LOSS}:
        raise ValueError("role-neutral BoW seal requested another family")
    payload = copy.deepcopy(dict(evidence_payload))
    payload_sha256 = _sha256_json(payload)
    owner = request.physical_owner
    events = [
        {
            "sequence": 1,
            "event": "fit_completed",
            "fit_state_artifact_sha256": fit_state_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
        },
        {
            "sequence": 2,
            "event": "fit_family_artifact_sealed",
            "fit_state_artifact_sha256": fit_state_sha256,
            "evidence_payload_sha256": payload_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
        },
    ]
    body = {
        "schema_version": LEGACY_STAGE1_FIT_ONLY_FAMILY_SEAL_SCHEMA,
        "plan_scientific_content_sha256": (request.plan_scientific_content_sha256),
        "physical_owner_scope_id": owner.scope_id,
        "physical_owner_scope_sha256": owner.as_dict()["scope_sha256"],
        "family": family,
        "fit_row_ids": list(owner.fit_row_ids),
        "fit_row_order_fingerprint": _row_order_fingerprint(owner.fit_row_ids),
        "canonical_group_seed": int(owner.scope_seed),
        "producer_identity_sha256": _producer_identity(),
        "configuration_identity_sha256": configuration_identity_sha256,
        "fit_state_artifact_sha256": fit_state_sha256,
        "evidence_payload_sha256": payload_sha256,
        "evidence_payload": payload,
        "event_order": events,
        "logical_view_transform_started": False,
        "registered_heldout_text_accessed": False,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _predict_live_and_replay(
    *,
    live: Sequence[_LiveFold],
    texts: tuple[str, ...],
    store: _ArrayStore,
    prediction_clip: float | None,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for fold in live:
        if fold.constant_prediction is not None:
            live_prediction = np.full(
                len(texts),
                fold.constant_prediction,
                dtype=np.float64,
            )
            replay_matrix = sparse.csr_matrix((len(texts), 0), dtype=np.float32)
        else:
            if fold.vectorizer is None or fold.learner is None:
                raise RuntimeError("live BoW fold lost fitted state before transform")
            live_matrix = fold.vectorizer.transform(texts)
            live_prediction = (
                fold.learner.predict_proba(live_matrix)[:, 1]
                if fold.classification
                else fold.learner.predict(live_matrix)
            )
            restored = _restore_vectorizer(
                fold.captured_vectorizer or {},
                store.arrays,
            )
            replay_matrix = restored.transform(texts)
        replay_prediction = _predict_learner(
            fold.captured_learner,
            store.arrays,
            replay_matrix,
        )
        live_prediction = np.asarray(live_prediction, dtype=np.float64)
        replay_prediction = np.asarray(replay_prediction, dtype=np.float64)
        if prediction_clip is not None:
            live_prediction = np.clip(
                live_prediction,
                prediction_clip,
                1.0 - prediction_clip,
            )
            replay_prediction = np.clip(
                replay_prediction,
                prediction_clip,
                1.0 - prediction_clip,
            )
        if not np.isfinite(live_prediction).all() or not np.isfinite(replay_prediction).all():
            raise RuntimeError("BoW logical transform emitted non-finite values")
        if not np.allclose(
            live_prediction,
            replay_prediction,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError("live BoW transform differs from sealed safe state")
        predictions.append(live_prediction)
    if not predictions:
        raise RuntimeError("BoW logical transform has no fitted folds")
    return np.mean(np.vstack(predictions), axis=0)


def execute_role_neutral_bow_physical_group(
    *,
    request: RoleNeutralBoWPhysicalGroupRequest,
    output_root: Path | str,
    fit_texts: Sequence[str],
    fit_treatment: Sequence[Any],
    fit_outcome: Sequence[Any],
    view_configs: Sequence[BoWViewConfig],
    nuisance_folds: int,
    effect_folds: int,
    e_clip: float,
    bow_fold_parallelism: int | str,
    bow_parallel_backend: str,
    owner_cpu_budget: int,
    exact_heldout_text_loader: Callable[[tuple[int, ...]], Sequence[str]],
) -> Mapping[str, Any]:
    """Fit once, seal, then produce exact and cumulative logical views.

    ``exact_heldout_text_loader`` is deliberately invoked only after the
    fit-state tree and compatible fit-only family seal have been written and
    freshly validated.  The live fitted vectorizer/learner objects remain in
    this call's worker memory and are used for the exact transform.
    """

    if not isinstance(request, RoleNeutralBoWPhysicalGroupRequest):
        raise TypeError("role-neutral BoW execution requires its typed request")
    request.as_dict()
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("role-neutral BoW output root must be absolute")
    if root.exists() or root.is_symlink():
        raise FileExistsError("role-neutral BoW output root must be fresh")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    owner = request.physical_owner
    texts = tuple(fit_texts)
    if len(texts) != len(owner.fit_row_ids) or any(not isinstance(text, str) for text in texts):
        raise ValueError("fit texts must align exactly to physical fit rows")
    treatment = _binary_vector(
        fit_treatment,
        label="fit treatment",
        length=len(texts),
    )
    outcome = _binary_vector(
        fit_outcome,
        label="fit outcome",
        length=len(texts),
    )
    views = tuple(view_configs)
    if (
        not views
        or any(type(view) is not BoWViewConfig for view in views)
        or len({view.name for view in views}) != len(views)
    ):
        raise ValueError("role-neutral BoW execution requires unique typed views")
    nuisance_fold_count = int(nuisance_folds)
    if nuisance_fold_count < 2:
        raise ValueError("configured nuisance_folds must be at least two")
    effect_fold_count = int(effect_folds)
    if effect_fold_count < 2:
        raise ValueError("configured effect_folds must be at least two")
    clip = float(e_clip)
    if not 0.0 < clip < 0.5:
        raise ValueError("configured e_clip must be in (0, 0.5)")
    owner_budget = int(owner_cpu_budget)
    if isinstance(owner_cpu_budget, bool) or owner_budget < 1:
        raise ValueError("BoW owner CPU budget must be a positive integer")
    _parallel_backend_name(bow_parallel_backend)
    _resolve_fold_parallelism(
        setting=bow_fold_parallelism,
        task_count=max(nuisance_fold_count, effect_fold_count),
        owner_cpu_budget=owner_budget,
    )
    if not callable(exact_heldout_text_loader):
        raise TypeError("exact held-out text loader must be callable")

    store = _ArrayStore()
    records: list[dict[str, Any]] = []
    live_by_objective_view: dict[str, list[_LiveFold]] = {}
    oof_by_objective_view: dict[str, np.ndarray] = {}
    for view_index, view in enumerate(views):
        for objective, labels in (
            ("treatment_nuisance", treatment),
            ("outcome_nuisance", outcome),
        ):
            live, objective_records, oof = _fit_binary_objective(
                request=request,
                fit_texts=texts,
                labels=labels,
                objective=objective,
                view=view,
                view_index=view_index,
                nuisance_folds=nuisance_fold_count,
                e_clip=clip,
                store=store,
                bow_fold_parallelism=bow_fold_parallelism,
                bow_parallel_backend=bow_parallel_backend,
                owner_cpu_budget=owner_budget,
            )
            key = f"{view.name}::{objective}"
            live_by_objective_view[key] = live
            oof_by_objective_view[key] = oof
            records.extend(objective_records)

    treatment_oof = [
        values
        for name, values in oof_by_objective_view.items()
        if name.endswith("::treatment_nuisance")
    ]
    outcome_oof = [
        values
        for name, values in oof_by_objective_view.items()
        if name.endswith("::outcome_nuisance")
    ]
    if len(treatment_oof) != len(views) or len(outcome_oof) != len(views):
        raise RuntimeError("role-neutral BoW nuisance OOF coverage changed")
    e_hat = np.mean(np.vstack(treatment_oof), axis=0)
    m_hat = np.mean(np.vstack(outcome_oof), axis=0)
    clipped_e_hat = np.clip(e_hat, clip, 1.0 - clip)
    t_residual = treatment.astype(np.float64) - clipped_e_hat
    y_residual = outcome.astype(np.float64) - m_hat
    pseudo_target = y_residual / t_residual
    r_weight = np.square(t_residual)
    derived_fit_quantities = {
        "fit_treatment": treatment.astype(np.float64),
        "fit_outcome": outcome.astype(np.float64),
        "ensemble_e_hat": e_hat,
        "ensemble_m_hat": m_hat,
        "clipped_e_hat": clipped_e_hat,
        "t_residual": t_residual,
        "y_residual": y_residual,
        "pseudo_target": pseudo_target,
        "r_weight": r_weight,
    }
    if any(
        value.shape != treatment.shape or not np.isfinite(value).all()
        for value in derived_fit_quantities.values()
    ):
        raise RuntimeError("role-neutral BoW residual quantities are invalid")
    for view_index, view in enumerate(views):
        for objective, weights in (
            ("effect_pseudo_target", None),
            ("effect_weighted_r", r_weight),
        ):
            live, objective_records, oof = _fit_regression_objective(
                request=request,
                fit_texts=texts,
                target=pseudo_target,
                sample_weight=weights,
                objective=objective,
                view=view,
                view_index=view_index,
                effect_folds=effect_fold_count,
                store=store,
                bow_fold_parallelism=bow_fold_parallelism,
                bow_parallel_backend=bow_parallel_backend,
                owner_cpu_budget=owner_budget,
            )
            key = f"{view.name}::{objective}"
            live_by_objective_view[key] = live
            oof_by_objective_view[key] = oof
            records.extend(objective_records)

    view_payloads = tuple(_bow_view_to_dict(view) for view in views)
    evidence_payload_by_family = {
        family: _evidence_payload(
            records=records,
            store=store,
            family=family,
        )
        for family in (BOW_NUISANCE, BOW_R_LOSS)
    }
    _metadata, fit_state_sha256, configuration_identity = _write_fit_state(
        root=root,
        request=request,
        fit_texts=texts,
        treatment=treatment,
        outcome=outcome,
        view_configs=view_payloads,
        nuisance_folds=nuisance_fold_count,
        effect_folds=effect_fold_count,
        e_clip=clip,
        records=records,
        oof_by_objective_view=oof_by_objective_view,
        derived_fit_quantities=derived_fit_quantities,
        store=store,
    )
    seals = {
        family: _fit_seal(
            request=request,
            family=family,
            evidence_payload=evidence_payload_by_family[family],
            fit_state_sha256=fit_state_sha256,
            configuration_identity_sha256=configuration_identity,
        )
        for family in (BOW_NUISANCE, BOW_R_LOSS)
    }
    seal_registrations: dict[str, dict[str, Any]] = {}
    for family, seal in seals.items():
        seal_path = root / _FIT_SEAL_FILES[family]
        _write_new_json(seal_path, seal)
        seal_sha256, seal_size = _sha256_file(seal_path)
        seal_registrations[family] = {
            "relative_path": _FIT_SEAL_FILES[family],
            "sha256": seal_sha256,
            "size_bytes": seal_size,
            "content_sha256": seal["content_sha256"],
        }
    # This reopens every fit-state byte before any registered held-out loader
    # can run.  A self-attested event record alone is not accepted.
    _validate_fit_side(
        root=root,
        request=request,
        expected_fit_texts=texts,
        expected_treatment=treatment,
        expected_outcome=outcome,
    )

    logical_root = root / _LOGICAL_VIEW_DIRECTORY
    logical_root.mkdir(parents=True, exist_ok=False)
    family_order = (BOW_NUISANCE, BOW_R_LOSS)
    events: list[dict[str, Any]] = [
        {
            "sequence": 1,
            "event": "fit_completed",
            "families": list(family_order),
            "fit_state_artifact_sha256": fit_state_sha256,
            "registered_heldout_text_accessed": False,
            "registered_heldout_labels_accessed": False,
        }
    ]
    for family in family_order:
        events.append(
            {
                "sequence": len(events) + 1,
                "event": "fit_family_artifact_sealed",
                "family": family,
                "fit_state_artifact_sha256": fit_state_sha256,
                "fit_only_family_seal_sha256": seal_registrations[family]["sha256"],
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
            }
        )
    logical_registrations: list[dict[str, Any]] = []
    for member in request.logical_members:
        if member.scope_id == owner.scope_id:
            continue
        if member.scope_kind != "cumulative_spent":
            raise RuntimeError("role-neutral BoW alias changed purpose")
        for family in family_order:
            seal = seals[family]
            seal_registration = seal_registrations[family]
            body = {
                "schema_version": ROLE_NEUTRAL_BOW_LOGICAL_VIEW_SCHEMA,
                "group_request_content_sha256": request.content_sha256,
                "logical_scope_id": member.scope_id,
                "logical_scope_sha256": member.as_dict()["scope_sha256"],
                "logical_purpose": member.scope_kind,
                "physical_owner_scope_id": owner.scope_id,
                "family": family,
                "fit_only_family_seal_sha256": seal_registration["sha256"],
                "fit_only_family_seal_content_sha256": seal["content_sha256"],
                "view_input_policy": ("sealed_row_ids_only_no_sealed_text_or_labels_v1"),
                "logical_heldout_row_ids": list(member.heldout_row_ids),
                "logical_transform_performed": False,
                "prediction_artifact": None,
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
                "reuses_live_physical_fit": True,
            }
            view = {**body, "content_sha256": _sha256_json(body)}
            suffix = "" if family == BOW_NUISANCE else f".{family}"
            path = logical_root / f"{member.scope_id}{suffix}.json"
            _write_new_json(path, view)
            digest, size = _sha256_file(path)
            logical_registrations.append(
                {
                    "logical_scope_id": member.scope_id,
                    "family": family,
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": digest,
                    "size_bytes": size,
                    "content_sha256": view["content_sha256"],
                }
            )
            events.append(
                {
                    "sequence": len(events) + 1,
                    "event": "cumulative_fit_only_view_published",
                    "logical_scope_id": member.scope_id,
                    "family": family,
                    "registered_heldout_text_accessed": False,
                    "registered_heldout_labels_accessed": False,
                }
            )

    # The only operation capable of materializing registered held-out text is
    # intentionally below the validated fit-only publication boundary.
    loaded = exact_heldout_text_loader(tuple(owner.heldout_row_ids))
    heldout_texts = tuple(loaded)
    if len(heldout_texts) != len(owner.heldout_row_ids) or any(
        not isinstance(text, str) for text in heldout_texts
    ):
        raise ValueError("exact held-out loader returned another row/text shape")
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "exact_heldout_text_opened",
            "logical_scope_id": owner.scope_id,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
        }
    )
    for family in family_order:
        selected = [
            (key, live)
            for key, live in live_by_objective_view.items()
            if live and live[0].family == family
        ]
        columns = [key for key, _live in selected]
        prediction_matrix = np.column_stack(
            [
                _predict_live_and_replay(
                    live=live,
                    texts=heldout_texts,
                    store=store,
                    prediction_clip=(clip if family == BOW_NUISANCE else None),
                )
                for _key, live in selected
            ]
        ).astype(np.float64, copy=False)
        suffix = "" if family == BOW_NUISANCE else f".{family}"
        prediction_path = logical_root / f"{owner.scope_id}{suffix}.predictions.npy"
        _write_new_npy(prediction_path, prediction_matrix)
        prediction_sha256, prediction_size = _sha256_file(prediction_path)
        prediction_registration = {
            "relative_path": prediction_path.relative_to(root).as_posix(),
            "sha256": prediction_sha256,
            "size_bytes": prediction_size,
            "dtype": prediction_matrix.dtype.str,
            "shape": list(prediction_matrix.shape),
            "columns": columns,
        }
        events.append(
            {
                "sequence": len(events) + 1,
                "event": "exact_heldout_transform_completed",
                "logical_scope_id": owner.scope_id,
                "family": family,
                "registered_heldout_text_accessed": True,
                "registered_heldout_labels_accessed": False,
            }
        )
        seal = seals[family]
        exact_body = {
            "schema_version": ROLE_NEUTRAL_BOW_LOGICAL_VIEW_SCHEMA,
            "group_request_content_sha256": request.content_sha256,
            "logical_scope_id": owner.scope_id,
            "logical_scope_sha256": owner.as_dict()["scope_sha256"],
            "logical_purpose": owner.scope_kind,
            "physical_owner_scope_id": owner.scope_id,
            "family": family,
            "fit_only_family_seal_sha256": seal_registrations[family]["sha256"],
            "fit_only_family_seal_content_sha256": seal["content_sha256"],
            "view_input_policy": "heldout_row_id_and_text_no_labels_v1",
            "logical_heldout_row_ids": list(owner.heldout_row_ids),
            "logical_heldout_text_sha256": _text_sha256(
                owner.heldout_row_ids,
                heldout_texts,
            ),
            "logical_transform_performed": True,
            "prediction_artifact": prediction_registration,
            "registered_heldout_text_accessed": True,
            "registered_heldout_labels_accessed": False,
            "reuses_live_physical_fit": True,
            "model_state_reloaded_for_primary_transform": False,
            "sealed_state_replay_checked": True,
        }
        exact_view = {
            **exact_body,
            "content_sha256": _sha256_json(exact_body),
        }
        exact_path = logical_root / f"{owner.scope_id}{suffix}.json"
        _write_new_json(exact_path, exact_view)
        exact_digest, exact_size = _sha256_file(exact_path)
        logical_registrations.append(
            {
                "logical_scope_id": owner.scope_id,
                "family": family,
                "relative_path": exact_path.relative_to(root).as_posix(),
                "sha256": exact_digest,
                "size_bytes": exact_size,
                "content_sha256": exact_view["content_sha256"],
            }
        )
        events.append(
            {
                "sequence": len(events) + 1,
                "event": "exact_logical_view_published",
                "logical_scope_id": owner.scope_id,
                "family": family,
                "registered_heldout_text_accessed": True,
                "registered_heldout_labels_accessed": False,
            }
        )
    logical_registrations.sort(
        key=lambda row: (
            next(
                index
                for index, member in enumerate(request.logical_members)
                if member.scope_id == row["logical_scope_id"]
            ),
            family_order.index(row["family"]),
        )
    )
    fit_projection_proof = build_role_neutral_stage2_fit_projection_proof(
        plan_scientific_content_sha256=(request.plan_scientific_content_sha256),
        physical_owner_scope_id=owner.scope_id,
        fit_row_ids=owner.fit_row_ids,
        fit_texts=texts,
        fit_treatment=treatment,
        fit_outcome=outcome,
    )
    terminal_body = {
        "schema_version": ROLE_NEUTRAL_BOW_GROUP_EXECUTION_SCHEMA,
        "status": "complete",
        "group_request": request.as_dict(),
        "families": list(family_order),
        "fit_state_artifact_sha256": fit_state_sha256,
        "fit_only_family_seals": seal_registrations,
        "logical_views": logical_registrations,
        ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD: (fit_projection_proof),
        "event_order": events,
        "fit_completed_before_registered_heldout_text_access": True,
        "fit_sealed_before_registered_heldout_text_access": True,
        "cumulative_views_published_without_sealed_text": True,
        "live_model_objects_reused_for_exact_transform": True,
        "model_state_reloaded_for_primary_transform": False,
        "registered_heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "text_truncation_applied": False,
    }
    terminal = {
        **terminal_body,
        "content_sha256": _sha256_json(terminal_body),
    }
    _write_new_json(root / _TERMINAL_FILE, terminal)
    return validate_role_neutral_bow_group_execution(
        root=root,
        request=request,
    )


def _validate_fit_side(
    *,
    root: Path,
    request: RoleNeutralBoWPhysicalGroupRequest,
    expected_fit_texts: Sequence[str] | None = None,
    expected_treatment: np.ndarray | None = None,
    expected_outcome: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    fit_root = root / _FIT_STATE_DIRECTORY
    metadata = _read_json(
        fit_root / _FIT_STATE_METADATA,
        label="role-neutral BoW fit metadata",
    )
    body = {key: copy.deepcopy(value) for key, value in metadata.items() if key != "content_sha256"}
    required = {
        "schema_version",
        "group_request_content_sha256",
        "plan_scientific_content_sha256",
        "physical_owner_scope_id",
        "physical_owner_scope_sha256",
        "fit_row_ids",
        "fit_row_order_fingerprint",
        "canonical_group_seed",
        "fit_text_sha256",
        "fit_treatment_sha256",
        "fit_outcome_sha256",
        "configuration",
        "configuration_identity_sha256",
        "producer_identity_sha256",
        "fold_records",
        "oof_predictions",
        "derived_fit_quantities",
        "array_inventory",
        "array_layout",
        "model_objects_retained_in_worker_memory",
        "registered_heldout_text_accessed",
        "registered_heldout_labels_accessed",
        "oracle_fields_accessed",
        "text_truncation_applied",
        "content_sha256",
    }
    inventory = metadata.get("array_inventory")
    configuration = metadata.get("configuration")
    expected_configuration_fields = {
        "view_configs",
        "nuisance_folds",
        "effect_folds",
        "e_clip",
        "outcome_type",
        "nuisance_source_policy",
        "pseudo_target_formula",
        "r_weight_formula",
        "residual_effect_objectives",
        "text_truncation_applied",
    }
    if (
        set(metadata) != required
        or metadata.get("schema_version") != ROLE_NEUTRAL_BOW_FIT_STATE_SCHEMA
        or metadata.get("group_request_content_sha256") != request.content_sha256
        or metadata.get("plan_scientific_content_sha256") != request.plan_scientific_content_sha256
        or metadata.get("physical_owner_scope_id") != request.physical_owner.scope_id
        or metadata.get("physical_owner_scope_sha256")
        != request.physical_owner.as_dict()["scope_sha256"]
        or metadata.get("fit_row_ids") != list(request.physical_owner.fit_row_ids)
        or metadata.get("fit_row_order_fingerprint")
        != _row_order_fingerprint(request.physical_owner.fit_row_ids)
        or metadata.get("canonical_group_seed") != request.physical_owner.scope_seed
        or not isinstance(configuration, Mapping)
        or set(configuration) != expected_configuration_fields
        or configuration.get("outcome_type") != "binary"
        or configuration.get("nuisance_source_policy") != "mean_of_fit_row_oof_bow_views_v1"
        or configuration.get("pseudo_target_formula") != "(outcome-m_hat)/(treatment-clipped_e_hat)"
        or configuration.get("r_weight_formula") != "(treatment-clipped_e_hat)^2"
        or configuration.get("residual_effect_objectives")
        != ["effect_pseudo_target", "effect_weighted_r"]
        or configuration.get("text_truncation_applied") is not False
        or metadata.get("configuration_identity_sha256") != _sha256_json(configuration)
        or metadata.get("producer_identity_sha256") != _producer_identity()
        or metadata.get("array_layout") != "one_npy_per_array_mmap_safe_v1"
        or metadata.get("model_objects_retained_in_worker_memory") is not True
        or metadata.get("registered_heldout_text_accessed") is not False
        or metadata.get("registered_heldout_labels_accessed") is not False
        or metadata.get("oracle_fields_accessed") is not False
        or metadata.get("text_truncation_applied") is not False
        or not isinstance(inventory, Mapping)
        or not inventory
        or metadata.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("role-neutral BoW fit metadata is invalid")
    view_configs = configuration.get("view_configs")
    if (
        not isinstance(view_configs, list)
        or not view_configs
        or any(not isinstance(view, Mapping) for view in view_configs)
        or any(not isinstance(view.get("name"), str) or not view["name"] for view in view_configs)
        or len({str(view["name"]) for view in view_configs}) != len(view_configs)
        or type(configuration.get("nuisance_folds")) is not int
        or int(configuration["nuisance_folds"]) < 2
        or type(configuration.get("effect_folds")) is not int
        or int(configuration["effect_folds"]) < 2
    ):
        raise ValueError("role-neutral BoW fit configuration is incomplete")
    clip = float(configuration.get("e_clip"))
    if not 0.0 < clip < 0.5:
        raise ValueError("role-neutral BoW fit clipping configuration is invalid")

    observed_arrays = {
        path.stem: path
        for path in (fit_root / "arrays").glob("*.npy")
        if path.is_file() and not path.is_symlink()
    }
    if set(observed_arrays) != set(inventory):
        raise ValueError("role-neutral BoW fit array coverage changed")
    arrays: dict[str, np.ndarray] = {}
    for key, registration in inventory.items():
        if (
            not isinstance(key, str)
            or not isinstance(registration, Mapping)
            or set(registration)
            != {
                "relative_path",
                "dtype",
                "shape",
                "content_sha256",
                "file_sha256",
                "size_bytes",
            }
            or registration.get("relative_path") != f"arrays/{key}.npy"
        ):
            raise ValueError("role-neutral BoW array registration changed")
        path = fit_root / str(registration["relative_path"])
        if path != observed_arrays[key] or path.parent != fit_root / "arrays":
            raise ValueError("role-neutral BoW array path changed")
        digest, size = _sha256_file(path)
        with path.open("rb") as handle:
            array = np.load(handle, allow_pickle=False)
        if (
            array.dtype.hasobject
            or registration.get("dtype") != array.dtype.str
            or registration.get("shape") != list(array.shape)
            or registration.get("content_sha256") != _array_sha256(array)
            or registration.get("file_sha256") != digest
            or registration.get("size_bytes") != size
        ):
            raise ValueError(f"role-neutral BoW fit array changed: {key}")
        arrays[key] = np.asarray(array)

    if expected_fit_texts is not None and metadata.get("fit_text_sha256") != _text_sha256(
        request.physical_owner.fit_row_ids, expected_fit_texts
    ):
        raise ValueError("role-neutral BoW fit text binding changed")
    if expected_treatment is not None and metadata.get("fit_treatment_sha256") != _float_hex_sha256(
        expected_treatment
    ):
        raise ValueError("role-neutral BoW treatment binding changed")
    if expected_outcome is not None and metadata.get("fit_outcome_sha256") != _float_hex_sha256(
        expected_outcome
    ):
        raise ValueError("role-neutral BoW outcome binding changed")

    oof_references = metadata.get("oof_predictions")
    derived_references = metadata.get("derived_fit_quantities")
    view_names = [str(view["name"]) for view in view_configs]
    objectives_by_family = {
        BOW_NUISANCE: ("treatment_nuisance", "outcome_nuisance"),
        BOW_R_LOSS: ("effect_pseudo_target", "effect_weighted_r"),
    }
    expected_oof_names = {
        f"{view_name}::{objective}"
        for view_name in view_names
        for objectives in objectives_by_family.values()
        for objective in objectives
    }
    expected_derived_names = {
        "fit_treatment",
        "fit_outcome",
        "ensemble_e_hat",
        "ensemble_m_hat",
        "clipped_e_hat",
        "t_residual",
        "y_residual",
        "pseudo_target",
        "r_weight",
    }
    if (
        not isinstance(oof_references, Mapping)
        or set(oof_references) != expected_oof_names
        or any(
            not isinstance(reference, str) or reference not in arrays
            for reference in oof_references.values()
        )
        or len(set(oof_references.values())) != len(oof_references)
        or not isinstance(derived_references, Mapping)
        or set(derived_references) != expected_derived_names
        or any(
            not isinstance(reference, str) or reference not in arrays
            for reference in derived_references.values()
        )
        or len(set(derived_references.values())) != len(derived_references)
    ):
        raise ValueError("role-neutral BoW fit proof references are incomplete")

    def proof_array(reference: str, *, label: str) -> np.ndarray:
        value = np.asarray(arrays[reference], dtype=np.float64)
        if value.shape != (len(request.physical_owner.fit_row_ids),):
            raise ValueError(f"{label} is not aligned to physical fit rows")
        if not np.isfinite(value).all():
            raise ValueError(f"{label} contains non-finite values")
        return value

    derived = {
        name: proof_array(str(derived_references[name]), label=name)
        for name in sorted(expected_derived_names)
    }
    treatment = derived["fit_treatment"]
    outcome = derived["fit_outcome"]
    if (
        not set(np.unique(treatment)).issubset({0.0, 1.0})
        or not set(np.unique(outcome)).issubset({0.0, 1.0})
        or metadata.get("fit_treatment_sha256") != _float_hex_sha256(treatment)
        or metadata.get("fit_outcome_sha256") != _float_hex_sha256(outcome)
    ):
        raise ValueError("role-neutral BoW persisted fit labels are invalid")
    if expected_treatment is not None and not np.array_equal(
        treatment,
        np.asarray(expected_treatment, dtype=np.float64),
    ):
        raise ValueError("persisted treatment differs from supplied fit rows")
    if expected_outcome is not None and not np.array_equal(
        outcome,
        np.asarray(expected_outcome, dtype=np.float64),
    ):
        raise ValueError("persisted outcome differs from supplied fit rows")

    treatment_oof = [
        proof_array(
            str(oof_references[f"{view_name}::treatment_nuisance"]),
            label=f"{view_name} treatment OOF",
        )
        for view_name in view_names
    ]
    outcome_oof = [
        proof_array(
            str(oof_references[f"{view_name}::outcome_nuisance"]),
            label=f"{view_name} outcome OOF",
        )
        for view_name in view_names
    ]
    expected_derived = {
        "fit_treatment": treatment,
        "fit_outcome": outcome,
        "ensemble_e_hat": np.mean(np.vstack(treatment_oof), axis=0),
        "ensemble_m_hat": np.mean(np.vstack(outcome_oof), axis=0),
    }
    expected_derived["clipped_e_hat"] = np.clip(
        expected_derived["ensemble_e_hat"],
        clip,
        1.0 - clip,
    )
    expected_derived["t_residual"] = treatment - expected_derived["clipped_e_hat"]
    expected_derived["y_residual"] = outcome - expected_derived["ensemble_m_hat"]
    expected_derived["pseudo_target"] = (
        expected_derived["y_residual"] / expected_derived["t_residual"]
    )
    expected_derived["r_weight"] = np.square(expected_derived["t_residual"])
    for name, expected in expected_derived.items():
        if not np.allclose(
            derived[name],
            expected,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(f"role-neutral BoW derived fit quantity changed: {name}")

    records = metadata.get("fold_records")
    if not isinstance(records, list) or not records:
        raise ValueError("role-neutral BoW fold proof is empty")
    config_by_view = {str(view["name"]): copy.deepcopy(dict(view)) for view in view_configs}
    fit_row_ids = tuple(request.physical_owner.fit_row_ids)
    fit_row_set = set(fit_row_ids)
    row_position = {row_id: index for index, row_id in enumerate(fit_row_ids)}
    grouped_records: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    common_record_fields = {
        "family",
        "objective",
        "view_name",
        "view_config",
        "fold",
        "seed",
        "fit_row_ids",
        "validation_row_ids",
        "fit_target",
        "validation_target",
        "validation_prediction",
        "vectorizer",
        "learner",
        "registered_heldout_text_accessed",
        "registered_heldout_labels_accessed",
    }
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("role-neutral BoW fold proof row is malformed")
        family = str(record.get("family"))
        objective = str(record.get("objective"))
        view_name = str(record.get("view_name"))
        expected_fields = (
            {*common_record_fields, "fit_sample_weight"}
            if family == BOW_R_LOSS
            else common_record_fields
        )
        if (
            family not in objectives_by_family
            or objective not in objectives_by_family[family]
            or view_name not in config_by_view
            or set(record) != expected_fields
            or record.get("view_config") != config_by_view[view_name]
            or type(record.get("fold")) is not int
            or int(record["fold"]) < 1
            or type(record.get("seed")) is not int
            or record.get("registered_heldout_text_accessed") is not False
            or record.get("registered_heldout_labels_accessed") is not False
            or not isinstance(record.get("learner"), Mapping)
        ):
            raise ValueError("role-neutral BoW fold proof semantics changed")
        fit_ids = record.get("fit_row_ids")
        validation_ids = record.get("validation_row_ids")
        if (
            not isinstance(fit_ids, list)
            or not isinstance(validation_ids, list)
            or not fit_ids
            or not validation_ids
            or len(fit_ids) != len(set(fit_ids))
            or len(validation_ids) != len(set(validation_ids))
            or set(fit_ids).intersection(validation_ids)
            or set(fit_ids).union(validation_ids) != fit_row_set
        ):
            raise ValueError("role-neutral BoW fold rows are not a fit-only partition")
        fit_positions = np.asarray([row_position[int(row_id)] for row_id in fit_ids])
        validation_positions = np.asarray([row_position[int(row_id)] for row_id in validation_ids])
        target = (
            treatment
            if objective == "treatment_nuisance"
            else outcome if objective == "outcome_nuisance" else derived["pseudo_target"]
        )
        for field, positions in (
            ("fit_target", fit_positions),
            ("validation_target", validation_positions),
        ):
            reference = record.get(field)
            if (
                not isinstance(reference, str)
                or reference not in arrays
                or not np.allclose(
                    np.asarray(arrays[reference], dtype=np.float64),
                    target[positions],
                    rtol=1e-12,
                    atol=1e-12,
                )
            ):
                raise ValueError(f"role-neutral BoW {objective} {field} changed")
        prediction_reference = record.get("validation_prediction")
        if (
            not isinstance(prediction_reference, str)
            or prediction_reference not in arrays
            or np.asarray(arrays[prediction_reference]).shape != (len(validation_positions),)
            or not np.isfinite(arrays[prediction_reference]).all()
        ):
            raise ValueError("role-neutral BoW validation prediction changed")
        vectorizer_state = record.get("vectorizer")
        learner_kind = str(record["learner"].get("kind") or "")
        if (
            (vectorizer_state is None) != learner_kind.startswith("constant_")
            or (vectorizer_state is not None and not isinstance(vectorizer_state, Mapping))
            or (
                family == BOW_R_LOSS
                and (vectorizer_state is None or learner_kind == "constant_regressor")
            )
        ):
            raise ValueError("role-neutral BoW fitted state kind changed")
        if family == BOW_R_LOSS:
            weight_reference = record.get("fit_sample_weight")
            if objective == "effect_pseudo_target":
                if weight_reference is not None:
                    raise ValueError("unweighted pseudo-target fit gained weights")
            elif (
                not isinstance(weight_reference, str)
                or weight_reference not in arrays
                or not np.allclose(
                    np.asarray(arrays[weight_reference], dtype=np.float64),
                    derived["r_weight"][fit_positions],
                    rtol=1e-12,
                    atol=1e-12,
                )
            ):
                raise ValueError("weighted R-loss fit weights changed")
        grouped_records.setdefault((family, view_name, objective), []).append(record)

    expected_group_keys = {
        (family, view_name, objective)
        for family, objectives in objectives_by_family.items()
        for view_name in view_names
        for objective in objectives
    }
    if set(grouped_records) != expected_group_keys:
        raise ValueError("role-neutral BoW fold objective coverage changed")
    for (family, view_name, objective), objective_records in grouped_records.items():
        ordered = sorted(objective_records, key=lambda row: int(row["fold"]))
        if [int(record["fold"]) for record in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError("role-neutral BoW fold numbering changed")
        expected_fold_count = (
            _bounded_fold_count(int(configuration["effect_folds"]), len(fit_row_ids))
            if family == BOW_R_LOSS
            else len(
                _binary_split_items(
                    treatment if objective == "treatment_nuisance" else outcome,
                    requested_folds=int(configuration["nuisance_folds"]),
                    random_state=_derived_seed(
                        request.physical_owner.scope_seed,
                        view_name=view_name,
                        objective=objective,
                        purpose="cross_fit_split",
                    ),
                )
            )
        )
        if len(ordered) != expected_fold_count:
            raise ValueError("role-neutral BoW configured fold count changed")
        seen_validation_rows: list[int] = []
        reconstructed_oof = np.full(len(fit_row_ids), np.nan, dtype=np.float64)
        for record in ordered:
            expected_seed = _derived_seed(
                request.physical_owner.scope_seed,
                view_name=view_name,
                objective=objective,
                purpose="fold_model",
                fold=int(record["fold"]),
            )
            if int(record["seed"]) != expected_seed:
                raise ValueError("role-neutral BoW fold seed changed")
            validation_ids = [int(row_id) for row_id in record["validation_row_ids"]]
            seen_validation_rows.extend(validation_ids)
            positions = np.asarray([row_position[row_id] for row_id in validation_ids])
            reconstructed_oof[positions] = np.asarray(
                arrays[str(record["validation_prediction"])],
                dtype=np.float64,
            )
        if (
            len(seen_validation_rows) != len(set(seen_validation_rows))
            or set(seen_validation_rows) != fit_row_set
            or not np.isfinite(reconstructed_oof).all()
            or not np.allclose(
                reconstructed_oof,
                proof_array(
                    str(oof_references[f"{view_name}::{objective}"]),
                    label=f"{view_name} {objective} OOF",
                ),
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise ValueError("role-neutral BoW OOF reconstruction changed")

    referenced_keys: set[str] = set()

    def collect_array_references(value: Any) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                collect_array_references(child)
        elif isinstance(value, list):
            for child in value:
                collect_array_references(child)
        elif isinstance(value, str) and value in arrays:
            referenced_keys.add(value)

    collect_array_references(records)
    collect_array_references(oof_references)
    collect_array_references(derived_references)
    if referenced_keys != set(arrays):
        raise ValueError("role-neutral BoW fit state contains unreferenced proof arrays")

    fit_state_sha256 = _tree_sha256(fit_root)
    expected_seal_fields = {
        "schema_version",
        "plan_scientific_content_sha256",
        "physical_owner_scope_id",
        "physical_owner_scope_sha256",
        "family",
        "fit_row_ids",
        "fit_row_order_fingerprint",
        "canonical_group_seed",
        "producer_identity_sha256",
        "configuration_identity_sha256",
        "fit_state_artifact_sha256",
        "evidence_payload_sha256",
        "evidence_payload",
        "event_order",
        "logical_view_transform_started",
        "registered_heldout_text_accessed",
        "registered_heldout_labels_accessed",
        "oracle_fields_accessed",
        "content_sha256",
    }
    evidence_store = _ArrayStore()
    evidence_store.arrays = arrays
    seals: dict[str, dict[str, Any]] = {}
    for family, filename in _FIT_SEAL_FILES.items():
        seal = _read_json(
            root / filename,
            label=f"role-neutral BoW {family} fit seal",
        )
        seal_body = {
            key: copy.deepcopy(value) for key, value in seal.items() if key != "content_sha256"
        }
        evidence_payload = seal.get("evidence_payload")
        expected_payload = _evidence_payload(
            records=records,
            store=evidence_store,
            family=family,
        )
        expected_events = [
            {
                "sequence": 1,
                "event": "fit_completed",
                "fit_state_artifact_sha256": fit_state_sha256,
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
                "oracle_fields_accessed": False,
            },
            {
                "sequence": 2,
                "event": "fit_family_artifact_sealed",
                "fit_state_artifact_sha256": fit_state_sha256,
                "evidence_payload_sha256": _sha256_json(expected_payload),
                "registered_heldout_text_accessed": False,
                "registered_heldout_labels_accessed": False,
                "oracle_fields_accessed": False,
            },
        ]
        if (
            set(seal) != expected_seal_fields
            or seal.get("schema_version") != LEGACY_STAGE1_FIT_ONLY_FAMILY_SEAL_SCHEMA
            or seal.get("plan_scientific_content_sha256") != request.plan_scientific_content_sha256
            or seal.get("physical_owner_scope_id") != request.physical_owner.scope_id
            or seal.get("physical_owner_scope_sha256")
            != request.physical_owner.as_dict()["scope_sha256"]
            or seal.get("family") != family
            or seal.get("fit_row_ids") != list(request.physical_owner.fit_row_ids)
            or seal.get("fit_row_order_fingerprint")
            != _row_order_fingerprint(request.physical_owner.fit_row_ids)
            or seal.get("canonical_group_seed") != request.physical_owner.scope_seed
            or seal.get("fit_state_artifact_sha256") != fit_state_sha256
            or seal.get("configuration_identity_sha256")
            != metadata["configuration_identity_sha256"]
            or seal.get("producer_identity_sha256") != metadata["producer_identity_sha256"]
            or evidence_payload != expected_payload
            or seal.get("evidence_payload_sha256") != _sha256_json(expected_payload)
            or seal.get("logical_view_transform_started") is not False
            or seal.get("registered_heldout_text_accessed") is not False
            or seal.get("registered_heldout_labels_accessed") is not False
            or seal.get("oracle_fields_accessed") is not False
            or seal.get("event_order") != expected_events
            or seal.get("content_sha256") != _sha256_json(seal_body)
        ):
            raise ValueError(f"role-neutral BoW {family} fit-only seal is invalid")
        seals[family] = seal
    return metadata, seals


def replay_role_neutral_bow_exact_transform(
    *,
    root: Path | str,
    request: RoleNeutralBoWPhysicalGroupRequest,
    exact_heldout_texts: Sequence[str],
) -> Mapping[str, Any]:
    """Replay an exact transform from JSON and non-object ``.npy`` state.

    This path deliberately has no access to the live sklearn objects retained
    by :func:`execute_role_neutral_bow_physical_group`.  It reopens and
    authenticates the fit-only tree, reconstructs each supported vectorizer
    and learner from closed numerical state, and never reads pickle/joblib.
    """

    source = Path(root)
    if source.is_symlink():
        raise ValueError("role-neutral BoW replay root cannot be a symbolic link")
    tree = source.resolve(strict=True)
    metadata, _seals = _validate_fit_side(root=tree, request=request)
    texts = tuple(exact_heldout_texts)
    if len(texts) != len(request.physical_owner.heldout_row_ids) or any(
        not isinstance(text, str) for text in texts
    ):
        raise ValueError("replay held-out texts differ from the exact logical rows")
    inventory = metadata["array_inventory"]
    arrays: dict[str, np.ndarray] = {}
    for key, registration in inventory.items():
        path = tree / _FIT_STATE_DIRECTORY / str(registration["relative_path"])
        with path.open("rb") as handle:
            arrays[str(key)] = np.asarray(np.load(handle, allow_pickle=False)).copy()
    grouped: dict[str, dict[str, list[np.ndarray]]] = {
        BOW_NUISANCE: {},
        BOW_R_LOSS: {},
    }
    for record in metadata["fold_records"]:
        if not isinstance(record, Mapping):
            raise ValueError("persisted BoW fold record is malformed")
        family = str(record.get("family"))
        if family not in grouped:
            raise ValueError("persisted BoW fold names another family")
        vectorizer_state = record.get("vectorizer")
        learner_state = record.get("learner")
        if not isinstance(learner_state, Mapping):
            raise ValueError("persisted BoW learner state is malformed")
        if vectorizer_state is None:
            matrix = sparse.csr_matrix((len(texts), 0), dtype=np.float32)
        else:
            if not isinstance(vectorizer_state, Mapping):
                raise ValueError("persisted BoW vectorizer state is malformed")
            vectorizer = _restore_vectorizer(vectorizer_state, arrays)
            matrix = vectorizer.transform(texts)
        prediction = _predict_learner(learner_state, arrays, matrix)
        key = f"{record['view_name']}::{record['objective']}"
        grouped[family].setdefault(key, []).append(np.asarray(prediction, dtype=np.float64))
    if any(not family_group for family_group in grouped.values()):
        raise RuntimeError("persisted BoW state has incomplete replayable families")
    clip = float(metadata["configuration"]["e_clip"])
    family_predictions: dict[str, dict[str, Any]] = {}
    for family in (BOW_NUISANCE, BOW_R_LOSS):
        columns = list(grouped[family])
        predictions = [np.mean(np.vstack(grouped[family][column]), axis=0) for column in columns]
        if family == BOW_NUISANCE:
            predictions = [np.clip(prediction, clip, 1.0 - clip) for prediction in predictions]
        matrix = np.column_stack(predictions).astype(np.float64, copy=False)
        if not np.isfinite(matrix).all():
            raise RuntimeError("persisted BoW replay emitted non-finite values")
        family_predictions[family] = {
            "columns": columns,
            "predictions": matrix,
        }
    return {
        # Keep the original nuisance aliases while the dual-family consumers
        # migrate to the explicit family mapping.
        "columns": family_predictions[BOW_NUISANCE]["columns"],
        "predictions": family_predictions[BOW_NUISANCE]["predictions"],
        "family_predictions": family_predictions,
        "fit_state_artifact_sha256": _tree_sha256(tree / _FIT_STATE_DIRECTORY),
        "state_source": "authenticated_json_and_npy_only",
        "live_model_objects_available": False,
        "pickle_or_joblib_loaded": False,
        "text_truncation_applied": False,
    }


def validate_role_neutral_bow_group_execution(
    *,
    root: Path | str,
    request: RoleNeutralBoWPhysicalGroupRequest,
) -> Mapping[str, Any]:
    """Fresh path-only validation of one completed two-phase BoW group."""

    source = Path(root)
    if source.is_symlink():
        raise ValueError("role-neutral BoW execution cannot be a symbolic link")
    tree = source.resolve(strict=True)
    if not tree.is_dir():
        raise ValueError("role-neutral BoW execution must be one real directory")
    metadata, seals = _validate_fit_side(root=tree, request=request)
    terminal = _read_json(
        tree / _TERMINAL_FILE,
        label="role-neutral BoW execution manifest",
    )
    body = {key: copy.deepcopy(value) for key, value in terminal.items() if key != "content_sha256"}
    logical_rows = terminal.get("logical_views")
    events = terminal.get("event_order")
    expected_terminal_fields = {
        "schema_version",
        "status",
        "group_request",
        "families",
        "fit_state_artifact_sha256",
        "fit_only_family_seals",
        "logical_views",
        ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD,
        "event_order",
        "fit_completed_before_registered_heldout_text_access",
        "fit_sealed_before_registered_heldout_text_access",
        "cumulative_views_published_without_sealed_text",
        "live_model_objects_reused_for_exact_transform",
        "model_state_reloaded_for_primary_transform",
        "registered_heldout_labels_accessed",
        "oracle_fields_accessed",
        "text_truncation_applied",
        "content_sha256",
    }
    family_order = (BOW_NUISANCE, BOW_R_LOSS)
    seal_registrations = terminal.get("fit_only_family_seals")
    fit_projection_proof = validate_role_neutral_stage2_fit_projection_proof(
        terminal.get(ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD),
        expected_plan_scientific_content_sha256=(request.plan_scientific_content_sha256),
        expected_physical_owner_scope_id=(request.physical_owner.scope_id),
        expected_fit_row_ids=request.physical_owner.fit_row_ids,
        expected_fit_text_sha256=metadata["fit_text_sha256"],
        expected_fit_treatment_sha256=metadata["fit_treatment_sha256"],
        expected_fit_outcome_sha256=metadata["fit_outcome_sha256"],
    )
    observed_seal_files: dict[str, tuple[str, int]] = {
        family: _sha256_file(tree / _FIT_SEAL_FILES[family]) for family in family_order
    }
    if (
        set(terminal) != expected_terminal_fields
        or terminal.get("schema_version") != ROLE_NEUTRAL_BOW_GROUP_EXECUTION_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("group_request") != request.as_dict()
        or terminal.get("families") != list(family_order)
        or terminal.get(ROLE_NEUTRAL_STAGE2_FIT_PROJECTION_TERMINAL_FIELD) != fit_projection_proof
        or terminal.get("fit_state_artifact_sha256")
        != seals[BOW_NUISANCE]["fit_state_artifact_sha256"]
        or any(
            seal["fit_state_artifact_sha256"] != terminal.get("fit_state_artifact_sha256")
            for seal in seals.values()
        )
        or not isinstance(seal_registrations, Mapping)
        or set(seal_registrations) != set(family_order)
        or not isinstance(logical_rows, list)
        or len(logical_rows) != len(request.logical_members) * len(family_order)
        or not isinstance(events, list)
        or [event.get("sequence") for event in events] != list(range(1, len(events) + 1))
        or [event.get("event") for event in events[:3]]
        != [
            "fit_completed",
            "fit_family_artifact_sealed",
            "fit_family_artifact_sealed",
        ]
        or [event.get("family") for event in events[1:3]] != list(family_order)
        or terminal.get("fit_completed_before_registered_heldout_text_access") is not True
        or terminal.get("fit_sealed_before_registered_heldout_text_access") is not True
        or terminal.get("cumulative_views_published_without_sealed_text") is not True
        or terminal.get("live_model_objects_reused_for_exact_transform") is not True
        or terminal.get("model_state_reloaded_for_primary_transform") is not False
        or terminal.get("registered_heldout_labels_accessed") is not False
        or terminal.get("oracle_fields_accessed") is not False
        or terminal.get("text_truncation_applied") is not False
        or terminal.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("role-neutral BoW execution manifest is invalid")
    expected_seal_registration_fields = {
        "relative_path",
        "sha256",
        "size_bytes",
        "content_sha256",
    }
    for family in family_order:
        registration = seal_registrations[family]
        digest, size = observed_seal_files[family]
        if (
            not isinstance(registration, Mapping)
            or set(registration) != expected_seal_registration_fields
            or registration.get("relative_path") != _FIT_SEAL_FILES[family]
            or registration.get("sha256") != digest
            or registration.get("size_bytes") != size
            or registration.get("content_sha256") != seals[family]["content_sha256"]
        ):
            raise ValueError(f"role-neutral BoW {family} seal registration changed")
    first_text_access = next(
        (
            index
            for index, event in enumerate(events)
            if event.get("registered_heldout_text_accessed") is True
        ),
        None,
    )
    expected_cumulative_events = (len(request.logical_members) - 1) * len(family_order)
    if (
        first_text_access is None
        or first_text_access != 3 + expected_cumulative_events
        or events[first_text_access].get("event") != "exact_heldout_text_opened"
    ):
        raise ValueError("held-out text access preceded sealed cumulative views")
    for event in events[:first_text_access]:
        if (
            event.get("registered_heldout_text_accessed") is not False
            or event.get("registered_heldout_labels_accessed") is not False
        ):
            raise ValueError("pre-transform event accessed held-out data")
    cumulative_events = events[3:first_text_access]
    expected_cumulative_keys = {
        (member.scope_id, family)
        for member in request.logical_members
        if member.scope_id != request.physical_owner.scope_id
        for family in family_order
    }
    if {
        (str(event.get("logical_scope_id")), str(event.get("family")))
        for event in cumulative_events
        if event.get("event") == "cumulative_fit_only_view_published"
    } != expected_cumulative_keys or len(cumulative_events) != len(expected_cumulative_keys):
        raise ValueError("cumulative views were not completely published before text access")
    registration_by_scope_family = {
        (str(row.get("logical_scope_id")), str(row.get("family"))): row
        for row in logical_rows
        if isinstance(row, Mapping)
    }
    expected_logical_keys = {
        (member.scope_id, family) for member in request.logical_members for family in family_order
    }
    if (
        len(registration_by_scope_family) != len(logical_rows)
        or set(registration_by_scope_family) != expected_logical_keys
    ):
        raise ValueError("role-neutral BoW logical view coverage changed")
    expected_registration_fields = {
        "logical_scope_id",
        "family",
        "relative_path",
        "sha256",
        "size_bytes",
        "content_sha256",
    }
    for member in request.logical_members:
        transformed = member.scope_id == request.physical_owner.scope_id
        for family in family_order:
            registration = registration_by_scope_family[(member.scope_id, family)]
            suffix = "" if family == BOW_NUISANCE else f".{family}"
            expected_relative_path = f"{_LOGICAL_VIEW_DIRECTORY}/{member.scope_id}{suffix}.json"
            if (
                not isinstance(registration, Mapping)
                or set(registration) != expected_registration_fields
                or registration.get("logical_scope_id") != member.scope_id
                or registration.get("family") != family
                or registration.get("relative_path") != expected_relative_path
            ):
                raise ValueError("role-neutral BoW logical registration is invalid")
            path = tree / str(registration.get("relative_path"))
            digest, size = _sha256_file(path)
            view = _read_json(
                path,
                label=f"{member.scope_id} {family} logical view",
            )
            view_body = {
                key: copy.deepcopy(value) for key, value in view.items() if key != "content_sha256"
            }
            common_view_fields = {
                "schema_version",
                "group_request_content_sha256",
                "logical_scope_id",
                "logical_scope_sha256",
                "logical_purpose",
                "physical_owner_scope_id",
                "family",
                "fit_only_family_seal_sha256",
                "fit_only_family_seal_content_sha256",
                "view_input_policy",
                "logical_heldout_row_ids",
                "logical_transform_performed",
                "prediction_artifact",
                "registered_heldout_text_accessed",
                "registered_heldout_labels_accessed",
                "reuses_live_physical_fit",
                "content_sha256",
            }
            expected_view_fields = (
                {
                    *common_view_fields,
                    "logical_heldout_text_sha256",
                    "model_state_reloaded_for_primary_transform",
                    "sealed_state_replay_checked",
                }
                if transformed
                else common_view_fields
            )
            seal_digest, _seal_size = observed_seal_files[family]
            if (
                set(view) != expected_view_fields
                or registration.get("sha256") != digest
                or registration.get("size_bytes") != size
                or registration.get("content_sha256") != view.get("content_sha256")
                or view.get("content_sha256") != _sha256_json(view_body)
                or view.get("schema_version") != ROLE_NEUTRAL_BOW_LOGICAL_VIEW_SCHEMA
                or view.get("group_request_content_sha256") != request.content_sha256
                or view.get("logical_scope_id") != member.scope_id
                or view.get("logical_scope_sha256") != member.as_dict()["scope_sha256"]
                or view.get("logical_purpose") != member.scope_kind
                or view.get("logical_heldout_row_ids") != list(member.heldout_row_ids)
                or view.get("physical_owner_scope_id") != request.physical_owner.scope_id
                or view.get("family") != family
                or view.get("fit_only_family_seal_sha256") != seal_digest
                or view.get("fit_only_family_seal_content_sha256")
                != seals[family]["content_sha256"]
                or view.get("registered_heldout_labels_accessed") is not False
            ):
                raise ValueError(
                    f"role-neutral BoW logical view changed: " f"{member.scope_id} {family}"
                )
            if not transformed:
                if (
                    view.get("view_input_policy")
                    != "sealed_row_ids_only_no_sealed_text_or_labels_v1"
                    or view.get("logical_transform_performed") is not False
                    or view.get("prediction_artifact") is not None
                    or view.get("registered_heldout_text_accessed") is not False
                    or "logical_heldout_text_sha256" in view
                ):
                    raise ValueError("cumulative BoW view accessed sealed text")
            else:
                prediction = view.get("prediction_artifact")
                expected_prediction_fields = {
                    "relative_path",
                    "sha256",
                    "size_bytes",
                    "dtype",
                    "shape",
                    "columns",
                }
                expected_columns: list[str] = []
                for record in metadata["fold_records"]:
                    if (
                        record["family"] == family
                        and (column := f"{record['view_name']}::{record['objective']}")
                        not in expected_columns
                    ):
                        expected_columns.append(column)
                expected_prediction_path = (
                    f"{_LOGICAL_VIEW_DIRECTORY}/{member.scope_id}" f"{suffix}.predictions.npy"
                )
                if (
                    view.get("view_input_policy") != "heldout_row_id_and_text_no_labels_v1"
                    or view.get("logical_transform_performed") is not True
                    or view.get("registered_heldout_text_accessed") is not True
                    or not isinstance(prediction, Mapping)
                    or set(prediction) != expected_prediction_fields
                    or prediction.get("relative_path") != expected_prediction_path
                    or prediction.get("columns") != expected_columns
                    or view.get("model_state_reloaded_for_primary_transform") is not False
                    or view.get("sealed_state_replay_checked") is not True
                ):
                    raise ValueError("exact BoW view lacks its held-out transform")
                prediction_path = tree / str(prediction.get("relative_path"))
                prediction_digest, prediction_size = _sha256_file(prediction_path)
                with prediction_path.open("rb") as handle:
                    matrix = np.load(handle, allow_pickle=False)
                if (
                    matrix.dtype.hasobject
                    or prediction.get("sha256") != prediction_digest
                    or prediction.get("size_bytes") != prediction_size
                    or prediction.get("dtype") != matrix.dtype.str
                    or prediction.get("shape") != list(matrix.shape)
                    or matrix.shape != (len(member.heldout_row_ids), len(expected_columns))
                    or not np.isfinite(matrix).all()
                ):
                    raise ValueError("exact BoW prediction artifact changed")
    post_open_events = events[first_text_access + 1 :]
    expected_exact_event_keys = {
        (event_name, family)
        for family in family_order
        for event_name in (
            "exact_heldout_transform_completed",
            "exact_logical_view_published",
        )
    }
    if (
        {(str(event.get("event")), str(event.get("family"))) for event in post_open_events}
        != expected_exact_event_keys
        or len(post_open_events) != len(expected_exact_event_keys)
        or any(
            event.get("registered_heldout_text_accessed") is not True
            or event.get("registered_heldout_labels_accessed") is not False
            for event in post_open_events
        )
    ):
        raise ValueError("exact BoW event coverage changed")
    return terminal


def load_authenticated_role_neutral_bow_nuisance_bank(
    *,
    root: Path | str,
    request: RoleNeutralBoWPhysicalGroupRequest,
) -> AuthenticatedRoleNeutralBoWNuisanceBank:
    """Reopen the sealed BoW fit and expose label-free nuisance probabilities.

    This is the only bridge intended for the matched-pair producer.  It
    authenticates the complete BoW execution first, then reads only the two
    registered fit-side derived arrays and the exact held-out nuisance
    prediction matrix.  Treatment and outcome arrays are neither returned nor
    accepted by this API.
    """

    source = Path(root)
    terminal = validate_role_neutral_bow_group_execution(
        root=source,
        request=request,
    )
    tree = source.resolve(strict=True)
    metadata, _seals = _validate_fit_side(
        root=tree,
        request=request,
    )
    fit_root = tree / _FIT_STATE_DIRECTORY
    inventory = metadata["array_inventory"]
    derived_references = metadata["derived_fit_quantities"]

    def load_fit_array(name: str) -> np.ndarray:
        reference = str(derived_references[name])
        registration = inventory[reference]
        path = fit_root / str(registration["relative_path"])
        digest, size = _sha256_file(path)
        with path.open("rb") as handle:
            values = np.load(handle, allow_pickle=False)
        if (
            values.dtype.hasobject
            or digest != registration["file_sha256"]
            or size != registration["size_bytes"]
            or values.dtype.str != registration["dtype"]
            or list(values.shape) != registration["shape"]
            or _array_sha256(values) != registration["content_sha256"]
            or values.shape != (len(request.physical_owner.fit_row_ids),)
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"authenticated BoW nuisance array changed: {name}")
        return np.asarray(values, dtype=np.float64)

    fit_propensity = load_fit_array("clipped_e_hat")
    fit_outcome_nuisance = load_fit_array("ensemble_m_hat")

    matching_registrations = [
        row
        for row in terminal["logical_views"]
        if row.get("logical_scope_id") == request.physical_owner.scope_id
        and row.get("family") == BOW_NUISANCE
    ]
    if len(matching_registrations) != 1:
        raise ValueError("BoW execution lacks one exact nuisance logical view")
    view_registration = matching_registrations[0]
    view_path = tree / str(view_registration["relative_path"])
    view_digest, view_size = _sha256_file(view_path)
    view = _read_json(
        view_path,
        label="role-neutral BoW exact nuisance view",
    )
    view_body = {
        key: copy.deepcopy(value) for key, value in view.items() if key != "content_sha256"
    }
    prediction = view.get("prediction_artifact")
    if (
        view_digest != view_registration.get("sha256")
        or view_size != view_registration.get("size_bytes")
        or view.get("content_sha256") != view_registration.get("content_sha256")
        or view.get("content_sha256") != _sha256_json(view_body)
        or not isinstance(prediction, Mapping)
    ):
        raise ValueError("BoW exact nuisance view changed after validation")
    prediction_path = tree / str(prediction["relative_path"])
    prediction_digest, prediction_size = _sha256_file(prediction_path)
    with prediction_path.open("rb") as handle:
        prediction_matrix = np.load(handle, allow_pickle=False)
    columns = tuple(map(str, prediction.get("columns") or ()))
    if (
        prediction_matrix.dtype.hasobject
        or prediction_digest != prediction.get("sha256")
        or prediction_size != prediction.get("size_bytes")
        or prediction_matrix.dtype.str != prediction.get("dtype")
        or list(prediction_matrix.shape) != prediction.get("shape")
        or prediction_matrix.shape != (len(request.physical_owner.heldout_row_ids), len(columns))
        or not np.isfinite(prediction_matrix).all()
        or len(columns) != len(set(columns))
    ):
        raise ValueError("BoW exact nuisance prediction matrix changed after validation")
    treatment_columns = [
        index for index, name in enumerate(columns) if name.endswith("::treatment_nuisance")
    ]
    outcome_columns = [
        index for index, name in enumerate(columns) if name.endswith("::outcome_nuisance")
    ]
    if (
        not treatment_columns
        or not outcome_columns
        or set(treatment_columns).intersection(outcome_columns)
        or len(treatment_columns) + len(outcome_columns) != len(columns)
    ):
        raise ValueError("BoW exact nuisance columns do not form the two required banks")
    clip = float(metadata["configuration"]["e_clip"])
    heldout_propensity = np.clip(
        np.mean(
            prediction_matrix[:, treatment_columns],
            axis=1,
        ),
        clip,
        1.0 - clip,
    )
    heldout_outcome_nuisance = np.mean(
        prediction_matrix[:, outcome_columns],
        axis=1,
    )
    probability_vectors = (
        fit_propensity,
        fit_outcome_nuisance,
        heldout_propensity,
        heldout_outcome_nuisance,
    )
    if any(
        not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0)
        for values in probability_vectors
    ):
        raise ValueError("authenticated BoW nuisance bank contains non-probability values")

    kwargs: dict[str, Any] = {
        "plan_scientific_content_sha256": (request.plan_scientific_content_sha256),
        "physical_owner_scope_id": request.physical_owner.scope_id,
        "fit_row_ids": tuple(request.physical_owner.fit_row_ids),
        "heldout_row_ids": tuple(request.physical_owner.heldout_row_ids),
        "fit_propensity_probability": tuple(map(float, fit_propensity)),
        "fit_outcome_nuisance_probability": tuple(map(float, fit_outcome_nuisance)),
        "heldout_propensity_probability": tuple(map(float, heldout_propensity)),
        "heldout_outcome_nuisance_probability": tuple(map(float, heldout_outcome_nuisance)),
        "source_terminal_content_sha256": str(terminal["content_sha256"]),
        "fit_state_artifact_sha256": str(terminal["fit_state_artifact_sha256"]),
    }
    identity_body = {
        "schema_version": "production_role_neutral_bow_nuisance_bank_v1",
        "plan_scientific_content_sha256": (kwargs["plan_scientific_content_sha256"]),
        "physical_owner_scope_id": kwargs["physical_owner_scope_id"],
        "fit_row_ids": list(kwargs["fit_row_ids"]),
        "heldout_row_ids": list(kwargs["heldout_row_ids"]),
        "fit_propensity_probability_sha256": _float_hex_sha256(fit_propensity),
        "fit_outcome_nuisance_probability_sha256": _float_hex_sha256(fit_outcome_nuisance),
        "heldout_propensity_probability_sha256": _float_hex_sha256(heldout_propensity),
        "heldout_outcome_nuisance_probability_sha256": (
            _float_hex_sha256(heldout_outcome_nuisance)
        ),
        "source_terminal_content_sha256": (kwargs["source_terminal_content_sha256"]),
        "fit_state_artifact_sha256": kwargs["fit_state_artifact_sha256"],
        "heldout_treatment_field_present": False,
        "heldout_outcome_field_present": False,
    }
    bank = AuthenticatedRoleNeutralBoWNuisanceBank(
        **kwargs,
        content_sha256=_sha256_json(identity_body),
    )
    if (
        _read_json(
            tree / _TERMINAL_FILE,
            label="role-neutral BoW execution manifest",
        )
        != terminal
    ):
        raise RuntimeError("BoW execution changed while its nuisance bank was opened")
    return bank


__all__ = [
    "AuthenticatedRoleNeutralBoWNuisanceBank",
    "ROLE_NEUTRAL_BOW_FIT_STATE_SCHEMA",
    "ROLE_NEUTRAL_BOW_GROUP_EXECUTION_SCHEMA",
    "ROLE_NEUTRAL_BOW_GROUP_REQUEST_SCHEMA",
    "ROLE_NEUTRAL_BOW_LOGICAL_VIEW_SCHEMA",
    "RoleNeutralBoWPhysicalGroupRequest",
    "execute_role_neutral_bow_physical_group",
    "load_authenticated_role_neutral_bow_nuisance_bank",
    "replay_role_neutral_bow_exact_transform",
    "validate_role_neutral_bow_group_execution",
]
