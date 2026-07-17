"""Authenticated providers for post-extraction untouched-gate review.

This module contains no model-launching code.  It composes injected nested
backends and already-authenticated neural-query banks into the two strict gate
views consumed by :mod:`all_evidence_post_extraction_review`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .all_evidence_post_extraction_review import (
    GateFeatureBankView,
    GateSourceSignalView,
    ObservableCausalRows,
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .fold_honest_r_stack import FitRowProvenance
from .nested_fold_signal_producer import (
    FoldPredictionRows,
    FoldTrainingRows,
    NestedEffectSignalBackend,
    NestedNuisanceBackend,
    SignalFoldPrediction,
)
from .neural_query_signal_fusion_adapter import (
    NeuralQueryFeatureBank,
    NeuralQueryFeatureBanks,
)

NESTED_GATE_SOURCE_CACHE_SCHEMA_VERSION = "nested_gate_source_cache_v2"
NESTED_GATE_SOURCE_PROVIDER_ID = "nested_gate_source_signal_provider_v2"
NEURAL_QUERY_GATE_ADAPTER_ID = "neural_query_single_inner_gate_adapter_v1"

_FORBIDDEN_FIELD = re.compile(r"(?:^|_)(?:true|oracle|ground_truth)(?:_|$)", flags=re.IGNORECASE)
_CACHE_FIELDS = frozenset(
    {
        "schema_version",
        "cache_key",
        "binding",
        "gate_row_ids",
        "source_names",
        "source_kinds",
        "values",
        "fit_row_provenance",
        "content_sha256",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _json_safe(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError(f"{path} contains an empty identity field")
            if _FORBIDDEN_FIELD.search(key):
                raise ValueError(f"{path} contains a forbidden benchmark identity field")
            if key in result:
                raise ValueError(f"{path} contains colliding identity fields")
            result[key] = _json_safe(raw_value, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, path=f"{path}[]") for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path=path)
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and _FORBIDDEN_FIELD.search(value):
            raise ValueError(f"{path} contains a forbidden benchmark identity value")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite identity value")
        return value
    raise TypeError(f"{path} must be closed JSON-compatible metadata")


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _canonical_integer_rows(values: Sequence[Any], *, name: str) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc
    if not raw:
        raise ValueError(f"{name} must be non-empty")
    result: list[int] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must contain canonical integer row IDs")
        normalized = int(value)
        if normalized < 0:
            raise ValueError(f"{name} cannot contain negative row IDs")
        result.append(normalized)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique row IDs")
    return tuple(result)


def _exact_texts(values: Sequence[Any], *, name: str, length: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of strings")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of strings") from exc
    if len(raw) != int(length) or not all(isinstance(value, str) for value in raw):
        raise ValueError(f"{name} must contain exactly {length} strings")
    return raw


def _float_hex_hash(values: Sequence[float]) -> str:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("observable vectors must be one-dimensional and finite")
    return _sha256([float(value).hex() for value in vector])


def _row_hash(values: Sequence[int]) -> str:
    return _sha256([int(value) for value in values])


def _text_hash(values: Sequence[str]) -> str:
    # Hash exact strings before FoldTrainingRows performs its model-facing text
    # normalization.  Whitespace/case changes therefore cannot alias a cache.
    return _sha256(list(values))


def _serialize_lineage(lineage: FitRowProvenance) -> dict[str, Any]:
    if not isinstance(lineage, FitRowProvenance):
        raise TypeError("lineage must be FitRowProvenance")
    rows = sorted(lineage.fit_row_ids)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in rows):
        raise TypeError("cached lineage must use canonical integer row IDs")
    return {
        "fit_row_ids": rows,
        "upstream": [_serialize_lineage(item) for item in lineage.upstream],
    }


def _deserialize_lineage(value: Any) -> FitRowProvenance:
    if not isinstance(value, Mapping) or set(value) != {"fit_row_ids", "upstream"}:
        raise ValueError("cached lineage does not match its closed schema")
    raw_rows = value["fit_row_ids"]
    if not isinstance(raw_rows, list):
        raise TypeError("cached fit_row_ids must be a list")
    rows_list: list[int] = []
    for item in raw_rows:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise TypeError("cached fit_row_ids must contain canonical integer row IDs")
        normalized = int(item)
        if normalized < 0:
            raise ValueError("cached fit_row_ids cannot contain negative row IDs")
        rows_list.append(normalized)
    if len(rows_list) != len(set(rows_list)):
        raise ValueError("cached fit_row_ids cannot contain duplicates")
    upstream = value["upstream"]
    if not isinstance(upstream, list):
        raise TypeError("cached lineage upstream must be a list")
    return FitRowProvenance(
        fit_row_ids=frozenset(rows_list),
        upstream=tuple(_deserialize_lineage(item) for item in upstream),
    )


def _validate_nested_prediction(
    prediction: SignalFoldPrediction,
    *,
    context_row_ids: tuple[int, ...],
    gate_row_ids: tuple[int, ...],
    source_id: str,
) -> tuple[np.ndarray, FitRowProvenance]:
    if not isinstance(prediction, SignalFoldPrediction):
        raise TypeError(f"{source_id} returned the wrong prediction type")
    values = np.asarray(prediction.values, dtype=float)
    if values.shape != (len(gate_row_ids),) or not np.isfinite(values).all():
        raise ValueError(f"{source_id} returned malformed or non-finite gate predictions")
    if not isinstance(prediction.provenance, FitRowProvenance):
        raise TypeError(f"{source_id} returned malformed fit-row provenance")
    recursive = prediction.provenance.recursive_fit_row_ids()
    if not recursive:
        raise ValueError(f"{source_id} returned empty fit-row provenance")
    if not recursive <= frozenset(context_row_ids):
        raise ValueError(f"{source_id} lineage leaves the supplied review context")
    if recursive & frozenset(gate_row_ids):
        raise ValueError(f"{source_id} lineage touches an untouched gate row")
    return values, prediction.provenance


class NestedGateSourceSignalProvider:
    """Cache context-only nested effect fits for sequential review gates.

    ``prepare_gate_source_view`` is the explicit data-bearing operation.  Once
    prepared, ``get_gate_source_view`` implements the runner's narrow provider
    protocol using only ``outer_fold`` and the exact gate IDs.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        nuisance_backend: NestedNuisanceBackend,
        effect_backends: Sequence[NestedEffectSignalBackend],
        inner_inner_folds: int = 3,
        random_state: int = 42,
        outcome_type: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.nuisance_backend = nuisance_backend
        self.effect_backends = tuple(effect_backends)
        self.inner_inner_folds = _positive_int(inner_inner_folds, name="inner_inner_folds")
        if self.inner_inner_folds < 2:
            raise ValueError("inner_inner_folds must be at least two")
        if isinstance(random_state, (bool, np.bool_)) or not isinstance(
            random_state, (int, np.integer)
        ):
            raise TypeError("random_state must be an integer")
        self.random_state = int(random_state)
        inferred_outcome_type = (
            getattr(getattr(self.nuisance_backend, "config", None), "outcome_type", None)
            if outcome_type is None
            else outcome_type
        )
        if inferred_outcome_type is None:
            inferred_outcome_type = "continuous"
        normalized_outcome_type = str(inferred_outcome_type).strip().lower()
        if normalized_outcome_type not in {"binary", "continuous"}:
            raise ValueError("outcome_type must be 'binary' or 'continuous'")
        self.outcome_type = normalized_outcome_type
        if not self.effect_backends:
            raise ValueError("at least one nested effect backend is required")
        if not callable(getattr(self.nuisance_backend, "fit_predict", None)) or not callable(
            getattr(self.nuisance_backend, "identity", None)
        ):
            raise TypeError("nuisance_backend does not implement the nested backend protocol")

        backend_rows: list[dict[str, Any]] = []
        declared_names: list[str] = []
        for index, backend in enumerate(self.effect_backends, start=1):
            if not callable(getattr(backend, "fit_predict", None)) or not callable(
                getattr(backend, "identity", None)
            ):
                raise TypeError("effect backend does not implement the nested backend protocol")
            name = str(getattr(backend, "signal_name", "")).strip()
            kind = str(getattr(backend, "source_kind", "")).strip()
            if not name or not kind:
                raise ValueError("effect backends require non-empty signal_name and source_kind")
            declared_names.append(name)
            backend_rows.append(
                {
                    "opaque_source_name": f"review_effect_source_{index:04d}",
                    "opaque_source_kind": f"nested_calibrated_effect_{index:04d}",
                    "declared_name_sha256": _sha256(name),
                    "declared_kind_sha256": _sha256(kind),
                    "backend_identity": _json_safe(
                        backend.identity(), path=f"effect_backends[{index - 1}].identity"
                    ),
                }
            )
        if len(declared_names) != len(set(declared_names)):
            raise ValueError("effect backend signal names must be unique")
        self._backend_rows = tuple(backend_rows)
        self._nuisance_identity = _json_safe(
            self.nuisance_backend.identity(), path="nuisance_backend.identity"
        )
        self._prepared: dict[tuple[int, tuple[int, ...]], GateSourceSignalView] = {}

    def _assert_backend_identities_stable(self) -> None:
        current_nuisance = _json_safe(
            self.nuisance_backend.identity(), path="nuisance_backend.identity"
        )
        if current_nuisance != self._nuisance_identity:
            raise ValueError("nuisance backend identity changed after provider construction")
        for index, (backend, frozen) in enumerate(
            zip(self.effect_backends, self._backend_rows), start=1
        ):
            current_identity = _json_safe(
                backend.identity(), path=f"effect_backends[{index - 1}].identity"
            )
            if current_identity != frozen["backend_identity"]:
                raise ValueError("effect backend identity changed after provider construction")
            if (
                _sha256(str(getattr(backend, "signal_name", "")).strip())
                != frozen["declared_name_sha256"]
                or _sha256(str(getattr(backend, "source_kind", "")).strip())
                != frozen["declared_kind_sha256"]
            ):
                raise ValueError("effect backend declaration changed after provider construction")

    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": NESTED_GATE_SOURCE_PROVIDER_ID,
            "provider_code_sha256": _module_sha256(),
            "inner_inner_folds": self.inner_inner_folds,
            "random_state": self.random_state,
            "outcome_type": self.outcome_type,
            "nuisance_backend": self._nuisance_identity,
            "effect_backends": list(self._backend_rows),
            "gate_bind_api": "exact_row_ids_and_text_only_v1",
            "adaptive_acceptance_conditional_context_supported": False,
            "intended_use": "legacy_gate_preservation_only",
        }

    def _binding(
        self,
        *,
        outer_fold: int,
        context: ObservableCausalRows,
        exact_gate_row_ids: tuple[int, ...],
        context_texts: tuple[str, ...],
        gate_texts: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "provider_identity": _json_safe(self.identity(), path="provider.identity"),
            "outer_fold": outer_fold,
            "context_row_ids_sha256": _row_hash(context.row_ids),
            "context_text_sha256": _text_hash(context_texts),
            "context_treatment_sha256": _float_hex_hash(context.treatment),
            "context_outcome_sha256": _float_hex_hash(context.outcome),
            "gate_row_ids_sha256": _row_hash(exact_gate_row_ids),
            "gate_text_sha256": _text_hash(gate_texts),
            "context_row_count": len(context.row_ids),
            "gate_row_count": len(exact_gate_row_ids),
            "gate_bind_api": "exact_row_ids_and_text_only_v1",
        }

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _load_cached(
        self,
        path: Path,
        *,
        cache_key: str,
        binding: Mapping[str, Any],
        context_row_ids: tuple[int, ...],
        gate_row_ids: tuple[int, ...],
    ) -> GateSourceSignalView:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("nested gate source cache is unreadable or malformed") from exc
        if not isinstance(payload, Mapping) or set(payload) != _CACHE_FIELDS:
            raise ValueError("nested gate source cache does not match its closed schema")
        if payload.get("schema_version") != NESTED_GATE_SOURCE_CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported nested gate source cache schema")
        if payload.get("cache_key") != cache_key or payload.get("binding") != binding:
            raise ValueError("nested gate source cache binding mismatch")
        content = {key: value for key, value in payload.items() if key != "content_sha256"}
        if payload.get("content_sha256") != _sha256(content):
            raise ValueError("nested gate source cache content SHA-256 mismatch")
        names = tuple(str(value) for value in payload["source_names"])
        kinds = tuple(str(value) for value in payload["source_kinds"])
        expected_names = tuple(row["opaque_source_name"] for row in self._backend_rows)
        expected_kinds = tuple(row["opaque_source_kind"] for row in self._backend_rows)
        if names != expected_names or kinds != expected_kinds:
            raise ValueError("nested gate source cache source identity mismatch")
        cached_gate_ids = _canonical_integer_rows(
            payload["gate_row_ids"], name="cached gate_row_ids"
        )
        if cached_gate_ids != gate_row_ids:
            raise ValueError("nested gate source cache gate row identity/order mismatch")
        lineage_rows = payload["fit_row_provenance"]
        if not isinstance(lineage_rows, list) or len(lineage_rows) != len(names):
            raise ValueError("nested gate source cache lineage count mismatch")
        lineage = tuple(_deserialize_lineage(value) for value in lineage_rows)
        context_set = frozenset(context_row_ids)
        gate_set = frozenset(gate_row_ids)
        for source_lineage in lineage:
            recursive = source_lineage.recursive_fit_row_ids()
            if not recursive or not recursive <= context_set or recursive & gate_set:
                raise ValueError("nested gate source cache lineage is not gate-honest")
        return GateSourceSignalView(
            row_ids=cached_gate_ids,
            source_names=names,
            source_kinds=kinds,
            values=np.asarray(payload["values"], dtype=float),
            fit_row_provenance=lineage,
        )

    def _write_cached(
        self,
        path: Path,
        *,
        cache_key: str,
        binding: Mapping[str, Any],
        view: GateSourceSignalView,
    ) -> None:
        content = {
            "schema_version": NESTED_GATE_SOURCE_CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "binding": binding,
            "gate_row_ids": list(view.row_ids),
            "source_names": list(view.source_names),
            "source_kinds": list(view.source_kinds),
            "values": view.values.tolist(),
            # Each fitted backend produces one lineage shared by all gate rows.
            "fit_row_provenance": [
                _serialize_lineage(source_lineages[0])
                for source_lineages in view.fit_row_provenance
            ],
        }
        payload = {**content, "content_sha256": _sha256(content)}
        encoded = (_canonical_json(payload) + "\n").encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as handle:
                handle.write(encoded)
                handle.flush()
                temporary = Path(handle.name)
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def prepare_gate_source_view(
        self,
        *,
        outer_fold: int,
        context: ObservableCausalRows,
        context_texts: Sequence[str],
        gate_texts: Sequence[str],
        exact_gate_row_ids: Sequence[int],
    ) -> GateSourceSignalView:
        outer_fold = _positive_int(outer_fold, name="outer_fold")
        self._assert_backend_identities_stable()
        if not isinstance(context, ObservableCausalRows):
            raise TypeError("context must be ObservableCausalRows")
        context_ids = _canonical_integer_rows(context.row_ids, name="context.row_ids")
        gate_ids = _canonical_integer_rows(exact_gate_row_ids, name="exact_gate_row_ids")
        if set(context_ids) & set(gate_ids):
            raise ValueError("review context and untouched gate row IDs must be disjoint")
        exact_context_texts = _exact_texts(
            context_texts, name="context_texts", length=len(context_ids)
        )
        exact_gate_texts = _exact_texts(gate_texts, name="gate_texts", length=len(gate_ids))
        binding = self._binding(
            outer_fold=outer_fold,
            context=context,
            exact_gate_row_ids=gate_ids,
            context_texts=exact_context_texts,
            gate_texts=exact_gate_texts,
        )
        cache_key = _sha256(binding)
        cache_path = self._cache_path(cache_key)
        if cache_path.exists():
            view = self._load_cached(
                cache_path,
                cache_key=cache_key,
                binding=binding,
                context_row_ids=context_ids,
                gate_row_ids=gate_ids,
            )
            if view.row_ids != gate_ids:
                raise ValueError("cached gate source row identity/order mismatch")
            self._prepared[(outer_fold, gate_ids)] = view
            return view

        training = FoldTrainingRows(
            row_ids=context_ids,
            texts=exact_context_texts,
            treatment=context.treatment,
            outcome=context.outcome,
            outcome_type=self.outcome_type,
        )
        prediction = FoldPredictionRows(row_ids=gate_ids, texts=exact_gate_texts)
        columns: list[np.ndarray] = []
        lineages: list[FitRowProvenance] = []
        for index, (backend, backend_row) in enumerate(
            zip(self.effect_backends, self._backend_rows), start=1
        ):
            result = backend.fit_predict(
                training,
                prediction,
                nuisance_backend=self.nuisance_backend,
                inner_inner_folds=self.inner_inner_folds,
                random_state=self.random_state + outer_fold * 100_000 + index * 1_000,
            )
            values, lineage = _validate_nested_prediction(
                result,
                context_row_ids=context_ids,
                gate_row_ids=gate_ids,
                source_id=backend_row["opaque_source_name"],
            )
            columns.append(values)
            lineages.append(lineage)
        view = GateSourceSignalView(
            row_ids=gate_ids,
            source_names=tuple(row["opaque_source_name"] for row in self._backend_rows),
            source_kinds=tuple(row["opaque_source_kind"] for row in self._backend_rows),
            values=np.column_stack(columns),
            fit_row_provenance=tuple(lineages),
        )
        self._write_cached(
            cache_path,
            cache_key=cache_key,
            binding=binding,
            view=view,
        )
        self._prepared[(outer_fold, gate_ids)] = view
        return view

    def bind_fold(
        self,
        *,
        outer_fold: int,
        context: ObservableCausalRows,
        context_texts: Sequence[str],
        gate_texts: Sequence[str],
        exact_gate_row_ids: Sequence[int],
    ) -> "BoundNestedGateSourceSignalProvider":
        """Prepare one gate and return a label-free runner-facing provider."""

        view = self.prepare_gate_source_view(
            outer_fold=outer_fold,
            context=context,
            context_texts=context_texts,
            gate_texts=gate_texts,
            exact_gate_row_ids=exact_gate_row_ids,
        )
        return BoundNestedGateSourceSignalProvider(
            outer_fold=_positive_int(outer_fold, name="outer_fold"),
            exact_gate_row_ids=view.row_ids,
            view=view,
            parent_identity=self.identity(),
        )

    def get_gate_source_view(
        self,
        *,
        outer_fold: int,
        exact_gate_row_ids: Sequence[int],
    ) -> GateSourceSignalView:
        fold = _positive_int(outer_fold, name="outer_fold")
        gate_ids = _canonical_integer_rows(exact_gate_row_ids, name="exact_gate_row_ids")
        try:
            return self._prepared[(fold, gate_ids)]
        except KeyError as exc:
            raise RuntimeError(
                "gate source view is not prepared; call prepare_gate_source_view with "
                "the exact spent context plus gate row IDs and texts first"
            ) from exc


class BoundNestedGateSourceSignalProvider:
    """One prepared gate with the runner's narrow label-free lookup API."""

    def __init__(
        self,
        *,
        outer_fold: int,
        exact_gate_row_ids: Sequence[int],
        view: GateSourceSignalView,
        parent_identity: Mapping[str, Any],
    ) -> None:
        self.outer_fold = _positive_int(outer_fold, name="outer_fold")
        self.exact_gate_row_ids = _canonical_integer_rows(
            exact_gate_row_ids, name="exact_gate_row_ids"
        )
        if not isinstance(view, GateSourceSignalView) or view.row_ids != self.exact_gate_row_ids:
            raise ValueError("bound gate view does not match exact gate row identity/order")
        self._view = view
        self._parent_identity = _json_safe(parent_identity, path="parent_identity")

    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": "bound_nested_gate_source_signal_provider_v2",
            "outer_fold": self.outer_fold,
            "gate_row_ids_sha256": _row_hash(self.exact_gate_row_ids),
            "parent_identity_sha256": _sha256(self._parent_identity),
            "adaptive_acceptance_conditional_context_supported": False,
        }

    def get_gate_source_view(
        self,
        *,
        outer_fold: int,
        exact_gate_row_ids: Sequence[int],
    ) -> GateSourceSignalView:
        fold = _positive_int(outer_fold, name="outer_fold")
        rows = _canonical_integer_rows(exact_gate_row_ids, name="exact_gate_row_ids")
        if fold != self.outer_fold or rows != self.exact_gate_row_ids:
            raise ValueError("bound provider requested with a different fold or gate")
        return self._view


def _single_complete_inner_gate_positions(
    bank: NeuralQueryFeatureBank,
    gate_row_ids: tuple[int, ...],
) -> tuple[int, tuple[int, ...]]:
    if not isinstance(bank, NeuralQueryFeatureBank):
        raise TypeError("neural query bank has the wrong authenticated type")
    if len(bank.outer_train_row_ids) != len(bank.inner_fold_ids):
        raise ValueError("neural query bank row/fold alignment was tampered")
    positions = {row_id: index for index, row_id in enumerate(bank.outer_train_row_ids)}
    missing = [row_id for row_id in gate_row_ids if row_id not in positions]
    if missing:
        raise ValueError("neural query gate contains rows outside authenticated outer train")
    gate_positions = tuple(positions[row_id] for row_id in gate_row_ids)
    fold_ids = {bank.inner_fold_ids[position] for position in gate_positions}
    if len(fold_ids) != 1:
        raise ValueError("neural query review gate spans multiple inner folds")
    fold_id = int(next(iter(fold_ids)))
    complete_fold_rows = tuple(
        row_id
        for row_id, candidate_fold in zip(bank.outer_train_row_ids, bank.inner_fold_ids)
        if candidate_fold == fold_id
    )
    if frozenset(complete_fold_rows) != frozenset(gate_row_ids):
        raise ValueError("neural query review gate must equal one complete inner fold")
    matrix = np.asarray(bank.outer_train_inner_oof, dtype=float)
    if matrix.shape != (len(bank.outer_train_row_ids), len(bank.feature_names)):
        raise ValueError("neural query bank activation matrix was tampered")
    if not np.isfinite(matrix).all():
        raise ValueError("neural query bank activation matrix contains non-finite values")
    if bank.outer_train_inner_oof.flags.writeable:
        raise ValueError("neural query bank activation matrix is no longer frozen")
    prefix = f"neural_query_{bank.bank}_"
    if not bank.feature_names or any(
        not str(name).startswith(prefix) for name in bank.feature_names
    ):
        raise ValueError("neural query bank feature identity was tampered")
    if len(bank.inner_fit_row_provenance) != len(bank.outer_train_row_ids):
        raise ValueError("neural query bank provenance alignment was tampered")
    gate_set = frozenset(gate_row_ids)
    for position in gate_positions:
        lineage = bank.inner_fit_row_provenance[position]
        if not isinstance(lineage, FitRowProvenance):
            raise TypeError("neural query bank provenance was tampered")
        recursive = lineage.recursive_fit_row_ids()
        if not recursive or not recursive <= frozenset(bank.outer_train_row_ids):
            raise ValueError("neural query bank provenance leaves authenticated outer train")
        if recursive & gate_set:
            raise ValueError("neural query bank provenance touches an untouched gate row")
    return fold_id, gate_positions


def neural_query_gate_feature_view(
    banks: NeuralQueryFeatureBanks,
    *,
    exact_gate_row_ids: Sequence[int],
) -> GateFeatureBankView:
    """Adapt one exact authenticated inner fold into a role-aware gate bank."""

    if not isinstance(banks, NeuralQueryFeatureBanks):
        raise TypeError("banks must be authenticated NeuralQueryFeatureBanks")
    gate_ids = _canonical_integer_rows(exact_gate_row_ids, name="exact_gate_row_ids")
    bank_rows = (
        (
            banks.treatment,
            PROPENSITY_NUISANCE_FEATURE_ROLE,
        ),
        (
            banks.outcome,
            OUTCOME_NUISANCE_FEATURE_ROLE,
        ),
        (
            banks.effect,
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        ),
    )
    values: list[np.ndarray] = []
    names: list[str] = []
    kinds: list[str] = []
    roles: list[str] = []
    source_major_lineage: list[tuple[FitRowProvenance, ...]] = []
    expected_fold: int | None = None
    for bank, role in bank_rows:
        if bank.outer_train_row_ids != banks.outer_train_row_ids:
            raise ValueError("neural query bank row identity/order was tampered")
        fold_id, positions = _single_complete_inner_gate_positions(bank, gate_ids)
        if expected_fold is None:
            expected_fold = fold_id
        elif fold_id != expected_fold:
            raise ValueError("neural query banks disagree on the gate inner fold")
        if bank.consumer_role != role:
            raise ValueError("neural query bank consumer role was tampered")
        values.append(np.asarray(bank.outer_train_inner_oof[list(positions)], dtype=float))
        names.extend(bank.feature_names)
        kinds.extend(["neural_query_moments"] * len(bank.feature_names))
        roles.extend([role] * len(bank.feature_names))
        per_row_lineage = tuple(bank.inner_fit_row_provenance[position] for position in positions)
        source_major_lineage.extend([per_row_lineage for _feature_name in bank.feature_names])
    return GateFeatureBankView(
        row_ids=gate_ids,
        feature_names=tuple(names),
        source_kinds=tuple(kinds),
        consumer_roles=tuple(roles),
        values=np.column_stack(values),
        fit_row_provenance=tuple(source_major_lineage),
    )


class NeuralQueryGateFeatureBankProvider:
    """Production wrapper over authenticated per-outer-fold query banks."""

    def __init__(
        self,
        banks_by_outer_fold: Mapping[int, NeuralQueryFeatureBanks],
    ) -> None:
        if not isinstance(banks_by_outer_fold, Mapping) or not banks_by_outer_fold:
            raise ValueError("banks_by_outer_fold must be a non-empty mapping")
        normalized: dict[int, NeuralQueryFeatureBanks] = {}
        for raw_fold, banks in banks_by_outer_fold.items():
            fold = _positive_int(raw_fold, name="banks_by_outer_fold key")
            if not isinstance(banks, NeuralQueryFeatureBanks):
                raise TypeError("all feature-bank registrations must be authenticated")
            if banks.outer_fold != fold:
                raise ValueError("feature-bank outer fold does not match its registry key")
            normalized[fold] = banks
            self._validated_assignments(banks)
        self._banks_by_outer_fold = normalized

    @staticmethod
    def _validated_assignments(
        banks: NeuralQueryFeatureBanks,
    ) -> dict[int, tuple[int, ...]]:
        reference_rows = tuple(banks.outer_train_row_ids)
        reference_folds = tuple(banks.treatment.inner_fold_ids)
        if len(reference_rows) != len(reference_folds) or len(set(reference_folds)) < 2:
            raise ValueError("neural query banks do not define a valid inner partition")
        expected = (
            (banks.treatment, PROPENSITY_NUISANCE_FEATURE_ROLE),
            (banks.outcome, OUTCOME_NUISANCE_FEATURE_ROLE),
            (banks.effect, UNCALIBRATED_EFFECT_MODIFIER_ROLE),
        )
        for bank, role in expected:
            if bank.outer_train_row_ids != reference_rows:
                raise ValueError("neural query banks do not share exact outer-train rows")
            if tuple(bank.inner_fold_ids) != reference_folds:
                raise ValueError("neural query banks do not share exact inner-fold assignments")
            if bank.consumer_role != role:
                raise ValueError("neural query bank has a mismatched consumer role")
        assignments: dict[int, list[int]] = {}
        for row_id, fold_id in zip(reference_rows, reference_folds):
            fold = _positive_int(fold_id, name="neural query inner_fold_id")
            assignments.setdefault(fold, []).append(int(row_id))
        return {fold: tuple(rows) for fold, rows in sorted(assignments.items())}

    def identity(self) -> Mapping[str, Any]:
        folds: list[dict[str, Any]] = []
        for outer_fold, banks in sorted(self._banks_by_outer_fold.items()):
            assignments = self._validated_assignments(banks)
            folds.append(
                {
                    "outer_fold": outer_fold,
                    "split_fingerprint": banks.split_fingerprint,
                    "manifest_sha256": banks.manifest_sha256,
                    "signal_parquet_sha256": banks.signal_parquet_sha256,
                    "outer_train_row_ids_sha256": _row_hash(banks.outer_train_row_ids),
                    "inner_partition_sha256": _sha256(
                        {str(key): list(value) for key, value in assignments.items()}
                    ),
                    "inner_fold_count": len(assignments),
                }
            )
        return {
            "provider": "neural_query_gate_feature_bank_provider_v1",
            "provider_code_sha256": _module_sha256(),
            "registered_outer_folds": folds,
            "raw_activations_are_calibrated_treatment_effects": False,
            "adaptive_acceptance_conditional_context_supported": False,
            "intended_use": "legacy_gate_preservation_only",
        }

    def _banks(self, outer_fold: int) -> NeuralQueryFeatureBanks:
        fold = _positive_int(outer_fold, name="outer_fold")
        try:
            return self._banks_by_outer_fold[fold]
        except KeyError as exc:
            raise ValueError(f"no authenticated neural query banks for outer fold {fold}") from exc

    def get_review_partition_assignments(
        self,
        *,
        outer_fold: int,
        exact_outer_train_row_ids: Sequence[int],
    ) -> Mapping[int, tuple[int, ...]]:
        banks = self._banks(outer_fold)
        requested = _canonical_integer_rows(
            exact_outer_train_row_ids, name="exact_outer_train_row_ids"
        )
        if requested != banks.outer_train_row_ids:
            raise ValueError(
                "exact_outer_train_row_ids must match authenticated row identity/order"
            )
        return self._validated_assignments(banks)

    def get_gate_feature_bank_view(
        self,
        *,
        outer_fold: int,
        exact_gate_row_ids: Sequence[int],
    ) -> GateFeatureBankView:
        banks = self._banks(outer_fold)
        # Revalidate shared assignments immediately before adapting so an
        # in-memory mutation cannot cross the provider boundary silently.
        self._validated_assignments(banks)
        return neural_query_gate_feature_view(
            banks,
            exact_gate_row_ids=exact_gate_row_ids,
        )


__all__ = [
    "NESTED_GATE_SOURCE_CACHE_SCHEMA_VERSION",
    "NESTED_GATE_SOURCE_PROVIDER_ID",
    "NEURAL_QUERY_GATE_ADAPTER_ID",
    "BoundNestedGateSourceSignalProvider",
    "NeuralQueryGateFeatureBankProvider",
    "NestedGateSourceSignalProvider",
    "neural_query_gate_feature_view",
]
