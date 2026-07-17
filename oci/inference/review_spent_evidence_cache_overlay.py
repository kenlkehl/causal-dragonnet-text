"""Authenticated, read-only reuse of spent-evidence JSON cache entries.

The adaptive-review provider's cache key already binds an entry to the exact
spent/sealed partition, spent text and labels, backend identities, and the
provider's code-bound identity.  This module reuses only historical entries
whose bytes and closed cache envelope authenticate independently and whose
binding is exactly the binding requested from the current provider.

Historical paths are never writable cache roots.  Each registered file is
read once into an immutable byte snapshot; hashing, JSON parsing, and later
materialization all use that same snapshot.  On an exact hit, those bytes are
atomically copied into the fresh output-local cache owned by the current
provider.  Misses are ordinary current-provider calls.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np

from .review_spent_evidence_provider import (
    REVIEW_SPENT_EVIDENCE_CACHE_VERSION,
    ContextFitReviewSpentEvidenceProvider,
    _exact_texts,
    _finite_vector,
    _integer_rows,
)

REVIEW_SPENT_CACHE_OVERLAY_IDENTITY_VERSION = "authenticated_review_spent_cache_overlay_identity_v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEY = re.compile(
    r"(?:^|_)(?:oracle|true|ground_truth|groundtruth)(?:_|$)", re.IGNORECASE
)
_CACHE_FIELDS = frozenset({"schema_version", "cache_key", "binding", "results", "content_sha256"})
_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "outer_fold",
        "review_round",
        "spent_row_ids_sha256",
        "sealed_row_ids_sha256",
        "ordered_spent_text_sha256",
        "spent_treatment_sha256",
        "spent_outcome_sha256",
        "backend_identities_sha256",
        "provider_identity_sha256",
    }
)
_RESULT_FIELDS = frozenset({"source_kind", "payload"})


class ReviewSpentCacheAuthenticationError(RuntimeError):
    """A registered spent-evidence cache snapshot failed authentication."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for raw_key, value in pairs:
        key = str(raw_key)
        if key in output:
            raise ValueError(f"duplicate JSON field {key!r}")
        output[key] = value
    return output


def _parse_snapshot(snapshot: bytes, *, path: Path) -> Mapping[str, Any]:
    try:
        parsed = json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_closed_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReviewSpentCacheAuthenticationError(
            f"spent-evidence cache source is not closed finite UTF-8 JSON: {path}"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ReviewSpentCacheAuthenticationError(
            "spent-evidence cache source root must be an object"
        )
    return parsed


def _reject_forbidden_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if not key or _FORBIDDEN_KEY.search(key):
                raise ReviewSpentCacheAuthenticationError(
                    f"spent-evidence cache source contains a forbidden field at {path}.{key}"
                )
            _reject_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, path=f"{path}[{index}]")


def _required_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256.fullmatch(digest):
        raise ReviewSpentCacheAuthenticationError(f"{label} must be one lowercase SHA-256")
    return digest


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ReviewSpentCacheAuthenticationError(f"{label} must be an integer")
    result = int(value)
    if result < 1:
        raise ReviewSpentCacheAuthenticationError(f"{label} must be positive")
    return result


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ReviewSpentCacheAuthenticationError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise ReviewSpentCacheAuthenticationError(f"{label} must be non-negative")
    return result


@dataclass(frozen=True)
class AuthenticatedReviewSpentCacheSource:
    """One immutable source snapshot plus its closed binding identity."""

    source_path: Path
    registered_sha256: str
    snapshot_sha256: str
    byte_count: int
    cache_key: str
    outer_fold: int
    review_round: int
    result_count: int
    provider_identity_sha256: str
    backend_identities_sha256: str
    _binding_json: str = field(repr=False)
    _snapshot: bytes = field(repr=False)

    @property
    def binding(self) -> dict[str, Any]:
        return json.loads(self._binding_json)

    @property
    def snapshot(self) -> bytes:
        # ``bytes`` is immutable; returning it cannot mutate the authenticated
        # buffer retained by this registration.
        return self._snapshot

    def identity(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "registered_sha256": self.registered_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "byte_count": int(self.byte_count),
            "cache_key": self.cache_key,
            "binding_sha256": self.cache_key,
            "outer_fold": int(self.outer_fold),
            "review_round": int(self.review_round),
            "result_count": int(self.result_count),
            "provider_identity_sha256": self.provider_identity_sha256,
            "backend_identities_sha256": self.backend_identities_sha256,
            "schema_version": REVIEW_SPENT_EVIDENCE_CACHE_VERSION,
        }


def _authenticate_source(
    path: Path,
    *,
    declared_sha256: str,
) -> AuthenticatedReviewSpentCacheSource:
    try:
        # This is the only read of the historical source.  Every subsequent
        # operation uses this exact immutable buffer.
        snapshot = path.read_bytes()
    except FileNotFoundError as exc:
        raise ReviewSpentCacheAuthenticationError(
            f"spent-evidence cache source does not exist: {path}"
        ) from exc
    except OSError as exc:
        raise ReviewSpentCacheAuthenticationError(
            f"could not read spent-evidence cache source: {path}"
        ) from exc
    snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
    if snapshot_sha256 != declared_sha256:
        raise ReviewSpentCacheAuthenticationError(
            f"spent-evidence cache source SHA-256 mismatch: {path}"
        )
    raw = _parse_snapshot(snapshot, path=path)
    _reject_forbidden_keys(raw, path="cache")
    if set(raw) != _CACHE_FIELDS:
        raise ReviewSpentCacheAuthenticationError(
            "spent-evidence cache source has an unsupported closed schema"
        )
    if raw["schema_version"] != REVIEW_SPENT_EVIDENCE_CACHE_VERSION:
        raise ReviewSpentCacheAuthenticationError(
            "spent-evidence cache source has an unsupported schema version"
        )
    content_sha256 = _required_sha256(raw["content_sha256"], label="content_sha256")
    content = {key: raw[key] for key in raw if key != "content_sha256"}
    if content_sha256 != _sha256_json(content):
        raise ReviewSpentCacheAuthenticationError(
            "spent-evidence cache source content hash mismatch"
        )
    cache_key = _required_sha256(raw["cache_key"], label="cache_key")
    if path.name != f"{cache_key}.json":
        raise ReviewSpentCacheAuthenticationError(
            "spent-evidence cache source filename does not match its cache key"
        )
    binding = raw["binding"]
    if not isinstance(binding, Mapping) or set(binding) != _BINDING_FIELDS:
        raise ReviewSpentCacheAuthenticationError(
            "spent-evidence cache source binding has an unsupported closed schema"
        )
    if binding["schema_version"] != REVIEW_SPENT_EVIDENCE_CACHE_VERSION:
        raise ReviewSpentCacheAuthenticationError(
            "spent-evidence cache source binding schema version mismatch"
        )
    if _sha256_json(binding) != cache_key:
        raise ReviewSpentCacheAuthenticationError(
            "spent-evidence cache source binding hash does not equal its cache key"
        )
    for field_name in sorted(_BINDING_FIELDS - {"schema_version", "outer_fold", "review_round"}):
        _required_sha256(binding[field_name], label=f"binding.{field_name}")
    outer_fold = _positive_integer(binding["outer_fold"], label="binding.outer_fold")
    review_round = _nonnegative_integer(binding["review_round"], label="binding.review_round")
    results = raw["results"]
    if not isinstance(results, list) or not results:
        raise ReviewSpentCacheAuthenticationError(
            "spent-evidence cache source results must be a non-empty list"
        )
    source_kinds: list[str] = []
    for index, result in enumerate(results):
        if not isinstance(result, Mapping) or set(result) != _RESULT_FIELDS:
            raise ReviewSpentCacheAuthenticationError(
                f"spent-evidence cache result {index} has an unsupported closed schema"
            )
        source_kind = str(result["source_kind"] or "").strip()
        if not source_kind or not isinstance(result["payload"], Mapping):
            raise ReviewSpentCacheAuthenticationError(
                f"spent-evidence cache result {index} is incomplete"
            )
        source_kinds.append(source_kind)
    if len(source_kinds) != len(set(source_kinds)):
        raise ReviewSpentCacheAuthenticationError(
            "spent-evidence cache source has duplicate result source kinds"
        )
    return AuthenticatedReviewSpentCacheSource(
        source_path=path,
        registered_sha256=declared_sha256,
        snapshot_sha256=snapshot_sha256,
        byte_count=len(snapshot),
        cache_key=cache_key,
        outer_fold=outer_fold,
        review_round=review_round,
        result_count=len(results),
        provider_identity_sha256=_required_sha256(
            binding["provider_identity_sha256"],
            label="binding.provider_identity_sha256",
        ),
        backend_identities_sha256=_required_sha256(
            binding["backend_identities_sha256"],
            label="binding.backend_identities_sha256",
        ),
        _binding_json=_canonical_json(binding),
        _snapshot=snapshot,
    )


def authenticate_review_spent_cache_registrations(
    entries: Sequence[str],
) -> tuple[AuthenticatedReviewSpentCacheSource, ...]:
    """Parse repeatable mandatory ``PATH::SHA256`` registrations."""

    sources: list[AuthenticatedReviewSpentCacheSource] = []
    source_paths: set[Path] = set()
    cache_keys: set[str] = set()
    for raw_entry in entries:
        entry = str(raw_entry).strip()
        raw_path, separator, raw_sha256 = entry.rpartition("::")
        if not separator or not raw_path.strip() or not raw_sha256.strip():
            raise ReviewSpentCacheAuthenticationError(
                "--read-only-review-spent-evidence-cache must use PATH::SHA256"
            )
        declared_sha256 = _required_sha256(
            raw_sha256, label="registered spent-evidence cache SHA-256"
        )
        path = Path(raw_path).expanduser().resolve()
        if path in source_paths:
            raise ReviewSpentCacheAuthenticationError(
                f"duplicate spent-evidence cache source path: {path}"
            )
        source = _authenticate_source(path, declared_sha256=declared_sha256)
        if source.cache_key in cache_keys:
            raise ReviewSpentCacheAuthenticationError(
                f"duplicate spent-evidence cache key: {source.cache_key}"
            )
        source_paths.add(path)
        cache_keys.add(source.cache_key)
        sources.append(source)
    return tuple(sources)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_provider_module_path() -> Path:
    import sys

    module = sys.modules[ContextFitReviewSpentEvidenceProvider.__module__]
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise ReviewSpentCacheAuthenticationError(
            "current spent-evidence provider has no code file identity"
        )
    return Path(module_file).resolve()


class AuthenticatedReviewSpentEvidenceCacheOverlay:
    """Materialize authenticated historical bytes only for exact current hits."""

    def __init__(
        self,
        *,
        provider: ContextFitReviewSpentEvidenceProvider,
        sources: Sequence[AuthenticatedReviewSpentCacheSource],
        output_root: Path | str,
    ) -> None:
        if type(provider) is not ContextFitReviewSpentEvidenceProvider:
            raise TypeError(
                "spent-evidence cache overlay requires the exact current context-fit provider"
            )
        self.provider = provider
        self.sources = tuple(sources)
        if not self.sources:
            raise ValueError("spent-evidence cache overlay requires at least one source")
        if not all(isinstance(item, AuthenticatedReviewSpentCacheSource) for item in self.sources):
            raise TypeError("spent-evidence cache overlay sources were not authenticated")
        self.output_root = Path(output_root).expanduser().resolve()
        self.cache_dir = Path(provider.cache_dir).resolve()
        if self.cache_dir.parent != self.output_root:
            raise ValueError(
                "spent-evidence writable cache must be a direct child of the fresh output root"
            )
        if not self.cache_dir.is_dir():
            raise ValueError("spent-evidence writable cache directory was not constructed")
        if any(self.cache_dir.iterdir()):
            raise ValueError(
                "spent-evidence writable cache must be empty when the overlay is constructed"
            )
        self._wrapper_code_sha256 = _sha256_path(Path(__file__).resolve())
        self._provider_code_path = _review_provider_module_path()
        self._provider_code_sha256 = _sha256_path(self._provider_code_path)
        self._provider_identity = self._detached_provider_identity()
        self._provider_identity_sha256 = _sha256_json(self._provider_identity)
        declared_code_sha256 = _required_sha256(
            self._provider_identity.get("provider_code_sha256"),
            label="current provider identity code hash",
        )
        if declared_code_sha256 != self._provider_code_sha256:
            raise ReviewSpentCacheAuthenticationError(
                "current spent-evidence provider identity is not bound to its current code"
            )
        self._backend_identities = self._current_backend_identities()
        self._backend_identities_sha256 = _sha256_json(list(self._backend_identities))
        declared_backends = self._provider_identity.get("backends")
        if declared_backends != list(self._backend_identities):
            raise ReviewSpentCacheAuthenticationError(
                "current spent-evidence provider identity does not match current backends"
            )
        by_key: dict[str, AuthenticatedReviewSpentCacheSource] = {}
        for source in self.sources:
            if source.provider_identity_sha256 != self._provider_identity_sha256:
                raise ReviewSpentCacheAuthenticationError(
                    "spent-evidence cache source provider identity does not match current provider"
                )
            if source.backend_identities_sha256 != self._backend_identities_sha256:
                raise ReviewSpentCacheAuthenticationError(
                    "spent-evidence cache source backend identities do not match current backends"
                )
            if source.result_count != len(self._backend_identities):
                raise ReviewSpentCacheAuthenticationError(
                    "spent-evidence cache source result count does not match current backends"
                )
            if source.cache_key in by_key:
                raise ReviewSpentCacheAuthenticationError(
                    f"duplicate spent-evidence cache key: {source.cache_key}"
                )
            by_key[source.cache_key] = source
        self._sources_by_key = by_key
        self._materialized_keys: set[str] = set()
        self._materialization_lock = threading.Lock()
        required_families = self._provider_identity.get("required_source_families")
        if not isinstance(required_families, list):
            raise ReviewSpentCacheAuthenticationError(
                "current spent-evidence provider identity lacks required source families"
            )
        self._identity = {
            "provider": REVIEW_SPENT_CACHE_OVERLAY_IDENTITY_VERSION,
            "wrapper_code_sha256": self._wrapper_code_sha256,
            "delegate_provider_identity": copy.deepcopy(self._provider_identity),
            "delegate_provider_identity_sha256": self._provider_identity_sha256,
            "delegate_provider_code_sha256": self._provider_code_sha256,
            "delegate_backend_identities_sha256": self._backend_identities_sha256,
            "required_source_families": list(required_families),
            "read_only_source_count": len(self.sources),
            "read_only_sources": [source.identity() for source in self.sources],
            "source_authentication": (
                "one_immutable_byte_snapshot_external_sha256_closed_json_and_binding"
            ),
            "materialization_policy": (
                "exact_binding_hit_to_fresh_output_local_writable_cache_only"
            ),
            "historical_source_writes_allowed": False,
            "extraction_or_checkpoint_reuse_enabled": False,
        }

    def _detached_provider_identity(self) -> dict[str, Any]:
        raw = self.provider.identity()
        if not isinstance(raw, Mapping):
            raise TypeError("current spent-evidence provider identity must be a mapping")
        _reject_forbidden_keys(raw, path="provider_identity")
        return json.loads(_canonical_json(raw))

    def _current_backend_identities(self) -> tuple[dict[str, Any], ...]:
        current = self.provider._current_backend_identities()
        return tuple(json.loads(_canonical_json(value)) for value in current)

    def _assert_current(self) -> None:
        if _sha256_path(Path(__file__).resolve()) != self._wrapper_code_sha256:
            raise RuntimeError("spent-evidence cache overlay code changed after binding")
        if _sha256_path(self._provider_code_path) != self._provider_code_sha256:
            raise RuntimeError("spent-evidence provider code changed after cache authentication")
        if self._detached_provider_identity() != self._provider_identity:
            raise RuntimeError(
                "spent-evidence provider identity changed after cache authentication"
            )
        if self._current_backend_identities() != self._backend_identities:
            raise RuntimeError("spent-evidence backend identity changed after cache authentication")

    def identity(self) -> Mapping[str, Any]:
        self._assert_current()
        return copy.deepcopy(self._identity)

    def _request_binding(
        self,
        *,
        outer_fold: int,
        review_round: int,
        exact_spent_row_ids: tuple[int, ...],
        exact_sealed_row_ids: tuple[int, ...],
        spent_texts: tuple[str, ...],
        spent_treatment: np.ndarray,
        spent_outcome: np.ndarray,
    ) -> dict[str, Any]:
        if (
            isinstance(outer_fold, (bool, np.bool_))
            or not isinstance(outer_fold, (int, np.integer))
            or int(outer_fold) < 1
        ):
            raise ValueError("outer_fold must be positive")
        if (
            isinstance(review_round, (bool, np.bool_))
            or not isinstance(review_round, (int, np.integer))
            or int(review_round) < 0
        ):
            raise ValueError("review_round must be non-negative")
        normalized_outer_fold = int(outer_fold)
        normalized_review_round = int(review_round)
        spent_ids = _integer_rows(exact_spent_row_ids, name="exact_spent_row_ids")
        sealed_ids = _integer_rows(exact_sealed_row_ids, name="exact_sealed_row_ids")
        if set(spent_ids) & set(sealed_ids):
            raise ValueError("spent and sealed review rows overlap")
        texts = _exact_texts(spent_texts, rows=len(spent_ids))
        treatment = _finite_vector(spent_treatment, name="spent_treatment", rows=len(spent_ids))
        outcome = _finite_vector(spent_outcome, name="spent_outcome", rows=len(spent_ids))
        if not set(np.unique(treatment)).issubset({0.0, 1.0}):
            raise ValueError("spent_treatment must be binary")
        # The current provider owns the binding semantics.  Calling its exact
        # binding implementation prevents this overlay from inventing a weaker
        # or parallel cache-key definition.
        return self.provider._binding(
            outer_fold=normalized_outer_fold,
            review_round=normalized_review_round,
            spent_ids=spent_ids,
            sealed_ids=sealed_ids,
            spent_texts=texts,
            treatment=treatment,
            outcome=outcome,
        )

    def _materialize(self, source: AuthenticatedReviewSpentCacheSource) -> None:
        target = self.cache_dir / f"{source.cache_key}.json"
        with self._materialization_lock:
            if source.cache_key in self._materialized_keys:
                try:
                    current = target.read_bytes()
                except OSError as exc:
                    raise RuntimeError(
                        "materialized spent-evidence cache entry became unavailable"
                    ) from exc
                if current != source.snapshot:
                    raise RuntimeError(
                        "materialized spent-evidence cache bytes changed during the run"
                    )
                return
            if target.exists():
                raise RuntimeError(
                    "fresh spent-evidence writable cache unexpectedly contains a hit target"
                )
            handle, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=self.cache_dir)
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(source.snapshot)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_name, target)
                directory_fd = os.open(self.cache_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            if hashlib.sha256(target.read_bytes()).hexdigest() != source.snapshot_sha256:
                raise RuntimeError(
                    "materialized spent-evidence cache bytes failed local verification"
                )
            self._materialized_keys.add(source.cache_key)

    def get_spent_evidence_inputs(
        self,
        *,
        outer_fold: int,
        review_round: int,
        exact_spent_row_ids: tuple[int, ...],
        exact_sealed_row_ids: tuple[int, ...],
        spent_texts: tuple[str, ...],
        spent_treatment: np.ndarray,
        spent_outcome: np.ndarray,
    ):
        self._assert_current()
        binding = self._request_binding(
            outer_fold=outer_fold,
            review_round=review_round,
            exact_spent_row_ids=exact_spent_row_ids,
            exact_sealed_row_ids=exact_sealed_row_ids,
            spent_texts=spent_texts,
            spent_treatment=spent_treatment,
            spent_outcome=spent_outcome,
        )
        cache_key = _sha256_json(binding)
        source = self._sources_by_key.get(cache_key)
        if source is not None:
            if source.binding != binding:
                raise ReviewSpentCacheAuthenticationError(
                    "spent-evidence cache hash matched but exact binding did not"
                )
            self._materialize(source)
        return self.provider.get_spent_evidence_inputs(
            outer_fold=outer_fold,
            review_round=review_round,
            exact_spent_row_ids=exact_spent_row_ids,
            exact_sealed_row_ids=exact_sealed_row_ids,
            spent_texts=spent_texts,
            spent_treatment=spent_treatment,
            spent_outcome=spent_outcome,
        )


__all__ = [
    "AuthenticatedReviewSpentCacheSource",
    "AuthenticatedReviewSpentEvidenceCacheOverlay",
    "REVIEW_SPENT_CACHE_OVERLAY_IDENTITY_VERSION",
    "ReviewSpentCacheAuthenticationError",
    "authenticate_review_spent_cache_registrations",
]
