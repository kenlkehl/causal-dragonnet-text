"""Authenticated, read-only reuse of historical explicit-feature caches.

The historical extraction cache format stores only positional feature columns.
That is not enough information to decide whether a file is safe to reuse in a
new run.  This module therefore consumes a separately generated v2 index whose
entries bind a cache artifact to all of the identities that affect extraction:

* the exact extraction contract;
* the remote model identity and prompt-template version;
* the ordered row/text fingerprint of the sanitized dataset; and
* the cache artifact's byte hash and expected row count.

The overlay never writes to a historical cache root.  A missing exact entry is
a normal cache miss and is delegated to the injected current provider.  A
present but invalid entry is treated as corruption and raises instead of
silently falling through.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

LEGACY_EXTRACTION_CACHE_INDEX_SCHEMA_VERSION = "legacy_extraction_cache_index_v2"
FROZEN_EXTRACTION_CACHE_OVERLAY_IDENTITY_VERSION = "frozen_extraction_cache_overlay_identity_v1"
CACHE_OVERLAY_REPORT_SCHEMA_VERSION = "frozen_extraction_cache_overlay_report_v2"
_ROW_INDEX_COLUMN = "__oci_cache_row_index"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_COLUMN = re.compile(r"(?:^|_)(?:true|oracle|ground_truth)(?:_|$)", flags=re.IGNORECASE)


class CacheAuthenticationError(RuntimeError):
    """A declared historical cache artifact failed an integrity check."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_extraction_contract(spec: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Return the exact JSON contract used by the overlay identity."""

    if is_dataclass(spec) and not isinstance(spec, type):
        spec = asdict(spec)
    if not isinstance(spec, Mapping):
        raise TypeError("an extraction contract must be a mapping or dataclass")
    contract = dict(spec)
    allowed = {"name", "type", "categories", "description", "roles", "value_aliases"}
    unexpected = set(contract) - allowed
    if unexpected:
        raise ValueError(f"extraction contract contains unsupported fields: {sorted(unexpected)}")
    required = {"name", "type", "roles"}
    missing = required - set(contract)
    if missing:
        raise ValueError(f"extraction contract is missing fields: {sorted(missing)}")
    name = str(contract.get("name") or "").strip()
    if not name:
        raise ValueError("extraction contract name must be non-empty")
    if _FORBIDDEN_COLUMN.search(name):
        raise ValueError("oracle/true feature contracts cannot enter the cache overlay")
    feature_type = str(contract.get("type") or "").strip().lower()
    if feature_type not in {"categorical", "continuous"}:
        raise ValueError("extraction contract type must be categorical or continuous")
    roles = contract.get("roles")
    if not isinstance(roles, (list, tuple)) or not roles:
        raise ValueError("extraction contract roles must be a non-empty list")

    # Normalize only JSON container types.  Semantically relevant fields such as
    # role order and category order intentionally remain part of the identity.
    contract["name"] = name
    contract["type"] = feature_type
    contract["roles"] = [str(value) for value in roles]
    contract.setdefault("categories", None)
    contract.setdefault("description", None)
    contract.setdefault("value_aliases", None)
    if contract.get("categories") is not None:
        contract["categories"] = [str(value) for value in contract["categories"]]
    aliases = contract.get("value_aliases")
    if isinstance(aliases, Mapping):
        contract["value_aliases"] = {
            str(key): [
                str(value) for value in (values if isinstance(values, (list, tuple)) else [values])
            ]
            for key, values in aliases.items()
        }
    # A JSON round trip rejects unserializable objects and returns a detached
    # representation that callers cannot mutate behind our back.
    return json.loads(_canonical_json(contract))


def extraction_contract_sha256(spec: Mapping[str, Any] | Any) -> str:
    contract = canonical_extraction_contract(spec)
    return hashlib.sha256(_canonical_json(contract).encode("utf-8")).hexdigest()


def ordered_dataset_text_fingerprint(
    dataset: pd.DataFrame,
    *,
    row_id_column: str = "_oci_row_id",
    text_column: str = "text",
) -> str:
    """Fingerprint exact ordered row IDs and their exact text bytes."""

    missing = {row_id_column, text_column} - set(dataset.columns)
    if missing:
        raise ValueError(f"dataset is missing fingerprint columns: {sorted(missing)}")
    if dataset[row_id_column].isna().any() or dataset[row_id_column].duplicated().any():
        raise ValueError("dataset row IDs must be complete and unique")
    rows = [
        {
            "row_id_type": type(row_id).__name__,
            "row_id": repr(row_id),
            "text_sha256": hashlib.sha256(str(text or "").encode("utf-8")).hexdigest(),
        }
        for row_id, text in zip(dataset[row_id_column].tolist(), dataset[text_column].tolist())
    ]
    return hashlib.sha256(_canonical_json(rows).encode("utf-8")).hexdigest()


def expected_extraction_columns(spec: Mapping[str, Any] | Any) -> tuple[str, str]:
    name = canonical_extraction_contract(spec)["name"]
    return f"explicit_feat_{name}", f"explicit_feat_{name}_missing"


@dataclass(frozen=True)
class CacheIndexIdentity:
    """Exact bytes and location of one parsed cache index."""

    path: str
    sha256: str
    byte_count: int
    schema_version: str
    entry_count: int


@dataclass(frozen=True)
class AuthenticatedCacheHit:
    """Provenance for artifact bytes authenticated and parsed for one hit."""

    contract_sha256: str
    cache_index_path: str
    cache_index_sha256: str
    cache_index_entry_position: int
    artifact_path: str
    artifact_sha256: str
    artifact_byte_count: int


@dataclass(frozen=True)
class CacheOverlayReport:
    dataset_text_fingerprint: str
    model_identity: str
    prompt_template_version: str
    cache_hit_contract_hashes: tuple[str, ...]
    cache_miss_contract_hashes: tuple[str, ...]
    authenticated_artifact_paths: tuple[str, ...]
    authenticated_artifact_sha256s: tuple[str, ...] = ()
    cache_index_identities: tuple[CacheIndexIdentity, ...] = ()
    authenticated_cache_hits: tuple[AuthenticatedCacheHit, ...] = ()
    overlay_identity_sha256: str | None = None
    schema_version: str = CACHE_OVERLAY_REPORT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _IndexEntry:
    contract: dict[str, Any]
    contract_sha256: str
    model_identity: str
    prompt_template_version: str
    dataset_text_fingerprint: str
    expected_row_count: int
    artifact_path: Path
    artifact_sha256: str
    cache_index_path: Path
    cache_index_sha256: str
    cache_index_entry_position: int

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.contract_sha256,
            self.model_identity,
            self.prompt_template_version,
            self.dataset_text_fingerprint,
        )


class FrozenExtractionCacheOverlay:
    """Resolve exact historical cache hits before invoking a current provider."""

    def __init__(
        self,
        index_paths: Sequence[Path | str],
        *,
        expected_row_count: int = 1000,
        row_id_column: str = "_oci_row_id",
        text_column: str = "text",
    ) -> None:
        if int(expected_row_count) < 1:
            raise ValueError("expected_row_count must be positive")
        self.expected_row_count = int(expected_row_count)
        self.row_id_column = str(row_id_column)
        self.text_column = str(text_column)
        self._entries: dict[tuple[str, str, str, str], _IndexEntry] = {}
        self._index_identities: list[CacheIndexIdentity] = []
        for raw_path in index_paths:
            self._load_index(Path(raw_path).resolve())
        self._identity = self._identity_payload()

    def _load_index(self, path: Path) -> None:
        try:
            index_bytes = path.read_bytes()
        except FileNotFoundError as exc:
            raise CacheAuthenticationError(f"cache index does not exist: {path}") from exc
        except OSError as exc:
            raise CacheAuthenticationError(f"could not read cache index: {path}") from exc
        index_sha256 = hashlib.sha256(index_bytes).hexdigest()
        try:
            # Parse the exact byte string whose digest becomes provenance. This
            # avoids separately hashing and parsing two different file reads.
            payload = json.loads(index_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CacheAuthenticationError(f"cache index is invalid JSON: {path}") from exc
        if not isinstance(payload, Mapping):
            raise CacheAuthenticationError("cache index root must be an object")
        if payload.get("schema_version") != LEGACY_EXTRACTION_CACHE_INDEX_SCHEMA_VERSION:
            raise CacheAuthenticationError(
                f"unsupported cache index schema: {payload.get('schema_version')!r}"
            )
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise CacheAuthenticationError("cache index entries must be a list")
        for position, raw in enumerate(entries):
            if not isinstance(raw, Mapping):
                raise CacheAuthenticationError(f"cache index entry {position} is not an object")
            contract = canonical_extraction_contract(raw.get("contract"))
            contract_digest = str(raw.get("contract_sha256") or "")
            if contract_digest != extraction_contract_sha256(contract):
                raise CacheAuthenticationError(
                    f"cache index entry {position} has an invalid contract hash"
                )
            artifact_digest = str(raw.get("artifact_sha256") or "")
            dataset_digest = str(raw.get("dataset_text_fingerprint") or "")
            if not _SHA256.fullmatch(artifact_digest) or not _SHA256.fullmatch(dataset_digest):
                raise CacheAuthenticationError(
                    f"cache index entry {position} has an invalid SHA-256 identity"
                )
            model = str(raw.get("model_identity") or "").strip()
            template = str(raw.get("prompt_template_version") or "").strip()
            if not model or not template:
                raise CacheAuthenticationError(
                    f"cache index entry {position} lacks model/template identity"
                )
            expected_rows = int(raw.get("expected_row_count", -1))
            if expected_rows != self.expected_row_count:
                raise CacheAuthenticationError(
                    f"cache index entry {position} row-count contract is "
                    f"{expected_rows}, expected {self.expected_row_count}"
                )
            requested_artifact = Path(str(raw.get("artifact_path") or ""))
            artifact_path = (
                requested_artifact
                if requested_artifact.is_absolute()
                else (path.parent / requested_artifact)
            ).resolve()
            entry = _IndexEntry(
                contract=contract,
                contract_sha256=contract_digest,
                model_identity=model,
                prompt_template_version=template,
                dataset_text_fingerprint=dataset_digest,
                expected_row_count=expected_rows,
                artifact_path=artifact_path,
                artifact_sha256=artifact_digest,
                cache_index_path=path,
                cache_index_sha256=index_sha256,
                cache_index_entry_position=position,
            )
            if entry.key in self._entries:
                raise CacheAuthenticationError(
                    "duplicate cache index identity across overlay manifests"
                )
            self._entries[entry.key] = entry

        self._index_identities.append(
            CacheIndexIdentity(
                path=str(path),
                sha256=index_sha256,
                byte_count=len(index_bytes),
                schema_version=LEGACY_EXTRACTION_CACHE_INDEX_SCHEMA_VERSION,
                entry_count=len(entries),
            )
        )

    def _identity_payload(self) -> dict[str, Any]:
        """Return every immutable input that can change conditional reuse."""

        entries = [
            {
                "cache_index_path": str(entry.cache_index_path),
                "cache_index_sha256": entry.cache_index_sha256,
                "cache_index_entry_position": entry.cache_index_entry_position,
                "contract": entry.contract,
                "contract_sha256": entry.contract_sha256,
                "model_identity": entry.model_identity,
                "prompt_template_version": entry.prompt_template_version,
                "dataset_text_fingerprint": entry.dataset_text_fingerprint,
                "expected_row_count": entry.expected_row_count,
                "artifact_path": str(entry.artifact_path),
                "artifact_sha256": entry.artifact_sha256,
            }
            for entry in self._entries.values()
        ]
        return json.loads(
            _canonical_json(
                {
                    "schema_version": FROZEN_EXTRACTION_CACHE_OVERLAY_IDENTITY_VERSION,
                    "expected_row_count": self.expected_row_count,
                    "row_id_column": self.row_id_column,
                    "text_column": self.text_column,
                    "cache_index_identities": [
                        asdict(identity) for identity in self._index_identities
                    ],
                    "indexed_entry_identities": entries,
                    "artifact_authentication": (
                        "exact_bytes_sha256_then_parquet_parse_from_same_bytes_on_hit"
                    ),
                    "unrequested_artifacts_opened": False,
                }
            )
        )

    def identity(self) -> Mapping[str, Any]:
        """Return a detached identity suitable for an immutable run manifest."""

        current = self._identity_payload()
        if current != self._identity:
            raise RuntimeError("frozen extraction-cache overlay state changed after binding")
        return json.loads(_canonical_json(self._identity))

    def _authenticate_entry(self, entry: _IndexEntry) -> tuple[pd.DataFrame, str, int]:
        try:
            artifact_bytes = entry.artifact_path.read_bytes()
        except FileNotFoundError as exc:
            raise CacheAuthenticationError(
                f"indexed cache artifact does not exist: {entry.artifact_path}"
            ) from exc
        except OSError as exc:
            raise CacheAuthenticationError(
                f"could not read indexed cache artifact: {entry.artifact_path}"
            ) from exc
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        if artifact_sha256 != entry.artifact_sha256:
            raise CacheAuthenticationError(
                f"indexed cache artifact was mutated: {entry.artifact_path}"
            )
        try:
            # Hash and Parquet-decode the same immutable byte buffer. A path
            # replacement between authentication and parsing cannot alter the
            # values projected into the extraction frame.
            frame = pd.read_parquet(BytesIO(artifact_bytes))
        except Exception as exc:  # pragma: no cover - engine-specific message
            raise CacheAuthenticationError(
                f"could not read indexed cache artifact: {entry.artifact_path}"
            ) from exc
        if len(frame) != self.expected_row_count:
            raise CacheAuthenticationError(
                f"cache artifact row count changed: {len(frame)} != {self.expected_row_count}"
            )
        if any(_FORBIDDEN_COLUMN.search(str(column)) for column in frame.columns):
            raise CacheAuthenticationError("cache artifact contains an oracle/true column")
        if _ROW_INDEX_COLUMN in frame.columns:
            positions = pd.to_numeric(frame[_ROW_INDEX_COLUMN], errors="coerce").to_numpy()
            expected = np.arange(self.expected_row_count, dtype=float)
            if not np.array_equal(positions, expected):
                raise CacheAuthenticationError("cache artifact row positions are not canonical")
        value_column, missing_column = expected_extraction_columns(entry.contract)
        missing = {value_column, missing_column} - set(frame.columns)
        if missing:
            raise CacheAuthenticationError(
                f"cache artifact is missing extraction columns: {sorted(missing)}"
            )
        missing_values = frame[missing_column]
        if missing_values.isna().any() or not missing_values.isin([True, False]).all():
            raise CacheAuthenticationError("cache missingness column is not boolean-valued")
        return (
            frame[[value_column, missing_column]].copy(),
            artifact_sha256,
            len(artifact_bytes),
        )

    def ensure_features(
        self,
        dataset: pd.DataFrame,
        specs: Sequence[Mapping[str, Any] | Any],
        *,
        model_identity: str,
        prompt_template_version: str,
        fallback_provider: Any,
    ) -> tuple[pd.DataFrame, CacheOverlayReport]:
        """Return requested columns, delegating only exact cache misses."""

        if len(dataset) != self.expected_row_count:
            raise ValueError(
                f"dataset row count is {len(dataset)}, expected {self.expected_row_count}"
            )
        model = str(model_identity).strip()
        template = str(prompt_template_version).strip()
        if not model or not template:
            raise ValueError("model_identity and prompt_template_version are required")
        contracts = [canonical_extraction_contract(spec) for spec in specs]
        hashes = [extraction_contract_sha256(contract) for contract in contracts]
        if len(hashes) != len(set(hashes)):
            raise ValueError("duplicate extraction contracts were requested")
        dataset_digest = ordered_dataset_text_fingerprint(
            dataset,
            row_id_column=self.row_id_column,
            text_column=self.text_column,
        )
        output = dataset.copy()
        hit_hashes: list[str] = []
        miss_hashes: list[str] = []
        miss_specs: list[Mapping[str, Any] | Any] = []
        paths: list[str] = []
        artifact_sha256s: list[str] = []
        authenticated_hits: list[AuthenticatedCacheHit] = []
        for original_spec, contract, contract_digest in zip(specs, contracts, hashes):
            key = (contract_digest, model, template, dataset_digest)
            entry = self._entries.get(key)
            if entry is None:
                miss_hashes.append(contract_digest)
                miss_specs.append(original_spec)
                continue
            if entry.contract != contract:
                raise CacheAuthenticationError("cache contract body/hash identity mismatch")
            cached, authenticated_sha256, artifact_byte_count = self._authenticate_entry(entry)
            for column in cached.columns:
                output[column] = cached[column].to_numpy(copy=True)
            hit_hashes.append(contract_digest)
            paths.append(str(entry.artifact_path))
            artifact_sha256s.append(authenticated_sha256)
            authenticated_hits.append(
                AuthenticatedCacheHit(
                    contract_sha256=contract_digest,
                    cache_index_path=str(entry.cache_index_path),
                    cache_index_sha256=entry.cache_index_sha256,
                    cache_index_entry_position=entry.cache_index_entry_position,
                    artifact_path=str(entry.artifact_path),
                    artifact_sha256=authenticated_sha256,
                    artifact_byte_count=artifact_byte_count,
                )
            )

        if miss_specs:
            if fallback_provider is None:
                raise RuntimeError("cache misses require an injected current extraction provider")
            if hasattr(fallback_provider, "ensure_features"):
                fallback = fallback_provider.ensure_features(output.copy(), list(miss_specs))
            elif callable(fallback_provider):
                fallback = fallback_provider(output.copy(), list(miss_specs))
            else:
                raise TypeError("fallback_provider must be callable or expose ensure_features")
            if not isinstance(fallback, pd.DataFrame):
                raise TypeError("fallback extraction provider must return a DataFrame")
            if len(fallback) != len(output):
                raise ValueError("fallback extraction changed the dataset row count")
            if self.row_id_column not in fallback.columns or not np.array_equal(
                fallback[self.row_id_column].to_numpy(),
                output[self.row_id_column].to_numpy(),
            ):
                raise ValueError("fallback extraction changed canonical row identity/order")
            for spec in miss_specs:
                for column in expected_extraction_columns(spec):
                    if column not in fallback.columns:
                        raise ValueError(f"fallback extraction omitted required column {column!r}")
                    if _FORBIDDEN_COLUMN.search(column):
                        raise ValueError("fallback extraction returned an oracle/true column")
                    output[column] = fallback[column].to_numpy(copy=True)

        overlay_identity = self.identity()
        report = CacheOverlayReport(
            dataset_text_fingerprint=dataset_digest,
            model_identity=model,
            prompt_template_version=template,
            cache_hit_contract_hashes=tuple(hit_hashes),
            cache_miss_contract_hashes=tuple(miss_hashes),
            authenticated_artifact_paths=tuple(paths),
            authenticated_artifact_sha256s=tuple(artifact_sha256s),
            cache_index_identities=tuple(self._index_identities),
            authenticated_cache_hits=tuple(authenticated_hits),
            overlay_identity_sha256=hashlib.sha256(
                _canonical_json(overlay_identity).encode("utf-8")
            ).hexdigest(),
        )
        return output, report
