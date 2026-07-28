"""Context-fitted, spent-only discovery evidence for adaptive review.

The post-extraction review loop is adaptive.  Consequently, neither a
full-outer discovery handoff nor a model fitted on a future review gate can be
shown to the reviewer.  This module rebuilds discovery evidence from the rows
that have already been spent and returns ordinary :class:`FoldEvidenceInput`
objects with exact spent/sealed provenance.

Two production backends are provided:

* ``HistoricalStage1SpentDiscoveryBackend`` reuses the integrated Stage-1
  BoW, HTR, matched-pair, and frozen-embedding discovery machinery.
* ``TfidfTopicOrphanSpentDiscoveryBackend`` reuses the exact-context TF-IDF
  topic fitter and the orphan n-gram safety/clustering adapter.

Only sanitized lexical concept phrases and aggregate fit-side scores leave a backend.
Patient/document identifiers, row IDs, retrieved note/chunk excerpts, raw
prediction vectors, and backend artifact paths are rejected before a fusion
request can be built.  A third backend (for example, spent-only neural query
moments) can be composed through the same closed protocol without changing the
provider.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from ..models.concept_embedding_utils import chunk_text_words
from .all_evidence_fusion import (
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
    HTR_NEURAL,
    LEGACY_ALL_SOURCE,
    MATCHED_PAIR_UPLIFT,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPICS,
    TFIDF_TOPIC_SOURCE,
    prepare_all_evidence_fusion,
    source_text_temporal_policy_audit,
)
from .embedding_contrast_discovery import EmbeddingContrastEvidenceGenerator
from .fold_honest_r_stack import FitRowProvenance
from .multi_model_agentic_forest import _build_role_grouped_evidence_digest
from .multi_model_forest_stage1 import MultiModelForestStage1Runner
from .production_stage1_scope_scheduler import derive_stage1_group_seed
from .stage1_upstream_gate_backend import (
    EFFECTIVE_STAGE1_CONFIG_ID,
    HistoricalStage1ConfigSnapshot,
    PrivateHTRModelTreeSnapshot,
    _FrozenEmbeddingGenerator,
    _effective_applied_config_sha256,
    _historical_stage1_config_snapshot,
    _module_file_sha256,
    _resolve_htr_model_path,
)
from .tfidf_orphan_evidence_adapter import (
    OrphanNgramEvidenceAdapterConfig,
    _cluster_records,
    _eligible_residual_records,
    _represented_topic_terms,
    _validate_score_frame,
)
from .tfidf_topic_discovery import fit_tfidf_topic_context
from .tfidf_upstream_gate_backend import TfidfTopicOrphanContextBackend

REVIEW_SPENT_EVIDENCE_PROVIDER_ID = "context_fit_review_spent_evidence_provider_v3"
REVIEW_SPENT_EVIDENCE_CACHE_VERSION = "context_fit_review_spent_evidence_cache_v4"
STAGE1_SPENT_DISCOVERY_BACKEND_ID = "historical_stage1_spent_discovery_v5"
TFIDF_SPENT_DISCOVERY_BACKEND_ID = "tfidf_topic_orphan_spent_discovery_v2"

ALL_NON_QUERY_DISCOVERY_FAMILIES = frozenset(
    {
        BOW_NUISANCE,
        BOW_R_LOSS,
        MATCHED_PAIR_UPLIFT,
        HTR_NEURAL,
        EMBEDDING_WHOLE_COHORT,
        EMBEDDING_CLUSTERED,
        TFIDF_TOPICS,
        TFIDF_ORPHAN_NGRAMS,
    }
)

_FORBIDDEN_KEY = re.compile(
    r"(?:^|_)(?:oracle|true|ground_truth|groundtruth)(?:_|$)", re.IGNORECASE
)
_ROW_ID_KEY = re.compile(
    r"(?:^|_)(?:row|patient|record|document|note|chunk)_(?:id|ids|index|indices)(?:_|$)",
    re.IGNORECASE,
)
_EXCERPT_KEY = re.compile(
    r"(?:^|_)(?:raw_?text|text|chunk_?text|excerpt|excerpts|snippet|snippets|"
    r"retrieved_?chunks?|retrieved_?notes?)(?:_|$)",
    re.IGNORECASE,
)
_IDENTIFIER_NOISE = re.compile(
    r"\b(?:patient\s+id|medical\s+record|record\s+number|mrn|account\s+number|"
    r"accession\s+number|claim\s+number|social\s+security|ssn|date\s+of\s+birth|"
    r"dob|patient\s+name|email\s+address|phone\s+number|street\s+address|"
    r"postal\s+code|zip\s+code)\b",
    re.IGNORECASE,
)
_PERSON_NAME_CONTEXT = re.compile(
    r"(?i:\b(?:named|name\s*[:=-])\s+[a-z][a-z'’-]+"
    r"(?:[\s,]+[a-z][a-z'’-]+){1,3}\b)|"
    r"\b(?i:patient)\s*[:=-]\s+(?:[A-Z][a-z]+|[A-Z]{3,})"
    r"(?:[\s,]+(?:[A-Z][a-z]+|[A-Z]{3,})){1,2}\b"
)

_EMAIL_URL_LONG_ID = re.compile(
    r"(?:\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b|https?://|www\.|"
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|"
    r"\b\d{7,}\b)",
    re.IGNORECASE,
)
_TERM_TOKEN = re.compile(r"[a-z0-9%<>+=-]+", re.IGNORECASE)
_PAYLOAD_TERM_KEYS = frozenset(
    {"term", "feature", "phrase", "concept", "token", "attended_token_summary"}
)

SEMANTIC_WITNESS_VECTORIZER_SCHEMA = "semantic_witness_tfidf_vectorizer_v1"
SEMANTIC_WITNESS_SCIENTIFIC_SCHEMA = "semantic_witness_scientific_config_v1"


def _closed_mapping(
    value: Any,
    *,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one configured object")
    observed = set(value)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"{label} must be closed and explicit; missing={missing}, extra={extra}"
        )
    return copy.deepcopy(dict(value))


def _document_frequency(value: Any, *, label: str, maximum: bool) -> int | float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{label} must be an integer count or floating proportion")
    if isinstance(value, (int, np.integer)):
        count = int(value)
        if count < 1:
            raise ValueError(f"{label} integer count must be positive")
        return count
    proportion = float(value)
    lower_bound = 0.0 if not maximum else 0.0
    if not math.isfinite(proportion) or not lower_bound <= proportion <= 1.0:
        raise ValueError(f"{label} floating proportion must be in [0, 1]")
    if maximum and proportion == 0.0:
        raise ValueError(f"{label} maximum proportion must be positive")
    return proportion


@dataclass(frozen=True)
class SemanticWitnessTfidfVectorizerConfig:
    """Closed TF-IDF behavior for one semantic-witness projection.

    No field has a default.  ``max_features`` is an overflow assertion, not a
    feature-selection instruction: the implementation always fits the complete
    configured vocabulary first and raises if a finite bound is exceeded.
    """

    schema_version: str
    input: str
    encoding: str
    decode_error: str
    strip_accents: str | None
    lowercase: bool
    preprocessor: None
    tokenizer: None
    analyzer: str
    stop_words: str | tuple[str, ...] | None
    token_pattern: str | None
    ngram_range_min: int
    ngram_range_max: int
    max_df: int | float
    min_df: int | float
    max_features: int | None
    vocabulary: None
    binary: bool
    dtype: str
    norm: str | None
    use_idf: bool
    smooth_idf: bool
    sublinear_tf: bool

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_WITNESS_VECTORIZER_SCHEMA:
            raise ValueError("semantic-witness vectorizer schema is unsupported")
        if self.input != "content":
            raise ValueError("semantic-witness vectorizer input must be 'content'")
        if not isinstance(self.encoding, str) or not self.encoding:
            raise ValueError("semantic-witness vectorizer encoding must be nonempty")
        if self.decode_error not in {"strict", "ignore", "replace"}:
            raise ValueError("semantic-witness vectorizer decode_error is unsupported")
        if self.strip_accents not in {None, "ascii", "unicode"}:
            raise ValueError("semantic-witness vectorizer strip_accents is unsupported")
        for name in ("lowercase", "binary", "use_idf", "smooth_idf", "sublinear_tf"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"semantic-witness vectorizer {name} must be boolean")
        if self.preprocessor is not None or self.tokenizer is not None:
            raise ValueError(
                "semantic-witness vectorizer preprocessor/tokenizer must be null"
            )
        if self.analyzer not in {"word", "char", "char_wb"}:
            raise ValueError("semantic-witness vectorizer analyzer is unsupported")
        if isinstance(self.stop_words, str):
            if self.stop_words != "english":
                raise ValueError(
                    "semantic-witness vectorizer stop_words string must be 'english'"
                )
        elif self.stop_words is not None:
            if not isinstance(self.stop_words, (list, tuple)):
                raise TypeError(
                    "semantic-witness vectorizer stop_words must be null, "
                    "'english', or an ordered list"
                )
            normalized = tuple(str(item) for item in self.stop_words)
            if (
                not normalized
                or any(not item for item in normalized)
                or len(set(normalized)) != len(normalized)
                or normalized != tuple(sorted(normalized))
            ):
                raise ValueError(
                    "semantic-witness explicit stop_words must be nonempty, "
                    "unique, and sorted"
                )
            object.__setattr__(self, "stop_words", normalized)
        if self.analyzer == "word":
            if not isinstance(self.token_pattern, str) or not self.token_pattern:
                raise ValueError(
                    "word semantic-witness vectorizer requires token_pattern"
                )
        else:
            if self.token_pattern is not None:
                raise ValueError(
                    "character semantic-witness vectorizer token_pattern must be null"
                )
            if self.stop_words is not None:
                raise ValueError(
                    "character semantic-witness vectorizer stop_words must be null"
                )
        if (
            isinstance(self.ngram_range_min, (bool, np.bool_))
            or isinstance(self.ngram_range_max, (bool, np.bool_))
            or not isinstance(self.ngram_range_min, (int, np.integer))
            or not isinstance(self.ngram_range_max, (int, np.integer))
            or int(self.ngram_range_min) < 1
            or int(self.ngram_range_max) < int(self.ngram_range_min)
        ):
            raise ValueError("semantic-witness vectorizer ngram range is invalid")
        object.__setattr__(self, "ngram_range_min", int(self.ngram_range_min))
        object.__setattr__(self, "ngram_range_max", int(self.ngram_range_max))
        object.__setattr__(
            self,
            "min_df",
            _document_frequency(self.min_df, label="min_df", maximum=False),
        )
        object.__setattr__(
            self,
            "max_df",
            _document_frequency(self.max_df, label="max_df", maximum=True),
        )
        if self.max_features is not None:
            if (
                isinstance(self.max_features, (bool, np.bool_))
                or not isinstance(self.max_features, (int, np.integer))
                or int(self.max_features) < 1
            ):
                raise ValueError(
                    "semantic-witness vectorizer max_features must be null "
                    "or a positive overflow assertion"
                )
            object.__setattr__(self, "max_features", int(self.max_features))
        if self.vocabulary is not None:
            raise ValueError(
                "semantic-witness vectorizer vocabulary must be null so the "
                "complete fit-scope vocabulary is learned"
            )
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("semantic-witness vectorizer dtype must be float32 or float64")
        if self.norm not in {None, "l1", "l2"}:
            raise ValueError("semantic-witness vectorizer norm is unsupported")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        label: str = "semantic-witness vectorizer",
    ) -> "SemanticWitnessTfidfVectorizerConfig":
        values = _closed_mapping(
            value,
            expected=frozenset(item.name for item in fields(cls)),
            label=label,
        )
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if isinstance(self.stop_words, tuple):
            value["stop_words"] = list(self.stop_words)
        return value

    @property
    def identity_sha256(self) -> str:
        return _sha256_json(self.as_dict())


@dataclass(frozen=True)
class SemanticWitnessScientificConfig:
    """All scientific choices used by safe embedding/HTR term projection."""

    schema_version: str
    retrieval_vectorizer: SemanticWitnessTfidfVectorizerConfig
    htr_vectorizer: SemanticWitnessTfidfVectorizerConfig
    retrieval_min_positive_documents: int
    retrieval_min_negative_documents: int
    htr_min_unique_sources: int
    htr_min_distinct_positive_documents: int
    htr_min_positive_source_support: int
    htr_attention_score_min_exclusive: float
    htr_direction_score_min_exclusive: float
    htr_require_strict_attention_separation: bool
    retrieval_document_weighting_policy: str
    htr_source_weighting_policy: str
    htr_extreme_chunk_tie_policy: str
    retrieval_ranking_policy: str
    htr_ranking_policy: str
    phrase_collision_policy: str
    htr_term_overlap_policy: str
    retrieval_score_eligibility_policy: str
    maximum_retrieval_terms: int | None
    maximum_htr_terms: int | None
    maximum_explicit_phrases_per_attention_row: int | None
    overflow_policy: str
    insufficient_source_policy: str
    empty_vocabulary_policy: str
    direction_numeric_dtype: str

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_WITNESS_SCIENTIFIC_SCHEMA:
            raise ValueError("semantic-witness scientific schema is unsupported")
        for name in ("retrieval_vectorizer", "htr_vectorizer"):
            if type(getattr(self, name)) is not SemanticWitnessTfidfVectorizerConfig:
                raise TypeError(f"{name} must be a closed semantic-witness vectorizer")
        for name in (
            "retrieval_min_positive_documents",
            "retrieval_min_negative_documents",
            "htr_min_unique_sources",
            "htr_min_distinct_positive_documents",
            "htr_min_positive_source_support",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        for name in (
            "htr_attention_score_min_exclusive",
            "htr_direction_score_min_exclusive",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if type(self.htr_require_strict_attention_separation) is not bool:
            raise TypeError("htr_require_strict_attention_separation must be boolean")
        expected_policies = {
            "retrieval_document_weighting_policy": "unweighted_document_mean_v1",
            "htr_source_weighting_policy": (
                "equal_source_mass_inverse_repeated_partition_count_v1"
            ),
            "htr_extreme_chunk_tie_policy": (
                "attention_then_chunk_index_then_casefolded_text_v1"
            ),
            "retrieval_ranking_policy": "absolute_score_desc_then_term_asc_v1",
            "htr_ranking_policy": (
                "score_desc_then_token_count_desc_then_term_asc_v1"
            ),
            "phrase_collision_policy": "highest_ranked_normalized_phrase_v1",
            "htr_term_overlap_policy": "retain_all_eligible_terms_v1",
            "retrieval_score_eligibility_policy": "all_finite_including_zero_v1",
            "overflow_policy": "fail_closed_without_selection_v1",
            "insufficient_source_policy": "return_empty_evidence_v1",
            "empty_vocabulary_policy": "return_empty_evidence_v1",
        }
        for name, expected in expected_policies.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must equal {expected!r}")
        for name in (
            "maximum_retrieval_terms",
            "maximum_htr_terms",
            "maximum_explicit_phrases_per_attention_row",
        ):
            value = getattr(self, name)
            if value is not None:
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or int(value) < 1
                ):
                    raise ValueError(f"{name} must be null or a positive overflow assertion")
                object.__setattr__(self, name, int(value))
        if self.direction_numeric_dtype not in {"float32", "float64"}:
            raise ValueError("direction_numeric_dtype must be float32 or float64")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        label: str = "semantic-witness scientific config",
    ) -> "SemanticWitnessScientificConfig":
        values = _closed_mapping(
            value,
            expected=frozenset(item.name for item in fields(cls)),
            label=label,
        )
        values["retrieval_vectorizer"] = (
            SemanticWitnessTfidfVectorizerConfig.from_mapping(
                values["retrieval_vectorizer"],
                label=f"{label}.retrieval_vectorizer",
            )
        )
        values["htr_vectorizer"] = SemanticWitnessTfidfVectorizerConfig.from_mapping(
            values["htr_vectorizer"],
            label=f"{label}.htr_vectorizer",
        )
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["retrieval_vectorizer"] = self.retrieval_vectorizer.as_dict()
        value["htr_vectorizer"] = self.htr_vectorizer.as_dict()
        return value

    @property
    def identity_sha256(self) -> str:
        return _sha256_json(self.as_dict())


def semantic_witness_config_from_portable_scientific_spec(
    value: Mapping[str, Any],
) -> SemanticWitnessScientificConfig:
    """Load the closed witness config from the authenticated portable profile."""

    if not isinstance(value, Mapping):
        raise TypeError("portable scientific identity must be one mapping")
    profiles = value.get("architecture_profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("portable scientific identity has no architecture_profiles")
    profile = profiles.get("lexical_semantic_retrieval")
    if (
        not isinstance(profile, Mapping)
        or profile.get("enabled") is not True
        or profile.get("shared_physical_producer") != "whole_cohort_embeddings"
    ):
        raise ValueError(
            "lexical semantic retrieval must be enabled and bind the "
            "whole-cohort physical producer"
        )
    return SemanticWitnessScientificConfig.from_mapping(
        profile.get("producer_configuration"),
        label=(
            "architecture_profiles.lexical_semantic_retrieval."
            "producer_configuration"
        ),
    )


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _snapshot_cache_file(
    path: Path,
) -> tuple[BinaryIO, str, tuple[int, int, int, int, int], int]:
    """Stream one canonical file into a retained anonymous private snapshot."""

    before = _stat_signature(path)
    snapshot = tempfile.TemporaryFile(mode="w+b")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                snapshot.write(chunk)
                size += len(chunk)
        snapshot.flush()
        snapshot.seek(0)
        after = _stat_signature(path)
        if before != after:
            raise RuntimeError(f"cache file changed while it was being authenticated: {path.name}")
    except Exception:
        snapshot.close()
        raise
    return snapshot, digest.hexdigest(), after, size


def _load_private_snapshot_npy(
    snapshot: BinaryIO,
    *,
    name: str,
    mmap: bool,
) -> np.ndarray:
    """Load an NPY only through its private snapshot descriptor."""

    snapshot_path = f"/proc/self/fd/{snapshot.fileno()}"
    try:
        loaded = np.load(
            snapshot_path,
            mmap_mode="r" if mmap else None,
            allow_pickle=False,
        )
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"frozen embedding cache contains an invalid {name} array") from exc
    if not isinstance(loaded, np.ndarray):
        raise ValueError(f"frozen embedding cache {name} must be one NumPy array")
    if mmap:
        loaded.setflags(write=False)
        return loaded
    output = np.array(loaded, copy=True)
    output.setflags(write=False)
    return output


def _snapshot_line_spans(
    snapshot: BinaryIO,
    *,
    size: int,
) -> tuple[tuple[int, int], ...]:
    """Index JSONL byte ranges without decoding or retaining row text."""

    spans: list[tuple[int, int]] = []
    line_start = 0
    cursor = 0
    fd = snapshot.fileno()
    while cursor < int(size):
        block = os.pread(fd, min(1024 * 1024, int(size) - cursor), cursor)
        if not block:
            raise RuntimeError("private chunk-text snapshot ended unexpectedly")
        search_from = 0
        while True:
            newline = block.find(b"\n", search_from)
            if newline < 0:
                break
            line_stop = cursor + newline + 1
            spans.append((line_start, line_stop))
            line_start = line_stop
            search_from = newline + 1
        cursor += len(block)
    if line_start < int(size):
        spans.append((line_start, int(size)))
    return tuple(spans)


def _module_sha256() -> str:
    return _sha256_file(Path(__file__))


def _json_value(value: Any, *, path: str) -> Any:
    """Return closed finite JSON and reject benchmark metadata."""

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key).strip()
            if not key or _FORBIDDEN_KEY.search(key):
                raise ValueError(f"{path} contains a forbidden or empty field")
            if key in output:
                raise ValueError(f"{path} contains colliding fields")
            output[key] = _json_value(child, path=f"{path}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [_json_value(child, path=f"{path}[]") for child in value]
    if isinstance(value, np.generic):
        return _json_value(value.item(), path=path)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, str):
        return value
    raise TypeError(f"{path} contains a non-JSON value")


def _integer_rows(
    values: Sequence[Any], *, name: str, allow_empty: bool = False
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of canonical row IDs")
    output: list[int] = []
    for value in tuple(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must contain integer row IDs")
        row_id = int(value)
        if row_id < 0:
            raise ValueError(f"{name} contains a negative row ID")
        output.append(row_id)
    if (not allow_empty and not output) or len(output) != len(set(output)):
        raise ValueError(
            f"{name} must be unique and {'possibly empty' if allow_empty else 'non-empty'}"
        )
    return tuple(output)


def _exact_texts(values: Sequence[Any], *, rows: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError("spent_texts must be a sequence")
    output = tuple(values)
    if len(output) != int(rows) or not all(isinstance(value, str) for value in output):
        raise ValueError("spent_texts must contain one exact string per spent row")
    return output


def _finite_vector(values: Any, *, name: str, rows: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) != int(rows) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector with length {rows}")
    output = vector.copy()
    output.setflags(write=False)
    return output


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _safe_concept_phrase(
    value: Any,
    *,
    max_tokens: int | None = None,
    max_chars: int | None = None,
) -> str:
    """Normalize a lexical phrase without an implicit size cutoff.

    Optional finite capacities are caller-owned assertions. They are not
    production defaults and never replace the complete term.
    """

    for name, capacity in (
        ("max_tokens", max_tokens),
        ("max_chars", max_chars),
    ):
        if capacity is not None and (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < 1
        ):
            raise ValueError(f"{name} must be null or a positive integer")
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n;,:|/\\")
    if max_chars is not None and len(text) > max_chars:
        raise ValueError(
            "safe concept phrase exceeds configured max_chars; refusing "
            "silent semantic omission"
        )
    if not text:
        return ""
    # Vectorizer terms can contain identifier-like phrases joined by
    # underscores or punctuation.  Probe a separator-normalized form before
    # returning the ordinary lexical normalization so ``patient_id`` receives
    # the same privacy treatment as ``patient id``.
    privacy_probe = re.sub(r"[\W_]+", " ", text).strip()
    if _IDENTIFIER_NOISE.search(privacy_probe):
        return ""
    if _EMAIL_URL_LONG_ID.search(text):
        return ""
    tokens = _TERM_TOKEN.findall(text)
    if max_tokens is not None and len(tokens) > max_tokens:
        raise ValueError(
            "safe concept phrase exceeds configured max_tokens; refusing "
            "silent semantic omission"
        )
    if not tokens:
        return ""
    if all(token.isdigit() for token in tokens):
        return ""
    # Concept phrases are lexical features, not sentence-like excerpts.
    if re.search(r"[.!?]", text):
        return ""
    return " ".join(tokens).lower()


def _validate_reviewer_payload(value: Any, *, path: str = "payload") -> None:
    """Fail closed on row identity, excerpts, vectors, and unsafe concepts."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.casefold()
            if _FORBIDDEN_KEY.search(key):
                raise ValueError(f"forbidden benchmark field at {path}.{key}")
            if _ROW_ID_KEY.search(key):
                raise ValueError(f"row-level identity entered reviewer payload at {path}.{key}")
            if _EXCERPT_KEY.search(key):
                raise ValueError(f"raw text/excerpt entered reviewer payload at {path}.{key}")
            if lowered in _PAYLOAD_TERM_KEYS:
                phrase = _safe_concept_phrase(child)
                if not phrase or phrase != str(child):
                    raise ValueError(f"unsafe or non-normalized concept phrase at {path}.{key}")
            _validate_reviewer_payload(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        # Anonymous numeric vectors are never reviewer evidence.
        if value and all(
            isinstance(item, (int, float, np.integer, np.floating))
            and not isinstance(item, (bool, np.bool_))
            for item in value
        ):
            raise ValueError(f"anonymous numeric vector entered reviewer payload at {path}")
        for index, child in enumerate(value):
            _validate_reviewer_payload(child, path=f"{path}[{index}]")


@dataclass(frozen=True)
class SpentDiscoveryEvidence:
    """One backend payload plus its exact recursive target-fit lineage."""

    source_kind: str
    _payload_json: str = field(repr=False)
    fit_row_provenance: FitRowProvenance = field(repr=False)

    @classmethod
    def create(
        cls,
        *,
        source_kind: str,
        payload: Mapping[str, Any],
        fit_row_provenance: FitRowProvenance,
    ) -> "SpentDiscoveryEvidence":
        if not isinstance(payload, Mapping):
            raise TypeError("spent discovery payload must be a mapping")
        if not isinstance(fit_row_provenance, FitRowProvenance):
            raise TypeError("spent discovery lineage must be FitRowProvenance")
        closed = _json_value(payload, path="payload")
        _validate_reviewer_payload(closed)
        return cls(
            source_kind=str(source_kind).strip().lower(),
            _payload_json=_canonical_json(closed),
            fit_row_provenance=fit_row_provenance,
        )

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._payload_json)


class SpentDiscoveryBackend(Protocol):
    """Closed extension point for Stage-1, TF-IDF, or neural-query discovery."""

    def identity(self) -> Mapping[str, Any]: ...

    def fit_discovery(
        self,
        *,
        outer_fold: int,
        review_round: int,
        exact_spent_row_ids: tuple[int, ...],
        spent_texts: tuple[str, ...],
        spent_treatment: np.ndarray,
        spent_outcome: np.ndarray,
        work_dir: Path,
    ) -> SpentDiscoveryEvidence: ...


class ContextFitReviewSpentEvidenceProvider:
    """Build and cache exact-spent evidence while future gates stay sealed."""

    def __init__(
        self,
        *,
        backends: Sequence[SpentDiscoveryBackend],
        cache_dir: Path | str,
        required_source_families: Sequence[str] = tuple(sorted(ALL_NON_QUERY_DISCOVERY_FAMILIES)),
    ) -> None:
        self.backends = tuple(backends)
        if not self.backends:
            raise ValueError("at least one spent discovery backend is required")
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.required_source_families = frozenset(
            str(value).strip() for value in required_source_families if str(value).strip()
        )
        identities: list[dict[str, Any]] = []
        for index, backend in enumerate(self.backends):
            identity = _json_value(backend.identity(), path=f"backend[{index}].identity")
            identities.append(identity)
        if len({_canonical_json(value) for value in identities}) != len(identities):
            raise ValueError("spent discovery backend identities must be unique")
        self._backend_identities = tuple(identities)
        self._identity = {
            "provider": REVIEW_SPENT_EVIDENCE_PROVIDER_ID,
            "cache_schema_version": REVIEW_SPENT_EVIDENCE_CACHE_VERSION,
            "provider_code_sha256": _module_sha256(),
            "backends": list(self._backend_identities),
            "required_source_families": sorted(self.required_source_families),
            "neural_query_extension_supported": True,
            "future_gate_text_or_labels_accepted": False,
            "reviewer_excerpts_allowed": False,
            "source_text_temporal_policy": source_text_temporal_policy_audit(),
        }

    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._identity)

    def _current_backend_identities(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _json_value(backend.identity(), path=f"backend[{index}].identity")
            for index, backend in enumerate(self.backends)
        )

    def _binding(
        self,
        *,
        outer_fold: int,
        review_round: int,
        spent_ids: tuple[int, ...],
        sealed_ids: tuple[int, ...],
        spent_texts: tuple[str, ...],
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "schema_version": REVIEW_SPENT_EVIDENCE_CACHE_VERSION,
            "outer_fold": int(outer_fold),
            "review_round": int(review_round),
            "spent_row_ids_sha256": _sha256_json(list(spent_ids)),
            "sealed_row_ids_sha256": _sha256_json(list(sealed_ids)),
            "ordered_spent_text_sha256": _sha256_json(list(spent_texts)),
            "spent_treatment_sha256": _array_digest(treatment),
            "spent_outcome_sha256": _array_digest(outcome),
            "backend_identities_sha256": _sha256_json(list(self._backend_identities)),
            "provider_identity_sha256": _sha256_json(self._identity),
        }

    def _load_cache(
        self,
        path: Path,
        *,
        cache_key: str,
        binding: Mapping[str, Any],
        exact_spent_row_ids: tuple[int, ...],
    ) -> tuple[SpentDiscoveryEvidence, ...] | None:
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        expected_fields = {"schema_version", "cache_key", "binding", "results", "content_sha256"}
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ValueError("spent evidence cache manifest has an unsupported schema")
        content = {key: raw[key] for key in raw if key != "content_sha256"}
        if raw["content_sha256"] != _sha256_json(content):
            raise ValueError("spent evidence cache manifest content hash mismatch")
        if (
            raw["schema_version"] != REVIEW_SPENT_EVIDENCE_CACHE_VERSION
            or raw["cache_key"] != cache_key
            or raw["binding"] != binding
        ):
            raise ValueError("spent evidence cache binding mismatch")
        results = raw["results"]
        if not isinstance(results, list) or len(results) != len(self.backends):
            raise ValueError("spent evidence cache has an invalid backend result count")
        lineage = FitRowProvenance(fit_row_ids=frozenset(exact_spent_row_ids))
        output = tuple(
            SpentDiscoveryEvidence.create(
                source_kind=str(row["source_kind"]),
                payload=row["payload"],
                fit_row_provenance=lineage,
            )
            for row in results
        )
        return output

    def _write_cache(
        self,
        path: Path,
        *,
        cache_key: str,
        binding: Mapping[str, Any],
        results: Sequence[SpentDiscoveryEvidence],
    ) -> None:
        body: dict[str, Any] = {
            "schema_version": REVIEW_SPENT_EVIDENCE_CACHE_VERSION,
            "cache_key": cache_key,
            "binding": dict(binding),
            "results": [
                {"source_kind": row.source_kind, "payload": row.payload} for row in results
            ],
        }
        body["content_sha256"] = _sha256_json(body)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(_canonical_json(body))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

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
    ) -> Sequence[FoldEvidenceInput]:
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
        spent_ids = _integer_rows(exact_spent_row_ids, name="exact_spent_row_ids")
        sealed_ids = _integer_rows(exact_sealed_row_ids, name="exact_sealed_row_ids")
        if set(spent_ids) & set(sealed_ids):
            raise ValueError("spent and sealed review rows overlap")
        texts = _exact_texts(spent_texts, rows=len(spent_ids))
        treatment = _finite_vector(spent_treatment, name="spent_treatment", rows=len(spent_ids))
        outcome = _finite_vector(spent_outcome, name="spent_outcome", rows=len(spent_ids))
        if not set(np.unique(treatment)).issubset({0.0, 1.0}):
            raise ValueError("spent_treatment must be binary")
        current_identities = self._current_backend_identities()
        if current_identities != self._backend_identities:
            raise RuntimeError("a spent discovery backend identity changed during the run")

        binding = self._binding(
            outer_fold=int(outer_fold),
            review_round=int(review_round),
            spent_ids=spent_ids,
            sealed_ids=sealed_ids,
            spent_texts=texts,
            treatment=treatment,
            outcome=outcome,
        )
        cache_key = _sha256_json(binding)
        cache_path = self.cache_dir / f"{cache_key}.json"
        results = self._load_cache(
            cache_path,
            cache_key=cache_key,
            binding=binding,
            exact_spent_row_ids=spent_ids,
        )
        if results is None:
            before_treatment = _array_digest(treatment)
            before_outcome = _array_digest(outcome)
            fitted: list[SpentDiscoveryEvidence] = []
            with tempfile.TemporaryDirectory(
                prefix=f"review_spent_{int(outer_fold):03d}_{int(review_round):02d}_",
                dir=self.cache_dir,
            ) as raw_work_dir:
                work_root = Path(raw_work_dir)
                for index, backend in enumerate(self.backends):
                    result = backend.fit_discovery(
                        outer_fold=int(outer_fold),
                        review_round=int(review_round),
                        exact_spent_row_ids=spent_ids,
                        spent_texts=texts,
                        spent_treatment=treatment,
                        spent_outcome=outcome,
                        work_dir=work_root / f"backend_{index:02d}",
                    )
                    if not isinstance(result, SpentDiscoveryEvidence):
                        raise TypeError("spent discovery backend returned an invalid result")
                    direct = result.fit_row_provenance.fit_row_ids
                    recursive = result.fit_row_provenance.recursive_fit_row_ids()
                    expected = frozenset(spent_ids)
                    if direct != expected or recursive != expected:
                        raise ValueError(
                            "spent discovery backend did not declare exact-spent FitRowProvenance"
                        )
                    fitted.append(result)
            if (
                _array_digest(treatment) != before_treatment
                or _array_digest(outcome) != before_outcome
            ):
                raise RuntimeError("spent discovery backend mutated its read-only input labels")
            results = tuple(fitted)
            self._write_cache(
                cache_path,
                cache_key=cache_key,
                binding=binding,
                results=results,
            )

        source_kinds = [row.source_kind for row in results]
        if len(source_kinds) != len(set(source_kinds)):
            raise ValueError("spent discovery backends returned duplicate source kinds")
        provenance = FoldEvidenceProvenance(
            outer_fold=int(outer_fold),
            train_row_ids=spent_ids,
            heldout_row_ids=sealed_ids,
            scope="inner_train",
            inner_fold=int(review_round) + 1,
            artifact_id=f"review-spent-{cache_key}",
        )
        inputs = tuple(
            FoldEvidenceInput(
                source_kind=row.source_kind,
                payload=row.payload,
                provenance=provenance,
            )
            for row in results
        )
        request = prepare_all_evidence_fusion(inputs)
        present = set(request.source_family_coverage["present_source_families"])
        missing = sorted(self.required_source_families - present)
        if missing:
            raise RuntimeError(
                "spent-only discovery omitted required source families: " + ", ".join(missing)
            )
        return inputs


class SpentOnlyFrozenChunkEmbeddingCache:
    """Authenticate a global cache while decoding only exact spent rows."""

    _FILES = (
        "metadata.json",
        "chunk_embeddings.npy",
        "offsets.npy",
        "chunk_texts.jsonl",
    )

    def __init__(self, cache_dir: Path | str) -> None:
        supplied_root = Path(cache_dir)
        if supplied_root.is_symlink():
            raise ValueError("frozen embedding cache root cannot be a symlink")
        if not supplied_root.is_dir():
            raise FileNotFoundError(f"frozen embedding cache does not exist: {supplied_root}")
        linked = [name for name in self._FILES if (supplied_root / name).is_symlink()]
        if linked:
            raise ValueError(f"frozen embedding cache files cannot be symlinks: {linked}")
        missing = [name for name in self._FILES if not (supplied_root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"frozen embedding cache is incomplete: {missing}")
        self.cache_dir = supplied_root.resolve(strict=True)
        snapshots: dict[str, BinaryIO] = {}
        snapshot_digests: dict[str, str] = {}
        snapshot_sizes: dict[str, int] = {}
        file_stats: dict[str, tuple[int, int, int, int, int]] = {}
        for filename in self._FILES:
            (
                snapshots[filename],
                snapshot_digests[filename],
                file_stats[filename],
                snapshot_sizes[filename],
            ) = _snapshot_cache_file(self.cache_dir / filename)
        # A path read early must still name exactly the snapshotted bytes after
        # all four private snapshots have completed.
        for filename in self._FILES:
            path = self.cache_dir / filename
            before = _stat_signature(path)
            digest = _sha256_file(path)
            after = _stat_signature(path)
            if (
                before != file_stats[filename]
                or after != before
                or digest != snapshot_digests[filename]
            ):
                raise RuntimeError(
                    f"cache file changed while it was being authenticated: {filename}"
                )
        self._snapshot_files = snapshots
        metadata_snapshot = snapshots["metadata.json"]
        metadata_bytes = os.pread(metadata_snapshot.fileno(), snapshot_sizes["metadata.json"], 0)
        if len(metadata_bytes) != snapshot_sizes["metadata.json"]:
            raise RuntimeError("private metadata snapshot ended unexpectedly")
        try:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("frozen embedding cache metadata is invalid JSON") from exc
        if not isinstance(metadata, dict):
            raise ValueError("frozen embedding cache metadata must be an object")
        self._metadata = metadata
        self._embeddings = _load_private_snapshot_npy(
            snapshots["chunk_embeddings.npy"],
            name="chunk_embeddings",
            mmap=True,
        )
        self._offsets = _load_private_snapshot_npy(
            snapshots["offsets.npy"],
            name="offsets",
            mmap=False,
        )
        row_count = int(self._metadata.get("num_samples", -1))
        hidden_size = int(self._metadata.get("hidden_size", -1))
        if row_count < 1 or self._embeddings.ndim != 2:
            raise ValueError("frozen embedding cache has invalid metadata or matrix rank")
        if self._offsets.ndim != 1 or len(self._offsets) != row_count + 1:
            raise ValueError("frozen embedding offsets do not match cache row count")
        if int(self._offsets[-1]) != int(self._embeddings.shape[0]):
            raise ValueError("frozen embedding offsets do not span the chunk matrix")
        if hidden_size != int(self._embeddings.shape[1]):
            raise ValueError("frozen embedding hidden size is inconsistent")

        # Keep only a private descriptor and integer line spans. No future-row
        # JSON/text bytes remain resident in a Python object after this scan.
        self._chunk_text_snapshot = snapshots["chunk_texts.jsonl"]
        self._line_spans = _snapshot_line_spans(
            self._chunk_text_snapshot,
            size=snapshot_sizes["chunk_texts.jsonl"],
        )
        if len(self._line_spans) != row_count:
            raise ValueError("frozen chunk-text registry does not match cache row count")
        self._identity = {
            "provider": "spent_only_frozen_chunk_embedding_cache_v2",
            "metadata_sha256": snapshot_digests["metadata.json"],
            "embeddings_sha256": snapshot_digests["chunk_embeddings.npy"],
            "offsets_sha256": snapshot_digests["offsets.npy"],
            "chunk_texts_sha256": snapshot_digests["chunk_texts.jsonl"],
            "row_count": row_count,
            "chunk_count": int(self._embeddings.shape[0]),
            "cache_snapshot_authentication": "streamed_private_fd_sha256_v1",
            "chunk_text_storage": "private_fd_pread_lazy_row_decode_v1",
            "embeddings_path_backed": False,
            "private_snapshot_embedding_mmap": True,
            "future_row_text_decoded": False,
            "novel_text_encoding_allowed": False,
        }
        self._file_stats = file_stats

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return metadata detached from the authenticated internal contract."""

        return copy.deepcopy(self._metadata)

    @property
    def row_count(self) -> int:
        return int(self._metadata["num_samples"])

    def authenticated_snapshot_identity(self) -> Mapping[str, Any]:
        """Return the identity of this already-authenticated private snapshot.

        Construction opens private file descriptors, hashes every registered
        byte, and verifies the path/stat inventory after all snapshots exist.
        Consumers holding this nonserializable handle may therefore reuse its
        identity without replaying the same multi-gigabyte cache merely to
        recover the digest that was just authenticated.
        """

        return copy.deepcopy(self._identity)

    def identity(self) -> Mapping[str, Any]:
        hash_fields = {
            "metadata.json": "metadata_sha256",
            "chunk_embeddings.npy": "embeddings_sha256",
            "offsets.npy": "offsets_sha256",
            "chunk_texts.jsonl": "chunk_texts_sha256",
        }
        for filename, field_name in hash_fields.items():
            path = self.cache_dir / filename
            try:
                before = _stat_signature(path)
                digest = _sha256_file(path)
                after = _stat_signature(path)
            except OSError as exc:
                raise RuntimeError(
                    "frozen embedding cache bytes changed or path changed during the run: "
                    f"{filename}"
                ) from exc
            if (
                before != self._file_stats[filename]
                or after != before
                or digest != self._identity[field_name]
            ):
                raise RuntimeError(
                    "frozen embedding cache bytes changed or path changed during the run: "
                    f"{filename}"
                )
        return copy.deepcopy(self._identity)

    def _cached_chunks(self, row_id: int) -> tuple[str, ...]:
        if not 0 <= int(row_id) < self.row_count:
            raise ValueError("spent row lies outside the frozen embedding cache")
        start, stop = self._line_spans[int(row_id)]
        line = os.pread(self._chunk_text_snapshot.fileno(), stop - start, start)
        if len(line) != stop - start:
            raise RuntimeError("private chunk-text snapshot ended unexpectedly")
        try:
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("spent frozen chunk-text row is invalid JSON") from exc
        chunks = payload.get("chunks")
        if not isinstance(chunks, list) or not all(isinstance(value, str) for value in chunks):
            raise ValueError("spent frozen chunk-text row has an invalid schema")
        expected_count = int(self._offsets[int(row_id) + 1]) - int(self._offsets[int(row_id)])
        if len(chunks) != expected_count:
            raise ValueError("spent frozen chunk text does not align with embeddings")
        return tuple(chunks)

    def bind_spent(
        self,
        row_ids: tuple[int, ...],
        texts: tuple[str, ...],
    ) -> "BoundSpentFrozenChunkEmbeddingProvider":
        row_ids = _integer_rows(row_ids, name="bind_spent.row_ids")
        texts = _exact_texts(texts, rows=len(row_ids))
        cached_by_row: dict[int, tuple[str, ...]] = {}
        token_bounded: list[int] = []
        chunk_size = int(self._metadata["chunk_size_words"])
        chunk_overlap = int(self._metadata["chunk_overlap_words"])
        max_chunks = int(self._metadata["max_chunks"])
        selection = str(self._metadata.get("chunk_selection") or "last")
        for row_id, text in zip(row_ids, texts):
            cached = self._cached_chunks(int(row_id))
            generated = tuple(
                chunk_text_words(
                    str(text),
                    chunk_size,
                    chunk_overlap,
                    max_chunks,
                    selection,
                )
            )
            if cached != generated:
                # Preserve the narrowly audited tokenizer-bound exception used
                # by the original cache adapter, but only on a supplied row.
                if not generated or not cached or cached[-1] != generated[-1]:
                    raise ValueError("spent text does not match its frozen embedding cache row")
                first_real = next(
                    (
                        chunk.replace("\u00ad", "").strip()
                        for chunk in cached
                        if chunk.replace("\u00ad", "").strip()
                    ),
                    "",
                )
                normalized_source = " ".join(str(text).split())
                normalized_first = " ".join(first_real.split())
                if not normalized_first or normalized_first not in normalized_source:
                    raise ValueError(
                        "token-bounded spent cache row cannot be bound to supplied text"
                    )
                token_bounded.append(int(row_id))
            cached_by_row[int(row_id)] = cached
        if len(token_bounded) > max(4, int(math.ceil(0.01 * len(row_ids)))):
            raise ValueError("too many token-bounded rows in spent embedding context")
        return BoundSpentFrozenChunkEmbeddingProvider(
            cache=self,
            row_ids=row_ids,
            cached_by_row=cached_by_row,
            token_bounded_row_ids=tuple(token_bounded),
        )


class BoundSpentFrozenChunkEmbeddingProvider:
    """Narrow cache view containing semantic text for spent rows only."""

    def __init__(
        self,
        *,
        cache: SpentOnlyFrozenChunkEmbeddingCache,
        row_ids: tuple[int, ...],
        cached_by_row: Mapping[int, tuple[str, ...]],
        token_bounded_row_ids: tuple[int, ...],
    ) -> None:
        self._cache = cache
        self.metadata = cache.metadata
        self._embeddings = cache._embeddings
        self._offsets = cache._offsets
        self.row_ids = tuple(row_ids)
        self.cached_by_row = dict(cached_by_row)
        self.token_bounded_row_ids = tuple(token_bounded_row_ids)

    def identity(self) -> Mapping[str, Any]:
        return {
            "cache": self._cache.identity(),
            "spent_row_ids_sha256": _sha256_json(list(self.row_ids)),
            "token_bounded_row_ids_sha256": _sha256_json(list(self.token_bounded_row_ids)),
        }

    def chunk_matrix(self, row_id: int) -> np.ndarray:
        if isinstance(row_id, (bool, np.bool_)) or not isinstance(row_id, (int, np.integer)):
            raise TypeError("chunk_matrix.row_id must be an integer")
        row_id = int(row_id)
        if row_id not in self.cached_by_row:
            raise ValueError("embedding provider refuses a non-spent row")
        start = int(self._offsets[row_id])
        stop = int(self._offsets[row_id + 1])
        return np.array(self._embeddings[start:stop], dtype=np.float32, copy=True)

    def chunk_matrices(self, row_ids: Sequence[int]) -> tuple[np.ndarray, ...]:
        """Return exact ordered chunk matrices, refusing every unbound row."""

        requested = _integer_rows(row_ids, name="chunk_matrices.row_ids", allow_empty=True)
        return tuple(self.chunk_matrix(row_id) for row_id in requested)

    def chunk_texts(self, row_ids: Sequence[int]) -> tuple[tuple[str, ...], ...]:
        """Return exact ordered cached chunks, refusing every unbound row."""

        requested = _integer_rows(row_ids, name="chunk_texts.row_ids", allow_empty=True)
        missing = [row_id for row_id in requested if row_id not in self.cached_by_row]
        if missing:
            raise ValueError("embedding provider refuses a non-spent row")
        return tuple(self.cached_by_row[row_id] for row_id in requested)


class _FrozenCacheEmbeddingEvidenceGenerator(EmbeddingContrastEvidenceGenerator):
    """Use the authenticated chunk cache without loading an embedding model."""

    def __init__(
        self,
        *,
        config: Any,
        embedding_provider: BoundSpentFrozenChunkEmbeddingProvider,
        dataset_row_count: int,
        output_dir: Path,
    ) -> None:
        config = copy.deepcopy(config)
        config.architecture.multi_model_agentic_forest.embedding_contrast = copy.deepcopy(
            config.architecture.multi_model_forest.embedding_contrast
        )
        embedding_config = config.architecture.multi_model_agentic_forest.embedding_contrast
        embedding_config.include_bow_phrases_as_concepts = False
        embedding_config.concept_phrases = []
        embedding_config.external_corpus_cache_dirs = []
        super().__init__(config=config, output_dir=output_dir, embedding_provider=None)
        self._spent_provider = embedding_provider
        self._dataset_row_count = int(dataset_row_count)

    def prepare(self, dataset: pd.DataFrame) -> None:
        if self._prepared:
            return
        provider = self._spent_provider
        if len(dataset) != self._dataset_row_count:
            raise ValueError("embedding evidence dataset does not match frozen cache")
        self._row_ids = dataset["_oci_row_id"].astype(int).tolist()
        self._row_id_to_position = {
            int(row_id): index for index, row_id in enumerate(self._row_ids)
        }
        self._chunks_by_position = [[] for _ in range(self._dataset_row_count)]
        for row_id, chunks in provider.cached_by_row.items():
            self._chunks_by_position[int(row_id)] = list(chunks)
        # Preserve the authenticated private-snapshot read-only mmap. The
        # inherited row/chunk accessors cast only requested slices to float32.
        self._flat_embeddings = provider._embeddings
        self._offsets = np.array(provider._offsets, dtype=np.int64, copy=True)
        self._offsets.setflags(write=False)
        self._external_corpora = []
        self._prepared = True

    def _concept_phrases(self, _importance: Mapping[str, Any]) -> list[str]:
        return []

    def _encode_concepts(self, _phrases: Sequence[str]) -> np.ndarray:
        raise RuntimeError("frozen spent discovery never encodes novel concept text")


def _semantic_vectorizer(
    config: SemanticWitnessTfidfVectorizerConfig,
) -> TfidfVectorizer:
    if type(config) is not SemanticWitnessTfidfVectorizerConfig:
        raise TypeError("semantic witness projection requires a typed vectorizer config")
    return TfidfVectorizer(
        input=config.input,
        encoding=config.encoding,
        decode_error=config.decode_error,
        strip_accents=config.strip_accents,
        lowercase=config.lowercase,
        preprocessor=config.preprocessor,
        tokenizer=config.tokenizer,
        analyzer=config.analyzer,
        stop_words=(
            list(config.stop_words)
            if isinstance(config.stop_words, tuple)
            else config.stop_words
        ),
        token_pattern=config.token_pattern,
        ngram_range=(config.ngram_range_min, config.ngram_range_max),
        max_df=config.max_df,
        min_df=config.min_df,
        # A finite configured value is checked after exhaustive fitting.  It
        # never authorizes sklearn to select a vocabulary prefix.
        max_features=None,
        vocabulary=None,
        binary=config.binary,
        dtype=np.float32 if config.dtype == "float32" else np.float64,
        norm=config.norm,
        use_idf=config.use_idf,
        smooth_idf=config.smooth_idf,
        sublinear_tf=config.sublinear_tf,
    )


def _fit_semantic_witness_matrix(
    documents: Sequence[str],
    *,
    config: SemanticWitnessTfidfVectorizerConfig,
    empty_vocabulary_policy: str,
    label: str,
) -> tuple[TfidfVectorizer, Any] | None:
    vectorizer = _semantic_vectorizer(config)
    try:
        matrix = vectorizer.fit_transform(list(documents))
    except ValueError as exc:
        message = str(exc)
        vocabulary_is_empty = (
            "empty vocabulary" in message
            or "After pruning, no terms remain" in message
        )
        if (
            empty_vocabulary_policy == "return_empty_evidence_v1"
            and vocabulary_is_empty
        ):
            return None
        raise
    feature_count = len(vectorizer.get_feature_names_out())
    if config.max_features is not None and feature_count > config.max_features:
        raise RuntimeError(
            f"{label} complete vocabulary has {feature_count} terms, exceeding "
            f"the configured fail-closed assertion {config.max_features}"
        )
    return vectorizer, matrix


def _fail_on_semantic_witness_overflow(
    values: Sequence[Any],
    *,
    capacity: int | None,
    label: str,
) -> None:
    if capacity is not None and len(values) > capacity:
        raise RuntimeError(
            f"{label} produced {len(values)} complete values, exceeding the "
            f"configured fail-closed assertion {capacity}"
        )


def _signed_contrastive_terms(
    positive: Sequence[str],
    negative: Sequence[str],
    *,
    scientific_config: SemanticWitnessScientificConfig,
) -> list[dict[str, Any]]:
    if type(scientific_config) is not SemanticWitnessScientificConfig:
        raise TypeError("contrastive terms require SemanticWitnessScientificConfig")
    positive_documents = [str(value) for value in positive if str(value).strip()]
    negative_documents = [str(value) for value in negative if str(value).strip()]
    documents = [*positive_documents, *negative_documents]
    if (
        len(positive_documents) < scientific_config.retrieval_min_positive_documents
        or len(negative_documents) < scientific_config.retrieval_min_negative_documents
    ):
        return []
    fitted = _fit_semantic_witness_matrix(
        documents,
        config=scientific_config.retrieval_vectorizer,
        empty_vocabulary_policy=scientific_config.empty_vocabulary_policy,
        label="retrieval semantic witness",
    )
    if fitted is None:
        return []
    vectorizer, matrix = fitted
    split = len(positive_documents)
    numeric_dtype = (
        np.float32
        if scientific_config.direction_numeric_dtype == "float32"
        else np.float64
    )
    direction = np.asarray(
        matrix[:split].mean(axis=0) - matrix[split:].mean(axis=0),
        dtype=numeric_dtype,
    ).ravel()
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(range(len(terms)), key=lambda index: (-abs(direction[index]), terms[index]))
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in ranked:
        phrase = _safe_concept_phrase(terms[index])
        score = float(direction[index])
        if not phrase or phrase in seen or not math.isfinite(score):
            continue
        seen.add(phrase)
        output.append({"concept": phrase, "score": score})
    _fail_on_semantic_witness_overflow(
        output,
        capacity=scientific_config.maximum_retrieval_terms,
        label="retrieval semantic witness",
    )
    return output


def _embedding_concepts_only(
    evidence: Mapping[str, Any],
    *,
    scientific_config: SemanticWitnessScientificConfig,
) -> dict[str, Any]:
    if type(scientific_config) is not SemanticWitnessScientificConfig:
        raise TypeError("embedding concept projection requires a typed scientific config")
    output: list[dict[str, Any]] = []
    for raw in evidence.get("contrasts") or ():
        if not isinstance(raw, Mapping):
            continue
        positive = [
            str(row.get("text") or "")
            for key in ("positive_aligned_chunks", "positive_external_chunks")
            for row in raw.get(key) or ()
            if isinstance(row, Mapping)
        ]
        negative = [
            str(row.get("text") or "")
            for key in ("negative_aligned_chunks", "negative_external_chunks")
            for row in raw.get(key) or ()
            if isinstance(row, Mapping)
        ]
        scores = _signed_contrastive_terms(
            positive,
            negative,
            scientific_config=scientific_config,
        )
        if not scores:
            continue
        item = {
            key: raw.get(key)
            for key in (
                "name",
                "role_hint",
                "contrast_family",
                "direction_source",
                "cluster_component_index",
            )
            if raw.get(key) is not None
        }
        item["concept_probe_scores"] = scores
        output.append(item)
    return {
        "enabled": True,
        "concept_derivation": "tfidf_ngrams_contrasting_frozen_embedding_retrieval_tails",
        "raw_retrieved_excerpts_retained": False,
        "contrasts": output,
    }


def _htr_attention_score(
    row: Mapping[str, Any],
    *,
    minimum_exclusive: float,
) -> float | None:
    for key in ("attention", "attention_score", "chunk_attention"):
        value = row.get(key)
        if isinstance(value, (bool, np.bool_)):
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score) and score > minimum_exclusive:
            return score
    return None


def _htr_attention_text(row: Mapping[str, Any]) -> str:
    # These are observed HTR attention-table aliases.  Raw text is used only
    # transiently to derive configured n-grams and never enters a payload.
    for key in ("chunk_text", "highlighted_chunk_text", "evidence_snippet"):
        text = unicodedata.normalize("NFKC", str(row.get(key) or ""))
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if (
            _IDENTIFIER_NOISE.search(text)
            or _PERSON_NAME_CONTEXT.search(text)
            or _EMAIL_URL_LONG_ID.search(text)
        ):
            continue
        return text
    return ""


def _htr_attention_source_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    pair_side = str(row.get("pair_side") or "")
    row_identity: Any = None
    fallback_keys = ("candidate_row_id", "control_row_id")
    if "control" in pair_side.casefold():
        fallback_keys = ("control_row_id", "candidate_row_id")
    for key in ("row_id", "_oci_row_id", *fallback_keys):
        if row.get(key) is not None:
            row_identity = row.get(key)
            break
    if row_identity is None:
        return None
    return (str(row_identity), pair_side)


def _htr_attention_group_key(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    source_key = _htr_attention_source_key(row)
    if source_key is None:
        return None
    model_partition = tuple(
        "" if row.get(key) is None else str(row.get(key))
        for key in ("outer_fold", "fold", "inner_fold")
    )
    return (*model_partition, *source_key)


def _htr_name_like_term_in_documents(term: str, documents: Sequence[str]) -> bool:
    """Conservatively reject terms observed inside Title-Case name-like runs."""

    term_tokens = tuple(token.casefold() for token in re.findall(r"(?u)\b\w\w+\b", term))
    if not term_tokens:
        return True

    def name_shaped(token: str) -> bool:
        return bool(re.fullmatch(r"[A-Z][a-z]+", token)) or bool(
            len(token) >= 3 and token.isalpha() and token.isupper()
        )

    width = len(term_tokens)
    for document in documents:
        source_tokens = re.findall(r"(?u)\b\w\w+\b", document)
        folded = tuple(token.casefold() for token in source_tokens)
        for start in range(len(source_tokens) - width + 1):
            if folded[start : start + width] != term_tokens:
                continue
            for offset in range(width):
                position = start + offset
                if not name_shaped(source_tokens[position]):
                    continue
                left_name = position > 0 and name_shaped(source_tokens[position - 1])
                right = position + 1
                right_name = right < len(source_tokens) and name_shaped(source_tokens[right])
                if left_name or right_name:
                    return True
    return False


def _htr_phrase_has_unsafe_numeric_fragment(phrase: str) -> bool:
    """Reject dates/IDs while preserving short synthetic identifiers such as AX4."""

    for token in phrase.split():
        if token.isdigit() or re.search(r"\d{4,}", token):
            return True
    return False


def _htr_attention_contrastive_terms(
    attention_rows: Sequence[Mapping[str, Any]],
    *,
    scientific_config: SemanticWitnessScientificConfig,
) -> list[dict[str, Any]]:
    """Derive recurrent concepts from patient-local high/low HTR attention.

    CLS-pooled HTR encoders expose chunk-level attention but no token spans.
    One highest- and one lowest-attention distinct chunk per spent row makes
    those local weights comparable without allowing prolific patients to
    dominate.  Only high-attention-enriched n-grams recurring across at least
    two rows survive; source chunks and row identities remain transient.
    """

    if type(scientific_config) is not SemanticWitnessScientificConfig:
        raise TypeError("HTR contrastive terms require a typed scientific config")
    grouped: dict[tuple[str, ...], list[tuple[float, int, str, str]]] = {}
    sources: dict[tuple[str, ...], tuple[str, str]] = {}
    for raw in attention_rows:
        if not isinstance(raw, Mapping):
            continue
        source_key = _htr_attention_source_key(raw)
        group_key = _htr_attention_group_key(raw)
        score = _htr_attention_score(
            raw,
            minimum_exclusive=scientific_config.htr_attention_score_min_exclusive,
        )
        text = _htr_attention_text(raw)
        if source_key is None or group_key is None or score is None or not text:
            continue
        try:
            chunk_index = int(raw.get("chunk_index"))
        except (TypeError, ValueError):
            chunk_index = 2**31 - 1
        canonical_text = text.casefold()
        grouped.setdefault(group_key, []).append((score, chunk_index, canonical_text, text))
        sources[group_key] = source_key

    positive_documents: list[str] = []
    negative_documents: list[str] = []
    source_keys: list[tuple[str, str]] = []
    for group_key in sorted(grouped):
        candidates = grouped[group_key]
        high = min(candidates, key=lambda row: (-row[0], row[1], row[2]))
        low = min(candidates, key=lambda row: (row[0], row[1], row[2]))
        if (
            scientific_config.htr_require_strict_attention_separation
            and high[0] <= low[0]
        ) or high[2] == low[2]:
            continue
        positive_documents.append(high[3])
        negative_documents.append(low[3])
        source_keys.append(sources[group_key])

    # The configured independent-source and distinct-high-document thresholds
    # prevent repeated cross-fit models or one template from manufacturing
    # recurrence for a patient-specific fragment.
    unique_sources = tuple(sorted(set(source_keys)))
    distinct_high_documents = {text.casefold() for text in positive_documents}
    if (
        len(unique_sources) < scientific_config.htr_min_unique_sources
        or len(distinct_high_documents)
        < scientific_config.htr_min_distinct_positive_documents
    ):
        return []
    documents = [*positive_documents, *negative_documents]
    fitted = _fit_semantic_witness_matrix(
        documents,
        config=scientific_config.htr_vectorizer,
        empty_vocabulary_policy=scientific_config.empty_vocabulary_policy,
        label="HTR semantic witness",
    )
    if fitted is None:
        return []
    vectorizer, matrix = fitted

    split = len(positive_documents)
    positive_matrix = matrix[:split]
    negative_matrix = matrix[split:]
    source_counts = Counter(source_keys)
    row_weights = np.asarray(
        [1.0 / (len(unique_sources) * source_counts[key]) for key in source_keys],
        dtype=(
            np.float32
            if scientific_config.direction_numeric_dtype == "float32"
            else np.float64
        ),
    )
    direction = np.asarray(
        positive_matrix.T.dot(row_weights) - negative_matrix.T.dot(row_weights),
        dtype=row_weights.dtype,
    ).ravel()
    positive_support = np.zeros(positive_matrix.shape[1], dtype=np.int64)
    for source_key in unique_sources:
        indices: set[int] = set()
        for row_index, key in enumerate(source_keys):
            if key == source_key:
                indices.update(positive_matrix.getrow(row_index).indices.tolist())
        if indices:
            positive_support[np.fromiter(sorted(indices), dtype=np.int64)] += 1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(
        range(len(terms)),
        key=lambda index: (-direction[index], -len(str(terms[index]).split()), terms[index]),
    )
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in ranked:
        score = float(direction[index])
        if (
            score <= scientific_config.htr_direction_score_min_exclusive
            or not math.isfinite(score)
            or int(positive_support[index])
            < scientific_config.htr_min_positive_source_support
        ):
            continue
        phrase = _safe_concept_phrase(terms[index])
        if (
            not phrase
            or phrase in seen
            or _htr_phrase_has_unsafe_numeric_fragment(phrase)
            or _htr_name_like_term_in_documents(phrase, positive_documents)
        ):
            continue
        seen.add(phrase)
        output.append({"concept": phrase, "score": score})
    _fail_on_semantic_witness_overflow(
        output,
        capacity=scientific_config.maximum_htr_terms,
        label="HTR semantic witness",
    )
    return output


def _htr_concepts_only(
    evidence: Mapping[str, Any],
    *,
    scientific_config: SemanticWitnessScientificConfig,
) -> dict[str, Any]:
    if type(scientific_config) is not SemanticWitnessScientificConfig:
        raise TypeError("HTR concept projection requires a typed scientific config")
    output: dict[str, Any] = {}
    for stage in ("nuisance", "effect", "pair_uplift"):
        raw_stage = evidence.get(stage)
        if not isinstance(raw_stage, Mapping):
            continue
        attention_rows = list(raw_stage.get("attention") or ())
        rows: list[dict[str, Any]] = []
        for raw in attention_rows:
            if not isinstance(raw, Mapping):
                continue
            candidates: list[Any] = []
            spans = raw.get("top_token_spans") or raw.get("top_token_spans_json")
            if isinstance(spans, str):
                try:
                    spans = json.loads(spans)
                except json.JSONDecodeError:
                    spans = []
            if isinstance(spans, (list, tuple)):
                for span in spans:
                    if isinstance(span, Mapping):
                        candidates.append(span.get("text") or span.get("token") or span.get("span"))
            summary = raw.get("attended_token_summary")
            if summary:
                candidates.extend(re.split(r"[;,|]", str(summary)))
            phrases: list[str] = []
            for candidate in candidates:
                phrase = _safe_concept_phrase(candidate)
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
            _fail_on_semantic_witness_overflow(
                phrases,
                capacity=(
                    scientific_config.maximum_explicit_phrases_per_attention_row
                ),
                label="HTR explicit attention phrases for one source row",
            )
            if phrases:
                # The historical compactor recognizes this field directly.
                # Each value is a normalized attention-derived concept,
                # never the source chunk or evidence snippet.
                rows.extend({"attended_token_summary": value} for value in phrases)
        if not rows:
            rows = [
                {
                    "attended_token_summary": str(item["concept"]),
                    "attention_score": float(item["score"]),
                }
                for item in _htr_attention_contrastive_terms(
                    attention_rows,
                    scientific_config=scientific_config,
                )
            ]
        if rows:
            output[stage] = {"attention": rows}
    return output


def _sanitize_digest_terms(digest: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(digest))
    # Prompt-size metadata mentions historical chunk-text fields even though
    # no chunk text is retained.  It is not concept evidence and must not cross
    # the strict reviewer payload boundary.
    result.pop("prompt_compaction", None)
    for role_key in ("confounders", "effect_modifiers"):
        section = result.get(role_key)
        if not isinstance(section, Mapping):
            continue
        section = dict(section)
        bow_groups: list[dict[str, Any]] = []
        for raw_group in section.get("bow_blurbs") or ():
            if not isinstance(raw_group, Mapping):
                continue
            rows: list[dict[str, Any]] = []
            for raw_row in raw_group.get("rows") or ():
                if not isinstance(raw_row, Mapping):
                    continue
                phrase = _safe_concept_phrase(
                    raw_row.get("feature") or raw_row.get("term") or raw_row.get("phrase"),
                )
                if not phrase:
                    continue
                row = {"feature": phrase}
                for key in (
                    "coefficient",
                    "importance",
                    "score",
                    "signed_score",
                    "frequency",
                    "source_count",
                    "rank",
                ):
                    value = raw_row.get(key)
                    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
                        float(value)
                    ):
                        row[key] = float(value)
                rows.append(row)
            if rows:
                group = {
                    key: value
                    for key, value in raw_group.items()
                    if key in {"source", "view_name", "bow_model", "evidence_type", "meaning"}
                }
                group["rows"] = rows
                bow_groups.append(group)
        section["bow_blurbs"] = bow_groups
        # These were already stripped to concept probes/token spans.  Keep no
        # unrecognized fields that could reintroduce row-level evidence.
        section["embedding_chunks"] = list(section.get("embedding_chunks") or ())
        section["htr_blurbs"] = list(section.get("htr_blurbs") or ())
        result[role_key] = section
    return result


class HistoricalStage1SpentDiscoveryBackend:
    """Refit Stage-1 on spent rows and expose concept-only legacy evidence."""

    def __init__(
        self,
        *,
        dataset_path: Path | str,
        stage1_config_path: Path | str | None = None,
        embedding_cache_dir: Path | str | None = None,
        stage1_config_snapshot: HistoricalStage1ConfigSnapshot | None = None,
        embedding_cache: SpentOnlyFrozenChunkEmbeddingCache | None = None,
        htr_model_snapshot: PrivateHTRModelTreeSnapshot | None = None,
        semantic_witness_scientific_config: (
            SemanticWitnessScientificConfig | Mapping[str, Any]
        ),
        device: str = "cuda:0",
        bow_fold_parallelism: int = 1,
        bow_parallel_backend: str = "threads",
    ) -> None:
        self.dataset_path = Path(dataset_path).resolve()
        if not self.dataset_path.is_file():
            raise FileNotFoundError("historical Stage-1 dataset must exist")
        self._stage1_config_snapshot = _historical_stage1_config_snapshot(
            stage1_config_path,
            stage1_config_snapshot,
        )
        self.stage1_config_path = self._stage1_config_snapshot.source_path
        self.config = self._stage1_config_snapshot.applied_config()
        self.config.dataset_path = str(self.dataset_path)
        if isinstance(semantic_witness_scientific_config, Mapping):
            semantic_witness_scientific_config = (
                SemanticWitnessScientificConfig.from_mapping(
                    semantic_witness_scientific_config
                )
            )
        if type(semantic_witness_scientific_config) is not SemanticWitnessScientificConfig:
            raise TypeError(
                "historical spent discovery requires one closed semantic-witness "
                "scientific config"
            )
        self.semantic_witness_scientific_config = semantic_witness_scientific_config
        if isinstance(bow_fold_parallelism, (bool, np.bool_)) or not isinstance(
            bow_fold_parallelism, (int, np.integer)
        ):
            raise TypeError("bow_fold_parallelism must be an integer")
        self.bow_fold_parallelism = int(bow_fold_parallelism)
        if self.bow_fold_parallelism < 1:
            raise ValueError("bow_fold_parallelism must be positive")
        self.bow_parallel_backend = str(bow_parallel_backend).strip().lower()
        if self.bow_parallel_backend == "loky":
            self.bow_parallel_backend = "processes"
        if self.bow_parallel_backend not in {"threads", "processes"}:
            raise ValueError("bow_parallel_backend must be 'threads' or 'processes'")
        forest = self.config.architecture.multi_model_forest
        forest.outer_parallelism = "1"
        forest.fold_parallelism = "1"
        forest.bow_fold_parallelism = str(self.bow_fold_parallelism)
        forest.htr_fold_parallelism = "1"
        forest.cpus_total = self.bow_fold_parallelism
        forest.bow_parallel_backend = self.bow_parallel_backend
        forest.embedding_contrast.include_bow_phrases_as_concepts = False
        forest.embedding_contrast.concept_phrases = []
        forest.embedding_contrast.external_corpus_cache_dirs = []
        htr_path = _resolve_htr_model_path(self.config)
        if htr_model_snapshot is not None:
            if not isinstance(htr_model_snapshot, PrivateHTRModelTreeSnapshot):
                raise TypeError("htr_model_snapshot must be PrivateHTRModelTreeSnapshot")
            if htr_model_snapshot.source_path != htr_path:
                raise ValueError("HTR model path does not match supplied private snapshot")
            htr_model_snapshot.verify()
            self._htr_model_snapshot = htr_model_snapshot
        else:
            self._htr_model_snapshot = PrivateHTRModelTreeSnapshot(htr_path)
        self.config.architecture.htr_sentence_model = str(self._htr_model_snapshot.path)
        self.device = str(device)
        if not self.device.startswith("cuda:") and self.device != "cpu":
            raise ValueError("device must be 'cpu' or one explicit CUDA device")
        if embedding_cache is not None:
            if not isinstance(embedding_cache, SpentOnlyFrozenChunkEmbeddingCache):
                raise TypeError("embedding_cache must be SpentOnlyFrozenChunkEmbeddingCache")
            if (
                embedding_cache_dir is not None
                and embedding_cache.cache_dir != Path(embedding_cache_dir).resolve()
            ):
                raise ValueError("embedding_cache_dir does not match supplied embedding_cache")
            self.embedding_cache = embedding_cache
        else:
            if embedding_cache_dir is None:
                raise ValueError("embedding_cache_dir or embedding_cache is required")
            self.embedding_cache = SpentOnlyFrozenChunkEmbeddingCache(embedding_cache_dir)
        # Blank future rows preserve the global positional index needed by the
        # frozen embedding offsets without materializing future-row text.
        self._dataset_frame = pd.DataFrame(
            {
                "_oci_row_id": np.arange(self.embedding_cache.row_count, dtype=int),
                self.config.text_column: [""] * self.embedding_cache.row_count,
            }
        )

        import oci.inference.embedding_contrast_discovery as embedding_module
        import oci.inference.multi_model_forest_stage1 as stage1_module
        import oci.inference.multi_model_pair_uplift as pair_module

        effective_config_sha256 = _effective_applied_config_sha256(self.config)
        self._identity = {
            "backend": STAGE1_SPENT_DISCOVERY_BACKEND_ID,
            "stage1_config_sha256": self._stage1_config_snapshot.sha256,
            "effective_config_schema_version": EFFECTIVE_STAGE1_CONFIG_ID,
            "effective_config_sha256": effective_config_sha256,
            "embedding_cache": self.embedding_cache.identity(),
            "htr_model_tree_sha256": self._htr_model_snapshot.sha256,
            "htr_model_path_basename": self._htr_model_snapshot.source_basename,
            "htr_model_source_path_used_after_snapshot": False,
            "stage1_code_sha256": _module_file_sha256(stage1_module.__file__),
            "pair_code_sha256": _module_file_sha256(pair_module.__file__),
            "embedding_code_sha256": _module_file_sha256(embedding_module.__file__),
            "device": self.device,
            "bow_fold_parallelism": self.bow_fold_parallelism,
            "bow_parallel_backend": self.bow_parallel_backend,
            "htr_fold_parallelism": 1,
            "semantic_witness_scientific_config": (
                self.semantic_witness_scientific_config.as_dict()
            ),
            "semantic_witness_scientific_config_sha256": (
                self.semantic_witness_scientific_config.identity_sha256
            ),
            "concept_projection": (
                "complete_configured_bow_terms_htr_tokens_or_per_row_chunk_"
                "attention_contrast_embedding_tail_ngrams_v3"
            ),
            "raw_attention_or_embedding_excerpts_retained": False,
            "embedding_language_model_launch_allowed": False,
            "future_row_text_decoded_or_materialized": False,
            "code_sha256": _module_sha256(),
        }

    def identity(self) -> Mapping[str, Any]:
        self._stage1_config_snapshot.verify_source()
        self._htr_model_snapshot.verify()
        if (
            _effective_applied_config_sha256(self.config)
            != self._identity["effective_config_sha256"]
        ):
            raise RuntimeError("effective spent Stage-1 runtime config changed")
        current_cache_identity = self.embedding_cache.identity()
        if current_cache_identity != self._identity["embedding_cache"]:
            raise RuntimeError("Stage-1 spent discovery cache identity changed")
        return copy.deepcopy(self._identity)

    def fit_discovery(
        self,
        *,
        outer_fold: int,
        review_round: int,
        exact_spent_row_ids: tuple[int, ...],
        spent_texts: tuple[str, ...],
        spent_treatment: np.ndarray,
        spent_outcome: np.ndarray,
        work_dir: Path,
    ) -> SpentDiscoveryEvidence:
        self.identity()
        embedding_provider = self.embedding_cache.bind_spent(exact_spent_row_ids, spent_texts)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(self.config)
        runner = MultiModelForestStage1Runner(
            dataset=self._dataset_frame,
            config=config,
            output_path=work_dir / "unused_predictions.parquet",
            device=torch.device(self.device),
            gpu_ids=(
                (int(self.device.split(":", 1)[1]),) if self.device.startswith("cuda:") else None
            ),
            num_workers=1,
            embedding_provider=None,
        )
        runner.embedding_evidence_generator = _FrozenEmbeddingGenerator(embedding_provider)
        train_df = pd.DataFrame(
            {
                "_oci_row_id": exact_spent_row_ids,
                config.text_column: spent_texts,
                config.treatment_column: np.asarray(spent_treatment, dtype=float),
                config.outcome_column: np.asarray(spent_outcome, dtype=float),
            }
        )
        # The builder requires a transform frame.  It contains one spent row,
        # with text only, and is never used to form discovery evidence.
        transform_df = train_df.iloc[:1][["_oci_row_id", config.text_column]].copy()
        self._htr_model_snapshot.verify()
        try:
            bundle = runner._build_feature_bundle(
                train_df=train_df,
                test_df=transform_df,
                outer_fold=int(outer_fold),
            )
        finally:
            self._htr_model_snapshot.verify()
        handoff = bundle.handoff_evidence or {}
        ensemble_frames = [
            frame
            for frame in bundle.prediction_frames
            if not frame.empty
            and "source_name" in frame
            and set(frame["source_name"].astype(str)) == {"ensemble_mean_nuisance"}
        ]
        if len(ensemble_frames) != 1:
            raise RuntimeError("Stage-1 spent discovery omitted ensemble nuisance lineage")
        ensemble = ensemble_frames[0]
        ensemble = ensemble.loc[ensemble["split_role"] == "train_inner_oof"].copy()
        positions = {int(row_id): index for index, row_id in enumerate(ensemble["_oci_row_id"])}
        if set(positions) != set(exact_spent_row_ids):
            raise ValueError("Stage-1 ensemble nuisance changed the spent row set")
        order = [positions[row_id] for row_id in exact_spent_row_ids]
        e_hat = ensemble.iloc[order]["e_hat"].to_numpy(dtype=float)
        m_hat = ensemble.iloc[order]["m_hat"].to_numpy(dtype=float)
        t_resid = np.asarray(spent_treatment, dtype=float) - np.clip(
            e_hat, float(runner.nn_config.e_clip), 1.0 - float(runner.nn_config.e_clip)
        )
        pseudo_target = (np.asarray(spent_outcome, dtype=float) - m_hat) / t_resid

        embedding_generator = _FrozenCacheEmbeddingEvidenceGenerator(
            config=self.config,
            embedding_provider=embedding_provider,
            dataset_row_count=self.embedding_cache.row_count,
            output_dir=work_dir / "embedding_concept_discovery",
        )
        embedding_generator.prepare(self._dataset_frame)
        embedding_generator.bind_cluster_physical_fit_authority(
            ordered_fit_row_ids=exact_spent_row_ids,
            canonical_group_seed=derive_stage1_group_seed(
                int(self.config.seed),
                exact_spent_row_ids,
            ),
        )
        raw_embedding = embedding_generator.build_evidence(
            discovery_df=train_df,
            y=np.asarray(spent_outcome, dtype=float),
            t=np.asarray(spent_treatment, dtype=float),
            pseudo_target=[pseudo_target],
            t_resid=[t_resid],
            pseudo_target_names=["spent_context_ensemble_nuisance"],
            importance=handoff.get("importance") or {},
        )
        digest = _build_role_grouped_evidence_digest(
            importance=handoff.get("importance") or {},
            embedding_evidence=_embedding_concepts_only(
                raw_embedding,
                scientific_config=self.semantic_witness_scientific_config,
            ),
            htr_evidence=_htr_concepts_only(
                handoff.get("htr_evidence") or {},
                scientific_config=self.semantic_witness_scientific_config,
            ),
        )
        digest = _sanitize_digest_terms(digest)
        payload = {
            "outer_fold": int(outer_fold),
            "scope": "inner_train",
            "inner_fold": int(review_round) + 1,
            "context": {"evidence_digest": digest},
        }
        return SpentDiscoveryEvidence.create(
            source_kind=LEGACY_ALL_SOURCE,
            payload=payload,
            fit_row_provenance=FitRowProvenance(fit_row_ids=frozenset(exact_spent_row_ids)),
        )


def _sanitize_topic_banks(value: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if not isinstance(value, Mapping):
        return output
    for bank in ("treatment", "outcome", "effect"):
        raw_bank = value.get(bank)
        if not isinstance(raw_bank, Mapping):
            continue
        topics: list[dict[str, Any]] = []
        for topic_index, raw_topic in enumerate(raw_bank.get("topics") or (), start=1):
            if not isinstance(raw_topic, Mapping):
                continue
            terms: list[dict[str, Any]] = []
            for raw_term in raw_topic.get("terms") or ():
                row = raw_term if isinstance(raw_term, Mapping) else {"term": raw_term}
                phrase = _safe_concept_phrase(
                    row.get("term") or row.get("feature") or row.get("ngram"),
                )
                if not phrase:
                    continue
                term = {"term": phrase}
                for key in (
                    "loading",
                    "signed_score",
                    "fit_signed_score",
                    "standardized_score",
                    "rank",
                    "fit_rank",
                ):
                    numeric = row.get(key)
                    if isinstance(numeric, (int, float, np.integer, np.floating)) and math.isfinite(
                        float(numeric)
                    ):
                        term[key] = float(numeric)
                terms.append(term)
            if terms:
                topics.append(
                    {
                        "topic_id": f"{bank}_topic_{topic_index:03d}",
                        "terms": terms,
                    }
                )
        output[bank] = {"topics": topics}
    return output


def _sanitize_orphan_clusters(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cluster_index, raw_cluster in enumerate(value, start=1):
        terms: list[dict[str, Any]] = []
        for raw_term in raw_cluster.get("terms") or ():
            if not isinstance(raw_term, Mapping):
                continue
            phrase = _safe_concept_phrase(raw_term.get("term"))
            if not phrase:
                continue
            term = {"term": phrase}
            for key in (
                "fit_signed_score",
                "signed_score",
                "combined_importance",
                "fit_rank",
                "support_control",
                "support_treated",
                "lexical_similarity_to_seed",
            ):
                numeric = raw_term.get(key)
                if isinstance(numeric, (int, float, np.integer, np.floating)) and math.isfinite(
                    float(numeric)
                ):
                    term[key] = float(numeric)
            terms.append(term)
        if terms:
            output.append(
                {
                    "cluster_id": f"effect_orphan_cluster_{cluster_index:03d}",
                    "terms": terms,
                }
            )
    return output


class TfidfTopicOrphanSpentDiscoveryBackend:
    """Fit exact-spent TF-IDF topics and safe fit-side orphan clusters."""

    def __init__(
        self,
        *,
        stage1_config_path: Path | str | None = None,
        stage1_config_snapshot: HistoricalStage1ConfigSnapshot | None = None,
        outcome_type: str = "binary",
        orphan_config: OrphanNgramEvidenceAdapterConfig,
    ) -> None:
        orphan_config.validate()
        self.source = TfidfTopicOrphanContextBackend(
            stage1_config_path=stage1_config_path,
            stage1_config_snapshot=stage1_config_snapshot,
            outcome_type=outcome_type,
            max_orphan_features=None,
        )
        self.orphan_config = orphan_config
        self._identity = {
            "backend": TFIDF_SPENT_DISCOVERY_BACKEND_ID,
            "source_backend": self.source.identity(),
            "orphan_adapter_config": asdict(orphan_config),
            "orphan_adapter_code_sha256": _sha256_file(
                Path(
                    __import__(
                        "oci.inference.tfidf_orphan_evidence_adapter",
                        fromlist=["__file__"],
                    ).__file__
                )
            ),
            "code_sha256": _module_sha256(),
            "heldout_score_tests_enabled": False,
            "reviewer_excerpts_allowed": False,
        }

    def identity(self) -> Mapping[str, Any]:
        if self.source.identity() != self._identity["source_backend"]:
            raise RuntimeError("spent TF-IDF source backend identity changed")
        return copy.deepcopy(self._identity)

    def fit_discovery(
        self,
        *,
        outer_fold: int,
        review_round: int,
        exact_spent_row_ids: tuple[int, ...],
        spent_texts: tuple[str, ...],
        spent_treatment: np.ndarray,
        spent_outcome: np.ndarray,
        work_dir: Path,
    ) -> SpentDiscoveryEvidence:
        self.identity()
        work_dir = Path(work_dir)
        text_column = str(self.source.config.text_column)
        treatment_column = str(self.source.config.treatment_column)
        outcome_column = str(self.source.config.outcome_column)
        fit_df = pd.DataFrame(
            {
                "_oci_row_id": exact_spent_row_ids,
                text_column: spent_texts,
                treatment_column: np.asarray(spent_treatment, dtype=float),
                outcome_column: np.asarray(spent_outcome, dtype=float),
            }
        )
        # A transform-only copy of one spent row keeps the exact-context fitter
        # on its normal code path without supplying a future row or any labels.
        heldout_df = fit_df.iloc[:1][["_oci_row_id", text_column]].copy()
        metadata = fit_tfidf_topic_context(
            fit_df=fit_df,
            heldout_df=heldout_df,
            text_column=text_column,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            outcome_type=self.source.outcome_type,
            views=copy.deepcopy(self.source._views),
            nuisance_folds=int(self.source.config.architecture.multi_model_forest.nuisance_folds),
            config=copy.deepcopy(self.source._topic_config),
            artifact_dir=work_dir,
            scope_id=(f"review_spent_outer_{int(outer_fold):03d}_round_{int(review_round):02d}"),
            enable_heldout_score_tests=False,
        )
        topic_banks = _sanitize_topic_banks(metadata.get("topic_banks"))
        artifacts = metadata.get("artifacts") or {}
        effect_path = Path((artifacts.get("ngram_scores") or {}).get("effect", ""))
        if not effect_path.is_file():
            raise ValueError("exact-spent TF-IDF discovery omitted effect n-gram scores")
        score_frame = pd.read_parquet(effect_path)
        _validate_score_frame(score_frame)
        represented = _represented_topic_terms(topic_banks)
        records, _counts = _eligible_residual_records(
            score_frame,
            represented_terms=represented,
            config=self.orphan_config,
        )
        clusters = _cluster_records(
            records,
            outer_fold=int(outer_fold),
            config=self.orphan_config,
        )
        sanitized_clusters = _sanitize_orphan_clusters(clusters)
        payload = {
            "outer_fold": int(outer_fold),
            "scope": "inner_train",
            "inner_fold": int(review_round) + 1,
            "discovery": {
                "topic_banks": topic_banks,
                "effect_orphan_ngram_branch": {
                    "selected_cluster_ids": [row["cluster_id"] for row in sanitized_clusters],
                    "selected_clusters": sanitized_clusters,
                },
            },
        }
        return SpentDiscoveryEvidence.create(
            source_kind=TFIDF_TOPIC_SOURCE,
            payload=payload,
            fit_row_provenance=FitRowProvenance(fit_row_ids=frozenset(exact_spent_row_ids)),
        )


__all__ = [
    "ALL_NON_QUERY_DISCOVERY_FAMILIES",
    "BoundSpentFrozenChunkEmbeddingProvider",
    "ContextFitReviewSpentEvidenceProvider",
    "HistoricalStage1SpentDiscoveryBackend",
    "REVIEW_SPENT_EVIDENCE_CACHE_VERSION",
    "REVIEW_SPENT_EVIDENCE_PROVIDER_ID",
    "STAGE1_SPENT_DISCOVERY_BACKEND_ID",
    "SpentDiscoveryBackend",
    "SpentDiscoveryEvidence",
    "SpentOnlyFrozenChunkEmbeddingCache",
    "TFIDF_SPENT_DISCOVERY_BACKEND_ID",
    "TfidfTopicOrphanSpentDiscoveryBackend",
]
