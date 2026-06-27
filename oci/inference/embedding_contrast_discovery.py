"""Embedding-contrast evidence for agentic explicit-variable discovery."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..config import AppliedInferenceConfig, EmbeddingContrastDiscoveryConfig
from ..models.concept_embedding_cache import (
    ConceptEmbeddingCache,
    clear_sentence_transformer_cache,
    load_sentence_transformer,
)
from ..models.concept_embedding_utils import chunk_text_words


logger = logging.getLogger(__name__)


class EmbeddingContrastEvidenceGenerator:
    """Build train-fold embedding contrasts and retrieve aligned text chunks."""

    def __init__(
        self,
        *,
        config: AppliedInferenceConfig,
        output_dir: Path,
        embedding_provider: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.embedding_config: EmbeddingContrastDiscoveryConfig = (
            config.architecture.multi_model_agentic_forest.embedding_contrast
        )
        self.output_dir = Path(output_dir)
        self.embedding_provider = embedding_provider
        self._prepared = False
        self._row_ids: List[Any] = []
        self._row_id_to_position: Dict[Any, int] = {}
        self._chunks_by_position: List[List[str]] = []
        self._flat_embeddings = None
        self._offsets = None
        self._cache = None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.embedding_config, "enabled", False))

    def prepare(self, dataset: pd.DataFrame) -> None:
        """Prepare chunk embeddings for the dataset order used by this runner."""
        if not self.enabled or self._prepared:
            return
        if self.config.text_column not in dataset.columns:
            raise ValueError(
                f"Embedding contrast requires text column {self.config.text_column!r}"
            )

        texts = [str(text or "") for text in dataset[self.config.text_column].fillna("")]
        if "_oci_row_id" in dataset.columns:
            self._row_ids = dataset["_oci_row_id"].tolist()
        else:
            self._row_ids = list(range(len(dataset)))
        self._row_id_to_position = {
            _row_key(row_id): idx for idx, row_id in enumerate(self._row_ids)
        }
        self._chunks_by_position = [
            chunk_text_words(
                text,
                int(self.embedding_config.chunk_size_words),
                int(self.embedding_config.chunk_overlap_words),
                int(self.embedding_config.max_chunks),
                str(self.embedding_config.chunk_selection),
            )
            for text in texts
        ]

        if self.embedding_provider is not None:
            self._prepare_from_provider()
        else:
            self._prepare_from_sentence_transformer_cache(texts)
        self._prepared = True

    def build_evidence(
        self,
        *,
        discovery_df: pd.DataFrame,
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: Any,
        t_resid: Any,
        pseudo_target_names: Optional[Sequence[str]] = None,
        importance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return agent-facing embedding contrast evidence for one discovery fold."""
        if not self.enabled:
            return {}
        if not self._prepared:
            self.prepare(discovery_df)

        positions = self._positions_for_frame(discovery_df)
        if not positions:
            return {"enabled": True, "skipped": "no_rows"}

        patient_embeddings = self._patient_embeddings(positions)
        patient_embeddings = _residualize_embeddings(
            patient_embeddings,
            discovery_df,
            self.embedding_config.residualize_columns,
        )
        patient_embeddings = _normalize_rows(patient_embeddings)

        concept_phrases = self._concept_phrases(importance or {})
        concept_embeddings = (
            self._encode_concepts(concept_phrases) if concept_phrases else None
        )
        contrasts = []
        for contrast in self._contrast_specs(
            y=np.asarray(y, dtype=float),
            t=np.asarray(t, dtype=float),
            pseudo_targets=_named_pseudo_targets(
                pseudo_target,
                t_resid,
                pseudo_target_names,
            ),
        ):
            contrasts.append(
                self._build_one_contrast(
                    positions=positions,
                    patient_embeddings=patient_embeddings,
                    concept_phrases=concept_phrases,
                    concept_embeddings=concept_embeddings,
                    **contrast,
                )
            )

        return {
            "enabled": True,
            "model_name": self.embedding_config.model_name,
            "unit": "patient_row",
            "chunking": {
                "chunk_size_words": int(self.embedding_config.chunk_size_words),
                "chunk_overlap_words": int(self.embedding_config.chunk_overlap_words),
                "max_chunks": int(self.embedding_config.max_chunks),
                "chunk_selection": str(self.embedding_config.chunk_selection),
            },
            "residualized_columns": [
                col
                for col in self.embedding_config.residualize_columns
                if col in discovery_df.columns
            ],
            "n_patients": int(len(positions)),
            "n_concept_phrases": int(len(concept_phrases)),
            "contrasts": contrasts,
        }

    def _prepare_from_provider(self) -> None:
        flat_chunks = [chunk for chunks in self._chunks_by_position for chunk in chunks]
        offsets = np.zeros(len(self._chunks_by_position) + 1, dtype=np.int64)
        cursor = 0
        for idx, chunks in enumerate(self._chunks_by_position):
            cursor += len(chunks)
            offsets[idx + 1] = cursor
        embeddings = self._encode_with_provider(flat_chunks)
        embeddings = _coerce_embedding_matrix(embeddings, expected_rows=len(flat_chunks))
        if bool(self.embedding_config.normalize_embeddings):
            embeddings = _normalize_rows(embeddings)
        self._flat_embeddings = embeddings.astype(np.float32, copy=False)
        self._offsets = offsets

    def _prepare_from_sentence_transformer_cache(self, texts: Sequence[str]) -> None:
        dataset_path = self.config.dataset_path or str(self.output_dir / "in_memory_dataset")
        cache_dir = (
            Path(str(self.embedding_config.cache_dir))
            if self.embedding_config.cache_dir
            else _default_embedding_cache_dir(dataset_path, self.output_dir)
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = ConceptEmbeddingCache(
            cache_dir=str(cache_dir),
            sentence_model_name=str(self.embedding_config.model_name),
            dataset_path=str(dataset_path),
            chunk_size_words=int(self.embedding_config.chunk_size_words),
            chunk_overlap_words=int(self.embedding_config.chunk_overlap_words),
            max_chunks=int(self.embedding_config.max_chunks),
            normalize_embeddings=bool(self.embedding_config.normalize_embeddings),
            chunk_selection=str(self.embedding_config.chunk_selection),
        )
        logger.info("Embedding contrast chunk cache: %s", cache.cache_path)
        cache_valid = cache.is_valid(expected_num_samples=len(texts))
        if cache_valid:
            logger.info("Reusing embedding contrast chunk cache")
        else:
            logger.info("Building embedding contrast chunk cache")
            try:
                cache.precompute(
                    list(texts),
                    device=_torch_device_or_none(self.embedding_config.device),
                    batch_size=int(self.embedding_config.batch_size),
                )
            finally:
                _release_sentence_transformer_model(str(self.embedding_config.model_name))
        cache.open()
        self._cache = cache
        self._flat_embeddings = cache.hidden_states_array.flat
        self._offsets = cache.hidden_states_array.offsets

    def _positions_for_frame(self, frame: pd.DataFrame) -> List[int]:
        if "_oci_row_id" in frame.columns:
            row_ids = frame["_oci_row_id"].tolist()
        else:
            row_ids = list(range(len(frame)))
        positions = []
        for row_id in row_ids:
            position = self._row_id_to_position.get(_row_key(row_id))
            if position is not None:
                positions.append(int(position))
        return positions

    def _chunk_matrix(self, position: int) -> np.ndarray:
        start = int(self._offsets[position])
        end = int(self._offsets[position + 1])
        return np.asarray(self._flat_embeddings[start:end], dtype=np.float32)

    def _patient_embeddings(self, positions: Sequence[int]) -> np.ndarray:
        pooled = []
        for position in positions:
            chunks = self._chunk_matrix(int(position))
            if chunks.size == 0:
                pooled.append(np.zeros(1, dtype=np.float32))
            else:
                pooled.append(np.mean(chunks, axis=0))
        matrix = np.vstack(pooled).astype(np.float32, copy=False)
        return _normalize_rows(matrix)

    def _contrast_specs(
        self,
        *,
        y: np.ndarray,
        t: np.ndarray,
        pseudo_targets: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        treatment_labels, treatment_mask = _binary_labels(t)
        specs.append(
            {
                "name": "treatment",
                "positive_label": "treated",
                "negative_label": "untreated",
                "labels": treatment_labels,
                "mask": treatment_mask,
                "sample_weights": None,
                "role_hint": "confounder",
            }
        )

        if str(self.config.outcome_type).lower() == "continuous":
            labels, mask = _tail_labels(y, float(self.embedding_config.pseudo_target_quantile))
            positive_label = "higher_outcome"
            negative_label = "lower_outcome"
        else:
            labels, mask = _binary_labels(y)
            positive_label = "outcome_present"
            negative_label = "outcome_absent"
        specs.append(
            {
                "name": "outcome",
                "positive_label": positive_label,
                "negative_label": negative_label,
                "labels": labels,
                "mask": mask,
                "sample_weights": None,
                "role_hint": "confounder",
            }
        )

        multiple_pseudo_targets = len(pseudo_targets) > 1
        for pseudo_name, pseudo_target, t_resid in pseudo_targets:
            pseudo_labels, pseudo_mask = _tail_labels(
                pseudo_target,
                float(self.embedding_config.pseudo_target_quantile),
            )
            pseudo_weights = None
            if bool(self.embedding_config.pseudo_target_weighted):
                pseudo_weights = np.square(np.asarray(t_resid, dtype=float))
            contrast_name = "r_pseudo_target"
            if multiple_pseudo_targets:
                contrast_name = f"r_pseudo_target__{_safe_contrast_suffix(pseudo_name)}"
            specs.append(
                {
                    "name": contrast_name,
                    "positive_label": f"higher_r_pseudo_target:{pseudo_name}",
                    "negative_label": f"lower_r_pseudo_target:{pseudo_name}",
                    "labels": pseudo_labels,
                    "mask": pseudo_mask,
                    "sample_weights": pseudo_weights,
                    "role_hint": "effect_modifier",
                }
            )
        return specs

    def _build_one_contrast(
        self,
        *,
        positions: Sequence[int],
        patient_embeddings: np.ndarray,
        concept_phrases: Sequence[str],
        concept_embeddings: Optional[np.ndarray],
        name: str,
        positive_label: str,
        negative_label: str,
        labels: np.ndarray,
        mask: np.ndarray,
        sample_weights: Optional[np.ndarray],
        role_hint: str,
    ) -> Dict[str, Any]:
        labels = np.asarray(labels, dtype=int)
        mask = np.asarray(mask, dtype=bool)
        usable = mask & np.all(np.isfinite(patient_embeddings), axis=1)
        pos_mask = usable & (labels == 1)
        neg_mask = usable & (labels == 0)
        n_pos = int(np.sum(pos_mask))
        n_neg = int(np.sum(neg_mask))
        record: Dict[str, Any] = {
            "name": name,
            "positive_label": positive_label,
            "negative_label": negative_label,
            "role_hint": role_hint,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "min_probe_auc": float(self.embedding_config.min_probe_auc),
        }
        if n_pos < 2 or n_neg < 2:
            record["retrieval_skipped"] = "too_few_examples_per_group"
            return record

        weights = None
        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=float)
            weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)

        mean_direction = _weighted_mean(patient_embeddings[pos_mask], _subset(weights, pos_mask))
        mean_direction -= _weighted_mean(patient_embeddings[neg_mask], _subset(weights, neg_mask))
        mean_norm = float(np.linalg.norm(mean_direction))
        record["mean_difference_norm"] = _finite_or_none(mean_norm)

        probe_auc, _probe_direction = _linear_probe_direction(
            patient_embeddings[usable],
            labels[usable],
        )
        record["probe_auc"] = _finite_or_none(probe_auc)

        min_auc = float(self.embedding_config.min_probe_auc)
        if min_auc > 0.0 and (
            probe_auc is None
            or not np.isfinite(probe_auc)
            or probe_auc < min_auc
        ):
            record["retrieval_skipped"] = "probe_auc_below_threshold"
            return record
        if mean_norm <= 0.0:
            record["retrieval_skipped"] = "zero_mean_difference_direction"
            return record

        direction = _normalize_vector(mean_direction)
        record["direction_source"] = "mean_difference"
        record["probe_auc_role"] = (
            "diagnostic_gate_only" if min_auc > 0.0 else "diagnostic_only"
        )
        record["direction_norm"] = _finite_or_none(float(np.linalg.norm(direction)))
        record["positive_aligned_chunks"] = self._retrieve_chunks(
            positions,
            direction,
            descending=True,
        )
        record["negative_aligned_chunks"] = self._retrieve_chunks(
            positions,
            direction,
            descending=False,
        )
        record["concept_probe_scores"] = self._score_concepts(
            concept_phrases,
            concept_embeddings,
            direction,
        )
        return record

    def _retrieve_chunks(
        self,
        positions: Sequence[int],
        direction: np.ndarray,
        *,
        descending: bool,
    ) -> List[Dict[str, Any]]:
        candidates: List[Tuple[float, int, int, str]] = []
        for position in positions:
            chunks = self._chunks_by_position[int(position)]
            chunk_embeddings = self._chunk_matrix(int(position))
            if chunk_embeddings.size == 0:
                continue
            scores = np.asarray(chunk_embeddings @ direction, dtype=float)
            for chunk_idx, score in enumerate(scores):
                if not np.isfinite(score):
                    continue
                text = chunks[chunk_idx] if chunk_idx < len(chunks) else ""
                if not _informative_chunk_text(text):
                    continue
                candidates.append((float(score), int(position), int(chunk_idx), text))

        candidates.sort(key=lambda item: item[0], reverse=descending)
        rows: List[Dict[str, Any]] = []
        per_patient_counts: Dict[Any, int] = {}
        max_per_patient = int(self.embedding_config.max_chunks_per_patient)
        for score, position, chunk_idx, text in candidates:
            row_id = self._row_ids[position]
            key = _row_key(row_id)
            if per_patient_counts.get(key, 0) >= max_per_patient:
                continue
            per_patient_counts[key] = per_patient_counts.get(key, 0) + 1
            rows.append(
                {
                    "row_id": _jsonable_scalar(row_id),
                    "chunk_index": int(chunk_idx),
                    "score": _finite_or_none(score),
                    "text": text,
                }
            )
            if len(rows) >= int(self.embedding_config.top_k_chunks_per_tail):
                break
        return rows

    def _concept_phrases(self, importance: Dict[str, Any]) -> List[str]:
        phrases: List[str] = list(self.embedding_config.concept_phrases)
        if bool(self.embedding_config.include_bow_phrases_as_concepts):
            for row in importance.get("phrase_features", []) or []:
                phrase = str(row.get("feature", "")).strip()
                if phrase:
                    phrases.append(phrase)
        deduped = list(dict.fromkeys(phrases))
        limit = int(self.embedding_config.max_concept_phrases)
        return deduped[:limit] if limit > 0 else []

    def _score_concepts(
        self,
        concept_phrases: Sequence[str],
        concept_embeddings: Optional[np.ndarray],
        direction: np.ndarray,
    ) -> List[Dict[str, Any]]:
        if not concept_phrases or concept_embeddings is None:
            return []
        embeddings = _normalize_rows(concept_embeddings)
        scores = np.asarray(embeddings @ direction, dtype=float)
        order = np.argsort(np.abs(scores))[::-1]
        rows = []
        for idx in order[: int(self.embedding_config.concept_probe_top_k)]:
            rows.append(
                {
                    "concept": str(concept_phrases[int(idx)]),
                    "score": _finite_or_none(float(scores[int(idx)])),
                }
            )
        return rows

    def _encode_concepts(self, phrases: Sequence[str]) -> np.ndarray:
        if self.embedding_provider is not None:
            embeddings = self._encode_with_provider(list(phrases))
        else:
            try:
                encoder = load_sentence_transformer(
                    str(self.embedding_config.model_name),
                    device=_torch_device_or_none(self.embedding_config.device),
                )
                embeddings = encoder.encode(
                    list(phrases),
                    batch_size=max(
                        1,
                        min(int(self.embedding_config.batch_size), len(phrases)),
                    ),
                    convert_to_numpy=True,
                    normalize_embeddings=bool(self.embedding_config.normalize_embeddings),
                    show_progress_bar=False,
                )
            finally:
                _release_sentence_transformer_model(str(self.embedding_config.model_name))
        return _coerce_embedding_matrix(embeddings, expected_rows=len(phrases))

    def _encode_with_provider(self, texts: Sequence[str]) -> np.ndarray:
        provider = self.embedding_provider
        if hasattr(provider, "encode_chunks"):
            return provider.encode_chunks(list(texts))
        if hasattr(provider, "encode_texts"):
            return provider.encode_texts(list(texts))
        if hasattr(provider, "encode"):
            return provider.encode(list(texts))
        raise TypeError(
            "embedding_provider must implement encode_chunks, encode_texts, or encode"
        )


def redact_embedding_contrast_evidence(evidence: Any) -> Any:
    """Return a copy of embedding evidence with retrieved raw text removed."""
    if not evidence:
        return evidence

    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            redacted: Dict[str, Any] = {}
            for key, item in value.items():
                if key == "text":
                    redacted[key] = None
                    redacted["text_redacted"] = True
                else:
                    redacted[key] = redact(item)
            return redacted
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    return redact(copy.deepcopy(evidence))


def _informative_chunk_text(text: str) -> bool:
    """Return True when a retrieved chunk has enough clinical text to show."""
    compact = "".join(ch for ch in str(text or "") if ch.isalnum())
    return len(compact) >= 12


def _named_pseudo_targets(
    pseudo_target: Any,
    t_resid: Any,
    names: Optional[Sequence[str]],
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    targets = _as_target_list(pseudo_target)
    residuals = _as_target_list(t_resid)
    if len(residuals) == 1 and len(targets) > 1:
        residuals = residuals * len(targets)
    if len(targets) != len(residuals):
        raise ValueError("pseudo_target and t_resid must have matching view counts")
    if names is None:
        target_names = [
            "r_pseudo_target" if len(targets) == 1 else f"view_{idx}"
            for idx in range(1, len(targets) + 1)
        ]
    else:
        target_names = [str(name) for name in names]
    if len(target_names) != len(targets):
        raise ValueError("pseudo_target_names must match pseudo_target view count")
    return [
        (
            target_names[idx],
            np.asarray(targets[idx], dtype=float),
            np.asarray(residuals[idx], dtype=float),
        )
        for idx in range(len(targets))
    ]


def _as_target_list(value: Any) -> List[Any]:
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, (list, tuple)):
        if not value:
            return [np.asarray([], dtype=float)]
        first = np.asarray(value[0])
        if first.ndim > 0:
            return list(value)
    return [value]


def _safe_contrast_suffix(value: Any) -> str:
    suffix = []
    for ch in str(value).strip().lower():
        if ch.isalnum():
            suffix.append(ch)
        elif suffix and suffix[-1] != "_":
            suffix.append("_")
    text = "".join(suffix).strip("_")
    return text or "view"


def _linear_probe_direction(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return None, None
    class_counts = np.bincount(labels, minlength=2)
    min_class = int(np.min(class_counts[:2]))
    if min_class < 2:
        return None, None

    auc = None
    folds = min(5, min_class)
    if folds >= 2:
        oof = np.full(len(labels), np.nan, dtype=float)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=83_001)
        for train_idx, heldout_idx in splitter.split(embeddings, labels):
            model = LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=1000,
            )
            model.fit(embeddings[train_idx], labels[train_idx])
            oof[heldout_idx] = model.predict_proba(embeddings[heldout_idx])[:, 1]
        if np.all(np.isfinite(oof)):
            try:
                auc = float(roc_auc_score(labels, oof))
            except ValueError:
                auc = None

    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
    model.fit(embeddings, labels)
    coef = np.asarray(model.coef_, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(coef)) or np.linalg.norm(coef) <= 0.0:
        return auc, None
    return auc, coef


def _binary_labels(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    labels = (values >= 0.5).astype(int)
    mask = np.isfinite(values)
    if len(np.unique(labels[mask])) < 2:
        mask = np.zeros_like(mask, dtype=bool)
    return labels, mask


def _tail_labels(values: np.ndarray, quantile: float) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    labels = np.zeros(len(values), dtype=int)
    mask = np.zeros(len(values), dtype=bool)
    if np.sum(finite) < 4:
        return labels, mask
    low, high = np.nanquantile(values[finite], [quantile, 1.0 - quantile])
    low_mask = finite & (values <= low)
    high_mask = finite & (values >= high)
    overlap = low_mask & high_mask
    low_mask[overlap] = False
    high_mask[overlap] = False
    labels[high_mask] = 1
    mask = low_mask | high_mask
    if np.sum(low_mask) < 2 or np.sum(high_mask) < 2:
        mask[:] = False
    return labels, mask


def _residualize_embeddings(
    embeddings: np.ndarray,
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> np.ndarray:
    usable_columns = [col for col in columns if col in frame.columns]
    if not usable_columns:
        return embeddings
    covariates = pd.get_dummies(
        frame[usable_columns].reset_index(drop=True),
        dummy_na=True,
        drop_first=True,
    )
    if covariates.shape[1] == 0:
        return embeddings
    x = covariates.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[0] != embeddings.shape[0]:
        return embeddings
    x = np.column_stack([np.ones(x.shape[0], dtype=float), x])
    try:
        beta, *_ = np.linalg.lstsq(x, embeddings, rcond=None)
    except np.linalg.LinAlgError:
        logger.warning("Embedding residualization failed; using raw embeddings")
        return embeddings
    return embeddings - x @ beta


def _weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    if weights is None or len(weights) != len(values) or np.sum(weights) <= 0.0:
        return np.mean(values, axis=0)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return np.sum(values * weights[:, None], axis=0)


def _subset(values: Optional[np.ndarray], mask: np.ndarray) -> Optional[np.ndarray]:
    if values is None:
        return None
    return np.asarray(values, dtype=float)[mask]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        return vector
    return vector / norm


def _coerce_embedding_matrix(embeddings: Any, expected_rows: int) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise RuntimeError(f"Embedding provider returned shape {matrix.shape}")
    if matrix.shape[0] != expected_rows:
        raise RuntimeError(
            f"Embedding provider returned {matrix.shape[0]} embeddings for "
            f"{expected_rows} texts"
        )
    return matrix


def _torch_device_or_none(device: Optional[str]):
    if device is None or str(device).strip().lower() in {"", "auto"}:
        return None
    import torch

    return torch.device(str(device))


def _release_sentence_transformer_model(model_name: str) -> None:
    try:
        clear_sentence_transformer_cache(model_name=model_name)
        logger.info("Released sentence-transformer model cache for %s", model_name)
    except Exception:
        logger.warning(
            "Failed to release sentence-transformer model cache for %s",
            model_name,
            exc_info=True,
        )


def _default_embedding_cache_dir(dataset_path: str, output_dir: Path) -> Path:
    path = Path(str(dataset_path))
    if str(path).endswith("in_memory_dataset"):
        return Path(output_dir) / "embedding_cache"
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser().absolute()
    dataset_dir = resolved if resolved.is_dir() else resolved.parent
    if not str(dataset_dir):
        return Path(output_dir) / "embedding_cache"
    return dataset_dir / ".oci_cache" / "embedding_contrast"


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _row_key(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value
