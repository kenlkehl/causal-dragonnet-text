"""ColBERT-style late-interaction scoring for short queries and long evidence.

The newest Sentence Transformers API exposes ``MultiVectorEncoder`` directly.
The project currently pins an earlier release for text-only runtime compatibility,
so this module also contains a small compatibility loader for Stanford ColBERT
checkpoints.  Both paths produce normalized token vectors and MeanMaxSim scores.
"""

from __future__ import annotations

import json
import logging
import string
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from .concept_embedding_utils import split_text_to_token_chunks


LOGGER = logging.getLogger(__name__)

_ENCODER_CACHE: dict[tuple[str, str], Any] = {}
_ENCODER_LOCK = threading.RLock()


def _resolved_device(device: str) -> str:
    requested = str(device).strip()
    if not requested:
        raise ValueError("late-interaction device must be nonempty")
    if requested.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _as_token_matrix(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise RuntimeError(
            "late-interaction encoder returned an invalid token matrix "
            f"with shape {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("late-interaction encoder returned non-finite token vectors")
    return matrix


def _score_token_matrices(
    query_matrices: Sequence[np.ndarray],
    document_matrix: np.ndarray,
    *,
    max_interaction_elements: int = 20_000_000,
) -> np.ndarray:
    """Score several queries against one document with MeanMaxSim."""

    if not query_matrices:
        return np.empty(0, dtype=np.float32)
    document = torch.from_numpy(_as_token_matrix(document_matrix))
    scores = np.empty(len(query_matrices), dtype=np.float32)
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, query in enumerate(query_matrices):
        by_length[int(_as_token_matrix(query).shape[0])].append(index)
    for query_length, indexes in by_length.items():
        per_query_elements = max(1, query_length * int(document.shape[0]))
        batch_size = max(1, int(max_interaction_elements) // per_query_elements)
        for start in range(0, len(indexes), batch_size):
            batch_indexes = indexes[start : start + batch_size]
            queries = torch.from_numpy(
                np.stack([_as_token_matrix(query_matrices[index]) for index in batch_indexes])
            )
            # Token embeddings are L2-normalized by both supported encoders, so
            # the dot product is token-level cosine similarity.
            batch_scores = torch.matmul(queries, document.T).amax(dim=2).mean(dim=1)
            scores[batch_indexes] = batch_scores.detach().cpu().numpy().astype(np.float32)
    return scores


class _SentenceTransformersMultiVectorAdapter:
    """Adapter around the released/upcoming native MultiVectorEncoder API."""

    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import MultiVectorEncoder

        self.model = MultiVectorEncoder(
            model_name,
            device=device,
            similarity_fn_name="meanmaxsim",
            model_kwargs={"torch_dtype": torch.float32},
        )
        self.model.float()
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        transformer = self.model[0]
        configured_length = getattr(transformer, "document_length", None)
        if configured_length is None:
            configured_length = getattr(self.tokenizer, "model_max_length", 300)
        try:
            configured_length = int(configured_length)
        except (TypeError, ValueError):
            configured_length = 300
        self.document_length = min(300, max(8, configured_length))
        # Native MultiVectorEncoder receives the unframed text.  Its tokenizer
        # therefore supplies the exact length check used for chunking.
        self.document_encoding_prefix = ""

    def encode_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        values = self.model.encode_query(
            list(texts),
            batch_size=min(32, len(texts)),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [_as_token_matrix(value) for value in values]

    def encode_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        values = self.model.encode_document(
            list(texts),
            batch_size=min(32, len(texts)),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [_as_token_matrix(value) for value in values]


def _model_file(model_name: str, filename: str, *, required: bool) -> Path | None:
    local = Path(model_name).expanduser()
    if local.is_dir():
        path = local / filename
        if path.is_file():
            return path
        if required:
            raise FileNotFoundError(f"ColBERT checkpoint is missing {path}")
        return None
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo_id=model_name, filename=filename))
    except Exception:
        if required:
            raise
        return None


class _StanfordColbertCompatibilityAdapter:
    """Load the common Stanford/HF_ColBERT checkpoint format without remote code."""

    def __init__(self, model_name: str, device: str) -> None:
        from safetensors import safe_open
        from transformers import AutoModel, AutoTokenizer

        metadata_path = _model_file(model_name, "artifact.metadata", required=False)
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path is not None
            else {}
        )
        self.query_marker = str(metadata.get("query_token_id") or "[unused0]")
        self.document_marker = str(metadata.get("doc_token_id") or "[unused1]")
        self.document_encoding_prefix = f"{self.document_marker} "
        self.query_length = int(metadata.get("query_maxlen") or 32)
        self.document_length = int(metadata.get("doc_maxlen") or 300)
        self.attend_to_expansion = bool(metadata.get("attend_to_mask_tokens"))
        self.mask_punctuation = bool(metadata.get("mask_punctuation", True))
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        existing_specials = list(
            getattr(self.tokenizer, "additional_special_tokens", None) or []
        )
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": list(
                    dict.fromkeys(
                        [*existing_specials, self.query_marker, self.document_marker]
                    )
                )
            }
        )
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        if len(self.tokenizer) > self.encoder.get_input_embeddings().num_embeddings:
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.encoder.to(self.device)
        self.encoder.float()
        self.encoder.eval()

        weights_path = _model_file(model_name, "model.safetensors", required=True)
        assert weights_path is not None
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            if "linear.weight" not in handle.keys():
                raise ValueError(
                    "the configured late-interaction model is not a supported Stanford "
                    "ColBERT checkpoint: model.safetensors has no linear.weight"
                )
            projection = handle.get_tensor("linear.weight").float()
        if projection.ndim != 2 or projection.shape[1] != self.encoder.config.hidden_size:
            raise ValueError(
                "ColBERT projection shape does not match the base encoder: "
                f"{tuple(projection.shape)} versus hidden_size="
                f"{self.encoder.config.hidden_size}"
            )
        self.projection = projection.to(self.device)

        punctuation_ids = []
        if self.mask_punctuation:
            unknown_id = self.tokenizer.unk_token_id
            for character in string.punctuation:
                token_id = self.tokenizer.convert_tokens_to_ids(character)
                if token_id is not None and token_id != unknown_id:
                    punctuation_ids.append(int(token_id))
        self.punctuation_ids = tuple(sorted(set(punctuation_ids)))

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return functional.normalize(
            functional.linear(hidden_states.float(), self.projection),
            p=2,
            dim=-1,
        )

    def encode_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), 32):
            batch = [f"{self.query_marker} {text}" for text in texts[start : start + 32]]
            encoded_lengths = [
                len(self.tokenizer.encode(text, add_special_tokens=True)) for text in batch
            ]
            if any(length >= self.query_length for length in encoded_lengths):
                raise ValueError(
                    "a candidate feature name does not fit the configured ColBERT query "
                    f"length of {self.query_length} tokens with expansion headroom"
                )
            encoded = self.tokenizer(
                batch,
                padding="max_length",
                truncation=False,
                max_length=self.query_length,
                return_tensors="pt",
            )
            attention_mask = encoded["attention_mask"]
            expansion_positions = attention_mask == 0
            expansion_id = self.tokenizer.mask_token_id
            if expansion_id is None:
                expansion_id = self.tokenizer.eos_token_id
            if expansion_id is None:
                raise ValueError("ColBERT query expansion requires a mask or EOS token")
            encoded["input_ids"] = encoded["input_ids"].masked_fill(
                expansion_positions,
                int(expansion_id),
            )
            if self.attend_to_expansion:
                encoded["attention_mask"] = attention_mask.masked_fill(
                    expansion_positions,
                    1,
                )
            model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                vectors = self._project(self.encoder(**model_inputs).last_hidden_state)
            for row in vectors:
                # Fixed ColBERT query expansion makes every position part of
                # scoring, including expansion positions excluded from attention.
                outputs.append(row.detach().cpu().numpy().astype(np.float32))
        return outputs

    def encode_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        outputs: list[np.ndarray] = []
        punctuation = (
            torch.tensor(self.punctuation_ids, dtype=torch.long)
            if self.punctuation_ids
            else None
        )
        for start in range(0, len(texts), 32):
            batch = [f"{self.document_marker} {text}" for text in texts[start : start + 32]]
            encoded_lengths = [
                len(self.tokenizer.encode(text, add_special_tokens=True)) for text in batch
            ]
            if any(length > self.document_length for length in encoded_lengths):
                raise ValueError(
                    "a losslessly chunked evidence passage exceeds the configured ColBERT "
                    f"document length of {self.document_length} tokens"
                )
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            scoring_mask = encoded["attention_mask"].bool()
            if punctuation is not None:
                scoring_mask &= ~torch.isin(encoded["input_ids"], punctuation)
            model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                vectors = self._project(self.encoder(**model_inputs).last_hidden_state)
            for row, mask in zip(vectors, scoring_mask):
                outputs.append(
                    row[mask.to(row.device)].detach().cpu().numpy().astype(np.float32)
                )
        return outputs


def _load_encoder(model_name: str, device: str) -> Any:
    key = (str(model_name).strip(), _resolved_device(device))
    if key in _ENCODER_CACHE:
        return _ENCODER_CACHE[key]
    try:
        encoder = _SentenceTransformersMultiVectorAdapter(*key)
        LOGGER.info(
            "loaded late-interaction model=%s with Sentence Transformers MultiVectorEncoder",
            key[0],
        )
    except (ImportError, AttributeError):
        encoder = _StanfordColbertCompatibilityAdapter(*key)
        LOGGER.info(
            "loaded late-interaction model=%s with the Stanford ColBERT compatibility adapter",
            key[0],
        )
    _ENCODER_CACHE[key] = encoder
    return encoder


def score_late_interaction_pairs(
    queries: Sequence[str],
    documents: Sequence[str],
    model_name: str,
    device: str = "cpu",
    *,
    document_chunk_overlap_tokens: int = 32,
) -> np.ndarray:
    """Return one MeanMaxSim score for each matched query/document pair.

    Unique queries and evidence documents are encoded once. Long evidence is
    losslessly split into overlapping token chunks, then its chunk token vectors
    are concatenated before MaxSim, so the logical document remains one packet.
    """

    if len(queries) != len(documents):
        raise ValueError("late-interaction queries and documents must have equal length")
    if not queries:
        return np.empty(0, dtype=np.float32)
    clean_queries = [str(value).strip() for value in queries]
    clean_documents = [str(value).strip() for value in documents]
    if any(not value for value in clean_queries):
        raise ValueError("late-interaction queries must be nonempty")
    if any(not value for value in clean_documents):
        raise ValueError("late-interaction documents must be nonempty")
    if document_chunk_overlap_tokens < 0:
        raise ValueError("document_chunk_overlap_tokens must be nonnegative")

    with _ENCODER_LOCK:
        encoder = _load_encoder(str(model_name).strip(), str(device).strip())
        unique_queries = list(dict.fromkeys(clean_queries))
        unique_documents = list(dict.fromkeys(clean_documents))
        query_matrices = dict(zip(unique_queries, encoder.encode_queries(unique_queries)))

        chunks: list[str] = []
        chunk_spans: dict[str, tuple[int, int]] = {}
        document_length = int(encoder.document_length)
        overlap = min(
            int(document_chunk_overlap_tokens),
            max(0, document_length - 3),
        )
        encoding_prefix = str(getattr(encoder, "document_encoding_prefix", ""))
        for document in unique_documents:
            document_chunks = split_text_to_token_chunks(
                document,
                encoder.tokenizer,
                max_seq_length=document_length,
                chunk_overlap_tokens=overlap,
                encoding_prefix=encoding_prefix,
            )
            begin = len(chunks)
            chunks.extend(document_chunks)
            chunk_spans[document] = (begin, len(chunks))
        chunk_matrices = encoder.encode_documents(chunks)
        document_matrices = {
            document: np.concatenate(chunk_matrices[begin:end], axis=0)
            for document, (begin, end) in chunk_spans.items()
        }

        scores = np.empty(len(clean_queries), dtype=np.float32)
        pairs_by_document: dict[str, list[int]] = defaultdict(list)
        for index, document in enumerate(clean_documents):
            pairs_by_document[document].append(index)
        for document, indexes in pairs_by_document.items():
            document_scores = _score_token_matrices(
                [query_matrices[clean_queries[index]] for index in indexes],
                document_matrices[document],
            )
            scores[indexes] = document_scores
    if not np.isfinite(scores).all():
        raise RuntimeError("late-interaction scoring returned non-finite values")
    return scores


def clear_late_interaction_cache() -> None:
    """Release cached late-interaction encoders (primarily useful in tests)."""

    with _ENCODER_LOCK:
        _ENCODER_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
