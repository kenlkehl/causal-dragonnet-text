from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from oci.models.lossless_tokenization import (
    SemanticTruncationError,
    require_nonbinding_chunk_capacity,
    required_overlapping_chunks,
    tokenize_losslessly,
)


class _Tokenizer:
    pad_token = "[PAD]"
    pad_token_id = 0
    eos_token = "[EOS]"
    eos_token_id = 0

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        texts,
        *,
        padding=False,
        truncation=None,
        max_length=None,
        return_tensors=None,
        return_length=False,
        **_kwargs,
    ):
        self.calls.append(
            {
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
            }
        )
        is_single = isinstance(texts, str)
        values = [texts] if is_single else list(texts)
        rows = [
            list(range(1, len(str(value).split()) + 3))
            for value in values
        ]
        lengths = [len(row) for row in rows]
        if padding:
            if padding == "max_length":
                target = int(max_length)
            else:
                target = max(lengths, default=0)
            rows = [row + [0] * (target - len(row)) for row in rows]
        masks = [
            [1] * length + [0] * (len(row) - length)
            for row, length in zip(rows, lengths)
        ]
        result = {
            "input_ids": rows[0] if is_single and return_tensors is None else rows,
            "attention_mask": masks[0] if is_single and return_tensors is None else masks,
        }
        if return_length:
            result["length"] = lengths[0] if is_single else lengths
        if return_tensors == "pt":
            result["input_ids"] = torch.as_tensor(rows, dtype=torch.long)
            result["attention_mask"] = torch.as_tensor(masks, dtype=torch.long)
        return result


def test_lossless_tokenizer_never_enables_truncation() -> None:
    tokenizer = _Tokenizer()

    encoded = tokenize_losslessly(
        tokenizer,
        ["one two", "three"],
        configured_max_length=4,
        context="test input",
        padding=True,
        return_tensors="pt",
    )

    assert tuple(encoded["input_ids"].shape) == (2, 4)
    assert tokenizer.calls == [
        {"padding": True, "truncation": False, "max_length": None}
    ]


def test_lossless_tokenizer_rejects_a_binding_configured_limit() -> None:
    tokenizer = _Tokenizer()

    with pytest.raises(
        SemanticTruncationError,
        match=r"configured_max_length=3.*semantic truncation is forbidden",
    ):
        tokenize_losslessly(
            tokenizer,
            ["one two", "one two three"],
            configured_max_length=3,
            context="test input",
            padding=True,
            return_tensors="pt",
        )

    assert tokenizer.calls[0]["truncation"] is False


def test_overlapping_chunk_capacity_accounts_for_overlap() -> None:
    assert required_overlapping_chunks(
        10,
        chunk_size=4,
        chunk_overlap=1,
    ) == 3
    with pytest.raises(
        SemanticTruncationError,
        match=r"requires 4 chunks.*max_chunks=3",
    ):
        require_nonbinding_chunk_capacity(
            11,
            chunk_size=4,
            chunk_overlap=1,
            max_chunks=3,
            context="test document",
        )


def test_neural_causal_forest_word_chunking_fails_closed() -> None:
    from oci.models.neural_causal_forest_extractor import (
        split_text_into_word_span_chunks,
    )

    text = " ".join(f"w{index}" for index in range(11))
    with pytest.raises(
        SemanticTruncationError,
        match=r"NeuralCausalForest note requires 4 chunks",
    ):
        split_text_into_word_span_chunks(
            text,
            chunk_size_words=4,
            chunk_overlap_words=1,
            max_chunks=3,
        )


def test_concept_word_chunking_rejects_first_and_last_selection() -> None:
    from oci.models.concept_embedding_utils import chunk_text_words

    text = " ".join(f"w{index}" for index in range(10))
    for selection in ("first", "last"):
        with pytest.raises(
            SemanticTruncationError,
            match=r"requires 4 chunks.*max_chunks=2",
        ):
            chunk_text_words(
                text,
                chunk_size_words=3,
                chunk_overlap_words=0,
                max_chunks=2,
                chunk_selection=selection,
            )


def test_learned_tokenizer_rejects_binding_max_length() -> None:
    from oci.models.learned_tokenizer import LearnedTokenizer

    tokenizer = LearnedTokenizer()
    tokenizer.fit(["one two three"], vocab_size=10, min_freq=1)

    with pytest.raises(
        SemanticTruncationError,
        match=r"semantic truncation is forbidden \(3 > 2\)",
    ):
        tokenizer.encode("one two three", max_length=2)


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def num_special_tokens_to_add(self, pair=False):
        del pair
        return 0


def test_concept_embedding_cache_rejects_token_split_chunk_overflow(
    tmp_path,
) -> None:
    from oci.models.concept_embedding_cache import ConceptEmbeddingCache

    cache = ConceptEmbeddingCache(
        cache_dir=str(tmp_path),
        sentence_model_name="unused-model",
        dataset_path=str(tmp_path / "dataset.parquet"),
        chunk_size_words=100,
        chunk_overlap_words=0,
        max_chunks=1,
    )

    with pytest.raises(
        SemanticTruncationError,
        match=r"token-bounded chunk plan.*\(2 > 1\)",
    ):
        cache._chunk_texts(
            ["abcdefghi"],
            tokenizer=_CharacterTokenizer(),
            max_seq_length=5,
        )


def test_slot_value_alignment_rejects_chunk_clipping() -> None:
    from oci.models.slot_value_discovery_extractor import (
        SlotValueDiscoveryExtractor,
    )

    extractor = SlotValueDiscoveryExtractor(
        chunk_size_words=2,
        chunk_overlap_words=0,
        max_chunks=4,
        num_free_slots=1,
        slot_dim=2,
        cached_embedding_dim=2,
    )

    with pytest.raises(
        SemanticTruncationError,
        match=r"value-feature chunk capacity.*\(2 > 1\)",
    ):
        extractor._chunk_texts_for_values(["one two three"], max_len=1)


def test_frozen_llm_pooler_rejects_binding_limit_before_model_forward() -> None:
    from oci.models.frozen_llm_pooler_extractor import FrozenLLMPoolerExtractor

    extractor = FrozenLLMPoolerExtractor(
        max_length=3,
        skip_llm=True,
        cached_hidden_size=4,
        gated_attention_dim=2,
        projection_dim=2,
    )
    extractor._tokenizer = _Tokenizer()
    extractor._model = object()

    with pytest.raises(SemanticTruncationError):
        extractor._forward_from_texts(["one two"])


def test_hierarchical_llm_rejects_binding_chunk_cap_before_model_forward() -> None:
    from oci.models.hierarchical_llm_extractor import HierarchicalLLMExtractor

    extractor = HierarchicalLLMExtractor(
        chunk_size=4,
        chunk_overlap=1,
        max_chunks=1,
        skip_llm=True,
        cached_hidden_size=4,
        gated_attention_dim=2,
        projection_dim=2,
    )
    extractor._tokenizer = _Tokenizer()
    extractor._model = object()

    with pytest.raises(SemanticTruncationError, match=r"requires 2 chunks"):
        extractor._forward_from_texts(["one two three"])


def test_hierarchical_transformer_rejects_negative_overlap_before_word_omission() -> None:
    from oci.models.hierarchical_transformer_extractor import (
        HierarchicalTransformerExtractor,
        split_text_into_word_chunks,
    )

    with pytest.raises(ValueError, match="nonnegative"):
        split_text_into_word_chunks(
            "one two three four",
            chunk_size_words=2,
            chunk_overlap_words=-1,
            max_chunks=4,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        HierarchicalTransformerExtractor(
            sentence_encoder_model="hash",
            chunk_size_words=2,
            chunk_overlap_words=-1,
            max_chunks=4,
            max_chunk_length=4,
        )


def test_gpu_store_estimate_rejects_binding_limit(monkeypatch) -> None:
    import transformers

    from oci.models.gpu_hidden_state_store import GPUHiddenStateStore

    tokenizer = _Tokenizer()
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(hidden_size=4),
    )

    with pytest.raises(SemanticTruncationError):
        GPUHiddenStateStore.estimate_vram_gb(
            ["one two"],
            "unused-model",
            max_length=3,
        )


def test_hidden_state_cache_rejects_binding_limit_before_model_load(
    monkeypatch,
    tmp_path,
) -> None:
    import transformers

    from oci.models.hidden_state_cache import HiddenStateCache

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _Tokenizer(),
    )
    cache = HiddenStateCache(
        cache_dir=str(tmp_path / "cache"),
        model_name="unused-model",
        max_length=3,
        dataset_path=str(tmp_path / "dataset.parquet"),
    )

    with pytest.raises(SemanticTruncationError):
        cache.precompute(["one two"], torch.device("cpu"))
    assert not cache.cache_path.exists()


def test_concept_token_encoder_rejects_binding_limit_before_model_forward() -> None:
    from oci.models.concept_token_cnn_extractor import LLMTokenHiddenStateEncoder

    encoder = LLMTokenHiddenStateEncoder("unused-model", torch.device("cpu"))
    encoder._tokenizer = _Tokenizer()
    encoder._model = object()
    encoder._compute_dtype = torch.float32

    with pytest.raises(SemanticTruncationError):
        encoder.encode_token_sequences(["one two"], max_length=3)


def test_hidden_state_cache_identity_excludes_old_truncating_policy(tmp_path) -> None:
    from oci.models.hidden_state_cache import HiddenStateCache

    first = HiddenStateCache.compute_cache_hash(
        "model",
        17,
        str(tmp_path / "dataset.parquet"),
    )
    second = HiddenStateCache.compute_cache_hash(
        "model",
        17,
        str(tmp_path / "dataset.parquet"),
    )

    assert first == second
    assert len(first) == 12


def test_production_sources_have_no_literal_truncation_true() -> None:
    repository = Path(__file__).resolve().parents[1]
    source_paths = tuple((repository / "oci").rglob("*.py")) + tuple(
        (repository / "scripts").rglob("*.py")
    )
    offenders: list[str] = []
    for path in sorted(source_paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "truncation"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    offenders.append(
                        f"{path.relative_to(repository).as_posix()}:{node.lineno}"
                    )
    assert offenders == []
