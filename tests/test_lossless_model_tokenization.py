from __future__ import annotations

import ast
from pathlib import Path

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
        rows = [list(range(1, len(str(value).split()) + 3)) for value in values]
        lengths = [len(row) for row in rows]
        if padding:
            target = int(max_length) if padding == "max_length" else max(lengths, default=0)
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


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def num_special_tokens_to_add(self, pair=False):
        del pair
        return 0


class _VerboseWarningCharacterTokenizer(_CharacterTokenizer):
    def __init__(self) -> None:
        self.verbose_values: list[bool] = []

    def encode(self, text, add_special_tokens=False, verbose=True):
        import warnings

        self.verbose_values.append(bool(verbose))
        if verbose and len(text) > 8:
            warnings.warn("sequence length exceeds model_max_length", UserWarning)
        return super().encode(text, add_special_tokens=add_special_tokens)


def test_lossless_tokenizer_never_enables_truncation():
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


def test_lossless_tokenizer_rejects_a_binding_limit():
    with pytest.raises(SemanticTruncationError, match="semantic truncation is forbidden"):
        tokenize_losslessly(
            _Tokenizer(),
            ["one two", "one two three"],
            configured_max_length=3,
            context="test input",
            padding=True,
            return_tensors="pt",
        )


def test_overlapping_chunk_capacity_accounts_for_overlap():
    assert required_overlapping_chunks(10, chunk_size=4, chunk_overlap=1) == 3
    with pytest.raises(SemanticTruncationError, match="requires 4 chunks"):
        require_nonbinding_chunk_capacity(
            11,
            chunk_size=4,
            chunk_overlap=1,
            max_chunks=3,
            context="test document",
        )


def test_concept_word_chunking_rejects_prefix_or_suffix_selection():
    from oci.models.concept_embedding_utils import chunk_text_words

    text = " ".join(f"word{index}" for index in range(10))
    for selection in ("first", "last"):
        with pytest.raises(SemanticTruncationError, match="requires 4 chunks"):
            chunk_text_words(
                text,
                chunk_size_words=3,
                chunk_overlap_words=0,
                max_chunks=2,
                chunk_selection=selection,
            )


def test_concept_embedding_cache_rejects_token_chunk_overflow(tmp_path):
    from oci.models.concept_embedding_cache import ConceptEmbeddingCache

    cache = ConceptEmbeddingCache(
        cache_dir=str(tmp_path),
        sentence_model_name="unused-model",
        dataset_path=str(tmp_path / "dataset.parquet"),
        chunk_size_words=100,
        chunk_overlap_words=0,
        max_chunks=1,
    )
    with pytest.raises(SemanticTruncationError, match="token-bounded chunk plan"):
        cache._chunk_texts(
            ["abcdefghi"],
            tokenizer=_CharacterTokenizer(),
            max_seq_length=5,
        )


def test_token_chunk_boundaries_suppress_length_warning_without_truncating():
    import warnings

    from oci.models.concept_embedding_utils import split_text_to_token_chunks

    tokenizer = _VerboseWarningCharacterTokenizer()
    text = "abcdefghijklmnopqrstuvwxyz"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chunks = split_text_to_token_chunks(
            text,
            tokenizer,
            max_seq_length=7,
            chunk_overlap_tokens=0,
        )

    assert tokenizer.verbose_values == [False]
    assert "".join(chunks) == text
    assert max(map(len, chunks)) <= 7


def test_production_sources_have_no_literal_truncation_true():
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
                    offenders.append(f"{path.relative_to(repository)}:{node.lineno}")
    assert offenders == []
