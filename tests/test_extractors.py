"""Unit tests for feature extractors.

Tests each extractor's forward shape, fit_tokenizer, get_state, get_num_parameters.
LLM-based extractors are tested separately with @pytest.mark.slow.
"""

import json
import threading
import time

import pytest
import pandas as pd
import torch

# Sample texts for testing
SAMPLE_TEXTS = [
    "Patient is a 65 year old male with stage IV NSCLC diagnosed in January 2024.",
    "ECOG performance status 1. Started pembrolizumab 200mg IV every 3 weeks.",
    "CT scan shows partial response after 4 cycles of chemotherapy.",
    "Lab results: WBC 5.2, hemoglobin 12.1, platelets 180.",
]


class TestSimpleCNN:
    def test_forward_shape(self):
        from oci.models.simple_cnn_extractor import SimpleCNNExtractor

        ext = SimpleCNNExtractor(
            embedding_dim=32,
            conv_dim=32,
            kernel_size=3,
            num_conv_blocks=2,
            max_length=100,
            vocab_size=500,
            gated_attention_dim=16,
            projection_dim=24,
            dropout=0.0,
        )
        ext.fit_tokenizer(SAMPLE_TEXTS)
        out = ext(SAMPLE_TEXTS)
        assert out.shape == (4, 24)

    def test_fit_tokenizer_required(self):
        from oci.models.simple_cnn_extractor import SimpleCNNExtractor

        ext = SimpleCNNExtractor(vocab_size=500, projection_dim=24)
        with pytest.raises(RuntimeError, match="not fitted"):
            ext(SAMPLE_TEXTS)

    def test_get_state(self):
        from oci.models.simple_cnn_extractor import SimpleCNNExtractor

        ext = SimpleCNNExtractor(vocab_size=500, projection_dim=24)
        ext.fit_tokenizer(SAMPLE_TEXTS)
        state = ext.get_state()
        assert state["extractor_type"] == "simple_cnn"
        assert state["output_dim"] == 24
        assert state["tokenizer_state"] is not None

    def test_get_num_parameters(self):
        from oci.models.simple_cnn_extractor import SimpleCNNExtractor

        ext = SimpleCNNExtractor(vocab_size=500, projection_dim=24)
        params = ext.get_num_parameters()
        assert params["trainable"] > 0
        assert params["frozen"] == 0

    def test_dict_input(self):
        from oci.models.simple_cnn_extractor import SimpleCNNExtractor

        ext = SimpleCNNExtractor(
            embedding_dim=32,
            conv_dim=32,
            kernel_size=3,
            num_conv_blocks=2,
            max_length=100,
            vocab_size=500,
            projection_dim=24,
        )
        ext.fit_tokenizer(SAMPLE_TEXTS)
        out = ext({"texts": SAMPLE_TEXTS})
        assert out.shape == (4, 24)

    def test_tokenized_dict_input(self):
        from oci.data.collators import SimpleCNNTokenizingCollator
        from oci.data.dataset import ClinicalTextDataset
        from oci.models.simple_cnn_extractor import SimpleCNNExtractor

        ext = SimpleCNNExtractor(
            embedding_dim=32,
            conv_dim=32,
            kernel_size=3,
            num_conv_blocks=2,
            max_length=100,
            vocab_size=500,
            projection_dim=24,
            dropout=0.0,
        )
        ext.fit_tokenizer(SAMPLE_TEXTS)
        df = pd.DataFrame(
            {
                "clinical_text": SAMPLE_TEXTS,
                "outcome_indicator": [0, 1, 0, 1],
                "treatment_indicator": [1, 0, 1, 0],
            }
        )
        dataset = ClinicalTextDataset(
            df,
            text_column="clinical_text",
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
        )
        collator = SimpleCNNTokenizingCollator(ext._tokenizer, max_length=100)
        batch = collator([dataset[i] for i in range(len(dataset))])
        out = ext(batch)
        assert out.shape == (4, 24)
        assert batch["input_ids"].dim() == 2
        assert batch["attention_mask"].shape == batch["input_ids"].shape


class TestHierarchicalCNN:
    def test_forward_shape(self):
        from oci.models.hierarchical_cnn_extractor import HierarchicalCNNExtractor

        ext = HierarchicalCNNExtractor(
            embedding_dim=32,
            conv_dim=32,
            kernel_size=3,
            num_conv_blocks=2,
            chunk_size=20,
            chunk_overlap=4,
            max_chunks=8,
            vocab_size=500,
            gated_attention_dim=16,
            projection_dim=24,
        )
        ext.fit_tokenizer(SAMPLE_TEXTS)
        out = ext(SAMPLE_TEXTS)
        assert out.shape == (4, 24)

    def test_fit_tokenizer_required(self):
        from oci.models.hierarchical_cnn_extractor import HierarchicalCNNExtractor

        ext = HierarchicalCNNExtractor(vocab_size=500, projection_dim=24)
        with pytest.raises(RuntimeError, match="not fitted"):
            ext(SAMPLE_TEXTS)

    def test_get_state(self):
        from oci.models.hierarchical_cnn_extractor import HierarchicalCNNExtractor

        ext = HierarchicalCNNExtractor(vocab_size=500, projection_dim=24)
        ext.fit_tokenizer(SAMPLE_TEXTS)
        state = ext.get_state()
        assert state["extractor_type"] == "hierarchical_cnn"
        assert "chunk_size" in state

    def test_tokenized_dict_input(self):
        from oci.data.collators import HierarchicalCNNTokenizingCollator
        from oci.data.dataset import ClinicalTextDataset
        from oci.models.hierarchical_cnn_extractor import HierarchicalCNNExtractor

        ext = HierarchicalCNNExtractor(
            embedding_dim=32,
            conv_dim=32,
            kernel_size=3,
            num_conv_blocks=2,
            chunk_size=20,
            chunk_overlap=4,
            max_chunks=8,
            vocab_size=500,
            gated_attention_dim=16,
            projection_dim=24,
        )
        ext.fit_tokenizer(SAMPLE_TEXTS)
        df = pd.DataFrame(
            {
                "clinical_text": SAMPLE_TEXTS,
                "outcome_indicator": [0, 1, 0, 1],
                "treatment_indicator": [1, 0, 1, 0],
            }
        )
        dataset = ClinicalTextDataset(
            df,
            text_column="clinical_text",
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
        )
        collator = HierarchicalCNNTokenizingCollator(
            ext._tokenizer,
            chunk_size=20,
            chunk_overlap=4,
            max_chunks=8,
        )
        batch = collator([dataset[i] for i in range(len(dataset))])
        out = ext(batch)
        assert out.shape == (4, 24)
        assert batch["input_ids"].dim() == 3
        assert batch["chunk_mask"].shape == batch["input_ids"].shape[:2]


class TestHierarchicalGRU:
    def test_forward_shape(self):
        from oci.models.hierarchical_gru_extractor import HierarchicalGRUExtractor

        ext = HierarchicalGRUExtractor(
            embedding_dim=32,
            gru_hidden_dim=24,
            num_gru_layers=1,
            chunk_size=20,
            chunk_overlap=4,
            max_chunks=8,
            vocab_size=500,
            gated_attention_dim=16,
            projection_dim=24,
        )
        ext.fit_tokenizer(SAMPLE_TEXTS)
        out = ext(SAMPLE_TEXTS)
        assert out.shape == (4, 24)

    def test_fit_tokenizer_required(self):
        from oci.models.hierarchical_gru_extractor import HierarchicalGRUExtractor

        ext = HierarchicalGRUExtractor(vocab_size=500, projection_dim=24)
        with pytest.raises(RuntimeError, match="not fitted"):
            ext(SAMPLE_TEXTS)

    def test_get_state(self):
        from oci.models.hierarchical_gru_extractor import HierarchicalGRUExtractor

        ext = HierarchicalGRUExtractor(vocab_size=500, projection_dim=24)
        ext.fit_tokenizer(SAMPLE_TEXTS)
        state = ext.get_state()
        assert state["extractor_type"] == "hierarchical_gru"
        assert "gru_hidden_dim" in state


class TestHierarchicalTransformer:
    def test_forward_shape_with_hash_backend(self):
        from oci.models.hierarchical_transformer_extractor import (
            HierarchicalTransformerExtractor,
        )

        ext = HierarchicalTransformerExtractor(
            sentence_encoder_model="hash",
            chunk_size_words=8,
            chunk_overlap_words=2,
            max_chunks=6,
            num_transformer_layers=1,
            num_attention_heads=2,
            transformer_dim=32,
            projection_dim=24,
            hash_embedding_dim=32,
            transformer_dropout=0.0,
        )
        out = ext(SAMPLE_TEXTS)
        assert out.shape == (4, 24)

    def test_attention_evidence_includes_chunk_text(self):
        from oci.models.hierarchical_transformer_extractor import (
            HierarchicalTransformerExtractor,
        )

        ext = HierarchicalTransformerExtractor(
            sentence_encoder_model="hash",
            chunk_size_words=6,
            chunk_overlap_words=1,
            max_chunks=6,
            num_transformer_layers=1,
            num_attention_heads=2,
            transformer_dim=32,
            projection_dim=16,
            hash_embedding_dim=32,
            transformer_dropout=0.0,
        )
        evidence = ext.get_attention_evidence(
            SAMPLE_TEXTS[:2],
            row_ids=[10, 11],
            fold=2,
            stage="nuisance",
            top_k=2,
        )
        assert evidence
        assert {row["row_id"] for row in evidence} == {10, 11}
        assert all(row["stage"] == "nuisance" for row in evidence)
        assert all(isinstance(row["chunk_text"], str) for row in evidence)
        assert all(0.0 <= row["attention"] <= 1.0 for row in evidence)

    def test_attention_evidence_falls_back_to_shared_attention_without_role_heads(self):
        from oci.models.hierarchical_transformer_extractor import (
            HierarchicalTransformerExtractor,
        )

        ext = HierarchicalTransformerExtractor(
            sentence_encoder_model="hash",
            chunk_size_words=3,
            chunk_overlap_words=0,
            max_chunks=4,
            num_transformer_layers=1,
            num_attention_heads=2,
            transformer_dim=32,
            projection_dim=16,
            hash_embedding_dim=32,
            transformer_dropout=0.0,
            role_attention=False,
        )

        evidence = ext.get_attention_evidence(
            ["alpha beta gamma delta epsilon zeta eta theta"],
            row_ids=[99],
            fold=1,
            stage="effect_modifier",
            top_k=2,
        )

        assert evidence
        assert all(row["attention_role"] == "x" for row in evidence)
        assert any(row["attention"] > 0.0 for row in evidence)

    def test_split_text_into_word_chunks(self):
        from oci.models.hierarchical_transformer_extractor import split_text_into_word_chunks

        chunks = split_text_into_word_chunks(
            "one two three four five six seven eight nine",
            chunk_size_words=4,
            chunk_overlap_words=1,
            max_chunks=3,
        )
        assert chunks == [
            "one two three four",
            "four five six seven",
            "seven eight nine",
        ]

    def test_split_text_into_word_chunks_keeps_tail_when_truncated(self):
        from oci.models.hierarchical_transformer_extractor import split_text_into_word_chunks

        text = " ".join(f"w{i}" for i in range(1, 21))
        chunks = split_text_into_word_chunks(
            text,
            chunk_size_words=4,
            chunk_overlap_words=1,
            max_chunks=3,
        )
        assert chunks == [
            "w11 w12 w13 w14",
            "w14 w15 w16 w17",
            "w17 w18 w19 w20",
        ]

    def test_sentence_encoder_backend_and_pooling_defaults(self):
        from oci.models.hierarchical_transformer_extractor import (
            HierarchicalTransformerExtractor,
        )

        bert = HierarchicalTransformerExtractor(sentence_encoder_model="prajjwal1/bert-tiny")
        assert bert._effective_sentence_encoder_backend() == "transformers"
        assert bert._effective_sentence_pooling() == "cls"

        qwen = HierarchicalTransformerExtractor(sentence_encoder_model="Qwen/Qwen3-Embedding-0.6B")
        assert qwen._effective_sentence_encoder_backend() == "sentence_transformers"
        assert qwen._effective_sentence_pooling() == "last"

        trainable_qwen = HierarchicalTransformerExtractor(
            sentence_encoder_model="Qwen/Qwen3-Embedding-0.6B",
            trainable_sentence_encoder_layers=1,
        )
        assert trainable_qwen._effective_sentence_encoder_backend() == "transformers"

    def test_tokenizer_loader_falls_back_to_slow_tokenizer(self):
        from oci.models.hierarchical_transformer_extractor import (
            HierarchicalTransformerExtractor,
        )

        calls = []

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(model_name, use_fast=True):
                calls.append((model_name, use_fast))
                if use_fast:
                    raise ValueError("fast tokenizer conversion failed")
                return "slow-tokenizer"

        ext = HierarchicalTransformerExtractor(sentence_encoder_model="some-bert")
        tokenizer = ext._load_tokenizer(FakeAutoTokenizer)

        assert tokenizer == "slow-tokenizer"
        assert calls == [("some-bert", True), ("some-bert", False)]

    def test_tokenizer_loader_uses_legacy_bert_snapshot(self, tmp_path):
        from oci.models.hierarchical_transformer_extractor import (
            HierarchicalTransformerExtractor,
        )

        (tmp_path / "vocab.txt").write_text(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\npatient\n",
            encoding="utf-8",
        )

        class BrokenAutoTokenizer:
            @staticmethod
            def from_pretrained(model_name, use_fast=True):
                calls.append((model_name, use_fast))
                raise ValueError(f"auto tokenizer failed: {model_name} {use_fast}")

        calls = []
        ext = HierarchicalTransformerExtractor(sentence_encoder_model="prajjwal1/bert-tiny")
        ext._resolved_sentence_encoder_path = str(tmp_path)
        tokenizer = ext._load_tokenizer(BrokenAutoTokenizer)

        assert tokenizer.__class__.__name__ == "BertTokenizer"
        assert tokenizer.cls_token == "[CLS]"
        assert calls == []

    def test_model_loader_uses_legacy_bert_snapshot(self, tmp_path):
        import json
        from transformers import BertConfig, BertModel

        from oci.models.hierarchical_transformer_extractor import (
            HierarchicalTransformerExtractor,
        )

        calls = []

        class BrokenAutoModel:
            @staticmethod
            def from_pretrained(model_name):
                calls.append(("auto", model_name))
                raise ValueError("missing model_type")

        config = BertConfig(
            vocab_size=6,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=16,
        )
        BertModel(config).save_pretrained(tmp_path)
        config_path = tmp_path / "config.json"
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config_data.pop("model_type", None)
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        ext = HierarchicalTransformerExtractor(sentence_encoder_model="prajjwal1/bert-tiny")
        ext._resolved_sentence_encoder_path = str(tmp_path)
        model = ext._load_transformers_model(BrokenAutoModel)

        assert isinstance(model, BertModel)
        assert model.config.hidden_size == 8
        assert calls == []

    def test_model_loader_strips_legacy_bert_pretraining_heads(self, tmp_path):
        import json
        from transformers import BertConfig, BertForPreTraining, BertModel

        from oci.models.hierarchical_transformer_extractor import (
            HierarchicalTransformerExtractor,
        )

        calls = []

        class BrokenAutoModel:
            @staticmethod
            def from_pretrained(model_name):
                calls.append(("auto", model_name))
                raise ValueError("missing model_type")

        config = BertConfig(
            vocab_size=6,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=16,
        )
        BertForPreTraining(config).save_pretrained(tmp_path)
        config_path = tmp_path / "config.json"
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config_data.pop("model_type", None)
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        ext = HierarchicalTransformerExtractor(sentence_encoder_model="prajjwal1/bert-tiny")
        ext._resolved_sentence_encoder_path = str(tmp_path)
        model = ext._load_transformers_model(BrokenAutoModel)

        assert isinstance(model, BertModel)
        assert model.config.hidden_size == 8
        assert not any(key.startswith("cls.") for key in model.state_dict())
        assert calls == []

    def test_transformer_encoder_initialization_is_thread_serialized(self, monkeypatch):
        from oci.models.hierarchical_transformer_extractor import (
            HierarchicalTransformerExtractor,
        )

        active = 0
        max_active = 0
        calls = 0
        lock = threading.Lock()

        def fake_init(self):
            nonlocal active, max_active, calls
            with lock:
                active += 1
                calls += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            self._sentence_dim = self._hash_embedding_dim
            self._input_projection = torch.nn.Linear(
                self._sentence_dim,
                self._transformer_dim,
            )
            with lock:
                active -= 1

        monkeypatch.setattr(
            HierarchicalTransformerExtractor,
            "_ensure_transformers_initialized",
            fake_init,
        )
        extractors = [
            HierarchicalTransformerExtractor(sentence_encoder_model="some-bert") for _ in range(3)
        ]
        threads = [
            threading.Thread(target=extractor._ensure_encoder_initialized)
            for extractor in extractors
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert calls == 3
        assert max_active == 1
        assert all(extractor._encoder_initialized for extractor in extractors)

    def test_transformer_tokenization_is_cached(self):
        from types import SimpleNamespace

        from oci.models.hierarchical_transformer_extractor import (
            HierarchicalTransformerExtractor,
        )

        calls = []

        class FakeTokenizer:
            pad_token_id = 0
            padding_side = "right"

            def __call__(self, text, padding=False, truncation=True, max_length=None):
                del padding, truncation
                calls.append(text)
                token_ids = [101]
                token_ids.extend(range(10, 10 + min(2, len(text.split()))))
                token_ids.append(102)
                if max_length is not None:
                    token_ids = token_ids[:max_length]
                return {
                    "input_ids": token_ids,
                    "attention_mask": [1] * len(token_ids),
                }

        class FakeEncoder(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                del attention_mask
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
                return SimpleNamespace(last_hidden_state=hidden)

        ext = HierarchicalTransformerExtractor(
            sentence_encoder_model="some-bert",
            chunk_size_words=3,
            chunk_overlap_words=0,
            max_chunks=4,
            max_chunk_length=8,
            num_transformer_layers=1,
            num_attention_heads=2,
            transformer_dim=8,
            projection_dim=4,
            transformer_dropout=0.0,
        )
        ext._encoder_initialized = True
        ext._tokenizer = FakeTokenizer()
        ext._sentence_encoder = FakeEncoder()
        ext._sentence_dim = 4
        ext._input_projection = torch.nn.Linear(4, 8)

        texts = [
            "one two three four five six",
            "one two three four five six",
        ]
        first = ext(texts)
        second = ext(texts)

        assert first.shape == (2, 4)
        assert second.shape == (2, 4)
        assert calls == ["one two three", "four five six"]
        assert len(ext._chunk_cache) == 1
        assert len(ext._tokenization_cache) == 2

    def test_transformer_tokenization_rejects_overflow_without_truncating(self):
        from oci.models.hierarchical_transformer_extractor import (
            HierarchicalTransformerExtractor,
        )

        calls = []

        class OverflowTokenizer:
            pad_token_id = 0
            padding_side = "right"

            def __call__(self, text, **kwargs):
                calls.append((text, dict(kwargs)))
                return {
                    "input_ids": list(range(9)),
                    "attention_mask": [1] * 9,
                }

        ext = HierarchicalTransformerExtractor(
            sentence_encoder_model="some-bert",
            chunk_size_words=3,
            chunk_overlap_words=0,
            max_chunks=2,
            max_chunk_length=8,
            num_transformer_layers=1,
            num_attention_heads=2,
            transformer_dim=8,
            projection_dim=4,
        )
        ext._encoder_initialized = True
        ext._tokenizer = OverflowTokenizer()

        with pytest.raises(ValueError, match="semantic truncation is forbidden"):
            ext(["one two three"])
        assert calls == [
            (
                "one two three",
                {"padding": False, "truncation": False},
            )
        ]

    def test_token_attention_pooling_exports_token_spans(self):
        import json
        import re
        from types import SimpleNamespace

        from oci.models.hierarchical_transformer_extractor import (
            HierarchicalTransformerExtractor,
        )

        class FakeTokenizer:
            pad_token_id = 0
            padding_side = "right"
            is_fast = True
            all_special_ids = [0, 101, 102]
            all_special_tokens = ["[PAD]", "[CLS]", "[SEP]"]

            def __call__(
                self,
                text,
                padding=False,
                truncation=True,
                max_length=None,
                return_offsets_mapping=False,
            ):
                del padding, truncation
                words = list(re.finditer(r"\S+", text))
                input_ids = [101] + list(range(10, 10 + len(words))) + [102]
                attention_mask = [1] * len(input_ids)
                offsets = (
                    [(0, 0)]
                    + [(int(match.start()), int(match.end())) for match in words]
                    + [(0, 0)]
                )
                if max_length is not None:
                    input_ids = input_ids[:max_length]
                    attention_mask = attention_mask[:max_length]
                    offsets = offsets[:max_length]
                result = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                if return_offsets_mapping:
                    result["offset_mapping"] = offsets
                return result

            def convert_ids_to_tokens(self, input_ids):
                tokens = []
                for token_id in input_ids:
                    if token_id == 101:
                        tokens.append("[CLS]")
                    elif token_id == 102:
                        tokens.append("[SEP]")
                    elif token_id == 0:
                        tokens.append("[PAD]")
                    else:
                        tokens.append(f"tok{token_id}")
                return tokens

        class FakeEncoder(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                del attention_mask
                hidden = input_ids.float().unsqueeze(-1)
                hidden = torch.cat(
                    [
                        hidden,
                        hidden / 10.0,
                        torch.sin(hidden),
                        torch.cos(hidden),
                    ],
                    dim=-1,
                )
                return SimpleNamespace(last_hidden_state=hidden)

        ext = HierarchicalTransformerExtractor(
            sentence_encoder_model="some-bert",
            sentence_pooling="token_attention",
            chunk_size_words=10,
            chunk_overlap_words=0,
            max_chunks=2,
            max_chunk_length=16,
            num_transformer_layers=1,
            num_attention_heads=2,
            transformer_dim=8,
            projection_dim=4,
            transformer_dropout=0.0,
        )
        ext._encoder_initialized = True
        ext._tokenizer = FakeTokenizer()
        ext._sentence_encoder = FakeEncoder()
        ext._sentence_dim = 4
        ext._input_projection = torch.nn.Linear(4, 8)
        ext._ensure_token_pooling_initialized()

        evidence = ext.get_attention_evidence(
            ["Encounter Record Timepoint: At age 84 PD-L1 1-49 percent"],
            row_ids=[123],
            fold=1,
            stage="nuisance",
            top_k=1,
        )

        assert evidence
        row = evidence[0]
        assert row["row_id"] == 123
        assert "top_token_spans_json" in row
        spans = json.loads(row["top_token_spans_json"])
        assert spans
        assert {"text", "focus_token", "token_attention", "salience"}.issubset(spans[0])
        assert row["attended_token_summary"]
        assert "[[" in row["highlighted_chunk_text"]

        features, attention_info = ext(
            ["Encounter Record Timepoint: At age 84 PD-L1 1-49 percent"],
            return_attention_tensors=True,
        )
        assert features.shape == (1, 4)
        assert attention_info["token_alpha"] is not None
        assert attention_info["token_alpha_sources"]
        features.square().sum().backward()
        assert attention_info["token_alpha_sources"][0].grad is not None

    def test_role_attention_has_distinct_token_and_chunk_heads(self):
        import re
        from types import SimpleNamespace

        from oci.models.hierarchical_transformer_extractor import (
            HierarchicalTransformerExtractor,
        )

        class FakeTokenizer:
            pad_token_id = 0
            padding_side = "right"
            is_fast = True
            all_special_ids = [0, 101, 102]
            all_special_tokens = ["[PAD]", "[CLS]", "[SEP]"]

            def __call__(
                self,
                text,
                padding=False,
                truncation=True,
                max_length=None,
                return_offsets_mapping=False,
            ):
                del padding, truncation
                words = list(re.finditer(r"\S+", text))
                input_ids = [101] + list(range(10, 10 + len(words))) + [102]
                attention_mask = [1] * len(input_ids)
                offsets = (
                    [(0, 0)]
                    + [(int(match.start()), int(match.end())) for match in words]
                    + [(0, 0)]
                )
                if max_length is not None:
                    input_ids = input_ids[:max_length]
                    attention_mask = attention_mask[:max_length]
                    offsets = offsets[:max_length]
                result = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                if return_offsets_mapping:
                    result["offset_mapping"] = offsets
                return result

            def convert_ids_to_tokens(self, input_ids):
                return [f"tok{token_id}" for token_id in input_ids]

        class FakeEncoder(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                del attention_mask
                hidden = input_ids.float().unsqueeze(-1)
                hidden = torch.cat(
                    [
                        hidden,
                        hidden / 10.0,
                        torch.sin(hidden),
                        torch.cos(hidden),
                    ],
                    dim=-1,
                )
                return SimpleNamespace(last_hidden_state=hidden)

        ext = HierarchicalTransformerExtractor(
            sentence_encoder_model="some-bert",
            sentence_pooling="token_attention",
            role_attention=True,
            w_attention_heads=2,
            x_attention_heads=3,
            chunk_size_words=3,
            chunk_overlap_words=0,
            max_chunks=4,
            max_chunk_length=8,
            num_transformer_layers=1,
            num_attention_heads=2,
            transformer_dim=8,
            projection_dim=5,
            transformer_dropout=0.0,
        )
        ext._encoder_initialized = True
        ext._tokenizer = FakeTokenizer()
        ext._sentence_encoder = FakeEncoder()
        ext._sentence_dim = 4
        ext._input_projection = torch.nn.Linear(4, 8)
        ext._ensure_token_pooling_initialized()

        texts = ["alpha beta gamma delta epsilon zeta"]
        role_features, attention_info = ext(
            texts,
            return_role_features=True,
            return_attention_tensors=True,
        )

        assert set(role_features) == {"features", "w_features", "x_features"}
        assert role_features["features"].shape == (1, 5)
        assert role_features["w_features"].shape == (1, 5)
        assert role_features["x_features"].shape == (1, 5)
        assert ext._w_token_pooling.num_heads == 2
        assert ext._x_token_pooling.num_heads == 3
        assert ext._w_chunk_pooling.num_heads == 2
        assert ext._x_chunk_pooling.num_heads == 3
        assert attention_info["role_token_alpha"]["w"].shape[0] == 2
        assert attention_info["role_token_alpha"]["x"].shape[0] == 2
        assert attention_info["role_chunk_alpha"]["w"].shape == (1, 2)
        assert attention_info["role_chunk_alpha"]["x"].shape == (1, 2)

        nuisance = ext.get_attention_evidence(texts, stage="nuisance", top_k=1)
        effect = ext.get_attention_evidence(texts, stage="effect_modifier", top_k=1)

        assert nuisance and nuisance[0]["attention_role"] == "w"
        assert effect and effect[0]["attention_role"] == "x"


class TestNeuralCausalForest:
    def _tokenizer_test_encoder(self, model_name="some-bert"):
        from oci.models.neural_causal_forest_extractor import (
            HierarchicalTokenAttentionEncoder,
            NeuralCausalForestConfig,
        )

        encoder = object.__new__(HierarchicalTokenAttentionEncoder)
        encoder.config = NeuralCausalForestConfig(encoder_model_name=model_name)
        encoder._resolved_encoder_model_path = None
        return encoder

    def test_tokenizer_loader_falls_back_to_slow_tokenizer(self):
        calls = []

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(model_name, use_fast=True):
                calls.append((model_name, use_fast))
                if use_fast:
                    raise ValueError("fast tokenizer conversion failed")
                return "slow-tokenizer"

        encoder = self._tokenizer_test_encoder("some-bert")
        tokenizer = encoder._load_tokenizer(FakeAutoTokenizer)

        assert tokenizer == "slow-tokenizer"
        assert calls == [("some-bert", True), ("some-bert", False)]

    def test_tokenizer_loader_uses_legacy_bert_snapshot(self, tmp_path):
        (tmp_path / "vocab.txt").write_text(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\npatient\n",
            encoding="utf-8",
        )

        class BrokenAutoTokenizer:
            @staticmethod
            def from_pretrained(model_name, use_fast=True):
                calls.append((model_name, use_fast))
                raise ValueError(f"auto tokenizer failed: {model_name} {use_fast}")

        calls = []
        encoder = self._tokenizer_test_encoder("prajjwal1/bert-tiny")
        encoder._resolved_encoder_model_path = str(tmp_path)
        tokenizer = encoder._load_tokenizer(BrokenAutoTokenizer)

        assert tokenizer.__class__.__name__ == "BertTokenizer"
        assert tokenizer.cls_token == "[CLS]"
        assert calls == []

    def _ncf_hash_config(self):
        from oci.models.neural_causal_forest_extractor import NeuralCausalForestConfig

        return NeuralCausalForestConfig(
            encoder_architecture="ncf_token_attention",
            encoder_backend="hash",
            representation_dim=8,
            token_attention_dim=8,
            chunk_attention_dim=8,
            nuisance_hidden_dim=8,
            chunk_size_words=8,
            chunk_overlap_words=2,
            max_chunks=4,
        )

    def _htr_hash_config(self):
        from oci.models.neural_causal_forest_extractor import NeuralCausalForestConfig

        return NeuralCausalForestConfig(
            encoder_architecture="hierarchical_transformer",
            encoder_model_name="hash",
            encoder_backend="hash",
            representation_dim=8,
            token_attention_dim=8,
            chunk_attention_dim=8,
            nuisance_hidden_dim=8,
            chunk_size_words=8,
            chunk_overlap_words=2,
            max_chunks=4,
            htr_num_layers=1,
            htr_num_heads=2,
            htr_transformer_dim=8,
            htr_sentence_pooling="auto",
            htr_sentence_encoder_backend="auto",
            htr_hash_embedding_dim=8,
        )

    def test_nuisance_model_uses_htr_encoder_by_default(self):
        from oci.models.neural_causal_forest_extractor import (
            HTRGradientAttentionEncoder,
            NuisanceTextModel,
        )

        config = self._htr_hash_config()
        model = NuisanceTextModel(config, device="cpu", outcome_type="binary")
        out = model(SAMPLE_TEXTS[:2])

        assert isinstance(model.encoder, HTRGradientAttentionEncoder)
        assert out["propensity_logit"].shape == (2,)
        assert out["outcome_raw"].shape == (2,)

    def test_ncf_nuisance_defaults_are_calibration_oriented(self):
        from oci.models.neural_causal_forest_extractor import NeuralCausalForestConfig

        config = NeuralCausalForestConfig()

        assert config.nuisance_epochs == 20
        assert config.nuisance_weight_decay == pytest.approx(0.05)
        assert config.nuisance_label_smoothing == pytest.approx(0.02)
        assert config.nuisance_calibration == "temperature_isotonic"

    def test_ncf_config_loader_ignores_unknown_fields(self, tmp_path):
        from oci.models.neural_causal_forest_extractor import NeuralCausalForestConfig

        path = tmp_path / "ncf_config.json"
        path.write_text(
            json.dumps(
                {
                    "encoder_backend": "hash",
                    "encoder_model_name": "hash",
                    "future_nuisance_only_field": 123,
                }
            ),
            encoding="utf-8",
        )

        config = NeuralCausalForestConfig.from_json(path)

        assert config.encoder_backend == "hash"
        assert not hasattr(config, "future_nuisance_only_field")

    def test_ncf_token_attention_encoder_still_selectable(self):
        from oci.models.neural_causal_forest_extractor import (
            HierarchicalTokenAttentionEncoder,
            NuisanceTextModel,
        )

        config = self._ncf_hash_config()
        model = NuisanceTextModel(config, device="cpu", outcome_type="binary")

        assert isinstance(model.encoder, HierarchicalTokenAttentionEncoder)

    def test_inner_fold_parallelism_resolver_is_conservative_on_cuda(self):
        from oci.models.neural_causal_forest_extractor import (
            _resolve_inner_fold_parallelism,
        )

        config = self._ncf_hash_config()
        config.inner_fold_parallelism = "auto"
        config.num_workers = 3

        assert _resolve_inner_fold_parallelism(config, 5, torch.device("cpu")) == 3
        assert _resolve_inner_fold_parallelism(config, 5, torch.device("cuda:0")) == 1

    def test_explicit_inner_fold_parallelism_overrides_cuda_serial_default(self):
        from oci.models.neural_causal_forest_extractor import (
            _resolve_inner_fold_parallelism,
        )

        config = self._ncf_hash_config()
        config.inner_fold_parallelism = "2"
        config.num_workers = 0

        assert _resolve_inner_fold_parallelism(config, 5, torch.device("cuda:0")) == 2

    def test_ncf_nuisance_attention_schema(self):
        from oci.models.neural_causal_forest_extractor import (
            NuisanceTextModel,
            nuisance_attention_evidence,
        )

        config = self._ncf_hash_config()
        model = NuisanceTextModel(config, device="cpu", outcome_type="binary")
        rows = nuisance_attention_evidence(
            model,
            SAMPLE_TEXTS[:2],
            row_ids=[10, 11],
            config=config,
            top_k=2,
            metadata=[{"nuisance_fold": 1}, {"nuisance_fold": 1}],
        )

        assert rows
        assert {row["row_id"] for row in rows} == {10, 11}
        assert all("token_text" in row for row in rows)
        assert all("evidence_score" in row for row in rows)
        assert all("snippet" in row for row in rows)
        assert all(row["nuisance_fold"] == 1 for row in rows)


class TestLearnedTokenizer:
    def test_fit_and_encode(self):
        from oci.models.learned_tokenizer import LearnedTokenizer

        tok = LearnedTokenizer()
        tok.fit(SAMPLE_TEXTS, vocab_size=200, min_freq=1)
        assert tok.vocab_size > 2  # at least PAD and UNK
        ids = tok.encode("patient is stage IV", max_length=10)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_batch(self):
        from oci.models.learned_tokenizer import LearnedTokenizer

        tok = LearnedTokenizer()
        tok.fit(SAMPLE_TEXTS, vocab_size=200, min_freq=1)
        input_ids, mask = tok.encode_batch(SAMPLE_TEXTS[:2], max_length=20)
        assert input_ids.shape[0] == 2
        assert mask.shape == input_ids.shape

    def test_state_roundtrip(self):
        from oci.models.learned_tokenizer import LearnedTokenizer

        tok = LearnedTokenizer()
        tok.fit(SAMPLE_TEXTS, vocab_size=200, min_freq=1)
        state = tok.get_state()

        tok2 = LearnedTokenizer()
        tok2.load_state(state)
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.encode("test", 5) == tok.encode("test", 5)

    def test_unk_token(self):
        from oci.models.learned_tokenizer import LearnedTokenizer

        tok = LearnedTokenizer()
        tok.fit(["hello world"], vocab_size=100, min_freq=1)
        ids = tok.encode("xyzzy_unknown_word", max_length=5)
        assert tok.unk_token_id in ids
