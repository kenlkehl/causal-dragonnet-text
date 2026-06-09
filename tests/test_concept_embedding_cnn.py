import numpy as np
import pytest
import torch

from oci.models.concept_embedding_utils import chunk_text_words


class FakeSentenceEncoder:
    def __init__(self, dim=6):
        self.dim = dim

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(
        self,
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        del batch_size, convert_to_numpy, show_progress_bar
        rows = []
        for text in texts:
            base = sum(ord(ch) for ch in text)
            row = np.array(
                [((base + 17 * i) % 97) / 97.0 for i in range(self.dim)],
                dtype=np.float32,
            )
            if normalize_embeddings:
                row = row / max(np.linalg.norm(row), 1e-12)
            rows.append(row)
        return np.vstack(rows)


class RecordingSentenceEncoder(FakeSentenceEncoder):
    def __init__(self, dim=6):
        super().__init__(dim=dim)
        self.batch_lengths = []

    def encode(self, texts, **kwargs):
        self.batch_lengths.append(len(texts))
        return super().encode(texts, **kwargs)


class OOMThenSingleSentenceEncoder(FakeSentenceEncoder):
    def __init__(self, dim=6):
        super().__init__(dim=dim)
        self.batch_lengths = []

    def encode(self, texts, **kwargs):
        self.batch_lengths.append(len(texts))
        if len(texts) > 1:
            raise RuntimeError("CUDA out of memory while testing")
        return super().encode(texts, **kwargs)


def test_chunk_text_words_overlap():
    chunks = chunk_text_words(
        "one two three four five six",
        chunk_size_words=3,
        chunk_overlap_words=1,
        max_chunks=10,
    )
    assert chunks == ["one two three", "three four five", "five six"]


def test_concept_embedding_cnn_cached_forward_and_anchor():
    from oci.models.concept_embedding_cnn_extractor import ConceptEmbeddingCNNExtractor

    extractor = ConceptEmbeddingCNNExtractor(
        sentence_encoder=FakeSentenceEncoder(dim=6),
        cached_embedding_dim=6,
        confounder_concepts=["patient age"],
        effect_modifier_concepts=["PD-L1 expression"],
        random_features=2,
        kernel_role="combined",
        projection_dim=5,
        anchor_weight=0.5,
        dropout=0.0,
    )
    batch = {
        "cached_hidden_states": torch.randn(4, 7, 6),
        "cached_attention_mask": torch.ones(4, 7),
    }
    out = extractor(batch)
    assert out.shape == (4, 5)
    assert extractor.get_state()["num_concept_features"] == 2
    assert extractor.get_state()["num_random_features"] == 2
    assert torch.isclose(extractor.compute_anchor_loss(), torch.tensor(0.0))

    with torch.no_grad():
        extractor._concept_conv.weight[0, 0, 0] += 1.0
    assert extractor.compute_anchor_loss() > 0


def test_concept_embedding_cnn_role_specific_counts():
    from oci.models.concept_embedding_cnn_extractor import ConceptEmbeddingCNNExtractor

    conf = ConceptEmbeddingCNNExtractor(
        sentence_encoder=FakeSentenceEncoder(dim=4),
        cached_embedding_dim=4,
        confounder_concepts=["age", "ECOG"],
        effect_modifier_concepts=["histology"],
        random_confounder_features=3,
        random_modifier_features=5,
        kernel_role="confounder",
        projection_dim=4,
    )
    mod = ConceptEmbeddingCNNExtractor(
        sentence_encoder=FakeSentenceEncoder(dim=4),
        cached_embedding_dim=4,
        confounder_concepts=["age", "ECOG"],
        effect_modifier_concepts=["histology"],
        random_confounder_features=3,
        random_modifier_features=5,
        kernel_role="effect_modifier",
        projection_dim=4,
    )
    assert conf.get_state()["num_concept_features"] == 2
    assert conf.get_state()["num_random_features"] == 3
    assert mod.get_state()["num_concept_features"] == 1
    assert mod.get_state()["num_random_features"] == 5


def test_concept_embedding_cache_roundtrip(tmp_path, monkeypatch):
    import oci.models.concept_embedding_cache as cache_mod

    encoder = RecordingSentenceEncoder(dim=5)
    monkeypatch.setattr(
        cache_mod,
        "load_sentence_transformer",
        lambda model_name, device=None: encoder,
    )
    cache = cache_mod.ConceptEmbeddingCache(
        cache_dir=str(tmp_path),
        sentence_model_name="fake-model",
        dataset_path=str(tmp_path / "dataset.parquet"),
        chunk_size_words=2,
        chunk_overlap_words=1,
        max_chunks=4,
    )
    texts = ["alpha beta gamma", "delta epsilon"]
    cache.precompute(texts, device=None, batch_size=2)
    assert cache.is_valid(expected_num_samples=2)
    cache.open()
    assert cache.hidden_size == 5
    assert cache.hidden_states_array[0].shape == (3, 5)
    assert cache.attention_mask_array[1].shape == (2,)
    assert encoder.batch_lengths == [2, 2, 1]


def test_concept_embedding_cache_reduces_batch_size_on_cuda_oom(tmp_path, monkeypatch):
    import oci.models.concept_embedding_cache as cache_mod

    encoder = OOMThenSingleSentenceEncoder(dim=4)
    monkeypatch.setattr(
        cache_mod,
        "load_sentence_transformer",
        lambda model_name, device=None: encoder,
    )
    cache = cache_mod.ConceptEmbeddingCache(
        cache_dir=str(tmp_path),
        sentence_model_name="fake-model",
        dataset_path=str(tmp_path / "dataset.parquet"),
        chunk_size_words=2,
        chunk_overlap_words=1,
        max_chunks=4,
    )
    cache.precompute(["alpha beta gamma", "delta epsilon"], device=None, batch_size=4)

    assert cache.is_valid(expected_num_samples=2)
    assert encoder.batch_lengths[:3] == [4, 2, 1]
    assert max(length for length in encoder.batch_lengths[2:]) == 1


def test_concept_embedding_cache_multi_gpu_shards_chunks(tmp_path, monkeypatch):
    import oci.models.concept_embedding_cache as cache_mod

    encoders = {}

    def load_encoder(model_name, device=None):
        del model_name
        key = str(device)
        if key not in encoders:
            encoders[key] = RecordingSentenceEncoder(dim=5)
        return encoders[key]

    monkeypatch.setattr(cache_mod, "load_sentence_transformer", load_encoder)
    cache = cache_mod.ConceptEmbeddingCache(
        cache_dir=str(tmp_path),
        sentence_model_name="fake-model",
        dataset_path=str(tmp_path / "dataset.parquet"),
        chunk_size_words=2,
        chunk_overlap_words=0,
        max_chunks=3,
    )
    texts = [
        "alpha beta gamma delta",
        "epsilon zeta eta theta",
        "iota kappa",
    ]
    cache.precompute_multi_gpu(
        texts,
        devices=[torch.device("cuda:0"), torch.device("cuda:1")],
        batch_size=2,
    )

    assert cache.is_valid(expected_num_samples=3)
    assert set(encoders) == {"cuda:0", "cuda:1"}
    assert sum(sum(encoder.batch_lengths) for encoder in encoders.values()) == 5
    cache.open()
    assert cache.hidden_states_array[0].shape == (2, 5)
    assert cache.hidden_states_array[1].shape == (2, 5)
    assert cache.hidden_states_array[2].shape == (1, 5)
    assert cache.chunk_counts == [2, 2, 1]
    assert cache._metadata["num_gpus_used"] == 2


def test_load_sentence_transformer_forces_float32(monkeypatch):
    import sentence_transformers
    import oci.models.concept_embedding_cache as cache_mod

    calls = []

    class DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
            self.float_called = False
            self.eval_called = False

        def float(self):
            self.float_called = True
            return self

        def eval(self):
            self.eval_called = True
            return self

    cache_mod.clear_sentence_transformer_cache()
    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        DummySentenceTransformer,
    )

    encoder = cache_mod.load_sentence_transformer(
        "fake-model",
        device=torch.device("cuda:0"),
    )

    assert calls[0][0] == ("fake-model",)
    assert calls[0][1]["device"] == "cuda:0"
    assert calls[0][1]["model_kwargs"]["torch_dtype"] is torch.float32
    assert encoder.float_called
    assert encoder.eval_called
    cache_mod.clear_sentence_transformer_cache()


def test_factory_normalizes_concept_embedding_alias(monkeypatch):
    import oci.models.concept_embedding_cnn_extractor as extractor_mod
    from oci.config import normalize_feature_extractor_type
    from oci.models.extractor_factory import create_feature_extractor

    monkeypatch.setattr(
        extractor_mod,
        "load_sentence_transformer",
        lambda model_name, device=None: FakeSentenceEncoder(dim=4),
    )
    assert normalize_feature_extractor_type("cecnn") == "concept_embedding_cnn"
    extractor = create_feature_extractor(
        extractor_type="cecnn",
        device=torch.device("cpu"),
        cecnn_confounder_concepts=["age"],
        cecnn_effect_modifier_concepts=["PD-L1"],
        cecnn_random_features=1,
        cecnn_projection_dim=3,
    )
    assert extractor.output_dim == 3
