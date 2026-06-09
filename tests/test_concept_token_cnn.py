import numpy as np
import torch


class FakeTokenEncoder:
    hidden_size = 4

    def __init__(self):
        self.vocab = {
            "alpha": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "beta": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "gamma": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "delta": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "noise": np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }

    def get_hidden_size(self):
        return self.hidden_size

    def encode_token_sequences(
        self,
        texts,
        add_special_tokens=True,
        max_length=None,
        normalize_embeddings=True,
    ):
        del add_special_tokens, normalize_embeddings
        result = []
        for text in texts:
            tokens = str(text).split()
            if max_length is not None:
                tokens = tokens[:max_length]
            rows = [
                self.vocab.get(token, np.zeros(self.hidden_size, dtype=np.float32))
                for token in tokens
            ]
            result.append(np.vstack(rows).astype(np.float32))
        return result


def _extractor(**kwargs):
    from oci.models.concept_token_cnn_extractor import ConceptTokenCNNExtractor

    params = dict(
        token_encoder=FakeTokenEncoder(),
        cached_hidden_size=4,
        confounder_concepts=["alpha beta"],
        effect_modifier_concepts=[],
        random_features=0,
        projection_dim=3,
        dropout=0.0,
        anchor_weight=0.5,
        chunk_size=2,
        max_chunks=4,
        normalize_embeddings=True,
        device=torch.device("cpu"),
    )
    params.update(kwargs)
    return ConceptTokenCNNExtractor(**params)


def test_concept_token_cnn_exact_window_response():
    extractor = _extractor()
    encoder = FakeTokenEncoder()
    sample_match = encoder.encode_token_sequences(["alpha beta gamma"])[0]
    sample_miss = encoder.encode_token_sequences(["alpha gamma beta"])[0]
    hidden = torch.tensor(np.stack([sample_match, sample_miss]), dtype=torch.float32)
    batch = {
        "cached_hidden_states": hidden,
        "cached_attention_mask": torch.ones(2, 3),
    }

    out = extractor(batch)

    assert out.shape == (2, 3)
    responses = extractor._last_response_maps[0]
    assert torch.isclose(responses[0, 0], torch.tensor(1.0))
    assert responses[0].max() > responses[1].max()
    assert extractor.get_state()["kernel_lengths"] == [2]


def test_concept_token_cnn_masks_chunk_boundary_windows():
    extractor = _extractor(chunk_size=2)
    encoder = FakeTokenEncoder()
    sequence = encoder.encode_token_sequences(["noise alpha beta delta"])[0]
    batch = {
        "cached_hidden_states": torch.tensor(sequence[None, :, :], dtype=torch.float32),
        "cached_attention_mask": torch.ones(1, 4),
        "sample_chunk_counts": [2],
    }

    extractor(batch)

    valid = extractor._last_valid_window_masks[0]
    responses = extractor._last_response_maps[0]
    assert valid.tolist() == [[True, False, True]]
    assert torch.isclose(responses[0, 1], torch.tensor(1.0))


def test_concept_token_cnn_anchor_loss_tracks_concept_filters():
    extractor = _extractor()
    assert torch.isclose(extractor.compute_anchor_loss(), torch.tensor(0.0))

    with torch.no_grad():
        extractor._filters[0][0, 0] += 1.0

    assert extractor.compute_anchor_loss() > 0


def test_concept_token_factory_alias_without_llm():
    from oci.config import normalize_feature_extractor_type
    from oci.models.extractor_factory import create_feature_extractor

    assert normalize_feature_extractor_type("ctcnn") == "concept_token_cnn"
    extractor = create_feature_extractor(
        extractor_type="ctcnn",
        device=torch.device("cpu"),
        ctcnn_random_features=2,
        ctcnn_cached_hidden_size=4,
        ctcnn_projection_dim=3,
    )
    assert extractor.output_dim == 3
    assert extractor.get_state()["num_random_features"] == 2


def test_cached_hidden_state_collator_preserves_explicit_mask():
    from oci.data.cached_hidden_state_dataset import collate_cached_batch

    batch = [
        {
            "text": "a",
            "outcome": torch.tensor(0.0),
            "treatment": torch.tensor(1.0),
            "text_id": 0,
            "hidden_states": np.ones((4, 3), dtype=np.float16),
            "attention_mask": np.array([1, 1, 0, 0], dtype=np.uint8),
        },
        {
            "text": "b",
            "outcome": torch.tensor(1.0),
            "treatment": torch.tensor(0.0),
            "text_id": 1,
            "hidden_states": np.ones((2, 3), dtype=np.float16),
            "attention_mask": np.array([1, 0], dtype=np.uint8),
        },
    ]

    result = collate_cached_batch(batch)

    assert result["cached_attention_mask"].tolist() == [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
