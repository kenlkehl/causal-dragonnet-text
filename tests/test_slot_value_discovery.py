import numpy as np
import torch


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
            base = sum(ord(ch) for ch in str(text))
            row = np.array(
                [((base + 17 * i) % 97) / 97.0 for i in range(self.dim)],
                dtype=np.float32,
            )
            if normalize_embeddings:
                row = row / max(np.linalg.norm(row), 1e-12)
            rows.append(row)
        return np.vstack(rows)


def test_slot_value_extractor_cached_forward_and_anchor():
    from oci.models.slot_value_discovery_extractor import SlotValueDiscoveryExtractor

    extractor = SlotValueDiscoveryExtractor(
        sentence_encoder=FakeSentenceEncoder(dim=6),
        cached_embedding_dim=6,
        confounder_concepts=["patient age"],
        effect_modifier_concepts=["PD-L1 expression"],
        num_free_slots=2,
        slot_dim=5,
        num_value_prototypes=3,
        dropout=0.0,
        anchor_weight=0.5,
        attention_entropy_weight=0.01,
        query_diversity_weight=0.01,
        device=torch.device("cpu"),
    )
    batch = {
        "texts": [
            "Age 72 years. PD-L1 80% positive.",
            "Age 55 years. PD-L1 0% negative.",
        ],
        "cached_hidden_states": torch.randn(2, 4, 6),
        "cached_attention_mask": torch.ones(2, 4),
    }

    out = extractor(batch)

    assert out.shape == (2, 20)
    assert extractor.get_state()["num_seed_slots"] == 2
    assert extractor.get_state()["num_free_slots"] == 2
    assert torch.isclose(extractor.compute_anchor_loss(), torch.tensor(0.0))
    reg = extractor.compute_regularization_losses()
    assert "slot_attention_entropy_loss" in reg
    assert "slot_query_diversity_loss" in reg

    with torch.no_grad():
        extractor._queries[0, 0] += 1.0
    assert extractor.compute_anchor_loss() > 0


def test_slot_value_features_capture_generic_values():
    from oci.models.slot_value_discovery_extractor import (
        VALUE_FEATURE_NAMES,
        value_features_for_chunk,
    )

    features = value_features_for_chunk("PD-L1 >= 50% positive; no brain mets")
    mapping = dict(zip(VALUE_FEATURE_NAMES, features))

    assert mapping["has_number"] == 1.0
    assert mapping["has_percent"] == 1.0
    assert mapping["first_percent"] == 0.5
    assert mapping["has_high_comparator"] == 1.0
    assert mapping["has_negation"] == 1.0
    assert mapping["has_positive"] == 1.0


def test_slot_value_factory_alias_without_seed_llm():
    from oci.config import normalize_feature_extractor_type
    from oci.models.extractor_factory import create_feature_extractor

    assert normalize_feature_extractor_type("svx") == "slot_value_discovery"
    extractor = create_feature_extractor(
        extractor_type="svx",
        device=torch.device("cpu"),
        svx_num_free_slots=3,
        svx_cached_embedding_dim=4,
        svx_slot_dim=5,
        svx_num_value_prototypes=2,
    )
    assert extractor.output_dim == 15
    assert extractor.get_state()["num_free_slots"] == 3


def test_slot_value_rlearner_gradients_reach_queries_and_gates():
    from oci.models.causal_text import CausalText

    model = CausalText(
        feature_extractor_type="slot_value_discovery",
        model_type="rlearner",
        svx_num_free_slots=2,
        svx_cached_embedding_dim=4,
        svx_slot_dim=6,
        svx_num_value_prototypes=2,
        svx_anchor_weight=0.0,
        svx_attention_entropy_weight=0.01,
        svx_query_diversity_weight=0.01,
        svx_gate_l1_weight=0.01,
        causal_head_representation_dim=8,
        causal_head_hidden_outcome_dim=6,
        causal_head_dropout=0.0,
        device="cpu",
    )
    batch = {
        "texts": [
            "Age 72 years. marker 80% positive.",
            "Age 50 years. marker 0% negative.",
            "Age 66 years. marker unknown.",
        ],
        "cached_hidden_states": torch.randn(3, 5, 4),
        "cached_attention_mask": torch.ones(3, 5),
        "treatment": torch.tensor([1.0, 0.0, 1.0]),
        "outcome": torch.tensor([1.0, 0.0, 1.0]),
    }

    losses = model.train_step(batch, gamma_rlearner=1.0)
    losses["loss"].backward()

    assert model.feature_extractor._queries.grad is not None
    assert model.feature_extractor._queries.grad.abs().sum() > 0
    assert model.net.nuisance_gate_logits.grad is not None
    assert model.net.nuisance_gate_logits.grad.abs().sum() > 0
    assert model.net.effect_gate_logits.grad is not None
    assert model.net.effect_gate_logits.grad.abs().sum() > 0
    assert "slot_gate_l1_loss" in losses
