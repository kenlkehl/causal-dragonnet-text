from __future__ import annotations

import numpy as np

from oci.models import late_interaction


def test_score_token_matrices_computes_mean_maxsim():
    queries = [
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    ]
    document = np.asarray([[1.0, 0.0], [0.0, 0.5]], dtype=np.float32)

    scores = late_interaction._score_token_matrices(queries, document)

    assert scores.tolist() == [0.75, 1.0]


def test_late_interaction_pair_scorer_deduplicates_and_chunks(monkeypatch):
    class Tokenizer:
        def __init__(self):
            self.tokens = {}
            self.reverse = {}

        def encode(self, text, add_special_tokens=False):
            values = []
            for token in text.split():
                if token not in self.tokens:
                    token_id = len(self.tokens) + 1
                    self.tokens[token] = token_id
                    self.reverse[token_id] = token
                values.append(self.tokens[token])
            return values

        def decode(self, token_ids, **_kwargs):
            return " ".join(self.reverse[token_id] for token_id in token_ids)

        def num_special_tokens_to_add(self, pair=False):
            return 2

    class Encoder:
        document_length = 6

        def __init__(self):
            self.tokenizer = Tokenizer()
            self.query_calls = []
            self.document_calls = []

        def encode_queries(self, texts):
            self.query_calls.append(list(texts))
            return [
                np.asarray(
                    [[1.0, 0.0] if text == "Age" else [0.0, 1.0]],
                    dtype=np.float32,
                )
                for text in texts
            ]

        def encode_documents(self, texts):
            self.document_calls.append(list(texts))
            return [
                np.asarray(
                    [[1.0, 0.0] if "age" in text else [0.0, 1.0]],
                    dtype=np.float32,
                )
                for text in texts
            ]

    encoder = Encoder()
    monkeypatch.setattr(late_interaction, "_load_encoder", lambda *_args: encoder)

    scores = late_interaction.score_late_interaction_pairs(
        ["Age", "Age", "Renal"],
        [
            "age one two three four five six seven",
            "renal evidence",
            "renal evidence",
        ],
        "test-model",
        "cpu",
        document_chunk_overlap_tokens=1,
    )

    assert scores.tolist() == [1.0, 0.0, 1.0]
    assert encoder.query_calls == [["Age", "Renal"]]
    assert len(encoder.document_calls) == 1
    assert len(encoder.document_calls[0]) > 2
