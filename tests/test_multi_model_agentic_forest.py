import json
from pathlib import Path

import numpy as np
import pandas as pd

from oci.config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    BoWViewConfig,
    EmbeddingContrastDiscoveryConfig,
    ExperimentConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    ModelArchitectureConfig,
    MultiModelAgenticForestConfig,
)
from oci.inference.agentic_explicit_feature_forest import (
    AgenticFeatureProposal,
    SplitEvaluation,
)
from oci.inference.embedding_contrast_discovery import (
    EmbeddingContrastEvidenceGenerator,
    _default_embedding_cache_dir,
    _informative_chunk_text,
    redact_embedding_contrast_evidence,
)
from oci.inference.multi_model_agentic_forest import (
    MultiModelAgenticForestRunner,
    _candidate_consistency_threshold,
    _compact_multi_model_agent_context,
    _evaluate_extracted_feature_set_diagnostic,
    _extracted_feature_review_gate,
    _feature_redundancy_review,
    _fallback_consistency_proposals,
    _fit_binary_bow_fold,
    run_multi_model_agentic_forest,
)
from oci.models.concept_embedding_utils import chunk_text_words


def _linear_test_bow_views() -> list[BoWViewConfig]:
    return [
        BoWViewConfig(
            name="linear_test",
            max_features=1000,
            min_df=1,
            max_df=1.0,
            ngram_range_min=1,
            ngram_range_max=3,
        )
    ]


def _two_linear_test_bow_views() -> list[BoWViewConfig]:
    return [
        BoWViewConfig(
            name="linear_test_unigram",
            max_features=1000,
            min_df=1,
            max_df=1.0,
            ngram_range_min=1,
            ngram_range_max=1,
        ),
        BoWViewConfig(
            name="linear_test_bigram",
            max_features=1000,
            min_df=1,
            max_df=1.0,
            ngram_range_min=1,
            ngram_range_max=2,
        ),
    ]


def _disable_htr_test_kwargs() -> dict:
    return {
        "htr_evidence_enabled": False,
        "htr_evidence_disable_reason": "unit test without HTR training",
    }


def _disable_required_evidence_test_kwargs() -> dict:
    return {
        "embedding_contrast": EmbeddingContrastDiscoveryConfig(
            enabled=False,
            disable_reason="unit test without embedding contrast",
        ),
        **_disable_htr_test_kwargs(),
    }


class FakeProposalAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        if context.get("prompt_version") == "multi_model_agentic_alias_resolution_v1":
            return {
                "groups": [
                    {
                        "canonical_name": "pd_l1_expression",
                        "member_names": [
                            "pd_l1_expression",
                            "pd_l1_expression_level",
                        ],
                        "type": "categorical",
                        "categories": ["<1%", "1-49%", ">=50%"],
                        "description": "Pretreatment tumor PD-L1 expression category.",
                        "roles": ["effect_modifier"],
                        "rationale": "The two names refer to the same extraction target.",
                    }
                ],
                "unmerged": [{"name": "age", "reason": "No alias proposed."}],
            }
        if context.get("prompt_version") == "multi_model_agentic_value_harmonization_v1":
            return {
                "features": [
                    {
                        "name": "age",
                        "type": "continuous",
                        "categories": None,
                        "description": "Patient age at treatment initiation in years.",
                        "missing_values": ["unknown", "not_reported", "high", "low"],
                        "rationale": "Age should remain numeric; qualitative labels are missing.",
                    },
                    {
                        "name": "pd_l1_expression",
                        "type": "categorical",
                        "categories": ["<1%", "1-49%", ">=50%", "unknown"],
                        "description": "Pretreatment tumor PD-L1 expression category.",
                        "value_aliases": {
                            "<1%": ["low negative"],
                            ">=50%": ["high", "50% or greater"],
                        },
                        "missing_values": ["unknown", "not_reported"],
                        "rationale": "Collapse high/low aliases into threshold categories.",
                    },
                ]
            }
        return [
            {
                "action": "add",
                "name": "age",
                "type": "continuous",
                "roles": ["confounder"],
                "description": "Patient age at treatment initiation in years.",
                "rationale": "Age-bearing terms appear in treatment and outcome models.",
                "expected_signal": "treatment and outcome",
            },
            {
                "action": "add",
                "name": "pd_l1_expression",
                "type": "categorical",
                "categories": ["low", "high", "unknown"],
                "roles": ["effect_modifier"],
                "description": "Pretreatment tumor PD-L1 expression category.",
                "rationale": "PD-L1 threshold terms appear in the pseudo-target model.",
                "expected_signal": "pseudo-target",
            },
            {
                "action": "add",
                "name": "pd_l1_expression_level",
                "type": "categorical",
                "categories": ["<1%", "1-49%", ">=50%"],
                "roles": ["effect_modifier"],
                "description": "Pretreatment tumor PD-L1 expression category.",
                "rationale": "PD-L1 threshold terms appear in the pseudo-target model.",
                "expected_signal": "pseudo-target",
            },
        ]


class ReviewRevisionAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        prompt_version = context.get("prompt_version")
        if prompt_version == "multi_model_agentic_value_harmonization_v1":
            return {"features": context.get("selected_features", [])}
        if prompt_version == "multi_model_agentic_extracted_feature_review_v1":
            return [
                {
                    "action": "add",
                    "name": "signal_marker",
                    "type": "categorical",
                    "categories": ["negative", "positive"],
                    "roles": ["confounder", "effect_modifier"],
                    "description": "Pretreatment signal marker status.",
                    "rationale": "BoW diagnostics show signal marker text captures the missed treatment and outcome signal.",
                    "expected_signal": "treatment, outcome, and pseudo-target",
                }
            ]
        return [
            {
                "action": "add",
                "name": "noise_marker",
                "type": "categorical",
                "categories": ["absent", "present"],
                "roles": ["confounder"],
                "description": "Pretreatment noise marker status.",
                "rationale": "Initial weak candidate.",
                "expected_signal": "treatment and outcome",
            }
        ]


class FakeExtractionProvider:
    def ensure_features(self, dataset, specs):
        dataset = dataset.copy()
        text = dataset["clinical_text"].astype(str)
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if spec.name == "age":
                dataset[value_col] = text.str.extract(r"age (\d+)").astype(float)
            elif spec.name == "pd_l1_expression":
                dataset[value_col] = np.where(
                    text.str.contains(">=50%"),
                    ">=50%",
                    np.where(text.str.contains("1-49%"), "1-49%", "<1%"),
                )
            else:
                dataset[value_col] = np.nan
            dataset[missing_col] = dataset[value_col].isna()
        return dataset


class ReviewExtractionProvider:
    def __init__(self):
        self.calls = []

    def ensure_features(self, dataset, specs):
        self.calls.append([spec.name for spec in specs])
        dataset = dataset.copy()
        text = dataset["clinical_text"].astype(str)
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if spec.name == "noise_marker":
                dataset[value_col] = "present"
            elif spec.name == "signal_marker":
                dataset[value_col] = np.where(
                    text.str.contains("signal positive"),
                    "positive",
                    "negative",
                )
            else:
                dataset[value_col] = np.nan
            dataset[missing_col] = dataset[value_col].isna()
        return dataset


class FakeEvaluator:
    def __init__(self):
        self.seen_specs = []

    def evaluate_split(self, train_df, test_df, specs, fold_id):
        self.seen_specs.append(specs)
        predictions = test_df.copy()
        predictions["pred_ite_prob"] = 0.1
        predictions["pred_y0_prob"] = 0.4
        predictions["pred_y1_prob"] = 0.5
        predictions["pred_propensity_prob"] = 0.5
        predictions["pred_outcome_prob"] = 0.5
        predictions["cv_fold"] = fold_id
        metrics = {
            "fold": fold_id,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_explicit_features": len(specs),
        }
        return SplitEvaluation(predictions=predictions, metrics=metrics)


class FakeHTREvidenceProvider:
    def __init__(self):
        self.seen_effect_nuisance_predictions = []

    def fit_nuisance(self, discovery_df, outer_fold):
        y = discovery_df["outcome_indicator"].to_numpy(dtype=float)
        t = discovery_df["treatment_indicator"].to_numpy(dtype=float)
        e_hat = np.clip(0.25 + 0.45 * t, 0.05, 0.95)
        m_hat = np.clip(0.20 + 0.50 * y, 0.05, 0.95)
        y_resid = y - m_hat
        t_resid = t - e_hat
        predictions = pd.DataFrame(
            {
                "_oci_row_id": discovery_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "e_hat": e_hat,
                "m_hat": m_hat,
                "y_residual": y_resid,
                "t_residual": t_resid,
                "r_pseudo_outcome": y_resid / np.where(np.abs(t_resid) < 1e-6, 1e-6, t_resid),
                "r_loss_at_zero_tau": y_resid**2,
                "nuisance_fold": 1,
            }
        )
        return {
            "predictions": predictions,
            "attention": self._attention_rows(discovery_df, outer_fold, "nuisance", e_hat=e_hat, m_hat=m_hat),
        }

    def fit_effect(self, discovery_df, nuisance_predictions, outer_fold):
        self.seen_effect_nuisance_predictions.append(nuisance_predictions.copy())
        predictions = nuisance_predictions.copy()
        tau_hat = np.linspace(0.10, 0.35, len(predictions))
        y_resid = predictions["y_residual"].to_numpy(dtype=float)
        t_resid = predictions["t_residual"].to_numpy(dtype=float)
        predictions["tau_hat_r_stage"] = tau_hat
        predictions["tau_logit_modifier"] = np.nan
        predictions["r_loss"] = (y_resid - tau_hat * t_resid) ** 2
        predictions["effect_loss"] = predictions["r_loss"]
        predictions["effect_loss_at_zero_tau"] = y_resid**2
        predictions["effect_fold"] = 1
        predictions["effect_objective"] = "squared_r_loss"
        predictions["r_stage_train_eligible"] = True
        return {
            "predictions": predictions,
            "attention": self._attention_rows(discovery_df, outer_fold, "effect_modifier", tau_hat_r_stage=tau_hat),
        }

    def _attention_rows(self, discovery_df, outer_fold, stage, **extra):
        rows = []
        for offset, row in discovery_df.head(4).reset_index(drop=True).iterrows():
            text = str(row["clinical_text"])
            token = text.split()[0] if text.split() else "note"
            start = max(0, text.find(token))
            evidence = {
                "row_id": int(row["_oci_row_id"]),
                "outer_fold": int(outer_fold),
                "fold": 1,
                "stage": stage,
                "chunk_index": 0,
                "chunk_text": text,
                "attended_token_summary": token,
                "top_token_spans_json": json.dumps(
                    [
                        {
                            "text": token,
                            "focus_token": token,
                            "char_start": start,
                            "char_end": start + len(token),
                            "salience": 0.9,
                        }
                    ]
                ),
            }
            for key, values in extra.items():
                evidence[key] = values[offset]
            rows.append(evidence)
        return rows


class EmptyProposalAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        if context.get("prompt_version") == "multi_model_agentic_value_harmonization_v1":
            return {"features": context.get("selected_features", [])}
        return []


class FailingConsistencyAgent:
    def propose(self, context):
        if context.get("prompt_version") == "multi_model_agentic_consistency_v1":
            raise AssertionError("consistency agent should not select features")
        return []


class RecordingExtractionProvider:
    def __init__(self):
        self.calls = []

    def ensure_features(self, dataset, specs):
        self.calls.append([(spec.name, tuple(spec.roles)) for spec in specs])
        dataset = dataset.copy()
        text = dataset["clinical_text"].astype(str)
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if spec.name == "age":
                dataset[value_col] = text.str.extract(r"age (\d+)").astype(float)
            elif spec.name == "biomarker":
                dataset[value_col] = np.where(
                    text.str.contains("biomarker positive"),
                    "positive",
                    "negative",
                )
            else:
                dataset[value_col] = np.nan
            dataset[missing_col] = dataset[value_col].isna()
        return dataset


class KeywordEmbeddingProvider:
    def encode_chunks(self, texts):
        rows = []
        for text in texts:
            lower = str(text).lower()
            row = np.zeros(8, dtype=np.float32)
            for token, index in {
                "brain": 0,
                "liver": 1,
                "cachexia": 2,
                "pd-l1": 3,
                "pdl1": 3,
                "high": 4,
                "low": 5,
                "age": 6,
            }.items():
                if token in lower:
                    row[index] += 1.0
            row[7] = 0.1
            rows.append(row)
        return np.vstack(rows)


class KeywordSentenceTransformer:
    def encode(
        self,
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        del batch_size, convert_to_numpy, show_progress_bar
        embeddings = KeywordEmbeddingProvider().encode_chunks(texts)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings.astype(np.float32, copy=False)


def _embedding_contrast_cache_test_config(tmp_path: Path) -> AppliedInferenceConfig:
    return AppliedInferenceConfig(
        dataset_path=str(tmp_path / "cohort.parquet"),
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=True,
                    model_name="fake-keyword",
                    cache_dir=str(tmp_path / "embedding_cache"),
                    chunk_size_words=4,
                    chunk_overlap_words=0,
                    max_chunks=2,
                    min_probe_auc=0.0,
                    top_k_chunks_per_tail=3,
                    max_chunks_per_patient=1,
                    concept_phrases=["brain metastases", "liver lesion"],
                    include_bow_phrases_as_concepts=False,
                ),
                **_disable_htr_test_kwargs(),
            ),
        ),
    )


def _embedding_contrast_cache_test_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_oci_row_id": np.arange(8),
            "clinical_text": [
                "brain metastases pd-l1 high",
                "brain metastases pd-l1 high",
                "liver lesion pd-l1 low",
                "liver lesion pd-l1 low",
                "brain metastases cachexia pd-l1 high",
                "liver lesion stable disease",
                "brain metastases cachexia",
                "liver lesion stable disease",
            ],
        }
    )


def _build_embedding_contrast_cache_test_evidence(
    generator: EmbeddingContrastEvidenceGenerator,
    dataset: pd.DataFrame,
):
    return generator.build_evidence(
        discovery_df=dataset,
        y=np.asarray([0, 1, 0, 0, 1, 0, 1, 0], dtype=float),
        t=np.asarray([1, 1, 0, 0, 1, 0, 1, 0], dtype=float),
        pseudo_target=np.asarray([1, 1, -1, -1, 1, -1, 1, -1], dtype=float),
        t_resid=np.ones(8, dtype=float),
        importance={},
    )


def test_embedding_contrast_retrieval_filters_low_content_chunks():
    assert not _informative_chunk_text("")
    assert not _informative_chunk_text("--- ### <new_note> ---")
    assert _informative_chunk_text("Brain MRI shows enhancing metastases.")


def test_embedding_contrast_chunk_selection_keeps_last_chunks():
    text = " ".join(f"w{i}" for i in range(1, 11))
    first = chunk_text_words(text, chunk_size_words=3, chunk_overlap_words=0, max_chunks=2)
    last = chunk_text_words(
        text,
        chunk_size_words=3,
        chunk_overlap_words=0,
        max_chunks=2,
        chunk_selection="last",
    )
    assert first == ["w1 w2 w3", "w4 w5 w6"]
    assert last == ["w7 w8 w9", "w10"]


def test_embedding_contrast_default_cache_dir_is_dataset_scoped(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    parquet_path = dataset_dir / "cohort.parquet"
    assert _default_embedding_cache_dir(
        str(parquet_path),
        tmp_path / "run",
    ) == dataset_dir / ".oci_cache" / "embedding_contrast"
    assert _default_embedding_cache_dir(
        str(dataset_dir),
        tmp_path / "run",
    ) == dataset_dir / ".oci_cache" / "embedding_contrast"


def test_embedding_contrast_reuses_warm_chunk_and_concept_phrase_caches_without_model_load(
    tmp_path: Path,
    monkeypatch,
):
    dataset = _embedding_contrast_cache_test_dataset()
    config = _embedding_contrast_cache_test_config(tmp_path)
    load_calls = []

    def fake_load_sentence_transformer(model_name, device=None):
        load_calls.append((model_name, str(device)))
        return KeywordSentenceTransformer()

    monkeypatch.setattr(
        "oci.models.concept_embedding_cache.load_sentence_transformer",
        fake_load_sentence_transformer,
    )
    monkeypatch.setattr(
        "oci.inference.embedding_contrast_discovery.load_sentence_transformer",
        fake_load_sentence_transformer,
    )

    generator = EmbeddingContrastEvidenceGenerator(config=config, output_dir=tmp_path)
    generator.prepare(dataset)
    evidence = _build_embedding_contrast_cache_test_evidence(generator, dataset)
    assert "concept_probe_skipped" not in evidence
    treatment = next(item for item in evidence["contrasts"] if item["name"] == "treatment")
    assert treatment["concept_probe_scores"]
    assert load_calls

    def fail_load_sentence_transformer(model_name, device=None):
        del model_name, device
        raise AssertionError("sentence-transformer should not load on warm caches")

    monkeypatch.setattr(
        "oci.models.concept_embedding_cache.load_sentence_transformer",
        fail_load_sentence_transformer,
    )
    monkeypatch.setattr(
        "oci.inference.embedding_contrast_discovery.load_sentence_transformer",
        fail_load_sentence_transformer,
    )

    warm_generator = EmbeddingContrastEvidenceGenerator(
        config=config,
        output_dir=tmp_path,
    )
    warm_generator.prepare(dataset)
    warm_evidence = _build_embedding_contrast_cache_test_evidence(
        warm_generator,
        dataset,
    )
    assert "concept_probe_skipped" not in warm_evidence
    warm_treatment = next(
        item for item in warm_evidence["contrasts"] if item["name"] == "treatment"
    )
    assert warm_treatment["concept_probe_scores"]


def test_embedding_contrast_skips_concept_probe_load_when_only_chunk_cache_is_warm(
    tmp_path: Path,
    monkeypatch,
):
    dataset = _embedding_contrast_cache_test_dataset()
    config = _embedding_contrast_cache_test_config(tmp_path)
    load_calls = []

    def fake_load_sentence_transformer(model_name, device=None):
        load_calls.append((model_name, str(device)))
        return KeywordSentenceTransformer()

    monkeypatch.setattr(
        "oci.models.concept_embedding_cache.load_sentence_transformer",
        fake_load_sentence_transformer,
    )
    generator = EmbeddingContrastEvidenceGenerator(config=config, output_dir=tmp_path)
    generator.prepare(dataset)
    assert load_calls

    def fail_load_sentence_transformer(model_name, device=None):
        del model_name, device
        raise AssertionError("sentence-transformer should not load for optional probes")

    monkeypatch.setattr(
        "oci.models.concept_embedding_cache.load_sentence_transformer",
        fail_load_sentence_transformer,
    )
    monkeypatch.setattr(
        "oci.inference.embedding_contrast_discovery.load_sentence_transformer",
        fail_load_sentence_transformer,
    )

    warm_generator = EmbeddingContrastEvidenceGenerator(
        config=config,
        output_dir=tmp_path,
    )
    warm_generator.prepare(dataset)
    evidence = _build_embedding_contrast_cache_test_evidence(warm_generator, dataset)
    assert (
        evidence["concept_probe_skipped"]
        == "concept_phrase_cache_miss_on_warm_chunk_cache"
    )
    treatment = next(item for item in evidence["contrasts"] if item["name"] == "treatment")
    assert treatment["concept_probe_scores"] == []


def test_embedding_contrast_evidence_retrieves_aligned_chunks(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "_oci_row_id": np.arange(8),
            "clinical_text": [
                "brain metastases pd-l1 high",
                "brain metastases pd-l1 high",
                "liver lesion pd-l1 low",
                "liver lesion pd-l1 low",
                "brain metastases cachexia pd-l1 high",
                "liver lesion stable disease",
                "brain metastases cachexia",
                "liver lesion stable disease",
            ],
            "site": ["a", "a", "b", "b", "a", "b", "a", "b"],
        }
    )
    config = AppliedInferenceConfig(
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=True,
                    model_name="fake-keyword",
                    chunk_size_words=4,
                    chunk_overlap_words=0,
                    max_chunks=2,
                    min_probe_auc=0.0,
                    top_k_chunks_per_tail=3,
                    max_chunks_per_patient=1,
                    concept_phrases=["brain metastases", "liver lesion"],
                ),
                **_disable_htr_test_kwargs(),
            ),
        ),
    )
    generator = EmbeddingContrastEvidenceGenerator(
        config=config,
        output_dir=tmp_path,
        embedding_provider=KeywordEmbeddingProvider(),
    )
    generator.prepare(dataset)
    evidence = generator.build_evidence(
        discovery_df=dataset,
        y=np.asarray([0, 1, 0, 0, 1, 0, 1, 0], dtype=float),
        t=np.asarray([1, 1, 0, 0, 1, 0, 1, 0], dtype=float),
        pseudo_target=np.asarray([1, 1, -1, -1, 1, -1, 1, -1], dtype=float),
        t_resid=np.ones(8, dtype=float),
        importance={"phrase_features": [{"feature": "pd-l1 high"}]},
    )

    treatment = next(item for item in evidence["contrasts"] if item["name"] == "treatment")
    assert treatment["direction_source"] == "mean_difference"
    assert treatment["probe_auc_role"] == "diagnostic_only"
    assert any("brain" in row["text"] for row in treatment["positive_aligned_chunks"])
    assert any("liver" in row["text"] for row in treatment["negative_aligned_chunks"])
    assert any(
        row["concept"] == "brain metastases"
        for row in treatment["concept_probe_scores"]
    )

    redacted = redact_embedding_contrast_evidence(evidence)
    redacted_treatment = next(
        item for item in redacted["contrasts"] if item["name"] == "treatment"
    )
    assert all(row["text"] is None for row in redacted_treatment["positive_aligned_chunks"])
    assert all(
        row["text_redacted"] is True
        for row in redacted_treatment["positive_aligned_chunks"]
    )


def test_embedding_contrast_adds_cell_and_orthogonal_r_score_contrasts(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "_oci_row_id": np.arange(8),
            "clinical_text": [
                "brain metastases pd-l1 high",
                "brain metastases pd-l1 high",
                "liver lesion pd-l1 low",
                "liver lesion pd-l1 low",
                "liver lesion pd-l1 low",
                "liver lesion pd-l1 low",
                "brain metastases pd-l1 high",
                "brain metastases pd-l1 high",
            ],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=True,
                    model_name="fake-keyword",
                    chunk_size_words=4,
                    chunk_overlap_words=0,
                    max_chunks=1,
                    min_probe_auc=0.0,
                    top_k_chunks_per_tail=3,
                    max_chunks_per_patient=1,
                    concept_phrases=["brain metastases", "liver lesion"],
                ),
                **_disable_htr_test_kwargs(),
            ),
        ),
    )
    generator = EmbeddingContrastEvidenceGenerator(
        config=config,
        output_dir=tmp_path,
        embedding_provider=KeywordEmbeddingProvider(),
    )
    generator.prepare(dataset)
    evidence = generator.build_evidence(
        discovery_df=dataset,
        y=np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=float),
        t=np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=float),
        pseudo_target=np.asarray([2, 2, -2, -2, -2, -2, 2, 2], dtype=float),
        t_resid=np.ones(8, dtype=float),
        importance={},
    )

    contrast_names = {item["name"] for item in evidence["contrasts"]}
    assert {
        "treated_outcome",
        "untreated_outcome",
        "treatment_outcome_interaction",
        "orthogonal_r_score",
    }.issubset(contrast_names)

    interaction = next(
        item
        for item in evidence["contrasts"]
        if item["name"] == "treatment_outcome_interaction"
    )
    assert interaction["contrast_family"] == "treatment_outcome_cell_interaction"
    assert interaction["direction_source"] == "cell_mean_difference_in_differences"
    assert {row["n"] for row in interaction["component_counts"]} == {2}
    assert any(
        "brain" in row["text"] for row in interaction["positive_aligned_chunks"]
    )
    assert any(
        "liver" in row["text"] for row in interaction["negative_aligned_chunks"]
    )

    orthogonal = next(
        item for item in evidence["contrasts"] if item["name"] == "orthogonal_r_score"
    )
    assert orthogonal["contrast_family"] == "orthogonal_r_score"
    assert "(Y - m_hat) * (T - e_hat)" in orthogonal["score_formula"]
    assert any(
        "brain" in row["text"] for row in orthogonal["positive_aligned_chunks"]
    )


def test_multi_model_agentic_forest_runs_with_fake_agent_and_extractor(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 baseline note pd-l1 >=50% high marker",
                "age 78 baseline note pd-l1 <1% low marker",
                "age 57 baseline note pd-l1 >=50% high marker",
                "age 76 baseline note pd-l1 <1% low marker",
                "age 61 baseline note pd-l1 1-49% intermediate marker",
                "age 81 baseline note pd-l1 <1% low marker",
                "age 54 baseline note pd-l1 >=50% high marker",
                "age 70 baseline note pd-l1 1-49% intermediate marker",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 0, 0, 1, 0],
            "true_ite_prob": [0.3, 0.0, 0.3, 0.0, -0.1, 0.0, 0.3, -0.1],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                max_iterations=1,
                max_additions_per_iter=4,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_linear_test_bow_views(),
                top_n_features=5,
                candidate_consistency_enabled=False,
                fold_parallelism="2",
                **_disable_required_evidence_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = FakeProposalAgent()
    evaluator = FakeEvaluator()
    output_path = tmp_path / "predictions.parquet"

    run_multi_model_agentic_forest(
        dataset,
        config,
        output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=evaluator,
    )

    predictions = pd.read_parquet(output_path)
    assert len(predictions) == len(dataset)
    assert "selected_feature_names" in predictions.columns
    assert "selected_feature_roles" in predictions.columns
    assert "selected_confounder_names" in predictions.columns
    assert "selected_effect_modifier_names" in predictions.columns
    assert set(predictions["selected_feature_roles"]) == {
        "age[confounder],pd_l1_expression[effect_modifier]"
    }
    assert set(predictions["selected_confounder_names"]) == {"age"}
    assert set(predictions["selected_effect_modifier_names"]) == {"pd_l1_expression"}
    assert agent.contexts
    assert agent.contexts[0]["prompt_version"] == "multi_model_agentic_forest_v1"
    assert agent.contexts[1]["prompt_version"] == "multi_model_agentic_alias_resolution_v1"
    assert agent.contexts[2]["prompt_version"] == "multi_model_agentic_value_harmonization_v1"
    assert "feature_importance" in agent.contexts[0]
    phrase_features = agent.contexts[0]["feature_importance"]["phrase_features"]
    assert phrase_features
    assert all(2 <= len(row["feature"].split()) <= 4 for row in phrase_features)
    assert "canonical_feature_name_guidance" not in agent.contexts[0]
    assert "true_" not in json.dumps(agent.contexts[0])
    seen_names = [[spec.name for spec in specs] for specs in evaluator.seen_specs]
    assert all({"age", "pd_l1_expression"}.issubset(set(names)) for names in seen_names)
    assert all(names.count("pd_l1_expression") == 1 for names in seen_names)
    pdl1_specs = [
        spec
        for specs in evaluator.seen_specs
        for spec in specs
        if spec.name == "pd_l1_expression"
    ]
    assert pdl1_specs
    assert all(spec.categories == ["<1%", "1-49%", ">=50%"] for spec in pdl1_specs)
    assert all("unknown" not in spec.categories for spec in pdl1_specs)
    assert all(
        spec.value_aliases[">=50%"] == ["high", "50% or greater"]
        for spec in pdl1_specs
    )
    age_specs = [
        spec
        for specs in evaluator.seen_specs
        for spec in specs
        if spec.name == "age"
    ]
    assert age_specs
    assert all(spec.type == "continuous" and spec.categories is None for spec in age_specs)
    assert all("numeric value only" in (spec.description or "") for spec in age_specs)
    artifact_dir = output_path.parent / "multi_model_agentic_forest"
    assert (artifact_dir / "bow_view_oof_predictions.parquet").exists()
    assert (artifact_dir / "bow_view_feature_importance_by_fold.jsonl").exists()
    assert (artifact_dir / "agent_candidate_proposals.jsonl").exists()


def test_multi_model_agentic_forest_adds_embedding_contrast_context(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 baseline note brain metastases pd-l1 high",
                "age 78 baseline note liver lesion pd-l1 low",
                "age 57 baseline note brain metastases pd-l1 high",
                "age 76 baseline note liver lesion pd-l1 low",
                "age 61 baseline note brain metastases cachexia",
                "age 81 baseline note liver lesion stable",
                "age 54 baseline note brain metastases cachexia",
                "age 70 baseline note liver lesion stable",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                max_iterations=1,
                max_additions_per_iter=4,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_linear_test_bow_views(),
                top_n_features=5,
                candidate_consistency_enabled=False,
                fold_parallelism="1",
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=True,
                    model_name="fake-keyword",
                    chunk_size_words=8,
                    chunk_overlap_words=0,
                    max_chunks=2,
                    min_probe_auc=0.0,
                    top_k_chunks_per_tail=2,
                    max_chunks_per_patient=1,
                    concept_phrases=["brain metastases", "liver lesion"],
                ),
                **_disable_htr_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = FakeProposalAgent()
    output_path = tmp_path / "predictions.parquet"

    run_multi_model_agentic_forest(
        dataset,
        config,
        output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
        embedding_provider=KeywordEmbeddingProvider(),
    )

    first_context = agent.contexts[0]
    assert "embedding_contrast_evidence" in first_context
    treatment = next(
        item
        for item in first_context["embedding_contrast_evidence"]["contrasts"]
        if item["name"] == "treatment"
    )
    assert any("brain" in row["text"] for row in treatment["positive_aligned_chunks"])

    artifact_dir = output_path.parent / "multi_model_agentic_forest"
    evidence_rows = [
        json.loads(line)
        for line in (artifact_dir / "embedding_contrast_evidence_by_fold.jsonl")
        .read_text()
        .splitlines()
    ]
    assert evidence_rows
    artifact_treatment = next(
        item
        for item in evidence_rows[0]["embedding_contrast_evidence"]["contrasts"]
        if item["name"] == "treatment"
    )
    assert all(row["text"] is None for row in artifact_treatment["positive_aligned_chunks"])
    assert all(
        row["text_redacted"] is True
        for row in artifact_treatment["positive_aligned_chunks"]
    )


def test_multi_model_agentic_forest_adds_htr_to_ensemble_and_agent_context(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 baseline note brain metastases pd-l1 high",
                "age 78 baseline note liver lesion pd-l1 low",
                "age 57 baseline note brain metastases pd-l1 high",
                "age 76 baseline note liver lesion pd-l1 low",
                "age 61 baseline note brain metastases cachexia",
                "age 81 baseline note liver lesion stable",
                "age 54 baseline note brain metastases cachexia",
                "age 70 baseline note liver lesion stable",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                max_iterations=1,
                max_additions_per_iter=4,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_linear_test_bow_views(),
                top_n_features=5,
                candidate_consistency_enabled=False,
                fold_parallelism="1",
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=False,
                    disable_reason="unit test focuses on HTR evidence",
                ),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = FakeProposalAgent()
    htr_provider = FakeHTREvidenceProvider()
    output_path = tmp_path / "predictions.parquet"

    run_multi_model_agentic_forest(
        dataset,
        config,
        output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
        htr_evidence_provider=htr_provider,
    )

    assert htr_provider.seen_effect_nuisance_predictions
    effect_nuisance = htr_provider.seen_effect_nuisance_predictions[0]
    assert set(effect_nuisance["target_source"]) == {"ensemble_mean_nuisance_with_htr"}
    first_context = agent.contexts[0]
    assert first_context["feature_importance"]["ensemble_r"]["target_source"] == (
        "ensemble_mean_nuisance_with_htr"
    )
    htr_evidence = first_context["htr_attention_evidence"]
    assert htr_evidence["nuisance"]["attention"][0]["evidence_snippet"]
    assert htr_evidence["effect"]["attention"][0]["top_token_spans"]

    artifact_dir = output_path.parent / "multi_model_agentic_forest"
    assert (artifact_dir / "htr_nuisance_oof_predictions.parquet").exists()
    assert (artifact_dir / "htr_effect_oof_predictions.parquet").exists()
    assert (artifact_dir / "htr_attention_evidence.parquet").exists()
    text_predictions = pd.read_parquet(artifact_dir / "text_model_oof_predictions.parquet")
    assert {"htr_nuisance", "htr_effect"}.issubset(
        set(text_predictions["view_name"].dropna())
    )
    importance_rows = [
        json.loads(line)
        for line in (artifact_dir / "bow_view_feature_importance_by_fold.jsonl")
        .read_text()
        .splitlines()
    ]
    consensus_row = next(row for row in importance_rows if row["record_type"] == "consensus")
    artifact_htr_row = consensus_row["context"]["htr_attention_evidence"]["nuisance"][
        "attention"
    ][0]
    assert artifact_htr_row["text_redacted"] is True
    assert "evidence_snippet" not in artifact_htr_row
    assert "top_token_spans" not in artifact_htr_row


def test_multi_model_agentic_forest_adds_ensemble_r_target_context(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 baseline note pd-l1 high brain metastases",
                "age 78 baseline note pd-l1 low liver lesion",
                "age 57 baseline note pd-l1 high brain metastases",
                "age 76 baseline note pd-l1 low liver lesion",
                "age 61 baseline note pd-l1 high cachexia",
                "age 81 baseline note pd-l1 low stable",
                "age 54 baseline note pd-l1 high cachexia",
                "age 70 baseline note pd-l1 low stable",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                max_iterations=1,
                max_additions_per_iter=4,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_two_linear_test_bow_views(),
                top_n_features=5,
                candidate_consistency_enabled=False,
                fold_parallelism="1",
                **_disable_required_evidence_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = FakeProposalAgent()
    output_path = tmp_path / "predictions.parquet"

    run_multi_model_agentic_forest(
        dataset,
        config,
        output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
    )

    importance = agent.contexts[0]["feature_importance"]
    assert "ensemble_r" in importance
    assert importance["ensemble_r"]["target_source"] == "ensemble_mean_nuisance"
    assert all(
        str(view["view_name"]).startswith("ensemble_r__")
        for view in importance["ensemble_r"]["views"]
    )

    artifact_dir = output_path.parent / "multi_model_agentic_forest"
    bow_oof = pd.read_parquet(artifact_dir / "bow_view_oof_predictions.parquet")
    assert "target_source" in bow_oof.columns
    assert "ensemble_mean_nuisance" in set(bow_oof["target_source"].dropna())
    importance_rows = [
        json.loads(line)
        for line in (artifact_dir / "bow_view_feature_importance_by_fold.jsonl")
        .read_text()
        .splitlines()
    ]
    assert any(row["record_type"] == "ensemble_r_view" for row in importance_rows)
    assert any(row["record_type"] == "ensemble_r_consensus" for row in importance_rows)


def test_multi_model_agent_context_compacts_large_evidence_payload():
    long_rows = [
        {
            "feature": f"feature phrase {idx}",
            "score": 0.123456789,
            "combined_score": 0.987654321,
            "confounder_overlap_score": 0.25,
            "treatment_score": 0.5,
            "outcome_score": 0.5,
            "pseudo_target_score": 0.1,
            "abs_pseudo_target_score": 0.1,
            "supporting_views": ["v1"],
        }
        for idx in range(100)
    ]
    long_text = "brain metastases " * 200
    context = {
        "prompt_version": "multi_model_agentic_forest_v1",
        "feature_importance": {
            "n_views": 1,
            "phrase_consensus": long_rows,
            "views": [
                {
                    "view_name": "v1",
                    "view_index": 0,
                    "view_config": {"name": "v1"},
                    "metrics": {},
                    "n_features": 1000,
                    "n_bow_features": 1000,
                    "n_prespecified_features": 0,
                    "phrase_features": long_rows,
                    "confounder_overlap": long_rows,
                    "treatment_positive": long_rows,
                    "treatment_negative": long_rows,
                    "outcome_positive": long_rows,
                    "outcome_negative": long_rows,
                    "pseudo_target_positive": long_rows,
                    "pseudo_target_negative": long_rows,
                }
            ],
        },
        "embedding_contrast_evidence": {
            "enabled": True,
            "contrasts": [
                {
                    "name": "treatment",
                    "positive_label": "treated",
                    "negative_label": "untreated",
                    "role_hint": "confounder",
                    "positive_aligned_chunks": [
                        {"row_id": idx, "chunk_index": idx, "score": 0.123456, "text": long_text}
                        for idx in range(10)
                    ],
                    "negative_aligned_chunks": [
                        {"row_id": idx, "chunk_index": idx, "score": -0.123456, "text": long_text}
                        for idx in range(10)
                    ],
                    "concept_probe_scores": [
                        {"concept": f"concept {idx}", "score": 0.111111}
                        for idx in range(20)
                    ],
                }
            ],
        },
    }
    compact = _compact_multi_model_agent_context(context)
    assert len(compact["feature_importance"]["phrase_consensus"]) == 40
    compact_view = compact["feature_importance"]["views"][0]
    assert len(compact_view["treatment_positive"]) == 12
    contrast = compact["embedding_contrast_evidence"]["contrasts"][0]
    assert len(contrast["positive_aligned_chunks"]) == 3
    assert len(contrast["concept_probe_scores"]) == 8
    assert len(contrast["positive_aligned_chunks"][0]["text"]) <= 600


def test_multi_model_bow_fold_uses_prespecified_explicit_features():
    texts = ["same baseline note"] * 6
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
    explicit_values = [
        {"age": 30.0, "age_missing": False},
        {"age": 31.0, "age_missing": False},
        {"age": 29.0, "age_missing": False},
        {"age": 80.0, "age_missing": False},
        {"age": 81.0, "age_missing": False},
        {"age": 79.0, "age_missing": False},
    ]
    heldout, pred = _fit_binary_bow_fold(
        texts,
        labels,
        fit_pos=np.asarray([0, 1, 3, 4]),
        heldout_pos=np.asarray([2, 5]),
        vectorizer_params={
            "ngram_range_min": 1,
            "ngram_range_max": 1,
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
            "max_features": 100,
        },
        model_params={
            "bow_model": "linear",
            "logistic_c": 1.0,
            "logistic_max_iter": 1000,
            "ridge_alpha": 1.0,
        },
        explicit_feature_dicts=explicit_values,
        explicit_specs=[
            ExplicitFeatureSpec(
                name="age",
                type="continuous",
                roles=["confounder"],
            )
        ],
        random_state=7,
    )
    assert heldout.tolist() == [2, 5]
    assert pred[1] > pred[0]


def test_extracted_feature_review_gate_flags_underperforming_features():
    dataset = pd.DataFrame(
        {
            "clinical_text": ["signal positive", "signal negative"] * 6,
            "treatment_indicator": [1, 0] * 6,
            "outcome_indicator": [1, 0] * 6,
            "explicit_feat_noise_marker": ["present"] * 12,
            "explicit_feat_noise_marker_missing": [False] * 12,
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
    )
    nn_config = MultiModelAgenticForestConfig(
        nuisance_folds=2,
        effect_folds=2,
        bow_views=_linear_test_bow_views(),
        extracted_feature_review_auc_margin=0.01,
        extracted_feature_review_loss_relative_margin=0.01,
        extracted_feature_review_min_benchmark_auc=0.55,
        **_disable_required_evidence_test_kwargs(),
    )
    diagnostic = _evaluate_extracted_feature_set_diagnostic(
        train_df=dataset,
        specs=[
            ExplicitFeatureSpec(
                name="noise_marker",
                type="categorical",
                categories=["absent", "present"],
                roles=["confounder"],
            )
        ],
        config=config,
        nn_config=nn_config,
        bow_metrics={
            "views": [
                {
                    "metrics": {
                        "treatment_auroc": 1.0,
                        "outcome_auroc": 1.0,
                        "r_loss_mean": 0.01,
                    }
                }
            ]
        },
        embedding_evidence={},
        random_state=123,
    )
    gate = _extracted_feature_review_gate(
        diagnostic=diagnostic,
        nn_config=nn_config,
    )
    failed_metrics = {item["metric"] for item in gate["failed_criteria"]}
    assert gate["passed"] is False
    assert {"treatment_auroc", "outcome_auroc"}.issubset(failed_metrics)


def test_multi_model_parsimony_redundancy_review_records_required_summaries():
    df = pd.DataFrame(
        {
            "explicit_feat_age": [50.0, 60.0, 70.0, 80.0],
            "explicit_feat_age_copy": [51.0, 61.0, 71.0, 81.0],
            "explicit_feat_marker": ["low", "low", "high", "high"],
            "explicit_feat_marker_copy": ["low", "low", "high", "high"],
            "explicit_feat_age_missing": [False, False, False, False],
            "explicit_feat_age_copy_missing": [False, False, False, False],
            "explicit_feat_marker_missing": [False, False, False, False],
            "explicit_feat_marker_copy_missing": [False, False, False, False],
        }
    )
    specs = [
        ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ExplicitFeatureSpec(name="age_copy", type="continuous", roles=["confounder"]),
        ExplicitFeatureSpec(
            name="marker",
            type="categorical",
            categories=["low", "high"],
            roles=["effect_modifier"],
        ),
        ExplicitFeatureSpec(
            name="marker_copy",
            type="categorical",
            categories=["low", "high"],
            roles=["effect_modifier"],
        ),
    ]

    review = _feature_redundancy_review(
        train_df=df,
        specs=specs,
        corr_threshold=0.75,
    )

    assert review["continuous_correlations_abs_ge_threshold"]
    assert review["categorical_contingency"]
    assert len(review["missingness_overlap"]) == 6


def test_multi_model_extracted_feature_review_revises_underperforming_specs(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
                "signal positive baseline note",
                "signal negative baseline note",
            ],
            "treatment_indicator": [1, 0] * 6,
            "outcome_indicator": [1, 0] * 6,
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                max_removals_per_iter=2,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_linear_test_bow_views(),
                top_n_features=5,
                candidate_consistency_enabled=False,
                extracted_feature_review_enabled=True,
                extracted_feature_review_max_rounds=1,
                extracted_feature_review_auc_margin=0.0,
                extracted_feature_review_loss_relative_margin=0.0,
                extracted_feature_review_min_benchmark_auc=0.55,
                fold_parallelism="1",
                **_disable_required_evidence_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = ReviewRevisionAgent()
    extractor = ReviewExtractionProvider()
    evaluator = FakeEvaluator()
    output_path = tmp_path / "predictions.parquet"

    run_multi_model_agentic_forest(
        dataset,
        config,
        output_path,
        proposal_agent=agent,
        extraction_provider=extractor,
        evaluator=evaluator,
    )

    assert any(
        context.get("prompt_version")
        == "multi_model_agentic_extracted_feature_review_v1"
        for context in agent.contexts
    )
    assert ["noise_marker"] in extractor.calls
    assert any("signal_marker" in call for call in extractor.calls)
    final_names = [spec.name for spec in evaluator.seen_specs[-1]]
    assert "signal_marker" in final_names
    artifact_dir = output_path.parent / "multi_model_agentic_forest"
    diagnostics = [
        json.loads(line)
        for line in (artifact_dir / "extracted_feature_diagnostics_by_fold.jsonl")
        .read_text()
        .splitlines()
    ]
    assert diagnostics
    selected_sets = json.loads((artifact_dir / "selected_feature_sets.json").read_text())
    assert selected_sets[0]["extracted_feature_review"]["review_rounds"] >= 1
    assert selected_sets[0]["parsimony_review"]["mandatory"] is True
    assert selected_sets[0]["parsimony_review"]["decision"] in {"retain_all", "prune"}
    parsimony_rows = [
        json.loads(line)
        for line in (artifact_dir / "parsimony_review_by_fold.jsonl")
        .read_text()
        .splitlines()
    ]
    assert parsimony_rows
    assert parsimony_rows[0]["event"] == "mandatory_parsimony_review"
    assert "redundancy_review" in parsimony_rows[0]
    assert "ablations" in parsimony_rows[0]


def test_multi_model_prespecified_features_extract_before_bow_and_merge_roles(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 biomarker positive baseline note",
                "age 78 biomarker negative baseline note",
                "age 57 biomarker positive baseline note",
                "age 76 biomarker negative baseline note",
                "age 61 biomarker positive baseline note",
                "age 81 biomarker negative baseline note",
                "age 54 biomarker positive baseline note",
                "age 70 biomarker negative baseline note",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                max_iterations=1,
                max_additions_per_iter=4,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                bow_views=_linear_test_bow_views(),
                top_n_features=5,
                candidate_consistency_enabled=False,
                fold_parallelism="1",
                prespecified_confounders=[
                    ExplicitFeatureSpec(
                        name="age",
                        type="continuous",
                        roles=["confounder"],
                    ),
                    {
                        "name": "biomarker",
                        "type": "categorical",
                        "categories": ["negative", "positive"],
                    },
                ],
                prespecified_effect_modifiers=[
                    {
                        "name": "biomarker",
                        "type": "categorical",
                        "categories": ["negative", "positive"],
                    },
                ],
                **_disable_required_evidence_test_kwargs(),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = EmptyProposalAgent()
    extractor = RecordingExtractionProvider()
    evaluator = FakeEvaluator()

    run_multi_model_agentic_forest(
        dataset,
        config,
        tmp_path / "predictions.parquet",
        proposal_agent=agent,
        extraction_provider=extractor,
        evaluator=evaluator,
    )

    assert extractor.calls
    assert extractor.calls[0] == [
        ("age", ("confounder",)),
        ("biomarker", ("confounder", "effect_modifier")),
    ]
    first_context = agent.contexts[0]
    assert first_context["prompt_version"] == "multi_model_agentic_forest_v1"
    assert first_context["current_features"] == [
        {
            "name": "age",
            "type": "continuous",
            "categories": None,
            "description": None,
            "roles": ["confounder"],
            "value_aliases": None,
        },
        {
            "name": "biomarker",
            "type": "categorical",
            "categories": ["negative", "positive"],
            "description": None,
            "roles": ["confounder", "effect_modifier"],
            "value_aliases": None,
        },
    ]
    importance = first_context["feature_importance"]
    assert importance["n_views"] == 1
    first_view = importance["views"][0]
    assert first_view["n_prespecified_features"] == 2
    assert "explicit:age_normalized" in first_view["prespecified_raw_feature_names"]
    assert "explicit:biomarker_positive" in first_view["prespecified_raw_feature_names"]
    seen_roles = {
        spec.name: spec.roles
        for specs in evaluator.seen_specs
        for spec in specs
    }
    assert seen_roles == {
        "age": ["confounder"],
        "biomarker": ["confounder", "effect_modifier"],
    }


def test_multi_model_agentic_forest_parses_bow_views_and_embedding_option():
    cfg = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": (
                    "synthetic_data/example_synthetic_datasets/"
                    "one_confounder_one_effect_modifier_nsclc_with_structured/"
                    "dataset.parquet"
                ),
                "architecture": {
                    "model_type": "multi_model_agentic_forest",
                    "multi_model_agentic_forest": {
                        "bow_views": [
                            {
                                "name": "trees",
                                "bow_model": "extratrees",
                                "ngram_range_min": 1,
                                "ngram_range_max": 3,
                                "min_df": 1,
                            }
                        ],
                        "nuisance_folds": 2,
                        "effect_folds": 2,
                        "candidate_consistency_enabled": True,
                        "candidate_consistency_inner_folds": 4,
                        "candidate_consistency_min_folds": 2,
                        "candidate_consistency_min_fold_fraction": 0.5,
                        "candidate_consistency_parallelism": "2",
                        "outer_parallelism": "3",
                        "bow_parallel_backend": "processes",
                        "extracted_feature_review_enabled": False,
                        "extracted_feature_review_max_rounds": 2,
                        "extracted_feature_review_auc_margin": 0.03,
                        "extracted_feature_review_loss_relative_margin": 0.07,
                        "extracted_feature_review_min_benchmark_auc": 0.6,
                        "parsimony_review_auc_tolerance": 0.02,
                        "parsimony_review_loss_relative_tolerance": 0.04,
                        "parsimony_review_corr_threshold": 0.8,
                        "parsimony_review_max_single_feature_ablations": 7,
                        "embedding_contrast": {
                            "enabled": True,
                            "model_name": "Qwen/Qwen3-Embedding-8B",
                            "chunk_size_words": 128,
                            "chunk_overlap_words": 32,
                            "max_chunks": 16,
                            "chunk_selection": "last",
                            "min_probe_auc": 0.55,
                            "include_cell_contrasts": False,
                            "include_orthogonal_r_score_contrasts": False,
                            "concept_phrases": ["brain metastases"],
                        },
                    },
                },
                "explicit_features": {"enabled": True, "features": []},
            }
        }
    )
    nn_cfg = cfg.applied_inference.architecture.multi_model_agentic_forest
    assert len(nn_cfg.bow_views) == 1
    assert nn_cfg.bow_views[0].name == "trees"
    assert nn_cfg.bow_views[0].bow_model == "extratrees"
    assert nn_cfg.bow_views[0].ngram_range_max == 3
    assert nn_cfg.candidate_consistency_enabled is True
    assert nn_cfg.candidate_consistency_inner_folds == 4
    assert nn_cfg.candidate_consistency_min_folds == 2
    assert nn_cfg.candidate_consistency_min_fold_fraction == 0.5
    assert nn_cfg.candidate_consistency_parallelism == "2"
    assert nn_cfg.outer_parallelism == "3"
    assert nn_cfg.bow_parallel_backend == "processes"
    assert nn_cfg.extracted_feature_review_enabled is False
    assert nn_cfg.extracted_feature_review_max_rounds == 2
    assert nn_cfg.extracted_feature_review_auc_margin == 0.03
    assert nn_cfg.extracted_feature_review_loss_relative_margin == 0.07
    assert nn_cfg.extracted_feature_review_min_benchmark_auc == 0.6
    assert nn_cfg.parsimony_review_auc_tolerance == 0.02
    assert nn_cfg.parsimony_review_loss_relative_tolerance == 0.04
    assert nn_cfg.parsimony_review_corr_threshold == 0.8
    assert nn_cfg.parsimony_review_max_single_feature_ablations == 7
    assert nn_cfg.embedding_contrast.enabled is True
    assert nn_cfg.embedding_contrast.model_name == "Qwen/Qwen3-Embedding-8B"
    assert nn_cfg.embedding_contrast.chunk_size_words == 128
    assert nn_cfg.embedding_contrast.chunk_overlap_words == 32
    assert nn_cfg.embedding_contrast.max_chunks == 16
    assert nn_cfg.embedding_contrast.chunk_selection == "last"
    assert nn_cfg.embedding_contrast.min_probe_auc == 0.55
    assert nn_cfg.embedding_contrast.include_cell_contrasts is False
    assert nn_cfg.embedding_contrast.include_orthogonal_r_score_contrasts is False
    assert nn_cfg.embedding_contrast.concept_phrases == ["brain metastases"]
    cfg.validate()


def test_multi_model_agentic_forest_default_bow_view_grid():
    cfg = MultiModelAgenticForestConfig(nuisance_folds=2, effect_folds=2)
    names = [view.name for view in cfg.bow_views]
    assert names == [
        "linear_unigram_c0p5",
        "linear_1_2",
        "linear_1_3",
        "linear_2_4_min_df3",
        "extratrees_1_3",
        "random_forest_1_2",
    ]
    assert {view.bow_model for view in cfg.bow_views} == {
        "linear",
        "extratrees",
        "random_forest",
    }


def test_old_non_neural_agentic_forest_model_type_rejected():
    cfg = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": (
                    "synthetic_data/example_synthetic_datasets/"
                    "one_confounder_one_effect_modifier_nsclc_with_structured/"
                    "dataset.parquet"
                ),
                "architecture": {"model_type": "non_neural_agentic_forest"},
            }
        }
    )
    try:
        cfg.validate()
    except ValueError as exc:
        assert "multi_model_agentic_forest" in str(exc)
    else:
        raise AssertionError("old non_neural_agentic_forest model_type was accepted")


def test_oracle_multi_model_script_builds_default_and_cli_bow_views():
    from oracle_experiment_scripts import run_oracle_multi_model_agentic_forest as script

    dataset_path = Path(
        "synthetic_data/example_synthetic_datasets/"
        "one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet"
    )
    cfg = script.MultiModelAgenticOracleConfig(
        dataset_path=str(dataset_path),
        dataset_name="smoke",
    )
    applied = script._make_applied_config(cfg, dataset_path)
    mm_cfg = applied.architecture.multi_model_agentic_forest
    assert len(mm_cfg.bow_views) == 6
    assert mm_cfg.embedding_contrast.enabled is True

    cfg.bow_view_grid = "cli_single"
    cfg.bow_model = "extratrees"
    cfg.embedding_contrast_enabled = True
    cfg.embedding_concept_phrases = ["brain metastases"]
    cfg.extracted_feature_review_enabled = False
    cfg.extracted_feature_review_max_rounds = 1
    applied = script._make_applied_config(cfg, dataset_path)
    mm_cfg = applied.architecture.multi_model_agentic_forest
    assert [view.name for view in mm_cfg.bow_views] == ["cli_view"]
    assert mm_cfg.bow_views[0].bow_model == "extratrees"
    assert mm_cfg.embedding_contrast.enabled is True
    assert mm_cfg.embedding_contrast.concept_phrases == ["brain metastases"]
    assert mm_cfg.extracted_feature_review_enabled is False
    assert mm_cfg.extracted_feature_review_max_rounds == 1


def test_multi_model_agentic_forest_parses_prespecified_feature_sources(tmp_path):
    features_json = tmp_path / "features.json"
    features_json.write_text(
        json.dumps(
            {
                "confounders": [
                    {
                        "name": "baseline_risk",
                        "type": "continuous",
                        "description": "Baseline clinical risk score.",
                    }
                ],
                "effect_modifiers": [
                    {
                        "name": "baseline_risk",
                        "type": "continuous",
                        "description": "Baseline clinical risk score.",
                    }
                ],
            }
        )
    )
    cfg = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": (
                    "synthetic_data/example_synthetic_datasets/"
                    "one_confounder_one_effect_modifier_nsclc_with_structured/"
                    "dataset.parquet"
                ),
                "architecture": {
                    "model_type": "multi_model_agentic_forest",
                    "multi_model_agentic_forest": {
                        "prespecified_confounders": [
                            {
                                "name": "age",
                                "type": "continuous",
                                "description": "Age in years.",
                            }
                        ],
                        "prespecified_effect_modifiers": [
                            {
                                "name": "pd_l1_expression",
                                "type": "categorical",
                                "categories": ["low", "high"],
                                "description": "PD-L1 expression.",
                            }
                        ],
                        "prespecified_features_json": str(features_json),
                    },
                },
                "explicit_features": {"enabled": True, "features": []},
            }
        }
    )
    nn_cfg = cfg.applied_inference.architecture.multi_model_agentic_forest
    assert nn_cfg.prespecified_confounders[0].roles == ["confounder"]
    assert nn_cfg.prespecified_effect_modifiers[0].roles == ["effect_modifier"]
    assert nn_cfg.prespecified_features_json == str(features_json)


def test_multi_model_candidate_consistency_fallback_prefers_stable_candidates():
    assert _candidate_consistency_threshold(
        3,
        min_folds=2,
        min_fold_fraction=0.5,
    ) == 2
    age = AgenticFeatureProposal(
        action="add",
        name="patient_age",
        type="continuous",
        roles=["confounder"],
    )
    noise = AgenticFeatureProposal(
        action="add",
        name="rare_noise",
        type="categorical",
        categories=["present", "absent"],
        roles=["effect_modifier"],
    )
    selected = _fallback_consistency_proposals(
        [
            {
                "name": "patient_age",
                "passes_consistency_gate": True,
                "proposed_on_full_outer_train": True,
            },
            {
                "name": "rare_noise",
                "passes_consistency_gate": False,
                "proposed_on_full_outer_train": True,
            },
        ],
        {"patient_age": age, "rare_noise": noise},
    )
    assert selected == [age]


def test_multi_model_consistency_selection_is_deterministic_gate(tmp_path):
    runner = MultiModelAgenticForestRunner(
        dataset=pd.DataFrame(
            {
                "clinical_text": ["ecog 0", "ecog 2"],
                "treatment_indicator": [1, 0],
                "outcome_indicator": [1, 0],
            }
        ),
        config=AppliedInferenceConfig(
            text_column="clinical_text",
            treatment_column="treatment_indicator",
            outcome_column="outcome_indicator",
            architecture=ModelArchitectureConfig(
                model_type="multi_model_agentic_forest",
                agentic_feature_search=AgenticFeatureSearchConfig(),
                multi_model_agentic_forest=MultiModelAgenticForestConfig(
                    candidate_proposals_per_fold=5,
                    **_disable_required_evidence_test_kwargs(),
                ),
            ),
        ),
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=FailingConsistencyAgent(),
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
    )
    ecog = AgenticFeatureProposal(
        action="add",
        name="ecog_performance_status",
        type="categorical",
        categories=[
            "0",
            "1",
            "2",
            "3",
            "4",
            "ECOG 0",
            "ECOG 1",
            "ECOG 2",
            "ECOG 3",
        ],
        roles=["confounder"],
        description="Baseline ECOG performance status.",
    )

    selected, artifact = runner._select_consistent_proposals(
        context={"prompt_version": "multi_model_agentic_consistency_v1"},
        candidate_summaries=[
            {
                "name": "ecog_performance_status",
                "passes_consistency_gate": True,
                "inner_support_count": 3,
                "proposed_on_full_outer_train": True,
            }
        ],
        canonical_proposals={"ecog_performance_status": ecog},
    )

    assert selected == [ecog]
    assert artifact["selection_method"] == "deterministic_consistency_gate"
    assert artifact["agent_selection_used"] is False
