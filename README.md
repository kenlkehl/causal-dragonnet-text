# Oncology Causal Inference (OCI)

OCI estimates treatment effects from clinical text. It uses a frozen pretrained LLM to extract patient-level feature vectors from unstructured EHR narratives, then feeds those features to causal inference heads (DragonNet, R-Learner, or Causal Forest) to estimate individual treatment effects (ITE) and average treatment effects (ATE).

## Installation

```bash
git clone https://github.com/kenlkehl/onc-causal-inference.git
cd onc-causal-inference
pip install -e .

# For LLM-based explicit feature extraction (optional)
pip install -e ".[extraction]"
```

Requires Python 3.12+. A CUDA GPU is required for the frozen LLM forward pass.

## Quick Start

```bash
# Generate a default config file
oci init --output config.json

# Run an experiment
oci run --config config.json --device cuda:0 --workers 4
```

Or use the oracle experiment script to run a grid of experiments on synthetic data:

```bash
python oracle_experiment_scripts/run_oracle_experiments.py \
    --output-dir ../my_results \
    --devices cuda:0 \
    --datasets one_confounder_twostage \
    --epochs 20 \
    --n-folds 5
```

## Architecture

OCI has five feature extractors and four causal heads.

### Feature Extractors

| Type | Description |
|------|-------------|
| `frozen_llm_pooler` | Frozen pretrained LLM + gated attention pooling (default) |
| `hierarchical_llm` | Frozen LLM on overlapping text chunks + two-level pooling |
| `hierarchical_cnn` | Trainable dilated CNN on chunks + two-level pooling |
| `hierarchical_gru` | Trainable BiGRU on chunks + two-level pooling |
| `simple_cnn` | Trainable dilated CNN on whole text + pooling |

Trainable extractors (`hierarchical_cnn`, `hierarchical_gru`, `simple_cnn`) learn from scratch and require `fit_tokenizer()` before training.

#### Frozen LLM Pooler (default)

A pretrained decoder-only LLM with frozen weights extracts per-token hidden states from clinical text. Gated attention pooling aggregates all tokens into a single patient-level feature vector. Only the downprojection, pooling, and causal head parameters are trained.

```
Clinical Text
  -> Pretrained Tokenizer (right-padded)
  -> Frozen LLM (no_grad, autocast float16)
  -> All Token Hidden States
  -> Trainable Downprojection (e.g., 1024 -> 256)
  -> Gated Attention Pooling (tanh x sigmoid gating)
  -> Projection MLP
  -> Patient Feature Vector
  -> Causal Head
```

Key parameters:

| Param | Description | Default |
|-------|-------------|---------|
| `flp_model_name` | HuggingFace model name | `"Qwen/Qwen3.5-0.8B-Base"` |
| `flp_max_length` | Maximum token length | `8192` |
| `flp_downprojection_dim` | Reduce hidden dim before pooling | `256` |
| `flp_gated_attention_dim` | Gated attention hidden dim | `128` |
| `flp_projection_dim` | Output projection dim | `128` |
| `flp_cache_hidden_states` | Pre-cache hidden states to disk | `false` |
| `flp_chat_template_prompt` | Chat template prompt for instruct models | `null` |

Handles documents up to 50K+ tokens with the pretrained tokenizer. No `fit_tokenizer()` step is needed.

**Instruct model support**: Set `flp_chat_template_prompt` to wrap clinical text in the model's chat template. Recommended prompt: `"You are an expert clinical cancer researcher. Read this patient history, and then extract a set of features that will predict the patient's next treatment and their outcome on that treatment. The history is: "`

### Causal Heads

| Type | Description | Key Output |
|------|-------------|------------|
| `dragonnet` | Propensity + Y0/Y1 potential outcomes | ITE = P(Y=1\|T=1,X) - P(Y=1\|T=0,X) |
| `rlearner` | Direct tau(X) optimization with detached nuisance functions | tau directly predicts ITE |
| `causal_forest` | Two-stage: neural features + econml CausalForestDML | tau with 95% confidence intervals |
| `tfidf_forest` | TF-IDF features + CausalForestDML (no neural network, no GPU) | tau with 95% confidence intervals |
| `explicit_feature_forest` | Role-tagged explicit features + CausalForestDML (no text model) | tau with 95% confidence intervals |
| `agentic_explicit_feature_forest` | Nested-CV LLM variable search + explicit-feature CausalForestDML | outer-CV tau, nuisance AUROC, R-loss |
| `multi_model_agentic_forest` | Multi-view BoW/embedding evidence + LLM extraction + explicit-feature CausalForestDML | outer-CV tau, per-view BoW diagnostics, selected variables |

**Recommended: Causal Forest** -- trains neural features with propensity + outcome losses (optionally with R-learner loss), then fits CausalForestDML on the learned representations for doubly-robust estimation with confidence intervals.

**TF-IDF Forest** is a non-neural baseline that uses sklearn `TfidfVectorizer` features directly. No GPU required.

## Dataset Format

OCI expects Parquet or CSV files with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `clinical_text` | string | Clinical narrative |
| `treatment_indicator` | int | Binary treatment (0/1) |
| `outcome_indicator` | int/float | Binary (0/1) or continuous outcome |
| `split` | string | Optional: "train"/"val"/"test" for fixed splits |

Set `outcome_type` in config: `"binary"` (default, BCE loss + sigmoid) or `"continuous"` (MSE loss, no sigmoid). Treatment is always binary.

## Configuration

### Causal Forest with R-Learner Representation Training

```json
{
  "output_dir": "./oci_results",
  "seed": 42,
  "device": "cuda:0",

  "applied_inference": {
    "dataset_path": "./data/clinical_notes.parquet",
    "cv_folds": 5,

    "architecture": {
      "model_type": "causal_forest",
      "feature_extractor_type": "frozen_llm_pooler",

      "flp_model_name": "Qwen/Qwen3.5-0.8B-Base",
      "flp_max_length": 50000,
      "flp_freeze_llm": true,
      "flp_downprojection_dim": 256,
      "flp_gated_attention_dim": 128,
      "flp_projection_dim": 128,
      "flp_gradient_checkpointing": true,

      "causal_forest": {
        "n_estimators": 200,
        "min_samples_leaf": 5,
        "honest": true,
        "inference": true,
        "use_rlearner_representation": true,
        "rlearner_nuisance_folds": 5
      }
    },

    "training": {
      "epochs": 30,
      "batch_size": 2,
      "learning_rate": 1e-4
    }
  }
}
```

### R-Learner

```json
{
  "applied_inference": {
    "architecture": {
      "model_type": "rlearner",
      "feature_extractor_type": "frozen_llm_pooler",

      "flp_model_name": "Qwen/Qwen3.5-0.8B-Base",
      "flp_max_length": 50000,
      "flp_downprojection_dim": 256
    },
    "training": {
      "epochs": 30,
      "gamma_rlearner": 1.0,
      "stop_grad_propensity": false
    }
  }
}
```

### DragonNet

```json
{
  "applied_inference": {
    "architecture": {
      "model_type": "dragonnet",
      "feature_extractor_type": "frozen_llm_pooler",

      "flp_model_name": "Qwen/Qwen3.5-0.8B-Base",
      "flp_max_length": 50000,
      "flp_downprojection_dim": 256
    },
    "training": {
      "epochs": 30,
      "beta_targreg": 0.1
    }
  }
}
```

### TF-IDF Forest (No GPU Baseline)

```json
{
  "device": "cpu",
  "applied_inference": {
    "architecture": {
      "model_type": "tfidf_forest",
      "tfidf_forest": {
        "max_features": 10000,
        "ngram_range_min": 1,
        "ngram_range_max": 2,
        "n_estimators": 200,
        "min_samples_leaf": 10,
        "honest": true,
        "inference": true
      }
    }
  }
}
```

See `example_configs/` for complete configuration files, including configs for each feature extractor type.

## Staged R-Learner Representation

For Causal Forest, `use_rlearner_representation=true` trains separate nuisance and effect representations. Nuisance features support propensity/outcome prediction; effect features are trained with R-loss from out-of-fold nuisance predictions and are used as forest `X`.

```json
{
  "architecture": {
    "model_type": "causal_forest",
    "feature_extractor_type": "frozen_llm_pooler",
    "causal_forest": {
      "use_rlearner_representation": true,
      "rlearner_nuisance_folds": 5
    }
  }
}
```

Note: staged R-learner representation training uses separate nuisance and effect extractors, so it approximately doubles feature extraction memory and compute when enabled.

## Hidden State Caching

When the LLM is frozen, hidden states can be pre-computed once and reused across CV folds and experiment runs. This avoids repeated LLM forward passes.

Enable in config:
```json
{
  "architecture": {
    "flp_cache_hidden_states": true
  }
}
```

Or pass `--cache` to the oracle experiment script. Cache files are stored in `{dataset_dir}/.oci_cache/` and are keyed by model name, max length, and dataset path. Different causal heads, learning rates, and fold counts all share the same cache.

Use `--gpu-cache` to keep hidden states in GPU VRAM for fastest access (requires sufficient VRAM).

## Explicit Feature Extraction

Researchers can specify structured variables to extract from clinical text using an LLM (via vLLM or an OpenAI-compatible API). Each feature declares its causal role with `roles`: `"confounder"`, `"effect_modifier"`, or both.

For neural heads, extracted features are featurized and concatenated to the text feature vector before the causal heads. For causal forests, raw role-specific features are also passed directly: confounder-role features go to `W`, and effect-modifier-role features go to `X`.

```json
{
  "applied_inference": {
    "explicit_features": {
      "enabled": true,
      "features": [
        {
          "name": "performance_status",
          "type": "categorical",
          "categories": ["0", "1", "2", "3", "4"],
          "description": "ECOG performance status",
          "roles": ["confounder", "effect_modifier"]
        },
        {
          "name": "age_at_diagnosis",
          "type": "continuous",
          "description": "Patient age at diagnosis in years",
          "roles": ["confounder"]
        },
        {
          "name": "pdl1_expression",
          "type": "continuous",
          "description": "Tumor PD-L1 expression percentage",
          "roles": ["effect_modifier"]
        }
      ],
      "vllm_mode": "python_api",
      "vllm_model_name": "Qwen/Qwen2.5-7B-Instruct",
      "cache_enabled": true,
      "featurizer_output_dim": 64
    }
  }
}
```

Install extraction support with `pip install -e ".[extraction]"`.

Results are cached to `{dataset_dir}/.oci_cache/` and invalidated automatically if the extraction config changes.

Use `model_type="explicit_feature_forest"` to fit CausalForestDML from extracted structured features only. In that mode, confounder-role features form `W`, effect-modifier-role features form `X`, and no text encoder is trained.

Use `model_type="agentic_explicit_feature_forest"` to let an LLM propose additional explicit variables after an initial explicit-feature forest fit. The agentic path uses nested CV: inner folds decide add/remove/re-role actions, and outer folds report the performance of the whole adaptive search process. Set `applied_inference.clinical_question` to the study question; the proposal agent receives that question plus treatment/outcome column metadata in every prompt.

For each proposed add/re-role candidate, the agentic path now records train-fold role diagnostics before inner-CV acceptance. It fits treatment and outcome regressions adjusted for the current confounder-role features, then fits an outcome model with treatment-by-candidate interactions. Candidates with both treatment and outcome association are flagged as likely confounders; candidates with interaction signal are flagged as likely effect modifiers. These diagnostics are advisory feedback to the proposal agent and artifacts; acceptance still uses nested-CV R-loss and nuisance-AUROC guardrails.

For an empty-start agentic search, use the normal repo runner:

```bash
oci run --config example_configs/agentic_explicit_feature_forest_config.json
```

Set `explicit_features.features` to a role-tagged starting list if a researcher wants to seed known confounders or effect modifiers; leave it empty to let the proposal LLM choose the first variables. The agent and extractor both use OpenAI-compatible endpoints, so the same running vLLM server can serve `agent_server_url` and `vllm_server_url`.

The agent receives a few train-fold text snippets to ground its proposals, but raw snippets are omitted from `agentic_feature_search/agent_decisions.jsonl` by default. Set `architecture.agentic_feature_search.save_agent_context=true` only for non-sensitive debugging runs. Raw proposal-model output is also omitted by default; set `architecture.agentic_feature_search.save_agent_raw_output=true` to persist the exact chat completion content and any provider-exposed reasoning field on each `agent_proposals` event.

Use `model_type="multi_model_agentic_forest"` for the multi-signal sparse
discovery path. This path does not train a neural text encoder. For each outer
training fold, it fits several configured TF-IDF/BoW views with different
learner and n-gram settings. Each view cross-fits treatment and outcome nuisance
models, computes its own R-learner pseudo-target, fits a sparse pseudo-target
model, and sends the per-view outputs plus a cross-view phrase consensus summary
to the proposal agent. The agent proposes explicit confounders and effect
modifiers, optional inner-fold consistency checks stabilize the candidate set,
the extractor materializes selected variables from text, and the final estimator
is an explicit-feature CausalForestDML.

You can seed this path with known variables using
`architecture.multi_model_agentic_forest.prespecified_confounders`,
`prespecified_effect_modifiers`, `prespecified_features`, or
`prespecified_features_json`. These entries use the same `ExplicitFeatureSpec`
shape as `explicit_features.features`; `confounders` and `effect_modifiers`
sections in the JSON file apply those roles automatically. If the same variable
is supplied in both roles, it is extracted once, included in the BoW nuisance and
pseudo-target models alongside the text features, and passed to both `W` and `X`
in the final causal forest.

By default this path uses a broad BoW grid: linear TF-IDF views over 1-1, 1-2,
1-3, and 2-4 word n-grams, plus ExtraTrees and RandomForest views. Set
`architecture.multi_model_agentic_forest.bow_views` to override the grid. The
agent context includes `feature_importance.views` for every view and
`feature_importance.phrase_consensus` for repeated 2-4 token n-gram signals.
Final artifacts are written under `multi_model_agentic_forest/`, including
`bow_view_oof_predictions.parquet`,
`bow_view_feature_importance_by_fold.jsonl`,
`embedding_contrast_evidence_by_fold.jsonl`, `agent_candidate_proposals.jsonl`,
`selected_feature_sets.json`, and `outer_cv_metrics.csv`.

This path can also add embedding-contrast retrieval evidence. Set
`architecture.multi_model_agentic_forest.embedding_contrast.enabled=true` to
pool document chunks into patient-level embeddings, build train-fold treatment,
outcome, and per-view R-pseudo-target contrast directions, and retrieve real text chunks
and concept phrases aligned with those directions. The proposal agent sees the
retrieved chunks as hypothesis-generation evidence; saved artifacts redact raw
retrieved chunk text by default unless
`architecture.agentic_feature_search.save_agent_context=true`.
Contrast directions use weighted patient-level mean differences, for example
mean treated embedding minus mean untreated embedding. Linear-probe AUC is
recorded as a diagnostic by default; set `min_probe_auc > 0` only if you want
to use it as an opt-in retrieval gate. It is not used as the retrieval direction.
When a patient has more chunks than `max_chunks`, embedding contrast keeps the
last chunks by default because later notes are often more recent.

The default local embedding model is `Qwen/Qwen3-Embedding-8B`, loaded through
`sentence-transformers` and cached on disk under the run artifact directory.
Use a smaller model or pre-populated local Hugging Face cache for constrained
hardware.

Minimal configuration:

```json
{
  "applied_inference": {
    "dataset_path": "path/to/dataset.parquet",
    "text_column": "clinical_text",
    "treatment_column": "treatment_indicator",
    "outcome_column": "outcome_indicator",
    "outcome_type": "binary",
    "cv_folds": 5,
    "architecture": {
      "model_type": "multi_model_agentic_forest",
      "multi_model_agentic_forest": {
        "bow_views": [
          {
            "name": "linear_1_3",
            "bow_model": "linear",
            "ngram_range_min": 1,
            "ngram_range_max": 3,
            "min_df": 5,
            "max_df": 0.95,
            "max_features": 30000,
            "logistic_c": 1.0,
            "ridge_alpha": 10.0
          },
          {
            "name": "extratrees_1_3",
            "bow_model": "extratrees",
            "ngram_range_min": 1,
            "ngram_range_max": 3,
            "min_df": 5,
            "max_df": 0.95,
            "max_features": 30000
          }
        ],
        "top_n_features": 100,
        "candidate_consistency_enabled": true,
        "embedding_contrast": {
          "enabled": true,
          "model_name": "Qwen/Qwen3-Embedding-8B",
          "chunk_size_words": 256,
          "chunk_overlap_words": 64,
          "max_chunks": 64,
          "chunk_selection": "last",
          "top_k_chunks_per_tail": 12,
          "max_chunks_per_patient": 2,
          "min_probe_auc": 0.0,
          "pseudo_target_quantile": 0.2,
          "pseudo_target_weighted": true,
          "concept_phrases": [
            "brain metastases",
            "liver metastases",
            "PD-L1 high",
            "poor performance status"
          ]
        },
        "prespecified_confounders": [
          {
            "name": "age",
            "type": "continuous",
            "description": "Patient age at treatment initiation in years."
          }
        ],
        "prespecified_effect_modifiers": [
          {
            "name": "pd_l1_expression",
            "type": "categorical",
            "categories": ["<1%", "1-49%", ">=50%"],
            "description": "Pretreatment tumor PD-L1 expression category."
          }
        ]
      }
    },
    "explicit_features": {
      "enabled": true,
      "features": []
    }
  }
}
```

An optional `prespecified_features_json` file can contain:

```json
{
  "confounders": [
    {"name": "age", "type": "continuous", "description": "Age in years."}
  ],
  "effect_modifiers": [
    {
      "name": "pd_l1_expression",
      "type": "categorical",
      "categories": ["<1%", "1-49%", ">=50%"]
    }
  ]
}
```

## Contrastive Learning

Optional supervised contrastive loss (SupCon) within similarity clusters encourages the model to discriminate treatment/outcome status among clinically similar patients. Features are clustered via K-means, then SupCon is computed within each cluster using treatment x outcome as 4-class labels.

```json
{
  "architecture": {
    "contrastive_enabled": true,
    "contrastive_num_clusters": 4,
    "contrastive_temperature": 0.1,
    "contrastive_label_mode": "joint"
  },
  "training": {
    "contrastive_weight": 0.1
  }
}
```

## Synthetic Data Generation

The `synthetic_data/` module generates synthetic clinical datasets with known causal structure for benchmarking. An LLM creates realistic confounders, treatment/outcome regression equations, and clinical narratives for each patient. Ground-truth treatment effects (ITE, ATE) are known by construction.

```bash
# Generate 500 patients with vLLM batch inference
python -m synthetic_data.cli --use-vllm-batch --dataset-size 500 \
  --output-dir ./my_synthetic_data

# Custom clinical question
python -m synthetic_data.cli --use-vllm-batch --dataset-size 500 \
  --clinical-question "Compare pembrolizumab with nivolumab for advanced NSCLC"
```

### Structured Clinical Data Events

By default, synthetic datasets contain only narrative clinical notes (progress notes, imaging reports, pathology reports, NGS reports). With `--structured-data`, the pipeline also generates structured clinical data events that simulate real-world EHR/claims data converted to text:

- **Encounter records** -- ICD-10 diagnosis codes and CPT/HCPCS procedure codes
- **Laboratory results** -- CBC, CMP, tumor markers with values, units, and reference ranges
- **Hospitalization records** -- principal diagnosis, length of stay, discharge disposition
- **Patient-reported outcomes** -- EORTC QLQ-C30 functional/symptom scores (0-100 scale) and PRO-CTCAE adverse event severity (0-4 scale)

These structured events are generated by the LLM as part of the patient's chronological event timeline (ensuring clinical coherence), then converted to standardized text using deterministic templates. The resulting text blocks are interleaved chronologically with the narrative notes in the final `clinical_text` column.

```bash
# Enable all structured data types
python -m synthetic_data.cli --use-vllm-batch --dataset-size 500 --structured-data

# Selective: only encounters and labs
python -m synthetic_data.cli --use-vllm-batch --dataset-size 500 \
  --structured-data --no-hospitalizations --no-pros
```

Or via JSON config:

```json
{
  "clinical_question": "Compare letrozole+palbociclib with letrozole+ribociclib ...",
  "dataset_size": 500,
  "generation_mode": "two_stage",
  "structured_data": {
    "enabled": true,
    "include_encounters": true,
    "include_labs": true,
    "include_hospitalizations": true,
    "include_pros": true,
    "pro_instruments": ["EORTC_QLQ_C30", "PRO_CTCAE"]
  }
}
```

| CLI Flag | Effect |
|----------|--------|
| `--structured-data` | Enable structured clinical data events |
| `--no-encounters` | Disable encounter records (ICD-10/CPT) |
| `--no-labs` | Disable laboratory results |
| `--no-hospitalizations` | Disable hospitalization records |
| `--no-pros` | Disable patient-reported outcomes |

## Oracle Experiment Script

The `run_oracle_experiments.py` script runs a grid of experiments on synthetic datasets with known ground-truth treatment effects. It compares causal heads (causal_forest, rlearner, dragonnet) and includes a `best_attainable` upper bound computed from ground-truth confounder columns.

Each configuration is repeated N times with different random seeds (default 10) so that summary statistics report mean +/- std.

```bash
# Full grid on 4 GPUs
python oracle_experiment_scripts/run_oracle_experiments.py \
    --output-dir ../my_results \
    --devices cuda:0 cuda:1 cuda:2 cuda:3

# Targeted run: causal forest only, one dataset
python oracle_experiment_scripts/run_oracle_experiments.py \
    --output-dir ../my_results \
    --devices cuda:0 \
    --datasets one_confounder_twostage \
    --model-types causal_forest \
    --max-lengths 50000

# Quick test: 1 experiment, 3 epochs
python oracle_experiment_scripts/run_oracle_experiments.py \
    --output-dir ../my_results \
    --devices cuda:0 \
    --max-experiments 1 --epochs 3

# With hidden state caching
python oracle_experiment_scripts/run_oracle_experiments.py \
    --output-dir ../my_results \
    --devices cuda:0 \
    --cache

# 5 repeats instead of default 10
python oracle_experiment_scripts/run_oracle_experiments.py \
    --output-dir ../my_results \
    --n-repeats 5
```

Multi-model agentic forest oracle run, assuming compatible agent/extraction
servers are already running:

```bash
python oracle_experiment_scripts/run_oracle_multi_model_agentic_forest.py \
  --dataset synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured \
  --output-dir ../pcori_experiments/oracle_multi_model_agentic_forest_smoke \
  --n-folds 5 \
  --nuisance-folds 5 \
  --effect-folds 5 \
  --bow-view-grid default_broad \
  --agent-model-name Qwen/Qwen3.5-35B-A3B \
  --extraction-model-name Qwen/Qwen3.5-35B-A3B \
  --extraction-reasoning-parser qwen3
```

Add `--enable-embedding-contrast --embedding-model-name Qwen/Qwen3-Embedding-8B`
to include the embedding-delta retrieval evidence in the same oracle run.

Agentic attention-variable forest smoke test, assuming a Qwen vLLM server is
already running with `--reasoning-parser qwen3`:

```bash
.venv/bin/python oracle_experiment_scripts/run_oracle_agentic_attention_variable_forest_experiments.py \
  --datasets synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured \
  --output-dir ../pcori_experiments/oracle_agentic_attention_variable_forest_smoke \
  --htr-sentence-model Qwen/Qwen3-Embedding-0.6B \
  --n-folds 5 \
  --nuisance-folds 5 \
  --effect-folds 5 \
  --epochs 25 \
  --extraction-batch-size 16 \
  --agent-model-name Qwen/Qwen3.5-35B-A3B \
  --extraction-model-name Qwen/Qwen3.5-35B-A3B \
  --extraction-reasoning-parser qwen3
```

Omit `--sample-size` and `--text-max-chars` to run on the full dataset with
untruncated clinical text.

Options:

| Flag | Description |
|------|-------------|
| `--output-dir` | Directory for results |
| `--devices` | GPU devices (e.g., `cuda:0 cuda:1`) |
| `--datasets` | Filter datasets (e.g., `one_confounder_twostage`) |
| `--model-types` | Filter model types (`causal_forest`, `rlearner`, `dragonnet`, `best_attainable`) |
| `--max-lengths` | Filter max sequence lengths (e.g., `5000 50000`) |
| `--epochs` | Training epochs (default: 30) |
| `--n-folds` | CV folds (default: 5) |
| `--n-repeats` | Repeats per config with different seeds (default: 10) |
| `--cache` | Pre-cache LLM hidden states to disk |
| `--gpu-cache` | Keep hidden states in GPU VRAM (implies `--cache`) |
| `--resume` | Resume from existing results |
| `--max-experiments` | Limit number of experiments |

### Analyzing Results

```bash
python oracle_experiment_scripts/analyze_results.py \
    --results-dir ../my_results/results
```

Produces a comprehensive analysis with group comparisons, pairwise t-tests, and cross-tabulated results across experimental factors.

## Output Files

```
output_dir/
  config.json
  applied_inference/
    predictions.parquet        # Treatment effect estimates
    training_log.csv
    psm_analysis/              # If matching analysis enabled
```

The `predictions.parquet` contains:

| Column | Description |
|--------|-------------|
| `pred_y0_prob` | Predicted outcome under no treatment |
| `pred_y1_prob` | Predicted outcome under treatment |
| `pred_ite_prob` | Individual treatment effect (y1 - y0) |
| `pred_propensity_prob` | Treatment propensity score |
| `pred_ite_lower` | Lower 95% CI bound (causal forest only) |
| `pred_ite_upper` | Upper 95% CI bound (causal forest only) |

## Dependencies

**Core**: torch, transformers, pandas, numpy, scikit-learn, econml, accelerate

**Optional**: openai (explicit feature extraction via `pip install -e ".[extraction]"`)

## Citation

```bibtex
@software{oci2024,
  author = {Kehl, Ken},
  title = {Oncology Causal Inference: Treatment Effect Estimation from Clinical Text},
  year = {2024},
  url = {https://github.com/kenlkehl/onc-causal-inference}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
