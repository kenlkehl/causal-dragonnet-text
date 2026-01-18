# CLAUDE.md - CDT (Causal DragonNet Text)

## Project Overview

CDT is a framework for **clinical causal inference using electronic health record (EHR) text**. It estimates treatment effects from unstructured clinical narratives by combining text feature extraction with DragonNet causal inference heads.

**Key purpose**: Extract confounders from clinical text to estimate individual treatment effects (ITEs) and average treatment effects (ATEs) in observational studies where critical confounders exist only in unstructured notes.

## Repository Structure

```
cdt/                          # Main package
├── cli.py                    # CLI entry point (`cdt init`, `cdt run`)
├── config.py                 # Dataclass configurations (ExperimentConfig, ModelArchitectureConfig, etc.)
├── data/
│   └── dataset.py            # ClinicalTextDataset, data loading/validation
├── experiments/
│   └── runner.py             # ExperimentRunner - orchestrates applied inference & plasmode
├── inference/
│   └── applied.py            # Run applied causal inference (CV or fixed split)
├── models/
│   ├── causal_cnn.py         # CausalCNNText - main model combining extractor + DragonNet
│   ├── cnn_extractor.py      # CNNFeatureExtractor - 1D CNN with semantic filter init
│   ├── bert_extractor.py     # BertFeatureExtractor - HuggingFace transformer CLS token
│   ├── gru_extractor.py      # GRUFeatureExtractor - BiGRU with attention pooling
│   ├── dragonnet.py          # DragonNet head (propensity + potential outcomes)
│   ├── uplift.py             # UpliftNet head (alternative parametrization)
│   ├── outcome_heads.py      # Outcome prediction heads
│   └── propensity_model.py   # Standalone propensity model
├── training/
│   ├── plasmode.py           # Plasmode simulation experiments
│   ├── propensity_trimming.py# Propensity-based dataset trimming
│   └── outcome_training.py   # Standalone outcome model training
└── utils/                    # Utilities (io, system, etc.)

synthetic_data/               # LLM-based synthetic data generation
├── generator.py              # Main pipeline for generating synthetic datasets
├── config.py                 # SyntheticDataConfig
├── prompts.py                # Clinical prompts for LLM
├── llm_client.py             # OpenAI API client
└── vllm_batch_client.py      # vLLM batch inference client

examples/                     # Example config files
├── semantic_cnn_config.json  # CNN with explicit clinical concepts
├── bert_config.json          # BERT feature extractor config
└── modernbert_config.json    # ModernBERT config
```

## Architecture

### Core Model: CausalCNNText (`cdt/models/causal_cnn.py`)

The main model combines:
1. **Feature Extractor** (one of three types):
   - `cnn`: 1D CNN with word-level tokenization (default, fastest)
   - `bert`: HuggingFace transformer (Bio_ClinicalBERT, ModernBERT, etc.)
   - `gru`: Bidirectional GRU with attention (O(N) for long sequences)

2. **Causal Inference Head** (one of two types):
   - `dragonnet`: Classic DragonNet (propensity + Y0/Y1 potential outcomes)
   - `uplift`: UpliftNet (base outcome + treatment effect parametrization)

### CNN Feature Extractor (`cdt/models/cnn_extractor.py`)

Key features:
- **Semantic filter initialization**: CNN filters can be initialized from explicit clinical concepts (e.g., "stage iv cancer", "performance status poor")
- **K-means filter initialization**: Additional filters derived from clustering training n-grams
- **BERT embedding initialization**: Word embeddings initialized from Bio_ClinicalBERT
- **Filter interpretability**: `interpret_filters()` method shows which n-grams activate each filter

**Important**: Call `fit_tokenizer(texts)` before training with CNN/GRU extractors.

### DragonNet (`cdt/models/dragonnet.py`)

Outputs:
- `y0_logit`: Predicted outcome under control (T=0)
- `y1_logit`: Predicted outcome under treatment (T=1)
- `t_logit`: Treatment propensity logit
- Individual Treatment Effect (ITE) = sigmoid(y1_logit) - sigmoid(y0_logit)

Loss function components:
- Outcome loss (factual only)
- Propensity loss (binary cross-entropy)
- Targeted regularization (R-loss)

## Key Commands

```bash
# Initialize default config
cdt init --output config.json

# Run experiment
cdt run --config config.json --device cuda:0 --workers 4

# CLI options
cdt run --config config.json \
    --device cuda:1 \
    --workers 4 \
    --output-dir ./results \
    --skip-plasmode \
    --verbose
```

## Configuration (`cdt/config.py`)

Main config classes:
- `ExperimentConfig`: Top-level config
- `AppliedInferenceConfig`: Dataset paths, column names, CV folds
- `ModelArchitectureConfig`: Feature extractor type, CNN/BERT/GRU params, DragonNet dims
- `TrainingConfig`: Learning rate, epochs, batch size, loss weights
- `PropensityTrimmingConfig`: Pre-trimming by propensity scores
- `PlasmodeConfig`: Plasmode simulation parameters

Key architecture settings:
```python
# Feature extractor type
feature_extractor_type: str = "cnn"  # "cnn", "bert", or "gru"

# CNN-specific
cnn_embedding_dim: int = 128
cnn_kernel_sizes: List[int] = [3, 4, 5, 7]
cnn_explicit_filter_concepts: Dict[str, List[str]]  # kernel_size -> concepts
cnn_num_kmeans_filters: int = 64
cnn_init_embeddings_from: str = "emilyalsentzer/Bio_ClinicalBERT"

# BERT-specific
bert_model_name: str = "bert-base-uncased"
bert_max_length: int = 512
bert_freeze_encoder: bool = False

# GRU-specific
gru_hidden_dim: int = 256
gru_num_layers: int = 2
gru_max_length: int = 8192  # Efficient for long sequences
```

## Dataset Format

Expected columns in Parquet/CSV:
| Column | Type | Description |
|--------|------|-------------|
| `clinical_text` | string | Clinical narrative text |
| `treatment_indicator` | int/float | Binary treatment (0 or 1) |
| `outcome_indicator` | int/float | Binary outcome (0 or 1) |
| `split` | string | Optional: "train", "val", "test" |

## Workflow Modes

### 1. Applied Inference
Estimates treatment effects on real data:
- K-fold cross-validation (default: 5 folds)
- Fixed train/val/test splits
- Outputs: `predictions.parquet` with `pred_ite_prob`, `pred_propensity_prob`, etc.

### 2. Plasmode Simulation
Generates synthetic outcomes with known ground truth for method validation:
1. Train "generator" model on real data
2. Generate synthetic outcomes with known ATE
3. Train "evaluator" on synthetic data
4. Compare estimated vs. true treatment effects

## Synthetic Data Generation (`synthetic_data/`)

LLM-based pipeline for generating synthetic clinical datasets:
1. Generate confounders from clinical question
2. Generate treatment/outcome regression equations
3. Generate summary statistics
4. Sample patient characteristics and generate clinical histories

Supports both OpenAI API and local vLLM batch inference.

## Key Files for Development

- **Main model**: `cdt/models/causal_cnn.py` (CausalCNNText)
- **Training loop**: `cdt/inference/applied.py` (_train_single_model, _train_epoch)
- **Config**: `cdt/config.py`
- **CLI**: `cdt/cli.py`
- **Dataset**: `cdt/data/dataset.py`

## Common Patterns

### Training a model manually
```python
from cdt.models.causal_cnn import CausalCNNText

model = CausalCNNText(
    feature_extractor_type="cnn",
    embedding_dim=128,
    kernel_sizes=[3, 4, 5, 7],
    num_kmeans_filters=64,
    device="cuda:0"
)

# IMPORTANT: Fit tokenizer before training
model.fit_tokenizer(train_texts)

# Optional: Initialize embeddings from BERT
model.feature_extractor.init_embeddings_from_bert("emilyalsentzer/Bio_ClinicalBERT")

# Optional: Initialize semantic filters
model.feature_extractor.init_filters(train_texts)

# Training loop
for batch in dataloader:
    losses = model.train_step(batch, alpha_propensity=1.0, beta_targreg=0.1)
    losses['loss'].backward()
    optimizer.step()
```

### Getting predictions
```python
preds = model.predict(texts)
# preds contains: y0_prob, y1_prob, propensity, y0_logit, y1_logit, t_logit

# Individual treatment effect (probability scale)
ite = preds['y1_prob'] - preds['y0_prob']
```

## Dependencies

Core: torch, transformers, pandas, numpy, scikit-learn, tqdm, pyarrow
Optional: openai (for synthetic data generation)

## Output Files

```
output_dir/
├── config.json                 # Experiment configuration
├── applied_inference/
│   ├── predictions.parquet     # Per-sample predictions
│   └── training_log.csv        # Training metrics
└── plasmode_experiments/       # If enabled
    └── results.csv             # Plasmode results
```

## Notes for Development

1. **Feature extractor type**: CNN is fastest, BERT is most expressive, GRU is efficient for long sequences
2. **Tokenizer fitting**: CNN/GRU require `fit_tokenizer()` before use; BERT uses pretrained tokenizer
3. **Filter initialization**: Semantic filters improve interpretability; k-means captures data patterns
4. **Propensity trimming**: Optional preprocessing to enforce positivity assumption
5. **All predictions are on probability scale**: ITE = P(Y=1|T=1,X) - P(Y=1|T=0,X)
