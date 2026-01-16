# CLAUDE.md - CDT (Causal DragonNet Text) Codebase Reference

## Project Overview

CDT is a framework for **clinical causal inference using electronic health record (EHR) text**. It estimates treatment effects on **continuous outcomes (survival in months)** from unstructured clinical narratives using neural networks with DragonNet-style causal inference heads.

**Core research goal**: Extract confounders from clinical text that may not be captured in structured EHR data (e.g., functional status, symptom severity, patient preferences) and use them for causal inference.

**Outcome type**: Continuous (survival in months, no censoring). Outcome heads use Softplus activation to ensure positive predictions.

## Repository Structure

```
causal-dragonnet-text/
├── cdt/                          # Main causal inference package
│   ├── cli.py                    # CLI entry point: `cdt init`, `cdt run`
│   ├── config.py                 # Configuration dataclasses
│   ├── models/                   # Neural network architectures
│   │   ├── cnn_extractor.py      # Word-level 1D CNN feature extractor
│   │   ├── bert_extractor.py     # HuggingFace transformer (CLS token)
│   │   ├── dragonnet.py          # DragonNet causal head
│   │   ├── uplift.py             # UpliftNet alternative architecture
│   │   ├── causal_cnn.py         # Combined model (extractor + head)
│   │   └── propensity_model.py   # Propensity-only model for trimming
│   ├── inference/
│   │   └── applied.py            # Applied inference on real data
│   ├── training/
│   │   ├── plasmode.py           # Plasmode simulation experiments
│   │   └── propensity_trimming.py # Propensity-based dataset trimming
│   ├── data/
│   │   └── dataset.py            # PyTorch datasets and loading
│   ├── experiments/
│   │   └── runner.py             # Main experiment orchestrator
│   └── utils/                    # Utility functions
├── synthetic_data/               # LLM-based synthetic data generation
│   ├── cli.py                    # CLI: `python -m synthetic_data.cli`
│   ├── config.py                 # Generation configuration
│   ├── generator.py              # Main generation pipeline
│   ├── llm_client.py             # OpenAI-compatible API client
│   ├── vllm_batch_client.py      # Direct vLLM batch inference
│   └── prompts.py                # LLM prompts for data generation
└── examples/                     # Example configuration files
```

## Key Concepts

### 1. Architecture: Text → Features → Causal Estimates

```
Clinical Text → Feature Extractor (CNN/BERT) → DragonNet → [P(T|X), E[Y|T=0,X], E[Y|T=1,X]]
                                                              ↓
                                                        ITE = E[Y|T=1,X] - E[Y|T=0,X]
```

- **Feature Extractor**: Converts text to fixed-dimensional vector
  - **CNN** (`cnn_extractor.py`): Word-level tokenization, 1D convolutions with multiple kernel sizes (n-gram detection), global max pooling. Faster to train.
  - **BERT** (`bert_extractor.py`): Any HuggingFace transformer, CLS token extraction, optional projection layer. Higher capacity.

- **DragonNet** (`dragonnet.py`): Jointly predicts:
  - Treatment propensity: P(T=1|X) (binary, sigmoid)
  - Potential outcomes: E[Y|T=0,X] and E[Y|T=1,X] (continuous, Softplus for positivity)
  - Uses shared representation layers, separate outcome heads
  - ITE = E[Y|T=1,X] - E[Y|T=0,X] (in months)

### 2. CNN Semantic Initialization

Key innovation: CNN filters can be initialized with clinical meaning:

1. **Explicit concepts**: User-specified phrases (e.g., "stage iv cancer", "performance status poor") become filter templates
2. **K-means filters**: Cluster training n-grams and use centroids as filters
3. **Random filters**: Standard random initialization

Embeddings can be initialized from BERT (e.g., Bio_ClinicalBERT).

### 3. Workflow Modes

**Applied Inference** (`cdt/inference/applied.py`):
- Run on real clinical data
- K-Fold CV or fixed train/val/test splits
- Outputs: predictions.parquet with ITE estimates

**Plasmode Simulation** (`cdt/training/plasmode.py`):
- Uses real text covariates, generates synthetic outcomes with known treatment effects
- Validates method: Can the model recover the true ATE?
- Generator model learns representations, synthetic outcomes generated, evaluator model trained on synthetic data

## CLI Commands

### Main CDT CLI

```bash
# Create default config
cdt init --output config.json

# Run experiment
cdt run --config config.json [--device cuda:0] [--workers 4] [--skip-plasmode]
```

### Synthetic Data Generation

```bash
# Using OpenAI-compatible API (e.g., vLLM server)
python -m synthetic_data.cli \
    --api-url http://localhost:8000/v1 \
    --model openai/gpt-oss-120b \
    --dataset-size 500 \
    --clinical-question "Compare treatment A with treatment B for condition X"

# Direct vLLM batch inference (faster, no server)
python -m synthetic_data.cli \
    --use-vllm-batch \
    --model path/to/model \
    --tensor-parallel-size 2 \
    --dataset-size 1000
```

## Configuration Reference

### Key Config Classes (`cdt/config.py`)

```python
ExperimentConfig                    # Top-level config
├── applied_inference: AppliedInferenceConfig
│   ├── dataset_path: str           # Path to parquet/csv
│   ├── text_column: str            # Column name for text
│   ├── outcome_column: str         # Column name for outcome
│   ├── treatment_column: str       # Column name for treatment
│   ├── cv_folds: int               # K-fold CV (0/1 = fixed splits)
│   ├── architecture: ModelArchitectureConfig
│   │   ├── feature_extractor_type: str  # "cnn" or "bert"
│   │   ├── cnn_*                   # CNN-specific params
│   │   ├── bert_*                  # BERT-specific params
│   │   └── dragonnet_*             # DragonNet head params
│   ├── training: TrainingConfig
│   │   ├── epochs, batch_size, learning_rate
│   │   ├── alpha_propensity: float # Weight for propensity loss
│   │   └── beta_targreg: float     # Targeted regularization weight
│   └── propensity_trimming: PropensityTrimmingConfig
└── plasmode_experiments: PlasmodeExperimentConfig
    ├── enabled: bool
    ├── num_repeats: int
    ├── plasmode_scenarios: List[PlasmodeConfig]
    │   ├── generation_mode: str    # e.g., "phi_linear"
    │   ├── baseline_control_outcome_mean: float  # Mean survival (months) for control
    │   ├── target_ate: float       # True ATE in months
    │   └── outcome_noise_std: float  # Noise std in log-space
    └── generator_*/evaluator_*     # Separate configs for gen/eval
```

### Dataset Format

Parquet or CSV with columns:
- `clinical_text`: Raw clinical narrative text
- `treatment_indicator`: Binary (0/1)
- `outcome_indicator`: Continuous (survival in months)
- `split` (optional): "train"/"val"/"test" for fixed splits

## Important Code Patterns

### Training a Model (from applied.py)

```python
# Create model
model = CausalCNNText(
    feature_extractor_type="cnn",  # or "bert"
    embedding_dim=128,
    kernel_sizes=[3, 4, 5, 7],
    dragonnet_representation_dim=128,
    device="cuda:0"
)

# For CNN: MUST fit tokenizer before training
model.fit_tokenizer(train_texts)

# Optional: Initialize embeddings from BERT
model.feature_extractor.init_embeddings_from_bert("emilyalsentzer/Bio_ClinicalBERT")

# Optional: Initialize filters from concepts/k-means
model.feature_extractor.init_filters(texts=train_texts)

# Training loop
for batch in train_loader:
    losses = model.train_step(batch, alpha_propensity=1.0, beta_targreg=0.1)
    losses['loss'].backward()
    optimizer.step()

# Prediction
preds = model.predict(texts)  # Returns dict with y0_pred, y1_pred, ite, propensity
```

### Key Model Methods

- `CausalCNNText.fit_tokenizer(texts)`: Required for CNN, builds vocabulary
- `CausalCNNText.train_step(batch, alpha_propensity, beta_targreg)`: Returns loss dict
- `CausalCNNText.predict(texts)`: Returns predictions (continuous outcomes in months)
  - `y0_pred`: Predicted survival under control
  - `y1_pred`: Predicted survival under treatment
  - `ite`: Individual treatment effect (y1 - y0, in months)
  - `propensity`: Treatment probability
- `CausalCNNText.get_features(texts)`: Extract feature representations
- `CNNFeatureExtractor.interpret_filters(texts)`: Post-hoc filter interpretation

## Synthetic Data Generation Pipeline

```
1. Generate confounders (LLM)
2. Generate regression equations (LLM) - treatment/outcome models
3. Generate summary statistics (LLM) - confounder distributions
4. Rescale treatment coefficients for target logit std
5. Calibrate treatment intercept for target treatment rate
6. For each patient:
   a. Sample characteristics from distributions
   b. Compute treatment probability, sample treatment (binary)
   c. Generate outcome using log-normal distribution (positive, continuous)
   d. Compute potential outcomes Y0, Y1 and ITE (in months)
   e. Generate clinical history text (LLM)
7. Save dataset + metadata
```

Outputs:
- `dataset.parquet`: Patient data with text and ground truth
- `metadata.json`: Confounders, equations, generation config
- `generation_config.json`: Input configuration

## Loss Function Components

```python
total_loss = outcome_loss + alpha_propensity * propensity_loss + beta_targreg * targreg_loss
```

- **outcome_loss**: MSE on factual outcome (continuous survival, Y given actual T)
- **propensity_loss**: BCE on treatment prediction (binary)
- **targreg_loss**: Targeted regularization (R-loss) for debiasing

## Propensity Trimming

Optional preprocessing to enforce positivity assumption:
1. Train propensity-only model with CV
2. Remove patients with P(T|X) outside [min, max] bounds
3. Proceed with DragonNet training on trimmed data

## Common Modifications

### Adding a new feature extractor
1. Create class in `cdt/models/` with `forward(texts) -> tensor`
2. Add to `CausalCNNText.__init__` initialization logic
3. Update config in `cdt/config.py`

### Modifying the loss function
Edit `CausalCNNText.train_step()` in `cdt/models/causal_cnn.py`

### Adding a new plasmode generation mode
Edit `_generate_plasmode_data()` in `cdt/training/plasmode.py`

## Development Commands

```bash
# Install in editable mode
uv pip install -e .

# Run with specific config
cdt run --config examples/bert_config.json --device cuda:0

# Generate synthetic data
python -m synthetic_data.cli --config my_config.json --dataset-size 500
```

## Key Files to Understand First

1. `cdt/models/causal_cnn.py` - Combined model architecture
2. `cdt/config.py` - All configuration options
3. `cdt/inference/applied.py` - Training and prediction workflow
4. `synthetic_data/generator.py` - Data generation pipeline
