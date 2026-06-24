# Evidence-First Causal DGP Workflow

## 1. Initial Exploration

- Confirm available files in the task folder and do not inspect parent folders unless the user permits it.
- Load the dataset and identify patient id, text, treatment, outcome, and any existing non-oracle covariates.
- Summarize:
  - row count and missing text
  - treatment and outcome rates
  - treatment/outcome cross-tab
  - note length distribution
  - repeated note sections and likely chronology
- Record all of this in `report.txt`.

## 2. Text Evidence Before Feature Extraction

Start with honest cross-fitted text models. The purpose is to learn which concepts the text suggests, not to let clinical priors define the variable list. Run both BoW/TF-IDF and HTR/attention evidence before finalizing candidate features: BoW gives broad lexical recurrence, while HTR/attention localizes the specific spans needed to recover baseline/index-time values in long longitudinal notes.

Run a BoW/TF-IDF pass:
- Fit treatment nuisance models on training folds and predict held-out rows.
- Fit outcome nuisance models on training folds and predict held-out rows.
- Compute held-out residuals and R-learner or pseudo-outcome targets.
- Fit text models for residual/pseudo-outcome signal.
- Save high-signal n-grams by fold for treatment, outcome, overlap, and effect heterogeneity.

Run an HTR/attention evidence pass after the BoW pass:
- Before launching neural jobs, inspect the local accelerator environment:
  - `nvidia-smi` or equivalent GPU inventory, free/used memory, and active processes.
  - `which python`, `sys.executable`, `sys.prefix`, `VIRTUAL_ENV`, `PATH`, `LD_LIBRARY_PATH`, `CUDA_VISIBLE_DEVICES`, available device count, and framework CUDA support from the intended Python environment.
  - `torch.__file__`, `torch.__version__`, `torch.cuda.is_available()`, `torch.cuda.device_count()`, and device names/properties when available.
  - Model size, sequence/chunk length, batch size, precision, and expected memory per worker.
- Handle CUDA mismatch before fallback:
  - If shell-level `nvidia-smi` works but the agent's Python process reports `torch.cuda.is_available() == False`, do not conclude that no GPU is usable.
  - Treat this as a likely sandbox, device-file, driver-library, or inherited-environment mismatch. Record both the shell result and the Python failure in `report.txt`.
  - Rerun the exact CUDA probe with escalated permissions in the same intended interpreter/venv. In Codex, use `sandbox_permissions="require_escalated"` with a concise justification such as "verify whether the workspace sandbox is blocking CUDA/NVML device access while using the intended venv."
  - If escalated PyTorch sees CUDA, run the neural HTR smoke test and full HTR/attention jobs under the escalated context with explicit device pinning.
  - Fall back to CPU span/attention diagnostics only if the escalated probe still cannot use CUDA, the user declines escalation, or GPU memory is genuinely insufficient after a smoke test.
- Choose and record an HTR parallelization plan:
  - Parallelize across CV folds, signal objectives, attention chunk shards, or dataset shards when memory allows.
  - Use explicit device assignment, such as per-process `CUDA_VISIBLE_DEVICES`, so concurrent jobs do not collide.
  - Prefer one model-loading process per GPU unless a smoke test shows multiple workers fit with stable memory headroom.
  - Stagger startup or run a small fold/chunk smoke test before launching all jobs.
  - If no usable GPU is available after the mismatch escalation path above, document the limitation and run the smallest honest HTR pass feasible, or explain why the HTR pass is blocked.
- Train cross-fitted nuisance/effect models with attention or span evidence.
- Inspect top spans for treatment, outcome, and heterogeneity objectives.
- Localize the note section, timepoint, and nearby value/assertion for each candidate concept.
- Use HTR spans to determine whether a signal is baseline/pre-treatment, current-regimen/index-time, historical, post-treatment, or a report-template artifact.
- Pay special attention to numeric slots and derived quantities that BoW cannot represent well, such as ratios, nearest pre-treatment labs, and categorical status values with unknown/missing categories.
- Prefer recurring concepts across folds over one-off patient-template fragments.

Do not extract a broad clinical inventory before this step. Extract only concepts supported by recurring text evidence.

## 3. Candidate Concept Translation

Translate high-signal phrases/spans into baseline patient-level concepts:
- Map aliases and near-duplicates into one extraction target.
- Preserve temporal meaning: baseline/pre-treatment values are valid; post-treatment response/progression text is not a baseline covariate unless the task explicitly asks for post-treatment prediction.
- Keep continuous concepts continuous by default.
- For categorical concepts, define categories before extraction and include an unknown/missing handling rule.
- For longitudinal notes, define the index time and the baseline window before extracting values. Prefer the value nearest to but not after treatment/regimen initiation unless the user asks for post-treatment prediction.
- Where evidence points to a lab or measurement family, consider clinically and textually supported derived quantities, such as neutrophil-to-lymphocyte ratio from paired neutrophil and lymphocyte values.

Examples of valid translation logic:
- Repeated high-weight phrases around "ECOG 2", "requires assistance", and "performance status" can become `baseline_ecog`.
- Repeated spans around neutrophils, lymphocytes, or "NLR" can become `baseline_nlr`.
- Repeated phrases around prior therapy can become `prior_systemic_therapy` or a more specific concept if supported.

## 3.5. LLM-Based Candidate Extraction

Do not make regex the primary method for extracting clinical variables from notes. The extraction step must read the patient document and return structured values, missingness flags, and brief evidence/rationale for each requested concept.

Before extraction, prompt the user to choose one backend unless they already provided it:
- **Coding agent extraction**: the agent reads each `clinical_text` document and produces structured values itself. Use this only when the dataset is small enough or endpoint-backed extraction is unavailable; still preserve the same feature schema and missingness flags.
- **OpenAI-compatible endpoint extraction**: the user provides a running endpoint such as vLLM, plus model name, API key if needed, batch/concurrency limits, and max text length. Prefer this for nontrivial datasets.

Use the repository's LLM extraction implementation when an endpoint is available:
- `oci/extraction/explicit_features.py`: `VLLMFeatureExtractor`, `extract_explicit_features()`, JSON parsing, categorical alias handling, retries, batching, and OpenAI-compatible server mode.
- `oci/extraction/llm_routing.py`: OpenAI-compatible client routing, endpoint pools, retry/backoff helpers.
- `oci/inference/agentic_explicit_feature_forest.py`: `VLLMExplicitFeatureExtractionProvider.ensure_features()` for grouped, cached, resumable extraction of `explicit_feat_*` columns.
- `oci/config.py`: `ExplicitFeatureSpec` and `ExplicitFeatureExtractionConfig`.

Configure each extraction target as an `ExplicitFeatureSpec`-shaped contract:
- `name`
- `type`: `continuous` or `categorical`
- `categories` and optional `value_aliases` for categorical variables
- `description`: exact baseline/pre-treatment extraction target and missingness rule
- `roles`: `confounder`, `effect_modifier`, or both

Extraction prompts must request baseline/pre-treatment values and instruct the model to return null when the value is not explicitly stated or cannot be inferred from pre-treatment information. Preserve raw LLM responses only when safe and explicitly useful; otherwise record summarized evidence, coverage, and missingness.

Regex may be used after LLM extraction for type coercion, category canonicalization, sanity checks, or targeted repair of a known formatting issue. If regex is used as a fallback extractor for any variable, record the reason, scope, and validation against LLM-extracted examples in `report.txt`.

## 4. Role Evaluation

Evaluate confounders and modifiers separately. Use univariable screens as fold-aware diagnostics before and during multivariable modeling; they are useful for prioritization and debugging, but not sufficient as final role evidence.

Univariable nuisance screens:
- For every extracted candidate, screen association with treatment in training folds or with out-of-fold predictions.
- For every extracted candidate, screen association with outcome and whether it improves a simple outcome nuisance model.
- Record effect size/direction, fold recurrence, missingness, and whether the association survives basic adjustment for already selected high-confidence confounders.

Confounder evidence:
- Predicts treatment in held-out folds.
- Predicts outcome or improves held-out outcome nuisance metrics after adjustment.
- Reduces residual treatment-outcome association or improves overlap diagnostics.

Univariable effect-modification screens:
- Fit treatment-by-candidate interaction screens for binary outcomes or residualized outcome models.
- Compare subgroup residual slopes or subgroup ATE proxies only when overlap is adequate.
- Screen candidate association with R-loss targets, logistic R-loss targets, or pseudo-outcomes using out-of-fold nuisance estimates.
- Require fold-stable direction and interpretable span/value evidence before promoting a modifier.

Effect-modifier evidence:
- Improves held-out treatment-by-feature interaction objectives.
- Improves R-loss, logistic R-loss, or pseudo-outcome MSE.
- Produces fold-stable heterogeneity and ITE rankings.
- Need not be strongly prognostic marginally.

Avoid promoting variables based only on marginal outcome association.

## 5. Iteration

After each iteration:
- Compare fold-level nuisance metrics, R-loss/pseudo-outcome metrics, causal-forest stability, and ITE distribution.
- Identify unexplained residual text signal and propose a narrow expansion only if metrics or fold evidence justify it.
- Re-extract or re-role candidates when aliasing, missingness, or temporal leakage is suspected.
- Stop when new variables do not improve honest metrics, do not recur across folds, do not stabilize univariable screens, do not resolve temporal anchoring, and do not improve ITE stability.

## 6. Functional Form and ITEs

Do not assume the final DGP is linear. Compare:
- flexible nuisance/effect models
- tree/forest-based heterogeneity
- interactions suggested by R-loss or pseudo-outcomes
- simple parametric summaries only as a compact explanation

For every patient, produce counterfactual outcome estimates and ITEs only from models that did not train on that row, or from a final model whose selection was completed using nested honest folds and whose report clearly states the refit convention.
