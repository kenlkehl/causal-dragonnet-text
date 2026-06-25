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

Start with honest cross-fitted text models. The purpose is to learn which concepts the text suggests, not to let clinical priors define the variable list. Run both BoW/TF-IDF and HTR/attention evidence before finalizing candidate features: BoW gives broad lexical recurrence, while HTR/attention localizes the specific spans needed to recover baseline/index-time values in long longitudinal notes. BoW evidence should come from multiple vectorization strategies so a signal is not missed merely because one n-gram or document-frequency setting hides it.

Run a BoW/TF-IDF suite:
- Use the same honest folds across vectorization variants when possible so feature recurrence can be compared directly.
- Include these variants when feasible:
  - unigram-focused terms for broad clinical concepts
  - the default broad `1-3` n-gram setup
  - phrase-focused `2-4` n-grams for multiword statuses, regimens, and lab names
  - lower-`min_df` or higher-`max_features` settings for rare but fold-stable signals
- Fit treatment nuisance models on training folds and predict held-out rows.
- Fit outcome nuisance models on training folds and predict held-out rows.
- Compute held-out residuals and R-learner or pseudo-outcome targets.
- Fit text models for residual/pseudo-outcome signal.
- Save high-signal n-grams by fold and vectorization run for treatment, outcome, overlap, and effect heterogeneity.
- Compare consensus, disagreement, and unique discoveries across vectorization runs before translating text evidence into candidate concepts.

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

Do not use regex or pattern-matching rules as the method for extracting clinical variables from notes. The extraction step must read the patient document and return structured values, missingness flags, and brief evidence/rationale for each requested concept. The coding agent itself is a valid LLM-based document-reading extractor. Values that cannot be recovered by document reading remain missing/null, but only after the patient text has actually been read for that concept.

Default to **coding agent extraction** unless the user supplied an endpoint, requested endpoint-backed extraction, or the dataset is too large for reliable coding-agent extraction after sharding has been attempted or explicitly bounded. Lack of a repo extraction package, local model, vLLM server, OpenAI-compatible endpoint, or API key is not a blocker and must not result in an all-missing feature table; it means the coding agent must perform the document-reading extraction directly.

Supported backends:
- **Coding agent extraction**: the agent reads the `clinical_text` documents and produces structured values itself. This is the required fallback when no external/model endpoint exists. For long documents, use targeted chunk sampling, section search, BoW/HTR-highlighted spans, and subagents sharded by patient, fold, concept, or document chunk to find relevant passages without loading every full document into one context window. A coordinating pass must reconcile chunk-level findings into one patient-level value per concept with missingness flags and evidence summaries. If subagents are unavailable, perform the same sharded extraction manually across repeated passes.
- **OpenAI-compatible endpoint extraction**: use a running endpoint such as vLLM, plus model name, API key if needed, batch/concurrency limits, and max text length. Use this when coding-agent extraction is infeasible or when the user explicitly provides or requests it.
- **Local Hugging Face GPU extraction**: prefer this as an endpoint-backed path, not as an ad hoc `transformers.generate()` loop. If local model weights and CUDA GPUs are available, first attempt to start a vLLM OpenAI-compatible server and use the repository's endpoint-backed extractor. Choose a current instruction-tuned model with enough context and extraction reliability for clinical notes; Google's `gemma-4-e2b-it`/Gemma 4 E2B instruct is an example of a better local extraction candidate when cached and supported by vLLM. Treat small/base models such as `Qwen/Qwen3-1.7B` or `Qwen/Qwen3-0.6B-Base` as smoke-test models only, unless stronger instruction models and server mode are unavailable and the limitation is explicitly documented.

For local vLLM-backed extraction:
- Probe GPUs as in the HTR section, then choose explicit `CUDA_VISIBLE_DEVICES`, dtype, tensor parallelism, GPU memory utilization, and max model length.
- Default to very long context for extraction. Aim for about 200,000 tokens of total server context when supported so full or near-full longitudinal patient histories can be included. Set vLLM `--max-model-len 200000` or the largest stable supported value near that target; set the extractor's max text/input token limit near the same value.
- Set generation `max_tokens`/`max_new_tokens` deliberately high enough for reasoning-model overhead and complete JSON output, not a small value such as a few hundred tokens. For reasoning-capable models, reserve thousands to tens of thousands of output tokens as needed while ensuring `input_tokens + max_new_tokens <= max_model_len`.
- Start vLLM with an OpenAI-compatible API server, for example adapting the local paths/options:
  `python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port <port> --model <local-or-hf-model> --served-model-name <name> --dtype bfloat16 --max-model-len 200000`.
- Run a small JSON extraction smoke test against the server before the full pass, including at least one numeric variable and one categorical variable from a long note.
- Configure the repo extractor with `explicit_features.vllm_mode="server"`, `explicit_features.vllm_server_url`, `explicit_features.vllm_model_name`, deterministic temperature, bounded concurrency, retries, cache enabled, a max input/text length near the 200k-token target, and a generation token cap large enough for reasoning overhead and JSON completion.
- Record the server command, model, context limit, max input/text tokens, max generation tokens, prompt version, smoke-test result, and any fallback reason in `report.txt` and `candidate_features` metadata.
- Use direct HF `transformers` generation only when vLLM/server mode is unavailable or incompatible; if used, keep the same model-quality standard and document why direct generation was necessary.

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

Extraction prompts/instructions must request baseline/pre-treatment values and instruct the extractor to return null when the value is not explicitly stated or cannot be inferred from pre-treatment information. Coding-agent and endpoint-backed extraction must both preserve the same feature schema, missingness flags, temporal labels, and brief evidence summaries. Preserve raw LLM responses only when safe and explicitly useful; otherwise record summarized evidence, coverage, and missingness.

Post-extraction type coercion, category canonicalization, and sanity checks may operate only on values already produced by document-reading extraction. Do not use regex as a fallback extractor, targeted repair extractor, or missing-value filler; unresolved patient/concept values remain missing/null and must be documented in `report.txt`. A run-level failure to find an endpoint is not a valid missingness rationale. If coding-agent extraction is genuinely infeasible after a concrete sharding attempt, stop and report the blocker instead of proceeding as though extraction succeeded.

## 4. Role Evaluation

Evaluate confounders and modifiers separately. Use univariable screens as fold-aware diagnostics before and during multivariable modeling; they are useful for prioritization and debugging, but not sufficient as final role evidence. Run these diagnostics only inside internal training folds for any selection decision.

Univariable nuisance screens:
- For every extracted candidate, screen association with treatment in training folds or with out-of-fold predictions.
- For every extracted candidate, screen association with outcome and whether it improves a simple outcome nuisance model.
- For plausible confounders, evaluate supported functional relationships such as monotone transforms, ratios, coarse categories, or interactions only when text evidence or diagnostics suggest them.
- Record standardized effect magnitude, score deltas, direction, fold recurrence, missingness, and whether the association survives basic adjustment for already selected high-confidence confounders.
- Include p values when useful, but do not rank or retain candidates mainly because a p value is small.

Confounder evidence:
- Predicts treatment in held-out folds.
- Predicts outcome or improves held-out outcome nuisance metrics after adjustment.
- Reduces residual treatment-outcome association or improves overlap diagnostics.
- Has a magnitude and direction that are stable enough to matter, not merely a nominal p value.

Univariable effect-modification screens:
- Fit treatment-by-candidate interaction screens for binary outcomes or residualized outcome models.
- Evaluate differential association with outcome by treatment using treatment-stratified outcome associations or interaction score deltas.
- Compare subgroup residual slopes or subgroup ATE proxies only when overlap is adequate.
- Screen candidate association with R-loss targets, logistic R-loss targets, or pseudo-outcomes using out-of-fold nuisance estimates.
- Require fold-stable direction and interpretable span/value evidence before promoting a modifier.

Effect-modifier evidence:
- Improves held-out treatment-by-feature interaction objectives.
- Improves R-loss, logistic R-loss, or pseudo-outcome MSE.
- Produces fold-stable heterogeneity and ITE rankings.
- Need not be strongly prognostic marginally.

Avoid promoting variables based only on marginal outcome association.

Parsimony and redundancy review:
- Before finalizing a candidate list, compute feature-feature correlations, contingency tables for categorical pairs, and missingness-overlap summaries within training folds.
- Group highly correlated or semantically duplicate candidates and prefer the most direct baseline variable unless a proxy improves honest treatment/outcome nuisance, heterogeneity, or ITE-stability metrics.
- Prefer a compact feature set that preserves held-out performance and fold-stable role evidence over a larger set of weakly distinct variables.
- Revisit role assignments after each review: a variable may be a confounder, an effect modifier, both, or neither.

## 5. Iteration

After each iteration:
- Compare fold-level nuisance metrics, R-loss/pseudo-outcome metrics, causal-forest stability, and ITE distribution.
- Identify unexplained residual text signal and propose a narrow expansion only if metrics or fold evidence justify it.
- Re-extract or re-role candidates when aliasing, missingness, or temporal leakage is suspected.
- Mull over the candidate list repeatedly before finalization: merge redundant variables, reject weak proxies, test supported transformations, and rerun role diagnostics on the revised list.
- Stop when new variables do not improve honest metrics, do not recur across folds, do not stabilize univariable screens, do not resolve temporal anchoring, and do not improve ITE stability.

## 6. Functional Form and ITEs

Do not assume the final DGP is linear. Compare:
- flexible nuisance/effect models
- tree/forest-based heterogeneity
- interactions suggested by R-loss or pseudo-outcomes
- simple parametric summaries only as a compact explanation

For every patient, produce counterfactual outcome estimates and ITEs only from models that did not train on that row, or from a final model whose selection was completed using nested honest folds and whose report clearly states the refit convention.
