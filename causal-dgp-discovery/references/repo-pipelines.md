# Repository Pipelines

This repository already contains `agentic` discovery and causal-forest code. Treat `agentic` as a repository workflow label: Codex remains the reasoning agent, and repo or vLLM LLM hooks must not be used as autonomous feature-proposal/review agents unless the user explicitly asks for that.

## Required Integrated Text-Evidence Pass

Use `model_type="multi_model_agentic_forest"` for the initial evidence pass. Configure multiple `bow_views` in one run rather than relying on one vectorizer setting, and keep embedding-contrast retrieval plus HTR attention/span evidence enabled unless a run has a documented opt-out reason.

Relevant implementation:
- `oci/inference/multi_model_agentic_forest.py`
- Config dataclass: `MultiModelAgenticForestConfig`
- README section beginning "Use `model_type=\"multi_model_agentic_forest\"`"

Current multi-model stages:
- Build shared honest folds, then fit every configured `bow_view` for treatment nuisance, outcome nuisance, and R-loss/pseudo-target evidence.
- Use the default broad BoW grid when feasible: linear 1-1, 1-2, 1-3, and 2-4 n-gram TF-IDF views plus ExtraTrees and RandomForest views. Supported `bow_model` values are `linear`, `extratrees`, `random_forest`, and `xgboost`.
- Add HTR nuisance treatment/outcome models and attention/span evidence. HTR nuisance predictions join the row-level ensemble treatment/outcome predictions used for R-loss and pseudo-target construction.
- Add an ensemble R target from BoW plus HTR nuisance predictions so Codex can inspect a signal that is less tied to one vectorizer, learner, or neural architecture.
- Add embedding-contrast evidence: treatment, outcome, per-view R-pseudo-target, ensemble R, within-arm outcome, treatment-outcome cell, orthogonal R-score, and concept-probe contrasts from real text chunks.
- Expose highly attended HTR tokens/spans from nuisance and R-stage/effect models to Codex as evidence for confounders and effect modifiers.
- Merge researcher-supplied `prespecified_features`, `prespecified_confounders`, `prespecified_effect_modifiers`, and `prespecified_features_json` into the same `ExplicitFeatureSpec` contract as Codex-proposed features. Duplicate names are harmonized and role lists are merged.
- Run inner-fold candidate consistency checks before extraction when enabled. Use these checks to stabilize proposed concepts, recover strong fold-local candidates, and reject unstable aliases without peeking at held-out reporting rows.
- Canonicalize aliases, categorical categories, and `value_aliases` before extraction so the extractor produces one patient-level column per concept rather than multiple synonymous variables.
- Extract selected features with the repo extractor or with coding-agent document reading, then run the extracted-feature review loop before final causal-forest fitting. Underperforming extracted-feature nuisance or R/pseudo-target diagnostics should trigger Codex revision and targeted re-extraction, capped by `extracted_feature_review_max_rounds`.

Recommended defaults for each run:
- `applied_inference.cv_folds >= 5`
- `architecture.explicit_feature_forest.honest = true`
- `architecture.multi_model_agentic_forest.nuisance_folds >= 5`
- `architecture.multi_model_agentic_forest.effect_folds >= 5`
- `candidate_consistency_enabled = true`
- `candidate_consistency_inner_folds = 3`
- `candidate_consistency_min_folds = 2`
- `candidate_consistency_min_fold_fraction = 0.5`
- `extracted_feature_review_enabled = true`
- `extracted_feature_review_max_rounds = 3`
- `extracted_feature_review_auc_margin = 0.02`
- `extracted_feature_review_loss_relative_margin = 0.05`
- `extracted_feature_review_min_benchmark_auc = 0.55`
- leave `prespecified_confounders`, `prespecified_effect_modifiers`, and `prespecified_features` empty unless the user supplied variables

Minimum `bow_views` suite when feasible:
- unigram-focused: `ngram_range_min = 1`, `ngram_range_max = 1`
- default broad: `ngram_range_min = 1`, `ngram_range_max = 3`
- phrase-focused: `ngram_range_min = 2`, `ngram_range_max = 4`
- rare-signal-friendly: keep broad or phrase n-grams but lower `min_df`, raise `max_features`, or compare `sublinear_tf` settings

Optional learner sensitivity checks can vary `bow_model` among supported values such as `linear`, `extratrees`, `random_forest`, or `xgboost` when runtime allows. Keep fold construction and outcome/treatment definitions fixed across views so recurrence and disagreement are interpretable. Record a view name and vectorizer params for every BoW artifact.

Key artifacts usually appear under `multi_model_agentic_forest/`:
- `bow_view_oof_predictions.parquet`
- `htr_nuisance_oof_predictions.parquet`
- `htr_effect_oof_predictions.parquet`
- `htr_attention_evidence.parquet`
- `text_model_oof_predictions.parquet`
- `bow_view_feature_importance_by_fold.jsonl`
- `embedding_contrast_evidence_by_fold.jsonl`
- `agent_candidate_proposals.jsonl`
- `extracted_feature_diagnostics_by_fold.jsonl`
- `selected_feature_sets.json`
- `outer_cv_metrics.csv`

Use BoW, embedding, and HTR outputs to propose concepts, not as final variables. BoW is mandatory for the broad lexical recurrence pass, embedding contrast is mandatory retrieval evidence for real text chunks and contrastive concept probes, and HTR is mandatory neural attention/span evidence for localization and row-level nuisance stabilization. Prefer concepts that recur across folds and across evidence sources, while preserving source-specific discoveries when the evidence is strong.

## LLM-Based Explicit Feature Extraction

Do not implement clinical variable extraction with ad hoc regex or pattern-matching fallback logic. After BoW/attention evidence identifies candidate concepts, materialize variables by having an LLM read each underlying `clinical_text` document and return structured values. When no external/model endpoint is available, the coding agent itself is the LLM document-reading extractor.

Default to coding-agent extraction unless the user supplied an endpoint, requested endpoint extraction, or the dataset is too large for reliable coding-agent extraction after a concrete sharding attempt:
- **Coding agent extraction**: Codex reads each document, or targeted chunks of each long document, and emits structured values itself. This is required when no repo-native extractor, vLLM server, OpenAI-compatible endpoint, local model, or API key is available. Use BoW/HTR spans, section search, targeted chunk sampling, and subagents when useful, but reconcile all evidence into one patient-level row per concept. For large datasets, shard by patient ranges, folds, concepts, or document chunks and spawn subagents when available; if subagents are unavailable, perform the same sharded passes manually. This must still be document-reading LLM extraction, not regex or pattern matching.
- **OpenAI-compatible endpoint extraction**: use `vllm_server_url` or another OpenAI-compatible base URL, model name, API key if needed, batch size/concurrency, and max text length when provided, requested, or needed after coding-agent extraction has been attempted or clearly bounded for scale. This endpoint is an extraction backend, not a feature-discovery, review, or synthesis agent.
- **Local HF weights on GPUs**: when using cached Hugging Face weights locally, prefer starting a vLLM OpenAI-compatible server and using the endpoint-backed extractor. Select a capable instruction-tuned model with sufficient context, not a small/base model chosen only because it loads quickly. Example preferred class: Google's `gemma-4-e2b-it`/Gemma 4 E2B instruct when cached and supported by vLLM. Reserve direct `transformers` generation with small models such as `Qwen/Qwen3-1.7B` for smoke tests or explicitly documented last-resort extraction.

Do not emit an all-missing explicit-feature table merely because endpoint-backed extraction is unavailable. Missing values are valid only for patient/concept pairs that document-reading extraction could not recover from the text. If extraction is genuinely infeasible even after sharding, stop and report the blocker instead of running downstream causal-forest or final ITE comparisons that require extracted features.

Prefer existing repo code:
- `oci/extraction/explicit_features.py`: `VLLMFeatureExtractor` and `extract_explicit_features()`.
- `oci/inference/agentic_explicit_feature_forest.py`: `VLLMExplicitFeatureExtractionProvider.ensure_features()` for grouped, cached, resumable extraction.
- `oci/extraction/llm_routing.py`: endpoint pooling and retry/backoff.
- `oci/config.py`: `ExplicitFeatureSpec`, `ExplicitFeatureExtractionConfig`.

Typical config fields:
- `applied_inference.explicit_features.enabled=true`
- `explicit_features.features`: role-tagged specs with `name`, `type`, `description`, `roles`, and categorical `categories`/`value_aliases` where needed
- `explicit_features.vllm_mode="server"` for a running vLLM/OpenAI-compatible endpoint
- `explicit_features.vllm_server_url`
- `explicit_features.vllm_model_name`
- `explicit_features.extraction_batch_size`
- `explicit_features.extraction_max_retries`
- `explicit_features.extraction_temperature=0.0`
- `explicit_features.extraction_max_text_length` or equivalent max input/token setting near 200,000 tokens when supported
- `explicit_features.extraction_max_tokens` / endpoint `max_tokens` / `max_new_tokens` set high enough for reasoning overhead and complete JSON output
- `explicit_features.cache_enabled=true`

Local vLLM launch guidance:
- Probe CUDA and GPU memory first, then choose `CUDA_VISIBLE_DEVICES`, `--dtype`, `--tensor-parallel-size`, `--gpu-memory-utilization`, and `--max-model-len` explicitly.
- Aim for a very long extraction context: `--max-model-len 200000` when the model and hardware support it, or the largest stable value near 200k after smoke testing. Keep room for output tokens by enforcing `input_tokens + max_new_tokens <= max_model_len`.
- Start an OpenAI-compatible server with a command shaped like:
  `python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port <port> --model <model-or-local-path> --served-model-name <name> --dtype bfloat16 --max-model-len 200000`.
- Run a JSON-format smoke test through the server before full extraction; the test must include at least one numeric baseline variable and one categorical variable.
- Record the server command, model name/path, context length, max input/text tokens, max generation tokens, prompt version, smoke-test output, and fallback reasons in `report.txt`.

Use `model_type="multi_model_agentic_forest"`, `agentic_explicit_feature_forest`, or `agentic_attention_variable_forest` to keep evidence generation, extraction, and forest fitting on the repo-native path. Do not use the path's LLM proposal/review hooks as the discovery agent by default: Codex should inspect the text evidence, author or revise the `ExplicitFeatureSpec` set, and use endpoint-backed models only for bounded document-reading extraction when allowed.

## Post-Extraction Feature Review

Do not treat a successfully extracted feature table as proof that the variables are useful confounders or effect modifiers. After extraction, run relatively simple fold-honest diagnostics on the extracted variables and compare them with the upstream BoW, embedding, and HTR evidence that motivated the variables.

The repo-native `multi_model_agentic_forest` path performs this review when `extracted_feature_review_enabled=true`:
- Fit extracted-feature treatment nuisance and outcome nuisance models on training folds and score only held-out rows.
- Fit extracted-feature R-loss, logistic R-loss, pseudo-target, or interaction-style effect-modifier diagnostics using out-of-fold nuisance quantities.
- Compare extracted-feature treatment/outcome AUROC or losses against BoW, embedding, and HTR benchmarks. The default gate allows a small AUROC gap (`extracted_feature_review_auc_margin=0.02`) and a small relative loss gap (`extracted_feature_review_loss_relative_margin=0.05`) before treating the extraction as underperforming; `extracted_feature_review_min_benchmark_auc=0.55` prevents weak benchmarks from forcing revisions.
- Review failed gates, missingness, weak role evidence, alias/category problems, and upstream evidence directly in Codex. Codex may drop variables, re-role them, merge aliases, improve categorical `value_aliases`, add narrow evidence-supported concepts, or request targeted re-extraction.
- Cap revisions with `extracted_feature_review_max_rounds`. If the cap is reached, document the remaining benchmark gaps and whether the final forest is exploratory or blocked.

For a custom coding-agent orchestrator, mirror the same discipline: all diagnostics used to accept, reject, or revise features must be computed inside training folds or out-of-fold for the scored rows. Never compare extracted features to BoW, embedding, or HTR models using in-sample predictions.

## Integrated Attention/HTR Evidence Path

`multi_model_agentic_forest` now incorporates HTR nuisance/effect training and attention/span evidence directly. Use `model_type="agentic_attention_variable_forest"` as a deeper standalone or sensitivity pass when additional neural objectives, consensus passes, or residual contrastive attention are needed. HTR is especially important when:
- BoW terms are mostly template artifacts.
- Long notes require localization to baseline/index-time spans.
- Candidate values must be extracted from repeated longitudinal mentions.
- Numeric slots, derived quantities, or categorical status values need nearby context.
- Effect modifiers remain unstable.
- Residual or pseudo-outcome signal is not well explained by extracted concepts.

Relevant implementation:
- `oci/inference/agentic_attention_variable_forest.py`
- Config dataclass: `AgenticAttentionVariableForestConfig`

Try multiple effect objectives when feasible:
- `squared_r_loss`
- `logistic_r_loss` for binary outcomes
- `pseudo_outcome_mse`

Important config fields:
- `nuisance_folds`
- `effect_folds`
- `signal_cv_folds`
- `attention_top_k_chunks`
- `candidate_proposals_per_fold`
- `consensus_min_folds`
- `consensus_min_fold_fraction`
- `neural_stage_mode`
- `effect_objective`

Keep attention runs honest: inspect only held-out or fold-specific evidence for candidate discovery.

## GPU Planning for HTR Runs

Before launching `agentic_attention_variable_forest` or any custom neural span/attention jobs, explore the local GPU environment and decide how to parallelize the work.

Minimum checks:
- Run `nvidia-smi` or the local equivalent to record GPU count, model names, total memory, free memory, utilization, and active processes.
- Check the actual interpreter and environment: `which python`, `sys.executable`, `sys.prefix`, `VIRTUAL_ENV`, `PATH`, `LD_LIBRARY_PATH`, `CUDA_VISIBLE_DEVICES`, `torch.__file__`, `torch.__version__`, `torch.cuda.is_available()`, device count, and device properties.
- Estimate memory pressure from model size, precision, maximum sequence/chunk length, batch size, number of simultaneous workers, and whether the model is loaded once per worker.
- Record the GPU inventory, constraints, and planned worker/device mapping in `report.txt`.

CUDA mismatch escalation:
- If shell `nvidia-smi` works but Python/PyTorch reports no CUDA, assume a sandbox or environment mismatch until proven otherwise.
- Confirm that the Python process is the intended user/project venv, then rerun the identical CUDA probe with escalated permissions. In Codex, request `sandbox_permissions="require_escalated"` and state that the probe checks whether the workspace sandbox is blocking CUDA/NVML access.
- If escalated PyTorch reports CUDA available, run the HTR smoke test and the neural attention pipeline in that escalated context, with explicit `CUDA_VISIBLE_DEVICES` for each worker.
- Only use a CPU fallback when the escalated probe also fails, escalation is declined, or a GPU smoke test shows insufficient memory/headroom.

Parallelization choices:
- Prefer parallelizing independent CV folds, effect objectives, signal types, or attention chunk shards across separate GPUs.
- Use explicit device pinning per process, such as `CUDA_VISIBLE_DEVICES=0`, `CUDA_VISIBLE_DEVICES=1`, etc.
- Do not run multiple full model workers on one GPU unless a smoke test verifies sufficient free memory and stable utilization.
- Stagger process startup to avoid simultaneous model-load spikes.
- If the model supports data parallelism or tensor/model parallelism, use it only when it is already supported by the local pipeline; otherwise prefer independent fold/objective workers.
- If no GPU is available or memory is insufficient, reduce batch size/chunk count, run a minimal honest HTR diagnostic, or document why HTR is blocked.

Use HTR outputs to:
- Identify the note section/timepoint that defines the index treatment or regimen decision.
- Separate baseline/pre-treatment facts from post-treatment response, progression, adverse events, and copied historical summaries.
- Link values to measurement labels, such as ECOG score, creatinine clearance, hemoglobin, neutrophils, lymphocytes, and derived ratios.
- Resolve categorical slots and unknown categories, such as mutation status, prior therapy category, and metastasis status.
- Compare span recurrence with BoW term recurrence before expanding the extraction set.

## Univariable Screening Diagnostics

After extracting an evidence-supported candidate set, run fold-aware univariable screens alongside the multivariable nuisance/effect models.

Recommended screens:
- Treatment nuisance screen: candidate-only model or standardized association with treatment.
- Outcome nuisance screen: candidate-only model or incremental outcome nuisance improvement.
- Confounder screen: candidate association with both treatment and outcome, plus residual balance or overlap impact.
- Effect-modifier screen: treatment-by-candidate interaction, subgroup residual slope, R-loss target association, logistic R-loss target association, or pseudo-outcome association.

Report fold recurrence, direction, standardized effect magnitude or score delta, missingness, overlap warnings, and p values where useful. Put at least as much weight on effect magnitude and fold stability as on p values. Do not select final confounders or modifiers solely from univariable screens; use them to prioritize extraction, diagnose temporal leakage, and decide which multivariable DGP forms to compare.

Before passing final variables to the causal forest:
- Evaluate plausible confounder transformations or functional relationships inside training folds, not on the reporting fold.
- Evaluate effect modifiers for differential association with outcome by treatment, using interaction, treatment-stratified, R-loss, or pseudo-outcome diagnostics.
- Run the mandatory parsimony gate after extracted-feature review and before final forest fitting. Inspect feature-feature correlations, categorical contingency tables, and missingness overlap to merge or reject redundant proxy variables.
- Test weak or redundant variables with honest removal/group-ablation diagnostics where feasible. Iterate on a parsimonious feature list until removing weak or redundant variables no longer preserves honest nuisance, heterogeneity, or ITE-stability metrics.
- Record `retain_all`, `prune`, or `blocked`. `retain_all` is valid when the current list is already compact enough or every tested removal violates tolerances; the pipeline must still write the parsimony artifact.

## Explicit-Feature Causal Forest

After the final extracted confounder and effect-modifier roles are settled, use the explicit feature forest as the required final ITE estimator when the repo-native path is available. Model-based ITEs from R-learners, generic random forests, or other meta-learners may be retained as sensitivity comparisons, but they do not replace the causal-forest fit.

Configuration guidance:
- `architecture.explicit_feature_forest.honest = true`
- confounder-role features become `W`
- effect-modifier-role features become `X`
- `multi_model_agentic_forest` must write `parsimony_review_by_fold.jsonl` and include the parsimony decision in `selected_feature_sets.json` before fitting the final forest.
- use nested or outer folds for all reported metrics

Do not treat causal forest ITEs as ground truth. They are the required final estimator for reported ITEs, and should also be interpreted as a stability and heterogeneity check against the inferred DGP. If the repo-native explicit feature forest is unavailable, use an equivalent honest causal-forest implementation such as `econml.dml.CausalForestDML`; if no real causal forest can be made to run, stop and report the blocker rather than presenting final ITE artifacts as complete.

## Minimal Config Pattern

Start from `example_configs/agentic_explicit_feature_forest_config.json` for endpoint and extraction settings, then change:
- `applied_inference.dataset_path`
- `applied_inference.text_column`
- `applied_inference.treatment_column`
- `applied_inference.outcome_column`
- `applied_inference.outcome_type`
- `applied_inference.architecture.model_type`

For command-line execution, prefer the repo runner if available:

```bash
oci run --config path/to/config.json
```

If the CLI is unavailable, import the pipeline function directly from Python and pass an `AppliedInferenceConfig`.
