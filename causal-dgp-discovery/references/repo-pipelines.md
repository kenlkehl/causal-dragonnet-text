# Repository Pipelines

This repository already contains agentic discovery and causal-forest code. Prefer these paths before writing custom modeling loops.

## Required First Pass: BoW-Guided Discovery

Use `model_type="multi_model_agentic_forest"` for the initial evidence pass. Configure multiple `bow_views` in one run rather than relying on one vectorizer setting.

Relevant implementation:
- `oci/inference/multi_model_agentic_forest.py`
- Config dataclass: `MultiModelAgenticForestConfig`
- README section beginning "Use `model_type=\"multi_model_agentic_forest\"`"

Recommended defaults for each run:
- `applied_inference.cv_folds >= 5`
- `architecture.explicit_feature_forest.honest = true`
- `architecture.multi_model_agentic_forest.nuisance_folds >= 5`
- `architecture.multi_model_agentic_forest.effect_folds >= 5`
- `candidate_consistency_enabled = true`
- leave `prespecified_confounders`, `prespecified_effect_modifiers`, and `prespecified_features` empty unless the user supplied variables

Minimum `bow_views` suite when feasible:
- unigram-focused: `ngram_range_min = 1`, `ngram_range_max = 1`
- default broad: `ngram_range_min = 1`, `ngram_range_max = 3`
- phrase-focused: `ngram_range_min = 2`, `ngram_range_max = 4`
- rare-signal-friendly: keep broad or phrase n-grams but lower `min_df`, raise `max_features`, or compare `sublinear_tf` settings

Optional learner sensitivity checks can vary `bow_model` among supported values such as `linear`, `extratrees`, `random_forest`, or `xgboost` when runtime allows. Keep fold construction and outcome/treatment definitions fixed across views so recurrence and disagreement are interpretable. Record a view name and vectorizer params for every BoW artifact.

Key artifacts usually appear under `multi_model_agentic_forest/`:
- `bow_view_oof_predictions.parquet`
- `bow_view_feature_importance_by_fold.jsonl`
- `embedding_contrast_evidence_by_fold.jsonl`
- `agent_candidate_proposals.jsonl`
- `selected_feature_sets.json`
- `outer_cv_metrics.csv`

Use BoW and embedding outputs to propose concepts, not as final variables. BoW is mandatory for the first lexical recurrence pass, but it is insufficient by itself for final feature extraction in long longitudinal notes. Prefer concepts that recur across folds and across views, while preserving view-specific discoveries when the evidence is strong.

## LLM-Based Explicit Feature Extraction

Do not implement clinical variable extraction with ad hoc regex or pattern-matching fallback logic. After BoW/attention evidence identifies candidate concepts, materialize variables by having an LLM read each underlying `clinical_text` document and return structured values. When no external/model endpoint is available, the coding agent itself is the LLM document-reading extractor.

Default to coding-agent extraction unless the user supplied an endpoint, requested endpoint extraction, or the dataset is too large for reliable coding-agent extraction after a concrete sharding attempt:
- **Coding agent extraction**: Codex reads each document, or targeted chunks of each long document, and emits structured values itself. This is required when no repo-native extractor, vLLM server, OpenAI-compatible endpoint, local model, or API key is available. Use BoW/HTR spans, section search, targeted chunk sampling, and subagents when useful, but reconcile all evidence into one patient-level row per concept. For large datasets, shard by patient ranges, folds, concepts, or document chunks and spawn subagents when available; if subagents are unavailable, perform the same sharded passes manually. This must still be document-reading LLM extraction, not regex or pattern matching.
- **OpenAI-compatible endpoint extraction**: use `vllm_server_url` or another OpenAI-compatible base URL, model name, API key if needed, batch size/concurrency, and max text length when provided, requested, or needed after coding-agent extraction has been attempted or clearly bounded for scale.
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

Use `model_type="multi_model_agentic_forest"`, `agentic_explicit_feature_forest`, or `agentic_attention_variable_forest` to keep proposal, extraction, and forest fitting on the repo-native path. The multi-model BoW path sends text evidence to the proposal agent, then the extractor materializes selected explicit variables from text before fitting the final causal forest.

## Required Second Pass: Attention/HTR Evidence Path

Use `model_type="agentic_attention_variable_forest"` after the BoW pass. This is a standard part of the discovery workflow, not only a fallback. It is especially important when:
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
- Inspect feature-feature correlations, categorical contingency tables, and missingness overlap to merge or reject redundant proxy variables.
- Iterate on a parsimonious feature list until removing weak or redundant variables no longer harms honest nuisance, heterogeneity, or ITE-stability metrics.

## Explicit-Feature Causal Forest

After the final extracted confounder and effect-modifier roles are settled, use the explicit feature forest as the required final ITE estimator when the repo-native path is available. Model-based ITEs from R-learners, generic random forests, or other meta-learners may be retained as sensitivity comparisons, but they do not replace the causal-forest fit.

Configuration guidance:
- `architecture.explicit_feature_forest.honest = true`
- confounder-role features become `W`
- effect-modifier-role features become `X`
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
