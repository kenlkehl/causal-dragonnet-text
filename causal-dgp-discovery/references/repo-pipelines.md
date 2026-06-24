# Repository Pipelines

This repository already contains agentic discovery and causal-forest code. Prefer these paths before writing custom modeling loops.

## Required First Pass: BoW-Guided Discovery

Use `model_type="non_neural_agentic_forest"` for the initial evidence pass.

Relevant implementation:
- `oci/inference/non_neural_agentic_forest.py`
- Config dataclass: `NonNeuralAgenticForestConfig`
- README section beginning "Use `model_type=\"non_neural_agentic_forest\"`"

Recommended defaults:
- `applied_inference.cv_folds >= 5`
- `architecture.explicit_feature_forest.honest = true`
- `architecture.non_neural_agentic_forest.nuisance_folds >= 5`
- `architecture.non_neural_agentic_forest.effect_folds >= 5`
- `ngram_range_min = 1`, `ngram_range_max = 3`
- `candidate_consistency_enabled = true`
- leave `prespecified_confounders`, `prespecified_effect_modifiers`, and `prespecified_features` empty unless the user supplied variables

Key artifacts usually appear under `non_neural_agentic_forest/`:
- `bow_feature_importance_by_fold.jsonl`
- `agent_candidate_proposals.jsonl`
- `selected_feature_sets.json`
- `outer_cv_metrics.csv`

Use BoW outputs to propose concepts, not as final variables. BoW is mandatory for the first lexical recurrence pass, but it is insufficient by itself for final feature extraction in long longitudinal notes.

## LLM-Based Explicit Feature Extraction

Do not implement primary clinical variable extraction with ad hoc regex. After BoW/attention evidence identifies candidate concepts, materialize variables by having an LLM read each underlying `clinical_text` document and return structured values.

Prompt the user before extraction unless they already specified a backend:
- **Coding agent extraction**: Codex reads each document and emits structured values itself. This is acceptable for small datasets or endpoint outages, but it must still be document-by-document LLM extraction, not regex matching.
- **OpenAI-compatible endpoint extraction**: the user provides `vllm_server_url` or another OpenAI-compatible base URL, model name, API key if needed, batch size/concurrency, and max text length. This is preferred for most datasets.

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
- `explicit_features.extraction_max_text_length`
- `explicit_features.cache_enabled=true`

Use `model_type="non_neural_agentic_forest"`, `agentic_explicit_feature_forest`, or `agentic_attention_variable_forest` to keep proposal, extraction, and forest fitting on the repo-native path. The non-neural BoW path sends text evidence to the proposal agent, then the extractor materializes selected explicit variables from text before fitting the final causal forest.

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

Report fold recurrence, direction, effect size, missingness, and overlap warnings. Do not select final confounders or modifiers solely from univariable screens; use them to prioritize extraction, diagnose temporal leakage, and decide which multivariable DGP forms to compare.

## Explicit-Feature Causal Forest

Use the explicit feature forest to compare final extracted DGP variables against model-based ITEs.

Configuration guidance:
- `architecture.explicit_feature_forest.honest = true`
- confounder-role features become `W`
- effect-modifier-role features become `X`
- use nested or outer folds for all reported metrics

Do not treat causal forest ITEs as ground truth. Use them as a stability and heterogeneity check against the inferred DGP.

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
