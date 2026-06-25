---
name: causal-dgp-discovery
description: Use when Codex needs to reverse engineer a synthetic clinical causal inference dataset from patient-level clinical text, treatment, and outcome columns; discover confounders and effect modifiers from empirical BoW/HTR/attention evidence rather than upfront variable lists; avoid assuming clinical realism or a linear DGP; run honest cross-fitted nuisance, R-loss, pseudo-outcome, and causal-forest analyses; estimate patient-level ITEs; and write a reproducible report.
---

# Causal DGP Discovery

Use this skill to investigate a synthetic patient-level causal inference dataset with `clinical_text`, a binary or continuous treatment column, and an outcome column. The goal is to infer the data-generating process, including confounders, effect modifiers, functional form, and honest patient-level ITEs.

## Core Rules

- Treat the true DGP, metadata, generation config, oracle columns, and parent-folder benchmark files as off-limits unless the user explicitly asks for evaluation against them.
- Start with empirical text-model evidence. Do not begin with a broad hand-built clinical feature list.
- Run both BoW/TF-IDF evidence and HTR/attention/span evidence before finalizing candidate features. Treat BoW as broad lexical discovery and HTR/attention as the span-localization step needed to recover baseline/index-time values inside longitudinal notes.
- For BoW discovery, train a small suite of vectorization strategies instead of relying on one n-gram setup. Compare fold recurrence across unigram-focused, default broad, phrase-focused, and rare-signal-friendly variants when feasible.
- Before running HTR/attention models, inspect the local GPU environment and current GPU load, then choose an explicit parallelization plan for folds, objectives, chunks, and devices. Record the hardware inventory and plan in `report.txt`.
- If shell-level GPU tools work but Python/PyTorch reports no CUDA, treat it as a likely agent sandbox or environment mismatch. Verify the intended interpreter/venv and rerun the CUDA probe, smoke test, and neural HTR job with escalated permissions before falling back to CPU.
- Use fold-aware univariable screens as supporting diagnostics for candidate discovery: feature-treatment association, feature-outcome association for nuisance modeling, and treatment-by-feature/effect-modification association. Emphasize effect magnitude, direction, stability, and missingness at least as much as p values. Do not promote variables from univariable screens alone.
- Use clinical knowledge only to translate recurring high-signal text evidence into extractable baseline concepts.
- Do not use regex or pattern-matching rules as a clinical variable extractor, fallback extractor, or value filler. Default to coding-agent LLM-based, document-by-document extraction that reads the underlying `clinical_text`; the coding agent itself is the LLM extraction backend when no external endpoint is available. Values may remain missing/null only when document-reading extraction cannot recover the value from the patient text, with documented rationale.
- For agentic variable extraction, do the extraction yourself by default. For long documents, use BoW/HTR evidence, targeted chunk sampling, section search, and subagents when useful to inspect relevant passages, then reconcile chunk-level evidence into one structured patient-level value. If no OpenAI-compatible endpoint or repo extraction backend is available, this requirement becomes stronger, not weaker: shard the coding-agent extraction by patient, fold, concept, or document chunk and continue. Ask for or use an external endpoint only when the user supplied one, requested one, or the dataset is too large for reliable coding-agent extraction after a concrete sharding attempt.
- If using local Hugging Face model weights on GPUs for extraction, prefer starting a vLLM OpenAI-compatible server and using the repo's endpoint-backed extractor instead of directly calling `transformers.generate()` in an ad hoc loop. Use a current, instruction-tuned model with enough context and extraction reliability for clinical notes; for example, prefer a model such as Google's `gemma-4-e2b-it`/Gemma 4 E2B instruct when available and compatible with the hardware. Do not use small/base models such as `Qwen/Qwen3-1.7B` or `Qwen/Qwen3-0.6B-Base` for production extraction except for smoke tests or as an explicitly documented last resort after stronger local/server options are unavailable.
- For endpoint-backed extraction, use very long context by default when the model and hardware support it. Aim for roughly a 200,000-token extraction context budget so the extractor can see as much patient history as possible; set vLLM `--max-model-len` near `200000`, set the extractor's max input/text tokens to a similarly large value, and set `max_tokens`/`max_new_tokens` high enough for reasoning-model overhead and complete structured JSON output rather than a few hundred tokens. If a shorter context or output cap is required, document the exact limit and why.
- Do not treat "no LLM extraction backend" as a reason to skip candidate extraction, emit an all-missing `candidate_features` table, or proceed to final causal-forest/ITE claims. In that situation, the coding agent must perform the document-reading extraction itself, preferably with subagents; if the extraction is genuinely infeasible even after sharding, stop and report the blocker instead of finalizing downstream causal results that depend on extracted features.
- Keep all model assessment honest: every nuisance prediction, residual, pseudo-outcome, effect estimate, and ITE used for selection or reporting must be out-of-fold for that row.
- Separate confounder discovery from effect-modifier discovery.
- Seek parsimonious final feature lists. Before passing variables to the causal forest, inspect feature-feature correlations, missingness overlap, treatment/outcome associations, plausible transformations among confounders, and redundant proxy variables; prefer the smallest evidence-supported set that preserves nuisance, heterogeneity, and ITE stability.
- After settling on final confounders and effect modifiers, fit a real honest causal forest as the final ITE estimator. R-learners, S/T/X-learners, generic `RandomForestRegressor`/ExtraTrees/XGBoost effect models, or other meta-learners may be used as diagnostics or sensitivity checks, but they do not satisfy the final causal-forest requirement. If no causal-forest implementation is available, use the repo-native explicit-feature forest path or an installed causal-forest library such as `econml`'s `CausalForestDML`; if neither can be made to run, stop and report the blocker rather than emitting complete final ITE artifacts.
- Evaluate candidate confounders and functional relationships among confounders within internal training folds for association with both treatment and outcome, with effect magnitude and fold stability emphasized at least as much as p value.
- Evaluate candidate effect modifiers for differential association with outcome by treatment, using treatment-by-feature interactions, treatment-stratified outcome associations, R-loss, or pseudo-outcome evidence.
- Do not assume a linear parametric DGP. Fit simple equations only as interpretable summaries after flexible/honest evidence supports them.
- Keep continuous variables continuous unless fold-level diagnostics justify thresholds.
- Maintain a running `report.txt` in the task folder with attempts, results, rejected hypotheses, and final outputs.

## Workflow

1. Inspect the dataset schema, row count, note length, missingness, treatment/outcome rates, and note chronology. Write these facts to `report.txt`.
2. Run honest cross-fitted text-model discovery before structured extraction:
   - Fit a suite of BoW/TF-IDF treatment and outcome nuisance models with distinct vectorization settings. When feasible, include unigram-focused, default `1-3` n-gram, phrase-focused `2-4` n-gram, and lower-`min_df`/higher-`max_features` variants for rare but stable signals.
   - Fit residual, R-loss, or pseudo-outcome models from out-of-fold nuisance predictions.
   - Collect fold-specific high-signal terms and phrases for treatment, outcome, confounder overlap, residuals, and pseudo-outcomes by vectorization run, then compare consensus and disagreement across runs.
3. Run HTR/attention evidence modeling as a standard second pass, not only as a fallback:
   - Inspect available GPUs, memory, CUDA visibility, framework CUDA support, and active GPU processes before launching neural jobs.
   - Record `which python`, `sys.executable`, `sys.prefix`, `VIRTUAL_ENV`, `CUDA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH`, `nvidia-smi`, `torch.__file__`, `torch.__version__`, `torch.cuda.is_available()`, device count, and device names.
   - If the intended Python environment is a user-active venv and sandboxed PyTorch cannot see CUDA while shell `nvidia-smi` can, rerun the same probe with `sandbox_permissions="require_escalated"` and document both results. Run neural HTR under the escalated context if CUDA becomes available.
   - Decide whether to parallelize by CV fold, effect objective, signal type, chunk shard, or dataset shard; avoid oversubscribing GPU memory.
   - Localize fold-specific high-signal spans for treatment, outcome nuisance, residuals, R-loss, and pseudo-outcome objectives.
   - Use HTR spans to distinguish baseline/index-time facts from post-treatment outcomes, repeated history, and report-template artifacts.
   - Prefer concepts whose aliases recur in both BoW and HTR evidence, while allowing HTR-only numeric or temporal slots when BoW cannot represent them.
4. Translate recurring text/span signals into a small first candidate set of extractable baseline concepts.
5. Extract only those evidence-supported concepts with LLM-based document reading:
   - Default to coding-agent extraction by reading the patient documents yourself. The absence of a repo-native extractor, vLLM server, OpenAI-compatible endpoint, or local model is not a valid reason to skip extraction; it means the coding agent must do the extraction directly from `clinical_text`.
   - For non-trivial datasets, plan a sharded extraction pass before starting: split by patient ranges, folds, concepts, or HTR-highlighted chunks; spawn subagents when available; and reconcile subagent outputs into one patient-level feature table. If subagents are unavailable, perform the same sharding manually across repeated passes.
   - Use a repo-native OpenAI-compatible/vLLM endpoint only when the user supplied one, requested it, or the dataset is too large for coding-agent extraction after sharded coding-agent extraction has been attempted or clearly bounded.
   - Prefer the repository's `oci.extraction.explicit_features.VLLMFeatureExtractor` or `VLLMExplicitFeatureExtractionProvider` path for endpoint-backed extraction; configure `explicit_features.features` with role-tagged `ExplicitFeatureSpec` entries.
   - When local HF weights and GPUs are available, first try to serve a strong instruction model through vLLM/OpenAI-compatible server mode and route extraction through the repo extractor. Document the exact server command, model path/name, tensor parallelism, dtype, max model length, GPU assignment, batch/concurrency limits, and a JSON smoke-test result before full extraction. Use direct Hugging Face `transformers` generation only for smoke tests or if vLLM/server mode is unavailable or incompatible, and then use a capable instruction model rather than a small/base model.
   - Configure vLLM/server extraction for long patient histories: target `--max-model-len 200000` or the largest supported value near 200k tokens, set extraction max text/input length near the same target, and set generation `max_tokens`/`max_new_tokens` generously for reasoning tokens and JSON completion. Keep `input_tokens + max_new_tokens <= max_model_len`; if the exact 200k target is impossible, choose the largest stable context after a smoke test and document the fallback.
   - For coding-agent extraction, use subagents or targeted chunk review when useful: search or sample likely relevant chunks, inspect nearby context, reconcile patient-level values, and emit the same structured feature table and missingness flags; do not use regex or pattern matching to extract or fill clinical values.
   - Do not create an all-missing feature table merely because no endpoint exists. Missingness is a patient/concept-level conclusion from reading the text, not a run-level substitute for extraction.
   - Record extraction backend, model/endpoint if used, prompt/version, missingness, and extraction rationale.
6. Run fold-aware univariable screens on extracted candidates:
   - Screen each candidate for treatment association to support treatment nuisance/confounder discovery.
   - Screen each candidate for outcome association or improvement in outcome nuisance prediction.
   - Screen each candidate for effect modification using treatment-by-feature interaction, subgroup residual slopes, R-loss, or pseudo-outcome association.
   - Screen plausible transformations or functional relationships among confounders when supported by text evidence or diagnostics.
   - Record standardized effect sizes, score deltas, fold recurrence, direction, missingness, and p values; do not rank candidates by p value alone.
   - Treat screens as prioritization and debugging tools; require cross-fitted multivariable/nuisance evidence before final role assignment.
7. Evaluate roles honestly:
   - Confounder candidates should improve treatment and outcome nuisance performance or residual balance across folds.
   - Effect-modifier candidates should improve treatment-by-feature interaction, treatment-stratified outcome association, R-loss, pseudo-outcome, or fold-stable heterogeneity objectives.
8. Run a parsimony and redundancy review before causal-forest fitting:
   - Explore correlations, missingness overlap, and surrogate relationships among extracted features.
   - Prefer direct, stable, baseline variables over redundant proxies or broad composite variables unless the composite clearly improves honest metrics.
   - Revisit candidate transformations and interactions among confounders only when they improve held-out nuisance or balance diagnostics.
9. Expand candidate extraction only when current candidates leave unexplained treatment assignment, outcome prediction, residual structure, heterogeneous effect signal, or unresolved baseline temporal anchoring.
10. Repeatedly mull over the candidate feature list before finalizing: revise roles, merge redundant variables, reject weak proxies, and compare the revised list against prior iterations.
11. Compare candidate DGP forms, including nonparametric/tree/forest models and interpretable summaries. Avoid finalizing after one plausible pass.
12. Estimate final ITEs with an honest causal forest fit on the finalized confounders and effect modifiers, using the same cross-fitting discipline. Other honest DGP, R-learner, or meta-learner estimates may be reported only as sensitivity analyses or comparisons, not as a replacement for the final causal-forest ITEs.
13. Stop when additional iterations do not improve holdout nuisance metrics, R-loss/pseudo-outcome metrics, fold recurrence, univariable-screen stability, parsimony, or ITE stability. Document remaining uncertainty.

## References

- Read [references/workflow.md](references/workflow.md) for the detailed evidence-first procedure.
- Read [references/repo-pipelines.md](references/repo-pipelines.md) when using this repository's BoW, attention, or causal-forest pipelines.
- Read [references/artifacts.md](references/artifacts.md) before writing final outputs.
