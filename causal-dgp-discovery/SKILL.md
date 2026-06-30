---
name: causal-dgp-discovery
description: Use when a coding agent needs to reverse engineer a synthetic clinical causal inference dataset from patient-level clinical text, treatment, and outcome columns; discover confounders and effect modifiers from empirical BoW, embedding-contrast, HTR, attention, attribution, and span evidence rather than upfront variable lists; extract and review candidate variables; run honest nuisance, R-loss, pseudo-outcome, parsimony, and causal-forest analyses; estimate patient-level ITEs; and write a reproducible report.
---

# Causal DGP Discovery

Use this skill to investigate a synthetic patient-level causal inference dataset with clinical text, treatment, and outcome columns. The goal is to infer the data-generating process: confounders, effect modifiers, functional form, uncertainty, and honest patient-level ITEs.

This skill is harness-agnostic. The invoking agent owns the evidence review, candidate translation, extraction decisions, role assignment, revision loop, and final synthesis. Repository modules with names such as `agentic` are implementation labels, not permission to delegate causal reasoning or final judgment to autonomous proposal/review prompts.

## Core Rules

- Treat the true DGP, metadata, generation config, oracle columns, and parent-folder benchmark files as off-limits unless the user explicitly asks for evaluation against them.
- Start with empirical text-model evidence. Do not begin with a broad hand-built clinical feature inventory.
- Run BoW/TF-IDF evidence, embedding-contrast retrieval, and HTR/attention evidence before finalizing candidate features.
- Use BoW as broad lexical discovery, embedding contrast as real-text chunk/concept retrieval, and HTR attention/attribution as the span-localization step for baseline or index-time values inside longitudinal notes.
- For BoW discovery, compare multiple vectorization views when feasible: unigram-focused, default broad, phrase-focused, and rare-signal-friendly variants.
- Keep discovery context and extraction context distinct: BoW terms, embedding-contrast chunks, and HTR spans are required evidence for considering candidate features, aliases, temporal anchors, and audit targets; they are not sufficient context for accepted downstream feature values.
- Use clinical knowledge only to translate recurring high-signal text evidence into extractable baseline concepts.
- Separate confounder discovery from effect-modifier discovery.
- Keep all model assessment honest: every nuisance prediction, residual, pseudo-outcome, effect estimate, and ITE used for selection or reporting must be out-of-fold for that row.
- Maintain `workflow_gate_status.json` from the start of the run. Every required gate must be `pass` before final ITE fitting; a failed gate must trigger a targeted retry and `gate_retry_log.jsonl` entry before it can be marked `blocked_after_retries`.
- Gate failures are repair signals, not stopping points. Diagnose the failed condition, retry the narrowest responsible stage, and re-run the gate. Declare a run blocked only after documented retries at a small enough scope that further progress is infeasible without user input or external access.
- Use fold-aware univariable screens as supporting diagnostics, not final selection rules. Emphasize effect magnitude, direction, stability, overlap, and missingness at least as much as p values.
- Do not assume a linear parametric DGP. Fit simple equations only as interpretable summaries after flexible/honest evidence supports them.
- Keep continuous variables continuous unless fold-level diagnostics justify thresholds.
- Do not use regex, short-window parsers, nearby-number rules, category heuristics, or other shortcut logic as a clinical variable extractor, fallback extractor, or value filler. These brittle rules are especially unsafe for numeric/categorical concepts embedded in dense clinical text, such as biomarker scores, lab values, stages, grades, and treatment regimens.
- Extraction must be document reading. Agent-based extraction must read each patient's complete `clinical_text` wholesale for the requested concepts, or use a recursive reading strategy whose passes cover the complete note before reconciling to one patient-level value. Evidence-highlighted chunks, BoW terms, HTR spans, or retrieved snippets may guide attention, candidate selection, temporal anchoring, and auditing, but they must not be the only context used to extract a downstream modeling value.
- Extraction must return structured patient-level values, missingness flags, temporal labels, and brief evidence that is consistent with the full patient note. Endpoint-backed models are extraction backends, not discovery agents.
- Before starting candidate extraction, ask the user how they want extraction performed. Present exactly these two routes: (1) the invoking agent performs full-document-reading extraction itself and acts as the LLM, using patient, fold, concept, or concept-family shards as needed while preserving complete-note coverage; or (2) the user provides a URL for a running vLLM server or other OpenAI-compatible endpoint to use as the document-reading extraction backend. Do not launch or use a local endpoint, vLLM process, or OpenAI-compatible server unless the user selects route (2) or explicitly authorizes that backend.
- If no endpoint-backed extractor is available, perform full-document-reading extraction directly with the available agent/harness. For larger datasets, shard by patient, fold, concept, or concept family, but every patient/concept extraction must be based on the complete note or a recursive complete-note pass. Do not shard extraction down to isolated snippets or short windows unless those snippets are only audit aids and the final value is verified against the full note.
- Audit extraction shards before accepting them. Check that each structured value is consistent with its evidence summary and source text, especially numeric values, categories, temporal anchors, and missingness flags. Audits must include stratified checks across categorical levels and suspicious boundary values, not only random rows.
- A failed or inconsistent shard is not a run-level blocker by itself. Quarantine the bad shard, diagnose the inconsistency, narrow the shard or concept scope, clarify the extraction prompt/spec, and retry. Escalate from large shards to small shards to single-patient or single-concept retries before declaring extraction blocked.
- Do not create an all-missing candidate table merely because an extraction endpoint is unavailable. Stop only if extraction remains genuinely infeasible after concrete retry attempts, including smaller shards and inconsistency-focused re-reads. Report the blocker instead of running downstream causal-forest claims.
- Build `ensemble_nuisance_predictions.parquet` after BoW, HTR, and extracted-feature nuisance predictions are available. Final pseudo-outcomes, R-loss diagnostics, candidate R-signal review, and role decisions must use this full out-of-fold nuisance ensemble. BoW-only R signals are allowed for early discovery, not final review.
- Before final causal-forest fitting, compare extracted-feature nuisance and effect diagnostics with upstream BoW, embedding-contrast, and HTR evidence. Revise specs, aliases, values, roles, or extraction when extracted features materially underperform.
- Run `candidate_signal_review.jsonl` before parsimony. Every candidate must have fold-honest treatment nuisance, outcome nuisance, R-pseudo-outcome/R-loss, interaction or treatment-stratified diagnostics, missingness/overlap, and a role decision. Missing candidate-level R diagnostics fail the gate and require retry.
- Run a mandatory parsimony review before passing variables to the causal forest. Record `retain_all`, `prune`, or `blocked`; `retain_all` is valid when the set is already compact or removals harm honest diagnostics.
- After final confounders and effect modifiers are settled, fit a real honest causal forest as the final ITE estimator. R-learners, S/T/X-learners, generic random forests, ExtraTrees, XGBoost, or other meta-learners may be diagnostics, but they do not satisfy the final causal-forest requirement.
- Maintain a running `report.txt` in the task folder with attempts, results, rejected hypotheses, revision decisions, and final outputs.

## Repository Components

Use these components directly or through any runner that exposes them without giving the runner responsibility for discovery decisions:

- **BoW modeling:** `BoWViewConfig`, `default_multi_model_bow_views()`, and sparse cross-fitted BoW helpers in `oci/inference/multi_model_agentic_forest.py`. Inspect fold-specific nuisance, residual, pseudo-target, R-loss, and feature-importance outputs.
- **Embedding contrasts:** `EmbeddingContrastEvidenceGenerator` in `oci/inference/embedding_contrast_discovery.py`, with chunking and cache helpers in `oci/models/concept_embedding_utils.py` and `oci/models/concept_embedding_cache.py`.
- **HTR modeling:** `AgenticAttentionVariableForestRunner` in `oci/inference/agentic_attention_variable_forest.py`, and `MultiModelHTREvidenceProvider` when a sparse-text workflow needs reusable HTR nuisance/effect predictions.
- **Feature attribution:** BoW feature importances, embedding-contrast retrieved chunks/concept probes, and HTR attention/token-span attribution outputs such as nuisance/effect attention evidence.
- **Extraction:** `ExplicitFeatureSpec`, `ExplicitFeatureExtractionConfig`, `VLLMFeatureExtractor`, `extract_explicit_features()`, `VLLMExplicitFeatureExtractionProvider`, and `oci/extraction/llm_routing.py`.
- **Final ITEs:** `CausalForestHead` and explicit-feature forest evaluators built on honest `CausalForestDML`.

## Workflow

1. Inspect the dataset schema, row count, note length, missingness, treatment/outcome rates, and note chronology. Write these facts to `report.txt`.
2. Build honest folds and reuse them across text evidence, extraction review, role evaluation, parsimony, and final reporting.
3. Run cross-fitted BoW/TF-IDF views for treatment nuisance, outcome nuisance, residual, R-loss, and pseudo-outcome evidence. Compare fold recurrence across vectorizer and learner views.
4. Run embedding-contrast retrieval on training-fold text chunks for treatment, outcome, per-view R-pseudo-target, ensemble R, within-arm outcome, treatment-outcome cell, orthogonal R-score, and concept-probe contrasts when supported.
5. Run HTR nuisance/effect modeling and inspect attention/span attributions. Use them to localize baseline/index-time facts, separate post-treatment or copied-history text, and identify numeric or temporal slots that BoW cannot represent well.
6. Translate recurring text, chunk, and span evidence into a small first candidate set of extractable baseline concepts shaped like `ExplicitFeatureSpec`: name, type, categories or value aliases, exact description, roles, temporal window, and missingness rule.
7. Before extracting evidence-supported concepts, ask the user to choose the extraction route: agent-as-LLM direct document reading, or a user-provided running vLLM/OpenAI-compatible endpoint URL. Then extract only evidence-supported concepts by the selected document-reading route. For agent-as-LLM extraction, read complete patient notes wholesale or use a recursive complete-note strategy; do not extract from short windows, regex matches, or rule-generated snippets. Audit shard outputs as they arrive; if a shard is internally inconsistent, retry with narrower patient/concept shards and an inconsistency-focused prompt before accepting or blocking extraction.
8. Review extracted features honestly. Train extracted-feature treatment and outcome nuisance models inside folds, then build or update the full `ensemble_nuisance_predictions.parquet` using all available BoW, HTR, and extracted-feature out-of-fold nuisance predictions.
9. Run `candidate_signal_review.jsonl` using the full ensemble R signal. For every candidate, record treatment nuisance signal, outcome nuisance signal, R-pseudo-outcome/R-loss signal, interaction or treatment-stratified signal, missingness/overlap, role decision, and retry decision. If any candidate lacks this review, retry the review stage before parsimony.
10. If extracted features underperform or candidate-level gates fail, revise candidate specs, merge aliases, fix categories/value harmonization, re-role variables, add narrow evidence-supported concepts, and re-extract or re-review only what changed.
11. Run fold-aware univariable screens for treatment association, outcome association, treatment-by-feature interaction, treatment-stratified outcome association, R-loss, or pseudo-outcome signal. Use them for debugging and prioritization only.
12. Run the mandatory parsimony/redundancy review after candidate signal review: correlations, categorical contingency tables, missingness overlap, and honest removal/group-ablation diagnostics where useful.
13. Run the final preflight gate. Verify that the full nuisance ensemble exists, candidate-level R-signal review exists for every candidate, extraction audit passed or was retried, parsimony passed, and the final estimator is a real causal forest. Retry failed stages before proceeding.
14. Compare candidate DGP forms with flexible/tree/forest models and simple summaries. Avoid finalizing after one plausible pass.
15. Estimate final ITEs with an honest causal forest fit on finalized confounders and effect modifiers only after all gates pass. Label any non-causal-forest estimates as sensitivity diagnostics.
16. Stop when additional iterations no longer improve honest nuisance metrics, R-loss/pseudo-outcome metrics, extracted-feature benchmark gaps, fold recurrence, parsimony, or ITE stability. Document remaining uncertainty.

## References

- Read [references/workflow.md](references/workflow.md) for the evidence-first procedure.
- Read [references/gates.md](references/gates.md) for mandatory gates, retry behavior, and preflight rules.
- Read [references/repo-pipelines.md](references/repo-pipelines.md) for the reusable repository component map.
- Read [references/artifacts.md](references/artifacts.md) before writing final outputs.
