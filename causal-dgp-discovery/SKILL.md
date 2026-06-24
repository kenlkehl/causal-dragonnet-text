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
- Before running HTR/attention models, inspect the local GPU environment and current GPU load, then choose an explicit parallelization plan for folds, objectives, chunks, and devices. Record the hardware inventory and plan in `report.txt`.
- If shell-level GPU tools work but Python/PyTorch reports no CUDA, treat it as a likely agent sandbox or environment mismatch. Verify the intended interpreter/venv and rerun the CUDA probe, smoke test, and neural HTR job with escalated permissions before falling back to CPU.
- Use fold-aware univariable screens as supporting diagnostics for candidate discovery: feature-treatment association, feature-outcome association for nuisance modeling, and treatment-by-feature/effect-modification association. Do not promote variables from univariable screens alone.
- Use clinical knowledge only to translate recurring high-signal text evidence into extractable baseline concepts.
- Do not default to regex for clinical variable extraction. Use LLM-based, document-by-document extraction that reads the underlying `clinical_text`; regex is allowed only for post-extraction normalization, validation, or narrow fallback after documenting why LLM extraction is unavailable.
- Before materializing candidate clinical variables, prompt the user to choose the extraction backend unless they already specified one: either the coding agent reads each document and extracts structured values itself, or the user provides an OpenAI-compatible endpoint such as a running vLLM server for repo-native extraction.
- Keep all model assessment honest: every nuisance prediction, residual, pseudo-outcome, effect estimate, and ITE used for selection or reporting must be out-of-fold for that row.
- Separate confounder discovery from effect-modifier discovery.
- Do not assume a linear parametric DGP. Fit simple equations only as interpretable summaries after flexible/honest evidence supports them.
- Keep continuous variables continuous unless fold-level diagnostics justify thresholds.
- Maintain a running `report.txt` in the task folder with attempts, results, rejected hypotheses, and final outputs.

## Workflow

1. Inspect the dataset schema, row count, note length, missingness, treatment/outcome rates, and note chronology. Write these facts to `report.txt`.
2. Run honest cross-fitted text-model discovery before structured extraction:
   - Fit BoW/TF-IDF treatment and outcome nuisance models.
   - Fit residual, R-loss, or pseudo-outcome models from out-of-fold nuisance predictions.
   - Collect fold-specific high-signal terms and phrases for treatment, outcome, confounder overlap, residuals, and pseudo-outcomes.
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
   - Ask the user to choose extraction mode if not already specified: coding-agent extraction by reading each note, or repo-native extraction through an OpenAI-compatible/vLLM endpoint.
   - Prefer the repository's `oci.extraction.explicit_features.VLLMFeatureExtractor` or `VLLMExplicitFeatureExtractionProvider` path for endpoint-backed extraction; configure `explicit_features.features` with role-tagged `ExplicitFeatureSpec` entries.
   - For coding-agent extraction, read each `clinical_text` document and emit the same structured feature table and missingness flags; do not use regex as the primary extractor.
   - Record extraction backend, model/endpoint if used, prompt/version, missingness, and extraction rationale.
6. Run fold-aware univariable screens on extracted candidates:
   - Screen each candidate for treatment association to support treatment nuisance/confounder discovery.
   - Screen each candidate for outcome association or improvement in outcome nuisance prediction.
   - Screen each candidate for effect modification using treatment-by-feature interaction, subgroup residual slopes, R-loss, or pseudo-outcome association.
   - Treat screens as prioritization and debugging tools; require cross-fitted multivariable/nuisance evidence before final role assignment.
7. Evaluate roles honestly:
   - Confounder candidates should improve treatment and outcome nuisance performance or residual balance across folds.
   - Effect-modifier candidates should improve treatment-by-feature interaction, R-loss, pseudo-outcome, or fold-stable heterogeneity objectives.
8. Expand candidate extraction only when current candidates leave unexplained treatment assignment, outcome prediction, residual structure, heterogeneous effect signal, or unresolved baseline temporal anchoring.
9. Compare candidate DGP forms, including nonparametric/tree/forest models and interpretable summaries. Avoid finalizing after one plausible pass.
10. Estimate ITEs using the best honest DGP and compare them with an honest causal forest using the same cross-fitting discipline.
11. Stop when additional iterations do not improve holdout nuisance metrics, R-loss/pseudo-outcome metrics, fold recurrence, univariable-screen stability, or ITE stability. Document remaining uncertainty.

## References

- Read [references/workflow.md](references/workflow.md) for the detailed evidence-first procedure.
- Read [references/repo-pipelines.md](references/repo-pipelines.md) when using this repository's BoW, attention, or causal-forest pipelines.
- Read [references/artifacts.md](references/artifacts.md) before writing final outputs.
