---
name: causal-dgp-discovery
description: Use when a coding agent needs to infer confounders and effect modifiers from patient-level clinical text, treatment, and outcome columns; train honest BoW, HTR, embedding-contrast, explicit-feature, and causal-forest models; extract full-document feature values without shortcuts; and produce reproducible patient-level ITEs.
---

# Causal DGP Discovery

Use this skill for clinical-text causal discovery tasks where each row has patient text, treatment, and outcome. The goal is to derive explicit confounders and effect modifiers from empirical text evidence, extract those variables from the complete notes, refine them until they are stable and useful, then fit an honest causal forest.

The coding agent owns the causal reasoning. Repository modules named `agentic` are tools, not authorities. Do not delegate final feature definitions, role decisions, or causal claims to an autonomous proposal prompt.


## Core Workflow

1. **Inspect the dataset.** Identify patient id, clinical text, treatment, outcome, row count, missingness, treatment/outcome rates, treatment-outcome table, note lengths, and chronology assumptions.
2. **Create folds.** Define external/outer folds and internal/inner folds. Record the split provenance.
3. **Run nuisance text evidence.** Train cross-fitted BoW/TF-IDF and HTR nuisance models for treatment and outcome. Run embedding contrasts for treatment, outcome, and nuisance residual evidence.
4. **Interpret confounding evidence.** Review high-signal BoW terms, embedding chunks, and HTR spans. Translate recurring evidence into candidate confounder definitions.
5. **Build nuisance ensembles.** For each patient, compute a mean out-of-fold treatment prediction and a mean out-of-fold outcome prediction from available BoW, HTR, and later extracted-feature nuisance models. Record source predictions and the averaging rule.
6. **Compute R signals.** Use the mean ensemble predictions to compute treatment residuals, outcome residuals, R-loss terms, and pseudo-outcomes for each patient. These signals drive effect-modifier discovery and candidate review.
7. **Run effect-modification evidence.** Use BoW R/pseudo-outcome/interaction views, embedding contrasts for R/orthogonal/within-arm/treatment-outcome signals, and a separate HTR effect/R-stage attribution pass.
8. **Interpret effect evidence.** Translate highly attended, relevant, or contrasting text into candidate effect-modifier definitions. Keep dual-role variables when evidence supports both confounding and modification.
9. **Deduplicate and harmonize candidates.** Merge aliases and near-duplicates, define type, categories, units, temporal anchor, missingness rule, value aliases, and roles. Keep continuous variables continuous unless fold-honest diagnostics justify a category.
10. **Extract feature values.** Extract only the harmonized evidence-supported concepts. Each patient/concept value must come from complete-document reading. 
11. **Audit and harmonize values.** Check sampled rows, suspicious values, boundary values, missingness, and category levels against the source text. Quarantine bad shards, narrow the scope, re-read the full note, and retry before downstream modeling.
12. **Review feature signal.** Train fold-honest extracted-feature treatment and outcome nuisance models. Update the nuisance ensemble and R signals. For each candidate, record treatment signal, outcome signal, R-loss/pseudo-outcome or interaction signal, missingness/overlap, upstream text evidence, and role decision.
13. **Check parsimony.** Inspect inter-feature correlations, categorical contingency, missingness overlap, semantic duplicates, and ablation impact. Retain all only when the set is compact or removals hurt honest diagnostics.
14. **Iterate.** If nuisance prediction, R-loss, pseudo-outcome, extracted-feature benchmarks, parsimony, overlap, or ITE stability are weak, return to the relevant text evidence. Revise only evidence-supported concepts, re-extract changed concepts, re-harmonize values, and rerun review.
15. **Fit the causal forest.** Once features and ontologies are stable, fit an honest causal forest using confounder-role features as controls and effect-modifier-role features as heterogeneity features. Save patient-level counterfactual predictions and ITEs with fold/model provenance.

Stop when additional evidence-supported revisions no longer improve honest diagnostics, parsimony, or ITE stability. Document remaining uncertainty instead of forcing a precise DGP.

## Interpreting Text Evidence

- **BoW/TF-IDF:** broad lexical discovery for treatment prediction, outcome prediction, residual structure, R targets, and feature recurrence. Compare multiple vectorization views when feasible.
- **Embedding contrasts:** retrieve real chunks for treatment, outcome, residual, R, within-arm, treatment-outcome cell, orthogonal, and concept-probe contrasts. Chunks are evidence for concepts, not extracted values.
- **HTR/neural attention:** localize spans for nuisance/confounding and separately for effect/R-stage heterogeneity. HTR must be a real neural attention, hidden-state, or attribution workflow. Sparse BoW chunk scoring, dense TF-IDF/SVD retrieval, embedding retrieval, or generic chunk localization is not HTR.
- Use clinical knowledge only to translate recurring evidence into extractable baseline concepts. Do not invent variables unsupported by the text evidence.
- When evidence points to broad families such as labs, monitoring, molecular markers, status labels, regimen history, eligibility, counts, ratios, or derived quantities, inspect the actual chunks/spans and define the specific extractable value.

## Non-Negotiables

- Do not inspect true DGP files, oracle columns, generation configs, benchmark parent folders, or other answer keys unless the user explicitly asks.
- Treat supplied `clinical_text` as baseline/pre-treatment/pre-outcome unless the user says otherwise. Treatment planning, prior regimen history, prognosis, eligibility, and severity can be valid baseline signals. The treatment and outcome label columns themselves are never features.
- Start from empirical text-model evidence. Do not begin with a broad hand-built clinical inventory.
- Use three text evidence families before finalizing candidates: BoW/TF-IDF, embedding contrasts, and HTR/neural attention or attribution.
- Keep confounder discovery separate from effect-modifier discovery, then harmonize both candidate lists before extraction.
- Every prediction, residual, pseudo-outcome, R-loss value, diagnostic, and ITE used for selection or reporting must be out-of-fold for that patient.
- Clinical variable extraction must read the complete patient document, or a recursive pass that covers the complete document. Do not extract final feature values from regex, short windows, nearby-number rules, isolated snippets, category heuristics, or BoW/HTR/embedding highlights alone.
- If extracted features underperform the text evidence or R-signal, revise feature definitions, aliases, categories, roles, or extraction. Do not just tune the final forest around weak variables.
- The final ITE estimator must be a real honest causal forest, such as `CausalForestDML` through `CausalForestHead` or the explicit-feature forest evaluator. Other learners are diagnostics only.

## Honest Fold Strategy

Use nested honesty whenever final performance or patient-level ITEs are reported.

- **External/outer folds:** hold out rows for final evaluation and reported ITE predictions. Do not use outer-held-out outcomes, treatment labels, or text evidence to choose features, ontologies, extraction fixes, parsimony decisions, tuning, or stopping rules for that fold.
- **Internal/inner folds:** run inside each outer-training split. Use them for text evidence, nuisance fitting, candidate decisions, extraction review, value harmonization, parsimony, and model tuning.
- **Out-of-fold nuisance predictions:** for each row being scored inside the current training split, `e_hat = E[T | text/features]` and `m_hat = E[Y | text/features]` must come from models that did not train on that row.
- **Final row estimates:** report one prediction per patient from a model that held the row out, or clearly label any post-selection full-data refit as a refit and do not use it as honest performance evidence.
- Reuse the same fold assignments across BoW, HTR, embedding evidence, extraction review, nuisance ensembles, parsimony, and final reporting whenever possible.

## Extraction Standard

Extraction is document reading, not pattern matching.

- The final value for each patient/concept must be based on the complete note or a recursive pass covering the complete note.
- Evidence highlights can guide attention, temporal anchoring, aliases, and audits, but they cannot be the only context for accepted values.
- Return structured values, missingness flags, temporal labels, and brief evidence summaries.
- Null is valid only when complete-document reading cannot recover the value.
- Endpoint absence is not a reason to create an all-missing table. Use direct agent reading, smaller shards, or report a blocker after concrete retries.
- Post-processing may canonicalize values already extracted by document reading; it must not create clinical values from regex or nearby-number rules.

## Minimal Artifacts

Write artifacts in the task folder unless the user asks otherwise.

- `report.txt`: running summary of dataset facts, folds, evidence, feature decisions, retries, final variables, model results, and uncertainty.
- `text_evidence.*`: fold-specific BoW, embedding, and HTR evidence with source/objective/provenance.
- `candidate_features.*`: extracted patient-level values, missingness, feature specs, backend, and evidence summaries.
- `ensemble_nuisance_predictions.*`: source-specific and mean out-of-fold `e_hat` and `m_hat`, residuals, pseudo-outcomes, and R-loss inputs.
- `candidate_signal_review.*`: per-candidate confounding/modification diagnostics and role decisions.
- `parsimony_review.*`: redundancy, correlation, missingness, and ablation decisions.
- `ite_estimates.*`: final causal-forest counterfactual predictions and ITEs with fold/model provenance.

## Useful Repository Components

- BoW and multi-model text evidence: `oci/inference/multi_model_agentic_forest.py`, `BoWViewConfig`, `default_multi_model_bow_views()`.
- Embedding contrasts: `oci/inference/embedding_contrast_discovery.py`, `oci/models/concept_embedding_utils.py`, `oci/models/concept_embedding_cache.py`.
- HTR evidence: `oci/inference/agentic_attention_variable_forest.py`, `MultiModelHTREvidenceProvider`.
- Feature specs and extraction: `ExplicitFeatureSpec`, `ExplicitFeatureExtractionConfig`, `VLLMFeatureExtractor`, `extract_explicit_features()`, `VLLMExplicitFeatureExtractionProvider`, `oci/extraction/llm_routing.py`.
- Final causal forest: `CausalForestHead`, `CausalForestDML`, and explicit-feature forest evaluators.
