# Evidence-First Causal DGP Workflow

## 1. Initial Exploration

- Confirm available files in the task folder and do not inspect parent folders unless the user permits it.
- Load the dataset and identify patient id, text, treatment, outcome, and any existing non-oracle covariates.
- Summarize row count, missing text, treatment/outcome rates, treatment-outcome cross-tab, note length distribution, repeated sections, and likely chronology.
- Start `report.txt` immediately and keep appending evidence, decisions, rejected hypotheses, and blockers.
- Start `workflow_gate_status.json` immediately. Initialize required gates as `pending`, update them to `pass`, `retrying`, or `blocked_after_retries`, and write every failed-gate retry to `gate_retry_log.jsonl`.

## 2. Text Evidence Before Feature Extraction

Start with honest cross-fitted text models. The purpose is to learn which concepts the text suggests, not to let clinical priors define the variable list. Run BoW/TF-IDF, embedding-contrast retrieval, and HTR/attention evidence before finalizing candidate features.

Run a BoW/TF-IDF suite:
- Use the same honest folds across vectorization variants so feature recurrence can be compared directly.
- Include unigram-focused, default broad `1-3` n-gram, phrase-focused `2-4` n-gram, and rare-signal-friendly variants.
- Fit treatment nuisance models on training folds and predict held-out rows.
- Fit outcome nuisance models on training folds and predict held-out rows.
- Compute held-out residuals, R-learner targets, and pseudo-outcomes.
- Save high-signal n-grams by fold and vectorization run for treatment, outcome, overlap, residual, and heterogeneity evidence.
- Compare consensus, disagreement, and unique discoveries across vectorization runs before translating text evidence into candidate concepts.

Run embedding-contrast retrieval as a required companion to BoW:
- Build training-fold text chunks only, preserving patient/fold provenance.
- Retrieve treatment, outcome, per-view R-pseudo-target, ensemble R, within-arm outcome, treatment-outcome cell, orthogonal R-score, and concept-probe contrasts.
- Treat retrieved chunks as real-text evidence for candidate concepts, not as direct vector interpretations.
- Keep embedding evidence active alongside BoW and HTR unless it is explicitly disabled with a documented reason.

Run an HTR/attention evidence pass:
- Before neural jobs, inspect accelerator availability, the intended Python environment, framework CUDA support, active GPU processes, memory headroom, and worker/device mapping.
- If GPU access is unavailable or inconsistent, follow the active harness's approval/escalation mechanism when needed, then document both the failed and resolved probes. Fall back to CPU only after the limitation is confirmed or the user declines the required access.
- Train cross-fitted nuisance and effect/heterogeneity models with attention or span evidence.
- Add HTR nuisance predictions to any row-level nuisance ensemble used for R-loss or pseudo-target construction when that ensemble exists. This is a gate: if HTR predictions are available but not included in the final ensemble R signal, mark `nuisance_ensemble_gate` as failed and retry ensemble construction.
- Inspect top chunks, token spans, and attribution targets for treatment, outcome, residual, R-loss, and pseudo-outcome objectives.
- Use HTR spans to determine whether a signal is baseline/pre-treatment, index-time, historical, post-treatment, or a report-template artifact.
- Pay special attention to numeric slots, lab values, temporal qualifiers, categorical status values, and derived quantities that BoW cannot represent well.

Do not extract a broad clinical inventory before this step. Extract only concepts supported by recurring text, chunk, or span evidence.

Discovery evidence is allowed and required to be local. BoW terms, embedding-contrast chunks, and HTR spans should guide which features to consider, what aliases/categories to define, where temporal anchors may appear, and which rows need auditing. That does not relax the extraction requirement: accepted patient-level feature values for downstream modeling must still come from full-document reading or a recursive pass covering the full document.

## 3. Candidate Concept Translation

Translate high-signal phrases, chunks, and spans into baseline patient-level concepts:
- Map aliases and near-duplicates into one extraction target.
- Preserve temporal meaning. Baseline/pre-treatment values are valid; post-treatment response/progression text is not a baseline covariate unless the user asks for post-treatment prediction.
- Keep continuous concepts continuous by default.
- For categorical concepts, define categories and unknown/missing handling before extraction.
- For longitudinal notes, define index time and baseline window before extracting values. Prefer the value nearest to but not after treatment/regimen initiation unless the task says otherwise.
- Where evidence points to a lab or measurement family, consider clinically and textually supported derived quantities.

Each concept should be representable as an `ExplicitFeatureSpec`-shaped contract:
- `name`
- `type`: `continuous` or `categorical`
- `categories` and optional `value_aliases` for categorical variables
- `description`: exact baseline/pre-treatment extraction target and missingness rule
- `roles`: `confounder`, `effect_modifier`, or both

## 4. Candidate Extraction

Do not use regex, pattern matching, short-window parsers, nearby-number rules, category heuristics, or other shortcut logic to extract clinical variables. Extraction must read patient documents and return structured values, missingness flags, and brief evidence/rationale for each requested concept. This matters most for numeric and categorical clinical concepts embedded near many unrelated numbers, such as biomarker scores, lab values, stages, grades, dates, assay names, and treatment regimens.

For agent/harness extraction, the extractor must read each patient's complete note wholesale for the requested concepts. If a complete note is too large for one pass, use a recursive full-document strategy: split the note into sections that cover the entire text, read every section, carry forward candidate findings with provenance, and reconcile them into one patient-level value only after the whole note has been covered. Evidence-highlighted chunks, BoW terms, HTR spans, or retrieved snippets may guide where to pay attention, which candidates to extract, and what to audit, and they may be cited in evidence summaries; they are not sufficient extraction context by themselves.

Supported extraction routes:
- **Agent/harness document reading:** The invoking agent reads complete patient documents and emits structured values directly. For larger datasets, shard by patient ranges, folds, concepts, or concept families, but each patient/concept extraction must be based on the full patient note or a recursive pass that covers the full note. HTR-highlighted chunks can prioritize attention and audit targets, but they cannot replace full-note reading.
- **OpenAI-compatible endpoint extraction:** Use a running endpoint such as vLLM when the user supplies it, requests it, or the dataset is too large for reliable direct extraction. The endpoint is a document-reading extraction backend, not a proposal, review, or synthesis agent unless the user explicitly asks for that.
- **Repository extraction components:** Use `VLLMFeatureExtractor`, `extract_explicit_features()`, `VLLMExplicitFeatureExtractionProvider.ensure_features()`, and endpoint routing helpers when endpoint-backed extraction is available.

Extraction requirements:
- Values that cannot be recovered by document reading remain null/missing, but only after the complete patient note, or a recursive pass covering the complete note, has actually been read for the concept.
- Endpoint absence is not a missingness rationale. It triggers direct agent/harness extraction or a documented blocker.
- Do not emit an all-missing table because no endpoint exists.
- Preserve feature schema, roles, categories, value aliases, missingness flags, temporal labels, and evidence summaries.
- Post-extraction type coercion and category canonicalization may operate only on values already produced by document-reading extraction. It must not introduce values from regex, short-window parsing, category heuristics, or nearby-number rules.
- Audit extraction output before accepting it:
  - Compare structured values against the cited evidence summary and source text for sampled rows and all suspicious rows.
  - Include stratified audits across categorical levels, missingness, and boundary values. For numeric-to-category concepts, audit each category and the values near cutpoints.
  - Treat mismatches between values and evidence, impossible values, off-by-field numeric copies, invalid categories, shortcut-parser artifacts, or temporal leakage as extraction failures, not as missingness.
  - Quarantine failed shard files so downstream scripts cannot consume them accidentally.
  - Diagnose the failure mode: incomplete note coverage, oversized shard, prompt ambiguity, concept overload, temporal confusion, value copied from a nearby field, category alias problem, shortcut/rule leakage, or insufficient context.
  - Retry the failed work with a smaller or simpler scope while preserving complete-note coverage. Prefer this escalation sequence: original shard -> smaller patient shard -> single patient -> single concept or concept family -> recursive full-note reading with evidence-highlighted audit targets.
  - Re-read the relevant text during retry and include a short inconsistency-resolution note in the artifact.
  - Accept a retried row only when the final value, missing flag, temporal label, and evidence summary agree.

Failed extraction policy:
- A single failed shard, malformed output, or inconsistent audited row is not enough to halt the run.
- Do not proceed with downstream causal-forest claims using unaudited or failed shards.
- Declare extraction blocked only after documented retry attempts have failed at a scope small enough that the active harness should reasonably be able to read the complete required patient text, or after the user declines required approval/access.
- When blocked, record what was attempted, what failed, which rows/concepts remain unrecovered, and the next viable recovery path.

For local vLLM-backed extraction:
- Probe hardware and choose model, context length, dtype, tensor parallelism, GPU assignment, batch/concurrency limits, and output-token budget explicitly.
- Run a JSON extraction smoke test before the full pass, including at least one numeric variable and one categorical variable from a realistic note.
- Record server command, model, context limit, max input/text tokens, max generation tokens, prompt version, smoke-test result, and fallback reason in `report.txt`.

If extraction is genuinely infeasible after concrete retry attempts, stop and report the blocker instead of proceeding as though extraction succeeded.

## 5. Post-Extraction Feature Review

Do not treat extraction success as causal relevance. Review extracted features against the upstream evidence that motivated them.

Before candidate review, build `ensemble_nuisance_predictions.parquet` from all available out-of-fold BoW, HTR, and extracted-feature nuisance predictions. Use this ensemble, not a BoW-only signal, for final pseudo-outcomes, R-loss, candidate R-signal review, and role decisions. If a source is unavailable, record the disable reason and retry if the source should have been available.

Run fold-honest diagnostics:
- treatment nuisance association/performance
- outcome nuisance association/performance
- R-loss, logistic R-loss, pseudo-target, interaction, or treatment-stratified effect-modifier diagnostics
- missingness, category/value coverage, overlap warnings, and role-specific failures

Run candidate-level signal review for every extracted feature before parsimony. Each row in `candidate_signal_review.jsonl` must include the candidate name, fold context, treatment nuisance signal, outcome nuisance signal, ensemble R-pseudo-outcome or R-loss signal, interaction or treatment-stratified signal, missingness/overlap, role decision, and retry decision. If any candidate lacks R-signal review, retry this stage rather than proceeding.

Compare extracted-feature diagnostics with BoW, embedding-contrast, and HTR benchmarks. Large gaps should trigger one of:
- merge aliases or duplicate concepts
- fix categorical values or value aliases
- revise temporal anchoring
- re-role variables
- reject weak proxies
- add narrow evidence-supported concepts
- re-extract only changed concepts

Cap review rounds only after at least one targeted retry for each failed gate. Document remaining benchmark gaps, retry attempts, and whether final causal claims are allowed (`pass`) or blocked after retries (`blocked_after_retries`).

## 6. Role Evaluation

Evaluate confounders and effect modifiers separately. Use univariable screens as fold-aware diagnostics before and during multivariable modeling; they are not sufficient final role evidence.

Confounder evidence:
- predicts treatment in held-out folds
- predicts outcome or improves held-out outcome nuisance metrics
- improves residual balance or overlap diagnostics
- has stable direction, effect magnitude, and missingness behavior

Effect-modifier evidence:
- improves treatment-by-feature interaction objectives
- improves R-loss, logistic R-loss, or pseudo-outcome MSE
- produces fold-stable heterogeneity and ITE rankings
- has interpretable text/span/value evidence

Avoid promoting variables based only on marginal outcome association or p value.

## 7. Parsimony And Redundancy Review

Before finalizing variables:
- compute feature-feature correlations within training folds
- inspect categorical contingency tables
- inspect missingness overlap
- identify semantic duplicates and broad proxies
- test single-feature or grouped removals when the set is not already minimal
- revisit roles after each accepted revision

Record `retain_all`, `prune`, or `blocked`:
- `retain_all` is valid when the current list is compact or removals harm honest nuisance, heterogeneity, or ITE-stability metrics.
- `prune` should list removed variables and honest metric impact.
- `blocked` should state what prevents a defensible final forest.

## 8. Functional Form And ITEs

Before fitting final ITEs, run the final preflight gate from `references/gates.md`. If the full nuisance ensemble, candidate signal review, extraction audit, parsimony review, or causal-forest implementation check fails, retry the failed stage and rerun preflight. Do not produce final `ite_estimates.parquet` until preflight passes or the run is documented as `blocked_after_retries`.

Do not assume the final DGP is linear. Compare flexible nuisance/effect models, tree/forest-based heterogeneity, interactions suggested by R-loss or pseudo-outcomes, and simple parametric summaries only as explanations.

After final confounders and effect modifiers are settled, fit an honest causal forest as the required final ITE estimator. Generic random forests, ExtraTrees, XGBoost, or meta-learner final-stage regressors are diagnostics only unless they are part of a real causal-forest implementation.

For every patient, produce counterfactual outcome estimates and ITEs only from models that did not train on that row, or from a final model whose selection was completed using nested honest folds and whose report clearly states the refit convention.

Stop when new variables do not improve honest metrics, fold recurrence, temporal anchoring, extracted-feature benchmark gaps, parsimony, or ITE stability.
