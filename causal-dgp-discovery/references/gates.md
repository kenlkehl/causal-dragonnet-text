# Workflow Gates And Retries

Use this reference to convert the workflow into hard gates. A gate failure means retry the responsible stage; it does not mean immediately stop the run.

## Gate Status Values

Use only these statuses in `workflow_gate_status.json`:
- `pending`: not yet checked
- `pass`: checked and usable downstream
- `retrying`: failed and a targeted retry is in progress or planned
- `blocked_after_retries`: failed after documented retries at a narrow/corrected scope

Final causal-forest fitting is allowed only when all upstream gates are `pass`.

## Retry Rule

When a gate fails:
1. Record the failed condition in `workflow_gate_status.json`.
2. Append a `gate_retry_log.jsonl` record with diagnosis and retry scope.
3. Retry the narrowest responsible stage, not the whole workflow.
4. Re-run the gate.
5. Repeat with a narrower scope or corrected spec/prompt/model if the gate still fails.
6. Use `blocked_after_retries` only when further progress requires user input, external access, or a genuinely unavailable capability.

## Required Gates

### `bow_evidence_gate`
Pass requires multiple honest BoW/TF-IDF views, out-of-fold treatment and outcome nuisance predictions, preliminary residual/R or pseudo-outcome evidence, and fold-specific text evidence.

Retry examples: reduce feature caps, fix fold leakage, rerun failed views, or replace a failed learner with a documented equivalent.

### `embedding_contrast_gate`
Pass requires training-fold chunk provenance and retrieved chunks for treatment, outcome, ensemble/preliminary R, within-arm outcome, and treatment-outcome contrasts when feasible.

Retry examples: rebuild chunks with fold labels, reduce chunk count, switch to cached/local embeddings, or document unavailable contrast support.

### `htr_evidence_gate`
Pass requires a hardware/runtime probe and either cross-fitted HTR nuisance/effect evidence with attention/span outputs or a documented retry path. If HTR is unavailable, retry with a smaller model/pass before disabling it.

Retry examples: smaller context, fewer folds, CPU/small-model span localization, or cached feature extractor.

### `extraction_audit_gate`
Pass requires candidate features produced by complete-note document reading, valid schema, no accepted quarantined shards, sampled value/evidence consistency checks, suspicious-row audits, stratified audits for categorical levels and boundary numeric values, and retry records for failures. BoW, embedding-contrast, and HTR evidence should guide candidate selection and audit targeting, but agent-based extraction must read each patient's complete note or use a recursive pass covering the complete note; isolated snippets, short windows, regex/pattern matching, nearby-number rules, or category heuristics do not satisfy this gate.

Retry examples: smaller shard, single patient, single concept family, clarified temporal prompt, or recursive full-note reading with evidence-highlighted audit targets.

### `nuisance_ensemble_gate`
Pass requires `ensemble_nuisance_predictions.parquet` containing all available out-of-fold nuisance sources and final ensemble residual/R-pseudo-outcome columns. HTR or extracted-feature nuisance predictions must be included if they exist and are used downstream. BoW-only R signals fail this gate for final review unless other sources are genuinely unavailable and retried/disabled with reasons.

Retry examples: align folds/patient ids, rebuild source predictions, average only passing sources with documented weights, or rerun missing nuisance source.

### `candidate_signal_review_gate`
Pass requires `candidate_signal_review.jsonl` records for every retained candidate. Each candidate needs fold-aware treatment nuisance, outcome nuisance, ensemble R-pseudo-outcome/R-loss, interaction or treatment-stratified signal, missingness/overlap, and role decision fields.

Retry examples: compute missing diagnostics, re-role variables, merge aliases, add narrow evidence-supported candidates, or re-extract/re-review changed concepts.

### `extracted_feature_review_gate`
Pass requires extracted-feature metrics compared against upstream BoW, embedding, and HTR benchmarks, with documented decisions for underperforming concepts.

Retry examples: fix values/categories, revise temporal anchoring, re-extract changed concepts, or rerun review after ensemble rebuild.

### `parsimony_gate`
Pass requires parsimony after candidate signal review, including redundancy, missingness overlap, categorical contingency summaries, and honest ablation/removal diagnostics where useful.

Retry examples: rerun after candidate re-role/drop/merge, test grouped removals, or retain all with metric-supported rationale.

### `final_preflight_gate`
Pass requires all upstream gates `pass`, final ensemble R signal present, candidate signal review complete, parsimony complete, and final estimator configuration set to a real honest causal forest.

Retry examples: rerun the specific failed gate. Do not fit final ITEs while preflight is `retrying`.

### `final_causal_forest_gate`
Pass requires final ITEs from `CausalForestDML` or an equivalent honest causal forest, with patient-level fold/model provenance and no in-sample row-level estimates used for reporting.

Retry examples: fix estimator, folds, feature matrices, missingness handling, or row provenance.
