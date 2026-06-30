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
Pass requires training-fold chunk provenance and retrieved chunks for treatment, outcome, ensemble/preliminary R, within-arm outcome, treatment-outcome cell, orthogonal R-score, and concept-probe contrasts when feasible. If a contrast family is unavailable, the gate must record the reason and a targeted retry or fallback. Outcome, R, orthogonal, and within-arm contrasts must be reviewed explicitly for numeric values, laboratory panels, treatment-history/regimen evidence, status categories, and derived quantities before candidate extraction.

Retry examples: rebuild chunks with fold labels, reduce chunk count, switch to cached/local embeddings, or document unavailable contrast support.

### `htr_evidence_gate`
Pass requires a hardware/runtime probe plus cross-fitted HTR evidence for both nuisance targets and effect/heterogeneity targets, with attention/span or neural hidden-state attribution outputs for each. Nuisance-only HTR evidence, generic chunk localization, or a hardware probe without effect/R-stage attribution fails this gate.

Accepted HTR evidence must come from an actual HTR/neural attention or hidden-state attribution workflow, such as `AgenticAttentionVariableForestRunner`, `MultiModelHTREvidenceProvider`, cached hidden states from a neural text model, or a smaller CPU/GPU run of the same class of neural HTR architecture. If system tools such as `nvidia-smi` see GPUs but the framework cannot initialize CUDA/NVML or reports zero devices, the gate must first record a failed sandboxed probe and a targeted retry using the active harness approval/escalation mechanism. Do not mark GPU unavailable until that escalation is attempted or the user declines it.

Sparse BoW/TF-IDF models, linear/logistic/Ridge coefficient chunk scoring, dense TF-IDF/SVD chunk retrieval, embedding/concept-probe retrieval, generic chunk localization, or any other lexical/sparse-text substitute must not be labeled HTR and must not pass this gate. These artifacts may be recorded under BoW or embedding evidence only. Mark `blocked_after_retries`, not `pass`, if real nuisance and effect-stage HTR attribution cannot be produced after documented GPU escalation and narrow neural retries.

Retry examples: escalated GPU probe/HTR command when sandboxing is suspected, smaller neural HTR context, fewer folds, fewer epochs, smaller neural HTR model, CPU execution of the neural HTR runner, cached hidden-state extraction from a neural model, or repaired device mapping.

### `extraction_audit_gate`
Pass requires candidate features produced by complete-note document reading, valid schema, no accepted quarantined shards, sampled value/evidence consistency checks, suspicious-row audits, stratified audits for categorical levels and boundary numeric values, and retry records for failures. BoW, embedding-contrast, and HTR evidence should guide candidate selection and audit targeting, but agent-based extraction must read each patient's complete note or use a recursive pass covering the complete note; isolated snippets, short windows, regex/pattern matching, nearby-number rules, or category heuristics do not satisfy this gate. The extraction plan must be the harmonized union of nuisance/confounder and effect-modifier candidate batches for the current iteration; extracting after only nuisance discovery or only effect discovery fails this gate.

Retry examples: smaller shard, single patient, single concept family, clarified temporal prompt, or recursive full-note reading with evidence-highlighted audit targets.

### `nuisance_ensemble_gate`
Pass requires `ensemble_nuisance_predictions.parquet` containing all available out-of-fold nuisance sources and final ensemble residual/R-pseudo-outcome columns. HTR or extracted-feature nuisance predictions must be included if they exist and are used downstream. BoW-only R signals fail this gate for final review unless other sources are genuinely unavailable and retried/disabled with reasons.

Retry examples: align folds/patient ids, rebuild source predictions, average only passing sources with documented weights, or rerun missing nuisance source.

### `candidate_signal_review_gate`
Pass requires `candidate_signal_review.jsonl` records for every retained candidate. Each candidate needs fold-aware treatment nuisance, outcome nuisance, ensemble R-pseudo-outcome/R-loss, interaction or treatment-stratified signal, missingness/overlap, and role decision fields.

The gate also requires a candidate-translation coverage record showing that high-rank/recurrent BoW, embedding, and HTR effect/R-stage evidence involving numeric values, laboratory panels, treatment-history/regimen evidence, status categories, and derived quantities was mapped, merged, or rejected with a concrete reason. A candidate set can be compact, but it cannot silently omit an evidence-supported extractable concept family.

The gate also requires `discovery_iteration_trace.jsonl` records showing the current iteration order: nuisance discovery with BoW/embedding/HTR, confounder translation, effect-modification discovery with BoW/embedding/HTR, effect-modifier translation, harmonized extraction plan, extraction, value harmonization, parsimony, and efficacy/ITE diagnostics.

Retry examples: compute missing diagnostics, re-role variables, merge aliases, add narrow evidence-supported candidates, or re-extract/re-review changed concepts.

### `extracted_feature_review_gate`
Pass requires extracted-feature metrics compared against upstream BoW, embedding, and HTR benchmarks, with documented decisions for underperforming concepts.

Retry examples: fix values/categories, revise temporal anchoring, re-extract changed concepts, or rerun review after ensemble rebuild.

### `parsimony_gate`
Pass requires parsimony after candidate signal review, including redundancy, missingness overlap, categorical contingency summaries, and honest ablation/removal diagnostics where useful.

Retry examples: rerun after candidate re-role/drop/merge, test grouped removals, or retain all with metric-supported rationale.

### `final_preflight_gate`
Pass requires all upstream gates `pass`, final ensemble R signal present, discovery iteration trace complete, candidate signal review complete, candidate-translation coverage review complete, parsimony complete, HTR effect/R-stage attribution present, and final estimator configuration set to a real honest causal forest. A run with missing, blocked, or nuisance-only HTR effect evidence must not pass preflight. A run where extraction occurred before harmonizing both nuisance/confounder and effect-modifier candidate batches must not pass preflight.

Retry examples: rerun the specific failed gate. Do not fit final ITEs while preflight is `retrying`.

### `final_causal_forest_gate`
Pass requires final ITEs from `CausalForestDML` or an equivalent honest causal forest, with patient-level fold/model provenance and no in-sample row-level estimates used for reporting.

Retry examples: fix estimator, folds, feature matrices, missingness handling, or row provenance.
