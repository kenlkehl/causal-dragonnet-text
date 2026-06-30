# Required Artifacts

Write outputs into the task dataset folder unless the user requests another location. Every artifact used for selection or reporting must preserve fold provenance and avoid in-sample predictions for scored rows.

## `report.txt`

Maintain this throughout the run. Include:
- dataset schema, row count, text length, missingness, treatment/outcome rates, and chronology assumptions
- fold construction and honesty rules
- BoW vectorization-suite settings, nuisance/effect metrics, and fold recurrence
- embedding-contrast retrieval settings and evidence
- HTR modeling, attention/span attribution evidence, hardware plan, and disable reasons if any
- discovery iteration trace showing nuisance discovery, confounder translation, effect-modification discovery, effect-modifier translation, harmonized extraction, value harmonization, parsimony, efficacy/ITE diagnostics, and loop-back decisions
- candidate concepts and why each was proposed from text evidence
- extraction backend, feature specs, missingness, and extraction rationale
- post-extraction diagnostics and benchmark-review decisions
- rejected hypotheses and revision rounds
- confounder/modifier role evidence
- feature-correlation, redundancy, and parsimony review with `retain_all`, `prune`, or `blocked`
- model comparison table
- final inferred DGP and remaining uncertainty
- workflow gate status, failed-gate retries, final preflight result, and whether any gate required retry before passing
- final causal-forest ITE summary and sensitivity comparisons

## `discovery_iteration_trace.jsonl`

Store one record per discovery/review iteration. This artifact makes the temporal order auditable and is required before final preflight.

Each record should include:
- iteration id and fold context
- nuisance/confounder discovery artifacts from BoW, embedding contrasts, and HTR nuisance attribution
- confounder candidate decisions from nuisance evidence
- effect-modification discovery artifacts from BoW R/interaction evidence, embedding R/orthogonal/within-arm/treatment-outcome/concept-probe contrasts, and HTR effect/R-stage attribution
- effect-modifier candidate decisions from effect evidence
- harmonized extraction plan after merging confounder and effect-modifier candidates
- extraction delta: new concepts, changed concepts, retried concepts, and unchanged concepts reused from prior iterations
- value harmonization decisions
- candidate signal review, parsimony, nuisance ensemble, efficacy/ITE diagnostic summaries
- loop-back trigger when diagnostics are suboptimal, or stop reason when no evidence-supported revision remains

The first extraction in an iteration must happen after both nuisance and effect-modification candidate batches are translated and harmonized. Later iterations should re-extract only changed or newly added concepts unless an audit failure requires a broader retry.


## `workflow_gate_status.json`

Maintain a machine-readable gate ledger from the beginning of the run. Each gate record should include:
- `status`: `pending`, `pass`, `retrying`, or `blocked_after_retries`
- `attempt_count`
- `last_checked_at` when available
- `required_artifacts`
- `failure_reason` when not passing
- `next_retry` or `retry_exhaustion_reason`

Required gates:
- `initial_exploration_gate`
- `fold_construction_gate`
- `bow_evidence_gate`
- `embedding_contrast_gate`
- `htr_evidence_gate`
- `extraction_audit_gate`
- `nuisance_ensemble_gate`
- `candidate_signal_review_gate`
- `extracted_feature_review_gate`
- `parsimony_gate`
- `final_preflight_gate`
- `final_causal_forest_gate`

A failed gate must trigger a targeted retry and a `gate_retry_log.jsonl` record. Do not mark a gate `blocked_after_retries` until retry attempts are documented at a narrower or corrected scope.

## `gate_retry_log.jsonl`

Store every failed-gate retry attempt:
- gate name
- failed condition
- diagnosis
- retry scope
- changed code/spec/prompt/data subset
- result
- next action

## `ensemble_nuisance_predictions.parquet`

Store full out-of-fold nuisance ensemble predictions used for final R signals:
- `patient_id`
- `fold`
- source-specific `e_hat_*` and `m_hat_*` columns for BoW, HTR, extracted features, and any other retained nuisance source
- `e_hat_ensemble`
- `m_hat_ensemble`
- `treatment_residual_ensemble`
- `outcome_residual_ensemble`
- `r_pseudo_outcome_ensemble`
- source weights or averaging rule
- disabled-source reasons when a planned source is absent

BoW-only pseudo-outcomes may be stored as early discovery artifacts, but final candidate review must use this ensemble artifact.

## `candidate_signal_review.jsonl`

Store one or more records per candidate before parsimony:
- candidate name, role under review, and fold/review context
- treatment nuisance association and metric delta
- outcome nuisance association and metric delta
- ensemble R-pseudo-outcome association, R-loss delta, logistic R-loss, or equivalent heterogeneity metric
- treatment-by-feature interaction or treatment-stratified outcome signal when applicable
- effect magnitude, direction, fold recurrence, missingness, category/value coverage, and overlap warnings
- upstream evidence links from BoW, embedding, and HTR
- decision: retained, dropped, re-roled, merged, targeted for re-extraction, or targeted for re-review
- retry status when diagnostics are missing or fail thresholds

This artifact is required before `parsimony_review_by_fold.jsonl`.

## `candidate_translation_coverage.jsonl`

Store a pre-extraction coverage review that proves candidate translation did not silently skip high-signal evidence. This artifact is required before extraction and is referenced by `candidate_signal_review_gate`.

Each record should include:
- evidence source: `bow`, `embedding_contrast`, `htr_attention`, `htr_attribution`, `ensemble`, or another documented source
- fold, contrast/objective, rank, score, and evidence identifier or row pointer
- evidence family: numeric value, laboratory panel, treatment-history/regimen evidence, status category, derived quantity, molecular marker, staging/anatomic finding, functional status, comorbidity, or another explicit family
- extracted concept decision: `mapped_to_candidate`, `merged_with_candidate`, `rejected`, or `needs_retry`
- candidate name when mapped or merged
- rejection reason when rejected
- reviewer note explaining why the concept is or is not an extractable baseline covariate or effect modifier under the task-level pre-treatment/pre-outcome text assumption

At minimum, cover high-rank/recurrent evidence from outcome, R-pseudo-outcome, orthogonal R-score, within-arm outcome, treatment-outcome cell, and HTR effect/R-stage attribution targets. Generic evidence labels such as routine monitoring or broad lab-panel names should be traced to the specific extractable value, count, ratio, category, or derived quantity present in the retrieved chunks/spans.

## `final_preflight_check.json`

Before final causal forest fitting, record:
- all required gates and statuses
- whether `discovery_iteration_trace.jsonl` exists and shows the required order: nuisance discovery, confounder translation, effect-modification discovery, effect-modifier translation, candidate harmonization, extraction, value harmonization, parsimony, efficacy/ITE diagnostics, and any loop-back iterations
- whether `ensemble_nuisance_predictions.parquet` exists and contains the final R signal
- whether every retained candidate appears in `candidate_signal_review.jsonl`
- whether `candidate_translation_coverage.jsonl` exists and covers high-rank/recurrent numeric, laboratory, treatment-history/regimen, status, and derived-quantity evidence from BoW, embedding, and HTR effect/R-stage outputs
- whether extraction audit failures were retried or quarantined
- whether parsimony ran after candidate signal review
- whether HTR effect/R-stage attribution exists from an accepted HTR/neural attention or hidden-state attribution workflow; nuisance-only HTR evidence is not sufficient, and BoW/TF-IDF chunk scoring, coefficient-based sparse attribution, embedding/concept-probe retrieval, or generic chunk localization must not satisfy this requirement
- whether the final estimator is a real honest causal forest
- `decision`: `pass`, `retry_required`, or `blocked_after_retries`

If decision is `retry_required`, retry the failed stage and rerun preflight. Do not write final ITE estimates until preflight passes.

## `text_evidence.parquet` Or `text_evidence.jsonl`

Store fold-specific text evidence:
- fold id and split provenance
- evidence source: `bow`, `embedding_contrast`, `htr_attention`, `htr_attribution`, `residual`, `pseudo_outcome`, `r_loss`, or `ensemble`
- vectorization run label and vectorizer parameters when source is BoW/TF-IDF
- term, phrase, retrieved chunk, token span, or attribution target
- direction, score, and model family
- patient/chunk provenance when available
- mapped candidate concept if any
- fold recurrence and cross-view recurrence
- embedding contrast, HTR nuisance/effect, or ensemble-R provenance when relevant

## `candidate_features.parquet` Or `.csv`

Store extracted concept values:
- `patient_id`
- extracted candidate variables
- missingness flags
- extraction backend, such as `agent_document_reading`, `openai_compatible_endpoint`, `vllm_server`, or another documented backend
- extraction model/endpoint/config hash when endpoint-backed extraction is used
- local server command, model name/path, served model name, dtype, tensor parallelism, max context/input/output settings, GPU assignment, batch/concurrency settings, and smoke-test status when applicable
- extraction confidence, source section/chunk/span summary, or evidence summary when available
- temporal label such as baseline/pre-treatment when relevant
- feature-spec version or hash

Clinical variable extraction must be document reading. If no endpoint-backed extractor is available, the invoking agent/harness must extract directly from complete patient documents, or from a recursive extraction pass whose sections cover each complete patient document, or report a blocker after a concrete sharding attempt. Evidence-supported chunks may guide feature candidacy, attention, temporal anchoring, and audits, but they must not be the only context for accepted extracted values. Endpoint absence is not a valid all-row missingness rationale.

Missing values are valid only for patient/concept pairs that complete-note document-reading extraction could not recover from the text. An all-missing artifact is valid only if the complete documents, or recursive passes covering complete documents, were actually reviewed for the requested concepts and the unrecoverability is justified.

Accepted candidate-feature artifacts must exclude quarantined failed shards. If a shard is retried, preserve only the accepted latest row per patient/concept in `candidate_features.parquet` or `.csv`; keep failed attempts in separate rejected/retry artifacts.

## `candidate_feature_review.jsonl`

Store each candidate-list review iteration:
- iteration id and fold/review context
- candidate variables, roles, categories, value aliases, and proposed transformations
- treatment association, outcome association, and effect-modifier diagnostics
- effect magnitude, score delta, direction, fold recurrence, missingness, overlap warnings, and p value when useful
- feature-feature correlations, categorical contingency summaries, and missingness overlap
- variables merged, rejected, re-roled, retained, or targeted for re-extraction, with rationale
- upstream BoW/embedding/HTR benchmark gaps
- final review decision for the iteration
- extraction audit and retry records, including inconsistent values, cited evidence, failure mode, retry scope, retry result, and whether the row/concept was accepted, quarantined, or still blocked

## `extracted_feature_diagnostics_by_fold.jsonl`

Store post-extraction feature-review records:
- outer fold, inner fold or review context, review round, and split provenance
- extracted feature specs, roles, categories, aliases, and missingness summaries
- extracted-feature treatment nuisance metrics
- extracted-feature outcome nuisance metrics
- extracted-feature R-loss, logistic R-loss, pseudo-target, interaction, or treatment-stratified diagnostics computed from the full ensemble R signal when used for role decisions
- upstream BoW/TF-IDF, embedding-contrast, and HTR benchmark metrics used for comparison
- gate thresholds or tolerance rules, pass/fail status, and margin by role/objective
- revision decision: retained, dropped, re-roled, merged, alias-harmonized, value-harmonized, newly added, or targeted for re-extraction
- stop reason when review rounds are capped or no evidence-supported revision remains

## `parsimony_review_by_fold.jsonl`

Store the mandatory pre-forest parsimony gate:
- outer fold and selected features before/after review
- continuous feature correlations above threshold
- categorical contingency summaries
- missingness-overlap summaries
- tested single-feature or grouped ablations, including metric deltas
- user-supplied/protected features that were not eligible for automatic removal
- final decision: `retain_all`, `prune`, or `blocked`, plus rationale

This artifact is required even when no variables are pruned. It must be written after `candidate_signal_review.jsonl`. No diagnostic in it should be based on in-sample predictions for scored rows.

## `crossfit_predictions.parquet`

Store honest fold predictions:
- `patient_id`
- fold id
- `e_hat`
- `m_hat`
- treatment residual
- outcome residual
- pseudo-outcome or R-loss target
- model family and iteration
- nuisance/effect source, such as BoW view, HTR nuisance/effect, or ensemble nuisance
- fold-specific ITE estimate when available

No row should contain a prediction from a model trained on that same row. If multiple nuisance sources are used, also write `ensemble_nuisance_predictions.parquet` with the final ensemble R signal.

## `ite_estimates.parquet`

Store final patient-level effects:
- `patient_id`
- `p_y_do_t0`
- `p_y_do_t1`
- `ite`
- fold/model provenance
- missingness reason if ITE is unavailable
- causal-forest implementation/provenance
- optional non-causal-forest sensitivity estimates, clearly named as such

Do not name a column `causal_forest_*` unless it was produced by a real causal forest implementation, such as `CausalForestDML` or an equivalent honest causal forest.

## `model_comparison.json` Or `.csv`

Store iteration/model diagnostics:
- BoW view labels and vectorizer parameters
- treatment nuisance metrics
- outcome nuisance metrics
- R-loss, logistic R-loss, or pseudo-outcome metrics
- embedding-contrast evidence summary
- HTR nuisance/effect metrics and attention/span attribution coverage
- extracted-feature metrics versus BoW/embedding/HTR benchmarks
- extracted-feature review pass/fail status and review-round count
- fold recurrence of candidate concepts
- parsimony decision summary
- ITE distribution and fold-to-fold stability
- final causal-forest metrics and clearly labeled non-causal-forest sensitivity comparisons
