# Required Artifacts

Write outputs into the task dataset folder unless the user requests another location.

## `report.txt`

Maintain this throughout the run. Include:
- dataset schema and basic rates
- text-model evidence, BoW vectorization-suite settings, and fold recurrence
- embedding-contrast retrieval evidence and HTR attention/span evidence, including any disable reasons
- candidate concepts and why each was proposed
- extraction rules and missingness
- post-extraction extracted-feature diagnostics and benchmark-review decisions
- rejected hypotheses
- confounder/modifier role evidence
- feature-correlation, redundancy, and parsimony reviews
- repeated candidate-list revision decisions
- model comparison table
- final inferred DGP, including uncertainty
- ITE summary and causal-forest comparison

## `text_evidence.parquet` or `text_evidence.jsonl`

Store fold-specific evidence:
- fold id
- vectorization run label and vectorizer params when source is BoW/TF-IDF
- evidence source (`bow`, `embedding_contrast`, `htr_attention`, `residual`, `pseudo_outcome`, `r_loss`)
- term or span
- direction and score
- mapped candidate concept if any
- whether it recurred across folds
- whether it recurred across vectorization strategies
- embedding-contrast, HTR nuisance/effect, or ensemble-R provenance when relevant

## `candidate_features.parquet` or `.csv`

Store extracted concept values:
- `patient_id`
- extracted candidate variables
- missingness flags
- extraction backend (`coding_agent`, `openai_compatible_endpoint`, `vllm_python_api`, or other documented backend)
- extraction model/endpoint/config hash when endpoint-backed extraction is used
- for local HF/vLLM extraction: server command, model name/path, served model name, dtype, tensor parallelism, max model length, max input/text tokens, max generation tokens, GPU assignment, batch/concurrency settings, and JSON smoke-test status
- extraction confidence/source/evidence summary when available
- source chunk, section, or span summary used for coding-agent extraction when available
- temporal label such as baseline/pre-treatment when relevant

Clinical variable extraction must be LLM-based document reading. If no endpoint-backed extractor is available, the coding agent itself must read the documents, using subagents or sharded manual passes when useful, and emit the structured feature table. If a value cannot be recovered by document reading for a specific patient/concept, leave it missing/null and include source/evidence or missingness rationale columns; do not fill values with regex or pattern-matching fallback logic.

Do not use a run-level explanation such as "no LLM endpoint/backend available" as the missingness rationale for all candidate values. Endpoint absence triggers coding-agent extraction; it does not satisfy extraction. An all-missing `candidate_features` artifact is valid only if the coding agent actually read the relevant documents and found every requested concept unrecoverable, which should be rare and must be justified patient-by-patient or concept-by-concept. If coding-agent extraction is genuinely infeasible after a concrete sharding/subagent attempt, report the blocker and do not present downstream causal-forest or final ITE artifacts as complete.

## `candidate_feature_review.jsonl`

Store each candidate-list review iteration:
- iteration id and training fold or outer-fold context
- candidate variables, roles, and proposed transformations
- treatment association, outcome association, and treatment-by-feature or treatment-stratified outcome diagnostics
- effect magnitude, score delta, direction, fold recurrence, missingness, and p value when applicable
- feature-feature correlations, categorical contingency summaries, and missingness overlap
- variables merged, rejected, re-roled, or retained, with rationale
- parsimony decision and impact on honest nuisance, R-loss/pseudo-outcome, heterogeneity, or ITE-stability metrics
- upstream BoW/embedding/HTR benchmark gaps and whether the candidate list requires revision or re-extraction

## `extracted_feature_diagnostics_by_fold.jsonl`

Store the post-extraction feature-review records that decide whether extracted variables are adequate for final causal-forest use:
- outer fold, inner fold or review context, review round, and honest split provenance
- extracted feature specs, roles, categories, aliases, and missingness summaries
- extracted-feature treatment nuisance metrics, outcome nuisance metrics, and overlap warnings
- extracted-feature R-loss, logistic R-loss, pseudo-target, interaction, or treatment-stratified effect-modifier diagnostics
- upstream BoW/TF-IDF, embedding-contrast, and HTR benchmark metrics used for comparison
- gate thresholds, pass/fail status, and margin by role/objective
- Codex revision decision: retained, dropped, re-roled, merged, alias-harmonized, value-harmonized, newly added, or targeted for re-extraction
- stop reason when review rounds are capped or no evidence-supported revision remains

No diagnostic in this artifact should be based on in-sample predictions for the scored rows.

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
- nuisance/effect source such as BoW view, HTR nuisance/effect, or ensemble nuisance
- any fold-specific ITE estimate

No row in this file should contain a prediction from a model trained on that same row.

## `ite_estimates.parquet`

Store final patient-level effects:
- `patient_id`
- `p_y_do_t0`
- `p_y_do_t1`
- `ite`
- fold/model provenance
- missingness reason if ITE is unavailable
- causal-forest implementation/provenance, because final ITEs must come from a real honest causal forest after final confounders and effect modifiers are settled
- optional non-causal-forest sensitivity estimates, clearly named as such

Do not name a column `causal_forest_*` unless it was produced by a real causal forest implementation, such as the repository's explicit-feature forest path, `econml.dml.CausalForestDML`, or an equivalent honest causal forest. A generic random-forest regressor used inside an R-learner or other meta-learner is not a causal forest.

## `model_comparison.json` or `.csv`

Store iteration/model diagnostics:
- vectorization strategy label and parameters for BoW-suite runs
- treatment nuisance AUROC/Brier/log loss
- outcome nuisance AUROC/Brier/log loss or RMSE
- R-loss, logistic R-loss, or pseudo-outcome MSE
- extracted-feature nuisance and R/pseudo-target metrics versus BoW/embedding/HTR benchmarks
- HTR nuisance/effect metrics and attention/span evidence coverage when available
- extracted-feature review pass/fail status and review-round count
- fold recurrence of candidate concepts
- parsimony/redundancy review summary
- ITE distribution and fold-to-fold correlation
- required final causal-forest metrics and any clearly labeled non-causal-forest sensitivity comparisons
