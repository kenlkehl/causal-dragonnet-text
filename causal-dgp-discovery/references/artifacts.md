# Required Artifacts

Write outputs into the task dataset folder unless the user requests another location.

## `report.txt`

Maintain this throughout the run. Include:
- dataset schema and basic rates
- text-model evidence, BoW vectorization-suite settings, and fold recurrence
- candidate concepts and why each was proposed
- extraction rules and missingness
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
- evidence source (`bow`, `attention`, `residual`, `pseudo_outcome`, `r_loss`)
- term or span
- direction and score
- mapped candidate concept if any
- whether it recurred across folds
- whether it recurred across vectorization strategies

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
- optional causal-forest ITE for comparison

## `model_comparison.json` or `.csv`

Store iteration/model diagnostics:
- vectorization strategy label and parameters for BoW-suite runs
- treatment nuisance AUROC/Brier/log loss
- outcome nuisance AUROC/Brier/log loss or RMSE
- R-loss, logistic R-loss, or pseudo-outcome MSE
- fold recurrence of candidate concepts
- parsimony/redundancy review summary
- ITE distribution and fold-to-fold correlation
- causal forest comparison metrics
