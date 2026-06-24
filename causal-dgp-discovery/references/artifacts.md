# Required Artifacts

Write outputs into the task dataset folder unless the user requests another location.

## `report.txt`

Maintain this throughout the run. Include:
- dataset schema and basic rates
- text-model evidence and fold recurrence
- candidate concepts and why each was proposed
- extraction rules and missingness
- rejected hypotheses
- confounder/modifier role evidence
- model comparison table
- final inferred DGP, including uncertainty
- ITE summary and causal-forest comparison

## `text_evidence.parquet` or `text_evidence.jsonl`

Store fold-specific evidence:
- fold id
- evidence source (`bow`, `attention`, `residual`, `pseudo_outcome`, `r_loss`)
- term or span
- direction and score
- mapped candidate concept if any
- whether it recurred across folds

## `candidate_features.parquet` or `.csv`

Store extracted concept values:
- `patient_id`
- extracted candidate variables
- missingness flags
- extraction backend (`coding_agent`, `openai_compatible_endpoint`, `vllm_python_api`, or other documented backend)
- extraction model/endpoint/config hash when endpoint-backed extraction is used
- extraction confidence/source/evidence summary when available
- temporal label such as baseline/pre-treatment when relevant

Primary clinical variable extraction should be LLM-based document reading. If regex was used for any value, include method/source columns or a companion audit table that identifies those variables and explains the fallback.

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
- treatment nuisance AUROC/Brier/log loss
- outcome nuisance AUROC/Brier/log loss or RMSE
- R-loss, logistic R-loss, or pseudo-outcome MSE
- fold recurrence of candidate concepts
- ITE distribution and fold-to-fold correlation
- causal forest comparison metrics
