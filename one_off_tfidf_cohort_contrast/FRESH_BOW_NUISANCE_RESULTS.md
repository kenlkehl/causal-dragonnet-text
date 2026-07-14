# Fresh BoW nuisance calibration benchmark

This benchmark asks whether training new fold-local TF-IDF/logistic nuisance
models improves calibration relative to the nuisance predictions saved by the
earlier multi-model experiment. It compares:

1. `fresh_bow_stacked`: 16 new word TF-IDF logistic models (four n-gram views
   and four regularization strengths), combined using honest inner-fold
   out-of-fold logistic stacking.
2. `reused_crossfit_stack`: the earlier BoW/HTR predictions, but recalibrated
   and combined using the same honest cross-fitted stacking procedure.
3. `reused_probability_mean`: the earlier experiment's uncalibrated ensemble
   probability mean.

All reported test results are outer-fold held out. The known synthetic-data
probabilities are used only for evaluation.

## Five-fold result

| model | treatment Brier | treatment slope | true e RMSE | outcome Brier | outcome slope | true m RMSE | residual effect |
|---|---:|---:|---:|---:|---:|---:|---:|
| fresh BoW stack | 0.1619 | 0.954 | 0.1231 | 0.1730 | 0.967 | 0.1160 | 0.1182 |
| reused cross-fit stack | **0.1597** | 1.091 | **0.0964** | **0.1711** | 1.089 | **0.1024** | 0.1203 |
| reused probability mean | 0.1842 | 2.526 | 0.1776 | 0.1929 | 2.570 | 0.1793 | 0.2309 |

The fresh models have calibration slopes closer to one than the recalibrated
earlier stack, but they have worse Brier scores and substantially worse RMSE
against the known true nuisance probabilities. Thus, retraining fixes the
large under-dispersion of the raw probability mean, but it does not provide a
better nuisance model than recalibrating and stacking the earlier candidates.

The true mean treatment effect is 0.0627. With oracle nuisance probabilities,
the same finite-sample residual-effect calculation averages 0.0896, so some of
the remaining difference is sampling noise. Fresh and reused cross-fit stacks
give almost identical residual effects (0.1182 versus 0.1203); the raw mean is
much worse (0.2309).

## Outer fold 1

| model | treatment Brier | treatment slope | true e RMSE | outcome Brier | outcome slope | true m RMSE | residual effect |
|---|---:|---:|---:|---:|---:|---:|---:|
| fresh BoW stack | 0.1501 | 1.099 | 0.1147 | 0.1821 | 0.898 | 0.1159 | 0.1807 |
| reused cross-fit stack | **0.1481** | 1.249 | **0.0951** | **0.1788** | 0.979 | **0.1059** | 0.1827 |
| reused probability mean | 0.1778 | 2.773 | 0.1786 | 0.1965 | 2.263 | 0.1761 | 0.2781 |

Fold 1 tells the same story: the fresh stack is dramatically better than the
raw mean, but not better overall than the recalibrated earlier stack. Its
treatment slope is closer to one, while its Brier scores and true-probability
RMSE are worse. The fold's true ATE is 0.0637 and its oracle-nuisance residual
estimate is 0.1366; the fresh and reused stacks are again nearly tied.

Full metrics and reusable nuisance predictions are in
`results_fresh_bow_nuisance/`.
