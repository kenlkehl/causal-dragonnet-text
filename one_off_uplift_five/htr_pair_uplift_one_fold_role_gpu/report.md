# One-fold HTR matched-pair uplift check

Dataset:
`synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet`

Run:

- Outer split: 800 train / 200 test, stratified by treatment/outcome.
- Test treated rows: 105.
- Nuisance inputs: prior honest TF-IDF propensity and outcome probabilities.
- Matching: +/- 0.05 propensity and +/- 0.05 outcome probability; up to 3 controls per candidate plus 1 nearest fallback.
- HTR: cached local `unsloth/bge-small-en-v1.5` transformer backend, token-attention pooling, role attention enabled, 2 inner folds, 4 epochs.

## Predictive result

The HTR uplift model did not learn a useful treatment-effect ranking in this one-fold run.

- Inner held-out treated AUROC: 0.738 and 0.754 by inner fold.
- Combined train OOF treated AUROC: 0.746.
- Outer-test actual-treated baseline AUROC: 0.680.
- Outer-test HTR uplift AUROC: 0.665.
- Outer-test HTR uplift Brier/log-loss: 0.234 / 0.660 versus baseline 0.234 / 0.661.
- Predicted delta logit mean/std: 0.098 / 0.006.
- Predicted delta logit vs true delta logit: Pearson 0.039, Spearman 0.069.
- Predicted delta probability vs true ITE probability: Pearson -0.088, Spearman -0.130.

The model mostly learned a small positive offset on top of the nuisance outcome model, not modifier-driven heterogeneity.

## Attention readout

Role-attention evidence produced 444 attention rows; 435 had token summaries. Modifier-family hits:

- Histology: 113 rows, mean attention 0.092.
- EGFR/molecular status: 119 rows, mean attention 0.088.
- Brain metastases/staging: 47 rows, mean attention 0.083.
- Hemoglobin/labs: 52 rows, mean attention 0.055.
- NLR/labs: 17 rows, mean attention 0.052.

The strongest nonempty attention chunks were dominated by pathology and molecular testing:

- Immunophenotype and histology: TTF-1, Napsin A, CK7, p40, adenocarcinoma/squamous language.
- EGFR mutation and broader molecular testing: EGFR negative/positive/unknown, KRAS/MET/STK11 mentions.
- Imaging/staging chunks: MRI brain, intracranial metastases, lymphadenopathy, pleural disease.
- Some lab chunks: hemoglobin/CBC and neutrophil/lymphocyte mentions, but weaker and less frequent.

There were 5 patients with empty `clinical_text`; a few of the highest raw attention rows came from those degenerate empty chunks and should be ignored for interpretation.

## Files

- `htr_pair_uplift_summary.json`
- `htr_pair_uplift_outer_test_predictions.parquet`
- `htr_pair_uplift_pair_predictions.parquet`
- `htr_pair_uplift_attention.parquet`
- `htr_pair_uplift_attention_modifier_hits.csv`
- `htr_pair_uplift_top_attention_rows.csv`
