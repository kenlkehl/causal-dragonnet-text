# Five-Confounder/Five-Modifier TF-IDF Matching/Uplift Report

Dataset:
`synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet`

## Nuisance And Matching

- Patients: 1000
- Treated: 521
- Control: 479
- Honest 5-fold propensity AUROC: 0.7922495281676878
- Honest 5-fold outcome AUROC: 0.7661005097029494
- Eligible cross-treatment pairs inside +/-0.05 propensity and +/-0.05 outcome calipers: 27726
- Maximum one-to-one matched pairs: 364

Known confounder hits in final TF-IDF nuisance coefficient tables:

| Confounder | Propensity top-table hit | Outcome top-table hit |
|---|---|---|
| age | age 71 | age 55 |
| sex | female | male |
| ecog_performance_status | ambulation | fatigued |
| creatinine_clearance | kidney | kidney disease |
| prior_platinum_therapy | cisplatin pemetrexed | platinum doublet |

## Pair Uplift

Pair-level observations: 364 matched treated/control pairs.

| Model | AUROC | Brier | Log loss |
|---|---:|---:|---:|
| Baseline untreated/control probability | 0.7284399224806202 | 0.22834492292518505 | 0.6485875593062461 |
| Ridge delta-probability uplift | 0.7300750968992248 | 0.22283380646420772 | 0.6374608957189735 |
| Offset-logit delta uplift | 0.7272589631782945 | 0.2266957697229453 | 0.6457152143812712 |

Honest uplift/true-effect association:

| Model delta | True pair delta Pearson | True pair delta Spearman | True treated ITE Pearson | True treated ITE Spearman |
|---|---:|---:|---:|---:|
| Ridge delta probability | 0.16787007460934597 | 0.20789770627877724 | 0.17810039788950988 | 0.20715473816345548 |
| Offset-logit implied delta probability | 0.054204961767857586 | 0.08783495458464326 | 0.07364088346876649 | 0.11684977538028596 |

Known effect-modifier hits in final pair-uplift coefficient tables:

| Effect modifier | Ridge top-table hit | Offset-logit top-table hit |
|---|---|---|
| histology_type | treated::large cell | treated::large cell |
| egfr_mutation_status | treated::deletion egfr | treated::deletion egfr |
| baseline_nlr | treated::neutrophils | treated::neutrophils |
| brain_metastases_status | treated::right cerebellar | treated::right cerebellar |
| baseline_hemoglobin | not found in top 250 per direction | not found in top 250 per direction |

## Interpretation

The nuisance models clearly recover text proxies for all five known confounders in
their top TF-IDF coefficients. The pair-uplift models recover recognizable text
features for four of the five known effect modifiers, but their honest uplift
predictions are still weak: the ridge model improves calibration/log-loss only
slightly and has low correlation with the true treatment effect. Hemoglobin did
not appear in the top uplift coefficient tables.
