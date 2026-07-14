# Cohort-Level TF-IDF Effect-Modifier Contrast

## Motivation

The direct text effect models in this repository usually create a target for
each patient (for example, an R-learner pseudo-outcome) and then train a model
to predict that target from the patient's text.  With binary treatment and a
binary outcome, those patient-level targets are extremely noisy.  A flexible
model can spend most of its capacity fitting that noise even when the text
plainly contains the true modifier.

This one-off experiment asks a narrower question:

> Can the treatment/outcome evidence be pooled across the entire cohort first,
> producing one signed effect-modification score for every TF-IDF n-gram, and
> can a causal forest work better after its eligible `X` features are limited
> to the most stable n-grams?

The experiment deliberately does **not** fit a model to a patient-level
effect target during feature discovery.  It creates one dataset-level score
vector by aggregating an orthogonal moment over all patients in an outer
training fold.

## Cohort contrast

For each patient, the experiment starts from honest cross-fitted nuisance
predictions:

- `treatment_residual = observed_treatment - predicted_treatment_probability`
- `outcome_residual = observed_outcome - predicted_outcome`

It estimates the fold's constant residualized treatment effect and then forms
the patient contribution

```text
treatment_residual
    * (outcome_residual - constant_effect * treatment_residual)
```

The contribution is not treated as a regression target.  Instead, the sparse
TF-IDF matrix is transposed and multiplied by the vector of patient
contributions.  The result is one pooled cohort-level moment for every n-gram.
Each moment is divided by a heteroskedastic influence-function standard error,
giving an approximately z-scaled contrast score.  Large positive and negative
scores indicate n-grams aligned with above-average and below-average treatment
effects, respectively.

This is the nuisance-adjusted vocabulary-space analogue of the repository's
dataset-level embedding interaction contrasts.

## Stability and eligibility

Ranking thousands of n-grams once would select noise.  The primary ranking
therefore combines the full-fold standardized score with:

- sign agreement across the individual nuisance-model views;
- sign agreement across stratified cohort subsamples;
- frequency of appearing in the top screening pool across those subsamples;
- agreement with a top-versus-bottom tail contrast;
- minimum document frequency and minimum treated/control support.

The ranking remains a screening statistic, not a claim that each selected
n-gram is itself a causal variable.

## Final forest

For every requested value of `top_k`, only those TF-IDF columns are passed as
the forest's heterogeneity features (`X`).  The final model is
`econml.grf.CausalForest` fit directly to the already residualized treatment
and outcome.  This is important: the repository's general `CausalForestHead`
currently refits nuisance models internally and would discard the performant
cross-fitted nuisances used to construct the contrast.

All vocabulary construction, nuisance use, contrast scoring, and feature
selection happen inside each outer training fold.  The outer test fold is used
only for evaluation.

## Current one-confounder/one-modifier run

The default command uses the completed stage-1 nuisance artifact from the
one-confounder/one-modifier multi-model run.  It builds an honest second-level
nuisance ensemble by fitting separate cross-fitted logistic stackers to the
individual BoW and HTR predictions.  The outer-test predictions come from
stackers fit only on the outer-training rows.  This was materially better
calibrated than simply averaging the nuisance probabilities.

```bash
~/thisenv/bin/python one_off_tfidf_cohort_contrast/run_experiment.py
```

Useful options include:

```bash
~/thisenv/bin/python one_off_tfidf_cohort_contrast/run_experiment.py \
  --top-k 25,50,100,200,400 \
  --primary-top-k 100 \
  --stability-repeats 30 \
  --forest-estimators 400
```

## Artifacts

The default output directory is
`one_off_tfidf_cohort_contrast/results_one_conf_one_mod/`.

- `run_config.json`: resolved arguments and input provenance.
- `contrast_feature_scores.parquet`: all eligible and ineligible n-gram scores
  in every outer fold.
- `selected_features.csv`: selected n-grams and diagnostics for every
  `top_k`/fold combination.
- `oof_predictions.parquet`: outer-fold predictions for every `top_k`.
- `fold_metrics.csv`: fold-level R-loss and oracle metrics.
- `aggregate_metrics.csv`: pooled outer-fold performance by `top_k`.
- `selection_stability.csv`: how often each n-gram was selected across outer
  folds.
- `ngram_relevance_summary.csv`: cohort-contrast relevance aggregated across
  outer folds, independent of a particular `top_k`.
- `summary.json`: concise primary-configuration summary.

The follow-up nuisance-calibration benchmark is described in
[`FRESH_BOW_NUISANCE_RESULTS.md`](FRESH_BOW_NUISANCE_RESULTS.md). Its script
trains new fold-local TF-IDF/logistic nuisance models and compares them with
both the original probability mean and an honestly recalibrated stack of the
earlier nuisance candidates.

## Per-topic LLM labeling

`batch_label_nmf_topics.py` creates one independent prompt per NMF topic. Each
prompt asks for both a broad topic label and the distinct normalized clinical
features represented by that topic's terms. It does not introduce causal
inference language. Responses are strict JSON and are checkpointed separately,
so rerunning the same command resumes incomplete topics.

Prepare prompts without contacting a server:

```bash
~/thisenv/bin/python one_off_tfidf_cohort_contrast/batch_label_nmf_topics.py \
  --prepare-only
```

Submit the batch to an OpenAI-compatible vLLM server:

```bash
~/thisenv/bin/python one_off_tfidf_cohort_contrast/batch_label_nmf_topics.py \
  --server-url http://camus.dfci.harvard.edu:8002/v1 \
  --concurrency 8
```

The topic-term and topic-summary inputs, clinical-task description, term count,
topic ranks, model, concurrency, and output directory can all be overridden by
command-line options.

## Canonical registry and bounded extraction

`build_canonical_feature_registry.py` harmonizes the per-topic candidate names
into a provenance-preserving registry. It distinguishes aliases, base
variables, derived variables, subfields, related-but-distinct variables,
review items, and drops. Model calls are checkpointed by stage and domain.

```bash
~/thisenv/bin/python \
  one_off_tfidf_cohort_contrast/build_canonical_feature_registry.py
```

`extract_canonical_features.py` reads that registry and creates domain-aware
request groups. It defaults to 10 variables per request and rejects any value
above 10. Without `--execute`, it only writes a plan:

```bash
~/thisenv/bin/python \
  one_off_tfidf_cohort_contrast/extract_canonical_features.py
```

To run extraction explicitly:

```bash
~/thisenv/bin/python \
  one_off_tfidf_cohort_contrast/extract_canonical_features.py \
  --variables-per-request 10 \
  --execute
```

Extraction progress is checkpointed per row and request group in SQLite.
Rerunning the same output directory resumes completed work. The current
registry and smoke-test result are described in
[`CANONICAL_REGISTRY_RESULTS.md`](CANONICAL_REGISTRY_RESULTS.md).

Oracle columns such as true ITE, age, and PD-L1 are used only after prediction
for evaluation.  They never participate in contrast construction or feature
selection.

## Limitations

- Marginal screening can miss a pure interaction whose component n-grams have
  no marginal orthogonal moment.
- Correlated synonyms can divide one concept's score among multiple n-grams.
- Selecting features and fitting the forest on the same outer-training fold is
  honest for outer-fold performance evaluation, but the forest's nominal
  confidence intervals do not account for adaptive feature selection.
- Good nuisance models reduce bias but do not remove the fundamental sampling
  noise of treatment-effect estimation.
