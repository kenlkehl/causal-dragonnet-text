# Contrast-screened NMF: five-confounder/five-modifier fold 1

## Method

This experiment used outer fold 1 only. It selected the top 10% of eligible
stability-adjusted cohort-contrast n-grams (2,996 of 29,951), mildly weighted
their TF-IDF columns by contrast score, and factorized the resulting
800-patient matrix into 100 nonnegative topics.

Three NMF fits used seeds 42, 43, and 44. Topics were aligned across seeds by
maximum loading cosine similarity, their patient scores were averaged, and the
same orthogonal cohort contrast was recalculated for each consensus topic.
Topic ranking used contrast magnitude plus seed, nuisance-source, and patient
subsample stability. The synthetic oracle nuisance probabilities were used
only after topic construction for diagnostics.

All 100 compact topic summaries were sent to the Gemma model on `camus:8002`.
This avoided an oracle-informed topic cutoff while still reducing roughly
3,000 phrases to 100 grouped representations. The agent was asked to return at
most 20 concepts.

## Modifier recovery

| Hidden modifier | Result | Representative topic evidence |
|---|---|---|
| Histology | Recovered | Topic rank 28: `histology.`, `squamous histology.`, `supports squamous` |
| EGFR status | Recovered | Topic ranks 10 and 50: `met amplification egfr`, `deletion egfr` |
| Brain metastases | Recovered | Topic rank 27: `brain metastases`, `surrounding edema`; related imaging topics |
| Baseline hemoglobin | Recovered broadly | Topic ranks 64, 75, and 79: distinct Hgb levels and anemia |
| Baseline NLR | Not recovered explicitly | A WBC topic ranked 33 and had weak NLR association, but no NLR phrase reached any topic's top 15 terms |

The agent therefore recovered evidence for four of five hidden modifier
concepts, compared with two of five when it received only the top 50 raw
n-grams. It merged histology and molecular alterations into one broad concept
and placed hemoglobin inside a general laboratory/hematologic concept rather
than identifying it as a modifier by itself.

## False positives and diagnostics

The agent also called functional/performance status a candidate modifier. ECOG
performance status is a confounder in this data-generating process. It further
promoted gastrointestinal/systemic symptoms as a candidate modifier and found
several plausible prognostic or proxy concepts such as pulmonary symptoms,
radiation history, and surgery history.

Learned- and oracle-nuisance topic contrasts were nevertheless very similar:

- learned/oracle z-score correlation: 0.981;
- sign agreement across all topics: 98%;
- sign agreement among the top 25 topics: 100%.

Thus, for this fold, the remaining false positives are not explained mainly by
the learned nuisance calibration. They can arise from finite-sample contrast
noise, correlated clinical proxies, and synthetic note-generation structure.
NMF materially improves concept coverage, but it does not by itself determine
which coherent concepts are genuine effect modifiers.
