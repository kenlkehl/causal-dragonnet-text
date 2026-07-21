# Project Brief and Continuation Plan: Discover Patient Features from Clinical Text

Date: 2026-07-18


## 2026-07-19 continuation checkpoint

The replacement discovery path now uses the required hierarchy instead of one
all-evidence prompt: each of the ten active Stage 1 architectures is interpreted,
consolidated, and coverage-audited separately; cross-architecture planning sees
ten compact dossiers; and raw evidence can return only through a bounded,
ID-addressed lookback. A second exact-spent hierarchy for review rounds after the
first gate is being integrated so newly visible evidence can add or revise
features without dumping the complete catalog into one prompt.

The historical projection omission for
`tfidf_semantic_retrieval_contrasts` is closed by an authenticated compatibility
view. It authenticates the exact provider or cache overlay object, exact request
rows/text/treatment/outcome, immutable cache bytes, provider/backend/helper code,
and a per-object before/after migration ledger. The provider and legacy grouping
helper remain byte-identical at SHA-256
`de11740a862c13d59d340e1dba26fb1202820dec4a0055c49819b7e01eccc1f1`
and `a46b801fc7a6f7dc66c42dc91a1bda9ddd93a79d871b9649aa89e25877ef633c`.

Both preserved real fold-1 caches were reauthenticated through the exact current
overlay with zero backend fits and the exact 480-spent/320-sealed schedule:

- one-confounder/one-modifier cache key
  `6d2123501feb4181fe2f72083b203e277d598bd3ef2408a0159e818c1a9a605a`,
  catalog SHA-256
  `24562e2b83fc3d9defbcaf4e3edbdf5f5748907d898ce5580b58b008befd86e4`,
  migration-ledger SHA-256
  `56f9a6f668081e01976ca92f46a87437cc90187d4c2441aa08f2d31af49f1e76`;
- five-confounder/five-modifier cache key
  `eaa08c3584581f4ce3bafac7e449ac3cb41e0ef8315c2006b93ad5dc1a0570b5`,
  catalog SHA-256
  `35b9fcb4ec32ac6fba11bf9592aa2c3a1fd004e3d27e19d8fcb16e7762f83aee`,
  migration-ledger SHA-256
  `10f50ede4514da1903c1fc7fa3e3be5b8a0dfbd0af223faba52e05c76970c227`.

Each cache reconstructed all ten active families, including exactly 19 semantic
retrieval contrasts. These fold-1 fits must be reused, not recomputed.

First-gate direct numerical materialization is now deferred. Zero-call
preparation freezes an authenticated `FirstGateMaterializationIntent` containing
the exact stable coordinate contract but performs no Stage 1 context fit, matrix
read, or gate-cache write. Only after exact batch approval and an immutable review
proposal freeze may the gate bank be realized; the resulting manifest, matrix
hashes, values, provider, and lineage must match the intent before gate evaluation.

Before restarting expensive preparation, finish the phased review-policy update:
round 1 may reuse the approved initial accepted-support evidence; rounds 2 and
later must require a refreshed exact accumulated-spent all-ten hierarchy, bounded
lookback, and pre-gate proposal freeze. Then run the consolidated regression suite.


## 1. The overall goal

The project is trying to develop an AI pipeline to learn and predict heterogeneous treatment effects (individual treatment effects, ITEs, aka conditional average treatment effects, CATEs) in oncology when most of the useful patient information is contained in clinical notes.

For each patient, the working dataset contains:

- clinical text;
- an observed treatment assignment;
- an observed outcome;
- and, in the synthetic benchmark only, hidden data-generating variables and the
  patient's true individual treatment effect.

The hidden synthetic variables and true treatment effects are called **oracle
information**. These exist so that a completed method can be evaluated. They must
never be used to discover features, write prompts, choose revisions, tune models,
or accept an adaptive change.

The intended method is:

1. Fit several different text models using only the patients currently allowed for
   training.
2. Use those models to surface many noisy clues about patient characteristics that
   may matter for treatment choice, outcome, or differences between treatment
   effects.
3. Have language-model agents determine which concrete patient characteristics are
   encoded in those noisy clues.
4. Define and extract those characteristics from complete clinical notes.
5. Use honest, observable validation data to find extraction failures, missing
   concepts, redundant variables, or mistaken interpretations, and revise them in
   a bounded loop.
6. Fit a causal forest using the extracted variables, with or without honest numerical outputs from the text
   models.
7. Produce one held-out treatment-effect prediction for every patient.
8. Only after the predictions are immutable, compare them with the synthetic true
   treatment effects and report oracle-feature recovery.


### Benchmark population, comparison, and estimand

The current synthetic benchmark represents 1,000 patients with advanced or
metastatic non-small-cell lung cancer. Its generating clinical question is a
comparison of vinorelbine with gemcitabine. The modeling interface contains a
binary `treatment_indicator` and a binary `outcome_indicator`. The archived
metadata does not give the binary outcome a more specific clinical endpoint or
window, so this brief does not invent one. Before applying the method to real data,
the population, treatment coding, and endpoint would need to be specified explicitly.

The repository calls its primary endpoint of interest an individual treatment effect, or ITE. More
precisely, `CausalForestDML` estimates a conditional average treatment effect given
the representation supplied to it. In this synthetic benchmark, the retrospective
column `true_ite_prob` records the difference between the two
potential binary-outcome probabilities for that row, based on the synthetic data generation process. That probability-scale target
is what the frozen forest predictions are compared with; it is not an observed
patient-level counterfactual outcome.

The causal interpretation relies on the usual assumptions: the recorded treatment
corresponds to the intervention being compared; the recovered adjustment variables
are sufficient for conditional exchangeability; both treatments have adequate
support over relevant patient strata; and the supplied note text is suitable by
construction. Note that this pipeline need not, and should not, try to enforce temporal ordering; assume the treatment and outcomes have not yet been given or occurred, such that any treatments or outcomes that are described are prior historical treatments, not the treatment/outcome we are trying to model. The benchmark metadata did not enforce positivity mechanically. The agents are not asked to police these
assumptions inside their feature-discovery or extraction prompts; in fact, feature discovery and extraction agents need not, and should not, know that they are part of a causal inference pipeline. The only agents that need to know this are those that evaluate sufficiency of currently extracted variables and iterate on them if inadequate for downstream causal inference use. Features do need to be assigned to confounder and/or effect modifier roles, but that assignment should be based on the stage 1 modeling objective that led to the noisy traces, which means it can/should be rules-based.

### The present research bottleneck

Stage 1 often contains useful signal, but it does not directly emit clean variable
names. It emits fragments: weighted words, short phrases, attention witnesses,
topic terms, retrieved semantic neighbors, residual contrasts, and numerical
scores. A clinically meaningful feature may be distributed over several weak clues
and several model families.

The central task for the discovery agents is therefore:

> Look across the supplied noisy signals and determine what specific, extractable
> patient characteristic they appear to have in common.

The agent is discovering variables. It is not estimating a treatment effect.

## 2. Terms used in this document

### Stage 1

Stage 1 is the collection of fold-local statistical and neural text models described
in Section 4. These models use observed treatment and outcome only from permitted
training patients.

Stage 1 produces two importantly different kinds of output:

1. **Concept-bearing evidence**: human-readable words, phrases, topic terms, and
   semantic witnesses that an agent can use to infer a patient characteristic.
2. **Direct numerical signals**: patient-level predictions or activations from the
   Stage 1 models. These can be passed directly to later validation and the causal
   forest, but an anonymous number cannot name or clinically ground a feature.

These two channels must not be confused. Concept-bearing evidence tells an agent
what a signal might mean. Direct numerical signals preserve predictive information
even when the explicit extraction step is imperfect. The direct numerical signals should not be encoded directly into prompts to agents tasked with feature discovery, but they could be useful traces for agents tasked with evaluating adequacy of currently extracted variables.

### Feature discovery

Feature discovery is the process of translating Stage 1 evidence into named patient
characteristics that could be read reliably from a note. A discovered feature is a
specific variable (eg age, sex, cancer_type, and so on), not merely a token, topic number, latent vector, or model score.

### Nuisance models and residual loss

In this project, a **nuisance model** is a model trained to predict treatment assignment
and/or expected outcome. Those predictions are used to remove the part of treatment and
outcome that the text already explains. The remaining treatment and outcome
residuals are then used in an **R-loss**, which asks whether a proposed feature or
model helps explain treatment-effect heterogeneity. These supporting quantities are
important, but they are not themselves the final treatment-effect estimate.

### TF-IDF

**Term frequency-inverse document frequency**, abbreviated TF-IDF, represents a
document with words and short phrases, giving more weight to terms that are common
in that document but not common in every document. It is used both for sparse
prediction and for making semantic retrieval results lexically interpretable.

### Extraction

Extraction is the later process of reading the value of a named feature from a
patient's note. The discovery agent and extraction agent have different jobs:

- discovery asks what feature the evidence describes;
- extraction asks what value the note reports for that already-defined feature.

### Causal roles

Roles are assigned based on the objective of the model that generated the stage I evidence traces. Generally, agents are tasked with idenfitying confounders and effect modifiers. 

- A **confounder** is a plausible cause of both treatment choice and outcome.
  Treatment or prediction alone is insufficient. Evidence connecting the same feature
  to treatment and outcome, plus sensible causal reasoning, can support this role.
- A **prognostic variable** helps predict outcome regardless of whether it predicts
  treatment. A prognostic variable can improve outcome adjustment without thereby
  becoming a confounder.
- An **instrument** is a variable that may predict treatment but does not also predict outcome.
- A **colider** is a variable that may be caused by both treatment and outcome. In this dataset, outcomes have not yet occurred, so colliders generally should not be identified.
- An **effect modifier** is a variable that change the relative effect of a treatment.
- An **extraction-support variable** helps define, distinguish, or reliably read a
  construct. It is not automatically a causal variable.

Roles may overlap. For example, a feature can be both a confounder and an effect
modifier. The statistical axis on which a clue appeared is evidence about the
feature; it is not automatically the feature's causal role.

### Outer and inner folds

An **outer fold** separates patients used to build the complete method from patients
used only for its final held-out prediction. Several kinds of **inner folds** are
used inside the outer training set. Section 5 distinguishes them because they serve
different purposes.

### Precommit

A **precommit** is an immutable manifest written before a model call or adaptive
decision. It records the inputs, split identities, prompt bytes, configuration, and
expected output location. It prevents a later result from being paired silently
with different evidence or instructions.

### Frozen and immutable artifacts

In this project, **frozen** or **immutable** means logically write-once and
content-addressed, not merely "saved to disk." A new run writes to a fresh path,
refuses overwrite, records the file's SHA-256 and all parent manifests, and requires
downstream readers to verify those hashes. Operating-system permissions alone are
not the integrity mechanism.

## 3. End-to-end design

The intended flow within one outer fold is:

```text
allowed outer-training notes + treatment + outcome
                    |
                    v
        multiple Stage 1 text models
          /                         \
         v                           v
human-readable evidence       row-level numerical signals
         |                           |
         v                           |
agents infer named patient           |
features from noisy clues            |
         |                           |
         v                           |
separate role review and             |
extraction-definition steps          |
         |                           |
         v                           |
extract values from notes            |
          \                         /
           v                       v
       observable adaptive review and ablations
                         |
                         v
       freeze the feature registry and all definitions
                         |
                         v
         fit a causal forest on outer-training data
                         |
                         v
    transform outer-held-out notes without using their labels
                         |
                         v
             immutable held-out ITE predictions
                         |
                         v
        retrospective oracle evaluation only afterward
```

This entire process is repeated independently for every outer fold. The method is
evaluated on the union of the held-out predictions, one prediction per patient. Note that prior to feature registry freezing, the intent is for agents to iterate on the feature library by doing preliminary modeling with extracted variables, and if nuisance and/or effect modification performance appears insufficient, go back to the original stage 1 evidence to consider how the library of variables should be expanded or contracted, and repeat.

## 4. Complete Stage 1 pipeline

No single Stage 1 family is expected to recover every useful feature. The families
use different modeling assumptions and expose different views of the same notes.
Their disagreement and their weaker tail evidence are potentially informative.

The following registry distinguishes predictive models, evidence adapters, and
integration layers. These identities should be made canonical in the continuation
work so that evidence-delivery tests have an exact set to check.

| Canonical identity | Kind | Supervised target | Agent-visible evidence | Row-level numerical output |
| --- | --- | --- | --- | --- |
| `bow_nuisance` | Predictive model family | Treatment and outcome separately | Signed exact terms and cross-view recurrence | Cross-fitted treatment and outcome predictions |
| `bow_r_loss` | Predictive model family | Residual-effect objectives | Signed residual-effect words and n-grams | Pseudo-outcome and direct R-loss predictions |
| `htr_neural` | Predictive model family | Treatment, outcome, and residual-effect objectives | Short attention phrases or spans | HTR nuisance and residual-effect predictions |
| `matched_pair_uplift` | Predictive model family using sparse and HTR variants | Treated outcome relative to matched controls | Sparse pair terms and HTR pair-attention phrases | Pair delta-logit and treated-outcome predictions |
| `embedding_whole_cohort` | Frozen-representation contrast model | Observable cohort contrasts | Semantic retrieval evidence, lexically rendered below | Patient projections on whole-cohort directions |
| `embedding_clustered` | Frozen-representation contrast model | Local observable contrasts | Cluster-local semantic retrieval evidence | Patient projections on clustered directions |
| `tfidf_semantic_retrieval_contrasts` | Evidence adapter; production identity still to add | Positive versus negative semantic-retrieval groups | Signed exact terms and bounded witnesses | No independent signal; shares the underlying embedding projection |
| `tfidf_topics` | Predictive model plus topic adapter | Treatment, outcome, and orthogonalized effect scores | Complete treatment, outcome, and effect topic term lists | Topic activations and nuisance-stack predictions |
| `tfidf_orphan_ngrams` | Deterministic residual-evidence adapter | Retained effect scores from the topic route | Exact high-scoring terms omitted by topics | Selected label-free TF-IDF term values |
| `neural_query_moments` | Learned-query model plus evidence adapter | Treatment, outcome, and orthogonalized residual moments | Semantic witnesses, lexical contrasts, and non-grounding aggregate strengths | Query activations and fixed bank summaries |
| `direct_upstream_numerical` | Authenticated integration layer; production identity still to add | None beyond its source models | Non-grounding family-level strength and stability only | Fold-honest row values collected from all source models |
| `sparse_query_moments` | Inactive fallback | Fixed sparse-query activation | Not permitted as neural-query evidence in this benchmark | None in the required benchmark path | --sparse query moments should be eliminated from the repo at this time--

HTR pair attention belongs operationally to `matched_pair_uplift`; it may also carry
HTR provenance, but it should be delivered once rather than duplicated. Likewise,
TF-IDF semantic-retrieval records retain joint provenance to their underlying whole
or clustered embedding direction while remaining a separate, first-class lexical
discovery view.

### 4.1 Sparse bag-of-words models for treatment and outcome

These models convert note text into TF-IDF-weighted word and phrase features. The
default grid contains six views:

1. a linear unigram model;
2. a linear one-to-two-word model;
3. a linear one-to-three-word model;
4. a linear two-to-four-word model with a higher minimum document frequency;
5. an ExtraTrees one-to-three-word model;
6. a random-forest one-to-two-word model.

Each view separately predicts:

- treatment assignment, producing an estimate of the probability of treatment;
- outcome, producing an estimate of expected outcome.

The fits are cross-fitted, so every outer-training patient's prediction comes from
a model that did not train on that patient.

Concept-bearing evidence includes, for every view:

- terms with strong positive or negative treatment association;
- terms with strong positive or negative outcome association;
- terms that appear on both axes, with both scores retained;
- repeated phrases that recur across multiple sparse views.

Treatment-associated terms are evidence about treatment assignment, not automatic
confounders. Outcome-associated terms are prognostic evidence. A common feature
supported on both axes may later be considered as a confounder, but only after the
feature itself is identified and causal plausibility is considered.

Direct numerical outputs include the cross-fitted treatment and outcome
predictions. These can contribute to nuisance adjustment and to later diagnostic
comparisons.

Primary implementation:
`oci/inference/multi_model_forest_stage1.py` and the default view definitions in
`oci/config.py`.

### 4.2 Sparse R-loss and residual-effect models

The nuisance models above provide, for each training patient:

- a treatment residual: observed treatment minus predicted treatment probability;
- an outcome residual: observed outcome minus predicted outcome.

Stage 1 then asks which text features help explain the remaining association
between those residuals. It does this in two related ways:

- a pseudo-outcome regression based on outcome residual divided by treatment
  residual;
- a direct weighted R-loss regression, which gives more weight to rows with more
  informative treatment residuals.

The ratio can become unstable when the treatment residual is near zero. The direct
R-loss form avoids treating every such ratio as equally reliable: it fits the
residual relationship directly and weights rows by squared treatment residual. All
clipping and weighting rules must be fixed within the training context.

Each sparse text view is fitted to these objectives. The resulting positive and
negative weighted words and phrases are noisy clues about treatment-effect
heterogeneity. They are not measurements of a patient's true treatment effect and
must not be described to an agent as established effect modifiers.

Concept-bearing evidence is the signed set of exact words and n-grams from the
residual-effect fits, with view identity and recurrence retained.

Direct numerical outputs are the patient-level pseudo-outcome-model and weighted
R-loss predictions. These are candidate heterogeneity inputs to the final forest
after fold-honest calibration and validation.

Primary implementation: `oci/inference/multi_model_forest_stage1.py`.

### 4.3 HTR neural text models

HTR here means the repository's hierarchical transformer text representation. A
clinical note is divided into many overlapping word chunks. A small pretrained
sentence encoder maps each chunk to a representation, and a document-level
transformer and attention mechanism combine the chunks.

The HTR family includes:

- nuisance models for treatment and outcome;
- effect models using pseudo-outcome or direct R-loss objectives;
- an HTR model used in the matched-pair branch described below.

The intended HTR configuration keeps the sentence encoder **unfrozen and
trainable**. In the current code, `htr_freeze_sentence_encoder=false` makes all of
the encoder parameters trainable; the separate
`htr_trainable_sentence_encoder_layers` setting does not partially freeze it in
that case. Any future HTR refit must preserve this behavior.

Concept-bearing evidence consists of short high-attention chunks, phrases, or token
spans from treatment, outcome, residual-effect, and matched-pair models. Attention
is a salience clue, not proof that a phrase names a causal variable. Patient IDs and
full raw notes are not part of the selector evidence.

Direct numerical outputs include cross-fitted HTR treatment, outcome,
residual-effect, and matched-pair predictions.

Primary implementations:
`oci/models/hierarchical_transformer_extractor.py`,
`oci/inference/multi_model_forest_stage1.py`, and
`oci/inference/context_prediction_htr_provider.py`.

### 4.4 Matched-patient and uplift models

This branch tries to create a different view of heterogeneity. Within permitted
training data, treated patients are matched to comparable control patients using
cross-fitted treatment-probability and outcome predictions. Matching uses fixed
calipers—a maximum permitted distance between matching scores—and may use one or
more nearby controls according to configuration.

Two types of model are then fitted:

- sparse bag-of-words models over matched treated/control text;
- a trainable HTR pair model that represents the two notes and their difference.

They predict the treated patient's observed outcome relative to information from
the matched control patient or patients. The branch emits a delta-logit (a
difference on the model's log-odds scale) and a
treated-outcome probability. These are pair-based association signals, not observed
individual treatment effects.

Concept-bearing evidence includes:

- signed bag-of-words terms from delta-logit, absolute-pair, and probability-
  difference fits across all sparse views;
- short high-attention phrases from the HTR pair model.

Direct numerical outputs include the patient-level pair delta-logit,
treated-outcome probability, and related matched-pair summaries.

Primary implementation: `oci/inference/multi_model_pair_uplift.py`, orchestrated by
`oci/inference/multi_model_forest_stage1.py`.

### 4.5 Frozen semantic-embedding contrasts

This family begins with precomputed embeddings of note chunks. In the historical
benchmark the embedding model is `Qwen/Qwen3-Embedding-8B`. The embedding encoder is
frozen for this branch: Stage 1 learns contrast directions in its semantic space; it
does not fine-tune that encoder.

Whole-cohort contrasts compare semantic representations along several observable
training-data directions, including:

- treatment assignment;
- outcome;
- outcome within treatment arms;
- treatment-by-outcome cell differences;
- pseudo-outcome and orthogonal R-score directions;
- combined treatment/outcome directions;
- residualized treatment/outcome interaction directions.

Here, an orthogonal R-score is a treatment-residual by outcome-residual signal
constructed so that ordinary errors in the treatment and outcome support models
have less first-order influence. A residualized interaction similarly removes the
main treatment and outcome patterns before looking for semantic directions related
to their interaction.

Clustered contrasts repeat related comparisons within learned regions of the
semantic space and combine local contrast directions. This is intended to recover
signals that a single global contrast can average away.

For each direction, Stage 1 identifies note chunks near opposing ends of the
semantic contrast. The raw vectors themselves are not useful model-facing evidence
and must never be sent to the discovery agent.

Concept-bearing evidence comes from the words and phrases that distinguish the two
retrieval sides, as described in the next subsection. Direct numerical outputs are
each patient's projections onto the whole-cohort and clustered contrast directions.

The historical fold-1 catalog had nine whole-cohort directions and ten clustered
directions. Those counts describe that artifact, not an invariant design rule.

Primary implementation: `oci/inference/embedding_contrast_discovery.py`.

### 4.6 TF-IDF semantic-retrieval contrasts

This must be treated as a first-class discovery family, not hidden inside a generic
"embedding" summary.

For each frozen semantic-embedding direction, Stage 1 takes the text retrieved near
the positive and negative ends and computes which exact TF-IDF words or phrases
distinguish those groups. The result connects a distributed semantic direction to
lexical evidence an agent can understand.

Concept-bearing evidence includes:

- the identity and plain meaning of the semantic comparison;
- signed exact words and phrases differentiating the two retrieval groups;
- bounded semantic witnesses when safe to expose;
- stable evidence IDs and joint provenance back to the whole-cohort or clustered
  embedding contrast.

This is distinct from the topic model in the next subsection. It is semantic
retrieval followed by a lexical comparison, not an NMF topic decomposition.

The numerical patient-level signal currently comes from the underlying embedding
projection. A remaining implementation gap is to give the TF-IDF semantic-retrieval
facet equally explicit treatment in numerical-family provenance and ablations,
rather than inheriting an opaque generic embedding label.

### 4.7 Standalone TF-IDF treatment, outcome, and effect topics

This is a separate text-analysis route. It fits honest cross-fitted treatment and
outcome models over TF-IDF views and scores terms in three banks:

- a treatment bank for treatment-associated terms;
- an outcome bank for prognostic terms;
- an effect bank based on an orthogonalized treatment-residual by outcome-residual
  moment after removing the constant-effect component.

The constant-effect component is the single average residual relationship that
would apply if treatment effect did not vary. Removing it makes the effect bank
look for deviations from that average rather than simply rediscovering a common
treatment effect.

Eligible terms are stability-tested over inner splits. Weighted terms are then
grouped into consensus non-negative-matrix-factorization topics across multiple
random seeds. Non-negative matrix factorization groups terms that tend to receive
weight together; consensus across seeds retains groupings that are not an accident
of one initialization.

Concept-bearing evidence includes every retained topic's terms, loadings, signed
scores, bank identity, and stability information. The current benchmark was
configured for 100 topics in each bank, for 300 topics total, with up to 15 terms
per topic. A selector must not receive only the first dozen topics because of a
global prompt limit.

Direct numerical outputs include patient-level topic activations and cross-fitted
treatment/outcome stack predictions.

Primary implementations: `oci/inference/tfidf_topic_discovery.py` and
`oci/inference/tfidf_topic_stage1.py`.

### 4.8 TF-IDF residual or "orphan" n-grams

Some strong effect-bank terms are not adequately represented by any selected NMF
topic. A deterministic residual branch keeps these terms, filters obvious
administrative or identifier noise, removes genuine nested duplicates, and groups
lexically related survivors.

There is no additional predictive model here. This branch prevents lower-ranked
exact lexical evidence from disappearing merely because topic formation did not
represent it well.

Concept-bearing evidence includes the exact surviving words or phrases, signed
scores, treatment-arm support, stability, and cluster membership. Label-free
patient TF-IDF values for selected terms can become heterogeneity inputs after the
feature set is frozen.

Primary implementations: `oci/inference/tfidf_orphan_evidence_adapter.py` and
`oci/inference/tfidf_topic_score_selection.py`.

### 4.9 Learned neural semantic queries and aggregate moments

This family uses the same frozen chunk-embedding cache as the semantic contrast
branch. It learns small latent query vectors whose smooth retrieval activations are
associated with one of three observable training targets:

- treatment assignment;
- outcome;
- an orthogonalized residual moment intended to surface heterogeneity clues.

The query learner is not a language model and does not fine-tune the embedding
encoder. Queries are learned across inner subfolds, grouped by similar activation
patterns, summarized by stable representatives, and refitted within the permitted
training context.

The raw learned vectors must never be exposed to the discovery agent. A raw vector
has no interpretable clinical meaning and would consume context without grounding a
feature.

Concept-bearing evidence instead includes:

- semantic text witnesses retrieved by each query;
- exact TF-IDF words or phrases contrasting foreground and background retrievals;
- the query's bank identity;
- understandable aggregate summaries such as signed strength, recurrence across
  subfolds, member count, and stability.

These summaries are sometimes called **query moments**. They describe how a query
behaved across allowed patients; they are not true treatment effects.


Direct numerical outputs include patient-level query activations and fixed
summaries of the treatment, outcome, and effect query banks. Treatment-bank outputs
can contribute to nuisance inputs, outcome-bank outputs to prognostic adjustment,
and effect-bank outputs to heterogeneity inputs, subject to honest validation.

The historical configuration learned five queries per bank, fifteen total. Its
cached handoff contained 270 semantic witnesses and 30 aggregate moments, with no
raw vectors.

In ablations, remove or retain the row-level query activation bank independently of
the explicit feature inferred from its witnesses. This distinguishes signal carried
directly by the neural query from signal recovered through named-feature extraction.

Primary implementations include:
`oci/inference/neural_query_discovery_runtime.py`,
`oci/inference/neural_query_agentic_forest.py`,
`oci/inference/neural_cohort_witness.py`,
`oci/inference/neural_query_context_backend.py`, and
`oci/inference/query_moment_evidence_adapter.py`.

There is infrastructure for a fixed sparse-query fallback when learned neural
queries are unavailable. That fallback is not an active Stage 1 family for this
benchmark and must not be allowed to masquerade as learned neural-query evidence. In fact, eliminate it from the repo.

### 4.10 Direct upstream numerical signals

This is an integration layer, not another concept-discovery model.

It collects authenticated row-level outputs from the Stage 1 families, including:

- sparse treatment and outcome predictions;
- sparse R-loss predictions;
- HTR nuisance and residual-effect predictions;
- matched-pair/uplift predictions;
- whole-cohort and clustered embedding projections;
- TF-IDF topic and selected residual-term values;
- neural-query activations and moments.

For an outer-training patient, every supervised value must be inner-fold
out-of-fold. For an outer-held-out patient, the value must come from a model fitted
only on outer-training patients.

The language model may receive only aggregate, family-level descriptions such as
dispersion, nonzero rate, association with the appropriate observable target,
direction, and stability across inner folds. Those summaries can tell the agent
that a family deserves attention. They cannot name or ground a patient feature, so
any proposed feature must cite separate lexical or semantic evidence from the same
family.

The causal forest may receive the corresponding row-level values directly as input features. This
preserves useful Stage 1 information when a named feature is extracted imperfectly
or when the language model has not fully aligned the evidence to a clinical
construct.

Primary implementations include:
`oci/inference/fold_honest_signal_fusion.py`,
`oci/inference/final_context_fit_upstream_bank.py`, and the context-fit upstream
provider modules.

## 5. Honest outer-fold and inner-fold strategy

The complete adaptive procedure must be evaluated as a method. It is not honest to
discover features on the whole dataset and then cross-validate only the final
forest.

### 5.1 Outer folds protect the final prediction

For each outer fold:

1. Divide the data into outer training and outer heldout patients.
2. Permit the entire Stage 1, agent discovery, extraction-definition, adaptive
   review, and causal-forest fitting process to use only outer-training information.
3. After every choice is frozen, apply the trained transforms and forest to the
   outer-heldout notes without using their treatment or outcome.
4. Atomically create the held-out predictions at a fresh write-once path, hash the
   bytes, bind the hash into the fold manifest, and reject later overwrite or hash
   mismatch.

The historical synthetic benchmark has 1,000 patients and five outer folds, so a
typical fold has 800 outer-training and 200 outer-heldout patients. These numbers
describe the benchmark rather than a general requirement.

### 5.2 Stage 1 inner cross-fitting

Inside an outer training set, each supervised Stage 1 family uses inner
cross-fitting:

1. Hold out one inner subset.
2. Fit the Stage 1 model on the complementary inner-training patients.
3. Predict the held-out inner patients.
4. Repeat until every outer-training patient has an out-of-fold prediction.

This produces honest row-level Stage 1 values and lets the pipeline measure whether
words, topics, queries, or directions recur across fits. Different Stage 1 families
currently use different numbers of inner contexts; they need not share one model
object or one fold count, but their row and fit lineage must be explicit.

Once the fold's feature choices are frozen, a separate fit on all outer-training
patients may transform the outer-heldout notes. It still may not use outer-heldout
treatment or outcome.

### 5.3 Sequential one-use validation during adaptive review

The post-extraction review loop needs a stricter boundary because each agent-guided
revision is adaptive.

Before agent calls, partition the outer-training set into:

- an initial development set whose text, treatment, and outcome may drive Stage 1,
  extraction diagnostics, and revision proposals;
- one fresh validation partition for each allowed review round.

The code has called development-visible rows **spent** and not-yet-used validation
rows **sealed**. These are bookkeeping terms for data reuse, not scientific
concepts and not a user preference.

For each review round:

1. Build the agent's proposal using only information already in the development
   set.
2. Run all schema, citation, grounding, and extraction-validity checks before
   touching the fresh validation partition.
3. Freeze the proposed change.
4. Have a deterministic evaluator compare the current and proposed feature sets
   once on the fresh validation partition using observed treatment and outcome.
5. Consume that partition whether the proposal is accepted or rejected. It may
   contribute sanitized aggregate feedback to the next round, but it can never
   serve as untouched validation again.
6. If another round is allowed, add the consumed rows to the development set and
   use a new fresh partition.

The current default implementation creates three initial partitions plus one fresh
partition per review round. In the historical 800-row outer training fold with two
review rounds, this happened to produce 480 initially spent rows and two 160-row
gates, described previously as 480 spent and 320 sealed. That ratio is not a design
requirement and should not be imposed on other datasets.

The three initial partitions have no three different scientific jobs; they are
pooled to form the initial development set. For the historical two-round schedule,
the exact lineage is:

| Point in the run | Rows allowed to train supervised Stage 1 and diagnostics | Rows allowed to form the proposal | Fresh one-use gate | What happens next |
| --- | --- | --- | --- | --- |
| Before round 1 | Partitions 1-3, 480 rows | The same 480 rows, through sanitized evidence and diagnostics | Partition 4, 160 rows | Freeze proposal 1; apply development-trained transforms to gate 1; evaluate once; mark gate 1 spent |
| Before round 2 | Partitions 1-4, 640 rows, using a new exact-scope cache or refit | The same 640 accumulated rows | Partition 5, 160 rows | Freeze proposal 2; apply accumulated-development transforms to gate 2; evaluate once; mark gate 2 spent |
| After review | No further feature choice is permitted | None | None | Freeze the registry; then use all 800 outer-training rows for the final cross-fitted bank and full outer-training forest fit |

The fresh gate's text can be transformed after a proposal is frozen, but its
treatment and outcome are available only to the deterministic evaluator. A
supervised producer for the current round cannot train on that gate.

If review stops early because the agent proposes no change or convergence is
declared, unused gates remain untouched during feature choice. After the registry is
frozen, they may join the rest of the outer-training set for final cross-fitting and
forest fitting. Their labels must never be used to reopen feature discovery.

### 5.4 Nested folds for observable diagnostics

Within the currently available development rows, diagnostics themselves must also
be out of sample. Treatment and outcome nuisance predictions are cross-fitted.
R-loss diagnostics use effect targets whose nuisance predictions were generated
without fitting on the evaluated row. This additional nesting is sometimes called
"inner-inner" fitting.

Observable diagnostics include:

- extraction coverage and missingness;
- whether continuous values and categories are valid and plausible;
- stability across development folds;
- cross-fitted treatment and outcome prediction;
- cross-fitted R-loss;
- preservation of useful Stage 1 source signals;
- preservation of neural-query moments and direct numerical inputs;
- redundancy among extracted variables;
- delete-one-variable and delete-one-source-family ablations;
- a fixed complexity penalty for added definitions or encoded columns.

The agent may propose dropping, merging, reinterpreting, replacing, or revising a
feature. Only changed definitions should be re-extracted. A deterministic gate—not
the language model—decides acceptance using the predeclared observable metrics.
Oracle variables and true treatment effects are unavailable to both.

### 5.5 Stage 1 caching and when recomputation is necessary

Changing a prompt does **not** by itself require refitting all Stage 1 models. Cached
Stage 1 evidence should be reused whenever it is authenticated to:

- the exact outer fold;
- the exact currently permitted development-row set;
- the exact not-yet-permitted row set;
- the ordered text and observed inputs used by the producer;
- the model code, configuration, and artifact identity;
- the cross-fitting lineage of every row-level value.

A full-outer-training Stage 1 artifact is not valid for the initial adaptive
selector if its supervised fits used rows reserved for future review gates. An
exact 480-row historical spent cache is valid for that 480-row development context.
If a later round adds a gate to development, the pipeline needs a cache or refit for
that new exact accumulated scope.

Frozen, label-free chunk embeddings can be reused broadly. Supervised BOW, HTR,
query directions, and other label-dependent fits cannot be borrowed from a context
that included a still-sealed gate. On a cache miss, recompute only the missing exact
scope and family rather than rebuilding everything globally.

A redesigned model-facing prompt requires a new prompt precommit, selector cache,
and output directory because the rendered messages changed. It can still consume
the same authenticated Stage 1 cache.

### 5.6 Final inner cross-fitting and causal-forest fit

After bounded review converges:

1. Freeze the feature registry, role assignments, and extraction definitions.
2. Complete extraction over the remaining outer-training notes and the
   outer-heldout notes. The outer-heldout transformation is label-free.
3. Fit structured encoders on outer training only.
4. Build a complete final numerical bank for outer training using a separate set of
   meta-inner fits, so every outer-training row has an out-of-fold upstream value.
5. Fit the corresponding Stage 1 transforms on all outer training and apply them to
   outer heldout.
6. Fit `CausalForestDML` on the full outer-training feature bank.
7. Predict the outer-heldout treatment effects once and freeze them.

"Meta-inner" only distinguishes these final-bank folds from Stage 1 discovery folds
and sequential review gates. Their sole purpose is to build the final forest's
outer-training input matrix without giving any row a supervised value from a model
that trained on that row.

Confounders and prognostic variables can be routed to adjustment or nuisance inputs.
Effect modifiers can be routed to the forest's heterogeneity inputs. Overlapping
roles can route a feature to both appropriate places. Direct numerical Stage 1
signals and neural-query moments may also enter their predeclared forest inputs.

There is no non-forest fallback.

### 5.7 Oracle evaluation happens last

Only after every relevant prompt, feature choice, extraction definition, trained
model, and held-out prediction is frozen may a separate evaluator read the
synthetic true treatment effects.

The retrospective evaluator should authenticate the frozen prediction bytes and
then report metrics such as:

- correlation between predicted and true individual treatment effect;
- mean absolute and root-mean-squared error;
- recovery of hidden benchmark features;
- fold-specific and aggregate results.

Feature recovery also needs a predeclared post-hoc matching procedure. Before any
oracle file is opened, freeze the matching code and, if a semantic judge is used,
its prompt and model identity. After prediction freeze, load the frozen candidate
registry and hidden benchmark definitions into a separate evaluator that cannot
feed results back to discovery. Report at least:

- strict normalized-name matches;
- a semantic match ledger with candidate, hidden definition, decision, and reason;
- precision and recall under both the strict and semantic definitions;
- every ambiguous or disputed match rather than silently tuning aliases after
  seeing the oracle list.

A semantic matcher should use two independent blinded decisions or another
predeclared agreement rule. Its outputs and exact inputs must be hashed. This keeps
the retrospective feature-recovery score informative without turning subjective
post-hoc naming into another adaptive optimization loop.

These metrics describe the completed method. They must never decide which prompt,
feature, review action, or model is accepted.

## 6. What the language-model agents need to do

The selector should no longer be conceived as one agent reading one globally
truncated object and immediately returning a final causal feature set. The evidence
is too large and too heterogeneous for that design.

### 6.1 First task: interpret what the noisy evidence encodes

Run complementary discovery passes over losslessly chunked evidence from each
family. Each pass should be told in ordinary language:

- how that family produced the supplied clues;
- what a positive, negative, or high score means;
- that the clues are uncertain discovery evidence rather than facts;
- that other passes will inspect other chunks;
- that it must inspect every supplied item and build a broad inventory before
  ranking;
- that it must identify the specific patient characteristic shared by related
  words, phrases, topics, or semantic witnesses;
- that every proposed characteristic must cite the supplied evidence IDs that
  actually contain its lexical or semantic grounding.

The first-pass output should be a role-independent concept record containing:

- a plain feature name;
- a one-sentence description of what would be measured in a patient;
- supporting evidence IDs;
- the evidence families represented;
- whether the supplied evidence suggests a numerical, categorical, or ambiguous
  representation, explicitly marked as a hypothesis for the later extraction-
  definition step to confirm;
- unresolved ambiguity or competing interpretations.

Do not require `confounder` or `effect_modifier` in this response. That requirement
made purely prognostic variables impossible to represent and distracted the model
from the actual discovery problem.

### 6.2 Consolidate concepts without erasing provenance

After every evidence chunk has been inspected, a separate consolidation step should
merge only genuine spelling, abbreviation, or formatting aliases. It must preserve:

- every supporting evidence ID;
- every source family;
- weaker and lower-ranked support;
- disagreement among evidence families;
- distinct specific variables that happen to sit under one broad clinical theme.

Prompt length should be controlled through complementary chunks and staged union,
not by silently taking one global top-k. The merged ledger is an inventory of named,
grounded patient characteristics, not yet the final model input list.

### 6.3 Use two independent critics for two different failure modes

The first is an **evidence-coverage critic**. It compares the complete supplied
evidence inventory with the merged concept ledger and asks:

- Which evidence items have no candidate representing them?
- Was a specific variable hidden by an overly broad alias merge?
- Did family disagreement cause a useful interpretation to disappear?
- Did lower-ranked evidence vanish through truncation?

The second is a **rejection critic**. It receives the explicitly rejected candidate
ledger together with each candidate's complete backing evidence and rejection
reason. It asks whether a candidate was rejected because evidence was missing from
a later prompt, two distinct variables were mistaken for duplicates, or the role or
extraction question was prematurely confused with concept discovery.

Do not combine these objectives in one prompt. Each critic may propose a bounded
addition, split, or reconsideration, but it must cite existing evidence. Repeating
the same compact prompt with random sampling is not a substitute for complementary
evidence coverage.

### 6.4 Assign roles only after features are grounded

For each named candidate, build a simple table of its actual backing evidence:

- evidence related to treatment assignment;
- evidence related to outcome;
- residual, pair-based, or other heterogeneity clues;
- extraction-specific evidence;
- supporting evidence IDs and families.

A later role-review agent can then apply the definitions in Section 2. It may use
general scientific knowledge to judge the plausible role of an already-grounded
feature. It may not use outside knowledge to invent a new benchmark feature,
unsupported alias, category, unit, patient fact, or treatment-effect direction.

Closed role endorsements must be passed to the final selection step. The current
scratch design sometimes used role responses only to reorder candidates and then
discarded the endorsements; that is not sufficient.

### 6.5 Define extraction in a separate step

For a retained feature, an extraction-definition agent should specify:

- exactly what value to read;
- continuous, categorical, or other supported representation;
- supported units or categories;
- evidence-grounded aliases;
- how to distinguish it from nearby concepts;
- what to return when the note is absent or ambiguous.

The extractor then applies that definition to the note. The extractor's job is to
read the requested variable, not perform feature discovery or causal reasoning.

Do not include temporal-boundary instructions or temporal-policy language in the
selector, reviewer, or extractor prompts. The notes supplied to this pipeline are
appropriate by construction. Keep extractor reasoning disabled.

### 6.6 Review extraction with observable feedback

The post-extraction review agent sees only sanitized aggregate diagnostics from
already-consumed development data and the current grounded evidence. It can propose
a bounded revision, which must pass the one-use deterministic gate in Section 5.

Post-extraction adaptive review must remain enabled. Its purpose is to repair the
major failure mode already observed: Stage 1 may point toward a useful construct,
but the selected variable name, categories, or extraction definition may fail to
capture it reliably.

## 7. Model-facing prompt design

Every prompt to agents developed for this workflow must be understandable to an open-weight model that knows nothing
about this repository.

Use a short, stable system message and a concise pass-specific user message. Keep
JSON as the evidence container and output format.

Prompt rules:

- State the agent goal plainly.
- Explain only the evidence families present in that pass.
- Define every output field.
- Say `Return exactly this JSON shape` and require JSON only.
- Use a complete, content-neutral structural example if an example is necessary.
- Do not include dataset-specific clinical examples that could seed benchmark
  answers.
- Do not use unexplained project phrases such as `response contract`, `ordinary
  response schema`, `producer node`, `nuisance axis`, or `sealed context`.
- Keep machine bookkeeping out of model-visible messages.
- Continue exposing stable evidence IDs because citations require them.
- Keep split fingerprints, hashes, schema versions, cache keys, and producer IDs in
  the authenticated internal envelope and audit trail.
- Hash both the complete internal request and the exact rendered system/user message
  sequence, including any repair attempt.
- Check every cumulative prompt against the unchanged context-size guard before an
  HTTP request.
- Enable selector/discovery reasoning with exactly 5,000 reasoning tokens.
- Disable extractor reasoning.

Direct numerical summaries must be visibly marked as non-grounding. Neural-query
evidence must be described as semantic witnesses and aggregate behavior, never as a
raw vector.

## 8. Work completed under this goal

### 8.1 Earlier components

Before the current all-evidence runner, the repository already had an agentic
feature-search path and a multi-model text path. Those established several core
ideas:

- use outer folds to protect final predictions;
- let inner observable metrics evaluate proposed explicit variables;
- extract named variables before fitting a structured causal model;
- use synthetic truth only for retrospective evaluation.

Stage 1 was then expanded progressively to include whole-cohort and clustered
embedding contrasts, multi-view sparse models, HTR evidence, matched-patient
uplift, TF-IDF topics, residual/orphan terms, and learned neural semantic queries.

### 8.2 The committed all-evidence integration

The current main commit is:

`d00e6a1dc8a7bb11294b3870a8e3c19e7979866b`  
`Integrate fold-honest all-evidence causal inference`

This is a large integration commit: 99 files and roughly 76,900 inserted lines. It
added or connected:

- the all-evidence request validator, command-line interface, and outer-fold runner;
- Stage 1 evidence sanitization, split sealing, provenance, and cache overlays;
- staged candidate proposal and union machinery;
- extraction grounding and coverage diagnostics;
- an oracle-free post-extraction adaptive review loop;
- nested cross-fitted treatment, outcome, and R-loss diagnostics;
- authenticated direct numerical Stage 1 banks;
- HTR context refitting with an unfrozen sentence encoder;
- TF-IDF topic and orphan-evidence adapters;
- neural-query evidence and moment adapters;
- final outer-honest upstream banks;
- a final `CausalForestDML` adapter;
- extensive tests for honesty, provenance, schemas, caching, and final prediction
  freezing.

This commit is a substantial engineering foundation. It is not a finished solution
to the feature-discovery problem. Some of its model-facing schemas and prompts
encode assumptions that the later audit found to be wrong.

### 8.3 Completed earlier benchmark result

The only completed all-fold oracle result found for the integrated family of work is
the older `benchmark_five_contractrag_v5` run. Its overall Pearson correlation
between frozen predictions and synthetic true individual treatment effect was:

`0.1898545804440651`

The five fold correlations were approximately:

- 0.2187;
- 0.0714;
- 0.1510;
- 0.2113;
- 0.2444.

This is the earlier result described as about 0.19. It is not a result from the
later v24 experiment.

The result can be authenticated at:

`artifacts/all_evidence_fusion/benchmark_five_contractrag_v5/posthoc_oracle_evaluation/posthoc_oracle_metrics.json`

File SHA-256:

`0c702cb7d1f7c3f92fb0ccd1d60be4f77209b18f169c81f744975ace9b2e6e8d`

The metrics file records the frozen prediction SHA-256 as
`079b58fa9af58b38b839cad4510a515415a2a24136d130de75ff3edc8912232c`.

Subsequent analysis indicated that a major loss of signal occurred between Stage 1
and final modeling: the pipeline did not always translate evidence into the right
clinical construct, and some selected constructs were extracted poorly.

### 8.4 Historical v24 experiment and evidence audit

`v24` is merely the directory/version label of a later benchmark configuration. It
is not a model family or scientific concept. Its control directory is:

`artifacts/all_evidence_fusion/benchmark_five_agentic_loop_v24_exact_lexical_thisenv_r2`

The run progressed only partway through the first outer fold:

- its staged proposal union contained 25 unique candidates;
- its frozen selector response contained 18 candidates;
- adaptive review round 1 proposed changes that the observable gate rejected;
- round 2 has a request and model response but no completed round audit;
- no immutable final outer-fold predictions were produced;
- consequently, there is no v24 oracle ITE correlation.

A retrospective audit of its fold-1 evidence found:

- all hidden benchmark concepts appeared somewhere in the raw Stage 1 evidence from
  the allowed development rows;
- the compact selector prompt completely lost several concepts and weakened others;
- some relatively strong evidence survived the prompt but the selector overlooked
  it;
- the full 800-row HTR artifact had relevant witnesses, while the compact 480-row
  development-only HTR blocks had none;
- extraction quality and alignment between a selected name and the underlying
  construct remained a second major loss point.

Specific hidden benchmark names were used only in this retrospective audit. They
must not be copied into future model-facing prompts or acceptance criteria.

The conclusion was that Stage-1-to-agent evidence delivery and agent comprehension
should be treated as the primary bottleneck before further extractor tuning.

### 8.5 Cached evidence-integration experiment

A later experiment used the authenticated v24 Stage 1 cache rather than recomputing
the expensive models. It constructed a fold-1 catalog with 441 concept-bearing
evidence atoms:

- 84 sparse bag-of-words groups;
- 18 HTR phrases;
- 19 semantic embedding contrasts;
- 300 TF-IDF topics;
- 5 TF-IDF orphan clusters;
- 15 neural-query entries.

The catalog also restored explicit first-class provenance for the 19 TF-IDF
semantic-retrieval contrasts and registered 316 direct numerical context signals.
It contained no raw dense vectors, raw patient excerpts, row IDs, sealed-row values,
oracle fields, or sparse-query fallback.

The intended comparison included:

1. the frozen current compact prompt;
2. larger stratified deterministic compaction;
3. hierarchical per-family prompting;
4. complementary chunked proposals followed by union;
5. the hierarchical design plus an independent critic.

Only 22 remote responses were obtained: 21 family/direct-numerical calls and one
larger global call. One response needed JSON repair. The comparison did not finish.
A later role-selection request rendered to 462,741 bytes and was stopped by the
unchanged context guard before any HTTP call. That guard used a 262,144-token model
context, a 25,000-token output reserve, an 8,192-token safety reserve, and a hard
220,000-byte rendered-prompt cap. No alternative arm reached completed discovery,
extraction, inner-fold evaluation, or oracle evaluation.

### 8.6 What the prompt and transport audit found

The audit found several mismatches between the implementation and the scientific
intent:

- Discovery asked for causal roles before plainly asking what patient feature the
  evidence described. Discovery agent should not even ask for causal roles at all; that should be handled in a rules based way, since stage 1 modeling objectives should allow assignment of roles (confounder or effect modifier) by definition.
- Pure prognostic variables were not representable because the output allowed only
  `confounder` or `effect_modifier`.
- Treatment-prediction evidence was sometimes treated mechanically as confounder
  evidence.
- Candidate summaries reached later role/meta prompts without their actual backing
  Stage 1 witnesses.
- Role-review endorsements did not reach the final selector as closed evidence.
- Some HTR witnesses stored under a `token` field disappeared during prompt
  projection.
- Arbitrary sampling of directions, aliases, and numerical summaries could erase
  evidence because list order was not a meaningful ranking.
- Critic instructions combined two different tasks: finding unrepresented evidence
  and reconsidering rejected candidates.
- Dense internal schema language and bookkeeping made the scientific task hard for
  an open-weight model to understand.
- Temporal-policy language remained in selector, reviewer, and extractor prompts
  despite not being part of the intended task.
- Exact message-sequence hashing, repair-prompt context checks, and cache provenance
  were only partially implemented in the scratch work.

These findings motivated the conceptual reset in Sections 6 and 7.

### 8.7 Uncommitted and scratch work


Recently committed main-tree changes include:

- raising the new selector reasoning budget from 4,096 to exactly 5,000;
- requiring post-extraction adaptive review for the v24 route;
- requiring final upstream, neural-query, and source-signal inputs;
- disabling sparse-query substitution for that benchmark;
- recording that no non-forest final fallback is permitted;
- related CLI and test changes.

These changes have not received a clean full-suite test run after all later edits.
The untracked probe/submission scripts are older experiments and do not satisfy the
current selector-reasoning requirement.

An incomplete copied tree exists at:

`/tmp/causal_stage1_ledger_20260717`

It contains prototypes for lossless evidence catalogs, per-family chunk plans, a
candidate ledger, hierarchical family/role/meta/critic passes, machine-field
projection, direct numerical sidecars, prompt hashing, cache auditing, and an
offline review packet. It is not a git repository and is not production code. It
compiled at the stopping point, but the latest focused record still included 81
passing and 8 failing tests, and a full clean suite was never run.

Known scratch defects include the role-first discovery schema, dense prompt
language, lost HTR token witnesses, dropped role endorsements, lossy candidate
sampling, conflated critic tasks, and incomplete transport/cache provenance.
Salvage small tested components only; do not copy the scratch tree wholesale.
Because this tree and the cached comparison directories are under `/tmp`, they are
not durable records. A fresh agent should verify that they still exist, record
hashes for any component it proposes to salvage, and move only reviewed material
into a new repository change or authenticated artifact directory.

### 8.8 Natural stopping point

No project language-model process, Camus selector request, or project GPU job is
running at this handoff. No new remote comparison has been launched. No commit has
been made after `d00e6a1`.

## 9. Remaining work: execution plan for a fresh agent

### Phase 0: Re-establish the baseline without changing it

1. Read this brief completely.
2. Inspect `git status --short`, commit `d00e6a1`, the dirty diffs, and all relevant
   artifact directories. Do not restrict the search to folders named `one_off`;
   recent iterations are stored in several layouts.
3. Do not reset or overwrite unrelated user changes.
4. Verify that no stale local or remote job from this project is running.
5. Authenticate the historical v24 prompt, response, spent Stage 1 cache, split
   registry, and hashes before using them.
6. Preserve the historical control byte-for-byte.

### Phase 1: Write the scientific interfaces before transport code

Create short specifications for four separate jobs:

1. evidence-chunk interpretation and role-independent feature discovery;
2. cross-chunk concept consolidation and omission criticism;
3. later causal-role review for already-grounded features;
4. extraction-definition generation for retained features.

For each specification, write:

- the plain-language mission;
- the evidence-family explanation needed for that pass;
- the exact JSON shape and field glossary;
- citation and grounding rules;
- what the agent may and may not infer;
- a content-neutral example only if necessary.

Remove all temporal-policy text. Do not begin implementation until a reader with no
repository context can explain each job correctly from the prompt alone.

### Phase 2: Build a role-independent, lossless evidence path

Starting from the committed code, implement in small reviewed pieces:

1. a fold-local evidence catalog that retains every eligible Stage 1 item;
2. explicit family identities for all families in Section 4, including TF-IDF
   semantic retrieval and neural-query moments;
3. complementary chunk plans with no arbitrary global top-k before discovery;
4. a role-independent concept ledger;
5. deterministic alias normalization that preserves provenance;
6. a mapping from every candidate back to its complete supporting evidence;
7. separate critic paths for unrepresented evidence and rejected candidates;
8. closed role endorsements that actually reach final selection;
9. explicit non-grounding treatment of direct numerical summaries;
10. semantic witnesses and aggregate moments for neural queries, never raw vectors;
11. complete support for HTR evidence stored as `token`, `phrase`, or `concept`.

Keep internal hashes, split identities, producer metadata, and schemas in an
authenticated envelope. Render only scientific decision fields and stable evidence
IDs to the model.

### Phase 3: Prove honesty and context safety offline

Use cached Stage 1 evidence. Do not recompute expensive models.

Add tests proving that:

1. every cataloged evidence item reaches at least one discovery pass;
2. every proposal citation resolves to supplied concept-bearing evidence;
3. direct numerical summaries cannot ground a feature name;
4. raw query vectors, row values, patient IDs, full notes, oracle fields, and fresh
   validation data cannot enter selector prompts;
5. purely prognostic concepts are representable before role assignment;
6. treatment evidence alone cannot mechanically produce a confounder endorsement;
7. HTR token/phrase/concept witnesses all survive rendering;
8. role endorsements reach the final selector;
9. exact internal requests and exact rendered first/repair messages are separately
   hashed and bound to the cache key;
10. every cumulative message sequence passes the unchanged context guard before
    transport;
11. closed JSON validation and repair remain fail-closed;
12. historical v24 bytes remain unchanged;
13. selector reasoning is exactly 5,000 and extractor reasoning is off;
14. post-extraction review is enabled;
15. the final estimator fails if a causal forest is unavailable;
16. oracle and not-yet-consumed validation information cannot affect adaptation.

Generate exact context-size audits for every planned pass, including repair
attempts. Compaction is allowed only through semantically complete complementary
chunks, not silent evidence loss.

### Phase 4: Prepare an offline review packet

Before any new remote run, show the user:

- the byte-exact historical model-facing prompt;
- the old hierarchical prompt retained only for the prompt-quality ablation;
- the proposed new plain-language discovery prompt;
- one real family-discovery prompt rendered from cached evidence;
- one role-review prompt for already-grounded candidates;
- one extraction-definition prompt;
- the complete list of internal fields hidden from model visibility;
- the exact context-size audits;
- the honesty, provenance, evidence-preservation, and closed-schema test results;
- new immutable precommit manifests and cache namespaces.

Stop for review at this point. Do not launch the remote comparison before approval.

### Phase 5: Run the cached-evidence discovery comparison

After approval, compare:

1. the frozen historical compact control;
2. larger stratified deterministic compaction;
3. hierarchical per-family prompting;
4. complementary chunked proposals followed by union;
5. hierarchical discovery plus an independent omission critic.

Also run the same hierarchical evidence architecture once under the old dense
instructions and once under the new plain instructions. This separates prompt
comprehension from evidence-delivery architecture.

Use new selector caches and output directories for every changed rendered prompt.
Do not reuse historical model responses under a new prompt.

Before oracle access, compare:

- first-response valid-JSON rate;
- repair frequency and success;
- breadth and specificity of named features;
- citation and grounding validity;
- source-family coverage;
- preservation of lower-ranked evidence;
- cross-pass stability;
- redundancy and over-broad merging;
- later role-review consistency.

Choose the simplest design that materially improves evidence preservation and agent
comprehension using only these observable and prompt-level measures.

### Phase 6: Test extraction and adaptive review on one outer fold

For the chosen frozen discovery design:

1. generate extraction definitions;
2. extract the selected features on the currently permitted development rows;
3. run the nested observable diagnostics;
4. enable the bounded post-extraction review loop;
5. re-extract only changed definitions;
6. use each fresh validation gate once;
7. include direct numerical signals and neural-query moments in source-preservation
   tests and ablations;
8. freeze the registry after convergence;
9. fit the final outer-fold causal forest with no fallback;
10. freeze the held-out predictions.

Only then perform retrospective feature-recovery and true-ITE evaluation for that
fold. Use those oracle results to describe the method, not to reopen or choose the
frozen design.

### Phase 7: Run all outer folds

If the one-fold pilot is honest, stable, and operationally sound:

1. precommit every outer and inner partition;
2. run the complete adaptive pipeline independently within each outer fold;
3. produce one immutable held-out prediction per patient;
4. authenticate the combined prediction file;
5. perform the post-hoc oracle join once;
6. report fold-level and aggregate treatment-effect performance, feature recovery,
   extraction quality, source preservation, and computational cost.

## 10. Existing artifacts and repository state

Committed baseline:

`d00e6a1dc8a7bb11294b3870a8e3c19e7979866b`

Synthetic benchmark metadata:

`synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/metadata.json`

Metadata SHA-256:

`52897cf1ca16241a141bd9b896757d4767df1177658bea2170ef48b82a8aa437`

Completed older benchmark metrics:

`artifacts/all_evidence_fusion/benchmark_five_contractrag_v5/posthoc_oracle_evaluation/posthoc_oracle_metrics.json`

Metrics SHA-256:

`0c702cb7d1f7c3f92fb0ccd1d60be4f77209b18f169c81f744975ace9b2e6e8d`

Historical v24 control directory:

`/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/all_evidence_fusion/benchmark_five_agentic_loop_v24_exact_lexical_thisenv_r2`

Historical immutable selector response:

`.../outer_fold_001/immutable_fusion_response.json`

Response SHA-256:

`55b5837ea815db5166c6ec02798c8f79ef30a5973dd4f01fab7de9152baffe04`

Authenticated fold-1 development-row Stage 1 cache:

`.../post_extraction_review_spent_evidence_cache/79ca229f95c81825dd271815d88d5bdda9b847a89afda2fd9278548574b3d8b9.json`

Cache file SHA-256:

`b28a423f9e7e1a5e53f6e0afe3b6fa68b24418e85a4c9d2c7ea284673e4df5bd`

Historical multi-model Stage 1 handoff:

`/data1/ken/pcori_dev/pcori_experiments/five_conf_five_mod_agent_refactor_7-9-26/multi_model_forest/25366977da7d/handoff/discovery_contexts.jsonl`

Handoff SHA-256:

`5cb4971d43adcf3c1d0396ab57b321d569479c5d132fc05f7eb751fbb8907cea`

Its Stage 1 configuration is the sibling `stage1_config.json`, SHA-256
`2867cbcde684873f939b81af39225434934501e4d734d9dee0357598b5a0f6bc`.

Resealed TF-IDF Stage 1 handoff:

`artifacts/five_conf_five_mod_tfidf_resealed_20260715/discovery_contexts.jsonl`

TF-IDF handoff SHA-256:

`87e4bc947e643b1b319874b4a5d3728c9db895de0522f0fe6ad2a7b83eed93e4`

Incomplete cached comparison output:

`/tmp/cached_stage1_evidence_integration_v5_real_fold1_20260718_r1`

Its selector cache must remain immutable and must not be reused for changed prompts:

`/tmp/cached_stage1_evidence_integration_v5_selector_cache_20260718_r1`

Incomplete scratch implementation:

`/tmp/causal_stage1_ledger_20260717`

Main worktree state at handoff:

- modified `oci/inference/all_evidence_fusion_cli.py`;
- modified `oci/inference/all_evidence_fusion_runner.py`;
- modified `tests/test_all_evidence_fusion_cli.py`;
- modified `tests/test_all_evidence_fusion_runner.py`;
- untracked `scripts/probe_all_evidence_fusion_remote.py`;
- untracked `scripts/submit_all_evidence_fusion_request_remote.py`;
- untracked this project brief.

Do not reset this worktree.

## 11. Execution constraints

- Use `/home/klkehl/thisenv/bin/python` for Python execution.
- Do not run a language model on the current host.
- Use authenticated cached Stage 1 evidence for the initial work.
- Do not recompute expensive Stage 1 models merely because prompts changed.
- If later Stage 1 computation genuinely needs GPUs, the current machine has two
  A6000 GPUs and execution may require approval outside the sandbox.
- Remote selector inference is available on Camus at port 8010.
- Do not contact Camus until the offline prompt/test/context packet has been shown
  to and approved by the user.
- Keep selector reasoning enabled at exactly 5,000 tokens.
- Keep extractor reasoning disabled.
- Do not add temporal-boundary enforcement or temporal-policy prompt text.
- If HTR is refitted, keep its sentence encoder unfrozen and trainable.
- Keep post-extraction adaptive review enabled.
- Use a causal forest as the final estimator and fail closed if it is unavailable.
- Preserve all unrelated user changes and immutable historical controls.

## 12. Definition of completion

This goal is complete only when:

1. every Stage 1 evidence family reaches the discovery process without hidden global
   truncation;
2. the agents can turn noisy distributed evidence into specific, grounded patient
   features before causal-role assignment;
3. the extractor can materialize those features with acceptable coverage and
   validity;
4. bounded adaptive review improves or preserves predeclared untouched observable
   metrics without oracle access;
5. direct numerical Stage 1 signals and neural-query moments are honestly retained
   and ablated;
6. a causal forest, with no non-forest fallback, produces frozen outer-held-out
   predictions for all folds;
7. a post-hoc-only oracle evaluation reports feature recovery and treatment-effect
   performance;
8. the final artifacts contain enough provenance to reproduce exactly which rows,
   evidence, prompts, extraction definitions, models, and predictions were used.

The main scientific question is not whether an agent can rank a dense JSON object.
It is whether the combined statistical evidence can be translated into the actual
patient characteristics encoded in the notes, extracted reliably, and used in an
honest causal treatment-effect model.

## 13. Resumed implementation checkpoint (2026-07-18)

The architecture-at-a-time implementation is now the authenticated production
path for both initial discovery and later adaptive reconsideration.  It covers all
ten active Stage 1 concept families in their exact declared order.  Each family is
interpreted, consolidated, and coverage-criticized independently.  Only ten compact
family dossiers reach cross-architecture planning.  Raw evidence can be revisited
only by exact requested evidence ID under the fixed 24-ID / 96,000-byte global
lookback bound.  The proposal and extraction definitions are frozen before any
untouched review gate is materialized or consumed.

The adaptive prompt contract authenticates the exact six-stage system text,
settings, top-level payload schema, static literals, dynamic paths and shapes, and
nested output schemas.  Selector reasoning is exactly 5,000 tokens.  Definition
extraction reasoning is disabled.  The definition-stage vocabulary grounding
policy is an exact static literal and is mutation-tested.  The current semantic
validator and the seven-file implementation bundle are revalidated for every cache
hit.  An outer adaptive artifact is comparison-only and cannot bypass current
execution authentication.

The runner bridge now supplies fresh accumulated-spent evidence to the adaptive
builder on later rounds, never to the legacy reviewer.  It compiles add, drop,
split, rename, revise-definition, merge, and role changes into a frozen executable
registry, with exact current-family citation coverage for merge/revision operations.
Rejected gates restore the baseline registry; accepted gates commit the exact
candidate registry.  Final audit fields describe observed adaptive executions and
their exact proposal/executable hashes.

Regression checkpoint after the final static-literal synchronization:

- 28 adaptive hierarchy and authenticated executor tests passed;
- 19 adaptive runner-bridge tests passed;
- 46 approved-batch and offline-packet tests passed;
- 86 core hierarchy/compiler/cache/catalog tests passed, with 4 intentional skips;
- 230 evidence provenance, deferred-gate, upstream, review, integration, and CLI
  tests passed;
- 31 OpenAI-compatible JSON runner identity tests passed;
- 55 adjacent discovery/review/upstream tests passed.

Total: 495 passed, 4 skipped.  Black single-worker checks and pyflakes checks for
the directly changed adaptive and packet modules/tests are clean.

Prepare-only must bind the exact future execution endpoint
`http://camus:8010/v1`, model
`RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic`.  The endpoint is part of the approved
runner identity, so the older TEST-NET placeholder must not be used.  This does not
contact Camus: prepare-only fails if a client pool is constructed, if runner
execution metadata changes, or if any JSON job is submitted.  It reports zero
remote calls.  The next action is local preparation for both synthetic datasets,
followed by publication of the exact offline packet and a hard stop for user
approval before any Camus request.

## 14. Active prepare-only checkpoint (2026-07-19 02:42 EDT)

The first sandboxed prepare attempt authenticated both 623 MiB handoffs, replayed
both exact fold-1 caches, and produced valid fold-1 preparation artifacts, but
failed closed when PyTorch first initialized HTR with `No CUDA GPUs are available`.
This was sandbox device isolation: an approved out-of-sandbox PyTorch probe returned
`True 2`.  The incomplete directories were preserved, not deleted, as:

- `hierarchical_all_arch_one_20260719_v1_{scratch,preparation}_sandbox_cuda_unavailable_20260719_0229`;
- `hierarchical_all_arch_five_20260719_v1_{scratch,preparation}_sandbox_cuda_unavailable_20260719_0229`.

The authoritative prepare-only jobs were relaunched with GPU access using the same
arguments and exact Camus runner identity, with one-by-one on `cuda:1` and
five-by-five on `cuda:0`.  Prepare-only still constructs no remote client and makes
zero JSON/model calls.  Both authoritative target paths were fresh at relaunch.
Both jobs replayed fold 1 and entered uncached fold 2 at 02:42:33 EDT.  Both passed
the prior CUDA failure boundary and loaded/train the unfrozen HTR sentence encoder.

No exact replay exists for folds 2--5.  Required cache keys are:

- one-by-one: fold 2 `5a35682310e8491630f750c4b58af384d68fe777ef087f7cbb218353e506ab69`,
  fold 3 `6ae859b093967dcf52bb612960d6a67c17de6e745425a7d6db2b3e6755e510aa`,
  fold 4 `e3dd9f03e4b2f2329c5fd6a726f6f84e2a739bc20dd46a70e7eebcca6a51f830`,
  fold 5 `80833c064a49462feb68fd23eacaefd424ad6eecfe1ee522ee9ed6e9a57bcbb1`;
- five-by-five: fold 2 `379bca2b911ef3cc22a2607434623360d0c33d4c06e774c140e2382151bec20c`,
  fold 3 `2a0922ccbc8483715d035d26e3212a34cca7c1b602df37dbf1ebc30c6762b710`,
  fold 4 `94d5d860cd761ba02393451afb59729806b3e2378cf99dcfb986a7483316b407`,
  fold 5 `385490e1f24feb6b2b4ed5a986a5fe5390b42baa9d4910b31ed835b119067216`.

The exact-current historical runtime was 121.9 minutes for one-by-one and 125.7
minutes for five-by-five per uncached initial-spent fold.  Four sequential misses
therefore imply about 8--8.5 hours wall time with both datasets running concurrently
on separate GPUs.  This is consistent with five nuisance folds and 1,000 bootstrap
replicates in the frozen Stage 1 configuration.  Do not interrupt a healthy fold:
the provider commits its complete cache JSON atomically only after all three
spent-evidence backends finish.

Fold-2 milestone:

- one-by-one cache key
  `5a35682310e8491630f750c4b58af384d68fe777ef087f7cbb218353e506ab69`
  committed at 04:45:36 EDT; file SHA-256
  `7e1f45048b5317f245b63728ec617d3c0f4023d5693356c490ac63a57667ede5`;
  `outer_fold_002/immutable_fold_preparation.json` completed at 04:46:29 EDT;
- five-by-five cache key
  `379bca2b911ef3cc22a2607434623360d0c33d4c06e774c140e2382151bec20c`
  committed at 04:50:44 EDT; file SHA-256
  `6938061e8ebedc742fb5dcf55b6086b8ded306e2e67eed90c918ec877884251b`;
  `outer_fold_002/immutable_fold_preparation.json` completed at 04:51:52 EDT.

Both jobs then continued to fold 3 without interruption.

Fold-3 milestone (one-by-one):

- cache key
  `6ae859b093967dcf52bb612960d6a67c17de6e745425a7d6db2b3e6755e510aa`
  committed atomically at 06:49:04 EDT; file SHA-256
  `da3de77e90cdbec64f6a70252c9b10a8999db458618633ada86dc8d114dafe83`;
- `outer_fold_003/immutable_fold_preparation.json` completed at 06:49:55 EDT;
- fold 4 entered its uncached spent-evidence build at 06:50:50 EDT.

Fold-3 milestone (five-by-five):

- cache key
  `2a0922ccbc8483715d035d26e3212a34cca7c1b602df37dbf1ebc30c6762b710`
  committed atomically at 07:01:00 EDT; file SHA-256
  `f3da27a8f0733c0d470b4a4acf5cf4c877901c559bdac7e630c6d04e191bd2b3`;
- `outer_fold_003/immutable_fold_preparation.json` completed at 07:02:07 EDT
  with file SHA-256
  `216fade98c5e37ced5f6ba0d28aee12d88da4ff933d1508dc3935ad9bb0a25d4`;
- fold 4 entered its uncached spent-evidence build at 07:03:03 EDT.

Fold-4 milestone (one-by-one):

- cache key
  `e3dd9f03e4b2f2329c5fd6a726f6f84e2a739bc20dd46a70e7eebcca6a51f830`
  committed atomically at 08:55:28 EDT; file SHA-256
  `558e44608a32f15583530f9101ae022b28ae36189aa1da40b86843a875cb20a0`;
- `outer_fold_004/immutable_fold_preparation.json` completed at 08:56:24 EDT
  with file SHA-256
  `c4be7e918aba0a0ab386b63766d2dae6077d1854edf96e5f6224fe90ad014aef`;
- fold 5 entered its uncached spent-evidence build at 08:57:18 EDT.

Fold-4 milestone (five-by-five):

- cache key
  `94d5d860cd761ba02393451afb59729806b3e2378cf99dcfb986a7483316b407`
  committed atomically at 09:12:04 EDT; file SHA-256
  `fc3f30970f0973e8602e2aeb32928aac6f6cb71f01fd46006d1d2b4d63b0504b`;
- `outer_fold_004/immutable_fold_preparation.json` completed at 09:13:12 EDT
  with file SHA-256
  `c508668bfb67671fa16f9e1c0feef1e781721d617d9728a9c23419bf7c50f0ef`;
- fold 5 entered its uncached spent-evidence build at 09:14:08 EDT.

Both datasets are therefore in their final preparation fold with exact fold 1--4
cache coverage.

Fold-5 and prepare-only completion milestone (one-by-one):

- cache key
  `80833c064a49462feb68fd23eacaefd424ad6eecfe1ee522ee9ed6e9a57bcbb1`
  committed atomically at 11:00:24 EDT; file SHA-256
  `da3f3aad38b5f905d4bb394cf610ce429ddbec2b595c6f631c6852909cac6358`;
- `outer_fold_005/immutable_fold_preparation.json` completed at 11:01:20 EDT
  with file SHA-256
  `cfea826a0f6f805c0cd19dbbd7a0cad1642aa7b0f679c6830378015b7823ef6a`;
- the authoritative process exited `0` with status
  `hierarchical_discovery_prepared_awaiting_approval`, approval SHA-256
  `750dbb3e072bac8ec41afb1123c0f3b0eb3dfc561e1d58273d58f437ed66763f`,
  and offline review packet SHA-256
  `d62746b13d2eaa141cc78433a497c094bf0610e470dd233cd645e37d5a38c206`;
- its terminal record says `hierarchical_json_jobs_executed: 0`,
  `remote_clients_constructed: false`, `remote_calls_made: false`,
  `oracle_columns_read: false`, `predictions_written: false`, and
  `final_run_manifest_written: false`.

The one-by-one packet is undergoing local cross-hash authentication.  Do not use
its approval digest until that audit passes and the exact packet is shown to the
user.  Five-by-five remains in fold 5.

Fold-5 and prepare-only completion milestone (five-by-five):

- cache key
  `385490e1f24feb6b2b4ed5a986a5fe5390b42baa9d4910b31ed835b119067216`
  committed atomically at 11:19:39 EDT; file SHA-256
  `a23b3489db1af0e4f059e27878504ac78df6103411ce7eeece27e362398a1cfa`;
- `outer_fold_005/immutable_fold_preparation.json` completed at 11:20:48 EDT
  with file SHA-256
  `d3c87aa86fd96b611e16ac795f151a5bb3c79f2e4a12c1926cef7ede5a26df1`;
- the authoritative process exited `0` with status
  `hierarchical_discovery_prepared_awaiting_approval`, approval SHA-256
  `79a8dbc1693b13501c89562e5599d18c6ee026058673d2c845ec969f731d48f4`,
  and offline review packet SHA-256
  `035223f539103e30fff1065522af747770bea204285e315b7a82daf91bc6c5d9`;
- its terminal record says `hierarchical_json_jobs_executed: 0`,
  `remote_clients_constructed: false`, `remote_calls_made: false`,
  `oracle_columns_read: false`, `predictions_written: false`, and
  `final_run_manifest_written: false`.

The one-by-one packet passed full local replay and cross-hash authentication.  The
five-by-five packet is undergoing the same audit.  Neither approval digest is
authorized until both exact packets are shown to the user and explicitly approved.

Final offline-packet authentication checkpoint:

- both packets pass current production-validator replay, envelope/cross-hash,
  current-source identity, comparison-byte, exact three-file manifest, file-mode,
  and post-audit side-effect checks;
- every fold contains nonzero evidence from all ten active architectures, delivered
  exactly once through family-pure chunks before ten-dossier integration; complete
  raw multi-architecture dumps, global top-k truncation, and sparse fallback are
  absent;
- the exact six-stage adaptive contract is current: architecture interpretation,
  consolidation, coverage audit, ten-dossier planning, bounded integration, and
  extraction definition; the first five stages use exactly 5,000 reasoning tokens,
  extraction reasoning is off, raw lookback is bounded to 24 IDs/96,000 bytes and
  8 IDs per target, and proposal freeze precedes the next gate;
- all first-gate intents validate against current code; labels/views were not
  exposed, materialization remains deferred until exact approval and proposal
  freeze, and all declared hierarchical job-cache roots remain absent;
- exact Camus identity is `http://camus:8010/v1` with model
  `RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic`; local identity reconstruction had no
  client pool or execution records;
- one-by-one approval SHA-256 is
  `750dbb3e072bac8ec41afb1123c0f3b0eb3dfc561e1d58273d58f437ed66763f`;
- five-by-five approval SHA-256 is
  `79a8dbc1693b13501c89562e5599d18c6ee026058673d2c845ec969f731d48f4`.

The required next action is to show the exact immutable packets to the user and wait
for explicit approval of these two batch approval digests.  Do not substitute the
offline packet SHA values for the approval SHA values.  Do not contact Camus before
that approval.

## 15. Exact user approval received (2026-07-19 16:49 EDT)

After the two authenticated execution digests were presented together and the goal
was stopped at the mandatory approval boundary, the user replied `approved`.  This
explicitly authorizes remote Camus execution of exactly these two immutable batches:

- one-by-one:
  `750dbb3e072bac8ec41afb1123c0f3b0eb3dfc561e1d58273d58f437ed66763f`;
- five-by-five:
  `79a8dbc1693b13501c89562e5599d18c6ee026058673d2c845ec969f731d48f4`.

This approval does not authorize a different digest, changed implementation,
different endpoint/model identity, nonfresh final output, premature oracle access,
or any relaxation of the proposal/prediction freeze.  Reconstruct each execution
from the authenticated preparation directory and exact replay-cache registrations,
use a fresh final output directory and fresh output-local neural-query cache, and
let the code reject any byte-identity mismatch before the first remote call.

## 16. Approved v1 packets retired after pre-network replay defect

The approved v1 batches above must not be executed.  A mandatory fresh-output
reconstruction exposed two preparation/replay defects before any Camus client was
constructed or any oracle data were read:

- the prepare CLI advertised newly copied output-local caches even though the
  immutable input manifest binds the exact historical read-only sources; and
- each immutable fold preparation embedded the prepare scratch absolute cache path
  in its semantic-compatibility migration ledger.  Replaying into a required fresh
  output root therefore made `_write_immutable_json` reject a mutation of the
  preparation manifest before the approval gate.

The path-bound compatibility packet, including the intermediate five-by-five v2
digest `8f42b0922e2c3558c68ba342a3567e7f28379552e1a0bcc2447bad7f73adb32a`,
is non-executable and was never presented for approval.  The user's section-15
approval covers only the retired v1 digests; it cannot authorize the corrected v3
packets below.

The correction is fail-closed and source-bound:

- compatibility ledger schema v3 replaces the output-local absolute cache path
  with an authenticated relative filename and records
  `absolute_location_recorded: false`; exact historical v1 source identities,
  runtime bytes, cache keys, provider identity, and source hashes remain bound;
- input/fold preparation schemas v2 bind the complete current compatibility
  identity, which is captured once and rechecked after all folds;
- prefix salvage validates that exact identity;
- the CLI now distinguishes exact approval-preserving historical registrations
  from output-local copies usable only for a new preparation and deduplicates
  context-fit registrations; and
- targeted compatibility, packet, hierarchy, CLI, and salvage tests pass, with
  the broad final targeted run at 85 passed and no failures (expected skips only).

Do not edit source before approved execution: the corrected packets bind its exact
hashes.

## 17. Corrected v3 packets and fresh-root replay proof

Both corrected prepare-only runs exited `0` with zero clients, calls, or jobs;
oracle visibility and prediction/final-output writes remained false.  Independent
read-only audits replayed the current production packet composers and validators
and passed every source, envelope, manifest, architecture-coverage, routing, and
offline-boundary check.

Corrected one-by-one identities:

- approval SHA-256:
  `4b0dcc810600ec963b688f0683eb98cf98bb47bf2596ecf34c0c9405dd386cd8`;
- input-manifest content SHA-256:
  `65f66fadb4d60a4eb956e752d078304282823a7b816d735749a95520b6611d15`;
- offline-packet logical SHA-256:
  `55e2ec15911e2c96ba4f1a06604a2d6e7271cd277941746fa532d0eb9da9e122`;
- preparation directory:
  `artifacts/all_evidence_fusion/hierarchical_all_arch_one_20260719_v3_preparation`.

Corrected five-by-five identities:

- approval SHA-256:
  `6e046fa5048d97f0a3c6b52a9518471270ca1cdde144e35fa9b1ac38fbcf1e7d`;
- input-manifest content SHA-256:
  `8c7a29ed19671bcf99f7112284817aa532fe410a5cb0d89dcd4741fb897767a4`;
- offline-packet logical SHA-256:
  `2e0474c7154a3a93918d2119d51f9786c59786b3a3fb953cf4e674495ed1baf8`;
- preparation directory:
  `artifacts/all_evidence_fusion/hierarchical_all_arch_five_20260719_v3_preparation`.

Each packet contains nonzero evidence from all ten required Stage-1 architectures
and preserves the architecture-local dossier -> coverage audit -> bounded raw-ID
lookback -> cross-architecture integration -> proposal-freeze workflow.  Direct
numerical evidence remains non-grounding, selector thinking is exactly 5,000,
extraction thinking is off, sparse fallback is forbidden, and the estimator is
strict `CausalForestDML`.

Before requesting approval, each preparation was reconstructed with the exact five
historical cache registrations into a different fresh output root while supplying
an all-zero approval digest.  Both probes reproduced all five cache files with the
ten exact approval-bound file hashes and then exited only at
`approved batch SHA-256 does not match the offline packet`.  Each probe tree
contains exactly those five spent-cache files; neither preparation nor probe tree
contains a hierarchy job-cache, remote result, prediction, final-output, or oracle
artifact.  This proves output-root portability through the last pre-network gate.

Mandatory next boundary: show both corrected approval digests to the user and
obtain fresh explicit approval naming or unambiguously covering both.  Do not treat
the section-15 `approved` reply as authorization for either v3 digest.  After that
approval, run one-by-one first and five-by-five second; never overlap the two full
runs because extraction concurrency can otherwise exceed the intended 128 calls.

## 18. V3 approval, live failures, and required bounded-response repair

The user explicitly approved both v3 packets and directed execution.  One-by-one
was started first; five-by-five never started.  The first command misspelled the
CLI approval option and exited in `argparse` before creating the fresh output root
or making a remote call.  The corrected option is
`--hierarchical-approved-batch-sha256`.

Two correct live invocations then each reproduced all five output-local spent
caches with the exact approval-bound hashes, passed immutable preparation and
batch validation, and reached the first fold's first architecture-local Camus job.
Both failed closed before any hierarchy result was cached:

1. the first response contained a duplicate JSON key, `feature_names`, and was
   rejected by `parse_strict_json_object`;
2. the recovery invocation used a new fresh execution directory, the same v3
   packet/digest/endpoint/model/source/arguments, and an unchanged empty hierarchy
   cache, but its regenerated first response cited an unsupplied evidence ID and
   was rejected by `validate_interpret_evidence_chunk_response`.

The failed execution roots are preserved as:

- `artifacts/all_evidence_fusion/hierarchical_all_arch_one_20260719_v3_execution`;
- `artifacts/all_evidence_fusion/hierarchical_all_arch_one_20260719_v3_execution_retry01`.

Neither contains predictions, a final run manifest, or oracle output.  The v3
preparation's hierarchy job-cache root remains absent because invalid responses
are never stored.

Do not continue whole-run v3 retries.  The hierarchy currently makes one remote
call per logical job: parse failures are classified as non-retryable inside the
JSON runner, and semantic-validator failures escape the orchestrator.  This
contradicts the interface specification's promised one bounded, authenticated
repair attempt and is not viable across roughly 468 initial hierarchy calls plus
dynamic work.

The active correction must add one fail-closed repair request for both strict-JSON
and semantic-validation failures, use only static/sanitized diagnostics, bind the
exact initial/repair message policy and implementation into the precommit/offline
packet/cache identities, retain the original selector/extraction thinking policy,
cache only validated results, and test duplicate-key repair, unsupplied-evidence
repair, and exhausted repair.  Do not modify Stage 1 model modules while building
the corrected packets: the expensive registered spent caches must remain exactly
reusable.  After the repair passes independent review, prepare and authenticate
new immutable packets from the same historical cache registrations; v3 approval
cannot authorize their changed identities.

## 19. V4 authentication, live failure, and retirement

The bounded one-repair implementation was prepared as v4 and independently
authenticated for both cohorts.  The machine approval identities were:

- one-by-one:
  `78c4cc8d0d62a555f3609c14b9e7a8f5566aaf04a29fcaca1260a03addab8459`;
- five-by-five:
  `98afe7f95c04dd1a0383ade201b0ce5a142a61c9a3e769a82d1a449a5b8bc6c8`.

The first sandboxed one-by-one execution reconstructed all five spent-evidence
caches and then failed DNS resolution before any remote request.  A fresh
network-enabled retry reconstructed the same caches, reached Camus, and exercised
the initial response plus its one authenticated privacy-preserving repair.  Both
responses failed semantic validation because they cited an unsupplied identifier.
No invalid response was cached.  No prediction, final manifest, or oracle output
was written.  The failed roots are preserved as:

- `artifacts/all_evidence_fusion/hierarchical_all_arch_one_20260719_v4_execution`;
- `artifacts/all_evidence_fusion/hierarchical_all_arch_one_20260719_v4_execution_retry01`.

The five-by-five v4 execution never started.  Both v4 packets are retired.

The root cause was not the hierarchical architecture-at-a-time design.  Static
model-facing output examples contained regex-valid placeholder strings such as
`supplied_member_id` and `candidate_id`.  The repair prompt instructed the model
to use only identifiers visible in the request, so copying those placeholders was
consistent with the prose but inconsistent with the supplied evidence domains.
The defect affected base and adaptive jobs, not only the first interpretation job.

## 20. Dynamic response contracts, source binding, and output-aware chunks

The post-v4 correction derives a strict JSON Schema and identifier-ownership map
from designated request fields for every base and adaptive job.  The exact schema,
canonical bytes, hashes, ownership map, rendered request, job ID, cache identity,
repair job, and offline packet are authenticated together.  Runtime transport uses
vLLM strict `json_schema`; semantic validators remain authoritative.  Identifier-
looking examples were removed from static and runtime prompt contracts.

An independent audit then found that the new response-contract helper was imported
by the base hierarchy but was not included in its implementation hash.  The base
hierarchy now binds a complete implementation bundle into inner precommit, every
initial and repair job, runtime drift checks, and the cache validator identity.
A helper-only change therefore rekeys preparation or fails closed rather than
silently changing a previously approved schema.

Live probing found a second independent limit.  Historical architecture chunks
contained roughly 163 semantic member IDs at the median, 175 at p90, and up to
191.  A production-equivalent 61-member probe using the first dynamic contract
still allowed every evidence disposition to repeat the chunk-wide member-ID
domain.  Camus exhausted all 25,000 completion tokens with
`finish_reason: length` and returned 83,353 bytes of incomplete JSON.

The corrected interpretation contract uses one `anyOf` branch per evidence atom:

- `evidence_id` is an exact `const`;
- member IDs are restricted to that evidence atom's owned enum;
- the member-disposition array has the exact owned length;
- generated names, prose, arrays, and per-member name fanout have finite bounds;
- semantic validation still enforces uniqueness, exact coverage, and name unions.

Rejection review uses the analogous candidate-owned evidence branches.  Camus was
also probed directly and accepts the required `maxLength` keyword; `uniqueItems`
remains forbidden because the deployed grammar does not implement it.

Architecture chunk plans now add a third authenticated packing dimension,
`max_semantic_member_ids_per_chunk`, fixed by default at 64.  It is bound through
the plan identity, audit, hierarchy config, approved fold/batch packet, CLI, base
runner, and adaptive policy.  Every evidence atom and every semantic member is
still delivered exactly once; an atom that individually exceeds the bound fails
closed and must be losslessly batched by its native adapter.

The corrected production-equivalent 61-member probe passed on Camus:

- `finish_reason: stop`;
- 24,928 prompt tokens and 14,200 completion tokens;
- 10 concepts, seven complete evidence dispositions, and all 61 members;
- strict JSON and semantic validation passed;
- response content SHA-256:
  `b01daddce57327d7ee76102cb6a841c92464928ed218c81b2afa1839def41248`;
- validated response SHA-256:
  `52fb5465781aabbf831a83c00070a4e335081a712173440cbe6f325d70169612`.

Only hashes, counts, and transport metadata from this diagnostic were retained;
raw model content was not persisted.  Fresh v5 packets must not be prepared until
the complete base/adaptive/cache/packet regression and independent source audit
pass on the final bytes.

The user's run authorization covers this benchmark workflow.  Packet digests are
machine integrity and resume controls, not an interactive approval ceremony.  The
operator-facing production wrapper must compute, authenticate, carry, and record
them internally; it must not require end users to approve digests manually.
