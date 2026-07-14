# Faster paths from TF-IDF/NMF evidence to treatment-effect modeling

The TF-IDF screening and three NMF topic banks are producing useful signal, but
LLM extraction of hundreds of variables for every nested context is too slow to
be the default bridge into modeling. The options below preserve the honest
outer-fold boundary: every representation is learned on the outer-training
rows, held-out text is transformed without refitting, and oracle columns are
used only after predictions are frozen.

## 1. Direct topic-score causal forest (implemented first)

Use the continuous NMF patient-topic scores as the model inputs:

- treatment-bank and outcome-bank scores are adjustment features (`W`);
- effect-bank scores are heterogeneity features (`X`);
- the existing structured `CausalForestDML` head estimates ITEs;
- each outer fold uses its own frozen Stage 1 vectorizer/NMF models and predicts
  its outer test fold once.

This is the fastest faithful baseline because Stage 1 already persisted all of
the required train and held-out transforms. It retains graded evidence rather
than forcing each topic into a brittle binary extraction. Its main drawback is
that an NMF topic is less clinically legible than a canonical variable, and a
single topic can mix several concepts.

## 2. Topic scores plus a small targeted structured supplement

Run the direct topic forest, then extract only a small set of variables tied to
the strongest, least-reconstructed, or most clinically important topics. This
keeps the NMF representation as the broad signal carrier while reserving slow
LLM extraction for perhaps 10--30 high-value concepts. It is likely the best
longer-term speed/interpretability compromise.

## 3. Distilled soft concept extractor

Use the completed LLM extractions as supervision for compact per-concept text
classifiers or regressors. A small model can then emit probabilistic concept
values for every patient in one batched pass. This resembles a soft
[concept bottleneck](https://arxiv.org/abs/2007.04612): fast and interpretable
at inference time, but it requires enough reliable teacher labels and careful
fold-local training to avoid distillation leakage.

## 4. Topic-routed retrieval and extraction

Index note sections or chunks, retrieve only passages relevant to a topic or a
small feature group, and send those passages to the extractor. This follows the
core retrieval-then-generation pattern of
[RAG](https://arxiv.org/abs/2005.11401). It reduces prompt length and should
improve batching, but it still pays an LLM call cost and retrieval recall must
be audited so that relevant pre-treatment evidence is not silently missed.

## 5. Neural soft topic extraction

Train a compact encoder to reproduce the frozen NMF topic-score vectors, then
optionally fine-tune or rotate that representation using only outer-training
data. Contextualized topic models are one related family
([CTM](https://arxiv.org/abs/2004.07737)). This can capture synonyms beyond
literal n-grams, but it adds another learned representation whose stability and
fold isolation need validation.

## 6. Deterministic anchor-based extraction

Turn high-loading topic terms into dictionaries, regexes, negation-aware
matches, unit parsers, and weak labeling functions. Systems such as
[Snorkel](https://arxiv.org/abs/1711.10160) provide a framework for combining
noisy labeling rules. This is extremely fast and auditable for explicit facts,
but weaker for temporality, implicit descriptions, and nuanced clinical states.

## Recommended sequence

1. Establish the direct topic-score causal forest as the no-LLM benchmark.
2. Compare its honest outer-fold ITE behavior with the structured extraction
   path after predictions are frozen.
3. Add targeted extraction only for topic evidence that materially improves
   interpretability or is not represented well by the score model.
4. If repeated large-scale inference is needed, distill those targeted
   variables into compact soft extractors.

The direct forest is based on the generalized random-forest framework
([Athey, Tibshirani, and Wager](https://arxiv.org/abs/1610.01271)); the important
experimental safeguard is not the representation choice itself, but preserving
the nested, fold-local fitting and post-hoc-only oracle evaluation.
