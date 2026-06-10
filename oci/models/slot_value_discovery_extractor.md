# Slot Value Discovery Extractor

This note describes how `SlotValueDiscoveryExtractor` turns clinical text into
slot-level features. The key point is that this extractor does not perform exact
span extraction. It finds relevant chunks with learned slot attention, then
summarizes generic value-bearing signals from those chunks.

## Data Flow

```text
clinical text
  |
  | chunk_text_words(...)
  | - whitespace word chunks
  | - default svx config: 64 words, 16-word overlap, max 128 chunks
  v
[chunk string 0, chunk string 1, ...]
  |
  | SentenceTransformer.encode(chunk strings)
  v
chunk embeddings: [n_chunks, embedding_dim]
  |
  | per slot query:
  | - seeded slots = embedding("patient age"), embedding("PD-L1 expression"), ...
  | - free slots = random learned query vectors
  |
  | score(slot, chunk) = dot(query, chunk_embedding) / temperature
  | attention = softmax over chunks
  v
slot attends to likely relevant chunks
  |
  | same attention weights are applied to chunk-level value features
  v
slot feature = projection(
    attended semantic embedding
    + attended generic value features
    + learned value-prototype probabilities
    + max attention score
    + attention entropy
)
```

## What Gets Embedded

The embedding model receives overlapping word chunks of the clinical note. The
chunker is `chunk_text_words()` in `concept_embedding_utils.py`; it uses simple
whitespace spans so the chunk boundaries are independent of any sentence
transformer tokenizer.

For live inference, `_chunks_to_tensor()` creates the chunks and embeds them with
`SentenceTransformer.encode()`.

For cached runs, `ConceptEmbeddingCache` precomputes the same chunk embeddings.
The batch then passes them to the extractor as:

```text
cached_hidden_states: [batch, max_chunks_in_batch, embedding_dim]
cached_attention_mask: [batch, max_chunks_in_batch]
```

The extractor still re-chunks the raw text to compute value features, so the
value-feature rows line up with the cached embedding rows.

## How A Slot Finds Its Associated Value

Each slot has a query vector. Seeded slots start as embeddings of configured
concept names, such as `patient age` or `PD-L1 expression`. Free slots start as
random unit vectors and are learned during training.

For every sample, the extractor computes a similarity score between each slot
query and each chunk embedding:

```text
scores[b, slot, chunk] = dot(normalized_chunk, normalized_query) / temperature
```

After masking padded chunks, the scores are softmaxed across chunks. This gives
each slot a soft distribution over the chunks in the note.

Separately, every chunk is converted to generic value features by
`value_features_for_chunk()`. These features include:

- number indicators: has number, count, first/max/min/mean number
- percent indicators: has percent, first percent
- comparator indicators: high/low words and symbols such as `>=`, `<`, `high`,
  `low`, `positive`, `negative`
- status indicators: negation, positive, unknown, none, yes, no
- demographic/unit/status words: male, female, time units, lab units, stage,
  grade, score, level

The slot's value summary is an attention-weighted average over chunk-level value
features:

```text
attended_values[b, slot] =
    sum_over_chunks attention[b, slot, chunk] * value_features[b, chunk]
```

So if the `PD-L1 expression` slot attends mostly to a chunk like:

```text
PD-L1 >= 50% positive
```

then its attended value summary will carry signals like `has_percent`,
`first_percent = 0.5`, `has_high_comparator`, and `has_positive`.

## Important Limitation

The association between a concept and a value is chunk-level, not span-level. If
one chunk contains both `Age 72` and `PD-L1 80%`, the value feature vector for
that chunk contains both values. The slot can learn to attend to the most useful
chunks, but within a chunk it does not identify the exact substring attached to
the concept.

In short:

```text
semantic attention chooses chunks;
generic regex/lexical features summarize values inside those chunks.
```

## Shared R-Learner Forest Cross-Fitting

The shared slot-value forest variant is meant to avoid hand-separating
confounders from effect modifiers. All learned slot features and all raw
explicit features are treated as `X`; the causal forest receives `W=None`.

The honest evaluation structure is nested:

```text
outer CV fold
  outer train rows
    |
    | inner K-fold nuisance cross-fitting
    | - fit nuisance model on inner-train rows
    | - predict e_hat and m_hat on inner-validation rows
    | - repeat until every outer-train row has OOF nuisance predictions
    v
  fixed OOF residuals for outer-train rows
    |
    | train final shared slot model from fresh initialization
    | - propensity head: e_final(X) -> T
    | - outcome head:    m_final(X) -> Y
    | - tau head:        tau_final(X) using fixed OOF e_hat/m_hat
    v
  extract final shared X features
    |
    | fit CausalForestDML(X=X, W=None, T, Y)
    v
  predict on untouched outer-test rows
```

The inner nuisance models are temporary residual generators. They are discarded
after producing out-of-fold `e_hat` and `m_hat` for the outer-train rows. The
final tau model is not copied from an inner nuisance model.

The final shared model starts from the normal slot seed/random initialization.
It then trains one extractor with three heads:

```text
single slot extractor -> X
  |
  +-- propensity head: e_final(X)
  +-- outcome head:    m_final(X)
  +-- tau head:        tau_final(X)
```

The R-loss for `tau_final` uses the fixed OOF nuisance predictions:

```text
(Y - m_oof(X) - tau_final(X) * (T - e_oof(X)))^2
```

This differs from the staged X/W model. In staged mode the final estimator has a
nuisance extractor for `W`, a separate effect extractor for `X`, and the forest
receives both matrices. In shared mode there is one final extractor and the
forest receives only `X`.

The purpose of the inner OOF residuals is to avoid in-sample nuisance/tau
collusion. If `m_hat` and `e_hat` are learned on the same row whose R-loss is
being optimized, they can overfit `Y` and `T`, leaving an easy residual for
`tau` to explain. OOF nuisance predictions make the residual target harder and
more honest while preserving the "everything is X" feature strategy.
