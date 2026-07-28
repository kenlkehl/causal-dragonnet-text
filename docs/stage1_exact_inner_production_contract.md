# Exact-inner Stage 1 production contract and migration

Status: executable contract plus artifact-authenticated native registration for
all ten architecture families. Structural 10/10 registration is complete,
including canonical label-lineage replay and drift rejection for embedding,
legacy, neural-query, and nested TF-IDF captures, plus truthful training-only
TF-IDF policies. Cumulative-spent component implementation and persisted reload
are also complete for all ten families (10/10). The arbitrary-cohort builder,
all-ten exact/cumulative root graph, authenticated loader, and no-approval
one-shot CLI are implemented. Final readiness remains false until a genuine
end-to-end cohort run completes and its identities are independently reviewed.
This document is not an authorization to change or replay an already approved
benchmark packet.

## Finding

`MultiModelForestStage1Runner._inner_model_handoff_rows()` did not create
exact-inner discovery evidence. It grouped architecture-local bookkeeping rows by
the integer named `inner_fold`, copied the full-outer `base_result`, and relabeled
that copy as `candidate_consistency_inner_train`.

The integer did not denote one shared patient partition:

| Stage 1 component | Existing split seed family |
|---|---:|
| HTR nuisance | `10_000 + outer_fold` |
| BoW nuisance | `11_000 + ...` |
| BoW effect | `13_000 + seed_offset + ...` |
| HTR effect | `20_000 + outer_fold` |
| Embedding directions | `40_000 + outer_fold` |
| Candidate consistency | `51_000 + outer_fold` |
| BoW matched-pair uplift | `91_000 + ...` |
| HTR matched-pair uplift | `92_000 + outer_fold` |

Those bookkeeping rows retained row counts but no ordered fit/held-out row IDs,
row fingerprints, split-registry identity, producer identity, fit audit, model
artifact identity, or evidence-payload hash. Therefore matching `inner_fold=1`
and matching counts could not establish a common fit scope. The downstream legacy
loader checked counts and fold numbers, so it could not distinguish these rows
from genuine refits.

This is a leakage and provenance defect even when the copied evidence happens to
be scientifically plausible. A full-outer model has seen the inner held-out labels
that the exact-inner recurrence check is intended to seal.

The investigation also found pre-posthoc reads of `true_ite_prob` in the primary
Stage 1 and matched-pair metrics. Those reads are unrelated to model fitting but
violate the stronger production rule that oracle columns are unavailable until
predictions are frozen.

## Isolated remediation

`oci/inference/stage1_exact_inner_evidence.py` provides the replacement boundary.

1. `CanonicalStage1SplitRegistry` constructs one ordered outer/inner registry.
   Every inner fit and held-out partition is content addressed. Every row is held
   out exactly once within an outer training fold, and every outer test partition
   covers the dataset exactly once.
2. `ExactInnerStage1FamilyRequest` is the only input passed to an architecture
   adapter. Fit rows contain row ID, text, treatment, and outcome. Held-out rows
   contain row ID and text only. Oracle and unrelated columns are never projected.
3. `produce_exact_inner_stage1_evidence_bundle()` requires exactly the ten active
   architecture producers. It invokes them in canonical family order on the same
   split registry and data projection.
4. Every producer identity must bind its name, version, code SHA-256, configuration
   SHA-256, and family. The identity is checked before and after fitting.
5. Every fit audit binds the exact request and split fingerprint, an execution
   identity, and a model-artifact identity. It must attest that held-out labels,
   oracle fields, and secrets were not accessed. An authenticated cache hit is
   permitted only for the identical split fingerprint.
6. Each nonempty concept-bearing payload is hashed. The protocol accepts an
   independently authenticated full-outer payload hash for every family; the
   native adapter layer derives those hashes only from a validated persisted
   full-outer catalog through
   `native_full_outer_payload_registry_from_catalog()`. Exact byte-semantic
   equality is rejected as a clone canary.
7. Each family artifact and the ten-family bundle are independently content
   addressed. The validator recomputes every binding and fails closed on mutation,
   omission, reordering, split drift, identity drift, forbidden fields, or reuse
   claims.

The old `_inner_model_handoff_rows()` path now raises instead of manufacturing a
false exact-inner row. Disabling candidate consistency still returns no rows, as
before. The primary Stage 1 constructor projects out `true_*`, `oracle_*`, and
`ground_truth*` columns, and the in-fit oracle metric reads were removed from the
Stage 1 and pair-uplift implementations.

## Why the complete model refit is a larger integration

The ten active concept families do not all originate in
`MultiModelForestStage1Runner`:

1. `bow_nuisance`
2. `bow_r_loss`
3. `htr_neural`
4. `matched_pair_uplift`
5. `embedding_whole_cohort`
6. `embedding_clustered`
7. `tfidf_semantic_retrieval_contrasts`
8. `tfidf_topics`
9. `tfidf_orphan_ngrams`
10. `neural_query_moments`

The first six are assembled through the legacy all-source path. The next three
come from the registry-sealed TF-IDF producer. Neural queries have a separate
producer/cache lifecycle. A truthful all-ten exact-inner artifact therefore
cannot be implemented by copying or rearranging fields inside the legacy runner;
it must be orchestrated by the production Stage 1 bundle builder.

## Exact-inner integration and remaining production sequence

### 1. Persist the one authoritative registry

The production wrapper must derive outer test rows from its one canonical split
source, construct `CanonicalStage1SplitRegistry` once, write its canonical JSON,
and retain `content_sha256`. Architecture adapters must not instantiate their own
`KFold` for the outer candidate-consistency scope.

Nested cross-fitting *inside* an architecture's exact fit rows remains allowed for
nuisance estimation, but it is an implementation detail. It must not be relabeled
as the shared candidate-consistency split.

### 2. Register the six legacy-family native fits — implemented

The native adapter mapping names the existing fit surfaces and binds their
code, configuration, fit metadata, execution record, model/output artifact, and
catalog payload. The combined builder must make those real fit surfaces emit the
record consumed by `bind_native_family_fit_proof()` for every registered scope:

- BoW nuisance fits treatment and outcome models on `request.fit_rows` and emits
  only fit-scope terms/phrases and metrics.
- BoW R-loss first creates nuisance predictions using fit-only nested cross-fitting,
  then fits its effect model and importance projection within that same scope.
- HTR nuisance/effect and HTR pair attention must be newly trained on the exact fit
  scope. Attention evidence must be derived only from fit-scope text or from nested
  held-outs wholly contained in the exact fit scope.
- Matched-pair uplift must build both candidate and control pools from exact fit
  rows. The canonical inner held-out rows may be transformed but their treatment
  and outcome are unavailable.
- Whole-cohort and clustered embedding directions may use a frozen, authenticated
  text encoder/cache, but supervised directions, cluster contrasts, and readable
  witness selection must use exact fit rows. A cache of evidence fitted on a
  broader scope is not valid merely because the underlying embeddings are frozen.

Each adapter writes a closed fit audit and model-artifact hash before returning its
draft. Row-aligned numerical signals remain in the separate authenticated direct
numerical channel and cannot ground feature names.

### 3. Register the three TF-IDF native fits — implemented

The TF-IDF producer must accept the same registry object rather than deriving its
own splits. Semantic-retrieval contrasts, topics, and orphan n-grams each return a
separate family draft. Their registries, vocabularies, fitted transforms, and
evidence outputs bind the exact request SHA-256.

Topic and orphan-n-gram registrations bind the real nested-fit context metadata,
fitted context, raw score-selection JSON, exact split, label-free held-out
projection, and immutable catalog-derived evidence payload. Their label-based
selection occurs inside disjoint nested training-only partitions. Semantic
retrieval is registered from the shared embedding-fit artifact with a truthful
deterministic exhaustive no-selection policy; its label-free partitions are
replay canaries and impose no vocabulary or output cap.

### 4. Register the neural-query native fit — implemented

The component learns query parameters and aggregate moments only on exact fit rows. Readable
witness selection must remain fit-local. An exact-scope authenticated cache hit is
allowed; a full-outer query bank is not. The family payload remains concept-bearing
evidence, while row-level query values are sealed in the direct numerical channel.

### 5. Write a composite Stage 1 bundle manifest — exact-inner graph implemented

For every outer and inner scope, the builder persists:

- canonical split-registry path and SHA-256;
- the sealed ten-family exact-inner evidence bundle;
- one path/SHA-256 per family artifact and model-fit audit;
- the separate direct numerical manifest or an explicit zero-signal reason;
- input dataset projection, code, configuration, environment, and model identities;
- an attestation that no remote discovery/extraction client and no oracle reader was
  constructed during Stage 1.

The cumulative-spent producers and artifact-revalidating reload paths cover
10/10 families. The builder now invokes them for every canonical hierarchy
epoch, writes the distinct root-registered cumulative graph, and seals every
scope catalog, proof, typed family artifact, and nested descriptor/model/source
registration. Exact-inner proofs cannot be relabeled or copied into that graph.
Compatibility handoffs remain projections of the validated composite bundle;
they are not independent evidence authorities.

The isolated offline frozen embedding-cache producer is implemented and wired
into the arbitrary-cohort CLI, with 57/57 focused builder/validator tests
passing. Its exact public API is:

```python
build_production_embedding_cache(
    *,
    dataset_path: Path | str,
    text_column: str,
    local_model_path: Path | str,
    sentence_model_name: str,
    chunk_configuration: Mapping[str, Any],
    target_dir: Path | str,
    device: str | None = None,
    batch_size: int = 32,
) -> ProductionEmbeddingCacheBuildResult
```

The logical `sentence_model_name` remains separate from the authenticated
absolute local model path/tree. The CLI can either validate an existing
production cache or atomically build a fresh one from a local, symlink-free
model tree. A genuine one-shot E2E remains pending, so neither 10/10 component
coverage nor the cache suite is by itself a production-readiness claim.

Before model authentication or encoding, the v2 producer computes the uncapped
chunk count for every row and fails closed unless the configured `max_chunks`
cap is nonbinding. Its closed v2 metadata and provenance persist the uncapped
total and SHA-256 hash of the uncapped per-row count vector, assert
`chunk_cap_nonbinding=true`, and set `semantic_truncation_allowed=false`;
semantic truncation is forbidden.

The v2 cache is also tokenizer-verified nontruncating. After constructing the
local encoder but before the first `encoder.encode`, the producer requires a
positive effective maximum sequence length and audits every ordered raw chunk
with the tokenizer using `truncation=False`; any token count above the limit
fails closed. Metadata and provenance persist `max_observed_token_count`,
`ordered_token_counts_sha256`, and `tokenizer_truncation_allowed=false`, with
`chunking_mode=whitespace_word_chunks_tokenizer_verified_nontruncating_v2`.

The bundle request is now `production_all_evidence_stage1_request_v5`. Its
closed `production_stage1_htr_input_nontruncation_audit_v1` applies to
`htr_neural` and the HTR subproducer of `matched_pair_uplift`. During
preparation, before an embedding cache is built or loaded and before any Stage 1
training begins, the wrapper computes every row's uncapped word-chunk count and
requires configured `htr_max_chunks` to be nonbinding. It then loads the same
authenticated local-only tokenizer used by the HTR runtime (`BertTokenizer` for
the production legacy-BERT tree) and tokenizes every ordered exact HTR chunk
with `padding=False` and `truncation=False`.

The audit derives `effective_max_chunk_length` as the smaller of configured
`htr_max_chunk_length` and the model/tokenizer sequence limit when one is
available, otherwise as the configured limit, and fails before cache work or
training on any overflow. The request persists
`normalized_text_projection_sha256`, `ordered_chunk_counts_sha256`,
`ordered_token_counts_sha256`, `max_observed_token_count`, tokenizer and model
identities, and the authenticated nontruncation flags
`chunk_cap_nonbinding=true`,
`all_chunks_within_effective_max_length=true`,
`semantic_truncation_allowed=false`, and
`tokenizer_truncation_allowed=false`, all under the audit's own
`content_sha256`.

Both HTR runtime tokenizer sites independently set `padding=False` and
`truncation=False` and explicitly reject sequences above `max_chunk_length`.
Batch collation pads only to its longest member, so
`max_chunk_length=512` raises the admissible ceiling without padding every chunk
to 512 tokens.

### 6. Upgrade consumers before enabling the wrapper — implemented

The all-evidence loader requires the new bundle manifest for production inputs
and validates it against the same authoritative registry before reading any
inner recurrence evidence. A v1 inner row with only `n_rows`, `heldout_rows`, and a
fold number must be rejected. Full-outer historical artifacts may remain readable
only in an explicitly named historical/ablation mode that cannot satisfy the
production exact-inner requirement.

The loader projects each family payload into the architecture-local catalog
interface. It does not concatenate raw family payloads. Discovery processes one
lossless architecture dossier at a time, then performs compact authenticated
integration across completed dossiers. ID-addressed lookback can return to the
retained raw evidence during reconsideration. Paging and recursive folding do
not sample, globally rank, cap, or truncate support, and no prompt contains the
raw evidence from all architectures.

The arbitrary-cohort request and root manifest must also authenticate the exact
hierarchy implementation that consumes those catalogs. A family-complete Stage 1
bundle is not compatible with an older discovery implementation that dumps all raw
families into one prompt or uses repeatable enum arrays for exact coverage. The
current wrapper pins the imported keyed hierarchy and adaptive bundles and rejects
such protocol drift independently of the native-proof readiness gate.

### 7. Keep approval policy outside the scientific artifact

Bundle and job digests remain useful for integrity, replay, and audit. The
production arbitrary-cohort invocation authenticates and carries them internally;
it does not ask an end user to inspect, approve, or type a digest. Its exact
same-process one-shot capability binds the concrete runner and runtime objects and
accepts no caller replay registration. The separate historical benchmark CLI may
retain its operator approval ceremony without changing production behavior.

## Required acceptance tests

- all ten adapters receive exactly the same ordered fit and held-out rows;
- no adapter request exposes held-out treatment/outcome, oracle, secret, or
  unrelated dataset fields;
- replacing one family with its full-outer payload fails;
- changing one split row, row order, producer identity, fit audit, payload byte, or
  family order fails;
- missing, duplicate, or zero-evidence families fail;
- cache replay from a broader or different scope fails;
- a family-specific internal nested fold cannot masquerade as the canonical inner
  fold;
- no v1 count-only exact-inner handoff is accepted in production mode;
- direct numerical manifests are authenticated separately and never enter a
  concept-grounding prompt;
- oracle evaluation remains impossible until the frozen prediction hash exists.

The retained nested TF-IDF coverage in
`tests/test_tfidf_nested_calibration_production.py` checks rejection of
label-leaky metadata, held-out-label permutation invariance, and the Stage 1 to
Stage 2 validator round trip. Topic/orphan proofs additionally require the
registered held-out projection
to be exactly row ID plus the configuration-bound text column and bind the raw
score-selection JSON as their source artifact. Embedding tests additionally
recompute canonical split/data/label lineage and reject treatment, outcome,
pseudo-target, or residual drift. These exact-inner tests intentionally do not
self-certify production readiness. Cumulative-spent implementation/reload is
10/10, the authenticated all-ten root graph and both production CLIs are
implemented, and the remaining release gate is a genuine no-approval one-shot
E2E run plus independent review of its sealed result.
