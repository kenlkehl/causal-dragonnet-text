# Cumulative-spent native Stage 1 adapter matrix

Status: genuine cumulative-spent implementation and persisted reload are
complete for all ten architecture families (10/10): four legacy families,
three shared embedding/semantic families, topic TF-IDF, orphan TF-IDF, and
neural query. The arbitrary-cohort wrapper now integrates and authenticates the
unified all-ten root scope graph, and both the bundle CLI and no-approval
one-shot CLI are implemented. This is still not a readiness declaration: a
genuine one-shot E2E run and independent review of its sealed result remain
pending.

## Non-negotiable boundary

Each producer is invoked with `CumulativeSpentStage1FamilyRequest`. It receives
text, treatment, and outcome for the ordered spent rows, and integer IDs only
for sealed rows. A component may never recover sealed text from the cohort or a
cache.

The exact-inner BoW, HTR, and matched-pair capture sinks require a non-empty
text transform to replay their fitted state. For a cumulative scope, that
transform is a deterministic alias of one already-spent text. The alias is not
a cohort row, carries no labels, and contributes neither evidence nor numerical
features. Its binding is persisted in the component execution record. This is
different from a registered held-out transform and must never be described as
one.

`stage1_cumulative_spent_native_adapters.py` implements this replay-canary
binding and request-bound execution-record/model/source replay for the four
legacy families. The existing exact-inner behavior is unchanged.

`stage1_cumulative_spent_embedding_adapters.py` now consumes a fresh live
spent-bound embedding capture, emits three opaque same-process family results,
and independently reloads the persisted capture to regenerate each payload,
count, identity, execution record, and fit audit. The production bundle writes
and revalidates the corresponding closed three-family cumulative index.

`stage1_cumulative_spent_remaining_adapters.py` implements persisted topic and
orphan TF-IDF producers with nested training-only calibration and the
neural-query producer with its authenticated local execution/source/model
artifacts. Reload reconstructs request-bound producers and revalidates their
artifacts on use. Together, the component implementation/reload count is
10/10. Root registration is now complete, but the still-pending E2E
certification gate cannot be inferred from that count.

## Component implementation matrix

| Family | Genuine cumulative fit and artifacts | Component implementation/reload status |
|---|---|---|
| `bow_nuisance` | Run the existing BoW nuisance cross-fits and full importance fits on all spent rows. Capture vectorizers, learners, fold partitions, OOF outputs, and the spent replay canary in JSON/NPZ. | Implemented and revalidated in the closed four-family cumulative legacy index. |
| `bow_r_loss` | Reuse the same BoW capture, including exact ensemble nuisance, residual, pseudo-target, weight, and both R-loss objectives fitted only on spent rows. | Implemented as a distinct payload/record in the closed four-family cumulative legacy index. |
| `htr_neural` | Run the existing nested HTR nuisance/effect fits on spent rows, retain tensor state and nested calibrators in JSON/NPZ, authenticate the local model tree, and replay the spent canary. | Implemented and revalidated against the authenticated local HTR model tree and the global HTR no-truncation audit. |
| `matched_pair_uplift` | Run both native subproducers—BoW offset/Ridge and HTR pair network—against the same spent-only nuisance vectors and deterministic pair builder. Capture both branches and replay the spent canary. | Implemented with complete BoW+HTR subproducer capture and replay; its HTR branch is covered by the same global no-truncation audit. |
| `embedding_whole_cohort` | Fit supervised directions only on spent labels using the spent-bound frozen embedding provider. Its existing capture already accepts sealed IDs without sealed text. Persist all uncapped concept members and numerical fit state. | Implemented from a fresh live capture and independently regenerated during closed-index reload. |
| `embedding_clustered` | Reuse the same spent-bound embedding capture and genuine cluster-local contrasts, with complete cluster/member evidence. | Implemented as a distinct lossless family payload from the shared capture. |
| `tfidf_semantic_retrieval_contrasts` | Derive the exhaustive label-free TF-IDF projection after spent-only supervised embedding directions are frozen. Training-only partitions are deterministic replay/stability canaries, not selectors. | Implemented with exhaustive no-selection policy, null caps, false label access, and source-derived payload replay. |
| `tfidf_topics` | Fit TF-IDF/NMF on all spent rows. Perform label-based model selection in disjoint nested model/calibration partitions wholly inside the spent scope, freeze selection, and use only the spent replay canary for transform replay. | Implemented and revalidated with cumulative fitted-context, selection, execution, source, and policy artifacts. Hierarchy partitions and interaction folds are not reused as calibration folds. |
| `tfidf_orphan_ngrams` | Reuse the topic context but perform orphan selection/deduplication from training-only score artifacts and nested calibration. Preserve every selected cluster member and alias in the family payload. | Implemented as a distinct persisted family record bound to the genuine fitted context and score-selection artifacts. No prompt compactor or global top-k intervenes. |
| `neural_query_moments` | Use `ContextFitNeuralQueryService.discovery_for_context` on spent rows only, persist its trusted learned-query/tensor snapshot, and emit safe lexical witnesses plus aggregate fit-side moments. | Implemented with a persisted cumulative execution record/binder that never transforms sealed text. Any later gate transform belongs to the separately frozen numerical channel. |

## Lossless hierarchy policy

Evidence discovery is architecture-at-a-time. Each architecture gets a lossless
dossier over all of its authenticated support. Only after those dossiers are
complete does the hierarchy perform compact cross-architecture integration.
The compact integration carries authenticated dossier identities and can use
ID-addressed lookback to the retained raw evidence during reconsideration. It
does not concatenate all raw architecture evidence into one prompt. Paging and
recursive folding structure the complete support; they do not sample, cap,
apply a global top-k, or otherwise truncate it.

## Remaining production integration order

1. Component capture, persistence, and reload for all ten families — complete.
2. For every canonical hierarchy epoch, invoke those component producers and
   build one lossless all-ten catalog through the typed cumulative boundary —
   complete in the arbitrary-cohort builder.
3. Root-register and reauthenticate every descriptor, model/source artifact,
   scope catalog, and proof bundle in the arbitrary-cohort output — complete.
4. Expose and validate the no-approval production CLI path — complete; the
   bundle suite passes 39/39 tests, and the hierarchy-loader, one-shot, and
   security suites pass 100/100 tests combined.
5. Run the genuine arbitrary-cohort one-shot hierarchy E2E — pending.

## Frozen embedding-cache prerequisite

The isolated offline cache producer is implemented and its focused
builder/validator suite passes 57/57 tests. Its exact public API is:

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

`sentence_model_name` is the required logical configuration identity;
`local_model_path` is the separate absolute offline model tree whose bytes are
authenticated. The root wrapper can atomically build this cache with
`--embedding-cache-output-dir` plus `--embedding-local-model-path`, or validate
an existing cache passed with `--embedding-cache-dir`. Exercising the complete
path in the genuine E2E remains pending. Production users are never asked to
approve or type artifact digests; hashes are internal integrity identities.

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

## Global HTR input no-truncation prerequisite

The bundle request is `production_all_evidence_stage1_request_v5`. It carries
the closed `production_stage1_htr_input_nontruncation_audit_v1`, which applies
to `htr_neural` and the HTR subproducer of `matched_pair_uplift`. During
preparation, before an embedding cache is built or loaded and before any Stage 1
training begins, the wrapper computes every row's uncapped word-chunk count and
requires configured `htr_max_chunks` to be nonbinding. It then loads the same
authenticated local-only tokenizer used by the HTR runtime (`BertTokenizer` for
the production legacy-BERT tree) and tokenizes every ordered exact HTR chunk
with `padding=False` and `truncation=False`.

The audit derives `effective_max_chunk_length` as the smaller of configured
`htr_max_chunk_length` and the model/tokenizer sequence limit when one exists,
otherwise as the configured limit, and fails before cache work or training on
any overflow. The request persists `normalized_text_projection_sha256`,
`ordered_chunk_counts_sha256`, `ordered_token_counts_sha256`,
`max_observed_token_count`, tokenizer and model identities, and the authenticated
nontruncation flags `chunk_cap_nonbinding=true`,
`all_chunks_within_effective_max_length=true`,
`semantic_truncation_allowed=false`, and
`tokenizer_truncation_allowed=false`, all under the audit's own
`content_sha256`.

Both HTR runtime tokenizer sites independently set `padding=False` and
`truncation=False` and explicitly reject sequences above `max_chunk_length`.
Dynamic batch collation pads only to its longest member; therefore,
`max_chunk_length=512` raises the admissible ceiling without padding every chunk
to 512 tokens.

The historical review-spent provider remains useful as a compatibility and
diagnostic implementation, but wrapping its final catalog in new hashes is not
native cumulative proof and cannot open production serving.
