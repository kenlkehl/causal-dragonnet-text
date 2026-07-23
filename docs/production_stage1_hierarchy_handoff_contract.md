# Production Stage 1 hierarchy handoff contract

Status: authenticated interface, direct prefit-catalog consumer, and typed
one-shot execution seam are implemented, with a non-bypassable native-proof
validation gate. All ten exact-inner registrations are present, including the
canonical fit-label lineage checks and negative drift canaries for embedding,
legacy, neural-query, and nested TF-IDF registrations. Genuine cumulative-spent
component implementation and persisted reload are also complete for all ten
families (10/10). The arbitrary-cohort root graph, strict loader, catalog
provider, and no-approval one-shot CLI are implemented. Candidate catalog
serving and same-process execution are enabled by the native-proof validation
substrate; final readiness remains false until a genuine one-shot E2E completes
and is independently reviewed.

## Why this handoff exists

The hierarchical discovery path cannot reconstruct its inputs by passing the
legacy, TF-IDF, and neural-query handoffs back through the historical review
provider. That provider creates and refits an independent accumulated-spent
schedule. Even if every source path is authenticated, the resulting catalogs
are not the Stage 1 scopes registered by the arbitrary-cohort wrapper.

The production handoff therefore requires Stage 1 to fit and persist every
cumulative-spent scope that hierarchical review will consume. Each scope has:

- one lossless role-neutral catalog with nonzero evidence from all ten active
  Stage 1 architecture families;
- one closed all-ten proof bundle;
- per-family component-emitted execution records plus closed descriptors for
  nested model/source artifacts; and
- exact spent-row labels plus sealed row IDs in its input binding. Sealed text,
  treatment, and outcome are unavailable to producers.

`stage1_cumulative_spent_evidence.py` is the typed component boundary for those
fits. It invokes all ten producers on a common authenticated scope. The request
contains spent-row text/treatment/outcome and sealed integer IDs only; there is
no field capable of carrying sealed text or labels. It rejects missing families,
identity changes, scope drift, forbidden identifiers in evidence, and any fit
audit claiming sealed-row access before a catalog or hierarchy index can be
assembled.

The hierarchy interprets each architecture separately and builds one lossless
architecture dossier at a time. Only after all architecture-local dossiers are
complete does compact cross-architecture integration run. Rejection
reconsideration and extraction can follow authenticated IDs back to the retained
raw support, reviewing every exact support item on one-raw-item pages and
recursively folding all page judgments. Paging and configured lookback structure
access; they never sample, cap, globally rank, or truncate support. A prompt
containing all raw architecture evidence is forbidden.

The request also contains one imported, content-addressed production hierarchy
contract identity. It authenticates the exact interface/normalizer, keyed dynamic
response contract, base hierarchy implementation bundle, standalone cache,
transport, approved agent/batch modules, adaptive implementation bundle, and the
frozen-review materializer policy used to create disjoint accepted/rejected/
planner-only review scopes. It also pins the exact spent-evidence provider/cache
generation whose semantic-retrieval and HTR fallback projections use exhaustive
vocabularies with no default term cap. The
identity SHA is repeated in the root manifest, cumulative-spent contract and
index, provider identity, handoff, and preparation input binding. Consumers
recompute it from current local source and reject missing, stale, array-form, or
flat-dump contracts before a remote client can be constructed.

## Three distinct fold domains

Three counts happen to be small integers, but they are not interchangeable.

| Domain | Authenticated field | Purpose |
|---|---|---|
| Hierarchy spent schedule | `canonical_hierarchy_partition_count = review_rounds + 3` | Three canonical partitions are initially spent. One additional canonical gate is consumed after each review round. Two rounds therefore require five partitions. |
| Interaction cross-fitting | `interaction_inner_folds` | Cross-fitting for interaction features and the final effect estimator. It neither creates hierarchy review partitions nor selects TF-IDF evidence. |
| TF-IDF training-scope policy | `tfidf_nested_calibration_folds` | Topic and orphan-n-gram selection occurs wholly inside each already registered Stage 1 fit scope. Semantic retrieval is deterministic and exhaustive: the same training-only partitions are replay/stability canaries, never selectors, and access no labels. No path reuses hierarchy partitions or interaction folds. |

The request, hierarchy index, all-ten proof bundle, and authenticated provider
identity bind these domains independently. A consumer that changes either the
interaction or TF-IDF count fails authentication even when the numeric value of
another domain happens to match it.

Each of `tfidf_semantic_retrieval_contrasts`, `tfidf_topics`, and
`tfidf_orphan_ngrams` must bind a truthful native training-scope policy record.
Topics and orphan n-grams identify the configured/effective fold count, selected
fold, disjoint model/calibration rows, and freeze-before-sealed-transform
attestations. Semantic retrieval instead records
`selection_kind=none_deterministic_exhaustive` and
`nested_calibration_applicability=no_label_or_hyperparameter_selection`; its
disjoint partitions are authenticated replay canaries only. They cannot select
or drop terms, their label-access flag is false, and both vocabulary and output
caps are null. The registered sealed text, treatment, and outcome never enter
any of these paths.

## Canonical schedule

`CanonicalHierarchySpentSchedule` derives partitions only from
`CanonicalStage1SplitRegistry`. It requires exactly
`inner_fold_count == review_rounds + 3`.

For context epoch `r`, the spent set is canonical partitions
`1 .. 3 + r`; the still-sealed set is the remainder. The schedule preserves
registry row order and authenticates both row-order fingerprints and the split
fingerprint. There is no wrapper-local KFold, random repartition, or schedule
reconstruction.

## Authenticated graph

The immutable build request declares
`production_stage1_hierarchy_spent_contract_v2`. The root manifest registers a
`production_stage1_hierarchy_spent_index_v2`, which transitively registers every
scope catalog, proof bundle, execution record, and native model descriptor by
relative path, size, and SHA-256. Each closed
`production_stage1_cumulative_native_model_descriptor_v1` in turn authenticates
the native model and source registrations plus the complete cumulative fit
audit. It also binds the canonical exact-inner evidence index.

`load_production_stage1_hierarchy_handoff` first authenticates the complete
Stage 1 bundle, then validates this hierarchy graph and constructs
`AuthenticatedProductionStage1HierarchyProvider`. At the schema boundary, the
provider:

- returns review assignments from the canonical registry;
- anchors the bundle root with a directory descriptor and opens every path
  component relative to that descriptor with symlink following disabled;
- reads each registered JSON file into one immutable byte snapshot and parses
  those exact authenticated bytes with duplicate keys rejected, rather than
  hashing a path and reopening it;
- carries the authenticated root-manifest, request, split-registry, and index
  snapshots into the handoff instead of reopening their public paths;
- authenticates the exact prefit catalog registered for the requested
  cumulative-spent scope;
- rehashes the registered graph when used;
- verifies the runtime spent text/treatment/outcome projection; and
- raises if the historical raw-input/refit method is called.

Catalog serving reconstructs every family proof with the genuine native
binders and validates its metadata, execution-record schema, code/config
identity, typed family artifact, and nested model/source artifacts. This
implemented boundary is represented by
`NATIVE_PROOF_VALIDATION_SUBSTRATE_READY=True`. The separate
`GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY=False` records only that a
genuine one-shot cohort E2E has not yet been certified. Neither is an operator
option, and the certification flag does not bypass or weaken proof validation.

`AuthenticatedStage1HierarchyInputs.compatibility_cli_arguments()` is retained
for diagnostics only. `hierarchy_cli_arguments()` deliberately raises because
authenticated historical paths are not an authenticated cumulative-spent
handoff.

## Remaining execution-certification gate

The interface and consumer must not be mistaken for a production-ready wrapper.
All ten cumulative-spent families now have implementations that accept the
canonical request, enforce the spent/sealed boundary, perform the genuine fit,
persist component-owned execution/model/source evidence, and reload through
artifact-revalidating producers. Topic and orphan selection use nested
training-only calibration; semantic retrieval remains deterministic,
exhaustive, and nonselecting. The component implementation/reload count is
10/10.

Exact-inner and cumulative-spent proofs remain distinct; neither can be relabeled
as the other. The arbitrary-cohort wrapper now invokes the 10/10 cumulative
producers for every canonical epoch, assembles the lossless all-ten catalogs,
and root-registers and reauthenticates every scope proof and nested
descriptor/model/source artifact. The production one-shot CLI constructs the
authenticated handoff and concrete runner, accepts no digest/approval/replay
argument, and calls the same-process internal one-shot seam. The remaining gate
is a genuine arbitrary-cohort E2E and independent review of its result;
production certification remains false until then.

The production wrapper accepts one explicitly supplied canonical HTTP(S)
OpenAI-compatible endpoint and one exact served-model name per invocation.
Those values are bound into a single-endpoint runner identity; pools,
fallbacks, comma-separated URLs, credentials, queries, fragments, and model
autodiscovery are rejected. Localhost is valid when it is the intentional
endpoint. The current Camus profile explicitly supplies
`http://camus:8010/v1` and
`RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic`; neither value is a generic-wrapper
constant.

No served-deployment identity JSON, container attestation, or compiled
deployment digest is required or accepted as execution authority. Before the
full run, `scripts/canary_production_stage1_hierarchy.py` loads the validated
Stage 1 bundle, prepares the ordinary hierarchy without a remote call, selects
the deterministic smallest real architecture-pure initial job, and runs only
that job with transport retries disabled and at most one schema repair. Every
initial, invalid, and repair response must report the exact requested model and
`finish_reason=stop` before semantic validation or a cache write. The resulting
runtime report is operational evidence, not an authorization token.

The hardened served-deployment collector remains available as optional static
certification tooling. Its model/server/container/package/listener evidence may
be archived when available, but cannot authorize, substitute for, or block the
required live endpoint/model checks.

The runner consumes the exact same handoff provider object as both the canonical
review-partition authority and prefit-catalog source, and rejects the historical
compatibility refit route. The one-shot seam executes only after the complete
root graph and mutable external Stage 1 runtime inputs authenticate.

## Frozen embedding-cache producer

The isolated offline cache producer is implemented, wired into the bundle CLI,
and its focused builder/validator suite passes 57/57 tests. Its exact public API
is:

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

The logical `sentence_model_name` is bound separately from the absolute local
model path/tree hash. Existing-cache validation and fresh offline cache
publication are both wired into the bundle CLI. The genuine E2E remains
pending.

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

## Global HTR input no-truncation contract

The bundle request is `production_all_evidence_stage1_request_v5`. It carries
the closed `production_stage1_htr_input_nontruncation_audit_v1`, which applies
to `htr_neural` and the HTR subproducer of `matched_pair_uplift`. During
preparation, before an embedding cache is built or loaded and before any Stage 1
training begins, the wrapper computes every row's uncapped word-chunk count and
requires configured `htr_max_chunks` to be nonbinding. It then loads the same
authenticated local-only tokenizer used by the HTR runtime (`BertTokenizer` for
the production legacy-BERT tree) and tokenizes every ordered exact HTR chunk
with `padding=False` and `truncation=False`.

The effective token ceiling is the smaller of configured
`htr_max_chunk_length` and the model/tokenizer sequence limit when one is
available; otherwise it is the configured limit. Any overflow fails before
cache work or training. The request persists
`normalized_text_projection_sha256`, `ordered_chunk_counts_sha256`,
`ordered_token_counts_sha256`, `max_observed_token_count`, tokenizer and model
identities, and the authenticated policy flags
`chunk_cap_nonbinding=true`,
`all_chunks_within_effective_max_length=true`,
`semantic_truncation_allowed=false`, and
`tokenizer_truncation_allowed=false`, all under the audit's own
`content_sha256`.

Both HTR runtime tokenizer sites independently use `padding=False` and
`truncation=False` and explicitly reject a sequence above `max_chunk_length`.
Collation pads only to the longest sequence in the current batch. Thus,
`max_chunk_length=512` sets an admissible ceiling without padding every chunk to
512 tokens.

## Digest handling

Preparation and low-level execution retain the exact batch-digest check.
The internal preparation seam accepts only the exact concrete, process-local
`PreparedHierarchicalDiscoveryBatch` instance issued by preparation. It rejects
caller mappings and bare expected digests, consumes that capability once, and
authenticates the input-manifest and batch-packet wrappers from descriptor-
anchored byte snapshots with both wrapper schema versions pinned.

The production seam accepts no caller replay registrations or replay arguments.
It binds the exact same-process runner and its hierarchy runner/config/policy,
spent-evidence and extraction providers, review agent, TF-IDF policy validator,
coordinator, precommit, and canonical unbound coordinator method. Before both
authorization and execution it reauthenticates those object identities plus the
current cohort, legacy/TF-IDF handoff, split, candidate/query, and orphan
artifact hashes. Object substitution, instance method overrides, or scientific
input mutation therefore invalidate the capability.

`run_internal_production_stage1_hierarchy_one_shot` now prepares the concrete
runner batch, creates the provider-bound capability, converts it to the exact
typed authorization, and passes both directly into the same runner invocation.
The runner rejects a copied batch, a mapping lookalike, a caller-supplied digest,
caller-supplied replay registration, substituted runtime object, noncanonical
coordinator execution, wrong result type, or authorization reuse. A serialized
authorization or subprocess adapter is not executable authority.
`internal_hierarchy_execution_authorization` requires the implemented native-
proof validation substrate and manufactures no authority from serialized
claims. `scripts/run_production_stage1_hierarchy_one_shot.py` carries the digest
internally under the single authorized cohort invocation; no end user is asked
to inspect, approve, or type it. A successful candidate run still records
`genuine_one_shot_e2e_certified=false` until its independent certification gate
is completed.
