# Production all-evidence Stage 1 bundle runbook

Status: executable arbitrary-cohort candidate builder with all ten exact-inner
registrations, all ten cumulative-spent implementation/reload paths, the sealed
all-ten root graph, strict cache/loader validation, and a no-approval hierarchy
one-shot CLI. The cache and bundle suites pass 57/57 and 39/39 tests; the
hierarchy-loader, one-shot, and security suites pass 100/100 tests combined. A
genuine cohort E2E and independent review of its result remain pending, so this
is not yet a final production-readiness declaration.

The authoritative hierarchy boundary is documented in
[`production_stage1_hierarchy_handoff_contract.md`](production_stage1_hierarchy_handoff_contract.md).

## Current release state

The ten existing architectures have native
`ExactInnerStage1FamilyProducer` adapter implementations in
`stage1_exact_inner_family_adapters.py`. They bind the real native APIs, code and
configuration identities, exact ordered rows, projected data, payload bytes,
native fit metadata, execution record, and model/source artifact hashes.
Exact-inner scopes can only be constructed from a validated persisted catalog,
and the comparison hashes used by the clone canary can only be constructed from
a validated persisted full-outer catalog. Catalog, fit-metadata, execution,
model, and source bytes are rehashed when producers are bound and invoked so a
post-binding mutation fails rather than relying on a stale digest. All three
TF-IDF family proof contracts bind a truthful training-scope policy. Topic and
orphan n-gram bind native nested training-only selection; semantic retrieval
binds deterministic exhaustive no-selection replay canaries. The production
TF-IDF component genuinely registers the topic and
orphan-n-gram adapters for every canonical exact-inner context. Those
registrations bind the native `context_metadata.json`,
`fitted_context.joblib`, and `topic_score_tests.json`, persist the
catalog-derived family payload and immutable execution record, and are covered
by the sealed component inventory. The orphan catalog projection mechanically
retains every fit-side cluster term and nested alias while leaving calibration
statistics in the authenticated raw score artifact. Semantic retrieval, both
embedding families, HTR neural, and both matched-pair subproducers now also have
direct exact-inner registration paths. Neural-query moments has a direct
registration: the component persists its trusted in-memory final
query and training-activation arrays as non-executable NPZ plus closed JSON,
persists row-aligned held-out moment arrays, and binds the exact fit inputs,
fit-e/fit-m outputs, catalog payload, execution record, and proof index without
retaining or reloading its temporary joblib cache.

The paired BoW nuisance and R-loss registrations are now genuine as well. In
proof mode, the native runner captures every fitted per-view/per-fold
`TfidfVectorizer` and learner, plus the full-fit treatment, outcome, and
weighted-R importance learners, as numerical NPZ arrays and closed JSON only.
Validation reconstructs the vectorizer transform and replays every validation
and registered held-out prediction from that state, checks the exact fold row
partitions, nuisance/residual/pseudo-target/weight identities, objectives,
classes, seeds, clipping configuration, and output hashes, and rejects missing
folds, changed objectives, tampering, or any held-out-label access. No BoW
pickle or joblib file is retained or loaded by this proof path.

The all-ten registration count is therefore 10/10, with no modeling
architecture omitted. Registration count alone is not production readiness.
The embedding proof path now recomputes its canonical split and modeling-data
projection, binds canonical treatment/outcome at capture and registration, and
compares native registered fit outputs with generator treatment, outcome,
pseudo-outcome, and residual arrays. Focused negative canaries reject drift in
each of those arrays and strict-envelope or duplicate-key tampering.

A follow-on lineage audit found and repaired the analogous issue in the legacy
and neural-query captures: their registration boundaries now compare captured
fit treatment/outcome to the canonical modeling-data/request values. Focused
negative canaries reject arbitrary treatment or outcome drift for BoW, HTR,
matched-pair, neural-query, embedding, and nested TF-IDF registrations. The
10/10 exact-inner registration count therefore includes completed canonical
label-lineage validation; it does not by itself certify the cumulative graph or
the final one-shot E2E.

The earlier TF-IDF label-boundary mismatch is resolved in the implementation.
Production forces deterministic nested calibration wholly inside each
registered fit partition where model selection exists (topics and orphan
n-grams). Selection is frozen before the registered held-out text is transformed,
and registered held-out treatment/outcome are never projected into the fitter.
Semantic retrieval remains a label-free deterministic exhaustive projection
after its supervised fit-scope directions are frozen. Its training-only
partitions are replay/stability canaries only: they neither select nor drop
terms, access no labels, and impose no vocabulary or output cap.

The builder can now execute a complete candidate Stage 1 run. It fails closed
on any missing family, provenance mismatch, partial component, cache drift, or
root-graph inconsistency. Automation must not interpret either preflight or a
completed candidate bundle as final E2E certification. There is no readiness
override, and compatibility handoffs are never an alternative evidence
authority.

Genuine native cumulative-spent component implementation, persistence, and
artifact-revalidating reload are now complete for all ten families (10/10).
The four legacy families, three embedding/semantic families, topic TF-IDF,
orphan TF-IDF, and neural-query moments all have request-bound producers.
Topic/orphan selection uses nested training-only calibration; semantic retrieval
is deterministic, exhaustive, and nonselecting.

The integration work is implemented: every canonical hierarchy epoch invokes
the typed cumulative producers, assembles a lossless all-ten catalog,
root-registers and reauthenticates every proof and nested
descriptor/model/source artifact, and is consumed by the no-approval one-shot
CLI. A catalog or execution JSON assembled later by unauthenticated glue is not
a substitute for a component-emitted record, and an exact-inner record cannot
be relabeled as cumulative-spent. Authenticating the old handoff paths and then
re-fitting through the historical review provider is not a compatibility
solution. The remaining release action is the genuine E2E run and independent
review of its sealed output.

The integrated path provides the following fail-closed hardening:

- the legacy concept projection does not use the prompt-oriented compactor, and
  each scope has an immutable authenticated raw sidecar for drill-back;
- dataset and config identities are checked before and after parsing and are
  rechecked, together with source code, dependency, HTR-tree, and
  embedding-cache identities, before the bundle is sealed;
- the hierarchy loader anchors every bundle, component, embedding-cache, and
  preparation path with directory descriptors, disables symlink traversal for
  every component, rejects duplicate JSON keys, and carries retained manifest
  bytes into the handoff instead of reopening the manifest path;
- the request binds the complete in-repository Python behavior surface,
  packaging/lock metadata, installed distribution versions, and relevant
  runtime identities;
- partial TF-IDF checkpoint reuse is prohibited; only a fully sealed component
  can be reused; and
- exact-inner TF-IDF topic and orphan-n-gram fits emit an immutable native
  family-proof index bound to their real fitted-context and score-selection
  artifacts; and
- neural-query-moment fits emit an immutable trusted-array snapshot, row-aligned
  held-out moments, and a native family-proof index without retaining executable
  joblib state; and
- paired BoW nuisance/R-loss fits emit replayable non-executable vectorizer and
  learner state, full-fit objective metadata, and an immutable native
  family-proof index for every canonical exact-inner scope; and
- all three production TF-IDF family paths persist a truthful fit-only policy:
  topics/orphans bind nested calibration, while semantic retrieval binds
  uncapped exhaustive no-selection replay canaries; all reject any claim that
  registered held-out labels were accessed;
- all ten adapters reject split/data/payload/code/config/artifact drift, old
  heldout-label-dependent TF-IDF metadata, and semantic clones of the
  catalog-authenticated full-outer payload; and
- the authenticated hierarchy loader validates the root/component byte graph,
  canonical exact-inner contract, every registered exact-inner family artifact,
  raw-sidecar linkage, and all-ten coverage;
- the immutable request imports and authenticates the production hierarchy rather
  than copying its response schemas: interface v10 with atomic-occurrence
  compiler normalization v3, dynamic response contract v8 with closed keyed
  exact coverage and lossless page/fold jobs, orchestrator v12 and precommit v11,
  base implementation bundle v5 and its response-attempt trace, the
  authenticated cache/runner and approved agent/batch surfaces, and adaptive
  hierarchy v7 with implementation bundle v8; and
- that hierarchy identity is carried by the root manifest, authenticated loader,
  cumulative-spent contract/index, provider, handoff, and implemented internal
  execution authorization. An old flat all-architecture evidence dump, mixed-
  family chunking, global top-k, sampled/capped raw support, or exact-coverage
  array is rejected instead of being treated as a compatibility path.

This hardening opens candidate construction and execution, not final
certification. Cumulative-spent implementation/reload is 10/10, the local
frozen-cache producer is integrated, and every cumulative scope catalog/proof
graph is root registered. The remaining release blocker is one genuine
arbitrary-cohort one-shot E2E plus independent review. Installed dependency
names/versions are bound, but their installed Python/native bytes are not yet
content-addressed; record the exact environment for every candidate run.

The strict hierarchy handoff loader and canonical provider authenticate the root
graph, recheck registered bytes on use, reject the historical independent-refit
input path, and bind the runtime accumulated-spent data projection. The runner
directly consumes that provider's prefit catalog and the one-shot seam carries
the prepared digest internally through a typed, exact-object authorization.
Catalog serving reconstructs and authenticates the 10/10 native cumulative
proofs with the genuine family binders. The separate certification flag remains
false until the E2E is reviewed. The wrapper does not treat schema-valid dummy
records as component emission or over-attest candidate execution as certified
production readiness.

## Outcome

`scripts/build_all_evidence_stage1_bundle.py` is the single entry point
for turning one new cohort into the authenticated Stage 1 inputs required by the
hierarchical all-evidence path. Its preflight runs locally, makes no
language-model or other remote calls, and does not ask an operator to inspect or
approve a digest. Hashes are internal integrity, cache, and resume controls.

The builder fits and verifies all ten active concept-bearing architectures:

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

Every architecture is later interpreted separately by the hierarchical
discovery pipeline. Each architecture-local dossier is lossless over its
authenticated support. Compact cross-architecture integration happens only
after those dossiers are complete, with authenticated ID-addressed lookback to
raw evidence when reconsideration needs it. Paging and recursive folding do not
sample, globally rank, cap, or truncate support. This command never constructs
one prompt containing the raw evidence from every architecture.

## Required inputs

- A Parquet cohort with four distinct configured fields: a stable unique unit
  ID, text, binary treatment, and observed binary outcome. Extra columns are not
  materialized into Stage 1; the Parquet container bytes are hashed for input
  identity. Accordingly, the security claim is that extra/oracle columns are
  not decoded or materialized—not that their container bytes are never read.
- Enough support for every canonical outer fit, exact-inner fit, and its
  deterministic nested TF-IDF model/calibration partition to contain both
  treatment arms and both binary outcome classes. Preflight constructs these
  partitions and rejects a small or imbalanced cohort before fitting.
- A JSON or YAML Stage 1 configuration whose `model_type` is
  `multi_model_forest`. BoW, HTR, embedding contrast, BoW+HTR matched-pair
  uplift, clustered embeddings, honest candidate consistency, TF-IDF score
  testing, TF-IDF topics, and orphan n-grams must all be enabled. The HTR
  sentence encoder must be unfrozen.
- A local HTR sentence-model directory named by the config. The complete model
  tree is authenticated and copied to a private read-only temporary snapshot
  for fitting.
- A frozen row-bound embedding cache containing `metadata.json`,
  `chunk_embeddings.npy`, `offsets.npy`, and `chunk_texts.jsonl`. Its row order,
  text, model/chunk configuration, and file bytes must agree with the cohort.

The isolated offline cache producer is implemented, integrated, and its focused
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

`sentence_model_name` must be the logical model identity from the Stage 1
configuration. `local_model_path` is the separate absolute local model tree;
its path and complete tree hash are retained independently. The producer is
offline, publishes a fresh four-file cache atomically, and all-row binds it
through `SpentOnlyFrozenChunkEmbeddingCache`. It is wired into the single
bundle CLI; exercising it in the genuine E2E remains pending. It never falls
back to loading a remote model.

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

### Global HTR input no-truncation contract

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
`htr_max_chunk_length` and the model/tokenizer sequence limit when the model
exposes one; otherwise it is the configured limit. Any overflow fails before
cache work or training. The request persists
`normalized_text_projection_sha256`, `ordered_chunk_counts_sha256`,
`ordered_token_counts_sha256`, `max_observed_token_count`, the tokenizer and
HTR-model identities, and the authenticated policy flags
`chunk_cap_nonbinding=true`,
`all_chunks_within_effective_max_length=true`,
`semantic_truncation_allowed=false`, and
`tokenizer_truncation_allowed=false`, all under the audit's own
`content_sha256`.

Both HTR runtime tokenizer sites independently use `padding=False` and
`truncation=False` and explicitly reject a sequence above `max_chunk_length`.
Collation pads only to the longest sequence in the current batch. Therefore,
`max_chunk_length=512` sets an admissible ceiling without padding every chunk to
512 tokens.

The current all-ten contract accepts binary outcomes only. Continuous-outcome
support remains blocked because matched-pair uplift is mandatory and has not
been established under an equivalent continuous-outcome contract.

Do not put API keys or other secrets in the Stage 1 config. Recognized secret
fields are redacted before parsing or persistence, and no remote client is
constructed.

## Build one candidate Stage 1 bundle

Side-effect-free preflight requires an existing authenticated cache:

```bash
uv run --frozen python scripts/build_all_evidence_stage1_bundle.py \
  --dataset /absolute/path/cohort.parquet \
  --stage1-config /absolute/path/stage1_config.json \
  --embedding-cache-dir /absolute/path/frozen_embedding_cache \
  --output-dir /absolute/path/new_stage1_bundle \
  --unit-id-column person_key \
  --device cuda:0 \
  --query-device cuda:0 \
  --dry-run
```

Build with an existing authenticated cache:

```bash
uv run --frozen python scripts/build_all_evidence_stage1_bundle.py \
  --dataset /absolute/path/cohort.parquet \
  --stage1-config /absolute/path/stage1_config.json \
  --embedding-cache-dir /absolute/path/frozen_embedding_cache \
  --output-dir /absolute/path/new_stage1_bundle \
  --unit-id-column person_key \
  --device cuda:0 \
  --gpu-id 0 \
  --query-device cuda:0 \
  --num-workers 1 \
  --tfidf-workers 8 \
  --tfidf-parallel-backend processes \
  --seed 42
```

Or atomically build a fresh cache and then run Stage 1 in the same invocation:

```bash
uv run --frozen python scripts/build_all_evidence_stage1_bundle.py \
  --dataset /absolute/path/cohort.parquet \
  --stage1-config /absolute/path/stage1_config.json \
  --embedding-cache-output-dir /absolute/path/new_embedding_cache \
  --embedding-local-model-path /absolute/path/symlink_free_local_model \
  --output-dir /absolute/path/new_stage1_bundle \
  --unit-id-column person_key \
  --device cuda:0 \
  --gpu-id 0 \
  --query-device cuda:0 \
  --num-workers 1 \
  --tfidf-workers 8 \
  --tfidf-parallel-backend processes \
  --seed 42
```

The cache modes are mutually exclusive. Fresh-cache mode requires both fresh
absolute output paths, cannot be combined with `--dry-run` or `--resume`, and
never follows a model-tree symlink. Resume accepts only an existing cache and
reuses only byte-verified sealed components from the identical request.

`--dry-run` authenticates and validates the cohort, config, model tree, cache,
canonical split feasibility, exact-inner contract availability, and adapter
readiness without writing outputs, using a GPU, or fitting the supervised Stage
1 estimators. It does run the frozen-cache cluster KMeans and label-conditioned
local-contrast SVD readiness calculation described below. A blocked result is
not a successful production readiness result.

Clustered-embedding readiness is an execution preflight, not a row-count
estimate. The wrapper runs the configured frozen-cache KMeans/local-contrast/SVD
path in every ordered full, exact-inner, and cumulative-spent scope and requires
both native local-contrast families to have genuine rank-two support and
exact uncapped component preservation across raw records, semantic records,
clustered catalog atoms, and their TF-IDF semantic-retrieval mirrors. Every
component must retain nonempty members and identical mirror-parent linkage.
The cache binding must also report zero token-bounded reconciliation rows. It
fails closed before bundle output or GPU model fitting if any scope is
infeasible. It never changes cluster counts or support thresholds, emits
rank-one substitutes, reconciles mismatched cache text, or drops the clustered
family.

After the bundle completes, run the generic runtime canary directly against the
one endpoint and model intended for this invocation. The canary performs the
ordinary authenticated local preparation, selects the deterministic smallest
real architecture-pure initial interpretation job, and makes only that logical
call (plus at most one schema repair):

```bash
uv run --frozen python scripts/canary_production_stage1_hierarchy.py \
  --bundle-manifest /absolute/path/new_stage1_bundle/bundle_manifest.json \
  --scratch-output-dir /absolute/path/new_canary_scratch \
  --hierarchical-preparation-dir /absolute/path/new_canary_preparation \
  --report-dir /absolute/path/new_canary_report \
  --endpoint http://camus:8010/v1 \
  --model RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic \
  --review-rounds 1 \
  --interaction-inner-folds 3 \
  --tfidf-nested-calibration-folds 3 \
  --review-stage1-device cuda:0 \
  --review-neural-query-device cuda:0
```

The Camus/Gemma values above are the current run profile, not constants embedded
in the generic wrapper. A deliberate local run may instead use, for example,
`--endpoint http://localhost:2345/v1 --model local/exact-model`. Every invocation
must supply exactly one canonical HTTP(S) OpenAI-compatible endpoint and one
explicit model name. Endpoint pools/fallbacks, comma-separated URLs,
credentials, whitespace aliases, queries, and fragments are rejected. The
runner identity must contain only that endpoint and model; autodiscovery is not
allowed.

Every initial response, schema-invalid response, and repair response must report
the exact requested model and `finish_reason=stop`. This metadata is checked at
the per-call boundary before semantic validation or an immutable cache write.
Transport retries are zero and the hierarchy permits no more than one schema
repair. The canary report is an operational result, not an authorization token,
and contains no model-identity, deployment-pin, digest-approval, prediction, or
oracle authority.

After the canary succeeds, run the hierarchy through the dedicated same-process
entry point using the same exact endpoint and model:

```bash
uv run --frozen python scripts/run_production_stage1_hierarchy_one_shot.py \
  --bundle-manifest /absolute/path/new_stage1_bundle/bundle_manifest.json \
  --output-dir /absolute/path/new_hierarchy_execution \
  --hierarchical-preparation-dir /absolute/path/new_hierarchy_preparation \
  --attestation-dir /absolute/path/new_hierarchy_execution_record \
  --endpoint http://camus:8010/v1 \
  --model RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic \
  --review-rounds 1 \
  --interaction-inner-folds 3 \
  --tfidf-nested-calibration-folds 3 \
  --review-stage1-device cuda:0 \
  --review-neural-query-device cuda:0
```

For every model-produced response, including a schema-invalid response eligible
for the single bounded repair and the repair response itself, authenticated
per-call metadata must report the exact requested served-model name and
`finish_reason=stop`. Missing or different response-model metadata and every
other finish reason fail before semantic validation or an immutable cache write.

The three output roots must be fresh, absolute, pairwise nonnested, and outside
the Stage 1 bundle. `review_rounds`, `interaction_inner_folds`, and
`tfidf_nested_calibration_folds` must equal the values sealed in the bundle
request; `candidate_consistency_inner_folds` in the Stage 1 configuration must
equal `review_rounds + 3`. The one-shot wrapper derives cohort, config, HTR,
cache, and neural-query inputs from the authenticated bundle and revalidates
them before constructing a client-capable object.

Neither production command exposes `--approve-digest`, `--approval-sha256`, or
a similar option. The Stage 1 command prints request and bundle hashes for audit
records only. The hierarchy command prepares, authorizes, and consumes its exact
same-process capability internally.

The standalone benchmark operations runbook still documents an explicit human
approval ceremony for its existing two-phase operator CLI. That is not the target
interaction for this arbitrary-cohort wrapper. The wrapper retains the exact
prepared-batch digest and low-level equality check. The implemented one-shot
provider-bound orchestrator carries it internally under the already authorized
cohort invocation. The authorization boundary accepts only the exact in-process
prepared-batch capability, consumes it once, pins both preparation wrapper
schemas, and reauthenticates the current scientific input file hashes. It binds
the exact same-process runner, providers, runtime policy/configuration objects,
coordinator, precommit, and canonical coordinator execution method. Production
accepts no caller replay registration or replay arguments. The runner accepts
and consumes the exact typed authorization object once and requires the exact
authenticated result type; a copied mapping, copied prepared batch, substituted
runtime object, or method override is not executable authority.
This policy difference does not open the current readiness gate or authorize use
of the historical compatibility/refit route.

## Fold and leakage contract

The exact-inner contract creates the sole canonical joint treatment/outcome
split registry before any architecture is fitted. The wrapper verifies and
persists that registry; the same ordered outer and exact-inner partitions are
used by the legacy all-source, TF-IDF, and neural-query components and by the
primary split artifact.

The hierarchy schedule is a separate registered use of that same partition
authority. It requires `inner_fold_count = review_rounds + 3`: exactly three
partitions are initially spent and one additional partition is consumed per
review round. `interaction_inner_folds` remains the downstream estimator's
cross-fit count. `tfidf_nested_calibration_folds` is used only inside each
already registered Stage 1 fit scope. Neither count may be repurposed as the
hierarchy partition count or as the other count, even when both are configured
to three.

The required contract for every full-outer and exact-inner scope is:

- treatment and outcome exist only on fit rows presented to each family
  producer;
- transform rows contain `_oci_row_id` and text, but no labels;
- rows outside an inner scope have blank text and no labels in that runner;
- embedding retrieval is limited to rows bound to the exact scope;
- neural-query definitions are learned independently from that scope;
- a fresh legacy runner executes the scope instead of copying full-outer
  importance, embedding, or HTR evidence into inner records; and
- the resulting lineage records bind exact ordered fit/held-out rows and the
  canonical registry hash.

Only short lexical contrast witnesses and HTR concepts cross the automatic
concept-grounding boundary. Raw retrieved chunks and attention excerpts are
retained only in authenticated prompt-hidden sidecars for ID-addressed
drill-back. Other transient runner artifacts and executable neural-query
checkpoints live in temporary directories and are deleted before a component
can be sealed. Full-outer row-aligned numerical matrices are persisted
separately for estimator use and cannot ground feature names.

Before the root bundle is sealed,
every full-outer, exact-inner, and cumulative-spent scope must pass through its
authenticated role-neutral catalog boundary. Both the atom and semantic member
counts for all ten active families must be nonzero. This coverage check is not
the catalog itself. Any missing family, truncated upstream evidence declaration,
lineage mismatch, or invalid payload aborts the build.

## Candidate bundle outputs

The output root contains:

- `immutable_build_request.json`: source paths, byte identities, effective
  secret-redacted config, code identities, runtime settings, and security
  assertions;
- `stage1_config.json`: downstream-compatible effective Stage 1 snapshot;
- `split_registry.json`: sole authoritative outer/inner partition registry;
- `row_registry.parquet`: `_oci_row_id` to operator-supplied unit ID mapping;
- `primary_predictions.parquet`: authoritative `_oci_row_id`, `outer_fold`, and
  `cv_fold` split columns (it is a split carrier, not a clinical prediction
  export);
- one content-addressed legacy component with the all-source handoff, exact
  scope index, and full-outer direct numerical matrices;
- one content-addressed TF-IDF component with registry-sealed full and exact-inner
  handoffs;
- one content-addressed neural-query component with safe per-scope evidence;
- one registered exact-inner evidence index plus an authenticated all-ten
  producer bundle for every outer/inner scope;
- one canonical cumulative-spent hierarchy index plus a lossless all-ten
  catalog and component-emitted proof bundle for every review context;
- immutable per-scope raw-evidence sidecars and separate matched-pair BoW/HTR
  output/evidence diagnostics (not the still-required producer fit/model
  proofs); and
- `bundle_manifest.json`: terminal registrations, component manifests, coverage
  counts, and the root bundle hash.

The production hierarchical command must call
`load_production_stage1_hierarchy_handoff` in
`oci.inference.production_stage1_hierarchy_handoff` and consume its direct
prefit-catalog provider. The older
`load_authenticated_stage1_bundle_for_hierarchy` path validates the historical
handoffs but exposes them only as diagnostic compatibility arguments;
`hierarchy_cli_arguments()` raises. Executing those paths would silently re-fit
a different accumulated-spent schedule and is forbidden. The returned
`legacy_scope_index_path` remains the authenticated drill-back boundary for raw
legacy evidence; those sidecars remain outside automatic prompt grounding.

## Resume and recovery

Use `--resume` only with the same output directory and identical request:

```bash
uv run --frozen python scripts/build_all_evidence_stage1_bundle.py \
  ...same arguments... \
  --resume
```

- A component is reusable only when its terminal manifest identity, manifest
  content hash, complete file set, sizes, and file hashes all validate.
- Legacy and neural-query partial components are not replayed or trusted. Use a
  fresh output directory after either one is interrupted.
- A nonempty, unsealed TF-IDF component is rejected. Native partial checkpoints
  are not reused because they are not independently authenticated; restart it
  in a fresh output directory. Only a complete sealed TF-IDF component may be
  reused.
- A completed bundle is reused only after every root registration, component,
  source cache, HTR tree, split artifact, and all-ten coverage assertion
  revalidates.
- A changed cohort, config, cache, model tree, code file, split, or runtime
  setting creates a different internal request hash and cannot reuse the prior
  bundle.

Never edit a sealed artifact to make resume pass. Preserve the failed directory
for diagnosis and start a new output directory.

## Downstream one-shot orchestration status

Stage 1 itself is deterministic local modeling and has no JSON-agent retry
surface. The downstream hierarchical feature-discovery runner does. Its strict
parser rejects duplicate keys and malformed JSON, then allows one bounded,
authenticated repair response for the same logical job. The repaired response
is rerun through the same semantic validator; both attempt and response hashes,
the repair-policy identity, and the final parsed-response hash are authenticated.
The sequence fails closed after that single repair.

This prevents one malformed model response from forcing a user to restart the
cohort workflow or approve a new digest. It does not authorize Stage 1 to
construct a remote client or weaken evidence, fold, cache, or response
validation. The remaining release requirement is a genuine one-shot E2E run
through this already-implemented repair path.
