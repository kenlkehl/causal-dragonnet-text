# Portable all-evidence causal inference pipeline

This is the active master record. The exact superseded 90,785-byte working
record is preserved at `todo_list_7-22-26.history.md` with SHA-256
`d4bcb596a4aef42a03eec3a2ce63e7d01a03e89ec69835317a860584dd508c59`.
Historical timestamps, PIDs, launch commands, and completed test narratives
belong only in that archive.

## Scientific mission

Estimate patient-level probability-scale treatment-effect heterogeneity from
complete narrative text:

`tau(X_i) = P(Y=1 | do(T=1), X_i) - P(Y=1 | do(T=0), X_i)`.

Version 1 supports binary treatment and binary outcome only. The final
estimator is strict `CausalForestDML`; no structured or nonforest fallback is
allowed. The benchmark forest profile is configuration: 200 trees, minimum
leaf 10, `max_features="sqrt"`, honest splitting, inference enabled, seed 42.

The initial acceptance cohort is the configured 1,000-patient synthetic NSCLC
dataset. Oracle fields, prompts, timelines, and pre-extracted truth remain
sealed until the 1,000 oracle-free outer-held-out predictions and their
manifest have been frozen, reopened, and authenticated. Poor oracle agreement
is an honest result and is not a completion failure.

## Required workflow

Stage 1 is an oracle-free evidence factory. Every required logical context must
retain all ten nonempty and separately interpretable evidence families:

1. Word treatment/outcome models.
2. Word residual-effect model.
3. Hierarchical transformer.
4. Matched-patient uplift models.
5. Whole-cohort embeddings.
6. Cluster-local embeddings.
7. Lexical semantic-retrieval contrasts.
8. TF-IDF topics.
9. Residual TF-IDF n-grams.
10. Learned neural queries.

Whole-cohort and cluster-local embeddings are independently mandatory. With
five outer folds and two review rounds the configured plan contains 40 logical
contexts: five full-outer, 25 exact-inner, and ten cumulative-review contexts.
Equivalent scopes are discovered by content. The benchmark has five
`hierarchy_epoch_001`/`inner_005` equivalence pairs, so it retains 40 logical
records while requiring at most 35 physical all-architecture fits.

For each outer fold, Stage 2 must reopen the sealed Stage 1 handoff, interpret
each family separately, account for every lossless page, integrate only after
all ten summaries, propose at most 20 note-readable variables, assign causal
roles, extract them from every complete prepared note, use the first three
training partitions for initial evaluation, reveal partitions four and five
in two bounded reviews, freeze definitions, fit one strict forest on 800
outer-training patients, and predict the 200 held-out patients once.

## Portable implementation contract

- Three typed layers: `ScientificWorkflowSpec`, `DeploymentProfile`, and
  `RunControl`.
- Critical scientific limits have no production defaults: paging and
  embedding geometry, reconciliation fan-in, candidate count, and strict
  forest settings are explicit configuration. Generic extraction defaults to
  no character limit, while any allocation cap must be proven nonbinding or
  abort before fitting.
- Scientific identity includes dataset/model content, row/split identity,
  scientific settings, prompts, seed policy, runtime compatibility class, and
  transitive producer code. It excludes paths, hostnames, GPU IDs/count,
  worker PIDs/count, completion order, and operational stop/resume controls.
- Path-neutral artifact DAG manifests contain exact upstream artifact IDs,
  ordered payload inventories, and one physical-location-independent content
  root. Locator/execution attestations are separate.
- Repeatable `--adopt-checkpoint PATH` reopens and authenticates every byte,
  validates dependencies and compatibility, rejects partial/loose/tampered or
  downgraded artifacts, and writes an immutable adoption attestation.
- A separate explicit research-only
  `--trust-prior-adoption-attestation PATH` control may reuse a preparation or
  embedding-cache artifact whose bytes were fully authenticated by that exact
  prior immutable adoption attestation. It revalidates the current controls,
  dependency identities, and filesystem-stat continuity but deliberately does
  not reopen, hash, or copy payload bytes. Its transitive attestation must state
  `payload_bytes_reauthenticated=false`,
  `fresh_full_byte_validation_achieved=false`, and
  `global_release_certified=false`; it is never silently substituted for
  ordinary checkpoint adoption.
- Active work uses configured local POSIX scratch and publishes terminal
  artifacts once. Dense arrays use mmap-safe `.npy`; tables use Parquet/Arrow;
  manifests and indexes use small canonical JSON. No pickle or large
  compressed NPZ defaults.
- One immutable shared embedding cache is exposed through
  `ScopedEmbeddingView`, which limits fit code to selected rows. Same-user
  hostile-process isolation is explicitly out of scope.
- `PhysicalFitKey` binds architecture, target, canonical fit-row order,
  scientific configuration, canonical seed, producer identity, and runtime
  class. One physical result may have multiple immutable logical bindings only
  after full ten-family equality is proven.
- Resource policy accepts `cpu`, explicit devices, or `auto`, discovers 0–N
  GPUs, records hardware only in execution attestations, supports CPU when
  models permit it, and never kills external processes.
- Telemetry covers wall/CPU time, GPU utilization and peak memory, and bytes
  read, written, copied, hashed, compressed, decompressed, JSON-encoded, and
  synchronized.
- `complete_paged_v1` requires its core size, side-context size, and page
  maximum in the scientific specification. The NSCLC acceptance configuration
  supplies 13,488/256/14,000, but production code supplies no hidden geometry
  defaults. Every configured geometry must provide exact-once core coverage,
  absolute prepared-text citations, overlap deduplication, and configured
  recursive reconciliation with complete child accounting. Reconciliation
  fan-in is likewise scientific configuration (16 only in the benchmark
  profile); patient and page counts are derived without a truncating cap.
- Production LLM calls use zero transport retries and at most one fixed-schema
  repair. Both responses must identify the configured model and finish with
  `stop`; any unresolved schema, model, citation, coverage, or transport
  failure aborts rather than becoming missing data.
- The path-only terminal validator reopens all Stage 1/2, forest, row-map, and
  frozen-prediction artifacts, proves oracle-open ordering, and records
  `execution_completed`, `run_validation_status`, and
  `global_release_certified=false`.

## Preservation and rollout

- [x] Preserve v1–v5 artifact trees unchanged.
- [x] Archive and verify the exact prior working record.
- [x] Complete typed portable configuration and public compatibility shims.
- [x] Complete portable artifact DAG, binary layout, telemetry, and adoption
  validation code.
- [x] Complete scoped-cache, physical-fit deduplication, authenticated
  single-execution owner validation, and resource-scheduler implementation.
  The optional lower-level legacy canary API is not invoked by production.
- [x] Complete paged Stage 2 and event-order terminal validation.
- [x] Implement the exclusive 35-physical/40-logical all-ten execution gate
  and forbid typed portable runs from silently entering the legacy
  40-attempt builder.
- [x] Implement the concrete deployment-bound six-producer factory, including
  authenticated sibling BoW nuisance banks for matched-patient uplift. Every
  producer field is supplied by the closed scientific profiles; constructor
  defaults cannot fill omitted settings.
- [x] Persist each canonical clustered-preflight KMeans/SVD state as safe
  JSON/NPY payloads during preflight and bind the 35 physical embedding
  producers to those canonical no-refit states.
- [x] Finish validation and workflow dispatch for the compact
  reference-only Stage-1 handoff format: scientific manifest, locator
  attestation, split plan, row map, and guarded in-place cumulative evidence
  provider. It copies no evidence or numerical payload.
- [x] Assemble exact-inner OOF and full-outer held-out numerical banks by
  authenticated reference, route them through the strict forest without
  constructing/refitting the historical hierarchy bundle, seal all five fold
  outputs, and reopen the complete direct graph in the terminal validator.
- [x] Implement the reference-only Stage-2 runtime canary as exactly one real
  architecture-level initial-interpretation request with no extraction,
  forest, prediction, or oracle access.
- [x] Finish the positive public five-fold/two-review one-shot exercise. The
  source-frozen reference-only integration completed all five folds, both
  bounded reviews and untouched gates per fold, five strict forest calls,
  sealed fold outputs, and the global frozen prediction assertions in
  1,640.24 seconds.
- [x] Replace the legacy multi-gigabyte inline clustered-preflight JSON with a
  compact v2 artifact: one authenticated request/audit reference, one ordered
  Parquet concept table per physical owner, individual `.npy` state arrays,
  and 40 small logical bindings to 35 physical owners. The reader must
  authenticate every registered byte and reject missing, reordered,
  substituted, extra, linked, or tampered payloads.
- [x] Seal one reusable prepared Stage-1 context and use persistent
  spawn-isolated resource slots. Each slot may authenticate the context once
  and execute multiple owners sequentially only after resetting the complete
  configured RNG/thread state at every owner boundary. Production, canary,
  and benchmarking must share this executor and perform one parent
  reauthentication per returned owner.
- [x] Remove producer-identity and checkpoint-proof amplification with
  process-local, stat/hasher-guarded authentication handles. Fresh path-only
  validators still reopen and hash every byte; source, imported dependency,
  missing-import creation, stat-only, callable-state, and hasher changes all
  invalidate reuse. The formerly stalled workflow DAG gate now passes in
  under 40 seconds.
- [x] Materialize the configured Gemma Stage-2 tokenizer as a symlink-free
  four-file deployment tree and bind both benchmark and acceptance profiles
  to that explicit locator. Its files match the original snapshot blobs
  byte-for-byte and load locally without endpoint access.
- [x] Finish the monolithic stable broad suite under
  `/home/klkehl/thisenv`.
- [x] Finish the resumable representative benchmark implementation, including
  authenticated workload deployment, one-observation pause/resume, all-ten
  replica equality, six-lane overlap telemetry, persistent-slot cleanup,
  compact durable publication, and measured deployment selection.
- [ ] Run representative 800-row and 640-row performance benchmarks and
  select the fastest deterministic resource profile that meets the configured
  memory/headroom and throughput criteria.
- [x] Create and validate the post-repair first-acceptance source snapshot and
  absent work root. The initial validated snapshot exposed a six-pass
  same-process model-tree authentication defect during live staging and is
  retained only as a superseded diagnostic record.
- [x] Implement the operator-requested no-rehash V5 cache path. The exact prior
  r2 full-byte adoption attestations are the trust roots; the current reader
  uses guarded read-only path-backed memory maps and retains the legacy V5
  encoder/chunk metadata only through an exact sealed 17-field projection.
  Focused cache, adoption, request-projection, and run-control validation is 83
  passes. This policy is intentionally ineligible for fresh-full-byte and
  global-release certification.
- [x] Replace clustered-preflight cache replication with one set-level shared
  cache reference and 35 row-restricted views. Per-scope recovery artifacts
  contain no embedding array or chunk-text payload; treatment/outcome and note
  text remain fit-scope-local. Persistent workers reopen the V5 cache through
  guarded read-only memory maps, reuse one authenticated chunk-text line index,
  and memoize one physical handle per worker. The focused layout,
  access-control, tamper, line-index, and row-fingerprint gate is 11 passes,
  and the representative serial-versus-loky complete scientific audit is
  exactly equal.
- [x] Prove the narrow V5 preparation/cache migration in rehearsal. The
  exact frozen V5-v2 producer is allowlisted by source, dependency lock,
  model-tree, builder-code, and payload identities; all registered bytes are
  reopened, and generic v2 artifacts remain rejected. The accepted portable
  rehearsal artifacts are recorded below. The productive fresh run still
  must publish immutable adoption attestations into its own work root.
- [x] Remove the remaining device-zero assumptions from canary descriptor
  preparation and the legacy dual-device canary boundary. Configured GPU IDs
  are authenticated as deployment assignments, while each
  `CUDA_VISIBLE_DEVICES`-isolated replica uses logical device zero. Generated
  legacy configs no longer prescribe `cuda:0` or GPU IDs `[0,1]`.
- [x] Migrate the legacy hierarchy cluster-fit index to 35 authenticated
  physical-owner records with complete logical-to-physical bindings for all
  40 contexts. Missing, reordered, substituted, or tampered aliases fail
  closed.
- [x] Recompute all 35 clustered-preflight equivalence groups through the
  shared V5 cache and retain 40 logical bindings. The legacy V4 state cannot
  prove its current seed, safe KMeans/SVD payload, and dependency identities,
  so direct reuse is rejected rather than silently accepting 40 historical
  outputs.
- [ ] Complete Stage 1 through fresh handoff validation while the Stage 2
  endpoint is offline. Preserve frozen r12 as launched; subsequent requests
  must not run a full duplicate production owner as a canary.
- [ ] Resume the identical request for Stage 2, strict forests, frozen
  predictions, optional post-freeze oracle evaluation, and terminal validation.
- [ ] Report performance telemetry and overall/per-fold Pearson, Spearman,
  MAE, RMSE, signed error, and truth/estimate variance.

## Acceptance evidence

Implemented validation gates:

- [Typed configuration and portable-contract tests](tests/test_portable_workflow_contracts.py)
  require every scientific/text-window field explicitly, prove that geometry
  affects scientific identity while paths/devices/workers do not, and cover
  relocation, adoption, scoped cache access, fit deduplication, and resource
  planning.
- [Source-snapshot tests](tests/test_production_source_snapshot.py) separate
  the operational repository locator attestation from the path-neutral code
  content root; byte-identical source trees at different absolute paths now
  have the same scientific source identity.
- [Complete-page tests](tests/test_complete_paged_extraction.py) exercise
  non-benchmark page geometry, derive request totals from 37 notes without a
  note/page cap, and verify exact-once page, citation, response, repair,
  reconciliation, and terminal-ledger accounting.
- [Embedding-cache tests](tests/test_production_embedding_cache_builder.py)
  prove configured chunk caps are nonbinding before fitting and abort on
  either chunk-level or tokenizer-level semantic truncation.
- The exact NSCLC page geometry `13,488/256/14,000` and reconciliation fan-in
  `16` occur only in the benchmark
  [scientific example](example_configs/portable_all_evidence_scientific_nsclc.json),
  not in `oci/` or production scripts.
- [Role-neutral producer tests](tests/test_role_neutral_all_ten_binding.py)
  and the six producer-specific test files cover every full-outer,
  exact-inner, deduplicated alias, and independent cumulative physical-owner
  shape. The latest integrated factory/producer/workflow gate is 58 passed;
  whole-cohort and
  cluster-local embeddings remain distinct native families, and no held-out
  treatment/outcome field is accepted by a producer transform.
- [Role-neutral execution tests](tests/test_production_stage1_role_neutral_execution.py)
  prove exactly-once execution and authentication of six producers per
  physical owner, completion-order-neutral scheduling, 35 physical fits, 40
  logical bindings, and five content-derived equivalence groups.
- [Direct numerical-bank tests](tests/test_direct_upstream_numerical_reference_bank.py)
  cover exact-inner OOF assembly, full-outer held-out transforms, gate-only
  cumulative references, mmap/sparse payload reuse, and strict fold-shape
  authorization without per-family replay or refitting.
- [Reference-only handoff tests](tests/test_production_role_neutral_stage2_handoff.py)
  reopen the no-copy provider/runtime capability, row map, prepared cohort
  projection, all-ten catalogs, and relocation/tamper boundaries.
- [Public legacy-checkpoint adoption tests](tests/test_public_legacy_checkpoint_adoption.py)
  reopen and migrate exact V5 preparation/cache terminal manifests through
  repeatable `--adopt-checkpoint`, reject partial/tampered/ambiguous inputs,
  and persist the V4 audit-only decision as 40 logical contexts, 35 physical
  recomputations, and five superseded duplicates. The focused file is nine
  passes with no xfail.
- The exact V5 preparation/cache rehearsal reopened the legacy manifests,
  prepared cohort, model evidence, and the complete 627-MiB numerical cache,
  scanned every embedding value for finite/unit-normalized content, and then
  passed a second fresh-process path-neutral validation boundary. Its portable
  artifact IDs are
  `5eab023992dbee7ab02ade15e5708edfce6f28b02600e0a87b18360f8bb24be6`
  (prepared cohort) and
  `25d9fd56ed98b5108e5c23cff55ec0409ff9a83436f318c88a24da394114988c`
  (embedding cache). The combined focused cache/adoption/portable gate is
  155 passes.
- [Terminal validation tests](tests/test_production_terminal_artifact_validation.py)
  now include a full five-fold direct CATE fixture and reopen all fold
  manifests/predictions, explicit strict-forest and nuisance settings,
  canary, Stage-1 graph, row order, closed inventory, and oracle phase order.
- [Workflow tests](tests/test_production_all_evidence_workflow.py) cover the
  explicit role-neutral integration seam and prove that typed mode rejects
  generic Stage 1 hooks/overrides and legacy-shaped adopted handoffs, then
  aborts before constructing the legacy builder when the seam is absent.
- The final source-stable broad gate completed outside the sandbox under
  `/home/klkehl/thisenv`: 3,155 passed, five expected skips, zero failures,
  and zero errors in 6,525.30 seconds. Before that gate, the cached
  failure-set rerun was 24 passed. Focused portability/nontruncation coverage
  was 161 passed; repaired compatibility, probe, TF-IDF, hierarchy-loader,
  and logical-binding clusters were independently rerun and reviewed
  fail-closed. `git diff --check` and Python compilation passed at the merge
  boundaries.
- After repairing process-local model-tree authentication and typed operational
  root overrides, the complete source-stable gate was rerun outside the sandbox:
  3,168 passed, five expected skips, zero failures, and zero errors in
  6,152.07 seconds. The independent focused gates were 126 passed for the
  workflow/authenticated-tree files and 11 passed for typed-profile root
  overrides and the direct-shim conflict matrix.
- The initial first-acceptance source snapshot was created at
  `artifacts/production_source_snapshot_20260725_portable_acceptance` and
  independently reopened with its own code in a separate process. Its 240-file
  content root is
  `7f08291c0c425187ae72cc65f13a19c3cce635812ae797c9dbf448c1c66a7e78`.
  The benchmark-staging and final-acceptance durable and local-scratch targets
  were absent before request creation. Live staging then proved that checkpoint
  adoption left the embedding-model revalidation policy on the uncached path:
  the same process would have hashed the 15,150,575,778-byte Qwen tree six
  times through preflight. That request was interrupted at the clean boundary
  after its two adopted preparation/cache phases and before any preflight
  output; the root is preserved but superseded.
- The post-repair source snapshot was created at
  `artifacts/production_source_snapshot_20260725_portable_acceptance_r2` and
  independently reopened using its own code. Its 240-file path-neutral content
  SHA-256 is
  `5a012aea9da4de111d8e9ad97c56c550255fdaad3cfe89ced8eed7e4d9956d9a`.
  Typed `--work-root` and `--scratch-root` overrides now select fresh
  operational targets without changing scientific or checkpoint compatibility
  identities; all other direct deployment shims remain rejected.
- The first operator-trusted cache-reuse snapshot was created at
  `artifacts/production_source_snapshot_20260725_portable_acceptance_r3` and
  independently reopened using its own code. Its 242-file path-neutral content
  SHA-256 is
  `73d8c0d0390e349963effab1c847c42f79c3e3e9bf1e2ea3f5d4e8b3d32ff815`.
  Live request initialization exposed one duplicated ordinary compatibility-key
  check after the trusted compatibility projection; it failed before either
  payload or preflight was opened and is preserved as a superseded diagnostic.
  The check was narrowed to the same producer-code-only exception while
  retaining exact artifact, kind, upstream, metadata, phase, and all other
  compatibility checks.
- The repaired operator-trusted snapshot is
  `artifacts/production_source_snapshot_20260725_portable_acceptance_r4`,
  independently reopened with 242 files and path-neutral content SHA-256
  `b7d6eec6bb3f8e23fd92f93f831e2e5c99cad33abe9af3c63b05657b0df83f9b`.
  Its request successfully published the two operator-trusted
  preparation/cache phases, but the old preflight publisher then created 35
  private cache copies (about 16 GiB) and all 35 cluster fits completed before
  state sealing exposed two incompatible row-fingerprint conventions. The
  failed root and scratch evidence are preserved unchanged and are not
  resumable checkpoints. The fingerprint comparison is repaired without
  weakening exact row-order checks, and the copy-producing preflight schema is
  superseded by the shared-cache design above. A new r5 immutable snapshot and
  absent work root are the next productive boundary.
- The shared-cache r5 source snapshot was independently validated with 242
  files and path-neutral content SHA-256
  `1905bf70d5482039a95a036ffa669749a3c3b95c6202ed5603b960a413ad6ff9`.
  Its fresh request reused the V5 preparation/cache, published only one shared
  cache descriptor, copied zero per-scope embedding arrays or chunk-text
  payloads, and completed all 35 physical preflight fits with 40 logical
  bindings. Scope-input publication fell from 42m49s in r4 to 1m44s. The
  complete durable preflight copy then exposed a cross-filesystem publisher
  defect: byte copying did not preserve deliberate `0444/0555` sealing, and
  post-commit deletion of the read-only scratch tree raised `PermissionError`.
  The failed r5 request and its complete unterminalized durable attempt remain
  preserved as diagnostic evidence; they are not resumed.
- Cross-filesystem publication now preserves every source file and directory
  mode, makes only workflow-owned scratch directories removable, and treats a
  cleanup error after durable rename/fsync as operational residue rather than
  invalidating the committed publication. One focused regression forces the
  cross-filesystem branch with a real sealed prepared context and proves exact
  modes, bytes, successful reopening, and non-fatal post-commit cleanup. It
  passes independently; no broad test campaign was repeated.
- The repaired publisher was exercised by the independently validated r6
  snapshot
  `artifacts/production_source_snapshot_20260725_portable_acceptance_r6`
  (242 files; path-neutral content SHA-256
  `636c2c2d4d82138de454b42fd6de07cc4d0b759be000bf05916b16f193200c74`).
  Its fresh request reused the operator-trusted V5 preparation/cache and
  terminally published all 35 physical preflight fits with 40 logical
  bindings, 429 durable files, zero per-scope embedding arrays or chunk texts,
  exact `0444/0555` sealing, and successful scratch cleanup. The durable
  preflight is retained as valid evidence for r6, but the immutable request
  stopped before GPU allocation when Stage 1 exposed a contradictory TF-IDF
  declaration: the portable profile said `registered_context_heldout` while
  the effective configuration silently rewrote it to
  `nested_fit_calibration`. The shared-cache refactor also removed durable
  replication but parent-process result IPC still read about 24.3 GiB; this is
  recorded performance debt, not a reason to weaken or delay the current
  scientific run.
- The hidden TF-IDF rewrite and the related
  `include_bow_phrases_as_concepts=true` to `false` rewrite are removed.
  Intended values are now explicit in both source profile trees and the
  portable TF-IDF profile, and production validation fails closed instead of
  mutating them. HTR live-attestation, empty external caches, and empty
  concept phrases are likewise validated rather than coerced. JSON validation
  plus the two focused effective-config/profile reconstruction tests pass
  (`2 passed`); no broad test campaign was run. Because r6 binds the old full
  scientific identity and producer identity, its preflight is not adopted
  into a corrected request. A fresh r7 request will reuse only the V5
  preparation/cache and recompute the 35 groups.
- The first corrected r7 launch stopped before creating a work or scratch root:
  its legacy V5 preparation/cache manifests carried the historical
  whole-workflow configuration digest, so the downstream-only TF-IDF
  correction was still treated as an upstream cache incompatibility. The
  exact comparison proved that dataset, row order, all model identities,
  prompts, folds, seed, and runtime were unchanged; only the global
  configuration digest differed. Operator-trusted legacy prefix reuse now
  replaces only that over-broad digest after validating each artifact's
  sealed typed migration expectation against current dataset/columns/text
  preprocessing and, for the cache, the exact upstream prepared artifact,
  embedding model/tree, and complete chunk/encoder configuration. The proof
  is content-hashed into the immutable adoption record and recomputed during
  every external-input reopen. Payload bytes remain unread, relevant upstream
  changes fail closed, and global certification remains false. The single
  focused downstream-drift/threshold/chunk-size regression plus the two
  existing TF-IDF profile gates pass (`3 passed`); no broad suite was run.
- The independently validated r8 snapshot is
  `artifacts/production_source_snapshot_20260725_portable_acceptance_r8`
  (242 files; path-neutral content SHA-256
  `e08c84a27c55a2b43e78a8f0369962caa131e08c3ee812ac68537aec5ca9a1fa`).
  Its fresh request reused the operator-trusted V5 preparation/cache and
  terminally published 35 physical preflight fits with 40 logical bindings,
  430 phase files, 494,391,455 bytes, and zero copied embedding arrays or
  chunk-text payloads. Stage 1 then failed before dispatching any owner:
  both GPU slots spent the fixed 600-second readiness window redundantly
  reopening all 35 preflight owner concept/array states. The terminal r8
  preflight remains valid and is an ordinary portable-adoption candidate.
- Persistent startup no longer has a production source default. The finite
  positive deadline is required by the typed deployment-only Stage 1
  execution profile, is passed explicitly into the executor, and is excluded
  from scientific identity while remaining operationally recorded. The NSCLC
  deployment profile selects 1,800 seconds; other deployments must choose
  their own value.
- Clustered-state bundle reconstruction now authenticates the ordered compact
  owner index, manifests, registrations, paths, sizes, and link safety without
  eagerly decoding concept Parquets or reading NPY arrays. Full concept/array
  authentication occurs once on the first actual use of each owner and is
  memoized within that process. Against the exact r8 bytes, all 35 owners
  indexed in 18.84 seconds with zero owner payloads loaded and about 1.14 GiB
  peak RSS; the first real owner then authenticated its ten arrays with the
  expected content identity. The combined focused lazy-access, tamper,
  timeout-cleanup, CLI-propagation, and science-neutrality gate is `5 passed`;
  no broad test campaign was run.
- The r9 snapshot
  `artifacts/production_source_snapshot_20260725_portable_acceptance_r9`
  was independently reopened with 242 files and path-neutral content SHA-256
  `4667a2e6a6d81afe41a46933dc1ee42d428ee9b41810618b84e7758ea37a7954`.
  Ordinary adoption of r8 preflight failed closed on its producer-code
  compatibility key, so no exception was added; the fresh r9 request
  `2ef20a5f0f4548eac90215543a9c2181e4b9c4e2756b6c93b6658cce2db4d990`
  reused only the operator-trusted V5 preparation/cache and recomputed
  preflight. It terminally published 35 physical/40 logical contexts in about
  49 minutes, with 430 files, 35 canonical state manifests, and zero
  `chunk_embeddings.npy` or `chunk_texts.jsonl` scope copies. Both persistent
  GPU slots then reconstructed the same artifact and entered actual Stage 1
  fitting in roughly 35 seconds, crossing the exact readiness boundary that
  had timed out after 600 seconds in r8.
- The r9 Stage 1 attempt was then stopped cleanly before replica B or any
  additional physical owner began. Its first owner exposed a real
  role-neutral BoW regression: all nuisance/effect folds were serial and the
  benchmark profile explicitly selected one fold worker, leaving the second
  persistent slot idle for about 30 minutes. The terminal r9 preflight remains
  intact and is an ordinary adoption candidate.
- Role-neutral BoW now uses the existing configured
  `bow_fold_parallelism`/`bow_parallel_backend` controls. Independent folds
  return live fitted models and predictions without sharing `_ArrayStore`;
  OOF placement, evidence records, fitted-model order, and proof capture are
  merged serially in canonical fold order. Each fold retains its prior derived
  seed and limits nested native pools to one thread. The two-owner/eight-CPU
  benchmark profile explicitly selects four fold workers per owner, while the
  library derives and enforces the owner CPU cap from the deployment policy
  without a source-level worker-count constant. The one complete-artifact
  serial/parallel equality-and-overlap test and the single control-propagation
  check pass; Python compilation and `git diff --check` are clean. No broad
  suite or benchmark matrix was run before the r10 relaunch.
- The independently validated r10 snapshot is
  `artifacts/production_source_snapshot_20260725_portable_acceptance_r10`
  (242 files; path-neutral content SHA-256
  `12d41e8d4f549dc62a2e624877d3bb089fb4c82b1e47c29d9a3ae9e38bd9822b`).
  Ordinary adoption of r9 preflight failed closed on producer compatibility
  before either fresh root was created; no exception was added. Fresh request
  `1be74af2ec5759839002ec4e083c06642fa87da51e518ed919147e4d5a4e4284`
  reused the two trusted V5 preparation/cache attestations and entered an
  eight-worker shared-cache preflight recomputation. All 35 scope inputs and
  the set manifest were published, and preflight terminally completed all 35
  physical/40 logical contexts with 35 canonical state manifests and zero
  per-scope embedding or chunk-text cache copies.
- The r10 canary was intentionally stopped before producing any BoW payload
  after live host counters showed that four overlapping fold threads consumed
  only 1.02 aggregate CPU cores. TF-IDF vocabulary/token counting is
  Python/GIL-bound, so the configured `threads` backend did not deliver the
  required throughput despite logical overlap. Both persistent slots exited
  and released their GPUs; the terminal r10 preflight remains an ordinary
  adoption candidate. The benchmark now selects the existing `processes`
  backend with the same configured four folds and scheduler-derived
  four-CPU owner budget. The same single focused test proves complete serial,
  thread-parallel, and process-parallel artifacts are byte-identical and that
  folds overlap; no new backend or test campaign was introduced.
- The fresh r11 snapshot independently validates with the same 242-file
  path-neutral source root as r10 because no production Python or lockfile
  changed after r10. Ordinary r10 preflight adoption passed general
  compatibility but failed closed on the prepared-context binding before
  either r11 root existed; no migration or exception was added. Fresh request
  `a17bbeec02a232bd7a548d68c672cf59eac0a24d7ae97480b2474182f8eab367`
  reused the trusted V5 preparation/cache and terminally published all 35
  physical/40 logical preflight contexts with 35 canonical state manifests
  and zero per-scope embedding-array or chunk-text cache copies. Both
  persistent GPU slots became ready in about 80 seconds. The first productive
  BoW canary used four process-backed folds, sustained about 2.6 aggregate CPU
  cores instead of r10's 1.02, completed fitting in about 12 minutes, and
  terminally published its two logical views after deterministic fold-order
  proof capture and fresh replay. The owner then failed closed before creating
  any HTR output: preparation and the HTR proof convention hashed model-tree
  rows with `size`, while the new role-neutral runtime checker used
  `size_bytes`. The identical unchanged five-file tree therefore produced
  `fb7b1e...` versus `fd242d...`. This was a deterministic schema mismatch,
  not a model mutation. The loose terminal BoW component is not reusable
  because its owner and Stage 1 phase never terminalized.
- The role-neutral HTR runtime tree inventory now uses the existing
  preparation/proof field name `size`; its one focused compatibility
  regression passes, both changed production modules compile, and the scoped
  diff check is clean. Fresh r12 snapshot
  `artifacts/production_source_snapshot_20260725_portable_acceptance_r12`
  independently reopens with 242 files and path-neutral content SHA-256
  `6f45aeea869b00f71a931cf1100a7b3019fdab8c38be19d2563c55c43690f72c`.
  One ordinary adoption attempt of r11's terminal preflight rejected
  scientific/producer compatibility before creating either r12 root; no
  exception was added. Fresh r12 request
  `b7ff1ba6f47296029f18c9b43674ff0b92e211c3ac479ddbfca7abedd595eab1`
  initialized with only the trusted V5 preparation/cache and entered
  shared-cache preflight at `2026-07-26T05:04:57Z`. Preflight terminally
  published 35 physical/40 logical contexts at
  `2026-07-26T05:54:26Z`, with 430 phase files, 35 canonical state
  manifests, canonical no-refit true, and zero per-scope embedding-array or
  chunk-text cache copies. Both persistent slots became ready in about seven
  seconds. Productive canary replica A entered `outer_001_full` BoW at
  `2026-07-26T05:56:06Z` with four process folds sustaining about 2.66
  aggregate CPU cores. BoW fitting completed in about 11 minutes 39 seconds,
  both authenticated logical family views and the terminal execution manifest
  published, and the fresh reopen completed. Replica A crossed the corrected
  model-tree gate and began HTR on GPU 0 at
  `2026-07-26T06:28:13Z`; complete all-architecture replica equality remains
  pending.
- At the operator's superseding direction, r12 was stopped on
  `2026-07-26` rather than spend several more days in serial neural folds.
  `SIGTERM` was sent only to the verified owned process groups
  `1100432`, `1113720`, and `1113721`; all exited and released both GPUs.
  Replica B never launched. Nothing was deleted: r12's terminal preflight,
  96,751-file completed BoW component, and incomplete HTR scratch remain.
  The incomplete owner is not being salvaged or migrated.
- The full production Replica A/B byte-identical canary is no longer the
  intended forward policy. The frozen r12 request cannot skip replica B:
  it has no owner-terminal pause/adoption seam, dispatches B immediately
  after A returns, and its handoff validators require both replicas. Stopping
  would leave A as loose scratch and same-request resume would rerun the
  entire Stage 1 phase. R12 is preserved on disk but stopped under the later
  operator instruction; its loose partial A is neither resumed nor migrated.
  Working source now uses one authenticated canonical
  execution per physical owner and honestly propagates disabled-canary
  `false`/`false`/`null` claims through the handoff and fresh validator.
  Configuration, ordered rows, seeds, prompts, schemas, provenance, and
  discrete evidence remain exact. HTR, matched-pair HTR, and learned-query
  within-artifact replay now authenticate stored bytes exactly and compare
  only recomputed neural floating outputs with explicit default-free
  per-profile tolerances; the NSCLC benchmark profile declares
  `rtol=3e-5` and `atol=3e-6`. Eleven neural replay/config/factory nodes and
  three canary/handoff nodes pass, as do targeted compilation and
  `git diff --check`; no broad suite, preflight recomputation, or migration
  framework was introduced.
- [x] Restore deployment-configured process parallelism in the current
  role-neutral HTR producer without reviving the legacy estimator. Five
  nuisance folds now run through bounded isolated leases, all nuisance OOF
  results merge in canonical fold order, a strict residual barrier completes,
  and five effect folds then run through the same leases. Device identities,
  total concurrency, slots per device, and CPU budget are operational
  configuration; spawned workers enforce and re-observe the Stage-1 Torch
  determinism policy and use one read-only complete tokenizer/chunk plan.
  Fold-local arrays merge serially, no mutable array store is shared, and all
  capacity limits fail closed instead of truncating text.
- [x] Complete only the focused HTR gates requested for this restoration.
  Three simulated/process nodes prove nuisance/effect overlap and barrier
  order, two tasks on one device, use of multiple devices, canonical
  serial/parallel equality, declared-tolerance neural equality, child-process
  determinism, and unchanged complete-text coverage. The one deployment
  propagation assertion passes after retaining encoder microbatch 16.
  Compilation and `git diff --check` pass; no broad suite was run.
- [x] Complete the narrow real-GPU check on the two configured 48-GiB A6000s.
  Two independent batch-16 runs each scheduled five folds concurrently
  (three leases on `cuda:0`, two on `cuda:1`), enforced the nuisance/effect
  barrier, used both devices in both stages, and produced identical terminal
  content under exact discrete and declared-tolerance neural comparison.
  The runs took 47.92 and 46.14 seconds on the 25 longest complete V5-prepared
  notes, covering 267,163 words and 3,735 chunks without truncation.
  Conservative overlapping-child peaks were 19.41 GB and 12.67 GB, leaving
  at least 31.49 GB headroom. A larger encoder microbatch completed without
  OOM but failed the complete-artifact equality gate, so it was rejected and
  batch 16 remains selected. No multi-GPU speedup claim is made without the
  required single-device baseline.
- [x] Freeze and independently reopen R13 source snapshot
  `artifacts/production_source_snapshot_20260725_portable_acceptance_r13`:
  243 files with path-neutral content SHA-256
  `50c2420f7ff4beb83791217e315298491e51bec055058d83b41048cfa559f5d9`.
  The one permitted ordinary adoption attempt of R12's terminal preflight
  failed closed on scientific/producer compatibility before either R13 work
  root existed. No retry, migration, exception, or partial-A salvage was
  introduced. R13 was immediately relaunched from the same frozen source with
  only the prior V5 preparation/cache attestations; it will recompute
  preflight against the shared cache with no per-scope embedding or chunk-text
  copies, then continue through Stage-1 handoff validation. The fresh request
  is `78dc44196bdae983fa09bf7d30b60f379bd5f93e162a18e19c3b8a9c53017ccf`;
  its preparation and cache phases are terminal through stat-continuity
  attestations that explicitly record no payload-byte reauthentication, and
  its fresh `stage1_preflight` phase is running.
- A read-only pre-resume Stage 2 audit found one concrete complete-page
  reconciliation contract mismatch without disturbing frozen r11: internally
  authenticated leaf citations contain `start`, `end`, `text`, and `sha256`,
  while the model-facing reconciliation response schema accepts only the
  first three fields. The working source now builds prompt-only three-field
  citation copies and states that closed item schema in both the initial and
  fixed-repair prompts; authenticated originals and system-side SHA
  regeneration remain unchanged. Its one focused copy-through regression
  passes, as do Python compilation and scoped `git diff --check`. This fix
  will enter a fresh immutable post-handoff snapshot/request; ordinary
  adoption of the validated r11 Stage 1 handoff will be attempted once, with
  no compatibility exception or migration rabbit hole.
- [Performance benchmark tests](tests/test_role_neutral_performance_benchmark.py),
  [publication tests](tests/test_role_neutral_performance_benchmark_publication.py),
  and [lane-overlap tests](tests/test_role_neutral_lane_overlap_analysis.py)
  cover 640/800-row selectors derived from the configured fold plan, 32
  observations/64 complete all-ten fits, resumable checkpoints, direct
  per-device memory telemetry, deterministic equality, and path-neutral
  durable selection. No 640/800 cohort-size constant appears in production
  library logic.
- The reference-only numerical assembler, public dispatch, strict forest,
  direct sealer, one-request canary, and fresh five-fold terminal validator
  are implemented. The positive public five-fold/two-review fixture passed
  under a frozen source tree (`1 passed` in 27 minutes 20 seconds); every fold
  crossed both authenticated review gates and sealed its forest output. All
  result-changing Stage 2 generation, review, paging, and real-estimator
  constructor choices are now explicit typed scientific configuration. The
  remaining gates are measured resource selection, productive checkpoint
  adoption/recompute, Stage 1 handoff, Stage 2, and terminal/oracle validation.
  V5 partial preflight and all V4 fitted preflight outputs remain categorically
  non-reusable.

The durable acceptance deployment is not considered complete until every
intended evidence path ran, discovery/review remained outer-training-only,
every patient received exactly one outer-held-out forest estimate, and any
oracle source was opened only after predictions were frozen.
