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
- [x] Complete scoped-cache, physical-fit deduplication, compute-canary, and
  resource-scheduler implementation.
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
- [ ] Replace the legacy multi-gigabyte inline clustered-preflight JSON with a
  compact v2 artifact: one authenticated request/audit reference, one ordered
  Parquet concept table per physical owner, individual `.npy` state arrays,
  and 40 small logical bindings to 35 physical owners. The reader must
  authenticate every registered byte and reject missing, reordered,
  substituted, extra, linked, or tampered payloads.
- [ ] Seal one reusable prepared Stage-1 context and use persistent
  spawn-isolated resource slots. Each slot may authenticate the context once
  and execute multiple owners sequentially only after resetting the complete
  configured RNG/thread state at every owner boundary. Production, canary,
  and benchmarking must share this executor and perform one parent
  reauthentication per returned owner.
- [ ] Finish the monolithic stable broad suite under
  `/home/klkehl/thisenv`.
- [ ] Run representative 800-row and 640-row performance benchmarks and
  select the fastest deterministic resource profile that meets the configured
  memory/headroom and throughput criteria.
- [ ] Create and validate a fresh source snapshot and absent work root.
- [x] Prove the narrow V5 preparation/cache migration in rehearsal. The
  exact frozen V5-v2 producer is allowlisted by source, dependency lock,
  model-tree, builder-code, and payload identities; all registered bytes are
  reopened, and generic v2 artifacts remain rejected. The accepted portable
  rehearsal artifacts are recorded below. The productive fresh run still
  must publish immutable adoption attestations into its own work root.
- [ ] Recompute all 35 V4 preflight-equivalence groups and retain 40 logical
  bindings. The legacy V4 state cannot prove its current seed, safe
  KMeans/SVD payload, and dependency identities, so direct reuse is rejected
  rather than silently accepting 40 historical outputs.
- [ ] Run the productive Stage 1 canary and Stage 1 through fresh handoff
  validation while the Stage 2 endpoint is offline.
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
  recomputations, and five superseded duplicates. The focused file is seven
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
- One monolithic broad run completed with 2,489 passes, five skips, nine
  failures, and 11 errors. Every reported failure/error was triaged: the
  sandbox-only HTR forkserver cases pass in their required execution
  environment, the forced-offline model-snapshot case passes normally, and
  the nested DataLoader and probe-schema regressions were fixed and rerun
  green. This is evidence of coverage, not the final gate: the complete suite
  must be rerun after the current integration edits. Targeted terminal/oracle
  validation is 16 passed, and Python bytecode/diff checks remain required at
  each merge boundary.
- The reference-only numerical assembler, public dispatch, strict forest,
  direct sealer, one-request canary, and fresh five-fold terminal validator
  are implemented. The positive public five-fold/two-review fixture passed
  under a frozen source tree (`1 passed` in 27 minutes 20 seconds); every fold
  crossed both authenticated review gates and sealed its forest output. Before
  the monolithic regression gate, the remaining local work is to make all
  result-changing Stage 2 generation, review, and real-estimator constructor
  choices explicit in typed scientific configuration. V5
  preparation/cache migration must then prove the exact typed columns,
  preprocessing, row/projection identities, model tree, chunk/token capacity,
  legacy builder identity, and every registered byte. V5 partial preflight and
  all V4 fitted preflight outputs remain categorically non-reusable.

The durable acceptance deployment is not considered complete until every
intended evidence path ran, discovery/review remained outer-training-only,
every patient received exactly one outer-held-out forest estimate, and any
oracle source was opened only after predictions were frozen.
