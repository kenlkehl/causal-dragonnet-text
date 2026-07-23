# TODO list — 2026-07-22

## Durable-work rule

Keep this file current as implementation, testing, and live execution expose
new work. Before ending a turn or allowing conversation compaction, add any
unresolved defect, design gap, failed acceptance check, recovery constraint, or
follow-up here. Mark items complete only with the validating command/artifact
recorded. Do not rely on conversation history as the sole record.

## Release embedding-model GPU memory after cache construction

Status: pending; do not edit the active cache-builder implementation until the
current production attempt reaches a terminal state.

Context:

- The 1,000-patient production cache completed embedding computation and was
  atomically published on 2026-07-22 at approximately 14:39 EDT.
- After publication, GPU 1 still showed approximately 30 GB allocated with 0%
  utilization while CPU/disk validation continued.
- `build_production_embedding_cache()` retains the local `encoder` variable
  until the builder function returns and does not explicitly perform garbage
  collection or call `torch.cuda.empty_cache()`.
- PyTorch's caching allocator can normally reuse this memory in the same
  process, so the allocation alone does not prove that later training will
  fail. Explicit cleanup is still preferable to reduce fragmentation, expose a
  clear phase boundary, and avoid accidental live model references.
- Do not patch `oci/inference/production_embedding_cache_builder.py` during the
  active attempt. Cache provenance contains the builder code hash; changing the
  file during validation could invalidate the current run.

Implementation after the active attempt terminates:

1. Scope encoder use so it is released immediately after tokenizer coverage
   validation and `_encode_chunks()` complete.
2. Explicitly delete the encoder reference.
3. Run `gc.collect()`.
4. For a CUDA device, call `torch.cuda.empty_cache()` after work has completed
   synchronously.
5. Ensure cleanup also occurs on encoding/validation exceptions without
   weakening atomic cache cleanup or replacing the original exception.
6. Record a non-secret cleanup audit in the cache result/manifest, including
   device, cleanup attempted/completed, and allocated/reserved memory before
   and after when available.
7. Keep the cache's builder/provenance hashes consistent with the new code.

Required tests:

- A mocked CUDA test proves encoder references are released and garbage
  collection plus `torch.cuda.empty_cache()` occur after successful encoding.
- The same cleanup occurs when encoding raises.
- CPU cache construction never invokes CUDA cleanup.
- Cleanup failure is handled fail-closed or attached diagnostically without
  hiding the original build error.
- Existing cache atomic-publication, validation, provenance, and tamper tests
  continue to pass.
- A focused live check confirms GPU memory falls after cache publication before
  later Stage 1 model fitting begins.

## Separate cache construction from Stage 1 and make the sealed cache resumable

Status: pending; P0 recovery issue.

- The workflow currently writes an `embedding_cache` phase manifest that only
  records the GPU resource check. Actual cache construction happens inside the
  `stage1_modeling` attempt.
- The current attempt successfully published a complete four-file cache under
  its Stage 1 attempt, but the Stage 1 phase has not sealed. An identical
  workflow resume would currently start another fresh cache instead of reusing
  the independently validated published cache.
- Implement cache construction as its own real phase/attempt, independently
  validate all four files and provenance, and make Stage 1 consume only that
  completed cache identity.
- Resume must reuse a cache only when dataset bytes, prepared ordered text,
  local model tree, tokenizer/chunk settings, code version, file hashes, and
  validation manifest all match exactly. Partial temporary caches remain
  non-reusable.
- Add interruption tests for: before publication, immediately after atomic
  rename, after cache validation, and during later Stage 1 work.

## Add durable progress ledgers for long local phases

Status: pending.

- `_encode_chunks()` exposes no cursor or completed-chunk count, making the
  five-hour 38,267-chunk cache build difficult to estimate or audit while live.
- Add an atomic progress ledger with planned/completed chunks, rows, bytes,
  timestamps, throughput, and current subphase. It must not authorize partial
  cache reuse.
- Add progress for model-tree hashing, cache hashing/validation, cluster
  preflight scopes, full/inner/cumulative Stage 1 fits, and Stage 2 planned vs.
  completed requests.
- Ensure useful phase transitions are written to the persistent console log as
  well as disk.

## Complete proof-safe CPU process parallelism across Stage 1 scopes

Status: partially implemented; pending audit and fixes after the active attempt
terminates.

What is already implemented:

- The production invocation binds `--tfidf-workers 8` and
  `--tfidf-parallel-backend processes` into the Stage 1 request.
- `run_tfidf_topic_stage1()` globally schedules the 5 full-outer plus 25
  exact-inner contexts with joblib `Parallel(backend="loky", n_jobs=8)`,
  `batch_size=1`, `pre_dispatch="all"`, and
  `inner_max_num_threads=1`. Thus the main 30 independent TF-IDF context fits
  are process-parallel rather than serialized by outer fold.

Known serial or restricted paths:

- The 40-scope clustered-embedding feasibility preflight currently loops over
  scopes serially. This is likely the current CPU-bound approximately one-core
  phase, despite the process owning many idle library threads.
- The 10 cumulative-review TF-IDF snapshot fits currently loop serially after
  the 30 exact contexts complete.
- Production BoW native-proof capture forces within-model fold work to one job
  because the current in-memory proof sink would be copied and lose child
  captures under loky.
- Multi-model outer-fold parallelism is disabled when custom embedding/HTR
  providers are supplied, and the current invocation also sets
  `--num-workers 1`.
- Neural-query and some cumulative native-family scope loops require a separate
  device/memory and proof-capture audit before parallel execution is safe.

Required implementation/audit:

1. Build a matrix of every full-outer, exact-inner, cumulative-review, nested
   calibration, and preflight workload with its CPU/GPU needs, current executor,
   requested worker count, and proof-output ownership.
2. Parallelize the 40 independent clustered-embedding preflight scopes using a
   bounded loky process pool or another measured process backend. Share the
   frozen embedding array read-only/memory-mapped, return closed per-scope
   results, and restore canonical scope order before hashing.
3. Parallelize the 10 independent cumulative TF-IDF fits with the same bounded
   loky policy and deterministic ordered collection.
4. Replace the BoW in-memory proof sink restriction with child-local persisted
   proof captures that the parent independently reloads, verifies, and orders,
   then permit bounded loky fold parallelism.
5. Avoid nested oversubscription: enforce a single native BLAS/OpenMP thread per
   worker and one authoritative global CPU budget.
6. Do not parallelize multiple GPU-heavy fits onto one GPU unless measured peak
   memory proves the declared concurrency safe. CPU and GPU scheduling should
   be separate.
7. Persist executor/backend, requested/effective workers, worker PIDs, task
   counts, canonical result ordering, timings, and failure propagation in each
   phase manifest.
8. Add equivalence tests proving serial and process-parallel outputs/proofs are
   identical and reordered/substituted child results are rejected.
9. Benchmark 1, 4, and 8 workers on representative 800/640-row scopes before
   fixing the production default; retain the fastest stable setting within the
   host memory budget.
10. Add a proof-safe cross-GPU scope scheduler for independent full-outer,
    exact-inner, and cumulative neural fits. Assign each scope to one declared
    GPU, keep child-local proof outputs isolated, restore canonical scope order
    before hashing, and fail closed on missing/duplicated scope results. Measure
    peak memory and equivalence before allowing GPU 0 and GPU 1 to run scopes
    concurrently. The active attempt is serial on GPU 1 and cannot adopt this
    change without a new code/run identity.

## Finish production `complete_paged_v1` extraction before any Stage 2 endpoint work

Status: pending; P0 scientific/production blocker.

- Current code losslessly plans all note pages, but the active provider does
  not yet implement the full required citation-bearing page response contract
  and bounded remote reconciliation protocol.
- It currently merges one unambiguous value locally and aborts on conflicting
  page values. This is not a substitute for contract-aware recursive
  reconciliation across every page.
- Require exact absolute-offset citations for every positive page result,
  validate citations against prepared text, deduplicate context-overlap evidence
  by offsets, and apply each feature's declared temporal/aggregation rule.
- Persist page plans, span hashes, request/response metadata, normalized parsed
  results, citation proofs, reconciliation trees, and planned/actual counts
  without copying raw notes into response caches.
- Transport failures, invalid model/finish metadata, exhausted single schema
  repair, missing/duplicated pages, invalid citations, and incomplete
  reconciliation must abort rather than become missing values.
- Do not allow the current one-shot workflow to contact the Stage 2 endpoint
  until this item and its acceptance tests are complete.

## Strengthen workflow phase boundaries and independent terminal validation

Status: pending.

- The `stage1_preflight` workflow phase currently records delegation rather
  than running and sealing a distinct preflight attempt before supervised fits.
- The terminal validator currently checks phase manifests but must independently
  reopen and validate the Stage 1 bundle, row map, all fold manifests, strict
  forest identity, frozen prediction bytes/schema, post-hoc evaluation, and
  oracle event order using paths only.
- Bind the complete behavior-affecting source tree/code identity into the top
  run request, not only a small subset of workflow modules.
- Add tests proving incomplete attempts are preserved and every skipped phase
  is accepted only from a complete matching manifest.

## Audit the checked-in full production profiles

Status: pending.

- The new profiles were derived from the balanced-200 effective fixture and
  mechanically updated to the requested 1,000-patient values.
- Perform a closed-schema comparison against the intended scientific profile:
  all ten sources, 5 outer folds, 5 inner partitions, 50 Stage 1 epochs,
  120/80 neural-query epochs, 10 fixed clusters, support thresholds, 128 chunk
  cap, HTR settings, strict 200-tree forest, review count, and devices.
- Confirm all stored credential fields are neutral placeholders and endpoint/
  model values used by Stage 2 come only from invocation configuration.

## Resolve or formally classify remaining broad-suite failures

Status: pending.

The broader pre-endpoint suite last reported 334 passes and two failures:

1. Qwen embedding model backend/pooling default in
   `tests/test_extractors.py::TestHierarchicalTransformer::test_sentence_encoder_backend_and_pooling_defaults`.
2. Packed contract-RAG configuration versus adaptive-review request-locality in
   `tests/test_contract_lexical_context.py::test_fusion_cli_wires_packed_contract_rag_and_composite_cache_identity`.

Determine whether each is pre-existing expectation drift or a production-path
defect, fix or update it deliberately, and rerun all required suites before any
real Stage 2 endpoint call.

## Live-run recovery record

Status: active; preserve evidence.

- Work root:
  `artifacts/production_all_evidence_one_conf_one_mod_1000_v1`
- Active Python PID at last check: `3644296`.
- Prepared cohort completed and validated: 1,000 rows.
- Fresh cache published at approximately 14:39 EDT with 38,267 chunks,
  4,096 dimensions, 128-chunk nonbinding cap, and production metadata v2.
- Stage 1 has not yet emitted a terminal bundle manifest.
- Preserve the earlier numeric-ID serialization failure root and incomplete
  Stage 1 attempts; do not relabel them complete or delete them.
- If the live attempt terminates before Stage 1 sealing, independently validate
  the published cache and recover it through the corrected separate-cache phase
  rather than recomputing it.
- 2026-07-22 17:42 EDT checkpoint: PID `3644296` remained alive after 8h29m,
  using approximately one CPU core (`95%` process CPU) with no active loky
  workers. GPU 1 retained approximately 30.5 GB but had 0% utilization. No
  Stage 1 bundle/output files had appeared beyond the completed cache and
  effective profile. This is consistent with the serial 40-scope clustered
  feasibility preflight and demonstrates the need for both per-scope progress
  records and bounded process parallelism. Main TF-IDF context fitting had not
  started.
- 2026-07-22 19:06 EDT checkpoint: the serial 40-scope cluster preflight had
  completed and the Stage 1 bundle root was initialized at 17:45. The active
  temporary scope was `outer_001_full`; its HTR evidence directory appeared at
  18:00. GPU 1 was actively used (approximately 9.1 GB, 66% utilization) while
  `bert-tiny` HTR/matched-pair neural work ran. No completed outer-scope artifact
  had yet been published. GPU 0 was not assigned to this workflow; it held an
  external `VLLM::EngineCore` allocation of approximately 45 GB at 0%
  utilization. Main TF-IDF context fitting had not started.
- 2026-07-23 05:52 EDT checkpoint: `outer_001_full` completed at 00:26 after
  approximately 6h40 of scoped modeling. `outer_001_inner_001` completed its
  20 HTR model fits by 05:21 and was publishing its BoW, HTR, matched-pair, and
  embedding native-proof captures through 05:52. The exact-inner scope was not
  yet terminally registered, but its expensive model fitting was complete.
  Thus the first two full/exact scopes consumed approximately twelve hours,
  or roughly six hours per scope, while the 30 full/exact scopes are scheduled
  serially. A simple extrapolation is about seven additional days for the
  remaining 28 full/exact scopes alone. Ten cumulative scopes, later Stage 1
  families/packaging, complete-note Stage 2 work, forests, and validation remain,
  so an uninterrupted end-to-end run can reasonably exceed one week and may
  approach two weeks. GPU 0 was idle; GPU 1 had released the retained Qwen
  allocation and held only a roughly 416 MiB CUDA context during CPU-side proof
  publication.
