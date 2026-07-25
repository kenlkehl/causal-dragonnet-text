# TODO list — 2026-07-22

## Durable-work rule

Keep this file current as implementation, testing, and live execution expose
new work. Before ending a turn or allowing conversation compaction, add any
unresolved defect, design gap, failed acceptance check, recovery constraint, or
follow-up here. Mark items complete only with the validating command/artifact
recorded. Do not rely on conversation history as the sole record.

## Accepted execution plan — parallel Stage 1 restart

Plan accepted: 2026-07-23.

Objective: replace the serial v1 Stage 1 attempt with a fresh, independently
recoverable Stage-1-only run that reuses the authenticated embedding cache,
executes all 40 full/exact/cumulative scopes with deterministic two-GPU
scheduling, seals complete scope subphases, and stops after an independent
handoff-loader validation. No Stage 2 endpoint may be contacted by this run.

Target work root:

`artifacts/production_all_evidence_one_conf_one_mod_1000_v2_parallel_stage1`

Freshness checkpoint on 2026-07-23: the target root is absent and the untouched
source Parquet still hashes to
`6566aef4350ac1f78589d87d6a383bb49dce4a63ae5c9661921e925e3e854fe0`.

### Milestone 0 — stop and preserve v1

Status: complete.

- Verified main PID `3644296`, session/process group `3644290`, and the exact v1
  command before signaling.
- Sent `SIGINT` to PID `3644296` on 2026-07-23. The complete process group exited
  without requiring `SIGTERM` or `SIGKILL`.
- The interrupt was recorded in
  `artifacts/production_all_evidence_one_conf_one_mod_1000_v1.console.retry2.log`
  during `outer_001_inner_001` matched-pair proof validation.
- GPU 0 returned to approximately 15 MiB and GPU 1 to approximately 188 MiB,
  with no workflow compute process.
- Preserve the entire v1 artifact tree and both incomplete Stage 1 attempts.
  Never relabel loose full/inner outputs complete and never resume v1 after the
  parallel code change.

### Milestone 1 — authenticated cache import

Status: implementation complete; live v2 relocation pending.

Implementation/test record (2026-07-23):

- Added `oci/inference/production_embedding_cache_relocation.py` with typed
  create/revalidate APIs and a fresh, closed relocation tree.
- The relocation binds both preparation manifests and ordered four-column
  cohorts, source/destination cache bytes, Qwen model-tree identity, chunk
  configuration, source cache provenance, and relocator code identity.
- Destination artifacts must be distinct single-link regular files; symlink,
  arbitrary hard-link, writable-mode, extra-entry, byte, manifest, source,
  model, and root-substitution attacks abort.
- Independent adversarial review plus
  `uv run --active --no-sync --frozen pytest -q
  tests/test_production_embedding_cache_relocation.py
  tests/test_production_source_snapshot.py` passed `19` tests.
- A fresh read-only authentication of the v1 cache completed on 2026-07-23.
  The surrounding diagnostic command then raised `KeyError: embedding_dim`
  while formatting its printout because that convenience key is not in the
  returned identity; validation itself had already returned. The live v2
  relocation remains the authoritative recorded acceptance check.

- Freshly prepare the four-column cohort in v2.
- Independently validate the v1 prepared cohort, preparation manifest, Qwen
  model tree, chunk policy, and atomically published four-file cache.
- Copy prepared/cache bytes into v2 without symlinks or hard links.
- Seal a relocation attestation binding source/destination hashes, ordered
  row/text identity, local model tree, builder identity, and companion files.
- Keep relocation code outside the original cache-builder module so the
  recorded builder-code hash remains verifiable.
- Make the cache a real completed phase and require Stage 1 to consume only its
  independently reopened manifest.

Acceptance: destination cache has 1,000 rows, 38,267 chunks, dimension 4,096,
all source/destination file hashes match, the 128-chunk cap remains nonbinding,
and every relocation-tamper test aborts.

### Milestone 2 — real parallel preflight

Status: implementation complete and integrated tests green; live 40-scope v2
artifact pending.

- A bounded loky implementation and canonical-order reducer are implemented.
- Serial-versus-two-worker fixture equality is covered in
  `test_embedding_cluster_preflight_enumerates_all_native_scopes_with_real_catalogs`.
- The public preflight phase publishes the one effective profile and a terminal
  40-scope scientific artifact. Stage 1 modeling consumes those exact bytes
  instead of regenerating the profile or recomputing the preflight.
- Added `production_stage1_cluster_preflight_artifact.py`: it seals the raw
  audit, exact preflight Stage 1 request, ordered per-scope records, and
  cluster-fit identities in a closed read-only tree bound to profile, cache,
  registry, configuration, runtime, and source-code identities. Workflow
  preflight owns the one effective profile and artifact, and modeling consumes
  those paths without recomputing it.
- Artifact tamper/order/substitution coverage plus workflow boundary coverage:
  `23 passed` in
  `tests/test_production_stage1_cluster_preflight_artifact.py` and
  `tests/test_production_all_evidence_workflow.py` on 2026-07-23.
- Do not mark complete merely because the worker-count setting is present:
  the terminal preflight artifact, all 40 exact identities, and comparison
  against all 40 supervised fits are still required.
- Independent isolation review previously found that the loky shard payload
  received the full label/text DataFrame and global cache path while processing
  several scopes. This was replaced with exactly 40 one-scope tasks. Each
  serialized task contains only its fit-row values, a sanitized config without
  prepared-cohort/global-cache paths, and a restricted cache capability
  containing only those fit rows. Parent aggregation alone sees all 40 outputs.
- Non-fit label/text/cache mutation-invariance tests, a worker visibility spy
  proving refusal of non-fit cache rows, an exact 40-task/order test, and a
  serialized-payload scan proving no global label/cache path are implemented.
- Implemented in review: exactly 40 one-scope loky submissions now receive a
  fit-only global-shaped data projection, physically restricted cache,
  path-neutral config, and canonical parent reorder. Per-scope preflight input
  publication is recoverable.
- Focused record: `3/3` input isolation/mutation/interruption tests and `1/1`
  submission-spy test passed. The spy confirms 40 tasks and no serialized
  global cohort/cache path.
- The last pre-acceptance defect was a private-config round trip that disabled
  `multi_model_forest.embedding_contrast` while leaving its historical alias
  enabled; it is fixed.
- Completed on 2026-07-23: the private-config round trip now preserves the
  enabled embedding contrast, the real-cluster fixture is exactly equal under
  serial and two-worker execution, and the sealed-consumer spy proves modeling
  never calls the cluster-preflight builder. The isolated-preflight,
  sealed-artifact, consumer, and workflow suite passed all `28` tests directly
  under `/home/klkehl/thisenv`.

- Replace the delegated preflight placeholder with a terminal preflight
  manifest covering exactly 5 full, 25 exact-inner, and 10 cumulative scopes.
- Run the 40 cluster-feasibility tasks with bounded process parallelism
  (`loky`, up to 8 workers, one native numerical thread per worker).
- Bind K-means inputs/results, SVD matrices/components, raw concepts, semantic
  concepts, and final catalog concepts into one closed per-scope identity.
- Do not start supervised Stage 1 fitting unless all 40 scopes pass.

Acceptance: 40/40 complete, canonical order restored after out-of-order worker
completion, and changed/missing/reordered/substituted cluster results abort.

### Milestone 3 — deterministic two-GPU scope scheduler

Status: implementation complete and integrated tests green; live dual-GPU
canary and production execution pending.

- Typed 40-scope planning, deterministic per-scope seeds, balanced static GPU
  assignments, spawn orchestration, sealed scope attempts, resume validation,
  and an operational progress ledger are implemented and validated.
- The real legacy scope-fragment emitter, collision-safe canonical merger, and
  exact preflight-versus-fit clustered-embedding checks are wired.
- Scope attempts and the progress ledger live at an explicit,
  request-bound recovery path that is stable across sibling
  `stage1_modeling` workflow attempts. Keeping them only under the current
  bundle attempt would silently lose sealed-scope reuse after `--resume`.
- A spawned fixture test now proves finite labels equal exactly the selected
  scope's fit rows, scope-kind text visibility is enforced, and a sibling
  modeling attempt reuses the same sealed recovery attempts (`1 passed` in
  `test_spawned_probe_and_sibling_attempt_resume_use_stable_recovery`).
- Independent review previously found a production leakage blocker in the
  shared descriptor: every worker could reach all 40 fit-label Parquets and the
  full 40-scope label-derived cluster-preflight artifact. The implementation
  now publishes a private per-scope descriptor with only exact-schema
  row-ID/text data, that scope's fit labels, and that scope's authenticated
  preflight projection. A child receives no path to the other 39 label or
  preflight projections; parent aggregation authenticates all 40 against the
  full sealed preflight.
- Exact physical Parquet schemas are enforced, and an adversarial child test
  enumerates all descriptor-supplied inputs and finds no non-fit treatment,
  outcome, or oracle data.
- The 40 private cache views total roughly 17 GiB, so descriptor publication is
  recoverable work. Each scope descriptor seals independently under the stable
  recovery identity; only complete matching scope views are reused, incomplete
  attempts are preserved, and the descriptor-set terminal manifest is written
  only after all 40 independently reopen. The interrupt-after-N/resume test
  proves completed view bytes are not rewritten.
- Implemented: descriptor publication now uses stable per-scope attempts,
  preserves partial attempts, reuses completed scope trees byte-for-byte, and
  writes the descriptor-set terminal marker last. The interruption/resume test
  passes.
- Implemented: fragment merge now performs a fresh path-only full
  reauthentication and bottom-up durability synchronization before terminal
  publication. Validation on 2026-07-23: `15` non-spawn adapter tests and `11`
  fragment tests passed; the spawned adapter suite had already passed `5/5`.
- Independent root-agent full-file rerun of both adapter and fragment suites
  passed all `16` collected tests in 82.94 seconds using `~/thisenv`.
- Scheduler validation record on 2026-07-23: `20 passed, 2 deselected` for the
  non-spawn suite, `2 passed` for adversarial inode/hash-seed spawn coverage,
  and `3 passed` for post-terminal/lost-message recovery. Request/manifest v4
  binds store/scope/attempt inode identities, streams inventories through
  anchored file descriptors, rejects same-path directory substitution, treats
  terminal publication as a no-mutation boundary, and uses bounded direct
  terminal authentication when a Queue completion message is lost.
- Independent root-agent rerun with
  `/home/klkehl/thisenv/bin/pytest -q
  tests/test_production_stage1_scope_scheduler.py` passed all `22` tests in
  82.61 seconds.
- Scope spawn sets `PYTHONHASHSEED` to the deterministic scope seed only around
  each serial `process.start()`, restores the exact parent environment, seeds
  only the selected CUDA device, and terminates/joins all children on
  `BaseException`, `KeyboardInterrupt`, or `SIGTERM`.
- The private process-group ready marker is now schema v2 and binds the Linux
  `/proc/<pid>/stat` process start-time ticks. Cleanup verifies the marker,
  leader PID, process group, and process birth identity before signaling, so a
  recycled PID/process-group number cannot cause an unrelated process to be
  killed. An unstarted child is handled without signaling.

- Add typed scope specs for 5 full, 25 exact-inner, and 10 cumulative fits.
- Derive schedule-independent scope RNG seeds from seed 42 and canonical scope
  IDs; reset Python, NumPy, and Torch in each isolated worker.
- Use deterministic largest-fit-row-first assignment with one active scope per
  GPU. The fixed 1,000-row registry balances to 20 scopes and 12,800 fit-row
  units per GPU:
  - GPU 0: 3 full, 12 exact-inner, 5 cumulative.
  - GPU 1: 2 full, 13 exact-inner, 5 cumulative.
- Launch fresh spawned subprocesses so CUDA state is never forked or retained
  across scopes.
- Write child output only beneath a private scope attempt; write the terminal
  scope manifest last; independently reopen it before aggregation.
- Reuse only complete matching scope subphases. Preserve and ignore incomplete
  attempts.
- Keep device/PID/timing data in a separate execution ledger; canonical
  scientific payloads and hashes must not depend on completion order or GPU.
- Require actual clustered-embedding identities from all 40 fits to equal
  preflight exactly.

Acceptance: serial and parallel fixture payloads/proofs are identical; all
leakage, worker-failure, CUDA-OOM, termination, disk-full, tamper, and resume
tests fail closed.

### Milestone 4 — parallel CPU and neural-query lanes

Status: implementation complete and integrated tests green; live production
execution pending.

- Global 8-worker `loky` execution covers the 30 main TF-IDF contexts and the
  10 cumulative TF-IDF contexts.
- CPU TF-IDF work runs concurrently with the two-GPU legacy lane without nested
  process pools or BLAS/OpenMP oversubscription.
- Keep BoW folds serial inside each scope for this first safe version; revisit
  only if measured at more than 20% of Stage 1 wall time.
- Use the existing single neural-query service with devices `cuda:0` and
  `cuda:1`, preserving its service identity while distributing inner-fold
  work.
- The TF-IDF lane is its own spawned, sealed recovery attempt and starts
  alongside the legacy two-GPU orchestrator. The child receives only a
  picklable TF-IDF projection (effective config, modeling rows, registry,
  request identity, and CPU settings), never the embedding-cache object. Seal
  the completed component in the stable recovery tree before atomically
  publishing it into the bundle; preserve incomplete attempts.
- Cross-lane failure/interrupt propagation ensures a failed CPU lane terminates
  and joins active GPU children, and a failed GPU lane terminates and joins the
  TF-IDF process. A completed sealed TF-IDF attempt must be reusable even when
  the parent was interrupted before publication.

Implementation/validation record (2026-07-23):

- The 30 main TF-IDF contexts and 10 cumulative spent-only contexts run through
  bounded, one-native-thread-per-worker loky execution with deterministic
  canonical collection.
- The sealed recoverable TF-IDF component starts concurrently with the
  dual-GPU legacy lane; cross-lane failure and interruption clean up owned
  descendants, while a completed matching component remains reusable.
- Neural-query work uses the declared `cuda:0` and `cuda:1` service devices.
  Each scientific scope still has one authoritative worker/device assignment.
- Recovery/security, held-out isolation, deterministic seed, cumulative
  contract, and real ten-scope loky serial/parallel equivalence tests are
  green; the combined parallel Stage 1 suite result is recorded under
  Milestone 6.

Acceptance: 40 TF-IDF and 40 neural-query scope records are complete and
canonical, both native embedding sources remain independently present, and no
TF-IDF source substitutes for a failed embedding source.

### Milestone 5 — public Stage-1-only workflow

Status: implementation complete; immutable snapshot accepted and live
preparation prefix active.

- The workflow/CLI now accept ordered plural GPU IDs, retain the singular
  compatibility alias with conflict rejection, check every requested GPU for
  exclusive availability, support authenticated cache relocation, release
  process CUDA allocations at the cache boundary, and provide a Stage-1-only
  phase sequence with no endpoint/model requirement.
- Stage-1-only handoff validation runs in a fresh process. The multi-GPU path
  uses the production fragment adapter and remains fail-closed on any missing
  or mismatched fragment.
- Initial workflow/cache/snapshot suite: `22 passed`; independent workflow
  resume/terminal-closure review is complete.
- Each phase now atomically seals its exact full attempt file tree; resume
  rejects changed or extra bytes. Terminal validation runs in a fresh,
  path-only subprocess. The CLI re-execs from the authenticated read-only
  source snapshot rather than merely recording its identity.
- Authenticated cache-relocation options now propagate into the Stage 1
  builder. Imported-cache validation no longer creates idle CUDA contexts;
  fresh embedding builds release only the GPU they actually used.
- Stage-1-only isolation tests block socket connections and verify the
  one-shot/canary modules and OpenAI client package remain unimported.
- CLI review, TF-IDF calibration, and interaction-fold values are now bound
  into both effective multi-model sections and the explicit-feature forest;
  a differing-values test prevents decorative CLI knobs.
- Source-snapshot re-exec now sets `PYTHONHASHSEED` to the configured global
  seed before the new interpreter starts and rejects an authenticated snapshot
  marker if the active hash seed differs. The two focused re-exec tests passed
  directly under `/home/klkehl/thisenv` on 2026-07-23.
- The workflow now reopens and rehashes every request-bound external input
  (source dataset, both profiles, local model trees, authenticated cache-import
  sources, implementation/hook files, and source snapshot) before and after
  every phase and again in the fresh path-only terminal validator. Mutating a
  profile or model input after immutable request publication aborts before the
  phase can be marked complete; the four focused hash-seed/input-boundary tests
  passed under `/home/klkehl/thisenv`.
- Added the operational
  `--prepare-stage1-canary-descriptors-only` mode. It is intentionally excluded
  from the typed scientific request: it initializes the exact immutable v2
  request, completes or reuses only input preparation, authenticated embedding
  cache relocation, and the 40-scope cluster preflight, invokes the supervised
  builder's preparation boundary to publish all 40 private scope descriptors,
  records one canonical full-outer canary descriptor, and stops before every
  legacy, TF-IDF, neural-query, Stage 2, or remote fit/request.
- Added a fresh path-only validator for that preparation prefix. It reopens the
  immutable request and all three completed phases, verifies all external-input
  and preflight identities, requires exactly 40 descriptors and the selected
  canary scope, and proves that no `stage1_modeling` or TF-IDF output exists.
- The finalized workflow suite passed `22` tests in 23.59 seconds under
  `/home/klkehl/thisenv` on 2026-07-23.
- The final adversarial pre-snapshot audit identified four recovery/source
  boundary defects; all four are implemented and covered:
  - Prep-only now requires an authenticated source snapshot before it creates
    the work root. A mutable-workspace descriptor preparation cannot be
    accepted.
  - The neural-query profile's scientific Stage 1 request identity is
    content-addressed by path and SHA-256. Inode and timestamp identity remains
    in the per-attempt input audit, so mutation during one attempt still
    aborts, while byte-identical replacement between prep and resume no longer
    invalidates sealed preflight/descriptors.
  - Immutable request initialization is staged in a sibling
    `.initialization_attempt_*` directory, durably written and reopened, then
    atomically renamed into the requested root. An interruption preserves the
    attempt while leaving the fixed production root fresh and reusable.
  - The fresh prep validator now explicitly supplies and reports
    `PYTHONHASHSEED`, `PYTHONPATH`, `PYTHONNOUSERSITE`, and the authenticated
    source-snapshot marker, uses `-P`, and rejects any mismatch in addition to
    checking that the validator module came from the snapshot.

- Add ordered plural Stage 1 GPU IDs while retaining a nonconflicting singular
  CLI compatibility alias.
- Add fixed one-scope-per-GPU, preflight-worker, cache-import, CPU-budget, and
  Stage-1-only settings to the typed request and immutable manifest.
- Require every declared GPU to be exclusively available; report conflicts and
  never kill external processes.
- The workflow GPU gate now checks both reported compute processes and the
  physical idle state for every declared GPU. It rejects memory above the
  larger of 512 MiB or 2% of capacity, less than 6 GiB headroom, or non-idle
  utilization even when the compute-app table is empty. The focused
  all-GPU/compute-occupant/unreported-memory tests passed `3/3`; the gate only
  reports and aborts and never signals an external process.
- After the hash-seed, external-input, and physical-GPU gate changes, the
  complete workflow, cache-relocation, source-snapshot, and checked-in-profile
  group passed `42` tests under `/home/klkehl/thisenv` on 2026-07-23.
- In Stage-1-only mode, endpoint/model are unnecessary, no remote client is
  constructed, and execution ends only after a fresh process accepts the
  terminal Stage 1 handoff.
- Launch from an immutable source snapshot so later workspace edits cannot
  alter the running Stage 1 code identity.

Acceptance: workflow phase and source identities validate from paths only, a
same-request restart skips only sealed work, and the command cannot enter
canary, Stage 2, oracle, or terminal inference phases.

### Milestone 6 — tests, benchmark, and production launch

Status: all implementation/test gates and the bounded fixture benchmark are
green; immutable snapshot accepted; live preparation prefix active before the
real dual-GPU canary and production launch.

- Dual-GPU reproducibility canary tooling is implemented in
  `oci/inference/production_stage1_dual_gpu_canary.py` with the public entry
  point `scripts/run_stage1_dual_gpu_reproducibility_canary.py`.
- The production CLI accepts no worker override: it runs the exact
  `run_legacy_stage1_scope_worker` path twice from one authenticated private
  descriptor, with one common scientific request/scope seed and only the
  operational GPU assignment changed.
- Each replica is a fresh spawned process with strict Torch determinism and one
  native numerical thread. Each leader first establishes an authenticated
  private POSIX process group, so peer failure or interruption terminates
  descendants as well as the leader. The parent independently reopens both
  sealed fragments and compares the complete canonical accumulator plus
  artifact identities, while PID/device/timing/resource values remain outside
  that scientific comparison.
- The canary requires execution from the authenticated source snapshot, rejects
  occupied GPUs without killing external processes, persists an atomic
  `nvidia-smi` resource ledger, enforces a conservative peak below 85%, at
  least 6 GiB headroom, and a minimum 1.5 concurrent-throughput factor, and
  terminates/joins both owned children on peer failure, cancellation,
  `KeyboardInterrupt`, or `SIGTERM`.
- CPU-only spawned-process contract tests cover identical scientific requests
  and output authentication, payload/descriptor/extra-file tampering, peer
  failure and cancellation cleanup, resource rejection, source-snapshot
  execution binding, production CLI defaults, fresh output, and prohibition of
  a nonproduction worker. Validation on 2026-07-23:
  `/home/klkehl/thisenv/bin/pytest -q
  tests/test_production_stage1_dual_gpu_canary.py` passed `9` tests in 22.91
  seconds. The peer-failure fixture starts a real delayed grandchild and proves
  that no orphan sentinel survives group termination.
- The adversarial canary audit defects are fixed. The canary reads the current
  descriptor API (`plan_content_sha256`, one-scope authority, and assignment);
  requires the descriptor's logical assignment and global seed; gives both
  replicas the identical logical `cuda:0` scientific assignment; maps physical
  GPU UUIDs separately through `CUDA_VISIBLE_DEVICES`; restores the parent
  environment; authenticates `PYTHONHASHSEED`; avoids circular descriptor
  setup through the workflow preparation-only prefix; streams large artifact
  inventories instead of retaining all bytes; never mutates ledgers after the
  global terminal publication; uses the scheduler's process-birth marker v2;
  and applies the same compute-process, idle-memory, utilization, reservation,
  and headroom gates as production.
- The real full-profile GPU 0/GPU 1 canary has deliberately not been run yet.
  The integration suites are now green; run the canary only after the immutable
  production source snapshot is created. Its accepted terminal
  manifest/resource ledger remains a launch gate.
- The bounded CPU parallelism benchmark is implemented in
  `oci/inference/production_stage1_parallel_benchmark.py`, with public CLI
  `scripts/run_stage1_parallelism_benchmark.py` and artifact contract
  documentation in `docs/production_stage1_parallel_benchmark.md`. It runs
  identical canonical jobs at exactly 1, 4, and 8 loky workers, limits every
  observed native numerical pool to one thread, records source/code/input
  hashes and timings, requires exact scientific equality, and writes
  `terminal_manifest.json` last. The real clustered-preflight mode consumes
  authenticated private scope-input manifests and accepts no cohort or oracle
  path; it has deliberately not been launched yet.
- `/home/klkehl/thisenv/bin/pytest -q
  tests/test_production_stage1_parallel_benchmark.py` passed `5` tests in
  39.33 seconds on 2026-07-23. The finalized bounded fixture artifact is
  `artifacts/production_stage1_parallel_cpu_fixture_20260723_v2`; its fresh
  path-only validator accepted terminal summary SHA-256
  `19a08e542a8ecf7a22f9bc8d9a074362d70563442a414a8417e91d099fbbc6e3`.
  Cluster-fixture wall times at 1/4/8 workers were
  0.1187/0.1722/0.0511 seconds (speedups 1.00/0.69/2.32), and TF-IDF-fixture
  wall times were 0.5777/0.2633/0.1616 seconds (speedups 1.00/2.19/3.57).
  Scientific identities matched exactly across all worker counts:
  `4da417e96a772f77cb7c2a1c93f4a4f818864a0ffcd9e823d63e1dcb304f0375`
  for the cluster fixture and
  `ad6245df38f05b397fb65ff0409788366f3a28d87699df7a23bc67a90056b6f7`
  for TF-IDF. Timings exclude a declared per-worker import warmup, and each
  reusable loky executor is shut down between worker-count trials. The earlier
  `...cpu_fixture_20260723_v1` artifact is preserved but superseded because it
  allowed loky pool warmth from the 4-worker trial to bias the 8-worker trial.
- Final integrated parallel Stage 1 validation on 2026-07-23 covered the scope
  scheduler, one-scope adapter/fragments, isolated preflight inputs and
  consumer, terminal cluster-preflight artifact, exact and cumulative TF-IDF
  parallel/recovery paths, legacy TF-IDF backend, and the benchmark contract:
  `79 passed, 1 skipped in 218.10s`.

Final pre-snapshot test gate (2026-07-23):

- The initial broad collection finished with `961 passed, 1 skipped, 22
  failed`. This was retained as triage evidence, not treated as an accepted
  result.
- Isolated classification resolved every failure category:
  - The single workflow target passed in isolation.
  - The parent-agent token-bounded cluster-preflight check passed in isolation
    (`1 passed`), confirming that any nonempty token-bounded cache binding is
    rejected rather than treated as complete coverage.
  - All six HTR targets passed outside the restricted sandbox (`6 passed`);
    their broad-run failures were execution-environment effects, not accepted
    scientific-path failures.
  - The hierarchy-loader suite passed all `46` tests after correcting two stale
    synthetic-fixture omissions: the effective embedding-cache binding and the
    now-required native cluster-fit records/index with proper resealing.
- The consolidated cache-relocation, source-snapshot, checked-in-profile,
  workflow, and dual-GPU-canary group passed `55` tests.
- The consolidated workflow and isolated/sealed cluster-preflight group passed
  `36` tests.
- The final integrated parallel Stage 1 group was rerun after all source edits
  stabilized and passed `79` tests with `1` skip in `218.10s`.

- Resource checkpoint on 2026-07-23: both RTX A6000 devices reported 49,140
  MiB total, GPU 0 at 15 MiB/0% and GPU 1 at 188 MiB/0%, with no compute-app
  rows. Recheck immediately before both the canary and production launch; this
  checkpoint is not permission to kill a later occupant.
- The same resource state was freshly reconfirmed at 08:23 EDT on 2026-07-23
  with an escalated read-only `nvidia-smi` check. The source dataset SHA-256
  also still equals
  `6566aef4350ac1f78589d87d6a383bb49dce4a63ae5c9661921e925e3e854fe0`,
  the v2 work root and source-snapshot root remain absent, and `/data1` has
  approximately 1.8 TiB available.
- At 09:17 EDT on 2026-07-23 both RTX A6000 devices were again idle with no
  compute applications. GPU 0 reported 15 MiB used and 0% utilization; GPU 1
  reported 188 MiB used and 0% utilization. A read-only 09:41 EDT check
  reconfirmed the same values. Recheck again immediately before the canary and
  production launch; never signal an external occupant.
- Focused scheduler/cache/import/preflight/progress/resource tests are green.
- The broad pre-endpoint failures are classified and their corrected/focused
  suites are green. No real endpoint was contacted.
- The two previously recorded broad-suite failures were classified as
  expectation drift, deliberately corrected, and their focused tests pass.
- Run a real full-profile scope simultaneously on GPUs 0 and 1. Require exact
  scientific equality, peak reservation below 85% of 48 GiB, at least 6 GiB
  headroom, and useful throughput improvement.
- The canary must use the same immutable source snapshot and one identical
  private scope descriptor on both GPUs, vary only the operational GPU
  assignment, compare the authenticated scope accumulator/artifact scientific
  identities (not PID/timing/attempt envelopes), and poll `nvidia-smi` into a
  separate resource ledger. It must terminate and join both replicas on either
  failure or interruption.
- Reopen and validate the imported cache, preflight, scope fragments, aggregate
  indexes, component manifests, bundle, and handoff through fresh readers.
- Start the v2 Stage-1-only run only after all gates pass. Persist its immutable
  command, source snapshot, console log, progress ledger, benchmark, and GPU
  resource report.
- Immutable source snapshot created and freshly validated on 2026-07-23:
  `artifacts/production_all_evidence_one_conf_one_mod_1000_v2_parallel_stage1_source_snapshot`.
  It contains 193 read-only files and has content SHA-256
  `4b2794e74ef176ea9d6186c8631440632eabf0a68996d526e3a7f94b0554d01f`.
- The exact preparation-only prefix started from that snapshot at
  approximately 09:43 EDT. Its immutable request SHA-256 is
  `87aa3190519cf91d093752351e57f5533dee3dadf4ef7dd26979940ceef2e1a5`;
  endpoint/model are null and the request is Stage-1-only.
- Input preparation sealed at 10:03:36 EDT. It retained 1,000 unique patients,
  exactly the four configured columns, binary treatment/outcome, seven audited
  text transformations, source SHA-256
  `6566aef4350ac1f78589d87d6a383bb49dce4a63ae5c9661921e925e3e854fe0`,
  and prepared SHA-256
  `3b01d5c5d3a5b6729756b511e3f7bd1f8790d01c6d1c2f6143c02291610b3514`.
- Authenticated cache relocation opened at 10:08:35 EDT and sealed successfully
  at 12:20:55 EDT. The phase manifest is
  `artifacts/production_all_evidence_one_conf_one_mod_1000_v2_parallel_stage1/phases/embedding_cache/complete_manifest.json`;
  the console log is
  `artifacts/production_all_evidence_one_conf_one_mod_1000_v2_parallel_stage1.preparation.console.log`;
  durable state is in the v2 work root's `workflow_progress.json`.
- Stage 1 preflight opened at 12:25:47 EDT and failed closed at 13:09:32 EDT
  before any clustering worker or supervised Stage 1 fit began. The private
  preflight-scope config round-trip emitted `bow_fold_parallelism`, while
  `ExperimentConfig.from_dict()` constructs `MultiModelAgenticForestConfig`
  using the canonical constructor field `fold_parallelism`; validation raised
  `TypeError` rather than accepting a descriptor with a changed schema. The
  v2 root and incomplete attempt remain preserved, both GPUs are clean, and
  the completed input-preparation and embedding-cache phase manifests remain
  intact. Fix the live source with a focused full-production-config round-trip
  regression test, create a new authenticated source snapshot and fresh work
  root as required by the changed implementation identity, and reuse only
  artifacts accepted through an explicitly supported authenticated import
  boundary.
- The config-wire defect was fixed in the live source, not in the immutable v2
  snapshot. `production_stage1_effective_config_payload()` now supplies the
  request, `stage1_config.json`, and private preflight wire boundaries; the
  prepared config keeps `multi_model_agentic_forest` as its declared base
  type, while existing fitting code creates integrated aliases only in
  transient runtime copies. `bow_fold_parallelism` and
  `htr_fold_parallelism` remain distinct integrated controls under
  `multi_model_forest` and are never reinterpreted as `fold_parallelism`.
- Regression evidence after the fix: 26 preflight/legacy tests passed; 48
  bundle/profile tests passed; the complete integrated parallel Stage 1 gate
  passed `81 passed, 1 skipped in 233.62s`; and the workflow, authenticated
  relocation, source-snapshot, checked-in-profile, and dual-GPU-canary gate
  passed `55 passed in 43.09s`. Python compilation and `git diff --check`
  passed.
- The replacement run must use fresh absent paths
  `artifacts/production_all_evidence_one_conf_one_mod_1000_v3_parallel_stage1`
  and
  `artifacts/production_all_evidence_one_conf_one_mod_1000_v3_parallel_stage1_source_snapshot`.
  The v2 root is preserved and must not be resumed: its request binds the old
  source snapshot and code identities. The only supported cross-root reuse is
  its four-file relocated cache at
  `artifacts/production_all_evidence_one_conf_one_mod_1000_v2_parallel_stage1/phases/embedding_cache/attempt_20260723T140835185509Z/relocated_cache/embedding_cache`,
  paired with the original v1 prepared cohort and preparation manifest still
  named in that cache's immutable provenance. V3 must freshly authenticate and
  relocate those bytes.
- The corrected v3 source snapshot was created and independently reopened on
  2026-07-23 at
  `artifacts/production_all_evidence_one_conf_one_mod_1000_v3_parallel_stage1_source_snapshot`.
  It contains 194 files, has content SHA-256
  `a0190509f2e82b950baf01f379851408e7d8a4c6a8579ce2d6030580f39fd476`,
  contains no symlinks, and has read-only root/file modes 555/444.
- V3 canary-descriptor preparation launched from that snapshot at 14:00:54 EDT
  on 2026-07-23. It uses the supported v2 four-file cache import with the
  original v1 prepared-cohort provenance, has no Stage 2 endpoint/model, and
  stops before supervised Stage 1 so the GPUs are expected to remain idle.
  At 14:03 EDT the process was healthy but disk-bound in its first immutable
  Qwen-model identity scan: it had read about 6.76 of 15.14 GB (44.7%), was
  advancing around 55 MB/s, and had not yet created the work root or emitted
  console output. After authentication and descriptor preparation seal, run
  the selected real dual-GPU canary before resuming the production Stage 1
  workflow.
- At the user's direction, v3 was stopped and preserved at 14:27 EDT on
  2026-07-23 so the repeated-authentication defect can be fixed before more
  time is spent. Its input-preparation phase is complete; its embedding-cache
  attempt failed before relocation because the workflow had been launched
  inside a restricted sandbox where its child `nvidia-smi` returned exit 9.
  `workflow_progress.json` records the failure, the incomplete attempt remains
  present, there are no surviving workflow children, and both GPUs are clean.
  Do not resume v3. Launch the future v4 workflow outside the restricted
  sandbox so its exclusive-GPU preflight sees the host driver.
- The live relocation exposed avoidable repeated hashing within one immutable
  process: validation repeatedly scans the approximately 14 GiB Qwen model
  tree even though its path, file inventory, stat identity, immutable request,
  and process lifetime have not changed. Add a nonserializable run-scoped authenticated
  tree capability that reuses a completed tree digest only while a complete
  per-file device/inode/mode/size/mtime/ctime inventory still matches, and
  invalidates/fails closed on any change. Keep independent fresh-reader and
  resume validation as real byte rereads rather than memoizing across
  processes: `/data1` is SSHFS, so stat identity alone is not a cryptographic
  cross-process proof. The current imported embedding cache is only about
  663 MiB; the dominant cost is the 15.15-GB local Qwen model tree, currently
  reread about 25 times in the embedding-cache phase and about 50 times across
  canary-descriptor preparation including its independent validator. Do not
  modify `production_embedding_cache_builder.py`, whose exact source identity
  is embedded in the historical cache provenance. Add the shared capability
  outside that module, let relocation compare its authenticated tree identity
  with the builder provenance, and call the historical builder validator
  without asking it to reopen the already-authenticated live model path. After
  a relocation is terminally sealed, downstream consumers should authenticate
  its self-contained destination
  cache/attestation without repeatedly reopening the now-unused source model
  and source cache; retain the frozen model digest as provenance. Target one
  full main-process model authentication plus one independent fresh-process
  reread. Add call-count, mutation, inode-replacement, child/fresh-process
  isolation, and benchmark tests. Longer term, prefer native-storage
  `fs-verity` (or another backend-enforced immutable object/version) so a fresh
  process can authenticate kernel Merkle roots without rereading all bytes;
  read-only modes or a plain content-addressed directory are not sufficient.
- The authentication optimization was implemented for v4 on 2026-07-23.
  `production_authenticated_tree_cache.py` performs one full stable byte hash,
  keeps a PID-only/nonserializable/fork-reset capability, compares the complete
  root/directory/file stat inventory on every reuse, poisons changed paths, and
  has bounded live/poison registries. The workflow enables it only for an
  imported embedding cache/model; fresh cache builds and the live HTR model
  retain the historical full-byte path. Relocation v2 still runs the complete
  historical cache validator with `expected_local_model_path=None`, separately
  binds all five authenticated model-provenance fields, brackets every
  validation/copy boundary with inventory checks, and records both the
  relocator and authenticated-tree module hashes. The historical cache builder
  was not edited and remains SHA-256
  `9af77ce3cc47ea77c819974f4b55885ddeb279f758bbac6ca5b987ac9d61aabd`.
- Real Qwen verification matched the sealed-cache provenance exactly: 17 files,
  one subdirectory, 15,150,575,778 bytes, and tree SHA-256
  `c905c538fb4ea49243eea098e68aa6f6d17a1e0c13c3e035c6b8521bde0caa53`.
  The first full read took 293.658 seconds on SSHFS; the second complete
  same-process inventory check took 0.002018 seconds (about 145,000 times
  faster). This is not described as adversarially equivalent to a rehash:
  every resume and independent validator starts in a fresh process and performs
  its own full byte authentication.
- Final verification after the optimization and edge-case fixes includes:
  80 workflow/cache/snapshot/profile/canary tests passed; 106
  authentication/relocation/workflow/bundle tests passed; 20
  source-snapshot/profile/canary tests passed; the integrated parallel Stage 1
  gate passed 127 with one skip; and the final targeted Stage 1 consumer gate
  passed 55. Compilation, Black, `git diff --check`, exact dataset/profile
  hashes, and the historical builder hash all passed.
- The immutable v4 source snapshot was created and independently reopened at
  `artifacts/production_all_evidence_one_conf_one_mod_1000_v4_parallel_stage1_source_snapshot`.
  It contains 195 files, no symlinks, read-only root/file modes 555/444, and
  content SHA-256
  `1caecd698ec2c637dc1cc999b1d5e626fa8c7c7a186762efb88fa1e581e999b7`.
  Use fresh work root
  `artifacts/production_all_evidence_one_conf_one_mod_1000_v4_parallel_stage1`
  and launch outside the restricted sandbox so the host `nvidia-smi`
  preflight is available. Reuse only the sealed v2 four-file cache with its
  original v1 preparation provenance; do not resume v3.
- V4 canary-descriptor preparation launched outside the restricted sandbox at
  15:35 EDT on 2026-07-23. Its immutable workflow request has SHA-256
  `8435b0fc37dfff7fe9d5b9752ec0f4516a8059a1f3c5be31558b7edffcea714c`.
  Input preparation sealed 1,000 rows with prepared-cohort SHA-256
  `3b01d5c5d3a5b6729756b511e3f7bd1f8790d01c6d1c2f6143c02291610b3514`;
  it preserved row order/non-text values, materialized no oracle values, and
  audited affected unit IDs 57, 186, 241, 637, 661, 784, and 942.
- The fresh v4 embedding-cache relocation sealed at 15:58 EDT on 2026-07-23
  and the workflow advanced to `stage1_preflight`. Process inspection proved
  that Qwen was fully byte-authenticated once and was not reopened during the
  repeated source/destination cache-validation passes. Both GPUs remained
  free. The cache phase has a complete v2 workflow manifest; resume may skip it
  only after the normal manifest/input revalidation.
- At 16:52 EDT on 2026-07-23, preflight had durably published all 40 physically
  restricted fit-only scope inputs in canonical order: five full-outer, 25
  exact-inner, and ten cumulative-review scopes. The closed
  `preflight_scope_input_set_manifest.json` exists. The coordinator was then
  performing the required full-set revalidation before launching the
  configured eight loky clustering workers; no supervised fit or GPU work had
  begun.
- At approximately 17:10 EDT, full-set revalidation passed and all eight
  configured loky preflight workers launched together. Host inspection showed
  eight distinct workers concurrently reopening their first private scope
  inputs at about 1 GB RSS each. The coordinator and workers remained
  CPU/disk-bound and both GPUs remained intentionally unused.
- All 40 parallel clustering fits returned successfully. Preflight
  finalization exposed a future storage/performance item: the canonical
  `cluster_feasibility_audit.json` is approximately 3.91 GB. Preserve the
  current immutable run, but later replace giant numeric JSON payloads with a
  closed, non-pickle binary array/artifact layout plus canonical manifests and
  per-array hashes. The replacement must retain every fitted/emitted identity,
  exact ordering, fresh-reader verification, and comparison of all 40 fits; it
  must not weaken or summarize away the clustered-embedding proof.
- Stage 1 clustered-embedding preflight sealed successfully at 18:18:46 EDT on
  2026-07-23. Its report records
  `scientific_cluster_preflight=accepted_and_independently_sealed_v1`, eight
  workers, and exact scope counts 5 full-outer, 25 exact-inner, ten
  cumulative-review, 40 total. `workflow_progress.json` lists
  `stage1_preflight` as completed and moved to
  `canary_descriptor_preparation`. The current proof format also writes a
  roughly 4.08-GB `stage1_preflight_request.json`; include that duplication in
  the future binary-manifest redesign.
- Live canary-descriptor preparation showed that a downstream
  `ProductionStage1BundleBuilder.prepare()` still reopens the historical v2
  source cache while validating the already sealed v4 relocation. This is
  scientifically safe and much smaller than the Qwen tree (about 627 MB of
  embeddings plus companions), but it is unnecessary repeated I/O. For a
  future run, add a source-free consumption validator for a terminally sealed
  relocation that authenticates only its closed destination
  cache/attestation/terminal manifest and retains the source-cache/model
  digests as provenance. Keep full source validation at the initial import
  boundary and in an explicitly independent audit, not every downstream
  consumer.
- V4 canary-descriptor preparation failed after safely publishing the first of
  40 descriptors. The clustered fit and sealed preflight remain accepted; the
  failure was a serialization/reader mismatch in the later private descriptor
  handoff. `outer_001_full/cluster_preflight_projection.json` was 87,668,273
  bytes when pretty-printed and exceeded the descriptor reader's defensive
  64-MiB JSON limit. V4 is stopped and preserved with no descriptor-set or
  canary terminal manifest. Do not resume it: its immutable snapshot would
  reproduce the failure, while a corrected snapshot changes the request
  identity.
- The immediate correction keeps the 64-MiB reader limit and writes only the
  per-scope cluster-preflight projection as canonical compact JSON. A streaming
  audit of all 40 accepted v4 identities found compact sizes from about
  52.0 MB through 59,850,458 bytes, versus pretty-printed sizes through
  88,906,851 bytes. Thus every real projection fits without dropping,
  summarizing, or changing any scientific evidence. The adapter suite passes
  `9 passed`; the focused compact-serialization/unchanged-size-guard regression
  passes; compilation and Black checks pass for the edited files.
- Next live action: create and independently validate a fresh immutable v5
  source snapshot and use a fresh absent v5 work root. The supported embedding
  cache import remains reusable, but the current workflow has no authenticated
  cross-request preflight-import boundary, so v5 must repeat the 40 clustered
  preflight fits. Prepare all 40 descriptors, run the selected full-outer
  two-GPU reproducibility canary, and only after it is accepted resume the same
  v5 request without the preparation-only flag to begin all ten Stage 1
  evidence architectures.
- V5 source snapshot creation and launch completed on 2026-07-23. The snapshot
  contains 195 files, no symlinks, read-only mode 555/444, and content SHA-256
  `ede2d093c5f51905b28fee73be1de47bb3a025b3142e068fd6394d491e61aa79`.
  The 54-test workflow/descriptor/snapshot/canary gate passed before freezing
  it. The fresh work root is
  `artifacts/production_all_evidence_one_conf_one_mod_1000_v5_parallel_stage1`;
  immutable request SHA-256 is
  `b79af856b80cb7a04563536e289ffea479ec57010ce37c3bdd4fe92473cf5357`.
  The request has no endpoint/model/oracle, uses GPUs 0 and 1, and differs from
  v4 only in new snapshot/work/recovery identities and implementation hashes.
  Input preparation is complete and the live preparation-only process has
  advanced to authenticated embedding-cache relocation. Do not edit the v5
  source snapshot or use the v4 work root.
- At the user's direction, the v5 preparation/preflight run was stopped cleanly
  at 20:28 EDT on 2026-07-23 so work can focus on checkpoint adoption and Stage
  1 simplification. The workflow exited 130 and durably recorded
  `KeyboardInterrupt`, status `failed`, with input preparation and embedding
  cache as its only two completed phases. Its one incomplete
  `stage1_preflight` attempt is preserved with 36 of 40 independently
  published private scope-input manifests, no preflight terminal manifest, no
  clustering-worker phase, and no supervised Stage 1 fit. A host-level process
  check found no remaining v5 process or descendant. GPU 0 was idle at 15 MiB
  and GPU 1 at 188 MiB, both at 0% utilization. Do not resume, relabel, delete,
  or modify v5 unless the user explicitly directs it; await the simplification
  work plan and use authenticated adoption rather than loose files if any of
  its completed artifacts are consumed later.
- Add authenticated cross-version phase adoption before treating this workflow
  as operationally mature. Existing `--resume` correctly recovers interruption
  under one byte-identical request, but every phase manifest is bound to the
  global request SHA. That request contains the work root, source-snapshot
  root/content identity, broad implementation hashes, and run-local recovery
  paths. Stage 1's current behavior identity is broader still and hashes every
  `oci/*.py`. Consequently, a downstream-only serializer fix invalidates
  preparation, cache, and clustered preflight even though their scientific
  inputs and producer code did not change.
- Implement a phase DAG and a two-level identity:
  a path-neutral scientific compatibility key for each phase, plus a run-local
  execution/locator attestation. Each portable phase artifact must bind its
  exact input artifact IDs, scientific configuration projection, seed/runtime
  settings that can affect results, phase-specific transitive producer-code
  identity, closed output inventory, schemas, ordering, and content hashes.
  Work-root paths, console paths, downstream implementation files, and
  unrelated source files must not enter the scientific compatibility key.
- Add an explicit `adopt-checkpoint` operation. A fresh reader from the new
  snapshot must reopen the old terminal manifest and all registered bytes,
  independently validate the scientific payload against the new phase inputs,
  compare the phase compatibility key, and write an immutable adoption
  attestation linking producer run/snapshot/artifact IDs to the consumer
  request. No loose-file copying, partial attempts, manual digest approval,
  `--force`, or schema downgrade is allowed. Exact phase-code matches may adopt
  automatically; changed producer code requires a narrow versioned migration
  whose new validator can prove the complete payload and dependencies.
- Make the artifact content address independent of its physical locator.
  Preserve an authenticated read-only source locator or publish into a
  content-addressed object store; do not rewrite or duplicate multi-gigabyte
  preflight JSON merely to change its work-root path. The consumer must
  reauthenticate the referenced object and fail closed if it disappears or
  changes.
- Split Stage 1 checkpoints below the monolithic phase: prepared cohort,
  embedding cache, 40-fit clustered preflight, descriptor/canary packaging,
  each per-scope all-source fragment, TF-IDF component, neural-query component,
  fragment merge, and terminal handoff. Invalidation should follow dependency
  edges only. The v4 descriptor serialization defect should therefore require
  rewriting descriptors only; its sealed preparation, cache, and 40 fitted
  clustered identities are candidates for authenticated adoption.
- Keep the current immutable v5 process running while this recovery layer is
  designed and tested. Do not restart v5 merely to add the future importer:
  the importer should be able to consume existing v4/v5 sealed artifacts
  retroactively after full validation. If v5 completes normally, continue it;
  if a later downstream defect appears, use the tested adoption path in a
  fresh run instead of recomputing accepted upstream phases.
- Longer term, avoid duplicating the very large final cluster catalogs in JSON
  at all. Move them to closed content-addressed non-pickle binary artifacts
  with per-array hashes and small canonical manifests while preserving exact
  raw, semantic, and final-catalog identities and ordering.

## Simplify Stage 1 coordination before the next full production run

Status: read-only audit complete on 2026-07-23; implementation and
representative benchmarks pending. Keep the immutable v5 preparation/preflight
run running. Do not patch its snapshot or stop it merely to apply this future
design. Any implementation requires a new authenticated snapshot/work root, or
the explicit checkpoint-adoption boundary described above.

Scientific guardrail: preserve all five full-outer, 25 exact-inner, and ten
cumulative-review contexts and all ten Stage 1 evidence architectures. The
goal is to remove redundant copying, serialization, authentication, replay,
and refitting—not to replace native embedding evidence, collapse the
training-only review design, inspect the oracle, or relax held-out isolation.

### Measured baseline and diagnosis

- V4 clustered preflight ran from 15:58:05 through 18:18:46, or 140 minutes 41
  seconds. Publishing the 40 private inputs took 54 minutes 52 seconds,
  whole-set revalidation took about 17 more minutes, and post-fit
  sealing/validation took about 27 minutes. The eight-worker fit window was at
  most about 41 minutes and included giant JSON encoding. Thus at least 99
  minutes, or at least 70% of the phase, was coordination rather than the
  clustered fits themselves.
- The immutable global embedding cache is approximately 0.647 GiB. The 40
  preflight scope inputs expand it to 17.177 GiB: 14.950 GiB of embedding
  matrices, 1.604 GiB of chunk text, and 0.621 GiB of note Parquets. Descriptor
  publication writes essentially the same private cache set again, reaching
  roughly 34 GiB, or 53 times the source cache, before supervised modeling.
- A live v5 checkpoint before clustering had already accumulated roughly 165
  GB of process reads and 34 GB of writes while using about 0.4 CPU core and
  neither GPU. It was repeatedly reopening a roughly 402-MB private
  `chunk_embeddings.npy`. The visible output tree was only about 12–13 GiB at
  that point, proving that most traffic was reread/rehash/revalidation rather
  than unique scientific output.
- The workspace is SSHFS, whereas `/tmp` is local ext4 with approximately 237
  GB free. Repeated copy, hash, JSON parse, and `fsync` operations therefore
  pay remote-filesystem latency and bandwidth.
- V4 wrote a 3,905,543,170-byte `cluster_feasibility_audit.json` and a
  4,082,573,444-byte `stage1_preflight_request.json`; the latter embeds the
  former's information. Sealing, reopening, hashing, parsing, deep-copying, and
  canonicalizing those payloads consumed tens of minutes. A single compact
  cluster-fit identity is still about 52–60 MB because the final catalogs
  expand tens of thousands of three-member evidence atoms.
- `ProductionStage1BundleBuilder.prepare()` is invoked for preflight, canary
  descriptor preparation, and production build. It repeats cohort parsing,
  cache/model binding, HTR coverage checks, and reopening of the giant
  preflight proof.
- A v1 exact-inner scope spent at least 69 minutes after its HTR fits writing
  and validating approximately 0.886 GiB of proof artifacts, before later
  merge/finalization copies. BoW, HTR, matched-patient, and evidence metadata
  duplicate numerical content already stored in arrays.
- Worker fragments are authenticated and copied into a merge tree, then hashed
  and copied again into the finalized component. Native proof validators can
  reconstruct/replay the same fitted models repeatedly at registration,
  per-family proof validation, merge, bundle validation, and terminal loading.
- The clustered preflight already computes all 40 native cluster outputs, but
  supervised Stage 1 refits them solely to compare equality. The real
  dual-GPU canary fits one full-outer scope twice and production would fit it a
  third time.
- TF-IDF main and cumulative contexts already use bounded `loky` process
  parallelism. Descriptor publication and validation are storage-bound, so
  adding more processes there before removing byte duplication is likely to
  increase contention rather than reduce wall time.

Primary code paths identified in the audit:

- Private cache materialization:
  `production_stage1_legacy_scope_adapter.py::_write_private_embedding_cache`
  and `production_stage1_preflight_scope_inputs.py::_write_scope`.
- Repeated per-scope/set validation:
  `publish_preflight_scope_inputs()`, `validate_preflight_scope_input()`, and
  `validate_preflight_scope_input_set()`.
- Giant proof sealing/loading:
  `production_stage1_cluster_preflight_artifact.py`.
- Repeated preparation:
  `ProductionStage1BundleBuilder.prepare()` and `build()`.
- Copy-heavy assembly:
  `merge_legacy_stage1_scope_fragments()` and
  `finalize_legacy_stage1_component_from_merge()`.
- Repeated model replay:
  `validate_htr_native_capture()` and the native-family proof-index validators.

### P0 — remove I/O and authentication amplification

1. Add a representative full-scope performance ledger before changing the
   design. Record wall and CPU time plus bytes read, written, copied, hashed,
   compressed, decompressed, JSON-encoded/decoded, and `fsync`ed by subphase;
   also record model-fit/replay time, process startup/import time, GPU
   utilization, and peak memory. The current microfixture benchmark is not a
   substitute for this measurement.
2. Replace 40 durable private cache copies with one immutable global cache and
   an authenticated exact-row scope capability. Use the same sealed capability
   for preflight and supervised Stage 1. A trusted row-view API, read-only
   memory map plus signed row manifest, or parent-owned shared-memory/memfd
   broker is acceptable. If hostile-child isolation is genuinely required,
   implement a real OS boundary; same-UID physical copies provide expensive
   pseudo-isolation without a kernel security boundary.
3. If a physical scope view remains necessary, materialize only active views
   on local ext4 scratch, publish the sealed scientific result once, and remove
   the transient view after successful publication. Do not persist 40 copies
   twice on SSHFS.
4. Replace the multi-gigabyte preflight JSON with independently closed
   per-scope binary artifacts using a safe non-pickle format such as NPY plus
   manifests, Arrow, or safetensors. Store arrays/numerical evidence once,
   retain canonical ordering and per-array hashes, and assemble a small ordered
   40-scope index. `stage1_preflight_request` must reference the audit artifact
   by immutable content identity rather than embedding it.
5. Seal one path-neutral typed `PreparedStage1Context` containing the prepared
   cohort identity, split plan, effective profiles, cache capability, HTR
   audit, and preflight index. Preflight, canary, modeling, and validators
   consume this context rather than rerunning the monolithic `prepare()`.
6. Authenticate each immutable byte tree once per fresh trust boundary. Within
   one process, reuse a nonserializable open-file/stat-inventory capability and
   fail closed on mutation. Set-level validators should authenticate sealed
   child manifests/Merkle roots rather than reread every matrix. Keep one
   explicit independent full-byte terminal audit; do not recursively rehash
   the same nested objects at every wrapper.
7. Hash bytes while they are first written/copied and bracket publication with
   mutation checks instead of immediately rereading them solely to create a
   registration. `fsync` each payload and directory once bottom-up, then write
   the terminal marker; do not rescan and resync already durable child trees.
8. Implement the phase-local compatibility keys and explicit
   `adopt-checkpoint` DAG above before another downstream-only defect can force
   accepted preparation, cache, or preflight work to restart.

### P1 — eliminate redundant proof/model work

1. Make each worker publish a collision-free immutable
   `scopes/<scope_id>` tree. The parent validates its terminal manifest and
   builds small aggregate indexes that reference those trees. Remove both
   full-byte merge and finalization copies while retaining one fresh final
   reader.
2. Store every numerical proof array exactly once. Evidence families,
   inventories, fitted-model records, and agent-facing views should reference
   immutable array IDs/slices rather than repeating vectors in NPZ, large JSON
   metadata, sidecars, and aggregate payloads.
3. Perform one independent deterministic replay per sealed fitted artifact,
   memoized by content hash within a process. Family-specific proofs should
   reference that authenticated capability. A fresh terminal process still
   performs one full replay; it must not replay the identical artifact once per
   wrapper or family.
4. Treat the sealed 40-scope clustered-preflight outputs as the actual
   cluster-local native evidence consumed downstream. Do not perform the same
   cluster fits again merely to prove that a refit matches. This strengthens
   exact provenance while preserving the mandatory native embedding source.
5. After a dual-GPU canary proves exact equality and resource safety, adopt one
   accepted replica as that production scope fragment. Alternatively move
   determinism checking to a small release-level fixture. Never fit and discard
   two full production scopes and then fit the identical scope a third time.
6. Benchmark persistent one-worker-per-GPU services against fresh
   per-scope processes. A service may load Python/Torch, HTR resources, and
   authenticated cache handles once, but it must reset RNG and model state,
   prove schedule-independent scientific equality, release memory between
   scopes, and preserve failure isolation. Also benchmark two concurrent
   roughly 9-GiB HTR scope workers per 48-GiB A6000, larger HTR
   training/encoder batches, data-loader workers, and tokenizer/chunk-plan
   caching.
7. Keep the working global TF-IDF `loky` lane. Use one resource-aware host CPU
   budget so eight TF-IDF processes, GPU data feeders, compression, and proof
   validation do not oversubscribe the eight physical cores. Benchmark one
   neural-query context per GPU against the current within-context two-GPU
   strategy, and give TF-IDF/query contexts their own sealed recovery records.
8. Replace single-threaded `np.savez_compressed` on large captures with a
   mmap-friendly safe layout and measured compression policy. Prefer
   uncompressed local scratch when compression CPU and repeated decompression
   cost more than the reduced one-time publication.

### P2 — deduplicate scientifically equivalent work

- Explicitly deduplicate the five set-identical scope pairs:
  `outer_001_hierarchy_epoch_001` through
  `outer_005_hierarchy_epoch_001` each have the same 640 fit rows and 160
  sealed rows as the corresponding `outer_*_inner_005`, although their current
  row ordering and scope-derived seeds differ. Keep all 40 logical scientific
  contexts, but canonicalize fit/sealed row order and key the underlying
  architecture fit by canonical row-set hash, target, scientific
  configuration, producer code, and a content-derived fit seed. Fit each
  equivalent pair once, then emit two separately bound scope records with
  their own purpose, evaluation view, and provenance. Do not merely relabel an
  artifact produced under a different seed. Prove equality for every one of
  the ten evidence families and all five outer folds, and retain fail-closed
  tests for a changed row, order-sensitive input, configuration, target,
  producer, or seed. This should reduce the 40 logical contexts to at most 35
  physical all-architecture fits without removing either review context.
- Generalize fitted-work caching around
  `(architecture, target, canonical fit-row hash, scientific config, seed,
  producer-code identity)`. Audit cross-fitting and nuisance-model keys for
  further exact duplicates. This is higher scientific-change risk than the P0
  byte-layout work and requires output-equivalence tests.
- Maintain the complete lossless evidence banks for audit and numerical
  estimation, but define a deterministic bounded agent-facing view with
  explicit per-architecture, per-contrast, and distribution-tail coverage.
  Evaluate and freeze that policy without oracle access. Do not equate “all ten
  evidence sources ran” with serializing every three-member catalog atom into
  every prompt.

### Performance and scientific acceptance

- Establish a budget that coordination/proof overhead is no more than 20–30%
  of measured model-fit wall time on representative scopes, and that a phase
  reads no more than about twice its unique immutable input bytes absent one
  explicitly recorded independent audit.
- Benchmark the current immutable implementation and each simplification on
  real 800/640-row scopes. Record end-to-end time, not just worker fit time.
- Require exact scientific equality wherever storage, copying,
  authentication, or scheduling alone changes. For intentional evidence-view
  compaction, require declared deterministic coverage and separate lossless
  backing artifacts.
- Preserve exact scope counts, all ten nonempty evidence families, native
  whole-cohort and cluster-local embeddings, frozen split identities, strict
  outer-held-out label isolation, deterministic seeds, fail-closed validation,
  and fresh terminal loading.
- Add interruption/tamper tests at every terminal boundary. A completed scope
  or component must be reusable by identity; partial trees, loose files,
  mismatched manifests, or cache/model substitutions remain non-reusable.
- Do not prioritize more descriptor-publication workers until the redundant
  SSHFS bytes are removed. The expected largest first wins are the shared
  scoped cache, binary/reference-based preflight proof, narrow cached
  validation handles, zero-copy fragment assembly, and productive reuse of
  preflight/canary computation.

- Historical test-run bookkeeping: a combined recovery suite on 2026-07-23 reached
  `56 passed` and one spawned adapter failure while
  `production_stage1_scope_scheduler.py` was being edited by the TF-IDF/process
  cleanup implementation between parent descriptor publication and child
  import. The child correctly rejected the now-different plan. This is not an
  accepted stable result. It has been superseded by the stable integrated
  `79 passed, 1 skipped` result above.

Expected Stage 1 wall time after validation: approximately 4–6 days. This is an
estimate, not an acceptance criterion.

Camus/vLLM timing: keep the Camus server off during the preparation prefix,
dual-GPU canary, and the entire v2 Stage-1-only run. This workflow has no
endpoint/model requirement and cannot contact Stage 2. The earliest likely
need is approximately 5–7 days after the production Stage 1 launch (roughly a
six-hour canary plus the estimated 4–6-day Stage 1 run and remaining Stage 2
gates). Re-estimate from the durable progress ledgers and explicitly notify the
operator before asking for Camus; do not start it speculatively.

### Deferred until the immutable Stage 1 run is underway

- Finish the production `complete_paged_v1` citation/reconciliation protocol.
- Strengthen independent terminal inference/oracle validation.
- Audit the checked-in production profiles.
- Complete embedding-builder cleanup tests. The immediate v2 workflow must
  release cache-model memory through a separate terminating cache process.
- Do not contact the real Stage 2 endpoint until all deferred P0 tests pass.

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

Status: complete for checked-in content; invocation overrides remain bound by
the workflow request and its tests.

- Preliminary closed read on 2026-07-23 confirmed 5 outer folds, 5 candidate
  consistency partitions, 50 nuisance/effect/Stage-1 epochs, 120 query epochs,
  80 final-refit epochs, the requested 10-cluster/20-initialization support
  settings, 128 embedding chunks, 512 HTR chunks, and the 200-tree honest
  inference-enabled forest.
- The only credential-shaped fields observed were the literal neutral
  placeholders `agent_api_key="EMPTY"` and `vllm_api_key="EMPTY"`; no endpoint
  or production model identity is stored in these profiles.
- Both JSON objects were accepted by `load_applied_stage1_config` and the
  production closed neural-query loader without unknown fields.
- Stage 1 profile SHA-256:
  `1af35bb0a107c28a79a76fa74319de105d2ee4352c12345d8bdbe97869b9cfc0`.
- Neural-query profile SHA-256:
  `2d465f6c2eae71d4c9f4d18716f0919aee954b0afde9ef4414a27c5ad4771997`.
- Confirmed outer/candidate folds `5/5`, training and native HTR
  nuisance/effect epochs `50`, TF-IDF calibration folds `3`, explicit
  interaction folds `3`, query epochs/refit/reviews `120/80/2`, and query inner
  folds `5`.
- Confirmed Qwen cache settings `256/64/128/1024`, batch size `1`, normalized
  embeddings, fixed K-means `10/20`, support `24/8/4`, five components, and
  every requested cell/residual/confounder/interaction contrast enabled.
- Confirmed trainable HTR settings `96/24/512/512`, encoder batch `16`, and
  confirmed both final forest profiles use 200 trees, leaf size 10, `sqrt`,
  honesty, and inference.
- The only raw credential-shaped values are literal `EMPTY`; production
  endpoint/model values are absent and stored Stage 1 model labels are
  explicitly unused placeholders.
- The audit is now executable in
  `tests/test_production_all_evidence_profiles.py`; direct execution with
  `/home/klkehl/thisenv/bin/pytest` passed `2` tests on 2026-07-23.

## Resolve or formally classify remaining broad-suite failures

Status: complete for the two previously recorded failures; full broad rerun is
still part of Milestone 6.

The broader pre-endpoint suite last reported 334 passes and two failures:

1. Qwen embedding model backend/pooling default in
   `tests/test_extractors.py::TestHierarchicalTransformer::test_sentence_encoder_backend_and_pooling_defaults`.
2. Packed contract-RAG configuration versus adaptive-review request-locality in
   `tests/test_contract_lexical_context.py::test_fusion_cli_wires_packed_contract_rag_and_composite_cache_identity`.

Both were expectation drift:

- HTR encoders are trainable by default, so automatic Qwen
  sentence-transformers selection is now tested only for an explicitly frozen
  encoder; the default trainable Qwen path is correctly `transformers`.
- Adaptive review is mandatory and requires one feature per extraction
  request, so the packed-contract test now proves a cap of four is rejected and
  verifies packed/RAG cache wiring with the required contract-local cap of one.

Validation on 2026-07-23:

`uv run --active --no-sync --frozen pytest -q
tests/test_extractors.py::TestHierarchicalTransformer::test_sentence_encoder_backend_and_pooling_defaults
tests/test_contract_lexical_context.py::test_fusion_cli_wires_packed_contract_rag_and_composite_cache_identity`

Result: `2 passed`.

## Live-run recovery record

Status: stopped and preserved; never resume v1 after the parallel code change.

- Work root:
  `artifacts/production_all_evidence_one_conf_one_mod_1000_v1`
- Former Python PID `3644296` exited cleanly after the recorded `SIGINT`; no v1
  workflow process remains.
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

## Close private Stage 1 workers to one-scope split authority

Status: implementation complete; focused and integrated consumer suites green
on 2026-07-23.

- Clustered-embedding preflight scope inputs now use
  `production_stage1_preflight_one_scope_authority_v1`. A child receives the
  selected scope only; `split_registry.json`, peer scope definitions, and peer
  row-identity lists are absent. The parent still owns the complete registry
  and restores canonical order before validating and hashing the 40 results.
- Legacy all-source descriptors now use
  `production_legacy_stage1_one_scope_authority_v1`. A child no longer receives
  either `split_registry.json` or `scope_plan.json`; it receives one selected
  scope/assignment, its split fingerprint where applicable, and opaque
  registry/plan/schedule hashes.
- `_run_legacy_component(selected_scope_id=...)` no longer reconstructs or
  reads the global registry/schedule. It runs from the selected authority.
  Scope fragments can be sealed with one `Stage1ScopeSpec` plus the opaque
  parent-plan hash, while the parent reopens every fragment against the full
  canonical plan during aggregation.
- Filesystem-enumeration and serialized-worker-payload tests prove that a
  selected descriptor has no peer scope ID, peer split definition, or peer
  fit/held-out row list. A scientific-path test makes global-registry access
  fatal and proves exact-inner and cumulative selected paths reach the model
  boundary without it.
- No all-scope split-row authority remains necessary in a child. The residual
  global-row-space facts are only cohort cardinality, empty non-fit cache
  offsets/rows, and opaque hashes. They are needed to preserve original
  logical row positions and frozen-cache/proof identities and contain no peer
  text, treatment, outcome, or oracle value.
- This is application-level capability confinement, not an adversarial OS
  sandbox: all loky children run under the same Unix identity. If scientific
  worker code itself becomes hostile, separate UIDs/containers or an
  equivalent kernel isolation boundary is still required.

Focused validation:

- `tests/test_production_stage1_preflight_scope_inputs.py`,
  `tests/test_production_stage1_preflight_parallel_isolation.py`, and the real
  serial/parallel fixture
  `test_embedding_cluster_preflight_enumerates_all_native_scopes_with_real_catalogs`:
  `5 passed` in 29.42 seconds.
- `tests/test_production_stage1_legacy_scope_adapter.py` plus
  `tests/test_production_stage1_legacy_scope_fragments.py`: `19 passed` in
  85.64 seconds, including spawned-worker/resume, closed-tree enumeration,
  one-scope child sealing, and parent full-plan reopening.
- Edited modules compile under `/home/klkehl/thisenv/bin/python`; Black checks
  pass for the new private-input/descriptor modules and tests, and
  `git diff --check` passes for the shared bundle edit.

## Parallel TF-IDF lane and recoverable component execution

Status: implementation complete; focused and integrated suites green on
2026-07-23.

- The exact TF-IDF stage now creates all 30 independent contexts (five
  full-outer plus 25 exact-inner fits) up front, schedules them globally with
  bounded loky workers, restores canonical order, and limits native numeric
  threads to one per child.
- Each production held-out context serialized to a worker contains exactly
  `_oci_row_id` and the configured text column. Treatment/outcome mutations on
  held-out rows cannot alter the serialized worker input. Fit labels remain
  available only where scientifically authorized.
- The ten cumulative review TF-IDF scopes are separate spent-only loky tasks.
  Their payloads bind the exact spent text/labels, sealed row IDs, canary,
  scientific TF-IDF configuration hash, global seed, and derived scope seed;
  no source cohort, embedding cache, external corpus, or full dataframe locator
  is accepted. Missing, duplicate, reordered, and output-alias results fail.
- Python, NumPy, and Torch are explicitly seeded in every exact/cumulative
  worker, strict Torch determinism is enabled, and the run seed is propagated
  from the production options instead of relying on the historical default.
- Loky imports are serialized only during worker bootstrap. This is necessary
  on the current FUSE/SSHFS workspace, where four simultaneous imports of the
  model stack intermittently returned `EPERM`; model fitting remains parallel.
- The whole TF-IDF component starts concurrently with the dual-GPU legacy
  all-source lane. It publishes a closed authenticated descriptor, executes in
  a fresh spawned process, seals the complete component in a stable recovery
  attempt, and is materialized into the current workflow attempt only after
  authentication. Incomplete publications/attempts are preserved and ignored
  on identical resume; only a complete matching seal is reusable.
- Descriptor and attempt validation binds current code hashes, request hash,
  exact closed schemas, file hashes/inventory, root and attempt device/inode,
  descriptor path/identity, seed, worker result, and terminal durability.
  Symlinks, hardlinks, extra files, code changes, directory substitution, and
  request/result tampering fail closed.
- Every spawned scope/component worker creates a private POSIX session and
  durable PID/PGID-ready marker before it may create descendants. Peer failure,
  parent interruption, or cancellation sends TERM and then KILL to that private
  group, preventing loky/DataLoader descendants from surviving.

Focused validation completed on 2026-07-23:

- TF-IDF component recovery/security suite: `8 passed in 33.81s`.
- Real process-group descendant sentinel: `1 passed` (with 22 deselected) in
  `31.19s`.
- Exact TF-IDF held-out serialization and process-worker seed path: `2 passed`
  (with 9 deselected) in `39.62s`.
- Cumulative scheduler contract tests: the two non-loky cases passed in the
  three-test run; after the SSHFS bootstrap fix, the real ten-scope loky
  serial-versus-parallel proof passed (`1 passed`, 2 deselected) in `80.20s`.
- All edited Python modules/tests compile, and `git diff --check` passes for the
  tracked shared edits. The combined integrated result is `79 passed, 1
  skipped in 218.10s`, as recorded under Milestone 6.
