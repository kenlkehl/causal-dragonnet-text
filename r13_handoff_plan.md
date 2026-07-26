  # Handoff: Complete the Portable All-Evidence Causal Pipeline on the Higher-Powered Host

  ## 1. Mission and scientific contract

  Work in /data1/ken/pcori_dev/causal-dragonnet-text and use only:

  /data1/ken/envs/gptoss3/bin/python

  First verify that interpreter resolves and imports Torch, NumPy, pandas, scikit-learn, EconML, PyArrow, Joblib, and Transformers. Its symlink is broken on the old host because /homes/klkehl/... is absent; it
  is expected to resolve on the new host. If it does not, repair this same environment before proceeding—do not silently substitute another Python environment.

  The scientific objective is patient-level probability-scale treatment-effect heterogeneity:

  # [
  \widehat{\tau}(X_i)

  ## P(Y=1\mid do(T=1),X_i)

  P(Y=1\mid do(T=0),X_i)
  ]

  Version 1 supports binary treatment and binary outcome. Causal interpretation relies on consistency, positivity, and conditional exchangeability after recovering adequate note-readable adjustment variables.

  The configured acceptance cohort is:

  /data1/ken/pcori_dev/causal-dragonnet-text/synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet
  SHA-256: 6566aef4350ac1f78589d87d6a383bb49dce4a63ae5c9661921e925e3e854fe0
  columns: patient_id, clinical_text, treatment_indicator, outcome_indicator
  oracle: patient_id, true_ite_prob

  The oracle is a separate input and must remain unopened until all 1,000 predictions and their manifest have been frozen, reopened, and authenticated. Correlation magnitude is not a completion gate.

  Stage 1 is an oracle-free evidence factory. Every logical context must contain all ten nonempty families:

  1. Word treatment/outcome models.
  2. Word residual-effect model.
  3. Hierarchical transformer.
  4. Matched-patient uplift.
  5. Whole-cohort embeddings.
  6. Cluster-local embeddings.
  7. Lexical semantic-retrieval contrasts.
  8. TF-IDF topics.
  9. Residual TF-IDF n-grams.
  10. Learned neural queries.

  The configured plan has 40 logical contexts but only 35 unique physical fits: five full-outer, 25 exact-inner, five cumulative epoch-0, and five epoch-1 aliases of inner_005.

  Stage 2 must independently interpret every family and every lossless evidence page, then propose at most 20 explicit note-readable variables, extract each variable from every complete note, use only outer-
  training partitions for two bounded review rounds, freeze definitions, and fit only the configured strict econml.dml.CausalForestDML. Each outer fold fits on 800 patients and predicts its 200 held-out
  patients once.

  No structured or nonforest fallback is permitted.

  ## 2. History and current state

  Preserve all historical trees. Do not reset the dirty working tree or delete incomplete runs.

  - V1–V3 are historical and must not be resumed.
  - V4 has a legacy preflight that cannot prove current state/seed identities.
  - V5 has the only reusable terminal preparation and embedding cache. Its preflight is incomplete and categorically non-reusable.
  - R4–R8 established the shared-cache preflight design: one V5 cache, row-restricted views, no per-scope embedding arrays or chunk-text copies.
  - R9–R11 exposed and repaired serial/GIL-bound BoW execution, persistent-worker startup, and HTR model-tree identity issues.
  - R12 was cleanly stopped rather than spend days in serial HTR. Replica B never ran. Its loose BoW/HTR owner is not reusable.
  - R13 restored process-parallel primary HTR folds: five nuisance folds, a strict residual barrier, then five effect folds, with deterministic canonical merging and complete nontruncated text plans.
  - R13 subsequently revealed that matched-pair uplift, TF-IDF nuisance folds, and neural-query folds remain serial. Embedding contrasts are not fold-based.

  Current R13 identifiers:

  snapshot:
    /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/production_source_snapshot_20260725_portable_acceptance_r13
  snapshot content root:
    50c2420f7ff4beb83791217e315298491e51bec055058d83b41048cfa559f5d9
  request:
    78dc44196bdae983fa09bf7d30b60f379bd5f93e162a18e19c3b8a9c53017ccf
  work root:
    /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/production_all_evidence_one_conf_one_mod_1000_v13_benchmark_staging_shared_cache_parallel_htr_r13
  old-host scratch:
    /tmp/causal_dragonnet_nsclc_v13_benchmark_staging_shared_cache_parallel_htr_r13
  log:
    /tmp/causal_dragonnet_nsclc_v13_stage1.recompute.log

  R13’s preparation, embedding cache, and clustered preflight are terminal. The preflight contains 35 physical fits, 40 logical bindings, 35 safe state manifests, 430 files, canonical-no-refit state, and zero
  per-scope embedding/chunk-text cache copies.

  Its portable preflight checkpoint is:

  artifact ID:
    aa6135fe24024e9ff7f6c6c8a3e205e7bcfa1a56aa5722bd1454b8d86edbb57f
  locator:
    /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/production_all_evidence_one_conf_one_mod_1000_v13_benchmark_staging_shared_cache_parallel_htr_r13/portable_checkpoints/stage1_preflight/
    artifact_locator.json

  Only the first owner’s BoW is terminal in loose scratch. Its HTR is incomplete; matched-pair, embeddings, TF-IDF, neural query, and the other 34 owners have not started. No partial owner artifact is
  adoptable.

  At takeover, stop R13 as requested:

  - Dynamically re-resolve its parent and process groups; do not trust stale PIDs.
  - Confirm command lines bind the R13 snapshot, request, and work root.
  - Send SIGTERM only to verified R13-owned groups.
  - Wait up to 120 seconds.
  - If verified processes remain, report them rather than using an automatic SIGKILL.
  - Do not delete or modify its work root, scratch, log, snapshot, terminal preflight, terminal BoW, or incomplete HTR.
  - If R13 is running only on the old host, record that it cannot be stopped locally and have the old-host operator perform the same verified termination.

  The tracked working tree is intentionally dirty relative to commit e8d4fe1; these changes contain the accepted HTR parallelism, disabled Replica-B policy, configuration propagation, tests, and updated master
  record. Preserve them as the implementation baseline. The complete-page citation repair is already present in R13 and the working tree.

  ## 3. Implement the remaining parallelism narrowly

  ### Public operational configuration

  Keep scientific model settings separate from execution settings.

  - Do not add parallelism fields to matched-pair, TF-IDF, or neural-query scientific configurations.
  - Bump the typed Stage 1 execution profile schema and the enclosing deployment-profile schema.
  - Require all new operational fields explicitly; provide no production defaults.
  - Keep devices, worker counts, slots, backends, timing, PIDs, and completion order out of scientific identities and scientific artifact bytes.
  - Preserve all current seeds, folds, prompts, schemas, model settings, and row orders.

  Add a required deployment-only RoleNeutralNeuralQueryOperationalControls containing:

  inner_fold_parallelism
  fold_parallel_backend
  fold_slots_per_device
  bank_parallelism
  worker_cpu_threads
  schema_version

  For production CUDA concurrency, require spawned processes when parallelism exceeds one. The benchmark profile may explicitly select five inner-fold workers and three bank workers, but these values must live
  in deployment configuration, not library constants.

  Expose tfidf_parallel_backend through the typed Stage 1 execution profile. Remove the current compiler literal "processes". Numeric TF-IDF concurrency remains derived from configured cpu_budget, then bounded
  by owner CPU budget and actual fold count.

  Use the existing HTR fold resource plan for matched-pair HTR leasing. Do not create a separate matched-pair concurrency framework.

  ### Matched-patient uplift

  Refactor the serial effect-fold loop in role_neutral_matched_pair_group_execution.py.

  - Build the canonical KFold task list in the parent.
  - Create spawn-pickleable fold task/result types.
  - Schedule all effect folds through the existing HTR device/slot resource plan.
  - Allow multiple configured slots on one GPU and use all selected GPUs.
  - Keep current split, BoW, and HTR seed derivations exactly unchanged.
  - Set deterministic Torch policy before each worker initializes CUDA or a model.
  - Set tokenizer/native Torch/BLAS pools to the configured single worker thread.
  - Each worker owns its matching tables, BoW views, HTR model, validation predictions, replay checks, and a private array store.
  - Return isolated CPU model state, arrays, pair tables, predictions, evidence records, and execution telemetry. Never share _ArrayStore or a live CUDA model between workers.
  - Parent-sort strictly by fold, reject missing/duplicate/substituted folds, merge OOF arrays and proof records serially, and persist in canonical fold order.
  - Authenticate every stored byte exactly. Compare BoW replay exactly and HTR numerical replay with the configured neural tolerances.
  - Open registered held-out text only after all fold fits are sealed. Reconstruct models from authenticated captured state for exact transform; do not require returned live CUDA objects.
  - Preserve complete note coverage and the existing fail-closed nontruncation checks.

  Resource placement, timing, peak memory, PID, and completion order belong only in the execution attestation.

  ### TF-IDF topics and residual n-grams

  The expensive parallel target is not tfidf_nested_calibration_folds: that setting constructs candidate splits and deterministically selects one calibration partition.

  Parallelize the top-level folds inside fit_joint_cross_fitted_nuisance_stacks.

  - Propagate existing tfidf_workers, configured backend, and owner CPU budget through the role-neutral factory, physical-group executor, nested-calibration wrapper, context fitter, and joint nuisance fitter.
  - Factor one top-level fold into a standalone worker returning exact positions, treatment/outcome base predictions, stacked predictions, and fit-row provenance.
  - Keep each fold’s subfold loop serial to prevent nested process/BLAS oversubscription.
  - Use configured Joblib threads or spawn-safe Loky processes; production uses the configured process backend.
  - Merge top-level results only in canonical fold order.
  - Perform full-data base fits and final stack fits after the fold barrier, in their current deterministic order.
  - When TF-IDF nuisance fitting runs inside an already-parallel neural-query fold, force its internal fold concurrency to one so there is only one global parallel layer.
  - Preserve all complete input strings, vocabulary settings, score tests, NMF settings, topics, residual terms, and seeds. Add no note, character, token, vocabulary, feature, or topic cap.

  ### Learned neural queries and query moments

  R13 gives each owner only one query GPU and groups folds into one serial queue per device. Replace that with bounded task-level process execution.

  - Configure the new deployment with one_context_spanning_all_selected_devices.
  - Build one complete, authorized, row-scoped shared embedding view per owner. Workers reopen mmap-backed cache arrays by authenticated reference and row ID; do not pickle/copy large embedding arrays or chunk
    texts through process IPC.

  - Keep treatment/outcome arrays owner-local and unavailable outside authorized fit rows.
  - Submit the five query inner folds independently across configured GPU slots, including multiple slots on one GPU.
  - One inner-fold task performs its nuisance fits and then its treatment, outcome, and effect query banks in the existing bank order. This minimizes code movement while restoring fold overlap.
  - After all inner folds complete, enforce a barrier and submit the three full-context consensus/final-refit bank tasks concurrently.
  - Use the same bounded bank executor for independent safe-evidence and held-out-moment bank work.
  - Merge folds in numeric order and banks in treatment, outcome, effect order regardless of completion order.
  - Remove device names from subfold_audit and all scientific payloads. Put placement and slot telemetry in an execution attestation.
  - Preserve these seed formulas exactly:

  base = scope_seed + 100_000 * outer_fold
  full-context nuisance = base + 10_000
  inner split = base
  inner fold = base + fold
  inner bank = fold_seed + 100 * bank_index
  consensus = base + 1000 + bank_index
  final refit = base + 2000 + bank_index
  evidence = scope_seed + 3000 + bank_index

  Require exact equality for rows, splits, seeds, query identities, schemas, discrete evidence, finite masks, shapes, and dtypes. Compare learned query vectors, activations, moments, and other recomputed neural
  floats only with the declared per-family rtol/atol. Stored bytes remain exactly hashed and within-artifact replay remains mandatory.

  ### Embedding contrasts

  Do not invent fold parallelism here.

  Whole-cohort, canonical cluster-local, and lexical semantic-retrieval evidence are already computed vectorially from the one shared V5 cache. Preserve that path, verify that it performs no Qwen re-encoding
  and creates no per-scope embedding/chunk-text copies, and leave it unchanged unless telemetry later proves it dominates runtime.

  ### Scheduling boundary

  Do not build a component-DAG scheduler or partial-owner migration framework before relaunch.

  The existing owner executor may continue reserving the union of selected GPUs for one owner. Primary HTR, matched-pair HTR, and neural queries will now all use that selected device set. CPU-only owner/GPU
  pipelining is future performance work, not an acceptance blocker.

  All fold executors must enforce:

  active fold workers <= configured fold parallelism
  active slots per device <= configured slots per device
  sum of active worker CPU threads <= owner/global CPU budget
  waiting tasks hold no lease
  external GPU occupants are never killed

  ## 4. Focused verification and host selection

  Run checks with /data1/ken/envs/gptoss3/bin/python. GPU inspection and CUDA tests must run outside the sandbox; a failed sandbox GPU probe is not evidence that GPUs are unavailable.

  Add only focused tests:

  - Matched-pair serial/parallel comparison:
      - folds overlap;
      - two tasks can share one simulated GPU;
      - multiple GPUs are used;
      - reversed completion still produces canonical records;
      - discrete/BoW state is exact;
      - HTR state and predictions satisfy declared tolerances;
      - complete long-note coverage and text_truncation_applied=false remain unchanged.

  - TF-IDF serial/process comparison:
      - at least two top-level nuisance folds overlap;
      - canonical scientific artifacts and predictions are exact;
      - subfolds do not spawn nested pools;
      - complete string coverage remains unchanged.

  - Neural-query serial/parallel comparison:
      - inner folds overlap;
      - multiple slots can share a simulated GPU;
      - multiple GPUs are used;
      - final refits start only after the inner-fold barrier;
      - canonical seeds/order/discrete evidence are exact;
      - neural arrays satisfy configured tolerances;
      - complete row/chunk coverage is identical;
      - peer-row cache access fails.

  - One typed propagation/budget test:
      - new profile fields are required and excluded from scientific identity;
      - TF-IDF backend reaches the producer;
      - oversubscribed CPU/GPU plans fail before work starts;
      - operational device metadata does not enter scientific artifacts.

  Run only those nodes, py_compile on touched modules, and git diff --check. Do not run a new broad suite or benchmark matrix before relaunch.

  Perform one bounded real-host calibration per neural producer:

  1. Inspect the higher-powered host using unsandboxed nvidia-smi and CPU-affinity information.
  2. Measure one production-shaped fold with the configured text/chunk plan and batch sizes.
  3. Derive a candidate slots-per-device value from the configured gpu_max_allocation_fraction=0.85 and gpu_minimum_headroom_bytes=6 GiB.
  4. Run that candidate twice; decrement only if it violates memory safety or OOMs.
  5. Require deterministic discrete outputs and tolerance-valid neural outputs.
  6. Record exact selected devices, slots, concurrency, peak memory, and throughput in deployment configuration/attestations.
  7. Do not claim multi-GPU acceleration unless measured throughput is at least the configured 1.5× single-device baseline.

  Keep current scientific batch sizes unless a changed batch passes the complete scientific equality/tolerance gate. Never change a batch merely because the new host has more memory.

  ## 5. Freeze, adopt, and run Stage 1

  Create a new high-powered deployment profile based on the acceptance profile:

  scientific spec:
    /data1/ken/pcori_dev/causal-dragonnet-text/example_configs/portable_all_evidence_scientific_nsclc.json
  deployment base:
    /data1/ken/pcori_dev/causal-dragonnet-text/example_configs/portable_all_evidence_deployment_nsclc.acceptance.json
  storage_backend:
    local_posix
  scratch:
    a fresh /tmp/...r14 directory on the new host
  durable work root:
    a fresh /data1/...r14 directory

  The profile must contain the calibrated CPU/device/slot controls and the final Stage 2 endpoint/model before request creation. Do not edit it between the Stage 1 pause and Stage 2 resume.

  Verify the configured model trees:

  Qwen embedding:
    /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/local_models/qwen3_embedding_8b_materialized
  HTR expected tree hash:
    fd242da536eef74b2c38f426c010353f35bd29c89323fa14787bbfbd40f8c3ba
  Stage-2 tokenizer:
    /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/local_models/gemma4_26b_a4b_it_fp8_dynamic_tokenizer_materialized
  tokenizer tree hash:
    4b5634239b53024f4e6359d8e1d48be064b5f71bcb627bfe0f

  Use the full tokenizer hash recorded in the existing immutable request if the abbreviated line above differs; the authoritative value is:

  4b5634239b53024f4e6359d8e1d48be064d2a6ecfe227e0efa764137b3f6cb23

  If the /home/.../bert-tiny locator is unavailable on the new host, materialize the exact five-file tree beneath /data1/.../artifacts/local_models/, verify the expected tree hash, and update only the
  deployment locator. Paths are operational; model content is scientific.

  Freeze and independently reopen:

  /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/production_source_snapshot_20260726_portable_acceptance_r14

  Create and validate the snapshot using gptoss3, then launch only from the frozen snapshot through PYTHONPATH=<snapshot> and its frozen script.

  Reuse V5 preparation/cache through R13’s trusted attestations when filesystem/stat continuity validates:

  prepared:
    .../production_all_evidence_one_conf_one_mod_1000_v13_benchmark_staging_shared_cache_parallel_htr_r13/checkpoint_adoptions/a39287a94bc3f8925c39d1edf51d0d3d9482a689e28c9d060c95d508c7934fed.adoption.json
  cache:
    .../production_all_evidence_one_conf_one_mod_1000_v13_benchmark_staging_shared_cache_parallel_htr_r13/checkpoint_adoptions/be8254746c50d47d3d25cb8fb65ee59fe7844935cfe5151995be15ad3df8c5fa.adoption.json

  Because /data1 is directly available on the target, try these exact stat-continuity attestations first. If they fail because filesystem identities changed, use ordinary full-byte adoption of their registered
  portable artifacts. Do not bypass the guard, edit locators manually, or rebuild embeddings unless ordinary adoption also rejects compatibility.

  Attempt ordinary adoption of R13’s terminal preflight exactly once. Expect that producer-code or Python/runtime compatibility may reject it.

  - If accepted, use it.
  - If rejected, add no exception, migration, schema downgrade, or retry.
  - Preserve the diagnostic attempt.
  - Select a second fresh absent work/scratch root and recompute all 35 preflight groups from the shared V5 cache.
  - Require 35 physical states, 40 logical bindings, canonical-no-refit true, and zero per-scope embedding arrays/chunk-text copies.

  Launch detached and outside the sandbox with:

  --scientific-spec <frozen scientific spec>
  --deployment-profile <frozen high-powered deployment profile>
  --source-snapshot-root <R14 snapshot>
  --trust-prior-adoption-attestation <prepared attestation>
  --trust-prior-adoption-attestation <cache attestation>
  [--adopt-checkpoint <R13 preflight locator> only in the one adoption attempt]
  --stop-after handoff_validation
  --validation-depth fresh_terminal_audit
  --log-level INFO

  Use setsid or an equivalent detached process, redirect stdin, retain the log path, and record the parent PID/PGID/SID. Do not launch Replica B.

  During the first complete physical owner, verify from execution attestations that matched, TF-IDF, and query folds actually overlap and use the configured leases. If configured parallelism silently executes
  serially, memory limits fail, or text coverage changes, stop cleanly and fix; do not salvage the partial owner.

  Then let Stage 1 complete all 35 physical owners, 40 logical bindings, six producer groups, ten evidence families, the reference-only Stage 1 handoff, and fresh path-only handoff_validation with the Stage 2
  endpoint offline.

  ## 6. Resume identical request for Stage 2 and oracle evaluation

  Only after Stage 1 handoff validation is terminal:

  1. Start or ask the operator to start the exact configured OpenAI-compatible/vLLM endpoint.
  2. Verify /v1/models returns exactly:

  RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic

  3. Verify the endpoint/tokenizer pair and context window using the configured local tokenizer.
  4. Resume the identical immutable request using the same snapshot, profiles, work root, and initial adoption argument array.
  5. Remove --stop-after and add --resume; change nothing else.

  The remaining phases must execute in order:

  - stage2_canary: exactly one real architecture-level initial-interpretation request; no extraction, forest, prediction, or oracle access.
  - stage2_inference: separately interpret all ten families and every lossless page, integrate only afterward, propose/review/extract/freeze variables, fit five strict forests, and seal one 1,000-row outer-
    held-out prediction.

  - oracle_evaluation: open the separate oracle only after prediction bytes, prediction manifest, row map, and Stage 1 mapping have been reopened and hashed.
  - terminal_validation: fresh path-only reopen of the entire graph and event order.

  Production LLM behavior remains:

  - zero transport retries;
  - at most one fixed-schema repair;
  - exact configured model;
  - finish_reason=stop;
  - complete page/request/reconciliation ledgers;
  - exact absolute-offset citations;
  - abort on omitted pages, invalid citations, wrong models, non-stop completions, second-invalid responses, or transport failure;
  - never convert failures to all-missing values.

  Paging and chunk geometry remain configuration, not code constants. The NSCLC profile may explicitly contain 13,488/256/14,000, reconciliation fan-in 16, and its current chunk geometry, but library code must
  not assume these values or any note/page count. Capacity limits must prove nonbinding or abort before fitting; they must never truncate.

  ## 7. Completion and reporting

  Completion requires:

  - exactly 35 physical Stage 1 fits and 40 logical contexts;
  - all ten nonempty evidence families in every required context;
  - independently present whole-cohort and cluster-local native embeddings;
  - no outer-held-out treatment/outcome access during discovery, review, or fitting;
  - complete nontruncated note/chunk/page coverage;
  - one strict CausalForestDML prediction per patient;
  - exactly 1,000 unique prediction rows in original row-map order;
  - oracle access only after frozen-prediction authentication;
  - execution_completed=true;
  - successful fresh terminal validation;
  - global_release_certified=false, because trusted no-rehash V5 adoption is a research-only trust boundary.

  Report:

  - Stage 1 wall time overall and by component/owner.
  - CPU/GPU utilization, peak allocated/reserved memory, configured versus actual concurrency, and fold overlap.
  - Bytes read, written, copied, hashed, synchronized, compressed, and decompressed.
  - Coordination/proof overhead ratio and ordinary read amplification.
  - Overall and per-fold Pearson, Spearman, MAE, RMSE, signed error, truth variance, and estimate variance.
  - Any inability to claim 1.5× multi-device acceleration.
  - The exact snapshot/request/artifact IDs and prediction hash.

  Finally update todo_list_7-22-26.md with concise terminal statuses and artifact links. Preserve todo_list_7-22-26.history.md byte-for-byte at SHA-256
  d4bcb596a4aef42a03eec3a2ce63e7d01a03e89ec69835317a860584dd508c59. Do not add historical logs, PIDs, or long test narratives back into the master record.

