# Hierarchical All-Evidence Pipeline Operations Runbook

Status: low-level two-phase operator and exact-replay guide for the
architecture-at-a-time benchmark path.

> **Deployment boundary:** this document describes the retained low-level
> prepare/approve/execute benchmark interface. It is not the end-user workflow
> for a new cohort. The arbitrary-cohort path uses
> `scripts/build_all_evidence_stage1_bundle.py` followed by
> `scripts/run_production_stage1_hierarchy_one_shot.py`, as documented in
> [`production_stage1_bundle_runbook.md`](production_stage1_bundle_runbook.md).
> Those wrappers create and authenticate integrity digests internally and never
> ask an end user to inspect, type, or approve one. Do not expose the approval
> steps below as a production UI.

This runbook is the durable operational companion to
[`all_evidence_discovery_interfaces.md`](all_evidence_discovery_interfaces.md).
Conversation history is not an execution dependency. For this low-level
interface, immutable manifests, the approved batch digest, and authenticated
caches are the sources of truth.

## Scope

The supported entry point is:

```text
scripts/run_all_evidence_fusion_benchmark.py
```

It orchestrates the pipeline from an **authenticated Stage 1 artifact bundle**
through:

```text
Stage 1 handoffs
  -> architecture-local evidence catalogs and chunks
  -> one dossier per active architecture
  -> bounded cross-architecture lookback and integration
  -> extraction definitions and patient-level extraction
  -> hierarchical adaptive review on accumulated-spent rows
  -> frozen feature registry and direct numerical bank
  -> honest causal forest
  -> frozen outer-held-out predictions
  -> optional post-hoc synthetic oracle evaluation
```

The entry point does **not** start from a raw Parquet file alone. It consumes
previously produced Stage 1 handoffs, authoritative outer splits, a Stage 1
configuration snapshot, and a frozen embedding cache. See
[Upstream Stage 1 bundle](#upstream-stage-1-bundle) before attempting a new cohort.

## Non-negotiable invariants

- Production discovery mode is `hierarchical`. The legacy staged mode is only for
  tests and historical ablations.
- Every active architecture is interpreted separately before integration. Never
  replace this with one prompt containing all raw evidence.
- The preparation scratch, preparation directory, and final execution output are
  separate paths. The preparation and execution paths must not contain or be
  contained by one another.
- Preparation makes zero remote model calls. It writes an inspectable packet and
  an approval SHA-256.
- Live execution requires explicit human approval of that exact SHA-256. A changed
  input, path-bound provider, code identity, endpoint, model, or configuration
  requires a new preparation and approval.
- Each execution uses a nonexistent or empty output directory. Never point a new
  invocation at a prior or partially populated execution directory.
- The proposal/extraction endpoint must be remote from the worker running this
  command. Loopback, wildcard, and the current host are rejected.
- Oracle columns cannot enter discovery, extraction, review, model fitting, or
  prediction. `--evaluate-oracle-posthoc` reads the synthetic oracle only after
  the prediction file has been frozen and hash-checked.
- Successful hierarchical JSON jobs are cached immutably under the approved
  preparation directory. A replay runs the same semantic validator before a cache
  hit is accepted.

## Upstream Stage 1 bundle

The final CLI requires the following inputs.

| Input | Required contract |
|---|---|
| Dataset | Parquet containing the configured text, binary treatment, and observed outcome columns. Synthetic oracle columns may coexist but are projected out until post-hoc evaluation. |
| Legacy all-source handoff | `multi_model_agentic_discovery_handoff_v1` JSONL with complete full-outer evidence for every authoritative fold. |
| Resealed TF-IDF handoff | Registry-sealed TF-IDF topic, contrast, and orphan evidence whose row registry agrees exactly with the other handoff and primary splits. |
| Primary predictions | Parquet containing canonical `_oci_row_id` and `outer_fold` or `cv_fold`; outer held-out rows must match the resealed TF-IDF registry exactly. |
| Historical Stage 1 config | JSON or YAML snapshot that enables the required BoW, HTR, matched-pair, embedding, and TF-IDF runtime. The HTR sentence encoder must remain unfrozen. |
| Frozen embedding cache | One row-bound cache containing `metadata.json`, `chunk_embeddings.npy`, `offsets.npy`, and `chunk_texts.jsonl`. |
| Prompt controls | Byte-exact historical flat prompt and prior hierarchy prompt, each registered as `PATH::SHA256`; used only to construct the offline comparison packet during preparation. |
| Optional authenticated overlays | Extraction-cache indexes, accumulated-spent evidence cache entries, and context-fit cache indexes registered with their declared hashes. |

For the synthetic benchmark, the existing Stage 1 paths were produced by the
integrated multi-model Stage 1 code and the dedicated TF-IDF producer:

```text
oracle_experiment_scripts/run_oracle_multi_model_forest.py --stage stage1
scripts/run_tfidf_topic_stage1_from_primary_splits.py
```

The TF-IDF producer writes `split_registry.json`,
`handoff/discovery_contexts.jsonl`, `primary_predictions.parquet`,
`stage1_invocation_audit.json`, and `stage1_result.json`. The migration-only
`scripts/reseal_tfidf_topic_handoff.py` may reseal an older compatible handoff; it
is not a substitute for fitting Stage 1 on a new cohort.

The stable general-purpose arbitrary-cohort builder is now
`scripts/build_all_evidence_stage1_bundle.py`. It fits all ten native
architectures, writes the exact-inner and cumulative-spent root graphs, and can
either validate an existing production embedding cache or atomically create one
from a local symlink-free model tree. Its output is consumed only through
`scripts/run_production_stage1_hierarchy_one_shot.py`, which authenticates the
bundle and constructs the same-process provider-bound runner without exposing a
digest or approval argument. The older `run_oracle_multi_model_forest.py` helper
remains synthetic-experiment infrastructure, not the production raw-data
ingestion interface. A genuine cohort E2E of the new path is still required
before final certification.

For a rerun of an existing benchmark, reuse the exact authenticated upstream files
named in its preparation manifest. Do not regenerate them merely to obtain new
paths: path and byte identity can be part of the approval.

## Preflight

Run from the repository root. Use the project lock file or the exact previously
recorded environment. `uv run --frozen` is the recommended generic invocation;
an already approved batch must use the interpreter and package identities bound
into its preparation manifest.

```bash
uv run --frozen python scripts/run_all_evidence_fusion_benchmark.py --help
```

Before an expensive run, check the relevant devices and endpoint independently.
Do not start a local language-model server through this CLI.

The examples below use deliberately pipeline-specific variable names. Replace
every placeholder with an absolute path. Keep the three output targets distinct,
and do not create the scratch or execution directories in advance.

```bash
export AE_REPO=/absolute/path/to/causal-dragonnet-text
export AE_BENCHMARK=descriptive_benchmark_name
export AE_DATASET=/absolute/path/to/dataset.parquet
export AE_LEGACY_HANDOFF=/absolute/path/to/legacy/discovery_contexts.jsonl
export AE_TFIDF_HANDOFF=/absolute/path/to/tfidf/handoff/discovery_contexts.jsonl
export AE_PRIMARY_SPLITS=/absolute/path/to/primary_predictions.parquet
export AE_STAGE1_CONFIG=/absolute/path/to/stage1_config.json
export AE_EMBEDDING_CACHE=/absolute/path/to/frozen_embedding_cache
export AE_HISTORICAL_PROMPT=/absolute/path/to/historical_flat_prompt.txt
export AE_HISTORICAL_PROMPT_SHA256=replace_with_64_lowercase_hex_characters
export AE_OLD_HIERARCHY_PROMPT=/absolute/path/to/old_hierarchy_prompt.txt
export AE_OLD_HIERARCHY_PROMPT_SHA256=replace_with_64_lowercase_hex_characters
export AE_ENDPOINT=http://remote-model-host:8010/v1
export AE_MODEL=publisher/model-name
export AE_API_KEY=EMPTY
export AE_STAGE1_DEVICE=cuda:0
export AE_NEURAL_QUERY_DEVICE=cuda:0
export AE_CONTROL_DIR=/absolute/path/to/run_control_record
export AE_PREP_SCRATCH=/absolute/path/to/fresh_prepare_scratch
export AE_PREPARATION=/absolute/path/to/fresh_immutable_preparation
export AE_EXECUTION=/absolute/path/to/fresh_final_execution
```

If a real API secret is required, do not write it into the control record or an
archived command transcript. The current CLI passes `--api-key` as a process
argument, so use it only under the host's approved secret-handling policy.

Create only the separate control directory:

```bash
mkdir -p "$AE_CONTROL_DIR"
cd "$AE_REPO"
```

### Current reference profile

The following array makes the important scientific and transport settings
explicit. It matches the current hierarchical benchmark profile as of
2026-07-19. A future intentional configuration change is allowed only before
preparation and yields a new approval digest.

```bash
AE_COMMON_ARGS=(
  --benchmark-name "$AE_BENCHMARK"
  --discovery-mode hierarchical
  --dataset "$AE_DATASET"
  --legacy-handoff "$AE_LEGACY_HANDOFF"
  --resealed-tfidf-handoff "$AE_TFIDF_HANDOFF"
  --primary-splits "$AE_PRIMARY_SPLITS"
  --endpoint "$AE_ENDPOINT"
  --model "$AE_MODEL"
  --api-key "$AE_API_KEY"
  --text-column clinical_text
  --treatment-column treatment_indicator
  --outcome-column outcome_indicator
  --outcome-type binary
  --expected-outer-folds 5
  --interaction-inner-folds 3
  --max-candidates 20
  --hierarchical-max-cross-architecture-lookback-ids 24
  --hierarchical-max-cross-architecture-lookback-bytes 96000
  --hierarchical-max-rejection-lookback-ids-per-candidate 24
  --hierarchical-max-rejection-lookback-bytes-per-candidate 48000
  --hierarchical-review-max-evidence-ids 512
  --hierarchical-review-max-evidence-bytes 2000000
  --post-extraction-review-rounds 2
  --post-extraction-review-max-operations 4
  --post-extraction-review-max-quality-retries 8
  --post-extraction-review-min-partition-rows 8
  --review-stage1-config "$AE_STAGE1_CONFIG"
  --review-embedding-cache-dir "$AE_EMBEDDING_CACHE"
  --review-stage1-device "$AE_STAGE1_DEVICE"
  --review-stage1-bow-fold-parallelism 1
  --review-stage1-bow-parallel-backend threads
  --review-neural-query-nuisance-folds 3
  --review-neural-query-device "$AE_NEURAL_QUERY_DEVICE"
  --final-upstream-meta-inner-folds 3
  --final-upstream-head-regularization 1.0
  --seed 42
  --proposal-max-tokens 25000
  --extraction-max-tokens 25000
  --proposal-schema-repair-attempts 2
  --request-max-retries 3
  --request-timeout 1800
  --extraction-batch-size 128
  --max-variables-per-extraction-request 1
  --extraction-grouping-strategy packed
  --extraction-context-strategy contract_lexical_rag
  --extraction-max-text-length 14000
  --extraction-prompt-version explicit_features_v5
  --require-neural-query-moments
  --require-orphan-ngrams
  --modifier-interactions-only
)
```

The retained `lookback` arguments are compatibility/profile fields, not semantic
cutoffs. Current base and adaptive discovery schedule one page for every exact
authenticated support item and recursively fold all pages; lowering a legacy
lookback field must not sample or discard evidence.

For exact replay, reconstruct this array from the original invocation record and
the immutable preparation manifest. Do not assume that the profile above is a
substitute for an older approved configuration.

## Phase 1: side-effect-free dry run

Dry run validates files, fold registries, endpoint identity, configuration,
authenticated overlays, and the fresh-output rule. It constructs no client and
makes no remote call.

```bash
uv run --frozen python scripts/run_all_evidence_fusion_benchmark.py \
  "${AE_COMMON_ARGS[@]}" \
  --output-dir "$AE_PREP_SCRATCH" \
  --review-neural-query-cache-dir \
    "$AE_PREP_SCRATCH/post_extraction_review_neural_query_cache" \
  --hierarchical-preparation-dir "$AE_PREPARATION" \
  --hierarchical-job-cache-root "$AE_PREPARATION/hierarchical_job_cache" \
  --hierarchical-offline-review-packet-dir "$AE_PREPARATION/offline_review_packet" \
  --historical-discovery-prompt \
    "${AE_HISTORICAL_PROMPT}::${AE_HISTORICAL_PROMPT_SHA256}" \
  --old-hierarchy-prompt \
    "${AE_OLD_HIERARCHY_PROMPT}::${AE_OLD_HIERARCHY_PROMPT_SHA256}" \
  --dry-run \
  | tee "$AE_CONTROL_DIR/dry_run.json"
```

Require all of the following in the output:

- `status` is `validated_dry_run`;
- `discovery_mode` is `hierarchical`;
- `hierarchical_all_active_stage1_architectures_required` is true;
- selector reasoning is enabled with a 5,000-token budget;
- extraction reasoning is false;
- the sparse-query fallback is false;
- final causal forest is required;
- `clients_constructed`, `remote_calls_made`, and `oracle_columns_read` are false.

Dry run does not reserve the output paths. Recheck that the scratch, preparation,
offline-packet, and final execution targets have not been populated by another
process before proceeding.

## Phase 2: prepare the immutable batch

Preparation may perform expensive local Stage 1 work and create output-local
provider caches, but it must execute zero hierarchical JSON jobs and zero remote
model calls.

```bash
uv run --frozen python scripts/run_all_evidence_fusion_benchmark.py \
  "${AE_COMMON_ARGS[@]}" \
  --output-dir "$AE_PREP_SCRATCH" \
  --review-neural-query-cache-dir \
    "$AE_PREP_SCRATCH/post_extraction_review_neural_query_cache" \
  --hierarchical-preparation-dir "$AE_PREPARATION" \
  --hierarchical-job-cache-root "$AE_PREPARATION/hierarchical_job_cache" \
  --hierarchical-offline-review-packet-dir "$AE_PREPARATION/offline_review_packet" \
  --historical-discovery-prompt \
    "${AE_HISTORICAL_PROMPT}::${AE_HISTORICAL_PROMPT_SHA256}" \
  --old-hierarchy-prompt \
    "${AE_OLD_HIERARCHY_PROMPT}::${AE_OLD_HIERARCHY_PROMPT_SHA256}" \
  --prepare-hierarchical-discovery \
  | tee "$AE_CONTROL_DIR/prepare_result.json"
```

A valid result has:

- `status: hierarchical_discovery_prepared_awaiting_approval`;
- one `approval_sha256`;
- paths and hashes for the batch packet, input manifest, first-gate intent index,
  and offline review packet;
- one fold preparation for every authoritative outer fold;
- `hierarchical_json_jobs_executed: 0`;
- `remote_clients_constructed: false`;
- `remote_calls_made: false`;
- `oracle_columns_read: false`;
- `predictions_written: false`;
- `final_run_manifest_written: false`.

The result also contains three similarly named replay fields. For execution of
the just-approved batch, preserve **exactly**
`authoritative_execution_replay_arguments`. Do not substitute
`next_authenticated_provider_replay_arguments` or paths copied into a different
scratch directory.

## Phase 3: inspect and approve

Stop before any live invocation.

1. Authenticate the packet and manifest hashes reported in
   `prepare_result.json` with `sha256sum`.
2. Read the exact Markdown packet under
   `$AE_PREPARATION/offline_review_packet`.
3. Confirm that all folds contain nonzero evidence from all ten active
   architectures, architecture chunks are family-pure, no global top-k or sparse
   fallback is active, and raw lookback bounds match the prepared configuration.
4. Confirm the exact remote endpoint and model, selector reasoning budget,
   extraction settings, causal-forest requirement, and deferred first-gate
   materialization.
5. Record explicit human approval of the reported **batch `approval_sha256`**.
   The offline packet's own SHA-256 is not the execution approval digest.

Set the approved value only after that review:

```bash
export AE_APPROVAL_SHA256=replace_with_the_approved_batch_sha256
```

Copy the JSON array named `authoritative_execution_replay_arguments` from the
preparation result into a Bash array. Each element is already a complete CLI
argument:

```bash
AE_REPLAY_ARGS=(
  # Paste each exact authoritative_execution_replay_arguments element here.
)
```

The actual array may be empty or contain several entries. Preserve its order and
bytes.

## Phase 4: execute the approved batch

Use the exact prepared configuration and a fresh final output. Do not pass the two
historical prompt-control arguments during live execution; they were inputs to the
offline comparison packet, not permission to change the approved hierarchy.

```bash
uv run --frozen python scripts/run_all_evidence_fusion_benchmark.py \
  "${AE_COMMON_ARGS[@]}" \
  --output-dir "$AE_EXECUTION" \
  --review-neural-query-cache-dir \
    "$AE_EXECUTION/post_extraction_review_neural_query_cache" \
  --hierarchical-preparation-dir "$AE_PREPARATION" \
  --hierarchical-job-cache-root "$AE_PREPARATION/hierarchical_job_cache" \
  --hierarchical-offline-review-packet-dir "$AE_PREPARATION/offline_review_packet" \
  --hierarchical-approved-batch-sha256 "$AE_APPROVAL_SHA256" \
  "${AE_REPLAY_ARGS[@]}" \
  | tee "$AE_CONTROL_DIR/execution_result.json"
```

For a synthetic benchmark, add `--evaluate-oracle-posthoc` to this live
invocation if the separate post-hoc report is required. That flag does not expose
oracle values until after the frozen prediction SHA-256 has been verified. Do not
use it for real data.

Live execution must fail closed before its first remote call if the approval,
preparation, input bytes, code identity, model, endpoint, provider identity, cache
registration, or scientific configuration differs.

## Phase 5: verify completion

Require `status: completed` in `execution_result.json` and preserve the complete
execution directory. The principal artifacts are:

```text
immutable_input_manifest.json
outer_fold_001/immutable_fold_manifest.json
...
outer_fold_NNN/immutable_fold_manifest.json
outer_fold_001/frozen_predictions.parquet
...
outer_fold_NNN/frozen_predictions.parquet
frozen_predictions.parquet
immutable_run_manifest.json
posthoc_oracle_evaluation/        # synthetic and requested only
```

Verify that:

- the run manifest lists every fold manifest;
- every dataset row appears exactly once in the combined prediction file;
- the combined prediction SHA-256 matches both the terminal result and run
  manifest;
- prediction columns contain no oracle or ground-truth fields;
- all fold manifests are present and immutable;
- post-hoc outputs, when requested, are separate from the prediction manifest.

Then follow
[`hierarchical_all_evidence_reproducibility_runbook.md`](hierarchical_all_evidence_reproducibility_runbook.md)
to preserve the run bundle.

## Interruption and restart rules

| Point of interruption | Supported action |
|---|---|
| Before preparation starts | Correct inputs and use fresh scratch, preparation, and execution targets. |
| During preparation | Preserve the failed directories for diagnosis. Start a new preparation with fresh targets, optionally using only formally authenticated read-only cache overlays. Do not treat an arbitrary partial cache as an input. |
| After preparation, before approval | Do not rerun preparation. Inspect the immutable packet and either approve its exact digest or prepare a new batch. |
| During hierarchical remote jobs | Preserve the failed execution. Restart with a **new fresh execution directory**, the same approved preparation directory, the same job-cache root, the same digest, and identical arguments. Successfully validated JSON jobs are replayed from the immutable hierarchy cache. |
| During extraction or adaptive review | Preserve the failed output. Restart in a fresh output. Downstream work is reusable only through an authenticated extraction-cache index, spent-evidence cache registration, or context-fit cache index accepted by the CLI; unsealed partial output is not a cache input. |
| After frozen predictions | Treat the prediction and run manifests as immutable. Do not rerun in place. Use a new output for any scientifically distinct run. |

An approval or cache mismatch is not something to override. It means the requested
execution is a different run and needs a new preparation and approval.

## Running multiple benchmarks

Each benchmark must have its own:

- control-record directory;
- preparation scratch directory;
- immutable preparation directory;
- hierarchy job-cache root below that preparation directory;
- fresh live execution directory;
- fresh output-local neural-query cache;
- explicitly selected Stage 1 and neural-query devices.

Read-only source artifacts may be shared only when their authenticated registration
matches the current request exactly. Never share writable execution caches between
concurrent benchmarks.
