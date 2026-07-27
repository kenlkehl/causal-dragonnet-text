# Hierarchical All-Evidence Pipeline Operations Runbook

Status: low-level two-phase operator and exact-replay guide for the
architecture-at-a-time benchmark path.

> **Deployment boundary:** this document describes a retained low-level
> prepare/approve/execute interface for exact historical replay. It is not the
> end-user workflow for a new cohort. New work must enter through
> `scripts/run_production_all_evidence_workflow.py` with a typed
> `ScientificWorkflowSpec` and `DeploymentProfile`. The older bundle and
> one-shot wrappers remain compatibility tools. Do not expose the approval steps
> below as a production UI or derive a new scientific profile from the
> historical commands.

This runbook is the durable operational companion to
[`all_evidence_discovery_interfaces.md`](all_evidence_discovery_interfaces.md).
Conversation history is not an execution dependency. For this low-level
interface, immutable manifests, the approved batch digest, and authenticated
caches are the sources of truth.

## Scope

The historical low-level entry point described by the replay sections is:

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
`stage1_invocation_audit.json`, and `stage1_result.json`.

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

### Typed portable profile for every new run

There is intentionally no source- or runbook-level reference array of numeric
scientific settings. For a new cohort, configure every column, fold/review
policy, model-facing token budget, hierarchy wire budget, architecture profile,
causal-estimator setting, and text geometry in an immutable
`ScientificWorkflowSpec`. Configure paths, endpoint/model locators, devices,
CPU/concurrency, and scratch/storage policy in a separate
`DeploymentProfile`.

In particular, extraction must use `complete_paged_v1`. Its configured core,
left/right context, maximum page size, and reconciliation fan-in are
scientific-spec fields—not universal constants. Every note character must occur
in exactly one page core, context overlap is deduplicated by absolute offsets,
and every page is recursively reconciled. A page/chunk/token capacity may page
or abort; it must never select a prefix, suffix, or top-k substitute. The typed
Stage 2 protocol also requires zero transport retries and exactly one fixed
schema-repair attempt.

Create and freshly validate the immutable source snapshot only after the
source and test gates are quiescent:

```bash
export AE_REPOSITORY_ROOT=/absolute/path/to/causal-dragonnet-text
export AE_SOURCE_SNAPSHOT_ROOT=/absolute/path/to/absent_source_snapshot

PYTHONPATH="$AE_REPOSITORY_ROOT" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONNOUSERSITE=1 \
/home/klkehl/thisenv/bin/python -P -B -c \
  'import os; from pathlib import Path; from oci.inference.production_source_snapshot import create_production_source_snapshot as create; print(create(repository_root=Path(os.environ["AE_REPOSITORY_ROOT"]), target_dir=Path(os.environ["AE_SOURCE_SNAPSHOT_ROOT"])).as_dict())'

PYTHONPATH="$AE_SOURCE_SNAPSHOT_ROOT" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONNOUSERSITE=1 \
/home/klkehl/thisenv/bin/python -P -B -c \
  'import os; from pathlib import Path; from oci.inference.production_source_snapshot import validate_production_source_snapshot as validate; print(validate(Path(os.environ["AE_SOURCE_SNAPSHOT_ROOT"])).as_dict())'
```

The typed portable invocation shape is:

```bash
export AE_SCIENTIFIC_SPEC=/absolute/path/to/scientific_workflow.json
export AE_DEPLOYMENT_PROFILE=/absolute/path/to/deployment_profile.json
export AE_SOURCE_SNAPSHOT_ROOT=/absolute/path/to/source_snapshot
export AE_PREPARED_CHECKPOINT=/absolute/path/to/preparation/complete_manifest.json
export AE_CACHE_CHECKPOINT=/absolute/path/to/embedding_cache/complete_manifest.json
export AE_PREFLIGHT_AUDIT=/absolute/path/to/cluster_preflight_manifest.json

uv run --frozen python scripts/run_production_all_evidence_workflow.py \
  --scientific-spec "$AE_SCIENTIFIC_SPEC" \
  --deployment-profile "$AE_DEPLOYMENT_PROFILE" \
  --source-snapshot-root "$AE_SOURCE_SNAPSHOT_ROOT" \
  --adopt-checkpoint "$AE_PREPARED_CHECKPOINT" \
  --adopt-checkpoint "$AE_CACHE_CHECKPOINT" \
  --adopt-checkpoint "$AE_PREFLIGHT_AUDIT" \
  --validation-depth fresh_terminal_audit \
  --log-level INFO \
  --stop-after handoff_validation
```

Dataset, work and scratch roots, device policy, and worker budgets come from
the typed deployment profile. Do not mix `--deployment-profile` with their
direct compatibility flags. The `stop-after`, validation, and logging values
are operational and excluded from scientific identity. After the configured
Stage 2 endpoint is available, repeat the same scientific, deployment,
snapshot, and checkpoint arguments, omit `--stop-after`, and add `--resume`.
Omitting an adoption input changes the immutable request and fails closed. Any
scientific code or configuration change requires checkpoint adoption into a
fresh request, not resume.

This is the generic post-selection invocation shape. It is not the
first-acceptance launch sequence: that sequence must first create the measured
deployment profile and resolve its checkpoint-adoption controls as described
below.

### Measured Stage 1 deployment selection

The first portable acceptance run must select its Stage 1 execution profile
from the configured representative benchmark before productive modeling. Run
the workflow and every benchmark utility from the same freshly validated
immutable source snapshot. Importing benchmark modules from the mutable
checkout would bind different producer-code evidence and is not a valid
selection authority.

The benchmark source workflow is a separate fresh request. Adopt the validated
preparation/cache checkpoints and the legacy preflight audit, then pause it
exactly at `stage1_preflight`:

For the configured NSCLC acceptance deployment, use one closed set of fresh
paths throughout the sequence:

```bash
export AE_REPOSITORY_ROOT=/data1/ken/pcori_dev/causal-dragonnet-text
export AE_SCIENTIFIC_SPEC="$AE_REPOSITORY_ROOT/example_configs/portable_all_evidence_scientific_nsclc.json"
export AE_BENCHMARK_STAGING_PROFILE="$AE_REPOSITORY_ROOT/example_configs/portable_all_evidence_deployment_nsclc.benchmark-staging.json"
export AE_BASE_ACCEPTANCE_PROFILE="$AE_REPOSITORY_ROOT/example_configs/portable_all_evidence_deployment_nsclc.acceptance.json"
export AE_BENCHMARK_CONFIG="$AE_REPOSITORY_ROOT/example_configs/portable_role_neutral_performance_benchmark_nsclc.deployment.json"

export AE_SOURCE_SNAPSHOT_ROOT="$AE_REPOSITORY_ROOT/artifacts/production_source_snapshot_20260725_portable_acceptance"
export AE_BENCHMARK_STAGING_WORK_ROOT="$AE_REPOSITORY_ROOT/artifacts/production_all_evidence_one_conf_one_mod_1000_v6_benchmark_staging"
export AE_BENCHMARK_STAGING_SCRATCH=/tmp/causal_dragonnet_nsclc_v6_benchmark_staging
export AE_ACCEPTANCE_WORK_ROOT="$AE_REPOSITORY_ROOT/artifacts/production_all_evidence_one_conf_one_mod_1000_v6_portable_acceptance"
export AE_ACCEPTANCE_SCRATCH=/tmp/causal_dragonnet_nsclc_v6_acceptance

export AE_PREPARED_CHECKPOINT="$AE_REPOSITORY_ROOT/artifacts/production_all_evidence_one_conf_one_mod_1000_v5_parallel_stage1/phases/input_preparation/complete_manifest.json"
export AE_CACHE_CHECKPOINT="$AE_REPOSITORY_ROOT/artifacts/production_all_evidence_one_conf_one_mod_1000_v5_parallel_stage1/phases/embedding_cache/complete_manifest.json"
export AE_PREFLIGHT_AUDIT="$AE_REPOSITORY_ROOT/artifacts/production_all_evidence_one_conf_one_mod_1000_v4_parallel_stage1/phases/stage1_preflight/attempt_20260723T195805360899Z/cluster_preflight/cluster_preflight_manifest.json"

export AE_BENCHMARK_PREPARED_CONTEXT=/tmp/causal_dragonnet_nsclc_v6_benchmark_prepared_context
export AE_BENCHMARK_CONTROL_DIR=/tmp/causal_dragonnet_nsclc_v6_benchmark_control
export AE_BENCHMARK_WORKLOAD_DEPLOYMENT="$AE_BENCHMARK_CONTROL_DIR/workload_deployment.json"
export AE_BENCHMARK_SCRATCH=/tmp/causal_dragonnet_nsclc_v6_role_neutral_benchmark
export AE_BENCHMARK_PUBLICATION="$AE_REPOSITORY_ROOT/artifacts/production_role_neutral_benchmark_publication_20260725"
export AE_SELECTED_DEPLOYMENT="$AE_REPOSITORY_ROOT/artifacts/portable_all_evidence_deployment_nsclc.acceptance.selected.20260725.json"
```

Before creating the source snapshot, require every output target to be absent.
Create only the workload file's parent after this check:

```bash
for path in \
  "$AE_SOURCE_SNAPSHOT_ROOT" \
  "$AE_BENCHMARK_STAGING_WORK_ROOT" \
  "$AE_BENCHMARK_STAGING_SCRATCH" \
  "$AE_ACCEPTANCE_WORK_ROOT" \
  "$AE_ACCEPTANCE_SCRATCH" \
  "$AE_BENCHMARK_PREPARED_CONTEXT" \
  "$AE_BENCHMARK_CONTROL_DIR" \
  "$AE_BENCHMARK_SCRATCH" \
  "$AE_BENCHMARK_PUBLICATION" \
  "$AE_SELECTED_DEPLOYMENT"
do
  test ! -e "$path" || {
    echo "refusing non-fresh rollout target: $path" >&2
    exit 1
  }
done
mkdir -p "$AE_BENCHMARK_CONTROL_DIR"
```

Create and independently validate `AE_SOURCE_SNAPSHOT_ROOT` with the two
snapshot commands above. Run the following GPU-consuming steps outside the
sandbox.

```bash
PYTHONPATH="$AE_SOURCE_SNAPSHOT_ROOT" /home/klkehl/thisenv/bin/python -P -u \
  "$AE_SOURCE_SNAPSHOT_ROOT/scripts/run_production_all_evidence_workflow.py" \
  --scientific-spec "$AE_SCIENTIFIC_SPEC" \
  --deployment-profile "$AE_BENCHMARK_STAGING_PROFILE" \
  --source-snapshot-root "$AE_SOURCE_SNAPSHOT_ROOT" \
  --adopt-checkpoint "$AE_PREPARED_CHECKPOINT" \
  --adopt-checkpoint "$AE_CACHE_CHECKPOINT" \
  --adopt-checkpoint "$AE_PREFLIGHT_AUDIT" \
  --stop-after stage1_preflight \
  --validation-depth fresh_terminal_audit \
  --log-level INFO
```

Do not advance this request to modeling or handoff: the authenticated workload
writer accepts only the exact preflight pause. It derives the representative
exact-inner and full-outer row counts from the configured split plan; neither
count is a library constant. Write those two selectors into the fresh workload
deployment and prepared-context roots:

```bash
PYTHONPATH="$AE_SOURCE_SNAPSHOT_ROOT" /home/klkehl/thisenv/bin/python -P -u \
  "$AE_SOURCE_SNAPSHOT_ROOT/scripts/write_role_neutral_benchmark_workload_deployment.py" \
  --workflow-root "$AE_BENCHMARK_STAGING_WORK_ROOT" \
  --benchmark-config "$AE_BENCHMARK_CONFIG" \
  --prepared-context-root "$AE_BENCHMARK_PREPARED_CONTEXT" \
  --scope-selector configured_exact_inner_fit exact_inner 0 \
  --scope-selector configured_full_outer_fit full_outer 0 \
  --output "$AE_BENCHMARK_WORKLOAD_DEPLOYMENT"
```

Run one observation with `--stop-after-observations 1`. Its sealed checkpoint
must be
`$AE_BENCHMARK_SCRATCH/checkpoints/observation_000000.json`. Then resume
without that stop option and publish the compact durable benchmark authority:

```bash
PYTHONPATH="$AE_SOURCE_SNAPSHOT_ROOT" /home/klkehl/thisenv/bin/python -P -u \
  "$AE_SOURCE_SNAPSHOT_ROOT/scripts/run_role_neutral_performance_benchmark.py" \
  --benchmark-config "$AE_BENCHMARK_CONFIG" \
  --workload-deployment "$AE_BENCHMARK_WORKLOAD_DEPLOYMENT" \
  --output-root "$AE_BENCHMARK_SCRATCH" \
  --stop-after-observations 1
```

```bash
PYTHONPATH="$AE_SOURCE_SNAPSHOT_ROOT" /home/klkehl/thisenv/bin/python -P -u \
  "$AE_SOURCE_SNAPSHOT_ROOT/scripts/run_role_neutral_performance_benchmark.py" \
  --benchmark-config "$AE_BENCHMARK_CONFIG" \
  --workload-deployment "$AE_BENCHMARK_WORKLOAD_DEPLOYMENT" \
  --output-root "$AE_BENCHMARK_SCRATCH" \
  --resume \
  --durable-publication-root "$AE_BENCHMARK_PUBLICATION"
```

Select the deployment with `--benchmark-publication`; a workload locator is
deliberately forbidden for durable selection. The resulting profile binds the
publication bytes, original result/workload identities, exact scientific spec,
dataset/model/settings identity, and transitive producer code without retaining
scratch locators.

```bash
PYTHONPATH="$AE_SOURCE_SNAPSHOT_ROOT" /home/klkehl/thisenv/bin/python -P -u \
  "$AE_SOURCE_SNAPSHOT_ROOT/scripts/select_role_neutral_benchmark_deployment.py" \
  --base-deployment "$AE_BASE_ACCEPTANCE_PROFILE" \
  --benchmark-publication "$AE_BENCHMARK_PUBLICATION" \
  --scientific-spec "$AE_SCIENTIFIC_SPEC" \
  --output "$AE_SELECTED_DEPLOYMENT"
```

Do not pass `--benchmark-workload-deployment` in this durable-publication
selection mode.

Before starting productive Stage 1, compare the selected preflight compression
with the staging profile:

- If they match, the fresh acceptance request must adopt four portable DAG
  nodes: migrated prepared cohort, migrated embedding cache, current clustered
  preflight, and granular `prepared_stage1_context`. Resolve each node from the
  authenticated staging manifests; never embed a generated node directory name
  in code or configuration.
- If they differ, the staged preflight and prepared context are incompatible.
  Adopt only the migrated prepared cohort/cache, pass the legacy preflight audit
  again, and recompute the current preflight/context under the selected codec.

Resolve the four controls from authenticated staging manifests. Do not derive
or embed the generated granular node directory name:

```bash
eval "$(
PYTHONPATH="$AE_SOURCE_SNAPSHOT_ROOT" \
/home/klkehl/thisenv/bin/python -P -B <<'PY'
import os
import shlex
from pathlib import Path

from oci.inference.production_all_evidence_workflow import (
    _read_json_object,
    validate_published_workflow_checkpoint_dag,
)

root = Path(
    os.environ["AE_BENCHMARK_STAGING_WORK_ROOT"]
).resolve(strict=True)
request = _read_json_object(
    root / "immutable_run_request.json",
    label="staging immutable request",
)
validated = validate_published_workflow_checkpoint_dag(
    work_root=root,
    expected_request_sha256=request["request_sha256"],
    expected_phases=(
        "input_preparation",
        "embedding_cache",
        "stage1_preflight",
    ),
)

phase_for_kind = {
    "prepared_cohort": "input_preparation",
    "embedding_cache": "embedding_cache",
}
portable = {}
for record, locator in zip(
    request["requested_checkpoint_adoptions"],
    request["checkpoint_adoption_locators"],
    strict=True,
):
    kind = record["artifact_kind"]
    if kind in phase_for_kind:
        phase = phase_for_kind[kind]
        assert (
            record["artifact_id"]
            == validated["checkpoint_artifact_ids"][phase]
        )
        assert kind not in portable
        portable[kind] = Path(locator).resolve(strict=True)
assert set(portable) == set(phase_for_kind)

preflight = (
    root / "portable_checkpoints" / "stage1_preflight"
).resolve(strict=True)
assert "stage1_preflight" in validated["local_publication_phases"]

granular_root = (
    root
    / "portable_granular_checkpoints"
    / "stage1_preflight"
)
index = _read_json_object(
    granular_root / "granular_index.json",
    label="Stage 1 preflight granular index",
)
locator = _read_json_object(
    granular_root / "granular_index_locator.json",
    label="Stage 1 preflight granular locator",
)
matches = [
    (node, control)
    for node, control in zip(
        index["nodes"],
        locator["node_controls"],
        strict=True,
    )
    if node["artifact_kind"] == "prepared_stage1_context"
    and node["artifact_id"] == control["artifact_id"]
]
assert len(matches) == 1
node, control = matches[0]
assert validated["granular_checkpoint_artifact_ids"][
    "stage1_preflight"
] == [node["artifact_id"]]
context = Path(control["control_root"]).resolve(strict=True)

for name, value in (
    ("AE_ADOPT_PREPARED", portable["prepared_cohort"]),
    ("AE_ADOPT_CACHE", portable["embedding_cache"]),
    ("AE_ADOPT_PREFLIGHT", preflight),
    ("AE_ADOPT_PREPARED_CONTEXT", context),
):
    print(f"export {name}={shlex.quote(str(value))}")
PY
)"
```

Compare the selected and staging codecs through their typed profiles:

```bash
eval "$(
PYTHONPATH="$AE_SOURCE_SNAPSHOT_ROOT" \
/home/klkehl/thisenv/bin/python -P -B <<'PY'
import os
import shlex
from pathlib import Path

from oci.inference.portable_workflow_spec import DeploymentProfile

staging = DeploymentProfile.from_json(
    Path(os.environ["AE_BENCHMARK_STAGING_PROFILE"])
)
selected = DeploymentProfile.from_json(
    Path(os.environ["AE_SELECTED_DEPLOYMENT"])
)
print(
    "export AE_STAGING_CODEC="
    + shlex.quote(staging.cluster_preflight_parquet_compression)
)
print(
    "export AE_SELECTED_CODEC="
    + shlex.quote(selected.cluster_preflight_parquet_compression)
)
PY
)"
```

Construct and retain one exact adoption array for both the initial request and
its resume:

```bash
if [ "$AE_SELECTED_CODEC" = "$AE_STAGING_CODEC" ]; then
  AE_ADOPTION_ARGS=(
    --adopt-checkpoint "$AE_ADOPT_PREPARED"
    --adopt-checkpoint "$AE_ADOPT_CACHE"
    --adopt-checkpoint "$AE_ADOPT_PREFLIGHT"
    --adopt-checkpoint "$AE_ADOPT_PREPARED_CONTEXT"
  )
else
  AE_ADOPTION_ARGS=(
    --adopt-checkpoint "$AE_ADOPT_PREPARED"
    --adopt-checkpoint "$AE_ADOPT_CACHE"
    --adopt-checkpoint "$AE_PREFLIGHT_AUDIT"
  )
fi
```

The productive request then pauses with `--stop-after handoff_validation`:

```bash
PYTHONPATH="$AE_SOURCE_SNAPSHOT_ROOT" /home/klkehl/thisenv/bin/python -P -u \
  "$AE_SOURCE_SNAPSHOT_ROOT/scripts/run_production_all_evidence_workflow.py" \
  --scientific-spec "$AE_SCIENTIFIC_SPEC" \
  --deployment-profile "$AE_SELECTED_DEPLOYMENT" \
  --source-snapshot-root "$AE_SOURCE_SNAPSHOT_ROOT" \
  "${AE_ADOPTION_ARGS[@]}" \
  --stop-after handoff_validation \
  --validation-depth fresh_terminal_audit \
  --log-level INFO
```

This is not `--stage1-only`: the full endpoint/model identity remains part of
the immutable request, but no endpoint call occurs before `stage2_canary`.
Only after the configured remote Stage 2 service is available, repeat the exact
profiles and adoption array with `--resume` and no `--stop-after`:

```bash
PYTHONPATH="$AE_SOURCE_SNAPSHOT_ROOT" /home/klkehl/thisenv/bin/python -P -u \
  "$AE_SOURCE_SNAPSHOT_ROOT/scripts/run_production_all_evidence_workflow.py" \
  --scientific-spec "$AE_SCIENTIFIC_SPEC" \
  --deployment-profile "$AE_SELECTED_DEPLOYMENT" \
  --source-snapshot-root "$AE_SOURCE_SNAPSHOT_ROOT" \
  "${AE_ADOPTION_ARGS[@]}" \
  --resume \
  --validation-depth fresh_terminal_audit \
  --log-level INFO
```

Do not occupy the local Stage 1 GPUs with vLLM when the resource policy rejects
external GPU occupants.

The remaining phase examples in this document apply only to an exact replay of
the retired low-level interface. Before using them, reconstruct
`AE_COMMON_ARGS` byte-for-byte from the original invocation record and immutable
preparation manifest. This document deliberately provides no fallback values:

```bash
# Historical replay only. Populate from authenticated records; never improvise.
AE_COMMON_ARGS=(
  # exact original arguments
)
```

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
