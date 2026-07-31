# Production all-evidence workflow quickstart

This is the operator guide for running the current integrated production path
without an interactive coding agent. The real entrypoint is
`scripts/run_production_all_evidence_workflow.py`. For the two included
synthetic benchmarks, use the repository-root launchers documented below; they
construct and invoke that entrypoint rather than running an alternate workflow.

The supplied launchers are deliberately specific to one deployment shape:

- one Linux VM;
- exactly eight visible NVIDIA Blackwell GPUs numbered `0` through `7`;
- Python 3.12 or 3.13 managed by `uv`;
- the included 1,000-patient synthetic NSCLC datasets; and
- a local eight-GPU vLLM server using the pinned
  `nvidia/Gemma-4-31B-IT-NVFP4` checkpoint for Stage 2.

For an arbitrary clinical cohort, read [Using a new cohort](#using-a-new-cohort)
before launching. A new cohort requires a reviewed scientific specification; it
is not safe to change only the dataset filename.

## What the workflow does

One successful invocation runs these authenticated phases in order:

1. `input_preparation`
2. `embedding_cache`
3. `stage1_preflight`
4. `stage1_modeling`
5. `handoff_validation`
6. `stage2_canary`
7. `stage2_inference`
8. `oracle_evaluation` for the included synthetic benchmarks
9. `terminal_validation`

Stage 1 retains all ten evidence families, nested fold honesty, token-attention
HTR evidence, canonical merge order, and component-level resume. Stage 2
interprets the authenticated handoff, extracts complete-note variables, fits
five strict outer-fold `econml.dml.CausalForestDML` models, and writes one
held-out prediction per patient.

The synthetic oracle is not available to preparation, evidence generation,
variable selection, extraction, forest fitting, or prediction. Its evaluation
phase runs only after the prediction bytes have been frozen and authenticated.

The Stage 2 prompt byte ceiling is a deterministic per-request packing limit.
It can create more lossless batches; it does not sample or truncate the
authenticated evidence.

## Host prerequisites

Before renting or configuring a VM, confirm the following:

- Eight NVIDIA Blackwell GPUs are visible as physical devices `0,1,2,3,4,5,6,7`.
- The installed NVIDIA driver supports the CUDA runtime used by the locked
  PyTorch and vLLM packages.
- `nvidia-smi` works for the unprivileged account that will run the workflow.
- The GPUs are otherwise idle. The production profile fails closed when it
  detects external GPU occupants.
- The host has at least eight logical CPUs. More CPUs and ample RAM improve
  preflight throughput.
- The open-file hard limit is at least 4,096. The launcher raises its soft limit
  as high as 65,536 when the host permits it.
- The repository and its `artifacts/` output tree live on a POSIX filesystem.
- There is substantial free storage. The pinned Stage 2 model alone is about
  32.7 GB, and complete Stage 1 token evidence can be much larger than the model
  files. Check both capacity and inode availability before launch.
- Outbound access to Hugging Face is available during `--prepare-only`, unless
  validated local model directories are supplied.
- `bash`, `awk`, `flock`, `nproc`, `nvidia-smi`, `ps`, `realpath`, `setsid`,
  `tail`, `tr`, and `uv` are installed.
- Loopback TCP ports 8002 and 8003 are free, or two different unprivileged ports
  are supplied through `VLLM_PROXY_PORT` and `VLLM_UPSTREAM_PORT`.

If Hugging Face requires authentication for the pinned checkpoint, authenticate
the VM account before preparation, for example by setting `HF_TOKEN` according
to the host's secret-management policy. Do not commit access tokens or place
them in deployment JSON.

Clinical data and production artifacts may contain sensitive text. Use an
appropriately controlled VM, encrypted storage, restrictive access controls,
and the study's required retention policy. The launchers set `umask 077`, but
that does not replace host-level security controls.

## Clone and install

Run from a committed repository revision. Uncommitted changes on another
machine are not transferred by Git.

```bash
git clone <repository-url> causal-dragonnet-text
cd causal-dragonnet-text
uv sync --frozen --extra extraction
```

The lock requires Python `>=3.12,<3.14`. `uv` creates `.venv/`; the production
launchers use `.venv/bin/python` and `.venv/bin/vllm` by default.

The launchers normally run the same frozen sync at every invocation. After the
explicit sync above, `SKIP_UV_SYNC=1` avoids repeating that check. Omit
`SKIP_UV_SYNC=1` if dependencies may have changed.

No pre-existing `artifacts/` directory is required. It is ignored by Git and is
created locally for models, immutable source snapshots, generated deployment
profiles, durable results, scratch data, locks, and vLLM logs.

## Choose one benchmark

Run the two benchmarks one at a time because each claims all eight GPUs.

| Benchmark | Launcher | Included dataset |
|---|---|---|
| One confounder, one effect modifier | `run_one_conf_one_mod_cloud_8gpu.sh` | `synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet` |
| Five confounders, five effect modifiers | `run_five_conf_five_mod_cloud_8gpu.sh` | `synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet` |

The examples below use the one-confounder launcher. Substitute the
five-confounder launcher verbatim for the other benchmark.

## Prepare, validate, and launch

### 1. Materialize models and deployment inputs

```bash
SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh --prepare-only
```

This command may take a while on first use. It:

- downloads exact pinned revisions of Qwen3-Embedding-8B, BERT-tiny, and the
  NVIDIA Gemma 4 NVFP4 Stage 2 model;
- copies them into real, symlink-free local model trees;
- validates the model identities and required files;
- validates all eight GPUs and host resource bounds;
- creates and validates an immutable production source snapshot; and
- generates the exact deployment profile for this run.

It does not fit a scientific model or contact the Stage 2 inference endpoint.
Rerunning it revalidates and reuses compatible materialized models.

### 2. Compile the exact cold-start request without running it

```bash
SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh --check-only
```

This verifies the local vLLM command, exact model identity, eight disjoint
Stage 1 owner lanes, token-attention HTR configuration, 256K Stage 2 context,
prompt batching limit, paths, and immutable request. It exits without starting
the workflow or vLLM.

Resolve every reported error before proceeding. Do not bypass an identity,
resource, occupancy, or compatibility failure by editing a generated manifest.

### 3. Run the complete workflow

Use a persistent terminal multiplexer such as `tmux`. To retain a readable
operator log while preserving the launcher's exit status:

```bash
mkdir -p artifacts/operator_logs
set -o pipefail
SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh \
  2>&1 | tee artifacts/operator_logs/one_conf_one_mod.log
```

The default invocation continues through Stage 2 and synthetic post-freeze
oracle evaluation. Do not start the other benchmark concurrently.

During preparation and Stage 1, the CPU-only loopback proxy does not initialize
CUDA. Stage 1 can therefore use all eight GPUs. On the first Stage 2 request,
the proxy starts vLLM with tensor parallelism across all eight GPUs, waits for
health and the exact served-model identity, and then forwards the request.

It is normal to see a transition period with little GPU activity while Stage 1
seals or authenticates an artifact and before vLLM finishes loading.

## Optional Stage 1-only pause

To stop after ordinary handoff validation without loading vLLM:

```bash
STOP_AFTER=handoff_validation \
SKIP_UV_SYNC=1 \
./run_one_conf_one_mod_cloud_8gpu.sh
```

Later, resume the same immutable request into Stage 2 by running the launcher
without `STOP_AFTER`:

```bash
SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh
```

The operational stop boundary is excluded from scientific identity. Do not
change the dataset, scientific profile, generated deployment, model trees, or
source snapshot between the two commands.

## Monitoring

The durable root for each supplied launcher is:

```text
artifacts/cloud_runs/one_conf_one_mod/
artifacts/cloud_runs/five_conf_five_mod/
```

The corresponding active scratch roots are:

```text
artifacts/cloud_scratch/one_conf_one_mod/
artifacts/cloud_scratch/five_conf_five_mod/
```

Useful checks from a second terminal include:

```bash
nvidia-smi

.venv/bin/python -m json.tool \
  artifacts/cloud_runs/one_conf_one_mod/workflow_progress.json

.venv/bin/python -m json.tool \
  artifacts/cloud_runs/one_conf_one_mod/recovery/stage1_scope_progress.json

find artifacts/cloud_runs/one_conf_one_mod/phases \
  -maxdepth 2 -name complete_manifest.json -print
```

The Stage 2 supervisor files are under:

```text
artifacts/cloud_scratch/vllm_one_conf_one_mod/proxy.log
artifacts/cloud_scratch/vllm_one_conf_one_mod/vllm.log
artifacts/cloud_scratch/vllm_one_conf_one_mod/status.json
```

Use `vllm_five_conf_five_mod` for the other benchmark. The vLLM log is absent
until the first Stage 2 request starts the server.

`workflow_progress.json` distinguishes active, completed, and failed phases.
Stage 1's recovery record reports sealed scope/component progress. A phase is
reusable only after its terminal manifest has been written and authenticated;
an attempt directory by itself is not completion.

## Interruption and resume

For an attached launcher, use `Ctrl-C` once. The launcher verifies its owned
process-group identities, forwards `SIGTERM` to the production group, and sends
`SIGTERM` to its owned Stage 2 proxy/vLLM group. It does not automatically use
`SIGKILL`.

Preserve the entire durable and scratch trees, including incomplete attempts.
To continue, rerun the exact same launcher with the same environment:

```bash
SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh
```

If its durable root already contains an immutable request, the launcher adds
`--resume` automatically. Compatible sealed phases and components are reused;
incomplete work is recomputed at the supported boundary.

Do not edit terminal manifests, generated deployment JSON, model files, or
sealed payloads. Do not point both hosts or both benchmark launchers at the same
active scratch or durable root.

## Starting a genuinely fresh run

The default run keys are stable so an ordinary rerun resumes. To create a new
attempt without deleting or overwriting the old one, first preserve the
generated deployment profile for the old request. For the one-confounder run:

```bash
mkdir -p /absolute/path/to/preserved_profiles
mv artifacts/runtime_profiles/current/one_conf_one_mod.json \
  /absolute/path/to/preserved_profiles/one_conf_one_mod.previous.json
```

Use `five_conf_five_mod.json` for the other benchmark. Then choose fresh run,
scratch, and source-snapshot paths:

```bash
export CLOUD_RUN_ROOT_BASE=/absolute/path/to/new_cloud_runs
export CLOUD_SCRATCH_ROOT_BASE=/absolute/path/to/new_cloud_scratch
export CLOUD_SOURCE_SNAPSHOT_ROOT=/absolute/path/to/new_source_snapshot

SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh --check-only
SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh
```

These paths must not identify an incompatible prior request. A compatible
`CLOUD_MODEL_ROOT` may be shared read-only or reused to avoid downloading model
bytes again, but the two active workflows must have distinct run and scratch
roots.

If source code changed after `production_source_snapshot_current` was created,
the launcher rejects the stale snapshot. Preserve it for the old run and select
a fresh `CLOUD_SOURCE_SNAPSHOT_ROOT` for the new code identity.

## Completion and outputs

A successful run prints `production workflow completed successfully`, and
`workflow_progress.json` reports completion with no error. It also has terminal
manifests for every configured phase, including `terminal_validation`.

Locate the frozen predictions and final run manifest without assuming an
attempt timestamp:

```bash
find artifacts/cloud_runs/one_conf_one_mod/phases/stage2_inference \
  -type f \( -name frozen_predictions.parquet \
  -o -name immutable_run_manifest.json \) -print

find artifacts/cloud_runs/one_conf_one_mod/phases/oracle_evaluation \
  -type f -name evaluation_metrics.json -print
```

The terminal validator checks, among other invariants, that the five held-out
fold outputs cover every patient exactly once, the estimator is the strict
causal-forest backend, prediction hashes agree, and oracle evaluation occurred
only after the frozen prediction checkpoint.

Keep at least these records together for reproducibility:

- the committed repository revision;
- the immutable source snapshot;
- the scientific and generated deployment profiles;
- the complete durable run root;
- the component store and required scratch artifacts;
- materialized model identities;
- the operator log; and
- the terminal validation and oracle-evaluation reports.

## Supported operational overrides

The launchers expose a small number of deployment-only controls. Set them before
`--prepare-only` and keep them unchanged for the corresponding immutable run.

| Variable | Purpose | Default |
|---|---|---|
| `CLOUD_CPU_BUDGET` | CPUs available to production | all CPUs reported by `nproc` |
| `PREFLIGHT_MEMORY_BUDGET_BYTES` | Total bounded preflight memory budget | 75% of host RAM |
| `PREFLIGHT_ESTIMATED_OWNER_PEAK_BYTES` | Per-owner preflight planning estimate | 8 GiB |
| `EMBEDDING_BATCH_SIZE` | Canonical per-worker embedding batch size | 8 |
| `VLLM_GPU_MEMORY_UTILIZATION` | vLLM GPU memory target | 0.90 |
| `VLLM_STARTUP_TIMEOUT_SECONDS` | Time allowed for vLLM startup | 600 seconds |
| `VLLM_PROXY_PORT` | CPU-only proxy listen port | 8002 |
| `VLLM_UPSTREAM_PORT` | private vLLM listen port | 8003 |
| `UV_CACHE_DIR` | uv download/build cache | `.uv-cache` in the repository |
| `CLOUD_MODEL_ROOT` | materialized pinned model root | `artifacts/local_models/current` |
| `CLOUD_RUN_ROOT_BASE` | durable run base | `artifacts/cloud_runs` |
| `CLOUD_SCRATCH_ROOT_BASE` | scratch base | `artifacts/cloud_scratch` |
| `CLOUD_SOURCE_SNAPSHOT_ROOT` | immutable source snapshot | `artifacts/production_source_snapshot_current` |

`CLOUD_GPU_COUNT` and `CLOUD_VISIBLE_DEVICES` are validated rather than
generalized by these wrappers: they must remain `8` and `0,1,2,3,4,5,6,7`.

Advanced operators may supply already materialized, compatible model directories
through `EMBEDDING_MODEL_DIR`, `HTR_MODEL_DIR`, `STAGE2_TOKENIZER_DIR`, and
`STAGE2_VLLM_MODEL_DIR`. Each directory is still authenticated and must be a
real local tree, not a symlink.

## Common failures

### An external process is using a GPU

Stop or relocate the external process, then rerun `--check-only`. Do not weaken
the external-occupant check: Stage 1 and Stage 2 are sized to own all eight GPUs.

### A loopback port is occupied

Choose two unused, distinct ports and use the same values on every invocation:

```bash
export VLLM_PROXY_PORT=18002
export VLLM_UPSTREAM_PORT=18003
```

### Model download failed

Check network access, storage, and Hugging Face credentials, then rerun
`--prepare-only`. Publication is atomic; a partial model tree is never accepted
as a completed pinned model.

### The source snapshot is stale

Code changed after the snapshot was sealed. Resume an existing run with its
original code and snapshot, or start a new run with fresh run, scratch, and
snapshot paths. Never overwrite the old snapshot.

### The deployment profile differs from an existing file

The requested dataset, paths, endpoint identity, or resource settings changed.
Use the original values to resume, or choose fresh run/scratch/profile roots for
a new request.

### GPUs are quiet

Read `workflow_progress.json` and the operator log before assuming a hang.
Input authentication, artifact sealing, process startup, CPU-only TF-IDF work,
and the Stage 1-to-vLLM transition may temporarily leave GPUs idle.

### Stage 2 failed

Inspect `proxy.log`, `vllm.log`, `status.json`, and the workflow error in
`workflow_progress.json`. Preserve the run and rerun the same launcher after
correcting an operational fault; validated Stage 2 requests and sealed upstream
work resume at their supported content-addressed boundaries.

## Using a new cohort

The two benchmark launchers are not generic cohort generators. In particular,
their deployment builder configures the included synthetic oracle columns
`patient_id` and `true_ite_prob`. Do not point one at a real cohort merely by
changing `CLOUD_DATASET_RELATIVE`.

A real-cohort production request needs two separately reviewed inputs:

1. A path-neutral `ScientificWorkflowSpec`, based on
   `example_configs/portable_all_evidence_scientific_nsclc.json`, defining the
   clinical question, estimand, four input columns, folds, all ten architecture
   profiles, text geometry, Stage 2 protocol, causal estimator, seed, and prompt
   identities.
2. A `DeploymentProfile`, based on
   `example_configs/portable_all_evidence_deployment_nsclc.stage1-only.example.json`,
   defining absolute data/model/output paths, selected devices, CPU and
   concurrency budgets, storage behavior, and the Stage 2 endpoint/model.

The production cohort must be Parquet and provide, at minimum:

- a non-null, unique, stable unit identifier;
- the complete clinical text used for evidence generation and extraction;
- a binary treatment indicator; and
- a binary observed outcome indicator.

The configured column names need not match the synthetic names. Every fold and
nested fitting partition must have enough treatment-arm and outcome-class
support; exact preflight rejects a cohort that does not. Extra columns are not
automatically authorized as covariates. Oracle or future-information columns
must remain excluded from the scientific projection.

For a real cohort, set `oracle_source`, `oracle_unit_id_column`, and
`oracle_ite_column` to `null` unless a separately authorized synthetic
post-prediction evaluation genuinely exists. Do not reinterpret an observed
clinical column as an oracle.

After peer review and immutable source-snapshot creation, the generic execution
path is non-interactive. Create a new snapshot at an absent target and validate
it immediately:

```bash
export AE_REPOSITORY_ROOT="$(pwd -P)"
export AE_SOURCE_SNAPSHOT_ROOT=/absolute/path/to/absent_source_snapshot

PYTHONPATH="$AE_REPOSITORY_ROOT" \
uv run --frozen --extra extraction python -P - <<'PY'
import os
from pathlib import Path

from oci.inference.production_source_snapshot import (
    create_production_source_snapshot,
    validate_production_source_snapshot,
)

repository = Path(os.environ["AE_REPOSITORY_ROOT"])
target = Path(os.environ["AE_SOURCE_SNAPSHOT_ROOT"])
created = create_production_source_snapshot(
    repository_root=repository,
    target_dir=target,
)
validated = validate_production_source_snapshot(target)
if created.content_sha256 != validated.content_sha256:
    raise SystemExit("source snapshot identity changed during validation")
print(validated.content_sha256)
PY
```

Then invoke the entrypoint from that snapshot:

```bash
PYTHONPATH="$AE_SOURCE_SNAPSHOT_ROOT" \
uv run --frozen --extra extraction python -P \
  "$AE_SOURCE_SNAPSHOT_ROOT/scripts/run_production_all_evidence_workflow.py" \
  --scientific-spec /absolute/path/to/scientific_workflow.json \
  --deployment-profile /absolute/path/to/deployment_profile.json \
  --source-snapshot-root "$AE_SOURCE_SNAPSHOT_ROOT" \
  --validation-depth fresh_terminal_audit \
  --log-level INFO
```

Use `--stop-after handoff_validation` for a deliberate Stage 1-only pause and
repeat the same command with `--resume` and without the stop boundary after the
configured Stage 2 endpoint is available.

Unlike the supplied benchmark launchers, this direct generic invocation does
not start a local vLLM supervisor. The endpoint and exact model named by the
deployment profile must already be available before Stage 2, or the run should
pause at `handoff_validation` until the endpoint is ready.

The repository does not currently claim that a generic script can choose an
appropriate clinical question, estimand, causal roles, fold design, or resource
profile for an arbitrary study. Those are scientific and deployment decisions,
not launch ceremony, and require review. Once the two typed profiles are fixed,
the production entrypoint itself is non-interactive and does not require Codex.

## Deeper references

- [Production Stage 1 bundle runbook](production_stage1_bundle_runbook.md) explains
  all ten evidence families and the Stage 1 scientific contracts.
- [Stage 1 hierarchy handoff contract](production_stage1_hierarchy_handoff_contract.md)
  specifies the authenticated Stage 1-to-Stage 2 boundary.
- [All-evidence discovery interfaces](all_evidence_discovery_interfaces.md)
  describes architecture-local interpretation and cumulative review.
- [Exact-inner production contract](stage1_exact_inner_production_contract.md)
  documents nested fitting and fold-honesty requirements.
- [Reproducibility and recovery runbook](hierarchical_all_evidence_reproducibility_runbook.md)
  provides background on preservation and replay. Its older low-level command
  examples are historical; use this quickstart for the current launch path.
