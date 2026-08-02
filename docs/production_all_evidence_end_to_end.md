# Production all-evidence workflow: end-to-end guide

This is the canonical guide to the current multi-model Stage 1 to Stage 2
production workflow. It covers a genuinely new cohort, the scientific and
deployment profiles, execution, artifact layout, monitoring, interruption,
resume, checkpoint reuse, and final outputs.

The only production entrypoint described here is:

```text
scripts/run_production_all_evidence_workflow.py
```

The repository's benchmark launchers compile and invoke this entrypoint. They
are deployment conveniences, not alternative scientific workflows.

For the supplied one-confounder and five-confounder benchmark launchers, see
[Production all-evidence quickstart](production_all_evidence_quickstart.md).
For a new cohort, use this document.

## Workflow in one view

```text
reviewed Parquet cohort + scientific profile + deployment profile
  |
  v
input preparation
  -> immutable row order and projected text/treatment/outcome cohort
  |
  v
multi-GPU embedding cache
  -> authenticated chunk embeddings and exact row/text binding
  |
  v
Stage 1 preflight
  -> split plan, non-truncation audit, reusable per-owner KMeans/SVD states
  |
  v
Stage 1 modeling
  -> all ten evidence families in nested fold-honest physical owners
  -> raw token/chunk attention and deterministic semantic catalogs
  -> direct numerical bank and Stage 1 handoff
  |
  v
fresh-process handoff validation
  |
  v
Stage 2 canary
  -> endpoint/model/schema/prompt checks on the authenticated handoff
  |
  v
Stage 2 inference
  -> hierarchical evidence review
  -> at most 20 note-readable variable definitions
  -> complete-note paged extraction
  -> post-extraction causal review
  -> one strict CausalForestDML fit per outer fold
  -> every patient predicted exactly once while held out
  -> frozen_predictions.parquet
  |
  v
optional post-freeze synthetic oracle evaluation
  |
  v
fresh-process terminal validation
```

The production phase order is:

1. `input_preparation`
2. `embedding_cache`
3. `stage1_preflight`
4. `stage1_modeling`
5. `handoff_validation`
6. `stage2_canary`
7. `stage2_inference`
8. `oracle_evaluation`
9. `terminal_validation`

`oracle_evaluation` is present in the phase graph but is a closed no-op when
no synthetic oracle is configured.

## Scientific design

### Stage 1 evidence families

Every logical context contains all ten families:

1. word treatment/outcome associations;
2. residual word effects;
3. hierarchical transformer evidence;
4. matched-patient uplift;
5. whole-cohort embeddings;
6. cluster-local embeddings;
7. lexical-retrieval contrasts;
8. TF-IDF topics;
9. residual TF-IDF n-grams; and
10. learned neural queries.

The physical component directories are `bow`, `htr`, `matched_pair`,
`embeddings`, `tfidf`, and `neural_query`. Some components produce more than
one evidence family, which is why there are six component directories but ten
families.

The HTR encoder is the fully unfrozen configured BERT model. The current
profile requires learned `token_attention` pooling within each chunk, followed
by the document-level chunk transformer and its learned document pool token.
The sealed HTR evidence retains complete raw token attention and chunk
attention. The product of those weights is a ranking heuristic, not a causal
attribution.

Raw token occurrences remain in authenticated columnar sidecars. Stage 2 sees
deterministic semantic aggregates and reverse indexes rather than millions of
occurrence-level prompt objects. Prompt byte ceilings create additional
lossless batches; they do not authorize top-k selection, sampling, or evidence
truncation.

### Nested fold domains

Let:

- `O` be `folds.outer_folds`;
- `I` be `folds.initial_training_partitions + folds.review_rounds`; and
- `R` be `folds.review_rounds`.

The workflow creates `O * (1 + I + R)` logical contexts:

- one full-outer context per outer fold;
- `I` exact-inner contexts per outer fold; and
- `R` cumulative-spent review contexts per outer fold.

Contexts with identical fit rows and scientific identities are deduplicated to
physical owners. The supplied five-outer-fold, five-inner-partition,
two-review-round profile therefore has 40 logical contexts and 35 physical
owners.

Exact-inner held-out transforms form the row-wise OOF Stage 1 bank for an
outer-training set. A full-outer model transforms only that outer fold's
held-out patients. Cumulative contexts expose only evidence available at their
review epoch. Stage 2 then performs its separate, configured nuisance
cross-fitting inside each strict outer-fold `econml.dml.CausalForestDML` fit.

## Inputs for a new cohort

### Cohort file

The input must be one Parquet file with four configured columns:

| Role | Required property |
|---|---|
| Unit ID | non-null, stable, and unique |
| Clinical text | complete temporally valid note text; non-null after the configured preprocessing policy |
| Treatment | binary `0/1` |
| Observed outcome | binary `0/1` for the current registered estimand |

The column names are declared in the scientific profile and do not need to use
the benchmark names. Row order is frozen and authenticated during input
preparation. The workflow constructs its own deterministic outer and inner
fold registry; a pre-existing split column is not the authority for this path.

Do not include post-treatment, future, or oracle information in any configured
input role. Extra Parquet columns are not automatically authorized as
features. For an ordinary real cohort, the deployment's `oracle_source`,
`oracle_unit_id_column`, and `oracle_ite_column` must all be `null`.

Before profile review, run a basic structural audit. Set the names to match the
prospective scientific profile:

```bash
export AE_DATASET=/absolute/path/to/cohort.parquet
export AE_UNIT_ID=patient_id
export AE_TEXT=clinical_text
export AE_TREATMENT=treatment_indicator
export AE_OUTCOME=outcome_indicator

uv run --frozen --extra extraction python - <<'PY'
import os
import pandas as pd

path = os.environ["AE_DATASET"]
columns = [
    os.environ["AE_UNIT_ID"],
    os.environ["AE_TEXT"],
    os.environ["AE_TREATMENT"],
    os.environ["AE_OUTCOME"],
]
frame = pd.read_parquet(path, columns=columns)
if frame.empty:
    raise SystemExit("cohort is empty")
if frame[columns[0]].isna().any() or not frame[columns[0]].is_unique:
    raise SystemExit("unit IDs must be non-null and unique")
if frame[columns[1]].isna().any():
    raise SystemExit("clinical text contains nulls")
for name in columns[2:]:
    observed = set(frame[name].dropna().unique().tolist())
    if frame[name].isna().any() or not observed <= {0, 1} or not observed:
        raise SystemExit(f"{name} is not non-null binary 0/1")
print({"rows": len(frame), "columns": columns})
PY
```

This is only a structural check. Production preflight performs the exact
row-order, class-support, fold-feasibility, text-coverage, and oracle-exclusion
checks.

### Scientific workflow profile

Start from:

```text
example_configs/portable_all_evidence_scientific_nsclc.json
```

Copy it to a study-controlled path and review every field. It contains choices
that may change the scientific answer, including:

- column roles and the clinical question;
- estimand and outcome type;
- outer folds, exact-inner partitions, and cumulative review rounds;
- all ten architecture profiles and their scientific hyperparameters;
- HTR tokenizer/chunking and `token_attention` pooling;
- complete-note paging and embedding text geometry;
- preprocessing and temporal-validity policy;
- Stage 2 prompts, generation policies, response schemas, and byte/token
  bounds;
- maximum candidate-variable count;
- post-extraction causal review;
- strict causal-forest and nuisance-model specifications; and
- canonical seed and seed policy.

Do not merely replace the dataset column names and assume the benchmark
science is appropriate. The question, estimand, temporal cutoff, fold support,
text geometry, prompts, and estimator settings require study review.

The separate Stage 1 and neural-query implementation profiles referenced by
the deployment normally start from:

```text
example_configs/production_all_evidence_stage1_full.json
example_configs/production_all_evidence_neural_query_full.json
```

They are bound into the compiled request and must agree with the portable
scientific profile. Do not independently change one copy of a scientific
setting to bypass profile validation.

### Deployment profile

Start from:

```text
example_configs/portable_all_evidence_deployment_nsclc.stage1-only.example.json
```

The deployment profile supplies physical and operational choices:

- absolute cohort, durable-root, scratch-root, and model paths;
- selected `cpu`, `auto`, or explicit `cuda:N` devices;
- embedding and HTR model locators;
- Stage 1 and neural-query profile locators;
- CPU, memory, I/O, GPU-safety, and concurrency budgets;
- Stage 1 owner-capacity policy;
- Stage 2 endpoint, served-model name, exact tokenizer tree, and response
  concurrency;
- causal-forest operational CPU/Ray settings; and
- optional synthetic-oracle paths.

For a full workflow, `endpoint`, `endpoint_model`, and
`stage2_tokenizer_locator` must be non-null. The tokenizer tree must match the
served model so prompt-length proofs are local and fail closed. For a permanent
Stage-1-only request, those three values may be null.

Use `stage1_execution.owner_capacity_policy.mode = "resource_autodetect"` for
generic deployments. The configured per-device and global owner counts are
hard ceilings. At launch, production lowers them as necessary using free VRAM
after the configured reserve, available host RAM, and the CPU budget. Resource
assignment and completion order are not part of scientific identity.

Model directories must be immutable, local, real directory trees rather than
symlinked Hugging Face cache views. The benchmark launchers use
`scripts/materialize_production_models.py` to create such trees; the same tool
may be used for a new deployment after selecting and reviewing exact model
revisions.

### Stage 2 endpoint transport and credentials

The deployment profile stores the endpoint and model identity, never a secret.
Select transport and authentication through the runtime environment before the
request is first created, and use the same values on resume:

| Endpoint | Environment |
|---|---|
| Local vLLM | `OCI_STAGE2_ENDPOINT_TRANSPORT=vllm`, `OCI_STAGE2_ENDPOINT_AUTH=none` |
| OpenAI-compatible bearer endpoint | `OCI_STAGE2_ENDPOINT_TRANSPORT=openai_compatible`, `OCI_STAGE2_ENDPOINT_AUTH=api_key`, `OCI_STAGE2_ENDPOINT_API_KEY=...` |
| Google Vertex OpenAI-compatible endpoint | `OCI_STAGE2_ENDPOINT_TRANSPORT=google_vertex`, `OCI_STAGE2_ENDPOINT_AUTH=google_adc` |

The credential value is resolved immediately before transport and is not
written into the immutable request. Use the host's secret manager rather than
committing it or placing it in shell history.

## Install and validate the request inputs

Use a committed repository revision and its lock:

```bash
git clone <repository-url> causal-dragonnet-text
cd causal-dragonnet-text
uv sync --frozen --extra extraction
```

Set canonical paths:

```bash
export AE_REPOSITORY_ROOT="$(pwd -P)"
export AE_SCIENTIFIC=/absolute/path/to/reviewed_scientific_workflow.json
export AE_DEPLOYMENT=/absolute/path/to/reviewed_deployment_profile.json
export AE_DURABLE=/absolute/path/to/new_durable_run
export AE_SCRATCH=/absolute/path/to/new_scratch_root
export AE_SOURCE_SNAPSHOT=/absolute/path/to/new_source_snapshot
```

`AE_DURABLE` and `AE_SCRATCH` must match the deployment JSON. For a genuinely
fresh run they must not contain a prior request or component store.

Parse both closed profiles before doing expensive work:

```bash
AE_SCIENTIFIC="$AE_SCIENTIFIC" AE_DEPLOYMENT="$AE_DEPLOYMENT" \
uv run --frozen --extra extraction python - <<'PY'
import os
from oci.inference.portable_workflow_spec import (
    DeploymentProfile,
    ScientificWorkflowSpec,
)

scientific = ScientificWorkflowSpec.from_json(os.environ["AE_SCIENTIFIC"])
deployment = DeploymentProfile.from_json(os.environ["AE_DEPLOYMENT"])
print(
    {
        "columns": scientific.columns,
        "outer_folds": scientific.folds.outer_folds,
        "devices": deployment.devices,
        "durable_root": str(deployment.durable_artifact_root),
        "scratch_root": str(deployment.scratch_root),
        "stage2_enabled": deployment.endpoint is not None,
    }
)
PY
```

Create one immutable source snapshot at an absent path:

```bash
AE_REPOSITORY_ROOT="$AE_REPOSITORY_ROOT" \
AE_SOURCE_SNAPSHOT="$AE_SOURCE_SNAPSHOT" \
PYTHONPATH="$AE_REPOSITORY_ROOT" \
uv run --frozen --extra extraction python -P - <<'PY'
import os
from pathlib import Path
from oci.inference.production_source_snapshot import (
    create_production_source_snapshot,
    validate_production_source_snapshot,
)

repository = Path(os.environ["AE_REPOSITORY_ROOT"])
target = Path(os.environ["AE_SOURCE_SNAPSHOT"])
created = create_production_source_snapshot(
    repository_root=repository,
    target_dir=target,
)
validated = validate_production_source_snapshot(target)
if created.content_sha256 != validated.content_sha256:
    raise SystemExit("source snapshot changed during validation")
print(validated.content_sha256)
PY
```

Never overwrite a source snapshot. A later source revision gets a new snapshot
path.

## Run a fresh workflow

### Option A: pause after validated Stage 1, then continue to Stage 2

Use this when the full deployment profile already names the eventual Stage 2
endpoint but you do not want to contact it yet:

```bash
PYTHONPATH="$AE_SOURCE_SNAPSHOT" \
uv run --frozen --extra extraction python -P -u \
  "$AE_SOURCE_SNAPSHOT/scripts/run_production_all_evidence_workflow.py" \
  --scientific-spec "$AE_SCIENTIFIC" \
  --deployment-profile "$AE_DEPLOYMENT" \
  --source-snapshot-root "$AE_SOURCE_SNAPSHOT" \
  --resume-trust trusted-local \
  --validation-depth fresh_terminal_audit \
  --log-level INFO \
  --stop-after handoff_validation
```

The workflow exits with `status: paused` after the Stage 1 handoff has passed
fresh-process validation. To continue, start the endpoint and rerun the same
request with `--resume` and without `--stop-after`:

```bash
PYTHONPATH="$AE_SOURCE_SNAPSHOT" \
uv run --frozen --extra extraction python -P -u \
  "$AE_SOURCE_SNAPSHOT/scripts/run_production_all_evidence_workflow.py" \
  --scientific-spec "$AE_SCIENTIFIC" \
  --deployment-profile "$AE_DEPLOYMENT" \
  --source-snapshot-root "$AE_SOURCE_SNAPSHOT" \
  --resume \
  --resume-trust trusted-local \
  --validation-depth fresh_terminal_audit \
  --log-level INFO
```

The stop boundary is operational and does not alter scientific identity.

### Option B: run the complete workflow immediately

Start the configured endpoint, then run the same command without
`--stop-after` and without `--resume`:

```bash
PYTHONPATH="$AE_SOURCE_SNAPSHOT" \
uv run --frozen --extra extraction python -P -u \
  "$AE_SOURCE_SNAPSHOT/scripts/run_production_all_evidence_workflow.py" \
  --scientific-spec "$AE_SCIENTIFIC" \
  --deployment-profile "$AE_DEPLOYMENT" \
  --source-snapshot-root "$AE_SOURCE_SNAPSHOT" \
  --resume-trust trusted-local \
  --validation-depth fresh_terminal_audit \
  --log-level INFO
```

### Option C: create a permanently Stage-1-only request

Use `--stage1-only` only when the deployment has no Stage 2 endpoint and the
request is intentionally not going to continue into Stage 2:

```bash
PYTHONPATH="$AE_SOURCE_SNAPSHOT" \
uv run --frozen --extra extraction python -P -u \
  "$AE_SOURCE_SNAPSHOT/scripts/run_production_all_evidence_workflow.py" \
  --scientific-spec "$AE_SCIENTIFIC" \
  --deployment-profile "$AE_DEPLOYMENT" \
  --source-snapshot-root "$AE_SOURCE_SNAPSHOT" \
  --stage1-only \
  --resume-trust trusted-local \
  --validation-depth fresh_terminal_audit \
  --log-level INFO
```

This is a different phase sequence. Do not use it as a substitute for
`--stop-after handoff_validation` when Stage 2 will later be required.

## What is saved where

The paths recorded in each `complete_manifest.json` are authoritative. The
tree below shows the stable layout; attempt timestamps and content hashes vary.

### Durable run root

```text
$AE_DURABLE/
  immutable_run_request.json
  workflow_progress.json
  phases/
    input_preparation/
      complete_manifest.json
      attempt_<UTC>/
        prepared/modeling_cohort.parquet
        prepared/preparation_manifest.json
    embedding_cache/
      complete_manifest.json
      attempt_<UTC>/
        prepared/modeling_cohort.parquet
        embedding_cache/...
    stage1_preflight/
      complete_manifest.json
      attempt_<UTC>/
        effective_stage1_profile.json
        stage1_preflight_report.json
        cluster_preflight/cluster_preflight_manifest.json
        cluster_preflight_states/cluster_state_bundle_manifest.json
        prepared_stage1_context/...
    stage1_modeling/
      complete_manifest.json
      attempt_<UTC>/
        role_neutral_stage1_execution/
        stage1_bundle/
          bundle_manifest.json
          row_registry.parquet
        direct_upstream_numerical_reference_bank/
          direct_upstream_numerical_manifest.json
          locator_attestation.json
        role_neutral_handoff_binding.json
    handoff_validation/
      complete_manifest.json
      attempt_<UTC>/fresh_handoff_validation.json
    stage2_canary/
      complete_manifest.json
      attempt_<UTC>/...
    stage2_inference/
      complete_manifest.json
      attempt_<UTC>/
        full_preparation/...
        full_output/
          frozen_predictions.parquet
          immutable_run_manifest.json
          ...
        full_attestation/...
    oracle_evaluation/
      complete_manifest.json
      attempt_<UTC>/evaluation/evaluation_metrics.json
    terminal_validation/
      complete_manifest.json
      attempt_<UTC>/validation.json
  portable_checkpoints/
    <phase>/artifact_manifest.json
    <phase>/artifact_locator.json
  portable_granular_checkpoints/
    stage1_preflight/...
    stage1_modeling/...
    stage2_inference/...
  execution_attestations/
    execution_epochs/...
    phase_payload_authentication/...
    portable_checkpoint_publications/...
    performance_telemetry.json
    run_control/...
  recovery/
    stage1_scope_progress.json
    stage1_scope_attempts/...
    cluster_preflight_scope_inputs/...
    ...
```

Important durable records are:

- `immutable_run_request.json`: the closed scientific request, concrete input
  identities, model-tree identities, phase producer identities, and initial
  operational binding;
- `workflow_progress.json`: current state, phase counts, current error,
  resolved owner capacity, and execution epoch;
- `phases/<phase>/complete_manifest.json`: the only phase-level completion
  authority;
- `portable_checkpoints/`: phase-level content-addressed adoption handles;
- `portable_granular_checkpoints/`: reusable Stage 1 owner/component and Stage
  2 response/extraction/review/fold nodes;
- `execution_attestations/`: resource epochs, authentication proofs, checkpoint
  publications, run-control selection, and performance telemetry; and
- `recovery/`: progress and incomplete-attempt records used by supported
  in-place recovery.

The Stage 2 phase also retains deterministic request/response caches,
model-facing evidence batches, extraction ledgers, per-fold review trees,
per-fold forest manifests, and per-fold held-out predictions. The exact paths
are registered in the Stage 2 phase and granular checkpoint manifests.

### Scratch root

```text
$AE_SCRATCH/production_all_evidence_workflow/
  <request_sha256>/
    <in-place-resumable-phase>/attempt_<UTC>/...
  stage1_component_store/
    <scientific_compatibility_key>/
      component_store_manifest.json
      components/
        <physical_owner_scope_id>/
          bow/...
          htr/...
          matched_pair/...
          embeddings/...
          tfidf/...
          neural_query/...
  stage1_reusable_preflight_store_v2/
    global_audits/...
    owner_artifacts/...
    assembled_contexts/...
    accepted_contexts/...
    authentication_proofs/...
    recovery/...
```

The stable component store is keyed by dataset and ordered rows, split plan,
scientific profiles, model identities, seeds, and component-producer
compatibility. GPU assignment, CPU budget, and owner concurrency are excluded.
Each component publishes through a temporary attempt directory; its component
`execution_manifest.json` marks successful completion. Incomplete attempts are
preserved but are never accepted as complete.

The reusable preflight store separates the global text/token non-truncation
audit, each physical owner's cluster state, and assembled contexts. Compatible
reopens can use protected prior byte proofs plus stat continuity and lazily
load owner states instead of retokenizing all notes or refitting KMeans/SVD.

Do not discard the scratch tree merely because the durable root exists. It is
part of component-granular Stage 1 resume and may contain the only accepted
copy of a large sealed component or an in-progress Stage 2 cache. Keep both
trees until terminal validation has completed and the study's archival plan
has been executed.

### Phase output summary

| Phase | Main scientific result | Resume boundary |
|---|---|---|
| Input preparation | projected, ordered modeling cohort and preparation manifest | sealed phase |
| Embedding cache | complete chunk embeddings, offsets/text inventory, cache identity | sealed phase; portable import supported |
| Stage 1 preflight | split/context plan, exact non-truncation audit, frozen cluster states, prepared context | sealed phase plus reusable sub-artifacts |
| Stage 1 modeling | six physical components covering ten families, logical bindings, raw HTR evidence, semantic catalogs, direct bank, handoff | component-granular plus sealed phase |
| Handoff validation | fresh-process report over the complete Stage 1 handoff | sealed phase |
| Stage 2 canary | endpoint/model/schema and bounded prompt acceptance | sealed phase and response cache |
| Stage 2 inference | evidence review, feature definitions, extraction ledgers, causal reviews, outer-fold forests and predictions | request/extraction/review/fold granular checkpoints plus sealed phase |
| Oracle evaluation | optional joined synthetic truth and metrics, opened after frozen predictions | sealed phase |
| Terminal validation | fresh-process validation of the whole requested DAG | sealed phase |

## Monitor a run

Set the durable and scratch roots in a second shell:

```bash
export RUN=/absolute/path/to/durable_run
export SCRATCH=/absolute/path/to/scratch_root
```

Current phase and error:

```bash
jq . "$RUN/workflow_progress.json"
```

Phase-level completion:

```bash
find "$RUN/phases" -mindepth 2 -maxdepth 2 \
  -name complete_manifest.json -print | sort
```

Stage 1 owner/component progress:

```bash
jq . "$RUN/recovery/stage1_scope_progress.json"

find "$SCRATCH/production_all_evidence_workflow/stage1_component_store" \
  -path '*/components/*/*/execution_manifest.json' -print | wc -l
```

Resolved Stage 1 concurrency and operational telemetry:

```bash
jq '{
  status,
  current_phase,
  stage1_owner_capacity_attestation,
  stage1_execution_profile,
  error
}' "$RUN/workflow_progress.json"

jq . "$RUN/execution_attestations/performance_telemetry.json"
```

GPU utilization alone is not a progress indicator. Input authentication,
artifact publication, CPU-only TF-IDF work, Stage 2 request preparation, and
fresh-process validation can legitimately leave GPUs quiet.

## Interrupt and resume

### Normal interruption

Send `SIGTERM` to the verified workflow-owned process group. The supplied
launchers do this when they receive one `Ctrl-C`. Do not kill unrelated GPU
processes and do not automatically escalate to `SIGKILL`.

Preserve:

- the entire durable root;
- the entire scratch root;
- the source snapshot;
- both profiles and referenced implementation profiles;
- model trees;
- endpoint transport/auth mode; and
- operator logs.

### Resume the same request

Rerun the same command and add `--resume`:

```bash
PYTHONPATH="$AE_SOURCE_SNAPSHOT" \
uv run --frozen --extra extraction python -P -u \
  "$AE_SOURCE_SNAPSHOT/scripts/run_production_all_evidence_workflow.py" \
  --scientific-spec "$AE_SCIENTIFIC" \
  --deployment-profile "$AE_DEPLOYMENT" \
  --source-snapshot-root "$AE_SOURCE_SNAPSHOT" \
  --resume \
  --resume-trust trusted-local \
  --validation-depth fresh_terminal_audit \
  --log-level INFO
```

On resume, the workflow:

1. reopens the immutable request;
2. authenticates completed phases;
3. skips compatible sealed phases;
4. authenticates and skips sealed Stage 1 components;
5. preserves but does not accept incomplete attempts;
6. resumes supported Stage 2 request/extraction/review/fold caches; and
7. recomputes only missing or incompatible work.

Do not add `--resume` to the first invocation of a genuinely absent durable
root. Do not edit a terminal manifest to make an attempt appear complete.

### What authentication means

Authentication proves that a manifest still points to the exact files,
schemas, hashes, row identities, split plan, producer/scientific identities,
and coverage that were sealed. It prevents a partial, moved, modified, or
scientifically incompatible artifact from silently entering a later fold.

The supported policies are:

- `trusted-local` (recommended on the same filesystem): after one full-byte
  validation, exact protected proof and file-stat continuity can avoid
  rereading multi-gigabyte payloads; any discontinuity falls back to deep
  authentication;
- `strict-portable`: reread and hash payload bytes after each process restart;
  use for a different or untrusted filesystem; and
- `manifest-local`: explicitly accept a private sealed manifest/stat inventory
  where no prior proof exists. It is faster and slightly higher risk because a
  metadata-preserving corruption may escape that local check. Fresh handoff and
  terminal validation still run.

`trusted-local` reports whether each phase used prior-proof/stat continuity,
full-byte authentication, or recomputation under
`execution_attestations/performance_telemetry.json`.

### Operational changes during resume

Device assignment, owner concurrency, CPU/I/O budget, scratch locator,
endpoint locator, and source-snapshot implementation closure are classified as
execution-epoch fields. A compatible resume records a new file under
`execution_attestations/execution_epochs/`; resource changes do not themselves
change scientific component identity.

Use a new deployment-profile path when its bytes change. Keep the same durable
root only when the workflow accepts the change as a compatible execution epoch.
If it rejects the request, do not weaken the check: use a new durable request
and adopt/import only explicitly compatible checkpoints.

### Source-code corrections

Never overwrite the old immutable source snapshot.

For a scientifically equivalent orchestration, authentication, scheduling, or
adapter correction whose declared phase producer versions remain compatible:

1. commit and deploy the correction;
2. create a new source snapshot at a new path;
3. preserve the durable and scratch roots;
4. keep the scientific profile unchanged; and
5. rerun with `--resume`.

The workflow either accepts the new implementation as an execution epoch or
fails closed. A code change that can alter scientific contents, schemas, or
coverage must bump the affected producer identity. Such a change requires a
new request or explicit adoption of only the checkpoints whose own
compatibility proves reuse is valid.

### Reuse across a new durable request

Portable phase checkpoints live under:

```text
$RUN/portable_checkpoints/<phase>/artifact_manifest.json
```

The entrypoint accepts repeatable `--adopt-checkpoint` arguments. Adoption
authenticates the artifact and its required ancestor DAG before substituting a
phase. Granular Stage 1 and Stage 2 nodes have separate indexes under
`portable_granular_checkpoints/`.

An embedding cache is path-bound to its prepared cohort. Moving it requires
the authenticated relocation route with all three arguments:

```text
--embedding-cache-import
--embedding-cache-import-source-prepared
--embedding-cache-import-source-preparation-manifest
```

Use `scripts/resolve_production_embedding_cache_import.py --run-root OLD_RUN`
to resolve those inputs from a preserved run. Never copy a cache and edit its
manifest paths manually.

For a truly new dataset, old preparation, embeddings, preflight, and Stage 1
components are normally incompatible. Sharing a scratch parent is not
permission to reuse them; content-specific keys and ordinary authentication
are the authority.

## Completion and downstream safeguards

A successful full run has:

- `workflow_progress.json` with `status: "complete"` and no error;
- a `complete_manifest.json` for every phase in its configured sequence;
- `frozen_predictions.parquet` and `immutable_run_manifest.json` in the Stage
  2 inference phase;
- one held-out prediction per input unit and no duplicate unit coverage;
- one strict causal-forest fit per configured outer fold;
- a complete Stage 1 handoff and Stage 2 granular checkpoint DAG; and
- a fresh `terminal_validation` report.

Locate the final files through registered manifests rather than assuming an
attempt timestamp:

```bash
find "$RUN/phases/stage2_inference" -type f \
  \( -name frozen_predictions.parquet \
     -o -name immutable_run_manifest.json \) -print

find "$RUN/phases/terminal_validation" \
  -type f -name validation.json -print
```

For a synthetic run, oracle evaluation may open its separately configured
source only after the frozen-prediction checkpoint has been published and
authenticated. A real cohort normally has no oracle phase payload. The
terminal validator checks this ordering.

Archive together:

- the exact input cohort or its governed immutable source;
- the committed revision and source snapshot;
- scientific, deployment, Stage 1, and neural-query profiles;
- materialized model trees or their exact immutable identities;
- the complete durable root;
- the scratch component and reusable-preflight stores until no further resume
  is required;
- endpoint/model and environment identity records, excluding secrets;
- operator logs; and
- the terminal validation report.

## Failure triage

1. Read `workflow_progress.json`; its `current_phase` and `error` are the first
   authority.
2. Read the operator traceback and the phase's latest attempt tree.
3. Check whether `phases/<phase>/complete_manifest.json` exists. If it does,
   that phase sealed; if it does not, the phase did not complete even if many
   subcomponents did.
4. For Stage 1, inspect `recovery/stage1_scope_progress.json` and component
   `execution_manifest.json` files before assuming all model work was lost.
5. For Stage 2, inspect the canary report, response caches, extraction ledgers,
   fold manifests, and endpoint logs.
6. Preserve both roots and rerun with `--resume` after correcting the actual
   operational or implementation fault.

Warnings from sklearn or tokenizers are not necessarily the failure. The last
structured `workflow_progress` error and final traceback identify the terminal
exception.

Never respond to an authentication failure by hand-editing hashes, terminal
manifests, row identities, or producer versions. Determine whether the cause is
a changed file, a path relocation, an incomplete attempt, an incompatible
scientific change, or an implementation bug, then use the corresponding
resume, relocation, adoption, or corrected-source path.

## Related specifications

- [Production all-evidence quickstart](production_all_evidence_quickstart.md):
  concrete eight-GPU benchmark launchers and cloud operations.
- [Production Stage 1 bundle runbook](production_stage1_bundle_runbook.md):
  detailed Stage 1 inputs, evidence families, and bundle contracts.
- [Exact-inner production contract](stage1_exact_inner_production_contract.md):
  nested fitting and fold-honesty rules.
- [Stage 1 hierarchy handoff contract](production_stage1_hierarchy_handoff_contract.md):
  authenticated Stage 1-to-Stage 2 boundary.
- [All-evidence discovery interfaces](all_evidence_discovery_interfaces.md):
  model-facing evidence, hierarchical review, and transport contracts.
- [Reproducibility and recovery runbook](hierarchical_all_evidence_reproducibility_runbook.md):
  lower-level preservation and replay background.
