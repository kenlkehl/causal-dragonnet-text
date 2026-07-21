# Hierarchical All-Evidence Reproducibility and Recovery Runbook

Status: preservation, exact replay, and context-independent recovery guide for the
hierarchical all-evidence pipeline.

Use this after the operational procedure in
[`hierarchical_all_evidence_operations_runbook.md`](hierarchical_all_evidence_operations_runbook.md).

> The approval records discussed here belong to low-level exact replay and
> operator-controlled benchmark execution. They are internal artifacts when a
> cohort uses `scripts/build_all_evidence_stage1_bundle.py` followed by
> `scripts/run_production_stage1_hierarchy_one_shot.py`; neither production
> wrapper requests manual digest approval from its user.

## Reproducibility guarantees

Three distinct guarantees matter.

| Guarantee | Meaning |
|---|---|
| Audit reproducibility | A reviewer can determine the exact rows, evidence, prompts, definitions, model/provider identities, settings, cache lineage, and predictions used. This is the primary guarantee of the immutable manifests. |
| Exact cache replay | A previously validated hierarchical JSON response can be reused byte-for-byte only when the exact job, runner identity, hierarchy precommit, validator implementation, cache identity, and response all authenticate. The response is semantically revalidated on replay. |
| Independent fresh rerun | The same algorithm and configuration can be run again on the same inputs. A fresh remote language-model call or GPU fit is not promised to be bit-identical, even with temperature zero and fixed seeds. It must create a new prepared batch and audit trail unless it is an exact approved replay. |

Conversation compaction affects none of these. Chat history, a continuation summary,
and directory modification times are non-authoritative.

## The minimum run record

Create a control-record directory outside the immutable preparation and execution
directories. Preserve these items there:

- the exact command or Bash argument array used for dry run, preparation, and live
  execution, with secrets redacted;
- the complete terminal JSON from all three phases;
- the approved batch SHA-256 and the exact human approval record;
- the exact `authoritative_execution_replay_arguments` array;
- repository commit ID and a binary patch or source snapshot for every uncommitted
  change that influenced the run;
- `pyproject.toml`, `uv.lock`, Python version, installed package inventory, CUDA and
  driver information, and GPU model;
- remote endpoint identity, exact served model identifier, model-weight revision or
  digest when available, and server/container build identity;
- paths and SHA-256 values for every upstream input and authenticated overlay;
- start/end timestamps and the host/device assignment;
- the complete preparation and final execution directories.

Never put API keys, access tokens, or other secrets into the archived command or
environment record.

For the arbitrary-cohort production path, preserve the exact builder and
one-shot invocations, the complete sealed Stage 1 bundle, the hierarchy
preparation/output/execution-record roots, and the separate generic runtime
canary report. Preserve the exact single endpoint and model supplied to both
commands and the canary's per-response model and finish-reason metadata. There
is no served-deployment identity input, compiled deployment pin, human approval
record, or caller replay argument for that path. Optional model revision,
server version, container, or deployment-certificate metadata may be archived
when available, but it is informational and cannot authorize or block a run.

The immutable manifests record much of this automatically, but they do not replace
the operator's exact invocation record or the external model/container inventory.

## Artifacts that must be retained

### Source and environment

- the exact repository source used by the run;
- `pyproject.toml` and `uv.lock`;
- local model code and any uncommitted patch;
- the HTR model tree and all referenced model weights;
- the frozen embedding-model weights or an immutable model revision;
- the remote inference server build and served model revision.

### Upstream data bundle

- the complete sealed arbitrary-cohort Stage 1 bundle, including
  `bundle_manifest.json`, `immutable_build_request.json`, the exact-inner and
  cumulative all-ten root indexes, every component directory, catalog, proof,
  descriptor, and raw sidecar;
- the dataset Parquet;
- legacy all-source discovery handoff and its manifest;
- resealed TF-IDF handoff, registry, manifest, and preflight/result audit;
- primary predictions/split file;
- historical Stage 1 configuration snapshot;
- the complete frozen embedding cache;
- any registered orphan-ngram or neural-query artifacts;
- every extraction, spent-evidence, or context-fit read-only cache index and every
  file to which that index refers;
- both byte-exact prompt-control files used during preparation.

### Preparation bundle

Retain the entire approved preparation directory, including:

```text
immutable_hierarchical_input_manifest.json
approved_hierarchical_batch_precommit.json
context_fit_overlay_companions/
first_gate_materialization_intent_indexes/
offline_review_packet/
outer_fold_NNN/role_neutral_evidence_catalog.json
outer_fold_NNN/architecture_chunk_plan.json
outer_fold_NNN/first_gate_materialization_intent.json
outer_fold_NNN/approved_hierarchical_wrapper_precommit.json
outer_fold_NNN/immutable_fold_preparation.json
hierarchical_job_cache/             # once live JSON jobs have run
```

Also retain every source cache path named by an authenticated overlay in the input
manifest or preparation result. A copied cache in a different directory is not
automatically equivalent.

### Final execution bundle

Retain the entire final execution directory, especially:

```text
immutable_input_manifest.json
outer_fold_NNN/immutable_fold_manifest.json
outer_fold_NNN/frozen_predictions.parquet
frozen_predictions.parquet
immutable_run_manifest.json
current_extraction_cache/
post_extraction_review_spent_evidence_cache/
post_extraction_review_gate_cache/
post_extraction_review_neural_query_cache/
final_context_fit_upstream_cache/
posthoc_oracle_evaluation/          # when present
```

Do not retain only `frozen_predictions.parquet`. Without the parent manifests,
fold artifacts, registries, and upstream identities, it is a result without a
reproducible derivation.

For the no-approval production path, also retain the entire separate
execution-record root and its `production_stage1_hierarchy_one_shot_result.json`,
plus `production_stage1_hierarchy_runtime_canary.json`. Together they record the
bundle, provider, exact endpoint/model runner identity, authenticated response
metadata, batch result, run manifest, fold manifests, and prediction hash. The
canary report is not executable authority.

## Path identity and portability

Several authenticated identities deliberately include absolute paths, including
the hierarchy job-cache root and some provider/cache envelopes. Therefore:

- preserving identical bytes at a new path does not guarantee that an old approval
  or cache identity remains valid;
- for an exact approved replay, keep the original paths and directory relationships;
- after moving to another machine or filesystem layout, expect the low-level
  benchmark path to perform a new dry run, preparation, packet review, and
  approval; the production path instead performs a fresh no-approval one-shot
  invocation under new absolute roots;
- never edit a manifest to replace old paths. That destroys its authenticated
  content identity.

The pipeline remains procedurally reproducible after relocation, but an old
path-bound approval is intentionally not portable by assertion.

## Build a preservation inventory

Choose one archive root containing copies or filesystem snapshots of the source,
upstream bundle, preparation bundle, execution bundle, and control record. Generate
a checksum inventory from inside that root:

```bash
export AE_ARCHIVE_ROOT=/absolute/path/to/preserved_run_bundle
export AE_ARCHIVE_CONTROL=/absolute/path/to/preserved_run_bundle/control

mkdir -p "$AE_ARCHIVE_CONTROL"
cd "$AE_ARCHIVE_ROOT"
find source upstream preparation execution control \
  -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$AE_ARCHIVE_CONTROL/SHA256SUMS"
```

If a cache or model tree is too large to duplicate, use a read-only filesystem
snapshot and include its snapshot identifier plus a checksum inventory. A path-only
reference is insufficient.

Verify later with:

```bash
cd "$AE_ARCHIVE_ROOT"
sha256sum -c control/SHA256SUMS
```

Store the checksum file independently as well, so corruption of the archive cannot
silently replace both an artifact and its only digest.

## Recovery after interruption

### Preparation failed

Preserve the failed scratch and preparation directories for diagnosis. Do not
delete, overwrite, or bless partial files as inputs. Correct the fault and prepare
a new batch under fresh target paths. Expensive prior computation may be reused
only through a cache format and authenticated registration explicitly accepted by
the production CLI.

### Live hierarchical discovery failed

Successful remote JSON jobs are written to the immutable job cache under the
approved preparation directory only after semantic validation. To continue:

1. preserve the failed execution directory;
2. keep the approved preparation and hierarchy job-cache root unchanged;
3. choose a new nonexistent final execution directory and fresh output-local
   neural-query cache;
4. use the same approval digest, code, endpoint/model identity, common arguments,
   and authoritative replay registrations;
5. rerun the live command from the operations runbook.

The runner will accept validated cache hits and submit only genuine misses. Never
copy an entry into the job cache manually.

### Extraction or adaptive review failed

The final output still cannot be resumed in place. A new live invocation needs a
fresh output directory. Reuse downstream work only if a closed artifact has been
exported and can be registered through one of:

```text
--read-only-cache-index
--read-only-review-spent-evidence-cache PATH::SHA256
--read-only-context-fit-cache-index INDEX_PATH::SHA256
```

Registering such an artifact changes provider identity unless it was already bound
into the approved preparation. If the approved batch did not include it, prepare
and approve a new batch rather than trying to force the old digest.

### Code, model, or prompt changed

Any scientific or identity-bearing change requires a new run:

1. use fresh scratch, preparation, and execution paths;
2. run side-effect-free dry validation;
3. prepare a new immutable packet;
4. inspect and explicitly approve its new digest;
5. execute and preserve it as a separate result.

Do not overwrite the prior run or reuse its approval digest.

## Exact replay checklist

Before attempting to reuse an existing approved batch, verify all of the following:

- [ ] The dataset, both handoffs, primary splits, Stage 1 config, frozen embedding
      cache, prompt controls, and registered overlays still match their declared
      hashes.
- [ ] The repository source and runtime package identities match the preparation
      manifest.
- [ ] The absolute preparation and hierarchy cache paths are unchanged.
- [ ] The endpoint and exact model identity match the approved runner identity.
- [ ] The live scientific arguments match `effective_runner_config` and
      `hierarchical_discovery_config` in the preparation manifest.
- [ ] The batch approval SHA-256 is the explicitly approved value, not an offline
      packet hash or a fold wrapper hash.
- [ ] The authoritative replay arguments are byte-for-byte the approved ones.
- [ ] The new final output and its neural-query cache are nonexistent or empty.
- [ ] Oracle evaluation is post-hoc only and is requested only for a synthetic
      dataset.

If any item fails, create a new prepared batch.

## Completed-run verification checklist

- [ ] Terminal status is `completed`.
- [ ] `immutable_run_manifest.json` exists and names every fold manifest.
- [ ] Every fold has an immutable manifest and frozen prediction file.
- [ ] Combined predictions cover every dataset row exactly once.
- [ ] Prediction SHA-256 agrees across the terminal result, run manifest, and file.
- [ ] Prediction columns contain no oracle values.
- [ ] The final estimator is the required causal-forest backend with no nonforest
      fallback.
- [ ] The complete input, preparation, execution, cache, source, environment, and
      model records are preserved.
- [ ] Synthetic oracle results, if any, live only in the separate post-hoc directory.
- [ ] The checksum inventory verifies successfully.

## Context-compaction recovery protocol

After a compacted conversation, a new operator or agent should not reconstruct
state from memory. Use this order:

1. Read this runbook and the operations runbook.
2. Identify the intended preparation directory by explicit path, never by newest
   timestamp or version-like directory name.
3. Read `immutable_hierarchical_input_manifest.json` and
   `approved_hierarchical_batch_precommit.json`.
4. Read the preserved `prepare_result.json`, exact invocation record, replay
   arguments, and human approval record.
5. Determine whether a final execution completed by checking for a valid
   `immutable_run_manifest.json`, not by inspecting chat summaries.
6. If execution must continue, use a fresh output and follow the interruption
   matrix in the operations runbook.
7. If any identity is ambiguous or missing, stop and prepare a new batch rather
   than guessing.

This makes conversation context disposable: the repository, immutable manifests,
approved digest, and preserved artifacts fully describe the authorized workflow.
