# Continuation handoff: all-ten Stage 1 through hierarchical discovery and ITEs

Status captured: 2026-07-21 19:34 EDT

This is the primary continuation document for a fresh agent. Read it before
launching anything. Then read `production_stage1_bundle_runbook.md`,
`production_stage1_hierarchy_handoff_contract.md`, and
`all_evidence_discovery_interfaces.md` in this directory. Conversation history
is not required to resume safely.

## 1. Executive status

The production-shaped arbitrary-cohort implementation is largely built, but no
end-to-end run has completed and no new ITE estimates exist.

The immediate state is:

- The repository is at commit
  `477808b97ae8c318b793338198715b0fe6b8ebf2` (`main`, commit subject
  `codex agent handoff 7-21-26`). The working tree was clean before this handoff
  file was added.
- The arbitrary-cohort Stage 1 wrapper exists at
  `scripts/build_all_evidence_stage1_bundle.py` and is wired to all ten required
  Stage 1 architecture families.
- The frozen balanced-200 engineering fixture and its complete frozen embedding
  cache exist and pass preflight.
- Several genuine all-ten Stage 1 attempts exposed and led to fixes for real
  replay/lineage bugs. Those failed outputs were preserved separately.
- The latest retry launched successfully on GPU 1, completed the first full-outer
  Stage 1 scope, then disappeared abruptly while beginning
  `outer_001_inner_001`. There is no retained traceback or exit status. It must
  be classified as an interrupted/unknown run, not a diagnosed new modeling
  defect.
- At handoff time there is no active Stage 1 process and no active GPU compute
  process. The current `stage1_bundle` directory is incomplete and unsealed: 7
  files, 5,652,120 bytes, and no `bundle_manifest.json`.
- Because there is no sealed Stage 1 bundle, the generic endpoint canary has not
  run, the new hierarchical discovery wrapper has not contacted Camus or any
  other endpoint, extraction and final causal-forest fitting have not run, and
  there are no ITEs from this new path.

The next safe action is not more design work. Preserve the incomplete target,
relaunch the byte-identical balanced-200 Stage 1 request into a fresh target with
persistent stdout/stderr logging, and actively monitor it to terminal exit.

## 2. User requirements that must not drift

### 2.1 All ten architectures, not a winner-take-all subset

The Stage 1 and discovery paths must incorporate all of these families:

1. `bow_nuisance`
2. `bow_r_loss`
3. `htr_neural`
4. `matched_pair_uplift`
5. `embedding_whole_cohort`
6. `embedding_clustered`
7. `tfidf_semantic_retrieval_contrasts`
8. `tfidf_topics`
9. `tfidf_orphan_ngrams`
10. `neural_query_moments`

Do not reinterpret "all of the above" as selecting the best architecture. The
families expose different views and all ten must enter the authenticated evidence
graph and final numerical path where applicable.

### 2.2 Hierarchical feature discovery is mandatory

A previous experiment that dumped raw evidence from every Stage 1 architecture
into one feature-discovery prompt did not work well. Do not restore that design.

The intended structure is:

1. Build a lossless authenticated atom catalog for all ten families.
2. Interpret one architecture at a time. If an architecture is too large, page
   it into complementary chunks whose union covers every atom and member.
3. Consolidate and coverage-criticize within that architecture to produce one
   compact architecture dossier.
4. Compare/integrate the ten compact dossiers across architectures.
5. During integration, rejection reconsideration, extraction-definition work,
   and adaptive review, look back to raw evidence by exact authenticated ID when
   needed.
6. Preserve every candidate in either a final concept or an explicit rejection
   ledger.

Context limits must be handled with lossless paging, recursive folding, and
hierarchical prompts. Do not sample, global-top-k, cap, semantically truncate, or
silently discard model output/evidence merely to fit a context window. The
current design's member-aware page size is an operational packing boundary, not
permission to omit support.

Direct row-level numerical signals are a separate non-grounding channel: they may
enter the final estimator but cannot, by themselves, justify a human-readable
feature name.

### 2.3 Fold honesty and TF-IDF calibration

- Outer held-out rows protect final ITE prediction.
- Exact-inner Stage 1 scopes use the one canonical joint treatment/outcome split
  registry.
- Topics and orphan n-grams perform model/term selection using nested
  training-only model/calibration partitions wholly inside the registered fit
  scope.
- Semantic-retrieval TF-IDF is deterministic, exhaustive, and nonselecting after
  its supervised fit-scope directions are frozen. Its training-only partitions
  are replay/stability canaries, not a held-out-label selector.
- Registered held-out treatment/outcome labels must never enter any of these
  fit/selection paths.
- Hierarchy review partitions, final estimator cross-fit folds, and nested
  TF-IDF calibration folds are distinct domains even when counts happen to be
  equal.

### 2.4 Endpoint and approval policy

The production wrapper must accept one explicit canonical HTTP(S)
OpenAI-compatible endpoint and one exact model name per invocation. The current
test profile is:

```text
endpoint: http://camus:8010/v1
model:    RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic
```

Those values are a run profile, not hard-coded generic defaults. Intentional
localhost endpoints are valid. Endpoint pools, failover URLs, model
autodiscovery/substitution, and silent fallbacks are not.

Every initial response and any bounded repair response must report the exact
requested model and `finish_reason=stop` before parsing, semantic validation, or
cache publication.

Production users must not approve artifact digests. Digests are internal
integrity/cache/resume identities. The one-shot wrapper carries its same-process
authorization internally. The older low-level benchmark runbook still documents
a manual approval ceremony; that is retained for historical operator tooling and
must not be exposed as the arbitrary-cohort production workflow.

A deployment/container/model-tree pin is not required execution authority.
`production_served_model_attestation.py` is optional static audit tooling only.

### 2.5 Final estimator and execution order

- Selector thinking budget: exactly 5,000 tokens.
- Extraction thinking: disabled.
- Proposal and extraction completion limit: 25,000 tokens.
- Transport retries for the intended production profile: zero.
- Semantic/schema repair: at most one bounded repair.
- Sparse-query fallback: forbidden.
- Final estimator: strict `CausalForestDML`; no structured-head or nonforest
  fallback.
- Synthetic oracle fields remain inaccessible until held-out predictions are
  frozen and hash-checked.

The user's rollout instruction is one arbitrary cohort end to end first, then five
additional cohorts sequentially. That means six cohort invocations, not one
five-fold job. The repository does not currently contain a six-cohort batch
orchestrator or an authoritative manifest enumerating those six dataset/config/
cache tuples. Do not guess the cohort list from directory names; obtain or record
the exact tuples before that rollout begins.

Separately, historical artifacts called "one-by-one" and "five-by-five" refer to
two synthetic benchmark variants: one-confounder/one-effect-modifier and
five-confounder/five-effect-modifier. Those names do not mean one cohort followed
by five cohorts and do not establish the user's intended six-cohort list.

## 3. What is implemented

### 3.1 Arbitrary-cohort Stage 1 wrapper

`scripts/build_all_evidence_stage1_bundle.py` is the single raw-cohort entry
point. It delegates to `ProductionStage1BundleBuilder` in
`oci/inference/production_stage1_bundle.py`.

It accepts:

- one Parquet cohort with configured stable unit ID, raw text, binary treatment,
  and binary outcome;
- one Stage 1 JSON/YAML configuration;
- either an existing authenticated four-file embedding cache or inputs to build
  a fresh local/offline cache atomically;
- a local HTR model tree;
- optional neural-query configuration;
- explicit CPU/GPU and worker settings; and
- a fresh output root.

It creates and authenticates the canonical outer/exact-inner split registry,
runs native full-outer and exact-inner fits, captures non-executable native proof
state, constructs cumulative-spent producers for the hierarchy epochs, and seals
the root graph only after all-ten coverage and byte identities validate.

This wrapper is aware of and uses the existing native Stage 1 implementations; it
does not replace them with ten toy reimplementations. The main integration
surfaces are:

- `stage1_exact_inner_family_adapters.py`
- `stage1_cumulative_spent_native_adapters.py`
- `stage1_cumulative_spent_embedding_adapters.py`
- `stage1_cumulative_spent_remaining_adapters.py`
- `bow_native_proof_capture.py`
- `htr_native_proof_capture.py`
- `matched_pair_native_proof_capture.py`
- `embedding_native_proof_capture.py`
- the native TF-IDF fitted-context/selection proof paths
- the neural-query trusted-array proof path

The terminal artifact is `bundle_manifest.json`. Nothing short of an exit-0,
loader-validated terminal manifest is a completed Stage 1 result.

### 3.2 Frozen embedding-cache producer

`oci/inference/production_embedding_cache_builder.py` implements an offline,
local-files-only cache build. It preserves exact raw text for embedding paths,
audits every ordered chunk with truncation disabled, requires the configured
chunk cap to be nonbinding, publishes atomically, and revalidates all rows.

The four cache files are:

```text
metadata.json
chunk_embeddings.npy
offsets.npy
chunk_texts.jsonl
```

### 3.3 Hierarchical discovery and one-shot wrapper

The generic production entry points are:

- `scripts/canary_production_stage1_hierarchy.py`
- `scripts/run_production_stage1_hierarchy_one_shot.py`

The old `scripts/canary_production_stage1_hierarchy_camus.py` is only a thin
compatibility entry point. New work should call the generic script.

The hierarchy implementation covers architecture-local interpretation,
within-family consolidation and coverage criticism, cross-family planning,
support-page review, recursive folds, rejection review, extraction definitions,
and adaptive reconsideration. Important implementation files include:

- `hierarchical_all_architecture_discovery.py`
- `hierarchical_discovery_compiler.py`
- `hierarchical_discovery_response_contract.py`
- `openai_compatible_json_discovery_job_runner.py`
- `adaptive_hierarchical_stage1_reconsideration.py`
- `lossless_stage1_evidence_catalog.py`
- `production_stage1_hierarchy_handoff.py`
- `production_stage1_hierarchy_loader.py`
- `production_stage1_hierarchy_one_shot.py`

The response contract uses exact identifier ownership, closed keyed coverage,
dynamic strict JSON schemas, at most one sanitized repair, and validated-only
immutable caching. Member-aware pages and recursive folds were introduced after
a 25,000-token truncation failure; they preserve complete support rather than
dropping it.

Important limitation: those losslessness guarantees currently cover Stage 1
evidence discovery/reconsideration, not the patient-note extraction compactor.
See the P0 extraction item in section 11.

### 3.4 Strict final causal forest

The production one-shot builds `AllEvidenceFusionRunnerConfig` with
`require_final_causal_forest=True` and supplies authenticated raw final-upstream
inputs. The runner uses `StrictOuterHonestFinalCausalForestAdapter`; the degraded
structured interaction head is not allowed on this path. Completion must verify
that every fold manifest reports the strict forest mode.

The locked environment can currently import EconML `CausalForestDML` (EconML
0.16.0, scikit-learn 1.6.1, NumPy 2.4.3). If the backend's internal
hyperparameter-tuning attempt fails, current code rebuilds the same
`CausalForestDML` class with configured parameters. This is not a nonforest
estimator fallback, but it should be reported distinctly in the audit if it is
ever exercised.

## 4. Current repository and source identity

Observed before adding this report:

```text
branch: main
HEAD:   477808b97ae8c318b793338198715b0fe6b8ebf2
tree:   clean
```

The latest interrupted request authenticated 160 source files. A fresh audit
found 0 missing or mismatched files relative to that request. Its identities are:

```text
request_sha256:       e2e2984ce752b3f893c691687fe1ca20ab1b0cf4013316963f4f2b37587b7fbb
behavior_identity:    547f15140a8b41204f1cf1dfedd6a779ae912f45590ab395d387830b8cb01ced
source_tree_sha256:   a524a802938de9e20fc346c4989408a1bba691ff77efa93379ed233c3a99ee96
```

Critical current file hashes:

```text
scripts/build_all_evidence_stage1_bundle.py
  c2795237812ad59ab69c46dc60cee25be6374b05652edcf8ff745e2a0bdd83d6
scripts/canary_production_stage1_hierarchy.py
  ffd37875896b2e9472f8fe3265206287e6ffc0e7c32ffa7357ebdbdc0c49acfd
scripts/run_production_stage1_hierarchy_one_shot.py
  c361af5d364d30cd41fe5665242cd0e6b1d41a4c708dc09b6738b18a187b4444
oci/inference/production_stage1_bundle.py
  4a0329e8d9dd4a5fa221c5eec24fbf6353de8476b73626fb962330024244a9a9
oci/inference/multi_model_forest_stage1.py
  b1f67dfff5832311d60a71f0e336a97a6580f5b52efa614db6af8fbd82352bd0
oci/inference/bow_native_proof_capture.py
  8eef3f554a63da26b797a1ed47e17b4a6ff4be88e088fd443fcb2e43989c59b3
oci/inference/htr_native_proof_capture.py
  0ba4aa10220dc371317494bc32f1f1ece2ed0ecf92cb0e977e9106dd333121ee
oci/inference/matched_pair_native_proof_capture.py
  6c53cd3260be7517ebb6ab848f2886c08c3ddc55a45b5b43d49b87e8e607aff6
oci/inference/embedding_native_proof_capture.py
  2a3ff1d8c6decb015f1a91960a0649a11d2445dd6db861f11052a54b11620e06
oci/inference/production_stage1_hierarchy_contract.py
  808ef87dfea9494e9d2669e7d9dc0d94046781709d62565cbe0f6a123ebb63b3
oci/inference/production_stage1_hierarchy_one_shot.py
  11e1812c09bb07c74e525b6f719fcd3bfbd348b79d81fee981732ed57b10dfde
oci/inference/hierarchical_all_architecture_discovery.py
  8895d594c72ceb6a9c9df59a72b2180286d385be18b6ec4f16fbf68947773db4
oci/inference/openai_compatible_json_discovery_job_runner.py
  f0dec595a9feb9406b5b12d8479a72bd57d6c6a897448016f87bb638dbd07172
```

Recompute these before resuming if HEAD or the working tree changes. A source
change intentionally creates a new Stage 1 request identity.

Portability warning: the balanced-200 cohort, embedding cache, interrupted Stage
1 directories, and other `artifacts/` contents are gitignored. They exist in this
workspace but are not contained in commit `477808b`. A fresh agent in this same
workspace can use them; a different clone cannot reconstruct this state from Git
alone and would need an explicit artifact transfer/checksum inventory.

## 5. Balanced-200 engineering fixture

Root:

```text
artifacts/production_stage1_e2e_fixture_balanced200_v1
```

This is a production-control-path fixture, not a scientific or performance
benchmark. It contains 200 rows, exactly 50 in each joint treatment/outcome cell,
two outer folds, and four exact-inner folds per outer fold. The minimum exact
inner fit scope is 75 rows. One malformed source row containing a pathological
25,458-character nonbreaking-hyphen token was deliberately excluded and replaced
within the same joint cell.

Key inputs and identities:

```text
cohort_balanced200.parquet
  sha256 cc575a322e40dbd542979295cc197eef69848def425638f0eb7b4f0d088c935c
stage1_config_balanced200.json
  sha256 f726cc58b4b79d7747c1dd43662a7e86a5c349f2b016f9a5ad35a99580fe21af
neural_query_config_balanced200.json
  sha256 e813258ecf0c0c002f3b05c48628a523b1417f60c5315a45113e7730ee6c852f
preflight_report.json (file bytes)
  sha256 4e4d5ad99c67312a19a2801c5bf585df4aa45c7edda645cccf1ecd38a146194a
preflight content_sha256
  eebc3dc35d8d50a2d97fa2626f22dd17760f8eed71e62292f91ca48e188a5bbb
canonical split registry sha256
  44b1c9b1afaefe3c8acfb33cd3ac5343cce371360dcb68dbc7ef1a01891e9f55
```

The preflight confirms all ten architectures remain enabled, no architecture was
removed for the fixture, HTR/embedding chunk caps are nonbinding, every chunk is
within its tokenizer/model limit, and all eight nested TF-IDF partitions have
both treatment arms and outcome classes.

### Frozen embedding cache

The cache contains 7,621 chunks by 4,096 float32 dimensions and is bound to all
200 ordered raw texts.

```text
chunk_embeddings.npy
  b71ddc1007303b42327e9822a2ce210f2967c297aba626b7d34d4ffb9570c3b4
chunk_texts.jsonl
  7df1e1eac798181d5ed7535bc007b8e360dcb9e6441527be02ee5348b5d70823
offsets.npy
  213dbc2e17f023f2a34dd04b7e4914f5de0065b5d42a5c54c57820bb62da2fa7
metadata.json
  3bd87b81f164619583dd6f304d91fb6b2e5ad475f729e8b3b00d2430b430c6ba
Qwen3 embedding model tree
  c905c538fb4ea49243eea098e68aa6f6d17a1e0c13c3e035c6b8521bde0caa53
cache configuration
  9df9e793e9adece7a5cc4920266e667715ba717a2c7b983758e6543a3a6ff226
production provenance
  75d7dc6a1dfdd9460adbad1f39ab9bf6eac8bad8d7f1720324f1800a52a2a533
```

The authenticated HTR `bert-tiny` model-tree SHA is
`fb7b1e91028d9543d0e603ec0069a173968d39317fcc60b79d3a96cd6b120f82`.

## 6. Stage 1 failure chronology and fixes

All failed artifacts are diagnostic evidence. Do not delete, edit, relabel, or
resume them.

### 6.1 Raw-text projection failure

```text
stage1_bundle.failed_raw_text_projection_20260721T131646
5 files; 230,710 bytes
```

Root cause: embedding cache/capture paths and legacy text-model paths had been
made to share a normalized text projection. The frozen embedding cache is bound
to exact raw text, while BoW/legacy normalization is a separate native behavior.

Fix: embedding cache, embedding capture, and embedding lineage now use exact raw
text. BoW and other native legacy paths retain their intended normalization.

### 6.2 GPU OOM while LM Studio occupied memory

```text
stage1_bundle.failed_gpu_oom_20260721T133037
5 files; 230,710 bytes
```

This was resource contention, not a Stage 1 architecture defect. A local LM
Studio process occupied the GPU during the attempt. The user stopped it. Later
runs crossed the previous approximately 33.3-GiB peak with substantial headroom.
Do not assume the local service is still running; inspect `nvidia-smi` each time.

### 6.3 BoW native replay dtype mismatch

```text
stage1_bundle.failed_bow_replay_20260721T171121
9 files; 108,314,097 bytes
```

The first full outer fit completed. The first exact-inner BoW replay differed by
about `2.24e-08`. Restoring the TF-IDF matrix as float32 while multiplying by
upcast Ridge coefficients produced float64 replay arithmetic that did not match
the native path.

Fix: reconstruct the native float32 arithmetic exactly. The tolerance was not
loosened. The preserved artifact's 72 fold states and 18 full-fit states were
replayed successfully after the fix.

### 6.4 Matched-pair HTR fold-coverage mismatch

```text
stage1_bundle.failed_matched_pair_fold_coverage_20260721T175232
11 files; 454,751,016 bytes
```

Root cause: `MultiModelForestStage1HTRProvider` forwarded an unrelated AVF fold
count (5) to the HTR matched-pair fitter while the production MMF effect-fold
contract required 3.

Fix: forward
`config.architecture.multi_model_forest.effect_folds`. Regression coverage
proved that native BoW and HTR branches use the intended fold IDs.

### 6.5 Embedding canonical-lineage validation

```text
stage1_bundle.failed_embedding_lineage_20260721T183557
27 files; 575,581,741 bytes
```

The run cleared the BoW and matched-pair failures, persisted HTR, matched-pair,
and embedding captures, then failed because the embedding lineage helper
required `_oci_row_id` as a stored column. Production `modeling_data`
intentionally stores only text, treatment, and outcome; `_oci_row_id` is a
derived positional namespace.

Fix: require the three real stored columns, validate positional bounds, and, if
an `_oci_row_id` column is supplied, still require exact equality with row
positions. Drifted supplied IDs remain a hard failure. The full embedding-proof
module passed 14/14, including mixed Unicode, raw-versus-normalized text, and
nonmonotonic row-order cases.

### 6.6 HTR proof raw-versus-normalized text binding

An offline registration replay of the preserved embedding-lineage artifact then
found that HTR prediction had run on raw `clinical_text`, while the HTR proof
scope was bound to normalized text. Raw text reproduced capture bit-exactly;
normalized text differed by up to `2.006292343e-4` on all 60 checked rows.

Fix: bind HTR proof lineage/replay to the actual raw text used by the runner and
add a Unicode regression. Preserved HTR replay then passed all 5 nuisance and 10
effect models (worst difference approximately `1.49e-7`), and preserved
matched-pair replay also passed.

### 6.7 Latest interrupted attempt: unknown process loss

Current incomplete target:

```text
artifacts/production_stage1_e2e_fixture_balanced200_v1/stage1_bundle
7 files; 5,652,120 bytes; no bundle_manifest.json
```

Timeline:

- 19:13:11: request/root files written; run began on GPU 1.
- The Python process was PID `3430575` under `uv` PID `3430571`.
- GPU 1 allocation reached approximately 6.4 GiB during model initialization.
- 19:25:49: `outer_001_full.json` finalized (5,374,251 bytes), with successful
  matched-pair BoW and HTR subproducer records.
- 19:25:50: `direct_numerical_outer_001.npz` finalized (47,159 bytes).
- 19:26:42: a temporary directory for `outer_001_inner_001` was active with HTR
  evidence subdirectories.
- The process then vanished. No process or GPU allocation remained.

There is no saved stdout/stderr log. The prior unified session is unavailable.
Kernel journal and cgroup checks showed no OOM-kill record, and the cgroup
`oom_kill` counter was zero. The surviving `TemporaryDirectory` strongly suggests
abrupt process loss; a normal Python exception unwinding its context would have
removed it. The exact cause is unknowable from retained evidence.

Do not call this another HTR bug, do not claim it completed, and do not use
`--resume`. The runbook forbids resuming interrupted legacy/neural partial
components.

## 7. Exact safe pickup procedure

### 7.1 Confirm quiescence and preserve the interrupted target

First verify twice that no matching process is alive, GPU usage is stable, and
the target mtimes are unchanged. Then move the current `stage1_bundle` to a
unique sibling such as:

```text
stage1_bundle.interrupted_unknown_exit_20260721T192642
```

Moving it is preferable to deletion because it preserves the only evidence from
the abrupt interruption. The new target path `stage1_bundle` must be absent
before launch. Do not pass `--resume`.

### 7.2 Rerun focused gates without starting another test marathon

At minimum, verify the working tree/HEAD, current critical hashes, and the native
proof/production-bundle tests affected by the last fixes. The last post-fix
focused gate was 44/44 and the preserved HTR/matched-pair replays passed. If
those bytes are unchanged, do not expand this into another broad speculative
test campaign before retrying the genuine E2E.

### 7.3 Relaunch with a persistent log and a maintained foreground session

The semantically exact command encoded by the latest request is:

```bash
uv run --frozen python -u scripts/build_all_evidence_stage1_bundle.py \
  --dataset /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/production_stage1_e2e_fixture_balanced200_v1/cohort_balanced200.parquet \
  --stage1-config /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/production_stage1_e2e_fixture_balanced200_v1/stage1_config_balanced200.json \
  --embedding-cache-dir /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/production_stage1_e2e_fixture_balanced200_v1/embedding_cache \
  --output-dir /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/production_stage1_e2e_fixture_balanced200_v1/stage1_bundle \
  --unit-id-column patient_id \
  --seed 42 \
  --device cuda:1 \
  --gpu-id 1 \
  --num-workers 1 \
  --tfidf-workers 1 \
  --tfidf-parallel-backend threads \
  --query-device cuda:1 \
  --query-nuisance-folds 3 \
  --query-config /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/production_stage1_e2e_fixture_balanced200_v1/neural_query_config_balanced200.json
```

Run this through a shell with `set -o pipefail` and `2>&1 | tee` to a timestamped
log in a sibling `run_logs` directory outside the bundle. Keep Python unbuffered.
Keep the launching unified execution session and its owning agent alive until a
terminal exit is observed. Do not launch it in a subagent that immediately marks
itself complete, and do not rely on a detached sandbox child whose parent may be
destroyed.

Monitor all three signals:

- session stdout/stderr;
- `nvidia-smi` plus the exact process PID; and
- artifact count/mtime progression.

GPU inactivity alone is not success. Success requires exit code 0 and a valid
terminal manifest.

### 7.4 Terminal Stage 1 acceptance checks

Require all of the following:

- `bundle_manifest.json` exists and its declared bundle hash validates;
- the authenticated production hierarchy loader accepts the bundle;
- 2 outer full scopes exist;
- 8 exact-inner scopes exist (2 outer by 4 inner);
- all 80 exact-inner family bindings exist (8 scopes by 10 families);
- every full/exact scope has nonzero atom and member coverage for all ten
  families;
- the 2 cumulative hierarchy scopes and all 20 cumulative native family proofs
  exist and reload;
- raw sidecars, direct numerical manifests, model/source descriptors, HTR tree,
  and embedding cache all rehash correctly;
- no oracle field entered Stage 1; and
- no partial component was silently reused.

## 8. Work after Stage 1 completes

### 8.1 Fix or explicitly close the one-shot retry/default discrepancy

This is a newly recorded pre-hierarchy blocker. The canary hard-enforces the
intended policy:

```text
transport retries = 0
schema repairs     = 1
```

However, the current production one-shot dataclass/parser defaults are:

```text
proposal_schema_repair_attempts = 2
request_max_retries             = 3
```

and the example command in `production_stage1_bundle_runbook.md` omits both
flags. This conflicts with the user's fixed production profile. Before the full
one-shot run, change the production defaults and runbook to 1 and 0, add a test
that the ordinary documented invocation has those exact values, and re-run the
one-shot/security/canary focused suites. Passing explicit
`--proposal-schema-repair-attempts 1 --request-max-retries 0` is a temporary
execution safeguard, not a reason to leave unsafe production defaults.

Any such source change will intentionally invalidate earlier prepared packets;
that is acceptable because no current arbitrary-cohort packet has been executed.

### 8.2 Run the generic one-job endpoint canary

Once Stage 1 is sealed, use fresh, absolute, pairwise nonnested paths:

```bash
uv run --frozen python scripts/canary_production_stage1_hierarchy.py \
  --bundle-manifest /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/production_stage1_e2e_fixture_balanced200_v1/stage1_bundle/bundle_manifest.json \
  --scratch-output-dir /absolute/fresh/canary_scratch \
  --hierarchical-preparation-dir /absolute/fresh/canary_preparation \
  --report-dir /absolute/fresh/canary_report \
  --endpoint http://camus:8010/v1 \
  --model RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic \
  --review-rounds 1 \
  --interaction-inner-folds 3 \
  --tfidf-nested-calibration-folds 3 \
  --review-stage1-device cuda:1 \
  --review-neural-query-device cuda:1
```

The canary loads the real bundle, prepares the ordinary hierarchy offline,
chooses the deterministic smallest real architecture-pure interpretation job,
and makes only that logical request plus at most its one repair. It is an
operational check, not an approval token or Camus-specific pin.

If Camus is unavailable, the user can restart it. Do not add endpoint fallback.
If a different endpoint/model is desired, pass that one exact pair explicitly and
let all runtime identity checks bind it.

### 8.3 Run the balanced-200 one-shot hierarchy

After a successful canary and after closing the retry/default discrepancy:

```bash
uv run --frozen python scripts/run_production_stage1_hierarchy_one_shot.py \
  --bundle-manifest /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/production_stage1_e2e_fixture_balanced200_v1/stage1_bundle/bundle_manifest.json \
  --output-dir /absolute/fresh/hierarchy_execution \
  --hierarchical-preparation-dir /absolute/fresh/hierarchy_preparation \
  --attestation-dir /absolute/fresh/hierarchy_execution_record \
  --endpoint http://camus:8010/v1 \
  --model RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic \
  --review-rounds 1 \
  --interaction-inner-folds 3 \
  --tfidf-nested-calibration-folds 3 \
  --review-stage1-device cuda:1 \
  --review-neural-query-device cuda:1 \
  --proposal-schema-repair-attempts 1 \
  --request-max-retries 0
```

The name `--attestation-dir` here means a non-authorizing run-result audit record;
it is not the optional container/deployment pin machinery.

Verify terminal status, exact all-row prediction coverage, per-fold immutable
manifests, strict causal-forest audit, and absence of oracle columns before any
post-hoc synthetic evaluation.

### 8.4 Historical synthetic variants and the one-plus-five rollout

The two source cohorts are:

```text
synthetic_data/example_synthetic_datasets/
  one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet

synthetic_data/example_synthetic_datasets/
  five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet
```

Do not reuse the old historical handoffs as if they were outputs from the new
arbitrary-cohort wrapper. If these two benchmark variants are selected as rollout
cohorts, build/validate each full embedding cache, run the new all-ten Stage 1
wrapper, run its generic canary, then run the new one-shot hierarchy to frozen
ITEs. Finish and verify one before starting the next.

These two historical variants are not a substitute for the user's requested
one-then-five cohort manifest. Before the six-cohort rollout, explicitly record
all six dataset, unit-ID, Stage 1 config, neural-query config, local model/cache,
device, endpoint, and model tuples. The current wrapper runs one cohort per
invocation; there is no verified batch driver.

Only after predictions are immutable may the synthetic `true_ite_prob` and
oracle feature definitions be read for correlation/recovery evaluation.

## 9. Historical v3/v4/v5 packet disposition

Do not resume the old v3, v4, or v5 execution roots.

- v3 reached the first architecture-local job but failed on duplicate JSON keys
  and then an unsupplied evidence ID. It lacked the promised bounded repair.
- v4 added one repair but both initial and repaired responses cited placeholder
  identifiers copied from static examples. The packets were retired.
- v5 added dynamic response schemas and member-aware chunking. Its one-by-one
  execution still exhausted its repair because the model duplicated a member
  disposition. The preserved failure record is:

```text
artifacts/all_evidence_fusion/
  hierarchical_all_arch_one_20260720_v5_control/live_failure_record.json
```

It reports `DiscoveryResponseRepairExhausted`, source family `bow_nuisance`, no
validated cache entries, no predictions, no final manifest, and no oracle read.
The five-by-five v5 execution never started.

Current code has since moved to closed keyed exact coverage and further source
binding. Old packets are path/source-bound historical diagnostics; they are not
authorization or cache inputs for the new arbitrary-cohort one-shot path.

## 10. Test evidence and what it does not prove

Verified checkpoints during this work include:

- Fresh current-tree native-proof/TF-IDF/Stage 1 bundle gate:
  `uv run --frozen pytest -q tests/test_embedding_native_proof_capture.py
  tests/test_bow_native_proof_capture.py tests/test_htr_native_proof_capture.py
  tests/test_matched_pair_native_proof_capture.py
  tests/test_tfidf_nested_calibration_production.py
  tests/test_production_stage1_bundle.py` -> 83 passed, 369 warnings, 188.47s.
- Fresh current-tree hierarchy contract/loader/one-shot/security/canary/JSON-runner
  gate: `uv run --frozen pytest -q
  tests/test_production_stage1_hierarchy_contract.py
  tests/test_production_stage1_hierarchy_loader.py
  tests/test_production_stage1_hierarchy_one_shot.py
  tests/test_production_stage1_hierarchy_one_shot_security_audit.py
  tests/test_canary_production_stage1_hierarchy.py
  tests/test_openai_compatible_json_discovery_job_runner.py` -> 177 passed,
  234.68s.
- Earlier combined generic production offline regression: 281/281 passed.
- Relevant embedding/cache/raw-projection regression checkpoint: 171 passed.
- BoW native proof plus adjacent combined checkpoint: 193 passed.
- Full embedding native-proof module after the positional-lineage fix: 14/14.
- Focused post-HTR-fix prelaunch suite: 44/44.
- Preserved real-artifact HTR replay: 5 nuisance plus 10 effect models passed.
- Preserved matched-pair replay passed.
- All 160 source files in the latest request matched current bytes at handoff.

The exact command behind the earlier reported 44/44 focused prelaunch gate is not
recoverable from retained transcripts. The two fresh commands above are the
reproducible current-tree evidence and supersede it for pickup purposes.

These are strong unit/integration and artifact-replay checks. They do not replace
the still-missing genuine all-ten Stage 1 terminal manifest or the missing
hierarchy/ITE E2E.

The runbooks mention other earlier suite totals (for example cache 57/57, bundle
39/39, hierarchy loader/one-shot/security 100/100, and a 495-pass hierarchy-era
checkpoint). Treat those as historical evidence tied to their then-current bytes,
not as a fresh test result after any new edit.

## 11. Known limitations and remaining implementation items

### P0: finish the balanced-200 all-ten Stage 1 run

This is the immediate blocker for every downstream action.

### P0: production one-shot retry defaults conflict with the fixed profile

Set and test ordinary defaults of one schema repair and zero transport retries,
as described above, before contacting the full endpoint workflow.

### P0: patient-level extraction still truncates/compacts note input

The architecture evidence hierarchy is lossless, but the current full one-shot
explicit-extraction path is not. It defaults to
`extraction_context_strategy="contract_lexical_rag"` and
`extraction_max_text_length=14_000`. `compact_contract_lexical_context()` ranks
contract-derived excerpts, hard-caps the rendered note context, and may shrink
the final excerpt to fit. A note longer than that cap is therefore not
exhaustively processed.

This violates the user's instruction to handle agentic context limits with
chunked/hierarchical prompts rather than semantic truncation if that instruction
applies across the whole workflow, as it should for production. Fix this before
the full hierarchy/ITE run:

1. losslessly page each patient's entire note for each extraction definition;
2. extract against every page;
3. reconcile all page results through a bounded hierarchical fold with complete
   page coverage;
4. bind the page plan, coverage, prompts, responses, and fold result into the
   extraction cache/manifest; and
5. add tests proving no character/token region is omitted and no page result is
   silently truncated.

Merely raising `extraction_max_text_length` is not a robust fix.

### P1: authenticated endpoint credentials are not yet a safe wrapper feature

The generic runner class can accept an API key, but the production canary/one-shot
path currently hardcodes `EMPTY` and exposes no safe environment-variable
credential option. Therefore the current production wrapper supports
unauthenticated OpenAI-compatible endpoints such as the present Camus profile.

Do not put a real key into Stage 1 JSON or CLI arguments. Before using an
authenticated endpoint, add an environment-variable reference whose secret value
is never serialized, hashed into user-visible artifacts, logged, or copied into
effective config snapshots; audit both proposal/review and extraction configs for
secret persistence.

This limitation does not make the endpoint Camus-specific. Any unauthenticated
compatible endpoint/model pair can be supplied now.

### P1: no genuine arbitrary-cohort E2E certification yet

The runbooks correctly state `production_hierarchy_execution_ready=false` until
a genuine sealed Stage 1 bundle, generic canary, one-shot hierarchy, strict forest,
and independent artifact review complete.

Concretely,
`production_stage1_hierarchy_handoff.py` currently fixes
`GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY=False`, and the one-shot result
therefore reports `genuine_one_shot_e2e_certified=false` even if it successfully
writes ITE predictions. Do not flip that constant merely because a run exits 0.
Complete an independent read-only audit of the genuine E2E artifacts, record its
acceptance criteria/results, and make any later certification change deliberately
with focused tests.

### P2: environment bytes are version-bound, not fully content-addressed

The request records installed distribution names/versions and source bytes, but
not every installed Python/native dependency byte. Preserve the exact environment
record alongside successful outputs.

## 12. Definition of completion

Engineering control-path completion requires:

- balanced-200 all-ten Stage 1 exits 0 with a validated terminal bundle;
- generic endpoint canary succeeds with exact model and `finish_reason=stop`;
- one-shot hierarchy interprets every architecture separately and integrates all
  ten without semantic truncation;
- extraction and adaptive review complete under the frozen proposal boundary;
- strict `CausalForestDML` produces exactly one held-out ITE per row;
- immutable run/fold manifests and prediction hashes validate; and
- an independent read-only audit confirms no oracle leakage or fallback.

Scientific benchmark completion additionally requires the same full sequence for
every explicitly selected cohort, run sequentially, followed only afterward by
oracle correlation and feature-recovery reporting for synthetic cohorts.

## 13. Fresh-agent first-hour checklist

1. Read this report and the three primary docs named at the top.
2. Confirm HEAD/working tree and recompute critical hashes.
3. Confirm no Stage 1 PID/GPU workload and stable incomplete-target mtimes.
4. Preserve the incomplete `stage1_bundle` under a unique interrupted name.
5. Re-run the focused 44-test/native-proof gate if bytes changed or evidence is
   uncertain.
6. Relaunch the exact Stage 1 command with `python -u`, `pipefail`, persistent
   `tee` logging, and a maintained foreground session.
7. Monitor until a real terminal exit; never infer success from GPU disappearance.
8. On failure, preserve the target and exact log before diagnosing one first
   error. On success, validate the terminal bundle before any remote call.
9. Fix/test the one-shot 1-repair/0-retry defaults and replace patient-note
   extraction compaction with lossless chunk/fold processing.
10. Run the generic single-job canary, then balanced-200 one-shot, then the
    explicitly enumerated one-plus-five cohorts sequentially.
