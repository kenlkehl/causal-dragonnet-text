# All-evidence workflow

The researcher workflow is one command:

```bash
uv run python scripts/run_all_evidence.py \
  --config example_configs/research_all_evidence.json
```

It reads one cohort and writes everything to one output directory. With no
Stage 2 transport configured it runs Stage 1 through the plain handoff. With an
external endpoint or a pipeline-managed vLLM pool it continues through
fold-scoped feature definitions, patient-level extraction, training-fold
empirical review, and cross-fitted causal estimation. Interruption and resume
are automatic: run the same command again.

This is the repository's only all-evidence orchestration path. It uses ordinary
files and completion markers rather than a separate source snapshot, immutable
request, checkpoint-adoption system, trust policy, artifact-authentication
layer, or content-hash protocol.

The former `MultiModelForestRunner` and its TF-IDF-topic agentic Stage 2 were
retired. `multi_model_forest` remains the name of the Stage 1 scientific
configuration, not a second runnable workflow. The `oci run` command is
reserved for the retained explicit-feature workflows; configurations using
`multi_model_forest` stop with a migration message and must use this workflow.

## Configuration

Copy
[`example_configs/research_all_evidence.json`](../example_configs/research_all_evidence.json)
and edit it. It keeps the settings a researcher normally changes in one place:

```json
{
  "dataset": "/data/cohort.parquet",
  "output_dir": "/results/my_stage1_run",
  "columns": {
    "unit_id": "patient_id",
    "text": "clinical_text",
    "treatment": "treatment_indicator",
    "outcome": "outcome_indicator"
  },
  "science": {
    "clinical_question": "Which pretreatment text features confound or modify treatment effect?",
    "outcome_type": "binary",
    "outer_folds": 5,
    "inner_folds": 5,
    "seed": 42,
    "stage1": {},
    "neural_queries": {}
  },
  "models": {
    "htr": "/models/bert-tiny",
    "embeddings": "/models/qwen3-embedding-8b"
  },
  "stage2": {
    "endpoint": "",
    "model": ""
  },
  "run": {
    "devices": ["cuda:0", "cuda:1"],
    "workers": 16,
    "components": [
      "embedding_cache",
      "tfidf",
      "text_models",
      "neural_queries",
      "handoff"
    ]
  }
}
```

Paths in the file are relative to the config file. `science.stage1` accepts a
nested override of any setting in the established all-evidence Stage 1 model
template. `science.neural_queries` does the same for neural-query settings.
The fully expanded settings actually used are written to the output directory
as `resolved_stage1_model_config.json` and
`resolved_neural_query_config.json`. Secret-valued API-key fields are redacted
in these readable copies but remain available to the in-memory model config.

JSON and YAML config files are accepted. YAML requires PyYAML in the current
environment.

### Architecture selection

By default, Stage 1 retains the historical behavior: it runs every architecture
enabled by the resolved model configuration. To make an architecture subset a
first-class scientific choice, set `science.stage1_architectures` to a JSON list
or pass a comma-separated CLI value:

```bash
uv run python scripts/run_all_evidence.py \
  --config my_run.json \
  --architectures embedding_whole_cohort,tfidf_semantic_retrieval_contrasts
```

The registry resolves private prerequisites such as the embedding cache and
whole-cohort contrast computation, but only selected architecture envelopes are
written to a targeted handoff and admitted to Stage 2. A saved explicit
selection cannot be changed while resuming. Omit the setting to resume an older
run unchanged, or choose a fresh output directory for a different subset.

## Full, Stage-1-only, and Stage-2-only runs

Stage 2 reads `handoff/evidence.jsonl` and completes the remainder of the
analysis. It first uses the `all_evidence_fusion` scientific projections to
compile exact-deduplicated, fold-local semantic evidence cards. It reuses the
Stage 1 embedding cache by memory map where compatible and retains card/member/
raw-path lineage for audit. Within each outer fold, it interprets the compiled evidence,
defines patient-level variables, extracts those variables from the clinical
text, and evaluates their usefulness on the outer training rows. A bounded
model-assisted review may clarify a measurement definition and repeat the
training-row extraction. The definition is then frozen before it is applied to
the outer held-out rows. Finally, Stage 2 fits nuisance and effect-modification
models on the training rows and writes held-out AIPW scores and conditional
effect estimates. It is enabled by specifying `stage2.endpoint` or
`stage2.vllm`.

The only supported compiler is `semantic_cluster_cards_v2`. It checks each
outer fold for every architecture in the frozen Stage 1 selection (or the
resolved legacy enable flags) before making an interpretation request. Missing evidence fails
with a readable Stage 1 rerun instruction. This is an in-process set comparison,
not an artifact-authentication, byte-attestation, or deployment-gate system.
The former `raw_packets_v1` compatibility option is intentionally unsupported
because it combined scientifically distinct architectures.

For feature discovery, Python projects every compiled card to a prompt-local
integer `item` and a list containing only its deduplicated
`representative_evidence.text` strings. Card and packet IDs, evidence kind,
detail objects, truncation flags, axes, polarity, semantic grouping,
architectures, scores, support counts, folds, and other provenance stay outside
the model prompt. The model returns feature names, descriptions, rationales,
and `supporting_items` such as `[1, 3]`; Python immediately maps those ordinal
labels back to the original packets and derives evidence axes and causal roles
without model-authored IDs. Discovery does not choose value types, units,
categories, or extraction ontologies; the later one-feature ontology request
makes those decisions from the retained feature name and supporting text.

An external endpoint configuration is:

```json
{
  "stage2": {
    "endpoint": "http://127.0.0.1:8010/v1",
    "model": "Qwen/Qwen3-32B",
    "workers": 32,
    "request_timeout": 7200,
    "max_tokens": 50000,
    "repetition_penalty": 1.1,
    "interpretation_reasoning_effort": "high",
    "extraction_reasoning_effort": "none",
    "evidence_compiler": "semantic_cluster_cards_v2",
    "evidence_max_cards_per_fold": 400,
    "evidence_max_exemplars_per_card": 4,
    "evidence_max_exemplar_chars": 2400,
    "operationalization_max_prompt_chars": 640000,
    "consolidation_batch_size": 20,
    "consolidation_alphabetical_rounds": 5,
    "consolidation_max_rounds": 55,
    "max_review_rounds": 2,
    "ontology_refinement_min_failure_patients": 3,
    "max_ontology_refinement_rounds": 2,
    "screening_trees": 200,
    "stability_selection_rounds": 3,
    "stability_selection_frequency": 0.6666666667,
    "effect_modifier_negative_margin_fraction": 0.01,
    "effect_modifier_negative_fold_fraction": 0.6,
    "estimation_trees": 200,
    "explicit_features": [
      {
        "name": "age_at_treatment_decision",
        "description": "Age at the pretreatment decision point.",
        "value_type": "continuous",
        "categories_or_unit": ["years"],
        "measurement_definition": "Extract age in years at the treatment-decision point.",
        "missing_value_rule": "Return null when age cannot be determined from the pretreatment record.",
        "roles": ["confounder"]
      }
    ]
  }
}
```

`stage2.explicit_features` is optional. Each entry is a complete, investigator-
specified variable definition, not merely a requested name. It requires
`name`, `description`, `value_type`, `categories_or_unit`,
`measurement_definition`, `missing_value_rule`, and one or more `roles` from
`confounder`, `prognostic`, and `effect_modifier`. Closed ontologies are
validated just like model-authored definitions: binary variables need exactly
two values, and categorical or ordinal variables need at least two. A
continuous feature may use an empty list when it has no meaningful unit.
`stability_summary` and `caveats` are optional. For compatibility with the
standalone explicit-feature vocabulary, `type`, `categories`, and `unit` are
accepted aliases; the ontology fields may also be placed inside an `ontology`
object while `name` and `roles` remain alongside it.

Configured features enter the iterative alias-consolidation batches in every
outer fold. In every round they are hard invariants: they cannot be excluded,
two distinct configured feature names cannot be merged, and a merge containing
one must use its exact configured name as the output. When Stage 1 discovers an
alias, Python retains one consolidated feature, attaches the discovered packet
and architecture provenance, keeps the configured name and roles, and uses the
supplied ontology without making the one-feature ontology request. They still
undergo training-fold extraction and empirical diagnostics, but review must
keep them without revising the supplied ontology. If a required feature cannot
be extracted well enough for the workflow's health checks, the run fails
visibly rather than silently dropping or redefining it.

`stage2.model` is optional. If it is empty or omitted, Stage 2 queries the
OpenAI-compatible `/models` endpoint once at startup and uses the result when
exactly one model ID is advertised. If the server advertises multiple model
IDs, set `stage2.model` explicitly to avoid an ambiguous selection.

### Pipeline-managed vLLM replicas

`stage2.vllm` tells the same pipeline process to own the local vLLM server
lifecycle. In this mode `stage2.model` is required and `stage2.endpoint` must be
empty or omitted, and the workflow environment must include the `local-llm`
optional dependency group. For example:

```json
{
  "stage2": {
    "model": "google/gemma-4-31B-it",
    "workers": 32,
    "vllm": {
      "server_count": 8,
      "gpus": [
        "cuda:0", "cuda:1", "cuda:2", "cuda:3",
        "cuda:4", "cuda:5", "cuda:6", "cuda:7"
      ],
      "base_port": 8010,
      "download_dir": "/models/huggingface",
      "startup_timeout": 7200,
      "extra_args": [
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "65536"
      ]
    }
  }
}
```

The pipeline interprets the GPU values as logical CUDA indices. When its parent
environment has `CUDA_VISIBLE_DEVICES=4,6`, for example, managed logical GPUs 0
and 1 are mapped to physical selections 4 and 6 for the child processes. The
list is divided into equal, nonoverlapping groups in its supplied order and the
tensor-parallel size of each server is set to its group size. Thus eight
servers over eight GPUs produces eight one-GPU replicas; four servers over
eight GPUs produces four two-GPU replicas. Uneven division, duplicate devices,
or more servers than GPUs is rejected before launch.

The runner checks every requested port before launch, starts all replicas with
separate `CUDA_VISIBLE_DEVICES` values, and waits for each `/v1/models` API to
advertise a model. `stage2.workers` remains the total request concurrency; a
thread-safe round-robin router spreads those requests over the ready endpoints.
A retry advances to the next endpoint, so one transiently unhealthy replica
does not receive every retry of the same request. Server logs and a redacted
manifest containing commands, PIDs, endpoints, GPU assignments, and exit codes
are stored in `stage2/vllm_servers/`. All managed process groups are terminated
when Stage 2 succeeds, fails, or is interrupted.

The managed vLLM fields are:

- `server_count`: number of server replicas. If omitted, it defaults to one per
  supplied GPU.
- `gpus`: required list or comma-separated string of logical CUDA indices.
- `host`, `base_port`, or `ports`: bind address and either consecutive or exact
  ports. Defaults are `127.0.0.1` and ports beginning at 8010.
- `startup_timeout`, `startup_poll_interval`, and `shutdown_timeout`: lifecycle
  timing controls in seconds.
- `download_dir`, `reasoning_parser`, `language_model_only`, and
  `default_chat_template_kwargs`: named vLLM serve options.
- `extra_args`: a list of remaining raw vLLM CLI tokens, such as
  `["--gpu-memory-utilization", "0.90"]`. Orchestration-owned and named options
  cannot be duplicated here.

Unless explicitly overridden, any model name containing `gemma` uses
`reasoning_parser: "gemma4"` and `language_model_only: true`; it does not set a
server-wide chat-template thinking default. Any model name containing `qwen`
uses `reasoning_parser: "qwen3"` and `language_model_only: true`.

Stage 2 selects reasoning per Chat Completions request. Evidence
interpretation and audit, consolidation, operationalization, feature review,
and ontology refinement send `reasoning_effort: "high"` by default and omit
`max_tokens`. Patient extraction and its note-free category-repair requests
send `reasoning_effort: "none"` and retain the configured `max_tokens` cap.
Every Stage 2 completion request also sends the configured
`repetition_penalty` (1.1 by default).
vLLM maps those request values to Gemma 4's `enable_thinking` chat-template
switch. The configured fields are `interpretation_reasoning_effort` and
`extraction_reasoning_effort`; request-scoped values take precedence over a
server default.

Python first coalesces only exact normalized-name duplicates; this is identity
bookkeeping and makes no semantic decision between distinct names. It then
sorts the distinct candidates by normalized feature name and sends
nonoverlapping batches of `consolidation_batch_size` candidates (20 by default).
Batches within a round are independent and may run concurrently. After applying
their directives, Python re-sorts the consolidated versions and repeats for up
to `consolidation_max_rounds` rounds (55 by default). The first
`consolidation_alphabetical_rounds` rounds (5 by default) shift alphabetical
boundaries so adjacent candidates split at one boundary can meet in another.
The remaining 50 default rounds assign the re-sorted pool to new pseudorandom
batches using the run seed and outer-fold number. These seeded shuffles are exactly
reproducible but allow lexically distant aliases to be considered together. A
no-change round does not stop the process until its complete partition repeats;
this prevents one boundary layout from declaring false convergence. The process
also stops when the pool is empty or only configured features remain. Identical
canonical output names produced by independent batches are coalesced exactly
while retaining all provenance and candidate descriptions.

The clinical question is deliberately absent from every consolidation batch.
The model returns `merge_directives`, each with an `inputs` list of exact names
from that batch and one canonical `output` name. Iterative consolidation is
strictly merge-only: every supplied feature survives each round either
unchanged or as a member of one merged alias family. The response contract has
no exclusion list, and a response that supplies `exclude_feature_names` is
invalid and falls back losslessly if repairs do not remove it. Each batch sees
no candidate or group IDs and does not restate unchanged features. Python
validates that every supplied name exists, maps names back to internal groups,
unions merged provenance, and passes every unmentioned name through unchanged.
Original candidate descriptions are carried through every round so later
prompts do not lose semantic evidence.
The response normalizer recognizes only unique names and descriptions actually
supplied in that batch; this permits a copied description to resolve back to its
exact feature without fuzzy or domain-specific matching. It restores a reused
canonical output omitted from its own input family and ignores a degenerate
one-feature no-op. If a batch is still structurally invalid after bounded
repairs, Python writes `fallback.json`, passes every member of that batch
through unchanged, and records the fallback in the round and root completion
summaries. This lossless fallback also preserves every configured feature.
There is no fuzzy blocker, neighbor selection, or pairwise LLM request. Python
derives causal roles only after all rounds, allowing complementary evidence
axes from different representations to combine before role filtering. That
later deterministic filter is a separate phase: groups whose Stage 1 evidence
supports no confounder, prognostic, or effect-modifier role are excluded there,
not during alias consolidation. Its group-level decisions are written to
`consolidation/causal_role_filter.json`; investigator-configured features use
their supplied roles and remain protected.

Every group remaining after consolidation and the subsequent causal-role
filter is operationalized for extraction. Each ontology request contains only the
canonical feature name, a deduplicated flat list of readable supporting text,
the ontology instructions, and the response contract. Python extracts only
`representative_evidence.text`
from the internally selected compiled packets. Packet boundaries, evidence
kind, truncation flags, details, the outer-fold number, clinical question,
candidate or group IDs, discovery value type, evidence axes, semantic grouping,
architecture names, scores, support counts, and candidate summaries are not
sent. The model chooses binary, categorical, continuous, ordinal, or ambiguous
at this point from the feature name and readable evidence, then supplies allowed
values or a unit and the extraction and missingness rules. Python validates the
closed-ontology shape but does not hard-code domain-specific features,
categories, or units. There is no LLM ranking, diversity selection, or
feature-count cap between alias grouping and operationalization. A group that
contains a configured explicit feature skips this request and uses its supplied
ontology instead.

Operationalization requests have an independent
`operationalization_max_prompt_chars` allowance (640,000 characters by
default). Python reserves repair headroom and deterministically packs whole
supporting excerpts under the remaining initial-prompt budget. The checkpoint
records counts and character totals for available, included, omitted, and
truncated evidence together with a fingerprint of the full available evidence
list, while the final feature retains complete packet provenance. If the model
still returns an invalid ontology after all bounded repairs, Stage 2 writes
`fallback.json` and uses a conservative `ambiguous` ontology for training-fold
extraction and review rather than aborting the fold.

The API key may be set as `stage2.api_key` or in `OCI_STAGE2_API_KEY`. Other
operational controls include `request_timeout`, `transport_max_attempts`,
`transport_retry_backoff`, `max_tokens`, `max_prompt_chars`,
`consolidation_max_prompt_chars`,
`operationalization_max_prompt_chars`,
`consolidation_batch_size`, `consolidation_alphabetical_rounds`,
`consolidation_max_rounds`,
`extraction_max_prompt_chars`, `extraction_feature_batch_size`,
`evidence_compiler`, `evidence_max_cards_per_fold`,
`evidence_max_exemplars_per_card`, `evidence_max_exemplar_chars`,
`max_review_rounds`, `ontology_refinement_min_failure_patients`,
`max_ontology_refinement_rounds`, `screening_trees`,
`stability_selection_rounds`, `stability_selection_frequency`,
`effect_modifier_negative_margin_fraction`,
`effect_modifier_negative_fold_fraction`, `estimation_trees`,
`propensity_clip`, `min_nonmissing_fraction`, `max_dominant_fraction`,
`temperature`, `repetition_penalty`, `interpretation_reasoning_effort`, and
`extraction_reasoning_effort`. The legacy `max_candidates_per_fold` and
`consolidation_oversample_factor` fields are still accepted in existing run
files but do not affect consolidation. A configured endpoint or managed vLLM
pool makes the default mode `full`. The modes can always be made explicit:

Each candidate-consolidation batch uses the independent
`consolidation_max_prompt_chars` limit (640,000 characters by default), each
one-feature ontology request uses `operationalization_max_prompt_chars`
(640,000 by default), and patient-variable extraction uses
`extraction_max_prompt_chars` (also 640,000 by default). Each extraction prompt
contains one patient and at most `extraction_feature_batch_size` frozen feature
definitions (10 by default); Stage 2 checkpoints and merges the feature batches.
`max_prompt_chars`
continues to bound interpretation and review planning. These character limits
are safety/planning guards, not claims about the model's token context. Every
interpretation-class completion omits `max_tokens`, allowing the model to use
the context window available after the prompt. Extraction completions send
`max_tokens` (50,000 by default), bounding their answer. An extraction response
that reaches that limit enters Stage 2's bounded repair or fallback path.
Extraction always sends exactly one patient's text per request and never sends
more than the configured feature batch; oversized notes are split into lossless
contiguous pages. Clinical
text remains Unicode instead of expanding into token-heavy ASCII escape
sequences. Independent outer folds execute concurrently. `stage2.workers`
controls their combined endpoint concurrency, including consolidation batches
and other Stage 2 request fan-outs, without combining patients.

```bash
# Run/resume Stage 1 and stop after the handoff.
uv run python scripts/run_all_evidence.py --config run.json --stage1-only

# Skip Stage 1 and run/resume Stage 2 from the existing handoff.
uv run python scripts/run_all_evidence.py --config run.json --stage2-only

# Run both when run.mode was explicitly set to stage1 in the file.
uv run python scripts/run_all_evidence.py \
  --config run.json --set run.mode=full
```

The same choice can be stored as `run.mode: "full"`, `"stage1"`, or
`"stage2"`. `--stage1-only` and `--stage2-only` override it. Endpoint and model
values can also be supplied as `--stage2-endpoint` and `--stage2-model`.
Managed mode additionally has `--stage2-vllm-servers`,
`--stage2-vllm-gpus`, `--stage2-vllm-base-port`,
`--stage2-vllm-download-dir`, `--stage2-vllm-reasoning-parser`,
`--stage2-vllm-language-model-only` (and its `--no-` form),
`--stage2-vllm-default-chat-template-kwargs`, and repeatable
`--stage2-vllm-extra-arg` options.

## Arguments instead of a config

The common settings can be supplied directly:

```bash
uv run python scripts/run_all_evidence.py \
  --dataset /data/cohort.parquet \
  --output-dir /results/my_stage1_run \
  --unit-id-column patient_id \
  --text-column clinical_text \
  --treatment-column treatment_indicator \
  --outcome-column outcome_indicator \
  --outcome-type binary \
  --outer-folds 5 \
  --inner-folds 5 \
  --seed 42 \
  --devices cuda:0,cuda:1 \
  --workers 16 \
  --htr-model /models/bert-tiny \
  --embedding-model /models/qwen3-embedding-8b \
  --stage2-endpoint http://127.0.0.1:8010/v1 \
  --stage2-model Qwen/Qwen3-32B \
  --stage2-review-rounds 2 \
  --stage2-estimation-trees 200
```

Any less-common nested value can be set without adding another CLI flag:

```bash
uv run python scripts/run_all_evidence.py \
  --config my_run.json \
  --set science.stage1.architecture.multi_model_forest.tfidf_topic.topic_count=50 \
  --set science.neural_queries.query_epochs=80
```

Command-line values override the config file.

## Parallel execution

Top-level components run in order. The embedding cache must exist before its
consumers run, and TF-IDF writes the shared outer and inner split definitions
used by the other evidence families. Within those boundaries, the runner uses
the available CPUs and GPUs as follows.

| Component | Unit of parallel work | Concurrency control |
|---|---|---|
| `embedding_cache` | A contiguous shard of the complete chunk corpus | One encoder worker per entry in `run.devices` when multiple CUDA devices are configured |
| `tfidf` | An outer/full or exact-inner context | Separate CPU processes, bounded by `run.workers` and the number of contexts; native numerical threads are limited to one per process |
| `text_models` | An outer/full or exact-inner context, with independent BoW folds inside it | One fixed process lane per configured CUDA device; `run.workers` is divided among active lanes and bounds their combined BoW fold threads |
| `neural_queries` | An outer/full or exact-inner context | One fixed process lane per configured CUDA device; CPU-only runs use at most `run.workers` lanes |
| `handoff` | None | The completed JSONL files are combined serially |
| `stage2` | Independent outer folds, with interpretation and patient-extraction batches inside each fold | Outer folds execute concurrently; one shared semaphore bounds their combined endpoint requests at `stage2.workers`, while review rounds within each fold remain ordered |

A fixed CUDA lane processes its assigned contexts serially on one GPU. This
provides device affinity and prevents a process queue from placing two
simultaneous contexts on the same GPU. The runner creates at most one lane per
configured CUDA device, bounded by the number of unfinished contexts, and
assigns the largest remaining context to the least-loaded lane. Each
neural-query context performs its inner-fold fits, final query-bank fits, and
evidence retrieval on its assigned device. Thus the principal neural-query
parallelism is across independent contexts.

The CPU budget is shared rather than repeated across the text-model lanes. For
`W` requested CPU workers and `L` active lanes, each lane receives at least
`floor(W / L)` workers, and the remainder is assigned one worker at a time.
BoW nuisance and effect cross-fitting use that lane allocation to execute
independent folds concurrently. They use threads so the enclosing process lane
does not create a nested process pool or copy the context dataset. The three
full-context feature-importance fits are also concurrent. Native linear-algebra
and estimator thread pools remain limited to one thread per fit. If `W < L`, one
controller worker per active lane is the unavoidable minimum.

The context directories remain the unit of recovery. Each lane writes the
context's `complete.json` immediately after its artifacts are durable and
before proceeding to its next context. Rerunning the command redistributes only
the unfinished contexts among the available lanes.

## Output and resume

There is one directory to inspect and preserve:

```text
my_stage1_run/
  run_config.json
  resolved_stage1_model_config.json
  resolved_neural_query_config.json
  progress.json
  logs/
    workflow.log
  components/
    embedding_cache/
      cache/...
      complete.json
    tfidf/
      predictions.parquet
      split_provenance.jsonl
      evidence.jsonl
      stage1_tfidf_topics/contexts/...
      complete.json
    text_models/
      outer_001_full/
        evidence.json
        worker_artifacts/...
        complete.json
      outer_001_inner_001/...
      evidence.jsonl
      complete.json
    neural_queries/
      outer_001_full/...
      outer_001_inner_001/...
      evidence.jsonl
      complete.json
  stage1_architectures/
    manifest.json
    <architecture>/evidence.jsonl
  handoff/
    text_models.jsonl
    tfidf.jsonl
    neural_queries.jsonl
    evidence.jsonl
    index.json
    complete.json
  evaluations/
    stage1/
      evaluation_manifest.json
      metrics.jsonl
      comparison.csv
      summary.json
      architectures/<architecture>/metrics.jsonl
  stage2/
    config.json
    evidence_compilation/
      packets.jsonl
      summary.json
      compile_complete.json
      outer_001/
        cards.jsonl
        members.jsonl
        lineage.jsonl
    outer_001/
      input_packets.jsonl
      interpretations/...
      interpreted_candidates.json
      feature_definitions.json
      definitions_complete.json
      review/
        round_001/
          definitions.json
          extraction/
            batches/...
            failure_summary.json
            extracted.csv
            complete.json
          ontology_refinement/
            round_001/
              feature_.../
              extraction/...
              result.json
              complete.json
            result.json
            complete.json
          definitions_after_ontology_refinement.json
          extraction_summary.json
          performance.json
          review.json
          complete.json
      final_definitions.json
      extraction/
        heldout/
          batches/...
          extracted.csv
          complete.json
        extracted_features.csv
      estimation/
        predictions.csv
        diagnostics.json
        complete.json
      complete.json
    features_by_outer_fold.jsonl
    cross_fitted_predictions.csv
    posthoc_predictions_with_oracle_ite.csv
    posthoc_oracle_ite_metrics.json
    causal_estimate.json
    summary.json
    complete.json
```

`progress.json` is the first place to look while a run is active. The model
outputs are under `components/<name>/`. The plain Stage 2 boundary is
`handoff/evidence.jsonl`; `handoff/index.json` explains the source files and
the original per-family JSONL files remain beside it. Python consumers can
stream the combined rows with:

```python
from oci.inference.research_all_evidence_workflow import iter_stage1_handoff

for evidence_context in iter_stage1_handoff("/results/my_stage1_run"):
    ...
```

Completion has one intentionally simple rule:

- if `components/<name>/complete.json` exists, that component is reused and its
  progress status is `already_complete`
  (`handoff/complete.json` and a causal-estimation `stage2/complete.json` serve
  the same purpose for those components);
- if it does not exist, the component runs in the existing directory;
- text-model and neural-query contexts use the same rule inside their context
  directories.

An interrupted component's partial files are left in place. There is no
`--resume` flag. Rerun the same command. Stage 2 skips completed interpretation
and extraction batches, completed review rounds, and completed fold estimates.
The compiled packet plan is cached under `stage2/evidence_compilation/`; on a
restart the runner hashes the handoff and normally loads this small cache rather
than reparsing and reclustering the raw Stage 1 evidence. Interpretation batches
are skipped only when their input fingerprint (cards plus clinical question)
matches.
It writes an outer-fold completion marker only after held-out estimation, and
writes the final top-level marker only after the cross-fitted estimates have
been assembled.

Raw-packet interpretation checkpoints do not match semantic-card inputs and are
therefore rerun automatically. If a prior run already produced
`feature_definitions.json`, the runner fails closed instead of silently mixing
those definitions with a new evidence plan or a changed explicit-feature
configuration; preserve that directory for audit and select a fresh Stage 2
output directory for the new definition inputs. The Stage 1 handoff itself does
not need to be regenerated when only `stage2.explicit_features` changes.

The review boundary is deliberately fold-honest. Extraction summaries and
performance metrics in `review/round_NNN/` contain outer-training rows only.
Outer-held-out outcomes are not made available to definition revision or
feature retention. Continuous definitions accept either a numeric scalar or a
documented categorical/threshold scalar when no exact number is reported. The
summary records numeric and fallback-category distributions, and the reviewer
chooses `continuous`, `categorical`, or
`continuous_with_categorical_fallback` modeling before estimation.

The performance file includes baseline, complete-feature, leave-one-feature-out,
and singleton `individual_feature_signal` measurements. Propensity, outcome,
and treatment-effect models use random forests whenever feature columns exist.
Repeated screens are fit on inner-fit rows and evaluated on inner-held-out rows.
Confounder and prognostic roles require stable positive support. Effect
modifiers use asymmetric pruning: they remain eligible unless a configured
fraction of repeated screens shows consistent R-loss deterioration larger than
the configured relative negative margin. LLM drop recommendations pass through
the same gate. Any drop, role change, measurement revision, or modeling-strategy
change starts another evaluation round. Evaluation-only convergence rounds may
therefore extend beyond `max_review_rounds`, which limits language-model reviews
rather than final empirical certification. Each round records the full votes,
margins, deterministic decisions, and per-role evidence in
`signal_pruning.json`.

When a continuous extraction contains both numeric and categorical values, a
separate LLM request receives only the feature ontology and aggregated
outer-training values. It chooses one validated continuous mapping or an
exhaustive categorical binning plan. That plan is frozen in the feature
definition and applied deterministically to final training and held-out values;
held-out values and outcomes are never sent back for harmonization.

Every single-patient extraction writes `extraction_issues.json`, including
feature-attributable invalid scalar/type values and values outside a declared
closed ontology. The extraction directory aggregates those events by feature,
failure kind, and distinct patient in `failure_summary.json`. A generic malformed
response is counted separately as a structural transport/format failure and
cannot trigger an ontology change. When the same attributable pattern occurs in
at least `ontology_refinement_min_failure_patients` outer-training patients (3
by default), a one-feature ontology-refinement component receives the frozen
definition, aggregate count, and a bounded list of failed model outputs—but no
patient text, treatment, outcome, or held-out information. It may keep the
definition or revise only its description, value type, categories or unit,
measurement rule, and missingness rule. The feature name, identity, roles, and
support remain fixed. The training patients are then re-extracted and monitored
again, for at most `max_ontology_refinement_rounds` revisions (2 by default),
before empirical review proceeds. Investigator-configured explicit ontologies
are immutable: their repeated failures are audited, but no refinement request is
made. The held-out extraction begins only after
`final_definitions.json` has been written. The reported average treatment effect
is the mean of the held-out AIPW scores across outer folds; its standard error
is the empirical standard error of those scores. `estimated_cate` in the
prediction file comes from a random-forest effect model trained on cross-fitted
training-fold pseudo-outcomes and the variables assigned an effect-modifier
role.

If the dataset supplies `true_ite_prob`, Stage 2 hashes and freezes
`cross_fitted_predictions.csv` before joining that oracle column. The final
`causal_estimate.json` includes overall Pearson and Spearman correlations
between `estimated_cate` and oracle ITE. Detailed overall and per-fold
correlations, MAE, RMSE, ATE bias, and ITE dispersion are written to
`posthoc_oracle_ite_metrics.json`; the joined rows are stored separately in
`posthoc_predictions_with_oracle_ite.csv`. Without an oracle column, the same
metrics file records `available: false`.

Stage 1 itself can be evaluated architecture by architecture after the handoff
has frozen its evidence:

```bash
uv run oci-evaluate-stage1 \
  --run-dir /results/my_stage1_run \
  --metadata /data/synthetic/metadata.json \
  --architectures all
```

The evaluator hashes all consumed evidence and row-score sidecars before it
loads oracle-bearing data. It never fits, ranks, or selects Stage 1 models. The
per-architecture files contain native metrics appropriate to each evidence
representation plus a common recovery view; `comparison.csv` is the compact
cross-architecture summary. A completed legacy handoff is backfilled into the
same additive `stage1_architectures/` contract without refitting.

To intentionally rerun one component, use:

```bash
uv run python scripts/run_all_evidence.py \
  --config my_run.json \
  --rerun tfidf
```

This removes that component's completion markers; it does not delete model
outputs. Contexts are then recomputed in their existing directories.

To print status without starting work:

```bash
uv run python scripts/run_all_evidence.py \
  --config my_run.json \
  --status
```

Because there is deliberately no config identity check, changing scientific
settings does not invalidate existing completion markers. Use a new output
directory for a scientifically different run, or explicitly rerun every
affected component.

## Components

- `embedding_cache` encodes reusable text chunks.
- `tfidf` runs first and writes the plainly readable fold definitions used by
  the remaining evidence families, plus topic and residual n-gram evidence.
- `text_models` runs the established BoW, HTR, matched-pair, whole-cohort
  embedding, cluster-local embedding, and lexical-retrieval evidence models.
- `neural_queries` fits and saves each outer/full or exact-inner context
  independently.
- `handoff` gathers the completed evidence into the stable Stage 2 input path.
- `stage2` interprets the plain handoff, extracts and empirically reviews
  fold-scoped variables, freezes the retained definitions, and writes held-out
  AIPW and conditional-effect estimates before aggregating the outer folds.

The Stage 1 scientific model implementations are reused. The plain Stage 2 path
replaces the former production control plane with readable directories and a
small dependency-light estimator: logistic or ridge nuisance models, a
random-forest effect-modification model, and cross-fitted AIPW scores. These
choices are recorded in the fold diagnostics and final estimate rather than
hidden in an authenticated deployment specification.
