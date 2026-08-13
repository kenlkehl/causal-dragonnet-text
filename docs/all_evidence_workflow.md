# All-evidence workflow

The researcher workflow is one command:

```bash
uv run python scripts/run_all_evidence.py \
  --config example_configs/research_all_evidence.json
```

It reads one cohort and writes everything to one output directory. With no
Stage 2 endpoint configured it runs Stage 1 through the plain handoff. With an
endpoint it continues through fold-scoped feature definitions,
patient-level extraction, training-fold empirical review, and cross-fitted
causal estimation. Interruption and resume are automatic: run the same command
again.

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
effect estimates. It is enabled by specifying `stage2.endpoint`. For example:

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

```json
{
  "stage2": {
    "endpoint": "http://127.0.0.1:8010/v1",
    "model": "Qwen/Qwen3-32B",
    "workers": 8,
    "request_timeout": 7200,
    "evidence_compiler": "semantic_cluster_cards_v2",
    "evidence_max_cards_per_fold": 400,
    "evidence_max_exemplars_per_card": 4,
    "evidence_max_exemplar_chars": 2400,
    "max_review_rounds": 2,
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

Configured features enter the full-pool alias-consolidation pass in every outer fold.
When Stage 1 discovers an alias, Python retains one consolidated feature,
attaches the discovered packet and architecture provenance, keeps the
configured name and roles, and uses the supplied ontology without making the
one-feature ontology request. Distinct configured feature names are never
merged with each other. They still undergo training-fold extraction and
empirical diagnostics, but review must keep them without revising the supplied
ontology. If a required feature cannot be extracted well enough for the
workflow's health checks, the run fails visibly rather than silently dropping
or redefining it.

`stage2.model` is optional. If it is empty or omitted, Stage 2 queries the
OpenAI-compatible `/models` endpoint once at startup and uses the result when
exactly one model ID is advertised. If the server advertises multiple model
IDs, set `stage2.model` explicitly to avoid an ambiguous selection.

Candidate alias consolidation is one semantic request over the complete pool.
Python first coalesces only exact normalized-name duplicates so response routes
remain unambiguous; this is identity bookkeeping and makes no semantic decision
between distinct names. It preserves every source candidate and supplies every
distinct name together with all of its distinct candidate descriptions. There
is no fuzzy blocker, neighbor selection, pairwise LLM request, or transitive
assembly from local decisions.

The clinical question is deliberately absent from the full-pool request. The
model sees all distinct candidates at once, including candidates whose evidence
does not independently establish a causal role. It returns `merge_directives`,
each with an `inputs` list of exact supplied names and one canonical `output`
name, plus `exclude_feature_names`. Exclusion is restricted to clear failures of
the patient-level scalar contract: patient-specific or value-encoded artifacts,
profiles and composites, and nonclinical analysis or documentation artifacts.
Borderline but valid clinical variables pass through, and
investigator-configured features cannot be excluded. The pass sees no candidate
or group IDs and does not restate unchanged features. Python validates that
every supplied name exists, prevents a name from being both merged and excluded,
maps names back to the internal groups, unions merged provenance, records
excluded-candidate dispositions, and passes every unmentioned name through
unchanged. The instructions explicitly treat a general measurement, quantitative
score, thresholded or categorical state, and value-encoded name as equivalent
representations when one underlying patient variable can encode them, while
keeping independently varying components separate. Python derives causal roles
only after these groups are formed, allowing complementary evidence axes from
different representations to combine before role filtering.

Every group remaining after those merge and exclusion directives is
operationalized for extraction. Each ontology request contains only the
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

The API key may be set as `stage2.api_key` or in `OCI_STAGE2_API_KEY`. Other
operational controls include `request_timeout`, `transport_max_attempts`,
`transport_retry_backoff`, `max_prompt_chars`,
`consolidation_max_prompt_chars`,
`extraction_max_prompt_chars`,
`evidence_compiler`, `evidence_max_cards_per_fold`,
`evidence_max_exemplars_per_card`, `evidence_max_exemplar_chars`,
`max_review_rounds`, `estimation_trees`,
`propensity_clip`, `min_nonmissing_fraction`, `max_dominant_fraction`,
`temperature`, and `enable_thinking`. The legacy `max_candidates_per_fold` and
`consolidation_oversample_factor` fields are still accepted in existing run
files but do not affect consolidation. A configured endpoint makes the default
mode `full`. The modes can always be made explicit:

The one-request candidate-pool pass uses the independent
`consolidation_max_prompt_chars` limit (640,000 characters by default), while
patient-variable extraction uses `extraction_max_prompt_chars` (also 640,000 by
default) because every extraction request includes the complete frozen feature
ontology. `max_prompt_chars` continues to bound interpretation and review
planning. These character limits are safety/planning guards, not claims about
the model's token context. Extraction always sends exactly one patient's text
per request; oversized notes are split into lossless contiguous pages. Clinical
text remains Unicode instead of expanding into token-heavy ASCII escape
sequences. `stage2.workers` provides concurrency without combining patients.

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
| `stage2` | Interpretation batches and patient-extraction batches within the current outer fold | Concurrent endpoint requests bounded by `stage2.workers`; review rounds, outer folds, and fold-level estimation remain ordered |

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
            extracted.csv
            complete.json
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
feature retention. The performance file includes baseline, complete-feature,
and leave-one-feature-out measurements so that retention decisions can be tied
to an individual extracted variable rather than to the feature set in the
aggregate. The held-out extraction begins only after
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
