# All-evidence workflow

The researcher workflow is one command:

```bash
uv run python scripts/run_all_evidence.py \
  --config example_configs/research_all_evidence.json
```

It reads one cohort and writes everything to one output directory. With no
Stage 2 command configured it runs Stage 1 through the plain handoff. With a
Stage 2 command it runs both phases. Interruption and resume are automatic:
run the same command again.

The older `run_production_all_evidence_workflow.py` control plane is retained
for old runs, but it is not required by this workflow. This orchestration has
no source snapshot, immutable request, checkpoint adoption, trust policy,
artifact authentication, or content-hash protocol.

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
    "command": []
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

## Full, Stage-1-only, and Stage-2-only runs

`stage2.command` is a list of arguments for the study's Stage 2 entry point.
It is executed directly, without a shell. The following placeholders are
expanded in any argument:

- `{dataset}`: configured input dataset;
- `{output_dir}`: the whole workflow output;
- `{handoff}`: `handoff/evidence.jsonl`;
- `{handoff_dir}`: directory containing the combined and per-family handoffs;
- `{stage2_output}`: the visible `stage2/` output directory.

For example:

```json
{
  "stage2": {
    "command": [
      "uv", "run", "python", "/study/run_stage2.py",
      "--dataset", "{dataset}",
      "--handoff", "{handoff}",
      "--output-dir", "{stage2_output}"
    ],
    "working_dir": "/study"
  }
}
```

A nonempty command makes the default mode `full`. The modes can always be made
explicit:

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
`"stage2"`. `--stage1-only` and `--stage2-only` override it. Stage 2 also
receives the five paths above as `OCI_DATASET`, `OCI_RUN_OUTPUT`,
`OCI_STAGE1_HANDOFF`, `OCI_STAGE1_HANDOFF_DIR`, and `OCI_STAGE2_OUTPUT`.

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
  --embedding-model /models/qwen3-embedding-8b
```

Any less-common nested value can be set without adding another CLI flag:

```bash
uv run python scripts/run_all_evidence.py \
  --config my_run.json \
  --set science.stage1.architecture.multi_model_forest.tfidf_topic.topic_count=50 \
  --set science.neural_queries.query_epochs=80
```

Command-line values override the config file.

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
  handoff/
    text_models.jsonl
    tfidf.jsonl
    neural_queries.jsonl
    evidence.jsonl
    index.json
    complete.json
  stage2/
    run.json
    ...study Stage 2 outputs...
    complete.json
```

`progress.json` is the first place to look while a run is active. The model
outputs are under `components/<name>/`. The plain Stage 2 boundary is
`handoff/evidence.jsonl`; `handoff/index.json` explains the source files and
the original per-family JSONL files remain beside it. Python consumers can
stream the combined rows with:

```python
from oci.inference.research_all_evidence_stage1 import iter_stage1_handoff

for evidence_context in iter_stage1_handoff("/results/my_stage1_run"):
    ...
```

Completion has one intentionally simple rule:

- if `components/<name>/complete.json` exists, that component is skipped
  (`handoff/complete.json` and `stage2/complete.json` for those components);
- if it does not exist, the component runs in the existing directory;
- text-model and neural-query contexts use the same rule inside their context
  directories.

An interrupted component's partial files are left in place. There is no
`--resume` flag. Rerun the same command. Stage 2 is reinvoked in the same
`stage2/` directory, so its command should use that directory for its own
granular checkpoints; the workflow adds only the final top-level marker.

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
- `stage2` invokes the configured Stage 2 command against that handoff and
  marks it complete only after a zero exit status.

The scientific model implementations are reused; this change removes their
production orchestration layer rather than substituting new estimators.
