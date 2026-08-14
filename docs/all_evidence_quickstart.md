# All-evidence quickstart

1. Copy the example config:

   ```bash
   cp example_configs/research_all_evidence.json my_all_evidence_run.json
   ```

2. Set `dataset`, `output_dir`, the four column names, model locations, devices,
   and the fold/seed parameters.

3. Run:

   ```bash
   uv run python scripts/run_all_evidence.py --config my_all_evidence_run.json
   ```

4. Watch progress:

   ```bash
   uv run python scripts/run_all_evidence.py \
     --config my_all_evidence_run.json --status
   tail -f /path/to/output/logs/workflow.log
   ```

5. If the process stops, run step 3 again. Completed components are reused and
   reported as `already_complete`.

Use `--stage1-only` to stop at the handoff or `--stage2-only` to consume an
existing handoff. Setting `stage2.endpoint` or `stage2.vllm` in the config makes
an unflagged invocation run both phases. `stage2.model` may be omitted when an
external endpoint's `/models` API advertises exactly one model ID; it is required
when the pipeline launches vLLM itself.

To run a scientific subset of Stage 1, add (for example)
`--architectures bow_nuisance,tfidf_topics`. Private prerequisites are resolved
automatically and Stage 2 receives only those selected lanes. Keep the same
selection when resuming; changing it requires a fresh output directory. Omit
the option for legacy all-enabled behavior.

Stage 2 does not stop at variable definitions. For each outer fold it extracts
the proposed variables on training records, reviews their empirical behavior by
inner validation, freezes the retained definitions, extracts the held-out
records, and computes held-out nuisance predictions, AIPW scores, and treatment
effect estimates. The common controls are:

```json
{
  "stage2": {
    "endpoint": "http://127.0.0.1:8010/v1",
    "model": "Qwen/Qwen3-32B",
    "workers": 32,
    "evidence_compiler": "semantic_cluster_cards_v2",
    "evidence_max_cards_per_fold": 400,
    "max_review_rounds": 2,
    "estimation_trees": 200,
    "explicit_features": []
  }
}
```

`stage2.explicit_features` may contain investigator-specified feature
definitions. Each entry must include its complete extraction ontology and
causal roles; see the complete workflow guide for the schema. Configured
features join Stage 2 alias consolidation in every outer fold, so an
automatically discovered alias does not create a second variable. They remain
fixed, required definitions during empirical review.

Independent outer folds run concurrently, and their combined interpretation and
extraction request concurrency is bounded by `stage2.workers`. Each extraction
request contains exactly one patient's text; this isolation is an invariant
rather than a configurable batch-size choice.

For eight independent vLLM replicas on eight GPUs, replace `endpoint` in the
example above with:

```json
{
  "stage2": {
    "model": "google/gemma-4-31B-it",
    "workers": 32,
    "vllm": {
      "server_count": 8,
      "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
      "download_dir": "/models/huggingface",
      "extra_args": ["--gpu-memory-utilization", "0.90"]
    }
  }
}
```

Stage 2 starts the servers, waits for all eight model endpoints, round-robins
work across them, and stops them on exit. Gemma defaults to the `gemma4`
reasoning parser, language-model-only mode, and thinking enabled. Qwen defaults
to the `qwen3` reasoning parser and language-model-only mode. See the complete
workflow guide for GPU partition rules and all managed-server settings.

Before interpretation, Stage 2 compiles the raw handoff into fold-local,
provenance-preserving semantic cards under `stage2/evidence_compilation/`.
Each completed request is saved beneath the relevant outer-fold directory, so
the same command resumes after interruption without repeating it.

The Stage 2 input is always:

```text
/path/to/output/handoff/evidence.jsonl
```

The final estimate and row-level cross-fitted results are:

```text
/path/to/output/stage2/causal_estimate.json
/path/to/output/stage2/cross_fitted_predictions.csv
```

For synthetic data with known truth, evaluate the frozen Stage 1 lanes in their
own right:

```bash
uv run oci-evaluate-stage1 \
  --run-dir /path/to/output \
  --metadata /path/to/metadata.json \
  --architectures all
```

See the [complete workflow guide](all_evidence_workflow.md) for
the config schema, Stage 2 endpoint contract, output layout, direct CLI
arguments, and component reruns.
