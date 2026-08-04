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

5. If the process stops, run step 3 again. Completed components are skipped.

Use `--stage1-only` to stop at the handoff or `--stage2-only` to consume an
existing handoff. Setting `stage2.endpoint` in the config makes an unflagged
invocation run both phases. `stage2.model` may be omitted when the endpoint's
`/models` API advertises exactly one model ID.

Stage 2 does not stop at variable definitions. For each outer fold it extracts
the proposed variables on training records, reviews their empirical behavior by
inner validation, freezes the retained definitions, extracts the held-out
records, and computes held-out nuisance predictions, AIPW scores, and treatment
effect estimates. The common controls are:

```json
{
  "stage2": {
    "endpoint": "http://127.0.0.1:8000/v1",
    "model": "Qwen/Qwen3-32B",
    "workers": 8,
    "extraction_batch_size": 12,
    "max_review_rounds": 2,
    "estimation_trees": 200
  }
}
```

Interpretation and extraction requests are concurrent up to `stage2.workers`.
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

See the [complete workflow guide](all_evidence_workflow.md) for
the config schema, Stage 2 endpoint contract, output layout, direct CLI
arguments, and component reruns.
