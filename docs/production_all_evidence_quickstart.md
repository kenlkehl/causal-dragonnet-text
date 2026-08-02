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
existing handoff. A nonempty `stage2.command` in the config makes an unflagged
invocation run both phases.

The Stage 2 input is always:

```text
/path/to/output/handoff/evidence.jsonl
```

See the [complete workflow guide](production_all_evidence_end_to_end.md) for
the config schema, Stage 2 command contract, output layout, direct CLI
arguments, and component reruns.
