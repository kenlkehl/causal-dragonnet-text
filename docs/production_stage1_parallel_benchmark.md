# Stage 1 parallelism benchmark artifacts

The public command is `scripts/run_stage1_parallelism_benchmark.py`. It always
runs the same ordered jobs at 1, 4, and 8 loky workers, caps native numerical
libraries at one thread per worker, and rejects any change in the canonical
scientific outputs.

Wall time is a steady-state measurement: each loky worker imports the measured
stack under the same one-thread cap before the clock starts. The reusable loky
executor is shut down between worker-count trials so one trial cannot silently
lend warm workers to the next. Import warmup time is explicitly excluded.

A CPU-only fixture can be created and run without opening a cohort:

```bash
/home/klkehl/thisenv/bin/python \
  scripts/run_stage1_parallelism_benchmark.py \
  --write-fixture-plan /absolute/path/fixture_plan.json

/home/klkehl/thisenv/bin/python \
  scripts/run_stage1_parallelism_benchmark.py \
  --fixture-plan /absolute/path/fixture_plan.json \
  --output-root /absolute/fresh/output
```

For the real clustered preflight, pass the terminal
`preflight_scope_input_set_manifest.json` published by input preparation. A
repeatable `--preflight-scope-id` selects representative authenticated scopes;
omitting it benchmarks all published scopes. The command receives no cohort or
oracle path.

The output remains incomplete until `terminal_manifest.json` is written. That
file is published last and binds the request, source/code hashes, input hashes,
per-family timings, speedups, native-thread observations, and exact scientific
equality result. A changed source file, input manifest, result byte, missing
file, extra file, or symlink causes validation to fail.
