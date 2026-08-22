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
when the pipeline launches vLLM itself. Dataset-backed Stage 2 always requires a
`stage2.extraction_llm` configuration. It may point to a different endpoint or
the same multi-model endpoint; its model is auto-discovered only when that
endpoint advertises exactly one model.

To run a scientific subset of Stage 1, add (for example)
`--architectures bow_nuisance,tfidf_topics`. Private prerequisites are resolved
automatically and Stage 2 receives only those selected lanes. Keep the same
selection when resuming; changing it requires a fresh output directory. Omit
the option for legacy all-enabled behavior.

Stage 2 does not stop at variable definitions. For each outer fold it exhaustively
lists clinical features from every semantic evidence card, performs merge-only
consolidation, uses a separate small model to extract all candidates on training
records, and lets the primary model review only aggregate extraction ontologies.
Fold-local regressions then assign confounder and effect-modifier roles before a
causal forest is fit and evaluated on outer-held-out records. The common controls
are:

```json
{
  "stage2": {
    "endpoint": "http://127.0.0.1:8010/v1",
    "model": "Qwen/Qwen3.8-27B",
    "workers": 32,
    "selection_workers": 32,
    "extraction_llm": {
      "endpoint": "http://127.0.0.1:8020/v1",
      "model": "small-extractor",
      "workers": 32
    },
    "max_tokens": 100000,
    "max_response_repairs": 10,
    "thinking_after_response_repairs": 5,
    "repetition_penalty": 1.1,
    "interpretation_reasoning_effort": "high",
    "extraction_reasoning_effort": "none",
    "evidence_compiler": "semantic_cluster_cards_v2",
    "evidence_max_cards_per_fold": 400,
    "extraction_feature_batch_size": 10,
    "max_review_rounds": 2,
    "confounder_p_value_threshold": 0.05,
    "confounder_min_inner_fold_fraction": 0.75,
    "effect_modifier_p_value_threshold": 0.05,
    "effect_modifier_min_inner_fold_fraction": 0.75,
    "estimation_trees": 200,
    "explicit_features": []
  }
}
```

Interpretation, consolidation, operationalization, category mapping, and
aggregate ontology-review requests go to the primary model with
`reasoning_effort: "high"`. One-patient value extraction alone goes to the
configured `extraction_llm` model with `reasoning_effort: "none"`. The two
models may use different endpoints or the same multi-model endpoint.
`max_tokens` is a 100,000-token output ceiling on every request; it does not ask
or force a model to generate that many tokens, and normal EOS stopping applies.
All Stage 2 completion requests send `repetition_penalty: 1.1` by default.
Stage 2 probes `/models`, recognizes Qwen 3 (including 3.8), Gemma 4, and LFM
2.5 IDs, and sends family-appropriate per-request thinking controls. It accepts
either server-parsed reasoning fields or inline reasoning delimiters.
The selected IDs are persisted in `stage2/model_identity.json`: endpoint URL
changes may resume, but changing either running model ID raises an error.
Invalid completed responses receive up to 10 validator-guided repair retries.
The first five repairs retain the request's normal reasoning policy; repairs
6–10 force `reasoning_effort` to at least `high`, enabling thinking.

`stage2.explicit_features` may contain investigator-specified feature
definitions. Each entry must include its complete extraction ontology and
causal roles; see the complete workflow guide for the schema. Configured
features join Stage 2 alias consolidation in every outer fold, so an
automatically discovered alias does not create a second variable. They remain
fixed and required regardless of evidence or p-values. Configure either role or
both; their ontologies and roles cannot be changed by the models.

Independent outer folds run concurrently. Primary request concurrency is bounded
by `stage2.workers`, and patient extraction by
`stage2.extraction_llm.workers`. Each extraction request contains exactly one
patient's text; this isolation is invariant. Statistical screens use
`stage2.selection_workers` loky processes.

Discovered confounders must predict both treatment and outcome with raw p-values
strictly below the configured threshold in at least the configured fraction of
inner folds. Modifier models adjust for all selected confounders and treatment,
then test one candidate's treatment interaction; the same threshold/fraction
vote rule applies. Categorical candidates enter with all estimable
nonreference-level interaction terms and receive one omnibus likelihood-ratio
test. The default for both p-values is `0.05`, and the default for both fold
fractions is `0.75` (rounded up to a whole-fold count).

For eight independent vLLM replicas on eight GPUs, replace `endpoint` in the
example above with:

```json
{
  "stage2": {
    "model": "google/gemma-4-31B-it",
    "workers": 32,
    "extraction_llm": {
      "endpoint": "http://127.0.0.1:8020/v1",
      "model": "small-extractor",
      "workers": 32
    },
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
reasoning parser and language-model-only mode; thinking is selected per request
as described above. Qwen defaults to the `qwen3` reasoning parser and
language-model-only mode. See the complete
workflow guide for GPU partition rules and all managed-server settings.

Stage 2 compiles the raw handoff into fold-local, provenance-preserving semantic
cards under `stage2/evidence_compilation/`, and candidate discovery reads all of
them. There is no ColBERT interaction filter, evidence-community graph,
candidate reranking, or feature-count cap. Consolidation may only merge aliases;
every unmerged candidate proceeds to extraction and the inner-fold screens.
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
