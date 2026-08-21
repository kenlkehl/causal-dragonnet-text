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
    "max_tokens": 50000,
    "max_response_repairs": 10,
    "thinking_after_response_repairs": 5,
    "repetition_penalty": 1.1,
    "interpretation_reasoning_effort": "high",
    "extraction_reasoning_effort": "none",
    "evidence_compiler": "semantic_cluster_cards_v2",
    "evidence_max_cards_per_fold": 400,
    "evidence_community_enabled": true,
    "evidence_community_model": "answerdotai/answerai-colbert-small-v1",
    "evidence_community_device": "cpu",
    "evidence_community_max_packets": 75,
    "evidence_community_min_per_causal_lane": 30,
    "max_candidates_per_fold": 50,
    "candidate_selection_top_n": 50,
    "candidate_registry_embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "candidate_registry_embedding_device": "cpu",
    "candidate_registry_similarity_threshold": 0.94,
    "candidate_selection_method": "late_interaction",
    "candidate_selection_late_interaction_model": "answerdotai/answerai-colbert-small-v1",
    "candidate_selection_late_interaction_device": "cpu",
    "candidate_selection_top_evidence_packets": 3,
    "max_review_rounds": 2,
    "max_evaluation_rounds": 10,
    "screening_trees": 200,
    "stability_selection_rounds": 3,
    "stability_selection_frequency": 0.6666666667,
    "effect_modifier_negative_margin_fraction": 0.01,
    "effect_modifier_negative_fold_fraction": 0.6,
    "estimation_trees": 200,
    "explicit_features": []
  }
}
```

Interpretation, consolidation, operationalization, and review requests send
`reasoning_effort: "high"` and do not send an output-token cap. Patient
extraction sends `reasoning_effort: "none"`; `max_tokens` is its response cap.
All Stage 2 completion requests send `repetition_penalty: 1.1` by default.
Managed Gemma 4 servers therefore use the `gemma4` reasoning parser without a
server-wide `enable_thinking` default.
Invalid completed responses receive up to 10 validator-guided repair retries.
The first five repairs retain the request's normal reasoning policy; repairs
6–10 force `reasoning_effort` to at least `high`, enabling thinking.

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
reasoning parser and language-model-only mode; thinking is selected per request
as described above. Qwen defaults to the `qwen3` reasoning parser and
language-model-only mode. See the complete
workflow guide for GPU partition rules and all managed-server settings.

Before interpretation, Stage 2 compiles the raw handoff into fold-local,
provenance-preserving semantic cards under `stage2/evidence_compilation/`.
It then turns each card representative into overlapping 16-word atoms, builds a
cross-architecture reciprocal-neighbor graph with symmetric document/document
ColBERT scoring, and clusters it. The best 30 confounder-lane and 30
modifier-lane communities are reserved independently, overlaps are
deduplicated, and the remaining capacity is filled by global rank up to 75.
Each LLM item contains community consensus phrases plus at most three
architecture-diverse exemplars. The full atom, edge, community, and source
lineage audits are under `stage2/evidence_communities/`; no oracle metadata is
used in selection.

After interpretation, Stage 2 first collapses exact normalized names, then uses
the registry embedding model for conservative, lexically anchored alias
merges. Each canonical name is rendered as natural language (`patient_age`
becomes `Patient Age`) and scored only against the evidence packets that cited
it. The default ColBERT-style scorer keeps at most five candidates per evidence
axis, and `max_candidates_per_fold` supplies a hard overall cap before global
LLM alias consolidation. Provenance associations are retained; the three
highest-scoring packets are separately chosen as ontology evidence. This means
2,000 packets can create 10,000 scored candidate-packet associations, but they
cannot send 10,000 candidate features downstream.
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
