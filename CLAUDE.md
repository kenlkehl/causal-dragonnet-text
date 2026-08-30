# CLAUDE.md — OCI repository guide

## Purpose

OCI implements a science-first, multi-model causal-text workflow:

1. Stage 1 produces frozen evidence from ten independent architectures.
2. A versioned plain handoff exposes only the selected architectures.
3. Stage 2 reviews and consolidates evidence before structured estimation.
4. Post-hoc evaluation compares each frozen Stage 1 architecture with oracle
   truth without allowing oracle information into discovery or fitting.

The supported production entry point is `scripts/run_all_evidence.py`. The
`oci run` compatibility command is retained only for explicit-feature
workflows.

## Retained package structure

```text
oci/
├── config.py                         # Stage 1 and explicit-feature configuration
├── evaluation/stage1.py              # Oracle-safe architecture evaluation
├── extraction/                       # Explicit-feature extraction contracts
├── inference/
│   ├── research_all_evidence_workflow.py
│   ├── multi_model_forest_stage1.py
│   ├── stage1_architectures.py
│   ├── stage1_architecture_artifacts.py
│   ├── plain_handoff_stage2.py
│   ├── vllm_server_pool.py
│   ├── plain_handoff_stage2_evidence.py
│   ├── all_evidence_fusion.py
│   ├── tfidf_topic_stage1.py
│   ├── embedding_contrast_discovery.py
│   ├── neural_query_discovery_runtime.py
│   └── agentic_explicit_feature_forest.py
├── models/
│   ├── causal_forest_head.py
│   ├── elastic_net_nuisance.py
│   ├── hierarchical_transformer_extractor.py
│   ├── concept_embedding_cache.py
│   ├── explicit_feature_featurizer.py
│   └── structured_interaction_head.py
└── utils/
```

The repository intentionally does not ship the retired DragonNet pipeline,
single-representation neural heads, CNN/GRU/slot extractors, generic hidden-
state cache, post-hoc matching package, or standalone TF-IDF forest wrapper.

## Stage 1 architectures

The canonical order in `oci/inference/stage1_architectures.py` is:

1. `bow_nuisance`
2. `bow_r_loss`
3. `matched_pair_uplift`
4. `htr_neural`
5. `embedding_whole_cohort`
6. `embedding_clustered`
7. `tfidf_semantic_retrieval_contrasts`
8. `tfidf_topics`
9. `tfidf_orphan_ngrams`
10. `neural_query_moments`

When `stage1_architectures` is omitted, all ten run in canonical order for
backward compatibility. An explicit selector is frozen into run state and may
not change on resume. Private prerequisites may run, but only selected lanes
may enter the Stage 2 handoff.

## Scientific invariants

- Never use oracle columns during discovery, fitting, selection, or handoff
  construction.
- Final Stage 2 role prompts may contain only the allowlisted aggregate evidence
  bundle. Never add dataset paths/names, synthetic-generation metadata, known
  data-generating roles, row values, identifiers, or outer-heldout information.
- Hash frozen Stage 1 artifacts before loading oracle data for evaluation.
- Keep outer-test rows unavailable to inner-fold discovery and review.
- Preserve exact row, split, component, architecture, and configuration
  provenance in sidecars.
- Fail closed when text capacity would truncate semantic content.
- Keep causal-forest nuisance models elastic-net-only, and retain fitted-clone
  audit records for selected regularization and optimizer iteration limits.
- Do not silently change the omitted-selector legacy path.

## Explicit-feature functionality

Explicit-feature extraction and modeling are supported and must not be removed
as incidental cleanup. Retained standalone model types are:

- `explicit_feature_forest`
- `agentic_explicit_feature_forest`
- `agentic_attention_variable_forest`
- `multi_model_agentic_forest`

Feature contracts are role-aware. Use `roles: ["confounder"]`,
`roles: ["effect_modifier"]`, or both. Preserve complete-note paging,
citation validation, extraction caching, post-extraction review, and the
explicit-feature featurizers.

## Common commands

```bash
# Full or resumed Stage 1 → Stage 2 workflow
python scripts/run_all_evidence.py --config example_configs/research_all_evidence.json

# Select a subset of Stage 1 architectures
python scripts/run_all_evidence.py --config example_configs/research_all_evidence.json \
  --architectures bow_nuisance,tfidf_topics

# Evaluate frozen Stage 1 outputs
python scripts/evaluate_stage1_architectures.py \
  --run-dir artifacts/research_all_evidence --architectures all

# Standalone explicit-feature workflow
oci run --config example_configs/agentic_explicit_feature_forest_config.json

# Tests
pytest -q
```

## Change checklist

When changing Stage 1 or Stage 2:

1. Update the canonical registry if architecture identity changes.
2. Preserve or deliberately version handoff and artifact schemas.
3. Test omitted, subset, resume, and missing-artifact behavior.
4. Add architecture-native evaluation metrics without refitting.
5. Verify lightweight imports do not initialize retired or optional runners.
6. Run `git diff --check`, focused tests, the full suite, and package build.

More detail is in `docs/all_evidence_workflow.md` and
`docs/all_evidence_quickstart.md`.
