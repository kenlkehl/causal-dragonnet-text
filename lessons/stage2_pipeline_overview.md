# Stage 2 pipeline overview

Stage 2 is a fold-honest evidence-to-variable-to-causal-estimate pipeline. It
converts Stage 1 model evidence into measurable clinical features, reviews and
refines those features using only the outer-training data, freezes their
definitions, and estimates effects on the untouched outer-heldout patients.

```text
$OUT/handoff/evidence.jsonl
  Stage 1 evidence from:
  - text models and matched pairs
  - TF-IDF topics and n-grams
  - neural queries
            |
            | all_evidence_fusion allowlisting
            | + exact fold-local deduplication
            | + cached-embedding / lexical clustering
            v
$OUT/stage2/evidence_compilation/
  packets.jsonl                 compact prompt cards
  outer_NNN/cards.jsonl         readable card inventory
  outer_NNN/members.jsonl       exact members -> Stage 1 paths
  outer_NNN/lineage.jsonl       card -> exact member IDs
  summary.json                  reduction audit
            |
            | prompt-size batching
            v
+---------------------- ONE OUTER FOLD --------------------------+
|                                                               |
|  input_packets.jsonl                                          |
|       |                                                       |
|       | group by evidence architecture, then prompt-size batch|
|       v                                                       |
|  Parallel LLM interpretation                                  |
|  packets ---> candidate concepts + grounded packet citations  |
|       |                                                       |
|       v                                                       |
|  interpreted_candidates.json                                  |
|       |                                                       |
|       | cross-architecture deduplication and consolidation     |
|       v                                                       |
|  feature_definitions.json                                     |
|  operational variables:                                      |
|  name, type, categories/unit, extraction rules, causal role   |
|       |                                                       |
|       +---- original OUTER-TRAINING patient notes             |
|       v                                                       |
|  LLM patient-level feature extraction                         |
|       |                                                       |
|       v                                                       |
|  Training feature matrix + missingness/variation summaries    |
|       |                                                       |
|       +---- treatment/outcome + inner-fold splits             |
|       v                                                       |
|  Empirical performance review                                 |
|       |                                                       |
|       v                                                       |
|  LLM keep / drop / revise decision --------+                  |
|       ^                                    |                  |
|       +----- re-extract if revised --------+                  |
|                                                               |
|  final_definitions.json   <- definitions are now frozen       |
|       |                                                       |
|       +---- OUTER-HELDOUT patient notes                       |
|       v                                                       |
|  Heldout feature extraction                                   |
|       |                                                       |
|       v                                                       |
|  Fit nuisance/effect models on training rows only             |
|  Predict propensity, mu0, mu1, and CATE on heldout rows       |
|       |                                                       |
|       v                                                       |
|  estimation/predictions.csv + diagnostics.json                |
+---------------------------------------------------------------+
            | repeat independently for every outer fold
            v
cross_fitted_predictions.csv
causal_estimate.json
features_by_outer_fold.jsonl
summary.json
```

## Inputs and outputs

| Stage | Main inputs | Main outputs |
|---|---|---|
| Evidence compilation | Stage 1 handoff rows plus the existing memory-mapped Stage 1 chunk-embedding cache, when available | Fold-local semantic cards, exact-member and raw-path lineage manifests, a reduction audit, and bounded prompt packets |
| Interpretation | One architecture-specific packet batch plus the clinical question | Candidate clinical concepts, packet dispositions, and citations to supplied packet IDs |
| Candidate assembly | Concepts from all interpretation batches | `interpreted_candidates.json`; evidence axes are recomputed from citations and mapped to possible causal roles |
| Consolidation | Candidates from all Stage 1 architectures | Deduplicated operational definitions in `feature_definitions.json` |
| Training extraction | Operational definitions plus outer-training patient text | A patient-by-feature matrix for the training patients |
| Review | Training-only extraction summaries and inner-fold treatment/outcome performance | Keep, drop, or measurement-revision decisions; a revision may trigger another extraction round |
| Freeze and heldout extraction | Final reviewed definitions plus outer-heldout text | `final_definitions.json` and heldout feature measurements |
| Fold estimation | Training/heldout features, treatment, outcome, and split provenance | Heldout propensity, `mu0`, `mu1`, AIPW score, estimated CATE, and fold diagnostics |
| Cross-fold aggregation | Exactly one heldout prediction for every patient | Cross-fitted ATE, standard error, 95% CI, mean estimated CATE, and feature stability counts |

Stage 2 keeps the outer folds separate throughout feature discovery and review.
Treatment/outcome performance used to review definitions is calculated within
the outer-training data. Feature definitions are frozen before heldout feature
extraction and heldout outcome estimation.

Potential causal roles are derived from the evidence supporting each feature:

- Treatment plus outcome evidence supports a potential confounder role.
- Outcome evidence supports a prognostic role.
- Residual-effect or matched-pair evidence supports a potential effect-modifier
  role.

## Evidence compilation before LLM interpretation

The default compiler is `semantic_cluster_cards_v2`. It reconnects the plain
Stage 2 route to the audited scientific projections in `all_evidence_fusion`:
large TF-IDF score arrays and operational diagnostics do not enter the prompt,
while topic terms, orphan n-grams, sparse terms, retrieved clinical text, HTR
attention evidence, and neural-query evidence do.

Reduction happens independently inside each outer fold:

1. Normalize the concept-bearing content and remove exact duplicates while
   unioning source families, evidence axes, polarity, full/inner-fold support,
   and raw JSON paths.
2. Stratify exact members by evidence kind, axes, polarity, and semantic-vector
   availability. This prevents a treatment-only term from being clustered into
   an unrelated residual-effect group merely because their wording overlaps.
3. Reuse the Stage 1 chunk embeddings by memory map for compatible retrieved
   chunks. Other evidence uses deterministic character n-gram projections; no
   second embedding model is loaded beside vLLM.
4. Cluster within each stratum and produce conservative cards containing
   representative text, support/stability counts, source families, axes, and
   score ranges. The complete raw evidence remains in Stage 1, and the manifests
   preserve the route from every card back to every raw occurrence.

The default ceiling is `evidence_max_cards_per_fold=400`, with up to four
representatives per card. This is deliberately an oversampled evidence atlas,
not the later variable limit. The LLM may recover many concepts from each card,
and `max_candidates_per_fold` is applied only after interpretation during
cross-card consolidation. If a sensitivity analysis is warranted, raise the
card ceiling rather than reverting to raw packetization. `raw_packets_v1`
remains available as an explicit comparator via `stage2.evidence_compiler`.

Compilation is cached under `stage2/evidence_compilation`. A restart hashes the
Stage 1 handoff, loads the compact packet cache when its compiler signature
matches, and avoids reparsing and reclustering the raw evidence. Interpretation
checkpoints also carry an input fingerprint: completed batches are reused only
when their exact card inputs and clinical question match.

## Progressive consolidation

When interpreted candidates cannot fit into one consolidation prompt, the
reducer partitions them into prompt-sized batches. Its first pass retains an
oversampled intermediate beam of up to

```text
consolidation_oversample_factor * max_candidates_per_fold
```

features across all shards, bounded by the number of candidates actually
available. The default oversample factor is 4. The beam slots are allocated
proportionally to each shard's candidate count using the full available budget,
and every nonempty shard receives at least one slot. Results are interleaved
round-robin so subsequent prompts compare candidates originating from different
shards. Later rounds halve the intermediate pool toward the final fold limit;
the final patient-level extraction cap remains `max_candidates_per_fold`.

This makes a one-feature shard limit possible only when the shard itself has one
candidate or the number of prompt shards exhausts the available beam. It is a
prompt-budget safeguard rather than a scientific conclusion that the evidence
supports only one feature.
