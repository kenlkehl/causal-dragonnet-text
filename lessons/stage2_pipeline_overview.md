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
|       +---- configured explicit features + supplied ontologies|
|       | local pairwise alias judgments                         |
|       | + causal-role routing                                  |
|       | + one global name-only merge-directive pass            |
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
| Interpretation | One architecture-specific batch projected to prompt-local item numbers and readable text only | Candidate clinical concepts and citations to supplied item numbers; Python maps these back to packet provenance |
| Candidate assembly | Concepts from all interpretation batches | `interpreted_candidates.json`; evidence axes are recomputed from citations and mapped to possible causal roles |
| Consolidation | Candidates from all Stage 1 architectures plus optional `stage2.explicit_features` | Deduplicated operational definitions in `feature_definitions.json`; configured groups use their supplied ontology |
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
not a variable limit. The LLM may recover many concepts from each card. If a
sensitivity analysis is warranted, raise the card ceiling rather than reverting
to raw packetization. `raw_packets_v1` remains available as an explicit
comparator via `stage2.evidence_compiler`.

Before interpretation, Python strips each card to
`{"item": N, "text": ["..."]}`. The integer is local to that request and is
mapped back to the original packet immediately after validation. The LLM does
not see packet or card IDs, evidence kind, detail objects, truncation flags,
axes, polarity, semantic grouping, architectures, scores, support counts,
folds, or other compiler metadata. It returns feature names, descriptions,
rationales, and `supporting_items`; it does not choose value types or any other
part of the extraction ontology during discovery.

Compilation is cached under `stage2/evidence_compilation`. A restart hashes the
Stage 1 handoff, loads the compact packet cache when its compiler signature
matches, and avoids reparsing and reclustering the raw evidence. Interpretation
checkpoints also carry an input fingerprint: completed batches are reused only
when their exact card inputs and clinical question match.

## Candidate consolidation

Optional investigator-specified features enter consolidation here rather than
bypassing it. This lets pairwise and global alias review merge a discovered
synonym into one configured feature. The configured feature's name, causal
roles, and complete extraction ontology remain authoritative, so its group
skips model-authored ontology definition while retaining any Stage 1 provenance
carried by the merged candidates. Training-fold diagnostics are still computed,
but review must keep the feature without revising its supplied ontology.

Consolidation first uses generic fuzzy signals to identify candidate pairs that
may be aliases. It retains bounded local neighbors plus a maximum-similarity
spanning forest for every plausible-alias component, ensuring that large alias
families remain connected for semantic review. The LLM judges one pair at a
time as the same or different scalar measurement. Python constructs transitive
groups from accepted pairs and prevents a group merge when an explicit negative
pair judgment would make it contradictory. A discovered candidate with exactly
the same normalized name as one configured feature is the one deterministic
identity case and joins that configured group directly.

Python then excludes groups without an evidence-supported causal role and makes
one global LLM request containing only the complete list of remaining unique
feature names. The response contains only residual merge directives of the form
`inputs -> output`; it contains no opaque IDs and does not enumerate unchanged
features. Python resolves the names back to internal groups, rejects unknown or
overlapping inputs, unions provenance for each accepted directive, and passes
all groups absent from `inputs` through unchanged.

All groups remaining after this residual semantic deduplication are
operationalized one at a time. The ontology model sees the canonical feature
name and a deduplicated flat list of `representative_evidence.text` strings.
It does not see packet boundaries, evidence kind, detail objects, truncation
flags, fold metadata, internal IDs, causal axes, semantic grouping, architecture
names, scores, support counts, candidate summaries, or an earlier proposed
value type. It chooses the value type and supplies allowed values or a unit
based on the readable text. Python keeps provenance and causal-role routing
outside the prompt and validates ontology shape without encoding domain-specific
clinical answers. There is no diversity-ranking prompt or feature-count pruning
step. The legacy `max_candidates_per_fold` and
`consolidation_oversample_factor` configuration fields remain readable only so
existing run files continue to parse.
