# Repository Component Map

This repository contains reusable evidence, extraction, review, and causal-forest components. Use them as building blocks for the skill workflow. Do not treat any module or runner name as the owner of discovery decisions: the invoking agent must inspect evidence, define candidate variables, review failures, and write the final synthesis.

## Shared Contracts

- `oci/config.py`
  - `ExplicitFeatureSpec`: one extractable patient-level concept with `name`, `type`, optional `categories`, optional `value_aliases`, `description`, and causal `roles`.
  - `ExplicitFeatureExtractionConfig`: endpoint-backed extraction settings, batching, retries, cache settings, and output-token/text-length controls.
  - `BoWViewConfig` and `default_multi_model_bow_views()`: sparse text-model view definitions.
  - `EmbeddingContrastDiscoveryConfig`: embedding model, chunking, cache, and contrast settings.
  - `AgenticAttentionVariableForestConfig`: HTR nuisance/effect, attention, attribution, and fold settings.
  - `ExplicitFeatureForestConfig`: final explicit-feature causal-forest settings.

Keep feature specs role-tagged. Confounder-role features become adjustment/control variables; effect-modifier-role features become heterogeneity variables. A feature may have both roles.

## BoW Modeling

Use BoW/TF-IDF as the broad lexical discovery pass. It should produce fold-specific evidence for treatment assignment, outcome prediction, residual structure, R-loss, and pseudo-outcomes.

Relevant repo pieces:
- `oci/inference/multi_model_agentic_forest.py`
  - sparse cross-fitted nuisance/effect helpers
  - BoW view handling via `BoWViewConfig`
  - feature-importance and phrase-consensus utilities
- `oci/config.py`
  - `BoWViewConfig`
  - `default_multi_model_bow_views()`

Recommended evidence views:
- unigram-focused view for broad clinical concepts
- broad `1-3` n-gram view
- phrase-focused `2-4` n-gram view
- rare-signal-friendly view with lower `min_df`, higher `max_features`, or alternate `sublinear_tf`
- learner sensitivity when runtime permits: `linear`, `extratrees`, `random_forest`, or `xgboost`

Useful artifacts and outputs:
- fold-level out-of-fold nuisance predictions
- treatment/outcome/R-loss/pseudo-target feature importances
- phrase consensus across folds and vectorizer views
- view names and vectorizer parameters for every reported term

Use BoW outputs to propose candidate concepts, not as final structured variables.

## Embedding Contrasts

Use embedding contrast to retrieve real text chunks aligned with contrasts that sparse n-grams may obscure. The evidence is chunk-level and contrast-level; it still needs translation into explicit patient-level variables.

Relevant repo pieces:
- `oci/inference/embedding_contrast_discovery.py`
  - `EmbeddingContrastEvidenceGenerator`
  - `redact_embedding_contrast_evidence()`
- `oci/models/concept_embedding_utils.py`
  - `chunk_text_words()`
- `oci/models/concept_embedding_cache.py`
  - `ConceptEmbeddingCache`

Useful contrast families:
- treatment
- outcome
- per-view R-pseudo-target
- ensemble R
- within-arm outcome
- treatment-outcome cell
- orthogonal R-score
- concept probes derived from BoW phrases or agent-authored candidate concepts

Useful evidence fields:
- retrieved chunk text
- contrast name and direction
- chunk/patient provenance
- concept-probe AUC or similarity when available
- embedding model, chunking, cache, and residualization settings

Use embedding chunks to identify extractable baseline concepts from real text. Under the default synthetic-task assumption, supplied clinical text is pre-treatment/pre-outcome unless the user states otherwise, so do not reject chunks merely because they mention treatment planning, regimen history, prognosis, or outcome-associated severity. For outcome, R, orthogonal, within-arm, and treatment-outcome contrasts, explicitly inspect chunks for numeric values, laboratory panels, counts, ratios, treatment-history/regimen evidence, status categories, eligibility markers, and derived quantities.

## HTR Modeling And Attribution

Use HTR modeling twice in each discovery iteration: first to localize spans that drive nuisance/confounding structure, then again to localize spans that drive effect/R-stage heterogeneity. HTR evidence is especially useful for numeric slots, nearby value labels, treatment-history/regimen spans, status categories, derived quantities, and repeated longitudinal mentions. Both nuisance and effect/R-stage HTR evidence are required by the skill gates.

Relevant repo pieces:
- `oci/inference/agentic_attention_variable_forest.py`
  - `AgenticAttentionVariableForestRunner`
  - nuisance/effect cross-fitting
  - attention row generation
  - token/span attribution helpers
- `oci/inference/multi_model_agentic_forest.py`
  - `MultiModelHTREvidenceProvider` adapter for reusing HTR nuisance/effect predictions in a sparse-text workflow
- `oci/models/extractor_factory.py`
  - `create_feature_extractor()` and `create_feature_extractor_from_config()` for configured text extractors

Useful HTR outputs:
- out-of-fold nuisance predictions
- out-of-fold effect/R-stage predictions
- attention evidence tables for nuisance and effect stages
- top chunks and compact token spans
- token/span attribution scores and attribution target labels

Use HTR outputs to:
- localize baseline/index-time facts under the task-level pre-treatment/pre-outcome text assumption
- locate the note section and local context for a candidate value
- identify lab labels, values, categories, and temporal qualifiers
- distinguish true clinical signals from note templates, copied histories, or broad family labels that need more specific candidate translation
- support effect-modifier candidates when BoW evidence is unstable

Before GPU-backed HTR runs, record the interpreter, CUDA visibility, device inventory, memory headroom, and worker/device mapping. If GPU access is unavailable or inconsistent, especially when `nvidia-smi` or equivalent system probes show devices but the framework reports no CUDA devices, use the active harness approval/escalation mechanism for the GPU probe and HTR command before declaring GPU unavailable. If GPU remains unavailable, run the smallest honest HTR/neural attention or hidden-state attribution pass that still produces both nuisance and effect/R-stage evidence, using CPU only for the same class of neural HTR workflow. A nuisance-only pass is a failed HTR gate, not a completed HTR stage.

Do not substitute sparse-text evidence for HTR. BoW/TF-IDF models, linear/logistic/Ridge coefficient chunk scoring, dense TF-IDF/SVD chunk retrieval, embedding/concept probes, or generic chunk localization are not HTR modeling or HTR attribution, even when fold-honest and targeted at R-pseudo-outcomes. Record those artifacts under BoW or embedding evidence. If actual HTR/neural attention or hidden-state attribution cannot be produced after documented escalation and narrow neural retries, mark `htr_evidence_gate` `blocked_after_retries` and do not pass final preflight.

## Feature Extraction

After evidence review, materialize only evidence-supported concepts as patient-level columns. BoW, embedding-contrast, and HTR outputs should guide candidate choice, aliases, temporal anchors, and audit targets. Extraction itself must read the complete patient text, or use a recursive pass covering the complete patient text; regex, short-window parsers, nearby-number rules, category heuristics, and pattern-matching fallback extraction are not acceptable.

Relevant repo pieces:
- `oci/extraction/explicit_features.py`
  - `VLLMFeatureExtractor`
  - `extract_explicit_features()`
  - JSON parsing, categorical alias handling, retries, batching
- `oci/inference/agentic_explicit_feature_forest.py`
  - `VLLMExplicitFeatureExtractionProvider.ensure_features()` for grouped, cached, resumable endpoint-backed extraction
  - feature-spec normalization, alias handling, value harmonization, extracted-feature review utilities
- `oci/extraction/llm_routing.py`
  - endpoint pooling and retry/backoff helpers
- `oci/models/explicit_feature_featurizer.py`
  - `get_raw_explicit_features()` for interpretable raw feature matrices

Extraction contract:
- one row per patient
- one structured value per selected concept
- missingness flag or null when the value is not recoverable from the text
- source/evidence summary when available
- backend and configuration provenance
- role-tagged feature specs preserved alongside values

Endpoint-backed extraction:
- Use OpenAI-compatible/vLLM endpoints only as document-reading extraction backends unless the user explicitly requests external proposal or review.
- Prefer deterministic temperature, retries, caching, bounded concurrency, and a large enough input/output budget for complete JSON extraction.
- Run a small JSON smoke test before full extraction when starting or switching an endpoint.

Agent/harness extraction:
- If no endpoint is available, the invoking agent may perform extraction directly by reading complete patient documents, or by a recursive reading strategy whose sections cover each complete patient document.
- For larger datasets, shard by patient, fold, concept, or concept family and reconcile findings into one patient-level table. Evidence-highlighted chunks may guide candidate selection, attention, and audit targets, but they must not be the only context for accepted extracted values.
- Do not proceed to downstream causal-forest claims with an all-missing table caused by backend absence.
- Extract only after the nuisance/confounder candidate batch and the effect-modifier candidate batch have been harmonized into one extraction specification. In later iterations, re-extract only changed or newly added concepts unless audits require broader retry.

## Post-Extraction Review

Do not treat a successfully extracted feature table as proof that variables are useful. Review extracted features against the upstream text evidence that motivated them.

Relevant repo pieces:
- extracted-feature review utilities in `oci/inference/agentic_explicit_feature_forest.py`
- review and parsimony utilities in `oci/inference/multi_model_agentic_forest.py`
- raw feature construction via `oci/models/explicit_feature_featurizer.py`

Run fold-honest diagnostics:
- extracted-feature treatment nuisance
- extracted-feature outcome nuisance
- R-loss, logistic R-loss, pseudo-target, interaction, or treatment-stratified effect-modifier diagnostics
- missingness, category/value coverage, overlap warnings, and role-specific failures

Compare extracted-feature performance to BoW, embedding-contrast, and HTR benchmarks. Large gaps should trigger spec revision, alias/value harmonization, re-role decisions, targeted additions, or targeted re-extraction before final causal-forest fitting.

When gaps remain after extracted-feature review, loop back to the responsible evidence stage rather than only tuning the final model: nuisance gaps go back to BoW/embedding/HTR nuisance evidence, and effect-modifier gaps go back to BoW R/interaction evidence, embedding R/orthogonal/within-arm evidence, and HTR effect/R-stage attribution.

## Parsimony And Role Review

Before final fitting:
- inspect continuous feature correlations
- inspect categorical contingency tables
- inspect missingness overlap
- test plausible single-feature or grouped removals when the candidate set is not already minimal
- revisit roles after every revision

Record one final decision:
- `retain_all`: current set is already compact or tested removals harm honest diagnostics
- `prune`: redundant or weak variables are removed
- `blocked`: the feature set cannot support a defensible final forest

## Final Causal Forest

Use a real honest causal forest for final ITEs.

Relevant repo pieces:
- `oci/models/causal_forest_head.py`
  - `CausalForestHead`
  - `tune_causal_forest_model()`
- `oci/inference/agentic_explicit_feature_forest.py`
  - `CausalForestExplicitEvaluator`
- `oci/inference/applied_explicit_feature_forest.py`
  - explicit-feature matrix helpers used by forest evaluators

Requirements:
- finalized confounder-role features are controls/adjustment variables
- finalized effect-modifier-role features are heterogeneity variables
- final reported ITEs come from `CausalForestDML` or an equivalent honest causal-forest implementation
- non-causal-forest learners may be reported only as sensitivity diagnostics
- no row-level prediction used for selection or reporting should be trained in-sample for that row unless the report explicitly describes a final refit after nested selection is complete

## Integrated Runners

Integrated runners can save time when their hooks expose the needed evidence and accept externally supplied candidate specs, extraction providers, or evaluators. Use them as execution conveniences, not as conceptual authorities.

When using an integrated runner:
- keep BoW, embedding, and HTR evidence active unless an opt-out is documented
- inspect the produced evidence yourself
- supply or override candidate specs when needed
- avoid autonomous feature proposal/review hooks unless the user explicitly asks for them
- verify that extracted-feature review and parsimony artifacts are written before final forest results are accepted
