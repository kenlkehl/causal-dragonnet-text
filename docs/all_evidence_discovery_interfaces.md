# All-evidence feature-discovery interfaces

Status: offline specification for the post-v24 discovery path.

This document defines the scientific decisions shown to a language model. Split
identities, hashes, cache keys, producer identities, row lineage, and transport
metadata remain in an authenticated machine envelope and are not rendered as part
of these jobs.

The discovery path must cover every active concept-bearing Stage 1 family, but
must never concatenate all of their raw evidence into one discovery prompt:

- sparse treatment/outcome nuisance evidence;
- sparse residual-effect (R-loss) evidence;
- HTR treatment, outcome, residual-effect, and matched-pair phrases;
- sparse and HTR matched-pair uplift evidence;
- whole-cohort embedding contrasts;
- clustered embedding contrasts;
- TF-IDF terms derived from semantic-retrieval contrasts;
- TF-IDF treatment, outcome, and effect topics;
- TF-IDF residual/orphan n-grams; and
- learned neural-query witnesses and aggregate moments.

Authenticated row-level numerical outputs from these families form a separate
direct-numerical channel. Aggregate summaries may be shown as non-grounding
context, and row-level values may be routed to the final forest, but neither can
name a patient feature. The inactive sparse-query fallback is not part of this
benchmark path.

The exact production wording for the ten architecture explanations lives in
`oci/inference/stage1_architecture_explanations.py`. It explains, in ordinary
scientific language, what each architecture's words, phrases, topics, contrasts,
or witnesses mean. The mapping is complete and follows the canonical family order;
callers should not substitute family names alone as explanations.

## Shared rules

Every model-facing request uses JSON evidence items with stable `evidence_id`
values. A response may name a feature only when it cites concept-bearing evidence
whose visible words, phrases, topics, or semantic witnesses support that name.
Numerical summaries alone cannot ground a name.

Model-facing JSON contains only scientific context and the requested output shape.
It does not contain `schema_version`, catalog or coverage digests, cache or producer
identities, split fingerprints, numerical-manifest digests, or deterministic role
routing. Those fields remain in the content-addressed job envelope and input
bindings. Candidate, evidence, and member IDs remain visible because the model
needs them for exact citations and complete dispositions.

The model may infer which specific patient characteristic several clues describe.
It may not invent a benchmark variable, unsupported alias, unit, category, patient
fact, or treatment-effect direction. Discovery is deliberately role-independent:
the response contains no confounder or effect-modifier field.

Every supplied item receives exactly one disposition. JSON validation and one
bounded repair attempt fail closed. The exact initial and repair message sequences
are separately context-checked, hashed, and bound to their cache records.

Every model-facing field that must cover an exact authenticated identifier domain
uses a closed JSON object keyed by those identifiers, with every key required and
additional properties forbidden. It does not use an exact-length enum-valued
array: such an array can repeat one legal identifier while omitting another. After
wire validation, the implementation deterministically normalizes keyed objects to
request-ordered internal lists for existing downstream consumers. The canonical
raw wire object and normalized validated projection have separate SHA-256 bindings
in the response-attempt trace, runner metadata, and immutable cache; cache replay
reruns the same normalizer from the stored wire object.

The initial sequence is exactly `system, user`. If strict JSON parsing or the
job-specific closed semantic validator fails, the only admitted repair sequence
is exactly `system, user, assistant, user`: the first two messages are unchanged;
the assistant message is a fixed category-specific notice that the invalid content
was intentionally omitted; and the final user message is one fixed prompt selected
from the trusted failure category. The failed transport text or canonical parsed
object is reduced to a SHA-256 in the hidden repair binding and is neither sent back
to the model nor persisted. Exception text, model-written diagnostics, and
model-proposed identifiers are never inserted into the repair sequence. The model
must reconstruct the response using only identifiers appearing verbatim in the
original request. The cumulative four-message sequence must independently pass the
fixed byte guard. A second invalid response exhausts repair, and neither invalid
response is cacheable; only the validated output, the hidden failed-response hash,
and the independently reconstructable one- or two-attempt job identity enter the
immutable cache.

## Hierarchy and raw-evidence lookback

The pipeline uses five explicit levels:

1. An authenticated atom catalog records every concept-bearing item without a
   global top-k. Each atom belongs to one canonical architecture and retains its
   observable axes.
2. Each architecture is interpreted independently. If one architecture is too
   large, it is divided into complementary chunks; all chunks together cover the
   architecture, and no other architecture is mixed into those prompts.
3. Chunk candidates are consolidated and coverage-criticized *within that
   architecture*. The result is an architecture dossier containing its complete
   candidate ledger, candidate-to-evidence citations, evidence dispositions,
   ambiguity, and coverage audit. It does not contain the entire raw atom payload.
4. The cross-architecture stage compares every candidate pair on bounded compact-
   descriptor pages and deterministically compiles complete-link groups. It does
   not ask a model to select a subset of raw evidence IDs.
5. For every provisional group, the integrator reviews each exact authenticated
   support atom on its own page and recursively folds all page judgments. Every
   candidate is then preserved either in a canonical concept or in an explicit
   rejection ledger. Rejection reconsideration and extraction use the same
   one-support-page plus recursive-fold pattern.

Thus all architectures contribute, while raw evidence is disclosed locally one
item at a time instead of being dumped into one prompt. Every page and fold is
ID-addressed, context-checked, hashed, and included in the frozen audit. No
configured legacy lookback bound samples or truncates semantic support. A per-
architecture coverage manifest proves that an architecture was not merely named
but actually interpreted.

## Job 1: interpret one complementary evidence chunk

### Mission

Inspect every supplied clue and identify concrete patient characteristics encoded
by related words, phrases, topics, or semantic witnesses. Build a broad inventory
before ranking. Do not decide causal roles or estimate treatment effects.

### Input

The user message contains:

```json
{
  "job": "interpret_evidence_chunk",
  "family_explanation": "Plain-language explanation of only the supplied family.",
  "evidence": [
    {
      "evidence_id": "evidence_0001",
      "source_family": "one canonical family",
      "observable_axes": ["treatment|outcome|heterogeneity|pair_uplift|extraction_support"],
      "member_ids": ["Stable IDs for every term, phrase, or witness inside this item."],
      "content": {}
    }
  ]
}
```

`observable_axes` explains why Stage 1 surfaced the clue. It is not a causal-role
label. Evidence derived from more than one axis lists all applicable axes.

### Output

Return exactly this JSON shape, with JSON only:

```json
{
  "concepts": [
    {
      "feature_name": "snake_case_name",
      "description": "One sentence describing what would be measured in one patient.",
      "value_shape_hypothesis": "continuous|categorical|ambiguous",
      "supporting_evidence_ids": ["evidence_0001"],
      "unresolved_ambiguity": "Empty string when there is none."
    }
  ],
  "evidence_dispositions": {
    "evidence_0001": {
      "status": "supports_concept|reviewed_no_specific_concept",
      "feature_names": ["snake_case_name"],
      "member_dispositions": {
        "member_0001": {
          "feature_names": ["snake_case_name"]
        }
      },
      "reason": "Short evidence-based reason."
    }
  }
}
```

Every supplied `evidence_id` is a required key in `evidence_dispositions`, and
every supplied `member_id` is a required key inside its parent's closed
`member_dispositions` object. A member cites every returned feature it supports, or an
empty list after it was reviewed but found non-specific. The parent
`feature_names` is the exact union of its member assignments. A
`supports_concept` disposition cites one or more returned feature names. A
`reviewed_no_specific_concept` disposition cites none and explains why the clue is
too broad, administrative, ambiguous, or otherwise non-specific. Every concept
must cite at least one supplied concept-bearing ID. This child-level accounting
prevents a response from claiming to review a 15-term topic while silently
ignoring most of its terms.

## Job 2a: consolidate complementary chunk inventories within one architecture

### Mission

Merge only spelling, abbreviation, and formatting aliases while preserving every
candidate and every supporting evidence ID. Keep distinct specific variables
separate even when they share a broad clinical theme. Do not select a small final
feature set.

### Input

The user message contains the complete validated chunk-candidate ledger for
exactly one architecture. Each candidate has an opaque `candidate_id`, its name
and description, all evidence citations, its source family, value-shape
hypothesis, and unresolved ambiguity. It also contains that architecture's full
evidence-disposition index, but no evidence from another architecture.

### Output

Return exactly this JSON shape, with JSON only:

```json
{
  "canonical_concepts": [
    {
      "canonical_name": "snake_case_name",
      "description": "One patient-level measurement.",
      "value_shape_hypothesis": "continuous|categorical|ambiguous",
      "unresolved_ambiguity": "Empty string when there is none."
    }
  ],
  "candidate_dispositions": {
    "candidate_0001": {
      "canonical_name": "snake_case_name",
      "reason": "Alias merge or keep-distinct reason."
    }
  }
}
```

Every input candidate is a required key in `candidate_dispositions`. The
deterministic normalizer groups dispositions by canonical name in request order
and derives `member_candidate_ids`, `supporting_evidence_ids`, and
`source_families` from the authenticated candidate ledger. The resulting internal
union must equal the union carried by its members; consolidation cannot drop
support or introduce new support.

## Job 2b: evidence-coverage critic

### Mission

Compare the complete supplied evidence inventory with the consolidated ledger.
Find evidence with no adequate concept, specific variables hidden by an overly
broad merge, family disagreement erased during consolidation, or lower-ranked
evidence lost between passes.

### Output

Return exactly this JSON shape, with JSON only:

```json
{
  "findings": [
    {
      "action": "add_concept|split_concept|restore_support|no_change",
      "affected_canonical_names": ["snake_case_name"],
      "proposed_name": "snake_case_name_or_empty",
      "description": "Patient-level measurement or empty for no_change.",
      "supporting_evidence_ids": ["evidence_0001"],
      "reason": "Short evidence-based reason."
    }
  ],
  "reviewed_evidence_ids": {
    "evidence_0001": true
  }
}
```

The within-architecture critic must review every concept-bearing ID cataloged for
that architecture. Additions and splits must cite existing evidence. A separate
rejection critic receives rejected candidates and rejection reasons; it is not
combined with omission detection.

## Job 2d: build one architecture dossier

This is deterministic and makes no model call. It joins the validated
within-architecture concepts, all candidate and evidence dispositions, unresolved
ambiguities, critic findings, catalog counts and hashes, and the availability of
that architecture's direct numerical channel. Numerical availability is bound to
an authenticated manifest hash and exact signal count; a zero count requires an
explicit reason (for example, the semantic-retrieval adapter shares its parent
embedding projection and has no independent numerical signal). A boolean alone is
not sufficient. The dossier includes citations and short descriptions, not the raw
evidence payload. A dossier is invalid unless its catalog coverage is complete.

The model-facing projection includes the scientific catalog and disposition counts,
the completeness flag, the direct numerical signal count and availability, and any
zero-signal explanation. Catalog, coverage, and numerical-manifest hashes remain in
the authenticated dossier identity and job binding; they are never rendered in the
planner or integrator messages.

## Job 3a: plan cross-architecture integration and lookback

### Mission

Compare the architecture dossiers. Propose only alias/formatting merges, flag
specific disagreements or broad concepts that may hide distinct measurements,
and request the minimum raw evidence needed to adjudicate them. Do not make final
rejections during this pass.

### Output

Return JSON only:

```json
{
  "provisional_groups": {
    "candidate_0001": {
      "provisional_name": "snake_case_name",
      "reason": "Why these dossier candidates may or may not be aliases."
    }
  },
  "raw_evidence_requests": [
    {
      "evidence_ids": ["evidence_0001"],
      "question": "Specific ambiguity the raw evidence can resolve.",
      "reason": "Why dossier-level information is insufficient."
    }
  ]
}
```

Every dossier candidate is a required key in `provisional_groups`. The
deterministic normalizer groups identical provisional names into the internal
request-ordered group list. Requested IDs must already be cited by members of the
relevant dossiers. A deterministic resolver returns only exact catalog entries for
requested IDs; it cannot perform semantic search, add evidence, or expose
row-level numerical data.

## Job 3b: integrate across architectures with exhaustive support pages

For each compiler-derived provisional group, one review job receives exactly one
raw support atom. Recursive fold jobs combine at most eight prior page/fold inputs
at a time, with at most seven fresh inputs after an accumulator exists. Every fold
returns a closed disposition keyed by every exact input ID. The terminal fold
accepts or rejects the group only after the complete support union has been
reviewed. Canonical membership, support, and source families remain compiler-owned,
so no candidate or architecture can disappear during a merge.

There is no model-selected final lookback. Every authenticated support ID is
scheduled, and an unresolved ambiguity remains explicit rather than causing
evidence sampling, invention, or silent deletion.

## Job 2c: rejection critic

### Mission

Review every explicitly rejected candidate with its complete backing evidence and
rejection reason. Determine whether evidence disappeared, distinct variables were
mistaken for duplicates, or discovery was prematurely confused with role or
extraction decisions.

### Output

Return exactly this JSON shape, with JSON only:

```json
{
  "reconsiderations": {
    "candidate_0001": {
      "decision": "uphold|restore|split",
      "proposed_name": "snake_case_name_or_empty",
      "supporting_evidence_ids": ["evidence_0001"],
      "reason": "Short evidence-based reason."
    }
  }
}
```

Every rejected candidate is a required key and receives one decision. Restoration
or splitting must cite that candidate's supplied concept-bearing evidence.

## Job 4: deterministic downstream role routing

Causal roles are not requested during discovery. After the feature is grounded,
the pipeline deterministically summarizes the observable axes represented in its
support:

- treatment and outcome evidence together can support adjustment as a confounder;
- outcome evidence without treatment evidence supports a prognostic adjustment
  role, not confounder status;
- residual-effect, orthogonal-effect, or matched-pair heterogeneity evidence
  supports an effect-modifier role;
- treatment-only evidence is retained as treatment-prediction support but does not
  mechanically become a confounder; and
- extraction-support evidence can improve a definition but does not create a
  causal role.

Overlapping supported roles are retained. The routing audit records every source
axis and rule that produced the result. A later adequacy reviewer may question
whether the resulting observable adjustment and heterogeneity banks are sufficient,
but it may not invent an unsupported feature.

## Job 5: generate one extraction definition

### Mission

Define exactly how to read one already-grounded patient characteristic from a
complete clinical record. Do not discover new features or perform causal reasoning.

### Input

The user message contains one canonical concept, its complete concept-bearing
evidence, and any validated aliases or value clues. It contains no treatment or
outcome values, numerical Stage 1 row values, or causal-role discussion.
Observable-axis routing and the deterministic routing audit stay in the internal
job binding and are not rendered for the extraction model.

### Output

Return exactly this JSON shape, with JSON only:

```json
{
  "feature_name": "snake_case_name",
  "measurement": "Exact patient-level value to read.",
  "representation": {
    "kind": "continuous|categorical|unresolved",
    "unit": "Supported unit or empty string.",
    "categories": ["Only evidence-supported categories."]
  },
  "aliases": ["Only evidence-supported aliases."],
  "distinguish_from": ["Nearby concepts that the evidence actually identifies."],
  "missing_or_ambiguous": "Exact value to return when absent or ambiguous.",
  "supporting_evidence_ids": {
    "evidence_0001": true
  }
}
```

The output must preserve the canonical name and cite supplied evidence. Continuous
definitions require a supported unit or an explicit unitless statement. Categorical
definitions require non-empty evidence-supported categories. If the evidence does
not support a valid representation, return `unresolved`; the definition cannot be
used for extraction until a bounded, evidence-grounded repair succeeds.

Extractor inference uses this frozen definition, returns structured JSON only, and
runs with reasoning disabled.

## Adaptive review must remain hierarchical

Post-extraction review must not receive either of these lossy extremes:

- only the raw atoms that supported features accepted by the initial selector; or
- one concatenation of every raw atom from every architecture.

The first option prevents review from recovering a construct that the initial
selector missed or that becomes visible only after a validation gate has been
consumed. The second repeats the overloaded-prompt failure that motivated this
hierarchy.

For review round 1, the exact initial-spent catalog and its ten completed
architecture dossiers are reused. After a gate has been consumed, Stage 1 is
authenticated to the new accumulated-spent scope. Every architecture is then
interpreted and coverage-checked separately against that new catalog before the
next review proposal is formed. An exact-scope cache hit may replace a refit; a
full-outer artifact or an artifact trained on a still-sealed gate may not.

Review uses a phased bounded hierarchy rather than two overloaded prompts.

### Adaptive phase A: exhaustive diagnostic/evidence planning pages

The compiler schedules bounded pages across every current/new-missing target,
authenticated evidence item, current candidate, and observable diagnostic that
requires consideration. Each atomic response addresses only its supplied page.
Recursive folds combine all page decisions and require an explicit disposition
for every input. Unknown or model-written evidence IDs are rejected. The retained
legacy per-target/total ID and byte fields are audit compatibility metadata; they
do not remove pages or cap semantic evidence.

### Adaptive phase B: proposal pages, judgments, and complete-link compilation

For every compiled review target, the proposer schedules singleton and merge-pair
pages over all requested authenticated atoms. A page may emit at most one drop,
merge, split, rename, definition repair, or evidence-grounded addition. Every
emitted proposal receives its own judgment before grouping. Pairwise relation
pages and recursive definition folds compile accepted proposals with complete-link
semantics, so an uncertain or distinct relation cannot silently merge them.

Only after every page and proposal has an explicit disposition may the configured
round capacity choose executable operations. Proposals beyond that capacity are
recorded as explicit capacity rejections, not sliced from the audit. Each accepted
operation's extraction bridge then reviews every exact support item on its own
page and recursively folds all reviews into the executable definition. An
addition must cite current exact-scope evidence; a definition repair may also cite
authenticated retained-feature support. Deterministic role routing is recomputed
from evidence axes after grounding. The complete proposal is schema-checked and
frozen before the next one-use gate is transformed or its labels are read.

The adaptive audit binds the accumulated-spent row scope, current catalog and
chunk-plan hashes, all ten dossier hashes, prior registry hash, diagnostic hash,
page schedules, recursive-fold records, proposal/disposition hashes, and the still-sealed gate
fingerprint. Round 2 cannot silently reuse round 1's semantic catalog when the
allowed training scope has changed.

## Context and transport invariants

- Complementary chunks cover every cataloged concept-bearing atom at least once;
  there is no global top-k before discovery. Chunks never mix architectures.
- Cross-architecture prompts contain bounded compact-descriptor comparison pages
  or one exact ID-addressed raw-support item, never an indiscriminate concatenation
  of all raw evidence.
- Large semantic member collections are partitioned by their closed source
  adapters into complete authenticated member batches; every member is audited
  exactly once and arbitrary JSON-fragment splitting is forbidden.
- Every cumulative initial or repair message sequence is checked against the fixed
  context guard before transport.
- Every system/user message array is serialized as canonical JSON and UTF-8, then
  its exact SHA-256 and byte count are stored in the authenticated job binding. A
  fixed 220,000-byte guard is checked before the runner can make a transport call;
  the limit is part of the reviewed hierarchy configuration and cannot be raised by
  a caller.
- Selector/discovery inference uses exactly 5,000 reasoning tokens. Extraction
  reasoning is disabled.
- Raw vectors, row-aligned numerical values, patient identifiers, full notes,
  backend paths, oracle fields, and not-yet-consumed validation information are
  forbidden from discovery prompts.
- Adaptive review sees all ten architectures through compact dossiers and bounded
  ID-addressed lookback. It never substitutes an accepted-support-only catalog for
  omission discovery and never dumps all raw architectures into one prompt.
- Prompt changes create new precommits, cache namespaces, and output paths. They do
  not mutate historical controls or require recomputing an authenticated exact-scope
  Stage 1 cache.
