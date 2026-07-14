# Canonical feature registry and bounded extraction

## Current registry

The 15-term per-topic labeling batch produced 375 candidate occurrences and
287 case-insensitive exact names. The harmonization pipeline assigned those
names to clinical domains and consolidated them into 152 canonical records:

- 139 directly extractable features;
- 10 records requiring review;
- 3 dropped redundant or vague records.

Every source candidate maps exactly once to a canonical record. Mapping
relations include 167 aliases, 98 base-variable mappings, 8 related-but-
distinct mappings, 5 subfields, 6 review mappings, and 3 drops. The registry
preserves source topic ranks, candidate IDs, aliases, descriptions, types,
categories, parent objects, and rationales.

The current model did not propose an executable derived feature. The registry
and extraction code support ratio and threshold derivations, but do not invent
them when required inputs or clinical thresholds are absent. In particular,
NLR cannot be calculated because no lymphocyte-count candidate was recovered.

## Extraction plan

Extraction is organized within clinical domains and hard-limited to 10
variables per patient request. The current 139 extractable fields form 18
groups:

- ten groups of 10 variables;
- two groups of 7;
- one group of 6;
- one group of 4;
- three groups of 2;
- one group of 9.

For 1,000 patients this is 18,000 LLM requests. The plan is written without
making any server call unless `--execute` is supplied. Row/group results are
checkpointed in SQLite, and each completed group is also materialized as
Parquet.

## Smoke validation

A two-patient smoke run made 36 requests across all 18 groups:

- 36 complete responses;
- 0 failed responses;
- 0 schema issues after category normalization;
- 139 value columns and 139 missingness columns;
- approximately 40% missing values, reflecting features not documented for
  those two patients rather than parsing failures.

The complete 1,000-patient extraction has not been started.
