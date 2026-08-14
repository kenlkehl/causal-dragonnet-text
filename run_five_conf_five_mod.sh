#!/usr/bin/env bash

# Run or resume the complete Stage 1 -> 2 workflow on the bundled
# five-confounder/five-effect-modifier NSCLC cohort.
#
# Usage:
#   ./run_five_conf_five_mod.sh
#   GPU_COUNT=2 ./run_five_conf_five_mod.sh
#   PHYSICAL_GPUS=1,3 STAGE2_ENDPOINT=http://127.0.0.1:8010/v1 ./run_five_conf_five_mod.sh
#   STAGE1_ARCHITECTURES=bow_nuisance,tfidf_topics ./run_five_conf_five_mod.sh
#   STAGE2_CONSOLIDATION_MAX_ROUNDS=12 ./run_five_conf_five_mod.sh
#   ./run_five_conf_five_mod.sh /persistent/results/my_run

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Stage 2 ontology preset for this example. Callers may override any setting
# through the corresponding environment variable.
export STAGE2_CONSOLIDATION_BATCH_SIZE="${STAGE2_CONSOLIDATION_BATCH_SIZE:-20}"
export STAGE2_CONSOLIDATION_ALPHABETICAL_ROUNDS="${STAGE2_CONSOLIDATION_ALPHABETICAL_ROUNDS:-5}"
export STAGE2_CONSOLIDATION_MAX_ROUNDS="${STAGE2_CONSOLIDATION_MAX_ROUNDS:-25}"
export STAGE2_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS="${STAGE2_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS:-3}"
export STAGE2_MAX_ONTOLOGY_REFINEMENT_ROUNDS="${STAGE2_MAX_ONTOLOGY_REFINEMENT_ROUNDS:-2}"

exec "${repo_root}/scripts/run_synthetic_all_evidence.sh" \
    synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet \
    five_conf_five_mod_nsclc_full \
    "$@"
