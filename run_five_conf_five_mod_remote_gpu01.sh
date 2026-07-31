#!/usr/bin/env bash
#
# Shared-/data1 entry point for the corrected five-confounder production run.
# All substantive launcher logic executes from the immutable source snapshot.

set -Eeuo pipefail
IFS=$'\n\t'

readonly REPO_ROOT="/data1/ken/pcori_dev/causal-dragonnet-text"
readonly SNAPSHOT_ROOT="${REPO_ROOT}/artifacts/production_source_snapshot_20260730_token_attention_htr_stage2_complete_semantic_catalog_fast_stat_auth_reusable_preflight_v8"
readonly SNAPSHOT_LAUNCHER="${SNAPSHOT_ROOT}/scripts/run_five_conf_five_mod_remote_gpu01.sh"

if [[ ! -f "${SNAPSHOT_LAUNCHER}" || -L "${SNAPSHOT_LAUNCHER}" ]]; then
    printf 'ERROR: immutable token-attention launcher is absent: %s\n' \
        "${SNAPSHOT_LAUNCHER}" >&2
    exit 1
fi

exec /usr/bin/bash "${SNAPSHOT_LAUNCHER}" "$@"
