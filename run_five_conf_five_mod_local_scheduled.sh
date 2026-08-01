#!/usr/bin/env bash

# Local five-confounder/five-modifier Stage 1 production schedule.
#
# Run from this repository with:
#
#   ./run_five_conf_five_mod_local_scheduled.sh
#
# Before 9:00 AM America/New_York on 2026-08-01, the workflow uses physical
# GPUs 0,1,2,3 as four disjoint owner/embedding lanes. At the deadline, if the
# Stage 1 handoff is not already complete, this supervisor sends SIGTERM only
# to its verified workflow process group, waits for it to exit, and starts a
# new production request on physical GPUs 2,3. It never sends SIGKILL.
#
# Both resource schedules are execution epochs of one durable scientific run
# and share one scratch store. The post-deadline epoch reopens the already
# sealed input, embedding, preflight, and Stage 1 work through protected
# proof/stat continuity; it does not rebuild embeddings merely because the GPU
# allocation changed. Incomplete attempts are preserved.
#
# This is a Stage 1-only path through ordinary handoff_validation. It does not
# contact an LLM endpoint or open the oracle. Re-running the script resumes the
# applicable immutable request. The pinned embedding and HTR model trees are
# materialized once under artifacts/local_models/current and then reused.
#
# GPU admission is based on aggregate VRAM, not process exclusivity. A selected
# GPU may have other compute processes so long as at least 90% of its VRAM is
# free when production performs its resource checks. The observed processes
# and memory state remain recorded in the production resource attestations.

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

local_script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
local_repo_root="$(realpath -e -- "$(dirname -- "${local_script_path}")")"
local_python="${LOCAL_PRODUCTION_PYTHON:-/data1/ken/envs/gptoss3/bin/python}"

note() {
    printf '[five-conf local scheduled] %s\n' "$*"
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" && ! -L "$1" && -r "$1" ]] \
        || fail "required readable file is missing: $1"
}

require_directory() {
    [[ -d "$1" && ! -L "$1" && -r "$1" ]] \
        || fail "required readable directory is missing: $1"
}

[[ $# == 0 ]] || fail "usage: $0"
[[ -x "${local_python}" && ! -d "${local_python}" ]] \
    || fail "production Python is unavailable: ${local_python}"
for local_command in awk date flock nice nproc ps realpath setsid sleep tail tee tr; do
    command -v "${local_command}" >/dev/null 2>&1 \
        || fail "required command is unavailable: ${local_command}"
done

"${local_python}" -P - <<'PY'
import sys

if not ((3, 12) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(
        f"production requires Python 3.12 or 3.13, observed {sys.version.split()[0]}"
    )
PY

local_dataset="${local_repo_root}/synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
local_scientific="${local_repo_root}/example_configs/portable_all_evidence_scientific_nsclc.json"
local_stage1="${local_repo_root}/example_configs/production_all_evidence_stage1_full.json"
local_query="${local_repo_root}/example_configs/production_all_evidence_neural_query_full.json"
local_deployment_base="${local_repo_root}/example_configs/portable_all_evidence_deployment_nsclc.stage1-only.example.json"
for local_required in \
    "${local_dataset}" \
    "${local_scientific}" \
    "${local_stage1}" \
    "${local_query}" \
    "${local_deployment_base}"; do
    require_file "${local_required}"
done

local_model_root="${LOCAL_MODEL_ROOT:-${local_repo_root}/artifacts/local_models/current}"
local_embedding_model="${EMBEDDING_MODEL_DIR:-${local_model_root}/qwen3_embedding_8b}"
local_htr_model="${HTR_MODEL_DIR:-${local_model_root}/bert_tiny}"
mkdir -p "${local_model_root}"

materialize_model() {
    local kind="$1"
    local repo_id="$2"
    local revision="$3"
    local target="$4"
    "${local_python}" -P \
        "${local_repo_root}/scripts/materialize_production_models.py" \
        --kind "${kind}" \
        --repo-id "${repo_id}" \
        --revision "${revision}" \
        --target "${target}"
}

if [[ -n "${EMBEDDING_MODEL_DIR:-}" ]]; then
    require_directory "${local_embedding_model}"
else
    materialize_model \
        embedding \
        "${EMBEDDING_MODEL_ID:-Qwen/Qwen3-Embedding-8B}" \
        "${EMBEDDING_MODEL_REVISION:-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af}" \
        "${local_embedding_model}"
fi
if [[ -n "${HTR_MODEL_DIR:-}" ]]; then
    require_directory "${local_htr_model}"
else
    materialize_model \
        htr \
        "${HTR_MODEL_ID:-prajjwal1/bert-tiny}" \
        "${HTR_MODEL_REVISION:-6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837}" \
        "${local_htr_model}"
fi
require_directory "${local_embedding_model}"
require_directory "${local_htr_model}"

local_run_base="${LOCAL_FIVE_CONF_RUN_ROOT_BASE:-${local_repo_root}/artifacts/local_runs/five_conf_five_mod_scheduled}"
local_scratch_root="${LOCAL_FIVE_CONF_SCRATCH_ROOT:-${local_repo_root}/artifacts/local_scratch/five_conf_five_mod_scheduled}"
local_profile_root="${LOCAL_FIVE_CONF_PROFILE_ROOT:-${local_repo_root}/artifacts/runtime_profiles/local_five_conf_five_mod_scheduled}"
local_snapshot="${LOCAL_FIVE_CONF_SOURCE_SNAPSHOT_ROOT:-${local_repo_root}/artifacts/production_source_snapshot_local_five_conf_five_mod_current}"
local_admission_namespace="at_least_90pct_vram_free_trusted_resume"
local_active_root="${local_run_base}/${local_admission_namespace}/active"
local_before_root="${local_active_root}"
local_after_root="${local_active_root}"
local_before_profile="${local_profile_root}/${local_admission_namespace}/gpu0123.json"
local_after_profile="${local_profile_root}/${local_admission_namespace}/gpu23.json"
local_log_root="${local_repo_root}/artifacts/operator_logs"
local_before_log="${local_log_root}/five_conf_five_mod_vram90_gpu0123.log"
local_after_log="${local_log_root}/five_conf_five_mod_vram90_gpu23.log"
local_lock_root="${local_repo_root}/artifacts/production_launch_locks"
mkdir -p \
    "${local_run_base}" \
    "${local_scratch_root}" \
    "${local_profile_root}" \
    "${local_log_root}" \
    "${local_lock_root}"

exec 9>"${local_lock_root}/five_conf_five_mod_local_scheduled.lock"
flock -n 9 || fail "another local five-confounder scheduler owns this run"

local_snapshot_identity="$({
    LOCAL_REPOSITORY="${local_repo_root}" \
    LOCAL_SNAPSHOT="${local_snapshot}" \
    PYTHONPATH="${local_repo_root}" \
        "${local_python}" -P - <<'PY'
import os
from pathlib import Path

from oci.inference.production_source_snapshot import (
    SOURCE_SNAPSHOT_SCHEMA,
    _relative_inventory,
    _required_source_files,
    _sha256_json,
    create_production_source_snapshot,
    validate_production_source_snapshot,
)

repository = Path(os.environ["LOCAL_REPOSITORY"]).resolve(strict=True)
target = Path(os.environ["LOCAL_SNAPSHOT"])
inventory = _relative_inventory(repository, _required_source_files(repository))
live_identity = _sha256_json(
    {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA,
        "files": list(inventory),
        "file_count": len(inventory),
        "python_bytecode_writes_allowed": False,
    }
)
if target.exists() or target.is_symlink():
    snapshot = validate_production_source_snapshot(target)
    if snapshot.content_sha256 != live_identity:
        raise SystemExit(
            "local production source snapshot is stale; choose a fresh "
            "LOCAL_FIVE_CONF_SOURCE_SNAPSHOT_ROOT"
        )
else:
    snapshot = create_production_source_snapshot(
        repository_root=repository,
        target_dir=target,
    )
    if snapshot.content_sha256 != live_identity:
        raise SystemExit("published source snapshot differs from live source")
print(snapshot.content_sha256)
PY
} | tail -n 1)"
[[ "${local_snapshot_identity}" =~ ^[0-9a-f]{64}$ ]] \
    || fail "source snapshot identity is invalid"

local_available_cpus="$(nproc)"
local_cpu_budget="${LOCAL_CPU_BUDGET:-16}"
[[ "${local_cpu_budget}" =~ ^[1-9][0-9]*$ ]] \
    || fail "LOCAL_CPU_BUDGET must be a positive integer"
(( local_cpu_budget <= local_available_cpus )) \
    || fail "LOCAL_CPU_BUDGET exceeds the ${local_available_cpus} available CPUs"
local_total_memory_bytes="$(( $(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo) * 1024 ))"
(( local_total_memory_bytes > 0 )) || fail "could not determine host memory"
local_preflight_memory_budget="${PREFLIGHT_MEMORY_BUDGET_BYTES:-$(( local_total_memory_bytes * 3 / 4 ))}"
local_preflight_owner_peak="${PREFLIGHT_ESTIMATED_OWNER_PEAK_BYTES:-8589934592}"
local_embedding_batch_size="${EMBEDDING_BATCH_SIZE:-8}"
local_gpu_minimum_free_fraction="0.90"
for local_positive in \
    "${local_preflight_memory_budget}" \
    "${local_preflight_owner_peak}" \
    "${local_embedding_batch_size}"; do
    [[ "${local_positive}" =~ ^[1-9][0-9]*$ ]] \
        || fail "memory and embedding controls must be positive integers"
done
local_memory_lanes="$(( local_preflight_memory_budget / local_preflight_owner_peak ))"
(( local_memory_lanes >= 1 )) || local_memory_lanes=1
local_before_lanes=4
local_after_lanes=2
(( local_before_lanes > local_cpu_budget )) \
    && local_before_lanes="${local_cpu_budget}"
(( local_before_lanes > local_memory_lanes )) \
    && local_before_lanes="${local_memory_lanes}"
(( local_after_lanes > local_cpu_budget )) \
    && local_after_lanes="${local_cpu_budget}"
(( local_after_lanes > local_memory_lanes )) \
    && local_after_lanes="${local_memory_lanes}"

build_profile() {
    local target="$1"
    local durable="$2"
    local lane_count="$3"
    shift 3
    local device_arguments=()
    local device
    for device in "$@"; do
        device_arguments+=(--device "${device}")
    done
    PYTHONPATH="${local_snapshot}" \
        "${local_python}" -P \
        "${local_snapshot}/scripts/build_local_stage1_deployment_profile.py" \
        --base "${local_deployment_base}" \
        --target "${target}" \
        --dataset "${local_dataset}" \
        --durable-root "${durable}" \
        --scratch-root "${local_scratch_root}" \
        --embedding-model "${local_embedding_model}" \
        --htr-model "${local_htr_model}" \
        --stage1-profile "${local_stage1}" \
        --query-profile "${local_query}" \
        --cpu-budget "${local_cpu_budget}" \
        --preflight-memory-budget "${local_preflight_memory_budget}" \
        --preflight-owner-peak "${local_preflight_owner_peak}" \
        --preflight-lanes "${lane_count}" \
        --embedding-batch-size "${local_embedding_batch_size}" \
        --gpu-minimum-free-fraction "${local_gpu_minimum_free_fraction}" \
        "${device_arguments[@]}"
}

build_profile \
    "${local_before_profile}" \
    "${local_before_root}" \
    "${local_before_lanes}" \
    cuda:0 cuda:1 cuda:2 cuda:3
build_profile \
    "${local_after_profile}" \
    "${local_after_root}" \
    "${local_after_lanes}" \
    cuda:0 cuda:1

local_target_open_files=65536
local_hard_open_files="$(ulimit -H -n)"
if [[ "${local_hard_open_files}" == "unlimited" ]]; then
    local_desired_open_files="${local_target_open_files}"
elif (( local_hard_open_files < 4096 )); then
    fail "hard open-file limit is below 4096"
elif (( local_hard_open_files < local_target_open_files )); then
    local_desired_open_files="${local_hard_open_files}"
else
    local_desired_open_files="${local_target_open_files}"
fi
ulimit -S -n "${local_desired_open_files}"

local_deadline_epoch="$({
    "${local_python}" -P - <<'PY'
from datetime import datetime
from zoneinfo import ZoneInfo

deadline = datetime(2026, 8, 1, 9, 0, 0, tzinfo=ZoneInfo("America/New_York"))
print(int(deadline.timestamp()))
PY
} | tail -n 1)"
[[ "${local_deadline_epoch}" =~ ^[0-9]+$ ]] \
    || fail "could not compile the GPU transition deadline"

validate_visible_gpus() {
    local physical_mask="$1"
    local expected_count="$2"
    note "checking physical GPUs ${physical_mask}"
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${physical_mask}" \
        "${local_python}" -P - "${expected_count}" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
observed = torch.cuda.device_count()
if observed != expected:
    raise SystemExit(f"expected {expected} visible GPUs, observed {observed}")
for index in range(observed):
    properties = torch.cuda.get_device_properties(index)
    print(
        f"[five-conf local GPU] logical cuda:{index}: {properties.name}; "
        f"memory={properties.total_memory / (1024 ** 3):.1f} GiB"
    )
PY
}

local_entrypoint="${local_snapshot}/scripts/run_production_all_evidence_workflow.py"
require_file "${local_entrypoint}"
local_active_pid=""
local_active_pgid=""
local_active_profile=""

active_group_alive() {
    [[ -n "${local_active_pgid}" ]] \
        && kill -0 -- "-${local_active_pgid}" 2>/dev/null
}

verify_active_group() {
    [[ -n "${local_active_pid}" && -n "${local_active_pgid}" ]] \
        || return 1
    local observed_pgid
    observed_pgid="$(ps -o pgid= -p "${local_active_pid}" | tr -d ' ')"
    [[ "${observed_pgid}" == "${local_active_pgid}" ]] || return 1
    [[ -r "/proc/${local_active_pid}/cmdline" ]] || return 1
    local command_line
    command_line="$(tr '\0' ' ' < "/proc/${local_active_pid}/cmdline")"
    [[ "${command_line}" == *"${local_entrypoint}"* ]] || return 1
    [[ "${command_line}" == *"${local_active_profile}"* ]] || return 1
}

stop_active_group() {
    local reason="$1"
    if ! active_group_alive; then
        return 0
    fi
    verify_active_group \
        || fail "refusing to signal an unverified process group"
    note "sending SIGTERM to workflow-owned PGID ${local_active_pgid}: ${reason}"
    kill -TERM -- "-${local_active_pgid}"
}

handle_operator_signal() {
    note "operator signal received"
    stop_active_group "operator interrupted the scheduler"
    exit 130
}
trap handle_operator_signal INT TERM HUP

start_workflow() {
    local label="$1"
    local physical_mask="$2"
    local profile="$3"
    local durable_root="$4"
    local log_path="$5"
    local resume_arguments=()
    if [[ -L "${durable_root}" ]]; then
        fail "durable root is a symlink: ${durable_root}"
    elif [[ -d "${durable_root}" ]]; then
        require_file "${durable_root}/immutable_run_request.json"
        resume_arguments=(--resume)
        note "resuming ${label} request"
    elif [[ -e "${durable_root}" ]]; then
        fail "durable root is not a directory: ${durable_root}"
    else
        note "starting fresh ${label} request"
    fi
    local mpl_root="${local_scratch_root}/mpl_${label}"
    mkdir -p "${mpl_root}"
    note "${label}: physical GPUs ${physical_mask}; log ${log_path}"
    setsid nice -n "${LOCAL_NICE_LEVEL:-5}" env \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES="${physical_mask}" \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        MPLCONFIGDIR="${mpl_root}" \
        OCI_PRODUCTION_SOURCE_SNAPSHOT_SHA256="${local_snapshot_identity}" \
        PYTHONHASHSEED=42 \
        PYTHONNOUSERSITE=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        TOKENIZERS_PARALLELISM=false \
        PYTHONPATH="${local_snapshot}" \
        "${local_python}" -P -u "${local_entrypoint}" \
        --scientific-spec "${local_scientific}" \
        --deployment-profile "${profile}" \
        --source-snapshot-root "${local_snapshot}" \
        --resume-trust trusted-local \
        --validation-depth fresh_terminal_audit \
        --log-level INFO \
        --stage1-only \
        "${resume_arguments[@]}" \
        > >(tee -a "${log_path}") 2>&1 &
    local_active_pid="$!"
    local_active_profile="${profile}"
    local attempts=0
    while (( attempts < 50 )); do
        local_active_pgid="$(
            ps -o pgid= -p "${local_active_pid}" 2>/dev/null \
                | tr -d ' '
        )"
        if [[ "${local_active_pgid}" == "${local_active_pid}" ]] \
            && verify_active_group; then
            break
        fi
        kill -0 "${local_active_pid}" 2>/dev/null \
            || fail "${label} workflow exited before process-group verification"
        sleep 0.1
        ((attempts += 1))
    done
    verify_active_group \
        || fail "${label} workflow process-group identity is invalid"
}

wait_for_active() {
    local status
    if wait "${local_active_pid}"; then
        status=0
    else
        status=$?
    fi
    local_active_pid=""
    local_active_pgid=""
    local_active_profile=""
    return "${status}"
}

local_now_epoch="$(date +%s)"
if (( local_now_epoch < local_deadline_epoch )); then
    validate_visible_gpus "0,1,2,3" 4
    start_workflow \
        "gpu0123" \
        "0,1,2,3" \
        "${local_before_profile}" \
        "${local_before_root}" \
        "${local_before_log}"
    while active_group_alive; do
        local_now_epoch="$(date +%s)"
        if (( local_now_epoch >= local_deadline_epoch )); then
            break
        fi
        local_remaining="$(( local_deadline_epoch - local_now_epoch ))"
        local_poll_seconds=5
        (( local_remaining < local_poll_seconds )) \
            && local_poll_seconds="${local_remaining}"
        (( local_poll_seconds >= 1 )) || local_poll_seconds=1
        sleep "${local_poll_seconds}"
    done
    if ! active_group_alive; then
        if wait_for_active; then
            note "four-GPU request completed before the deadline"
            exit 0
        fi
        fail "four-GPU request failed before the deadline; fallback was not started"
    fi

    stop_active_group \
        "9:00 AM America/New_York deadline reached; retire GPUs 0 and 1"
    local_term_deadline="$(( $(date +%s) + 120 ))"
    while active_group_alive && (( $(date +%s) < local_term_deadline )); do
        sleep 1
    done
    if active_group_alive; then
        fail "workflow group did not exit after SIGTERM; no SIGKILL was sent and the two-GPU request was not started"
    fi
    wait_for_active || true
    note "four-GPU workflow stopped; sealed work and incomplete attempts were preserved"
else
    note "launch is at or after the deadline; GPUs 0 and 1 will not be used"
fi

validate_visible_gpus "2,3" 2
start_workflow \
    "gpu23" \
    "2,3" \
    "${local_after_profile}" \
    "${local_after_root}" \
    "${local_after_log}"
if wait_for_active; then
    note "two-GPU request completed Stage 1 and handoff_validation"
    exit 0
fi
fail "two-GPU request failed; rerun this script to resume sealed work"
