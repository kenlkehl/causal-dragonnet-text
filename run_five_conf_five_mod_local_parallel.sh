#!/usr/bin/env bash

# Resumable five-confounder/five-modifier Stage 1 production launcher.
#
# With no RUN_TAG this launcher starts from scratch under a unique name. An
# explicit RUN_TAG reopens an interrupted run at its sealed boundaries when
# its immutable request exists; otherwise it safely retries an interrupted
# pre-initialization launch. It never adopts, deletes, or overwrites another
# run. Stop an older launcher first, then run, for example:
#
#   GPU_LIST=0,1,2,3 \
#   ./run_five_conf_five_mod_local_parallel.sh
#
# GPU_LIST contains physical CUDA indices. CUDA_VISIBLE_DEVICES remaps them to
# logical cuda:0..N-1 for the production profile. The production entrypoint
# detects a safe uniform owner-lane count from live free VRAM, currently
# available host RAM, and the CPU budget. SCOPE_WORKERS_PER_DEVICE and
# MAX_PARALLEL_OWNERS are hard ceilings, not promises to launch that many
# workers. HTR and neural-query fold parallelism remain one per owner, avoiding
# an additional multiplication of model processes within each owner. Initial
# embedding uses every selected GPU through the ordinary production
# embedding-cache phase. The resolved capacity and its inputs are recorded in
# workflow_progress.json under stage1_owner_capacity_attestation.
#
# Useful overrides:
#
#   RUN_TAG                         filesystem-safe new or existing run name
#   GPU_LIST                       comma-separated physical GPU indices
#   SCOPE_WORKERS_PER_DEVICE       hard owner-lane ceiling per GPU (default 4)
#   MAX_PARALLEL_OWNERS            hard global owner ceiling (default GPUs * ceiling)
#   CPU_BUDGET                     global CPU budget (default 16)
#   PREFLIGHT_LANES                preflight owner lanes (default GPU count)
#   EMBEDDING_BATCH_SIZE           per embedding worker batch size (default 8)
#   GPU_MINIMUM_FREE_FRACTION      admission threshold (default 0.90)
#   STAGE1_ESTIMATED_DEVICE_MEMORY_BYTES_PER_OWNER  VRAM estimate (default 8 GiB)
#   STAGE1_DEVICE_MEMORY_RESERVE_BYTES              VRAM reserve (default 6 GiB)
#   STAGE1_ESTIMATED_HOST_MEMORY_BYTES_PER_OWNER    RAM estimate (default 8 GiB)
#   STAGE1_HOST_MEMORY_BUDGET_FRACTION              usable available RAM (default 0.75)
#   STAGE1_MINIMUM_CPU_THREADS_PER_OWNER            CPU estimate (default 1)
#   LOCAL_PRODUCTION_PYTHON        Python executable
#   LOCAL_MODEL_ROOT               materialized production model directory
#   FIVE_CONF_RUN_ROOT_BASE        durable-root parent
#   FIVE_CONF_SCRATCH_ROOT_BASE    scratch-root parent
#   FIVE_CONF_SNAPSHOT_ROOT_BASE   immutable source-snapshot parent
#   FIVE_CONF_PROFILE_ROOT         generated deployment-profile parent
#   FIVE_CONF_LOG_ROOT             operator-log parent
#
# Ctrl-C once. The trap verifies the owned workflow process group and sends
# SIGTERM. This launcher never sends SIGKILL.

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repo_root="$(realpath -e -- "$(dirname -- "${script_path}")")"
production_python="${LOCAL_PRODUCTION_PYTHON:-/data1/ken/envs/gptoss3/bin/python}"

note() {
    printf '[five-conf local parallel] %s\n' "$*"
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

positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] \
        || fail "$1 must be a positive integer"
}

[[ $# == 0 ]] || fail "usage: $0"
[[ -x "${production_python}" && ! -d "${production_python}" ]] \
    || fail "production Python is unavailable: ${production_python}"
for required_command in awk date flock nice nproc ps realpath seq setsid sleep tail tee tr; do
    command -v "${required_command}" >/dev/null 2>&1 \
        || fail "required command is unavailable: ${required_command}"
done

"${production_python}" -P - <<'PY'
import sys

if not ((3, 12) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(
        f"production requires Python 3.12 or 3.13, observed {sys.version.split()[0]}"
    )
PY

gpu_list="${GPU_LIST:-0,1,2,3}"
scope_workers_per_device="${SCOPE_WORKERS_PER_DEVICE:-4}"
cpu_budget="${CPU_BUDGET:-16}"
embedding_batch_size="${EMBEDDING_BATCH_SIZE:-8}"
gpu_minimum_free_fraction="${GPU_MINIMUM_FREE_FRACTION:-0.90}"
preflight_owner_peak="${PREFLIGHT_ESTIMATED_OWNER_PEAK_BYTES:-8589934592}"
estimated_device_owner_bytes="${STAGE1_ESTIMATED_DEVICE_MEMORY_BYTES_PER_OWNER:-8589934592}"
device_memory_reserve_bytes="${STAGE1_DEVICE_MEMORY_RESERVE_BYTES:-6442450944}"
estimated_host_owner_bytes="${STAGE1_ESTIMATED_HOST_MEMORY_BYTES_PER_OWNER:-8589934592}"
host_memory_budget_fraction="${STAGE1_HOST_MEMORY_BUDGET_FRACTION:-0.75}"
minimum_cpu_threads_per_owner="${STAGE1_MINIMUM_CPU_THREADS_PER_OWNER:-1}"
if [[ -n "${RUN_TAG:-}" ]]; then
    run_tag_was_supplied=1
    run_tag="${RUN_TAG}"
else
    run_tag_was_supplied=0
    run_tag="five_conf_five_mod_parallel_$(date +%Y%m%dT%H%M%S)"
fi

[[ "${run_tag}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || fail "RUN_TAG must contain only letters, digits, dot, underscore, and dash"
positive_integer SCOPE_WORKERS_PER_DEVICE "${scope_workers_per_device}"
positive_integer CPU_BUDGET "${cpu_budget}"
positive_integer EMBEDDING_BATCH_SIZE "${embedding_batch_size}"
positive_integer PREFLIGHT_ESTIMATED_OWNER_PEAK_BYTES "${preflight_owner_peak}"
positive_integer STAGE1_ESTIMATED_DEVICE_MEMORY_BYTES_PER_OWNER "${estimated_device_owner_bytes}"
positive_integer STAGE1_ESTIMATED_HOST_MEMORY_BYTES_PER_OWNER "${estimated_host_owner_bytes}"
positive_integer STAGE1_MINIMUM_CPU_THREADS_PER_OWNER "${minimum_cpu_threads_per_owner}"
[[ "${device_memory_reserve_bytes}" =~ ^[0-9]+$ ]] \
    || fail "STAGE1_DEVICE_MEMORY_RESERVE_BYTES must be a nonnegative integer"
"${production_python}" -P - "${host_memory_budget_fraction}" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or not 0 < value <= 1:
    raise SystemExit(
        "STAGE1_HOST_MEMORY_BUDGET_FRACTION must be in (0, 1]"
    )
PY

old_ifs="${IFS}"
IFS=',' read -r -a physical_gpus <<< "${gpu_list}"
IFS="${old_ifs}"
(( ${#physical_gpus[@]} >= 1 )) || fail "GPU_LIST must select at least one GPU"
declare -A observed_gpu_ids=()
for physical_gpu in "${physical_gpus[@]}"; do
    [[ "${physical_gpu}" =~ ^(0|[1-9][0-9]*)$ ]] \
        || fail "GPU_LIST contains an invalid physical index: ${physical_gpu}"
    [[ -z "${observed_gpu_ids[${physical_gpu}]+present}" ]] \
        || fail "GPU_LIST contains a duplicate physical index: ${physical_gpu}"
    observed_gpu_ids["${physical_gpu}"]=1
done
gpu_count="${#physical_gpus[@]}"
max_parallel_owners="${MAX_PARALLEL_OWNERS:-$((gpu_count * scope_workers_per_device))}"
preflight_lanes="${PREFLIGHT_LANES:-${gpu_count}}"
positive_integer MAX_PARALLEL_OWNERS "${max_parallel_owners}"
positive_integer PREFLIGHT_LANES "${preflight_lanes}"
(( max_parallel_owners <= gpu_count * scope_workers_per_device )) \
    || fail "MAX_PARALLEL_OWNERS exceeds GPU owner-lane capacity"
(( max_parallel_owners <= cpu_budget )) \
    || fail "MAX_PARALLEL_OWNERS exceeds CPU_BUDGET"
(( preflight_lanes <= max_parallel_owners )) \
    || fail "PREFLIGHT_LANES exceeds MAX_PARALLEL_OWNERS"
available_cpus="$(nproc)"
(( cpu_budget <= available_cpus )) \
    || fail "CPU_BUDGET exceeds the ${available_cpus} available CPUs"

dataset="${repo_root}/synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
scientific_profile="${repo_root}/example_configs/portable_all_evidence_scientific_nsclc.json"
stage1_profile="${repo_root}/example_configs/production_all_evidence_stage1_full.json"
query_profile="${repo_root}/example_configs/production_all_evidence_neural_query_full.json"
deployment_base="${repo_root}/example_configs/portable_all_evidence_deployment_nsclc.stage1-only.example.json"
for required_file in \
    "${dataset}" \
    "${scientific_profile}" \
    "${stage1_profile}" \
    "${query_profile}" \
    "${deployment_base}"; do
    require_file "${required_file}"
done

model_root="${LOCAL_MODEL_ROOT:-${repo_root}/artifacts/local_models/current}"
embedding_model="${EMBEDDING_MODEL_DIR:-${model_root}/qwen3_embedding_8b}"
htr_model="${HTR_MODEL_DIR:-${model_root}/bert_tiny}"
require_directory "${embedding_model}"
require_directory "${htr_model}"

run_root_base="${FIVE_CONF_RUN_ROOT_BASE:-${repo_root}/artifacts/local_runs}"
scratch_root_base="${FIVE_CONF_SCRATCH_ROOT_BASE:-${repo_root}/artifacts/local_scratch}"
snapshot_root_base="${FIVE_CONF_SNAPSHOT_ROOT_BASE:-${repo_root}/artifacts/production_source_snapshots}"
profile_root="${FIVE_CONF_PROFILE_ROOT:-${repo_root}/artifacts/runtime_profiles}"
log_root="${FIVE_CONF_LOG_ROOT:-${repo_root}/artifacts/operator_logs}"
durable_root="${run_root_base}/${run_tag}"
scratch_root="${scratch_root_base}/${run_tag}"
snapshot_root="${snapshot_root_base}/${run_tag}"
deployment_profile="${profile_root}/${run_tag}.json"
log_path="${log_root}/${run_tag}.log"
mpl_root="${scratch_root}/matplotlib"

run_paths=(
    "${durable_root}"
    "${scratch_root}"
    "${snapshot_root}"
    "${deployment_profile}"
    "${log_path}"
)
if (( ! run_tag_was_supplied )); then
    for fresh_path in "${run_paths[@]}"; do
        [[ ! -e "${fresh_path}" && ! -L "${fresh_path}" ]] \
            || fail "fresh run path already exists: ${fresh_path}"
    done
fi

for existing_directory in "${scratch_root}" "${snapshot_root}"; do
    if [[ -e "${existing_directory}" || -L "${existing_directory}" ]]; then
        [[ -d "${existing_directory}" && ! -L "${existing_directory}" ]] \
            || fail "run directory is not one real directory: ${existing_directory}"
    fi
done
for existing_file in "${deployment_profile}" "${log_path}"; do
    if [[ -e "${existing_file}" || -L "${existing_file}" ]]; then
        [[ -f "${existing_file}" && ! -L "${existing_file}" ]] \
            || fail "run file is not one real regular file: ${existing_file}"
    fi
done

resume_arguments=()
if [[ -e "${durable_root}" || -L "${durable_root}" ]]; then
    (( run_tag_was_supplied )) \
        || fail "fresh durable run root already exists: ${durable_root}"
    [[ -d "${durable_root}" && ! -L "${durable_root}" ]] \
        || fail "durable run root is not one real directory: ${durable_root}"
    require_file "${durable_root}/immutable_run_request.json"
    require_directory "${snapshot_root}"
    require_file "${deployment_profile}"
    resume_arguments=(--resume)
fi

mkdir -p \
    "${run_root_base}" \
    "${scratch_root}" \
    "${snapshot_root_base}" \
    "${profile_root}" \
    "${log_root}" \
    "${mpl_root}"

lock_root="${repo_root}/artifacts/production_launch_locks"
mkdir -p "${lock_root}"
exec 9>"${lock_root}/five_conf_five_mod_local_parallel.lock"
flock -n 9 || fail "another local parallel five-confounder launcher is active"

snapshot_identity="$({
    REPOSITORY="${repo_root}" SNAPSHOT="${snapshot_root}" \
    PYTHONPATH="${repo_root}" "${production_python}" -P - <<'PY'
import os
from pathlib import Path

from oci.inference.production_source_snapshot import (
    create_production_source_snapshot,
    validate_production_source_snapshot,
)

target = Path(os.environ["SNAPSHOT"])
if target.exists() or target.is_symlink():
    snapshot = validate_production_source_snapshot(target)
else:
    snapshot = create_production_source_snapshot(
        repository_root=Path(os.environ["REPOSITORY"]),
        target_dir=target,
    )
print(snapshot.content_sha256)
PY
} | tail -n 1)"
[[ "${snapshot_identity}" =~ ^[0-9a-f]{64}$ ]] \
    || fail "source snapshot identity is invalid"

total_memory_bytes="$(( $(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo) * 1024 ))"
(( total_memory_bytes > 0 )) || fail "could not determine host memory"
preflight_memory_budget="${PREFLIGHT_MEMORY_BUDGET_BYTES:-$((total_memory_bytes * 3 / 4))}"
positive_integer PREFLIGHT_MEMORY_BUDGET_BYTES "${preflight_memory_budget}"

device_arguments=()
for logical_index in "${!physical_gpus[@]}"; do
    device_arguments+=(--device "cuda:${logical_index}")
done

PYTHONPATH="${snapshot_root}" \
"${production_python}" -P \
    "${snapshot_root}/scripts/build_local_stage1_deployment_profile.py" \
    --base "${deployment_base}" \
    --target "${deployment_profile}" \
    --dataset "${dataset}" \
    --durable-root "${durable_root}" \
    --scratch-root "${scratch_root}" \
    --embedding-model "${embedding_model}" \
    --htr-model "${htr_model}" \
    --stage1-profile "${stage1_profile}" \
    --query-profile "${query_profile}" \
    --scope-workers-per-device "${scope_workers_per_device}" \
    --max-parallel-owners "${max_parallel_owners}" \
    --cpu-budget "${cpu_budget}" \
    --preflight-memory-budget "${preflight_memory_budget}" \
    --preflight-owner-peak "${preflight_owner_peak}" \
    --preflight-lanes "${preflight_lanes}" \
    --embedding-batch-size "${embedding_batch_size}" \
    --gpu-minimum-free-fraction "${gpu_minimum_free_fraction}" \
    --owner-capacity-mode resource_autodetect \
    --estimated-device-memory-per-owner "${estimated_device_owner_bytes}" \
    --device-memory-reserve "${device_memory_reserve_bytes}" \
    --estimated-host-memory-per-owner "${estimated_host_owner_bytes}" \
    --host-memory-budget-fraction "${host_memory_budget_fraction}" \
    --minimum-cpu-threads-per-owner "${minimum_cpu_threads_per_owner}" \
    "${device_arguments[@]}"

target_open_files=65536
hard_open_files="$(ulimit -H -n)"
if [[ "${hard_open_files}" == "unlimited" ]]; then
    desired_open_files="${target_open_files}"
elif (( hard_open_files < 4096 )); then
    fail "hard open-file limit is below 4096"
elif (( hard_open_files < target_open_files )); then
    desired_open_files="${hard_open_files}"
else
    desired_open_files="${target_open_files}"
fi
ulimit -S -n "${desired_open_files}"

note "run tag: ${run_tag}"
note "physical GPUs: ${gpu_list}; logical devices: cuda:0..$((gpu_count - 1))"
note "owner ceilings: ${scope_workers_per_device} per GPU, ${max_parallel_owners} global"
note "owner autodetection: live free VRAM, available RAM, and CPU; effective lanes will be recorded in workflow_progress.json"
note "HTR/neural folds: one per owner; embedding workers: ${gpu_count}"
note "durable root: ${durable_root}"
note "scratch root: ${scratch_root}"
note "source snapshot: ${snapshot_root} (${snapshot_identity})"
note "deployment profile: ${deployment_profile}"
note "log: ${log_path}"
if (( ${#resume_arguments[@]} )); then
    note "resume: authenticated durable request will reopen at sealed boundaries"
else
    note "resume: starting a new durable request"
fi

note "checking selected GPUs and logical remapping"
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES="${gpu_list}" \
"${production_python}" -P - "${gpu_count}" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
observed = torch.cuda.device_count()
if observed != expected:
    raise SystemExit(f"expected {expected} visible GPUs, observed {observed}")
for index in range(observed):
    properties = torch.cuda.get_device_properties(index)
    print(
        f"[five-conf local parallel GPU] logical cuda:{index}: "
        f"{properties.name}; memory={properties.total_memory / (1024 ** 3):.1f} GiB"
    )
PY

entrypoint="${snapshot_root}/scripts/run_production_all_evidence_workflow.py"
require_file "${entrypoint}"
active_pid=""
active_pgid=""

verify_active_group() {
    [[ -n "${active_pid}" && -n "${active_pgid}" ]] || return 1
    local observed_pgid command_line
    observed_pgid="$(ps -o pgid= -p "${active_pid}" 2>/dev/null | tr -d ' ')"
    [[ "${observed_pgid}" == "${active_pgid}" ]] || return 1
    [[ -r "/proc/${active_pid}/cmdline" ]] || return 1
    command_line="$(tr '\0' ' ' < "/proc/${active_pid}/cmdline")"
    [[ "${command_line}" == *"${entrypoint}"* ]] || return 1
    [[ "${command_line}" == *"${deployment_profile}"* ]] || return 1
}

stop_active_group() {
    local reason="$1"
    [[ -n "${active_pgid}" ]] || return 0
    if ! kill -0 -- "-${active_pgid}" 2>/dev/null; then
        return 0
    fi
    verify_active_group || fail "refusing to signal an unverified workflow group"
    note "sending SIGTERM to workflow-owned PGID ${active_pgid}: ${reason}"
    kill -TERM -- "-${active_pgid}"
}

handle_operator_signal() {
    trap - INT TERM HUP
    stop_active_group "operator interrupted the launcher"
    if [[ -n "${active_pid}" ]]; then
        wait "${active_pid}" 2>/dev/null || true
    fi
    exit 130
}
trap handle_operator_signal INT TERM HUP

setsid nice -n "${LOCAL_NICE_LEVEL:-5}" env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${gpu_list}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MPLCONFIGDIR="${mpl_root}" \
    OCI_PRODUCTION_SOURCE_SNAPSHOT_SHA256="${snapshot_identity}" \
    PYTHONHASHSEED=42 \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="${snapshot_root}" \
    "${production_python}" -P -u "${entrypoint}" \
    --scientific-spec "${scientific_profile}" \
    --deployment-profile "${deployment_profile}" \
    --source-snapshot-root "${snapshot_root}" \
    --resume-trust trusted_local \
    --validation-depth fresh_terminal_audit \
    --log-level INFO \
    --stage1-only \
    "${resume_arguments[@]}" \
    > >(tee -a "${log_path}") 2>&1 &
active_pid="$!"

for _attempt in $(seq 1 50); do
    active_pgid="$(ps -o pgid= -p "${active_pid}" 2>/dev/null | tr -d ' ')"
    if [[ "${active_pgid}" == "${active_pid}" ]] && verify_active_group; then
        break
    fi
    kill -0 "${active_pid}" 2>/dev/null \
        || fail "workflow exited before process-group verification"
    sleep 0.1
done
verify_active_group || fail "workflow process-group identity is invalid"

if wait "${active_pid}"; then
    note "Stage 1 and handoff validation completed"
else
    status="$?"
    fail "workflow exited with status ${status}; artifacts were preserved"
fi
