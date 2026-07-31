#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

cloud_common_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
cloud_repo_root="$(realpath -e -- "$(dirname -- "${cloud_common_path}")/..")"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '[%s cloud] %s\n' "${CLOUD_RUN_KEY:-all-evidence}" "$*"
}

require_file() {
    [[ -f "$1" && ! -L "$1" && -r "$1" ]] \
        || fail "required readable file is missing: $1"
}

require_directory() {
    [[ -d "$1" && ! -L "$1" && -r "$1" ]] \
        || fail "required readable directory is missing: $1"
}

[[ -n "${CLOUD_RUN_KEY:-}" ]] || fail "CLOUD_RUN_KEY is unset"
[[ "${CLOUD_RUN_KEY}" =~ ^[a-z0-9][a-z0-9_]*$ ]] \
    || fail "CLOUD_RUN_KEY contains unsupported characters"
[[ -n "${CLOUD_DATASET_RELATIVE:-}" ]] || fail "CLOUD_DATASET_RELATIVE is unset"

cloud_check_only=0
cloud_prepare_only=0
case "${1:-}" in
    "") ;;
    --check-only) cloud_check_only=1 ;;
    --prepare-only) cloud_prepare_only=1 ;;
    *) fail "usage: $0 [--check-only|--prepare-only]" ;;
esac

cloud_vllm_model_id="nvidia/Gemma-4-31B-IT-NVFP4"
cloud_vllm_model_revision="4135a98a9b728a548947683219633b25682223ac"
cloud_endpoint_model="${cloud_vllm_model_id}@${cloud_vllm_model_revision}"
cloud_vllm_proxy_port="${VLLM_PROXY_PORT:-8002}"
cloud_vllm_upstream_port="${VLLM_UPSTREAM_PORT:-8003}"
for cloud_port_name in cloud_vllm_proxy_port cloud_vllm_upstream_port; do
    cloud_port_value="${!cloud_port_name}"
    [[ "${cloud_port_value}" =~ ^[0-9]+$ ]] \
        || fail "${cloud_port_name} must be an integer"
    (( cloud_port_value >= 1024 && cloud_port_value <= 65535 )) \
        || fail "${cloud_port_name} must be an unprivileged TCP port"
done
[[ "${cloud_vllm_proxy_port}" != "${cloud_vllm_upstream_port}" ]] \
    || fail "VLLM_PROXY_PORT and VLLM_UPSTREAM_PORT must differ"
cloud_endpoint="http://127.0.0.1:${cloud_vllm_proxy_port}/v1"
if [[ -n "${STAGE2_ENDPOINT:-}" && "${STAGE2_ENDPOINT}" != "${cloud_endpoint}" ]]; then
    fail "STAGE2_ENDPOINT cannot replace the launcher-owned local vLLM endpoint"
fi
if [[ -n "${STAGE2_ENDPOINT_MODEL:-}" \
    && "${STAGE2_ENDPOINT_MODEL}" != "${cloud_endpoint_model}" ]]; then
    fail "STAGE2_ENDPOINT_MODEL cannot replace the pinned NVIDIA NVFP4 model"
fi
cloud_endpoint_auth="${OCI_STAGE2_ENDPOINT_AUTH:-none}"
cloud_endpoint_transport="${OCI_STAGE2_ENDPOINT_TRANSPORT:-vllm}"
[[ "${cloud_endpoint_auth}" == "none" ]] \
    || fail "OCI_STAGE2_ENDPOINT_AUTH must be none for the loopback-only vLLM server"
[[ "${cloud_endpoint_transport}" == "vllm" ]] \
    || fail "OCI_STAGE2_ENDPOINT_TRANSPORT must be vllm"
[[ -z "${OCI_STAGE2_ENDPOINT_API_KEY:-}" ]] \
    || fail "OCI_STAGE2_ENDPOINT_API_KEY must be unset for local vLLM"
cloud_stop_after="${STOP_AFTER-}"
if [[ -n "${cloud_stop_after}" && "${cloud_stop_after}" != "handoff_validation" ]]; then
    fail "these launchers support STOP_AFTER only when set to handoff_validation"
fi

for cloud_command in awk flock nproc nvidia-smi ps realpath setsid tail tr uv; do
    command -v "${cloud_command}" >/dev/null 2>&1 \
        || fail "required command is unavailable: ${cloud_command}"
done

cloud_uv_cache="${UV_CACHE_DIR:-${cloud_repo_root}/.uv-cache}"
if [[ "${SKIP_UV_SYNC:-0}" != "1" ]]; then
    note "syncing the locked Python environment"
    UV_CACHE_DIR="${cloud_uv_cache}" uv sync \
        --frozen \
        --extra extraction
fi
cloud_python="${CLOUD_PYTHON:-${cloud_repo_root}/.venv/bin/python}"
[[ -x "${cloud_python}" && ! -d "${cloud_python}" ]] \
    || fail "uv environment Python is absent: ${cloud_python}"
cloud_vllm_command="${VLLM_COMMAND:-$(dirname -- "${cloud_python}")/vllm}"
[[ -x "${cloud_vllm_command}" && -f "${cloud_vllm_command}" \
    && ! -L "${cloud_vllm_command}" ]] \
    || fail "the locked environment does not expose a real vLLM executable: ${cloud_vllm_command}"

"${cloud_python}" -P - <<'PY'
import sys

if not ((3, 12) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(
        f"production requires Python 3.12 or 3.13, observed {sys.version.split()[0]}"
    )
PY

cloud_dataset="${cloud_repo_root}/${CLOUD_DATASET_RELATIVE}"
cloud_scientific="${cloud_repo_root}/example_configs/portable_all_evidence_scientific_nsclc.json"
cloud_stage1="${cloud_repo_root}/example_configs/production_all_evidence_stage1_full.json"
cloud_query="${cloud_repo_root}/example_configs/production_all_evidence_neural_query_full.json"
cloud_deployment_base="${cloud_repo_root}/example_configs/portable_all_evidence_deployment_nsclc.stage1-only.example.json"
for cloud_required in \
    "${cloud_dataset}" \
    "${cloud_scientific}" \
    "${cloud_stage1}" \
    "${cloud_query}" \
    "${cloud_deployment_base}"; do
    require_file "${cloud_required}"
done

cloud_model_root="${CLOUD_MODEL_ROOT:-${cloud_repo_root}/artifacts/local_models/current}"
cloud_embedding_model="${EMBEDDING_MODEL_DIR:-${cloud_model_root}/qwen3_embedding_8b}"
cloud_htr_model="${HTR_MODEL_DIR:-${cloud_model_root}/bert_tiny}"
cloud_stage2_tokenizer="${STAGE2_TOKENIZER_DIR:-${cloud_model_root}/stage2_tokenizer}"
cloud_stage2_vllm_model="${STAGE2_VLLM_MODEL_DIR:-${cloud_model_root}/gemma4_31b_it_nvfp4}"
mkdir -p "${cloud_model_root}"

materialize_model() {
    local kind="$1"
    local repo_id="$2"
    local revision="$3"
    local target="$4"
    "${cloud_python}" -P \
        "${cloud_repo_root}/scripts/materialize_production_models.py" \
        --kind "${kind}" \
        --repo-id "${repo_id}" \
        --revision "${revision}" \
        --target "${target}"
}

if [[ -n "${EMBEDDING_MODEL_DIR:-}" ]]; then
    require_directory "${cloud_embedding_model}"
else
    materialize_model \
        embedding \
        "${EMBEDDING_MODEL_ID:-Qwen/Qwen3-Embedding-8B}" \
        "${EMBEDDING_MODEL_REVISION:-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af}" \
        "${cloud_embedding_model}"
fi
if [[ -n "${HTR_MODEL_DIR:-}" ]]; then
    require_directory "${cloud_htr_model}"
else
    materialize_model \
        htr \
        "${HTR_MODEL_ID:-prajjwal1/bert-tiny}" \
        "${HTR_MODEL_REVISION:-6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837}" \
        "${cloud_htr_model}"
fi
materialize_model \
    tokenizer \
    "${cloud_vllm_model_id}" \
    "${cloud_vllm_model_revision}" \
    "${cloud_stage2_tokenizer}"
materialize_model \
    stage2_vllm \
    "${cloud_vllm_model_id}" \
    "${cloud_vllm_model_revision}" \
    "${cloud_stage2_vllm_model}"
require_directory "${cloud_embedding_model}"
require_directory "${cloud_htr_model}"
require_directory "${cloud_stage2_tokenizer}"
require_directory "${cloud_stage2_vllm_model}"

cloud_gpu_count="${CLOUD_GPU_COUNT:-8}"
[[ "${cloud_gpu_count}" =~ ^[1-9][0-9]*$ ]] \
    || fail "CLOUD_GPU_COUNT must be a positive integer"
[[ "${cloud_gpu_count}" == "8" ]] \
    || fail "these launchers are sized for exactly eight GPUs"
cloud_visible_devices="${CLOUD_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[[ "${cloud_visible_devices}" == "0,1,2,3,4,5,6,7" ]] \
    || fail "CUDA_VISIBLE_DEVICES must be exactly 0,1,2,3,4,5,6,7"

note "checking eight visible GPUs"
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES="${cloud_visible_devices}" \
    "${cloud_python}" -P - <<'PY'
import torch

if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 visible CUDA devices, found {torch.cuda.device_count()}")
for index in range(8):
    properties = torch.cuda.get_device_properties(index)
    capability = torch.cuda.get_device_capability(index)
    if capability[0] < 10:
        raise SystemExit(
            f"cuda:{index} is not NVIDIA Blackwell-class: capability={capability}"
        )
    print(
        f"[cloud gpu] cuda:{index}: {properties.name}; "
        f"memory={properties.total_memory / (1024 ** 3):.1f} GiB; "
        f"capability={capability[0]}.{capability[1]}"
    )
PY

cloud_available_cpus="$(nproc)"
[[ "${cloud_available_cpus}" =~ ^[1-9][0-9]*$ ]] \
    || fail "nproc returned an invalid CPU count"
cloud_cpu_budget="${CLOUD_CPU_BUDGET:-${cloud_available_cpus}}"
[[ "${cloud_cpu_budget}" =~ ^[1-9][0-9]*$ ]] \
    || fail "CLOUD_CPU_BUDGET must be a positive integer"
(( cloud_cpu_budget >= 8 && cloud_cpu_budget <= cloud_available_cpus )) \
    || fail "CLOUD_CPU_BUDGET must be between 8 and ${cloud_available_cpus}"

cloud_total_memory_bytes="$(( $(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo) * 1024 ))"
(( cloud_total_memory_bytes > 0 )) || fail "could not determine host memory"
cloud_preflight_memory_budget="${PREFLIGHT_MEMORY_BUDGET_BYTES:-$(( cloud_total_memory_bytes * 3 / 4 ))}"
cloud_preflight_owner_peak="${PREFLIGHT_ESTIMATED_OWNER_PEAK_BYTES:-8589934592}"
[[ "${cloud_preflight_memory_budget}" =~ ^[1-9][0-9]*$ ]] \
    || fail "PREFLIGHT_MEMORY_BUDGET_BYTES must be positive"
[[ "${cloud_preflight_owner_peak}" =~ ^[1-9][0-9]*$ ]] \
    || fail "PREFLIGHT_ESTIMATED_OWNER_PEAK_BYTES must be positive"
cloud_memory_lanes="$(( cloud_preflight_memory_budget / cloud_preflight_owner_peak ))"
(( cloud_memory_lanes >= 1 )) || cloud_memory_lanes=1
cloud_preflight_lanes=8
(( cloud_cpu_budget < cloud_preflight_lanes )) \
    && cloud_preflight_lanes="${cloud_cpu_budget}"
(( cloud_memory_lanes < cloud_preflight_lanes )) \
    && cloud_preflight_lanes="${cloud_memory_lanes}"

cloud_target_open_files=65536
cloud_hard_open_files="$(ulimit -H -n)"
if [[ "${cloud_hard_open_files}" == "unlimited" ]]; then
    cloud_desired_open_files="${cloud_target_open_files}"
elif (( cloud_hard_open_files < 4096 )); then
    fail "hard open-file limit is below 4096"
elif (( cloud_hard_open_files < cloud_target_open_files )); then
    cloud_desired_open_files="${cloud_hard_open_files}"
else
    cloud_desired_open_files="${cloud_target_open_files}"
fi
ulimit -S -n "${cloud_desired_open_files}"

mkdir -p "${cloud_repo_root}/artifacts"
cloud_snapshot="${CLOUD_SOURCE_SNAPSHOT_ROOT:-${cloud_repo_root}/artifacts/production_source_snapshot_current}"
CLOUD_REPOSITORY="${cloud_repo_root}" \
CLOUD_SNAPSHOT="${cloud_snapshot}" \
PYTHONPATH="${cloud_repo_root}" \
    "${cloud_python}" -P - <<'PY'
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

repository = Path(os.environ["CLOUD_REPOSITORY"]).resolve(strict=True)
target = Path(os.environ["CLOUD_SNAPSHOT"])
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
            "production_source_snapshot_current is stale; choose a fresh "
            "CLOUD_SOURCE_SNAPSHOT_ROOT rather than overwriting it"
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
cloud_snapshot_identity="$(
    CLOUD_SNAPSHOT="${cloud_snapshot}" \
    PYTHONPATH="${cloud_snapshot}" \
        "${cloud_python}" -P - <<'PY'
import os
from oci.inference.production_source_snapshot import validate_production_source_snapshot

print(validate_production_source_snapshot(os.environ["CLOUD_SNAPSHOT"]).content_sha256)
PY
)"
[[ "${cloud_snapshot_identity}" =~ ^[0-9a-f]{64}$ ]] \
    || fail "source snapshot identity is invalid"

cloud_run_base="${CLOUD_RUN_ROOT_BASE:-${cloud_repo_root}/artifacts/cloud_runs}"
cloud_scratch_base="${CLOUD_SCRATCH_ROOT_BASE:-${cloud_repo_root}/artifacts/cloud_scratch}"
cloud_durable_root="${cloud_run_base}/${CLOUD_RUN_KEY}"
cloud_scratch_root="${cloud_scratch_base}/${CLOUD_RUN_KEY}"
cloud_profile_directory="${cloud_repo_root}/artifacts/runtime_profiles/current"
cloud_deployment="${cloud_profile_directory}/${CLOUD_RUN_KEY}.json"
cloud_embedding_batch_size="${EMBEDDING_BATCH_SIZE:-8}"
cloud_vllm_runtime_root="${cloud_scratch_base}/vllm_${CLOUD_RUN_KEY}"
cloud_vllm_log="${cloud_vllm_runtime_root}/vllm.log"
cloud_vllm_status="${cloud_vllm_runtime_root}/status.json"
cloud_vllm_proxy_log="${cloud_vllm_runtime_root}/proxy.log"
cloud_vllm_gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
cloud_vllm_startup_timeout="${VLLM_STARTUP_TIMEOUT_SECONDS:-600}"
mkdir -p \
    "${cloud_run_base}" \
    "${cloud_scratch_base}" \
    "${cloud_profile_directory}" \
    "${cloud_vllm_runtime_root}"

PYTHONPATH="${cloud_snapshot}" \
    "${cloud_python}" -P \
    "${cloud_snapshot}/scripts/build_cloud_deployment_profile.py" \
    --base "${cloud_deployment_base}" \
    --target "${cloud_deployment}" \
    --dataset "${cloud_dataset}" \
    --durable-root "${cloud_durable_root}" \
    --scratch-root "${cloud_scratch_root}" \
    --embedding-model "${cloud_embedding_model}" \
    --htr-model "${cloud_htr_model}" \
    --stage2-tokenizer "${cloud_stage2_tokenizer}" \
    --stage1-profile "${cloud_stage1}" \
    --query-profile "${cloud_query}" \
    --cpu-budget "${cloud_cpu_budget}" \
    --preflight-memory-budget "${cloud_preflight_memory_budget}" \
    --preflight-owner-peak "${cloud_preflight_owner_peak}" \
    --preflight-lanes "${cloud_preflight_lanes}" \
    --embedding-batch-size "${cloud_embedding_batch_size}" \
    --endpoint "${cloud_endpoint}" \
    --endpoint-model "${cloud_endpoint_model}"

cloud_lock_directory="${cloud_repo_root}/artifacts/production_launch_locks"
cloud_lock_path="${cloud_lock_directory}/${CLOUD_RUN_KEY}.lock"
mkdir -p "${cloud_lock_directory}"
exec 9>"${cloud_lock_path}"
flock -n 9 || fail "another launcher owns ${CLOUD_RUN_KEY}"

cloud_resume_arguments=()
if [[ -L "${cloud_durable_root}" ]]; then
    fail "durable run root is a symlink"
elif [[ -d "${cloud_durable_root}" ]]; then
    require_file "${cloud_durable_root}/immutable_run_request.json"
    cloud_resume_arguments=(--resume)
    note "resuming the same cold-start request at sealed boundaries"
elif [[ -e "${cloud_durable_root}" ]]; then
    fail "durable run root exists but is not a directory"
else
    note "starting a new request with no adopted checkpoints or cache imports"
fi

cloud_stop_arguments=()
[[ -n "${cloud_stop_after}" ]] \
    && cloud_stop_arguments=(--stop-after "${cloud_stop_after}")
cloud_workflow_arguments=(
    --scientific-spec "${cloud_scientific}"
    --deployment-profile "${cloud_deployment}"
    --source-snapshot-root "${cloud_snapshot}"
    --validation-depth fresh_terminal_audit
    --log-level INFO
    "${cloud_stop_arguments[@]}"
    "${cloud_resume_arguments[@]}"
)

note "dataset: ${cloud_dataset}"
note "GPUs: cuda:0 through cuda:7; eight owner lanes"
note "embedding: eight model workers, canonical batch size ${cloud_embedding_batch_size}"
note "CPU budget: ${cloud_cpu_budget}; preflight lanes: ${cloud_preflight_lanes}"
note "durable root: ${cloud_durable_root}"
note "scratch root: ${cloud_scratch_root}"
note "deployment profile: ${cloud_deployment}"
note "source snapshot: ${cloud_snapshot} (${cloud_snapshot_identity})"
note "HTR pooling: token_attention; HTR folds per owner/GPU: 1"
note "Stage 2 endpoint: ${cloud_endpoint}"
note "Stage 2 model: ${cloud_endpoint_model}"
note "Stage 2 vLLM: ModelOpt NVFP4, tensor parallel 8, 256K context"
note "Stage 2 scheduling: vLLM starts lazily only after the first post-Stage-1 request"
note "stop-after: ${cloud_stop_after:-none}"

if (( cloud_prepare_only == 1 )); then
    note "environment, models, source snapshot, and deployment are ready"
    exit 0
fi

cloud_mpl_directory="${cloud_scratch_base}/mpl_${CLOUD_RUN_KEY}"
mkdir -p "${cloud_mpl_directory}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${cloud_visible_devices}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${cloud_mpl_directory}"
export OCI_PRODUCTION_SOURCE_SNAPSHOT_SHA256="${cloud_snapshot_identity}"
export PYTHONHASHSEED=42
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${cloud_snapshot}"
export OCI_STAGE2_ENDPOINT_AUTH="${cloud_endpoint_auth}"
export OCI_STAGE2_ENDPOINT_TRANSPORT="${cloud_endpoint_transport}"
unset OCI_STAGE2_ENDPOINT_API_KEY

cloud_vllm_proxy_arguments=(
    --listen-port "${cloud_vllm_proxy_port}"
    --upstream-port "${cloud_vllm_upstream_port}"
    --vllm-command "${cloud_vllm_command}"
    --model-dir "${cloud_stage2_vllm_model}"
    --served-model-name "${cloud_endpoint_model}"
    --tensor-parallel-size 8
    --max-model-len 262144
    --gpu-memory-utilization "${cloud_vllm_gpu_memory_utilization}"
    --max-num-seqs 8
    --startup-timeout-seconds "${cloud_vllm_startup_timeout}"
    --request-timeout-seconds 900
    --log-path "${cloud_vllm_log}"
    --status-path "${cloud_vllm_status}"
)

if (( cloud_check_only == 1 )); then
    "${cloud_python}" -P \
        "${cloud_snapshot}/scripts/run_local_vllm_stage2_proxy.py" \
        "${cloud_vllm_proxy_arguments[@]}" \
        --check-only >/dev/null
    "${cloud_python}" -P - "${cloud_workflow_arguments[@]}" <<'PY'
import sys

from oci.inference.production_all_evidence_workflow import (
    ProductionAllEvidenceWorkflow,
    _default_portable_role_neutral_hooks,
    build_parser,
    options_from_args,
)

arguments = build_parser().parse_args(sys.argv[1:])
options = options_from_args(arguments)
workflow = ProductionAllEvidenceWorkflow(
    options,
    hooks=_default_portable_role_neutral_hooks(options),
)
request = workflow._request_body()
if request["requested_checkpoint_adoptions"]:
    raise SystemExit("cold cloud request unexpectedly adopted checkpoints")
if workflow.query_devices != tuple(f"cuda:{index}" for index in range(8)):
    raise SystemExit("cloud request did not compile eight devices")
if "stage2_endpoint_authentication" in request:
    raise SystemExit("local vLLM request unexpectedly configured endpoint auth")
if "stage2_endpoint_transport" in request:
    raise SystemExit("local vLLM request unexpectedly changed the default vLLM transport")
if options.stage2_prompt_protocol.model_context_window_tokens != 262_144:
    raise SystemExit("cloud request did not compile the complete 256K Stage 2 context")
if options.stage2_prompt_protocol.max_rendered_discovery_prompt_bytes != 440_000:
    raise SystemExit("cloud request did not compile the expanded prompt batching ceiling")
print("[cloud check] exact cold-start request compiled successfully")
PY
    note "all checks passed; workflow was not started"
    exit 0
fi

cloud_proxy_pid=""
cloud_proxy_pgid=""
cloud_workflow_pid=""
cloud_workflow_pgid=""

stop_owned_proxy() {
    if [[ -z "${cloud_proxy_pid}" ]] || ! kill -0 "${cloud_proxy_pid}" 2>/dev/null; then
        return 0
    fi
    local observed_pgid
    observed_pgid="$(ps -o pgid= -p "${cloud_proxy_pid}" | tr -d '[:space:]')"
    if [[ "${observed_pgid}" != "${cloud_proxy_pgid}" \
        || "${cloud_proxy_pgid}" != "${cloud_proxy_pid}" ]]; then
        note "refusing to signal Stage 2 proxy because its process-group identity changed"
        return 0
    fi
    note "sending SIGTERM to the workflow-owned Stage 2 proxy/vLLM supervisor"
    kill -TERM -- "-${cloud_proxy_pgid}" 2>/dev/null || true
    local poll
    for ((poll = 0; poll < 60; poll++)); do
        kill -0 "${cloud_proxy_pid}" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "${cloud_proxy_pid}" 2>/dev/null; then
        note "Stage 2 supervisor is still shutting down; no SIGKILL was sent"
    else
        wait "${cloud_proxy_pid}" 2>/dev/null || true
    fi
}

terminate_owned_workflow() {
    if [[ -z "${cloud_workflow_pid}" ]] \
        || ! kill -0 "${cloud_workflow_pid}" 2>/dev/null; then
        return 0
    fi
    local observed_pgid
    observed_pgid="$(ps -o pgid= -p "${cloud_workflow_pid}" | tr -d '[:space:]')"
    if [[ "${observed_pgid}" == "${cloud_workflow_pgid}" \
        && "${cloud_workflow_pgid}" == "${cloud_workflow_pid}" ]]; then
        note "forwarding SIGTERM to the workflow-owned production process group"
        kill -TERM -- "-${cloud_workflow_pgid}" 2>/dev/null || true
    else
        note "refusing to signal production because its process-group identity changed"
    fi
}

handle_launcher_signal() {
    local status="$1"
    terminate_owned_workflow
    stop_owned_proxy
    exit "${status}"
}

trap stop_owned_proxy EXIT
trap 'handle_launcher_signal 130' INT
trap 'handle_launcher_signal 143' TERM

if [[ -z "${cloud_stop_after}" ]]; then
    note "starting the CPU-only lazy Stage 2 proxy; it will not touch CUDA during Stage 1"
    setsid "${cloud_python}" -P -u \
        "${cloud_snapshot}/scripts/run_local_vllm_stage2_proxy.py" \
        "${cloud_vllm_proxy_arguments[@]}" \
        >"${cloud_vllm_proxy_log}" 2>&1 &
    cloud_proxy_pid="$!"
    cloud_proxy_pgid="$(ps -o pgid= -p "${cloud_proxy_pid}" | tr -d '[:space:]')"
    [[ "${cloud_proxy_pgid}" == "${cloud_proxy_pid}" ]] \
        || fail "Stage 2 proxy did not acquire a disjoint owned process group"
    CLOUD_PROXY_PID="${cloud_proxy_pid}" \
    CLOUD_PROXY_PORT="${cloud_vllm_proxy_port}" \
        "${cloud_python}" -P - <<'PY'
import json
import os
import time
import urllib.error
import urllib.request

pid = int(os.environ["CLOUD_PROXY_PID"])
url = f"http://127.0.0.1:{os.environ['CLOUD_PROXY_PORT']}/proxy-health"
deadline = time.monotonic() + 30.0
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError as exc:
        raise SystemExit("lazy Stage 2 proxy exited during startup") from exc
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if response.status == 200 and payload.get("vllm_started") is False:
            break
    except (OSError, UnicodeError, json.JSONDecodeError, urllib.error.URLError):
        pass
    time.sleep(0.25)
else:
    raise SystemExit("lazy Stage 2 proxy did not become ready")
PY
    note "lazy Stage 2 proxy is ready; vLLM has not started and GPUs remain free"
fi

note "starting production workflow"
setsid "${cloud_python}" -P -u \
    "${cloud_snapshot}/scripts/run_production_all_evidence_workflow.py" \
    "${cloud_workflow_arguments[@]}" &
cloud_workflow_pid="$!"
cloud_workflow_pgid="$(ps -o pgid= -p "${cloud_workflow_pid}" | tr -d '[:space:]')"
[[ "${cloud_workflow_pgid}" == "${cloud_workflow_pid}" ]] \
    || fail "production workflow did not acquire a disjoint owned process group"
set +e
wait "${cloud_workflow_pid}"
cloud_workflow_status="$?"
set -e
cloud_workflow_pid=""
cloud_workflow_pgid=""
if (( cloud_workflow_status != 0 )); then
    if [[ -f "${cloud_vllm_proxy_log}" ]]; then
        note "recent Stage 2 proxy log follows"
        tail -n 80 -- "${cloud_vllm_proxy_log}" >&2 || true
    fi
    if [[ -f "${cloud_vllm_log}" ]]; then
        note "recent vLLM log follows"
        tail -n 120 -- "${cloud_vllm_log}" >&2 || true
    fi
    exit "${cloud_workflow_status}"
fi
note "production workflow completed successfully"
