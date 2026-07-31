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

cloud_endpoint="${STAGE2_ENDPOINT:-https://generativelanguage.googleapis.com/v1beta/openai/}"
cloud_endpoint_model="${STAGE2_ENDPOINT_MODEL:-gemini-3.6-flash}"
cloud_endpoint_auth="${OCI_STAGE2_ENDPOINT_AUTH:-api_key}"
cloud_endpoint_transport="${OCI_STAGE2_ENDPOINT_TRANSPORT:-openai_compatible}"
[[ "${cloud_endpoint_auth}" == "api_key" ]] \
    || fail "OCI_STAGE2_ENDPOINT_AUTH must be api_key for this Gemini launcher"
[[ "${cloud_endpoint_transport}" == "openai_compatible" ]] \
    || fail "OCI_STAGE2_ENDPOINT_TRANSPORT must be openai_compatible for this Gemini launcher"
if [[ -n "${OCI_STAGE2_ENDPOINT_API_KEY:-}" \
    && -n "${GEMINI_API_KEY:-}" \
    && "${OCI_STAGE2_ENDPOINT_API_KEY}" != "${GEMINI_API_KEY}" ]]; then
    fail "OCI_STAGE2_ENDPOINT_API_KEY and GEMINI_API_KEY disagree"
fi
cloud_endpoint_api_key="${OCI_STAGE2_ENDPOINT_API_KEY:-${GEMINI_API_KEY:-}}"
if (( cloud_prepare_only == 0 )) && [[ -z "${cloud_endpoint_api_key}" ]]; then
    fail "GEMINI_API_KEY is required for check-only and full workflow execution"
fi

for cloud_command in awk flock nproc nvidia-smi realpath uv; do
    command -v "${cloud_command}" >/dev/null 2>&1 \
        || fail "required command is unavailable: ${cloud_command}"
done

cloud_uv_cache="${UV_CACHE_DIR:-${cloud_repo_root}/.uv-cache}"
if [[ "${SKIP_UV_SYNC:-0}" != "1" ]]; then
    note "syncing the locked Python environment"
    UV_CACHE_DIR="${cloud_uv_cache}" uv sync \
        --frozen \
        --extra extraction \
        --extra gemini
fi
cloud_python="${CLOUD_PYTHON:-${cloud_repo_root}/.venv/bin/python}"
[[ -x "${cloud_python}" && ! -d "${cloud_python}" ]] \
    || fail "uv environment Python is absent: ${cloud_python}"

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
if [[ -n "${STAGE2_TOKENIZER_DIR:-}" ]]; then
    require_directory "${cloud_stage2_tokenizer}"
else
    materialize_model \
        tokenizer \
        "${STAGE2_TOKENIZER_MODEL_ID:-RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic}" \
        "${STAGE2_TOKENIZER_REVISION:-30dd81263d7400b11032161ea3d8a6765557a4a1}" \
        "${cloud_stage2_tokenizer}"
fi
require_directory "${cloud_embedding_model}"
require_directory "${cloud_htr_model}"
require_directory "${cloud_stage2_tokenizer}"

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
    print(
        f"[cloud gpu] cuda:{index}: {properties.name}; "
        f"memory={properties.total_memory / (1024 ** 3):.1f} GiB"
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
mkdir -p \
    "${cloud_run_base}" \
    "${cloud_scratch_base}" \
    "${cloud_profile_directory}"

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

cloud_stop_after="${STOP_AFTER-}"
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
note "Stage 2 model: ${cloud_endpoint_model} (Gemini OpenAI-compatible transport)"
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
export OCI_STAGE2_ENDPOINT_API_KEY="${cloud_endpoint_api_key}"

if (( cloud_check_only == 1 )); then
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
if request.get("stage2_endpoint_authentication", {}).get("mode") != "api_key":
    raise SystemExit("cloud request did not compile secret-free API-key auth")
if request.get("stage2_endpoint_transport", {}).get("mode") != "openai_compatible":
    raise SystemExit("cloud request did not compile OpenAI-compatible transport")
print("[cloud check] exact cold-start request compiled successfully")
PY
    note "all checks passed; workflow was not started"
    exit 0
fi

note "starting production workflow"
exec "${cloud_python}" -P -u \
    "${cloud_snapshot}/scripts/run_production_all_evidence_workflow.py" \
    "${cloud_workflow_arguments[@]}"
