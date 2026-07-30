#!/usr/bin/env bash
#
# Immutable-snapshot launcher for the five-confounder/five-effect-modifier
# all-evidence workflow on physical GPUs 0 and 1 of the shared-/data1 host.
#
# The repository-level wrapper invokes this copy from the selected production
# source snapshot.  Repeating the normal invocation resumes the same immutable
# request at phase and Stage 1 component boundaries.

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

readonly REPO_ROOT="/data1/ken/pcori_dev/causal-dragonnet-text"
readonly SCRIPT_PATH="$(realpath -e -- "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIRECTORY="$(dirname -- "${SCRIPT_PATH}")"
readonly SNAPSHOT_ROOT="$(realpath -e -- "${SCRIPT_DIRECTORY}/..")"
readonly SCIENTIFIC_PROFILE_DIRECTORY="${REPO_ROOT}/artifacts/runtime_profiles/portable_all_evidence_r15_token_attention_complete_evidence_v1"
readonly SCIENTIFIC_SPEC="${SCIENTIFIC_PROFILE_DIRECTORY}/portable_all_evidence_scientific_nsclc.json"
readonly STAGE1_PROFILE="${SCIENTIFIC_PROFILE_DIRECTORY}/production_all_evidence_stage1_full.json"
readonly QUERY_PROFILE="${SCIENTIFIC_PROFILE_DIRECTORY}/production_all_evidence_neural_query_full.json"
readonly BASE_DEPLOYMENT_PROFILE="${REPO_ROOT}/artifacts/runtime_profiles/portable_all_evidence_deployment_nsclc.r14-parallel-owner-component-resume-gpu012-until-0900.json"
readonly DATASET="${REPO_ROOT}/synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
readonly EMBEDDING_MODEL="${REPO_ROOT}/artifacts/local_models/qwen3_embedding_8b_materialized"
readonly HTR_MODEL="${REPO_ROOT}/artifacts/local_models/bert_tiny_6f75de8b60a9_materialized"
readonly STAGE2_TOKENIZER="${REPO_ROOT}/artifacts/local_models/gemma4_26b_a4b_it_fp8_dynamic_tokenizer_materialized"

# The already authenticated five-confounder embedding publication.  The
# embedding cache predates the HTR pooling change and is reusable because its
# phase-specific compatibility projection excludes HTR model/evidence state.
readonly ORIGINAL_FIVE_CONF_ROOT="${REPO_ROOT}/artifacts/production_all_evidence_five_conf_five_mod_1000_r14_high_powered_v1_gpu01"
readonly EMBEDDING_ATTEMPT="${ORIGINAL_FIVE_CONF_ROOT}/phases/embedding_cache/attempt_20260728T002827455911Z"
readonly EMBEDDING_CACHE="${EMBEDDING_ATTEMPT}/embedding_cache"
readonly CACHE_SOURCE_PREPARED_DURABLE="${EMBEDDING_ATTEMPT}/prepared/modeling_cohort.parquet"
readonly CACHE_SOURCE_PREPARATION_MANIFEST="${ORIGINAL_FIVE_CONF_ROOT}/recovery/embedding_cache_source_preparation_manifest.json"
readonly CACHE_SOURCE_PREPARED_HISTORICAL="/tmp/causal_dragonnet_nsclc_five_conf_five_mod_r14_high_powered_v1_gpu01/production_all_evidence_workflow/de716f8bc19d165e18b3dff68a9bea81b7070539a922cd3fb1004edaee8464d2/embedding_cache/attempt_20260728T002827455911Z/prepared/modeling_cohort.parquet"
readonly CACHE_SOURCE_PREPARED_SHA256="ab80ebf1d860086e7087e170bffd65573a20b6971411a091f1815e8bcc52825d"

readonly TARGET_OPEN_FILE_LIMIT=65536
readonly MINIMUM_OPEN_FILE_LIMIT=4096

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '[five-conf token-attention remote] %s\n' "$*"
}

require_file() {
    local path="$1"
    [[ -f "${path}" && ! -L "${path}" && -r "${path}" ]] \
        || fail "required readable regular file is missing: ${path}"
}

require_directory() {
    local path="$1"
    [[ -d "${path}" && ! -L "${path}" && -r "${path}" ]] \
        || fail "required readable directory is missing: ${path}"
}

[[ -n "${HOME:-}" ]] || fail "HOME is unset; cannot resolve ~/thisenv"
readonly REMOTE_PYTHON="${FIVE_CONF_REMOTE_PYTHON:-${HOME}/thisenv/bin/python}"
[[ -x "${REMOTE_PYTHON}" && ! -d "${REMOTE_PYTHON}" ]] \
    || fail "Python is not executable at ${REMOTE_PYTHON}"

case "${1:-}" in
    --inspect-obsolete-token-run|--stop-obsolete-token-run)
        readonly OBSOLETE_CONTROL="${SNAPSHOT_ROOT}/scripts/control_obsolete_five_conf_token_catalog_workflow.py"
        require_file "${OBSOLETE_CONTROL}"
        if [[ "${1}" == "--inspect-obsolete-token-run" ]]; then
            exec "${REMOTE_PYTHON}" -P "${OBSOLETE_CONTROL}" inspect
        fi
        exec "${REMOTE_PYTHON}" -P "${OBSOLETE_CONTROL}" stop
        ;;
    "")
        readonly CHECK_ONLY=0
        ;;
    --check-only)
        readonly CHECK_ONLY=1
        ;;
    *)
        fail "usage: $0 [--check-only|--inspect-obsolete-token-run|--stop-obsolete-token-run]"
        ;;
esac

for command_name in flock hostname install nproc nvidia-smi realpath sha256sum tr; do
    command -v "${command_name}" >/dev/null 2>&1 \
        || fail "${command_name} is unavailable"
done

[[ "${SNAPSHOT_ROOT}" == "${REPO_ROOT}/artifacts/production_source_snapshot_"* ]] \
    || fail "launcher is not executing from an immutable production snapshot"
require_file "${SNAPSHOT_ROOT}/source_snapshot_manifest.json"

readonly REMOTE_HOSTNAME="$(hostname -s)"
[[ -n "${REMOTE_HOSTNAME}" ]] || fail "could not determine the remote hostname"
readonly REMOTE_HOSTNAME_SAFE="$(
    printf '%s' "${REMOTE_HOSTNAME}" | LC_ALL=C tr -c 'A-Za-z0-9._-' '_'
)"
[[ -n "${REMOTE_HOSTNAME_SAFE}" ]] \
    || fail "remote hostname has no safe characters"

readonly RUN_TAG="${FIVE_CONF_RUN_TAG:-r15_token_attention_htr_stage2_complete_semantic_catalog_v4_remote_${REMOTE_HOSTNAME_SAFE}_gpu01}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || fail "FIVE_CONF_RUN_TAG contains unsupported characters"

readonly SUPERSEDED_TOKEN_ROOT="${REPO_ROOT}/artifacts/production_all_evidence_five_conf_five_mod_1000_r15_token_attention_complete_evidence_cache_import_v3_remote_${REMOTE_HOSTNAME_SAFE}_gpu01"
readonly SUPERSEDED_TOKEN_SCRATCH_ROOT="${REPO_ROOT}/artifacts/production_scratch/five_conf_five_mod_1000_r15_token_attention_complete_evidence_cache_import_v3_remote_${REMOTE_HOSTNAME_SAFE}_gpu01"
readonly DURABLE_ROOT="${FIVE_CONF_DURABLE_ROOT:-${REPO_ROOT}/artifacts/production_all_evidence_five_conf_five_mod_1000_${RUN_TAG}}"
readonly SCRATCH_ROOT="${FIVE_CONF_SCRATCH_ROOT:-${SUPERSEDED_TOKEN_SCRATCH_ROOT}}"
readonly GENERATED_PROFILE_DIRECTORY="${REPO_ROOT}/artifacts/runtime_profiles/generated"
readonly DEPLOYMENT_PROFILE="${GENERATED_PROFILE_DIRECTORY}/portable_all_evidence_deployment_nsclc.five-conf-five-mod.${RUN_TAG}.json"
readonly ENDPOINT="${FIVE_CONF_ENDPOINT:-http://127.0.0.1:8002/v1}"
readonly ENDPOINT_MODEL="${FIVE_CONF_ENDPOINT_MODEL:-gemma4-26B}"
readonly MPL_CONFIG_DIRECTORY="/tmp/causal_dragonnet_mpl_${RUN_TAG}"
readonly LAUNCH_LOCK_DIRECTORY="${REPO_ROOT}/artifacts/production_launch_locks"
readonly LAUNCH_LOCK_PATH="${LAUNCH_LOCK_DIRECTORY}/five_conf_five_mod_1000_${RUN_TAG}.lock"
readonly ADOPT_INPUT_PREPARATION_CHECKPOINT="${SUPERSEDED_TOKEN_ROOT}/portable_checkpoints/input_preparation"
readonly ADOPT_EMBEDDING_CACHE_CHECKPOINT="${SUPERSEDED_TOKEN_ROOT}/portable_checkpoints/embedding_cache"

[[ "${DURABLE_ROOT}" == "${REPO_ROOT}/artifacts/"* ]] \
    || fail "durable root must remain below ${REPO_ROOT}/artifacts"
[[ "${SCRATCH_ROOT}" == "${REPO_ROOT}/artifacts/"* ]] \
    || fail "scratch root must remain below ${REPO_ROOT}/artifacts"
[[ "${DURABLE_ROOT}" != "${SCRATCH_ROOT}" ]] \
    || fail "durable and scratch roots must differ"
[[ "${RUN_TAG}" == *token_attention* ]] \
    || fail "fresh run tag must explicitly identify token-attention science"

require_file "${SCIENTIFIC_SPEC}"
require_file "${STAGE1_PROFILE}"
require_file "${QUERY_PROFILE}"
require_file "${BASE_DEPLOYMENT_PROFILE}"
require_file "${DATASET}"
require_directory "${EMBEDDING_MODEL}"
require_directory "${HTR_MODEL}"
require_directory "${STAGE2_TOKENIZER}"
require_directory "${EMBEDDING_CACHE}"
require_file "${EMBEDDING_CACHE}/metadata.json"
require_file "${CACHE_SOURCE_PREPARED_DURABLE}"
require_file "${CACHE_SOURCE_PREPARATION_MANIFEST}"
require_directory "${ADOPT_INPUT_PREPARATION_CHECKPOINT}"
require_file "${ADOPT_INPUT_PREPARATION_CHECKPOINT}/artifact_manifest.json"
require_file "${ADOPT_INPUT_PREPARATION_CHECKPOINT}/artifact_locator.json"
require_directory "${ADOPT_EMBEDDING_CACHE_CHECKPOINT}"
require_file "${ADOPT_EMBEDDING_CACHE_CHECKPOINT}/artifact_manifest.json"
require_file "${ADOPT_EMBEDDING_CACHE_CHECKPOINT}/artifact_locator.json"
readonly OBSOLETE_CONTROL="${SNAPSHOT_ROOT}/scripts/control_obsolete_five_conf_token_catalog_workflow.py"
require_file "${OBSOLETE_CONTROL}"
"${REMOTE_PYTHON}" -P "${OBSOLETE_CONTROL}" assert-stopped

ADOPTION_ARGUMENTS=(
    --adopt-checkpoint "${ADOPT_INPUT_PREPARATION_CHECKPOINT}"
    --adopt-checkpoint "${ADOPT_EMBEDDING_CACHE_CHECKPOINT}"
)
readonly -a ADOPTION_ARGUMENTS
readonly EXPECTED_ADOPTED_PHASES="input_preparation,embedding_cache"

readonly SNAPSHOT_CONTENT_SHA256="$(
    SNAPSHOT_TO_VALIDATE="${SNAPSHOT_ROOT}" \
    PYTHONPATH="${SNAPSHOT_ROOT}" \
        "${REMOTE_PYTHON}" -P - <<'PY'
import os
from oci.inference.production_source_snapshot import (
    validate_production_source_snapshot,
)

snapshot = validate_production_source_snapshot(
    os.environ["SNAPSHOT_TO_VALIDATE"]
)
print(snapshot.content_sha256)
PY
)"
[[ "${SNAPSHOT_CONTENT_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
    || fail "source snapshot did not yield a valid authenticated identity"

SCIENTIFIC_SPEC_PATH="${SCIENTIFIC_SPEC}" \
STAGE1_PROFILE_PATH="${STAGE1_PROFILE}" \
    "${REMOTE_PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

scientific = json.loads(
    Path(os.environ["SCIENTIFIC_SPEC_PATH"]).read_text(encoding="utf-8")
)
main_htr = scientific["architecture_profiles"][
    "hierarchical_transformer"
]["producer_configuration"]
matched_htr = scientific["architecture_profiles"][
    "matched_patient_uplift"
]["producer_configuration"]["htr_extractor"]
stage1 = json.loads(
    Path(os.environ["STAGE1_PROFILE_PATH"]).read_text(encoding="utf-8")
)
legacy_pooling = stage1["config"]["architecture"]["htr_sentence_pooling"]
if (
    main_htr.get("sentence_pooling") != "token_attention"
    or matched_htr.get("sentence_pooling") != "token_attention"
    or legacy_pooling != "token_attention"
):
    raise SystemExit("scientific profile does not require token_attention everywhere")
if (
    main_htr.get("freeze_sentence_encoder") is not False
    or matched_htr.get("freeze_sentence_encoder") is not False
):
    raise SystemExit("bert-tiny must remain fully unfrozen")
PY

[[ ! -L "${LAUNCH_LOCK_DIRECTORY}" ]] \
    || fail "launch-lock directory is a symlink"
mkdir -p "${LAUNCH_LOCK_DIRECTORY}"
[[ ! -L "${LAUNCH_LOCK_PATH}" ]] || fail "launch-lock path is a symlink"
exec 9>"${LAUNCH_LOCK_PATH}"
flock -n 9 || fail "another launcher owns this run tag: ${RUN_TAG}"

[[ ! -L "${MPL_CONFIG_DIRECTORY}" ]] \
    || fail "Matplotlib configuration directory is a symlink"
mkdir -p "${MPL_CONFIG_DIRECTORY}"
chmod 700 "${MPL_CONFIG_DIRECTORY}"

readonly AVAILABLE_CPU_COUNT="$(nproc)"
[[ "${AVAILABLE_CPU_COUNT}" =~ ^[1-9][0-9]*$ ]] \
    || fail "nproc returned an invalid CPU count"
if (( AVAILABLE_CPU_COUNT < 16 )); then
    DEFAULT_CPU_BUDGET="${AVAILABLE_CPU_COUNT}"
else
    DEFAULT_CPU_BUDGET=16
fi
readonly CPU_BUDGET="${FIVE_CONF_CPU_BUDGET:-${DEFAULT_CPU_BUDGET}}"
[[ "${CPU_BUDGET}" =~ ^[1-9][0-9]*$ ]] \
    || fail "FIVE_CONF_CPU_BUDGET must be a positive integer"
(( CPU_BUDGET >= 2 && CPU_BUDGET <= AVAILABLE_CPU_COUNT )) \
    || fail "CPU budget must be between 2 and ${AVAILABLE_CPU_COUNT}"

readonly HARD_OPEN_FILE_LIMIT="$(ulimit -H -n)"
readonly CURRENT_OPEN_FILE_LIMIT="$(ulimit -S -n)"
if [[ "${HARD_OPEN_FILE_LIMIT}" == "unlimited" ]]; then
    DESIRED_OPEN_FILE_LIMIT="${TARGET_OPEN_FILE_LIMIT}"
elif [[ "${HARD_OPEN_FILE_LIMIT}" =~ ^[1-9][0-9]*$ ]] \
    && (( HARD_OPEN_FILE_LIMIT >= TARGET_OPEN_FILE_LIMIT )); then
    DESIRED_OPEN_FILE_LIMIT="${TARGET_OPEN_FILE_LIMIT}"
elif [[ "${HARD_OPEN_FILE_LIMIT}" =~ ^[1-9][0-9]*$ ]] \
    && (( HARD_OPEN_FILE_LIMIT >= MINIMUM_OPEN_FILE_LIMIT )); then
    DESIRED_OPEN_FILE_LIMIT="${HARD_OPEN_FILE_LIMIT}"
else
    fail "hard open-file limit is below ${MINIMUM_OPEN_FILE_LIMIT}"
fi
if [[ "${CURRENT_OPEN_FILE_LIMIT}" != "unlimited" ]] \
    && (( CURRENT_OPEN_FILE_LIMIT < DESIRED_OPEN_FILE_LIMIT )); then
    ulimit -S -n "${DESIRED_OPEN_FILE_LIMIT}" \
        || fail "could not raise the soft open-file limit"
fi
readonly OPEN_FILE_LIMIT_EFFECTIVE="$(ulimit -S -n)"

printf '%s  %s\n' \
    "${CACHE_SOURCE_PREPARED_SHA256}" \
    "${CACHE_SOURCE_PREPARED_DURABLE}" \
    | sha256sum -c - >/dev/null
readonly HISTORICAL_PARENT="$(dirname -- "${CACHE_SOURCE_PREPARED_HISTORICAL}")"
if [[ -L "${CACHE_SOURCE_PREPARED_HISTORICAL}" ]]; then
    fail "historical prepared-cohort target is a symlink"
elif [[ ! -e "${CACHE_SOURCE_PREPARED_HISTORICAL}" ]]; then
    install -d -m 700 "${HISTORICAL_PARENT}"
    install -m 600 \
        "${CACHE_SOURCE_PREPARED_DURABLE}" \
        "${CACHE_SOURCE_PREPARED_HISTORICAL}"
    note "restored the provenance-bound prepared cohort under local /tmp"
fi
require_file "${CACHE_SOURCE_PREPARED_HISTORICAL}"
printf '%s  %s\n' \
    "${CACHE_SOURCE_PREPARED_SHA256}" \
    "${CACHE_SOURCE_PREPARED_HISTORICAL}" \
    | sha256sum -c - >/dev/null

mapfile -t PHYSICAL_GPU_INDICES < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits
)
for REQUIRED_GPU in 0 1; do
    GPU_FOUND=0
    for PHYSICAL_GPU_INDEX in "${PHYSICAL_GPU_INDICES[@]}"; do
        if [[ "${PHYSICAL_GPU_INDEX//[[:space:]]/}" == "${REQUIRED_GPU}" ]]; then
            GPU_FOUND=1
            break
        fi
    done
    (( GPU_FOUND == 1 )) || fail "physical GPU ${REQUIRED_GPU} is absent"
done

note "checking that ~/thisenv exposes physical GPUs 0 and 1"
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1 \
    "${REMOTE_PYTHON}" -P - <<'PY'
import torch

if torch.cuda.device_count() != 2:
    raise SystemExit("physical GPUs 0 and 1 did not map to two logical devices")
for index in range(2):
    print(
        "[five-conf token-attention remote] "
        f"logical cuda:{index}: {torch.cuda.get_device_name(index)}"
    )
PY

[[ ! -L "${GENERATED_PROFILE_DIRECTORY}" ]] \
    || fail "generated-profile directory is a symlink"
mkdir -p "${GENERATED_PROFILE_DIRECTORY}"
FIVE_CONF_PROFILE_BASE="${BASE_DEPLOYMENT_PROFILE}" \
FIVE_CONF_PROFILE_DATASET="${DATASET}" \
FIVE_CONF_PROFILE_DURABLE_ROOT="${DURABLE_ROOT}" \
FIVE_CONF_PROFILE_SCRATCH_ROOT="${SCRATCH_ROOT}" \
FIVE_CONF_PROFILE_CPU_BUDGET="${CPU_BUDGET}" \
FIVE_CONF_PROFILE_ENDPOINT="${ENDPOINT}" \
FIVE_CONF_PROFILE_ENDPOINT_MODEL="${ENDPOINT_MODEL}" \
FIVE_CONF_PROFILE_STAGE1="${STAGE1_PROFILE}" \
FIVE_CONF_PROFILE_QUERY="${QUERY_PROFILE}" \
FIVE_CONF_PROFILE_TARGET="${DEPLOYMENT_PROFILE}" \
    "${REMOTE_PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["FIVE_CONF_PROFILE_BASE"])
target = Path(os.environ["FIVE_CONF_PROFILE_TARGET"])
profile = json.loads(base.read_text(encoding="utf-8"))
dataset = os.environ["FIVE_CONF_PROFILE_DATASET"]
cpu_budget = int(os.environ["FIVE_CONF_PROFILE_CPU_BUDGET"])
profile.update(
    {
        "dataset_path": dataset,
        "oracle_source": dataset,
        "durable_artifact_root": os.environ[
            "FIVE_CONF_PROFILE_DURABLE_ROOT"
        ],
        "scratch_root": os.environ["FIVE_CONF_PROFILE_SCRATCH_ROOT"],
        "devices": ["cuda:0", "cuda:1"],
        "cpu_budget": cpu_budget,
        "endpoint": os.environ["FIVE_CONF_PROFILE_ENDPOINT"],
        "endpoint_model": os.environ["FIVE_CONF_PROFILE_ENDPOINT_MODEL"],
        "stage1_profile_locator": os.environ[
            "FIVE_CONF_PROFILE_STAGE1"
        ],
        "query_profile_locator": os.environ["FIVE_CONF_PROFILE_QUERY"],
    }
)
profile["forest_operational"]["requested_host_cpu_budget"] = cpu_budget
profile["resource_performance_safety"][
    "fail_on_external_gpu_occupants"
] = True
stage1 = profile["stage1_execution"]
stage1.update(
    {
        "resource_kind": "accelerator",
        "device_count": 2,
        "scope_workers_per_device": 1,
        "max_parallel_owners": 2,
    }
)
stage1["neural_query_topology"][
    "mode"
] = "one_context_per_selected_device"
# Each remote A6000 has 48 GiB.  The superseded run proved that two
# simultaneously resident HTR fold models consume the entire device.  Keep
# the two owner lanes (one per GPU), but execute one HTR fold at a time within
# each owner's disjoint lease.
htr_controls = stage1["htr_operational_controls"]
htr_controls["fold_parallelism"] = 1
htr_controls["fold_slots_per_device"] = 1

encoded = (
    json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
).encode("utf-8")
target.parent.mkdir(parents=True, exist_ok=True)
try:
    descriptor = os.open(
        target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
except FileExistsError:
    if target.read_bytes() != encoded:
        raise SystemExit(
            "generated deployment profile differs; choose a fresh run tag"
        )
else:
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
PY

FIVE_CONF_PROFILE_TO_VALIDATE="${DEPLOYMENT_PROFILE}" \
PYTHONPATH="${SNAPSHOT_ROOT}" \
    "${REMOTE_PYTHON}" -P - <<'PY'
import os
from oci.inference.portable_workflow_spec import DeploymentProfile

profile = DeploymentProfile.from_json(
    os.environ["FIVE_CONF_PROFILE_TO_VALIDATE"]
)
if (
    tuple(profile.devices) != ("cuda:0", "cuda:1")
    or profile.stage1_execution.max_parallel_owners != 2
    or profile.stage1_execution.scope_workers_per_device != 1
    or (
        profile.stage1_execution.htr_operational_controls.fold_parallelism
        != 1
    )
    or (
        profile.stage1_execution.htr_operational_controls.fold_slots_per_device
        != 1
    )
):
    raise SystemExit(
        "compiled deployment does not provide two disjoint one-fold lanes"
    )
PY

RESUME_ARGUMENTS=()
if [[ -L "${DURABLE_ROOT}" ]]; then
    fail "durable root is a symlink"
elif [[ -e "${DURABLE_ROOT}" ]]; then
    [[ -d "${DURABLE_ROOT}" ]] \
        || fail "durable root exists but is not a directory"
    require_file "${DURABLE_ROOT}/immutable_run_request.json"
    RESUME_ARGUMENTS=(--resume)
    note "existing immutable request found; component-granular resume is enabled"
else
    note "fresh complete-semantic-catalog request will adopt compatible completed checkpoints"
fi

WORKFLOW_ARGUMENTS=(
    --scientific-spec "${SCIENTIFIC_SPEC}"
    --deployment-profile "${DEPLOYMENT_PROFILE}"
    --source-snapshot-root "${SNAPSHOT_ROOT}"
    --embedding-cache-import "${EMBEDDING_CACHE}"
    --embedding-cache-import-source-prepared
        "${CACHE_SOURCE_PREPARED_HISTORICAL}"
    --embedding-cache-import-source-preparation-manifest
        "${CACHE_SOURCE_PREPARATION_MANIFEST}"
    "${ADOPTION_ARGUMENTS[@]}"
    --validation-depth fresh_terminal_audit
    --log-level INFO
    --stop-after handoff_validation
    "${RESUME_ARGUMENTS[@]}"
)
readonly -a WORKFLOW_ARGUMENTS

note "host: ${REMOTE_HOSTNAME}"
note "physical GPUs: 0,1 (workflow devices cuda:0,cuda:1)"
note "parallel Stage 1 owners: 2"
note "HTR folds per owner/GPU: 1 (A6000 memory-safe)"
note "CPU budget: ${CPU_BUDGET}"
note "open-file limit: ${OPEN_FILE_LIMIT_EFFECTIVE}"
note "source snapshot: ${SNAPSHOT_ROOT}"
note "source identity: ${SNAPSHOT_CONTENT_SHA256}"
note "durable root: ${DURABLE_ROOT}"
note "scratch root: ${SCRATCH_ROOT}"
note "superseded request is stopped; compatible sealed Stage 1 components share this scratch store"
note "Stage 1 preflight will be freshly authenticated for the corrected producer"
note "deployment profile: ${DEPLOYMENT_PROFILE}"
note "HTR pooling: token_attention; all CLS HTR components are ineligible"
note "HTR Stage 2 delivery: exhaustive semantic aggregates and complete reverse indexes"
note "bounded non-mmap .npy replay/authentication: required by this snapshot"

if (( CHECK_ONLY == 1 )); then
    note "validating the exact request/checkpoint graph without creating run state"
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0,1 \
    MPLCONFIGDIR="${MPL_CONFIG_DIRECTORY}" \
    OCI_PRODUCTION_SOURCE_SNAPSHOT_SHA256="${SNAPSHOT_CONTENT_SHA256}" \
    PYTHONHASHSEED=42 \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${SNAPSHOT_ROOT}" \
    FIVE_CONF_EXPECTED_ADOPTED_PHASES="${EXPECTED_ADOPTED_PHASES}" \
        "${REMOTE_PYTHON}" -P - "${WORKFLOW_ARGUMENTS[@]}" <<'PY'
import os
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
adopted = {
    record["substituted_phase"]
    for record in request["requested_checkpoint_adoptions"]
}
expected = set(
    os.environ["FIVE_CONF_EXPECTED_ADOPTED_PHASES"].split(",")
)
if adopted != expected:
    raise SystemExit(f"unexpected adopted phases: {sorted(adopted)}")
print(
    "[five-conf token-attention remote] exact request compatibility "
    "passed; only non-adopted work will be rebuilt"
)
PY
    note "all checks passed; workflow was not started"
    exit 0
fi

note "starting corrected production workflow through handoff_validation"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export MPLCONFIGDIR="${MPL_CONFIG_DIRECTORY}"
export OCI_PRODUCTION_SOURCE_SNAPSHOT_SHA256="${SNAPSHOT_CONTENT_SHA256}"
export PYTHONHASHSEED=42
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SNAPSHOT_ROOT}"

exec "${REMOTE_PYTHON}" -P -u \
    "${SNAPSHOT_ROOT}/scripts/run_production_all_evidence_workflow.py" \
    "${WORKFLOW_ARGUMENTS[@]}"
