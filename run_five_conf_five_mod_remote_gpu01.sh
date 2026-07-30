#!/usr/bin/env bash
#
# One-off launcher for the five-confounder/five-effect-modifier production
# workflow on physical GPUs 0 and 1 of another machine sharing /data1.
#
# First invocation:
#   ./run_five_conf_five_mod_remote_gpu01.sh
#
# Later invocation after an interruption:
#   Run the same command again.  The immutable request is detected and
#   --resume is added automatically.
#
# Optional preflight without starting the workflow:
#   ./run_five_conf_five_mod_remote_gpu01.sh --check-only
#
# Optional overrides (set them before the first invocation):
#   FIVE_CONF_RUN_TAG
#   FIVE_CONF_CPU_BUDGET
#   FIVE_CONF_DURABLE_ROOT
#   FIVE_CONF_SCRATCH_ROOT
#   FIVE_CONF_ENDPOINT
#   FIVE_CONF_ENDPOINT_MODEL
#   FIVE_CONF_REMOTE_PYTHON

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

readonly REPO_ROOT="/data1/ken/pcori_dev/causal-dragonnet-text"
readonly SNAPSHOT_ROOT="${REPO_ROOT}/artifacts/production_source_snapshot_20260729_parallel_owner_component_resume_htr_bounded_npy_v4"
readonly SNAPSHOT_EXPECTED_SHA256="b011caf2cd68139c5bf202d52ac3cbefbbb4bde46a62b01144b09b16ea1113e0"
readonly SCIENTIFIC_SPEC="${REPO_ROOT}/artifacts/runtime_profiles/portable_all_evidence_r14_science_first_frozen_v1/portable_all_evidence_scientific_nsclc.json"
readonly BASE_DEPLOYMENT_PROFILE="${REPO_ROOT}/artifacts/runtime_profiles/portable_all_evidence_deployment_nsclc.r14-parallel-owner-component-resume-gpu012-until-0900.json"
readonly DATASET="${REPO_ROOT}/synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
readonly OLD_FIVE_CONF_ROOT="${REPO_ROOT}/artifacts/production_all_evidence_five_conf_five_mod_1000_r14_high_powered_v1_gpu01"
readonly EMBEDDING_ATTEMPT="${OLD_FIVE_CONF_ROOT}/phases/embedding_cache/attempt_20260728T002827455911Z"
readonly EMBEDDING_CACHE="${EMBEDDING_ATTEMPT}/embedding_cache"
readonly CACHE_SOURCE_PREPARED_DURABLE="${EMBEDDING_ATTEMPT}/prepared/modeling_cohort.parquet"
readonly CACHE_SOURCE_PREPARATION_MANIFEST="${OLD_FIVE_CONF_ROOT}/recovery/embedding_cache_source_preparation_manifest.json"
readonly CACHE_SOURCE_PREPARED_HISTORICAL="/tmp/causal_dragonnet_nsclc_five_conf_five_mod_r14_high_powered_v1_gpu01/production_all_evidence_workflow/de716f8bc19d165e18b3dff68a9bea81b7070539a922cd3fb1004edaee8464d2/embedding_cache/attempt_20260728T002827455911Z/prepared/modeling_cohort.parquet"
readonly CACHE_SOURCE_PREPARED_SHA256="ab80ebf1d860086e7087e170bffd65573a20b6971411a091f1815e8bcc52825d"
readonly EMBEDDING_MODEL="${REPO_ROOT}/artifacts/local_models/qwen3_embedding_8b_materialized"
readonly HTR_MODEL="${REPO_ROOT}/artifacts/local_models/bert_tiny_6f75de8b60a9_materialized"
readonly STAGE2_TOKENIZER="${REPO_ROOT}/artifacts/local_models/gemma4_26b_a4b_it_fp8_dynamic_tokenizer_materialized"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '[five-conf remote] %s\n' "$*"
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

case "${1:-}" in
    "")
        readonly CHECK_ONLY=0
        ;;
    --check-only)
        readonly CHECK_ONLY=1
        ;;
    *)
        fail "usage: $0 [--check-only]"
        ;;
esac

[[ -n "${HOME:-}" ]] || fail "HOME is unset; cannot resolve ~/thisenv"
readonly REMOTE_PYTHON="${FIVE_CONF_REMOTE_PYTHON:-${HOME}/thisenv/bin/python}"
[[ -x "${REMOTE_PYTHON}" && ! -d "${REMOTE_PYTHON}" ]] \
    || fail "Python is not executable at ${REMOTE_PYTHON}"

command -v hostname >/dev/null 2>&1 || fail "hostname is unavailable"
command -v flock >/dev/null 2>&1 || fail "flock is unavailable"
command -v install >/dev/null 2>&1 || fail "install is unavailable"
command -v nproc >/dev/null 2>&1 || fail "nproc is unavailable"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is unavailable"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"
command -v tr >/dev/null 2>&1 || fail "tr is unavailable"

remote_hostname="$(hostname -s)"
[[ -n "${remote_hostname}" ]] || fail "could not determine the remote hostname"
remote_hostname_safe="$(
    printf '%s' "${remote_hostname}" | LC_ALL=C tr -c 'A-Za-z0-9._-' '_'
)"
[[ -n "${remote_hostname_safe}" ]] || fail "remote hostname has no safe characters"

readonly RUN_TAG="${FIVE_CONF_RUN_TAG:-r14_remote_${remote_hostname_safe}_gpu01_htr_bounded_npy_v4}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || fail "FIVE_CONF_RUN_TAG must contain only letters, digits, dot, underscore, and hyphen"

readonly PRIOR_PHASEPUB_RUN_TAG="r14_remote_${remote_hostname_safe}_gpu01_phasepub_v3"
readonly PRIOR_PHASEPUB_ROOT="${REPO_ROOT}/artifacts/production_all_evidence_five_conf_five_mod_1000_${PRIOR_PHASEPUB_RUN_TAG}"
readonly PRIOR_PHASEPUB_SCRATCH_ROOT="${REPO_ROOT}/artifacts/production_scratch/five_conf_five_mod_1000_${PRIOR_PHASEPUB_RUN_TAG}"
readonly DURABLE_ROOT="${FIVE_CONF_DURABLE_ROOT:-${REPO_ROOT}/artifacts/production_all_evidence_five_conf_five_mod_1000_${RUN_TAG}}"
# Reuse the failed v3 request's shared component-store parent.  Its two sealed
# BOW components were byte-copied and freshly validated under the corrected
# v4 producer-code key; incomplete HTR attempts remain in the preserved v3
# namespace and are recomputed.
readonly SCRATCH_ROOT="${FIVE_CONF_SCRATCH_ROOT:-${PRIOR_PHASEPUB_SCRATCH_ROOT}}"
readonly GENERATED_PROFILE_DIRECTORY="${REPO_ROOT}/artifacts/runtime_profiles/generated"
readonly DEPLOYMENT_PROFILE="${GENERATED_PROFILE_DIRECTORY}/portable_all_evidence_deployment_nsclc.five-conf-five-mod.${RUN_TAG}.json"
readonly ENDPOINT="${FIVE_CONF_ENDPOINT:-http://127.0.0.1:8002/v1}"
readonly ENDPOINT_MODEL="${FIVE_CONF_ENDPOINT_MODEL:-gemma4-26B}"
readonly MPL_CONFIG_DIRECTORY="/tmp/causal_dragonnet_mpl_${RUN_TAG}"
readonly LAUNCH_LOCK_DIRECTORY="${REPO_ROOT}/artifacts/production_launch_locks"
readonly LAUNCH_LOCK_PATH="${LAUNCH_LOCK_DIRECTORY}/five_conf_five_mod_1000_${RUN_TAG}.lock"
readonly TARGET_OPEN_FILE_LIMIT=65536
readonly MINIMUM_OPEN_FILE_LIMIT=4096
readonly FAILED_REMOTE_ROOT="${REPO_ROOT}/artifacts/production_all_evidence_five_conf_five_mod_1000_r14_remote_${remote_hostname_safe}_gpu01"
readonly ADOPT_INPUT_PREPARATION_CHECKPOINT="${FAILED_REMOTE_ROOT}/portable_checkpoints/input_preparation"
readonly ADOPT_EMBEDDING_CACHE_CHECKPOINT="${FAILED_REMOTE_ROOT}/portable_checkpoints/embedding_cache"
readonly FAILED_EMBEDDING_PHASE_MANIFEST="${FAILED_REMOTE_ROOT}/phases/embedding_cache/complete_manifest.json"
readonly EXPECTED_COMPONENT_STORE_KEY="4168d2649f1ace5e990538b0956e0d8be465ced33d6837fb3fc47b0e461998e1"
readonly COMPONENT_STORE_ROOT="${SCRATCH_ROOT}/production_all_evidence_workflow/stage1_component_store/${EXPECTED_COMPONENT_STORE_KEY}"
readonly COMPONENT_MIGRATION_ATTESTATION="${REPO_ROOT}/artifacts/operational_controls/five_conf_v3_to_htr_bounded_npy_v4_bow_component_migration.json"

[[ "${DURABLE_ROOT}" == "${REPO_ROOT}/artifacts/"* ]] \
    || fail "durable root must remain below ${REPO_ROOT}/artifacts"
[[ "${SCRATCH_ROOT}" == "${REPO_ROOT}/artifacts/"* ]] \
    || fail "scratch root must remain below ${REPO_ROOT}/artifacts"
[[ "${DURABLE_ROOT}" != "${OLD_FIVE_CONF_ROOT}" ]] \
    || fail "the failed five-confounder durable root must not be reused"
[[ "${DURABLE_ROOT}" != "${PRIOR_PHASEPUB_ROOT}" ]] \
    || fail "the immutable v3 request cannot be reused with the corrected v4 source snapshot"
[[ "${DURABLE_ROOT}" != "${SCRATCH_ROOT}" ]] \
    || fail "durable and scratch roots must be different"

[[ ! -L "${LAUNCH_LOCK_DIRECTORY}" ]] \
    || fail "launch-lock directory is a symlink: ${LAUNCH_LOCK_DIRECTORY}"
mkdir -p "${LAUNCH_LOCK_DIRECTORY}"
[[ ! -L "${LAUNCH_LOCK_PATH}" ]] \
    || fail "launch-lock path is a symlink: ${LAUNCH_LOCK_PATH}"
exec 9>"${LAUNCH_LOCK_PATH}"
flock -n 9 \
    || fail "another launcher already owns this run tag: ${RUN_TAG}"

[[ ! -L "${MPL_CONFIG_DIRECTORY}" ]] \
    || fail "Matplotlib configuration directory is a symlink: ${MPL_CONFIG_DIRECTORY}"
mkdir -p "${MPL_CONFIG_DIRECTORY}"
chmod 700 "${MPL_CONFIG_DIRECTORY}"

available_cpu_count="$(nproc)"
[[ "${available_cpu_count}" =~ ^[1-9][0-9]*$ ]] \
    || fail "nproc returned an invalid CPU count: ${available_cpu_count}"
if (( available_cpu_count < 64 )); then
    default_cpu_budget="${available_cpu_count}"
else
    default_cpu_budget=64
fi
readonly CPU_BUDGET="${FIVE_CONF_CPU_BUDGET:-${default_cpu_budget}}"
[[ "${CPU_BUDGET}" =~ ^[1-9][0-9]*$ ]] \
    || fail "FIVE_CONF_CPU_BUDGET must be a positive integer"
(( CPU_BUDGET >= 2 )) \
    || fail "the two-owner deployment requires a CPU budget of at least 2"
(( CPU_BUDGET <= available_cpu_count )) \
    || fail "CPU budget ${CPU_BUDGET} exceeds the available affinity of ${available_cpu_count}"

# Keep generous descriptor headroom for the model runtimes.  The corrected v4
# HTR validator itself performs bounded read-once array authentication and no
# longer retains one filesystem-backed memory map per immutable array.
hard_open_file_limit="$(ulimit -H -n)"
current_open_file_limit="$(ulimit -S -n)"
if [[ "${hard_open_file_limit}" == "unlimited" ]]; then
    desired_open_file_limit="${TARGET_OPEN_FILE_LIMIT}"
elif [[ "${hard_open_file_limit}" =~ ^[1-9][0-9]*$ ]] \
    && (( hard_open_file_limit >= TARGET_OPEN_FILE_LIMIT )); then
    desired_open_file_limit="${TARGET_OPEN_FILE_LIMIT}"
elif [[ "${hard_open_file_limit}" =~ ^[1-9][0-9]*$ ]] \
    && (( hard_open_file_limit >= MINIMUM_OPEN_FILE_LIMIT )); then
    desired_open_file_limit="${hard_open_file_limit}"
else
    fail "hard open-file limit ${hard_open_file_limit} is below the required ${MINIMUM_OPEN_FILE_LIMIT}"
fi
if [[ "${current_open_file_limit}" != "unlimited" ]]; then
    [[ "${current_open_file_limit}" =~ ^[1-9][0-9]*$ ]] \
        || fail "soft open-file limit is invalid: ${current_open_file_limit}"
    if (( current_open_file_limit < desired_open_file_limit )); then
        ulimit -S -n "${desired_open_file_limit}" \
            || fail "could not raise the soft open-file limit to ${desired_open_file_limit}"
    fi
fi
readonly OPEN_FILE_LIMIT_EFFECTIVE="$(ulimit -S -n)"
if [[ "${OPEN_FILE_LIMIT_EFFECTIVE}" != "unlimited" ]]; then
    [[ "${OPEN_FILE_LIMIT_EFFECTIVE}" =~ ^[1-9][0-9]*$ ]] \
        || fail "effective open-file limit is invalid: ${OPEN_FILE_LIMIT_EFFECTIVE}"
    (( OPEN_FILE_LIMIT_EFFECTIVE >= MINIMUM_OPEN_FILE_LIMIT )) \
        || fail "effective open-file limit ${OPEN_FILE_LIMIT_EFFECTIVE} is below the required ${MINIMUM_OPEN_FILE_LIMIT}"
fi

require_file "${SCIENTIFIC_SPEC}"
require_file "${BASE_DEPLOYMENT_PROFILE}"
require_file "${DATASET}"
require_file "${CACHE_SOURCE_PREPARED_DURABLE}"
require_file "${CACHE_SOURCE_PREPARATION_MANIFEST}"
require_directory "${SNAPSHOT_ROOT}"
require_file "${SNAPSHOT_ROOT}/source_snapshot_manifest.json"
require_directory "${EMBEDDING_CACHE}"
require_file "${EMBEDDING_CACHE}/metadata.json"
require_directory "${EMBEDDING_MODEL}"
require_directory "${HTR_MODEL}"
require_directory "${STAGE2_TOKENIZER}"
require_directory "${ADOPT_INPUT_PREPARATION_CHECKPOINT}"
require_file "${ADOPT_INPUT_PREPARATION_CHECKPOINT}/artifact_manifest.json"
require_file "${ADOPT_INPUT_PREPARATION_CHECKPOINT}/artifact_locator.json"
require_directory "${ADOPT_EMBEDDING_CACHE_CHECKPOINT}"
require_file "${ADOPT_EMBEDDING_CACHE_CHECKPOINT}/artifact_manifest.json"
require_file "${ADOPT_EMBEDDING_CACHE_CHECKPOINT}/artifact_locator.json"
require_file "${FAILED_EMBEDDING_PHASE_MANIFEST}"
require_directory "${COMPONENT_STORE_ROOT}"
require_file "${COMPONENT_STORE_ROOT}/component_store_manifest.json"
require_file "${COMPONENT_STORE_ROOT}/components/outer_001_full/bow/execution_manifest.json"
require_file "${COMPONENT_STORE_ROOT}/components/outer_002_full/bow/execution_manifest.json"
require_file "${COMPONENT_MIGRATION_ATTESTATION}"

adopted_historical_input_path="$(
    "${REMOTE_PYTHON}" -P -c '
import json
import sys
from pathlib import Path

phase = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
attestation_path = Path(
    phase["result"]["cache_identity"]["attestation_path"]
)
attestation = json.loads(
    attestation_path.read_text(encoding="utf-8")
)
print(attestation["fresh_preparation"]["prepared_cohort"]["path"])
' "${FAILED_EMBEDDING_PHASE_MANIFEST}"
)"
require_file "${adopted_historical_input_path}"

actual_snapshot_sha256="$(
    SNAPSHOT_MANIFEST="${SNAPSHOT_ROOT}/source_snapshot_manifest.json" \
        "${REMOTE_PYTHON}" -P -c \
        'import json, os; print(json.load(open(os.environ["SNAPSHOT_MANIFEST"], encoding="utf-8"))["content_sha256"])'
)"
[[ "${actual_snapshot_sha256}" == "${SNAPSHOT_EXPECTED_SHA256}" ]] \
    || fail "source snapshot identity differs from the fixed snapshot"

FIVE_CONF_COMPONENT_STORE_ROOT="${COMPONENT_STORE_ROOT}" \
FIVE_CONF_COMPONENT_STORE_KEY="${EXPECTED_COMPONENT_STORE_KEY}" \
FIVE_CONF_COMPONENT_MIGRATION_ATTESTATION="${COMPONENT_MIGRATION_ATTESTATION}" \
PYTHONPATH="${SNAPSHOT_ROOT}" \
    "${REMOTE_PYTHON}" -P - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from oci.inference.production_all_evidence_workflow import (
    _scientific_callable_identity,
    _sha,
)
from oci.inference.production_role_neutral_producer_factories import (
    PreparedBuildRoleNeutralProducerFactoriesBuilder,
)

store = Path(os.environ["FIVE_CONF_COMPONENT_STORE_ROOT"])
expected_key = os.environ["FIVE_CONF_COMPONENT_STORE_KEY"]
manifest = json.loads(
    (store / "component_store_manifest.json").read_text(encoding="utf-8")
)
manifest_body = {
    key: value for key, value in manifest.items() if key != "content_sha256"
}
if (
    manifest.get("component_store_key") != expected_key
    or manifest.get("content_sha256") != _sha(manifest_body)
    or _sha(manifest.get("compatibility")) != expected_key
):
    raise SystemExit("migrated v4 component-store manifest is invalid")
registered_producer = manifest["compatibility"][
    "component_producer_compatibility"
]
scientific_identity = registered_producer["behavior_state"][
    "scientific_identity"
]
observed_producer = _scientific_callable_identity(
    PreparedBuildRoleNeutralProducerFactoriesBuilder(
        architecture_profiles=scientific_identity[
            "architecture_profiles"
        ],
        runtime_compatibility_class=scientific_identity[
            "runtime_compatibility_class"
        ],
    ),
    explicit_scientific_identity=scientific_identity,
)
if observed_producer != registered_producer:
    raise SystemExit(
        "remote Python computes a different v4 component-producer identity"
    )

attestation = json.loads(
    Path(
        os.environ["FIVE_CONF_COMPONENT_MIGRATION_ATTESTATION"]
    ).read_text(encoding="utf-8")
)
attestation_body = {
    key: value for key, value in attestation.items() if key != "content_sha256"
}
if (
    attestation.get("content_sha256") != _sha(attestation_body)
    or attestation.get("target_component_store") != str(store)
    or attestation.get("target_component_store_key") != expected_key
    or attestation.get("scientific_behavior_identity_equal") is not True
    or attestation.get("incomplete_htr_attempts_migrated") is not False
    or attestation.get("source_store_preserved") is not True
):
    raise SystemExit("v4 BOW component-migration attestation is invalid")

records = attestation.get("migrated_components")
if not isinstance(records, list) or len(records) != 2:
    raise SystemExit("v4 BOW component-migration inventory is invalid")
for record in records:
    owner = record.get("physical_owner_scope_id")
    if owner not in {"outer_001_full", "outer_002_full"}:
        raise SystemExit("v4 BOW migration contains an unexpected owner")
    terminal = (
        store / "components" / owner / "bow" / "execution_manifest.json"
    )
    if (
        hashlib.sha256(terminal.read_bytes()).hexdigest()
        != record.get("execution_manifest_sha256")
    ):
        raise SystemExit(f"migrated BOW terminal changed: {owner}")
PY

printf '%s  %s\n' \
    "${CACHE_SOURCE_PREPARED_SHA256}" \
    "${CACHE_SOURCE_PREPARED_DURABLE}" \
    | sha256sum -c - >/dev/null

historical_parent="$(dirname "${CACHE_SOURCE_PREPARED_HISTORICAL}")"
historical_ancestor="${historical_parent}"
while [[ "${historical_ancestor}" != "/" ]]; do
    [[ ! -L "${historical_ancestor}" ]] \
        || fail "historical prepared-cohort ancestor is a symlink: ${historical_ancestor}"
    historical_ancestor="$(dirname "${historical_ancestor}")"
done
if [[ -L "${CACHE_SOURCE_PREPARED_HISTORICAL}" ]]; then
    fail "historical prepared-cohort target is a symlink: ${CACHE_SOURCE_PREPARED_HISTORICAL}"
elif [[ -e "${CACHE_SOURCE_PREPARED_HISTORICAL}" ]]; then
    [[ -f "${CACHE_SOURCE_PREPARED_HISTORICAL}" ]] \
        || fail "historical prepared-cohort target exists but is not a regular file"
else
    install -d -m 700 "${historical_parent}"
    install -m 600 \
        "${CACHE_SOURCE_PREPARED_DURABLE}" \
        "${CACHE_SOURCE_PREPARED_HISTORICAL}"
    note "restored the provenance-bound prepared cohort under local /tmp"
fi
printf '%s  %s\n' \
    "${CACHE_SOURCE_PREPARED_SHA256}" \
    "${CACHE_SOURCE_PREPARED_HISTORICAL}" \
    | sha256sum -c - >/dev/null
resolved_historical_path="$(
    HISTORICAL_PATH="${CACHE_SOURCE_PREPARED_HISTORICAL}" \
        "${REMOTE_PYTHON}" -P -c \
        'import os; from pathlib import Path; print(Path(os.environ["HISTORICAL_PATH"]).resolve(strict=True))'
)"
[[ "${resolved_historical_path}" == "${CACHE_SOURCE_PREPARED_HISTORICAL}" ]] \
    || fail "historical prepared-cohort path is not canonical"

mapfile -t physical_gpu_indices < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits
)
for required_gpu in 0 1; do
    gpu_found=0
    for physical_gpu_index in "${physical_gpu_indices[@]}"; do
        if [[ "${physical_gpu_index//[[:space:]]/}" == "${required_gpu}" ]]; then
            gpu_found=1
            break
        fi
    done
    (( gpu_found == 1 )) || fail "physical GPU ${required_gpu} is not present"
done

note "checking that ~/thisenv exposes physical GPUs 0 and 1"
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1 \
    "${REMOTE_PYTHON}" -P - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 2:
    raise SystemExit(
        f"CUDA_VISIBLE_DEVICES=0,1 exposed {count} CUDA devices instead of 2"
    )
for index in range(count):
    print(f"[five-conf remote] logical cuda:{index}: {torch.cuda.get_device_name(index)}")
PY

[[ ! -L "${GENERATED_PROFILE_DIRECTORY}" ]] \
    || fail "generated-profile directory is a symlink: ${GENERATED_PROFILE_DIRECTORY}"
mkdir -p "${GENERATED_PROFILE_DIRECTORY}"
FIVE_CONF_PROFILE_BASE="${BASE_DEPLOYMENT_PROFILE}" \
FIVE_CONF_PROFILE_DATASET="${DATASET}" \
FIVE_CONF_PROFILE_DURABLE_ROOT="${DURABLE_ROOT}" \
FIVE_CONF_PROFILE_SCRATCH_ROOT="${SCRATCH_ROOT}" \
FIVE_CONF_PROFILE_CPU_BUDGET="${CPU_BUDGET}" \
FIVE_CONF_PROFILE_ENDPOINT="${ENDPOINT}" \
FIVE_CONF_PROFILE_ENDPOINT_MODEL="${ENDPOINT_MODEL}" \
FIVE_CONF_PROFILE_TARGET="${DEPLOYMENT_PROFILE}" \
    "${REMOTE_PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

base_path = Path(os.environ["FIVE_CONF_PROFILE_BASE"])
target_path = Path(os.environ["FIVE_CONF_PROFILE_TARGET"])
if target_path.is_symlink():
    raise SystemExit(f"generated deployment profile is a symlink: {target_path}")
profile = json.loads(base_path.read_text(encoding="utf-8"))

dataset = os.environ["FIVE_CONF_PROFILE_DATASET"]
cpu_budget = int(os.environ["FIVE_CONF_PROFILE_CPU_BUDGET"])
profile["dataset_path"] = dataset
profile["oracle_source"] = dataset
profile["durable_artifact_root"] = os.environ["FIVE_CONF_PROFILE_DURABLE_ROOT"]
profile["scratch_root"] = os.environ["FIVE_CONF_PROFILE_SCRATCH_ROOT"]
profile["devices"] = ["cuda:0", "cuda:1"]
profile["cpu_budget"] = cpu_budget
profile["endpoint"] = os.environ["FIVE_CONF_PROFILE_ENDPOINT"]
profile["endpoint_model"] = os.environ["FIVE_CONF_PROFILE_ENDPOINT_MODEL"]
profile["forest_operational"]["requested_host_cpu_budget"] = cpu_budget
profile["resource_performance_safety"]["fail_on_external_gpu_occupants"] = True

stage1 = profile["stage1_execution"]
stage1["resource_kind"] = "accelerator"
stage1["device_count"] = 2
stage1["scope_workers_per_device"] = 1
stage1["max_parallel_owners"] = 2
stage1["neural_query_topology"]["mode"] = "one_context_per_selected_device"

encoded = (
    json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
).encode("utf-8")
target_path.parent.mkdir(parents=True, exist_ok=True)
try:
    descriptor = os.open(
        target_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
except FileExistsError:
    existing = target_path.read_bytes()
    if existing != encoded:
        raise SystemExit(
            "generated deployment profile already exists with different bytes; "
            "use a new FIVE_CONF_RUN_TAG or restore the original profile"
        )
else:
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
PY

FIVE_CONF_PROFILE_TO_VALIDATE="${DEPLOYMENT_PROFILE}" \
MPLCONFIGDIR="${MPL_CONFIG_DIRECTORY}" \
PYTHONNOUSERSITE=1 \
PYTHONPATH="${SNAPSHOT_ROOT}" \
    "${REMOTE_PYTHON}" -P -c \
    'import os; from oci.inference.portable_workflow_spec import DeploymentProfile; DeploymentProfile.from_json(os.environ["FIVE_CONF_PROFILE_TO_VALIDATE"])'

resume_arguments=()
if [[ -L "${DURABLE_ROOT}" ]]; then
    fail "durable root is a symlink: ${DURABLE_ROOT}"
elif [[ -e "${DURABLE_ROOT}" ]]; then
    [[ -d "${DURABLE_ROOT}" ]] \
        || fail "durable root exists but is not a directory: ${DURABLE_ROOT}"
    [[ -f "${DURABLE_ROOT}/immutable_run_request.json" ]] \
        || fail "durable root exists without an immutable request; refusing to reuse it"
    resume_arguments=(--resume)
    note "existing immutable request found; component-granular resume is enabled"
else
    note "fresh corrected request will adopt completed input and embedding checkpoints"
    note "Stage 1 preflight will be rebuilt under the corrected v4 producer identity"
fi

workflow_arguments=(
    --scientific-spec "${SCIENTIFIC_SPEC}"
    --deployment-profile "${DEPLOYMENT_PROFILE}"
    --source-snapshot-root "${SNAPSHOT_ROOT}"
    --embedding-cache-import "${EMBEDDING_CACHE}"
    --embedding-cache-import-source-prepared
        "${CACHE_SOURCE_PREPARED_HISTORICAL}"
    --embedding-cache-import-source-preparation-manifest
        "${CACHE_SOURCE_PREPARATION_MANIFEST}"
    --adopt-checkpoint "${ADOPT_INPUT_PREPARATION_CHECKPOINT}"
    --adopt-checkpoint "${ADOPT_EMBEDDING_CACHE_CHECKPOINT}"
    --validation-depth fresh_terminal_audit
    --log-level INFO
    --stop-after handoff_validation
    "${resume_arguments[@]}"
)
readonly -a workflow_arguments

note "host: ${remote_hostname}"
note "physical GPUs: 0,1 (workflow devices cuda:0,cuda:1)"
note "parallel Stage 1 owners: 2"
note "CPU budget: ${CPU_BUDGET}"
note "open-file limit: ${OPEN_FILE_LIMIT_EFFECTIVE}"
note "source snapshot: ${SNAPSHOT_ROOT}"
note "durable root: ${DURABLE_ROOT}"
note "shared scratch root: ${SCRATCH_ROOT}"
note "deployment profile: ${DEPLOYMENT_PROFILE}"
note "adopting input checkpoint: ${ADOPT_INPUT_PREPARATION_CHECKPOINT}"
note "adopting embedding checkpoint: ${ADOPT_EMBEDDING_CACHE_CHECKPOINT}"
note "Stage 1 preflight: recompute under the corrected v4 source snapshot"
note "migrated component store: ${COMPONENT_STORE_ROOT}"
note "two sealed v3 BOW components will be freshly authenticated before reuse"

if (( CHECK_ONLY == 1 )); then
    note "validating the exact immutable request and checkpoint graph without creating run state"
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0,1 \
    MPLCONFIGDIR="${MPL_CONFIG_DIRECTORY}" \
    OCI_PRODUCTION_SOURCE_SNAPSHOT_SHA256="${SNAPSHOT_EXPECTED_SHA256}" \
    PYTHONHASHSEED=42 \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${SNAPSHOT_ROOT}" \
        "${REMOTE_PYTHON}" -P - "${workflow_arguments[@]}" <<'PY'
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
adopted_phases = {
    record["substituted_phase"]
    for record in request["requested_checkpoint_adoptions"]
}
expected_adopted_phases = {"input_preparation", "embedding_cache"}
if adopted_phases != expected_adopted_phases:
    raise SystemExit(
        "exact request adopted unexpected phases: "
        f"{sorted(adopted_phases)}"
    )
print(
    "[five-conf remote] exact request compatibility passed; "
    "input and embedding are reusable and Stage 1 preflight will recompute"
)
PY
    note "all launcher and exact-request checks passed; workflow was not started"
    exit 0
fi

note "starting production workflow through handoff_validation"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export MPLCONFIGDIR="${MPL_CONFIG_DIRECTORY}"
export PYTHONHASHSEED=42
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SNAPSHOT_ROOT}"

exec "${REMOTE_PYTHON}" -P -u \
    "${SNAPSHOT_ROOT}/scripts/run_production_all_evidence_workflow.py" \
    "${workflow_arguments[@]}"
