"""Content-addressed attestation for one running production model deployment.

The attestation collected on the inference host is evidence, not its own trust
anchor.  Production callers must compare the raw root-document SHA-256 with a
digest pinned outside this module.  The root then authenticates five closed
sidecars and their cross-document relationships:

* every regular file in the served model tree;
* every regular file in the server implementation tree and its executable;
* the exact OCI image-manifest bytes and their immutable digest;
* the exact package-inventory source plus its normalized package inventory; and
* the running process, launch arguments, network namespace, and owned listener.

The collector runs with host PID/cgroup visibility but interprets model and
server paths inside the target process mount namespace. It hashes the exact
``/proc/PID/exe`` inode, holds pid/root/namespace descriptors, authenticates the
target UTS hostname, and revalidates the process epoch before publication. It
does not contact the inference API, restart a service, or mutate a deployment.
Its only write is atomic publication into a caller-selected fresh output
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import select
import shutil
import socket
import stat
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

SERVED_DEPLOYMENT_ATTESTATION_IMPLEMENTATION_VERSION = (
    "production_served_deployment_attestation_implementation_v2"
)
SERVED_DEPLOYMENT_ATTESTATION_SCHEMA_VERSION = "production_openai_served_deployment_attestation_v3"
SERVED_MODEL_FILE_MANIFEST_SCHEMA_VERSION = "served_model_file_manifest_v1"
SERVER_IMPLEMENTATION_MANIFEST_SCHEMA_VERSION = "inference_server_implementation_manifest_v1"
CONTAINER_IMAGE_MANIFEST_SCHEMA_VERSION = "inference_container_image_manifest_v1"
PACKAGE_INVENTORY_MANIFEST_SCHEMA_VERSION = "inference_package_inventory_manifest_v1"
LAUNCH_LISTENER_BINDING_SCHEMA_VERSION = "inference_launch_listener_binding_v2"
AUTHENTICATED_SERVED_DEPLOYMENT_IDENTITY_SCHEMA_VERSION = (
    "authenticated_served_deployment_identity_v1"
)

ROOT_FILENAME = "served_deployment_attestation.json"
EVIDENCE_FILENAMES = {
    "model_manifest": "model_manifest.json",
    "server_implementation_manifest": "server_implementation_manifest.json",
    "container_image_manifest": "container_image_manifest.json",
    "package_inventory_manifest": "package_inventory_manifest.json",
    "launch_listener_binding": "launch_listener_binding.json",
}
EVIDENCE_SCHEMAS = {
    "model_manifest": SERVED_MODEL_FILE_MANIFEST_SCHEMA_VERSION,
    "server_implementation_manifest": SERVER_IMPLEMENTATION_MANIFEST_SCHEMA_VERSION,
    "container_image_manifest": CONTAINER_IMAGE_MANIFEST_SCHEMA_VERSION,
    "package_inventory_manifest": PACKAGE_INVENTORY_MANIFEST_SCHEMA_VERSION,
    "launch_listener_binding": LAUNCH_LISTENER_BINDING_SCHEMA_VERSION,
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OCI_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}\Z")
_BOOT_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_PACKAGE_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,127})\Z")
_MAX_ATTESTATION_FILE_BYTES = 64 * 1024 * 1024
_MAX_PROC_RECORD_BYTES = 16 * 1024 * 1024
_ROOT_KEYS = frozenset({"schema_version", "body", "content_sha256"})
_REFERENCE_KEYS = frozenset({"relative_path", "sha256", "size_bytes", "schema_version"})
_FILE_RECORD_KEYS = frozenset({"relative_path", "size_bytes", "sha256"})
_ABSOLUTE_FILE_RECORD_KEYS = frozenset({"path", "size_bytes", "sha256"})
_COLLECTOR_KEYS = frozenset(
    {
        "implementation_version",
        "attestation_module_sha256",
        "helper_script_sha256",
    }
)
_RELATIONSHIP_KEYS = frozenset(
    {
        "model_files_sha256",
        "server_implementation_files_sha256",
        "container_image_digest",
        "container_instance_id",
        "packages_sha256",
        "launch_binding_sha256",
    }
)
_ROOT_BODY_KEYS = frozenset(
    {
        "attestation_scope",
        "endpoint",
        "served_model_name",
        "deployment_instance_id",
        "collector",
        "evidence",
        "relationships",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from exc


def _closed_mapping(value: Any, keys: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} must be a closed object with keys {sorted(keys)!r}")
    return dict(value)


def _nonempty_text(value: Any, *, label: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be bounded nonempty text without surrounding whitespace")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError(f"{label} cannot contain control characters")
    return value


def _sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _positive_size(value: Any, *, label: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _absolute_path_text(value: Any, *, label: str) -> str:
    text = _nonempty_text(value, label=label)
    path = PurePosixPath(text)
    if (
        path.anchor != "/"
        or not path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != text
    ):
        raise ValueError(f"{label} must be a normalized absolute POSIX path")
    return text


def _relative_path_text(value: Any, *, label: str) -> str:
    text = _nonempty_text(value, label=label)
    if "\\" in text:
        raise ValueError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or len(path.parts) == 0
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    if str(path) != text:
        raise ValueError(f"{label} is not canonical")
    return text


def _validate_endpoint(value: Any) -> str:
    endpoint = _nonempty_text(value, label="endpoint", maximum=512)
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1"
        or parsed.port is None
        or parsed.hostname.lower() != parsed.hostname
    ):
        raise ValueError("endpoint must be one exact credential-free http://host:port/v1 URL")
    if endpoint != f"http://{parsed.hostname}:{parsed.port}/v1":
        raise ValueError("endpoint must use its canonical spelling")
    return endpoint


def _validate_wrapped_document(value: Any, *, expected_schema: str, label: str) -> dict[str, Any]:
    document = _closed_mapping(value, _ROOT_KEYS, label=label)
    if document["schema_version"] != expected_schema:
        raise ValueError(f"{label} schema_version is unsupported")
    body = document["body"]
    declared = _sha256(document["content_sha256"], label=f"{label}.content_sha256")
    if content_sha256(body) != declared:
        raise ValueError(f"{label} content hash is invalid")
    if not isinstance(body, Mapping):
        raise ValueError(f"{label}.body must be an object")
    return dict(body)


def _validate_file_records(value: Any, *, label: str) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _closed_mapping(raw, _FILE_RECORD_KEYS, label=f"{label}[{index}]")
        rows.append(
            {
                "relative_path": _relative_path_text(
                    row["relative_path"], label=f"{label}[{index}].relative_path"
                ),
                "size_bytes": _positive_size(
                    row["size_bytes"], label=f"{label}[{index}].size_bytes", allow_zero=True
                ),
                "sha256": _sha256(row["sha256"], label=f"{label}[{index}].sha256"),
            }
        )
    paths = [row["relative_path"] for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{label} paths must be unique and lexicographically sorted")
    return rows, sum(int(row["size_bytes"]) for row in rows)


def _validate_absolute_file_record(value: Any, *, label: str) -> dict[str, Any]:
    row = _closed_mapping(value, _ABSOLUTE_FILE_RECORD_KEYS, label=label)
    return {
        "path": _absolute_path_text(row["path"], label=f"{label}.path"),
        "size_bytes": _positive_size(
            row["size_bytes"], label=f"{label}.size_bytes", allow_zero=True
        ),
        "sha256": _sha256(row["sha256"], label=f"{label}.sha256"),
    }


def _validate_model_manifest(body: Mapping[str, Any], *, model_name: str) -> dict[str, Any]:
    keys = frozenset(
        {
            "served_model_name",
            "model_root",
            "files",
            "file_count",
            "total_file_bytes",
            "files_sha256",
        }
    )
    value = _closed_mapping(body, keys, label="model manifest body")
    if value["served_model_name"] != model_name:
        raise ValueError("model manifest served_model_name does not match the root")
    _absolute_path_text(value["model_root"], label="model_manifest.model_root")
    files, total = _validate_file_records(value["files"], label="model_manifest.files")
    if value["file_count"] != len(files) or value["total_file_bytes"] != total:
        raise ValueError("model manifest file counts or byte totals are inconsistent")
    if value["files_sha256"] != content_sha256(files):
        raise ValueError("model manifest files_sha256 is invalid")
    paths = {row["relative_path"] for row in files}
    if "config.json" not in paths:
        raise ValueError("model manifest must include the exact root config.json")
    if not any(path.endswith((".safetensors", ".bin")) for path in paths):
        raise ValueError("model manifest must include at least one model-weight file")
    return value


def _validate_server_manifest(body: Mapping[str, Any]) -> dict[str, Any]:
    keys = frozenset(
        {
            "server_runtime",
            "implementation_root",
            "server_executable",
            "files",
            "file_count",
            "total_file_bytes",
            "files_sha256",
        }
    )
    value = _closed_mapping(body, keys, label="server implementation manifest body")
    runtime = _nonempty_text(value["server_runtime"], label="server_runtime", maximum=128)
    if runtime != "vllm_openai_compatible":
        raise ValueError("server_runtime must be the pinned vLLM OpenAI-compatible runtime")
    _absolute_path_text(value["implementation_root"], label="implementation_root")
    _validate_absolute_file_record(value["server_executable"], label="server_executable")
    files, total = _validate_file_records(value["files"], label="server.files")
    if value["file_count"] != len(files) or value["total_file_bytes"] != total:
        raise ValueError("server implementation counts or byte totals are inconsistent")
    if value["files_sha256"] != content_sha256(files):
        raise ValueError("server implementation files_sha256 is invalid")
    if not any(row["relative_path"].endswith((".py", ".so")) for row in files):
        raise ValueError("server implementation manifest has no Python or shared-object code")
    return value


def _validate_container_manifest(body: Mapping[str, Any]) -> dict[str, Any]:
    keys = frozenset(
        {
            "container_runtime",
            "image_reference",
            "immutable_image_digest",
            "container_instance_id",
            "oci_manifest_source",
            "runtime_inspect_source",
        }
    )
    value = _closed_mapping(body, keys, label="container image manifest body")
    runtime = _nonempty_text(value["container_runtime"], label="container_runtime", maximum=64)
    if runtime not in {"docker", "containerd", "podman", "kubernetes"}:
        raise ValueError("container_runtime is unsupported")
    _nonempty_text(value["image_reference"], label="image_reference", maximum=512)
    source = _validate_absolute_file_record(
        value["oci_manifest_source"], label="oci_manifest_source"
    )
    _validate_absolute_file_record(value["runtime_inspect_source"], label="runtime_inspect_source")
    if _CONTAINER_ID.fullmatch(str(value["container_instance_id"])) is None:
        raise ValueError("container_instance_id must be one exact lowercase 64-hex ID")
    digest = str(value["immutable_image_digest"])
    if _OCI_DIGEST.fullmatch(digest) is None or digest != f"sha256:{source['sha256']}":
        raise ValueError("immutable image digest must equal the exact OCI manifest byte digest")
    return value


def _validate_package_manifest(body: Mapping[str, Any]) -> dict[str, Any]:
    keys = frozenset(
        {
            "python_executable",
            "inventory_source",
            "packages",
            "package_count",
            "packages_sha256",
        }
    )
    value = _closed_mapping(body, keys, label="package inventory manifest body")
    _validate_absolute_file_record(value["python_executable"], label="python_executable")
    _validate_absolute_file_record(value["inventory_source"], label="package inventory source")
    packages = value["packages"]
    if not isinstance(packages, list) or not packages:
        raise ValueError("package inventory must be nonempty")
    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(packages):
        item = _closed_mapping(raw, frozenset({"name", "version"}), label=f"packages[{index}]")
        name = _nonempty_text(item["name"], label=f"packages[{index}].name", maximum=128)
        if _PACKAGE_NAME.fullmatch(name) is None:
            raise ValueError("package name is invalid")
        version = _nonempty_text(item["version"], label=f"packages[{index}].version", maximum=256)
        normalized.append({"name": name, "version": version})
    ordering = [
        (row["name"].lower().replace("_", "-"), row["name"], row["version"]) for row in normalized
    ]
    if ordering != sorted(ordering) or len({row[0] for row in ordering}) != len(ordering):
        raise ValueError("package inventory must have unique canonically sorted names")
    required = {"vllm", "torch", "transformers"}
    observed = {row["name"].lower().replace("_", "-") for row in normalized}
    if not required.issubset(observed):
        raise ValueError("package inventory must include vllm, torch, and transformers")
    if value["package_count"] != len(normalized):
        raise ValueError("package_count is inconsistent")
    if value["packages_sha256"] != content_sha256(normalized):
        raise ValueError("packages_sha256 is invalid")
    return value


def _deployment_instance_id(body: Mapping[str, Any]) -> str:
    process = body["process"]
    listener = body["listener"]
    relationships = {
        key: value for key, value in body["relationships"].items() if key != "launch_binding_sha256"
    }
    identity = {
        "endpoint": body["endpoint"],
        "served_model_name": body["served_model_name"],
        "hostname": body["hostname"],
        "boot_id": body["boot_id"],
        "pid": process["pid"],
        "start_time_ticks": process["start_time_ticks"],
        "executable_device": process["executable_device"],
        "executable_inode": process["executable_inode"],
        "process_root_device": process["process_root_device"],
        "process_root_inode": process["process_root_inode"],
        "mount_namespace_inode": process["mount_namespace_inode"],
        "network_namespace_inode": process["network_namespace_inode"],
        "uts_namespace_inode": process["uts_namespace_inode"],
        "cmdline_sha256": process["cmdline_sha256"],
        "listener_records_sha256": listener["records_sha256"],
        "relationships": relationships,
    }
    return content_sha256(identity)


def _launch_binding_sha256(body: Mapping[str, Any]) -> str:
    projection = json.loads(canonical_json(body))
    projection.pop("deployment_instance_id", None)
    relationships = dict(projection["relationships"])
    relationships.pop("launch_binding_sha256", None)
    projection["relationships"] = relationships
    return content_sha256(projection)


def _validate_launch_binding(
    body: Mapping[str, Any],
    *,
    endpoint: str,
    model_name: str,
    relationships: Mapping[str, Any],
) -> dict[str, Any]:
    keys = frozenset(
        {
            "endpoint",
            "served_model_name",
            "hostname",
            "boot_id",
            "deployment_instance_id",
            "process",
            "model_launch",
            "listener",
            "relationships",
        }
    )
    value = _closed_mapping(body, keys, label="launch/listener binding body")
    if value["endpoint"] != endpoint or value["served_model_name"] != model_name:
        raise ValueError("launch binding endpoint/model does not match the root")
    parsed = urlsplit(endpoint)
    hostname = _nonempty_text(value["hostname"], label="hostname", maximum=253).lower()
    if hostname != parsed.hostname and not hostname.startswith(f"{parsed.hostname}."):
        raise ValueError("launch host does not match the endpoint hostname")
    boot_id = str(value["boot_id"])
    if _BOOT_ID.fullmatch(boot_id) is None:
        raise ValueError("launch boot_id is invalid")
    process = _closed_mapping(
        value["process"],
        frozenset(
            {
                "pid",
                "start_time_ticks",
                "executable_path",
                "executable_sha256",
                "cmdline_sha256",
                "cgroup_sha256",
                "executable_device",
                "executable_inode",
                "process_root_device",
                "process_root_inode",
                "mount_namespace_inode",
                "network_namespace_inode",
                "uts_namespace_inode",
                "container_instance_id",
            }
        ),
        label="launch process",
    )
    _positive_size(process["pid"], label="process.pid")
    _positive_size(process["start_time_ticks"], label="process.start_time_ticks")
    _absolute_path_text(process["executable_path"], label="process.executable_path")
    for key in ("executable_sha256", "cmdline_sha256", "cgroup_sha256"):
        _sha256(process[key], label=f"process.{key}")
    for key in (
        "executable_device",
        "executable_inode",
        "process_root_device",
        "process_root_inode",
        "mount_namespace_inode",
        "network_namespace_inode",
        "uts_namespace_inode",
    ):
        _positive_size(process[key], label=f"process.{key}", allow_zero=key.endswith("device"))
    if _CONTAINER_ID.fullmatch(str(process["container_instance_id"])) is None:
        raise ValueError("process.container_instance_id is invalid")
    model_launch = _closed_mapping(
        value["model_launch"],
        frozenset({"model_argument", "served_model_names", "served_name_flag_present"}),
        label="model_launch",
    )
    _nonempty_text(model_launch["model_argument"], label="model_launch.model_argument")
    if (
        model_launch["served_model_names"] != [model_name]
        or model_launch["served_name_flag_present"] is not True
    ):
        raise ValueError("launch must explicitly expose exactly the attested served-model name")
    listener = _closed_mapping(
        value["listener"],
        frozenset({"transport", "port", "records", "records_sha256"}),
        label="listener",
    )
    if listener["transport"] != "tcp" or listener["port"] != parsed.port:
        raise ValueError("listener transport/port does not match the endpoint")
    records = listener["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("listener must contain at least one owned LISTEN socket")
    normalized_records: list[dict[str, Any]] = []
    record_keys = frozenset({"address", "port", "state", "socket_inode", "owned_by_process"})
    for index, raw in enumerate(records):
        record = _closed_mapping(raw, record_keys, label=f"listener.records[{index}]")
        try:
            ipaddress.ip_address(str(record["address"]))
        except ValueError as exc:
            raise ValueError("listener address is invalid") from exc
        if (
            record["port"] != parsed.port
            or record["state"] != "LISTEN"
            or record["owned_by_process"] is not True
        ):
            raise ValueError("listener record is not an owned endpoint LISTEN socket")
        _positive_size(record["socket_inode"], label="listener socket inode")
        normalized_records.append(record)
    sort_key = lambda row: (str(row["address"]), int(row["port"]), int(row["socket_inode"]))
    if normalized_records != sorted(normalized_records, key=sort_key):
        raise ValueError("listener records must be canonically sorted")
    if listener["records_sha256"] != content_sha256(normalized_records):
        raise ValueError("listener records_sha256 is invalid")
    if value["relationships"] != dict(relationships):
        raise ValueError("launch relationships do not match the root")
    if value["deployment_instance_id"] != _deployment_instance_id(value):
        raise ValueError("deployment_instance_id is not derived from launch/listener evidence")
    return value


def _module_and_helper_hashes() -> tuple[str, str]:
    module_path = Path(__file__).resolve(strict=True)
    helper = module_path.parents[2] / "scripts" / "build_production_served_model_attestation.py"
    if not helper.is_file():
        raise RuntimeError("served-deployment attestation helper script is unavailable")
    return _hash_file_path(module_path), _hash_file_path(helper)


def _validate_root_and_sidecars(
    root: Mapping[str, Any],
    sidecars: Mapping[str, Mapping[str, Any]],
    *,
    expected_model_name: str,
    expected_endpoint: str,
) -> dict[str, Any]:
    root_body = _validate_wrapped_document(
        root,
        expected_schema=SERVED_DEPLOYMENT_ATTESTATION_SCHEMA_VERSION,
        label="served-deployment attestation root",
    )
    root_body = _closed_mapping(root_body, _ROOT_BODY_KEYS, label="attestation root body")
    if root_body["attestation_scope"] != "one_exact_running_openai_compatible_deployment":
        raise ValueError("served-deployment attestation scope is unsupported")
    endpoint = _validate_endpoint(root_body["endpoint"])
    if endpoint != expected_endpoint:
        raise ValueError("served-deployment attestation endpoint differs from production")
    model_name = _nonempty_text(root_body["served_model_name"], label="served_model_name")
    if model_name != expected_model_name:
        raise ValueError("served-deployment attestation model differs from --model")

    collector = _closed_mapping(root_body["collector"], _COLLECTOR_KEYS, label="collector")
    if collector["implementation_version"] != SERVED_DEPLOYMENT_ATTESTATION_IMPLEMENTATION_VERSION:
        raise ValueError("served-deployment attestation collector version is unsupported")
    observed_module, observed_helper = _module_and_helper_hashes()
    if (
        collector["attestation_module_sha256"] != observed_module
        or collector["helper_script_sha256"] != observed_helper
    ):
        raise ValueError("served-deployment attestation collector bytes differ from local code")

    evidence = root_body["evidence"]
    if not isinstance(evidence, Mapping) or set(evidence) != set(EVIDENCE_FILENAMES):
        raise ValueError("served-deployment evidence reference set is incomplete")
    for role, expected_filename in EVIDENCE_FILENAMES.items():
        reference = _closed_mapping(evidence[role], _REFERENCE_KEYS, label=f"evidence.{role}")
        if (
            reference["relative_path"] != expected_filename
            or reference["schema_version"] != EVIDENCE_SCHEMAS[role]
        ):
            raise ValueError(f"served-deployment evidence reference {role!r} is not pinned")
        _sha256(reference["sha256"], label=f"evidence.{role}.sha256")
        _positive_size(reference["size_bytes"], label=f"evidence.{role}.size_bytes")

    relationships = _closed_mapping(
        root_body["relationships"], _RELATIONSHIP_KEYS, label="attestation relationships"
    )
    for key in (
        "model_files_sha256",
        "server_implementation_files_sha256",
        "packages_sha256",
        "launch_binding_sha256",
    ):
        _sha256(relationships[key], label=f"relationships.{key}")
    if _OCI_DIGEST.fullmatch(str(relationships["container_image_digest"])) is None:
        raise ValueError("relationships.container_image_digest is invalid")
    if _CONTAINER_ID.fullmatch(str(relationships["container_instance_id"])) is None:
        raise ValueError("relationships.container_instance_id is invalid")

    model = _validate_model_manifest(sidecars["model_manifest"], model_name=model_name)
    server = _validate_server_manifest(sidecars["server_implementation_manifest"])
    container = _validate_container_manifest(sidecars["container_image_manifest"])
    packages = _validate_package_manifest(sidecars["package_inventory_manifest"])
    launch = _validate_launch_binding(
        sidecars["launch_listener_binding"],
        endpoint=endpoint,
        model_name=model_name,
        relationships=relationships,
    )
    expected_relationships = {
        "model_files_sha256": model["files_sha256"],
        "server_implementation_files_sha256": server["files_sha256"],
        "container_image_digest": container["immutable_image_digest"],
        "container_instance_id": container["container_instance_id"],
        "packages_sha256": packages["packages_sha256"],
        "launch_binding_sha256": _launch_binding_sha256(launch),
    }
    if relationships != expected_relationships:
        raise ValueError("served-deployment cross-document relationships are invalid")
    if (
        launch["process"]["executable_path"] != server["server_executable"]["path"]
        or launch["process"]["executable_sha256"] != server["server_executable"]["sha256"]
    ):
        raise ValueError("launch process executable is not the attested server executable")
    if packages["python_executable"] != server["server_executable"]:
        raise ValueError("package inventory is not bound to the running server executable")
    if launch["model_launch"]["model_argument"] != model["model_root"]:
        raise ValueError("launch model argument is not the exact attested model root")
    if launch["deployment_instance_id"] != root_body["deployment_instance_id"]:
        raise ValueError("root deployment_instance_id differs from launch evidence")
    if launch["process"]["container_instance_id"] != container["container_instance_id"]:
        raise ValueError("launch process is not bound to the attested container instance")
    return {
        "root_body": root_body,
        "model": model,
        "server": server,
        "container": container,
        "packages": packages,
        "launch": launch,
    }


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _path_descriptor_flags() -> int:
    if not hasattr(os, "O_PATH"):
        raise RuntimeError("served-deployment collection requires Linux O_PATH support")
    return os.O_PATH | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_absolute_directory(path: Path, *, label: str) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be an absolute traversal-free path")
    descriptor = os.open(path.anchor, _directory_flags())
    try:
        for part in path.parts[1:]:
            next_descriptor = os.open(part, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_descriptor_file(directory_fd: int, filename: str, *, label: str) -> tuple[bytes, str]:
    if PurePosixPath(filename).name != filename:
        raise ValueError(f"{label} filename must be one canonical basename")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(filename, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a singly linked regular file")
        if before.st_size <= 0 or before.st_size > _MAX_ATTESTATION_FILE_BYTES:
            raise ValueError(f"{label} has an invalid byte size")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_key = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_key = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_key != after_key or size != before.st_size:
        raise RuntimeError(f"{label} changed while its bytes were authenticated")
    return b"".join(chunks), digest.hexdigest()


def _read_bundle(path: Path | str) -> tuple[Path, dict[str, bytes], dict[str, str]]:
    root = Path(path)
    if not root.is_absolute() or ".." in root.parts or root.name != ROOT_FILENAME:
        raise ValueError(
            f"served-deployment attestation path must be absolute and end in {ROOT_FILENAME!r}"
        )
    directory_fd = _open_absolute_directory(root.parent, label="served-deployment bundle directory")
    try:
        expected_entries = {ROOT_FILENAME, *EVIDENCE_FILENAMES.values()}
        observed_entries = set(os.listdir(directory_fd))
        if observed_entries != expected_entries:
            raise ValueError("served-deployment bundle directory is not closed")
        payloads: dict[str, bytes] = {}
        digests: dict[str, str] = {}
        for filename in sorted(expected_entries):
            payload, digest = _read_descriptor_file(
                directory_fd, filename, label=f"served-deployment bundle file {filename!r}"
            )
            payloads[filename] = payload
            digests[filename] = digest
        return Path(os.path.abspath(os.fspath(root))), payloads, digests
    finally:
        os.close(directory_fd)


@dataclass(frozen=True)
class AuthenticatedServedDeploymentIdentity:
    """Compact identity derived only from a pinned and fully validated bundle."""

    path: Path
    file_sha256: str
    served_model_name: str
    endpoint: str
    deployment_instance_id: str
    evidence_file_sha256: Mapping[str, str]
    model_files_sha256: str
    server_implementation_files_sha256: str
    container_image_digest: str
    packages_sha256: str
    launch_binding_sha256: str
    trust_anchor_source: str
    _snapshot_file_sha256: Mapping[str, str] = field(repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": AUTHENTICATED_SERVED_DEPLOYMENT_IDENTITY_SCHEMA_VERSION,
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "served_model_name": self.served_model_name,
            "endpoint": self.endpoint,
            "deployment_instance_id": self.deployment_instance_id,
            "evidence_file_sha256": dict(sorted(self.evidence_file_sha256.items())),
            "model_files_sha256": self.model_files_sha256,
            "server_implementation_files_sha256": self.server_implementation_files_sha256,
            "container_image_digest": self.container_image_digest,
            "packages_sha256": self.packages_sha256,
            "launch_binding_sha256": self.launch_binding_sha256,
            "trust_anchor": self.trust_anchor_source,
        }
        return {**body, "content_sha256": content_sha256(body)}

    def validate_current(self) -> None:
        current = load_authenticated_served_deployment_identity(
            self.path,
            expected_model_name=self.served_model_name,
            expected_endpoint=self.endpoint,
            trusted_attestation_sha256=self.file_sha256,
            trust_anchor_source=self.trust_anchor_source,
        )
        if current.as_dict() != self.as_dict() or dict(current._snapshot_file_sha256) != dict(
            self._snapshot_file_sha256
        ):
            raise RuntimeError("served-deployment attestation changed during execution")


def load_authenticated_served_deployment_identity(
    path: Path | str,
    *,
    expected_model_name: str,
    expected_endpoint: str,
    trusted_attestation_sha256: str,
    trust_anchor_source: str = "explicit_external_validation_pin",
) -> AuthenticatedServedDeploymentIdentity:
    """Load one bundle only when its raw root bytes match an external trust pin."""

    trusted = _sha256(
        trusted_attestation_sha256, label="trusted served-deployment attestation SHA-256"
    )
    trust_source = _nonempty_text(trust_anchor_source, label="trust_anchor_source", maximum=128)
    resolved, payloads, digests = _read_bundle(path)
    if digests[ROOT_FILENAME] != trusted:
        raise ValueError("served-deployment attestation is not the compiled trusted deployment")
    root = _strict_json_bytes(payloads[ROOT_FILENAME], label="served-deployment root")
    if not isinstance(root, Mapping):
        raise ValueError("served-deployment root must be an object")
    root_body = _validate_wrapped_document(
        root,
        expected_schema=SERVED_DEPLOYMENT_ATTESTATION_SCHEMA_VERSION,
        label="served-deployment attestation root",
    )
    references = _closed_mapping(root_body, _ROOT_BODY_KEYS, label="attestation root body")[
        "evidence"
    ]
    if not isinstance(references, Mapping) or set(references) != set(EVIDENCE_FILENAMES):
        raise ValueError("served-deployment evidence reference set is incomplete")
    sidecar_bodies: dict[str, dict[str, Any]] = {}
    evidence_hashes: dict[str, str] = {}
    for role, filename in EVIDENCE_FILENAMES.items():
        reference = _closed_mapping(references[role], _REFERENCE_KEYS, label=f"evidence.{role}")
        if reference.get("relative_path") != filename:
            raise ValueError(f"served-deployment evidence reference {role!r} changed path")
        if reference.get("sha256") != digests[filename] or reference.get("size_bytes") != len(
            payloads[filename]
        ):
            raise ValueError(f"served-deployment evidence bytes differ for {role!r}")
        document = _strict_json_bytes(payloads[filename], label=f"evidence {role!r}")
        sidecar_bodies[role] = _validate_wrapped_document(
            document,
            expected_schema=EVIDENCE_SCHEMAS[role],
            label=f"evidence {role!r}",
        )
        evidence_hashes[role] = digests[filename]
    validated = _validate_root_and_sidecars(
        root,
        sidecar_bodies,
        expected_model_name=_nonempty_text(expected_model_name, label="expected_model_name"),
        expected_endpoint=_validate_endpoint(expected_endpoint),
    )
    relationships = validated["root_body"]["relationships"]
    return AuthenticatedServedDeploymentIdentity(
        path=resolved,
        file_sha256=trusted,
        served_model_name=validated["root_body"]["served_model_name"],
        endpoint=validated["root_body"]["endpoint"],
        deployment_instance_id=validated["root_body"]["deployment_instance_id"],
        evidence_file_sha256=evidence_hashes,
        model_files_sha256=relationships["model_files_sha256"],
        server_implementation_files_sha256=relationships["server_implementation_files_sha256"],
        container_image_digest=relationships["container_image_digest"],
        packages_sha256=relationships["packages_sha256"],
        launch_binding_sha256=relationships["launch_binding_sha256"],
        trust_anchor_source=trust_source,
        _snapshot_file_sha256=dict(digests),
    )


def _hash_file_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_record(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError(f"attested path must be a regular non-symlink file: {path}")
    digest = _hash_file_path(path)
    after = path.stat(follow_symlinks=False)
    before_key = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_key != after_key:
        raise RuntimeError(f"attested file changed while hashing: {path}")
    key = "relative_path" if relative_path is not None else "path"
    value = relative_path if relative_path is not None else str(path.resolve(strict=True))
    return {key: value, "size_bytes": int(after.st_size), "sha256": digest}


def _read_strict_source(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    resolved = path.resolve(strict=True)
    record = _stable_file_record(resolved)
    payload = resolved.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise RuntimeError(f"{label} changed after authentication")
    value = _strict_json_bytes(payload, label=label)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return dict(value), record, payload


def _canonical_process_absolute_path(value: Path | str, *, label: str) -> str:
    raw = os.fspath(value)
    if not isinstance(raw, str):
        raise ValueError(f"{label} must be text")
    text = _nonempty_text(raw, label=label)
    if "\\" in text:
        raise ValueError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if (
        path.anchor != "/"
        or not path.is_absolute()
        or path == PurePosixPath("/")
        or any(part in ("", ".", "..") for part in path.parts[1:])
        or str(path) != text
    ):
        raise ValueError(f"{label} must be one canonical non-root absolute process path")
    return text


def _stat_key(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _inode_key(value: os.stat_result) -> tuple[int, int, int]:
    return (int(value.st_dev), int(value.st_ino), int(value.st_mode))


def _read_open_descriptor(
    descriptor: int,
    *,
    label: str,
    maximum_bytes: int | None = None,
) -> tuple[bytes, os.stat_result, os.stat_result]:
    before = os.fstat(descriptor)
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if maximum_bytes is not None and size > maximum_bytes:
            raise ValueError(f"{label} exceeds its fixed byte guard")
        chunks.append(chunk)
    after = os.fstat(descriptor)
    return b"".join(chunks), before, after


def _stable_descriptor_file_record(
    descriptor: int,
    *,
    path_key: str,
    path_value: str,
    label: str,
) -> dict[str, Any]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{label} must be a singly linked regular file")
    digest = hashlib.sha256()
    observed_size = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        observed_size += len(chunk)
    after = os.fstat(descriptor)
    if _stat_key(before) != _stat_key(after) or observed_size != before.st_size:
        raise RuntimeError(f"{label} changed while its exact inode was hashed")
    return {
        path_key: path_value,
        "size_bytes": int(before.st_size),
        "sha256": digest.hexdigest(),
    }


@dataclass
class _AnchoredTargetDirectory:
    target_path: str
    descriptors: list[int]
    links: list[tuple[int, str, int, tuple[int, ...]]]

    @property
    def directory_fd(self) -> int:
        return self.descriptors[-1]

    def validate(self, *, label: str) -> None:
        for parent_fd, name, child_fd, expected in self.links:
            if _stat_key(os.fstat(child_fd)) != expected:
                raise RuntimeError(f"{label} directory inode changed during traversal")
            try:
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise RuntimeError(f"{label} path was renamed during traversal") from exc
            if _stat_key(current) != expected:
                raise RuntimeError(f"{label} path was replaced during traversal")

    def close(self) -> None:
        while self.descriptors:
            os.close(self.descriptors.pop())


def _open_anchored_target_directory(
    process_root_fd: int,
    process_path: Path | str,
    *,
    label: str,
) -> _AnchoredTargetDirectory:
    target_path = _canonical_process_absolute_path(process_path, label=label)
    descriptors = [os.dup(process_root_fd)]
    links: list[tuple[int, str, int, tuple[int, ...]]] = []
    try:
        for name in PurePosixPath(target_path).parts[1:]:
            parent_fd = descriptors[-1]
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                raise ValueError(f"{label} cannot traverse a symlink or non-directory component")
            child_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
            opened = os.fstat(child_fd)
            if _stat_key(before) != _stat_key(opened):
                os.close(child_fd)
                raise RuntimeError(f"{label} component changed while it was opened")
            descriptors.append(child_fd)
            links.append((parent_fd, name, child_fd, _stat_key(opened)))
        anchor = _AnchoredTargetDirectory(
            target_path=target_path,
            descriptors=descriptors,
            links=links,
        )
        anchor.validate(label=label)
        return anchor
    except BaseException:
        while descriptors:
            os.close(descriptors.pop())
        raise


def _snapshot_directory_descriptor(
    directory_fd: int,
    *,
    label: str,
    relative_parent: PurePosixPath | None = None,
) -> list[dict[str, Any]]:
    root_before = os.fstat(directory_fd)
    if not stat.S_ISDIR(root_before.st_mode):
        raise ValueError(f"{label} must be a directory")
    before_names = sorted(os.listdir(directory_fd))
    rows: list[dict[str, Any]] = []
    parent = relative_parent or PurePosixPath()
    for name in before_names:
        try:
            name.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{label} contains a non-UTF-8 entry name") from exc
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"{label} contains a non-canonical entry name")
        relative = parent / name
        relative_text = _relative_path_text(str(relative), label=f"{label} relative path")
        entry_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(entry_before.st_mode):
            raise ValueError(f"{label} cannot contain symlinks: {relative_text}")
        if stat.S_ISDIR(entry_before.st_mode):
            child_fd = os.open(name, _directory_flags(), dir_fd=directory_fd)
            try:
                opened = os.fstat(child_fd)
                if _stat_key(entry_before) != _stat_key(opened):
                    raise RuntimeError(f"{label} directory changed while opened: {relative_text}")
                rows.extend(
                    _snapshot_directory_descriptor(
                        child_fd,
                        label=label,
                        relative_parent=relative,
                    )
                )
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if _stat_key(current) != _stat_key(opened):
                    raise RuntimeError(f"{label} directory changed while read: {relative_text}")
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(entry_before.st_mode):
            raise ValueError(f"{label} contains a non-regular entry: {relative_text}")
        if entry_before.st_nlink != 1:
            raise ValueError(f"{label} contains a multiply linked file: {relative_text}")
        path_fd = os.open(
            name,
            _path_descriptor_flags(),
            dir_fd=directory_fd,
        )
        file_fd: int | None = None
        try:
            opened = os.fstat(path_fd)
            if _stat_key(entry_before) != _stat_key(opened):
                raise RuntimeError(f"{label} file changed while opened: {relative_text}")
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise ValueError(f"{label} contains a non-regular or multiply linked file")
            file_fd = os.open(
                f"/proc/self/fd/{path_fd}",
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
            )
            if _stat_key(os.fstat(file_fd)) != _stat_key(opened):
                raise RuntimeError(f"{label} file inode changed while reopened: {relative_text}")
            rows.append(
                _stable_descriptor_file_record(
                    file_fd,
                    path_key="relative_path",
                    path_value=relative_text,
                    label=f"{label} file {relative_text!r}",
                )
            )
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if _stat_key(current) != _stat_key(opened):
                raise RuntimeError(f"{label} file changed while read: {relative_text}")
        finally:
            if file_fd is not None:
                os.close(file_fd)
            os.close(path_fd)
    after_names = sorted(os.listdir(directory_fd))
    root_after = os.fstat(directory_fd)
    if before_names != after_names or _stat_key(root_before) != _stat_key(root_after):
        raise RuntimeError(f"{label} directory entries changed while being authenticated")
    rows.sort(key=lambda row: str(row["relative_path"]))
    return rows


def _snapshot_target_tree(
    process_root_fd: int,
    process_path: Path | str,
    *,
    label: str,
) -> tuple[str, list[dict[str, Any]]]:
    target_path, rows, _root_identity = _snapshot_target_tree_with_identity(
        process_root_fd,
        process_path,
        label=label,
    )
    return target_path, rows


def _snapshot_target_tree_with_identity(
    process_root_fd: int,
    process_path: Path | str,
    *,
    label: str,
) -> tuple[str, list[dict[str, Any]], tuple[int, ...]]:
    anchor = _open_anchored_target_directory(process_root_fd, process_path, label=label)
    try:
        root_identity = _stat_key(os.fstat(anchor.directory_fd))
        rows = _snapshot_directory_descriptor(anchor.directory_fd, label=label)
        anchor.validate(label=label)
        if not rows:
            raise ValueError(f"{label} cannot be empty")
        return anchor.target_path, rows, root_identity
    finally:
        anchor.close()


def _validate_target_tree_root_identity(
    process_root_fd: int,
    process_path: Path | str,
    *,
    expected_identity: tuple[int, ...],
    label: str,
) -> None:
    anchor = _open_anchored_target_directory(process_root_fd, process_path, label=label)
    try:
        if _stat_key(os.fstat(anchor.directory_fd)) != expected_identity:
            raise RuntimeError(f"{label} root changed after its file snapshot")
        anchor.validate(label=label)
    finally:
        anchor.close()


def _read_proc_file(process_fd: int, filename: str, *, label: str) -> bytes:
    if PurePosixPath(filename).name != filename:
        raise ValueError(f"{label} must be one proc basename")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(filename, flags, dir_fd=process_fd)
    try:
        payload, before, after = _read_open_descriptor(
            descriptor,
            label=label,
            maximum_bytes=_MAX_PROC_RECORD_BYTES,
        )
        if not stat.S_ISREG(before.st_mode) or _inode_key(before) != _inode_key(after):
            raise RuntimeError(f"{label} proc inode changed while read")
        return payload
    finally:
        os.close(descriptor)


def _open_proc_namespace_fd(process_fd: int, name: str) -> int:
    namespace_directory = os.open("ns", _directory_flags(), dir_fd=process_fd)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            dir_fd=namespace_directory,
        )
    finally:
        os.close(namespace_directory)
    value = os.fstat(descriptor)
    if value.st_ino <= 0:
        os.close(descriptor)
        raise RuntimeError(f"target {name} namespace has no stable inode")
    return descriptor


def _hostname_from_uts_namespace_fd(namespace_fd: int) -> str:
    current_fd = os.open(
        "/proc/self/ns/uts",
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        if _inode_key(os.fstat(current_fd)) == _inode_key(os.fstat(namespace_fd)):
            return _nonempty_text(
                socket.gethostname().lower(), label="target UTS hostname", maximum=253
            )
    finally:
        os.close(current_fd)
    if not hasattr(os, "setns"):
        raise RuntimeError("target UTS authentication requires Linux setns support")
    thread_namespace_fd = os.dup(namespace_fd)
    result: dict[str, Any] = {}

    def _read_in_target_namespace() -> None:
        try:
            os.setns(thread_namespace_fd, getattr(os, "CLONE_NEWUTS", 0x04000000))
            result["hostname"] = socket.gethostname().lower()
        except BaseException as exc:
            result["error"] = exc
        finally:
            os.close(thread_namespace_fd)

    worker = threading.Thread(
        target=_read_in_target_namespace,
        name="served-attestation-target-uts",
        daemon=False,
    )
    try:
        worker.start()
    except BaseException:
        os.close(thread_namespace_fd)
        raise
    worker.join()
    if "error" in result:
        error = result["error"]
        raise RuntimeError(
            f"could not enter target UTS namespace: {type(error).__name__}: {error}"
        ) from error
    return _nonempty_text(
        result.get("hostname"),
        label="target UTS hostname",
        maximum=253,
    )


def _target_uts_hostname(process_fd: int) -> tuple[str, tuple[int, int, int]]:
    namespace_fd = _open_proc_namespace_fd(process_fd, "uts")
    try:
        identity = _inode_key(os.fstat(namespace_fd))
        hostname = _hostname_from_uts_namespace_fd(namespace_fd)
        if _inode_key(os.fstat(namespace_fd)) != identity:
            raise RuntimeError("target UTS namespace changed while hostname was read")
        return hostname, identity
    finally:
        os.close(namespace_fd)


def _snapshot_process_executable(
    process_fd: int,
    *,
    expected_path: str,
    expected_inode: tuple[int, ...],
) -> dict[str, Any]:
    before_path = os.readlink("exe", dir_fd=process_fd)
    if before_path.endswith(" (deleted)"):
        raise ValueError("server executable was deleted after launch")
    canonical_path = _canonical_process_absolute_path(before_path, label="server executable path")
    if canonical_path != expected_path:
        raise RuntimeError("server executable path changed before hashing")
    descriptor = os.open(
        "exe",
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        dir_fd=process_fd,
    )
    try:
        if _stat_key(os.fstat(descriptor)) != expected_inode:
            raise RuntimeError("server executable inode changed before hashing")
        record = _stable_descriptor_file_record(
            descriptor,
            path_key="path",
            path_value=canonical_path,
            label="server executable",
        )
    finally:
        os.close(descriptor)
    after_path = os.readlink("exe", dir_fd=process_fd)
    current = os.stat("exe", dir_fd=process_fd, follow_symlinks=True)
    if after_path != before_path or _stat_key(current) != expected_inode:
        raise RuntimeError("server executable changed while it was authenticated")
    return record


def _validate_executable_under_process_root(
    process_root_fd: int,
    executable_path: str,
    *,
    expected_inode: tuple[int, ...],
) -> None:
    path = PurePosixPath(
        _canonical_process_absolute_path(executable_path, label="server executable path")
    )
    parent_path = str(path.parent)
    if parent_path == "/":
        parent_fd = os.dup(process_root_fd)
        anchor: _AnchoredTargetDirectory | None = None
    else:
        anchor = _open_anchored_target_directory(
            process_root_fd,
            parent_path,
            label="server executable parent",
        )
        parent_fd = anchor.directory_fd
    path_descriptor: int | None = None
    read_descriptor: int | None = None
    try:
        path_descriptor = os.open(
            path.name,
            _path_descriptor_flags(),
            dir_fd=parent_fd,
        )
        anchored = os.fstat(path_descriptor)
        if not stat.S_ISREG(anchored.st_mode) or _stat_key(anchored) != expected_inode:
            raise RuntimeError("process-root executable path does not name the running inode")
        read_descriptor = os.open(
            f"/proc/self/fd/{path_descriptor}",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        if _stat_key(os.fstat(read_descriptor)) != expected_inode:
            raise RuntimeError("process-root executable inode changed while reopened")
        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if _stat_key(current) != expected_inode:
            raise RuntimeError("process-root executable path changed during validation")
        if anchor is not None:
            anchor.validate(label="server executable parent")
    finally:
        if read_descriptor is not None:
            os.close(read_descriptor)
        if path_descriptor is not None:
            os.close(path_descriptor)
        if anchor is None:
            os.close(parent_fd)
        else:
            anchor.close()


def _parse_cmdline(raw: bytes, *, model_name: str, model_root: str) -> dict[str, Any]:
    try:
        arguments = [part.decode("utf-8") for part in raw.split(b"\0") if part]
    except UnicodeDecodeError as exc:
        raise ValueError("server process command line is not UTF-8") from exc
    if not arguments:
        raise ValueError("server process command line is empty")
    served_names: list[str] = []
    served_flag_count = 0
    model_arguments: list[str] = []
    model_flag_count = 0
    for index, argument in enumerate(arguments):
        if argument.startswith("--served-model-name=") or argument.startswith("--model="):
            raise ValueError("server launch must use unambiguous space-separated model flags")
        if argument == "--served-model-name" and index + 1 < len(arguments):
            served_flag_count += 1
            cursor = index + 1
            while cursor < len(arguments) and not arguments[cursor].startswith("--"):
                served_names.append(arguments[cursor])
                cursor += 1
        if argument == "--model" and index + 1 < len(arguments):
            model_flag_count += 1
            model_arguments.append(arguments[index + 1])
        if (
            argument == "serve"
            and index == 1
            and PurePosixPath(arguments[0]).name == "vllm"
            and index + 1 < len(arguments)
        ):
            model_arguments.append(arguments[index + 1])
    legacy_vllm_entrypoint = any(
        argument == "-m"
        and index + 1 < len(arguments)
        and arguments[index + 1] == "vllm.entrypoints.openai.api_server"
        for index, argument in enumerate(arguments)
    )
    if model_flag_count and not legacy_vllm_entrypoint:
        raise ValueError("--model is accepted only for the exact legacy vLLM entrypoint")
    if served_flag_count != 1 or served_names != [model_name]:
        raise ValueError("server launch must explicitly contain exactly one expected served name")
    if len(model_arguments) != 1:
        raise ValueError("server launch must identify exactly one model argument")
    model_argument = _canonical_process_absolute_path(
        model_arguments[0], label="server launch model argument"
    )
    if model_argument != model_root:
        raise ValueError("server launch model argument is not the exact attested process path")
    return {
        "model_argument": model_argument,
        "served_model_names": served_names,
        "served_name_flag_present": True,
    }


def _process_start_ticks(stat_payload: str) -> int:
    close = stat_payload.rfind(")")
    if close < 0:
        raise ValueError("process stat record is malformed")
    fields_from_three = stat_payload[close + 2 :].split()
    if len(fields_from_three) <= 19:
        raise ValueError("process stat record is incomplete")
    value = int(fields_from_three[19])
    if value <= 0:
        raise ValueError("process start time is invalid")
    return value


def _read_boot_id() -> str:
    directory_fd = _open_absolute_directory(
        Path("/proc/sys/kernel/random"), label="host kernel random proc directory"
    )
    try:
        try:
            value = (
                _read_proc_file(directory_fd, "boot_id", label="host boot_id")
                .decode("ascii", errors="strict")
                .strip()
                .lower()
            )
        except UnicodeDecodeError as exc:
            raise ValueError("host boot_id is not ASCII") from exc
    finally:
        os.close(directory_fd)
    if _BOOT_ID.fullmatch(value) is None:
        raise ValueError("host boot_id is invalid")
    return value


def _open_live_pidfd(pid: int) -> int:
    if not hasattr(os, "pidfd_open"):
        raise RuntimeError("served-deployment collection requires Linux pidfd_open support")
    try:
        descriptor = os.pidfd_open(pid, 0)
    except ProcessLookupError as exc:
        raise ValueError("server_pid does not identify a running process") from exc
    try:
        _assert_pidfd_live(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _assert_pidfd_live(pidfd: int) -> None:
    poller = select.poll()
    poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
    if poller.poll(0):
        raise RuntimeError("server process exited during attestation")


def _process_epoch(process_fd: int) -> dict[str, Any]:
    try:
        stat_payload = _read_proc_file(process_fd, "stat", label="server process stat").decode(
            "utf-8", errors="strict"
        )
    except UnicodeDecodeError as exc:
        raise ValueError("server process stat is not UTF-8") from exc
    cmdline = _read_proc_file(process_fd, "cmdline", label="server process cmdline")
    cgroup = _read_proc_file(process_fd, "cgroup", label="server process cgroup")
    executable_path = os.readlink("exe", dir_fd=process_fd)
    if executable_path.endswith(" (deleted)"):
        raise ValueError("server executable was deleted after launch")
    executable_path = _canonical_process_absolute_path(
        executable_path, label="server executable path"
    )
    executable_stat = os.stat("exe", dir_fd=process_fd, follow_symlinks=True)
    if not stat.S_ISREG(executable_stat.st_mode) or executable_stat.st_nlink != 1:
        raise ValueError("server executable must be a singly linked regular file")
    process_root_stat = os.stat("root", dir_fd=process_fd, follow_symlinks=True)
    if not stat.S_ISDIR(process_root_stat.st_mode):
        raise ValueError("server process root is not a directory")
    mount_fd = _open_proc_namespace_fd(process_fd, "mnt")
    net_fd = _open_proc_namespace_fd(process_fd, "net")
    uts_fd = _open_proc_namespace_fd(process_fd, "uts")
    try:
        mount_namespace = _inode_key(os.fstat(mount_fd))
        net_namespace = _inode_key(os.fstat(net_fd))
        uts_namespace = _inode_key(os.fstat(uts_fd))
    finally:
        os.close(uts_fd)
        os.close(net_fd)
        os.close(mount_fd)
    return {
        "boot_id": _read_boot_id(),
        "start_time_ticks": _process_start_ticks(stat_payload),
        "cmdline": cmdline,
        "cgroup": cgroup,
        "executable_path": executable_path,
        "executable_stat": _stat_key(executable_stat),
        "process_root_stat": _stat_key(process_root_stat),
        "mount_namespace": mount_namespace,
        "net_namespace": net_namespace,
        "uts_namespace": uts_namespace,
    }


def _assert_process_epoch_unchanged(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    if dict(before) != dict(after):
        changed = sorted(
            key for key in set(before) | set(after) if before.get(key) != after.get(key)
        )
        raise RuntimeError("server process epoch changed during attestation: " + ", ".join(changed))


def _open_process_root(
    process_fd: int,
    *,
    expected_stat: tuple[int, ...],
) -> int:
    descriptor = os.open(
        "root",
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=process_fd,
    )
    current = os.fstat(descriptor)
    if not stat.S_ISDIR(current.st_mode) or _stat_key(current) != expected_stat:
        os.close(descriptor)
        raise RuntimeError("server process root changed before target traversal")
    return descriptor


def _validate_process_root(
    process_fd: int,
    process_root_fd: int,
    *,
    expected_stat: tuple[int, ...],
) -> None:
    held = os.fstat(process_root_fd)
    current = os.stat("root", dir_fd=process_fd, follow_symlinks=True)
    if _stat_key(held) != expected_stat or _stat_key(current) != expected_stat:
        raise RuntimeError("server process root changed during target traversal")


def _decode_proc_address(value: str, *, ipv6: bool) -> str:
    raw = bytes.fromhex(value)
    if not ipv6:
        return socket.inet_ntop(socket.AF_INET, raw[::-1])
    reordered = b"".join(raw[index : index + 4][::-1] for index in range(0, 16, 4))
    return socket.inet_ntop(socket.AF_INET6, reordered)


def _owned_listener_records(process_fd: int, *, port: int) -> list[dict[str, Any]]:
    fd_root = os.open("fd", _directory_flags(), dir_fd=process_fd)
    owned: set[int] = set()
    try:
        for entry in os.listdir(fd_root):
            try:
                target = os.readlink(entry, dir_fd=fd_root)
            except FileNotFoundError:
                continue
            match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
            if match:
                owned.add(int(match.group(1)))
    finally:
        os.close(fd_root)
    records: list[dict[str, Any]] = []
    net_root = os.open("net", _directory_flags(), dir_fd=process_fd)
    try:
        for filename, ipv6 in (("tcp", False), ("tcp6", True)):
            table = _read_proc_file(net_root, filename, label=f"target net/{filename}").decode(
                "ascii", errors="strict"
            )
            for line in table.splitlines()[1:]:
                fields = line.split()
                if len(fields) < 10:
                    raise ValueError(f"target net/{filename} contains a malformed socket row")
                local_hex, state, inode_text = fields[1], fields[3], fields[9]
                address_hex, port_hex = local_hex.split(":", 1)
                observed_port = int(port_hex, 16)
                inode = int(inode_text)
                if observed_port == port and state == "0A" and inode in owned:
                    records.append(
                        {
                            "address": _decode_proc_address(address_hex, ipv6=ipv6),
                            "port": port,
                            "state": "LISTEN",
                            "socket_inode": inode,
                            "owned_by_process": True,
                        }
                    )
    finally:
        os.close(net_root)
    records.sort(key=lambda row: (str(row["address"]), int(row["port"]), int(row["socket_inode"])))
    if not records:
        raise ValueError("server process owns no LISTEN socket on the endpoint port")
    return records


def _container_instance_id_from_cgroup(payload: bytes) -> str:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("server process cgroup record is not UTF-8") from exc
    candidates = sorted(set(re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", text)))
    if len(candidates) != 1:
        raise ValueError(
            "server process cgroup must identify exactly one lowercase 64-hex container instance"
        )
    return candidates[0]


def _wrapped(schema: str, body: Mapping[str, Any]) -> dict[str, Any]:
    copied = json.loads(canonical_json(body))
    return {"schema_version": schema, "body": copied, "content_sha256": content_sha256(copied)}


def seal_served_deployment_attestation_bundle(
    *,
    output_dir: Path | str,
    endpoint: str,
    served_model_name: str,
    sidecar_bodies: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Validate and atomically publish a closed attestation bundle."""

    output = Path(output_dir)
    if not output.is_absolute() or ".." in output.parts or output.exists():
        raise ValueError("output_dir must be an absolute fresh nonexistent path")
    if set(sidecar_bodies) != set(EVIDENCE_FILENAMES):
        raise ValueError("sidecar body set is incomplete")
    parent = output.parent.resolve(strict=True)
    if parent.is_symlink():
        raise ValueError("output_dir parent cannot be a symlink")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        references: dict[str, dict[str, Any]] = {}
        documents: dict[str, dict[str, Any]] = {}
        for role, filename in EVIDENCE_FILENAMES.items():
            document = _wrapped(EVIDENCE_SCHEMAS[role], sidecar_bodies[role])
            payload = (
                json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            path = temporary / filename
            path.write_bytes(payload)
            references[role] = {
                "relative_path": filename,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "schema_version": EVIDENCE_SCHEMAS[role],
            }
            documents[role] = document
        module_sha, helper_sha = _module_and_helper_hashes()
        relationships = dict(sidecar_bodies["launch_listener_binding"]["relationships"])
        root_body = {
            "attestation_scope": "one_exact_running_openai_compatible_deployment",
            "endpoint": endpoint,
            "served_model_name": served_model_name,
            "deployment_instance_id": sidecar_bodies["launch_listener_binding"][
                "deployment_instance_id"
            ],
            "collector": {
                "implementation_version": SERVED_DEPLOYMENT_ATTESTATION_IMPLEMENTATION_VERSION,
                "attestation_module_sha256": module_sha,
                "helper_script_sha256": helper_sha,
            },
            "evidence": references,
            "relationships": relationships,
        }
        root = _wrapped(SERVED_DEPLOYMENT_ATTESTATION_SCHEMA_VERSION, root_body)
        _validate_root_and_sidecars(
            root,
            {role: dict(document["body"]) for role, document in documents.items()},
            expected_model_name=served_model_name,
            expected_endpoint=endpoint,
        )
        root_path = temporary / ROOT_FILENAME
        root_path.write_text(
            json.dumps(root, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
        return output / ROOT_FILENAME
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def collect_served_deployment_attestation(
    *,
    output_dir: Path | str,
    endpoint: str,
    served_model_name: str,
    model_root: Path | str,
    server_pid: int,
    server_implementation_root: Path | str,
    container_runtime: str,
    image_reference: str,
    oci_manifest_json: Path | str,
    container_runtime_inspect_json: Path | str,
    package_inventory_json: Path | str,
) -> Path:
    """Read one live Linux deployment and publish its deterministic evidence bundle."""

    endpoint = _validate_endpoint(endpoint)
    model_name = _nonempty_text(served_model_name, label="served_model_name")
    parsed = urlsplit(endpoint)
    pid = _positive_size(server_pid, label="server_pid")
    model_path = _canonical_process_absolute_path(model_root, label="model_root")
    server_root = _canonical_process_absolute_path(
        server_implementation_root, label="server_implementation_root"
    )
    pidfd = _open_live_pidfd(pid)
    procfs_fd: int | None = None
    process_fd: int | None = None
    process_root_fd: int | None = None
    mount_namespace_fd: int | None = None
    network_namespace_fd: int | None = None
    uts_namespace_fd: int | None = None
    try:
        procfs_fd = _open_absolute_directory(Path("/proc"), label="host proc filesystem")
        process_fd = os.open(str(pid), _directory_flags(), dir_fd=procfs_fd)
        _assert_pidfd_live(pidfd)
        epoch_before = _process_epoch(process_fd)

        mount_namespace_fd = _open_proc_namespace_fd(process_fd, "mnt")
        network_namespace_fd = _open_proc_namespace_fd(process_fd, "net")
        uts_namespace_fd = _open_proc_namespace_fd(process_fd, "uts")
        held_mount_namespace = _inode_key(os.fstat(mount_namespace_fd))
        held_network_namespace = _inode_key(os.fstat(network_namespace_fd))
        held_uts_namespace = _inode_key(os.fstat(uts_namespace_fd))
        if (
            held_mount_namespace != epoch_before["mount_namespace"]
            or held_network_namespace != epoch_before["net_namespace"]
            or held_uts_namespace != epoch_before["uts_namespace"]
        ):
            raise RuntimeError("server namespaces changed before collection")

        process_root_fd = _open_process_root(
            process_fd,
            expected_stat=epoch_before["process_root_stat"],
        )
        executable = _snapshot_process_executable(
            process_fd,
            expected_path=epoch_before["executable_path"],
            expected_inode=epoch_before["executable_stat"],
        )
        _validate_executable_under_process_root(
            process_root_fd,
            executable["path"],
            expected_inode=epoch_before["executable_stat"],
        )
        hostname_before = _hostname_from_uts_namespace_fd(uts_namespace_fd)
        if hostname_before != parsed.hostname and not hostname_before.startswith(
            f"{parsed.hostname}."
        ):
            raise ValueError("target UTS hostname does not match the production endpoint")
        model_launch = _parse_cmdline(
            epoch_before["cmdline"],
            model_name=model_name,
            model_root=model_path,
        )
        listeners_before = _owned_listener_records(
            process_fd,
            port=int(parsed.port or 0),
        )
        model_path, model_files, model_root_identity = _snapshot_target_tree_with_identity(
            process_root_fd,
            model_path,
            label="model root",
        )
        server_root, server_files, server_root_identity = _snapshot_target_tree_with_identity(
            process_root_fd,
            server_root,
            label="server implementation root",
        )
        _validate_process_root(
            process_fd,
            process_root_fd,
            expected_stat=epoch_before["process_root_stat"],
        )
        oci_value, oci_source, oci_payload = _read_strict_source(
            Path(oci_manifest_json), label="OCI image manifest"
        )
        if not oci_value:
            raise ValueError("OCI image manifest cannot be empty")
        inspect_value, inspect_source, _inspect_payload = _read_strict_source(
            Path(container_runtime_inspect_json), label="container runtime inspection"
        )
        expected_image_digest = f"sha256:{hashlib.sha256(oci_payload).hexdigest()}"
        expected_inspect = {
            "container_instance_id": _container_instance_id_from_cgroup(epoch_before["cgroup"]),
            "container_runtime": container_runtime,
            "image_reference": image_reference,
            "immutable_image_digest": expected_image_digest,
        }
        if inspect_value != expected_inspect:
            raise ValueError(
                "container runtime inspection does not bind the process cgroup to the exact image"
            )
        container_instance_id = str(inspect_value["container_instance_id"])
        package_value, package_source, _package_payload = _read_strict_source(
            Path(package_inventory_json), label="package inventory"
        )
        if set(package_value) != {"packages"}:
            raise ValueError("package inventory source must be a closed {'packages'} object")
        packages = package_value["packages"]

        hostname_after = _hostname_from_uts_namespace_fd(uts_namespace_fd)
        listeners_after = _owned_listener_records(process_fd, port=int(parsed.port or 0))
        epoch_after = _process_epoch(process_fd)
        _assert_process_epoch_unchanged(epoch_before, epoch_after)
        _assert_pidfd_live(pidfd)
        _validate_process_root(
            process_fd,
            process_root_fd,
            expected_stat=epoch_before["process_root_stat"],
        )
        _validate_target_tree_root_identity(
            process_root_fd,
            model_path,
            expected_identity=model_root_identity,
            label="model root",
        )
        _validate_target_tree_root_identity(
            process_root_fd,
            server_root,
            expected_identity=server_root_identity,
            label="server implementation root",
        )
        if hostname_after != hostname_before:
            raise RuntimeError("target UTS hostname changed during attestation")
        if (
            _inode_key(os.fstat(mount_namespace_fd)) != held_mount_namespace
            or _inode_key(os.fstat(network_namespace_fd)) != held_network_namespace
            or _inode_key(os.fstat(uts_namespace_fd)) != held_uts_namespace
        ):
            raise RuntimeError("held server namespace changed during attestation")
        if listeners_after != listeners_before:
            raise RuntimeError("server listener changed during attestation")

        relationships = {
            "model_files_sha256": content_sha256(model_files),
            "server_implementation_files_sha256": content_sha256(server_files),
            "container_image_digest": expected_image_digest,
            "container_instance_id": container_instance_id,
            "packages_sha256": content_sha256(packages),
        }
        executable_stat = epoch_before["executable_stat"]
        process_root_stat = epoch_before["process_root_stat"]
        launch = {
            "endpoint": endpoint,
            "served_model_name": model_name,
            "hostname": hostname_before,
            "boot_id": epoch_before["boot_id"],
            "deployment_instance_id": "0" * 64,
            "process": {
                "pid": pid,
                "start_time_ticks": epoch_before["start_time_ticks"],
                "executable_path": executable["path"],
                "executable_sha256": executable["sha256"],
                "cmdline_sha256": hashlib.sha256(epoch_before["cmdline"]).hexdigest(),
                "cgroup_sha256": hashlib.sha256(epoch_before["cgroup"]).hexdigest(),
                "executable_device": executable_stat[0],
                "executable_inode": executable_stat[1],
                "process_root_device": process_root_stat[0],
                "process_root_inode": process_root_stat[1],
                "mount_namespace_inode": held_mount_namespace[1],
                "network_namespace_inode": held_network_namespace[1],
                "uts_namespace_inode": held_uts_namespace[1],
                "container_instance_id": container_instance_id,
            },
            "model_launch": model_launch,
            "listener": {
                "transport": "tcp",
                "port": int(parsed.port or 0),
                "records": listeners_before,
                "records_sha256": content_sha256(listeners_before),
            },
            "relationships": relationships,
        }
        launch["deployment_instance_id"] = _deployment_instance_id(launch)
        launch_binding_sha = _launch_binding_sha256(launch)
        relationships["launch_binding_sha256"] = launch_binding_sha
        launch["relationships"] = dict(relationships)

        sidecars = {
            "model_manifest": {
                "served_model_name": model_name,
                "model_root": model_path,
                "files": model_files,
                "file_count": len(model_files),
                "total_file_bytes": sum(int(row["size_bytes"]) for row in model_files),
                "files_sha256": content_sha256(model_files),
            },
            "server_implementation_manifest": {
                "server_runtime": "vllm_openai_compatible",
                "implementation_root": server_root,
                "server_executable": executable,
                "files": server_files,
                "file_count": len(server_files),
                "total_file_bytes": sum(int(row["size_bytes"]) for row in server_files),
                "files_sha256": content_sha256(server_files),
            },
            "container_image_manifest": {
                "container_runtime": container_runtime,
                "image_reference": image_reference,
                "immutable_image_digest": expected_image_digest,
                "container_instance_id": container_instance_id,
                "oci_manifest_source": oci_source,
                "runtime_inspect_source": inspect_source,
            },
            "package_inventory_manifest": {
                "python_executable": executable,
                "inventory_source": package_source,
                "packages": packages,
                "package_count": len(packages),
                "packages_sha256": content_sha256(packages),
            },
            "launch_listener_binding": launch,
        }
        return seal_served_deployment_attestation_bundle(
            output_dir=output_dir,
            endpoint=endpoint,
            served_model_name=model_name,
            sidecar_bodies=sidecars,
        )
    finally:
        for descriptor in (
            uts_namespace_fd,
            network_namespace_fd,
            mount_namespace_fd,
            process_root_fd,
            process_fd,
            procfs_fd,
            pidfd,
        ):
            if descriptor is not None:
                os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read and attest one running Camus OpenAI-compatible model deployment"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument(
        "--model-root",
        required=True,
        type=Path,
        help="canonical absolute path inside the server process; must equal its live model argv",
    )
    parser.add_argument(
        "--server-pid",
        required=True,
        type=int,
        help="host-PID-namespace PID of the exact running inference process",
    )
    parser.add_argument(
        "--server-implementation-root",
        required=True,
        type=Path,
        help="canonical absolute vLLM code-tree path inside the server process",
    )
    parser.add_argument(
        "--container-runtime",
        required=True,
        choices=("docker", "containerd", "podman", "kubernetes"),
    )
    parser.add_argument("--image-reference", required=True)
    parser.add_argument("--oci-manifest-json", required=True, type=Path)
    parser.add_argument("--container-runtime-inspect-json", required=True, type=Path)
    parser.add_argument("--package-inventory-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = collect_served_deployment_attestation(
        output_dir=args.output_dir,
        endpoint=args.endpoint,
        served_model_name=args.served_model_name,
        model_root=args.model_root,
        server_pid=args.server_pid,
        server_implementation_root=args.server_implementation_root,
        container_runtime=args.container_runtime,
        image_reference=args.image_reference,
        oci_manifest_json=args.oci_manifest_json,
        container_runtime_inspect_json=args.container_runtime_inspect_json,
        package_inventory_json=args.package_inventory_json,
    )
    digest = _hash_file_path(path)
    print(canonical_json({"attestation_path": str(path), "attestation_sha256": digest}))
    return 0


__all__ = [
    "AUTHENTICATED_SERVED_DEPLOYMENT_IDENTITY_SCHEMA_VERSION",
    "AuthenticatedServedDeploymentIdentity",
    "CONTAINER_IMAGE_MANIFEST_SCHEMA_VERSION",
    "EVIDENCE_FILENAMES",
    "EVIDENCE_SCHEMAS",
    "LAUNCH_LISTENER_BINDING_SCHEMA_VERSION",
    "PACKAGE_INVENTORY_MANIFEST_SCHEMA_VERSION",
    "ROOT_FILENAME",
    "SERVED_DEPLOYMENT_ATTESTATION_IMPLEMENTATION_VERSION",
    "SERVED_DEPLOYMENT_ATTESTATION_SCHEMA_VERSION",
    "SERVED_MODEL_FILE_MANIFEST_SCHEMA_VERSION",
    "SERVER_IMPLEMENTATION_MANIFEST_SCHEMA_VERSION",
    "build_parser",
    "canonical_json",
    "collect_served_deployment_attestation",
    "content_sha256",
    "load_authenticated_served_deployment_identity",
    "main",
    "seal_served_deployment_attestation_bundle",
]
