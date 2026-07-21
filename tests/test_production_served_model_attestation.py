from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oci.inference import production_served_model_attestation as subject

MODEL = "publisher/exact-served-model"
ENDPOINT = "http://camus:8010/v1"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture_bodies(*, pid: int = 4242) -> dict[str, dict[str, object]]:
    model_files = [
        {"relative_path": "config.json", "size_bytes": 11, "sha256": "1" * 64},
        {
            "relative_path": "model-00001-of-00001.safetensors",
            "size_bytes": 1024,
            "sha256": "2" * 64,
        },
    ]
    server_files = [
        {
            "relative_path": "vllm/entrypoints/openai/api_server.py",
            "size_bytes": 20,
            "sha256": "3" * 64,
        }
    ]
    packages = [
        {"name": "torch", "version": "2.9.0"},
        {"name": "transformers", "version": "5.4.0"},
        {"name": "vllm", "version": "0.20.1"},
    ]
    executable = {"path": "/usr/bin/python3", "size_bytes": 100, "sha256": "4" * 64}
    oci_sha = "5" * 64
    relationships: dict[str, object] = {
        "model_files_sha256": subject.content_sha256(model_files),
        "server_implementation_files_sha256": subject.content_sha256(server_files),
        "container_image_digest": f"sha256:{oci_sha}",
        "container_instance_id": "9" * 64,
        "packages_sha256": subject.content_sha256(packages),
    }
    listener_records = [
        {
            "address": "0.0.0.0",
            "port": 8010,
            "state": "LISTEN",
            "socket_inode": 9001,
            "owned_by_process": True,
        }
    ]
    launch: dict[str, object] = {
        "endpoint": ENDPOINT,
        "served_model_name": MODEL,
        "hostname": "camus",
        "boot_id": "01234567-89ab-cdef-0123-456789abcdef",
        "deployment_instance_id": "0" * 64,
        "process": {
            "pid": pid,
            "start_time_ticks": 123456,
            "executable_path": executable["path"],
            "executable_sha256": executable["sha256"],
            "cmdline_sha256": "6" * 64,
            "cgroup_sha256": "7" * 64,
            "executable_device": 31,
            "executable_inode": 32,
            "process_root_device": 41,
            "process_root_inode": 42,
            "mount_namespace_inode": 7070,
            "network_namespace_inode": 8080,
            "uts_namespace_inode": 9090,
            "container_instance_id": "9" * 64,
        },
        "model_launch": {
            "model_argument": "/srv/models/exact-model",
            "served_model_names": [MODEL],
            "served_name_flag_present": True,
        },
        "listener": {
            "transport": "tcp",
            "port": 8010,
            "records": listener_records,
            "records_sha256": subject.content_sha256(listener_records),
        },
        "relationships": dict(relationships),
    }
    launch["deployment_instance_id"] = subject._deployment_instance_id(launch)
    relationships["launch_binding_sha256"] = subject._launch_binding_sha256(launch)
    launch["relationships"] = dict(relationships)
    return {
        "model_manifest": {
            "served_model_name": MODEL,
            "model_root": "/srv/models/exact-model",
            "files": model_files,
            "file_count": len(model_files),
            "total_file_bytes": 1035,
            "files_sha256": subject.content_sha256(model_files),
        },
        "server_implementation_manifest": {
            "server_runtime": "vllm_openai_compatible",
            "implementation_root": "/opt/vllm",
            "server_executable": executable,
            "files": server_files,
            "file_count": 1,
            "total_file_bytes": 20,
            "files_sha256": subject.content_sha256(server_files),
        },
        "container_image_manifest": {
            "container_runtime": "containerd",
            "image_reference": "registry.example/inference@sha256:" + oci_sha,
            "immutable_image_digest": "sha256:" + oci_sha,
            "container_instance_id": "9" * 64,
            "oci_manifest_source": {
                "path": "/run/attestation/oci-manifest.json",
                "size_bytes": 200,
                "sha256": oci_sha,
            },
            "runtime_inspect_source": {
                "path": "/run/attestation/container-inspect.json",
                "size_bytes": 201,
                "sha256": "a" * 64,
            },
        },
        "package_inventory_manifest": {
            "python_executable": executable,
            "inventory_source": {
                "path": "/run/attestation/packages.json",
                "size_bytes": 300,
                "sha256": "8" * 64,
            },
            "packages": packages,
            "package_count": len(packages),
            "packages_sha256": subject.content_sha256(packages),
        },
        "launch_listener_binding": launch,
    }


def _bundle(tmp_path: Path, *, name: str = "deployment", pid: int = 4242) -> tuple[Path, str]:
    root = subject.seal_served_deployment_attestation_bundle(
        output_dir=tmp_path / name,
        endpoint=ENDPOINT,
        served_model_name=MODEL,
        sidecar_bodies=_fixture_bodies(pid=pid),
    )
    return root, _sha(root.read_bytes())


def _load(root: Path, digest: str):
    return subject.load_authenticated_served_deployment_identity(
        root,
        expected_model_name=MODEL,
        expected_endpoint=ENDPOINT,
        trusted_attestation_sha256=digest,
        trust_anchor_source="test_fixture_external_pin",
    )


def test_valid_bundle_authenticates_every_closed_sidecar_and_relationship(tmp_path: Path) -> None:
    root, digest = _bundle(tmp_path)
    identity = _load(root, digest)
    observed = identity.as_dict()
    assert observed["schema_version"] == (
        subject.AUTHENTICATED_SERVED_DEPLOYMENT_IDENTITY_SCHEMA_VERSION
    )
    assert observed["file_sha256"] == digest
    assert observed["endpoint"] == ENDPOINT
    assert observed["served_model_name"] == MODEL
    assert observed["trust_anchor"] == "test_fixture_external_pin"
    assert set(observed["evidence_file_sha256"]) == set(subject.EVIDENCE_FILENAMES)
    assert observed["content_sha256"] == subject.content_sha256(
        {key: value for key, value in observed.items() if key != "content_sha256"}
    )
    identity.validate_current()


def test_self_asserted_bundle_is_not_its_own_trust_anchor(tmp_path: Path) -> None:
    root, _digest = _bundle(tmp_path)
    with pytest.raises(ValueError, match="not the compiled trusted deployment"):
        _load(root, "f" * 64)


def test_legacy_nonempty_identity_strings_are_not_a_deployment_attestation(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "legacy"
    bundle.mkdir()
    root = bundle / subject.ROOT_FILENAME
    root.write_text(
        json.dumps(
            {
                "schema_version": "production_openai_served_model_identity_v1",
                "served_model_name": MODEL,
                "model_artifact_identity": "anything-nonempty",
                "server_implementation_identity": "anything-nonempty",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="directory is not closed"):
        subject.load_authenticated_served_deployment_identity(
            root,
            expected_model_name=MODEL,
            expected_endpoint=ENDPOINT,
            trusted_attestation_sha256=_sha(root.read_bytes()),
        )


def test_replay_from_another_deployment_epoch_is_rejected_by_exact_root_pin(
    tmp_path: Path,
) -> None:
    first_root, first_digest = _bundle(tmp_path, name="first", pid=4242)
    second_root, second_digest = _bundle(tmp_path, name="second", pid=4343)
    assert first_digest != second_digest
    assert (
        _load(first_root, first_digest).deployment_instance_id
        != _load(second_root, second_digest).deployment_instance_id
    )
    with pytest.raises(ValueError, match="not the compiled trusted deployment"):
        _load(second_root, first_digest)


def test_sidecar_byte_tamper_is_rejected_before_semantic_use(tmp_path: Path) -> None:
    root, digest = _bundle(tmp_path)
    sidecar = root.parent / subject.EVIDENCE_FILENAMES["model_manifest"]
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(ValueError, match="evidence bytes differ"):
        _load(root, digest)


def test_root_rehash_after_sidecar_tamper_cannot_bypass_external_pin(tmp_path: Path) -> None:
    root, digest = _bundle(tmp_path)
    raw = json.loads(root.read_text(encoding="utf-8"))
    raw["body"]["served_model_name"] = "attacker/self-asserted"
    raw["content_sha256"] = subject.content_sha256(raw["body"])
    root.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")
    assert _sha(root.read_bytes()) != digest
    with pytest.raises(ValueError, match="not the compiled trusted deployment"):
        _load(root, digest)


def test_duplicate_json_key_in_referenced_evidence_is_rejected(tmp_path: Path) -> None:
    root, digest = _bundle(tmp_path)
    sidecar = root.parent / subject.EVIDENCE_FILENAMES["package_inventory_manifest"]
    sidecar.write_text('{"schema_version":"x","schema_version":"y"}\n', encoding="utf-8")
    root_value = json.loads(root.read_text(encoding="utf-8"))
    reference = root_value["body"]["evidence"]["package_inventory_manifest"]
    reference["sha256"] = _sha(sidecar.read_bytes())
    reference["size_bytes"] = len(sidecar.read_bytes())
    root_value["content_sha256"] = subject.content_sha256(root_value["body"])
    root.write_text(json.dumps(root_value, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        _load(root, _sha(root.read_bytes()))


def test_cross_document_relationship_mismatch_is_rejected_even_with_new_root_pin(
    tmp_path: Path,
) -> None:
    root, _digest = _bundle(tmp_path)
    root_value = json.loads(root.read_text(encoding="utf-8"))
    root_value["body"]["relationships"]["model_files_sha256"] = "a" * 64
    root_value["content_sha256"] = subject.content_sha256(root_value["body"])
    root.write_text(json.dumps(root_value, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="launch relationships do not match the root"):
        _load(root, _sha(root.read_bytes()))


def test_launch_model_alias_cannot_replace_exact_process_namespace_model_root(
    tmp_path: Path,
) -> None:
    bodies = _fixture_bodies()
    launch = bodies["launch_listener_binding"]
    launch["model_launch"]["model_argument"] = MODEL  # type: ignore[index]
    launch["deployment_instance_id"] = subject._deployment_instance_id(launch)
    relationships = launch["relationships"]  # type: ignore[assignment]
    relationships["launch_binding_sha256"] = subject._launch_binding_sha256(launch)  # type: ignore[index]
    with pytest.raises(ValueError, match="exact attested model root"):
        subject.seal_served_deployment_attestation_bundle(
            output_dir=tmp_path / "alias",
            endpoint=ENDPOINT,
            served_model_name=MODEL,
            sidecar_bodies=bodies,
        )


def test_package_inventory_cannot_name_a_different_python_executable(tmp_path: Path) -> None:
    bodies = _fixture_bodies()
    bodies["package_inventory_manifest"]["python_executable"] = {
        "path": "/host/usr/bin/python3",
        "size_bytes": 100,
        "sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="running server executable"):
        subject.seal_served_deployment_attestation_bundle(
            output_dir=tmp_path / "wrong-package-environment",
            endpoint=ENDPOINT,
            served_model_name=MODEL,
            sidecar_bodies=bodies,
        )


def test_bundle_directory_is_closed_and_symlink_sidecars_are_rejected(tmp_path: Path) -> None:
    root, digest = _bundle(tmp_path, name="extra")
    (root.parent / "unattested.txt").write_text("injected", encoding="utf-8")
    with pytest.raises(ValueError, match="directory is not closed"):
        _load(root, digest)

    second_root, second_digest = _bundle(tmp_path, name="symlink")
    sidecar = second_root.parent / subject.EVIDENCE_FILENAMES["model_manifest"]
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(sidecar.read_bytes())
    sidecar.unlink()
    sidecar.symlink_to(replacement)
    with pytest.raises(OSError):
        _load(second_root, second_digest)


def test_validate_current_detects_temporal_replacement(tmp_path: Path) -> None:
    root, digest = _bundle(tmp_path)
    identity = _load(root, digest)
    sidecar = root.parent / subject.EVIDENCE_FILENAMES["launch_listener_binding"]
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises((ValueError, RuntimeError)):
        identity.validate_current()


def test_model_endpoint_and_trust_digest_are_not_accepted_as_loose_strings(tmp_path: Path) -> None:
    root, digest = _bundle(tmp_path)
    with pytest.raises(ValueError, match="model differs"):
        subject.load_authenticated_served_deployment_identity(
            root,
            expected_model_name="other/model",
            expected_endpoint=ENDPOINT,
            trusted_attestation_sha256=digest,
        )
    with pytest.raises(ValueError, match="endpoint differs"):
        subject.load_authenticated_served_deployment_identity(
            root,
            expected_model_name=MODEL,
            expected_endpoint="http://camus:8011/v1",
            trusted_attestation_sha256=digest,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        subject.load_authenticated_served_deployment_identity(
            root,
            expected_model_name=MODEL,
            expected_endpoint=ENDPOINT,
            trusted_attestation_sha256="nonempty-but-not-content-addressed",
        )


def test_cgroup_container_instance_extraction_is_exact_and_unambiguous() -> None:
    instance = "a" * 64
    assert (
        subject._container_instance_id_from_cgroup(
            f"0::/system.slice/containerd-{instance}.scope\n".encode()
        )
        == instance
    )
    with pytest.raises(ValueError, match="exactly one"):
        subject._container_instance_id_from_cgroup(b"0::/\n")
    with pytest.raises(ValueError, match="exactly one"):
        subject._container_instance_id_from_cgroup(f"0::/{instance}/{('b' * 64)}\n".encode())


def test_collector_cli_requires_explicit_offline_runtime_evidence() -> None:
    parser = subject.build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert {
        "--server-pid",
        "--model-root",
        "--server-implementation-root",
        "--oci-manifest-json",
        "--container-runtime-inspect-json",
        "--package-inventory-json",
    }.issubset(options)
    assert not any("url" in option or "discover" in option for option in options)
