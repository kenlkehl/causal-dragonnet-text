from __future__ import annotations

import hashlib
import os
import socket
from pathlib import Path
from typing import Any

import pytest

from oci.inference import production_served_model_attestation as subject
from tests.test_production_served_model_attestation import (
    ENDPOINT,
    MODEL,
    _fixture_bodies,
)


def _directory_fd(path: Path) -> int:
    return os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )


def _root_document(sidecars: dict[str, dict[str, object]]) -> dict[str, Any]:
    module_sha, helper_sha = subject._module_and_helper_hashes()
    launch = sidecars["launch_listener_binding"]
    body = {
        "attestation_scope": "one_exact_running_openai_compatible_deployment",
        "endpoint": ENDPOINT,
        "served_model_name": MODEL,
        "deployment_instance_id": launch["deployment_instance_id"],
        "collector": {
            "implementation_version": (
                subject.SERVED_DEPLOYMENT_ATTESTATION_IMPLEMENTATION_VERSION
            ),
            "attestation_module_sha256": module_sha,
            "helper_script_sha256": helper_sha,
        },
        "evidence": {
            role: {
                "relative_path": filename,
                "sha256": "a" * 64,
                "size_bytes": 1,
                "schema_version": subject.EVIDENCE_SCHEMAS[role],
            }
            for role, filename in subject.EVIDENCE_FILENAMES.items()
        },
        "relationships": launch["relationships"],
    }
    return subject._wrapped(subject.SERVED_DEPLOYMENT_ATTESTATION_SCHEMA_VERSION, body)


@pytest.mark.parametrize(
    "value",
    [
        "/",
        "srv/model",
        "./srv/model",
        "/srv/../model",
        "/srv/./model",
        "/srv//model",
        "/srv/model/",
        r"/srv\model",
    ],
)
def test_process_paths_must_be_canonical_nonroot_absolute_posix_paths(value: str) -> None:
    with pytest.raises(
        ValueError,
        match="canonical non-root absolute process path|POSIX separators",
    ):
        subject._canonical_process_absolute_path(value, label="test path")


def test_target_tree_is_read_only_through_the_held_process_root_descriptor(
    tmp_path: Path,
) -> None:
    process_root = tmp_path / "target-process-root"
    target_model = process_root / "srv" / "model"
    target_model.mkdir(parents=True)
    target_payload = b"bytes loaded inside the target mount namespace"
    (target_model / "config.json").write_bytes(target_payload)

    collector_decoy = tmp_path / "collector-namespace-decoy" / "srv" / "model"
    collector_decoy.mkdir(parents=True)
    decoy_payload = b"different collector namespace bytes"
    (collector_decoy / "config.json").write_bytes(decoy_payload)

    root_fd = _directory_fd(process_root)
    try:
        canonical, rows = subject._snapshot_target_tree(
            root_fd,
            "/srv/model",
            label="model root",
        )
    finally:
        os.close(root_fd)

    assert canonical == "/srv/model"
    assert rows == [
        {
            "relative_path": "config.json",
            "size_bytes": len(target_payload),
            "sha256": hashlib.sha256(target_payload).hexdigest(),
        }
    ]
    assert rows[0]["sha256"] != hashlib.sha256(decoy_payload).hexdigest()


def test_target_tree_rejects_a_symlinked_path_component(tmp_path: Path) -> None:
    process_root = tmp_path / "process-root"
    process_root.mkdir()
    outside = tmp_path / "outside"
    (outside / "model").mkdir(parents=True)
    (outside / "model" / "weights.bin").write_bytes(b"outside")
    (process_root / "srv").symlink_to(outside, target_is_directory=True)

    root_fd = _directory_fd(process_root)
    try:
        with pytest.raises((ValueError, OSError), match="symlink|non-directory|Not a directory"):
            subject._snapshot_target_tree(root_fd, "/srv/model", label="model root")
    finally:
        os.close(root_fd)


def test_target_tree_rejects_a_symlinked_leaf(tmp_path: Path) -> None:
    process_root = tmp_path / "process-root"
    model = process_root / "srv" / "model"
    model.mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (model / "weights.bin").symlink_to(outside)

    root_fd = _directory_fd(process_root)
    try:
        with pytest.raises(ValueError, match="cannot contain symlinks"):
            subject._snapshot_target_tree(root_fd, "/srv/model", label="model root")
    finally:
        os.close(root_fd)


def test_target_tree_rejects_a_multiply_linked_file(tmp_path: Path) -> None:
    process_root = tmp_path / "process-root"
    model = process_root / "srv" / "model"
    model.mkdir(parents=True)
    original = tmp_path / "outside.bin"
    original.write_bytes(b"same inode")
    os.link(original, model / "weights.bin")

    root_fd = _directory_fd(process_root)
    try:
        with pytest.raises(ValueError, match="multiply linked"):
            subject._snapshot_target_tree(root_fd, "/srv/model", label="model root")
    finally:
        os.close(root_fd)


def test_target_tree_rejects_a_fifo_without_opening_or_blocking_on_it(
    tmp_path: Path,
) -> None:
    process_root = tmp_path / "process-root"
    model = process_root / "srv" / "model"
    model.mkdir(parents=True)
    os.mkfifo(model / "weights.pipe")

    root_fd = _directory_fd(process_root)
    try:
        with pytest.raises(ValueError, match="non-regular entry"):
            subject._snapshot_target_tree(root_fd, "/srv/model", label="model root")
    finally:
        os.close(root_fd)


def test_held_anchor_detects_namespace_path_replacement(tmp_path: Path) -> None:
    process_root = tmp_path / "process-root"
    (process_root / "srv" / "model").mkdir(parents=True)
    root_fd = _directory_fd(process_root)
    anchor = subject._open_anchored_target_directory(
        root_fd,
        "/srv/model",
        label="model root",
    )
    try:
        (process_root / "srv").rename(process_root / "srv-original")
        (process_root / "srv" / "model").mkdir(parents=True)
        with pytest.raises(
            RuntimeError,
            match="replaced during traversal|inode changed during traversal",
        ):
            anchor.validate(label="model root")
    finally:
        anchor.close()
        os.close(root_fd)


def test_target_tree_root_identity_is_rechecked_after_snapshot(tmp_path: Path) -> None:
    process_root = tmp_path / "process-root"
    model = process_root / "srv" / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_bytes(b"original")
    root_fd = _directory_fd(process_root)
    try:
        canonical, _rows, identity = subject._snapshot_target_tree_with_identity(
            root_fd,
            "/srv/model",
            label="model root",
        )
        model.rename(process_root / "srv" / "original-model")
        replacement = process_root / "srv" / "model"
        replacement.mkdir()
        (replacement / "config.json").write_bytes(b"replacement")
        with pytest.raises(RuntimeError, match="root changed after its file snapshot"):
            subject._validate_target_tree_root_identity(
                root_fd,
                canonical,
                expected_identity=identity,
                label="model root",
            )
    finally:
        os.close(root_fd)


@pytest.mark.parametrize(
    "raw",
    [
        (
            b"python\0-m\0vllm.entrypoints.openai.api_server\0"
            b"--model\0/srv/models/exact\0"
            b"--served-model-name\0publisher/exact-served-model\0"
        ),
        (
            b"vllm\0serve\0/srv/models/exact\0"
            b"--served-model-name\0publisher/exact-served-model\0"
        ),
    ],
)
def test_cmdline_accepts_only_supported_absolute_vllm_model_forms(raw: bytes) -> None:
    assert subject._parse_cmdline(
        raw,
        model_name=MODEL,
        model_root="/srv/models/exact",
    ) == {
        "model_argument": "/srv/models/exact",
        "served_model_names": [MODEL],
        "served_name_flag_present": True,
    }


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (
            b"vllm\0serve\0publisher/repository-id\0--served-model-name\0"
            b"publisher/exact-served-model\0",
            "canonical non-root absolute process path",
        ),
        (
            b"vllm\0serve\0relative/model\0--served-model-name\0" b"publisher/exact-served-model\0",
            "canonical non-root absolute process path",
        ),
        (
            b"vllm\0serve\0/srv/./models/exact\0--served-model-name\0"
            b"publisher/exact-served-model\0",
            "canonical non-root absolute process path",
        ),
        (
            b"vllm\0serve\0/srv/models/exact\0--model\0/srv/models/exact\0"
            b"--served-model-name\0publisher/exact-served-model\0",
            "exact legacy vLLM entrypoint|exactly one model argument",
        ),
        (
            b"vllm\0serve\0/srv/models/exact\0--served-model-name\0"
            b"publisher/exact-served-model\0--served-model-name\0"
            b"publisher/exact-served-model\0",
            "exactly one expected served name",
        ),
        (
            b"vllm\0serve=/srv/models/exact\0--served-model-name\0"
            b"publisher/exact-served-model\0",
            "exactly one model argument",
        ),
        (
            b"vllm\0serve\0/srv/models/exact\0"
            b"--served-model-name=publisher/exact-served-model\0",
            "space-separated model flags",
        ),
    ],
)
def test_cmdline_rejects_aliases_relative_noncanonical_and_ambiguous_forms(
    raw: bytes,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        subject._parse_cmdline(
            raw,
            model_name=MODEL,
            model_root="/srv/models/exact",
        )


@pytest.mark.parametrize(
    "raw",
    [
        (
            b"python\0/tmp/unrelated.py\0--model\0/srv/models/exact\0"
            b"--served-model-name\0publisher/exact-served-model\0"
        ),
        (
            b"python\0/tmp/unrelated.py\0serve\0/srv/models/exact\0"
            b"--served-model-name\0publisher/exact-served-model\0"
        ),
    ],
)
def test_cmdline_rejects_vllm_flag_spoofing_by_an_unrelated_program(raw: bytes) -> None:
    with pytest.raises(ValueError, match="vLLM|exactly one model argument"):
        subject._parse_cmdline(
            raw,
            model_name=MODEL,
            model_root="/srv/models/exact",
        )


def test_cross_document_validation_rejects_launch_model_root_mismatch() -> None:
    sidecars = _fixture_bodies()
    launch = sidecars["launch_listener_binding"]
    launch["model_launch"]["model_argument"] = "/srv/models/different-model"  # type: ignore[index]
    launch["deployment_instance_id"] = subject._deployment_instance_id(launch)
    launch["relationships"]["launch_binding_sha256"] = (  # type: ignore[index]
        subject._launch_binding_sha256(launch)
    )
    root = _root_document(sidecars)

    with pytest.raises(ValueError, match="exact attested model root"):
        subject._validate_root_and_sidecars(
            root,
            sidecars,
            expected_model_name=MODEL,
            expected_endpoint=ENDPOINT,
        )


def test_descriptor_hashing_reads_the_held_inode_not_a_path_label(tmp_path: Path) -> None:
    exact = tmp_path / "exact-running-file"
    decoy = tmp_path / "collector-decoy-file"
    payload = b"exact bytes held by the descriptor"
    exact.write_bytes(payload)
    decoy.write_bytes(b"different decoy bytes")

    descriptor = os.open(exact, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        record = subject._stable_descriptor_file_record(
            descriptor,
            path_key="path",
            path_value=str(decoy),
            label="exact descriptor",
        )
    finally:
        os.close(descriptor)

    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["sha256"] != hashlib.sha256(decoy.read_bytes()).hexdigest()


def test_process_executable_snapshot_hashes_the_proc_exe_inode(tmp_path: Path) -> None:
    executable = tmp_path / "target-python"
    payload = b"target executable bytes"
    executable.write_bytes(payload)
    process_directory = tmp_path / "fake-proc-pid"
    process_directory.mkdir()
    (process_directory / "exe").symlink_to(executable)
    expected = subject._stat_key(executable.stat())

    process_fd = _directory_fd(process_directory)
    try:
        record = subject._snapshot_process_executable(
            process_fd,
            expected_path=str(executable),
            expected_inode=expected,
        )
    finally:
        os.close(process_fd)

    assert record == {
        "path": str(executable),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_process_executable_snapshot_rejects_a_deleted_exe_link(tmp_path: Path) -> None:
    process_directory = tmp_path / "fake-proc-pid"
    process_directory.mkdir()
    (process_directory / "exe").symlink_to("/usr/bin/server (deleted)")

    process_fd = _directory_fd(process_directory)
    try:
        with pytest.raises(ValueError, match="executable was deleted"):
            subject._snapshot_process_executable(
                process_fd,
                expected_path="/usr/bin/server",
                expected_inode=(1, 2, 3),
            )
    finally:
        os.close(process_fd)


def test_process_executable_snapshot_rejects_readlink_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "target-python"
    executable.write_bytes(b"target executable bytes")
    process_directory = tmp_path / "fake-proc-pid"
    process_directory.mkdir()
    (process_directory / "exe").symlink_to(executable)
    expected = subject._stat_key(executable.stat())
    real_readlink = os.readlink
    calls = 0

    def changing_readlink(path: str, *, dir_fd: int | None = None) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_readlink(path, dir_fd=dir_fd)
        return "/different/executable"

    monkeypatch.setattr(subject.os, "readlink", changing_readlink)
    process_fd = _directory_fd(process_directory)
    try:
        with pytest.raises(RuntimeError, match="changed while it was authenticated"):
            subject._snapshot_process_executable(
                process_fd,
                expected_path=str(executable),
                expected_inode=expected,
            )
    finally:
        os.close(process_fd)


def test_process_root_executable_must_resolve_to_the_same_inode(tmp_path: Path) -> None:
    process_root = tmp_path / "process-root"
    executable = process_root / "usr" / "bin" / "server"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"running executable")
    expected = subject._stat_key(executable.stat())
    root_fd = _directory_fd(process_root)
    try:
        subject._validate_executable_under_process_root(
            root_fd,
            "/usr/bin/server",
            expected_inode=expected,
        )
        replacement = executable.with_name("replacement")
        replacement.write_bytes(b"replacement executable")
        os.replace(replacement, executable)
        with pytest.raises(RuntimeError, match="running inode"):
            subject._validate_executable_under_process_root(
                root_fd,
                "/usr/bin/server",
                expected_inode=expected,
            )
    finally:
        os.close(root_fd)


def test_same_uts_namespace_hostname_is_read_without_a_fallback() -> None:
    namespace_fd = os.open(
        "/proc/self/ns/uts",
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        assert subject._hostname_from_uts_namespace_fd(namespace_fd) == (
            socket.gethostname().lower()
        )
    finally:
        os.close(namespace_fd)


def test_denied_distinct_uts_namespace_has_no_collector_hostname_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_namespace = tmp_path / "different-uts-namespace"
    fake_namespace.write_bytes(b"not the current namespace inode")

    def deny_setns(_descriptor: int, _namespace_type: int) -> None:
        raise PermissionError("setns deliberately denied")

    def forbidden_fallback() -> str:
        raise AssertionError("collector hostname fallback was attempted")

    monkeypatch.setattr(subject.os, "setns", deny_setns, raising=False)
    monkeypatch.setattr(subject.socket, "gethostname", forbidden_fallback)
    namespace_fd = os.open(
        fake_namespace,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        with pytest.raises(
            RuntimeError,
            match="PermissionError: setns deliberately denied",
        ):
            subject._hostname_from_uts_namespace_fd(namespace_fd)
    finally:
        os.close(namespace_fd)


def test_epoch_comparison_reports_exact_changed_fields() -> None:
    before = {"start_time_ticks": 10, "cmdline": b"vllm"}
    subject._assert_process_epoch_unchanged(before, dict(before))
    with pytest.raises(RuntimeError, match="cmdline, start_time_ticks"):
        subject._assert_process_epoch_unchanged(
            before,
            {"start_time_ticks": 11, "cmdline": b"other"},
        )


def test_listener_evidence_requires_an_inode_owned_by_the_exact_process(
    tmp_path: Path,
) -> None:
    fake_process = tmp_path / "fake-proc-pid"
    fd_directory = fake_process / "fd"
    net_directory = fake_process / "net"
    fd_directory.mkdir(parents=True)
    net_directory.mkdir()
    (fd_directory / "3").symlink_to("socket:[12345]")
    header = "sl local_address rem_address st tx_rx tr tm retr uid timeout inode\n"
    listening = (
        "0: 00000000:1F4A 00000000:0000 0A 00000000:00000000 " "00:00000000 00000000 1000 0 12345\n"
    )
    (net_directory / "tcp").write_text(header + listening, encoding="ascii")
    (net_directory / "tcp6").write_text(header, encoding="ascii")

    process_fd = _directory_fd(fake_process)
    try:
        assert subject._owned_listener_records(process_fd, port=8010) == [
            {
                "address": "0.0.0.0",
                "port": 8010,
                "state": "LISTEN",
                "socket_inode": 12345,
                "owned_by_process": True,
            }
        ]
        (fd_directory / "3").unlink()
        (fd_directory / "3").symlink_to("socket:[99999]")
        with pytest.raises(ValueError, match="owns no LISTEN socket"):
            subject._owned_listener_records(process_fd, port=8010)
    finally:
        os.close(process_fd)
