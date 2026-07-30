#!/usr/bin/env python3
"""Inspect or terminate the superseded remote token-attention catalog run.

This controller is deliberately tied to one immutable request.  It refuses to
signal anything unless the workflow parent and persistent worker process groups
can be authenticated from /proc and the workflow-owned marker files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/data1/ken/pcori_dev/causal-dragonnet-text")
SNAPSHOT_ROOT = REPO_ROOT / (
    "artifacts/production_source_snapshot_20260730_"
    "token_attention_complete_htr_evidence_remote_cache_import_v3"
)
DURABLE_ROOT = REPO_ROOT / (
    "artifacts/production_all_evidence_five_conf_five_mod_1000_"
    "r15_token_attention_complete_evidence_cache_import_v3_"
    "remote_kehl-lab_gpu01"
)
DEPLOYMENT_PROFILE = REPO_ROOT / (
    "artifacts/runtime_profiles/generated/"
    "portable_all_evidence_deployment_nsclc.five-conf-five-mod."
    "r15_token_attention_complete_evidence_cache_import_v3_"
    "remote_kehl-lab_gpu01.json"
)
SCRATCH_ROOT = REPO_ROOT / (
    "artifacts/production_scratch/five_conf_five_mod_1000_"
    "r15_token_attention_complete_evidence_cache_import_v3_"
    "remote_kehl-lab_gpu01"
)
REQUEST_SHA256 = (
    "907b6d60fb0281391462f7afcc1021f7262e2160a1353994390ff73bde9582f7"
)
MARKER_SCHEMA = "production_stage1_worker_process_group_ready_v2"
REQUIRED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "PYTHONPATH": str(SNAPSHOT_ROOT),
}
PARENT_CMDLINE_FRAGMENTS = (
    str(SNAPSHOT_ROOT / "scripts/run_production_all_evidence_workflow.py"),
    "--deployment-profile",
    str(DEPLOYMENT_PROFILE),
)
MARKER_ROOT = (
    SCRATCH_ROOT
    / "production_all_evidence_workflow"
    / REQUEST_SHA256
    / "stage1_modeling"
)
EVENT_LOG = (
    REPO_ROOT
    / "artifacts/operational_controls/"
    "five_conf_obsolete_token_catalog_remote_stop_20260730.events.jsonl"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _append_event(event: str, **details: Any) -> None:
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": event,
        "recorded_at": datetime.now().astimezone().isoformat(),
        **details,
    }
    with EVENT_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _process_identity(pid: int) -> dict[str, Any] | None:
    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        right_parenthesis = raw_stat.rfind(")")
        if right_parenthesis < 0:
            raise ValueError("malformed /proc stat")
        fields = raw_stat[right_parenthesis + 2 :].split()
        cmdline = (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", errors="replace")
            .strip()
        )
        proc_stat = os.stat(f"/proc/{pid}")
        return {
            "pid": pid,
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "start_ticks": int(fields[19]),
            "uid": int(proc_stat.st_uid),
            "cmdline": cmdline,
        }
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def _process_environment(pid: int) -> dict[str, str] | None:
    try:
        entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    result: dict[str, str] = {}
    for entry in entries:
        if b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def _environment_matches(pid: int) -> bool:
    environment = _process_environment(pid)
    return environment is not None and all(
        environment.get(key) == value
        for key, value in REQUIRED_ENVIRONMENT.items()
    )


def _all_processes() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        identity = _process_identity(int(path.name))
        if identity is not None:
            result.append(identity)
    return result


def _authenticate_parent(
    processes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    matches = [
        process
        for process in processes
        if all(
            fragment in str(process["cmdline"])
            for fragment in PARENT_CMDLINE_FRAGMENTS
        )
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(
            "refusing control because the superseded immutable request has "
            f"{len(matches)} matching workflow parents"
        )
    parent = matches[0]
    if int(parent["uid"]) != os.getuid() or not _environment_matches(
        int(parent["pid"])
    ):
        raise RuntimeError(
            "matching workflow parent ownership or environment is invalid"
        )
    if int(parent["pid"]) != int(parent["pgid"]):
        raise RuntimeError(
            "matching workflow parent does not lead a private process group"
        )

    group_members = [
        process
        for process in processes
        if int(process["pgid"]) == int(parent["pgid"])
    ]
    member_by_pid = {
        int(process["pid"]): process for process in processes
    }
    for member in group_members:
        if int(member["uid"]) != os.getuid() or not _environment_matches(
            int(member["pid"])
        ):
            raise RuntimeError(
                "workflow parent process group contains an unauthenticated member"
            )
        cursor = member
        visited: set[int] = set()
        while int(cursor["pid"]) != int(parent["pid"]):
            cursor_pid = int(cursor["pid"])
            if cursor_pid in visited:
                raise RuntimeError("cycle while authenticating process ancestry")
            visited.add(cursor_pid)
            next_process = member_by_pid.get(int(cursor["ppid"]))
            if next_process is None:
                raise RuntimeError(
                    "workflow parent process group contains a non-descendant"
                )
            cursor = next_process
    return {
        **parent,
        "authenticated_group_members": [
            {
                key: member[key]
                for key in ("pid", "ppid", "pgid", "sid", "start_ticks", "cmdline")
            }
            for member in sorted(group_members, key=lambda value: value["pid"])
        ],
    }


def _read_worker_marker(path: Path) -> dict[str, Any]:
    resolved_root = MARKER_ROOT.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    resolved_path.relative_to(resolved_root)
    before = os.lstat(path)
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or int(before.st_uid) != os.getuid()
    ):
        raise ValueError(f"marker is not private regular data: {path}")
    payload = path.read_bytes()
    after = os.lstat(path)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"marker changed while reading: {path}")
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_strict_json_object,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {token}")
        ),
    )
    required_fields = {
        "schema_version",
        "pid",
        "process_group_id",
        "process_start_time_ticks",
        "content_sha256",
    }
    if not isinstance(value, dict) or set(value) != required_fields:
        raise ValueError(f"marker fields are invalid: {path}")
    body = dict(value)
    declared_sha256 = body.pop("content_sha256")
    calculated_sha256 = hashlib.sha256(
        _canonical_json(body).encode("utf-8")
    ).hexdigest()
    if (
        value["schema_version"] != MARKER_SCHEMA
        or type(value["pid"]) is not int
        or int(value["pid"]) <= 0
        or value["process_group_id"] != value["pid"]
        or type(value["process_start_time_ticks"]) is not int
        or int(value["process_start_time_ticks"]) < 0
        or declared_sha256 != calculated_sha256
    ):
        raise ValueError(f"marker authentication failed: {path}")
    return value


def _authenticate_workers() -> list[dict[str, Any]]:
    if not MARKER_ROOT.exists():
        return []
    marker_paths = sorted(
        MARKER_ROOT.glob(
            "attempt_*/role_neutral_stage1_execution/"
            ".persistent-owner-execution-session/process-group-slot-*.json"
        )
    )
    live_workers: list[dict[str, Any]] = []
    for marker_path in marker_paths:
        marker = _read_worker_marker(marker_path)
        pid = int(marker["pid"])
        process = _process_identity(pid)
        if process is None or int(process["start_ticks"]) != int(
            marker["process_start_time_ticks"]
        ):
            continue
        if (
            int(process["uid"]) != os.getuid()
            or int(process["pid"]) != int(process["pgid"])
            or int(process["pid"]) != int(process["sid"])
            or not _environment_matches(pid)
        ):
            raise RuntimeError(
                f"live marked worker {pid} failed process authentication"
            )
        live_workers.append(
            {
                "marker_path": str(marker_path),
                **{
                    key: process[key]
                    for key in (
                        "pid",
                        "ppid",
                        "pgid",
                        "sid",
                        "start_ticks",
                        "cmdline",
                    )
                },
            }
        )
    unique = {int(worker["pid"]): worker for worker in live_workers}
    return [unique[pid] for pid in sorted(unique)]


def _verify_immutable_request() -> None:
    path = DURABLE_ROOT / "immutable_run_request.json"
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_json_object,
    )
    if (
        value.get("request_sha256") != REQUEST_SHA256
        or value.get("source_snapshot_root") != str(SNAPSHOT_ROOT)
        or value.get("scratch_root") != str(SCRATCH_ROOT)
    ):
        raise RuntimeError("obsolete immutable request identity is invalid")


def _still_same(process: dict[str, Any]) -> bool:
    observed = _process_identity(int(process["pid"]))
    return observed is not None and int(observed["start_ticks"]) == int(
        process["start_ticks"]
    )


def _inspect() -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    _verify_immutable_request()
    processes = _all_processes()
    return _authenticate_parent(processes), _authenticate_workers()


def _stop(
    parent: dict[str, Any] | None,
    workers: list[dict[str, Any]],
    *,
    grace_seconds: float,
) -> None:
    targets: list[dict[str, Any]] = []
    if parent is not None:
        targets.append(
            {
                "kind": "workflow_parent",
                "pid": int(parent["pid"]),
                "pgid": int(parent["pgid"]),
                "start_ticks": int(parent["start_ticks"]),
            }
        )
    targets.extend(
        {
            "kind": "persistent_owner_worker",
            "pid": int(worker["pid"]),
            "pgid": int(worker["pgid"]),
            "start_ticks": int(worker["start_ticks"]),
            "marker_path": worker["marker_path"],
        }
        for worker in workers
    )
    for target in targets:
        try:
            os.killpg(int(target["pgid"]), signal.SIGTERM)
        except ProcessLookupError:
            continue
    _append_event(
        "sigterm_sent_to_authenticated_obsolete_token_catalog_groups",
        request_sha256=REQUEST_SHA256,
        targets=targets,
    )

    deadline = time.monotonic() + grace_seconds
    remaining = [target for target in targets if _still_same(target)]
    while remaining and time.monotonic() < deadline:
        time.sleep(1.0)
        remaining = [target for target in remaining if _still_same(target)]
    if remaining:
        _append_event(
            "authenticated_groups_remain_after_sigterm_grace",
            targets=remaining,
            no_sigkill_policy=True,
        )
        raise RuntimeError(
            "authenticated groups remain after SIGTERM grace; no SIGKILL was "
            f"sent: {[target['pid'] for target in remaining]}"
        )
    _append_event(
        "authenticated_obsolete_token_catalog_groups_exited",
        target_count=len(targets),
        no_sigkill_policy=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("inspect", "stop", "assert-stopped"),
        help="inspect authentic targets or SIGTERM them",
    )
    parser.add_argument("--grace-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if not (1.0 <= args.grace_seconds <= 300.0):
        raise SystemExit("--grace-seconds must be between 1 and 300")

    parent, workers = _inspect()
    summary = {
        "request_sha256": REQUEST_SHA256,
        "workflow_parent": parent,
        "persistent_owner_workers": workers,
        "remote_process_visibility_required": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.mode == "stop":
        _stop(parent, workers, grace_seconds=args.grace_seconds)
        print(
            "superseded token-catalog workflow-owned groups exited after "
            "SIGTERM; "
            "no SIGKILL was sent"
        )
    elif args.mode == "assert-stopped" and (parent is not None or workers):
        raise RuntimeError(
            "superseded token-catalog workflow still owns live process "
            "groups; inspect and stop it before launching the corrected "
            "request"
        )
    elif parent is None and not workers:
        print(
            "no live superseded token-catalog workflow-owned process was found"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
