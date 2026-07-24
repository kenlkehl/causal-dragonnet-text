from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import oci.inference.production_authenticated_tree_cache as tree_module
import oci.inference.production_embedding_cache_builder as builder_module
from oci.inference.production_authenticated_tree_cache import (
    AuthenticatedDirectoryTreeDriftError,
    authenticate_directory_tree,
    clear_authenticated_directory_tree_cache,
)


@pytest.fixture(autouse=True)
def _fresh_process_tree_cache():
    clear_authenticated_directory_tree_cache()
    yield
    clear_authenticated_directory_tree_cache()


def _tree(tmp_path: Path) -> Path:
    root = (tmp_path / "model").resolve()
    (root / "tokenizer").mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"safe"}\n', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"safe model weights")
    (root / "tokenizer" / "vocab.json").write_text(
        '{"safe":0}\n',
        encoding="utf-8",
    )
    return root


def test_projection_exactly_matches_historical_builder_and_workflow_wire(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    snapshot = authenticate_directory_tree(root)
    historical_path, historical = builder_module._model_tree_snapshot(root)

    assert snapshot.path == historical_path
    assert snapshot.local_model_provenance() == {
        "path": str(historical_path),
        **historical,
    }
    workflow = snapshot.workflow_path_identity()
    assert set(workflow) == {
        "kind",
        "path",
        "file_count",
        "total_size_bytes",
        "tree_sha256",
        "files",
    }
    assert workflow["kind"] == "directory"
    assert workflow["path"] == str(root)
    assert workflow["file_count"] == 3
    expected_tree_sha = tree_module._sha256_json(workflow["files"])
    assert workflow["tree_sha256"] == expected_tree_sha
    assert workflow["total_size_bytes"] == sum(row["size_bytes"] for row in workflow["files"])


def test_workflow_projection_preserves_legacy_unicode_json_bytes(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    (root / "é.json").write_text('{"unicode":true}\n', encoding="utf-8")
    workflow = authenticate_directory_tree(root).workflow_path_identity()
    legacy_payload = json.dumps(
        workflow["files"],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )
    expected = hashlib.sha256(legacy_payload.encode("utf-8")).hexdigest()

    assert workflow["tree_sha256"] == expected
    assert tree_module._sha256_json(workflow["files"]) != expected


def test_repeated_and_concurrent_calls_hash_content_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _tree(tmp_path)
    calls: list[str] = []
    original = tree_module._stable_file_authentication

    def counted(tree_root: Path, relative_path: str):
        calls.append(relative_path)
        return original(tree_root, relative_path)

    monkeypatch.setattr(tree_module, "_stable_file_authentication", counted)
    first = authenticate_directory_tree(root)
    expected_calls = first.local_model_provenance()["file_count"]
    assert len(calls) == expected_calls

    second = authenticate_directory_tree(root)
    assert second is first
    assert len(calls) == expected_calls

    with ThreadPoolExecutor(max_workers=4) as pool:
        observed = tuple(pool.map(lambda _index: authenticate_directory_tree(root), range(8)))
    assert all(value is first for value in observed)
    assert len(calls) == expected_calls


def test_same_size_mutation_poisons_same_process_capability(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    authenticate_directory_tree(root)
    artifact = root / "model.safetensors"
    before = artifact.stat()
    replacement = b"changed weights!!!"
    assert len(replacement) == len(artifact.read_bytes())
    artifact.write_bytes(replacement)
    os.utime(
        artifact,
        ns=(
            int(before.st_atime_ns),
            int(before.st_mtime_ns) + 2_000_000_000,
        ),
    )

    with pytest.raises(AuthenticatedDirectoryTreeDriftError, match="inventory"):
        authenticate_directory_tree(root)
    with pytest.raises(AuthenticatedDirectoryTreeDriftError, match="previously"):
        authenticate_directory_tree(root)


def test_inode_replacement_and_root_substitution_fail_closed(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    authenticate_directory_tree(root)
    artifact = root / "model.safetensors"
    replacement = root / "replacement.safetensors"
    replacement.write_bytes(artifact.read_bytes())
    os.replace(replacement, artifact)

    with pytest.raises(AuthenticatedDirectoryTreeDriftError):
        authenticate_directory_tree(root)

    clear_authenticated_directory_tree_cache()
    authenticate_directory_tree(root)
    backup = tmp_path / "old-model"
    root.rename(backup)
    shutil.copytree(backup, root)
    with pytest.raises(AuthenticatedDirectoryTreeDriftError):
        authenticate_directory_tree(root)


@pytest.mark.parametrize("substitution", ["missing", "file", "symlink"])
def test_root_loss_or_special_substitution_is_poisoned(
    tmp_path: Path,
    substitution: str,
) -> None:
    root = _tree(tmp_path)
    authenticate_directory_tree(root)
    backup = tmp_path / "authenticated-backup"
    root.rename(backup)
    if substitution == "file":
        root.write_text("not a directory", encoding="utf-8")
    elif substitution == "symlink":
        root.symlink_to(backup, target_is_directory=True)

    with pytest.raises(AuthenticatedDirectoryTreeDriftError, match="root changed"):
        authenticate_directory_tree(root)
    with pytest.raises(AuthenticatedDirectoryTreeDriftError, match="previously"):
        authenticate_directory_tree(root)


def test_poison_registry_is_bounded(tmp_path: Path) -> None:
    poisoned: list[Path] = []
    for index in range(tree_module._MAX_CACHE_ENTRIES + 4):
        root = (tmp_path / f"model-{index:02d}").resolve()
        root.mkdir()
        artifact = root / "model.safetensors"
        artifact.write_bytes(b"safe")
        authenticate_directory_tree(root)
        artifact.write_bytes(b"changed")
        with pytest.raises(AuthenticatedDirectoryTreeDriftError):
            authenticate_directory_tree(root)
        poisoned.append(root)

    assert len(tree_module._POISONED_PATHS) == tree_module._MAX_CACHE_ENTRIES
    assert tuple(tree_module._POISONED_PATHS) == tuple(poisoned[-tree_module._MAX_CACHE_ENTRIES :])


@pytest.mark.parametrize("change", ["add", "remove"])
def test_membership_drift_fails_closed(tmp_path: Path, change: str) -> None:
    root = _tree(tmp_path)
    authenticate_directory_tree(root)
    if change == "add":
        (root / "added.json").write_text("{}\n", encoding="utf-8")
    else:
        (root / "config.json").unlink()

    with pytest.raises(AuthenticatedDirectoryTreeDriftError):
        authenticate_directory_tree(root)


def test_initial_authentication_rejects_links_specials_and_executables(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    (root / "linked.json").symlink_to(root / "config.json")
    with pytest.raises(ValueError, match="linked or special"):
        authenticate_directory_tree(root)

    (root / "linked.json").unlink()
    clear_authenticated_directory_tree_cache()
    script = root / "unsafe.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="executable"):
        authenticate_directory_tree(root)

    script.unlink()
    clear_authenticated_directory_tree_cache()
    fifo = root / "unsafe.fifo"
    os.mkfifo(fifo)
    try:
        with pytest.raises(ValueError, match="linked or special"):
            authenticate_directory_tree(root)
    finally:
        fifo.unlink()


@pytest.mark.parametrize("mutation_point", ["before_later_file", "after_last_file"])
def test_initial_authentication_rejects_tree_drift_during_hashing(
    tmp_path: Path,
    monkeypatch,
    mutation_point: str,
) -> None:
    root = _tree(tmp_path)
    original = tree_module._stable_file_authentication
    calls = 0

    def mutate_during_hash(tree_root: Path, relative_path: str):
        nonlocal calls
        calls += 1
        if mutation_point == "before_later_file" and calls == 1:
            (tree_root / "tokenizer" / "vocab.json").write_text(
                '{"changed":1}\n',
                encoding="utf-8",
            )
        result = original(tree_root, relative_path)
        if mutation_point == "after_last_file" and calls == 3:
            (tree_root / "config.json").write_text(
                '{"model_type":"changed"}\n',
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(
        tree_module,
        "_stable_file_authentication",
        mutate_during_hash,
    )
    with pytest.raises(RuntimeError, match="changed during full authentication"):
        authenticate_directory_tree(root)


def test_snapshot_cannot_be_copied_pickled_or_used_after_fork(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("fork is unavailable")
    root = _tree(tmp_path)
    snapshot = authenticate_directory_tree(root)

    with pytest.raises(TypeError, match="copied"):
        copy.copy(snapshot)
    with pytest.raises(TypeError, match="copied"):
        copy.deepcopy(snapshot)
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(snapshot)

    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        result: dict[str, object] = {}
        try:
            snapshot.local_model_provenance()
        except RuntimeError:
            result["inherited_rejected"] = True
        else:
            result["inherited_rejected"] = False
        try:
            child_snapshot = authenticate_directory_tree(root)
            result["fresh_child_authenticated"] = (
                child_snapshot.local_model_provenance() == snapshot._local_model_provenance
            )
        except BaseException as exc:  # pragma: no cover - surfaced in parent
            result["error"] = f"{type(exc).__name__}: {exc}"
        os.write(write_fd, json.dumps(result).encode("utf-8"))
        os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    payload = os.read(read_fd, 4096)
    os.close(read_fd)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    result = json.loads(payload)
    assert result == {
        "inherited_rejected": True,
        "fresh_child_authenticated": True,
    }


def test_explicit_clear_requires_a_new_full_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _tree(tmp_path)
    calls = 0
    original = tree_module._stable_file_authentication

    def counted(tree_root: Path, relative_path: str):
        nonlocal calls
        calls += 1
        return original(tree_root, relative_path)

    monkeypatch.setattr(tree_module, "_stable_file_authentication", counted)
    first = authenticate_directory_tree(root)
    file_count = first.local_model_provenance()["file_count"]
    clear_authenticated_directory_tree_cache()
    authenticate_directory_tree(root)
    assert calls == 2 * file_count
