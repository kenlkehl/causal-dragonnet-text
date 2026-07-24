from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import pytest

import oci.inference.production_source_snapshot as snapshot_module
from oci.inference.production_source_snapshot import (
    SOURCE_SNAPSHOT_MANIFEST,
    create_production_source_snapshot,
    validate_production_source_snapshot,
)


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    (root / "oci").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "oci" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "scripts" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return root


def test_source_snapshot_is_closed_and_read_only(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    target = (tmp_path / "snapshot").resolve()

    created = create_production_source_snapshot(
        repository_root=repository,
        target_dir=target,
    )
    validated = validate_production_source_snapshot(target)

    assert created == validated
    assert created.file_count == 4
    assert not (target / "oci" / "module.py").stat().st_mode & stat.S_IWUSR
    manifest = json.loads((target / SOURCE_SNAPSHOT_MANIFEST).read_text())
    assert manifest["python_bytecode_writes_allowed"] is False


def test_source_snapshot_rejects_changed_or_extra_files(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    target = (tmp_path / "snapshot").resolve()
    create_production_source_snapshot(
        repository_root=repository,
        target_dir=target,
    )

    source_file = target / "oci" / "module.py"
    source_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    source_file.write_text("VALUE = 2\n", encoding="utf-8")
    source_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    with pytest.raises(ValueError, match="changed"):
        validate_production_source_snapshot(target)

    source_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    source_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    target.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    extra = target / "unexpected.py"
    extra.write_text("pass\n", encoding="utf-8")
    target.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    with pytest.raises(ValueError, match="unregistered"):
        validate_production_source_snapshot(target)


def test_source_snapshot_rejects_symlinked_source(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    target_file = repository / "outside.py"
    target_file.write_text("pass\n", encoding="utf-8")
    (repository / "oci" / "link.py").symlink_to(target_file)

    with pytest.raises(ValueError, match="symlinks"):
        create_production_source_snapshot(
            repository_root=repository,
            target_dir=(tmp_path / "snapshot").resolve(),
        )


@pytest.mark.parametrize("entry_kind", ["directory", "symlink", "fifo"])
def test_source_snapshot_rejects_every_unregistered_entry_type(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    repository = _repository(tmp_path)
    target = (tmp_path / "snapshot").resolve()
    create_production_source_snapshot(
        repository_root=repository,
        target_dir=target,
    )
    target.chmod(stat.S_IRWXU)
    extra = target / "unregistered"
    if entry_kind == "directory":
        extra.mkdir()
        extra.chmod(
            stat.S_IRUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH
        )
    elif entry_kind == "symlink":
        extra.symlink_to(repository)
    else:
        os.mkfifo(extra)
    target.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )

    with pytest.raises(ValueError, match="unregistered"):
        validate_production_source_snapshot(target)


def test_source_snapshot_rejects_arbitrary_hardlink_and_writable_mode(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    target = (tmp_path / "snapshot").resolve()
    create_production_source_snapshot(
        repository_root=repository,
        target_dir=target,
    )
    copied = target / "oci" / "module.py"
    copied.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with pytest.raises(ValueError, match="writable"):
        validate_production_source_snapshot(target)

    external = tmp_path / "same-bytes.py"
    shutil.copyfile(copied, external)
    parent = copied.parent
    parent.chmod(stat.S_IRWXU)
    copied.unlink()
    os.link(external, copied)
    copied.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    parent.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    with pytest.raises(ValueError, match="linked"):
        validate_production_source_snapshot(target)


def test_source_snapshot_manifest_is_closed_and_duplicate_free(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    target = (tmp_path / "snapshot").resolve()
    create_production_source_snapshot(
        repository_root=repository,
        target_dir=target,
    )
    manifest_path = target / SOURCE_SNAPSHOT_MANIFEST
    manifest_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unregistered"] = True
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    with pytest.raises(ValueError, match="closed schema"):
        validate_production_source_snapshot(target)

    manifest_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    manifest_path.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    manifest_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    with pytest.raises(ValueError, match="duplicate key"):
        validate_production_source_snapshot(target)


def test_source_snapshot_rejects_root_inode_substitution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = _repository(tmp_path)
    target = (tmp_path / "snapshot").resolve()
    create_production_source_snapshot(
        repository_root=repository,
        target_dir=target,
    )
    original_stable_file = snapshot_module._stable_file
    substituted = False

    def substitute_then_hash(path: Path):
        nonlocal substituted
        if not substituted:
            substituted = True
            backup = tmp_path / "snapshot-original"
            target.rename(backup)
            shutil.copytree(backup, target)
        return original_stable_file(path)

    monkeypatch.setattr(snapshot_module, "_stable_file", substitute_then_hash)
    with pytest.raises(RuntimeError, match="root changed"):
        validate_production_source_snapshot(target)
