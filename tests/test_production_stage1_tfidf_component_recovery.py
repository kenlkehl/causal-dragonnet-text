from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

import oci.inference.production_stage1_tfidf_component_recovery as recovery
from oci.config import AppliedInferenceConfig
from oci.inference.production_stage1_bundle import _seal_component
from oci.inference.production_stage1_tfidf_component_recovery import (
    TfidfComponentAttemptHandle,
    TfidfComponentAttemptManager,
    ValidatedTfidfComponentAttempt,
    publish_tfidf_component_descriptor,
)


def _descriptor(tmp_path: Path):
    split = tmp_path / "split_registry.json"
    split.write_text(
        json.dumps({"schema_version": "test", "rows": [0, 1, 2, 3]}),
        encoding="utf-8",
    )
    modeling = pd.DataFrame(
        {
            "clinical_text": ["a", "b", "c", "d"],
            "treatment_indicator": [0, 1, 0, 1],
            "outcome_indicator": [0, 0, 1, 1],
        }
    )
    return publish_tfidf_component_descriptor(
        descriptor_root=tmp_path / "descriptor",
        scientific_request_sha256="a" * 64,
        modeling_data=modeling,
        effective_config=asdict(AppliedInferenceConfig()),
        registry={"dataset_row_count": 4, "outer_folds": []},
        registry_content_sha256="b" * 64,
        split_registry_path=split,
        tfidf_workers=4,
        seed=42,
    )


def test_incomplete_descriptor_publication_is_preserved_and_does_not_block_resume(
    tmp_path: Path,
):
    incomplete = tmp_path / "descriptor" / "publication_interrupted"
    incomplete.mkdir(parents=True)
    (incomplete / "partial").write_text("preserve", encoding="utf-8")

    published = _descriptor(tmp_path)
    replayed = _descriptor(tmp_path)

    assert replayed.root == published.root
    assert (incomplete / "partial").read_text(encoding="utf-8") == "preserve"
    assert len(list((tmp_path / "descriptor").glob("publication_*"))) == 2


@pytest.mark.parametrize(
    "attack",
    ("extra_file", "symlink", "hardlink", "directory_substitution"),
)
def test_descriptor_loader_rejects_tree_and_path_substitution(
    tmp_path: Path,
    attack: str,
):
    descriptor = _descriptor(tmp_path)
    if attack == "extra_file":
        (descriptor.root / "unregistered").write_text("x", encoding="utf-8")
    elif attack in {"symlink", "hardlink"}:
        target = descriptor.root / "effective_config.json"
        external = tmp_path / "external-config.json"
        shutil.copyfile(target, external)
        target.unlink()
        if attack == "symlink":
            os.symlink(external, target)
        else:
            os.link(external, target)
    else:
        original = tmp_path / "original-publication"
        descriptor.root.rename(original)
        shutil.copytree(original, descriptor.root)

    with pytest.raises(ValueError):
        recovery.load_tfidf_component_descriptor(
            descriptor.manifest_path,
            expected_request_sha256="a" * 64,
        )


def test_descriptor_loader_rejects_changed_code_identity(
    tmp_path: Path,
    monkeypatch,
):
    descriptor = _descriptor(tmp_path)
    monkeypatch.setattr(
        recovery,
        "_current_code_identity",
        lambda: {"substituted.py": "f" * 64},
    )

    with pytest.raises(ValueError, match="invalid binding"):
        recovery.load_tfidf_component_descriptor(
            descriptor.manifest_path,
            expected_request_sha256="a" * 64,
        )


class _SealingFakeProcess:
    def __init__(self, *, target, args, name):
        del target, name
        self.request = args[0]
        self.pid = 44001
        self.exitcode = None
        self.alive = False
        self.terminated = False

    def start(self):
        attempt = Path(self.request.attempt_dir)
        component = attempt / "payload" / "tfidf"
        component.mkdir(parents=True)
        (component / "proof.json").write_text(
            json.dumps({"request": self.request.scientific_request_sha256}),
            encoding="utf-8",
        )
        sealed = _seal_component(
            component,
            request_sha256=self.request.scientific_request_sha256,
            component="tfidf",
        )
        recovery._seal_tfidf_attempt(
            self.request,
            result={
                "component_relative_path": "payload/tfidf",
                "component_manifest_sha256": sealed["content_sha256"],
            },
        )
        self.exitcode = 0

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        del timeout

    def terminate(self):
        self.terminated = True
        self.alive = False
        self.exitcode = -15

    def kill(self):
        self.alive = False
        self.exitcode = -9


class _FakeContext:
    def Process(self, *, target, args, name):
        return _SealingFakeProcess(target=target, args=args, name=name)


def test_complete_attempt_reuses_across_managers_and_materializes(
    tmp_path: Path,
    monkeypatch,
):
    descriptor = _descriptor(tmp_path)
    attempt_root = tmp_path / "recovery" / "attempts"
    progress = tmp_path / "recovery" / "progress.json"
    incomplete = attempt_root / "attempt_old"
    incomplete.mkdir(parents=True)
    (incomplete / "partial").write_text("preserved", encoding="utf-8")
    monkeypatch.setattr(
        recovery.mp,
        "get_context",
        lambda method: _FakeContext()
        if method == "spawn"
        else (_ for _ in ()).throw(AssertionError(method)),
    )
    monkeypatch.setattr(
        recovery,
        "_start_spawned_process_with_scope_hash_seed",
        lambda process, *, scope_seed: process.start(),
    )
    manager = TfidfComponentAttemptManager(
        attempt_root=attempt_root,
        progress_path=progress,
        descriptor=descriptor,
        scientific_request_sha256="a" * 64,
        seed=42,
    )

    started = manager.start()
    assert isinstance(started, TfidfComponentAttemptHandle)
    completed = manager.wait(started)
    assert isinstance(completed, ValidatedTfidfComponentAttempt)
    assert (incomplete / "partial").read_text(encoding="utf-8") == "preserved"
    target = tmp_path / "sibling_workflow_attempt" / "tfidf"
    target.parent.mkdir()
    manager.materialize(completed, target=target)
    assert (target / "component_manifest.json").is_file()

    sibling = TfidfComponentAttemptManager(
        attempt_root=attempt_root,
        progress_path=progress,
        descriptor=descriptor,
        scientific_request_sha256="a" * 64,
        seed=42,
    )
    reused = sibling.start()
    assert isinstance(reused, ValidatedTfidfComponentAttempt)
    assert reused.attempt_dir == completed.attempt_dir
    assert len(list(attempt_root.glob("attempt_*"))) == 2
    request = json.loads(
        (completed.attempt_dir / "attempt_request.json").read_text(
            encoding="utf-8"
        )
    )
    assert str(progress) not in json.dumps(request)
    assert request["scientific_request_sha256"] == "a" * 64

    request["unexpected"] = "substitution"
    (completed.attempt_dir / "attempt_request.json").write_text(
        json.dumps(request),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="request changed"):
        sibling.reusable()


def test_terminate_preserves_incomplete_attempt_and_records_failure(
    tmp_path: Path,
):
    descriptor = _descriptor(tmp_path)
    manager = TfidfComponentAttemptManager(
        attempt_root=tmp_path / "attempts",
        progress_path=tmp_path / "progress.json",
        descriptor=descriptor,
        scientific_request_sha256="a" * 64,
        seed=42,
    )
    attempt = manager.attempt_root / "attempt_manual"
    attempt.mkdir()
    request = recovery.TfidfComponentAttemptRequest(
        attempt_dir=str(attempt),
        scientific_request_sha256="a" * 64,
        descriptor_manifest_path=str(descriptor.manifest_path),
        descriptor_content_sha256=descriptor.content_sha256,
        attempt_request_sha256="c" * 64,
        seed=42,
    )
    process = _SealingFakeProcess(
        target=None,
        args=(request,),
        name="fake",
    )
    process.alive = True
    handle = TfidfComponentAttemptHandle(request=request, process=process)

    manager.terminate(handle, reason="legacy peer interrupted")

    assert attempt.is_dir()
    assert process.terminated is True
    progress = json.loads(manager.progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert progress["failure"]["exception_type"] == "PeerComponentFailure"
