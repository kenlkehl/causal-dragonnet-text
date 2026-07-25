from __future__ import annotations

import copy
import inspect
import json
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import oci.inference.prepared_stage1_context as context_module
import oci.inference.production_stage1_cluster_preflight_artifact as preflight_module
from oci.inference.prepared_stage1_context import (
    PREPARED_STAGE1_CONTEXT_LOCATOR_SCHEMA,
    PREPARED_STAGE1_CONTEXT_MANIFEST_NAME,
    PREPARED_STAGE1_CONTEXT_SCIENTIFIC_SCHEMA,
    PreparedStage1ContextArtifact,
    load_prepared_stage1_context,
    rebind_prepared_stage1_context_locators,
)
from oci.inference.production_role_neutral_process_executor import (
    _option_mapping,
)
from oci.inference.production_stage1_bundle import Stage1BundleBuildOptions


def _request(*, dataset_path: str, runtime_root: str) -> dict:
    body = {
        "schema_version": "fixture_stage1_request_v1",
        "scientific_key": "same-science",
        "dataset_path": dataset_path,
        "runtime_root": runtime_root,
    }
    return {
        **body,
        "request_sha256": context_module._sha256_json(body),
    }


def _projection(request) -> dict:
    body = {
        "schema_version": "fixture_path_neutral_projection_v1",
        "scientific_key": request["scientific_key"],
        "split_registry_content_sha256": "5" * 64,
    }
    return {**body, "content_sha256": context_module._sha256_json(body)}


def _options(tmp_path: Path, prefix: str) -> dict:
    root = tmp_path / prefix
    options = Stage1BundleBuildOptions(
        dataset_path=root / "cohort.parquet",
        config_path=root / "stage1.json",
        embedding_cache_dir=root / "cache",
        output_dir=root / "output",
        unit_id_column="person_id",
        initial_training_partitions=3,
        query_config_path=root / "query.json",
        cluster_preflight_manifest_path=root / "preflight.json",
        cluster_preflight_state_bundle_manifest_path=root / "state.json",
    )
    return _option_mapping(SimpleNamespace(options=options))


def _scientific() -> dict:
    projection = _projection(
        _request(dataset_path="/old/cohort", runtime_root="/old/runtime")
    )
    body = {
        "schema_version": PREPARED_STAGE1_CONTEXT_SCIENTIFIC_SCHEMA,
        "stage1_request_scientific_projection": projection,
        "stage1_request_scientific_compatibility_sha256": projection[
            "content_sha256"
        ],
        "split_registry": {
            "dataset_row_count": 1,
            "content_sha256": "5" * 64,
        },
        "split_registry_content_sha256": "5" * 64,
    }
    return {**body, "content_sha256": context_module._sha256_json(body)}


def _locator(tmp_path: Path, prefix: str) -> dict:
    return context_module._locator_payload(
        stage1_build_options=_options(tmp_path, prefix),
        architecture_profiles={"fixture": {"closed": True}},
        runtime_compatibility_class="fixture-runtime",
        scientific_compatibility_sha256=_scientific()[
            "stage1_request_scientific_compatibility_sha256"
        ],
        exact_stage1_request=_request(
            dataset_path=f"/{prefix}/cohort",
            runtime_root=f"/{prefix}/runtime",
        ),
    )


@pytest.fixture(autouse=True)
def _path_neutral_projection(monkeypatch):
    monkeypatch.setattr(
        preflight_module,
        "stage1_request_scientific_compatibility_projection",
        _projection,
    )


def _make_writable(root: Path) -> None:
    root.chmod(stat.S_IRWXU)
    for child in root.iterdir():
        child.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_scientific_root_excludes_locators_and_supports_rebinding(
    tmp_path: Path,
) -> None:
    first = context_module._publish_prepared_stage1_context_payloads(
        root=(tmp_path / "first").resolve(),
        scientific=_scientific(),
        locator=_locator(tmp_path, "old"),
    )
    rebound = rebind_prepared_stage1_context_locators(
        source_manifest_path=first.manifest_path,
        output_root=(tmp_path / "rebound").resolve(),
        stage1_build_options=_options(tmp_path, "new"),
        exact_stage1_request=_request(
            dataset_path="/new/cohort",
            runtime_root="/new/runtime",
        ),
    )
    assert first.content_root_sha256 == rebound.content_root_sha256
    assert (
        first.manifest["execution_locator_sha256"]
        != rebound.manifest["execution_locator_sha256"]
    )
    assert (
        first.execution_locators["stage1_build_options"]["dataset_path"]
        != rebound.execution_locators["stage1_build_options"]["dataset_path"]
    )
    assert rebound.execution_locators["schema_version"] == (
        PREPARED_STAGE1_CONTEXT_LOCATOR_SCHEMA
    )
    _make_writable(first.root)
    _make_writable(rebound.root)


def test_context_directory_is_byte_relocatable_and_tamper_fails(
    tmp_path: Path,
) -> None:
    original = context_module._publish_prepared_stage1_context_payloads(
        root=(tmp_path / "original").resolve(),
        scientific=_scientific(),
        locator=_locator(tmp_path, "same"),
    )
    moved_root = (tmp_path / "moved").resolve()
    shutil.copytree(original.root, moved_root)
    reopened = load_prepared_stage1_context(
        moved_root / PREPARED_STAGE1_CONTEXT_MANIFEST_NAME
    )
    assert reopened.content_root_sha256 == original.content_root_sha256

    _make_writable(moved_root)
    locator = moved_root / "execution_locators.json"
    locator.write_bytes(locator.read_bytes() + b" ")
    for child in moved_root.iterdir():
        child.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    moved_root.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    with pytest.raises(ValueError, match="payload changed"):
        load_prepared_stage1_context(
            moved_root / PREPARED_STAGE1_CONTEXT_MANIFEST_NAME
        )
    _make_writable(original.root)
    _make_writable(moved_root)


def test_context_rehydration_source_cannot_call_monolithic_prepare() -> None:
    source = inspect.getsource(PreparedStage1ContextArtifact.reconstruct)
    assert "ProductionStage1BundleBuilder(options).prepare" not in source
    assert ".prepare()" not in source
