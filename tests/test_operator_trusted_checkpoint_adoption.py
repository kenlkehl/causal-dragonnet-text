from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import oci.inference.portable_artifacts as portable_artifacts_module
from oci.inference.operator_trusted_checkpoint_adoption import (
    OPERATOR_TRUSTED_ADOPTION_ATTESTATION,
    OPERATOR_TRUSTED_VALIDATION_POLICY,
    adopt_checkpoint_from_prior_full_byte_attestation,
    validate_operator_trusted_checkpoint_adoption,
    validate_operator_trusted_portable_artifact,
)
from oci.inference.portable_artifacts import (
    ArtifactCompatibility,
    adopt_checkpoint,
    publish_portable_artifact,
)
from oci.inference.portable_identity import identity_sha256


def _digest(label: str) -> str:
    return identity_sha256({"label": label})


def _compatibility() -> ArtifactCompatibility:
    return ArtifactCompatibility(
        dataset_identity=_digest("dataset"),
        split_identity=_digest("split"),
        row_order_identity=_digest("rows"),
        model_identities={"embed": _digest("embed")},
        prompt_identities={"extract": _digest("prompt")},
        configuration_identity=_digest("config"),
        seed_identity=_digest("seed"),
        producer_code_identity=_digest("producer"),
        runtime_compatibility_class="python-posix-test-v1",
    )


def _artifact_and_prior(tmp_path: Path):
    root = tmp_path / "artifact"
    root.mkdir()
    payload = root / "values.bin"
    payload.write_bytes(b"previously authenticated payload bytes")
    artifact = publish_portable_artifact(
        root=root,
        artifact_kind="embedding_cache",
        artifact_schema="operator_trust_test_v1",
        compatibility=_compatibility(),
        upstream_artifact_ids=(_digest("prepared"),),
        payload_paths=("values.bin",),
    )
    prior_root = tmp_path / "prior_adoptions"
    prior = adopt_checkpoint(
        source=artifact.root,
        attestation_root=prior_root,
        consumer_request_sha256=_digest("prior consumer"),
        validated_artifact=artifact,
    )
    prior_path = prior_root / f"{artifact.artifact_id}.adoption.json"
    return artifact, payload, prior, prior_path


def test_operator_trusted_adoption_skips_payload_reads_and_marks_limitations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, payload, prior, prior_path = _artifact_and_prior(tmp_path)
    original_hash = portable_artifacts_module._safe_file_hash_with_identity
    original_read = portable_artifacts_module._safe_read_with_identity

    def guarded_hash(path: Path, *, label: str):
        assert Path(path).resolve() != payload.resolve(), label
        return original_hash(path, label=label)

    def guarded_read(path: Path, *, label: str):
        assert Path(path).resolve() != payload.resolve(), label
        return original_read(path, label=label)

    monkeypatch.setattr(
        portable_artifacts_module,
        "_safe_file_hash_with_identity",
        guarded_hash,
    )
    monkeypatch.setattr(
        portable_artifacts_module,
        "_safe_read_with_identity",
        guarded_read,
    )
    trusted = validate_operator_trusted_portable_artifact(
        source=artifact.root,
        prior_attestation_path=prior_path,
        expected_kind="embedding_cache",
        expected_compatibility_key=artifact.compatibility_key,
        expected_upstream_artifact_ids=(_digest("prepared"),),
    )
    current_request = _digest("current consumer")
    adopted = adopt_checkpoint_from_prior_full_byte_attestation(
        source=artifact.root,
        prior_attestation_path=prior_path,
        attestation_root=tmp_path / "current_adoptions",
        consumer_request_sha256=current_request,
        expected_kind="embedding_cache",
        expected_compatibility_key=artifact.compatibility_key,
        expected_upstream_artifact_ids=(_digest("prepared"),),
        trusted_checkpoint=trusted,
    )

    assert adopted["schema_version"] == OPERATOR_TRUSTED_ADOPTION_ATTESTATION
    assert adopted["validation_policy"] == OPERATOR_TRUSTED_VALIDATION_POLICY
    assert adopted["producer_artifact_id"] == artifact.artifact_id
    assert adopted["consumer_request_sha256"] == current_request
    assert adopted["prior_adoption_attestation_content_sha256"] == prior[
        "content_sha256"
    ]
    assert adopted["payload_bytes_reauthenticated"] is False
    assert adopted["fresh_full_byte_validation_achieved"] is False
    assert adopted["global_release_certified"] is False
    assert adopted["operator_trust_explicit"] is True

    reopened = validate_operator_trusted_checkpoint_adoption(
        attestation_path=(
            tmp_path
            / "current_adoptions"
            / f"{artifact.artifact_id}.adoption.json"
        ),
        source=artifact.root,
        prior_attestation_path=prior_path,
        consumer_request_sha256=current_request,
        expected_kind="embedding_cache",
        expected_compatibility_key=artifact.compatibility_key,
        expected_upstream_artifact_ids=(_digest("prepared"),),
    )
    assert reopened == adopted


def test_operator_trusted_adoption_rejects_payload_stat_discontinuity(
    tmp_path: Path,
) -> None:
    artifact, payload, _prior, prior_path = _artifact_and_prior(tmp_path)
    payload.write_bytes(b"x" * len(b"previously authenticated payload bytes"))

    with pytest.raises(ValueError, match="newer than"):
        validate_operator_trusted_portable_artifact(
            source=artifact.root,
            prior_attestation_path=prior_path,
        )


def test_operator_trusted_adoption_rejects_control_or_prior_substitution(
    tmp_path: Path,
) -> None:
    artifact, _payload, prior, prior_path = _artifact_and_prior(tmp_path)
    manifest_path = artifact.manifest_path
    original_manifest = manifest_path.read_text(encoding="utf-8")
    os.chmod(manifest_path, 0o644)
    manifest_path.write_text(original_manifest + " ", encoding="utf-8")
    os.chmod(manifest_path, 0o444)
    with pytest.raises(ValueError, match="controls differ"):
        validate_operator_trusted_portable_artifact(
            source=artifact.root,
            prior_attestation_path=prior_path,
        )

    os.chmod(manifest_path, 0o644)
    manifest_path.write_text(original_manifest, encoding="utf-8")
    os.chmod(manifest_path, 0o444)
    mutated = dict(prior)
    mutated["consumer_request_sha256"] = _digest("substituted consumer")
    os.chmod(prior_path, 0o644)
    prior_path.write_text(
        json.dumps(mutated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(prior_path, 0o444)
    with pytest.raises(ValueError, match="prior full-byte"):
        validate_operator_trusted_portable_artifact(
            source=artifact.root,
            prior_attestation_path=prior_path,
        )


def test_ordinary_adoption_still_requires_a_payload_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, payload, _prior, _prior_path = _artifact_and_prior(tmp_path)
    original_hash = portable_artifacts_module._safe_file_hash_with_identity

    def reject_payload_hash(path: Path, *, label: str):
        if Path(path).resolve() == payload.resolve():
            raise AssertionError("ordinary adoption attempted the required payload hash")
        return original_hash(path, label=label)

    monkeypatch.setattr(
        portable_artifacts_module,
        "_safe_file_hash_with_identity",
        reject_payload_hash,
    )
    with pytest.raises(
        AssertionError,
        match="ordinary adoption attempted the required payload hash",
    ):
        adopt_checkpoint(
            source=artifact.root,
            attestation_root=tmp_path / "ordinary_again",
            consumer_request_sha256=_digest("ordinary consumer"),
        )
