from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from oci.inference.all_evidence_fusion import TFIDF_TOPIC_SOURCE
from oci.inference.all_evidence_fusion_runner import _review_provider_identity
from oci.inference.fold_honest_r_stack import FitRowProvenance
from oci.inference.review_spent_evidence_cache_overlay import (
    AuthenticatedReviewSpentEvidenceCacheOverlay,
    ReviewSpentCacheAuthenticationError,
    authenticate_review_spent_cache_registrations,
)
from oci.inference.review_spent_evidence_provider import (
    ContextFitReviewSpentEvidenceProvider,
    SpentDiscoveryEvidence,
)
import oci.inference.review_spent_evidence_cache_overlay as overlay_module

_FORBIDDEN_KEY = re.compile(
    r"(?:^|_)(?:oracle|true|ground_truth|groundtruth)(?:_|$)", re.IGNORECASE
)


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _tfidf_payload(outer_fold: int, review_round: int) -> dict:
    def topic(phrase: str) -> dict:
        return {"terms": [{"term": phrase, "loading": 0.8}]}

    return {
        "outer_fold": outer_fold,
        "scope": "inner_train",
        "inner_fold": review_round + 1,
        "discovery": {
            "topic_banks": {
                "treatment": {"topics": [topic("baseline care pattern")]},
                "outcome": {"topics": [topic("baseline risk pattern")]},
                "effect": {"topics": [topic("pretreatment marker phrase")]},
            },
            "effect_orphan_ngram_branch": {
                "selected_cluster_ids": ["cluster_001"],
                "selected_clusters": [
                    {
                        "cluster_id": "cluster_001",
                        "terms": [{"term": "unmodeled baseline phrase", "fit_rank": 2}],
                    }
                ],
            },
        },
    }


class _Backend:
    def __init__(self, identity: str = "test_tfidf_backend_v1") -> None:
        self.identity_value = identity
        self.calls = 0

    def identity(self):
        return {"backend": self.identity_value}

    def fit_discovery(
        self,
        *,
        outer_fold,
        review_round,
        exact_spent_row_ids,
        spent_texts,
        spent_treatment,
        spent_outcome,
        work_dir,
    ):
        del spent_texts, spent_treatment, spent_outcome, work_dir
        self.calls += 1
        return SpentDiscoveryEvidence.create(
            source_kind=TFIDF_TOPIC_SOURCE,
            payload=_tfidf_payload(outer_fold, review_round),
            fit_row_provenance=FitRowProvenance(fit_row_ids=frozenset(exact_spent_row_ids)),
        )


def _request(*, review_round: int = 0, texts: tuple[str, ...] | None = None) -> dict:
    return {
        "outer_fold": 1,
        "review_round": review_round,
        "exact_spent_row_ids": (10, 11, 12),
        "exact_sealed_row_ids": (20, 21),
        "spent_texts": texts or ("spent a", "spent b", "spent c"),
        "spent_treatment": np.asarray([0.0, 1.0, 0.0]),
        "spent_outcome": np.asarray([0.0, 0.0, 1.0]),
    }


def _make_source(tmp_path: Path):
    backend = _Backend()
    provider = ContextFitReviewSpentEvidenceProvider(
        backends=(backend,),
        cache_dir=tmp_path / "historical" / "spent-cache",
        required_source_families=(),
    )
    provider.get_spent_evidence_inputs(**_request())
    source_path = next(provider.cache_dir.glob("*.json"))
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source = authenticate_review_spent_cache_registrations([f"{source_path}::{source_sha256}"])[0]
    return source, source_path, source_bytes


def _target_provider(output_root: Path, *, backend_identity: str = "test_tfidf_backend_v1"):
    backend = _Backend(backend_identity)
    provider = ContextFitReviewSpentEvidenceProvider(
        backends=(backend,),
        cache_dir=output_root / "post_extraction_review_spent_evidence_cache",
        required_source_families=(),
    )
    return provider, backend


def _assert_no_forbidden_keys(value) -> None:
    if isinstance(value, dict):
        assert not any(_FORBIDDEN_KEY.search(str(key)) for key in value)
        for child in value.values():
            _assert_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden_keys(child)


def test_exact_binding_hit_materializes_authenticated_snapshot_and_skips_refit(
    tmp_path: Path,
) -> None:
    source, source_path, source_bytes = _make_source(tmp_path)
    # The historical path is no longer authoritative after authentication.
    # Replacing it proves that materialization uses the retained byte snapshot.
    source_path.write_bytes(b"mutated after authentication")
    output_root = tmp_path / "fresh-output"
    provider, backend = _target_provider(output_root)
    overlay = AuthenticatedReviewSpentEvidenceCacheOverlay(
        provider=provider,
        sources=(source,),
        output_root=output_root,
    )

    result = overlay.get_spent_evidence_inputs(**_request())

    assert backend.calls == 0
    assert len(result) == 1
    target = provider.cache_dir / f"{source.cache_key}.json"
    assert target.read_bytes() == source_bytes == source.snapshot
    identity = overlay.identity()
    assert identity["read_only_source_count"] == 1
    assert identity["read_only_sources"][0]["source_path"] == str(source_path.resolve())
    assert identity["read_only_sources"][0]["snapshot_sha256"] == source.snapshot_sha256
    assert identity["delegate_provider_identity_sha256"] == (source.provider_identity_sha256)
    _assert_no_forbidden_keys(identity)
    manifest_binding = _review_provider_identity(overlay, label="review_spent_evidence_provider")
    assert manifest_binding["identity"] == identity


def test_nonmatching_binding_is_a_miss_and_delegates_to_current_backend(
    tmp_path: Path,
) -> None:
    source, _, _ = _make_source(tmp_path)
    output_root = tmp_path / "fresh-output"
    provider, backend = _target_provider(output_root)
    overlay = AuthenticatedReviewSpentEvidenceCacheOverlay(
        provider=provider,
        sources=(source,),
        output_root=output_root,
    )

    overlay.get_spent_evidence_inputs(**_request(review_round=1))

    assert backend.calls == 1
    files = tuple(provider.cache_dir.glob("*.json"))
    assert len(files) == 1
    assert files[0].stem != source.cache_key
    assert overlay.identity()["read_only_sources"][0]["cache_key"] == source.cache_key


def test_source_path_is_read_once_then_only_the_snapshot_is_used(
    tmp_path: Path, monkeypatch
) -> None:
    _, source_path, source_bytes = _make_source(tmp_path)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    real_read_bytes = Path.read_bytes
    source_reads = 0

    def tracking_read_bytes(path: Path):
        nonlocal source_reads
        if path.resolve() == source_path.resolve():
            source_reads += 1
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tracking_read_bytes)
    source = authenticate_review_spent_cache_registrations([f"{source_path}::{source_sha256}"])[0]
    source_path.write_bytes(b"replacement")
    output_root = tmp_path / "fresh-output"
    provider, backend = _target_provider(output_root)
    overlay = AuthenticatedReviewSpentEvidenceCacheOverlay(
        provider=provider, sources=(source,), output_root=output_root
    )
    overlay.get_spent_evidence_inputs(**_request())

    assert source_reads == 1
    assert backend.calls == 0


def _variant_path(
    tmp_path: Path,
    source_bytes: bytes,
    mutate,
    *,
    recompute_content: bool,
    filename_key: str | None = None,
) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = json.loads(source_bytes)
    mutate(raw)
    if recompute_content:
        raw["content_sha256"] = _sha256_json(
            {key: value for key, value in raw.items() if key != "content_sha256"}
        )
    snapshot = _canonical_json(raw).encode("utf-8")
    path = tmp_path / f"{filename_key or raw['cache_key']}.json"
    path.write_bytes(snapshot)
    return path, hashlib.sha256(snapshot).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "recompute_content", "message"),
    [
        (
            lambda raw: raw["results"][0]["payload"].update({"changed": "value"}),
            False,
            "content hash mismatch",
        ),
        (
            lambda raw: raw.update({"unexpected": "field"}),
            True,
            "unsupported closed schema",
        ),
        (
            lambda raw: raw["binding"].update({"review_round": 7}),
            True,
            "binding hash does not equal",
        ),
        (
            lambda raw: raw["results"][0]["payload"].update({"oracle_score": 0.9}),
            True,
            "forbidden field",
        ),
    ],
)
def test_closed_cache_authentication_fails_on_tamper(
    tmp_path: Path, mutation, recompute_content: bool, message: str
) -> None:
    _, _, source_bytes = _make_source(tmp_path)
    path, digest = _variant_path(
        tmp_path / "variant",
        source_bytes,
        mutation,
        recompute_content=recompute_content,
    )
    with pytest.raises(ReviewSpentCacheAuthenticationError, match=message):
        authenticate_review_spent_cache_registrations([f"{path}::{digest}"])


def test_registration_requires_external_hash_and_matching_filename_and_bytes(
    tmp_path: Path,
) -> None:
    _, _, source_bytes = _make_source(tmp_path)
    raw = json.loads(source_bytes)
    cache_key = raw["cache_key"]
    wrong_name = tmp_path / "wrong-name" / f"{'0' * 64}.json"
    wrong_name.parent.mkdir()
    wrong_name.write_bytes(source_bytes)
    digest = hashlib.sha256(source_bytes).hexdigest()

    with pytest.raises(ReviewSpentCacheAuthenticationError, match="PATH::SHA256"):
        authenticate_review_spent_cache_registrations([str(wrong_name)])
    with pytest.raises(ReviewSpentCacheAuthenticationError, match="SHA-256 mismatch"):
        authenticate_review_spent_cache_registrations([f"{wrong_name}::{'1' * 64}"])
    with pytest.raises(ReviewSpentCacheAuthenticationError, match="filename"):
        authenticate_review_spent_cache_registrations([f"{wrong_name}::{digest}"])

    correct = tmp_path / "duplicate" / f"{cache_key}.json"
    correct.parent.mkdir()
    correct.write_bytes(source_bytes)
    with pytest.raises(ReviewSpentCacheAuthenticationError, match="duplicate"):
        authenticate_review_spent_cache_registrations(
            [f"{correct}::{digest}", f"{correct}::{digest}"]
        )


def test_overlay_rejects_provider_identity_code_and_nonfresh_output(
    tmp_path: Path, monkeypatch
) -> None:
    source, _, _ = _make_source(tmp_path)
    output_root = tmp_path / "fresh-output"
    wrong_provider, _ = _target_provider(output_root, backend_identity="different_backend_v1")
    with pytest.raises(ReviewSpentCacheAuthenticationError, match="provider identity"):
        AuthenticatedReviewSpentEvidenceCacheOverlay(
            provider=wrong_provider,
            sources=(source,),
            output_root=output_root,
        )

    code_output = tmp_path / "code-output"
    code_provider, _ = _target_provider(code_output)
    real_sha256_path = overlay_module._sha256_path

    def wrong_provider_code(path: Path) -> str:
        if path.name == "review_spent_evidence_provider.py":
            return "0" * 64
        return real_sha256_path(path)

    monkeypatch.setattr(overlay_module, "_sha256_path", wrong_provider_code)
    with pytest.raises(ReviewSpentCacheAuthenticationError, match="current code"):
        AuthenticatedReviewSpentEvidenceCacheOverlay(
            provider=code_provider,
            sources=(source,),
            output_root=code_output,
        )
    monkeypatch.setattr(overlay_module, "_sha256_path", real_sha256_path)

    dirty_output = tmp_path / "dirty-output"
    dirty_provider, _ = _target_provider(dirty_output)
    (dirty_provider.cache_dir / "preexisting.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        AuthenticatedReviewSpentEvidenceCacheOverlay(
            provider=dirty_provider,
            sources=(source,),
            output_root=dirty_output,
        )

    outside_provider, _ = _target_provider(tmp_path / "outside-parent")
    with pytest.raises(ValueError, match="direct child"):
        AuthenticatedReviewSpentEvidenceCacheOverlay(
            provider=outside_provider,
            sources=(source,),
            output_root=tmp_path / "different-output",
        )
