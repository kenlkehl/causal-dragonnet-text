from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

from oci.config import BoWViewConfig
from oci.inference.tfidf_safe_artifacts import (
    load_fitted_topic_context,
    load_named_array_bank,
    write_fitted_topic_context,
    write_named_array_bank,
)
from oci.inference.tfidf_topic_discovery import CrossFittedStack, FittedTopicContext


def _canonical_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _rewrite_index(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    value["content_sha256"] = _canonical_sha(body)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _fitted_context() -> tuple[FittedTopicContext, list[str]]:
    texts = [
        "alpha baseline stable",
        "beta treated response",
        "alpha untreated risk",
        "beta treated stable",
        "alpha response risk",
        "beta untreated stable",
    ]
    treatment = np.asarray([0, 1, 0, 1, 0, 1])
    outcome = np.arange(len(texts), dtype=float)
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)[a-z]+",
        dtype=np.float32,
    ).fit(texts)
    matrix = vectorizer.transform(texts)
    treatment_base = ExtraTreesClassifier(n_estimators=3, random_state=17).fit(
        matrix,
        treatment,
    )
    outcome_base = RandomForestRegressor(n_estimators=3, random_state=19).fit(
        matrix,
        outcome,
    )
    treatment_meta = treatment_base.predict_proba(matrix)[:, 1].reshape(-1, 1)
    outcome_meta = outcome_base.predict(matrix).reshape(-1, 1)
    treatment_stack = LogisticRegression(random_state=23).fit(
        treatment_meta,
        treatment,
    )
    outcome_stack = Ridge(alpha=2.0).fit(outcome_meta, outcome)
    treatment_view = BoWViewConfig(
        name="extra",
        max_features=100,
        min_df=1,
        max_df=1.0,
        ngram_range_min=1,
        ngram_range_max=1,
        bow_model="extratrees",
    )
    outcome_view = BoWViewConfig(
        name="forest",
        max_features=100,
        min_df=1,
        max_df=1.0,
        ngram_range_min=1,
        ngram_range_max=1,
        bow_model="random_forest",
    )
    fitted = FittedTopicContext(
        common_vectorizer=vectorizer,
        treatment_stack=CrossFittedStack(
            views=[treatment_view],
            binary=True,
            base_models=[(vectorizer, treatment_base, 0.5)],
            stack_model=treatment_stack,
            stack_constant=0.5,
            config_hash="treatment-config",
        ),
        outcome_stack=CrossFittedStack(
            views=[outcome_view],
            binary=False,
            base_models=[(vectorizer, outcome_base, 2.5)],
            stack_model=outcome_stack,
            stack_constant=2.5,
            config_hash="outcome-config",
        ),
        topic_banks={},
        config_hash="context-config",
    )
    return fitted, texts


def test_fitted_context_round_trip_is_exact_and_uses_no_pickle_or_npz(tmp_path: Path) -> None:
    fitted, texts = _fitted_context()
    index = write_fitted_topic_context(fitted, tmp_path / "context")
    replay = load_fitted_topic_context(index)

    original_treatment = fitted.treatment_stack.predict(texts)[0]
    replay_treatment = replay.treatment_stack.predict(texts)[0]
    original_outcome = fitted.outcome_stack.predict(texts)[0]
    replay_outcome = replay.outcome_stack.predict(texts)[0]
    assert np.array_equal(original_treatment, replay_treatment)
    assert np.array_equal(original_outcome, replay_outcome)
    assert fitted.common_vectorizer.vocabulary_ == replay.common_vectorizer.vocabulary_
    assert not tuple(tmp_path.rglob("*.joblib"))
    assert not tuple(tmp_path.rglob("*.pkl"))
    assert not tuple(tmp_path.rglob("*.pickle"))
    assert not tuple(tmp_path.rglob("*.npz"))
    arrays = tuple((tmp_path / "context").glob("*.npy"))
    assert arrays
    assert all(np.load(path, allow_pickle=False, mmap_mode="r").dtype.hasobject is False for path in arrays)


def test_safe_context_preserves_a_term_beyond_old_prompt_sized_prefix(tmp_path: Path) -> None:
    fitted, _texts = _fitted_context()
    long_text = ("ordinaryprefix " * 2_000) + "sentinelsuffix"
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)[a-z]+",
        dtype=np.float32,
    ).fit([long_text, "ordinaryprefix control"])
    fitted.common_vectorizer = vectorizer
    index = write_fitted_topic_context(fitted, tmp_path / "long-context")
    replay = load_fitted_topic_context(index)
    assert "sentinelsuffix" in replay.common_vectorizer.vocabulary_
    assert replay.common_vectorizer.transform([long_text]).shape[1] == len(
        replay.common_vectorizer.vocabulary_
    )


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "reordered", "tampered", "symlink", "hardlink", "dtype", "shape"],
)
def test_array_bank_rejects_any_inventory_or_payload_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    index = write_named_array_bank(
        {
            "effect": np.arange(12, dtype=np.float64).reshape(6, 2),
            "treatment": np.arange(6, dtype=np.float32).reshape(6, 1),
        },
        tmp_path / mutation,
        row_count=6,
    )
    root = index.parent
    manifest = json.loads(index.read_text(encoding="utf-8"))
    first = root / manifest["payload_inventory"][0]["relative_path"]
    second = root / manifest["payload_inventory"][1]["relative_path"]

    if mutation == "missing":
        first.unlink()
    elif mutation == "extra":
        np.save(root / "extra.npy", np.zeros((1,), dtype=float), allow_pickle=False)
    elif mutation == "reordered":
        _rewrite_index(index, lambda value: value["payload_inventory"].reverse())
    elif mutation == "tampered":
        first.write_bytes(first.read_bytes() + b"x")
    elif mutation == "symlink":
        first.unlink()
        first.symlink_to(second.name)
    elif mutation == "hardlink":
        first.unlink()
        os.link(second, first)
    elif mutation in {"dtype", "shape"}:
        replacement = (
            np.arange(12, dtype=np.float32).reshape(6, 2)
            if mutation == "dtype"
            else np.arange(12, dtype=np.float64).reshape(3, 4)
        )
        with first.open("wb") as handle:
            np.save(handle, replacement, allow_pickle=False)

        def update_payload_digest(value) -> None:
            entry = value["payload_inventory"][0]
            raw = first.read_bytes()
            entry["size_bytes"] = len(raw)
            entry["sha256"] = hashlib.sha256(raw).hexdigest()

        _rewrite_index(index, update_payload_digest)

    with pytest.raises(ValueError):
        load_named_array_bank(index, expected_row_count=6)


def test_named_array_bank_is_ordered_mmap_safe_and_closed(tmp_path: Path) -> None:
    expected = {
        "effect": np.arange(8, dtype=np.float64).reshape(4, 2),
        "treatment": np.arange(4, dtype=np.float32).reshape(4, 1),
    }
    index = write_named_array_bank(expected, tmp_path / "bank", row_count=4)
    loaded = load_named_array_bank(index, expected_row_count=4)
    assert list(loaded) == ["effect", "treatment"]
    assert all(isinstance(values, np.memmap) for values in loaded.values())
    assert all(values.flags.writeable is False for values in loaded.values())
    for name, values in expected.items():
        assert np.array_equal(loaded[name], values)
