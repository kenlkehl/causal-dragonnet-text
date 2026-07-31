#!/usr/bin/env python3
"""Materialize pinned Hugging Face snapshots as real, local-only trees."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Iterable


MATERIALIZATION_SCHEMA = "production_model_materialization_v1"
MARKER_NAME = "materialization.json"
TOKENIZER_PATTERNS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "*.model",
    "*.txt",
)
REQUIRED_FILES = {
    "embedding": (
        "config.json",
        "modules.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ),
    "htr": ("config.json", "pytorch_model.bin", "vocab.txt"),
    "tokenizer": ("config.json", "tokenizer.json", "tokenizer_config.json"),
    "stage2_vllm": (
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "hf_quant_config.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
}


def _closed_marker(*, repo_id: str, revision: str, resolved_revision: str, kind: str) -> dict:
    return {
        "schema_version": MATERIALIZATION_SCHEMA,
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "kind": kind,
        "local_files_only_after_publication": True,
        "symlinks_allowed": False,
        "atomic_publication": True,
    }


def _validate_tree(
    target: Path,
    *,
    repo_id: str,
    revision: str,
    kind: str,
) -> None:
    if target.is_symlink() or not target.is_dir():
        raise ValueError(f"model target is not one real directory: {target}")
    marker_path = target / MARKER_NAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError(f"model materialization marker is absent: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        not isinstance(marker, dict)
        or set(marker) != set(
            _closed_marker(
                repo_id=repo_id,
                revision=revision,
                resolved_revision=str(marker.get("resolved_revision", "")),
                kind=kind,
            )
        )
        or marker.get("schema_version") != MATERIALIZATION_SCHEMA
        or marker.get("repo_id") != repo_id
        or marker.get("requested_revision") != revision
        or marker.get("kind") != kind
        or not isinstance(marker.get("resolved_revision"), str)
        or not marker["resolved_revision"]
        or marker.get("local_files_only_after_publication") is not True
        or marker.get("symlinks_allowed") is not False
        or marker.get("atomic_publication") is not True
    ):
        raise ValueError(f"model materialization marker is incompatible: {marker_path}")
    for path in target.rglob("*"):
        state = os.lstat(path)
        if stat.S_ISLNK(state.st_mode):
            raise ValueError(f"materialized model contains a symlink: {path}")
        if not stat.S_ISDIR(state.st_mode) and not stat.S_ISREG(state.st_mode):
            raise ValueError(f"materialized model contains a special file: {path}")
    missing = [name for name in REQUIRED_FILES[kind] if not (target / name).is_file()]
    if missing:
        raise ValueError(f"materialized {kind} model is missing files: {missing}")
    if kind == "stage2_vllm":
        try:
            config = json.loads((target / "config.json").read_text(encoding="utf-8"))
            legacy_quant = json.loads(
                (target / "hf_quant_config.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Stage 2 vLLM model metadata is unreadable") from exc
        quantization = config.get("quantization_config")
        text_config = config.get("text_config")
        if (
            config.get("architectures") != ["Gemma4ForConditionalGeneration"]
            or config.get("model_type") != "gemma4"
            or not isinstance(text_config, dict)
            or text_config.get("max_position_embeddings") != 262_144
            or not isinstance(quantization, dict)
            or quantization.get("quant_method") != "modelopt"
            or quantization.get("quant_algo") != "NVFP4"
            or legacy_quant.get("producer", {}).get("name") != "modelopt"
            or legacy_quant.get("quantization", {}).get("quant_algo") != "NVFP4"
        ):
            raise ValueError(
                "Stage 2 model is not the expected 256K Gemma 4 ModelOpt NVFP4 checkpoint"
            )


def _copy_snapshot(source: Path, target: Path) -> None:
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        destination = target / relative
        if source_path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif source_path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination, follow_symlinks=True)
            destination.chmod(0o600)


def materialize(
    *,
    repo_id: str,
    revision: str,
    kind: str,
    target: Path,
) -> None:
    if not target.is_absolute():
        raise ValueError("model materialization target must be absolute")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _validate_tree(
            target,
            repo_id=repo_id,
            revision=revision,
            kind=kind,
        )
        print(f"[models] reused {kind}: {target}")
        return

    from huggingface_hub import snapshot_download

    source = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=(TOKENIZER_PATTERNS if kind == "tokenizer" else None),
        )
    ).resolve(strict=True)
    resolved_revision = source.name
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.materializing-", dir=target.parent)
    )
    try:
        _copy_snapshot(source, temporary)
        marker = _closed_marker(
            repo_id=repo_id,
            revision=revision,
            resolved_revision=resolved_revision,
            kind=kind,
        )
        marker_path = temporary / MARKER_NAME
        marker_path.write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        marker_path.chmod(0o600)
        _validate_tree(
            temporary,
            repo_id=repo_id,
            revision=revision,
            kind=kind,
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"model target appeared during publication: {target}")
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _validate_tree(target, repo_id=repo_id, revision=revision, kind=kind)
    print(f"[models] materialized {kind}: {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--kind", choices=tuple(REQUIRED_FILES), required=True)
    parser.add_argument("--target", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    materialize(
        repo_id=args.repo_id,
        revision=args.revision,
        kind=args.kind,
        target=args.target,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
