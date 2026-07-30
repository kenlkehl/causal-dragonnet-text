"""Authenticate a relocation after workflow scratch-to-durable publication.

The relocation producer deliberately seals exact absolute locators.  The
generic workflow phase publisher subsequently moves every authenticated byte
from its scratch attempt into the durable phase tree without rewriting nested
JSON.  This validator preserves the producer's implementation hash and
scientific checks while proving the one allowed locator transformation.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

from . import production_embedding_cache_relocation as relocation


def _canonical_historical_root(value: Path | str) -> Path:
    root = Path(value)
    if (
        not root.is_absolute()
        or Path(os.path.normpath(str(root))) != root
    ):
        raise ValueError(
            "relocation prepublication root must be one absolute lexically "
            "canonical historical path"
        )
    return root


def validate_phase_publication_locator_mapping(
    *,
    terminal: Mapping[str, Any],
    attestation: Mapping[str, Any],
    durable_root: Path | str,
    prepublication_root: Path | str,
) -> None:
    """Validate only the exact scratch-to-durable locator mapping."""

    durable = Path(durable_root).resolve(strict=True)
    historical = _canonical_historical_root(prepublication_root)
    if durable == historical:
        raise ValueError(
            "phase-publication mapping requires distinct historical and "
            "durable roots"
        )
    terminal_attestation = terminal.get("attestation")
    destination = attestation.get("destination")
    cache_build_identity = (
        destination.get("cache_build_identity")
        if isinstance(destination, Mapping)
        else None
    )
    prepared = (
        destination.get("prepared_cohort")
        if isinstance(destination, Mapping)
        else None
    )
    if (
        terminal.get("root") != str(historical)
        or not isinstance(terminal_attestation, Mapping)
        or terminal_attestation.get("path")
        != str(
            historical
            / relocation.RELOCATION_ATTESTATION_NAME
        )
        or not isinstance(destination, Mapping)
        or destination.get("root") != str(historical)
        or destination.get("cache_dir")
        != str(
            historical
            / relocation.RELOCATED_CACHE_RELATIVE_PATH
        )
        or not isinstance(prepared, Mapping)
        or prepared.get("path")
        != str(
            historical
            / relocation.RELOCATED_PREPARED_RELATIVE_PATH
        )
        or not isinstance(cache_build_identity, Mapping)
        or cache_build_identity.get("cache_path")
        != str(
            historical
            / relocation.RELOCATED_CACHE_RELATIVE_PATH
        )
    ):
        raise ValueError(
            "relocation payload does not bind its exact prepublication root"
        )


def validate_phase_published_production_embedding_cache_relocation(
    options: relocation.ProductionEmbeddingCacheRelocationOptions,
    *,
    prepublication_root: Path | str,
) -> relocation.AuthenticatedProductionEmbeddingCacheRelocation:
    """Freshly validate a byte-preserving workflow publication of relocation."""

    columns, configuration = relocation._validated_options(
        options,
        target_must_exist=True,
    )
    target = relocation._real_canonical_directory(
        options.target_dir,
        label="published relocation target",
    )
    historical = _canonical_historical_root(prepublication_root)
    target_signature = relocation._directory_signature(
        target,
        label="published relocation target",
    )
    expected_root_entries = {
        relocation.RELOCATED_PREPARED_RELATIVE_PATH.parent.name,
        relocation.RELOCATED_CACHE_RELATIVE_PATH.name,
        relocation.RELOCATION_ATTESTATION_NAME,
        relocation.RELOCATION_TERMINAL_MANIFEST_NAME,
    }
    if {path.name for path in target.iterdir()} != expected_root_entries:
        raise ValueError(
            "published relocation root contains missing or unregistered entries"
        )
    prepared_root = relocation._real_canonical_directory(
        target
        / relocation.RELOCATED_PREPARED_RELATIVE_PATH.parent,
        label="published relocated prepared root",
    )
    prepared_root_signature = relocation._directory_signature(
        prepared_root,
        label="published relocated prepared root",
    )
    if {path.name for path in prepared_root.iterdir()} != {
        relocation.RELOCATED_PREPARED_RELATIVE_PATH.name
    }:
        raise ValueError("published relocated prepared root is not closed")
    cache_root = relocation._real_canonical_directory(
        target / relocation.RELOCATED_CACHE_RELATIVE_PATH,
        label="published relocated cache root",
    )
    cache_root_signature = relocation._directory_signature(
        cache_root,
        label="published relocated cache root",
    )
    actual_cache_registrations = relocation._cache_registrations(
        cache_root,
        require_single_link=True,
    )
    relocation._require_single_link_regular_file(
        target / relocation.RELOCATION_ATTESTATION_NAME,
        label="published relocation attestation",
    )
    relocation._require_single_link_regular_file(
        target / relocation.RELOCATION_TERMINAL_MANIFEST_NAME,
        label="published relocation terminal manifest",
    )

    attestation, attestation_snapshot = relocation._read_json_snapshot(
        target / relocation.RELOCATION_ATTESTATION_NAME,
        label="published relocation attestation",
    )
    relocation._validate_attestation_shape(attestation)
    terminal, terminal_snapshot = relocation._read_json_snapshot(
        target / relocation.RELOCATION_TERMINAL_MANIFEST_NAME,
        label="published relocation terminal manifest",
    )
    if set(terminal) != set(relocation._TERMINAL_FIELDS):
        raise ValueError(
            "published relocation terminal manifest is not a closed schema"
        )
    terminal_body = {
        key: copy.deepcopy(value)
        for key, value in terminal.items()
        if key != "content_sha256"
    }
    if (
        terminal.get("schema_version")
        != relocation.PRODUCTION_EMBEDDING_CACHE_RELOCATION_TERMINAL_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("relocator_version")
        != relocation.PRODUCTION_EMBEDDING_CACHE_RELOCATOR_VERSION
        or terminal.get("relocator_code_sha256")
        != relocation._relocator_code_sha256()
        or terminal.get("authenticated_tree_code_sha256")
        != relocation._authenticated_tree_code_sha256()
        or relocation._require_sha256(
            terminal.get("content_sha256"),
            label="published terminal.content_sha256",
        )
        != relocation._sha256_json(terminal_body)
    ):
        raise ValueError(
            "published relocation terminal manifest identity is invalid"
        )
    validate_phase_publication_locator_mapping(
        terminal=terminal,
        attestation=attestation,
        durable_root=target,
        prepublication_root=historical,
    )
    terminal_attestation = terminal["attestation"]
    if (
        set(terminal_attestation)
        != {"path", "sha256", "size_bytes", "content_sha256"}
        or relocation._require_registration(
            {
                "sha256": terminal_attestation.get("sha256"),
                "size_bytes": terminal_attestation.get("size_bytes"),
            },
            label="published terminal.attestation",
        )
        != attestation_snapshot.registration()
        or terminal_attestation.get("content_sha256")
        != attestation.get("content_sha256")
    ):
        raise ValueError(
            "published terminal does not bind the relocation attestation"
        )
    relative_artifacts = (
        relocation.RELOCATED_PREPARED_RELATIVE_PATH,
        *(
            relocation.RELOCATED_CACHE_RELATIVE_PATH / name
            for name in relocation._CACHE_FILE_NAMES
        ),
        Path(relocation.RELOCATION_ATTESTATION_NAME),
    )
    actual_artifacts = relocation._registrations_for_relative_paths(
        target,
        relative_artifacts,
        require_single_link=True,
    )
    if terminal.get("artifacts") != actual_artifacts:
        raise ValueError(
            "published relocation artifacts differ from terminal manifest"
        )

    validated = relocation._validate_inputs(
        options,
        columns,
        configuration,
    )
    destination = attestation["destination"]
    source = attestation["source"]
    fresh = attestation["fresh_preparation"]
    expected_source_prepared = {
        "path": str(validated["source_prepared"]),
        **validated["source_prepared_snapshot"].registration(),
    }
    expected_source_manifest = {
        "path": str(validated["source_manifest_path"]),
        **validated["source_manifest_snapshot"].registration(),
    }
    expected_fresh_prepared = {
        "path": str(validated["fresh_prepared"]),
        **validated["fresh_prepared_snapshot"].registration(),
    }
    expected_fresh_manifest = {
        "path": str(validated["fresh_manifest_path"]),
        **validated["fresh_manifest_snapshot"].registration(),
    }
    if (
        source.get("cache_dir") != str(validated["source_cache"])
        or source.get("cache_build_identity")
        != validated["source_cache_identity"]
        or source.get("prepared_cohort") != expected_source_prepared
        or source.get("preparation_manifest") != expected_source_manifest
        or source.get("preparation_content_sha256")
        != validated["source_manifest"]["content_sha256"]
        or source.get("prepared_projection_sha256")
        != validated["source_projection"]
        or source.get("local_model_path")
        != str(validated["model_path"])
        or source.get("local_model_tree_sha256")
        != validated["source_cache_identity"][
            "local_model_tree_sha256"
        ]
        or fresh.get("prepared_cohort") != expected_fresh_prepared
        or fresh.get("preparation_manifest") != expected_fresh_manifest
        or fresh.get("preparation_content_sha256")
        != validated["fresh_manifest"]["content_sha256"]
        or fresh.get("prepared_projection_sha256")
        != validated["fresh_projection"]
    ):
        raise ValueError(
            "published relocation attestation differs from authenticated inputs"
        )

    copied_prepared_path, copied_prepared_snapshot = (
        relocation._stable_file_snapshot(
            target / relocation.RELOCATED_PREPARED_RELATIVE_PATH,
            label="published relocated prepared cohort",
        )
    )
    relocation._require_distinct_file_objects(
        validated["source_prepared"],
        copied_prepared_path,
        label="published relocated prepared cohort",
    )
    for name in relocation._CACHE_FILE_NAMES:
        relocation._require_distinct_file_objects(
            validated["source_cache"] / name,
            cache_root / name,
            label=f"published relocated cache {name}",
        )
    try:
        copied_frame = relocation.pd.read_parquet(
            copied_prepared_path,
            columns=list(validated["source_frame"].columns),
        )
        relocation.assert_frame_equal(
            validated["source_frame"],
            copied_frame,
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
            check_frame_type=True,
            check_names=True,
            check_exact=True,
            check_like=False,
        )
    except Exception as exc:
        raise ValueError(
            "published relocated cohort differs from authenticated source"
        ) from exc
    copied_projection = relocation._ordered_projection_sha256(
        copied_frame
    )
    if (
        copied_prepared_snapshot.registration()
        != validated["source_prepared_snapshot"].registration()
        or copied_projection != validated["source_projection"]
    ):
        raise ValueError(
            "published relocated cohort bytes or rows changed"
        )

    relocation._authenticate_expected_local_model(
        local_model_path=validated["model_path"],
        expected_model_provenance=validated["model_provenance"],
        expected_workflow_inventory=validated[
            "model_workflow_inventory"
        ],
    )
    destination_cache_identity = (
        relocation.validate_published_production_embedding_cache(
            cache_dir=cache_root,
            dataset_path=validated["source_prepared"],
            text_column=options.text_column,
            sentence_model_name=options.sentence_model_name,
            chunk_configuration=configuration,
            expected_local_model_path=None,
        )
    )
    relocation._authenticate_local_model_against_builder_cache(
        local_model_path=validated["model_path"],
        cache_root=cache_root,
        cache_identity=destination_cache_identity,
        expected_model_provenance=validated["model_provenance"],
        expected_workflow_inventory=validated[
            "model_workflow_inventory"
        ],
    )
    historical_cache_identity = copy.deepcopy(
        dict(destination_cache_identity)
    )
    historical_cache_identity["cache_path"] = str(
        historical / relocation.RELOCATED_CACHE_RELATIVE_PATH
    )
    if (
        actual_cache_registrations
        != validated["source_cache_registrations"]
        or relocation._without_cache_path(destination_cache_identity)
        != relocation._without_cache_path(
            validated["source_cache_identity"]
        )
        or destination.get("root") != str(historical)
        or destination.get("prepared_cohort")
        != {
            "path": str(
                historical
                / relocation.RELOCATED_PREPARED_RELATIVE_PATH
            ),
            **copied_prepared_snapshot.registration(),
        }
        or destination.get("prepared_projection_sha256")
        != copied_projection
        or destination.get("cache_dir")
        != str(
            historical
            / relocation.RELOCATED_CACHE_RELATIVE_PATH
        )
        or destination.get("cache_files")
        != actual_cache_registrations
        or destination.get("cache_build_identity")
        != historical_cache_identity
    ):
        raise ValueError(
            "published relocated cache differs from source or attestation"
        )

    result_identity = {
        "schema_version": (
            relocation.PRODUCTION_EMBEDDING_CACHE_RELOCATION_RESULT_SCHEMA
        ),
        "relocator_version": (
            relocation.PRODUCTION_EMBEDDING_CACHE_RELOCATOR_VERSION
        ),
        "relocator_code_sha256": relocation._relocator_code_sha256(),
        "authenticated_tree_code_sha256": (
            relocation._authenticated_tree_code_sha256()
        ),
        "root": str(target),
        "cache_dir": str(cache_root),
        "prepared_cohort_path": str(copied_prepared_path),
        "attestation_path": str(
            target / relocation.RELOCATION_ATTESTATION_NAME
        ),
        "terminal_manifest_path": str(
            target / relocation.RELOCATION_TERMINAL_MANIFEST_NAME
        ),
        "row_count": len(copied_frame),
        "prepared_projection_sha256": copied_projection,
        "source_cache_identity_sha256": relocation._sha256_json(
            validated["source_cache_identity"]
        ),
        "cache_build_identity": copy.deepcopy(
            dict(destination_cache_identity)
        ),
        "attestation_sha256": attestation_snapshot.sha256,
        "terminal_manifest_sha256": terminal_snapshot.sha256,
    }
    if (
        relocation._directory_signature(
            target,
            label="published relocation target",
        )
        != target_signature
        or relocation._directory_signature(
            prepared_root,
            label="published relocated prepared root",
        )
        != prepared_root_signature
        or relocation._directory_signature(
            cache_root,
            label="published relocated cache root",
        )
        != cache_root_signature
        or relocation._cache_registrations(
            cache_root,
            require_single_link=True,
        )
        != actual_cache_registrations
        or relocation._registrations_for_relative_paths(
            target,
            relative_artifacts,
            require_single_link=True,
        )
        != actual_artifacts
        or relocation._stable_file_snapshot(
            target / relocation.RELOCATION_TERMINAL_MANIFEST_NAME,
            label="published relocation terminal manifest",
        )[1]
        != terminal_snapshot
    ):
        raise RuntimeError(
            "published relocation artifacts changed during validation"
        )
    return relocation.AuthenticatedProductionEmbeddingCacheRelocation(
        root=target,
        cache_dir=cache_root,
        prepared_cohort_path=copied_prepared_path,
        attestation_path=(
            target / relocation.RELOCATION_ATTESTATION_NAME
        ),
        terminal_manifest_path=(
            target / relocation.RELOCATION_TERMINAL_MANIFEST_NAME
        ),
        cache_build_identity=destination_cache_identity,
        _identity=result_identity,
    )


__all__ = [
    "validate_phase_publication_locator_mapping",
    "validate_phase_published_production_embedding_cache_relocation",
]
