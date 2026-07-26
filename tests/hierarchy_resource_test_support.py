"""Explicit hierarchy resource policies used by test fixtures."""

from oci.inference.first_untouched_gate_direct_numerical_preparation import (
    FirstUntouchedGatePreparationBounds,
)
from oci.inference.hierarchical_discovery_job_cache import (
    HierarchicalDiscoveryJobCacheConfig,
)


HIERARCHY_JOB_CACHE_CONFIG = HierarchicalDiscoveryJobCacheConfig(
    max_entry_bytes=32_000_000,
)

FIRST_UNTOUCHED_GATE_BOUNDS = FirstUntouchedGatePreparationBounds(
    max_initial_spent_rows=1_000_000,
    max_first_gate_rows=1_000_000,
    max_total_text_utf8_bytes=1_073_741_824,
    max_catalog_atoms=100_000,
    max_source_manifest_bytes=16_777_216,
    max_direct_numerical_signals=16_384,
    max_single_matrix_file_bytes=1_073_741_824,
    max_total_matrix_file_bytes=4_294_967_296,
)

