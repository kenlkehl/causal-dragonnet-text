from __future__ import annotations

import json
import subprocess
import sys


def _loaded_after(statement: str) -> set[str]:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import json, sys; {statement}; print(json.dumps(sorted(sys.modules)))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(json.loads(completed.stdout.splitlines()[-1]))


def test_importing_package_does_not_initialize_model_runtimes():
    loaded = _loaded_after("import oci")

    assert "torch" not in loaded
    assert "oci.experiments.runner" not in loaded


def test_active_text_stage_does_not_import_retired_orchestrators():
    loaded = _loaded_after("import oci.inference.multi_model_forest_stage1")

    assert "oci.inference.multi_model_agentic_forest" not in loaded
    assert "oci.inference.agentic_attention_variable_forest" not in loaded
    assert "oci.inference.agentic_explicit_feature_forest" not in loaded
