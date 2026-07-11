"""Pytest configuration for the agent-app test suite.

A few test modules instantiate live Databricks clients / Lakebase config at
import time — they are integration tests, not hermetic unit tests. When the
workspace is not configured (e.g. CI without secrets) importing them fails at
collection with errors like "SQL_WAREHOUSE_ID cannot be empty" or "account_id is
required to ...", which aborts the entire run.

To keep the hermetic unit tests gating every PR, skip only those integration
modules when the Databricks environment is absent. They run normally whenever
DATABRICKS_HOST / DATABRICKS_TOKEN / SQL_WAREHOUSE_ID are set (local dev or a CI
job with secrets configured).
"""

import os
import warnings

# Integration modules that require a configured Databricks workspace at import.
_REQUIRES_DATABRICKS = (
    "unit/test_agent_checkpointer_retry.py",
    "unit/test_sequential_output_fixes.py",
    "unit/test_tabular_ltm.py",
    "unit/test_agent_stream_terminal_short_circuit.py",
)


def _databricks_configured() -> bool:
    return bool(
        os.environ.get("DATABRICKS_HOST")
        and os.environ.get("DATABRICKS_TOKEN")
        and os.environ.get("SQL_WAREHOUSE_ID")
    )


collect_ignore: list[str] = []
if not _databricks_configured():
    collect_ignore = list(_REQUIRES_DATABRICKS)
    warnings.warn(
        "Databricks env not configured (DATABRICKS_HOST/DATABRICKS_TOKEN/"
        "SQL_WAREHOUSE_ID) — skipping integration test modules: "
        + ", ".join(_REQUIRES_DATABRICKS),
        stacklevel=2,
    )
