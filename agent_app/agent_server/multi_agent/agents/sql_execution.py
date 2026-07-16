"""
SQL Execution Agent Node

This module provides the SQL execution node for the multi-agent system.
It wraps the SQLExecutionAgent class and executes SQL queries using Databricks SQL Warehouse.

Supports:
- Parallel execution with retry on failure
- Sequential execution (one query at a time, feeding results back to synthesis)
"""

import json as _json
from typing import Dict, Any, List
from langgraph.config import get_stream_writer
from langchain_core.messages import SystemMessage

from ..core.state import AgentState
from ..core.config import get_config
from .sql_execution_agent import SQLExecutionAgent


def _build_retry_feedback(
    successes: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    retry_count: int = 1,
    retry_max: int = 1,
) -> str:
    """Build structured feedback string for the synthesis agent on retry."""
    parts = [f"## Retry Context (attempt {retry_count + 1} of {retry_max + 1})\n"]

    if successes:
        parts.append("### Successfully executed queries (DO NOT regenerate these):")
        for r in successes:
            qn = r.get("query_number", "?")
            rows = r.get("row_count", 0)
            cols = r.get("columns", [])
            sql = r.get("sql", "")
            sample = r.get("result", [])[:5]
            sample_json = _json.dumps(sample, default=str)
            if len(sample_json) > 800:
                sample_json = sample_json[:800] + "..."
            parts.append(
                f"Query {qn}: {rows} rows returned\n"
                f"SQL: {sql}\n"
                f"Columns: {', '.join(cols)}\n"
                f"Sample: {sample_json}\n"
            )

    if failures:
        parts.append("### Failed queries (regenerate these):")
        for r in failures:
            qn = r.get("query_number", "?")
            sql = r.get("sql", "")
            error = r.get("error", "Unknown error")
            parts.append(f"Query {qn}: FAILED\nSQL: {sql}\nError: {error}\n")

    parts.append(
        "### Instructions:\n"
        "- Only regenerate the FAILED queries. The successful queries will be kept.\n"
        "- Use the successful query results above as context (column names, sample data).\n"
        "- Fix the error and generate corrected SQL.\n"
        "- Each regenerated query must be self-contained (include its own subqueries/CTEs)."
    )
    return "\n".join(parts)


def _build_sequential_feedback(
    preserved: List[Dict[str, Any]],
    step: int,
    total: int,
) -> str:
    """Build structured feedback for sequential continuation."""
    from ..utils.executed_result_literals import (
        build_executed_literal_package,
        format_executed_literals_block,
    )

    parts = [f"## Sequential Step {step + 1} of {total}\n"]
    if preserved:
        parts.append("### Previous results:")
        for r in preserved:
            qn = r.get("query_number", "?")
            status = r.get("status", "success")
            if status == "skipped":
                parts.append(
                    f"Query {qn}: skipped\n"
                    f"Reason: {r.get('skip_reason', 'already covered')}\n"
                )
                continue
            if status == "failed":
                parts.append(
                    f"Query {qn}: failed\n"
                    f"SQL: {r.get('sql', '')}\n"
                    f"Error: {r.get('error', 'unknown')}\n"
                )
                continue
            rows = r.get("row_count", 0)
            cols = r.get("columns", [])
            sql = r.get("sql", "")
            sample = r.get("result", [])[:10]
            sample_json = _json.dumps(sample, default=str)
            if len(sample_json) > 1200:
                sample_json = sample_json[:1200] + "..."
            parts.append(
                f"Query {qn}: {rows} rows returned\n"
                f"SQL: {sql}\n"
                f"Columns: {', '.join(cols)}\n"
                f"Data: {sample_json}\n"
            )

        literal_package = build_executed_literal_package(preserved)
        literal_block = format_executed_literals_block(literal_package)
        if literal_block:
            parts.append("### Concrete literals for the next Genie / SQL question:")
            parts.append(literal_block)
            parts.append(
                "\n### Instructions for next step:\n"
                "- Embed these concrete literals in the next Genie question or SQL filter "
                "(e.g. IN lists / exact codes).\n"
                "- Do NOT ask Genie to rediscover the set already returned above "
                "(for example, do not re-ask for 'top N' if those IDs are already listed)."
            )
    return "\n".join(parts)


def _filter_out_already_succeeded(
    sql_queries: List[str],
    preserved: List[Dict[str, Any]],
) -> List[str]:
    """Skip queries whose leading comment label matches an already-successful result."""
    if not preserved:
        return sql_queries
    succeeded_labels = set()
    for r in preserved:
        sql = r.get("sql", "")
        first_line = sql.strip().split("\n")[0] if sql else ""
        if first_line.startswith("--"):
            succeeded_labels.add(first_line.strip().lower())
    filtered = []
    for q in sql_queries:
        first_line = q.strip().split("\n")[0] if q else ""
        if first_line.startswith("--") and first_line.strip().lower() in succeeded_labels:
            print(f"⏭ Skipping already-succeeded query: {first_line.strip()}")
            continue
        filtered.append(q)
    return filtered


def _fallback_query_label(state: AgentState, query_index: int) -> str | None:
    """Best-effort label fallback when synthesis did not return a label."""
    sub_questions = state.get("sub_questions") or []
    if 0 <= query_index < len(sub_questions):
        return sub_questions[query_index]
    return None


def _infer_row_grain_hint(label: str | None, columns: List[str]) -> str | None:
    """Infer when rows likely repeat a higher-level entity across detail rows."""
    lower_label = (label or "").lower()
    lower_columns = {column.lower() for column in columns}

    if "diagnosis_code" in lower_columns or "comorbidit" in lower_label:
        return (
            "Rows are diagnosis-level detail. Patient-level metrics may repeat across "
            "multiple diagnosis rows, so repeated totals should not be summed."
        )

    if (
        {"procedure_code", "cpt_code", "hcpcs_code"} & lower_columns
        or "procedure" in lower_label
    ):
        return (
            "Rows may be procedure-level detail. Higher-level patient or claim totals "
            "may repeat across multiple procedure rows."
        )

    if {"benefit_type", "pay_type"} & lower_columns or "coverage" in lower_label:
        return (
            "Rows are coverage-level detail. Patient-level metrics may repeat across "
            "multiple benefit or pay-type rows."
        )

    if {"diagnosis_code", "benefit_type", "pay_type"} & lower_columns and "patient_id" in lower_columns:
        return "Rows repeat patient-level values across a more detailed grain."

    return None


def _attach_query_metadata(
    state: AgentState,
    result: Dict[str, Any],
    *,
    query_index: int,
    sql_query: str,
    label: str | None = None,
) -> Dict[str, Any]:
    """Attach stable per-query metadata to an execution result."""
    metadata_label = label or _fallback_query_label(state, query_index)
    enriched = dict(result)
    enriched["status"] = "success" if enriched.get("success") else "failed"
    enriched["query_number"] = query_index + 1
    enriched["sql"] = enriched.get("sql") or sql_query
    if metadata_label:
        enriched["query_label"] = metadata_label
    explanation = state.get("sql_synthesis_explanation")
    if explanation:
        enriched["sql_explanation"] = explanation
    row_grain_hint = _infer_row_grain_hint(metadata_label, enriched.get("columns", []) or [])
    if row_grain_hint:
        enriched["row_grain_hint"] = row_grain_hint
    return enriched


def _build_skipped_artifact(
    state: AgentState,
    *,
    query_index: int,
    reason: str,
) -> Dict[str, Any]:
    """Create an explicit placeholder artifact for skipped sequential steps."""
    label = _fallback_query_label(state, query_index)
    artifact: Dict[str, Any] = {
        "status": "skipped",
        "success": False,
        "query_number": query_index + 1,
        "query_label": label,
        "sql": "",
        "columns": [],
        "result": [],
        "row_count": 0,
        "skip_reason": reason,
        "sql_explanation": reason,
    }
    row_grain_hint = _infer_row_grain_hint(label, [])
    if row_grain_hint:
        artifact["row_grain_hint"] = row_grain_hint
    return artifact


def _append_remaining_skipped_artifacts(
    state: AgentState,
    preserved: List[Dict[str, Any]],
    *,
    start_step: int,
    reason: str,
) -> List[Dict[str, Any]]:
    """Pad sequential results so every planned sub-question has an artifact."""
    total = state.get("total_sub_questions", 0) or 0
    if start_step >= total:
        return preserved

    artifacts = list(preserved)
    for query_index in range(start_step, total):
        artifacts.append(_build_skipped_artifact(state, query_index=query_index, reason=reason))
    return artifacts


def sql_execution_node(state: AgentState) -> dict:
    """
    SQL execution node wrapping SQLExecutionAgent class.
    Supports parallel execution with retry, and sequential one-at-a-time mode.
    """
    writer = get_stream_writer()

    print("\n" + "=" * 80)
    print("SQL EXECUTION AGENT")
    print("=" * 80)

    sql_queries = state.get("sql_queries", [])
    if not sql_queries:
        single_query = state.get("sql_query")
        if single_query:
            sql_queries = [single_query]

    # Handle empty queries (e.g. NO_MORE_QUERIES from sequential synthesis)
    if not sql_queries:
        preserved = list(state.get("preserved_results") or [])
        if preserved:
            print("No new queries — returning preserved results")
            step = state.get("sequential_step", len(preserved))
            completion_reason = (
                "Skipped because the remaining sub-question was considered already covered by prior results."
            )
            padded = _append_remaining_skipped_artifacts(
                state,
                preserved,
                start_step=step,
                reason=completion_reason,
            )
            return {
                "execution_results": padded,
                "execution_result": padded[0],
                "preserved_results": [],
                "sql_retry_feedback": None,
                "loop_reason": None,
                "next_agent": "summarize",
                "messages": [SystemMessage(content=f"Sequential complete: {len(padded)} result sets")],
            }
        return {
            "execution_error": "No SQL queries provided",
            "next_agent": "summarize",
        }

    config = get_config()
    sql_warehouse_id = config.table_metadata.sql_warehouse_id
    if not sql_warehouse_id:
        return {"execution_error": "SQL_WAREHOUSE_ID is not configured"}

    try:
        execution_agent = SQLExecutionAgent(warehouse_id=sql_warehouse_id)
    except Exception as e:
        print(f"Failed to load SQLExecutionAgent: {e}")
        result = _execute_sql_fallback(sql_queries[0], sql_warehouse_id)
        return {
            "execution_result": result,
            "execution_results": [result],
            "next_agent": "summarize",
            "messages": [SystemMessage(content=f"Execution {'successful' if result['success'] else 'failed'}")],
        }

    execution_mode = state.get("execution_mode", "parallel")
    route = state.get("join_strategy_route")

    # ── Sequential mode ──────────────────────────────────────────────────
    if execution_mode == "sequential" and len(sql_queries) >= 1:
        return _execute_sequential(state, execution_agent, sql_queries, route, writer)

    # ── Parallel mode (default) ──────────────────────────────────────────
    return _execute_parallel(state, execution_agent, sql_queries, route, writer)


def _execute_parallel(
    state: AgentState,
    agent: SQLExecutionAgent,
    sql_queries: List[str],
    route: str | None,
    writer,
) -> dict:
    """Run all queries in parallel. On partial failure, retry once via synthesis loop."""
    is_retry = state.get("loop_reason") == "retry"
    preserved = list(state.get("preserved_results") or [])
    labels = list(state.get("sql_query_labels") or [])

    if is_retry:
        sql_queries = _filter_out_already_succeeded(sql_queries, preserved)
        if not sql_queries:
            return {
                "execution_results": preserved,
                "execution_result": preserved[0] if preserved else None,
                "preserved_results": [],
                "sql_retry_feedback": None,
                "loop_reason": None,
                "next_agent": "summarize",
                "messages": [SystemMessage(content="Retry: all queries already succeeded")],
            }

    for i, query in enumerate(sql_queries, 1):
        writer({"type": "sql_execution_start", "estimated_complexity": "standard", "query_number": i})

    execution_results = agent.execute_sql_parallel(sql_queries)
    execution_results = [
        _attach_query_metadata(
            state,
            result,
            query_index=i,
            sql_query=sql_queries[i],
            label=labels[i] if i < len(labels) else None,
        )
        for i, result in enumerate(execution_results)
    ]

    for result in execution_results:
        i = result.get("query_number", 0)
        if result["success"]:
            print(f"✓ Query {i} succeeded: {result['row_count']} rows")
            writer({"type": "sql_execution_complete", "rows": result["row_count"], "columns": result["columns"], "query_number": i})
        else:
            print(f"❌ Query {i} failed: {result.get('error')}")

    if is_retry:
        merged = preserved + execution_results
        return {
            "execution_results": merged,
            "execution_result": merged[0] if merged else None,
            "preserved_results": [],
            "sql_retry_feedback": None,
            "loop_reason": None,
            "next_agent": "summarize",
            "messages": [SystemMessage(content=f"Retry complete: {len(merged)} result sets")],
        }

    successes = [r for r in execution_results if r.get("success")]
    failures = [r for r in execution_results if not r.get("success")]

    if not failures:
        total_rows = sum(r["row_count"] for r in execution_results)
        return {
            "execution_results": execution_results,
            "execution_result": execution_results[0],
            "preserved_results": [],
            "sql_retry_feedback": None,
            "loop_reason": None,
            "next_agent": "summarize",
            "messages": [SystemMessage(content=f"Executed {len(sql_queries)} queries successfully. Total rows: {total_rows}")],
        }

    retry_count = state.get("sql_retry_count", 0)
    retry_max = state.get("sql_retry_max", 1)

    if retry_count < retry_max and route:
        return {
            "execution_results": execution_results,
            "preserved_results": successes,
            "sql_retry_count": retry_count + 1,
            "sql_retry_feedback": _build_retry_feedback(successes, failures, retry_count, retry_max),
            "loop_reason": "retry",
            "next_agent": route,
            "messages": [SystemMessage(content=f"Partial failure ({len(failures)} failed) — retrying via {route}")],
        }

    return {
        "execution_results": execution_results,
        "execution_result": execution_results[0],
        "preserved_results": [],
        "sql_retry_feedback": None,
        "loop_reason": None,
        "next_agent": "summarize",
        "messages": [SystemMessage(content=f"{len(failures)} queries failed, retries exhausted — summarizing partial results")],
    }


def _execute_sequential(
    state: AgentState,
    agent: SQLExecutionAgent,
    sql_queries: List[str],
    route: str | None,
    writer,
) -> dict:
    """Execute one query at a time, looping back to synthesis for the next sub-question."""
    preserved = list(state.get("preserved_results") or [])
    step = state.get("sequential_step", 0)
    total = state.get("total_sub_questions", 1)
    retry_count = state.get("sql_retry_count", 0)
    retry_max = state.get("sql_retry_max", 1)

    query = sql_queries[0]
    labels = list(state.get("sql_query_labels") or [])
    label = labels[0] if labels else None
    writer({"type": "sql_execution_start", "estimated_complexity": "standard", "query_number": step + 1})

    result = agent.execute_sql(query)
    result = _attach_query_metadata(
        state,
        result,
        query_index=step,
        sql_query=query,
        label=label,
    )

    if result["success"]:
        writer({"type": "sql_execution_complete", "rows": result["row_count"], "columns": result["columns"], "query_number": step + 1})
        print(f"✓ Sequential step {step + 1}/{total} succeeded: {result['row_count']} rows")
    else:
        print(f"❌ Sequential step {step + 1}/{total} failed: {result.get('error')}")

    if result.get("success"):
        preserved.append(result)
        next_step = step + 1

        if next_step >= total:
            return {
                "execution_results": preserved,
                "execution_result": preserved[0] if preserved else None,
                "preserved_results": [],
                "executed_result_literals": None,
                "sql_retry_feedback": None,
                "loop_reason": None,
                "next_agent": "summarize",
                "messages": [SystemMessage(content=f"Sequential complete: {len(preserved)} result sets")],
            }
        from ..utils.executed_result_literals import build_executed_literal_package

        literal_package = build_executed_literal_package(preserved)
        return {
            "preserved_results": preserved,
            "sequential_step": next_step,
            "sql_retry_count": 0,
            "executed_result_literals": literal_package if literal_package.get("has_literals") else None,
            "sql_retry_feedback": _build_sequential_feedback(preserved, step=next_step, total=total),
            "loop_reason": "sequential_next",
            "next_agent": route or "summarize",
            "messages": [SystemMessage(content=f"Step {next_step}/{total} — continuing to next sub-question")],
        }

    # Failure path
    if retry_count < retry_max and route:
        return {
            "preserved_results": preserved,
            "sql_retry_count": retry_count + 1,
            "sql_retry_feedback": _build_retry_feedback(preserved, [result], retry_count, retry_max),
            "loop_reason": "retry",
            "next_agent": route,
            "messages": [SystemMessage(content=f"Step {step + 1}/{total} failed — retrying")],
        }

    # Retry exhausted — skip this step
    preserved.append(result)
    next_step = step + 1
    if next_step >= total:
        return {
            "execution_results": preserved,
            "execution_result": preserved[0] if preserved else None,
            "preserved_results": [],
            "executed_result_literals": None,
            "sql_retry_feedback": None,
            "loop_reason": None,
            "next_agent": "summarize",
            "messages": [SystemMessage(content=f"Sequential complete (with skipped failures): {len(preserved)} result sets")],
        }
    from ..utils.executed_result_literals import build_executed_literal_package

    literal_package = build_executed_literal_package(preserved)
    return {
        "preserved_results": preserved,
        "sequential_step": next_step,
        "sql_retry_count": 0,
        "executed_result_literals": literal_package if literal_package.get("has_literals") else None,
        "sql_retry_feedback": _build_sequential_feedback(preserved, step=next_step, total=total),
        "loop_reason": "sequential_next",
        "next_agent": route or "summarize",
        "messages": [SystemMessage(content=f"Step {step + 1} failed (retries exhausted), skipping to step {next_step + 1}")],
    }


def _execute_sql_fallback(sql_query: str, warehouse_id: str) -> Dict[str, Any]:
    """
    Fallback SQL execution function.
    
    This is a simplified implementation. In production, use SQLExecutionAgent class.
    """
    try:
        from databricks import sql
        from databricks.sdk.core import Config
        import re
        
        # Extract SQL from markdown code blocks if present
        extracted_sql = sql_query.strip()
        if "```sql" in extracted_sql.lower():
            sql_match = re.search(r'```sql\s*(.*?)\s*```', extracted_sql, re.IGNORECASE | re.DOTALL)
            if sql_match:
                extracted_sql = sql_match.group(1).strip()
        elif "```" in extracted_sql:
            sql_match = re.search(r'```\s*(.*?)\s*```', extracted_sql, re.DOTALL)
            if sql_match:
                extracted_sql = sql_match.group(1).strip()
        
        # Enforce trailing LIMIT clause (not inner CTE LIMITs)
        max_rows = 1000
        trailing_limit = re.search(r'\s+LIMIT\s+(\d+)(?:\s+OFFSET\s+\d+)?\s*;?\s*$', extracted_sql, re.IGNORECASE)
        if trailing_limit:
            existing_limit = int(trailing_limit.group(1))
            if existing_limit > max_rows:
                extracted_sql = extracted_sql[:trailing_limit.start()] + f' LIMIT {max_rows}' + extracted_sql[trailing_limit.end():]
        else:
            extracted_sql = f"{extracted_sql.rstrip(';')} LIMIT {max_rows}"
        
        # Initialize Databricks Config
        cfg = Config()
        
        # Execute SQL query
        print(f"\n{'='*80}")
        print("🔍 EXECUTING SQL QUERY (via SQL Warehouse)")
        print(f"{'='*80}")
        print(f"Warehouse ID: {warehouse_id}")
        print(f"SQL:\n{extracted_sql}")
        print(f"{'='*80}\n")
        
        with sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            credentials_provider=lambda: cfg.authenticate,
            session_configuration={"ansi_mode": "true"},
            socket_timeout=900,
            http_retry_delay_min=1,
            http_retry_delay_max=60,
            http_retry_max_redirects=5,
            http_retry_stop_after_attempts=30,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(extracted_sql)
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                row_count = len(results)
                
                print(f"✅ Query executed successfully!")
                print(f"📊 Rows returned: {row_count} (LIMIT enforced at {max_rows})")
                print(f"📋 Columns: {', '.join(columns)}\n")
                
                # Convert results to list of dicts
                result_data = SQLExecutionAgent._normalize_result_rows(columns, results)
                
                return {
                    "success": True,
                    "sql": extracted_sql,
                    "result": result_data,
                    "row_count": row_count,
                    "columns": columns,
                }
                
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        print(f"\n{'='*80}")
        print(f"❌ SQL EXECUTION FAILED - {error_type}")
        print(f"{'='*80}")
        print(f"Error Message: {error_msg}")
        print(f"Warehouse ID: {warehouse_id}")
        print(f"{'='*80}\n")
        
        return {
            "success": False,
            "sql": extracted_sql if 'extracted_sql' in locals() else sql_query,
            "result": None,
            "row_count": 0,
            "columns": [],
            "error": f"{error_type}: {error_msg}",
            "error_type": error_type,
        }
