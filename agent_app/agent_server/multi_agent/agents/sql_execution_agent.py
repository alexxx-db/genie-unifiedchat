"""
SQL Execution Agent

This module provides the SQLExecutionAgent class for executing SQL queries
using Databricks SQL Warehouse.

PRODUCTION-READY DESIGN:
- Uses databricks-sql-connector with unified authentication (Config + credentials_provider)
- Automatically handles OAuth credentials when deployed with registered resources
- Supports both development and deployed app environments

AUTHENTICATION WITH AUTOMATIC PASSTHROUGH:
When you register resources during agent deployment:

    resources = [
        DatabricksSQLWarehouse(warehouse_id=SQL_WAREHOUSE_ID),
        # ... other resources
    ]
    mlflow.langchain.log_model(..., resources=resources)

Databricks automatically:
1. Creates a service principal for your agent
2. Manages OAuth token generation and rotation
3. Injects credentials into the runtime environment

The Config() class automatically reads workspace host and injected OAuth credentials,
eliminating the need for manual DATABRICKS_HOST/DATABRICKS_TOKEN configuration.

Reference: https://docs.databricks.com/generative-ai/agent-framework/agent-authentication

LEGACY MANUAL AUTHENTICATION (if not using automatic passthrough):
If you're not using resource registration, you can still manually configure:
- DATABRICKS_HOST and DATABRICKS_TOKEN via environment variables
- Config() will still read them from the environment
"""

import re
import json
from typing import Dict, Any, List, Tuple
from datetime import date, datetime
from decimal import Decimal


# Expression types that represent read-only (non-mutating) SQL. Anything else
# — INSERT/UPDATE/DELETE/MERGE/CREATE/DROP/ALTER/GRANT/TRUNCATE/… — is rejected
# before it ever reaches the warehouse.
_READ_ONLY_COMMAND_PREFIXES = ("SHOW", "DESCRIBE", "DESC", "EXPLAIN")


def _is_read_only_sql(sql: str) -> Tuple[bool, str]:
    """
    Return (is_read_only, reason). Defense-in-depth guard so LLM-synthesized
    SQL (which can be steered by prompt injection) cannot run mutating
    statements even if the warehouse principal has write grants.

    Uses sqlglot to parse each statement and allows only SELECT/WITH/UNION-style
    read queries plus a small set of read-only commands (SHOW/DESCRIBE/EXPLAIN).
    If parsing fails, falls back to a conservative first-keyword denylist.
    """
    import sqlglot
    from sqlglot import exp

    stripped = sql.strip()
    if not stripped:
        return False, "empty SQL"

    # exp.SetOperation is the base of Union/Intersect/Except in sqlglot 30.x
    # (Intersect/Except do NOT subclass Union). exp.Use / exp.Set are session
    # context, not data mutations, so leading USE/SET before a query is allowed.
    set_op = getattr(exp, "SetOperation", exp.Union)
    read_only_types = (
        exp.Select, set_op, exp.Union, exp.Subquery, exp.With,
        exp.Describe, exp.Use, exp.Set,
    )
    write_keywords = {
        "INSERT", "UPDATE", "DELETE", "MERGE", "REPLACE", "UPSERT",
        "CREATE", "DROP", "ALTER", "TRUNCATE", "GRANT", "REVOKE",
        "COPY", "CALL", "REFRESH", "OPTIMIZE", "VACUUM",
    }

    try:
        statements = [s for s in sqlglot.parse(stripped, read="databricks") if s]
    except Exception:
        statements = None

    if statements:
        for stmt in statements:
            if isinstance(stmt, read_only_types):
                continue
            if isinstance(stmt, exp.Command):
                # sqlglot represents unsupported/uncommon statements as Command;
                # allow only known read-only commands (SHOW/DESCRIBE/EXPLAIN).
                name = (stmt.name or "").upper()
                if name in _READ_ONLY_COMMAND_PREFIXES:
                    continue
                return False, f"non-read-only command: {name or type(stmt).__name__}"
            return False, f"non-read-only statement: {type(stmt).__name__}"
        return True, ""

    # Parsing failed — fail closed. Reject if ANY write keyword appears as a
    # standalone token (catches e.g. `WITH ... INSERT ...` where a leading read
    # keyword would otherwise mask a trailing mutation), and only allow when the
    # statement clearly begins with a read verb.
    tokens = {t.upper() for t in re.findall(r"[A-Za-z_]+", stripped)}
    hit = tokens & write_keywords
    if hit:
        return False, f"non-read-only statement: {sorted(hit)[0]}"
    # Guard on the residue after stripping leading whitespace/parens: re.sub can
    # yield "" from a non-empty input like "((", so checking `stripped` would
    # crash on [0]. Fail closed on empty residue.
    residue = re.sub(r"^[\s(]*", "", stripped)
    first_word = residue.split(None, 1)[0].upper() if residue else ""
    if first_word in ("SELECT", "WITH", "TABLE", "VALUES", "FROM") or first_word in _READ_ONLY_COMMAND_PREFIXES:
        return True, ""
    return False, f"could not verify statement is read-only (starts with {first_word or 'nothing'})"


def _is_limitable_sql(sql: str) -> bool:
    """
    True only when it is safe to append a trailing ``LIMIT`` — i.e. the final
    statement is a row-returning query. SHOW/DESCRIBE/EXPLAIN/USE/SET do not
    accept a trailing LIMIT and would become invalid SQL.
    """
    import sqlglot
    from sqlglot import exp

    set_op = getattr(exp, "SetOperation", exp.Union)
    query_types = (exp.Select, set_op, exp.Union, exp.Subquery, exp.With)
    try:
        statements = [s for s in sqlglot.parse(sql, read="databricks") if s]
    except Exception:
        statements = None
    if statements:
        return isinstance(statements[-1], query_types)
    # Parse failed: only append LIMIT for clearly query-shaped SQL. Guard on the
    # residue after re.sub (which can be empty for input like "((").
    residue = re.sub(r"^[\s(]*", "", sql.strip())
    first_word = residue.split(None, 1)[0].upper() if residue else ""
    return first_word in ("SELECT", "WITH", "TABLE", "VALUES", "FROM")


class SQLExecutionAgent:
    """
    Agent responsible for executing SQL queries using Databricks SQL Warehouse.
    
    PRODUCTION-READY DESIGN:
    - Uses databricks-sql-connector with unified authentication (Config + credentials_provider)
    - Automatically handles OAuth credentials when deployed with registered resources
    - Supports both development and deployed app environments
    
    AUTHENTICATION WITH AUTOMATIC PASSTHROUGH:
    When you register resources during agent deployment:
    
        resources = [
            DatabricksSQLWarehouse(warehouse_id=SQL_WAREHOUSE_ID),
            # ... other resources
        ]
        mlflow.langchain.log_model(..., resources=resources)
    
    Databricks automatically:
    1. Creates a service principal for your agent
    2. Manages OAuth token generation and rotation
    3. Injects credentials into the runtime environment
    
    The Config() class automatically reads workspace host and injected OAuth credentials,
    eliminating the need for manual DATABRICKS_HOST/DATABRICKS_TOKEN configuration.
    
    Reference: https://docs.databricks.com/generative-ai/agent-framework/agent-authentication
    
    LEGACY MANUAL AUTHENTICATION (if not using automatic passthrough):
    If you're not using resource registration, you can still manually configure:
    - DATABRICKS_HOST and DATABRICKS_TOKEN via environment variables
    - Config() will still read them from the environment
    """
    
    def __init__(self, warehouse_id: str):
        """
        Initialize SQL Execution Agent.
        
        Args:
            warehouse_id: Databricks SQL Warehouse ID for query execution
        """
        self.name = "SQLExecution"
        self.warehouse_id = warehouse_id

    @staticmethod
    def _normalize_result_value(value: Any) -> Any:
        """Convert connector return values into JSON-safe Python types."""
        if isinstance(value, dict):
            return {
                str(key): SQLExecutionAgent._normalize_result_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [SQLExecutionAgent._normalize_result_value(item) for item in value]
        if isinstance(value, tuple):
            return [SQLExecutionAgent._normalize_result_value(item) for item in value]
        if isinstance(value, set):
            return [SQLExecutionAgent._normalize_result_value(item) for item in value]
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        try:
            import numpy as np  # type: ignore

            if isinstance(value, np.ndarray):
                return SQLExecutionAgent._normalize_result_value(value.tolist())
            if isinstance(value, np.generic):
                return SQLExecutionAgent._normalize_result_value(value.item())
        except ImportError:
            pass

        if hasattr(value, "tolist") and callable(value.tolist):
            try:
                return SQLExecutionAgent._normalize_result_value(value.tolist())
            except Exception:
                pass

        if hasattr(value, "item") and callable(value.item):
            try:
                return SQLExecutionAgent._normalize_result_value(value.item())
            except Exception:
                pass

        if hasattr(value, "isoformat") and callable(value.isoformat):
            try:
                return value.isoformat()
            except Exception:
                pass

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)

    @staticmethod
    def _normalize_result_rows(columns: List[str], rows: List[Any]) -> List[Dict[str, Any]]:
        """Build row dicts with normalized values safe for JSON serialization."""
        return [
            {
                column: SQLExecutionAgent._normalize_result_value(value)
                for column, value in zip(columns, row)
            }
            for row in rows
        ]
    
    def execute_sql(
        self, 
        sql_query: str, 
        max_rows: int = 1000,
        return_format: str = "dict"
    ) -> Dict[str, Any]:
        """
        Execute SQL query using Databricks SQL Warehouse and return formatted results.
        
        PRODUCTION BEST PRACTICES IMPLEMENTED:
        1. Context Managers: Uses 'with' statements for automatic resource cleanup
        2. Connection Resilience: Configures timeouts and retry logic for transient failures
        3. Proper Error Handling: Categorizes errors for better production debugging
        4. ANSI SQL Mode: Ensures consistent SQL behavior across environments
        5. App Runtime Compatible: Works without Spark session via REST API
        
        Connection Configuration:
        - Socket timeout: 900s (balances app/runtime request limits with warehouse query time)
        - HTTP retries: 30 attempts with exponential backoff (1-60s)
        - Session config: ANSI mode enabled for SQL compliance
        
        Args:
            sql_query: Support two types: 
                1) The result from invoke the SQL synthesis agent (dict with messages)
                2) The SQL query string (can be raw SQL or contain markdown code blocks)
            max_rows: Maximum number of rows to return (default: 1000)
            return_format: Format of the result - "dict", "json", or "markdown"
            
        Returns:
            Dictionary containing:
            - success: bool - Whether execution was successful
            - sql: str - The executed SQL query
            - result: Any - Query results in requested format
            - row_count: int - Number of rows returned
            - columns: List[str] - Column names
            - error: str - Error message if failed (optional)
            - error_type: str - Exception type for debugging (only on failure)
            - error_hint: str - Suggested resolution (only on failure)
        """
        from databricks import sql
        from databricks.sdk.core import Config
        
        # Step 1: Extract SQL from agent result or markdown code blocks if present
        if sql_query and isinstance(sql_query, dict) and "messages" in sql_query:
            sql_query = sql_query["messages"][-1].content
        
        extracted_sql = sql_query.strip()
        
        if "```sql" in extracted_sql.lower():
            # Extract content between ```sql and ```
            sql_match = re.search(r'```sql\s*(.*?)\s*```', extracted_sql, re.IGNORECASE | re.DOTALL)
            if sql_match:
                extracted_sql = sql_match.group(1).strip()
        elif "```" in extracted_sql:
            # Extract any code block
            sql_match = re.search(r'```\s*(.*?)\s*```', extracted_sql, re.DOTALL)
            if sql_match:
                extracted_sql = sql_match.group(1).strip()
        
        # Step 1b: Reject non-read-only SQL before it reaches the warehouse.
        # This is defense-in-depth: the SQL is LLM-generated and could be steered
        # toward mutating statements via prompt injection.
        is_read_only, ro_reason = _is_read_only_sql(extracted_sql)
        if not is_read_only:
            print(f"⛔ Rejected non-read-only SQL: {ro_reason}")
            return {
                "success": False,
                "sql": extracted_sql,
                "result": None,
                "row_count": 0,
                "columns": [],
                "error": f"Only read-only SQL is permitted (SELECT/WITH/UNION/SHOW/DESCRIBE/EXPLAIN/USE/SET): {ro_reason}",
                "error_type": "ReadOnlyPolicyViolation",
                "error_hint": "Rephrase the request as a read-only query (e.g., SELECT/WITH) rather than a mutating statement.",
            }

        # Step 2: Enforce LIMIT clause (for safety and token management)
        # Only match a trailing LIMIT at the end of the statement, not inside CTEs/subqueries
        trailing_limit = re.search(r'\s+LIMIT\s+(\d+)(?:\s+OFFSET\s+\d+)?\s*;?\s*$', extracted_sql, re.IGNORECASE)
        if trailing_limit:
            existing_limit = int(trailing_limit.group(1))
            if existing_limit > max_rows:
                extracted_sql = extracted_sql[:trailing_limit.start()] + f' LIMIT {max_rows}' + extracted_sql[trailing_limit.end():]
                print(f"⚠️  Reduced trailing LIMIT from {existing_limit} to {max_rows} (max_rows enforcement)")
        elif _is_limitable_sql(extracted_sql):
            # Only append LIMIT to row-returning queries. Appending it to
            # SHOW/DESCRIBE/EXPLAIN/USE/SET would produce invalid SQL.
            extracted_sql = f"{extracted_sql.rstrip(';')} LIMIT {max_rows}"
        
        try:
            # Step 3: Initialize Databricks Config for unified authentication
            # BEST PRACTICE: Config() automatically reads workspace host and OAuth credentials
            # - In deployed runtimes with automatic passthrough: reads injected service principal credentials
            # - In notebooks: reads from notebook context or environment variables
            # - With manual config: reads DATABRICKS_HOST and DATABRICKS_TOKEN from environment
            cfg = Config()
            
            # Step 4: Execute the SQL query using SQL Warehouse
            print(f"\n{'='*80}")
            print("🔍 EXECUTING SQL QUERY (via SQL Warehouse)")
            print(f"{'='*80}")
            print(f"Warehouse ID: {self.warehouse_id}")
            print(f"SQL:\n{extracted_sql}")
            print(f"{'='*80}\n")
            
            # Connect to SQL Warehouse using context manager (production best practice)
            # Context managers ensure proper cleanup even if exceptions occur
            # credentials_provider=cfg lets the connector fetch OAuth tokens transparently
            with sql.connect(
                server_hostname=cfg.host,
                http_path=f"/sql/1.0/warehouses/{self.warehouse_id}",
                credentials_provider=lambda: cfg.authenticate,  # Unified authentication - handles OAuth automatically
                # Production settings for resilience
                session_configuration={
                    "ansi_mode": "true"  # Enable ANSI SQL compliance for consistent behavior
                },
                socket_timeout=900,  # 15 minutes to accommodate longer warehouse queries
                http_retry_delay_min=1,  # Minimum retry delay in seconds
                http_retry_delay_max=60,  # Maximum retry delay in seconds
                http_retry_max_redirects=5,  # Max HTTP redirects
                http_retry_stop_after_attempts=30,  # Max retry attempts for transient failures
            ) as connection:
                
                # Use nested context manager for cursor (ensures cursor cleanup)
                with connection.cursor() as cursor:
                    
                    # Execute query
                    cursor.execute(extracted_sql)
                    
                    # PHASE 2 OPTIMIZATION: Get row count efficiently
                    # Try to use cursor.rowcount if available (more efficient than len(fetchall()))
                    columns = [desc[0] for desc in cursor.description]
                    
                    # Fetch results (limited by LIMIT clause already enforced)
                    results = cursor.fetchall()
                    
                    # Post-execution safety truncation
                    if len(results) > max_rows:
                        results = results[:max_rows]
                    row_count = len(results)
                    
                    print(f"✅ Query executed successfully!")
                    print(f"📊 Rows returned: {row_count} (LIMIT enforced at {max_rows})")
                    print(f"📋 Columns: {', '.join(columns)}\n")
                    
                    # Step 5: Convert results to list of dicts for compatibility
                    result_data = self._normalize_result_rows(columns, results)
                    
                # Cursor automatically closed here by context manager
            
            # Connection automatically closed here by context manager
            
            # Step 6: Format results based on return_format
            if return_format == "json":
                # Convert to JSON strings (matching old spark behavior)
                result_data = [json.dumps(row) for row in result_data]
            elif return_format == "markdown":
                # Create markdown table
                import pandas as pd
                pandas_df = pd.DataFrame(result_data)
                result_data = pandas_df.to_markdown(index=False)
            # else: dict format (default) - already in correct format
            
            return {
                "success": True,
                "sql": extracted_sql,
                "result": result_data,
                "row_count": row_count,
                "columns": columns,
            }
            
        except Exception as e:
            # Step 8: Handle errors with specific exception types for better diagnostics
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Provide production-grade error categorization
            if "DatabaseError" in error_type or "OperationalError" in error_type:
                error_category = "SQL Execution Error"
                error_hint = "Check SQL syntax and table/column permissions"
            elif "ConnectionError" in error_type or "timeout" in error_msg.lower():
                error_category = "Connection Error"
                error_hint = "Verify SQL Warehouse is running and network connectivity"
            elif "Authentication" in error_msg or "Unauthorized" in error_msg:
                error_category = "Authentication Error"
                error_hint = "Verify access token and warehouse permissions"
            else:
                error_category = "General Error"
                error_hint = "Review full error details below"
            
            print(f"\n{'='*80}")
            print(f"❌ SQL EXECUTION FAILED - {error_category}")
            print(f"{'='*80}")
            print(f"Error Type: {error_type}")
            print(f"Error Message: {error_msg}")
            print(f"Hint: {error_hint}")
            print(f"Warehouse ID: {self.warehouse_id}")
            print(f"{'='*80}\n")
            
            return {
                "success": False,
                "sql": extracted_sql,
                "result": None,
                "row_count": 0,
                "columns": [],
                "error": f"{error_category}: {error_msg}",
                "error_type": error_type,
                "error_hint": error_hint
            }
    
    def execute_sql_parallel(
        self,
        sql_queries: List[str],
        max_rows: int = 1000,
        return_format: str = "dict",
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple SQL queries in parallel using ThreadPoolExecutor.
        
        Each query runs in its own thread with an independent sql.connect() connection,
        so there is no shared state between threads. This is safe because execute_sql()
        creates and closes its own connection/cursor via context managers per call.
        
        ThreadPoolExecutor is used instead of asyncio because databricks-sql-connector
        is synchronous (no native async API), and the work is I/O-bound (waiting on
        SQL Warehouse HTTP responses), so the GIL is not a bottleneck.
        
        Args:
            sql_queries: List of SQL query strings to execute
            max_rows: Maximum rows per query (default: 1000)
            return_format: Result format - "dict", "json", or "markdown"
            max_workers: Maximum concurrent threads (default: 4, tune to warehouse concurrency)
        
        Returns:
            List of result dicts (same format as execute_sql), ordered to match input queries.
            Each result includes a "query_number" field (1-indexed).
        """
        import concurrent.futures
        import contextvars
        
        # Fast path: skip threading overhead for single query
        if len(sql_queries) <= 1:
            if sql_queries:
                result = self.execute_sql(sql_queries[0], max_rows, return_format)
                result["query_number"] = 1
                return [result]
            return []
        
        print(f"⚡ Executing {len(sql_queries)} queries in parallel (max_workers={min(len(sql_queries), max_workers)})")
        
        results = [None] * len(sql_queries)  # Pre-allocate to preserve ordering
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sql_queries), max_workers)) as executor:
            future_to_idx = {
                executor.submit(contextvars.copy_context().run, self.execute_sql, query, max_rows, return_format): idx
                for idx, query in enumerate(sql_queries)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:
                    # Catch unexpected errors not handled inside execute_sql
                    result = {
                        "success": False,
                        "sql": sql_queries[idx],
                        "result": None,
                        "row_count": 0,
                        "columns": [],
                        "error": f"Parallel execution error: {type(e).__name__}: {str(e)}"
                    }
                result["query_number"] = idx + 1
                results[idx] = result
        
        succeeded = sum(1 for r in results if r["success"])
        print(f"⚡ Parallel execution complete: {succeeded}/{len(sql_queries)} succeeded")
        
        return results
    
    def __call__(self, sql_query: str, max_rows: int = 1000, return_format: str = "dict") -> Dict[str, Any]:
        """Make agent callable."""
        return self.execute_sql(sql_query, max_rows, return_format)
