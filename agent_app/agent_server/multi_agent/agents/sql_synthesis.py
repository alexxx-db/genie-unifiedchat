"""
SQL Synthesis Agent Nodes

This module provides two SQL synthesis strategies for the multi-agent system:

1. Table Route (sql_synthesis_table_node):
   - Fast SQL synthesis using Unity Catalog (UC) function tools
   - Queries metadata directly from UC functions (get_space_summary, get_table_overview, get_space_instructions, etc.)
   - Optimized for single-space or simple multi-space queries
   - Uses cached SQLSynthesisTableAgent instance for performance

2. Genie Route (sql_synthesis_genie_node):
   - Slower but more powerful SQL synthesis using Genie agents as tools
   - Orchestrates multiple Genie agents in parallel to gather SQL fragments
   - Best for complex queries requiring coordination across multiple spaces
   - Uses dedicated SQL_SYNTHESIS_GENIE endpoint for stronger reasoning

Both functions:
- Use minimal state extraction to reduce token usage
- Emit streaming events for observability
- Return state updates as dictionaries for clean MLflow traces
- Handle errors gracefully with appropriate error messages
"""

import json
import logging
import time
from functools import wraps
from typing import Dict, List, Optional, Any, Callable

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer

# Type imports
from ..core.state import AgentState

# Agent class imports
from .sql_synthesis_agents import (
    SQLSynthesisTableAgent,
    SQLSynthesisGenieAgent
)

# SQL extraction utilities for multi-query support
from ..utils.sql_extraction import extract_sql_queries_from_agent_result

# LLM and utility imports
try:
    from databricks_langchain import ChatDatabricks
except ImportError:
    ChatDatabricks = None  # type: ignore

# Performance metrics storage (module-level)
_performance_metrics = {
    "node_timings": {},
    "agent_model_usage": {},
    "cache_stats": {}
}

logger = logging.getLogger(__name__)


def _debug_log(location: str, message: str, data: Optional[dict] = None) -> None:
    """Emit routing/synthesis debug info via the standard logger.

    Previously this wrote to a hard-coded developer path on every call, which
    leaked routing data locally and silently failed everywhere else.
    """
    logger.debug("%s | %s | %s", location, message, data or {})

# Agent cache (module-level)
_agent_cache = {}

# LLM connection pool (module-level)
_llm_connection_pool = {}

SQL_SYNTHESIS_MAX_TOKENS = 8192 * 2


# ==============================================================================
# Helper Functions
# ==============================================================================

def extract_synthesis_table_context(state: AgentState) -> dict:
    """Extract minimal context for table-based SQL synthesis."""
    return {
        "plan": state.get("plan", {}),
        "relevant_space_ids": state.get("relevant_space_ids", [])
    }


def extract_synthesis_genie_context(state: AgentState) -> dict:
    """Extract minimal context for genie-based SQL synthesis."""
    return {
        "plan": state.get("plan", {}),
        "relevant_spaces": state.get("relevant_spaces", []),
        "genie_route_plan": state.get("genie_route_plan"),
        "genie_execution_mode": state.get("genie_execution_mode"),
        "dependency_edges": state.get("dependency_edges"),
        "genie_conversation_ids": state.get("genie_conversation_ids"),
    }


def record_cache_hit(cache_type: str):
    """Record a cache hit for monitoring."""
    key = f"{cache_type}_hits"
    if key in _performance_metrics["cache_stats"]:
        _performance_metrics["cache_stats"][key] += 1
    else:
        _performance_metrics["cache_stats"][key] = 1


def record_cache_miss(cache_type: str):
    """Record a cache miss for monitoring."""
    key = f"{cache_type}_misses"
    if key in _performance_metrics["cache_stats"]:
        _performance_metrics["cache_stats"][key] += 1
    else:
        _performance_metrics["cache_stats"][key] = 1


def get_pooled_llm(endpoint_name: str, temperature: float = 0.1, max_tokens: int = None):
    """
    Get or create a pooled LLM connection.
    Reuses connections across requests to avoid connection overhead.
    
    Args:
        endpoint_name: Name of the LLM endpoint
        temperature: Temperature for generation (default 0.1)
        max_tokens: Maximum tokens to generate (default None)
    
    Returns:
        ChatDatabricks instance from pool
    """
    if ChatDatabricks is None:
        raise ImportError("ChatDatabricks is not available. Install databricks-langchain.")
    
    # Create a cache key that includes temperature and max_tokens
    cache_key = f"{endpoint_name}_{temperature}_{max_tokens}"
    
    if cache_key not in _llm_connection_pool:
        record_cache_miss("llm_pool")
        print(f"⚡ Creating pooled LLM connection: {endpoint_name} (temperature={temperature})")
        kwargs = {"endpoint": endpoint_name, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        _llm_connection_pool[cache_key] = ChatDatabricks(**kwargs)
        print(f"✓ LLM connection pooled: {cache_key}")
    else:
        record_cache_hit("llm_pool")
        print(f"♻️ Reusing pooled LLM connection: {cache_key} (-50ms to -200ms)")
    
    return _llm_connection_pool[cache_key]


def track_agent_model_usage(agent_name: str, model_endpoint: str):
    """
    Track which LLM model is used by each agent for monitoring and cost analysis.
    
    Args:
        agent_name: Name of the agent (e.g., "sql_synthesis_table", "sql_synthesis_genie")
        model_endpoint: LLM endpoint being used (e.g., "databricks-claude-haiku-4-5")
    """
    if "agent_model_usage" not in _performance_metrics:
        _performance_metrics["agent_model_usage"] = {}
    
    if agent_name not in _performance_metrics["agent_model_usage"]:
        _performance_metrics["agent_model_usage"][agent_name] = {
            "model": model_endpoint,
            "invocations": 0
        }
    
    _performance_metrics["agent_model_usage"][agent_name]["invocations"] += 1
    print(f"📊 Agent '{agent_name}' using model: {model_endpoint}")


def measure_node_time(node_name: str):
    """
    Decorator to measure node execution time.
    Expected use: Track per-node performance for optimization.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # Record timing
                if node_name not in _performance_metrics["node_timings"]:
                    _performance_metrics["node_timings"][node_name] = []
                _performance_metrics["node_timings"][node_name].append(elapsed)
                
                # Print timing
                print(f"⏱️  {node_name}: {elapsed:.3f}s")
                
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"⏱️  {node_name}: {elapsed:.3f}s (FAILED)")
                raise
        return wrapper
    return decorator


def get_cached_sql_table_agent(llm_endpoint=None, catalog=None, schema=None):
    """
    Get or create cached SQLSynthesisTableAgent instance.
    Expected gain: -500ms to -1s per request
    """
    if SQLSynthesisTableAgent is None:
        raise ImportError("SQLSynthesisTableAgent is not available. Import it from the appropriate module.")
    
    if llm_endpoint is None or catalog is None or schema is None:
        from ..core.config import get_config
        config = get_config()
        if llm_endpoint is None:
            llm_endpoint = config.llm.sql_synthesis_table_endpoint
        if catalog is None:
            catalog = config.unity_catalog.catalog_name
        if schema is None:
            schema = config.unity_catalog.schema_name
            
    if llm_endpoint is None:
        raise ValueError("LLM_ENDPOINT_SQL_SYNTHESIS_TABLE must be configured")
    
    if catalog is None or schema is None:
        raise ValueError("CATALOG and SCHEMA must be configured")
    
    if "sql_table" not in _agent_cache:
        record_cache_miss("agent_cache")
        print("⚡ Creating SQLSynthesisTableAgent (first use)...")
        llm = get_pooled_llm(llm_endpoint, max_tokens=SQL_SYNTHESIS_MAX_TOKENS)
        _agent_cache["sql_table"] = SQLSynthesisTableAgent(llm, catalog, schema)
        print("✓ SQLSynthesisTableAgent cached")
    else:
        record_cache_hit("agent_cache")
        print("✓ Using cached SQLSynthesisTableAgent")
    return _agent_cache["sql_table"]


# ==============================================================================
# SQL Synthesis Node Functions
# ==============================================================================

def _preserved_as_execution_results(state: AgentState) -> dict:
    """If there are preserved_results from a loop, surface them as execution_results
    so the summarize node can access accumulated data."""
    preserved = list(state.get("preserved_results") or [])
    if preserved:
        return {
            "execution_results": preserved,
            "execution_result": preserved[0],
            "preserved_results": [],
        }
    return {}


def _build_loop_prompt_prefix(state: AgentState) -> Optional[str]:
    """Build a prompt prefix based on loop_reason (retry or sequential continuation)."""
    loop_reason = state.get("loop_reason")
    feedback = state.get("sql_retry_feedback")
    if not loop_reason or not feedback:
        return None

    if loop_reason == "retry":
        return f"RETRY CONTEXT:\n{feedback}\n\nFix the failed query and regenerate."

    if loop_reason == "sequential_next":
        step = state.get("sequential_step", 0)
        sub_questions = state.get("sub_questions") or []
        remaining = sub_questions[step:] if step < len(sub_questions) else []
        remaining_str = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(remaining))
        return (
            f"SEQUENTIAL CONTEXT:\n{feedback}\n\n"
            f"### Remaining sub-questions (soft guidance, not rigid):\n{remaining_str}\n\n"
            f"Based on the previous results above, decide what to do next:\n"
            f"- If the next sub-question is already answered by prior results, skip it.\n"
            f"- If you can write a better query using data from prior results "
            f"(e.g., exact codes, IDs), do so.\n"
            f"- If ALL remaining questions are already addressed, return the "
            f"special marker: NO_MORE_QUERIES\n"
            f"- Otherwise, generate ONLY ONE query for the most relevant "
            f"next sub-question.\n"
            f"- The query must be independently executable "
            f"(no temp tables from prior queries)."
        )

    return None


def _check_no_more_queries(result: dict) -> bool:
    """Check if the synthesis agent returned NO_MORE_QUERIES."""
    raw = result.get("raw_content", "") or result.get("explanation", "")
    return "NO_MORE_QUERIES" in raw


def _build_synthesis_explanation_entry(
    *,
    route: str,
    state: AgentState,
    explanation: str,
    query_labels: Optional[List[str]] = None,
    has_sql: bool = False,
    synthesis_error: Optional[str] = None,
) -> dict:
    """Build a structured explanation entry for later UI rendering."""
    return {
        "route": route,
        "loop_reason": state.get("loop_reason"),
        "retry_count": state.get("sql_retry_count", 0),
        "sequential_step": state.get("sequential_step", 0),
        "has_sql": has_sql,
        "synthesis_error": synthesis_error,
        "query_labels": query_labels or [],
        "explanation": explanation or "",
    }


def _append_synthesis_explanation(
    state: AgentState,
    *,
    route: str,
    explanation: str,
    query_labels: Optional[List[str]] = None,
    has_sql: bool = False,
    synthesis_error: Optional[str] = None,
) -> List[dict]:
    existing = list(state.get("sql_synthesis_explanations") or [])
    existing.append(
        _build_synthesis_explanation_entry(
            route=route,
            state=state,
            explanation=explanation,
            query_labels=query_labels,
            has_sql=has_sql,
            synthesis_error=synthesis_error,
        )
    )
    return existing


@measure_node_time("sql_synthesis_table")
def sql_synthesis_table_node(state: AgentState) -> dict:
    """
    Fast SQL synthesis node wrapping SQLSynthesisTableAgent class.
    Supports retry and sequential modes via loop_reason state field.
    """
    writer = get_stream_writer()
    
    print("\n" + "="*80)
    print("SQL SYNTHESIS AGENT - TABLE ROUTE")
    print("="*80)
    _debug_log(
        "sql_synthesis.py:sql_synthesis_table_node:entry",
        "entered table synthesis node",
        {
            "force_synthesis_route": state.get("force_synthesis_route"),
            "join_strategy": state.get("join_strategy"),
            "next_agent": state.get("next_agent"),
            "plan_join_strategy": state.get("plan", {}).get("join_strategy") if isinstance(state.get("plan"), dict) else None,
        },
    )
    
    context = extract_synthesis_table_context(state)
    
    plan = context.get("plan", {})
    relevant_space_ids = context.get("relevant_space_ids", [])
    
    # Emit synthesis start event
    writer({"type": "sql_synthesis_start", "route": "table", "spaces": relevant_space_ids})
    
    # Get configuration
    from ..core.config import get_config
    config = get_config()
    llm_endpoint = config.llm.sql_synthesis_table_endpoint
    catalog = config.unity_catalog.catalog_name
    schema = config.unity_catalog.schema_name
    
    # OPTIMIZATION: Use cached agent instance
    sql_agent = get_cached_sql_table_agent(llm_endpoint, catalog, schema)
    track_agent_model_usage("sql_synthesis_table", llm_endpoint)
    
    print("plan loaded from state is:", plan)
    print(json.dumps(plan, indent=2))
    
    try:
        # Inject retry / sequential context into the plan if looping
        loop_prefix = _build_loop_prompt_prefix(state)
        if loop_prefix:
            print(f"Loop reason: {state.get('loop_reason')} — injecting context into plan")
            plan = dict(plan)
            original_query = plan.get("original_query", "")
            plan["original_query"] = f"{loop_prefix}\n\nOriginal question: {original_query}"
            writer({"type": "agent_thinking", "agent": "sql_synthesis_table", "content": f"Retry/sequential context injected (loop_reason={state.get('loop_reason')})"})
        
        writer({"type": "agent_thinking", "agent": "sql_synthesis_table", "content": "Starting SQL synthesis using UC function tools..."})
        
        uc_functions = config.unity_catalog.uc_function_names
        writer({"type": "tools_available", "agent": "sql_synthesis_table", "tools": uc_functions, "content": f"Available UC functions: {', '.join(uc_functions)}"})
        
        result = sql_agent(plan, writer=writer)
        
        # Check for early completion signal in sequential mode
        if _check_no_more_queries(result):
            print("Synthesis returned NO_MORE_QUERIES — early completion")
            return {
                **_preserved_as_execution_results(state),
                "sql_queries": [],
                "next_agent": "summarize",
                "loop_reason": None,
                "sql_retry_feedback": None,
                "messages": [AIMessage(content="All sub-questions addressed — no more queries needed")],
            }
        
        sql_query = result.get("sql")
        explanation = result.get("explanation", "")
        has_sql = result.get("has_sql", False)
        
        sql_queries, query_labels = extract_sql_queries_from_agent_result(
            result, "sql_synthesis_table"
        )

        if sql_queries:
            print(f"✓ Extracted {len(sql_queries)} SQL quer{'y' if len(sql_queries) == 1 else 'ies'}")
            for i, query in enumerate(sql_queries, 1):
                label_info = f" [{query_labels[i-1]}]" if i <= len(query_labels) and query_labels[i-1] else ""
                print(f"  Query {i}{label_info} preview: {query[:100]}...")
            
            writer({"type": "sql_generated", "agent": "sql_synthesis_table", "query": sql_queries[0], "content": f"{len(sql_queries)} SQL Queries Generated"})
            
            return {
                "sql_queries": sql_queries,
                "sql_query_labels": query_labels,
                "sql_query": sql_queries[0],
                "has_sql": True,
                "sql_synthesis_explanation": explanation,
                "sql_synthesis_explanations": _append_synthesis_explanation(
                    state,
                    route="table",
                    explanation=explanation,
                    query_labels=query_labels,
                    has_sql=True,
                ),
                "next_agent": "sql_execution",
                "messages": [
                    AIMessage(content=f"SQL Synthesis (Table Route):\n{explanation}")
                ]
            }
        else:
            print("⚠ No SQL generated - agent explanation:")
            print(f"  {explanation}")
            
            writer({"type": "agent_result", "agent": "sql_synthesis_table", "result": "no_sql", "content": f"Could not generate SQL: {explanation[:150]}..."})
            
            return {
                **_preserved_as_execution_results(state),
                "synthesis_error": "Cannot generate SQL query",
                "sql_synthesis_explanation": explanation,
                "sql_synthesis_explanations": _append_synthesis_explanation(
                    state,
                    route="table",
                    explanation=explanation,
                    query_labels=query_labels,
                    has_sql=False,
                    synthesis_error="Cannot generate SQL query",
                ),
                "next_agent": "summarize",
                "messages": [
                    AIMessage(content=f"SQL Synthesis Failed (Table Route):\n{explanation}")
                ]
            }
        
    except Exception as e:
        print(f"❌ SQL synthesis failed: {e}")
        error_msg = str(e)
        return {
            **_preserved_as_execution_results(state),
            "synthesis_error": error_msg,
            "sql_synthesis_explanation": error_msg,
            "sql_synthesis_explanations": _append_synthesis_explanation(
                state,
                route="table",
                explanation=error_msg,
                has_sql=False,
                synthesis_error=error_msg,
            ),
            "messages": [
                AIMessage(content=f"SQL Synthesis Failed (Table Route):\n{error_msg}")
            ]
        }


@measure_node_time("sql_synthesis_genie")
def sql_synthesis_genie_node(state: AgentState) -> dict:
    """
    SQL synthesis node wrapping SQLSynthesisGenieAgent class.
    Supports retry and sequential modes via loop_reason state field.
    """
    writer = get_stream_writer()
    _debug_log(
        "sql_synthesis.py:sql_synthesis_genie_node:entry",
        "entered genie synthesis node",
        {
            "force_synthesis_route": state.get("force_synthesis_route"),
            "join_strategy": state.get("join_strategy"),
            "next_agent": state.get("next_agent"),
            "plan_join_strategy": state.get("plan", {}).get("join_strategy") if isinstance(state.get("plan"), dict) else None,
        },
    )
    
    print("\n" + "="*80)
    print("SQL SYNTHESIS AGENT - GENIE ROUTE")
    print("="*80)
    
    context = extract_synthesis_genie_context(state)
    
    # Get relevant spaces from state (already discovered by PlanningAgent)
    relevant_spaces = context.get("relevant_spaces", [])
    relevant_space_ids = [s.get("space_id") for s in relevant_spaces if s.get("space_id")]
    
    # Emit synthesis start event
    writer({"type": "sql_synthesis_start", "route": "genie", "spaces": relevant_space_ids})
    
    # Get configuration
    from ..core.config import get_config
    config = get_config()
    llm_endpoint = config.llm.sql_synthesis_genie_endpoint
    
    # Use dedicated SQL_SYNTHESIS_GENIE endpoint for orchestrating multiple Genie agents
    # This agent requires stronger reasoning for complex coordination
    llm = get_pooled_llm(
        llm_endpoint,
        temperature=0.1,
        max_tokens=SQL_SYNTHESIS_MAX_TOKENS,
    )
    
    if not relevant_spaces:
        print("❌ No relevant_spaces found in state")
        return {
            **_preserved_as_execution_results(state),
            "synthesis_error": "No relevant spaces available for genie route",
            "next_agent": "summarize",
        }
    
    # Use OOP agent - only creates Genie agents for relevant spaces
    if SQLSynthesisGenieAgent is None:
        raise ImportError("SQLSynthesisGenieAgent is not available. Import it from the appropriate module.")
    
    sql_agent = SQLSynthesisGenieAgent(llm, relevant_spaces)
    track_agent_model_usage("sql_synthesis_genie", llm_endpoint)
    
    # Use minimal context (already extracted)
    plan = dict(context.get("plan") or {})
    genie_route_plan = context.get("genie_route_plan") or plan.get("genie_route_plan", {})
    genie_execution_mode = context.get("genie_execution_mode") or plan.get("genie_execution_mode")
    dependency_edges = context.get("dependency_edges") or plan.get("dependency_edges")
    genie_conversation_ids = (
        context.get("genie_conversation_ids")
        or plan.get("genie_conversation_ids")
        or state.get("genie_conversation_ids")
    )
    if genie_route_plan:
        plan["genie_route_plan"] = genie_route_plan
    if genie_execution_mode:
        plan["genie_execution_mode"] = genie_execution_mode
    if dependency_edges is not None:
        plan["dependency_edges"] = dependency_edges
    if genie_conversation_ids:
        plan["genie_conversation_ids"] = genie_conversation_ids

    if not genie_route_plan:
        print("❌ No genie_route_plan found in plan")
        return {
            **_preserved_as_execution_results(state),
            "synthesis_error": "No routing plan available for genie route",
            "next_agent": "summarize",
        }

    from ..utils.genie_route_dag import (
        compute_execution_waves,
        legacy_question_map,
        normalize_genie_route_plan,
        resolve_genie_execution_mode,
    )

    try:
        # Inject retry / sequential context into the plan if looping
        loop_prefix = _build_loop_prompt_prefix(state)
        if loop_prefix:
            loop_reason = state.get("loop_reason")
            print(f"Loop reason: {loop_reason} — injecting context into plan")
            original_query = plan.get("original_query", "")

            room_info = "\n".join(
                f"  - Genie room '{s.get('space_title', 'unknown')}' (ID: {s.get('space_id')})"
                for s in relevant_spaces
            )

            if loop_reason == "sequential_next":
                # Null out the stale full-plan so the LLM doesn't re-invoke
                # parallel with all original questions. The sub-question texts in
                # genie_route_plan may not match sub_questions (different phrasing),
                # so index-based trimming is unreliable -- just disable it.
                plan["genie_route_plan"] = None
                plan["genie_execution_mode"] = None
                print("  Nulled genie_route_plan for sequential step — LLM will use individual tools")

                genie_guidance = (
                    f"\n\nGENIE ROUTE GUIDANCE (Sequential Mode):\n"
                    f"Available Genie rooms:\n{room_info}\n"
                    f"- Do NOT call invoke_parallel_genie_agents. The original plan is stale.\n"
                    f"- Use an INDIVIDUAL Genie agent tool for this single adapted query.\n"
                    f"- Choose the room most relevant to the adapted question content.\n"
                    f"  The adapted question may need a DIFFERENT room than originally assigned.\n"
                    f"- You MUST produce only ONE SQL query for this step."
                )
            elif loop_reason == "retry":
                prior_cids = plan.get("genie_conversation_ids") or {}
                cid_lines = "\n".join(
                    f"  - {sid}: {cid}" for sid, cid in prior_cids.items()
                ) or "  - (none cached yet)"
                genie_guidance = (
                    f"\n\nGENIE ROUTE GUIDANCE (Retry Mode):\n"
                    f"Available Genie rooms:\n{room_info}\n"
                    f"- Retry only the FAILED query.\n"
                    f"- Prefer SAME-SPACE Genie conversation continuity:\n"
                    f"  Cached conversation_ids:\n{cid_lines}\n"
                    f"- Pass conversation_ids on invoke_parallel_genie_agents, or "
                    f"conversation_id on the individual Genie tool for that space.\n"
                    f"- Only start a fresh conversation when switching to a different room.\n"
                    f"- Try rephrasing the question for the same room, or pick a different room.\n"
                    f"- Use individual Genie agent tools for precise control."
                )
            else:
                genie_guidance = ""

            plan["original_query"] = f"{loop_prefix}{genie_guidance}\n\nOriginal question: {original_query}"
            writer({"type": "agent_thinking", "agent": "sql_synthesis_genie", "content": f"Retry/sequential context injected (loop_reason={loop_reason})"})

        active_grp = plan.get("genie_route_plan") or {}
        if active_grp:
            normalized_grp = normalize_genie_route_plan(active_grp)
            mode = resolve_genie_execution_mode(
                plan.get("genie_execution_mode"),
                normalized_grp,
            )
            plan["genie_execution_mode"] = mode
            plan["genie_route_plan"] = normalized_grp or active_grp
            waves = compute_execution_waves(normalized_grp) if normalized_grp else []
            question_map = legacy_question_map(normalized_grp) if normalized_grp else {}
            print(
                f"Querying {len(question_map)} Genie agents "
                f"(mode={mode}, waves={len(waves)})..."
            )
            writer({
                "type": "agent_thinking",
                "agent": "sql_synthesis_genie",
                "content": (
                    f"Genie execution mode={mode}; "
                    f"{len(question_map)} space(s); {len(waves)} wave(s)"
                ),
            })
            for idx, (space_id, query) in enumerate(question_map.items(), 1):
                space_title = next(
                    (
                        s.get("space_title", space_id)
                        for s in relevant_spaces
                        if s.get("space_id") == space_id
                    ),
                    space_id,
                )
                deps = (normalized_grp.get(space_id) or {}).get("depends_on") or []
                dep_note = f" (depends_on={deps})" if deps else ""
                writer({
                    "type": "genie_agent_call",
                    "agent": "sql_synthesis_genie",
                    "space_id": space_id,
                    "space_title": space_title,
                    "query": query,
                    "depends_on": deps,
                    "genie_execution_mode": mode,
                    "content": (
                        f"[{idx}/{len(question_map)}] Genie '{space_title}'"
                        f"{dep_note}"
                    ),
                })
        else:
            print("Sequential step — LLM will pick a Genie room via individual tools")
            writer({"type": "agent_thinking", "agent": "sql_synthesis_genie", "content": "Sequential step — agent will select the appropriate Genie room"})
        
        result = sql_agent(plan, writer=writer)
        
        # Check for early completion signal in sequential mode
        if _check_no_more_queries(result):
            print("Synthesis returned NO_MORE_QUERIES — early completion")
            return {
                **_preserved_as_execution_results(state),
                "sql_queries": [],
                "next_agent": "summarize",
                "loop_reason": None,
                "sql_retry_feedback": None,
                "messages": [AIMessage(content="All sub-questions addressed — no more queries needed")],
            }
        
        sql_query = result.get("sql")
        explanation = result.get("explanation", "")
        has_sql = result.get("has_sql", False)
        from ..utils.genie_route_dag import merge_conversation_ids

        updated_conversation_ids = merge_conversation_ids(
            state.get("genie_conversation_ids"),
            plan.get("genie_conversation_ids"),
            result.get("genie_conversation_ids"),
        ) or None

        sql_queries, query_labels = extract_sql_queries_from_agent_result(
            result, "sql_synthesis_genie"
        )

        if sql_queries:
            print(f"✓ Extracted {len(sql_queries)} SQL quer{'y' if len(sql_queries) == 1 else 'ies'}")
            for i, query in enumerate(sql_queries, 1):
                label_info = f" [{query_labels[i-1]}]" if i <= len(query_labels) and query_labels[i-1] else ""
                print(f"  Query {i}{label_info} preview: {query[:100]}...")
            
            writer({"type": "sql_generated", "agent": "sql_synthesis_genie", "query": sql_queries[0], "content": f"{len(sql_queries)} SQL Queries Generated"})
            
            return {
                "sql_queries": sql_queries,
                "sql_query_labels": query_labels,
                "sql_query": sql_queries[0],
                "has_sql": True,
                "genie_conversation_ids": updated_conversation_ids,
                "sql_synthesis_explanation": explanation,
                "sql_synthesis_explanations": _append_synthesis_explanation(
                    state,
                    route="genie",
                    explanation=explanation,
                    query_labels=query_labels,
                    has_sql=True,
                ),
                "next_agent": "sql_execution",
                "messages": [
                    AIMessage(content=f"SQL Synthesis (Genie Route):\n{explanation}")
                ]
            }
        else:
            print("⚠ No SQL generated - agent explanation:")
            print(f"  {explanation}")
            
            writer({"type": "agent_result", "agent": "sql_synthesis_genie", "result": "no_sql", "content": f"Could not generate SQL from Genie agents: {explanation[:150]}..."})
            
            return {
                **_preserved_as_execution_results(state),
                "synthesis_error": "Cannot generate SQL query from Genie agent fragments",
                "genie_conversation_ids": updated_conversation_ids,
                "sql_synthesis_explanation": explanation,
                "sql_synthesis_explanations": _append_synthesis_explanation(
                    state,
                    route="genie",
                    explanation=explanation,
                    query_labels=query_labels,
                    has_sql=False,
                    synthesis_error="Cannot generate SQL query from Genie agent fragments",
                ),
                "next_agent": "summarize",
                "messages": [
                    AIMessage(content=f"SQL Synthesis Failed (Genie Route):\n{explanation}")
                ]
            }
        
    except Exception as e:
        print(f"❌ SQL synthesis failed: {e}")
        error_msg = str(e)
        return {
            **_preserved_as_execution_results(state),
            "synthesis_error": error_msg,
            "sql_synthesis_explanation": error_msg,
            "sql_synthesis_explanations": _append_synthesis_explanation(
                state,
                route="genie",
                explanation=error_msg,
                has_sql=False,
                synthesis_error=error_msg,
            ),
            "messages": [
                AIMessage(content=f"SQL Synthesis Failed (Genie Route):\n{error_msg}")
            ]
        }


# Export both functions and configuration helpers
__all__ = [
    "sql_synthesis_table_node",
    "sql_synthesis_genie_node",
    "extract_synthesis_table_context",
    "extract_synthesis_genie_context",
    "get_cached_sql_table_agent",
    "get_pooled_llm",
    "track_agent_model_usage",
    "measure_node_time"
]
