"""
Planning agent node for the multi-agent system.

This module contains the planning_node function which is responsible for:
- Analyzing user queries
- Searching for relevant Genie spaces using vector search
- Creating execution plans with join strategies
- Determining the next agent in the workflow

The planning node uses turn-based context management and includes optimizations
for vector search result caching and token usage reduction.

Dependencies:
- PlanningAgent class: Must be available via get_cached_planning_agent()
- Configuration: VECTOR_SEARCH_INDEX and LLM_ENDPOINT_PLANNING must be set
  (either imported from config module or set directly)

Example usage:
    from src.multi_agent.agents.planning import planning_node
    from src.multi_agent.core.state import AgentState
    
    # Set configuration
    from src.multi_agent.agents.planning import VECTOR_SEARCH_INDEX, LLM_ENDPOINT_PLANNING
    VECTOR_SEARCH_INDEX = "catalog.schema.index_name"
    LLM_ENDPOINT_PLANNING = "databricks-claude-sonnet-4-5"
    
    # Use the planning node
    result = planning_node(state)
"""

import json
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from functools import wraps

from langchain_core.messages import SystemMessage
from langgraph.config import get_stream_writer

from ..core.state import AgentState


logger = logging.getLogger(__name__)


def _debug_log(location: str, message: str, data: Optional[dict] = None) -> None:
    """Emit routing/planning debug info via the standard logger.

    Previously this wrote to a hard-coded developer path on every call, which
    leaked routing data locally and silently failed everywhere else.
    """
    logger.debug("%s | %s | %s", location, message, data or {})


# ==============================================================================
# Module-level caches and configuration
# ==============================================================================

# Vector search result cache (for refinement queries)
# Bounded OrderedDict: evicts expired + oldest entries when full.
_vector_search_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
_vector_search_cache_lock = threading.Lock()
_VECTOR_SEARCH_CACHE_TTL = timedelta(minutes=10)
_VS_CACHE_MAX_SIZE = 200


def _vs_cache_evict_expired() -> None:
    """Remove all expired entries. Caller must hold _vector_search_cache_lock."""
    now = datetime.now()
    expired = [
        k for k, v in _vector_search_cache.items()
        if now - v["timestamp"] >= _VECTOR_SEARCH_CACHE_TTL
    ]
    for k in expired:
        del _vector_search_cache[k]


def _vs_cache_get(thread_id: str) -> Optional[Dict[str, Any]]:
    """Thread-safe TTL-aware cache lookup. Returns entry or None."""
    with _vector_search_cache_lock:
        entry = _vector_search_cache.get(thread_id)
        if entry is None:
            return None
        if datetime.now() - entry["timestamp"] >= _VECTOR_SEARCH_CACHE_TTL:
            del _vector_search_cache[thread_id]
            return None
        _vector_search_cache.move_to_end(thread_id)
        return entry


def _vs_cache_put(thread_id: str, query: str, results: List) -> None:
    """Thread-safe bounded cache insert. Evicts oldest if at capacity."""
    with _vector_search_cache_lock:
        if thread_id in _vector_search_cache:
            del _vector_search_cache[thread_id]
        elif len(_vector_search_cache) >= _VS_CACHE_MAX_SIZE:
            _vs_cache_evict_expired()
            while len(_vector_search_cache) >= _VS_CACHE_MAX_SIZE:
                _vector_search_cache.popitem(last=False)
        _vector_search_cache[thread_id] = {
            "query": query,
            "results": results,
            "timestamp": datetime.now(),
        }

# Performance metrics tracking
_performance_metrics: Dict[str, Any] = {
    "cache_stats": {},
    "node_timings": {},
    "agent_model_usage": {}
}


# ==============================================================================
# Configuration constants (should be imported from config module)
# ==============================================================================

# These should be imported from a config module in production
# For now, they are placeholders that need to be set
VECTOR_SEARCH_INDEX: Optional[str] = None
LLM_ENDPOINT_PLANNING: Optional[str] = None


# ==============================================================================
# Helper Functions
# ==============================================================================

# Agent cache (module-level)
_agent_cache = {}

def get_cached_planning_agent():
    """
    Get or create cached PlanningAgent instance.
    Expected gain: -500ms to -1s per request
    """
    global LLM_ENDPOINT_PLANNING, VECTOR_SEARCH_INDEX
    
    # Lazy load config if not set
    if LLM_ENDPOINT_PLANNING is None or VECTOR_SEARCH_INDEX is None:
        try:
            from ..core.config import get_config
            config = get_config()
            if LLM_ENDPOINT_PLANNING is None:
                LLM_ENDPOINT_PLANNING = config.llm.planning_endpoint
            if VECTOR_SEARCH_INDEX is None:
                VECTOR_SEARCH_INDEX = config.vs_index_fq
        except Exception as e:
            print(f"⚠️ Failed to load config: {e}")
            
    if "planning" not in _agent_cache:
        record_cache_miss("agent_cache")
        print("⚡ Creating PlanningAgent (first use)...")
        
        try:
            from databricks_langchain import ChatDatabricks
            from .planning_agent import PlanningAgent
            
            if LLM_ENDPOINT_PLANNING is None:
                raise ValueError("LLM_ENDPOINT_PLANNING must be configured")
            if VECTOR_SEARCH_INDEX is None:
                raise ValueError("VECTOR_SEARCH_INDEX must be configured")
                
            llm = ChatDatabricks(endpoint=LLM_ENDPOINT_PLANNING, temperature=0.1)
            _agent_cache["planning"] = PlanningAgent(llm, VECTOR_SEARCH_INDEX)
            print("✓ PlanningAgent cached")
        except ImportError:
            raise ImportError(
                "Failed to import required dependencies (ChatDatabricks or PlanningAgent)."
            )
    else:
        record_cache_hit("agent_cache")
        print("✓ Using cached PlanningAgent")
        
    return _agent_cache["planning"]

def extract_planning_context(state: AgentState) -> dict:
    """Extract minimal context for planning."""
    return {
        "current_turn": state.get("current_turn"),
        "original_query": state.get("original_query")  # Backward compat
    }


def measure_node_time(node_name: str):
    """
    Decorator to measure node execution time.
    Expected use: Track per-node performance for optimization.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
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


def track_agent_model_usage(agent_name: str, model_endpoint: str):
    """
    Track which LLM model is used by each agent for monitoring and cost analysis.
    
    Args:
        agent_name: Name of the agent (e.g., "clarification", "planning")
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


def record_cache_hit(cache_type: str):
    """Record a cache hit for monitoring."""
    key = f"{cache_type}_hits"
    if key not in _performance_metrics["cache_stats"]:
        _performance_metrics["cache_stats"][key] = 0
    _performance_metrics["cache_stats"][key] += 1


def record_cache_miss(cache_type: str):
    """Record a cache miss for monitoring."""
    key = f"{cache_type}_misses"
    if key not in _performance_metrics["cache_stats"]:
        _performance_metrics["cache_stats"][key] = 0
    _performance_metrics["cache_stats"][key] += 1


# ==============================================================================
# Planning Node Function
# ==============================================================================

@measure_node_time("planning")
def planning_node(state: AgentState) -> dict:
    """
    Planning node wrapping PlanningAgent class using turn-based context.
    
    IMPROVEMENTS:
    - Uses current_turn.context_summary (LLM-generated) instead of manual combined_query_context
    - Intent-aware planning (different strategies for refinements vs new questions)
    - Clean separation from clarification logic
    
    OPTIMIZED: Uses minimal state extraction to reduce token usage
    
    Returns: Dictionary with only the state updates (for clean MLflow traces)
    
    Note: This function requires PlanningAgent class and get_cached_planning_agent function
    to be available. These should be imported from appropriate modules or defined elsewhere.
    """
    writer = get_stream_writer()
    
    print("\n" + "="*80)
    print("📋 PLANNING AGENT (Token Optimized)")
    print("="*80)
    
    # OPTIMIZATION: Extract only minimal context needed for planning
    context = extract_planning_context(state)
    print(f"📊 State optimization: Using {len(context)} fields (vs {len([k for k in state.keys() if state.get(k) is not None])} in full state)")
    
    # Get current turn from state
    current_turn = context.get("current_turn")
    if not current_turn:
        print("No current_turn found, falling back to legacy behavior")
        query = context.get("original_query", "")
        is_followup = False
        context_summary = None
    else:
        query = current_turn["query"]
        is_followup = bool(current_turn.get("parent_turn_id"))
        context_summary = current_turn.get("context_summary")
    
    # Use context_summary if available (LLM-generated from intent detection)
    # This replaces the manual combined_query_context template
    planning_query = context_summary or query
    
    # Emit agent start event
    writer({"type": "agent_start", "agent": "planning", "query": planning_query})
    
    print(f"Query: {query}")
    print(f"Is follow-up: {is_followup}")
    if context_summary:
        print(f"✓ Using context summary from intent detection")
        print(f"  Summary: {context_summary[:200]}...")
    else:
        print(f"✓ Using query directly (no context needed)")
    
    # OPTIMIZATION: Use cached agent instance
    # NOTE: get_cached_planning_agent must be imported or defined elsewhere
    # This function should return a cached PlanningAgent instance
    try:
        planning_agent = get_cached_planning_agent()
    except Exception as e:
        # Fallback: PlanningAgent and get_cached_planning_agent need to be provided
        # This is a placeholder - in production, these should be properly imported
        raise ImportError(
            f"Failed to get planning agent: {e}"
        )
    track_agent_model_usage("planning", LLM_ENDPOINT_PLANNING)
    
    # PHASE 2 OPTIMIZATION: Vector search result caching for follow-ups
    thread_id = state.get("thread_id", "default")
    can_reuse_cache = is_followup
    
    relevant_spaces_full = None
    cache_hit = False

    if can_reuse_cache:
        cache_entry = _vs_cache_get(thread_id)
        if cache_entry is not None:
            record_cache_hit("vector_search")
            relevant_spaces_full = cache_entry["results"]
            cache_hit = True
            cache_age = datetime.now() - cache_entry["timestamp"]
            print(f"VECTOR SEARCH CACHE HIT (thread: {thread_id}, age: {cache_age.seconds}s)")
            print(f"   Reusing {len(relevant_spaces_full)} spaces for follow-up query")
            print(f"   Expected gain: -300 to -800ms")

            writer({
                "type": "vector_search_cache_hit",
                "thread_id": thread_id,
                "is_followup": is_followup,
                "space_count": len(relevant_spaces_full)
            })

    if relevant_spaces_full is None:
        record_cache_miss("vector_search")
        writer({"type": "vector_search_start", "index": VECTOR_SEARCH_INDEX})

        print(f"🔍 Performing vector search (cache miss or new question)...")
        relevant_spaces_full = planning_agent.search_relevant_spaces(planning_query)

        _vs_cache_put(thread_id, planning_query, relevant_spaces_full)
        print(f"✓ Cached vector search results for thread: {thread_id}")
    
    # Emit vector search results
    writer({"type": "vector_search_results", "spaces": relevant_spaces_full, "count": len(relevant_spaces_full)})
    
    # Emit plan formulation start
    writer({"type": "agent_thinking", "agent": "planning", "content": "Creating execution plan..."})
    
    # Create execution plan
    # IMPORTANT: Use planning_query (with context_summary) not just query
    # Pass original_query so it can be shown in the prompt before context_summary
    force_route = state.get("force_synthesis_route", "auto")
    plan = planning_agent.create_execution_plan(
        planning_query,
        relevant_spaces_full,
        original_query=query,
        force_synthesis_route=force_route,
    )
    
    # Extract plan components
    join_strategy = plan.get("join_strategy")
    
    # Honor force_synthesis_route override from UI
    if force_route == "table_route":
        join_strategy = "table_route"
        next_agent = "sql_synthesis_table"
        print(f"✓ Plan complete - FORCED TABLE ROUTE (UI override)")
    elif force_route == "genie_route":
        join_strategy = "genie_route"
        next_agent = "sql_synthesis_genie"
        print(f"✓ Plan complete - FORCED GENIE ROUTE (UI override)")
    elif join_strategy == "genie_route":
        next_agent = "sql_synthesis_genie"
        print("✓ Plan complete - using GENIE ROUTE (Genie agents)")
    else:
        next_agent = "sql_synthesis_table"
        print("✓ Plan complete - using TABLE ROUTE (direct SQL synthesis)")

    # Finalize Genie planner fields (mode, structured route plan, dependency_edges).
    # Distinct from UI SQL execution_mode. Idempotent with planning_agent finalize.
    from ..utils.genie_route_dag import (
        finalize_planner_genie_plan,
        normalize_genie_route_plan,
        summarize_plan_for_logging,
    )

    plan = dict(plan)
    # Align join_strategy with the resolved route before finalize so Genie
    # fields are cleared on table_route (including UI force overrides).
    plan["join_strategy"] = (
        "genie_route" if next_agent == "sql_synthesis_genie" else "table_route"
    )
    plan = finalize_planner_genie_plan(plan)
    genie_route_plan = plan.get("genie_route_plan")
    genie_execution_mode = plan.get("genie_execution_mode")
    dependency_edges = plan.get("dependency_edges") or []

    from ..utils.join_contract import (
        build_join_contract_from_plan,
        join_contract_summary,
    )

    join_contract = build_join_contract_from_plan(
        plan,
        relevant_spaces=relevant_spaces_full,
    )
    plan["join_contract"] = join_contract

    if next_agent == "sql_synthesis_genie" and genie_route_plan:
        normalized_grp = normalize_genie_route_plan(genie_route_plan)
        print(
            "  Genie plan finalized: "
            f"{json.dumps(summarize_plan_for_logging(normalized_grp, genie_execution_mode or 'parallel'))}"
        )
    print(f"  Join contract seeded: {json.dumps(join_contract_summary(join_contract))}")

    _debug_log(
        "planning.py:planning_node:route_decision",
        "planning route resolved",
        {
            "force_synthesis_route": force_route,
            "plan_join_strategy": plan.get("join_strategy"),
            "resolved_join_strategy": join_strategy,
            "next_agent": next_agent,
            "genie_execution_mode": genie_execution_mode,
            "dependency_edge_count": len(dependency_edges),
            "join_contract": join_contract_summary(join_contract),
            "relevant_space_count": len(relevant_spaces_full),
        },
    )
    
    # Emit plan formulation result
    writer({
        "type": "plan_formulation",
        "force_route": force_route,
        "strategy": join_strategy,
        "requires_join": plan.get("requires_join", False),
        "genie_execution_mode": genie_execution_mode,
        "dependency_edges": dependency_edges,
        "join_contract": join_contract_summary(join_contract),
    })
    
    sub_questions = plan.get("sub_questions", [])
    
    # Return only updates (no in-place modifications)
    return {
        "plan": plan,
        "sub_questions": sub_questions,
        "total_sub_questions": len(sub_questions),
        "requires_multiple_spaces": plan.get("requires_multiple_spaces", False),
        "relevant_space_ids": plan.get("relevant_space_ids", []),
        "requires_join": plan.get("requires_join", False),
        "join_strategy": join_strategy,
        "join_strategy_route": next_agent,
        "execution_plan": plan.get("execution_plan", ""),
        "genie_route_plan": genie_route_plan,
        "genie_execution_mode": genie_execution_mode,
        "dependency_edges": dependency_edges,
        "join_contract": join_contract,
        "vector_search_relevant_spaces_info": plan.get("vector_search_relevant_spaces_info", []),
        "relevant_spaces": relevant_spaces_full,
        "next_agent": next_agent,
        "messages": [
            SystemMessage(content=f"Execution plan: {json.dumps(plan, indent=2)}")
        ]
    }


# ==============================================================================
# Cache Management Functions
# ==============================================================================

def clear_vector_search_cache(thread_id: str = None):
    """Clear vector search cache for a specific thread or all threads."""
    with _vector_search_cache_lock:
        if thread_id:
            if thread_id in _vector_search_cache:
                del _vector_search_cache[thread_id]
                print(f"✓ Vector search cache cleared for thread: {thread_id}")
        else:
            _vector_search_cache.clear()
            print("✓ All vector search caches cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring."""
    with _vector_search_cache_lock:
        size = len(_vector_search_cache)
        threads = list(_vector_search_cache.keys())
    return {
        "vector_search_cache_size": size,
        "vector_search_cache_max_size": _VS_CACHE_MAX_SIZE,
        "vector_search_threads": threads,
        "performance_metrics": _performance_metrics,
    }
