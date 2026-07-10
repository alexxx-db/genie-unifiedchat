"""
LangGraph workflow construction for the multi-agent system.

This module defines the graph structure, routing logic, and workflow compilation.
"""

import json
import logging
from functools import wraps
from typing import Any, Callable, Optional, Union

import mlflow
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from mlflow.entities import SpanType

logger = logging.getLogger(__name__)

from ..core.state import AgentState
from ..agents.clarification import ClarificationAgent
from ..agents.planning import planning_node
from ..agents.sql_synthesis import sql_synthesis_table_node, sql_synthesis_genie_node
from ..agents.sql_execution import sql_execution_node
from ..agents.summarize import summarize_node


def _trace_state_snapshot(payload: Any) -> dict[str, Any]:
    """Capture agent state for traces without repeating full transcripts on each span."""
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}

    from langchain_core.messages import BaseMessage
    MAX_RECENT_MESSAGES = 5
    MAX_RESULT_SAMPLE_ROWS = 5
    MAX_SQL_PREVIEW_CHARS = 2000

    def _preview_text(value: Any, max_chars: int = 400) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, default=str)
            except Exception:
                text = str(value)
        return text if len(text) <= max_chars else f"{text[:max_chars]}... [truncated]"

    def _serialize_message(message: Any) -> Any:
        if isinstance(message, BaseMessage):
            message_dict: dict[str, Any] = {
                "type": message.__class__.__name__,
                "content_preview": _preview_text(message.content),
                "content_length": len(_preview_text(message.content, max_chars=100000)),
            }
            if getattr(message, "id", None):
                message_dict["id"] = str(message.id)
            if getattr(message, "name", None):
                message_dict["name"] = message.name
            if getattr(message, "tool_calls", None):
                message_dict["tool_call_count"] = len(message.tool_calls)
            if getattr(message, "additional_kwargs", None):
                message_dict["additional_kwargs_keys"] = sorted(
                    list(message.additional_kwargs.keys())
                )
            return message_dict
        return {
            "type": type(message).__name__,
            "content_preview": _preview_text(message),
            "content_length": len(_preview_text(message, max_chars=100000)),
        }

    def _summarize_messages(messages: list[Any]) -> dict[str, Any]:
        serialized_messages = [_serialize_message(message) for message in messages]
        last_user_message = next(
            (
                message
                for message in reversed(serialized_messages)
                if message.get("type") == "HumanMessage"
            ),
            None,
        )
        last_assistant_message = next(
            (
                message
                for message in reversed(serialized_messages)
                if message.get("type") in ("AIMessage", "AIMessageChunk")
            ),
            None,
        )

        return {
            "count": len(messages),
            "recent_messages": serialized_messages[-MAX_RECENT_MESSAGES:],
            "last_user_message_preview": (
                last_user_message.get("content_preview") if last_user_message else None
            ),
            "last_assistant_message_preview": (
                last_assistant_message.get("content_preview")
                if last_assistant_message
                else None
            ),
            "message_ids": [
                message["id"] for message in serialized_messages if message.get("id")
            ][-MAX_RECENT_MESSAGES:],
            "truncated": len(messages) > MAX_RECENT_MESSAGES,
        }

    def _summarize_execution_result(result: Any) -> Any:
        if not isinstance(result, dict):
            return result

        sample_rows = result.get("result")
        sample_rows_summary = None
        if isinstance(sample_rows, list):
            sample_rows_summary = sample_rows[:MAX_RESULT_SAMPLE_ROWS]

        sql_text = result.get("sql")
        sql_preview = (
            _preview_text(sql_text, max_chars=MAX_SQL_PREVIEW_CHARS)
            if isinstance(sql_text, str)
            else None
        )

        return {
            "status": result.get("status"),
            "success": result.get("success"),
            "query_number": result.get("query_number"),
            "query_label": result.get("query_label"),
            "row_count": result.get("row_count"),
            "columns": result.get("columns"),
            "column_count": len(result.get("columns", []) or []),
            "sql_preview": sql_preview,
            "sql_length": len(sql_text) if isinstance(sql_text, str) else None,
            "sql_truncated": isinstance(sql_text, str)
            and len(sql_text) > MAX_SQL_PREVIEW_CHARS,
            "error_preview": _preview_text(result.get("error"))
            if result.get("error")
            else None,
            "skip_reason_preview": _preview_text(result.get("skip_reason"))
            if result.get("skip_reason")
            else None,
            "sql_explanation_preview": _preview_text(result.get("sql_explanation"))
            if result.get("sql_explanation")
            else None,
            "row_grain_hint": result.get("row_grain_hint"),
            "result_sample_rows": sample_rows_summary,
            "result_sample_row_count": len(sample_rows_summary)
            if sample_rows_summary is not None
            else None,
            "result_total_row_count": len(sample_rows)
            if isinstance(sample_rows, list)
            else None,
            "result_payload_truncated": isinstance(sample_rows, list)
            and len(sample_rows) > MAX_RESULT_SAMPLE_ROWS,
        }

    snapshot: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "messages":
            if isinstance(value, list):
                snapshot["messages"] = _summarize_messages(value)
            else:
                snapshot["messages"] = {"payload_type": type(value).__name__}
        elif key == "execution_result":
            snapshot["execution_result"] = _summarize_execution_result(value)
        elif key == "execution_results":
            if isinstance(value, list):
                snapshot["execution_results"] = [
                    _summarize_execution_result(result) for result in value
                ]
            else:
                snapshot["execution_results"] = {"payload_type": type(value).__name__}
        else:
            snapshot[key] = value
    return snapshot


def _sync_turn_metadata(state: Any, result: Any) -> None:
    """Attach turn identifiers to the active trace once intent detection resolves them."""
    current_turn = None
    if isinstance(result, dict):
        current_turn = result.get("current_turn")
    if not isinstance(current_turn, dict) and isinstance(state, dict):
        current_turn = state.get("current_turn")
    if not isinstance(current_turn, dict):
        return

    trace_metadata = {}
    if turn_id := current_turn.get("turn_id"):
        trace_metadata["chat.turn_id"] = turn_id
    if parent_turn_id := current_turn.get("parent_turn_id"):
        trace_metadata["chat.parent_turn_id"] = parent_turn_id

    if trace_metadata and mlflow.get_current_active_span() is not None:
        mlflow.update_current_trace(metadata=trace_metadata)


def _with_node_trace(
    node_name: str,
    node_fn: Callable[[AgentState], Any],
    span_type: str = SpanType.AGENT,
) -> Callable[[AgentState], Any]:
    """Wrap a LangGraph node in a manual MLflow span."""

    @wraps(node_fn)
    def traced_node(state: AgentState) -> Any:
        with mlflow.start_span(
            name=node_name,
            span_type=span_type,
            attributes={"langgraph.node": node_name},
        ) as span:
            span.set_inputs(_trace_state_snapshot(state))
            try:
                result = node_fn(state)
            except Exception as exc:
                span.set_status("ERROR")
                span.set_attribute("error.type", type(exc).__name__)
                span.set_attribute("error.message", str(exc))
                raise

            span.set_outputs(_trace_state_snapshot(result))
            _sync_turn_metadata(state, result)
            return result

    return traced_node


def create_super_agent_hybrid(config=None) -> StateGraph:
    """
    Create the Hybrid Super Agent LangGraph workflow.

    Args:
        config: Optional configuration object (uses default if None)

    Returns:
        StateGraph: Uncompiled LangGraph workflow
    """
    if config is None:
        from .config import get_config
        config = get_config()

    table_name = get_space_context_table_name(config)
    clarification_agent = ClarificationAgent(
        llm_endpoint=config.llm.clarification_endpoint,
        table_name=table_name,
        warehouse_id=config.table_metadata.sql_warehouse_id,
    )

    logger.info("Building hybrid super agent workflow")

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "unified_intent_context_clarification",
        _with_node_trace(
            "unified_intent_context_clarification",
            clarification_agent.run,
            SpanType.AGENT,
        ),
    )
    workflow.add_node("planning", _with_node_trace("planning", planning_node, SpanType.AGENT))
    workflow.add_node(
        "sql_synthesis_table",
        _with_node_trace("sql_synthesis_table", sql_synthesis_table_node, SpanType.AGENT),
    )
    workflow.add_node(
        "sql_synthesis_genie",
        _with_node_trace("sql_synthesis_genie", sql_synthesis_genie_node, SpanType.AGENT),
    )
    workflow.add_node(
        "sql_execution",
        _with_node_trace("sql_execution", sql_execution_node, SpanType.TOOL),
    )
    workflow.add_node("summarize", _with_node_trace("summarize", summarize_node, SpanType.AGENT))

    def route_after_unified(state: AgentState) -> str:
        """Route after unified node: planning or END (meta-question/irrelevant).

        Clarification no longer routes to END here -- unclear queries pause inside
        the subgraph via interrupt() and resume directly into planning.
        """
        if state.get("is_irrelevant", False):
            return END
        if state.get("is_meta_question", False):
            return END
        return "planning"

    def route_after_planning(state: AgentState) -> str:
        """Route after planning: determine SQL synthesis route or direct summarize"""
        next_agent = state.get("next_agent", "summarize")
        if next_agent == "sql_synthesis_table":
            return "sql_synthesis_table"
        elif next_agent == "sql_synthesis_genie":
            return "sql_synthesis_genie"
        return "summarize"

    def route_after_synthesis(state: AgentState) -> str:
        """Route after SQL synthesis: execution or summarize (if error)"""
        next_agent = state.get("next_agent", "summarize")
        if next_agent == "sql_execution":
            return "sql_execution"
        return "summarize"

    workflow.set_entry_point("unified_intent_context_clarification")

    workflow.add_conditional_edges(
        "unified_intent_context_clarification",
        route_after_unified,
        {
            "planning": "planning",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "planning",
        route_after_planning,
        {
            "sql_synthesis_table": "sql_synthesis_table",
            "sql_synthesis_genie": "sql_synthesis_genie",
            "summarize": "summarize"
        }
    )

    workflow.add_conditional_edges(
        "sql_synthesis_table",
        route_after_synthesis,
        {
            "sql_execution": "sql_execution",
            "summarize": "summarize"
        }
    )

    workflow.add_conditional_edges(
        "sql_synthesis_genie",
        route_after_synthesis,
        {
            "sql_execution": "sql_execution",
            "summarize": "summarize"
        }
    )

    def route_after_execution(state: AgentState) -> str:
        next_agent = state.get("next_agent", "summarize")
        if next_agent in ("sql_synthesis_table", "sql_synthesis_genie"):
            return next_agent
        return "summarize"

    workflow.add_conditional_edges(
        "sql_execution",
        route_after_execution,
        {
            "sql_synthesis_table": "sql_synthesis_table",
            "sql_synthesis_genie": "sql_synthesis_genie",
            "summarize": "summarize",
        },
    )

    workflow.add_edge("summarize", END)

    logger.info(
        "Workflow nodes added: intent/context/clarification, planning, "
        "sql-synthesis (table + genie routes), sql-execution, summarize (final). "
        "Conditional routing configured; all paths route to summarize before END."
    )
    logger.info("Hybrid super agent workflow created successfully")

    return workflow


def get_space_context_table_name(config) -> str:
    """Return the UC table used to store Genie space summary chunks."""
    return (
        f"{config.unity_catalog.catalog_name}"
        f".{config.unity_catalog.schema_name}"
        f".{config.table_metadata.source_table}"
    )


def create_agent_graph(
    config=None,
    with_checkpointer: Union[bool, str] = False,
):
    """
    Create and optionally compile the agent graph.

    Args:
        config: Optional configuration object (uses default if None)
        with_checkpointer:
            False  -- return uncompiled StateGraph (serving path compiles at runtime)
            True   -- compile with DatabricksCheckpointSaver (CLI on Databricks)
            "memory" -- compile with MemorySaver (local dev without Lakebase)

    Returns:
        StateGraph or CompiledStateGraph depending on with_checkpointer
    """
    workflow = create_super_agent_hybrid(config)

    if with_checkpointer is True:
        from databricks_langchain.checkpoint import DatabricksCheckpointSaver
        from databricks.sdk import WorkspaceClient

        if config is None:
            from .config import get_config
            config = get_config()

        w = WorkspaceClient()
        checkpointer = DatabricksCheckpointSaver(
            w.lakebase, database_instance_name=config.lakebase.instance_name
        )
        return workflow.compile(checkpointer=checkpointer)

    if with_checkpointer == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return workflow.compile(checkpointer=MemorySaver())

    return workflow
