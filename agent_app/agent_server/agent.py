"""
Multi-agent Genie system async @invoke/@stream entry point for Databricks Apps.
"""

import atexit
import contextvars
import json
import logging
import os
import sys
import threading
import time
from typing import Any, AsyncGenerator, Optional
from uuid import uuid4

import litellm
import mlflow
from mlflow.entities import SpanType
from mlflow.genai.agent_server import get_request_headers, invoke, stream
from mlflow.tracing.provider import with_active_span
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from agent_server.mlflow_span_context import mlflow_span_if
from agent_server.utils import get_session_id
from agent_server.multi_agent.core.graph import create_super_agent_hybrid, get_space_context_table_name
from agent_server.multi_agent.core.state import RESET_STATE_TEMPLATE

logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)


def _enable_mlflow_langchain_autolog() -> None:
    """Best-effort autolog setup that does not break unit-test imports."""
    try:
        # Uncomment this when you also want to include autologging.
        # Many autologged spans will overlap with manual tracing, so you may
        # want to disable autologging in that case. `run_tracer_inline=True`
        # forces MLflow to process traces synchronously, which helps avoid
        # losing them in async execution.
        
        mlflow.langchain.autolog(run_tracer_inline=True)
        # mlflow.langchain.autolog(disable=True)
    except Exception as exc:
        logger.warning("Skipping mlflow.langchain.autolog setup: %s", exc)


_enable_mlflow_langchain_autolog()

PRIVACY_QUERY_POSTFIX = (
    " ALWAYS REPORT PATIENT COUNT ONLY "
    "(NEVER REVEAL PATIENT-LEVEL DATA TO AVOID PII LEAKAGE)"
)


def _append_privacy_query_postfix(query: str, enabled: bool) -> str:
    if not enabled or not query:
        return query
    if query.endswith(PRIVACY_QUERY_POSTFIX):
        return query
    return f"{query}{PRIVACY_QUERY_POSTFIX}"

# ---------------------------------------------------------------------------
# Module-level setup (replaces __init__ of SuperAgentHybridResponsesAgent)
# ---------------------------------------------------------------------------
try:
    _workflow = create_super_agent_hybrid()
except Exception as e:
    logger.warning(f"Failed to create workflow at import time: {e}")
    _workflow = None

LAKEBASE_INSTANCE_NAME = None
LAKEBASE_PROJECT = None
LAKEBASE_BRANCH = None
LAKEBASE_AUTOSCALING_ENDPOINT = None
LAKEBASE_RUNTIME_KWARGS: dict[str, str] = {}
EMBEDDING_ENDPOINT = None
EMBEDDING_DIMS = None
SPACE_CONTEXT_TABLE_NAME = None
SQL_WAREHOUSE_ID = None
try:
    from agent_server.multi_agent.core.config import get_config
    _cfg = get_config()
    LAKEBASE_INSTANCE_NAME = _cfg.lakebase.instance_name or None
    LAKEBASE_PROJECT = _cfg.lakebase.project or None
    LAKEBASE_BRANCH = _cfg.lakebase.branch or None
    LAKEBASE_AUTOSCALING_ENDPOINT = _cfg.lakebase.autoscaling_endpoint or None
    LAKEBASE_RUNTIME_KWARGS = _cfg.lakebase.runtime_kwargs()
    EMBEDDING_ENDPOINT = _cfg.lakebase.embedding_endpoint
    EMBEDDING_DIMS = _cfg.lakebase.embedding_dims
    SPACE_CONTEXT_TABLE_NAME = get_space_context_table_name(_cfg)
    SQL_WAREHOUSE_ID = _cfg.table_metadata.sql_warehouse_id
    logger.info("Using Lakebase connection: %s", _cfg.lakebase.connection_label())
except Exception as e:
    logger.warning(f"Failed to load config at import time: {e}")

_store = None
_store_lock = threading.Lock()
_compiled_workflow_app = None
_compiled_workflow_checkpointer = None
_compiled_workflow_checkpointer_exit = None
_compiled_workflow_last_used_monotonic: Optional[float] = None
_compiled_workflow_lock = threading.Lock()
_keep_warm_thread = None
_keep_warm_lock = threading.Lock()
_keep_warm_stop = threading.Event()
_RECOVERABLE_CHECKPOINTER_ERROR_MARKERS = (
    "adminshutdown",
    "terminating connection due to administrator command",
    "server closed the connection unexpectedly",
    "ssl connection has been closed unexpectedly",
    "ssl syscall error",
    "connection is closed",
    "connection not open",
    "broken pipe",
    "connection reset by peer",
    "operation timed out",
    "could not receive data from server",
    "consuming input failed",
)


def _get_store():
    """Lazy initialization of DatabricksStore for long-term memory."""
    global _store
    if _store is not None:
        return _store
    if not LAKEBASE_RUNTIME_KWARGS:
        return None
    with _store_lock:
        if _store is not None:
            return _store
        from databricks_langchain import DatabricksStore
        logger.info("Initializing DatabricksStore with Lakebase kwargs: %s", LAKEBASE_RUNTIME_KWARGS)
        store = DatabricksStore(
            **LAKEBASE_RUNTIME_KWARGS,
            embedding_endpoint=EMBEDDING_ENDPOINT,
            embedding_dims=EMBEDDING_DIMS,
        )
        store.setup()
        _store = store
        logger.info("DatabricksStore initialized")
    return _store


def _close_compiled_workflow_resources() -> None:
    """Best-effort cleanup for the long-lived compiled workflow checkpointer."""
    global _compiled_workflow_checkpointer_exit, _compiled_workflow_checkpointer
    exit_fn = _compiled_workflow_checkpointer_exit
    if exit_fn is None:
        return
    try:
        exit_fn(None, None, None)
    except Exception as exc:
        logger.warning(f"Failed to close compiled workflow checkpointer: {exc}")
    finally:
        _compiled_workflow_checkpointer_exit = None
        _compiled_workflow_checkpointer = None


def _get_workflow_cache_max_idle_seconds() -> int:
    raw_value = os.getenv("AGENT_CHECKPOINTER_MAX_IDLE_SECONDS", "300").strip() or "300"
    try:
        return max(0, int(raw_value))
    except ValueError:
        logger.warning(
            "Invalid AGENT_CHECKPOINTER_MAX_IDLE_SECONDS=%r; defaulting to 300 seconds",
            raw_value,
        )
        return 300


def _get_compiled_workflow_idle_seconds() -> Optional[float]:
    if _compiled_workflow_last_used_monotonic is None:
        return None
    return time.monotonic() - _compiled_workflow_last_used_monotonic


def _reset_compiled_workflow_app(reason: Optional[str] = None) -> None:
    """Drop cached workflow state so the next request rebuilds it."""
    global _compiled_workflow_app, _compiled_workflow_last_used_monotonic
    with _compiled_workflow_lock:
        if reason:
            logger.warning(f"Resetting cached workflow app: {reason}")
        _compiled_workflow_app = None
        _compiled_workflow_last_used_monotonic = None
        _close_compiled_workflow_resources()


def _is_recoverable_checkpointer_error(exc: Exception) -> bool:
    """Detect transient DB/checkpointer connection failures worth retrying once."""
    seen: set[int] = set()
    current: Optional[BaseException] = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = f"{type(current).__name__}: {current}".lower()
        if any(marker in message for marker in _RECOVERABLE_CHECKPOINTER_ERROR_MARKERS):
            return True
        current = current.__cause__ or current.__context__

    return False


atexit.register(_close_compiled_workflow_resources)


def _get_compiled_workflow_app(*, record_trace: bool = True):
    """Compile the workflow once and reuse it across requests."""
    global _compiled_workflow_app, _compiled_workflow_checkpointer
    global _compiled_workflow_checkpointer_exit, _compiled_workflow_last_used_monotonic
    global _workflow
    with mlflow_span_if(
        record_trace,
        name="get_compiled_workflow_app",
        span_type=SpanType.AGENT,
    ) as span:
        with _compiled_workflow_lock:
            cache_max_idle_seconds = _get_workflow_cache_max_idle_seconds()
            if _compiled_workflow_app is not None:
                idle_seconds = _get_compiled_workflow_idle_seconds()
                if (
                    cache_max_idle_seconds > 0
                    and idle_seconds is not None
                    and idle_seconds >= cache_max_idle_seconds
                ):
                    logger.warning(
                        "Cached workflow app idle for %.1f seconds (limit=%ss); recreating checkpointer",
                        idle_seconds,
                        cache_max_idle_seconds,
                    )
                    _compiled_workflow_app = None
                    _compiled_workflow_last_used_monotonic = None
                    _close_compiled_workflow_resources()
                else:
                    _compiled_workflow_last_used_monotonic = time.monotonic()
                    span.set_outputs(
                        {
                            "cache_hit": True,
                            "idle_seconds": idle_seconds,
                            "cache_max_idle_seconds": cache_max_idle_seconds,
                        }
                    )
                    return _compiled_workflow_app

            from databricks_langchain import CheckpointSaver

            logger.info("Compiling workflow app for reuse")
            try:
                with mlflow_span_if(
                    record_trace,
                    name="workflow_checkpointer_init",
                    span_type=SpanType.TOOL,
                    attributes={
                        "lakebase_instance_name": LAKEBASE_INSTANCE_NAME or "",
                        "lakebase_project": LAKEBASE_PROJECT or "",
                        "lakebase_branch": LAKEBASE_BRANCH or "",
                        "lakebase_autoscaling_endpoint": LAKEBASE_AUTOSCALING_ENDPOINT or "",
                    },
                ):
                    checkpointer_init_started = time.perf_counter()
                    cm = CheckpointSaver(**LAKEBASE_RUNTIME_KWARGS)
                logger.info(
                    "Initialized workflow checkpointer in %.1f ms",
                    (time.perf_counter() - checkpointer_init_started) * 1000,
                )
            except Exception as exc:
                logger.warning("Failed to initialize workflow checkpointer, continuing without it: %s", exc)
                cm = None
            checkpointer = cm
            exit_fn = None

            if cm is not None and hasattr(cm, "__enter__") and hasattr(cm, "__exit__"):
                with mlflow_span_if(
                    record_trace,
                    name="workflow_checkpointer_enter",
                    span_type=SpanType.TOOL,
                ):
                    checkpointer_enter_started = time.perf_counter()
                    entered = cm.__enter__()
                logger.info(
                    "Entered workflow checkpointer context in %.1f ms",
                    (time.perf_counter() - checkpointer_enter_started) * 1000,
                )
                if entered is not None:
                    checkpointer = entered
                exit_fn = cm.__exit__

            try:
                if _workflow is None:
                    _workflow = create_super_agent_hybrid()
                with mlflow_span_if(
                    record_trace,
                    name="workflow_compile",
                    span_type=SpanType.AGENT,
                ) as compile_span:
                    workflow_compile_started = time.perf_counter()
                    compiled = _workflow.compile(checkpointer=checkpointer)
                    workflow_compile_ms = (time.perf_counter() - workflow_compile_started) * 1000
                    compile_span.set_outputs({"compiled": True})
                    logger.info("Compiled workflow in %.1f ms", workflow_compile_ms)
            except Exception:
                if exit_fn is not None:
                    try:
                        exit_fn(*sys.exc_info())
                    except Exception as cleanup_exc:
                        logger.warning(
                            f"Failed to clean up workflow checkpointer after compile error: {cleanup_exc}"
                        )
                raise

            _compiled_workflow_app = compiled
            _compiled_workflow_checkpointer = checkpointer
            _compiled_workflow_checkpointer_exit = exit_fn
            _compiled_workflow_last_used_monotonic = time.monotonic()
            logger.info("Workflow app compiled and cached")
            span.set_outputs(
                {
                    "cache_hit": False,
                    "compiled": True,
                    "cache_max_idle_seconds": cache_max_idle_seconds,
                }
            )
            return _compiled_workflow_app


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_keep_warm_interval_seconds() -> int:
    raw_value = os.getenv("AGENT_KEEP_WARM_INTERVAL_SECONDS", "0").strip() or "0"
    try:
        return int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid AGENT_KEEP_WARM_INTERVAL_SECONDS=%r; disabling keep-warm",
            raw_value,
        )
        return 0


def prewarm_agent_resources(
    *,
    include_space_context: Optional[bool] = None,
    reason: str = "startup",
    record_trace: bool = True,
) -> dict[str, bool]:
    """Eagerly initialize workflow resources and optional clarification cache."""
    if include_space_context is None:
        include_space_context = _env_flag("AGENT_PREWARM_SPACE_CONTEXT", True)

    results = {"compiled_workflow": False, "space_context": False}
    with mlflow_span_if(
        record_trace,
        name="agent_prewarm",
        span_type=SpanType.TOOL,
        attributes={"reason": reason, "include_space_context": include_space_context},
    ) as span:
        _get_compiled_workflow_app(record_trace=record_trace)
        results["compiled_workflow"] = True

        if include_space_context and SPACE_CONTEXT_TABLE_NAME and SQL_WAREHOUSE_ID:
            from agent_server.multi_agent.agents.clarification import load_space_context

            load_space_context(
                table_name=SPACE_CONTEXT_TABLE_NAME,
                warehouse_id=SQL_WAREHOUSE_ID,
                record_trace=record_trace,
            )
            results["space_context"] = True

        span.set_outputs(results)
    return results


def prewarm_agent_resources_from_env() -> Optional[dict[str, bool]]:
    """Warm the workflow on startup when enabled."""
    if not _env_flag("AGENT_PREWARM_ON_STARTUP", True):
        logger.info("Startup prewarm disabled by AGENT_PREWARM_ON_STARTUP")
        return None

    logger.info("Prewarming agent resources on startup")
    return prewarm_agent_resources(reason="startup")


def _keep_warm_loop(interval_seconds: int, include_space_context: bool) -> None:
    while not _keep_warm_stop.wait(interval_seconds):
        try:
            prewarm_agent_resources(
                include_space_context=include_space_context,
                reason="periodic_keep_warm",
                record_trace=False,
            )
        except Exception as exc:
            logger.warning("Periodic keep-warm failed: %s", exc)


def start_background_keep_warm() -> None:
    """Refresh prewarmed resources periodically while the process stays alive."""
    global _keep_warm_thread

    interval_seconds = _get_keep_warm_interval_seconds()
    if interval_seconds <= 0:
        return

    include_space_context = _env_flag("AGENT_PREWARM_SPACE_CONTEXT", True)
    with _keep_warm_lock:
        if _keep_warm_thread is not None and _keep_warm_thread.is_alive():
            return
        _keep_warm_stop.clear()
        _keep_warm_thread = threading.Thread(
            target=_keep_warm_loop,
            args=(interval_seconds, include_space_context),
            name="agent-keep-warm",
            daemon=True,
        )
        _keep_warm_thread.start()
        logger.info(
            "Started periodic keep-warm thread (interval=%ss, include_space_context=%s)",
            interval_seconds,
            include_space_context,
        )


def _stop_keep_warm() -> None:
    _keep_warm_stop.set()


atexit.register(_stop_keep_warm)


# ---------------------------------------------------------------------------
# Helper: extract thread_id / user_id from request
# ---------------------------------------------------------------------------

def _get_or_create_thread_id(request: ResponsesAgentRequest) -> str:
    ci = dict(request.custom_inputs or {})
    if "thread_id" in ci:
        return ci["thread_id"]
    if request.context and getattr(request.context, "conversation_id", None):
        return request.context.conversation_id
    return str(uuid4())


def _get_user_id(request: ResponsesAgentRequest):
    if request.context and getattr(request.context, "user_id", None):
        return request.context.user_id
    if request.custom_inputs and "user_id" in request.custom_inputs:
        return request.custom_inputs["user_id"]
    return None


def _get_request_header(name: str) -> Optional[str]:
    try:
        value = get_request_headers().get(name)
    except Exception:
        return None
    return value if isinstance(value, str) and value else None


def _format_request_preview(content, limit: int = 500) -> str:
    if isinstance(content, str):
        preview = content
    else:
        try:
            preview = json.dumps(content, default=str)
        except Exception:
            preview = str(content)
    preview = " ".join(preview.split())
    if len(preview) <= limit:
        return preview
    return preview[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Helper: format custom streaming events
# ---------------------------------------------------------------------------

def _make_json_serializable(obj):
    from langchain_core.messages import BaseMessage
    from uuid import UUID

    if obj is None:
        return None
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="ignore")
        except Exception:
            return f"<bytes:{len(obj)}>"
    if isinstance(obj, set):
        return [_make_json_serializable(item) for item in obj]
    if isinstance(obj, BaseMessage):
        msg_dict = {"type": obj.__class__.__name__, "content": str(obj.content) if obj.content else ""}
        if hasattr(obj, "id") and obj.id:
            msg_dict["id"] = str(obj.id)
        if hasattr(obj, "tool_calls") and obj.tool_calls:
            msg_dict["tool_calls"] = [_make_json_serializable(tc) for tc in obj.tool_calls]
        return msg_dict
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    try:
        return str(obj)
    except Exception:
        return f"<{type(obj).__name__}>"


def _extract_merged_state(app: Any, run_config: dict, last_state: dict) -> dict:
    merged_state: dict[str, Any] = {}

    if isinstance(last_state, dict):
        merged_state.update(last_state)
        for value in last_state.values():
            if isinstance(value, dict):
                for key, nested_value in value.items():
                    merged_state.setdefault(key, nested_value)

    try:
        latest_state = app.get_state(run_config)
    except Exception:
        logger.debug("Unable to fetch merged workflow state for trace outputs", exc_info=True)
        return merged_state

    state_values = getattr(latest_state, "values", None)
    if isinstance(state_values, dict):
        merged_state.update(state_values)

    return merged_state


def _build_summary_trace_metadata(summary: Optional[str]) -> dict[str, Any]:
    if not summary:
        return {}
    workspace_count = summary.count("```viz-workspace")
    return {
        "final_summary_length": len(summary),
        "has_viz_workspace": workspace_count > 0,
        "viz_workspace_count": workspace_count,
    }


_CUSTOM_FORMATTERS = {
    "agent_thinking": lambda d: f"{d['agent'].upper()}: {d['content']}",
    "agent_start": lambda d: f"Starting {d['agent']} agent for: {d.get('query', '')}",
    "intent_detection": lambda d: f"Intent: {d['result']} - {d.get('reasoning', '')}",
    "clarity_analysis": lambda d: f"Query {'clear' if d['clear'] else 'unclear'}: {d.get('reasoning', '')}",
    "vector_search_start": lambda d: f"Searching vector index: {d['index']}",
    "vector_search_results": lambda d: f"Found {d['count']} relevant spaces",
    "plan_formulation": lambda d: f"Execution plan: {d.get('strategy', 'unknown')} strategy",
    "sql_generated": lambda d: f"SQL generated: {d.get('query', d.get('query_preview', ''))}",
    "sql_validation_start": lambda d: "Validating SQL query...",
    "sql_execution_start": lambda d: "Executing SQL query...",
    "sql_execution_complete": lambda d: f"Query complete: {d.get('rows', 0)} rows",
    "summary_start": lambda d: "Generating summary...",
    "genie_agent_call": lambda d: f"Calling Genie agent for space: {d.get('space_id', 'unknown')}",
    "meta_answer_content": lambda d: f"\n\n{d.get('content', '')}",
    "clarification_content": lambda d: f"\n\n{d.get('content', '')}",
    "clarification_result": lambda d: d.get("content", f"Clarification result: {d.get('result', 'unknown')}"),
    "tool_call_start": lambda d: d.get("content", "Calling " + d.get("tool", "tool") + "..."),
    "tool_call_end": lambda d: d.get("content", "Tool returned result"),
    "agent_step": lambda d: d.get("content", f"Step: {d.get('step', '')}"),
    "agent_result": lambda d: d.get("content", f"Result: {d.get('result', '')}"),
    "tools_available": lambda d: d.get("content", "Tools loaded"),
    "sql_synthesis_start": lambda d: f"SQL synthesis starting ({d.get('route', 'unknown')} route)",
    "summarize_step": lambda d: d.get("content", "Summarize step"),
    "summary_complete": lambda d: d.get("content", "Summary complete"),
}


def _format_custom_event(custom_data: dict) -> str:
    event_type = custom_data.get("type", "unknown")
    formatter = _CUSTOM_FORMATTERS.get(
        event_type,
        lambda d: f"info {event_type}: {json.dumps(_make_json_serializable(d), indent=2, default=str)}",
    )
    try:
        return formatter(custom_data)
    except Exception:
        return f"info {event_type}: {str(custom_data)}"


def _create_text_output_item(text: str, id: str):
    return {
        "type": "message",
        "id": id,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _create_text_delta(delta: str, id: str):
    return {
        "type": "response.output_text.delta",
        "item_id": id,
        "content_index": 0,
        "delta": delta,
    }


def _create_function_call_item(id: str, call_id: str, name: str, arguments: str):
    return {
        "type": "function_call",
        "id": id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


# ---------------------------------------------------------------------------
# @invoke / @stream entry points
# ---------------------------------------------------------------------------


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Collect all streaming events and return the final response."""
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Stream the multi-agent workflow — async wrapper around sync LangGraph execution."""
    session_id = get_session_id(request)
    thread_id = _get_or_create_thread_id(request)
    user_id = _get_user_id(request)
    trace_kind = _get_request_header("x-chat-request-kind") or "chat-turn"
    trace_source = _get_request_header("x-chat-trace-source") or "chat-route"
    original_trace_kind = _get_request_header("x-chat-original-request-kind")
    request_id = _get_request_header("x-chat-request-id") or thread_id
    user_message_id = _get_request_header("x-chat-user-message-id")
    retry_attempt_header = _get_request_header("x-chat-retry-attempt") or "0"
    try:
        retry_attempt = int(retry_attempt_header)
    except ValueError:
        retry_attempt = 0

    ci = dict(request.custom_inputs or {})
    ci["thread_id"] = thread_id
    ci["request_id"] = request_id
    if user_id:
        ci["user_id"] = user_id
    request.custom_inputs = ci

    logger.info(f"Processing request - thread_id: {thread_id}, user_id: {user_id}")

    workflow_start_time = time.time()
    first_token_time = None

    cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
    latest_query_input = cc_msgs[-1]["content"] if cc_msgs else ""
    request_preview = _format_request_preview(latest_query_input)
    if trace_kind not in ("chat-turn", "chat-fallback") and request_preview:
        request_preview = f"[{trace_kind}] {request_preview}"

    request_span = mlflow.start_span_no_context(
        name="chat_request",
        span_type=SpanType.AGENT,
        inputs={
            "thread_id": thread_id,
            "user_id": user_id,
            "request_id": request_id,
            "request_kind": trace_kind,
            "retry_attempt": retry_attempt,
            "latest_query": latest_query_input,
        },
        attributes={
            "chat.thread_id": thread_id,
            "chat.request_kind": trace_kind,
            "chat.trace_source": trace_source,
            "chat.request_id": request_id,
            "chat.retry_attempt": str(retry_attempt),
            "chat.is_retry": str(trace_kind == "chat-fallback").lower(),
            **({"chat.original_request_kind": original_trace_kind} if original_trace_kind else {}),
            **({"chat.user_id": user_id} if user_id else {}),
            **({"chat.user_message_id": user_message_id} if user_message_id else {}),
        },
    )
    with with_active_span(request_span):
        if session_id:
            mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

        mlflow.update_current_trace(
            request_preview=request_preview,
            metadata={
                "chat.thread_id": thread_id,
                "chat.request_kind": trace_kind,
                "chat.trace_source": trace_source,
                "chat.request_id": request_id,
                "chat.retry_attempt": str(retry_attempt),
                "chat.is_retry": str(trace_kind == "chat-fallback").lower(),
                **({"chat.original_request_kind": original_trace_kind} if original_trace_kind else {}),
                **({"chat.user_id": user_id} if user_id else {}),
                **({"chat.user_message_id": user_message_id} if user_message_id else {}),
            },
        )

    latest_query = latest_query_input

    run_config = {"configurable": {"thread_id": thread_id}}
    if user_id:
        run_config["configurable"]["user_id"] = user_id

    # Agent settings from UI (via custom_inputs)
    execution_mode = ci.get("execution_mode", "parallel")
    force_synthesis_route = ci.get("force_synthesis_route", "auto")
    clarification_sensitivity = ci.get("clarification_sensitivity", "medium")
    count_only = ci.get("count_only", False)
    latest_query = _append_privacy_query_postfix(latest_query, count_only)

    initial_state = {
        **RESET_STATE_TEMPLATE,
        "original_query": latest_query,
        "execution_mode": execution_mode,
        "force_synthesis_route": force_synthesis_route,
        "clarification_sensitivity": clarification_sensitivity,
        "count_only": count_only,
        "messages": [
            SystemMessage(content="""You are a multi-agent Q&A analysis system.
Your role is to help users query and analyze cross-domain data.

Guidelines:
- Always explain your reasoning and execution plan
- Validate SQL queries before execution
- Provide clear, comprehensive summaries
- If information is missing, ask for clarification (max once)
- Use UC functions and Genie agents to generate accurate SQL
- Return results with proper context and explanations"""),
            HumanMessage(content=latest_query),
        ],
    }
    if user_id:
        initial_state["user_id"] = user_id
        initial_state["thread_id"] = thread_id

    first_message = True
    seen_ids: set = set()

    import asyncio

    _SENTINEL = object()
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    parent_span = request_span

    def _run_workflow():
        """Stream events from the sync LangGraph workflow into an asyncio.Queue."""
        workflow_span = mlflow.start_span_no_context(
            name="langgraph_workflow",
            span_type=SpanType.AGENT,
            parent_span=parent_span,
            inputs={
                "original_query": latest_query,
                "thread_id": thread_id,
                "user_id": user_id,
                "request_id": request_id,
                "execution_mode": execution_mode,
                "force_synthesis_route": force_synthesis_route,
                "clarification_sensitivity": clarification_sensitivity,
                "count_only": count_only,
            },
            attributes={
                "thread_id": thread_id,
                "request_id": request_id,
                "execution_mode": execution_mode,
                "force_synthesis_route": force_synthesis_route,
                "clarification_sensitivity": clarification_sensitivity,
                "count_only": str(count_only).lower(),
            },
        )
        try:
            with with_active_span(workflow_span):
                span = workflow_span
                span.set_inputs(
                    {
                        "original_query": latest_query,
                        "thread_id": thread_id,
                        "user_id": user_id,
                        "request_id": request_id,
                        "execution_mode": execution_mode,
                        "force_synthesis_route": force_synthesis_route,
                        "clarification_sensitivity": clarification_sensitivity,
                        "count_only": count_only,
                    }
                )
                last_state: dict = {}
                from langgraph.types import Command

                for attempt in range(2):
                    attempt_emitted_events = False
                    workflow_stage = "app_load"
                    try:
                        attempt_started = time.perf_counter()
                        app_load_started = time.perf_counter()
                        app = _get_compiled_workflow_app()
                        app_load_ms = (time.perf_counter() - app_load_started) * 1000
                        logger.info(
                            "Executing cached workflow app (thread: %s, attempt: %s, app_load_ms=%.1f)",
                            thread_id,
                            attempt + 1,
                            app_load_ms,
                        )

                        workflow_stage = "state_load"
                        state_load_started = time.perf_counter()
                        existing_state = app.get_state(run_config)
                        state_load_ms = (time.perf_counter() - state_load_started) * 1000
                        logger.info(
                            "Loaded workflow state for thread %s on attempt %s in %.1f ms",
                            thread_id,
                            attempt + 1,
                            state_load_ms,
                        )
                        if existing_state.tasks and any(
                            hasattr(t, "interrupts") and t.interrupts for t in existing_state.tasks
                        ):
                            logger.info(f"Resuming from interrupt on thread {thread_id}")
                            # Resume with only the user's clarification answer.
                            # Re-sending scalar runtime overrides here can collide with
                            # the clarification subgraph's parallel fan-out and trigger
                            # INVALID_CONCURRENT_GRAPH_UPDATE on keys like execution_mode.
                            input_data = Command(resume=latest_query)
                        else:
                            input_data = initial_state

                        workflow_stage = "stream"
                        stream_started = time.perf_counter()
                        for raw_event in app.stream(
                            input_data,
                            run_config,
                            stream_mode=["updates", "messages", "custom", "tasks"],
                            subgraphs=True,
                        ):
                            # subgraphs=True: 3-tuple (ns, mode, data)
                            # or 2-tuple (ns, (mode, data)) depending on LangGraph version
                            if len(raw_event) == 3:
                                ns, mode, data = raw_event
                            elif isinstance(raw_event[0], tuple):
                                ns = raw_event[0]
                                mode, data = raw_event[1]
                            else:
                                ns, mode, data = (), raw_event[0], raw_event[1]

                            if mode == "updates" and not ns and isinstance(data, dict):
                                last_state.update(data)
                            # Only retry before any stream events were emitted to avoid duplicates.
                            if not attempt_emitted_events:
                                logger.info(
                                    "First workflow stream event for thread %s on attempt %s after %.1f ms "
                                    "(app_load_ms=%.1f, state_load_ms=%.1f)",
                                    thread_id,
                                    attempt + 1,
                                    (time.perf_counter() - stream_started) * 1000,
                                    app_load_ms,
                                    state_load_ms,
                                )
                            attempt_emitted_events = True
                            loop.call_soon_threadsafe(
                                queue.put_nowait, (ns, mode, data)
                            )
                        logger.info(
                            "Workflow stream completed for thread %s on attempt %s in %.1f ms",
                            thread_id,
                            attempt + 1,
                            (time.perf_counter() - attempt_started) * 1000,
                        )
                        break
                    except Exception as exc:
                        timeout_shaped_state_load_failure = (
                            workflow_stage == "state_load"
                            and _is_recoverable_checkpointer_error(exc)
                        )
                        if timeout_shaped_state_load_failure:
                            _reset_compiled_workflow_app(
                                reason=(
                                    f"timeout-shaped workflow state load failure on thread {thread_id}: {exc}"
                                )
                            )

                        is_retryable = (
                            attempt == 0
                            and not attempt_emitted_events
                            and _is_recoverable_checkpointer_error(exc)
                        )
                        if not is_retryable:
                            raise

                        logger.warning(
                            "Recoverable workflow checkpointer error on thread %s; "
                            "resetting cached workflow app and retrying once: %s",
                            thread_id,
                            exc,
                        )
                        if not timeout_shaped_state_load_failure:
                            _reset_compiled_workflow_app(reason=str(exc))
                        last_state = {}
                        continue

                trace_state = _extract_merged_state(app, run_config, last_state)
                workflow_output: dict = {"status": "completed"}
                if summary := trace_state.get("final_summary"):
                    workflow_output.update(_build_summary_trace_metadata(summary))
                if sql := trace_state.get("sql_query"):
                    workflow_output["sql_query"] = sql
                if er := trace_state.get("execution_result"):
                    workflow_output["execution_result_status"] = er.get("status")
                    workflow_output["row_count"] = er.get("row_count")
                span.set_outputs(workflow_output)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            try:
                workflow_span.end()
            except Exception:
                logger.debug("Failed to end workflow span cleanly", exc_info=True)
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    trace_context = contextvars.copy_context()
    loop.run_in_executor(None, trace_context.run, _run_workflow)

    progress_steps: list[str] = []
    progress_item_id = str(uuid4())
    progress_started = False
    details_finalized = False
    in_summarize = False
    streaming_item_id = str(uuid4())
    text_deltas_emitted = False
    seen_content_events: set[tuple[str, str]] = set()

    def _emit_progress_step(step_text: str):
        """Append a step to the live progress block (returns events to yield)."""
        nonlocal progress_started
        events = []
        if not progress_started:
            progress_started = True
            events.append(ResponsesAgentStreamEvent(
                **_create_text_delta(
                    delta='<details open>\n<summary>Processing Steps</summary>\n',
                    id=progress_item_id,
                ),
            ))
        events.append(ResponsesAgentStreamEvent(
            **_create_text_delta(
                delta=f'<div class="ps">{step_text}</div>\n',
                id=progress_item_id,
            ),
        ))
        return events

    def _finalize_progress():
        """Close the live progress block with a closing tag delta."""
        nonlocal details_finalized
        events = []
        if progress_started and not details_finalized:
            details_finalized = True
            events.append(ResponsesAgentStreamEvent(
                **_create_text_delta(delta="\n</details>\n\n", id=progress_item_id),
            ))
        return events

    while True:
        event = await queue.get()
        if event is _SENTINEL:
            break
        if isinstance(event, Exception):
            raise event

        # Normalised 3-tuple from _run_workflow: (namespace, mode, data)
        ns, event_type, event_data = event

        # Skip subgraph-internal updates (avoid noise and premature final-node detection)
        if event_type == "updates" and ns:
            continue

        # ── tasks: detect errors ──
        if event_type == "tasks":
            try:
                ev = event_data if isinstance(event_data, dict) else {}
                if ev.get("event") == "error":
                    node_name = ev.get("name", "unknown")
                    error = ev.get("error", "Unknown error")
                    logger.error(f"Task failed: {node_name} - {error}")
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item=_create_text_output_item(text=f"Error in {node_name}: {error}", id=str(uuid4())),
                    )
            except Exception as e:
                logger.warning(f"Error processing task event: {e}")
            continue

        # ── custom: stream progress live, handle text deltas and special events ──
        if event_type == "custom":
            try:
                et = event_data.get("type", "") if isinstance(event_data, dict) else ""
                if et == "text_delta":
                    delta_content = event_data.get("content", "")
                    if delta_content:
                        if not text_deltas_emitted:
                            for ev in _finalize_progress():
                                yield ev
                        if first_token_time is None:
                            first_token_time = time.time()
                            logger.info(f"TTFT: {first_token_time - workflow_start_time:.3f}s")
                        text_deltas_emitted = True
                        yield ResponsesAgentStreamEvent(
                            **_create_text_delta(delta=delta_content, id=streaming_item_id),
                        )
                elif et in ("meta_answer_content", "clarification_content", "clarification_requested"):
                    content_key = (et, str(event_data.get("content", "")) if isinstance(event_data, dict) else str(event_data))
                    if content_key in seen_content_events:
                        continue
                    seen_content_events.add(content_key)
                    if not details_finalized:
                        for ev in _finalize_progress():
                            yield ev
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item=_create_text_output_item(text=_format_custom_event(event_data), id=str(uuid4())),
                    )
                    if et == "clarification_content" and isinstance(event_data, dict):
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=_create_text_output_item(text="", id=str(uuid4())),
                            databricks_output={
                                "clarification": {
                                    "reason": event_data.get("reason", ""),
                                    "options": event_data.get("options", []),
                                }
                            },
                        )
                elif et.endswith("_error") or et == "error":
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item=_create_text_output_item(text=_format_custom_event(event_data), id=str(uuid4())),
                    )
                elif et == "code_enrichment_progress":
                    step_text = event_data.get("content", "")
                    if step_text:
                        progress_steps.append(step_text)
                        for ev in _emit_progress_step(step_text):
                            yield ev
                elif et == "summary_start":
                    in_summarize = True
                    streaming_item_id = str(uuid4())
                    text_deltas_emitted = False
                elif et == "summary_complete":
                    pass
                else:
                    step_text = _format_custom_event(event_data)
                    progress_steps.append(step_text)
                    for ev in _emit_progress_step(step_text):
                        yield ev
            except Exception as e:
                logger.warning(f"Error processing custom event: {e}")
            continue

        # ── messages: skip (text_delta custom events handle all streaming) ──
        if event_type == "messages":
            continue

        # ── updates: stream node progress, only emit AIMessages from final nodes ──
        if event_type == "updates":
            events_dict = event_data
            if isinstance(events_dict, tuple):
                flattened = {}
                for item in events_dict:
                    if isinstance(item, dict):
                        flattened.update(item)
                events_dict = flattened
            if not isinstance(events_dict, dict):
                logger.warning(f"Skipping malformed updates event: {type(event_data).__name__}")
                continue
            node_names = list(events_dict.keys())
            new_msgs = [
                msg
                for v in events_dict.values()
                if isinstance(v, dict)
                for msg in v.get("messages", [])
                if hasattr(msg, "id") and msg.id not in seen_ids
            ]

            seen_ids.update(msg.id for msg in new_msgs)

            routing_nodes = {
                "planning",
                "sql_synthesis_table",
                "sql_synthesis_genie",
                "sql_execution",
            }
            for node_name, update in events_dict.items():
                if not isinstance(update, dict):
                    continue
                if node_name != "summarize":
                    step = f"Step: {node_name}"
                    extra = [k for k in update if k != "messages"]
                    if extra:
                        step += f" ({', '.join(extra)})"
                    progress_steps.append(step)
                    for ev in _emit_progress_step(step):
                        yield ev
                    if node_name in routing_nodes and "next_agent" in update:
                        routing = f"Routing → {update['next_agent']}"
                        progress_steps.append(routing)
                        for ev in _emit_progress_step(routing):
                            yield ev

            is_final_node = "summarize" in node_names
            if not is_final_node:
                for update in events_dict.values():
                    if not isinstance(update, dict):
                        continue
                    if update.get("is_meta_question") or update.get("is_irrelevant"):
                        is_final_node = True
                        break

            if is_final_node:
                for ev in _finalize_progress():
                    yield ev
                if not text_deltas_emitted:
                    for msg in new_msgs:
                        if isinstance(msg, AIMessage):
                            for se in output_to_responses_items_stream([msg]):
                                yield se
            continue

    for ev in _finalize_progress():
        yield ev

    ttcl = time.time() - workflow_start_time
    logger.info(
        f"Workflow completed (thread: {thread_id}) "
        f"TTFT={first_token_time - workflow_start_time if first_token_time else 'N/A'}s, TTCL={ttcl:.3f}s"
    )
    final_state = {}
    try:
        final_app = _get_compiled_workflow_app()
        final_state = _extract_merged_state(final_app, run_config, {})
    except Exception:
        logger.debug("Unable to fetch final workflow state for request span outputs", exc_info=True)

    request_output = {
        "thread_id": thread_id,
        "request_id": request_id,
        "ttft_seconds": first_token_time - workflow_start_time if first_token_time else None,
        "ttcl_seconds": ttcl,
        "first_token_emitted": first_token_time is not None,
    }
    if final_summary := final_state.get("final_summary"):
        request_output.update(_build_summary_trace_metadata(final_summary))
    request_span.set_outputs(request_output)
    request_span.end()
