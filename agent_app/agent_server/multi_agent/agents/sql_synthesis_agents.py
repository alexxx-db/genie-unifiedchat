"""
SQL Synthesis Agents for Multi-Agent System

This module contains two SQL synthesis agent classes:

1. SQLSynthesisTableAgent:
   - Fast SQL synthesis using Unity Catalog (UC) function tools
   - Direct table metadata access via UC functions
   - Best for table_route execution strategy

2. SQLSynthesisGenieAgent:
   - SQL synthesis using Genie agents as tools
   - Supports both parallel and sequential execution
   - Best for genie_route execution strategy

Both agents receive execution plans from PlanningAgent and generate
executable SQL queries with proper error handling and explanation.

Example usage:
    from langchain_core.runnables import Runnable
    from databricks_langchain import ChatDatabricks
    
    llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")
    
    # Table Route Agent
    table_agent = SQLSynthesisTableAgent(
        llm=llm,
        catalog="catalog_name",
        schema="schema_name"
    )
    sql_result = table_agent(execution_plan)
    
    # Genie Route Agent
    genie_agent = SQLSynthesisGenieAgent(
        llm=llm,
        relevant_spaces=[{"space_id": "...", "space_title": "...", "searchable_content": "..."}]
    )
    sql_result = genie_agent(execution_plan)
"""

import contextvars
import hashlib
import json
import re
import unicodedata
from typing import Dict, List, Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from pydantic import BaseModel, Field


class _ProgressCallbackHandler(BaseCallbackHandler):
    """Emits real-time progress events for tool calls within SQL synthesis agents."""

    def __init__(self, writer, agent_label: str = "sql_synthesis"):
        self._writer = writer
        self._agent = agent_label
        self._tool_round = 0

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._tool_round += 1
        name = serialized.get("name", "unknown_tool")
        short_name = name.split("__")[-1] if "__" in name else name
        self._writer({
            "type": "tool_call_start",
            "agent": self._agent,
            "tool": short_name,
            "content": f"🔧 Calling {short_name}...",
        })

    def on_tool_end(self, output, **kwargs):
        size = len(str(output)) if output else 0
        self._writer({
            "type": "tool_call_end",
            "agent": self._agent,
            "content": f"✅ Tool returned ({size} chars)",
        })

    def on_chat_model_start(self, serialized, messages, **kwargs):
        if self._tool_round > 0:
            self._writer({
                "type": "agent_thinking",
                "agent": self._agent,
                "content": "🤔 Analyzing tool results...",
            })

from databricks_langchain import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
    GenieAgent,
)

from ..utils.genie_route_dag import (
    build_context_packages,
    compute_execution_waves,
    extract_conversation_ids,
    legacy_question_map,
    merge_conversation_ids,
    normalize_genie_route_plan,
    questions_for_wave,
    resolve_genie_execution_mode,
    resolve_space_conversation_id,
    summarize_plan_for_logging,
)


# ==============================================================================
# Genie Agent Pool (for SQLSynthesisGenieAgent)
# ==============================================================================

# Global pool for caching Genie agents across requests
_genie_agent_pool: Dict[str, Any] = {}
_tool_label_translation_cache: Dict[str, str] = {}


def _response_content_to_text(response: Any) -> str:
    """Extract text from a LangChain model response."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return " ".join(parts)
    return str(content or "")


def _translate_tool_label_to_english(label: Any, llm: Optional[Runnable] = None) -> str:
    """Best-effort English label for readable API-safe tool names."""
    raw_label = str(label or "").strip()
    if not raw_label:
        return ""

    if raw_label.isascii():
        return raw_label

    if raw_label in _tool_label_translation_cache:
        return _tool_label_translation_cache[raw_label]

    if llm is None:
        return raw_label

    prompt = (
        "Translate this analytics/BI Genie space title into a concise English tool label.\n"
        "Return ONLY 2-6 English words. Use ASCII letters, numbers, spaces, hyphens, or underscores only.\n"
        "Do not include quotes, explanations, punctuation, or non-English characters.\n\n"
        f"Title: {raw_label}"
    )

    try:
        translated = _response_content_to_text(llm.invoke(prompt)).strip()
        translated = translated.strip("\"'` \n\t")
        translated = re.sub(r"[^a-zA-Z0-9 _-]+", " ", translated)
        translated = re.sub(r"\s+", " ", translated).strip()
        if re.search(r"[a-zA-Z]", translated):
            _tool_label_translation_cache[raw_label] = translated
            return translated
    except Exception as e:
        print(f"⚠ Tool label translation failed for '{raw_label}': {e}")

    return raw_label


def _safe_tool_name(prefix: str, label: Any, fallback_id: Any = "", max_length: int = 128) -> str:
    """Return an API-safe tool name matching ^[a-zA-Z0-9_-]{1,128}$."""
    raw_label = str(label or fallback_id or "tool")
    ascii_label = (
        unicodedata.normalize("NFKD", raw_label)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", ascii_label)
    slug = re.sub(r"_+", "_", slug).strip("_-")

    digest_source = f"{raw_label}:{fallback_id}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    base = re.sub(r"[^a-zA-Z0-9_-]+", "_", prefix).strip("_-") or "tool"

    if slug:
        reserved = len(base) + len(digest) + 2
        slug = slug[: max(0, max_length - reserved)].strip("_-")
        if slug:
            return f"{base}_{slug}_{digest}"

    return f"{base}_{digest}"[:max_length]


def _message_content_to_text(content: Any) -> str:
    """Normalize LangChain message content into a plain text string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _get_recent_assistant_texts(messages: List[Any], limit: int = 2) -> List[str]:
    """Collect the most recent assistant-authored text messages."""
    assistant_texts: List[str] = []

    for message in reversed(messages or []):
        role = getattr(message, "type", None)
        if role is None and isinstance(message, dict):
            role = message.get("role")

        if role not in {"ai", "assistant"}:
            continue

        if isinstance(message, dict):
            content = message.get("content")
            text = _message_content_to_text(content).strip()
        else:
            text_attr = getattr(message, "text", None)
            if isinstance(text_attr, str):
                text = text_attr.strip()
            else:
                content = getattr(message, "content", "")
                text = _message_content_to_text(content).strip()
        if not text:
            continue

        assistant_texts.append(text)
        if len(assistant_texts) >= limit:
            break

    return list(reversed(assistant_texts))


def get_or_create_genie_agent(
    space_id: str,
    space_title: str,
    description: str,
    genie_agent_name: Optional[str] = None,
):
    """
    Get existing Genie agent from pool or create new one if not cached.
    
    OPTIMIZATION: Reuses Genie agents across requests to avoid expensive initialization.
    Expected gain: -1 to -3s on genie route (creating 3-5 agents)
    
    Args:
        space_id: Genie space ID
        space_title: Space title for agent name
        description: Space description
    
    Returns:
        Cached or newly created GenieAgent instance
    """
    global _genie_agent_pool
    
    if space_id not in _genie_agent_pool:
        print(f"⚡ Creating Genie agent for space: {space_title} (first use)")
        safe_genie_agent_name = genie_agent_name or _safe_tool_name("Genie", space_title, space_id)
        
        def enforce_limit(messages, n=5):
            """Enforce result limit in Genie queries."""
            last = messages[-1] if messages else {"content": ""}
            content = last.get("content", "") if isinstance(last, dict) else last.content
            return f"{content}\n\nPlease limit the result to at most {n} rows."
        
        genie_agent = GenieAgent(
            genie_space_id=space_id,
            genie_agent_name=safe_genie_agent_name,
            description=description,
            include_context=True,
            message_processor=lambda msgs: enforce_limit(msgs, n=5)
        )
        
        _genie_agent_pool[space_id] = genie_agent
        print(f"✓ Genie agent cached for {space_title}")
    else:
        print(f"✓ Using cached Genie agent for {space_title}")
    
    return _genie_agent_pool[space_id]


# ==============================================================================
# SQLSynthesisTableAgent
# ==============================================================================

class SQLSynthesisTableAgent:
    """
    Agent responsible for fast SQL synthesis using UC function tools.
    
    OOP design with UC toolkit integration.
    """
    
    def __init__(
        self, 
        llm: Runnable, 
        catalog: str, 
        schema: str
    ):
        """
        Initialize SQLSynthesisTableAgent.
        
        Args:
            llm: Language model for SQL synthesis
            catalog: Unity Catalog catalog name
            schema: Unity Catalog schema name
        """
        self.llm = llm
        self.catalog = catalog
        self.schema = schema
        self.name = "SQLSynthesisTable"
        
        # Initialize UC Function Client
        client = DatabricksFunctionClient()
        set_uc_function_client(client)
        
        from ..core.config import get_config
        config = get_config()
        uc_function_names = config.unity_catalog.uc_function_names_fq
        
        self.uc_toolkit = UCFunctionToolkit(function_names=uc_function_names)
        self.tools = self.uc_toolkit.tools
        
        # Create SQL synthesis agent with tools
        self.agent = create_agent(
            model=llm,
            tools=self.tools,
            system_prompt=(
                "You are a specialized SQL synthesis agent in a multi-agent system.\n\n"
                "ROLE: You receive execution plans from the planning agent and generate SQL queries.\n\n"

                "## WORKFLOW:\n"
                "1. Review the execution plan and provided metadata\n"
                "2. If metadata is sufficient → Generate SQL immediately\n"
                "3. If insufficient, call UC function tools in this order to gather metadata:\n"
                "   a) get_space_summary for space information\n"
                "   b) get_table_overview for table schemas\n"
                "   c) get_column_detail for specific columns\n"
                "   d) get_space_details ONLY as last resort (token intensive)\n"
                "4. If still cannot find enough metadata in relevant spaces, expand searching scope to all spaces\n"
                "   mentioned in the execution plan's 'vector_search_relevant_spaces_info' field\n"
                "5. Generate complete, executable SQL using the gathered metadata, print out the final SQL\n\n"

                "## UC FUNCTION USAGE:\n"
                "- Pass arguments as JSON array strings: e.g., '[\"space_id_1\", \"space_id_2\"]' or passing a NULL without any quote\n"
                "- Always explicitly passing all required arguments, even it is a NULL\n"
                "- Only query spaces from execution plan's relevant_space_ids\n"
                "- Use minimal sufficiency: only query what you need\n"
                "- OPTIMIZATION: When possible, call multiple UC functions in parallel by returning multiple tool calls\n"
                "  Example: If you need table_overview for space_1 AND column_detail for space_2, call both tools at once\n"
                "- This enables parallel execution and reduces latency by 1-2 seconds\n\n"

                "## SQL FINETUNE INSTRUCTIONS:\n"
                "- **Additional SQL Finetune Step** After you already generated the SQL, take a reflection first, and then you are ready to call **get_space_instructions** to extract the space instructions taught by human; only use the most related instruction parts to finetune the SQL if necessary.\n"
                "- This provides essential human-taught SQL patterns and best practices for the specific space.\n\n"

                "## OUTPUT REQUIREMENTS:\n"
                "- Generate complete, executable SQL with:\n"
                "  * Proper JOINs based on execution plan\n"
                "  * WHERE clauses for filtering\n"
                "  * Appropriate aggregations\n"
                "  * Clear column aliases\n"
                "  * Always use real column names, never make up ones\n\n"
                "## MULTI-QUERY STRATEGY:\n"
                "- If the question has multiple parts (sub_questions) and you think it's better to report\n"
                "  each query and result separately instead of combining into one big complex query:\n"
                "  * Generate MULTIPLE separate SQL queries (one per sub-question)\n"
                "  * This is preferred when: sub-questions are independent, results are easier to interpret\n"
                "    separately, or combining would create overly complex SQL\n"
                "- If sub-questions are closely related and naturally combine (e.g., same table, similar filters):\n"
                "  * You may generate a single combined SQL query\n\n"
                "- CRITICAL RULE for 'top N items + their details' patterns:\n"
                "  When the question asks for 'top N items and their associated details'\n"
                "  (e.g., 'top 10 medicines and their diagnoses', 'top 5 providers and their procedures'),\n"
                "  you MUST use multiple separate SQL queries:\n"
                "  * Query 1: The top N items with their aggregate metric (e.g., top 10 by total cost)\n"
                "  * Query 2+: For each detail dimension, a separate query\n"
                "  * IMPORTANT: Each query must be SELF-CONTAINED and independently executable.\n"
                "    If Query 2 needs the same top-N list as Query 1, embed the top-N selection\n"
                "    as a subquery or CTE within Query 2 -- do NOT reference Query 1's results.\n"
                "  * This avoids a single cross-join that explodes rows\n"
                "    (e.g., 10 medicines x 362 diagnoses = 3,620 rows) and lets each result\n"
                "    be summarized clearly.\n"
                "  * NEVER combine top-N aggregation with detail expansion in a single query.\n\n"
                "## OUTPUT FORMAT:\n"
                "- Return your response with:\n"
                "0. Your overall explanation.\n"
                "1. Your detailed explanations for each query how you generated the SQL; If SQL cannot be generated, explain what metadata is missing\n"
                "2. SQL queries formatted as follows:\n"
                "   * For SINGLE-part questions: One ```sql code block with query ending in semicolon\n"
                "   * For MULTI-part questions: Use SEPARATE ```sql code blocks (one per query)\n"
                "   * Each query MUST end with a semicolon (;)\n"
                "   * Add a leading comment before each query: -- Query N: <brief description>\n"
                "   * Example for multi-part:\n"
                "     ```sql\n"
                "     -- Query 1: Most common diagnoses\n"
                "     SELECT diagnosis_code, COUNT(*) AS freq FROM diagnosis GROUP BY diagnosis_code;\n"
                "     ```\n"
                "     ```sql\n"
                "     -- Query 2: Top procedures\n"
                "     SELECT procedure_code, COUNT(*) AS count FROM procedures GROUP BY procedure_code;\n"
                "     ```\n\n"
            )
        )
    
    def synthesize_sql(self, plan: Dict[str, Any], writer=None) -> Dict[str, Any]:
        """
        Synthesize SQL query based on execution plan.
        
        Args:
            plan: Execution plan from planning agent
            writer: Optional LangGraph stream writer for real-time progress events
            
        Returns:
            Dictionary with:
            - sql: str - Extracted SQL query (None if cannot generate)
            - explanation: str - Agent's explanation/reasoning
            - has_sql: bool - Whether SQL was successfully extracted
        """
        plan_result = plan
        agent_message = {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Generate a SQL query to answer the question according to the Query Plan:
{json.dumps(plan_result, indent=2)}

Use your available UC function tools to gather metadata intelligently.
"""
                }
            ]
        }
        
        config = {}
        if writer:
            config["callbacks"] = [_ProgressCallbackHandler(writer, "sql_synthesis_table")]
        result = self.agent.invoke(agent_message, config=config)
        
        # Extract SQL and explanation from response
        if result and "messages" in result:
            assistant_texts = _get_recent_assistant_texts(result["messages"], limit=2)
            final_content = assistant_texts[-1] if assistant_texts else _message_content_to_text(result["messages"][-1].content)
            explanation_source = "\n\n".join(assistant_texts) if assistant_texts else final_content
            original_content = explanation_source
            
            sql_query = None
            has_sql = False
            
            # Try to extract SQL from markdown - use findall to capture ALL code blocks
            if "```sql" in final_content.lower():
                # Find all ```sql blocks
                sql_blocks = re.findall(r'```sql\s*(.*?)\s*```', final_content, re.IGNORECASE | re.DOTALL)
                if sql_blocks:
                    # Join all SQL blocks with newlines to preserve multi-query structure
                    sql_query = '\n\n'.join(block.strip() for block in sql_blocks if block.strip())
                    has_sql = True
            elif "```" in final_content:
                # Find all generic code blocks
                code_blocks = re.findall(r'```\s*(.*?)\s*```', final_content, re.DOTALL)
                # Filter for SQL-like blocks
                sql_blocks = [
                    block.strip() for block in code_blocks 
                    if block.strip() and any(keyword in block.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'WITH'])
                ]
                if sql_blocks:
                    # Join all SQL blocks
                    sql_query = '\n\n'.join(sql_blocks)
                    has_sql = True
            
            # Clean up explanation
            explanation = explanation_source
            if "```sql" in explanation.lower():
                explanation = re.sub(r'```sql\s*.*?\s*```', '', explanation, flags=re.IGNORECASE | re.DOTALL)
            elif "```" in explanation:
                explanation = re.sub(r'```\s*.*?\s*```', '', explanation, flags=re.DOTALL)
            explanation = explanation.strip()
            if not explanation:
                explanation = original_content if not has_sql else "SQL query generated successfully."
            
            return {
                "sql": sql_query,
                "explanation": explanation,
                "has_sql": has_sql
            }
        else:
            raise Exception("No response from agent")
    
    def __call__(self, plan: Dict[str, Any], writer=None) -> Dict[str, Any]:
        """Make agent callable."""
        return self.synthesize_sql(plan, writer=writer)


# ==============================================================================
# SQLSynthesisGenieAgent
# ==============================================================================

class SQLSynthesisGenieAgent:
    """
    Agent responsible for Genie Route SQL synthesis using Genie agents as tools.
    
    EXECUTION MODES:
    ---------------
    1. LangGraph Agent Mode (default via synthesize_sql()):
       - Uses LangGraph agent with tool calling
       - Supports retries, disaster recovery, and adaptive routing
       - Agent decides which tools to call and when
       - Best for complex queries requiring orchestration
    
    2. RunnableParallel Mode (via invoke_genie_agents_parallel()):
       - Uses RunnableParallel for direct parallel execution
       - Faster for simple parallel queries
       - No retry logic or adaptive routing
       - Best for straightforward parallel execution
    
    ARCHITECTURE:
    ------------
    - Upgraded from RunnableLambda to RunnableParallel pattern
    - Each Genie agent is wrapped as both a tool and a parallel executor
    - Supports efficient parallel invocation using LangChain's RunnableParallel
    - Optimized to only create Genie agents for relevant spaces (not all spaces)
    """
    
    def __init__(self, llm: Runnable, relevant_spaces: List[Dict[str, Any]]):
        """
        Initialize SQL Synthesis Genie Agent with tool-calling pattern.
        
        Args:
            llm: Language model for SQL synthesis
            relevant_spaces: List of relevant spaces from PlanningAgent's Vector Search.
                            Each dict should have: space_id, space_title, searchable_content
        """
        self.llm = llm
        self.relevant_spaces = relevant_spaces
        self.name = "SQLSynthesisGenie"
        # Per-space Genie Conversation API ids for same-space retries/follow-ups.
        self._genie_conversation_ids: Dict[str, str] = {}
        # Latest Genie batch results (for join_contract updates after tool calls).
        self._last_genie_results: Dict[str, Any] = {}
        
        # Create Genie agents and their tool representations
        self.genie_agents = []
        self.genie_agent_tools = []
        self.space_id_to_tool = {}
        self._create_genie_agent_tools()
        
        # Create SQL synthesis agent with Genie agent tools
        self.sql_synthesis_agent = self._create_sql_synthesis_agent()

    def seed_conversation_ids(self, conversation_ids: Optional[Dict[str, Any]]) -> None:
        """Seed the per-space conversation cache (e.g. graph-level SQL retry)."""
        self._genie_conversation_ids = merge_conversation_ids(
            self._genie_conversation_ids,
            conversation_ids,
        )

    def get_conversation_ids(self) -> Dict[str, str]:
        """Return a copy of cached Genie conversation_ids."""
        return dict(self._genie_conversation_ids)

    def get_last_genie_results(self) -> Dict[str, Any]:
        """Return the most recent parallel/DAG Genie result map."""
        return dict(self._last_genie_results)
    
    def _create_genie_agent_tools(self):
        """
        Create Genie agents as tools only for relevant spaces.
        
        OPTIMIZED: Uses cached Genie agents from pool to avoid expensive initialization.
        Expected gain: -1 to -3s on genie route (when agents are already cached)
        
        Creates both:
        1. Individual tool wrappers for LangGraph agent tool calling
        2. A parallel executor mapping for efficient batch invocation
        
        Uses LangChain preferred syntax with Pydantic BaseModel and StructuredTool.
        """
        print(f"  Creating Genie agent tools for {len(self.relevant_spaces)} relevant spaces...")
        
        for space in self.relevant_spaces:
            space_id = space.get("space_id")
            space_title = space.get("space_title", space_id)
            searchable_content = space.get("searchable_content", "")
            
            if not space_id:
                print(f"  ⚠ Warning: Space missing space_id, skipping: {space}")
                continue
            
            english_tool_label = _translate_tool_label_to_english(space_title, self.llm)
            genie_agent_name = _safe_tool_name("Genie", english_tool_label, space_id)
            description = searchable_content
            
            # OPTIMIZATION: Get Genie agent from pool (cached or newly created)
            genie_agent = get_or_create_genie_agent(
                space_id,
                space_title,
                description,
                genie_agent_name=genie_agent_name,
            )
            self.genie_agents.append(genie_agent)
            
            # Define tool input schema using Pydantic
            class GenieToolInput(BaseModel):
                question: str = Field(..., description="Natural-language query to run in the Genie Space")
                conversation_id: Optional[str] = Field(
                    None,
                    description=(
                        "Genie conversation_id for SAME-SPACE continuity. "
                        "On retry/reframe in this space, pass the conversation_id from the "
                        "prior result. Omit only for a fresh conversation or a different space."
                    ),
                )
            
            # Create tool function using factory pattern to capture agent + space_id
            def make_genie_tool_call(agent, sid: str):
                """Factory function to capture agent/space in closure properly"""
                def _genie_tool_call(question: str, conversation_id: Optional[str] = None):
                    """
                    StructuredTool with args_schema expects individual field arguments,
                    not a single Pydantic object.
                    """
                    cid = resolve_space_conversation_id(
                        sid,
                        explicit=conversation_id,
                        cached=self._genie_conversation_ids,
                    )
                    # GenieAgent expects a LangChain-style message list
                    result = agent.invoke({
                        "messages": [{"role": "user", "content": question}],
                        "conversation_id": cid,
                    })
                    # Extract final output + optional context
                    out_cid = str(result.get("conversation_id") or "").strip() or cid
                    out = {"conversation_id": out_cid or ""}
                    if out_cid:
                        self._genie_conversation_ids[sid] = out_cid
                    msgs = result["messages"]
                    def _get(name): 
                        return next((getattr(m, "content", "") for m in msgs if getattr(m, "name", None) == name), None)
                    out["answer"] = _get("query_result") or ""
                    reasoning = _get("query_reasoning")
                    sql = _get("query_sql")
                    if reasoning: out["reasoning"] = reasoning
                    if sql: out["sql"] = sql
                    return out
                return _genie_tool_call
            
            # Create StructuredTool
            genie_tool = StructuredTool(
                name=genie_agent_name,
                description=(
                    f"Use for governed analytics queries (NL→SQL) in {space_title}. "
                    f"{description}. "
                    "Returns an answer and, when available, the generated SQL, reasoning, "
                    "and conversation_id. For same-space retry/reframe, pass conversation_id "
                    "from the prior call so Genie keeps multi-turn context."
                ),
                args_schema=GenieToolInput,
                func=make_genie_tool_call(genie_agent, space_id),
            )
            self.genie_agent_tools.append(genie_tool)
            self.space_id_to_tool[space_id] = genie_tool
            
            print(f"  ✓ Created Genie agent tool: {genie_agent_name} ({space_id})")
    
    def _create_parallel_execution_tool(self):
        """
        Create a tool that allows the agent to invoke multiple Genie agents in parallel.
        
        This tool gives the agent control over parallel execution with the same
        disaster recovery capabilities as individual tool calls.
        
        Uses RunnableParallel pattern with StructuredTool for type safety.
        """
        
        # Define input schema for parallel / DAG execution
        class ParallelGenieInput(BaseModel):
            genie_route_plan: Dict[str, Any] = Field(
                ...,
                description=(
                    "Map of space_id to question string OR structured step "
                    "{question, depends_on, inject}. "
                    "Example parallel: {'space_a': 'Get member demographics'}. "
                    "Example DAG: {'space_a': {'question': 'Top 10 drugs', 'depends_on': []}, "
                    "'space_b': {'question': 'Diagnoses for those drugs', "
                    "'depends_on': ['space_a'], "
                    "'inject': ['few_shot', 'ids', 'filters']}}."
                ),
            )
            genie_execution_mode: Optional[str] = Field(
                None,
                description=(
                    "Optional 'parallel' or 'dag'. If omitted, DAG is inferred when any "
                    "step has depends_on. Distinct from UI SQL execution_mode."
                ),
            )
            conversation_ids: Optional[Dict[str, str]] = Field(
                None,
                description=(
                    "Optional map of space_id → Genie conversation_id for same-space "
                    "retries. On retry/reframe, pass ids from the prior tool result "
                    "(_genie_conversation_ids or each space's conversation_id). "
                    "Fresh first attempts may omit this."
                ),
            )

        def merge_genie_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
            """Normalize per-space Genie tool outputs into a unified result map."""
            merged_results = {}

            for space_id, result in outputs.items():
                extracted = {
                    "space_id": space_id,
                    "question": "",
                    "sql": "",
                    "reasoning": "",
                    "answer": "",
                    "conversation_id": "",
                    "error": "",
                    "success": False,
                }

                # Every value from _safe_invoke is a plain dict: either the
                # _genie_tool_call result or a per-task error dict.
                if isinstance(result, dict):
                    extracted["question"] = result.get("question", "")
                    extracted["answer"] = result.get("answer", "")
                    extracted["sql"] = result.get("sql", "")
                    extracted["reasoning"] = result.get("reasoning", "")
                    extracted["conversation_id"] = result.get("conversation_id", "")
                    extracted["error"] = result.get("error", "")
                    extracted["success"] = bool(result.get("sql") or result.get("answer"))

                merged_results[space_id] = extracted

            return merged_results

        space_id_to_tool = dict(self.space_id_to_tool)

        def _run_question_map(
            question_map: Dict[str, str],
            conversation_ids: Optional[Dict[str, str]] = None,
        ) -> Dict[str, Any]:
            """Execute a flat space_id→question map concurrently with optional continuity."""
            if not question_map:
                return {}

            cid_map = merge_conversation_ids(self._genie_conversation_ids, conversation_ids)
            parallel_tasks = {}
            for space_id, question in question_map.items():
                tool = space_id_to_tool[space_id]
                ctx = contextvars.copy_context()
                prior_cid = cid_map.get(space_id)

                def _safe_invoke(inp, sid=space_id, t=tool, c=ctx, cid=prior_cid):
                    try:
                        q = inp.get(sid, "")
                        out = c.run(t.func, question=q, conversation_id=cid)
                        if isinstance(out, dict):
                            return {"question": q, **out}
                        return {"question": q, "answer": str(out)}
                    except Exception as task_err:  # noqa: BLE001
                        return {
                            "space_id": sid,
                            "question": inp.get(sid, ""),
                            "conversation_id": cid or "",
                            "success": False,
                            "error": f"Genie call failed for {sid}: {task_err}",
                        }

                parallel_tasks[space_id] = RunnableLambda(_safe_invoke)

            parallel = RunnableParallel(**parallel_tasks)
            composed = parallel | RunnableLambda(merge_genie_outputs)
            results = composed.invoke(question_map)
            # Persist any conversation_ids returned for later same-space retries.
            self._genie_conversation_ids = merge_conversation_ids(
                self._genie_conversation_ids,
                extract_conversation_ids(results),
            )
            return results

        def invoke_parallel_genie_agents(
            genie_route_plan: Dict[str, Any],
            genie_execution_mode: Optional[str] = None,
            conversation_ids: Optional[Dict[str, str]] = None,
        ) -> Dict[str, Any]:
            """
            Invoke Genie agents in parallel, or as dependency waves with structured inject.

            StructuredTool with args_schema expects individual field arguments.
            """
            try:
                normalized = normalize_genie_route_plan(genie_route_plan)
                if not normalized:
                    return {"error": "No valid Genie route tasks to execute"}

                missing = [sid for sid in normalized if sid not in space_id_to_tool]
                if missing:
                    return {
                        "error": f"No tool found for space_id(s): {', '.join(missing)}",
                        "available_space_ids": list(space_id_to_tool.keys()),
                    }

                if conversation_ids:
                    self.seed_conversation_ids(conversation_ids)

                mode = resolve_genie_execution_mode(genie_execution_mode, normalized)
                plan_summary = summarize_plan_for_logging(normalized, mode)
                print(f"  Genie route execution: {json.dumps(plan_summary)}")
                if self._genie_conversation_ids:
                    print(
                        "  Reusing Genie conversation_ids for spaces: "
                        f"{sorted(self._genie_conversation_ids)}"
                    )

                if mode == "parallel":
                    question_map = legacy_question_map(normalized)
                    results = _run_question_map(question_map, conversation_ids)
                else:
                    # DAG: run topological waves; inject compact upstream context
                    # into dependent Genie questions between waves.
                    results: Dict[str, Any] = {}
                    packages: Dict[str, Any] = {}
                    waves = compute_execution_waves(normalized)
                    for wave_idx, wave in enumerate(waves, 1):
                        question_map = questions_for_wave(wave, normalized, packages)
                        print(
                            f"  🌊 Genie DAG wave {wave_idx}/{len(waves)}: "
                            f"{len(wave)} space(s) — {wave}"
                        )
                        wave_results = _run_question_map(question_map, conversation_ids)
                        results.update(wave_results)
                        packages.update(build_context_packages(wave_results))

                space_results = [
                    v for v in results.values()
                    if isinstance(v, dict) and "success" in v
                ]
                if space_results and all(not v.get("success") for v in space_results):
                    errors = "; ".join(
                        str(v.get("error") or "unknown error") for v in space_results
                    )
                    return {"error": f"All Genie tasks failed: {errors}"}

                # Continuity map for the orchestrator LLM / graph-level retries.
                results["_genie_conversation_ids"] = self.get_conversation_ids()

                # Attach lightweight DAG metadata for the orchestrator LLM.
                if mode == "dag":
                    results["_genie_dag"] = {
                        "mode": mode,
                        "waves": plan_summary["waves"],
                        "dependencies": plan_summary["dependencies"],
                    }

                from ..utils.join_contract import (
                    build_join_contract_from_genie_results,
                    format_join_contract_block,
                )

                join_contract = build_join_contract_from_genie_results(
                    results,
                    relevant_spaces=self.relevant_spaces,
                    dependency_edges=plan_summary.get("dependency_edges"),
                )
                results["_join_contract"] = join_contract
                contract_block = format_join_contract_block(join_contract)
                if contract_block:
                    results["_join_contract_block"] = contract_block
                self._last_genie_results = {
                    k: v for k, v in results.items() if not str(k).startswith("_") or k == "_genie_dag"
                }
                # Keep metadata keys too for downstream merge.
                self._last_genie_results.update(
                    {
                        "_genie_conversation_ids": results.get("_genie_conversation_ids"),
                        "_genie_dag": results.get("_genie_dag"),
                        "_join_contract": join_contract,
                    }
                )
                return results

            except Exception as e:
                return {"error": f"Genie route execution failed: {str(e)}"}

        parallel_tool = StructuredTool(
            name="invoke_parallel_genie_agents",
            description=(
                "Invoke Genie agents for SQL generation. Supports: "
                "(1) PARALLEL independent questions — space_id→question string map; "
                "(2) DAG dependent questions — structured steps with depends_on + inject "
                "(few_shot, ids, filters, sql_preview, answer_summary) so upstream "
                "Genie Q/SQL/keys are chained as few-shot examples into downstream prompts. "
                "Pass genie_execution_mode='dag' when steps have dependencies "
                "(or omit and let depends_on infer DAG). "
                "When any step has depends_on, you MUST use this tool — individual "
                "Genie tools skip wave inject. "
                "Returns per-space SQL/reasoning/answer/conversation_id plus "
                "_genie_conversation_ids and _join_contract (entities/keys/time/metrics/"
                "sql_by_space). On SAME-SPACE retry, pass conversation_ids "
                "from the prior result (or use individual Genie tools with conversation_id). "
                "Do not reuse a conversation_id across different spaces. "
                "Use _join_contract when framing dependent follow-up questions."
            ),
            args_schema=ParallelGenieInput,
            func=invoke_parallel_genie_agents,
        )

        return parallel_tool
    
    def _create_sql_synthesis_agent(self):
        """
        Create LangGraph SQL Synthesis Agent with Genie agent tools.
        
        Uses Databricks LangGraph SDK with create_agent pattern.
        Includes both individual Genie agent tools AND a parallel execution tool.
        """
        tools = []
        tools.extend(self.genie_agent_tools)
        
        # Add parallel execution tool
        parallel_tool = self._create_parallel_execution_tool()
        tools.append(parallel_tool)
        
        print(f"✓ Created SQL Synthesis Agent with {len(self.genie_agent_tools)} Genie agent tools + 1 parallel execution tool")
        
        # Create SQL Synthesis Agent (specialized for multi-agent system)
        sql_synthesis_agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=(
"""You are a SQL synthesis agent with access to both INDIVIDUAL and PARALLEL/DAG Genie agent execution tools.

The Plan given to you is a JSON:
{
'original_query': 'The User's Question',
'vector_search_relevant_spaces_info': [{'space_id': 'space_id_1', 'space_title': 'space_title_1'}, ...],
"question_clear": true,
"sub_questions": ["sub-question 1", "sub-question 2", ...],
"requires_multiple_spaces": true/false,
"relevant_space_ids": ["space_id_1", "space_id_2", ...],
"requires_join": true/false,
"join_strategy": "table_route" or "genie_route" or null,
"execution_plan": "Brief description of execution plan",
"genie_execution_mode": "parallel" or "dag",
"genie_route_plan": {
  // legacy parallel form:
  // 'space_id_1': 'partial_question_1',
  // OR structured DAG form:
  // 'space_id_1': {'question': '...', 'depends_on': [], 'inject': ['few_shot','ids','filters']},
  // 'space_id_2': {'question': '...', 'depends_on': ['space_id_1'], 'inject': ['few_shot','ids','filters']}
} or null
}

## TOOL EXECUTION STRATEGY:

### OPTION 1: PARALLEL (independent spaces)
When genie_execution_mode is "parallel" AND no step has depends_on:
1. Extract genie_route_plan
2. Call invoke_parallel_genie_agents(genie_route_plan=..., genie_execution_mode="parallel")
3. Combine successful SQL fragments

### OPTION 2: DAG WITH FEW-SHOT INJECT (dependent spaces) — REQUIRED when depends_on is non-empty
When genie_execution_mode is "dag" OR any step has depends_on (e.g. "top N items, then details for those items"):
1. Pass the structured genie_route_plan as-is (keep depends_on + inject, including few_shot)
2. Call invoke_parallel_genie_agents(genie_route_plan=..., genie_execution_mode="dag")
3. The tool runs topological waves and injects labeled few-shot examples
   (upstream Q / SQL / keys / filters) into downstream Genie questions
4. Do NOT call individual Genie tools for dependent spaces — they skip wave inject
5. Do NOT manually re-ask upstream spaces unless a wave failed
6. Do NOT pass a conversation_id from one space to a different space

### OPTION 3: INDIVIDUAL TOOLS (rare)
Use only for granular SAME-SPACE retry after DAG/parallel failure.
Never use individual tools for a step that has depends_on.

**NOTE**: Prefer OPTION 1 for independent multi-space work; OPTION 2 whenever
one Genie answer must feed another Genie's prompt. Do not flatten DAG plans
into string-only maps — that drops dependencies and few-shot inject.

## DISASTER RECOVERY (DR) — SAME-SPACE conversation_id CONTINUITY:

1. **First Attempt**: Try the planned parallel or DAG call AS IS (no conversation_ids)
2. **If fails / empty SQL**: Analyze the error message
   - If agent says "I don't have information for X", remove X from the question
   - If agent returns empty/incomplete SQL, try rephrasing the question
3. **Retry Once (SAME SPACE)**: Prefer Genie Conversation API continuity
   - From the prior tool result, collect each space's conversation_id
     (or use `_genie_conversation_ids`)
   - Re-call with reframed question(s) AND
     `conversation_ids={space_id: prior_conversation_id}`
   - Or call the individual Genie tool with `conversation_id` set
   - Preserve depends_on for DAG retries
   - Only omit conversation_id when switching to a *different* space
4. **If still fails**: Try alternative Genie spaces (fresh conversation; no old conversation_id)
5. **Final fallback**: Work with what you have and explain limitations

## EXAMPLE SAME-SPACE RETRY:

Step 1: invoke_parallel_genie_agents(...) → space_a returns conversation_id="abc"
Step 2: space_a SQL empty → reframe question
Step 3: invoke_parallel_genie_agents(..., conversation_ids={"space_a": "abc"})
        OR call the individual Genie tool with conversation_id="abc"

## EXAMPLE DAG EXECUTION:

Step 1: Call invoke_parallel_genie_agents with structured depends_on plan (mode=dag)
Step 2: Wave 1 runs independent spaces; wave 2+ receive few-shot Q/SQL/keys
Step 3: Combine all successful SQL fragments (keep self-contained queries)

## SQL SYNTHESIS:

MULTI-QUERY STRATEGY:
- If the question has multiple parts and you think it's better to report each query
  and result separately instead of combining into one big complex query:
  * Generate MULTIPLE separate SQL queries (one per sub-question)
  * This is preferred when: sub-questions are independent, results are easier to interpret
    separately, or combining would create overly complex SQL
- If sub-questions are closely related and naturally combine (e.g., same Genie space, similar context):
  * You may combine SQL fragments into a single query

CRITICAL RULE for "top N items + their details" patterns:
  When the question asks for "top N items and their associated details"
  (e.g., "top 10 medicines and their diagnoses", "top 5 providers and their procedures"),
  you MUST use multiple separate SQL queries:
  * Query 1: The top N items with their aggregate metric (e.g., top 10 by total cost)
  * Query 2+: For each detail dimension, a separate query
  * IMPORTANT: Each query must be SELF-CONTAINED and independently executable.
    If Query 2 needs the same top-N list as Query 1, embed the top-N selection
    as a subquery or CTE within Query 2 -- do NOT reference Query 1's results.
  * This avoids a single cross-join that explodes rows
    (e.g., 10 medicines x 362 diagnoses = 3,620 rows) and lets each result
    be summarized clearly.
  * NEVER combine top-N aggregation with detail expansion in a single query.

OUTPUT REQUIREMENTS:
- Generate complete, executable SQL with:
  * Proper JOINs based on execution plan strategy
  * WHERE clauses for filtering  
  * Appropriate aggregations
  * Clear column aliases
  * Always use real column names from the data
- Return your response with:
  0. Your overall explanation including which execution strategy you used
  1. Your detailed explanations for each query how you generated the SQL; If a particular SQL cannot be generated, explain what metadata is missing\n"
  2. SQL queries formatted as follows:
     * For SINGLE-part questions: One ```sql code block with query ending in semicolon
     * For MULTI-part questions: Use SEPARATE ```sql code blocks (one per query)
     * Each query MUST end with a semicolon (;)
     * Add a leading comment before each query: -- Query N: <brief description>
     * Example for multi-part:
       ```sql
       -- Query 1: Most common diagnoses
       SELECT diagnosis_code, COUNT(*) AS freq FROM diagnosis GROUP BY diagnosis_code;
       ```
       ```sql
       -- Query 2: Top procedures
       SELECT procedure_code, COUNT(*) AS count FROM procedures GROUP BY procedure_code;
       ```"""
            )
        )
        
        return sql_synthesis_agent
    
    def invoke_genie_agents_parallel(
        self,
        genie_route_plan: Dict[str, Any],
        genie_execution_mode: Optional[str] = None,
        conversation_ids: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke Genie agents via the parallel/DAG tool (RunnableParallel waves).

        Prefer synthesize_sql() for full orchestration; this is a direct helper.
        """
        if not genie_route_plan:
            return {}

        parallel_tool = self._create_parallel_execution_tool()
        return parallel_tool.func(
            genie_route_plan=genie_route_plan,
            genie_execution_mode=genie_execution_mode,
            conversation_ids=conversation_ids,
        )
    
    def synthesize_sql(
        self, 
        plan: Dict[str, Any],
        writer=None,
    ) -> Dict[str, Any]:
        """
        Synthesize SQL using Genie agents with intelligent tool selection.
        
        The agent has access to:
        1. invoke_parallel_genie_agents tool - Parallel or DAG few-shot inject
        2. Individual Genie agent tools - Same-space retry only
        
        The agent autonomously decides which strategy to use and handles
        disaster recovery with retry logic for both parallel and sequential execution
        
        Args:
            plan: Complete plan dictionary from PlanningAgent containing:
                - original_query: Original user question
                - execution_plan: Execution plan description
                - genie_route_plan: Mapping of space_id to question or structured DAG step
                - genie_execution_mode: "parallel" | "dag" (optional; inferred from depends_on)
                - vector_search_relevant_spaces_info: List of relevant spaces
                - relevant_space_ids: List of relevant space IDs
                - requires_join: Whether join is needed
                - join_strategy: Join strategy (table_route/genie_route)
            
        Returns:
            Dictionary with:
            - sql: str - Combined SQL query (None if cannot generate)
            - explanation: str - Agent's explanation/reasoning
            - has_sql: bool - Whether SQL was successfully extracted
        """
        plan_result = dict(plan or {})
        normalized = normalize_genie_route_plan(plan_result.get("genie_route_plan"))
        mode = resolve_genie_execution_mode(
            plan_result.get("genie_execution_mode"),
            normalized,
        )
        plan_result["genie_execution_mode"] = mode
        if normalized:
            # Persist structured steps so the tool keeps depends_on/inject.
            plan_result["genie_route_plan"] = normalized

        # Seed same-space Genie conversation continuity (graph-level retries).
        self.seed_conversation_ids(plan_result.get("genie_conversation_ids"))
        if self._genie_conversation_ids:
            plan_result["genie_conversation_ids"] = self.get_conversation_ids()

        from ..utils.join_contract import format_join_contract_block

        join_contract_block = format_join_contract_block(plan_result.get("join_contract"))
        if join_contract_block:
            plan_result["join_contract_block"] = join_contract_block

        print(f"\n{'='*80}")
        print("🤖 SQL Synthesis Agent - Starting (parallel/DAG Genie tool)...")
        print(f"{'='*80}")
        print(f"Plan: {json.dumps(plan_result, indent=2)}")
        print(f"{'='*80}\n")

        dag_hint = ""
        if mode == "dag":
            dag_hint = (
                "This plan is a DEPENDENCY DAG. You MUST call invoke_parallel_genie_agents "
                "with genie_execution_mode='dag' and keep structured depends_on/inject "
                "(including few_shot). Do not flatten steps to bare question strings. "
                "Do not call individual Genie tools for dependents — that skips few-shot inject. "
                "Do not reuse conversation_id across different spaces.\n"
            )
        else:
            dag_hint = (
                "Spaces are independent. Call invoke_parallel_genie_agents with "
                "genie_execution_mode='parallel' for fastest fan-out.\n"
            )

        continuity_hint = ""
        if self._genie_conversation_ids:
            continuity_hint = (
                "SAME-SPACE RETRY: genie_conversation_ids are already available. "
                "Pass them as conversation_ids on invoke_parallel_genie_agents "
                "(or conversation_id on individual Genie tools) so Genie continues "
                "the prior conversation instead of starting a new one.\n"
            )

        contract_hint = ""
        if join_contract_block:
            contract_hint = (
                "JOIN CONTRACT is available in the plan (and as join_contract_block). "
                "When asking dependent Genie follow-ups, ground questions in that contract "
                "(keys, time window, metrics, sql_by_space) instead of rediscovering context.\n"
            )

        agent_message = {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Generate a SQL query to answer the question according to the Query Plan:
{json.dumps(plan_result, indent=2)}

RECOMMENDED APPROACH:
{dag_hint}{continuity_hint}{contract_hint}
Use invoke_parallel_genie_agents on genie_route_plan, then combine SQL fragments
into final executable queries. On same-space retry, reuse conversation_ids.
"""
                }
            ]
        }
        
        try:
            config = {}
            if writer:
                config["callbacks"] = [_ProgressCallbackHandler(writer, "sql_synthesis_genie")]
            result = self.sql_synthesis_agent.invoke(agent_message, config=config)
            
            # Extract SQL from agent result
            # The agent returns {"messages": [...]}
            # Last message contains the final response
            assistant_texts = _get_recent_assistant_texts(result["messages"], limit=2)
            final_message = result["messages"][-1]
            final_content = assistant_texts[-1] if assistant_texts else _message_content_to_text(final_message.content).strip()
            explanation_source = "\n\n".join(assistant_texts) if assistant_texts else final_content
            
            print(f"\n{'='*80}")
            print("✅ SQL Synthesis Agent completed")
            print(f"{'='*80}")
            print(f"Result: {final_content[:500]}...")
            print(f"{'='*80}\n")
            
            # Extract SQL and explanation from the result
            sql_query = None
            has_sql = False
            explanation = explanation_source
            
            # Clean markdown if present and extract SQL - use findall to capture ALL code blocks
            if "```sql" in final_content.lower():
                # Find all ```sql blocks
                sql_blocks = re.findall(r'```sql\s*(.*?)\s*```', final_content, re.IGNORECASE | re.DOTALL)
                if sql_blocks:
                    # Join all SQL blocks with newlines to preserve multi-query structure
                    sql_query = '\n\n'.join(block.strip() for block in sql_blocks if block.strip())
                    has_sql = True
            elif "```" in final_content:
                # Find all generic code blocks
                code_blocks = re.findall(r'```\s*(.*?)\s*```', final_content, re.DOTALL)
                # Filter for SQL-like blocks
                sql_blocks = [
                    block.strip() for block in code_blocks 
                    if block.strip() and any(keyword in block.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'WITH'])
                ]
                if sql_blocks:
                    # Join all SQL blocks
                    sql_query = '\n\n'.join(sql_blocks)
                    has_sql = True
            else:
                # No markdown, check if the entire content is SQL
                if any(keyword in final_content.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'JOIN']):
                    sql_query = final_content
                    has_sql = True
                    explanation = "SQL query generated successfully by Genie agent tools."
            
            if "```sql" in explanation.lower():
                explanation = re.sub(r'```sql\s*.*?\s*```', '', explanation, flags=re.IGNORECASE | re.DOTALL)
            elif "```" in explanation:
                explanation = re.sub(r'```\s*.*?\s*```', '', explanation, flags=re.DOTALL)
            explanation = explanation.strip()
            if not explanation:
                explanation = final_content if not has_sql else "SQL query generated successfully by Genie agent tools."
            
            from ..utils.join_contract import (
                build_join_contract_from_genie_results,
                merge_join_contracts,
            )

            join_contract = merge_join_contracts(
                plan_result.get("join_contract"),
                build_join_contract_from_genie_results(
                    self.get_last_genie_results(),
                    relevant_spaces=self.relevant_spaces,
                    dependency_edges=plan_result.get("dependency_edges"),
                ),
            )
            return {
                "sql": sql_query,
                "explanation": explanation,
                "has_sql": has_sql,
                "genie_conversation_ids": self.get_conversation_ids(),
                "join_contract": join_contract,
            }
            
        except Exception as e:
            print(f"\n{'='*80}")
            print("❌ SQL Synthesis Agent failed")
            print(f"{'='*80}")
            print(f"Error: {str(e)}")
            print(f"{'='*80}\n")
            
            return {
                "sql": None,
                "explanation": f"SQL synthesis failed: {str(e)}",
                "has_sql": False,
                "genie_conversation_ids": self.get_conversation_ids(),
                "join_contract": plan_result.get("join_contract"),
            }
    
    def __call__(
        self, 
        plan: Dict[str, Any],
        writer=None,
    ) -> Dict[str, Any]:
        """Make agent callable with plan dictionary."""
        return self.synthesize_sql(plan, writer=writer)
