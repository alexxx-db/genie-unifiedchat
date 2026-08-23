"""
Planning Agent for Multi-Agent System

This module contains the PlanningAgent class responsible for:
- Query analysis and execution planning
- Vector search for relevant Genie spaces
- Creating execution plans with join strategies
- Determining execution routes (table_route vs genie_route)

The PlanningAgent uses vector search to identify relevant Genie spaces and
creates detailed execution plans that guide subsequent SQL synthesis agents.

Example usage:
    from langchain_core.runnables import Runnable
    from databricks_langchain import ChatDatabricks
    
    llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")
    agent = PlanningAgent(
        llm=llm,
        vector_search_index="catalog.schema.index_name"
    )
    
    plan = agent("How many active plan members over 50 are on Lexapro?")
"""

import json
import logging
import re
from typing import Any, Dict, List

import mlflow
from databricks_langchain import VectorSearchRetrieverTool
from langchain_core.runnables import Runnable
from mlflow.entities import SpanType

from agent_server.databricks_runtime_auth import get_vector_search_workspace_client

logger = logging.getLogger(__name__)
_VECTOR_SEARCH_TOOL_NAME = "search_genie_spaces"


def _is_invalid_token_error(exc: Exception) -> bool:
    """Detect auth failures that are safe to retry with a fresh client."""
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = f"{type(current).__name__}: {current}".lower()
        if "invalid token" in message:
            return True
        current = current.__cause__ or current.__context__

    return False


class PlanningAgent:
    """
    Agent responsible for query analysis and execution planning.
    
    OOP design with vector search integration.
    """
    
    def __init__(self, llm: Runnable, vector_search_index: str):
        """
        Initialize PlanningAgent.
        
        Args:
            llm: Language model for planning (must support streaming)
            vector_search_index: Name of the vector search index to use
        """
        self.llm = llm
        self.vector_search_index = vector_search_index
        self.name = "Planning"

    def _build_vs_tool(self) -> VectorSearchRetrieverTool:
        return VectorSearchRetrieverTool(
            index_name=self.vector_search_index,
            tool_name=_VECTOR_SEARCH_TOOL_NAME,
            tool_description="Search for relevant Genie spaces by semantic similarity",
            num_results=5,
            columns=["space_id", "space_title", "searchable_content"],
            filters={"chunk_type": "space_summary"},
            query_type="ANN",
            include_score=True,
            workspace_client=get_vector_search_workspace_client(),
        )

    def search_relevant_spaces(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant Genie spaces using VectorSearchRetrieverTool.

        Args:
            query: User's question
            num_results: Number of results to return
            
        Returns:
            List of relevant space dictionaries with keys:
            - space_id: Genie space identifier
            - space_title: Human-readable space title
            - searchable_content: Space description/content
            - score: Relevance score from vector search
        """
        with mlflow.start_span(
            name="vector_search_spaces",
            span_type=SpanType.RETRIEVER,
            attributes={
                "index_name": self.vector_search_index,
                "num_results": num_results,
                "tool_name": _VECTOR_SEARCH_TOOL_NAME,
            },
        ) as span:
            span.set_inputs({"query": query, "k": num_results})
            docs = None
            for attempt in range(2):
                try:
                    vs_tool = self._build_vs_tool()
                    docs = vs_tool._vector_store.similarity_search(
                        query=query,
                        k=num_results,
                        filter={"chunk_type": "space_summary"},
                        query_type="ANN",
                    )
                    break
                except Exception as exc:
                    should_retry = attempt == 0 and _is_invalid_token_error(exc)
                    if not should_retry:
                        raise
                    logger.warning(
                        "Vector search auth failed for planning agent; "
                        "rebuilding client and retrying once: %s",
                        exc,
                    )

            if docs is None:
                raise RuntimeError("Vector search returned no results after retry.")

            relevant_spaces = []
            for doc in docs:
                space_id = doc.metadata.get("space_id", "")
                if not space_id:
                    continue
                relevant_spaces.append({
                    "space_id": space_id,
                    "space_title": doc.metadata.get("space_title", ""),
                    "searchable_content": doc.page_content,
                    "score": doc.metadata.get("score", 0.0),
                })

            span.set_outputs(
                {
                    "result_count": len(relevant_spaces),
                    "spaces": [
                        {"space_id": s["space_id"], "space_title": s["space_title"], "score": s["score"]}
                        for s in relevant_spaces
                    ],
                }
            )

        return relevant_spaces
    
    def create_execution_plan(
        self, 
        query: str, 
        relevant_spaces: List[Dict[str, Any]],
        original_query: str = None,
        force_synthesis_route: str = "auto",
    ) -> Dict[str, Any]:
        """
        Create execution plan based on query and relevant spaces.
        
        Args:
            query: User's question (may be context_summary if available)
            relevant_spaces: List of relevant Genie spaces from vector search
            original_query: Original user query from this turn (before context enrichment)
            
        Returns:
            Dictionary with execution plan containing:
            - original_query: Original user query
            - vector_search_relevant_spaces_info: Mapping of space_id to space_title
            - question_clear: Boolean indicating if question is clear
            - sub_questions: List of sub-questions identified
            - requires_multiple_spaces: Boolean indicating if multiple spaces needed
            - relevant_space_ids: List of space IDs needed for execution
            - requires_join: Boolean indicating if data join is needed
            - join_strategy: "table_route" or "genie_route"
            - execution_plan: Brief description of execution plan
            - genie_execution_mode: "parallel" or "dag" (genie_route only; not UI SQL execution_mode)
            - genie_route_plan: space_id → question string OR structured DAG step
            - dependency_edges: [{from, to}, ...] for genie_route DAG (optional; derived if omitted)
        """
        # Use original_query if provided, otherwise use query as original
        original_query_display = original_query if original_query is not None else query
        forced_route_display = force_synthesis_route or "auto"
        forced_route_instructions = ""
        if forced_route_display == "genie_route":
            forced_route_instructions = """
- UI override: Genie Route is selected.
- You must use Genie Route and must create `genie_route_plan`.
"""
        elif forced_route_display == "table_route":
            forced_route_instructions = """
- UI override: Table Route is selected.
- You must use Table Route and must keep `genie_route_plan` null.
- Set genie_execution_mode to null.
"""

        spaces_info = [
            {"space_id": sp["space_id"], "space_title": sp["space_title"]}
            for sp in relevant_spaces
        ]
        expected_json_example = json.dumps(
            {
                "original_query": query,
                "vector_search_relevant_spaces_info": spaces_info,
                "question_clear": True,
                "sub_questions": ["sub-question 1", "sub-question 2"],
                "requires_multiple_spaces": False,
                "relevant_space_ids": ["space_id_1", "space_id_2"],
                "requires_join": False,
                "join_strategy": "table_route",
                "execution_plan": "Brief description of execution plan",
                "genie_execution_mode": None,
                "genie_route_plan": None,
                "dependency_edges": [],
            },
            indent=2,
        )
        dag_example = json.dumps(
            {
                "space_id_1": {
                    "question": "Top 10 drugs by total cost. Please limit to top 10 rows",
                    "depends_on": [],
                    "inject": ["few_shot", "ids", "filters"],
                },
                "space_id_2": {
                    "question": (
                        "Diagnoses associated with the provided drug IDs. "
                        "Please limit to top 10 rows"
                    ),
                    "depends_on": ["space_id_1"],
                    "inject": ["few_shot", "ids", "filters"],
                },
            },
            indent=2,
        )

        planning_prompt = f"""
You are a query planning expert. Analyze the following question and create an execution plan.

User original query this turn: {original_query_display}

Question: {query}

Forced synthesis route from UI: {forced_route_display}

Potentially relevant Genie spaces:
{json.dumps(relevant_spaces, indent=2)}

Break down the question and determine:
1. What are the sub-questions or analytical components?
2. How many Genie spaces are needed to answer completely? (List their space_ids)
3. If multiple spaces are needed, do we need to JOIN data across them? Reasoning whether the sub-questions are totally independent without joining need.
    - JOIN needed: E.g., "How many active plan members over 50 are on Lexapro?" requires joining member data with pharmacy claims.
    - No need for JOIN: E.g., "How many active plan members over 50? How much total cost for all Lexapro claims?" - Two independent questions.
4. If JOIN is needed, what's the best strategy:
    - "table_route": Directly synthesize SQL across multiple tables
    - "genie_route": Query each Genie Space Agent separately, then combine SQL queries
    - If user explicitly asks for "genie_route", use it; otherwise, use "table_route"
    - If Genie Route is selected from UI, you must use Genie Route and must create `genie_route_plan`
    - always populate the join_strategy field in the JSON output.
5. Execution plan: A brief description of how to execute the plan.
6. For genie_route ONLY — choose genie_execution_mode, genie_route_plan, and dependency_edges:
    - "parallel": Independent space questions (no answer from space A needed to ask space B).
      Use flat map: {{"space_id_1": "partial_question_1", "space_id_2": "partial_question_2"}}
      Set dependency_edges to [].
    - "dag": Staged / dependent questions where later Genie prompts need concrete values
      from earlier Genie answers (IDs, filters, top-N lists, SQL predicates).
      Examples that REQUIRE dag: "top 10 drugs and their diagnoses",
      "highest-cost members then their claims", "find codes then look up descriptions in another space".
      Use structured steps:
{dag_example}
      - depends_on: list of upstream space_ids (empty for roots)
      - inject: subset of ["few_shot","ids","filters","sql_preview","answer_summary"]
        - few_shot: labeled Q/SQL/keys example from the upstream space (default for dependents)
        - ids + few_shot: "find X then look up Y" (top-N codes → details in another space)
        - filters + sql_preview: preserve a time window / grain / predicate
        - answer_summary: only when a short narrative helps; omit if it is noise
      - ALSO set dependency_edges, e.g. [{{"from": "space_id_1", "to": "space_id_2"}}]
        (must match depends_on; do not invent spaces not in relevant_space_ids)
    - For table_route: genie_route_plan=null, genie_execution_mode=null, dependency_edges=[]
    - Each question should be scoped to that space
    - Add "Please limit to top 10 rows" to each question
    - Prefer dag over parallel when one space's results constrain another space's question
    - Never mark staged top-N / "and their" / "for those" questions as parallel
    - Do NOT reuse a Genie conversation_id across different spaces
{forced_route_instructions}

Return your analysis as JSON:
{expected_json_example}

Only return valid JSON, no explanations.
"""
        
        # Stream LLM response for immediate first token emission
        print("🤖 Streaming planning LLM call...")
        content = ""
        for chunk in self.llm.stream(planning_prompt):
            if chunk.content:
                content += chunk.content
        
        content = content.strip()
        print(f"✓ Planning stream complete ({len(content)} chars)")
        
        # Use regex to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # No code blocks, assume entire content is JSON
            json_str = content
        
        # Remove any trailing commas before ] or }
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        from ..utils.genie_route_dag import finalize_planner_genie_plan

        try:
            plan_result = json.loads(json_str)
            return finalize_planner_genie_plan(plan_result)
        except json.JSONDecodeError as e:
            print(f"❌ Planning JSON parsing error at position {e.pos}: {e.msg}")
            print(f"Raw content (first 500 chars):\n{content[:500]}")
            print(f"Cleaned JSON (first 500 chars):\n{json_str[:500]}")
            
            # Try one more time with even more aggressive cleaning
            try:
                # Remove comments
                json_str_clean = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
                # Remove trailing commas again
                json_str_clean = re.sub(r',(\s*[}\]])', r'\1', json_str_clean)
                plan_result = json.loads(json_str_clean)
                print("✓ Successfully parsed JSON after aggressive cleaning")
                return finalize_planner_genie_plan(plan_result)
            except Exception:
                raise e  # Re-raise original error
    
    def __call__(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and create execution plan.
        
        This method combines vector search and execution plan creation.
        
        Args:
            query: User's question
            
        Returns:
            Complete execution plan with relevant spaces
        """
        # Search for relevant spaces
        relevant_spaces = self.search_relevant_spaces(query)
        
        # Create execution plan
        plan = self.create_execution_plan(query, relevant_spaces)
        
        return plan
