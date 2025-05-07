import asyncio
import logging
import json
import os
from typing import TypedDict, List, Optional, Dict, Any, Annotated # Added Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
# datetime and timezone are not strictly needed for this first step,
# but can be kept if your agent functions use them.
# from datetime import datetime, timezone

# --- Define Pydantic Models for the outputs of the first three agents ---
# From competitor.py (assuming these are the correct structures)
class CompetitorInfo(BaseModel): # Renamed from Competitor to avoid clash if Competitor is defined elsewhere
    name: str = Field(..., description="Name of the competitor")
    app_store_url: Optional[str] = Field(None, description="App Store URL (for apps)")
    google_play_url: Optional[str] = Field(None, description="Google Play URL (for apps)")

class CompetitorAgentResponse(BaseModel): # Renamed from CompetitorResponse
    query: str = Field(..., description="The user's original query")
    competitors: List[CompetitorInfo] = Field(..., description="List of competitors")

# From blog_url.py
class WebResult(BaseModel):
    title: str
    url: str

# From trend_analyzer.py
class RelatedQuery(BaseModel):
    query: str = Field(..., description="The related search query text")

class GoogleTrendsResult(BaseModel):
    keyword: str = Field(..., description="The keyword searched on Google Trends")
    related_queries_top: List[RelatedQuery] = Field(default_factory=list, description="List of top related queries found")
    related_queries_rising: List[RelatedQuery] = Field(default_factory=list, description="List of rising related queries found")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered during scraping")

# --- Import run functions from agent modules ---
# Ensure these paths are correct relative to where workflow.py is or adjust sys.path if needed
from competitor import run_agent as competitor_run_agent
from blog_url import run_search as blog_url_run_search
from trend_analyzer import run_trends_agent

# --- Reducer function for Annotated state fields ---
# Keeps the first value encountered, ignoring subsequent updates in the same step.
def _first_value(left: Any, right: Any) -> Any:
    return left if left is not None else right

# --- State Definition with Annotated fields ---
class LangGraphState(TypedDict):
    user_query: str
    scrapfly_api_key: Annotated[Optional[str], _first_value] # For blog_url_node
    competitor_data: Annotated[Optional[CompetitorAgentResponse], _first_value]
    blog_urls: Annotated[Optional[List[WebResult]], _first_value]
    trends: Annotated[Optional[GoogleTrendsResult], _first_value]
    error: Annotated[Optional[List[str]], lambda a, b: (a or []) + (b or [])] # Accumulate errors in a list
    # status: str # Status might be less relevant if we just run three and end

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Helper Function for Error Appending ---
def append_error_list(state: LangGraphState, new_error_msg: str):
    """Appends an error message to the state['error'] list."""
    current_errors = state.get('error', [])
    if not isinstance(current_errors, list): # Ensure it's a list
        current_errors = [str(current_errors)] if current_errors is not None else []
    current_errors.append(new_error_msg.strip())
    state['error'] = current_errors


# --- Node Functions (Return relevant part of the state to be updated) ---

# 1. Competitor Node
async def competitor_node(state: LangGraphState) -> Dict[str, Any]:
    logger.info(f"Competitor Node: Received state['user_query'] = '{state.get('user_query', 'NOT_FOUND')}'")
    logger.info("Running Competitor Agent...")
    try:
        result: Optional[CompetitorAgentResponse] = await competitor_run_agent(state['user_query'])
        if result:
            logger.info(f"Competitor Agent finished. Found: {[c.name for c in result.competitors]}")
            return {"competitor_data": result}
        else:
            logger.warning("Competitor Agent returned None.")
            append_error_list(state, "Competitor Agent returned None") # Modify state directly for error
            return {"competitor_data": CompetitorAgentResponse(query=state['user_query'], competitors=[])} # Return default
    except Exception as e:
        logger.error(f"Error in competitor_node: {e}", exc_info=True)
        append_error_list(state, f"Competitor Agent failed: {e}")
        return {"competitor_data": CompetitorAgentResponse(query=state['user_query'], competitors=[])} # Return default

# 2. Blog URL Node
async def blog_url_node(state: LangGraphState) -> Dict[str, Any]:
    logger.info(f"Blog URL Node: Received state['user_query'] = '{state.get('user_query', 'NOT_FOUND')}'")
    logger.info("Running Blog URL Agent...")
    scrapfly_api_key = state.get('scrapfly_api_key')
    if not scrapfly_api_key:
        logger.error("Missing Scrapfly API key for Blog URL Agent.")
        append_error_list(state, "Missing Scrapfly API key for Blog URL search")
        return {"blog_urls": []}
    try:
        result: List[WebResult] = await blog_url_run_search(state['user_query'], scrapfly_api_key)
        logger.info(f"Blog URL Agent finished. Found {len(result)} URLs.")
        return {"blog_urls": result}
    except Exception as e:
        logger.error(f"Error in blog_url_node: {e}", exc_info=True)
        append_error_list(state, f"Blog URL Agent failed: {e}")
        return {"blog_urls": []}

# 3. Trend Analyzer Node
async def trend_analyzer_node(state: LangGraphState) -> Dict[str, Any]:
    logger.info(f"Trend Analyzer Node: Received state['user_query'] = '{state.get('user_query', 'NOT_FOUND')}'")
    logger.info("Running Trend Analyzer Agent...")
    keyword = state['user_query']
    country_code = "US" # Or make this configurable from state if needed
    try:
        result: Optional[GoogleTrendsResult] = await run_trends_agent(keyword=keyword, country=country_code)
        if result:
            found_top = len(result.related_queries_top)
            found_rising = len(result.related_queries_rising)
            errors_count = len(result.errors)
            logger.info(f"Trend Analyzer finished. Top:{found_top}, Rising:{found_rising}, Errors:{errors_count}")
            if errors_count > 0:
                error_details = result.errors
                logger.warning(f"Trend Analyzer reported errors: {error_details}")
                append_error_list(state, f"Trend Analyzer Errors: {error_details}")
            return {"trends": result}
        else:
            logger.error("Trend Analyzer Agent returned None unexpectedly.")
            append_error_list(state, "Trend Analyzer node received None")
            return {"trends": GoogleTrendsResult(keyword=keyword, errors=["Agent returned None"])}
    except Exception as e:
        logger.error(f"Error in trend_analyzer_node: {e}", exc_info=True)
        error_msg = f"Trend Analyzer Node failed: {e}"
        append_error_list(state, error_msg)
        return {"trends": GoogleTrendsResult(keyword=state['user_query'], errors=[error_msg])}

# --- Graph Definition and Execution ---
workflow = StateGraph(LangGraphState)

# Add nodes for the first step
workflow.add_node("competitor", competitor_node)
workflow.add_node("blog_url", blog_url_node)
workflow.add_node("trend_analyzer", trend_analyzer_node)

# Set entry points: all three will run in parallel from the start
workflow.add_edge("__start__", "competitor")
workflow.add_edge("__start__", "blog_url")
workflow.add_edge("__start__", "trend_analyzer")

# End the graph after these three nodes complete
# When nodes don't have outgoing edges to other nodes within the graph (excluding END),
# their branches effectively conclude. The graph itself finishes when all
# started branches have concluded or reached an explicit END.
# For clarity, we can explicitly connect them to END.
workflow.add_edge("competitor", END)
workflow.add_edge("blog_url", END)
workflow.add_edge("trend_analyzer", END)

app = workflow.compile()

# --- Workflow Runner ---
async def run_workflow(query: str, scrapfly_key: Optional[str] = None):
    logger.info(f"DEBUG: run_workflow called with query: '{query}'")
    initial_state = LangGraphState(
        user_query=query,
        scrapfly_api_key=scrapfly_key,
        competitor_data=None,
        blog_urls=None,
        trends=None,
        error=None
    )
    logger.info(f"DEBUG: initial_state created: {initial_state}")
    
    logger.info(f"Starting workflow for query: '{initial_state['user_query']}'")
    final_state_values = {}
    try:
        final_output = await app.ainvoke(initial_state, {"recursion_limit": 10})
        final_state_values = final_output

        logger.info("Graph invocation complete.")
        final_errors = final_state_values.get('error')
        if final_errors:
            logger.error(f"Workflow finished with {len(final_errors)} error(s): {final_errors}")
        else:
            logger.info("Workflow finished successfully (initial three agents).")

    except Exception as graph_exec_error:
        logger.error(f"Exception during graph execution: {graph_exec_error}", exc_info=True)
        if not final_state_values:  # If invoke failed before returning anything
            final_state_values = initial_state  # Use initial state as a base
        current_errors = final_state_values.get('error', [])
        if not isinstance(current_errors, list): current_errors = []
        current_errors.append(f"Graph execution error: {graph_exec_error}")
        final_state_values['error'] = current_errors

    logger.info("--- Final State (First Step) ---")
    # Ensure complex Pydantic models are serializable for JSON output
    serializable_state = {}
    for key, value in final_state_values.items():
        if isinstance(value, BaseModel):
            serializable_state[key] = value.model_dump(exclude_none=True)
        elif isinstance(value, list) and value and all(isinstance(i, BaseModel) for i in value):
            serializable_state[key] = [i.model_dump(exclude_none=True) for i in value]
        else:
            serializable_state[key] = value
            
    print(json.dumps(serializable_state, indent=2, default=str))
    logger.info("--- End of Final State (First Step) ---")
    return final_state_values

# --- Main Execution Block ---
if __name__ == "__main__":
    user_query = "app for music" # Example query
    
    # Ensure API keys are loaded from .env or environment
    scrapfly_api_key_env = os.getenv("SCRAPFLY_API_KEY")
    openrouter_api_key_env = os.getenv("OPENROUTER_API_KEY") # Used by agents internally

    if not scrapfly_api_key_env:
        logger.warning("SCRAPFLY_API_KEY not found in environment. Blog URL agent will fail or be skipped.")
    if not openrouter_api_key_env:
        logger.error("OPENROUTER_API_KEY not found in environment. Agents will likely fail.")
        # Consider exiting if critical keys are missing for agent operation
        # exit(1)

    asyncio.run(run_workflow(query=user_query, scrapfly_key=scrapfly_api_key_env))