import asyncio
import logging
import json
import os
# --- Add Annotated import ---
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from datetime import datetime, timezone

# --- Define Shared Pydantic Models ---
class AgentTaskContext(BaseModel):
    business_type: Optional[str] = None
    target_location: Optional[str] = None
    target_audience: Optional[str] = None

# --- Corrected Imports from Agent Modules ---
# From competitor.py
from competitor import run_agent as competitor_run_agent
from competitor import CompetitorResponse as CompetitorAgentResponse # Alias
from competitor import Competitor as CompetitorInfo # Alias

# From blog_url.py
from blog_url import run_search as blog_url_run_search, WebResult

# From trend_analyzer.py
from trend_analyzer import run_trends_agent, GoogleTrendsResult, RelatedQuery

# From dummy_reviews.py
from dummy_reviews import generate_fake_reviews_for_app, AppReviewData, Review
from dummy_reviews import Competitor as DummyReviewsCompetitorInput # Alias

# From craw4ai.py
from craw4ai import scraper_agent, UrlInput, ScrapeResponse, ScrapeResult

# From summarizer.py
from summarizer import run_synthesizer, MarketSummaryReport, FullMarketInput
from summarizer import TrendsInput as SummarizerTrendsInput # Alias
from summarizer import TrendItem as SummarizerTrendItem # Alias
from summarizer import WebpageInsightsInput as SummarizerWebpageInsightsInput # Alias
from summarizer import WebpageResult as SummarizerWebpageResult # Alias
from summarizer import CompetitorReviews as SummarizerCompetitorReviews # Alias
from summarizer import ReviewItem as SummarizerReviewItem # Alias

# From diagram_agent.py
from diagram_agent import diagram_agent, ChartDataResponse
# Define missing models used by diagram_node locally
class CompetitorData(BaseModel):
    name: str
    review_count: int
    rating: Optional[float] = None
    market_share: Optional[float] = None

class MarketGapData(BaseModel):
    feature: str
    unmet_need_level: str
    competitor_support: List[str]

# --- Reducer function for Annotated state fields ---
# Keeps the first value encountered, ignoring subsequent updates in the same step.
def _first_value(left: Any, right: Any) -> Any:
    return left if left is not None else right # Prioritize existing value if not None

# --- State Definition with Annotated fields ---
class LangGraphState(TypedDict):
    # Annotate all fields potentially updated by parallel initial nodes
    user_query: Annotated[str, _first_value]
    competitor_data: Annotated[Optional[CompetitorAgentResponse], _first_value]
    blog_urls: Annotated[Optional[List[WebResult]], _first_value]
    trends: Annotated[Optional[GoogleTrendsResult], _first_value]
    # --- Fields updated later don't need annotation for this specific issue ---
    reviews: Optional[List[AppReviewData]]
    webpage_insights: Optional[ScrapeResponse]
    summary_report: Optional[MarketSummaryReport]
    chart_data: Optional[ChartDataResponse]
    # --- Error field might receive updates from multiple branches, needs careful handling ---
    # Using a simple list accumulator might be better for errors
    # error: Optional[str] # Let's change this
    error: Annotated[Optional[List[str]], lambda a, b: (a or []) + (b or [])] # Accumulate errors in a list
    status: str # Assume status is managed sequentially or okay with last value
    scrapfly_api_key: Annotated[Optional[str], _first_value]


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Helper Function for Error Appending (Now appends to list) ---
def append_error_list(state: LangGraphState, new_error_msg: str):
    """Appends an error message to the state['error'] list."""
    if 'error' not in state or state['error'] is None:
        state['error'] = []
    # Ensure state['error'] is treated as a list for append
    if isinstance(state['error'], list):
        state['error'].append(new_error_msg.strip())
    else: # Should not happen with Annotated lambda, but safeguard
        state['error'] = [str(state['error']), new_error_msg.strip()]


# --- Node Functions (Return full state, rely on Annotated merge) ---

# 1. Competitor Node
async def competitor_node(state: LangGraphState) -> LangGraphState:
    logger.info("Running Competitor Agent...")
    try:
        result: Optional[CompetitorAgentResponse] = await competitor_run_agent(state['user_query'])
        if result:
            state['competitor_data'] = result
            logger.info(f"Competitor Agent finished. Found: {[c.name for c in result.competitors]}")
        else:
             logger.warning("Competitor Agent returned None.")
             append_error_list(state, "Competitor Agent returned None")
             # Ensure field exists even if None, Annotated handles merge
             state['competitor_data'] = CompetitorAgentResponse(query=state['user_query'], competitors=[])
    except Exception as e:
        logger.error(f"Error in competitor_node: {e}", exc_info=True)
        append_error_list(state, f"Competitor Agent failed: {e}")
        state['competitor_data'] = CompetitorAgentResponse(query=state['user_query'], competitors=[])
    return state

# 2. Blog URL Node
async def blog_url_node(state: LangGraphState) -> LangGraphState:
    logger.info("Running Blog URL Agent...")
    scrapfly_api_key = state.get('scrapfly_api_key', None)
    if not scrapfly_api_key:
         logger.error("Missing Scrapfly API key for Blog URL Agent.")
         append_error_list(state, "Missing Scrapfly API key")
         state['blog_urls'] = []
         return state
    try:
        result: List[WebResult] = await blog_url_run_search(state['user_query'], scrapfly_api_key)
        state['blog_urls'] = result
        logger.info(f"Blog URL Agent finished. Found {len(result)} URLs.")
    except Exception as e:
        logger.error(f"Error in blog_url_node: {e}", exc_info=True)
        append_error_list(state, f"Blog URL Agent failed: {e}")
        state['blog_urls'] = []
    return state

# 3. Trend Analyzer Node
async def trend_analyzer_node(state: LangGraphState) -> LangGraphState:
    logger.info("Running Trend Analyzer Agent (run_trends_agent)...")
    keyword = state['user_query']; country_code = "US"
    try:
        result: Optional[GoogleTrendsResult] = await run_trends_agent(keyword=keyword, country=country_code)
        if result:
             state['trends'] = result
             found_top = len(result.related_queries_top); found_rising = len(result.related_queries_rising); errors_count = len(result.errors)
             logger.info(f"Trend Analyzer finished. Top:{found_top}, Rising:{found_rising}, Errors:{errors_count}")
             if errors_count > 0:
                  error_details = result.errors
                  logger.warning(f"Trend Analyzer reported errors: {error_details}")
                  append_error_list(state, f"Trend Analyzer Errors: {error_details}")
        else:
             logger.error("Trend Analyzer Agent returned None unexpectedly.")
             append_error_list(state, "Trend Analyzer node received None")
             state['trends'] = GoogleTrendsResult(keyword=keyword, errors=["Agent returned None"])
    except Exception as e:
        logger.error(f"Error in trend_analyzer_node: {e}", exc_info=True)
        error_msg = f"Trend Analyzer Node failed: {e}"
        append_error_list(state, error_msg)
        state['trends'] = GoogleTrendsResult(keyword=state['user_query'], errors=[error_msg])
    return state

# 4. Dummy Reviews Node
async def dummy_reviews_node(state: LangGraphState) -> LangGraphState:
    logger.info("Running Dummy Reviews Agent...")
    if not state.get('competitor_data') or not state['competitor_data'].competitors:
        logger.warning("Missing or empty competitor data for Dummy Reviews. Skipping.")
        state['reviews'] = []
        return state
    try:
        tasks = [generate_fake_reviews_for_app(DummyReviewsCompetitorInput(name=c.name, app_store_url=c.app_store_url, google_play_url=c.google_play_url)) for c in state['competitor_data'].competitors]
        results: List[Optional[AppReviewData]|Exception] = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = []; task_errors = []
        for i, res in enumerate(results):
             comp_name = state['competitor_data'].competitors[i].name
             if isinstance(res, AppReviewData): valid_results.append(res); logger.info(f"Generated reviews for {comp_name}")
             elif isinstance(res, Exception): err_detail = f"Reviews failed for {comp_name}: {res}"; logger.error(err_detail, exc_info=False); task_errors.append(err_detail)
             elif res is None: warn_detail = f"Reviews skipped/failed for {comp_name}"; logger.warning(warn_detail); task_errors.append(warn_detail)
             else: err_detail = f"Reviews unexpected result for {comp_name}"; logger.error(f"Unexpected result type '{type(res)}'"); task_errors.append(err_detail)
        state['reviews'] = valid_results
        if task_errors: append_error_list(state, f"Dummy Reviews errors: {'; '.join(task_errors)}")
        logger.info(f"Dummy Reviews Agent finished. Generated data for {len(valid_results)} competitors.")
    except Exception as e:
        logger.error(f"Error in dummy_reviews_node: {e}", exc_info=True)
        append_error_list(state, f"Dummy Reviews Agent failed globally: {e}")
        state['reviews'] = []
    return state

# 5. Craw4AI Node
async def craw4ai_node(state: LangGraphState) -> LangGraphState:
    logger.info("Running Craw4AI Agent...")
    if not state.get('blog_urls'):
        logger.warning("Missing blog URLs for Craw4AI. Skipping.")
        state['webpage_insights'] = ScrapeResponse(results=[])
        return state
    urls_to_process: List[UrlInput] = []
    try:
        valid_urls = [url for url in state['blog_urls'] if url and url.url and url.title]
        if not valid_urls: logger.info("No valid blog URLs found."); state['webpage_insights'] = ScrapeResponse(results=[]); return state
        urls_to_process = [UrlInput(url=r.url, title=r.title) for r in valid_urls]
    except Exception as e:
         logger.error(f"Failed to create UrlInput objects: {e}")
         append_error_list(state, f"Craw4AI invalid blog_urls structure: {e}")
         state['webpage_insights'] = ScrapeResponse(results=[])
         return state
    try:
        input_str = f"Process URLs for {state['user_query']} research:\n" + "\n".join([f"- {u.title}: {u.url}" for u in urls_to_process])
        response_container = await scraper_agent.run(input_str)
        if response_container and isinstance(response_container.data, ScrapeResponse):
            state['webpage_insights'] = response_container.data
            logger.info(f"Craw4AI finished. Processed {len(urls_to_process)} URLs, got {len(response_container.data.results)} results.")
        else:
             error_details = getattr(response_container, 'error', 'Unknown error'); data_type = type(getattr(response_container, 'data', None))
             logger.error(f"Craw4AI failed or returned unexpected data. Type: {data_type}, Error: {error_details}")
             append_error_list(state, f"Craw4AI Agent failed: {error_details}")
             state['webpage_insights'] = ScrapeResponse(results=[])
    except Exception as e:
        logger.error(f"Error in craw4ai_node: {e}", exc_info=True)
        append_error_list(state, f"Craw4AI Agent failed: {e}")
        state['webpage_insights'] = ScrapeResponse(results=[])
    return state

# 6. Summarizer Node
async def summarizer_node(state: LangGraphState) -> LangGraphState:
    logger.info("Running Summarizer Agent...")
    required = ['webpage_insights', 'reviews']; missing = [k for k in required if state.get(k) is None]
    if missing: append_error_list(state, f"Missing data for Summarizer: {missing}"); return state
    if state.get('trends') is None: logger.warning("Trends data missing for Summarizer.")
    try:
        # Convert Trends
        summarizer_trends = None; trend_items = []
        trends_res = state.get('trends')
        if trends_res:
            if trends_res.related_queries_top: trend_items.extend([SummarizerTrendItem(description=f"Top related query: '{q.query}'", source="Google Trends", metric="Top") for q in trends_res.related_queries_top])
            if trends_res.related_queries_rising: trend_items.extend([SummarizerTrendItem(description=f"Rising related query: '{q.query}'", source="Google Trends", metric="Rising") for q in trends_res.related_queries_rising])
            if trends_res.errors: trend_items.extend([SummarizerTrendItem(description=f"Trends scraping issue: {err}", source="Google Trends Error") for err in trends_res.errors])
        if trend_items: summarizer_trends = SummarizerTrendsInput(trends=trend_items)
        # Convert Webpage Insights
        summarizer_insights = None
        if state['webpage_insights'] and state['webpage_insights'].results: insights = [SummarizerWebpageResult(url=str(r.url), title=r.title, summary=r.summary, insight=r.insight, relevance=r.relevance) for r in state['webpage_insights'].results]; summarizer_insights = SummarizerWebpageInsightsInput(results=insights)
        # Convert Reviews
        summarizer_reviews = None
        if state['reviews']:
            reviews_list = []
            for rd in state['reviews']:
                asr = [SummarizerReviewItem(rating=r.rating, text=r.text) for r in rd.app_store_reviews or []]; gpr = [SummarizerReviewItem(rating=r.rating, text=r.text) for r in rd.google_play_reviews or []]
                reviews_list.append(SummarizerCompetitorReviews(competitor_name=rd.competitor_name, app_store_reviews=asr, google_play_reviews=gpr))
            summarizer_reviews = reviews_list
        # Run Synthesizer
        full_input = FullMarketInput(user_query=state['user_query'], trends=summarizer_trends, webpage_insights=summarizer_insights, reviews=summarizer_reviews)
        result: MarketSummaryReport = await run_synthesizer(full_input)
        state['summary_report'] = result
        logger.info("Summarizer Agent finished.")
    except Exception as e:
        logger.error(f"Error in summarizer_node: {e}", exc_info=True)
        append_error_list(state, f"Summarizer Agent failed: {e}")
    return state

# 7. Diagram Node
async def diagram_node(state: LangGraphState) -> LangGraphState:
    logger.info("Running Diagram Agent...")
    required = ['webpage_insights', 'reviews']; missing = [k for k in required if state.get(k) is None]
    if missing: append_error_list(state, f"Missing data for Diagram Agent: {missing}"); return state
    if state.get('trends') is None: logger.warning("Trends data missing for Diagram Agent.")
    try:
        # Prepare input
        trends_for_prompt = None; trends_res = state.get('trends')
        if trends_res:
            trends_list = []
            if trends_res.related_queries_top: trends_list.append({"source": "Top Queries", "queries": [q.query for q in trends_res.related_queries_top]})
            if trends_res.related_queries_rising: trends_list.append({"source": "Rising Queries", "queries": [q.query for q in trends_res.related_queries_rising]})
            if trends_res.errors: trends_list.append({"source": "Errors", "details": trends_res.errors})
            if trends_list: trends_for_prompt = {"google_trends_summary": trends_list}
        input_dict = {"reviews": [r.model_dump(exclude_none=True) for r in state['reviews']] if state['reviews'] else None, "trends": trends_for_prompt, "webpage_insights": state['webpage_insights'].model_dump(exclude_none=True) if state['webpage_insights'] else None}
        # Format prompt
        prompt = f"Analyze market data for '{state['user_query']}' & generate ChartDataResponse JSON:\n\n"; prompt_parts = []
        if input_dict.get("reviews"): prompt_parts.append("--- Reviews ---\n" + json.dumps(input_dict["reviews"], indent=2))
        if input_dict.get("trends"): prompt_parts.append("--- Trends (Google Queries/Errors) ---\n" + json.dumps(input_dict["trends"], indent=2))
        if input_dict.get("webpage_insights"): prompt_parts.append("--- Web Insights ---\n" + json.dumps(input_dict["webpage_insights"], indent=2))
        prompt += "\n\n".join(prompt_parts) + "\n\n--- End Data ---\nGenerate JSON."
        # Run agent
        response_container = await diagram_agent.run(prompt)
        if response_container and isinstance(response_container.data, ChartDataResponse):
            state['chart_data'] = response_container.data; logger.info("Diagram Agent finished.")
        else:
            error_details = getattr(response_container, 'error', 'Unknown'); data_type = type(getattr(response_container, 'data', None))
            logger.error(f"Diagram Agent failed. Type: {data_type}, Error: {error_details}")
            append_error_list(state, f"Diagram Agent failed: {error_details}")
    except Exception as e:
        logger.error(f"Error in diagram_node: {e}", exc_info=True)
        append_error_list(state, f"Diagram Agent failed: {e}")
    return state

# --- Graph Definition and Execution ---
workflow = StateGraph(LangGraphState)
# Add nodes
workflow.add_node("competitor", competitor_node)
workflow.add_node("blog_url", blog_url_node)
workflow.add_node("trend_analyzer", trend_analyzer_node)
workflow.add_node("dummy_reviews", dummy_reviews_node)
workflow.add_node("craw4ai", craw4ai_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("diagram", diagram_node)
# Define edges
workflow.add_edge("competitor", "dummy_reviews")
workflow.add_edge("blog_url", "craw4ai")
workflow.add_edge("trend_analyzer", "summarizer")
workflow.add_edge("dummy_reviews", "summarizer")
workflow.add_edge("craw4ai", "summarizer")
workflow.add_edge("trend_analyzer", "diagram")
workflow.add_edge("dummy_reviews", "diagram")
workflow.add_edge("craw4ai", "diagram")
workflow.add_edge("summarizer", END)
workflow.add_edge("diagram", END)
# Set entry points
workflow.set_entry_point("competitor")
workflow.add_edge("__start__", "blog_url")
workflow.add_edge("__start__", "trend_analyzer")
app = workflow.compile()

# --- Workflow Runner ---
async def run_workflow(query: str, api_key: Optional[str] = None):
    logger.info(f"Starting workflow for query: '{query}'")
    # Initialize state with error as None
    initial_state = LangGraphState(
        user_query=query, status="pending", competitor_data=None, blog_urls=None,
        trends=None, reviews=None, webpage_insights=None, summary_report=None,
        chart_data=None, error=None, scrapfly_api_key=api_key
    )
    logger.info("Invoking graph...")
    final_state = {}
    try:
        final_state = await app.ainvoke(initial_state, {"recursion_limit": 25})
        logger.info("Graph invocation complete.")
        # Check final state for errors accumulated in the list
        final_errors = final_state.get('error')
        if final_errors:
            logger.error(f"Workflow finished with {len(final_errors)} error(s): {final_errors}")
        else:
            logger.info("Workflow finished successfully.")
    except Exception as graph_exec_error:
        logger.error(f"Exception during graph execution: {graph_exec_error}", exc_info=True)
        if not final_state: final_state = initial_state
        # Ensure error is a list before appending
        current_errors = final_state.get('error')
        if not isinstance(current_errors, list): current_errors = []
        current_errors.append(f"Graph execution error: {graph_exec_error}")
        final_state['error'] = current_errors # Assign the list back
    logger.info("--- Final State ---")
    print(json.dumps(final_state, indent=2, default=str)) # default=str handles complex types
    logger.info("--- End of Final State ---")
    return final_state

# --- Main Execution Block ---
if __name__ == "__main__":
    user_query = "app for music"
    api_key = os.getenv("SCRAPFLY_API_KEY")
    if not api_key: logger.warning("SCRAPFLY_API_KEY not found.")
    if not os.getenv("OPENROUTER_API_KEY"): logger.error("OPENROUTER_API_KEY not found."); exit(1)
    asyncio.run(run_workflow(query=user_query, api_key=api_key))