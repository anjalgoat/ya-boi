import asyncio
import logging
import json
import os
import sys
import subprocess
from typing import TypedDict, List, Optional, Dict, Any, Annotated

from pydantic import BaseModel, Field, HttpUrl # HttpUrl for summarizer's WebpageResult

from langgraph.graph import StateGraph, END

# --- Define Pydantic Models for ALL agent inputs/outputs handled by workflow ---

# From competitor.py
class CompetitorInfo(BaseModel):
    name: str
    app_store_url: Optional[str] = None
    google_play_url: Optional[str] = None

class CompetitorAgentResponse(BaseModel):
    query: str
    competitors: List[CompetitorInfo]

# From blog_url.py
class WebResult(BaseModel):
    title: str
    url: str # blog_url.py uses str

# From trend_analyzer.py
class RelatedQuery(BaseModel):
    query: str

class GoogleTrendsResult(BaseModel):
    keyword: str
    related_queries_top: List[RelatedQuery] = Field(default_factory=list)
    related_queries_rising: List[RelatedQuery] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

# For craw4ai.py output
class Craw4aiScrapeResult(BaseModel):
    url: str # Matched to craw4ai.py's ScrapeResult model which uses str
    title: str
    summary: str
    insight: str
    relevance: str

class Craw4aiScrapeResponse(BaseModel):
    results: List[Craw4aiScrapeResult]

# For dummy_reviews.py output
class DummyReview(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    text: str

class DummyAppReviewData(BaseModel): # Output of dummy_reviews.py
    competitor_name: str
    app_store_reviews: List[DummyReview] = Field(default_factory=list)
    google_play_reviews: List[DummyReview] = Field(default_factory=list)

# --- Models for summarizer.py and diagram_agent.py ---
# These are the *input* structures these scripts expect.

# Input structure for summarizer.py (FullMarketInput)
class SummarizerTrendItem(BaseModel): # As defined in summarizer.py
    description: str
    source: str
    metric: Optional[str] = None
    timestamp: Optional[str] = None

class SummarizerTrendsInput(BaseModel): # As defined in summarizer.py
    trends: List[SummarizerTrendItem]

class SummarizerWebpageResult(BaseModel): # As defined in summarizer.py
    url: HttpUrl # summarizer.py uses HttpUrl
    title: str
    summary: str
    insight: str
    relevance: str

class SummarizerWebpageInsightsInput(BaseModel): # As defined in summarizer.py
    results: List[SummarizerWebpageResult]

class SummarizerReviewItem(BaseModel): # As defined in summarizer.py
    rating: Optional[int] = None
    text: str

class SummarizerCompetitorReviews(BaseModel): # As defined in summarizer.py
    competitor_name: str
    app_store_reviews: Optional[List[SummarizerReviewItem]] = Field(default_factory=list)
    google_play_reviews: Optional[List[SummarizerReviewItem]] = Field(default_factory=list)

class SummarizerFullMarketInput(BaseModel): # Main input for summarizer.py
    user_query: str
    trends: Optional[SummarizerTrendsInput] = None
    webpage_insights: Optional[SummarizerWebpageInsightsInput] = None
    reviews: Optional[List[SummarizerCompetitorReviews]] = None

# Output structure from summarizer.py
class MarketSummaryReport(BaseModel): # As defined in summarizer.py
    original_query: str
    overall_market_summary: str
    key_market_trends: List[str]
    competitor_positioning_summary: str
    identified_gaps: List[str]
    strategic_opportunities: List[str]

# Input structure for diagram_agent.py (RawMarketDataInput is similar to FullMarketInput without user_query)
# Re-using Summarizer models for DiagramAgent input for simplicity as structures are identical
class DiagramAgentRawMarketDataInput(BaseModel): # Main input for diagram_agent.py
    reviews: Optional[List[SummarizerCompetitorReviews]] = None # from dummy_reviews
    trends: Optional[SummarizerTrendsInput] = None # from trend_analyzer
    webpage_insights: Optional[SummarizerWebpageInsightsInput] = None # from craw4ai

# Output structure from diagram_agent.py
class BarChartDataItem(BaseModel): # As defined in diagram_agent.py
    name: str
    review_count: int
    rating: Optional[float] = None
    market_share: Optional[float] = None

class GapMatrixDataItem(BaseModel): # As defined in diagram_agent.py
    feature: str
    unmet_need: str
    competitor_status: Dict[str, str]

class ChartDataResponse(BaseModel): # As defined in diagram_agent.py
    bar_chart_data: List[BarChartDataItem]
    gap_matrix_data: List[GapMatrixDataItem]
    suggested_bar_chart_metric: Optional[str] = None


# --- Import run functions ---
from competitor import run_agent as competitor_run_agent
from blog_url import run_search as blog_url_run_search
from trend_analyzer import run_trends_agent # Re-added

# --- Reducer ---
def _first_value(left: Any, right: Any) -> Any:
    return left if left is not None else right

# --- State Definition ---
class LangGraphState(TypedDict):
    user_query: str
    # scrapfly_api_key: Annotated[Optional[str], _first_value] # No longer primary for blog_url
    scraperapi_key: Annotated[Optional[str], _first_value] # ADDED for ScraperAPI
    competitor_data: Annotated[Optional[CompetitorAgentResponse], _first_value]
    blog_urls: Annotated[Optional[List[WebResult]], _first_value]
    trends_data: Annotated[Optional[GoogleTrendsResult], _first_value] # Re-added for trend_analyzer output

    scraped_web_content: Annotated[Optional[List[Craw4aiScrapeResult]], _first_value]
    generated_reviews: Annotated[Optional[List[DummyAppReviewData]], _first_value] # Output of dummy_reviews_node

    # Outputs of the new summarizer and diagram agent nodes
    market_summary_report: Annotated[Optional[MarketSummaryReport], _first_value]
    chart_data: Annotated[Optional[ChartDataResponse], _first_value]
    
    error: Annotated[Optional[List[str]], lambda a, b: (a or []) + (b or [])]

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Error Helper ---
def append_to_errors(current_errors: Optional[List[str]], new_error_msg: str) -> List[str]:
    current_errors = list(current_errors) if current_errors is not None else []
    current_errors.append(new_error_msg.strip())
    return current_errors

# --- Node Functions ---
async def competitor_node(state: LangGraphState) -> Dict[str, Any]:
    logger.critical("COMPETITOR NODE: If you see 401 errors, CHECK YOUR OPENROUTER_API_KEY!")
    logger.info(f"Competitor Node: Running for query '{state.get('user_query', 'NOT_FOUND')}'")
    errors_list = list(state.get('error', []))
    try:
        result: Optional[CompetitorAgentResponse] = await competitor_run_agent(state['user_query'])
        if result:
            logger.info(f"Competitor Agent finished. Found: {[c.name for c in result.competitors]}")
            if any("Exception" in c.name or "Failed" in c.name for c in result.competitors):
                 errors_list = append_to_errors(errors_list, "Competitor agent returned placeholder/error names (likely due to 401 API key error).")
            return {"competitor_data": result, "error": errors_list or None}
        else: # Should not happen if competitor_run_agent always returns a CompetitorAgentResponse
            logger.warning("Competitor Agent returned None unexpectedly.")
            errors_list = append_to_errors(errors_list, "Competitor Agent returned None")
            return {"competitor_data": CompetitorAgentResponse(query=state['user_query'], competitors=[]), "error": errors_list}
    except Exception as e:
        logger.error(f"Error in competitor_node: {e}", exc_info=True)
        errors_list = append_to_errors(errors_list, f"Competitor Agent failed: {e}")
        return {"competitor_data": CompetitorAgentResponse(query=state['user_query'], competitors=[]), "error": errors_list}

async def blog_url_node(state: LangGraphState) -> Dict[str, Any]:
    logger.info(f"Blog URL Node: Running for query '{state.get('user_query', 'NOT_FOUND')}'")
    errors_list = list(state.get('error', []))
    # MODIFIED: Use scraperapi_key from state
    key_for_blog_url = state.get('scraperapi_key') 
    if not key_for_blog_url:
        # MODIFIED: Updated error message
        logger.error("Missing ScraperAPI key for Blog URL Agent.")
        errors_list = append_to_errors(errors_list, "Missing ScraperAPI key for Blog URL search")
        return {"blog_urls": [], "error": errors_list}
    try:
        # MODIFIED: Pass the correct key
        result: List[WebResult] = await blog_url_run_search(state['user_query'], key_for_blog_url)
        logger.info(f"Blog URL Agent finished. Found {len(result)} URLs.")
        return {"blog_urls": result, "error": errors_list or None}
    except Exception as e:
        logger.error(f"Error in blog_url_node: {e}", exc_info=True)
        errors_list = append_to_errors(errors_list, f"Blog URL Agent failed: {e}")
        return {"blog_urls": [], "error": errors_list}

async def trend_analyzer_node(state: LangGraphState) -> Dict[str, Any]: # Re-added
    logger.critical("TREND ANALYZER NODE: If you see 401 errors, CHECK YOUR OPENROUTER_API_KEY!")
    logger.info(f"Trend Analyzer Node: Running for query '{state.get('user_query', 'NOT_FOUND')}'")
    errors_list = list(state.get('error', []))
    keyword = state['user_query']
    country_code = "US"
    # Note: trend_analyzer.py directly uses os.getenv("SCRAPERAPI_KEY").
    # If you want this node to also use a key from the state for consistency,
    # you'd need to modify trend_analyzer.py to accept the key as a parameter.
    # For now, it will continue to use its environment variable.
    # You could also pass state.get('scraperapi_key') to it if trend_analyzer.py is modified.
    try:
        result: Optional[GoogleTrendsResult] = await run_trends_agent(keyword=keyword, country=country_code)
        if result:
            logger.info(f"Trend Analyzer finished. Top:{len(result.related_queries_top)}, Rising:{len(result.related_queries_rising)}, Errors:{len(result.errors)}")
            if result.errors:
                errors_list.extend([f"Trend Analyzer Reported Error: {err}" for err in result.errors])
            return {"trends_data": result, "error": errors_list or None}
        else: # Should not happen if run_trends_agent always returns GoogleTrendsResult
            logger.error("Trend Analyzer Agent returned None unexpectedly.")
            errors_list = append_to_errors(errors_list, "Trend Analyzer node received None")
            return {"trends_data": GoogleTrendsResult(keyword=keyword, errors=["Agent returned None"]), "error": errors_list}
    except Exception as e:
        logger.error(f"Error in trend_analyzer_node: {e}", exc_info=True)
        error_msg = f"Trend Analyzer Node failed: {e}"
        errors_list = append_to_errors(errors_list, error_msg)
        return {"trends_data": GoogleTrendsResult(keyword=state['user_query'], errors=[error_msg]), "error": errors_list}

async def craw4ai_node(state: LangGraphState) -> Dict[str, Any]:
    logger.info("Craw4ai Node: Processing blog URLs...")
    errors_list = list(state.get('error', []))
    blog_urls_data = state.get('blog_urls')

    # craw4ai.py uses os.getenv("SCRAPFLY_API_KEY") internally.
    # If you want craw4ai to use ScraperAPI, you would need to:
    # 1. Modify craw4ai.py to accept an API key and use ScraperAPI.
    # 2. Pass state.get('scraperapi_key') to it.
    # For now, it will attempt to use Scrapfly key from its env vars.

    if not blog_urls_data:
        logger.warning("Craw4ai Node: No blog URLs found in state to process.")
        errors_list = append_to_errors(errors_list, "Craw4ai Node: No blog URLs to process")
        return {"scraped_web_content": [], "error": errors_list}

    input_for_craw4ai = [{"title": item.title, "url": item.url} for item in blog_urls_data]
    input_json_str = json.dumps(input_for_craw4ai)
    script_path = os.path.join(os.path.dirname(__file__), 'craw4ai.py')

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=input_json_str.encode())

        if process.returncode == 0:
            output_json_str = stdout.decode()
            parsed_output = Craw4aiScrapeResponse.model_validate_json(output_json_str)
            logger.info(f"Craw4ai Node: Successfully processed {len(parsed_output.results)} URLs from script output.")
            return {"scraped_web_content": parsed_output.results, "error": errors_list or None}
        else:
            error_message = stderr.decode()
            logger.error(f"Craw4ai Node: Script execution failed. Code: {process.returncode}. Error: {error_message}")
            errors_list = append_to_errors(errors_list, f"Craw4ai script failed: {error_message}")
            return {"scraped_web_content": [], "error": errors_list}
    except Exception as e:
        logger.error(f"Error in craw4ai_node: {e}", exc_info=True)
        errors_list = append_to_errors(errors_list, f"Craw4ai Node failed: {e}")
        return {"scraped_web_content": [], "error": errors_list}

async def dummy_reviews_node(state: LangGraphState) -> Dict[str, Any]:
    logger.info("Dummy Reviews Node: Processing competitor data...")
    errors_list = list(state.get('error', []))
    competitor_data_response = state.get('competitor_data')

    if not competitor_data_response or not competitor_data_response.competitors or \
       any("Exception" in c.name or "Failed" in c.name for c in competitor_data_response.competitors):
        logger.warning("Dummy Reviews Node: No valid competitor data from previous step. Skipping review generation.")
        errors_list = append_to_errors(errors_list, "Dummy Reviews Node: No valid competitor data to process")
        return {"generated_reviews": [], "error": errors_list}

    input_json_str = competitor_data_response.model_dump_json()
    script_path = os.path.join(os.path.dirname(__file__), 'dummy_reviews.py')

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=input_json_str.encode())

        if process.returncode == 0:
            output_json_str = stdout.decode()
            parsed_output_list = json.loads(output_json_str)
            generated_reviews = [DummyAppReviewData.model_validate(item) for item in parsed_output_list]
            logger.info(f"Dummy Reviews Node: Successfully generated reviews for {len(generated_reviews)} competitors.")
            return {"generated_reviews": generated_reviews, "error": errors_list or None}
        else:
            error_message = stderr.decode()
            logger.error(f"Dummy Reviews Node: Script execution failed. Code: {process.returncode}. Error: {error_message}")
            errors_list = append_to_errors(errors_list, f"Dummy Reviews script failed: {error_message}")
            return {"generated_reviews": [], "error": errors_list}
    except Exception as e:
        logger.error(f"Error in dummy_reviews_node: {e}", exc_info=True)
        errors_list = append_to_errors(errors_list, f"Dummy Reviews Node failed: {e}")
        return {"generated_reviews": [], "error": errors_list}

def _transform_google_trends_to_summarizer_trends(google_trends_data: Optional[GoogleTrendsResult]) -> Optional[SummarizerTrendsInput]:
    if not google_trends_data:
        return None
    
    trend_items: List[SummarizerTrendItem] = []
    for rq_top in google_trends_data.related_queries_top:
        trend_items.append(SummarizerTrendItem(description=rq_top.query, source="Google Trends - Top", metric=None, timestamp=None))
    for rq_rising in google_trends_data.related_queries_rising:
        trend_items.append(SummarizerTrendItem(description=rq_rising.query, source="Google Trends - Rising", metric=None, timestamp=None))
    
    if not trend_items and google_trends_data.keyword: # Add keyword itself if no related queries found but keyword exists
        trend_items.append(SummarizerTrendItem(description=f"Keyword: {google_trends_data.keyword}", source="Google Trends - Searched Keyword", metric=None, timestamp=None))
        
    return SummarizerTrendsInput(trends=trend_items) if trend_items else None

def _transform_craw4ai_to_summarizer_web(craw4ai_data: Optional[List[Craw4aiScrapeResult]]) -> Optional[SummarizerWebpageInsightsInput]:
    if not craw4ai_data:
        return None
    summarizer_web_results = []
    for item in craw4ai_data:
        try:
            if item.url and item.url.lower() != "n/a" and "example.com/not_available" not in item.url :
                 summarizer_web_results.append(SummarizerWebpageResult(
                    url=HttpUrl(item.url), 
                    title=item.title,
                    summary=item.summary,
                    insight=item.insight,
                    relevance=item.relevance
                ))
            else:
                logger.warning(f"Skipping invalid URL for summarizer input: {item.url}")
        except Exception as e_url: 
            logger.warning(f"Could not convert URL '{item.url}' to HttpUrl for summarizer: {e_url}")
            
    return SummarizerWebpageInsightsInput(results=summarizer_web_results) if summarizer_web_results else None

def _transform_dummy_reviews_to_summarizer_reviews(dummy_reviews_data: Optional[List[DummyAppReviewData]]) -> Optional[List[SummarizerCompetitorReviews]]:
    if not dummy_reviews_data:
        return None
    s_reviews: List[SummarizerCompetitorReviews] = []
    for dr_data in dummy_reviews_data:
        app_store = [SummarizerReviewItem(rating=r.rating, text=r.text) for r in dr_data.app_store_reviews]
        google_play = [SummarizerReviewItem(rating=r.rating, text=r.text) for r in dr_data.google_play_reviews]
        s_reviews.append(SummarizerCompetitorReviews(
            competitor_name=dr_data.competitor_name,
            app_store_reviews=app_store,
            google_play_reviews=google_play
        ))
    return s_reviews if s_reviews else None

async def summarizer_node(state: LangGraphState) -> Dict[str, Any]:
    logger.info("Summarizer Node: Preparing data and running summarizer.py...")
    errors_list = list(state.get('error', []))
    user_query = state.get('user_query', "N/A")
    
    trends_input = _transform_google_trends_to_summarizer_trends(state.get('trends_data'))
    web_insights_input = _transform_craw4ai_to_summarizer_web(state.get('scraped_web_content'))
    reviews_input = _transform_dummy_reviews_to_summarizer_reviews(state.get('generated_reviews'))

    if not trends_input and not web_insights_input and not reviews_input:
        logger.warning("Summarizer Node: No data from previous steps to summarize.")
        errors_list = append_to_errors(errors_list, "Summarizer Node: No input data available.")
        return {"market_summary_report": None, "error": errors_list}

    summarizer_payload = SummarizerFullMarketInput(
        user_query=user_query,
        trends=trends_input,
        webpage_insights=web_insights_input,
        reviews=reviews_input
    )
    input_json_str = summarizer_payload.model_dump_json(exclude_none=True)
    script_path = os.path.join(os.path.dirname(__file__), 'summarizer.py')

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=input_json_str.encode())

        if process.returncode == 0:
            output_json_str = stdout.decode()
            report = MarketSummaryReport.model_validate_json(output_json_str)
            logger.info("Summarizer Node: Successfully generated market summary report.")
            return {"market_summary_report": report, "error": errors_list or None}
        else:
            error_message = stderr.decode()
            logger.error(f"Summarizer Node: Script execution failed. Code: {process.returncode}. Error: {error_message}")
            errors_list = append_to_errors(errors_list, f"Summarizer script failed: {error_message}")
            return {"market_summary_report": None, "error": errors_list}
    except Exception as e:
        logger.error(f"Error in summarizer_node: {e}", exc_info=True)
        errors_list = append_to_errors(errors_list, f"Summarizer Node failed: {e}")
        return {"market_summary_report": None, "error": errors_list}

async def diagram_agent_node(state: LangGraphState) -> Dict[str, Any]:
    logger.info("Diagram Agent Node: Preparing data and running diagram_agent.py...")
    errors_list = list(state.get('error', []))

    trends_input = _transform_google_trends_to_summarizer_trends(state.get('trends_data')) 
    web_insights_input = _transform_craw4ai_to_summarizer_web(state.get('scraped_web_content')) 
    reviews_input = _transform_dummy_reviews_to_summarizer_reviews(state.get('generated_reviews')) 

    if not trends_input and not web_insights_input and not reviews_input:
        logger.warning("Diagram Agent Node: No data from previous steps to create diagrams from.")
        errors_list = append_to_errors(errors_list, "Diagram Agent Node: No input data available.")
        return {"chart_data": None, "error": errors_list}
        
    diagram_payload = DiagramAgentRawMarketDataInput(
        trends=trends_input,
        webpage_insights=web_insights_input,
        reviews=reviews_input
    )
    input_json_str = diagram_payload.model_dump_json(exclude_none=True)
    script_path = os.path.join(os.path.dirname(__file__), 'diagram_agent.py')

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=input_json_str.encode())

        if process.returncode == 0:
            output_json_str = stdout.decode()
            charts = ChartDataResponse.model_validate_json(output_json_str)
            logger.info("Diagram Agent Node: Successfully generated chart data.")
            return {"chart_data": charts, "error": errors_list or None}
        else:
            error_message = stderr.decode()
            logger.error(f"Diagram Agent Node: Script execution failed. Code: {process.returncode}. Error: {error_message}")
            errors_list = append_to_errors(errors_list, f"Diagram Agent script failed: {error_message}")
            return {"chart_data": None, "error": errors_list}
    except Exception as e:
        logger.error(f"Error in diagram_agent_node: {e}", exc_info=True)
        errors_list = append_to_errors(errors_list, f"Diagram Agent Node failed: {e}")
        return {"chart_data": None, "error": errors_list}

# --- Graph Definition ---
workflow = StateGraph(LangGraphState)
workflow.add_node("competitor", competitor_node)
workflow.add_node("blog_url", blog_url_node)
workflow.add_node("trend_analyzer", trend_analyzer_node) 
workflow.add_node("craw4ai_scraper", craw4ai_node)
workflow.add_node("dummy_review_generator", dummy_reviews_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("diagram_generator", diagram_agent_node)

# Entry points
# MODIFIED: Fully sequential start for initial agents
workflow.add_edge("__start__", "competitor")        # 1st: competitor starts
workflow.add_edge("competitor", "blog_url")       # 2nd: blog_url starts after competitor
workflow.add_edge("blog_url", "trend_analyzer")   # 3rd: trend_analyzer starts after blog_url

# Sequence
workflow.add_edge("blog_url", "craw4ai_scraper")
workflow.add_edge("competitor", "dummy_review_generator")

workflow.add_edge("craw4ai_scraper", "summarizer")
workflow.add_edge("dummy_review_generator", "summarizer")
workflow.add_edge("trend_analyzer", "summarizer")

workflow.add_edge("craw4ai_scraper", "diagram_generator")
workflow.add_edge("dummy_review_generator", "diagram_generator")
workflow.add_edge("trend_analyzer", "diagram_generator")

workflow.add_edge("summarizer", END)
workflow.add_edge("diagram_generator", END)

app = workflow.compile()

# --- Custom JSON Serializer ---
def pydantic_model_dumper(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump(exclude_none=True)
    try:
        return str(obj)
    except TypeError:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable and str() failed")

# --- Workflow Runner ---
# MODIFIED: run_workflow now accepts scraperapi_key
async def run_workflow(query: str, scraperapi_key_param: Optional[str] = None):
    logger.critical("IMPORTANT: Ensure OPENROUTER_API_KEY is valid and accessible to avoid 401 errors.")
    initial_state = LangGraphState(
        user_query=query,
        scraperapi_key=scraperapi_key_param, # MODIFIED: Use scraperapi_key_param
        competitor_data=None, blog_urls=None, trends_data=None, 
        scraped_web_content=None, generated_reviews=None,
        market_summary_report=None, chart_data=None, 
        error=[]
    )
    logger.info(f"DEBUG: initial_state created: {initial_state}")
    logger.info(f"Starting workflow for query: '{initial_state['user_query']}'")
    
    reconstructed_final_state = initial_state.copy()
    try:
        async for s_update_dict in app.astream(initial_state, {"recursion_limit": 20}): 
            logger.debug(f"Graph stream update (node output dict): {s_update_dict}")
            reconstructed_final_state = s_update_dict 

        logger.info("Graph invocation complete.")
        final_errors = reconstructed_final_state.get('error', [])
        if final_errors:
            logger.error(f"Workflow finished with {len(final_errors)} error(s): {final_errors}")
        else:
            logger.info("Workflow finished successfully (individual agent errors like 401 might still be present in 'error' list).")
        
        final_state_to_print = reconstructed_final_state

    except Exception as graph_exec_error:
        logger.error(f"Exception during graph execution: {graph_exec_error}", exc_info=True)
        final_state_to_print = initial_state.copy() 
        current_errors = list(final_state_to_print.get('error', []))
        current_errors = append_to_errors(current_errors, f"Graph execution error: {graph_exec_error}")
        final_state_to_print['error'] = current_errors
        
    logger.info("--- Final State ---")
    print(json.dumps(final_state_to_print, indent=2, default=pydantic_model_dumper))
    logger.info("--- End of Final State ---")
    return final_state_to_print

# --- Main Execution ---
if __name__ == "__main__":
    user_query = "app for music" 
    # REMOVED: scrapfly_api_key_env
    # ADDED: Get ScraperAPI key from environment
    scraperapi_key_env = os.getenv("SCRAPERAPI_KEY") 
    openrouter_api_key_env = os.getenv("OPENROUTER_API_KEY") 

    if not openrouter_api_key_env:
        logger.critical("FATAL: OPENROUTER_API_KEY not found. LLM agents WILL FAIL. Please set it.")
    # MODIFIED: Check for ScraperAPI key
    if not scraperapi_key_env:
        logger.warning("SCRAPERAPI_KEY not found in environment. Blog URL and Trend Analyzer agents might fail or use fallback.")
    else:
        logger.info("SCRAPERAPI_KEY found in environment and will be passed to the workflow.")


    scripts_to_check = ['craw4ai.py', 'dummy_reviews.py', 'competitor.py', 'blog_url.py', 'trend_analyzer.py', 'summarizer.py', 'diagram_agent.py']
    for script in scripts_to_check:
        if not os.path.exists(os.path.join(os.path.dirname(__file__), script)):
            logger.error(f"{script} not found. Subprocess/import calls may fail.")

    # MODIFIED: Pass the scraperapi_key_env to run_workflow
    asyncio.run(run_workflow(query=user_query, scraperapi_key_param=scraperapi_key_env))