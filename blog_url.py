import logging
import os
import sys
import asyncio
import json
from typing import List, Optional, Dict

from bs4 import BeautifulSoup # Keep for potential pre-cleaning if needed
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, HttpUrl, field_validator
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
import httpx
# MODIFIED: Import urlencode
from urllib.parse import quote_plus, unquote, urlencode

# --- Standard Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# SCRAPERAPI_KEY will be passed into run_search

# --- Pydantic Models ---
class WebResult(BaseModel):
    """Represents a single found web result (URL and title)."""
    title: str = Field(..., description="The accurately extracted title of the search result.")
    url: str = Field(..., description="The fully qualified URL of the search result.")

    @field_validator('url')
    def validate_url_format(cls, value):
        if not value.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return value

class HTMLParseResult(BaseModel):
    """Structured output from the LLM that parses Google Search HTML."""
    extracted_results: List[WebResult] = Field(default_factory=list, description="List of web results extracted from HTML.")
    parsing_errors: List[str] = Field(default_factory=list)

class BlogURLsOutput(BaseModel):
    """Defines the structured output for the main Blog URL Finder Agent."""
    user_query: str = Field(description="The original user query.")
    search_keywords_used: str = Field(description="The effective search keywords used by the agent.")
    found_urls: List[WebResult] = Field(default_factory=list, description="List of relevant blog URLs and titles found, max 5.")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered by the main agent or its tools.")
    tool_logs: List[str] = Field(default_factory=list, description="Log messages from the search and parsing tool.")

# --- Global LLM Model Instance (Initialize once) ---
LLM_MODEL_INSTANCE = None
HTML_PARSING_LLM_INSTANCE = None 

if OPENROUTER_API_KEY:
    try:
        LLM_MODEL_INSTANCE = OpenAIModel(
            api_key=OPENROUTER_API_KEY,
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        )
        HTML_PARSING_LLM_INSTANCE = OpenAIModel(
            api_key=OPENROUTER_API_KEY,
            model_name=os.getenv("HTML_PARSING_MODEL", "gpt-3.5-turbo"), 
            base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        )
        logger.info("LLM instances for Blog URL agent initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAIModel instances for Blog URL Agent: {e}")
else:
    logger.error("OPENROUTER_API_KEY not found. Blog URL Agent will not function.")


# --- Tool Implementation: Fetch and Parse Search Results ---
async def _fetch_google_search_html(
    search_keywords: str,
    scraperapi_key: str,
    http_client: httpx.AsyncClient,
    tool_logs: List[str] 
) -> Optional[str]:
    """Fetches HTML content of Google search results page using ScraperAPI."""
    tool_logs.append(f"Fetching Google search HTML for keywords: '{search_keywords}'")
    if not scraperapi_key:
        tool_logs.append("ScraperAPI key missing in _fetch_google_search_html.")
        return None

    # MODIFIED: Robust URL construction using urlencode
    base_google_search_domain = "https://www.google.com/search"
    excluded_sites_and_paths = [
        "-site:youtube.com", "-site:amazon.com", "-site:wikipedia.org",
        "-inurl:(signup|login|cart|shop|product|download|jobs|careers|forum|community|support)",
        "-filetype:pdf", "-filetype:xml", "-filetype:doc", "-filetype:ppt",
    ]
    # The query string for Google's 'q' parameter
    google_q_param_value = search_keywords + " " + " ".join(excluded_sites_and_paths)
    
    # Parameters for the Google Search URL
    google_url_params = {
        'q': google_q_param_value, # urlencode will handle spaces and special chars
        'hl': 'en',
        'gl': 'us'
    }
    # Construct the final Google Search URL that ScraperAPI will be asked to fetch
    # This is the URL we want ScraperAPI to visit.
    google_search_url_to_scrape = f"{base_google_search_domain}?{urlencode(google_url_params)}"
    
    tool_logs.append(f"DEBUG: Constructed Google Search URL (this is what ScraperAPI should fetch): {google_search_url_to_scrape}")

    scraper_api_endpoint = "http://api.scraperapi.com"
    # Parameters for the request to ScraperAPI itself
    params_for_scraperapi_request = {
        'api_key': scraperapi_key,
        'url': google_search_url_to_scrape, # Pass the fully formed Google URL here
        'render': 'true', 
        'country_code': 'us',
    }
    tool_logs.append(f"Requesting ScraperAPI endpoint '{scraper_api_endpoint}' with PARAMS: {json.dumps(params_for_scraperapi_request)}")

    try:
        response = await http_client.get(scraper_api_endpoint, params=params_for_scraperapi_request, timeout=90.0)
        response.raise_for_status()
        # Log the URL ScraperAPI was asked to fetch for clarity in case of errors
        tool_logs.append(f"ScraperAPI returned status {response.status_code} for its target URL '{google_search_url_to_scrape}'.")
        return response.text
    except httpx.HTTPStatusError as e:
        tool_logs.append(f"ScraperAPI HTTP error {e.response.status_code} when trying to fetch '{google_search_url_to_scrape}'. Response: {e.response.text[:200]}...")
    except httpx.RequestError as e:
        tool_logs.append(f"ScraperAPI Request error when trying to fetch '{google_search_url_to_scrape}': {e}")
    except Exception as e:
        tool_logs.append(f"Unexpected error fetching HTML via ScraperAPI (target was '{google_search_url_to_scrape}'): {e}")
    return None

async def _parse_search_html_with_llm(
    html_content: str,
    search_keywords: str,
    tool_logs: List[str]
) -> HTMLParseResult:
    """Uses an LLM to parse HTML and extract search results."""
    tool_logs.append(f"Attempting to parse HTML (length: {len(html_content)}) with LLM for keywords: '{search_keywords}'.")
    if not HTML_PARSING_LLM_INSTANCE:
        tool_logs.append("HTML Parsing LLM instance not available.")
        return HTMLParseResult(parsing_errors=["HTML Parsing LLM not initialized."])

    MAX_HTML_FOR_PARSING_LLM = 20000 
    if len(html_content) > MAX_HTML_FOR_PARSING_LLM:
        html_content_to_parse = html_content[:MAX_HTML_FOR_PARSING_LLM]
        tool_logs.append(f"HTML content truncated from {len(html_content)} to {len(html_content_to_parse)} chars for LLM parsing.")
    else:
        html_content_to_parse = html_content
    
    try:
        soup = BeautifulSoup(html_content_to_parse, 'html.parser')
        for tag_name in ['script', 'style', 'nav', 'footer', 'aside', 'form']:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        cleaned_html_for_llm = str(soup) 
        if len(cleaned_html_for_llm) > MAX_HTML_FOR_PARSING_LLM: 
            cleaned_html_for_llm = cleaned_html_for_llm[:MAX_HTML_FOR_PARSING_LLM]
        tool_logs.append(f"Cleaned HTML for LLM (length: {len(cleaned_html_for_llm)})")
    except Exception as e_clean:
        tool_logs.append(f"Error during HTML pre-cleaning: {e_clean}. Using raw (truncated) HTML for LLM.")
        cleaned_html_for_llm = html_content_to_parse

    parsing_prompt = (
        "You are an expert HTML parsing assistant. Your task is to analyze the provided HTML content, "
        "which is from a Google search results page, and extract relevant organic search result links and their titles. \n"
        "Focus on main search results and try to ignore ads, 'People also ask' sections, related searches, or navigation links.\n"
        "For each valid search result you find, extract:\n"
        "1. 'title': The main clickable title text of the search result.\n"
        "2. 'url': The fully qualified URL that the title links to.\n"
        "Return your findings as a JSON object matching the 'HTMLParseResult' schema, specifically the 'extracted_results' list. "
        "If you encounter issues or cannot find results, return an empty list for 'extracted_results' and describe issues in 'parsing_errors'.\n"
        f"The original search keywords were: '{search_keywords}'. This might give context to the type of links to prioritize.\n"
        "HTML Content to parse:\n```html\n"
        f"{cleaned_html_for_llm}\n"
        "```"
    )

    try:
        response = await HTML_PARSING_LLM_INSTANCE.chat.completions.create(
            messages=[{"role": "user", "content": parsing_prompt}],
            response_model=HTMLParseResult, 
            max_retries=1 
        )
        tool_logs.append(f"LLM HTML parsing successful. Found {len(response.extracted_results)} potential items.")
        return response
    except Exception as e:
        tool_logs.append(f"LLM HTML parsing failed: {e}")
        return HTMLParseResult(parsing_errors=[f"LLM HTML parsing exception: {e}"])

# --- Pydantic AI Agent Definition ---
BLOG_URL_FINDER_AGENT = None
if LLM_MODEL_INSTANCE:
    try:
        system_prompt_blog_agent = (
            "You are an intelligent Blog URL Finder Agent. Your primary goal is to find relevant blog posts and articles "
            "for market research based on a user's query. \n"
            "1. Analyze the user's query to understand the core topic and intent. The query will be provided by the user. \n"
            "2. Formulate effective search keywords. Append terms like 'market research', 'industry analysis', 'trends', 'overview', 'statistics', 'report' to the core topic. "
            "   For example, if the query is 'app for music', keywords could be 'music app market research' or 'streaming music industry trends report'. \n"
            "3. You MUST use the 'fetch_and_parse_search_results_tool' to get blog URLs using these keywords. Provide ONLY the refined search keywords to the tool. \n"
            "4. The tool will fetch the Google search results page for your keywords and then use another AI to parse that page to extract titles and URLs. \n"
            "5. Compile the results into the 'BlogURLsOutput' schema. Ensure 'search_keywords_used' reflects the keywords you decided to search with. "
            "   If the tool returns errors or no URLs, reflect this in your output. Prioritize quality over quantity. Limit to a maximum of 5 relevant URLs. \n"
            "   Include any logs or errors from the tool in the 'tool_logs' and 'errors' fields respectively."
        )
        BLOG_URL_FINDER_AGENT = Agent(
            model=LLM_MODEL_INSTANCE,
            system_prompt=system_prompt_blog_agent,
            result_type=BlogURLsOutput,
            result_tool_name="final_blog_url_list_formatter",
            result_tool_description="Formats the final list of found blog URLs, keywords used, and any errors or logs.",
            result_retries=1
        )
    except Exception as e:
        logger.error(f"Failed to initialize BlogURLFinderAgent: {e}")
        BLOG_URL_FINDER_AGENT = None
else:
    logger.error("Main LLM_MODEL_INSTANCE not available for BlogURLFinderAgent.")

# --- Main Entry Point ---
async def run_search(user_query: str, scraperapi_key: str) -> List[WebResult]:
    if not BLOG_URL_FINDER_AGENT or not HTML_PARSING_LLM_INSTANCE:
        logger.error("Blog URL Finder Agent or HTML Parsing LLM is not initialized. Cannot proceed.")
        return []

    async with httpx.AsyncClient() as http_client_for_tool: 
        @BLOG_URL_FINDER_AGENT.tool 
        async def fetch_and_parse_search_results_tool(
            ctx: RunContext, 
            search_keywords: str = Field(..., description="The specific keywords formulated by the main agent to search Google for relevant blog posts/articles.")
        ) -> str: 
            tool_logs_accumulator = [f"Tool 'fetch_and_parse_search_results_tool' called with keywords: '{search_keywords}'."]
            tool_errors_accumulator = []
            extracted_web_results: List[WebResult] = []
            html_content = await _fetch_google_search_html(
                search_keywords, scraperapi_key, http_client_for_tool, tool_logs_accumulator
            )
            if html_content:
                html_parse_agent_result = await _parse_search_html_with_llm(
                    html_content, search_keywords, tool_logs_accumulator
                )
                extracted_web_results.extend(html_parse_agent_result.extracted_results)
                tool_errors_accumulator.extend(html_parse_agent_result.parsing_errors)
            else:
                tool_errors_accumulator.append("Failed to fetch HTML content from Google Search via ScraperAPI.")
            tool_output_dict = {
                "search_keywords_provided_to_tool": search_keywords, 
                "found_urls": [wr.model_dump() for wr in extracted_web_results],
                "errors": tool_errors_accumulator,
                "tool_logs": tool_logs_accumulator
            }
            return json.dumps(tool_output_dict)

        try:
            agent_prompt = (
                f"Based on the user's query: '{user_query}', identify the best keywords for finding market research blogs/articles "
                "and then use the 'fetch_and_parse_search_results_tool' with those keywords."
            )
            agent_response_container = await BLOG_URL_FINDER_AGENT.run(agent_prompt)
            if agent_response_container and isinstance(agent_response_container.data, BlogURLsOutput):
                final_output: BlogURLsOutput = agent_response_container.data
                logger.info(f"Blog URL Agent completed. User Query: '{final_output.user_query}', Keywords Used: '{final_output.search_keywords_used}'. Found {len(final_output.found_urls)} URLs.")
                if final_output.errors:
                    for err in final_output.errors: logger.error(f"BlogURLAgent Error: {err}")
                if final_output.tool_logs:
                    for log_msg in final_output.tool_logs: logger.info(f"BlogURLAgent ToolLog: {log_msg}")
                valid_found_urls = []
                for item_data in final_output.found_urls:
                    try:
                        if isinstance(item_data, dict):
                            valid_found_urls.append(WebResult(**item_data))
                        elif isinstance(item_data, WebResult):
                             valid_found_urls.append(item_data)
                    except ValidationError as e_val:
                        logger.warning(f"Skipping an invalid WebResult item from agent output: {item_data}. Error: {e_val}")
                return valid_found_urls[:5] 
            else:
                logger.error(f"Blog URL Agent returned an unexpected response or no data. Response: {agent_response_container}")
                return []
        except ModelRetry as e_retry:
            logger.error(f"Blog URL Agent failed after retries for query '{user_query}': {e_retry}")
            return []
        except Exception as e:
            logger.error(f"Error running Blog URL Agent for query '{user_query}': {e}", exc_info=True)
            return []

    if BLOG_URL_FINDER_AGENT: 
        @BLOG_URL_FINDER_AGENT.result_validator
        async def validate_blog_agent_output(ctx: RunContext, result: BlogURLsOutput) -> BlogURLsOutput:
            logger.info(f"Validating BlogURLFinderAgent output for query: {result.user_query}")
            if not result.search_keywords_used:
                logger.warning("Validator: Agent did not specify search_keywords_used.")
                result.errors.append("Validation: Agent did not specify search_keywords_used.")
            if not result.found_urls and not result.errors:
                logger.info("Validator: Agent found no URLs and reported no explicit errors from tool.")
            valid_urls_after_validation = []
            for item in result.found_urls:
                try:
                    validated_item = WebResult(title=item.title, url=item.url) 
                    if "google.com/search?q=" in validated_item.url.lower() or \
                       "google.com/imgres?imgurl=" in validated_item.url.lower() or \
                       not validated_item.url.startswith(("http://", "https://")): 
                        logger.warning(f"Validator: Filtering out likely invalid/search URL: {validated_item.url}")
                        result.tool_logs.append(f"Validator: Filtered out invalid URL: {validated_item.url}")
                        continue
                    valid_urls_after_validation.append(validated_item)
                except ValidationError as e_val:
                    logger.warning(f"Validator: Invalid WebResult found in agent output: {item}. Error: {e_val}")
                    result.errors.append(f"Validator: Found invalid URL structure: {item.url or 'URL missing'}")
            result.found_urls = valid_urls_after_validation[:5] 
            if len(result.found_urls) == 0 and not result.errors:
                result.tool_logs.append("Validator: No valid URLs remained after validation, and no explicit tool errors were reported.")
            logger.info(f"Validation complete. Returning {len(result.found_urls)} URLs.")
            return result
    else: 
        pass


if __name__ == "__main__":
    example_query_main = "latest trends in music streaming apps"
    logger.info(f"--- Starting Blog URL Agent (AI Parsing) for query: '{example_query_main}' ---")
    
    test_scraperapi_key_main = os.getenv("SCRAPERAPI_KEY")
    if not test_scraperapi_key_main:
        logger.critical("SCRAPERAPI_KEY not found in environment for standalone agent test.")
        sys.exit(1)
    if not OPENROUTER_API_KEY:
        logger.critical("OPENROUTER_API_KEY not found. LLM part of agent will fail.")
        sys.exit(1)
    if not LLM_MODEL_INSTANCE or not HTML_PARSING_LLM_INSTANCE:
        logger.critical("LLM Model instances not created, cannot run agent.")
        sys.exit(1)
        
    final_web_results_list = asyncio.run(run_search(example_query_main, test_scraperapi_key_main))
    
    logger.info(f"--- Blog URL Agent (AI Parsing) run finished for query: '{example_query_main}' ---")
    logger.info(f"Final List[WebResult] (to be passed to craw4ai):")
    
    output_list_for_craw4ai = [wr.model_dump() for wr in final_web_results_list]
    print(json.dumps(output_list_for_craw4ai, indent=2))
