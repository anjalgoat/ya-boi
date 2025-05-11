import logging
import os
import sys
import asyncio
import json
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, HttpUrl, field_validator
from pydantic_ai import Agent, RunContext, ModelRetry # Pydantic-AI for agent framework
from pydantic_ai.models.openai import OpenAIModel # OpenAI model via Pydantic-AI
import httpx # Async HTTP client
from urllib.parse import quote_plus, urlencode # For URL encoding

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
# BRAVE_SEARCH_API_KEY will be passed into run_search function by the workflow.

# --- Pydantic Models ---
class WebResult(BaseModel):
    """Represents a single found web result (URL and title)."""
    title: str = Field(..., description="The accurately extracted title of the search result.")
    url: str = Field(..., description="The fully qualified URL of the search result.")

    @field_validator('url')
    def validate_url_format(cls, value):
        """Ensures the URL starts with http:// or https://."""
        if not value.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return value

# Defines the final structured output for the main Blog URL Finder Agent.
class BlogURLsOutput(BaseModel):
    """Defines the structured output for the main Blog URL Finder Agent."""
    user_query: str = Field(description="The original user query.")
    search_keywords_used: str = Field(description="The effective search keywords used by the agent.")
    found_urls: List[WebResult] = Field(default_factory=list, description="List of relevant blog URLs and titles found, max 5.")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered by the main agent or its tools.")
    tool_logs: List[str] = Field(default_factory=list, description="Log messages from the search and parsing tool.")

# --- Global LLM Model Instance (Initialize once) ---
LLM_MODEL_INSTANCE = None

if OPENROUTER_API_KEY:
    try:
        LLM_MODEL_INSTANCE = OpenAIModel(
            api_key=OPENROUTER_API_KEY,
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        )
        logger.info("LLM instance for Blog URL agent initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAIModel instance for Blog URL Agent: {e}")
else:
    logger.error("OPENROUTER_API_KEY not found. Blog URL Agent's LLM capabilities will be disabled.")


# --- Tool Implementation: Fetch Search Results with Brave Search API ---

async def _fetch_brave_search_results(
    search_keywords: str,
    brave_api_key: str,
    http_client: httpx.AsyncClient,
    tool_logs: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Fetches search results from Brave Search API.
    
    Args:
        search_keywords: Keywords to search on Brave.
        brave_api_key: The API key for Brave Search API.
        http_client: An active httpx.AsyncClient instance.
        tool_logs: A list to append log messages to.

    Returns:
        A dictionary containing the JSON response from Brave Search API if successful, None otherwise.
    """
    tool_logs.append(f"Attempting to fetch Brave Search results for keywords: '{search_keywords}'.")
    if not brave_api_key:
        tool_logs.append("Error: Brave Search API key is missing in _fetch_brave_search_results.")
        logger.error("Brave Search API key missing. Cannot fetch search results.")
        return None

    brave_search_api_endpoint = "https://api.search.brave.com/res/v1/web/search"
    
    # Parameters for the Brave Search API request
    params = {
        'q': search_keywords,
        'country': 'us', # Specify country, e.g., 'us'
        'search_lang': 'en', # Specify search language, e.g., 'en'
        'count': 20, # Number of results to return
        'safesearch': 'moderate',
        # 'result_filter': 'web', # To prefer web results, though Brave API primarily returns web.
        # 'extra_snippets': 'true' # Potentially useful for relevance assessment later
    }
    
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': brave_api_key
    }

    tool_logs.append(f"Requesting Brave Search API endpoint '{brave_search_api_endpoint}' with query: '{search_keywords}'")

    try:
        response = await http_client.get(brave_search_api_endpoint, params=params, headers=headers, timeout=30.0)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        json_response = response.json()
        tool_logs.append(f"Brave Search API successfully returned JSON response. Status: {response.status_code}.")
        return json_response
    except httpx.HTTPStatusError as e:
        error_message = f"Brave Search API HTTP error {e.response.status_code} for keywords '{search_keywords}'."
        response_text_snippet = e.response.text[:500] if e.response else "No response body."
        tool_logs.append(f"{error_message} Response snippet: {response_text_snippet}...")
        logger.error(f"{error_message} Response: {response_text_snippet}")
    except httpx.RequestError as e:
        error_message = f"Brave Search API Request error (e.g., network issue) for keywords '{search_keywords}': {e}"
        tool_logs.append(error_message)
        logger.error(error_message)
    except json.JSONDecodeError as e:
        error_message = f"Failed to decode JSON response from Brave Search API for keywords '{search_keywords}': {e}. Response text: {response.text[:500]}"
        tool_logs.append(error_message)
        logger.error(error_message)
    except Exception as e:
        error_message = f"Unexpected error during Brave Search API call for keywords '{search_keywords}': {type(e).__name__} - {e}"
        tool_logs.append(error_message)
        logger.error(error_message, exc_info=True)
    return None

# --- Pydantic AI Agent Definition ---
BLOG_URL_FINDER_AGENT = None
if LLM_MODEL_INSTANCE:
    try:
        system_prompt_blog_agent = (
            "You are an intelligent Blog URL Finder Agent. Your primary goal is to find relevant blog posts and articles "
            "for market research based on a user's query. \n"
            "1. Analyze the user's query to understand the core topic and intent. \n"
            "2. Formulate effective search keywords. Append terms like 'market research', 'industry analysis', 'trends', 'overview', 'statistics', 'report', 'insights', 'blog', 'article' to the core topic. "
            "   For example, if the query is 'app for music', keywords could be 'music app market research blog' or 'streaming music industry trends report article'. \n"
            "3. You MUST use the 'fetch_search_results_via_brave_tool' to get blog URLs using these keywords. Provide ONLY the refined search keywords to the tool. \n"
            "4. The tool will call the Brave Search API and directly return structured search results (titles and URLs). \n"
            "5. Compile the results into the 'BlogURLsOutput' schema. Ensure 'search_keywords_used' reflects the keywords you decided to search with. "
            "   If the tool returns errors or no URLs, reflect this in your output. Prioritize quality over quantity. Limit to a maximum of 5 relevant URLs. \n"
            "   Filter the results from the tool to include only those that appear to be actual blog posts or articles, avoiding product pages, main company sites unless they are clearly blog sections, or forums. \n"
            "   Include any logs or errors from the tool in the 'tool_logs' and 'errors' fields respectively of the BlogURLsOutput."
        )
        BLOG_URL_FINDER_AGENT = Agent(
            model=LLM_MODEL_INSTANCE,
            system_prompt=system_prompt_blog_agent,
            result_type=BlogURLsOutput,
            result_tool_name="final_blog_url_list_formatter",
            result_tool_description="Formats the final list of found blog URLs, keywords used, and any errors or logs from the process.",
            result_retries=1
        )
    except Exception as e:
        logger.error(f"Failed to initialize BlogURLFinderAgent: {e}")
        BLOG_URL_FINDER_AGENT = None
else:
    logger.error("Main LLM_MODEL_INSTANCE not available for BlogURLFinderAgent. Agent not created.")

# --- Main Entry Point for the script/module ---
async def run_search(user_query: str, brave_search_api_key: str) -> List[WebResult]:
    """
    Main function to run the blog URL search process using Brave Search API.
    """
    if not BLOG_URL_FINDER_AGENT:
        logger.error("Blog URL Finder Agent is not initialized. Cannot proceed.")
        return []
    if not brave_search_api_key:
        logger.error("Brave Search API key not provided to run_search. Cannot fetch URLs.")
        return []

    async with httpx.AsyncClient() as http_client_for_tool:
        @BLOG_URL_FINDER_AGENT.tool
        async def fetch_search_results_via_brave_tool(
            ctx: RunContext,
            search_keywords: str = Field(..., description="The specific keywords formulated by the main agent to search for relevant blog posts/articles using Brave Search API.")
        ) -> str: # Tool returns a JSON string for the agent to process
            """
            Tool to fetch search results using Brave Search API and extract blog URLs and titles.
            """
            tool_logs_accumulator = [f"Tool 'fetch_search_results_via_brave_tool' called with keywords: '{search_keywords}'."]
            tool_errors_accumulator = []
            extracted_web_results: List[WebResult] = []

            brave_response_json = await _fetch_brave_search_results(
                search_keywords, brave_search_api_key, http_client_for_tool, tool_logs_accumulator
            )

            if brave_response_json:
                # Brave Search API returns results in `web.results` or `mixed.main.results` etc.
                # We'll primarily check `web.results`.
                results_list = brave_response_json.get("web", {}).get("results", [])
                
                # Also check 'mixed' results if 'web.results' is empty or insufficient
                if not results_list and "mixed" in brave_response_json:
                    mixed_main_results = brave_response_json.get("mixed", {}).get("main", [])
                    for item_group in mixed_main_results: # mixed.main can be a list of groups
                        if item_group.get("type") == "web_search_results" and "results" in item_group:
                             results_list.extend(item_group.get("results",[]))
                    tool_logs_accumulator.append(f"Checked 'mixed.main' results, found {len(results_list)} potential items there.")


                tool_logs_accumulator.append(f"Brave API returned {len(results_list)} raw results for keywords: '{search_keywords}'.")

                for item in results_list:
                    title = item.get("title")
                    url = item.get("url")
                    # Basic filtering: ensure title and URL are present
                    if title and url:
                        try:
                            # Validate with Pydantic model
                            web_res = WebResult(title=title, url=url)
                            # Additional filtering for common non-blog patterns
                            if not any(domain_part in url.lower() for domain_part in [
                                "google.com", "youtube.com", "amazon.com", "wikipedia.org", 
                                "facebook.com", "twitter.com", "linkedin.com",
                                ".pdf", ".xml", ".doc", ".ppt" # File extensions
                            ]) and ("blog" in url.lower() or "article" in url.lower() or "news" in url.lower() or "insights" in url.lower() or "/" in url.split("://",1)[-1]): # Heuristic for content pages
                                extracted_web_results.append(web_res)
                            else:
                                tool_logs_accumulator.append(f"Tool: Filtered out potential non-blog/article URL: {url}")
                        except ValidationError as ve:
                            tool_logs_accumulator.append(f"Tool: Invalid WebResult from Brave API - Title: '{title}', URL: '{url}'. Error: {ve}")
                    else:
                        tool_logs_accumulator.append(f"Tool: Skipping Brave API result with missing title or URL: {item}")
                tool_logs_accumulator.append(f"Tool: Extracted {len(extracted_web_results)} WebResult items after initial filtering.")
            else:
                tool_errors_accumulator.append(f"Failed to fetch or parse search results from Brave API for keywords: '{search_keywords}'.")
                logger.warning(f"Tool: No results from Brave API for keywords '{search_keywords}'.")
            
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
                "and then use the 'fetch_search_results_via_brave_tool' with those keywords. "
                "Critically evaluate the URLs returned by the tool. Select up to 5 URLs that are most likely to be actual blog posts, articles, or detailed reports. "
                "Avoid generic homepages, product listings, or forum discussions unless the title clearly indicates a relevant article. "
                "Ensure your final output (BlogURLsOutput) correctly includes the 'user_query', 'search_keywords_used', "
                "the filtered 'found_urls' (max 5), and any 'errors' or 'tool_logs' reported by the tool."
            )
            agent_response_container = await BLOG_URL_FINDER_AGENT.run(agent_prompt)

            if agent_response_container and isinstance(agent_response_container.data, BlogURLsOutput):
                final_output: BlogURLsOutput = agent_response_container.data
                logger.info(f"Blog URL Agent completed. User Query: '{final_output.user_query}', Keywords Used: '{final_output.search_keywords_used}'. Found {len(final_output.found_urls)} URLs from agent.")
                
                if final_output.errors:
                    for err in final_output.errors: logger.error(f"BlogURLAgent Final Output Error: {err}")
                if final_output.tool_logs:
                    for log_msg in final_output.tool_logs: logger.info(f"BlogURLAgent Final Output ToolLog: {log_msg}")
                
                # The agent itself is now responsible for the final filtering to 5 URLs.
                # The validator can do a final sanity check.
                validated_urls = []
                for item_data in final_output.found_urls:
                    try:
                        if isinstance(item_data, dict):
                            web_res = WebResult(**item_data)
                        elif isinstance(item_data, WebResult):
                            web_res = item_data
                        else: continue # Skip if not a valid type
                        validated_urls.append(web_res)
                    except ValidationError:
                        logger.warning(f"Skipping invalid WebResult in agent's final list: {item_data}")

                logger.info(f"Returning {len(validated_urls[:5])} URLs after agent processing.")
                return validated_urls[:5]
            else:
                logger.error(f"Blog URL Agent returned an unexpected response or no data. Response type: {type(agent_response_container.data if hasattr(agent_response_container, 'data') else agent_response_container)}")
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
        logger.info(f"Validator: Validating BlogURLFinderAgent output for query: {result.user_query}")
        
        if not result.search_keywords_used:
            logger.warning("Validator: Agent did not specify 'search_keywords_used'.")
            result.errors.append("Validation Error: Agent did not specify 'search_keywords_used'.")

        validated_urls_after_final_check = []
        for item in result.found_urls:
            try:
                if isinstance(item, dict): validated_item = WebResult(**item)
                elif isinstance(item, WebResult): validated_item = item
                else: continue

                if any(domain in validated_item.url.lower() for domain in [
                    "google.com/search?q=", "google.com/imgres?imgurl=",
                    "microsofttranslator.com", "translate.google.com"
                ]) or not validated_item.url.startswith(("http://", "https://")):
                    logger.warning(f"Validator: Filtering out likely invalid/search/translation URL: {validated_item.url}")
                    result.tool_logs.append(f"Validator: Filtered out invalid URL: {validated_item.url}")
                    continue
                validated_urls_after_final_check.append(validated_item)
            except ValidationError as e_val:
                logger.warning(f"Validator: Invalid WebResult in agent output: {item}. Error: {e_val}")
                result.errors.append(f"Validator Error: Invalid URL structure: {getattr(item, 'url', 'URL missing')}")
        
        result.found_urls = validated_urls_after_final_check[:5]
        
        if len(result.found_urls) == 0 and not result.errors:
            result.tool_logs.append("Validator: No valid URLs remained after final validation, and no tool errors reported.")
        
        logger.info(f"Validator: Complete. Final BlogURLsOutput has {len(result.found_urls)} URLs.")
        return result
else:
    logger.warning("BLOG_URL_FINDER_AGENT is None, result_validator not attached.")

# --- Example Usage (for standalone testing) ---
if __name__ == "__main__":
    example_query_main = "market research on AI powered code generation tools"
    logger.info(f"--- Starting Blog URL Agent (Brave Search API) for query: '{example_query_main}' ---")
    
    test_brave_api_key_main = os.getenv("BRAVE_SEARCH_API_KEY")
    if not test_brave_api_key_main:
        logger.critical("FATAL: BRAVE_SEARCH_API_KEY not found in environment. Cannot run standalone agent test.")
        sys.exit(1)
    if not OPENROUTER_API_KEY:
        logger.critical("FATAL: OPENROUTER_API_KEY not found. LLM part of agent will fail.")
        sys.exit(1)
    if not LLM_MODEL_INSTANCE:
        logger.critical("FATAL: LLM Model instance not created, cannot run agent.")
        sys.exit(1)
        
    final_web_results_list = asyncio.run(run_search(example_query_main, test_brave_api_key_main))
    
    logger.info(f"--- Blog URL Agent (Brave Search API) run finished for query: '{example_query_main}' ---")
    logger.info(f"Final List[WebResult] (to be passed to craw4ai.py by workflow.py):")
    
    output_list_for_craw4ai = [wr.model_dump() for wr in final_web_results_list]
    print(json.dumps(output_list_for_craw4ai, indent=2))
