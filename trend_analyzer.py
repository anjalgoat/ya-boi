import os
import sys
import asyncio
import logging
import json
from typing import List, Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent, ModelRetry, RunContext, UserError
from pydantic_ai.models.openai import OpenAIModel
import httpx
from urllib.parse import quote_plus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

load_dotenv()

class RelatedQuery(BaseModel):
    query: str = Field(..., description="The related search query text")

class GoogleTrendsResult(BaseModel):
    keyword: str = Field(..., description="The keyword searched on Google Trends")
    related_queries_top: List[RelatedQuery] = Field(default_factory=list)
    related_queries_rising: List[RelatedQuery] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

llm_api_key = os.getenv("OPENROUTER_API_KEY")
scraperapi_key_global = os.getenv("SCRAPERAPI_KEY") # Use this for functions in this file
llm_model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
openai_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

keys_available = True
if not llm_api_key:
    logger.error("OPENROUTER_API_KEY not found. LLM agent for Google Trends will fail.")
    keys_available = False
if not scraperapi_key_global:
    logger.error("SCRAPERAPI_KEY not found. Google Trends scraping will fail.")
    keys_available = False # Scraping itself will fail

async def scrape_google_trends_with_scraperapi(
    keyword: str,
    country: str = "US",
    http_client: httpx.AsyncClient = None
) -> GoogleTrendsResult:
    logger.info(f"Scraping Google Trends via ScraperAPI for: '{keyword}' in '{country}'")

    if not scraperapi_key_global:
        return GoogleTrendsResult(keyword=keyword, errors=["SCRAPERAPI_KEY not configured for scraping."])

    encoded_keyword = quote_plus(keyword)
    target_trends_url = f"https://trends.google.com/trends/explore?q={encoded_keyword}&geo={country}&hl=en"
    
    scraper_api_endpoint = "http://api.scraperapi.com"
    params = {
        'api_key': scraperapi_key_global,
        'url': target_trends_url,
        'render': 'true',       # JS Rendering is essential for Google Trends
        'country_code': country,
        'premium': 'true',      # Often needed for Google services
    }
    
    result_data = GoogleTrendsResult(keyword=keyword)
    client_to_use = http_client if http_client else httpx.AsyncClient()

    try:
        logger.info(f"Requesting Google Trends URL via ScraperAPI: {target_trends_url}")
        response = await client_to_use.get(scraper_api_endpoint, params=params, timeout=90.0)
        response.raise_for_status()

        logger.info(f"ScraperAPI Google Trends response status: {response.status_code}")
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
        
        # --- HTML PARSING LOGIC (HIGHLY FRAGILE - VERIFY SELECTORS) ---
        processed_any_widget = False
        # Updated selectors attempt to be more robust.
        # Google Trends often wraps related queries in sections identifiable by attributes.
        widget_containers = soup.select("div.widget.concepts-widget, div.details-widgets-container, related-searches, div[feed-name='RELATED_QUERIES']")

        if not widget_containers:
             logger.warning(f"No potential related query widget containers found for '{keyword}'. HTML structure might have changed.")
             result_data.errors.append("No related query widget containers found (check selectors/HTML).")
        
        for container_idx, widget_container in enumerate(widget_containers):
            # Check for explicit "Related queries" title if possible
            title_el = widget_container.select_one(".widget-header-title, .title, header .title")
            is_related_widget = title_el and "related queries" in title_el.get_text(strip=True).lower()
            
            # Check for Top queries sections
            # Look for elements that might indicate "Top" or "Rising" data lists
            top_sections = widget_container.select("div[ng-repeat*='Dataবহুল TOP'], div[aria-label*='Top related'], list-item[aria-label*='Top related topics']")
            for top_section in top_sections:
                processed_any_widget = True
                items = top_section.select(".item a, .label-text, list-item span[aria-hidden='true']") # Common item text selectors
                for item in items:
                    query_text = item.get_text(strip=True)
                    if query_text and not any(rq.query == query_text for rq in result_data.related_queries_top):
                        result_data.related_queries_top.append(RelatedQuery(query=query_text))
            
            # Check for Rising queries sections
            rising_sections = widget_container.select("div[ng-repeat*='Dataবহুল RISING'], div[aria-label*='Rising related'], list-item[aria-label*='Rising related topics']")
            for rising_section in rising_sections:
                processed_any_widget = True
                items = rising_section.select(".item a, .label-text, list-item span[aria-hidden='true']")
                for item in items:
                    query_text = item.get_text(strip=True)
                    if query_text and not any(rq.query == query_text for rq in result_data.related_queries_rising):
                        result_data.related_queries_rising.append(RelatedQuery(query=query_text))
        
        if not processed_any_widget and not result_data.errors:
            logger.warning(f"No specific Top/Rising query sections processed for '{keyword}'. The page might not contain them or selectors need update.")
            result_data.errors.append("No Top/Rising query sections processed (check selectors/HTML).")

        if not result_data.related_queries_top and not result_data.related_queries_rising and not result_data.errors:
            no_data_msg = soup.select_one(".widget-error-title, .feed-item.no-data, .auxilaryMessage")
            if no_data_msg:
                msg = no_data_msg.get_text(strip=True)
                logger.warning(f"Google Trends reported no data for '{keyword}': {msg}")
                result_data.errors.append(f"Google Trends: {msg}")
            elif not processed_any_widget: # Only add this if we didn't find any widgets at all
                 result_data.errors.append("No related queries found (page might not have data or selectors failed).")
        # --- END HTML PARSING ---

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error from ScraperAPI for Google Trends '{keyword}': Status {e.response.status_code}"
        logger.error(error_msg)
        if e.response: logger.debug(f"ScraperAPI Error Response: {e.response.text[:200]}...")
        result_data.errors.append(error_msg)
    except httpx.RequestError as e:
        error_msg = f"Network error with ScraperAPI for Google Trends '{keyword}': {e}"
        logger.error(error_msg)
        result_data.errors.append(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error during Google Trends scraping for '{keyword}': {e}"
        logger.error(error_msg, exc_info=True)
        result_data.errors.append(error_msg)
    finally:
        if not http_client and client_to_use:
            await client_to_use.aclose()

    logger.info(f"Google Trends scraping via ScraperAPI finished for '{keyword}'. Top: {len(result_data.related_queries_top)}, Rising: {len(result_data.related_queries_rising)}. Errors: {len(result_data.errors)}")
    return result_data

google_trends_agent = None
agent_init_error_msg = None

if llm_api_key: # LLM Agent setup
    try:
        _llm_model_instance = OpenAIModel(api_key=llm_api_key, model_name=llm_model_name, base_url=openai_base_url)
        system_prompt_trends = (
            "You are an agent processing Google Trends data. A tool will provide scraped data as a JSON string. "
            "Your task is to parse this JSON string and ensure the final output strictly matches the GoogleTrendsResult structure. "
            "The JSON string from the tool IS the GoogleTrendsResult. Simply parse it and return it. "
            "If the tool reports errors in its JSON output, include them in your final GoogleTrendsResult's 'errors' field."
        )
        google_trends_agent = Agent(
            model=_llm_model_instance,
            system_prompt=system_prompt_trends,
            result_type=GoogleTrendsResult,
            result_tool_name="google_trends_data_parser",
            result_tool_description="Parses the JSON string output from the Google Trends scraper tool (ScraperAPI).",
            result_retries=1 # Parsing should be straightforward
        )

        @google_trends_agent.tool
        async def get_scraped_google_trends_data_tool(ctx: RunContext[None], keyword: str, country_code: str = 'US') -> str:
            logger.info(f"Tool: Scraping Google Trends for keyword: '{keyword}', country: '{country_code}' via ScraperAPI")
            # This tool now directly calls the async scraping function
            try:
                async with httpx.AsyncClient() as client:
                    result_obj = await scrape_google_trends_with_scraperapi(keyword=keyword, country=country_code, http_client=client)
                # Return the Pydantic model as a JSON string
                return result_obj.model_dump_json(exclude_none=True)
            except Exception as e:
                logger.error(f"Error in get_scraped_google_trends_data_tool: {e}", exc_info=True)
                error_res = GoogleTrendsResult(keyword=keyword, errors=[f"Tool's internal scraping call failed: {e}"])
                return error_res.model_dump_json(exclude_none=True)

    except UserError as e: # Pydantic-AI specific error
        agent_init_error_msg = f"Pydantic-AI UserError during Trend Agent setup: {e}"
        logger.error(agent_init_error_msg, exc_info=True)
    except Exception as e:
        agent_init_error_msg = f"Unexpected error during Trend Agent setup: {e}"
        logger.error(agent_init_error_msg, exc_info=True)
else:
    agent_init_error_msg = "Trend Agent not initialized: OPENROUTER_API_KEY missing."
    logger.error(agent_init_error_msg)

async def run_trends_agent(keyword: str, country: str = "US") -> GoogleTrendsResult:
    if not google_trends_agent:
        error_msg = f"Google Trends Agent not initialized. Last error: {agent_init_error_msg}"
        logger.error(error_msg)
        return GoogleTrendsResult(keyword=keyword, errors=[error_msg])
    if not keys_available: # If scraping key was also missing
         return GoogleTrendsResult(keyword=keyword, errors=["Cannot run: SCRAPERAPI_KEY is missing."])


    logger.info(f"Running Google Trends LLM agent for keyword: '{keyword}', country: '{country}'")
    try:
        # The prompt instructs the LLM to use the tool and parse its JSON string output.
        agent_prompt = (
            f"Use the tool to get Google Trends data for the keyword '{keyword}' in country '{country}'. "
            f"The tool will return a JSON string. Parse this JSON string and structure your final response "
            f"as a GoogleTrendsResult object. Ensure all fields from the tool's JSON output are correctly mapped."
        )
        response = await google_trends_agent.run(agent_prompt)

        if hasattr(response, 'data') and isinstance(response.data, GoogleTrendsResult):
            result = response.data
        elif isinstance(response, GoogleTrendsResult): # Direct return
            result = response
        elif isinstance(response, str): # LLM might just return the string if tool is simple
            logger.warning("Trend agent returned a string; attempting to parse as GoogleTrendsResult.")
            try:
                result = GoogleTrendsResult.model_validate_json(response)
            except Exception as e_parse:
                err = f"Failed to parse string response from Trend agent: {e_parse}. Response: {response[:300]}"
                logger.error(err)
                return GoogleTrendsResult(keyword=keyword, errors=[err])
        else:
            err = f"Trend agent returned unexpected response type: {type(response)}"
            logger.error(err)
            return GoogleTrendsResult(keyword=keyword, errors=[err])
        
        logger.info(f"Trend LLM agent run completed for '{keyword}'.")
        return result
        
    except ModelRetry as e_retry:
        err = f"Trend agent failed after retries for '{keyword}': {e_retry}"
        logger.error(err)
        return GoogleTrendsResult(keyword=keyword, errors=[err])
    except Exception as e_run:
        err = f"Unexpected error during Trend agent run for '{keyword}': {e_run}"
        logger.error(err, exc_info=True)
        return GoogleTrendsResult(keyword=keyword, errors=[err])

if __name__ == "__main__":
    example_keyword = "generative ai applications"
    example_country = "US"
    logger.info(f"--- Running Trend Analyzer (ScraperAPI) for: '{example_keyword}', Country: '{example_country}' ---")

    if not llm_api_key or not scraperapi_key_global:
        logger.critical("Cannot run standalone: OPENROUTER_API_KEY or SCRAPERAPI_KEY is missing from .env")
        sys.exit(1)
    if not google_trends_agent:
        logger.critical(f"Cannot run standalone: Google Trends Agent not initialized. Error: {agent_init_error_msg}")
        sys.exit(1)

    final_result = asyncio.run(run_trends_agent(example_keyword, example_country))
    logger.info(f"--- Trend Analyzer finished for: '{example_keyword}' ---")

    if final_result:
        print(final_result.model_dump_json(indent=2, exclude_none=True))
    else:
        # run_trends_agent should always return a GoogleTrendsResult, even if with errors
        print(json.dumps({"keyword": example_keyword, "errors": ["Agent run unexpectedly returned None."]}, indent=2))