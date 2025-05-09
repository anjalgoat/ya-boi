import os
import sys
import asyncio
import logging
import json
from typing import List, Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
import httpx
import requests # For iTunes API
from urllib.parse import quote_plus, unquote

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

load_dotenv()

class CompetitorInfo(BaseModel): # Renamed from Competitor to avoid Pydantic-AI conflict if any
    name: str = Field(..., description="Name of the competitor")
    app_store_url: Optional[str] = Field(None, description="App Store URL (for apps)")
    google_play_url: Optional[str] = Field(None, description="Google Play URL (for apps)")

class CompetitorAgentResponse(BaseModel): # Renamed from CompetitorResponse
    query: str = Field(..., description="The user's original query")
    competitors: List[CompetitorInfo] = Field(..., description="List of exactly 3 competitors", min_items=3, max_items=3)

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
scraperapi_key_global = os.getenv("SCRAPERAPI_KEY") # For use within this module's functions
openai_model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini") # Default to a known good model
openai_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

if not openrouter_api_key:
    logger.error("OPENROUTER_API_KEY not found. LLM agent will fail.")
if not scraperapi_key_global:
    logger.warning("SCRAPERAPI_KEY not found. Google Play URL scraping will fail.")

_model = None
if openrouter_api_key:
    try:
        _model = OpenAIModel(
            api_key=openrouter_api_key,
            model_name=openai_model_name,
            base_url=openai_base_url
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAIModel: {e}")

system_prompt = (
    "You are an AI agent. Process a user query to identify top competitors. Your task is:\n"
    "1. Analyze if the query is app-related (e.g., 'app for music') or local business-related (e.g., 'restaurant in London').\n"
    "2. Identify exactly 3 top competitors based on the query.\n"
    "3. For apps: Include plausible (but not necessarily validated) Google Play Store and App Store URLs if known or easily inferable. Use common patterns.\n"
    "4. For local businesses: Only provide competitor names.\n"
    "5. Return results using the provided tool, ensuring exactly 3 competitors.\n"
    "Guidelines:\n"
    "- If the query is very unclear, return 3 placeholders like 'Unknown Competitor 1'.\n"
    "- For apps, if you are unsure about specific URLs, you can omit them or use a placeholder format like 'SEARCH_APP_STORE' or 'SEARCH_GOOGLE_PLAY'. They will be validated/fetched later."
)

competitor_finder_agent = None
if _model:
    try:
        competitor_finder_agent = Agent(
            _model,
            system_prompt=system_prompt,
            result_type=CompetitorAgentResponse,
            result_tool_name="competitor_results_logger",
            result_tool_description="Logs the identified competitors and their potential app store URLs based on the query.",
            result_retries=2,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Pydantic AI Agent for competitors: {e}")

def get_app_store_url_itunes_api(competitor_name: str) -> Optional[str]:
    logger.info(f"Fetching App Store URL via iTunes API for: {competitor_name}")
    try:
        search_term_encoded = quote_plus(f"{competitor_name} app")
        itunes_url = f"https://itunes.apple.com/search?term={search_term_encoded}&entity=software&limit=1&country=US"
        response = requests.get(itunes_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            app_info = data["results"][0]
            found_name = app_info.get("trackName", "").lower()
            # Simple name check; could be improved with fuzzy matching
            if competitor_name.lower() in found_name or found_name in competitor_name.lower():
                url = app_info.get("trackViewUrl")
                logger.info(f"App Store URL found via iTunes API: {url} for '{competitor_name}' (found: '{found_name}')")
                return url
            else:
                logger.warning(f"iTunes API found app '{found_name}' but it doesn't closely match query '{competitor_name}'")
        else:
            logger.warning(f"No results in iTunes API search for '{competitor_name}'")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching App Store URL via iTunes API for '{competitor_name}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error during iTunes API call for '{competitor_name}': {e}", exc_info=True)
    return None

async def get_google_play_url_scraperapi(competitor_name: str, http_client: httpx.AsyncClient) -> Optional[str]:
    logger.info(f"Fetching Google Play URL via ScraperAPI for: {competitor_name}")
    if not scraperapi_key_global:
        logger.error("SCRAPERAPI_KEY missing for Google Play search in competitor.py.")
        return None

    search_term_encoded = quote_plus(f"{competitor_name} app review") # Adding "review" might help target app page
    target_search_url = f"https://play.google.com/store/search?q={search_term_encoded}&c=apps&hl=en&gl=US"
    
    scraper_api_endpoint = "http://api.scraperapi.com"
    params = {
        'api_key': scraperapi_key_global,
        'url': target_search_url,
        'render': 'false', # JS might not be strictly needed if links are in initial HTML
        'country_code': 'us'
    }

    try:
        response = await http_client.get(scraper_api_endpoint, params=params, timeout=30.0)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # --- HTML SELECTOR for Google Play (CRITICAL: Verify & Maintain) ---
        # This looks for an 'a' tag whose href starts with '/store/apps/details?id='
        # Common parent selectors for app cards: div.ULeU3b, div.Vpfmgd, div.wXUyZd
        app_link_tag = soup.select_one('div.Vpfmgd a[href^="/store/apps/details?id="], div.wXUyZd a[href^="/store/apps/details?id="]')
        if not app_link_tag: # Fallback to a broader search within common item containers
            app_link_tag = soup.select_one('div[role="listitem"] a[href^="/store/apps/details?id="]')
        # --- END HTML SELECTOR ---

        if app_link_tag:
            href = app_link_tag.get('href')
            if href and href.startswith('/store/apps/details?id='):
                full_url = f"https://play.google.com{unquote(href)}" # Unquote the URL
                # Optional: More robust title matching from elements near the link
                # For now, we assume the first link is the most relevant if the search query is good.
                logger.info(f"Google Play URL found via ScraperAPI: {full_url} for '{competitor_name}'")
                return full_url
            else:
                logger.warning(f"Found link tag for '{competitor_name}', but href '{href}' is not a valid Play Store app details link.")
        else:
            logger.warning(f"No app link found in Google Play search results for '{competitor_name}' via ScraperAPI.")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching Google Play URL via ScraperAPI for '{competitor_name}': {e.response.status_code}")
        if e.response : logger.debug(f"ScraperAPI Error Response: {e.response.text[:200]}...")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching Google Play URL via ScraperAPI for '{competitor_name}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching Google Play URL for '{competitor_name}': {e}", exc_info=True)
    return None

@competitor_finder_agent.result_validator
async def validate_competitor_result(ctx: RunContext[None], result: CompetitorAgentResponse) -> CompetitorAgentResponse:
    logger.info(f"Validating LLM competitor result for query: {result.query}")
    query = result.query.lower()
    llm_competitors = result.competitors
    
    is_app_query = "app" in query or any(c.app_store_url or c.google_play_url for c in llm_competitors if c) # Check if c is not None
    logger.info(f"Query identified as app-related: {is_app_query}")

    validated_competitors = []
    # Filter out any None competitors that might have come from LLM
    current_competitors = [c for c in llm_competitors if c and c.name and 'unknown' not in c.name.lower()]

    # Placeholder filling logic (simplified)
    if len(current_competitors) < 3:
        logger.warning(f"LLM returned {len(current_competitors)} valid competitors, expected 3. Filling placeholders.")
        placeholders_needed = 3 - len(current_competitors)
        for i in range(placeholders_needed):
            current_competitors.append(CompetitorInfo(name=f"Placeholder Competitor {i + 1}"))
    
    final_competitors_for_urls = current_competitors[:3] # Ensure we only process 3

    url_fetch_tasks = []
    if is_app_query:
        logger.info("Fetching/Validating App Store and Google Play URLs for app competitors...")
        async with httpx.AsyncClient() as client: # Use a single client for all ScraperAPI calls here
            for comp in final_competitors_for_urls:
                # App Store URL (uses sync requests.get, run in thread)
                if not comp.app_store_url or "apple.com" not in comp.app_store_url.lower():
                    url_fetch_tasks.append(asyncio.to_thread(get_app_store_url_itunes_api, comp.name))
                else:
                    url_fetch_tasks.append(asyncio.sleep(0, result=comp.app_store_url)) # Keep valid looking URL

                # Google Play URL (uses async ScraperAPI)
                if not comp.google_play_url or "play.google.com" not in comp.google_play_url.lower():
                    url_fetch_tasks.append(get_google_play_url_scraperapi(comp.name, client))
                else:
                    url_fetch_tasks.append(asyncio.sleep(0, result=comp.google_play_url))
            
            fetched_urls_results = await asyncio.gather(*url_fetch_tasks)

            for i, comp_info in enumerate(final_competitors_for_urls):
                app_store_url = fetched_urls_results[i * 2]
                google_play_url = fetched_urls_results[i * 2 + 1]
                validated_competitors.append(CompetitorInfo(
                    name=comp_info.name,
                    app_store_url=app_store_url if isinstance(app_store_url, str) else None,
                    google_play_url=google_play_url if isinstance(google_play_url, str) else None
                ))
    else: # Non-app query
        for comp_info in final_competitors_for_urls:
            validated_competitors.append(CompetitorInfo(name=comp_info.name, app_store_url=None, google_play_url=None))

    final_response = CompetitorAgentResponse(query=result.query, competitors=validated_competitors)
    logger.info(f"Competitor validation complete. Final data: {final_response.model_dump(exclude_none=True)}")
    return final_response

async def run_agent(query: str) -> CompetitorAgentResponse:
    logger.info(f"Running competitor agent with query: {query}")
    error_competitors = [CompetitorInfo(name=f"Error Comp {i+1}") for i in range(3)]
    
    if not competitor_finder_agent:
        logger.error("Competitor finder agent not initialized (likely API key or model issue).")
        return CompetitorAgentResponse(query=query, competitors=error_competitors)
        
    try:
        response = await competitor_finder_agent.run(query)
        if hasattr(response, 'data') and isinstance(response.data, CompetitorAgentResponse):
            result = response.data
        elif isinstance(response, CompetitorAgentResponse):
            result = response
        else:
            logger.error(f"Competitor agent returned unexpected response type: {type(response)}")
            return CompetitorAgentResponse(query=query, competitors=error_competitors)
        
        # Ensure there are always 3 competitors, even if placeholders
        if len(result.competitors) < 3:
            logger.warning(f"LLM returned {len(result.competitors)}, padding to 3.")
            while len(result.competitors) < 3:
                result.competitors.append(CompetitorInfo(name=f"LLM Placeholder {len(result.competitors)+1}"))
        result.competitors = result.competitors[:3]

        logger.info(f"Competitor agent LLM run completed for query: {query}")
        return result
    except ModelRetry as e:
        logger.error(f"Competitor agent failed after retries for query '{query}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error during competitor agent LLM run for query '{query}': {e}", exc_info=True)
    
    # Fallback error response
    return CompetitorAgentResponse(query=query, competitors=error_competitors)

if __name__ == "__main__":
    example_query = "music streaming app"
    logger.info(f"--- Running Competitor Agent (ScraperAPI) for query: '{example_query}' ---")

    if not openrouter_api_key:
        logger.critical("OPENROUTER_API_KEY is missing. Cannot run standalone LLM test.")
        sys.exit(1)
    if not scraperapi_key_global:
        logger.warning("SCRAPERAPI_KEY is missing. Google Play URL fetching will fail in standalone test.")

    result_data = asyncio.run(run_agent(example_query))
    logger.info(f"--- Competitor Agent run finished for: '{example_query}' ---")

    if result_data:
        print(result_data.model_dump_json(indent=2, exclude_none=True))
    else: # Should be handled by run_agent returning a default error response
        logger.error(f"Agent returned None for query: {example_query}, which is unexpected.")
        print(json.dumps({"query": example_query, "competitors": [], "error_message": "Agent returned None"}, indent=2))