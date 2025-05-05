import os
import asyncio
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from typing import List, Optional
import requests
from scrapfly import ScrapflyClient, ScrapeConfig
from bs4 import BeautifulSoup
import json # Added for potential JSON operations, though model_dump_json handles output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define Pydantic models
class Competitor(BaseModel):
    name: str = Field(..., description="Name of the competitor")
    app_store_url: Optional[str] = Field(None, description="App Store URL (for apps)")
    google_play_url: Optional[str] = Field(None, description="Google Play URL (for apps)")

class CompetitorResponse(BaseModel):
    query: str = Field(..., description="The user's original query")
    competitors: List[Competitor] = Field(..., description="List of exactly 3 competitors", min_items=3, max_items=3)

# Environment variable checks
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
scrapfly_api_key = os.getenv("SCRAPFLY_API_KEY")
openai_model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo") # Provide a default model
openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") # Provide a default base URL if needed

if not openrouter_api_key:
    # Assuming OPENROUTER_API_KEY might be intended for OpenAI key via OpenRouter
    logger.warning("OPENROUTER_API_KEY not found. Ensure OpenAI compatible API key is set if not using OpenRouter.")
    # You might want to use a different variable like OPENAI_API_KEY depending on your setup
    # For now, we'll proceed assuming _model might use a default or other env var internally if needed
    pass # Or raise ValueError("API key environment variable is required.")
if not scrapfly_api_key:
    raise ValueError("SCRAPFLY_API_KEY environment variable is required.")


_model = OpenAIModel(
    # Use appropriate key var name here depending on your .env setup
    api_key=openrouter_api_key,
    model_name=openai_model_name,
    base_url=openai_base_url
)


# System prompt for the agent
system_prompt = (
    "You are an AI agent designed to process a user's query and identify top competitors. Your task is to:\n"
    "1. Analyze the query to determine if it’s an app-related request (e.g., 'app for music') or a local business request (e.g., 'restaurant for Nepali cuisine in London').\n"
    "2. Identify exactly 3 top competitors based on the query.\n"
    "3. For apps: Include their Google Play Store and App Store URLs (to be validated later).\n"
    "4. For local businesses: Only provide competitor names, no URLs.\n"
    "5. Return the results in a structured format with exactly 3 competitors using the provided tool.\n\n"
    "Guidelines:\n"
    "- If the query is unclear, return 3 placeholder competitors (e.g., 'Unknown 1', 'Unknown 2', 'Unknown 3').\n"
    "- Do not fabricate competitor names unless the query is unclear—rely on common knowledge or search-like logic.\n"
    "- For apps, provide initial URL guesses; they will be validated later.\n"
    "- Ensure the final output structure matches the CompetitorResponse model precisely."
)

# Create the agent
competitor_finder_agent = Agent(
    _model,
    system_prompt=system_prompt,
    result_type=CompetitorResponse,
    result_tool_name="competitor_results",
    result_tool_description="Log the competitors based on the query",
    result_retries=3,
)

# Helper functions for web scraping and API calls
def search_web_with_scrapfly(query: str) -> List[str]:
    logger.info(f"Searching Scrapfly for query: {query}")
    try:
        client = ScrapflyClient(key=scrapfly_api_key)
        # Use a more targeted search query if possible
        search_url = f"https://www.google.com/search?q=top+{query}+competitors"
        logger.info(f"Scraping URL: {search_url}")
        result = client.scrape(ScrapeConfig(
            url=search_url,
            render_js=True, # JS rendering might be needed for some results
            # asp=True, # Consider using Anti Scraping Protection if needed
            # country="US" # Set country if relevant
        ))
        soup = BeautifulSoup(result.content, "html.parser")
        
        # Try to find more reliable elements (this is brittle and depends on Google's changing layout)
        competitors = set() # Use a set to avoid duplicates easily
        
        # Example: Look for organic result titles/links (selectors need frequent updates)
        # This is a guess, inspect Google search results HTML for better selectors
        for link in soup.select('div.g a h3'): # Example selector, likely needs adjustment
             parent_a = link.find_parent('a')
             if parent_a and parent_a.get('href'):
                 href = parent_a.get('href')
                 # Basic filtering: avoid google links, try to get domain
                 if 'google.com' not in href and 'http' in href:
                     try:
                        # Extract domain or a meaningful name part
                        domain = href.split('/')[2].replace('www.', '')
                        name = domain.split('.')[0].capitalize() # Simple name extraction
                        if name and len(name) > 2: # Basic check
                            competitors.add(name)
                     except IndexError:
                         pass # Ignore malformed URLs

        competitor_list = list(competitors)[:3] # Take up to 3 unique names found
        
        # Fallback if parsing fails
        if not competitor_list:
             logger.warning("Could not parse specific competitors, trying simpler link text.")
             for link in soup.select("a[href*='.com']")[:5]: # Look broader if specific parse fails
                 name = link.text.strip() or link.get("href", "").split("/")[2].split(".")[0].capitalize()
                 if name and name not in competitor_list and len(competitor_list) < 3 and 'Google' not in name:
                     competitor_list.append(name)

        logger.info(f"Found potential competitors via Scrapfly: {competitor_list}")
        return competitor_list

    except Exception as e:
        logger.error(f"Error searching Scrapfly for query '{query}': {e}")
        return []


def get_app_store_url(competitor_name: str) -> Optional[str]:
    logger.info(f"Fetching App Store URL for: {competitor_name}")
    try:
        # Use a more specific search term for apps
        search_term = f"{competitor_name} app"
        itunes_url = f"https://itunes.apple.com/search?term={search_term}&entity=software&limit=1&country=US" # Limit to 1, specify country
        response = requests.get(itunes_url, timeout=10) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        if data.get("results"):
            # Check if the found app name closely matches the competitor name
            found_name = data["results"][0].get("trackName", "").lower()
            if competitor_name.lower() in found_name:
                url = data["results"][0].get("trackViewUrl")
                logger.info(f"App Store URL found: {url}")
                return url
            else:
                logger.warning(f"Found app '{found_name}' but it doesn't closely match '{competitor_name}'")
                return None
        else:
             logger.warning(f"No results in App Store search for {competitor_name}")
             return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching App Store URL for '{competitor_name}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching App Store URL for '{competitor_name}': {e}")
        return None


def get_google_play_url(competitor_name: str) -> Optional[str]:
    logger.info(f"Fetching Google Play URL for: {competitor_name}")
    try:
        client = ScrapflyClient(key=scrapfly_api_key)
        # Use a more specific search term
        search_term = f"{competitor_name} app"
        search_url = f"https://play.google.com/store/search?q={search_term}&c=apps&hl=en&gl=US" # Specify lang/country
        
        logger.info(f"Scraping Google Play search: {search_url}")
        result = client.scrape(ScrapeConfig(
            url=search_url,
            render_js=False, # JS might not be needed for Play Store search results
            # asp=True, # Consider enabling ASP
            # country="US"
        ))
        soup = BeautifulSoup(result.content, "html.parser")

        # Google Play structure changes; selector needs verification
        # Find the first app link in the search results
        # This selector targets the card containing the app link
        app_card = soup.select_one('div[role="listitem"] a') # Adjust selector based on current Play Store HTML structure

        if app_card and app_card.get('href'):
            href = app_card.get('href')
            if href.startswith('/store/apps/details?id='):
                # Extract package ID
                package_id = href.split('id=')[-1]
                url = f"https://play.google.com/store/apps/details?id={package_id}&hl=en&gl=US"

                # Optional: Check if the app title roughly matches
                app_title_element = app_card.select_one('div > div:nth-of-type(2) > div:nth-of-type(1) > div') # Very fragile selector!
                app_title = app_title_element.text.strip().lower() if app_title_element else ""
                if competitor_name.lower() in app_title or not app_title: # Check if name matches or if title wasn't found
                    logger.info(f"Google Play URL found: {url}")
                    return url
                else:
                    logger.warning(f"Found Google Play app '{app_title}' ({url}) but it doesn't closely match '{competitor_name}'")
                    return None
            else:
                logger.warning(f"Found link '{href}' but it's not a details link.")
                return None
        else:
            logger.warning(f"No app link found in Google Play search results for {competitor_name}")
            return None

    except Exception as e:
        logger.error(f"Error fetching Google Play URL for '{competitor_name}' via Scrapfly: {e}")
        return None


@competitor_finder_agent.result_validator
async def validate_result(ctx: RunContext[None], result: CompetitorResponse) -> CompetitorResponse:
    """
    Validates the result from the LLM.
    Ensures exactly 3 competitors.
    Fetches/validates URLs for app queries.
    Clears URLs for non-app queries.
    Uses Scrapfly for missing competitors.
    """
    logger.info(f"Validating LLM result for query: {result.query}")
    query = result.query.lower()
    competitors = result.competitors

    # Determine if it's likely an app query (simple check)
    is_app_query = "app" in query or any(c.app_store_url or c.google_play_url for c in competitors)
    logger.info(f"Query identified as app-related: {is_app_query}")

    # Ensure exactly 3 competitors - Use Scrapfly if needed
    if len(competitors) != 3:
        logger.warning(f"LLM returned {len(competitors)} competitors, expected 3. Attempting to fetch from Scrapfly.")
        # Prepare a search query for Scrapfly (remove generic terms like 'app')
        search_query = query.replace(" app ", " ").replace("best ", "").replace(" top ", "").strip()
        
        try:
             competitor_names = await asyncio.to_thread(search_web_with_scrapfly, search_query)
             
             # Combine LLM results (if any valid ones) with scraped results
             existing_names = {c.name for c in competitors if 'unknown' not in c.name.lower()}
             combined_names = list(existing_names)
             for name in competitor_names:
                 if name not in combined_names and len(combined_names) < 3:
                      combined_names.append(name)

             # Fill remaining slots with placeholders if still needed
             while len(combined_names) < 3:
                 combined_names.append(f"Unknown {len(combined_names) + 1}")

             # Rebuild the competitor list
             competitors = [Competitor(name=name) for name in combined_names[:3]]
             logger.info(f"Adjusted competitor list: {[c.name for c in competitors]}")

        except Exception as e:
             logger.error(f"Scrapfly search failed during validation: {e}")
             # Fallback to placeholders if Scrapfly fails
             if len(competitors) < 3:
                  competitors.extend([Competitor(name=f"Unknown {i+1}") for i in range(len(competitors), 3)])
             competitors = competitors[:3] # Ensure exactly 3

    # Process URLs based on query type
    validated_competitors = []
    url_fetch_tasks = []

    if is_app_query:
        logger.info("Processing as app query: Fetching/Validating URLs...")
        for competitor in competitors:
            # Create tasks to fetch URLs concurrently
            if not competitor.app_store_url or "apple.com" not in competitor.app_store_url:
                url_fetch_tasks.append(asyncio.to_thread(get_app_store_url, competitor.name))
            else:
                url_fetch_tasks.append(asyncio.sleep(0, result=competitor.app_store_url)) # Keep valid URL

            if not competitor.google_play_url or "play.google.com" not in competitor.google_play_url:
                 url_fetch_tasks.append(asyncio.to_thread(get_google_play_url, competitor.name))
            else:
                 url_fetch_tasks.append(asyncio.sleep(0, result=competitor.google_play_url)) # Keep valid URL
        
        # Run URL fetching tasks concurrently
        fetched_urls = await asyncio.gather(*url_fetch_tasks)

        # Reconstruct competitors with fetched/validated URLs
        url_idx = 0
        for competitor in competitors:
            app_store_url = fetched_urls[url_idx]
            google_play_url = fetched_urls[url_idx + 1]
            url_idx += 2

            # Basic validation of fetched URLs
            if app_store_url and "apple.com" not in app_store_url:
                 logger.warning(f"Invalid App Store URL structure for {competitor.name}: {app_store_url}")
                 app_store_url = None
            if google_play_url and "play.google.com" not in google_play_url:
                 logger.warning(f"Invalid Google Play URL structure for {competitor.name}: {google_play_url}")
                 google_play_url = None

            validated_competitors.append(Competitor(
                name=competitor.name,
                app_store_url=app_store_url,
                google_play_url=google_play_url
            ))

    else:
        logger.info("Processing as non-app query: Clearing URLs...")
        # For non-app queries, ensure no URLs are present
        for competitor in competitors:
            validated_competitors.append(Competitor(
                name=competitor.name,
                app_store_url=None,
                google_play_url=None
            ))

    # Final check and return
    try:
        final_result = CompetitorResponse(query=result.query, competitors=validated_competitors)
        # Log the final validated data before returning
        logger.info(f"Validation successful. Final data: {final_result.model_dump(exclude_none=True)}")
        return final_result
    except ValidationError as e:
        logger.error(f"Final validation failed after adjustments: {e}")
        # This shouldn't happen if logic above is correct, but raise retry if it does
        raise ModelRetry(f"Result structure became invalid after validation: {e}")


async def run_agent(query: str) -> Optional[CompetitorResponse]:
    """Runs the agent and returns the CompetitorResponse object or None on failure."""
    logger.info(f"Running agent with query: {query}")
    try:
        response = await competitor_finder_agent.run(query)
        # Assuming response.data holds the CompetitorResponse object based on pydantic-ai structure
        # Adjust if the structure is different (e.g., response directly is the model)
        if isinstance(response, CompetitorResponse):
             result = response
        elif hasattr(response, 'data') and isinstance(response.data, CompetitorResponse):
             result = response.data
        else:
             logger.error(f"Agent returned unexpected response type: {type(response)}")
             # Attempt to create a placeholder error response
             return CompetitorResponse(query=query, competitors=[Competitor(name=f"Error {i+1}") for i in range(3)])

        logger.info(f"Agent run completed successfully for query: {query}")
        return result
    except ModelRetry as e:
         logger.error(f"Agent failed after retries for query '{query}': {e}")
         # Return a placeholder error response consistent with the structure
         return CompetitorResponse(query=query, competitors=[Competitor(name=f"Failed {i+1}") for i in range(3)])
    except Exception as e:
        logger.error(f"An unexpected error occurred during agent run for query '{query}': {e}", exc_info=True)
        # Return a placeholder error response
        return CompetitorResponse(query=query, competitors=[Competitor(name=f"Exception {i+1}") for i in range(3)])


if __name__ == "__main__":
    # Example 1: App query
    query1 = "best music app"
    logger.info(f"--- Running Example: Query = '{query1}' ---")
    result1 = asyncio.run(run_agent(query1))
    logger.info(f"--- Agent run finished for: '{query1}' ---")

    # --- OUTPUT MODIFICATION ---
    # Instead of printing parts, serialize the entire result object to a JSON string
    if result1:
        # Use model_dump_json() for direct JSON string output
        # indent=2 makes it pretty-printed like the example
        print(result1.model_dump_json(indent=2, exclude_none=True)) # exclude_none=True cleans up nulls if desired
    else:
        # Handle cases where the agent run might fail unexpectedly
        logger.error(f"Agent failed to return a result for query: {query1}")
        # Output an error JSON structure to stdout
        error_output = {
            "query": query1,
            "competitors": [
                 {"name": "Processing Error 1", "app_store_url": None, "google_play_url": None},
                 {"name": "Processing Error 2", "app_store_url": None, "google_play_url": None},
                 {"name": "Processing Error 3", "app_store_url": None, "google_play_url": None}
            ],
             "error_message": "Agent failed to produce a valid competitor list."
        }
        print(json.dumps(error_output, indent=2))

