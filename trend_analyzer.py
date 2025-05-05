# --- trend_analyzer.py (Full Code - Requires: python-dotenv, pydantic, pydantic-ai, scrapfly, beautifulsoup4) ---

import os
import asyncio
import logging
import json
from typing import List, Optional

# --- Attempt to import required libraries ---
lib_errors = []
try:
    from dotenv import load_dotenv
except ImportError:
    lib_errors.append("python-dotenv (pip install python-dotenv)")
try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    lib_errors.append("pydantic (pip install pydantic)")
try:
    from pydantic_ai import Agent, RunContext, ModelRetry
    from pydantic_ai.models.openai import OpenAIModel # Or choose another model if preferred
    from pydantic_ai.exceptions import UserError
except ImportError:
    lib_errors.append("pydantic-ai (pip install pydantic-ai)")
try:
    from scrapfly import ScrapflyClient, ScrapeConfig, ScrapeApiResponse
except ImportError:
    lib_errors.append("scrapfly (pip install scrapfly)")
try:
    from bs4 import BeautifulSoup
except ImportError:
    lib_errors.append("beautifulsoup4 (pip install beautifulsoup4)")

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Exit if libraries are missing ---
if lib_errors:
    logger.error("Required libraries are missing:")
    for error in lib_errors:
        logger.error(f" - {error}")
    print("\nERROR: Required libraries are missing. Please install them:")
    for error in lib_errors:
        print(f" - {error}")
    exit() # Stop execution if core libraries are missing
# --- End Library Check ---

# --- Load .env file ---
try:
    # load_dotenv() looks for .env in the current working directory or parent directories
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env') # More specific path finding
    if os.path.exists(dotenv_path):
         load_dotenv(dotenv_path=dotenv_path)
         logger.info(f".env file found at {dotenv_path} and loading attempted.")
    else:
         # Try loading from current directory as fallback
         if load_dotenv():
              logger.info(".env file found in current directory and loading attempted.")
         else:
              logger.warning(".env file not found in script directory or current directory.")
except NameError:
     # This happens if dotenv import failed earlier but we didn't exit
     logger.warning("python-dotenv library not available, cannot load .env file.")
except Exception as e:
     logger.error(f"An error occurred during dotenv loading: {e}")

# --- Environment Variable Checks (Now reading after attempting load_dotenv) ---
llm_api_key = os.getenv("OPENROUTER_API_KEY")
scrapfly_api_key = os.getenv("SCRAPFLY_API_KEY")
llm_model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo") # Default model is okay

# Add checks *after* attempting load_dotenv
keys_loaded = True
if not scrapfly_api_key:
    logger.error("SCRAPFLY_API_KEY not found after attempting to load .env. Check .env file and location.")
    keys_loaded = False
else:
    logger.info("SCRAPFLY_API_KEY loaded successfully.")

if not llm_api_key:
     logger.error("OPENAI_API_KEY not found after attempting to load .env. Check .env file and location.")
     keys_loaded = False
else:
    logger.info("OPENAI_API_KEY loaded successfully.")

# --- Pydantic Models ---
class RelatedQuery(BaseModel):
    """Represents a related query found on Google Trends."""
    query: str = Field(..., description="The related search query text")

class GoogleTrendsResult(BaseModel):
    """Structured data scraped from Google Trends for a keyword."""
    keyword: str = Field(..., description="The keyword searched on Google Trends")
    related_queries_top: List[RelatedQuery] = Field(default_factory=list, description="List of top related queries found")
    related_queries_rising: List[RelatedQuery] = Field(default_factory=list, description="List of rising related queries found")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered during scraping")

# --- Scraping Function ---
def scrape_google_trends_with_scrapfly(keyword: str, country: str = "US") -> GoogleTrendsResult:
    """
    Uses Scrapfly to scrape Google Trends for a given keyword.
    Returns a GoogleTrendsResult object.
    """
    current_scrapfly_key = os.getenv("SCRAPFLY_API_KEY") # Re-check in case it wasn't loaded initially
    if not current_scrapfly_key:
        logger.error("Scrapfly API Key is missing inside scraping function. Cannot perform scrape.")
        return GoogleTrendsResult(keyword=keyword, errors=["Scrapfly API Key (SCRAPFLY_API_KEY) not configured."])

    logger.info(f"Attempting to scrape Google Trends for keyword: '{keyword}' in country: {country}")
    client = ScrapflyClient(key=current_scrapfly_key)
    # Use urlencode for safer keyword handling
    from urllib.parse import quote_plus
    encoded_keyword = quote_plus(keyword)
    trends_url = f"https://trends.google.com/trends/explore?q={encoded_keyword}&geo={country}&hl=en"
    logger.info(f"Scraping URL: {trends_url}")

    result_data = GoogleTrendsResult(keyword=keyword)

    try:
        scrape_result: ScrapeApiResponse = client.scrape(ScrapeConfig(
            url=trends_url,
            render_js=True,       # Crucial for Google Trends
            asp=True,             # Enable Anti Scraping Protection
            country=country,      # Set proxy location
            # timeout=20000,      # Optional: Increase timeout if needed
            # wait_for_selector="div.widget-container" # Example: wait for widgets
        ))

        if not scrape_result.success or scrape_result.status_code >= 400:
             error_msg = f"Scrapfly request failed. URL: {scrape_result.context.get('url')}, Status: {scrape_result.status_code}. Reason: {scrape_result.error}"
             logger.error(error_msg)
             result_data.errors.append(error_msg)
             # Sometimes Google returns error pages, try to get the title
             try:
                  soup_error = BeautifulSoup(scrape_result.content, "html.parser")
                  error_title = soup_error.title.string if soup_error.title else "No Title Found"
                  result_data.errors.append(f"Scraped page title (potential error): {error_title}")
             except Exception:
                   pass # Ignore errors during error page parsing
             return result_data

        logger.info(f"Scrapfly request successful (Status: {scrape_result.status_code}). Parsing content...")
        soup = BeautifulSoup(scrape_result.content, "html.parser")

        # --- !!! PARSING LOGIC - HIGHLY FRAGILE !!! ---
        # Selectors WILL likely break. Adjust based on current Google Trends HTML.
        related_query_widgets = []
        # Find all containers that might hold the "Related queries" widget
        # This selector might need refinement based on actual structure
        all_widgets = soup.select("div.details-widgets-container, div.widget.concepts-widget")
        logger.debug(f"Found {len(all_widgets)} potential widget containers.")

        for widget in all_widgets:
             # Check for a title explicitly containing "Related queries"
             title_element = widget.select_one("div.widget-header-title, h LTR-title") # Multiple possible title selectors
             if title_element and "Related queries" in title_element.get_text(strip=True):
                   logger.info("Found a 'Related queries' widget based on title.")
                   related_query_widgets.append(widget)
                   continue # Found via title, no need to check structure below

             # Fallback: Check structure if title match failed
             # Look for sections clearly labeled "Top" and "Rising" within the widget
             top_section = widget.select_one(".widget-top-entities, div:has(> .widget-title-label:contains('Top'))")
             rising_section = widget.select_one(".widget-rising-entities, div:has(> .widget-title-label:contains('Rising'))")
             if top_section or rising_section:
                  logger.info("Found a potential 'Related queries' widget based on Top/Rising sections.")
                  related_query_widgets.append(widget)

        if not related_query_widgets:
             logger.warning("Could not find any widgets definitively identified as 'Related queries'.")
             result_data.errors.append("Could not find 'Related queries' widgets (adjust selectors).")
        else:
             logger.info(f"Processing {len(related_query_widgets)} identified 'Related queries' widget(s)...")
             processed_widget_count = 0
             for widget in related_query_widgets:
                  processed_widget_count += 1
                  logger.info(f"Processing widget {processed_widget_count}/{len(related_query_widgets)}")
                  # Try to find Top/Rising sections within this specific widget
                  # Selectors are more specific now, adjust as needed
                  top_section = widget.select_one("div:has(> .widget-title-label:contains('Top')), .widget-top-entities")
                  rising_section = widget.select_one("div:has(> .widget-title-label:contains('Rising')), .widget-rising-entities")

                  # Process Top queries
                  if top_section:
                       logger.info("  Processing 'Top' queries section.")
                       # Look for list items or specific query elements
                       query_items = top_section.select(".item .label-text, .entity-info-container .label") # Multiple possible selectors
                       if query_items:
                           found_count = 0
                           for item in query_items:
                               query_text = item.get_text(strip=True)
                               if query_text:
                                   result_data.related_queries_top.append(RelatedQuery(query=query_text))
                                   found_count += 1
                           logger.info(f"    Found {found_count} 'Top' query items.")
                       else:
                            logger.warning("    Found 'Top' section but no query items inside (check selectors like '.item .label-text').")
                  # else: # Don't log if not found, might be normal
                  #      logger.debug("  'Top' queries section not found in this widget.")

                  # Process Rising queries
                  if rising_section:
                      logger.info("  Processing 'Rising' queries section.")
                      query_items = rising_section.select(".item .label-text, .entity-info-container .label") # Multiple possible selectors
                      if query_items:
                          found_count = 0
                          for item in query_items:
                               query_text = item.get_text(strip=True)
                               if query_text:
                                   result_data.related_queries_rising.append(RelatedQuery(query=query_text))
                                   found_count += 1
                          logger.info(f"    Found {found_count} 'Rising' query items.")
                      else:
                           logger.warning("    Found 'Rising' section but no query items inside (check selectors).")
                  # else: # Don't log if not found, might be normal
                  #      logger.debug("  'Rising' queries section not found in this widget.")

        if not result_data.related_queries_top and not result_data.related_queries_rising and not result_data.errors:
             # Check if the page content suggests no data is available
             no_data_element = soup.select_one(".widget-error-title, .feed-item.no-data") # Add selectors for "no data" messages
             if no_data_element:
                  no_data_text = no_data_element.get_text(strip=True)
                  logger.warning(f"Scraping finished, but Google Trends reported no data: '{no_data_text}'")
                  result_data.errors.append(f"Google Trends reported no data: {no_data_text}")
             else:
                  logger.warning("Scraping finished, but no related queries found (check selectors or page content).")
                  result_data.errors.append("Scraping finished, but no related queries found (check selectors/content).")
        # --- End Parsing Logic ---

    except Exception as e:
        error_msg = f"An error occurred during scraping or parsing: {e}"
        logger.error(error_msg, exc_info=True)
        result_data.errors.append(error_msg)

    logger.info(f"Finished scraping attempt for '{keyword}'. Found {len(result_data.related_queries_top)} top, {len(result_data.related_queries_rising)} rising queries. Errors: {len(result_data.errors)}")
    return result_data


# --- Pydantic AI Agent ---
# Initialize agent only if keys were loaded successfully
google_trends_agent = None
agent_init_error = None

if keys_loaded: # Only proceed if keys were found
    try:
        _model = OpenAIModel(
            api_key=llm_api_key, # Use the key loaded from .env
            model_name=llm_model_name,
        )

        system_prompt_trends = (
            "You are an agent designed to fetch Google Trends data for a specific keyword. "
            "Use the provided tool to scrape the data using Scrapfly. The tool will return a JSON string "
            "representing the scraped data or errors encountered. Your final output MUST be the structured "
            "GoogleTrendsResult object, parsing the JSON string returned by the tool."
        )

        google_trends_agent = Agent(
            model=_model, # Corrected parameter name
            system_prompt=system_prompt_trends,
            result_type=GoogleTrendsResult, # Agent's *final* output is still GoogleTrendsResult
            result_tool_name="google_trends_scraper_results",
            result_tool_description="Logs the results from scraping Google Trends for a keyword.",
            # No deps_type needed here as the tool context is None
        )

        @google_trends_agent.tool
        def scrape_google_trends_tool(ctx: RunContext[None], keyword: str, country_code: str = 'US') -> str: # Added ctx, returns str
            """
            Triggers the Scrapfly function to scrape Google Trends for a given keyword and country code (default US).
            Returns the scraped data (or errors) as a JSON string.
            The 'ctx' parameter provides run context but is not used in this tool.
            """
            # We don't use ctx here, but it satisfies the pydantic-ai requirement
            logger.info(f"Tool triggered for keyword: '{keyword}', country: '{country_code}' (Context not used)")
            try:
                # Call the actual scraping function which returns a Pydantic object
                result_obj = scrape_google_trends_with_scrapfly(keyword=keyword, country=country_code)
                # Return the result as a JSON string
                # Use model_dump_json (Pydantic v2)
                json_output = result_obj.model_dump_json()
                logger.debug(f"Tool returning JSON string: {json_output[:200]}...") # Log truncated output
                return json_output
            except Exception as e:
                 logger.error(f"Error executing scrape_google_trends_tool: {e}", exc_info=True)
                 # Return an error structure as a JSON string
                 error_result = GoogleTrendsResult(keyword=keyword, errors=[f"Tool execution failed: {e}"])
                 # Use model_dump_json (Pydantic v2)
                 return error_result.model_dump_json()

    except UserError as e:
         agent_init_error = f"Failed to initialize Agent or register tool: {e}"
         logger.error(agent_init_error, exc_info=True)
    except Exception as e: # Catch potential OpenAIError here too if key is invalid format
         agent_init_error = f"An unexpected error occurred during agent setup: {e}"
         logger.error(agent_init_error, exc_info=True)

if not keys_loaded:
     logger.warning("Agent not initialized because required API keys were not loaded.")
elif agent_init_error:
     logger.error(f"Agent not initialized due to error: {agent_init_error}")


# --- Main Execution Logic ---
async def run_trends_agent(keyword: str, country: str = "US") -> Optional[GoogleTrendsResult]:
    """Runs the Google Trends agent for a given keyword."""
    if not google_trends_agent:
         error_msg = "Google Trends Agent was not initialized successfully. Cannot run."
         logger.error(error_msg)
         final_errors = [error_msg]
         if not keys_loaded:
              final_errors.append("Missing API keys (check .env file).")
         if agent_init_error:
              final_errors.append(f"Initialization error: {agent_init_error}")
         return GoogleTrendsResult(keyword=keyword, errors=final_errors)

    logger.info(f"Running Google Trends agent for keyword: '{keyword}', country: '{country}'")
    try:
        # The agent will call the tool, which now returns a JSON string.
        # The agent framework (or LLM) should process this string and format the final output
        # according to the agent's result_type (GoogleTrendsResult).
        response = await google_trends_agent.run(
            # Provide clear instructions to the LLM
            f"Scrape Google Trends using the tool for the keyword '{keyword}' in country '{country}'. "
            f"The tool will return a JSON string. Parse this JSON string and return the final result "
            f"strictly formatted as the GoogleTrendsResult structure. Include any errors reported by the tool.",
        )

        # Response handling: pydantic-ai might return the parsed object or the raw data
        if isinstance(response, GoogleTrendsResult):
            result = response
            logger.info("Agent returned a parsed GoogleTrendsResult object.")
        # Handle potential nesting if using older pydantic-ai versions or specific configs
        elif hasattr(response, 'data') and isinstance(response.data, GoogleTrendsResult):
            result = response.data
            logger.info("Agent returned nested data containing a GoogleTrendsResult object.")
        # Handle case where LLM might just return the JSON string from the tool
        elif isinstance(response, str):
             logger.warning("Agent returned a string, attempting to parse as GoogleTrendsResult.")
             try:
                  # Use model_validate_json (Pydantic v2)
                  result = GoogleTrendsResult.model_validate_json(response)
                  logger.info("Successfully parsed string response into GoogleTrendsResult.")
             except ValidationError as ve:
                  parse_error_msg = f"Failed to parse agent's string response into GoogleTrendsResult: {ve}"
                  logger.error(parse_error_msg)
                  logger.debug(f"Raw string response from agent: {response}")
                  # Return the keyword and the parsing error
                  return GoogleTrendsResult(keyword=keyword, errors=[parse_error_msg, f"Raw agent response: {response[:500]}..."])
             except Exception as pe: # Catch other potential parsing errors (e.g., JSONDecodeError)
                  parse_error_msg = f"Unexpected error parsing agent's string response: {pe}"
                  logger.error(parse_error_msg)
                  logger.debug(f"Raw string response from agent: {response}")
                  return GoogleTrendsResult(keyword=keyword, errors=[parse_error_msg, f"Raw agent response: {response[:500]}..."])
        else:
            # Handle unexpected response types
            error_msg = f"Agent returned unexpected response type: {type(response)}"
            logger.error(error_msg)
            logger.debug(f"Unexpected response content: {response}")
            return GoogleTrendsResult(keyword=keyword, errors=[error_msg])

        logger.info(f"Agent run completed successfully for keyword: '{keyword}'")
        return result

    except ModelRetry as e:
         retry_error_msg = f"Agent failed after retries for keyword '{keyword}': {e}"
         logger.error(retry_error_msg)
         return GoogleTrendsResult(keyword=keyword, errors=[retry_error_msg])
    except Exception as e:
        run_error_msg = f"An unexpected error occurred during agent run for keyword '{keyword}': {e}"
        logger.error(run_error_msg, exc_info=True)
        return GoogleTrendsResult(keyword=keyword, errors=[run_error_msg])


if __name__ == "__main__":
    # --- Script Entry Point ---
    search_keyword = "artificial intelligence"
    search_country = "US" # Example: Use 'NP' for Nepal if desired

    print(f"\n--- Attempting to run Google Trends Agent ---")
    print(f"--- Keyword: '{search_keyword}', Country: {search_country} ---")
    print("--- Requires: .env file with API keys, installed libraries ---")

    # Check if keys were loaded and agent initialized before running
    if not keys_loaded:
        print("\nERROR: Required API keys were not loaded. Cannot proceed.")
        print("Ensure the .env file is in the correct directory and contains valid keys.")
    elif google_trends_agent is None:
         print(f"\nError: Agent could not be initialized. Check logs. Last error: {agent_init_error}")
    else:
        try:
            # Run the main async function
            agent_result = asyncio.run(run_trends_agent(search_keyword, search_country))

            print(f"\n--- Agent Final Result for '{search_keyword}' ---")
            if agent_result:
                # Use model_dump_json (Pydantic v2) or .json() (Pydantic v1)
                # exclude_none=True makes the output cleaner if some fields are empty
                print(agent_result.model_dump_json(indent=2, exclude_none=True))
            else:
                # This case should ideally be handled by run_trends_agent returning an error object
                print("Agent did not return a result (Check logs for errors during run_trends_agent).")

        except Exception as e:
            # Catch errors during the asyncio.run call itself
            print(f"\nFATAL ERROR: An unexpected error occurred during the main execution block: {e}")
            logger.error("Error in __main__ execution block", exc_info=True)

    # --- Final Reminders ---
    print("\n--- Important Reminders ---")
    print("1. Google Trends Scraping: The HTML parsing selectors are very likely to break over time. They need regular inspection and updates.")
    print("2. Check Logs: Review the full console output/logs for detailed INFO, WARNING, and ERROR messages from the scraping process.")
    print("3. Alternatives: If scraping is unreliable, consider the unofficial 'pytrends' library.")