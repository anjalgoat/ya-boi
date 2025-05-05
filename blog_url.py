import logging
import os
import asyncio
from typing import List
from urllib.parse import quote_plus
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from scrapfly import ScrapflyClient, ScrapeConfig, ScrapflyError
from bs4 import BeautifulSoup
import json # Import the json module for JSON output

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
scrapfly_api_key = os.getenv("SCRAPFLY_API_KEY")
if not scrapfly_api_key:
    raise ValueError("SCRAPFLY_API_KEY environment variable is required")

# Pydantic model for web results
class WebResult(BaseModel):
    title: str
    url: str

# Updated Scrapfly search function (with minor robustness improvements)
def search_web_with_scrapfly(keyword: str, scrapfly_api_key: str, search_engine: str = "google") -> List[WebResult]:
    logger.info(f"Searching {search_engine} for keyword: {keyword}")
    client = ScrapflyClient(key=scrapfly_api_key)

    # Search engine URLs
    search_urls = {
        "google": "https://www.google.com/search?q={}&hl=en&gl=us", # Add lang/country
        "bing": "https://www.bing.com/search?q={}&cc=us" # Add country
    }
    base_url = search_urls.get(search_engine, search_urls["google"])

    # Encode keyword (consider refining filters based on needs)
    # Site filter might be too restrictive - removed for broader search
    # encoded_keyword = quote_plus(keyword + " site:*.edu | site:*.org | site:*.com -inurl:(signup | login | shop)")
    encoded_keyword = quote_plus(keyword + " -site:youtube.com -site:amazon.com -inurl:(signup|login|cart|shop|product)") # Example refinement
    url = base_url.format(encoded_keyword)
    logger.info(f"Requesting URL: {url}")

    results = []
    # Simplified retry logic: Try full features, then minimal if failed
    configs_to_try = [
        {"render_js": True, "asp": True, "country": "US"},  # Full features
        {"render_js": False, "asp": False, "country": "US"} # Minimal
    ]
    unique_urls = set() # Track unique URLs

    for attempt, config in enumerate(configs_to_try, 1):
        try:
            scrape_config = ScrapeConfig(
                url=url,
                render_js=config["render_js"],
                country=config["country"],
                asp=config["asp"],
                cache=True  # Enable caching
                # Timeout can sometimes cause issues with complex pages/JS rendering
            )
            logger.info(f"Scrapfly attempt {attempt} with config: {config}")
            result = client.scrape(scrape_config)

            logger.info(f"Scrapfly response status: {result.status_code} (Attempt {attempt})")
            # logger.debug(f"Raw HTML snippet (Attempt {attempt}): {result.content[:500]}") # Keep debug minimal

            soup = BeautifulSoup(result.content, "html.parser")

            # Selectors need constant maintenance based on search engine HTML structure
            link_elements = []
            # Google Selectors (Example - VERIFY THESE)
            if search_engine == "google":
                 # Try finding organic results containers first
                 # This is very fragile - Inspect actual Google HTML
                 organic_results = soup.select('div.g, div.Ww4FFb, div[jscontroller="SC7lYd"]')
                 for res_container in organic_results:
                     link_tag = res_container.select_one('a[href^="http"]')
                     if link_tag:
                         title_tag = link_tag.select_one('h3')
                         if title_tag:
                              link_elements.append((link_tag, title_tag)) # Store pair

            # Bing Selectors (Example - VERIFY THESE)
            elif search_engine == "bing":
                 bing_results = soup.select('li.b_algo')
                 for res_container in bing_results:
                     link_tag = res_container.select_one('h2 a[href^="http"]')
                     if link_tag:
                          title_tag = link_tag # Bing often has title directly in link
                          link_elements.append((link_tag, title_tag))

            logger.info(f"Found {len(link_elements)} potential result elements (Attempt {attempt})")

            for link, title_elem in link_elements:
                href = link.get("href", "")
                if not href: continue

                # Clean Google redirects (if still necessary)
                if "/url?q=" in href:
                    href = href.split("/url?q=")[-1].split("&")[0]

                title = title_elem.get_text(strip=True) if title_elem else "No Title Found"
                # Basic title cleanup
                title = ' '.join(title.split()) # Remove excess whitespace
                if len(title) > 200:
                    title = title[:197] + "..."

                # Filter URLs and ensure uniqueness
                if (href.startswith(("http://", "https://")) and
                    href not in unique_urls and
                    all(x not in href.lower() for x in ["google.com", "bing.com", "/signup", "/login", "/shop", "/cart", "/product"]) and
                    title and title != "No Title Found"):
                    try:
                         web_result = WebResult(title=title, url=href)
                         results.append(web_result)
                         unique_urls.add(href)
                         if len(results) >= 5: # Limit to 5 results
                             break
                    except ValidationError as ve:
                         logger.warning(f"Validation failed for result: title='{title}', url='{href}'. Error: {ve}")


                if len(results) >= 5: break # Exit inner loop

            if results:
                logger.info(f"Found {len(results)} valid results on attempt {attempt}. Stopping search.")
                break  # Exit outer loop if results found

        except ScrapflyError as e:
            logger.error(f"Scrapfly API error (attempt {attempt}): {e}")
            if e.response: # Check if response object exists
                 logger.error(f"Status code: {e.response.status_code}, Response body snippet: {e.response.text[:200]}")
        except Exception as e:
            logger.error(f"Unexpected error during scraping (attempt {attempt}): {e}", exc_info=True)

        if attempt < len(configs_to_try) and not results:
            logger.info(f"No results found yet, retrying with config: {configs_to_try[attempt]}")
        elif not results:
            logger.warning("All scraping attempts failed or yielded no valid results.")


    # Handle case where no results were found after all attempts
    if not results:
        logger.warning("No relevant results found after all attempts.")

    logger.info(f"Returning {len(results)} results.")
    return results[:5] # Ensure maximum 5 results


# Main execution function (modified to return results)
async def run_search(query: str, scrapfly_api_key: str) -> List[WebResult]:
    """
    Runs the web search for the given query and returns a list of WebResult objects.
    """
    # Simple query cleaning (can be expanded)
    cleaned_query = query.lower().replace("app for", "").strip()
    # Construct a keyword phrase suitable for search
    keyword = f"{cleaned_query} market research OR analysis OR overview OR trends" # Example keyword construction
    logger.info(f"Using effective keyword: '{keyword}' derived from query: '{query}'")

    results = []
    # Try engines sequentially, stopping if results are found
    for engine in ["google", "bing"]:
        try:
             # Run synchronous function in thread pool for asyncio compatibility
             engine_results = await asyncio.to_thread(search_web_with_scrapfly, keyword, scrapfly_api_key, engine)
             if engine_results:
                 results = engine_results
                 logger.info(f"Successfully retrieved {len(results)} results from {engine}.")
                 break # Stop searching if results found
             else:
                  logger.info(f"No results retrieved from {engine}, trying next engine if available.")
        except Exception as e:
            logger.error(f"Error occurred while searching with {engine}: {e}", exc_info=True)
            # Optionally continue to the next engine or break

    return results


if __name__ == "__main__":
    # Example query
    query = "dating app" # Simplified query

    logger.info(f"--- Starting search for query: '{query}' ---")
    # Run the async search function
    final_results = asyncio.run(run_search(query, scrapfly_api_key))
    logger.info(f"--- Search finished for query: '{query}' ---")

    # --- OUTPUT MODIFICATION ---
    # Convert the list of Pydantic WebResult models to a list of Python dictionaries
    results_dict_list = [result.model_dump() for result in final_results]

    # Convert the list of dictionaries to a JSON formatted string
    # Use indent=2 for pretty-printing as shown in the desired format
    json_output = json.dumps(results_dict_list, indent=2)

    # Print the final JSON string to standard output
    print(json_output)