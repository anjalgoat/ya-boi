import logging
import os
import sys
import asyncio
import json
from typing import List
from urllib.parse import quote_plus, unquote

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

load_dotenv()

class WebResult(BaseModel):
    title: str
    url: str

async def search_web_with_scraperapi(
    keyword: str,
    scraperapi_key: str,
    search_engine: str = "google",
    http_client: httpx.AsyncClient = None
) -> List[WebResult]:
    logger.info(f"Searching {search_engine} via ScraperAPI for keyword: {keyword}")

    if not scraperapi_key:
        logger.error("ScraperAPI key is missing for web search.")
        return []

    search_urls = {
        "google": "https://www.google.com/search?q={}&hl=en&gl=us",
        "bing": "https://www.bing.com/search?q={}&cc=us"
    }
    base_search_url = search_urls.get(search_engine, search_urls["google"])

    # Add some common negative keywords to the search to filter results directly
    excluded_sites_and_paths = [
        "-site:youtube.com", # Exclude specific irrelevant domains
        "-site:amazon.com",
        "-inurl:(signup|login|cart|shop|product|download|jobs|careers)", # Exclude common non-informational paths
        "-filetype:pdf", # Exclude PDF files
        "-filetype:xml",
        "-site:wikipedia.org", # Often too general for specific market research
        "-site:support.google.com",
        "-site:microsoft.com" # Unless specifically relevant
    ]
    encoded_keyword = quote_plus(keyword + " " + " ".join(excluded_sites_and_paths))
    target_url = base_search_url.format(encoded_keyword)
    logger.info(f"Target search URL for ScraperAPI: {target_url}")

    scraper_api_endpoint = "http://api.scraperapi.com"
    params = {
        'api_key': scraperapi_key,
        'url': target_url,
        'render': 'true',
        'country_code': 'us',
        # 'premium': 'true', # Consider for harder targets
    }

    results = []
    unique_urls = set()
    client_to_use = http_client if http_client else httpx.AsyncClient()

    try:
        logger.info(f"Making request to ScraperAPI for {search_engine} search.")
        response = await client_to_use.get(scraper_api_endpoint, params=params, timeout=60.0)
        response.raise_for_status()

        logger.info(f"ScraperAPI response status: {response.status_code} for {search_engine} search.")
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
        link_elements_data = []

        # --- HTML Selectors (Need Frequent Verification) ---
        if search_engine == "google":
            result_blocks = soup.select('div.g, div.Ww4FFb, div.Gx5Zad, div.VCmPNe, div.sATSHe, div.Zn_bQc, div.xpd') # Common Google result blocks
            for block in result_blocks:
                link_tag = block.select_one('a[href][data-ved]') or block.select_one('a[href][jsname]') or block.select_one('a[href]')
                if link_tag and link_tag.get('href','').startswith('http'):
                    href = link_tag['href']
                    title_tag = block.select_one('h3')
                    title_text = title_tag.get_text(strip=True) if title_tag else "Title Not Found"
                    if title_text and title_text != "Title Not Found" and len(title_text) > 5:
                        link_elements_data.append({'link': href, 'title': title_text})
        elif search_engine == "bing":
            result_blocks = soup.select('li.b_algo')
            for block in result_blocks:
                link_tag = block.select_one('h2 a[href]')
                if link_tag and link_tag.get('href','').startswith('http'):
                    link_elements_data.append({'link': link_tag['href'], 'title': link_tag.get_text(strip=True)})
        # --- End HTML Selectors ---

        logger.info(f"Found {len(link_elements_data)} potential result elements via ScraperAPI for {search_engine}.")

        for elem_data in link_elements_data:
            href = elem_data['link']
            title = elem_data['title']
            title = ' '.join(title.split())
            if len(title) > 150: title = title[:147] + "..." # Slightly shorter title limit

            cleaned_href = unquote(href)

            # Stricter filtering for relevance
            if (cleaned_href.startswith(("http://", "https://")) and
                cleaned_href not in unique_urls and
                not any(skip_domain in cleaned_href.lower() for skip_domain in [
                    "google.com/search", "bing.com/search", "microsoft.com/search",
                    "google.com/aclk", "googleadservices.com", "doubleclick.net",
                    "facebook.com/sharer", "twitter.com/intent", "linkedin.com/share",
                    "pinterest.com/pin", "youtube.com", "youtube.com/watch?", "amazon.com"
                ]) and
                not any(cleaned_href.lower().endswith(ext) for ext in ['.pdf', '.xml', '.ppt', '.doc', '.xls'])
            ):
                try:
                    web_result = WebResult(title=title, url=cleaned_href)
                    results.append(web_result)
                    unique_urls.add(cleaned_href)
                    if len(results) >= 5: # Limit to 5 high-quality results
                        break
                except ValidationError as ve:
                    logger.warning(f"Pydantic validation failed for web result: title='{title}', url='{cleaned_href}'. Error: {ve}")

        logger.info(f"Found {len(results)} valid and unique results after filtering for {search_engine}.")

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from ScraperAPI during {search_engine} search: {e.response.status_code} for {target_url}")
        if e.response: logger.debug(f"ScraperAPI Error Response: {e.response.text[:500]}...")
    except httpx.RequestError as e:
        logger.error(f"Network error connecting to ScraperAPI for {search_engine} search: {e} for {target_url}")
    except Exception as e:
        logger.error(f"Unexpected error during ScraperAPI {search_engine} search for '{keyword}': {e}", exc_info=True)
    finally:
        if not http_client and client_to_use:
            await client_to_use.aclose()
    return results[:5]

async def run_search(query: str, scraper_api_key: str) -> List[WebResult]:
    cleaned_query = query.lower().replace("app for", "").strip()
    keyword = f"{cleaned_query} market research OR analysis OR overview OR trends"
    logger.info(f"Using effective keyword: '{keyword}' derived from query: '{query}' for blog URL search.")

    results = []
    async with httpx.AsyncClient() as client:
        for engine in ["google", "bing"]: # Try Google first, then Bing as fallback
            try:
                engine_results = await search_web_with_scraperapi(keyword, scraper_api_key, engine, http_client=client)
                if engine_results:
                    results = engine_results
                    logger.info(f"Successfully retrieved {len(results)} results from {engine} via ScraperAPI.")
                    break
                logger.info(f"No results from {engine} via ScraperAPI, trying next if available.")
            except Exception as e:
                logger.error(f"Error searching {engine} via ScraperAPI: {e}", exc_info=True)
    return results

if __name__ == "__main__":
    example_query = "ai in healthcare"
    logger.info(f"--- Starting Blog URL Search (ScraperAPI) for query: '{example_query}' ---")
    
    api_key = os.getenv("SCRAPERAPI_KEY")
    if not api_key:
        logger.error("SCRAPERAPI_KEY not found in environment for standalone test.")
        sys.exit(1)
        
    final_results = asyncio.run(run_search(example_query, api_key))
    logger.info(f"--- Blog URL Search finished for query: '{example_query}' ---")

    results_dict_list = [result.model_dump() for result in final_results]
    json_output = json.dumps(results_dict_list, indent=2)
    print(json_output)