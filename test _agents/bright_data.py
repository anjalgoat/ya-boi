import asyncio
import logging
import os
from typing import List, Optional

import httpx                      # Async HTTP client
import instructor                 # For Pydantic structured output from LLM
from bs4 import BeautifulSoup      # Keep for potential HTML pre-processing/stripping
from dotenv import load_dotenv     # To load API keys from .env file
from openai import AsyncOpenAI     # Async OpenAI client
from pydantic import BaseModel, Field, HttpUrl

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") # Prioritize OpenAI key directly
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
bright_data_api_token = os.getenv("BRIGHT_DATA_API_TOKEN")
bright_data_zone = os.getenv("BRIGHT_DATA_ZONE")

# Determine LLM settings (Default to OpenAI, fallback to OpenRouter)
llm_api_key = openai_api_key or openrouter_api_key
llm_base_url = None
default_model = "gpt-4o" # Default to a capable OpenAI model

if not openai_api_key and openrouter_api_key:
    llm_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    default_model = os.getenv("OPENAI_MODEL", "mistralai/mistral-7b-instruct") # Example OpenRouter model
    logger.info(f"Using OpenRouter: Base URL={llm_base_url}, Model={default_model}")
elif openai_api_key:
     default_model = os.getenv("OPENAI_MODEL", default_model) # Allow overriding default OpenAI model
     logger.info(f"Using OpenAI: Model={default_model}")
else:
    raise ValueError("Either OPENAI_API_KEY or OPENROUTER_API_KEY environment variable is required.")

if not bright_data_api_token:
    raise ValueError("BRIGHT_DATA_API_TOKEN environment variable is required.")
if not bright_data_zone:
    raise ValueError("BRIGHT_DATA_ZONE environment variable is required.")

# Max characters of HTML to send to LLM (adjust based on model context window)
MAX_HTML_CHARS_FOR_LLM = 30000 # Example limit

# --- Pydantic Models ---
class GooglePlayAppInput(BaseModel):
    """Input defining the app and region to scrape."""
    app_id: str = Field(..., description="Google Play app identifier (e.g., 'com.google.android.gm')")
    country: str = Field(..., description="2-letter country code (e.g., 'us')")
    lang: str = Field(..., description="2-letter language code (e.g., 'en')")

class ReviewDetail(BaseModel):
    """Represents the extracted details of a single app review."""
    author: Optional[str] = Field(None, description="Name of the reviewer")
    rating_text: Optional[str] = Field(None, description="Rating description (e.g., 'Rated 4 stars out of 5') or numeric score")
    date: Optional[str] = Field(None, description="Date the review was posted")
    review_text: Optional[str] = Field(None, description="The main content of the review")

class ScrapingResponse(BaseModel):
    """Represents the result of scraping reviews for a specific app, extracted by an LLM."""
    app_id: str
    country: str
    language: str
    reviews: List[ReviewDetail] = Field(..., description="List of reviews extracted by the AI from the HTML")
    review_count: int = Field(..., description="Number of reviews successfully extracted by the AI")

# --- LLM Client Setup ---
try:
    # Use the standard AsyncOpenAI client, Instructor will patch its create method
    openai_client = AsyncOpenAI(
        api_key=llm_api_key,
        base_url=llm_base_url, # Will be None if using default OpenAI
    )
    # Patch the client instance
    instructor_client = instructor.patch(openai_client, mode=instructor.Mode.TOOLS) # Use TOOLS mode for better model support
    LLM_AVAILABLE = True
    logger.info("Instructor patched AsyncOpenAI client successfully.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
    LLM_AVAILABLE = False

# --- Core Scraping and Parsing Function (The "Tool") ---
async def scrape_and_parse_google_play_reviews(
    app_input: GooglePlayAppInput,
    httpx_client: httpx.AsyncClient,
    ai_client: AsyncOpenAI, # Expecting the patched client
    ai_model: str = default_model
) -> Optional[ScrapingResponse]:
    """
    Tool function: Scrapes Google Play reviews using Bright Data Direct API
    and parses the HTML using an LLM via Instructor.
    """
    if not LLM_AVAILABLE:
        logger.error("LLM Client not available. Cannot perform AI parsing.")
        return None

    # --- !!! CRITICAL STEP: Determine the Correct Google Play URL !!! ---
    # This placeholder URL needs to be replaced with one that actually
    # yields review HTML when fetched by Bright Data. Requires investigation.
    target_url = f"https://play.google.com/store/apps/details?id={app_input.app_id}&hl={app_input.lang}&gl={app_input.country}&showAllReviews=true"
    logger.info(f"Attempting to fetch target URL via Bright Data: {target_url}")
    # --- !!! END CRITICAL STEP !!! ---

    brightdata_api_endpoint = "https://api.brightdata.com/request"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bright_data_api_token}"
    }
    payload = {
        "zone": bright_data_zone,
        "url": target_url,
        "format": "raw",
        "render": "true", # Essential if reviews load via JS
        "country": app_input.country,
    }

    html_content = None
    logger.info(f"Sending request to Bright Data API for {app_input.app_id}...")
    try:
        response = await httpx_client.post(
            brightdata_api_endpoint, headers=headers, json=payload, timeout=120.0 # Use float timeout
        )
        response.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)
        html_content = response.text
        logger.info(f"Received response from Bright Data (Status: {response.status_code}, Length: {len(html_content)})")

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching from Bright Data API: {e.response.status_code}")
        try:
            # Attempt to log response body snippet on error
             logger.error(f"Bright Data Response Body: {e.response.text[:500]}...")
        except Exception:
             logger.error("Could not read error response body.")
        return None
    except httpx.RequestError as e:
        logger.error(f"Network error connecting to Bright Data API: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during Bright Data fetch: {e}", exc_info=True)
        return None

    if not html_content:
        logger.warning("No HTML content received from Bright Data.")
        return None

    # --- Optional: Pre-process / Truncate HTML ---
    # Consider more sophisticated stripping of irrelevant tags (nav, footer, scripts) using BeautifulSoup
    # For now, simple truncation:
    if len(html_content) > MAX_HTML_CHARS_FOR_LLM:
        logger.warning(f"HTML too long ({len(html_content)} chars), truncating to {MAX_HTML_CHARS_FOR_LLM} for LLM analysis.")
        # Potentially use BS4 here for smarter stripping before truncation
        html_to_parse = html_content[:MAX_HTML_CHARS_FOR_LLM]
    else:
        html_to_parse = html_content
    # --- End Pre-processing ---

    # --- Parse with LLM using Instructor ---
    logger.info(f"Attempting to parse HTML with AI model: {ai_model}")
    try:
        ai_response : ScrapingResponse = await ai_client.chat.completions.create(
            model=ai_model,
            response_model=instructor.Partial[ScrapingResponse], # Use Partial for streaming-like or robust parsing
            messages=[
                {"role": "system", "content": f"You are an expert web scraping assistant. Analyze the provided HTML content for Google Play app '{app_input.app_id}' (country: {app_input.country}, lang: {app_input.lang}). Your goal is to extract all user reviews. For each review, find: author name, rating text/score, date posted, and the full review text. Return the data structured strictly according to the 'ScrapingResponse' schema. Include the original app_id, country, and language. If no reviews are found, return an empty 'reviews' list and 0 for 'review_count'."},
                {"role": "user", "content": f"Parse this HTML:\n\n```html\n{html_to_parse}\n```"}
            ],
            max_retries=1,
        )

        logger.info(f"AI parsing completed. Extracted {len(ai_response.reviews)} reviews.")
        # Ensure review_count reflects actual count
        ai_response.review_count = len(ai_response.reviews)
        # Add app_id, country, language if model didn't include them (though it should based on prompt)
        ai_response.app_id = app_input.app_id
        ai_response.country = app_input.country
        ai_response.language = app_input.lang

        return ai_response # Now it's a complete ScrapingResponse object

    except Exception as e:
        logger.error(f"Error during AI Parsing: {e}", exc_info=True)
        return None
    # --- End AI Parsing ---


# --- Main Execution Logic ---
async def main():
    """Main async function to run the scraping tool."""
    if not LLM_AVAILABLE:
        logger.error("Cannot run main function: LLM Client (e.g., OpenAI) is not available.")
        return

    # Initialize shared clients
    async with httpx.AsyncClient() as httpx_client:
        # Define test input
        test_input = GooglePlayAppInput(
            app_id='com.google.android.gm', # Example: Gmail
            country='us',
            lang='en'
        )

        logger.info(f"--- Starting Google Play Review Scraping Tool for {test_input.app_id} ---")
        result = await scrape_and_parse_google_play_reviews(
            app_input=test_input,
            httpx_client=httpx_client,
            ai_client=instructor_client # Pass the patched client
        )

        if result:
            print("\n" + "="*20 + " Scraping Result " + "="*20)
            print(f"App ID: {result.app_id}")
            print(f"Country: {result.country}")
            print(f"Language: {result.language}")
            print(f"Reviews Extracted by AI: {result.review_count}")
            print("-"*55)
            if result.reviews:
                for i, review in enumerate(result.reviews[:5]): # Print first 5
                    print(f"\n--- Review {i+1} ---")
                    print(f"  Author: {review.author}")
                    print(f"  Rating: {review.rating_text}")
                    print(f"  Date: {review.date}")
                    print(f"  Text: {review.review_text}")
                if result.review_count > 5:
                    print("\n[... and potentially more ...]")
            else:
                print("\nNo reviews were successfully extracted by the AI.")
                print("(Check Bright Data Zone/Token, Target URL validity, and AI model capability/prompt)")
            print("="*55)
        else:
            print("\n" + "="*20 + " Scraping Failed " + "="*20)
            print("The scraping and parsing process did not complete successfully.")
            print("Check logs for errors related to Bright Data API or AI Parsing.")
            print("="*55)

    logger.info("--- Google Play Review Scraping Tool Finished ---")

# --- Run the main async function ---
if __name__ == "__main__":
    asyncio.run(main())