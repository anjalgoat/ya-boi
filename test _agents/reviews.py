import os
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, AnyHttpUrl
from typing import List, Optional, Dict, Any
from apify_client import ApifyClient

# --- Configuration ---

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Apify Configuration (Using user-provided Actor names)
APP_STORE_ACTOR_ID = "agents~appstore-reviews"  # User-provided Actor ID/Name
GOOGLE_PLAY_ACTOR_ID = "agents~googleplay-reviews" # User-provided Actor ID/Name
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

# Check for Apify Token
if not APIFY_API_TOKEN:
    logger.error("APIFY_API_TOKEN environment variable is required but not set.")
    apify_client = None # Indicate client is not available
else:
    # Initialize Apify Client
    apify_client = ApifyClient(APIFY_API_TOKEN)

# --- Pydantic Models ---

# Input Model (expects data similar to CompetitorResponse output)
class CompetitorInput(BaseModel):
    """Represents a single competitor's data as input."""
    name: str = Field(..., description="Name of the competitor")
    app_store_url: Optional[AnyHttpUrl] = Field(None, description="Validated App Store URL (optional)")
    google_play_url: Optional[AnyHttpUrl] = Field(None, description="Validated Google Play URL (optional)")

class CompetitorDataInput(BaseModel):
    """Input structure for the reviews agent."""
    query: str = Field(..., description="The user's original query")
    competitors: List[CompetitorInput] = Field(..., description="List of competitors with potential store URLs")

# Output Models
class ReviewResult(BaseModel):
    """Holds the review results for a single competitor."""
    name: str = Field(..., description="Name of the competitor")
    app_store_url_used: Optional[str] = Field(None, description="App Store URL passed to Apify or status")
    google_play_url_used: Optional[str] = Field(None, description="Google Play URL passed to Apify or status")
    app_store_status: str = Field(..., description="Status of App Store fetch (success, failed, skipped)")
    google_play_status: str = Field(..., description="Status of Google Play fetch (success, failed, skipped)")
    app_store_reviews: List[Dict[str, Any]] = Field(default_factory=list, description="List of reviews from App Store")
    google_play_reviews: List[Dict[str, Any]] = Field(default_factory=list, description="List of reviews from Google Play")

class ReviewResponse(BaseModel):
    """Final output structure containing reviews for all competitors."""
    query: str = Field(..., description="The user's original query")
    competitor_reviews: List[ReviewResult] = Field(..., description="List of review results for each competitor")

# --- Apify Interaction Functions ---

def fetch_reviews_from_store(actor_id: str, store_url: Optional[AnyHttpUrl], store_name: str, run_input_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetches reviews from a specific app store using an Apify actor.

    Args:
        actor_id: The ID or name of the Apify actor to use.
        store_url: The validated URL for the app store listing.
        store_name: Name of the store (e.g., "App Store", "Google Play") for logging.
        run_input_params: Base parameters for the actor run input.

    Returns:
        A dictionary containing 'status' and 'reviews'.
    """
    if not apify_client:
        logger.error(f"Apify client not initialized. Skipping {store_name} fetch.")
        return {"status": "failed", "reviews": []}

    if not store_url:
        logger.info(f"{store_name} URL is missing. Skipping review fetch.")
        return {"status": "skipped", "reviews": []}

    # Prepare the specific input for this run
    run_input = {**run_input_params, "startUrls": [{"url": str(store_url)}]} # Ensure URL is string

    logger.info(f"Calling Apify actor '{actor_id}' for {store_name} URL: {store_url} with input: {run_input}") # Log input
    try:
        # Use .call() instead of .call_sync()
        actor_call = apify_client.actor(actor_id).call(run_input=run_input)

        # Check if the actor run itself provided dataset items
        reviews = actor_call.get("items", []) if isinstance(actor_call, dict) else []
        if not isinstance(reviews, list):
             logger.warning(f"Apify actor '{actor_id}' for {store_name} returned unexpected data format. Expected list under 'items'. Got: {type(reviews)}")
             reviews = []

        # Check if reviews were actually fetched
        if reviews:
            logger.info(f"Successfully fetched {len(reviews)} reviews from {store_name} using actor '{actor_id}'.")
            return {"status": "success", "reviews": reviews}
        else:
            # Check if the call result indicates success even without items
            run_status = actor_call.get("status") if isinstance(actor_call, dict) else None
            if run_status == 'SUCCEEDED':
                 logger.info(f"Apify actor '{actor_id}' for {store_name} ran successfully but found no reviews.")
                 return {"status": "success", "reviews": []}
            else:
                 # Log the actual response if it failed and didn't return items
                 logger.warning(f"Apify actor '{actor_id}' for {store_name} did not return items or failed. Run details: {actor_call}")
                 return {"status": "failed", "reviews": []}

    except Exception as e:
        logger.error(f"An error occurred during Apify call for {store_name} URL {store_url}: {e}", exc_info=True)
        return {"status": "failed", "reviews": []}

# --- Main Agent Logic ---

def get_reviews_for_competitors(input_data: CompetitorDataInput) -> ReviewResponse:
    """
    Orchestrates fetching reviews for a list of competitors.

    Args:
        input_data: CompetitorDataInput object containing query and competitor list.

    Returns:
        ReviewResponse object containing the results.
    """
    all_review_results = []

    for competitor in input_data.competitors:
        logger.info(f"Processing competitor: {competitor.name}")

        # --- Fetch App Store Reviews ---
        app_store_result = fetch_reviews_from_store(
            actor_id=APP_STORE_ACTOR_ID,
            store_url=competitor.app_store_url,
            store_name="App Store",
            run_input_params={ # Define base parameters for App Store actor
                "maxReviews": 5, # <-- Changed to 5
                "country": "us"
            }
        )

        # --- Fetch Google Play Reviews ---
        google_play_result = fetch_reviews_from_store(
            actor_id=GOOGLE_PLAY_ACTOR_ID,
            store_url=competitor.google_play_url,
            store_name="Google Play",
            run_input_params={ # Define base parameters for Google Play actor
                "maxReviews": 5, # <-- Changed to 5
                "language": "en"
            }
        )

        # --- Compile Result for this Competitor ---
        competitor_result = ReviewResult(
            name=competitor.name,
            app_store_url_used=str(competitor.app_store_url) if competitor.app_store_url else "skipped_missing_url",
            google_play_url_used=str(competitor.google_play_url) if competitor.google_play_url else "skipped_missing_url",
            app_store_status=app_store_result["status"],
            google_play_status=google_play_result["status"],
            app_store_reviews=app_store_result["reviews"],
            google_play_reviews=google_play_result["reviews"]
        )
        all_review_results.append(competitor_result)

    # --- Final Response ---
    final_response = ReviewResponse(
        query=input_data.query,
        competitor_reviews=all_review_results
    )

    logger.info("Finished processing all competitors.")
    return final_response

# --- Example Usage ---

if __name__ == "__main__":

    # Example input data - Modified to include only ONE competitor for testing
    sample_input_dict = {
        "query": "best music app",
        "competitors": [
            { # <-- Only processing the first competitor
                "name": "Spotify",
                "app_store_url": "https://apps.apple.com/us/app/spotify-music-and-podcasts/id324684580",
                "google_play_url": "https://play.google.com/store/apps/details?id=com.spotify.music" # Using a potentially more valid URL
            }
        ]
    }

    # Example demonstrating invalid URL input handling by Pydantic (kept for reference)
    sample_input_invalid_url = {
         "query": "test invalid url",
         "competitors": [
              {
                "name": "Test Invalid",
                "app_store_url": "this-is-not-a-valid-url", # Invalid URL string
                "google_play_url": "https://play.google.com/store/apps/details?id=test.app"
              }
         ]
    }


    logger.info("--- Running Example with Modified Valid Input (Single Competitor, Max 5 Reviews) ---")
    try:
        # Validate input using the Pydantic model
        validated_input = CompetitorDataInput(**sample_input_dict)
        logger.info("Input data validated successfully.")

        # Run the reviews agent
        if apify_client: # Only run if client was initialized
             review_data = get_reviews_for_competitors(validated_input)

             # Print the results (using model_dump for clean JSON-like output)
             import json
             print("\n--- Review Agent Output ---")
             print(json.dumps(review_data.model_dump(), indent=2))
             print("--- End of Output ---")
        else:
            logger.error("Cannot run reviews agent because Apify client failed to initialize (check APIFY_API_TOKEN).")

    except ValidationError as e:
        logger.error(f"Input data failed validation: {e}")
    except Exception as e:
         logger.error(f"An unexpected error occurred during example execution: {e}", exc_info=True)