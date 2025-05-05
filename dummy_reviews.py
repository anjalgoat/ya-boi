import os
import sys # Import sys to read from stdin
import json # Import json to parse input and dump output
import asyncio
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from typing import List, Optional, Dict

# --- Existing Setup (Logging, Env Vars, OpenAIModel) ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # Use stderr for logs to keep stdout clean for JSON output
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Environment variable checks and Model setup
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

_model_name = os.getenv("OPENAI_MODEL", "mistralai/mistral-7b-instruct") # Default model if not set
_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1") # Default base URL

if not openrouter_api_key:
    # Log error to stderr and exit if critical env var is missing
    logger.critical("CRITICAL: OPENROUTER_API_KEY environment variable is required.")
    sys.exit(1) # Exit with error code

_model = OpenAIModel(
    model_name=_model_name,
    api_key=openrouter_api_key,
    base_url=_base_url
)
logger.info(f"Using LLM: {_model_name} via {_base_url}")

# --- Pydantic Models ---

# Input Models (Based on the expected stdin JSON)
class Competitor(BaseModel):
    name: str = Field(..., description="Name of the competitor")
    app_store_url: Optional[str] = Field(None, description="App Store URL (for apps)")
    google_play_url: Optional[str] = Field(None, description="Google Play URL (for apps)")

class CompetitorResponse(BaseModel):
    query: str = Field(..., description="The user's original query")
    competitors: List[Competitor] = Field(..., description="List of competitors")

# Output Models (For the generated reviews and the final JSON output)
class Review(BaseModel):
    """Represents a single fake app review."""
    rating: int = Field(..., description="Star rating from 1 to 5", ge=1, le=5)
    text: str = Field(..., description="The text content of the review.")

class AppReviewData(BaseModel):
    """Holds the generated fake reviews for a single app."""
    competitor_name: str = Field(..., description="Name of the app competitor")
    app_store_reviews: List[Review] = Field(default_factory=list, description="List of fake App Store reviews")
    google_play_reviews: List[Review] = Field(default_factory=list, description="List of fake Google Play reviews")

# --- Review Extractor Agent Definition ---

# System prompt for the review generation agent
review_agent_system_prompt = (
    "You are an AI assistant specialized in generating realistic fake app reviews based on an app's name and target platforms. "
    "Given an app name (e.g., 'DraftKings') and the platforms it's available on (e.g., 'App Store and Google Play'), "
    "generate exactly 3 distinct, plausible-sounding fake reviews for EACH specified platform. "
    "Each review must include:\n"
    "1. A 'rating' (integer between 1 and 5).\n"
    "2. 'text' (1-3 sentences long, varied in tone - positive, negative, mentioning features, bugs, etc.).\n"
    "Structure your entire response strictly as an AppReviewData JSON object, containing 'competitor_name', 'app_store_reviews' (list), and 'google_play_reviews' (list)."
    " Ensure the competitor_name matches the input app name."
)

# Create the review extractor agent
# Expects data for a single app as output
review_extractor_agent = Agent(
    _model,
    system_prompt=review_agent_system_prompt,
    result_type=AppReviewData, # Expects structured data for one app
    result_tool_name="generated_reviews",
    result_tool_description="Formats the generated fake reviews for the specified app",
    result_retries=2, # Retry if generation fails validation
)

# --- Function to Run Review Extraction for a Single Competitor ---

async def generate_fake_reviews_for_app(competitor: Competitor) -> Optional[AppReviewData]:
    """
    Uses the ReviewExtractorAgent to generate fake reviews for a given competitor app.
    Returns an AppReviewData object on success, None on failure or if no URLs provided.
    """
    logger.info(f"Processing competitor: {competitor.name}")

    # Determine which platforms to generate reviews for based on URL presence
    platforms = []
    if competitor.app_store_url:
        platforms.append("App Store")
    if competitor.google_play_url:
        platforms.append("Google Play")

    if not platforms:
        logger.warning(f"No App Store or Google Play URL found for '{competitor.name}'. Skipping review generation.")
        return None

    # Construct the prompt for the review agent
    platform_string = " and ".join(platforms)
    prompt = (
        f"Generate exactly 3 fake reviews for the app named '{competitor.name}'. "
        f"Provide reviews for the following platform(s): {platform_string}. "
        f"Output the result strictly as an AppReviewData JSON object."
    )
    logger.debug(f"Review generation prompt for '{competitor.name}':\n{prompt}")

    try:
        # Run the agent
        response = await review_extractor_agent.run(prompt)

        # --- Result Extraction and Validation ---
        # Access the validated Pydantic object from the agent's response
        if isinstance(response, AppReviewData):
             result_data = response
        elif hasattr(response, 'data') and isinstance(response.data, AppReviewData):
             result_data = response.data
        else:
             logger.error(f"Agent for '{competitor.name}' returned unexpected structure: {type(response)}. Skipping.")
             return None # Return None if the agent failed structurally

        # Override the name just in case the LLM hallucinates it, using the known input name
        result_data.competitor_name = competitor.name

        # Clean up reviews for platforms that weren't requested (safety check)
        if "App Store" not in platforms:
            logger.debug(f"Removing App Store reviews for {competitor.name} as URL was not provided.")
            result_data.app_store_reviews = []
        if "Google Play" not in platforms:
            logger.debug(f"Removing Google Play reviews for {competitor.name} as URL was not provided.")
            result_data.google_play_reviews = []

        # Validate that we got the expected number of reviews per requested platform
        if "App Store" in platforms and len(result_data.app_store_reviews) != 3:
             logger.warning(f"Expected 3 App Store reviews for {competitor.name}, but got {len(result_data.app_store_reviews)}.")
        if "Google Play" in platforms and len(result_data.google_play_reviews) != 3:
             logger.warning(f"Expected 3 Google Play reviews for {competitor.name}, but got {len(result_data.google_play_reviews)}.")


        logger.info(f"Successfully generated review data for '{competitor.name}'")
        return result_data

    except ModelRetry as e:
        logger.error(f"Agent failed validation/retries for '{competitor.name}': {e}", exc_info=True)
        return None # Return None if agent fails after retries
    except Exception as e:
        logger.error(f"An unexpected error occurred during review generation for '{competitor.name}': {e}", exc_info=True)
        return None # Return None on other unexpected errors


# --- Main Execution Logic ---
async def main():
    """
    Reads competitor data from stdin, generates fake reviews, and prints results as JSON to stdout.
    """
    logger.info("Review Generation Agent started. Reading competitor data from stdin...")
    output_json = [] # Initialize empty list for the final JSON output

    # --- Phase 1: Read and Parse Competitor Data from Stdin ---
    input_data: Optional[CompetitorResponse] = None
    competitors_to_process: List[Competitor] = []
    try:
        input_json_str = sys.stdin.read()
        if not input_json_str:
            logger.error("No input received from stdin.")
            # Output empty list as JSON and exit if no input
            print("[]")
            return

        # Validate JSON structure using Pydantic model
        input_data = CompetitorResponse.model_validate_json(input_json_str)
        competitors_to_process = input_data.competitors
        logger.info(f"Successfully parsed input for query: '{input_data.query}'. Found {len(competitors_to_process)} competitors.")

    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from stdin.")
        output_json = {"error": "Invalid JSON input"} # Prepare error output
    except ValidationError as e:
        logger.error(f"Input JSON does not match expected CompetitorResponse schema: {e}")
        output_json = {"error": "Input validation failed", "details": e.errors()} # Prepare error output
    except Exception as e:
        logger.error(f"An unexpected error occurred during input processing: {e}", exc_info=True)
        output_json = {"error": "Unexpected error processing input"} # Prepare error output

    # If input parsing failed, print error JSON and exit
    if isinstance(output_json, dict) and "error" in output_json:
         print(json.dumps(output_json, indent=2))
         return

    # --- Phase 2: Generate Fake Reviews for Each Competitor Concurrently ---
    all_review_data: List[AppReviewData] = []
    if competitors_to_process:
        logger.info(f"Generating fake reviews for {len(competitors_to_process)} competitors concurrently...")

        # Create tasks for concurrent review generation
        tasks = [generate_fake_reviews_for_app(comp) for comp in competitors_to_process]
        # Use return_exceptions=True to capture errors from individual tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect valid results and log errors/skipped competitors
        for i, result in enumerate(results):
            competitor_name = competitors_to_process[i].name # Get name for logging
            if isinstance(result, AppReviewData):
                all_review_data.append(result) # Add successful result
            elif isinstance(result, Exception):
                # Log exception captured by return_exceptions=True
                logger.error(f"Task for '{competitor_name}' failed with exception: {result}", exc_info=result)
            elif result is None:
                 # Log cases where generate_fake_reviews_for_app returned None (e.g., no URLs, agent failure)
                 logger.warning(f"Review generation skipped or failed for '{competitor_name}' (task returned None).")
            else:
                 # Log unexpected return types from asyncio.gather tasks
                 logger.error(f"Unexpected result type received for '{competitor_name}': {type(result)}")


        logger.info(f"Finished review generation. Processed {len(competitors_to_process)} competitors, successfully generated data for {len(all_review_data)}.")
    else:
        logger.info("No competitors found in the input to process.")

    # --- Phase 3: Prepare and Output Final JSON ---
    # Convert the list of AppReviewData Pydantic objects to a list of Python dictionaries
    # This list directly matches the desired output structure: [ { "competitor_name": ..., "reviews": ... }, ... ]
    output_list = [review_data.model_dump(exclude_none=True) for review_data in all_review_data] # exclude_none for cleaner output

    # Convert the list of dictionaries to a JSON formatted string
    # indent=2 makes it pretty-printed like the example output
    output_json_str = json.dumps(output_list, indent=2)

    # Print the final JSON string to standard output
    print(output_json_str)
    logger.info("Successfully wrote final JSON output to stdout.")


if __name__ == "__main__":
    # Ensure the main async function is run
    asyncio.run(main())