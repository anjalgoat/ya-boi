# diagram_agent.py
import asyncio
import json
import logging
import os
import sys  # Import sys for stdin
from typing import List, Optional, Dict, Any
import math

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, HttpUrl
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Load environment variables
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Validate environment variables
if not openrouter_api_key:
    # Log error to stderr if key is missing, but allow script to potentially run if called unexpectedly
    logging.error("CRITICAL: diagram_agent.py requires OPENROUTER_API_KEY environment variable.")
    # Avoid raising ValueError here so workflow.py can capture the failure if called via subprocess

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)] # Log to stderr to keep stdout clean for JSON
)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_NAME = os.getenv("OPENAI_MODEL", "openai/gpt-4o")
OPENROUTER_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

# --- Pydantic Input Models (Representing data received from workflow.py) ---
# These should match the structures defined in workflow.py's transformation helpers

class TrendItem(BaseModel):
    description: str
    source: str
    metric: Optional[str] = None
    timestamp: Optional[str] = None

class TrendsInput(BaseModel):
    trends: List[TrendItem]

class WebpageResult(BaseModel):
    url: HttpUrl # Expecting HttpUrl from workflow's transformation
    title: str
    summary: str
    insight: str
    relevance: str

class WebpageInsightsInput(BaseModel):
    results: List[WebpageResult]

class ReviewItem(BaseModel):
    rating: Optional[int] = None
    text: str

class CompetitorReviews(BaseModel):
    competitor_name: str
    app_store_reviews: Optional[List[ReviewItem]] = Field(default_factory=list)
    google_play_reviews: Optional[List[ReviewItem]] = Field(default_factory=list)

class RawMarketDataInput(BaseModel):
    """ Represents the complete raw-like input structure received via stdin """
    reviews: Optional[List[CompetitorReviews]] = None
    trends: Optional[TrendsInput] = None
    webpage_insights: Optional[WebpageInsightsInput] = None

# --- Pydantic Output Models (Define the desired JSON structure for charts) ---
class BarChartDataItem(BaseModel):
    name: str = Field(..., description="Competitor name")
    review_count: int = Field(..., description="Total number of reviews analyzed for this competitor")
    rating: Optional[float] = Field(None, description="Average rating calculated from reviews (null if no ratings available)")
    market_share: Optional[float] = Field(None, description="Estimated market share (if inferrable, otherwise null)")

class GapMatrixDataItem(BaseModel):
    feature: str = Field(..., description="Feature name identified from inputs")
    unmet_need: str = Field(..., description="Level of unmet need ('High', 'Medium', 'Low') based on analysis")
    competitor_status: Dict[str, str] = Field(..., description="Dictionary mapping competitor names to feature support status ('Yes', 'No', 'Unknown').")

class ChartDataResponse(BaseModel): # This is what the script should output as JSON
    bar_chart_data: List[BarChartDataItem] = Field(default_factory=list, description="Data for competitor comparison bar chart.")
    gap_matrix_data: List[GapMatrixDataItem] = Field(default_factory=list, description="Data for feature gap analysis matrix.")
    suggested_bar_chart_metric: Optional[str] = Field(None, description="Suggested metric for bar chart Y-axis (e.g., 'review_count' or 'rating').")


# --- Agent Setup ---
# Only initialize if API key is present
diagram_agent = None
if openrouter_api_key:
    try:
        diagram_agent_model = OpenAIModel(
            model_name=MODEL_NAME,
            api_key=openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
        )
        # System Prompt as previously defined
        diagram_agent_system_prompt = """
You are an AI agent specializing in analyzing market research data (reviews, trends, web insights) for betting apps
and structuring the findings into JSON format suitable for frontend charting libraries.

You will receive raw-like input data containing competitor reviews, market trends, and webpage insights.
Your task is to **ANALYZE and AGGREGATE** this data to generate the structured JSON output.

**Instructions:**

1.  **Identify Competitors:** List all unique competitors mentioned in the `reviews` section.
2.  **Analyze Competitor Reviews:** For each unique competitor found in the `reviews` data:
    * Calculate the total `review_count` (sum of app_store_reviews and google_play_reviews).
    * Calculate the average `rating`. Sum all available ratings and divide by the number of reviews *with* ratings. If no ratings are available, set rating to null. Handle potential division by zero. Avoid NaN values, use null instead.
    * Attempt to infer `market_share` only if explicitly mentioned or strongly implied in trends/web insights for that competitor; otherwise, set it to null.
    * Compile this information into `BarChartDataItem` objects for the `bar_chart_data` list.
3.  **Identify Key Features:** Analyze all input sections (`reviews` text, `trends` descriptions, `webpage_insights` summaries/insights) to identify key features relevant to betting apps (e.g., "Live betting", "Cash-out", "Parlay options", "Live streaming", "Esports betting", "UI/UX", "Payout speed", "App stability/crashes", "Customer support").
4.  **Analyze Feature Gaps & Competitor Support:** For each identified key feature:
    * Determine the `unmet_need` level ('High', 'Medium', 'Low'). Consider negative sentiment in reviews mentioning the feature, lack of the feature mentioned as a con, or explicit statements in insights/trends. 'High' indicates significant user complaints or a clear market gap. 'Low' indicates general satisfaction or wide availability.
    * For each unique competitor, determine their support status ('Yes', 'No', 'Unknown').
        * 'Yes': Strong positive mentions of the feature in reviews OR explicit mention of support in trends/insights.
        * 'No': Explicit complaints about the feature *missing* OR insights stating they lack it.
        * 'Unknown': Insufficient information to determine support.
    * Create a `competitor_status` dictionary mapping each competitor name to their 'Yes'/'No'/'Unknown' status for this feature.
    * Compile this into `GapMatrixDataItem` objects for the `gap_matrix_data` list.
5.  **Suggest Bar Chart Metric:** Based on the calculated competitor data, suggest the most informative metric for the bar chart's primary axis (`suggested_bar_chart_metric`). Prioritize 'review_count' if the counts are substantial and varied, otherwise suggest 'rating'. If unsure, set to null.
6.  **Format Output:** Structure the results *strictly* according to the `ChartDataResponse` JSON schema, using the exact field names (`bar_chart_data`, `gap_matrix_data`, `suggested_bar_chart_metric`). Handle potential NaN/Infinity values by outputting null.

**IMPORTANT:** Your sole output must be a single, valid JSON object conforming to the `ChartDataResponse` schema. Do NOT include explanations, apologies, or any text outside the JSON structure. Do NOT generate Python or any other code.
"""
        diagram_agent = Agent(
            diagram_agent_model,
            system_prompt=diagram_agent_system_prompt,
            result_type=ChartDataResponse,
            result_tool_name="analyze_and_format_chart_data",
            result_tool_description="Analyzes raw market data and formats it into JSON suitable for frontend charts.",
            result_retries=2,
        )
    except Exception as e:
         logger.error(f"Failed to initialize Pydantic AI Agent in diagram_agent.py: {e}", exc_info=True)
         diagram_agent = None # Ensure agent is None if init fails
else:
    logger.error("Diagram Agent cannot be initialized because OPENROUTER_API_KEY is missing.")


# --- Main Execution ---
async def main():
    """Main function to parse input from stdin, run the agent, and generate chart data JSON to stdout."""
    logger.info("Starting diagram_agent.py execution")
    input_json_string_from_stdin = ""
    final_output_json = "" # Variable to hold the JSON string to be printed

    # Define a fallback error response structure
    def create_error_response(message: str) -> str:
         error_resp = ChartDataResponse(
             bar_chart_data=[],
             gap_matrix_data=[],
             suggested_bar_chart_metric=f"Error: {message}"
         )
         return error_resp.model_dump_json(indent=2, exclude_none=True)

    # 1. Read and Parse Input from Stdin
    try:
        input_json_string_from_stdin = sys.stdin.read()
        if not input_json_string_from_stdin.strip():
            logger.error("Diagram Agent: No input received from stdin.")
            final_output_json = create_error_response("No input received from stdin.")
            print(final_output_json)
            return

        raw_market_data = RawMarketDataInput.model_validate_json(input_json_string_from_stdin)
        logger.info("Diagram Agent: Successfully parsed and validated input JSON from stdin.")

    except Exception as e:
        logger.error(f"Diagram Agent: Error parsing input JSON from stdin: {e}", exc_info=True)
        final_output_json = create_error_response(f"Invalid input JSON - {e}")
        print(final_output_json)
        return

    # 2. Check if Agent was Initialized
    if not diagram_agent:
        logger.error("Diagram Agent: Agent was not initialized (likely missing API key). Cannot proceed.")
        final_output_json = create_error_response("Agent not initialized (API key missing?)")
        print(final_output_json)
        return

    # 3. Construct Prompt for the Agent using model_dump_json for nested models
    try:
        prompt_str = "Analyze the following market data and generate the ChartDataResponse JSON:\n\n"
        # Use model_dump_json() for nested Pydantic models to handle types like HttpUrl
        if raw_market_data.reviews:
            prompt_str += "--- Reviews Data ---\n"
            # reviews is List[CompetitorReviews], dump each model or dump the list itself if the agent handles lists of objects
            # Assuming agent can handle list of dicts:
            prompt_str += json.dumps([r.model_dump(mode='json') for r in raw_market_data.reviews], indent=2)
            # If you need to dump the whole Pydantic list object, you'd need a wrapper model or custom handling.
            prompt_str += "\n\n"

        if raw_market_data.trends:
            prompt_str += "--- Trends Data ---\n"
            prompt_str += raw_market_data.trends.model_dump_json(indent=2, exclude_none=True)
            prompt_str += "\n\n"

        if raw_market_data.webpage_insights:
            prompt_str += "--- Webpage Insights Data ---\n"
            prompt_str += raw_market_data.webpage_insights.model_dump_json(indent=2, exclude_none=True)
            prompt_str += "\n\n"

        prompt_str += "--- End of Data ---\n"
        prompt_str += "Now, generate the JSON output according to the instructions."
        logger.debug(f"Diagram Agent: Prompt constructed (length: {len(prompt_str)})")

    except Exception as e_prompt:
        logger.error(f"Diagram Agent: Error constructing prompt string: {e_prompt}", exc_info=True)
        final_output_json = create_error_response(f"Error building agent prompt - {e_prompt}")
        print(final_output_json)
        return

    # 4. Run the Pydantic AI Agent
    logger.info("Diagram Agent: Running Pydantic AI Agent to analyze data...")
    try:
        response_container = await diagram_agent.run(prompt_str)

        if response_container and isinstance(response_container.data, ChartDataResponse):
            logger.info("Diagram Agent: Agent execution successful.")
            # Ensure final output is clean JSON
            final_output_json = response_container.data.model_dump_json(indent=2, exclude_none=True)
        else:
            error_details = getattr(response_container, 'error', 'Unknown agent error')
            data_type = type(getattr(response_container, 'data', None))
            logger.error(f"Diagram Agent: Agent did not return expected ChartDataResponse. Type: {data_type}. Error: {error_details}")
            final_output_json = create_error_response(f"Agent failed or returned unexpected data type: {data_type}")

    except Exception as e_agent:
        logger.error(f"Diagram Agent: An error occurred during agent execution: {e_agent}", exc_info=True)
        final_output_json = create_error_response(f"Agent execution exception - {e_agent}")

    # 5. Print final JSON output to stdout
    print(final_output_json)
    logger.info("Diagram Agent: Finished execution.")


if __name__ == "__main__":
    asyncio.run(main())