import asyncio
import json
import logging
import os
from typing import List, Optional, Dict, Any
import math # Import math for potential calculations like NaN checks

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, HttpUrl # Added HttpUrl, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Load environment variables
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Validate environment variables
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is required.")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Using a more capable model is highly recommended for this complex analysis task
MODEL_NAME = os.getenv("OPENAI_MODEL", "openai/gpt-4o")
OPENROUTER_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

# --- Pydantic Input Models (Adapted from previous script) ---
# These models represent the RAW-LIKE input structure provided

class TrendItem(BaseModel):
    description: str
    source: str
    metric: Optional[str] = None
    timestamp: Optional[str] = None

class TrendsInput(BaseModel):
    trends: List[TrendItem]

class WebpageResult(BaseModel):
    url: HttpUrl
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
    """ Represents the complete raw-like input structure for chart data generation """
    reviews: Optional[List[CompetitorReviews]] = None
    trends: Optional[TrendsInput] = None
    webpage_insights: Optional[WebpageInsightsInput] = None


# --- Pydantic Output Models (Define the desired JSON structure for charts) ---
class BarChartDataItem(BaseModel):
    name: str = Field(..., description="Competitor name")
    review_count: int = Field(..., description="Total number of reviews analyzed for this competitor")
    # Allow float for rating, handle potential division by zero later if needed
    rating: Optional[float] = Field(None, description="Average rating calculated from reviews (null if no ratings available)")
    # Market share is unlikely to be derivable purely from this input, making it optional
    market_share: Optional[float] = Field(None, description="Estimated market share (if inferrable, otherwise null)")

class GapMatrixDataItem(BaseModel):
    feature: str = Field(..., description="Feature name identified from inputs")
    unmet_need: str = Field(..., description="Level of unmet need ('High', 'Medium', 'Low') based on analysis")
    # Competitor status dictionary: Key = competitor name, Value = 'Yes'/'No'/'Unknown'
    competitor_status: Dict[str, str] = Field(..., description="Dictionary mapping competitor names to feature support status ('Yes', 'No', 'Unknown').")

class ChartDataResponse(BaseModel):
    bar_chart_data: List[BarChartDataItem] = Field(..., description="Data for competitor comparison bar chart.")
    gap_matrix_data: List[GapMatrixDataItem] = Field(..., description="Data for feature gap analysis matrix.")
    suggested_bar_chart_metric: Optional[str] = Field(None, description="Suggested metric for bar chart Y-axis (e.g., 'review_count' or 'rating').")


# --- Agent Setup ---
diagram_agent_model = OpenAIModel(
    model_name=MODEL_NAME,
    api_key=openrouter_api_key,
    base_url=OPENROUTER_BASE_URL,
)

# --- UPDATED System Prompt for Analysis and JSON Output ---
diagram_agent_system_prompt = """
You are an AI agent specializing in analyzing market research data (reviews, trends, web insights) for betting apps
and structuring the findings into JSON format suitable for frontend charting libraries.

You will receive raw-like input data containing competitor reviews, market trends, and webpage insights.
Your task is to **ANALYZE and AGGREGATE** this data to generate the structured JSON output.

**Instructions:**

1.  **Identify Competitors:** List all unique competitors mentioned in the `reviews` section.
2.  **Analyze Competitor Reviews:** For each unique competitor found in the `reviews` data:
    * Calculate the total `review_count` (sum of app_store_reviews and google_play_reviews).
    * Calculate the average `rating`. Sum all available ratings and divide by the number of reviews *with* ratings. If no ratings are available, set rating to null. Handle potential division by zero.
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
6.  **Format Output:** Structure the results *strictly* according to the `ChartDataResponse` JSON schema, using the exact field names (`bar_chart_data`, `gap_matrix_data`, `suggested_bar_chart_metric`).

**IMPORTANT:** Your sole output must be a single, valid JSON object conforming to the `ChartDataResponse` schema. Do NOT include explanations, apologies, or any text outside the JSON structure. Do NOT generate Python or any other code.
"""

# Initialize the Agent
diagram_agent = Agent(
    diagram_agent_model,
    system_prompt=diagram_agent_system_prompt,
    result_type=ChartDataResponse, # Expect the chart data JSON structure
    result_tool_name="analyze_and_format_chart_data",
    result_tool_description="Analyzes raw market data and formats it into JSON suitable for frontend charts.",
    result_retries=2, # Analysis can be complex, allow retries
)

# --- Removed Helper Functions: create_competitor_data, create_market_gap_data ---

# --- Main Execution ---
async def main():
    """Main function to parse input, run the agent, and generate chart data JSON."""
    logger.info("Starting diagram agent execution (Analysis & JSON Output Mode)")

    # --- Example Input JSON (as a string, matching the user's format) ---
    input_json_string = """
{
  "reviews": [
    {
      "competitor_name": "DraftKings",
      "app_store_reviews": [
        {
          "rating": 5,
          "text": "Amazing app! Live betting is seamless and the odds are great."
        },
        {
          "rating": 3,
          "text": "Good app but crashes sometimes during peak events."
        }
      ],
      "google_play_reviews": [
        {
          "rating": 4,
          "text": "Solid betting app, fast payouts!"
        },
        {
          "rating": null,
          "text": "Needs better customer support response times."
        }
      ]
    },
    {
      "competitor_name": "FanDuel",
      "app_store_reviews": [
        {
          "rating": 4,
          "text": "Great for live streaming games, but the app can lag."
        },
        {
          "rating": 5,
          "text": "User interface is very intuitive and easy to navigate."
        }
      ],
      "google_play_reviews": [
        {
          "rating": 5,
          "text": "Love the parlay options!"
        }
      ]
    }
  ],
  "trends": {
    "trends": [
      {
        "description": "DraftKings is a popular app with features: Live betting and cash-out options",
        "source": "App Store",
        "metric": "4.6 rating",
        "timestamp": "2025-04-18T12:00:00Z"
      },
       {
        "description": "Industry trend shows increasing demand for comprehensive parlay builders.",
        "source": "Market Report",
        "metric": null,
        "timestamp": "2025-04-15T10:00:00Z"
      }
    ]
  },
  "webpage_insights": {
    "results": [
      {
        "url": "https://www.grandviewresearch.com/industry-analysis/sports-betting-market-report",
        "title": "Sports Betting Market Size & Share Analysis Report, 2030",
        "summary": "The report forecasts significant growth in the sports betting market. In-play betting and esports betting are trending. Mobile UX is key.",
        "insight": "Focus on in-play and esports betting to capture emerging market segments.",
        "relevance": "Highly relevant - offers market trends and competitive insights."
      },
      {
        "url": "https://www.examplecomparison.com/betting-apps",
        "title": "Betting App Comparison 2025",
        "summary": "Compared top apps. FanDuel leads in UI, DraftKings offers wide markets. BetMGM strong in loyalty programs. Stability issues noted for DraftKings during high traffic.",
        "insight": "App stability under load is a key differentiator and pain point.",
        "relevance": "Relevant - provides direct competitor comparison points."
      }
    ]
  }
}
"""

    # --- Parse Input JSON and Validate with Pydantic ---
    try:
        input_dict = json.loads(input_json_string)
        # Validate the structure using the top-level input model
        raw_market_data = RawMarketDataInput(**input_dict)
        logger.info("Successfully parsed and validated input JSON.")
    except json.JSONDecodeError:
        logger.error("Invalid JSON input provided.")
        print(json.dumps({"error": "Invalid JSON input format."}, indent=2))
        return # Exit if input is invalid
    except ValidationError as e:
        logger.error(f"Input data validation error: {e}")
        print(json.dumps({"error": f"Input data validation failed: {e}"}, indent=2))
        return # Exit if input is invalid
    except Exception as e:
        logger.error(f"Error processing input JSON: {e}", exc_info=True)
        print(json.dumps({"error": f"Error processing input: {e}"}, indent=2))
        return # Exit on other processing errors

    # --- Prepare Input for the Agent ---
    # The agent's prompt is designed to handle the raw structure.
    # We can pass the Pydantic model directly or format it as a string/dict.
    # Passing the dictionary representation is often robust.
    # Let's construct a detailed prompt string as the system prompt gives detailed instructions based on sections.

    prompt_str = "Analyze the following market data and generate the ChartDataResponse JSON:\n\n"

    if raw_market_data.reviews:
        prompt_str += "--- Reviews Data ---\n"
        prompt_str += json.dumps([r.model_dump() for r in raw_market_data.reviews], indent=2)
        prompt_str += "\n\n"

    if raw_market_data.trends:
        prompt_str += "--- Trends Data ---\n"
        prompt_str += json.dumps(raw_market_data.trends.model_dump(), indent=2)
        prompt_str += "\n\n"

    if raw_market_data.webpage_insights:
        prompt_str += "--- Webpage Insights Data ---\n"
        prompt_str += json.dumps(raw_market_data.webpage_insights.model_dump(), indent=2)
        prompt_str += "\n\n"

    prompt_str += "--- End of Data ---\n"
    prompt_str += "Now, generate the JSON output according to the instructions."


    # --- Run the agent ---
    logger.info("Running Pydantic AI Agent to analyze data and generate chart JSON...")
    try:
        # Pass the constructed prompt string
        response_container = await diagram_agent.run(prompt_str)

        # --- Print the generated JSON data ---
        if response_container and isinstance(response_container.data, ChartDataResponse):
            logger.info("Agent execution successful. Generated Chart Data (JSON):")
            # Use Pydantic's built-in JSON serialization for clean output
            print(response_container.data.model_dump_json(indent=2))

            if response_container.data.suggested_bar_chart_metric:
                logger.info(f"LLM suggested metric: {response_container.data.suggested_bar_chart_metric}")
            else:
                logger.warning("LLM did not provide 'suggested_bar_chart_metric'. Frontend may need a default.")

        else:
            # Log detailed error info if available
            error_details = getattr(response_container, 'error', 'Unknown error')
            raw_output = getattr(response_container, 'raw_output', 'N/A') # Get raw LLM output if possible
            data_type = type(getattr(response_container, 'data', None))
            logger.error(
                f"Agent execution failed or returned unexpected data type: {data_type}. "
                f"Error: {error_details}."
                # Log raw output carefully - it might be large or contain errors
                # f" Raw Output: {str(raw_output)[:1000]}..." # Log truncated raw output
            )
            # Attempt to print raw output if it exists and is useful
            if isinstance(raw_output, dict) and 'choices' in raw_output:
                 print(f"Raw LLM Output (potentially useful): {raw_output['choices'][0].get('message', {}).get('content', 'N/A')}")

            print("\n=== Agent Execution Failed ===")
            print(f"The agent did not return the expected ChartDataResponse data. Type: {data_type}")
            print(f"Error details: {error_details}")

    except Exception as e:
        logger.error(f"An error occurred during agent execution: {e}", exc_info=True)
        print(f"\n=== An Error Occurred During Agent Execution ===\n{e}")

if __name__ == "__main__":
    asyncio.run(main())