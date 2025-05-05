import os
import asyncio
import logging
import json # Added for JSON handling
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, HttpUrl
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from typing import List, Optional, Dict, Any

# --- Standard Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
# Using a more capable model might be beneficial for complex synthesis
_model_name = os.getenv("OPENAI_MODEL", "openai/gpt-4o")
_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is required.")

# --- Setup LLM ---
_model = OpenAIModel(
    model_name=_model_name,
    api_key=openrouter_api_key,
    base_url=_base_url
)

# --- Define DETAILED Input Data Structures (Matching the new input format) ---

class TrendItem(BaseModel):
    description: str
    source: str
    metric: Optional[str] = None # Made optional as not always present/structured
    timestamp: Optional[str] = None # Assuming timestamp is ISO format string

class TrendsInput(BaseModel):
    trends: List[TrendItem]

class WebpageResult(BaseModel):
    url: HttpUrl # Use HttpUrl for validation
    title: str
    summary: str
    insight: str
    relevance: str

class WebpageInsightsInput(BaseModel):
    results: List[WebpageResult]

class ReviewItem(BaseModel):
    rating: Optional[int] = None # Can be optional
    text: str

class CompetitorReviews(BaseModel):
    competitor_name: str
    app_store_reviews: Optional[List[ReviewItem]] = [] # Default to empty list
    google_play_reviews: Optional[List[ReviewItem]] = [] # Default to empty list

class FullMarketInput(BaseModel):
    """ Represents the complete input structure """
    user_query: str
    trends: Optional[TrendsInput] = None # Make sections optional
    webpage_insights: Optional[WebpageInsightsInput] = None
    reviews: Optional[List[CompetitorReviews]] = None

# --- Define Output Data Structure (Pydantic Model - same as before) ---
class MarketSummaryReport(BaseModel):
    original_query: str = Field(..., description="The initial user query being addressed")
    overall_market_summary: str = Field(..., description="A high-level synthesis of the current market state based on all inputs")
    key_market_trends: List[str] = Field(..., description="List of the most significant market trends identified from all sources")
    competitor_positioning_summary: str = Field(..., description="Brief summary comparing the key competitors based on provided insights and reviews")
    identified_gaps: List[str] = Field(..., description="Specific gaps identified between user needs/feedback (reviews) and current market offerings/trends (webpages, trends data)")
    strategic_opportunities: List[str] = Field(..., description="Actionable opportunities derived from the analysis and identified gaps")

# --- Updated System Prompt for the Summarizer Agent ---
summarizer_system_prompt = (
    "You are an expert Market Analysis Synthesizer AI. Your task is to analyze comprehensive market data provided in multiple sections: "
    "1. 'User Query': The original context for the analysis. "
    "2. 'Trends': Snippets of market trends from various sources. "
    "3. 'Webpage Insights': Summaries and insights derived from relevant web articles/reports. "
    "4. 'Reviews': User feedback (pros/cons) for specific competitors. "
    "Your goal is to synthesize ALL this information into a concise and actionable 'Market Summary Report'.\n\n"
    "Instructions:\n"
    "1.  Thoroughly analyze all provided sections (Trends, Webpage Insights, Reviews) in the context of the 'User Query'.\n"
    "2.  **Overall Market Summary**: Create a high-level synthesis. What is the main state of the market related to the query? Combine insights from webpages, trends, and overarching themes from reviews.\n"
    "3.  **Key Market Trends**: Extract and list the *most significant* trends impacting this market. Use information from the 'Trends' section AND the 'Webpage Insights'. Prioritize trends mentioned in multiple places or with clear metrics.\n"
    "4.  **Competitor Positioning Summary**: Summarize the strengths and weaknesses of each competitor mentioned in the 'Reviews' section. Corroborate findings with information from 'Webpage Insights' if available. Mention key differentiating factors or common criticisms.\n"
    "5.  **Identified Gaps**: Critically analyze the data to find gaps. Where do user needs/complaints (from 'Reviews') contrast with market offerings or reported trends ('Webpage Insights', 'Trends')? What seems to be missing in the market based on the collective data?\n"
    "6.  **Strategic Opportunities**: Based *only* on the identified trends and gaps, list actionable strategic opportunities. These should directly address the gaps or leverage the trends.\n"
    "7.  Use the 'original_query' field from the input for the corresponding field in the report.\n"
    "8.  Be concise and focused. Base your report *strictly* on the provided input data.\n"
    "9.  Ensure your output strictly adheres to the 'MarketSummaryReport' JSON format."
)

# --- Create the Summarizer Agent ---
summarizer_agent = Agent(
    _model,
    system_prompt=summarizer_system_prompt,
    result_type=MarketSummaryReport,
    result_tool_name="market_synthesis_generator",
    result_tool_description="Generates the market summary report by synthesizing trends, web insights, and reviews.",
    result_retries=2,
)

# --- Updated Function to Run the Summarizer Agent ---
async def run_synthesizer(input_data: FullMarketInput) -> MarketSummaryReport:
    """
    Runs the Synthesizer Agent with the structured market input data.
    """
    logger.info("Running Market Synthesizer Agent...")

    # Construct the detailed prompt from the input data object
    prompt_content = f"Synthesize the following market data related to the user query: '{input_data.user_query}'\n\n"

    # Add Trends section if present
    if input_data.trends and input_data.trends.trends:
        prompt_content += "--- Market Trends Data ---\n"
        for i, trend in enumerate(input_data.trends.trends):
            prompt_content += f"Trend {i+1}:\n"
            prompt_content += f"  Description: {trend.description}\n"
            if trend.source: prompt_content += f"  Source: {trend.source}\n"
            if trend.metric: prompt_content += f"  Metric: {trend.metric}\n"
            if trend.timestamp: prompt_content += f"  Timestamp: {trend.timestamp}\n"
        prompt_content += "\n"

    # Add Webpage Insights section if present
    if input_data.webpage_insights and input_data.webpage_insights.results:
        prompt_content += "--- Webpage Insights Data ---\n"
        for i, insight in enumerate(input_data.webpage_insights.results):
            prompt_content += f"Webpage {i+1}:\n"
            prompt_content += f"  URL: {insight.url}\n"
            prompt_content += f"  Title: {insight.title}\n"
            prompt_content += f"  Summary: {insight.summary}\n"
            prompt_content += f"  Insight: {insight.insight}\n"
            prompt_content += f"  Relevance: {insight.relevance}\n"
        prompt_content += "\n"

    # Add Reviews section if present
    if input_data.reviews:
        prompt_content += "--- Competitor Reviews Data ---\n"
        for i, competitor in enumerate(input_data.reviews):
            prompt_content += f"Competitor {i+1}: {competitor.competitor_name}\n"
            if competitor.app_store_reviews:
                prompt_content += "  App Store Reviews:\n"
                for review in competitor.app_store_reviews:
                     prompt_content += f"    - Rating: {review.rating if review.rating else 'N/A'}, Text: {review.text}\n"
            if competitor.google_play_reviews:
                prompt_content += "  Google Play Reviews:\n"
                for review in competitor.google_play_reviews:
                     prompt_content += f"    - Rating: {review.rating if review.rating else 'N/A'}, Text: {review.text}\n"
        prompt_content += "\n"

    prompt_content += "--- Task ---\n"
    prompt_content += "Based *only* on the detailed information provided above, generate the Market Summary Report following the specified JSON structure."

    try:
        logger.debug(f"Sending prompt to LLM: \n{prompt_content[:1000]}...") # Log start of prompt
        agent_result = await summarizer_agent.run(prompt_content)

        if hasattr(agent_result, 'data') and isinstance(agent_result.data, MarketSummaryReport):
            final_report: MarketSummaryReport = agent_result.data
            # Ensure the original query is correctly transferred
            if not final_report.original_query:
                 final_report.original_query = input_data.user_query
            logger.info("Synthesizer Agent finished successfully and parsed data.")
            return final_report
        else:
            actual_type = type(agent_result.data) if hasattr(agent_result, 'data') else type(agent_result)
            logger.error(f"Agent returned unexpected data structure. Expected MarketSummaryReport in .data, got {actual_type}")
            # Log the raw response if helpful (be careful with large outputs)
            raw_llm_response = getattr(agent_result, 'raw_output', {}).get('choices', [{}])[0].get('message', {}).get('content', 'N/A')
            logger.debug(f"Raw LLM Response: {raw_llm_response}")
            raise TypeError(f"Expected MarketSummaryReport, got {actual_type}")

    except Exception as e:
        logger.error(f"Synthesizer Agent failed during execution or data extraction: {e}", exc_info=True)
        raise


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Use the New Example Input JSON (as a string) ---
    input_json_string = """
{
  "user_query": "best betting app",
  "trends": {
    "trends": [
      {
        "description": "Search interest for betting app global boxing enthusiasts, gambler",
        "source": "Google Trends",
        "metric": "75 search interest",
        "timestamp": "2025-04-18T12:00:00Z"
      },
      {
        "description": "DraftKings is a popular app with features: Live betting and cash-out options",
        "source": "App Store",
        "metric": "4.6 rating",
        "timestamp": "2025-04-18T12:00:00Z"
      },
      {
          "description": "Regulatory changes opening up markets in North America",
          "source": "Industry Report",
          "metric": null,
          "timestamp": "2025-04-15T10:00:00Z"
      }
    ]
  },
  "webpage_insights": {
    "results": [
      {
        "url": "https://www.apptunix.com/blog/how-to-develop-a-sports-betting-app-like-fanduel-a-detailed-guide/",
        "title": "Build a Sports Betting App Like FanDuel: A Detailed Guide!",
        "summary": "The article outlines steps to develop a sports betting app like FanDuel, emphasizing user-friendly design, live betting features, and secure payment systems. Highlights regulatory compliance.",
        "insight": "Prioritize live betting and real-time odds to compete with leaders like FanDuel.",
        "relevance": "Highly relevant - provides actionable development insights for betting apps."
      },
      {
        "url": "https://www.grandviewresearch.com/industry-analysis/sports-betting-market-report",
        "title": "Sports Betting Market Size & Share Analysis Report, 2030",
        "summary": "The report forecasts significant growth in the sports betting market, driven by legalization in the US and Europe. Mobile betting apps dominate due to convenience. Growing trend towards in-play and esports betting.",
        "insight": "Focus on in-play and esports betting to capture emerging market segments.",
        "relevance": "Highly relevant - offers market trends and competitive insights."
      }
    ]
  },
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
  ]
}
"""

    # --- Parse Input JSON and Validate with Pydantic ---
    try:
        input_dict = json.loads(input_json_string)
        # Validate the entire structure using the top-level model
        market_input_data = FullMarketInput(**input_dict)
        logger.info("Successfully parsed and validated input JSON.")
    except json.JSONDecodeError:
        logger.error("Invalid JSON input provided.")
        print(json.dumps({"error": "Invalid JSON input format."}, indent=2))
        exit(1)
    except ValidationError as e:
        logger.error(f"Input data validation error: {e}")
        print(json.dumps({"error": f"Input data validation failed: {e}"}, indent=2))
        exit(1)
    except Exception as e:
        logger.error(f"Error processing input JSON: {e}", exc_info=True)
        print(json.dumps({"error": f"Error processing input: {e}"}, indent=2))
        exit(1)

    # --- Run the Agent Asynchronously ---
    try:
        # Pass the validated Pydantic object to the synthesizer function
        final_report: MarketSummaryReport = asyncio.run(run_synthesizer(market_input_data))

        # --- Print the Resulting JSON ---
        # Use .model_dump_json() for correct serialization
        output_json = final_report.model_dump_json(indent=2)
        print("\n" + "="*20 + " Synthesized Market Summary Report (JSON) " + "="*20)
        print(output_json)
        print("\n" + "="*70)

    except Exception as e:
        # Error already logged in run_synthesizer or during the call
        print(json.dumps({"error": f"Failed to generate report: {e}"}, indent=2))