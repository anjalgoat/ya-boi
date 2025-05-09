# summarizer.py
import os
import asyncio
import logging
import json
import sys # Import sys for stdin reading
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, HttpUrl
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from typing import List, Optional, Dict, Any

# --- Standard Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)] # Log to stderr
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
# Using a more capable model might be beneficial for complex synthesis
_model_name = os.getenv("OPENAI_MODEL", "openai/gpt-4o")
_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

if not openrouter_api_key:
    logging.error("CRITICAL: summarizer.py requires OPENROUTER_API_KEY environment variable.")
    # Avoid raising ValueError here so workflow.py can capture the failure

# --- Setup LLM ---
# Only initialize if API key is present
summarizer_agent = None
_model = None
if openrouter_api_key:
    try:
        _model = OpenAIModel(
            model_name=_model_name,
            api_key=openrouter_api_key,
            base_url=_base_url
        )
        # --- Define DETAILED Input Data Structures (Matching the expected input format) ---
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

        class FullMarketInput(BaseModel):
            """ Represents the complete input structure expected via stdin """
            user_query: str
            trends: Optional[TrendsInput] = None
            webpage_insights: Optional[WebpageInsightsInput] = None
            reviews: Optional[List[CompetitorReviews]] = None

        # --- Define Output Data Structure ---
        class MarketSummaryReport(BaseModel):
            original_query: str = Field(..., description="The initial user query being addressed")
            overall_market_summary: str = Field(..., description="A high-level synthesis of the current market state based on all inputs")
            key_market_trends: List[str] = Field(..., description="List of the most significant market trends identified from all sources")
            competitor_positioning_summary: str = Field(..., description="Brief summary comparing the key competitors based on provided insights and reviews")
            identified_gaps: List[str] = Field(..., description="Specific gaps identified between user needs/feedback (reviews) and current market offerings/trends (webpages, trends data)")
            strategic_opportunities: List[str] = Field(..., description="Actionable opportunities derived from the analysis and identified gaps")

        # --- Updated System Prompt for the Summarizer Agent ---
        summarizer_system_prompt = (
            # ...(Your existing detailed system prompt)...
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
            "9.  Ensure your output strictly adheres to the 'MarketSummaryReport' JSON format.\n"
            "10. **IMPORTANT**: Your final response MUST be ONLY the JSON object representing the MarketSummaryReport. Do not include any introductory text, explanations, apologies, or markdown formatting like ```json ... ```."

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
    except Exception as e:
        logger.error(f"Failed to initialize Pydantic AI Agent in summarizer.py: {e}", exc_info=True)
        summarizer_agent = None # Ensure agent is None if init fails
else:
    logger.error("Summarizer Agent cannot be initialized because OPENROUTER_API_KEY is missing.")


# --- Function to Run the Summarizer Agent ---
async def run_synthesizer(input_data: FullMarketInput) -> MarketSummaryReport:
    """
    Runs the Synthesizer Agent with the structured market input data.
    """
    logger.info("Running Market Synthesizer Agent...")

    if not summarizer_agent:
        logger.error("Summarizer Agent not initialized. Cannot run.")
        # Raise an exception or return a specific error structure
        # For now, raising exception which will be caught in main
        raise RuntimeError("Summarizer Agent failed to initialize (check API key).")

    # Construct the detailed prompt from the input data object
    prompt_content = f"Synthesize the following market data related to the user query: '{input_data.user_query}'\n\n"

    # Add Trends section if present
    if input_data.trends and input_data.trends.trends:
        prompt_content += "--- Market Trends Data ---\n"
        for i, trend in enumerate(input_data.trends.trends):
            prompt_content += f"Trend {i+1}: "
            prompt_content += f"Description: {trend.description}, Source: {trend.source}"
            if trend.metric: prompt_content += f", Metric: {trend.metric}"
            if trend.timestamp: prompt_content += f", Timestamp: {trend.timestamp}"
            prompt_content += "\n"
        prompt_content += "\n"
    else:
         prompt_content += "--- Market Trends Data ---\nNo specific trend data provided.\n\n"


    # Add Webpage Insights section if present
    if input_data.webpage_insights and input_data.webpage_insights.results:
        prompt_content += "--- Webpage Insights Data ---\n"
        for i, insight in enumerate(input_data.webpage_insights.results):
            prompt_content += f"Webpage {i+1} ({insight.url}):\n"
            prompt_content += f"  Title: {insight.title}\n"
            prompt_content += f"  Summary: {insight.summary}\n"
            prompt_content += f"  Insight: {insight.insight}\n"
            prompt_content += f"  Relevance: {insight.relevance}\n"
        prompt_content += "\n"
    else:
         prompt_content += "--- Webpage Insights Data ---\nNo specific webpage insight data provided.\n\n"


    # Add Reviews section if present
    if input_data.reviews:
        prompt_content += "--- Competitor Reviews Data ---\n"
        for i, competitor in enumerate(input_data.reviews):
            prompt_content += f"Competitor {i+1}: {competitor.competitor_name}\n"
            review_count = 0
            if competitor.app_store_reviews:
                prompt_content += "  App Store Reviews:\n"
                for review in competitor.app_store_reviews:
                     prompt_content += f"    - Rating: {review.rating if review.rating else 'N/A'}, Text: {review.text}\n"
                     review_count += 1
            if competitor.google_play_reviews:
                prompt_content += "  Google Play Reviews:\n"
                for review in competitor.google_play_reviews:
                     prompt_content += f"    - Rating: {review.rating if review.rating else 'N/A'}, Text: {review.text}\n"
                     review_count += 1
            if review_count == 0:
                 prompt_content += "  No specific reviews provided for this competitor.\n"

        prompt_content += "\n"
    else:
         prompt_content += "--- Competitor Reviews Data ---\nNo specific competitor review data provided.\n\n"


    prompt_content += "--- Task ---\n"
    prompt_content += "Based *only* on the detailed information provided above, generate the Market Summary Report following the specified JSON structure. Ensure your entire output is ONLY the valid JSON object for MarketSummaryReport."

    # No need for try-except here as agent.run() errors are caught in main()
    logger.debug(f"Sending prompt to LLM (length {len(prompt_content)}): \n{prompt_content[:1000]}...")
    agent_result = await summarizer_agent.run(prompt_content)

    if hasattr(agent_result, 'data') and isinstance(agent_result.data, MarketSummaryReport):
        final_report: MarketSummaryReport = agent_result.data
        # Ensure the original query is correctly transferred if model misses it
        if not final_report.original_query:
             final_report.original_query = input_data.user_query
        logger.info("Synthesizer Agent finished successfully and parsed data.")
        return final_report
    else:
        actual_type = type(agent_result.data) if hasattr(agent_result, 'data') else type(agent_result)
        logger.error(f"Agent returned unexpected data structure. Expected MarketSummaryReport in .data, got {actual_type}")
        raw_llm_response = "N/A"
        if hasattr(agent_result, 'raw_output') and isinstance(agent_result.raw_output, dict):
             raw_llm_response = agent_result.raw_output.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')
        logger.debug(f"Raw LLM Response: {raw_llm_response}")
        # Raise specific error to be caught in main
        raise TypeError(f"Agent returned unexpected data: {actual_type}. Raw response: {raw_llm_response[:500]}...")


# --- Main Execution Block ---
async def main():
    """Reads FullMarketInput JSON from stdin, runs synthesizer, prints MarketSummaryReport JSON to stdout."""
    logger.info("Starting summarizer.py execution.")
    input_json_str = ""
    final_output_json = "" # Variable to hold the JSON string to be printed

    # Define a fallback error response structure
    def create_error_response(query: str, message: str) -> str:
         error_resp = MarketSummaryReport(
             original_query=query,
             overall_market_summary=f"Error: {message}",
             key_market_trends=["Error generating report"],
             competitor_positioning_summary="Error generating report",
             identified_gaps=["Error generating report"],
             strategic_opportunities=["Error generating report"]
         )
         return error_resp.model_dump_json(indent=2, exclude_none=True)

    # 1. Read and Parse Input from Stdin
    market_input_data = None
    parsed_query = "Input Query Error"
    try:
        input_json_str = sys.stdin.read()
        if not input_json_str.strip():
            logger.error("Summarizer: No input received from stdin.")
            final_output_json = create_error_response("N/A", "No input received from stdin.")
            print(final_output_json)
            return

        market_input_data = FullMarketInput.model_validate_json(input_json_str)
        parsed_query = market_input_data.user_query # Store query for potential error messages
        logger.info("Summarizer: Successfully parsed and validated input JSON from stdin.")

    except Exception as e:
        logger.error(f"Summarizer: Error parsing input JSON from stdin: {e}", exc_info=True)
        final_output_json = create_error_response("N/A", f"Invalid input JSON - {e}")
        print(final_output_json)
        return

    # 2. Run the Synthesizer Agent
    try:
        final_report: MarketSummaryReport = await run_synthesizer(market_input_data)
        # Serialize the successful result to JSON for printing
        final_output_json = final_report.model_dump_json(indent=2, exclude_none=True)

    except Exception as e:
        # Catch errors from run_synthesizer (like agent init failure or LLM call failure)
        logger.error(f"Summarizer: Failed to generate report: {e}", exc_info=True)
        # Create an error JSON using the parsed query
        final_output_json = create_error_response(parsed_query, f"Failed to generate report - {e}")

    # 3. Print the Resulting JSON (either success or error structure) to stdout
    # Ensure nothing else is printed to stdout
    print(final_output_json)
    logger.info("Summarizer: Finished execution.")


if __name__ == "__main__":
    asyncio.run(main())