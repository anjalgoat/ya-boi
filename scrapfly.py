import asyncio
import logging
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel # Still needed for Agent
from scrapfly import ScrapflyClient, ScrapeConfig, ScrapflyScrapeError
from readability import Document
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI # Import the standard OpenAI async client

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
scrapfly_api_key = os.getenv("SCRAPFLY_API_KEY")

# Validate environment variables
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is required.")
if not scrapfly_api_key:
    raise ValueError("SCRAPFLY_API_KEY environment variable is required.")

# Pydantic models
class UrlInput(BaseModel):
    url: str = Field(..., description="URL to scrape")
    title: str = Field(..., description="Title of the URL")

class ScrapeResult(BaseModel):
    url: str = Field(..., description="Scraped URL")
    title: str = Field(..., description="Title of the page")
    summary: str = Field(..., description="Detailed summary of the content (4-5 sentences)")
    insight: str = Field(..., description="Actionable insight for betting app market research")
    relevance: str = Field(..., description="Relevance verdict with explanation")

class ScrapeResponse(BaseModel):
    results: List[ScrapeResult] = Field(..., description="List of scrape results", min_items=1)

class InitialScrapeResponse(BaseModel):
    urls: List[UrlInput] = Field(..., description="List of URLs to process")

# --- Agent Setup ---
# We still need the pydantic-ai OpenAIModel for the Agent itself
_model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo") # Or your preferred OpenRouter model
_agent_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1") # Default to OpenRouter

_agent_model = OpenAIModel( # Use pydantic-ai's model for the Agent
    model_name=_model_name,
    api_key=openrouter_api_key,
    base_url=_agent_base_url
)

system_prompt = (
    "You are an AI agent for betting app market research. Given a list of URLs with titles:\n"
    "1. Validate that URLs and titles are properly formatted.\n"
    "2. Return the URLs and titles for further processing.\n"
    "Do not scrape or analyze content; that will be handled separately."
)

scraper_agent = Agent(
    _agent_model, # Pass the pydantic-ai model instance here
    system_prompt=system_prompt,
    result_type=InitialScrapeResponse,
    result_tool_name="scrape_urls",
    result_tool_description="Log URLs to be scraped",
    result_retries=3, # Retries for the initial LLM call if needed
)
# --- End Agent Setup ---


class ScraperProcessor:
    """Handles scraping via ScrapFly and analysis via OpenAI."""
    def __init__(self):
        # Use the standard openai AsyncOpenAI client for direct calls
        # Ensure it points to the correct base URL for OpenRouter
        self.openai_client = AsyncOpenAI(
            api_key=openrouter_api_key,
            base_url=_agent_base_url # Use the same base URL as the agent
        )
        # Initialize ScrapflyClient once
        self.scrapfly_client = ScrapflyClient(key=scrapfly_api_key)
        # Store model name for direct openai calls
        self.model_name = _model_name
        logger.info(f"Initialized ScraperProcessor with AsyncOpenAI (model: {self.model_name}, base_url: {_agent_base_url}) and ScrapFly clients")

    async def cleanup(self):
        """Clean up async resources if any were explicitly created and need closing."""
        # await self.openai_client.close() # Typically not needed unless managing client lifecycle strictly
        logger.info("Cleanup called - AsyncOpenAI client managed by library, ScrapFly has no async cleanup.")

    async def scrape_url(self, url: str, title: str) -> ScrapeResult:
        """Scrape a single URL using ScrapFly and analyze its content using OpenAI."""
        extracted_text = ""
        try:
            logger.info(f"Scraping {url} using ScrapFly")
            scrape_config = ScrapeConfig(
                url=url,
                asp=True,  # Enable Anti Scraping Protection
                # country="US", # Optional: Specify proxy country
            )
            result = await self.scrapfly_client.async_scrape(scrape_config)

            # Check ScrapFly result validity and access content using dictionary keys
            if result.success and isinstance(result.scrape_result, dict) and result.scrape_result.get('content'):
                html_content = result.scrape_result['content']

                # --- Content Extraction ---
                try:
                    doc = Document(html_content)
                    # Use readability's summary html and clean it with BeautifulSoup
                    summary_html = doc.summary()
                    soup = BeautifulSoup(summary_html, 'html.parser')
                    extracted_text = soup.get_text(separator=' ', strip=True)

                    # Fallback if readability fails to get significant text
                    if not extracted_text or len(extracted_text) < 50: # Check length as well
                        logger.warning(f"Readability extracted minimal/no text ({len(extracted_text)} chars) from {url}, trying full body text.")
                        soup_full = BeautifulSoup(html_content, 'html.parser')
                        body_tag = soup_full.find('body') # Find body tag
                        if body_tag:
                             body_text = body_tag.get_text(separator=' ', strip=True)
                             if body_text:
                                 extracted_text = body_text
                             else:
                                 logger.error(f"Could not extract text content from body tag of {url}.")
                                 return ScrapeResult(url=url, title=title, summary="Failed: No text found in body.", insight="No insight available.", relevance="Relevance unclear - no text.")
                        else:
                             logger.error(f"Could not extract any text content from {url} (no body tag found).")
                             return ScrapeResult(url=url, title=title, summary="Failed: No body tag found.", insight="No insight available.", relevance="Relevance unclear - no text.")

                except Exception as e_extract:
                    logger.error(f"Error extracting content with readability/bs4 for {url}: {e_extract}", exc_info=True)
                    return ScrapeResult(url=url, title=title, summary="Failed during content extraction phase.", insight="No insight available.", relevance="Relevance unclear - extraction error.")
                # --- End Content Extraction ---

                logger.info(f"Extracted ~{len(extracted_text)} characters from {url} using ScrapFly/Readability")
                # Limit content size before sending to LLM
                content_to_analyze = extracted_text[:4000]

                # --- OpenAI Analysis using standard openai library ---
                prompt = (
                    f"Summarize the following content in 4-5 sentences, focusing on betting app market research:\n\n{content_to_analyze}\n\n"
                    f"Then, provide one actionable insight and assess relevance to 'app for betting app market research' "
                    f"(Highly relevant, Partially relevant, Not relevant, with explanation).\n"
                    f"Return as:\nSummary: ...\nInsight: ...\nRelevance: ..."
                 )
                try:
                    # Use the standard AsyncOpenAI client call with create()
                    response = await self.openai_client.chat.completions.create(
                        model=self.model_name, # Use the stored model name
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                        # Optional: Add other parameters like temperature, max_tokens
                        # temperature=0.7,
                        # max_tokens=300
                    )

                    if not response or not response.choices:
                         logger.error(f"Invalid response structure from OpenAI API for {url}")
                         raise ValueError("Invalid OpenAI response structure") # Raise error

                    # Extract response text
                    response_text = response.choices[0].message.content

                except Exception as e_openai:
                     logger.error(f"OpenAI API call failed for {url}: {e_openai}", exc_info=True)
                     # Return failure but indicate scraping was successful
                     return ScrapeResult(
                        url=url, title=title,
                        summary=f"Scraped successfully, but OpenAI analysis failed. Raw text length: {len(extracted_text)}",
                        insight="OpenAI analysis failed.",
                        relevance="OpenAI analysis failed."
                    )

                # Parse OpenAI output
                summary = insight = relevance = ""
                current_field = None
                for line in response_text.split("\n"):
                    line = line.strip()
                    if line.startswith("Summary:"): current_field = "summary"; summary = line.replace("Summary:", "").strip()
                    elif line.startswith("Insight:"): current_field = "insight"; insight = line.replace("Insight:", "").strip()
                    elif line.startswith("Relevance:"): current_field = "relevance"; relevance = line.replace("Relevance:", "").strip()
                    elif current_field and line: # Append multi-line content
                        if current_field == "summary": summary += " " + line
                        elif current_field == "insight": insight += " " + line
                        elif current_field == "relevance": relevance += " " + line

                return ScrapeResult(
                    url=url, title=title,
                    summary=summary or "Summary parsing failed.",
                    insight=insight or "Insight parsing failed.",
                    relevance=relevance or "Relevance parsing failed."
                )
                # --- End OpenAI Analysis ---

            else:
                # Handle cases where ScrapFly succeeded technically but content wasn't usable
                status_code = result.scrape_result.get('status_code', 'N/A') if isinstance(result.scrape_result, dict) else 'N/A'
                logger.warning(f"ScrapFly reported success=False or missing content for {url}. Success: {result.success}, Status: {status_code}, Reason: {result.error or 'Unknown'}")
                return ScrapeResult(url=url, title=title, summary="Failed to retrieve content via ScrapFly (Success=False or missing content).", insight="No insight available.", relevance="Not relevant - content not retrieved.")

        # Handle ScrapFly-specific errors during the scrape attempt
        except ScrapflyScrapeError as e:
            status_code = e.response.status_code if hasattr(e, 'response') and e.response else 'N/A'
            logger.error(f"ScrapFly scraping error for {url}: {e.message} (Status: {status_code}, Attempt: {e.attempt})", exc_info=False)
            return ScrapeResult(url=url, title=title, summary=f"Failed to retrieve content due to ScrapFly error: {e.message}", insight="No insight available.", relevance="Not relevant - ScrapFly error.")
        # Handle any other unexpected errors during the process
        except Exception as e:
            logger.error(f"Generic error processing {url}: {e}", exc_info=True) # Log full traceback
            return ScrapeResult(url=url, title=title, summary="Failed to retrieve content due to an unexpected error.", insight="No insight available.", relevance="Not relevant - unexpected error.")


    async def process_urls(self, urls: List[UrlInput]) -> List[ScrapeResult]:
        """Process multiple URLs sequentially using the single ScraperProcessor instance."""
        results = []
        for url_input in urls:
             # scrape_url now returns a ScrapeResult even on failure
             result = await self.scrape_url(url_input.url, url_input.title)
             results.append(result)
        return results

# Result validator (triggered after the initial agent call)
@scraper_agent.result_validator
async def validate_result(ctx: RunContext[None], result: InitialScrapeResponse) -> ScrapeResponse:
    """Validates the URLs from the agent and triggers the scraping process."""
    logger.info(f"Validating URLs from agent: {[u.url for u in result.urls]}")
    validated_urls = []
    for url_input in result.urls:
        # Basic URL format check
        if not (url_input.url.startswith("http://") or url_input.url.startswith("https://")):
            logger.error(f"Invalid URL format skipped: {url_input.url}")
            continue # Skip this URL
        # Basic title check
        if not url_input.title or not url_input.title.strip():
            logger.error(f"Empty title for URL skipped: {url_input.url}")
            continue # Skip this URL
        validated_urls.append(url_input)

    # Handle case where no URLs passed validation
    if not validated_urls:
        logger.error("No valid URLs received from agent after validation.")
        # Return a response indicating failure
        return ScrapeResponse(results=[ScrapeResult(
            url="N/A", title="No Valid URLs",
            summary="No valid URLs were provided by the agent or passed validation.",
            insight="No insights generated.", relevance="Not relevant."
        )])

    # Initialize the processor that uses ScrapFly and OpenAI
    processor = ScraperProcessor()
    scrape_results = []
    try:
        # Process only the validated URLs
        scrape_results = await processor.process_urls(validated_urls)
    except Exception as e:
        logger.error(f"Critical error during processor.process_urls: {e}", exc_info=True)
        # Create fallback results for all validated URLs if processing fails globally
        scrape_results = [
            ScrapeResult(
                url=url_input.url, title=url_input.title,
                summary="Failed due to overarching processing error.",
                insight="No insight available.", relevance="Not relevant - processing error."
            ) for url_input in validated_urls
        ]
    finally:
        # Call cleanup if the processor has any resources needing explicit async cleanup
        await processor.cleanup()


    # Handle case where processing somehow returned no results
    if not scrape_results:
        logger.warning("No results obtained after processing; returning fallback")
        scrape_results = [ScrapeResult(
            url="https://example.com/fallback", title="Fallback Result",
            summary="No content was retrieved from any attempted URLs.",
            insight="No insights available due to scraping failures.",
            relevance="Not relevant - no content retrieved."
        )]

    # Ensure all elements in the list are ScrapeResult objects (should be guaranteed now)
    final_results = [res for res in scrape_results if isinstance(res, ScrapeResult)]
    if len(final_results) != len(scrape_results):
        logger.warning("Some results were filtered out during final validation check.")


    return ScrapeResponse(results=final_results)


# Print results
def print_results(response: ScrapeResponse):
    """Prints the formatted results from the ScrapeResponse."""
    print("\n=== Betting App Scrape Results ===")
    if not response or not response.results:
        print("No results were generated or provided.")
        return

    for i, result in enumerate(response.results, 1):
        print(f"\n{i}. URL: {result.url}")
        print(f"   Title: {result.title}")
        print(f"   Summary: {result.summary}")
        print(f"   Insight: {result.insight}")
        print(f"   Relevance: {result.relevance}")
        print("-" * 60)

# Test URLs
test_urls = [
    UrlInput(
        url="https://www.apptunix.com/blog/how-to-develop-a-sports-betting-app-like-fanduel-a-detailed-guide/",
        title="Build a Sports Betting App Like FanDuel: A Detailed Guide!"
    ),
    UrlInput(
        url="https://themindstudios.com/blog/how-to-make-sport-betting-app/",
        title="Sports Betting App Development like William Hill or Paddy ..."
    ),
    UrlInput(
        url="https://bettinghero.com/",
        title="Betting Hero: Sports Betting Help, Odds, and Insights"
    ),
    UrlInput(
        url="https://ekgamingllc.com/",
        title="Eilers & Krejcik Gaming Consulting and Market Research"
    ),
    UrlInput(
        url="https://www.grandviewresearch.com/industry-analysis/sports-betting-market-report",
        title="Sports Betting Market Size & Share Analysis Report, 2030"
    )
]

# Main execution
async def main():
    """Main function to run the agent and process results."""
    logger.info("Starting agent execution")
    try:
        # Format input for the agent - simple list format
        input_str = f"Process these URLs for betting market research:\n" + \
                    "\n".join([f"- {u.title}: {u.url}" for u in test_urls])

        # Run the initial agent call (which triggers the validator and scraping)
        response_container = await scraper_agent.run(input_str)

        # Check the final response from the validator
        if response_container and isinstance(response_container.data, ScrapeResponse):
             print_results(response_container.data)
        else:
             # Log details if the agent run failed or returned unexpected data
             error_details = getattr(response_container, 'error', 'Unknown error')
             data_type = type(getattr(response_container, 'data', None))
             logger.error(f"Agent execution failed or returned unexpected data type: {data_type}. Error: {error_details}")
             print("\n=== Agent Execution Failed ===")
             print(f"The agent did not return the expected ScrapeResponse data. Type: {data_type}")
             print(f"Error details: {error_details}")
             # You might want to inspect response_container itself here for more clues

    except Exception as e:
        logger.error(f"An error occurred during main execution: {e}", exc_info=True)
        print(f"\n=== An Error Occurred During Main Execution ===\n{e}")


if __name__ == "__main__":
    # Ensure you have a .env file with OPENROUTER_API_KEY and SCRAPFLY_API_KEY
    # Run the main asynchronous function
    asyncio.run(main())