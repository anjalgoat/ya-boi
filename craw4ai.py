import sys
import asyncio
import logging
import json 
from typing import List
from pydantic import BaseModel, Field, ValidationError 
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel # 
from scrapfly import ScrapflyClient, ScrapeConfig, ScrapflyScrapeError
from readability import Document
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI 

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
                content_to_analyze = extracted_text[:4000] # Increased limit slightly

                # --- OpenAI Analysis using standard openai library ---
                prompt = (
                    f"Analyze the following content from the page titled '{title}' ({url}). "
                    f"Focus on aspects relevant to betting app market research.\n\n"
                    f"Content:\n{content_to_analyze}\n\n"
                    f"Provide the following:\n"
                    f"1. Summary: A detailed summary (4-5 sentences).\n"
                    f"2. Insight: One actionable insight specifically for building or marketing a betting app.\n"
                    f"3. Relevance: Assess relevance to 'betting app market research' (e.g., Highly relevant, Partially relevant, Not relevant) and briefly explain why.\n\n"
                    f"Format the output exactly like this, with each item on a new line:\n"
                    f"Summary: ...\n"
                    f"Insight: ...\n"
                    f"Relevance: ..."
                )
                try:
                    # Use the standard AsyncOpenAI client call with create()
                    response = await self.openai_client.chat.completions.create(
                        model=self.model_name, # Use the stored model name
                        messages=[
                            {"role": "system", "content": "You are an expert market research analyst specializing in the online betting industry."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.5, # Slightly adjusted for more focused output
                        max_tokens=350 # Increased slightly
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

                # Parse OpenAI output more robustly
                summary = insight = relevance = ""
                lines = response_text.strip().split('\n')
                for line in lines:
                    if line.lower().startswith("summary:"):
                        summary = line[len("summary:"):].strip()
                    elif line.lower().startswith("insight:"):
                        insight = line[len("insight:"):].strip()
                    elif line.lower().startswith("relevance:"):
                        relevance = line[len("relevance:"):].strip()
                # Handle cases where parsing might have missed something
                if not summary and not insight and not relevance and response_text:
                     # Fallback: use the whole text if parsing failed but got response
                     summary = f"Could not parse OpenAI response. Full response: {response_text[:200]}..."
                     insight = "Parsing failed."
                     relevance = "Parsing failed."
                elif not summary: summary = "Summary not found in response."
                elif not insight: insight = "Insight not found in response."
                elif not relevance: relevance = "Relevance not found in response."

                return ScrapeResult(
                    url=url, title=title,
                    summary=summary,
                    insight=insight,
                    relevance=relevance
                )
                # --- End OpenAI Analysis ---

            else:
                # Handle cases where ScrapFly succeeded technically but content wasn't usable
                status_code = result.scrape_result.get('status_code', 'N/A') if isinstance(result.scrape_result, dict) else 'N/A'
                reason = result.error or 'Unknown (content missing or success=False)'
                logger.warning(f"ScrapFly reported success={result.success} or missing content for {url}. Status: {status_code}, Reason: {reason}")
                return ScrapeResult(url=url, title=title, summary=f"Failed: ScrapFly could not retrieve usable content (Status: {status_code}, Reason: {reason}).", insight="No insight available.", relevance="Not relevant - content not retrieved.")

        # Handle ScrapFly-specific errors during the scrape attempt
        except ScrapflyScrapeError as e:
            status_code = e.response.status_code if hasattr(e, 'response') and e.response else 'N/A'
            logger.error(f"ScrapFly scraping error for {url}: {e.message} (Status: {status_code}, Attempt: {e.attempt})", exc_info=False) # Don't need full trace here
            return ScrapeResult(url=url, title=title, summary=f"Failed: ScrapFly error during scrape ({e.message}, Status: {status_code}).", insight="No insight available.", relevance="Not relevant - ScrapFly error.")
        # Handle any other unexpected errors during the process
        except Exception as e:
            logger.error(f"Generic error processing {url}: {e}", exc_info=True) # Log full traceback
            return ScrapeResult(url=url, title=title, summary="Failed: Unexpected error during processing.", insight="No insight available.", relevance="Not relevant - unexpected error.")


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


    # Handle case where processing somehow returned no results (e.g., all URLs failed individually)
    if not scrape_results:
        logger.warning("No results obtained after processing (all URLs might have failed); returning empty list")
        # According to the desired output, an empty list is valid if all fail
        scrape_results = []


    # Ensure all elements in the list are ScrapeResult objects (should be guaranteed now)
    final_results = [res for res in scrape_results if isinstance(res, ScrapeResult)]
    if len(final_results) != len(scrape_results):
        logger.warning("Some results were filtered out during final validation check.")


    return ScrapeResponse(results=final_results)


# Main execution
async def main():
    """Main function to parse input JSON from stdin, run the agent, process results, and print final JSON."""
    logger.info("Starting craw4ai.py agent execution. Reading URLs from stdin...")

    parsed_urls: List[UrlInput] = []
    input_json_from_stdin_str: str = ""
    error_response_sent = False

    # 1. Parse Input JSON from Stdin
    try:
        input_json_from_stdin_str = sys.stdin.read()
        
        if not input_json_from_stdin_str.strip():
            logger.error("craw4ai.py: No input received from stdin.")
            error_output = ScrapeResponse(results=[
                ScrapeResult(url="N/A", title="Input Error", summary="No input URLs received via stdin.", insight="N/A", relevance="N/A")
            ])
            print(error_output.model_dump_json(indent=2))
            error_response_sent = True
            return

        input_data = json.loads(input_json_from_stdin_str)
        
        if not isinstance(input_data, list) or not all(isinstance(item, dict) for item in input_data):
            logger.error("craw4ai.py: Input from stdin must be a JSON list of objects.")
            raise ValueError("Input from stdin must be a JSON list of objects.")
        
        temp_parsed_urls = []
        for item in input_data:
            try:
                temp_parsed_urls.append(UrlInput(**item))
            except ValidationError as e_item:
                logger.warning(f"craw4ai.py: Skipping an item due to validation error: {item}. Error: {e_item.errors()}")
        parsed_urls = temp_parsed_urls
        
        if not parsed_urls:
            logger.error("craw4ai.py: No valid URL items to process after parsing stdin.")
            error_output = ScrapeResponse(results=[
                 ScrapeResult(url="N/A", title="Input Error", summary="No valid URL items found in stdin input.", insight="N/A", relevance="N/A")
            ])
            print(error_output.model_dump_json(indent=2))
            error_response_sent = True
            return

        logger.info(f"craw4ai.py: Successfully parsed {len(parsed_urls)} URLs from stdin.")

    except json.JSONDecodeError:
        logger.error(f"craw4ai.py: Invalid JSON input provided via stdin. Content: {input_json_from_stdin_str[:500]}...")
        error_output = ScrapeResponse(results=[
            ScrapeResult(url="N/A", title="JSON Error", summary="Invalid JSON input format from stdin.", insight="N/A", relevance="N/A")
        ])
        print(error_output.model_dump_json(indent=2))
        error_response_sent = True
        return
    except ValidationError as e:
        logger.error(f"craw4ai.py: Input data validation error for items from stdin: {e}")
        error_output = ScrapeResponse(results=[
            ScrapeResult(url="N/A", title="Validation Error", summary=f"Input data validation failed: {e}", insight="N/A", relevance="N/A")
        ])
        print(error_output.model_dump_json(indent=2))
        error_response_sent = True
        return
    except Exception as e:
        logger.error(f"craw4ai.py: Error parsing input JSON from stdin: {e}", exc_info=True)
        error_output = ScrapeResponse(results=[
            ScrapeResult(url="N/A", title="Parsing Error", summary=f"Error processing input from stdin: {e}", insight="N/A", relevance="N/A")
        ])
        print(error_output.model_dump_json(indent=2))
        error_response_sent = True
        return

    if not parsed_urls and not error_response_sent:
         logger.warning("craw4ai.py: No valid URLs to process after parsing input from stdin.")
         print(ScrapeResponse(results=[]).model_dump_json(indent=2))
         return

    # 2. Format input for the agent
    agent_prompt_str = "For betting app market research, please process the following URLs and their titles:\n"
    for u in parsed_urls:
        agent_prompt_str += f"- Title: '{u.title}', URL: '{u.url}'\n"
    agent_prompt_str += "\nReturn these as an InitialScrapeResponse."

    # 3. Run the agent and processing pipeline
    final_response_data = None
    try:
        response_container = await scraper_agent.run(agent_prompt_str)

        if response_container and isinstance(response_container.data, ScrapeResponse):
            final_response_data = response_container.data
            logger.info(f"craw4ai.py: Agent execution and processing successful. Got {len(final_response_data.results)} results.")
        else:
            error_details = getattr(response_container, 'error', 'Unknown error')
            data_type = type(getattr(response_container, 'data', None))
            logger.error(f"craw4ai.py: Agent execution failed or returned unexpected data type: {data_type}. Error: {error_details}")
            final_response_data = ScrapeResponse(results=[ScrapeResult(
                url="N/A", title="Agent Error",
                summary=f"Agent execution failed. Type: {data_type}, Error: {error_details}",
                insight="No insight available.", relevance="N/A - agent processing error."
            )])

    except Exception as e:
        logger.error(f"craw4ai.py: An error occurred during main agent execution: {e}", exc_info=True)
        final_response_data = ScrapeResponse(results=[ScrapeResult(
             url="N/A", title="Critical Execution Error",
             summary=f"A critical error occurred during execution: {e}",
             insight="No insight available.", relevance="N/A - critical error."
        )])

    # 4. Output the final result as JSON
    if final_response_data:
         output_json = final_response_data.model_dump_json(indent=2, exclude_none=True)
         print(output_json)
    else:
         logger.error("craw4ai.py: Final response data was unexpectedly None after all processing.")
         fallback_error = ScrapeResponse(results=[
             ScrapeResult(url="N/A", title="Internal Error", summary="Failed to generate final response.", insight="N/A", relevance="N/A")
         ])
         print(fallback_error.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    asyncio.run(main())