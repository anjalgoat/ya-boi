import os
import asyncio
import logging
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define constants for research agents
TREND_ANALYZER = "TREND_ANALYZER"
COMPETITOR_INTELLIGENCE = "COMPETITOR_INTELLIGENCE"
SOCIAL_SENTIMENT = "SOCIAL_SENTIMENT"
LOCAL_CONTEXT = "LOCAL_CONTEXT"
TASK_CHECKER = "TASK_CHECKER"
AI_SUMMARIZER = "AI_SUMMARIZER"
GAP_FINDER = "GAP_FINDER"
PRESENTATION = "PRESENTATION"

# Define Pydantic models for the orchestrator's output
class ResearchQuestion(BaseModel):
    question: str
    priority: int = Field(description="Priority level from 1 (highest) to 5 (lowest)", ge=1, le=5)
    relevant_agent: str = Field(description="Primary agent responsible for this question")

    @field_validator("relevant_agent")
    @classmethod
    def validate_relevant_agent(cls, v):
        valid_agents = [
            TREND_ANALYZER, COMPETITOR_INTELLIGENCE, SOCIAL_SENTIMENT, LOCAL_CONTEXT,
            TASK_CHECKER, AI_SUMMARIZER, GAP_FINDER, PRESENTATION
        ]
        if v not in valid_agents:
            raise ValueError(f"relevant_agent must be one of {valid_agents}")
        return v

class AgentTaskContext(BaseModel):
    business_type: Optional[str] = None
    target_location: Optional[str] = None
    target_audience: Optional[str] = None

class AgentTask(BaseModel):
    specific_agent: str
    task: str
    dependencies: List[str] = Field(default_factory=list)
    priority: int = Field(ge=1, le=5)
    context: Optional[AgentTaskContext] = None

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, v, info):
        # Ensure no self-dependencies
        agent_name = info.data.get("specific_agent", "unknown")
        if agent_name in v:
            raise ValueError(f"Agent {agent_name} cannot depend on itself")
        return v

class ResearchPlanContext(BaseModel):
    business_type: str
    target_location: str
    target_audience: Optional[str] = None
    research_questions: List[ResearchQuestion]

class ResearchPlan(BaseModel):
    context: ResearchPlanContext
    agent_tasks: Dict[str, AgentTask]
    model_config = ConfigDict(extra='forbid')

    @field_validator("agent_tasks")
    @classmethod
    def validate_agent_tasks(cls, v, info):
        # Check for circular dependencies
        def detect_cycle(graph, node, visited, stack):
            visited.add(node)
            stack.add(node)
            for dep in graph.get(node, {}).dependencies:
                if dep not in visited:
                    if detect_cycle(graph, dep, visited, stack):
                        return True
                elif dep in stack:
                    return True
            stack.remove(node)
            return False

        visited = set()
        stack = set()
        for agent_name in v:
            if agent_name not in visited:
                if detect_cycle(v, agent_name, visited, stack):
                    raise ValueError(f"Circular dependency detected in agent_tasks")

        # Ensure context consistency
        high_level_context = info.data.get("context", None)
        if high_level_context:
            for agent_name, task in v.items():
                if task.context:
                    if task.context.business_type and task.context.business_type != high_level_context.business_type:
                        raise ValueError(f"business_type mismatch for {agent_name}")
                    if task.context.target_location and task.context.target_location != high_level_context.target_location:
                        raise ValueError(f"target_location mismatch for {agent_name}")
                    if task.context.target_audience and high_level_context.target_audience and task.context.target_audience != high_level_context.target_audience:
                        raise ValueError(f"target_audience mismatch for {agent_name}")
        return v

# Environment variable checks with defaults
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is required.")

model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Default to a known model
base_url = os.getenv("OPENAI_BASE_URL", "https://api.openrouter.ai/v1")  # Default base URL

_model = OpenAIModel(
    model_name=model_name,
    api_key=openrouter_api_key,
    base_url=base_url
)

system_prompt = (
    "You are the Orchestrator Agent for a market research system. Your role is to analyze user input about a business idea, "
    "create a comprehensive research strategy, and output a structured plan matching the ResearchPlan schema.\n\n"
    
    "Output Requirements:\n"
    "1. context: Include business type, target location, target audience, and research questions\n"
    "2. agent_tasks: A dictionary with agent names as keys, each containing:\n"
    "   - specific_agent: Actual agent performing the task\n"
    "   - task: Detailed description of the task\n"
    "   - dependencies: List of agent names that must complete before this agent (ensure no circular dependencies)\n"
    "   - priority: Task priority (1-5)\n"
    "   - context: Business context details (optional for some agents like TASK_CHECKER, AI_SUMMARIZER, GAP_FINDER, PRESENTATION)\n\n"
    
    "Essential Agents to Include:\n"
    f"1. {TREND_ANALYZER}: Market trend analysis\n"
    f"2. {COMPETITOR_INTELLIGENCE}: Competitor research\n"
    f"3. {SOCIAL_SENTIMENT}: Public sentiment analysis\n"
    f"4. {LOCAL_CONTEXT}: Location-specific insights\n"
    f"5. {TASK_CHECKER}: Verify task completions\n"
    f"6. {AI_SUMMARIZER}: Summarize research findings\n"
    f"7. {GAP_FINDER}: Identify market opportunities\n"
    f"8. {PRESENTATION}: Create final presentation (depends on AI_SUMMARIZER and GAP_FINDER)\n\n"
    
    "Ensure that the context in agent_tasks matches the high-level context where applicable. "
    "Your output must be a valid, comprehensive market research plan with no circular dependencies."
)

orchestrator_agent = Agent(
    _model,
    system_prompt=system_prompt,
    result_type=ResearchPlan,
    result_tool_name="research_plan",
    result_tool_description="Creates a structured market research plan based on the user's business idea",
    result_retries=3,
)

@orchestrator_agent.result_validator
def validate_result(ctx: RunContext[None], result: ResearchPlan) -> ResearchPlan:
    """Validate that the research plan is comprehensive and well-structured."""
    try:
        # Validate context
        if not result.context.business_type or not result.context.target_location:
            raise ModelRetry("Research plan must include business type and target location.")
        
        # Validate research questions
        if len(result.context.research_questions) < 3:
            raise ModelRetry("At least 3 research questions are required.")
        
        # Validate essential agents
        essential_agents = [
            TREND_ANALYZER, COMPETITOR_INTELLIGENCE, SOCIAL_SENTIMENT, LOCAL_CONTEXT,
            TASK_CHECKER, AI_SUMMARIZER, GAP_FINDER, PRESENTATION
        ]
        for agent in essential_agents:
            if agent not in result.agent_tasks:
                raise ModelRetry(f"Essential agent {agent} must be included.")
        
        # Validate PRESENTATION dependencies
        presentation_task = result.agent_tasks.get(PRESENTATION)
        if presentation_task and not all(dep in presentation_task.dependencies for dep in [AI_SUMMARIZER, GAP_FINDER]):
            raise ModelRetry(f"{PRESENTATION} must depend on both {AI_SUMMARIZER} and {GAP_FINDER}")

        return result
    except Exception as e:
        logger.error(f"Validation Error: {str(e)}")
        raise

async def process_user_query(query: str) -> ResearchPlan:
    """Process a user query to generate a market research plan."""
    try:
        logger.info(f"Processing user query: {query}")
        agent_result = await orchestrator_agent.run(query)
        
        # Extract the ResearchPlan from the AgentRunResult
        if not hasattr(agent_result, 'data'):
            logger.error(f"AgentRunResult attributes: {dir(agent_result)}")
            raise AttributeError("AgentRunResult does not have a 'data' attribute. Check pydantic-ai documentation for the correct property.")
        research_plan = agent_result.data  # Use 'data' instead of 'value'
        if not isinstance(research_plan, ResearchPlan):
            raise TypeError(f"Expected ResearchPlan, got {type(research_plan)}")
        
        logger.info("Research plan generated successfully")
        return research_plan
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise

# Example usage (for standalone testing)
async def main():
    query = "I want to start a dating app for gay people in San Francisco targeting young professionals"
    try:
        research_plan = await process_user_query(query)
        print(research_plan.model_dump_json(indent=2))
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())