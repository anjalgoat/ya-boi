import logging
from typing import Dict, Any
from copy import deepcopy

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
INITIAL_AGENTS = ["TREND_ANALYZER", "COMPETITOR_INTELLIGENCE", "SOCIAL_SENTIMENT", "LOCAL_CONTEXT"]
TASK_CHECKER = "TASK_CHECKER"
AI_SUMMARIZER = "AI_SUMMARIZER"

# Simulated LangGraph state (base state for all test scenarios)
BASE_STATE = {
    "task_id": "task_id_1",
    "orchestrator_output": {
        "context": {
            "business_type": "dating app",
            "target_location": "San Francisco",
            "target_audience": "young professionals",
            "research_questions": [
                {"question": "What are the current trends in dating apps for the LGBTQ+ community in San Francisco?", "priority": 1, "relevant_agent": "TREND_ANALYZER"},
                {"question": "Who are the main competitors targeting gay professionals in the dating app market in San Francisco?", "priority": 1, "relevant_agent": "COMPETITOR_INTELLIGENCE"},
                {"question": "What is the public sentiment regarding dating apps among young gay professionals in San Francisco?", "priority": 2, "relevant_agent": "SOCIAL_SENTIMENT"},
                {"question": "What cultural and social factors should be considered when launching a dating app for gay professionals in San Francisco?", "priority": 3, "relevant_agent": "LOCAL_CONTEXT"}
            ]
        },
        "agent_tasks": {
            "TREND_ANALYZER": {
                "specific_agent": "Google Search Agent",
                "task": "Scrape Google Search and app store data for trends in LGBTQ+ dating apps (user growth, popular features, market size)",
                "dependencies": [],
                "priority": 1,
                "context": {"business_type": "dating app", "target_location": "San Francisco", "target_audience": "young professionals"}
            },
            "COMPETITOR_INTELLIGENCE": {
                "specific_agent": "Reviews Agent",
                "task": "Identify competitors targeting gay professionals via app store reviews and Google Reviews",
                "dependencies": ["TREND_ANALYZER"],
                "priority": 2,
                "context": {"business_type": "dating app", "target_location": "San Francisco", "target_audience": "young professionals"}
            },
            "SOCIAL_SENTIMENT": {
                "specific_agent": "Twitter Sentiment Agent",
                "task": "Analyze Twitter sentiment for dating apps among gay professionals",
                "dependencies": ["COMPETITOR_INTELLIGENCE"],
                "priority": 3,
                "context": {"business_type": "dating app", "target_location": "San Francisco", "target_audience": "young professionals"}
            },
            "LOCAL_CONTEXT": {
                "specific_agent": "Demographic Data Agent",
                "task": "Analyze cultural events, community values, and legal considerations for the gay community in San Francisco using demographic databases",
                "dependencies": ["SOCIAL_SENTIMENT"],
                "priority": 4,
                "context": {"business_type": "dating app", "target_location": "San Francisco", "target_audience": "young professionals"}
            },
            "TASK_CHECKER": {
                "specific_agent": "Task Checker Agent",
                "task": "Verify completion of initial agents and notify AI Summarizer",
                "dependencies": ["TREND_ANALYZER", "COMPETITOR_INTELLIGENCE", "SOCIAL_SENTIMENT", "LOCAL_CONTEXT"],
                "priority": 5,
                "context": {}
            },
            "AI_SUMMARIZER": {
                "specific_agent": "AI Summarizer Agent",
                "task": "Summarize the results from initial agents into a concise narrative",
                "dependencies": ["TASK_CHECKER"],
                "priority": 6,
                "context": {}
            }
        }
    },
    "agent_results": {},
    "current_agent": "TASK_CHECKER",
    "status": "pending",
    "final_output": None
}

class TaskCheckerAgent:
    def __init__(self, state: Dict[str, Any]):
        self.state = state
        self.dependencies = self.state["orchestrator_output"]["agent_tasks"]["TASK_CHECKER"]["dependencies"]

    def fetch_state(self) -> Dict[str, Any]:
        """Simulate fetching the state (in a real system, this would be an API call)."""
        logger.info("Fetching state for task_id: %s", self.state["task_id"])
        return self.state

    def verify_completions(self) -> tuple[bool, Dict[str, Any]]:
        """
        Verify that all Initial Agents have completed their tasks.
        Returns a tuple of (success, errors), where success is True if all tasks are complete,
        and errors is a dictionary of agents with errors.
        """
        logger.info("Verifying completions for Initial Agents: %s", self.dependencies)
        agent_results = self.state["agent_results"]
        errors = {}
        all_complete = True

        for agent in self.dependencies:
            if agent not in agent_results:
                logger.warning("Agent %s has not completed its task.", agent)
                all_complete = False
            elif "error" in agent_results[agent]:
                logger.warning("Agent %s failed with error: %s", agent, agent_results[agent]["error"])
                errors[agent] = agent_results[agent]["error"]
                all_complete = False
            else:
                logger.info("Agent %s completed successfully.", agent)

        return all_complete, errors

    def update_state(self, next_agent: str, status: str) -> Dict[str, Any]:
        """Update the state to transition to the next agent."""
        logger.info("Updating state: setting current_agent to %s and status to %s", next_agent, status)
        self.state["current_agent"] = next_agent
        self.state["status"] = status
        return self.state

    def notify_next_agent(self, next_agent: str) -> None:
        """Simulate notifying the next agent (in a real system, this would be an API call)."""
        logger.info("Notifying next agent: %s for task_id: %s", next_agent, self.state["task_id"])

    def run(self, retry_on_error: bool = False) -> Dict[str, Any]:
        """Run the Task Checker Agent's logic."""
        # Fetch the state
        self.fetch_state()

        # Verify completions
        all_complete, errors = self.verify_completions()

        if all_complete:
            logger.info("All Initial Agents completed successfully.")
            # Update state to transition to AI_SUMMARIZER
            self.update_state(AI_SUMMARIZER, "pending")
            # Notify the next agent
            self.notify_next_agent(AI_SUMMARIZER)
        else:
            if errors and retry_on_error:
                # Retry logic: Reset current_agent to the first failed agent
                failed_agent = list(errors.keys())[0]
                logger.info("Retrying failed agent: %s", failed_agent)
                self.update_state(failed_agent, "pending")
                self.notify_next_agent(failed_agent)
            else:
                # Proceed with partial results if retry is not enabled
                logger.warning("Proceeding with partial results due to errors: %s", errors)
                self.update_state(AI_SUMMARIZER, "pending_with_errors")
                self.notify_next_agent(AI_SUMMARIZER)

        return self.state

# Test scenarios
def test_task_checker():
    # Scenario 1: All Initial Agents completed successfully
    logger.info("\n=== Scenario 1: All Initial Agents Completed Successfully ===")
    state1 = deepcopy(BASE_STATE)
    state1["agent_results"] = {
        "TREND_ANALYZER": {"trends": ["Grindr: 1M users, growing", "Scruff: 500K users, feature-rich"]},
        "COMPETITOR_INTELLIGENCE": {"competitors": ["Grindr", "Scruff", "Tinder"]},
        "SOCIAL_SENTIMENT": {"sentiment": "Mostly positive, privacy concerns noted"},
        "LOCAL_CONTEXT": {"cultural_factors": ["Pride events", "inclusivity focus"]}
    }
    agent1 = TaskCheckerAgent(state1)
    final_state1 = agent1.run()
    logger.info("Final state: %s", final_state1)

    # Scenario 2: One Initial Agent failed
    logger.info("\n=== Scenario 2: One Initial Agent Failed (Retry Enabled) ===")
    state2 = deepcopy(BASE_STATE)
    state2["agent_results"] = {
        "TREND_ANALYZER": {"trends": ["Grindr: 1M users, growing", "Scruff: 500K users, feature-rich"]},
        "COMPETITOR_INTELLIGENCE": {"error": "Failed to fetch reviews"},
        "SOCIAL_SENTIMENT": {"sentiment": "Mostly positive, privacy concerns noted"},
        "LOCAL_CONTEXT": {"cultural_factors": ["Pride events", "inclusivity focus"]}
    }
    agent2 = TaskCheckerAgent(state2)
    final_state2 = agent2.run(retry_on_error=True)
    logger.info("Final state: %s", final_state2)

    # Scenario 3: One Initial Agent not completed (Proceed with partial results)
    logger.info("\n=== Scenario 3: One Initial Agent Not Completed (Proceed with Partial Results) ===")
    state3 = deepcopy(BASE_STATE)
    state3["agent_results"] = {
        "TREND_ANALYZER": {"trends": ["Grindr: 1M users, growing", "Scruff: 500K users, feature-rich"]},
        "COMPETITOR_INTELLIGENCE": {"competitors": ["Grindr", "Scruff", "Tinder"]},
        "SOCIAL_SENTIMENT": {"sentiment": "Mostly positive, privacy concerns noted"}
        # LOCAL_CONTEXT is missing
    }
    agent3 = TaskCheckerAgent(state3)
    final_state3 = agent3.run(retry_on_error=False)
    logger.info("Final state: %s", final_state3)

if __name__ == "__main__":
    test_task_checker()