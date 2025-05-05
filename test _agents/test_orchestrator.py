import asyncio
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Assuming the Orchestrator Agent code is in a file called `orchestrator.py`
from orchestrator import orchestrator_agent, process_user_query, ResearchPlan

# Test query
TEST_QUERY = "I want to start a dating app for gay people in San Francisco targeting young professionals."

async def run_test():
    """Run a test query through the Orchestrator Agent and print the result."""
    print(f"Processing query: {TEST_QUERY}\n")
    
    try:
        # Process the query using the orchestrator agent
        agent_result = await process_user_query(TEST_QUERY)
        
        # First, print the agent_result type and its attributes to understand its structure
        print(f"Agent result type: {type(agent_result)}")
        print("Agent result attributes:")
        for attr in dir(agent_result):
            if not attr.startswith('_'):  # Skip private attributes
                try:
                    value = getattr(agent_result, attr)
                    print(f"  - {attr}: {type(value)}")
                except:
                    print(f"  - {attr}: <error accessing>")
        
        # Try to access common attributes that might contain the result
        possible_attributes = ['value', 'data', 'content', 'result', 'response', 'output']
        research_plan = None
        
        for attr in possible_attributes:
            if hasattr(agent_result, attr):
                print(f"\nTrying to use {attr} attribute:")
                try:
                    value = getattr(agent_result, attr)
                    # Check if this looks like our ResearchPlan
                    if isinstance(value, dict) and 'business_type' in value:
                        research_plan = ResearchPlan(**value)
                        print(f"Found ResearchPlan in {attr}!")
                        break
                    elif isinstance(value, ResearchPlan):
                        research_plan = value
                        print(f"Found ResearchPlan directly in {attr}!")
                        break
                    else:
                        print(f"  Value type: {type(value)}")
                        if hasattr(value, "__dict__"):
                            print(f"  Value contents: {value.__dict__}")
                        else:
                            try:
                                print(f"  Value contents: {value}")
                            except:
                                print("  Cannot display value contents")
                except Exception as e:
                    print(f"  Error accessing {attr}: {e}")
        
        # If we can't find the research plan in common attributes, 
        # try the direct object as the research plan
        if not research_plan and isinstance(agent_result, ResearchPlan):
            research_plan = agent_result
            print("\nUsing agent_result directly as ResearchPlan")
        
        # If we can't find the research plan in any attribute, print the entire object
        if not research_plan:
            print("\nCould not identify ResearchPlan in result. Printing full result:")
            try:
                if hasattr(agent_result, "__dict__"):
                    print(json.dumps(agent_result.__dict__, indent=2, default=str))
                else:
                    print(agent_result)
            except:
                print("Could not serialize agent_result")
        else:
            # Print the research plan in a readable format
            print("\n=== Generated Research Plan ===")
            print(f"Business Type: {research_plan.business_type}")
            print(f"Target Location: {research_plan.target_location}")
            print(f"Target Audience: {research_plan.target_audience}")
            print("\nResearch Questions:")
            for idx, question in enumerate(research_plan.research_questions, 1):
                print(f"{idx}. {question.question} (Priority: {question.priority})")
                print(f"   Relevant Agents: {', '.join(question.relevant_agents)}")
            print("\nAgent Tasks:")
            for idx, task in enumerate(research_plan.agent_tasks, 1):
                print(f"{idx}. Agent: {task.agent}")
                print(f"   Tasks: {', '.join(task.tasks)}")
                print(f"   Dependencies: {', '.join(task.dependencies) if task.dependencies else 'None'}")
            print("\nTimeline:")
            for agent, time in research_plan.timeline.items():
                print(f"{agent}: {time}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

# Run the test
if __name__ == "__main__":
    asyncio.run(run_test())