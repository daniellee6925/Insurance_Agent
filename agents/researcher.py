from typing import Literal
from langgraph.types import Command
from langchain.schema import HumanMessage
from state import InsuranceAgentState
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from pathlib import Path

from llm import llm

tavily_search = TavilySearchResults(max_results=4)

# Load system prompt
prompt_path = Path(__file__).parent / "../prompts/validator_prompt.txt"
system_prompt = prompt_path.read_text()


def research_node(state: InsuranceAgentState) -> Command[Literal["validator"]]:
    """
    Research agent node that gathers up-to-date information using Tavily search.
    This agent focuses only on finding and formatting relevant information,
    not interpreting or acting on it.
    """

    research_agent = create_react_agent(
        llm,
        tools=[tavily_search],
        state_modifier=system_prompt,
    )

    result = research_agent.invoke(state)

    print("--- Workflow Transition: Researcher -> Validator ---")

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="validator",
    )
