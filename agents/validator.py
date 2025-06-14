"""
validator.py

Defines the Validator agent, responsible for assessing whether an agent's response
adequately addresses the user's original question. If the response is acceptable,
the workflow ends. If the response is incorrect or off-topic, control is routed
back to the supervisor for reassignment.
"""

from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import END
from langgraph.types import Command
from pathlib import Path
from langchain.schema import HumanMessage

from state import InsuranceAgentState
from llm import llm

# Load system prompt
prompt_path = Path(__file__).parent / "../prompts/validator_prompt.txt"
system_prompt = prompt_path.read_text()


class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Where to route next: 'supervisor' or 'FINISH'."
    )
    reason: str = Field(description="Why this decision was made.")


def validator_node(state: InsuranceAgentState) -> Command[Literal["supervisor", END]]:
    user_question = state.messages[0]["content"]
    agent_answer = state.messages[-1]["content"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": agent_answer},
    ]

    response = llm.with_structured_output(Validator).invoke(messages)
    goto = response.next
    reason = response.reason

    if goto == "FINISH":
        print("--- Transition to END ---")
        goto = END
    else:
        print("--- Workflow Transition: Validator -> Supervisor ---")

    return Command(
        update={
            "messages": [HumanMessage(content=reason, name="validator")],
        },
        goto=goto,
    )
