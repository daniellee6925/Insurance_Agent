"""
validator.py

Defines the Verifier agent, responsible for evaluating the underwriting agent’s
risk classification and reasoning. If the risk assessment is appropriate and
justified, the workflow proceeds to the consultant agent. If the assessment is
inadequate or inconsistent with the reasoning, control is routed back to the
underwriter agent for revision.
"""

from pydantic import BaseModel, Field
from typing import Literal
from langgraph.types import Command
from pathlib import Path
from langchain.schema import HumanMessage

from state import InsuranceAgentState
from llm import llm

# Load system prompt
prompt_path = Path(__file__).parent / "../prompts/verifier_prompt.txt"
system_prompt = prompt_path.read_text()


class Validator(BaseModel):
    next: Literal["underwriter", "consultant"] = Field(
        description="Route to 'underwriter' if the risk classification is not supported by the reasoning. Otherwise, route to 'consultant'."
    )
    reason: str = Field(description="Why this decision was made.")


def validator_node(
    state: InsuranceAgentState,
) -> Command[Literal["underwriter", "consultant"]]:
    underwriter_summary = state.messages[-1]["content"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": underwriter_summary},
    ]

    response = llm.with_structured_output(Validator).invoke(messages)
    goto = response.next
    reason = response.reason

    print(f"--- Workflow Transition: Validator -> {goto.capitalize()}  ---")

    return Command(
        update={
            "messages": [HumanMessage(content=reason, name="verifier")],
        },
        goto=goto,
    )
