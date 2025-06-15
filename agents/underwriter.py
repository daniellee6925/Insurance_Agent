"""
underwriter.py

Defines the Underwriter agent, responsible for assessing whether an agent's response
adequately addresses the user's original question. If the response is acceptable,
the workflow ends. If the response is incorrect or off-topic, control is routed
back to the supervisor for reassignment.

Risk Class Description
Low Risk - Excellent health, no chronic conditions, healthy lifestyle
Moderate Risk - Minor conditions (e.g., mild asthma), good control, no major risk factors
High Risk - Serious conditions (e.g., diabetes, heart disease), or risky lifestyle
Major Risk - Conditions that are very costly (e.g., terminal illness)

"""

from langgraph.types import Command
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path

from llm import llm
from state import InsuranceAgentState

# Load system prompt
prompt_path = Path(__file__).parent / "../prompts/underwriter_prompt.txt"
system_prompt = prompt_path.read_text()


class UnderwritingAssessment(BaseModel):
    risk_class: Literal["Low Risk", "Moderate Risk", "High Risk", "Major Risk"] = Field(
        description="Risk classification based on user input"
    )
    reasoning: str = Field(description="Reason for the risk classification")


REQUIRED_FIELDS = [
    "age",
    "gender",
    "height",
    "weight",
    "tobacco_use",
    "alcohol_use",
    "pre_existing_conditions",
    "current_medications",
    "occupation",
    "risky_activities",
    "recent_hospitalizations",
    "family_medical_history",
]

questions_map = {
    "age": "What is your age?",
    "gender": "What is your gender?",
    "height": "What is your height (in feet/inches or cm)?",
    "weight": "What is your weight (in pounds or kg)?",
    "tobacco_use": "Do you currently use any tobacco products (e.g., cigarettes, vapes) Yes/No?",
    "alcohol_use": "Do you consume alcohol at least once every week? Yes/No",
    "pre_existing_conditions": "Do you have any pre-existing medical conditions (e.g., diabetes, asthma, heart disease)?",
    "current_medications": "Are you currently taking any prescription medications?",
    "occupation": "What is your current occupation?",
    "risky_activities": "Do you participate in any high-risk activities (e.g., extreme sports, hazardous work) Yes/No?",
    "recent_hospitalizations": "Have you been hospitalized in the past 5 years? If so, please provide details.",
    "family_medical_history": "Does your family have a history of serious medical conditions (e.g., cancer, heart disease, diabetes)?",
}


def underwriter_node(state: InsuranceAgentState) -> Command[Literal["Verifier"]]:
    user_info = state.user_info
    last_message = state.messages[-1].content if state.messages else ""

    # Store the user's last answer if we were asking a question
    if state.next_field:
        user_info[state.next_field] = last_message
        state.next_field = None

    # Ask next missing question
    for field in REQUIRED_FIELDS:
        if field not in user_info or not user_info[field]:
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=questions_map[field], name="underwriter")
                    ],
                    "user_info": user_info,
                    "next_field": field,
                },
                goto="underwriter",
            )

    summary = "\n".join(
        [f"{k.replace('_', ' ').title()}: {v}" for k, v in user_info.items()]
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User information:\n{summary}"},
    ]

    result = llm.with_structured_output(UnderwritingAssessment).invoke(messages)

    return Command(
        update={
            "messages": [
                HumanMessage(
                    name="underwriter",
                    content=f"Risk Class: {result.risk_class}\nReasoning: {result.reasoning}",
                )
            ],
            "user_info": user_info,
        },
        goto="verifier",
    )
