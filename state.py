"""State.py

It defines
- The data structure each agent reads and writes.
- The initial state that gets passed into the LangGraph.
- Optional typing for better structure and debugging.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

# Each message in the conversation history follows ChatOpenAI format - {"role": "user", "content": "..."}
ChatMessage = Dict[str, str]


class InsuranceAgentState(BaseModel):
    # Full chat history between user and assistant
    messages: list[BaseMessage] = Field(default_factory=list)

    # Original user question
    user_question: Optional[str] = None

    # Clarified question from the enhancer agent
    enhanced_question: Optional[str] = None

    # RAG or search context used to generate answers
    rag_context: Optional[str] = None
    search_context: Optional[str] = None

    # Raw assistant response before reflection/validation
    raw_answer: Optional[str] = None

    # Improved response after validator or reflection agent
    validated_answer: Optional[str] = None

    # Calculated actuarial value (numerical result)
    actuarial_value: Optional[float] = None

    # List of suggested plans from consultant agent
    plan_recommendations: Optional[List[str]] = None

    # Underwriting decision or risk level
    risk_assessment: Optional[str] = None
    user_info: Dict[str, str] = Field(default_factory=dict)  # [REQUIRED_FIELDS, ANSWER]
    next_field: Optional[str] = None  # Track the field currently being asked

    # Extra info or routing/debugging data
    metadata: Dict[str, Any] = {}
