"""State.py

It defines
- The data structure each agent reads and writes.
- The initial state that gets passed into the LangGraph.
- Optional typing for better structure and debugging.
"""

from typing import List, Dict, Optional, Any
from langgraph.graph import State

# Each message in the conversation history follows ChatOpenAI format - {"role": "user", "content": "..."}
ChatMessage = Dict[str, str]


class InsuranceAgentState(State):
    # Full chat history between user and assistant
    messages: List[ChatMessage] = []

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

    # Extra info or routing/debugging data
    metadata: Dict[str, Any] = {}
