"""
main.py

Entry point for the Insurance Agent multi-agent system using LangGraph.
This sets up the graph, initializes state, and runs an example conversation.
"""

from langgraph.graph import StateGraph, END
from state import InsuranceAgentState
from agents.supervisor import supervisor_node
from agents.validator import validator_node


def build_graph():
    builder = StateGraph(InsuranceAgentState)

    # Add all agent nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("validator", validator_node)
    # builder.add_node("enhancer", enhancer_node)
    # builder.add_node("consultant", consultant_node)
    # builder.add_node("actuary", actuary_node)
    # builder.add_node("underwriting", underwriting_node)
    # builder.add_node("search", search_node)
    # builder.add_node("rag", rag_node)

    # Example: Route supervisor to validator for testing
    builder.set_entry_point("supervisor")
    builder.add_edge("supervisor", "validator")
    builder.add_edge("validator", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()

    # Example input state
    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": "What’s the best health insurance for my family?",
            }
        ]
    }

    final_state = graph.invoke(initial_state)
    print("\nFinal Output:\n", final_state)
