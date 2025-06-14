# Multi-Agent Insurance Assistant with LangGraph

## Objective
Create a LangGraph-based multi-agent system to assist users with insurance-related queries. A central **Supervisor Agent** routes user queries to specialized agents.

---

## Agents Overview

| Agent | Role | Input | Output | Tools |
|-------|------|-------|--------|-------|
| **Supervisor** | Routes queries | User input | Agent route | Router logic (LLM or rules) |
| **Enhancer** | Rephrases ambiguous questions | User query | Clarified query | Prompt template + LLM |
| **Validator** | Reflects and improves responses | Agent output | Refined output | Self-reflection prompt |
| **RAG Agent** | Provides policy info | Clarified query | Retrieved info | VectorDB + RAG chain |
| **Search Agent** | Fetches real-time info | Query | Latest info | Web search (Tavily, SerpAPI) |
| **Actuary Agent** | Computes actuarial value | Plan details | Actuarial value | Python formula |
| **Consultant Agent** | Recommends plans | User preferences | Suggested plan | Ranking + prompt |
| **Underwriting Agent** | Assesses risk | User profile | Risk score | LLM or rule-based |

---

## Phase 1: Define Shared State


state = State(keys={
    "messages": list,
    "user_query": str,
    "enhanced_query": str,
    "validated_response": str,
    "policy_docs": str,
    "search_results": str,
    "actuarial_value": float,
    "recommendation": str,
    "risk_assessment": str,
})


## Phase 2: Create Agents as Nodes
- Each agent is a function that reads and updates the state.

async def enhancer_agent(state):
    clarified = await clarifier_llm.ainvoke(f"Rephrase to clarify: {state['user_query']}")
    return {"enhanced_query": clarified}


## Phase 2: Build LangGraph
graph = StateGraph(state)

# Add agents as nodes
graph.add_node("supervisor", supervisor_agent)
graph.add_node("enhancer", enhancer_agent)
graph.add_node("validator", validator_agent)
graph.add_node("rag", rag_agent)
graph.add_node("search", search_agent)
graph.add_node("actuary", actuary_agent)
graph.add_node("consultant", consultant_agent)
graph.add_node("underwriter", underwriter_agent)

# Entry point
graph.set_entry_point("supervisor")

# Routing logic
graph.add_conditional_edges(
    "supervisor",
    condition_function=route_query,
    path_map={
        "clarify": "enhancer",
        "retrieve": "rag",
        "search": "search",
        "calculate": "actuary",
        "recommend": "consultant",
        "assess": "underwriter",
    }
)

# Optional validation stage
graph.add_edge("rag", "validator")
graph.add_edge("search", "validator")
graph.add_edge("consultant", "validator")
graph.add_edge("validator", END)


Tools by Agent
- Enhancer: Clarifier prompt + OpenAI/Anthropic
- Validator: Chain-of-thought self-reflection + confidence threshold
- RAG: FAISS/ChromaDB + LangChain retriever
- Search: LangChain web search tool (e.g., Tavily)
- Actuary: Local calculator for actuarial value
- Consultant: Plan matcher using scoring logic
- Underwriter: LLM or rule-based risk rating system

Recommended Steps
1. Scaffold with LangGraph, basic state + supervisor.
2. Implement 1–2 agents (e.g., Enhancer + RAG).
3. Validate pipeline.
4. Add complexity iteratively (Actuary, Validator, etc.).
5. Optional: add memory, human-in-the-loop, UI.



Folder Structure
insurance_agent/
├── agents/
│   ├── __init__.py
│   ├── supervisor.py
│   ├── enhancer.py
│   ├── validator.py
│   ├── rag.py
│   ├── search.py
│   ├── actuary.py
│   ├── consultant.py
│   └── underwriter.py
├── tools/
│   ├── actuarial_calculator.py
│   ├── search_tool.py
│   └── rag_tool.py
├── prompts/
│   ├── enhancer_prompt.txt
│   ├── validator_prompt.txt
│   └── consultant_prompt.txt
├── graph/
│   ├── __init__.py
│   └── main_graph.py
├── state.py
├── config.py
├── requirements.txt
└── README.md

# agents/
- Each file should define a LangGraph-compatible async def function that accepts a state and returns a partial state update.

# tools/
- Custom logic for actuarial calculation, search, or document retrieval.

# prompts/
- Store prompt templates separately to improve modularity and make fine-tuning easier.

# graph/main_graph.py
    - Your LangGraph orchestration file:
    - Import all agents.
    - Build the full state graph.
    - Define conditional routing.
    - Set entry and end points.

# state.py
- Define your shared LangGraph State once and import it as needed.