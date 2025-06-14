from typing import TypedDict, Sequence, Literal, List
from langgraph.graph import add_messages, StateGraph, END, START, MessagesState
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from PIL import Image
import io
import pprint


load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

tavily_search = TavilySearchResults(max_results=2)

python_repl_tool = PythonREPLTool()


class Supervisor(BaseModel):
    next: Literal["enhancer", "researcher", "coder"] = Field(
        description="Determines which specialist to activate next in the workflow sequence:"
        "'enhancer' when user input requires clarification, expansion or refinement, "
        "'researcher' when additional facts, context, or data collection is necessary, "
        "'coder' when implementation, computation, or technical problem-solving is required"
    )

    reason: str = Field(
        description="Detailed justification for the routing decision, explaing the rationale behind selecting the particular specialist and how this advances the task toward completion."
    )


def supervisor_node(
    state: MessagesState,
) -> Command[Literal["enhancer", "researcher", "coder"]]:
    system_prompt = """You are a workflow supervisor managing a team of three specialized agents: Prompt Enhancer, Researcher, and Coder. Your role is to orchestrate the workflow by selecting the most appropriate next agent based on the current state and needs of the task. Provide a clear, concise rationale for each decision to ensure transparency in your decision-making process.
        
        **Team Members**:
        1. **Prompt Enhancer**: Always consider this agent first. They clarify ambiguous requests, improve poorly defined queries, and ensure the task is well-structured before deeper processing begins.
        2. **Researcher**: Specializes in information gathering, fact-finding, and collecting relavent data needed to address the user's request. 
        3. **Prompt Enhancer**: Focuses on technical implementation, calculations, data analysis, algorithm development, and coding solutions.
        
        **Your Responsibilities**:
        1. Analyze each user request and agent response for completeness, accuracy, and relevance. 
        2. Route the task to the most appropriate agent at the decision point.
        3. Maintain workflow momentum by avoding redundant agent assignments
        4. Continue the process until the user's request is fully and satisfactorily resolved.
        
        Your objective is to create an efficient workflow that leverages each agent's strengths while minimizing unneccessary steps, ultimately delivering complete and accurate solutions to user requests.
        
        """

    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    # want output to always have next and reason
    response = llm.with_structured_output(Supervisor).invoke(messages)

    goto = response.next
    reason = response.reason

    print(f"--- Workflow Transition: Supervisor -> {goto.upper()} ---")

    return Command(
        update={"messages": [HumanMessage(content=reason, name="supervisor")]},
        goto=goto,
    )


def enhancer_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    Enhancer agent node that improves and clarifies users queries. Takes the orignal user input and transforms it into a more precise, actionable request before passing it to the supervisor
    """

    system_prompt = "You are a Query Refinement Specialist with expertise in transfroming vague requests into precise instructions. Your responsibilities include:\n\n"
    "1. Analyzing the original query to identify key intent and requirements\n"
    "2. Resolving any ambiguities without requesting additional user input\n"
    "3. Expanding underdeveloped aspects of the query with reasonable assumptions\n"
    "4. Restructuring the query for clarity and actionability\n"
    "5. Ensuring all the technical terminology is properly defined in context\n\n"
    "Important: Never ask questions back to the user. Instead, make informed assumptions and create the most comprehensive version of their request possible"

    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    enhanced_query = llm.invoke(messages)

    print("--- Workflow Transition: Prompt Enhancer -> Supervisor ---")

    return Command(
        update={
            "messages": [HumanMessage(content=enhanced_query.content, name="enhancer")]
        },
        goto="supervisor",
    )


def research_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Research agent node that gathers information using Tavily search. Takes the current task state, performs research, and returns findings for valiation
    """

    research_agent = create_react_agent(
        llm,
        tools=[tavily_search],
        state_modifier="You are an Information Specialist with expertise in comprehensive research. Your responsibilities include: \n\n"
        "1. Identifying key information needs based on the query context \n"
        "2. Gathering relevant, accurate, and up-to-date information from reliable sources \n"
        "3. Organizaing findings in a structured, easily digestible format\n"
        "4. Citing sources when possible to establish credibility \n"
        "5. Focusing exclusively on information gathering - avoid analysis or implementation \n\n"
        "Provide thorough, factual responses without speculation where information is unavailable",
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


def code_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Code agent node that focuses on mathematical calculations. Takes the current task state, executes code, and returns findings for valiation
    """

    research_agent = create_react_agent(
        llm,
        tools=[python_repl_tool],
        # state_modifier="You are a code and analyst. Focus on mathematical calculations, analyzing, solving math questions, and executing code. Handle Technical probelm-solving and data tasks.",
    )

    result = research_agent.invoke(state)

    print("--- Workflow Transition: Coder -> Validator ---")

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        goto="validator",
    )


class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Specifies the next worker in the pipeline: 'supervisor' to continue or 'FINISH' to terminate."
    )

    reason: str = Field(description="The reason for the decision.")


def validator_node(
    state: MessagesState,
) -> Command[Literal["supervisor", END]]:
    user_question = state["messages"][0].content
    agent_answer = state["messages"][-1].content

    system_prompt = """Your task is to ensure reasonable quality. 
                Specifically, you must:
                - Review the user's question (the first message in the workflow).
                - Review the answer (the last message in the workflow)
                - If the answer addresses the core intent of the question, even if not perfectly, signal to end the workflow with 'FINISH'.
                - Only route back to the supervisor if the answer is completely off-topic, harmful, or fundamentally misunderstands the question.
                
                - Accept answers that are 'good enough' rather than perfect
                - Prioritize workflow completion over perfect responses
                - Give benefit of the doubt to borderline answers
                
                Routing Guidelines:
                1. 'supervisor' Agent: ONLY for responses that are completely incorrect of off-topic.
                2. Respond with 'FINISH' in all other cases to end the workflow
                """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": agent_answer},
    ]

    response = llm.with_structured_output(Validator).invoke(messages)

    goto = response.next
    reason = response.reason

    if goto == "FINISH" or goto == END:
        goto = END
        print("--- Transition to END ---")
    else:
        print("--- Workflow Transition: Validator -> Supervisor ---")

    return Command(
        update={
            "messages": [HumanMessage(content=reason, name="validator")],
        },
        goto=goto,
    )


graph = StateGraph(MessagesState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("enhancer", enhancer_node)
graph.add_node("researcher", research_node)
graph.add_node("coder", code_node)
graph.add_node("validator", validator_node)

graph.add_edge(START, "supervisor")
app = graph.compile()


# Generate
inputs = {"messages": ["user", "Give me the 20th fibonacci number"]}
for event in app.stream(inputs):
    for key, value in event.items():
        if value is None:
            continue
        last_message = value.get("messages", [])[-1] if "messages" in value else None
        if last_message:
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint(last_message, indent=2, width=80, depth=None)
            print()


"""Create Agent Flow Graph"""
# img_bytes = app.get_graph(xray=True).draw_mermaid_png()
# image = Image.open(io.BytesIO(img_bytes))
# image.show()  # Opens the default image viewer
