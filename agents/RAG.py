from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import TypedDict, Annotated, Sequence, Literal, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import add_messages, StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

embedding_function = OpenAIEmbeddings()

docs = [
    Document(page_content="some content here", metadata={"source": "about.txt"}),
    Document(page_content="some content here", metadata={"source": "hours.txt"}),
    Document(page_content="some content here", metadata={"source": "membership.txt"}),
    Document(page_content="some content here", metadata={"source": "classes.txt"}),
    Document(page_content="some content here", metadata={"source": "trainers.txt"}),
    Document(page_content="some content here", metadata={"source": "facilities.txt"}),
]

# chroma is an open source vector DB
# Indexes vectors and lets you search for similar content
db = Chroma.from_documents(docs, embedding_function)

# MMR: Maximal Marginal Relevance (balances relavence and diversity), k:3 -> top 3 results
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})


llm = ChatOpenAI(model="gpt-4o")

template = """Answer the question based on the followng context and the ChatHistory. Especially take the latest question into consideration:
ChatHistory: {history} Context: {contex} Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = prompt | llm


class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage


# BaseModel: automatic validation of input types
class GradeQeustion(BaseModel):
    """Boolean value to check wheter a question is related othe the topic"""

    score: str = Field(
        description="Question is aobut topic? If yes-> 'YES' if not -> 'NO'"
    )  # Field: add meta data


def question_rewriter(state: AgentState):
    print(f"Entering question_rewrite with the following state: {state}")

    # reset the state variable except for 'question' and 'messages'
    # every question goes from start to end: need to reset variables from previous question
    state["documents"] = []
    state["on_topic"] = ""
    state["rephrased_question"] = ""
    state["proceed_to_generate"] = False
    state["rephrase_count"] = 0

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if "question" not in state["messages"]:
        state["messages"].append(state["question"])

    if len(state["messages"]) > 1:
        # might want to rephrase the question
        conversation = state["essages"][
            :-1
        ]  # extract everything other than the last message
        current_question = state["question"].content
        messages = [
            SystemMessage(
                content="You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval"
            )
        ]
        messages.extend(conversation)  # past conversation history
        messages.append(HumanMessage(content=current_question))  # latest user question
        rephrase_prompt = ChatPromptTemplate.from_messages(messages)
        llm = ChatOpenAI(model="gpt-4o-mini")
        prompt = rephrase_prompt.format()
        response = llm.invoke(prompt)
        better_question = response.content.strip()
        print(f"question_rewriter: Rephrased question: {better_question}")
        state["rephrased_question"] = better_question
    else:  # first question
        state["rephrased_question"] = state["question"].content
    return state


def question_classifier(state: AgentState):
    print("Entering question_classifier")
    system_message = SystemMessage(
        content="""You are a classifier that determines whether a user's question is about one of the following topics
    1. topic one
    2. topic two
    3. topic three
    
    If the quetion IS about any of these  topics, respond with 'Yes'. Otherwise, respond 'No'
    """
    )

    human_message = HumanMessage(
        content=f"User question: {state['rephrased_question']}"
    )
    grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    llm = ChatOpenAI(model="gpt-4o")
    structured_llm = llm.with_structured_output(
        GradeQeustion
    )  # output matches gradequestion schema
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({})
    state["on_topic"] = result.score.strip()
    print(f"question_classifier: on_topic = {state['on_topic']}")
    return state


def on_topic_router(state: AgentState):
    print("Entering on_topic_router")
    on_topic = state.get("on_topic", "").strip().lower()
    if on_topic.lower() == "yes":
        print("routing to retrieve")
        return "retrieve"
    else:
        print("routing to off_topic_response")
        return "off_topic_response"


def retrieve(state: AgentState):
    print("Entering retrieve")
    documents = retriever.invoke(state["rephrased_question"])
    print(f"retrieve: Retrieved {len(documents)} documents")
    state["documents"] = documents
    return state


class GradeDocument(BaseModel):
    """Boolean value to check wheter the document is related to the question"""

    score: str = Field(
        description="Document is relavent to the question? If yes-> 'YES' if not -> 'NO'"
    )  # Field: add meta data


def retrieval_grader(state: AgentState):
    print("Entering retrieval_grader")
    system_message = SystemMessage(
        content="""You are a grader assessing the relavence of the retrieved document to a user question. Only Answer with 'YES' or 'NO'.
        If the document contains information relavent to the user's question, respond with 'Yes'. Otherwise, respond 'No'
    """
    )

    llm = ChatOpenAI(model="gpt-4o")
    structured_llm = llm.with_structured_output(
        GradeDocument
    )  # output matches GradeDocument schema
    relavent_docs = []
    for doc in state["documents"]:
        human_message = HumanMessage(
            content=f"User question: {state['rephrased_question']}\n\nRetrieved document \n{doc.page_conent}"
        )
        grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        grader_llm = grade_prompt | structured_llm
        result = grader_llm.invoke({})
        state["on_topic"] = result.score.strip()
        print(
            f"Grading document: {doc.page_content[:30]}...Result: {result.score.strip()}"
        )
        if result.score.strip().lower() == "yes":
            relavent_docs.append(docs)
    state["documents"] = relavent_docs
    state["proceed_to_generate"] = (
        len(relavent_docs) > 0
    )  # if there are no relavent docs, do not generate
    print(f"retrieval_grader: proceed_to_generate = {state['proceed_to_generate']}")
    return state


def proceed_router(state: AgentState):
    print("Entering proceed_router")
    rephrase_count = state.get("rephrase_count", 0)
    if state.get("proceed_to_generate", False):  # if None default to False
        print("Routing to generate_answer")
        return "generate_answer"
    elif rephrase_count >= 2:
        print("Maximum rephrase attempts reached. Cannot find relavent docs")
        return "cannt_answer"
    else:
        print("Routing to refine_question")
        return "refine_question"


def refine_question(state: AgentState):
    print("Entering refine_question")
    rephrase_count = state.get("rephrase_count", 0)
    if rephrase_count >= 2:
        print("Maximum rephrase attempts reached. Cannot find relavent docs")
        return state
    question_to_refine = state["rephrased_question"]
    system_message = SystemMessage(
        "You are a helpful assistant that slightly refines the user's question to improve retrieval results. Provide a slightly adjusted version of this question"
    )
    human_message = HumanMessage(
        content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question"
    )
    refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    llm = ChatOpenAI(model="gpt-4o")
    prompt = refine_prompt.format()  # returns a ChatPromptValue object (not just a string), which can be passed to llm.invoke()
    response = llm.invoke(prompt)
    refined_question = response.content.strip()
    print(f"refine_question: Refined question: {refined_question}")
    state["rephrased_question"] = refine_question
    state["rephrase_count"] = rephrase_count + 1
    return state


def generate_answer(state: AgentState):
    print("Entering generate_answer")
    if "messages" not in state or state["messages"] is None:
        raise ValueError("State must include 'messages' before generating an anaswer.")
    history = state["messages"]
    documents = state["documents"]
    rephrased_question = state["rephrased_question"]

    response = rag_chain.invoke(
        {"history": history, "context": documents, "question": rephrased_question}
    )

    generation = response.content.strip()

    state["messages"].append(AIMessage(content=generation))
    print(f"generate_answer: Generated Response: {generation}")
    return state


def cannot_answer(state: AgentState):
    print("Entering cannot_answer")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(
        AIMessage(
            content="I'm sorry, but I cannot find the information you're looking for"
        )
    )
    return state


def off_topic_response(state: AgentState):
    print("Entering off_topic_response")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(
        AIMessage(content="I'm sorry, I cannot answer this question")
    )
    return state


checkpointer = MemorySaver()


worklow = StateGraph(AgentState)
worklow.add_node("question_rewriter", question_rewriter)
worklow.add_node("question_classifier", question_classifier)
worklow.add_node("off_topic_response", off_topic_response)
worklow.add_node("retrieve", retrieve)
worklow.add_node("retrieval_grader", retrieval_grader)
worklow.add_node("generate_answer", generate_answer)
worklow.add_node("refine_question", refine_question)
worklow.add_node("cannot_answer", cannot_answer)

worklow.add_edge("question_rewriter", "question_classifier")
worklow.add_conditional_edges(
    "question_classifier",
    on_topic_router,
    {"retrieve": "retrieve", "off_topic_response": "off_topic_response"},
)
worklow.add_edge("refine_question", "retrieve")
worklow.add_edge("generate_answer", END)
worklow.add_edge("cannot_answer", END)
worklow.add_edge("off_topic_response", END)
worklow.set_entry_point("question_rewriter")

graph = worklow.compile(checkpointer=checkpointer)


"""
input_data = {
    "question": HumanMessage(
        content="Ask question here"
    )
}
graph.invoke(input=input_data, config={"configurable" : {"thread_id" : 1}})
"""
