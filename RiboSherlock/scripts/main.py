from typing import Annotated, Sequence, List, Literal 
from pydantic import BaseModel, Field 
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults 
from langgraph.types import Command 
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent 
from IPython.display import Image, display 
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
import pprint

load_dotenv(dotenv_path="/Users/michael/Documents/Bioinformatics/Personal_Projects/ML/Trenchware/LLM/input/.env")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

tavily_search = TavilySearchResults(max_results=2)

python_repl_tool = PythonREPLTool()


class Supervisor(BaseModel):
    next: Literal["enhancer", "request_grader", "researcher", "coder"] = Field(
        description="Determines which specialist to activate next in the workflow sequence: "
                    "'enhancer' when user input requires clarification, expansion, or refinement, "
                    "'request_grader' when user submits a request it checks for relevance to your purpose"
                    "'researcher' when additional facts, context, or data collection is necessary, "
                    "'coder' when implementation, computation, or technical problem-solving is required."
    )
    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
    )

def supervisor_node(state: MessagesState) -> Command[Literal["enhancer", "request_grader", "researcher", "coder"]]:

    system_prompt = ('''
                 
        You are a workflow supervisor specialized in Bulk RNAseq quality control managing a team of four specialized agents: Prompt Enhancer, Request Grader, Researcher, and Coder. Your role is to orchestrate the workflow by selecting the most appropriate next agent based on the current state and needs of the task. Provide a clear, concise rationale for each decision to ensure transparency in your decision-making process.

        **Team Members**:
        1. **Prompt Enhancer**: Always consider this agent first. They clarify ambiguous requests, improve poorly defined queries, and ensure the task is well-structured before deeper processing begins.
        2. **Request Grader** Ensures all requests are on topic and are about QC analysis of Bulk RNAseq data and related topics.
        3. **Researcher**: Specializes in information gathering, fact-finding, and collecting relevant data needed to address the user's request.
        4. **Coder**: Focuses on technical implementation, calculations, data analysis, algorithm development, and coding solutions.

        **Your Responsibilities**:
        1. Analyze each user request and agent response for completeness, accuracy, and relevance.
        2. Route the task to the most appropriate agent at each decision point.
        3. Maintain workflow momentum by avoiding redundant agent assignments.
        4. Continue the process until the user's request is fully and satisfactorily resolved.

        Your objective is to create an efficient workflow that leverages each agent's strengths while minimizing unnecessary steps, ultimately delivering complete and accurate solutions to user requests.
                 
    ''')
    
    messages = [
        {"role": "system", "content": system_prompt},  
    ] + state["messages"] 

    response = llm.with_structured_output(Supervisor).invoke(messages)

    goto = response.next
    reason = response.reason

    print(f"--- Workflow Transition: Supervisor → {goto.upper()} ---")
    
    return Command(
        update={
            "messages": [
                HumanMessage(content=reason, name="supervisor")
            ]
        },
        goto=goto,  
    )

def enhancer_node(state: MessagesState) -> Command[Literal["supervisor"]]:

    """
        Enhancer agent node that improves and clarifies user queries.
        Takes the original user input and transforms it into a more precise,
        actionable request before passing it to the supervisor.
    """
   
    system_prompt = (
        "You are a Query Refinement Specialist with expertise in transforming vague requests into precise instructions. Your responsibilities include:\n\n"
        "1. Analyzing the original query to identify key intent and requirements\n"
        "2. Resolving any ambiguities without requesting additional user input\n"
        "3. Expanding underdeveloped aspects of the query with reasonable assumptions\n"
        "4. Restructuring the query for clarity and actionability\n"
        "5. Ensuring all technical terminology is properly defined in context\n\n"
        "Important: Never ask questions back to the user. Instead, make informed assumptions and create the most comprehensive version of their request possible."
    )

    messages = [
        {"role": "system", "content": system_prompt},  
    ] + state["messages"]  

    enhanced_query = llm.invoke(messages)

    print(f"--- Workflow Transition: Prompt Enhancer → Supervisor ---")

    return Command(
        update={
            "messages": [  
                HumanMessage(
                    content=enhanced_query.content, 
                    name="enhancer"  
                )
            ]
        },
        goto="supervisor", #always go back to supervisor
    )

class GradeRequest(BaseModel):
    score: str = Field(
        description="Is the request about the topic? If yes -> 'yes' if not -> 'no'"
    )


# Request grader for RiboSherlock
    
def request_grader(state: MessagesState) -> Command[Literal["supervisor"]]:
    print("Entering Request Grader")

    system_prompt = (
        " You are a classifier that determines whether a user's question is about bulk RNA seq data analysis, Quality control, and related topics."
        "'If the question IS about that, respond with 'Yes'. Otherwise, respond with 'No'."
    )
    messages = [
        {"role": "system", "content": system_prompt},  
    ] + state["messages"]  

    grader_response = llm.with_structured_output(GradeRequest).invoke(messages)

    score = grader_response.score

    #print(f"--- Workflow Transition: Request Grader → Supervisor ---")

    if score.lower() == "no":
        print("Request Grader determined the request is off-topic. Ending workflow.")
        return Command(
            update={
                "messages": [
                    HumanMessage(content="This request is off-topic and not related to Bulk RNAseq QC. Ending workflow.", name="grader")
                ]
            },
            goto=END
        )

    else:
        print("--- Workflow Transition: Request Grader → Supervisor ---")
        return Command(
            update={
                "messages": [
                    HumanMessage(content="This request is about Bulk RNAseq QC. Continuing workflow.", name="grader")
                ]
            },
            goto="supervisor"
        )




def qc_researcher_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
        This node specializes in gathering information using Tavilyu search tool.
        It takes the current task state and performs relevant research and returns findings for validations.
    """

    research_agent =  create_react_agent(
        llm,
        tools=[tavily_search],
        state_modifier="You are an INFORMATION SPECIALIST with great expertise in comprehensive research. Your key responsibilities include: \n\n"
            "1. Identifying the key information needs based on the request or query context\n"
            "2. Gathering relevant, accurate, and up-to-date information from reliable sources\n"
            "3. Organizing findings in a structured, easily digestible format\n"
            "4. Citing sources when possible to establish credibility\n"
            "5. Focusing exclusively on information gathering - avoid analysis or implementation\n\n"
            "Provide thorough, factual responses without speculation where information is unavailable."
    )

    research_results = research_agent.invoke(state)

    print(f"--- RiboSherlock Workflow Transition: Researcher → Validator ---")

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=research_results["messages"][-1].content,
                    name="researcher"
                )
            ]
        },
        goto="validator"
    )

def code_node(state: MessagesState) -> Command[Literal["validator"]]:

    sherlock_coder_agent = create_react_agent(
        llm, 
        tools=[python_repl_tool],
        state_modifier=(
            "You are an expert coder and Quality Control analyst. Focus on mathematical calculations, analyzing, solving math questions, "
            "and execting code. Handle technical problem-solving and data tasks."
        )
    )

    results = sherlock_coder_agent.invoke(state)

    print(f"--- RiboSherlock Workflow Transition: Coder → Validator ---")

    return Command(
        update={
            "messages": [
                AIMessage(content=results["messages"][-1].content,
                          name="coder")
            ]
        },
        goto="validator"
    )


Validator_system_prompt = """
    Your purpose and task is to ensure reasonable quality.
    Specifically, you must:
    - Review the user's question (the first message in the workflow).
    - Review the answer (the last message in the workflow).
    - If the answer addresses the core intent of the question, even if not perfectly, signal to end the workflow with 'FINISH'.
    - Only route back to the supervisor if the answer is completely off-topic, harmful, or fundamentally misunderstands the question.
    
    - Accept answers that are "good enough" rather than perfect
    - Prioritize workflow completion over perfect responses
    - Give benefit of doubt to borderline answers
    
    Routing Guidelines:
    1. 'supervisor' Agent: ONLY for responses that are completely incorrect or off-topic.
    2. Respond with 'FINISH' in all other cases to end the workflow.
"""


class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Specifies the next worker in the pipeline: 'supervisor' to continue or 'FINISH' to terminate"
    )
    reason: str = Field(
        description="The reason for the decision."
    )

def validator_node(state: MessagesState) -> Command[Literal["supervisor", "__end__"]]:

    user_question = state["messages"][0].content
    agent_answer = state["messages"][-1].content

    messages = [
        {"role": "system", "content": Validator_system_prompt},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": agent_answer},
    ]

    response = llm.with_structured_output(Validator).invoke(messages)

    goto=response.next
    reason=response.reason

    if goto == "FINISH" or goto == END:
        goto = END
        print(" --- Transitioning to End of workflow --- ")
    
    else:
        print(" --- RiboSherlock Workflow Transition: Validator -> Supervisor --- ")

    return Command(
        update={
            "messages": [
                HumanMessage(content=reason, name="validator")
            ]
        },
        goto=goto
    )

graph = StateGraph(MessagesState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("request_grader", request_grader)
graph.add_node("researcher", qc_researcher_node)
graph.add_node("coder", code_node)
graph.add_node("validator", validator_node)
graph.add_node("enhancer", enhancer_node)

graph.add_edge(START, "supervisor")
app = graph.compile()


final_user_prompt = input("How can RiboSherlock Help you today? ")

inputs = {
    "messages": [
        ("user", final_user_prompt),
    ]
}

for event in app.stream(inputs):
    for key, value in event.items():
        if value is None:
            continue
        last_message = value.get("messages", [])[-1] if "messages" in value else None
        if last_message:
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint(last_message, indent=2, width=80, depth=None)
            print()