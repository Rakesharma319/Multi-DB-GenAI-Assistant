from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import os
### Working With Tools
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
## Graph
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document

from langgraph.graph import END, StateGraph, START

import streamlit as st

# Define the parameters
api_key = "123654789ghjm"
db_password = "jgvmsdbf11333"
st = "col1"

GROQ_API_KEY = st.sidebar.text_input("Enter groqu password",type="password")

# Run
# question = "List all movies by Imdb ratings , and sort by imdb rating ascending?" #--- GraphDB
# question = "Who is Virat Kohali?" #-- wiki
question = "visualize all albums by month?" # -- rdbms
# question = "write a quote to become richer, and aslo show tags and author?" #--- vector

## --------- Tools Code start -----

### 1. wikipedia start

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

### 1. wikipedia end

### 2. RDBMS_main start

def rdbms_main(api_key,db_password,question,st):
  v1 = api_key
  v2 = db_password
  v3 = question
  v4 = st
  return v1,v2,v3,v4
  
### 2. RDBMS_main End

### 3. Graph DB start

def graphDB_main(api_key,db_password,question,st):
  v1 = api_key
  v2 = db_password
  v3 = question
  v4 = st
  return v1,v2,v3,v4
  
### 3. Graph DB end

### 4. VectorStore(RAG application) start

def fun_retriver(api_key,db_password,question,st):
  v1 = api_key
  v2 = db_password
  v3 = question
  v4 = st
  return v1,v2,v3,v4
  
### 4. VectorStore(RAG application)  end


## ----- Lag Graph Start --------
# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["relationalDB","graphDB","vectorstore", "wikipedia"] = Field(
        ...,
        description="Given a user question choose to route it to relationalDB or graphDB or vectorDB or wikipedia.",
    )

# LLM with function call
from langchain_groq import ChatGroq

groq_api_key=GROQ_API_KEY
os.environ["GROQ_API_KEY"]=groq_api_key
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to relationalDB or graphDB or vectorDB or wikipedia.
relationalDB conatain data related to musics, albums, singer etc.
graphDB contains US movies data related to movies, acted_in, directors, Gener, IMDB ratings, title etc.
vectorDB contains data related to author, quotes and tags.
if user question is not related to above datasources then use wikipedia.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

# question = "List all movies by Imdb ratings , and sort by imdb rating ascending?"
# question = "Who is Virat Kohali?"
# question = "visualize all albums by month?"
question = "write a quote to become richer, and aslo show tags and author?"

print(
    question_router.invoke(
        {"question": question}
    )
)
# print(question_router.invoke({"question": "What are the types of agent memory?"}))

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    # documents = retriever.invoke(question)
    documents = fun_retriver("123654789ghjm_retriver","jgvmsdbf11333",question,"col1")
    return {"documents": documents, "question": question}

def rdbms_query(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RDBMS Query---")
    question = state["question"]

    # Retrieval
    # documents = retriever.invoke(question)
    # documents = rdbms_main("123654789ghjm_rdbms","jgvmsdbf11333",question,"col1")
    documents = rdbms_main(api_key,db_password,question,st)
    return {"documents": documents, "question": question}

def graphDB_query(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---Graph DB Query---")
    question = state["question"]

    # Retrieval
    # documents = retriever.invoke(question)
    # documents = graphDB_main("123654789ghjm_graphDB","jgvmsdbf11333",question,"col1")
    documents = graphDB_main(api_key,db_password,question,st)
    return {"documents": documents, "question": question}

def wiki_search(state):
    """
    wiki search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---wikipedia---")
    print("---HELLO--")
    question = state["question"]
    print(question)

    # Wiki search
    docs = wiki.invoke({"query": question})
    #print(docs["summary"])
    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)

    return {"documents": wiki_results, "question": question}

### Edges ###
def route_question(state):
    """
    Route question to wiki search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wikipedia":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wikipedia"
    elif source.datasource == "relationalDB":
        print("---ROUTE QUESTION TO RDBMS---")
        return "relationalDB"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    elif source.datasource == "graphDB":
        print("---ROUTE QUESTION TO GRAPHDB---")
        return "graphDB"
    else:
        raise ValueError(f"Unknown datasource: {source.datasource}")


workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("wiki_search", wiki_search)  # web search
workflow.add_node("rdbms_query", rdbms_query)
workflow.add_node("graphDB_query", graphDB_query)
workflow.add_node("retrieve", retrieve)

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wikipedia": "wiki_search",
        "vectorstore": "retrieve",
        "relationalDB": "rdbms_query",
        "graphDB": "graphDB_query",
    },
)
workflow.add_edge( "retrieve", END)
workflow.add_edge( "wiki_search", END)
workflow.add_edge( "rdbms_query", END)
workflow.add_edge( "graphDB_query", END)
# Compile
app = workflow.compile()

inputs = {
    "question": question
}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        st.write(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    st.write("\n---\n")

# Final generation
# pprint(value['documents'][0].dict()['metadata']['description'])
st.write(value['documents'])
