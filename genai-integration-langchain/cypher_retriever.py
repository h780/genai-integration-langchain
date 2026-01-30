import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from langchain_neo4j import Neo4jGraph

# Initialize the LLM
model = init_chat_model("gpt-4o", model_provider="openai")

# Create a prompt
cypher_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, for example "The 39 Steps" becomes "39 Steps, The".
Exclude NULL values when finding the highest value of a property.

Schema:
{schema}
Examples:
1. Question: Get user ratings?
   Cypher: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE u.name = "User name" RETURN r.rating AS userRating
2. Question: Get average rating for a movie?
   Cypher: MATCH (m:Movie)<-[r:RATED]-(u:User) WHERE m.title = 'Movie Title' RETURN avg(r.rating) AS userRating
3. Question: Get movies for a genre?
   Cypher: MATCH ((m:Movie)-[:IN_GENRE]->(g:Genre) WHERE g.name = 'Genre Name' RETURN m.title AS movieTitle

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=cypher_template
)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

# Define state for application
class State(TypedDict):
    question: str
    context: List[dict]
    answer: str

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
)

# Create the Cypher QA chain
from langchain_neo4j import GraphCypherQAChain
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=model,
    allow_dangerous_requests=True,
    return_direct=True,
    verbose=True
)

# Define functions for each step in the application

# Retrieve context
def retrieve(state: State):
    context = cypher_qa.invoke(
        {"query": state["question"]}
    )
    return {"context": context}

# Generate the answer based on the question and context
def generate(state: State):
    messages = prompt.invoke({"question": state["question"], "context": state["context"]})
    response = model.invoke(messages)
    return {"answer": response.content}

# Define application steps
workflow = StateGraph(State).add_sequence([retrieve, generate])
workflow.add_edge(START, "retrieve")
app = workflow.compile()

# Run the application
question = "What movies has Tom Hanks acted in?"
response = app.invoke({"question": question})
print("Answer:", response["answer"])
print("Context:", response["context"])

question = "How much money did the movie Forrest Gump make at the box office?"
response = app.invoke({"question": question})
print("Answer:", response["answer"])
print("Context:", response["context"])

question = "How long is the movie Clueless?"
response = app.invoke({"question": question})
print("Answer:", response["answer"])
print("Context:", response["context"])

question = "What languages are spoken in the movie GoldenEye"
response = app.invoke({"question": question})
print("Answer:", response["answer"])
print("Context:", response["context"])