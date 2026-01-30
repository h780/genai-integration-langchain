import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_neo4j import Neo4jGraph

# Initialize the LLM
model = init_chat_model(
    "gpt-4o",
    model_provider="openai"
)

cypher_model = init_chat_model(
    "gpt-4o",
    model_provider="openai",
    temperature=0.0
)

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
)

# Cypher template
from langchain_core.prompts.prompt import PromptTemplate
cypher_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, for example "The 39 Steps" becomes "39 Steps, The".

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

# Create the Cypher QA chain
from langchain_neo4j import GraphCypherQAChain
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=cypher_model,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True,
    verbose=True,
)

# Invoke the chain
question = "What is the highest user rated movie in the Horror genre?"
response = cypher_qa.invoke({"query": question})
print(response["result"])

question = "Who acted in the movie Aliens?"
response = cypher_qa.invoke({"query": question})
print(response["result"])

question = "Who directed the movie Superman?"
response = cypher_qa.invoke({"query": question})
print(response["result"])

question = "What is the plot of the movie Toy Story?"
response = cypher_qa.invoke({"query": question})
print(response["result"])