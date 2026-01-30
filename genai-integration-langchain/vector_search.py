import os
from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
)

# Create the embedding model
from langchain_openai import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create Vector
from langchain_neo4j import Neo4jVector
plot_vector = Neo4jVector.from_existing_index(embedding_model,
                                              graph=graph,
                                              index_name="moviePlots",
                                              embedding_node_property="plotEmbedding",
                                              text_node_property="plot",
                                            )
# Search for similar movie plots
plot = "Love conquers all"
result = plot_vector.similarity_search(plot, k=3)
print(result)

# Parse the documents
for doc in result:
    print(f"Title: {doc.metadata['title']}")
    print(f"Plot: {doc.page_content}\n")

result = plot_vector.similarity_search(
    plot,
    k=3,
    filter={"revenue": {"$gte": 200000000}}
)

print(result)