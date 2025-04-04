import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv
from typing import List


# Load environment variables from .env file and get credentials
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")


def ConnectToQuadrant():
    return QdrantClient(url=QDRANT_HOST,
                        api_key=QDRANT_API_KEY
                        )


def create_collection(client:QdrantClient, colName:str, vector_dim:int, distance_metric:str="Cosine"):
    """
    Create a collection if it does not exist in Qdrant yet.

    Args
    client (QdrantClient): Qdrant client instance.
    colName (str): Name of the collection to be created.
    vector_dim (int): Dimension of the vectors to be stored in the collection.
    distance_metric (str): Distance metric to be used for vector similarity search. Default is "Cosine".

    Returns
    None
    """

    # check if the collection already exists
    if not client.collection_exists(colName):
        client.recreate_collection(
            collection=colName,
            vectors_config=VectorParams(size=vector_dim, distance=distance_metric),
        )
        print(f"Collection '{colName}' created with vector dimension {vector_dim} and distance metric '{distance_metric}'.")
    else:
        print(f"Collection '{colName}' already exists.")


def upsert_embeddings(client:QdrantClient, colName:str, embeddings:list, texts:list[str], topic: str):

    assert len(embeddings) == len(texts), "Embeddings and texts must have the same length."

    points= []
    for i, (embedding, chunk) in enumerate(zip(embeddings, texts)):
        point= PointStruct(
            id=uuid.uuid4(),  # Generate a unique ID for each point
            vector=embedding,  # The embedding vector
            payload={"text": chunk,
                     "topic": topic,
                     "chunk_id":i
                     }  # Additional metadata (text and topic)
        )
        points.append(point)
    
    # Upsert the points into the collection
    client.upsert(
        collection_name=colName,
        points=points
    )
    print(f"Upserted {len(points)} points into collection '{colName}'.")