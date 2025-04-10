from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import os
import uuid
from typing import List

load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
os.environ.pop("SSL_CERT_FILE", None)

def ConnectToQdrant():
    return QdrantClient(url=QDRANT_HOST,
                        api_key=QDRANT_API_KEY
                        )


def topic_exists(client: QdrantClient, colName: str, topic: str) -> bool:
    """
    Check if a topic exists in the Qdrant collection.

    Args:
        client (QdrantClient): Qdrant client instance.
        colName (str): Name of the collection to check.
        topic (str): Topic to check for existence.

    Returns:
        bool: True if the topic exists, False otherwise.
    """
    results = client.search(
        collection_name=colName,
        query_vector=[0.0] * 384,  # Dummy vector (required but unused)
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="topic",
                    match=MatchValue(value=topic)
                )
            ]
        ),
        limit=1,
    )
    return len(results) > 0

def create_collection(client:QdrantClient, colName:str, vector_dim:int):
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
            collection_name=colName,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        print(f"Collection '{colName}' created with vector dimension {vector_dim}.")
    else:
        print(f"Collection '{colName}' already exists.")


def upsert_embeddings(client:QdrantClient, colName:str, embeddings:list, texts:list[str], topic: str):

    assert len(embeddings) == len(texts), "Embeddings and texts must have the same length."

    points= []
    for i, (embedding, chunk) in enumerate(zip(embeddings, texts)):
        point= PointStruct(
            id=str(uuid.uuid4()),  # Generate a unique ID for each point
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


def search_qdrant(client:QdrantClient, colName:str, query_vector:List[float], topic, limit:int=5):
    results= client.search(
        collection_name=colName,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="topic",
                    match=MatchValue(value=topic)
                )
            ]
        ),
        limit=limit,
    )
    return results

