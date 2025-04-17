from sentence_transformers import SentenceTransformer
from typing import List
from utils.wikipedia_api import *
from utils.chunking import *

def get_model(model_name='all-MiniLM-L6-v2'):
    model= SentenceTransformer(model_name)
    return model


def embed_wiki_content(topic:str, model:SentenceTransformer) -> List[List[float]]:
    """
    Embed a list of texts using the provided model.

    Args:
        texts (List[str]): List of texts to embed.
        model: The embedding model to use.

    Returns:
        List[List[float]]: List of embeddings for each text.
    """
    # get information from wikipedia
    topic= fetch_content(topic)
    topic= clean_text(topic)

    # split into chunks
    chunks= chunk_text(topic, model.tokenizer, max_tokens=512)

    # embed chunks
    embeddings = model.encode(chunks).tolist()
    return chunks, embeddings

def embed_text(texts:List[str], model:SentenceTransformer) -> List[List[float]]:
    """
    Embed a list of texts using the provided model.

    Args:
        texts (List[str]): List of texts to embed.
        model: The embedding model to use.

    Returns:
        List[List[float]]: List of embeddings for each text.
    """
    # embed chunks
    embeddings = model.encode(texts).tolist()
    return embeddings


# test the function
# if __name__ == "__main__":
#     model = get_model()
#     chunks = ["This is a test sentence.", "This is another test sentence."]
#     embeddings = embed_chunks(chunks, model)
#     print(embeddings)
#     print(len(embeddings))
#     print(len(embeddings[0]))