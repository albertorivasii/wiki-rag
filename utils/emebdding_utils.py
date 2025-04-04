from sentence_transformers import SentenceTransformer
from typing import List

def get_model(model_name='all-MiniLM-L6-v2'):
    model= SentenceTransformer(model_name)
    return model


def embed_text(texts:List[str], model:SentenceTransformer) -> List[List[float]]:
    """
    Embed a list of texts using the provided model.

    Args:
        texts (List[str]): List of texts to embed.
        model: The embedding model to use.

    Returns:
        List[List[float]]: List of embeddings for each text.
    """
    embeddings = model.encode(texts)
    return embeddings


# test the function
# if __name__ == "__main__":
#     model = get_model()
#     chunks = ["This is a test sentence.", "This is another test sentence."]
#     embeddings = embed_chunks(chunks, model)
#     print(embeddings)
#     print(len(embeddings))
#     print(len(embeddings[0]))