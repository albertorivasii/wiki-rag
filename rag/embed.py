from sentence_transformers import SentenceTransformer

def get_model(model_name='all-MiniLM-L6-v2'):
    model= SentenceTransformer(model_name)
    return model


def embed_chunks(chunks, model):

    return model.encode(chunks, show_progress_bar=True).tolist()


# test the function
if __name__ == "__main__":
    model = get_model()
    chunks = ["This is a test sentence.", "This is another test sentence."]
    embeddings = embed_chunks(chunks, model)
    print(embeddings)
    print(len(embeddings))
    print(len(embeddings[0]))