from utils.emebdding_utils import get_model, embed_chunks
from utils.chunking import chunk_text, count_tokens
from utils.wikipedia_api import fetch_content, clean_text
from utils.qdrant_helpers import *
import os

# Setup
topic = "Artificial Intelligence"
collection_name = "wiki_chunks"
max_tokens = 512

# 1. Fetch Wikipedia content
raw_text = fetch_content(topic)
if not raw_text:
    print(f"❌ Could not fetch content for topic: {topic}")
    exit()

# 2. Clean + Chunk
cleaned = clean_text(raw_text)
model = get_model()
tokenizer = model.tokenizer  # get the tokenizer from MiniLM

chunks = chunk_text(cleaned, tokenizer, max_tokens=max_tokens)
print(f"✅ Chunked into {len(chunks)} sections")

# 3. Embed
vectors = embed_chunks(chunks, model)

# 4. Connect to Qdrant and upsert
client = ConnectToQuadrant()
create_collection(client, collection_name, vector_dim=len(vectors[0]))
upsert_embeddings(client, collection_name, vectors, chunks, topic)
