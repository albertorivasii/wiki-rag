from utils.emebdding_utils import *
from utils.qdrant_helpers import *
from utils.llm import *

query= "Why is Artificial Intelligence useful?"

model= get_model()
query_vector= embed_text([query], model)[0]

client= ConnectToQuadrant()

results= search_qdrant(client, "wiki_chunks", query_vector, limit=5)

mistral= load_mistral(r"C:\Users\thesm\Documents\Personal Website\RAG Project\models\mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_gpu_layers=20, n_ctx=2048)

answer= run_llm(mistral, results, query)

print("Context:\n", results)

print("Answer:\n", answer)

