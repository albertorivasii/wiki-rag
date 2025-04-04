from utils.emebdding_utils import *
from utils.qdrant_helpers import *
from utils.llm import run_llm

query= input("Ask Anything: ")

model= get_model()
query_vector= model.endcode([query])[0].tolist()

client= ConnectToQuadrant()

results= search_qdrant(client, "wiki_chunks", query_vector, limit=5)

llm_context= "\n".join([result.payload["text"] for result in results])

prompt= f"""
<s>
[INST]
Answer the question based on the context provided below. If the answer is not in the context, say "I don't know".
Context:
{llm_context}
Question: {query}
Answer:
[\INST]
"""

response= run_llm(prompt)

print("Answer:\n", response)