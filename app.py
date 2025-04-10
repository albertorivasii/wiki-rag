# front-end ui with streamlit
import streamlit as st
from utils.qdrant_helpers import ConnectToQdrant
from utils.llm import load_mistral
from utils.emebdding_utils import get_model
from rag.rag_pipeline import rag_pipeline
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
MODEL_PATH = os.environ.get("MODEL_PATH")

if "model_run" not in st.session_state:
    st.session_state["model_run"] = False

# caching reusable resources to improve performance
@st.cache_resource
def load_qdrant():
    return ConnectToQdrant()

@st.cache_resource
def load_llm():
    return load_mistral(model_path=MODEL_PATH, n_gpu_layers=28, n_ctx=2048)

@st.cache_resource
def embedding_model():
    return get_model()

embedding_llm = embedding_model()
client = load_qdrant()
llm = load_llm()

st.title("WikiGPT: A Chatbot for Wikipedia")

st.write("This is a chatbot that can answer questions based on Wikipedia articles.")
st.write("You can ask it any question, and it will try to find the answer in Wikipedia.")
st.write("Please enter your question below:")

# helper to identify the correct wiki page.
# future iterations will not use this. Will use keyword extraction to find relevant topics.
topic= st.text_input("Topic: ", key="topic")

# acts as the query vector.
query= st.text_input("Question: ", key="question")

submitButton= st.button("Submit", key="submit")

if submitButton:
    st.session_state["model_run"] = True

    answer, results= rag_pipeline(embedding_llm, client, llm, topic, query, limit=5)
    
    # display context in df format
    data= [
        {
            "Text": hit.payload["text"],
            "Score": hit.score
        }
    for hit in results]

    df= pd.DataFrame(data)
    df= df.sort_values(by="Score", ascending=False)

    # display answer
    st.write("Answer: ")
    st.write(answer)
    st.markdown("### Top Matching Results:")
    st.dataframe(df, use_container_width=True, hide_index=True)

