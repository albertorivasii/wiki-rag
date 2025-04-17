import streamlit as st
from rag.rag_pipeline import *

st.set_page_config(page_title="RAG App", page_icon=":guardsman:", layout="wide")
st.title("RAG App")

# cache the model loading to improve performance
@st.cache_resource
def load_model():
    return load_mistral("")

