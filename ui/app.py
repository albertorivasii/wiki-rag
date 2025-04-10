# front-end ui with streamlit
import streamlit as st


if "model_run" not in st.session_state:
    st.session_state["model_run"] = False

st.title("WikiGPT: A Chatbot for Wikipedia")

st.write("This is a chatbot that can answer questions based on Wikipedia articles.")
st.write("You can ask it any question, and it will try to find the answer in Wikipedia.")
st.write("Please enter your question below:")

st.text_input("Question: ", key="question")

submitButton= st.button("Submit", key="submit")

if submitButton:
    st.session_state["model_run"] = True

    # call model


    # display answer
    st.write("Answer: ")
    st.write("This is where the answer will be displayed.")
    st.write("Context: ")
    st.write("This is where the context will be displayed.")



