import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#LangSmith Tracing
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

#PromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistance. Please respond to the questions asked"),
        ("user", "Question:{question}")
    ]
)

#streamlit Framework
st.title("Langchain Demo with Gemma2")
inputText = st.text_input("Prompt here!")

#Ollama model
llm = Ollama(model="gemma2:2b")
output_parse = StrOutputParser()
chain = prompt|llm|output_parse

if inputText:
    st.write(chain.invoke({"question":inputText}))