import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv() # Load environment variables from .env file


token = os.getenv("SECRET") # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"


loader = WebBaseLoader(
    web_path=('https://en.wikipedia.org/wiki/Kėdainiai',),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(id="mw-content-text")
    ),
)

docs = loader.load()

text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
splits = text_spliter.split_documents(docs)


vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://models.inference.ai.azure.com",
    api_key=token,
))

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)



st.title("Streamlit Langchain Demo")

def generate_response(input_text):
    llm = ChatOpenAI(base_url=endpoint, temperature=0.7, api_key=token, model=model)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )      
    st.info(rag_chain.invoke(input_text))

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)