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
from langchain_community.document_loaders import PyPDFLoader



load_dotenv() # Load environment variables from .env file


token = os.getenv("SECRET") # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

# Load each source separately and tag them
loader1 = WebBaseLoader("https://en.wikipedia.org/wiki/Kėdainiai")
docs1 = loader1.load()
for doc in docs1:
    doc.metadata["source"] = "Wikipedia"

loader2 = WebBaseLoader("https://visitkedainiai.lt")
docs2 = loader2.load()
for doc in docs2:
    doc.metadata["source"] = "VisitKedainiai"

from langchain_community.document_loaders import PyPDFLoader
pdf_loader = PyPDFLoader("kedainiaiEN.pdf")
docs3 = pdf_loader.load()
for doc in docs3:
    doc.metadata["source"] = "Kedainiai PDF"

# Combine all with metadata preserved
docs = docs1 + docs2 + docs3

text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
splits = text_spliter.split_documents(docs)


vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://models.inference.ai.azure.com",
    api_key=token,
))

retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
#prompt = hub.pull("rlm/rag-prompt")
prompt = hub.pull("rlm/rag-prompt").partial(
    instructions=(
        "You are an assistant who ONLY answers questions using information from the following sources: "
        "1) the Wikipedia article at https://en.wikipedia.org/wiki/Kėdainiai, "
        "2) the website https://visitkedainiai.lt, and "
        "3) the PDF document kedainiaiEN.pdf. "
        "These are your ONLY sources. "
        "When you provide your answer, clearly state which source each piece of information comes from. "
        "If the question is not about Kėdainiai, or the answer is NOT found exactly in these sources, "
        "respond ONLY with this exact sentence: 'Your question is not valid. I can only answer about Kėdainiai.' "
        "Do NOT add anything else, no explanations, no apologies, no extra sentences."
    )
)







def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)



st.title("Streamlit Langchain Demo")

def generate_response(input_text):
    llm = ChatOpenAI(base_url=endpoint, temperature=0.7, api_key=token, model=model)

    #fetched_docs = vectorstore. search(input_text, search_type="similarity", k=3)
    fetched_docs = retriever.invoke(input_text)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    ) 

    result = rag_chain.invoke(input_text)     
    st.info(result)

    st.subheader("📚 Sources")
    for i, doc in enumerate(fetched_docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        with st.expander(f"Source {i} ({source})"):
            st.write(f"**Content:** {doc.page_content}")

with st.form("my_form"):
    text = st.text_area(
        "I am Kedainiai expert:",
        "Ask me anything about Kėdainiai, Lithuania.",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
