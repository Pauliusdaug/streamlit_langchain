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

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
#prompt = hub.pull("rlm/rag-prompt")
prompt = hub.pull("rlm/rag-prompt").partial(
    instructions=(
        "Before answering, determine if the question is ONLY about Kėdainiai based on the following three sources: "
        "1) Wikipedia article at https://en.wikipedia.org/wiki/Kėdainiai, "
        "2) website https://visitkedainiai.lt, "
        "3) PDF kedainiaiEN.pdf. "
        "If the question is NOT about Kėdainiai or the answer is NOT found exactly in these sources, "
        "do NOT answer or attempt to generate any information. "
        "Respond ONLY with this exact sentence: 'Your question is not valid. I can only answer about Kėdainiai.' "
        "Do NOT add anything else. "
        "If the question is about Kėdainiai, answer ONLY using these sources and clearly cite each piece of information's source."
    )
)




def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)



#st.title("Streamlit Langchain Kedainiai Chat Demo")
st.image("Kedainiai.jpg", use_container_width=True)
st.title("Streamlit Langchain Kedainiai Chat Demo")



def generate_response(input_text):
    llm = ChatOpenAI(base_url=endpoint, temperature=0.7, api_key=token, model=model)

    fetched_docs = retriever.invoke(input_text)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    ) 

    result = rag_chain.invoke(input_text)     
    st.markdown(
    f"<div style='background-color:#1e3a5f; color:#ffffff; padding:15px; border-radius:8px; margin-bottom:20px;'>"
    f"{result}</div>", 
    unsafe_allow_html=True
    )


    st.subheader("📚 Sources")
    for i, doc in enumerate(fetched_docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        with st.expander(f"Source {i} ({source})"):
            st.markdown(
                f"<div style='background-color:#e3f2fd; color:#000000; padding:15px; border-radius:8px; margin-bottom:20px;'>"
                f"{result}</div>", 
                unsafe_allow_html=True
    )




with st.form("my_form"):
    text = st.text_area(
        "I am Kedainiai expert:",
        "Ask me anything about Kėdainiai, Lithuania.",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
