from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import requests
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# FastAPI instance
app = FastAPI()

# Configuration
GOOGLE_API_KEY = "your-google-api-key"  # You should use environment variables for sensitive data
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Data storage
parsed_docs = []
COLLECTION_NAME = "my_collection"

# Models
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

connection_args = {
    "host": "localhost",
    "port": "19530"
}

# Define the request models
class LoadRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str

# Fetch and parse content
def fetch_and_parse_content(url: str):
    loader = WebBaseLoader([url])
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    return text_splitter.split_documents(docs)

# Create vector store
def create_vector_store(docs):
    global parsed_docs
    parsed_docs = docs
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name=COLLECTION_NAME,
        drop_old=True
    ).from_documents(
        parsed_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args=connection_args
    )

# Define the endpoints
@app.post("/load")
async def load_content(request: LoadRequest):
    global parsed_docs
    parsed_docs = fetch_and_parse_content(request.url)
    create_vector_store(parsed_docs)
    return {"message": "Content loaded successfully"}

@app.post("/query")
async def query_content(request: QueryRequest):
    retriever = Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name=COLLECTION_NAME
    ).as_retriever()

    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    
    rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
    )

    answer = rag_chain.invoke(request.question)
    return {"answer": answer}

