from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
import os
import getpass

app = FastAPI()

# Set API key
#os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")  Set this as environment variable 

# Initialize Google Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=  ********************************
)

# Define global variables
parsed_docs = None
vector_store = None

# Define models for request payloads
class LoadRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str

def fetch_and_parse_content(url: str):
    loader = WebBaseLoader([url])
    docs = loader.load()
    
    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)
    
    return all_splits

def create_vector_store(docs):
    global vector_store
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    connection_args = {
        "host": "localhost",
        "port": "19530"
    }

    COLLECTION_NAME = "my_collection"

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name=COLLECTION_NAME,
        drop_old=True
    ).from_documents(
        docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args=connection_args
    )

def process_query(question: str):
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No content loaded. Please load content first.")
    
    retriever = vector_store.as_retriever()
    
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
    
    return rag_chain.invoke(question)

@app.post("/load")
async def load_content(request: LoadRequest):
    global parsed_docs
    parsed_docs = fetch_and_parse_content(request.url)
    create_vector_store(parsed_docs)
    return {"message": "Content loaded successfully"}

@app.post("/query")
async def query(request: QueryRequest):
    if parsed_docs is None:
        raise HTTPException(status_code=400, detail="No content loaded. Please load content first.")
    
    response = process_query(request.question)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
