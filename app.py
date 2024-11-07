# import chainlit as cl

# @cl.on_message
# async def main(message: str):
#     # Your logic will be here
#     result = message.title()
#     await cl.send_message(content=f"Sure, here is your analysis: {result}")

import chainlit as cl
import os
import time
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # Correct import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

@cl.on_message
async def main(message: cl.Message):
    response = llm(message.content)
    await cl.Message(content=response).send()

# Define the LLM function that handles the query and returns the response
def llm(query):
    # Configure Chroma vector store directory and embedding model
    persist_directory = 'db'
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load persisted Chroma vector store
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # Set up the retriever with top-k document retrieval
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    
    # Initialize the Hugging Face Hub language model and RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=HuggingFaceHub(repo_id="gpt2", model_kwargs={"max_length": 100}),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Get the response and handle the case when no documents are retrieved
    llm_response = qa_chain(query)
    if not llm_response["source_documents"]:
        response = llm_response['result'] + "\n\nNo relevant sources found."
    else:
        response = (
            llm_response['result'] + 
            "\n\nSource of the information is: " + 
            json.dumps(llm_response["source_documents"][0].metadata)
        )

    print(llm_response)
    return response

# Define function to load and process resource documents
def loadResourceDocuments():
    documents = []

    # Set directory paths for document loaders
    project_directory = "C:/Users/User/OneDrive - Texas Tech University/Desktop/Fall- 24/CS 5342/Project"
    lecture_loader = DirectoryLoader(f"{project_directory}/Lectures", loader_cls=PyPDFLoader)
    
    # Load and split documents from the Lectures folder
    documents.extend(lecture_loader.load_and_split())
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)

    # Initialize Chroma with Hugging Face embeddings and automatically persist data
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_documents(documents=texts, embedding=embedding, persist_directory="db")

# Run the function to load and process documents
loadResourceDocuments()

