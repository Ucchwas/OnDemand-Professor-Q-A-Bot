# import chainlit as cl

# @cl.on_message
# async def main(message: str):
#     # Your logic will be here
#     result = message.title()
#     await cl.send_message(content=f"Sure, here is your analysis: {result}")

import chainlit as cl
import os
import json
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, PyPDFLoader

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# Load documents and build Chroma vector store
def loadResourceDocuments():
    try:
        project_directory = "C:/Users/User/Downloads/OnDemand-Professor-Q-A-Bot"
        lecture_loader = DirectoryLoader(f"{project_directory}/Lectures", loader_cls=PyPDFLoader)
        documents = lecture_loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        texts = text_splitter.split_documents(documents)

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Chroma.from_documents(documents=texts, embedding=embedding, persist_directory="db")
        print("Resource documents loaded and vector store created successfully.")
    except Exception as e:
        print(f"Error loading documents: {e}")

# Query LLM
def llm(query):
    try:
        persist_directory = 'db'
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

        retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=HuggingFaceHub(repo_id="gpt2", model_kwargs={"max_length": 100}),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        llm_response = qa_chain(query)
        
        # Extract only the main answer and filter out redundant or irrelevant context
        answer = llm_response['result'].split("Helpful Answer: ")[-1].strip()  # Remove repeated prompt prefixes if present
        
        # Collect unique sources
        unique_sources = {json.dumps(doc.metadata) for doc in llm_response.get("source_documents", [])}
        sources = "\n".join(f"Source of the information: {source}" for source in unique_sources)
        
        # Format response with answer and source only
        response = f"{answer}\n\n{sources}" if sources else answer
        
        return response

    except Exception as e:
        print(f"Error during LLM query: {e}")
        return "An error occurred during processing."


# Initialize Chainlit message handling
@cl.on_message
async def main(message: cl.Message):
    response = llm(message.content)
    await cl.Message(content=response).send()

# Load resources at startup
loadResourceDocuments()



