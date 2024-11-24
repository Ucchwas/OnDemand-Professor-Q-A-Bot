# import chainlit as cl

# @cl.on_message
# async def main(message: str):
#     # Your logic will be here
#     result = message.title()
#     await cl.send_message(content=f"Sure, here is your analysis: {result}")

import chainlit as cl
import os
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

os.environ["OPENAI_API_KEY"] = ""

def loadResourceDocuments():
    try:
        project_directory = "C:/Users/User/Downloads/OnDemand-Professor-Q-A-Bot"
        lecture_loader = DirectoryLoader(f"{project_directory}/Lectures", loader_cls=PyPDFLoader)
        documents = lecture_loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        texts = text_splitter.split_documents(documents)

        embedding = OpenAIEmbeddings()
        Chroma.from_documents(documents=texts, embedding=embedding, persist_directory="db")
        print("Resource documents loaded and vector store created successfully.")
    except Exception as e:
        print(f"Error loading documents: {e}")

# Query LLM
def llm(query):
    try:
        persist_directory = 'db'
        embedding = OpenAIEmbeddings()
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

        retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        llm_response = qa_chain(query)
        
        answer = llm_response['result']
        
        source_documents = llm_response.get("source_documents", [])
        
        if source_documents:
            unique_sources = set()
            for doc in source_documents:
                metadata = doc.metadata.copy()
                if "page" in metadata:
                    metadata["page"] = metadata["page"] + 1
                unique_sources.add(json.dumps(metadata))
            
            sources = "\n".join(f"Source of the information: {source}" for source in unique_sources)
            response = f"{answer}\n\n{sources}"
        else:
            response = answer
        
        return response

    except Exception as e:
        print(f"Error during LLM query: {e}")
        return "An error occurred during processing."



@cl.on_message
async def main(message: cl.Message):
    response = llm(message.content)
    await cl.Message(content=response).send()

loadResourceDocuments()






