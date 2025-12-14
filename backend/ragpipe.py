import os
from langchain_core.documents import Document
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import chromadb
from typing import List

from dotenv import load_dotenv
load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

def setAllConfig(embedName: str, apiversion: str, apikey: str, collectionname: str, azureEndpoint: str):
    print("**********************************************")
    print(embedName, apiversion, apikey, collectionname)
    print(os.environ.get("AZURE_OPENAI_API_KEY"))
    print("**********************************************")
    embeddings = AzureOpenAIEmbeddings(
        deployment=embedName,
        model="text-embedding-3-small",
        openai_api_version=apiversion,
        api_key=apikey,
        azure_endpoint=azureEndpoint
    )
    
    vector_store = Chroma(
        collection_name=collectionname,
        embedding_function=embeddings,
        chroma_cloud_api_key="ck-3jRzkSTiS6DagyR7QPtgTz67cjgw11aZjV85WstutzWn",
        tenant="1860447b-3aa0-4821-9244-98baa592b7a4",
        database="dev",
        create_collection_if_not_exists=True,
    )
    return embeddings, vector_store


def get_loader(file_path: str):
    return PyPDFLoader(file_path)    

def process_document_and_index(file_path: str, embedName: str, apiversion: str, apikey: str, collectionname: str, azureEndpoint: str) -> int:
    embeddings, vector_store = setAllConfig(embedName, apiversion, apikey, collectionname, azureEndpoint)
    loader = get_loader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata['source_file'] = file_path
        doc.metadata['timestamp'] = os.path.getmtime(file_path) # Example metadata

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(documents)
    print(f"Indexing {len(all_splits)} chunks from {file_path}")
    vector_store.add_documents(all_splits)
    
    return len(all_splits)

def index_documents(file_paths: List[str], embedName: str, apiversion: str, apikey: str, collectionname: str, azureEndpoint: str) -> None:
    for path in file_paths:
        process_document_and_index(path, embedName, apiversion, apikey, collectionname, azureEndpoint)
        
def query_qa(question: str, embedName: str, apiversion: str, apikey: str, collectionname: str, chatname: str, endpoint: str, k: int = 4) -> Tuple[str, List[dict]]:
    embeddings, vector_store = setAllConfig(embedName, apiversion, apikey, collectionname, endpoint)
    docs: List[Document] = vector_store.similarity_search(question, k=k)
    
    if not docs:
        return "No documents indexed or no relevant results.", []

    context = "\n\n".join([d.page_content for d in docs])
    sources = [d.metadata for d in docs]
    print(chatname)
    print(embedName)
    llm = AzureChatOpenAI(azure_endpoint= endpoint, api_key= apikey, azure_deployment= chatname, api_version= apiversion, temperature=0)
    prompt = f"Use the following context to answer the question. If the answer is not contained, say you don't know.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer concisely and cite sources by filename."

    resp = llm.invoke(prompt)
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return answer, sources
