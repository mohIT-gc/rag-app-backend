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

def setConfig():
    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("OPENAI_EMBED_DEPLOYMENT"),
        model="text-embedding-3-small",
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    vector_store = Chroma(
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag-collection"),
        embedding_function=embeddings,
        chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
        create_collection_if_not_exists=True,
    )
    return embeddings, vector_store


def get_loader(file_path: str):
    return PyPDFLoader(file_path)    

def process_document_and_index(file_path: str) -> int:
    embeddings, vector_store = setConfig()
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

def index_documents(file_paths: List[str]) -> None:
    for path in file_paths:
        process_document_and_index(path)
        
def query_qa(question: str, k: int = 4) -> Tuple[str, List[dict]]:
    embeddings, vector_store = setConfig()
    docs: List[Document] = vector_store.similarity_search(question, k=k)
    
    if not docs:
        return "No documents indexed or no relevant results.", []

    context = "\n\n".join([d.page_content for d in docs])
    sources = [d.metadata for d in docs]

    llm = AzureChatOpenAI(azure_deployment= os.getenv("OPENAI_CHAT_DEPLOYMENT"), api_version= os.getenv("OPENAI_CHAT_API_VERSION"),temperature=0)
    prompt = f"Use the following context to answer the question. If the answer is not contained, say you don't know.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer concisely and cite sources by filename."

    resp = llm.invoke(prompt)
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return answer, sources
