import os
from typing import List
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

PERSIST_DIR = "storage/llamaindex"
CHROMA_DIR = "storage/chroma"
COLLECTION = "lab1_kb"

def build_or_load_index (data_dir: str = "kb"):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    try:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR, vector_store=vector_store)
        index = load_index_from_storage(storage_context)
        return index
    except Exception:
        pass

    docs = SimpleDirectoryReader(data_dir).load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

def retrieve_context (query: str, k: int = 3) -> List[str]:
    index = build_or_load_index()
    qe = index.as_query_engine(similarity_top_k=k)
    res = qe.query(query)
    chunks = []
    for sn in getattr(res, "source_models", [])[:k]:
        chunks.append(sn.get_text())
    return chunks
