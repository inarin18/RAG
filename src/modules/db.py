import os

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def load_vectorstore(persist_directory) -> Chroma:
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore


def fetch_contexts_from_db(db_dir: os.path, query: str, top_k: int, is_ordinal: bool = False) -> list:
    
    vectorstore = load_vectorstore(db_dir)
    
    results = vectorstore.similarity_search_with_score(
        query=query, 
        k=top_k,
        filter=None
    )
    
    if is_ordinal:
        results = sorted(results, key=lambda x: x[0].metadata.get('chunk_index', 0))
        
    return results
