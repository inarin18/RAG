import os

import pyarrow.parquet as pq
import pandas as pd

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


def print_head_parquet(db_path):
    table = pq.read_table(db_path)
    df = table.to_pandas()
    print(df.head())

def print_content_parquet(db_path):
    table = pq.read_table(db_path)
    df = table.to_pandas()
    print(df['clustered_graph'][0][:10000])


if __name__ == '__main__':
    
    for root, dirs, files in os.walk('db/graphrag/'):
        for file in files:
            if file.endswith('.parquet'):
                print("Testing", file)
                if file == 'create_base_entity_graph.parquets':
                    print_content_parquet(os.path.join(root, file))
                else:
                    print_head_parquet(os.path.join(root, file))