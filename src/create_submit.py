import os
import csv
import pprint

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from modules.utils import setup_logging, load_config
from modules.chat_models import ChatModel
from modules.query import fetch_queries
from modules.db import fetch_contexts_from_db
from prompts.base_prompt import chat_prompt


def generate_ansewr(
        model: ChatOpenAI | ChatAnthropic, 
        prompt: ChatPromptTemplate, 
        query: str, 
        db_dir: os.path, 
        top_k: int, 
        is_ordinal: bool
    ) -> tuple[str, list, str]:
    
    contexts: list = fetch_contexts_from_db(db_dir, query, top_k, is_ordinal)

    results = model.invoke(
        input=prompt.format_messages(
            query=query,
            contexts=contexts
        )
    )
    
    return query, contexts, results.content


def main():
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    DOCS_DIR = os.path.join(ROOT_DIR, 'DOCS', 'novels')
    CONF_DIR = os.path.join(ROOT_DIR, 'conf')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    
    config = load_config(os.path.join(CONF_DIR, 'config.yml'))
    
    PERSIST_DIRECTORY = ROOT_DIR + 'db/vs_cnk_{}_ovlp_{}'.format(config['chunk_size'], config['chunk_overlap'])
    
    model = ChatModel(
        provider = config['model_provider'],
        model_name = config['model_name'],
        temperature = config['temperature'],
        max_tokens = config['max_tokens'],
    ).fetch_model()
    
    queries = fetch_queries(DATA_DIR)[1:2]
    
    with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        
        futures = [
            executor.submit(
                generate_ansewr, 
                model = model, 
                prompt = chat_prompt, 
                query = query, 
                db_dir = PERSIST_DIRECTORY,
                top_k = config['top_k'],
                is_ordinal = config['is_ordinal']
            ) for query in queries
        ]
        
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(queries), desc="クエリ処理中"):
            results.append(future.result())
    
    for query, contexts, result in results:
        print(f"Query: {query}")
        pprint.pprint(contexts)
        print(f"    Result: {result}")
            
    



if __name__ == '__main__':
    main()
