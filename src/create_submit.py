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
from _graphrag.keywords_chain import generate_keywords_chains_from_graphrag


def generate_ansewr(
        idx: int,
        model: ChatOpenAI | ChatAnthropic, 
        prompt: ChatPromptTemplate, 
        query: str, 
        db_dir: os.path, 
        top_k: int, 
        is_ordinal: bool
    ) -> tuple[str, list, str]:
    
    # クエリに関連する文脈を取得
    contexts_with_metadata: list = fetch_contexts_from_db(db_dir, query, top_k, is_ordinal)
    contexts = [context[0].page_content for context in contexts_with_metadata]
    
    # 最も関連性の高い文書を取得
    most_relevant_doc_path = contexts_with_metadata[0][0].metadata['source']
    with open(most_relevant_doc_path, 'r', encoding='shift-jis') as f:
        most_relevant_doc = f.read()
    
    # keywords_chains = generate_keywords_chains_from_graphrag()

    results = model.invoke(
        input=prompt.format_messages(
            query=query,
            all_text=most_relevant_doc,
            contexts=contexts
        )
    )
    
    answer = results.content[1]['input']['answer']
    evidence = results.content[1]['input']['evidence']
    
    return idx, query, contexts, answer, evidence


def main():
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    DOCS_DIR = os.path.join(ROOT_DIR, 'DOCS', 'novels')
    CONF_DIR = os.path.join(ROOT_DIR, 'conf')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    
    config = load_config(os.path.join(CONF_DIR, 'config.yml'))
    
    PERSIST_DIRECTORY = ROOT_DIR + 'db/vs_cnk_{}_ovlp_{}'.format(config['chunk_size'], config['chunk_overlap'])
    
    model_with_tool = ChatModel(
        provider = config['model_provider'],
        model_name = config['model_name'],
        temperature = config['temperature'],
        max_tokens = config['max_tokens'],
    ).fetch_model().bind_tools([config['tools']['generate_answer']])
    
    queries = fetch_queries(DATA_DIR)[0:1]
    
    with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        
        futures = [
            executor.submit(
                generate_ansewr, 
                idx = idx,
                model = model_with_tool, 
                prompt = chat_prompt, 
                query = query, 
                db_dir = PERSIST_DIRECTORY,
                top_k = config['top_k'],
                is_ordinal = config['is_ordinal']
            ) for idx, query in enumerate(queries)
        ]
        
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(queries), desc="クエリ処理中"):
            results.append(future.result())
            
    sorted_results = sorted(results, key=lambda x: x[0])
    for idx, query, contexts, answer, evidence in sorted_results:
        print(f"no.{idx} Query: {query}")
        print(f"    Answer: {answer}")
        print(f"    Evidence: {evidence}")
            
    



if __name__ == '__main__':
    main()
