import os
import time
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
        conductor: ChatOpenAI | ChatAnthropic,
        extractor: ChatOpenAI | ChatAnthropic, 
        prompt: ChatPromptTemplate, 
        query: str, 
        db_dir: os.path, 
        top_k: int, 
        is_ordinal: bool
    ) -> tuple[str, list, str]:
    
    time.sleep(20)
    
    # クエリに関連する文脈を取得
    contexts_with_metadata: list = fetch_contexts_from_db(db_dir, query, top_k, is_ordinal)
    contexts = [context[0].page_content for context in contexts_with_metadata]
    
    # 最も関連性の高い文書を取得
    most_relevant_doc_path = contexts_with_metadata[0][0].metadata['source']
    with open(most_relevant_doc_path, 'r', encoding='utf-8') as f:
        most_relevant_doc = f.read()
    
    keywords_chains = generate_keywords_chains_from_graphrag(
        extractor=extractor,
        query=query
    )

    results = conductor.invoke(
        input=prompt.format_messages(
            query=query,
            all_text=most_relevant_doc,
            keywords_chains=keywords_chains,
            contexts=contexts
        )
    )
    
    try:
        answer = results.content[1]['input']['answer']
    except :
        print(results)
        answer = '質問誤り'
    try:
        evidence = results.content[1]['input']['evidence']
    except KeyError:
        evidence = 'なし'
    except TypeError:
        evidence = 'なし'
    
    return idx, query, contexts, answer, evidence


def main():
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    DOCS_DIR = os.path.join(ROOT_DIR, 'DOCS', 'novels')
    CONF_DIR = os.path.join(ROOT_DIR, 'conf')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    
    config = load_config(os.path.join(CONF_DIR, 'config.yml'))
    
    PERSIST_DIRECTORY = os.path.join(ROOT_DIR, 'db', config['persist_dir_name'])
    
    conductor = ChatModel(
        provider = config['conductor']['model_provider'],
        model_name = config['conductor']['model_name'],
        temperature = config['conductor']['temperature'],
        max_tokens = config['conductor']['max_tokens'],
    ).fetch_model().bind_tools([config['tools']['generate_answer']])
    
    extrator = ChatModel(
        provider = config['extractor']['model_provider'],
        model_name = config['extractor']['model_name'],
        temperature = config['extractor']['temperature'],
        max_tokens = config['extractor']['max_tokens'],    
    ).fetch_model().bind_tools([config['tools']['extracting_keywords']]) 
    
    queries = fetch_queries(DATA_DIR)
    
    with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        
        futures = [
            executor.submit(
                generate_ansewr, 
                idx = idx,
                conductor = conductor, 
                extractor = extrator,
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
    
    SUBMIT_DIR = os.path.join(ROOT_DIR, 'submit')
    with open(os.path.join(SUBMIT_DIR, 'predictions.csv'), 'w', newline='') as f:
    
        for idx, query, contexts, answer, evidence in sorted_results:
            writer = csv.writer(f)
            writer.writerow([
                idx+1, 
                answer.replace('\n', ' '), 
                evidence.replace('\n', ' ')
            ])
    
    SUBMIT_BACKUP_DIR = os.path.join(ROOT_DIR, 'data', 'backup')
    with open(os.path.join(SUBMIT_BACKUP_DIR, f'predictions_{time.time()}.csv'), 'w', newline='') as f:
    
        for idx, query, contexts, answer, evidence in sorted_results:
            writer = csv.writer(f)
            writer.writerow([
                idx+1, 
                answer.replace('\n', ' '), 
                evidence.replace('\n', ' ')
            ])



if __name__ == '__main__':
    main()
