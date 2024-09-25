import os

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from prompts.extract_keywords_prompt import extract_keywords_prompt
from _graphrag.relationships import get_entity_relationships


def extract_keywords_from_query(extractor: ChatOpenAI | ChatAnthropic, query: str) -> list:

    results = extractor.invoke(
        input=extract_keywords_prompt.format_messages(query=query)
    )
    
    keywords = results.content[1]['input']['keywords']
    
    return keywords

def generate_keywords_chains_from_graphrag(extractor: ChatOpenAI | ChatAnthropic, query: str, top_k: int = 2, max_depth: int = 3) -> list:
    
    keywords = extract_keywords_from_query(extractor=extractor, query=query)
    
    all_chains = []
    for keyword in keywords:
        df_sorted_by_weight = get_entity_relationships(
            db_path=os.path.join(
                os.environ['RAG_ROOT'], 
                'db',
                'graphrag',
                'create_final_relationships.parquet'
            ),
            entity_name=keyword
        )
        chains = []
        for i in range(top_k):
            if i >= len(df_sorted_by_weight):
                break
            if df_sorted_by_weight.iloc[i]['source'] == keyword:
                dist = df_sorted_by_weight.iloc[i]['target']
            else:
                dist = df_sorted_by_weight.iloc[i]['source']
            
            chains.append(f"{keyword} -> " + dist)
        
        all_chains.extend(chains)
    
    return all_chains