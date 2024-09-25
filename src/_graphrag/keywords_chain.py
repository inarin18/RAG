
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from prompts.extract_keywords_prompt import extract_keywords_prompt


def extract_keywords_from_query(extractor: ChatOpenAI | ChatAnthropic, query: str) -> list:

    results = extractor.invoke(
        input=extract_keywords_prompt.format_messages(query=query)
    )
    
    keywords = results.content[1]['input']['keywords']
    
    return keywords

def generate_keywords_chains_from_graphrag(extractor: ChatOpenAI | ChatAnthropic, query: str) -> list:
    
    keywords = extract_keywords_from_query(extractor=extractor, query=query)
    
    for keyword in keywords:
        print(keyword)