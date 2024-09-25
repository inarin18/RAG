
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from prompts.extract_keywords_prompt import extract_keywords_prompt


def extract_keywords_from_query(model: ChatOpenAI | ChatAnthropic, query: str) -> list:

    results = model.invoke(
        input=extract_keywords_prompt.format_messages(query=query)
    )

def generate_keywords_chains_from_graphrag(query: str) -> list:
    
    pass