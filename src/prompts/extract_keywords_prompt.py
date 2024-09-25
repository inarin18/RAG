from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate


system_prompt_template = """
<role>
You are a professional novelist.
We give you a query about the novels. 
You should extract some keywords to answer to the query correctly.
</role>
<constraints>
    <constraint>
        We give you a query.
    </constraint>
</constraints>
"""

extract_keywords_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt_template),
    HumanMessagePromptTemplate.from_template("<query>{query}</query>")
])
