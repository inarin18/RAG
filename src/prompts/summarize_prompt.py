from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate


system_prompt_template = """
<role>
You are a professional novelist.
We give you a query about the novels and the answer to it. 
You should summarize the answer.
</role>
<constraints>
    <constraint>
        You should make the given answer shorten.
    </constraint>
    <constraint>
        Up to 50 strings in Japanese.
    </constraint>
</constraints>
"""

summarize_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt_template),
    HumanMessagePromptTemplate.from_template("<query>{query}</query><answer>{answer}</answer>")
])