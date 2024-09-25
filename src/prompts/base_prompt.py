from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate


system_prompt_template = """
<role>
You are a helpful assistant.
We give you a query about the novels. 
You should provide a correct, exact answer to the query and evidence.
</role>
<constraints>
    <constraint>
        We give you some contexts about the novels and the query.
    </constraint>
    <constraint>
        Use the contexts.
    </constraint>
    <constraint>
        If there are no evidence to answer for the query, then you should output "質問誤り" only.
    </constraint>
</constraints>
<contexts>
{contexts}
</contexts>
"""

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt_template),
    HumanMessagePromptTemplate.from_template("<query>{query}</query>")
])