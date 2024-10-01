from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate


chunker_prompt_template = """
<role>
You are a professional novelist.
We give you a document of the novels. 
You should split the document follwing the #constraints.
</role>
<constraints>
    <constraint>Split the document into meaningful units, such as paragraphs, scenes, or chapters.</constraint>
    <constraint>Each chunk should be approximately 500-2000 characters long.</constraint>
    <constraint>Never split document to one characters.</constraint>
    <constraint>Allow for slight overlap between chunks if necessary to avoid losing important information.</constraint>
    <constraint>Provide a short title for each chunk that indicates its main topic or characters.</constraint>
    <constraint>Preserve important proper nouns and key concepts within chunks to maintain context and facilitate later question-answering.</constraint>
    <constraint>Add a 1-2 sentence summary at the beginning of each chunk to help determine relevance to future queries.</constraint>
</constraints>
"""

chunker_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(chunker_prompt_template),
    HumanMessagePromptTemplate.from_template("<document>\n{document}\n</document>")
])