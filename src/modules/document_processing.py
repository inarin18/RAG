import os
from typing import Dict, List

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document

from prompts.chunker_prompt import chunker_prompt 


def split_documents_using_llm(
        chunker: ChatOpenAI | ChatAnthropic, 
        documents: List[Document], 
    ) -> List[Document]:
    
    new_docs = []
    for doc in documents:
        
        results: list = chunker.invoke(
            input=chunker_prompt.format_messages(
                doc.page_content
            )
        )
        
        try:
            chunks = results.content[1]['input']['chunks']
        except:
            raise ValueError("Chunking failed")
            
        for chunk in chunks:
            new_doc = Document(page_content=chunk, metadata=doc.metadata)
            new_docs.append(new_doc)
    
    return new_docs



if __name__ == '__main__':
    
    from langchain_community.document_loaders import TextLoader
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    DOCS_DIR = os.path.join(ROOT_DIR, 'DOCS', 'novels')
    CONF_DIR = os.path.join(ROOT_DIR, 'conf')
    
    documents = []
    for root, dirs, files in os.walk(DOCS_DIR):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, DOCS_DIR)
                
                # load documents from file
                loader = TextLoader(file_path, encoding='shift-jis')
                docs = loader.load()
                
                # add source metadata to each document
                for doc in docs:
                    doc.metadata['source'] = relative_path
                    
                documents.extend(loader.load())
                
    print(documents[0].page_content[:100])
