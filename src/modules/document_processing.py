import os
import json
from typing import Dict, List
from pprint import pprint

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from prompts.chunker_prompt import chunker_prompt 


def split_documents_using_llm(
        chunker: ChatOpenAI | ChatAnthropic, 
        documents: List[Document], 
    ) -> List[Document]:
    
    new_docs = []
    for doc in documents:
        
        # get base document path
        doc_path: os.path = doc.metadata['source']
        doc_path_contents = doc_path.split('/')
        base_doc_dir = "/".join(doc_path_contents[:-2])
        base_doc_name = doc_path_contents[-1].split('.')[0]
        
        print(f"\n@ Splitting document: {base_doc_name}")
        
        # split document into short documents
        short_docs = doc.page_content\
            .replace('\n\n\n\n\n\n', '\n\n')\
            .replace('\n\n\n\n\n', '\n\n')\
            .replace('\n\n\n\n', '\n\n')\
            .replace('\n\n\n', '\n\n')\
            .split('\n\n')
        short_docs = [Document(page_content=short_doc, metadata=doc.metadata) for short_doc in short_docs]
        
        for short_doc_idx, short_doc in enumerate(short_docs):
            
            chunk_file_dir = os.path.join(base_doc_dir, 'chunked_docs', base_doc_name, f'chunk_{short_doc_idx}')  
            os.makedirs(chunk_file_dir, exist_ok=True)
            
            print(len(short_doc.page_content), end=' ') 
            
            prompt: list[BaseMessage] = chunker_prompt.format_messages(document=short_doc.page_content)
            results: BaseMessage = chunker.invoke(input=prompt)
            
            with open(os.path.join(chunk_file_dir, 'results.json'), 'w') as f:
                retval = {
                    'prompt': prompt[0].content.split('\n'),
                    'given_document' : prompt[1].content.split('\n'),
                    'results': results.to_json()
                }
                json.dump(retval, f, indent=4, ensure_ascii=False)
            
            # check if chunking was successful
            if results.response_metadata['stop_reason'] == 'tool_use':
                try:
                    chunks = results.tool_calls[0]['args']['chunks']
                except:
                    print("\n", results)
                    raise ValueError("Chunking failed")
            elif results.response_metadata['stop_reason'] == 'end_turn':
                print('- reached end of turn')
                continue
            elif results.response_metadata['stop_reason'] == 'max_tokens':
                raise ValueError("Chunking failed - max tokens reached")
            else:
                raise ValueError("Chunking failed - unknown reason -> ", results.response_metadata['stop_reason'])
                
            print('length of the chunks =', len(chunks))
            for chunk_idx, chunk in enumerate(chunks):
                
                if len(chunk) <= 5:
                    print(f"Chunk {chunk_idx} is too short, skipping")
                    raise ValueError("Chunk is too short")
                
                if isinstance(chunk, dict):
                    chunk_content = " ".join(chunk.values())
                else:
                    chunk_content = chunk
                
                new_doc = Document(page_content=chunk_content, metadata=doc.metadata)
                new_docs.append(new_doc)
                
                # save chunk to file
                chunk_file_name = f"{base_doc_name}_chunk_{short_doc_idx}_{chunk_idx}.txt"
                with open(os.path.join(chunk_file_dir, chunk_file_name), 'w', encoding='utf-8') as f:
                    f.write(chunk_content)
        
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
