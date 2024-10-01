import sys
import os
import json
from typing import Dict, Tuple
from pprint import pprint

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


def fetch_doc_dir_and_name(doc: Document) -> Tuple[str, str]:
    
    doc_path: os.path = doc.metadata['source']
    doc_path_contents = doc_path.split('/')
    base_doc_dir = "/".join(doc_path_contents[:-2])
    base_doc_name = doc_path_contents[-1].split('.')[0]
    
    return base_doc_dir, base_doc_name


def split_docs_into_short_docs(doc: Document) -> list[Document]:
    
    short_docs = doc.page_content\
        .replace('\n\n\n\n\n\n', '\n\n')\
        .replace('\n\n\n\n\n', '\n\n')\
        .replace('\n\n\n\n', '\n\n')\
        .replace('\n\n\n', '\n\n')\
        .split('\n\n')
    
    short_docs = [Document(page_content=short_doc, metadata=doc.metadata) for short_doc in short_docs]
    
    return short_docs


def validate_chunking_results_then_get_chunks(results: BaseMessage) -> list[str|dict] | None:
    if results.response_metadata['stop_reason'] == 'tool_use':
        try:
            chunks = results.tool_calls[0]['args']['chunks']
        except:
            print("\n", results)
            raise ValueError("Chunking failed")
    elif results.response_metadata['stop_reason'] == 'end_turn':
        print('- reached end of turn')
        return None
    elif results.response_metadata['stop_reason'] == 'max_tokens':
        raise ValueError("Chunking failed - max tokens reached")
    else:
        raise ValueError("Chunking failed - unknown reason -> ", results.response_metadata['stop_reason'])
    
    # 稀に chunks が文字列で返ってくることがあるので、リストに変換
    if isinstance(chunks, str):
        chunks = chunks.replace('[', '').replace(']', '').split(',')

    return chunks


def genarate_results_using_chunker(short_doc: Document, chunker: ChatOpenAI | ChatAnthropic, chunk_file_dir: os.path) -> BaseMessage:
    
    prompt: list[BaseMessage] = chunker_prompt.format_messages(document=short_doc.page_content)
    results: BaseMessage = chunker.invoke(input=prompt)
    
    with open(os.path.join(chunk_file_dir, 'results.json'), 'w') as f:
        retval = {
            'prompt': prompt[0].content.split('\n'),
            'given_document' : prompt[1].content.split('\n'),
            'results': results.to_json()
        }
        json.dump(retval, f, indent=4, ensure_ascii=False)
        
    return results


def validate_chunk_content_then_get_chunk(chunk_idx: int, chunk: str | dict) -> str:
    if isinstance(chunk, dict):
        chunk_content = " ".join(chunk.values())
    else:
        chunk_content = chunk
    
    if len(chunk_content) <= 5:
        print(f"Chunk {chunk_idx} is too short, skipping")
        raise ValueError("Chunk is too short")
    
    return chunk_content


def split_documents_using_llm(
        chunker: ChatOpenAI | ChatAnthropic, 
        documents: list[Document], 
    ) -> list[Document]:
    
    new_docs = []
    for doc in documents:
        
        base_doc_dir, base_doc_name = fetch_doc_dir_and_name(doc)
        
        print(f"\n@ Splitting document: {base_doc_name}")
        
        short_docs: list[Document] = split_docs_into_short_docs(doc)
        
        for short_doc_idx, short_doc in enumerate(short_docs):
            
            chunk_file_dir = os.path.join(base_doc_dir, 'chunked_docs', base_doc_name, f'chunk_{short_doc_idx}')  
            os.makedirs(chunk_file_dir, exist_ok=True)
            
            print(len(short_doc.page_content), end=' ') 
            
            results: BaseMessage = genarate_results_using_chunker(short_doc, chunker, chunk_file_dir)
            
            # check if chunking was successful
            chunks = validate_chunking_results_then_get_chunks(results)
            if chunks is None:
                continue
                
            print('length of the chunks =', len(chunks))
            for chunk_idx, chunk in enumerate(chunks):
                
                chunk_content = validate_chunk_content_then_get_chunk(chunk_idx, chunk)
                
                new_doc = Document(page_content=chunk_content, metadata=doc.metadata)
                new_docs.append(new_doc)
                
                # save chunk to file
                chunk_file_name = f"{base_doc_name}_chunk_{short_doc_idx}_{chunk_idx}.txt"
                with open(os.path.join(chunk_file_dir, chunk_file_name), 'w', encoding='utf-8') as f:
                    f.write(chunk_content)
        
    return new_docs


def fetch_documents_from_directory(doc_dir: os.path) -> list[Document]:
    
    documents = []
    for root, dirs, files in os.walk(doc_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, doc_dir)
                
                # load documents from file
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                
                # add source metadata to each document
                for doc in docs:
                    doc.metadata['source'] = relative_path
                
                documents.extend(loader.load())
    
    return documents


def restore_chunks_from_directory(chunk_file_dir: os.path) -> list[Document]:
    
    chunked_docs = []
    for doc_dir in os.listdir(chunk_file_dir):
        sorted_chunk_dirs = sorted(
            os.listdir(
                os.path.join(chunk_file_dir, doc_dir)
            ), 
            key=lambda x: int(x.split('_')[-1])
        )
        for chunk_dir in sorted_chunk_dirs:
            for root, _, files in os.walk(os.path.join(chunk_file_dir, doc_dir, chunk_dir)):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        
                        with open(file_path, 'r', encoding='utf-8') as f:
                            chunk_content = f.read()
                        
                        chunked_docs.append(
                            Document(
                                page_content=chunk_content, 
                                metadata={'source': file_path}
                            )
                        )
    
    return chunked_docs



if __name__ == '__main__':
    
    from langchain_community.document_loaders import TextLoader
    
    from utils import setup_logging, load_config
    from chat_models import ChatModel
    
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    DOCS_DIR = os.path.join(ROOT_DIR, 'DOCS', 'chunked_docs')
    CONF_DIR = os.path.join(ROOT_DIR, 'conf')
    
    sys.path.append(os.path.join(ROOT_DIR, 'src'))
    from prompts.chunker_prompt import chunker_prompt 
    
    config = load_config(os.path.join(CONF_DIR, 'config.yml'))
    
    if config['split_docs']['use_llm']:
        chunker = ChatModel(
            role = 'chunker',
            provider = config['chunker']['model_provider'],
            model_name = config['chunker']['model_name'],
            temperature = config['chunker']['temperature'],
            max_tokens = config['chunker']['max_tokens'],
        ).fetch_model().bind_tools([config['tools']['for_embedding_chunks']])
    else:
        chunker = None
    
    # documents = fetch_documents_from_directory(DOCS_DIR)
    # chunked_documents = split_documents_using_llm(chunker, documents)
    
    chunks = restore_chunks_from_directory(DOCS_DIR)
    
    print(len(chunks))



