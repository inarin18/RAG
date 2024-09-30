import os

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from modules.document_processing import split_documents_using_llm


def create_vectorstore_from_directory(
        docs_dir: os.path,
        use_llm: bool,
        persist_directory: os.path,
        chunker: ChatOpenAI | ChatAnthropic = None,
        split_config: dict = None
    ):
    
    # load documents from directory
    documents = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, docs_dir)
                
                # load documents from file
                loader = TextLoader(file_path, encoding='shift-jis')
                docs = loader.load()
                
                # add source metadata to each document
                for doc in docs:
                    doc.metadata['source'] = relative_path
                
                documents.extend(loader.load())

    # split documents into chunks
    if use_llm:
        chunked_documents = split_documents_using_llm(chunker, documents)
    else :
        splitter = RecursiveCharacterTextSplitter(
            separators=split_config['separators'],
            chunk_size=split_config['chunk_size'], 
            chunk_overlap=split_config['chunk_overlap']
        )
        chunked_documents = splitter.split_documents(documents)
    
    # add chunk index to metadata
    for i, chunk in enumerate(chunked_documents):
        chunk.metadata['chunk_index'] = i
    
    # create vectorstore  
    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(
        embedding=embeddings, 
        documents=chunked_documents,
        persist_directory=persist_directory
    )


def load_vectorstore(persist_directory: os.path) -> Chroma:
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore


def fetch_contexts_from_db(db_dir: os.path, query: str, top_k: int, is_ordinal: bool = False) -> list:
    
    vectorstore = load_vectorstore(db_dir)
    
    results = vectorstore.similarity_search_with_score(
        query=query, 
        k=top_k,
        filter=None
    )
    
    if is_ordinal:
        results = sorted(results, key=lambda x: x[0].metadata.get('chunk_index', 0))
        
    return results
