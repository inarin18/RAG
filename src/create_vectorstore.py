import os
import yaml
import logging

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from modules.utils import setup_logging, load_config


def main():
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    DOCS_DIR = os.path.join(ROOT_DIR, 'DOCS', 'novels')
    CONF_DIR = os.path.join(ROOT_DIR, 'conf')
    
    config = load_config(CONF_DIR + '/config.yml')
    PERSIST_DIRECTORY = ROOT_DIR + 'db/vs_cnk_{}_ovlp_{}'.format(config['chunk_size'], config['chunk_overlap'])
    
    logger = setup_logging(config['log_level'])
    
    logger.info("Starting vector store creation process")
    
    logger.info(f"Scanning documents in {DOCS_DIR}")
    documents = []
    for root, dirs, files in os.walk(DOCS_DIR):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, DOCS_DIR)
                logger.debug(f"Loading document: {file_path}")
                
                # load documents from file
                loader = TextLoader(file_path, encoding='shift-jis')
                docs = loader.load()
                
                # add source metadata to each document
                for doc in docs:
                    doc.metadata['source'] = relative_path
                    
                documents.extend(loader.load())
    
    logger.info(f"Loaded {len(documents)} documents")
    
    logger.info("Splitting documents into chunks")
    splitter = RecursiveCharacterTextSplitter(
        separators=config['separators'],
        chunk_size=config['chunk_size'], 
        chunk_overlap=config['chunk_overlap']
    )
    texts = splitter.split_documents(documents)
    
    # add chunk index to metadata
    for i, text in enumerate(texts):
        text.metadata['chunk_index'] = i
    
    logger.info(f"Created {len(texts)} text chunks")

    logger.info("Creating embeddings")
    embeddings = OpenAIEmbeddings()
    
    logger.info("Creating and saving vector store")
    vectorstore = Chroma.from_documents(
        embedding=embeddings, 
        documents=texts,
        persist_directory=PERSIST_DIRECTORY
    )
    
    logger.info(f"Vector store saved to {PERSIST_DIRECTORY}")


if __name__ == '__main__':
    main()
