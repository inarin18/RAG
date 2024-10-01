import os
from datetime import datetime
import yaml
import logging

from modules.utils import setup_logging, load_config
from modules.db import create_vectorstore_from_directory
from modules.chat_models import ChatModel


def main():
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    DOCS_DIR = os.path.join(ROOT_DIR, 'DOCS', 'novels')
    CONF_DIR = os.path.join(ROOT_DIR, 'conf')
    
    config = load_config(os.path.join(CONF_DIR, 'config.yml'))
    PERSIST_DIRECTORY = os.path.join(
        ROOT_DIR, 'db', 
        f'vs_llm_{datetime.today().strftime("%Y-%m-%d-%H-%M-%S")}' if config['split_docs']['use_llm'] else \
        f'vs_cnk_{config['split_docs']['chunk_size']}_ovlp_{config['split_docs']['chunk_overlap']}'
    )
    
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
    
    create_vectorstore_from_directory(
        docs_dir = DOCS_DIR,
        use_llm  = config['split_docs']['use_llm'],
        persist_directory = PERSIST_DIRECTORY,
        chunker = chunker,
        split_config = config['split_docs']
    )
    


if __name__ == '__main__':
    main()
