import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from utils import load_config


def load_vectorstore(persist_directory):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore

def main():
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    DOCS_DIR = os.path.join(ROOT_DIR, 'DOCS', 'novels/')
    CONF_DIR = os.path.join(ROOT_DIR, 'conf/')
    
    config = load_config(CONF_DIR + 'config.yml')
    PERSIST_DIRECTORY = ROOT_DIR + 'db/vs_cnk_{}_ovlp_{}'.format(config['chunk_size'], config['chunk_overlap'])
    
    loaded_vectorstore = load_vectorstore(PERSIST_DIRECTORY)
    
    query = "骸骨男の正体は誰ですか？作中で言及されている氏名で答えること。"
    results = loaded_vectorstore.similarity_search(query, k=5)
    print(f"\n'{query}' に類似する上位5件のドキュメント:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:100]}...")  # 最初の100文字を表示
    
    
if __name__ == '__main__':
    main()