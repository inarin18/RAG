import os

from modules.db import load_vectorstore
from modules.utils import load_config


def main():
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    DOCS_DIR = os.path.join(ROOT_DIR, 'DOCS', 'novels/')
    CONF_DIR = os.path.join(ROOT_DIR, 'conf/')
    
    config = load_config(CONF_DIR + 'config.yml')
    PERSIST_DIRECTORY = ROOT_DIR + 'db/vs_cnk_{}_ovlp_{}'.format(config['chunk_size'], config['chunk_overlap'])
    
    loaded_vectorstore = load_vectorstore(PERSIST_DIRECTORY)
    
    query = "骸骨男の正体は誰ですか？作中で言及されている氏名で答えること。"
    results = loaded_vectorstore.similarity_search_with_score(
        query, 
        k=5,
        filter=None
    )
    
    # chunk_indexでソート
    sorted_results = sorted(results, key=lambda x: x[0].metadata.get('chunk_index', 0))
    
    print(f"\n'{query}' に類似する上位5件のドキュメント (chunk_index順):")
    for i, (doc, score) in enumerate(sorted_results[:5], 1):
        chunk_index = doc.metadata.get('chunk_index', 'N/A')
        similarity = 1 - score
        print(f"{i}. [Chunk {chunk_index}] - [Similarity {similarity:.4f}] \n  {doc.page_content[:100]}...")  # 最初の100文字を表示
    
    
if __name__ == '__main__':
    main()