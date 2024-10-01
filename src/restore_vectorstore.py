import os

from modules.db import load_vectorstore
from modules.utils import load_config


def main():
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    DOCS_DIR = os.path.join(ROOT_DIR, 'DOCS', 'novels/')
    CONF_DIR = os.path.join(ROOT_DIR, 'conf/')
    
    config = load_config(CONF_DIR + 'config.yml')
    PERSIST_DIRECTORY = os.path.join(
        ROOT_DIR, 'db',
        #'vs_cnk_{}_ovlp_{}'.format(config['chunk_size'], config['chunk_overlap'])
        'vs_llm_2024-10-02-05-05-24'
    )
    loaded_vectorstore = load_vectorstore(PERSIST_DIRECTORY)
    
    query = "骸骨男の正体は誰ですか？作中で言及されている氏名で答えること。"
    query = "競漕会の三日前のレースコースでの結果は、農科と文科でどれくらいの秒数差があったか？"
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
        print(f"{i}. [Chunk {chunk_index}] - [Similarity {similarity:.4f}] \n  {doc.page_content}")
    
    
if __name__ == '__main__':
    main()