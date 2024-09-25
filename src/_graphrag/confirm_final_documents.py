import os

import pyarrow.parquet as pq


def main():
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    
    db_name = 'create_final_documents.parquet'
    
    fin_docs_df = pq.read_table(os.path.join(ROOT_DIR, 'db', 'graphrag', db_name)).to_pandas()
    
    print(fin_docs_df['raw_content'][0])


if __name__ == '__main__':
    
    main()