import os

import pyarrow.parquet as pq


def main():
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    
    db_name = 'create_final_entities.parquet'
    
    fin_ents_df = pq.read_table(os.path.join(ROOT_DIR, 'db', 'graphrag', db_name)).to_pandas()
    
    # print(fin_ents_df.columns)
    # >> ['id', 'name', 'type', 'description', 'human_readable_id', 'graph_embedding', 'text_unit_ids', 'description_embedding']
    
    print(fin_ents_df[['name', 'description']].iloc[0])
    print(fin_ents_df['description'].iloc[0])


if __name__ == '__main__':
    
    main()