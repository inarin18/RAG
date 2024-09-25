import os
import pyarrow.parquet as pq


def get_entity_relationships(db_path='db/graphrag/create_final_relationships.parquet', entity_name=None):
    
    assert entity_name is not None, 'entity_name must be specified'
    
    relationships_df = pq.read_table(db_path).to_pandas()
    
    filtered_df = relationships_df[
        (relationships_df['source'] == entity_name) | (relationships_df['target'] == entity_name)
    ]
    
    # weightカラムの値の降順にソート
    sorted_df = filtered_df.sort_values(by='weight', ascending=False)
    
    return sorted_df


if __name__ == '__main__':
    print(
        get_entity_relationships(
            db_path=os.path.join(
                os.environ['RAG_ROOT'], 
                'db',
                'graphrag',
                'create_final_relationships.parquet'
            ),
            entity_name='秒数差'
        )
    )