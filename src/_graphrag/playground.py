import os

import pyarrow.parquet as pq
import pandas as pd


def print_head_parquet(db_path):
    table = pq.read_table(db_path)
    df = table.to_pandas()
    print(df.head())

def print_content_parquet(db_path):
    table = pq.read_table(db_path)
    df = table.to_pandas()
    print(df['clustered_graph'][0][:10000])


if __name__ == '__main__':
    
    for root, dirs, files in os.walk('db/graphrag/'):
        for file in files:
            if file.endswith('.parquet'):
                print("Testing", file)
                if file == 'create_base_entity_graph.parquets':
                    print_content_parquet(os.path.join(root, file))
                else:
                    print_head_parquet(os.path.join(root, file))
