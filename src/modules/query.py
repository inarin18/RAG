import os
import csv
import pprint


def fetch_queries(data_dir="data/"):
    queries = []
    with open(os.path.join(data_dir, 'query.csv'), 'r') as f:
        reader = csv.reader(f)
        next(reader)  # ヘッダーをスキップ
        for row in reader:
            queries.append(row[1])
    return queries

if __name__ == '__main__':
    pprint.pprint(fetch_queries(os.environ.get('RAG_ROOT') + 'data/'))
    