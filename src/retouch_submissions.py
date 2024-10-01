import os
import time
import csv

from modules.utils import load_config
from modules.chat_models import ChatModel
from modules.query import fetch_queries
from prompts.summarize_prompt import summarize_prompt


def main():
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    DOCS_DIR = os.path.join(ROOT_DIR, 'DOCS', 'novels')
    CONF_DIR = os.path.join(ROOT_DIR, 'conf')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    
    config = load_config(os.path.join(CONF_DIR, 'config.yml'))
    
    shortener = ChatModel(
        provider = config['shortener']['model_provider'],
        model_name = config['shortener']['model_name'],
        temperature = config['shortener']['temperature'],
        max_tokens = config['shortener']['max_tokens'],
    ).fetch_model().bind_tools([config['tools']['shorten_answer']])
    
    with open(os.path.join(DATA_DIR, 'backup', 'predictions_03_unshorten.csv'), 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # csv を初期化
    curr_time = time.time()
    with open(os.path.join(DATA_DIR, 'backup', 'predictions_{}.csv'.format(curr_time)), 'w') as f:
        writer = csv.writer(f)
        writer.writerow([])
    with open(os.path.join(ROOT_DIR, 'submit', 'predictions.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow([])
        
    queries = fetch_queries(DATA_DIR)  
    for query, row in zip(queries, rows):
        idx, answer, evidence = row
        
        if len(answer) >= 45:
            
            results = shortener.invoke(
                input=summarize_prompt.format_messages(
                    query=query,
                    answer=answer
                )
            )
            try:
                short_answer = results.content[1]['input']['shorten_answer']
            except :
                short_answer = '質問誤り'
                
        else :
            short_answer = answer
        
        short_evidence = evidence if evidence != "" else 'なし'
        
        with open(os.path.join(DATA_DIR, 'backup', 'predictions_{}.csv'.format(curr_time)), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([idx, short_answer, short_evidence])
            
        with open(os.path.join(ROOT_DIR, 'submit', 'predictions.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([idx, short_answer, short_evidence])


if __name__ == '__main__':
    main()
