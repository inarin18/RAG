import os
import csv


def main():
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    
    backup_dir = os.path.join(DATA_DIR, 'backup')
    response_dir = os.path.join(DATA_DIR, 'responses')
    result_dir = os.path.join(DATA_DIR, 'results')
    
    SUFFIX = '01'
    
    with open(os.path.join(backup_dir, f'predictions_{SUFFIX}.csv'), 'r') as f:
        reader = csv.reader(f)
        predictions = list(reader)
        
    with open(os.path.join(response_dir, f'score_result_{SUFFIX}.csv'), 'r') as f:
        reader = csv.reader(f)
        responses = list(reader)
    
    with open(os.path.join(result_dir, f'result_{SUFFIX}.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'prediction', 'score', 'evidence'])
    
    for pred, resp in zip(predictions, responses):
        idx, prediction, evidence = pred
        _, score = resp
        
        with open(os.path.join(result_dir, f'result_{SUFFIX}.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([idx, prediction, score, evidence])

if __name__ == '__main__':
    main()