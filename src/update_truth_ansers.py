import os
import csv


SCORE_DICT = {
    'perfect': 1,
    'acceptable': 0.5,
    'missing': 0,
    'incorrect': -1
}


def main():
    
    ROOT_DIR = os.environ.get('RAG_ROOT')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    
    result_dir = os.path.join(DATA_DIR, 'results')
    
    SUFFIX = '01'
    
    with open(os.path.join(result_dir, f'result_{SUFFIX}.csv'), 'r') as f:
        reader = csv.reader(f)
        next(reader)
        results = list(reader)
    
    if os.path.exists(os.path.join(DATA_DIR, f'truth_answer.csv')):
        reader = csv.reader(open(os.path.join(DATA_DIR, f'truth_answer.csv'), 'r'))
        next(reader)
        rows_before = list(reader)
        if len(rows_before) != len(results):
            rows_before = [['', '', 'incorrect', ''] for _ in range(len(results))]
    else:
        rows_before = [['', '', 'incorrect', ''] for _ in range(len(results))]
        
    with open(os.path.join(DATA_DIR, f'truth_answer.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'truth_answer', 'score', 'evidence'])
        
    for result, row_before in zip(results, rows_before):
        idx, prediction, score, evidence = result
        _, prediction_before, score_before, evidence_before = row_before
        
        with open(os.path.join(DATA_DIR, f'truth_answer.csv'), 'a') as f:
            writer = csv.writer(f)
            if score.lower() == 'perfect':
                writer.writerow([idx, prediction, score, evidence])
            elif SCORE_DICT.get(score.lower()) >= SCORE_DICT.get(score_before.lower()):
                writer.writerow([idx, prediction, score, evidence])
            elif SCORE_DICT.get(score.lower()) < SCORE_DICT.get(score_before.lower()):
                writer.writerow([idx, prediction_before, score_before, evidence_before])
            else:
                writer.writerow([idx, "", score, evidence])

if __name__ == '__main__':
    main()