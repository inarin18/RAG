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
    
    SUFFIX = '03_shorten'
    
    with open(os.path.join(result_dir, f'result_{SUFFIX}.csv'), 'r') as f:
        reader = csv.reader(f)
        next(reader)
        results = list(reader)
    
    with open(os.path.join(DATA_DIR, 'truth_answer.csv'), 'r') as f:
        reader = csv.reader(f)
        next(reader)
        truths = list(reader)
        
    with open(os.path.join(DATA_DIR, 'query.csv'), 'r') as f:
        reader = csv.reader(f)
        next(reader)
        queries = list(reader)
    
    compare_dict = {
        'perfect' : [],
        'acceptable' : [],
        'missing' : [],
        'incorrect' : []
    }
    for result, truth, idx_query in zip(results, truths, queries):
        idx, prediction, score, evidence = result
        _, truth_answer, truth_score, _ = truth
        _, query = idx_query
        
        if SCORE_DICT.get(score.lower()) < 1 : #SCORE_DICT.get(truth_score.lower()):
            compare_dict[score.lower()].append([idx, query, prediction, score, evidence, truth_answer, truth_score])
            
    for key, values in compare_dict.items():
        if len(values) > 0:
            print(f'@ My Answer\'s score -> {key}')
            for value in values:
                print('\n| index =', value[0])
                print('    query        =', value[1])
                print('    prediction   =', value[2])
                print('    score        =', value[3])
                print('    truth answer =', value[5])
                print('    truth score  =', value[6])
            print()


if __name__ == '__main__':
    main()