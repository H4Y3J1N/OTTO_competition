
import pandas as pd
import numpy as np


def evaluation(pred_df):
    # COMPUTE METRIC
    score = 0
    weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}
    for t in ['clicks','carts','orders']:
        sub = pred_df.loc[pred_df.session_type.str.contains(t)].copy()
        sub['session'] = sub.session_type.apply(lambda x: int(x.split('_')[0]))
        sub.labels = sub.labels.apply(lambda x: [int(i) for i in x.split(' ')[:20]])
        test_labels = pd.read_parquet('../data/valid/test_labels.parquet')
        test_labels = test_labels.loc[test_labels['type']==t]
        test_labels = test_labels.merge(sub, how='left', on=['session'])
        test_labels['hits'] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1)
        test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0,20)
        recall = test_labels['hits'].sum() / test_labels['gt_count'].sum()
        score += weights[t]*recall
        print(f'{t} recall =',recall)
        
    print('=============')
    print('Overall Recall =',score)
    print('=============')