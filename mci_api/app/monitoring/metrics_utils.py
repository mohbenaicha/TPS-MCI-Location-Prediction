import numpy as np
import pandas as pd
import scipy.stats as stats
import warnings
from loguru import logger


def setup_metrics_table(train_data):
    metrics_index = ['Median', 
                      'Mean', 
                      'Stdev', 
                      "New Cat Detected", 
                      "Null Val Detected", 
                      "Fisher p-val (Missing/Non-Missing)", 
                      "Extreme Val Detected",
                      'Chi Square p-val (Cat. Col. Dist.)',
                      'KS 2-Sample p-val (Cont. Col. Dist.)',
                      'Drift Detected'
                     ]

    columns = list(train_data.columns[1:])


    arr = np.zeros(shape=(len(metrics_index), len(columns)))
    metrics_table = pd.DataFrame(arr, index = metrics_index, columns = columns)
    return metrics_table



def calc_metrics(metrics_table, train_data, live_data, col):
#     metrics_table, train_data, live_data, col = args
    cat_columns = ['premises_type', 'occurrencemonth', 'occurrencehour',
       'occurrencedayofweek', 'MCI', 'Neighbourhood', 'occurrenceday', 'Pub_Id', 'Park_Id', 'PS_Id', 'occurrencedayofyear']
    dtypes={'occurrencedate':'object',
    'occurrencehour':'int',
    'premises_type':'object',
    'occurrencemonth':'object',
    'occurrencedayofweek':'object',
    'MCI':'object',
    'Neighbourhood':'object', 
    'occurrenceday':'int',
    'occurrencedayofyear':'int',
    'Pub_Id':'int',
    'Park_Id':'int',
    'PS_Id':'int',
    'Lat':'float',
    'Long':'float'}
        

    num_columns = ['Lat', 'Long']
    target_columns = ['Lat', 'Long']
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
#         for col in metrics_table.columns:
        try:
            live_data[col] = live_data[col].astype(train_data[col].dtype)
        except:
            live_data[col] = live_data[col].apply(lambda x: float(x))
            live_data[col].astype(train_data[col].dtype)
            

        # median, mean, std. dev.

        if col in cat_columns:
            metrics_table[col]['Median'] = live_data[col].value_counts().index[0]
            metrics_table[col]['Mean'] = 'N/A'
            metrics_table[col]['Stdev'] = 'N/A'
        else: 
            metrics_table[col]['Mean'] = live_data[col].mean()
            metrics_table[col]['Median'] = 'N/A'
            metrics_table[col]['Stdev'] = live_data[col].std()




        # New categories

        uniques = list(live_data[col].unique())
        if len([x for x in live_data[col] if x not in uniques]) > 0:
            metrics_table[col]['New Cat Detected'] = True
        else:
            metrics_table[col]['New Cat Detected'] = False



        # Missing values dist: Fisher
        train_data[f'{col}_na'] = np.where(train_data[col].isnull(), 1, 0)
        live_data[f'{col}_na'] = np.where(live_data[col].isnull(), 1, 0)
        ct = pd.concat([train_data.groupby(f'{col}_na')[f'{col}_na'].count(),
                        live_data.groupby(f'{col}_na')[f'{col}_na'].count()], axis=1)
        ct.fillna(0, inplace=True)

        if live_data[f'{col}_na'].sum() > 0 or train_data[f'{col}_na'].sum() > 0:
            metrics_table[col]['Null Val Detected'] = True
            _, metrics_table[col]['Fisher p-val (Missing/Non-Missing)'] = stats.fisher_exact(ct)
        else:
            metrics_table[col]['Null Val Detected'] = False
            metrics_table[col]['Fisher p-val (Missing/Non-Missing)'] = False



        # Extreme values cont. col.:

        if col in num_columns:   
            train_extremes = train_data[col].agg(['min', 'max'])
            live_extremes = live_data[col].agg(['min', 'max'])
            if train_extremes[0] > live_extremes[0] or train_extremes[1] < live_extremes[1]:
                metrics_table[col]['Extreme Val Detected'] = True
            else:
                metrics_table[col]['Extreme Val Detected'] = False
        else: 
            metrics_table[col]['Extreme Val Detected'] = 'N/A'



        # Cat feature distributions: Chi Square
        if col in cat_columns:
            ct = train_data[col].fillna('Missing').value_counts()
            cl = live_data[col].fillna('Missing').value_counts()
            if 'Missing' not in ct.index:
                ct.at['Missing'] = 0.1
            if 'Missing' not in cl:
                cl.at['Missing'] = 0.1
            for element in ct.index:
                if element not in cl.index:
                    cl.at[element] = 0.1

            for element in cl.index:
                if element not in ct.index:
                    ct.at[element] = 0.1
            ct = ct.apply(lambda x: (x/sum(ct))*sum(cl)) # scale to the size of cl to use a chisquare test
            p_val = stats.chisquare(f_obs=ct, f_exp=cl)[1]
            metrics_table[col]['Chi Square p-val (Cat. Col. Dist.)'] = round(p_val, 4)
        else:
            metrics_table[col]['Chi Square p-val (Cat. Col. Dist.)'] = 'N/A'

        # Cont. vals (Long, Lat):  2 sample K-S distance p-val.
        if col in num_columns:
            _, p_val = stats.ks_2samp(train_data[col], live_data[col])
            metrics_table[col]['KS 2-Sample p-val (Cont. Col. Dist.)'] = round(p_val, 4)
        else: 
            metrics_table[col]['KS 2-Sample p-val (Cont. Col. Dist.)'] = 'N/A'

        # Drift detected
        if col in cat_columns:
            metrics_table[col]['Drift Detected'] = (metrics_table[col]['Chi Square p-val (Cat. Col. Dist.)'] < 0.05)
        elif col in num_columns:
            metrics_table[col]['Drift Detected'] = (metrics_table[col]['KS 2-Sample p-val (Cont. Col. Dist.)'] < 0.05)

    return metrics_table