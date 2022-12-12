import numpy as np
import pandas as pd


def full_dataload(df_name, path:str='../../data/'):
    df_name = df_name
    df = pd.read_parquet(path + f"{df_name}.parquet")
    
    return df



def train_chunk_dataload(number, path:str='../../data/', number):
    number = number
    # LOAD TRAIN DATA. RANDOM SAMPLE 10%
    train = pd.read_parquet(path + 'train.parquet')
    sessions = train.session.unique()
    sample = np.random.choice(sessions,len(sessions)//number,replace=False)
    train = train.loc[train.session.isin(sample)]
    print(f'We are using random 1/{number} of users. Truncated train data has shape', train.shape )
    
    return train