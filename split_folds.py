import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import argparse

def run_split(df_path):
    df = pd.read_csv(df_path)
    skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    indices = []
    for train_index, test_index in skf.split(df, df['Crop_Id_Ne']):
        indices.append(test_index)

    for num, idx in enumerate(indices):
        df.loc[idx, 'Fold'] = num

    df['Fold'] = df['Fold'].astype(int)
    df.to_csv(df_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str,
                        default='data',
                        help='path to dataset')
    parser.add_argument('--transfer', type=str,
                        default='data',
                        help='path to dataset')
    args = parser.parse_args()
    run_split(args.df_path)
