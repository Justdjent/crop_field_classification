import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np

csv_path = 'data/train_rgb.csv'

df = pd.read_csv(csv_path)

random_state = 42
skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

indices = []
for train_index, test_index in skf.split(df, df['Crop_Id_Ne']):
    indices.append(test_index)

df.loc[indices[0], 'Fold'] = 0
df.loc[indices[1], 'Fold'] = 1
df.loc[indices[2], 'Fold'] = 2

df['Fold'] = df['Fold'].astype(int)

df.to_csv(csv_path, index=False)