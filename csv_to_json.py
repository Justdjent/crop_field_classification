import pandas as pd

frame = pd.read_csv('data/train_rgb.csv')
frame.to_json('data/train_rgb.json')
