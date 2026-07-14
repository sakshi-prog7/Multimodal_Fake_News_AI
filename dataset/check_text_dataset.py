import pandas as pd

df = pd.read_csv("bert_dataset.csv")

print(df.columns)
print(df.head())
print(df.isnull().sum())