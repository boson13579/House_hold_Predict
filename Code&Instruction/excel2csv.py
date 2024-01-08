import pandas as pd

df = pd.read_excel('../data/raw/all.xlsx')
old_cols = df.columns

df = df.dropna(axis=1)
new_cols = df.columns

df.to_csv('../data/raw/all.csv', index=False)

print([col for col in old_cols if col not in new_cols])