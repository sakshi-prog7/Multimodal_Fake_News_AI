import pandas as pd

df = pd.read_csv("data.csv")

# Combine title + text
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

# Keep only needed columns
df = df[["content", "target"]]

print(df.head())
print(df.shape)

# Save cleaned dataset
df.to_csv("bert_dataset.csv", index=False)

print("bert_dataset.csv created")