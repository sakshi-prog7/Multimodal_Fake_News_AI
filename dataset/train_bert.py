import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("bert_dataset.csv")

# Train/Test Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["content"],
    df["target"],
    test_size=0.2,
    random_state=42,
    stratify=df["target"]
)

print("Train Size:", len(train_texts))
print("Test Size:", len(test_texts))