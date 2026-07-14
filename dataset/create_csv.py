import os
import pandas as pd

data = []

valid_ext = (".jpg", ".jpeg", ".png", ".webp")

# Real Images
for image in os.listdir("real news"):
    if image.lower().endswith(valid_ext):
        data.append([f"real news/{image}", 1])

# Fake Images
for image in os.listdir("fake image 2.0"):
    if image.lower().endswith(valid_ext):
        data.append([f"fake image 2.0/{image}", 0])

df = pd.DataFrame(data, columns=["image_path", "label"])

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("dataset.csv", index=False)

print("CSV file created successfully!")
print("Total Images:", len(df))
print(df["label"].value_counts())