import os
import joblib
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from transformers import BertTokenizer, BertModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

print("=" * 70)
print("TEXT MODEL TRAINING")
print("=" * 70)

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------

DATASET_PATH = os.path.join(
    "dataset",
    "bert_dataset.csv",
)

MODEL_DIR = "model"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "text_fake_news_model.pkl",
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "text_feature_scaler.pkl",
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)

# -------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

df = df.dropna()

df["content"] = df["content"].astype(str)

print(df.head())

print("\nDataset Shape :", df.shape)

print("\nLabel Distribution")

print(df["target"].value_counts())

# -------------------------------------------------------
# TRAIN TEST SPLIT
# -------------------------------------------------------

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["content"].tolist(),
    df["target"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["target"],
)

print("\nTrain Size :", len(train_texts))

print("Test Size :", len(test_texts))

# -------------------------------------------------------
# LOAD BERT
# -------------------------------------------------------

print("\nLoading BERT...")

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device :", DEVICE)

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

bert = BertModel.from_pretrained(
    "bert-base-uncased"
)

bert.to(DEVICE)

bert.eval()

# -------------------------------------------------------
# BERT EMBEDDING FUNCTION
# -------------------------------------------------------

def bert_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = bert(**inputs)

    feature = outputs.last_hidden_state[:, 0, :]

    feature = (
        feature
        .cpu()
        .numpy()
        .flatten()
    )

    return feature

# -------------------------------------------------------
# GENERATE TRAIN FEATURES
# -------------------------------------------------------

print("\nGenerating Train Features...")

X_train = []

for text in tqdm(train_texts):

    feature = bert_embedding(text)

    X_train.append(feature)

X_train = np.asarray(
    X_train,
    dtype=np.float32,
)

print("Train Feature Shape :", X_train.shape)

# -------------------------------------------------------
# GENERATE TEST FEATURES
# -------------------------------------------------------

print("\nGenerating Test Features...")

X_test = []

for text in tqdm(test_texts):

    feature = bert_embedding(text)

    X_test.append(feature)

X_test = np.asarray(
    X_test,
    dtype=np.float32,
)

print("Test Feature Shape :", X_test.shape)
# -------------------------------------------------------
# STANDARD SCALING
# -------------------------------------------------------

print("\nScaling Features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("Scaling Complete")

# -------------------------------------------------------
# TRAIN LOGISTIC REGRESSION
# -------------------------------------------------------

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=5000,
    random_state=42,
    class_weight="balanced",
    solver="lbfgs",
)

model.fit(
    X_train,
    train_labels,
)

print("Training Complete")

# -------------------------------------------------------
# MODEL EVALUATION
# -------------------------------------------------------

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    test_labels,
    predictions,
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy : {accuracy*100:.2f}%"
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        test_labels,
        predictions,
    )
)

print("\nClassification Report")

print(
    classification_report(
        test_labels,
        predictions,
        target_names=[
            "FAKE NEWS",
            "REAL NEWS",
        ],
    )
)

# -------------------------------------------------------
# SAVE MODEL
# -------------------------------------------------------

print("\nSaving Model...")

joblib.dump(
    model,
    MODEL_FILE,
)

joblib.dump(
    scaler,
    SCALER_FILE,
)

print("Model Saved")

print(MODEL_FILE)

print(SCALER_FILE)

# -------------------------------------------------------
# TEST SINGLE SAMPLE
# -------------------------------------------------------

sample = test_texts[0]

feature = bert_embedding(sample)

feature = feature.reshape(
    1,
    -1,
)

feature = scaler.transform(
    feature,
)

prediction = model.predict(
    feature,
)[0]

probability = model.predict_proba(
    feature,
)[0]

print("\n" + "=" * 70)
print("SAMPLE TEST")
print("=" * 70)

print(
    sample[:250],
)

print("\nPrediction :")

if prediction == 0:
    print("FAKE NEWS")
else:
    print("REAL NEWS")

print(
    "\nConfidence :",
    round(
        np.max(probability) * 100,
        2,
    ),
    "%",
)

print("\nDone.")