import joblib
import numpy as np
from collections import Counter

print("=" * 70)
print("MODEL BIAS DIAGNOSTIC")
print("=" * 70)

X = joblib.load("model/X_features.pkl")
y = joblib.load("model/y_labels.pkl")

model = joblib.load(
    "model/multimodal_fake_news_model_v2.pkl"
)

X = np.asarray(X)
y = np.asarray(y)

print("\nDATASET INFORMATION")
print("-" * 70)

print("Feature Shape :", X.shape)
print("Label Shape   :", y.shape)

label_counts = Counter(y)

print("\nLABEL DISTRIBUTION")

print(
    "FAKE NEWS (0) :",
    label_counts.get(0, 0)
)

print(
    "REAL NEWS (1) :",
    label_counts.get(1, 0)
)

print("\nMODEL PREDICTION DISTRIBUTION")
print("-" * 70)

predictions = model.predict(X)

prediction_counts = Counter(predictions)

print(
    "Predicted FAKE (0) :",
    prediction_counts.get(0, 0)
)

print(
    "Predicted REAL (1) :",
    prediction_counts.get(1, 0)
)

print("\nPROBABILITY ANALYSIS")
print("-" * 70)

probabilities = model.predict_proba(X)

fake_probs = probabilities[:, 0]
real_probs = probabilities[:, 1]

print(
    "Average Fake Probability :",
    round(float(np.mean(fake_probs)) * 100, 2),
    "%"
)

print(
    "Average Real Probability :",
    round(float(np.mean(real_probs)) * 100, 2),
    "%"
)

print("\nCONFIDENCE RANGE")

print(
    "Fake Probability Min :",
    round(float(np.min(fake_probs)) * 100, 2)
)

print(
    "Fake Probability Max :",
    round(float(np.max(fake_probs)) * 100, 2)
)

print("\nMISCLASSIFICATION CHECK")
print("-" * 70)

wrong_indices = np.where(
    predictions != y
)[0]

print(
    "Total Misclassified Training Samples :",
    len(wrong_indices)
)

if len(wrong_indices) > 0:

    print(
        "Misclassified Indices :",
        wrong_indices
    )

print("\nMODEL COEFFICIENT ANALYSIS")
print("-" * 70)

if hasattr(model, "coef_"):

    coefficients = model.coef_[0]

    image_weights = coefficients[:512]

    text_weights = coefficients[512:]

    print(
        "Average Image Weight :",
        np.mean(np.abs(image_weights))
    )

    print(
        "Average Text Weight  :",
        np.mean(np.abs(text_weights))
    )

    image_importance = np.sum(
        np.abs(image_weights)
    )

    text_importance = np.sum(
        np.abs(text_weights)
    )

    total = (
        image_importance
        + text_importance
    )

    print("\nFEATURE IMPORTANCE")

    print(
        "Image Contribution :",
        round(
            image_importance / total * 100,
            2
        ),
        "%"
    )

    print(
        "Text Contribution :",
        round(
            text_importance / total * 100,
            2
        ),
        "%"
    )

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)