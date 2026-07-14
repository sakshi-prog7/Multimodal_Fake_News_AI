import joblib
import numpy as np
import sys
import torch

from feature_extractor import extract_features


if len(sys.argv) < 2:
    print(
        "Usage: python diagnose_live_feature.py "
        "\"image_path\" \"optional text\""
    )
    sys.exit()


image_path = sys.argv[1]

text = ""

if len(sys.argv) >= 3:
    text = sys.argv[2]


print("\n" + "=" * 70)
print("LIVE FEATURE DIAGNOSTIC")
print("=" * 70)


# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------

X_raw = joblib.load(
    "model/X_features.pkl"
)

y = joblib.load(
    "model/y_labels.pkl"
)

X_train = joblib.load(
    "model/X_train_v2.pkl"
)

model = joblib.load(
    "model/multimodal_fake_news_model_v2.pkl"
)

scaler_data = joblib.load(
    "model/feature_scaler.pkl"
)


X_raw = np.asarray(X_raw)
X_train = np.asarray(X_train)
y = np.asarray(y)


# --------------------------------------------------
# EXTRACT LIVE FEATURE
# --------------------------------------------------

feature = extract_features(
    image_path,
    text
)


if isinstance(feature, torch.Tensor):

    feature = (
        feature
        .detach()
        .cpu()
        .numpy()
    )


feature = np.asarray(
    feature,
    dtype=np.float32
)


print("\nLIVE RAW FEATURE")
print("-" * 70)

print("Shape :", feature.shape)
print("Mean  :", feature.mean())
print("Std   :", feature.std())
print("Min   :", feature.min())
print("Max   :", feature.max())


# --------------------------------------------------
# SPLIT FEATURES
# --------------------------------------------------

live_image = feature[:512]

live_text = feature[512:]


training_image = X_raw[:, :512]

training_text = X_raw[:, 512:]


print("\nLIVE IMAGE FEATURE")
print("-" * 70)

print("Mean :", live_image.mean())
print("Std  :", live_image.std())


print("\nTRAINING IMAGE FEATURE")
print("-" * 70)

print("Mean :", training_image.mean())
print("Std  :", training_image.std())
print("Min  :", training_image.min())
print("Max  :", training_image.max())


print("\nLIVE TEXT FEATURE")
print("-" * 70)

print("Mean :", live_text.mean())
print("Std  :", live_text.std())


print("\nTRAINING TEXT FEATURE")
print("-" * 70)

print("Mean :", training_text.mean())
print("Std  :", training_text.std())
print("Min  :", training_text.min())
print("Max  :", training_text.max())


# --------------------------------------------------
# SCALE LIVE FEATURE
# --------------------------------------------------

image_mean = np.asarray(
    scaler_data["image_mean"]
)

image_std = np.asarray(
    scaler_data["image_std"]
)

text_mean = np.asarray(
    scaler_data["text_mean"]
)

text_std = np.asarray(
    scaler_data["text_std"]
)


image_std[image_std == 0] = 1
text_std[text_std == 0] = 1


live_image_scaled = (
    live_image - image_mean
) / image_std


live_text_scaled = (
    live_text - text_mean
) / text_std


live_scaled = np.concatenate(
    (
        live_image_scaled,
        live_text_scaled
    )
)


print("\nSCALED LIVE FEATURE")
print("-" * 70)

print("Mean :", live_scaled.mean())
print("Std  :", live_scaled.std())
print("Min  :", live_scaled.min())
print("Max  :", live_scaled.max())


print("\nTRAINING SCALED FEATURE")
print("-" * 70)

print("Mean :", X_train.mean())
print("Std  :", X_train.std())
print("Min  :", X_train.min())
print("Max  :", X_train.max())


# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------

prediction = model.predict(
    live_scaled.reshape(1, -1)
)[0]


probability = model.predict_proba(
    live_scaled.reshape(1, -1)
)[0]


print("\nMODEL RESULT")
print("-" * 70)

print("Prediction :", prediction)
print("Classes    :", model.classes_)
print("Probability:", probability)


# --------------------------------------------------
# DISTANCE FROM TRAINING SAMPLES
# --------------------------------------------------

distances = np.linalg.norm(
    X_train - live_scaled,
    axis=1
)


nearest_indices = np.argsort(
    distances
)[:10]


print("\nNEAREST TRAINING FEATURES")
print("-" * 70)


for rank, index in enumerate(
    nearest_indices,
    start=1
):

    print(
        f"{rank}. "
        f"Training Index = {index} | "
        f"Distance = {distances[index]:.4f}"
    )


print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)