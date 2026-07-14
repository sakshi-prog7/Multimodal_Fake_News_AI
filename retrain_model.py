import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


X = joblib.load("model/X_features_rebuilt.pkl")
y = joblib.load("model/y_labels_rebuilt.pkl")

X = np.asarray(X)
y = np.asarray(y)


print("\n" + "=" * 70)
print("MULTIMODAL MODEL RETRAINING")
print("=" * 70)

print("Feature Shape :", X.shape)
print("Label Shape   :", y.shape)

print("Unique Labels :", np.unique(y))


X_image = X[:, :512]
X_text = X[:, 512:]


print("\nIMAGE FEATURES")
print("-" * 70)

print("Shape :", X_image.shape)
print("Mean  :", X_image.mean())
print("Std   :", X_image.std())


print("\nTEXT FEATURES")
print("-" * 70)

print("Shape :", X_text.shape)
print("Mean  :", X_text.mean())
print("Std   :", X_text.std())



image_mean = X_image.mean(axis=0)
image_std = X_image.std(axis=0)

text_mean = X_text.mean(axis=0)
text_std = X_text.std(axis=0)

image_std[image_std == 0] = 1
text_std[text_std == 0] = 1


X_image_scaled = (
    X_image - image_mean
) / image_std

X_text_scaled = (
    X_text - text_mean
) / text_std


X_multimodal = np.concatenate(
    (
        X_image_scaled,
        X_text_scaled
    ),
    axis=1
)


print("\nFINAL MULTIMODAL FEATURES")
print("-" * 70)

print("Shape :", X_multimodal.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X_multimodal,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


print("\nTRAINING MODEL...")
print("-" * 70)


model.fit(
    X_train,
    y_train
)


print("MODEL TRAINING COMPLETE")



predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

matrix = confusion_matrix(
    y_test,
    predictions
)


print("\nMODEL RESULTS")
print("-" * 70)

print(
    f"Accuracy : "
    f"{accuracy * 100:.2f}%"
)

print("\nConfusion Matrix")

print(matrix)


importances = model.feature_importances_

image_importance = (
    importances[:512].sum()
)

text_importance = (
    importances[512:].sum()
)


print("\nMULTIMODAL CONTRIBUTION")
print("-" * 70)

print(
    f"IMAGE CONTRIBUTION : "
    f"{image_importance * 100:.2f}%"
)

print(
    f"TEXT CONTRIBUTION  : "
    f"{text_importance * 100:.2f}%"
)




joblib.dump(
    model,
    "model/multimodal_fake_news_model_v2.pkl"
)

joblib.dump(
    {
        "image_mean": image_mean,
        "image_std": image_std,
        "text_mean": text_mean,
        "text_std": text_std
    },
    "model/feature_scaler.pkl"
)


# Save validation data
joblib.dump(
    X_train,
    "model/X_train_v2.pkl"
)

joblib.dump(
    X_test,
    "model/X_test_v2.pkl"
)

joblib.dump(
    y_train,
    "model/y_train_v2.pkl"
)

joblib.dump(
    y_test,
    "model/y_test_v2.pkl"
)


print("\n" + "=" * 70)

print("MODEL SAVED SUCCESSFULLY")

print(
    "model/"
    "multimodal_fake_news_model_v2.pkl"
)

print(
    "model/"
    "feature_scaler.pkl"
)

print("=" * 70)