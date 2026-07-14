import os
import joblib
import numpy as np
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model",
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "video_features.pkl",
)

LABEL_PATH = os.path.join(
    MODEL_DIR,
    "video_labels.pkl",
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "video_fake_news_model.pkl",
)



print("=" * 70)
print("VIDEO MODEL RETRAINING")
print("=" * 70)

print(
    "\nScikit-learn Version :",
    sklearn.__version__,
)


if not os.path.exists(FEATURE_PATH):
    raise FileNotFoundError(
        f"Video feature file not found: {FEATURE_PATH}"
    )

if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(
        f"Video label file not found: {LABEL_PATH}"
    )


X_video = joblib.load(
    FEATURE_PATH
)

y_video = joblib.load(
    LABEL_PATH
)


X_video = np.asarray(
    X_video,
    dtype=np.float32,
)

y_video = np.asarray(
    y_video,
    dtype=np.int64,
)

print(
    "\nVideo Feature Shape :",
    X_video.shape,
)

print(
    "Video Label Shape   :",
    y_video.shape,
)

print(
    "Unique Labels       :",
    np.unique(
        y_video,
        return_counts=True,
    ),
)

print(
    "NaN Count           :",
    np.isnan(X_video).sum(),
)

print(
    "Inf Count           :",
    np.isinf(X_video).sum(),
)


if len(X_video) != len(y_video):
    raise ValueError(
        "Feature and label sample count mismatch."
    )

if np.isnan(X_video).any():
    raise ValueError(
        "Video features contain NaN values."
    )

if np.isinf(X_video).any():
    raise ValueError(
        "Video features contain Inf values."
    )


print("\n" + "=" * 70)
print("LOOCV VALIDATION")
print("=" * 70)


loo = LeaveOneOut()

true_labels = []
predicted_labels = []


for fold_number, (train_index, test_index) in enumerate(
    loo.split(X_video),
    start=1,
):

    X_train = X_video[train_index]
    X_test = X_video[test_index]

    y_train = y_video[train_index]
    y_test = y_video[test_index]


    fold_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


    fold_model.fit(
        X_train,
        y_train,
    )


    prediction = int(
        fold_model.predict(
            X_test
        )[0]
    )


    actual_label = int(
        y_test[0]
    )


    true_labels.append(
        actual_label
    )

    predicted_labels.append(
        prediction
    )


    actual_name = (
        "FAKE"
        if actual_label == 0
        else "REAL"
    )

    predicted_name = (
        "FAKE"
        if prediction == 0
        else "REAL"
    )


    print(
        f"Fold {fold_number:02d}",
        "| Actual:",
        actual_name,
        "| Predicted:",
        predicted_name,
    )




true_labels = np.asarray(
    true_labels,
    dtype=np.int64,
)

predicted_labels = np.asarray(
    predicted_labels,
    dtype=np.int64,
)


validation_accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)


print("\n" + "=" * 70)
print("LOOCV RESULTS")
print("=" * 70)


print(
    "\nValidation Accuracy :",
    f"{validation_accuracy * 100:.2f}%",
)


print(
    "\nPrediction Distribution:"
)

print(
    np.unique(
        predicted_labels,
        return_counts=True,
    )
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=[
            "FAKE VIDEO",
            "REAL VIDEO",
        ],
        zero_division=0,
    )
)



print("\n" + "=" * 70)
print("FINAL VIDEO MODEL TRAINING")
print("=" * 70)


video_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


video_model.fit(
    X_video,
    y_video,
)


print(
    "\nFinal Video Model Training Complete"
)



training_predictions = video_model.predict(
    X_video
)


training_accuracy = accuracy_score(
    y_video,
    training_predictions,
)


print(
    "\nTraining Accuracy :",
    f"{training_accuracy * 100:.2f}%",
)


print(
    "Training Prediction Distribution :",
    np.unique(
        training_predictions,
        return_counts=True,
    ),
)

joblib.dump(
    video_model,
    MODEL_PATH,
)


verified_model = joblib.load(
    MODEL_PATH
)


verified_predictions = verified_model.predict(
    X_video
)


verification_match = np.array_equal(
    training_predictions,
    verified_predictions,
)

print("\n" + "=" * 70)
print("VIDEO MODEL SAVED SUCCESSFULLY")
print("=" * 70)


print(
    "\nModel Path :",
    MODEL_PATH,
)

print(
    "Scikit-learn Version :",
    sklearn.__version__,
)

print(
    "Feature Shape :",
    X_video.shape,
)

print(
    "Label Shape :",
    y_video.shape,
)

print(
    "LOOCV Accuracy :",
    f"{validation_accuracy * 100:.2f}%",
)

print(
    "Training Accuracy :",
    f"{training_accuracy * 100:.2f}%",
)

print(
    "Saved Model Verification :",
    verification_match,
)


print("\nLABEL MAPPING")

print(
    "0 = FAKE VIDEO"
)

print(
    "1 = REAL VIDEO"
)


print("\n" + "=" * 70)
print("RETRAINING COMPLETE")
print("=" * 70)