import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


X = joblib.load("model/X_features.pkl")
y = joblib.load("model/y_labels.pkl")

X = np.asarray(X)
y = np.asarray(y)

X_image = X[:, :512]
X_text = X[:, 512:]


def test_modality(name, features):

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nPredictions:")
    print(predictions)

    print("\nActual:")
    print(y_test)


print("\nMULTIMODAL DATASET DIAGNOSIS")

test_modality(
    "IMAGE ONLY MODEL",
    X_image
)

test_modality(
    "TEXT ONLY MODEL",
    X_text
)

test_modality(
    "IMAGE + TEXT MODEL",
    X
)