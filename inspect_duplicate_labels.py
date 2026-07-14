import joblib
import numpy as np
from collections import defaultdict


X = joblib.load("model/X_features.pkl")
y = joblib.load("model/y_labels.pkl")

X = np.asarray(X)
y = np.asarray(y)

X_text = X[:, 512:]


print("\n" + "=" * 80)
print("DUPLICATE TEXT FEATURE LABEL INSPECTION")
print("=" * 80)

groups = defaultdict(list)

for index, feature in enumerate(X_text):

    key = feature.tobytes()

    groups[key].append(index)


duplicate_groups = [
    indices
    for indices in groups.values()
    if len(indices) > 1
]


duplicate_groups.sort(
    key=len,
    reverse=True
)


print("\nTotal Samples          :", len(X_text))
print("Unique Text Features   :", len(groups))
print("Duplicate Groups       :", len(duplicate_groups))


for group_number, indices in enumerate(
    duplicate_groups,
    start=1
):

    labels = y[indices]

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    print("\n" + "-" * 80)

    print("GROUP :", group_number)

    print(
        "Duplicate Count :",
        len(indices)
    )

    print(
        "Sample Indices  :",
        indices
    )

    print(
        "Labels          :",
        labels
    )

    print("Label Distribution:")

    for label, count in zip(
        unique_labels,
        counts
    ):

        label_name = (
            "FAKE NEWS"
            if label == 0
            else "REAL NEWS"
        )

        print(
            f"Label {label} "
            f"({label_name}) : "
            f"{count}"
        )


print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)