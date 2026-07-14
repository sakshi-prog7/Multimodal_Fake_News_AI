import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


X = joblib.load("model/X_features.pkl")
y = joblib.load("model/y_labels.pkl")

X = np.asarray(X)
y = np.asarray(y)

# Text features only
X_text = X[:, 512:]


print("\n" + "=" * 80)
print("TEXT FEATURE DUPLICATE / LEAKAGE DIAGNOSIS")
print("=" * 80)

print("Total Samples :", len(X_text))
print("Text Shape    :", X_text.shape)
print("Labels        :", np.unique(y))


unique_features, counts = np.unique(
    X_text,
    axis=0,
    return_counts=True
)

duplicate_groups = counts[counts > 1]


print("\nEXACT DUPLICATE ANALYSIS")
print("-" * 80)

print("Unique Text Features :", len(unique_features))
print("Duplicate Groups     :", len(duplicate_groups))

if len(duplicate_groups) > 0:
    print("Duplicate Counts     :", duplicate_groups)
else:
    print("No exact duplicate text features found.")


similarity_matrix = cosine_similarity(X_text)

np.fill_diagonal(
    similarity_matrix,
    0
)


thresholds = [
    0.90,
    0.95,
    0.99,
    0.999
]


print("\nTEXT SIMILARITY ANALYSIS")
print("-" * 80)

for threshold in thresholds:

    pairs = np.argwhere(
        similarity_matrix >= threshold
    )

    unique_pairs = [
        (i, j)
        for i, j in pairs
        if i < j
    ]

    same_label = 0
    different_label = 0

    for i, j in unique_pairs:

        if y[i] == y[j]:
            same_label += 1
        else:
            different_label += 1

    print(
        f"\nThreshold >= {threshold}"
    )

    print(
        "Total Similar Pairs :",
        len(unique_pairs)
    )

    print(
        "Same Label Pairs    :",
        same_label
    )

    print(
        "Different Label     :",
        different_label
    )


print("\nNEAREST SAMPLE ANALYSIS")
print("-" * 80)

correct_label_match = 0

for i in range(len(X_text)):

    nearest_index = np.argmax(
        similarity_matrix[i]
    )

    similarity = similarity_matrix[
        i,
        nearest_index
    ]

    label_match = (
        y[i] == y[nearest_index]
    )

    if label_match:
        correct_label_match += 1

    print(
        f"Sample {i:3} | "
        f"Label {y[i]} | "
        f"Nearest {nearest_index:3} | "
        f"Nearest Label {y[nearest_index]} | "
        f"Similarity {similarity:.4f} | "
        f"Match {label_match}"
    )


match_percentage = (
    correct_label_match /
    len(X_text)
) * 100


print("\n" + "=" * 80)

print(
    "NEAREST TEXT LABEL MATCH : "
    f"{match_percentage:.2f}%"
)

print("=" * 80)