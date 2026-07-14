import joblib
import numpy as np

from config import MODEL_FILE


model = joblib.load(MODEL_FILE)

importances = model.feature_importances_

# Feature positions
# 0 - 511     = ResNet image features
# 512 - 1279  = BERT text features

image_importance = importances[:512]
text_importance = importances[512:]


print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

print("\nIMAGE FEATURES")
print("-" * 70)

print("Feature Count :", len(image_importance))
print("Total Importance :", image_importance.sum())
print("Mean Importance  :", image_importance.mean())
print("Active Features  :", np.count_nonzero(image_importance))


print("\nTEXT FEATURES")
print("-" * 70)

print("Feature Count :", len(text_importance))
print("Total Importance :", text_importance.sum())
print("Mean Importance  :", text_importance.mean())
print("Active Features  :", np.count_nonzero(text_importance))


print("\nIMPORTANCE PERCENTAGE")
print("-" * 70)

image_percentage = image_importance.sum() * 100
text_percentage = text_importance.sum() * 100

print(f"IMAGE CONTRIBUTION : {image_percentage:.2f}%")
print(f"TEXT CONTRIBUTION  : {text_percentage:.2f}%")


print("\nTOP 20 IMPORTANT FEATURES")
print("-" * 70)

top_indices = np.argsort(importances)[-20:][::-1]

for index in top_indices:

    importance = importances[index]

    if index < 512:
        feature_type = "IMAGE"
        feature_number = index
    else:
        feature_type = "TEXT"
        feature_number = index - 512

    print(
        f"{feature_type:5} | "
        f"Feature {feature_number:4} | "
        f"Importance {importance:.6f}"
    )


print("\n" + "=" * 70)