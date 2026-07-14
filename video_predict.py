import os
import sys
import joblib
import numpy as np

from video_feature_extractor import extract_video_features




BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "video_fake_news_model.pkl",
)


print("=" * 70)
print("VIDEO FAKE NEWS PREDICTION")
print("=" * 70)



if not os.path.exists(MODEL_PATH):

    print(
        f"\nERROR: Video model not found: {MODEL_PATH}"
    )

    sys.exit(1)


video_model = joblib.load(
    MODEL_PATH
)

print("\nVideo Model Loaded Successfully")

print(
    "Model Path :",
    MODEL_PATH,
)


if len(sys.argv) < 2:

    print(
        "\nERROR: Video path not provided"
    )

    print(
        '\nUsage: python video_predict.py "FULL_VIDEO_PATH"'
    )

    sys.exit(1)


video_path = sys.argv[1]



if not os.path.exists(video_path):

    print(
        f"\nERROR: Video not found: {video_path}"
    )

    sys.exit(1)


print(
    "\nVideo Path :",
    video_path,
)

print(
    "\nExtracting Video Features..."
)


try:

    video_feature = extract_video_features(
        video_path
    )

except Exception as error:

    print(
        "\nVIDEO FEATURE EXTRACTION ERROR"
    )

    print(
        "Error :",
        error,
    )

    sys.exit(1)



video_feature = np.asarray(
    video_feature,
    dtype=np.float32,
)


print(
    "\nRaw Video Feature Shape :",
    video_feature.shape,
)


video_feature = video_feature.reshape(
    1,
    -1,
)


print(
    "Model Input Feature Shape :",
    video_feature.shape,
)

if hasattr(
    video_model,
    "n_features_in_",
):

    expected_features = (
        video_model.n_features_in_
    )

    actual_features = (
        video_feature.shape[1]
    )


    print(
        "Expected Feature Size :",
        expected_features,
    )

    print(
        "Actual Feature Size   :",
        actual_features,
    )


    if actual_features != expected_features:

        print(
            "\nERROR: Feature size mismatch"
        )

        print(
            "Model expects:",
            expected_features,
        )

        print(
            "Extractor returned:",
            actual_features,
        )

        sys.exit(1)



print(
    "\nRunning Video Prediction..."
)


prediction = int(
    video_model.predict(
        video_feature
    )[0]
)


probabilities = video_model.predict_proba(
    video_feature
)[0]


classes = video_model.classes_




probability_map = {
    int(label): float(probability)
    for label, probability in zip(
        classes,
        probabilities,
    )
}


fake_probability = (
    probability_map.get(
        0,
        0.0,
    )
    * 100
)


real_probability = (
    probability_map.get(
        1,
        0.0,
    )
    * 100
)



if prediction == 0:

    prediction_label = "FAKE VIDEO"

    confidence = fake_probability

elif prediction == 1:

    prediction_label = "REAL VIDEO"

    confidence = real_probability

else:

    prediction_label = "UNKNOWN"

    confidence = 0.0



print("\n" + "=" * 70)

print(
    "VIDEO PREDICTION RESULT"
)

print("=" * 70)


print(
    "\nPrediction       :",
    prediction_label,
)


print(
    "Predicted Class  :",
    prediction,
)


print(
    "Confidence       :",
    f"{confidence:.2f}%",
)


print(
    "Fake Probability :",
    f"{fake_probability:.2f}%",
)


print(
    "Real Probability :",
    f"{real_probability:.2f}%",
)


print(
    "Model Classes    :",
    classes,
)


print("\n" + "=" * 70)

print(
    "PREDICTION COMPLETE"
)

print("=" * 70)