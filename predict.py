import joblib
import torch
import numpy as np

from config import MODEL_FILE
from feature_extractor import extract_features




model = joblib.load(MODEL_FILE)

scaler = joblib.load(
    "model/feature_scaler.pkl"
)


LABELS = {
    0: "FAKE NEWS",
    1: "REAL NEWS"
}


def predict_news(image_path, text):

    # Extract ResNet + BERT features
    feature = extract_features(
        image_path,
        text
    )

    # Convert tensor to numpy
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


    image_feature = feature[:512]

    text_feature = feature[512:]


    image_mean = np.asarray(
        scaler["image_mean"]
    )

    image_std = np.asarray(
        scaler["image_std"]
    )

    text_mean = np.asarray(
        scaler["text_mean"]
    )

    text_std = np.asarray(
        scaler["text_std"]
    )


    # Prevent division by zero
    image_std = np.where(
        image_std == 0,
        1,
        image_std
    )

    text_std = np.where(
        text_std == 0,
        1,
        text_std
    )

    image_scaled = (
        image_feature - image_mean
    ) / image_std


    text_scaled = (
        text_feature - text_mean
    ) / text_std


    scaled_feature = np.concatenate(
        (
            image_scaled,
            text_scaled
        )
    )


    scaled_feature = scaled_feature.reshape(
        1,
        -1
    )


    prediction = model.predict(
        scaled_feature
    )[0]


    probabilities = model.predict_proba(
        scaled_feature
    )[0]


    class_list = list(
        model.classes_
    )


    fake_index = class_list.index(0)

    real_index = class_list.index(1)


    fake_probability = round(
        float(
            probabilities[fake_index]
        ) * 100,
        2
    )


    real_probability = round(
        float(
            probabilities[real_index]
        ) * 100,
        2
    )


    predicted_index = class_list.index(
        prediction
    )


    confidence = round(
        float(
            probabilities[predicted_index]
        ) * 100,
        2
    )

    print("\n" + "=" * 60)

    print("LIVE PREDICTION DEBUG")

    print("=" * 60)

    print(
        "Raw Feature Shape :",
        feature.shape
    )

    print(
        "Scaled Feature Shape :",
        scaled_feature.shape
    )

    print(
        "Prediction :",
        prediction
    )

    print(
        "Fake Probability :",
        fake_probability
    )

    print(
        "Real Probability :",
        real_probability
    )

    print("=" * 60)



    return {

        "prediction":
            LABELS[int(prediction)],

        "confidence":
            confidence,

        "fake_probability":
            fake_probability,

        "real_probability":
            real_probability

    }