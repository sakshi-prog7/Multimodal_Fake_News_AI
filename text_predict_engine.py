import joblib
import numpy as np
import torch

from feature_extractor import bert_embedding




TEXT_MODEL_FILE = "model/text_fake_news_model.pkl"
TEXT_SCALER_FILE = "model/text_feature_scaler.pkl"


text_model = joblib.load(
    TEXT_MODEL_FILE
)

text_scaler = joblib.load(
    TEXT_SCALER_FILE
)


LABELS = {
    0: "FAKE NEWS",
    1: "REAL NEWS",
}


print("=" * 70)
print("TEXT PREDICTION ENGINE")
print("=" * 70)

print(
    "Model Classes :",
    text_model.classes_,
)

print(
    "Text model loaded successfully"
)

print(
    "Text scaler loaded successfully"
)

print("=" * 70)



def predict_text_news(text):

    

    if text is None:
        raise ValueError(
            "Text cannot be None"
        )

    text = str(text).strip()

    if len(text) < 10:
        raise ValueError(
            "Please enter sufficient news text."
        )


    feature = bert_embedding(text)


    if isinstance(feature, torch.Tensor):

        feature = (
            feature
            .detach()
            .cpu()
            .numpy()
        )


    feature = np.asarray(
        feature,
        dtype=np.float32,
    )


    feature = feature.reshape(
        1,
        -1,
    )


    print(
        "\nTEXT FEATURE SHAPE :",
        feature.shape,
    )

    print(
        "TEXT FEATURE MEAN  :",
        feature.mean(),
    )

    print(
        "TEXT FEATURE STD   :",
        feature.std(),
    )

    scaled_feature = (
        text_scaler.transform(
            feature
        )
    )


    print(
        "SCALED FEATURE MEAN :",
        scaled_feature.mean(),
    )

    print(
        "SCALED FEATURE STD  :",
        scaled_feature.std(),
    )


    prediction = text_model.predict(
        scaled_feature
    )[0]


    probabilities = (
        text_model.predict_proba(
            scaled_feature
        )[0]
    )


    class_list = list(
        text_model.classes_
    )


    fake_index = class_list.index(0)

    real_index = class_list.index(1)


    fake_probability = round(
        float(
            probabilities[fake_index]
        ) * 100,
        2,
    )


    real_probability = round(
        float(
            probabilities[real_index]
        ) * 100,
        2,
    )


    predicted_index = (
        class_list.index(
            prediction
        )
    )


    confidence = round(
        float(
            probabilities[
                predicted_index
            ]
        ) * 100,
        2,
    )



    print(
        "\nMODEL RAW PREDICTION :",
        prediction,
    )

    print(
        "FAKE PROBABILITY :",
        fake_probability,
    )

    print(
        "REAL PROBABILITY :",
        real_probability,
    )


    

    return {
        "prediction": LABELS[
            int(prediction)
        ],
        "confidence": confidence,
        "fake_probability": (
            fake_probability
        ),
        "real_probability": (
            real_probability
        ),
    }