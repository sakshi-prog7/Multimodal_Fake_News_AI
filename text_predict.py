import joblib
import torch

from feature_extractor import bert_embedding


TEXT_MODEL_FILE = "model/text_fake_news_model.pkl"

LABELS = {
    0: "FAKE NEWS",
    1: "REAL NEWS",
}


text_model = joblib.load(TEXT_MODEL_FILE)


def predict_text_news(text):

    if not text or not text.strip():
        raise ValueError("News text cannot be empty.")

   
    text_feature = bert_embedding(text)


    if isinstance(text_feature, torch.Tensor):
        text_feature = (
            text_feature
            .detach()
            .cpu()
            .numpy()
        )

    text_feature = text_feature.reshape(1, -1)

    
    prediction = text_model.predict(
        text_feature
    )[0]

  
    probabilities = text_model.predict_proba(
        text_feature
    )[0]

    class_list = list(text_model.classes_)

    fake_index = class_list.index(0)
    real_index = class_list.index(1)

    fake_probability = round(
        float(probabilities[fake_index]) * 100,
        2
    )

    real_probability = round(
        float(probabilities[real_index]) * 100,
        2
    )

    predicted_index = class_list.index(
        prediction
    )

    confidence = round(
        float(probabilities[predicted_index]) * 100,
        2
    )

    return {
        "analysis_mode": "TEXT",
        "prediction": LABELS[int(prediction)],
        "confidence": confidence,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
    }