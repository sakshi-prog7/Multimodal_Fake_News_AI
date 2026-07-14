# python -m uvicorn app:app --reload --port 8001
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import shutil
import time
import joblib
import numpy as np

from predict import predict_news
from text_predict import predict_text_news
from ocr_engine import extract_text
from video_feature_extractor import extract_video_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "model")
VIDEO_MODEL_PATH = os.path.join(MODEL_DIR, "video_fake_news_model.pkl")

os.makedirs(UPLOAD_DIR, exist_ok=True)

if not os.path.exists(VIDEO_MODEL_PATH):
    raise FileNotFoundError(f"Video model not found: {VIDEO_MODEL_PATH}")

video_model = joblib.load(VIDEO_MODEL_PATH)

app = FastAPI(
    title="Multimodal Fake News Detection API",
    description="AI fake news detection for text, image and video.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Multimodal Fake News Detection API is Running",
        "version": "3.0.0",
        "available_modes": ["TEXT", "MULTIMODAL", "VIDEO"],
    }


@app.post("/predict-text")
async def predict_text(text: str = Form(...)):
    try:
        start_time = time.time()
        cleaned_text = text.strip()

        if not cleaned_text:
            raise HTTPException(
                status_code=400,
                detail="News text cannot be empty.",
            )

        result = predict_text_news(cleaned_text)
        processing_time = round(time.time() - start_time, 2)

        return {
            **result,
            "analysis_mode": "TEXT",
            "input_text": cleaned_text,
            "text_length": len(cleaned_text),
            "processing_time": f"{processing_time} sec",
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    text: str = Form(""),
):
    save_path = None

    try:
        start_time = time.time()

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Image file is required.",
            )

        safe_filename = os.path.basename(file.filename)
        file_extension = os.path.splitext(safe_filename)[1].lower()

        allowed_image_extensions = {".jpg", ".jpeg", ".png"}

        if file_extension not in allowed_image_extensions:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, JPEG and PNG image files are supported.",
            )

        save_path = os.path.join(UPLOAD_DIR, safe_filename)

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ocr_text = extract_text(save_path) or ""

        text_parts = []

        if ocr_text.strip():
            text_parts.append(ocr_text.strip())

        if text.strip():
            text_parts.append(text.strip())

        final_text = "\\n".join(text_parts)

        result = predict_news(save_path, final_text)
        processing_time = round(time.time() - start_time, 2)

        return {
            **result,
            "analysis_mode": "MULTIMODAL",
            "ocr_text": ocr_text,
            "user_text": text,
            "combined_text": final_text,
            "processing_time": f"{processing_time} sec",
            "uploaded_file": safe_filename,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if save_path and os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    save_path = None

    try:
        start_time = time.time()

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Video file is required.",
            )

        safe_filename = os.path.basename(file.filename)
        file_extension = os.path.splitext(safe_filename)[1].lower()

        allowed_video_extensions = {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
        }

        if file_extension not in allowed_video_extensions:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Only video files are supported. "
                    "Allowed formats: MP4, MOV, AVI, MKV, WEBM."
                ),
            )

        save_path = os.path.join(UPLOAD_DIR, safe_filename)

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        video_feature = extract_video_features(save_path)

        video_feature = np.asarray(
            video_feature,
            dtype=np.float32,
        ).reshape(1, -1)

        if hasattr(video_model, "n_features_in_"):
            expected_features = int(video_model.n_features_in_)
            actual_features = int(video_feature.shape[1])

            if actual_features != expected_features:
                raise ValueError(
                    "Video feature size mismatch. "
                    f"Expected {expected_features}, "
                    f"received {actual_features}."
                )

        prediction = int(video_model.predict(video_feature)[0])
        probabilities = video_model.predict_proba(video_feature)[0]

        class_probability = {
            int(class_label): float(probability)
            for class_label, probability in zip(
                video_model.classes_,
                probabilities,
            )
        }

        fake_probability = class_probability.get(0, 0.0) * 100
        real_probability = class_probability.get(1, 0.0) * 100
        confidence = max(fake_probability, real_probability)

        prediction_label = (
            "FAKE VIDEO" if prediction == 0 else "REAL VIDEO"
        )

        processing_time = round(time.time() - start_time, 2)

        return {
            "prediction": prediction_label,
            "predicted_class": prediction,
            "confidence": round(confidence, 2),
            "fake_probability": round(fake_probability, 2),
            "real_probability": round(real_probability, 2),
            "analysis_mode": "VIDEO",
            "uploaded_file": safe_filename,
            "feature_size": int(video_feature.shape[1]),
            "processing_time": f"{processing_time} sec",
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if save_path and os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass
