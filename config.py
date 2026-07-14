import os


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# --------------------------------------------------
# PROJECT FOLDERS
# --------------------------------------------------

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

ASSET_DIR = os.path.join(
    BASE_DIR,
    "assets"
)


# --------------------------------------------------
# MULTIMODAL MODEL FILE
# --------------------------------------------------

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "multimodal_fake_news_model_v2.pkl"
)


# --------------------------------------------------
# FEATURE SCALER
# --------------------------------------------------

FEATURE_SCALER_FILE = os.path.join(
    MODEL_DIR,
    "feature_scaler.pkl"
)


# --------------------------------------------------
# REBUILT DATASET FILES
# --------------------------------------------------

X_FEATURES = os.path.join(
    MODEL_DIR,
    "X_features_rebuilt.pkl"
)

Y_LABELS = os.path.join(
    MODEL_DIR,
    "y_labels_rebuilt.pkl"
)


# --------------------------------------------------
# VALIDATION FILES
# --------------------------------------------------

X_TRAIN = os.path.join(
    MODEL_DIR,
    "X_train_v2.pkl"
)

X_TEST = os.path.join(
    MODEL_DIR,
    "X_test_v2.pkl"
)

Y_TRAIN = os.path.join(
    MODEL_DIR,
    "y_train_v2.pkl"
)

Y_TEST = os.path.join(
    MODEL_DIR,
    "y_test_v2.pkl"
)


# --------------------------------------------------
# IMAGE CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = (
    224,
    224
)


# --------------------------------------------------
# PREDICTION LABELS
# --------------------------------------------------

LABELS = {
    0: "FAKE NEWS",
    1: "REAL NEWS"
}