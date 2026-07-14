import os
import joblib
import numpy as np

from feature_extractor import extract_features
from ocr_engine import extract_text



FAKE_FOLDER = (
    r"C:\Users\Sakshi Tiwari\OneDrive\Desktop"
    r"\fake real news img\fake image 2.0"
)

REAL_FOLDER = (
    r"C:\Users\Sakshi Tiwari\OneDrive\Desktop"
    r"\fake real news img\real news image"
)

MODEL_FOLDER = "model"

X_OUTPUT = os.path.join(
    MODEL_FOLDER,
    "X_features_rebuilt.pkl"
)

Y_OUTPUT = os.path.join(
    MODEL_FOLDER,
    "y_labels_rebuilt.pkl"
)


FAKE_LABEL = 0
REAL_LABEL = 1

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
)

features = []
labels = []

processed_fake = 0
processed_real = 0
failed_files = []

def process_folder(folder_path, label, label_name):

    global processed_fake
    global processed_real

    print("\n" + "=" * 70)
    print(f"PROCESSING {label_name}")
    print("=" * 70)

    if not os.path.exists(folder_path):

        print(
            f"ERROR: Folder does not exist:\n"
            f"{folder_path}"
        )

        return

    image_files = [
        file_name
        for file_name in os.listdir(folder_path)
        if file_name.lower().endswith(
            IMAGE_EXTENSIONS
        )
    ]

    print(
        "Images Found :",
        len(image_files)
    )

    for index, file_name in enumerate(
        image_files,
        start=1
    ):

        image_path = os.path.join(
            folder_path,
            file_name
        )

        print(
            f"\n[{index}/{len(image_files)}] "
            f"{file_name}"
        )

        try:

            

            try:

                ocr_text = extract_text(
                    image_path
                )

                if ocr_text is None:
                    ocr_text = ""

            except Exception as ocr_error:

                print(
                    "OCR Warning :",
                    ocr_error
                )

                ocr_text = ""


            print(
                "OCR Characters :",
                len(ocr_text)
            )


            

            feature = extract_features(
                image_path,
                ocr_text
            )


           

            if hasattr(
                feature,
                "detach"
            ):

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


            
            if feature.shape[0] != 1280:

                raise ValueError(
                    "Invalid feature size: "
                    f"{feature.shape}"
                )


    

            features.append(
                feature
            )

            labels.append(
                label
            )


            if label == FAKE_LABEL:

                processed_fake += 1

            else:

                processed_real += 1


            print(
                f"SUCCESS -> {label_name}"
            )

            print(
                "Feature Shape :",
                feature.shape
            )


        except Exception as error:

            print(
                "FAILED :",
                error
            )

            failed_files.append(
                (
                    image_path,
                    str(error)
                )
            )


print(
    "\n" + "=" * 70
)

print(
    "MULTIMODAL FEATURE REBUILD"
)

print(
    "=" * 70
)



process_folder(
    FAKE_FOLDER,
    FAKE_LABEL,
    "FAKE NEWS"
)


process_folder(
    REAL_FOLDER,
    REAL_LABEL,
    "REAL NEWS"
)


X = np.asarray(
    features,
    dtype=np.float32
)

y = np.asarray(
    labels,
    dtype=np.int64
)


print(
    "\n" + "=" * 70
)

print(
    "FINAL DATASET INFORMATION"
)

print(
    "=" * 70
)


print(
    "Feature Shape :",
    X.shape
)

print(
    "Label Shape   :",
    y.shape
)


if len(y) > 0:

    unique_labels, counts = np.unique(
        y,
        return_counts=True
    )

    print(
        "Label Distribution:"
    )

    for label, count in zip(
        unique_labels,
        counts
    ):

        label_name = (
            "FAKE NEWS"
            if label == FAKE_LABEL
            else "REAL NEWS"
        )

        print(
            f"{label_name} ({label}) : "
            f"{count}"
        )


print(
    "\nProcessed Fake Images :",
    processed_fake
)

print(
    "Processed Real Images :",
    processed_real
)

print(
    "Failed Images         :",
    len(failed_files)
)



if len(features) == 0:

    raise RuntimeError(
        "No features were generated."
    )


os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


joblib.dump(
    X,
    X_OUTPUT
)

joblib.dump(
    y,
    Y_OUTPUT
)


print(
    "\n" + "=" * 70
)

print(
    "REBUILT FEATURES SAVED SUCCESSFULLY"
)

print(
    "=" * 70
)

print(
    X_OUTPUT
)

print(
    Y_OUTPUT
)

if failed_files:

    print(
        "\nFAILED FILES"
    )

    print(
        "-" * 70
    )

    for file_path, error in failed_files:

        print(
            "\nFile :",
            file_path
        )

        print(
            "Error:",
            error
        )


print(
    "\n" + "=" * 70
)

print(
    "FEATURE REBUILD COMPLETE"
)

print(
    "=" * 70
)