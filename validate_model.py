import os
from predict import predict_news


SAMPLE_FOLDER = "sample_images"



def get_actual_label(filename):
    filename = filename.lower()

    if "fake" in filename:
        return "FAKE NEWS"

    if "real" in filename:
        return "REAL NEWS"

    return "UNKNOWN"


def validate_model():

    print("\n" + "=" * 100)
    print("MULTIMODAL FAKE NEWS MODEL VALIDATION")
    print("=" * 100)

    total = 0
    correct = 0

    for filename in os.listdir(SAMPLE_FOLDER):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            continue

        image_path = os.path.join(
            SAMPLE_FOLDER,
            filename
        )

        actual = get_actual_label(filename)

        if actual == "UNKNOWN":
            print(
                f"\nSKIPPED: {filename} "
                f"(filename must contain fake or real)"
            )
            continue

        # Temporary validation text
        text = "news article validation test"

        try:

            result = predict_news(
                image_path,
                text
            )

            predicted = result["prediction"]
            confidence = result["confidence"]

            is_correct = actual == predicted

            if is_correct:
                status = "CORRECT"
                correct += 1
            else:
                status = "WRONG"

            total += 1

            print("\n" + "-" * 100)

            print(f"Image       : {filename}")
            print(f"Actual      : {actual}")
            print(f"Predicted   : {predicted}")
            print(f"Confidence  : {confidence}%")
            print(
                f"Fake Prob   : "
                f"{result['fake_probability']}%"
            )
            print(
                f"Real Prob   : "
                f"{result['real_probability']}%"
            )
            print(f"Status      : {status}")

        except Exception as error:

            print("\n" + "-" * 100)
            print(f"ERROR IMAGE : {filename}")
            print(f"ERROR       : {error}")

    print("\n" + "=" * 100)

    if total > 0:

        accuracy = (correct / total) * 100

        print("VALIDATION SUMMARY")
        print("-" * 100)
        print(f"Total Images   : {total}")
        print(f"Correct        : {correct}")
        print(f"Wrong          : {total - correct}")
        print(f"Accuracy       : {accuracy:.2f}%")

    else:

        print("NO VALIDATION IMAGES FOUND")

    print("=" * 100)


if __name__ == "__main__":
    validate_model()