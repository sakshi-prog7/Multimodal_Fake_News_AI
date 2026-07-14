import os
import numpy as np

from video_feature_extractor import (
    extract_video_features
)

video_path = os.path.join(
    "dataset",
    "real video",
    "real video1.mp4"
)


print("\n" + "=" * 70)

print("VIDEO FEATURE EXTRACTION TEST")

print("=" * 70)


print(
    "\nVideo Path :",
    video_path
)


feature = extract_video_features(
    video_path
)


print("\nVIDEO FEATURE INFORMATION")

print("-" * 70)


print(
    "Feature Shape :",
    feature.shape
)

print(
    "Feature Mean :",
    np.mean(feature)
)

print(
    "Feature Std :",
    np.std(feature)
)

print(
    "Feature Min :",
    np.min(feature)
)

print(
    "Feature Max :",
    np.max(feature)
)


print("\n" + "=" * 70)

print("VIDEO FEATURE EXTRACTION SUCCESSFUL")

print("=" * 70)