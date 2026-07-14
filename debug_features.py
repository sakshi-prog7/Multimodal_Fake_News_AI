from feature_extractor import image_feature
import os
import torch

folder = "sample_images"

for file in os.listdir(folder):

    if file.endswith((".png", ".jpg", ".jpeg")):

        path = os.path.join(folder, file)

        feat = image_feature(path)

        print("=" * 80)
        print(file)
        print("Shape :", feat.shape)
        print("Mean  :", feat.mean().item())
        print("Std   :", feat.std().item())
        print("First 10 Values:")
        print(feat[:10])