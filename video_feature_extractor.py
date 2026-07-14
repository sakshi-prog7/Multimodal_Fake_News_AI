import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms



resnet = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

resnet.fc = nn.Identity()

resnet.eval()


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_frame_feature(frame):

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(frame)

    image = image_transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():
        feature = resnet(image)

    feature = feature.squeeze()

    return feature.cpu().numpy()


def extract_video_features(
    video_path,
    max_frames=20
):

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError(
            f"Unable to open video: {video_path}"
        )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(
        "Total Video Frames :",
        total_frames
    )

    if total_frames <= 0:

        capture.release()

        raise ValueError(
            "Video contains no frames."
        )

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        min(max_frames, total_frames),
        dtype=int
    )

    frame_features = []

    for frame_index in frame_indices:

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        success, frame = capture.read()

        if not success:
            continue

        feature = extract_frame_feature(frame)

        frame_features.append(feature)

    capture.release()

    if len(frame_features) == 0:

        raise ValueError(
            "No valid video frames extracted."
        )

    frame_features = np.asarray(
        frame_features,
        dtype=np.float32
    )

    video_feature = np.mean(
        frame_features,
        axis=0
    )

    print(
        "Frames Processed :",
        len(frame_features)
    )

    print(
        "Video Feature Shape :",
        video_feature.shape
    )

    return video_feature