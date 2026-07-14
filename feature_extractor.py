import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from PIL import Image



resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet.eval()

image_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

def image_feature(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        feature = resnet(image)

    return feature.squeeze()


# -----------------------------
# BERT Feature Extractor
# -----------------------------

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()


def bert_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = bert(**inputs)

    return outputs.last_hidden_state[:,0,:].squeeze()




def extract_features(image_path, text):

    img = image_feature(image_path)
    txt = bert_embedding(text)

    feature = torch.cat((img, txt), dim=0)

    return feature