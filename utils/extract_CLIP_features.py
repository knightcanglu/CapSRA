import torch
import json
import os
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

device = torch.device("cuda")
clip_model_name = "openai/clip-vit-base-patch16"
image_dir = "data/"
data_files = ["data/train.jsonl"]
save_path = "clip_features_train.pt"

print("Loading CLIP model...")
clip = CLIPModel.from_pretrained(clip_model_name).to(device)
processor = CLIPProcessor.from_pretrained(clip_model_name)

def load_data(data_files):
    all_data = []
    for file in data_files:
        with open(file, "r") as f:
            all_data.extend([json.loads(line) for line in f])
    return all_data

def extract_clip_features(data):
    features = {}
    for item in tqdm(data, desc="Extracting CLIP features"):
        meme_id = str(item["id"])
        image_path = os.path.join(image_dir, item["img"])
        text = item["text"]

        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping...")
            continue

        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clip(**inputs)
            image_feature = outputs.image_embeds.squeeze(0)
            text_feature = outputs.text_embeds.squeeze(0)
            feature = (image_feature + text_feature) / 2.0

        features[meme_id] = feature.cpu()
    
    return features

print("Loading dataset...")
data = load_data(data_files)
features = extract_clip_features(data)

torch.save(features, save_path)
print(f"CLIP features saved to {save_path}")
