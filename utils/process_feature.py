import torch
import json
import os
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import cv2  
from PIL import Image
import torch.nn.functional as F

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
clip_model_name   = "openai/clip-vit-base-patch16"
image_dir         = "data/"
train_file        = "data/train.jsonl"
dev_file          = "data/dev.jsonl"
train_feat_path   = "clip_features_trai.pt"
dev_feat_path     = "clip_features_dev.pt"
top_n             = 20  
output_train_path = "fhm_train_neighbors_20_llava.json"
output_dev_path   = "fhm_dev_neighbors_20_llava.json"
train_desc_file = "processed_memes_captions_llava_train.json"
dev_desc_file = "processed_memes_captions_llava_dev.json"

clip = CLIPModel.from_pretrained(clip_model_name).to(device)
processor = CLIPProcessor.from_pretrained(clip_model_name)

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl(train_file)
dev_data   = load_jsonl(dev_file)

def load_desc(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
raw_train_desc = load_desc(train_desc_file)  
train_desc = {str(item['meme_id']): item['description'] for item in raw_train_desc}
raw_dev_desc = load_desc(dev_desc_file)  
dev_desc = {str(item['meme_id']): item['description'] for item in raw_dev_desc}

def extract_features(data, desc_dict):
    feats = {}
    for item in tqdm(data, desc="Extracting features"):
        mid = str(item['id'])
        img_path = os.path.join(image_dir, item['img'])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: {img_path} not found or cannot be read, skipping...")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        description = desc_dict.get(mid, "")
        if not description:
            print(f"[WARN] 没有 caption: id={mid}")
        inputs = processor(text=description, images=img, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = clip(**inputs)
            img_f = out.image_embeds.squeeze(0)
            txt_f = out.text_embeds.squeeze(0)
            feats[mid] = ((img_f + txt_f) / 2.0).cpu()
    return feats

train_feats = extract_features(train_data, train_desc)
dev_feats   = extract_features(dev_data, dev_desc)
torch.save(train_feats, train_feat_path)
torch.save(dev_feats, dev_feat_path)

t_ids = list(train_feats.keys())
train_matrix = torch.stack([train_feats[i] for i in t_ids], dim=0)

def compute_neighbors(feats_query, matrix, matrix_ids):
    results = {}
    for qid, qfeat in tqdm(feats_query.items(), desc="Computing neighbors"):
        sims = F.cosine_similarity(qfeat.unsqueeze(0), matrix, dim=1)
        top_vals, top_idxs = torch.topk(sims, k=min(top_n, len(matrix_ids)), largest=True)
        neighbors = [matrix_ids[idx] for idx in top_idxs]
        similarities = [float(top_vals[i]) for i in range(len(neighbors))]
        results[qid] = {
            "neighbors": neighbors,
            "similarities": similarities
        }
    return results

train_neighbors = compute_neighbors(train_feats, train_matrix, t_ids)
with open(output_train_path, 'w', encoding='utf-8') as ft:
    json.dump(train_neighbors, ft, ensure_ascii=False, indent=2)
print(f"Train-train neighbors saved to {output_train_path}")

dev_neighbors = compute_neighbors(dev_feats, train_matrix, t_ids)
with open(output_dev_path, 'w', encoding='utf-8') as fd:
    json.dump(dev_neighbors, fd, ensure_ascii=False, indent=2)
print(f"Dev-train neighbors saved to {output_dev_path}")