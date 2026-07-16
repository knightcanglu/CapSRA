import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from models.backbones import (
    FrozenMultimodalBackbone,
    build_processor,
    get_default_model_name,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CLIP features and build CapSRA retrieval neighbors."
    )
    parser.add_argument("--train_jsonl", type=str, required=True, help="Reference split jsonl.")
    parser.add_argument("--query_jsonl", type=str, required=True, help="Query split jsonl.")
    parser.add_argument(
        "--train_caption_json",
        type=str,
        required=True,
        help="Caption json generated for the reference split.",
    )
    parser.add_argument(
        "--query_caption_json",
        type=str,
        required=True,
        help="Caption json generated for the query split.",
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Image directory.")
    parser.add_argument(
        "--train_feature_out",
        type=str,
        required=True,
        help="Output .pt path for reference features.",
    )
    parser.add_argument(
        "--query_feature_out",
        type=str,
        required=True,
        help="Output .pt path for query features.",
    )
    parser.add_argument(
        "--train_neighbor_out",
        type=str,
        required=True,
        help="Output json path for reference-reference neighbors.",
    )
    parser.add_argument(
        "--query_neighbor_out",
        type=str,
        required=True,
        help="Output json path for query-reference neighbors.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="clip",
        choices=["clip", "vilt"],
        help="Feature backbone used for retrieval construction.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional Hugging Face model name for the selected base model.",
    )
    parser.add_argument("--top_k", type=int, default=20, help="Number of retrieved neighbors.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device such as cuda:0 or cpu.",
    )
    parser.add_argument(
        "--exclude_self",
        action="store_true",
        help="Exclude self-match when building reference-reference neighbors.",
    )
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_caption_map(path):
    with open(path, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    return {str(item["meme_id"]): item.get("description", "") for item in data}


def resolve_image_name(item):
    return item.get("img") or item.get("image") or ""


def encode_sample(base_model, backbone, processor, image, description, device):
    if base_model == "clip":
        inputs = processor(
            text=description,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
    elif base_model == "vilt":
        inputs = processor(
            image,
            description,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
    else:
        raise ValueError(f"Unsupported base_model: {base_model}")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = backbone.forward_features(inputs, device)
    return features.squeeze(0).cpu()


def extract_features(data, caption_map, image_dir, base_model, processor, backbone, device):
    features = {}
    for item in tqdm(data, desc="Extracting features"):
        meme_id = str(item["id"])
        image_name = resolve_image_name(item)
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: image not found, skip {image_path}")
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            print(f"Warning: failed to open image, skip {image_path}")
            continue

        description = caption_map.get(meme_id, "")
        features[meme_id] = encode_sample(
            base_model=base_model,
            backbone=backbone,
            processor=processor,
            image=image,
            description=description,
            device=device,
        )
    return features


def build_neighbors(query_features, ref_matrix, ref_ids, top_k, exclude_self=False):
    results = {}
    for query_id, query_feature in tqdm(query_features.items(), desc="Building neighbors"):
        similarities = F.cosine_similarity(query_feature.unsqueeze(0), ref_matrix, dim=1)
        ranked = torch.argsort(similarities, descending=True).tolist()

        neighbors = []
        scores = []
        for index in ranked:
            neighbor_id = ref_ids[index]
            if exclude_self and query_id == neighbor_id:
                continue
            neighbors.append(neighbor_id)
            scores.append(float(similarities[index]))
            if len(neighbors) >= top_k:
                break

        results[query_id] = {
            "neighbors": neighbors,
            "similarities": scores,
        }
    return results


def ensure_parent(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_name = args.model_name or get_default_model_name(args.base_model)
    backbone = FrozenMultimodalBackbone(args.base_model, model_name).to(device)
    processor = build_processor(args.base_model, model_name)

    train_data = load_jsonl(args.train_jsonl)
    query_data = load_jsonl(args.query_jsonl)
    train_captions = load_caption_map(args.train_caption_json)
    query_captions = load_caption_map(args.query_caption_json)

    train_features = extract_features(
        train_data,
        train_captions,
        args.image_dir,
        args.base_model,
        processor,
        backbone,
        device,
    )
    query_features = extract_features(
        query_data,
        query_captions,
        args.image_dir,
        args.base_model,
        processor,
        backbone,
        device,
    )

    train_ids = list(train_features.keys())
    train_matrix = torch.stack([train_features[meme_id] for meme_id in train_ids], dim=0)

    train_neighbors = build_neighbors(
        train_features,
        train_matrix,
        train_ids,
        top_k=args.top_k,
        exclude_self=args.exclude_self,
    )
    query_neighbors = build_neighbors(
        query_features,
        train_matrix,
        train_ids,
        top_k=args.top_k,
        exclude_self=False,
    )

    for path in [
        args.train_feature_out,
        args.query_feature_out,
        args.train_neighbor_out,
        args.query_neighbor_out,
    ]:
        ensure_parent(path)

    torch.save(train_features, args.train_feature_out)
    torch.save(query_features, args.query_feature_out)
    with open(args.train_neighbor_out, "w", encoding="utf-8") as fout:
        json.dump(train_neighbors, fout, ensure_ascii=False, indent=2)
    with open(args.query_neighbor_out, "w", encoding="utf-8") as fout:
        json.dump(query_neighbors, fout, ensure_ascii=False, indent=2)

    print(f"Saved reference features to {args.train_feature_out}")
    print(f"Saved query features to {args.query_feature_out}")
    print(f"Saved reference neighbors to {args.train_neighbor_out}")
    print(f"Saved query neighbors to {args.query_neighbor_out}")


if __name__ == "__main__":
    main()