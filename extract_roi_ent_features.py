
"""
Usage:
    python extract_momenta_roi_ent.py \
        --train_json path/to/train.jsonl \
        --val_json   path/to/val.jsonl \
        --test_json  path/to/test.jsonl \
        --image_dir  path/to/images \
        --output_dir path/to/save/features \
        --device     cuda \
        --max_rois   10 \
        [--use_web_entities]
        python extract_roi_ent_features_fhm.py --train_json data/train.jsonl --val_json data/dev.jsonl --test_json data/test.jsonl --image_dir data/ --output_dir FHM_features_momenta_roi
        python extract_roi_ent_features_fhm.py --train_json data/MAMI_Dataset/data/train.jsonl --test_json data/MAMI_Dataset/data/test.jsonl --image_dir data/MAMI_Dataset/data --output_dir FHM_features_momenta_roi

        """
import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import spacy
from sentence_transformers import SentenceTransformer

try:
    from google.cloud import vision
    GCV_AVAILABLE = True
except ImportError:
    GCV_AVAILABLE = False

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                continue
    return data

class ROIExtractor:
    def __init__(self, device, pooled_size=(7,7), feat_dim=256, out_dim=512,
                 score_thresh=0.5, max_rois=10):
        self.device = torch.device(device)
        self.detector = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.detector.eval()
        self.score_thresh = score_thresh
        self.max_rois = max_rois
        self.pooled_size = pooled_size
        flatten_dim = feat_dim * pooled_size[0] * pooled_size[1]
        self.fc = nn.Linear(flatten_dim, out_dim).to(self.device)

    @torch.no_grad()
    def extract(self, pil_image):
        transform = T.ToTensor()
        img_tensor = transform(pil_image).to(self.device)
        outputs = self.detector([img_tensor])[0]
        boxes = outputs['boxes']
        scores = outputs['scores']
        keep = scores >= self.score_thresh
        if keep.sum() == 0:
            return []
        boxes = boxes[keep]
        scores = scores[keep]
        scores, idxs = scores.sort(descending=True)
        idxs = idxs[:self.max_rois]
        boxes = boxes[idxs]
        features = self.detector.backbone(img_tensor.unsqueeze(0))
        image_shapes = [img_tensor.shape[-2:]]
        box_list = [boxes]
        pooled_feats = self.detector.roi_heads.box_roi_pool(features, box_list, image_shapes)
        pooled_feats = pooled_feats.to(self.device)
        num_keep = pooled_feats.shape[0]
        flattened = pooled_feats.view(num_keep, -1)
        out = self.fc(flattened)
        return [feat.cpu() for feat in out]

class EntityExtractor:
    def __init__(self, device, spacy_model="en_core_web_sm",
                 sentencemodel_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            from spacy.cli import download as spacy_download
            print(f"spaCy model {spacy_model} not found and try to download...")
            spacy_download(spacy_model)
            self.nlp = spacy.load(spacy_model)
        self.sentencemodel = SentenceTransformer(sentencemodel_name, device=device)
        if GCV_AVAILABLE:
            try:
                self.gcv_client = vision.ImageAnnotatorClient()
            except Exception:
                self.gcv_client = None
        else:
            self.gcv_client = None

    def extract_text_entities(self, text):
        if not text or not text.strip():
            D = self.sentencemodel.get_sentence_embedding_dimension()
            return torch.zeros(D)
        doc = self.nlp(text)
        ents = [ent.text for ent in doc.ents]
        if len(ents) == 0:
            emb = self.sentencemodel.encode(text, convert_to_tensor=True)
            return emb.cpu()
        embeddings = self.sentencemodel.encode(ents, convert_to_tensor=True)
        emb = torch.mean(embeddings, dim=0)
        return emb.cpu()

    def extract_web_entities(self, pil_image):
        D = self.sentencemodel.get_sentence_embedding_dimension()
        if self.gcv_client is None:
            return torch.zeros(D)
        try:
            import io
            buf = io.BytesIO()
            pil_image.save(buf, format='JPEG')
            content = buf.getvalue()
            image_req = vision.Image(content=content)
            response = self.gcv_client.web_detection(image=image_req)
            if response.error.message:
                print(f"GCV Error: {response.error.message}")
                return torch.zeros(D)
            web_entities = response.web_detection.web_entities
            descs = [ent.description for ent in web_entities if ent.description]
            if len(descs) == 0:
                return torch.zeros(D)
            embeddings = self.sentencemodel.encode(descs, convert_to_tensor=True)
            emb = torch.mean(embeddings, dim=0)
            return emb.cpu()
        except Exception as e:
            print(f"Web Entities wrong: {e}")
            return torch.zeros(D)

def process_split(split_name, json_path, image_dir, roi_extractor, ent_extractor,
                  output_dir, use_web_entities=False):
    data = load_jsonl(json_path)
    roi_feats = {}
    text_ent_feats = {}
    web_ent_feats = {}
    for item in tqdm(data, desc=f"Extracting {split_name} features", ncols=80):
        meme_id = str(item.get("id", ""))
        img_name = item.get("img") or item.get("image")
        text = item.get("text", "")
        if not meme_id or not img_name:
            print(f"Warning: ID 或 图像字段缺失，跳过样本: {item}")
            continue
        image_path = os.path.join(image_dir, img_name)
        if not os.path.exists(image_path):
            print(f"Warning: 图像不存在 {image_path}，跳过 id={meme_id}")
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: 打开图像失败 {image_path}: {e}, 跳过 id={meme_id}")
            continue
        rois = roi_extractor.extract(image)
        if len(rois) > 0:
            roi_tensor = torch.stack(rois, dim=0)
        else:
            roi_tensor = torch.zeros((0, roi_extractor.fc.out_features))
        roi_feats[meme_id] = roi_tensor
        text_ent = ent_extractor.extract_text_entities(text)
        text_ent_feats[meme_id] = text_ent
        if use_web_entities:
            web_ent = ent_extractor.extract_web_entities(image)
        else:
            D = ent_extractor.sentencemodel.get_sentence_embedding_dimension()
            web_ent = torch.zeros(D)
        web_ent_feats[meme_id] = web_ent
    roi_path = os.path.join(output_dir, f"harmeme_FHM_{split_name}_ROI.pt")
    ent_path = os.path.join(output_dir, f"harmeme_FHM_{split_name}_ent.pt")
    torch.save(roi_feats, roi_path)
    print(f"Saved ROI features for {split_name} to {roi_path}")
    torch.save(text_ent_feats, ent_path)
    print(f"Saved Text Entity features for {split_name} to {ent_path}")
    if use_web_entities:
        web_ent_path = os.path.join(output_dir, f"harmeme_FHM_{split_name}_web_ent.pt")
        torch.save(web_ent_feats, web_ent_path)
        print(f"Saved Web Entity features for {split_name} to {web_ent_path}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', type=str, required=True, help="train.jsonl")
    parser.add_argument('--val_json',   type=str, help="val.jsonl")
    parser.add_argument('--test_json',  type=str, required=True, help="test.jsonl")
    parser.add_argument('--image_dir',  type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device',     type=str, default='cuda:1', help="e.g. cuda or cpu")
    parser.add_argument('--max_rois',   type=int, default=10, help="top-K ROIs")
    parser.add_argument('--score_thresh', type=float, default=0.5)
    parser.add_argument('--use_web_entities', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'
    roi_extractor = ROIExtractor(device=device, pooled_size=(7,7),
                                 feat_dim=256, out_dim=512,
                                 score_thresh=args.score_thresh,
                                 max_rois=args.max_rois)
    ent_extractor = EntityExtractor(device=device)

    process_split('train', args.train_json, args.image_dir,
                  roi_extractor, ent_extractor, args.output_dir,
                  use_web_entities=args.use_web_entities)
    # process_split('val', args.val_json, args.image_dir,
    #               roi_extractor, ent_extractor, args.output_dir,
    #               use_web_entities=args.use_web_entities)
    process_split('test', args.test_json, args.image_dir,
                  roi_extractor, ent_extractor, args.output_dir,
                  use_web_entities=args.use_web_entities)

if __name__ == '__main__':
    main()
