import os
import math
import random
import argparse
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image

from transformers import CLIPModel, CLIPProcessor, get_linear_schedule_with_warmup
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_neighbors(neighbor_path):
    try:
        with open(neighbor_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: load_neighbors error: {e}. Returning empty dict.")
        return {}

class HatefulMemeDataset(Dataset):
    def __init__(self, data_file, image_dir):
        self.data = []
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        self.data.append(item)
                    except:
                        continue
        except Exception as e:
            print(f"Error loading data_file {data_file}: {e}")
        self.image_dir = image_dir
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item.get('image', ""))
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: cannot open image {image_path}: {e}. Using blank image.")
            image = Image.new("RGB", (224,224), color=(0,0,0))
        text = item.get('text', "")
        meme_id = item.get('id', "")
        image = image_transform(image)
        text_inputs = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        label = torch.tensor(item.get('labels', 0), dtype=torch.long)
        
        inputs = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": image
        }
        return [meme_id], inputs, label

def collate_fn(batch):
    meme_ids, inputs_list, labels = zip(*batch)
    meme_ids = [mid for sub in meme_ids for mid in sub]
    input_ids = pad_sequence([x["input_ids"] for x in inputs_list], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([x["attention_mask"] for x in inputs_list], batch_first=True, padding_value=0)
    pixel_values = torch.stack([x["pixel_values"] for x in inputs_list])
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values
    }
    labels = torch.tensor(labels)
    return meme_ids, inputs, labels

class GATLayer(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=512, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.fc = nn.Linear(hidden_dim * 4, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x  

class HatefulMemeClassifierIB(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch16",
        feature_path_train=None,
        feature_path_dev=None,
        neighbor_path_train=None,
        neighbor_path_dev=None,
        ib_beta=1e-3,
        ib_dim=512,
        use_ib=True
    ):
        super().__init__()
        self.use_ib = use_ib
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        for param in self.clip.parameters():
            param.requires_grad = False
        
        self.gat = GATLayer(in_dim=512, hidden_dim=256, out_dim=512)
        
        self.mu_proj = nn.Linear(512, ib_dim)
        self.logsigma_proj = nn.Linear(512, ib_dim)
        if ib_dim != 512:
            self.ib_to_fuse = nn.Linear(ib_dim, 512)
        else:
            self.ib_to_fuse = None
        self.ib_beta = ib_beta
        
        self.fusion_proj = nn.Linear(512 * 2, 512)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 2)
        )
        
        self.feature_path_train = feature_path_train
        self.feature_path_dev = feature_path_dev
        self.neighbor_path_train = neighbor_path_train
        self.neighbor_path_dev = neighbor_path_dev
        
        self.clip_features = {}
        self.neighbors = {}
        
        if not self.use_ib:
            print("INFO: IB module is disabled. Using GAT outputs directly without KL regularization.")
    
    def load_clip_features(self, feature_path, device):
        if feature_path is None:
            self.clip_features = {}
            return
        try:
            loaded = torch.load(feature_path, map_location=device)
            self.clip_features = {str(k): v.to(device) for k, v in loaded.items()}
        except Exception as e:
            print(f"Warning: load_clip_features error for {feature_path}: {e}. Using empty clip_features.")
            self.clip_features = {}
    
    def load_neighbors(self, neighbor_path):
        if neighbor_path is None:
            self.neighbors = {}
            return
        self.neighbors = load_neighbors(neighbor_path)
    
    def reparametrize(self, mu, log_sigma2):
        if self.training:
            sigma = torch.exp(0.5 * log_sigma2)
            eps = torch.randn_like(sigma)
            return mu + eps * sigma
        else:
            return mu
    
    def forward(self, meme_ids, inputs, labels=None):
        device = next(self.parameters()).device
        batch_size = len(meme_ids)
        
        outputs = self.clip(
            pixel_values=inputs["pixel_values"].to(device),
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device)
        )
        image_features = outputs.image_embeds    
        text_features = outputs.text_embeds      
        clip_features = (image_features + text_features) / 2  
        
        all_nodes = [clip_features[i].unsqueeze(0) for i in range(batch_size)]
        edge_index = []
        for i, meme_id in enumerate(meme_ids):
            mid = str(meme_id)
            if mid not in self.neighbors:
                continue
            for nb in self.neighbors[mid].get("neighbors", []):
                nbs = str(nb)
                if nbs in self.clip_features:
                    all_nodes.append(self.clip_features[nbs].unsqueeze(0))
                    edge_index.append([i, len(all_nodes) - 1])
        x = torch.cat(all_nodes, dim=0).to(device)  
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)  
        else:
            edge_index = torch.empty((2,0), dtype=torch.long, device=device)
        
        gat_output = self.gat(x, edge_index)  
        batch_gat_features = gat_output[:batch_size]  
        
        if self.use_ib:
            mu = self.mu_proj(batch_gat_features)            
            log_sigma2 = self.logsigma_proj(batch_gat_features)  
            z0 = self.reparametrize(mu, log_sigma2)          
            if self.ib_to_fuse is not None:
                z = self.ib_to_fuse(z0)                     
            else:
                z = z0                                      
            if labels is not None:
                kl_per = -0.5 * torch.sum(1 + log_sigma2 - mu.pow(2) - torch.exp(log_sigma2), dim=1)  
                ib_loss = self.ib_beta * torch.mean(kl_per)
            else:
                ib_loss = torch.tensor(0.0, device=device)
        else:
            z = batch_gat_features  
            ib_loss = torch.tensor(0.0, device=device)
        
        fused_input = torch.cat([clip_features, z], dim=1)  
        w = torch.sigmoid(self.fusion_proj(fused_input))   
        combined = w * clip_features + (1 - w) * z          
        
        logits = self.classifier(combined)  
        return logits, ib_loss

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device,
                    neighbor_path_train, feature_path_train, global_step, max_updates,
                    log_every_batches=50, debug=False):
    model.train()
    model.load_neighbors(neighbor_path_train)
    model.load_clip_features(feature_path_train, device)
    total_loss = total_ce = total_ib = 0.0
    correct = total_samples = 0
    batch_count = 0

    for batch_idx, (meme_ids, inputs, labels) in enumerate(tqdm(dataloader, desc="Training", unit="batch")):
        if global_step >= max_updates:
            break
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, ib_loss = model(meme_ids, inputs, labels)
        loss_ce = F.cross_entropy(logits, labels)
        loss = loss_ce + ib_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_ib += ib_loss.item()
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        batch_count += 1

        if (batch_idx + 1) % log_every_batches == 0:
            avg_ce = total_ce / batch_count
            avg_ib = total_ib / batch_count
            acc = correct / total_samples if total_samples > 0 else 0.0
            print(f"[Batch {batch_idx+1}] CE={avg_ce:.6f}, IB={avg_ib:.6f}, Acc={acc:.4f}")

    if batch_count > 0:
        avg_loss = total_loss / batch_count
        avg_ce = total_ce / batch_count
        avg_ib = total_ib / batch_count
        train_acc = correct / total_samples if total_samples > 0 else 0.0
    else:
        avg_loss = avg_ce = avg_ib = train_acc = 0.0

    return avg_loss, train_acc, avg_ce, avg_ib, global_step

def evaluate(model, dataloader, device, neighbor_path_dev, feature_path_dev):
    model.eval()
    model.load_neighbors(neighbor_path_dev)
    model.load_clip_features(feature_path_dev, device)
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for meme_ids, inputs, labels in dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            logits, _ = model(meme_ids, inputs, labels=None)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return macro_f1, auc, acc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Hateful Meme Classification with optional IB")
    
    parser.add_argument('--task', choices=['harm-c', 'harm-p'], default='harm-c')
    parser.add_argument('--train_json', default='MOMENTA-main/HarMeme_V1/Annotations/Harm-C/train_fix.jsonl')
    parser.add_argument('--val_json', default='MOMENTA-main/HarMeme_V1/Annotations/Harm-C/val_fix.jsonl')
    parser.add_argument('--test_json', default='MOMENTA-main/HarMeme_V1/Annotations/Harm-C/test_fix.jsonl')
    parser.add_argument('--image_dir', default='MOMENTA-main/HarMeme_V1/Annotations/Harm-C/img')
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda')
    
    parser.add_argument('--use_pretrained_features', action='store_true', default=True)
    parser.add_argument('--roi_features_path', default=r"harmeme_ROI_MOMENTA/cov/harmfulness/harmeme_cov_{split}_ROI.pt")
    parser.add_argument('--ent_features_path', default=r"harmeme_ENT_MOMENTA/cov/harmeme_cov_harmfulness/harmeme_cov_{split}_ent.pt")
    
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--patience', type=int, default=5)
    
    parser.add_argument("--feature_path_train", type=str, default="harmc_features_momenta/train_fused_feats.pt")
    parser.add_argument("--feature_path_dev", type=str, default="harmc_features_momenta/test_fused_feats.pt")
    parser.add_argument("--neighbor_path_train", type=str, default="top_10_neighbors_covid19_qwen_train.json")
    parser.add_argument("--neighbor_path_dev", type=str, default="top_10_neighbors_covid19_qwen_test.json")
    
    parser.add_argument("--ib_beta", type=float, default=0.0001)
    parser.add_argument("--ib_dim", type=int, default=256)
    parser.add_argument("--disable_ib", action="store_true", default=False)
    
    parser.add_argument("--max_updates", type=int, default=22000)
    parser.add_argument("--num_warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--log_every", type=int, default=50)
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    use_ib = not args.disable_ib

    train_dataset = HatefulMemeDataset(args.train_json, args.image_dir)
    val_dataset = HatefulMemeDataset(args.val_json, args.image_dir)
    test_dataset = HatefulMemeDataset(args.test_json, args.image_dir)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = HatefulMemeClassifierIB(
        clip_model_name="openai/clip-vit-base-patch16",
        feature_path_train=args.feature_path_train,
        feature_path_dev=args.feature_path_dev,
        neighbor_path_train=args.neighbor_path_train,
        neighbor_path_dev=args.neighbor_path_dev,
        ib_beta=args.ib_beta,
        ib_dim=args.ib_dim,
        use_ib=use_ib
    ).to(device)

    optimizer = optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "clip" not in n and p.requires_grad], 
             "lr": args.lr, "weight_decay": args.weight_decay},
        ],
        eps=1e-8
    )
    
    total_steps = args.max_updates
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=total_steps
    )

    best_val_acc = 0.0
    patience_counter = 0
    
    global_step = 0
    for epoch in range(args.epochs):
        train_loss, train_acc, avg_ce, avg_ib, global_step = train_one_epoch(
            model, train_dataloader, optimizer, scheduler, None, device,
            neighbor_path_train=args.neighbor_path_train,
            feature_path_train=args.feature_path_train,
            global_step=global_step,
            max_updates=args.max_updates,
            log_every_batches=args.log_every
        )
        
        val_f1, val_auc, val_acc = evaluate(
            model, val_dataloader, device,
            neighbor_path_dev=args.neighbor_path_dev,
            feature_path_dev=args.feature_path_dev
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {patience_counter} epochs without improvement.")
                break
        
        log_path = os.path.join(args.output_dir, f"training_log.txt")
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(
                f"Epoch {epoch+1}: TrainLoss={train_loss:.6f}, CE={avg_ce:.6f}, IB={avg_ib:.6f}, "
                f"TrainAcc={train_acc:.4f}, ValF1={val_f1:.4f}, ValAUC={val_auc:.4f}, ValAcc={val_acc:.4f}, GlobalStep={global_step}\n"
            )
        
        print(
            f"Epoch {epoch+1}: TrainLoss={train_loss:.6f}, CE={avg_ce:.6f}, IB={avg_ib:.6f}, "
            f"TrainAcc={train_acc:.4f}, ValF1={val_f1:.4f}, ValAUC={val_auc:.4f}, ValAcc={val_acc:.4f}, GlobalStep={global_step}"
        )
        
        if global_step >= args.max_updates:
            print("Reached max_updates. Early stopping.")
            break

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt")))
    test_f1, test_auc, test_acc = evaluate(
        model, test_dataloader, device,
        neighbor_path_dev=args.neighbor_path_dev,
        feature_path_dev=args.feature_path_dev
    )
    
    print(f"Test Results: F1={test_f1:.4f}, AUC={test_auc:.4f}, Acc={test_acc:.4f}")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"Test Results: F1={test_f1:.4f}, AUC={test_auc:.4f}, Acc={test_acc:.4f}\n")

    print(f"Training finished. Logs saved to {log_path}")

if __name__ == "__main__":
    main()