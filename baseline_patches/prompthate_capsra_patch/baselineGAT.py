import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import RobertaConfig, RobertaForMaskedLM


def load_neighbors(neighbor_path):
    with open(neighbor_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}


def load_precomputed_features(feature_path, device):
    data = torch.load(feature_path, map_location=device)
    return {str(k): v.to(device) for k, v in data.items()}


class GATLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.fc = nn.Linear(hidden_dim * heads, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc(x)


class RobertaPromptWithGAT(nn.Module):
    def __init__(
        self,
        label_list,
        neighbor_path=None,
        feature_path=None,
        alpha=0.5,
        device=None,
        gat_hidden_dim=256,
        gat_dropout=0.2,
        gat_heads=4,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = RobertaConfig.from_pretrained("roberta-large", output_hidden_states=True)
        self.roberta = RobertaForMaskedLM.from_pretrained("roberta-large", config=self.config).to(self.device)
        self.label_word_list = label_list
        self.alpha = alpha
        self.neighbors = load_neighbors(neighbor_path) if neighbor_path else {}
        self.precomp_features = load_precomputed_features(feature_path, self.device) if feature_path else {}
        hidden_size = self.roberta.config.hidden_size
        self.gat = GATLayer(hidden_size, gat_hidden_dim, hidden_size, dropout=gat_dropout, heads=gat_heads).to(self.device)
        self.gat_classifier = nn.Linear(hidden_size, len(self.label_word_list)).to(self.device)

    def set_split_data(self, neighbor_path, feature_path):
        self.neighbors = load_neighbors(neighbor_path) if neighbor_path else {}
        self.precomp_features = load_precomputed_features(feature_path, self.device) if feature_path else {}

    def forward(self, meme_ids, tokens, attention_mask, mask_pos):
        batch_size = tokens.size(0)
        tokens = tokens.to(self.device)
        attention_mask = attention_mask.to(self.device)
        mask_pos = mask_pos.to(self.device).squeeze()

        outputs = self.roberta(tokens, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-1]
        text_feats = hidden_states[torch.arange(batch_size, device=self.device), mask_pos]
        prediction_mask_scores = outputs.logits[torch.arange(batch_size, device=self.device), mask_pos]
        text_logits = torch.stack([prediction_mask_scores[:, wid] for wid in self.label_word_list], dim=-1)

        if not self.neighbors or not self.precomp_features or self.alpha == 1.0:
            return text_logits

        all_nodes = []
        edge_index = []
        idx_map = {}
        for i, mid in enumerate(meme_ids):
            key = str(mid)
            idx_map[key] = len(all_nodes)
            all_nodes.append(text_feats[i].unsqueeze(0))

        for mid in meme_ids:
            key = str(mid)
            if key not in self.neighbors:
                continue
            for nb in self.neighbors[key].get("neighbors", []):
                nb_key = str(nb)
                if nb_key not in idx_map and nb_key in self.precomp_features:
                    idx_map[nb_key] = len(all_nodes)
                    all_nodes.append(self.precomp_features[nb_key].unsqueeze(0))
                if key in idx_map and nb_key in idx_map:
                    edge_index.append([idx_map[key], idx_map[nb_key]])

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).T.to(self.device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long).to(self.device)
        x = torch.cat(all_nodes, dim=0).to(self.device)
        gat_out = self.gat(x, edge_index)
        core_gat = gat_out[:batch_size]
        gat_logits = self.gat_classifier(core_gat)
        return self.alpha * text_logits + (1.0 - self.alpha) * gat_logits


def build_baseline_with_gat(opt, label_list):
    return RobertaPromptWithGAT(
        label_list=label_list,
        neighbor_path=getattr(opt, "neighbor_path_train", None),
        feature_path=getattr(opt, "feature_path_train", None),
        alpha=getattr(opt, "alpha", 1),
        device=getattr(opt, "device", None),
        gat_hidden_dim=getattr(opt, "gat_hidden_dim", 256),
        gat_dropout=getattr(opt, "gat_dropout", 0.2),
        gat_heads=getattr(opt, "gat_heads", 4),
    )