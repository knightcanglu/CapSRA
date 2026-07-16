import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import RobertaForMaskedLM, RobertaTokenizer


def load_neighbors(neighbor_path):
    with open(neighbor_path, "r") as f:
        return json.load(f)


def load_retrieval_features(feature_path, device):
    data = torch.load(feature_path, map_location=device)
    return {str(k): v.to(device) for k, v in data.items()}


class GATLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
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
        return self.fc(x)


class PromptHateModel(nn.Module):
    def __init__(
        self,
        label_words,
        max_length=320,
        model_name="roberta-large",
        neighbor_path=None,
        feature_path=None,
        alpha=0.5,
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.roberta = RobertaForMaskedLM.from_pretrained(model_name).to(self.device)
        self.roberta.config.output_hidden_states = True
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.mask_token_id = self.tokenizer.mask_token_id
        self.label_word_list = [
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(" " + word)[0])
            for word in label_words
        ]
        hidden_size = self.roberta.config.hidden_size
        self.gat = GATLayer(hidden_size, 256, hidden_size).to(self.device)
        self.gat_classifier = nn.Linear(hidden_size, len(label_words)).to(self.device)
        self.alpha = alpha
        self.neighbors = load_neighbors(neighbor_path) if neighbor_path else {}
        self.retrieval_features = load_retrieval_features(feature_path, self.device) if feature_path else {}

    def set_split_data(self, neighbor_path, feature_path):
        self.neighbors = load_neighbors(neighbor_path)
        self.retrieval_features = load_retrieval_features(feature_path, self.device)

    def generate_input_tokens(self, sents):
        token_info = self.tokenizer(
            sents,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = token_info.input_ids
        attention_mask = token_info.attention_mask
        mask_pos = []
        for token_ids in tokens:
            positions = (token_ids == self.mask_token_id).nonzero(as_tuple=False)
            mask_pos.append(positions[0].item() if positions.numel() else self.max_length - 1)
        return tokens, attention_mask, torch.LongTensor(mask_pos)

    def forward(self, meme_ids, all_texts):
        batch_size = len(all_texts)
        tokens, attention_mask, mask_pos = self.generate_input_tokens(all_texts)
        tokens = tokens.to(self.device)
        attention_mask = attention_mask.to(self.device)
        mask_pos = mask_pos.to(self.device)

        outputs = self.roberta(tokens, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-1]
        text_feats = hidden_states[torch.arange(batch_size, device=self.device), mask_pos]
        mlm_scores = outputs.logits[torch.arange(batch_size, device=self.device), mask_pos, :]
        text_logits = torch.stack([mlm_scores[:, wid] for wid in self.label_word_list], dim=-1)

        if not self.neighbors or not self.retrieval_features or self.alpha == 1.0:
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
                if nb_key not in idx_map and nb_key in self.retrieval_features:
                    idx_map[nb_key] = len(all_nodes)
                    all_nodes.append(self.retrieval_features[nb_key].unsqueeze(0))
                if key in idx_map and nb_key in idx_map:
                    edge_index.append([idx_map[key], idx_map[nb_key]])

        edge_index = (
            torch.tensor(edge_index, dtype=torch.long).T.to(self.device)
            if edge_index
            else torch.zeros((2, 0), dtype=torch.long).to(self.device)
        )
        x = torch.cat(all_nodes, dim=0).to(self.device)
        gat_out = self.gat(x, edge_index)[:batch_size]
        gat_logits = self.gat_classifier(gat_out)
        return self.alpha * text_logits + (1 - self.alpha) * gat_logits


def build_baseline(
    label_words,
    max_length,
    model_name="roberta-large",
    neighbor_path=None,
    feature_path=None,
    alpha=0.5,
    device=None,
):
    return PromptHateModel(label_words, max_length, model_name, neighbor_path, feature_path, alpha, device)