import torch
import torch.nn as nn
from transformers import CLIPModel
from utils.utils import load_neighbors

class HatefulMemeClassifierIB(nn.Module):
    def __init__(
        self,
        aggregation_module,
        fusion_module,
        clip_model_name="openai/clip-vit-base-patch16",
        ib_beta=1e-3,
        ib_dim=512,
        use_ib=True
    ):
        super().__init__()
        
        self.use_ib = use_ib
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        for param in self.clip.parameters():
            param.requires_grad = False
        
        self.aggregation_module = aggregation_module
        self.fusion_module = fusion_module
        agg_out_dim = aggregation_module.out_dim
        self.mu_proj = nn.Linear(agg_out_dim, ib_dim)
        self.logsigma_proj = nn.Linear(agg_out_dim, ib_dim)
        if ib_dim != fusion_module.dim2:
            self.ib_to_fuse = nn.Linear(ib_dim, fusion_module.dim2)
        else:
            self.ib_to_fuse = nn.Identity()
        self.ib_beta = ib_beta
        self.classifier = nn.Sequential(
            nn.Linear(fusion_module.fusion_dim, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 2)
        )
        self.clip_features_cache = {}
        self.neighbors_cache = {}
        if not self.use_ib:
            print("INFO: IB module is disabled. Using aggregation outputs directly.")

    def load_external_data(self, feature_path, neighbor_path, device):
        self.load_clip_features(feature_path, device)
        self.load_neighbors(neighbor_path)

    def load_clip_features(self, feature_path, device):
        if feature_path is None:
            self.clip_features_cache = {}
            return
        try:
            loaded = torch.load(feature_path, map_location=device)
            self.clip_features_cache = {str(k): v.to(device) for k, v in loaded.items()}
        except Exception as e:
            print(f"Warning: load_clip_features error for {feature_path}: {e}. Using empty clip_features.")
            self.clip_features_cache = {}
    
    def load_neighbors(self, neighbor_path):
        if neighbor_path is None:
            self.neighbors_cache = {}
            return
        self.neighbors_cache = load_neighbors(neighbor_path)
    
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
        edge_index_list = []
        node_map = {str(mid): i for i, mid in enumerate(meme_ids)}
        
        for i, meme_id in enumerate(meme_ids):
            mid_str = str(meme_id)
            if mid_str not in self.neighbors_cache:
                continue
            for nb_id in self.neighbors_cache[mid_str].get("neighbors", []):
                nb_str = str(nb_id)
                if nb_str in self.clip_features_cache:
                    if nb_str not in node_map:
                        node_map[nb_str] = len(all_nodes)
                        all_nodes.append(self.clip_features_cache[nb_str].unsqueeze(0))
                    neighbor_idx = node_map[nb_str]
                    edge_index_list.append([i, neighbor_idx])
        
        x = torch.cat(all_nodes, dim=0).to(device)
        if len(edge_index_list) > 0:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        
        aggregated_features = self.aggregation_module(x, edge_index)
        batch_aggregated_features = aggregated_features[:batch_size]
        
        ib_loss = torch.tensor(0.0, device=device)
        if self.use_ib:
            mu = self.mu_proj(batch_aggregated_features)            
            log_sigma2 = self.logsigma_proj(batch_aggregated_features)  
            z0 = self.reparametrize(mu, log_sigma2)          
            z = self.ib_to_fuse(z0)                     
            if self.training:
                kl_div = -0.5 * torch.sum(1 + log_sigma2 - mu.pow(2) - torch.exp(log_sigma2), dim=1)  
                ib_loss = self.ib_beta * torch.mean(kl_div)
        else:
            z = batch_aggregated_features
        combined_features = self.fusion_module(clip_features, z)
        logits = self.classifier(combined_features)
        
        return logits, ib_loss
