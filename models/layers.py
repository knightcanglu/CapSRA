import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv
from torch_scatter import scatter_mean

# --- (Aggregation Modules) ---

class AggregationModuleBase(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x, edge_index):
        raise NotImplementedError

class GATLayer(AggregationModuleBase):
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

class TransformerConvLayer(AggregationModuleBase):
    def __init__(self, in_dim=512, out_dim=512, n_heads=4, dropout=0.2):
        super().__init__(in_dim, out_dim)
        self.transformer_conv = TransformerConv(in_dim, out_dim, heads=n_heads, concat=True, dropout=dropout)
        self.linear = nn.Linear(out_dim * n_heads, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        x_attn = self.transformer_conv(x, edge_index)
        x = self.linear(x_attn)
        x = self.dropout(x)
        x = self.layer_norm(x + x_attn[:, :self.out_dim])
        return x

class WeightedAverageLayer(AggregationModuleBase):
    def __init__(self, in_dim=512, out_dim=512):
        super().__init__(in_dim, out_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        source_nodes, target_nodes = edge_index
        source_features = x[source_nodes]
        aggregated_features = scatter_mean(source_features, target_nodes, dim=0, dim_size=x.size(0))
        
        transformed_features = self.linear(aggregated_features)
        
        return x + transformed_features

# --- (Fusion Modules) ---

class FusionModuleBase(nn.Module):
    def __init__(self, dim1, dim2, fusion_dim):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.fusion_dim = fusion_dim

    def forward(self, x1, x2):
        raise NotImplementedError

class GatedFusion(FusionModuleBase):
    def __init__(self, dim1=512, dim2=512, fusion_dim=512):
        super().__init__(dim1, dim2, fusion_dim)
        assert dim1 == dim2 == fusion_dim, "For GatedFusion, all dims must be equal."
        self.gate_linear = nn.Linear(dim1 + dim2, fusion_dim)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.gate_linear(torch.cat([x1, x2], dim=1)))
        combined = gate * x1 + (1 - gate) * x2
        return combined

class ConcatFusion(FusionModuleBase):
    def __init__(self, dim1=512, dim2=512, fusion_dim=512):
        super().__init__(dim1, dim2, fusion_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim1 + dim2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.fusion_layer(x)

class AttentionFusion(FusionModuleBase):
    def __init__(self, dim1=512, dim2=512, fusion_dim=512):
        super().__init__(dim1, dim2, fusion_dim)
        self.attention_proj = nn.Linear(dim1 + dim2, 2)
        self.proj1 = nn.Linear(dim1, fusion_dim) if dim1 != fusion_dim else nn.Identity()
        self.proj2 = nn.Linear(dim2, fusion_dim) if dim2 != fusion_dim else nn.Identity()

    def forward(self, x1, x2):
        x1_proj = self.proj1(x1)
        x2_proj = self.proj2(x2)
        
        attention_input = torch.cat([x1_proj, x2_proj], dim=1)
        weights = F.softmax(self.attention_proj(attention_input), dim=1)
        
        combined = weights[:, 0].unsqueeze(1) * x1_proj + weights[:, 1].unsqueeze(1) * x2_proj
        return combined
