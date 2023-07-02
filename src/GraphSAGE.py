import numpy as np
import gc
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GCNConv,SAGEConv,GAE
from torch_geometric.data import Data
from tqdm import tqdm
import polars as pl
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import NeighborLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

aid_features = pl.read_parquet("/kaggle/input/otto-100xfeatures/aid_features.parquet").fill_null(0)
train = pl.read_parquet("/kaggle/input/otto-full-optimized-memory-footprint/train.parquet")

std = StandardScaler().fit(aid_features)
node_features_scaled = std.transform(aid_features)
edge_index = torch.tensor(train[['session', 'aid']].to_numpy(), dtype=torch.long).t().contiguous()

data = Data(x=torch.tensor(node_features_scaled),
            edge_index= edge_index,)
data.n_id = torch.arange(data.num_nodes)

gSAGE_loader = NeighborLoader(
    data,
    num_neighbors=[5,3],
    batch_size=128)

# kernel dead

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x