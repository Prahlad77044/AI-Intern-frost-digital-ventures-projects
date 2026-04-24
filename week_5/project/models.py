import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class GCN(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.c1 = GCNConv(in_c, hid_c)
        self.c2 = GCNConv(hid_c, hid_c)
        self.c3 = GCNConv(hid_c, out_c)

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, 0.3, training=self.training)
        x = F.relu(self.c2(x, edge_index))
        x = F.dropout(x, 0.3, training=self.training)
        return self.c3(x, edge_index)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.c1 = SAGEConv(in_c, hid_c)
        self.c2 = SAGEConv(hid_c, hid_c)
        self.c3 = SAGEConv(hid_c, out_c)

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, 0.3, training=self.training)
        x = F.relu(self.c2(x, edge_index))
        x = F.dropout(x, 0.3, training=self.training)
        return self.c3(x, edge_index)
