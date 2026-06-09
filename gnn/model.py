"""
Graph Attention Network for BaSISS clone prediction.
Two GAT layers with skip connection, dropout regularisation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm


class CloneGAT(nn.Module):
    """
    Two-layer Graph Attention Network for clone identity classification.

    Architecture:
        Input  →  GAT(hidden)  →  BatchNorm  →  ReLU  →  Dropout
               →  GAT(hidden)  →  BatchNorm  →  ReLU  →  Dropout
               →  Linear(n_classes)
    """
    def __init__(self, in_channels, hidden_channels, n_classes,
                 heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # Layer 1: multi-head attention
        self.conv1 = GATConv(in_channels, hidden_channels,
                             heads=heads, dropout=dropout, concat=True)
        self.bn1   = BatchNorm(hidden_channels * heads)

        # Layer 2: single-head attention (concat=False → averages heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels,
                             heads=1, dropout=dropout, concat=False)
        self.bn2   = BatchNorm(hidden_channels)

        # Final classifier
        self.lin = nn.Linear(hidden_channels, n_classes)

    def forward(self, x, edge_index):
        # Layer 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)

        # Layer 2
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        return self.lin(x)   # logits: (n_nodes, n_classes)
