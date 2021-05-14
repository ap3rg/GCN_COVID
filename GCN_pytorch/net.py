import torch
import torch.nn as nn
import torch.nn.functional as F

from GCN_pytorch.layers import GCN


class GCN(nn.Module):
    
    def __init__(self, n_features, emb, dropout):
        super(GCN, self).__init__()
        self.mlp1 = nn.Linear(emb, n_features, bias=False)        # input: (Nx8), output: (Nx1)
        self.gc1 = GCN(n_features, n_features)                    # input: (Nx1), output: (Nx1)
        self.gc2 = GCN(n_features + 1, n_features)                # input: (Nx2), output: (Nx1)
        self.pred = nn.Linear(n_features + 1, 1, bias=False)      # input: (Nx2), output: (Nx1)
        self.dropout = dropout

    # Forward receives the embedded node attribute vector
    def forward(self, x, adj):
        H_0 = F.relu(self.mlp1(x))                                  # First embedding

        H_1 = F.relu(self.gc1(H_0, adj))                            # H_0: (Nx1)
        H_1 = F.dropout(H_1, self.dropout, training=self.training)  # Dropout
        H_1 = torch.cat([H_1 , H_0], dim=-1)                        # Concat H_0

        H_2 = F.relu(self.gc2(H_1, adj))                            # H_1: (Nx2)
        H_2 = F.dropout(H_2, self.dropout, training=self.training)  # Dropout
        H_2 = torch.cat([H_2 , H_0], dim=-1)                        # Concat H_0

        P = F.relu(self.pred(x))                                    # Prediction

        return P