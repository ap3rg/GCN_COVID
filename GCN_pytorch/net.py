import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GCN_impl
import model_constants as cons


class GCN(nn.Module):
    
    def __init__(self, n_features, n_nodes, dropout):
        super(GCN, self).__init__()
        self.mlp1 = nn.Linear(n_features, n_nodes, bias=False)  # input: (Nx8), output: (Nx1)
        self.gc1 = GCN_impl(1, n_nodes)                    # input: (Nx1), output: (Nx1)
        self.gc2 = GCN_impl(n_nodes + 1, n_nodes)                # input: (Nx2), output: (Nx1)
        self.pred = nn.Linear((n_nodes + 1) * n_nodes, n_nodes, bias=False)      # input: (Nx2), output: (Nx1)
        self.dropout = dropout

    # Forward receives the embedded node attribute vector
    def forward(self, x, adj):
        H_0 = F.relu(self.mlp1(x))                                  # x: (1x(num_attr*embedding_w*N))
        H_0 = torch.transpose(H_0, 0, 1)                            # H_0: (Nx1)
        H_0 = H_0.type(torch.FloatTensor)                           # H_0: (1xN)

        H_1 = F.relu(self.gc1(H_0, adj))                            # H_1: (NxN)
        H_1 = F.dropout(H_1, self.dropout, training=self.training)  # Dropout
        H_1 = torch.cat([H_1 , H_0], dim=-1)                        # Concat H_0 -> H_1: (NxN+1)

        H_2 = F.relu(self.gc2(H_1, adj))                            # H_2: (NxN)
        H_2 = F.dropout(H_2, self.dropout, training=self.training)  # Dropout
        H_2 = torch.cat([H_2 , H_0], dim=-1)                        # Concat H_0 -> H_2: (NxN+1)

        H_2 = H_2.view(1, -1)                                       # H_2: (1x(N*N+1))

        P = F.relu(self.pred(H_2))                                  # P: (1xN)

        return P