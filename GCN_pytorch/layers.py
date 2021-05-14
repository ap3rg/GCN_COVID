import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GCN(Module):
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.FloatTensor(in_features, out_features))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)     # init from uniform distribution

    def forward(self, input, adj):
        support = torch.mm(input, self.weights)      # (H_i x W_i)
        output = torch.spmm(adj, support)           # (A X H_i x W_i)
        return output
