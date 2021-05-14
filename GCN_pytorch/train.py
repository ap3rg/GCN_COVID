import os
from networkx.readwrite.json_graph.adjacency import adjacency_data
import pandas as pd
import networkx as nx

# Torch imports
import torch
import torch.optim as optim

# Local imports
from GCN_pytorch.net import GCN
import GCN_pytorch.functions as fun
import GCN_pytorch.model_config as conf
import GCN_pytorch.model_constants as cons

# get data
times = [1, 2, 3, 4]
graphs = []             # Ordered timeseries
adj_matrices = []       # Ordered timeseries
node_attrs = []         # Ordered timeseries

for t in times:
    edge_data = os.path.join(conf.raw_dir, t, "edgelist.csv")
    node_data = os.path.join(conf.raw_dir, t, "node_attrs.csv")
    
    # Get node attrs
    node_attrs.append(pd.read_csv(node_data)["cases"].values)

    # load graph
    Graph = nx.Graph()
    G = nx.from_pandas_edgelist(edge_data, edge_attr='weight', create_using=Graph)
    graphs.append(G)

    # Calculate spectral normalized matrix
    A_norm = fun.calculate_spectral_norm_matrix(G)
    adj_matrices.append(A_norm)

num_nodes = len(node_attrs[0])
first_embedding = num_nodes*cons.EMB_WINDOW

# Model and optimizer
model = GCN(emb=first_embedding,
            h_0=num_nodes,
            hop_1=num_nodes,
            hop_2=num_nodes,
            p=num_nodes,
            dropout=cons.DROPOUT)

optimizer = optim.Adam(model.parameters(),
                       lr=cons.LEARNING_RATE, weight_decay=cons.L2_REGULARIZATION)

