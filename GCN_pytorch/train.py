import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# Torch imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports
from net import GCN
import functions as fun
import model_config as conf
import model_constants as cons
from dataset import GCN_Dataset

def train_model(num_epochs):

    # Node attributes
    node_attrs = ["num_cases", "num_diseased"]
    node_attrs = ["num_cases"]

    # Get dataset
    dataset = GCN_Dataset(node_attrs=node_attrs, 
                        prediciton_attr="num_cases", 
                        node_attrs_file="cases.csv", 
                        movement_file="movement.csv")


    num_nodes = dataset.num_nodes
    num_attrs = len(node_attrs)

    n_features = num_nodes * num_attrs * (cons.EMB_WINDOW + 1)

    # Model
    model = GCN(n_features=n_features,
                n_nodes=num_nodes, 
                dropout=cons.DROPOUT)

    model.zero_grad()     # zeroes the gradient buffers of all parameters

    # Optimizer
    optimizer = optim.Adam(model.parameters(),
                        lr=cons.LEARNING_RATE, weight_decay=cons.L2_REGULARIZATION)

    # Loss function
    loss_function = torch.nn.MSELoss()
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        loss_values = []

        if epoch % 20 == 0:
            print(f"Epoch {epoch}")

        for idx in range(len(dataset)):
            if idx % 100 == 0:
                if cons.DEBUG: print(f"Iter: {idx}")

            data = dataset[idx]
            adj_matrix = data["adj_matrix"]
            first_embedding = data["first_embedding"]
            target = data["prediction"].unsqueeze(0)    # Add batch size

            x = first_embedding.view(cons.BATCH_SIZE, -1)
            x = x.type(torch.FloatTensor)
            
            optimizer.zero_grad()

            # Calculate loss
            prediction = model(x, adj_matrix)
            loss = loss_function(prediction, target)

            # Get loss values for eval
            running_loss += loss.item()
            loss_values.append(loss.detach().numpy())

            # Update
            loss.backward()
            optimizer.step()

        losses.append(running_loss)
        if cons.DEBUG: print(running_loss)

    plt.plot(np.array(losses), 'r')
    plt.show()



train_model(100)

