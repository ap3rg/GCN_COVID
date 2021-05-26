import os
import math
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

def train_model(num_epochs, dataset, end_idx, num_nodes, num_attrs, save=True):


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
            print(f"\tEpoch {epoch}")

        for idx in range(len(dataset)):
            if idx == end_idx:
                break

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

    if save:
        export_folder = os.path.join(conf.model_path, f"pred_window_{cons.PRED_WINDOW}")
        if not os.path.exists(export_folder):
            os.makedirs(export_folder) 
        out_path = os.path.join(export_folder, "base_0.pth")
        torch.save(model.state_dict(), out_path)
        plt.plot(np.array(losses), 'r')
        plt.savefig(os.path.join(export_folder, "trainning_losses.png"))

    return model

def test_model(model, dataset, start_idx):

    targets = []
    predictions = []
    errors = []

    model.eval()

    with torch.no_grad():
        for idx in range(len(dataset)):
            if idx <= start_idx:
                continue

            data = dataset[idx]
            adj_matrix = data["adj_matrix"]
            first_embedding = data["first_embedding"]
            target = data["prediction"].unsqueeze(0)    # Add batch size

            x = first_embedding.view(cons.BATCH_SIZE, -1)
            x = x.type(torch.FloatTensor)

            prediction = model(x, adj_matrix)

            error_function = torch.nn.MSELoss()
            error = error_function(prediction, target)

            targets.append(target[0])
            predictions.append(prediction[0])
            errors.append(error)
           
    return targets, predictions, errors


def main(train=True, test=True):

    # Node attributes
    node_attrs = ["num_cases", "num_diseased"]
    node_attrs = ["num_cases"]

    # Get dataset
    print("Building dataset ...")
    dataset = GCN_Dataset(node_attrs=node_attrs, 
                        prediciton_attr="num_cases", 
                        node_attrs_file="cases.csv", 
                        movement_file="movement.csv")


    num_nodes = dataset.num_nodes
    num_attrs = len(node_attrs)

    train_length = math.ceil(len(dataset) * 0.75)
    export_folder = os.path.join("out", f"pred_window_{cons.PRED_WINDOW}")
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    if train:
        print(f"Trainning model with prediction window {cons.PRED_WINDOW} days...")
        trained_model = train_model(50, dataset, train_length, num_nodes, num_attrs)
    else:
        n_features = num_nodes * num_attrs * (cons.EMB_WINDOW + 1)

        # Model
        trained_model = GCN(n_features=n_features,
                    n_nodes=num_nodes, 
                    dropout=cons.DROPOUT)

        state_dict_path = os.path.join(conf.model_path, f"pred_window_{cons.PRED_WINDOW}", "base_0.pth")
        trained_model.load_state_dict(torch.load(state_dict_path))

    if test:   
        print("Testing model...") 

        targets, predictions, errors = test_model(trained_model, dataset, train_length)

        node_list = list(range(num_nodes))
        targets = ["\t".join([str(x) for x in t.numpy()]) for t in targets]              # from tensors to array to tsv 
        predictions = ["\t".join([str(x) for x in t.numpy()]) for t in predictions]
        errors = [str(t.item()) for t in errors]    # from tensors to array to tsv 

        with open(os.path.join(export_folder, "target.csv"), "w") as target_file:
            target_file.write("\t".join([str(n) for n in node_list]) + "\n")
            target_file.write("\n".join(targets))
        with open(os.path.join(export_folder, "predictions.csv"), "w") as pred_file:
            pred_file.write("\t".join([str(n) for n in node_list]) + "\n")
            pred_file.write("\n".join(predictions))
        with open(os.path.join(export_folder, "errors.csv"), "w") as error_file:
            error_file.write("\n".join(errors))



main(train=True, test=True)