from datetime import date
import os
import numpy as np
import pandas as pd
import networkx as nx

# Torch imports 
import torch
from torch.nn.functional import embedding
from torch.utils.data import Dataset, DataLoader


# Local imports
import model_config as conf
import model_constants as cons
import functions as fun

indent = "  "

# default type
# torch.set_default_dtype(torch.float64)

class GCN_Dataset(Dataset):

    def __init__(self, node_attrs, prediciton_attr, node_attrs_file="cases.csv", movement_file="movement.csv"):
        """
        Args:
            node_attrs (list): List of node attributes to add as features 
                as they appear in node_attrs_file. e.g. ["num_cases"].
            node_attrs_file (string): Path to the csv file with node attributes.
            movement_file (string): Path to the csv with movement data.
        """
        self.node_attrs_file = node_attrs_file
        self.movement_file = movement_file
        self.node_attrs = node_attrs
        self.prediciton_attr = prediciton_attr

        self.df_data = self.construct_dataset()
    

    def __len__(self):
        """Accounts for embedding window and prediction window. i.e. real number
        of usable training examples.
        """
        # len = self.df_data.shape[0] - cons.EMB_WINDOW - cons.PRED_WINDOW
        return len(self.dates)

    def __getitem__(self, idx):
        """
        Args:
            date_time (pd.Timestamp or string): date to fetch training example from
        Returns:
            sample (dict [Tensor, Tensor, Tensor]): 
                    adj_matrix : adjacency graph at time date_time,
                    first_embedding: node_features corresponding to embedding window
                    cases: associate number of cases based on prediction window
        """
        date_time = self.dates[idx]
        ajd_matrix = self.df_data.set_index("date_time").at[date_time, "adj_matrix"]
        ajd_matrix_tensor = torch.tensor(ajd_matrix.todense())
        first_embedding = self.construct_first_embedding(date_time)

        # Get prediction
        # TODO: maybe make prediction dimensional based on node attributes?
        prediction = self.get_prediction(date_time)
        
        sample = {'adj_matrix': ajd_matrix_tensor.type(torch.FloatTensor), 
                'first_embedding': first_embedding.type(torch.FloatTensor),
                'prediction': prediction.type(torch.FloatTensor)}

        return sample

    def get_dates(self):
        return self.dates

    def get_slice(self, start, end):
        assert type(start) == type(end)
        if isinstance(start, pd.Timestamp):
            start_date = start
            end_date = end
        else:
            start_date = self.dates[start]
            end_date = self.dates[end]
        
        df_slice = self.df_data[(self.df_data["date_time"] >= start_date) & (self.df_data["date_time"] < end_date)]
        return df_slice

    def imputate_datapoint(self, df, date_times):
        """
        Args:
            date_times (list of pd.Timestamp): list of dates to imputate
            df (DataFrame): dataframe to imputate
        Returns:
            DataFrame with date rows replaced by the equivalent row of the day before
        """
        #TODO 
        #MOCK
        print(f"Imputating {date_times} dates")
        for d in date_times:
            imputation_date = d - pd.Timedelta(1, unit="d")
            df_tmp = df.loc[df["date_time"] == imputation_date].copy()
            df_tmp["date_time"] = d
            df = df.append(df_tmp, ignore_index=True)
        return df


    def construct_dataset(self):

        df_node_attrs = pd.read_csv(os.path.join(conf.raw_dir, self.node_attrs_file), parse_dates=["date_time"])
        df_movement = pd.read_csv(os.path.join(conf.raw_dir, self.movement_file), parse_dates=["date_time"])

        node_order = sorted(df_node_attrs.poly_id.unique())
        self.num_nodes = len(node_order)

        # Determine min datapoint to determine imputation 
        min_avail = df_movement.date_time.min()
        max_avail = df_movement.date_time.max()

        dates = pd.date_range(min_avail, max_avail)

        # Imputates data
        df_movement = pd.read_csv(os.path.join(conf.raw_dir, "movement.csv"), parse_dates=["date_time"])
        df_movement.set_index(["date_time", cons.SOURCE_NAME, cons.TARGET_NAME], inplace=True)
        df_movement = df_movement.reindex(pd.MultiIndex.from_product([dates, node_order, node_order])).fillna(0).reset_index()
        df_movement.rename(columns={"level_0": "date_time", "level_1": cons.SOURCE_NAME, "level_2": cons.TARGET_NAME}, inplace=True)

        # df_node_attrs = self.imputate_datapoint(df_node_attrs, node_attrs_dates_to_imputate)
        # df_movement = self.imputate_datapoint(df_movement, movement_dates_to_imputate)

        dates = np.intersect1d(df_movement.date_time.unique(), df_node_attrs.date_time.unique())

        # init general DataFrame
        columns = ["date_time", "adj_matrix"]

        for node_attr in self.node_attrs:
            columns.append(node_attr)

        # Init general DataFrame
        df_data = pd.DataFrame(columns=columns)
        
        # Construct dataset
        for i, d in enumerate(dates):
            
            if i % 50 == 0:
                print(f"{indent}{indent}Processing {d}.")

            # Add empty edges with cero weight to keep graphs consistent
            edge_data = df_movement[df_movement["date_time"] == d]
            # edge_data.set_index([cons.SOURCE_NAME, cons.TARGET_NAME], inplace=True)
            # edge_data = edge_data.reindex(pd.MultiIndex.from_product([node_order, node_order])).fillna(0).reset_index()
            # edge_data.rename(columns={"level_0": cons.SOURCE_NAME, "level_1": cons.TARGET_NAME}, inplace=True)
            # edge_data["date_time"] = d

            # Create graph
            Graph = nx.Graph()
            G = nx.from_pandas_edgelist(edge_data, source=cons.SOURCE_NAME, target=cons.TARGET_NAME, edge_attr=cons.EDGE_WEIGHT_NAME, create_using=Graph)

            # Calculate spectral adj matrix
            A_norm = fun.calculate_spectral_norm_matrix(G, nodelist=node_order)

            # append to lists
            nodes_attrs_dict = {}

            # Get node_attributes
            for node_attr in self.node_attrs:
                
                df_node_attr_tmp = df_node_attrs[df_node_attrs["date_time"] == d][[cons.NODE_NAME, node_attr]].groupby(cons.NODE_NAME).sum()
                df_node_attr_tmp = df_node_attr_tmp.reindex(node_order).fillna(0)
                node_attrs = df_node_attr_tmp[node_attr].values.tolist()

                nodes_attrs_dict[node_attr] = node_attrs

            nodes_attrs_dict["date_time"] = d
            nodes_attrs_dict["adj_matrix"] = A_norm

            df_data_tmp = pd.DataFrame.from_records([nodes_attrs_dict], columns=columns)
            
            # Append to general DataFrame
            df_data = df_data.append(df_data_tmp, ignore_index=True)

        # Set date attributes
        self.start_time = df_data["date_time"].min() + pd.Timedelta(cons.EMB_WINDOW, unit="d")
        self.end_time = df_data["date_time"].max() - pd.Timedelta(cons.PRED_WINDOW, unit="d")

        available_dates = []
        for d in sorted(df_data.date_time.unique()):
            if (d > self.start_time) and (d < self.end_time):
                available_dates.append(d)
        
        self.dates = available_dates

        print(f"{indent}Processed {len(dates)} distinct graphs. Added {len(self.node_attrs)} node attributes.")

        return df_data

    def construct_first_embedding(self, date_time):
        # For now only use one attr to construct embedding

        # Get window for embedding
        embedding_window = pd.date_range(date_time - pd.Timedelta(cons.EMB_WINDOW, unit="d"), date_time)
        df = self.df_data[self.df_data["date_time"].isin(embedding_window)].copy()
        df.set_index("date_time", inplace=True)
        df.sort_index(inplace=True, ascending=False)
        
        tensors = []
        for node_attr in self.node_attrs:
            node_attrs = []
            for d in df.index:
                attrs = df.at[d, node_attr]
                node_attrs.append(attrs)

            tensor = torch.tensor(node_attrs)
            tensors.append(tensor)
         
        return torch.dstack(tensors)


    def get_prediction(self, date_time):
        pred_date = date_time + pd.Timedelta(cons.PRED_WINDOW, unit="d")
        if pred_date not in self.df_data.date_time.unique():
            print(f"WARNING: prediction date {pred_date} not found in index. Imputing.")
            return self.get_prediction(pred_date)
        else:
            prediction = self.df_data.set_index("date_time").at[pred_date, self.prediciton_attr]
        return torch.tensor(prediction)