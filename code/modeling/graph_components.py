import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv, RGATConv


class Net(torch.nn.Module):
    def __init__(self, num_node_features, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(
            in_channels=num_node_features,
            out_channels=num_node_features,
            num_relations=num_relations,
        )

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        return x


class GNNDropout(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.drop = nn.Dropout(params.hidden_dropout_prob)

    def forward(self, inp):
        x, edge_index, edge_type = inp
        return self.drop(x), edge_index, edge_type


class GNNReLu(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, inp):
        x, edge_index, edge_type = inp
        return self.relu(x), edge_index, edge_type


class GNNConv(torch.nn.Module):
    def __init__(self, params, num_rels):
        super().__init__()
        self.params = params
        if self.params.gnn == "rgcn":
            self.gnn = RGCNConv(
                in_channels=self.params.node_emb_dim,
                out_channels=self.params.node_emb_dim,
                num_relations=num_rels,
            )
        elif self.params.gnn == "rgat":
            self.gnn = RGATConv(
                in_channels=self.params.node_emb_dim,
                out_channels=self.params.node_emb_dim,
                num_relations=num_rels,
            )

    def forward(self, inp):
        x, edge_index, edge_type = inp
        return self.gnn(x, edge_index, edge_type), edge_index, edge_type


class DeepNet(torch.nn.Module):
    def __init__(self, params, num_rels):
        super().__init__()
        self.params = params
        self.rgcn_layers = []
        for i in range(self.params.gnn_depth - 1):
            self.rgcn_layers.append(GNNConv(self.params, num_rels))
            self.rgcn_layers.append(GNNReLu(self.params))
            self.rgcn_layers.append(GNNDropout(self.params))

        self.rgcn_layers.append(GNNConv(self.params, num_rels))
        self.rgcn_layers.append(GNNReLu(self.params))
        self.rgcn_module = nn.Sequential(*self.rgcn_layers)

    def forward(self, x, edge_index, edge_type):
        x, edge_index, edge_type = self.rgcn_module((x, edge_index, edge_type))
        return x
