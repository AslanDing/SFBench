import torch
import torch.nn as nn
from torch.nn import Module, ModuleList
from abc import ABC, abstractmethod
from torch.nn.functional import mse_loss, relu
from torch_geometric.nn import GATConv, GCNConv, GCN2Conv, Linear
from torch_geometric.utils import add_self_loops
import numpy as np
from scipy.spatial import Delaunay

class BaseModel(Module, ABC):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, layerfun, edge_orientation, edge_weights):
        super().__init__()
        self.encoder = Linear(in_channels, hidden_channels, weight_initializer="kaiming_uniform")
        self.decoder = Linear(hidden_channels, 1, weight_initializer="kaiming_uniform")
        if param_sharing:
            self.layers = ModuleList(num_hidden * [layerfun()])
        else:
            self.layers = ModuleList([layerfun() for _ in range(num_hidden)])
        self.edge_weights = edge_weights
        self.edge_orientation = edge_orientation
        if self.edge_weights is not None:
            self.loop_fill_value = 1.0 if (self.edge_weights == 0).all() else "mean"

    def forward(self, x, edge_index=None, evo_tracking=False):
        if edge_index == None:
            edge_index = self.edge_index.to(x.device)
        x = x.flatten(1)
        if self.edge_weights is not None:
            num_graphs = edge_index.size(1) // len(self.edge_weights)
            edge_weights = torch.cat(num_graphs * [self.edge_weights], dim=0).to(x.device)
            edge_weights = edge_weights.abs()  # relevant when edge weights are learned
        else:
            edge_weights = torch.zeros(edge_index.size(1)).to(x.device)

        if self.edge_orientation is not None:
            if self.edge_orientation == "upstream":
                edge_index = edge_index[[1, 0]].to(x.device)
            elif self.edge_orientation == "bidirectional":
                edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1).to(x.device)
                edge_weights = torch.cat(2 * [edge_weights], dim=0).to(x.device)
            elif self.edge_orientation != "downstream":
                raise ValueError("unknown edge direction", self.edge_orientation)
        if self.edge_weights is not None:
            edge_index, edge_weights = add_self_loops(edge_index, edge_weights, fill_value=self.loop_fill_value)

        x_0 = self.encoder(x)
        evolution = [x_0.detach()] if evo_tracking else None

        x = x_0
        for layer in self.layers:
            x = self.apply_layer(layer, x, x_0, edge_index, edge_weights)
            if evo_tracking:
                evolution.append(x.detach())
        x = self.decoder(x)

        if evo_tracking:
            return x, evolution
        return x

    @abstractmethod
    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        pass

class GCN(BaseModel):
    def __init__(self, input_length,span_length,output_length,enc_in, dec_in, c_out,
                 edge_index,
                 hidden_channels=128, num_hidden=2, param_sharing=False, edge_orientation='bidirectional',
                 edge_weights=None): #
        in_channels = enc_in
        self.edge_index= edge_index
        layer_gen = lambda: GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return relu(layer(x, edge_index, edge_weights))


def generate_edfe_weights(points): # [nums:2]

    tri = Delaunay(points)
    edges = tri.simplices # [2,edges] numpy
    edge_index = torch.from_numpy(edges)
    edge_weights = nn.Parameter(torch.nn.init.uniform_(torch.empty(edge_index.shape[1]), 0.9, 1.1))
    return edge_index, edge_weights

