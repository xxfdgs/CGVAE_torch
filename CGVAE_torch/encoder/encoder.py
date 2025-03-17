import torch
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, BatchNorm
import torch.nn as nn
import torch.nn.functional as F


# GCNEncoder (unchanged)
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels,training=True,dropout=0.2):
        super(GCNEncoder, self).__init__()
        self.bns = nn.ModuleList()
        self.training = training
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = nn.ModuleList()
        self.dropout = dropout
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            #self.bns.append(BatchNorm(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                #self.bns.append(BatchNorm(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs):
            out = conv(x, edge_index, edge_attr)
            #if i != 0:
                #out = F.dropout(out, p=self.dropout, training=self.training)
            #    x = out + x
            #else:
            x = out
            #if i < len(self.bns):
            #    x = self.bns[i](x)
            x = F.relu(x)
            #x = F.dropout(x, p=self.dropout, training=self.training)
            if i == len(self.convs) - 3:
                break
        # Use the final two layers for mu and logvar
        mu = self.convs[-2](x, edge_index, edge_attr)
        logvar = F.softplus(self.convs[-1](x, edge_index, edge_attr))
        return mu, logvar

# GraphSAGE Encoder with BN and residual connections
class GraphSAGEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels):
        super(GraphSAGEncoder, self).__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_attr=None):
        # Process all layers except the last two (for mu and logvar)
        for i, conv in enumerate(self.convs):
            out = conv(x, edge_index, edge_attr)
            #if i != 0:
            #    x = out + x
            #else:
            x = out
            if i < len(self.bns):
                x = self.bns[i](x)
            x = F.relu(x)
            if i == len(self.convs) - 3:
                break
        mu = torch.sigmoid(self.convs[-2](x, edge_index, edge_attr))
        logvar = torch.sigmoid(self.convs[-1](x, edge_index, edge_attr))
        return mu, logvar

# Utility function for MLP in GINEncoder
def build_mlp(input_dim=64, hidden_dim=128, output_dim=64):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )

# GINEncoder with BN and residual connections
class GINEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels):
        super(GINEncoder, self).__init__()
        #print(f"in_channels: {in_channels}, hidden_channels: {hidden_channels}, num_layers: {num_layers}, out_channels: {out_channels}")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            mlp_mu = build_mlp(in_channels, hidden_channels, out_channels)
            self.convs.append(GINConv(mlp_mu))
            mlp_logvar = build_mlp(in_channels, hidden_channels, out_channels)
            self.convs.append(GINConv(mlp_logvar))
        else:
            print(in_channels)
            mlp = build_mlp(in_channels, hidden_channels, hidden_channels)
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                mlp = build_mlp(hidden_channels, hidden_channels, hidden_channels)
                self.convs.append(GINConv(mlp))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            mlp_mu = build_mlp(hidden_channels, hidden_channels, out_channels)
            self.convs.append(GINConv(mlp_mu))
            mlp_logvar = build_mlp(hidden_channels, hidden_channels, out_channels)
            self.convs.append(GINConv(mlp_logvar))

    def forward(self, x, edge_index, edge_attr=None):
        print(x.shape)
        for i, conv in enumerate(self.convs):
            out = conv(x, edge_index,edge_attr)
            if i != 0:
                x = out + x
            else:
                x = out
            if i < len(self.bns):
                x = self.bns[i](x)
            x = F.relu(x)
            print(x.shape)
            if i == len(self.convs) - 3:
                break
        print(x.shape)
        mu = torch.sigmoid(self.convs[-2](x, edge_index,edge_attr))
        logvar = torch.sigmoid(self.convs[-1](x, edge_index,edge_attr))
        return mu, logvar