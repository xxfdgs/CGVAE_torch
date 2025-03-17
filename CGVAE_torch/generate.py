import torch

#!/usr/bin/env python3
import torch.nn as nn
import torch.nn.functional as F

from models.GVAE import GraphVAE
from torch_geometric.data import Data
import networkx as nx

from utils.load_args import load_params
from torch_geometric.explain import Explanation
import os

def load_model(checkpoint_path, device, config):
    model = GraphVAE(config)
    # Load checkpoint; adjust key if necessary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def generate_molecule(model, device='cpu',embed_dim=64):
    # Sample a 64-d Gaussian random embedding
    #随便选的节点数
    n = 5
    embedding = torch.randn((n, embed_dim), device=device)
    model.eval()
    with torch.no_grad():
        atom_type, edge_index, edge_attr = model.generate(embedding)
        print(atom_type,edge_index, edge_attr)

    # Create a PyG Data object
    data = Data(x=atom_type, edge_index=edge_index, edge_attr=edge_attr)

    #explain = Explanation(x=atom_type,edge_index=edge_index, edge_mask = edge_attr)
    
    #explain.visualize_graph(path="molecular.png")

    '''
    # Convert to a NetworkX graph for visualization
    G = nx.Graph()
    num_nodes = atom_type.size(0)
    for i in range(num_nodes):
        # Optionally, use atom type information as a node label
        G.add_node(i, label=str(atom_type[i].item() if atom_type.dim() == 1 else atom_type[i].cpu().numpy()))

    edge_index_np = edge_index.cpu().numpy()
    for src, dst in zip(edge_index_np[0], edge_index_np[1]):
        G.add_edge(src, dst)

    # Draw the graph
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='skyblue', node_size=500, edge_color='gray')
    plt.savefig("molecular.png")
    plt.show()

    molecule_rep = data
    '''


    #return molecule_rep

if __name__ == '__main__':
    config = load_params()
    checkpoint_dir = "/home/puzexuan/study/code/CGVAE/CGVAE_torch/out/checkpoints/result_GVAE_ZINC_cuda:0_19h14m07s_on_Mar_14_2025"
    pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not pth_files:
        raise FileNotFoundError("No .pth files found in the directory")
    checkpoint_path = os.path.join(checkpoint_dir, pth_files[0])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(checkpoint_path, device,config)
    molecule = generate_molecule(model, device,config["net_params"]["out_channels"])
    print("Generated molecule representation:")
    print(molecule)