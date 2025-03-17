import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import torch.nn as nn
import torch.nn.functional as F

from encoder.encoder import GCNEncoder, GraphSAGEncoder, GINEncoder
def check_nan_inf(tensor, name=""):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in tensor: {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in tensor: {name}")


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.bn(self.linear1(x)))
        x = self.linear2(x)
        return x
    
class GraphVAE(nn.Module):
    def __init__(self, config):
        super(GraphVAE, self).__init__()
        self.config = config
        num_layers = config["net_params"]["num_layers"]
        in_channels = config["net_params"]["in_channels"]
        hidden_channels = config["net_params"]["hidden_channels"]
        out_channels = config["net_params"]["out_channels"]
        #根据提前统计的数据集中的原子种类数量，设置输入通道数
        if config["data"]["dataset_name"] == "ZINC":
            
            num_atom_type = 28
        elif config["data"]["dataset_name"] == "QM9":
            num_atom_type = 9
        self.atom_embedding = nn.Embedding(num_atom_type, in_channels)
        

        if config["params"]["encoder"] == "GraphSAGE":
            self.encoder = GraphSAGEncoder(in_channels, hidden_channels, num_layers,out_channels)
        elif config["params"]["encoder"] == "GIN":
            self.encoder = GINEncoder(in_channels, hidden_channels, num_layers,out_channels)
        elif config["params"]["encoder"] == "GCN":
            self.encoder = GCNEncoder(in_channels, hidden_channels, num_layers,out_channels)
        else:
            raise ValueError(f"Unsupported encoder: {config['params']['encoder']}. Please choose 'GraphSAGE', 'GIN', or 'GCN'.")
        # Decoder can be implemented according to the desired task.
        # Often in graph VAEs, the decoder computes node similarity, e.g., using dot product.

        self.atom_type_predictor = MLP(out_channels, out_channels*2, num_atom_type)
        # 生成边的链接预测器
        #考虑0，1，2，3四种边的情况，所以输出通道数为4
        self.edge_link_predictor = MLP(out_channels*2, out_channels*4, 4)

    def reparameterize(self, mu, logvar):
        # 根据logvar计算标准差
        std = torch.exp(0.5 * logvar)
        # 生成与标准差相同形状的正态分布随机噪声
        eps = torch.randn_like(std)
        # 使用重参数化技巧重新参数化，以便进行反向传播
        return mu + eps * std

    def forward(self, x, edge_index,edge_attr, sampled_edge_index):
        """
        计算图的前向传播过程。
        参数:
            x (torch.Tensor): 节点特征张量，形状为 [N, feature_dim]，其中 N 为节点数。
                该张量会先经过原子嵌入层转换为嵌入向量。
            edge_index (torch.Tensor): 表示图中真实边连接关系的张量，形状为 [2, E]，
                其中 E 为边的数量。用于构建真实边的连接信息并传入编码器。
            neg_edge_index (torch.Tensor): 表示负采样边的张量，形状为 [2, E_neg]，
                用于辅助训练时构建负采样边。注意：如果其中含有 -1，则对应的边不参与计算。
        流程:
            1. 将输入节点特征 x 转换为长整型，并通过 atom_embedding 层进行嵌入，得到嵌入后的节点特征。
            2. 通过 encoder 层对嵌入后的节点特征以及边连接关系 edge_index 进行编码，得到均值 mu 和对数方差 logvar。
            3. 利用 reparameterize 操作根据 mu 和 logvar 生成隐变量 z，实现变分自编码器中的采样过程。
            4. 通过 atom_type_predictor 对隐变量 z 进行处理，预测每个节点的原子类型。
            5. 对于真实边:
                 - 使用 edge_index 对应的节点对，从隐变量 z 中提取对应节点的嵌入，
                 - 将两端节点的嵌入拼接后通过 edge_link_predictor，预测真实边的连接概率或评分 (true_edge_pred)。
            6. 对于负采样边:
                 - 检查 neg_edge_index 中是否存在 -1；
                 - 如果 neg_edge_index 中有 -1 的值，则将这些边过滤掉，不参与负样本计算；
                 - 对过滤后的 neg_edge_index，类似真实边处理方式，拼接对应节点嵌入后通过 edge_link_predictor 预测负边的连接评分 (neg_edge_pred)；
                 - 如果所有 neg_edge_index 都无效，则直接返回空张量。
        返回:
            tuple: 包含以下元素的元组:
                - atom_type (torch.Tensor): 预测的节点原子类型，形状为 [N, atom_type_dim] 或相应形式。
                - true_edge_pred (torch.Tensor): 真实边的连接评分，形状依赖于 edge_link_predictor 的输出。
                - neg_edge_pred (torch.Tensor): 负采样边的连接评分，如果没有有效负边则为空张量。
                - mu (torch.Tensor): 编码器输出的隐分布均值。
                - logvar (torch.Tensor): 编码器输出的隐分布对数方差。
        """
        x = self.atom_embedding(x.long()).squeeze().float()
        mu, logvar = self.encoder(x, edge_index,edge_attr.float())
        z = self.reparameterize(mu, logvar)
        
        atom_type = self.atom_type_predictor(z)
        #print(atom_type[0])
        # 拼接所有节点的embedding（注意，此处z已经是形状为[N, embed_dim]的矩阵）
        # 计算所有节点两两之间的内积，生成重构的邻接矩阵
        #reconstructed_adj = torch.matmul(z, z.t())

        true_edge_pred = self.edge_link_predictor(torch.cat([z[sampled_edge_index[0]], z[sampled_edge_index[1]]], dim=1))
        
        # 过滤掉含有-1的边
        #print(sampled_neg_edge_index)
        #mask = (sampled_neg_edge_index[0] != -1) & (sampled_neg_edge_index[1] != -1)
        #if mask.sum() > 0:
        #    filtered_neg_edge_index = sampled_neg_edge_index[:, mask].long()
        #    neg_edge_pred = self.edge_link_predictor(torch.cat([z[filtered_neg_edge_index[0]], 
        #                            z[filtered_neg_edge_index[1]]], dim=1))
        #else:
        #    neg_edge_pred = torch.empty(0, 4, device=z.device)
        #print(f"True mask elements: {torch.sum(mask)}")
        #print(f"False mask elements: {torch.sum(~mask)}")

        return atom_type, true_edge_pred, mu, logvar
    
    def generate(self,z):
        #暂时只考虑无向图
        edge_index = [[],[]]
        edge_attr = []
        atom_type = torch.argmax(self.atom_type_predictor(z),dim=-1)
        print(atom_type)
        for i in range(z.shape[0]):
            for j in range(i+1,z.shape[0]):
                feature = torch.cat((z[i],z[j]),dim=0)
                edge_probability = self.edge_link_predictor(feature.unsqueeze(0))
                print(f"edge for {i} and {j}:{edge_probability}\n")
                edge_pred = torch.argmax(edge_probability)
                if edge_pred!=0:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_attr.append(edge_pred)

                    edge_index[0].append(j)
                    edge_index[1].append(i)
                    edge_attr.append(edge_pred)
        
        return torch.tensor(atom_type), torch.tensor(edge_index), torch.tensor(edge_attr)


        
