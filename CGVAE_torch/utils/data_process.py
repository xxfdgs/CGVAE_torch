import os
import random
from torch_geometric.datasets import QM9, ZINC
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.datasets import ZINC
from tqdm import tqdm

def create_dataset(root, dataset_name,atom_sample_size=1.0, 
                   edge_sample_size=1.0, 
                   neg_edge_sample_size=0.25):
    """
    根据提供的字符串加载QM9或ZINC数据集。
    
    参数:
        root: 数据集的根目录。
        dataset_name: 数据集名称，暂时可选的值为"QM9"或"ZINC"。
        atom_sample_size:每个分子图采样的参与训练的原子个数，用于训练预测原子种类。
        edge_sample_size:每个分子图采样的参与训练的边个数，用于训练预测化学键。
        neg_edge_sample_size:每个分子图采样的参与训练的负边比例，用于训练预测化学键。
        
    返回:
        数据集对象。
    
    异常:
        ValueError: 如果提供了不支持的数据集名称。
    """


    os.makedirs(root, exist_ok=True)
    root = os.path.join(root, dataset_name)
    
    #不从邻接矩阵直接入手，而是生成正确边与错误边的索引作为额外特征，让模型进行预测
    #我们先考虑所有正向边均采样的情况
    #将18及18以上的原子合为一种
    def sample_edge(data, atom_sample_size=1.0, 
                   edge_sample_size=1.0, 
                   neg_edge_sample_size=0.5):
        #for atom in data.x:
        #    if atom>torch.tensor(18):
        #         atom=torch.tensor(18,dtype=torch.long)
        atom_edge_dict = dict()
        for i in range(len(data.edge_attr)):
            if data.edge_index[0][i].item() not in atom_edge_dict.keys():
                atom_edge_dict[data.edge_index[0][i].item()] = [data.edge_index[1][i].item()]
            else:
                atom_edge_dict[data.edge_index[0][i].item()].append(data.edge_index[1][i].item())
        #data.atom_edge_dict = atom_edge_dict

        num_nodes = data.x.size()[0]
        k_atoms = max(int(num_nodes * atom_sample_size),1)
        weights = torch.ones(num_nodes)
        #weights[data.x.squeeze() == 0] = 0.1
        sample_indices = torch.multinomial(weights, k_atoms, replacement=False)
        data.sample_atom_indices = sample_indices
        #print(data.x[sample_indices])
        #data.sample_atom = sample_indices

        #print(f"edge_sample_size:{edge_sample_size}")
        #print(f"neg_edge_sample_size:{neg_edge_sample_size}")

        sampled_neg_edge_index = [[],[]]
        neg_edge_len = 0
        num_edges = len(data.edge_index[0])
        try_times=0
        while neg_edge_len < int(neg_edge_sample_size*num_edges):
            i = random.randint(0, data.x.shape[0] - 1)
            j = random.randint(0, data.x.shape[0] - 1)
            try:
                if i != j and j not in atom_edge_dict[i] and i not in atom_edge_dict[j]:

                    sampled_neg_edge_index[0].append(i)
                    sampled_neg_edge_index[1].append(j)
                    sampled_neg_edge_index[0].append(j)
                    sampled_neg_edge_index[1].append(i)
                    neg_edge_len += 2
                try_times += 1
                if try_times > 10000:
                    break
            except KeyError:
                print(atom_edge_dict)
                print(i,j)
        #print(f"neg_edge_len:{neg_edge_len}")
        sampled_neg_edge_index = torch.tensor(sampled_neg_edge_index, dtype=torch.long)
        sampled_neg_edge_attr = torch.zeros(sampled_neg_edge_index.size()[1],dtype=torch.long)
        
        data.edge_index = torch.cat([data.edge_index,sampled_neg_edge_index],dim=1)
        data.edge_attr = torch.cat([data.edge_attr,sampled_neg_edge_attr],dim=0)
        
        num_edges = len(data.edge_index[0])
        k = int(num_edges * edge_sample_size)
        indices = list(range(num_edges))
        sampled_indices = random.sample(indices, k)
        data.sampled_edge_index = data.edge_index[:, sampled_indices]
        data.sampled_edge_attr = data.edge_attr[sampled_indices]
        return data

        
    if dataset_name == "QM9":
        dataset = QM9(root=root, transform=lambda data: sample_edge(data,atom_sample_size, 
                                                   edge_sample_size, 
                                                   neg_edge_sample_size))
        total_len = len(dataset)
        train_len = int(total_len * 0.8)
        val_len = int(total_len * 0.1)
        test_len = total_len - train_len - val_len
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
        
        return train_dataset, val_dataset, test_dataset
        

    elif dataset_name == "ZINC":
        
        train_dataset = ZINC(root=root, split="train", transform=lambda data: sample_edge(data,atom_sample_size, 
                                                   edge_sample_size, 
                                                   neg_edge_sample_size), subset=True)
        val_dataset = ZINC(root=root, split="val", transform=lambda data: sample_edge(data,atom_sample_size, 
                                                   edge_sample_size, 
                                                   neg_edge_sample_size), subset=True)
        test_dataset = ZINC(root=root, split="test", transform=lambda data: sample_edge(data,atom_sample_size, 
                                                   edge_sample_size, 
                                                   neg_edge_sample_size), subset=True)
        """
        
        train_dataset = ZINC(root=root, split="train",
                            transform=lambda data: sample_edge(data,atom_sample_size, 
                                                   edge_sample_size, 
                                                   neg_edge_sample_size))
        val_dataset = ZINC(root=root, split="val", transform=lambda data: sample_edge(data,atom_sample_size, 
                                                   edge_sample_size, 
                                                   neg_edge_sample_size))
        test_dataset = ZINC(root=root, split="test", transform=lambda data: sample_edge(data,atom_sample_size, 
                                                   edge_sample_size, 
                                                   neg_edge_sample_size))
        """
        print(test_dataset[0])
        
        return train_dataset, val_dataset, test_dataset
        
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Please choose 'QM9' or 'ZINC'.")
    
def count_atom(dataset):
    """
    计算数据集中每种原子的数量。
    
    参数:
        dataset: 数据集对象。
        
    返回:
        包含原子数量的字典。
    """
    atom_dict = {}
    for data in tqdm(dataset, desc="Counting atoms"):
        for atom in data.x:
            atom = atom.item()
            if atom not in atom_dict:
                atom_dict[atom] = 1
            else:
                atom_dict[atom] += 1
    return atom_dict
def count_edge(dataset):
    edge_dict = {}
    for data in tqdm(dataset, desc="Counting edges"):
        for edge_attr in data.edge_attr:
            edge_attr = edge_attr.item()
            if edge_attr not in edge_dict:
                edge_dict[edge_attr] = 1
            else:
                edge_dict[edge_attr] += 1
    return edge_dict

# 示例使用代码：
if __name__ == "__main__":
    atom_dict = {}
    edge_dict = {}
    train_dataset = ZINC(root="data/ZINC", split="train")
    val_dataset = ZINC(root="data/ZINC", split="val")
    test_dataset = ZINC(root="data/ZINC", split="test")
    atom_dict["train"] = count_atom(train_dataset)
    atom_dict["val"] = count_atom(val_dataset)
    atom_dict["test"] = count_atom(test_dataset)
    new_dict = {}
    sum = 0
    for key in atom_dict["train"].keys():
        new_dict[key] = atom_dict["train"][key]
        if key in atom_dict["val"].keys():
            new_dict[key] += atom_dict["val"][key]
        if key in atom_dict["test"].keys():
            new_dict[key] += atom_dict["test"][key]
        sum+=new_dict[key]
    atom_dict["full"] = new_dict
    print(sum)
    print(atom_dict)
    edge_dict["train"] = count_edge(train_dataset)
    edge_dict["val"] = count_edge(val_dataset)
    edge_dict["test"] = count_edge(test_dataset)
    new_dict = {}
    sum = 0
    for key in edge_dict["train"].keys():
        new_dict[key] = edge_dict["train"][key] + edge_dict["val"][key] + edge_dict["test"][key]
        sum+=new_dict[key]
    edge_dict["full"] = new_dict
    print(sum)
    print(edge_dict)
    #data = train_dataset[0]
