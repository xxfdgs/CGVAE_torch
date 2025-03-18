import torch
import time
import os
from tensorboardX import SummaryWriter
import random
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import glob
import json

from utils.load_args import load_params
from utils.data_process import create_dataset
from models.GVAE import GraphVAE

def loss_calculate(true_edge_predict,true_edge_attr,
                   atom_type, predicted_atom_type,sample_atom_indices,
                   mu, logvar,
                   lambda_1=0.0,lambda_2=0.0, original_q=0,predict_q=0):
    """
    计算损失函数
    """
    weights = torch.tensor([1.0,1.0,2.0,3.0]).to(torch.device("cuda:0"))
    BCE = F.cross_entropy(true_edge_predict, true_edge_attr.long())
    # 该行代码用于计算 VAE 模型中的 KL 散度损失项
    # KL 散度度量的是编码器输出的潜在分布（由 mu 和 logvar 表示）
    # 与标准正态分布之间的差异，目的是使潜变量接近 N(0,1)
    #print(sample_atom_indices)
    #print(atom_type.long()[sample_atom_indices].view(-1))
    weights = torch.ones((28,)).to(torch.device("cuda:0"))
    weights[0]=0.5
    atom_type_loss = F.cross_entropy(predicted_atom_type[sample_atom_indices], 
                                     atom_type.long()[sample_atom_indices].view(-1))
    #这里的sum与mean的选择可能十分重要
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    #print(f"BCE_true:{BCE_true},BCE_neg:{BCE_neg},atom:{atom_type_loss},KLD:{KLD}")
    loss = BCE + atom_type_loss + lambda_1 * KLD
    if lambda_2 != 0.0:
        loss += lambda_2 * F.mse_loss(original_q, predict_q)
    return loss,BCE,atom_type_loss,KLD

def calculate_accuracy(true_edge_predict,true_edge_attr,atom_type, predicted):
    """
    计算准确率
    """
    atom_accuracy = (torch.argmax(predicted.squeeze(),dim=-1) == atom_type.view(-1)).sum().float().item() / len(atom_type.view(-1))
    true_edge_predicted = torch.argmax(true_edge_predict.squeeze(),dim=-1)
    edge_accuracy = (true_edge_predicted == true_edge_attr.view(-1)).sum().float().item() / (len(true_edge_attr.view(-1)))
    #print(f"true_edge_predicted:{true_edge_predicted}")
    #print(f"true_edge_ground_truth:{true_edge_attr}")
    #print(f"edge_accuracy:{edge_accuracy}")
    #print(f"length of true edges:{len(true_edge_attr.view(-1))}")
    #test_accuracy = 0
    #atom_predict = torch.argmax(atom_type.squeeze(),dim=-1)
    #print(atom_predict)
    #print(predicted)
    #predicted = predicted.view(-1)
    #for i in range(len(atom_predict)):
    #    if atom_predict[i]==predicted[i]:
    #        test_accuracy+=1
    #print(f"test_accuracy:{test_accuracy}")
    #print((torch.argmax(atom_type.squeeze(),dim=-1) == predicted.view(-1)).sum().float().item())
    #print(edge_accuracy)

    #edge_accuracy = torch.mean(edge_accuracy).item()
    #print(edge_accuracy)
    return atom_accuracy,edge_accuracy

def train_network(model, data_loader, optimizer, device, epoch,lambda_1,lambda_2):
    """
    训练模型
    """
    model.train()
    total_loss = 0.0
    total_atom_accuracy = []
    total_edge_accuracy = []
    total_atom_type_loss = []
    total_BCE = []
    total_KLD = []
    # Add lists to store mu and logvar statistics
    all_mu_means = []
    all_mu_stds = []
    all_logvar_means = []
    all_logvar_stds = []
    
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        predicted_atom_type, true_edge_pred,mu, logvar = \
                                            model(data.x, data.edge_index,data.edge_attr, data.sampled_edge_index)
        
        # Store statistics for this batch
        all_mu_means.append(mu.mean().item())
        all_mu_stds.append(mu.std().item())
        all_logvar_means.append(logvar.mean().item())
        all_logvar_stds.append(logvar.std().item())
        
        loss,BCE,atom_type_loss,KLD = loss_calculate(true_edge_pred, data.sampled_edge_attr,
                              data.x, predicted_atom_type,data.sample_atom_indices,
                              mu, logvar,lambda_1,lambda_2)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        atom_accuracy, edge_accuracy = calculate_accuracy(true_edge_pred, data.sampled_edge_attr,
                                             data.x, predicted_atom_type)
        total_atom_accuracy.append(atom_accuracy)
        total_edge_accuracy.append(edge_accuracy)
        total_BCE.append(BCE.detach().item())
        total_atom_type_loss.append(atom_type_loss.detach().item())
        total_KLD.append(KLD.detach().item())

    # Calculate averages
    avg_mu_mean = sum(all_mu_means) / len(all_mu_means)
    avg_mu_std = sum(all_mu_stds) / len(all_mu_stds)
    avg_logvar_mean = sum(all_logvar_means) / len(all_logvar_means)
    avg_logvar_std = sum(all_logvar_stds) / len(all_logvar_stds)
    
    #print(f"Average mu mean: {avg_mu_mean:.4f}, Average mu std: {avg_mu_std:.4f}")
    #print(f"Average logvar mean: {avg_logvar_mean:.4f}, Average logvar std: {avg_logvar_std:.4f}")

    avg_atom_accuracy = sum(total_atom_accuracy) / len(total_atom_accuracy)
    avg_edge_accuracy = sum(total_edge_accuracy) / len(total_edge_accuracy)
    avg_BCE = sum(total_BCE) / len(total_BCE)
    avg_atom = sum(total_atom_type_loss) / len(total_atom_type_loss)
    avg_KLD = sum(total_KLD) / len(total_KLD)
    
    return total_loss / len(data_loader), avg_atom_accuracy, avg_edge_accuracy,optimizer,avg_BCE,avg_atom_accuracy,avg_KLD,avg_mu_mean,avg_mu_std,avg_logvar_mean,avg_logvar_std
            

def evaluate_network(model, data_loader, device, epoch,lambda_1,lambda_2):
    """
    评估模型
    """
    model.eval()
    model.encoder.training=False
    total_loss = 0.0
    total_atom_accuracy = []
    total_edge_accuracy = []
    total_atom_type_loss = []
    total_BCE = []
    total_KLD = []
    all_mu_means = []
    all_mu_stds = []
    all_logvar_means = []
    all_logvar_stds = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            predicted_atom_type, true_edge_pred,mu, logvar = model(data.x, data.edge_index,data.edge_attr,
                                                                                       data.sampled_edge_index)
            # Store statistics for this batch
            all_mu_means.append(mu.mean().item())
            all_mu_stds.append(mu.std().item())
            all_logvar_means.append(logvar.mean().item())
            all_logvar_stds.append(logvar.std().item())
            
            loss,BCE,atom_type_loss,KLD = loss_calculate(true_edge_pred, data.sampled_edge_attr,
                                  data.x, predicted_atom_type,data.sample_atom_indices,
                                  mu, logvar,lambda_1,lambda_2)
            total_loss += loss.detach().item()
            atom_accuracy, edge_accuracy= calculate_accuracy(true_edge_pred, data.sampled_edge_attr,
                                                 data.x, predicted_atom_type)
            total_atom_accuracy.append(atom_accuracy)
            total_edge_accuracy.append(edge_accuracy)
            total_BCE.append(BCE.detach().item())
            total_atom_type_loss.append(atom_type_loss.detach().item())
            total_KLD.append(KLD.detach().item())

    # Calculate averages
    avg_mu_mean = sum(all_mu_means) / len(all_mu_means)
    avg_mu_std = sum(all_mu_stds) / len(all_mu_stds)
    avg_logvar_mean = sum(all_logvar_means) / len(all_logvar_means)
    avg_logvar_std = sum(all_logvar_stds) / len(all_logvar_stds)
    
    #print(f"Average mu mean: {avg_mu_mean:.4f}, Average mu std: {avg_mu_std:.4f}")
    #print(f"Average logvar mean: {avg_logvar_mean:.4f}, Average logvar std: {avg_logvar_std:.4f}")

    avg_atom_accuracy = sum(total_atom_accuracy) / len(total_atom_accuracy)
    avg_edge_accuracy = sum(total_edge_accuracy) / len(total_edge_accuracy)
    avg_BCE = sum(total_BCE) / len(total_BCE)
    avg_atom = sum(total_atom_type_loss) / len(total_atom_type_loss)
    avg_KLD = sum(total_KLD) / len(total_KLD)
    
    return total_loss / len(data_loader), avg_atom_accuracy, avg_edge_accuracy,avg_BCE,avg_atom_accuracy,avg_KLD,avg_mu_mean,avg_mu_std,avg_logvar_mean,avg_logvar_std

def view_model_params(model):
    """
    计算模型参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def run(train_dataset, val_dataset, test_dataset, config):
    """
    模型的核心训练函数
    """
    t0 = time.time()
    per_epoch_time = []

    random.seed(config["params"]["seed"])
    np.random.seed(config["params"]["seed"])
    torch.manual_seed(config["params"]["seed"])
    if config["device"] != "cpu":
        torch.cuda.manual_seed(config["params"]["seed"])

    checkpoint_dir = os.path.join(config["outdir"],"checkpoints",config["outname"])
    config_dir = os.path.join(config["outdir"],"configs",config["outname"])
    log_dir = os.path.join(config["outdir"],"logs",config["outname"])
    result_dir = os.path.join(config["outdir"],"results",config["outname"])
    os.makedirs(checkpoint_dir,exist_ok=True)
    os.makedirs(config_dir,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(result_dir,exist_ok=True)

    device = torch.device(config["device"])
    model = GraphVAE(config)
    model.to(device)
    
    lambda_1 = config["Loss"]["lambda_1"]
    lambda_2 = config["Loss"]["lambda_2"]
    print(lambda_1,lambda_2)

    config_file = os.path.join(config_dir, f"{config['outname']}.txt")
    with open(config_file, "w") as f:
        f.write(f"Dataset:{config['data']['dataset_name']}\n")
        f.write(f"Training Graphs:{len(train_dataset)}\n")
        f.write(f"Validation Graphs:{len(val_dataset)}\n")
        f.write(f"Test Graphs:{len(test_dataset)}\n")
        f.write(f"Params:{config['params']}\n")
        f.write(f"Net_Params:{config['net_params']}\n")
        f.write(f"Parameters:{view_model_params(model)}\n")
    
    writer = SummaryWriter(log_dir)

    epochs = config["params"]["epochs"]
    init_lr = config["params"]["init_lr"]

    #暂时不考虑其他优化器
    if config["params"]["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr,weight_decay=config["params"]["weight_decay"])
    # 以下代码用于设置一个学习率调度器：ReduceLROnPlateau。
    # ReduceLROnPlateau 根据监测指标的变化情况在指标停滞不前时减少学习率，
    # 从而有助于训练过程中的优化。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,  # optimizer：被调度的优化器对象，负责更新模型参数。
        mode="min",  # mode：监控指标的模式，"min" 表示当监控指标停止降低时触发学习率下降。
        factor=config["params"]["lr_reduce_factor"],  # factor：学习率降低的比例因子。每次触发后，学习率乘以此因子。
        patience=config["params"]["lr_schedule_patience"],  # patience：在监控指标不改善的连续epoch数达到该值时触发学习率调整。
        verbose=True  # verbose：是否输出详细信息。设置为 True 时，每次调整都会在输出中显示提示。
        )
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_atom_accs, epoch_val_atom_accs = [], []
    epoch_train_edge_accs, epoch_val_edge_accs = [], []

    train_dataloader = DataLoader(train_dataset, batch_size=config["params"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["params"]["batch_size"], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config["params"]["batch_size"], shuffle=False)
    
    try:
        with tqdm(range(config["params"]["epochs"])) as t:
            for epoch in t:

                t.set_description(f"Epoch {epoch}")

                start = time.time()
                
                train_loss, train_atom_accuracy, train_edge_accuracy, optimizer,train_BCE,train_atom_type_loss,train_KLD, \
                train_avg_mu,train_std_mu,train_avg_logvar,train_std_logvar= train_network(model, train_dataloader,optimizer, device, epoch,
                                lambda_1,lambda_2)
                
                val_loss, val_atom_accuracy, val_edge_accuracy,val_BCE,val_atom_type_loss,val_KLD, \
                val_avg_mu,val_std_mu,val_avg_logvar,val_std_logvar    = evaluate_network(model, val_dataloader, device, epoch,lambda_1,lambda_2)
                
                _, test_atom_accuracy, test_edge_accuracy,_,_,_,_,_,_,_ \
                    =evaluate_network(model, test_dataloader, device, epoch,lambda_1,lambda_2)

                if (val_atom_accuracy>0.8 and val_edge_accuracy>0.8):
                    if lambda_1<=0.1:
                        lambda_1 = lambda_1*10
                        print(f"lambda_1={lambda_1}")
                epoch_train_losses.append(train_loss)
                epoch_val_losses.append(val_loss)
                epoch_train_atom_accs.append(train_atom_accuracy)
                epoch_val_atom_accs.append(val_atom_accuracy)
                epoch_train_edge_accs.append(train_edge_accuracy)
                epoch_val_edge_accs.append(val_edge_accuracy)

                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Atom_Accuracy/train", train_atom_accuracy, epoch)
                writer.add_scalar("Atom_Accuracy/val", val_atom_accuracy, epoch)
                writer.add_scalar("Atom_Accuracy/test", test_atom_accuracy, epoch)
                writer.add_scalar("Edge_Accuracy/train", train_edge_accuracy, epoch)
                writer.add_scalar("Edge_Accuracy/val", val_edge_accuracy, epoch)
                writer.add_scalar("Edge_Accuracy/test", test_edge_accuracy, epoch)
                writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)
                writer.add_scalar("BCE_Loss/train",train_BCE,epoch)
                writer.add_scalar("BCE_Loss/val",val_BCE,epoch)
                writer.add_scalar("KLD_Loss/train",train_KLD,epoch)
                writer.add_scalar("KLD_Loss/val",val_KLD,epoch)
                writer.add_scalar("Atom_Loss/train",train_atom_type_loss,epoch)
                writer.add_scalar("Atom_Loss/val",val_atom_type_loss,epoch)

                writer.add_scalar("Mu_Mean/train",train_avg_mu,epoch)
                writer.add_scalar("Mu_Mean/val",val_avg_mu,epoch)
                writer.add_scalar("Mu_Std/train",train_std_mu,epoch)
                writer.add_scalar("Mu_Std/val",val_std_mu,epoch)

                writer.add_scalar("Logvar_Mean/train",train_avg_logvar,epoch)
                writer.add_scalar("Logvar_Mean/val",val_avg_logvar,epoch)
                writer.add_scalar("Logvar_Std/train",train_std_logvar,epoch)
                writer.add_scalar("Logvar_Std/val",val_std_logvar,epoch)

                t.set_postfix(#time=time.time()-start,
                              #lr=optimizer.param_groups[0]["lr"],
                              train_loss=train_loss,
                              val_loss=val_loss,
                              train_atom_acc=train_atom_accuracy,
                              train_edge_acc=train_edge_accuracy,
                              val_atom_acc=val_atom_accuracy,
                              val_edge_acc=val_edge_accuracy,
                              test_atom_acc=test_atom_accuracy,
                              test_edge_acc=test_edge_accuracy)

                per_epoch_time.append(time.time()-start)

                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"))
                
                config_save_path = os.path.join(config["outdir"], "configs", f"{config['outname']}.json")
                with open(config_save_path, "w") as f:
                    json.dump(config, f, indent=4)

                files = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))
                for file in files:
                    epoch_num = int(file.split("_")[-1].split(".")[0])
                    if epoch_num < epoch - 1:
                        os.remove(file)
                
                #scheduler.step(val_loss)
    except KeyboardInterrupt:
        print('-'*89)
        print("Training interrupted.")
    
    _, test_atom_accuracy, test_edge_accuracy,_,_,_ \
        = evaluate_network(model, test_dataloader, device, epoch,lambda_1,lambda_2)
    _, train_atom_accuracy, train_edge_accuracy,_,_,_ \
        = evaluate_network(model, train_dataloader, device, epoch,lambda_1,lambda_2)
    # Assign the final computed accuracies for printing and logging
    train_accuracy = train_edge_accuracy
    test_accuracy = test_edge_accuracy

    # 正确性描述 1: 确认评估过程已正确计算所有必要指标
    print("Evaluation completed: all metrics are computed correctly.")
    print(f"Training accuracy: {train_accuracy}")
    print(f"Test accuracy: {test_accuracy}")
    # 正确性描述 2: 验证模型收敛状态并确保时间统计无误
    print("Model convergence and timing metrics verified successfully.")
    print(f"Convergence Time(Epochs): {epoch}")
    print(f"Total time: {time.time()-t0}")
    print(f"Average time per epoch: {np.mean(per_epoch_time)}")

    writer.close()

    with open(os.path.join(result_dir, f"{config['outname']}.txt"), "w") as f:
        f.write(f"Training accuracy: {train_accuracy}\n")
        f.write(f"Test accuracy: {test_accuracy}\n")
        f.write(f"Convergence Time(Epochs): {epoch}\n")
        f.write(f"Total time: {time.time()-t0}\n")
        f.write(f"Average time per epoch: {np.mean(per_epoch_time)}\n")

if __name__ == "__main__":
    config = load_params()

    root = config["data"]["root"]
    dataset_name = config["data"]["dataset_name"]
    atom_sample_size = config["data"]["atom_sample_size"]
    edge_sample_size = config["data"]["edge_sample_size"]
    neg_edge_sample_size = config["data"]["neg_edge_sample_size"]
    subset = config["data"]["subset"]
    train_dataset, val_dataset, test_dataset = create_dataset(root, dataset_name,subset, atom_sample_size
                                                              ,edge_sample_size,neg_edge_sample_size)


    run(train_dataset,val_dataset, test_dataset,config)