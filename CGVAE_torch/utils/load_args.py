import argparse
import json
import os
import datetime
import torch

def parse_args():
    """
    Parse command-line arguments and return the processed arguments.
    """
    parser = argparse.ArgumentParser(description="Command line argument parser for CGVAE.")

    # Example arguments, add more as necessary.
    parser.add_argument("--config", type=str,help="Config to use.")
    parser.add_argument("--epochs", type=int,  help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int,  help="Batch size for training.")
    parser.add_argument("--init_lr", type=float, help="Learning rate for training.")
    parser.add_argument("--model", type=str,  help="Model to train.")
    parser.add_argument("--outdir",type=str,help="Output directory for saving model and logs.")
    parser.add_argument("--device", type=str, default="gpu",help="Device to use for training.")

    args = parser.parse_args()
    return args

def load_params():
    """
    从命令行参数指定的config文件中读取参数,根据命令行参数更新后返回
    """
    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    
    if args.outdir:
        config["outdir"] = args.outdir
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["params"]["batch_size"] = args.batch_size
    if args.init_lr:
        config["params"]["init_lr"] = args.init_lr
    if args.model:
        config["params"]["model"] = args.model  
    
    if args.device != "cpu":
        try:
            gpu_id = config["gpu_id"]
            if torch.cuda.is_available() and int(gpu_id) < torch.cuda.device_count():
                #torch.cuda.set_device(gpu_id)
                device = f"cuda:{gpu_id}"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"
        config["device"] = device
    else:
        config["device"] = "cpu"

    os.makedirs(config["outdir"],exist_ok=True)
    os.makedirs(os.path.join(config["outdir"],"checkpoints"),exist_ok=True)
    os.makedirs(os.path.join(config["outdir"],"logs"),exist_ok=True)
    os.makedirs(os.path.join(config["outdir"],"results"),exist_ok=True)
    os.makedirs(os.path.join(config["outdir"],"configs"),exist_ok=True)

    now = datetime.datetime.now()
    time_str = now.strftime("%Hh%Mm%Ss")
    date_str = now.strftime("on_%b_%d_%Y")

    model_str = config["params"]["model"]
    dataset_str = config["data"]["dataset_name"]

    outname = f"result_{model_str}_{dataset_str}_{device}_{time_str}_{date_str}"
    config["outname"] = outname

    config_save_path = os.path.join(config["outdir"], "configs", f"{config['outname']}.json")
    print(config)
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=4)

    return config

