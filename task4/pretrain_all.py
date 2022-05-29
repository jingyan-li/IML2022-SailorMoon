import argparse
import gc
import getpass
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models


def setup_wandb(usr_home_dir: str):
    username = getpass.getuser()
    print(username)
    wandb_key_dir = os.path.join(usr_home_dir, "configs")
    if not os.path.exists(wandb_key_dir):
        os.makedirs(wandb_key_dir)
    wandb_key_filename = os.path.join(wandb_key_dir, username + "_wandb.key")
    if not os.path.exists(wandb_key_filename):
        wandb_key = input(
            "[You need to firstly setup and login wandb] Please enter your wandb key (https://wandb.ai/authorize):")
        with open(wandb_key_filename, "w") as fh:
            fh.write(wandb_key)
    else:
        print("wandb key already set")
    os.system("export WANDB_API_KEY=$(cat \"" + wandb_key_filename + "\")")


def main(opt: argparse.Namespace) -> None:
    # Lock seed
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.benchmark = True

    """ ********************* Config ********************* """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # beginning timestamp
    run_name = opt.name + "_" + ts  # modified to a name that is easier to index
    root_path = Path(opt.data_root)
    output_root = Path(opt.output_root)  # .../out/
    output_path = Path / run_name  # .../out/run_name/

    pretrain_features = pd.read_csv(root_path/"pretrain_features.csv")
    pretrain_labels = pd.read_csv(root_path/"pretrain_labels.csv")

    # check directories
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    assert os.access(root_path, os.R_OK)
    assert os.access(output_root, os.W_OK)
    assert os.access(output_path, os.W_OK)
    print("Directory Permission checked.")

    # setup wandb
    setup_wandb(usr_home_dir=opt.user_home)
    wandb.init(project="IMLTask3 - v2", dir=output_path)
    wandb.run.name = run_name
    wandb.config.update(opt)

    # specify GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device: ", device)

