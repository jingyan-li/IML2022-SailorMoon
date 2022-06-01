"""
Dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import warnings
import numpy as np
from typing import Tuple, List
import random
from pathlib import Path

import pandas as pd



class MolecularData():
    def __init__(self,
                 root_path: str = None,
                 train_val_split: bool = True,
                 load_sets: list = ['pretrain', 'test', 'train'],
                 loader_cfg: dict = None
                 ):

        self.datasets = {'pretrain': None, 'test': None, 'train': None}
        self.dataloaders = {'pretrain': None, 'test': None, 'train': None}
        self.data_root = Path(root_path)
        self.arr_dict = {'pretrain': None, 'test': None, 'train': None}
        self.data_all = {'pretrain': None, 'test': None, 'train': None}

        # load scan scenes as train/test list
        for k in load_sets:
            self.arr_dict[k] = self.getFeatures(self.data_root, k)

        # split pretrain into train & val
        if train_val_split and 'pretrain' in load_sets:
            print('PRETRAINING: PRETRAIN DATA SPLIT INTO TRAIN AND VAL')
            arr_feat, arr_label, arr_id = self.arr_dict['pretrain']['feature'], self.arr_dict['pretrain']['label'], self.arr_dict['pretrain']['id']
            idx = np.arange(arr_feat.shape[0])
            random.shuffle(idx)
            arr_feat, arr_label, arr_id = arr_feat[idx], arr_label[idx], arr_id[idx]
            idx_end_train = int(len(idx) * 0.85)
            self.arr_dict['pretrain']= {"feature": arr_feat[:idx_end_train], "label": arr_label[:idx_end_train], "id":arr_id[:idx_end_train]}
            self.arr_dict['test'] = {"feature": arr_feat[idx_end_train:], "label": arr_label[idx_end_train:], "id": arr_id[idx_end_train:]}

        # TODO: set sampler or shuffle
        samplers = {'pretrain': None, 'train': None, 'test': None}
        shuffles = {'pretrain': True, 'train': False, 'test': False}

        # Build dataset
        for k in self.arr_dict.keys():
            if k not in load_sets:
                continue
            self.datasets[k] = _MolecularDataset(
                data_dict=self.arr_dict[k],
            )
            self.dataloaders[k] = DataLoader(self.datasets[k], shuffle=shuffles[k], sampler=samplers[k], **loader_cfg)
            print('Build {} dataset and loader with length {}'.format(k, len(self.datasets[k])))

        # Read train data as whole
        if "train" in load_sets:
            self.data_all['train'] = {
                "feature": torch.tensor(self.arr_dict['train']["feature"]).float(),
                "label": self.arr_dict['train']["label"],
                "index": self.arr_dict['train']["id"]
            }


    def getFeatures(self, path: Path, split: str = "train"):
        df_feat = pd.read_csv(path / f"{split}_features.csv").sort_values(by="Id").set_index("Id", drop=True)
        # TODO: Reserve SMILES?
        df_feat = df_feat.drop(labels="smiles", axis="columns")
        arr_feat = df_feat.values
        if split != "test":
            df_label = pd.read_csv(path / f"{split}_labels.csv").sort_values(by="Id").set_index("Id", drop=True)
            arr_label = df_label.values
            print(f"{split} - features {arr_feat.shape}, labels {arr_label.shape}")
        else:
            arr_label = None
            print(f"{split} - features {arr_feat.shape}, labels NONE")
        return {"feature": arr_feat, "label": arr_label, "id": np.array(df_feat.index)}


class _MolecularDataset(Dataset):
    def __init__(self, data_dict: dict):
        self.feature = torch.tensor(data_dict["feature"]).float()
        self.label = torch.tensor(data_dict["label"]).float() if data_dict["label"] is not None else None
        self.index = torch.tensor(data_dict["id"])
        print("label type", type(self.label))

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx: int):
        if self.label is not None:
            return {"feature": self.feature[idx], "label": self.label[idx], "index": self.index[idx]}
        else:
            return {"feature": self.feature[idx], "index": self.index[idx]}

