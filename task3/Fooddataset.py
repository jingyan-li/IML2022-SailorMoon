"""
Dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from torchvision import transforms as tr
import warnings
import numpy as np
from PIL import Image
from typing import Tuple, List
import random
from pathlib import Path


class FoodData():
    def __init__(self,
                 root_path: str = None,
                 train_val_split: bool = True,
                 load_sets: list = ['train', 'test'],
                 image_resize: tuple = (240,240),
                 loader_cfg: dict = None
                 ):

        self.datasets = {'train': None, 'test': None}
        self.dataloaders = {'train': None, 'test': None}
        self.data_root = {'triplet_path': Path(root_path), 'image_path': Path(root_path)/"food"}
        self.list_dict = {'train': None, 'test': None}

        # load scan scenes as train/test list
        for k in load_sets:
            self.list_dict[k] = self.getList(self.data_root["triplet_path"], k)

        # split train into train & val
        if train_val_split and 'train' in load_sets:
            print('TRAIN & VAL MODE: TRAIN DATA SPLIT INTO TRAIN AND VAL')
            random.shuffle(self.list_dict['train'])
            idx_end_train = int(len(self.list_dict['train']) * 0.8)
            self.list_dict['train'], self.list_dict['test'] = self.list_dict['train'][:idx_end_train], self.list_dict[
                                                                                                           'train'][
                                                                                                       idx_end_train:]

        # TODO: set sampler or shuffle
        samplers = {'train': None, 'test': None}
        shuffles = {'train': True, 'test': False}

        # Build dataset
        for k in self.list_dict.keys():
            if k not in load_sets:
                continue
            self.datasets[k] = _FoodDataset(
                img_dir= self.data_root["image_path"],
                image_list= self.list_dict[k],
                image_resize= image_resize
            )
            self.dataloaders[k] = DataLoader(self.datasets[k], shuffle=shuffles[k], sampler=samplers[k], **loader_cfg)
            print('Build {} dataset and loader with length {}'.format(k, len(self.datasets[k])))


    def getList(self, path: Path, split: str = "train"):
        return np.loadtxt(path / f"{split}_triplets.txt", dtype=str, delimiter=" ").tolist()



class _FoodDataset(Dataset):
    def __init__(self, img_dir : Path, image_list : list, image_resize : tuple):
        ''' Initialize the dataset
        Args:
            img_dir (str): Images directory
            image_list (list): [n_records, 3 (anchor img_id, positive img_id, negative img_id)]
            image_resize ((H, W)): To resize the images with the same size
        '''
        self.img_dir = img_dir
        self.image_list = image_list
        # TODO Check transform
        print(f"_FoodDataset image resize.shape {image_resize}")
        self.transform = tr.Compose([
            tr.Resize(image_resize),
            tr.ToTensor(),
            tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List]:
        ''' Read anchor_emb, positive_emb, negative_emb images
        '''
        anchor, positive, negative = self.image_list[idx]

        anchor_img: torch.Tensor = self.preprocess_image(anchor)
        positive_img: torch.Tensor = self.preprocess_image(positive)
        negative_img: torch.Tensor = self.preprocess_image(negative)

        return anchor_img, positive_img, negative_img, anchor

    def preprocess_image(self, filename) -> torch.Tensor:
        img: Image = Image.open(self.img_dir / f'{filename}.jpg')
        img: Image = img.convert('RGB')

        img_tensor: torch.Tensor = self.transform(img)
        return img_tensor
