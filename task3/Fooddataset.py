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
        augmentation = {'train': False, 'test': False}

        # Build dataset
        for k in self.list_dict.keys():
            if k not in load_sets:
                continue
            self.datasets[k] = _FoodDataset(
                img_dir= self.data_root["image_path"],
                image_list= self.list_dict[k],
                image_resize= image_resize,
                augmentation= augmentation[k]
            )
            self.dataloaders[k] = DataLoader(self.datasets[k], shuffle=shuffles[k], sampler=samplers[k], **loader_cfg)
            print('Build {} dataset and loader with length {}'.format(k, len(self.datasets[k])))


    def getList(self, path: Path, split: str = "train"):
        return np.loadtxt(path / f"{split}_triplets.txt", dtype=str, delimiter=" ").tolist()



class _FoodDataset(Dataset):
    def __init__(self, img_dir : Path, image_list : list, image_resize : tuple, augmentation: bool):
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
        ]) if not augmentation else tr.Compose([
            tr.Resize(image_resize),
            tr.RandomHorizontalFlip(),
            tr.RandomVerticalFlip(),
            tr.ColorJitter(),
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



class ImageData(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_resize):
        ''' Initialize the dataset
        Args:
            img_dir (str): Images directory
            img_resize ((H, W)): To resize the images with the same size
        '''
        self.path = img_dir

        self.transform = tr.Compose([
            tr.Resize(img_resize),
            tr.ToTensor(),
            tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.file_list = sorted(glob(os.path.join(self.path, '*.jpg')))

        if len(self.file_list) != 0:
            print(
                '\nImage dataset created successfully with {} images.'.format(len(self.file_list)))
        else:
            print('\nError creating dataset. Please ensure the path to the dataset folder is correct.\n')
            raise Exception

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        ''' Read anchor_emb, positive_emb, negative_emb images

        '''
        img_name = self.file_list[idx]
        img_index = self.file_list[idx][-9:-4]  # 00000 - 09999
        img_index = torch.tensor([int(img_index)])

        img: torch.Tensor = self.preprocess_image(img_name)

        sample = {'image': img, 'idx': img_index}

        return sample

    def preprocess_image(self, filename) -> torch.Tensor:
        img: Image = Image.open(filename)
        img: Image = img.convert('RGB')
        # img = torch.from_numpy(img).float()

        img_tensor: torch.Tensor = self.transform(img)
        # print(type(img))
        return img_tensor


class TripletData(torch.utils.data.Dataset):
    def __init__(self, features, triplet_label_arr):
        '''Initialize the dataset
        Args:
            features: feature tensor, look like (10000, num_features)
            feature_dir: file path storing embedding features
            triplet_label_arr: numpy.array [N,3] (each line represents a triplet looking like [0, 9999, 1])
        '''

        self.triplet_arr: np.array = triplet_label_arr

        # self.features: torch.tensor = torch.load(feature_dir)
        self.features: torch.tensor = features

    def __len__(self):
        return len(self.triplet_arr)

    def __getitem__(self, idx):
        triplet_idx_arr = self.triplet_arr[idx]
        anchor_embedding, positive_embedding, negative_embedding = self.features[triplet_idx_arr]

        return (anchor_embedding, positive_embedding, negative_embedding), triplet_idx_arr

