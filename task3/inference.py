"""
Main script for training
"""

import argparse

import torch
import os
import numpy as np
import wandb
import yaml
from tqdm import tqdm

from Fooddataset import FoodData
from SiameseNetwork import EmbeddingNet, EmbeddingNetL2, TripletModel, EmbeddingNetPretrain


def config():
    a = argparse.ArgumentParser()
    a.add_argument("--train_config", help="path to train config", default='configs/train.yaml')
    a.add_argument("--pred_config", help="path to inference config", default="configs/inference.yaml")
    a.add_argument("--output", default="predict.csv", help="path and filename of output file")
    args = a.parse_args()
    return args


if __name__ == "__main__":
    args = config()

    with open(args.train_config) as fh:
        train_config = yaml.safe_load(fh)

    with open(args.pred_config) as fh:
        pred_config = yaml.safe_load(fh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Initialize model
    img_size = train_config['data_cfg']['image_resize']

    USE_PRETRAIN = False if train_config['model_cfg']['embedding_model']['pretrain_model'] == "" else True
    FEATURE_EXTRACT = train_config['model_cfg']['embedding_model']['feature_extract']

    embedding_model = EmbeddingNet(img_size, train_config
        ['model_cfg']['embedding_model']['out_channels']) if not USE_PRETRAIN else EmbeddingNetPretrain(
        **train_config['model_cfg']['embedding_model'])
    triplet_model = TripletModel(embedding_model, **train_config['model_cfg']['triplet_model'])

    # To gpu
    embedding_model = embedding_model.to(device)
    triplet_model = triplet_model.to(device)

    checkpoint_path = os.path.join(pred_config['inference_cfg']['resume_checkpoint'], 'checkpoints/best')
    triplet_model.load_state_dict(torch.load(checkpoint_path)['model'])
    triplet_model.eval()

    # create dataset
    data = FoodData(load_sets=['test'], **pred_config['data_cfg'])
    dataloader = data.dataloaders['test']

    # run model
    results = []
    for step, batch in enumerate(tqdm(dataloader)):
        # if step > 2:
        #     break
        anchor_img, positive_img, negative_img = (_.to(device) for _ in batch[:-1])
        # anchor = batch[-1]
        B, C, W, H = anchor_img.shape
        # print(f"Batch {step}, anchor {anchor}")
        pred_y, loss = triplet_model.predict(anchor_img, positive_img, negative_img)
        results.append(pred_y.to('cpu').numpy().tolist())
        # print(f"Batch {step} - Loss {loss} - pred_y {results[-1]}")

    print("writing...")
    results = [item for sublist in results for item in sublist]
    # print(results)
    # Save results
    with open(args.output, "w") as file:
        file.writelines("\n".join([str(int(_)) for _ in results]))
    print("Saved!")
