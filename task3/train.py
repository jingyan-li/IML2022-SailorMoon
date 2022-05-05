"""
Main script for training
"""

import argparse

import torch
import os
import sys
import numpy as np
import wandb

from blowtorch2.blowtorch import Run
from blowtorch2.blowtorch.loggers import WandbLogger

from Fooddataset import FoodData
from SiameseNetwork import EmbeddingNet, EmbeddingNetL2, TripletModel
from TripletLoss import triplet_loss

def config():
    a = argparse.ArgumentParser()
    a.add_argument("--config", help="path to train config", default='configs/train.yaml')
    args = a.parse_args()
    return args

if __name__ == "__main__":

    args = config()
    run = Run(config_files=[args.config])
    run.set_deterministic(run['train_cfg.deterministic'])
    run.seed_all(run['train_cfg.random_seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loggers = WandbLogger(project='IML-Food') if run['log_cfg.use_wandb_logger'] else None

    # Create dataset
    data = FoodData(**run['data_cfg'])
    # print("Test dataloader")
    # print(type(data.dataloaders['train']))
    # print(data.datasets["test"].__getitem__(0))

    # Initialize model
    img_size = run['data_cfg.image_resize']

    embedding_model = EmbeddingNet(img_size, **run['model_cfg.embedding_model'])
    triplet_model = TripletModel(embedding_model, **run['model_cfg.triplet_model'])

    @run.train_step(data.dataloaders['train'])
    def train_step(batch, model):
        anchor_img, positive_img, negative_img = batch
        # print(f"anchor {anchor_img.shape}, positive {positive_img.shape}, negative {negative_img.shape}")
        anchor_emb, positive_emb, negative_emb = model(anchor_img, positive_img, negative_img)
        loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
        return {"loss": loss}


    @run.validate_step(data.dataloaders['test'], every=2, at=0)
    def val_step(batch, model, epoch, batch_index):
        anchor_img, positive_img, negative_img = batch
        pred_y, loss = model.predict(anchor_img, positive_img, negative_img)
        print(f"Validation: pred_y.shape {pred_y.shape}")
        return {'loss': loss}

    # TODO: define optimizer
    @run.configure_optimizers
    def configure_optimizers(model):
        optim = torch.optim.Adam(model.parameters(), lr=run['train_cfg.lr'],
                                 weight_decay=run['train_cfg.weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, **run['train_cfg.scheduler'])
        return (optim, scheduler)

    run(
        triplet_model,
        loggers=loggers,
        optimize_first=False,
        resume_checkpoint=run['train_cfg.resume_checkpoint'],
        num_epochs=run['train_cfg.epochs']
    )