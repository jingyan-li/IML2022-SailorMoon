"""
Main script for pre=training
"""

import argparse
from datetime import datetime
import torch
from blowtorch2.blowtorch import Run
from blowtorch2.blowtorch.loggers import WandbLogger

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

import torch.nn as nn
import numpy as np

from src.PretrainDataset import MolecularData
from src.PretrainModel import PretrainEmbeddingNet


def config():
    a = argparse.ArgumentParser()
    a.add_argument("--config", help="path to train config", default='configs/pretrain.yaml')
    args = a.parse_args()
    return args


if __name__ == "__main__":

    args = config()
    run = Run(config_files=[args.config])
    run.set_deterministic(run['train_cfg.deterministic'])
    run.seed_all(run['train_cfg.random_seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loggers = WandbLogger(project='IML-TASK4', name=f"{run['name']}-{datetime.now().strftime('%d/%m/%H:%M:%S')}") if \
    run[
        'log_cfg.use_wandb_logger'] else None

    # Create dataset
    pretrain_data = MolecularData(load_sets=['pretrain', 'test'], train_val_split=True, **run['data_cfg'])
    train_data = MolecularData(load_sets=['train'], train_val_split=False, **run["data_cfg"])

    # Initialize pretrain model
    pretrain_model = PretrainEmbeddingNet(**run['pretrain_model_cfg'])

    # Train model
    train_model = RandomForestRegressor(n_estimators=100, random_state=run['train_cfg.random_seed'])

    # Loss
    LossF = nn.L1Loss()

    @run.train_step(pretrain_data.dataloaders['pretrain'])
    def train_step(batch, model):
        feature, target = batch["feature"], batch["label"]
        output = model(feature)
        loss = LossF(output, target)
        return loss


    @run.validate_step(pretrain_data.dataloaders['test'], every=5, at=0)
    def val_step(batch, model, epoch, batch_index):
        feature, target = batch["feature"], batch["label"]
        output = model(feature)
        loss = LossF(output, target)
        # TODO: Add training of TRAIN DATASET
        if batch_index == 0:
            pred_embedding = []
            pred_y_target = []
            for batch_idx, sample_dict in enumerate(train_data.dataloaders["train"]):
                feature, target, indices = sample_dict["feature"], sample_dict["label"], sample_dict["index"]
                embedding = model.get_embedding(feature.to(device))
                pred_embedding.append(embedding)
                pred_y_target.append(target)
            pred_embedding = torch.cat(pred_embedding, 0).cpu().numpy()
            pred_y_target = torch.cat(pred_y_target, 0).numpy()
            # Multiply by -1 since sklearn calculates *negative* RMSE
            scores = -1 * cross_val_score(train_model, pred_embedding, pred_y_target.ravel(),
                                          cv=5,
                                          scoring='neg_root_mean_squared_error')
            # Average over cv results
            scores_avg = np.mean(scores)
            print(f"VALIDATION epoch{epoch} CV MSE {scores_avg}; scores {scores}")
        return {"loss": loss}, None


    @run.configure_optimizers
    def configure_optimizers(model):
        optim = torch.optim.AdamW(model.parameters(), lr=run['train_cfg.lr'],
                                  weight_decay=run['train_cfg.weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, **run['train_cfg.scheduler'])
        return (optim, scheduler)


    run(
        pretrain_model,
        loggers=loggers,
        optimize_first=False,
        resume_checkpoint=run['train_cfg.resume_checkpoint'],
        num_epochs=run['train_cfg.epochs']
    )
