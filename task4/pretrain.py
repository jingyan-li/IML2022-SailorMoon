"""
Main script for pre=training
"""

import argparse
from datetime import datetime
import torch
from sklearn import svm

from blowtorch2.blowtorch import Run
from blowtorch2.blowtorch.loggers import WandbLogger

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

import torch.nn as nn
import numpy as np

from src.PretrainDataset import MolecularData
from src import get_pretrainmodel


def config():
    a = argparse.ArgumentParser()
    a.add_argument("--config", help="path to train config", default='configs/pretrain_simple.yaml')
    args = a.parse_args()
    return args


def get_embedding(model, dataloader):
    pred_embedding = []
    pred_y_target = []
    for batch_idx, sample_dict in enumerate(dataloader):
        feature, target, indices = sample_dict["feature"], sample_dict["label"], sample_dict["index"]
        embedding = model.get_embedding(feature.to(device))
        pred_embedding.append(embedding)
        pred_y_target.append(target)
    pred_embedding = torch.cat(pred_embedding, 0).cpu().numpy()
    pred_y_target = torch.cat(pred_y_target, 0).numpy()
    return pred_embedding, pred_y_target


def get_embedding_all(model, data_dict):
    feature, target, indices = data_dict["feature"], data_dict["label"], data_dict["index"]
    embedding = model.get_embedding(feature.to(device))
    return embedding.cpu(), target


def get_train_model(model_type):
    if model_type == "svm":
        cv_param = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1.14, 1.13, 1.135], 'epsilon': [0.016, 0.018, 0.017]}
        model = svm.SVR()
    elif model_type == "rf":
        cv_param = {'n_estimators': (30, 50, 100), 'max_depth':(3,5,7,9)}
        model = RandomForestRegressor()
    clf = GridSearchCV(model, cv_param, error_score='raise', scoring='neg_root_mean_squared_error')
    return clf


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
    pretrain_model = get_pretrainmodel(run)

    # Train model
    train_model = get_train_model(run['train_model_type'])

    # Loss
    LossF = nn.L1Loss()

    @run.train_step(pretrain_data.dataloaders['pretrain'])
    def train_step(batch, model):
        feature, target = batch["feature"], batch["label"]
        output = model(feature)
        loss = LossF(output, target)
        # # ADD Train data fitting to loss
        # pred_embedding, pred_y_target = get_embedding(model, train_data.dataloaders["train"])
        # scores = -1 * cross_val_score(train_model, pred_embedding, pred_y_target.ravel(),
        #                              cv=5,
        #                              scoring='neg_root_mean_squared_error')
        # # Average over cv results
        # scores_avg = np.mean(scores)
        # loss = loss + torch.tensor(scores_avg/10.).to(device)
        return loss


    @run.validate_step(pretrain_data.dataloaders['test'], every=5, at=0)
    def val_step(batch, model, epoch, batch_index):
        feature, target = batch["feature"], batch["label"]
        output = model(feature)
        loss = LossF(output, target)
        # TODO: Add training of TRAIN DATASET
        # if batch_index == 0:
        pred_embedding, pred_y_target = get_embedding_all(model, train_data.data_all["train"])
        train_model.fit(pred_embedding, pred_y_target.ravel())
        scores_avg = - train_model.best_score_
            # print("Best fitted Train model: ", train_model.best_params_)
            # print(f"VALIDATION epoch{epoch} CV-RMSE {scores_avg}")
        return {"loss": loss, "train_rmse": scores_avg}, None


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
