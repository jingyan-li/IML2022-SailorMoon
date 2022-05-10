"""
Main script for training
"""

import argparse
from datetime import datetime
import torch
from blowtorch2.blowtorch import Run
from blowtorch2.blowtorch.loggers import WandbLogger

from Fooddataset import FoodData
from SiameseNetwork import getEmbeddingModel, TripletModel


def config():
    a = argparse.ArgumentParser()
    a.add_argument("--config", help="path to train config", default='configs/train_pretrained.yaml')
    args = a.parse_args()
    return args


if __name__ == "__main__":

    args = config()
    run = Run(config_files=[args.config])
    run.set_deterministic(run['train_cfg.deterministic'])
    run.seed_all(run['train_cfg.random_seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loggers = WandbLogger(project='IML-Food', name=f"{run['name']}-{datetime.now().strftime('%d/%m/%H:%M:%S')}") if run['log_cfg.use_wandb_logger'] else None

    # Create dataset
    data = FoodData(**run['data_cfg'])

    # Initialize model
    embedding_model, USE_PRETRAIN, FEATURE_EXTRACT = getEmbeddingModel(run, device)
    triplet_model = TripletModel(embedding_model, **run['model_cfg.triplet_model'])

    @run.train_step(data.dataloaders['train'])
    def train_step(batch, model):
        anchor_img, positive_img, negative_img, _ = batch
        # print(f"anchor {anchor_img.shape}, positive {positive_img.shape}, negative {negative_img.shape}")
        anchor_emb, positive_emb, negative_emb = model(anchor_img, positive_img, negative_img)
        loss = model.loss(anchor_emb, positive_emb, negative_emb)
        B, C, W, H = anchor_img.shape
        pred_y, pred_loss = model.predict(anchor_img, positive_img, negative_img)
        acc = torch.sum(pred_y) / B
        return {"loss": loss, "acc": acc}


    @run.validate_step(data.dataloaders['test'], every=5, at=0)
    def val_step(batch, model):
        anchor_img, positive_img, negative_img, _ = batch
        B, C, W, H = anchor_img.shape
        pred_y, loss = model.predict(anchor_img, positive_img, negative_img)
        # print(f"Validation: pred_y.shape {pred_y.size()}")
        acc = torch.sum(pred_y) / B
        loss_batch_mean = torch.mean(loss)
        return {'loss': loss_batch_mean, 'acc': acc}, None

    # TODO: define optimizer
    @run.configure_optimizers
    def configure_optimizers(model):
        print(model)
        params_to_update = model.parameters()
        print("Params to learn:")
        if USE_PRETRAIN and FEATURE_EXTRACT:
            print("Use pretrained model as feature extractor, optimize last layer")
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
                else:
                    print(f"Freezing {name}")
        else:
            if USE_PRETRAIN:
                print("Use pretrained model and fine tuning all layers!")
            else:
                print("Train self-model from scratch. Not pretrained model!")
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)
        optim = torch.optim.Adam(params_to_update, lr=run['train_cfg.lr'],
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