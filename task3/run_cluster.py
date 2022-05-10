
import argparse
import gc
import getpass
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models

from Fooddataset import ImageData, TripletData
from SiameseNetwork import Embedding1D, TripletModel_v2 as TripletModel


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
    root_path = opt.data_root
    img_dir = os.path.join(root_path, 'food/')
    output_root = opt.output_root  # .../out/
    output_path = os.path.join(output_root, run_name)  # .../out/run_name/

    train_filename = "train_triplets.txt"
    test_filename = "test_triplets.txt"

    img_size = (348, 512)

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

    # Hyper parameters
    train_val_split = opt.train_data_percentage
    batch_size = opt.batch_size
    n_epochs = opt.num_epochs
    margin = opt.triplet_margin
    lr = opt.learning_rate
    log_interval_batch = opt.log_frequency

    """ ********************* Extract Features ********************* """
    features: torch.Tensor
    feature_path = os.path.join(opt.output_root, opt.feature_path)
    if os.path.isfile(feature_path):
        # Load existing features
        features = torch.load(feature_path)
        print(f"Load features from {feature_path}")
    else:
        # Generate features
        img_dataset = ImageData(img_dir, img_size)
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        img_dataloader: DataLoader = DataLoader(img_dataset, batch_size=opt.feature_batch_size, shuffle=False, **kwargs)

        resnet: nn.Module = models.resnet152(pretrained=True, progress=True)
        print("Model Loaded: resnet")

        inception_v3: nn.Module = models.inception_v3(pretrained=True, progress=True)
        print("Model Loaded: inception_v3")

        vgg: nn.Module = models.vgg19_bn(pretrained=True, progress=True)
        print("Model Loaded: vgg")

        for model in [resnet, inception_v3, vgg]:
            for parameters in model.parameters():
                parameters.requires_grad = False

        resnet.eval()
        inception_v3.eval()
        vgg.eval()

        resnet.to(device)
        inception_v3.to(device)
        vgg.to(device)

        pred_y_result = []
        pred_y_index = []
        with torch.no_grad():
            for batch_idx, sample_dict in enumerate(img_dataloader, start=1):
                # Load data
                img: torch.Tensor = sample_dict['image'].to(device)
                idx: torch.Tensor = sample_dict['idx']
                # Forward
                emb_resnet = resnet(img)
                emb_incept = inception_v3(img)
                emb_vgg = vgg(img)
                embedding = torch.cat([emb_resnet, emb_incept, emb_vgg], 1)

                pred_y_result.append(embedding)
                pred_y_index.append(idx)

                wandb.log({'feature/batch_idx': batch_idx})
                if 0 == batch_idx % opt.log_frequency:
                    print(f"Extracting features: {batch_idx}/{len(img_dataloader)}")

                gc.collect()
                del img
        pred_y_result = torch.cat(pred_y_result, 0).cpu()
        pred_y_index = torch.cat(pred_y_index, 0).cpu()
        features = torch.zeros(([pred_y_result.size()[0]] + list(embedding.size()[1:]))).cpu()

        for i in range(len(pred_y_result)):
            features[pred_y_index[i]] = pred_y_result[i]

        torch.save(features, feature_path)
        print(f"Features saved to {feature_path}")

    print("Features size: ", features.size())

    """ ********************* Data ********************* """

    train = np.loadtxt(os.path.join(root_path, train_filename), delimiter=" ", dtype=int)
    val_img = np.random.randint(5000, size=round(5000 * (1 - train_val_split)))

    train_mask = np.sum(np.isin(train, val_img), axis=1) == 0
    val_mask = np.isin(train[:, 0], val_img)

    train_triplet = train[train_mask]
    val_triplet = train[val_mask]

    train_triplet_dataset = TripletData(features=features, triplet_label_arr=train_triplet)
    val_triplet_dataset = TripletData(features=features, triplet_label_arr=val_triplet)

    kwargs = {'num_workers': opt.num_workers, 'pin_memory': True} if use_cuda else {}
    train_triplet_loader: DataLoader = DataLoader(train_triplet_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_triplet_loader: DataLoader = DataLoader(val_triplet_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # Test data
    test = np.loadtxt(os.path.join(root_path, test_filename), delimiter=" ", dtype=int)
    test_triplet_dataset = TripletData(features=features, triplet_label_arr=test)
    test_triplet_loader: DataLoader = DataLoader(test_triplet_dataset, batch_size=opt.test_batch_size, shuffle=False,
                                                 **kwargs)
    print(
        f"Load {len(train_triplet_dataset)} train data, {len(val_triplet_dataset)} validation data, {len(test_triplet_dataset)} test data.")
    del train, val_img, val_mask, train_mask, train_triplet, val_triplet, kwargs, test
    print('Dataloader ready.')

    """ ********************* Model ********************* """
    # Set up the network and training parameters

    embedding_model = Embedding1D(in_features=features.size()[1])
    embedding_model.to(device)

    # load existing model
    if os.path.isfile(opt.model_path):
        embedding_model.load_state_dict(torch.load(opt.model_path))
        print(f'Load model from {opt.model_path}')
    else:
        print(f'No existing model on {opt.model_path}')

    triplet_model: TripletModel = TripletModel(embedding_model, margin=margin, device=device)
    triplet_model.to(device)

    wandb.watch(embedding_model)

    optimizer = AdamW(triplet_model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, opt.scheduler_step, gamma=opt.scheduler_gamma, last_epoch=-1)
    print('Models ready.')

    """ ********************* Fit ********************* """

    epoch_train_loss_ls = []
    epoch_val_loss_ls = []
    epoch_val_acc_ls = []
    best_accuracy = -1.0
    best_epoch = -1
    for epoch in range(1, n_epochs + 1):
        """ Train """
        embedding_model.train()
        triplet_model.train()

        train_loss_ls = []
        for batch_idx, (triplet_data, _) in enumerate(train_triplet_loader, start=1):
            # Load data
            anchor: torch.Tensor = triplet_data[0].to(device)
            positive: torch.Tensor = triplet_data[1].to(device)
            negative: torch.Tensor = triplet_data[2].to(device)

            # Forward
            # triplet_loss: torch.Tensor
            # pred_y: int
            triplet_loss = triplet_model(anchor, positive, negative)

            train_loss_ls.append(triplet_loss.item())

            # Backward
            optimizer.zero_grad()
            triplet_loss.backward()
            optimizer.step()

            # log
            wandb.log(
                {'train_batch/epoch': epoch, 'train_batch/batch': batch_idx, 'train_batch/train_loss': triplet_loss})
            if 0 == batch_idx % log_interval_batch:
                print(f"Train: {batch_idx:04d}/{len(train_triplet_loader)} \tLoss: {triplet_loss: .6f}")

                gc.collect()

        """ Validate """
        embedding_model.eval()
        triplet_model.eval()
        with torch.no_grad():
            val_loss_ls = []
            val_y_ls = []
            for batch_idx, (triplet_data, _) in enumerate(val_triplet_loader, start=1):
                # Load Data
                anchor: torch.Tensor = triplet_data[0].to(device)
                positive: torch.Tensor = triplet_data[1].to(device)
                negative: torch.Tensor = triplet_data[2].to(device)
                # Forward

                pred_y: torch.Tensor
                pred_y, triplet_loss = triplet_model.predict_with_loss(anchor, positive, negative)
                val_loss_ls.append(triplet_loss.item())
                val_y_ls.append(pred_y.int())

                # Log
                wandb.log(
                    {'val_batch/epoch': epoch, 'val_batch/batch': batch_idx, 'val_batch/val_loss': triplet_loss})
                if 0 == batch_idx % log_interval_batch:
                    print(f"Validation: {batch_idx:04d}/{len(val_triplet_loader)} \tLoss: {triplet_loss: .6f}")

                    gc.collect()

        # log loss and accuracy
        train_loss = np.mean(train_loss_ls)
        val_loss = np.mean(val_loss_ls)
        val_y = torch.cat(val_y_ls, 0)
        val_accuracy = torch.sum(val_y).item() / val_y.size()[0]

        epoch_train_loss_ls.append(train_loss)
        epoch_val_loss_ls.append(val_loss)
        epoch_val_acc_ls.append(val_accuracy)

        scheduler.step()

        wandb.log({'epoch/epoch': epoch, 'epoch/train_loss': train_loss, 'epoch/val_loss': val_loss,
                   'epoch/val_accuracy': val_accuracy, 'epoch/lr': optimizer.param_groups[0]['lr']})  #
        print(
            f"epoch: {epoch:03d}/{n_epochs}  train_loss: {train_loss}, test_loss: {val_loss}, val_accuracy: {val_accuracy}")  #

        """ Save Model """
        # save best
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            print("Best model updated", f"[epoch {epoch:04d}]")
            # Save the best model up-to-now
            checkpoint_name = "best"
            torch.save(embedding_model.state_dict(),
                       os.path.join(output_path, f"{checkpoint_name}_embedding.pt"))
            print(f"Save best model at {output_path}/{checkpoint_name}_embedding.pt")

        # regularly save checkpoint
        if 0 == epoch % opt.checkpoint_frequency:
            checkpoint_name = f"{epoch:05d}"
            torch.save(embedding_model.state_dict(),
                       os.path.join(output_path, f"{checkpoint_name}_embedding.pt"))
            print(f"Regularly save model at {output_path}/{checkpoint_name}_embedding.pt")

            gc.collect()

    print(f"Train Finished. Best epoch: {best_epoch}. Overall validation accuracy: {np.mean(epoch_val_acc_ls)}")

    """ ********************* Predict ********************* """
    # Load best model
    if os.path.isfile(opt.model_path):
        embedding_model.load_state_dict(torch.load(opt.model_path))
        print(f'Load model from {opt.model_path}')
    else:
        print(f'Load best model.')
        embedding_model.load_state_dict(torch.load(os.path.join(output_path, "best_embedding.pt")))

    embedding_model.eval()
    triplet_model.eval()

    pred_y_ls = []
    for batch_idx, (triplet_data, _) in enumerate(test_triplet_loader, start=1):
        # Load data
        anchor: torch.Tensor = triplet_data[0].to(device)
        positive: torch.Tensor = triplet_data[1].to(device)
        negative: torch.Tensor = triplet_data[2].to(device)
        # Forward
        pred_y, triplet_loss = triplet_model.predict_with_loss(anchor, positive, negative)
        pred_y_ls.append(pred_y.int())

        wandb.log({'predict/batch_idx': batch_idx})
        if 0 == batch_idx % opt.log_frequency:
            print(f"Predicting: {batch_idx:05d}/{len(test_triplet_loader):05d}")

    gc.collect()
    pred_y_result = torch.cat(pred_y_ls, 0)
    print("cat")
    pred_y_result = pred_y_result.cpu()
    print("cpu")
    pred_y_result = pred_y_result.numpy()
    print("length", len(pred_y_result))
    print("numpy")
    y_df = pd.DataFrame(pred_y_result)
    csv_path = os.path.join(output_path, 'prediction.csv')
    y_df.to_csv(csv_path, index=False, header=False)
    print(f"Result saved to {csv_path}")


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="IML Task 3 v2")
    parser.add_argument("--name", type=str, default="",
                        help="name of this test")
    # parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--output_root", type=str, default="/cluster/scratch/jingyli/iml/data")
    parser.add_argument("--user_home", type=str)
    parser.add_argument("--model_path", type=str, help="Existing model, full path, set 0 if not to use")  # TODO
    parser.add_argument("--feature_path", type=str, help="Existing features, full path, set 0 if not to use")  # TODO
    parser.add_argument("--train_data_percentage", type=float, default=0.8)
    # parser.add_argument("--freeze_pretrained", action='store_true', default=False)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="how many samples per batch")
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--triplet_margin", type=float, default=0.1)
    parser.add_argument("--scheduler_step", type=int, default=10)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_frequency", type=int, default=10,
                        help="unit: iteration, current training loss would be printed")
    # parser.add_argument("--validate_frequency", type=int, default=200,
    #                     help="unit: iteration, current triplet_model would be tested on the validation set")
    parser.add_argument("--checkpoint_frequency", type=int, default=2,
                        help="unit: epoch, current triplet_model would be saved in the output folder")

    parser.add_argument("--only_first_n", type=int, default=0)
    args = parser.parse_args()

    print(args)
    main(args)
