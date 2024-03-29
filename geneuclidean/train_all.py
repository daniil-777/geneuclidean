import multiprocessing
import os
import pickle

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from network import EuclideanNet
from network_utils import Loss, Pdb_Dataset
from utils import Utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = int(multiprocessing.cpu_count() / 2)

DATA_PATH = os.path.realpath(os.path.dirname(__file__))

RES_PATH = os.path.join(DATA_PATH, "results_core")

N_EPOCHS = 200
N_SPLITS = 5
BATCH_SIZE = 32
EVAL_MODES = ["normal"]

utils = Utils(DATA_PATH)

writer = SummaryWriter("runs_all_core/pdbbind_all_experiment_152")


def training_loop(loader, model, loss_cl, opt, epoch):
    """
    Training loop of `model` using data from `loader` and
    loss functions from `loss_cl` using optimizer `opt`.
    """

    model = model.train()
    progress = tqdm(loader)

    for idx, features, geometry, target_pkd in progress:
        idx = idx.to(DEVICE)
        features = features.to(DEVICE)
        geometry = geometry.to(DEVICE)

        target_pkd = target_pkd.to(DEVICE)

        opt.zero_grad()

        out1 = model(features, geometry)
        loss_rmsd_pkd = loss_cl(out1, target_pkd).float()
     
        writer.add_scalar("training_loss", loss_rmsd_pkd.item(), epoch)

        # ...log a Matplotlib Figure showing the model's predictions on a
        # random mini-batch
        # writer.add_figure('predictions vs. actuals',
        #                   out1,
        #                   target_pkd)
        # print("type loss_rmsd ")
        # print(type(loss_rmsd_pkd))
        # print(
        #     type(loss_rmsd_pkd)
        # )  # I output in forward of NET just a feature x of one particluar complex

        loss_rmsd_pkd.backward()
        opt.step()

        progress.set_postfix(
            {"loss_rmsd_pkd": loss_rmsd_pkd.item(),}
        )


def eval_loop(loader, model, epoch):
    """
    Evaluation loop using `model` and data from `loader`.
    """
    model = model.eval()
    progress = tqdm(loader)

    target_pkd_all = []
    pkd_pred = []

    for idx, features, geometry, target_pkd in progress:
        with torch.no_grad():
            features = features.to(DEVICE)
            geometry = geometry.to(DEVICE)

            out1 = model(features, geometry)
            target_pkd_all.append(target_pkd)
            pkd_pred.append(out1.cpu())

            loss_rmsd_pkd = loss_cl(out1, target_pkd).float()
            progress.set_postfix(
                {"loss_rmsd_pkd": loss_rmsd_pkd.item(),}
            )
            writer.add_scalar("test_loss", loss_rmsd_pkd.item(), epoch)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            # writer.add_figure('predictions vs. actuals',
            #                 out1,
            #                 target_pkd)

    return torch.cat(target_pkd_all), torch.cat(pkd_pred)


if __name__ == "__main__":
    # get indexes of all complexes and "nick names"
    data_ids, data_names = utils._get_refined_data()
    # print("furst data names")
    # print(data_names)
    data_names = utils._get_names_refined_core()
    # print("second data names")
    # print(data_names)
    split_pdbids = {}
    print(DATA_PATH)
    featuriser = Pdb_Dataset(DATA_PATH)
    for mode in EVAL_MODES:
        split_pdbids.setdefault(mode, [])

        # get indices of train and test data
        # train_data, test_data = utils._get_train_test_data(data_ids)
        # train_data, test_data = utils._get_dataset_preparation()
        train_data, test_data = utils._get_core_train_test()
        # print(train_data)
        # print(len(train_data))
        # print("---------")
        # print(test_data)
        # print(len(test_data))
        train_data = train_data[1:100]
        pdbids = [
            data_names[t] for t in test_data
        ]  # names of pdb corresponding to test data indexes
        split_pdbids[mode].append(pdbids)

        # indexes of training/test data

        #############################################################
        # does not work ???
        # feat_train = Pdb_Dataset(*train_data)
        # feat_test = Pdb_Dataset(*test_data)
        # print(train_data)
        feat_train = [featuriser[data] for data in train_data]
        feat_test = [featuriser[data] for data in test_data]
        #############################################################

        loader_train = DataLoader(
            feat_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True
        )

        loader_test = DataLoader(
            feat_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
        )

        model = EuclideanNet().to(DEVICE).float()
        loss_cl = Loss()
        opt = Adam(model.parameters())
        scheduler = ExponentialLR(opt, gamma=0.95)

        print("Training model...")
        for i in range(N_EPOCHS):
            print("Epoch {}/{}...".format(i + 1, N_EPOCHS))
            epoch = i + 1
            training_loop(loader_train, model, loss_cl, opt, epoch)
            scheduler.step()

        print("Evaluating model...")
        target_pkd_all, pkd_pred = eval_loop(loader_test, model, epoch)

        os.makedirs(RES_PATH, exist_ok=True)

        # Save results for later evaluation
        np.save(
            os.path.join(RES_PATH, "target_pkd_all_{}.npy".format(mode)),
            arr=target_pkd_all.numpy(),
        )
        np.save(
            os.path.join(RES_PATH, "pkd_pred_{}.npy".format(mode)),
            arr=pkd_pred.numpy(),
        )

    with open(os.path.join(RES_PATH, "split_pdbids.pt"), "wb") as handle:
        pickle.dump(split_pdbids, handle)
