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
import argparse
import sys

# parser = argparse.ArgumentParser()
# parser.add_argument("path_config", help="display a path to the config file",
#                     type=str)
# args = parser.parse_args()


# parse config file as an argument
args = str(sys.argv[1])
print(args)
DATA_PATH = os.path.realpath(os.path.dirname(__file__))

utils = Utils(DATA_PATH)

configuration = utils.parse_configuration(args)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = int(multiprocessing.cpu_count() / 2)


N_EPOCHS = configuration["model_params"]["N_EPOCHS"]
N_SPLITS = configuration["model_params"]["n_splits"]
BATCH_SIZE = configuration["model_params"]["batch_size"]
EVAL_MODES = ["normal"]
RES_PATH = os.path.join(DATA_PATH, configuration["output_parameters"]["result_path"])

PATH_LOSS = configuration["output_parameters"]["path_losses_output"]

# create folders for results if not exist
if not os.path.exists(PATH_LOSS):
    os.makedirs(PATH_LOSS)
if not os.path.exists(RES_PATH):
    os.makedirs(RES_PATH)

writer = SummaryWriter(configuration["output_parameters"]["path_tesnorboard_output"])


def training_loop(loader, model, loss_cl, opt, epoch):
    """
    Training loop of `model` using data from `loader` and
    loss functions from `loss_cl` using optimizer `opt`.
    """
    target_pkd_all = []
    model = model.train()
    progress = tqdm(loader)
    all_rmsd = []
    pkd_pred = []
    for idx, features, geometry, target_pkd in progress:
        idx = idx.to(DEVICE)
        features = features.to(DEVICE)
        geometry = geometry.to(DEVICE)

        target_pkd = target_pkd.to(DEVICE)
        target_pkd_all.append(target_pkd)
        opt.zero_grad()

        out1 = model(features, geometry)
        pkd_pred.append(out1.cpu())
        loss_rmsd_pkd = loss_cl(out1, target_pkd).float()

        writer.add_scalar("training_loss", loss_rmsd_pkd.item(), epoch)

        loss_rmsd_pkd.backward()
        opt.step()

        progress.set_postfix(
            {"loss_rmsd_pkd": loss_rmsd_pkd.item(),}
        )
        all_rmsd.append(loss_rmsd_pkd.item())
    return torch.cat(target_pkd_all), torch.cat(pkd_pred), sum(all_rmsd) / len(all_rmsd)


def eval_loop(loader, model, epoch):
    """
    Evaluation loop using `model` and data from `loader`.
    """
    model = model.eval()
    progress = tqdm(loader)

    target_pkd_all = []
    pkd_pred = []
    all_rmsd = []
    for idx, features, geometry, target_pkd in progress:
        with torch.no_grad():
            features = features.to(DEVICE)
            geometry = geometry.to(DEVICE)

            out1 = model(features, geometry).to(DEVICE)
            target_pkd = target_pkd.to(DEVICE)
            target_pkd_all.append(target_pkd)
            pkd_pred.append(out1.cpu())

            loss_rmsd_pkd = loss_cl(out1, target_pkd).float()
            progress.set_postfix(
                {"loss_rmsd_pkd": loss_rmsd_pkd.item(),}
            )
            writer.add_scalar("test_loss", loss_rmsd_pkd.item(), epoch)
            all_rmsd.append(loss_rmsd_pkd.item())

    return torch.cat(target_pkd_all), torch.cat(pkd_pred), sum(all_rmsd) / len(all_rmsd)


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

    # os.makedirs(RES_PATH, parents = True, exist_ok=True)
    for mode in EVAL_MODES:
        split_pdbids.setdefault(mode, [])

        # get indices of train and test data
        # train_data, test_data = utils._get_train_test_data(data_ids)
        # train_data, test_data = utils._get_dataset_preparation()
        if configuration["train_dataset_params"]["splitting"] == "casf":
            train_data, test_data = utils._get_core_train_test_casf()
        else:
            # train and test from refined set (4850 pdb)
            train_data, test_data = utils._get_train_test_data(data_ids)

        # train_data = train_data[1:3]
        # test_data = test_data[1:3]

        pdbids = [
            data_names[t] for t in test_data
        ]  # names of pdb corresponding to test data indexes
        split_pdbids[mode].append(pdbids)

        feat_train = [featuriser[data] for data in train_data]
        feat_test = [featuriser[data] for data in test_data]

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
        losses_to_write = []
        for i in range(N_EPOCHS):

            print("Epoch {}/{}...".format(i + 1, N_EPOCHS))
            epoch = i + 1
            target_pkd_all, pkd_pred, loss = training_loop(
                loader_train, model, loss_cl, opt, epoch
            )
            losses_to_write.append(loss)
            np.save(
                os.path.join(RES_PATH, "target_pkd_all_train.npy"),
                arr=target_pkd_all.detach().cpu().clone().numpy(),
            )
            np.save(
                os.path.join(RES_PATH, "pkd_pred_train_{}.npy".format(str(i))),
                arr=pkd_pred.detach().cpu().clone().numpy(),
            )
            scheduler.step()
        losses_to_write = np.asarray(losses_to_write, dtype=np.float32)
        np.savetxt(
            os.path.join(PATH_LOSS, "losses_train_2016.out"),
            losses_to_write,
            delimiter=",",
        )

        print("Evaluating model...")
        target_pkd_all, pkd_pred, loss_test_to_write = eval_loop(
            loader_test, model, epoch
        )
        loss_test_to_write = np.asarray(loss_test_to_write, dtype=np.float32)
        loss_test_to_write = np.asarray([loss_test_to_write])
        np.savetxt(
            os.path.join(PATH_LOSS, "losses_test_2016.out"),
            loss_test_to_write,
            delimiter=",",
        )

        os.makedirs(RES_PATH, exist_ok=True)

        # Save results for later evaluation
        np.save(
            os.path.join(RES_PATH, "target_pkd_all_test.npy"),
            arr=target_pkd_all.detach().cpu().clone().numpy(),
        )
        np.save(
            os.path.join(RES_PATH, "pkd_pred_test.npy"),
            arr=pkd_pred.detach().cpu().clone().numpy(),
        )

    with open(os.path.join(RES_PATH, "split_pdbids.pt"), "wb") as handle:
        pickle.dump(split_pdbids, handle)

    # utils.plot_statistics(
    #     RES_PATH, configuration['output_parameters']['name_plot'])
