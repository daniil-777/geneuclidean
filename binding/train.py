import multiprocessing
import os
import pickle
import json
import numpy as np
from numpy import savetxt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from network import EuclideanNet, SE3Net
from ACNE3 import se3ACN
# from network1 import EuclideanNet, SE3Net
# from network_utils import Loss, Pdb_Dataset
from data_loader import Loss, Pdb_Dataset
from utils import Utils
import argparse
import sys

# parser = argparse.ArgumentParser()
# parser.add_argument("path_config", help="display a path to the config file",
#                     type=str)
# args = parser.parse_args()


# parse config file as an argument
args = str(sys.argv[1])
# args = "configs/tetris_simple.json"
print(args)
# ags = "configs/tetris_simple.json"
# DATA_PATH = os.path.realpath(os.path.dirname(__file__))
# DATA_PATH = '/Volumes/Ubuntu'


with open(args) as json_file:
    config =  json.load(json_file)

# config = utils.parse_configuration(args)

utils = Utils(config)

DATA_PATH = config["preprocessing"]["path_root"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = int(multiprocessing.cpu_count() / 2)
# NUM_WORKERS = 0


N_EPOCHS = config["model_params"]["N_EPOCHS"]
print(N_EPOCHS)
N_SPLITS = config["model_params"]["n_splits"]
BATCH_SIZE = config["model_params"]["batch_size"]
EVAL_MODES = ["normal"]
RES_PATH = os.path.join(DATA_PATH, config["output_parameters"]["path_results"])
PKD_PATH = os.path.join(RES_PATH, config["output_parameters"]["pkd_path"])

PATH_LOSS = os.path.join(RES_PATH, config["output_parameters"]["path_losses_output"])

PATH_PLOTS = config["output_parameters"]["output_plots"]
# create folders for results if not exist
if not os.path.exists(RES_PATH):
    os.makedirs(RES_PATH)


if not os.path.exists(PATH_LOSS):
    os.makedirs(PATH_LOSS)
if not os.path.exists(PKD_PATH):
    os.makedirs(PKD_PATH)

if not os.path.exists(PATH_PLOTS):  
    os.makedirs(PATH_PLOTS)


# writer = SummaryWriter(config["output_parameters"]["path_tesnorboard_output"])


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
        # num_atoms= num_atoms.to(DEVICE)
        target_pkd = target_pkd.to(DEVICE)
        target_pkd_all.append(target_pkd)
        opt.zero_grad()

        # out1 = model(features, geometry)
        out1 = model(geometry, features)
        pkd_pred.append(out1.cpu())
        # print(out1.cpu())
        loss_rmsd_pkd = loss_cl(out1, target_pkd).float()

        # writer.add_scalar("training_loss", loss_rmsd_pkd.item(), epoch)

        loss_rmsd_pkd.backward()
        opt.step()

        # progress.set_postfix(
        #     {"loss_rmsd_pkd": loss_rmsd_pkd.item(),}
        # )
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

            # out1 = model(features, geometry).to(DEVICE)
            out1 = model(geometry, features).to(DEVICE)
            target_pkd = target_pkd.to(DEVICE)
            target_pkd_all.append(target_pkd)
            pkd_pred.append(out1.cpu())

            loss_rmsd_pkd = loss_cl(out1, target_pkd).float()
            # progress.set_postfix(
                # {"loss_rmsd_pkd": loss_rmsd_pkd.item(),}
            # )
            # writer.add_scalar("test_loss", loss_rmsd_pkd.item(), epoch)
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
    featuriser = Pdb_Dataset(config)

    # os.makedirs(PKD_PATH, parents = True, exist_ok=True)
    for mode in EVAL_MODES:
        split_pdbids.setdefault(mode, [])

        # get indices of train and test data
        # train_data, test_data = utils._get_train_test_data(data_ids)
        # train_data, test_data = utils._get_dataset_preparation()
        if config["train_dataset_params"]["splitting"] == "casf":
            train_data, test_data = utils._get_core_train_test_casf()
            # print("train data casf", train_data)
            # print(len(train_data))
            # print("------------")
            # # print(test_data)
            # print(len(test_data))
        else:
            # train and test from refined set (4850 pdb)
            train_data, test_data = utils._get_train_test_data(data_ids)
            print("train data", train_data)

        # train_data = train_data[1:5]
        # test_data = test_data[1:5]
        print("test_data", test_data)
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

        # model = EuclideanNet(
        #     config["model_params"]["input_channels"]).to(DEVICE).float()
        if(config["model_params"]["network"] == "Euclidean_atoms"):
            model = EuclideanNet().to(DEVICE).float() #how on the main page of e3nn github
        elif(config["model_params"]["network"] == "tetris"):
            print("tetris!!!")
            # model = SE3Net(config["model_params"]["representations"]).to(DEVICE).float()
            model = SE3Net(config["model_params"]["representations"]).to(
                DEVICE).float()
        elif(config["model_params"]["network"] == "kenneth"):
            print("kenneth!!!")
            model = se3ACN().to(DEVICE).double()
            

        loss_cl = Loss()
        opt = Adam(model.parameters(),
                   lr=config["model_params"]["learning_rate"])
        scheduler = ExponentialLR(opt, gamma=0.95)

        print("Training model...")
        losses_to_write_train = []
        for i in range(N_EPOCHS):

            print("Epoch {}/{}...".format(i + 1, N_EPOCHS))
            epoch = i + 1
            target_pkd_all, pkd_pred, loss = training_loop(
                loader_train, model, loss_cl, opt, epoch
            )
            print("pkd_pred", pkd_pred)
            losses_to_write_train.append(loss)

            if i == N_EPOCHS - 1:
                # for local debugging csv
                # savetxt(
                #     os.path.join(
                #         PKD_PATH, "pkd_pred_train_{}.csv".format(str(i))),
                #     pkd_pred.detach().cpu().clone().numpy(),
                # )

                np.save(
                    os.path.join(PKD_PATH, "pkd_pred_train_{}.npy".format(str(i))),
                    arr=pkd_pred.detach().cpu().clone().numpy(),
                )
            scheduler.step()
        losses_to_write_train = np.asarray(losses_to_write_train, dtype=np.float32)
        # save losses for the train
        np.savetxt(
            os.path.join(PATH_LOSS, "losses_train_2016.out"),
            losses_to_write_train,
            delimiter=",",
        )
        # save true values of training target
        savetxt(
            os.path.join(PKD_PATH, "target_pkd_all_train.csv"),
            target_pkd_all.detach().cpu().clone().numpy(),
        )

        np.save(
            os.path.join(PKD_PATH, "target_pkd_all_train"),
            arr=target_pkd_all.detach().cpu().clone().numpy(),
        )

        print("Evaluating model...")
        target_pkd_all_test, pkd_pred_test, loss_test_to_write = eval_loop(
            loader_test, model, epoch
        )
        print("pkd_pred", pkd_pred_test)
        loss_test_to_write = np.asarray(loss_test_to_write, dtype=np.float32)
        loss_test_to_write = np.asarray([loss_test_to_write])
        np.savetxt(
            os.path.join(PATH_LOSS, "losses_test_2016.out"),
            loss_test_to_write,
            delimiter=",",
        )

        os.makedirs(PKD_PATH, exist_ok=True)

        # Save results for later evaluation

        # for local debugging csv
        # savetxt(
        #     os.path.join(PKD_PATH, "target_pkd_all_test.csv"),
        #     target_pkd_all_test.detach().cpu().clone().numpy(),
        # )

        # savetxt(
        #     os.path.join(PKD_PATH, "pkd_pred_test.csv"),
        #     pkd_pred_test.detach().cpu().clone().numpy(),
        # )

        np.save(
            os.path.join(PKD_PATH, "target_pkd_all_test"),
            arr=target_pkd_all_test.detach().cpu().clone().numpy(),
        )
        np.save(
            os.path.join(PKD_PATH, "pkd_pred_test"),
            arr=pkd_pred_test.detach().cpu().clone().numpy(),
        )

    with open(os.path.join(PKD_PATH, "split_pdbids.pt"), "wb") as handle:
        pickle.dump(split_pdbids, handle)

    utils.plot_statistics(
        PKD_PATH,
        PATH_PLOTS,
        N_EPOCHS,
        config["output_parameters"]["name_plot"],
        "train",
        losses_to_write_train[-1],
        loss_test_to_write[0],
    )

    utils.plot_statistics(
        PKD_PATH,
        PATH_PLOTS,
        N_EPOCHS,
        config["output_parameters"]["name_plot"],
        "test",
        losses_to_write_train[-1],
        loss_test_to_write[0],

    )

    utils.plot_losses(
        PATH_LOSS, PATH_PLOTS, N_EPOCHS, config["output_parameters"]["name_plot"]
    )
