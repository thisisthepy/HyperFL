import torch
from torch import nn
import tools
import numpy as np
import copy
import math
import json
from tools import average_weights_weighted
import os
import random


def one_round_training(rule):
    # gradient aggregation rule
    Train_Round = {'FedAvg': train_round_fedavg,
                   'FedAvg-Adapter': train_round_fedavg,
                   'Local': train_round_standalone,
                   'Local-Adapter': train_round_standalone,
                   'HyperFL-LPM': train_round_hyperfl,
                   }
    return Train_Round[rule]


## training methods -------------------------------------------------------------------
# local training only
def train_round_standalone(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_losses1, local_losses2 = [], []
    local_acc1 = []
    local_acc2 = []

    global_weight = global_model.state_dict()

    for idx in idx_users:
        local_client = local_clients[idx]
        local_epoch = args.local_epoch
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch)
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2


# vanila FedAvg
def train_round_fedavg(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_grads = []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()

    for idx in idx_users:
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2


def train_round_hyperfl(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)

    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []  # aggregation weights for hypernetwork

    if rnd <= args.epochs:
        for idx in idx_users:
            local_client = local_clients[idx]

            ## local training
            local_epoch = args.local_epoch
            # w here denotes hypernetwork's weight
            w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, round=rnd)
            local_weights.append(copy.deepcopy(w))
            local_losses1.append(copy.deepcopy(loss1))
            local_losses2.append(copy.deepcopy(loss2))
            local_acc1.append(acc1)
            local_acc2.append(acc2)
            agg_weight.append(local_client.agg_weight)

        # get weight for hypernetwork aggregation
        agg_weight = torch.stack(agg_weight).to(args.device)

        # update global hypernetwork
        global_weight_new = average_weights_weighted(local_weights, agg_weight)

        for idx in range(num_users):
            local_client = local_clients[idx]
            local_client.update_hypernetwor(global_weight=global_weight_new)  # update local hypernetwork

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2
