import numpy as np
import torch
import torch.nn as nn
import math
from data_loader import get_dataset
from running import one_round_training
from methods import local_update
from models import ModelResNet, ModelResNet_Adapter, ModelResNet_Hyper
from options import args_parser
import tools
import copy
import time
import os

torch.set_num_threads(4)

if __name__ == '__main__':
    args = args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    # load dataset and user groups
    train_loader, test_loader, global_test_loader = get_dataset(args)
    seed = 520
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # construct model
    if args.dataset in ['cifar', 'cifar10']:
        if args.train_rule == 'HyperFL-LPM':
            global_model = ModelResNet_Hyper(args=args).to(device)
        elif args.train_rule == 'Local-Adapter' or args.train_rule == 'FedAvg-Adapter':
            global_model = ModelResNet_Adapter(args=args).to(device)
        else:
            global_model = ModelResNet(num_classes=args.num_classes).to(device)
    elif args.dataset == 'emnist':
        args.num_classes = 62
        if args.train_rule == 'HyperFL-LPM':
            global_model = ModelResNet_Hyper(args=args).to(device)
        elif args.train_rule == 'Local-Adapter' or args.train_rule == 'FedAvg-Adapter':
            global_model = ModelResNet_Adapter(args=args).to(device)
        else:
            global_model = ModelResNet(num_classes=args.num_classes).to(device)
    else:
        raise NotImplementedError()

    print(args)

    # Training Rule
    LocalUpdate = local_update(args.train_rule)
    # One Round Training Function
    train_round_parallel = one_round_training(args.train_rule)

    # Training
    train_loss = []
    local_accs1, local_accs2 = [], []
    local_clients = []
    for idx in range(args.num_users):
        if args.dataset == 'covid_fl' or args.dataset == 'retina':
            local_clients.append(LocalUpdate(idx=idx, args=args, train_set=train_loader[idx], test_set=test_loader,
                                             model=copy.deepcopy(global_model)))
        else:
            local_clients.append(LocalUpdate(idx=idx, args=args, train_set=train_loader[idx], test_set=test_loader[idx],
                                             model=copy.deepcopy(global_model)))

    for round in range(args.epochs):
        loss1, loss2, local_acc1, local_acc2 = train_round_parallel(args, global_model, local_clients, round)
        train_loss.append(loss1)
        print("Train Loss: {}, {}".format(loss1, loss2))
        print("Local Accuracy on Local Data: {}%, {}%".format(local_acc1, local_acc2))
        local_accs1.append(local_acc1)
        local_accs2.append(local_acc2)

    print(max(local_accs1))