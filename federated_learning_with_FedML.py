import argparse
import logging
import os
import random
import sys
from typing import DefaultDict
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "FedML")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "cifar_gm")))
# sys.path.insert(1, os.getcwd())
# sys.path.insert(2, os.path.abspath(os.path.join(os.getcwd(), "FedML/fedml_api")))
# from graph_matching_based_alignment.FedML import *

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS


# my own model:
import model_gm


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit

    each model trained for [epochs] epochs per communication
    the total epochs trained, therefore, is [comm_round * epochs]

    """
    # Training settings
    parser.add_argument('--model-name', type=str, default='vgg11_nobias', metavar='N',
                        help='neural network used in training')
    parser.add_argument('--second-model-name', type=str, default=None, action='store', help='name of second model!')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./FedML/data/cifar10',
                        help='data directory')
    # adjust
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers (default: hetero, namely non-iid)')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)
    # adjust
    parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=5, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=5, metavar='NN',
                        help='number of workers')
    # adjust
    parser.add_argument('--comm_round', type=int, default=50,
                        help='how many round of communications we shoud use')
    # adjust
    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu-id', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--fusion_mode', type=str, default='fusion_gamf_multi',
                        help='the method used to fuse different models, [traditional, ot, fusion, fusion_gamf]')

    parser.add_argument('--reg', default=1e-2, type=float, help='regularization strength for sinkhorn (default: 1e-2)')
    parser.add_argument('--reg-m', default=1e-3, type=float,
                        help='regularization strength for marginals in unbalanced sinkhorn (default: 1e-3)')
    parser.add_argument('--ground-metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                        help='ground metric for OT calculations.')
    parser.add_argument('--ground-metric-normalize', type=str, default='log',
                        choices=['log', 'max', 'none', 'median', 'mean'],
                        help='ground metric normalization to consider! ')
    parser.add_argument('--not-squared', action='store_true', help='dont square the ground metric')
    parser.add_argument('--clip-gm', action='store_true', help='to clip ground metric')
    parser.add_argument('--clip-min', action='store', type=float, default=0,
                        help='Value for clip-min for gm')
    parser.add_argument('--clip-max', action='store', type=float, default=5,
                        help='Value for clip-max for gm')
    parser.add_argument('--tmap-stats', action='store_true', help='print tmap stats')
    parser.add_argument('--ensemble-step', type=float, default=0.5, action='store',
                        help='rate of adjustment towards the second model')

    parser.add_argument('--ground-metric-eff', action='store_true',
                        help='memory efficient calculation of ground metric')

    parser.add_argument('--weight-stats', action='store_true', help='log neuron-wise weight vector stats.')
    parser.add_argument('--sinkhorn-type', type=str, default='normal',
                        choices=['normal', 'stabilized', 'epsilon', 'gpu'],
                        help='Type of sinkhorn algorithm to consider.')
    parser.add_argument('--geom-ensemble-type', type=str, default='wts', choices=['wts', 'acts'],
                        help='Ensemble based on weights (wts) or activations (acts).')
    parser.add_argument('--act-bug', action='store_true',
                        help='simulate the bug in ground metric calc for act based averaging')
    parser.add_argument('--standardize-acts', action='store_true',
                        help='subtract mean and divide by standard deviation across the samples for use in act based alignment')
    parser.add_argument('--transform-acts', action='store_true',
                        help='transform activations by transport map for later use in bi_avg mode ')
    parser.add_argument('--center-acts', action='store_true',
                        help='subtract mean only across the samples for use in act based alignment')
    parser.add_argument('--prelu-acts', action='store_true',
                        help='do activation based alignment based on pre-relu acts')
    parser.add_argument('--pool-acts', action='store_true',
                        help='do activation based alignment based on pooling acts')
    parser.add_argument('--pool-relu', action='store_true',
                        help='do relu first before pooling acts')
    parser.add_argument('--normalize-acts', action='store_true',
                        help='normalize the vector of activations')
    parser.add_argument('--normalize-wts', action='store_true',
                        help='normalize the vector of weights')
    parser.add_argument('--gromov', action='store_true', help='use gromov wasserstein distance and barycenters')
    parser.add_argument('--gromov-loss', type=str, default='square_loss', action='store',
                        choices=['square_loss', 'kl_loss'],
                        help="choice of loss function for gromov wasserstein computations")
    parser.add_argument('--tensorboard-root', action='store', default="./tensorboard", type=str,
                        help='Root directory of tensorboard logs')
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard to plot the loss values')

    parser.add_argument('--same-model', action='store', type=int, default=-1,
                        help='Index of the same model to average with itself')
    parser.add_argument('--dist-normalize', action='store_true', help='normalize distances by act num samples')
    parser.add_argument('--update-acts', action='store_true', help='update acts during the alignment of model0')
    parser.add_argument('--past-correction', action='store_true',
                        help='use the current weights aligned by multiplying with past transport map')
    parser.add_argument('--partial-reshape', action='store_true',
                        help='partially reshape the conv layers in ground metric calculation')
    parser.add_argument('--choice', type=str, default='0 2 4 6 8', action='store',
                        help="choice of how to partition the labels")
    parser.add_argument('--diff-init', action='store_true',
                        help='different initialization for models in data separated mode')

    return parser


def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    data_loader = load_partition_data_cifar10
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                            args.partition_alpha, args.client_num_in_total, args.batch_size)

    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = model_gm.get_model_from_name(name=model_name, args=args)
    return model


def custom_model_trainer(args, model):
    return MyModelTrainerCLS(model)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    wandb.init(
        project="fedml-cifar10",
        name="FedAVG-" + str(args.fusion_mode) + "-r" + str(args.comm_round) + "-e" + str(
            args.epochs) + "-lr" + str(args.lr) + "-c" + str(args.client_num_per_round),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    # load data
    dataset = load_data(args, args.dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model_name, output_dim=dataset[7])
    model_trainer = custom_model_trainer(args, model)
    logging.info(model)

    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer)
    fedavgAPI.train()  # _aggregate: aggregate the parameters
