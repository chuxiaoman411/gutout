import pdb
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import sys
import time
import copy

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.gutout.gutout_utils import BatchGradCam
from src.utils.data_utils import get_dataloaders
from src.training.training_utils import (
    get_args,
    get_optimizer_and_schedular,
    get_csv_logger,
    get_model,
    create_experiment_dir,
    run_epoch,
    test,
)


if __name__ == "__main__":

    # parse arguments
    args, max_num_batches = get_args()

    # create train and test dataloaders
    train_loader, test_loader = get_dataloaders(args)

    # create model and optimizer
    model_a = get_model(args, weights_path=args.model_a_path)
    model_b = get_model(args, weights_path=args.model_b_path)
    optimizer_a, scheduler_a = get_optimizer_and_schedular(model_a, args)
    optimizer_b, scheduler_b = get_optimizer_and_schedular(model_b, args)

    # create experiment dir, csv logg and criterion
    experiment_dir, experiment_id = create_experiment_dir(args)
    csv_logger_a = get_csv_logger(experiment_dir, experiment_id, args, model_flag="a")
    csv_logger_b = get_csv_logger(experiment_dir, experiment_id, args, model_flag="b")
    criterion = nn.CrossEntropyLoss()

    # cast to gpu if needed
    if args.use_cuda:
        model_a = model_a.cuda()
        model_b = model_b.cuda()
        criterion.cuda()

    # for a single model, the gutout model is the model that is being trained
    best_acc_a = -1
    best_acc_b = -1

    # set the model that will train for the first set of epochs
    training_model = model_a
    optimizer = optimizer_a
    scheduler = scheduler_a
    csv_logger = csv_logger_a
    best_acc = copy.copy(best_acc_a)
    training_flag = "a"

    # in sequential training: if model a is training then also model_a is the gutout model
    gutout_model = model_a

    # create the gradCAM model
    grad_cam = BatchGradCam(
        model=gutout_model,
        feature_module=getattr(gutout_model, args.feature_module),
        target_layer_names=[args.target_layer_names],
        use_cuda=args.use_cuda,
    )

    ##### train model a #######
    if args.model_a_path:
        # run test epoch
        test_acc = test(model_a, test_loader, args, max_num_batches)

        # write row
        tqdm.write("model a test_acc: %.3f" % (test_acc))

    for epoch in range(0 if args.model_a_path else args.epochs):
        # run the training loop on a single model
        print(f"running epoch with model: {training_flag}")
        best_acc = run_epoch(
            training_model,
            grad_cam,
            criterion,
            optimizer,
            scheduler,
            csv_logger,
            train_loader,
            test_loader,
            epoch,
            best_acc,
            max_num_batches,
            experiment_dir,
            experiment_id,
            args,
            model_flag=training_flag,
        )

    # switch to training model b
    training_model = model_b
    optimizer = optimizer_b
    scheduler = scheduler_b
    csv_logger = csv_logger_b
    best_acc = copy.copy(best_acc_b)
    training_flag = "b"

    # if model b is training, the model a is the gutout model
    gutout_model = model_a

    ##### train model a #######
    for epoch in range(args.epochs):
        # run the training loop on a single model
        print(f"running epoch with model: {training_flag}")
        best_acc = run_epoch(
            training_model,
            grad_cam,
            criterion,
            optimizer,
            scheduler,
            csv_logger,
            train_loader,
            test_loader,
            epoch,
            best_acc,
            max_num_batches,
            experiment_dir,
            experiment_id,
            args,
            model_flag=training_flag,
        )
