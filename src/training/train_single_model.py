import pdb
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import sys
import time

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
)

# python src/training/train_single_model.py --smoke_test 0 --use_cuda --model torchvision_resnet18
# python src/training/train_single_model.py --smoke_test 0 --use_cuda --model cutout_resnet18

if __name__ == "__main__":

    # parse arguments
    args, max_num_batches = get_args()

    # create train and test dataloaders
    train_loader, test_loader = get_dataloaders(args)

    # create model and optimizer
    model = get_model(args, weights_path=args.model_a_path)
    optimizer, scheduler = get_optimizer_and_schedular(model, args)

    # create experiment dir, csv logg and criterion
    experiment_dir, experiment_id = create_experiment_dir(args)
    csv_logger = get_csv_logger(experiment_dir, experiment_id, args, model_flag="a")
    criterion = nn.CrossEntropyLoss()

    # cast to gpu if needed
    if args.use_cuda:
        model = model.cuda()
        criterion.cuda()

    # for a single model, the gutout model is the model that is being trained
    gutout_model = model
    training_model = model
    best_acc = -1

    # create a gradcam
    grad_cam = BatchGradCam(
        model=gutout_model,
        feature_module=getattr(gutout_model, args.feature_module),
        target_layer_names=[args.target_layer_names],
        use_cuda=args.use_cuda,
    )

    # run the training loop on a single model
    for epoch in range(args.epochs):
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
            model_flag="a",
        )
