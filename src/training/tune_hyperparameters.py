import pdb
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import sys
import time
import random

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.gutout.gutout_utils import BatchGradCam
from src.utils.data_utils import get_dataloaders
from src.utils.misc import CSVLogger
from src.training.training_utils import (
    get_args,
    get_optimizer_and_schedular,
    get_model,
    create_experiment_dir,
    run_epoch,
    train,
    test
)

if __name__ == "__main__":

    # parse arguments
    args, max_num_batches = get_args(hypterparameters_tune=True)

    # create train and test dataloaders
    train_loader, test_loader = get_dataloaders(args)

    # create model and optimizer
    model = get_model(args, weights_path=args.model_a_path)
    optimizer, scheduler = get_optimizer_and_schedular(model, args)

    # create experiment dir, csv logger and criterion
    experiment_dir, experiment_id = create_experiment_dir(args)
    csv_filename = os.path.join(
        experiment_dir, f"HP_tune_" + experiment_id + f".csv")
    csv_logger = CSVLogger(
        args=args,
        fieldnames=[
            "decision",
            "mu",
            "sigma",
            "threshold",
            "epoch",
            "train_acc",
            "test_acc",
            "train_loss",
            "train_num_masked_pixel",
            "train_mean_gradcam_values",
            "train_std_gradcam_values",
        ],
        filename=csv_filename,
    )
    criterion = nn.CrossEntropyLoss()
    # cast to gpu if needed
    if args.use_cuda:
        model = model.cuda()
        criterion.cuda()

    # for a single model, the gutout model is the model that is being trained
    gutout_model = model
    training_model = model
    best_acc = -1

    

    def range_parser(range_string:str):
        temp = [float(num) for num in range_string.replace('[', '').replace(']','').split(',')]
        return np.arange(temp[0], temp[1], temp[2])

    if args.decision == "deterministic":
        threshold_range = range_parser(args.deterministic_range)
        args.gutout == True
    elif args.decision == "stochastic":
        mu_range = range_parser(args.mu_range)
        sigma_range = range_parser(args.sigma_range)
        args.gutout == True
    
    # create a gradcam
    grad_cam = BatchGradCam(
        model=gutout_model,
        feature_module=getattr(gutout_model, args.feature_module),
        target_layer_names=[args.target_layer_names],
        use_cuda=args.use_cuda,
    )


    # grid search
    if args.decision == "deterministic":
        for threshold in threshold_range:
            training_model = get_model(args, weights_path=args.model_a_path)
            if args.use_cuda:
                training_model = training_model.cuda()
            args.threshold = threshold
            print("Trying threshold ",args.threshold)
            for epoch in range(args.epochs):
                (
                    train_accuracy,
                    mean_loss,
                    mean_num_masked_pixel,
                    mean_gradcam_values,
                    mean_std_gradcam_values,
                ) = train(
                    training_model,
                    grad_cam,
                    criterion,
                    optimizer,
                    train_loader,
                    epoch,
                    args,
                    max_num_batches,
                )
                test_acc = test(training_model, test_loader, args, max_num_batches)
                tqdm.write( 
                    " test_acc: %.3f" % (test_acc))
                row = {
                    "decision": "deterministic",
                    "mu": "N/A",
                    "sigma": "N/A",
                    "threshold": round(args.threshold,3),
                    "epoch": str(epoch),
                    "train_acc": str(round(train_accuracy,3)),
                    "test_acc": str(round(test_acc,3)),
                    "train_loss": str(round(mean_loss,3)),
                    "train_num_masked_pixel": str(round(float(mean_num_masked_pixel),3)),
                    "train_mean_gradcam_values": str(round(float(mean_gradcam_values),3)),
                    "train_std_gradcam_values": str(round(float(mean_std_gradcam_values),3)),
                }
                scheduler.step()
                if epoch % args.log_interval == 0:
                    csv_logger.writerow(row)
    elif args.decision == "stochastic":
        for mu in mu_range:
            for sigma in sigma_range:
                if mu == 0.6 and sigma == 0.05:
                    continue
                training_model = get_model(args, weights_path=args.model_a_path)
                if args.use_cuda:
                    training_model = training_model.cuda()
                args.threshold = min(random.gauss(mu,sigma) ,1)
                print("mu:",mu,"sigma:",sigma)
                print("Trying threshold ", args.threshold)
                for epoch in range(args.epochs):
                    (
                        train_accuracy,
                        mean_loss,
                        mean_num_masked_pixel,
                        mean_gradcam_values,
                        mean_std_gradcam_values,
                    ) = train(
                        training_model,
                        grad_cam,
                        criterion,
                        optimizer,
                        train_loader,
                        epoch,
                        args,
                        max_num_batches,
                    )
                    test_acc = test(training_model, test_loader,
                                    args, max_num_batches)
                    tqdm.write( " test_acc: %.3f" % (test_acc))
                    row = {
                        "decision": "stochastic",
                        "mu": round(mu,3),
                        "sigma": round(sigma,3),
                        "threshold": round(args.threshold,3),
                        "epoch": str(epoch),
                        "train_acc": str(round(train_accuracy,3)),
                        "test_acc": str(round(test_acc,3)),
                        "train_loss": str(round(mean_loss,3)),
                        "train_num_masked_pixel": str(round(float(mean_num_masked_pixel),3)),
                        "train_mean_gradcam_values": str(round(float(mean_gradcam_values),3)),
                        "train_std_gradcam_values": str(round(float(mean_std_gradcam_values),3)),
                    }
                    scheduler.step()
                    if epoch % args.log_interval == 0:
                        csv_logger.writerow(row)

                    
