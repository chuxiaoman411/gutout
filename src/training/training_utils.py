import pdb
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import make_grid
from torchvision import datasets, transforms
import os
import sys
import random
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.utils.misc import CSVLogger
from src.utils.cutout import Cutout
from src.gutout.gutout_utils import BatchGradCam, get_gutout_samples, gutout_images
from src.models.resnet_cutout import ResNet18 as cutout_resnet18
from src.models.resnet_torchvision import resnet18 as torchvision_resnet18
from src.utils.data_utils import get_dataloaders

# python src/training/train_twin_nets_jointly.py --use_cuda --smoke_test 0 --model cutout_resnet18 --epochs 200 --num_workers 4

def get_args():

    model_options = ["torchvision_resnet18", "cutout_resnet18"]
    print(f"model options = {model_options}")
    dataset_options = ["cifar10", "cifar100", "svhn"]

    parser = argparse.ArgumentParser(description="CNN")
    parser.add_argument("--dataset", "-d", default="cifar10", choices=dataset_options)
    parser.add_argument("--model", "-a", default="torchvision_resnet18", choices=model_options)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="the number of workers for fetching data using the dataloaders (default: 4",
    )
    parser.add_argument(
        "--smoke_test",
        type=int,
        default=1,
        help="set this to 1 if debugging or to 0 if running full training session",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="learning rate"
    )
    parser.add_argument(
        "--data_augmentation",
        action="store_true",
        default=False,
        help="augment data by flipping and cropping",
    )
    parser.add_argument(
        "--model_a_path",
        default="",
        help="path to the Resnet model used to generate gutout mask",
    )
    parser.add_argument(
        "--model_b_path",
        default="",
        help="path to the Resnet model used to generate gutout mask",
    )

    parser.add_argument("--length", type=int, default=16, help="length of the holes")
    parser.add_argument(
        "--use_cuda", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 1)")

    # cutout arguments
    parser.add_argument(
        "--cutout", action="store_true", default=False, help="apply cutout"
    )
    parser.add_argument(
        "--n_holes", type=int, default=1, help="number of holes to cut out from image"
    )

    # GutOut arguments
    parser.add_argument(
        "--gutout", action="store_true", default=True, help="apply gutout"
    )
    parser.add_argument(
        "--img_size", type=int, default=32, help="the size of the input images"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.9, help="threshold for gutout"
    )
    parser.add_argument(
        "--random_threshold",
        action="store_true",
        default=False,
        help="whether to choose threshold randomly obeying Gaussian distribution",
    )
    parser.add_argument(
        "--mu", type=float, default=0.9, help="mu for Gaussian Distribution"
    )
    parser.add_argument(
        "--sigma", type=float, default=0.1, help="sigma for Gaussian Distribution"
    )

    # gradcam args
    parser.add_argument(
        "--feature_module",
        type=str,
        default="layer1",
        help="the resnet block from which to take the gradCAM",
    )
    parser.add_argument(
        "--target_layer_names",
        type=str,
        default="0",
        help="the layer of the selected resnet block from which to take the gradCAM",
    )

    # Joint training arguments
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of epochs to train each network"
    )
    parser.add_argument(
        "--switch_interval",
        type=int,
        default=2,
        help="frequency of switching between the training model and the gutout model",
    )

    args = parser.parse_args()
    max_num_batches = None
    args.cuda = args.use_cuda
    cudnn.benchmark = True  # Should make training go faster for large models
    torch.manual_seed(args.seed)

    if args.smoke_test:
        args.batch_size = 2
        args.epochs = 6
        max_num_batches = 2

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.random_threshold:
        args.threshold = random.gauss(float(args.mu), float(args.sigma))
        print("Randomly generated threshold ", args.threshold)

    print(args)

    return args, max_num_batches


def get_optimizer_and_schedular(model, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
    )
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    return optimizer, scheduler


def get_model(args, weights_path=""):
    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100

    if args.model == "torchvision_resnet18":
        model = torchvision_resnet18(num_classes=num_classes)
    elif args.model == "cutout_resnet18":
        model = cutout_resnet18(num_classes=num_classes)
    else:
        raise ValueError("got invalid model type, allowed models are ['out_resnet18', 'cutout_resnet18']")

    if os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"loaded weights from {weights_path}")
    else:
        print(f"didn't load weights into model, got path: {weights_path}")

    return model


def create_experiment_dir(args):
    experiment_id = args.dataset + "_" + args.model
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y__%H-%M-%S")

    experiment_dir = dt_string + "_experiment_" + experiment_id

    os.makedirs(experiment_dir)
    os.makedirs(os.path.join(experiment_dir, "checkpoints/"), exist_ok=True)
    return experiment_dir, experiment_id


def get_csv_logger(experiment_dir, experiment_id, args, model_flag="a"):
    csv_filename = os.path.join(experiment_dir, experiment_id + f"_{model_flag}.csv")
    csv_logger = CSVLogger(
        args=args,
        fieldnames=[
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
    return csv_logger


def train(
    model,
    grad_cam,
    criterion,
    optimizer,
    train_loader,
    epoch,
    args,
    max_num_batches=None,
):
    model.train()
    xentropy_loss_sum = 0.0
    avg_num_masked_pixel_sum = 0.0
    avg_gradcam_values_sum = 0.0
    std_gradcam_values_sum = 0.0
    correct = 0.0
    total = 0

    progress_bar = tqdm(train_loader)

    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description("Epoch " + str(epoch))

        if args.use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        # conduct gutout
        if args.gutout:
            (
                images,
                avg_num_masked_pixel,
                avg_gradcam_values,
                std_gradcam_values,
            ) = gutout_images(grad_cam, images, args=args)

        optimizer.zero_grad()
        pred = model(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        optimizer.step()

        xentropy_loss_sum += xentropy_loss.item()
        if args.gutout:
            avg_num_masked_pixel_sum += avg_num_masked_pixel
            avg_gradcam_values_sum += avg_gradcam_values
            std_gradcam_values_sum += std_gradcam_values

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()

        accuracy = correct / total
        mean_loss = xentropy_loss_sum / (i + 1)
        
        mean_num_masked_pixel = avg_num_masked_pixel_sum / (i + 1)
        mean_gradcam_values = avg_gradcam_values_sum / (i + 1)
        mean_std_gradcam_values = std_gradcam_values_sum / (i + 1)

        progress_bar.set_postfix(
            xentropy="%.3f" % (mean_loss),
            acc="%.3f" % (accuracy),
            mean_num_masked_pixel="%.3f" % (mean_num_masked_pixel),
            mean_gradcam_values="%.3f" % (mean_gradcam_values),
            mean_std_gradcam_values="%.3f" % (mean_std_gradcam_values),
        )

        if max_num_batches is not None and i >= max_num_batches:
            break

    return (
        accuracy,
        mean_loss,
        mean_num_masked_pixel,
        mean_gradcam_values,
        mean_std_gradcam_values,
    )


def test(model, test_loader, args, max_num_batches=None):
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.0
    total = 0.0
    i = 0
    for images, labels in test_loader:
        if args.use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        i += 1

        if max_num_batches is not None and i >= max_num_batches:
            break
    val_acc = correct / total
    return val_acc


def test_joint(model_a, model_b, test_loader, args, max_num_batches=None):
    model_a.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    model_b.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    correct = 0.0
    total = 0.0
    i = 0
    for images, labels in test_loader:
        if args.use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            pred_a = model_a(images)
            pred_b = model_b(images)

            pred_a = torch.softmax(pred_a, 1)
            pred_b = torch.softmax(pred_b, 1)
            pred = pred_a + pred_b

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        i += 1

        if max_num_batches is not None and i >= max_num_batches:
            break
    val_acc = correct / total
    return val_acc


def run_epoch(
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
):

    # run train epoch
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

    # run test epoch
    test_acc = test(training_model, test_loader, args, max_num_batches)

    # write row
    tqdm.write(model_flag + " test_acc: %.3f" % (test_acc))
    # row = {'epoch': str(epoch), 'train_acc': str(train_accuracy), 'test_acc': str(test_acc)}
    row = {
        "epoch": str(epoch),
        "train_acc": str(train_accuracy),
        "test_acc": str(test_acc),
        "train_loss": str(mean_loss),
        "train_num_masked_pixel": str(mean_num_masked_pixel),
        "train_mean_gradcam_values": str(mean_gradcam_values),
        "train_std_gradcam_values": str(mean_std_gradcam_values),
    }

    # step in schedualer, logger, and save checkpoint if needed
    scheduler.step()
    csv_logger.writerow(row)
    is_best = test_acc > best_acc
    if is_best:
        torch.save(
            training_model.state_dict(),
            os.path.join(
                experiment_dir, "checkpoints/" + experiment_id + f"_{model_flag}.pth"
            ),
        )
        best_acc = test_acc

    # save images
    get_gutout_samples(training_model, grad_cam, epoch, experiment_dir, args)

    return best_acc
