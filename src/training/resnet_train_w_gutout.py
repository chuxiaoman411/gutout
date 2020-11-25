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

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.utils.data_utils import get_dataloaders
# from src.models.resnet import resnet18
from src.models.resnet_cutout import ResNet18 as resnet18

from src.gutout.gutout_utils import BatchGradCam, get_gutout_samples, gutout_images
from src.utils.cutout import Cutout
from src.utils.misc import CSVLogger

model_options = ['resnet18']
dataset_options = ['cifar10', 'cifar100', 'svhn']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='the number of workers for fetching data using the dataloaders (default: 4')
parser.add_argument('--smoke_test', type=int, default=1,
                    help='set this to 1 if debugging or to 0 if running full training session')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--use_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')

# GutOut arguments
parser.add_argument('--gutout', action='store_true', default=True, #was False
                    help='apply gutout')
parser.add_argument('--model_path', default=r'cifar10_resnet18_acc0.7985_.pth', #originally: 'cifar10_resnet18_acc0.7985_.pth'
                    help='path to the Resnet model used to generate gutout mask')

parser.add_argument('--threshold', type=float, default=0.9,
                    help='threshold for gutout')
parser.add_argument('--random_threshold', action='store_true', default=False,
                    help='whether to choose threshold randomly obeying Gaussian distribution')
parser.add_argument('--mu',type=float, default=0.9,
                    help='mu for Gaussian Distribution')
parser.add_argument('--sigma', type=float, default=0.1,
                    help='sigma for Gaussian Distribution')



def train(model, grad_cam, criterion, optimizer, train_loader, max_num_batches=None):
    model.train()
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0

    progress_bar = tqdm(train_loader)

    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        if args.use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        # conduct gutout
        if args.gutout:
            images, _ = gutout_images(grad_cam, images, args=args)

        optimizer.zero_grad()
        pred = model(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

        if max_num_batches is not None and i >= max_num_batches:
            break

    return accuracy


def test(model, test_loader, max_num_batches=None):
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
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


args = parser.parse_args()
max_num_batches = None
if args.smoke_test == 1:
    args.batch_size = 2
    args.epochs = 115 #originally 3
    max_num_batches = 2
print(args)

args.cuda = args.use_cuda
cudnn.benchmark = True  # Should make training go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.random_threshold:
    args.threshold = random.gauss(float(args.mu),float(args.sigma))
    print("Randomly generated threshold ",args.threshold)


# get dataloaders
train_loader, test_loader = get_dataloaders(args)
if args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'cifar100':
    num_classes = 100

# create model
if args.model == 'resnet18':
    model = resnet18(num_classes=num_classes)
if args.gutout:
    model_for_gutout = resnet18(num_classes=num_classes)
    # model_for_gutout.load_state_dict(torch.load(args.model_path, map_location='cpu'))


# create optimizer, loss function and schedualer
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                            momentum=0.9, nesterov=False, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)
criterion = nn.CrossEntropyLoss()

# cast to gpu if needed
if args.use_cuda:
    model = model.cuda()
    criterion.cuda()

# create csv logger
experiment_id = args.dataset + '_' + args.model
current_time = time.localtime()
current_time = time.strftime(
    "%H-%M-%S", current_time)
experiment_dir = current_time + "-experiment_" + experiment_id

os.makedirs(experiment_dir)
os.makedirs(os.path.join(experiment_dir,"checkpoints/"), exist_ok=True)
csv_filename = os.path.join(experiment_dir, experiment_id + '.csv')
csv_logger = CSVLogger(args=args, fieldnames=[
                       'epoch', 'train_acc', 'test_acc'], filename=csv_filename)

best_acc = -1
grad_cam = BatchGradCam(model=model, feature_module=model.layer1,
                        target_layer_names=["0"], use_cuda=args.use_cuda)

# run training loop
for epoch in range(args.epochs):
    train_accuracy = train(model, grad_cam, criterion,
                           optimizer, train_loader, max_num_batches)
    test_acc = test(model, test_loader, max_num_batches)
    is_best = test_acc > best_acc
    tqdm.write('test_acc: %.3f' % (test_acc))

    # scheduler.step(epoch)  # Use this line for PyTorch <1.4
    scheduler.step()     # Use this line for PyTorch >=1.4

    row = {'epoch': str(epoch), 'train_acc': str(
        train_accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)
    if is_best:
        torch.save(model.state_dict(), os.path.join(experiment_dir,'checkpoints/' + experiment_id + '_best.pth'))
    if epoch % 5 == 0 and epoch != 0:
        torch.save(model.state_dict(), os.path.join(
            experiment_dir, 'checkpoints/' + experiment_id +"_Epoch_"+ str(epoch)+"_acc"+str(test_acc)+ '_.pth'))
    get_gutout_samples(model, grad_cam, epoch, experiment_dir, args)

csv_logger.close()
