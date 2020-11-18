# run resnet_train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16
# run resnet_train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8
# run resnet_train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160 --cutout --length 20

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

from misc import CSVLogger
from cutout import Cutout
from gutout import Gutout

from resnet import ResNet18
# from model.wide_resnet import WideResNet

# model_options = ['resnet18', 'wideresnet']
model_options = ['resnet18']
dataset_options = ['cifar10', 'cifar100', 'svhn']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
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
parser.add_argument('--use-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')

##GutOut arguments
parser.add_argument('--gutout', action='store_true', default=True,
                    help='apply gutout')
# parser.add_argument('--model_path', default='./model.pt',
#                     help='path to the Resnet model used to generate gutout mask')      
parser.add_argument('--model_path', default=r'gutout\basic_scripts\model.pt',
                    help='path to the Resnet model used to generate gutout mask')          
parser.add_argument('--threshold', type=float, default=0.9,
                    help='threshold for gutout')   
                     
args = parser.parse_args()
args.cuda = args.use_cuda
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_id = args.dataset + '_' + args.model

print(args)

# Image Preprocessing
if args.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
else:
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)

if args.cutout:
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
##Gutout
if args.gutout:
    if args.dataset == 'cifar10':
        Gutout = Gutout(model_path=args.model_path,model_num_classes=10, threshold=args.threshold, use_cuda=args.use_cuda)
    elif args.dataset == 'cifar100':
        Gutout = Gutout(model_path=args.model_path,model_num_classes=100, threshold=args.threshold, use_cuda=args.use_cuda)
    # train_transform.transforms.append(Gutout)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

print("Start preparing dataset")
if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='data/',
                                      train=True,
                                      transform=train_transform,
                                      download=True)

    test_dataset = datasets.CIFAR100(root='data/',
                                     train=False,
                                     transform=test_transform,
                                     download=True)
print("Prepared dataset:",train_dataset)
print("Dataset Prepared")
# elif args.dataset == 'svhn':
#     num_classes = 10
#     train_dataset = datasets.SVHN(root='data/',
#                                   split='train',
#                                   transform=train_transform,
#                                   download=True)

#     extra_dataset = datasets.SVHN(root='data/',
#                                   split='extra',
#                                   transform=train_transform,
#                                   download=True)

#     # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
#     data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
#     labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
#     train_dataset.data = data
#     train_dataset.labels = labels

#     test_dataset = datasets.SVHN(root='data/',
#                                  split='test',
#                                  transform=test_transform,
#                                  download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=0)

if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
# elif args.model == 'wideresnet':
#     if args.dataset == 'svhn':
#         cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
#                          dropRate=0.4)
#     else:
#         cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
#                          dropRate=0.3)

if args.use_cuda:
    cnn = cnn.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
else:
    cnn = cnn
    criterion = nn.CrossEntropyLoss()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

if args.dataset == 'svhn':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
else:
    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

filename = test_id + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        if args.use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc


for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0

    progress_bar = tqdm(train_loader)

    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        if args.use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        cnn_optimizer.zero_grad()
        pred = cnn(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

        i += 1

    test_acc = test(test_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))

    #scheduler.step(epoch)  # Use this line for PyTorch <1.4
    scheduler.step()     # Use this line for PyTorch >=1.4

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
csv_logger.close()
