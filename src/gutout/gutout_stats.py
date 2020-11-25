import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.models.resnet import resnet18
import torch
from tqdm import tqdm
from gutout_utils import GradCam, gutout_images, show_cam_on_image
from src.utils.data_utils import get_dataloaders
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


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
parser.add_argument('--gutout', action='store_true', default=False,
                    help='apply gutout')
# parser.add_argument('--model_path', default=r'checkpoints/model.pt',
#                     help='path to the Resnet model used to generate gutout mask')

parser.add_argument('--threshold', type=float, default=0.9,
                    help='threshold for gutout')
parser.add_argument('--random_threshold', action='store_true', default=False,
                    help='whether to choose threshold randomly obeying Gaussian distribution')
parser.add_argument('--mu', type=float, default=0.9,
                    help='mu for Gaussian Distribution')
parser.add_argument('--sigma', type=float, default=0.1,
                    help='sigma for Gaussian Distribution')

# Joint training arguments
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train each network')
parser.add_argument('--switch_interval', type=int, default=2,
                    help='frequency of switching between the training model and the gutout model')
args = parser.parse_args()

model_b = resnet18(num_classes=10) #generalize num_classes later
model_b.load_state_dict(torch.load('./../training/cifar10_resnet18_acc0.7985_.pth',map_location='cpu'))

grad_cam = GradCam(model=model_b, feature_module=model_b.layer1,
                        target_layer_names=["0"], use_cuda=False) #last argument was: use_cuda=args.use_cuda

train_loader, test_loader = get_dataloaders(args)
progress_bar = tqdm(train_loader)

def show_images(images):
    n = len(images)
    f = plt.figure()
    axes = []
    for i in range(n):
        ax = f.add_subplot(1, n, i + 1)
        axes.append(ax)
        plt.imshow(images[i])

    axes[0].set_title("Original image")
    axes[1].set_title("Grad-cam on image")
    axes[2].set_title("GutOut on image")
    plt.show(block=True)

for i, (images, labels) in enumerate(progress_bar): #shape of images: [128, 3, 32, 32]
    #progress_bar.set_description('Epoch ' + str(epoch))
    print("type of images", type(images))
    print("shape of images", images.size())
    target_index = None
    first_img = images[0,:,:,:]
    #first_img = np.transpose(first_img, [1,2,0])
    print("min", torch.min(first_img))
    print("max", torch.max(first_img))
    #first_img_as_batch = torch.unsqueeze(first_img, 0)
    first_img_as_batch = first_img.unsqueeze(0)
    #first_img = np.float32(cv2.resize(first_img, (224, 224))) #/ 255
    print("first_img", first_img)
    mask = grad_cam(first_img_as_batch)
    first_img = np.transpose(first_img, (1,2,0))
    #first_mask = mask[0,:,:,:]
    #print("shape of mask", mask.size())
    cam_on_image = show_cam_on_image(first_img, mask)
    gutout_imgs, _ = gutout_images(grad_cam, first_img_as_batch, args)
    gutout_img = gutout_imgs.squeeze(0)
    gutout_img = np.transpose(gutout_img, (1,2,0))
    print("shape of first img", first_img.size())
    print("shape of cam on image", cam_on_image.shape)
    print("shape of gut out image", gutout_img.shape)
    show_images([first_img, cam_on_image, gutout_img])
    input("pause")
