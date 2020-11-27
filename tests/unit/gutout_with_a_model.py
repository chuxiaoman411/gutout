import sys
import os
import torch
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.models.resnet import resnet18
from src.gutout.gutout_utils import BatchGradCam, get_gutout_samples, gutout_images

num_classes = 10
model = resnet18(num_classes=num_classes)

path = "run/checkpoints/cifar10_resnet18_Epoch_45acc0.7484_.pth"
model.load_state_dict(torch.load(path, map_location="cpu"))

grad_cam = BatchGradCam(
    model=model, feature_module=model.layer3, target_layer_names=["0"], use_cuda=False
)

img_path = "sample_imgs_cifar10/plane.png"
img = cv2.imread(img_path, 1)
img = np.float32(cv2.resize(img, (32, 32)))
img = np.expand_dims(img, 0)
img = torch.from_numpy(img).permute(0, 3, 1, 2)

mask = grad_cam(img).numpy()

print(mask)
gutout_mask = mask <= 0.9
img = img.numpy()
img = np.squeeze(img, axis=0)
gutout_mask = np.squeeze(gutout_mask, axis=0)
img *= gutout_mask

img = np.transpose(img, (1, 2, 0))

cv2.imwrite("showing.png", img)
