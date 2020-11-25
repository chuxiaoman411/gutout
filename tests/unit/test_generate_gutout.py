import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.gutout.generate_gutout_mask import (
    get_args,
    GradCam,
    preprocess_image,
    show_cam_on_image,
    apply_gutout_mask,
    show_images,
    generate_gutout_mask,
)


def test_generate_gutout():
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    if args.image_path is None:
        args.image_path = r"tests\unit\blueno.jpeg"
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model = models.resnet18(pretrained=True)
    grad_cam = GradCam(
        model=model,
        feature_module=model.layer4,
        target_layer_names=["1"],
        use_cuda=args.use_cuda,
    )

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)

    cam_on_image = show_cam_on_image(img, mask)

    gutout_mask = generate_gutout_mask(0.7, mask)
    img_after_gutout = apply_gutout_mask(img, gutout_mask)

    show_images([img, cam_on_image, img_after_gutout])


if __name__ == "__main__":
    test_generate_gutout()
