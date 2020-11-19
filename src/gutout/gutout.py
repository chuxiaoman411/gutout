import torch
import numpy as np

from src.models.resnet import resnet18
from .gutout_utilities import FeatureExtractor, ModelOutputs, GradCam

class Gutout(object):

    def __init__(self, model_path, model_num_classes, threshold, use_cuda):
        self.model = ResNet18(model_num_classes)
        # self.model.load_state_dict(torch.load(model_path))
        self.threshold = threshold
        self.use_cuda = use_cuda
        
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            
        """
        print("gutting out")
        grad_cam = GradCam(model=self.model, feature_module=self.model.layer4, \
                       target_layer_names=["2"], use_cuda=self.use_cuda)
        print("grad cam ready")
        target_index = None
        img = torch.unsqueeze(img,0)
        mask = grad_cam(img, target_index)
        print("mask ready")
        gutout_mask = generate_gutout_mask(self.threshold,mask)
        img_after_gutout = apply_gutout_mask(img,gutout_mask)

        return img_after_gutout
