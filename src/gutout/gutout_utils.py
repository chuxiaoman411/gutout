import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models
import matplotlib.pyplot as plt
import os
import sys
from torchvision import transforms


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(
            self.model, self.feature_module, target_layer_names
        )

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class BatchGradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(
            self.model, self.feature_module, target_layer_names
        )

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            # TODO perhaps try different labels (such as worst predictions)
            predicted_labes = torch.max(output.data, 1)[1]

        # create one hot vector to only use the specific
        one_hot = torch.zeros_like(output)
        for i, label in enumerate(predicted_labes):
            one_hot[i, label] = 1
        one_hot.requires_grad_(True)

        # prepare for backward
        self.feature_module.zero_grad()
        self.model.zero_grad()

        # extract gradients and features
        proxy_loss = (
            torch.sum(one_hot.cuda() * output)
            if self.cuda
            else torch.sum(one_hot * output)
        )
        proxy_loss.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1]
        target = features[-1]

        # calculate grad cam
        weights = torch.mean(grads_val, (2, 3), keepdim=True)
        grad_weighted_cam = torch.sum(target * weights, 1, keepdim=True)

        # clip (relu)
        grad_weighted_cam = torch.clamp(grad_weighted_cam, 0)

        # resize
        grad_weighted_cam = F.interpolate(grad_weighted_cam, input.shape[2:])

        # batch normalize heat map - to 0-1
        grad_weighted_cam = grad_weighted_cam / torch.max(grad_weighted_cam)

        # remove gradients from grad cam
        grad_weighted_cam = grad_weighted_cam.detach()

        return grad_weighted_cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input), input, positive_mask
        )
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(
                torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1
            ),
            positive_mask_2,
        )

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == "ReLU":
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=False,
        help="Use NVIDIA GPU acceleration",
    )
    parser.add_argument("--image-path", type=str, help="Input image path")
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def generate_gutout_mask(threshold, mask):

    gutout_mask = np.less(mask, threshold).astype(np.float32)
    return gutout_mask


def generate_batch_gutout_mask(threshold, masks, filter_out="greater_than_thresh"): #used to be "greater_than_thresh"
    if filter_out == "greater_than_thresh":
        gutout_mask = (masks <= threshold).float()
    elif filter_out == "less_than_thresh":
        gutout_mask = (masks >= threshold).float()
    else:
        raise ValueError(
            "recieved unsuppored value for 'filter_out' entry, allowed values are: [greater_than_thresh, less_than_thresh]"
        )

    return gutout_mask


def apply_batch_gutout_mask(images, masks, args):
    if args.use_cuda:
        images = images.cuda()
        masks = masks.cuda()
    return images * masks

def gutout_images(grad_cam, images, args, report_stats=False):
    img_only_for_heatmaps = preprocess_image_for_heatmap(images[0,:,:,:]) #newly added
    if args.print_output > 0:
        print("shape of imgs only for heatmaps", img_only_for_heatmaps.shape)
    img_only_for_heatmaps = img_only_for_heatmaps.permute(0, 3, 1, 2) #this is done to reverse the transposition occurred in preprocess_image_for_heatmap
    #print("IMG_ONLY_FOR_HEATMAPS", img_only_for_heatmaps)
    #print("IMAGE", images[0,:,:,:])
    heatmap_masks = grad_cam(img_only_for_heatmaps) #newly added
    #with np.printoptions(threshold=sys.maxsize):
        #print("heatmap_masks[0,:,:]", heatmap_masks[0,:,:])
    masks = grad_cam(images)
    if args.print_output > 0:
        print("type of images", type(images))
        print("size of images", images.size())
        print("type of masks", type(masks))
        print("size of masks", masks.size())
        print("masks", masks)
        with np.printoptions(threshold=sys.maxsize):
            print("masks[0,:,:]", masks[0,:,:])
    cam = show_cam_on_image(img_only_for_heatmaps, heatmap_masks) #newly added
    gutout_masks = generate_batch_gutout_mask(args.threshold, masks)
    avg_num_masked_pixel = np.sum(gutout_masks.clone().cpu().detach().numpy() == 0) / gutout_masks.shape[0]
    img_after_gutout = apply_batch_gutout_mask(images, gutout_masks, args)

    avg_gradcam_values = masks.mean()
    std_gradcam_values = masks.std()

    if args.print_output > 0:
        print("image", images[0,:,:,:])
        print("shape of original", images[0,:,:,:].shape)
        print("shape of cam", cam.shape)
        print("shape of img after gutout", img_after_gutout[0,:,:,:].shape)
        show_images([images[0,:,:,:], cam, img_after_gutout[0,:,:,:]], stats=
        (avg_num_masked_pixel,
        avg_gradcam_values,
        std_gradcam_values))
    if report_stats==True:
        return (
            img_after_gutout,
            avg_num_masked_pixel,
            avg_gradcam_values,
            std_gradcam_values,
            cam
        )
    return (
        img_after_gutout,
        avg_num_masked_pixel,
        avg_gradcam_values,
        std_gradcam_values,
    )


def get_gutout_samples(model, grad_cam, epoch, experiment_dir, args):
    if args.dataset == "cifar10":
        path = "../sample_data/sample_imgs_cifar10"
    elif args.dataset == "cifar100":
        path = "../sample_data/sample_imgs_cifar100"

    # grad_cam = BatchGradCam(model=model, feature_module=model.layer3,
    #                         target_layer_names=["0"], use_cuda=args.use_cuda)
    mean= np.asarray([x for x in [125.3, 123.0, 113.9]])
    std= np.asarray([x for x in [63.0, 62.1, 66.7]])
    normalize = transforms.Normalize(
        mean=mean, # got rid of x / "255.0"
        std=std # got rid of x / "255.0"
    )
    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))
    images = []
    names = []
    #print("looking for pictures", os.listdir(path))
    for f in os.listdir(path):
        if "png" in f or "jpeg" in f: #newly added: "or jpeg in f"
            names.append(f)
            img = cv2.imread(os.path.join(path, f), 1)
            img = np.float32(cv2.resize(img, (32, 32)))
            img = np.expand_dims(img, 0)
            images.append(img)

    images = np.concatenate(images, 0)

    if args.print_output > 0:
        print("image before normalize", images[0,:,:,:])
    images = images[:,:,:,::-1].copy() #newly added
    images = torch.from_numpy(images).permute(0, 3, 1, 2)
    images = normalize(images)
    if args.print_output > 0:
        print("image after normalize", images[0,:,:,:])
    # if args.use_cuda:
    #     img_after_gutout, avg_num_masked_pixel = gutout_images(grad_cam, images, args)
    #     img_after_gutout = img_after_gutout.cpu().numpy()
    # else:
    (
        img_after_gutout,
        avg_num_masked_pixel,
        avg_gradcam_values,
        std_gradcam_values,
        cam #newly added
    ) = gutout_images(grad_cam, images, args, report_stats=True) #used to not include "report_stats=True"
    img_after_gutout = img_after_gutout.cpu().numpy()

    print(
        "Average number of pixels per image get gutout during sampling:",
        avg_num_masked_pixel,
    )
    for i in range(len(names)):
        fn = "Epoch-" + str(epoch) + "-" + names[i]
        path = os.path.join(experiment_dir, fn)
        img = img_after_gutout[i]
        img = torch.from_numpy(img) #newly added
        img = denormalize(img) #newly added
        img = img.detach().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img[:,:,::-1] #newly added
        cv2.imwrite(path, img)
        print("i: ", i)
        print("img numbers", img)
    cam = np.transpose(cam, (1,2,0)) #newly added----currently cam for the first image only; will rework to include the whole set
    path = os.path.join(experiment_dir, "cam" + str(epoch) + ".jpeg")
    cv2.imwrite(path, cam)          #newly added

# this is preprocessing just for heatmap! temp use
def preprocess_image_for_heatmap(img):
    #means = [0.485, 0.456, 0.406]
    #stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.numpy().copy()[:, :, :] #the last index used to be '::-1'
    #for i in range(3):
        #preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        #preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    #print("shape of preprocessed img", preprocessed_img.shape)
    #preprocessed_img = \
    #    np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    #preprocessed_img = np.ascontiguousarray(preprocessed_img) #newly added
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (1, 2, 0)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def show_cam_on_image(img, mask):
    #print("shape of img", img.shape)
    #print("shape of mask", mask.shape)
    mask = mask.squeeze() #newly added
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = heatmap[:,:,::-1]
    heatmap = np.float32(heatmap) / 255
    heatmap = np.transpose(heatmap, (2,0,1)) #newly added
    img = img.squeeze() #newly added
    #print("shape of heatmap before np", heatmap.shape)
    #cam = heatmap + np.float32(img)
    cam = heatmap #+ img.detach().numpy()
    #print("shape of heatmap after np", heatmap.shape)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    #with np.printoptions(threshold=sys.maxsize):
        #print("cam", cam)

    return cam

def show_images(images, epoch=None, stats=None):
    n = len(images)
    f = plt.figure()
    if epoch:
        f.suptitle('epoch:' +str(epoch), fontsize=14, fontweight='bold')
    axes = []
    for i in range(n):
        ax = f.add_subplot(1, n, i + 1)
        axes.append(ax)
        plt.imshow(np.transpose(images[i], (1,2,0)))#used to be just "images[i]" as a parameter
    if stats:
        (avg_num_masked_pixel,avg_gradcam_values,std_gradcam_values) = stats
        axes[0].text(28, 46, 'average number of masked pixel: '+str(avg_num_masked_pixel), fontsize=10)
        axes[0].text(28, 52, 'average gradcam values: '+str(avg_gradcam_values.detach().numpy()), fontsize=10)
        axes[0].text(28, 58, 'std gradcam values: '+str(std_gradcam_values.detach().numpy()), fontsize=10)

    axes[0].set_title("Original image")
    axes[1].set_title("Grad-cam on image")
    axes[2].set_title("GutOut on image")
    plt.show(block=True)
