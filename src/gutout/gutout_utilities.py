from resnet import ResNet18

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        print("calling feature extractor in FeatureExtractor")
        print("x shape",x.shape)
        outputs = []
        self.gradients = []
        # print("self.model._modules.items()",self.model._modules.items())
        for name, module in self.model._modules.items():
            print("name ",name)
            print("module ",module)
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
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
            print("name",name)
            print("module",module)
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                print("case start")
                print("x shape is",x.shape)
                x = module(x)
                print("case end")
        print("for loop ends")
        return target_activations, x

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):

        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        print("features output ready")
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

def generate_gutout_mask(threshold,mask):
    """threshold is a percentage, pixels with attention larger than threshold*max(atttention) will be masked"""
    attention_threshold = np.min(mask) + threshold * (np.max(mask)-np.min(mask))
    gutout_mask=np.less(mask,attention_threshold).astype(np.float32)
    return gutout_mask

def apply_gutout_mask(image,mask):
    print("Apply image shape",image.shape)
    print("Apply mask shape",mask.shape)
    return image * np.repeat(np.expand_dims(mask,-1),3,-1)

# import torch
# model = ResNet18()

# # grad_cam = GradCam(model=model, feature_module=model.layer4, \
# #                        target_layer_names=["2"], use_cuda=False)

# # print(grad_cam)

# torch.save(model.state_dict(),'model.pt')

# model2 = ResNet18()
# model2.load_state_dict(torch.load('model.pt'))
# print(model2)
