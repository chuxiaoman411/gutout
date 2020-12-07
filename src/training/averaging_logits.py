from torchvision import datasets, transforms
from torchvision import datasets
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.models.resnet_cutout import ResNet18 as cutout_resnet18

modelA = cutout_resnet18(10)
modelB = cutout_resnet18(10)

modelA.load_state_dict(torch.load("./ensemble/cifar10_cutout_resnet18_a.pth", map_location="cpu"))
modelB.load_state_dict(torch.load("./ensemble/cifar10_cutout_resnet18_b.pth", map_location="cpu"))

# sdA = modelA.state_dict()
# sdB = modelB.state_dict()
# for key in sdA:
#     sdB[key] = (sdB[key]+sdA[key])/2
# modelC = cutout_resnet18(10)
# modelC.load_state_dict(sdB)


normalize = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
)
test_transform = transforms.Compose([transforms.ToTensor(), normalize])
test_dataset = datasets.CIFAR10(
    root="data/", train=False, transform=test_transform, download=True
)
# test_dataset = datasets.CIFAR100(
#     root="data/", train=False, transform=test_transform, download=True
# )
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=128,
    shuffle=False,
    pin_memory=True
)


def test(models, test_loader):
    for model in models:
        model.eval()
    correct = 0.0
    total = 0.0
    i = 0
    total_batch = 10000//128
    for images, labels in test_loader:
        if i %10 == 0:
            print("batch ",i,"/",total_batch)
        with torch.no_grad():
            pred = models[0](images)
            for j in range(1,len(models)):
                pred += models[j](images)
        pred /= len(models)
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        i += 1
    val_acc = correct / total
    return val_acc

acc = test([modelA,modelB],test_loader)
print("Accuracy:",acc)
