import torch
import numpy as np
from .cutout import Cutout
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_datasets(args, need_validate = False):
    if args.dataset == "svhn":
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
            std=[x / 255.0 for x in [50.1, 50.6, 50.8]],
        )
    else:
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

    train_transform = transforms.Compose([])
    if args.data_augmentation:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())

    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    if args.cutout:
        train_transform.transforms.append(
            Cutout(n_holes=args.n_holes, length=args.length)
        )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    if args.dataset == "cifar10":
        num_classes = 10
        train_dataset = datasets.CIFAR10(
            root="data/", train=True, transform=train_transform, download=True
        )

        test_dataset = datasets.CIFAR10(
            root="data/", train=False, transform=test_transform, download=True
        )

        if need_validate:
            validate_dataset = datasets.CIFAR10(
                root="data/", train=True, transform=train_transform, download=True
            )

    elif args.dataset == "cifar100":
        num_classes = 100
        train_dataset = datasets.CIFAR100(
            root="data/", train=True, transform=train_transform, download=True
        )

        test_dataset = datasets.CIFAR100(
            root="data/", train=False, transform=test_transform, download=True
        )

        if need_validate:
            validate_dataset = datasets.CIFAR100(
                root="data/", train=True, transform=train_transform, download=True
            )

    elif args.dataset == "svhn":
        num_classes = 10
        train_dataset = datasets.SVHN(
            root="data/", split="train", transform=train_transform, download=True
        )

        extra_dataset = datasets.SVHN(
            root="data/", split="extra", transform=train_transform, download=True
        )

        # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
        data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
        labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
        train_dataset.data = data
        train_dataset.labels = labels

        test_dataset = datasets.SVHN(
            root="data/", split="test", transform=test_transform, download=True
        )
    if need_validate:
        return train_dataset, validate_dataset, test_dataset

    return train_dataset, test_dataset


def get_dataloaders(args, need_validate=False, validate_proportion=0.8):
    if need_validate:
        train_dataset, validate_dataset, test_dataset = get_datasets(args,need_validate)
    else:
        train_dataset, test_dataset = get_datasets(args)

    if need_validate:
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(validate_proportion * num_train))
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=validate_dataset, batch_size=args.batch_size, sampler=valid_sampler,
            num_workers=args.num_workers
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
        )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    if need_validate:
        return train_loader,valid_loader,test_loader
    return train_loader, test_loader
