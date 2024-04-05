import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
import torch


class CIFAR10Dataset(datasets.CIFAR10):
    """
    Custom dataset class for CIFAR-10 dataset.

    Args:
        root (str): Root directory where the dataset exists or will be saved.
        train (bool): If True, creates a dataset from the training set, otherwise from the test set.
        download (bool): If True, downloads the dataset from the internet and puts it in the root directory.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
            Default: None.

    Attributes:
        transform (callable): A function/transform that takes in an PIL image and returns a transformed version.

    """

    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        if transform == "train":
            self.transform = A.Compose(
                [
                    A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                    A.HorizontalFlip(p=0.5),
                    A.CoarseDropout(
                        max_holes=1,
                        max_height=8,
                        max_width=8,
                        min_holes=1,
                        min_height=8,
                        min_width=8,
                        fill_value=(0.4914, 0.4822, 0.4465),
                        mask_fill_value=None,
                    ),
                    A.RandomBrightnessContrast(p=0.2),
                    A.CenterCrop(32, 32, always_apply=True),
                    A.Normalize(
                        mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)
                    ),
                    ToTensorV2(),
                ]
            )
        elif transform == "test":
            self.transform = A.Compose(
                [
                    A.Normalize(
                        mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = transform

    def __getitem__(self, index):
        """
        Retrieves the image and label at the given index.

        Args:
            index (int): Index of the image.

        Returns:
            tuple: A tuple containing the transformed image and its corresponding label.

        """
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def setup_cifar10(config):
    """
    Sets up the CIFAR-10 dataset and data loaders.

    Args:
        config (dict): A dictionary containing configuration parameters.

    Returns:
        tuple: A tuple containing the CIFAR-10 train dataset, CIFAR-10 test dataset,
               CIFAR-10 train data loader, and CIFAR-10 test data loader.
    """
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(1)

    if cuda:
        torch.cuda.manual_seed(1)
    train_data = CIFAR10Dataset(
        root="./data", train=True, download=True, transform="train"
    )
    test_data = CIFAR10Dataset(
        root="./data", train=False, download=True, transform="test"
    )

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = (
        dict(
            shuffle=True,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=True,
        )
        if cuda
        else dict(shuffle=True, batch_size=64)
    )
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)

    return train_data, test_data, train_loader, test_loader