from models.custom_resnet import CustomResNet
from models.model_utils import model_summary
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.optim as optim
import torch
import argparse
from setup_cifar10 import CIFAR10Dataset
from config import get_config


def test_model_sanity(model_):
    """
    Function to perform a sanity check on a model by training it on a small dataset and checking if the loss decreases.

    Args:
        model_ (torch.nn.Module): The model to be tested.

    Returns:
        None
    """
    from tqdm import tqdm

    # Sanity check for model
    # Check if the model is capable of overfitting on a small dataset
    # Load CIFAR10 dataset
    cifar_train = CIFAR10Dataset(
        root="./data", train=True, transform="test", download=True
    )
    cifar_subset = Subset(cifar_train, range(100))

    # Set the seed
    torch.manual_seed(1)

    # Create model
    model = model_
    loss_function = F.nll_loss
    # Using SGD as the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Create data loader
    train_loader = DataLoader(cifar_subset, batch_size=10, shuffle=True)

    # Train the model on the small subset
    # Calculate initial loss
    model.eval()  # Set the model to evaluation mode for initial loss calculation
    with torch.no_grad():
        data, target = next(iter(train_loader))
        initial_loss = loss_function(model(data), target).item()

    # Train the model on the small subset
    model.train()  # Set the model back to train mode
    for epoch in range(1, 5):  # Running for 4 epochs just for testing
        print(f"Epoch {epoch}")
        pbar = tqdm(train_loader)

        for batch_idx, (data, target) in enumerate(pbar):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f} Batch_id={batch_idx}")

    # Perform a sanity check: the loss should decrease after training
    model.eval()  # Set the model to evaluation mode for final loss calculation
    with torch.no_grad():
        data, target = next(iter(train_loader))
        final_loss = loss_function(model(data), target).item()

    print("Initial loss:", initial_loss)
    print("Final loss:", final_loss)

    assert (
        final_loss < initial_loss
    ), "Sanity check failed: Loss did not decrease after training."

    print(
        "Sanity check passed: Model is capable of overfitting to a small subset of the data."
    )


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Model Testing")
    parser.add_argument("--summary", action="store_true", help="Print model summary")
    parser.add_argument(
        "--sanity", action="store_true", help="Perform model sanity check"
    )
    parser.add_argument(
        "--shapes", action="store_true", help="Print shapes of all layers"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create
    config = get_config()
    config["debug"] = True
    config["n_channels"] = 38
    model = CustomResNet(config)

    if args.summary:
        model_summary(model, input_size=(3, 32, 32))

    if args.sanity:
        test_model_sanity(model)

    if args.shapes:
        # print shapes of all layers (excluding bias)
        for name, param in model.named_parameters():
            print(name, param.shape)
