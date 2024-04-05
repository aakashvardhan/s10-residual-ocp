import numpy as np
import random
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

def set_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


train_losses = []
test_losses = []
train_acc = []
test_acc = []


def train(model, device, train_loader, optimizer,criterion, ocp_scheduler,epoch):
    """
    Trains the model on the training data for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        device (torch.device): The device (CPU or GPU) to be used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        epoch (int): The current epoch number.

    Returns:
        None
    """
    from tqdm import tqdm

    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)
        train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
        )
        ocp_scheduler.step()
        train_acc.append(100 * correct / processed)
        lrs = ocp_scheduler.get_last_lr()
        print(f"Max Learning Rate: {max(lrs)}")


def test(model, device, test_loader):
    """
    Function to test the model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model.
        device (torch.device): The device to run the model on.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.

    Returns:
        float: The average loss on the test dataset.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    test_acc.append(100.0 * correct / len(test_loader.dataset))

    return test_loss


# Function to plot the training and testing graphs for loss and accuracy
def plt_fig():
    """
    Plots the training and test metrics on a 2x2 grid.

    This function creates a figure with four subplots to visualize the training and test metrics.
    The subplots display the training loss, training accuracy, test loss, and test accuracy.

    Args:
        None

    Returns:
        None
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot([train_loss.cpu().detach().numpy() for train_loss in train_losses])
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc[4000:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")