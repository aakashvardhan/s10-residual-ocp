from models.custom_resnet import CustomResNet
from models.model_utils import model_summary, adam_optimizer, save_model
from utils import train, test
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR
from setup_cifar10 import setup_cifar10
import utils
from torch_lr_finder import LRFinder
import torch.nn as nn

def main(config):
    """
    Main function for training and testing a model on CIFAR-10 dataset.

    Args:
        config (dict): Configuration parameters for training and testing.

    Returns:
        tuple: A tuple containing the trained model, test data loader, and a list of learning rates.
    """
    criterion = nn.CrossEntropyLoss()
    train_data, test_data, train_loader, test_loader = setup_cifar10(config)
    model = CustomResNet(config).to(config["device"])
    model_summary(model, input_size=(3, 32, 32))
    optimizer = adam_optimizer(model, config)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda:0")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode="exp")
    _, max_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()
    config["max_lr"] = max_lr
    ocp_scheduler = OneCycleLR(
        optimizer,
        max_lr=config["max_lr"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
        pct_start=5 / config["epochs"],
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy="linear",
    )
    lr = []
    for epoch in range(1, config["epochs"] + 1):
        print("EPOCH:", epoch)
        train(model, config["device"], train_loader, optimizer, criterion, epoch)
        test_loss = test(model, config["device"], test_loader)

        if config["lr_scheduler"] == "one_cycle":
            ocp_scheduler.step()
            lr.append(optimizer.param_groups[0]["lr"])
            print("Learning rate:", optimizer.param_groups[0]["lr"])
        elif config["lr_scheduler"] == "none":
            continue

    # format name of model file according to config['norm']
    model_file = "model_" + config["norm"] + ".pth"
    save_model(model, model_file)

    return model, test_loader, lr
