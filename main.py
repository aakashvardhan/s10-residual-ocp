from models.model import CustomResNet
from models.model_utils import model_summary, adam_optimizer, save_model
from utils import train, test
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR
from setup_cifar10_data import setup_cifar10
import utils
from torch_lr_finder import LRFinder

def main(config):
    """
    Main function for training and testing a model on CIFAR-10 dataset.

    Args:
        config (dict): Configuration parameters for training and testing.

    Returns:
        tuple: A tuple containing the trained model, test data loader, and a list of learning rates.
    """
    train_data, test_data, train_loader, test_loader = setup_cifar10(config)
    model = Net(config).to(config["device"])
    model_summary(model, input_size=(3, 32, 32))
    optimizer = sgd_optimizer(model, lr=config["lr"])
    scheduler = StepLR(optimizer, step_size=config["step_size"], gamma=0.5)
    lr_plateau = ReduceLROnPlateau(optimizer, mode="min", patience=5, verbose=True)
    lr = []
    for epoch in range(1, config["epochs"] + 1):
        print("EPOCH:", epoch)
        train(model, config["device"], train_loader, optimizer, epoch)
        test_loss = test(model, config["device"], test_loader)
        if config["lr_scheduler"] == "step_lr":
            scheduler.step()
            lr.append(optimizer.param_groups[0]["lr"])
            print("Learning rate:", optimizer.param_groups[0]["lr"])
        elif config["lr_scheduler"] == "plateau":
            lr_plateau.step(utils.test_losses[-1])
            lr.append(optimizer.param_groups[0]["lr"])
            print("Learning rate:", optimizer.param_groups[0]["lr"])
        elif config["lr_scheduler"] == "none":
            continue

    # format name of model file according to config['norm']
    model_file = "model_" + config["norm"] + ".pth"
    save_model(model, model_file)

    return model, test_loader, lr