from torch.optim.lr_scheduler import OneCycleLR
from models.model_utils import save_model
from utils import test, train


def main(config, model, train_loader, test_loader, optimizer, criterion):
    """
    Main function that trains and tests a model using the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.
        model: The model to be trained and tested.
        train_loader: The data loader for the training dataset.
        test_loader: The data loader for the testing dataset.
        optimizer: The optimizer used for training the model.
        criterion: The loss criterion used for training the model.

    Returns:
        list: A list containing the learning rates used during training.
    """

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
        train(
            model,
            config["device"],
            train_loader,
            optimizer,
            criterion,
            ocp_scheduler=ocp_scheduler,
            epoch=epoch,
        )
        test(model, config["device"], test_loader)

    # format name of model file according to config['norm']
    model_file = "model_" + config["norm"] + ".pth"
    save_model(model, model_file)

    return lr
