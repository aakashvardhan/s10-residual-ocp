import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
import torch
import torch.nn.functional as F


def imshow(img):
    """
    Display an image.

    Args:
      img (torch.Tensor): The image tensor to be displayed.

    Returns:
      None
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_misclassified_images(model, test_loader, config):
    """
    Displays the misclassified images along with their corresponding targets and predictions.

    Args:
      model (torch.nn.Module): The trained model.
      test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
      config (dict): A dictionary containing configuration parameters.

    Returns:
      misclass_imgs (list): A list of misclassified images.
      misclass_targets (list): A list of misclassified targets.
      misclass_preds (list): A list of misclassified predictions.
    """
    model.eval()
    misclass_imgs, misclass_targets, misclass_preds = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config["device"]), labels.to(config["device"])
            outputs = model(images)
            pred = outputs.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log probability
            misclassified_mask = ~(pred == labels.view_as(pred)).cpu().numpy()
            # squeeze
            misclassified_mask = misclassified_mask.squeeze()
            misclass_imgs.extend(images[misclassified_mask])
            misclass_targets.extend(labels.view_as(pred)[misclassified_mask])
            misclass_preds.extend(pred[misclassified_mask])

    return misclass_imgs, misclass_targets, misclass_preds


def plt_misclassified_images(
    config, misclass_imgs, misclass_targets, misclass_preds, max_images=10
):
    """
    Plot misclassified images along with their predicted and actual labels.

    Args:
      config (dict): A dictionary containing configuration settings.
      misclass_imgs (list): A list of misclassified images.
      misclass_targets (list): A list of misclassified targets (actual labels).
      misclass_preds (list): A list of misclassified predictions.
      max_images (int, optional): The maximum number of images to plot. Defaults to 10.
    """

    # Determine the number of images to plot (max 10)
    n_images = min(len(misclass_imgs), max_images)
    classes = config["classes"]
    fig = plt.figure(figsize=(20, 4))
    for i in range(n_images):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        im = misclass_imgs[i].cpu().numpy().transpose((1, 2, 0))
        label = misclass_targets[i].cpu().numpy().item()
        pred = misclass_preds[i].cpu().numpy().item()

        # Normalize
        im_ = (im - np.min(im)) / (np.max(im) - np.min(im))
        ax.imshow(im_)
        ax.set_title(f"Prediction: {classes[pred]}\nActual: {classes[label]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()