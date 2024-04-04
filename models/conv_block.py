import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import sys
# # add parent directory to path
# sys.path.append('/Users/aakashvardhan/Library/CloudStorage/GoogleDrive-vardhan.aakash1@gmail.com/My Drive/ERA v2/s8-normalization/config.py')


GROUP_SIZE_GN = 2
GROUP_SIZE_LN = 1


class LayerNorm(nn.Module):
    """
    A module that applies Layer Normalization to the input tensor.

    Args:
        num_features (int): The number of input features.

    Attributes:
        layer_norm (nn.GroupNorm): The Group Normalization layer.

    """

    def __init__(self, num_features):
        """
        Initializes a LayerNorm module.

        Args:
            num_features (int): The number of input features.

        """
        super().__init__()
        self.layer_norm = nn.GroupNorm(
            num_groups=GROUP_SIZE_LN, num_channels=num_features
        )

    def forward(self, x):
        """
        Performs a forward pass through the LayerNorm module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized output tensor.

        """
        return self.layer_norm(x)


class ConvBlock(nn.Module):
    """
    Convolutional block module.

    This module represents a convolutional block that consists of a convolutional layer,
    followed by an activation function, normalization, and dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm (str): Type of normalization to be applied. Supported values are 'bn' (BatchNorm2d),
                    'gn' (GroupNorm), and 'ln' (LayerNorm).
        kernel_size (tuple, optional): Size of the convolutional kernel. Defaults to (3, 3).
        dropout_value (float, optional): Dropout probability. Defaults to 0.
        **kwargs: Additional keyword arguments to be passed to the nn.Conv2d layer.

    Raises:
        ValueError: If the norm type is not supported.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        norm="bn",
        kernel_size=(3, 3),
        dropout_value=0,
        **kwargs
    ):
        super().__init__()

        if norm == "bn":
            self.norm = lambda num_features: nn.BatchNorm2d(num_features)
        elif norm == "gn":
            self.norm = lambda num_features: nn.GroupNorm(GROUP_SIZE_GN, num_features)
        elif norm == "ln":
            self.norm = lambda num_features: LayerNorm(num_features)
        else:
            raise ValueError("Norm type {} not supported".format(norm))

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=False,
                **kwargs
            ),
            nn.MaxPool2d(2, 2),
            self.norm(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

    def forward(self, x):
        """
        Forward pass of the ConvBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)
