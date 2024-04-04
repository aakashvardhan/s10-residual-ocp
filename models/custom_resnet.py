import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conv_block import ConvBlock
from models.res_block import ResBlock

torch.manual_seed(1)


class CustomResNet(nn.Module):
    """
    Custom implementation of the ResNet model.
    
    Args:
        config (dict): Configuration parameters for the model.
            - in_channels (int): Number of input channels.
            - n_channels (int): Number of channels in the convolutional layers.
            - dropout (float): Dropout probability.
            - classes (list): List of class labels.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        in_channels = config["in_channels"]
        n_channels = config["n_channels"]
        dropout_prob = config["dropout"]

        # Prep layer
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
        )  # output_size = 32, RF = 3

        # Layer 1
        self.layer1 = ConvBlock(n_channels, n_channels * 2)  # output_size = 16, RF = 5
        self.res_block1 = ResBlock(
            n_channels * 2, n_channels * 2
        )  # output_size = 16, RF = 7

        # Layer 2
        self.layer2 = ConvBlock(
            n_channels * 2, n_channels * 4
        )  # output_size = 8, RF = 11

        # Layer 3
        self.layer3 = ConvBlock(
            n_channels * 4, n_channels * 8
        )  # output_size = 4, RF = 19
        self.res_block3 = ResBlock(
            n_channels * 8, n_channels * 8
        )  # output_size = 4, RF = 27

        # Output layer
        self.mp4 = nn.MaxPool2d(4)  # output_size = 1, RF = 35
        self.fc = nn.Linear(
            n_channels * 8, len(config["classes"])
        )  # output_size = 1, RF = 35

    def forward(self, x):
        """
        Forward pass of the custom ResNet model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor after passing through the model's layers.
        """
        prep = self.prep_layer(x)
        if self.config["debug"]:
            print("After prep_layer shape:", prep.shape)
        x = self.layer1(prep)
        if self.config["debug"]:
            print("After layer1 shape:", x.shape)
        r1 = self.res_block1(x)
        if self.config["debug"]:
            print("After res_block1 shape:", r1.shape)
        x = x + r1
        if self.config["debug"]:
            print("After adding res_block1 shape:", x.shape)
        x = self.layer2(x)
        if self.config["debug"]:
            print("After layer2 shape:", x.shape)
        x = self.layer3(x)
        if self.config["debug"]:
            print("After layer3 shape:", x.shape)
        r3 = self.res_block3(x)
        if self.config["debug"]:
            print("After res_block3 shape:", r3.shape)
        x = x + r3
        if self.config["debug"]:
            print("After adding res_block3 shape:", x.shape)
        x = self.mp4(x)
        if self.config["debug"]:
            print("After MaxPool shape:", x.shape)
        x = x.view(-1, x.size(1))
        if self.config["debug"]:
            print("After reshaping shape:", x.shape)
        x = self.fc(x)
        if self.config["debug"]:
            print("After fc shape:", x.shape)
        return F.log_softmax(x, dim=-1)
