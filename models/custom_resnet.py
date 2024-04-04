import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conv_block import ConvBlock
from models.res_block import ResBlock

torch.manual_seed(1)


class CustomResNet(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = config["in_channels"]
        dropout_prob = config["dropout_prob"]