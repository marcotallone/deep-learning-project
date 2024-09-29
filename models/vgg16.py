# VGG-16 model architecture
# Adapted from the original VGG-16 architecture to work with 128x128 images

# Imports ----------------------------------------------------------------------

# Common Python imports
import numpy as np

# Torch imports
import torch as th
from torch import Tensor

# Typining hints
from typing import List, Union, Callable, Tuple


# Convolutional blocks for VGG-16 -----------------------------------------------
class double_conv(th.nn.Module):
    """Double convolutional block for VGG-16"""

    # Constructor
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: int = 3, 
                 padding: int = 1,
                 activation: th.nn.Module = th.nn.ReLU(inplace=True)
    ) -> None:
        super().__init__()
        self.conv_block: th.nn.Sequential = th.nn.Sequential(
            # Convolutional layer 1
            th.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            activation,
            # Convolutional layer 2
            th.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            activation,
        )

    # Forward pass
    def forward(self, x: Tensor) -> Tensor:
        return self.conv_block(x)
    

class triple_conv(th.nn.Module):
    """Triple convolutional block for VGG-16"""

    # Constructor
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: int = 3, 
                 padding: int = 1,
                 activation: th.nn.Module = th.nn.ReLU(inplace=True)
    ) -> None:
        super().__init__()
        self.conv_block: th.nn.Sequential = th.nn.Sequential(
            # Convolutional layer 1
            th.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            activation,
            # Convolutional layer 2
            th.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            activation,
            # Convolutional layer 3
            th.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            activation,
        )

    # Forward pass
    def forward(self, x: Tensor) -> Tensor:
        return self.conv_block(x)


# VGG-16 Network for 128x128 images --------------------------------------------
class VGG16_128(th.nn.Module):
    """VGG-16 model for 128x128 images

    Parameters
    ----------
    output_classes: int, optional (default: 4)
        Number of output classes
    dropout: float, optional (default: 0.5)
        Dropout rate to use in the dense layers
    activation: th.nn.Module, optional (default: th.nn.ReLU(inplace=True))
        Activation function to use in the convolutional layers
    """

    # Constructor
    def __init__(self,
                 output_classes: int = 4,
                 dropout: float = 0.5,
                 activation: th.nn.Module = th.nn.ReLU(inplace=True)
    ) -> None:
        super().__init__()

        # Downsampling method
        self.pool: th.nn.MaxPool2d = th.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional layers
        self.conv1: double_conv = double_conv(3, 64, activation=activation)
        self.conv2: double_conv = double_conv(64, 128, activation=activation)
        self.conv3: triple_conv = triple_conv(128, 256, activation=activation)
        self.conv4: triple_conv = triple_conv(256, 512, activation=activation)
        self.conv5: triple_conv = triple_conv(512, 512, activation=activation)

        # Fully connected layers
        self.fully_connected = th.nn.Sequential(
            th.nn.Linear(
                in_features=512 * 4 * 4, # Adjusted for 128x128 input size
                out_features=4096
            ),
            activation,
            th.nn.Dropout(p=dropout),
            th.nn.Linear(
                in_features=4096,
                out_features=4096
            ),
            activation,
            th.nn.Dropout(p=dropout),
            th.nn.Linear(
                in_features=4096,
                out_features=output_classes
            ),
        )

    # Forward pass
    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fully_connected(x)
        return x
