# AlexNet Network
# Adapted from the original AlexNet architecture to work with 128x128 images

# Imports ----------------------------------------------------------------------

# Common Python imports
import numpy as np

# Torch imports
import torch as th
from torch import Tensor

# Typining hints
from typing import List, Union, Callable, Tuple


# AlexNet Network for 128x128 images -------------------------------------------
class AlexNet_128(th.nn.Module):
    """AlexNet model for 128x128 images

    Parameters
    ----------
    output_classes: int, optional (default: 4)
        Number of output classes
    activation: th.nn.Module, optional (default: th.nn.ReLU())
        Activation function to use in the convolutional layers
    """

    # Constructor
    def __init__(self,
                 output_classes: int = 4,
                 activation: th.nn.Module = th.nn.ReLU()
    ) -> None:
        super().__init__()
        
       # Convolutional layers 
        self.conv: th.nn.Sequential = th.nn.Sequential( 
            # Convolutional layer 1
            th.nn.Conv2d(
                in_channels=3, 
                out_channels=96, 
                kernel_size=9, 
                stride=4, 
                padding=0
            ),
            activation,
            th.nn.MaxPool2d(kernel_size=2),
            # Convolutional layer 2
            th.nn.Conv2d(
                in_channels=96, 
                out_channels=256,
                kernel_size=5, 
                stride=1, 
                padding=0
            ),
            activation,
            th.nn.MaxPool2d(kernel_size=2),
            # Convolutional layer 3
            th.nn.Conv2d(
                in_channels=256, 
                out_channels=384, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            activation,
            # Convolutional layer 4
            th.nn.Conv2d(
                in_channels=384, 
                out_channels=384, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            activation,
            # Convolutional layer 5
            th.nn.Conv2d(
                in_channels=384, 
                out_channels=256, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            activation,
            th.nn.MaxPool2d(kernel_size=2),
            # Flatten for the dense layers
            th.nn.Flatten(),
        )

        # Dense / Fully connected layers
        self.head = th.nn.Sequential(
            th.nn.Linear(
                in_features=1024,
                out_features=512
            ),
            activation,
            th.nn.Linear(
                in_features=512,
                out_features=512
            ),
            activation,
            th.nn.Linear(
                in_features=512,
                out_features=128
            ),
            activation,
            th.nn.Linear(
                in_features=128,
                out_features=output_classes
            )
            #th.nn.Softmax(dim=1),
        )

    # Forward pass
    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.conv(x))
