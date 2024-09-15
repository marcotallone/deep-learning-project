# U-Net Model architecture definition

# Imports ----------------------------------------------------------------------

# Common Python imports
import numpy as np

# Torch imports
import torch as th
from torch import Tensor

# Typining hints
from typing import List, Union, Callable, Tuple


# Encoder block ----------------------------------------------------------------
class encoder(th.nn.Module):
    """Encoder block of the U-Net model
    
    Parameters
    ----------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    activation: th.nn.Module, optional (default: th.nn.ReLU())
        Activation function to use in the convolutional layers
    """
    # Constructor
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 activation: th.nn.Module = th.nn.ReLU(),
    ) -> None:
        super().__init__()
        self.encoder_block: th.nn.Sequential = th.nn.Sequential(
            # Convolutional layer 1
            th.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            activation,
            # Convolutional layer 2
            th.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            activation,
        )

    # Forward pass
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder_block(x)


# Decoder block ----------------------------------------------------------------
class decoder(th.nn.Module):
    """Decoder block of the U-Net model

    Parameters
    ----------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    activation: th.nn.Module, optional (default: th.nn.ReLU())
        Activation function to use in the convolutional layers
    """

    # Constructor
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 activation: th.nn.Module = th.nn.ReLU(),
    ) -> None:
        super().__init__()
        self.decoder_block: th.nn.Sequential = th.nn.Sequential(
            # Convolutional layer 1
            th.nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            activation,
            # Convolutional layer 2
            th.nn.Conv2d(
                in_channels=in_channels // 2,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            activation,
        )

    # Forward pass
    def forward(self, x: Tensor) -> Tensor:
        return self.decoder_block(x)


# Bottleneck block -------------------------------------------------------------
class bottleneck(th.nn.Module):
    """Bottleneck block of the U-Net model

    Parameters
    ----------
    n_filters: int
        Number of filters to use in the convolutional layers
    activation: th.nn.Module, optional (default: th.nn.ReLU())
        Activation function to use in the convolutional layers
    """

    # Constructor
    def __init__(self, 
                 n_filters: int,
                 activation: th.nn.Module = th.nn.ReLU(),
    ) -> None:
        super().__init__()
        self.bottleneck_block: th.nn.Sequential = th.nn.Sequential(
            th.nn.Conv2d(
                in_channels = 8 * n_filters,
                out_channels = 16 * n_filters,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            activation,
            th.nn.Conv2d(
                in_channels = 16 * n_filters,
                out_channels = 8 * n_filters,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            activation,
        )

    # Forward pass
    def forward(self, x: Tensor) -> Tensor:
        return self.bottleneck_block(x)
    

# U-Net model ------------------------------------------------------------------
class UNet(th.nn.Module):

    # Constructor
    def __init__(self,
                in_channels: int = 4, # BraTS2020 dataset images channels
                out_channels: int = 3,
                n_filters: int = 32,
                activation: th.nn.Module = th.nn.ReLU(),
    ) -> None:

        # Call parent constructor
        super().__init__()

        # Downsampling and Upsampling methods
        self.downsample: th.nn.MaxPool2d = th.nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample: th.nn.Upsample = th.nn.UpsamplingBilinear2d(scale_factor=2)

        # Encoder blocks
        self.encoder1: encoder = encoder(in_channels, n_filters, activation)
        self.encoder2: encoder = encoder(1 * n_filters, 2 * n_filters, activation)
        self.encoder3: encoder = encoder(2 * n_filters, 4 * n_filters, activation)
        self.encoder4: encoder = encoder(4 * n_filters, 8 * n_filters, activation)

        # Bottolneck block
        self.bottleneck: bottleneck = bottleneck(n_filters, activation)

        # Decoder blocks
        self.decoder4: decoder = decoder(16 * n_filters, 4 * n_filters, activation)
        self.decoder3: decoder = decoder(8 * n_filters, 2 * n_filters, activation)
        self.decoder2: decoder = decoder(4 * n_filters, 1 * n_filters, activation)
        self.decoder1: decoder = decoder(2 * n_filters, 1 * n_filters, activation)

        # Output convolutional layer
        self.output: th.nn.Conv2d = th.nn.Conv2d(
            in_channels = n_filters,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
        )

    # Forward pass
    def forward(self, x: Tensor) -> Tensor:

        # Encoder
        skip_1 = self.encoder1(x)
        x      = self.downsample(skip_1)
        skip_2 = self.encoder2(x)
        x      = self.downsample(skip_2)
        skip_3 = self.encoder3(x)
        x      = self.downsample(skip_3)
        skip_4 = self.encoder4(x)
        x      = self.downsample(skip_4)
        
        # Bottleneck
        x      = self.bottleneck(x)
        
        # Decoder
        x      = self.upsample(x)
        x      = th.cat((x, skip_4), axis=1)  # Skip connection
        x      = self.decoder4(x)
        x      = self.upsample(x)
        x      = th.cat((x, skip_3), axis=1)  # Skip connection
        x      = self.decoder3(x)
        x      = self.upsample(x)
        x      = th.cat((x, skip_2), axis=1)  # Skip connection
        x      = self.decoder2(x)
        x      = self.upsample(x)
        x      = th.cat((x, skip_1), axis=1)  # Skip connection
        x      = self.decoder1(x)
        x      = self.output(x)
        return x