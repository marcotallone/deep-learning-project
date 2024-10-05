# U-Net with Attention Gates (AG) model architecture
# Inspired by the paper: 'Attention U-Net: Learning Where to Look for the Pancreas'
# by Oktay et al. (2018)

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
                 activation: th.nn.Module = th.nn.ReLU()
    ) -> None:
        super().__init__()
        expansion_ratio: int = 4
        self.encoder_block: th.nn.Sequential = th.nn.Sequential(
            # Convolutional layer 1: 3 convolutions
            th.nn.Conv2d( # 1. Depthwise convolution
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=7, 
                stride=1, 
                padding=3, 
                groups=in_channels,
            ),
            th.nn.BatchNorm2d(num_features=in_channels),
            th.nn.Conv2d( # 2. Pointwise convolution
                in_channels=in_channels,
                out_channels=expansion_ratio*out_channels, 
                kernel_size=1,
                stride=1
            ),
            activation,
            th.nn.Conv2d( # 3. Pointwise convolution
                in_channels=expansion_ratio*out_channels,
                out_channels=out_channels, 
                kernel_size=1, 
                stride=1
            ),
            # Convolutional layer 2: 3 convolutions
            th.nn.Conv2d( # 1. Depthwise convolution
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=7,
                stride=1, 
                padding=3, 
                groups=out_channels
            ),
            th.nn.BatchNorm2d(num_features=out_channels),
            th.nn.Conv2d( # 2. Pointwise convolution
                in_channels=out_channels,
                out_channels=expansion_ratio*out_channels,
                kernel_size=1, 
                stride=1
            ),
            activation,
            th.nn.Conv2d( # 3. Pointwise convolution
                in_channels=expansion_ratio*out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1
            ),
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
                 activation: th.nn.Module = th.nn.ReLU()
    ) -> None:
        super().__init__()
        expansion_ratio: int = 4
        self.decoder_block: th.nn.Sequential = th.nn.Sequential(
            # Convolutional layer 1: 3 convolutions
            th.nn.Conv2d( # 1. Depthwise convolution
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=7, 
                stride=1, 
                padding=3, 
                groups=in_channels
            ),
            th.nn.BatchNorm2d(num_features=in_channels),
            th.nn.Conv2d( # 2. Pointwise convolution
                in_channels=in_channels,
                out_channels=expansion_ratio*in_channels,
                kernel_size=1, 
                stride=1
            ),
            activation,
            th.nn.Conv2d( # 3. Pointwise convolution
                in_channels=expansion_ratio*in_channels,
                out_channels=out_channels, 
                kernel_size=1, 
                stride=1
            ),
            # Convolutional layer 2: 3 convolutions
            th.nn.Conv2d( # 1. Depthwise convolution
                in_channels=out_channels,
                out_channels=out_channels, 
                kernel_size=7, 
                stride=1, 
                padding=3, 
                groups=out_channels
            ),
            th.nn.BatchNorm2d(num_features=out_channels),
            th.nn.Conv2d( # 2. Pointwise convolution
                in_channels=out_channels,
                out_channels=expansion_ratio*out_channels, 
                kernel_size=1, 
                stride=1
            ),
            activation,
            th.nn.Conv2d( # 3. Pointwise convolution
                in_channels=expansion_ratio*out_channels, 
                out_channels=out_channels,
                kernel_size=1, 
                stride=1
            ),
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
                 activation: th.nn.Module = th.nn.ReLU()
    ) -> None:
        super().__init__()
        self.bottleneck_block: th.nn.Sequential = th.nn.Sequential(
            # Convolutional layer 1: 3 convolutions
            th.nn.Conv2d( # 1. Depthwise convolution
                in_channels=8*n_filters,  
                out_channels=8*n_filters,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=8*n_filters
            ),
            th.nn.BatchNorm2d(num_features=8*n_filters),
            th.nn.Conv2d( # 2. Pointwise convolution
                in_channels=8*n_filters,
                out_channels=4*8*n_filters,
                kernel_size=1, 
                stride=1
            ),
            activation,
            th.nn.Conv2d( # 3. Pointwise convolution
                in_channels=4*8*n_filters,
                out_channels=8*n_filters,
                kernel_size=1,
                stride=1
            ),
            # Convolutional layer 2: 3 convolutions
            th.nn.Conv2d( # 1. Depthwise convolution
                in_channels=8*n_filters,
                out_channels=8*n_filters,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=8*n_filters
            ),
            th.nn.BatchNorm2d(num_features=8*n_filters),
            th.nn.Conv2d( # 2. Pointwise convolution
                in_channels=8*n_filters,
                out_channels=4*8*n_filters,
                kernel_size=1,
                stride=1
            ),
            activation,
            th.nn.Conv2d( # 3. Pointwise convolution
                in_channels=4*8*n_filters,
                out_channels=8*n_filters,
                kernel_size=1,
                stride=1
            ),
        )

    # Forward pass
    def forward(self, x: Tensor) -> Tensor:
        return self.bottleneck_block(x)


# Attention Residual Block -----------------------------------------------------
class attention_residual(th.nn.Module):
    """Attention residual block of the U-Net model

    Parameters
    ----------
    in_channels: int
        Number of input channels
    activation: th.nn.Module, optional (default: th.nn.ReLU())
        Activation function to use in the convolutional layers
    """

    # Constructor
    def __init__(self, 
                 in_channels: int, 
                 activation: th.nn.Module = th.nn.ReLU()
    ) -> None:
        super().__init__()
        self.query_conv: th.nn.Conv2d = th.nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=1,
            stride=1
        )
        self.key_conv: th.nn.Conv2d = th.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=2
        )
        self.attention_conv: th.nn.Conv2d = th.nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1
        )
        self.upsample: th.nn.UpsamplingBilinear2d = th.nn.UpsamplingBilinear2d(scale_factor=2)
        self.activation: th.nn.Module = activation
    
    # Forward pass
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        query: Tensor = self.query_conv(query)
        key: Tensor = self.key_conv(key)
        combined_attention: Tensor = self.activation(query + key)
        attention_map: Tensor = th.sigmoid(self.attention_conv(combined_attention))
        upsampled_attention_map: Tensor = self.upsample(attention_map)
        attention_scores: Tensor = value * upsampled_attention_map
        return attention_scores
        

# Attention U-Net --------------------------------------------------------------
class AttentionUNet(th.nn.Module):
    """Attention U-Net model architecture (Oktay et al., 2018)

    Parameters
    ----------
    in_channels: int, optional (default: 4 [BraTS2020])
        Number of input channels
    out_channels: int, optional (default: 3 [RGB])
        Number of output channels
    n_filters: int, optional (default: 32)
        Number of filters to use in the convolutional layers
    activation: th.nn.Module, optional (default: th.nn.ReLU())
        Activation function to use in the convolutional layers
    name: str, optional (default: "AttentionUNet")
        Name of the model
    """

    # Constructor#    
    def __init__(self,
                 in_channels: int = 4,
                 out_channels: int = 3,
                 n_filters: int = 32,
                 activation: th.nn.Module = th.nn.ReLU(),
                 name: str = "AttentionUNet"
    ) -> None:
        super().__init__()

        # Model name
        self.name: str = name
        
        # Downsampling and Upsampling methods
        self.downsample: th.nn.MaxPool2d = th.nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample: th.nn.Upsample = th.nn.UpsamplingBilinear2d(scale_factor=2)

        # Encoder blocks
        self.encoder1: encoder = encoder(in_channels, 1*n_filters, activation)
        self.encoder2: encoder = encoder(1*n_filters, 2*n_filters, activation)
        self.encoder3: encoder = encoder(2*n_filters, 4*n_filters, activation)
        self.encoder4: encoder = encoder(4*n_filters, 8*n_filters, activation)
        
        # Bottleneck
        self.bottleneck: bottleneck = bottleneck(n_filters, activation)
        
        # Decoder
        self.decoder4: decoder = decoder(8*n_filters, 4*n_filters, activation)
        self.decoder3: decoder = decoder(4*n_filters, 2*n_filters, activation)
        self.decoder2: decoder = decoder(2*n_filters, 1*n_filters, activation)
        self.decoder1: decoder = decoder(1*n_filters, 1*n_filters, activation)
        
        # Output convolutional layer
        self.output: th.nn.Conv2d = th.nn.Conv2d(
            in_channels = 1*n_filters,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
        )     

        # Attention res blocks
        self.attention_residual1: attention_residual = attention_residual(1*n_filters)
        self.attention_residual2: attention_residual = attention_residual(2*n_filters)
        self.attention_residual3: attention_residual = attention_residual(4*n_filters)
        self.attention_residual4: attention_residual = attention_residual(8*n_filters)

    # Forward pass
    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        enc_1 = self.encoder1(x)
        x     = self.downsample(enc_1)
        enc_2 = self.encoder2(x)
        x     = self.downsample(enc_2)
        enc_3 = self.encoder3(x)
        x     = self.downsample(enc_3)
        enc_4 = self.encoder4(x)
        x     = self.downsample(enc_4)
        
        # Bottleneck
        dec_4 = self.bottleneck(x)
        
        # Decoder
        x     = self.upsample(dec_4)
        att_4 = self.attention_residual4(dec_4, enc_4, enc_4)
        x     = th.add(x, att_4) # Add attention wweights
        
        dec_3 = self.decoder4(x)
        x     = self.upsample(dec_3)
        att_3 = self.attention_residual3(dec_3, enc_3, enc_3)
        x     = th.add(x, att_3)  # Add attention wweights
        
        dec_2 = self.decoder3(x)
        x     = self.upsample(dec_2)
        att_2 = self.attention_residual2(dec_2, enc_2, enc_2)
        x     = th.add(x, att_2)  # Add attention wweights
        
        dec_1 = self.decoder2(x)
        x     = self.upsample(dec_1)
        att_1 = self.attention_residual1(dec_1, enc_1, enc_1)
        x     = th.add(x, att_1)  # Add attention wweights
        
        x     = self.decoder1(x)
        x     = self.output(x)
        return x


# Model for attention blocks visualization -------------------------------------

class visual_attention_residual(th.nn.Module):
    """Attention residual block of the U-Net model

    Parameters
    ----------
    in_channels: int
        Number of input channels
    activation: th.nn.Module, optional (default: th.nn.ReLU())
        Activation function to use in the convolutional layers
    """

    # Constructor
    def __init__(self, 
                 in_channels: int, 
                 activation: th.nn.Module = th.nn.ReLU()
    ) -> None:
        super().__init__()
        self.query_conv: th.nn.Conv2d = th.nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=1,
            stride=1
        )
        self.key_conv: th.nn.Conv2d = th.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=2
        )
        self.attention_conv: th.nn.Conv2d = th.nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1
        )
        self.upsample: th.nn.UpsamplingBilinear2d = th.nn.UpsamplingBilinear2d(scale_factor=2)
        self.activation: th.nn.Module = activation
    
    # Forward pass
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        query: Tensor = self.query_conv(query)
        key: Tensor = self.key_conv(key)
        combined_attention: Tensor = self.activation(query + key)
        attention_map: Tensor = th.sigmoid(self.attention_conv(combined_attention))
        upsampled_attention_map: Tensor = self.upsample(attention_map)
        attention_scores: Tensor = value * upsampled_attention_map
        return attention_scores, attention_map


class VisualAttentionUNet(th.nn.Module):
    """Attention U-Net with attention block visualization feaature

    Parameters
    ----------
    in_channels: int, optional (default: 4 [BraTS2020])
        Number of input channels
    out_channels: int, optional (default: 3 [RGB])
        Number of output channels
    n_filters: int, optional (default: 32)
        Number of filters to use in the convolutional layers
    activation: th.nn.Module, optional (default: th.nn.ReLU())
        Activation function to use in the convolutional layers
    name: str, optional (default: "AttentionUNet")
        Name of the model
    """

    # Constructor#    
    def __init__(self,
                 in_channels: int = 4,
                 out_channels: int = 3,
                 n_filters: int = 32,
                 activation: th.nn.Module = th.nn.ReLU(),
                 name: str = "AttentionUNet"
    ) -> None:
        super().__init__()

        # Model name
        self.name: str = name
        
        # Downsampling and Upsampling methods
        self.downsample: th.nn.MaxPool2d = th.nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample: th.nn.Upsample = th.nn.UpsamplingBilinear2d(scale_factor=2)

        # Encoder blocks
        self.encoder1: encoder = encoder(in_channels, 1*n_filters, activation)
        self.encoder2: encoder = encoder(1*n_filters, 2*n_filters, activation)
        self.encoder3: encoder = encoder(2*n_filters, 4*n_filters, activation)
        self.encoder4: encoder = encoder(4*n_filters, 8*n_filters, activation)
        
        # Bottleneck
        self.bottleneck: bottleneck = bottleneck(n_filters, activation)
        
        # Decoder
        self.decoder4: decoder = decoder(8*n_filters, 4*n_filters, activation)
        self.decoder3: decoder = decoder(4*n_filters, 2*n_filters, activation)
        self.decoder2: decoder = decoder(2*n_filters, 1*n_filters, activation)
        self.decoder1: decoder = decoder(1*n_filters, 1*n_filters, activation)
        
        # Output convolutional layer
        self.output: th.nn.Conv2d = th.nn.Conv2d(
            in_channels = 1*n_filters,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
        )     

        # Attention res blocks
        self.attention_residual1: visual_attention_residual = visual_attention_residual(1*n_filters)
        self.attention_residual2: visual_attention_residual = visual_attention_residual(2*n_filters)
        self.attention_residual3: visual_attention_residual = visual_attention_residual(4*n_filters)
        self.attention_residual4: visual_attention_residual = visual_attention_residual(8*n_filters)

    # Forward pass
    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        enc_1 = self.encoder1(x)
        x     = self.downsample(enc_1)
        enc_2 = self.encoder2(x)
        x     = self.downsample(enc_2)
        enc_3 = self.encoder3(x)
        x     = self.downsample(enc_3)
        enc_4 = self.encoder4(x)
        x     = self.downsample(enc_4)
        
        # Bottleneck
        dec_4 = self.bottleneck(x)
        
        # Decoder
        x     = self.upsample(dec_4)
        att_4, map4 = self.attention_residual4(dec_4, enc_4, enc_4)
        x     = th.add(x, att_4)
        
        dec_3 = self.decoder4(x)
        x     = self.upsample(dec_3)
        att_3, map3 = self.attention_residual3(dec_3, enc_3, enc_3)
        x     = th.add(x, att_3)
        
        dec_2 = self.decoder3(x)
        x     = self.upsample(dec_2)
        att_2, map2 = self.attention_residual2(dec_2, enc_2, enc_2)
        x     = th.add(x, att_2)
        
        dec_1 = self.decoder2(x)
        x     = self.upsample(dec_1)
        att_1, map1 = self.attention_residual1(dec_1, enc_1, enc_1)
        x     = th.add(x, att_1)
        
        x     = self.decoder1(x)
        x     = self.output(x)
        return x, map1, map2, map3, map4