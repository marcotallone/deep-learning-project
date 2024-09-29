# Vision Transformer (VIT) model implementation

# Imports ----------------------------------------------------------------------

# Common Python imports
import numpy as np

# Torch imports
import torch as th
from torch import Tensor
from einops import rearrange

# Typining hints
from typing import List, Union, Callable, Tuple


# Vision Transformer (VIT) Network ---------------------------------------------
class VisionTransformer(th.nn.Module):
    """Vision Transformer model

    Parameters
    ----------
    img_size: int, optional (default: 128)
        Image size
    patch_size: int, optional (default: 16)
        Patch size
    output_classes: int, optional (default: 4)
        Number of output classes
    dim: int, optional (default: 512)
        Dimension of the model
    depth: int, optional (default: 10)
        Number of transformer encoder layers
    heads: int, optional (default: 8)
        Number of attention heads
    mlp_dim: int, optional (default: 1024)
        Dimension of the feedforward network
    dropout: float, optional (default: 0.2)
        Dropout rate to use in the model
    """

    # Constructor
    def __init__(self, 
                 img_size: int = 128,
                 patch_size: int = 16,
                 output_classes: int = 4,
                 dim: int = 512,
                 depth: int = 10,
                 heads: int = 8,
                 mlp_dim: int = 1024,
                 dropout: float = 0.2
    ):
        super().__init__()

        # Check if the image dimensions are divisible by the patch size
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches: int = (img_size // patch_size) ** 2
        patch_dim: int = 3 * patch_size * patch_size  # 3 channels for RGB

        # Patch embedding
        self.patch_embed = th.nn.Linear(patch_dim, dim)

        # Positional encoding
        self.pos_embedding = th.nn.Parameter(th.randn(1, num_patches + 1, dim))
        self.cls_token = th.nn.Parameter(th.randn(1, 1, dim))
        self.dropout = th.nn.Dropout(dropout)

        # Transformer encoder layers
        self.transformer_layers = th.nn.ModuleList([
            th.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(depth)
        ])
        self.norm = th.nn.LayerNorm(dim)

        # Classification head
        self.to_cls_token = th.nn.Identity()
        self.mlp_head = th.nn.Sequential(
            th.nn.LayerNorm(dim),
            th.nn.Linear(
                in_features=dim,
                out_features=output_classes
            ),
            th.nn.Dropout(dropout) # Dropout before the classifier
        )


    # Forward pass
    def forward(self, x):
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)  # Flatten patches
        x = self.patch_embed(x)  # [batch_size, num_patches, dim]

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = th.cat((cls_tokens, x), dim=1)  # Add class token
        x += self.pos_embedding  # Add positional encoding
        x = self.dropout(x)

        # Transformer encoder
        for layer in self.transformer_layers:
            x = layer(x) + x  # Residual connection
            x = self.norm(x)

        # Classify
        cls_token_final = self.to_cls_token(x[:, 0])  # Use class token

        return self.mlp_head(cls_token_final)
