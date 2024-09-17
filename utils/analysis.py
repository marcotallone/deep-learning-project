# Utilities functions to conduct model analysis and data visualization
# TODO: complete functions to visualize layers and add typing hints

# Imports ----------------------------------------------------------------------

# Common Python imports
import numpy as np
import matplotlib.pyplot as plt

# Torch imports
import torch as th
from torch import Tensor

# Typining hints
from typing import List


# Parameters counter -----------------------------------------------------------
def count_parameters(model: th.nn.Module) -> int:
    """Function to count the total number of parameters of a model
    
    Parameters
    ----------
    model: th.nn.Module
        Model to analyze
        
    Returns
    -------
    int
        Number of parameters of the model
    """

    return sum(p.numel() for p in model.parameters())


# Remove 'module' prefix from the keys of a state dict -------------------------
def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys if it exists.
    Use this function only if you trained your model with the DataParallel
    wrapper module (e.g. multiple GPUs) and saved the model with the
    safetensors library.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict


# Channels visualizer functions for BraTS2020 dataset --------------------------
def display_image_channels(image: Tensor, title='Image Channels'):
    """Function to display the different channels of an MRI image.

    The diplayed channels are:
    - channel 1 [C1]: T1-weighted (T1)
    - channel 2 [C2]: T1-weighted post contrast (T1c)
    - channel 3 [C3]: T2-weighted (T2)
    - channel 4 [C4]: Fluid Attenuated Inversion Recovery (FLAIR)

    Parameters
    ----------
    image: Tensor
        Image to display the channels
    title: str, optional (default='Image Channels')
        Title of the plot
    """

    channel_names: list[str] = ['T1-weighted (T1) [C1]', 
                                'T1-weighted post contrast (T1c) [C2]', 
                                'T2-weighted (T2) [C3]', 
                                'Fluid Attenuated Inversion Recovery (FLAIR) [C4]'
                                ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, ax in enumerate(axes.flatten()):
        channel_image = image[idx, :, :]  # Transpose the array to display the channel
        ax.imshow(channel_image, cmap='magma')
        ax.axis('off')
        ax.set_title(channel_names[idx], fontsize=15)
    plt.tight_layout()
    plt.suptitle(title, fontsize=15, y=1.03)
    plt.show()


def display_mask_channels(mask: Tensor, title='Mask Channels'):
    """Function to display the different channels of a mask as RGB images.

    The diplayed channels are:
    - red channel [R]: Necrotic (NEC)
    - green channel [G]: Edema (ED)
    - blue channel [B]: Tumour (ET)

    Parameters
    ----------
    mask: Tensor
        Mask to display the channels
    title: str, optional (default='Mask Channels')
        Title of the plot
    """

    channel_names: list[str] = ['Necrotic (NEC) [R]', 
                                'Edema (ED) [G]', 
                                'Tumour (ET) [B]'
                                ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    for idx, ax in enumerate(axes):
        rgb_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        rgb_mask[..., idx] = mask[idx, :, :] * 255  # Transpose the array to display the channel
        ax.imshow(rgb_mask)
        ax.axis('off')
        ax.set_title(channel_names[idx], fontsize=15)
    plt.suptitle(title, fontsize=15, y=0.9)
    plt.tight_layout()
    plt.show()


def display_overlay(image: Tensor, 
                    mask: Tensor, 
                    title='Brain MRI with Tumor Masks Overlay [RGB]'
                    ):
    """Function to display the MRI image with the tumor masks overlayed as RGB image.

    Parameters
    ----------
    image: Tensor
        MRI image to display
    mask: Tensor
        Mask to overlay on the MRI image
    title: str, optional (default='Brain MRI with Tumor Masks Overlay [RGB]')
        Title of the plot
    """

    t1_image = image[0, :, :]  # Use the first channel of the image as background
    t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())

    rgb_image = np.stack([t1_image_normalized] * 3, axis=-1)
    color_mask = np.stack([mask[0, :, :], mask[1, :, :], mask[2, :, :]], axis=-1)
    rgb_image = np.where(color_mask, color_mask, rgb_image)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.title(title, fontsize=15, y=1.02)
    plt.axis('off')
    plt.show()


# Prediction visualizer for U-Net models ---------------------------------------
def display_prediction(model: th.nn.Module, 
                       test_input: Tensor,
                       test_target: Tensor, 
                       device: th.device = th.device("cpu")
                       ):

    # Move the data to the device
    test_input, test_target = test_input.to(device), test_target.to(device)

    # Obtain the model's prediction
    test_pred = th.sigmoid(model(test_input))

    # Process the image and masks for visualization
    mask_pred = test_pred.detach().cpu().numpy().squeeze(0)
    mask_target = test_target.detach().cpu().numpy().squeeze(0)

    # Display the input image, predicted mask, and target mask
    display_mask_channels(mask_pred, title='Predicted Mask Channels [RGB]')
    display_mask_channels(mask_target, title='Ground Truth Mask Channels [RGB]')