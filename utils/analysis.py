# Utilities functions to conduct model analysis and data visualization

# Imports ----------------------------------------------------------------------

# Common Python imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from IPython.display import Image

# Torch imports
import torch as th
from torch import Tensor


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
                       image: Tensor,
                       mask: Tensor,
                       threshold: float = 0.5,
                       device: th.device = th.device("cpu")
                       ):
    """Function to display the input image, predicted mask, and target mask.

    Parameters
    ----------
    model: th.nn.Module
        U-Net model to use for prediction
    image: Tensor
        Input image to predict the mask of shape [batch_size, channels, height, width]
    mask: Tensor
        Target mask of shape [batch_size, channels, height, width]
    threshold: float, optional (default=0.5)
        Threshold to binarize the predicted mask
    device: th.device, optional (default=th.device("cpu"))
        Device to use for the prediction
    """

    # Move the data to the device
    image, mask = image.to(device), mask.to(device)

    # Obtain the model's prediction
    prediction = th.sigmoid(model(image))

    # Binarize the prediction using a threshold of 0.5
    bin_prediction = (prediction > threshold).float()

    # Process the image and masks for visualization
    pred_mask = prediction.detach().cpu().numpy().squeeze(0)
    bin_pred_mask = bin_prediction.detach().cpu().numpy().squeeze(0)
    true_mask = mask.detach().cpu().numpy().squeeze(0)

    # Display the input image, predicted mask, and target mask
    display_mask_channels(pred_mask, title='Predicted Mask Channels [RGB]')
    display_mask_channels(bin_pred_mask, title='Binarized Predicted Mask Channels [RGB]')
    display_mask_channels(true_mask, title='Ground Truth Mask Channels [RGB]')


# Display single MRI scan ------------------------------------------------------
def display_scan(scan_index: int, 
                 patient_index: int,
                 images: Tensor, 
                 masks: Tensor, 
                 title: str = None,
                 subtitle: str = None,
                 save_path: str = None
):
    """Function to display a single MRI scan with its image channels and mask channels.

    Parameters
    ----------
    scan_index: int
        Index of the scan to display
    patient_index: int
        Index of the patient to display
    images: Tensor
        MRI images of shape [scans, channels, height, width]
    masks: Tensor
        Tumor masks of shape [scans, channels, height, width]
    title: str, optional
        Title of the plot
    subtitle: str, optional
        Subtitle of the plot
    save_path: str, optional
        Path to save the plot
    """

    # Define title and subtitle
    if title is None: title = 'Brain MRI Scan'
    if subtitle is None: subtitle = f'Patient {patient_index} - Scan {scan_index}'

    # Pick the image and mask of the scan
    image = images[scan_index]
    mask = masks[scan_index]

    # Other variables
    title_fontsize = 15
    subtitle_fontsize = 12
    labels_fontsize = 12

    # Channels names
    image_channels: list[str] = ['T1 [C1]', 'T1c [C2]', 'T2 [C3]', 'FLAIR [C4]']
    mask_channels: list[str] = ['NEC [R]', 'EDEMA [G]', 'ET [B]']

    # Build the big plot with everything
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1])

    # Plot image channels (4x4 grid)
    for idx in range(4):
        ax = fig.add_subplot(gs[0, idx])
        channel_image = image[idx, :, :]
        ax.imshow(channel_image, cmap='magma')
        ax.axis('off')
        ax.set_title(image_channels[idx], fontsize=labels_fontsize)

    # Plot mask channels (1x3 grid)
    for idx in range(3):
        ax = fig.add_subplot(gs[1, idx])
        rgb_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        rgb_mask[..., idx] = mask[idx, :, :] * 255
        ax.imshow(rgb_mask)
        ax.axis('off')
        ax.set_title(mask_channels[idx], fontsize=labels_fontsize)

    # Plot overlay (single plot)
    ax = fig.add_subplot(gs[1, 3])
    t1_image = image[0, :, :]
    t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())
    rgb_image = np.stack([t1_image_normalized] * 3, axis=-1)
    color_mask = np.stack([mask[0, :, :], mask[1, :, :], mask[2, :, :]], axis=-1)
    rgb_image = np.where(color_mask, color_mask, rgb_image)
    ax.imshow(rgb_image)
    ax.axis('off')
    ax.set_title('Overlay [RGB]', fontsize=labels_fontsize)

    plt.suptitle(title, fontsize=title_fontsize, y=1)
    fig.text(0.5, 0.9, subtitle, ha='center', fontsize=subtitle_fontsize)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


# Display animated MRI scan ----------------------------------------------------
def display_animated(patient_index: int, 
                     images: Tensor, 
                     masks: Tensor, 
                     output_file: str, 
                     scan_range: range, 
                     interval: float = 200
) -> Image:
    """
    Create an animated GIF that loops over a range of scan indexes for a given patient.

    Parameters
    ----------
    patient_index: int
        Index of the patient to display
    images: Tensor
        MRI images of shape [scans, channels, height, width]
    masks: Tensor
        Tumor masks of shape [scans, channels, height, width]
    output_file: str
        Path to save the animated GIF
    scan_range: range
        Range of scan indexes to display
    interval: float, optional (default=200)
        Interval between each frame in milliseconds

    Returns
    -------
    Image
        Display the animated GIF
    """
  
    # Create a figure
    fig = plt.figure(figsize=(8, 6))

    def update(scan_index):
        # Clear the figure
        fig.clf()

        # Use the display_scan function to create the plot
        temp_file = f'temp_{scan_index}.png'
        display_scan(scan_index, patient_index, images, masks, save_path=temp_file)

        # Read the saved image and display it
        img = plt.imread(temp_file)
        plt.tight_layout()
        plt.imshow(img)
        plt.axis('off')

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=scan_range, interval=interval)

    # Save the animation as a GIF
    ani.save(output_file, writer='imagemagick')

    # Close the figure
    plt.close(fig)

    # Delete temporary images
    for scan_index in scan_range:
        temp_file = f'temp_{scan_index}.png'
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Display the GIF
    return Image(filename=output_file)
