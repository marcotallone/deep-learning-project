# Utilities functions to conduct model analysis and data visualization
# Some of these functions are pretty long but it's just a bunch of plotting 
# commands to produce nice plots for better data visualization

# Imports ----------------------------------------------------------------------

# Common Python imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import seaborn as sns
from IPython.display import Image

# Torch imports
import torch as th
from torch import Tensor

# Define custom palette
tokyo = {
	"red": "#F7768E",
	"orange": "#FF9E64",
	"yellow": "#FFCB30",
	"green": "#9ECE6A",
	"cyan": "#2AC3DE",
	"blue": "#7AA2F7",
	"purple": "#BB9AF7",
}

# Apply the custom color palette globally
sns.set_palette(list(tokyo.values()))


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
def display_image_channels(image: Tensor, title='Image Channels', flat: bool = False, save_path: str = None):
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
	flat: bool, optional (default=False)
		Display the channels as a flat grid
	save_path: str, optional
		Path to save the plot
	"""

	channel_names: list[str] = ['T1-weighted (T1) [C1]', 
								'T1-weighted post contrast (T1c) [C2]', 
								'T2-weighted (T2) [C3]', 
								'Fluid Attenuated Inversion\nRecovery (FLAIR) [C4]'
								]
	
	# Display the channels as a flat grid
	if flat:
		fig, ax = plt.subplots(1, 4, figsize=(15, 5))
		for idx, ax in enumerate(ax):
			channel_image = image[idx, :, :]
			ax.imshow(channel_image, cmap='gray')
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_title(channel_names[idx], fontsize=15)
		# plt.suptitle(title, fontsize=15, y=1.03)
		plt.tight_layout()
		plt.savefig(save_path, format='png', dpi=600, transparent=True)
		plt.close(fig)

	# Display the channels as a 2x2 grid
	else:
		fig, axes = plt.subplots(2, 2, figsize=(10, 10))
		for idx, ax in enumerate(axes.flatten()):
			channel_image = image[idx, :, :]  # Transpose the array to display the channel
			ax.imshow(channel_image, cmap='gray')
			# ax.axis('off')
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_title(channel_names[idx], fontsize=15)
		plt.tight_layout()
		plt.suptitle(title, fontsize=15, y=1.03)
		# plt.show()
		plt.savefig(save_path, format='png', dpi=600, transparent=True)
		plt.close(fig)


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
		# ax.axis('off')
		ax.set_xticks([])
		ax.set_yticks([])
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
	# plt.axis('off')
	plt.xticks([])
	plt.yticks([])
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
	channel_names: list[str] = ['Necrotic (NEC) [R]', 
								'Edema (ED) [G]', 
								'Tumour (ET) [B]'
								]

	# Build the big plot with everything
	fig = plt.figure(figsize=(8, 6))
	gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1])

	# Plot image channels (4x4 grid)
	for idx in range(4):
		ax = fig.add_subplot(gs[0, idx])
		channel_image = image[idx, :, :]
		ax.imshow(channel_image, cmap='gray')
		# ax.axis('off')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title(image_channels[idx], fontsize=labels_fontsize)

	# Plot mask channels (1x3 grid)
	for idx in range(3):
		ax = fig.add_subplot(gs[1, idx])
		rgb_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
		rgb_mask[..., idx] = mask[idx, :, :] * 255
		ax.imshow(rgb_mask)
		# ax.axis('off')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title(mask_channels[idx], fontsize=labels_fontsize)

	# Plot overlay (single plot)
	ax = fig.add_subplot(gs[1, 3])
	t1_image = image[0, :, :]
	t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())
	rgb_image = np.stack([t1_image_normalized] * 3, axis=-1)
	color_mask = np.stack([mask[0, :, :], mask[1, :, :], mask[2, :, :]], axis=-1)
	rgb_image = np.where(color_mask, color_mask, rgb_image)
	ax.imshow(rgb_image)
	# ax.axis('off')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title('Overlay [RGB]', fontsize=labels_fontsize)

	plt.suptitle(title, fontsize=title_fontsize, y=1)
	fig.text(0.5, 0.9, subtitle, ha='center', fontsize=subtitle_fontsize)
	plt.tight_layout()

	# Only for presentation -----------------------------------------
	# Plot only ground thruth mask channels and overlay in a 1x4 grid
	# fig, axes = plt.subplots(1, 4, figsize=(15, 5))

	# # Plot mask channels (1x3 grid)
	# for idx, ax in enumerate(axes[:3]):
	# 	rgb_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
	# 	rgb_mask[..., idx] = mask[idx, :, :] * 255
	# 	ax.imshow(rgb_mask)
	# 	# ax.axis('off')
	# 	ax.set_xticks([])
	# 	ax.set_yticks([])
	# 	ax.set_title(channel_names[idx], fontsize=labels_fontsize)

	# # Plot overlay (single plot)
	# ax = axes[3]
	# t1_image = image[0, :, :]
	# t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())
	# rgb_image = np.stack([t1_image_normalized] * 3, axis=-1)
	# color_mask = np.stack([mask[0, :, :], mask[1, :, :], mask[2, :, :]], axis=-1)
	# rgb_image = np.where(color_mask, color_mask, rgb_image)
	# ax.imshow(rgb_image)
	# # ax.axis('off')
	# ax.set_xticks([])
	# ax.set_yticks([])
	# ax.set_title('Overlay [RGB]', fontsize=labels_fontsize)
	
	if save_path:
		plt.tight_layout()
		plt.savefig(save_path, format='png', dpi=600, transparent=True)
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
		# plt.axis('off')
		plt.xticks([])
		plt.yticks([])

		# Adjust layout to remove white borders
		plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)

	# Create the animation
	ani = animation.FuncAnimation(fig, update, frames=scan_range, interval=interval)

	# Save the animation as a GIF
	ani.save(output_file, writer='pillow', dpi=600, savefig_kwargs={'transparent': True})

	# Close the figure
	plt.close(fig)

	# Delete temporary images
	for scan_index in scan_range:
		temp_file = f'temp_{scan_index}.png'
		if os.path.exists(temp_file):
			os.remove(temp_file)

	# Display the GIF
	return Image(filename=output_file)


# Plot performance metrics vs epochs -------------------------------------------
def plot_metrics(df: pd.DataFrame, model: th.nn.Module, set_name: str) -> None:
	"""
	Function to plot the performance metrics of a U-Net model against the epochs
	for either the training or validation set.
	"""

	# Plot all the classic U-Net metrics against the epochs

	if set_name == 'train' or set_name == 'training':

		fig, ax = plt.subplots(4, 2, figsize=(15, 20))
		fig.suptitle(f"{model.module.name if isinstance(model, th.nn.DataParallel) else model.name} Training Metrics through Epochs", fontsize=16, y=1)

		# Set all the axes xticks as the epochs
		epochs = df['epoch'].values
		for i in range(4):
			for j in range(2):
				ax[i, j].set_xticks(epochs)

		# Loss
		sns.lineplot(data=df, x='epoch', y='train_loss', label='Train Loss', ax=ax[0, 0], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='validation_loss', label='Validation Loss', ax=ax[0, 0], marker='o', color='orange')
		ax[0, 0].set_title("Train and Validation Loss")
		ax[0, 0].set_xlabel("Epochs")
		ax[0, 0].set_ylabel("Loss")
		ax[0, 0].legend()


		# Accuracy
		sns.lineplot(data=df, x='epoch', y='accuracy_red', label='NEC', ax=ax[0, 1], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='accuracy_green', label='ED', ax=ax[0, 1], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='accuracy_blue', label='ET', ax=ax[0, 1], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='accuracy_average', label='Average', ax=ax[0, 1], marker='o', color='black', linestyle='--')
		ax[0, 1].set_title("Accuracy")
		ax[0, 1].set_xlabel("Epochs")
		ax[0, 1].set_ylabel("Accuracy")
		ax[0, 1].legend()

		# Dice
		sns.lineplot(data=df, x='epoch', y='dice_red', label='NEC', ax=ax[1, 0], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='dice_green', label='ED', ax=ax[1, 0], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='dice_blue', label='ET', ax=ax[1, 0], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='dice_average', label='Average', ax=ax[1, 0], marker='o', color='black', linestyle='--')
		ax[1, 0].set_title("Dice Coefficient")
		ax[1, 0].set_xlabel("Epochs")
		ax[1, 0].set_ylabel("Dice")
		ax[1, 0].legend()

		# IoU
		sns.lineplot(data=df, x='epoch', y='iou_red', label='NEC', ax=ax[1, 1], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='iou_green', label='ED', ax=ax[1, 1], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='iou_blue', label='ET', ax=ax[1, 1], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='iou_average', label='Average', ax=ax[1, 1], marker='o', color='black', linestyle='--')
		ax[1, 1].set_title("Intersection over Union")
		ax[1, 1].set_xlabel("Epochs")
		ax[1, 1].set_ylabel("IoU")
		ax[1, 1].legend()

		# FPR
		sns.lineplot(data=df, x='epoch', y='fpr_red', label='NEC', ax=ax[2, 0], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='fpr_green', label='ED', ax=ax[2, 0], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='fpr_blue', label='ET', ax=ax[2, 0], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='fpr_average', label='Average', ax=ax[2, 0], marker='o', color='black', linestyle='--')
		ax[2, 0].set_title("False Positive Rate")
		ax[2, 0].set_xlabel("Epochs")
		ax[2, 0].set_ylabel("FPR")
		ax[2, 0].legend()

		# FNR
		sns.lineplot(data=df, x='epoch', y='fnr_red', label='NEC', ax=ax[2, 1], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='fnr_green', label='ED', ax=ax[2, 1], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='fnr_blue', label='ET', ax=ax[2, 1], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='fnr_average', label='Average', ax=ax[2, 1], marker='o', color='black', linestyle='--')
		ax[2, 1].set_title("False Negative Rate")
		ax[2, 1].set_xlabel("Epochs")
		ax[2, 1].set_ylabel("FNR")
		ax[2, 1].legend()

		# Precision
		sns.lineplot(data=df, x='epoch', y='precision_red', label='NEC', ax=ax[3, 0], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='precision_green', label='ED', ax=ax[3, 0], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='precision_blue', label='ET', ax=ax[3, 0], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='precision_average', label='Average', ax=ax[3, 0], marker='o', color='black', linestyle='--')
		ax[3, 0].set_title("Precision")
		ax[3, 0].set_xlabel("Epochs")
		ax[3, 0].set_ylabel("Precision")
		ax[3, 0].legend()

		# Recall
		sns.lineplot(data=df, x='epoch', y='recall_red', label='NEC', ax=ax[3, 1], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='recall_green', label='ED', ax=ax[3, 1], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='recall_blue', label='ET', ax=ax[3, 1], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='recall_average', label='Average', ax=ax[3, 1], marker='o', color='black', linestyle='--')
		ax[3, 1].set_title("Recall")
		ax[3, 1].set_xlabel("Epochs")
		ax[3, 1].set_ylabel("Recall")
		ax[3, 1].legend()

		plt.tight_layout()
		plt.show()

	elif set_name == 'valid' or set_name == 'validation':

		fig, ax = plt.subplots(4, 2, figsize=(15, 20))
		fig.suptitle(f"{model.module.name if isinstance(model, th.nn.DataParallel) else model.name} Validation Metrics through Epochs", fontsize=16, y=1)

		# Set all the axes xticks as the epochs
		epochs = df['epoch'].values
		for i in range(4):
			for j in range(2):
				ax[i, j].set_xticks(epochs)

		# Loss
		sns.lineplot(data=df, x='epoch', y='train_loss', label='Train Loss', ax=ax[0, 0], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='validation_loss', label='Validation Loss', ax=ax[0, 0], marker='o', color='orange')
		ax[0, 0].set_title("Train and Validation Loss")
		ax[0, 0].set_xlabel("Epochs")
		ax[0, 0].set_ylabel("Loss")
		ax[0, 0].legend()

		# Accuracy
		sns.lineplot(data=df, x='epoch', y='accuracy_red_e', label='NEC', ax=ax[0, 1], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='accuracy_green_e', label='ED', ax=ax[0, 1], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='accuracy_blue_e', label='ET', ax=ax[0, 1], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='accuracy_average_e', label='Average', ax=ax[0, 1], marker='o', color='black', linestyle='--')
		ax[0, 1].set_title("Accuracy")
		ax[0, 1].set_xlabel("Epochs")
		ax[0, 1].set_ylabel("Accuracy")
		ax[0, 1].legend()

		# Dice
		sns.lineplot(data=df, x='epoch', y='dice_red_e', label='NEC', ax=ax[1, 0], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='dice_green_e', label='ED', ax=ax[1, 0], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='dice_blue_e', label='ET', ax=ax[1, 0], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='dice_average_e', label='Average', ax=ax[1, 0], marker='o', color='black', linestyle='--')
		ax[1, 0].set_title("Dice Coefficient")
		ax[1, 0].set_xlabel("Epochs")
		ax[1, 0].set_ylabel("Dice")
		ax[1, 0].legend()

		# IoU
		sns.lineplot(data=df, x='epoch', y='iou_red_e', label='NEC', ax=ax[1, 1], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='iou_green_e', label='ED', ax=ax[1, 1], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='iou_blue_e', label='ET', ax=ax[1, 1], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='iou_average_e', label='Average', ax=ax[1, 1], marker='o', color='black', linestyle='--')
		ax[1, 1].set_title("Intersection over Union")
		ax[1, 1].set_xlabel("Epochs")
		ax[1, 1].set_ylabel("IoU")
		ax[1, 1].legend()

		# FPR
		sns.lineplot(data=df, x='epoch', y='fpr_red_e', label='NEC', ax=ax[2, 0], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='fpr_green_e', label='ED', ax=ax[2, 0], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='fpr_blue_e', label='ET', ax=ax[2, 0], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='fpr_average_e', label='Average', ax=ax[2, 0], marker='o', color='black', linestyle='--')
		ax[2, 0].set_title("False Positive Rate")
		ax[2, 0].set_xlabel("Epochs")
		ax[2, 0].set_ylabel("FPR")
		ax[2, 0].legend()

		# FNR
		sns.lineplot(data=df, x='epoch', y='fnr_red_e', label='NEC', ax=ax[2, 1], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='fnr_green_e', label='ED', ax=ax[2, 1], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='fnr_blue_e', label='ET', ax=ax[2, 1], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='fnr_average_e', label='Average', ax=ax[2, 1], marker='o', color='black', linestyle='--')
		ax[2, 1].set_title("False Negative Rate")
		ax[2, 1].set_xlabel("Epochs")
		ax[2, 1].set_ylabel("FNR")
		ax[2, 1].legend()

		# Precision
		sns.lineplot(data=df, x='epoch', y='precision_red_e', label='NEC', ax=ax[3, 0], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='precision_green_e', label='ED', ax=ax[3, 0], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='precision_blue_e', label='ET', ax=ax[3, 0], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='precision_average_e', label='Average', ax=ax[3, 0], marker='o', color='black', linestyle='--')
		ax[3, 0].set_title("Precision")
		ax[3, 0].set_xlabel("Epochs")
		ax[3, 0].set_ylabel("Precision")
		ax[3, 0].legend()

		# Recall
		sns.lineplot(data=df, x='epoch', y='recall_red_e', label='NEC', ax=ax[3, 1], marker='o', color='red')
		sns.lineplot(data=df, x='epoch', y='recall_green_e', label='ED', ax=ax[3, 1], marker='o', color='green')
		sns.lineplot(data=df, x='epoch', y='recall_blue_e', label='ET', ax=ax[3, 1], marker='o', color='blue')
		sns.lineplot(data=df, x='epoch', y='recall_average_e', label='Average', ax=ax[3, 1], marker='o', color='black', linestyle='--')
		ax[3, 1].set_title("Recall")
		ax[3, 1].set_xlabel("Epochs")
		ax[3, 1].set_ylabel("Recall")
		ax[3, 1].legend()

		plt.tight_layout()
		plt.show()

	else:
		raise ValueError("The set_name parameter must be either 'train' or 'valid'.")


# Barplot of the performance metrics -------------------------------------------
def barplot_metrics(df: pd.DataFrame, model: th.nn.Module, set_name: str) -> None:
	"""
	Function to plot the final performance metrics of a U-Net model as a barplot.
	for either the training or validation set.
	"""

	if set_name == 'train' or set_name == 'training':

		# Convert dataframe format collecting final metrics
		data = {
			'Channel': ['Red', 'Green', 'Blue', 'Average'],
			'Dice': [df['dice_red'].iloc[-1], df['dice_green'].iloc[-1], df['dice_blue'].iloc[-1], df['dice_average'].iloc[-1]],
			'IoU': [df['iou_red'].iloc[-1], df['iou_green'].iloc[-1], df['iou_blue'].iloc[-1], df['iou_average'].iloc[-1]],
			'Accuracy': [df['accuracy_red'].iloc[-1], df['accuracy_green'].iloc[-1], df['accuracy_blue'].iloc[-1], df['accuracy_average'].iloc[-1]],
			'FPR': [df['fpr_red'].iloc[-1], df['fpr_green'].iloc[-1], df['fpr_blue'].iloc[-1], df['fpr_average'].iloc[-1]],
			'FNR': [df['fnr_red'].iloc[-1], df['fnr_green'].iloc[-1], df['fnr_blue'].iloc[-1], df['fnr_average'].iloc[-1]],
			'Precision': [df['precision_red'].iloc[-1], df['precision_green'].iloc[-1], df['precision_blue'].iloc[-1], df['precision_average'].iloc[-1]],
			'Recall': [df['recall_red'].iloc[-1], df['recall_green'].iloc[-1], df['recall_blue'].iloc[-1], df['recall_average'].iloc[-1]],
		}

		# Create a new DataFrame
		new_df = pd.DataFrame(data)
		
		# Melt the DataFrame
		df_melted = pd.melt(new_df, id_vars="Channel", var_name="metric", value_name="value")

		# Define a palettes
		palette = {
			'Red': 'red',
			'Blue': 'blue',
			'Green': 'green',
			'Average': 'gray',
		}

		# Plot using catplot for combined accuracies and losses
		g = sns.catplot(x='metric', y='value', hue='Channel', data=df_melted, kind='bar', height=5, aspect=1.1, palette=palette)
		g.set(ylim=(0, 1))
		g.set_axis_labels("", "")
		for ax in g.axes.flat: ax.set_yticks([i * 0.1 for i in range(11)])
		title = f"{model.module.name if isinstance(model, th.nn.DataParallel) else model.name} Final  Training Metrics"
		g.fig.suptitle(title, y=1.08)

		# Conver the legent orientation to horizontal and move it on top + remove legend title
		sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, title=None)

		plt.show()

	elif set_name == 'valid' or set_name == 'validation':

		# Convert dataframe format collecting final metrics
		data = {
			'Channel': ['Red', 'Green', 'Blue', 'Average'],
			'Dice': [df['dice_red_e'].iloc[-1], df['dice_green_e'].iloc[-1], df['dice_blue_e'].iloc[-1], df['dice_average_e'].iloc[-1]],
			'IoU': [df['iou_red_e'].iloc[-1], df['iou_green_e'].iloc[-1], df['iou_blue_e'].iloc[-1], df['iou_average_e'].iloc[-1]],
			'Accuracy': [df['accuracy_red_e'].iloc[-1], df['accuracy_green_e'].iloc[-1], df['accuracy_blue_e'].iloc[-1], df['accuracy_average_e'].iloc[-1]],
			'FPR': [df['fpr_red_e'].iloc[-1], df['fpr_green_e'].iloc[-1], df['fpr_blue_e'].iloc[-1], df['fpr_average_e'].iloc[-1]],
			'FNR': [df['fnr_red_e'].iloc[-1], df['fnr_green_e'].iloc[-1], df['fnr_blue_e'].iloc[-1], df['fnr_average_e'].iloc[-1]],
			'Precision': [df['precision_red_e'].iloc[-1], df['precision_green_e'].iloc[-1], df['precision_blue_e'].iloc[-1], df['precision_average_e'].iloc[-1]],
			'Recall': [df['recall_red_e'].iloc[-1], df['recall_green_e'].iloc[-1], df['recall_blue_e'].iloc[-1], df['recall_average_e'].iloc[-1]],
		}

		# Create a new DataFrame
		new_df = pd.DataFrame(data)
		
		# Melt the DataFrame
		df_melted = pd.melt(new_df, id_vars="Channel", var_name="metric", value_name="value")

		# Define a palettes
		palette = {
			'Red': 'red',
			'Blue': 'blue',
			'Green': 'green',
			'Average': 'gray',
		}

		# Plot using catplot for combined accuracies and losses
		g = sns.catplot(x='metric', y='value', hue='Channel', data=df_melted, kind='bar', height=5, aspect=1.1, palette=palette)
		g.set(ylim=(0, 1))
		g.set_axis_labels("", "")
		for ax in g.axes.flat: ax.set_yticks([i * 0.1 for i in range(11)])
		title = f"{model.module.name if isinstance(model, th.nn.DataParallel) else model.name} Final Validation Metrics"
		g.fig.suptitle(title, y=1.08)

		# Conver the legent orientation to horizontal and move it on top + remove legend title
		sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, title=None)

		plt.show()

	else:
		raise ValueError("The set_name parameter must be either 'train' or 'valid'.")


def barplot_metrics_multiple(dfs: list, models: list) -> None:
	"""
	Function to plot the final performance metrics of multiple U-Net models as barplots in a 2xn_model grid.
	"""
	assert len(dfs) == len(models), "The number of DataFrames must match the number of models."

	# Create a figure and axes for the 1x3 grid
	fig, axes = plt.subplots(2, len(dfs), figsize=(6*len(dfs), 10), squeeze=True)

	for i, (df, model) in enumerate(zip(dfs, models)):
		# Convert dataframe format collecting final metrics
		data = {
			'Channel': ['Red', 'Green', 'Blue', 'Average'],
			'Dice': [df['dice_red'].iloc[-1], df['dice_green'].iloc[-1], df['dice_blue'].iloc[-1], df['dice_average'].iloc[-1]],
			'IoU': [df['iou_red'].iloc[-1], df['iou_green'].iloc[-1], df['iou_blue'].iloc[-1], df['iou_average'].iloc[-1]],
			'Accuracy': [df['accuracy_red'].iloc[-1], df['accuracy_green'].iloc[-1], df['accuracy_blue'].iloc[-1], df['accuracy_average'].iloc[-1]],
			'FPR': [df['fpr_red'].iloc[-1], df['fpr_green'].iloc[-1], df['fpr_blue'].iloc[-1], df['fpr_average'].iloc[-1]],
			'FNR': [df['fnr_red'].iloc[-1], df['fnr_green'].iloc[-1], df['fnr_blue'].iloc[-1], df['fnr_average'].iloc[-1]],
			'Precision': [df['precision_red'].iloc[-1], df['precision_green'].iloc[-1], df['precision_blue'].iloc[-1], df['precision_average'].iloc[-1]],
			'Recall': [df['recall_red'].iloc[-1], df['recall_green'].iloc[-1], df['recall_blue'].iloc[-1], df['recall_average'].iloc[-1]],
		}
		data_e = {
			'Channel': ['Red', 'Green', 'Blue', 'Average'],
			'Dice': [df['dice_red_e'].iloc[-1], df['dice_green_e'].iloc[-1], df['dice_blue_e'].iloc[-1], df['dice_average_e'].iloc[-1]],
			'IoU': [df['iou_red_e'].iloc[-1], df['iou_green_e'].iloc[-1], df['iou_blue_e'].iloc[-1], df['iou_average_e'].iloc[-1]],
			'Accuracy': [df['accuracy_red_e'].iloc[-1], df['accuracy_green_e'].iloc[-1], df['accuracy_blue_e'].iloc[-1], df['accuracy_average_e'].iloc[-1]],
			'FPR': [df['fpr_red_e'].iloc[-1], df['fpr_green_e'].iloc[-1], df['fpr_blue_e'].iloc[-1], df['fpr_average_e'].iloc[-1]],
			'FNR': [df['fnr_red_e'].iloc[-1], df['fnr_green_e'].iloc[-1], df['fnr_blue_e'].iloc[-1], df['fnr_average_e'].iloc[-1]],
			'Precision': [df['precision_red_e'].iloc[-1], df['precision_green_e'].iloc[-1], df['precision_blue_e'].iloc[-1], df['precision_average_e'].iloc[-1]],
			'Recall': [df['recall_red_e'].iloc[-1], df['recall_green_e'].iloc[-1], df['recall_blue_e'].iloc[-1], df['recall_average_e'].iloc[-1]],
		}

		# Create a new DataFrame
		new_df = pd.DataFrame(data)
		new_df_e = pd.DataFrame(data_e)
		
		# Melt the DataFrame
		df_melted = pd.melt(new_df, id_vars="Channel", var_name="metric", value_name="value")
		df_melted_e = pd.melt(new_df_e, id_vars="Channel", var_name="metric", value_name="value")

		# Define a palettes
		palette = {
			'Red': 'red',
			'Blue': 'blue',
			'Green': 'green',
			'Average': 'gray',
		}

		# Plot using barplot training metrics in first row
		sns.barplot(x='metric', y='value', hue='Channel', data=df_melted, ax=axes[0, i], palette=palette, linewidth=0.5)
		axes[0, i].set(ylim=(0, 1))
		axes[0, i].set_yticks([i * 0.1 for i in range(11)])
		axes[0, i].set_xlabel("")
		axes[0, i].set_ylabel("Training Metrics")
		axes[0, i].set_title(f"{model.module.name if isinstance(model, th.nn.DataParallel) else model.name} Training Metrics", y=1.16)
		axes[0, i].legend(title="", loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=4)

		# Plot using barplot validation metrics in second row
		sns.barplot(x='metric', y='value', hue='Channel', data=df_melted_e, ax=axes[1, i], palette=palette, linewidth=0.5)
		axes[1, i].set(ylim=(0, 1))
		axes[1, i].set_yticks([i * 0.1 for i in range(11)])
		axes[1, i].set_xlabel("")
		axes[1, i].set_ylabel("Validation Metrics")
		axes[1, i].set_title(f"{model.module.name if isinstance(model, th.nn.DataParallel) else model.name} Validation Metrics", y=1.16)
		axes[1, i].legend(title="", loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=4)

	# Adjust layout
	plt.tight_layout()
	# plt.show()
	plt.savefig("images/metrics_multiple.png", format='png', dpi=600, transparent=True)
	plt.close()


# Plot chosen metrics for all channels and average -----------------------------

# Plots of dice, precision and recall over epochs for red channel
def plot_red_metrics(classic_df: pd.DataFrame, improved_df: pd.DataFrame, attention_df: pd.DataFrame) -> None:
	"""
	Function to plot dice, precision and recall of the red channel 
	for the classic, improved and attention U-Net models.
	"""

	fig, ax = plt.subplots(1, 3, figsize=(18, 7))

	# Set all the axes xticks as the epochs
	epochs = classic_df['epoch'].values
	ax[0].set_xticks(epochs)
	ax[1].set_xticks(epochs)
	ax[2].set_xticks(epochs)

	# Set yticks from 0.0 to 1.0 onsteps of 0.2
	ax[0].set_yticks(np.arange(0.0, 1.1, 0.1))
	ax[1].set_yticks(np.arange(0.0, 1.1, 0.1))
	ax[2].set_yticks(np.arange(0.0, 1.1, 0.1))

	# Set yrange [0,1] included
	ax[0].set_ylim([0, 1])
	ax[1].set_ylim([0, 1])
	ax[2].set_ylim([0, 1])

	# Dice
	sns.lineplot(data=classic_df, x='epoch', y='dice_red_e', label='Classic U-Net', marker='o', color=tokyo['red'], markeredgewidth=0, ax=ax[0])
	sns.lineplot(data=improved_df, x='epoch', y='dice_red_e', label='Improved U-Net', marker='o', color=tokyo['cyan'], markeredgewidth=0, ax=ax[0])
	sns.lineplot(data=attention_df, x='epoch', y='dice_red_e', label='Attention U-Net', marker='o', color=tokyo['green'], markeredgewidth=0, ax=ax[0])
	red_bullet = '\u2B24'  # Unicode for red circle
	ax[0].set_title(f"Dice for Red Channel {red_bullet}", fontsize=16, color=tokyo['red'])
	ax[0].set_xlabel("Epochs")
	ax[0].set_ylabel("")
	ax[0].legend(loc='lower right')

	# Precision
	sns.lineplot(data=classic_df, x='epoch', y='precision_red_e', label='Classic U-Net', marker='o', color=tokyo['red'], markeredgewidth=0, ax=ax[1])
	sns.lineplot(data=improved_df, x='epoch', y='precision_red_e', label='Improved U-Net', marker='o', color=tokyo['cyan'], markeredgewidth=0, ax=ax[1])
	sns.lineplot(data=attention_df, x='epoch', y='precision_red_e', label='Attention U-Net', marker='o', color=tokyo['green'], markeredgewidth=0, ax=ax[1])
	ax[1].set_title(f"Precision for Red Channel {red_bullet}", fontsize=16, color=tokyo['red'])
	ax[1].set_xlabel("Epochs")
	ax[1].set_ylabel("")
	ax[1].legend(loc='lower right')

	# Recall
	sns.lineplot(data=classic_df, x='epoch', y='recall_red_e', label='Classic U-Net', marker='o', color=tokyo['red'], markeredgewidth=0, ax=ax[2])
	sns.lineplot(data=improved_df, x='epoch', y='recall_red_e', label='Improved U-Net', marker='o', color=tokyo['cyan'], markeredgewidth=0, ax=ax[2])
	sns.lineplot(data=attention_df, x='epoch', y='recall_red_e', label='Attention U-Net', marker='o', color=tokyo['green'], markeredgewidth=0, ax=ax[2])
	ax[2].set_title(f"Recall for Red Channel {red_bullet}", fontsize=16, color=tokyo['red'])
	ax[2].set_xlabel("Epochs")
	ax[2].set_ylabel("")
	ax[2].legend(loc='lower right')

	# Save the plot
	plt.tight_layout()
	plt.savefig("images/metrics-red.png", dpi=600, transparent=True)
	plt.close()


# Plots for dice, precision, and recall over epochs for green channel
def plot_green_metrics(classic_df: pd.DataFrame, improved_df: pd.DataFrame, attention_df: pd.DataFrame) -> None:
	fig, ax = plt.subplots(1, 3, figsize=(18, 7))

	# Set all the axes xticks as the epochs
	epochs = classic_df['epoch'].values
	ax[0].set_xticks(epochs)
	ax[1].set_xticks(epochs)
	ax[2].set_xticks(epochs)

	# Set yticks from 0.0 to 1.0 on steps of 0.1
	ax[0].set_yticks(np.arange(0.0, 1.1, 0.1))
	ax[1].set_yticks(np.arange(0.0, 1.1, 0.1))
	ax[2].set_yticks(np.arange(0.0, 1.1, 0.1))

	# Set yrange [0,1] included
	ax[0].set_ylim([0, 1])
	ax[1].set_ylim([0, 1])
	ax[2].set_ylim([0, 1])

	# Dice
	sns.lineplot(data=classic_df, x='epoch', y='dice_green_e', label='Classic U-Net', marker='o', color=tokyo['red'], markeredgewidth=0, ax=ax[0])
	sns.lineplot(data=improved_df, x='epoch', y='dice_green_e', label='Improved U-Net', marker='o', color=tokyo['cyan'], markeredgewidth=0, ax=ax[0])
	sns.lineplot(data=attention_df, x='epoch', y='dice_green_e', label='Attention U-Net', marker='o', color=tokyo['green'], markeredgewidth=0, ax=ax[0])
	green_bullet = '\u2B24'  # Unicode for green circle
	ax[0].set_title(f"Dice for Green Channel {green_bullet}", fontsize=16, color=tokyo['green'])
	ax[0].set_xlabel("Epochs")
	ax[0].set_ylabel("")
	ax[0].legend(loc='lower right')

	# Precision
	sns.lineplot(data=classic_df, x='epoch', y='precision_green_e', label='Classic U-Net', marker='o', color=tokyo['red'], markeredgewidth=0, ax=ax[1])
	sns.lineplot(data=improved_df, x='epoch', y='precision_green_e', label='Improved U-Net', marker='o', color=tokyo['cyan'], markeredgewidth=0, ax=ax[1])
	sns.lineplot(data=attention_df, x='epoch', y='precision_green_e', label='Attention U-Net', marker='o', color=tokyo['green'], markeredgewidth=0, ax=ax[1])
	ax[1].set_title(f"Precision for Green Channel {green_bullet}", fontsize=16, color=tokyo['green'])
	ax[1].set_xlabel("Epochs")
	ax[1].set_ylabel("")
	ax[1].legend(loc='lower right')

	# Recall
	sns.lineplot(data=classic_df, x='epoch', y='recall_green_e', label='Classic U-Net', marker='o', color=tokyo['red'], markeredgewidth=0, ax=ax[2])
	sns.lineplot(data=improved_df, x='epoch', y='recall_green_e', label='Improved U-Net', marker='o', color=tokyo['cyan'], markeredgewidth=0, ax=ax[2])
	sns.lineplot(data=attention_df, x='epoch', y='recall_green_e', label='Attention U-Net', marker='o', color=tokyo['green'], markeredgewidth=0, ax=ax[2])
	ax[2].set_title(f"Recall for Green Channel {green_bullet}", fontsize=16, color=tokyo['green'])
	ax[2].set_xlabel("Epochs")
	ax[2].set_ylabel("")
	ax[2].legend(loc='lower right')

	# Save the plot
	plt.tight_layout()
	plt.savefig("images/metrics-green.png", dpi=600, transparent=True)
	plt.close()


# Plots for dice, precision, and recall over epochs for blue channel
def plot_blue_metrics(classic_df: pd.DataFrame, improved_df: pd.DataFrame, attention_df: pd.DataFrame) -> None:
	fig, ax = plt.subplots(1, 3, figsize=(18, 7))

	# Set all the axes xticks as the epochs
	epochs = classic_df['epoch'].values
	ax[0].set_xticks(epochs)
	ax[1].set_xticks(epochs)
	ax[2].set_xticks(epochs)

	# Set yticks from 0.0 to 1.0 on steps of 0.1
	ax[0].set_yticks(np.arange(0.0, 1.1, 0.1))
	ax[1].set_yticks(np.arange(0.0, 1.1, 0.1))
	ax[2].set_yticks(np.arange(0.0, 1.1, 0.1))

	# Set yrange [0,1] included
	ax[0].set_ylim([0, 1])
	ax[1].set_ylim([0, 1])
	ax[2].set_ylim([0, 1])

	# Dice
	sns.lineplot(data=classic_df, x='epoch', y='dice_blue_e', label='Classic U-Net', marker='o', color=tokyo['red'], markeredgewidth=0, ax=ax[0])
	sns.lineplot(data=improved_df, x='epoch', y='dice_blue_e', label='Improved U-Net', marker='o', color=tokyo['cyan'], markeredgewidth=0, ax=ax[0])
	sns.lineplot(data=attention_df, x='epoch', y='dice_blue_e', label='Attention U-Net', marker='o', color=tokyo['green'], markeredgewidth=0, ax=ax[0])
	blue_bullet = '\u2B24'  # Unicode for blue circle
	ax[0].set_title(f"Dice for Blue Channel {blue_bullet}", fontsize=16, color=tokyo['blue'])
	ax[0].set_xlabel("Epochs")
	ax[0].set_ylabel("")
	ax[0].legend(loc='lower right')

	# Precision
	sns.lineplot(data=classic_df, x='epoch', y='precision_blue_e', label='Classic U-Net', marker='o', color=tokyo['red'], markeredgewidth=0, ax=ax[1])
	sns.lineplot(data=improved_df, x='epoch', y='precision_blue_e', label='Improved U-Net', marker='o', color=tokyo['cyan'], markeredgewidth=0, ax=ax[1])
	sns.lineplot(data=attention_df, x='epoch', y='precision_blue_e', label='Attention U-Net', marker='o', color=tokyo['green'], markeredgewidth=0, ax=ax[1])
	ax[1].set_title(f"Precision for Blue Channel {blue_bullet}", fontsize=16, color=tokyo['blue'])
	ax[1].set_xlabel("Epochs")
	ax[1].set_ylabel("")
	ax[1].legend(loc='lower right')

	# Recall
	sns.lineplot(data=classic_df, x='epoch', y='recall_blue_e', label='Classic U-Net', marker='o', color=tokyo['red'], markeredgewidth=0, ax=ax[2])
	sns.lineplot(data=improved_df, x='epoch', y='recall_blue_e', label='Improved U-Net', marker='o', color=tokyo['cyan'], markeredgewidth=0, ax=ax[2])
	sns.lineplot(data=attention_df, x='epoch', y='recall_blue_e', label='Attention U-Net', marker='o', color=tokyo['green'], markeredgewidth=0, ax=ax[2])
	ax[2].set_title(f"Recall for Blue Channel {blue_bullet}", fontsize=16, color=tokyo['blue'])
	ax[2].set_xlabel("Epochs")
	ax[2].set_ylabel("")
	ax[2].legend(loc='lower right')

	# Save the plot
	plt.tight_layout()
	plt.savefig("images/metrics-blue.png", dpi=600, transparent=True)
	plt.close()


# Plots for dice, precision, and recall over epochs for average channel
def plot_average_metrics(classic_df: pd.DataFrame, improved_df: pd.DataFrame, attention_df: pd.DataFrame) -> None:
	fig, ax = plt.subplots(1, 3, figsize=(18, 7))

	# Set all the axes xticks as the epochs
	epochs = classic_df['epoch'].values
	ax[0].set_xticks(epochs)
	ax[1].set_xticks(epochs)
	ax[2].set_xticks(epochs)

	# Set yticks from 0.0 to 1.0 on steps of 0.1
	ax[0].set_yticks(np.arange(0.0, 1.1, 0.1))
	ax[1].set_yticks(np.arange(0.0, 1.1, 0.1))
	ax[2].set_yticks(np.arange(0.0, 1.1, 0.1))

	# Set yrange [0,1] included
	ax[0].set_ylim([0, 1])
	ax[1].set_ylim([0, 1])
	ax[2].set_ylim([0, 1])

	# Dice
	sns.lineplot(data=classic_df, x='epoch', y='dice_average_e', label='Classic U-Net', marker='o', color=tokyo['red'], markeredgewidth=0, ax=ax[0])
	sns.lineplot(data=improved_df, x='epoch', y='dice_average_e', label='Improved U-Net', marker='o', color=tokyo['cyan'], markeredgewidth=0, ax=ax[0])
	sns.lineplot(data=attention_df, x='epoch', y='dice_average_e', label='Attention U-Net', marker='o', color=tokyo['green'], markeredgewidth=0, ax=ax[0])
	ax[0].set_title(f"Dice Average ", fontsize=16)
	ax[0].set_xlabel("Epochs")
	ax[0].set_ylabel("")
	ax[0].legend(loc='lower right')

	# Precision
	sns.lineplot(data=classic_df, x='epoch', y='precision_average_e', label='Classic U-Net', marker='o', color=tokyo['red'], markeredgewidth=0, ax=ax[1])
	sns.lineplot(data=improved_df, x='epoch', y='precision_average_e', label='Improved U-Net', marker='o', color=tokyo['cyan'], markeredgewidth=0, ax=ax[1])
	sns.lineplot(data=attention_df, x='epoch', y='precision_average_e', label='Attention U-Net', marker='o', color=tokyo['green'], markeredgewidth=0, ax=ax[1])
	ax[1].set_title(f"Precision Average ", fontsize=16)
	ax[1].set_xlabel("Epochs")
	ax[1].set_ylabel("")
	ax[1].legend(loc='lower right')

	# Recall
	sns.lineplot(data=classic_df, x='epoch', y='recall_average_e', label='Classic U-Net', marker='o', color=tokyo['red'], markeredgewidth=0, ax=ax[2])
	sns.lineplot(data=improved_df, x='epoch', y='recall_average_e', label='Improved U-Net', marker='o', color=tokyo['cyan'], markeredgewidth=0, ax=ax[2])
	sns.lineplot(data=attention_df, x='epoch', y='recall_average_e', label='Attention U-Net', marker='o', color=tokyo['green'], markeredgewidth=0, ax=ax[2])
	ax[2].set_title(f"Recall Average ", fontsize=16)
	ax[2].set_xlabel("Epochs")
	ax[2].set_ylabel("")
	ax[2].legend(loc='lower right')

	# Save the plot
	plt.tight_layout()
	plt.savefig("images/metrics-average.png", dpi=600, transparent=True)
	plt.close()
