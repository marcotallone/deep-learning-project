# Training second unet model for segmentation task on brats2020 dataset

# Imports ----------------------------------------------------------------------

# Common Python imports
import os
import numpy as np
import pandas as pd
from datetime import datetime

# h5py to import h5 dataset files
import h5py

# Torch imports
import torch as th
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Typining hints
from typing import List, Union, Callable, Tuple

# Utils imports
from utils.datasets import BrainScanDataset
from utils.train import train_unet
from models.unet2 import UNet2
from utils.analysis import count_parameters


# Hyperparameters ---------------------------------------------------------------
DEVICE_AUTODETECT: bool = True
BATCH_TRAIN: int = 32
BATCH_VALID: int = 32
EPOCHS: int = 12
# CRITERION: Union[th.nn.Module, Callable[[th.Tensor], th.Tensor]] = (
# 	th.nn.CrossEntropyLoss(reduction="mean")
# )
CRITERION: Union[th.nn.Module, Callable[[th.Tensor, th.Tensor], th.Tensor]] = (
    th.nn.BCEWithLogitsLoss()
)
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-2

# Device setup
device: th.device = th.device(
    "cuda" if th.cuda.is_available() and DEVICE_AUTODETECT else "cpu"
)

print("U-Net for Brain Tumor Segmentation Started")
print("==========================================")
print(f"\nUsing device: {device}")

# Load the BraTS2020 dataset ---------------------------------------------------
DATASETS_DIR: str = "../datasets/segmentation/data"
print("\nLoading the BraTS2020 dataset...")

# Build .h5 file paths from directory containing .h5 files
h5_files = [os.path.join(DATASETS_DIR, f) for f in os.listdir(DATASETS_DIR) if f.endswith('.h5')]
np.random.seed(42)
np.random.shuffle(h5_files)

# Split the dataset into train and validation sets (90:10)
split_idx = int(0.9 * len(h5_files))
train_files = h5_files[:split_idx]
valid_files = h5_files[split_idx:]

# Create the train and val datasets
train_dataset = BrainScanDataset(train_files)
valid_dataset = BrainScanDataset(valid_files, deterministic=True)

# Sample dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_TRAIN, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_VALID, shuffle=False)

# Use this to generate test images to view later
# test_input_iterator = iter(DataLoader(valid_dataset, batch_size=1, shuffle=False))

# Define and train the model ---------------------------------------------------
print("\nBuilding the model...")

# Initialize the U-Net model
model: th.nn.Module = UNet2()
model.to(device)
total_params = count_parameters(model)
print(f'Total Parameters: {total_params:,}\n')

# Initialize the optimizer
optimizer: th.optim.Optimizer = th.optim.Adam(model.parameters(), lr=LR)

# Train the model
train_loss_history, valid_loss_history = train_unet(
    model, CRITERION, optimizer, train_dataloader, valid_dataloader, EPOCHS, device
)

# Save losses to a file
data = {
    'epoch': list(range(1, len(train_loss_history) + 1)),
    'train': train_loss_history,
    'validation': valid_loss_history
}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
now = datetime.now()
df.to_csv(f"loss_history_{now.strftime('%d%m%Y%H%M%S')}.csv", index=False)

# Print final losses
print("\nFinal losses:")
print(f"Train loss: {train_loss_history[-1]:.4f}")
print(f"Validation loss: {valid_loss_history[-1]:.4f}")