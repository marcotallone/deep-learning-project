# U-Net models training script for segmentation task on the BraTS2020 dataset


# Imports ----------------------------------------------------------------------
print("\nImporting libraries...")

# Common Python imports
import os
import sys
from token import PERCENT
import pandas as pd

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch imports
import torch as th

# Typining hints
from typing import Union, Callable

# Model import
from models.classic_unet import ClassicUNet
# from models.improved_unet import ImprovedUNet
# from models.attention_unet import AttentionUNet

# Dataset loader
from utils.datasets import load_segmentation

# Utils imports
from utils.train import train_unet
from utils.analysis import count_parameters


# Datasets directories (relative to the project root) --------------------------
DATASETS: str = "datasets"
SEGMENTATION: str = os.path.join(DATASETS, "segmentation/data")
SAVE_PATH: str = "models/saved_models"


# Hyperparameters --------------------------------------------------------------
print("\nSetting hyperparameters...")
DEVICE_AUTODETECT: bool = True
PERCENTAGE: float = 0.1
SPLIT: int = 0.6
IMG_SIZE: int = 128
N_FILTERS: int = 2
BATCH_TRAIN: int = 64
BATCH_VALID: int = 64
EPOCHS: int = 1
CRITERION: Union[th.nn.Module, Callable[[th.Tensor, th.Tensor], th.Tensor]] = (
    th.nn.BCEWithLogitsLoss()
)
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-2


# Device setup -----------------------------------------------------------------
print("\nSetting up the device...")
device: th.device = th.device(
    "cuda" if th.cuda.is_available() and DEVICE_AUTODETECT else "cpu"
)
print(f"Using device: {device}")


# Load the BraTS2020 dataset ---------------------------------------------------
print("\nLoading the BraTS2020 dataset...")

# Load the dataset
train_dataloader, valid_dataloader = load_segmentation(
    directory = SEGMENTATION,
    split = SPLIT,
    train_batch_size = BATCH_TRAIN,
    valid_batch_size = BATCH_VALID,
    percentage = PERCENTAGE,
    resize = (IMG_SIZE, IMG_SIZE),
    deterministic = True
)

# Count examples in the dataset
print(f"Total examples:      {len(train_dataloader.dataset) + len(valid_dataloader.dataset)} examples")
print(f"Train-Test split:    {SPLIT*100:.0f}% - {(1-SPLIT)*100:.0f}%") 
print(f"Train-Test examples: {len(train_dataloader.dataset)} - {len(valid_dataloader.dataset)}")

# Verify image sizes
for images, masks in train_dataloader:
    print("Training batch   - Images shape:", images.shape, "Masks shape:", masks.shape)
    break
for images, masks in valid_dataloader:
    print("Validation batch - Images shape:", images.shape, "Masks shape:", masks.shape)
    break


# Define and train the model ---------------------------------------------------
print("\nBuilding the model...")

# Select and initialize the U-Net model
model: th.nn.Module = ClassicUNet(n_filters=N_FILTERS)
# model: th.nn.Module = ImprovedUNet(n_filters=N_FILTERS)
# model: th.nn.Module = AttentionUNet(n_filters=N_FILTERS)

# Move the model to the device (or devices)
if th.cuda.device_count() > 1:
    print(f"Using {th.cuda.device_count()} GPUs")
    model = th.nn.DataParallel(model)
model.to(device)

# Count the total number of parameters
total_params = count_parameters(model)
print(f'Total Parameters: {total_params:,}')

# Initialize the optimizer
optimizer: th.optim.Optimizer = th.optim.Adam(model.parameters(), lr=LR)

# Train the model
train_loss_history, valid_loss_history = train_unet(
    model=model,
    loss_fn=CRITERION,
    optimizer=optimizer,
    train_loader=train_dataloader,
    valid_loader=valid_dataloader,
    n_epochs=EPOCHS,
    device=device,
    save_path=SAVE_PATH
)


# Save losses ------------------------------------------------------------------
data = {
    'epoch': list(range(1, len(train_loss_history) + 1)),
    'train': train_loss_history,
    'validation': valid_loss_history
}
df = pd.DataFrame(data)
df.to_csv(f"loss_history_{model.name}.csv", index=False)

# Print final losses
print("\nFinal losses:")
print(f"Train loss: {train_loss_history[-1]:.4f}")
print(f"Validation loss: {valid_loss_history[-1]:.4f}")
