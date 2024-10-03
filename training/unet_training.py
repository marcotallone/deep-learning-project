# U-Net models training script for segmentation task on the BraTS2020 dataset


# Imports ----------------------------------------------------------------------
print("\nImporting libraries...")

# Common Python imports
import os
import sys

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Typining hints
from typing import Callable, Union, List

# Torch imports
import torch as th
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LRScheduler, ExponentialLR
# from safetensors.torch import load_model, load_file

# Model import
from models.classic_unet import ClassicUNet
from models.improved_unet import ImprovedUNet
from models.attention_unet import AttentionUNet

# Utils imports
from utils.datasets import load_segmentation
from utils.train import train_unet
from utils.analysis import count_parameters


# Hyperparameters --------------------------------------------------------------
print("\nSetting hyperparameters...")

PERCENTAGE: float   = 0.5
SPLIT: float        = 0.9
N_FILTERS: int      = 32
IMG_SIZE: int       = 240
EPOCHS: int         = 10
BATCH_TRAIN: int    = 32
BATCH_VALID: int    = 32
LR: float           = 2e-3
WEIGHT_DECAY: float = 1e-2
GAMMA: float        = 0.9

print(f"Percentage:   {PERCENTAGE}")
print(f"Split:        {SPLIT}")
print(f"N Filters:    {N_FILTERS}")
print(f"Image Size:   {IMG_SIZE}")
print(f"Epochs:       {EPOCHS}")
print(f"Train Batch:  {BATCH_TRAIN}")
print(f"Valid Batch:  {BATCH_VALID}")
print(f"LR:           {LR}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Gamma:        {GAMMA}")


# Other training options -------------------------------------------------------
# Resume previous training from checkpoint
RESUME_TRAINING: bool = False
if RESUME_TRAINING:
    START_EPOCH: int = 10
else:
    START_EPOCH: int = 0

print(f"Resume Training: {RESUME_TRAINING}")
print(f"Start Epoch:     {START_EPOCH}")

# Loss function
CRITERION: Union[th.nn.Module, Callable[[th.Tensor, th.Tensor], th.Tensor]] = (
    th.nn.BCEWithLogitsLoss()
)


# Datasets and Checkpoints directories (relative to the project root) ----------
DATASETS: str = "datasets"
SEGMENTATION: str = os.path.join(DATASETS, "segmentation/data")
SAVE_MODELS_PATH: str = f"models/saved_models"
SAVE_METRICS_PATH: str = f"models/saved_metrics"

# Create the directories if they do not exist
os.makedirs(SAVE_MODELS_PATH, exist_ok=True)
os.makedirs(SAVE_METRICS_PATH, exist_ok=True)


# Device setup -----------------------------------------------------------------
print("\nSetting up the device...")

device: th.device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load the BraTS2020 dataset ---------------------------------------------------
print("\nLoading the BraTS2020 dataset...")

# Load the dataset
train_dataloader, valid_dataloader = load_segmentation(
    directory=SEGMENTATION,
    split=SPLIT,
    train_batch_size=BATCH_TRAIN,
    valid_batch_size=BATCH_VALID,
    percentage=PERCENTAGE,
    resize=(IMG_SIZE, IMG_SIZE),
    deterministic=True,
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


# Define the model -------------------------------------------------------------
print("\nBuilding the model...")

# Select and initialize the U-Net model
# model: th.nn.Module = ClassicUNet(n_filters=N_FILTERS).to(device)
# model: th.nn.Module = ImprovedUNet(n_filters=N_FILTERS).to(device)
model: th.nn.Module = AttentionUNet(n_filters=N_FILTERS).to(device)
print(f"Model: {model.name}")

# Count the total number of parameters
total_params = count_parameters(model)
print(f"Total Parameters: {total_params}")


# Define the optimizer and scheduler -------------------------------------------
optimizer: Optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler: LRScheduler = ExponentialLR(optimizer=optimizer, gamma=GAMMA)

_ = train_unet(
    model=model,
    loss_fn=CRITERION,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_dataloader,
    valid_loader=valid_dataloader,
    n_epochs=EPOCHS,
    start_epoch=START_EPOCH,
    device=device,
    save_path=SAVE_MODELS_PATH,
    save_metrics_path=SAVE_METRICS_PATH
)
