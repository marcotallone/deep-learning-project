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
from torch.optim.lr_scheduler import StepLR

# Model import
from models.vit import VisionTransformer

# Utils imports
from utils.datasets import load_classification
from utils.train import train_multiclass, get_device
from utils.analysis import count_parameters


# Hyperparameters --------------------------------------------------------------
print("\nSetting hyperparameters...")

IMG_SIZE: int       = 128
EPOCHS: int         = 50
BATCH_TRAIN: int    = 64
BATCH_VALID: int    = 64
LR: float           = 1e-4
WEIGHT_DECAY: float = 1e-5
GAMMA: float        = 0.5
STEP: int           = 10

print(f"Image Size:   {IMG_SIZE}")
print(f"Epochs:       {EPOCHS}")
print(f"Train Batch:  {BATCH_TRAIN}")
print(f"Valid Batch:  {BATCH_VALID}")
print(f"LR:           {LR}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Gamma:        {GAMMA}")
print(f"Step:         {STEP}")


# Other training options -------------------------------------------------------
# Resume previous training from checkpoint
RESUME_TRAINING: bool = False
if RESUME_TRAINING:
    START_EPOCH: int = 10
else:
    START_EPOCH: int = 0

print(f"Resume Training: {RESUME_TRAINING}")
print(f"Start Epoch:     {START_EPOCH}")

# Loss functions
TRAIN_CRITERION: Union[th.nn.Module, Callable[[th.Tensor], th.Tensor]] = (
    th.nn.CrossEntropyLoss(reduction="mean")
)
EVAL_CRITERION: Union[th.nn.Module, Callable[[th.Tensor], th.Tensor]] = (
    th.nn.CrossEntropyLoss(reduction="sum")
)


# Datasets and CHeckpoints directories (relative to the project root) ----------
DATASETS: str = "datasets"
CLASSIFICATION: str = os.path.join(DATASETS, "classification")
TEST_DIR: str = os.path.join(CLASSIFICATION, "Testing")
TRAIN_DIR: str = os.path.join(CLASSIFICATION, "Training")
SAVE_MODELS_PATH: str = f"models/saved_models"
SAVE_METRICS_PATH: str = f"models/saved_metrics"

# Create the directories if they do not exist
os.makedirs(SAVE_MODELS_PATH, exist_ok=True)
os.makedirs(SAVE_METRICS_PATH, exist_ok=True)


# Device setup -----------------------------------------------------------------
print("\nSetting up the device...")

device: th.device = get_device()
print(f"Using device: {device}")


# Load the classification dataset ----------------------------------------------
print("\nLoading the classification dataset...")

# Load the dataset
train_dataloader, valid_dataloader = load_classification(
    train_directory=TRAIN_DIR,
    valid_directory=TEST_DIR,
    train_batch_size=BATCH_TRAIN,
    valid_batch_size=BATCH_VALID,
    resize=(IMG_SIZE, IMG_SIZE),
    shuffle=True,
    data_augmentation=True
)

# Count the total number of examples in the datasets
total_examples: int = len(train_dataloader.dataset) + len(valid_dataloader.dataset)
train_examples: int = len(train_dataloader.dataset)
valid_examples: int = len(valid_dataloader.dataset)
train_split: float = (train_examples / total_examples) * 100
valid_split: float = (valid_examples / total_examples) * 100
print(f"Total examples:      {total_examples} examples")
print(f"Train-Test split:    {train_split:.0f}% - {valid_split:.0f}%")
print(f"Train-Test examples: {train_examples} - {valid_examples}")

# Verify image sizes
for images, labels in train_dataloader:
    print("Training batch   - Images shape:", images.shape, "Labels shape:", labels.shape)
    break
for images, labels in valid_dataloader:
    print("Validation batch - Images shape:", images.shape, "Labels shape:", labels.shape)
    break


# Define the model -------------------------------------------------------------
print("\nBuilding the model...")

# Select and initialize the U-Net model
model: th.nn.Module = VisionTransformer(
    img_size=IMG_SIZE,
    patch_size=16,
    output_classes=4,
    dim=512,
    depth=10,
    heads=8,
    mlp_dim=1024,
    dropout=0.2
).to(device)

print(f"Model: {model.name}")

# Count the total number of parameters
total_params = count_parameters(model)
print(f"Total Parameters: {total_params}")


# Define the optimizer and scheduler -------------------------------------------
optimizer: Optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler: StepLR = StepLR(optimizer, step_size=STEP, gamma=GAMMA)

_ = train_multiclass(
    model=model,
    loss_fns=(TRAIN_CRITERION, EVAL_CRITERION),
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

