# U-Net models training script for segmentation task on the BraTS2020 dataset


# Imports ----------------------------------------------------------------------
print("\nImporting libraries...")

# Common Python imports
import os
from re import L
import sys
from tkinter import LAST

import pandas as pd

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Typining hints
from typing import Callable, Union, List

# Torch imports
import torch as th
from safetensors.torch import load_model, load_file

# Model import
from models.classic_unet import ClassicUNet
from models.improved_unet import ImprovedUNet
from models.attention_unet import AttentionUNet

# Dataset loader
from utils.datasets import load_segmentation

# Utils imports
from utils.train import train_unet
from utils.analysis import count_parameters, remove_module_prefix

# Hyperparameters --------------------------------------------------------------
print("\nSetting hyperparameters...")
DEVICE_AUTODETECT: bool = True
PERCENTAGE: float = 0.5
SPLIT: float = 0.9
IMG_SIZE: int = 240
N_FILTERS: int = 64
BATCH_TRAIN: int = 64
BATCH_VALID: int = 64
EPOCHS: int = 10
CRITERION: Union[th.nn.Module, Callable[[th.Tensor, th.Tensor], th.Tensor]] = (
    th.nn.BCEWithLogitsLoss()
)
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-2

# Datasets directories (relative to the project root) --------------------------
DATASETS: str = "datasets"
SEGMENTATION: str = os.path.join(DATASETS, "segmentation/data")
SAVE_PATH: str = f"models/saved_models_e{EPOCHS}_n{N_FILTERS}_{IMG_SIZE}"
SAVE_METRICS_PATH: str = f"models/saved_metrics_e{EPOCHS}_n{N_FILTERS}_{IMG_SIZE}"

# Create the directories if they do not exist
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_METRICS_PATH, exist_ok=True)

# Pre-load interrupted training
LOAD_SAVED_MODEL: bool = False
SAVED_MODEL_FILE: str = os.path.join(SAVE_PATH, "AttentionUNet_e3.pth")
LAST_EPOCH: int = 0

if not LOAD_SAVED_MODEL: # Set the last epoch to 0 if not loading a saved model
    LAST_EPOCH = 0

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
    directory=SEGMENTATION,
    split=SPLIT,
    train_batch_size=BATCH_TRAIN,
    valid_batch_size=BATCH_VALID,
    percentage=PERCENTAGE,
    resize=(IMG_SIZE, IMG_SIZE),
    deterministic=True,
)

# Count examples in the dataset
print(
    f"Total examples:      {len(train_dataloader.dataset) + len(valid_dataloader.dataset)} examples"
)
print(f"Train-Test split:    {SPLIT*100:.0f}% - {(1-SPLIT)*100:.0f}%")
print(
    f"Train-Test examples: {len(train_dataloader.dataset)} - {len(valid_dataloader.dataset)}"
)

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
# model: th.nn.Module = ClassicUNet(n_filters=N_FILTERS)
# model: th.nn.Module = ImprovedUNet(n_filters=N_FILTERS)
model: th.nn.Module = AttentionUNet(n_filters=N_FILTERS)

# Move the model to the device (or devices)
if th.cuda.device_count() > 1:
    print(f"Using {th.cuda.device_count()} GPUs")
    model = th.nn.DataParallel(model)
model.to(device)

# Count the total number of parameters
total_params = count_parameters(model)
print(f"Total Parameters: {total_params:,}")

# Initialize the optimizer
optimizer: th.optim.Optimizer = th.optim.Adam(model.parameters(), lr=LR)

# Load a saved model if required
if LOAD_SAVED_MODEL:
    saved_dict = load_file(SAVED_MODEL_FILE)
    model_dict = remove_module_prefix(saved_dict)
    try:
        model.load_state_dict(model_dict)
        print(f"Model loaded from {SAVED_MODEL_FILE}")
    except Exception as e:
        print(f"Model not loaded: {e}")

# Train the model
(
    train_losses,
    valid_losses,
    dice_scores,
    iou_scores,
    accuracy_scores,
    fpr_scores,
    fnr_scores,
    precision_scores,
    recall_scores,
) = train_unet(
    model=model,
    loss_fn=CRITERION,
    optimizer=optimizer,
    train_loader=train_dataloader,
    valid_loader=valid_dataloader,
    n_epochs=EPOCHS,
    start_epoch=LAST_EPOCH,
    device=device,
    save_path=SAVE_PATH,
)


# Save losses and metrics in a dataframe ---------------------------------------

# Create lists
dice_red: List[float] = []
dice_green: List[float] = []
dice_blue: List[float] = []
dice_average: List[float] = []
iou_red: List[float] = []
iou_green: List[float] = []
iou_blue: List[float] = []
iou_average: List[float] = []
accuracy_red: List[float] = []
accuracy_green: List[float] = []
accuracy_blue: List[float] = []
accuracy_average: List[float] = []
fpr_red: List[float] = []
fpr_green: List[float] = []
fpr_blue: List[float] = []
fpr_average: List[float] = []
fnr_red: List[float] = []
fnr_green: List[float] = []
fnr_blue: List[float] = []
fnr_average: List[float] = []
precision_red: List[float] = []
precision_green: List[float] = []
precision_blue: List[float] = []
precision_average: List[float] = []
recall_red: List[float] = []
recall_green: List[float] = []
recall_blue: List[float] = []
recall_average: List[float] = []

# Fill the lists
for i in range(EPOCHS-LAST_EPOCH):
    dice_red.append(dice_scores[i][0])
    dice_green.append(dice_scores[i][1])
    dice_blue.append(dice_scores[i][2])
    dice_average.append(dice_scores[i][3])
    iou_red.append(iou_scores[i][0])
    iou_green.append(iou_scores[i][1])
    iou_blue.append(iou_scores[i][2])
    iou_average.append(iou_scores[i][3])
    accuracy_red.append(accuracy_scores[i][0])
    accuracy_green.append(accuracy_scores[i][1])
    accuracy_blue.append(accuracy_scores[i][2])
    accuracy_average.append(accuracy_scores[i][3])
    fpr_red.append(fpr_scores[i][0])
    fpr_green.append(fpr_scores[i][1])
    fpr_blue.append(fpr_scores[i][2])
    fpr_average.append(fpr_scores[i][3])
    fnr_red.append(fnr_scores[i][0])
    fnr_green.append(fnr_scores[i][1])
    fnr_blue.append(fnr_scores[i][2])
    fnr_average.append(fnr_scores[i][3])
    precision_red.append(precision_scores[i][0])
    precision_green.append(precision_scores[i][1])
    precision_blue.append(precision_scores[i][2])
    precision_average.append(precision_scores[i][3])
    recall_red.append(recall_scores[i][0])
    recall_green.append(recall_scores[i][1])
    recall_blue.append(recall_scores[i][2])
    recall_average.append(recall_scores[i][3])


data = {
    "epoch": list(range(1, len(train_losses) + 1)),
    "train_loss": train_losses,
    "validation_loss": valid_losses,
    "dice_red": dice_red,
    "dice_green": dice_green,
    "dice_blue": dice_blue,
    "dice_average": dice_average,
    "iou_red": iou_red,
    "iou_green": iou_green,
    "iou_blue": iou_blue,
    "iou_average": iou_average,
    "accuracy_red": accuracy_red,
    "accuracy_green": accuracy_green,
    "accuracy_blue": accuracy_blue,
    "accuracy_average": accuracy_average,
    "fpr_red": fpr_red,
    "fpr_green": fpr_green,
    "fpr_blue": fpr_blue,
    "fpr_average": fpr_average,
    "fnr_red": fnr_red,
    "fnr_green": fnr_green,
    "fnr_blue": fnr_blue,
    "fnr_average": fnr_average,
    "precision_red": precision_red,
    "precision_green": precision_green,
    "precision_blue": precision_blue,
    "precision_average": precision_average,
    "recall_red": recall_red,
    "recall_green": recall_green,
    "recall_blue": recall_blue,
    "recall_average": recall_average,
}
df = pd.DataFrame(data)
model_name = model.module.name if isinstance(model, th.nn.DataParallel) else model.name
df.to_csv(
    os.path.join(SAVE_METRICS_PATH, f"metrics_{model_name}_e{EPOCHS}.csv"), index=False
)


# Print final losses and metrics ----------------------------------------------
print("\nFinal losses:")
print(f"Train loss: {train_losses[-1]:.4f}")
print(f"Validation loss: {valid_losses[-1]:.4f}")

# Convert final metrics to percentage format
dice_final = [score * 100 for score in dice_scores[-1]]
iou_final = [score * 100 for score in iou_scores[-1]]
accuracy_final = [score * 100 for score in accuracy_scores[-1]]
fpr_final = [score * 100 for score in fpr_scores[-1]]
fnr_final = [score * 100 for score in fnr_scores[-1]]
precision_final = [score * 100 for score in precision_scores[-1]]
recall_final = [score * 100 for score in recall_scores[-1]]

# Create a DataFrame to display the metrics
metrics = {
    "Dice": dice_final,
    "IoU": iou_final,
    "Accuracy": accuracy_final,
    "FPR": fpr_final,
    "FNR": fnr_final,
    "Precision": precision_final,
    "Recall": recall_final,
}
rows = ["Red", "Green", "Blue", "Average"]
df = pd.DataFrame(metrics, index=rows)
pd.options.display.float_format = "{:.2f}".format

print("\nPerformance metrics (%)")
print("------------------------------------------------------------")
print(df)
