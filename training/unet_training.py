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
from models.improved_unet import ImprovedUNet
from models.attention_unet import AttentionUNet

# Dataset loader
from utils.datasets import load_segmentation

# Utils imports
from utils.train import train_unet
from utils.analysis import count_parameters


# Datasets directories (relative to the project root) --------------------------
DATASETS: str = "datasets"
SEGMENTATION: str = os.path.join(DATASETS, "segmentation/data")
SAVE_PATH: str = "models/saved_models"
SAVE_METRICS_PATH: str = "models/saved_metrics"

# Create the directories if they do not exist
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_METRICS_PATH, exist_ok=True)


# Hyperparameters --------------------------------------------------------------
print("\nSetting hyperparameters...")
DEVICE_AUTODETECT: bool = True
PERCENTAGE: float = 0.5
SPLIT: int = 0.7
IMG_SIZE: int = 128
N_FILTERS: int = 16
BATCH_TRAIN: int = 64
BATCH_VALID: int = 64
EPOCHS: int = 10
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

# Stop here (debug)
sys.exit(0)

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
train_losses, valid_losses, dice_scores, iou_scores, accuracy_scores, fpr_scores, fnr_scores, precision_scores, recall_scores = train_unet(
    model=model,
    loss_fn=CRITERION,
    optimizer=optimizer,
    train_loader=train_dataloader,
    valid_loader=valid_dataloader,
    n_epochs=EPOCHS,
    device=device,
    save_path=SAVE_PATH
)


# Save losses and metrics in a dataframe ---------------------------------------
data = {
    'epoch': list(range(1, len(train_losses) + 1)),
    'train_loss': train_losses,
    'validation_loss': valid_losses,
    'dice_red': dice_scores[0][0], 'dice_green': dice_scores[0][1], 'dice_blue': dice_scores[0][2], 'dice_average': dice_scores[0][3],
    'iou_red': iou_scores[0][0], 'iou_green': iou_scores[0][1], 'iou_blue': iou_scores[0][2], 'iou_average': iou_scores[0][3],
    'accuracy_red': accuracy_scores[0][0], 'accuracy_green': accuracy_scores[0][1], 'accuracy_blue': accuracy_scores[0][2], 'accuracy_average': accuracy_scores[0][3],
    'fpr_red': fpr_scores[0][0], 'fpr_green': fpr_scores[0][1], 'fpr_blue': fpr_scores[0][2], 'fpr_average': fpr_scores[0][3],
    'fnr_red': fnr_scores[0][0], 'fnr_green': fnr_scores[0][1], 'fnr_blue': fnr_scores[0][2], 'fnr_average': fnr_scores[0][3],
    'precision_red': precision_scores[0][0], 'precision_green': precision_scores[0][1], 'precision_blue': precision_scores[0][2], 'precision_average': precision_scores[0][3],
    'recall_red': recall_scores[0][0], 'recall_green': recall_scores[0][1], 'recall_blue': recall_scores[0][2], 'recall_average': recall_scores[0][3]
}
df = pd.DataFrame(data)
model_name = model.module.name if isinstance(model, th.nn.DataParallel) else model.name
df.to_csv(os.path.join(SAVE_METRICS_PATH, f"metrics_{model_name}_e{EPOCHS}.csv"), index=False)


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
	'Dice': dice_final,
	'IoU': iou_final,
	'Accuracy': accuracy_final,
	'FPR': fpr_final,
	'FNR': fnr_final,
	'Precision': precision_final,
	'Recall': recall_final
}
rows = ['Red', 'Green', 'Blue', 'Average']
df = pd.DataFrame(metrics, index=rows)
pd.options.display.float_format = '{:.2f}'.format

print("\nPerformance metrics (%)")
print("------------------------------------------------------------")
print(df)