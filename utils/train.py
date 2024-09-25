# General training functions for PyTorch models

# Imports ----------------------------------------------------------------------

# Common Python imports
import os
import sys
import numpy as np
import pandas as pd
import tqdm as tqdm

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch imports
import torch as th
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

# Typining hints
from typing import List, Tuple

# Assessment metrics
from utils.metrics import *


# Save training at a given checkpoint ------------------------------------------
def save_checkpoint(model: Module,
                    optimizer: Optimizer,
                    scheduler: LRScheduler,
                    epoch: int,
                    save_path: str
) -> None:
    """Save a training checkpoint
    
    Parameters
    ----------
    model: th.nn.Module
        Model to save
    optimizer: Optimizer
        Optimizer to save
    scheduler: LRScheduler
        Scheduler to save
    epoch: int
        Epoch to save
    save_path: str
        Path to save the checkpoint
    """
    save_name: str = os.path.join(save_path, f"{model.name}_e{epoch}.pth")
    th.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }, save_name)


# Load training from a given checkpoint ----------------------------------------
def load_checkpoint(model: Module,
                    optimizer: Optimizer,
                    scheduler: LRScheduler,
                    save_path: str
) -> Tuple[int, Module, Optimizer, LRScheduler]:
    """Load a training checkpoint

    Parameters
    ----------
    model: th.nn.Module
        Model to load
    optimizer: Optimizer
        Optimizer to load
    scheduler: LRScheduler
        Scheduler to load
    save_path: str
        Path to load the checkpoint from

    Returns
    -------
    Tuple[int, th.nn.Module, Optimizer, LRScheduler]
        Epoch, model, optimizer, and scheduler loaded from the checkpoint
    """
    load_name: str = os.path.join(save_path, f"{model.name}_e{epoch}.pth")
    map_location = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    checkpoint = th.load(load_name, weights_only=False, map_location=map_location)
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return epoch, model, optimizer, scheduler

                    
# Training function for U-Net models -------------------------------------------
def train_unet(model: Module,
               loss_fn: Module,
               optimizer: Optimizer,
               scheduler: LRScheduler,
               train_loader: DataLoader,
               valid_loader: DataLoader,
               n_epochs: int,
               start_epoch: int = 0,
               device: th.device = th.device("cpu"),
               save_path: str = None,
               save_metrics_path: str = None
) -> pd.DataFrame:
    """Training function for U-Net models

    Parameters
    ----------
    model: th.nn.Module
        U-Net model to train
    loss_fn: th.nn.Module
        Loss function to use
    optimizer: Optimizer
        Optimizer to use
    train_loader: DataLoader
        DataLoader for the training set
    valid_loader: DataLoader
        DataLoader for the validation set
    n_epochs: int
        Number of epochs to train the model
    start_epoch: int, optional (default=0)
        Epoch to start training from (useful for resuming training)
    device: th.device, optional (default=th.device("cpu"))
        Device to use for training
    save_path: str, optional (default=None)
        Path to save the model weights
    save_metrics_path: str, optional (default=None)
        Path to save the training metrics

    Returns
    -------
    pd.DataFrame
        Dataframe containing the training losses, validation losses, Dice scores, 
        IoU scores, accuracy scores, FPR scores, FNR scores, precision scores, and recall scores
        for each epoch
    """
    
    # If given create the directory to store the model weights and metrics
    if save_path is not None: os.makedirs(save_path, exist_ok=True)
    if save_metrics_path is not None: os.makedirs(save_metrics_path, exist_ok=True)

    # Initialize the DataFrame for tracking metrics
    columns = ["epoch", "lr", "train_loss", "validation_loss",
               "dice_red", "dice_green", "dice_blue", "dice_average",
               "iou_red", "iou_green", "iou_blue", "iou_average", 
               "accuracy_red", "accuracy_green", "accuracy_blue", "accuracy_average", 
               "fpr_red", "fpr_green", "fpr_blue", "fpr_average", 
               "fnr_red", "fnr_green", "fnr_blue", "fnr_average",
               "precision_red", "precision_green", "precision_blue", "precision_average", 
               "recall_red", "recall_green", "recall_blue", "recall_average",
               "dice_red_e", "dice_green_e", "dice_blue_e", "dice_average_e",
                "iou_red_e", "iou_green_e", "iou_blue_e", "iou_average_e",
                "accuracy_red_e", "accuracy_green_e", "accuracy_blue_e", "accuracy_average_e",
                "fpr_red_e", "fpr_green_e", "fpr_blue_e", "fpr_average_e",
                "fnr_red_e", "fnr_green_e", "fnr_blue_e", "fnr_average_e",
                "precision_red_e", "precision_green_e", "precision_blue_e", "precision_average_e",
                "recall_red_e", "recall_green_e", "recall_blue_e", "recall_average_e"]
    metrics_df = pd.DataFrame(columns=columns)
    
    # Either resume training from a checkpoint or start from scratch
    if start_epoch > 0:
        load_name: str = os.path.join(save_path, f"{model.name}_e{start_epoch}.pth")
        checkpoint = th.load(load_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"\nModel loaded from checkpoint file {load_name}")
        print(f"Resuming training from epoch {start_epoch}")
    
    # Move the model to the device
    model.to(device)
    
    print("\nTraining the model...")

    # Loop over the epochs
    for epoch in tqdm.tqdm(range(start_epoch + 1, n_epochs + 1), desc="Epoch", position=0): 
        # or:   for epoch in range(start_epoch + 1, n_epochs + 1):

        # Training loop --------------------------------------------------------
        model.train() # Set the model to training mode
        dice_coeff: List[List[float]] = [] # Trach Dice coeff for each batch
        iou: List[List[float]] = [] # Track IoU for each batch
        accuracy: List[List[float]] = [] # Track accuracy for each batch
        fpr: List[List[float]] = [] # Track false positive rate for each batch
        fnr: List[List[float]] = [] # Track false negative rate for each batch
        precision: List[List[float]] = [] # Track precision for each batch
        recall: List[List[float]] = [] # Track recall for each batch
        train_loss: float = 0.0 # Track epoch loss
        for _, (x, y) in tqdm.tqdm(enumerate(train_loader), 
                                   desc="Training Batch", 
                                   position=1, 
                                   leave=False, 
                                   total=len(train_loader)):
            # or:   for x, y in train_loader:
            x, y = x.to(device), y.to(device) # Move the data to the device
            optimizer.zero_grad() # Zero out past gradients
            yhat: Tensor = model(x) # Forward pass
            loss: Tensor = loss_fn(yhat, y) # Loss computation
            train_loss += loss.item() # Track the loss
            dice_coeff.append(dice(yhat, y)) # Track the Dice coefficient
            iou.append(IoU(yhat, y)) # Track the IoU
            accuracy.append(accuracy2D(yhat, y)) # Track the accuracy
            fpr.append(fpr2D(yhat, y)) # Track the false positive rate
            fnr.append(fnr2D(yhat, y)) # Track the false negative rate
            precision.append(precision2D(yhat, y)) # Track the precision
            recall.append(recall2D(yhat, y)) # Track the recall
            loss.backward() # Backward pass
            optimizer.step() # Update the weights

        # Validation loop ------------------------------------------------------
        model.eval() # Set the model to evaluation mode
        valid_loss = 0.0 # Track validation loss
        dice_coeff_e: List[List[float]] = [] # Trach Dice coeff for each batch
        iou_e: List[List[float]] = [] # Track IoU for each batch
        accuracy_e: List[List[float]] = [] # Track accuracy for each batch
        fpr_e: List[List[float]] = [] # Track false positive rate for each batch
        fnr_e: List[List[float]] = [] # Track false negative rate for each batch
        precision_e: List[List[float]] = [] # Track precision for each batch
        recall_e: List[List[float]] = [] # Track recall for each batch
        with th.no_grad():
            for _, (x_e, y_e) in tqdm.tqdm(enumerate(valid_loader), 
                                           desc="Validation Batch", 
                                           position=2, 
                                           leave=False, 
                                           total=len(valid_loader)):
                # or:   for x_e, y_e in valid_loader:
                x_e, y_e = x_e.to(device), y_e.to(device) # Move the data to the device
                yhat_e: Tensor = model(x_e) # Forward pass
                loss_e: Tensor = loss_fn(yhat_e, y_e) # Loss computation
                valid_loss += loss_e.item() # Track the loss
                dice_coeff_e.append(dice(yhat_e, y_e)) # Track the Dice coefficient
                iou_e.append(IoU(yhat_e, y_e)) # Track the IoU
                accuracy_e.append(accuracy2D(yhat_e, y_e)) # Track the accuracy
                fpr_e.append(fpr2D(yhat_e, y_e)) # Track the false positive rate
                fnr_e.append(fnr2D(yhat_e, y_e)) # Track the false negative rate
                precision_e.append(precision2D(yhat_e, y_e)) # Track the precision
                recall_e.append(recall2D(yhat_e, y_e)) # Track the recall

        # Compute the averages of other metrics across all batches
        def compute_avg(metrics: List[List[Optional[float]]]) -> List[Optional[float]]:
            avg_metrics = []
            for i in range(len(metrics[0])):
                valid_metrics = [m[i] for m in metrics if m[i] is not None]
                if valid_metrics:
                    avg_metrics.append(sum(valid_metrics) / len(valid_metrics))
                else:
                    avg_metrics.append(None)
            return avg_metrics

        dice_avg = compute_avg(dice_coeff)
        iou_avg = compute_avg(iou)
        accuracy_avg = compute_avg(accuracy)
        fpr_avg = compute_avg(fpr)
        fnr_avg = compute_avg(fnr)
        precision_avg = compute_avg(precision)
        recall_avg = compute_avg(recall)
        dice_avg_e = compute_avg(dice_coeff_e)
        iou_avg_e = compute_avg(iou_e)
        accuracy_avg_e = compute_avg(accuracy_e)
        fpr_avg_e = compute_avg(fpr_e)
        fnr_avg_e = compute_avg(fnr_e)
        precision_avg_e = compute_avg(precision_e)
        recall_avg_e = compute_avg(recall_e)

        # Replace None with pd.NA
        def replace_none_with_na(metrics: List[Optional[float]]) -> List[Optional[float]]:
            return [m if m is not None else pd.NA for m in metrics] 
        
        dice_avg = replace_none_with_na(dice_avg)
        iou_avg = replace_none_with_na(iou_avg)
        accuracy_avg = replace_none_with_na(accuracy_avg)
        fpr_avg = replace_none_with_na(fpr_avg)
        fnr_avg = replace_none_with_na(fnr_avg)
        precision_avg = replace_none_with_na(precision_avg)
        recall_avg = replace_none_with_na(recall_avg)
        dice_avg_e = replace_none_with_na(dice_avg_e)
        iou_avg_e = replace_none_with_na(iou_avg_e)
        accuracy_avg_e = replace_none_with_na(accuracy_avg_e)
        fpr_avg_e = replace_none_with_na(fpr_avg_e)
        fnr_avg_e = replace_none_with_na(fnr_avg_e)
        precision_avg_e = replace_none_with_na(precision_avg_e)
        recall_avg_e = replace_none_with_na(recall_avg_e)

        # Create a new DataFrame for the current epoch's metrics
        epoch_metrics_df = pd.DataFrame([{
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation_loss": valid_loss,
            "dice_red": dice_avg[0],
            "dice_green": dice_avg[1],
            "dice_blue": dice_avg[2],
            "dice_average": dice_avg[3],
            "iou_red": iou_avg[0],
            "iou_green": iou_avg[1],
            "iou_blue": iou_avg[2],
            "iou_average": iou_avg[3],
            "accuracy_red": accuracy_avg[0],
            "accuracy_green": accuracy_avg[1],
            "accuracy_blue": accuracy_avg[2],
            "accuracy_average": accuracy_avg[3],
            "fpr_red": fpr_avg[0],
            "fpr_green": fpr_avg[1],
            "fpr_blue": fpr_avg[2],
            "fpr_average": fpr_avg[3],
            "fnr_red": fnr_avg[0],
            "fnr_green": fnr_avg[1],
            "fnr_blue": fnr_avg[2],
            "fnr_average": fnr_avg[3],
            "precision_red": precision_avg[0],
            "precision_green": precision_avg[1],
            "precision_blue": precision_avg[2],
            "precision_average": precision_avg[3],
            "recall_red": recall_avg[0],
            "recall_green": recall_avg[1],
            "recall_blue": recall_avg[2],
            "recall_average": recall_avg[3],
            "dice_red_e": dice_avg_e[0],
            "dice_green_e": dice_avg_e[1],
            "dice_blue_e": dice_avg_e[2],
            "dice_average_e": dice_avg_e[3],
            "iou_red_e": iou_avg_e[0],
            "iou_green_e": iou_avg_e[1],
            "iou_blue_e": iou_avg_e[2],
            "iou_average_e": iou_avg_e[3],
            "accuracy_red_e": accuracy_avg_e[0],
            "accuracy_green_e": accuracy_avg_e[1],
            "accuracy_blue_e": accuracy_avg_e[2],
            "accuracy_average_e": accuracy_avg_e[3],
            "fpr_red_e": fpr_avg_e[0],
            "fpr_green_e": fpr_avg_e[1],
            "fpr_blue_e": fpr_avg_e[2],
            "fpr_average_e": fpr_avg_e[3],
            "fnr_red_e": fnr_avg_e[0],
            "fnr_green_e": fnr_avg_e[1],
            "fnr_blue_e": fnr_avg_e[2],
            "fnr_average_e": fnr_avg_e[3],
            "precision_red_e": precision_avg_e[0],
            "precision_green_e": precision_avg_e[1],
            "precision_blue_e": precision_avg_e[2],
            "precision_average_e": precision_avg_e[3],
            "recall_red_e": recall_avg_e[0],
            "recall_green_e": recall_avg_e[1],
            "recall_blue_e": recall_avg_e[2],
            "recall_average_e": recall_avg_e[3]
        }])

        # Concatenate the new DataFrame with the existing one
        metrics_df = pd.concat([metrics_df, epoch_metrics_df], ignore_index=True)

        # Convert metrics to percentage format, handling pd.NA appropriately
        def convert_to_percentage(metrics: List[Optional[float]]) -> List[Optional[float]]:
            return [m * 100 if m is not pd.NA else pd.NA for m in metrics]
        
        dice_avg = convert_to_percentage(dice_avg)
        iou_avg = convert_to_percentage(iou_avg)
        accuracy_avg = convert_to_percentage(accuracy_avg)
        fpr_avg = convert_to_percentage(fpr_avg)
        fnr_avg = convert_to_percentage(fnr_avg)
        precision_avg = convert_to_percentage(precision_avg)
        recall_avg = convert_to_percentage(recall_avg)
        dice_avg_e = convert_to_percentage(dice_avg_e)
        iou_avg_e = convert_to_percentage(iou_avg_e)
        accuracy_avg_e = convert_to_percentage(accuracy_avg_e)
        fpr_avg_e = convert_to_percentage(fpr_avg_e)
        fnr_avg_e = convert_to_percentage(fnr_avg_e)
        precision_avg_e = convert_to_percentage(precision_avg_e)
        recall_avg_e = convert_to_percentage(recall_avg_e)

        # Create a DataFrame to display the metrics
        metrics = {
            "Dice": dice_avg,
            "IoU": iou_avg,
            "Accuracy": accuracy_avg,
            "FPR": fpr_avg,
            "FNR": fnr_avg,
            "Precision": precision_avg,
            "Recall": recall_avg
        }
        metrics_e = {
            "Dice": dice_avg_e,
            "IoU": iou_avg_e,
            "Accuracy": accuracy_avg_e,
            "FPR": fpr_avg_e,
            "FNR": fnr_avg_e,
            "Precision": precision_avg_e,
            "Recall": recall_avg_e
        }
        rows = ["Red", "Green", "Blue", "Average"]
        metrics_display = pd.DataFrame(metrics, index=rows)
        metrics_display_e = pd.DataFrame(metrics_e, index=rows)
        pd.options.display.float_format = "{:.2f}".format

        # Display current epoch metrics in a nice format
        print("------------------------------------------------------------")
        print(f"Epoch:              {epoch}/{n_epochs}")
        print(f"Training loss:      {train_loss:.4f}")
        print(f"Validation loss:    {valid_loss:.4f}")
        print(f"Current LR:         {optimizer.param_groups[0]['lr']:.2e}")
        print("\nTraining Metrics (%)")
        print("------------------------------------------------------------")
        print(metrics_display)
        print("------------------------------------------------------------")
        print("\nValidation Metrics (%)")
        print("------------------------------------------------------------")
        print(metrics_display_e)
        print("------------------------------------------------------------")

       # Save the DataFrame to a CSV file
        if save_metrics_path is not None:
            csv_path = os.path.join(save_metrics_path, f"metrics_{model.name}.csv")
        else:
            csv_path = f"metrics_{model_name}.csv"
        
        # Write new csv if it's the first epoch, otherwise append to the existing csv
        if epoch == 1:
            metrics_df.to_csv(csv_path, index=False)
        else:
            epoch_metrics_df.to_csv(csv_path, mode='a', header=False, index=False)

        # Update the learning rate
        scheduler.step()

        # Save checkpoint at the end of the epoch
        if save_path is not None:
            save_checkpoint(model, optimizer, scheduler, epoch, save_path)

    print("\nTraining concluded")

    # Return the tracking metrics
    return metrics_df
