# General training functions for PyTorch models

# Imports ----------------------------------------------------------------------

# Common Python imports
import os
from re import L
import sys
import numpy as np
import tqdm as tqdm

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch imports
import torch as th
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from safetensors.torch import save_model

# Typining hints
from typing import List, Tuple

# Assessment metrics
from utils.metrics import *

# Training function for U-Net models -------------------------------------------
def train_unet(model: th.nn.Module,
               loss_fn: th.nn.Module,
               optimizer: Optimizer,
               train_loader: DataLoader,
               valid_loader: DataLoader,
               n_epochs: int,
               start_epoch: int = 0,
               device: th.device = th.device("cpu"),
               save_path: str = None
) -> Tuple[List[float], 
           List[float],
           List[List[float]], 
           List[List[float]], 
           List[List[float]], 
           List[List[float]], 
           List[List[float]], 
           List[List[float]]]:
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
        Epoch to start training from
    device: th.device, optional (default=th.device("cpu"))
        Device to use for training
    save_path: str, optional (default=None)
        Path to save the model weights

    Returns
    -------
    Tuple[List[float], ..., List[List[float]]
        Tuple with lists containing the training losses, validation losses, Dice scores, 
        IoU scores, accuracy scores, FPR scores, FNR scores, precision scores, and recall scores
        for each epoch
    """
    
    # If given create the directory to store the model weights
    if save_path is not None: os.makedirs(save_path, exist_ok=True)
    
    # Initialize the lists for the tracking metrics
    train_losses: List[float] = []
    valid_losses: List[float] = []
    dice_scores: List[List[float]] = []
    iou_scores: List[List[float]] = []
    accuracy_scores: List[List[float]] = []
    fpr_scores: List[List[float]] = []
    fnr_scores: List[List[float]] = []
    precision_scores: List[List[float]] = []
    recall_scores: List[List[float]] = []
    
    # Move the model to the device
    model.to(device)
    
    print("\nTraining the model...")

    # Loop over the epochs
    for epoch in tqdm.tqdm(range(start_epoch + 1, n_epochs + 1), desc="Epoch", position=0): 
        # or:   for epoch in range(start_epoch + 1, n_epochs + 1):

        # Training loop ------------------------------------
        model.train() # Set the model to training mode
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
            loss.backward() # Backward pass
            optimizer.step() # Update the weights

        # Validation loop ------------------------------------------
        model.eval() # Set the model to evaluation mode
        valid_loss = 0.0 # Track validation loss
        dice_coeff: List[List[float]] = [] # Trach Dice coeff for each batch
        iou: List[List[float]] = [] # Track IoU for each batch
        accuracy: List[List[float]] = [] # Track accuracy for each batch
        fpr: List[List[float]] = [] # Track false positive rate for each batch
        fnr: List[List[float]] = [] # Track false negative rate for each batch
        precision: List[List[float]] = [] # Track precision for each batch
        recall: List[List[float]] = [] # Track recall for each batch
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
                dice_coeff.append(dice(yhat_e, y_e)) # Track the Dice coefficient
                iou.append(IoU(yhat_e, y_e))
                accuracy.append(accuracy2D(yhat_e, y_e)) # Track the accuracy
                fpr.append(fpr2D(yhat_e, y_e)) # Track the false positive rate
                fnr.append(fnr2D(yhat_e, y_e)) # Track the false negative rate
                precision.append(precision2D(yhat_e, y_e)) # Track the precision
                recall.append(recall2D(yhat_e, y_e)) # Track the recall

        # Track the validation loss at the end of the epoch
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Compute the averages of other metrics across all batches
        dice_scores.append(np.mean(dice_coeff, axis=0).tolist())
        iou_scores.append(np.mean(iou, axis=0).tolist())
        accuracy_scores.append(np.mean(accuracy, axis=0).tolist())
        fpr_scores.append(np.mean(fpr, axis=0).tolist())
        fnr_scores.append(np.mean(fnr, axis=0).tolist())
        precision_scores.append(np.mean(precision, axis=0).tolist())
        recall_scores.append(np.mean(recall, axis=0).tolist())

        # Display current epoch loss
        print("\n------------------------")
        print(f"Epoch {epoch}/{n_epochs}")
        print(f"Current LR:      {optimizer.param_groups[0]['lr']:.8f}")
        print("Metrics:")
        print(f"Training loss:   {train_loss:.4f}")
        print(f"Validation loss: {valid_loss:.4f}")
        print(f"Dice [R]:        {dice_scores[-1][0]:.4f}")
        print(f"Dice [G]:        {dice_scores[-1][1]:.4f}")
        print(f"Dice [B]:        {dice_scores[-1][2]:.4f}")
        print(f"Dice [Avg]:      {dice_scores[-1][3]:.4f}")
        print(f"IoU [R]:         {iou_scores[-1][0]:.4f}")
        print(f"IoU [G]:         {iou_scores[-1][1]:.4f}")
        print(f"IoU [B]:         {iou_scores[-1][2]:.4f}")
        print(f"IoU [Avg]:       {iou_scores[-1][3]:.4f}")
        print(f"Accuracy [R]:    {accuracy_scores[-1][0]:.4f}")
        print(f"Accuracy [G]:    {accuracy_scores[-1][1]:.4f}")
        print(f"Accuracy [B]:    {accuracy_scores[-1][2]:.4f}")
        print(f"Accuracy [Avg]:  {accuracy_scores[-1][3]:.4f}")
        print(f"FPR [R]:         {fpr_scores[-1][0]:.4f}")
        print(f"FPR [G]:         {fpr_scores[-1][1]:.4f}")
        print(f"FPR [B]:         {fpr_scores[-1][2]:.4f}")
        print(f"FPR [Avg]:       {fpr_scores[-1][3]:.4f}")
        print(f"FNR [R]:         {fnr_scores[-1][0]:.4f}")
        print(f"FNR [G]:         {fnr_scores[-1][1]:.4f}")
        print(f"FNR [B]:         {fnr_scores[-1][2]:.4f}")
        print(f"FNR [Avg]:       {fnr_scores[-1][3]:.4f}")
        print(f"Precision [R]:   {precision_scores[-1][0]:.4f}")
        print(f"Precision [G]:   {precision_scores[-1][1]:.4f}")
        print(f"Precision [B]:   {precision_scores[-1][2]:.4f}")
        print(f"Precision [Avg]: {precision_scores[-1][3]:.4f}")
        print(f"Recall [R]:      {recall_scores[-1][0]:.4f}")
        print(f"Recall [G]:      {recall_scores[-1][1]:.4f}")
        print(f"Recall [B]:      {recall_scores[-1][2]:.4f}")
        print(f"Recall [Avg]:    {recall_scores[-1][3]:.4f}")
        print("------------------------")

        # Save the model parameters at the current epoch using safetensors
        model_name = model.module.name if isinstance(model, th.nn.DataParallel) else model.name
        if save_path is not None:
            save_model(model, os.path.join(save_path, f"{model_name}_e{epoch}.pth"))
        else:
            save_model(model, f"{model_name}_e{epoch}.pth")

    print("\nTraining concluded")

    # Return the tracking metrics
    return train_losses, valid_losses, dice_scores, iou_scores, accuracy_scores, fpr_scores, fnr_scores, precision_scores, recall_scores
            

# Evaluation function for U-Net models -----------------------------------------
def validate_unet(model: th.nn.Module,
                  loss_fn: th.nn.Module,
                  valid_loader: DataLoader,
                  device: th.device = th.device("cpu")
) -> Tuple[List[float], 
           List[float],
           List[List[float]], 
           List[List[float]], 
           List[List[float]], 
           List[List[float]], 
           List[List[float]]]:
    """Evaluation function for U-Net models"""

    # Initialize the lists for the tracking metrics
    valid_losses: List[float] = []
    dice_scores: List[List[float]] = []
    iou_scores: List[List[float]] = []
    accuracy_scores: List[List[float]] = []
    fpr_scores: List[List[float]] = []
    fnr_scores: List[List[float]] = []
    precision_scores: List[List[float]] = []
    recall_scores: List[List[float]] = []

    # Move the model to the device
    model.to(device)

    print("\nEvaluating the model...")

    # Validation loop ------------------------------------------
    model.eval() # Set the model to evaluation mode

    valid_loss = 0.0 # Track validation loss
    dice_coeff: List[List[float]] = [] # Trach Dice coeff for each batch
    iou: List[List[float]] = [] # Track IoU for each batch
    accuracy: List[List[float]] = [] # Track accuracy for each batch
    fpr: List[List[float]] = [] # Track false positive rate for each batch
    fnr: List[List[float]] = [] # Track false negative rate for each batch
    precision: List[List[float]] = [] # Track precision for each batch
    recall: List[List[float]] = [] # Track recall for each batch

    with th.no_grad():
        for _, (x_e, y_e) in tqdm.tqdm(enumerate(valid_loader), 
                                       desc="Validation Batch", 
                                       position=0, 
                                       leave=False, 
                                       total=len(valid_loader)):
            # or:   for x_e, y_e in valid_loader:
            x_e, y_e = x_e.to(device), y_e.to(device) # Move the data to the device
            yhat_e: Tensor = model(x_e) # Forward pass
            loss_e: Tensor = loss_fn(yhat_e, y_e)
            valid_loss += loss_e.item()
            dice_coeff.append(dice(yhat_e, y_e))
            iou.append(IoU(yhat_e, y_e))
            accuracy.append(accuracy2D(yhat_e, y_e))
            fpr.append(fpr2D(yhat_e, y_e))
            fnr.append(fnr2D(yhat_e, y_e))
            precision.append(precision2D(yhat_e, y_e))
            recall.append(recall2D(yhat_e, y_e))

    # Track the validation loss at the end of the epoch
    valid_losses.append(valid_loss)

    # Compute the averages of other metrics across all batches
    dice_scores.append(np.mean(dice_coeff, axis=0).tolist())
    iou_scores.append(np.mean(iou, axis=0).tolist())
    accuracy_scores.append(np.mean(accuracy, axis=0).tolist())
    fpr_scores.append(np.mean(fpr, axis=0).tolist())
    fnr_scores.append(np.mean(fnr, axis=0).tolist())
    precision_scores.append(np.mean(precision, axis=0).tolist())
    recall_scores.append(np.mean(recall, axis=0).tolist())

    # Return the tracking metrics
    return valid_losses, dice_scores, iou_scores, accuracy_scores, fpr_scores, fnr_scores, precision_scores, recall_scores
    
