# General training functions for PyTorch models

# Imports ----------------------------------------------------------------------

# Common Python imports
import os
import tqdm as tqdm

# Torch imports
import torch as th
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from safetensors.torch import save_model

# Typining hints
from typing import List, Tuple


# Training function for U-Net models -------------------------------------------
def train_unet(model: th.nn.Module,
               loss_fn: th.nn.Module,
               optimizer: Optimizer,
               train_loader: DataLoader,
               valid_loader: DataLoader,
               n_epochs: int,
               device: th.device = th.device("cpu"),
               save_path: str = None
) -> Tuple[List[float], List[float]]:
    
    # If given create the directory to store the weights
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    # Initialize the loss history
    train_loss_history: List[float] = []
    valid_loss_history: List[float] = []
    
    # Move the model to the device
    model.to(device)
    
    print("\nTraining the model...")

    # Loop over the epochs
    for epoch in tqdm.tqdm(range(1, n_epochs + 1), desc="Epoch", position=0): 
        # or:   for epoch in range(1, n_epochs + 1):

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

        # Track the validation loss at the end of the epoch
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        # Display current epoch loss
        print("\n------------------------")
        print(f"Epoch {epoch}/{n_epochs}")
        print(f"Training loss:   {train_loss:.4f}")
        print(f"Validation loss: {valid_loss:.4f}")

        # Save the model parameters at the current epoch using safetensors
        model_name = model.module.name if isinstance(model, th.nn.DataParallel) else model.name
        if save_path is not None:
            save_model(model, os.path.join(save_path, f"{model_name}_e{epoch}.pth"))
        else:
            save_model(model, f"{model_name}_e{epoch}.pth")

    print("\nTraining concluded")

    # Return the loss history
    return train_loss_history, valid_loss_history
            