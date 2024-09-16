# General training functions for PyTorch models

# Imports ----------------------------------------------------------------------

# Common Python imports
import numpy as np
from tqdm import trange
from datetime import datetime

# Torch imports
import torch as th
from torch import Tensor
from torch.utils.data import DataLoader
from safetensors.torch import save_model

# Typining hints
from typing import List, Union, Callable, Tuple


# Training function for U-Net models -------------------------------------------
def train_unet(model: th.nn.Module,
               loss_fn: th.nn.Module,
               optimizer: th.optim.Optimizer,
               train_loader: th.utils.data.DataLoader,
               valid_loader: th.utils.data.DataLoader,
               n_epochs: int,
               device: th.device = th.device("cpu"),
) -> Tuple[List[float], List[float]]:
    
    # Initialize the loss history
    train_loss_history: List[float] = []
    valid_loss_history: List[float] = []
    
    # Move the model to the device
    model.to(device)
    
    # Loop over the epochs
    print("\nTraining the model...")
    for epoch in trange(1, n_epochs + 1):
    # for epoch in range(1, n_epochs + 1):

        print(f"\nEpoch {epoch}/{n_epochs}") 

        # Set the model to training mode
        model.train()

        # Track epoch loss
        train_loss = 0.0

        # Loop over the batches in the training set
        for x, y in train_loader:

            # Move the data to the device
            x, y = x.to(device), y.to(device)

            # Zero out past gradients
            optimizer.zero_grad()

            # Forward pass + loss computation
            yhat: th.Tensor = model(x)
            loss: th.Tensor = loss_fn(yhat, y)
            train_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

        # Save the loss
        train_loss_history.append(train_loss)

        # Validate the model
        model.eval()

        # Track validation loss
        valid_loss = 0.0

        # Loop over the batches in the validation set
        with th.no_grad():
            for x_e, y_e in valid_loader:

                # Move the data to the device
                x_e, y_e = x_e.to(device), y_e.to(device)

                # Forward pass + loss computation
                yhat_e: th.Tensor = model(x_e)
                loss_e: th.Tensor = loss_fn(yhat_e, y_e)
                valid_loss += loss_e.item()

        # Save the loss
        valid_loss_history.append(valid_loss)

        print("\nTraining finished")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {valid_loss:.4f}")

        # Save the model
        now = datetime.now()
        save_model(model, f"unet1_{epoch}epochs_{now.strftime('%d%m%Y%H%M%S')}.pth")

    # Return the loss history
    return train_loss_history, valid_loss_history
            