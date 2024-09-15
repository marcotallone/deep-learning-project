import os
import numpy as np

import torch as th
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import datasets, transforms
from safetensors.torch import save_model, load_model

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange

from typing import List, Union, Callable, Tuple
from torch import Tensor

# Set the default style
sns.set_theme(
	style="whitegrid",
	palette="tab10",
	rc={
		"grid.linestyle": "--",
		"grid.color": "gray",
		"grid.alpha": 0.3,
		"grid.linewidth": 0.3,
	},
)

# # Constant hyperparameters
DEVICE_AUTODETECT: bool = True
TRAIN_BATCH_SIZE: int = 32	# Batch size for training
# TEST_BS: int = 1024	# Batch size for testing
EPOCHS: int = 10	# Number of epochs

# Hyperparameters
# TRAIN_BATCH_SIZE: int = 64
# TEST_BATCH_SIZE: int = 1000
# EPOCHS: int = 15
CRITERION: Union[th.nn.Module, Callable[[th.Tensor], th.Tensor]] = (
    th.nn.CrossEntropyLoss(reduction="mean")
)
EVAL_CRITERION: Union[th.nn.Module, Callable[[th.Tensor], th.Tensor]] = (
    th.nn.CrossEntropyLoss(reduction="sum")
)
LR: float = 2e-3

# Device setup
device: th.device = th.device(
    "cuda" if th.cuda.is_available() and DEVICE_AUTODETECT else "cpu"
)
print(f"Using device: {device}")

# Define the transformations for the training and testing datasets
transform = transforms.Compose([
    transforms.Resize((150, 150)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert images to PyTorch tensors
])

# Define the paths to the training and testing datasets
train_dir = '../datasets/classification/Training'
test_dir = '../datasets/classification/Testing'

# Create the datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Create the DataLoaders
train_loader: DataLoader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=TEST_BS, shuffle=False)

# Second Model (from Archlinux)
class CNN_Gus(th.nn.Module):
    def __init__(self, cls_out: int = 4) -> None:
        super().__init__()

        self.conv = th.nn.Sequential(
            th.nn.Conv2d( # convolutional 1
                in_channels=3, out_channels=32, kernel_size=4, stride=1, padding=0
            ),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=3), # max pooling 1
            th.nn.Conv2d( # convolutional 2
                in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0
            ),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=3), # max pooling 2
            th.nn.Conv2d( # convolutional 3
                in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=0
            ),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=3), # max pooling 3
            th.nn.Conv2d( # convolutional 4
                in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0
            ),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )

        self.head = th.nn.Sequential(
            th.nn.Linear(128, 512),
            th.nn.ReLU(),
            
			# +++
            th.nn.Dropout(p=0.5),
            th.nn.Linear(512, 512),
            th.nn.ReLU(),
			# +++

            th.nn.Dropout(p=0.5),
            th.nn.Linear(512, cls_out),
            th.nn.Softmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.conv(x))

model: CNN_Gus = CNN_Gus().to(device)
model.train()

# Let's define the optimizer
optimizer: th.optim.Optimizer = th.optim.Adam(
    params=model.parameters(), lr=0.001, betas=(0.869, 0.995), eps=1e-7
)

eval_losses: List[float] = []
eval_acc: List[float] = []
test_acc: List[float] = []

print("----------------------------------------")
print("Training model...")
# Loop over epochs
for epoch in trange(EPOCHS, desc="Training epoch"):
# for epoch in range(EPOCHS):

    # print("Training epoch", epoch + 1)
    model.train()  # Remember to set the model in training mode before actual training

    # Loop over data
    for batch_idx, batched_datapoint in enumerate(train_loader):

        x, y = batched_datapoint
        x, y = x.to(device), y.to(device)

        # Forward pass + loss computation
        yhat = model(x)
        loss = CRITERION(yhat, y)

        # Zero-out past gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

    # Log the loss and accuracy on the training set...
    num_elem: int = 0
    trackingmetric: float = 0
    trackingcorrect: int = 0

    model.eval()  # Remember to set the model in evaluation mode before evaluating it

    # Since we are just evaluating the model, we don't need to compute gradients
    with th.no_grad():
        # ... by looping over training data again
        for _, batched_datapoint_e in enumerate(train_loader):
            x_e, y_e = batched_datapoint_e
            x_e, y_e = x_e.to(device), y_e.to(device)
            modeltarget_e = model(x_e)
            ypred_e = th.argmax(modeltarget_e, dim=1, keepdim=True)
            trackingmetric += EVAL_CRITERION(modeltarget_e, y_e).item()
            trackingcorrect += ypred_e.eq(y_e.view_as(ypred_e)).sum().item()
            num_elem += x_e.shape[0]
        eval_losses.append(trackingmetric / num_elem)
        eval_acc.append(trackingcorrect / num_elem)

print("----------------------------------------")
print(f"Final training loss: {eval_losses[-1]}")
print(f"Final training accuracy: {eval_acc[-1]}")
print("----------------------------------------")

# Plot results
loss_color = "tab:red"
acc_color = "tab:blue"

fig, ax1 = plt.subplots()

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color=loss_color)
ax1.plot(eval_losses, color=loss_color)
ax1.tick_params(axis="y", labelcolor=loss_color)

ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy", color=acc_color)
ax2.plot(eval_acc, color=acc_color)
ax2.tick_params(axis="y", labelcolor=acc_color)

fig.tight_layout()

plt.title("Training loss and accuracy")
# plt.show()

# Save image as PNG
plt.savefig("training_loss_accuracy.png")

# Save model with safetensors
save_model(model, "./model_cnn_gus_safe.safetensors")

# --------------------------------------------------------
# Loading with `safetensors` (example...)
# --------------------------------------------------------
# model_loaded_safe: LinearRegressor = LinearRegressor(
#     in_features=P, out_features=1, bias=True
# )
# _ = load_model(model_loaded_safe, "./model_ols_safe.safetensors")