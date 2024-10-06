# Converter of model weights fro Pytorch
# Converts weights saved with pickle to safetensors

# Common Python imports
import os
import sys
import tqdm as tqdm
import copy
import warnings

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch imports
import torch as th
from torch import Tensor
from torch.nn import Module
from safetensors.torch import save_model as safe_save_model
from safetensors.torch import load_model as safe_load_model

# Assessment metrics
from utils.metrics import *

# Model import
from models.alexnet import AlexNet_128
from models.vgg16 import VGG16_128
from models.custom_cnn import CustomCNN
from models.vit import VisionTransformer
from models.classic_unet import ClassicUNet
from models.improved_unet import ImprovedUNet
from models.attention_unet import AttentionUNet, VisualAttentionUNet

# Constants
N_FILTERS: int      = 32
EPOCHS: int         = 20
SAVE_MODELS_PATH: str = f"models/saved_models"

# Choose the model to load
# model: th.nn.Module = AlexNet_128()
# model: th.nn.Module = VGG16_128()
# model: th.nn.Module = CustomCNN()
# model: th.nn.Module = VisionTransformer()
model: th.nn.Module = ClassicUNet(n_filters=N_FILTERS)
# model: th.nn.Module = ImprovedUNet(n_filters=N_FILTERS)
# model: th.nn.Module = AttentionUNet(n_filters=N_FILTERS)

# Load the models from the last training epoch

# File names
model_name: str = os.path.join(SAVE_MODELS_PATH, f"{model.name}_e{EPOCHS}.pth")

# Loading location
map_location = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

# Training checkpoints
with warnings.catch_warnings(): # (to avoid pickle module warning)
    warnings.simplefilter("ignore", FutureWarning)
    classic_checkpoint = th.load(model_name, map_location=map_location)

# Load the models
model.load_state_dict(classic_checkpoint["model_state_dict"])

# Save the model weights now using safetensors
safe_save_path: str = os.path.join(SAVE_MODELS_PATH, f"{model.name}_e{EPOCHS}.safetensors")
safe_save_model(model, safe_save_path)

# Deep copy the model
loaded_model = copy.deepcopy(model)

# Load the model weights using safetensors to check if it works
loaded_model = safe_load_model(loaded_model, safe_save_path)

print(f"Model weights saved and loaded successfully using safetensors.")