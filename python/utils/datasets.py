# Datasets Loader utility functions

# Imports ----------------------------------------------------------------------

# Common Python imports
import os
import numpy as np

# h5py to import h5 dataset files
import h5py

# Torch imports
import torch as th
from torch.utils.data import Dataset
from torchvision import transforms

# Typining hints
from typing import List, Union, Callable, Tuple

# Datasets directory (relative to inside python/ directory)
DATASETS_DIR: str = "../datasets"
CLASSIFICATION_DIR: str = os.path.join(DATASETS_DIR, "classification")
SEGMENTATION_DIR: str = os.path.join(DATASETS_DIR, "segmentation/data")


# BraTS2020 Dataset ------------------------------------------------------------

# Scanner class for BraTS2020 dataset
class BrainScanDataset(Dataset):
    def __init__(self, file_paths, deterministic=False):
        self.file_paths = file_paths
        if deterministic:  # To always generate the same test images for consistency
            np.random.seed(1)
        np.random.shuffle(self.file_paths)

		# Define the resize transformation to halven image size (less parameters later)
        self.resize = transforms.Resize((128, 128))
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load h5 file, get image and mask
        file_path = self.file_paths[idx]
        with h5py.File(file_path, 'r') as file:
            image = file['image'][()]
            mask = file['mask'][()]
            
            # Reshape: (H, W, C) -> (C, H, W)
            image = image.transpose((2, 0, 1))
            mask = mask.transpose((2, 0, 1))
            
            # Adjusting pixel values for each channel in the image so they are between 0 and 255
            for i in range(image.shape[0]):    # Iterate over channels
                min_val = np.min(image[i])     # Find the min value in the channel
                image[i] = image[i] - min_val  # Shift values to ensure min is 0
                max_val = np.max(image[i]) + 1e-4     # Find max value to scale max to 1 now.
                image[i] = image[i] / max_val
            
            # Convert to float and scale the whole image
            image = th.tensor(image, dtype=th.float32)
            mask = th.tensor(mask, dtype=th.float32) 

			# Resize the image and mask
            image = self.resize(image)
            mask = self.resize(mask)

			# Ensure the mask is binary after resizing
            mask = (mask > 0.5).float()
            
            
        return image, mask