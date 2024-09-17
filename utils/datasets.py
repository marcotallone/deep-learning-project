# Datasets loader utility functions

# Imports ----------------------------------------------------------------------

# Common Python imports
import os
import numpy as np

# h5py to import h5 dataset files
import h5py

# Torch imports
import torch as th
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Typining hints
from typing import List, Tuple


# Segmentation task: BraTS2020 Dataset -----------------------------------------

# Pre-processing scanning class for BraTS2020 dataset images
class SegmentationPreprocessing(Dataset):
    """Class to preprocess the BraTS2020 dataset images for segmentation task

    Parameters
    ----------
    file_paths: List[str]
        List of file paths to the .h5 files containing the images and masks
    resize: Tuple[int, int], optional (default=(240, 240))
        Tuple with the new size of the images (height, width)
    deterministic: bool, optional (default=False)
        Boolean to shuffle the files randomly or not when loading the dataset
    """

    # Constructor
    def __init__(self, 
                 file_paths: List[str],
                 resize: Tuple[int, int] = (240,240),
                 deterministic: bool = False,
    ) -> None:
        
        # Store the file paths
        self.file_paths = file_paths

        # Generate the same test images (for consistency)
        if deterministic: np.random.seed(1)
        np.random.shuffle(self.file_paths)

		# Define the resize transformation
        self.resize = transforms.Resize((resize[0], resize[1]))
        
    # Len method
    def __len__(self) -> int:
        return len(self.file_paths)
    
    # Get item method
    def __getitem__(self, index: int) -> Tuple[th.Tensor, th.Tensor]:

        # Load h5 file, get image and mask (ground truth)
        file_path: str = self.file_paths[index]
        with h5py.File(file_path, 'r') as file:
            image = file['image'][()]
            mask  = file['mask'][()]
            
            # Reshape: (H, W, C) -> (C, H, W)
            image = image.transpose((2, 0, 1))
            mask = mask.transpose((2, 0, 1))
            
            # Adjusting pixel values for each channel in the image between 0 and 255
            for i in range(image.shape[0]):         # Iterate over channels
                min_val  = np.min(image[i])         # Find the min value in the channel
                image[i] = image[i] - min_val       # Shift values to ensure min is 0
                max_val  = np.max(image[i]) + 1e-4  # Find max value to scale max to 1 now.
                image[i] = image[i] / max_val       # Scale values to ensure max is 1
            
            # Convert to float and scale the whole image
            image: Tensor = th.tensor(image, dtype=th.float32)
            mask: Tensor = th.tensor(mask, dtype=th.float32) 

			# Resize the image and mask
            image = self.resize(image)
            mask = self.resize(mask)

			# Ensure the mask is binary after resizing
            mask = (mask > 0.5).float()
            
        return image, mask

# Loading function for the BraTS2020 dataset
def load_segmentation(directory: str,
                      split: float,
                      train_batch_size: int,
                      valid_batch_size: int,
                      percentage: float = 1,
                      resize: Tuple[int, int] = (240, 240),
                      deterministic: bool = True
) -> Tuple[th.utils.data.DataLoader, th.utils.data.DataLoader]:
    """Function to load the BraTS2020 dataset for segmentation task

    Parameters
    ----------
    directory: str
        Directory containing the .h5 files
    split: float
        Fraction of the dataset to use for training (train-test split)
    train_batch_size: int
        Batch size for the training dataloader
    valid_batch_size: int
        Batch size for the validation dataloader
    percentage: float, optional (default=1)
        Percentage of the dataset to use (1 = 100%)
    resize: Tuple[int, int], optional (default=(240, 240))
        Tuple with the new size of the images (height, width)
    deterministic: bool, optional (default=True)
        Boolean to shuffle the files randomly or not when loading the dataset

    Returns
    -------
    Tuple[th.utils.data.DataLoader, th.utils.data.DataLoader]
        Tuple containing the training and validation dataloaders
    """

    # Build .h5 file paths from directory containing .h5 files
    h5_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]
    np.random.seed(42)
    np.random.shuffle(h5_files)

    if deterministic: 
        # Remove the last files to use only a percentage of the dataset
        h5_files = h5_files[:int(percentage * len(h5_files))]
    else:
        # Randomly sample a percentage of the dataset
        h5_files = np.random.choice(h5_files, int(percentage * len(h5_files)), replace=False)

    # Split the dataset into train and validation sets
    split_index = int(split * len(h5_files))
    train_files = h5_files[:split_index]
    valid_files = h5_files[split_index:]

    # Create the train and validation datasets
    train_dataset = SegmentationPreprocessing(train_files, resize=resize)
    valid_dataset = SegmentationPreprocessing(valid_files, resize=resize, deterministic=deterministic)

    # Sample dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

    return train_dataloader, valid_dataloader