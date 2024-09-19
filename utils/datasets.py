# Datasets loader utility functions

# Imports ----------------------------------------------------------------------

# Common Python imports
import os
from re import L
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
                 directory: List[str],
                 resize: Tuple[int, int] = (240,240),
                 deterministic: bool = False,
    ) -> None:
        
        # Store the file paths
        self.directory = directory

        # Generate the same test images (for consistency)
        if deterministic: np.random.seed(1)
        np.random.shuffle(self.directory)

		# Define the resize transformation
        self.resize = transforms.Resize((resize[0], resize[1]))
        
    # Len method
    def __len__(self) -> int:
        return len(self.directory)
    
    # Get item method
    def __getitem__(self, index: int) -> Tuple[th.Tensor, th.Tensor]:

        # Load h5 file, get image and mask (ground truth)
        file_path: str = self.directory[index]
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
    if np.isclose(percentage, 1):
        # Pick all the files
        h5_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]
    else:
        # Pick just a percentage of the files starting from the "middle" ones for each patient
        h5_files = []
        num_slices = 155  # Total number of slices per patient
        num_patients = 369  # Total number of patients
        num_files_per_patient = int(num_slices * percentage)  # Number of files to pick per patient
        middle_index = num_slices // 2  # Middle index

        for i in range(1, num_patients + 1):
            patient_files = [f for f in os.listdir(directory) if f.startswith(f'volume_{i}_slice_') and f.endswith('.h5')]
            patient_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by slice number

            start_index = middle_index - num_files_per_patient // 2
            end_index = start_index + num_files_per_patient

            selected_files = patient_files[start_index:end_index]
            h5_files.extend([os.path.join(directory, f) for f in selected_files])

            # Print the selected files in order for debugging
            # print(f"Selected files for patient {i}:")
            # for f in selected_files:
            #     print(f"\t{f}")


    # Shuffle the files randomly
    np.random.seed(42)
    np.random.shuffle(h5_files)

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
    

# Load single ------------------------------------------------------------------
def load_single(directory: str,
                index: int,
                resize: Tuple[int, int] = (240, 240)
) -> Tuple[List[th.Tensor], List[th.Tensor]]:
    """Function to load the whole MRI scan for a single patient from the dataset.

    Parameters
    ----------
    directory : str
        Directory containing the .h5 files
    index : int
        Index of the file to load, i.e. the patient number. Can range from 1 to 369.
    resize : Tuple[int, int], optional
        Tuple with the new size of the images (height, width), by default (240, 240)

    Returns
    -------
    Tuple[List[th.Tensor], List[th.Tensor]]
        Tuple containing the list of images and masks for the MRI scan.   
        Each MRI scan consists in ~155 images and masks.
    """

    # Check the index is within the range otherwise raise an error
    if index < 1 or index > 369:
        raise ValueError("Index must be between 1 and 369.")

    # Collect the .h5 files for the patient: then start with 'volume_{index}' and end with '.h5' 
    h5_files = [os.path.join(directory, f'volume_{index}_slice_{i}.h5') for i in range(155)]

    # Preprocess the images and masks
    images, masks = [], []
    for i in range(len(h5_files)):

        # Load h5 file, get image and mask (ground truth)
        file_path: str = h5_files[i]
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
            resize_transformation = transforms.Resize((resize[0], resize[1]))
            image = resize_transformation(image)
            mask = resize_transformation(mask)

            # Ensure the mask is binary after resizing
            mask = (mask > 0.5).float()

            # Append the images and masks
            images.append(image)
            masks.append(mask)

    return images, masks
