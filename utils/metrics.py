# Collection of assessment metrics functions for PyTorch models

# Imports ----------------------------------------------------------------------

# Torch imports
import torch as th
from torch import Tensor

# Typining hints
from typing import List


# Dice Similarity Coefficient --------------------------------------------------
def dice(pred_mask: Tensor, true_mask: Tensor, epsilon: float = 1e-6) -> List[float]:
    """
    Compute the Dice Coefficient for each channel separately and return them
    individually as well as the average across all channels.

    Parameters
    ----------
    pred_mask : Tensor
        The predicted mask of shape [batch_size, channels, height, width].
    true_mask : Tensor
        The ground truth mask of shape [batch_size, channels, height, width].
    epsilon : float, optional
        A small value to avoid division by zero, by default 1e-6.

    Returns
    -------
    List[float]
        The Dice Coefficient for each RGB channel and the average Dice in 
        the following order: [Dice_Red, Dice_Green, Dice_Blue, Average_Dice]
    """

    # Apply sigmoid to the predictions to get probabilities
    pred_mask = th.sigmoid(pred_mask)

    # Flatten the tensors to [batch_size, channels, height * width]
    pred_flat = pred_mask.view(pred_mask.size(0), pred_mask.size(1), -1)
    target_flat = true_mask.view(true_mask.size(0), true_mask.size(1), -1)

    # Compute the intersection and union
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

    # Compute the Dice Coefficient for each channel
    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    # Compute the average Dice Coefficient across all channels
    avg_dice = dice.mean().item()

    # Return the Dice Coefficient for each channel and the average Dice Coefficient
    return [dice[:, 0].mean().item(), dice[:, 1].mean().item(), dice[:, 2].mean().item(), avg_dice] 


# 2D Acuracy -------------------------------------------------------------------
def accuracy2D(pred_mask: Tensor, true_mask: Tensor, threshold: float = 0.5) -> list[float]:
    """
    Compute the pixel-wise accuracy for each channel separately and return a list containing
    the accuracy for the red, green, and blue channels, as well as the average accuracy.

    Parameters
    ----------
    pred : Tensor
        The predicted tensor of shape [batch_size, channels, height, width].
    target : Tensor
        The ground truth tensor of shape [batch_size, channels, height, width].
    threshold : float, optional
        The threshold to binarize the predicted mask, by default 0.5.

    Returns
    -------
    List[float]
        The pixel-wise accuracy for each RGB channel and the average accuracy in 
        the following order: [Accuracy_Red, Accuracy_Green, Accuracy_Blue, Average_Accuracy]
    """

    # Apply sigmoid to the predictions to get probabilities
    pred_mask = th.sigmoid(pred_mask)

    # Binarize the predictions using the threshold
    pred_bin = (pred_mask > threshold).float()

    # Flatten the tensors to compute the accuracy
    pred_flat = pred_bin.view(pred_mask.size(0), pred_mask.size(1), -1)
    target_flat = true_mask.view(true_mask.size(0), true_mask.size(1), -1)

    # Compute the number of correct predictions
    correct = (pred_flat == target_flat).float().sum(dim=2)

    # Compute the total number of pixels
    total = target_flat.size(2)

    # Compute the accuracy for each channel
    accuracy = correct / total

    # Compute the average accuracy across all channels
    avg_accuracy = accuracy.mean().item()

    # Return the accuracy for each channel and the average accuracy
    return [accuracy[:, 0].mean().item(), accuracy[:, 1].mean().item(), accuracy[:, 2].mean().item(), avg_accuracy]


# 2D False Positive Rate -------------------------------------------------------
def fpr2D(pred_mask: Tensor, true_mask: Tensor, threshold: float = 0.5) -> List[float]:
    """
    Compute the pixel-wise False Positive Rate (FPR) for each channel separately and return a list containing
    the FPR for the red, green, and blue channels, as well as the average FPR.

    Parameters
    ----------
    pred_mask : Tensor
        The predicted tensor of shape [batch_size, channels, height, width].
    true_mask : Tensor
        The ground truth tensor of shape [batch_size, channels, height, width].
    threshold : float, optional
        The threshold to binarize the predicted mask, by default 0.5.

    Returns
    -------
    List[float]
        The pixel-wise FPR for each RGB channel and the average FPR in 
        the following order: [FPR_Red, FPR_Green, FPR_Blue, Average_FPR]
    """

    # Apply sigmoid to the predictions to get probabilities
    pred_mask = th.sigmoid(pred_mask)

    # Binarize the predictions using the threshold
    pred_bin = (pred_mask > threshold).float()

    # Flatten the tensors to compute the FPR
    pred_flat = pred_bin.view(pred_mask.size(0), pred_mask.size(1), -1)
    target_flat = true_mask.view(true_mask.size(0), true_mask.size(1), -1)

    # Compute False Positives and True Negatives
    false_positives = ((pred_flat == 1) & (target_flat == 0)).float().sum(dim=2)
    true_negatives = ((pred_flat == 0) & (target_flat == 0)).float().sum(dim=2)

    # Compute the FPR for each channel
    fpr = false_positives / (false_positives + true_negatives + 1e-6)

    # Compute the average FPR across all channels
    avg_fpr = fpr.mean().item()

    # Return the FPR for each channel and the average FPR
    return [fpr[:, 0].mean().item(), fpr[:, 1].mean().item(), fpr[:, 2].mean().item(), avg_fpr]


# 2D False Negative Rate -------------------------------------------------------
def fnr2D(pred_mask: Tensor, true_mask: Tensor, threshold: float = 0.5) -> List[float]:
    """
    Compute the pixel-wise False Negative Rate (FNR) for each channel separately and return a list containing
    the FNR for the red, green, and blue channels, as well as the average FNR.

    Parameters
    ----------
    pred_mask : Tensor
        The predicted tensor of shape [batch_size, channels, height, width].
    true_mask : Tensor
        The ground truth tensor of shape [batch_size, channels, height, width].
    threshold : float, optional
        The threshold to binarize the predicted mask, by default 0.5.

    Returns
    -------
    List[float]
        The pixel-wise FNR for each RGB channel and the average FNR in 
        the following order: [FNR_Red, FNR_Green, FNR_Blue, Average_FNR]
    """

    # Apply sigmoid to the predictions to get probabilities
    pred_mask = th.sigmoid(pred_mask)

    # Binarize the predictions using the threshold
    pred_bin = (pred_mask > threshold).float()

    # Flatten the tensors to compute the FNR
    pred_flat = pred_bin.view(pred_mask.size(0), pred_mask.size(1), -1)
    target_flat = true_mask.view(true_mask.size(0), true_mask.size(1), -1)

    # Compute False Negatives and True Positives
    false_negatives = ((pred_flat == 0) & (target_flat == 1)).float().sum(dim=2)
    true_positives = ((pred_flat == 1) & (target_flat == 1)).float().sum(dim=2)

    # Compute the FNR for each channel
    fnr = false_negatives / (false_negatives + true_positives + 1e-6)

    # Compute the average FNR across all channels
    avg_fnr = fnr.mean().item()

    # Return the FNR for each channel and the average FNR
    return [fnr[:, 0].mean().item(), fnr[:, 1].mean().item(), fnr[:, 2].mean().item(), avg_fnr]


# 2D Precision -----------------------------------------------------------------
def precision2D(pred_mask: Tensor, true_mask: Tensor, threshold: float = 0.5) -> List[float]:
    """
    Compute the pixel-wise Precision for each channel separately and return a list containing
    the Precision for the red, green, and blue channels, as well as the average Precision.

    Parameters
    ----------
    pred_mask : Tensor
        The predicted tensor of shape [batch_size, channels, height, width].
    true_mask : Tensor
        The ground truth tensor of shape [batch_size, channels, height, width].
    threshold : float, optional
        The threshold to binarize the predicted mask, by default 0.5.

    Returns
    -------
    List[float]
        The pixel-wise Precision for each RGB channel and the average Precision in 
        the following order: [Precision_Red, Precision_Green, Precision_Blue, Average_Precision]
    """

    # Apply sigmoid to the predictions to get probabilities
    pred_mask = th.sigmoid(pred_mask)

    # Binarize the predictions using the threshold
    pred_bin = (pred_mask > threshold).float()

    # Flatten the tensors to compute the Precision
    pred_flat = pred_bin.view(pred_mask.size(0), pred_mask.size(1), -1)
    target_flat = true_mask.view(true_mask.size(0), true_mask.size(1), -1)

    # Compute True Positives and False Positives
    true_positives = ((pred_flat == 1) & (target_flat == 1)).float().sum(dim=2)
    false_positives = ((pred_flat == 1) & (target_flat == 0)).float().sum(dim=2)

    # Compute the Precision for each channel
    precision = true_positives / (true_positives + false_positives + 1e-6)

    # Compute the average Precision across all channels
    avg_precision = precision.mean().item()

    # Return the Precision for each channel and the average Precision
    return [precision[:, 0].mean().item(), precision[:, 1].mean().item(), precision[:, 2].mean().item(), avg_precision]


# 2D Recall --------------------------------------------------------------------
def recall2D(pred_mask: Tensor, true_mask: Tensor, threshold: float = 0.5) -> List[float]:
    """
    Compute the pixel-wise Recall for each channel separately and return a list containing
    the Recall for the red, green, and blue channels, as well as the average Recall.

    Parameters
    ----------
    pred_mask : Tensor
        The predicted tensor of shape [batch_size, channels, height, width].
    true_mask : Tensor
        The ground truth tensor of shape [batch_size, channels, height, width].
    threshold : float, optional
        The threshold to binarize the predicted mask, by default 0.5.

    Returns
    -------
    List[float]
        The pixel-wise Recall for each RGB channel and the average Recall in 
        the following order: [Recall_Red, Recall_Green, Recall_Blue, Average_Recall]
    """

    # Apply sigmoid to the predictions to get probabilities
    pred_mask = th.sigmoid(pred_mask)

    # Binarize the predictions using the threshold
    pred_bin = (pred_mask > threshold).float()

    # Flatten the tensors to compute the Recall
    pred_flat = pred_bin.view(pred_mask.size(0), pred_mask.size(1), -1)
    target_flat = true_mask.view(true_mask.size(0), true_mask.size(1), -1)

    # Compute True Positives and False Negatives
    true_positives = ((pred_flat == 1) & (target_flat == 1)).float().sum(dim=2)
    false_negatives = ((pred_flat == 0) & (target_flat == 1)).float().sum(dim=2)

    # Compute the Recall for each channel
    recall = true_positives / (true_positives + false_negatives + 1e-6)

    # Compute the average Recall across all channels
    avg_recall = recall.mean().item()

    # Return the Recall for each channel and the average Recall
    return [recall[:, 0].mean().item(), recall[:, 1].mean().item(), recall[:, 2].mean().item(), avg_recall]