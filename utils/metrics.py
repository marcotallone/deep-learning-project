# Collection of assessment metrics functions for PyTorch models

# Imports ----------------------------------------------------------------------

# Torch imports
import torch as th
from torch import Tensor

# Typining hints
from typing import List, Optional


# Dice Similarity Coefficient --------------------------------------------------
def dice(pred_mask: Tensor, 
         true_mask: Tensor,
         threshold: float = 0.5,
         epsilon: float = 1e-6
) -> List[Optional[float]]:
    """
    Compute the Dice Coefficient for each channel separately and return them
    individually as well as the average across all channels.

    Parameters
    ----------
    pred_mask : Tensor
        The predicted mask of shape [batch_size, channels, height, width].
    true_mask : Tensor
        The ground truth mask of shape [batch_size, channels, height, width].
    threshold : float, optional
        The threshold to binarize the predicted mask, by default 0.5.
    epsilon : float, optional
        A small value to avoid division by zero, by default 1e-6.

    Returns
    -------
    List[Optional[float]]
        The Dice Coefficient for each RGB channel and the average Dice in 
        the following order: [Dice_Red, Dice_Green, Dice_Blue, Average_Dice].
        If for any channel both the predicted and true masks are all zeros,
        the corresponding Dice Coefficient will be None and the average is done on the rest.
    """

    # Apply sigmoid to the predictions to get probabilities
    pred_mask = th.sigmoid(pred_mask)

    # Binarize the predicted mask
    pred_mask = (pred_mask > threshold).float()

    # Flatten the tensors to [batch_size, channels, height * width]
    pred_flat = pred_mask.view(pred_mask.size(0), pred_mask.size(1), -1)
    target_flat = true_mask.view(true_mask.size(0), true_mask.size(1), -1)

    # Initialize the list to store Dice coefficients
    dice_coeffs = []

    # Iterate over each channel
    for channel in range(pred_flat.size(1)):
        pred_channel = pred_flat[:, channel, :]
        target_channel = target_flat[:, channel, :]

        # Check if both the true mask and the predicted mask are all zeros for this channel
        if th.all(target_channel == 0) and th.all(pred_channel == 0):
            dice_coeffs.append(None)
        else:
            # Compute the intersection and union
            intersection = (pred_channel * target_channel).sum()
            union = pred_channel.sum() + target_channel.sum()

            # Compute the Dice Coefficient for this channel
            dice_coeff = (2.0 * intersection + epsilon) / (union + epsilon)
            dice_coeffs.append(dice_coeff.item())

    # Compute the average Dice Coefficient across all channels that are not None
    valid_dice_coeffs = [d for d in dice_coeffs if d is not None]
    avg_dice = sum(valid_dice_coeffs) / len(valid_dice_coeffs) if valid_dice_coeffs else None

    # Append the average Dice Coefficient to the list
    dice_coeffs.append(avg_dice)

    return dice_coeffs


# IoU (Intersection over Union or Jaccard Index) -------------------------------
def IoU(pred_mask: Tensor, 
        true_mask: Tensor, 
        threshold: float = 0.5,
        epsilon: float = 1e-6
) -> List[Optional[float]]:
    """
    Compute the pixel-wise Intersection over Union (IoU) for each channel separately and return a list containing
    the IoU for the red, green, and blue channels, as well as the average IoU.

    Parameters
    ----------
    pred_mask : Tensor
        The predicted tensor of shape [batch_size, channels, height, width].
    true_mask : Tensor
        The ground truth tensor of shape [batch_size, channels, height, width].
    threshold : float, optional
        The threshold to binarize the predicted mask, by default 0.5.
    epsilon : float, optional
        A small value to avoid division by zero, by default 1e-6.

    Returns
    -------
    List[Optional[float]]
        The pixel-wise IoU for each RGB channel and the average IoU in 
        the following order: [IoU_Red, IoU_Green, IoU_Blue, Average_IoU].
        If for any channel both the predicted and true masks are all zeros,
        the corresponding IoU will be None and the average is done on the rest.
    """

    # Apply sigmoid to the predictions to get probabilities
    pred_mask = th.sigmoid(pred_mask)

    # Binarize the predicted mask
    pred_mask = (pred_mask > threshold).float()

    # Flatten the tensors to [batch_size, channels, height * width]
    pred_flat = pred_mask.view(pred_mask.size(0), pred_mask.size(1), -1)
    target_flat = true_mask.view(true_mask.size(0), true_mask.size(1), -1)

    # Initialize the list to store IoU values
    iou_values = []

    # Iterate over each channel
    for channel in range(pred_flat.size(1)):
        pred_channel = pred_flat[:, channel, :]
        target_channel = target_flat[:, channel, :]

        # Check if both the true mask and the predicted mask are all zeros for this channel
        if th.all(target_channel == 0) and th.all(pred_channel == 0):
            iou_values.append(None)
        else:
            # Compute the intersection and union
            intersection = (pred_channel * target_channel).sum()
            union = pred_channel.sum() + target_channel.sum() - intersection

            # Compute the IoU for this channel
            iou = intersection / (union + epsilon)
            iou_values.append(iou.item())

    # Compute the average IoU across all channels that are not None
    valid_iou_values = [iou for iou in iou_values if iou is not None]
    avg_iou = sum(valid_iou_values) / len(valid_iou_values) if valid_iou_values else None

    # Append the average IoU to the list
    iou_values.append(avg_iou)

    return iou_values


# 2D Accuracy -------------------------------------------------------------------
def accuracy2D(pred_mask: Tensor, 
               true_mask: Tensor, 
               threshold: float = 0.5
) -> list[Optional[float]]:
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
    List[Optional[float]]
        The pixel-wise accuracy for each RGB channel and the average accuracy in 
        the following order: [Accuracy_Red, Accuracy_Green, Accuracy_Blue, Average_Accuracy].
        If for any channel both the predicted and true masks are all zeros,
        the corresponding accuracy will be None and the average is done on the rest.
    """

    # Apply sigmoid to the predictions to get probabilities
    pred_mask = th.sigmoid(pred_mask)

    # Binarize the predictions using the threshold
    pred_bin = (pred_mask > threshold).float()

    # Flatten the tensors to [batch_size, channels, height * width]
    pred_flat = pred_bin.view(pred_mask.size(0), pred_mask.size(1), -1)
    target_flat = true_mask.view(true_mask.size(0), true_mask.size(1), -1)

    # Initialize the list to store accuracy values
    accuracy_values = []

    # Iterate over each channel
    for channel in range(pred_flat.size(1)):
        pred_channel = pred_flat[:, channel, :]
        target_channel = target_flat[:, channel, :]

        # Check if both the true mask and the predicted mask are all zeros for this channel
        if th.all(target_channel == 0) and th.all(pred_channel == 0):
            accuracy_values.append(None)
        else:
            # Compute the number of correct predictions
            correct = (pred_channel == target_channel).float().sum()

            # Compute the total number of pixels
            total = target_channel.numel()

            # Compute the accuracy for this channel
            accuracy = correct / total
            accuracy_values.append(accuracy.item())

    # Compute the average accuracy across all channels that are not None
    valid_accuracy_values = [a for a in accuracy_values if a is not None]
    avg_accuracy = sum(valid_accuracy_values) / len(valid_accuracy_values) if valid_accuracy_values else None

    # Append the average accuracy to the list
    accuracy_values.append(avg_accuracy)

    return accuracy_values


# 2D False Positive Rate -------------------------------------------------------
def fpr2D(pred_mask: Tensor, 
          true_mask: Tensor, 
          threshold: float = 0.5,
          epsilon: float = 1e-6
) -> List[Optional[float]]:
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
    epsilon : float, optional
        A small value to avoid division by zero, by default 1e-6.

    Returns
    -------
    List[Optional[float]]
        The pixel-wise FPR for each RGB channel and the average FPR in 
        the following order: [FPR_Red, FPR_Green, FPR_Blue, Average_FPR].
        If for any channel both the predicted and true masks are all zeros,
        the corresponding FPR will be None and the average is done on the rest.
    """

    # Apply sigmoid to the predictions to get probabilities
    pred_mask = th.sigmoid(pred_mask)

    # Binarize the predictions using the threshold
    pred_bin = (pred_mask > threshold).float()

    # Flatten the tensors to compute the FPR
    pred_flat = pred_bin.view(pred_mask.size(0), pred_mask.size(1), -1)
    target_flat = true_mask.view(true_mask.size(0), true_mask.size(1), -1)

    # Initialize the list to store FPR values
    fpr_values = []

    # Iterate over each channel
    for channel in range(pred_flat.size(1)):
        pred_channel = pred_flat[:, channel, :]
        target_channel = target_flat[:, channel, :]

        # Check if both the true mask and the predicted mask are all zeros for this channel
        if th.all(target_channel == 0) and th.all(pred_channel == 0):
            fpr_values.append(None)
        else:
            # Compute False Positives and True Negatives
            false_positives = ((pred_channel == 1) & (target_channel == 0)).float().sum()
            true_negatives = ((pred_channel == 0) & (target_channel == 0)).float().sum()

            # Compute the FPR for this channel
            fpr = false_positives / (false_positives + true_negatives + epsilon)
            fpr_values.append(fpr.item())

    # Compute the average FPR across all channels that are not None
    valid_fpr_values = [f for f in fpr_values if f is not None]
    avg_fpr = sum(valid_fpr_values) / len(valid_fpr_values) if valid_fpr_values else None

    # Append the average FPR to the list
    fpr_values.append(avg_fpr)

    return fpr_values


# 2D False Negative Rate -------------------------------------------------------
def fnr2D(pred_mask: Tensor, 
          true_mask: Tensor, 
          threshold: float = 0.5,
          epsilon: float = 1e-6
) -> Optional[List[float]]:
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
    epsilon : float, optional
        A small value to avoid division by zero, by default 1e-6.

    Returns
    -------
    Optional[List[float]]
        The pixel-wise FNR for each RGB channel and the average FNR in 
        the following order: [FNR_Red, FNR_Green, FNR_Blue, Average_FNR].
        If for any channel both the predicted and true masks are all zeros,
        the corresponding FNR will be None and the average is done on the rest.
    """

    # Apply sigmoid to the predictions to get probabilities
    pred_mask = th.sigmoid(pred_mask)

    # Binarize the predictions using the threshold
    pred_bin = (pred_mask > threshold).float()

    # Flatten the tensors to compute the FNR
    pred_flat = pred_bin.view(pred_mask.size(0), pred_mask.size(1), -1)
    target_flat = true_mask.view(true_mask.size(0), true_mask.size(1), -1)

    # Initialize the list to store FNR values
    fnr_values = []

    # Iterate over each channel
    for channel in range(pred_flat.size(1)):
        pred_channel = pred_flat[:, channel, :]
        target_channel = target_flat[:, channel, :]

        # Check if both the true mask and the predicted mask are all zeros for this channel
        if th.all(target_channel == 0) and th.all(pred_channel == 0):
            fnr_values.append(None)
        else:
            # Compute False Negatives and True Positives
            false_negatives = ((pred_channel == 0) & (target_channel == 1)).float().sum()
            true_positives = ((pred_channel == 1) & (target_channel == 1)).float().sum()

            # Compute the FNR for this channel
            fnr = false_negatives / (false_negatives + true_positives + epsilon)
            fnr_values.append(fnr.item())

    # Compute the average FNR across all channels that are not None
    valid_fnr_values = [f for f in fnr_values if f is not None]
    avg_fnr = sum(valid_fnr_values) / len(valid_fnr_values) if valid_fnr_values else None

    # Append the average FNR to the list
    fnr_values.append(avg_fnr)

    return fnr_values


# 2D Precision -----------------------------------------------------------------
def precision2D(pred_mask: Tensor, 
                true_mask: Tensor, 
                threshold: float = 0.5,
                epsilon: float = 1e-6
) -> List[Optional[float]]:
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
    epsilon : float, optional
        A small value to avoid division by zero, by default 1e-6.

    Returns
    -------
    List[Optional[float]]
        The pixel-wise Precision for each RGB channel and the average Precision in 
        the following order: [Precision_Red, Precision_Green, Precision_Blue, Average_Precision].
        If for any channel both the predicted and true masks are all zeros,
        the corresponding precision will be None and the average is done on the rest.
    """

    # Apply sigmoid to the predictions to get probabilities
    pred_mask = th.sigmoid(pred_mask)

    # Binarize the predicted mask
    pred_mask = (pred_mask > threshold).float()

    # Flatten the tensors to [batch_size, channels, height * width]
    pred_flat = pred_mask.view(pred_mask.size(0), pred_mask.size(1), -1)
    target_flat = true_mask.view(true_mask.size(0), true_mask.size(1), -1)

    # Initialize the list to store Precision values
    precision_values = []

    # Iterate over each channel
    for channel in range(pred_flat.size(1)):
        pred_channel = pred_flat[:, channel, :]
        target_channel = target_flat[:, channel, :]

        # Check if both the true mask and the predicted mask are all zeros for this channel
        if th.all(target_channel == 0) and th.all(pred_channel == 0):
            precision_values.append(None)
        else:
            # Compute True Positives and False Positives
            true_positives = ((pred_channel == 1) & (target_channel == 1)).float().sum()
            false_positives = ((pred_channel == 1) & (target_channel == 0)).float().sum()

            # Compute the Precision for this channel
            precision = true_positives / (true_positives + false_positives + epsilon)
            precision_values.append(precision.item())

    # Compute the average Precision across all channels that are not None
    valid_precision_values = [p for p in precision_values if p is not None]
    avg_precision = sum(valid_precision_values) / len(valid_precision_values) if valid_precision_values else None

    # Append the average Precision to the list
    precision_values.append(avg_precision)

    return precision_values


# 2D Recall --------------------------------------------------------------------
def recall2D(pred_mask: Tensor, 
             true_mask: Tensor, 
             threshold: float = 0.5,
             epsilon: float = 1e-6
) -> List[Optional[float]]:
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
    epsilon : float, optional
        A small value to avoid division by zero, by default 1e-6.

    Returns
    -------
    List[Optional[float]]
        The pixel-wise Recall for each RGB channel and the average Recall in 
        the following order: [Recall_Red, Recall_Green, Recall_Blue, Average_Recall].
        If for any channel both the predicted and true masks are all zeros,
        the corresponding Recall will be None and the average is done on the rest.
    """

    # Apply sigmoid to the predictions to get probabilities
    pred_mask = th.sigmoid(pred_mask)

    # Binarize the predicted mask
    pred_mask = (pred_mask > threshold).float()

    # Flatten the tensors to [batch_size, channels, height * width]
    pred_flat = pred_mask.view(pred_mask.size(0), pred_mask.size(1), -1)
    target_flat = true_mask.view(true_mask.size(0), true_mask.size(1), -1)

    # Initialize the list to store Recall values
    recall_values = []

    # Iterate over each channel
    for channel in range(pred_flat.size(1)):
        pred_channel = pred_flat[:, channel, :]
        target_channel = target_flat[:, channel, :]

        # Check if both the true mask and the predicted mask are all zeros for this channel
        if th.all(target_channel == 0) and th.all(pred_channel == 0):
            recall_values.append(None)
        else:
            # Compute True Positives and False Negatives
            true_positives = ((pred_channel == 1) & (target_channel == 1)).float().sum()
            false_negatives = ((pred_channel == 0) & (target_channel == 1)).float().sum()

            # Compute the Recall for this channel
            recall = true_positives / (true_positives + false_negatives + epsilon)
            recall_values.append(recall.item())

    # Compute the average Recall across all channels that are not None
    valid_recall_values = [r for r in recall_values if r is not None]
    avg_recall = sum(valid_recall_values) / len(valid_recall_values) if valid_recall_values else None

    # Append the average Recall to the list
    recall_values.append(avg_recall)

    return recall_values
