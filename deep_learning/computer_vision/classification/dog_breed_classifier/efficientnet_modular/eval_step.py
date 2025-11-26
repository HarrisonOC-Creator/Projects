"""Evaluate the models loss and accuracy."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
def eval_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              device: torch.device):
    """
    Evaluate the model on a given dataset.
    Runs inference without gradient updates, computes average loss and accuracy
    across the dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing validation or test batches.
        loss_fn (torch.nn.Module): Loss function to compute evaluation loss.
        device (torch.device): Target device for computation ("cuda" or "cpu").

    Returns:
        tuple(float, float): Average evaluation loss and average evaluation accuracy.
    """

    model.to(device)
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            preds = torch.softmax(logits, dim=1).argmax(dim=1)
            total_acc += (preds == y).sum().item() / len(y)
    return total_loss / len(dataloader), total_acc / len(dataloader)
