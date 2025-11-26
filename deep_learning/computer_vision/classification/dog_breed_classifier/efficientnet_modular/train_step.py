"""Creates the training step for one epoch."""
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device):
    """
    Perform one training epoch step.
    Passes batches of data through the model, computes loss, updates weights,
    and tracks average loss and accuracy across the dataloader.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader providing training batches.
        loss_fn (torch.nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        device (torch.device): Target device for computation ("cuda" or "cpu").

    Returns:
        tuple(float, float): Average training loss and average training accuracy for the epoch.

    """
    model.to(device)
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        train_loss += loss.item()
        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        train_acc += (preds == y).sum().item() / len(y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / len(dataloader), train_acc / len(dataloader)
