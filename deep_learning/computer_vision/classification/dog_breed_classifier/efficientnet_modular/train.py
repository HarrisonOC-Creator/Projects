"""Trains model and returns model results with epoch tracking."""
import copy
from tqdm import tqdm
from timeit import default_timer as timer
import torch
import torch.nn as nn
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module, 
          optimizer: torch.nn.Module, 
          scheduler: torch.optim.lr_scheduler._LRScheduler, 
          epochs: int, 
          device: torch.device, 
          early_stopping_patience: int=15):
    """
    Train a model with early stopping and learning rate scheduling.
    Runs a training loop for a specified number of epochs, evaluates on
    validation data each epoch, applies a scheduler based on validation loss,
    and stops early if validation loss does not improve for a given patience.
    Tracks and returns training, validation, and test metrics.

    Args:
        model (torch.nn.Module): The model to train and evaluate.
        train_dataloader (torch.utils.data.DataLoader): DataLoader providing training batches.
        val_dataloader (torch.utils.data.DataLoader): DataLoader providing validation batches.
        test_dataloader (torch.utils.data.DataLoader): DataLoader providing test batches.
        loss_fn (torch.nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler adjusting learning rate based on validation loss.
        epochs (int): Maximum number of epochs to train.
        device (torch.device or str): Target device for computation ("cuda" or "cpu").
        early_stopping_patience (int): Number of epochs to wait for validation loss improvement before stopping.

    Returns:
        tuple:
            model (torch.nn.Module): The trained model (with best weights restored if early stopping triggered).
            results (dict): Dictionary containing lists of metrics:
                {
                    "train_loss": [...],
                    "train_acc": [...],
                    "val_loss": [...],
                    "val_acc": [...],
                    "test_loss": [...],
                    "test_acc": [...]
                }
    """
    
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": []}
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    best_model_wts = copy.deepcopy(model.state_dict())
    start_time = timer()

    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Training
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        # Validation
        val_loss, val_acc = eval_step(model, val_dataloader, loss_fn, device)

        # Scheduler reacts to validation loss
        scheduler.step()

        # Early stopping check
        if early_stopping.best_loss is None or val_loss < early_stopping.best_loss - early_stopping.min_delta:
            best_model_wts = copy.deepcopy(model.state_dict())
        early_stopping.step(val_loss)

        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(best_model_wts)
            break

        # Progress print
        if epoch == 0 or (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc*100:.2f}% "
                  f"| val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}% "
                  f"| LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc*100)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc*100)

    # Final test evaluation
    test_loss, test_acc = eval_step(model, test_dataloader, loss_fn, device)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc*100)

    end_time = timer()
    print(f"Training finished in {end_time-start_time:.2f} seconds")
    print(f"Final test loss: {test_loss:.4f}, test acc: {test_acc*100:.2f}%")

    plot_results(results)
    return model, results
