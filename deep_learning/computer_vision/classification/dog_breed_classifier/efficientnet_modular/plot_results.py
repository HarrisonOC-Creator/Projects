"""Plots the evaluation metrics for training and validation."""
import matplotlib.pyplot as plt
def plot_results(results):
    """
    Plot training and validation loss and accuracy curves.
    Displays side-by-side plots for loss and accuracy across epochs,
    and prints final test metrics if available.

    Args:
        results (dict): Dictionary containing lists of metrics with keys
            "train_loss", "train_acc", "val_loss", "val_acc",
            optionally "test_loss" and "test_acc".

    Returns:
        None
    """
    epochs = range(1, len(results["train_loss"]) + 1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, results["train_loss"], label="Train Loss")
    plt.plot(epochs, results["val_loss"], label="Val Loss")
    plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(epochs, results["train_acc"], label="Train Acc")
    plt.plot(epochs, results["val_acc"], label="Val Acc")
    plt.legend(); plt.title("Accuracy")
    plt.show()
