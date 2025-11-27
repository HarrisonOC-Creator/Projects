"""Saves a training checkpoint including model weights, class names, and results."""
from pathlib import Path
import torch
def save_fn(model, class_names, results, filename="efficientnet_checkpoint.pth"):
    """
    Save a training checkpoint including model weights, class names, and results.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model to save.
    class_names : list
        List of class names used during training.
    results : dict or any serializable object
        Training results (e.g., metrics, history).
    filename : str, optional
        Path to save the checkpoint file.
    """
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "results": results
    }, models_dir / filename)
