"""Loads the saved model and saved class names and results."""
import torch
from efficientnet_modular.efficientnet_model import efficientnet_model
def load_fn(num_classes, filename="efficientnet_checkpoint.pth", dropout_p=0.3, fine_tune=True):
    """
    Load a training checkpoint and reconstruct the model, class names, and results.

    Parameters
    ----------
    num_classes : int
        Number of output classes for the model architecture.
    filename : str, optional
        Path to the checkpoint file.
    dropout_p : float, optional
        Dropout probability for the classifier.
    fine_tune : bool, optional
        Whether to unfreeze classifier and last block.

    Returns
    -------
    model : torch.nn.Module
        The reconstructed model with loaded weights.
    class_names : list
        The class names from training.
    results : dict or any object
        The training results from the checkpoint.
    """
    checkpoint = torch.load(Path("models") / filename)

    model = efficientnet_model(num_classes=num_classes, dropout_p=dropout_p, fine_tune=fine_tune)
    model.load_state_dict(checkpoint["model_state_dict"])

    class_names = checkpoint["class_names"]
    results = checkpoint["results"]

    return model, class_names, results
