"""Generate prediction for a single input image (Django-ready)."""

import torch
from torchvision.transforms import v2
from PIL import Image

def pred_image(model: torch.nn.Module,
               image_file,
               class_names: list,
               device: torch.device):
    """
    Generate a prediction for a single uploaded image.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model used for inference.
    image_file : file-like object
        Uploaded image file (e.g., from Django request.FILES).
    class_names : list
        List of class names corresponding to model output indices.
    device : torch.device
        Device on which to run inference (e.g., "cpu" or "cuda").

    Returns
    -------
    pred_idx : int
        Predicted class index.
    pred_label : str
        Predicted class name.
    prob : float
        Confidence score (probability) for the prediction.
    """

    # Preprocessing pipeline
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])

    # Load and preprocess image
    image = Image.open(image_file).convert("RGB")
    X = transform(image).unsqueeze(0).to(device)

    # Run inference
    model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(X)
        pred_probs = torch.softmax(logits, dim=1)
        prob, pred = torch.max(pred_probs, dim=1)

    pred_idx = pred.item()
    pred_label = class_names[pred_idx]
    return pred_idx, pred_label, prob.item()
