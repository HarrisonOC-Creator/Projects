"""generate predicitons for input images"""
import os
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from PIL import Image
import numpy as np

def pred_images(model: torch.nn.Module, 
                image_path: str, 
                class_names: list, 
                device: torch.device,
                image_extensions: list[str] = [".jpg", ".jpeg", ".png"]):
    """
    Generate predictions for a batch of images and visualize results in a grid.

    This function loads all valid image files from a given directory, applies a
    preprocessing pipeline, performs inference with a trained model, and plots
    the images alongside their predicted labels and probabilities using Matplotlib
    subplots. The breed name is displayed in bold, and the probability is shown
    in smaller text within the title.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model used for inference.
    image_path : str
        Path to the directory containing input images.
    class_names : list
        List of class names corresponding to model output indices.
    device : torch.device
        Device on which to run inference (e.g., "cpu" or "cuda").

    Returns
    -------
    predictions : list of int
        Predicted class indices for each image.
    prediction_labels : list of str
        Predicted class names for each image.
    probabilities : list of float
        Confidence scores (probabilities) for each prediction.
    images : list of PIL.Image.Image
        Original loaded images corresponding to the predictions.

    Notes
    -----
    - Only files with extensions `.jpg`, `.jpeg`, or `.png` are processed.
    - Images are resized to 224x224 and normalized to float32 before inference.
    - Results are displayed in a grid of up to 5 images per row, with unused
      subplot axes hidden.
    - Titles above each image show the predicted breed in bold and the probability
      percentage in smaller text.
    """

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])

    model.to(device)
    model.eval()

    predictions, prediction_labels, probabilities, images = [], [], [], []

    with torch.inference_mode():
        for img_file in os.listdir(image_path):
            # Skip non-image files or directories
            if not any(img_file.lower().endswith(ext) for ext in image_extensions):
                continue

            img_path = os.path.join(image_path, img_file)
            if not os.path.isfile(img_path):
                continue

            # Load and preprocess image
            image = Image.open(img_path).convert("RGB")
            X = transform(image).unsqueeze(0).to(device)

            # Forward pass
            logit = model(X)
            pred_probs = torch.softmax(logit, dim=1)
            prob, pred = torch.max(pred_probs, dim=1)

            # Store results
            predictions.append(pred.item())
            prediction_labels.append(class_names[pred.item()])
            probabilities.append(prob.item())
            images.append(image)

    names_spaced = [name.replace("_", " ").title() for name in prediction_labels]

    # Plot results in a grid
    n_images = len(images)
    cols = min(n_images, 5)  # up to 5 images per row
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), dpi=150)

    # Flatten axes for easy iteration
    if rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()

    for ax, img, label, prob in zip(axes, images, names_spaced, probabilities):
        percentage = prob * 100
        ax.imshow(img)

        # Bold breed name, smaller probability in the same title
        ax.set_title(
            f"{label}\nProbability: {percentage:.2f}%",
            fontsize=10
        )
        ax.set_title(
            f"{label}\nProbability: {percentage:.2f}%",
            fontweight="bold", fontsize=12
        )

        ax.axis("off")

    # Hide unused subplots
    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return predictions, prediction_labels, probabilities, images
