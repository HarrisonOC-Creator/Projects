"""Transfer Learning model instantiation using efficientnet.v2"""
import torchvision
from torchvision import models
import torch.nn as nn

def efficientnet_model(num_classes: int, 
                       dropout_p: float=0.3, 
                       fine_tune: bool=True):
    """
    Create an EfficientNetV2-S model adapted for custom classification tasks.

    This function loads a pretrained EfficientNetV2-S backbone, freezes all layers by default,
    and replaces the classifier head with a new dropout + linear layer suitable for the given
    number of output classes. Optionally, the classifier and the last feature block can be
    unfrozen for fine-tuning.

    Parameters
    ----------
    num_classes : int
        Number of output classes for the final classification layer.
    dropout_p : float, optional (default=0.3)
        Dropout probability applied before the final linear layer.
    fine_tune : bool, optional (default=True)
        If True, unfreezes the classifier and the last feature block ("features.6")
        to allow fine-tuning. If False, all layers remain frozen.

    Returns
    -------
    torch.nn.Module
        A modified EfficientNetV2-S model ready for training or inference on the specified
        number of classes.

    Notes
    -----
    - Uses pretrained weights (`EfficientNet_V2_S_Weights.DEFAULT`) for initialization.
    - Freezing most layers helps retain pretrained feature extraction while reducing
      training time and risk of overfitting.
    - Fine-tuning the classifier and last block allows adaptation to new datasets
      while leveraging pretrained representations.
    """

    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Optionally unfreeze classifier + last block
    if fine_tune:
        for name, param in model.named_parameters():
            if "classifier" in name or "features.6" in name:  # last block
                param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes)
    )
    return model
