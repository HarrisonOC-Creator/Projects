"""The pipeline for deep learning process."""
import torch
from efficientnet_modular.efficientnet_model import efficientnet_model
from efficientnet_modular.train import train
from efficientnet_modular.load_kaggle import load_kaggle
def process(device: torch.device="cuda",
            num_classes: int=5,
            batch_size: int=32,
            seed: int=42,
            dropout: float=0.5,
            learning_rate: float=5e-4,
            loss_fn: torch.nn.Module=nn.CrossEntropyLoss(label_smoothing=0.1),
            epochs: int=50,
            early_stopping_patience: int=15,
            scheduling_patience: int=3,
            weight_decay: float=1e-4,
            factor: float=0.5,
            val_split: float=0.1):
    """
    End-to-end pipeline for transfer learning with EfficientNetV2-S.
    Loads and prepares the Stanford Dogs dataset, builds an EfficientNetV2-S model
    with a custom classifier head, sets up optimizer and scheduler, and
    trains the model using the defined training loop.

    Args:
        device (str): Target device for computation ("cuda" or "cpu").
        num_classes (int): Number of output classes to subset and train on.
        batch_size (int): Batch size for dataloaders.
        seed (int): Random seed for reproducibility.
        dropout (float): Dropout probability applied before the final linear layer.
        learning_rate (float): Initial learning rate for the optimizer.
        loss_fn (torch.nn.Module): Loss function to optimize (default: CrossEntropyLoss with label smoothing).
        epochs (int): Maximum number of epochs to train.
        early_stopping_patience (int): Number of epochs to wait for validation loss improvement before stopping.
        scheduling_patience (int): Number of epochs with no improvement before scheduler reduces learning rate.
        weight_decay (float): Weight decay (L2 regularization) applied by the optimizer.
        factor (float): Factor by which the scheduler reduces learning rate.
        val_split (float): Fraction of training data reserved for validation.

    Returns:
        tuple:
            model (torch.nn.Module): The trained EfficientNetV2-S model.
            results (dict): Dictionary containing training, validation, and test metrics.
            class_names (list[str]): List of class names used in training.
    """

    # Load data
    train_dataloader, val_dataloader, test_dataloader, class_names = load_kaggle(
        num_classes=num_classes,
        batch_size=batch_size,
        seed=seed,
        val_split=val_split
    )

    # Build model
    model = efficientnet_model(num_classes=num_classes, dropout_p=dropout)

    # Optimizer: AdamW with layer-wise learning rates
    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": learning_rate},   # classifier head
        {"params": model.features[-1].parameters(), "lr": learning_rate * 0.02}  # last block, smaller LR
    ], weight_decay=weight_decay)

    # Scheduler: cosine annealing for smooth LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train
    model, results = train(model, train_dataloader, val_dataloader, test_dataloader,
                           loss_fn, optimizer, scheduler, epochs, device,
                           early_stopping_patience=early_stopping_patience)

    return model, results, class_names
