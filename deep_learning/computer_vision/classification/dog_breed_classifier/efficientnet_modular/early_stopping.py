"""Halts training when validation loss stops improving."""
class EarlyStopping:
    """
    Implements early stopping to halt training when validation loss stops improving.
    Tracks the best validation loss and stops training after a patience threshold
    if no improvement is observed.

    Args:
        patience (int): Number of epochs to wait after last improvement before stopping.
        min_delta (float): Minimum change in validation loss to qualify as improvement.

    Attributes:
        best_loss (float or None): Best validation loss observed so far.
        counter (int): Number of consecutive epochs without improvement.
        should_stop (bool): Flag indicating whether training should stop.
    """
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        """
        Update early stopping state based on current validation loss.

        Args:
            val_loss (float): Current epoch's validation loss.

        Returns:
            None
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
