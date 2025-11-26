"""Remaps labels for subsets of the data"""
class RemapSubset(Dataset):
    """Wraps a Subset and remaps labels to contiguous [0..num_classes-1]."""
    def __init__(self, subset, selected_indices):
        self.subset = subset
        self.label_map = {old: new for new, old in enumerate(selected_indices)}

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, self.label_map[y]
