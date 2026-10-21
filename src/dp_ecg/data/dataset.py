"""PyTorch dataset over the cached waveform array."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """Rows of a (N, leads, samples) array (often a float16 memmap) with 0/1 labels."""

    def __init__(self, waveforms: np.ndarray, labels: np.ndarray, indices: np.ndarray | None = None):
        self.waveforms = waveforms
        self.labels = np.asarray(labels, dtype=np.float32)
        self.indices = np.arange(len(labels)) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        x = torch.from_numpy(np.asarray(self.waveforms[j], dtype=np.float32))
        return x, torch.tensor(self.labels[j])

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Whole split as tensors (the cache is small enough to hold in memory)."""
        x = torch.from_numpy(np.asarray(self.waveforms[self.indices], dtype=np.float32))
        return x, torch.from_numpy(self.labels[self.indices])
