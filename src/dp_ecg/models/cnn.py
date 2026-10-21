"""Compact 1D CNN for 12-lead ECG, compatible with Opacus.

Input (batch, 12, 1000) -> one logit. Each block is
Conv1d (stride 2) -> GroupNorm -> ReLU, which halves the time axis:
1000 -> 500 -> 250 -> 125 -> 63, then global average pooling over time and a
linear head.

Why GroupNorm and not BatchNorm: BatchNorm normalises with statistics of the
whole batch, so one example's output (and gradient) depends on the other
examples. DP-SGD needs per-example gradients whose influence can be bounded
by clipping, so BatchNorm is not allowed. GroupNorm normalises within each
example and is supported by Opacus.
"""

from __future__ import annotations

from torch import nn


class ECGCNN(nn.Module):
    def __init__(self, in_leads: int = 12, channels=(32, 64, 64, 128), kernel_sizes=(7, 5, 5, 3),
                 groupnorm_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        if len(channels) != len(kernel_sizes):
            raise ValueError("channels and kernel_sizes must have the same length")
        layers, c_in = [], in_leads
        for c_out, k in zip(channels, kernel_sizes):
            if c_out % groupnorm_groups:
                raise ValueError(f"{c_out} channels not divisible into {groupnorm_groups} groups")
            layers += [nn.Conv1d(c_in, c_out, k, stride=2, padding=k // 2),
                       nn.GroupNorm(groupnorm_groups, c_out), nn.ReLU()]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, 1))

    def forward(self, x):
        return self.head(self.pool(self.features(x))).squeeze(-1)  # logits, shape (batch,)


def build_model(model_cfg: dict, in_leads: int = 12) -> ECGCNN:
    return ECGCNN(in_leads, tuple(model_cfg["channels"]), tuple(model_cfg["kernel_sizes"]),
                  model_cfg["groupnorm_groups"], model_cfg["dropout"])
