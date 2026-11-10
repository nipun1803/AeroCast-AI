"""Network building blocks.

Step 4 adds the sequence encoder; the embedding MLP (Step 3) and the Wide &
Deep hybrid (Step 5) are added in their own steps and reuse it.
"""
from __future__ import annotations

import torch
from torch import nn


class SequenceEncoder(nn.Module):
    """GRU over one entity's daily history window, oldest day first.

    Input  [batch, seq_len, n_channels] (``sequences.SequenceNormalizer`` output)
    Output [batch, hidden]: the final hidden state, i.e. a summary of the
    window as of the newest slot (outcomes known by the decision clock).

    The window is already causal, so the GRU's direction carries no leakage
    risk; running it oldest -> newest lets the last state weight recent days.
    """

    def __init__(self, n_channels: int, hidden: int = 16, n_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)          # h: [n_layers, batch, hidden]
        return h[-1]


class MultiEntityEncoder(nn.Module):
    """One ``SequenceEncoder`` per entity type (origin, carrier, route);
    outputs are concatenated in a fixed order."""

    def __init__(self, names: list[str], n_channels: int, hidden: int = 16, **kw):
        super().__init__()
        self.names = list(names)
        self.encoders = nn.ModuleDict({n: SequenceEncoder(n_channels, hidden, **kw)
                                       for n in self.names})
        self.out_dim = hidden * len(self.names)

    def forward(self, windows: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.encoders[n](windows[n]) for n in self.names], dim=1)
