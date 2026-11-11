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


def embedding_dim(cardinality: int, cap: int = 16) -> int:
    """Rule-of-thumb embedding width 1.6 * n^0.56 (fastai), between 2 and ``cap``."""
    return int(min(cap, max(2, round(1.6 * cardinality ** 0.56))))


class DelayNet(nn.Module):
    """Embedding MLP (Step 3), optionally with GRU history encoders (Step 4)
    and a wide linear part (Step 5, Wide & Deep).

    logit = head(MLP([embeddings, numerics, GRU states])) [+ wide(x_wide)]

    * Categoricals: learned embeddings (``cat_mode="embed"``) or fixed one-hot
      vectors (``"onehot"``, the ablation that isolates what embedding adds).
    * LayerNorm, not BatchNorm: a row's output never depends on the other
      rows of its batch (docs/01_inputs_and_leakage.md).
    * With a wide part the head starts at zero, so the untrained network
      equals the LogReg it is initialised from and learns a correction.
    """

    def __init__(self, cardinalities: list[int], n_num: int, hidden: int = 128,
                 depth: int = 2, dropout: float = 0.2, emb_dropout: float = 0.0,
                 emb_dims: list[int] | None = None, cat_mode: str = "embed",
                 seq_names: list[str] | None = None, seq_channels: int = 6,
                 seq_hidden: int = 16, seq_dropout: float = 0.0, wide_dim: int = 0):
        super().__init__()
        self.cat_mode, self.cardinalities = cat_mode, list(cardinalities)
        if cat_mode == "embed":
            dims = emb_dims or [embedding_dim(c) for c in cardinalities]
            self.embs = nn.ModuleList(nn.Embedding(c, d) for c, d in zip(cardinalities, dims))
            cat_dim = sum(dims)
        elif cat_mode == "onehot":
            self.embs, cat_dim = None, sum(cardinalities)
        else:
            raise ValueError(cat_mode)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.seq = (MultiEntityEncoder(seq_names, seq_channels, seq_hidden)
                    if seq_names else None)
        self.seq_drop = nn.Dropout(seq_dropout)
        layers, d = [], cat_dim + n_num + (self.seq.out_dim if self.seq else 0)
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        self.wide = nn.Linear(wide_dim, 1) if wide_dim else None
        if self.wide is not None:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def init_wide(self, coef, intercept) -> None:
        """Copy a fitted LogReg (coefficients on its own design matrix)."""
        with torch.no_grad():
            self.wide.weight.copy_(torch.as_tensor(coef, dtype=torch.float32).reshape(1, -1))
            self.wide.bias.fill_(float(intercept))

    def forward(self, cat, num, seq=None, wide=None):
        if self.cat_mode == "embed":
            parts = [self.emb_drop(e(cat[:, k])) for k, e in enumerate(self.embs)]
        else:
            parts = [nn.functional.one_hot(cat[:, k], c).float()
                     for k, c in enumerate(self.cardinalities)]
        parts.append(num)
        if self.seq is not None:
            parts.append(self.seq_drop(self.seq(seq)))
        z = self.head(self.mlp(torch.cat(parts, dim=1))).squeeze(1)
        if self.wide is not None:
            z = z + self.wide(wide).squeeze(1)
        return z
