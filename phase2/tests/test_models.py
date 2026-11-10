"""Network building blocks: shapes and per-row independence."""
import torch

from aerocast_dl import models, sequences


def test_encoder_shapes_and_no_batch_coupling():
    """Each row's encoding depends only on its own window: encoding a row
    alone equals encoding it inside a batch (no cross-row information)."""
    torch.manual_seed(0)
    enc = models.MultiEntityEncoder(["origin", "route"], n_channels=6, hidden=8).eval()
    x = {n: torch.randn(5, sequences.SEQ_LEN, 6) for n in enc.names}
    out = enc(x)
    assert out.shape == (5, enc.out_dim) == (5, 16)
    single = enc({n: v[2:3] for n, v in x.items()})
    torch.testing.assert_close(single[0], out[2])
