"""The exogenous window at step t must cover draws t .. T_imag-2, nothing before."""

import torch
from omegaconf import OmegaConf

from causal_a2c import CausalBaseline

CONFIG = OmegaConf.create(
    {
        "shape": [255],
        "rnn_units": 8,
        "layers": 1,
        "units": 16,
        "act": "SiLU",
        "norm": 1.0,
        "device": "cpu",
        "dist": {"name": "symexp_twohot", "bin_num": 255},
        "outscale": 1.0,
        "symlog_inputs": False,
        "name": "causal_baseline",
    }
)


def test_window_is_the_tail():
    torch.manual_seed(0)
    B, T, S, K, F = 2, 6, 3, 4, 5
    net = CausalBaseline(CONFIG, F, S * K)
    feat = torch.randn(B, T, F)
    noise = torch.randn(B, T - 1, S, K)

    base = net(feat, noise).mode()
    for i in range(T - 1):
        perturbed = noise.clone()
        perturbed[:, i] += 10.0
        changed = ~torch.isclose(net(feat, perturbed).mode(), base, atol=1e-6)
        # draw i belongs to the windows of steps 0 .. i, and to no later step
        assert changed[:, : i + 1].all(), f"draw {i} missing from window of an earlier step"
        assert not changed[:, i + 1 :].any(), f"draw {i} leaked into a later step"


if __name__ == "__main__":
    test_window_is_the_tail()
    print("ok")
