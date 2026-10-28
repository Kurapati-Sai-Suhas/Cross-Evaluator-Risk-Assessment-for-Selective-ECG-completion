"""Unit tests for src/models/l2g.py (architecture lock v0.1, Milestone 2b smoke gate).

CPU-only and small, so they run anywhere PyTorch is installed. The parameter counts are the
values required by the lock; a mismatch is a FAULT, never a reason to change the model.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data import missingness as mz  # noqa: E402
from src.models import l2g  # noqa: E402
from tests.conftest import REPO_ROOT  # noqa: E402

T = 2048


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    return l2g.ECGen(in_channels=24), l2g.ECGen(in_channels=12), l2g.VTTD()


# --------------------------------------------------------------------------- #
# Parameter counts required by the lock (D-021, Section 6 step 2)
# --------------------------------------------------------------------------- #
def test_parameter_counts_match_the_lock(models):
    gen24, gen12, vttd = models
    expected = l2g.expected_parameter_counts()
    assert l2g.count_parameters(gen24) == expected["ecgen_24_inputs"] == 7_980_076
    assert l2g.count_parameters(gen12) == expected["ecgen_12_inputs"] == 7_972_396  # = paper's 7.97M
    assert l2g.count_parameters(vttd) == expected["vttd"] == 14_184


def test_encoder_decoder_layout(models):
    gen24 = models[0]
    convs = [m for m in gen24.modules() if isinstance(m, torch.nn.Conv1d)]
    kernels = [c.kernel_size[0] for c in convs]
    assert kernels[0] == 10 and kernels[-1] == 10  # first and output conv: kernel 10
    assert set(kernels[1:-1]) == {3}  # inner convs: kernel 3 (CODE, AD-5)
    assert sum(isinstance(m, torch.nn.BatchNorm1d) for m in gen24.decoder.modules()) == 0  # no BN in decoder
    assert sum(isinstance(m, torch.nn.BatchNorm1d) for m in gen24.encoder.modules()) == 8


# --------------------------------------------------------------------------- #
# Shapes and basic behaviour
# --------------------------------------------------------------------------- #
def test_generator_and_discriminator_shapes(models):
    gen24, _, vttd = models
    gen24.eval()
    with torch.no_grad():
        out = gen24(torch.rand(2, 24, T))
        probs = vttd.eval()(torch.rand(2, 24, 400))
    assert out.shape == (2, 12, T)
    assert probs.shape == (2, 24) and bool(((probs > 0) & (probs < 1)).all())
    with pytest.raises(ValueError):
        gen24(torch.rand(1, 24, T - 1))


def test_minmax_per_lead_range_and_flat_lead():
    y = torch.randn(3, 12, 100)
    y[1, 4] = 2.5  # constant lead
    z = l2g.minmax_per_lead(y)
    assert torch.isfinite(z).all()
    assert torch.allclose(z.amin(-1)[0], torch.zeros(12)) and torch.allclose(z.amax(-1)[0], torch.ones(12))
    assert (z[1, 4] == 0).all()


def test_channel_shuffle_keeps_pairs_and_labels():
    g = torch.Generator().manual_seed(1)
    x_gen, x_real = torch.rand(4, 12, 50), torch.rand(4, 12, 50)
    d, labels = l2g.channel_shuffle(x_gen, x_real, generator=g)
    assert d.shape == (4, 24, 50) and labels.shape == (4, 24)
    pairs = d.reshape(4, 12, 2, 50)
    lab = labels.reshape(4, 12, 2)
    assert torch.equal(lab.sum(-1), torch.ones(4, 12))  # exactly one generated per pair
    for b in range(4):
        for lead in range(12):
            gen_pos = int(lab[b, lead].argmax())
            assert torch.equal(pairs[b, lead, gen_pos], x_gen[b, lead])
            assert torch.equal(pairs[b, lead, 1 - gen_pos], x_real[b, lead])
    assert 0 < float(lab[..., 0].mean()) < 1  # both orders occur
    d2, labels2 = l2g.channel_shuffle(x_gen, x_real, generator=torch.Generator().manual_seed(1))
    assert torch.equal(d, d2) and torch.equal(labels, labels2)  # reproducible with a seed


def test_ssim_properties():
    ssim = l2g.SSIM1D()
    w = l2g.gaussian_window()
    assert w.shape == (50,) and abs(float(w.sum()) - 1) < 1e-6
    x = torch.rand(2, 12, 512)
    assert abs(float(ssim(x, x)) - 1.0) < 1e-4
    assert float(ssim(x, torch.rand(2, 12, 512))) < 0.5
    assert float(l2g.reconstruction_loss(ssim, x * 3 + 1, x)) < 1e-4  # min-max makes the scale irrelevant


def test_losses_finite_and_backward_flows(models):
    _, _, _ = models
    torch.manual_seed(2)
    gen, vttd, ssim = l2g.ECGen(24), l2g.VTTD(), l2g.SSIM1D()
    target = torch.rand(2, 12, T)
    out = gen(torch.rand(2, 24, T))
    d_loss = l2g.discriminator_loss(vttd, l2g.minmax_per_lead(out).detach(), target,
                                    generator=torch.Generator().manual_seed(3))
    g_loss = l2g.generator_total_loss(ssim, vttd, out, target, epoch=10,
                                      generator=torch.Generator().manual_seed(4))
    assert torch.isfinite(d_loss) and torch.isfinite(g_loss)
    g_loss.backward()
    grads = [p.grad for p in gen.parameters()]
    assert all(gr is not None and torch.isfinite(gr).all() for gr in grads)
    stage1 = l2g.generator_total_loss(ssim, vttd, out.detach(), target, epoch=9)
    assert torch.isclose(stage1, l2g.reconstruction_loss(ssim, out.detach(), target))  # SSIM only before epoch 10


def test_copy_through_on_generator_output(models):
    gen24 = models[0].eval()
    rng = np.random.default_rng(5)
    x = rng.random((12, T)).astype(np.float32)
    m = mz.signal_mask(("I", "III", "V2"), T)
    x_obs = mz.apply_mask(x, m)
    with torch.no_grad():
        raw = gen24(l2g.model_input(torch.from_numpy(x_obs)[None], torch.from_numpy(m)[None]))
    x_gen = l2g.minmax_per_lead(raw)[0].numpy()
    x_hat = mz.copy_through(x_obs, x_gen, m)
    assert np.array_equal(x_hat[m == 1].view(np.uint32), x[m == 1].view(np.uint32))
    assert np.array_equal(x_hat[m == 0], x_gen[m == 0])


def test_model_input_rejects_unmasked_signal():
    x = torch.rand(1, 12, 64)
    m = torch.zeros(1, 12, 64)
    m[:, 0] = 1
    with pytest.raises(ValueError):
        l2g.model_input(x, m)
    assert l2g.model_input(x * m, m).shape == (1, 24, 64)


# --------------------------------------------------------------------------- #
# Real-data smoke report (written by src/train/smoke_l2g.py; skipped until it exists)
# --------------------------------------------------------------------------- #
def test_smoke_report_passes_every_gate_step():
    path = REPO_ROOT / "artifacts" / "reports" / "smoke_l2g_v0_1.json"
    if not path.is_file():
        pytest.skip("Run: python -m src.train.smoke_l2g")
    report = json.loads(path.read_text(encoding="utf-8"))
    failed = [s["step"] for s in report["steps"] if s["status"] != "PASS"]
    assert failed == [], failed
    assert report["roles_loaded"] == ["A_fit"]
    assert report["full_training_started"] is False  # only the 8-record overfit check ran
