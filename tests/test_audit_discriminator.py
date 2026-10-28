"""Tests for the M2d audit checks: each check must pass on the real implementation and FAIL on a planted bug.
Small random models on CPU; no checkpoint is read."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from src.models import l2g  # noqa: E402
from src.train import audit_discriminator as ad  # noqa: E402
from tests.conftest import REPO_ROOT  # noqa: E402

T = 512


@pytest.fixture
def models():
    torch.manual_seed(0)
    gen, vttd = l2g.ECGen(24).train(), l2g.VTTD().train()
    x = torch.rand(4, 12, T)
    m = (torch.rand(4, 12, 1) > 0.5).float().expand_as(x).contiguous()
    m[:, 0] = 1  # at least one observed lead
    return gen, vttd, torch.cat([x * m, m], dim=1), x


def test_discriminator_step_updates_d_and_isolates_g(models):
    gen, vttd, inp, x = models
    res = ad.check_discriminator_step(gen, vttd, inp, x, seed=1)
    assert res["C1_pass"] and res["C2_pass"]


def test_adversarial_gradient_present_and_planted_detach_is_caught(models):
    gen, vttd, inp, x = models
    assert ad.check_adversarial_gradient(gen, vttd, inp, x, seed=1)["C3_pass"]

    def buggy(vttd_, x_gen, x_real, generator=None):  # mimics the released code: copy without autograd history
        return l2g.discriminator_loss(vttd_, x_gen.detach().clone(), x_real, generator=generator)

    res = ad.check_adversarial_gradient(gen, vttd, inp, x, seed=1, adv_fn=buggy)
    assert not res["C3_pass"] and res["grad_norm_adversarial_term"] == 0


def test_adversarial_sign_is_minus_alpha(models):
    gen, vttd, inp, x = models
    res = ad.check_adversarial_sign(gen, vttd, inp, x, seed=1)
    assert res["C4_pass"] and abs(res["coefficient_on_vttd_bce"] + 0.1) < 1e-4


def test_training_loop_source_has_detach_and_sign():
    assert ad.check_training_loop_source()["C4b_pass"]


def test_shuffle_convention():
    assert ad.check_shuffle_convention(seed=2)["C5_pass"]


def test_pair_accuracy():
    labels = torch.tensor([[1.0, 0.0] * 12])
    assert ad.pair_accuracy(torch.tensor([[0.9, 0.1] * 12]), labels) == 1.0
    assert ad.pair_accuracy(torch.tensor([[0.1, 0.9] * 12]), labels) == 0.0


def test_saved_audit_report_if_present():
    path = REPO_ROOT / "artifacts" / "reports" / "m2d_discriminator_audit.json"
    if not path.is_file():
        pytest.skip("Run: python -m src.train.audit_discriminator")
    rep = json.loads(path.read_text(encoding="utf-8"))
    assert rep["frozen_checkpoint_unchanged_after_audit"] is True
    assert rep["classification"] in {"A", "B", "C"}
