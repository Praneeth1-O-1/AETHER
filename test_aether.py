"""AETHER — regression suite.

Run with pytest (``pytest test_aether.py -v``) or directly
(``python test_aether.py``).

The suite is weighted toward the invariants that were silently violated before,
because those are the ones that cost real experiments:

- **Absence must be absence.** A modality marked absent must contribute exactly
  nothing. It used to contribute its learned modality embedding, and -- through
  cross-attention on a constant query -- the *other* modality's content, so
  every ablation number produced by the project was measuring a model that
  still had the modality it claimed to have removed.
- **Synthetic and real absence must be the same signal.** Modality dropout used
  to zero the encoded features, producing an exact zero that no real tile ever
  produces, while a genuinely unacquired tile produced ``encoder(zeros)`` -- a
  nonzero learned constant.
- **A cloud must not move the label.** Cloud injection is only sound because
  the LULC label came from the clear anchor and stays true underneath.

The channel counts are imported rather than hardcoded. The previous revision
asserted ``in_channels == 13`` against a config that said 11, so the suite
aborted on its first test and had been failing silently since.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data.clouds import (
    CloudConfig,
    CloudInjector,
    CloudMaskBank,
    curriculum_coverage,
    fractal_cloud_mask,
)
from data.dataset import (
    DEM_CHANNELS,
    IGNORE_INDEX,
    NUM_LULC_CLASSES,
    OPTICAL_CHANNELS,
    SAR_CHANNELS,
    location_split,
    TileRecord,
)
from models.aether import AETHERModel
from models.losses import (
    alpha_entropy_penalty,
    masked_cross_entropy,
    presence_masked_cross_entropy,
)
from utils.config import load_config

H = W = 64


@pytest.fixture(scope="module")
def model() -> AETHERModel:
    cfg = load_config("configs/model.yaml")
    cfg.model.optical_encoder.in_channels = OPTICAL_CHANNELS
    cfg.model.sar_encoder.in_channels = SAR_CHANNELS
    cfg.model.dem_encoder.in_channels = DEM_CHANNELS
    cfg.model.optical_encoder.pretrained = False
    cfg.model.sar_encoder.pretrained = False
    cfg.model.use_aux_heads = True
    torch.manual_seed(0)
    return AETHERModel.build_from_dict(cfg.model).eval()


@pytest.fixture(scope="module")
def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return (
        torch.randn(3, OPTICAL_CHANNELS, H, W),
        torch.randn(3, SAR_CHANNELS, H, W),
        torch.randn(3, DEM_CHANNELS, H, W),
    )


# =========================================================================
# Config and shapes
# =========================================================================


def test_config_channels_match_dataset():
    """The config must describe the tensors the dataset actually emits.

    A drift here does not crash -- the encoder is built to whatever the config
    says and then fed something else, or the config value is silently
    overridden at train time -- so it has to be asserted.
    """
    cfg = load_config("configs/model.yaml")
    assert cfg.model.optical_encoder.in_channels == OPTICAL_CHANNELS
    assert cfg.model.sar_encoder.in_channels == SAR_CHANNELS
    assert cfg.model.dem_encoder.in_channels == DEM_CHANNELS
    assert cfg.model.task_heads.lulc.num_classes == NUM_LULC_CLASSES


def test_forward_shapes(model, inputs):
    out = model(*inputs)
    assert out["lulc"].shape == (3, NUM_LULC_CLASSES, H, W)
    assert out["alpha_maps"].shape[:2] == (3, 3)
    for name in ("optical", "sar", "dem"):
        assert out[f"aux_{name}"].shape == (3, NUM_LULC_CLASSES, H, W)


def test_alpha_sums_to_one(model, inputs):
    alpha = model(*inputs)["alpha_maps"]
    assert torch.allclose(alpha.sum(dim=1), torch.ones_like(alpha.sum(dim=1)), atol=1e-5)


# =========================================================================
# Presence: absence must be absence
# =========================================================================


def test_absent_modality_gets_zero_alpha(model, inputs):
    """Alpha must renormalize over PRESENT modalities only.

    A convex combination that spends weight on an absent modality subtracts
    that weight from the informative ones and replaces it with a constant bias.
    """
    presence = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    alpha = model(*inputs, presence=presence)["alpha_maps"]

    assert alpha[1, 0].abs().max() < 1e-6, "absent optical still receives alpha"
    assert alpha[2, 1].abs().max() < 1e-6, "absent SAR still receives alpha"
    assert alpha[2, 2].abs().max() < 1e-6, "absent DEM still receives alpha"
    # Present modalities must still form a valid distribution among themselves.
    assert torch.allclose(alpha.sum(dim=1), torch.ones_like(alpha.sum(dim=1)), atol=1e-5)


@pytest.mark.parametrize("modality,idx", [("optical", 0), ("sar", 1), ("dem", 2)])
def test_absent_modality_cannot_influence_output(model, inputs, modality, idx):
    """The regression test for the defect that invalidated every ablation.

    Perturbing an absent modality's input must change nothing. Previously it
    changed a great deal: the modality embedding was added after masking, and
    cross-attention turned a constant query into a function of the other
    modality, so "removed" modalities kept feeding the fusion.
    """
    optical, sar, dem = (t.clone() for t in inputs)
    tensors = {"optical": optical, "sar": sar, "dem": dem}

    presence = torch.ones(3, 3)
    presence[0, idx] = 0.0                       # sample 0 lacks this modality
    baseline = model(optical, sar, dem, presence=presence)["lulc"][0].clone()

    tensors[modality][0] = torch.randn_like(tensors[modality][0]) * 50.0
    perturbed = model(optical, sar, dem, presence=presence)["lulc"][0]

    delta = (perturbed - baseline).abs().max().item()
    assert delta == 0.0, f"absent {modality} leaked into the prediction (delta {delta:.3e})"


def test_present_modality_does_influence_output(model, inputs):
    """The converse. Without this, the test above passes on a broken model
    that ignores every input."""
    optical, sar, dem = (t.clone() for t in inputs)
    presence = torch.ones(3, 3)
    baseline = model(optical, sar, dem, presence=presence)["lulc"][0].clone()
    sar[0] = torch.randn_like(sar[0]) * 5.0
    delta = (model(optical, sar, dem, presence=presence)["lulc"][0] - baseline).abs().max()
    assert delta > 1e-4, "present SAR has no influence at all -- encoder collapsed?"


def test_all_absent_stays_finite(model, inputs):
    """The 'nothing' ablation row must not produce NaN.

    Masking alpha logits with -inf would make softmax NaN when every modality
    is absent; a finite floor is used instead.
    """
    out = model(*inputs, presence=torch.zeros(3, 3))
    assert torch.isfinite(out["lulc"]).all()
    assert torch.isfinite(out["alpha_maps"]).all()


def test_skips_are_gated_by_presence(model, inputs):
    """Skip tensors bypass fusion, so they need their own presence gate.

    An ungated skip would carry ``encoder(zeros)`` -- a nonzero learned
    constant -- straight into the decoder and quietly undo the masking. This is
    covered by the leak test above only when skips are enabled, so assert the
    configuration too.
    """
    assert model.use_skips, "config disabled skips; the gating path is untested"
    presence = torch.ones(3, 3)
    presence[0, 1] = 0.0
    optical, sar, dem = (t.clone() for t in inputs)
    base = model(optical, sar, dem, presence=presence)["lulc"][0].clone()
    sar[0] += 100.0
    assert (model(optical, sar, dem, presence=presence)["lulc"][0] - base).abs().max() == 0.0


# =========================================================================
# Cloud injection
# =========================================================================


@pytest.mark.parametrize("coverage", [0.1, 0.5, 0.9])
def test_fractal_mask_hits_requested_coverage(coverage):
    rng = np.random.default_rng(0)
    mask = fractal_cloud_mask(128, 128, coverage, rng)
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    assert abs((mask > 0.5).mean() - coverage) < 0.06


def test_fractal_mask_edge_cases():
    rng = np.random.default_rng(0)
    assert fractal_cloud_mask(32, 32, 0.0, rng).max() == 0.0
    assert fractal_cloud_mask(32, 32, 1.0, rng).min() == 1.0


def test_opaque_cloud_destroys_reflectance_and_validity():
    """Opaque cloud must zero validity where it covers, so the model is told."""
    optical = np.full((10, 32, 32), 0.2, dtype=np.float32)
    valid = np.ones((1, 32, 32), dtype=np.float32)
    inj = CloudInjector(CloudConfig(prob=1.0, fixed_coverage=1.0, opaque_frac=1.0,
                                    deterministic=True))
    out, out_valid, cov = inj(optical.copy(), valid.copy(), index=0)
    assert cov > 0.95
    assert out_valid.max() == 0.0, "opaque cloud must clear the validity channel"
    assert out.min() > 0.5, "opaque cloud must raise reflectance toward cloud albedo"


def test_haze_preserves_validity():
    """Semi-transparent cloud must NOT clear validity -- the model has to infer
    unreliability from the data, which is the harder and more realistic case."""
    optical = np.full((10, 32, 32), 0.2, dtype=np.float32)
    valid = np.ones((1, 32, 32), dtype=np.float32)
    inj = CloudInjector(CloudConfig(prob=1.0, fixed_coverage=1.0, opaque_frac=0.0,
                                    deterministic=True))
    out, out_valid, _ = inj(optical.copy(), valid.copy(), index=0)
    assert out_valid.min() == 1.0, "haze must leave the validity channel alone"
    assert out.mean() > optical.mean(), "haze must brighten the scene"


def test_cloud_injection_is_deterministic_when_asked():
    """The evaluation curve has to be reproducible across runs and checkpoints."""
    optical = np.random.rand(10, 32, 32).astype(np.float32)
    valid = np.ones((1, 32, 32), dtype=np.float32)
    cfg = CloudConfig(prob=1.0, fixed_coverage=0.5, deterministic=True, seed=7)
    a = CloudInjector(cfg)(optical.copy(), valid.copy(), index=3)
    b = CloudInjector(cfg)(optical.copy(), valid.copy(), index=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = CloudInjector(cfg)(optical.copy(), valid.copy(), index=4)
    assert not np.array_equal(a[0], c[0]), "different tiles must get different clouds"


def test_zero_coverage_is_a_noop():
    optical = np.random.rand(10, 16, 16).astype(np.float32)
    valid = np.ones((1, 16, 16), dtype=np.float32)
    inj = CloudInjector(CloudConfig(prob=1.0, fixed_coverage=0.0))
    out, out_valid, cov = inj(optical.copy(), valid.copy(), index=0)
    assert cov == 0.0 and np.array_equal(out, optical) and np.array_equal(out_valid, valid)


def test_curriculum_ramps_then_holds():
    assert curriculum_coverage(1, 40, start=0.3, ramp_frac=0.5) == pytest.approx(0.3)
    mid = curriculum_coverage(20, 40, start=0.3, ramp_frac=0.5)
    assert 0.3 < mid < 1.0
    assert curriculum_coverage(40, 40, start=0.3, ramp_frac=0.5) == pytest.approx(1.0)


def test_empty_mask_bank_falls_back_to_fractal():
    """A missing or empty bank must degrade to synthetic masks, not crash."""
    bank = CloudMaskBank(np.zeros((0, 32, 32), dtype=np.uint8))
    assert bank.sample(32, 32, 0.5, np.random.default_rng(0)) is None
    inj = CloudInjector(CloudConfig(prob=1.0, fixed_coverage=0.5), bank)
    out, _, cov = inj(np.random.rand(10, 32, 32).astype(np.float32),
                      np.ones((1, 32, 32), dtype=np.float32), index=0)
    assert cov > 0 and np.isfinite(out).all()


# =========================================================================
# Losses
# =========================================================================


def test_masked_ce_returns_zero_not_nan_when_nothing_observed():
    """352 tiles have no valid LULC pixel. Under mean reduction they evaluate
    0/0 -> NaN, and one backward pass poisons every weight in the model."""
    logits = torch.randn(2, NUM_LULC_CLASSES, 8, 8)
    target = torch.full((2, 8, 8), IGNORE_INDEX, dtype=torch.long)
    loss = masked_cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    assert torch.isfinite(loss) and loss.item() == 0.0


def test_aux_loss_ignores_absent_samples():
    """An auxiliary head must not be scored on a modality that was dropped --
    that asks it to classify an all-zero tensor, and the gradient teaches the
    encoder to emit a class prior instead of reading its input."""
    torch.manual_seed(0)
    logits = torch.randn(2, NUM_LULC_CLASSES, 8, 8)
    target = torch.randint(0, NUM_LULC_CLASSES, (2, 8, 8))

    both = presence_masked_cross_entropy(logits, target, torch.tensor([1.0, 1.0]))
    first_only = presence_masked_cross_entropy(logits, target, torch.tensor([1.0, 0.0]))
    none = presence_masked_cross_entropy(logits, target, torch.tensor([0.0, 0.0]))

    assert none.item() == 0.0, "all-absent batch must contribute zero, not NaN"
    assert not torch.isclose(both, first_only), "presence mask had no effect"
    # Masking to sample 0 must equal scoring sample 0 alone.
    alone = presence_masked_cross_entropy(logits[:1], target[:1], torch.tensor([1.0]))
    assert torch.isclose(first_only, alone, atol=1e-5)


def test_alpha_entropy_penalty_prefers_uniform():
    """Lower (more negative) penalty for a more uniform alpha, so adding it
    with a positive weight pushes away from collapse."""
    presence = torch.ones(1, 3)
    uniform = torch.full((1, 3, 4, 4), 1 / 3)
    collapsed = torch.zeros(1, 3, 4, 4)
    collapsed[:, 0] = 1.0
    assert alpha_entropy_penalty(uniform, presence) < alpha_entropy_penalty(collapsed, presence)
    assert alpha_entropy_penalty(uniform, presence).item() == pytest.approx(-1.0, abs=1e-4)


def test_alpha_entropy_ignores_single_present_modality():
    """With one modality present, alpha is forced to 1.0 and its entropy is
    structurally zero -- penalizing that would be penalizing the mask."""
    presence = torch.tensor([[1.0, 0.0, 0.0]])
    alpha = torch.zeros(1, 3, 4, 4)
    alpha[:, 0] = 1.0
    assert alpha_entropy_penalty(alpha, presence).item() == 0.0


# =========================================================================
# Split integrity
# =========================================================================


def test_split_is_disjoint_by_location():
    """The split groups on AOI with the year stripped. Splitting on the
    _rXXX_cYYY tile suffix -- as an earlier revision did -- puts every AOI in
    every split, because there are only 16 distinct suffixes archive-wide."""
    recs = [
        TileRecord(path=__import__("pathlib").Path(f"/tmp/loc{i}_{y}/t{j}"),
                   location=f"loc{i}", year=str(y),
                   has_optical=True, has_sar=True, has_lulc=True,
                   has_road=True, has_building=True)
        for i in range(200) for y in (2021, 2023) for j in range(2)
    ]
    train, val, test = location_split(recs, 0.1, 0.1)
    assert len(train) + len(val) + len(test) == len(recs)
    locs = [{r.location for r in s} for s in (train, val, test)]
    assert not (locs[0] & locs[1]) and not (locs[0] & locs[2]) and not (locs[1] & locs[2])


# =========================================================================
# Gradients
# =========================================================================


def test_backward_reaches_every_encoder(model, inputs):
    """Every encoder must receive gradient, including through the aux heads."""
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(*inputs, presence=torch.ones(3, 3))
    loss = out["lulc"].mean() + sum(out[f"aux_{k}"].mean() for k in ("optical", "sar", "dem"))
    loss.backward()

    for name, enc in (("optical", model.optical_encoder), ("sar", model.sar_encoder),
                      ("dem", model.dem_encoder)):
        grads = [p.grad for p in enc.parameters() if p.requires_grad and p.grad is not None]
        assert grads, f"{name} encoder received no gradient at all"
        assert any(g.abs().sum() > 0 for g in grads), f"{name} encoder gradient is all zero"
    model.eval()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
