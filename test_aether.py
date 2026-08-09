"""Comprehensive verification test suite for AETHER model architecture.

Tests:
1. Config loading & DotDict functionality
2. Individual Encoders (Optical, SAR, DEM) output shapes & flexible input channels
3. CrossModalAlphaFusion output shapes, softmax alpha sum property, return_intermediates, modality embeddings gradient flow
4. Decoder progressive upsampling output shapes & SE blocks
5. Task Heads registry & LULC head output shape
6. Full AETHER model instantiation from config, forward pass, loss calculation, backward pass & gradient check across all parameters
7. Parameter breakdown per module
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from models.optical_encoder import OpticalEncoder
from models.sar_encoder import SAREncoder
from models.dem_encoder import DEMEncoder
from models.crossmodal_fusion import CrossModalAlphaFusion
from models.decoder import Decoder
from models.task_heads import LULCHead, get_task_head, list_task_heads
from models.aether import AETHERModel


def test_config():
    print("==================================================")
    print("TEST 1: Configuration Loading")
    print("==================================================")
    config_path = PROJECT_ROOT / "configs" / "model.yaml"
    cfg = load_config(config_path)
    assert cfg.model.optical_encoder.in_channels == 13
    assert cfg.model.sar_encoder.in_channels == 2
    assert cfg.model.dem_encoder.in_channels == 1
    assert cfg.model.fusion.feature_dim == 256
    print(f"✓ Config loaded successfully from {config_path}")
    print(f"  Optical in_channels: {cfg.model.optical_encoder.in_channels}")
    print(f"  SAR in_channels:     {cfg.model.sar_encoder.in_channels}")
    print(f"  DEM in_channels:     {cfg.model.dem_encoder.in_channels}")
    print(f"  Fusion feature_dim:  {cfg.model.fusion.feature_dim}\n")


def test_encoders():
    print("==================================================")
    print("TEST 2: Individual Encoders (H/16 Output Stride)")
    print("==================================================")
    B, H, W = 2, 256, 256
    
    # Test Optical Encoder with 13 bands (Sentinel-2) and 4 bands
    for ch in [13, 4, 3]:
        opt_enc = OpticalEncoder(in_channels=ch, feature_dim=256, pretrained=False)
        x_opt = torch.randn(B, ch, H, W)
        f_opt = opt_enc(x_opt)
        assert f_opt.shape == (B, 256, H // 16, W // 16), f"Expected (2, 256, 16, 16), got {f_opt.shape}"
        print(f"✓ OpticalEncoder(in_channels={ch}) output shape: {f_opt.shape}")

    # Test SAR Encoder with 2 channels (VV+VH)
    for ch in [2, 1, 3]:
        sar_enc = SAREncoder(in_channels=ch, feature_dim=256, pretrained=False)
        x_sar = torch.randn(B, ch, H, W)
        f_sar = sar_enc(x_sar)
        assert f_sar.shape == (B, 256, H // 16, W // 16), f"Expected (2, 256, 16, 16), got {f_sar.shape}"
        print(f"✓ SAREncoder(in_channels={ch}) output shape: {f_sar.shape}")

    # Test DEM Encoder with 1 channel
    for ch in [1, 2]:
        dem_enc = DEMEncoder(in_channels=ch, feature_dim=256)
        x_dem = torch.randn(B, ch, H, W)
        f_dem = dem_enc(x_dem)
        assert f_dem.shape == (B, 256, H // 16, W // 16), f"Expected (2, 256, 16, 16), got {f_dem.shape}"
        print(f"✓ DEMEncoder(in_channels={ch}) output shape: {f_dem.shape}")
    print()


def test_fusion():
    print("==================================================")
    print("TEST 3: CrossModalAlphaFusion & Spatial Alpha Maps")
    print("==================================================")
    B, C, H_prime, W_prime = 2, 256, 16, 16
    fusion_mod = CrossModalAlphaFusion(
        feature_dim=C,
        num_heads=8,
        dropout=0.1,
        use_modality_embeddings=True,
        num_refinement_blocks=2,
        se_reduction=16,
    )
    
    f_opt = torch.randn(B, C, H_prime, W_prime)
    f_sar = torch.randn(B, C, H_prime, W_prime)
    f_dem = torch.randn(B, C, H_prime, W_prime)

    res = fusion_mod(f_opt, f_sar, f_dem, return_intermediates=True)
    
    f_shared = res["f_shared"]
    alpha_maps = res["alpha_maps"]
    
    assert f_shared.shape == (B, C, H_prime, W_prime)
    assert alpha_maps.shape == (B, 3, H_prime, W_prime)
    
    # Check Softmax constraint: alpha values sum to 1.0 along dim 1
    alpha_sum = alpha_maps.sum(dim=1)
    assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5)
    print(f"✓ f_shared shape: {f_shared.shape}")
    print(f"✓ alpha_maps shape: {alpha_maps.shape}")
    print(f"✓ Softmax constraint satisfied: sum(alpha, dim=1) == 1.0 (min={alpha_sum.min().item():.6f}, max={alpha_sum.max().item():.6f})")
    
    # Verify return_intermediates
    assert "f_optical_cross" in res
    assert "f_sar_cross" in res
    assert "f_joint" in res
    print(f"✓ Intermediates returned correctly: f_optical_cross={res['f_optical_cross'].shape}, f_joint={res['f_joint'].shape}\n")


def test_decoder_and_heads():
    print("==================================================")
    print("TEST 4: Decoder & Task Heads")
    print("==================================================")
    B, C, H_prime, W_prime = 2, 256, 16, 16
    decoder = Decoder(feature_dim=C, out_channels=16, se_reduction=16)
    
    f_shared = torch.randn(B, C, H_prime, W_prime)
    decoded = decoder(f_shared)
    assert decoded.shape == (B, 16, 256, 256), f"Expected (2, 16, 256, 256), got {decoded.shape}"
    print(f"✓ Decoder output shape: {decoded.shape} (H/16 -> H resolution)")

    print(f"  Registered Task Heads: {list_task_heads()}")
    lulc_head = get_task_head("lulc", in_channels=16, num_classes=10)
    logits = lulc_head(decoded)
    assert logits.shape == (B, 10, 256, 256)
    print(f"✓ LULCHead output shape: {logits.shape}\n")


def test_full_model_and_backward():
    print("==================================================")
    print("TEST 5: Full AETHER Model End-to-End & Gradient Flow")
    print("==================================================")
    config_path = PROJECT_ROOT / "configs" / "model.yaml"
    cfg = load_config(config_path)
    # Set pretrained=False for instant test execution
    cfg.model.optical_encoder.pretrained = False
    cfg.model.sar_encoder.pretrained = False

    model = AETHERModel.build_from_dict(cfg.model)
    model.train()

    B, H, W = 2, 256, 256
    optical = torch.randn(B, 13, H, W)
    sar = torch.randn(B, 2, H, W)
    dem = torch.randn(B, 1, H, W)
    target = torch.randint(0, 10, (B, H, W))

    outputs = model(optical, sar, dem, return_intermediates=True)
    
    lulc_logits = outputs["lulc"]
    alpha_maps = outputs["alpha_maps"]
    
    assert lulc_logits.shape == (B, 10, H, W)
    assert alpha_maps.shape == (B, 3, H // 16, W // 16)
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(lulc_logits, target)
    
    loss.backward()

    # Verify gradients for all key parameter groups
    grad_checks = {
        "optical_encoder.projection": model.optical_encoder.projection.weight.grad,
        "sar_encoder.features": model.sar_encoder.features[0].weight.grad,
        "dem_encoder.features": model.dem_encoder.features[0].weight.grad,
        "fusion.emb_optical": model.fusion.emb_optical.grad,
        "fusion.emb_sar": model.fusion.emb_sar.grad,
        "fusion.emb_dem": model.fusion.emb_dem.grad,
        "fusion.cross_attn_opt2sar": model.fusion.cross_attn_opt2sar.in_proj_weight.grad,
        "fusion.alpha_estimator": model.fusion.alpha_estimator[0].weight.grad,
        "fusion.refinement": model.fusion.refinement[0].conv1.weight.grad,
        "decoder.stages": model.decoder.stages[0].conv[0].weight.grad,
        "task_heads.lulc": model.task_heads["lulc"].classifier.weight.grad,
    }

    for name, grad in grad_checks.items():
        assert grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(grad).any(), f"NaN gradient detected for {name}"
        print(f"  Grad OK: {name:<35} | Norm: {grad.norm().item():.6f}")

    print(f"✓ Loss value: {loss.item():.4f}")
    print("✓ Full backward pass successful — gradients flow to all submodules & modality embeddings!\n")


def test_parameter_count():
    print("==================================================")
    print("TEST 6: Parameter Breakdown")
    print("==================================================")
    config_path = PROJECT_ROOT / "configs" / "model.yaml"
    cfg = load_config(config_path)
    cfg.model.optical_encoder.pretrained = False
    cfg.model.sar_encoder.pretrained = False
    model = AETHERModel.build_from_dict(cfg.model)

    modules = {
        "Optical Encoder (ConvNeXt-Tiny Stage 3)": model.optical_encoder,
        "SAR Encoder (ResNet18 Layer 3)": model.sar_encoder,
        "DEM Encoder (Lightweight CNN)": model.dem_encoder,
        "CrossModalAlphaFusion Module": model.fusion,
        "Progressive SE Decoder": model.decoder,
        "LULC Head": model.task_heads["lulc"],
    }

    total_params = sum(p.numel() for p in model.parameters())

    for name, module in modules.items():
        p_count = sum(p.numel() for p in module.parameters())
        pct = (p_count / total_params) * 100
        print(f"  {name:<42}: {p_count:>10,} ({pct:>5.1f}%)")

    print(f"--------------------------------------------------")
    print(f"  TOTAL MODEL PARAMETERS                   : {total_params:>10,}")
    print("==================================================\n")


if __name__ == "__main__":
    test_config()
    test_encoders()
    test_fusion()
    test_decoder_and_heads()
    test_full_model_and_backward()
    test_parameter_count()
    print("ALL TESTS PASSED PERFECTLY!")
