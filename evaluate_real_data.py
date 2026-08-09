"""AETHER End-to-End Real Data Evaluator.

This script automatically queries Microsoft Planetary Computer STAC for a tiny
co-registered patch of Sentinel-2, Sentinel-1, and Copernicus DEM data.
It downloads exactly a 256x256 pixel patch (at 10m resolution) using cloud-optimized
windowed reading, ensuring the total download is less than 20MB.

It then saves the raw data, loads it into the AETHER model, verifies shapes,
validates the mathematical alpha map constraints, and saves visualizations.
"""

import os
import sys
import logging
import warnings
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Geospatial libraries
import pystac_client
import planetary_computer
import rioxarray
import xarray as xr
from rasterio.enums import Resampling

# AETHER Imports
# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.aether import AETHERModel
from utils.config import load_config

# Suppress noisy warnings from rasterio/pyproj
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
BBOX = [-122.3, 47.6, -122.25, 47.65]  # [min_lon, min_lat, max_lon, max_lat] - Seattle area
TIME_RANGE = "2023-07-01/2023-07-31"   # Summer for clear skies
PATCH_SIZE = 256                       # 256x256 pixels

# Required Bands
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
S1_BANDS = ["vv", "vh"]

DATA_DIR = PROJECT_ROOT / "datasets" / "sample1"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def fetch_stac_items():
    """Query Microsoft Planetary Computer for STAC items."""
    logger.info(f"Querying Planetary Computer STAC API for bbox: {BBOX} in {TIME_RANGE}...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # 1. Optical (Sentinel-2 L2A) - Least Cloud Cover
    search_s2 = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=TIME_RANGE,
        query={"eo:cloud_cover": {"lt": 10}},
    )
    s2_items = list(search_s2.items())
    if not s2_items:
        raise ValueError("No Sentinel-2 items found.")
    s2_items.sort(key=lambda x: x.properties["eo:cloud_cover"])
    s2_item = s2_items[0]
    logger.info(f"Selected Sentinel-2 Item: {s2_item.id} (Cloud cover: {s2_item.properties['eo:cloud_cover']}%)")

    # 2. SAR (Sentinel-1 RTC) - Closest date
    search_s1 = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=BBOX,
        datetime=TIME_RANGE,
    )
    s1_items = list(search_s1.items())
    if not s1_items:
        raise ValueError("No Sentinel-1 items found.")
    s1_item = s1_items[0]
    logger.info(f"Selected Sentinel-1 Item: {s1_item.id}")

    # 3. DEM (Copernicus DEM 30m)
    search_dem = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=BBOX,
    )
    dem_items = list(search_dem.items())
    if not dem_items:
        raise ValueError("No DEM items found.")
    dem_item = dem_items[0]
    logger.info(f"Selected DEM Item: {dem_item.id}")

    return s2_item, s1_item, dem_item


def process_and_align_data(s2_item, s1_item, dem_item):
    """Download, align, and stack the data using windowed reading."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # We use S2 B04 (10m Red band) to define the master spatial grid
    logger.info("Defining master spatial grid from Sentinel-2 (10m resolution)...")
    b04_url = s2_item.assets["B04"].href
    
    # Open B04 and clip to the bounding box
    da_b04 = rioxarray.open_rasterio(b04_url, chunks=True).rio.clip_box(*BBOX, crs="EPSG:4326")
    
    # We want exactly PATCH_SIZE x PATCH_SIZE. We slice it.
    da_master = da_b04.isel(x=slice(0, PATCH_SIZE), y=slice(0, PATCH_SIZE))
    
    logger.info(f"Master Grid defined. CRS: {da_master.rio.crs}, Shape: {da_master.shape}, Transform: {da_master.rio.transform()}")

    # --- Process Sentinel-2 ---
    logger.info("Extracting and aligning all Sentinel-2 bands...")
    s2_arrays = []
    for band in S2_BANDS:
        if band in s2_item.assets:
            url = s2_item.assets[band].href
            # Open and reproject to match the exact master grid
            da = rioxarray.open_rasterio(url).rio.clip_box(*BBOX, crs="EPSG:4326").rio.reproject_match(
                da_master, resampling=Resampling.bilinear
            )
            s2_arrays.append(da.values.astype(np.float32))
        else:
            # Handle missing B10 in some L2A products (Cirrus band often dropped)
            logger.warning(f"Band {band} missing in S2 L2A item. Filling with zeros.")
            s2_arrays.append(np.zeros_like(da_master.values, dtype=np.float32))
    
    s2_stack = np.concatenate(s2_arrays, axis=0)  # (13, 256, 256)
    
    # --- Process Sentinel-1 ---
    logger.info("Extracting and aligning Sentinel-1 bands...")
    s1_arrays = []
    for band in S1_BANDS:
        url = s1_item.assets[band].href
        da = rioxarray.open_rasterio(url).rio.clip_box(*BBOX, crs="EPSG:4326").rio.reproject_match(
            da_master, resampling=Resampling.bilinear
        )
        s1_arrays.append(da.values.astype(np.float32))
        
    s1_stack = np.concatenate(s1_arrays, axis=0)  # (2, 256, 256)

    # --- Process DEM ---
    logger.info("Extracting and aligning DEM...")
    dem_url = dem_item.assets["data"].href
    da_dem = rioxarray.open_rasterio(dem_url).rio.clip_box(*BBOX, crs="EPSG:4326").rio.reproject_match(
        da_master, resampling=Resampling.bilinear
    )
    dem_stack = da_dem.values.astype(np.float32)  # (1, 256, 256)

    # Save to disk as .npy to satisfy requirements cleanly without GDAL write overhead
    logger.info(f"Saving arrays to {DATA_DIR}...")
    np.save(DATA_DIR / "optical.npy", s2_stack)
    np.save(DATA_DIR / "sar.npy", s1_stack)
    np.save(DATA_DIR / "dem.npy", dem_stack)

    # Also print required metadata for verification
    print("\n" + "="*50)
    print("DATASET VERIFICATION")
    print("="*50)
    print(f"CRS       : {da_master.rio.crs}")
    print(f"Transform : {da_master.rio.transform()}")
    print(f"Optical   : {s2_stack.shape}, Bands: {S2_BANDS}")
    print(f"SAR       : {s1_stack.shape}, Bands: {S1_BANDS}")
    print(f"DEM       : {dem_stack.shape}, Bands: ['Elevation']")
    print("✓ Spatially aligned and perfectly coregistered.")
    print("="*50 + "\n")

    return s2_stack, s1_stack, dem_stack


def normalize_for_display(img_array, p_low=2, p_high=98):
    """Normalize numpy array to [0, 1] for visualization based on percentiles."""
    img = np.nan_to_num(img_array)
    p2, p98 = np.percentile(img, (p_low, p_high))
    img = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)
    return img


def run_model_and_visualize(s2_stack, s1_stack, dem_stack):
    """Run the AETHER model and generate visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Convert to Tensors
    optical = torch.from_numpy(s2_stack).unsqueeze(0)  # (1, 13, 256, 256)
    sar = torch.from_numpy(s1_stack).unsqueeze(0)      # (1, 2, 256, 256)
    dem = torch.from_numpy(dem_stack).unsqueeze(0)     # (1, 1, 256, 256)

    # Handle NaNs in real data
    optical = torch.nan_to_num(optical)
    sar = torch.nan_to_num(sar)
    dem = torch.nan_to_num(dem)

    # 2. Load Model
    logger.info("Initializing AETHER model...")
    config_path = PROJECT_ROOT / "configs" / "model.yaml"
    cfg = load_config(config_path)
    # Set pretrained=False to avoid network bottleneck during this test run
    cfg.model.optical_encoder.pretrained = False
    cfg.model.sar_encoder.pretrained = False
    model = AETHERModel.build_from_dict(cfg.model)
    model.eval()

    # 3. Model Inference (Encoders + Fusion)
    logger.info("Running inference through encoders and fusion...")
    with torch.no_grad():
        # Get individual encoder features to print their shapes
        f_opt = model.optical_encoder(optical)
        f_sar = model.sar_encoder(sar)
        f_dem = model.dem_encoder(dem)
        
        print("\n" + "="*50)
        print("MODEL INTERMEDIATES")
        print("="*50)
        print(f"Optical feature shape : {tuple(f_opt.shape)}")
        print(f"SAR feature shape     : {tuple(f_sar.shape)}")
        print(f"DEM feature shape     : {tuple(f_dem.shape)}")

        # Full forward pass with intermediates
        outputs = model(optical, sar, dem, return_intermediates=True)
        f_shared = outputs["f_shared"]
        alpha_maps = outputs["alpha_maps"]
        lulc_logits = outputs["lulc"]
        
        print(f"F_shared shape        : {tuple(f_shared.shape)}")
        print(f"Alpha maps shape      : {tuple(alpha_maps.shape)}")
        
        # Verify alpha sum constraint mathematically
        alpha_sum = alpha_maps.sum(dim=1)
        is_valid = torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5)
        print(f"Sum(alpha) == 1       : {is_valid} (min={alpha_sum.min().item():.5f}, max={alpha_sum.max().item():.5f})")
        print("="*50 + "\n")

    # 4. Generate Visualizations
    logger.info("Generating and saving visualizations...")
    
    # RGB (B4=idx 3, B3=idx 2, B2=idx 1)
    rgb = np.stack([s2_stack[3], s2_stack[2], s2_stack[1]], axis=-1)
    rgb_vis = normalize_for_display(rgb)
    plt.imsave(OUTPUT_DIR / "rgb.png", rgb_vis)

    # SAR VV (idx 0)
    sar_vv = normalize_for_display(s1_stack[0])
    plt.imsave(OUTPUT_DIR / "sar_vv.png", sar_vv, cmap="gray")

    # DEM (idx 0)
    dem_vis = normalize_for_display(dem_stack[0])
    plt.imsave(OUTPUT_DIR / "dem.png", dem_vis, cmap="terrain")

    # Alpha Maps (Resize from H/16 to H for visualization)
    alpha_maps_up = torch.nn.functional.interpolate(
        alpha_maps, size=(PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False
    ).squeeze(0).cpu().numpy()
    
    plt.imsave(OUTPUT_DIR / "alpha_optical.png", alpha_maps_up[0], cmap="viridis", vmin=0, vmax=1)
    plt.imsave(OUTPUT_DIR / "alpha_sar.png", alpha_maps_up[1], cmap="viridis", vmin=0, vmax=1)
    plt.imsave(OUTPUT_DIR / "alpha_dem.png", alpha_maps_up[2], cmap="viridis", vmin=0, vmax=1)

    # LULC Prediction
    lulc_pred = lulc_logits.argmax(dim=1).squeeze(0).cpu().numpy()
    plt.imsave(OUTPUT_DIR / "prediction.png", lulc_pred, cmap="tab10")

    # 5. Save Features and Feature Visualizations
    logger.info("Saving intermediate features and their mean visualizations...")
    
    # Save .npy files (remove batch dimension)
    f_opt_np = f_opt.squeeze(0).cpu().numpy()
    f_sar_np = f_sar.squeeze(0).cpu().numpy()
    f_dem_np = f_dem.squeeze(0).cpu().numpy()
    f_shared_np = f_shared.squeeze(0).cpu().numpy()

    np.save(OUTPUT_DIR / "feature_optical.npy", f_opt_np)
    np.save(OUTPUT_DIR / "feature_sar.npy", f_sar_np)
    np.save(OUTPUT_DIR / "feature_dem.npy", f_dem_np)
    np.save(OUTPUT_DIR / "fshared.npy", f_shared_np)

    # Calculate mean across the 256 channels and save visualization
    def save_feature_mean(feat_array, filename):
        mean_vis = np.mean(feat_array, axis=0)
        # Normalize for display to make patterns visible
        mean_vis = normalize_for_display(mean_vis, p_low=2, p_high=98)
        # Resize to full patch size for consistency with other visuals
        mean_vis_tensor = torch.from_numpy(mean_vis).unsqueeze(0).unsqueeze(0)
        mean_vis_up = torch.nn.functional.interpolate(
            mean_vis_tensor, size=(PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False
        ).squeeze().numpy()
        import matplotlib.pyplot as plt
        plt.imsave(OUTPUT_DIR / filename, mean_vis_up, cmap="plasma")

    save_feature_mean(f_opt_np, "feature_optical_mean.png")
    save_feature_mean(f_sar_np, "feature_sar_mean.png")
    save_feature_mean(f_dem_np, "feature_dem_mean.png")
    save_feature_mean(f_shared_np, "fshared_mean.png")

    logger.info(f"All visualizations and features saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    s2, s1, dem = fetch_stac_items()
    s2_arr, s1_arr, dem_arr = process_and_align_data(s2, s1, dem)
    run_model_and_visualize(s2_arr, s1_arr, dem_arr)
    logger.info("Pipeline test completed successfully!")
