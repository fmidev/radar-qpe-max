#!/usr/bin/env python
"""Sanity-check a composite QPE max-accumulation COG."""

# builtin
import sys

# pypi
import numpy as np
import rasterio

from qpemax.constants import LWE_SCALE_FACTOR, UINT16_FILLVAL


def check_cog(path: str) -> None:
    with rasterio.open(path) as ds:
        data = ds.read(1)
        profile = ds.profile
        bounds = ds.bounds
        nodata = ds.nodata

    print(f"File: {path}")
    print(f"Driver: {profile.get('driver')}")
    print(f"CRS: {ds.crs}")
    print(f"Size: {profile['width']}x{profile['height']}")
    print(f"Dtype: {profile['dtype']}")
    print(f"Compress: {profile.get('compress')}")
    print(f"Nodata: {nodata}")
    print(f"Bounds: {bounds}")
    print(f"Resolution: {ds.res}")
    print()

    total = data.size
    if nodata is not None:
        nodata_mask = data == int(nodata)
    else:
        nodata_mask = np.zeros_like(data, dtype=bool)
    n_nodata = int(nodata_mask.sum())
    valid = data[~nodata_mask]
    n_valid = valid.size

    print(f"Total pixels: {total}")
    print(f"Nodata pixels: {n_nodata} ({100 * n_nodata / total:.1f}%)")
    print(f"Valid pixels: {n_valid} ({100 * n_valid / total:.1f}%)")
    print()

    if n_valid == 0:
        print("WARNING: No valid data pixels found!")
        return

    print("Raw uint16 statistics:")
    print(f"  min: {valid.min()}")
    print(f"  max: {valid.max()}")
    print(f"  mean: {valid.mean():.1f}")
    print(f"  median: {np.median(valid):.1f}")
    print()

    scaled = valid.astype(np.float64) * LWE_SCALE_FACTOR
    print(f"Scaled values (x{LWE_SCALE_FACTOR}, mm):")
    print(f"  min: {scaled.min():.2f}")
    print(f"  max: {scaled.max():.2f}")
    print(f"  mean: {scaled.mean():.2f}")
    print(f"  median: {np.median(scaled):.2f}")
    print()

    n_zero = int((valid == 0).sum())
    print(f"Zero pixels: {n_zero} ({100 * n_zero / total:.1f}%)")

    # histogram of non-zero valid values
    nonzero = valid[valid > 0]
    if nonzero.size > 0:
        pcts = [25, 50, 75, 90, 95, 99]
        percentiles = np.percentile(nonzero, pcts)
        print("\nNon-zero valid value percentiles (raw):")
        for p, v in zip(pcts, percentiles):
            print(f"  p{p}: {v:.0f} ({v * LWE_SCALE_FACTOR:.2f} mm)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cog_path> [cog_path ...]", file=sys.stderr)
        sys.exit(1)
    for path in sys.argv[1:]:
        check_cog(path)
        if len(sys.argv) > 2:
            print("\n" + "=" * 60 + "\n")
