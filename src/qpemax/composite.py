# SPDX-FileCopyrightText: 2023-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Composite QPE max-accumulation from multiple radars.

Merges single-radar QPE max COGs into a national composite using
pixel-wise maximum (highest accumulation wins).
"""

# builtin
import logging
from pathlib import Path

# pypi
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT

# local
from qpemax.constants import COG_COMPRESS, EPSG_TARGET, LWE_SCALE_FACTOR, UINT16_FILLVAL

logger = logging.getLogger("airflow.task")


def composite_max(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    resolution: float | None = None,
    crs: str = f"EPSG:{EPSG_TARGET}",
) -> None:
    """Create a composite max-accumulation field from single-radar COGs.

    Merges all input rasters onto a fixed grid using pixel-wise maximum.
    Cells not covered by any radar are set to nodata.

    Parameters
    ----------
    input_paths
        Paths to single-radar max-accumulation COGs (uint16).
    output_path
        Path for the output composite COG.
    bounds
        Grid extent (xmin, ymin, xmax, ymax) in the target CRS.
        If None, the union of input extents is used.
    resolution
        Grid cell size [m]. If None, inferred from the input COGs
        (finest resolution among inputs).
    crs
        Target CRS.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_paths:
        raise ValueError("No input paths provided for compositing.")

    raw_datasets = [rasterio.open(p) for p in input_paths]
    # WarpedVRT normalizes upside-down (positive pixel height) transforms
    # to standard north-up orientation required by rasterio.merge.
    datasets = [WarpedVRT(ds) for ds in raw_datasets]
    try:
        merge_kwargs: dict = {
            "nodata": UINT16_FILLVAL,
            "method": "max",
        }
        if bounds is not None:
            merge_kwargs["bounds"] = bounds
        if resolution is not None:
            merge_kwargs["res"] = resolution
        mosaic, transform = merge(datasets, **merge_kwargs)
    finally:
        for ds in datasets:
            ds.close()
        for ds in raw_datasets:
            ds.close()

    height, width = mosaic.shape[1], mosaic.shape[2]

    profile = {
        "driver": "COG",
        "dtype": "uint16",
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": UINT16_FILLVAL,
        "compress": COG_COMPRESS,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.scales = (LWE_SCALE_FACTOR,)
        dst.offsets = (0.0,)
        dst.write(mosaic[0].astype(np.uint16), 1)

    logger.info(
        "Wrote composite COG: %s (%d inputs, %dx%d)",
        output_path,
        len(input_paths),
        width,
        height,
    )
