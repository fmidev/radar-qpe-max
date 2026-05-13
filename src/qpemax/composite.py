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
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import Affine, array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, Resampling

# local
from qpemax.constants import COG_COMPRESS, EPSG_TARGET, LWE_SCALE_FACTOR, UINT16_FILLVAL

logger = logging.getLogger("airflow.task")

_EPOCH = np.datetime64("2000-01-01T00:00", "m")


def _parse_time_units(units_str: str) -> np.datetime64 | None:
    """Parse 'minutes since <ref>' to a datetime64, or None if invalid."""
    ref_str = units_str.replace("minutes since ", "").strip()
    if not ref_str or ref_str == "NaT":
        return None
    ref = np.datetime64(ref_str, "m")
    if np.isnat(ref):
        return None
    return ref


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


def _resolve_target_grid(
    datasets: list,
    bounds: tuple[float, float, float, float] | None,
    resolution: float | None,
    crs: str,
) -> tuple[Affine, int, int]:
    """Determine the output grid transform, width, and height.

    Uses rasterio.merge to infer the target grid from the inputs (same logic
    as composite_max) so both products are pixel-aligned.
    """
    merge_kwargs: dict = {"nodata": UINT16_FILLVAL, "method": "max"}
    if bounds is not None:
        merge_kwargs["bounds"] = bounds
    if resolution is not None:
        merge_kwargs["res"] = resolution
    _, transform = merge(datasets, **merge_kwargs)
    # Re-derive shape from bounds + transform
    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
    else:
        xmin, ymin, xmax, ymax = array_bounds(
            datasets[0].height, datasets[0].width, datasets[0].transform
        )
        for ds in datasets[1:]:
            b = array_bounds(ds.height, ds.width, ds.transform)
            xmin, ymin = min(xmin, b[0]), min(ymin, b[1])
            xmax, ymax = max(xmax, b[2]), max(ymax, b[3])
    res_x = transform.a
    res_y = -transform.e
    width = int(round((xmax - xmin) / res_x))
    height = int(round((ymax - ymin) / res_y))
    return transform, width, height


def composite_max_with_time(
    acc_paths: list[str | Path],
    time_paths: list[str | Path],
    acc_output_path: str | Path,
    time_output_path: str | Path,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    resolution: float | None = None,
    crs: str = f"EPSG:{EPSG_TARGET}",
) -> None:
    """Create composite max-accumulation and matching timestamp COGs.

    For each pixel the timestamp is taken from the radar that contributed
    the highest accumulation value.  Timestamp encoding (uint16 minutes
    since a reference) is decoded, the correct time selected, and then
    re-encoded in the output.

    Parameters
    ----------
    acc_paths
        Paths to single-radar max-accumulation COGs (uint16, scale=0.01).
    time_paths
        Paths to matching timestamp COGs (uint16, minutes since per-file ref).
        Must be in the same order as *acc_paths*.
    acc_output_path
        Path for the output composite accumulation COG.
    time_output_path
        Path for the output composite timestamp COG.
    bounds, resolution, crs
        Target grid parameters (same semantics as composite_max).
    """
    acc_output_path = Path(acc_output_path)
    time_output_path = Path(time_output_path)
    acc_output_path.parent.mkdir(parents=True, exist_ok=True)
    time_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not acc_paths:
        raise ValueError("No input paths provided for compositing.")
    if len(acc_paths) != len(time_paths):
        raise ValueError("acc_paths and time_paths must have the same length.")

    n = len(acc_paths)

    # Open all inputs, wrap in WarpedVRT to normalize orientation
    raw_acc = [rasterio.open(p) for p in acc_paths]
    raw_time = [rasterio.open(p) for p in time_paths]
    vrt_acc = [WarpedVRT(ds) for ds in raw_acc]
    vrt_time = [WarpedVRT(ds) for ds in raw_time]

    try:
        # Determine target grid from accumulation inputs
        transform, width, height = _resolve_target_grid(
            vrt_acc, bounds, resolution, crs
        )
        dst_crs = CRS.from_user_input(crs)

        # Reproject all inputs onto the target grid
        acc_stack = np.full((n, height, width), UINT16_FILLVAL, dtype=np.uint16)
        time_stack = np.zeros((n, height, width), dtype=np.uint16)

        for i, (acc_ds, time_ds) in enumerate(zip(vrt_acc, vrt_time)):
            reproject(
                source=rasterio.band(acc_ds, 1),
                destination=acc_stack[i],
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=UINT16_FILLVAL,
                resampling=Resampling.nearest,
            )
            reproject(
                source=rasterio.band(time_ds, 1),
                destination=time_stack[i],
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
    finally:
        for ds in vrt_acc + vrt_time + raw_acc + raw_time:
            ds.close()

    # Compute composite: pixel-wise max of accumulation
    # Replace nodata with 0 for argmax comparison so nodata never wins
    acc_compare = np.where(acc_stack == UINT16_FILLVAL, 0, acc_stack)
    best_idx = np.argmax(acc_compare, axis=0)
    composite_acc = np.take_along_axis(
        acc_stack, best_idx[np.newaxis], axis=0
    )[0]

    # Decode timestamps to absolute minutes-since-epoch, pick the one
    # matching the best radar, then re-encode.
    # Each time COG stores minutes since its own reference time (from 'units').
    # We decode to int64 minutes since a common epoch, select, and re-encode.
    abs_time = np.zeros((n, height, width), dtype=np.int64)
    for i, tp in enumerate(time_paths):
        with rasterio.open(tp) as ds:
            units_str = ds.tags().get("units", "") or ds.tags(1).get("units", "")
        ref = _parse_time_units(units_str)
        if ref is None:
            # No valid reference — this radar has no timestamp data
            continue
        offset_from_epoch = int((ref - _EPOCH) / np.timedelta64(1, "m"))
        # time_stack[i] has encoded values: 0=nodata, valid = minutes + 1
        valid = time_stack[i] > 0
        abs_time[i] = np.where(valid, time_stack[i].astype(np.int64) - 1 + offset_from_epoch, 0)

    # Select timestamp from the best radar at each pixel
    composite_abs_time = np.take_along_axis(
        abs_time, best_idx[np.newaxis], axis=0
    )[0]

    # Re-encode: find new tmin, encode as uint16 minutes since tmin + 1
    valid_mask = composite_abs_time > 0
    if valid_mask.any():
        tmin_minutes = int(composite_abs_time[valid_mask].min())
    else:
        tmin_minutes = 0
    tmin_dt = _EPOCH + np.timedelta64(tmin_minutes, "m")
    tunits = f"minutes since {tmin_dt}"

    composite_time_encoded = np.where(
        valid_mask,
        (composite_abs_time - tmin_minutes + 1).astype(np.uint16),
        0,
    ).astype(np.uint16)

    # Where all radars had nodata for accumulation, set both to nodata
    all_nodata = np.all(acc_stack == UINT16_FILLVAL, axis=0)
    composite_acc[all_nodata] = UINT16_FILLVAL
    composite_time_encoded[all_nodata] = 0

    # Write accumulation COG
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
    with rasterio.open(acc_output_path, "w", **profile) as dst:
        dst.scales = (LWE_SCALE_FACTOR,)
        dst.offsets = (0.0,)
        dst.write(composite_acc, 1)

    # Write timestamp COG
    time_profile = {
        "driver": "COG",
        "dtype": "uint16",
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": COG_COMPRESS,
    }
    with rasterio.open(time_output_path, "w", **time_profile) as dst:
        dst.update_tags(units=tunits)
        dst.update_tags(1, units=tunits)
        dst.write(composite_time_encoded, 1)

    logger.info(
        "Wrote composite COGs: %s, %s (%d inputs, %dx%d)",
        acc_output_path,
        time_output_path,
        n,
        width,
        height,
    )


def composite_time_from_max(
    acc_composite_path: str | Path,
    acc_paths: list[str | Path],
    time_paths: list[str | Path],
    time_output_path: str | Path,
    *,
    crs: str = f"EPSG:{EPSG_TARGET}",
) -> None:
    """Derive a composite timestamp COG by matching radars to an existing composite.

    Reads the pre-computed accumulation composite and determines which
    single-radar COG contributed each pixel by matching values.  The
    timestamp is then taken from the corresponding radar's time COG.

    Parameters
    ----------
    acc_composite_path
        Path to the composite accumulation COG (output of composite_max).
    acc_paths
        Paths to single-radar max-accumulation COGs (same inputs used for
        composite_max).
    time_paths
        Paths to matching timestamp COGs.  Must be in the same order as
        *acc_paths*.
    time_output_path
        Path for the output composite timestamp COG.
    crs
        Target CRS (should match the composite).
    """
    time_output_path = Path(time_output_path)
    time_output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(acc_paths) != len(time_paths):
        raise ValueError("acc_paths and time_paths must have the same length.")

    n = len(acc_paths)

    # Read composite grid properties
    with rasterio.open(acc_composite_path) as comp_ds:
        comp_acc = comp_ds.read(1)
        transform = comp_ds.transform
        width = comp_ds.width
        height = comp_ds.height
        dst_crs = comp_ds.crs
        comp_nodata = comp_ds.nodata
        if comp_nodata is None:
            comp_nodata = UINT16_FILLVAL

    # Reproject each radar's acc and time onto the composite grid
    acc_stack = np.full((n, height, width), UINT16_FILLVAL, dtype=np.uint16)
    time_stack = np.zeros((n, height, width), dtype=np.uint16)

    for i, (acc_p, time_p) in enumerate(zip(acc_paths, time_paths)):
        with rasterio.open(acc_p) as ds:
            with WarpedVRT(ds) as vrt:
                reproject(
                    source=rasterio.band(vrt, 1),
                    destination=acc_stack[i],
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    dst_nodata=UINT16_FILLVAL,
                    resampling=Resampling.nearest,
                )
        with rasterio.open(time_p) as ds:
            with WarpedVRT(ds) as vrt:
                reproject(
                    source=rasterio.band(vrt, 1),
                    destination=time_stack[i],
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )

    # For each pixel, find which radar matches the composite value.
    # Use argmax on a boolean match array; ties go to the first radar.
    comp_nodata_int = int(comp_nodata)
    matches = acc_stack == comp_acc[np.newaxis, :, :]
    # Where composite is nodata, no radar should match
    nodata_mask = comp_acc == comp_nodata_int
    matches[:, nodata_mask] = False
    # If no radar matches (shouldn't happen, but be safe), fall back to argmax
    any_match = matches.any(axis=0)
    best_idx = np.argmax(matches, axis=0)
    # Fallback: where no match found, use pixel-wise argmax of acc
    acc_compare = np.where(acc_stack == UINT16_FILLVAL, 0, acc_stack)
    fallback_idx = np.argmax(acc_compare, axis=0)
    best_idx = np.where(any_match, best_idx, fallback_idx)

    # Decode timestamps to absolute time, select, and re-encode
    abs_time = np.zeros((n, height, width), dtype=np.int64)
    for i, tp in enumerate(time_paths):
        with rasterio.open(tp) as ds:
            units_str = ds.tags().get("units", "") or ds.tags(1).get("units", "")
        ref = _parse_time_units(units_str)
        if ref is None:
            continue
        offset_from_epoch = int((ref - _EPOCH) / np.timedelta64(1, "m"))
        valid = time_stack[i] > 0
        abs_time[i] = np.where(valid, time_stack[i].astype(np.int64) - 1 + offset_from_epoch, 0)

    composite_abs_time = np.take_along_axis(
        abs_time, best_idx[np.newaxis], axis=0
    )[0]

    # Re-encode
    valid_mask = composite_abs_time > 0
    if valid_mask.any():
        tmin_minutes = int(composite_abs_time[valid_mask].min())
    else:
        tmin_minutes = 0
    tmin_dt = _EPOCH + np.timedelta64(tmin_minutes, "m")
    tunits = f"minutes since {tmin_dt}"

    composite_time_encoded = np.where(
        valid_mask,
        (composite_abs_time - tmin_minutes + 1).astype(np.uint16),
        0,
    ).astype(np.uint16)

    # Nodata where composite acc is nodata
    composite_time_encoded[nodata_mask] = 0

    # Write timestamp COG
    time_profile = {
        "driver": "COG",
        "dtype": "uint16",
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": COG_COMPRESS,
    }
    with rasterio.open(time_output_path, "w", **time_profile) as dst:
        dst.update_tags(units=tunits)
        dst.update_tags(1, units=tunits)
        dst.write(composite_time_encoded, 1)

    logger.info(
        "Wrote composite time COG: %s (%d inputs, %dx%d)",
        time_output_path,
        n,
        width,
        height,
    )
