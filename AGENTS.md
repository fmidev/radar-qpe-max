# Agent Instructions

## Project

`qpemax` computes QPE (Quantitative Precipitation Estimation) max-accumulation statistics from ODIM HDF5 radar files, producing Cloud Optimized GeoTIFFs. Key pipeline: ODIM H5 → `pyart` Radar → gridded `xarray`/`rioxarray` (EPSG:3067) → netCDF cache → sliding-window max → COG GeoTIFF. The process is really resource intensive requiring advanced dask-fu and chunking to even run on most machines.

CLI entry point: `qpe` (subcommands: `grid`, `winmax`). See [README.md](README.md).

## Key principles
- Don't assume. Don't hide confusion. Surface tradeoffs.
- No speculation-driven extrapolation. If you don't know, say so. Or ask the user.

## Commands

```sh
# Run in development environment
/home/tiira/.virtualenvs/qpemax/bin/python

# Build container
podman build -t qpemax .

# Run the main program
/home/tiira/.virtualenvs/qpemax/bin/qpe
```

## Testing and debugging
* Prefer agent hooks to run tests
* Install only in the virtualenv

## Code Style

* **Python ≥ 3.14**: use modern syntax freely (e.g. `X | Y` unions, `list[X]`/`dict[K, V]` generics, `type` aliases)
* **Formatter: [Black](https://black.readthedocs.io/)** (default line length: 88)
* **Type hints**: Use the most modern convention supported
* Avoid trivial functions
* Succinct, to the point documentation

### Airflow integration

This package is deployed as a containerized service in FMI's Airflow v2.11 radar production system. The integration pattern:

- **Deployment**: Docker container with this package installed (built via `Containerfile`, image `quay.io/fmi/sademaksit:vx.y.z`).
- **Airflow tasks**: `@task.docker` decorator is used to invoke Python API.
- **No DAGs in this repo**: Workflow orchestration lives in the separate Airflow radar production repository
- **Robustness**: Handle missing/corrupted input files and edge cases gracefully, log processing steps

## Project Layout

| Path | Purpose |
|------|---------|
| `src/qpemax/__init__.py` | Package init; exports version, constants, public API |
| `src/qpemax/cli.py` | CLI entry point (`grid`, `winmax` subcommands) |
| `src/qpemax/grid.py` | ODIM H5 → pyart → gridded xarray (EPSG:3067) |
| `src/qpemax/accumulate.py` | Sliding-window max-accumulation via xarray/dask |
| `src/qpemax/composite.py` | Multi-radar national composite (pixel-wise max) |
| `src/qpemax/output.py` | Write xarray datasets to COG GeoTIFF |
| `src/qpemax/constants.py` | EPSG, field names, encoding/scale defaults |
| `src/qpemax/callbacks.py` | Dask progress/memory logging |
| `src/qpemax/logs.py` | Logging utilities |
| `src/qpemax/utils.py` | Filename generation, suffix/cache helpers |
| `tests/test_qpemax.py` | Pytest tests |

## Version

Version is derived from git tags via `hatch-vcs`; do not edit `src/qpemax/_version.py` manually.
