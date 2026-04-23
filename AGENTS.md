# Agent Instructions

## Project

`qpemax` computes QPE (Quantitative Precipitation Estimation) max-accumulation statistics from ODIM HDF5 radar files, producing Cloud Optimized GeoTIFFs. Key pipeline: ODIM H5 → `pyart` Radar → gridded `xarray`/`rioxarray` (EPSG:3067) → netCDF cache → sliding-window max → COG GeoTIFF. The process is really resource intensive requiring advanced dask-fu and chunking to even run on most machines.

CLI entry point: `qpe` (subcommands: `grid`, `winmax`). See [README.md](README.md).

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
* Install only in virtualenv

## Code Style

* **Python ≥ 3.14**: use modern syntax freely (e.g. `X | Y` unions, `list[X]`/`dict[K, V]` generics, `type` aliases)
* **Formatter: [Black](https://black.readthedocs.io/)** (default line length: 88)
* **Type hints**: Use the most modern convention supported
* Avoid trivial functions
* Succinct, to the point documentation

### Import grouping

Imports are manually grouped with inline comments:

```python
# builtin
import os

# pypi
import numpy as np

# local
from qpemax import something
```

### Airflow integration

This package is deployed as a containerized service in FMI's Airflow v2.11 radar production system. The integration pattern:

- **Deployment**: Docker container with this package installed (built via `Containerfile`, image `quay.io/fmi/sademaksit:vx.y.z`).
- **Airflow tasks**: `@task.docker` decorator is used to invoke Python API.
- **No DAGs in this repo**: Workflow orchestration lives in the separate Airflow radar production repository
- **Robustness**: Handle missing/corrupted input files and edge cases gracefully, log processing steps

## Project Layout

| Path | Purpose |
|------|---------|
| `src/qpemax/cli.py` | CLI entry point |
| `src/qpemax/callbacks.py` | Dask progress/memory logging |
| `src/qpemax/logs.py` | Logging utilities |
| `tests/test_qpemax.py` | Pytest tests |

## Version

Version is derived from git tags via `hatch-vcs`; do not edit `src/qpemax/_version.py` manually.
