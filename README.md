# radar-qpe-max
[![Docker Repository on Quay](https://quay.io/repository/fmi/sademaksit/status "Docker Repository on Quay")](https://quay.io/repository/fmi/sademaksit)

Statistical indicators of radar QPE over moving temporal windows.

The tool can be used to find the temporal window with the maximum precipitation accumulation for each grid point.
The output is cloud optimized GeoTIFF (COG) files with the maximum precipitation accumulation and the time of the maximum precipitation accumulation.
Additionally, 5 minute precipitation accumulation COG files are created.

## Installation

### Locally

```shell
pip install .
```

### Container

```shell
# Build
podman build -t qpemax .
# Verify
podman run --rm qpemax --help
```

## Usage
Python module API:

```python
import qpemax
```

Command line interface:

```shell
qpe --help
```
