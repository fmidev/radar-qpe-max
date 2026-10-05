# radar-qpe-max
[![Docker Repository on Quay](https://quay.io/repository/fmi/sademaksit/status "Docker Repository on Quay")](https://quay.io/repository/fmi/sademaksit)

Statistical indicators of radar QPE over moving temporal windows.

The tool can be used to find the temporal window with the maximum precipitation accumulation for each grid point.
The output is cloud optimized GeoTIFF (COG) files with the maximum precipitation accumulation and the time of the maximum precipitation accumulation.
Additionally, 5 minute precipitation accumulation COG files are created.

## Grid geometry

Per-site grids are `size` x `size` pixels (`size` must be even) of exactly `resolution` metres in EPSG:3067, written north-up.
Pixel edges lie on a lattice of multiples of `resolution`, and each grid is centred on the lattice node nearest to the radar.
The grids of different radars therefore line up with each other and with the finrad composite bounds
`[-208000, 6390000, 1072000, 7926000]` (5120 x 6144 px at 250 m), so compositing needs no resampling.

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
