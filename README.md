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

## Output files

Daily max products follow the FMI radar GeoTIFF naming convention
(see `qpemax.max_tif_name`):

```
{timestamp}_{site}_{product}_{parameter}_{quantity}_{geoconf}_{filter}.tif
202610050000_fikor_max_1h_acrr_finrad250_raw.tif
```

| Segment | Values |
|---------|--------|
| timestamp | End of the UTC day, `%Y%m%d%H%M`: the product of 2026-10-04 is `202610050000`. The max is taken over all sliding windows, in 5 min steps, whose last time step falls on that day. |
| site | Radar node name, e.g. `fikor`, or `composite` |
| product | `max` (max accumulation) or `maxtime` (time of the max) |
| parameter | Window length in whole hours, e.g. `1h`, `24h` (`-w 1D`) |
| quantity | `acrr` |
| geoconf | `finrad{resolution}`, e.g. `finrad250` |
| filter | `raw` for DBZH, `rawac` for attenuation corrected DBZHC |

`qpemax.max_tif_glob` gives a glob pattern for the per-site `max` files of a date and window.
It also matches the `composite` file, so exclude paths containing `_composite_` when it shares the directory.

5 min single-scan GeoTIFFs and cache files keep their previous naming.

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
