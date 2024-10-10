# radar-qpe-max
[![Docker Repository on Quay](https://quay.io/repository/fmi/sademaksit/status "Docker Repository on Quay")](https://quay.io/repository/fmi/sademaksit)

Statistical indicators of radar QPE over moving temporal windows.

The tool can be used to find the temporal window with the maximum precipitation accumulation for each grid point.
The output is cloud optimized GeoTIFF (COG) files with the maximum precipitation accumulation and the time of the maximum precipitation accumulation.
Additionally, 5 minute precipitation accumulation COG files are created.

## Installation

```shell
pip install .
```

## Usage
see

```shell
qpe --help
```

Container example:
```shell
podman run --rm -v=$HOME/data/polar/fivih/:/data:z -v=$HOME/results/sademaksit/:/output:z --tmpfs=/tmp quay.io/fmi/sademaksit:v0.11.0 winmax -i /data/{date}*.h5 -o /output -s 512 -w 1H 20220805
```
