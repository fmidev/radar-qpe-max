# maksitiirain
[![Docker Repository on Quay](https://quay.io/repository/fmi/sademaksit/status "Docker Repository on Quay")](https://quay.io/repository/fmi/sademaksit)

Statistical indicators of radar QPE over moving temporal windows 

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
podman run --rm -v=$HOME/data/polar/fivih/:/data:z -v=$HOME/results/sademaksit/:/output:z --tmpfs=/tmp quay.io/fmi/sademaksit:v0.10.2 winmax -i /data/{date}*.h5 -o /output -s 512 -w 1H 20220805
```

## License

`maksitiirain` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
