# maksitiirain
Statistical indicators of radar QPE over moving temporal windows 

## Installation

```console
pip install .
```

## Usage
see

```console
qpe --help
```

Container example:
```console
podman run --rm -v=$HOME/data/polar/fivih/:/data:z -v=$HOME/results/sademaksit/:/output:z --tmpfs=/tmp quay.io/fmi/sademaksit:v0.10.2 winmax -i /data/{date}*.h5 -o /output -s 512 -w 1H 20220805
```

## License

`maksitiirain` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
