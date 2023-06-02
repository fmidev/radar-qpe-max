# sademaksit
Statistical indicators of radar QPE over moving temporal windows 

-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

```console
pip install .
```

## Usage

Container example:
```console
podman run --rm -v=$HOME/data/polar/fivih/:/data:z -v=$HOME/results/sademaksit/:/output:z --tmpfs=/tmp maksit:latest -i /data/{date}*.h5 -o /output -s 512 -w 1H 20220805
```

## License

`sademaksit` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
