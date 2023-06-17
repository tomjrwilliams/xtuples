# xtuples

[![PyPI - Version](https://img.shields.io/pypi/v/xtuples.svg)](https://pypi.org/project/xtuples)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xtuples.svg)](https://pypi.org/project/xtuples)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install xtuples
```

## xtuples

xtuples is designed to make functional programming easier in Python.

The two key constructs are:

- xtuples.iTuple: a base class for iterable data types, equipped with methods like .map(), .fold(), .pipe() and .partial()
- xuples.nTuple.decorate: a decorator to inject .pipe() and .partial() into user-defined NamedTuples (as we can't subclass them directly).

We expose:

- xtuples.iTuple: an iterable wrapper around tuple, 
- xtuples.nTuple.decorate(): a decorator for user-defined named tuples (as we can't subclass them), exposing .pipe() and .partial()
- xtuples.json.JSONEncoder and ...JSONDecoder: base classes for weakly-rich json encoding / decoding (rich in that classes are preserved, weak in that this is on class name alone and no further checks or guarantees are provided)

## License

`xtuples` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
