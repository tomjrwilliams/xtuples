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

- xtuples.iTuple: a base class for iterable data types, equipped with methods like .map(), .fold(), .pipe() and .partial().

```python
assert (
    iTuple.range(5)
    .map(lambda x: x * 2)
    .filter(lambda x: x < 5)
    .accumulate(operator.sum)
    .pipe(sum)
) == 8
```

- xuples.nTuple.decorate: a decorator to inject .pipe() and .partial() into user-defined NamedTuples (as we can't subclass them directly).

```python
@nTuple.decorate
class Example(collections.NamedTuple):
    x: int

assert Example(1, "a").pipe(lambda obj: obj.int) == 1
```

As briefly demonstrated above, by equipping our data structures with .pipe() and .partial(), we're able to use method chaining to mimic the functional pipelines seen in languages like f#.

This then tends to lead us away from inheritance, and more towards composition: to a code base comprised entirely of either free functions, or descendants of either of the above.

### Performance

Performance should generally be at worst, not worse than a non-optimised canonical equivalent (and can often be significantly better).

For instance, NamedTuples provide faster access, and are more memory efficient, than standard classes (or even raw dicts).

< Insert demo >

Similarly, iTuple - as a subclass of tuple - is more memory efficient, and provides as fast access, than a standard raw list.

< Insert demo >

The one caveat to this is that some methods which are canonically lazy, returning generators, are treated eagerly - however this is something I am actively working on.

### xtuples.json

xtuples.json provides base classes for weakly-rich json encoding / decoding (rich in that classes are preserved, weak in that this is based on class name alone and no further checks or guarantees are provided).

## License

`xtuples` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
