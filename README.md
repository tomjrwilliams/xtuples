# xtuples

[![PyPI - Version](https://img.shields.io/pypi/v/xtuples.svg)](https://pypi.org/project/xtuples)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xtuples.svg)](https://pypi.org/project/xtuples)


-----

**Table of Contents**

- [Installation](#installation)
- [Overview](#overview)
- [Performance](#performance)
- [License](#license)

## Installation

```console
pip install xtuples
```

## Overview

```python
import typing
import xtuples as xt
```

xtuples (xt) provides:

- #1: xt.iTuple: a bunch of (primarily) itertools / functools boilerplate, wrapped into a tuple subclass.
- #2: xt.nTuple.decorate: a decorator to inject .pipe(), .partial(), and a dict of any user defined methodsm, into a NamedTuple.

### iTuple

The idea with #1 is to facilitate a functional style of programming, utilising method chaining to mimic the function pipelines seen in languages like f#.

For instance, a naive way to get all the squared primes under x, might be:

```python
def primes_under(x):
    return (
        xt.iTuple.range(x)
        .filter(lambda v: v > 1)
        .fold(lambda primes, v: (
            primes.append(v)
            if not primes.any(lambda prime: v % prime == 0)
            else primes
        ), initial=xt.iTuple())
    )

sq_primes_under_10 = primes_under(10).map(lambda v: v ** 2)
```

iTuple has type annotations for a reasonable range of overloads of common methods, such that mypy should, without requiring explicit annotations, be able to track types through methods like zip(), map(), filter(), and so on.

See ./tests/test_mypy.py for particular examples.

### nTuple

The idea with #2 is to ease method re-use between, and interface definitions on, NamedTuples, where rather than inheriting methods, we inject them in (on top of a type signature stub).

For instance, we can share the function f between NamedTuples A and B, as so:

```python
class Has_X(typing.Protocol):
    x: int

def f(self: Has_X) -> int:
    return self.x + 1

@xt.nTuple.decorate(f = reusable_f)
class A(typing.NamedTuple):

    x: int
    y: float

    def f(self) -> int: ...

@xt.nTuple.decorate(f = reusable_f)
class B(typing.NamedTuple):

    x: int
    y: float

    def f(self) -> int: ...
```

### JAX

Worth highlighting is the compatibility this promotes with [JAX](https://jax.readthedocs.io/en/latest/index.html), an auto-grad / machine learning framework from the Google Brain / Deepmind folks.

Because both iTuple and nTuple are kinds of tuple, JAX can take derivatives of / through without any further work (TODO: have a jax.register function on iTuple, so we don't need to call .pipe(tuple)).

Furthermore, because both iTuple and nTuple are immutable, generally very little refactoring is required for an xtuples code-base to be compliant with the (somewhat opinionated) functional purity requirements of the JAX jit compiler.

### xt.f

xtuples also exposes:

#3: xt.f: a module of sister methods to those in xt.iTuple, designed to provide / be used as arguments for their bound iTuple cousins.

For instance, instead of:

```python
cum_sq_primes_under_10 = (
    iTuple.range(10)
    .map(primes_under)
    .map(lambda it: it.map(lambda v: v ** 2))
)
```

We can write:

```python
cum_sq_primes_under_10 = (
    iTuple.range(10)
    .map(primes_under)
    .map(xt.f.map(lambda v: v ** 2))
)
```

Ie. it simply saves us from writing out quite so many lambdas.

## Performance

Performance using xtuples is generally not worse than a canonical equivalent implementation, and can sometimes be significantly better.

### iTuple

For instance, as iTuple is simply a subclass of the built-in tuple, it has very similar performance characteristics.

#### Creation

Creation is slightly slower than for an equivalent length list:


```python
%timeit xtuples.iTuple.range(10 ** 2)
%timeit list(range(10 ** 2))
```

    1.56 µs ± 9.4 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    588 ns ± 11.3 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    


```python
%timeit xtuples.iTuple.range(10 ** 3)
%timeit list(range(10 ** 3))
```

    11.9 µs ± 128 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    7.97 µs ± 60.2 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    


```python
%timeit xtuples.iTuple.range(10 ** 4)
%timeit list(range(10 ** 4))
```

    132 µs ± 1.51 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    89.6 µs ± 817 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    


```python
%timeit xtuples.iTuple.range(10 ** 6)
%timeit list(range(10 ** 6))
```

    40.9 ms ± 1.12 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    24.5 ms ± 424 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    

#### Memory

Whereas memory usage (comparable for small sizes), gets increasingly more efficient with size:


```python
memory = {}
for i in range(5):
    memory[i] = dict(
        iTuple=asizeof(xtuples.iTuple.range(10 ** i)),
        list=asizeof(list(range(10 ** i))),
    )
memory
```




    {0: {'iTuple': 160, 'list': 88},
     1: {'iTuple': 232, 'list': 448},
     2: {'iTuple': 952, 'list': 4048},
     3: {'iTuple': 8152, 'list': 40048},
     4: {'iTuple': 80152, 'list': 400048}}




```python
ex_iTuple = xtuples.iTuple.range(100)
ex_list = list(range(100))
ex_range = range(100)
```

#### Iteration & Indexing

Iteration is very similar:


```python
%timeit for x in ex_iTuple: pass
%timeit for x in ex_list: pass
```

    764 ns ± 4.57 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    721 ns ± 24.1 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    

And whilst elementwise indexing is clearly slower:


```python
%timeit for i in range(100): ex_iTuple[i]
%timeit for i in range(100): ex_list[i]
```

    18.6 µs ± 240 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    2.84 µs ± 75.3 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    

And so is slice indexing:


```python
%timeit ex_iTuple[10:20]
%timeit ex_list[10:20]
```

    667 ns ± 10.3 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    93.9 ns ± 0.998 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    

It is worth noting that per element indexing is not all that common using xtuples (as the canonical implementation is much more likely to use .map() and co).

#### Function application

Elementwise function application with .map() is *much* faster than the equivalent loop or list comprehension:


```python
add_2 = functools.partial(operator.add, 2)

def f_loop_map(f, l):
    res = []
    for v in l:
        res.append(f(v))
    return res

%timeit ex_iTuple.map(add_2)
%timeit f_loop_map(add_2, ex_list)
%timeit [add_2(x) for x in ex_list]
%timeit list(map(add_2, ex_list))
```

    5.57 µs ± 104 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    8.33 s ± 248 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    5.96 s ± 181 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    5.12 s ± 609 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

As is elementwise filtering:


```python
def f_loop_filter(f):
    res = []
    for i in ex_list:
        if f(i):
            res.append(i)
    return res

f = lambda x: x % 2 == 0

%timeit ex_iTuple.filter(f)
%timeit f_loop_filter(f)
%timeit [v for v in ex_list if f(v)]
```

    15.2 µs ± 41.6 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    6.68 s ± 139 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    7.13 s ± 113 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

And, so are both fold and cumulative fold:


```python
def f_loop_fold():
    acc = 0
    for i in ex_list:
        acc = operator.add(acc, i)
    return acc

%timeit ex_iTuple.fold(operator.add)
%timeit f_loop_fold()
```

    3.11 µs ± 48.5 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    4.87 s ± 89.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

So, as mentioned below, the answer to the poor .append() performance is just to use .foldcum() instead:


```python
def f_loop_foldcum():
    res = []
    acc = 0
    for i in ex_list:
        acc = operator.add(acc, i)
        res.append(acc)
    return res

%timeit ex_iTuple.foldcum(operator.add)
%timeit f_loop_foldcum()
```

    5.91 µs ± 19 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    15.2 s ± 1.67 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

#### Append

Appending is *much* slower, which is clearly to some extent a 'gotcha'.


```python
%timeit ex_iTuple.append(1)
%timeit ex_list.append(1)
```

    1.56 µs ± 22.9 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    69.3 ns ± 6.18 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    

Having said that, the canonical xtuples implementation is much more likely to use .map() .foldcum() or similar than .append().

And, as we've already seen, .map() and .foldcum() are *much* faster than the for-loop & append() implementations (so just do that instead - I personally also find it much more readable).

#### Prepend / Extend

Prepending to the tuple is *much* faster than with the list, though the relevant comparison is probably a deque (given that list is not at all optimised for left-append):


```python
%timeit ex_iTuple.prepend(1)
%timeit ex_list.insert(0, 1)
```

    1.44 µs ± 11 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    109 ms ± 10.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

Extend is somewhat slower (but is nowhere near as bad as append):


```python
%timeit xtuples.iTuple.range(100).extend([1])
%timeit xtuples.iTuple.range(100).extend(list(range(100)))
%timeit list(range(100)).extend([1])
%timeit list(range(100)).extend(list(range(100)))
```

    3.93 µs ± 32.3 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    5.58 µs ± 43.4 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    769 ns ± 8.37 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    1.48 µs ± 19 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    

And flatten is *much* faster:


```python
ex_iTuple_nested = ex_iTuple.map(lambda v: [v])
ex_list_nested = [[v] for v in ex_list]

def f_loop_flatten(l):
    for v in l:
        yield from v

%timeit ex_iTuple_nested.flatten()
%timeit list(f_loop_flatten(ex_list_nested))
%timeit list(itertools.chain(*ex_list_nested))
```

    5.38 µs ± 81.1 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    29.5 s ± 1.14 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    1min 19s ± 14.9 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

#### Summary

Overall, iTuple performance and memory usage is comparable - if not better - than a raw list.

The one clear weakness is .append() - however, the canonical xtuples implementation would use .map() .foldcum() etc. instead (which are actually *faster* than the equivalent .append() implementation).

### Named Tuple

nTuple does *not* (in comparison to iTuple) define a base class for us to sub-class.

Rather, it provides a decorator - nTuple.decorate - that adds .pipe() .partial() and a dict of user defined methods to a given NamedTuple.

As such, performance is essentially just that of built-in NamedTuples (ie. generally very strong).


```python
@xtuples.nTuple.decorate(
    update_x = lambda self, x: self._replace(x=x), 
    update_s = lambda self, s: self._replace(s=s),
)
class Example(typing.NamedTuple):
    x: int
    s: str
    
class Example_Cls:
    x: int
    s: str
    
    def __init__(self, x, s):
        self.x = x
        self.s = s

@dataclasses.dataclass(frozen=True, eq=True)
class Example_DC:
    x: int
    s: str
    
ex_nTuple = Example(1, "a")
ex_dict = dict(x=1, s="a")
ex_cls = Example_Cls(1, "a")
ex_dc = Example_DC(1, "a")
```

    NOTE, re-registering: Example
    

For instance, NamedTuples are significantly more memory efficient than any of the possible alternatives:


```python
dict(
    nTuple=asizeof(ex_nTuple),
    dict=asizeof(ex_dict),
    cls=asizeof(ex_cls),
    dataclass=asizeof(ex_dc),
)
```




    {'nTuple': 144, 'dict': 432, 'cls': 352, 'dataclass': 352}




```python
%timeit Example(1, "a")
%timeit dict(x=1, s="a")
%timeit Example_Cls(1, "a")
%timeit Example_DC(1, "a")
```

    257 ns ± 8.92 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    111 ns ± 1.53 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    209 ns ± 2.58 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    443 ns ± 3.05 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    

Whilst providing comparable (if not slightly faster) field access times:


```python
%timeit ex_nTuple.x
%timeit ex_nTuple[0]
%timeit ex_dict["x"]
%timeit ex_cls.x
%timeit ex_dc.x
```

    27.7 ns ± 0.303 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    25.1 ns ± 0.0815 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    29.7 ns ± 0.181 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    30.3 ns ± 0.492 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    34.9 ns ± 0.125 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    

Writes are, however, slower - the price we pay for immutability (but are still notably faster than the frozen dataclass equivalent):


```python
%timeit ex_dict["x"] = 1
%timeit ex_cls.x = 1
%timeit ex_nTuple._replace(x = 1)
&timeit ex_nTuple.update_x(x)
%timeit ex_nTuple.update(x=1)
%timeit dataclasses.replace(ex_dc, x=1)
```

    32.4 ns ± 0.467 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    43.8 ns ± 0.524 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    735 ns ± 5.93 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    1.03 µs ± 3.35 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    1.28 µs ± 4.37 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    

Like frozen dataclasses, NamedTuples are conveniently hashable (in comparison to dicts, for instance, which aren't), and do so based on value (versus standard classes which use object ids by default):


```python
dict(
    nTuple= hash(ex_nTuple) == hash(Example(1, "a")),
    cls= hash(ex_cls) == hash(Example_Cls(1, "a")),
    dataclass= hash(ex_dc) == hash(Example_DC(1, "a")),
)
```




    {'nTuple': True, 'cls': False, 'dataclass': True}



This is particularly useful in combination with iTuple, which is also hashable (making combinations of the two recursively hashable):


```python
@xtuples.nTuple.decorate
class Example_Nested(typing.NamedTuple):
    x: int
    s: str
    
    it: xtuples.iTuple
    
hash(Example_Nested(1, "s", xtuples.iTuple())) == hash(Example_Nested(1, "s", xtuples.iTuple()))
```




    True



Finally, sorting is both provided by default (again, in comparison to dicts and classes), and works as one would expect (ie. by the first field, then the second field, etc.):


```python
xtuples.iTuple([
    Example(2, "a"),
    Example(1, "b"),
    Example(1, "a"),
]).sort()
```




    iTuple(Example(x=1, s='a'), Example(x=1, s='b'), Example(x=2, s='a'))



## License

`xtuples` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
