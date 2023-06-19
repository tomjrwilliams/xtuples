# xtuples

[![PyPI - Version](https://img.shields.io/pypi/v/xtuples.svg)](https://pypi.org/project/xtuples)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xtuples.svg)](https://pypi.org/project/xtuples)


-----

**Table of Contents**

- [Installation](#installation)
- [Overview](#overview)
- [Examples](#examples)
- [Performance](#performance)
- [JSON](#xtuples.json)
- [License](#license)

## Installation

```console
pip install xtuples
```

## Overview

xtuples is designed to make functional programming easier in Python.

In particular, it is designed to enable one to mimic the function pipelines seen in languages like f#, but using method chaining.

The two key constructs are:

- xtuples.iTuple: a base class for iterable data types, equipped with methods like .map() .filter() and .fold().

- xuples.nTuple.decorate: a decorator to inject .pipe() and .partial() into user-defined NamedTuples (as we can't subclass them directly).

Taken together, these tend to lead us away from inheritance, and more towards composition: to a code base comprised entirely of either free functions, or (immutable) data structures implemented using either of the above.

## Examples

The value of this approach is best illustrated with a couple of examples.


```python
from src import xtuples
```


```python
import typing
import operator
import functools
```

For instance, let's imagine that we're reading in a list of prices from a csv file (which we'll mock here for convenience):


```python
def mock_read_prices():
    for ticker, price in {
        "IBM US Equity": 100,
        "AAPL US Equity": 105,
        "F US Equity": 95,
    }.items():
        yield ticker, price
```

We can read each row into a decorated NamedTuple:


```python
@xtuples.nTuple.decorate
class Ticker_Price(typing.NamedTuple):
    ticker: str
    price: float
```

By mapping over the (mocked) csv file iterator:


```python
prices = (
    xtuples.iTuple(mock_read_prices())
    .mapstar(Ticker_Price)
    .pipe(print, discard=True)
)
```

    iTuple(Ticker_Price(ticker='IBM US Equity', price=100), Ticker_Price(ticker='AAPL US Equity', price=105), Ticker_Price(ticker='F US Equity', price=95))
    

Here, we pipe into but discard the output of print(), so we still get back out and iTuple of Ticker_Prices (even whilst print() would otherwise return None).

Let's say that we want to map the prices into some new currency:


```python
def convert_price(ticker, price, fx):
    return price * fx
```

We can do this with a combination of .map() and nTuple.replace(), where we again pipe into but discard the output of print():


```python
prices = (
    prices.mapstar(functools.partial(convert_price, fx=0.9))
    .map(xtuples.nTuple.replace("price"), prices, at = 1)
    .pipe(print, discard=True)
)
```

    iTuple(Ticker_Price(ticker='IBM US Equity', price=90.0), Ticker_Price(ticker='AAPL US Equity', price=94.5), Ticker_Price(ticker='F US Equity', price=85.5))
    

As one can see, this code is significantly more concise than the canonical for-loop implementation would be.


```python
def f_loop_convert_price(prices, fx):
    res = []
    for obj in prices:
        obj = obj._replace(price=convert_price(ojb.ticker, obj.price, fx))
        res.append(obj)
    return res
# prices = f_loop_conver_price(prices, 0.9)
# print(prices)
```

Personally, it is also more readable - though that's obviously (to some extent) personal taste.

## Performance

Performance using xtuples should generally be, at worst, not (materially) worse than a non-optimised canonical equivalent (and can sometimes be somewhat better).


```python
import dataclasses
import collections
import functools
import itertools
import timeit
from pympler.asizeof import asizeof
```

### iTuple

For instance, iTuple is a relatively minimal wrapper around the built in tuple.

#### Creation & Memory

As such, creation takes a fairly similar time to that of a raw list (a raw tuple would probably be faster):


```python
%timeit xtuples.iTuple.range(100)
%timeit list(range(100))
```

    1.85 µs ± 26 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    560 ns ± 2.41 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    

Memory usage is very similar:


```python
memory = {}
for i in range(5):
    memory[i] = dict(
        iTuple=asizeof(xtuples.iTuple.range(10 ** i)),
        list=asizeof(list(range(10 ** i))),
    )
memory
```




    {0: {'iTuple': 280, 'list': 88},
     1: {'iTuple': 640, 'list': 448},
     2: {'iTuple': 4240, 'list': 4048},
     3: {'iTuple': 40240, 'list': 40048},
     4: {'iTuple': 400240, 'list': 400048}}




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

    875 ns ± 7.52 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    713 ns ± 9.48 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    

And whilst elementwise indexing is clearly slower (although referencing the .data field directly is not too far behind):


```python
%timeit for i in range(100): ex_iTuple[i]
%timeit for i in range(100): ex_iTuple.data[i]
%timeit for i in range(100): ex_list[i]
```

    11.1 µs ± 198 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    4.58 µs ± 66.8 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    3.18 µs ± 41.8 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    

And so is slice indexing:


```python
%timeit ex_iTuple[10:20]
%timeit ex_iTuple.data[10:20]
%timeit ex_list[10:20]
```

    165 ns ± 0.993 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    98.9 ns ± 0.787 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    96.1 ns ± 1.07 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    

It is worth noting that per element indexing is not all that common using xtuples (as the canonical implementation is much more likely to use .map() and co).

#### Append / Extend

Appending is *much* slower, which is clearly to some extent a 'gotcha'.


```python
%timeit ex_iTuple.append(1)
%timeit ex_list.append(1)
```

    1.86 µs ± 14.6 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    67.5 ns ± 4.63 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    

Having said that, the canonical xtuples implementation is much more likely to use .map() .foldcum() or similar than .append().

And, as you can see below, .map() and .foldcum() are actually *faster* than the for-loop & append() implementations.

So, as with elementwise indexing, in the context of a canonical implementation of an entire function, performance is generally on par (if not better) with xtuples as with the equivalent built-ins.

Prepending to the tuple is *much* faster than with the list, though the relevant comparison is probably a deque (given that list is not at all optimised for left-append):


```python
%timeit ex_iTuple.prepend(1)
%timeit ex_list.insert(0, 1)
```

    1.77 µs ± 13.2 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    113 ms ± 4.71 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    

Extend is somewhat slower (but is nowhere near as bad as append):


```python
%timeit xtuples.iTuple.range(100).extend([1])
%timeit xtuples.iTuple.range(100).extend(list(range(100)))
%timeit list(range(100)).extend([1])
%timeit list(range(100)).extend(list(range(100)))
```

    6.07 µs ± 85.2 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    8.79 µs ± 139 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    749 ns ± 3.83 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    1.44 µs ± 11.6 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    

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

    6.95 µs ± 266 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    22.8 s ± 1.49 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    45.8 s ± 5.07 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

#### Function application

Finally, elementwise function application with .map() is *much* faster than the equivalent loop or list comprehension:


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

    6.87 µs ± 389 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    8.54 s ± 65.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    6.02 s ± 47.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    4.76 s ± 62.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

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

    11.7 µs ± 146 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    6.9 s ± 18.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    7.23 s ± 39.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

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

    3.03 µs ± 18.6 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    5.21 s ± 3.05 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

Hence, as mentioned above, the answer to the poor .append() performance is often just to use .foldcum() instead:


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

    7.08 µs ± 34 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    11.4 s ± 293 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

#### Summary

Overall, iTuple performance and memory usage is comparable - if not better - than a raw list.

The one clear weak point is .append().

But, *if used as intended*, the canonical xtuples implementation would instead likely be using .map() .foldcum() and co.

Given that .map() .filter() .fold() and .foldcum() are generally *much* faster than the equivalent for loops or list comprehensions, performance is often actually *better* than the equivalent implementation using only built-ins.

### Eager evaluation

The one caveat worth highlighting is that many canonically lazy methods, which would standardly return generators (or similar), are instead treated eagerly - however this is something I am actively working on.

### Named Tuple

nTuple does *not* (in comparison to iTuple) define a base class for us to sub-class.

Rather, it provides a decorator - nTuple.decorate - that adds .pipe() and .partial() to user defined NamedTuples.

As such, performance is essentially just that of built-in NamedTuples (ie. generally very strong).


```python
@xtuples.nTuple.decorate
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



Finally, sorting is both provided by default (again, in comparison to dicts and classes), and works as one would expect (ie. by the first field, then the second field, and so on):


```python
xtuples.iTuple([
    Example(2, "a"),
    Example(1, "b"),
    Example(1, "a"),
]).sort()
```




    iTuple(Example(x=1, s='a'), Example(x=1, s='b'), Example(x=2, s='a'))



## xtuples.json

xtuples.json provides base classes for weakly-rich json encoding / decoding (rich in that classes are preserved, weak in that this is based on class name alone and no further checks or guarantees are provided).

## License

`xtuples` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
