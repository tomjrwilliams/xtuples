
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsDunderLT, SupportsDunderGT

# ---------------------------------------------------------------

import abc
import typing
import dataclasses
import collections

import operator
import itertools
import functools

from .i import iTuple, T, CT

# these are for use in .map() etc.
# so should return a func that accepts the relevant value, where it takes args / kwargs
# if no args / kwargs, not necessary

# ---------------------------------------------------------------

def empty(cls=iTuple) -> iTuple:
    return cls.empty()

def one(v, cls=iTuple) -> iTuple:
    return cls.one(v)

def none(n, cls = iTuple):
    return cls.none(n)

def irange(*args, cls = iTuple, **kwargs):
    return cls.range(*args, **kwargs)

def from_keys(d, cls = iTuple):
    return cls.from_keys(d)
    
def from_values(d, cls = iTuple):
    return cls.from_values(d)
    
def from_items(d, cls = iTuple):
    return cls.from_items(d)

def from_index(s, cls = iTuple):
    return cls.from_index(s)

def from_index_values(s, cls = iTuple):
    return cls.from_index_values(s)

def from_columns(s, cls = iTuple):
    return cls.from_columns(s)

# -----

def partial(self, f, *args, **kwargs):
    return self.partial(f, *args, **kwargs)

# -----

def index_of(self, v):
    return self.index_of(v)

def len_range(self):
    return self.len_range()

def append(self, value, *values):
    return self.append(value, *values)

def prepend(self, value, *values):
    return self.prepend(value, *values)

def zip(self, *itrs, lazy = False, at = None):
    return self.zip(*itrs, lazy=lazy, at=at)

def flatten(self):
    return self.flatten()

def extend(self, value, *values):
    return self.extend(value, *values)

def pretend(self, value, *values):
    return self.pretend(value, *values)

def any(self, f = None, star = False):
    return self.any(f=f, star=star)

def anystar(self, f):
    return self.anystar(f)

def all(self, f = None, star = False):
    return self.all(f=f, star=star)

def allstar(self, f):
    return self.allstar(f)

def assert_all(self, f, f_error = None):
    return self.assert_all(f, f_error=f_error)

def assert_any(self, f, f_error = None):
    return self.assert_any(f, f_error=f_error)

def filter_eq(self, v, f = None, eq = None, lazy = False):
    return self.filter_eq(v, f=f, eq=eq, lazy=lazy)

def filter(self, f, eq = None, lazy = False, **kws):
    return self.filter(f, eq=eq, lazy=lazy, **kws)

def filterstar(self, f, eq = None, lazy = False, **kws):
    return self.filterstar(f, eq=eq, lazy=lazy, **kws)

def is_none(self):
    return self.is_none()

def not_none(self):
    return self.not_none()

def i_min(self, key = None):
    return self.i_min(key=key)

def i_max(self, key = None):
    return self.i_max(key=key)

def map(
    self,
    f,
    *iterables,
    at = None,
    lazy = False,
    **kwargs,
) -> iTuple:
    return self.map(f, *iterables, at=at, lazy=lazy, **kwargs)

# TODO: args, kwargs
def mapstar(self, f):
    return self.mapstar(f)

def enumerate(self):
    return self.enumerate()

def chunkby(
    self, 
    f, 
    lazy = False, 
    keys = False,
    pipe= None,
):
    return self.chunkby(f, lazy=lazy, keys=keys, pipe=pipe)

def groupby(
    self, f, lazy = False, keys = False, pipe = None
):
    return self.groupby(f, lazy=lazy, keys=keys, pipe=pipe)

def first(self):
    return self.first()

def last(self):
    return self.last()

def insert(self, i, v):
    return self.insert(i, v)

def instend(self, i, v):
    return self.instend(i, v)

def pop_first(self):
    return self.pop_first()

def pop_last(self):
    return self.pop_last()

def pop(self, i):
    return self.pop(i)

def first_where(self, f, default = None, star = False):
    return self.first_where(f, default=default, star=star)

def last_where(self, f, default = None, star = False):
    return self.last_where(f, default=default, star=star)

def n_from(gen, n, cls = iTuple):
    return cls.n_from(gen, n)

def from_while(
    gen, 
    f, 
    n = None, 
    star = False,
    iters=None,
    value=True,
    cls=iTuple,
):
    return cls.from_while(
        gen,
        f,
        n=n,
        star=star,
        iters=iters,
        value=value,
    )

def from_where(
    gen, 
    f, 
    n = None, 
    star = False,
    iters=None,
    value=True,
    cls=iTuple,
):
    return cls.from_where(
        gen,
        f,
        n=n,
        star=star,
        iters=iters,
        value=value,
    )

def clear(self):
    return self.clear()

def take(self, n):
    return self.take(n)

def tail(self, n):
    return self.tail(n)

def reverse(self, lazy = False):
    return self.reverse(lazy=lazy)

def take_while(self, f, n = None, lazy = False):
    return self.take_while(f, n=n, lazy=lazy)

def tail_while(self, f, n = None):
    return self.tail_while(f, n=n)

# NOTE: from as in, starting from first true
# versus above, which is until first false
def take_after(self, f, n = None, lazy = False):
    return self.take_after(f, n=n, lazy=lazy)

def tail_after(self, f, n = None):
    return self.tail_after(f, n=n)

def islice(self, left = None, right = None):
    return self.islice(left=left, right=right)

def unique(self):
    return self.unique()

def argsort(self, f = lambda v: v, star = False, reverse = False):
    return self.argsort(f=f, star=star, reverse=reverse)

def sort(self, f = lambda v: v, reverse = False, star = False):
    return self.sort(f=f, reverse=reverse, star=star)

# TODO: sortby etc.

def sortstar(self, f = lambda v: v, reverse = False):
    return self.sortstar(f=f, reverse=reverse)

# NOTE: ie. for sorting back after some other transformation
def sort_with_indices(
    self, f = lambda v: v, reverse=False
):
    return self.sort_with_indices(f=f, reverse=reverse)

def sortstar_with_indices(
    self, f = lambda v: v, reverse=False
):
    return self.sortstar_with_indices(f=f, reverse=reverse)

def accumulate(self, f, initial = None, lazy = False):
    return self.accumulate(f, initial=initial, lazy=lazy)

def foldcum(self, *args, initial=None, **kwargs):
    return self.foldcum(*args, initial=initial, **kwargs)

def fold(self, f, initial=None):
    return self.fold(f,initial =initial )

def foldstar(self, f, initial=None):
    return self.foldstar(f,initial =initial )

# -----

def product(self):
    return self.product()

def product_with(self, *iters):
    return self.product_with()

# TODO: other combinatorics

# TODO: auto gen the tests by inspecting methods on ituple?
# to ensure all have de-bound versions

# ---------------------------------------------------------------
