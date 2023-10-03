
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

# ---------------------------------------------------------------

_all = all
_any = any
_zip = zip
_map = map
_filter = filter
_enumerate = enumerate

tuple_getitem = tuple.__getitem__

# ---------------------------------------------------------------

T = typing.TypeVar('T')

if TYPE_CHECKING:
    CT = typing.Union[
        SupportsDunderLT,
        SupportsDunderGT,
    ]
else:
    class SupportsDunderLT(typing.Protocol):
        def __lt__(self, __other: Any) -> bool: ...

    class SupportsDunderGT(typing.Protocol):
        def __gt__(self, __other: Any) -> bool: ...

    CT = typing.Union[
        SupportsDunderLT,
        SupportsDunderGT,
    ]

# ---------------------------------------------------------------

# NOTE: at worst, not worse, than an un-optimised canonical solution

# where I cribbed from the itertools recipes (and other python docs), _all credit to the original authors.

# where i didn't, i probably should have.

# ---------------------------------------------------------------


class nTuple(abc.ABC):

    @abc.abstractmethod
    def __abstract__(self):
        # NOTE: here to prevent initialise instances of this
        # but rather use the decorator and typing.NamedTuple
        return

    @staticmethod
    def pipe(obj, f, *args, at = None, **kwargs):
        """
        >>> _Example = _Example(1, "a")
        >>> _Example.pipe(lambda a, b: a, None)
        _Example(x=1, s='a', it=iTuple())
        >>> _Example.pipe(lambda a, b: a, None, at = 1)
        >>> _Example.pipe(lambda a, b: a, None, at = 'b')
        >>> _Example.pipe(lambda a, b: a, a=None, at = 'b')
        >>> _Example.pipe(lambda a, b: a, b=None, at = 'a')
        _Example(x=1, s='a', it=iTuple())
        >>> _Example.pipe(lambda a, b: a, None, at = 0)
        _Example(x=1, s='a', it=iTuple())
        """
        return pipe(f, obj, *args, at = at, **kwargs)

    @staticmethod
    def partial(obj, f, *args, **kwargs):
        return functools.partial(f, obj, *args, **kwargs)

    @classmethod
    def is_subclass(cls, t):
        """
        >>> nTuple.is_subclass(tuple)
        False
        >>> nTuple.is_subclass(_Example(1, "a"))
        False
        >>> nTuple.is_subclass(_Example)
        True
        """
        try:
            is_sub = issubclass(t, tuple)
        except:
            is_sub = False
        return (
            is_sub and
            hasattr(t, "cls") and
            hasattr(t, "pipe") and
            hasattr(t, "partial")
        )

    @classmethod
    def is_instance(cls, obj):
        """
        >>> nTuple.is_instance(tuple)
        False
        >>> nTuple.is_instance(_Example)
        False
        >>> nTuple.is_instance(_Example(1, "a"))
        True
        """
        return (
            cls.is_subclass(type(obj)) and
            hasattr(obj, '_asdict') and
            hasattr(obj, '_fields')
        )


    @staticmethod
    def annotations(obj):
        """
        >>> ex = _Example(1, "a")
        >>> ex.pipe(ex.cls.annotations)
        {'x': ForwardRef('int'), 's': ForwardRef('str'), 'it': ForwardRef('iTuple')}
        """
        return obj.__annotations__

    @classmethod
    def as_dict(cls, obj):
        """
        >>> ex = _Example(1, "a")
        >>> ex.pipe(ex.cls.as_dict)
        {'x': 1, 's': 'a', 'it': iTuple()}
        """
        return obj._asdict()

    @classmethod
    def decorate(meta, **methods):
        def decorator(cls):
            cls.pipe = meta.pipe
            cls.partial = meta.partial
            cls.cls = meta
            for k, f in methods.items():
                setattr(cls, k, f)
            return cls
        return decorator

    @classmethod
    def enum(meta, cls):
        cls = meta.decorate()(cls)
        return functools.lru_cache(maxsize=1)(cls)

# ---------------------------------------------------------------

class iTuple(tuple, typing.Generic[T]):

    # -----

    @staticmethod
    def __new__(cls, *args):
        if len(args) == 1:
            v = args[0]
            if isinstance(v, cls):
                return v
            elif not isinstance(v, collections.Iterable):
                return ().__new__(cls, (v,))
        return super().__new__(cls, *args)

    def __repr__(self):
        """
        >>> iTuple()
        iTuple()
        >>> iTuple(iTuple((3, 2,)))
        iTuple(3, 2)
        """
        s = tuple.__repr__(self)
        return "{}({})".format(
            type(self).__name__,
            s[1:-2 if s[-2] == "," else -1],
        )

    def __str__(self):
        return self.__repr__()

    # -----

    @classmethod
    def empty(cls):
        return cls(tuple())

    @classmethod
    def one(cls: typing.Type[I], v: T) -> iTuple[T]:
        return cls((v,))

    @classmethod
    def none(cls, n):
        """
        >>> iTuple.none(3)
        iTuple(None, None, None)
        """
        return cls((None for _ in range(n)))

    @classmethod
    def range(cls, *args, **kwargs):
        """
        >>> iTuple.range(3)
        iTuple(0, 1, 2)
        """
        return cls(range(*args, **kwargs))

    @classmethod
    def from_keys(cls, d):
        """
        >>> iTuple.from_keys({i: i + 1 for i in range(2)})
        iTuple(0, 1)
        """
        return cls(d.keys())
        
    @classmethod
    def from_values(cls, d):
        """
        >>> iTuple.from_values({i: i + 1 for i in range(2)})
        iTuple(1, 2)
        """
        return cls(d.values())
        
    @classmethod
    def from_items(cls, d):
        """
        >>> iTuple.from_items({i: i + 1 for i in range(2)})
        iTuple((0, 1), (1, 2))
        """
        return cls(d.items())
    
    @classmethod
    def from_index(cls, s):
        return cls(s.index)

    @classmethod
    def from_index_values(cls, s):
        return cls(s.index.values)

    @classmethod
    def from_columns(cls, s):
        return cls(s.columns)

    # -----

    def pipe(self, f, *args, at = None, **kwargs):
        """
        >>> iTuple.range(2).pipe(lambda it: it)
        iTuple(0, 1)
        >>> iTuple.range(2).pipe(
        ...     lambda it, v: it.map(lambda x: x * v), 2
        ... )
        iTuple(0, 2)
        """
        return pipe(f, self, *args, at = at, **kwargs)

    def partial(self, f, *args, **kwargs):
        """
        >>> f = iTuple.range(2).partial(
        ...     lambda it, v: it.map(lambda x: x * v)
        ... )
        >>> f(2)
        iTuple(0, 2)
        >>> f(3)
        iTuple(0, 3)
        """
        return functools.partial(f, self, *args, **kwargs)

    # -----

    # def __len__(self):
    #     return len(self)

    # def __contains__(self, v):
    #     return v in self

    def index_of(self, v):
        return self.index(v)

    def len(self):
        """
        >>> iTuple.range(3).len()
        3
        """
        return len(self)

    def len_range(self):
        """
        >>> iTuple.range(3).len()
        3
        """
        return iTuple.range(self.len())

    def append(self, value, *values):
        """
        >>> iTuple().append(1)
        iTuple(1)
        >>> iTuple.range(1).append(1)
        iTuple(0, 1)
        >>> iTuple.range(1).append(1, 2)
        iTuple(0, 1, 2)
        >>> iTuple.range(1).append(1, 2, 3)
        iTuple(0, 1, 2, 3)
        >>> iTuple.range(1).append(1, (2,))
        iTuple(0, 1, (2,))
        """
        return iTuple((*self, value, *values,))

    def prepend(self, value, *values):
        """
        >>> iTuple().prepend(1)
        iTuple(1)
        >>> iTuple.range(1).prepend(1)
        iTuple(1, 0)
        >>> iTuple.range(1).prepend(1, 2)
        iTuple(1, 2, 0)
        >>> iTuple.range(1).prepend(1, 2, 3)
        iTuple(1, 2, 3, 0)
        >>> iTuple.range(1).prepend(1, (2,))
        iTuple(1, (2,), 0)
        """
        return iTuple((value, *values, *self,))

    def zip(self, *itrs, lazy = False, at = None):
        """
        >>> iTuple([[1, 1], [2, 2], [3, 3]]).zip()
        iTuple((1, 2, 3), (1, 2, 3))
        >>> iTuple([iTuple.range(3), iTuple.range(1, 4)]).zip()
        iTuple((0, 1), (1, 2), (2, 3))
        >>> iTuple.range(3).zip(iTuple.range(1, 4))
        iTuple((0, 1), (1, 2), (2, 3))
        """
        if len(itrs) == 0:
            res = _zip(*self)
        elif at is None:
            res = _zip(self, *itrs)
        elif isinstance(at, int):
            res = _zip(*itrs[:at], self, *itrs[at:])
        else:
            assert False, at
        return res if lazy else iTuple(res)

    def flatten(self):
        """
        >>> iTuple.range(3).map(lambda x: [x]).flatten()
        iTuple(0, 1, 2)
        """
        return iTuple(itertools.chain(*self))

    def extend(self, value, *values):
        """
        >>> iTuple.range(1).extend((1,))
        iTuple(0, 1)
        >>> iTuple.range(1).extend([1])
        iTuple(0, 1)
        >>> iTuple.range(1).extend([1], [2])
        iTuple(0, 1, 2)
        >>> iTuple.range(1).extend([1], [[2]])
        iTuple(0, 1, [2])
        >>> iTuple.range(1).extend([1], [[2]], [2])
        iTuple(0, 1, [2], 2)
        """
        return iTuple(itertools.chain.from_iterable(
            (self, value, *values)
        ))

    def pretend(self, value, *values):
        """
        >>> iTuple.range(1).pretend((1,))
        iTuple(1, 0)
        >>> iTuple.range(1).pretend([1])
        iTuple(1, 0)
        >>> iTuple.range(1).pretend([1], [2])
        iTuple(1, 2, 0)
        >>> iTuple.range(1).pretend([1], [[2]])
        iTuple(1, [2], 0)
        >>> iTuple.range(1).pretend([1], [[2]], [2])
        iTuple(1, [2], 2, 0)
        """
        return iTuple(itertools.chain.from_iterable(
            (value, *values, self)
        ))

    def any(self, f = None, star = False):
        if f is None:
            return _any(self)
        elif star:
            return _any(self.map(lambda v: f(*v), lazy=True))
        return _any(self.map(f, lazy=True))

    def anystar(self, f):
        return _any(self.mapstar(f))
    
    def all(self, f = None, star = False):
        if f is None:
            return _all(self)
        elif star:
            return _all(self.map(lambda v: f(*v), lazy=True))
        return _all(self.map(f, lazy=True))

    def allstar(self, f):
        return _all(self.mapstar(f))

    def assert_all(self, f, f_error = None):
        if f_error:
            assert self.all(f), f_error(self)
        else:
            assert self.all(f)
        return self

    def assert_any(self, f, f_error = None):
        if f_error:
            assert self.any(f), f_error(self)
        else:
            assert self.any(f)
        return self

    def filter_eq(self, v, f = None, eq = None, lazy = False):
        """
        >>> iTuple.range(3).filter_eq(1)
        iTuple(1)
        """
        if f is None and eq is None:
            res = _filter(lambda x: x == v, self)
        elif f is not None:
            res = _filter(lambda x: f(x) == v, self)
        elif eq is not None:
            res = _filter(lambda x: eq(x, v), self)
        elif f is not None and eq is not None:
            res = _filter(lambda x: eq(f(x), v), self)
        else:
            assert False
        return res if lazy else type(self)(res)

    def filter(self, f, eq = None, lazy = False, **kws):
        """
        >>> iTuple.range(3).filter(lambda x: x > 1)
        iTuple(2)
        """
        # res = []
        # for v in self.iter():
        #     if f(v):
        #         res.append(v)
        return type(self)((
            v for v in self.iter() if f(v, **kws)
        ))
        # return self.filter_eq(True, f = f, eq = eq, lazy = lazy)

    def filterstar(self, f, eq = None, lazy = False, **kws):
        """
        >>> iTuple.range(3).filter(lambda x: x > 1)
        iTuple(2)
        """
        # res = []
        # for v in self.iter():
        #     if f(v):
        #         res.append(v)
        return type(self)((
            v for v in self.iter() if f(*v, **kws)
        ))

    def is_none(self):
        return self.filter(lambda v: v is None)

    def not_none(self):
        return self.filter(lambda v: v is not None)

    def i_min(
        self: iTuple[T], 
        f: typing.Optional[typing.Callable[[T], CT]] = None
    ) -> int:
        if f is not None:
            key = lambda _, v: f(v)
        else:
            key = lambda _, v: v
        return self.enumerate().sortby(key).first()[0]

    def i_max(
        self: iTuple[T], 
        f: typing.Optional[typing.Callable[[T], CT]] = None
    ) -> int:
        if f is not None:
            key = lambda _, v: f(v)
        else:
            key = lambda _, v: v
        return self.enumerate().sortby(key).last()[0]

    def map(
        self,
        f,
        *iterables,
        at = None,
        lazy = False,
        **kwargs,
    ) -> iTuple:
        """
        >>> iTuple.range(3).map(lambda x: x * 2)
        iTuple(0, 2, 4)
        """
        if len(kwargs):
            f = functools.partial(f, **kwargs)
        # if lazy and at is None:
        #     return map(f, self, *iterables)
        if at is None:
            return iTuple(_map(f, self, *iterables))
        elif isinstance(at, int):
            return iTuple(_map(
                f, *iterables[:at], self, *iterables[at:]
            ))
        elif isinstance(at, str):
            return iTuple(_map(
                f, *iterables, **{at: self}
            ))
        else:
            assert False, at

    # args, kwargs
    def mapstar(self, f):
        return iTuple(itertools.starmap(f, self))

    def get(self, i):
        if isinstance(i, slice):
            return type(self)(self[i])
        return self[i]

    def __getitem__(self, i):
        if isinstance(i, slice):
            return type(self)(tuple_getitem(self, i))
        return tuple_getitem(self, i)

    def __add__(self, v):
        if isinstance(v, typing.Iterable):
            return self.extend(v)
        assert False, type(v)

    # def __iter__(self):
    #     return iter(self)

    def iter(self):
        """
        >>> for x in iTuple.range(3).iter(): print(x)
        0
        1
        2
        """
        return iter(self)

    def enumerate(self: iTuple[T]) -> iTuple[tuple[int, T]]:
        """
        >>> iTuple.range(3).enumerate()
        iTuple((0, 0), (1, 1), (2, 2))
        """
        # TODO: _allow lazy
        return iTuple(_enumerate(self))

    def chunkby(
        self, 
        f, 
        lazy = False, 
        keys = False,
        pipe= None,
    ):
        """
        >>> iTuple.range(3).chunkby(lambda x: x < 2)
        iTuple(iTuple(0, 1), iTuple(2))
        >>> iTuple.range(3).chunkby(
        ...    lambda x: x < 2, keys=True, pipe=dict
        ... )
        {True: iTuple(0, 1), False: iTuple(2)}
        """
        # TODO: lazy no keys
        res = itertools.groupby(self, key=f)
        if lazy and keys and pipe is None:
            return res
        if pipe is None:
            pipe = iTuple
        if keys:
            return pipe((k, iTuple(g),) for k, g in res)
        else:
            return pipe(iTuple(g) for k, g in res)

    def groupby(
        self, f, lazy = False, keys = False, pipe = None
    ):
        """
        >>> iTuple.range(3).groupby(lambda x: x < 2)
        iTuple(iTuple(2), iTuple(0, 1))
        >>> iTuple.range(3).groupby(
        ...    lambda x: x < 2, keys=True, pipe=dict
        ... )
        {False: iTuple(2), True: iTuple(0, 1)}
        """
        res = (
            self.chunkby(f, keys=True)
            .sortby(lambda kg: kg[0])
            .chunkby(lambda kg: kg[0], keys = True)
            .map(lambda k_kgs: (
                k_kgs[0],
                k_kgs[1].mapstar(lambda k, g: g).flatten(),
            ))
        )
        if pipe is None:
            pipe = iTuple
        if keys:
            return pipe((k, iTuple(g),) for k, g in res)
        else:
            return pipe(iTuple(g) for k, g in res)

    def first(self):
        """
        >>> iTuple.range(3).first()
        0
        """
        return self[0]
    
    def last(self):
        """
        >>> iTuple.range(3).last()
        2
        """
        return self[-1]

    def insert(self, i, v):
        """
        >>> iTuple.range(3).insert(2, 4)
        iTuple(0, 1, 4, 2)
        """
        return self[:i].append(v).extend(self[i:])

    def instend(self, i, v):
        return self[:i].extend(v).extend(self[i:])

    def pop_first(self):
        return self[1:]

    def pop_last(self):
        return self[:-1]

    def pop(self, i):
        return self[:i] + self[i + 1:]

    def first_where(self, f, default = None, star = False):
        """
        >>> iTuple.range(3).first_where(lambda v: v > 0)
        1
        """
        if star:
            for v in self:
                if f(*v):
                    return v
        else:
            for v in self:
                if f(v):
                    return v
        return default

    def last_where(self, f, default = None, star = False):
        """
        >>> iTuple.range(3).last_where(lambda v: v < 2)
        1
        """
        if star:
            for v in reversed(self):
                if f(*v):
                    return v
        for v in reversed(self):
            if f(v):
                return v
        return default

    @classmethod
    def n_from(cls, gen, n):
        return cls.range(n).zip(gen).mapstar(
            lambda i, v: v
        )

    @classmethod
    def from_while(
        cls, 
        gen, 
        f, 
        n = None, 
        star = False,
        iters=None,
        value=True,
    ):
        def _gen():
            _n = 0
            for i, v in _enumerate(gen):
                if iters is not None and i == iters:
                    return
                if n == _n:
                    return
                if star:
                    if not f(*v) == value:
                        return
                else:
                    if not f(v) == value:
                        return
                yield v
                _n += 1
        return cls(_gen())

    @classmethod
    def from_where(
        cls, 
        gen, 
        f, 
        n = None, 
        star = False,
        iters=None,
        value=True,
    ):
        def _gen():
            _n = 0
            for i, v in _enumerate(gen):
                if iters is not None and i == iters:
                    return
                if n == _n:
                    return
                if star:
                    if not f(*v) == value:
                        continue
                else:
                    if not f(v) == value:
                        continue
                yield v
                _n += 1
        return cls(_gen())

    def clear(self):
        return type(self)()

    def take(self, n):
        """
        >>> iTuple.range(3).take(2)
        iTuple(0, 1)
        """
        return self[:n]

    def tail(self, n):
        """
        >>> iTuple.range(3).tail(2)
        iTuple(1, 2)
        """
        return self[-n:]

    def reverse(self, lazy = False):
        """
        >>> iTuple.range(3).reverse()
        iTuple(2, 1, 0)
        """
        if lazy:
            return reversed(self)
        return type(self)(reversed(self))

    def take_while(self, f, n = None, lazy = False):
        """
        >>> iTuple.range(3).take_while(lambda v: v < 1)
        iTuple(0)
        """
        def iter():
            i = 0
            for v in self:
                if f(v) and (n is None or i < n):
                    yield v
                    i += 1
                else:
                    return
        res = iter()
        return res if lazy else type(self)(res)

    def tail_while(self, f, n = None):
        """
        >>> iTuple.range(3).tail_while(lambda v: v > 1)
        iTuple(2)
        """
        i = 0
        for v in reversed(self):
            if f(v) and (n is None or i < n):
                i += 1
            else:
                break
        return self.tail(i)

    # NOTE: from as in, starting from first true
    # versus above, which is until first false
    def take_after(self, f, n = None, lazy = False):
        """
        >>> iTuple.range(3).take_after(lambda v: v < 1)
        iTuple(1, 2)
        >>> iTuple.range(3).take_after(lambda v: v < 1, n = 1)
        iTuple(1)
        """
        def iter():
            i = 0
            for v in self:
                if f(v):
                    pass
                elif n is None or i < n:
                    yield v
                    i += 1
                else:
                    return
        res = iter()
        return res if lazy else type(self)(res)

    def tail_after(self, f, n = None):
        """
        >>> iTuple.range(3).tail_after(lambda v: v < 2)
        iTuple(0, 1)
        >>> iTuple.range(3).tail_after(lambda v: v < 2, 1)
        iTuple(1)
        """
        l = 0
        r = 0
        for v in reversed(self):
            if not f(v):
                l += 1
            elif n is None or r < n:
                r += 1
            else:
                break
        return self.tail(l + r).take(r)

    def islice(self, left = None, right = None):
        """
        >>> iTuple.range(5).islice(1, 3)
        iTuple(1, 2)
        """
        return self[left:right]

    def unique(self):
        """
        >>> iTuple([1, 1, 3, 2, 4, 2, 3]).unique()
        iTuple(1, 3, 2, 4)
        """
        def iter():
            seen: typing.Set = set()
            seen_add = seen.add
            seen_contains = seen.__contains__
            for v in itertools.filterfalse(seen_contains, self):
                seen_add(v)
                yield v
        return type(self)(iter())

    def argsort(self, f = lambda v: v, star = False, reverse = False):
        if star:
            f_sort = lambda i, v: f(*v)
        else:
            f_sort = lambda i, v: f(v)
        return self.enumerate().sortstar(
            f=f_sort, reverse=reverse,
        ).mapstar(lambda i, v: i)

    def sort(self: iTuple[CT], reverse=False) -> iTuple[CT]:
        return type(self)(sorted(self, reverse=reverse))
    
    def sortby(
        self: iTuple[T], 
        f: typing.Union[
            typing.Callable[[T], CT],
            typing.Callable[..., CT],
        ],
        reverse = False,
        star = False,
    ):
        """
        >>> iTuple.range(3).reverse().sort()
        iTuple(0, 1, 2)
        >>> iTuple.range(3).sort()
        iTuple(0, 1, 2)
        """
        if star:
            return self.sortstar(f=f, reverse=reverse)
        return type(self)(
            sorted(self, key = f, reverse=reverse)
            #
        )
    
    def sortstar(
        self: iTuple[T], 
        f: typing.Callable[..., CT],
        reverse = False
    ):
        """
        >>> iTuple.range(3).reverse().sort()
        iTuple(0, 1, 2)
        >>> iTuple.range(3).sort()
        iTuple(0, 1, 2)
        """
        return type(self)(
            sorted(self, key = lambda v: f(*v), reverse=reverse)
            #
        )

    # NOTE: ie. for sorting back after some other transformation
    def sort_with_indices(
        self, f = lambda v: v, reverse=False
    ):
        return self.enumerate().sortstar(
            lambda i, v: f(v), reverse=reverse
        )

    def sortstar_with_indices(
        self, f = lambda v: v, reverse=False
    ):
        return self.enumerate().sortstar(
            lambda i, v: f(*v), reverse=reverse
        )

    def accumulate(self, f, initial = None, lazy = False):
        """
        >>> iTuple.range(3).accumulate(lambda acc, v: v)
        iTuple(0, 1, 2)
        >>> iTuple.range(3).accumulate(lambda acc, v: v, initial=0)
        iTuple(0, 0, 1, 2)
        >>> iTuple.range(3).accumulate(operator.add)
        iTuple(0, 1, 3)
        """
        if lazy:
            return itertools.accumulate(
                self, func=f, initial=initial
                #
            )
        return iTuple(itertools.accumulate(
            self, func=f, initial=initial
        ))

    def foldcum(self, *args, initial=None, **kwargs):
        """
        >>> iTuple.range(3).foldcum(lambda acc, v: v)
        iTuple(0, 1, 2)
        >>> iTuple.range(3).foldcum(operator.add)
        iTuple(0, 1, 3)
        """
        # res = []
        # acc = initial
        # for x in self.iter():
        #     acc = f(acc, x)
        #     res.append(acc)
        # return iTuple(tuple(res))
        return self.accumulate(*args, **kwargs)

    def fold(self, f, initial=None):
        """
        >>> iTuple.range(3).fold(lambda acc, v: v)
        2
        >>> iTuple.range(3).fold(lambda acc, v: v, initial=0)
        2
        >>> iTuple.range(3).fold(operator.add)
        3
        """
        # acc = initial
        # for v in self.iter():
        #     acc = f(acc, v)
        # return iTuple(tuple(acc))
        if initial is not None:
            return functools.reduce(f, self, initial)
        else:
            return functools.reduce(f, self)

    def foldstar(self, f, initial=None):
        """
        >>> iTuple.range(3).fold(lambda acc, v: v)
        2
        >>> iTuple.range(3).fold(lambda acc, v: v, initial=0)
        2
        >>> iTuple.range(3).fold(operator.add)
        3
        """
        # acc = initial
        # for v in self.iter():
        #     acc = f(acc, v)
        # return iTuple(tuple(acc))
        fstar = lambda acc, v: f(acc, *v)
        if initial is not None:
            return functools.reduce(fstar, self, initial)
        else:
            return functools.reduce(fstar, self)

    # -----

    def product(self):
        return iTuple(itertools.product(*self))

    def product_with(self, *iters):
        return iTuple(itertools.product(self, *iters))

    # combinatorics

    # -----

ituple = iTuple

# ---------------------------------------------------------------

def pipe(f, obj, *args, at = None, discard=False, **kwargs):
    if at is None:
        res = f(obj, *args, **kwargs)
    elif isinstance(at, int):
        res = f(*args[:at], obj, *args[at:], **kwargs)
    elif isinstance(at, str):
        res = f(*args, **{at: obj}, **kwargs)
    else:
        assert False, at
    if not discard:
        return res
    return obj

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


@nTuple.decorate()
class _Example(typing.NamedTuple):
    """
    >>> ex = _Example(1, "a")
    >>> ex
    _Example(x=1, s='a', it=iTuple())
    >>> ex.cls
    <class 'xtuples.xtuples.nTuple'>
    >>> ex.pipe(lambda nt: nt.x)
    1
    >>> f = ex.partial(lambda nt, v: nt.x * v)
    >>> f(2)
    2
    >>> f(3)
    3
    """
    # NOTE: cls, pipe, partial are mandatory boilerplate

    x: int
    s: str
    it: iTuple = iTuple([])

    @property
    def cls(self):
        ...

    def pipe(self, f, *args, at = None, **kwargs):
        ...

    def partial(self, f, *args, at = None, **kwargs):
        ...

# ---------------------------------------------------------------

I = typing.TypeVar('I', bound=iTuple)

__all__ = [
    "iTuple",
    "nTuple",
    "map",
]

# ---------------------------------------------------------------
