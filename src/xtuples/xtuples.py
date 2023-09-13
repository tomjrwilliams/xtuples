
from __future__ import annotations

# ---------------------------------------------------------------

import abc
import typing
import dataclasses
import collections

import operator
import itertools
import functools

# ---------------------------------------------------------------

# NOTE: at worst, not worse, than an un-optimised canonical solution

# where I cribbed from the itertools recipes (and other python docs), all credit to the original authors.

# where i didn't, i probably should have.

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


# TODO: some kind of validation placeholder?
# called in init, eg. quarter in [1 .. 4]

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
        return fDict(obj.__annotations__)

    @classmethod
    def as_dict(cls, obj):
        """
        >>> ex = _Example(1, "a")
        >>> ex.pipe(ex.cls.as_dict)
        {'x': 1, 's': 'a', 'it': iTuple()}
        """
        return fDict(obj._asdict())

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

class fDict(collections.UserDict):
    __slots__ = ()

    data: dict

    def pipe(self, f, *args, at=None, **kwargs):
        """
        >>> fDict({0: 1}).pipe(lambda d: d.map_values(
        ...     lambda v: v + 1
        ... ))
        {0: 2}
        """
        res = pipe(f, self, *args, at = at, **kwargs)
        if isinstance(res, dict):
            return fDict(res)
        return res

    def partial(self, f, *args, **kwargs):
        """
        >>> f = fDict({0: 1}).partial(
        ...     lambda d, n: d.map_values(lambda v: v + n)
        ... )
        >>> f(1)
        {0: 2}
        >>> f(2)
        {0: 3}
        """
        return functools.partial(f, self, *args, **kwargs)

    def keys_tuple(self):
        """
        >>> fDict({0: 1}).keys_tuple()
        iTuple(0)
        """
        return iTuple.from_keys(self)

    def values_tuple(self):
        """
        >>> fDict({0: 1}).values_tuple()
        iTuple(1)
        """
        return iTuple.from_values(self)
    
    def items_tuple(self):
        """
        >>> fDict({0: 1}).items_tuple()
        iTuple((0, 1))
        """
        return iTuple.from_items(self)

    # NOTE: we have separate map implementations 
    # as they are constant size, dict to dict
    # other iterator functions should use iTuple (from the above)

    def map_keys(self, f, *args, **kwargs):
        """
        >>> fDict({0: 1}).map_keys(lambda v: v + 1)
        {1: 1}
        """
        return fDict(dict(
            (f(k, *args, **kwargs), v) for k, v in self.items()
        ))

    def map_values(self, f, *args, **kwargs):
        """
        >>> fDict({0: 1}).map_values(lambda v: v + 1)
        {0: 2}
        """
        return fDict(dict(
            (k, f(v, *args, **kwargs)) for k, v in self.items()
        ))

    def map_items(self, f, *args, **kwargs):
        """
        >>> fDict({0: 1}).map_items(lambda k, v: (v, k))
        {1: 0}
        """
        return fDict(dict(
            f(k, v, *args, **kwargs) for k, v in self.items()
        ))

    def invert(self):
        """
        >>> fDict({0: 1}).invert()
        {1: 0}
        """
        return fDict(dict((v, k) for k, v in self.items()))

# ---------------------------------------------------------------

tuple_getitem = tuple.__getitem__

class iTuple(tuple):

    # -----

    @staticmethod
    def __new__(cls, *args):
        if len(args) == 1:
            v = args[0]
            if isinstance(v, cls):
                return v
            elif not isinstance(v, collections.abc.Iterable):
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
    def one(cls, v):
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
        return type(self).range(self.len())

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
            res = zip(*self)
        elif at is None:
            res = zip(self, *itrs)
        elif isinstance(at, int):
            res = zip(*itrs[:at], self, *itrs[at:])
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
            return any(self)
        elif star:
            return any(self.map(lambda v: f(*v), lazy=True))
        return any(self.map(f, lazy=True))

    def anystar(self, f):
        return any(self.mapstar(f, lazy=True))
    
    def all(self, f = None, star = False):
        if f is None:
            return all(self)
        elif star:
            return all(self.map(lambda v: f(*v), lazy=True))
        return all(self.map(f, lazy=True))

    def allstar(self, f):
        return all(self.mapstar(f))

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
            res = filter(lambda x: x == v, self)
        elif f is not None:
            res = filter(lambda x: f(x) == v, self)
        elif eq is not None:
            res = filter(lambda x: eq(x, v), self)
        elif f is not None and eq is not None:
            res = filter(lambda x: eq(f(x), v), self)
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

    def i_min(self, key = None):
        if key is None:
            return min(self.enumerate(), key = lambda i, v: v)
        return min(self.enumerate(), key = lambda i, v: key(v))

    def i_max(self, key = None):
        if key is None:
            return max(self.enumerate(), key = lambda i, v: v)
        return min(self.enumerate(), key = lambda i, v: key(v))

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
            return iTuple(map(f, self, *iterables))
        elif isinstance(at, int):
            return iTuple(map(
                f, *iterables[:at], self, *iterables[at:]
            ))
        elif isinstance(at, str):
            return iTuple(map(
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

    def enumerate(self):
        """
        >>> iTuple.range(3).enumerate()
        iTuple((0, 0), (1, 1), (2, 2))
        """
        # TODO: allow lazy
        return iTuple(enumerate(self))

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
        ...    lambda x: x < 2, keys=True, pipe=fDict
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
        ...    lambda x: x < 2, keys=True, pipe=fDict
        ... )
        {False: iTuple(2), True: iTuple(0, 1)}
        """
        res = (
            self.chunkby(f, keys=True)
            .sort(f = lambda kg: kg[0])
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
            for i, v in enumerate(gen):
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
            for i, v in enumerate(gen):
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
    
    def sort(self, f = lambda v: v, reverse = False, star = False):
        """
        >>> iTuple.range(3).reverse().sort()
        iTuple(0, 1, 2)
        >>> iTuple.range(3).sort()
        iTuple(0, 1, 2)
        """
        if star:
            return self.sortstar(f=f, reverse=reverse)
        return type(self)(sorted(self, key = f, reverse=reverse))
    
    def sortstar(self, f = lambda v: v, reverse = False):
        """
        >>> iTuple.range(3).reverse().sort()
        iTuple(0, 1, 2)
        >>> iTuple.range(3).sort()
        iTuple(0, 1, 2)
        """
        return type(self)(sorted(self, key = lambda v: f(*v), reverse=reverse))

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

    # combinatorics

    # -----

ituple = iTuple

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

import contextlib

@nTuple.decorate()
class Flags(typing.NamedTuple):
    
    values: typing.Optional[dict] = {}

    def add(self, v):
        t = type(v)
        if t not in self.values:
            self.values[t] = iTuple()
        self.values[t] = self.values[t].append(v)

    def remove(self, t):
        assert t in self.values
        self.values[t] = self.values[t][:-1]

    @contextlib.contextmanager
    def context(self, *vs):
        vs = iTuple(vs)
        ts = vs.map(type)
        assert ts.unique().len() == ts.len(), ts
        try:
            vs.map(self.add)
            res = vs
            yield res if res.len() > 1 else res[0]
        finally:
            ts.map(self.remove)

    def set(self, *vs):
        vs = iTuple(vs)
        ts = vs.map(type)
        assert ts.unique().len() == ts.len(), ts
        vs.map(self.add)
        res = vs
        return res if res.len() > 1 else res[0]

    def get(self, *ts):
        ts = iTuple(ts)
        assert ts.unique().len() == ts.len(), ts
        res = ts.map(self.values.get).map(
            lambda v: v if v is None else v.last()
        )
        return res if res.len() > 1 else res[0]

# ---------------------------------------------------------------

__all__ = [
    "iTuple",
    "nTuple",
    "fDict",
    "Flags",
    # "_Example",
]

# ---------------------------------------------------------------
