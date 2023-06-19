
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

REGISTRY = {}

# ---------------------------------------------------------------

def pipe(f, obj, *args, at = None, **kwargs):
    if at is None:
        return f(obj, *args, **kwargs)
    elif isinstance(at, int):
        return f(*args[:at], obj, *args[at:], **kwargs)
    elif isinstance(at, str):
        return f(*args, **{at: obj}, **kwargs)
    else:
        assert False, at

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
        >>> example = Example(1, "a")
        >>> example.pipe(lambda a, b: a, None)
        Example(x=1, s='a', it=iTuple())
        >>> example.pipe(lambda a, b: a, None, at = 1)
        >>> example.pipe(lambda a, b: a, None, at = 'b')
        >>> example.pipe(lambda a, b: a, a=None, at = 'b')
        >>> example.pipe(lambda a, b: a, b=None, at = 'a')
        Example(x=1, s='a', it=iTuple())
        >>> example.pipe(lambda a, b: a, None, at = 0)
        Example(x=1, s='a', it=iTuple())
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
        >>> nTuple.is_subclass(Example(1, "a"))
        False
        >>> nTuple.is_subclass(Example)
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
        >>> nTuple.is_instance(Example)
        False
        >>> nTuple.is_instance(Example(1, "a"))
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
        >>> ex = Example(1, "a")
        >>> ex.pipe(ex.cls.annotations)
        {'x': <class 'int'>, 's': <class 'str'>, 'it': <class 'xtuples.xtuples.iTuple'>}
        """
        return fDict(obj.__annotations__)

    @classmethod
    def as_dict(cls, obj):
        """
        >>> ex = Example(1, "a")
        >>> ex.pipe(ex.cls.as_dict)
        {'x': 1, 's': 'a', 'it': iTuple()}
        """
        return fDict(obj._asdict())

    @classmethod
    def decorate(meta, cls):
        assert cls.__name__ not in REGISTRY
        cls.pipe = meta.pipe
        cls.partial = meta.partial
        cls.cls = meta
        REGISTRY[cls.__name__] = cls
        return cls

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
        return fDict(
            (f(k, *args, **kwargs), v) for k, v in self.items()
        )

    def map_values(self, f, *args, **kwargs):
        """
        >>> fDict({0: 1}).map_values(lambda v: v + 1)
        {0: 2}
        """
        return fDict(
            (k, f(v, *args, **kwargs)) for k, v in self.items()
        )

    def map_items(self, f, *args, **kwargs):
        """
        >>> fDict({0: 1}).map_items(lambda k, v: (v, k))
        {1: 0}
        """
        return fDict(
            f(k, v, *args, **kwargs) for k, v in self.items()
        )

    def invert(self):
        """
        >>> fDict({0: 1}).invert()
        {1: 0}
        """
        return fDict((v, k) for k, v in self.items())

# ---------------------------------------------------------------

@dataclasses.dataclass(init = False, repr=True)
class iTuple(collections.UserList, tuple): # type: ignore
    __slots__ = ()

    data: tuple # type: ignore

    # -----

    @staticmethod
    def __new__(cls, data = None):
        # NOTE: we use cls not array
        # so sub-classing *does* change identity
        if isinstance(data, cls):
            return data
        return super().__new__(cls, data=data)
    
    def __init__(self, data = None):
        # TODO: option for lazy init?
        self.data = (
            tuple() if data is None
            else data if isinstance(data, tuple)
            else tuple(data)
        )
        __getitem__ = self.data.__getitem__
        self.__getitem__ = __getitem__
        self.get = __getitem__

    def __repr__(self):
        s = super().__repr__()
        return "{}({})".format(
            type(self).__name__,
            s[1:-2 if s[-2] == "," else -1],
        )

    def __hash__(self):
        return hash(self.data)

    @classmethod
    def decorate(meta, cls):
        assert cls.__name__ not in REGISTRY
        REGISTRY[cls.__name__] = cls
        return cls

    # -----

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

    def len(self):
        """
        >>> iTuple.range(3).len()
        3
        """
        return len(self)

    def append(self, value, *values):
        """
        >>> iTuple.range(1).append(1)
        iTuple(0, 1)
        >>> iTuple.range(1).append(1, 2)
        iTuple(0, 1, 2)
        >>> iTuple.range(1).append(1, 2, 3)
        iTuple(0, 1, 2, 3)
        >>> iTuple.range(1).append(1, (2,))
        iTuple(0, 1, (2,))
        """
        return self + (value, *values)

    def prepend(self, value, *values):
        """
        >>> iTuple.range(1).prepend(1)
        iTuple(1, 0)
        >>> iTuple.range(1).prepend(1, 2)
        iTuple(1, 2, 0)
        >>> iTuple.range(1).prepend(1, 2, 3)
        iTuple(1, 2, 3, 0)
        >>> iTuple.range(1).prepend(1, (2,))
        iTuple(1, (2,), 0)
        """
        return (value, *values) + self

    def zip(self, *itrs, lazy = False):
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
        else:
            res = zip(self, *itrs)
        return res if lazy else iTuple(data=res)

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
        else:
            res = filter(lambda x: eq(f(x), v), self)
        return res if lazy else type(self)(data=res)

    def filter(self, f, eq = None, lazy = False):
        """
        >>> iTuple.range(3).filter(lambda x: x > 1)
        iTuple(2)
        """
        # res = []
        # for v in self.iter():
        #     if f(v):
        #         res.append(v)
        return iTuple(data=(
            v for v in self.iter() if f(v)
        ))
        # return self.filter_eq(True, f = f, eq = eq, lazy = lazy)

    def map(self, f, *iterables, lazy = False):
        """
        >>> iTuple.range(3).map(lambda x: x * 2)
        iTuple(0, 2, 4)
        """
        if lazy:
            return map(f, self.data, *iterables)
        return iTuple(data = map(f, self.data, *iterables))

    # def __getitem__(self, i):
    #     return self.data[i]

    def iter(self):
        """
        >>> for x in iTuple.range(3).iter(): print(x)
        0
        1
        2
        """
        return iter(self.data)

    def enumerate(self):
        """
        >>> iTuple.range(3).enumerate()
        iTuple((0, 0), (1, 1), (2, 2))
        """
        # TODO: allow lazy
        return iTuple(enumerate(self))

    def groupby(
        self, 
        f, 
        lazy = False, 
        keys = False,
        pipe= None,
    ):
        """
        >>> iTuple.range(3).groupby(lambda x: x < 2)
        iTuple((0, 1), (2,))
        >>> iTuple.range(3).groupby(
        ...    lambda x: x < 2, keys=True, pipe=fDict
        ... )
        {True: (0, 1), False: (2,)}
        """
        # TODO: lazy no keys
        res = itertools.groupby(self, key=f)
        if lazy and keys and pipe is None:
            return res
        if pipe is None:
            pipe = iTuple
        if keys:
            return pipe((k, tuple(g),) for k, g in res)
        else:
            return pipe(tuple(g) for k, g in res)

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

    def first_where(self, f):
        """
        >>> iTuple.range(3).first_where(lambda v: v > 0)
        1
        """
        for v in self:
            if f(v):
                return v
        return None

    def last_where(self, f):
        """
        >>> iTuple.range(3).last_where(lambda v: v < 2)
        1
        """
        for v in reversed(self):
            if f(v):
                return v
        return None

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
        return type(self)(data=reversed(self))

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
        return res if lazy else type(self)(data=res)

    def __iter__(self):
        return iter(self.data)

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
        return res if lazy else type(self)(data=res)

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
            seen = set()
            seen_add = seen.add
            seen_contains = seen.__contains__
            for v in itertools.filterfalse(seen_contains, self):
                seen_add(v)
                yield v
        return type(self)(data=iter())
    
    def sort(self, f = lambda v: v):
        """
        >>> iTuple.range(3).reverse().sort()
        iTuple(0, 1, 2)
        >>> iTuple.range(3).sort()
        iTuple(0, 1, 2)
        """
        return type(self)(data=sorted(self, key = f))

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
            return itertools.accumulate(self, func=f, initial=initial)
        return iTuple(data=itertools.accumulate(
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
        # return iTuple(data=tuple(res))
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
        # return iTuple(data=tuple(acc))
        if initial is not None:
            return functools.reduce(f, self, initial)
        else:
            return functools.reduce(f, self)

    # -----

    # combinatorics

    # -----

# ---------------------------------------------------------------

@nTuple.decorate
class _Example(typing.NamedTuple):
    """
    >>> ex = Example(1, "a")
    >>> ex
    Example(x=1, s='a', it=iTuple())
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

# TODO: context manager to control
# if we add the type information when writing to json or not

# TODO: context mananger to control
# lazy default behaviour (ie. default to lazy or not)

__all__ = [
    "iTuple",
    "nTuple",
    "fDict",
    # "_Example",
]

# ---------------------------------------------------------------
