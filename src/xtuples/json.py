
# ---------------------------------------------------------------

import json

from .xtuples import REGISTRY, nTuple, iTuple, _Example

# -----

def cast_json_nTuple(obj):
    """
    >>> ex = _Example(1, "a")
    >>> ex.pipe(cast_json_nTuple)
    {'x': 1, 's': 'a', 'it': {'__t__': 'iTuple', 'data': []}, '__t__': '_Example'}
    """
    d = {
        k: cast_json(v)
        for k, v in obj._asdict().items()
        #
    }
    d["__t__"] = type(obj).__name__
    return d

def uncast_json_nTuple(cls, obj):
    """
    >>> ex = _Example(1, "a")
    >>> uncast_json_nTuple(_Example, ex.pipe(cast_json_nTuple))
    _Example(x=1, s='a', it=iTuple())
    """
    assert obj["__t__"] == cls.__name__
    return cls(
        *(
            uncast_json(v)
            for k, v in obj.items() if k != "__t__"
        )
    )

def cast_json_iTuple(self):
    """
    >>> iTuple.range(1).pipe(cast_json_iTuple)
    {'__t__': 'iTuple', 'data': [0]}
    """
    return dict(
        __t__ = type(self).__name__,
        data = list(self.map(cast_json)),
    )

def uncast_json_iTuple(cls, obj):
    """
    >>> uncast_json_iTuple(iTuple, iTuple.range(1).pipe(cast_json_iTuple))
    iTuple(0)
    """
    assert obj["__t__"] == cls.__name__
    return cls(data=obj["data"])


def cast_json(obj, default = lambda obj: obj):
    """
    >>> ex = _Example(1, "a")
    >>> ex.pipe(cast_json)
    {'x': 1, 's': 'a', 'it': {'__t__': 'iTuple', 'data': []}, '__t__': '_Example'}
    >>> iTuple.range(1).pipe(cast_json)
    {'__t__': 'iTuple', 'data': [0]}
    """
    if nTuple.is_instance(obj):
        return cast_json_nTuple(obj)
    elif isinstance(obj, iTuple):
        return cast_json_iTuple(obj)
    return default(obj)

def uncast_json(obj):
    """
    >>> ex = _Example(1, "a")
    >>> uncast_json(ex.pipe(cast_json))
    _Example(x=1, s='a', it=iTuple())
    >>> uncast_json(iTuple.range(1).pipe(cast_json))
    iTuple(0)
    """
    if not isinstance(obj, dict):
        return obj
    __t__ = obj.get("__t__", None)
    if __t__ is None:
        return obj
    cls = iTuple if __t__ == "iTuple" else REGISTRY[__t__]
    if cls is iTuple or issubclass(cls, iTuple):
        return uncast_json_iTuple(cls, obj)
    return uncast_json_nTuple(cls, obj)

# ---------------------------------------------------------------

class JSONEncoder(json.JSONEncoder):

    def iterencode(self, o, *args, **kwargs):
        for chunk in super().iterencode(
            cast_json(o), *args, **kwargs
        ):
            yield chunk

    # def meta_default(self, obj):
    #     return json.JSONEncoder.default(self, obj)

    # def default(self, obj):
    #     if isinstance(obj, fDict):
    #         return self.meta_default(obj.data)
    #     return cast_json(obj, default=self.meta_default)

# -----

class JSONDecoder(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self,
            object_hook=self.object_hook,
            *args,
            **kwargs
            #
        )

    @classmethod
    def xtuple_object_hook(cls, d):
        return uncast_json(d)

    def object_hook(self, d):
        return self.xtuple_object_hook(d)

# ---------------------------------------------------------------

# TODO: fString so can do .pipe ?
def to_json(v, **kwargs):
    """
    >>> print(iTuple([_Example(1, "a")]).pipe(to_json, indent=2))
    {
      "__t__": "iTuple",
      "data": [
        {
          "x": 1,
          "s": "a",
          "it": {
            "__t__": "iTuple",
            "data": []
          },
          "__t__": "_Example"
        }
      ]
    }
    >>> print(iTuple([
    ...     iTuple([_Example(1, "a")])
    ... ]).pipe(to_json, indent=2))
    {
      "__t__": "iTuple",
      "data": [
        {
          "__t__": "iTuple",
          "data": [
            {
              "x": 1,
              "s": "a",
              "it": {
                "__t__": "iTuple",
                "data": []
              },
              "__t__": "_Example"
            }
          ]
        }
      ]
    }
    >>> print(_Example(2, "b", iTuple([
    ...     iTuple([_Example(1, "a")])
    ... ])).pipe(to_json, indent=2))
    {
      "x": 2,
      "s": "b",
      "it": {
        "__t__": "iTuple",
        "data": [
          {
            "__t__": "iTuple",
            "data": [
              {
                "x": 1,
                "s": "a",
                "it": {
                  "__t__": "iTuple",
                  "data": []
                },
                "__t__": "_Example"
              }
            ]
          }
        ]
      },
      "__t__": "_Example"
    }
    """
    return json.dumps(v, cls=JSONEncoder, **kwargs)

def from_json(v: str, **kwargs):
    """
    >>> ex = iTuple([_Example(1, "a")])
    >>> from_json(ex.pipe(to_json))
    iTuple(_Example(x=1, s='a', it=iTuple()))
    >>> from_json(
    ...     iTuple([iTuple([_Example(1, "a")])]).pipe(to_json)
    ... )
    iTuple(iTuple(_Example(x=1, s='a', it=iTuple())))
    >>> from_json(
    ...     _Example(2, "b", iTuple([
    ...         iTuple([_Example(1, "a")])
    ...     ])).pipe(to_json)
    ... )
    _Example(x=2, s='b', it=iTuple(iTuple(_Example(x=1, s='a', it=iTuple()))))
    """
    return json.loads(v, cls=JSONDecoder, **kwargs)

def load_json(f):
    return json.load(f, cls=JSONDecoder)

def dump_json(f, v):
    return json.dump(f, v, cls=JSONEncoder)

# ---------------------------------------------------------------

__all__ = [
    "JSONEncoder",
    "JSONDecoder",
]

# ---------------------------------------------------------------
