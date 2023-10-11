
import sys
import doctest

import xtuples as xt

from . import utils

# ---------------------------------------------------------------

def unpack_doctests(module = xt):
    return doctest.DocTestFinder().find(module)

def unpack_docstrings(module):
    doctests = xt.iTuple(unpack_doctests(module))
    return doctests.map(
        lambda dt: dt.__dict__["docstring"]
    )

def parse_docstring(s: str):
    return "\n\t" + "\n\t".join(
        xt.iTuple(s.split("\n"))
        .map(lambda s: s.strip())
        .filter(
            lambda s: s.startswith(">>>") or s.startswith("...")
        )
        .map(lambda s: s[4:])
    )

def gen_example(i: int, s: str):
    return '''
def example_{}():
{}
'''.format(
    i, s
)

def gen_examples(mod):
    return (
        unpack_docstrings(mod)
        .map(parse_docstring)
        .enumerate()
        .mapstar(gen_example)
    )

# ---------------------------------------------------------------

boilerplate = """

import operator
from xtuples.n import _Example

"""

def test_i():
    utils.run_mypy(
        "\n\n".join(gen_examples(xt.i)),
        extra = boilerplate
    )

def test_n():
    utils.run_mypy(
        "\n\n".join(gen_examples(xt.n)),
        extra = boilerplate
    )

def test_f():
    utils.run_mypy(
        "\n\n".join(gen_examples(xt.f)),
        extra = boilerplate
    )

# ---------------------------------------------------------------
