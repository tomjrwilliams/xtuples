import typing

import os
import tempfile
import subprocess

import PATHS

SUCCESS = "Success: "
FAILURE = "error: "

def add_boilerplate(s):
    return """

import os
import sys
sys.path.append("{}")

import typing

from xtuples import iTuple, nTuple, iLazy

import xtuples as xt

""".format(PATHS.XTUPLES) + s

def run_mypy(s: str, asserting = None):
    try:
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(bytes(add_boilerplate(s), "utf-8"))
        f.close()
        res = subprocess.run(
            [
                "python",
                "-m",
                "mypy",
                f.name,
                "--check-untyped-defs",
                "--soft-error-limit=-1",
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            env=dict(os.environ, MYPYPATH=PATHS.XTUPLES)
        ).stdout
    finally:
        os.remove(f.name)

    if asserting is None:
        assert SUCCESS in res, s

    elif isinstance(asserting, str):
        assert asserting in res, s

    else:
        assert asserting(res), s

    return True
    
f_map = lambda res: """
f: typing.Callable[..., int] = lambda v: v * 2
res: {} = iTuple.range(3).map(f)
""".format(res)

f_map_lazy = lambda res: """
f: typing.Callable[..., int] = lambda v: v * 2
res: {} = iTuple.range(3).map(f, lazy = True)
""".format(res)

def test_map():
    run_mypy(f_map("iTuple[int]"))
    run_mypy(f_map("int"), asserting=FAILURE)
    run_mypy(f_map("iTuple[str]"), asserting=FAILURE)

    run_mypy(f_map_lazy("iLazy[int]"))
    run_mypy(f_map_lazy("int"), asserting=FAILURE)
    run_mypy(f_map_lazy("iTuple[int]"), asserting=FAILURE)

f_filter = lambda res: """
it: iTuple[int] = iTuple.range(3)
filt = lambda v: v < 3
res: {} = it.filter(filt)
""".format(res)

f_filter_lazy = lambda res: """
it: iTuple[int] = iTuple.range(3)
filt = lambda v: v < 3
res: {} = it.filter(filt, lazy = True)
""".format(res)

def test_filter():
    run_mypy(f_filter("iTuple[int]"))
    run_mypy(f_filter("int"), asserting=FAILURE)
    run_mypy(f_filter("iTuple[str]"), asserting=FAILURE)

    run_mypy(f_filter_lazy("iLazy[int]"))
    run_mypy(f_filter_lazy("int"), asserting=FAILURE)
    run_mypy(f_filter_lazy("iTuple[int]"), asserting=FAILURE)

f_fold = lambda res: """
it: iTuple[int] = iTuple.range(3)
acc = lambda v_acc, v: v_acc + v
i: {} = it.fold(acc, initial=0)
""".format(res)

f_fold_cum = lambda res: """
it: iTuple[int] = iTuple.range(3)
acc = lambda v_acc, v: v_acc + v
i: {} = it.foldcum(acc, initial=0)
""".format(res)

f_fold_star = lambda res: """
it = iTuple.range(3).zip(range(3))
acc = lambda v_acc, v, y: v_acc + v + y
i: {} = it.fold(acc, initial=0, star = True)
""".format(res)

f_fold_cum_star = lambda res: """
it = iTuple.range(3).zip(range(3))
acc = lambda v_acc, v, y: v_acc + v + y
i: {} = it.foldcum(acc, initial=0, star = True)
""".format(res)

f_fold_cum_star_lazy = lambda res: """
it = iTuple.range(3).zip(range(3))
acc = lambda v_acc, v, y: v_acc + v + y
i: {} = it.foldcum(acc, initial=0, star = True,lazy= True)
""".format(res)

def test_fold():
    run_mypy(f_fold("int"))
    run_mypy(f_fold("iTuple"), asserting=FAILURE)
    run_mypy(f_fold("str"), asserting=FAILURE)

    run_mypy(f_fold_star("int"))
    run_mypy(f_fold_star("iTuple"), asserting=FAILURE)
    run_mypy(f_fold_star("str"), asserting=FAILURE)

def test_fold_cum():
    run_mypy(f_fold_cum("iTuple[int]"))
    run_mypy(f_fold_cum("int"), asserting=FAILURE)
    run_mypy(f_fold_cum("iTuple[str]"), asserting=FAILURE)

    run_mypy(f_fold_cum_star("iTuple[int]"))
    run_mypy(f_fold_cum_star("int"), asserting=FAILURE)
    run_mypy(f_fold_cum_star("iTuple[str]"), asserting=FAILURE)

    run_mypy(f_fold_cum_star_lazy("iLazy[int]"))
    run_mypy(f_fold_cum_star_lazy("iTuple[int]"), asserting=FAILURE)