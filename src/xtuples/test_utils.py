
import inspect
import itertools
import datetime

from pympler.asizeof import asizeof  # type: ignore

import xtuples as xt

# ---------------------------------------------------------------

def lineno(n_back = 0):
    frame = inspect.currentframe()
    for _ in range(n_back + 1):
        assert frame is not None
        frame = frame.f_back
    assert frame is not None
    return frame.f_lineno

def outer_func_name(n_back = 0):
    frame = inspect.currentframe()
    for _ in range(n_back + 1):
        assert frame is not None
        frame = frame.f_back
    assert frame is not None
    return frame.f_code.co_name

# ---------------------------------------------------------------

def log_time(start, times):
    end = datetime.datetime.now()
    t = (end - start).microseconds
    times.append(t)
    return end, t

def time_funcs(*fs, iters = 10e4, max_time = 10e6):
    
    n = len(fs)

    times: list[list[float]] = [[] for _ in fs]
    mems: list[list[float]] = [[] for _ in fs]

    # f_times_mems = itertools.cycle(zip(
    #     range(n), fs, times, mems, 
    # )) # type: ignore

    runs = int(iters / 100)
    start = datetime.datetime.now()

    total = 0
    incr = 0

    for _ in range(100):
        for i in range(n):

            f = fs[i]
            ts = times[i]
            ms = mems[i]

            start = datetime.datetime.now()

            for _ in range(runs):
                res = f()

            start, t = log_time(start, ts)
            ms.append(asizeof(res))
            
            total += t

        if total > max_time:
            break
        
        # discard one instance
        # so each time we start with a different func
        # i, _, _, _ = next(f_times_mems)
        incr += 1

    return (incr,) + tuple([f() for f in fs]) + tuple(
        (sum(ts) / len(ts)) / 1000 for ts in times
    ) + tuple(
        (sum(ms) / len(ms)) / 1000 for ms in mems
    )

    # in milliseconds

# ---------------------------------------------------------------

def within_multiple(m, fastest=1):
    assert fastest in (0, 1,)
    def f_compare(v1, v2, m1, m2):
        v1, v2 = (
            (v1, v2,) 
            if fastest == 1 
            else (v2, v1,)
        )
        if v2 < v1:
            return True
        return (v1 * m) >= v2
    return f_compare

def excess_within_percent(pct, fastest=1):
    assert fastest in (0, 1,)
    def f_compare(v1, v2, m1, m2):
        v1, v2 = (
            (v1, v2,) 
            if fastest == 1 
            else (v2, v1,)
        )
        if v2 < v1:
            return True
        return (v2 - v1) < (v1 * (pct / 100))
    return f_compare

def compare(
    f1,
    f2,
    fastest = 1,
    f_compare = None,
    f_eq = None,
    **kwargs,
):

    print("--")

    (
        loops,
        res_1,
        res_2,
        millis_1,
        millis_2,
        mem_1,
        mem_2
    ) = time_funcs(f1, f2, **kwargs)

    passed = ((
        round(millis_2, 1) <= round(millis_1, 1)
        if fastest == 1
        else round(millis_1, 1) <= round(millis_2, 1)
        if fastest == 0
        else True
    ) if f_compare is None else f_compare(
        millis_1, 
        millis_2,
        mem_1,
        mem_2,
    )) and (
        True if f_eq is None else f_eq(res_1, res_2)
    )

    result = {
        **{
            "iters": "{}%".format(loops),
            "pass": passed,
            "line": lineno(n_back=1),
        },
        **({} if fastest is None else {
            "fastest": fastest,
        }),
        **{
            "memory": {
                f1.__name__: mem_1,
                f2.__name__: mem_2,
            },
            "milliseconds": {
                f1.__name__: round(millis_1, 2),
                f2.__name__: round(millis_2, 2),
            },
            "result": res_1 if f_eq(res_1, res_2) else {
                f1.__name__: res_1,
                f2.__name__: res_2,
            },
        },
    }
    for k, v in result.items():
        print(k, v)

    assert passed, result

# ---------------------------------------------------------------

import os

import tempfile
import subprocess

import pathlib

assert "xtuples" in os.environ

XTUPLES = str(pathlib.Path(os.environ["xtuples"]))

def run_py(s: str):
    try:
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(bytes(s.replace("\\", "\\\\"), "utf-8"))
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
            env=dict(os.environ, MYPYPATH=XTUPLES)
        ).stdout
    finally:
        os.remove(f.name)
    return res

# ---------------------------------------------------------------

SUCCESS = "Success: "
FAILURE = "error: "

def annotate(s: str, offset = 0):
    return "\n".join(
        xt.iTuple(s.split("\n"))
        .enumerate()
        .mapstar(lambda i, s: (
            s if not len(s.strip())
            else s + " # line {}".format(i + offset)
        ))
    )

def gen_boilerplate(extra: str = ''):
    return """

import os
import sys
sys.path.append("{}")

import typing

from xtuples import iTuple, nTuple, iLazy

import xtuples as xt

{}

""".format(XTUPLES, extra)


def run_mypy(s: str, asserting = None, extra: str = ''):
    
    boiler = gen_boilerplate(extra=extra)
    res = run_py(boiler + s)

    s = annotate(s, offset = len(boiler.split("\n")))

    try:
        if asserting is None:
            assert SUCCESS in res

        elif isinstance(asserting, str):
            assert asserting in res

        else:
            assert asserting(res)

    except Exception as e:

        print(s)
        print(res)

        raise e

    return True

# ---------------------------------------------------------------
