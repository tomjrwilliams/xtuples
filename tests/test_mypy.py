
import xtuples.test_utils as test_utils
    
# ---------------------------------------------------------------

f_zip = lambda rng_0_type, rng_1_type, v: """
i_rng_0: {}
i_rng_1: {}
i_rng_0, i_rng_1 = {}
""".format(rng_0_type, rng_1_type, v)

def test_zip():
    z = "iTuple.range(3).zip(range(3)).zip()"
    test_utils.run_mypy(f_zip(
        "typing.Iterable[int]",
        "typing.Iterable[int]",
        z
    ))
    test_utils.run_mypy(f_zip(
        "typing.Iterable[str]",
        "typing.Iterable[int]",
        z
    ), asserting=test_utils.FAILURE)
    test_utils.run_mypy(f_zip(
        "int",
        "typing.Iterable[int]",
        z
    ), asserting=test_utils.FAILURE)

# ---------------------------------------------------------------

f_map = lambda res_type: """
f: typing.Callable[..., int] = lambda v: v * 2
res: {} = iTuple.range(3).map(f, lazy=False)
""".format(res_type)

f_map_lazy = lambda res_type: """
f: typing.Callable[..., int] = lambda v: v * 2
res: {} = iTuple.range(3).map(f, lazy = True)
""".format(res_type)

f_map_n = lambda f_type, f, res_type, it: """
f: {} = {}
v = {}
res: {} = v.map(f)
""".format(f_type, f, it, res_type)

f_map_n_star = lambda f_type, f, res_type, it: """
f: {} = {}
v = {}
res: {} = v.map(f, star=True)
""".format(f_type, f, it, res_type)

def test_map():
    test_utils.run_mypy(f_map("iTuple[int]"))
    test_utils.run_mypy(f_map("int"), asserting=test_utils.FAILURE)
    test_utils.run_mypy(f_map("iTuple[str]"), asserting=test_utils.FAILURE)

    test_utils.run_mypy(f_map_lazy("iLazy[int]"))
    test_utils.run_mypy(f_map_lazy("int"), asserting=test_utils.FAILURE)
    test_utils.run_mypy(f_map_lazy("iTuple[int]"), asserting=test_utils.FAILURE)

    test_utils.run_mypy(
        f_map_n(
            "typing.Callable[[int], int]",
            "lambda v: v * 2",
            "iTuple[int]",
            "iTuple.range(3).zip(range(3))"
        ),
        asserting=test_utils.FAILURE
    )
    test_utils.run_mypy(
        f_map_n_star(
            "typing.Callable[[int], int]",
            "lambda v0, v1: v0 * 2",
            "iTuple[int]",
            "iTuple.range(3).zip(range(3))"
        ),
        asserting=test_utils.FAILURE
    )
    test_utils.run_mypy(
        f_map_n(
            "typing.Callable[[int, int], int]",
            "lambda v0, v1: v0 * 2",
            "iTuple[int]",
            "iTuple.range(3).zip(range(3))"
        ),
        asserting=test_utils.FAILURE
    )
    test_utils.run_mypy(
        f_map_n_star(
            "typing.Callable[..., int]",
            "lambda v0, v1: v0 * 2",
            "iTuple[int]",
            "iTuple.range(3).zip(range(3))"
        )
    )

f_filter = lambda res_type: """
it: iTuple[int] = iTuple.range(3)
filt = lambda v: v < 3
res: {} = it.filter(filt, lazy=False)
""".format(res_type)

f_filter_lazy = lambda res_type: """
it: iTuple[int] = iTuple.range(3)
filt = lambda v: v < 3
res: {} = it.filter(filt, lazy = True)
""".format(res_type)

def test_filter():
    test_utils.run_mypy(f_filter("iTuple[int]"))
    test_utils.run_mypy(f_filter("int"), asserting=test_utils.FAILURE)
    test_utils.run_mypy(f_filter("iTuple[str]"), asserting=test_utils.FAILURE)

    test_utils.run_mypy(f_filter_lazy("iLazy[int]"))
    test_utils.run_mypy(f_filter_lazy("int"), asserting=test_utils.FAILURE)
    test_utils.run_mypy(f_filter_lazy("iTuple[int]"), asserting=test_utils.FAILURE)

f_fold = lambda res_type: """
it: iTuple[int] = iTuple.range(3)
acc = lambda v_acc, v: v_acc + v
i: {} = it.fold(acc, initial=0)
""".format(res_type)

f_fold_cum = lambda res_type: """
it: iTuple[int] = iTuple.range(3)
acc = lambda v_acc, v: v_acc + v
i: {} = it.foldcum(acc, initial=0)
""".format(res_type)

f_fold_star = lambda res_type: """
it = iTuple.range(3).zip(range(3))
acc = lambda v_acc, v, y: v_acc + v + y
i: {} = it.fold(acc, initial=0, star = True)
""".format(res_type)

f_fold_cum_star = lambda res_type: """
it = iTuple.range(3).zip(range(3))
acc = lambda v_acc, v, y: v_acc + v + y
i: {} = it.foldcum(acc, initial=0, star = True)
""".format(res_type)

f_fold_cum_star_lazy = lambda res_type: """
it = iTuple.range(3).zip(range(3))
acc = lambda v_acc, v, y: v_acc + v + y
i: {} = it.foldcum(acc, initial=0, star = True,lazy= True)
""".format(res_type)

def test_fold():
    test_utils.run_mypy(f_fold("int"))
    test_utils.run_mypy(f_fold("iTuple"), asserting=test_utils.FAILURE)
    test_utils.run_mypy(f_fold("str"), asserting=test_utils.FAILURE)

    test_utils.run_mypy(f_fold_star("int"))
    test_utils.run_mypy(f_fold_star("iTuple"), asserting=test_utils.FAILURE)
    test_utils.run_mypy(f_fold_star("str"), asserting=test_utils.FAILURE)

def test_fold_cum():
    test_utils.run_mypy(
        f_fold_cum("iTuple[int]")
    )
    test_utils.run_mypy(
        f_fold_cum("int"), asserting=test_utils.FAILURE
    )
    test_utils.run_mypy(
        f_fold_cum("iTuple[str]"), asserting=test_utils.FAILURE
    )

    test_utils.run_mypy(
        f_fold_cum_star("iTuple[int]")
    )
    test_utils.run_mypy(
        f_fold_cum_star("int"), asserting=test_utils.FAILURE
    )
    test_utils.run_mypy(
        f_fold_cum_star("iTuple[str]"), asserting=test_utils.FAILURE
    )

    test_utils.run_mypy(
        f_fold_cum_star_lazy("iLazy[int]")
    )
    test_utils.run_mypy(
        f_fold_cum_star_lazy("iTuple[int]"), asserting=test_utils.FAILURE
    )

f_append = lambda v_type, v_val, res_type: """
it: iTuple[int] = iTuple.range(3)
v: {} = {}
res: {} = it.append(v)
""".format(v_type, v_val, res_type)

def test_append():
    test_utils.run_mypy(
        f_append("int", 1, "iTuple[int]")
    )
    test_utils.run_mypy(
        f_append("str", '"s"', "iTuple[int]"), asserting=test_utils.FAILURE
    )
    test_utils.run_mypy(
        f_append("str", '"s"', "iTuple[typing.Union[int, str]]")
    )

# ---------------------------------------------------------------
