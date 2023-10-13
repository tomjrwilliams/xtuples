
import xtuples as xt
import xtuples.test_utils as test_utils

# ---------------------------------------------------------------

def create_range_it(l):
    def f():
        return xt.iTuple.range(l)
    f.__name__ = test_utils.outer_func_name()
    return f

def create_range_list(l):
    def f():
        return list(range(l))
    f.__name__ = test_utils.outer_func_name()
    return f

# ---------------------------------------------------------------

def test_create_range():
    test_utils.compare(
        create_range_list(10 ** 3),
        create_range_it(10 ** 3),
        fastest=1,
        f_compare=test_utils.within_multiple(2, fastest=1)
    )
    test_utils.compare(
        create_range_list(10 ** 4),
        create_range_it(10 ** 4),
        fastest=1,
        f_compare=test_utils.within_multiple(2, fastest=1)
    )
    test_utils.compare(
        create_range_list(10 ** 5),
        create_range_it(10 ** 5),
        fastest=1,
        f_compare=test_utils.within_multiple(2, fastest=1),
        iters=10e3
    )
    test_utils.compare(
        create_range_list(10 ** 6),
        create_range_it(10 ** 6),
        fastest=1,
        f_compare=test_utils.within_multiple(2, fastest=1),
        iters=10e2
    )
       
# ---------------------------------------------------------------

# TODO: other creation
# ie. in case there's some magic going on with list(range(n))

# ---------------------------------------------------------------

def map_add_it(l):
    it: xt.iTuple[int] = xt.iTuple.range(l)
    def f():
        return it.map(lambda x: x + 1)
    f.__name__ = test_utils.outer_func_name()
    return f

def map_add_list(l):
    it: xt.iTuple[int] = xt.iTuple.range(l)
    def f():
        res = []
        for x in it:
            res.append(x + 1)
        return res
    f.__name__ = test_utils.outer_func_name()
    return f

def test_map_add():
    test_utils.compare(
        map_add_list(10 ** 3),
        map_add_it(10 ** 3),
        fastest=1,
        f_compare=test_utils.within_multiple(1.5, fastest=1),
    )
    test_utils.compare(
        map_add_list(10 ** 4),
        map_add_it(10 ** 4),
        fastest=1,
        f_compare=test_utils.within_multiple(1.5, fastest=1),
    )
    test_utils.compare(
        map_add_list(10 ** 5),
        map_add_it(10 ** 5),
        fastest=1,
        f_compare=test_utils.within_multiple(1.5, fastest=1),
        iters=10e2
    )

# ---------------------------------------------------------------
