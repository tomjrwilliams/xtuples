
# ---------------------------------------------------------------

import sys
import pathlib
import importlib.util

path = pathlib.Path("./src/xtuples/__init__.py").resolve()
spec = importlib.util.spec_from_file_location(
    "xtuples", str(path)
    #
)

xtuples = importlib.util.module_from_spec(spec)
sys.modules["xtuples"] = xtuples
spec.loader.exec_module(xtuples)

# ---------------------------------------------------------------

print(xtuples.iTuple.range(3))

# ---------------------------------------------------------------
