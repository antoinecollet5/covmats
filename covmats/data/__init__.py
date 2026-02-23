from importlib import resources

import scipy as sp


def load_precision_example_4225x() -> sp.sparse.csc_array:
    with (
        resources.files("covmats.data")
        .joinpath("precision_example_4225x.mtx")
        .open("rb")
    ) as f:
        return sp.io.mmread(f).tocsc()
