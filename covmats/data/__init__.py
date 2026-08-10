from importlib import resources

import numpy as np
import scipy as sp

from covmats._sparse_helpers import SparseCholeskyFactor


def load_precision_example_4225x() -> sp.sparse.csc_array:
    with (
        resources.files("covmats.data")
        .joinpath("precision_example_4225x.mtx")
        .open("rb")
    ) as f:
        return sp.io.mmread(f, spmatrix=False).tocsc()


def load_precision_example_4225x_SCF() -> SparseCholeskyFactor:
    L = (
        resources.files("covmats.data")
        .joinpath("precision_example_4225x_L.mtx")
        .open("rb")
    )
    D = (
        resources.files("covmats.data")
        .joinpath("precision_example_4225x_D.mtx")
        .open("rb")
    )
    P = np.loadtxt(
        resources.files("covmats.data").joinpath("precision_example_4225x_P.txt")  # ty:ignore[invalid-argument-type]
    )
    return SparseCholeskyFactor(
        sp.io.mmread(L, spmatrix=False).tocsc(),
        sp.io.mmread(D, spmatrix=False).tocsc(),
        P,
    )
