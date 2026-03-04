# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Provide covariance matrix representation.

It is an adaptation of Scipy's implementation adding some representation types.


Abstract Covariance class
^^^^^^^^^^^^^^^^^^^^^^^^^

To represent covariance matrices.

.. autosummary::
   :toctree: _autosummary

    CovarianceMatrix
    CovKernelAsLinop

Specialized Covariance classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Various representation of covariance matrices.

.. autosummary::
   :toctree: _autosummary

    CovViaDiagonal
    CovViaCholesky
    CovViaSparseCholesky
    CovViaPrecisionCholesky
    CovViaSparsePrecisionCholesky
    CovViaEigenFactorization
    CovViaEnsemble
    CovKernelAsLinopViaFFT

Matrix compression
^^^^^^^^^^^^^^^^^^^

Eigen decomposition

.. autosummary::
   :toctree: _autosummary

    get_linop_eigen_factorization
    eigen_factorize_cov_mat

Sparse Helpers
^^^^^^^^^^^^^^

Helpers to work with sparse matrices and covariances.

.. autosummary::
   :toctree: _autosummary

    SparseCholeskyFactor

Working with priors and trends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To represent trend through drift matrix. To use along with geostatistical regularizator.

.. autosummary::
   :toctree: _autosummary

    PriorTerm
    NullPriorTerm
    ConstantPriorTerm
    MeanPriorTerm
    EnsembleMeanPriorTerm
    DriftMatrix
    ConstantDriftMatrix
    LinearDriftMatrix

Other utility functions
^^^^^^^^^^^^^^^^^^^^^^^

To work with covariance matrices and low rank approximations.

.. autosummary::
   :toctree: _autosummary

    get_explained_var

Test data
^^^^^^^^^
Functions providing test data.

.. autosummary::
   :toctree: _autosummary

    load_precision_example_4225x
    get_SPD_sparse_n11_example,
    get_SPD_sparse_example,


"""

from covmats.__about__ import __author__, __email__, __version__
from covmats._covariances import (
    CovarianceMatrix,
    CovKernelAsLinop,
    CovKernelAsLinopViaFFT,
    CovViaCholesky,
    CovViaDiagonal,
    CovViaEigenFactorization,
    CovViaEnsemble,
    CovViaPrecisionCholesky,
    CovViaSparseCholesky,
    CovViaSparsePrecisionCholesky,
    eigen_factorize_cov_mat,
    get_explained_var,
    get_linop_eigen_factorization,
)
from covmats._priors import (
    ConstantDriftMatrix,
    ConstantPriorTerm,
    DriftMatrix,
    EnsembleMeanPriorTerm,
    LinearDriftMatrix,
    MeanPriorTerm,
    NullPriorTerm,
    PriorTerm,
)
from covmats._sparse_helpers import (
    SparseCholeskyFactor,
    get_SPD_sparse_example,
    get_SPD_sparse_n11_example,
)
from covmats.data import load_precision_example_4225x

__all__ = [
    "CovarianceMatrix",
    "CovViaDiagonal",
    "CovViaCholesky",
    "CovViaSparseCholesky",
    "CovViaPrecisionCholesky",
    "CovViaSparsePrecisionCholesky",
    "CovViaEigenFactorization",
    "CovViaEnsemble",
    "CovKernelAsLinopViaFFT",
    "CovKernelAsLinop",
    "PriorTerm",
    "NullPriorTerm",
    "ConstantPriorTerm",
    "MeanPriorTerm",
    "EnsembleMeanPriorTerm",
    "DriftMatrix",
    "ConstantDriftMatrix",
    "LinearDriftMatrix",
    "get_explained_var",
    "get_linop_eigen_factorization",
    "eigen_factorize_cov_mat",
    "__version__",
    "__email__",
    "__author__",
    "load_precision_example_4225x",
    "SparseCholeskyFactor",
    "get_SPD_sparse_n11_example",
    "get_SPD_sparse_example",
]
