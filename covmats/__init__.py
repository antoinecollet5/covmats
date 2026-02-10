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
    CovViaKernel

Specialized Covariance classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Various representation of covariance matrices.

.. autosummary::
   :toctree: _autosummary

    CovViaDense
    CovViaDiagonal
    CovViaCholesky
    CovViaSparseCholesky
    CovViaPrecision
    CovViaSparsePrecision
    CovViaEigenFactorization
    CovViaEnsemble
    CovViaFFT

Matrix compression
^^^^^^^^^^^^^^^^^^^

Eigen decomposition

.. autosummary::
   :toctree: _autosummary

    get_matrix_eigen_factorization
    eigen_factorize_cov_mat

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

    sample_from_sparse_cov_factor
    get_explained_var


"""

from covmats.__about__ import __author__, __email__, __version__
from covmats._covariances import (
    CovarianceMatrix,
    CovViaCholesky,
    CovViaDense,
    CovViaDiagonal,
    CovViaEigenFactorization,
    CovViaEnsemble,
    CovViaFFT,
    CovViaKernel,
    CovViaPrecision,
    CovViaSparseCholesky,
    CovViaSparsePrecision,
    eigen_factorize_cov_mat,
    generate_dense_matrix,
    get_explained_var,
    get_matrix_eigen_factorization,
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
from covmats._sparse_helpers import sample_from_sparse_cov_factor

__all__ = [
    "CovarianceMatrix",
    "CovViaDense",
    "CovViaDiagonal",
    "CovViaCholesky",
    "CovViaSparseCholesky",
    "CovViaPrecision",
    "CovViaSparsePrecision",
    "CovViaEigenFactorization",
    "CovViaEnsemble",
    "CovViaFFT",
    "CovViaKernel",
    "PriorTerm",
    "NullPriorTerm",
    "ConstantPriorTerm",
    "MeanPriorTerm",
    "EnsembleMeanPriorTerm",
    "DriftMatrix",
    "ConstantDriftMatrix",
    "LinearDriftMatrix",
    "get_matrix_eigen_factorization",
    "get_explained_var",
    "eigen_factorize_cov_mat",
    "sample_from_sparse_cov_factor",
    "generate_dense_matrix",
    "__version__",
    "__email__",
    "__author__",
]
