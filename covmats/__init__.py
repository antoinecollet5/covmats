# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Covariance matrix representations and priors for large scale inverse problems.

Abstract Covariance class
^^^^^^^^^^^^^^^^^^^^^^^^^

The common interface `CovarianceMatrix
<https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovarianceMatrix.html#covmats.CovarianceMatrix>`_
can be seen as a extension of the class `scipy.stats.Covariance
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Covariance.html>`_
as it inherits from it (thus making it compatible with all
`scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_
functions and classes)
and dope it with `LinearOperator
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_
capabilities.

.. autosummary::
   :toctree: _autosummary

    CovarianceMatrix


Full-rank covariance matrix decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Several full-rank representation of covariance matrices.

.. autosummary::
   :toctree: _autosummary

    CovViaDiagonal
    CovViaCholesky
    CovViaSparseCholesky
    CovViaPrecisionCholesky
    CovViaSparsePrecisionCholesky

Low-rank approximations
^^^^^^^^^^^^^^^^^^^^^^^

Several low-rank (approximate) representation of covariance matrices.

.. autosummary::
   :toctree: _autosummary

    CovViaEigenFactorization
    CovViaEnsemble

Kernel based approximations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Low-rank approximations from a given Kernel. Only provides the linear operations
capabilities (no sampling not statistical calculations).

.. autosummary::
   :toctree: _autosummary

    CovKernelAsLinop
    CovKernelAsLinopViaFFT

Matrix compression
^^^^^^^^^^^^^^^^^^^

Allow to factorize :py:class:`CovarianceMatrix` and :py:class:`CovKernelAsLinop` using
Eigen low-rank approximation.

.. autosummary::
   :toctree: _autosummary

    get_linop_eigen_factorization
    eigen_factorize_cov_mat

Sparse Helpers
^^^^^^^^^^^^^^

Helper to work with sparse matrices and covariances.

.. autosummary::
   :toctree: _autosummary

    SparseCholeskyFactor

Grid utilities
^^^^^^^^^^^^^^

Helper to build regular grids of points, typically used with
:py:class:`CovKernelAsLinop` and :py:class:`CovKernelAsLinopViaFFT`.

.. autosummary::
   :toctree: _autosummary

    get_pts_coords_regular_grid

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
    load_precision_example_4225x_SCF
    get_SPD_sparse_n11_example
    get_SPD_sparse_example


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
from covmats._helpers import (
    get_pts_coords_regular_grid,
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
from covmats.data import (
    load_precision_example_4225x,
    load_precision_example_4225x_SCF,
)

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
    "load_precision_example_4225x_SCF",
    "SparseCholeskyFactor",
    "get_SPD_sparse_n11_example",
    "get_SPD_sparse_example",
    "get_pts_coords_regular_grid",
]
