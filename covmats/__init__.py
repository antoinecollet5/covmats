"""Provide covariance matrix representation.

It is an adaptation of Scipy's implementation adding some representation types.

Note: add some notes about:
https://github.com/arvindks/kle/blob/master/covariance/covariance.py

And cite Saibaba's phd thesis about the uncertainty and all.


Covariance classes
^^^^^^^^^^^^^^^^^^

To represent covariance matrices.

.. autosummary::
   :toctree: _autosummary

    CovarianceMatrix
    CovViaDense
    CovViaDiagonal
    CovViaEnsemble
    CovViaCholesky
    CovViaSparseCholesky
    CovViaPrecision
    CovViaSparsePrecision
    CovViaFFT
    CovViaEigendecomposition
    CovViaSparsePrecision
    CovViaHierarchical
    CovViaSparsePrecision


Covariance functions
^^^^^^^^^^^^^^^^^^^^

To work with covariance matrices and low rank approximations.

.. autosummary::
   :toctree: _autosummary

    eigen_factorize_cov_mat
    generate_dense_matrix
    get_matrix_eigen_factorization
    sample_from_sparse_cov_factor
    get_explained_var


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

Matrix compression
^^^^^^^^^^^^^^^^^^^

Eigen decomposition

.. autosummary::
   :toctree: _autosummary

    get_matrix_eigen_factorization
    eigen_factorize_cov_mat

"""

from covmats._covariances import (
    CovarianceMatrix,
    CovViaCholesky,
    CovViaDense,
    CovViaEigendecomposition,
    CovViaEnsemble,
    CovViaFFT,
    CovViaHierarchical,
    CovViaPrecision,
    CovViaPSD,
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
    "CovViaCholesky",
    "CovViaSparseCholesky",
    "CovViaPrecision",
    "CovViaSparsePrecision",
    "CovViaEigendecomposition",
    "CovViaEnsemble",
    "CovViaFFT",
    "CovViaHierarchical",
    "CovViaPSD",
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
]
