=======
covmats
=======

|License| |Stars| |Python| |PyPI| |Downloads| |Build Status| |Documentation Status| |Coverage| |Codacy| |Precommit: enabled| |Ruff| |ty| |DOI|

🐍 Covariance matrices representation.

**The complete and up to date documentation can be found here**: https://covmats.readthedocs.io.

===============
🎯 Motivations
===============

Calculations involving covariance matrices (e.g. linear algebra, data whitening,
multivariate normal function evaluation) are often performed more efficiently using
a decomposition of the covariance matrix instead of the covariance matrix itself.
For large scale application, a dense covariance matrix would not even fit in memory and
one must rely on low-rank approximations.
This package allows the user to construct an object representing a covariance matrix
using any of several decompositions/approximations and perform calculations using a
common interface.

The common interface `CovarianceMatrix <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovarianceMatrix.html#covmats.CovarianceMatrix>`_
can be seen as a extension of the class `scipy.stats.Covariance <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Covariance.html>`_
as it inherits from it (thus making it compatible with all
`scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ functions and classes)
and dope it with `LinearOperator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_ capabilities.

The package is used in large-scale inversion packages such as
`pypcga <https://github.com/antoinecollet5/pypcga>`_,
`pyesmda <https://github.com/antoinecollet5/pyesmda>`_ and
`pyrtid <https://github.com/antoinecollet5/pyrtid>`_.

===============
🚀 Quick start
===============

To install `covmats`, the easiest way is through `pip`:

.. code-block::

    pip install covmats

Or alternatively using `conda`

.. code-block::

    conda install covmats

You might also clone the repository and install from source

.. code-block::

    pip install -e .

Once the installation is done, `covmats` is straightforward to use and proposes the following full-rank covariance representations:

- `CovViaDiagonal <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaDiagonal.html#covmats.CovViaDiagonal>`_
- `CovViaCholesky <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaCholesky.html#covmats.CovViaCholesky>`_
- `CovViaSparseCholesky <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaSparseCholesky.html#covmats.CovViaSparseCholesky>`_
- `CovViaPrecisionCholesky <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaPrecisionCholesky.html#covmats.CovViaPrecisionCholesky>`_
- `CovViaSparsePrecisionCholesky <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaSparsePrecisionCholesky.html#covmats.CovViaSparsePrecisionCholesky>`_

It also provides low-rank approximations suitable for large scale problems:

- `CovViaEigenFactorization <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaEigenFactorization.html#covmats.CovViaEigenFactorization>`_
- `CovViaEnsemble <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaEnsemble.html#covmats.CovViaEnsemble>`_

as well as kernel-based, matrix-free linear operators for point-cloud data:

- `CovKernelAsLinop <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovKernelAsLinop.html#covmats.CovKernelAsLinop>`_
- `CovKernelAsLinopViaFFT <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovKernelAsLinopViaFFT.html#covmats.CovKernelAsLinopViaFFT>`_

and a small hierarchy of prior/drift terms for geostatistical regularization:

- `PriorTerm <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.PriorTerm.html#covmats.PriorTerm>`_,
  `NullPriorTerm <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.NullPriorTerm.html#covmats.NullPriorTerm>`_,
  `ConstantPriorTerm <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.ConstantPriorTerm.html#covmats.ConstantPriorTerm>`_,
  `MeanPriorTerm <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.MeanPriorTerm.html#covmats.MeanPriorTerm>`_,
  `EnsembleMeanPriorTerm <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.EnsembleMeanPriorTerm.html#covmats.EnsembleMeanPriorTerm>`_
- `DriftMatrix <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.DriftMatrix.html#covmats.DriftMatrix>`_,
  `ConstantDriftMatrix <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.ConstantDriftMatrix.html#covmats.ConstantDriftMatrix>`_,
  `LinearDriftMatrix <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.LinearDriftMatrix.html#covmats.LinearDriftMatrix>`_

The two companion tutorial notebooks, `examples_covariances.py <https://github.com/antoinecollet5/covmats/blob/master/examples_covariances.py>`_
and `examples_priors.py <https://github.com/antoinecollet5/covmats/blob/master/examples_priors.py>`_,
walk through every one of these classes in detail. The rest of this section gives a
condensed overview.

Let's start by importing `numpy`, `scipy` and `covmats` for the tests and define a
random number generator seed for reproducibility:

.. code-block:: python

    import numpy as np
    import scipy as sp
    import covmats

    rng_seed = 2026

First example with a diagonal covariance matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following, we define a `(3 x 3)` covariance matrix defining only its diagonal (i.e., all elements of the random vectors are independent).
Using scipy, it is possible to compute the pdf.

.. code-block:: python

    d = [1, 2, 3]
    A33 = np.diag(d)  # a diagonal covariance matrix
    x = [4, -2, 5]  # a point of interest
    dist = sp.stats.multivariate_normal(mean=[0, 0, 0], cov=A33)
    dist.pdf(x)

.. code-block:: python

    np.float64(4.9595685102808205e-08)

It is possible to obtain a dense representation in a straightforward manner:

.. code-block:: python

    cov_diag33 = covmats.CovViaDiagonal(d)
    dist = sp.stats.multivariate_normal(mean=[0, 0, 0], cov=cov_diag33)
    dist.pdf(x)

.. code-block:: python

    np.float64(4.9595685102808205e-08)


It is compatible with the stats API from scipy since the base class inherit from `Covariance`.

Every representation also exposes ``todense``, ``solve``, ``precision`` and
``log_pdet``, as well as ``sample_mvnormal``, ``whiten`` and ``colorize`` for
fast Monte-Carlo sampling and data whitening:

.. code-block:: python

    samples = cov_diag33.sample_mvnormal(shape=[10000], random_state=rng_seed)
    np.round(np.cov(samples, rowvar=False), 1)

.. code-block:: python

    array([[ 1.,  0., -0.],
           [ 0.,  2., -0.],
           [-0., -0.,  3.]])

Cholesky and precision-based representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`CovViaCholesky` wraps a dense Cholesky factor `L` such that `A = L @ L.T`, while
`CovViaPrecisionCholesky` does the same for the *precision* matrix `Q = A^-1`
(the natural representation for Gaussian Markov Random Fields, where `Q` is
sparse). Sparse counterparts, `CovViaSparseCholesky` and
`CovViaSparsePrecisionCholesky`, build on a `SparseCholeskyFactor`
(an `LDL'` factorization) for large, sparse problems:

.. code-block:: python

    rng = np.random.default_rng(rng_seed)
    n = 4
    B = rng.random((n, n))
    A = B @ B.T + n * np.eye(n)  # a random SPD matrix
    cov_cho = covmats.CovViaCholesky(np.linalg.cholesky(A))
    np.allclose(cov_cho.todense(), A)

.. code-block:: python

    True

Low-rank representations
~~~~~~~~~~~~~~~~~~~~~~~~~

For very large problems, `CovViaEigenFactorization` (a truncated eigen
decomposition) and `CovViaEnsemble` (an ensemble of anomalies, as used in
ensemble Kalman filtering) never require forming the dense covariance matrix:

.. code-block:: python

    ensemble = rng.multivariate_normal(np.zeros(6), np.eye(6) * 2 + 0.3, size=200)
    cov_ens = covmats.CovViaEnsemble(ensemble)
    cov_ens.shape

.. code-block:: python

    (6, 6)

`get_linop_eigen_factorization` and `eigen_factorize_cov_mat` build a
randomized low-rank eigen factorization directly from any `CovarianceMatrix`
or `CovKernelAsLinop` instance, and `get_explained_var` reports how much
variance each retained mode captures -- useful to pick a truncation rank.

Kernel-based linear operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the covariance is defined analytically through a kernel evaluated on a
point cloud, `CovKernelAsLinop` (dense evaluation, any point cloud) and
`CovKernelAsLinopViaFFT` (FFT-based, regular grids only) expose a
`LinearOperator` without requiring the covariance matrix to be assembled
up front:

.. code-block:: python

    pts = covmats.get_pts_coords_regular_grid(mesh_dim=1.0, shape=(6, 6))
    cov_kernel = covmats.CovKernelAsLinop(
        pts, lambda d: np.exp(-d), len_scale=np.array([2.0, 2.0])
    )
    cov_kernel.shape

.. code-block:: python

    (36, 36)

Priors and drift matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~

Alongside covariance representations, `covmats` provides `PriorTerm`
subclasses to describe the deterministic mean/trend of a field, from a simple
`NullPriorTerm` to a `LinearDriftMatrix` expressing a spatially-varying
trend as `m = X @ beta`:

.. code-block:: python

    drift = covmats.LinearDriftMatrix(pts)
    drift.beta = np.array([1.0, 0.5, -0.5])
    drift.get_values(np.zeros(pts.shape[0])).shape

.. code-block:: python

    (36,)

Conditional and unconditional simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See `https://github.com/antoinecollet5/covmats/blob/main/notebooks/examples_simulations.ipynb <https://github.com/antoinecollet5/covmats/blob/main/notebooks/examples_simulations.ipynb>`_

🏗️ Complete example with supporting paper coming Q1 2027.

===========
🔑 License
===========

This project is released under the **BSD 3-Clause License**.

Copyright (c) 2026, Antoine COLLET. All rights reserved.

For more details, see the `LICENSE <https://github.com/antoinecollet5/covmats/blob/master/LICENSE>`_ file included in this repository.

==============
⚠️ Disclaimer
==============

This software is provided "as is", without warranty of any kind, express or implied,
including but not limited to the warranties of merchantability, fitness for a particular purpose,
or non-infringement. In no event shall the authors or copyright holders be liable for
any claim, damages, or other liability, whether in an action of contract, tort,
or otherwise, arising from, out of, or in connection with the software or the use
or other dealings in the software.

By using this software, you agree to accept full responsibility for any consequences,
and you waive any claims against the authors or contributors.

==========
📧 Contact
==========

For questions, suggestions, or contributions, you can reach out via:

- Email: antoinecollet5@gmail.com
- GitHub: https://github.com/antoinecollet5/covmats

We welcome contributions!

=============
📚 References
=============

TODO

* Free software: SPDX-License-Identifier: BSD-3-Clause

.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
    :target: https://github.com/antoinecollet5/covmats/blob/master/LICENSE

.. |Stars| image:: https://img.shields.io/github/stars/antoinecollet5/covmats.svg?style=social&label=Star&maxAge=2592000
    :target: https://github.com/antoinecollet5/covmats/stargazers
    :alt: Stars

.. |Python| image:: https://img.shields.io/pypi/pyversions/covmats.svg
    :target: https://pypi.org/pypi/covmats
    :alt: Python

.. |PyPI| image:: https://img.shields.io/pypi/v/covmats.svg
    :target: https://pypi.org/pypi/covmats
    :alt: PyPI

.. |Downloads| image:: https://static.pepy.tech/badge/covmats
    :target: https://pepy.tech/project/covmats
    :alt: Downloads

.. |Build Status| image:: https://github.com/antoinecollet5/covmats/actions/workflows/main.yml/badge.svg
    :target: https://github.com/antoinecollet5/covmats/actions/workflows/main.yml
    :alt: Build Status

.. |Documentation Status| image:: https://readthedocs.org/projects/covmats/badge/?version=latest
    :target: https://covmats.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Coverage| image:: https://codecov.io/gh/antoinecollet5/covmats/graph/badge.svg?token=8lE90wylXL
    :target: https://codecov.io/gh/antoinecollet5/covmats
    :alt: Coverage

.. |Codacy| image:: https://app.codacy.com/project/badge/Grade/122673cd1d104aa28ada0c44b1f4e7d6
    :target: https://app.codacy.com/gh/antoinecollet5/covmats/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
    :alt: codacy

.. |Precommit: enabled| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :target: https://github.com/pre-commit/pre-commit

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. |ty| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json
    :target: https://github.com/astral-sh/ty
    :alt: Checked with ty

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.18900358.svg
   :target: https://doi.org/10.5281/zenodo.18900358
