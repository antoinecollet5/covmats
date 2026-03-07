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
one must rely on low-rank approxiations.
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

Once the installation is done, `covmats` is straighforward to use and proposes the following full-rank covariance representations:

- `CovViaDiagonal <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaDiagonal.html#covmats.CovViaDiagonal>`_
- `CovViaCholesky <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaCholesky.html#covmats.CovViaCholesky>`_
- `CovViaSparseCholesky <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaSparseCholesky.html#covmats.CovViaSparseCholesky>`_
- `CovViaPrecisionCholesky <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaPrecisionCholesky.html#covmats.CovViaPrecisionCholesky>`_
- `CovViaSparsePrecisionCholesky <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaSparsePrecisionCholesky.html#covmats.CovViaSparsePrecisionCholesky>`_

It also provides low-rank approximations suitable for large scale problems:

- `CovViaEigenFactorization <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaEigenFactorization.html#covmats.CovViaEigenFactorization>`_
- `CovViaEnsemble <https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaEnsemble.html#covmats.CovViaEnsemble>`_

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

It is possible to obtain a dense representation in a straighforward manner:

.. code-block:: python

    cov_diag33 = covmats.CovViaDiagonal(d)
    dist = sp.stats.multivariate_normal(mean=[0, 0, 0], cov=cov_diag33)
    dist.pdf(x)

.. code-block:: python

    np.float64(4.9595685102808205e-08)


It is compatible with the stats API from scipy since the base class inherit from `Covariance`.

🏗️ Complete example with supporting paper coming Q1 2026.

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

.. [scipy| text:: scipy.stats.Covariance
    :target: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Covariance.html>`_

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
    :alt: Downoads

.. |Build Status| image:: https://github.com/antoinecollet5/covmats/actions/workflows/main.yml/badge.svg
    :target: https://github.com/antoinecollet5/covmats/actions/workflows/main.yml
    :alt: Build Status

.. |Documentation Status| image:: https://readthedocs.org/projects/covmats/badge/?version=latest
    :target: https://covmats.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Coverage| image:: https://codecov.io/gh/antoinecollet5/covmats/branch/master/graph/badge.svg?token=ISE874MMOF
    :target: https://codecov.io/gh/antoinecollet5/covmats
    :alt: Coverage

.. |Codacy| image:: https://app.codacy.com/project/badge/Grade/c41f65d98b824de394162520b0d8a17a
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

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.11384588.svg
   :target: https://doi.org/10.5281/zenodo.11384588
