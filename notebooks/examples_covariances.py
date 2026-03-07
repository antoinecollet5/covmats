import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Covariance matrices representations
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    For this tutorial, it is required to install `covmats` with the required modules
    for the notebook. The easiest way is through `pip`:

    ```bash
        pip install covmats[examples]
    ```

    Or alternatively using `conda`

    ```bash
        conda install covmats[examples]
    ```

    You might also clone the repository and install from source

    ```bash
        pip install -e ".[examples]"
    ```

    Then import the modules
    """)
    return


@app.cell
def _():
    import covmats
    import marimo as mo
    import numpy as np
    import scipy as sp

    # potential licence issue under the cholesky factorization section.
    return covmats, mo, np, sp


@app.cell
def _(mo):
    mo.md(r"""
    Once the installation is done, `covmats` gives access to various covariance matrix
    representations for large scale inversion, fast linear algebra and fast sampling.

    The common interface
    [CovarianceMatrix](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovarianceMatrix.html#covmats.CovarianceMatrix)
    can be seen as a extension of the class
    [scipy.stats.Covariance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Covariance.html)
    as it inherits from it (thus making it compatible with all
    [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) functions and
    classes) and dope it with
    [LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)
    capabilities.

    Derived from this base class, `covmats` provides several full-rank representation of
    covariance matrices:

    - [CovViaDiagonal](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaDiagonal.html#covmats.CovViaDiagonal)
    - [CovViaCholesky](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaCholesky.html#covmats.CovViaCholesky)
    - [CovViaSparseCholesky](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaSparseCholesky.html#covmats.CovViaSparseCholesky)
    - [CovViaPrecisionCholesky](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaPrecisionCholesky.html#covmats.CovViaPrecisionCholesky)
    - [CovViaSparsePrecisionCholesky](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaSparsePrecisionCholesky.html#covmats.CovViaSparsePrecisionCholesky)

    as well as several low-rank (approximate) representation of covariance matrices:

    - [CovViaEigenFactorization](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaEigenFactorization.html#covmats.CovViaEigenFactorization)
    - [CovViaEnsemble](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovViaEnsemble.html#covmats.CovViaEnsemble)

    In addition, it is possible to build low-rank approximations from a given Kernel.
    But it only provides the linear operations
    capabilities (no sampling not statistical calculations).

    - [CovKernelAsLinop](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovKernelAsLinop.html#covmats.CovKernelAsLinop)
    - [CovKernelAsLinopViaFFT](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.CovKernelAsLinopViaFFT.html#covmats.CovKernelAsLinopViaFFT)

    It is also possible to obtain low-rank factorization from any
    :py:class:`CovarianceMatrix` and :py:class:`CovKernelAsLinop` using
    randomized Eigen factorization:

    - [get_linop_eigen_factorization](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.get_linop_eigen_factorization.html#covmats.get_linop_eigen_factorization)
    - [eigen_factorize_cov_mat](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.eigen_factorize_cov_mat.html#covmats.eigen_factorize_cov_mat)

    By convention, we will refer to all covariance matrices as $\mathbf{A}$ and their
    respective inverses, aka precision matrices, as $\mathbf{Q}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Full-rank covariance matrix decomposition

    ### CovViaDiagonal

    Let's start with the simplest case, i.e., a diagonal matrix, to explore the API:
    """)
    return


@app.cell
def _(covmats, np):
    d3 = [1, 2, 3]
    A33 = np.diag(d3)  # a diagonal covariance matrix
    cov_diag_33 = covmats.CovViaDiagonal(d3)
    # a point of interest
    v3 = [4, -2, 5]
    # repreat 4 times to obtain an array
    V34 = np.vstack([v3] * 4).T
    return A33, V34, cov_diag_33, v3


@app.cell
def _(mo):
    mo.md(r"""
    It behaves as a linear operator:
    """)
    return


@app.cell
def _(A33, V34, cov_diag_33, np, v3):
    # Test the equality between a dense array and the diagonal cov representation
    np.testing.assert_allclose(A33 @ v3, cov_diag_33 @ v3)
    np.testing.assert_allclose(A33 @ V34, cov_diag_33 @ V34)
    cov_diag_33 @ V34
    return


@app.cell
def _(mo):
    mo.md(r"""
    It is possible to get the inverse (get $\mathbf{Q}$), and solve
    $\mathbf{Ax} = \mathbf{b}$:
    """)
    return


@app.cell
def _(A33, V34, cov_diag_33, np):
    np.testing.assert_allclose(np.linalg.inv(A33), cov_diag_33.precision)
    np.testing.assert_allclose(cov_diag_33.solve(cov_diag_33 @ V34), V34)
    return


@app.cell
def _(A33, sp, v3):
    dist = sp.stats.multivariate_normal(mean=[0, 0, 0], cov=A33)
    dist.pdf(v3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    TODO: continue the tutorial
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ⚠️ About the following code that performs sparse cholesky factorization

    [Scikit-sparse](https://github.com/scikit-sparse/scikit-sparse) depends on external
    libraries with GPL licenses, such as
    [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse?tab=License-1-ov-file).
    As a consequence the following piece of code must adopt that license as well. Please
    look into the terms of this license before creating a dynamic link to this pieces in
    your downstream package and understand commercial use limitations. We are not
    lawyers and cannot provide any guidance on the terms of this license.

    Please see https://www.gnu.org/licenses/licenses.html#LGPL
    """)
    return


@app.cell
def _(sp):
    from typing import no_type_check

    @no_type_check  # for ty checks
    def _get_L_D_P(A: sp.sparse.sparray):
        """
        Return L, D and P from the factorization L @ D @ L' = P @ A @ P' using sksparse.

        Note that sksparse uses SuiteSparse which is LGPL licence.
        """
        import sksparse.cholmod as cholmod

        # Need to take the API change into account
        try:
            # sksparse 4.x
            L, D, P = cholmod.ldl(A, order="amd")
        except AttributeError:
            # sksparse 5.x
            f = cholmod.cholesky(A)
            (L, D), P = f.L_D(), f.P()
        return L, D, P

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Low-rank covariance matrix decomposition
    """)
    return


if __name__ == "__main__":
    app.run()
