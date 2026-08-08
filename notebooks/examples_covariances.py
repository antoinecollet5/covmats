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
def _(mo):
    mo.md(r"""
    Being a `scipy.stats.Covariance` subclass, it also plugs directly into
    `scipy.stats` distributions, e.g. `scipy.stats.multivariate_normal`:
    """)
    return


@app.cell
def _(A33, sp, v3):
    dist = sp.stats.multivariate_normal(mean=[0, 0, 0], cov=A33)
    dist.pdf(v3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Finally, all representations share a common sampling, whitening and colorizing
    API through :py:meth:`CovarianceMatrix.sample_mvnormal`,
    :py:meth:`CovarianceMatrix.whiten` and
    :py:meth:`CovarianceMatrix.colorize`:

    - `sample_mvnormal` draws samples from $\mathcal{N}(0, \mathbf{A})$.
    - `whiten(x)` maps correlated draws $\mathbf{x} \sim \mathcal{N}(0, \mathbf{A})$
      to (approximately) uncorrelated, unit-variance draws.
    - `colorize(x)` is the inverse operation: it maps independent standard normal
      draws to draws from $\mathcal{N}(0, \mathbf{A})$.
    """)
    return


@app.cell
def _(cov_diag_33, np):
    samples = cov_diag_33.sample_mvnormal(shape=[10000], random_state=0)
    # The empirical covariance should be close to A33
    np.round(np.cov(samples, rowvar=False), 1)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### CovViaCholesky

    `CovViaCholesky` represents a dense covariance matrix through the (dense)
    lower Cholesky factor $\mathbf{L}$ such that $\mathbf{A} = \mathbf{L}\mathbf{L}^T$.
    This is the natural representation to use once a covariance matrix has already
    been Cholesky-factorized, e.g. for fast sampling or fast solves through
    forward/backward substitution.
    """)
    return


@app.cell
def _(covmats, np):
    rng_cho = np.random.default_rng(2026)
    n_cho = 4
    B_cho = rng_cho.random((n_cho, n_cho))
    A_cho = B_cho @ B_cho.T + n_cho * np.eye(n_cho)  # a random SPD matrix
    L_cho = np.linalg.cholesky(A_cho)
    cov_cho = covmats.CovViaCholesky(L_cho)
    return A_cho, cov_cho, rng_cho


@app.cell
def _(mo):
    mo.md(r"""
    As with `CovViaDiagonal`, we can check that the dense representation, the
    solve and the log-determinant are all consistent with plain numpy/scipy:
    """)
    return


@app.cell
def _(A_cho, cov_cho, np, rng_cho):
    b_cho = rng_cho.random(cov_cho.shape[0])
    np.testing.assert_allclose(cov_cho.todense(), A_cho)
    np.testing.assert_allclose(cov_cho.solve(b_cho), np.linalg.solve(A_cho, b_cho))
    np.testing.assert_allclose(cov_cho.log_pdet, np.linalg.slogdet(A_cho)[-1])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### CovViaSparseCholesky

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
    To keep this tutorial free of GPL-licensed dependencies, we instead build the
    $\mathbf{L}\mathbf{D}\mathbf{L}^T$ factorization with `scipy.linalg.ldl`
    (dense, but fine for a small illustrative example) and wrap it into a
    :py:class:`SparseCholeskyFactor`, which `CovViaSparseCholesky` and
    `CovViaSparsePrecisionCholesky` both build upon. In practice, for a sparse
    precision/covariance matrix, you would instead obtain `L`, `D` and `P` from a
    real sparse solver (e.g. `sksparse.cholmod`, subject to the license notice
    above).
    """)
    return


@app.cell
def _(covmats, sp):
    Q_sp = covmats.get_SPD_sparse_example(n=8, seed=2026)  # a sparse SPD matrix
    L_ldl, D_ldl, P_ldl = sp.linalg.ldl(Q_sp.toarray())
    scf_sp = covmats.SparseCholeskyFactor(
        sp.sparse.csc_array(L_ldl), sp.sparse.csc_array(D_ldl), P_ldl
    )
    return Q_sp, scf_sp


@app.cell
def _(mo):
    mo.md(r"""
    Here `Q_sp` plays the role of a covariance matrix (not a precision matrix):
    `CovViaSparseCholesky` factorizes the covariance matrix itself, as opposed to
    `CovViaSparsePrecisionCholesky` below, which factorizes its inverse.
    """)
    return


@app.cell
def _(Q_sp, covmats, np, scf_sp):
    cov_sp = covmats.CovViaSparseCholesky(scf_sp)
    v_sp = np.random.default_rng(1).random(Q_sp.shape[0])
    np.testing.assert_allclose(cov_sp.todense(), Q_sp.toarray())
    np.testing.assert_allclose(
        cov_sp.solve(v_sp), np.linalg.solve(Q_sp.toarray(), v_sp)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### CovViaPrecisionCholesky

    `CovViaPrecisionCholesky` is the mirror image of `CovViaCholesky`: instead of
    factorizing the covariance matrix $\mathbf{A}$, it factorizes its inverse, the
    precision matrix $\mathbf{Q} = \mathbf{A}^{-1} = \mathbf{L}_Q\mathbf{L}_Q^T$.
    This is the natural representation for Gaussian Markov Random Fields, where the
    precision matrix is typically sparse (see `CovViaSparsePrecisionCholesky` below
    for the sparse counterpart) while the covariance matrix is dense.
    """)
    return


@app.cell
def _(covmats, np):
    rng_prec = np.random.default_rng(2026)
    n_prec = 4
    B_prec = rng_prec.random((n_prec, n_prec))
    Q_prec = B_prec @ B_prec.T + n_prec * np.eye(n_prec)  # a random SPD precision
    L_prec = np.linalg.cholesky(Q_prec)
    cov_prec = covmats.CovViaPrecisionCholesky(L_prec)
    return Q_prec, cov_prec, rng_prec


@app.cell
def _(Q_prec, cov_prec, np, rng_prec):
    b_prec = rng_prec.random(cov_prec.shape[0])
    np.testing.assert_allclose(cov_prec.todense(), np.linalg.inv(Q_prec))
    np.testing.assert_allclose(cov_prec.precision, Q_prec)
    # cov_prec.solve applies the precision matrix directly (no dense inversion)
    np.testing.assert_allclose(cov_prec.solve(b_prec), Q_prec @ b_prec)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### CovViaSparsePrecisionCholesky

    Combining the two ideas above, `CovViaSparsePrecisionCholesky` factorizes a
    *sparse* precision matrix. It is the representation of choice for large,
    sparse Gaussian Markov Random Fields (e.g. from an SPDE discretization),
    where the dense covariance matrix would be far too large to store.
    """)
    return


@app.cell
def _(covmats, sp):
    Qprec_sp = covmats.get_SPD_sparse_example(n=8, seed=2026)  # a sparse precision
    L_qldl, D_qldl, P_qldl = sp.linalg.ldl(Qprec_sp.toarray())
    scf_qprec = covmats.SparseCholeskyFactor(
        sp.sparse.csc_array(L_qldl), sp.sparse.csc_array(D_qldl), P_qldl
    )
    cov_sp_prec = covmats.CovViaSparsePrecisionCholesky(scf_qprec)
    return Qprec_sp, cov_sp_prec


@app.cell
def _(Qprec_sp, cov_sp_prec, np):
    v_sp_prec = np.random.default_rng(1).random(Qprec_sp.shape[0])
    np.testing.assert_allclose(cov_sp_prec.todense(), np.linalg.inv(Qprec_sp.toarray()))
    np.testing.assert_allclose(cov_sp_prec.solve(v_sp_prec), Qprec_sp @ v_sp_prec)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Low-rank covariance matrix decomposition

    When the number of points grows very large, even a sparse full-rank
    representation may not be tractable. `covmats` provides two low-rank
    (approximate) representations:

    - `CovViaEigenFactorization`, from a truncated eigen decomposition
      $\mathbf{A} \approx \mathbf{V}\mathbf{W}\mathbf{V}^T$.
    - `CovViaEnsemble`, from an ensemble of anomalies (as used e.g. in
      ensemble Kalman filtering), which never builds the dense covariance
      matrix at all.

    ### CovViaEigenFactorization

    Given eigenvalues and eigenvectors (e.g. only the leading few, for a
    genuine low-rank approximation), `CovViaEigenFactorization` reconstructs a
    covariance-matrix-like linear operator:
    """)
    return


@app.cell
def _(covmats, np):
    rng_eig = np.random.default_rng(2026)
    n_eig = 5
    B_eig = rng_eig.random((n_eig, n_eig))
    A_eig = B_eig @ B_eig.T + n_eig * np.eye(n_eig)
    w_eig, v_eig = np.linalg.eigh(A_eig)  # full decomposition here (n_pc = n_eig)
    cov_eig = covmats.CovViaEigenFactorization((w_eig, v_eig))
    return A_eig, cov_eig


@app.cell
def _(A_eig, cov_eig, np):
    np.testing.assert_allclose(cov_eig.todense(), A_eig)
    np.testing.assert_allclose(cov_eig.get_trace(), np.trace(A_eig))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### get_linop_eigen_factorization and eigen_factorize_cov_mat

    Rather than computing an eigen decomposition by hand, `covmats` provides
    helpers to build a *low-rank, randomized* eigen factorization directly from
    any `CovarianceMatrix` (or `CovKernelAsLinop`) instance:

    - `get_linop_eigen_factorization` returns the `(eigenvalues, eigenvectors)`
      pair.
    - `eigen_factorize_cov_mat` wraps the result directly into a
      `CovViaEigenFactorization`.
    """)
    return


@app.cell
def _(covmats, np):
    cov_diag_5 = covmats.CovViaDiagonal(np.array([5.0, 10.0, 15.0, 20.0, 25.0]))
    eigvals_lr, eigvects_lr = covmats.get_linop_eigen_factorization(
        cov_diag_5, size=cov_diag_5.shape[0], n_pc=3, random_state=0
    )
    # cov_lowrank = covmats.eigen_factorize_cov_mat(cov_diag_5, n_pc=3, random_state=0)
    return cov_diag_5, eigvals_lr


@app.cell
def _(mo):
    mo.md(r"""
    `get_explained_var` then tells us how much of the total variance is captured
    by each retained mode -- useful to pick a truncation rank `n_pc`:
    """)
    return


@app.cell
def _(cov_diag_5, covmats, eigvals_lr):
    covmats.get_explained_var(eigvals_lr, cov_mat=cov_diag_5)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### CovViaEnsemble

    `CovViaEnsemble` represents the (empirical) covariance of an ensemble of
    $N_e$ realizations directly from the $(N_e, N_s)$ anomalies matrix, without
    ever forming the dense $(N_s, N_s)$ covariance matrix. This is the
    representation typically used in ensemble-based data assimilation
    (e.g. the Ensemble Kalman Filter), where $N_e \ll N_s$.
    """)
    return


@app.cell
def _(covmats, np):
    rng_ens = np.random.default_rng(2026)
    n_s = 6
    n_e = 200
    ensemble = rng_ens.multivariate_normal(
        np.zeros(n_s), np.eye(n_s) * 2 + 0.3, size=n_e
    )
    cov_ens = covmats.CovViaEnsemble(ensemble)
    return cov_ens, n_s, rng_ens


@app.cell
def _(cov_ens, n_s, np, rng_ens):
    v_ens = rng_ens.random(n_s)
    A_ens = cov_ens.todense()  # the empirical ensemble covariance
    np.testing.assert_allclose(cov_ens @ v_ens, A_ens @ v_ens)
    np.round(A_ens, 1)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Kernel-based low-rank linear operators

    When the covariance is defined analytically through a stationary kernel
    (e.g. exponential, Gaussian, Matérn) evaluated on a point cloud, `covmats`
    can wrap it into a `LinearOperator` on the fly, with matrix-vector products
    evaluated directly from the kernel -- no covariance matrix needs to be
    provided up front. Only the linear-operator interface is available here
    (matrix-vector products and `solve`, via GMRES): there is no `sample_mvnormal`,
    `whiten` or `colorize` for these classes.

    First, `get_pts_coords_regular_grid` builds the coordinates of a regular grid,
    which is a convenient way to generate points for these examples:
    """)
    return


@app.cell
def _(covmats):
    pts_grid = covmats.get_pts_coords_regular_grid(mesh_dim=1.0, shape=(6, 6))
    pts_grid.shape
    return (pts_grid,)


@app.cell
def _(mo):
    mo.md(r"""
    ### CovKernelAsLinop

    `CovKernelAsLinop` evaluates the kernel densely, in $O(N_{pts}^2)$, the
    first time a matrix-vector product is requested, and then caches it. It
    is well suited to moderate numbers of points, or to point clouds that
    do not lie on a regular grid.
    """)
    return


@app.cell
def _(np):
    def exponential_kernel(d, sill=1.0):
        return sill * np.exp(-d)

    return (exponential_kernel,)


@app.cell
def _(covmats, exponential_kernel, np, pts_grid):
    cov_kernel = covmats.CovKernelAsLinop(
        pts_grid, exponential_kernel, len_scale=np.array([2.0, 2.0])
    )
    cov_kernel.shape
    return (cov_kernel,)


@app.cell
def _(cov_kernel, np):
    v_kernel = np.random.default_rng(0).random(cov_kernel.shape[0])
    A_kernel = cov_kernel.todense()
    np.testing.assert_allclose(cov_kernel @ v_kernel, A_kernel @ v_kernel)
    # Solving A x = b recovers the original vector
    x_kernel = cov_kernel.solve(A_kernel @ v_kernel)
    np.testing.assert_allclose(x_kernel, v_kernel, atol=1e-6)
    return (A_kernel,)


@app.cell
def _(mo):
    mo.md(r"""
    ### CovKernelAsLinopViaFFT

    When the points form a regular grid and the kernel is stationary, the
    resulting covariance matrix is block-Toeplitz with Toeplitz blocks (BTTB).
    `CovKernelAsLinopViaFFT` exploits this structure to evaluate matrix-vector
    products through FFTs, in $O(N_{pts}\log N_{pts})$ time and without ever
    forming (or even needing) the dense matrix -- a substantial improvement over
    `CovKernelAsLinop` for large regular grids.
    """)
    return


@app.cell
def _(covmats, exponential_kernel, np):
    cov_fft = covmats.CovKernelAsLinopViaFFT(
        exponential_kernel,
        mesh_dim=1.0,
        domain_shape=(6, 6),
        len_scale=np.array([2.0, 2.0]),
    )
    cov_fft.shape
    return (cov_fft,)


@app.cell
def _(A_kernel, cov_fft, np):
    v_fft = np.random.default_rng(0).random(cov_fft.shape[0])
    # Matches the dense, kernel-based representation on the same grid
    np.testing.assert_allclose(cov_fft @ v_fft, A_kernel @ v_fft, atol=1e-6)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going further

    All full-rank and low-rank `CovarianceMatrix` representations expose the same
    API (`todense`, `solve`, `precision`, `log_pdet`, `sample_mvnormal`, `whiten`,
    `colorize`, ...), so switching from one representation to another -- e.g. going
    from a dense `CovViaCholesky` prototype to a sparse
    `CovViaSparsePrecisionCholesky` for a production-scale problem -- requires no
    change to downstream code. See the
    [API reference](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.html)
    for the full list of classes and functions, and the companion `examples_priors`
    notebook for the prior/drift-matrix API used alongside covariance
    representations in geostatistical regularization.
    """)
    return


if __name__ == "__main__":
    app.run()
