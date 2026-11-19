==============
Changelog
==============


0.6.0 (2026-11-19)
------------------

Bug fixes
~~~~~~~~~

* FIX: `SparseCholeskyFactor.get_diagonal` ignored the permutation `P` and returned
  the diagonal of :math:`PAP^T` instead of the diagonal of :math:`A` whenever
  :math:`P` is not the identity. It is now also computed in O(nnz(L)) without
  forming :math:`L D L^T`.
* FIX: `SparseCholeskyFactor.whiten_inv` applied the transposition at the wrong place:
  it raised a shape error for batches with a number of rows different from `n`, and
  silently returned a wrong result when the batch was square.
* FIX: `SparseCholeskyFactor.log_pdet` computed `log(prod(diag(D)))`, which overflows
  (`inf`) or underflows (`-inf`) for large matrices. It is now `sum(log(diag(D)))`.
* FIX: `SparseCholeskyFactor` cached properties (`invD`, `sqrtD`, `sqrtinvD`, `Pt`)
  were never invalidated when `D` or `P` were set again, contrary to what the
  documentation says. Setting `L`, `D` or `P` now resets everything derived from it.
* FIX: setting an invalid `L` (not lower triangular, or without unit diagonal) used to
  leave the instance with the invalid value. The value is now validated before being
  assigned.
* FIX: `SparseCholeskyFactor` now checks that `P` is a permutation of `range(n)`.
  The shape error message mentioned `L` twice instead of `P`.
* FIX: `assert_allclose_sparse` relied on a bare `assert` for the shape check, which
  is stripped by `python -O`.
* FIX: `get_SPD_sparse_example` ignored its `seed` argument (a hard-coded seed of 42
  was used) and reset the global numpy random state. It now uses a local generator
  seeded with `seed`. **Matrices generated for a given seed differ from previous
  versions.**
* FIX: `CovViaEnsemble` kept stale `anomalies` and `n_ens` when the `ensemble` was
  assigned a second time, and always reported `rank == 0` and `log_pdet == 0.0`.
  They are now consistent with `CovViaEigenFactorization` (rank is the number of
  strictly positive eigenvalues and `log_pdet` the log of their product).
* FIX: `get_eigen_factorization_gram_matrix_trick` truncation tolerance was at the
  level of the numerical noise: the null space of centered anomalies (rank
  :math:`\leq N_e - 1`) could survive as a spurious mode, and the associated
  eigenvector was obtained by dividing noise by a near-zero singular value.
  The tolerance now keeps a safety factor of 10.
* FIX: `CovKernelAsLinop.solve` did not reset the GMRES iteration counter between
  right-hand sides, so `itercount` over-counted for multiple columns. It also
  truncated `b` when given as integers, and now emits a `RuntimeWarning` when GMRES
  does not converge instead of failing silently.
* FIX: the sparse preconditioner of `CovKernelAsLinop` was built from the raw point
  coordinates whereas the kernel is evaluated on coordinates scaled by `len_scale`
  everywhere else. It is now built from the scaled coordinates.
* FIX: `CovKernelAsLinop.todense` failed for a scalar `len_scale` or integer
  coordinates, although both are documented as supported.
* FIX: `toeplitz_product` failed for a 1D grid with a single cell.

Behaviour changes
~~~~~~~~~~~~~~~~~

* FIX (**breaking**): the `nugget` of `CovKernelAsLinop` and `CovKernelAsLinopViaFFT`
  is now consistently an *absolute variance added to the diagonal*:
  :math:`C = K + \tau^2 I`, as documented. Previously `todense`, `get_diagonal` and
  the preconditioner ignored it, and the FFT matrix-vector product returned
  :math:`(1 + \tau^2) K` (a rescaling, which neither regularizes nor models a
  nugget effect), so that `todense`, `@` and `solve` did not describe the same
  matrix. Results obtained with a non-zero nugget differ from previous versions.
* ENH: the nugget is handled once, in `CovKernelAsLinop`. A new matrix type only has to
  override `_kernel_matvec` (:math:`Kx`, without nugget) and, optionally,
  `_kernel_todense`. `todense`, `get_diagonal`, `solve`, the preconditioner and the
  `count` matrix-vector counter (now also incremented by the FFT operator) are
  inherited.
* ENH: `nugget` must be finite and non-negative (`ValueError` otherwise) and is
  exposed as the read-only `nugget` property. The smallest eigenvalue of
  :math:`C` is at least :math:`\tau^2`.

Performance
~~~~~~~~~~~

* ENH: `CovKernelAsLinopViaFFT` computes the FFT of the circulant embedding once at
  creation (new `toeplitz_circulant_spectrum` / `toeplitz_product_from_spectrum`
  helpers) and uses real FFTs: matrix-vector products are about 2.7 times faster
  (256 x 256 grid: 14 ms -> 5 ms, 48 x 48 x 48 grid: 101 ms -> 39 ms).
* ENH: `SparseCholeskyFactor.get_invdiagonal` uses blocked multi right-hand side
  forward solves (a single triangular solve per unit vector instead of a full
  `solve`), about 6 times faster for n = 1500. The new `block_size` argument bounds
  the memory.
* ENH: `SparseCholeskyFactor.get_diagonal` is about 35 times faster (10 000 x 10 000
  Laplacian: 18.6 ms -> 0.5 ms).
* ENH: `SparseCholeskyFactor` keeps the CSR copies of `L` and `L^T` required by
  `scipy.sparse.linalg.spsolve_triangular` and the diagonal of `D` (and its inverse and
  square roots) as vectors, instead of converting and rebuilding sparse diagonal
  matrices at each call.
* ENH: `CovViaEigenFactorization.get_sparse_LLT_factor` is built from a COO array
  rather than a resized LIL array (about 4 times faster).

New features
~~~~~~~~~~~~

* ENH: add `sparse_cholesky_superlu`, a sparse
  :math:`LDL^T = PAP^T` factorization of a symmetric positive definite matrix relying on
  SuperLU, shipped with SciPy (BSD): it provides a `SparseCholeskyFactor` without
  the LGPL `scikit-sparse` / SuiteSparse dependency. Fill-reducing orderings
  (`"amd"`, `"rcm"`, `"natural"` or a user permutation) can be selected, and non
  positive definite matrices are rejected.

Tests
~~~~~

* TEST: the test suite no longer requires `scikit-sparse`: when it is missing,
  the factorizations are obtained with `sparse_cholesky_superlu`.
* TEST: add `tests/test_v060_regressions.py` covering all the fixes above.
* TEST: seed the random generator of `test_CovViaPrecisionCholesky`, which failed
  randomly (about once in 5 000 runs) for ill-conditioned random precision matrices.

0.5.0 (2026-09-08)
------------------

* ENH: add `colorize_adjoint` methods to all CovarianceMatrix derived classes.

0.4.0 (2026-09-07)
------------------

* ENH: add conditional Gaussian simulation on top of any `CovarianceMatrix` (or
  matrix-free operator exposing `matvec`/`colorize`): `covmats.conditional_simulate`
  and `covmats.conditional_mean` implement Matheron's rule / pathwise conditioning,
  producing exact realizations and the kriging mean from noisy point (or general
  linear) observations with per-observation or correlated uncertainty, without ever
  forming or factorizing the posterior covariance. `covmats.make_point_observation_operator`
  builds the observation operator for the common case of point data at
  discretization nodes.
* ENH: `conditional_simulate`/`conditional_mean` accept a `solver` argument.
  `solver="cg"` (default) is fully matrix-free and scales to arbitrarily large
  numbers of observations. `solver="direct"` assembles and Cholesky-factors the
  dense `(n_obs, n_obs)` observation-space system once, then solves every
  realization in a single vectorized call, removing the per-realization Python
  loop entirely; typically much faster whenever the number of observations is
  small to moderate compared to the field size.
* DOC: added the `conditional_simulation_tutorial` notebook, demonstrating
  unconditional simulation, conditioning on noisy point observations, and the
  convergence of the conditional mean to the true field as more observations
  are added.

0.3.2 (2026-08-10)
------------------

* FIX: implement `_PickleSafeLinearOperator`, a `LinearOperator`` subclass whose pickling
  correctly preserves state stored in `__slots__`, on every scipy/Python version.

0.3.1 (2026-08-09)
------------------

* FIX: use cached_property instead of precomputation in setters to be safer when using multiprocessing

0.3.0 (2026-08-08)
------------------
* ENH: expose `get_pts_coords_regular_grid()` in the public API (`covmats.get_pts_coords_regular_grid`);
  it was previously only used internally by `CovKernelAsLinopViaFFT` and unreachable by users.
* ENH: `CovViaPrecisionCholesky.get_diagonal()` and the kernel preconditioner construction
  (`CovKernelAsLinop(..., is_use_preconditioner=True)`) are now noticeably faster
  (blocked triangular solves and direct LAPACK calls instead of a per-column/per-point
  Python loop), with a tunable `block_size` on `get_diagonal` to bound the extra memory used.
* FIX: `CovKernelAsLinop` did not implement `_matvec`; any matrix-vector product
  (and therefore `CovKernelAsLinop.solve`) would recurse indefinitely. It is now backed
  by a cached dense evaluation of the kernel.
* FIX: `CovViaCholesky.L` and `CovViaPrecisionCholesky.L` setters called an undefined
  `_validate_matrix` method, raising an `AttributeError` on assignment.
* FIX: `get_pts_coords_regular_grid()` raised an `IndexError` when given a scalar
  `mesh_dim` together with a grid `shape` of more than one dimension; it now
  broadcasts the scalar across all dimensions as documented.
* FIX: `CovKernelAsLinopViaFFT` cast `domain_shape` to `np.int8`, silently corrupting
  grids with more than 127 cells along an axis; it now uses `np.int64`.
* DOC: filled in numerous placeholder/incomplete docstrings and fixed several broken or
  non-reproducible doctests across `covmats._covariances`, `covmats._sparse_helpers`,
  and `covmats._priors`.
* DOC: completed the `examples_covariances` and `examples_priors` tutorial notebooks
  to cover the full public API, and expanded the README quick start accordingly.
* TEST: added unit tests for `get_pts_coords_regular_grid()`.

0.2.2 (2026-02-10)
------------------

* FIX: make sure tests data are included in distributions (second fix).

0.2.1 (2026-02-10)
------------------

* FIX: make sure tests data are included in distributions.

0.2.0 (2026-02-08)
------------------

* ENH: add `load_precision_example_4225x_SCF()`.

0.1.0 (2026-02-07)
------------------

* First release on PyPI.
