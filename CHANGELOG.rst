==============
Changelog
==============

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
