import marimo

__generated_with = "0.20.4"

app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Priors and drift matrices
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

    return covmats, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    In geostatistical regularization (e.g. universal kriging, or regularized
    inversion with an uncertain trend), the field being estimated is often
    modeled as the sum of a random, zero-mean part -- described by one of the
    `CovarianceMatrix` representations covered in the companion
    `examples_covariances` notebook -- and a deterministic *prior* (or *drift*,
    or *trend*) term:

    $$\mathbf{z} = \mathbf{m}(\boldsymbol{\beta}) + \mathbf{\epsilon}, \quad
    \mathbf{\epsilon} \sim \mathcal{N}(0, \mathbf{A})$$

    `covmats` provides a small hierarchy of `PriorTerm` classes to describe the
    mean function $\mathbf{m}$:

    - [PriorTerm](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.PriorTerm.html#covmats.PriorTerm)
     --
      the abstract base class.
    - [NullPriorTerm](
    https://covmats.readthedocs.io/en/latest/_autosummary/covmats.NullPriorTerm.html#covmats.NullPriorTerm)
     --
      no prior at all (a purely zero-mean field).
    - [ConstantPriorTerm](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.ConstantPriorTerm.html#covmats.ConstantPriorTerm)
     --
      a fixed, known set of values.
    - [MeanPriorTerm](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.MeanPriorTerm.html#covmats.MeanPriorTerm)
     --
      the (running) mean of the current parameter values.
    - [EnsembleMeanPriorTerm](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.EnsembleMeanPriorTerm.html#covmats.EnsembleMeanPriorTerm)
     --
      the per-point mean across ensemble members.
    - [DriftMatrix](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.DriftMatrix.html#covmats.DriftMatrix)
      and its specializations
      [ConstantDriftMatrix](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.ConstantDriftMatrix.html#covmats.ConstantDriftMatrix)
      and
      [LinearDriftMatrix](https://covmats.readthedocs.io/en/latest/_autosummary/covmats.LinearDriftMatrix.html#covmats.LinearDriftMatrix)
       --
      a mean expressed as a linear combination
      $\mathbf{m} = \mathbf{X}\boldsymbol{\beta}$
      of known basis functions (drift terms) and unknown coefficients.

    Every `PriorTerm` shares two methods:

    - `get_values(params)`, returning the current value of the prior term.
    - `get_gradient_dot_product(input)`, returning the dot product of the
      prior's gradient with a given vector -- useful when the prior term is
      plugged into a gradient-based inversion scheme.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## NullPriorTerm

    The simplest prior: it always contributes zero, regardless of the
    parameters. It is useful as a default/placeholder when no informative
    prior is available.
    """)
    return


@app.cell
def _(covmats, np):
    null_prior = covmats.NullPriorTerm()
    params_null = np.array([1.0, 2.0, 3.0])
    null_prior.get_values(params_null)
    return (null_prior,)


@app.cell
def _(null_prior, params_null):
    # The gradient contribution is also zero, with the shape of the input
    null_prior.get_gradient_dot_product(params_null)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ConstantPriorTerm

    Represents a fixed, known prior (e.g. values coming from an external,
    trusted source), with no dependency on the current parameter values --
    only their shape is checked for consistency.
    """)
    return


@app.cell
def _(covmats, np):
    prior_values = np.array([1.0, 2.0, 3.0])
    const_prior = covmats.ConstantPriorTerm(prior_values)
    params_const = np.array([10.0, 20.0, 30.0])  # only the shape matters here
    const_prior.get_values(params_const)
    return (const_prior,)


@app.cell
def _(const_prior, params_const):
    # Since the prior does not depend on params, its gradient contribution is zero
    const_prior.get_gradient_dot_product(params_const)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## MeanPriorTerm

    Represents the prior as the (running) mean of the parameters themselves --
    every point is pulled towards the overall average value.
    """)
    return


@app.cell
def _(covmats, np):
    mean_prior = covmats.MeanPriorTerm()
    params_mean = np.array([1.0, 2.0, 3.0, 4.0])
    mean_prior.get_values(params_mean)  # the constant mean value, broadcast
    return (mean_prior,)


@app.cell
def _(mean_prior, params_mean):
    mean_prior.get_gradient_dot_product(params_mean)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## EnsembleMeanPriorTerm

    The ensemble counterpart of `MeanPriorTerm`: given an `(Ns, Ne)` matrix of
    parameter values across an ensemble of `Ne` members, it returns the
    per-point mean across members -- as used e.g. in ensemble-based
    geostatistical inversion.
    """)
    return


@app.cell
def _(covmats, np):
    n_s_ens, n_e_ens = 3, 4
    ens_prior = covmats.EnsembleMeanPriorTerm(shape=(n_s_ens, n_e_ens))
    params_ens = np.arange(n_s_ens * n_e_ens, dtype=float).reshape(n_s_ens, n_e_ens)
    ens_prior.get_values(params_ens)  # shape (Ns, 1): the per-point ensemble mean
    return (ens_prior,)


@app.cell
def _(ens_prior, n_s_ens, np):
    ens_prior.get_gradient_dot_product(np.ones(n_s_ens))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## DriftMatrix

    `DriftMatrix` represents the prior as a linear combination of known basis
    (drift) functions:

    $$\mathbf{m} = \mathbf{X}\boldsymbol{\beta}$$

    with $\mathbf{X}$ (`mat`) a `(Ns, Nbeta)` matrix of basis functions and
    $\boldsymbol{\beta}$ (`beta`) the `(Nbeta,)` vector of coefficients. This
    is the building block behind universal kriging. `covmats` ships two ready
    made specializations, `ConstantDriftMatrix` and `LinearDriftMatrix`, but a
    custom basis (e.g. polynomial, or problem-specific) can be passed directly
    to `DriftMatrix`.
    """)
    return


@app.cell
def _(covmats, np):
    pts_priors = covmats.get_pts_coords_regular_grid(mesh_dim=1.0, shape=(3, 3))
    # A custom quadratic drift basis: [1, x^2]
    X_custom = np.column_stack([np.ones(pts_priors.shape[0]), pts_priors[:, 0] ** 2])
    drift_custom = covmats.DriftMatrix(X_custom, beta=np.array([1.0, 0.1]))
    drift_custom.get_values(np.zeros(pts_priors.shape[0]))
    return (pts_priors,)


@app.cell
def _(mo):
    mo.md(r"""
    ### ConstantDriftMatrix

    A constant (order 0) drift: the basis is a single, normalized column of
    ones, so $\mathbf{X}\boldsymbol{\beta}$ represents a spatially uniform
    mean -- this is exactly the setting of ordinary/simple kriging with an
    unknown constant mean.
    """)
    return


@app.cell
def _(covmats, pts_priors):
    const_drift = covmats.ConstantDriftMatrix(n_pts=pts_priors.shape[0])
    const_drift.mat.ravel()  # a single, normalized column
    return (const_drift,)


@app.cell
def _(const_drift, np, pts_priors):
    const_drift.beta = np.array([2.0])
    const_drift.get_values(np.zeros(pts_priors.shape[0]))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### LinearDriftMatrix

    A linear (order 1) drift: the basis is a constant column followed by the
    raw point coordinates, so the mean varies linearly across space -- the
    setting of universal kriging with a linear trend.
    """)
    return


@app.cell
def _(covmats, pts_priors):
    lin_drift = covmats.LinearDriftMatrix(pts_priors)
    lin_drift.mat.shape  # (Npts, 1 + Ndim)
    return (lin_drift,)


@app.cell
def _(lin_drift, np, pts_priors):
    lin_drift.beta = np.array([1.0, 0.5, -0.5])
    lin_drift.get_values(np.zeros(pts_priors.shape[0]))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going further

    Because every `PriorTerm` shares the same `get_values`/
    `get_gradient_dot_product` interface, they can be swapped freely in a
    downstream inversion or kriging routine -- for instance starting with a
    `NullPriorTerm` while prototyping, then moving to a `LinearDriftMatrix`
    once a spatial trend is identified in the data, without further code
    changes. See the companion `examples_covariances` notebook for the
    covariance-matrix representations these priors are typically combined
    with.
    """)
    return


if __name__ == "__main__":
    app.run()
