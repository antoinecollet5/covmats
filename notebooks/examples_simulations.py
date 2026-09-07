import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Using covariance matrices to perform unconditional and conditional simulations
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
    import matplotlib.pyplot as plt
    import numpy as np

    return covmats, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following is a toy example to illustrate how to use `covmats` to perform
    unconditional and conditional simulations. Let's use an example provided by
    `covmats`. Here, the prior covariance matrix, $\mathbf{C}_{\mathrm{prior}}$ is
    represented as a sparse factorization of its inverse
    $\mathbf{C}_{\mathrm{prior}}^{-1}$ with
    $\mathbf{LDL}^{\mathrm{T}} =$
    $ \mathbf{PC}_{\mathrm{prior}}^{-1}\mathbf{P}^{\mathrm{T}}$.
    This is wrapped in the `covmats.CovViaSparsePrecisionCholesky` instance we create:
    """)
    return


@app.cell
def _(covmats):
    cov = covmats.CovViaSparsePrecisionCholesky(
        covmats.load_precision_example_4225x_SCF()
    )
    cov
    return (cov,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The covariance matrix has shape (4225, 4225), let's define a square domain
    (65, 65) and perform a non conditional simulation using our prior. We set a
    mean @ 50 and display 16 independent realizations:
    """)
    return


@app.cell
def _(cov, np):
    # Domain dimensions
    nx = ny = int(np.sqrt(cov.shape[0]))
    mean = 50.0

    # Non conditional simulation -> change the random state (seed) to obtain
    # different fields.
    n_uc_reals = 16
    z_uc = cov.sample_mvnormal(shape=(n_uc_reals,), random_state=2026) + mean

    # Reference field used for the rest of the tutorial: the "true", unknown
    # field we will pretend to sample noisy point observations from below.
    z_ref = z_uc[0]
    return mean, nx, ny, z_ref, z_uc


@app.cell
def _(np, nx, ny, plt, z_uc):
    def field_to_image(z_flat):
        """Reshape a flat (nx*ny,) field into a (ny, nx) image for imshow."""
        return z_flat.reshape(nx, ny, order="F").T

    def idx_to_xy(idx):
        """Map flat field indices to (x, y) image coordinates for scatter/imshow."""
        idx = np.asarray(idx)
        return idx % nx, idx // nx

    nrows_uc = ncols_uc = 4
    fig_uc, axes_uc = plt.subplots(
        nrows_uc,
        ncols_uc,
        figsize=(9.0, 9.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    vmin_uc, vmax_uc = np.min(z_uc), np.max(z_uc)
    for i_uc, ax_uc in enumerate(axes_uc.ravel()):
        im_uc = ax_uc.imshow(
            field_to_image(z_uc[i_uc]),
            origin="lower",
            cmap=plt.get_cmap("jet"),
            aspect="equal",
            vmin=vmin_uc,
            vmax=vmax_uc,
        )
        ax_uc.set_title(f"Realization #{i_uc}", fontsize=9, fontweight="bold")
        ax_uc.set_xticks([])
        ax_uc.set_yticks([])
    fig_uc.colorbar(im_uc, ax=axes_uc, shrink=0.6, pad=0.01, label="Parameter value")
    fig_uc.suptitle(
        "16 unconditional realizations of the prior field", fontweight="bold"
    )
    return field_to_image, idx_to_xy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conditional simulation

    Now let's pretend the first unconditional realization above (`z_ref`) is
    the "true", unknown field, and that we only get to observe it at a handful
    of locations, with some measurement uncertainty. We sample a random subset
    of grid points from `z_ref`, add independent Gaussian noise to mimic real
    measurement error, and use `covmats.conditional_simulate` /
    `covmats.conditional_mean` (Matheron's rule / pathwise conditioning) to
    produce realizations and the kriging mean that honor those noisy
    observations while remaining consistent with the prior everywhere else.
    """)
    return


@app.cell
def _(cov, covmats, mean, np, z_ref):
    rng = np.random.default_rng(0)
    sigma_obs = 0.5  # known measurement uncertainty (std) at every sampled point

    n_obs = 150
    obs_idx = rng.choice(cov.shape[0], size=n_obs, replace=False)
    obs_values = z_ref[obs_idx] + sigma_obs * rng.standard_normal(n_obs)

    H = covmats.make_point_observation_operator(obs_idx, cov.shape[0])

    # `solver="direct"` is a great fit here: n_obs (150) is tiny compared to
    # the field size (4225), so building & factoring the (n_obs, n_obs) system
    # once and solving every realization in a single vectorized call is both
    # exact and far cheaper than a CG solve per realization.
    n_cond_reals = 4
    z_cond = covmats.conditional_simulate(
        cov,
        H,
        obs_values,
        sigma_obs**2,
        n_reals=n_cond_reals,
        mean=mean,
        random_state=1,
        solver="direct",
    )
    z_cond_mean = covmats.conditional_mean(
        cov, H, obs_values, sigma_obs**2, mean=mean, solver="direct"
    )
    return obs_idx, sigma_obs, z_cond, z_cond_mean


@app.cell
def _(field_to_image, idx_to_xy, np, obs_idx, plt, z_cond, z_cond_mean, z_ref):
    obs_x, obs_y = idx_to_xy(obs_idx)

    assert not isinstance(z_cond, tuple)  # for ty

    fig_cond, axes_cond = plt.subplots(
        3,
        2,
        figsize=(4, 7),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    vmin_cond = min(z_ref.min(), np.min(z_cond), np.mean(z_cond_mean))
    vmax_cond = max(z_ref.max(), np.max(z_cond), np.max(z_cond_mean))

    panels_cond = [("Reference (truth)", z_ref)]
    panels_cond += [("Conditional mean", z_cond_mean)]
    panels_cond += [
        (f"Conditional Real #{i}", z_cond[i]) for i in range(z_cond.shape[0])
    ]

    for ax_cond, (title_cond, field_cond) in zip(axes_cond.ravel(), panels_cond):
        ax_cond.imshow(
            field_to_image(field_cond),
            origin="lower",
            cmap=plt.get_cmap("jet"),
            aspect="equal",
            vmin=vmin_cond,
            vmax=vmax_cond,
        )
        ax_cond.scatter(
            obs_x, obs_y, s=6, facecolors="none", edgecolors="k", linewidths=0.5
        )
        ax_cond.set_title(title_cond, fontsize=9, fontweight="bold")
        ax_cond.set_xticks([])
        ax_cond.set_yticks([])

    fig_cond.suptitle(
        f"Conditioning on {len(obs_idx)} noisy point\n observations "
        "(black circles):\n realizations honor the data,\n the mean smooths it out",
        fontweight="bold",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The more points sampled, the closer to the reference

    Let's now grow the number of conditioning points (keeping every smaller
    set nested inside the larger ones, so this is a genuine "more information"
    comparison) and look at both the resulting kriging mean field and the RMSE
    against the true reference field.
    """)
    return


@app.cell
def _(cov, covmats, mean, np, sigma_obs, z_ref):
    rng_conv = np.random.default_rng(0)
    all_idx = rng_conv.permutation(cov.shape[0])

    n_obs_list = [5, 20, 80, 300, 1000, 3000]
    conv_means = []
    conv_rmses = []
    for n_obs_i in n_obs_list:
        idx_i = all_idx[:n_obs_i]
        d_i = z_ref[idx_i] + sigma_obs * rng_conv.standard_normal(n_obs_i)
        H_i = covmats.make_point_observation_operator(idx_i, cov.shape[0])
        m_i = covmats.conditional_mean(
            cov, H_i, d_i, sigma_obs**2, mean=mean, solver="direct"
        )
        conv_means.append(m_i)
        conv_rmses.append(float(np.sqrt(np.mean((m_i - z_ref) ** 2))))
    return conv_means, conv_rmses, n_obs_list


@app.cell
def _(conv_means, field_to_image, n_obs_list, plt, z_ref):
    fig_conv, axes_conv = plt.subplots(
        4,
        2,  # int(np.ceil((len(n_obs_list) + 1) / 2)),
        figsize=(4.0, 8.5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    vmin_conv, vmax_conv = z_ref.min(), z_ref.max()

    ax = axes_conv.ravel()[0]
    ax.imshow(
        field_to_image(z_ref),
        origin="lower",
        cmap=plt.get_cmap("jet"),
        aspect="equal",
        vmin=vmin_conv,
        vmax=vmax_conv,
    )
    ax.set_title("Reference (truth)", fontsize=9, fontweight="bold")

    ax = axes_conv.ravel()[1]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    for ax_conv, n_obs_j, m_j in zip(axes_conv.ravel()[2:], n_obs_list, conv_means):
        ax_conv.imshow(
            field_to_image(m_j),
            origin="lower",
            cmap=plt.get_cmap("jet"),
            aspect="equal",
            vmin=vmin_conv,
            vmax=vmax_conv,
        )
        ax_conv.set_title(f"Mean, n_obs={n_obs_j}", fontsize=9, fontweight="bold")

    for ax_conv in axes_conv.ravel():
        ax_conv.set_xticks([])
        ax_conv.set_yticks([])

    fig_conv.suptitle(
        "Conditional mean field as the number\n of observations grows",
        fontweight="bold",
    )
    return


@app.cell
def _(conv_rmses, n_obs_list, plt):
    fig_rmse, ax_rmse = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    ax_rmse.plot(n_obs_list, conv_rmses, marker="o")
    ax_rmse.set_xscale("log")
    ax_rmse.set_xlabel("Number of conditioning points (log scale)", fontweight="bold")
    ax_rmse.set_ylabel("RMSE(conditional mean, reference)", fontweight="bold")
    ax_rmse.grid(True, which="both", alpha=0.3)
    ax_rmse.set_title(
        "More data -> the conditional mean converges to the truth", fontweight="bold"
    )
    return


if __name__ == "__main__":
    app.run()
