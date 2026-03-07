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


if __name__ == "__main__":
    app.run()
