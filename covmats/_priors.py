# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

from covmats._types import NDArrayFloat


class PriorTerm(ABC):
    """Represent a prior term for the geostatistical regularization."""

    @abstractmethod
    def get_values(self, params: NDArrayFloat) -> Union[float, NDArrayFloat]:
        """
        Return the values of the prior term.

        Parameters
        ----------
        params : NDArrayFloat
            Values of the parameters for which to compute the prior.

        Returns
        -------
        NDArrayFloat
            The prior term values.
        """

    @abstractmethod
    def get_gradient_dot_product(
        self, input: NDArrayFloat
    ) -> Union[float, NDArrayFloat]:
        """
        Return the dot product of the gradient of the prior and the given input vector.

        Parameters
        ----------
        params : NDArrayFloat
            Values with which to compute the prior gradient dot product.

        Returns
        -------
        NDArrayFloat
            Prior gradient-input vector dot product.
        """


class NullPriorTerm(PriorTerm):
    """Represent a null prior term."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()

    def get_values(self, params: NDArrayFloat) -> float:
        """
        Return the values of the prior term.

        Parameters
        ----------
        params : NDArrayFloat
            Values of the parameters for which to compute the prior. It has no effect
            with `NullPriorTerm`.

        Returns
        -------
        NDArrayFloat
            The prior term values.
        """
        return 0.0

    def get_gradient_dot_product(
        self, input: NDArrayFloat
    ) -> Union[float, NDArrayFloat]:
        """
        Return the dot product of the gradient of the prior and the given input vector.

        Parameters
        ----------
        params : NDArrayFloat
            Values with which to compute the prior gradient dot product.

        Returns
        -------
        NDArrayFloat
            Prior gradient-input vector dot product.
        """
        return np.zeros(input.shape)


class ConstantPriorTerm(PriorTerm):
    """Represent a prior (no influence of beta)."""

    def __init__(self, prior_values: NDArrayFloat) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        prior_values : NDArrayFloat
            Values given to the prior term.
        """
        super().__init__()
        self.prior_values: NDArrayFloat = prior_values.ravel("F")

    def get_values(self, params: NDArrayFloat) -> NDArrayFloat:
        """
        Return the values of the prior term.

        Parameters
        ----------
        params : NDArrayFloat
            Values of the parameters for which to compute the prior. It has no effect
            with `ConstantPriorTerm`.

        Returns
        -------
        NDArrayFloat
            The prior term values.
        """
        if params.shape != self.prior_values.shape:
            raise ValueError(
                f"The given values have shape {params.shape} while the constant prior "
                f"has been defined with shape {self.prior_values.shape}!"
            )
        return self.prior_values

    def get_gradient_dot_product(
        self, input: NDArrayFloat
    ) -> Union[float, NDArrayFloat]:
        """
        Return the dot product of the gradient of the prior and the given input vector.

        Parameters
        ----------
        params : NDArrayFloat
            Values with which to compute the prior gradient dot product.

        Returns
        -------
        NDArrayFloat
            Prior gradient-input vector dot product.
        """
        return np.zeros(input.shape)


class MeanPriorTerm(PriorTerm):
    """Represent a mean prior."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()

    def get_values(self, params: NDArrayFloat) -> NDArrayFloat:
        """
        Return the values of the prior term.

        Parameters
        ----------
        params : NDArrayFloat
            Values of the parameters for which to compute the prior mean.

        Returns
        -------
        NDArrayFloat
            The prior term values.
        """
        return np.full(params.size, fill_value=np.sum(params)) / params.size

    def get_gradient_dot_product(
        self, input: NDArrayFloat
    ) -> Union[float, NDArrayFloat]:
        """
        Return the dot product of the gradient of the prior and the given input vector.

        Parameters
        ----------
        params : NDArrayFloat
            Values with which to compute the prior gradient dot product.

        Returns
        -------
        NDArrayFloat
            Prior gradient-input vector dot product.
        """
        return np.full(input.size, fill_value=np.sum(input)) / input.size


class EnsembleMeanPriorTerm(PriorTerm):
    """Represent a mean prior."""

    def __init__(self, shape: Tuple[int, ...]) -> None:
        """Initialize the instance."""
        super().__init__()
        if len(shape) != 2:
            raise ValueError(
                "The shape of an EnsembleMeanPriorTerm should be (N_s, N_e)"
                " with N_s the number of adjuted values and N_e the number of"
                " members in the ensemble."
            )
        self.shape: Tuple[int, ...] = shape

    def get_values(self, params: NDArrayFloat) -> NDArrayFloat:
        """
        Return the values of the prior term.

        Parameters
        ----------
        params : NDArrayFloat
            Values of the parameters for which to compute the prior mean.

        Returns
        -------
        NDArrayFloat
            The prior term values.
        """
        if params.shape != self.shape:
            raise ValueError(f"Expected shape {self.shape}, got {params.shape}.")
        return np.mean(params, axis=1, keepdims=True)

    def get_gradient_dot_product(
        self, input: NDArrayFloat
    ) -> Union[float, NDArrayFloat]:
        """
        Return the dot product of the gradient of the prior and the given input vector.

        Parameters
        ----------
        params : NDArrayFloat
            Values with which to compute the prior gradient dot product.

        Returns
        -------
        NDArrayFloat
            Prior gradient-input vector dot product.
        """
        if input.shape[0] != self.shape[0]:
            raise ValueError(
                f"Expected a vector of size {self.shape[0]}, got {input.shape}."
            )
        return input / self.shape[1]


class DriftMatrix(PriorTerm):
    r"""
    Represent a drift (trend) matrix prior term.

    A drift matrix represents the prior mean of a spatial field as a linear
    combination of known basis functions (drift terms), such as a constant or
    a linear trend in the point coordinates:

    .. math::
        \mathbf{m} = \mathbf{X} \boldsymbol{\beta}

    with :math:`\mathbf{X}` (`mat`) the matrix of drift basis functions with
    shape :math:`(N_s, N_{\beta})`, and :math:`\boldsymbol{\beta}` (`beta`) the
    associated drift coefficients with shape :math:`(N_{\beta},)`. This is
    commonly used in universal kriging / geostatistical regularization to
    represent non-stationary trends on top of a stationary covariance model.

    See Also
    --------
    ConstantDriftMatrix : Constant (order 0) drift.
    LinearDriftMatrix : Linear (order 1) drift.
    """

    __slots__: List[str] = ["mat"]

    def __init__(
        self, mat: NDArrayFloat, beta: Optional[Union[NDArrayFloat, float]] = None
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        mat : NDArrayFloat
            Matrix of drift basis functions (coefficients) :math:`\\mathbf{X}`
            with shape (Ns, Nbeta), Ns being the number of points and Nbeta the
            number of drift terms (e.g., 1 for a constant drift, 1 + Ndim for a
            linear drift).
        beta : Optional[Union[NDArrayFloat, float]], optional
            Drift coefficients :math:`\\boldsymbol{\\beta}` with shape
            (Nbeta,), by default None. Must be provided before calling
            :py:meth:`DriftMatrix.get_values`.

        Raises
        ------
        ValueError
            If the shape of `beta` does not match the number of columns of
            `mat`.
        """
        self.mat: NDArrayFloat = mat
        self.beta: Optional[Union[NDArrayFloat, float]] = beta

        if beta is not None:
            if isinstance(beta, float):
                shape = (1,)
            else:
                shape = np.shape(beta)
            if shape[0] != mat.shape[1]:
                raise ValueError(
                    f"beta has shape {shape} while it should be shape "
                    f"({mat.shape[1]},) to match the given coefficient matrix."
                )

    @property
    def s_dim(self) -> int:
        return self.mat.shape[0]

    @property
    def beta_dim(self) -> int:
        return self.mat.shape[1]

    def dot(self, beta: Union[float, NDArrayFloat]) -> NDArrayFloat:
        """Return the dot product."""
        return np.dot(self.mat, beta)

    def get_values(self, params: NDArrayFloat) -> NDArrayFloat:
        """
        Return the values of the prior term.

        Parameters
        ----------
        params : NDArrayFloat
            Values of the parameters for which to compute the prior. It has no effect
            with `DriftMatrix`.

        Returns
        -------
        NDArrayFloat
            The prior term values.
        """
        if params.size != self.mat.shape[0]:
            raise ValueError(
                f"The given values have size {params.size} while the X matrix "
                f"has been defined with shape {self.mat.shape}!"
            )
        if self.beta is None:
            raise ValueError("beta is None! A value must be given.")
        return self.dot(self.beta)

    def get_gradient_dot_product(
        self, input: NDArrayFloat
    ) -> Union[float, NDArrayFloat]:
        """
        Return the dot product of the gradient of the prior and the given input vector.

        Parameters
        ----------
        params : NDArrayFloat
            Values with which to compute the prior gradient dot product.

        Returns
        -------
        NDArrayFloat
            Prior gradient-input vector dot product.
        """
        return 0.0


class ConstantDriftMatrix(DriftMatrix):
    r"""
    Represent a constant (order 0) drift matrix (trend).

    The drift basis is a single, normalized constant column
    :math:`\mathbf{X} = \frac{1}{\sqrt{N_s}}\mathbf{1}`, so that
    :math:`\mathbf{X}\boldsymbol{\beta}` represents a spatially uniform mean
    (i.e., ordinary/simple kriging with an unknown constant mean).
    """

    def __init__(self, n_pts: int) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        n_pts : int
            Number of points (`Ns`) covered by the drift.
        """
        mat: NDArrayFloat = np.ones((n_pts, 1), dtype="d") / np.sqrt(n_pts)
        super().__init__(mat)


class LinearDriftMatrix(DriftMatrix):
    r"""
    Represent a linear (order 1) drift matrix (trend).

    The drift basis is made of a constant column followed by the raw point
    coordinates, :math:`\mathbf{X} = [\mathbf{1}, \mathbf{pts}]`, with shape
    :math:`(N_s, 1 + N_{dim})`, so that :math:`\mathbf{X}\boldsymbol{\beta}`
    represents a mean varying linearly with the spatial coordinates (i.e.,
    universal kriging with a linear trend).
    """

    def __init__(self, pts: NDArrayFloat) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        pts : NDArrayFloat
            Coordinates of the points, with shape (Ns, Ndim), Ndim being the
            number of spatial dimensions.
        """
        mat: NDArrayFloat = np.ones((pts.shape[0], 1 + pts.shape[1]), dtype=np.float64)
        mat[:, 1 : mat.shape[1]] = np.copy(pts)
        super().__init__(mat)
