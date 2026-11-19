# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""
Toeplitz matrix-vector multiplication adapted from Arvind Saibaba's code
https://github.com/arvindks/kle/blob/master/covariance/toeplitz/toeplitz.py
and PyPCGA:
https://github.com/jonghyunharrylee/pyPCGA/blob/master/pyPCGA/covariance/toeplitz.py
"""

from typing import Callable, List, Sequence, Union

import numpy as np

from covmats._types import NDArrayFloat, NDArrayInt


def create_toepliz_first_row(
    pts: NDArrayFloat,
    kernel: Callable,
    lenscale: NDArrayFloat,
) -> NDArrayFloat:
    """
    Create the first row of the covariance matrix.

    Note
    ----
    The first row is also the first column of the matrix
    because covariance matrices are symmetric.

    Parameters
    ----------
    pts : NDArrayFloat
        Coordinates of the grid mesh centers of shape (Npts, Ndim), with Ndim the number
        of spatial dimensions between 1 and n.
    kernel : Callable
        Covariance kernel.
    lenscale : NDArrayFloat
        Correlation length.

    Returns
    -------
    NDArrayFloat
        Array of correlations between the first point of the array and all gridblocks
        with size Npts.
    """
    # We scale the points coordinates by the correlation length
    scaled_pts = pts / np.array(lenscale).reshape(1, -1)
    # Then we compute the distance for the first row of the matrix (between) the first
    # mesh center and all others. We square the difference over all axis and take the
    # sqrt of the sum (euclidean distance).
    scaled_distances = np.sqrt(np.sum((scaled_pts - scaled_pts[0]) ** 2, axis=1))
    return kernel(scaled_distances)


def _circulant_embedding(
    first_row: NDArrayFloat, shape: Union[int, Sequence[int], NDArrayInt]
) -> NDArrayFloat:
    """Embed the (BTTB) matrix described by its first row in a block circulant one."""
    _shape = np.array([shape], dtype=np.int64).ravel()
    dim: int = _shape.size

    if dim > 3 or dim == 0:
        raise ValueError("Support 1,2 and 3 dimensions.")

    circ = np.reshape(first_row, _shape, order="F")

    mask: List[slice] = [slice(None) for i in range(dim)]
    for i in range(dim):
        # remove first and last element on that axis
        mask[dim - i - 1] = slice(1, -1)
        # flip the axis and remove first and last element
        circ = np.concatenate(
            (circ, np.flip(circ, axis=dim - i - 1)[tuple(mask)]), axis=dim - i - 1
        )
        # restore the mask to its initial state
        mask[dim - i - 1] = slice(None)
    return circ


def toeplitz_circulant_spectrum(
    first_row: NDArrayFloat,
    shape: Union[int, Sequence[int], NDArrayInt],
) -> NDArrayFloat:
    """
    Return the (real-input) FFT of the circulant embedding of the matrix.

    It only depends on the matrix, not on the vector to multiply: compute it once
    and reuse it with :py:func:`toeplitz_product_from_spectrum` to save one of the
    two forward FFTs at each matrix-vector product.

    Parameters
    ----------
    first_row : NDArrayFloat
        First row of the covariance matrix.
    shape : Union[int, Sequence[int], NDArrayInt]
        Shape of the grid (nx, [ny, nz])

    Returns
    -------
    NDArrayFloat
        Complex array (output of ``numpy.fft.rfftn``) with the embedding shape,
        halved on the last axis.
    """
    circ = _circulant_embedding(first_row, shape)
    return np.fft.rfftn(circ, axes=list(range(circ.ndim)))


def toeplitz_product_from_spectrum(
    x: NDArrayFloat,
    spectrum: NDArrayFloat,
    shape: Union[int, Sequence[int], NDArrayInt],
) -> NDArrayFloat:
    """
    Return the product of the covariance matrix with `x` from a precomputed spectrum.

    See :py:func:`toeplitz_circulant_spectrum` and :py:func:`toeplitz_product`.
    """
    _shape = np.array([shape], dtype=np.int64).ravel()
    dim: int = _shape.size
    # Embedding shape: 2n - 2 along each axis (n for n == 1)
    full_shape = tuple(int(max(n, 2 * n - 2)) for n in _shape)
    axes = list(range(dim))

    # Zero-padding to the embedding shape is done by the FFT itself (`s`)
    padded = np.reshape(np.asarray(x, dtype=np.float64), _shape, order="F")
    result = np.fft.irfftn(
        spectrum * np.fft.rfftn(padded, s=full_shape, axes=axes),
        s=full_shape,
        axes=axes,
    )

    # Get the result and return
    mask = tuple(slice(0, _shape[i]) for i in range(dim))
    return np.reshape(result[mask], -1, order="F")


def toeplitz_product(
    x: NDArrayFloat,
    first_row: NDArrayFloat,
    shape: Union[int, Sequence[int], NDArrayInt],
) -> NDArrayFloat:
    """
    Return the product of the covariance matrix with a vector using toeplitz trick.

    It is easy to see that covariance matrices corresponding to regular grids
    (in 2D and 3D) result in block Toeplitz matrices, with Toeplitz sub-blocks (BTTB).
    We can exploit this to perform fast matrix-vector products.
    See :cite:t:`saibabaFastAlgorithmsGeostatistical2013`.

    Adapted from
    https://github.com/jonghyunharrylee/pyPCGA/blob/master/pyPCGA/covariance/toeplitz.py

    Note
    ----
    For repeated products with the same matrix, prefer
    :py:func:`toeplitz_circulant_spectrum` + :py:func:`toeplitz_product_from_spectrum`.

    Parameters
    ----------
    x : NDArrayFloat
        Input vector to multiply.
    first_row : NDArrayFloat
        First row of the covariance matrix.
    shape : Union[int, Sequence[int], NDArrayInt]
        Shape of the grid (nx, [ny, nz])

    Returns
    -------
    NDArrayFloat
        The matrix-vector product.

    Raises
    ------
    ValueError
        If the given shape does not match 1D, 2D or 3D.
    """
    return toeplitz_product_from_spectrum(
        x, toeplitz_circulant_spectrum(first_row, shape), shape
    )
