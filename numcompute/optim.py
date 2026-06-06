from __future__ import annotations
from collections.abc import Callable
import numpy as np


def grad(
    f: Callable[[np.ndarray], float | np.floating],
    x: np.ndarray,
    h: float = 1e-5,
    method: str = "central",
) -> np.ndarray:
    """Estimate the gradient of a scalar field ``f`` at ``x`` using finite differences.

    Parameters
    ----------
    f
        Scalar function mapping a 1-D array of shape ``(n,)`` to a real scalar.
    x
        Evaluation point, shape ``(n,)``. Must be one-dimensional.
    h
        Step size for the finite-difference stencil.
    method
        ``"central"`` for symmetric ``O(h**2)`` truncation error, or ``"forward"`` for one-sided
        ``O(h)`` differences.

    Returns
    -------
    np.ndarray
        Gradient vector, shape ``(n,)``, same dtype as ``x`` (promoted as needed for arithmetic).

    Raises
    ------
    ValueError
        If ``method`` is not supported, ``h`` is non-positive, or ``x`` is not 1-D.

    Notes
    -----
    Time complexity: ``O(n * C_f)`` where ``C_f`` is the cost of one evaluation of ``f``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array of shape (n,).")
    if h <= 0:
        raise ValueError("Step size h must be positive.")
    n = x.shape[0]
    basis = np.eye(n, dtype=float)
    g = np.empty(n, dtype=float)

    if method == "central":
        for i in range(n):
            ei = basis[i]
            g[i] = (float(f(x + h * ei)) - float(f(x - h * ei))) / (2.0 * h)
    elif method == "forward":
        fx = float(f(x))
        for i in range(n):
            ei = basis[i]
            g[i] = (float(f(x + h * ei)) - fx) / h
    else:
        raise ValueError("method must be 'central' or 'forward'.")

    return g


def jacobian(
    F: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float = 1e-5,
    method: str = "central",
) -> np.ndarray:
    """Estimate the Jacobian of a vector-valued function ``F`` at ``x``.

    Parameters
    ----------
    F
        Function mapping ``x`` of shape ``(n,)`` to a 1-D array of shape ``(m,)``.
    x
        Evaluation point, shape ``(n,)``.
    h
        Finite-difference step size, forwarded to :func:`grad`.
    method
        ``"central"`` or ``"forward"``, forwarded to :func:`grad`.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape ``(m, n)`` where row ``i`` is the gradient of component ``F_i``.

    Raises
    ------
    ValueError
        If ``F(x)`` is not 1-D or ``x`` is not 1-D.

    Notes
    -----
    Implemented by applying :func:`grad` to each scalar component ``F_i``. Time complexity is
    ``O(m * n * C_scalar)`` for ``m`` outputs, ``n`` inputs, and scalar-evaluation cost ``C_scalar``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array of shape (n,).")
    if h <= 0:
        raise ValueError("Step size h must be positive.")

    y0 = np.asarray(F(x), dtype=float)
    if y0.ndim != 1:
        raise ValueError("F(x) must return a 1-D array of shape (m,).")
    m = y0.shape[0]
    n = x.shape[0]
    J = np.empty((m, n), dtype=float)

    def scalar_component(i: int) -> Callable[[np.ndarray], float]:
        def fi(z: np.ndarray) -> float:
            y = np.asarray(F(z), dtype=float)
            if y.ndim != 1 or y.shape != (m,):
                raise ValueError("F(x) must consistently return a 1-D array of shape (m,).")
            return float(y[i])

        return fi

    for i in range(m):
        J[i] = grad(scalar_component(i), x, h=h, method=method)

    return J
