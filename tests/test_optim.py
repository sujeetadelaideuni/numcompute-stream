import numpy as np
import pytest
from numcompute.optim import grad, jacobian

def test_grad_quadratic_matches_analytic() -> None:
    def f(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    x = np.array([3.0, -1.5, 2.0])
    g = grad(f, x, h=1e-6, method="central")
    np.testing.assert_allclose(g, 2.0 * x, rtol=1e-5, atol=1e-5)


def test_central_smaller_error_than_forward_for_smooth_f() -> None:
    def f(x: np.ndarray) -> float:
        return float(x[0] ** 3)

    x = np.array([0.7])
    h = 1e-3
    g_c = grad(f, x, h=h, method="central")[0]
    g_f = grad(f, x, h=h, method="forward")[0]
    truth = 3.0 * x[0] ** 2
    assert abs(g_c - truth) < abs(g_f - truth)


def test_grad_bad_method() -> None:
    with pytest.raises(ValueError, match="central"):
        grad(lambda z: float(z[0]), np.ones(2), method="sideways")


def test_jacobian_shape_and_known_linear_model() -> None:
    def F(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2, x[0] * x[1]], dtype=float)

    x = np.array([2.0, 3.0])
    J = jacobian(F, x, h=1e-6, method="central")
    assert J.shape == (2, 2)
    analytic = np.array([[2.0 * x[0], 0.0], [x[1], x[0]]], dtype=float)
    np.testing.assert_allclose(J, analytic, rtol=1e-4, atol=1e-4)


def test_jacobian_forward_matches_known_values() -> None:
    def F(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] + 2.0 * x[1], x[0] * x[1]], dtype=float)

    x = np.array([1.2, -0.5])
    J = jacobian(F, x, h=1e-6, method="forward")
    analytic = np.array([[1.0, 2.0], [x[1], x[0]]], dtype=float)
    np.testing.assert_allclose(J, analytic, rtol=1e-4, atol=1e-4)


def test_jacobian_rejects_non_positive_step() -> None:
    with pytest.raises(ValueError, match="positive"):
        jacobian(lambda z: np.array([z[0]]), np.array([1.0]), h=0.0)
