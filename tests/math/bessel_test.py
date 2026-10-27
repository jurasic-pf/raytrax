import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import jv as scipy_jv
from scipy.special import kv as scipy_kv
from scipy.special import kve as scipy_kve

from raytrax.math.bessel import jv_jax, kv_jax, kve_jax

jax.config.update("jax_enable_x64", True)
_MACHINE_PRECISION = float(jnp.finfo(float).eps)


def _jv_log_space_series(v, z):
    """Previous J_v(z) series (log-space terms, 50 terms), kept as a reference."""
    half_z = z / 2.0
    k = jnp.arange(50, dtype=jnp.float64).reshape((-1,) + (1,) * jnp.ndim(z))
    log_terms = (2 * k + v) * jnp.log(jnp.maximum(jnp.abs(half_z), 1e-300)) - (
        jax.scipy.special.gammaln(k + 1) + jax.scipy.special.gammaln(k + v + 1)
    )
    return jnp.sum((-1.0) ** k * jnp.exp(log_terms), axis=0)


@pytest.mark.parametrize("v", [0, 1, 5, -1])
@pytest.mark.parametrize("z", [0.0, 1.0, 2.0, 5.0, 10.0])
def test_jv_jax_real(v, z):
    z_jax = jnp.float64(z)
    # The pure-JAX series implementation loses some precision for large z due to
    # cancellation in the alternating series.  For z <= 2 (the application range)
    # the error is < 1e-15; at z = 10 it reaches ~1e-12.
    np.testing.assert_allclose(jv_jax(v, z_jax), scipy_jv(v, z), rtol=0, atol=1e-11)


@pytest.mark.parametrize("v", [0, 1, 2, 3])
def test_jv_jax_matches_previous_series(v):
    # k_perp rho stays below 2 on the public test rays; the series is checked to 10.
    z = jnp.linspace(0, 10, 2001)
    got = np.asarray(jax.vmap(lambda zi: jv_jax(v, zi))(z))
    # The previous series loses ~1e-12 to cancellation near z = 10; the new one ~1e-13.
    np.testing.assert_allclose(got, _jv_log_space_series(v, z), rtol=0, atol=5e-12)
    np.testing.assert_allclose(got[z <= 2], scipy_jv(v, z[z <= 2]), rtol=0, atol=1e-15)


@pytest.mark.parametrize("v", [0, 1, 5])
def test_jv_jax_derivative(v):
    z = jnp.linspace(-10, 10, 2000, dtype=jnp.float64)
    dx = z[1] - z[0]

    def jv_scalar(zi):
        return jv_jax(v, zi)

    value = jv_scalar(z)

    derivative = jax.vmap(jax.grad(jv_scalar))(z)

    expected_derivative = (value[2:] - value[:-2]) / (2 * dx)

    np.testing.assert_allclose(derivative[1:-1], expected_derivative, rtol=0, atol=0.01)


@pytest.mark.parametrize("v", [0, 1, 5, -1])
@pytest.mark.parametrize("z", [1.0, 2.0, 5.0, 10.0])  # Note: kv not defined at z=0
def test_kv_jax_real(v, z):
    z_complex = jnp.float64(z)
    # Series + recurrence methods accumulate numerical errors, so we use a more
    # realistic tolerance than machine precision
    np.testing.assert_allclose(
        kv_jax(v, z_complex), scipy_kv(v, z_complex), rtol=1e-11, atol=1e-13
    )


@pytest.mark.parametrize("v", [0, 1, 5])
def test_kv_jax_derivative(v):
    # Avoiding small z values where kv can diverge
    z = jnp.linspace(0.5, 10, 2000, dtype=jnp.float64)
    dx = z[1] - z[0]

    def kv_scalar(zi):
        return kv_jax(v, zi)

    value = kv_scalar(z)

    derivative = jax.vmap(jax.grad(kv_scalar))(z)

    # Finite difference for comparison
    expected_derivative = (value[2:] - value[:-2]) / (2 * dx)

    # Check only for the inner points and with relatively loose tolerance
    np.testing.assert_allclose(derivative[1:-1], expected_derivative, rtol=1e-3, atol=0)


@pytest.mark.parametrize("v", [0, 1, 2, 5, -1])
@pytest.mark.parametrize("z", [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
def test_kve_jax_real(v, z):
    z_complex = jnp.float64(z)
    # For small z (< switchover), we use series which has limited precision
    # For large z (>= switchover), asymptotic expansion is very accurate
    if z < 10.0:
        np.testing.assert_allclose(
            kve_jax(v, z_complex), scipy_kve(v, z_complex), rtol=1e-11, atol=1e-13
        )
    elif z == 10.0:
        # At the switchover boundary, asymptotic expansion is less accurate
        np.testing.assert_allclose(
            kve_jax(v, z_complex), scipy_kve(v, z_complex), rtol=1e-9, atol=1e-10
        )
    else:
        # For z > 10, asymptotic expansion is accurate to near machine precision
        np.testing.assert_allclose(
            kve_jax(v, z_complex), scipy_kve(v, z_complex), rtol=1e-13, atol=1e-15
        )


@pytest.mark.parametrize("v", [0, 1, 5])
def test_kve_jax_derivative(v):
    # Test with a range that includes large values where kve is useful
    z = jnp.linspace(0.5, 20, 2000, dtype=jnp.float64)
    dx = z[1] - z[0]

    def kve_scalar(zi):
        return kve_jax(v, zi)

    value = kve_scalar(z)

    derivative = jax.vmap(jax.grad(kve_scalar))(z)

    # Finite difference for comparison
    expected_derivative = (value[2:] - value[:-2]) / (2 * dx)

    # Check only for the inner points with relatively loose tolerance
    # Higher order Bessel functions have steeper derivatives near small z, so need higher atol
    np.testing.assert_allclose(
        derivative[1:-1], expected_derivative, rtol=1e-3, atol=5e2
    )
