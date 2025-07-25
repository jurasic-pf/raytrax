import jax
import jax.numpy as jnp
import numpy as np
from raytrax.faddeeva import (
    plasma_dispersion_function,
    plasma_dispersion_function_derivative,
    wofz_jax,
)
from scipy.special import wofz as scipy_wofz

jax.config.update("jax_enable_x64", True)


def test_wofz_jax():
    # Test with real values
    real_inputs = jnp.array([0.0, 1.0, 2.0, -1.0, -2.0], jnp.complex128)
    for x in real_inputs:
        np.testing.assert_allclose(wofz_jax(x), scipy_wofz(x), rtol=0, atol=1e-16)

    # test with entire array
    np.testing.assert_allclose(
        wofz_jax(real_inputs), scipy_wofz(real_inputs), rtol=0, atol=1e-16
    )

    # Test with complex values
    complex_inputs = jnp.array(
        [
            1.0 + 1.0j,
            2.0 + 3.0j,
            -1.0 - 1.0j,
            0.5 - 0.5j,
            10.0 + 10.0j,
            1.0e-5 + 1.0e-5j,
        ]
    )
    for z in complex_inputs:
        np.testing.assert_allclose(wofz_jax(z), scipy_wofz(z), rtol=0, atol=1e-16)

    # test with entire array
    np.testing.assert_allclose(
        wofz_jax(complex_inputs), scipy_wofz(complex_inputs), rtol=0, atol=1e-16
    )


def test_plasma_dispersion_function_derivative():
    z = jnp.linspace(-10, 10, 1000, dtype=jnp.complex128)
    dx = z[1] - z[0]
    value = plasma_dispersion_function(z)
    derivative = plasma_dispersion_function_derivative(z)
    expected_derivative = (value[2:] - value[:-2]) / (2 * dx)
    np.testing.assert_allclose(derivative[1:-1], expected_derivative, rtol=0, atol=1e-3)


def test_wofz_jax_derivative():
    z = jnp.linspace(-10, 10, 1000, dtype=jnp.complex128)
    dx = z[1] - z[0]
    wofz_scalar = lambda zi: wofz_jax(zi)
    value = wofz_scalar(z)
    derivative = jax.vmap(jax.grad(wofz_scalar, holomorphic=True))(z)
    expected_derivative = (value[2:] - value[:-2]) / (2 * dx)
    np.testing.assert_allclose(derivative[1:-1], expected_derivative, rtol=0, atol=1e-3)
