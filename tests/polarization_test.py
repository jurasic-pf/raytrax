import jax
import jax.numpy as jnp
import numpy as np
import pytest
from raytrax import dielectric_tensor, dispersion, polarization

jax.config.update("jax_enable_x64", True)


def test_polarization():
    eps = dielectric_tensor.cold_dielectric_tensor(
        frequency=220.0,
        plasma_frequency=260.0,
        cyclotron_frequency=232.0,
    )
    pol = polarization.polarization(
        dielectric_tensor=eps,
        refractive_index_perp=1 - 0.4**2,
        refractive_index_para=0.4,
        frequency=220.0,
        cyclotron_frequency=232.0,
        mode="X",
    )
    assert pol.shape == (3,)
    assert jnp.isclose(jnp.linalg.norm(pol), 1.0)
    # TODO(dstraub): more meaningful functional tests


@pytest.mark.parametrize("mode", ["X", "O"])
def test_polarization_low_density(mode):
    frequency = 220e9
    plasma_frequency = 2e9
    cyclotron_frequency = 232e9
    n_para = 0.2
    pol1 = polarization._polarization_low_density(
        refractive_index_para=n_para,
        frequency=frequency,
        cyclotron_frequency=cyclotron_frequency,
        plasma_frequency=plasma_frequency,
        mode=mode,
    )
    np.testing.assert_allclose(jnp.linalg.norm(pol1), 1.0, rtol=0, atol=1e-8)
    eps = dielectric_tensor.cold_dielectric_tensor(
        frequency=frequency,
        plasma_frequency=plasma_frequency,
        cyclotron_frequency=cyclotron_frequency,
    )
    tan_theta = 0.5
    X = plasma_frequency**2 / frequency**2
    Y = cyclotron_frequency / frequency
    n2 = dispersion._dispersion_appleton_hartee(
        X=X,
        Y=Y,
        sin2theta=tan_theta**2 / (1 + tan_theta**2),
        mode=mode,
    )
    assert n2 > n_para**2, "n2 must be larger than n_para^2"
    n_perp = jnp.sqrt(n2 - n_para**2)
    pol2 = polarization.polarization(
        dielectric_tensor=eps,
        refractive_index_perp=n_perp,
        refractive_index_para=n_para,
        frequency=frequency,
        cyclotron_frequency=cyclotron_frequency,
        mode=mode,
    )
    np.testing.assert_allclose(jnp.linalg.norm(pol2), 1.0, rtol=0, atol=1e-8)

    np.testing.assert_allclose(pol1, pol2, rtol=0, atol=0.08)
