import jax
import jax.numpy as jnp
import numpy as np
from raytrax import dielectric_tensor as dielectric_tensor_module
from raytrax import polarization, power_flux, quantities
from scipy.optimize import root_scalar

jax.config.update("jax_enable_x64", True)


def test_hamiltonian():
    refractive_index_perp = 0.6
    frequency = 220e9
    plasma_frequency = 2e9
    cyclotron_frequency = 232e9
    electron_temperature_keV = 1.0
    mode = "X"
    thermal_velocity = quantities.normalized_electron_thermal_velocity(
        electron_temperature_keV=electron_temperature_keV
    )

    @jax.jit
    def _h(n_para):
        dielectric_tensor = (
            dielectric_tensor_module.weakly_relativistic_dielectric_tensor(
                frequency=frequency,
                plasma_frequency=plasma_frequency,
                cyclotron_frequency=cyclotron_frequency,
                thermal_velocity=thermal_velocity,
                refractive_index_para=n_para,
                max_s=1,
                max_k=1,
            )
        )
        polarization_vector = polarization.polarization(
            dielectric_tensor=dielectric_tensor,
            refractive_index_perp=refractive_index_perp,
            refractive_index_para=n_para,
            frequency=frequency,
            cyclotron_frequency=cyclotron_frequency,
            mode=mode,
        )
        return power_flux.power_flux_hamiltonian_stix(
            refractive_index=jnp.array([refractive_index_perp, 0.0, n_para]),
            dielectric_tensor=dielectric_tensor,
            polarization_vector=polarization_vector,
        )

    # determine n_para such that hamiltonian is zero
    sol = root_scalar(_h, bracket=[0.1, 1.0], method="bisect")
    refractive_index_para = sol.root

    np.testing.assert_allclose(
        _h(refractive_index_para),
        0.0,
        rtol=0,
        atol=1e-6,
    )


def test_power_flux_vector():
    refractive_index_perp = 0.6
    refractive_index_para = 0.8
    frequency = 220e9
    plasma_frequency = 2e9
    cyclotron_frequency = 232e9
    electron_temperature_keV = 1.0
    mode = "X"
    thermal_velocity = quantities.normalized_electron_thermal_velocity(
        electron_temperature_keV=electron_temperature_keV
    )
    dielectric_tensor = dielectric_tensor_module.weakly_relativistic_dielectric_tensor(
        frequency=frequency,
        plasma_frequency=plasma_frequency,
        cyclotron_frequency=cyclotron_frequency,
        thermal_velocity=thermal_velocity,
        refractive_index_para=refractive_index_para,
        max_s=1,
        max_k=1,
    )
    polarization_vector = polarization.polarization(
        dielectric_tensor=dielectric_tensor,
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        frequency=frequency,
        cyclotron_frequency=cyclotron_frequency,
        mode=mode,
    )
    power_flux_vector = power_flux.power_flux_vector_stix(
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        dielectric_tensor=dielectric_tensor,
        polarization_vector=polarization_vector,
    )

    assert power_flux_vector.shape == (3,)
    assert np.isfinite(power_flux_vector).all()
