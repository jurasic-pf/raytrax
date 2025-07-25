import jax
import jax.numpy as jnp
import numpy as np
from raytrax import absorption

jax.config.update("jax_enable_x64", True)


def test_absorption():
    refractive_index_perp = 0.6
    refractive_index_para = 0.8
    refractive_index = jnp.array([refractive_index_perp, 0.0, refractive_index_para])

    magnetic_field = jnp.array([0.0, 0.0, 8.3])

    frequency = 220e9
    electron_density_1e20_per_m3 = 0.1
    electron_temperature_keV = 1.0
    mode = "X"

    alpha = absorption.absorption_coefficient(
        refractive_index=refractive_index,
        magnetic_field=magnetic_field,
        electron_density_1e20_per_m3=electron_density_1e20_per_m3,
        electron_temperature_keV=electron_temperature_keV,
        frequency=frequency,
        mode=mode,
    )

    print(f"Absorption coefficient: {alpha}")

    assert not np.isnan(alpha), "Absorption coefficient is NaN"
    assert alpha >= 0, "Absorption coefficient should be non-negative"
    print(alpha)
