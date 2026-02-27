"""
Radial profiles and power deposition
=====================================

`plot_radial_electron_density`, `plot_radial_electron_temperature`, and
`plot_radial_power_density` visualise the plasma state and ECRH heating
profile as functions of the normalised radius $\\rho$.

`plot_linear_power_density` shows power absorbed per metre of beam arc length.

This example uses the bundled W7-X equilibrium from `raytrax.examples.w7x`.
"""

# %%
# ## Setup: equilibrium, profiles, and beam trace
#
# We load the equilibrium, define parabolic profiles, and run `trace`
# to obtain both the beam trajectory and the radial deposition profile.

import contextlib
import io
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="equinox")
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-interactive.*")

import jax.numpy as jnp
import matplotlib.pyplot as plt

from raytrax import Beam, RadialProfiles, trace
from raytrax.examples.w7x import (
    PortA,
    get_w7x_magnetic_configuration,
    w7x_aiming_angles_to_direction,
)
from raytrax.plot.plot1d import (
    plot_linear_power_density,
    plot_radial_electron_density,
    plot_radial_electron_temperature,
    plot_radial_power_density,
)

with contextlib.redirect_stdout(io.StringIO()):
    mag_conf = get_w7x_magnetic_configuration()

rho_prof = jnp.linspace(0, 1, 200)
profiles = RadialProfiles(
    rho=rho_prof,
    electron_density=0.5 * (1.0 - rho_prof**2),
    electron_temperature=3.0 * (1.0 - rho_prof**2),
)

beam = Beam(
    position=jnp.array(PortA.D1.cartesian),
    direction=jnp.array(
        w7x_aiming_angles_to_direction(
            theta_pol_deg=-10.0,
            theta_tor_deg=0.0,
            antenna_phi_deg=PortA.D1.phi_deg,
        )
    ),
    frequency=jnp.array(140e9),
    mode="O",
    power=1e6,
)

result = trace(mag_conf, profiles, beam)

# %%
# ## Plasma profiles and power deposition
#
# Three-panel figure: electron density $n_e(\rho)$, electron temperature
# $T_e(\rho)$, and volumetric ECRH power density $\mathrm{d}P/\mathrm{d}V(\rho)$.

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
plot_radial_electron_density(profiles, ax=axes[0])
plot_radial_electron_temperature(profiles, ax=axes[1])
plot_radial_power_density(result.radial_profile, ax=axes[2])
plt.tight_layout()
plt.show()

# %%
# ## Linear power density
#
# `plot_linear_power_density` shows how much power is deposited per unit
# arc length along the beam path.

fig, ax = plt.subplots(figsize=(6, 3))
plot_linear_power_density(result.beam_profile, ax=ax)
plt.tight_layout()
plt.show()
