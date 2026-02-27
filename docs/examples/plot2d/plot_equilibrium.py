"""
Equilibrium visualisation
=========================

The `raytrax.plot.plot2d` module provides four helpers for inspecting
a magnetic configuration on a poloidal R–Z cross-section:

- `plot_magnetic_field_rz` — contour lines of $|B|$
- `plot_effective_radius_rz` — flux-surface contours ($\\rho$)
- `plot_electron_density_rz` — filled colour map of $n_e$
- `interpolate_rz_slice` — raw interpolation data

This example uses the bundled W7-X equilibrium from `raytrax.examples.w7x`.
"""

# %%
# ## Loading the W7-X equilibrium
#
# `get_w7x_magnetic_configuration` builds the magnetic configuration from the
# bundled W7-X VMEC++ equilibrium.  The result is cached on disk after the
# first run.

import contextlib
import io
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="equinox")
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-interactive.*")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from raytrax import RadialProfiles
from raytrax.examples.w7x import PortA, get_w7x_magnetic_configuration
from raytrax.plot.plot2d import (
    plot_effective_radius_rz,
    plot_electron_density_rz,
    plot_magnetic_field_rz,
)

with contextlib.redirect_stdout(io.StringIO()):
    mag_conf = get_w7x_magnetic_configuration()

# Evaluate at the toroidal angle of Port A (≈ 0°)
phi = np.deg2rad(PortA.D1.phi_deg)

rho_prof = jnp.linspace(0, 1, 200)
profiles = RadialProfiles(
    rho=rho_prof,
    electron_density=0.5 * (1.0 - rho_prof**2),
    electron_temperature=3.0 * (1.0 - rho_prof**2),
)

# %%
# ## Magnetic field strength
#
# `plot_magnetic_field_rz` draws contour lines of $|B|$ in the R–Z plane.
# Points outside the last closed flux surface ($\rho > 1$) are masked.

fig, ax = plt.subplots(figsize=(4, 5))
plot_magnetic_field_rz(mag_conf, phi=phi, ax=ax, levels=8)
ax.set_title(r"$|B|$ (T)")
plt.tight_layout()
plt.show()

# %%
# ## Flux-surface contours
#
# `plot_effective_radius_rz` overlays the $\rho = 0.1, 0.2, \ldots, 1.0$
# contours.  The W7-X cross-section is non-circular — the bean-shaped geometry
# is clearly visible.

fig, ax = plt.subplots(figsize=(4, 5))
plot_effective_radius_rz(mag_conf, phi=phi, ax=ax)
ax.set_title(r"$\rho$ contours")
plt.tight_layout()
plt.show()

# %%
# ## Electron density
#
# `plot_electron_density_rz` maps $n_e(\rho)$ onto the poloidal plane via a
# filled contour plot.

fig, ax = plt.subplots(figsize=(4, 5))
plot_electron_density_rz(mag_conf, profiles, phi=phi, ax=ax)
ax.set_title(r"$n_e$ [$10^{20}$ m$^{-3}$]")
plt.tight_layout()
plt.show()

# %%
# ## Combining plots
#
# All helpers return the active `Axes` object, so they can be layered on a
# single figure.

fig, ax = plt.subplots(figsize=(4, 5))
plot_electron_density_rz(mag_conf, profiles, phi=phi, ax=ax)
plot_effective_radius_rz(mag_conf, phi=phi, ax=ax)
ax.set_title(r"$n_e$ with $\rho$ contours")
plt.tight_layout()
plt.show()
