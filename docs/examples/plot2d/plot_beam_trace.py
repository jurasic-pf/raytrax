"""
Beam trace visualisation
========================

`plot_beamtrace_rz` overlays the ray-tracing trajectory on a poloidal
cross-section.

This example traces the W7-X Port-A ECRH beam through the bundled W7-X
equilibrium and draws the result on top of the flux-surface contours.
"""

# %%
# ## Setup: W7-X equilibrium, profiles, and beam
#
# `get_w7x_magnetic_configuration` loads the equilibrium (cached after the
# first run). `w7x_aiming_angles_to_direction` converts poloidal/toroidal
# aiming angles into a Cartesian unit vector.  The density peak is set to
# $2.0 \times 10^{20}\ \mathrm{m}^{-3}$ — well below the 140 GHz O-mode
# cutoff (${\approx}2.43 \times 10^{20}\ \mathrm{m}^{-3}$) but high enough
# that refraction curves the ray path visibly.

import contextlib
import io
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="equinox")
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-interactive.*")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from raytrax import Beam, RadialProfiles, trace
from raytrax.examples.w7x import (
    PortA,
    get_w7x_magnetic_configuration,
    w7x_aiming_angles_to_direction,
)
from raytrax.plot.plot2d import plot_beamtrace_rz, plot_effective_radius_rz

with contextlib.redirect_stdout(io.StringIO()):
    mag_conf = get_w7x_magnetic_configuration()
phi = np.deg2rad(PortA.D1.phi_deg)

rho_prof = jnp.linspace(0, 1, 200)
profiles = RadialProfiles(
    rho=rho_prof,
    electron_density=2.0 * (1.0 - rho_prof**2),
    electron_temperature=3.0 * (1.0 - rho_prof**2),
)

# %%
# ## Running the ray tracer
#
# `trace` returns a `TraceResult`.  `trim=True` (the default) removes padding
# from the output arrays.

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

tau = float(result.beam_profile.optical_depth[-1])
print(f"Optical depth tau     = {tau:.3f}")
print(f"Absorbed fraction     = {1.0 - float(jnp.exp(-tau)):.1%}")

# %%
# ## Overlaying the beam on a cross-section
#
# `plot_beamtrace_rz` takes a `BeamProfile` and projects the 3-D trajectory
# onto the R–Z plane.

fig, ax = plt.subplots(figsize=(4, 5))
plot_effective_radius_rz(mag_conf, phi=phi, ax=ax)
plot_beamtrace_rz(result.beam_profile, phi=phi, ax=ax, lw=2)
ax.set_title(r"ECRH beam trace — D1, $\theta_\mathrm{pol} = -10°$")
plt.tight_layout()
plt.show()

# %%
# ## Comparing two poloidal aiming angles
#
# Both traces share the same plasma colourmap.  `add_colorbar=False` suppresses
# the per-trace colorbar; a single shared colorbar is added manually so both
# traces are on the same scale.

beam_up = Beam(
    position=jnp.array(PortA.D1.cartesian),
    direction=jnp.array(
        w7x_aiming_angles_to_direction(
            theta_pol_deg=5.0,
            theta_tor_deg=0.0,
            antenna_phi_deg=PortA.D1.phi_deg,
        )
    ),
    frequency=jnp.array(140e9),
    mode="O",
    power=1e6,
)
result_up = trace(mag_conf, profiles, beam_up)

import matplotlib.cm as cm
import matplotlib.colors as mcolors

p_all = (
    np.concatenate(
        [
            np.array(result.beam_profile.linear_power_density),
            np.array(result_up.beam_profile.linear_power_density),
        ]
    )
    / 1e6
)  # MW/m
norm = mcolors.Normalize(vmin=float(p_all.min()), vmax=float(p_all.max()))

fig, ax = plt.subplots(figsize=(4, 5))
plot_effective_radius_rz(mag_conf, phi=phi, ax=ax)
lc1 = plot_beamtrace_rz(
    result.beam_profile,
    phi=phi,
    ax=ax,
    lw=2,
    add_colorbar=False,
    norm=norm,
    label=r"$\theta_\mathrm{pol} = -10°$",
)
lc2 = plot_beamtrace_rz(
    result_up.beam_profile,
    phi=phi,
    ax=ax,
    lw=2,
    add_colorbar=False,
    norm=norm,
    label=r"$\theta_\mathrm{pol} = +5°$",
)
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(
    cm.ScalarMappable(norm=norm, cmap="plasma"),
    cax=cax,
    label="Linear power density [MW/m]",
)
ax.set_title("Two poloidal aiming angles")
plt.tight_layout()
plt.show()
