"""
3D flux surface and beam tube
==============================

`plot_flux_surface_3d` renders the W7-X last closed flux surface using PyVista.
`plot_beam_profile_3d` adds the ECRH beam as a tube coloured by linear power density.

Images are captured from an off-screen renderer and embedded in the gallery
via matplotlib.
"""

# %%
# ## Setup

import contextlib
import io
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="equinox")
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-interactive.*")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv

from raytrax import Beam, RadialProfiles, trace
from raytrax.examples.w7x import (
    PortA,
    get_w7x_magnetic_configuration,
    w7x_aiming_angles_to_direction,
)
from raytrax.plot.plot3d import plot_beam_profile_3d, plot_flux_surface_3d

pv.OFF_SCREEN = True

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
# ## Last closed flux surface
#
# The LCFS ($\rho = 1$) of the W7-X equilibrium, showing the characteristic
# bean-shaped and triangular cross-sections around the torus.

plotter = pv.Plotter(off_screen=True)
plotter.add_axes()
plotter.view_isometric()
plot_flux_surface_3d(mag_conf, rho_value=1.0, plotter=plotter)
img = plotter.screenshot(return_img=True)
plotter.close()

fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(img)
ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# ## Combined scene: flux surface and beam tube
#
# The LCFS rendered semi-transparent with the ECRH beam tube overlaid,
# coloured by linear power density along the ray path.

plotter = pv.Plotter(off_screen=True)
plotter.add_axes()
plotter.view_isometric()
plot_flux_surface_3d(mag_conf, rho_value=1.0, plotter=plotter, opacity=0.25)
plot_beam_profile_3d(result.beam_profile, plotter=plotter, tube_radius=0.02)
img = plotter.screenshot(return_img=True)
plotter.close()

fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(img)
ax.axis("off")
plt.tight_layout()
plt.show()
