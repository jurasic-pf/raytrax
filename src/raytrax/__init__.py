"""Main module for Raytrax."""

import jax

jax.config.update("jax_enable_x64", True)

from .api import trace
from .equilibrium.interpolate import MagneticConfiguration
from .types import Beam, Interpolators, RadialProfiles
