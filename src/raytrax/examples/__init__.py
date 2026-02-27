"""Example data and utilities for raytrax.

Quick start::

    from raytrax.examples import get_w7x_equilibrium, get_w7x_magnetic_configuration

    eq = get_w7x_magnetic_configuration()
"""

from .w7x import (
    AntennaPosition,
    PortA,
    get_w7x_equilibrium,
    get_w7x_magnetic_configuration,
    w7x_aiming_angles_to_direction,
)

__all__ = [
    "AntennaPosition",
    "PortA",
    "get_w7x_equilibrium",
    "get_w7x_magnetic_configuration",
    "w7x_aiming_angles_to_direction",
]
