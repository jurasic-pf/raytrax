"""Main functions to interact with Raytrax."""

import jax
import jax.numpy as jnp
import jaxtyping as jt

from .equilibrium.interpolate import (
    build_magnetic_field_interpolator,
    build_rho_interpolator,
    build_electron_density_profile_interpolator,
    build_electron_temperature_profile_interpolator,
    MagneticConfiguration,
)
from .ray import RaySetting
from .solver import trace_jitted
from .types import (
    Beam,
    BeamProfile,
    Interpolators,
    RadialProfile,
    RadialProfiles,
    TraceResult,
)


def _bin_power_deposition(
    rho_grid: jt.Float[jax.Array, " nrho"],
    dvolume_drho: jt.Float[jax.Array, " nrho"],
    rho_trajectory: jt.Float[jax.Array, " n"],
    arc_length: jt.Float[jax.Array, " n"],
    linear_power_density: jt.Float[jax.Array, " n"],
) -> jt.Float[jax.Array, " nrho"]:
    """Compute volumetric power deposition profile from ray trajectory.

    Implements δP_abs/δV = Σ_i (dP/ds)_i * δs_i / δV_bin, summing over all
    trajectory segments i that cross the flux-surface shell at rho.
    """
    # Upsample to a fine arc-length grid so each sub-segment spans << one rho bin,
    # giving a smooth histogram without the staircase artifact.
    n_fine = 50 * len(rho_grid)
    s_fine = jnp.linspace(arc_length[0], arc_length[-1], n_fine)
    rho_fine = jnp.interp(s_fine, arc_length, rho_trajectory)
    dpds_fine = jnp.interp(s_fine, arc_length, linear_power_density)

    ds_fine = (arc_length[-1] - arc_length[0]) / (n_fine - 1)
    dP_fine = dpds_fine[:-1] * ds_fine
    rho_mid_fine = 0.5 * (rho_fine[:-1] + rho_fine[1:])

    edges = jnp.concatenate(
        [rho_grid[:1], 0.5 * (rho_grid[:-1] + rho_grid[1:]), rho_grid[-1:]]
    )
    indices = jnp.clip(jnp.searchsorted(edges, rho_mid_fine) - 1, 0, len(rho_grid) - 1)
    power_per_bin = jnp.zeros_like(rho_grid).at[indices].add(dP_fine)

    dV = dvolume_drho * jnp.diff(edges)
    return power_per_bin / dV


def trace(
    magnetic_configuration: MagneticConfiguration,
    radial_profiles: RadialProfiles,
    beam: Beam,
) -> TraceResult:
    """Trace a single beam through the plasma.

    Args:
        magnetic_configuration: Magnetic configuration with gridded data
        radial_profiles: Radial profiles of plasma parameters
        beam: Beam initial conditions (position, direction, frequency, mode)

    Returns:
        TraceResult with beam profile and radial deposition profile.
    """
    setting = RaySetting(frequency=beam.frequency, mode=beam.mode)

    interpolators = Interpolators(
        magnetic_field=build_magnetic_field_interpolator(magnetic_configuration),
        rho=build_rho_interpolator(magnetic_configuration),
        electron_density=build_electron_density_profile_interpolator(radial_profiles),
        electron_temperature=build_electron_temperature_profile_interpolator(
            radial_profiles
        ),
        is_axisymmetric=magnetic_configuration.is_axisymmetric,
    )

    result = trace_jitted(
        jnp.asarray(beam.position),
        jnp.asarray(beam.direction),
        setting,
        interpolators,
        magnetic_configuration.nfp,
        magnetic_configuration.rho_1d,
        magnetic_configuration.dvolume_drho,
    )

    # Trim padded buffer to valid entries
    n = int(jnp.sum(jnp.isfinite(result.arc_length)).item())

    # Prepend antenna position as first point (with zero values)
    antenna_position = jnp.asarray(beam.position).reshape(1, 3)
    antenna_direction = jnp.asarray(beam.direction).reshape(1, 3)
    zero_scalar = jnp.zeros(1)
    zero_vector = jnp.zeros((1, 3))

    beam_profile = BeamProfile(
        position=jnp.concatenate([antenna_position, result.ode_state[:n, :3]], axis=0),
        arc_length=jnp.concatenate([zero_scalar, result.arc_length[:n]], axis=0),
        refractive_index=jnp.concatenate(
            [antenna_direction, result.ode_state[:n, 3:6]], axis=0
        ),
        optical_depth=jnp.concatenate([zero_scalar, result.ode_state[:n, 6]], axis=0),
        absorption_coefficient=jnp.concatenate(
            [zero_scalar, result.absorption_coefficient[:n]], axis=0
        ),
        electron_density=jnp.concatenate(
            [zero_scalar, result.electron_density[:n]], axis=0
        ),
        electron_temperature=jnp.concatenate(
            [zero_scalar, result.electron_temperature[:n]], axis=0
        ),
        magnetic_field=jnp.concatenate(
            [zero_vector, result.magnetic_field[:n]], axis=0
        ),
        normalized_effective_radius=jnp.concatenate(
            [jnp.array([jnp.nan]), result.normalized_effective_radius[:n]], axis=0
        ),
        linear_power_density=jnp.concatenate(
            [zero_scalar, result.linear_power_density[:n]], axis=0
        ),
    )

    power_binned = _bin_power_deposition(
        magnetic_configuration.rho_1d,
        magnetic_configuration.dvolume_drho,
        result.normalized_effective_radius[:n],
        result.arc_length[:n],
        result.linear_power_density[:n],
    )

    radial_profile = RadialProfile(
        rho=magnetic_configuration.rho_1d,
        volumetric_power_density=power_binned,
    )
    return TraceResult(beam_profile=beam_profile, radial_profile=radial_profile)
