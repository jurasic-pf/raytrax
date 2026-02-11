"""Main functions to interact with Raytrax."""

import jax.numpy as jnp

from .fourier import dvolume_drho
from .interpolate import (
    build_magnetic_field_interpolator,
    build_rho_interpolator,
    build_electron_density_profile_interpolator,
    build_electron_temperature_profile_interpolator,
    cylindrical_grid_for_equilibrium,
)
from .ray import RaySetting, RayState
from .solver import solve
from .type_conversion import ray_states_to_beam_profile, ray_states_to_radial_profile
from .types import (
    Beam,
    MagneticConfiguration,
    RadialProfiles,
    TracingResult,
    WoutLike,
)


def get_interpolator_for_equilibrium(
    equilibrium: WoutLike,
    magnetic_field_scale: float = 1.0,
) -> MagneticConfiguration:
    """Generate interpolators for the given MHD equilibrium.

    Args:
        equilibrium: an MHD equilibrium compatible with `vmecpp.VmecWOut`
        magnetic_field_scale: Factor to multiply all magnetic field values by.
            Use this to normalize the field strength, e.g. to match TRAVIS's
            ``B0_normalization_type at angle on magn.axis`` setting.

    Returns:
        A MagneticConfiguration object containing interpolation data.
    """
    # TODO add settings for grid resolution
    interpolated_array = cylindrical_grid_for_equilibrium(
        equilibrium=equilibrium, n_rho=40, n_theta=45, n_phi=50, n_r=45, n_z=55
    )
    rphiz = interpolated_array[..., :3]
    rho = interpolated_array[..., 3]
    magnetic_field = interpolated_array[..., 4:] * magnetic_field_scale

    # Compute volume derivative on 1D radial grid
    rho_1d = jnp.linspace(0, 1, 200)
    dv_drho = dvolume_drho(equilibrium, rho_1d)

    return MagneticConfiguration(
        rphiz=rphiz,
        magnetic_field=magnetic_field,
        rho=rho,
        nfp=equilibrium.nfp,
        stellarator_symmetric=not equilibrium.lasym,
        rho_1d=rho_1d,
        dvolume_drho=dv_drho,
    )


def trace(
    equilibrium_interpolator: MagneticConfiguration,
    radial_profiles: RadialProfiles,
    beam: Beam,
) -> TracingResult:
    """Solve the ray tracing equations for a given beam given an MHD equilibrium.

    Args:
        equilibrium_interpolator: Magnetic configuration with gridded data
        radial_profiles: Radial profiles of plasma parameters
        beam: A Beam object containing the initial conditions of the beam

    Returns:
        A TracingResult object containing the results of the tracing.

    Note:
        For best performance when tracing multiple beams, reuse the same
        MagneticConfiguration and RadialProfiles objects. Their cached
        interpolator properties avoid ~1-2s JAX recompilation per call.

    Example:
        >>> # Trace multiple beams efficiently
        >>> eq = get_interpolator_for_equilibrium(wout)
        >>> profiles = RadialProfiles(...)
        >>> for beam in beams:
        ...     result = trace(eq, profiles, beam)  # Fast after first call
    """
    # Use the beam direction as the initial refractive index direction
    initial_state = RayState(
        position=jnp.asarray(beam.position),
        refractive_index=jnp.asarray(beam.direction),
        optical_depth=jnp.array(0.0),
        arc_length=jnp.array(0.0),
    )
    setting = RaySetting(frequency=beam.frequency, mode=beam.mode)

    # Access cached interpolator properties. These build and cache the interpolators
    # on first access, then return the same callable objects on subsequent calls.
    # JAX uses object identity (id()) to cache compiled traces, so returning the
    # same callable object avoids ~1-2s recompilation overhead.
    magnetic_field_interpolator = equilibrium_interpolator.magnetic_field_interpolator
    rho_interpolator = equilibrium_interpolator.rho_interpolator
    electron_density_profile_interpolator = (
        radial_profiles.electron_density_interpolator
    )
    electron_temperature_profile_interpolator = (
        radial_profiles.electron_temperature_interpolator
    )
    # Solve ray tracing equations with augmented state vector
    # This computes all quantities (magnetic field, density, temperature, absorption, power)
    # in a single pass during ODE integration, avoiding expensive post-processing
    ray_states, additional_quantities = solve(
        state=initial_state,
        setting=setting,
        magnetic_field_interpolator=magnetic_field_interpolator,
        rho_interpolator=rho_interpolator,
        electron_density_profile_interpolator=electron_density_profile_interpolator,
        electron_temperature_profile_interpolator=electron_temperature_profile_interpolator,
    )
    beam_profile = ray_states_to_beam_profile(ray_states, additional_quantities)
    radial_profile = ray_states_to_radial_profile(
        ray_states,
        additional_quantities,
        equilibrium_interpolator.rho_1d,
        equilibrium_interpolator.dvolume_drho,
    )
    return TracingResult(beam_profile=beam_profile, radial_profile=radial_profile)
