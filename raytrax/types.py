from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable, Callable

import jax
import jaxtyping as jt


@runtime_checkable
class WoutLike(Protocol):
    """Protocol for objects that can be used as VmecWOut."""

    rmnc: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    zmns: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    xm: jt.Int[jax.Array, "n_fourier_coefficients"]
    xn: jt.Int[jax.Array, "n_fourier_coefficients"]
    gmnc: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    gmns: jt.Float[jax.Array, "n_fourier_coefficients n_surfaces"]
    bsupumnc: jt.Float[jax.Array, "n_fourier_coefficients_nyquist n_surfaces"]
    bsupvmnc: jt.Float[jax.Array, "n_fourier_coefficients_nyquist n_surfaces"]
    xm_nyq: jt.Int[jax.Array, "n_fourier_coefficients_nyquist"]
    xn_nyq: jt.Int[jax.Array, "n_fourier_coefficients_nyquist"]
    ns: int
    nfp: int
    lasym: bool


@dataclass
class Beam:
    """Dataclass for a beam to be traced."""

    position: jt.Float[jax.Array, "3"]
    """The starting position of the beam in cartesian coordinates."""

    direction: jt.Float[jax.Array, "3"]
    """The starting direction of the beam in cartesian coordinates."""

    frequency: jt.Float[jax.Array, ""]
    """The frequency of the beam in Hz."""

    mode: Literal["X", "O"]
    """The polarization mode of the beam, either 'X' or 'O'."""


@dataclass
class BeamProfile:
    """Dataclass for a traced beam profile."""

    position: jt.Float[jax.Array, "npoints 3"]
    """The position of the beam in cartesian coordinates."""

    arc_length: jt.Float[jax.Array, "npoints"]
    """The arc length along the beam."""

    refractive_index: jt.Float[jax.Array, "npoints 3"]
    """The refractive index at each point along the beam."""

    optical_depth: jt.Float[jax.Array, "npoints"]
    """The optical depth along the beam."""

    absorption_coefficient: jt.Float[jax.Array, "npoints"]
    """The absorption coefficient along the beam."""

    electron_density: jt.Float[jax.Array, "npoints"]
    """The electron density along the beam in units of 10^20 m^-3."""

    electron_temperature: jt.Float[jax.Array, "npoints"]
    """The electron temperature along the beam in keV."""

    magnetic_field: jt.Float[jax.Array, "npoints 3"]
    """The magnetic field vector along the beam in T."""

    normalized_effective_radius: jt.Float[jax.Array, "npoints"]
    """The normalized effective minor radius (rho) along the beam."""

    linear_power_density: jt.Float[jax.Array, "npoints"]
    """The linear power density along the beam."""


@dataclass
class RadialProfile:
    """The deposition profile projected onto the radial coordinate."""

    rho: jt.Float[jax.Array, "npoints"]
    """The normalized effective minor radius."""

    volumetric_power_density: jt.Float[jax.Array, "npoints"]
    """The volumetric power density in W/m³."""


@dataclass
class TracingResult:
    """The result of a ray tracing calculation."""

    beam_profile: BeamProfile
    """The traced beam profile."""

    radial_profile: RadialProfile
    """The radial deposition profile."""


@dataclass
class MagneticConfiguration:
    """Magnetic configuration and geometry on a cylindrical grid.

    This dataclass contains the interpolation grid data and provides cached
    interpolator properties. Interpolators are cached as properties to avoid
    JAX recompilation overhead (~1-2s) when the same object is reused across
    multiple trace() calls.

    The dataclass is designed to be pytree-compatible (can register with
    jax.tree_util in the future if needed for transformations). Cached
    interpolators are excluded from pytree structure via field configuration.
    """

    rphiz: jt.Float[jax.Array, "npoints 3"]
    """The (r, phi, z) coordinates of the points on the interpolation grid."""

    magnetic_field: jt.Float[jax.Array, "npoints 3"]
    """The magnetic field at each point on the interpolation grid."""

    rho: jt.Float[jax.Array, "npoints"]
    """The normalized effective minor radius at each point on the interpolation grid."""

    nfp: int
    """Number of field periods (toroidal periodicity)."""

    stellarator_symmetric: bool
    """Whether the configuration has stellarator symmetry."""

    rho_1d: jt.Float[jax.Array, "nrho_1d"]
    """1D radial grid for volume derivative."""

    dvolume_drho: jt.Float[jax.Array, "nrho_1d"]
    """Volume derivative dV/drho on the 1D radial grid."""

    # Private cached interpolators (not part of pytree)
    _magnetic_field_interpolator: Callable | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _rho_interpolator: Callable | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def magnetic_field_interpolator(
        self,
    ) -> Callable[[jt.Float[jax.Array, "3"]], jt.Float[jax.Array, "3"]]:
        """Magnetic field interpolator B(x,y,z) (cached on first access).

        The interpolator is built once and cached. Returning the same callable
        object on subsequent accesses allows JAX to reuse its compiled trace,
        avoiding ~1-2s recompilation overhead per trace() call.
        """
        if self._magnetic_field_interpolator is None:
            from .interpolate import build_magnetic_field_interpolator

            object.__setattr__(
                self,
                "_magnetic_field_interpolator",
                build_magnetic_field_interpolator(self),
            )
        assert self._magnetic_field_interpolator is not None
        return self._magnetic_field_interpolator

    @property
    def rho_interpolator(
        self,
    ) -> Callable[[jt.Float[jax.Array, "3"]], jt.Float[jax.Array, ""]]:
        """Normalized effective radius interpolator rho(x,y,z) (cached on first access).

        The interpolator is built once and cached. Returning the same callable
        object on subsequent accesses allows JAX to reuse its compiled trace,
        avoiding ~1-2s recompilation overhead per trace() call.
        """
        if self._rho_interpolator is None:
            from .interpolate import build_rho_interpolator

            object.__setattr__(self, "_rho_interpolator", build_rho_interpolator(self))
        assert self._rho_interpolator is not None
        return self._rho_interpolator


@dataclass
class RadialProfiles:
    """Dataclass for holding the electron radial profiles.

    This dataclass contains 1D radial profile data and provides cached
    interpolator properties. Interpolators are cached as properties to avoid
    JAX recompilation overhead (~1-2s) when the same object is reused across
    multiple trace() calls.

    The dataclass is designed to be pytree-compatible (can register with
    jax.tree_util in the future if needed for transformations). Cached
    interpolators are excluded from pytree structure via field configuration.
    """

    rho: jt.Float[jax.Array, "nrho"]
    """The normalized effective minor radius grid."""

    electron_density: jt.Float[jax.Array, "nrho"]
    """The electron density profile in units of 10^20 m^-3."""

    electron_temperature: jt.Float[jax.Array, "nrho"]
    """The electron temperature profile in keV."""

    # Private cached interpolators (not part of pytree)
    _electron_density_interpolator: Callable | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _electron_temperature_interpolator: Callable | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def electron_density_interpolator(
        self,
    ) -> Callable[[jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]]:
        """Electron density interpolator ne(rho) (cached on first access).

        The interpolator is built once and cached. Returning the same callable
        object on subsequent accesses allows JAX to reuse its compiled trace,
        avoiding ~1-2s recompilation overhead per trace() call.
        """
        if self._electron_density_interpolator is None:
            from .interpolate import build_electron_density_profile_interpolator

            object.__setattr__(
                self,
                "_electron_density_interpolator",
                build_electron_density_profile_interpolator(self),
            )
        assert self._electron_density_interpolator is not None
        return self._electron_density_interpolator

    @property
    def electron_temperature_interpolator(
        self,
    ) -> Callable[[jt.Float[jax.Array, ""]], jt.Float[jax.Array, ""]]:
        """Electron temperature interpolator Te(rho) (cached on first access).

        The interpolator is built once and cached. Returning the same callable
        object on subsequent accesses allows JAX to reuse its compiled trace,
        avoiding ~1-2s recompilation overhead per trace() call.
        """
        if self._electron_temperature_interpolator is None:
            from .interpolate import build_electron_temperature_profile_interpolator

            object.__setattr__(
                self,
                "_electron_temperature_interpolator",
                build_electron_temperature_profile_interpolator(self),
            )
        assert self._electron_temperature_interpolator is not None
        return self._electron_temperature_interpolator
