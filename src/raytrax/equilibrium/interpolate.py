r"""Utilities for interpolating flux surface quantities on a cylindrical grid.

Since ray tracing is computed in cartesian space (which can easily be transformed
to/from cylindrical coordinates), the effective minor radius $\rho$ and the magnetic
field $B$ need to be extrapolated from toroidal coordinates ($\rho$, $\theta$, $\phi$)
to cylindrical coordinates ($r$, $\phi$, $z$).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv

import interpax
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from beartype import beartype as typechecker
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from raytrax.equilibrium.protocol import WoutLike
from raytrax.types import RadialProfiles, SafetensorsMixin

from .fourier import dvolume_drho as compute_dvolume_drho


@dataclass
class CylindricalGridResolution:
    """Output cylindrical grid resolution, shared by all equilibrium importers.

    Defines the $(R, \\phi, Z)$ cylindrical grid on which the magnetic
    configuration is stored and the ray tracer interpolates. For axisymmetric
    (tokamak) configurations ``n_phi`` is ignored.

    Attributes:
        n_r: Number of grid points along the major radius $R$.
        n_z: Number of grid points along the vertical coordinate $Z$.
        n_phi: Number of toroidal planes $\\phi$. Ignored for axisymmetric
            configurations.
        n_rho_profile: Number of radial points for the 1-D $dV/d\\rho$ profile.
    """

    n_r: int = 95
    n_z: int = 105
    n_phi: int = 50
    n_rho_profile: int = 200


@dataclass
class VmecGridResolution:
    """Grid resolution for VMEC-based equilibrium imports.

    Combines the shared :class:`CylindricalGridResolution` output grid with
    VMEC-specific intermediate flux-coordinate sampling parameters.

    The VMEC pipeline first evaluates Fourier series on an intermediate
    curvilinear $(\\rho, \\theta, \\phi)$ grid, then scatter-interpolates the
    result onto the cylindrical output grid. The intermediate grid parameters
    ``n_rho`` and ``n_theta`` control the accuracy of that Fourier evaluation
    step; they are invisible to the ray tracer.

    Attributes:
        cylindrical: Output cylindrical grid shared with all other importers.
        n_rho: Number of radial (flux-surface) points on the intermediate
            VMEC curvilinear grid.
        n_theta: Number of poloidal points on the intermediate VMEC
            curvilinear grid.
    """

    cylindrical: CylindricalGridResolution = dataclass_field(
        default_factory=CylindricalGridResolution
    )
    n_rho: int = 40
    n_theta: int = 45


@jax.tree_util.register_dataclass
@dataclass
class MagneticConfiguration(SafetensorsMixin):
    r"""Magnetic configuration and geometry on a cylindrical grid.

    Contains the magnetic field $\boldsymbol{B}$ and normalized effective radius $\rho$ on a
    3D cylindrical grid ($R$, $\phi$, $Z$), along with volume information for
    computing deposition profiles.
    """

    rphiz: jt.Float[jax.Array, "npoints 3"]
    r"""The ($R$, $\phi$, $Z$) coordinates of the points on the interpolation grid."""

    magnetic_field: jt.Float[jax.Array, "npoints 3"]
    r"""The magnetic field $(B_R, B_\phi, B_Z)$ in cylindrical components at each grid point."""

    rho: jt.Float[jax.Array, " npoints"]
    """The normalized effective minor radius at each point on the interpolation grid."""

    nfp: int = dataclass_field(metadata={"static": True})
    """Number of field periods (toroidal periodicity)."""

    is_stellarator_symmetric: bool = dataclass_field(metadata={"static": True})
    """Whether the configuration has stellarator symmetry."""

    rho_1d: jt.Float[jax.Array, " nrho_1d"]
    """1D radial grid for volume derivative."""

    dvolume_drho: jt.Float[jax.Array, " nrho_1d"]
    r"""Volume derivative $dV/d\rho$ on the 1D radial grid."""

    is_axisymmetric: bool = dataclass_field(default=False, metadata={"static": True})
    """Whether the configuration is axisymmetric (tokamak) or 3D (stellarator)."""

    @classmethod
    def from_vmec_wout(
        cls,
        equilibrium: WoutLike,
        magnetic_field_scale: float = 1.0,
        grid: VmecGridResolution | None = None,
    ) -> MagneticConfiguration:
        """Create a MagneticConfiguration from a VMEC++ equilibrium.

        Optionally, apply a uniform scale factor to the magnetic field magnitude
        with respect to the equilibrium data.

        Args:
            equilibrium: an MHD equilibrium compatible with `vmecpp.VmecWOut`
            magnetic_field_scale: Factor to multiply all magnetic field values by.
            grid: Grid resolution settings. Defaults to
                :class:`VmecGridResolution` with its default values.

        Returns:
            A MagneticConfiguration object containing interpolation data.
        """
        if grid is None:
            grid = VmecGridResolution()

        interpolated_array = cylindrical_grid_for_equilibrium(
            equilibrium=equilibrium,
            n_rho=grid.n_rho,
            n_theta=grid.n_theta,
            n_phi=grid.cylindrical.n_phi,
            n_r=grid.cylindrical.n_r,
            n_z=grid.cylindrical.n_z,
        )
        rphiz = interpolated_array[..., :3]
        rho = interpolated_array[..., 3]
        magnetic_field = interpolated_array[..., 4:] * magnetic_field_scale

        # Compute volume derivative on 1D radial grid
        rho_1d = jnp.linspace(0, 1, grid.cylindrical.n_rho_profile)
        dv_drho = compute_dvolume_drho(equilibrium, rho_1d)

        return cls(
            rphiz=rphiz,
            magnetic_field=magnetic_field,
            rho=rho,
            nfp=equilibrium.nfp,
            is_stellarator_symmetric=not equilibrium.lasym,
            rho_1d=rho_1d,
            dvolume_drho=dv_drho,
        )

    def to_pyvista_grid(self) -> pv.StructuredGrid:
        """Convert to a PyVista StructuredGrid in Cartesian coordinates.

        Point arrays written:

        - ``rho``: normalised effective minor radius (NaN outside LCFS)
        - ``B``: magnetic field vector :math:`(B_R, B_\\phi, B_Z)` in T
        - ``absB``: magnetic field magnitude :math:`|\\boldsymbol{B}|` in T

        Requires the ``pyvista`` package.

        Returns:
            A :class:`pyvista.StructuredGrid` that can be visualised or saved
            directly via ``.save("output.vts")``.
        """
        import pyvista as pv

        rphiz = np.array(self.rphiz)
        R, phi, Z = rphiz[..., 0], rphiz[..., 1], rphiz[..., 2]

        B = np.array(self.magnetic_field)
        rho = np.array(self.rho)

        grid = pv.StructuredGrid(R * np.cos(phi), R * np.sin(phi), Z)
        grid["rho"] = rho.flatten(order="F")  # NaN outside LCFS
        grid["B"] = B.reshape(-1, 3, order="F")
        grid["absB"] = np.linalg.norm(B, axis=-1).flatten(order="F")

        return grid


@jt.jaxtyped(typechecker=typechecker)
def interpolate_toroidal_to_cylindrical_grid(
    rphiz_toroidal: jt.Float[jax.Array, "n_rho n_theta n_phi rphiz=3"],
    rz_cylindrical: jt.Float[jax.Array, "n_r n_z rz=2"],
    value_toroidal: jt.Float[jax.Array, "n_rho n_theta n_phi n_values"],
) -> jt.Float[jax.Array, "n_r n_phi n_z n_values"]:
    """Interpolate toroidal coordinates to a cylindrical grid.

    Args:
    - rphiz_toroidal: An array of cylindrical coordinates (the last dimension
        are the three coordinates R, phi, Z) on the toroidal grid.
    - rz_cylindrical: The cylindrical coordinates (R, Z) on the cylindrical grid.
        Note that phi does not need to be provided because we use the same phi
        grid as for the toroidal grid, so no interpolation in phi is needed.
    - value_toroidal: The values of the quantities to be interpolated on the
        toroidal grid nodes.

    Returns:
    The values of the quantities interpolated to the cylindrical grid as a JAX
    array.
    """
    n_r, n_z, _ = rz_cylindrical.shape
    n_values = value_toroidal.shape[-1]
    values = []
    phis = rphiz_toroidal[0, 0, :, 1]
    for i_phi, _ in enumerate(phis):
        # this uses plain numpy, because griddata is anyway not JAX compatible
        value = griddata(
            np.array(rphiz_toroidal[:, :, i_phi, ::2]).reshape(-1, 2),
            np.array(value_toroidal[:, :, i_phi]).reshape(-1, n_values),
            np.array(rz_cylindrical).reshape(-1, 2),
            method="linear",
        ).reshape(n_r, n_z, n_values)
        values.append(value)
    return jnp.stack(values, axis=1)


# Absolute (R, Z) residual in metres: the round-off level of the Fourier sums.
_NEWTON_TOLERANCE = 1e-12
_NEWTON_MAX_ITERATIONS = 30


@dataclass(frozen=True)
class _PlaneHarmonics:
    """VMEC geometry and field per phi plane as complex theta harmonics a_m(rho):
    a cosine series is Re sum_m a_m exp(i m theta), a sine series the Im part."""

    sqrt_s_full: np.ndarray
    sqrt_s_half: np.ndarray
    # Cubic pieces in rho per plane, interval, power, quantity, m.
    geometry: np.ndarray  # (R, Z)
    geometry_phi: np.ndarray  # (dR/dphi, dZ/dphi)
    field: np.ndarray  # (B^theta, B^phi)

    @classmethod
    def of(cls, equilibrium: WoutLike, phis: np.ndarray) -> _PlaneHarmonics:
        ns = equilibrium.ns
        ds = 1 / (ns - 1)
        sqrt_s_full = np.sqrt(np.linspace(0, 1, ns))
        sqrt_s_half = np.sqrt(np.linspace(ds / 2, 1 - ds / 2, ns - 1))
        xm, xn = np.asarray(equilibrium.xm), np.asarray(equilibrium.xn)
        rz = [np.asarray(equilibrium.rmnc), np.asarray(equilibrium.zmns)]
        b = [np.asarray(c)[:, 1:] for c in (equilibrium.bsupumnc, equilibrium.bsupvmnc)]
        return cls(
            sqrt_s_full=sqrt_s_full,
            sqrt_s_half=sqrt_s_half,
            geometry=_cubic_pieces(sqrt_s_full, _harmonics(rz, xm, xn, phis, 1)),
            geometry_phi=_cubic_pieces(
                sqrt_s_full, _harmonics(rz, xm, xn, phis, -1j * xn)
            ),
            field=_cubic_pieces(
                sqrt_s_half,
                _harmonics(
                    b,
                    np.asarray(equilibrium.xm_nyq),
                    np.asarray(equilibrium.xn_nyq),
                    phis,
                    1,
                ),
            ),
        )

    def rz(
        self, plane: int, rho: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """R, Z, dR/drho, dR/dtheta, dZ/drho, dZ/dtheta, with the C¹ linear
        continuation beyond rho = 1 that `evaluate_rphiz_on_toroidal_grid` uses."""
        rho_inside = np.minimum(rho, 1.0)
        a, a_rho = _cubic(self.sqrt_s_full, self.geometry[plane], rho_inside)
        a = a + (rho - rho_inside)[:, None, None] * a_rho
        m = np.arange(a.shape[-1])
        e = np.exp(1j * np.outer(theta, m))[:, None]
        f, f_rho, f_theta = ((c * e).sum(-1) for c in (a, a_rho, 1j * m * a))
        return (
            f[:, 0].real,
            f[:, 1].imag,
            f_rho[:, 0].real,
            f_theta[:, 0].real,
            f_rho[:, 1].imag,
            f_theta[:, 1].imag,
        )

    def b_cylindrical(
        self, plane: int, rho: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """(B_R, B_phi, B_Z) with B held at its LCMS value beyond rho = 1."""
        rho = np.minimum(rho, 1.0)
        r, _, _, r_theta, _, z_theta = self.rz(plane, rho, theta)
        rz_phi = _series(self.sqrt_s_full, self.geometry_phi[plane], rho, theta)
        b_theta, b_phi = _series(self.sqrt_s_half, self.field[plane], rho, theta).real.T
        return np.stack(
            [
                b_theta * r_theta + b_phi * rz_phi[:, 0].real,
                b_phi * r,
                b_theta * z_theta + b_phi * rz_phi[:, 1].imag,
            ],
            axis=-1,
        )


def _harmonics(
    coefficients: list[np.ndarray],
    xm: np.ndarray,
    xn: np.ndarray,
    phis: np.ndarray,
    weight: complex | np.ndarray,
) -> np.ndarray:
    """a_m per plane, knot and quantity with
    sum_k weight_k c_k exp(i (m_k theta - n_k phi)) = sum_m a_m exp(i m theta)."""
    phase = weight * np.exp(-1j * np.outer(phis, xn))
    one_hot = np.eye(xm.max() + 1)[xm]
    return np.einsum(
        "qkj,pk,km->pjqm", np.stack(coefficients), phase, one_hot, optimize=True
    )


def _cubic_pieces(knots: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Per interval, the coefficients of t^0..t^3 (t in [0, 1] across the interval)
    of the cubic Hermite spline interpax's "cubic" interp1d builds along axis 1."""
    x = jnp.asarray(knots)
    slopes = np.asarray(
        interpax.approx_df(x, jnp.asarray(values.real), "cubic", axis=1)
    ) + 1j * np.asarray(
        interpax.approx_df(x, jnp.asarray(values.imag), "cubic", axis=1)
    )
    dx = np.diff(knots)[:, None, None]
    f0, f1 = values[:, :-1], values[:, 1:]
    d0, d1 = slopes[:, :-1] * dx, slopes[:, 1:] * dx
    return np.stack(
        [f0, d0, 3 * (f1 - f0) - 2 * d0 - d1, 2 * (f0 - f1) + d0 + d1], axis=2
    )


def _series(
    knots: np.ndarray, pieces: np.ndarray, rho: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    a, _ = _cubic(knots, pieces, rho)
    e = np.exp(1j * np.outer(theta, np.arange(a.shape[-1])))[:, None]
    return (a * e).sum(-1)


def _cubic(
    knots: np.ndarray, pieces: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Spline value and x-derivative, extrapolating the end pieces like interpax."""
    i = np.clip(np.searchsorted(knots, x, side="right"), 1, len(knots) - 1) - 1
    dx = (knots[i + 1] - knots[i])[:, None, None]
    t = (x - knots[i])[:, None, None] / dx
    a0, a1, a2, a3 = np.moveaxis(pieces[i], 1, 0)
    value = ((a3 * t + a2) * t + a1) * t + a0
    derivative = ((3 * a3 * t + 2 * a2) * t + a1) / dx
    return value, derivative


@jt.jaxtyped(typechecker=typechecker)
def cylindrical_grid_for_equilibrium(
    equilibrium: WoutLike, n_rho: int, n_theta: int, n_phi: int, n_r: int, n_z: int
) -> jt.Float[jax.Array, "n_r n_phi n_z rphizrhoBxyz=7"]:
    """Compute the effective minor radius and the magnetic field on a cylindrical grid.

    Args:
        equilibrium: The MHD equilibrium.
        n_rho: Number of radial points.
        n_theta: Number of poloidal points.
        n_phi: Number of toroidal points (equal in both grids!).
        n_r: Number of radial points in the cylindrical grid.
        n_z: Number of vertical points in the cylindrical grid.

    Returns:
        A JAX array of values on the cylindrical grid with shape (n_r, n_phi, n_z, 7).
        The last dimension is a vector with the following components:
        (R, phi, Z, rho, B_R, B_phi, B_Z) where the magnetic field components
        are in cylindrical coordinates.
    """
    if equilibrium.lasym:
        raise NotImplementedError(
            "Non stellarator symmetric equilibria are not supported yet."
        )
    # Each node's flux coordinates are found by Newton on (R, Z)(rho, theta) in the
    # node's phi plane, so rho and B are exact at the nodes rather than interpolated
    # across a triangulation. Per plane the Fourier sums collapse to harmonics in
    # theta; splines are linear in their data, so this reproduces the radial
    # interpolation of the coefficients exactly while costing O(m_max) per point.
    phis = np.linspace(0, np.pi / equilibrium.nfp, n_phi)
    harmonics = _PlaneHarmonics.of(equilibrium, phis)
    rho_max = 1.2  # extend beyond the LCMS to allow extrapolation
    rho_sample, theta_sample = np.meshgrid(
        np.linspace(0, rho_max, n_rho), np.linspace(0, 2 * np.pi, n_theta)
    )
    sample = [
        harmonics.rz(p, rho_sample.ravel(), theta_sample.ravel())[:2]
        for p in range(n_phi)
    ]
    r_grid = np.linspace(
        min(r.min() for r, _ in sample), max(r.max() for r, _ in sample), n_r
    )
    z_grid = np.linspace(
        min(z.min() for _, z in sample), max(z.max() for _, z in sample), n_z
    )
    r_nodes, z_nodes = (a.ravel() for a in np.meshgrid(r_grid, z_grid, indexing="ij"))
    values = np.full((n_phi, n_r * n_z, 4), np.nan)
    for p, (r, z) in enumerate(sample):
        _, nearest = cKDTree(np.stack([r, z], -1)).query(
            np.stack([r_nodes, z_nodes], -1)
        )
        rho, theta = _invert(
            harmonics,
            p,
            r_nodes,
            z_nodes,
            rho_sample.ravel()[nearest],
            theta_sample.ravel()[nearest],
            rho_max,
        )
        inside = np.isfinite(rho)
        values[p, inside, 0] = rho[inside]
        values[p, inside, 1:] = harmonics.b_cylindrical(p, rho[inside], theta[inside])
    rphiz = np.stack(np.meshgrid(r_grid, phis, z_grid, indexing="ij"), axis=-1)
    values = values.reshape(n_phi, n_r, n_z, 4).transpose(1, 0, 2, 3)
    return jnp.asarray(np.concatenate([rphiz, values], axis=-1))


def _invert(
    geometry: _PlaneHarmonics,
    plane: int,
    r: np.ndarray,
    z: np.ndarray,
    rho: np.ndarray,
    theta: np.ndarray,
    rho_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(rho, theta) of the points (r, z) in a phi plane by Newton; NaN beyond rho_max."""
    # The (rho, theta) map is singular on the axis, so start Newton just off it.
    rho, theta = np.maximum(rho, 1e-3), theta.copy()
    active = np.arange(len(r))
    residual = np.full(len(r), np.inf)
    for _ in range(_NEWTON_MAX_ITERATIONS):
        rr, zz, r_rho, r_theta, z_rho, z_theta = geometry.rz(
            plane, rho[active], theta[active]
        )
        dr, dz = rr - r[active], zz - z[active]
        residual[active] = np.hypot(dr, dz)
        det = r_rho * z_theta - r_theta * z_rho
        # A singular step makes the point non-finite, which drops it as unconverged.
        with np.errstate(divide="ignore", invalid="ignore"):
            rho[active] -= (z_theta * dr - r_theta * dz) / det
            theta[active] -= (r_rho * dz - z_rho * dr) / det
        # A step through the axis lands at rho < 0; (|rho|, theta + pi) is the same
        # point to leading order in rho.
        flip = rho[active] < 0
        rho[active[flip]] *= -1
        theta[active[flip]] += np.pi
        keep = (residual[active] > _NEWTON_TOLERANCE) & (rho[active] < rho_max + 0.1)
        active = active[keep]
        if active.size == 0:
            break
    converged = (residual <= _NEWTON_TOLERANCE) & (rho <= rho_max)
    return np.where(converged, rho, np.nan), theta


def build_magnetic_field_interpolator(
    magnetic_configuration: MagneticConfiguration,
) -> interpax.Interpolator3D | interpax.Interpolator2D:
    """Build a magnetic field interpolator from the magnetic configuration.

    For stellarators, returns an interpax.Interpolator3D that interpolates
    (B_R, B_phi, B_Z) in cylindrical coordinates (R, phi, Z) on the
    fundamental domain [0, pi/nfp].

    For axisymmetric equilibria, returns an interpax.Interpolator2D that
    interpolates (B_R, B_phi, B_Z) as a function of (R, Z).
    """
    if magnetic_configuration.is_axisymmetric:
        B_cyl = magnetic_configuration.magnetic_field
        rphiz = magnetic_configuration.rphiz
        return interpax.Interpolator2D(
            x=rphiz[:, 0, 0, 0],
            y=rphiz[0, 0, :, 2],
            f=jnp.nan_to_num(B_cyl[:, 0, :, :], nan=0.0),
            method="linear",
            extrap=0.0,
        )

    if not magnetic_configuration.is_stellarator_symmetric:
        raise NotImplementedError(
            "Non stellarator-symmetric equilibria not yet supported"
        )

    B_cyl = magnetic_configuration.magnetic_field
    rphiz = magnetic_configuration.rphiz

    return interpax.Interpolator3D(
        x=rphiz[:, 0, 0, 0],
        y=rphiz[0, :, 0, 1],
        z=rphiz[0, 0, :, 2],
        f=jnp.nan_to_num(B_cyl, nan=0.0),
        method="linear",
        extrap=0.0,
    )


def build_rho_interpolator(
    magnetic_configuration: MagneticConfiguration,
) -> interpax.Interpolator3D | interpax.Interpolator2D:
    """Build rho interpolator for the given magnetic configuration.

    For stellarators, returns an interpax.Interpolator3D that interpolates rho
    in cylindrical coordinates (R, phi, Z) on the fundamental domain [0, pi/nfp].

    For axisymmetric equilibria, returns an interpax.Interpolator2D that
    interpolates rho as a function of (R, Z).
    """
    if magnetic_configuration.is_axisymmetric:
        rho = magnetic_configuration.rho
        rphiz = magnetic_configuration.rphiz
        return interpax.Interpolator2D(
            x=rphiz[:, 0, 0, 0],
            y=rphiz[0, 0, :, 2],
            f=jnp.nan_to_num(rho[:, 0, :], nan=1.1),
            method="linear",
            extrap=1.1,
        )

    if not magnetic_configuration.is_stellarator_symmetric:
        raise NotImplementedError(
            "Non stellarator-symmetric equilibria not yet supported"
        )

    rho = magnetic_configuration.rho
    rphiz = magnetic_configuration.rphiz

    return interpax.Interpolator3D(
        x=rphiz[:, 0, 0, 0],
        y=rphiz[0, :, 0, 1],
        z=rphiz[0, 0, :, 2],
        f=jnp.nan_to_num(rho, nan=1.1),
        method="linear",
        extrap=1.1,
    )


_VACUUM_RAMP_WIDTH = 0.02  # width in rho of the density ramp beyond the profile grid
_VACUUM_RAMP_POINTS = 9  # number of knots resolving the ramp


def build_electron_density_profile_interpolator(
    radial_profiles: RadialProfiles,
) -> interpax.Interpolator1D:
    r"""Build electron density profile interpolator.

    Beyond the last grid point $\rho_\mathrm{edge}$ (normally the LCFS), the
    density decreases from $n_e(\rho_\mathrm{edge})$ to zero over a thin layer of
    width $\Delta\rho = 0.02$, so that rays are refracted at the plasma–vacuum
    interface for profiles with a finite edge density. The ramp is sampled from a
    cubic smoothstep and linearly interpolated, like the profile inside the grid,
    which is unchanged.

    Args:
        radial_profiles: The radial profiles.

    Returns:
        An interpax.Interpolator1D that maps rho to electron density.
    """
    rho = radial_profiles.rho
    ne = radial_profiles.electron_density
    s = jnp.linspace(0.0, 1.0, _VACUUM_RAMP_POINTS + 1)[1:]
    rho_ramp = rho[-1] + _VACUUM_RAMP_WIDTH * s
    ne_ramp = ne[-1] * (1.0 - s**2 * (3.0 - 2.0 * s))

    return interpax.Interpolator1D(
        x=jnp.concatenate([rho, rho_ramp]),
        f=jnp.concatenate([ne, ne_ramp]),
        method="linear",
        extrap=0.0,
    )


def build_electron_temperature_profile_interpolator(
    radial_profiles: RadialProfiles,
) -> interpax.Interpolator1D:
    """Build electron temperature profile interpolator.

    Args:
        radial_profiles: The radial profiles.

    Returns:
        An interpax.Interpolator1D that maps rho to electron temperature.
    """
    return interpax.Interpolator1D(
        x=radial_profiles.rho,
        f=radial_profiles.electron_temperature,
        method="linear",
        extrap=0.0,
    )


def build_radial_interpolators(
    magnetic_configuration: MagneticConfiguration,
    radial_profiles: RadialProfiles,
) -> tuple[
    interpax.Interpolator3D | interpax.Interpolator2D,
    interpax.Interpolator1D,
    interpax.Interpolator1D,
]:
    """Build radial interpolators for the given magnetic configuration and radial profiles.

    This is a convenience function that combines the individual interpolator builders.
    For cleaner code, consider using the individual functions directly.

    Args:
        magnetic_configuration: The magnetic configuration.
        radial_profiles: The radial profiles.

    Returns:
        A tuple of three functions:
        - rho_interpolator: maps position to radial coordinate (rho)
        - electron_density_profile_interpolator: maps rho to electron density
        - electron_temperature_profile_interpolator: maps rho to electron temperature
    """
    rho_interpolator = build_rho_interpolator(magnetic_configuration)
    ne_profile_interpolator = build_electron_density_profile_interpolator(
        radial_profiles
    )
    Te_profile_interpolator = build_electron_temperature_profile_interpolator(
        radial_profiles
    )

    return rho_interpolator, ne_profile_interpolator, Te_profile_interpolator
