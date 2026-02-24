"""Tests for the raytrax.api module."""

import jax.numpy as jnp
import numpy as np

from raytrax.api import trace, _bin_power_deposition
from raytrax.equilibrium.interpolate import MagneticConfiguration
from raytrax.types import Beam, RadialProfiles
from tests.fixtures import w7x_wout


def test_from_vmec_wout_w7x(w7x_wout):
    """Test the MagneticConfiguration.from_vmec_wout classmethod with the W7-X equilibrium."""
    # Generate the interpolator
    interpolator = MagneticConfiguration.from_vmec_wout(w7x_wout)

    # Check that the interpolator has the expected attributes
    assert hasattr(interpolator, "rphiz")
    assert hasattr(interpolator, "magnetic_field")
    assert hasattr(interpolator, "rho")

    # Check the shapes match
    # The actual shape is (n_r, n_phi, n_z, 3) for rphiz and magnetic_field
    # and (n_r, n_phi, n_z) for rho
    assert interpolator.rphiz.shape == (45, 50, 55, 3)
    assert interpolator.magnetic_field.shape == (45, 50, 55, 3)
    assert interpolator.rho.shape == (45, 50, 55)

    # Check that non-NaN rho values are non-negative
    # Note: rho > 1 is expected for points outside the LCMS (in the vacuum region)
    valid_rho = interpolator.rho[~jnp.isnan(interpolator.rho)]
    assert valid_rho.size > 0, "All rho values are NaN"
    assert jnp.all(valid_rho >= 0.0)
    # Check that we have some points inside the plasma (rho < 1)
    assert jnp.any(valid_rho < 1.0), "No points found inside plasma (rho < 1)"
    # Check that we have some points outside the plasma (rho > 1)
    assert jnp.any(valid_rho > 1.0), "No points found outside plasma (rho > 1)"

    # Verify that the magnetic field has non-zero values
    assert not jnp.all(interpolator.magnetic_field == 0.0)

    # Test that the rphiz coordinates make sense physically
    # For W7-X, R (major radius) should be around 5.5m
    r_values = interpolator.rphiz[..., 0]  # First component is r
    assert jnp.min(r_values) > 0.0  # Should be positive
    assert jnp.max(r_values) < 7.0  # Should be less than ~7m for W7-X

    # Z values should be within reasonable range for W7-X
    z_values = interpolator.rphiz[..., 2]  # Third component is z
    assert jnp.min(z_values) > -1.5  # Lower bound
    assert jnp.max(z_values) < 1.5  # Upper bound


def test_trace_w7x_beam(w7x_wout):
    """Test the trace function with W7-X equilibrium and a specific beam position/direction."""
    # Beam position
    r = 6.50866
    phi = np.deg2rad(-6.56378)
    z = 0.38
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    position = jnp.array([x, y, z])

    # Beam direction starting from W7-X aiming angles
    alpha = np.deg2rad(15.7)
    beta = np.deg2rad(19.7001)
    d_r = -np.cos(alpha) * np.cos(beta)
    d_phi = np.cos(alpha) * np.sin(beta)
    d_z = np.sin(alpha)
    dir_x = d_r * np.cos(phi) - d_phi * np.sin(phi)
    dir_y = d_r * np.sin(phi) + d_phi * np.cos(phi)
    dir_z = d_z
    direction = np.array([dir_x, dir_y, dir_z])
    direction = direction / np.linalg.norm(direction)

    rho = np.linspace(0, 1, 45)
    quadratic_profile = (1 - rho) ** 2
    electron_temperature = 5 * quadratic_profile  # 5 keV on axis
    electron_density = 0.75 * quadratic_profile  # 0.75e20/m³ on axis
    radial_profiles = RadialProfiles(
        rho=rho,
        electron_density=electron_density,
        electron_temperature=electron_temperature,
    )

    beam = Beam(
        position=position,
        direction=direction,
        frequency=140e9,
        mode="O",
    )

    interpolator = MagneticConfiguration.from_vmec_wout(w7x_wout)
    result = trace(interpolator, radial_profiles, beam)

    assert hasattr(result, "beam_profile")
    assert hasattr(result, "radial_profile")
    assert result.beam_profile is not None
    assert result.radial_profile is not None


def test_bin_power_deposition():
    """Test the _bin_power_deposition helper function."""
    # Create a regular rho grid
    rho_grid = jnp.linspace(0.0, 1.0, 11)  # 11 points: 0.0, 0.1, ..., 1.0

    # Mock dV/drho (increasing volume with rho)
    dvolume_drho = jnp.linspace(1.0, 2.0, 11)

    # Create a non-monotonic trajectory that crosses same rho values multiple times
    # Simulates beam entering plasma (0.9 → 0.3), then exiting (0.3 → 0.7)
    rho_trajectory = jnp.array([0.9, 0.7, 0.5, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    arc_length = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    linear_power_density = jnp.ones(9) * 10.0  # Constant 10 W/m

    # Bin the power
    result = _bin_power_deposition(
        rho_grid, dvolume_drho, rho_trajectory, arc_length, linear_power_density
    )

    # Check output shape matches rho_grid
    assert result.shape == rho_grid.shape

    # Check that result contains no NaN/inf values
    assert jnp.all(jnp.isfinite(result))

    # Check that rho=0.5 bin receives contributions from TWO segments
    # (crossing rho=0.5 twice means power is summed, not averaged)
    assert result[5] > 0  # Should have deposited power

    # Total power = sum(dP/dV * dV); dV uses the same Voronoi edges as the function
    edges = jnp.concatenate(
        [rho_grid[:1], 0.5 * (rho_grid[:-1] + rho_grid[1:]), rho_grid[-1:]]
    )
    total_deposited = jnp.sum(result * dvolume_drho * jnp.diff(edges))
    expected_total = 10.0 * 0.8  # 10 W/m * 0.8 m = 8 W
    np.testing.assert_allclose(total_deposited, expected_total, rtol=0.05)


def test_bin_power_deposition_power_conservation():
    """Total deposited power must equal integral of dP/ds to within numerical precision.

    With fine-grid upsampling, conservation is tight (rtol~1e-3), not just
    order-of-magnitude as with coarse histograms.
    """
    rho_grid = jnp.linspace(0.0, 1.0, 200)
    dvolume_drho = jnp.ones(200)  # dV/drho = 1 → dV = drho, simplifies check

    # Monotone trajectory: ray enters at rho=0.9 and passes through to rho=0.1
    rho_trajectory = jnp.linspace(0.9, 0.1, 20)
    arc_length = jnp.linspace(0.0, 1.0, 20)
    linear_power_density = jnp.ones(20) * 5.0  # 5 W/m, constant

    result = _bin_power_deposition(
        rho_grid, dvolume_drho, rho_trajectory, arc_length, linear_power_density
    )

    assert result.shape == rho_grid.shape

    # No bin along the traversed rho range [0.1, 0.9] should be zero
    traversed = (rho_grid >= 0.1) & (rho_grid <= 0.9)
    assert jnp.all(
        result[traversed] > 0
    ), "Bins along trajectory path should be populated"

    # Power conservation: sum(dP/dV * dV) = 5 W/m * 1 m = 5 W
    edges = jnp.concatenate(
        [rho_grid[:1], 0.5 * (rho_grid[:-1] + rho_grid[1:]), rho_grid[-1:]]
    )
    total_deposited = jnp.sum(result * dvolume_drho * jnp.diff(edges))
    np.testing.assert_allclose(total_deposited, 5.0, rtol=1e-3)


def test_bin_power_with_nan_values():
    """Test that _bin_power_deposition does not crash with NaN/inf in trajectory."""
    rho_grid = jnp.linspace(0.0, 1.0, 11)
    dvolume_drho = jnp.ones(11)

    # Trajectory with some NaN values
    rho_trajectory = jnp.array([0.5, jnp.nan, 0.7, 0.5, jnp.inf])
    arc_length = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4])
    linear_power_density = jnp.array([2.0, 3.0, 4.0, jnp.nan, 5.0])

    result = _bin_power_deposition(
        rho_grid, dvolume_drho, rho_trajectory, arc_length, linear_power_density
    )
    assert result.shape == rho_grid.shape
