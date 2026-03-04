"""Visualization functions for Raytrax."""

import numpy as np

from raytrax.equilibrium.interpolate import MagneticConfiguration
from raytrax.types import BeamProfile


def plot_flux_surface_3d(
    magnetic_configuration: MagneticConfiguration,
    rho_value: float = 1.0,
    plotter=None,
    **kwargs,
):
    """Plot a 3D flux surface using PyVista.

    Args:
        magnetic_configuration: The magnetic configuration object.
        rho_value: The value of rho to plot (default: 1.0 for LCFS).
        plotter: Optional PyVista plotter to add the mesh to. If None, creates a new one.
        **kwargs: Additional keyword arguments passed to plotter.add_mesh().

    Returns:
        A PyVista plotter object. Call .show() to display.
    """
    import pyvista as pv

    grid = magnetic_configuration.to_pyvista_grid()

    rho = grid["rho"].copy()
    rho[~np.isfinite(rho)] = rho_value + 0.5
    grid["rho"] = rho

    contour = grid.contour(isosurfaces=[rho_value], scalars="rho")

    if plotter is None:
        plotter = pv.Plotter(notebook=True)
        plotter.add_axes()
        plotter.view_isometric()

    mesh_kwargs = {"color": "lightblue", "opacity": 0.8, "smooth_shading": True}
    mesh_kwargs.update(kwargs)

    plotter.add_mesh(contour, **mesh_kwargs)

    return plotter


def plot_b_surface_3d(
    magnetic_configuration: MagneticConfiguration,
    b_value: float,
    plotter=None,
    **kwargs,
):
    """Plot a 3D surface of constant magnetic field magnitude |B| using PyVista.

    Useful for visualising the electron-cyclotron resonance layer, e.g. the
    2nd-harmonic resonance surface at
    ``B = f_0 / (2 × 27.99 GHz/T)``.

    Args:
        magnetic_configuration: The magnetic configuration object.
        b_value: The |B| value (T) at which to extract the isosurface.
        plotter: Optional PyVista plotter to add the mesh to.
            If None, a new one is created.
        **kwargs: Additional keyword arguments forwarded to
            ``plotter.add_mesh()`` (e.g. ``color``, ``opacity``).

    Returns:
        A PyVista plotter object. Call ``.show()`` to display.
    """
    import pyvista as pv

    grid = magnetic_configuration.to_pyvista_grid()

    # Fill NaN values so clip_scalar and contour work correctly:
    # rho=2.0 → outside-LCFS cells are removed by clip_scalar(value=1.0)
    # absB=0.0 → safely below any realistic b_value, so those cells are inert
    rho = grid["rho"].copy()
    rho[~np.isfinite(rho)] = 2.0
    grid["rho"] = rho

    absB = grid["absB"].copy()
    absB[~np.isfinite(absB)] = 0.0
    grid["absB"] = absB

    # clip_scalar cuts along rho=1 and interpolates scalar values at the cut,
    # unlike threshold which only removes whole cells.  After clipping, the B
    # isosurface terminates naturally at the LCFS edge without an artificial cap.
    # invert=True: clip (remove) values above 1.0, keeping rho < 1 (inside plasma)
    inside = grid.clip_scalar(scalars="rho", invert=True, value=1.0)
    contour = inside.contour(isosurfaces=[b_value], scalars="absB")
    # Taubin smoothing removes faceting from the coarse cylindrical grid without
    # shrinking the surface (unlike Laplacian smoothing).
    contour = contour.smooth_taubin(n_iter=50, pass_band=0.1)

    if plotter is None:
        plotter = pv.Plotter(notebook=True)
        plotter.add_axes()
        plotter.view_isometric()

    mesh_kwargs = {"color": "gold", "opacity": 0.55, "smooth_shading": True}
    mesh_kwargs.update(kwargs)

    plotter.add_mesh(contour, **mesh_kwargs)

    return plotter


def plot_beam_profile_3d(
    beam_profile: BeamProfile,
    plotter=None,
    tube_radius=0.01,
    n_spline_points=100,
    **kwargs,
):
    """Plot a beam profile as a 3D tube colored by linear power density.

    Args:
        beam_profile: The beam profile to plot.
        plotter: Optional PyVista plotter to add the mesh to. If None, creates a new one.
        tube_radius: Radius of the tube in meters (default: 0.01).
        n_spline_points: Number of points for spline interpolation (default: 100).
        **kwargs: Additional keyword arguments passed to plotter.add_mesh().

    Returns:
        A PyVista plotter object. Call .show() to display.
    """
    import pyvista as pv

    # Get beam data
    position = np.array(beam_profile.position)
    power_density = np.array(beam_profile.linear_power_density)

    # Create spline through beam points
    spline = pv.Spline(position, n_spline_points)

    # Interpolate power density onto spline points
    from scipy.interpolate import interp1d

    arc_length = np.array(beam_profile.arc_length)
    power_interp = interp1d(
        arc_length,
        power_density,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    # Calculate arc length for spline points
    spline_points = spline.points
    cumulative_dist = np.zeros(len(spline_points))
    for i in range(1, len(spline_points)):
        cumulative_dist[i] = cumulative_dist[i - 1] + np.linalg.norm(
            spline_points[i] - spline_points[i - 1]
        )

    # Interpolate power density to spline
    power_on_spline = power_interp(cumulative_dist)

    # Add scalar data to spline first, then create tube
    spline["linear_power_density"] = power_on_spline
    tube = spline.tube(radius=tube_radius)

    # Create plotter if not provided
    if plotter is None:
        plotter = pv.Plotter(notebook=True)
        plotter.add_axes()
        plotter.view_isometric()

    # Default mesh styling
    mesh_kwargs = {
        "scalars": "linear_power_density",
        "cmap": "plasma",
        "show_scalar_bar": True,
        "scalar_bar_args": {"title": "Linear Power Density [W/m]"},
    }
    mesh_kwargs.update(kwargs)

    plotter.add_mesh(tube, **mesh_kwargs)

    return plotter
