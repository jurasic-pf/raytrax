---
icon: lucide/rocket
---

# Getting Started

## Installation

To install Raytrax, simply use `pip`:

```bash
python -m pip install raytrax
```

We recommend using a virtual environment to manage your dependencies.

## Your First Trace

Before starting to simulate fusion heating, it's important to understand some of the technical characteristics of Raytrax. Since it's based on [JAX](https://docs.jax.dev), it profits from just-in-time compilation and automatic differentiability. This comes with a cost: the first invocation of the `trace` command will be fairly slow since the computation needs to be compiled first. That's why it wouldn't make much sense to use Raytrax as a command line script. Instead, it's meant to be used in a Python program where 

### Prepare Inputs

Raytrax requires three inputs:

1. A magnetic configuration
2. Radial plasma profiles
3. Beam settings.

The **magnetic configuration** is a grid in cylindrical coordinates holding the values of the magnetic field $\vec{B}$, the effective minor radius $\rho$, and some other geometric quantities. Raytrax can compute this configuration from a [VMEC++](https://proximafusion.github.io/vmecpp/) MHD equilibrium.

An example where an equilibrium is loaded from a NetCDF file:

```python
import raytrax, vmecpp

vmec_wout = vmecpp.VmecWOut.from_wout_file("w7x.nc")
mag_conf = raytrax.MagneticConfiguration.from_vmec_wout(vmec_wout)
```

You can save the configuration to a file and load it back with the object's `.save` and `.load` methods.

The **radial plasma profiles** are gridded one-dimensional profiles for the electron density $n_e$ (in units of 10<sup>20</sup>/m³) and temperature $T_e$ (in units of keV) as a function of the effective minor radius $\rho$, which must extend from 0 to 1. You should ensure that both density and temperature are zero at the plasma boundary ($\rho$). Example:

```python
import raytrax, jax.numpy as jnp

rho = jnp.linspace(0, 1, 40)
n_e = 1.0 * (1 - rho**2)
T_e = 2.0 * (1 - rho**1.5)
profiles = raytrax.RadialProfiles(
    rho=rho,
    electron_density=n_e,
    electron_temperature=T_e
)
```

The **beam settings** define the properties of the microwave beam to be traced: its starting position (a vector in Cartesian coordinates), initial direction (a unit 3-vector), frequency (in Hz, not GHz!), wave mode (ordinary or extraordinary mode), and initial power (in W). Example:

```python
import raytrax, jax.numpy as jnp

beam = raytrax.Beam(
    position=jnp.array([1.0, 2.0, 3.0]),
    direction=jnp.array([0.0, -1.0, 0.0]),
    frequency=140e9,  # Hz!
    mode="O",
    power=1e6,  # W
)
```

### Trace

Once the inputs are ready, you can run the ray tracer:

```python
import raytrax

tracing_result = raytrax.trace(
    magnetic_configuration=mag_conf,
    radial_profiles=profiles,
    beam=beam,
)
```

The `TracingResult` object holds two properties:

- The `beam_profile` contains the quantities along the beam's trajectory through Cartesian space
- The `radial_profile` contains the quantities projected unto the radial coordinate $\rho$