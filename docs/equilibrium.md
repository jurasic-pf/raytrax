---
icon: lucide/magnet
---

# Loading a MHD Equilibrium

One of the inputs to Raytrax is a magnetic configuration, which is a grid in cylindrical coordinates holding the values of the magnetic field $\vec{B}$, the effective minor radius $\rho$, and some other geometric quantities. Raytrax can compute this configuration a magnetohydrodynamic (MHD) equilibrium.

At present, Raytrax supports loading equilibria from the [VMEC++](https://proximafusion.github.io/vmecpp/) code, which is a modern implementation of the widely used VMEC code for computing stellarator equilibria.

!!! info
    Support for Tokamak equilibria is planned for a future release.

### Using VMEC++

To instantiate a magnetic configuration from a VMEC++ equilibrium, you need a `vmecpp.VmecWOut` object, which can be either obtained by running VMEC++ on an input file, or by loading an existing equilibrium file in NetCDF "wout" format (created with VMEC++ or the original VMEC).

Example for loading an existing wout file:

```python
import vmecpp

vmec_wout = vmecpp.VmecWOut.from_wout_file("w7x.nc")
```

Alternatively, you can run VMEC++ on an input file yourself:

```python
import vmecpp

vmec_input = vmecpp.VmecInput.from_file("input.w7x")
vmec_output = vmecpp.run(vmec_input)
vmec_wout = vmec_output.wout
```

For more options and details, see the [VMEC++ documentation](https://proximafusion.github.io/vmecpp/).


Once you have the `VmecWOut` object, you can create a `raytrax.MagneticConfiguration` from it:

```python
mag_conf = raytrax.MagneticConfiguration.from_vmec_wout(vmec_wout)
```