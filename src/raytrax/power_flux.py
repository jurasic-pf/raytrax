import jax
import jax.numpy as jnp
import jaxtyping as jt
from raytrax import dispersion, utils

ScalarFloat = float | jt.Float[jax.Array, " "]


def power_flux_vector_stix(
    refractive_index_perp: ScalarFloat,
    refractive_index_para: ScalarFloat,
    dielectric_tensor: jt.Complex[jax.Array, "3 3"],
    polarization_vector: jt.Complex[jax.Array, "3"],
) -> jt.Float[jax.Array, "3"]:
    r"""Compute the dielectric power flux vector in Stix coordinates.

    The corrected dielectric wave energy flux vector is given by:

    .. math::
        \mathbf{S}=\frac{c}{16\pi} |A|^2 \frac{\partial}{\partial \mathbf{N}}\left( e_i^* D_{ij}^{H} e_j \right)

    (See Tokman, Westerhof, and Gavrilova,
    https://dx.doi.org/10.1088/0741-3335/42/2/302)

    This function computes a vector that is equal to this expression up to
    normalization, namely

    .. math::
        \mathbf{F} = \frac{1}{2} \frac{\partial}{\partial \mathbf{N}} \left( e_i^* D_{ij}^{H} e_j \right)
    """
    refractive_index = jnp.array(
        [
            refractive_index_perp,
            0.0,
            refractive_index_para,
        ],
        dtype=jnp.float64,
    )
    return 0.5 * power_flux_hamiltonian_gradient_n(
        refractive_index,
        dielectric_tensor=dielectric_tensor,
        polarization_vector=polarization_vector,
    )


def power_flux_hamiltonian_stix(
    refractive_index: jt.Float[jax.Array, "3"],
    dielectric_tensor: jt.Complex[jax.Array, "3 3"],
    polarization_vector: jt.Complex[jax.Array, "3"],
) -> ScalarFloat:
    """Compute the Hamiltonian for the power flux.

    This is the real part of the eigenvalue of the dispersion tensor,
    see eq. (15) in Tokman, Westerhof, and Gavrilova,
    https://dx.doi.org/10.1088/0741-3335/42/2/302.

    The function is named Hamiltonian as this can also be used as the Hamiltonian
    for ray tracing.

    Args:
        refractive_index: Refractive index in Stix coordinates. Must have the
            form [n_perp, 0, n_para].
        dielectric_tensor: Dielectric tensor as a 3x3 complex array.
        polarization_vector: Polarization vector as a 3-element complex array.

    Returns:
        The scalar & real value of the Hamiltonian
    """
    # We are in the Stix frame
    refractive_index_perp = refractive_index[0]
    refractive_index_para = refractive_index[2]
    dispersion_tensor = dispersion.dispersion_tensor_stix(
        refractive_index_perp=refractive_index_perp,
        refractive_index_para=refractive_index_para,
        dielectric_tensor=dielectric_tensor,
    )

    dispersion_tensor_h = utils.hermitian_part(dispersion_tensor)
    return jnp.real(
        polarization_vector.conj().T @ dispersion_tensor_h @ polarization_vector
    )


power_flux_hamiltonian_gradient_n = jax.grad(power_flux_hamiltonian_stix, argnums=0)
