from functools import partial

import jax
import jax.numpy as jnp
from scipy.special import jv as scipy_jv
from scipy.special import jvp as scipy_jvp
from scipy.special import kv as scipy_kv


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def jv_jax(v, z):
    """Compute the Bessel function of the first kind of real order and real argument.

    Args:
        v: order of the Bessel function
        z: complex argument
    """
    shape = jnp.shape(z)
    dtype = jnp.float64
    return jax.pure_callback(
        lambda z, v=v: scipy_jv(v, z),
        jax.ShapeDtypeStruct(shape=shape, dtype=dtype),
        z,
        vmap_method="sequential",
    )


@jv_jax.defjvp
def jv_jax_jvp(v, primals, tangents):
    """Custom JVP rule for the Bessel function."""
    (z,) = primals
    (dz,) = tangents
    jv = jv_jax(v, z)
    djv_dz = 0.5 * (jv_jax(v - 1, z) - jv_jax(v + 1, z))
    return jv, djv_dz * dz


djv_jax = jax.grad(jv_jax, argnums=1)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def kv_jax(v, z):
    """Compute the modified Bessel function of the second kind of real order and real
    argument.

    Args:
        v: order of the Bessel function
        z: complex argument
    """
    shape = jnp.shape(z)
    dtype = jnp.float64

    scipy_kv_v = partial(scipy_kv, v)

    return jax.pure_callback(
        lambda z: scipy_kv_v(z),
        jax.ShapeDtypeStruct(shape=shape, dtype=dtype),
        z,
        vmap_method="sequential",
    )


@kv_jax.defjvp
def kv_jax_jvp(v, primals, tangents):
    """Custom JVP rule for the modified Bessel function of the second kind."""
    (z,) = primals
    (dz,) = tangents
    kv_val = kv_jax(v, z)
    dkv_dz = -0.5 * (kv_jax(v - 1, z) + kv_jax(v + 1, z))
    return kv_val, dkv_dz * dz


dkv_jax = jax.grad(kv_jax, argnums=1)
