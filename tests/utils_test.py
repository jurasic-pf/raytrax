import jax
import jax.numpy as jnp
import numpy as np
from raytrax.utils import anti_hermitian_part, hermitian_part

jax.config.update("jax_enable_x64", True)
_MACHINE_PRECISION = float(jnp.finfo(float).eps)


def test_hermitian_antihermitian_part():
    matrix = jax.random.normal(jax.random.PRNGKey(42), (3, 3)) + 1j * jax.random.normal(
        jax.random.PRNGKey(43), (3, 3)
    )
    matrix_h = hermitian_part(matrix)
    matrix_ah = anti_hermitian_part(matrix)

    # h is Hermitian
    np.testing.assert_allclose(
        matrix_h,
        matrix_h.T.conj(),
        rtol=0,
        atol=_MACHINE_PRECISION,
    )

    # ah is anti-Hermitian
    np.testing.assert_allclose(
        matrix_ah,
        -matrix_ah.T.conj(),
        rtol=0,
        atol=_MACHINE_PRECISION,
    )

    # matrix = h + ah
    np.testing.assert_allclose(
        matrix,
        matrix_h + matrix_ah,
        rtol=0,
        atol=2 * _MACHINE_PRECISION,
    )
