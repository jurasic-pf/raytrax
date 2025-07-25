import jax
import jax.numpy as jnp
from raytrax import ray, solver

jax.config.update("jax_enable_x64", True)


def test_y_to_state_roundtrip():
    y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    state = solver._y_to_state(y, s=0.0)
    assert jnp.allclose(y, solver._state_to_y(state))


def test_ray_tracing():
    state = ray.RayState(
        position=jnp.array([0.0, 0.0, 0.0]),
        refractive_index=jnp.array([1.0, 1.0, 1.0]),
        optical_depth=jnp.array(0.0),
        arc_length=jnp.array(0.0),
    )
    setting = ray.RaySetting(
        frequency=jnp.array(238e9),
        mode="X",
    )

    def magnetic_field_interpolator(position):
        return jnp.array([10.0, 0.0, 0.0])

    def electron_density_interpolator(position):
        return jnp.array(0.1)

    def electron_temperature_interpolator(position):
        return jnp.array(1.0)

    solution = solver.solve(
        state,
        setting,
        magnetic_field_interpolator,
        electron_density_interpolator,
        electron_temperature_interpolator,
    )
    print(solution)
