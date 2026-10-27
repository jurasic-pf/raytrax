"""Tests for the Faddeeva function implementation.

Covers the real axis, all four quadrants, near-origin, large |z|, and
physics-relevant Shkarofsky argument ranges (|z| ~ 1–10).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import wofz as scipy_wofz

jax.config.update("jax_enable_x64", True)

from raytrax.math.faddeeva import wofz_jax  # noqa: E402

# ---------------------------------------------------------------------------
# Reference: the previous implementation
# ---------------------------------------------------------------------------

_ERRF = 1.12837916709551257  # 2/sqrt(pi)
_XLIM = 5.33
_YLIM = 4.29


def _wofz_region2(x: jax.Array, y: jax.Array):
    """9-step backward continued fraction for |z| outside the inner region."""
    xh = y
    yh = x

    def body(i, state):
        n = jnp.float64(10 - i)  # i=1..9  →  n=9..1
        Rx, Ry = state
        Tx = xh + n * Rx
        Ty = yh - n * Ry
        Tn = Tx * Tx + Ty * Ty
        return 0.5 * Tx / Tn, 0.5 * Ty / Tn

    Rx, Ry = jax.lax.fori_loop(1, 10, body, (jnp.zeros_like(x), jnp.zeros_like(x)))
    return _ERRF * Rx, _ERRF * Ry


def _wofz_region1(x: jax.Array, y: jax.Array):
    """Inner-region Shuman-sum algorithm for y < 4.29 and x < 5.33.

    The loop runs at most 31 steps (nu ≤ 31); a static fori_loop with
    masking replaces the data-dependent while_loop.
    """
    q = (1.0 - y / _YLIM) * jnp.sqrt(jnp.maximum(0.0, 1.0 - (x / _XLIM) ** 2))
    h = 1.0 / (3.2 * q)
    nc = 7.0 + jnp.floor(23.0 * q)  # ∈ [7, 30]
    nu = 10.0 + jnp.floor(21.0 * q)  # ∈ [10, 31]
    xl = jnp.power(h, 1.0 - nc)
    xh = y + 0.5 / h
    yh = x

    _MAX = 32

    def body(i, state):
        n = jnp.float64(_MAX - 1 - i)
        Rx, Ry, xl_, Sx, Sy = state

        Tx = xh + n * Rx
        Ty = yh - n * Ry
        Tn = Tx * Tx + Ty * Ty
        Rx2 = 0.5 * Tx / Tn
        Ry2 = 0.5 * Ty / Tn

        active = (n >= 1.0) & (n <= nu)
        accum = active & (n <= nc)

        Saux = Sx + xl_
        Sx2 = jnp.where(accum, Rx2 * Saux - Ry2 * Sy, Sx)
        Sy2 = jnp.where(accum, Rx2 * Sy + Ry2 * Saux, Sy)
        xl2 = jnp.where(accum, h * xl_, xl_)
        Rx2 = jnp.where(active, Rx2, Rx)
        Ry2 = jnp.where(active, Ry2, Ry)

        return Rx2, Ry2, xl2, Sx2, Sy2

    init = (
        jnp.zeros_like(x),
        jnp.zeros_like(x),
        xl,
        jnp.zeros_like(x),
        jnp.zeros_like(x),
    )
    _, _, _, Sx, Sy = jax.lax.fori_loop(0, _MAX, body, init)
    return _ERRF * Sx, _ERRF * Sy


def _wofz_poppe_wijers(z):
    """Previous implementation (Poppe & Wijers 1990, Algorithm 680), kept as a reference."""
    z = jnp.asarray(z, dtype=jnp.complex128)
    x_re = jnp.real(z)
    x_im = jnp.imag(z)
    x = jnp.abs(x_re)
    y = jnp.abs(x_im)

    inner = (y < _YLIM) & (x < _XLIM)
    Wx1, Wy1 = _wofz_region1(x, y)
    Wx2, Wy2 = _wofz_region2(x, y)
    Wx = jnp.where(inner, Wx1, Wx2)
    Wy = jnp.where(inner, Wy1, Wy2)

    # Im(z) = 0: Re(w(t)) = exp(-t²)
    Wx = jnp.where(y == 0.0, jnp.exp(-x * x), Wx)

    # Odd symmetry in Re(z): Im(w(-x+iy)) = -Im(w(x+iy))
    Wy = jnp.sign(x_re) * Wy

    # Reflection for Im(z) < 0: w(a-iy) = 2exp(-(a-iy)²) - w(-a+iy)
    e2 = 2.0 * jnp.exp(y * y - x * x)
    Wx_neg = e2 * jnp.cos(2.0 * x * y) - Wx
    Wy_neg = jnp.sign(x_re) * e2 * jnp.sin(2.0 * x * y) + Wy
    Wx = jnp.where(x_im < 0.0, Wx_neg, Wx)
    Wy = jnp.where(x_im < 0.0, Wy_neg, Wy)

    return Wx + 1j * Wy


# ---------------------------------------------------------------------------
# Test points covering the full complex plane
# ---------------------------------------------------------------------------

# fmt: off
_CASES = [
    # (z,                       description)
    # Real axis
    ( 0.0+0j,                  "origin"),
    ( 0.1+0j,                  "real small pos"),
    ( 1.0+0j,                  "real unit"),
    ( 2.0+0j,                  "real moderate"),
    ( 5.0+0j,                  "real large"),
    (10.0+0j,                  "real very large"),
    (-0.5+0j,                  "real small neg"),
    (-2.0+0j,                  "real neg moderate"),
    (-5.0+0j,                  "real large neg"),
    # Pure imaginary
    ( 0+0.1j,                  "imag small pos"),
    ( 0+1.0j,                  "imag unit"),
    ( 0+3.0j,                  "imag moderate"),
    ( 0+8.0j,                  "imag large"),
    ( 0-0.5j,                  "imag small neg"),
    ( 0-2.0j,                  "imag neg"),
    ( 0-8.0j,                  "imag large neg"),
    # Q1: Re>0, Im>0
    ( 1.0+1.0j,                "Q1 symmetric"),
    ( 2.0+0.5j,                "Q1 small Im"),
    ( 0.5+3.0j,                "Q1 large Im"),
    ( 3.0+3.0j,                "Q1 equal large"),
    ( 1.5+2.5j,                "Q1 general"),
    # Q2: Re<0, Im>0
    (-1.0+1.0j,                "Q2 symmetric"),
    (-3.0+0.5j,                "Q2 small Im"),
    (-0.5+4.0j,                "Q2 large Im"),
    # Q3: Re<0, Im<0
    (-1.0-1.0j,                "Q3 symmetric"),
    (-2.0-3.0j,                "Q3 general"),
    (-0.5-5.0j,                "Q3 large neg Im"),
    # Q4: Re>0, Im<0
    ( 1.0-1.0j,                "Q4 symmetric"),
    ( 3.0-0.5j,                "Q4 small neg Im"),
    ( 0.5-4.0j,                "Q4 large neg Im"),
    ( 2.0-2.0j,                "Q4 equal"),
    # Near real axis (Im → 0+)
    ( 1.0+1e-3j,               "near real axis pos Q1"),
    ( 3.0+1e-3j,               "near real axis pos Q1 large"),
    (-2.0+1e-3j,               "near real axis pos Q2"),
    # Near real axis (Im → 0-)
    ( 1.0-1e-3j,               "near real axis neg Q4"),
    ( 3.0-1e-3j,               "near real axis neg Q4 large"),
    (-2.0-1e-3j,               "near real axis neg Q3"),
    # Near origin
    ( 0.01+0.01j,              "near origin Q1"),
    ( 0.001+0j,                "near origin real"),
    ( 0+0.001j,                "near origin imag"),
    # Large |z| (asymptotic regime)
    (10.0+1.0j,                "large |z| Q1"),
    ( 1.0+10.0j,               "large imag Q1"),
    (15.0+0.1j,                "large real near real axis"),
    (-10.0+1.0j,               "large Q2"),
    (10.0-1.0j,                "large Q4"),
    (-5.0-5.0j,                "large Q3"),
    # Physics-relevant: Shkarofsky arguments psi-phi, -psi-phi, -phi
    # Typical: psi~1.5 (real), phi~1.6 (can be complex)
    ( 1.5+0j,                  "psi typical"),
    (-1.5+0j,                  "-psi typical"),
    ( 0-1.6j,                  "-phi imaginary"),
    ( 1.5-1.6j,                "psi-phi Q4"),
    (-1.5-1.6j,                "-(psi+phi) Q3"),
    ( 1.5+1.6j,                "psi-phi* Q1"),
    (-1.5+1.6j,                "-(psi+phi*) Q2"),
    # mu=51 (T=10 keV): psi up to ~5
    ( 4.0-2.0j,                "high mu Q4"),
    (-4.0-2.0j,                "high mu Q3"),
    # Harmonic resonance region: phi near real (alpha_s > 0, large)
    ( 2.0+0j,                  "phi real large"),
    ( 5.0+0j,                  "phi real very large"),
]
# fmt: on

_IDS = [desc for _, desc in _CASES]

_RTOL = 1e-10
_ATOL = 1e-12  # absolute floor for small values


@pytest.mark.parametrize("z, desc", _CASES, ids=_IDS)
def test_wofz_accuracy(z, desc):
    """wofz_jax must match scipy.special.wofz to _RTOL relative error."""
    z_arr = jnp.array(z, dtype=jnp.complex128)
    got = complex(wofz_jax(z_arr))
    expected = scipy_wofz(complex(z))

    err = abs(got - expected)
    tol = _ATOL + _RTOL * abs(expected)
    assert err <= tol, (
        f"[{desc}] z={z}\n"
        f"  got      = {got}\n"
        f"  expected = {expected}\n"
        f"  abs err  = {err:.3e}  (tol={tol:.3e})"
    )


def test_wofz_reflection_identity():
    """w(z) + w(-z) = 2 exp(-z²) must hold for all z."""
    test_z = [1 + 1j, 2 - 3j, -1 + 0.5j, 0 + 2j, 3 + 0j, 1.5 - 1.6j]
    for z in test_z:
        z_arr = jnp.array(z, dtype=jnp.complex128)
        w_pos = complex(wofz_jax(z_arr))
        w_neg = complex(wofz_jax(-z_arr))
        expected = 2 * np.exp(-(z**2))
        actual = w_pos + w_neg
        assert abs(actual - expected) < 1e-9, (
            f"z={z}: w(z)+w(-z)={actual!r}, 2exp(-z²)={expected!r}"
        )


def test_wofz_jit_compatible():
    """wofz_jax must compile and run under jax.jit."""
    f = jax.jit(wofz_jax)
    z = jnp.array(1.5 + 2.3j, dtype=jnp.complex128)
    got = complex(f(z))
    expected = scipy_wofz(1.5 + 2.3j)
    assert abs(got - expected) < 1e-10


def test_wofz_vmap_compatible():
    """wofz_jax must support batched evaluation via vmap."""
    raw = [complex(z) for z, _ in _CASES[:20]]
    zs = jnp.array(raw, dtype=jnp.complex128)
    got = np.array(jax.vmap(wofz_jax)(zs))
    expected = np.array([scipy_wofz(z) for z in raw])
    np.testing.assert_allclose(got, expected, rtol=_RTOL, atol=_ATOL)


def test_wofz_jvp():
    """The custom JVP w'(z) = 2i/√π - 2z·w(z) must be consistent with finite differences."""
    z0 = jnp.array(1.0 + 0.5j, dtype=jnp.complex128)
    dz = jnp.array(0.3 + 0.2j, dtype=jnp.complex128)
    _, dw = jax.jvp(wofz_jax, (z0,), (dz,))

    eps = 1e-6
    dw_fd = (wofz_jax(z0 + eps * dz) - wofz_jax(z0 - eps * dz)) / (2 * eps)

    assert abs(complex(dw) - complex(dw_fd)) < 1e-8, (
        f"JVP mismatch: analytic={complex(dw)!r}, fd={complex(dw_fd)!r}"
    )


def test_wofz_no_nan_or_inf():
    """wofz_jax must never return NaN or Inf for any of the test points."""
    for z, desc in _CASES:
        z_arr = jnp.array(z, dtype=jnp.complex128)
        w = complex(wofz_jax(z_arr))
        assert np.isfinite(w.real) and np.isfinite(w.imag), (
            f"[{desc}] z={z}: got non-finite result {w}"
        )


def test_plasma_dispersion_function_values():
    """Z(z) = i√π·w(z): spot-check against scipy."""
    from raytrax.math.faddeeva import plasma_dispersion_function

    test_pts = [0.5 + 0j, 1.0 + 1.0j, 0 + 2.0j, 1.5 - 1.6j]
    for z in test_pts:
        z_arr = jnp.array(z, dtype=jnp.complex128)
        got = complex(plasma_dispersion_function(z_arr))
        expected = 1j * np.sqrt(np.pi) * scipy_wofz(z)
        assert abs(got - expected) < 1e-10, (
            f"Z({z}): got {got!r}, expected {expected!r}"
        )


def test_plasma_dispersion_function_derivative():
    """Z'(z) = -2(1 + z·Z(z)) must match finite differences."""
    from raytrax.math.faddeeva import (
        plasma_dispersion_function,
        plasma_dispersion_function_derivative,
    )

    z = jnp.linspace(-5, 5, 1000, dtype=jnp.complex128)
    dx = z[1] - z[0]
    Z = plasma_dispersion_function(z)
    Z_prime = plasma_dispersion_function_derivative(z)
    Z_prime_fd = (Z[2:] - Z[:-2]) / (2 * dx)
    np.testing.assert_allclose(Z_prime[1:-1], Z_prime_fd, rtol=0, atol=1e-3)


# Grid over the region the Shkarofsky functions visit on the public test rays
# (Re in [-123, 39], Im in [0, 152]) plus the lower half-plane near the origin.
_GRID = np.concatenate(
    [
        (
            np.linspace(-130, 130, 261)[None] + 1j * np.linspace(0, 160, 161)[:, None]
        ).ravel(),
        (
            np.linspace(-6, 6, 121)[None] + 1j * np.linspace(-6, -1e-3, 61)[:, None]
        ).ravel(),
    ]
)


def test_wofz_matches_scipy_on_grid():
    # Measured maximum relative error 3.3e-14 (lower half-plane, via the reflection).
    got = np.asarray(jax.jit(jax.vmap(wofz_jax))(jnp.asarray(_GRID)))
    np.testing.assert_allclose(got, scipy_wofz(_GRID), rtol=1e-13, atol=0)


def test_wofz_matches_previous_implementation_on_grid():
    # Limited by the reference, which is off scipy by up to 4e-10.
    got = np.asarray(jax.jit(jax.vmap(wofz_jax))(jnp.asarray(_GRID)))
    reference = np.asarray(jax.jit(jax.vmap(_wofz_poppe_wijers))(jnp.asarray(_GRID)))
    np.testing.assert_allclose(got, reference, rtol=1e-9, atol=0)
