"""
Closed-form and semiclassical transmission.

These are the solutions you can write down, as opposed to the numerical
integration in :mod:`getelec.transmission_solver`. Three of them:

- :class:`AiryTriangular` -- the exact solution for a pure triangular barrier,
  in terms of Airy functions. No image charge, so it is not the barrier a real
  metal presents, but it is the one case where the Schrodinger equation has a
  closed form, which makes it the natural check on a numerical solver.
- :class:`WKB` -- the semiclassical result for the Schottky-Nordheim barrier.
  The turning points are roots of a quadratic, so the Gamow exponent needs only
  a quadrature, and Gauss-Chebyshev matches its endpoint behaviour exactly.
- :func:`calculate_gamow_numeric` -- the same semiclassical physics for a barrier of any
  shape, where the turning points have to be found numerically. Used as the
  reference function for the learned solvers, whose job is to correct it.

All of them are cheap. None of them is exact for a real barrier: WKB is accurate
to a factor of order unity deep in the tunnelling regime, which is fine for
exploration and for use as a reference, and not fine for a published number.
"""

from __future__ import annotations

import numpy as np
from scipy.special import airy

from getelec import constants as const
from getelec.transmission_solver import TransmissionSolver

__all__ = ["WKB", "AiryTriangular", "calculate_gamow_numeric", "calculate_log_kemble_numeric",
           "get_log_kemble_schottky"]

#: Gauss-Chebyshev nodes and pre-divided weights, cached by node count. The
#: rule is fixed, so recomputing the trigonometry on every call is pure waste.
_NODE_CACHE = {}

class WKB(TransmissionSolver):
    """
    Semiclassical (WKB) transmission for the Schottky-Nordheim barrier.

    The Gamow exponent is

        G = sqrt(2m)/hbar * integral from x1 to x2 of sqrt(V(x) - E) dx,

    with ``V(x) = E_F + phi - F x - k_e / (4x)`` and ``x1, x2`` the classical
    turning points, i.e. the roots of ``F x^2 + (E - E_F - phi) x + k_e / 4``.
    Transmission then follows from the Kemble form, ``T = 1 / (1 + exp(2G))``.

    The integrand is ``sqrt(F (x2 - x)(x - x1) / x)``, which vanishes like a
    square root at both turning points. Gauss-Chebyshev quadrature of the second
    kind carries exactly that weight, so it converges geometrically where an
    adaptive general-purpose rule fights the endpoint behaviour: 64 nodes give
    about 12 significant figures. It is also fully vectorised over energy, with
    no Python-level loop.

    Above the barrier top the turning points become complex. There the exponent
    is continued using the inverted-parabola (Kemble) approximation at the
    barrier maximum, so ``T -> 1`` smoothly instead of the formula breaking down.

    Parameters
    ----------
    fermi_level, work_function, electric_field : float
        Barrier parameters (eV, eV, V/nm).
    n_nodes : int, default 64
        Quadrature nodes. 64 is converged to round-off for typical parameters.

    """

    def __init__(self, fermi_level: float = 9.5, work_function: float = 4.5,
                 electric_field: float = 3.0, n_nodes: int = 64):
        self.fermi_level = fermi_level
        self.work_function = work_function
        self.electric_field = electric_field
        self.n_nodes = int(n_nodes)
        self._nodes_cache = None

    def _nodes(self):
        """Gauss-Chebyshev (second kind) nodes and weights on [-1, 1]."""
        if self._nodes_cache is None or self._nodes_cache[0] != self.n_nodes:
            n = self.n_nodes
            theta = np.arange(1, n + 1) * np.pi / (n + 1)
            self._nodes_cache = (n, np.cos(theta), (np.pi / (n + 1)) * np.sin(theta) ** 2)
        return self._nodes_cache[1], self._nodes_cache[2]

    def _parameters(self, potential=None):
        """
        Barrier parameters to use, preferring the barrier actually passed in.

        Reading them off the argument rather than from stored state means the
        solver cannot silently fall out of sync with the barrier it is being
        asked about -- which is exactly what happens in a sweep, where the
        caller holds several barriers but only one solver.
        """
        if potential is None:
            return self.fermi_level, self.work_function, self.electric_field
        return (getattr(potential, "fermi_level", self.fermi_level),
                getattr(potential, "work_function", self.work_function),
                getattr(potential, "electric_field", self.electric_field))

    def get_gamow_exponent(self, energies: np.ndarray, potential=None) -> np.ndarray:
        """
        Gamow exponent G at each energy (dimensionless).

        Exposed separately because G, not T, is the quantity that appears in
        Fowler-Nordheim analysis and in slope/intercept fitting.

        Parameters
        ----------
        energies : np.ndarray
        potential : potential_barrier.Barrier, optional
            If given, its parameters take precedence over the ones stored on
            this solver.
        """
        E = np.asarray(energies, dtype=np.float64)
        fermi_level, work_function, F = self._parameters(potential)
        Q = const.COULOMB_CONST

        if F <= 0:
            raise ValueError("electric_field must be positive for the WKB solver.")

        # V(x) - E = -(F x^2 + b x + Q/4) / x, so the turning points are the
        # roots of F x^2 + b x + Q/4.  The sign of the Q/4 term matters: with the
        # opposite sign one root is negative and the integration path crosses the
        # x = 0 pole.
        b = E - fermi_level - work_function
        disc = b * b - F * Q
        # Tunnelling requires the barrier maximum to lie ABOVE the electron
        # energy: E < V_max = W - sqrt(QF), i.e. b < -sqrt(QF). Testing only
        # disc > 0 is not enough -- for E well above the vacuum level b is large
        # and positive, disc turns positive again, and both turning points come
        # out negative, which sends the integrand imaginary.
        below_barrier = b < -np.sqrt(F * Q)
        root = np.sqrt(np.where(below_barrier, disc, 0.0))
        x1 = (-b - root) / (2 * F)
        x2 = (-b + root) / (2 * F)

        u, w = self._nodes()
        mid = 0.5 * (x1 + x2)
        half = 0.5 * (x2 - x1)
        xq = mid[:, None] + half[:, None] * u[None, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            integral = np.sum(w[None, :] / np.sqrt(np.where(xq > 0, xq, 1.0)), axis=1)
        G = const.WKB_PREFACTOR * np.sqrt(F) * half ** 2 * integral

        # Above the barrier top: inverted parabola at the maximum of V.
        v_max = fermi_level + work_function - np.sqrt(Q * F)
        x_max = np.sqrt(Q / (4 * F))
        curvature = Q / (2 * x_max ** 3)  # |V''(x_max)|
        hbar_omega = np.sqrt(2 * const.HBAR2_2M * curvature)
        G_above = -np.pi * (E - v_max) / hbar_omega

        return np.where(below_barrier, G, G_above)

    def calculate_transmission(self, potential, energies: np.ndarray) -> np.ndarray:
        """Transmission probability at each energy, via the Kemble form."""
        G = self.get_gamow_exponent(energies, potential)
        # Clip before exp: 2G > 700 underflows T to 0 anyway, and this keeps the
        # exponential from overflowing and emitting a warning.
        return 1.0 / (1.0 + np.exp(np.clip(2.0 * G, -700.0, 700.0)))


def get_log_kemble_schottky(W, F, energies):
    """
    ``ln D`` from the semiclassical Kemble form, evaluated stably.

    ``ln(1/(1+exp(2G)))`` is ``-logaddexp(0, 2G)``, which is exact in both tails.
    Taking ``log`` of the probability instead bottoms out at about ``-745`` and
    then returns ``-inf``, which would make the residual undefined exactly where
    the table is most useful.
    """
    energies = np.asarray(energies, dtype=np.float64)
    gamow = WKB(fermi_level=W - 4.5, work_function=4.5,
                electric_field=F).get_gamow_exponent(energies)
    return -np.logaddexp(0.0, 2.0 * gamow)


def _chebyshev_nodes(n):
    """Gauss-Chebyshev (second kind) nodes and weights on [-1, 1], cached."""
    if n not in _NODE_CACHE:
        theta = np.arange(1, n + 1) * np.pi / (n + 1)
        u = np.cos(theta)
        w = (np.pi / (n + 1)) * np.sin(theta) ** 2
        # The Chebyshev weight already carries sqrt((x2-x)(x-x1)); dividing it
        # out leaves a smooth integrand, so few nodes suffice.
        _NODE_CACHE[n] = (u, w / np.sqrt(np.maximum(1.0 - u ** 2, 1e-30)))
    return _NODE_CACHE[n]


def _profile(potential, energy_min, n_scan):
    """Log-spaced sample of the barrier, extended until it drops below E."""
    fermi = getattr(potential, "fermi_level", 0.0)
    work = getattr(potential, "work_function", 0.0)
    field = max(getattr(potential, "electric_field", 1.0), 1e-6)

    x_max = max((fermi + work) / field * 1.2, 1.0)
    for _ in range(10):
        # Log spacing: the image term varies on the scale of x itself near the
        # surface, and hardly at all far out.
        x = np.geomspace(1e-4, x_max, n_scan)
        V = potential.get_potential(x)
        if V[-1] < energy_min:
            break
        x_max *= 3.0
    return x, V


def calculate_gamow_numeric(potential, energies, n_scan=1024, n_nodes=24):
    """
    Gamow exponent for an arbitrary barrier, by numerical quadrature.

    Parameters
    ----------
    potential : potential_barrier.Barrier
    energies : array_like
        Energies in eV.
    n_scan : int
        Points used to bracket the turning points.
    n_nodes : int
        Quadrature nodes. 24 is ample for a reference function.

    Returns
    -------
    np.ndarray
        ``G`` at each energy. Negative above the barrier top, where it is
        continued with the inverted parabola at the maximum so that ``G`` passes
        smoothly through zero instead of jumping.
    """
    energies = np.asarray(energies, dtype=np.float64)
    if energies.size == 0:
        return np.zeros(0)

    x, V = _profile(potential, float(energies.min()), n_scan)
    peak = int(np.argmax(V))
    v_max = V[peak]

    # The rising branch is increasing in V and the falling branch decreasing;
    # reversing the latter makes both searchable, so the turning points come
    # from two vectorised searchsorted calls instead of a scan per energy.
    x1 = _interp_crossing(V[:peak + 1], x[:peak + 1], energies)
    x2 = _interp_crossing(V[peak:][::-1], x[peak:][::-1], energies)

    below = (energies < v_max) & np.isfinite(x1) & np.isfinite(x2) & (x2 > x1)

    gamow = np.empty(energies.shape)
    if below.any():
        u, w = _chebyshev_nodes(n_nodes)
        mid = 0.5 * (x1[below] + x2[below])
        half = 0.5 * (x2[below] - x1[below])
        xq = mid[:, None] + half[:, None] * u[None, :]
        integrand = np.sqrt(np.maximum(
            potential.get_potential(xq.ravel()).reshape(xq.shape)
            - energies[below][:, None], 0.0))
        gamow[below] = const.WKB_PREFACTOR * half * (integrand * w[None, :]).sum(axis=1)

    if not below.all():
        gamow[~below] = _above_barrier(x, V, energies[~below], peak, v_max)
    return gamow


def _interp_crossing(V_sorted, x_sorted, energies):
    """Where a monotonically increasing branch of V crosses each energy."""
    if V_sorted.size < 2:
        return np.full(energies.shape, np.nan)
    idx = np.clip(np.searchsorted(V_sorted, energies), 1, V_sorted.size - 1)
    v0, v1 = V_sorted[idx - 1], V_sorted[idx]
    x0, x1 = x_sorted[idx - 1], x_sorted[idx]
    span = np.where(v1 == v0, 1.0, v1 - v0)
    return x0 + (energies - v0) * (x1 - x0) / span


def _above_barrier(x, V, energies, peak, v_max):
    """Inverted-parabola continuation at the barrier maximum."""
    lo, hi = max(peak - 2, 0), min(peak + 3, x.size)
    if hi - lo < 3:
        return -(energies - v_max)
    curvature = abs(2.0 * np.polyfit(x[lo:hi], V[lo:hi], 2)[0])
    hbar_omega = np.sqrt(2.0 * const.HBAR2_2M * max(curvature, 1e-12))
    return -np.pi * (energies - v_max) / max(hbar_omega, 1e-12)


def calculate_log_kemble_numeric(potential, energies, **kwargs):
    """
    ``ln D`` from the Kemble form, using the numerical Gamow exponent.

    Evaluated as ``-logaddexp(0, 2G)``, exact in both tails; taking ``log`` of
    the probability would bottom out near -745 and then return ``-inf``, which
    is exactly where the residual is needed.
    """
    return -np.logaddexp(0.0, 2.0 * calculate_gamow_numeric(potential, energies, **kwargs))


class AiryTriangular(TransmissionSolver):
    """
    Exact transmission through a pure triangular barrier, via Airy functions.

    For ``V(x) = E_F + phi - F x`` outside the metal and ``V = 0`` inside, with
    no image term -- the barrier of
    :class:`~getelec.potential_barrier.TriangularPotential` -- the Schrodinger
    equation outside is Airy's equation, so the transmission has a closed form.
    A plane wave in the metal, matched at the surface to the outgoing wave
    ``Bi + i Ai`` outside, gives

        T = (4k / (pi beta)) / [ (k/beta)^2 (Ai^2 + Bi^2) + Ai'^2 + Bi'^2 + 2k / (pi beta) ]

    with ``k`` the wavevector in the metal, ``beta = (F / (hbar^2/2m))^(1/3)``,
    and the Airy functions evaluated at the surface, ``zeta = beta (E_F + phi - E) / F``.
    The last term is the Wronskian ``Ai Bi' - Ai' Bi = 1/pi``. It is negligible in
    deep tunnelling, where ``Bi`` is exponentially large, and it is what takes
    the result to the step-barrier value ``4 k q / (k + q)^2`` far above the
    barrier, ``q`` being the wavevector just outside the surface.

    This is not the barrier a metal presents -- dropping the image charge
    removes the Schottky lowering, so it overestimates the barrier and
    underestimates emission. Its value is as a check: it is the one case where
    an exact answer exists, so it pins the numerical solver against something
    that is not another numerical solver.

    Parameters
    ----------
    fermi_level, work_function, electric_field : float
        eV, eV, V/nm. Read from the barrier when one is passed.
    """

    def __init__(self, fermi_level=9.5, work_function=4.5, electric_field=3.0):
        self.fermi_level = fermi_level
        self.work_function = work_function
        self.electric_field = electric_field

    def _parameters(self, potential):
        if potential is None:
            return self.fermi_level, self.work_function, self.electric_field
        return (getattr(potential, "fermi_level", self.fermi_level),
                getattr(potential, "work_function", self.work_function),
                getattr(potential, "electric_field", self.electric_field))

    def calculate_transmission(self, potential, energies) -> np.ndarray:
        """Transmission probability at each energy."""
        energies = np.asarray(energies, dtype=np.float64)
        fermi, work, field = self._parameters(potential)
        if field <= 0:
            raise ValueError("electric_field must be positive.")

        vacuum_level = fermi + work
        # beta sets the Airy length scale: V - E = -hbar^2/2m * beta^3 * (x - x0)
        beta = (field / const.HBAR2_2M) ** (1.0 / 3.0)
        # Wavevector in the metal (V = 0) and far outside, one field-length on.
        k_in = np.sqrt(np.maximum(energies, 0.0) / const.HBAR2_2M)

        # Dimensionless coordinate at the surface, zeta = beta * (V0 - E) / F.
        zeta = beta * (vacuum_level - energies) / field
        ai, aip, bi, bip = airy(zeta)

        # Wronskian-normalised matching of a plane wave in the metal onto the
        # outgoing wave Bi + i Ai outside. Ai and Bi satisfy Ai Bi' - Ai' Bi =
        # 1/pi, which is where the last term of the denominator comes from.
        # Without it T is right in deep tunnelling but too large near and above
        # the barrier top: by up to 47% just above it, and above 1 (so clipped)
        # further up.
        with np.errstate(over="ignore", invalid="ignore"):
            numerator = 4.0 * k_in / (np.pi * beta)
            denominator = ((k_in / beta) ** 2 * (ai ** 2 + bi ** 2)
                           + (aip ** 2 + bip ** 2)
                           + 2.0 * k_in / (np.pi * beta))
            transmission = numerator / denominator

        transmission[~np.isfinite(transmission)] = 0.0
        return np.clip(transmission, 0.0, 1.0)
