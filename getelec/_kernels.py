"""
Low-level numerical kernels for GETELEC.

This module contains the performance-critical inner loops, kept separate from the
physics so that the physics modules stay readable. Nothing here knows about work
functions or Fermi levels -- it only integrates an ODE.

The Noumerov recurrence for -psi'' + (V - E)/HBAR2_2M psi = 0 on a uniform grid is

    (1 - T[j+1]) psi[j+1] = (2 + 10 T[j]) psi[j] - (1 - T[j-1]) psi[j-1],
    T[j] = -(h^2 / 12) (E - V[j]) / HBAR2_2M = alpha (E - V[j]).

Three structural choices make this fast:

1. **Rolling state.** Only psi at the final two grid points is needed to extract
   the transmission coefficient, so the full (n_energy, n_grid) wavefunction is
   never materialised. That removes ~190 MB of allocation per call at typical
   settings and turns the kernel from memory-bound into compute-bound.

2. **Grid loop outside, energy loop inside.** The recurrence carries a dependency
   along the grid axis but is embarrassingly parallel along the energy axis.
   Putting energies innermost lets the compiler emit SIMD for the inner loop and
   hides the latency of the divide, which is the critical path.

3. **T recomputed, not stored.** ``T[i, j] = alpha * (E[i] - V[j])`` is a single
   fused multiply-add from two streams that both sit in cache. Reading it back
   from an (n_energy, n_grid) array instead costs far more memory bandwidth than
   the arithmetic it saves.

The kernels also accept a *stack* of potentials, so an entire field sweep or a
multi-band semiconductor calculation is a single parallel launch rather than one
launch per point.

Falls back to a vectorised NumPy implementation if Numba is unavailable; results
are identical to floating-point round-off.

:func:`run_noumerov_integration` is the exception to all of the above: the same
recurrence for a single energy, written out plainly and keeping the whole
wavefunction. It is the kernel of
:class:`~getelec.transmission_solver.NoumerovReference`.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - trivial import guard
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn

        return wrap(args[0]) if args and callable(args[0]) else wrap

    prange = range


__all__ = ["noumerov_endpoints", "noumerov_wavefunction", "run_noumerov_integration",
           "plan_chunks", "HAVE_NUMBA"]

# Energies per parallel work item. Sized so the six rolling state vectors stay
# resident in L1 while still giving the scheduler enough work items to spread.
_CHUNK = 128


def plan_chunks(pot_index: np.ndarray, chunk: int = _CHUNK):
    """
    Split the energy axis into work items that never straddle two potentials.

    Each chunk therefore streams a single contiguous row of the potential stack,
    which keeps the grid-axis access pattern sequential.

    Parameters
    ----------
    pot_index : np.ndarray
        ``pot_index[i]`` is the row of the potential stack that energy ``i`` uses.
        Must be sorted (non-decreasing).
    chunk : int
        Maximum number of energies per work item.

    Returns
    -------
    lo, hi, pot : np.ndarray
        Half-open energy ranges ``[lo[c], hi[c])`` and the potential row each uses.
    """
    pot_index = np.ascontiguousarray(pot_index, dtype=np.int64)
    # Boundaries where the potential changes, plus the array ends.
    edges = np.flatnonzero(np.diff(pot_index)) + 1
    bounds = np.concatenate(([0], edges, [pot_index.size]))

    lo, hi, pot = [], [], []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        for a in range(start, stop, chunk):
            lo.append(a)
            hi.append(min(a + chunk, stop))
            pot.append(pot_index[start])
    return (
        np.asarray(lo, dtype=np.int64),
        np.asarray(hi, dtype=np.int64),
        np.asarray(pot, dtype=np.int64),
    )


@njit(cache=True, fastmath=True, parallel=True)
def _noumerov_numba(energies, minus_V, alpha, psi0_re, psi0_im, psi1_re, psi1_im,
                   chunk_lo, chunk_hi, chunk_pot, out_re, out_im):
    """Numba implementation. See :func:`noumerov_endpoints`."""
    n_grid = minus_V.shape[1]

    for c in prange(chunk_lo.shape[0]):
        i0 = chunk_lo[c]
        i1 = chunk_hi[c]
        p = chunk_pot[c]
        m = i1 - i0

        # Rolling state: T at j-1 and j, psi at j-1 and j, for every energy in
        # the chunk. Seven short vectors instead of two (m, n_grid) matrices.
        t_prev = np.empty(m)
        t_curr = np.empty(m)
        p_prev_re = np.empty(m)
        p_prev_im = np.empty(m)
        p_curr_re = np.empty(m)
        p_curr_im = np.empty(m)
        e_loc = np.empty(m)

        # Hoist the row out of the loop: a 1D contiguous view lets the inner
        # loop index with a single offset instead of recomputing a 2D address
        # on every one of the ~14000 grid steps.
        mv = minus_V[p]
        mv0 = mv[0]
        mv1 = mv[1]
        for i in range(m):
            e = energies[i0 + i]
            e_loc[i] = e
            t_prev[i] = alpha * (e + mv0)
            t_curr[i] = alpha * (e + mv1)
            p_prev_re[i] = psi0_re[i0 + i]
            p_prev_im[i] = psi0_im[i0 + i]
            p_curr_re[i] = psi1_re[i0 + i]
            p_curr_im[i] = psi1_im[i0 + i]

        for j in range(1, n_grid - 1):
            mv_next = mv[j + 1]
            for i in range(m):  # vectorisable: no dependence across energies
                t_next = alpha * (e_loc[i] + mv_next)
                inv = 1.0 / (1.0 - t_next)
                c_curr = 2.0 + 10.0 * t_curr[i]
                c_prev = 1.0 - t_prev[i]

                nxt_re = (c_curr * p_curr_re[i] - c_prev * p_prev_re[i]) * inv
                nxt_im = (c_curr * p_curr_im[i] - c_prev * p_prev_im[i]) * inv

                p_prev_re[i] = p_curr_re[i]
                p_prev_im[i] = p_curr_im[i]
                p_curr_re[i] = nxt_re
                p_curr_im[i] = nxt_im
                t_prev[i] = t_curr[i]
                t_curr[i] = t_next

        for i in range(m):
            out_re[i0 + i, 0] = p_curr_re[i]  # psi at the last grid point
            out_im[i0 + i, 0] = p_curr_im[i]
            out_re[i0 + i, 1] = p_prev_re[i]  # psi at the second-to-last point
            out_im[i0 + i, 1] = p_prev_im[i]


def _noumerov_numpy(energies, minus_V, alpha, psi0, psi1, pot_index):
    """
    Pure-NumPy fallback with the same rolling structure.

    Vectorised over energies, looping over the grid. Slower than the compiled
    kernel but has no build-time dependency and gives identical results.
    """
    n_grid = minus_V.shape[1]
    mv = minus_V[pot_index]  # (n_energy, n_grid) view-gather

    t_prev = alpha * (energies + mv[:, 0])
    t_curr = alpha * (energies + mv[:, 1])
    p_prev = psi0.copy()
    p_curr = psi1.copy()

    for j in range(1, n_grid - 1):
        t_next = alpha * (energies + mv[:, j + 1])
        nxt = ((2.0 + 10.0 * t_curr) * p_curr - (1.0 - t_prev) * p_prev) / (1.0 - t_next)
        p_prev, p_curr = p_curr, nxt
        t_prev, t_curr = t_curr, t_next

    return np.stack((p_curr, p_prev), axis=1)


def noumerov_endpoints(energies, minus_V, alpha, psi0, psi1, pot_index=None):
    """
    Integrate the Noumerov recurrence and return psi at the final two grid points.

    Parameters
    ----------
    energies : np.ndarray, shape (n_energy,)
        Energies to solve for, in eV. Must be grouped by ``pot_index``.
    minus_V : np.ndarray, shape (n_potential, n_grid)
        **Negated** potential profiles, one row per potential. Negated so the
        inner loop is a fused multiply-add rather than a subtract-then-multiply.
    alpha : float
        ``-(h**2 / 12) / HBAR2_2M``, folding the step size and the kinetic
        prefactor into a single constant.
    psi0, psi1 : np.ndarray, shape (n_energy,), complex
        Seed values of psi at the first two grid points.
    pot_index : np.ndarray, shape (n_energy,), optional
        Row of ``minus_V`` used by each energy. Defaults to all zeros.

    Returns
    -------
    np.ndarray, shape (n_energy, 2), complex
        Column 0 is psi at the last grid point, column 1 at the second-to-last.
    """
    energies = np.ascontiguousarray(energies, dtype=np.float64)
    minus_V = np.ascontiguousarray(np.atleast_2d(minus_V), dtype=np.float64)
    psi0 = np.ascontiguousarray(psi0, dtype=np.complex128)
    psi1 = np.ascontiguousarray(psi1, dtype=np.complex128)

    if pot_index is None:
        pot_index = np.zeros(energies.size, dtype=np.int64)
    else:
        pot_index = np.ascontiguousarray(pot_index, dtype=np.int64)

    if not HAVE_NUMBA:
        return _noumerov_numpy(energies, minus_V, alpha, psi0, psi1, pot_index)

    lo, hi, pot = plan_chunks(pot_index)
    out_re = np.empty((energies.size, 2))
    out_im = np.empty((energies.size, 2))
    _noumerov_numba(
        energies, minus_V, alpha,
        np.ascontiguousarray(psi0.real), np.ascontiguousarray(psi0.imag),
        np.ascontiguousarray(psi1.real), np.ascontiguousarray(psi1.imag),
        lo, hi, pot, out_re, out_im,
    )
    return out_re + 1j * out_im


@njit(cache=True, fastmath=True, parallel=True)
def _wavefunction_numba(energies, minus_V, alpha, psi0_re, psi0_im,
                        psi1_re, psi1_im, out_re, out_im):
    """Numba implementation. See :func:`noumerov_wavefunction`."""
    n_energy = energies.shape[0]
    n_grid = minus_V.shape[0]
    for i in prange(n_energy):
        energy = energies[i]
        t_prev = alpha * (energy + minus_V[0])
        t_curr = alpha * (energy + minus_V[1])
        out_re[i, 0] = psi0_re[i]
        out_im[i, 0] = psi0_im[i]
        out_re[i, 1] = psi1_re[i]
        out_im[i, 1] = psi1_im[i]
        p_prev_re, p_prev_im = psi0_re[i], psi0_im[i]
        p_curr_re, p_curr_im = psi1_re[i], psi1_im[i]
        for j in range(1, n_grid - 1):
            t_next = alpha * (energy + minus_V[j + 1])
            inv = 1.0 / (1.0 - t_next)
            c_curr = 2.0 + 10.0 * t_curr
            c_prev = 1.0 - t_prev
            nxt_re = (c_curr * p_curr_re - c_prev * p_prev_re) * inv
            nxt_im = (c_curr * p_curr_im - c_prev * p_prev_im) * inv
            out_re[i, j + 1] = nxt_re
            out_im[i, j + 1] = nxt_im
            p_prev_re, p_prev_im = p_curr_re, p_curr_im
            p_curr_re, p_curr_im = nxt_re, nxt_im
            t_prev, t_curr = t_curr, t_next


def noumerov_wavefunction(energies, minus_V, alpha, psi0, psi1):
    """
    Integrate the Noumerov recurrence and keep the whole wavefunction.

    The companion to :func:`noumerov_endpoints`, which keeps only the last two
    grid points. That one is far faster and uses O(n_grid) memory; this one is
    what you need when the wavefunction itself is the answer -- a charge density
    for a Poisson solve, a probability current, a plot of the decaying tail
    inside the barrier.

    The cost is memory, and it is not small: an ``(n_energy, n_grid)`` complex
    array is 16 bytes per element, so 800 energies on a 14000-point grid is
    180 MB. Ask for the energies you need rather than the whole band.

    Unlike the endpoint kernel this cannot invert the loop order -- the output
    is indexed by both axes, so the grid loop has to be innermost and the
    energies are parallelised across instead.

    Parameters
    ----------
    energies : np.ndarray, shape (n_energy,)
    minus_V : np.ndarray, shape (n_grid,)
        **Negated** potential on the grid.
    alpha : float
        ``-(h**2 / 12) / HBAR2_2M``.
    psi0, psi1 : np.ndarray, shape (n_energy,), complex
        Seed values at the first two grid points.

    Returns
    -------
    np.ndarray, shape (n_energy, n_grid), complex
    """
    energies = np.ascontiguousarray(energies, dtype=np.float64)
    minus_V = np.ascontiguousarray(np.ravel(minus_V), dtype=np.float64)
    psi0 = np.ascontiguousarray(psi0, dtype=np.complex128)
    psi1 = np.ascontiguousarray(psi1, dtype=np.complex128)

    shape = (energies.size, minus_V.size)
    if not HAVE_NUMBA:
        psi = np.empty(shape, dtype=np.complex128)
        psi[:, 0], psi[:, 1] = psi0, psi1
        t_prev = alpha * (energies + minus_V[0])
        t_curr = alpha * (energies + minus_V[1])
        for j in range(1, minus_V.size - 1):
            t_next = alpha * (energies + minus_V[j + 1])
            psi[:, j + 1] = ((2.0 + 10.0 * t_curr) * psi[:, j]
                             - (1.0 - t_prev) * psi[:, j - 1]) / (1.0 - t_next)
            t_prev, t_curr = t_curr, t_next
        return psi

    out_re = np.empty(shape)
    out_im = np.empty(shape)
    _wavefunction_numba(energies, minus_V, alpha,
                        np.ascontiguousarray(psi0.real),
                        np.ascontiguousarray(psi0.imag),
                        np.ascontiguousarray(psi1.real),
                        np.ascontiguousarray(psi1.imag), out_re, out_im)
    return out_re + 1j * out_im


@njit(cache=True)
def run_noumerov_integration(psi, T_vals):
    """
    The Noumerov recurrence for one energy, written out plainly.

    The kernel of
    :func:`~getelec.transmission_solver.calculate_noumerov_reference`.
    ``psi[0]`` and ``psi[1]`` hold the seed, and every later element is
    overwritten with

        psi[i+1] = ((2 + 10 T[i]) psi[i] - (1 - T[i-1]) psi[i-1]) / (1 - T[i+1]),

    so the whole wavefunction comes back. Unlike :func:`noumerov_endpoints` and
    :func:`noumerov_wavefunction` it runs serially and is compiled without
    ``fastmath``: it is written to be checked against, not to be fast.

    Parameters
    ----------
    psi : np.ndarray, shape (n_grid,), complex
        The seeded wavefunction, filled in place.
    T_vals : np.ndarray, shape (n_grid,)
        ``(h**2 / 12) (V - E) / HBAR2_2M`` at each grid point.

    Returns
    -------
    np.ndarray
        ``psi``.
    """
    N = len(psi)
    for i in range(1, N - 1):
        numerator = (2 + 10 * T_vals[i]) * psi[i] - (1 - T_vals[i - 1]) * psi[i - 1]
        denominator = 1 - T_vals[i + 1]
        psi[i + 1] = numerator / denominator
    return psi
