"""
Numerical solvers for quantum transmission probabilities.

Provides the :class:`TransmissionSolver` interface and its numerical
implementations:

- :class:`Noumerov` / :class:`NoumerovFast` -- direct integration of the 1D
  Schrodinger equation, with no approximation beyond discretisation, whose
  local truncation error is O(h^6).
- :class:`NoumerovReference` -- the same integration written out plainly, one
  energy at a time, for checking the solvers above and debugging a result.
- :class:`NeuralSolver` -- a trained network standing in for the Noumerov
  result, worth it for barriers with several parameters. Train one with
  :mod:`getelec.training`.

The closed-form and semiclassical solutions (WKB, the Airy-function solution)
live in :mod:`getelec.transmission_solutions`. Every solver takes an array of
energies and returns an array of transmission probabilities, so they are
interchangeable inside an emitter.
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from importlib import resources

from pathlib import Path

import numpy as np

from scipy.interpolate import CubicSpline

from getelec import constants as const
from getelec._kernels import (noumerov_endpoints, noumerov_wavefunction,
                              run_noumerov_integration)


def _solutions():
    """
    Import the semiclassical solutions lazily.

    They live in :mod:`getelec.transmission_solutions`, which imports
    ``TransmissionSolver`` from this module -- so importing it at module level
    here would be circular. The learned solver only needs it at
    call time, by which point both modules are loaded.
    """
    from getelec import transmission_solutions
    return transmission_solutions

__all__ = ["TransmissionSolver", "Noumerov", "NoumerovFast",
           "NoumerovReference", "calculate_noumerov_reference",
           "NeuralSolver", "NeuralModel", "get_shipped_model",
           "register_features", "get_schottky_features", "get_small_radii_features"]


class TransmissionSolver(ABC):
    """
    Abstract base class for all transmission solvers.

    Subclasses must implement :meth:`calculate_transmission`. Implementing
    :meth:`calculate_transmission_batch` as well is optional but lets sweeps and
    multi-band calculations be evaluated in a single pass.
    """

    @abstractmethod
    def calculate_transmission(self, potential, energies: np.ndarray) -> np.ndarray:
        """
        Transmission probability at each energy.

        Parameters
        ----------
        potential : potential_barrier.Barrier
            Barrier defining the spatial potential profile.
        energies : np.ndarray
            1D array of energies, in eV.

        Returns
        -------
        np.ndarray
            Transmission probabilities in [0, 1], same shape as ``energies``.
        """

    def calculate_transmission_batch(self, potentials, energy_arrays):
        """
        Transmission for several (potential, energies) pairs at once.

        The base implementation just loops. Solvers that can genuinely batch
        override this. Callers should prefer it over writing their own loop, so
        they benefit automatically when a solver does support batching.

        Parameters
        ----------
        potentials : sequence of potential_barrier.Barrier
        energy_arrays : sequence of np.ndarray

        Returns
        -------
        list of np.ndarray
        """
        return [self.calculate_transmission(p, e)
                for p, e in zip(potentials, energy_arrays)]


class NoumerovFast(TransmissionSolver):
    """
    Noumerov integration of the 1D Schrodinger equation, endpoints only.

    Keeps only psi at the last two grid points, which is all the transmission
    coefficient needs. That is what makes it fast and what bounds its memory to
    O(n_grid) instead of O(n_energy x n_grid). If you need the wavefunction
    itself -- a charge density, a probability current, the decaying tail inside
    the barrier -- use :class:`Noumerov`, which keeps all of it.

    Integrates right-to-left from deep vacuum into the metal, seeding an outgoing
    plane wave on the vacuum side and projecting the result onto incoming and
    outgoing waves inside the metal. The transmission probability follows from
    the flux ratio.

    Noumerov's method has a local truncation error of O(h^6), which makes it one
    of the most accurate methods for this equation, and it needs only a single
    three-term recurrence per grid point, which makes it fast.

    Parameters
    ----------
    x_metal : float, default -0.01
        Left-hand end of the integration domain, inside the metal (nm). The
        potential is identically zero there and the wave is matched at the last
        two grid points, so this only has to be negative: measured over the
        planar, sharp-tip and triangular barriers at 0.2-12 V/nm, the current
        density from ``x_metal=-0.01`` and from ``-1.0`` differ by 4e-10.
    x_vac_plus : float, default 3.0
        Distance past the barrier at which to start the integration (nm). The
        seed is only exact for a flat potential, so this has to be large enough
        that the residual gradient does not matter. With the default WKB seed,
        3 nm changes the current density by 0.0024%, the Nottingham heat by
        0.0034% and the transmission itself by 0.025% against a 20 nm domain,
        measured over the same barriers and fields. Check a particular case
        with :meth:`calculate_convergence_report` rather than assuming.
    h : float, default 1e-3
        Grid spacing (nm). The local truncation error is O(h^6).
    max_barrier_width : float, default 3.0
        Expected barrier width (nm); together with ``x_vac_plus`` this sets the
        right-hand end of the domain.

    Notes
    -----
    The spatial grid and the potential profile are cached and rebuilt only when
    the geometry or the barrier parameters actually change, so repeated calls at
    a fixed field cost almost nothing in setup.
    """

    def __init__(self, x_metal: float = -0.01, x_vac_plus: float = 3.0,
                 h: float = 1e-3, max_barrier_width: float = 3.0,
                 seed: str = "wkb", energy_nodes=None, auto_domain: bool = True,
                 interpolation_tolerance: float = 0.01):
        self.x_metal = float(x_metal)
        self.x_vac_plus = float(x_vac_plus)
        self.h = float(h)
        self.max_barrier_width = float(max_barrier_width)
        self.seed = seed
        self.energy_nodes = energy_nodes
        self.auto_domain = auto_domain
        #: Largest tolerated error in log(T) at the probe points before the
        #: energy reduction is abandoned. 0.01 is 1% in T.
        self.interpolation_tolerance = float(interpolation_tolerance)
        self._n_probes = 24
        self._grid_cache = None
        self._potential_cache = {}

    @classmethod
    def fast(cls, **kwargs):
        """
        Preset tuned for speed, with the accuracy trade-offs made explicit.

        Three changes relative to the defaults, each justified by a convergence
        study:

        - ``x_vac_plus=2.0``, one nanometre less vacuum than the default. Worth
          7% of the grid, at a current density 0.0072% from a 20 nm domain
          instead of 0.0024%, and a transmission 0.066% from it instead of
          0.025%.
        - ``h=2e-3``, half the grid points. Against h = 1.25e-4 this costs
          0.015% in current density and 0.035% in the transmission itself,
          measured for the planar and sharp-tip barriers over work functions of
          2.5 to 6 eV, fields of 0.5 to 12 V/nm and 300 to 3000 K. The
          triangular barrier, the one with a jump at the surface, costs 0.01%
          on the aligned grid (see :meth:`_grid`).
        - ``energy_nodes=48``. Transmission is solved on a reduced energy set and
          log-interpolated onto the full grid.

        The short metal side, which this preset used to carry alone, is now the
        default for every Noumerov solve.

        All three together, against a converged reference (h = 1.25e-4 on a
        20 nm domain) at 1 to 12 V/nm and 300 to 1500 K: current density within
        5e-5, energy distributions within 2e-3, pointwise transmission within
        7e-4. The default settings reach 2e-5, 2e-4 and 2e-4 on the same cases,
        so the preset costs a factor of a few in accuracy for a factor of three
        in time -- and both are a long way inside the 1% that separates
        transmission algorithms from each other.

        Use the default constructor when you want the transmission itself to be
        converged pointwise across the whole band.
        """
        opts = dict(x_metal=-0.01, x_vac_plus=2.0, h=2e-3,
                    seed="wkb", energy_nodes=48)
        opts.update(kwargs)
        # Always the endpoint solver: the preset exists for speed, and keeping
        # the wavefunction would defeat it. Ask Noumerov for psi explicitly.
        return NoumerovFast(**opts)

    @classmethod
    def reference(cls, **kwargs):
        """
        The reference implementation, :class:`NoumerovReference`.

        The same integration written out plainly, one energy at a time, to
        check a result rather than to produce many: a metal current density
        takes about half a second. Keyword arguments (``x_start``, ``x_end``,
        ``h``, ``seed``) go to :class:`NoumerovReference`.
        """
        return NoumerovReference(**kwargs)

    # -- grid and potential caching ---------------------------------------

    def get_required_barrier_width(self, potentials, energy_min):
        """
        Outermost classical turning point over the given barriers and energies.

        The Schottky-Nordheim barrier extends to x2 ~ (E_F + phi - E) / F, which
        for a low field and a deep energy is tens of nm -- far beyond the 3 nm
        default. Integrating only part of a barrier does not produce a slightly
        wrong answer, it produces a meaningless one, so the domain is measured
        rather than assumed.

        Returns
        -------
        float
            Required barrier width in nm, or ``None`` if it cannot be determined.
        """
        widths = []
        for potential in potentials:
            work = getattr(potential, "work_function", None)
            fermi = getattr(potential, "fermi_level", None)
            field = getattr(potential, "electric_field", None)
            if None in (work, fermi, field) or field <= 0:
                return None
            # Without the image term the turning point is exactly (W - E) / F;
            # the image term only lowers V, so this is a strict upper bound.
            upper = (fermi + work - energy_min) / field
            if not np.isfinite(upper) or upper <= 0:
                return None
            # (W - E)/F is a strict upper bound for the planar barrier, but not
            # for a curved tip, where the field falls off away from the apex and
            # the barrier reaches further. Extend the scan until the potential
            # has actually dropped below the energy.
            limit = upper * 1.02 + 1.0
            for _ in range(8):
                x_scan = np.linspace(1e-3, limit, 4096)
                above = potential.get_potential(x_scan) > energy_min
                if not above[-1]:
                    break
                limit *= 4.0
            widths.append(x_scan[np.flatnonzero(above)[-1]] if above.any() else 0.0)
        return max(widths) if widths else None

    def _grid(self, barrier_width=None) -> np.ndarray:
        """
        Uniform right-to-left grid, with the surface halfway between two nodes.

        The offset matters for a barrier that jumps at x = 0, which the
        triangular one does: Noumerov resolves a jump only to first order in h,
        by an amount set by where the surface falls between nodes. A node
        sitting on the surface costs 1.3% in current density at h = 1e-3 and
        the midpoint costs 0.003%, measured against the exact Airy solution for
        work functions of 2.5 to 6 eV, fields of 1 to 12 V/nm and 300 to
        2000 K. For a continuous barrier the same shift is worth about 1e-5,
        so the alignment is free where it is not needed.
        """
        width = self.max_barrier_width if barrier_width is None else barrier_width
        key = (self.x_metal, self.x_vac_plus, self.h, width)
        if self._grid_cache is None or self._grid_cache[0] != key:
            # Half-integer multiples of h put x = 0 exactly between two nodes.
            x_end = (np.floor((width + self.x_vac_plus) / self.h) + 0.5) * self.h
            # Two nodes inside the metal are what the plane-wave match needs, so
            # the metal side never gets shorter than that however coarse h is.
            x_metal = min(self.x_metal, -2.0 * self.h)
            # An integer count avoids the off-by-one that np.arange can produce
            # when (stop - start) is not an exact multiple of the step.
            n = int(np.ceil((x_end - x_metal) / self.h)) + 1
            x = x_end - self.h * np.arange(n)
            self._grid_cache = (key, x)
        return self._grid_cache[1]

    @staticmethod
    def _barrier_key(potential):
        """Hashable snapshot of everything the barrier's profile depends on."""
        from getelec.potential_barrier import get_barrier_parameters
        return get_barrier_parameters(potential)

    def _potential_profile(self, potential, x: np.ndarray) -> np.ndarray:
        key = (self._barrier_key(potential), x.shape[0], x[0], x[-1])
        cached = self._potential_cache.get(key)
        if cached is None:
            cached = np.ascontiguousarray(potential.get_potential(x), dtype=np.float64)
            if len(self._potential_cache) > 64:  # bound the cache
                self._potential_cache.clear()
            self._potential_cache[key] = cached
        return cached

    # -- public API --------------------------------------------------------

    def calculate_transmission(self, potential, energies: np.ndarray) -> np.ndarray:
        """Transmission probability at each energy for a single barrier."""
        return self.calculate_transmission_batch([potential], [energies])[0]

    def calculate_transmission_batch(self, potentials, energy_arrays):
        """
        Solve several barriers in one parallel launch.

        Used by field sweeps and by the semiconductor emitter, which needs four
        energy grids against the same barrier. Batching amortises the kernel
        launch and gives the thread pool enough work to saturate.
        """
        if self.energy_nodes:
            return self._interpolated_batch(potentials, energy_arrays)
        return self._solve_batch(potentials, energy_arrays)

    def _energy_nodes(self, n_energy):
        """
        Reduced energy set: uniform, plus extra nodes at the bottom of the band.

        log(T) is smooth and close to linear over most of the range, so uniform
        nodes interpolate it well. The exception is the lowest ~0.1 eV, where the
        metal-side wavelength grows longer than the integration domain and the
        curvature of log(T) jumps by four orders of magnitude. Clustering there
        instead of uniformly would starve the region around the Fermi level,
        which is where the current actually comes from -- so do both.
        """
        n = int(self.energy_nodes)
        if n_energy <= n + 12:
            return None
        uniform = np.round(np.linspace(0, n_energy - 1, n)).astype(np.int64)
        low = np.round(np.geomspace(1, max(uniform[1], 2), 12)).astype(np.int64)
        return np.unique(np.concatenate(([0], low, uniform)))

    def _interpolated_batch(self, potentials, energy_arrays):
        """
        Solve on a reduced energy set and spline log(T) onto the full grid,
        but only where that is demonstrably safe.
        """
        energy_arrays = [np.ascontiguousarray(e, dtype=np.float64) for e in energy_arrays]
        node_sets = [self._energy_nodes(e.size) for e in energy_arrays]

        probes = []
        combined = []
        for full, idx in zip(energy_arrays, node_sets):
            if idx is None:
                probes.append(None)
                combined.append(full)
                continue
            # Probe points sit between consecutive nodes, where a spline is
            # least constrained and aliasing shows up first.
            mid = ((idx[:-1] + idx[1:]) // 2)
            mid = np.unique(mid[(mid > idx[0]) & (mid < idx[-1])])
            if mid.size > self._n_probes:
                mid = mid[np.round(np.linspace(0, mid.size - 1, self._n_probes)).astype(int)]
            probes.append(mid)
            combined.append(full[np.union1d(idx, mid)])

        solved = self._solve_batch(potentials, combined)

        out = []
        redo_index, redo_potentials, redo_energies = [], [], []
        for i, (full, idx, probe, grid, trans) in enumerate(
                zip(energy_arrays, node_sets, probes, combined, solved)):
            if idx is None:
                out.append(trans)
                continue
            node_mask = np.isin(grid, full[idx])
            coarse, coarse_t = grid[node_mask], trans[node_mask]
            positive = coarse_t > 0
            if positive.sum() < 4:
                out.append(None)
                redo_index.append(i)
                redo_potentials.append(potentials[i])
                redo_energies.append(full)
                continue

            # Energy grids are not guaranteed to arrive sorted or distinct: a
            # non-parabolic band structure can produce a scaled grid that is
            # descending or has repeated points, and CubicSpline needs a
            # strictly increasing abscissa.
            order = np.argsort(coarse[positive])
            abscissa = coarse[positive][order]
            ordinate = np.log(coarse_t[positive])[order]
            distinct = np.concatenate(([True], np.diff(abscissa) > 0))
            if distinct.sum() < 4:
                out.append(None)
                redo_index.append(i)
                redo_potentials.append(potentials[i])
                redo_energies.append(full)
                continue
            abscissa, ordinate = abscissa[distinct], ordinate[distinct]
            spline = CubicSpline(abscissa, ordinate)

            accurate = True
            if probe is not None and probe.size:
                probe_mask = np.isin(grid, full[probe])
                probe_t = trans[probe_mask]
                usable = probe_t > 0
                if usable.any():
                    deviation = np.abs(spline(grid[probe_mask][usable])
                                       - np.log(probe_t[usable]))
                    accurate = bool(deviation.max() < self.interpolation_tolerance)

            if not accurate:
                out.append(None)
                redo_index.append(i)
                redo_potentials.append(potentials[i])
                redo_energies.append(full)
                continue

            interpolated = np.exp(spline(full))
            # Below the lowest solved node the spline extrapolates; transmission
            # is monotonic in energy there, so clamp rather than trust it.
            interpolated[full < abscissa[0]] = np.exp(ordinate[0])
            out.append(np.minimum(interpolated, 1.0))

        if redo_index:
            exact = self._solve_batch(redo_potentials, redo_energies)
            for position, values in zip(redo_index, exact):
                out[position] = values
        return out

    def _solve_batch(self, potentials, energy_arrays, log_output=False):
        energy_arrays = [np.ascontiguousarray(e, dtype=np.float64) for e in energy_arrays]
        sizes = [e.size for e in energy_arrays]
        if not sizes or sum(sizes) == 0:
            return [np.full(s, -np.inf) if log_output else np.zeros(s) for s in sizes]

        width = None
        if self.auto_domain:
            # Sized from the lowest energy actually requested. Sizing from the
            # bottom of the band would be more reproducible, but at low field it
            # is ruinous: the barrier for E = 0 at 0.005 V/nm reaches 2800 nm,
            # which is 2.8 million grid points for energies that only need 60.
            # The consequence is that a batch shares one grid sized to the
            # widest barrier in the set -- see calculate_transmission_batch.
            energy_min = min(float(e.min()) for e in energy_arrays if e.size)
            width = self.get_required_barrier_width(potentials, max(energy_min, 0.0))
        x = self._grid(width)
        V = np.stack([self._potential_profile(p, x) for p in potentials])

        flat_E = np.concatenate(energy_arrays)
        pot_index = np.repeat(np.arange(len(energy_arrays), dtype=np.int64), sizes)

        # k(x) = sqrt((E - V(x)) / (hbar^2 / 2m)). Only three grid points matter:
        # the seed point in vacuum and the two matching points in the metal.
        # Evaluating sqrt on the full (n_energy, n_grid) array, as the original
        # did, discards better than 99.9% of the work.
        k_vac = np.sqrt(((flat_E - V[pot_index, 0]) / const.HBAR2_2M).astype(np.complex128))
        k_a = np.sqrt(((flat_E - V[pot_index, -1]) / const.HBAR2_2M).astype(np.complex128))
        k_b = np.sqrt(((flat_E - V[pot_index, -2]) / const.HBAR2_2M).astype(np.complex128))

        if self.seed == "plane":
            # Original convention: unit-amplitude plane wave with the local
            # wavevector. Exact only where the potential is flat, which it is
            # not, so this needs a lot of vacuum to converge.
            psi0 = np.exp(1j * k_vac * x[0])
            psi1 = np.exp(1j * k_vac * x[1])
            outgoing_flux = k_vac.real
        elif self.seed == "wkb":
            # psi = k^(-1/2) exp(i int k dx) solves the equation to first order
            # in the slowly-varying-envelope sense, so it stays accurate much
            # closer to the barrier. It also carries unit flux by construction,
            # which is why the numerator below becomes 1 instead of Re(k).
            x_mid = 0.5 * (x[0] + x[1])
            V_mid = np.stack([np.atleast_1d(p.get_potential(np.array([x_mid])))[0]
                              for p in potentials])
            k_mid = np.sqrt(((flat_E - V_mid[pot_index]) / const.HBAR2_2M).astype(np.complex128))
            k_1 = np.sqrt(((flat_E - V[pot_index, 1]) / const.HBAR2_2M).astype(np.complex128))
            # Simpson's rule for the phase accumulated over the single step.
            phase = (self.h / 6.0) * (k_vac + 4.0 * k_mid + k_1)
            psi0 = k_vac ** -0.5
            psi1 = (k_1 ** -0.5) * np.exp(-1j * phase)
            outgoing_flux = np.ones(flat_E.size)
        else:
            raise ValueError(f"seed must be 'wkb' or 'plane', got {self.seed!r}")

        alpha = -(self.h ** 2 / 12.0) / const.HBAR2_2M
        psi = noumerov_endpoints(flat_E, -V, alpha, psi0, psi1, pot_index)

        # Project psi onto exp(+/- i k x) at the two matching points. Solving the
        # 2x2 system in closed form is faster than assembling and inverting a
        # stack of matrices, and avoids a needless LAPACK round trip.
        xa, xb = x[-1], x[-2]
        m00 = np.exp(1j * k_a * xa)
        m01 = np.exp(-1j * k_a * xa)
        m10 = np.exp(1j * k_b * xb)
        m11 = np.exp(-1j * k_b * xb)
        det = m00 * m11 - m01 * m10

        with np.errstate(divide="ignore", invalid="ignore"):
            amp_incident = (m11 * psi[:, 0] - m01 * psi[:, 1]) / det
            # T = (v_transmitted / v_incident) / |A|^2, with v proportional to
            # Re(k). Computed in logs: deep in the tunnelling regime |A| reaches
            # 1e150 and |A|**2 overflows to inf, silently returning T = 0 for a
            # transmission that is small but perfectly representable. Taking the
            # log first roughly doubles the reachable exponent range.
            log_transmission = (np.log(outgoing_flux) - np.log(k_a.real)
                                - 2.0 * np.log(np.abs(amp_incident)))

        # Evanescent channels (imaginary k) carry no flux.
        log_transmission[~np.isfinite(log_transmission)] = -np.inf

        if log_output:
            return list(np.split(log_transmission, np.cumsum(sizes)[:-1]))
        transmission = np.exp(log_transmission)
        transmission[~np.isfinite(transmission)] = 0.0
        return list(np.split(transmission, np.cumsum(sizes)[:-1]))

    def calculate_log_transmission(self, potential, energies: np.ndarray) -> np.ndarray:
        """
        Natural log of the transmission probability.

        Preferred over ``log(calculate_transmission(...))`` whenever the answer
        may be very small: this reaches ln T of about -1200, where taking the
        log of the returned probability bottoms out near -700 and then returns
        -inf. Useful for deep tunnelling and for building interpolation tables.

        Returns
        -------
        np.ndarray
            ln T, with -inf for channels that carry no flux.
        """
        return self._solve_batch([potential], [energies], log_output=True)[0]

    def calculate_convergence_report(self, potential, energies, factors=(1.0, 2.0, 4.0)):
        """
        Refine the grid and report how much the answer moves.

        The default ``h`` is not a guarantee of accuracy at every field and
        energy range. This reruns the calculation at ``h``, ``h/2``, ``h/4`` and
        returns the relative change, so a claimed digit count can be checked
        rather than assumed.

        Returns
        -------
        dict
            Maps each step size to the largest relative deviation from the
            finest grid.
        """
        base_h = self.h
        results = {}
        try:
            for f in sorted(factors, reverse=True):
                self.h = base_h / f
                results[self.h] = self.calculate_transmission(potential, energies)
        finally:
            self.h = base_h
            self._grid_cache = None
            self._potential_cache.clear()
        finest = results[min(results)]
        mask = finest > 0
        return {
            h: (float(np.max(np.abs(t[mask] - finest[mask]) / finest[mask]))
                if mask.any() else 0.0)
            for h, t in results.items()
        }


class Noumerov(NoumerovFast):
    """
    Noumerov integration that keeps the whole wavefunction.

    Same physics and same settings as :class:`NoumerovFast`, and it returns the
    same transmission coefficients -- but it retains psi(x) at every grid point,
    so the wavefunction is available afterwards through :meth:`calculate_psi`.

    Use it when the wavefunction is the answer: a charge density for a Poisson
    solve, a probability current, or simply looking at how the wave decays
    through the barrier. Use :class:`NoumerovFast` when only the transmission
    matters.

    The cost is memory and speed. The array is ``n_energy x n_grid`` complex,
    16 bytes an element, so 800 energies on a 14000-point grid is 180 MB; and
    the loop order cannot be inverted the way the endpoint kernel does, so it is
    several times slower per energy. Ask for the energies you actually need.

    Examples
    --------
    >>> solver = Noumerov()                                    # doctest: +SKIP
    >>> x, psi = solver.calculate_psi(barrier, energies)       # doctest: +SKIP
    >>> density = np.abs(psi) ** 2                             # doctest: +SKIP
    """

    def calculate_psi(self, potential, energies, x_points=None):
        """
        The wavefunction on the spatial grid.

        Parameters
        ----------
        potential : potential_barrier.Barrier
        energies : array_like
            Energies in eV. Each gets its own row.
        x_points : array_like, optional
            Positions at which to return psi. Defaults to the solver's own grid.
            Supplying a coarser set does not make the integration cheaper -- the
            recurrence has to step through every grid point regardless -- but it
            does cut the memory that comes back.

        Returns
        -------
        x, psi : np.ndarray
            ``x`` has shape (n_x,), ordered from vacuum into the metal as the
            integration runs. ``psi`` has shape (n_energy, n_x), complex, and is
            normalised to unit outgoing flux in the vacuum.

        Notes
        -----
        The seed is imposed at the vacuum end and the recurrence runs inward, so
        psi grows by many orders of magnitude across a thick barrier. That is
        the physical behaviour of the growing solution, not an instability, but
        it does mean ``|psi|`` inside the metal can be enormous; divide by the
        incident amplitude if you want a normalised scattering state.
        """
        energies = np.ascontiguousarray(np.atleast_1d(energies), dtype=np.float64)
        x, psi = self._integrate_wavefunction(potential, energies)
        if x_points is None:
            return x, psi
        wanted = np.asarray(x_points, dtype=np.float64)
        # The grid runs vacuum -> metal, i.e. decreasing, so reverse for interp.
        index = np.searchsorted(x[::-1], wanted)
        index = np.clip(x.size - 1 - index, 0, x.size - 1)
        return x[index], psi[:, index]

    def _integrate_wavefunction(self, potential, energies):
        """Run the recurrence, keeping every grid point."""
        width = None
        if self.auto_domain:
            width = self.get_required_barrier_width(
                [potential], max(float(np.min(energies)), 0.0))
        x = self._grid(width)
        V = self._potential_profile(potential, x)

        k_vac = np.sqrt(((energies - V[0]) / const.HBAR2_2M).astype(np.complex128))
        if self.seed == "plane":
            psi0 = np.exp(1j * k_vac * x[0])
            psi1 = np.exp(1j * k_vac * x[1])
        elif self.seed == "wkb":
            x_mid = 0.5 * (x[0] + x[1])
            V_mid = float(np.atleast_1d(potential.get_potential(np.array([x_mid])))[0])
            k_mid = np.sqrt(((energies - V_mid) / const.HBAR2_2M).astype(np.complex128))
            k_1 = np.sqrt(((energies - V[1]) / const.HBAR2_2M).astype(np.complex128))
            phase = (self.h / 6.0) * (k_vac + 4.0 * k_mid + k_1)
            psi0 = k_vac ** -0.5
            psi1 = (k_1 ** -0.5) * np.exp(-1j * phase)
        else:
            raise ValueError(f"seed must be 'wkb' or 'plane', got {self.seed!r}")

        alpha = -(self.h ** 2 / 12.0) / const.HBAR2_2M
        return x, noumerov_wavefunction(energies, -V, alpha, psi0, psi1)

    def calculate_probability_current(self, potential, energies):
        """
        Probability current at every grid point, for each energy.

        For a stationary scattering state this must be independent of position,
        so its variation across the grid is a direct check on the integration --
        more informative than any residual, because it tests a conservation law
        the scheme does not enforce.

        Returns
        -------
        x, current : np.ndarray
        """
        x, psi = self.calculate_psi(potential, energies)
        # j = (hbar/m) Im(psi* dpsi/dx); HBAR2_2M = hbar^2/2m, so hbar/m carries
        # a factor of 2/hbar which is common to every energy and cancels in the
        # ratios this is used for.
        gradient = np.gradient(psi, x, axis=1)
        return x, 2.0 * const.HBAR2_2M * np.imag(np.conj(psi) * gradient)


def calculate_noumerov_reference(potential, electron_energy: float, x_start: float = -1.0,
                                 x_end: float = 20.0, h: float = 1e-3, seed: str = "plane"):
    """
    Transmission at one energy by Noumerov integration, with everything behind it.

    The computation of :class:`NoumerovReference`, written to be read line by
    line. The grid runs from ``x_end`` in the vacuum to ``x_start`` in the
    metal; an outgoing wave is seeded at its first two points; the Schrodinger
    equation is integrated into the metal by
    :func:`getelec._kernels.run_noumerov_integration`; and psi at the last two
    points is matched to ``A exp(ikx) + B exp(-ikx)``. The transmission is the
    outgoing flux of the seed divided by the incident flux, ``k |A|^2``.

    Parameters
    ----------
    potential : potential_barrier.Barrier
        The barrier, carrying its own parameters.
    electron_energy : float
        Energy in eV, measured from the bottom of the band, where V = 0.
    x_start : float, default -1.0
        End of the integration inside the metal (nm).
    x_end : float, default 20.0
        Start of the integration in the vacuum (nm). It has to lie past the
        barrier: where V(x_end) >= E the seed is not a travelling wave and the
        transmission comes out as 0.
    h : float, default 1e-3
        Grid spacing (nm).
    seed : {'plane', 'wkb'}, default 'plane'
        The outgoing wave imposed at ``x_end``: a plane wave with the local
        wavevector, or the WKB form ``k^(-1/2) exp(i int k dx)`` that
        :class:`NoumerovFast` uses by default.

    Returns
    -------
    T_current : float
        Transmission probability.
    x_points : np.ndarray
        The grid, from the vacuum into the metal (nm).
    V : np.ndarray
        The potential on the grid (eV).
    psi : np.ndarray
        The wavefunction on the grid, complex: a unit-amplitude plane wave, or
        a unit-flux WKB wave, at ``x_end``.
    k_metal : float
        Wavevector inside the metal (1/nm).

    Notes
    -----
    Only energies above the bottom of the band have a travelling state in the
    metal. Below it this returns nan, and at exactly zero the matching matrix
    is singular and :func:`numpy.linalg.solve` raises;
    :class:`NoumerovReference` returns 0 there without calling it.
    """
    E = electron_energy  # eV

    x_points = np.arange(x_end, x_start - h, -h)
    N = len(x_points)

    V = potential.get_potential(x_points)

    psi = np.zeros(N, dtype=np.complex128)

    f_vals = (V - E) / const.HBAR2_2M
    T_vals = (h**2 / 12) * f_vals
    k = np.sqrt(-f_vals.astype(np.complex128))

    if seed == "plane":
        psi[0] = np.exp(1j * k[0] * x_points[0])
        psi[1] = np.exp(1j * k[0] * x_points[1])
        # Outgoing flux Re(k) |psi|^2, with |psi| = 1.
        flux_vac = np.real(k[0])
    elif seed == "wkb":
        # psi = k^(-1/2) exp(i int k dx), with the phase over the first step by
        # Simpson's rule: the seed NoumerovFast uses.
        x_mid = 0.5 * (x_points[0] + x_points[1])
        V_mid = potential.get_potential(np.array([x_mid]))[0]
        k_mid = np.sqrt(complex((E - V_mid) / const.HBAR2_2M))
        phase = (h / 6) * (k[0] + 4 * k_mid + k[1])
        psi[0] = k[0] ** -0.5
        psi[1] = k[1] ** -0.5 * np.exp(-1j * phase)
        # Outgoing flux Re(k) |psi|^2, with |psi|^2 = 1/|k|: 1 for a real k.
        flux_vac = np.real(k[0]) / np.abs(k[0])
    else:
        raise ValueError(f"seed must be 'plane' or 'wkb', got {seed!r}")

    psi = run_noumerov_integration(psi, T_vals)

    x_a, psi_a = x_points[-1], psi[-1]
    x_b, psi_b = x_points[-2], psi[-2]
    k_metal = np.sqrt(E / const.HBAR2_2M)

    matrix = np.array([
        [np.exp(1j * k_metal * x_a), np.exp(-1j * k_metal * x_a)],
        [np.exp(1j * k_metal * x_b), np.exp(-1j * k_metal * x_b)]
    ])
    vector = np.array([psi_a, psi_b])
    A, B = np.linalg.solve(matrix, vector)

    T_current = (flux_vac / np.real(k_metal)) * (1 / np.abs(A)**2)

    return T_current, x_points, V, psi, k_metal


class NoumerovReference(TransmissionSolver):
    """
    Noumerov integration written out plainly: the reference implementation.

    Solves each energy on its own with :func:`calculate_noumerov_reference`,
    from full arrays, on a fixed grid. Nothing is cached, batched, interpolated
    or reordered, and the kernel is compiled without ``fastmath``. That makes it
    slow -- about half a second for a metal current density, where
    :class:`Noumerov` takes a few milliseconds -- and easy to follow, which is
    what it is for: checking the fast solvers and debugging a result. Select it
    with ``reference=True`` in the one-line API or with
    :meth:`Noumerov.reference`.

    For the planar barrier (E_F = 7.5 eV, phi = 4.5 eV, 300 K, 3-7 V/nm) the
    current density and Nottingham heat agree with :class:`Noumerov` to 3e-5
    with either seed, and the transmission to 5e-4 with the plane-wave seed and
    3e-4 with the WKB seed.

    The grid is ``np.arange(x_end, x_start - h, -h)``, fixed rather than sized
    to the barrier, and two things follow from that:

    - **It has to enclose the barrier.** Where ``x_end`` is still inside it,
      V(x_end) >= E, the seed is not a travelling wave and the transmission
      comes out as 0. The solver warns when that happens.
    - **The defaults put a node on the surface.** A continuous barrier does not
      notice, but :class:`~getelec.potential_barrier.TriangularPotential` jumps
      there, which costs about 1% in current density. ``x_end=20.0005`` puts
      the surface halfway between two nodes and brings it within 2e-5 of the
      exact solution.

    Energies at or below zero, the bottom of the band, have no travelling state
    in the metal, so no incident flux: they return 0 without being solved.

    Parameters
    ----------
    x_start : float, default -1.0
        End of the integration inside the metal (nm).
    x_end : float, default 20.0
        Start of the integration in the vacuum (nm).
    h : float, default 1e-3
        Grid spacing (nm).
    seed : {'plane', 'wkb'}, default 'plane'
        The outgoing wave imposed at ``x_end``. See
        :func:`calculate_noumerov_reference`.

    Examples
    --------
    >>> getelec.current_density(field=5.0, reference=True)                  # doctest: +SKIP
    >>> solver = NoumerovReference(x_end=20.0005, seed="wkb")               # doctest: +SKIP
    >>> T, x, V, psi, k_metal = calculate_noumerov_reference(barrier, 7.5)  # doctest: +SKIP
    """

    def __init__(self, x_start: float = -1.0, x_end: float = 20.0, h: float = 1e-3,
                 seed: str = "plane"):
        if seed not in ("plane", "wkb"):
            raise ValueError(f"seed must be 'plane' or 'wkb', got {seed!r}")
        self.x_start = float(x_start)
        self.x_end = float(x_end)
        self.h = float(h)
        self.seed = seed

    def calculate_transmission(self, potential, energies) -> np.ndarray:
        """
        Transmission probability at each energy, solved one energy at a time.

        Energies at or below zero return 0 without being solved. Warns when
        ``x_end`` lies inside the barrier for any of the others.
        """
        energies = np.asarray(energies, dtype=np.float64)
        flat = energies.ravel()
        transmission = np.zeros(flat.size)
        travelling = flat > 0
        self._check_domain(potential, flat[travelling])
        for index in np.flatnonzero(travelling):
            transmission[index] = calculate_noumerov_reference(
                potential, flat[index], x_start=self.x_start, x_end=self.x_end,
                h=self.h, seed=self.seed)[0]
        return transmission.reshape(energies.shape)

    def _check_domain(self, potential, energies):
        """Warn about energies for which the barrier has not ended by ``x_end``."""
        if energies.size == 0:
            return
        V_end = float(potential.get_potential(np.array([self.x_end]))[0])
        inside = energies <= V_end
        if inside.any():
            warnings.warn(
                f"x_end = {self.x_end} nm lies inside the barrier for "
                f"{int(inside.sum())} of {energies.size} energies, up to "
                f"E = {energies[inside].max():.4g} eV, where V(x_end) = "
                f"{V_end:.4g} eV. The seed there is not a travelling wave, so "
                f"their transmission comes out as 0. Move x_end past the outer "
                f"turning point.", UserWarning, stacklevel=3)


def get_schottky_features(barrier_height, total_height, field):
    """
    Feature vector for a planar Schottky-Nordheim barrier.

    The first three entries are the scaled physical parameters. The rest are
    physics-derived: ``f = k_e F / h**2`` is the scaled barrier field, equal to 1
    exactly at the barrier top, so ``tanh(log f)`` and ``tanh(f - 1)`` give the
    network a coordinate that locates the top regardless of where the other
    parameters put it. The products let it represent the leading cross terms
    without having to build them from scratch.

    Parameters
    ----------
    barrier_height : array_like
        ``h = E_F + phi - E``, in eV.
    total_height : array_like
        ``W = E_F + phi``, in eV.
    field : array_like
        Field, V/nm.

    Returns
    -------
    np.ndarray, shape (n, 8)
    """
    h, W, F = np.broadcast_arrays(
        *[np.asarray(v, dtype=np.float64) for v in
          (barrier_height, total_height, field)])
    h_n = (h - 2.0) / 5.0
    W_n = (W - 16.5) / 7.0
    F_n = (F - 8.25) / 6.75
    scaled_field = const.COULOMB_CONST * F / np.maximum(h, 0.05) ** 2
    return np.column_stack([
        h_n, W_n, F_n,
        np.tanh(np.log(np.maximum(scaled_field, 1e-300))),
        np.tanh(scaled_field - 1.0),
        np.sqrt(const.COULOMB_CONST * F) / 5.0,
        h_n * F_n, h_n * h_n,
    ])


def get_small_radii_features(barrier_height, total_height, field, radius, gamma):
    """
    Feature vector for the curved-tip barrier: the planar features plus seven
    terms in tip radius and field enhancement.

    The radius enters as its inverse, ``20/R``: the curvature corrections are a
    series in ``x/R``, so the transmission is smooth -- to leading order linear
    -- in ``1/R``, and the planar limit ``R -> infinity`` is the finite point 0.
    It is scaled to [-1, 1] over R = 20-1000 nm, the radii over which
    :class:`~getelec.potential_barrier.SmallRadiiPotential` is valid; gamma is
    taken in its logarithm, scaled to [-1, 1] over 1-200.

    Parameters
    ----------
    barrier_height, total_height, field : array_like
        As in :func:`get_schottky_features`.
    radius : array_like
        Tip radius, nm.
    gamma : array_like
        Field enhancement factor.

    Returns
    -------
    np.ndarray, shape (n, 15)
    """
    h, W, F, R, g = np.broadcast_arrays(
        *[np.asarray(v, dtype=np.float64) for v in
          (barrier_height, total_height, field, radius, gamma)])
    planar = get_schottky_features(h, W, F)
    h_n, F_n = planar[:, 0], planar[:, 2]
    r_n = (20.0 / np.maximum(R.ravel(), 1e-6) - 0.51) / 0.49
    g_n = (np.log(np.maximum(g.ravel(), 1e-6)) - np.log(14.0)) / 2.6
    return np.column_stack([
        planar,
        r_n, g_n, r_n * g_n, r_n * h_n, r_n * F_n, g_n * h_n, g_n * F_n,
    ])


# --------------------------------------------------------------------------
# Learned transmission
# --------------------------------------------------------------------------

#: Feature functions, by the name a model file records. A model stores the name
#: rather than the function so that the file stays plain arrays; register a new
#: one with :func:`register_features` before loading a model that uses it.
FEATURES = {"schottky": get_schottky_features, "small_radii": get_small_radii_features}


def register_features(name, function):
    """
    Make a feature function available to models that record ``name``.

    ``function(barrier_height, total_height, field, *parameters)`` must return
    an ``(n, k)`` array, where ``parameters`` are the model's extra barrier
    attributes in order.
    """
    FEATURES[str(name)] = function


def _reference_kemble_schottky(potential, energies):
    W = potential.fermi_level + potential.work_function
    return _solutions().get_log_kemble_schottky(W, potential.electric_field, energies)


def _reference_kemble_numeric(potential, energies):
    return _solutions().calculate_log_kemble_numeric(potential, energies)


#: Semiclassical references a model's residual is measured against. The
#: closed-form one is exact for the planar barrier and cheap; the numerical one
#: works for any barrier with a single maximum.
REFERENCES = {"kemble_schottky": _reference_kemble_schottky,
              "kemble_numeric": _reference_kemble_numeric}

#: Models shipped in ``getelec/data``, keyed by the barrier class they were
#: trained on. Matched on the exact class: a subclass may change the
#: potential, so it must not silently get its parent's model.
SHIPPED_MODELS = {"SchottkyPotential": "neural_schottky.npz",
                  "SmallRadiiPotential": "neural_small_radii.npz"}

_SHIPPED_CACHE = {}


class NeuralModel:
    """
    A trained network and everything needed to use it, stored as plain arrays.

    :meth:`save` writes an ``.npz`` that loads with ``allow_pickle=False``: the
    weights and biases, the names of the feature function, the semiclassical
    reference and the barrier class, and the domain the model was trained on.
    No pickle means no scikit-learn at load time and no coupling to the version
    that trained it.

    Parameters
    ----------
    weights, biases : list of np.ndarray
        Layer by layer. Hidden layers use tanh, the output is linear.
    features : str
        Key into :data:`FEATURES`.
    reference : str
        Key into :data:`REFERENCES`.
    barrier : str
        Class name of the barrier the model was trained on.
    domain : dict
        ``name -> (lo, hi)`` for ``barrier_height``, ``total_height``, ``field``
        and each entry of ``parameters``.
    parameters : sequence of str
        Extra barrier attributes the features take, in order -- for example
        ``("radius", "gamma")``.
    metadata : dict, optional
        Provenance: how the model was trained and how well it validated.
    """

    def __init__(self, weights, biases, features, reference, barrier, domain,
                 parameters=(), metadata=None):
        self.weights = [np.ascontiguousarray(w, dtype=np.float64) for w in weights]
        self.biases = [np.ascontiguousarray(b, dtype=np.float64) for b in biases]
        if len(self.weights) != len(self.biases):
            raise ValueError("need one bias vector per weight matrix")
        self.features = str(features)
        self.reference = str(reference)
        self.barrier = str(barrier)
        self.domain = {str(k): (float(lo), float(hi)) for k, (lo, hi) in domain.items()}
        self.parameters = tuple(str(p) for p in parameters)
        self.metadata = dict(metadata or {})
        self._single = None

    def predict(self, features):
        """
        Forward pass: tanh hidden layers, linear output.

        Runs in single precision, which cuts its cost by about a third. The
        rounding this adds, ~1e-6 on an output of order unity, is three orders
        of magnitude below the model's own error; the stored weights stay in
        double precision.
        """
        if self._single is None:
            self._single = ([w.astype(np.float32) for w in self.weights],
                            [b.astype(np.float32) for b in self.biases])
        weights, biases = self._single
        activation = np.asarray(features, dtype=np.float32)
        for w, b in zip(weights[:-1], biases[:-1]):
            activation = np.tanh(activation @ w + b)
        return (activation @ weights[-1] + biases[-1]).ravel().astype(np.float64)

    def save(self, path):
        """Write the model as a pickle-free ``.npz``."""
        names = list(self.domain)
        arrays = {f"weights_{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"biases_{i}": b for i, b in enumerate(self.biases)})
        arrays.update(
            n_layers=np.array(len(self.weights)),
            features=np.array(self.features),
            reference=np.array(self.reference),
            barrier=np.array(self.barrier),
            parameters=np.array(self.parameters, dtype=str),
            domain_names=np.array(names, dtype=str),
            domain_bounds=np.array([self.domain[n] for n in names], dtype=np.float64),
            metadata=np.array(json.dumps(self.metadata)),
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, source):
        """Read a model from a path or an open binary file."""
        with np.load(source, allow_pickle=False) as data:
            n = int(data["n_layers"])
            return cls(
                weights=[data[f"weights_{i}"] for i in range(n)],
                biases=[data[f"biases_{i}"] for i in range(n)],
                features=str(data["features"]),
                reference=str(data["reference"]),
                barrier=str(data["barrier"]),
                domain=dict(zip(data["domain_names"].tolist(),
                                data["domain_bounds"].tolist())),
                parameters=data["parameters"].tolist(),
                metadata=json.loads(str(data["metadata"])),
            )

    def __repr__(self):
        sizes = [self.weights[0].shape[0]] + [w.shape[1] for w in self.weights]
        return (f"NeuralModel({self.barrier}, layers={sizes}, "
                f"features={self.features!r}, reference={self.reference!r})")


def get_shipped_model(barrier_name):
    """
    The model shipped for a barrier class, or ``None`` if there is none.

    Read through :mod:`importlib.resources`, so it works from a source
    checkout, an installed wheel and a frozen application alike. Loaded once
    and shared.
    """
    filename = SHIPPED_MODELS.get(barrier_name)
    if filename is None:
        return None
    if barrier_name not in _SHIPPED_CACHE:
        resource = resources.files("getelec").joinpath("data").joinpath(filename)
        with resource.open("rb") as handle:
            _SHIPPED_CACHE[barrier_name] = NeuralModel.load(handle)
    return _SHIPPED_CACHE[barrier_name]


class NeuralSolver(TransmissionSolver):
    """
    Transmission from a trained neural network.

    The network never predicts ``D``. It predicts the residual

        R = ln D_noumerov - ln D_kemble

    against a semiclassical reference for the same barrier. The reference
    already carries the exponential -- hundreds of e-folds -- so the network
    only learns a correction of order unity, and the error budget is explicit:
    relative error in ``D`` is exactly absolute error in ``R``.

    **When it is worth it.** For the planar Schottky barrier the network is
    several times faster than ``Noumerov.fast()``, but that calculation was
    already cheap: the model there is the worked example, not the reason. Its cost does not grow with the number
    of barrier parameters, while exact solves or a table of them do, so it pays
    off for barriers with several -- tip radius and field enhancement, as in the
    shipped curved-tip model. The shipped Schottky model is there as a running
    example of the workflow; :mod:`getelec.training` trains one for your own
    barrier.

    Parameters
    ----------
    model : None, str, Path or NeuralModel, optional
        ``None`` picks the shipped model for the barrier's class (see
        :data:`SHIPPED_MODELS`). A path or a :class:`NeuralModel` uses that
        model, for barriers of the class it was trained on.
    fallback : {'exact', 'wkb', 'error'}, default 'exact'
        What to do for a barrier with no model, or outside the trained domain.
        ``'exact'`` solves it with Noumerov, keeping the answer right at the
        cost of speed. ``'wkb'`` uses the numerical semiclassical result.
        ``'error'`` raises, which is what a pipeline that must not silently
        change accuracy wants.
    exact_solver : optional
        Used by the ``'exact'`` fallback. Defaults to ``Noumerov.fast()``.
    energy_nodes : int, default 48
        Evaluate on this many energies and spline ``ln D`` onto the rest. Both
        the reference and the forward pass cost time proportional to the number
        of energies, and ``ln D`` is smooth in energy, so this makes the cost
        flat in grid size and changes the answer by well under 1%. Set to 0 to
        evaluate at every energy.

    Examples
    --------
    >>> solver = NeuralSolver()                                     # doctest: +SKIP
    >>> barrier = SmallRadiiPotential(7.5, 4.5, 5.0, radius=50.0, gamma=100.0)
    >>> solver.calculate_transmission(barrier, energies)            # doctest: +SKIP
    """

    _WARNED = set()
    #: Node indices depend only on the grid size; the spline map on the grid.
    #: Both are shared across instances, so a sweep or a fit builds them once.
    _NODE_CACHE = {}
    _SPLINE_CACHE = {}

    def __init__(self, model=None, fallback="exact", exact_solver=None,
                 energy_nodes=48):
        if fallback not in ("exact", "wkb", "error"):
            raise ValueError(f"fallback must be 'exact', 'wkb' or 'error', not {fallback!r}")
        if isinstance(model, (str, Path)):
            model = NeuralModel.load(model)
        self._fixed_model = model
        self.fallback = fallback
        self.exact_solver = exact_solver if exact_solver is not None else Noumerov.fast()
        self.energy_nodes = energy_nodes

    # -- model and domain -------------------------------------------------

    def get_model(self, potential):
        """The model this solver uses for ``potential``, or ``None``."""
        name = type(potential).__name__
        if self._fixed_model is not None:
            return self._fixed_model if self._fixed_model.barrier == name else None
        return get_shipped_model(name)

    @staticmethod
    def _parameters(potential, model):
        values = {"total_height": potential.fermi_level + potential.work_function,
                  "field": potential.electric_field}
        for name in model.parameters:
            values[name] = getattr(potential, name, np.nan)
        return values

    def barrier_in_domain(self, potential) -> bool:
        """Whether a model exists for this barrier and its parameters were trained on."""
        model = self.get_model(potential)
        if model is None:
            return False
        return all(np.isfinite(value) and lo <= value <= hi
                   for name, value in self._parameters(potential, model).items()
                   for lo, hi in [model.domain[name]])

    def in_validated_band(self, potential, energies):
        """Energies whose barrier height falls inside the trained range."""
        energies = np.asarray(energies, dtype=np.float64)
        model = self.get_model(potential)
        if model is None:
            return np.zeros(energies.shape, dtype=bool)
        h = potential.fermi_level + potential.work_function - energies
        lo, hi = model.domain["barrier_height"]
        # Round-off slack: energies built as W - h come back as h only to
        # within a few ulp, and must not be rejected at the range ends.
        return (h >= lo - 1e-9) & (h <= hi + 1e-9)

    def _fall_back(self, potential, energies, model):
        name = type(potential).__name__
        if self.fallback == "error":
            if model is None:
                raise ValueError(
                    f"No trained model for {name}. Shipped models cover "
                    f"{sorted(SHIPPED_MODELS)}; train one with getelec.training, "
                    f"or use method='noumerov'.")
            values = {k: round(float(v), 4) for k, v in
                      self._parameters(potential, model).items()}
            raise ValueError(
                f"Barrier is outside the trained domain: {values}. "
                f"Trained ranges: {model.domain}.")
        if model is None and name not in NeuralSolver._WARNED:
            NeuralSolver._WARNED.add(name)
            warnings.warn(
                f"No trained model for {name}; using the {self.fallback!r} "
                f"fallback instead. Shipped models cover {sorted(SHIPPED_MODELS)}.",
                stacklevel=3)
        if self.fallback == "wkb":
            return _solutions().calculate_log_kemble_numeric(potential, energies)
        return self.exact_solver.calculate_log_transmission(potential, energies)

    # -- queries ----------------------------------------------------------

    def calculate_log_transmission(self, potential, energies) -> np.ndarray:
        """``ln D`` at each energy."""
        energies = np.asarray(energies, dtype=np.float64)
        if energies.size == 0:
            return np.zeros(0)

        model = self.get_model(potential)
        if model is None or not self.barrier_in_domain(potential):
            return self._fall_back(potential, energies, model)

        if self.fallback == "error":
            outside = ~self.in_validated_band(potential, energies)
            if outside.any():
                raise ValueError(
                    f"{int(outside.sum())} of {energies.size} energies have a "
                    f"barrier height outside the trained range "
                    f"{model.domain['barrier_height']} eV.")

        nodes = self._nodes(energies)
        if nodes is None:
            return self._evaluate(potential, energies, model)

        # Evaluate on the reduced set and spline ln D onto the full grid -- the
        # same trick Noumerov.fast() uses, and what makes the cost flat in grid
        # size rather than linear.
        coarse = energies[nodes]
        log_d = self._evaluate(potential, coarse, model)
        if np.isfinite(log_d).all():
            # For a given grid the spline is a fixed linear map from the node
            # values to the whole grid, so it is built once and every call is a
            # single matrix-vector product: the same numbers as fitting the
            # spline each time, without paying for its construction.
            matrix = self._spline_matrix(energies, nodes)
            if matrix is not None:
                return np.minimum(matrix @ log_d, 0.0)

        finite = np.isfinite(log_d)
        if finite.sum() < 4:
            return self._evaluate(potential, energies, model)

        # Callers pass energies in either direction -- band structures often
        # run downwards -- and CubicSpline needs a strictly increasing abscissa.
        order = np.argsort(coarse[finite])
        x, y = coarse[finite][order], log_d[finite][order]
        keep = np.concatenate(([True], np.diff(x) > 0))
        if keep.sum() < 4:
            return self._evaluate(potential, energies, model)

        interpolated = CubicSpline(x[keep], y[keep])(energies)
        # Below the lowest node the spline extrapolates; ln D is monotonic in
        # energy, so clamp rather than trust it.
        interpolated[energies < x[keep][0]] = y[keep][0]
        return np.minimum(interpolated, 0.0)

    def _nodes(self, energies):
        """
        Reduced energy set, or ``None`` if the grid is already small enough.

        Uniform nodes, plus a few geometrically spaced at the low-energy end
        where ln D has most of its curvature.
        """
        n = int(self.energy_nodes or 0)
        if n <= 0 or energies.size <= n + 12:
            return None
        key = (energies.size, n)
        nodes = NeuralSolver._NODE_CACHE.get(key)
        if nodes is None:
            uniform = np.round(np.linspace(0, energies.size - 1, n)).astype(np.int64)
            low = np.round(np.geomspace(1, max(uniform[1], 2), 12)).astype(np.int64)
            nodes = np.unique(np.concatenate(([0], low, uniform)))
            NeuralSolver._NODE_CACHE[key] = nodes
        return nodes

    @staticmethod
    def _spline_matrix(energies, nodes):
        """
        The linear map from ln D at the nodes to ln D on the whole grid.

        Exactly the cubic spline, with the clamp below the lowest node, that
        the per-call path fits, written as a matrix: a spline is linear in the
        values it interpolates, so fitting it to the identity gives the map.
        Cached per grid (at most 16, about 1.4 MB each at 3000 energies).
        ``None`` if the nodes cannot carry a spline.
        """
        key = (energies.size, float(energies[0]), float(energies[-1]),
               hash(energies.tobytes()))
        cache = NeuralSolver._SPLINE_CACHE
        if key in cache:
            return cache[key]
        # Callers pass energies in either direction -- band structures often
        # run downwards -- and CubicSpline needs a strictly increasing abscissa.
        x = energies[nodes]
        order = np.argsort(x)
        xs = x[order]
        keep = np.concatenate(([True], np.diff(xs) > 0))
        matrix = None
        if keep.sum() >= 4:
            k = int(keep.sum())
            basis = CubicSpline(xs[keep], np.eye(k))(energies)
            basis[energies < xs[keep][0]] = np.eye(k)[0]
            matrix = np.zeros((energies.size, nodes.size))
            matrix[:, order[keep]] = basis
        if len(cache) >= 16:
            cache.clear()
        cache[key] = matrix
        return matrix

    def _evaluate(self, potential, energies, model):
        """Reference plus learned residual, at every energy given."""
        function = FEATURES.get(model.features)
        if function is None:
            raise KeyError(
                f"The model uses features {model.features!r}, which are not "
                f"registered. Call getelec.transmission_solver.register_features("
                f"{model.features!r}, function) before using it.")
        values = self._parameters(potential, model)
        n = energies.size
        W = values["total_height"]
        extra = [np.full(n, values[name]) for name in model.parameters]
        features = function(W - energies, np.full(n, W),
                            np.full(n, values["field"]), *extra)
        # Beyond the trained band in barrier height the network extrapolates,
        # but only where D is tens of e-folds below the peak and so cannot move
        # any integral; the reference still carries the exponential there.
        return REFERENCES[model.reference](potential, energies) + model.predict(features)

    def calculate_transmission(self, potential, energies) -> np.ndarray:
        """Transmission probability at each energy."""
        with np.errstate(over="ignore"):
            transmission = np.exp(np.clip(
                self.calculate_log_transmission(potential, energies), -745.0, 0.0))
        return np.clip(transmission, 0.0, 1.0)
