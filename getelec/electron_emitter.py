"""
Electron emission from metal and semiconductor surfaces.

An emitter wires together four interchangeable pieces:

    barrier   -- the spatial potential profile           (potential_barrier)
    band      -- which energies to sample                (band_structure)
    supply    -- how many electrons arrive at each energy (electron_supply)
    solver    -- how likely each one is to tunnel        (transmission_solver)

and turns them into current density, energy distributions, and Nottingham heat.

Two structural points, both of which used to cost real time:

- The expensive step -- energy grid, supply, transmission -- is computed once
  and memoised against a fingerprint of every parameter it depends on. Asking
  for Nottingham heat after the current density no longer redoes the tunnelling
  calculation, and a fingerprint (rather than a dirty flag) means the cache is
  still correct if a sub-object is mutated directly.

- The semiconductor path needs transmission on four energy grids against the
  same barrier. Those go to the solver as one batched call rather than four
  separate ones.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid

from getelec import constants as const
from getelec.constants import Q

__all__ = ["MetalEmitter", "SemiconductorEmitter"]

# Converts the internal per-nm^2 per-fs electron flux to A/cm^2:
#   1e14 for nm^-2 -> cm^-2, 1e15 for fs^-1 -> s^-1, Q for electrons -> coulombs.
_FLUX_TO_A_PER_CM2 = const.J_PREFACTOR_CONST * 1e14 * 1e15 * Q

#: Applied to the supply function, which already carries its own k_B T (and,
#: at T = 0, correctly carries none). Dividing k_B out of the flux prefactor
#: keeps every result identical to the previous convention for T > 0 while
#: making the T = 0 limit finite instead of collapsing to zero.
_SUPPLY_TO_A_PER_CM2 = _FLUX_TO_A_PER_CM2 / const.KB

# Aliases accepted by update_params, so fitting code can use short names.
_ALIAS_MAP = {
    "field": "electric_field",
    "temp": "temperature",
    "wf": "work_function",
    "fermi": "fermi_level",
    "lower_energy_limit": "lower_energy_lim",
    "upper_energy_limit": "upper_energy_lim",
}

# Attributes that, if present on a sub-object, feed into the cache fingerprint.
# Every public attribute of a shipped component has to be listed: one that is
# not is silently ignored when changed directly on the component.
# test_every_component_setting_enters_the_cache_key enforces it.
_FINGERPRINT_FIELDS = (
    "fermi_level", "work_function", "electric_field", "radius", "gamma",
    "temperature", "energy_resolution", "lower_energy_lim", "upper_energy_lim",
    "top_valence", "band_gap", "electron_eff_mass", "hole_eff_mass",
    "barrier_width", "supply_threshold", "x_metal", "x_vac_plus", "h",
    "max_barrier_width", "n_nodes", "x_start", "x_end", "seed",
    "energy_nodes", "auto_domain", "interpolation_tolerance",
    "fallback", "exact_solver",
    "custom_energy_array", "custom_conduction_array", "custom_valence_array",
    "dos_energies", "dos_values",
    "potential_function", "table_positions", "table_potential", "zero_inside",
    "extra_parameters",
)


def _fingerprint_value(value):
    """
    A value as it enters the cache fingerprint.

    Arrays enter by their contents, copied into bytes: a user-supplied grid
    mutated in place, or replaced by another array, must invalidate the cache,
    and a tuple holding the array itself would compare element-wise. A solver
    held by another solver, as NeuralSolver holds the one it falls back to,
    enters by its own fingerprint, so its settings count as well.
    """
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        # A Customised barrier's extra parameters. Compared item by item, since
        # two dicts holding arrays compare element-wise and raise instead of
        # answering; the values go through this function for the same reason.
        return tuple(sorted((name, _fingerprint_value(item))
                            for name, item in value.items()))
    if hasattr(value, "calculate_transmission"):
        return _component_fingerprint(value)
    return value


def _component_fingerprint(component):
    """
    One component's part of the cache fingerprint: the object, then its settings.

    The object itself comes first, and compares by identity, so replacing a
    component with another of the same class and settings -- a NeuralSolver
    built on a different model, for one -- is still a change.
    """
    return (component, type(component).__name__) + tuple(
        _fingerprint_value(getattr(component, name, None))
        for name in _FINGERPRINT_FIELDS)


def get_energy_integral(energies, integrand):
    """
    Integrate an emission integrand over an energy grid, trapezoidally.

    Endpoints get half weight, which is the right treatment of a boundary the
    integral stops at -- and the valence integrand stops exactly at the band
    edge, where it is largest. A rectangle sum gives the last point full weight
    and the answer then depends on where the grid happens to land.

    Parameters
    ----------
    energies, integrand : np.ndarray

    Returns
    -------
    float
    """
    energies = np.asarray(energies, dtype=float)
    if energies.size < 2:
        return 0.0
    return float(np.trapezoid(integrand, energies))


def _cumulative_transmission(energies, transmission):
    """
    Running integral of D from the bottom of the grid, plus the tail below it.

    For a metal the grid starts where the transmission is already negligible,
    so the tail correction is a small exponential extrapolation. It is not used
    for semiconductors, where the window has a moving lower limit and
    :func:`_band_window_distributions` evaluates it exactly instead.

    Returns
    -------
    np.ndarray
    """
    energies = np.asarray(energies, dtype=float)
    if energies.size < 2:
        raise ValueError("need at least 2 energy points to integrate over energy.")
    running = cumulative_trapezoid(transmission, energies, initial=0.0)
    if transmission[0] > 0 and transmission[1] > transmission[0]:
        rate = ((np.log(transmission[1]) - np.log(transmission[0]))
                / (energies[1] - energies[0]))
        if rate > 0:
            running = running + transmission[0] / rate
    return running


def is_free_electron_weighting(weights) -> bool:
    """
    Whether a band structure's state weights are the free electron ones.

    ``BandStructure.get_state_weights`` returns exactly 1.0 for every band
    structure that models a free electron gas, so the test is on the weights
    themselves rather than on the type of the band: a tabulated density of states
    that happens to match the free electron one is free electron-like, and should
    take the same path.

    Returns
    -------
    bool
    """
    return bool(np.all(np.asarray(weights) == 1.0))


def get_metal_total_energy_distribution(energies, transmission, supply,
                                        weights=None) -> np.ndarray:
    """
    Total energy distribution for a metal, in A/(eV cm^2).

    ``TED(E) = R(E) f(E) integral of D(E_z) dE_z``, the bare occupancy times the
    transmission integrated over normal energy, times the number of states the
    band structure puts at E relative to a free electron gas.

    The one place the metal TED is assembled: :class:`MetalEmitter` and the
    batched sweeps behind :func:`getelec.current_density` both call it, so a
    change to this integrand cannot reach one path and miss the other.

    Parameters
    ----------
    energies, transmission : np.ndarray
        The energy grid, in eV, and D on it.
    supply : electron_supply.Supply
    weights : np.ndarray, optional
        State weights ``R(E)`` from
        :meth:`~getelec.band_structure.BandStructure.get_state_weights`. The
        free electron weights of one are the default.

    Returns
    -------
    np.ndarray
    """
    distribution = (supply.get_occupancy(energies)
                    * _cumulative_transmission(energies, transmission))
    if weights is not None:
        distribution = weights * distribution
    return distribution * _SUPPLY_TO_A_PER_CM2


def get_weighted_normal_supply(energies, supply, weights) -> np.ndarray:
    """
    ``S(E_z)``, what the normal energy distribution multiplies D by, in eV.

    Swapping the order of the double integral over total and normal energy puts
    the supply inside:

        S(E_z) = integral from E_z to the top of the grid of R(E) f(E) dE ,

    so that ``integral NED dE_z`` and ``integral TED dE`` are the same double
    integral evaluated two ways and cannot disagree. With ``R == 1`` this is
    ``l(E_z) - l(E_top)``, the difference of log supplies the normal
    distribution has always used, and the emitter keeps the closed form of ``l``
    there instead -- exact, and the free electron numbers stay untouched.

    With a state weight there is no closed form, so this is the trapezoid rule
    on the grid. The two integration orders then agree to round-off, at the cost
    of the usual ``(h / k_B T)^2 / 12`` in the pointwise distribution: about
    0.5% at ``h = 0.01 eV`` and 300 K.

    The integral is accumulated downwards from the top of the grid, and not as
    the total minus a running integral from the bottom. The two are the same
    algebraically, but S has to fall through the whole Fermi tail -- 1e-55 of
    its value at the bottom by the top of a metal's grid -- while the total and
    the running integral both stay of order one, so the subtraction leaves an
    absolute floor of a few times the machine epsilon, around 4e-16 eV. That
    floor is reached well below the top of the barrier, where D is of order one
    and multiplies it straight into the distribution: at 1.5 V/nm and 300 K the
    normal distribution integrated to 945 times the current density before this
    was accumulated the other way round.

    Returns
    -------
    np.ndarray
    """
    integrand = weights * supply.get_occupancy(energies)
    # Integrating the reversed grid against decreasing energy accumulates from
    # the top down, so every partial sum is of the size of the answer.
    return cumulative_trapezoid(integrand[::-1], -energies[::-1],
                                initial=0.0)[::-1]


def _band_window_distributions(energies, lower_limits, solver, potential, supply):
    """
    Energy distributions for one band, from the double integral itself.

    The emission current for a band is a double integral over total energy E and
    normal energy E_z, on the wedge ``lower(E) <= E_z <= E``. Doing the E_z
    integral first gives the **total** energy distribution; doing them the other
    way round gives the **normal** one. Same integral, same current, different
    functions -- each has to be built its own way.

    Returns
    -------
    ted : np.ndarray
        On ``energies``, non-negative.
    window : np.ndarray
        The window integral itself, ``g(E) = integral of D dE_z`` between the
        limits, on ``energies``, in eV. The TED is this times the occupancy;
        it is returned separately only so the interface can plot it. See
        :meth:`SemiconductorEmitter.calculate_window_integrated_transmission`.
    normal : np.ndarray
        The normal-energy grid the NED lives on.
    ned : np.ndarray
        On ``normal``, non-negative.
    """
    if energies.size < 2:
        zeros = np.zeros_like(energies)
        return zeros, zeros, energies, zeros

    low = np.minimum(lower_limits, energies)
    high = np.maximum(lower_limits, energies)

    step = abs(energies[1] - energies[0])
    grid = np.arange(low.min(), high.max() + step, step)
    transmission = solver.calculate_transmission(potential, grid)

    # The window integral from one running integral of D, read off at both
    # ends. Fourth order throughout: the running integral carries its
    # Euler-Maclaurin correction, and it is read off between grid points by
    # cubic Hermite interpolation using its known derivative, D itself. Linear
    # interpolation there was the dominant error -- ~(h^2/8) D' against a window
    # that shrinks to zero at the band edge, so up to 0.3% near the edges.
    antiderivative = _antiderivative(grid, transmission)
    window = (_hermite(energies, grid, antiderivative, transmission)
              - _hermite(lower_limits, grid, antiderivative, transmission))
    ted = supply.get_occupancy(energies) * window

    # The NED lives on its own grid. It is a distribution in normal energy, and
    # normal energy spans every value the windows reach -- which for m* > m
    # extends *below* the band edge, because the transverse energy can then
    # exceed the total kinetic energy. Evaluating it only on the total-energy
    # grid truncates it there and loses ~17% of the current at m* = 1.64.
    normal, normal_transmission = grid, transmission
    e_min, e_max = energies.min(), energies.max()
    span = energies[-1] - energies[0]
    slope = (lower_limits[-1] - lower_limits[0]) / span if span else 1.0

    # At a conduction edge (lower(E_C) = E_C, slope 1 - m*/m) the NED changes
    # over |1 - m*/m| k_B T: it rises from zero above E_C for m* < m, and its
    # tail into the gap has that width for m* > m. Near m* = m that is far below
    # any sensible grid step -- 0.0005 eV at m* = 0.98 -- and the trapezoid
    # integral of the NED missed the edge by up to 2.6%. So when the feature is
    # narrower than the grid, resolve it with points of its own; the NED is
    # exact pointwise, so this only costs a few dozen extra transmissions.
    # For m* > m the NED also has a kink at E_C, where the tail meets the band,
    # so E_C itself goes on the grid: a kink between grid points costs the
    # trapezoid integral its second-order accuracy.
    edge_index = np.argmin(energies)
    extra = np.empty(0)
    if slope < 1 and np.isclose(lower_limits[edge_index], e_min):
        width = abs(slope) * const.KB * supply.temperature
        if 0 < width < 5 * step:
            offsets = width * np.geomspace(1e-3, 40.0, 60)
            extra = e_min + offsets if slope > 0 else e_min - offsets
        if slope < 0:
            extra = np.append(extra, e_min)
        extra = np.setdiff1d(extra[(extra > grid[0]) & (extra < grid[-1])], grid)
    if extra.size:
        normal = np.concatenate([grid, extra])
        normal_transmission = np.concatenate(
            [transmission, solver.calculate_transmission(potential, extra)])
        order = np.argsort(normal)
        normal, normal_transmission = normal[order], normal_transmission[order]

    order = np.argsort(lower_limits)
    inverse = np.interp(normal, lower_limits[order], energies[order],
                        left=energies[order][0], right=energies[order][-1])
    if abs(slope) < 1e-12:
        # m* = m exactly: the window is [E_C, E] for every E, so E_z must lie
        # above E_C and the condition is simply E >= E_z -- the metal-like case.
        start = np.where(normal >= lower_limits[0], np.maximum(normal, e_min), np.inf)
        end = np.full_like(normal, e_max)
    elif slope > 0:
        start = np.maximum(normal, e_min)
        end = np.minimum(inverse, e_max)
    else:
        start = np.maximum(np.maximum(normal, inverse), e_min)
        end = np.full_like(normal, e_max)

    weight = np.zeros_like(normal)
    filled = start < end
    weight[filled] = (supply.get_log_supply(start[filled])
                      - supply.get_log_supply(end[filled]))
    ned = normal_transmission * weight

    # The window is clamped where the TED is, and for the same reason: a
    # valence grid runs to ``top_valence + dE`` (the published convention), so
    # its last point can sit just above E_V, where the limits invert and the
    # window comes out negative. Clamping both keeps ``TED == f * window``
    # exact, so the plotted g cannot disagree with the distribution.
    return (np.maximum(ted, 0.0), np.maximum(window, 0.0),
            normal, np.maximum(ned, 0.0))


def _antiderivative(grid, values):
    """
    Running integral of ``values`` on a uniform grid, to fourth order.

    The trapezoid rule's error is ``-(h^2/12) (f'(x) - f'(x_0)) + O(h^4)``
    (Euler-Maclaurin), so subtracting that term, with f' from second-order
    differences, leaves a running integral accurate to O(h^4).
    """
    running = cumulative_trapezoid(values, grid, initial=0.0)
    h = grid[1] - grid[0]
    derivative = np.gradient(values, grid, edge_order=2)
    return running - (h * h / 12.0) * (derivative - derivative[0])


def _hermite(x, grid, values, derivative):
    """Cubic Hermite interpolation of ``values``, whose derivative is known."""
    i = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, grid.size - 2)
    h = grid[i + 1] - grid[i]
    t = (x - grid[i]) / h
    t2 = t * t
    t3 = t2 * t
    return ((2.0 * t3 - 3.0 * t2 + 1.0) * values[i]
            + (t3 - 2.0 * t2 + t) * h * derivative[i]
            + (-2.0 * t3 + 3.0 * t2) * values[i + 1]
            + (t3 - t2) * h * derivative[i + 1])


class _EmitterBase:
    """Shared parameter handling and caching for the concrete emitters."""

    def __init__(self, potential, solver, supply, band):
        self.potential = potential
        self.solver = solver
        self.supply = supply
        self.band = band
        self._cache = None

    # -- parameters --------------------------------------------------------

    def update_params(self, **kwargs):
        """
        Set a parameter everywhere it appears.

        ``fermi_level`` lives on the barrier, the supply and (for semiconductors)
        the band structure. Setting it in one place only is a common and silent
        source of inconsistent results, so this sets it on every sub-object that
        has it.

        Accepts the short aliases ``field``, ``temp``, ``wf``, ``fermi``.

        Raises
        ------
        KeyError
            If no sub-object has the requested attribute. A silently ignored
            typo is worse than a loud failure.

        Examples
        --------
        >>> emitter.update_params(field=4.0, temperature=1000)   # doctest: +SKIP
        >>> emitter.update_params(temp=300, wf=4.5)              # doctest: +SKIP
        """
        targets = (self.potential, self.supply, self.band, self.solver)
        for key, value in kwargs.items():
            attr = _ALIAS_MAP.get(key, key)
            hits = [obj for obj in targets if hasattr(obj, attr)]
            if not hits:
                raise KeyError(
                    f"No component of this emitter has a parameter named "
                    f"{key!r} (resolved to {attr!r}). Components: "
                    + ", ".join(type(o).__name__ for o in targets)
                )
            for obj in hits:
                setattr(obj, attr, value)
        self._cache = None

    def __repr__(self):
        """
        The parameters, not the memory address.

        An emitter is a bundle of settings, and the first thing anyone does at a
        prompt is print it to see what those settings are.
        """
        params = self.get_params()
        shown = [f"{name}={params[name]:g}"
                 for name in ("work_function", "fermi_level", "electric_field",
                              "temperature")
                 if name in params]
        pieces = "+".join(type(getattr(self, part)).__name__
                          for part in ("potential", "band", "supply", "solver"))
        return f"{type(self).__name__}({', '.join(shown)})  [{pieces}]"

    def get_params(self) -> dict:
        """Every parameter currently set on any sub-object, for logging or reuse."""
        params = {}
        for obj in (self.potential, self.supply, self.band, self.solver):
            for field in _FINGERPRINT_FIELDS:
                if hasattr(obj, field):
                    params[field] = getattr(obj, field)
        return params

    def _fingerprint(self):
        """Hashable snapshot of everything ``_calculate_base_data`` depends on."""
        return (
            type(self).__name__,
            tuple(_component_fingerprint(obj)
                  for obj in (self.potential, self.supply, self.band, self.solver)),
        )

    def _base_data(self):
        """Memoised wrapper around the subclass's expensive computation."""
        key = self._fingerprint()
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, self._calculate_base_data())
        return self._cache[1]

    def _calculate_base_data(self):  # pragma: no cover - interface
        raise NotImplementedError


    # -- wavefunction ------------------------------------------------------

    def _psi_solver(self):
        """
        The solver to use for a wavefunction, or a clear error.

        Not every solver has one to give: the learned and semiclassical
        solvers return a transmission coefficient without ever forming psi, and
        the fast Noumerov preset deliberately discards it. Rather than quietly
        substituting a different solver -- which would return a wavefunction
        that did not correspond to the transmission this emitter reports -- say
        so and name the fix.
        """
        solver = self.solver
        if hasattr(solver, "calculate_psi"):
            return solver
        raise TypeError(
            f"{type(solver).__name__} does not produce a wavefunction. "
            f"Build the emitter with a full Noumerov solver, either "
            f"method='noumerov' (the default) or by passing "
            f"transmission_solver.Noumerov() directly. NoumerovFast, "
            f"NeuralSolver and WKB all return transmission without forming "
            f"psi. NoumerovReference forms it one energy at a time, through "
            f"transmission_solver.calculate_noumerov_reference.")

    def _psi_memory_note(self, n_energy, n_grid):
        megabytes = 16.0 * n_energy * n_grid / 1e6
        if megabytes > 100.0:
            warnings.warn(
                f"calculate_psi will allocate about {megabytes:.0f} MB "
                f"({n_energy} energies x {n_grid} grid points, 16 bytes each). "
                f"Pass energies=... to ask for fewer, or x_points=... to return "
                f"fewer positions.", ResourceWarning, stacklevel=3)


class _MetalPsiMixin:
    """Wavefunction access for a single-band emitter."""

    def calculate_psi(self, energies=None, x_points=None):
        """
        The wavefunction psi(E, x) for this emitter's barrier.

        Companion to :meth:`calculate_transmission_coefficient`: same barrier,
        same solver, same energies by default -- but the wavefunction itself
        rather than the flux ratio extracted from it.

        Parameters
        ----------
        energies : array_like, optional
            Energies in eV. Defaults to the emitter's own energy grid, which is
            typically several hundred points; pass a subset when you do not need
            all of it, because the array scales as energies x grid points.
        x_points : array_like, optional
            Positions in nm. Defaults to the solver's spatial grid. Values are
            snapped to the nearest grid node; this trims what is returned but
            not the work, since the recurrence steps through every node anyway.

        Returns
        -------
        energies, x, psi : np.ndarray
            ``psi`` has shape ``(len(energies), len(x))``, complex, normalised
            to unit outgoing flux in the vacuum.

        Raises
        ------
        TypeError
            If the emitter's solver does not form a wavefunction.

        Examples
        --------
        >>> emitter = getelec.metal_emitter(field=5.0)          # doctest: +SKIP
        >>> energies, x, psi = emitter.calculate_psi()          # doctest: +SKIP
        >>> density = np.abs(psi) ** 2                          # doctest: +SKIP
        """
        solver = self._psi_solver()
        if energies is None:
            energies = self._base_data()[0]
        energies = np.atleast_1d(np.asarray(energies, dtype=float))
        self._psi_memory_note(energies.size, solver._grid().size
                              if x_points is None else np.size(x_points))
        x, psi = solver.calculate_psi(self.potential, energies, x_points=x_points)
        return energies, x, psi

    def calculate_probability_current(self, energies=None):
        """
        Probability current j(E, x), which must not depend on x.

        The Noumerov scheme does not enforce current conservation, so how much
        this varies across the grid is an independent check on the integration.

        Returns
        -------
        energies, x, current : np.ndarray
        """
        solver = self._psi_solver()
        if energies is None:
            energies = self._base_data()[0]
        energies = np.atleast_1d(np.asarray(energies, dtype=float))
        x, current = solver.calculate_probability_current(self.potential, energies)
        return energies, x, current


class MetalEmitter(_MetalPsiMixin, _EmitterBase):
    """
    Thermal-field emission from a metal surface.

    Parameters
    ----------
    potential : potential_barrier.Barrier
        Barrier profile (work function, field, Fermi level).
    solver : transmission_solver.TransmissionSolver
        How transmission probabilities are evaluated.
    supply : electron_supply.Supply
        Electron supply function.
    band : band_structure.BandStructure
        Energy grid generator.

    Examples
    --------
    >>> from getelec import potential_barrier, band_structure  # doctest: +SKIP
    >>> from getelec import transmission_solver, electron_supply  # doctest: +SKIP
    >>> emitter = MetalEmitter(                                 # doctest: +SKIP
    ...     potential_barrier.SchottkyPotential(7.5, 4.5, 5.0),
    ...     transmission_solver.Noumerov(),
    ...     electron_supply.LogFermiDirac(7.5, 300.0),
    ...     band_structure.SmartMetal(),
    ... )
    >>> emitter.calculate_current_density()                     # doctest: +SKIP
    """

    def _calculate_base_data(self):
        """
        Energy grid, supply, and transmission for the current parameters.

        Returns
        -------
        energies, supply_vals, trans_vals : np.ndarray
        """
        energies = self.band.generate_band_structure(
            fermi_level=self.potential.fermi_level,
            work_function=self.potential.work_function,
            electric_field=self.potential.electric_field,
            temperature=self.supply.temperature,
        )
        # The log supply l(E), whichever supply class is attached: get_supply
        # is l for LogFermiDirac but the bare occupancy f for FermiDirac, and
        # f D is not the normal energy distribution -- it lacks the integral
        # over transverse energy, which put its integral 1/d times above J
        # (4.5x at 5 V/nm).
        supply_vals = self.supply.get_log_supply(energies)
        trans_vals = self.solver.calculate_transmission(self.potential, energies)
        return energies, supply_vals, trans_vals

    def calculate_transmission_coefficient(self):
        """
        Transmission probability D(E) on the emitter's energy grid.

        Returns
        -------
        energies, transmission : np.ndarray
        """
        energies, _, transmission = self._base_data()
        return energies, transmission

    def calculate_supply_function(self):
        """
        Supply function l(E) on the emitter's energy grid, in eV.

        Returns
        -------
        energies, supply : np.ndarray
        """
        energies, supply, _ = self._base_data()
        return energies, supply

    def calculate_total_energy_distribution(self):
        """
        Total energy distribution, in A/(eV cm^2).

        ``TED(E) = R(E) f(E) * integral of D(E_z) dE_z`` up to E -- the bare
        occupancy times the transmission integrated over normal energy. See
        :meth:`~getelec.electron_supply.Supply.get_occupancy` for why this uses
        ``f`` while the normal distribution uses ``l``.

        ``R(E)`` is the band structure's state weight, one everywhere for a free
        electron gas and so for every band structure but
        :class:`~getelec.band_structure.DensityOfStatesMetal`.

        Returns
        -------
        energies, ted : np.ndarray
        """
        energies, _, transmission = self._base_data()
        return energies, get_metal_total_energy_distribution(
            energies, transmission, self.supply,
            self.band.get_state_weights(energies))

    def calculate_normal_energy_distribution(self):
        """
        Normal energy distribution, in A/(eV cm^2).

        ``NED(E_z) = S(E_z) * D(E_z)``. For a metal the effective mass equals
        the free mass, so the transverse window is unrestricted and this is the
        whole of it.

        For a free electron gas ``S`` is the log supply ``l(E_z)``. When the
        band structure weights its states -- a tabulated density of states --
        the supply is instead integrated over the weighted occupancy above
        ``E_z``, which is what keeps ``integral NED dE_z == integral TED dE``.
        See :func:`get_weighted_normal_supply`.

        Returns
        -------
        energies, ned : np.ndarray
        """
        energies, supply, transmission = self._base_data()
        weights = self.band.get_state_weights(energies)
        if not is_free_electron_weighting(weights):
            supply = get_weighted_normal_supply(energies, self.supply, weights)
        return energies, supply * transmission * _SUPPLY_TO_A_PER_CM2

    def calculate_current_density(self) -> float:
        """
        Emitted current density, in A/cm^2.

        Returns
        -------
        float
        """
        energies, ted = self.calculate_total_energy_distribution()
        return get_energy_integral(energies, ted)

    def calculate_nottingham_heat(self) -> float:
        """
        Nottingham heat P_N, in W/cm^2.

        The net power carried away by the emitted electrons relative to the
        replacement energy, taken at the Fermi level:

            P_N = integral of (E - E_F) * TED(E) dE .

        **Negative means the emitter heats, positive that it cools.** An
        electron leaving from below E_F is replaced by a hotter one at E_F, so
        cold field emission heats the tip (P_N < 0); at high temperature the
        emission moves above E_F and the tip cools (P_N > 0).

        Returns
        -------
        float

        Notes
        -----
        Reported in **W/cm^2**, the same area unit as the current density:
        the TED is in A/(eV cm^2), so its energy moment is in W/cm^2 directly.
        """
        energies, ted = self.calculate_total_energy_distribution()
        moment = get_energy_integral(energies,
                                       (energies - self.supply.fermi_level) * ted)
        return moment


class SemiconductorEmitter(_MetalPsiMixin, _EmitterBase):
    """
    Thermal-field emission from a semiconductor surface.

    Conduction and valence band channels are treated separately, each with its
    own effective mass, and summed.

    Parameters
    ----------
    potential : potential_barrier.Barrier
    solver : transmission_solver.TransmissionSolver
    supply : electron_supply.Supply
    band : band_structure.Semiconductor
    """

    def _calculate_base_data(self):
        """
        Band grids, supplies, and transmission probabilities.

        The two transmission calculations share one barrier, so they are issued
        as a single batched solver call.

        The effective masses enter through the window limits of Eq. (8), which
        the distributions integrate D between -- not through the transmission.
        So the scaled grids are returned as limits, and D is solved for on the
        band grids only.

        Returns
        -------
        e_cb, e_vb, supply_cb, supply_vb, trans_cb, trans_vb : np.ndarray
        e_cb2, e_vb2 : np.ndarray
            The scaled grids: the lower window limit belonging to each point of
            ``e_cb`` and ``e_vb``.
        """
        e_cb1, e_cb2, e_vb1, e_vb2 = self.band.generate_band_structure(
            fermi_level=self.potential.fermi_level,
            work_function=self.potential.work_function,
            electric_field=self.potential.electric_field,
            temperature=self.supply.temperature,
        )

        # l(E) explicitly: the current integral needs the log supply whichever
        # supply object is attached.
        supply_cb = self.supply.get_log_supply(e_cb1)
        supply_vb = self.supply.get_log_supply(e_vb1)

        # One batched solver call for the two band grids.
        trans_cb, trans_vb = self.solver.calculate_transmission_batch(
            [self.potential] * 2, [e_cb1, e_vb1])

        return (e_cb1, e_vb1, supply_cb, supply_vb, trans_cb, trans_vb,
                e_cb2, e_vb2)


    def calculate_psi(self, energies=None, x_points=None):
        """
        The wavefunction for each band, on that band's energy grid.

        Parameters
        ----------
        energies : tuple of array_like, optional
            ``(conduction, valence)``. Defaults to the emitter's own grids.
        x_points : array_like, optional
            Positions in nm, shared by both bands.

        Returns
        -------
        energies_cb, x_cb, psi_cb, energies_vb, x_vb, psi_vb : np.ndarray
            Each band gets its own spatial grid: the domain is sized to enclose
            the barrier down to the lowest energy requested, and the two bands
            start at different energies.
        """
        solver = self._psi_solver()
        base = self._base_data()
        if energies is None:
            grids = (base[0], base[1])
        else:
            grids = tuple(np.atleast_1d(np.asarray(g, dtype=float)) for g in energies)

        results = []
        for grid in grids:
            if grid.size == 0:
                results.append((np.empty(0), np.empty((0, 0), dtype=complex)))
                continue
            self._psi_memory_note(grid.size, solver._grid().size
                                  if x_points is None else np.size(x_points))
            results.append(solver.calculate_psi(self.potential, grid,
                                                x_points=x_points))
        return (grids[0], results[0][0], results[0][1],
                grids[1], results[1][0], results[1][1])

    def calculate_probability_current(self, energies=None):
        """
        Probability current for each band, on that band's grid.

        A stationary scattering state carries a current that does not depend on
        position, and the Noumerov scheme does not enforce that -- so how much
        this varies across the grid is an independent check on the integration,
        band by band.

        Returns
        -------
        energies_cb, x_cb, current_cb, energies_vb, x_vb, current_vb : np.ndarray
        """
        solver = self._psi_solver()
        base = self._base_data()
        if energies is None:
            grids = (base[0], base[1])
        else:
            grids = tuple(np.atleast_1d(np.asarray(g, dtype=float)) for g in energies)

        results = []
        for grid in grids:
            if grid.size == 0:
                results.append((np.empty(0), np.empty((0, 0))))
                continue
            self._psi_memory_note(grid.size, solver._grid().size)
            results.append(solver.calculate_probability_current(self.potential, grid))
        return (grids[0], results[0][0], results[0][1],
                grids[1], results[1][0], results[1][1])

    def calculate_transmission_coefficient(self):
        """
        Transmission for each band, on its own energy grid.

        D(E) itself, the tunnelling probability, in [0, 1]. The effective masses
        do not appear here: they set the window limits of Eq. (8) that
        :meth:`calculate_total_energy_distribution` integrates D between.

        Returns
        -------
        energies_cb, trans_cb, energies_vb, trans_vb : np.ndarray
        """
        e_cb, e_vb, _, _, trans_cb, trans_vb, _, _ = self._base_data()
        return e_cb, trans_cb, e_vb, trans_vb

    def calculate_window_integrated_transmission(self):
        """
        Transmission integrated over the emission window, for each band.

        ``g(E) = integral of D(E_z) dE_z`` from ``lower(E)`` to ``E``, the
        window of Eq. (8), in **eV** -- a transmission times the width of the
        window it was integrated over, so it is *not* a probability and is not
        bounded by 1.

        Why it exists. :meth:`calculate_transmission_coefficient` returns
        ``D(E)``, which does not depend on the effective masses at all: the
        barrier contains no mass. The masses enter only through the limits this
        integral runs between, so ``g(E)`` is the smallest object that shows
        how the transmission enters the current *as the masses change it*. A
        heavier mass opens the window wider and raises ``g``, which is the same
        direction the current moves in.

        ``TED(E) = f(E) g(E)`` -- this is the same ``g`` the distributions and
        the current are built from, read out rather than recomputed, so the
        plotted curve cannot drift from the physics.

        Not to be confused with the by-parts integrand ``g'(E)`` of Eq. (15),
        which is negative throughout the valence band for every hole mass and
        is not returned anywhere.

        Returns
        -------
        energies_cb, window_cb, energies_vb, window_vb : np.ndarray
        """
        e_cb, e_vb, _, _, _, _, scaled_cb, scaled_vb = self._base_data()
        _, window_cb, _, _ = _band_window_distributions(
            e_cb, scaled_cb, self.solver, self.potential, self.supply)
        _, window_vb, _, _ = _band_window_distributions(
            e_vb, scaled_vb, self.solver, self.potential, self.supply)
        return e_cb, window_cb, e_vb, window_vb

    def calculate_supply_function(self):
        """
        Supply function l(E) for each band, on its own energy grid.

        Returns
        -------
        energies_cb, supply_cb, energies_vb, supply_vb : np.ndarray
        """
        e_cb, e_vb, supply_cb, supply_vb, _, _, _, _ = self._base_data()
        return e_cb, supply_cb, e_vb, supply_vb

    def calculate_current_density(self) -> float:
        """
        Total current density summed over both bands, in A/cm^2.

        The integral of the total energy distribution, band by band. 

        Returns
        -------
        float
        """
        e_cb, ted_cb, e_vb, ted_vb = self.calculate_total_energy_distribution()
        return (get_energy_integral(e_cb, ted_cb)
                + get_energy_integral(e_vb, ted_vb))

    def calculate_total_energy_distribution(self):
        """
        Total energy distribution for each band, in A/(eV cm^2).

        ``TED(E) = f(E) * integral of D(E_z) dE_z`` over the band's window,
        evaluated directly rather than recovered from its derivative. See
        :func:`_band_window_distributions`.

        Returns
        -------
        energies_cb, ted_cb, energies_vb, ted_vb : np.ndarray
        """
        e_cb, e_vb, _, _, _, _, scaled_cb, scaled_vb = self._base_data()
        ted_cb, _, _, _ = _band_window_distributions(e_cb, scaled_cb, self.solver,
                                                     self.potential, self.supply)
        ted_vb, _, _, _ = _band_window_distributions(e_vb, scaled_vb, self.solver,
                                                     self.potential, self.supply)
        return (e_cb, ted_cb * _SUPPLY_TO_A_PER_CM2,
                e_vb, ted_vb * _SUPPLY_TO_A_PER_CM2)

    def calculate_normal_energy_distribution(self):
        """
        Normal energy distribution for each band, in A/(eV cm^2).

        ``NED(E_z) = D(E_z) * [ l(E_z) - l(E_upper) ]`` -- Stratton Eq. (95) for
        the valence band, Eq. (49) for the conduction band. Non-negative by
        construction.

        Returned on its own ``E_z`` grid, which extends **below** the
        total-energy grid. For the conduction band with ``m_e* > m`` this puts
        part of the NED in the band gap, and that is correct: ``E_z`` is the
        normal energy in vacuum, and the conserved parallel momentum carries
        up to ``(m_e*/m)(E - E_C)`` of transverse energy there, more than the
        electron's whole kinetic energy in the band. Truncating at ``E_C``
        would lose that part of the current. See the GUIDE section
        "Semiconductors: the two energy distributions" and
        :func:`_band_window_distributions`.

        Returns
        -------
        energies_cb, ned_cb, energies_vb, ned_vb : np.ndarray
        """
        e_cb, e_vb, _, _, _, _, scaled_cb, scaled_vb = self._base_data()
        _, _, x_cb, ned_cb = _band_window_distributions(e_cb, scaled_cb, self.solver,
                                                        self.potential, self.supply)
        _, _, x_vb, ned_vb = _band_window_distributions(e_vb, scaled_vb, self.solver,
                                                        self.potential, self.supply)
        return (x_cb, ned_cb * _SUPPLY_TO_A_PER_CM2,
                x_vb, ned_vb * _SUPPLY_TO_A_PER_CM2)

    def calculate_nottingham_heat(self) -> float:
        """
        Nottingham heat P_N summed over both bands, in W/cm^2.

        ``P_N = integral of (E - E_R) * TED(E) dE`` with the replacement energy
        ``E_R`` at the Fermi level, per Eqs. (13) and (19). **Negative means the
        emitter heats, positive that it cools**, exactly as for a metal: an
        electron leaving from below E_F is replaced by a hotter one.

        At low field the valence band leads: electrons leave from below E_F, so
        the tip heats (P_N < 0). As the field rises the conduction band, above
        E_F, takes over and the sign flips to cooling (P_N > 0).

        Returns
        -------
        float
        """
        e_cb, ted_cb, e_vb, ted_vb = self.calculate_total_energy_distribution()
        fermi = self.supply.fermi_level
        moment = (get_energy_integral(e_cb, (e_cb - fermi) * ted_cb)
                  + get_energy_integral(e_vb, (e_vb - fermi) * ted_vb))
        return moment
