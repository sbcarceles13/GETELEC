"""
Test suite for GETELEC.

Run with ``pytest``. No virtual environment layout is assumed and no scripts are
launched as subprocesses, so this works the same on Linux, macOS and Windows and
in CI.

The important test is :func:`test_noumerov_matches_reference_implementation`,
which reimplements the original dense-array Noumerov independently, in plain
NumPy, and checks the optimised kernel against it. That is what makes the
rewrite safe: if an optimisation ever changes the physics, this fails.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

#: Repository root, derived from this file rather than the working directory,
#: so the file-based tests below pass wherever pytest is invoked from.
ROOT = Path(__file__).resolve().parent.parent
from scipy.integrate import quad

import getelec
from getelec import constants as const
from getelec.band_structure import Metal, SmartMetal, Semiconductor
from getelec.electron_supply import FermiDirac, LogFermiDirac
from getelec.potential_barrier import SchottkyPotential
from getelec.transmission_solver import (Noumerov, NoumerovFast, NoumerovReference,
                                         calculate_noumerov_reference)
from getelec.transmission_solutions import WKB
# Numba's njit, or a pass-through when Numba is missing, as for the package kernels.
from getelec._kernels import njit


# --------------------------------------------------------------------------
# Reference implementations, written for clarity rather than speed.
# --------------------------------------------------------------------------

def reference_noumerov(potential, energies, x_metal=-1.0, x_vac_plus=10.0,
                      h=1e-3, max_barrier_width=3.0, x=None):
    """
    Dense-array Noumerov, the way the original code did it.

    Materialises the full (n_energy, n_grid) wavefunction and inverts a stack of
    2x2 matrices with LAPACK. Slow and memory-hungry, which is the point: it
    shares no code with the optimised kernel. Pass ``x`` to run it on the
    solver's own grid; the recurrence and the matching stay independent.
    """
    if x is None:
        x_end = max_barrier_width + x_vac_plus
        x = np.arange(x_end, x_metal - h, -h)
    V = potential.get_potential(x)
    n_energy = energies.shape[0]

    e_minus_v = (energies[:, None] - V[None, :]) / const.HBAR2_2M
    t_vals = -(h ** 2 / 12) * e_minus_v
    k = np.sqrt(e_minus_v.astype(np.complex128))

    psi = np.zeros_like(k, dtype=np.complex128)
    psi[:, 0] = np.exp(1j * k[:, 0] * x[0])
    psi[:, 1] = np.exp(1j * k[:, 0] * x[1])

    for i in range(n_energy):
        for j in range(1, x.size - 1):
            num = ((2 + 10 * t_vals[i, j]) * psi[i, j]
                   - (1 - t_vals[i, j - 1]) * psi[i, j - 1])
            psi[i, j + 1] = num / (1 - t_vals[i, j + 1])

    k_a, k_b, k_vac = k[:, -1], k[:, -2], k[:, 0]
    xa, xb = x[-1], x[-2]
    mat = np.zeros((n_energy, 2, 2), dtype=np.complex128)
    mat[:, 0, 0] = np.exp(1j * k_a * xa)
    mat[:, 0, 1] = np.exp(-1j * k_a * xa)
    mat[:, 1, 0] = np.exp(1j * k_b * xb)
    mat[:, 1, 1] = np.exp(-1j * k_b * xb)
    coeffs = np.einsum("...ij,...j->...i", np.linalg.inv(mat), psi[:, [-1, -2]])

    with np.errstate(divide="ignore", invalid="ignore"):
        trans = (k_vac.real / k_a.real) / np.abs(coeffs[:, 0]) ** 2
    trans[~np.isfinite(trans)] = 0.0
    return trans


def reference_gamow(energy, fermi_level, work_function, field):
    """Gamow exponent by adaptive quadrature, independent of the fast path."""
    b = energy - fermi_level - work_function
    disc = b * b - field * const.COULOMB_CONST
    if disc <= 0:
        return None
    root = np.sqrt(disc)
    x1 = (-b - root) / (2 * field)
    x2 = (-b + root) / (2 * field)

    def integrand(x):
        v = fermi_level + work_function - field * x - const.COULOMB_CONST / (4 * x)
        return np.sqrt(max(v - energy, 0.0))

    value, _ = quad(integrand, x1, x2, limit=200)
    return const.WKB_PREFACTOR * value


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def barrier():
    return SchottkyPotential(fermi_level=7.5, work_function=4.5, electric_field=5.0)


@pytest.fixture
def emitter():
    return getelec.metal_emitter(work_function=4.5, fermi_level=7.5, temperature=300.0,
                                 field=5.0)


# --------------------------------------------------------------------------
# Noumerov solver
# --------------------------------------------------------------------------

def test_noumerov_matches_reference_implementation(barrier):
    """
    The optimised kernel must reproduce the dense reference to round-off.

    Pinned to ``seed="plane"`` and ``auto_domain=False`` because the reference
    reproduces the original boundary condition and fixed 13 nm domain. The default seed is now WKB, which is a
    deliberate accuracy improvement and legitimately gives different numbers --
    see test_wkb_seed_is_closer_to_the_converged_answer. The reference runs on
    the solver's grid, which is offset by half a step so that the surface falls
    between two nodes; what is compared is the recurrence and the matching.
    """
    energies = np.arange(2.0, 9.0, 0.25)
    solver = Noumerov(x_metal=-1.0, x_vac_plus=10.0, max_barrier_width=3.0,
                      h=1e-3, seed="plane", auto_domain=False)
    fast = solver.calculate_transmission(barrier, energies)
    slow = reference_noumerov(barrier, energies, x=solver._grid())

    mask = slow > 0
    assert mask.sum() > 5, "reference produced no usable values"
    rel = np.abs(fast[mask] - slow[mask]) / slow[mask]
    # The recurrence amplifies exponentially, so 1e-9 is already deep into
    # round-off territory for transmission values spanning 20 decades.
    assert rel.max() < 1e-9, f"max relative deviation {rel.max():.3e}"


def test_noumerov_batch_matches_individual_calls(barrier):
    """Batched and one-at-a-time solves must agree exactly."""
    solver = Noumerov()
    grids = [np.arange(3.0, 8.0, 0.1), np.arange(4.0, 9.0, 0.2), np.array([5.0, 6.0])]
    barriers = [SchottkyPotential(7.5, 4.5, f) for f in (4.0, 5.0, 6.0)]

    batched = solver.calculate_transmission_batch(barriers, grids)
    for b, g, got in zip(barriers, grids, batched):
        # Not bit-identical: a batch shares one spatial grid, sized to enclose
        # the widest barrier in the set, while a single solve uses that
        # barrier's own width. The resulting difference (~1e-5) is two orders
        # below the discretisation error, but it is not zero.
        np.testing.assert_allclose(got, solver.calculate_transmission(b, g),
                                   rtol=1e-4)


def test_transmission_is_bounded_and_monotonic(barrier):
    """Physical sanity: T in [0, 1] and rising with energy below the barrier."""
    energies = np.arange(2.0, 9.0, 0.1)
    trans = Noumerov().calculate_transmission(barrier, energies)
    assert np.all(trans >= 0.0)
    assert np.all(trans <= 1.0 + 1e-9)
    assert np.all(np.diff(trans) > 0), "transmission should increase with energy"


def test_noumerov_converges_under_grid_refinement(barrier):
    """Halving h must change the answer by much less than the answer itself."""
    energies = np.arange(4.0, 8.0, 0.5)
    report = Noumerov(h=1e-3).calculate_convergence_report(barrier, energies, factors=(1.0, 2.0))
    assert report[1e-3] < 1e-3, f"coarse grid deviates by {report[1e-3]:.2e}"


def test_grid_is_reproducible_across_geometries():
    """
    Point count must follow from h exactly, with no arange off-by-one, and the
    surface must sit halfway between two nodes whatever the geometry.

    The half-cell offset is what keeps a barrier with a jump at x = 0 accurate;
    see test_triangular_barrier_current_is_within_one_percent_of_the_exact_solution.
    """
    for h in (1e-3, 2e-3, 5e-4, 1e-4):
        solver = Noumerov(x_metal=-1.0, x_vac_plus=10.0, h=h, max_barrier_width=3.0)
        x = solver._grid()
        assert x.size == int(round((x[0] - x[-1]) / h)) + 1
        np.testing.assert_allclose(np.diff(x), -h, rtol=1e-9)
        assert 13.0 - h <= x[0] < 13.0 + h and -1.0 - h < x[-1] <= -1.0
        # The two nodes straddling the surface are half a step away from it.
        assert abs(np.abs(x).min() - h / 2) < 1e-9 * h


# --------------------------------------------------------------------------
# WKB solver
# --------------------------------------------------------------------------

def test_wkb_gamow_matches_adaptive_quadrature():
    """Gauss-Chebyshev nodes must agree with scipy.quad on the same integral."""
    solver = WKB(fermi_level=7.5, work_function=4.5, electric_field=5.0)
    energies = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    fast = solver.get_gamow_exponent(energies)
    for e, g in zip(energies, fast):
        ref = reference_gamow(e, 7.5, 4.5, 5.0)
        assert ref is not None
        assert abs(g - ref) / ref < 1e-6, f"E={e}: {g} vs {ref}"


def test_wkb_quadrature_is_converged():
    """32 nodes must already be at round-off relative to 256."""
    energies = np.array([4.0, 6.0, 8.0])
    coarse = WKB(7.5, 4.5, 5.0, n_nodes=32).get_gamow_exponent(energies)
    fine = WKB(7.5, 4.5, 5.0, n_nodes=256).get_gamow_exponent(energies)
    np.testing.assert_allclose(coarse, fine, rtol=1e-10)


def test_wkb_is_not_identically_one():
    """Regression guard for the sign errors that made the original return 1.0."""
    trans = WKB(7.5, 4.5, 5.0).calculate_transmission(None, np.array([4.0, 5.0, 6.0]))
    assert np.all(trans < 1e-3), f"deep tunnelling should be tiny, got {trans}"


def test_wkb_tracks_noumerov_within_semiclassical_error(barrier):
    """WKB should sit within an order of magnitude of the exact result."""
    energies = np.arange(4.0, 9.0, 0.5)
    exact = Noumerov().calculate_transmission(barrier, energies)
    approx = WKB(7.5, 4.5, 5.0).calculate_transmission(barrier, energies)
    ratio = approx / exact
    assert np.all(ratio > 0.3) and np.all(ratio < 3.0), f"ratios {ratio}"


def test_wkb_approaches_unity_above_the_barrier():
    """Above the barrier top the continuation must give T -> 1, not break."""
    solver = WKB(7.5, 4.5, 5.0)
    v_max = 7.5 + 4.5 - np.sqrt(const.COULOMB_CONST * 5.0)
    trans = solver.calculate_transmission(None, np.array([v_max + 2.0, v_max + 5.0]))
    assert np.all(trans > 0.95) and np.all(np.isfinite(trans))


def test_wkb_at_barrier_top_is_near_one_half():
    """Kemble's form gives T = 1/2 exactly at the barrier maximum."""
    v_max = 7.5 + 4.5 - np.sqrt(const.COULOMB_CONST * 5.0)
    trans = WKB(7.5, 4.5, 5.0).calculate_transmission(None, np.array([v_max]))
    assert abs(trans[0] - 0.5) < 0.05


# --------------------------------------------------------------------------
# Emitters
# --------------------------------------------------------------------------

def test_current_density_is_positive_and_finite(emitter):
    j = emitter.calculate_current_density()
    assert np.isfinite(j) and j > 0


def test_current_density_rises_steeply_with_field(emitter):
    """Fowler-Nordheim: log J vs 1/F is close to a straight line with negative slope."""
    fields = np.arange(4.0, 8.0, 0.5)
    j = np.array([_j_at(emitter, f) for f in fields])
    assert np.all(np.diff(j) > 0)
    slope = np.polyfit(1.0 / fields, np.log(j / fields ** 2), 1)[0]
    assert slope < 0, "Fowler-Nordheim slope must be negative"


def _j_at(emitter, field):
    emitter.update_params(field=field)
    return emitter.calculate_current_density()


def test_base_data_is_cached_not_recomputed(emitter, monkeypatch):
    """Nottingham heat must not re-run the tunnelling calculation."""
    calls = {"n": 0}
    original = emitter.solver.calculate_transmission_batch

    def counting(potentials, grids):
        calls["n"] += 1
        return original(potentials, grids)

    monkeypatch.setattr(emitter.solver, "calculate_transmission_batch", counting)
    emitter.calculate_nottingham_heat()
    assert calls["n"] == 1, f"expected a single solve, got {calls['n']}"


def test_cache_invalidates_on_direct_mutation(emitter):
    """Editing a sub-object directly must not leave a stale cached result."""
    first = emitter.calculate_current_density()
    emitter.potential.electric_field = 6.5
    emitter.solver.electric_field = 6.5
    second = emitter.calculate_current_density()
    assert second > first * 2, "cache did not notice the field change"


def test_cache_follows_a_custom_grid_changed_directly():
    """
    A user-supplied grid is part of the fingerprint too. It used to be left
    out, so replacing a CustomMetal's array, or editing it in place, returned
    the current of the old grid.
    """
    from getelec.band_structure import CustomMetal, CustomSemiconductor
    from getelec.electron_emitter import MetalEmitter, SemiconductorEmitter

    def metal(grid):
        return MetalEmitter(SchottkyPotential(7.5, 4.5, 5.0), Noumerov(),
                            LogFermiDirac(7.5, 300.0), CustomMetal(grid))

    def semiconductor(valence):
        band = CustomSemiconductor(np.linspace(13.62, 14.5, 89), valence,
                                   top_valence=12.5, band_gap=1.12)
        return SemiconductorEmitter(SchottkyPotential(13.0, 4.5, 5.0), Noumerov(),
                                    LogFermiDirac(13.0, 300.0), band)

    emitter = metal(np.arange(5.0, 8.5, 0.01))
    emitter.calculate_current_density()
    emitter.band.custom_energy_array = np.arange(7.0, 8.5, 0.01)
    expected = metal(np.arange(7.0, 8.5, 0.01)).calculate_current_density()
    assert emitter.calculate_current_density() == pytest.approx(expected, rel=1e-12)
    emitter.band.custom_energy_array += 0.05
    expected = metal(np.arange(7.0, 8.5, 0.01) + 0.05).calculate_current_density()
    assert emitter.calculate_current_density() == pytest.approx(expected, rel=1e-12)

    emitter = semiconductor(np.linspace(10.0, 12.5, 251))
    before = emitter.calculate_current_density()
    emitter.band.custom_valence_array[:] = np.linspace(12.0, 12.5, 251)
    expected = semiconductor(np.linspace(12.0, 12.5, 251)).calculate_current_density()
    assert emitter.calculate_current_density() == pytest.approx(expected, rel=1e-12)
    assert expected < before


def test_every_component_setting_enters_the_cache_key():
    """
    The emitter caches against a fixed list of attribute names, so a setting
    left off the list is silently ignored when changed directly on its
    component. Custom grids, the Noumerov seed and energy nodes, and the
    learned solver's fallback all were, one after another. Every public
    attribute of every shipped component class must be on the list. A new
    class that needs constructor arguments has to be given them here.
    """
    import inspect

    from getelec import (band_structure, electron_supply, potential_barrier,
                         transmission_solutions, transmission_solver)
    from getelec.electron_emitter import _FINGERPRINT_FIELDS

    grid = np.arange(0.01, 20.0, 0.01)
    arguments = {
        "Customised": (lambda x: np.zeros_like(x),),
        "CustomMetal": (grid,),
        "CustomSemiconductor": (np.arange(13.7, 20.0, 0.01), np.arange(1.0, 12.5, 0.01)),
        "DensityOfStatesMetal": (grid, np.sqrt(grid), 7.5, 4.5),
    }
    bases = (potential_barrier.Barrier, band_structure.BandStructure,
             electron_supply.Supply, transmission_solver.TransmissionSolver)
    classes = {cls for module in (potential_barrier, band_structure, electron_supply,
                                  transmission_solver, transmission_solutions)
               for _, cls in inspect.getmembers(module, inspect.isclass)
               if issubclass(cls, bases) and not inspect.isabstract(cls)}
    # Guards the check itself: a discovery that found nothing would pass.
    assert len(classes) >= 18, sorted(cls.__name__ for cls in classes)
    for cls in sorted(classes, key=lambda c: c.__name__):
        component = cls(*arguments.get(cls.__name__, ()))
        missing = sorted(name for name in vars(component)
                         if not name.startswith("_") and name not in _FINGERPRINT_FIELDS)
        assert not missing, f"{cls.__name__}: {missing} are not in the cache key"


def test_emitter_recomputes_when_a_solver_setting_or_the_solver_changes():
    """
    Changing a solver setting directly -- including a setting of the solver a
    NeuralSolver falls back to -- or swapping the solver for another of the
    same class and settings must recompute, once. Counted rather than compared
    by value, since several of these changes move the answer by less than a
    test can see.
    """
    from getelec.transmission_solver import NeuralSolver

    def count_recomputations(emitter):
        calls = {"n": 0}
        original = emitter._calculate_base_data

        def counting():
            calls["n"] += 1
            return original()

        emitter._calculate_base_data = counting
        emitter.calculate_current_density()
        return calls

    def assert_recomputes_once(emitter, calls, change, label):
        before = calls["n"]
        change()
        emitter.calculate_current_density()
        emitter.calculate_current_density()
        assert calls["n"] == before + 1, f"{label}: {calls['n'] - before} recomputations"

    emitter = getelec.metal_emitter(field=5.0)
    calls = count_recomputations(emitter)
    for name, value in (("energy_nodes", 48), ("auto_domain", False),
                        ("interpolation_tolerance", 1e-3), ("seed", "plane")):
        assert_recomputes_once(emitter, calls,
                               lambda: setattr(emitter.solver, name, value), name)

    emitter = getelec.metal_emitter(field=5.0, method="ml")
    calls = count_recomputations(emitter)
    # First, while every setting still matches: only the object differs.
    assert_recomputes_once(emitter, calls,
                           lambda: setattr(emitter, "solver", NeuralSolver()),
                           "a new NeuralSolver")
    assert_recomputes_once(emitter, calls,
                           lambda: setattr(emitter.solver, "energy_nodes", 0),
                           "energy_nodes")
    assert_recomputes_once(emitter, calls,
                           lambda: setattr(emitter.solver, "fallback", "wkb"),
                           "fallback")
    assert_recomputes_once(emitter, calls,
                           lambda: setattr(emitter.solver.exact_solver, "h", 5e-4),
                           "the fallback solver's h")


def test_update_params_sets_fermi_level_everywhere(emitter):
    emitter.update_params(fermi=8.25)
    assert emitter.potential.fermi_level == 8.25
    assert emitter.supply.fermi_level == 8.25


def test_update_params_rejects_unknown_names(emitter):
    with pytest.raises(KeyError, match="workfunction"):
        emitter.update_params(workfunction=4.5)


def test_nottingham_heat_is_finite(emitter):
    assert np.isfinite(emitter.calculate_nottingham_heat())


def test_nottingham_heat_sign_flips_with_temperature():
    """
    Cold field emission heats the tip, hot thermionic emission cools it.

    Electrons leaving from below E_F are replaced by hotter ones at E_F, so
    the emitter gains energy: P_N = integral of (E - E_F) TED dE < 0.
    """
    cold = getelec.nottingham_heat(field=6.0, temperature=300.0)
    hot = getelec.nottingham_heat(field=1.0, temperature=2500.0)
    assert cold < 0 < hot


def test_reference_current_and_nottingham_values():
    """
    The verified reference numbers.

    Pins absolute values, including the unit of P_N: W/cm^2, the same area
    unit as J. Off by 1e14 means W/nm^2 has come back.

    These are the shipped defaults' own numbers. They sit within 1.3e-5 of a
    run at h = 1.25e-4 on a 20 nm domain, which is the discretisation floor at
    the default step, so the last digit here is arithmetic and not physics.
    """
    for field, j_ref, p_ref in ((3.0, 4.106103e+01, -4.7227e+00),
                                (5.0, 4.087404e+05, -8.4180e+04),
                                (7.0, 2.383137e+07, -6.9694e+06)):
        params = dict(field=field, work_function=4.5, fermi_level=7.5,
                      temperature=300.0)
        assert np.isclose(getelec.current_density(**params), j_ref, rtol=1e-5)
        assert np.isclose(getelec.nottingham_heat(**params), p_ref, rtol=1e-3)
        emitter = getelec.metal_emitter(**params)
        assert np.isclose(emitter.calculate_nottingham_heat(), p_ref, rtol=1e-3)


def test_gui_supply_is_on_the_current_density_scale():
    """
    The GUI's N(E), times its D(E), must integrate to J. N(E) was plotted a
    factor k_B = 8.6e-5 too small. (The product is a check here, not a plot.)
    """
    import gui_backend

    params = dict(fermi_level=7.5, work_function=4.5, field=5.0, temperature=300.0)
    supply = gui_backend.calculate("N(E) (supply function)", "Metal",
                                 params, -3.0, 1.5, 0.005)
    transmission = gui_backend.calculate("D(E) (transmission)", "Metal",
                                       params, -3.0, 1.5, 0.005)
    area = np.trapezoid(supply["y"] * transmission["y"], supply["x"])
    assert abs(area / getelec.current_density(**params) - 1) < 0.01


def test_energy_distributions_have_matching_shapes(emitter):
    e1, ted = emitter.calculate_total_energy_distribution()
    e2, ned = emitter.calculate_normal_energy_distribution()
    assert e1.shape == ted.shape == e2.shape == ned.shape
    assert np.all(np.isfinite(ted)) and np.all(np.isfinite(ned))


def test_semiconductor_emitter_runs():
    em = getelec.semiconductor_emitter(work_function=4.5, fermi_level=13.0,
                                       temperature=300.0, field=5.0)
    j = em.calculate_current_density()
    assert np.isfinite(j)
    e_cb, ted_cb, e_vb, ted_vb = em.calculate_total_energy_distribution()
    assert e_cb.size and e_vb.size


def test_semiconductor_batches_its_band_solves(monkeypatch):
    """
    The two band grids are solved in one batched call, not two separate ones.

    It used to solve four. The two extra were the scaled grids, and they fed a
    mass-weighted combination that only the transmission accessor read -- which
    is not what that accessor should return, so they are no longer solved on.
    See test_semiconductor_transmission_is_the_bare_probability.

    The current density additionally evaluates the window antiderivative per
    band, so the batched call is the one that has to stay batched rather than
    the only solve that happens.
    """
    em = getelec.semiconductor_emitter()
    calls = {"n": 0}
    original = em.solver.calculate_transmission_batch

    def counting(potentials, grids):
        calls["n"] += 1
        assert len(grids) == 2
        return original(potentials, grids)

    monkeypatch.setattr(em.solver, "calculate_transmission_batch", counting)
    em._base_data()
    assert calls["n"] == 1


# --------------------------------------------------------------------------
# Convenience API
# --------------------------------------------------------------------------

def test_scalar_call_returns_a_float():
    j = getelec.current_density(field=5.0)
    assert isinstance(j, float) and j > 0


def test_array_call_matches_scalar_calls():
    fields = np.array([4.0, 5.0, 6.0])
    swept = getelec.current_density(field=fields)
    one_by_one = np.array([getelec.current_density(field=float(f)) for f in fields])
    # See test_noumerov_batch_matches_individual_calls: a batched solve shares
    # one spatial grid across the sweep, so agreement is to ~1e-5, not to bits.
    np.testing.assert_allclose(swept, one_by_one, rtol=1e-4)


def test_broadcasting_across_two_parameters():
    out = getelec.current_density(field=np.array([[4.0], [6.0]]),
                                  temperature=np.array([300.0, 1000.0]))
    assert out.shape == (2, 2)
    assert np.all(out > 0)


def test_wkb_method_is_selectable():
    j_exact = getelec.current_density(field=5.0, method="noumerov")
    j_wkb = getelec.current_density(field=5.0, method="wkb")
    assert 0.2 < j_wkb / j_exact < 5.0


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        getelec.current_density(field=5.0, method="magic")


def test_sharp_tip_emits_less_than_planar_at_fixed_apex_field():
    """
    Away from the apex the tip field decays as F/gamma, so the barrier is wider
    than the planar one at the same apex field and the current is lower.
    """
    planar = getelec.current_density(field=5.0, barrier="schottky")
    tip = getelec.current_density(field=5.0, barrier="sharp_tip", radius=20.0)
    assert tip < planar


def test_sharp_tip_recovers_the_planar_limit_at_large_radius():
    """R -> infinity must reproduce the planar Schottky barrier."""
    planar = getelec.current_density(field=5.0, barrier="schottky")
    # Radii far beyond the curvature expansion's validity range on purpose: this
    # checks the R -> infinity limit of the formula, not a physical emitter, so
    # the validity warning is expected and suppressed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tip = np.array([getelec.current_density(field=5.0, barrier="sharp_tip",
                                                radius=r)
                        for r in (20.0, 200.0, 2000.0)])
    assert np.all(np.diff(tip) > 0), "current must rise monotonically towards planar"
    assert tip[-1] < planar
    assert abs(tip[-1] - planar) / planar < 0.02


# --------------------------------------------------------------------------
# Supporting components
# --------------------------------------------------------------------------

def test_fermi_dirac_is_one_half_at_the_fermi_level():
    supply = FermiDirac(fermi_level=7.5, temperature=300.0)
    assert abs(supply.get_supply(np.array([7.5]))[0] - 0.5) < 1e-12


def test_fermi_dirac_is_a_step_at_zero_temperature():
    supply = FermiDirac(fermi_level=7.5, temperature=0.0)
    got = supply.get_supply(np.array([7.0, 8.0]))
    np.testing.assert_allclose(got, [1.0, 0.0])


def test_fermi_dirac_does_not_overflow_far_from_the_fermi_level():
    supply = FermiDirac(fermi_level=7.5, temperature=10.0)
    got = supply.get_supply(np.array([-100.0, 100.0]))
    assert np.all(np.isfinite(got))


def test_log_fermi_dirac_is_finite_over_a_wide_range():
    supply = LogFermiDirac(fermi_level=7.5, temperature=300.0)
    got = supply.get_supply(np.linspace(-50, 50, 501))
    assert np.all(np.isfinite(got)) and np.all(got >= 0)


def test_schottky_barrier_peaks_at_the_expected_height():
    """Barrier maximum must be E_F + phi - sqrt(k_e F)."""
    field = 5.0
    barrier = SchottkyPotential(7.5, 4.5, field)
    x = np.linspace(0.01, 3.0, 200001)
    expected = 7.5 + 4.5 - np.sqrt(const.COULOMB_CONST * field)
    assert abs(barrier.get_potential(x).max() - expected) < 1e-3


def test_barrier_accepts_integer_grids():
    """get_potential must not silently truncate when handed an int array."""
    got = SchottkyPotential(7.5, 4.5, 5.0).get_potential(np.array([1, 2, 3]))
    assert got.dtype.kind == "f"


def test_smart_metal_grid_brackets_the_fermi_level():
    grid = SmartMetal().generate_band_structure(fermi_level=7.5, work_function=4.5,
                                                electric_field=5.0, temperature=300.0)
    assert grid.size > 10 and grid[0] < 7.5 < grid[-1]


def test_metal_grid_respects_its_limits():
    grid = Metal(lower_energy_limit=1.0, upper_energy_limit=3.0,
                 energy_resolution=0.5).generate_band_structure()
    np.testing.assert_allclose(grid, [1.0, 1.5, 2.0, 2.5, 3.0])


def test_too_few_energy_points_raises_a_clear_error():
    from getelec.electron_emitter import _cumulative_transmission
    with pytest.raises(ValueError, match="at least 2 energy points"):
        _cumulative_transmission(np.array([1.0]), np.array([1e-10]))


# --------------------------------------------------------------------------
# Boundary condition and reduced-cost modes
# --------------------------------------------------------------------------

def _converged_reference(barrier, energies):
    """Large domain, fine grid: the answer both seeds should be converging to."""
    return Noumerov(x_metal=-1.0, x_vac_plus=60.0, h=2e-4,
                    seed="wkb").calculate_transmission(barrier, energies)


def test_wkb_seed_is_closer_to_the_converged_answer(barrier):
    """
    The WKB seed must beat the plane-wave seed on the *same* grid.

    A plane wave is only exact where the potential is flat, which it is not, so
    it needs a lot of vacuum to converge. This is the justification for changing
    the default.
    """
    energies = np.arange(2.0, 9.0, 0.25)
    reference = _converged_reference(barrier, energies)
    mask = reference > 0

    def deviation(seed):
        got = Noumerov(x_vac_plus=10.0, h=1e-3, seed=seed).calculate_transmission(
            barrier, energies)
        return np.max(np.abs(got[mask] - reference[mask]) / reference[mask])

    assert deviation("wkb") < deviation("plane") / 3


def test_unknown_seed_is_rejected(barrier):
    with pytest.raises(ValueError, match="seed must be"):
        Noumerov(seed="magic").calculate_transmission(barrier, np.array([5.0]))


def test_metal_region_length_does_not_affect_the_answer(barrier):
    """
    The potential is identically zero inside the metal, so shortening that part
    of the domain must change nothing. This is what licenses x_metal=-0.01.
    """
    energies = np.arange(3.0, 9.0, 0.5)
    long_domain = Noumerov(x_metal=-1.0).calculate_transmission(barrier, energies)
    short_domain = Noumerov(x_metal=-0.01).calculate_transmission(barrier, energies)
    # Not bit-identical: dropping ~1000 steps also drops their accumulated
    # round-off. 1e-8 is still six orders below the physical truncation error.
    np.testing.assert_allclose(short_domain, long_domain, rtol=1e-8)


def test_default_vacuum_length_keeps_the_answer_of_a_much_longer_one():
    """
    The default starts the integration 3 nm past the barrier, not 10.

    The seed is a WKB wave, which is already accurate close to the barrier, so
    the extra vacuum is grid points spent on nothing. What "nothing" means is
    measured here rather than assumed: against a 20 nm run, at both a low and a
    high field and for a curved tip as well as the planar barrier.
    """
    from getelec.band_structure import SmartMetal
    from getelec.potential_barrier import SmallRadiiPotential

    long_domain = Noumerov(x_metal=-1.0, x_vac_plus=20.0)
    for potential in (SchottkyPotential(7.5, 4.5, 1.0), SchottkyPotential(7.5, 4.5, 5.0),
                      SmallRadiiPotential(7.5, 4.5, 5.0, radius=20.0, gamma=100)):
        energies = SmartMetal().generate_band_structure(
            fermi_level=7.5, work_function=4.5,
            electric_field=potential.electric_field, temperature=300.0)
        supply = LogFermiDirac(7.5, 300.0).get_supply(energies)
        reference = long_domain.calculate_transmission(potential, energies)
        default = Noumerov().calculate_transmission(potential, energies)

        current = (supply * default).sum() / (supply * reference).sum()
        assert abs(current - 1) < 1e-4, f"current density moved by {current - 1:.2e}"
        # Pointwise, over the energies that carry the current.
        weight = supply * reference
        window = (reference > 0) & (weight >= 1e-6 * weight.max())
        pointwise = np.abs(default[window] / reference[window] - 1).max()
        assert pointwise < 1e-3, f"transmission moved by {pointwise:.2e}"


def test_fast_preset_is_accurate_where_it_matters(barrier):
    """
    Noumerov.fast() must not degrade the physical outputs.

    Current density and the energy distribution are checked against a converged
    reference; both must come out at least as good as the default settings do.
    """
    from getelec.electron_supply import LogFermiDirac
    from scipy.integrate import cumulative_trapezoid

    energies = SmartMetal().generate_band_structure(
        fermi_level=7.5, work_function=4.5, electric_field=5.0, temperature=300.0)
    supply = LogFermiDirac(7.5, 300.0).get_supply(energies)
    reference = _converged_reference(barrier, energies)

    def outputs(trans):
        current = np.sum(supply * trans)
        ted = supply * cumulative_trapezoid(trans, energies, initial=0)
        return current, ted

    j_ref, ted_ref = outputs(reference)
    j_fast, ted_fast = outputs(Noumerov.fast().calculate_transmission(barrier, energies))

    significant = ted_ref > ted_ref.max() * 1e-10
    assert abs(j_fast - j_ref) / j_ref < 1e-4
    assert np.max(np.abs(ted_fast[significant] - ted_ref[significant])
                  / ted_ref[significant]) < 1e-3


def test_fast_preset_is_substantially_cheaper(barrier):
    """
    The preset must actually do less, not just be differently configured.

    Counted as recurrence steps -- energies solved times grid points -- rather
    than timed. The wall-clock ratio at this problem size is between 1.5 and 3
    from run to run: both configurations are small enough that thread
    scheduling and Python overhead, not the recurrence, set the time, and a
    threshold on it was this suite's one persistently flaky test. The step
    count is what the preset actually controls, and it is deterministic. The
    timing itself is still checked, but only for its sign.
    """
    import time

    energies = SmartMetal().generate_band_structure(
        fermi_level=7.5, work_function=4.5, electric_field=5.0, temperature=300.0)
    exact, quick = Noumerov(), Noumerov.fast()
    for solver in (exact, quick):  # warm the JIT and the caches
        solver.calculate_transmission(barrier, energies)

    def steps(solver, n_energy):
        width = solver.get_required_barrier_width([barrier], float(energies[0]))
        return n_energy * solver._grid(width).size

    nodes = quick._energy_nodes(energies.size)
    assert nodes is not None, "the preset should be interpolating on this grid"
    quick_steps = steps(quick, nodes.size + quick._n_probes)
    exact_steps = steps(exact, energies.size)
    assert quick_steps < exact_steps / 10, (
        f"{exact_steps / quick_steps:.1f}x fewer steps is not enough")

    def elapsed(solver):
        start = time.perf_counter()
        solver.calculate_transmission(barrier, energies)
        return time.perf_counter() - start

    samples = [[elapsed(solver) for solver in (quick, exact)] for _ in range(9)]
    quick_time, exact_time = (float(np.median([row[i] for row in samples])) for i in (0, 1))
    assert quick_time < exact_time, f"{exact_time / quick_time:.2f}x is not faster at all"


def test_energy_interpolation_falls_back_on_small_grids(barrier):
    """With fewer energies than nodes there is nothing to interpolate."""
    energies = np.arange(4.0, 6.0, 0.5)
    quick = Noumerov(energy_nodes=48).calculate_transmission(barrier, energies)
    exact = Noumerov().calculate_transmission(barrier, energies)
    np.testing.assert_allclose(quick, exact, rtol=1e-12)


def test_interpolated_transmission_stays_physical(barrier):
    """Spline extrapolation must not push transmission above 1 or below 0."""
    energies = SmartMetal().generate_band_structure(
        fermi_level=7.5, work_function=4.5, electric_field=7.0, temperature=300.0)
    trans = Noumerov.fast().calculate_transmission(barrier, energies)
    assert np.all(trans >= 0.0) and np.all(trans <= 1.0)


# --------------------------------------------------------------------------
# NoumerovReference: the integration written out plainly
# --------------------------------------------------------------------------

@njit
def _original_noumerov_integration(psi, T_vals):
    N = len(psi)
    for i in range(1, N - 1):
        numerator = (2 + 10 * T_vals[i]) * psi[i] - (1 - T_vals[i - 1]) * psi[i - 1]
        denominator = 1 - T_vals[i + 1]
        psi[i + 1] = numerator / denominator
    return psi


def original_noumerov_reference(potential, electron_energy=7.5, fermi_level=7.5,
                                work_function=4.5, field=3):
    """
    The reference integration as it was first written, verbatim apart from names.

    Here ``potential(x, E_F, phi, F)`` is a function rather than a barrier
    object, and the domain, the step and the plane-wave seed are fixed. With
    its defaults, calculate_noumerov_reference must reproduce it bit for bit.
    """
    F = field  # V/nm
    PHI = work_function  # eV
    E_F = fermi_level  # eV
    E = electron_energy  # eV

    x_start = -1
    x_end = 20.
    h = 1E-3
    x_points = np.arange(x_end, x_start - h, -h)
    N = len(x_points)

    V = potential(x_points, E_F, PHI, F)

    psi = np.zeros(N, dtype=np.complex128)

    f_vals = (V - E) / const.HBAR2_2M
    T_vals = (h**2 / 12) * f_vals
    k = np.sqrt(-f_vals.astype(np.complex128))

    psi[0] = np.exp(1j * k[0] * x_points[0])
    psi[1] = np.exp(1j * k[0] * x_points[1])

    psi = _original_noumerov_integration(psi, T_vals)

    x_a, psi_a = x_points[-1], psi[-1]
    x_b, psi_b = x_points[-2], psi[-2]
    k_metal = np.sqrt(E / const.HBAR2_2M)

    matrix = np.array([
        [np.exp(1j * k_metal * x_a), np.exp(-1j * k_metal * x_a)],
        [np.exp(1j * k_metal * x_b), np.exp(-1j * k_metal * x_b)]
    ])
    vector = np.array([psi_a, psi_b])
    A, B = np.linalg.solve(matrix, vector)

    k_vac = k[0]
    T_current = (np.real(k_vac) / np.real(k_metal)) * (1 / np.abs(A)**2)

    return T_current, x_points, V, psi, k_metal


def test_noumerov_reference_reproduces_the_original_integration_bit_for_bit(barrier):
    """
    The grid, the potential, the wavefunction, the wavevector and the
    transmission, all identical to the bit. The only change the package made is
    that the barrier is an object carrying its own parameters.
    """
    def potential(x, fermi_level, work_function, field):
        return SchottkyPotential(fermi_level, work_function, field).get_potential(x)

    for energy in (0.5, 4.0, 7.5, 9.0, 12.0):
        original = original_noumerov_reference(potential, energy, 7.5, 4.5, 5.0)
        adapted = calculate_noumerov_reference(barrier, energy)
        assert adapted[0] == original[0], f"transmission differs at E = {energy}"
        for got, want in zip(adapted[1:], original[1:]):
            assert np.array_equal(got, want), f"arrays differ at E = {energy}"


def test_noumerov_reference_gives_the_reference_numbers():
    """
    The pinned values of test_reference_current_and_nottingham_values, from the
    plain integration through the emitter. It shares no solver code with
    Noumerov, and the two differ by 3e-5 at most at these fields.
    """
    for field, j_ref, p_ref in ((3.0, 4.106103e+01, -4.7227e+00),
                                (5.0, 4.087404e+05, -8.4180e+04),
                                (7.0, 2.383137e+07, -6.9694e+06)):
        emitter = getelec.metal_emitter(field=field, reference=True)
        assert np.isclose(emitter.calculate_current_density(), j_ref, rtol=1e-4)
        assert np.isclose(emitter.calculate_nottingham_heat(), p_ref, rtol=1e-4)


def test_noumerov_reference_transmission_agrees_with_noumerov(barrier):
    """
    Pointwise over the default grid at 5 V/nm (measured: 4.4e-4 with the plane
    wave, 1.5e-4 with WKB). The two solvers differ in their domain, 20 nm
    against 3 nm past the barrier, and with the plane wave in the seed too.
    """
    energies = SmartMetal().generate_band_structure(
        fermi_level=7.5, work_function=4.5, electric_field=5.0, temperature=300.0)
    exact = Noumerov().calculate_transmission(barrier, energies)
    for seed, tolerance in (("plane", 1e-3), ("wkb", 5e-4)):
        got = NoumerovReference(seed=seed).calculate_transmission(barrier, energies)
        deviation = np.abs(got / exact - 1).max()
        assert deviation < tolerance, f"seed={seed}: {deviation:.2e}"


def test_noumerov_reference_needs_the_surface_between_nodes_for_a_jump():
    """
    np.arange(20, -1.001, -0.001) puts a node on the surface, where the
    triangular barrier jumps by E_F + phi: about 1% in current density. Half a
    step more of vacuum moves the surface to the middle of a cell and recovers
    the exact result, without touching the integration. x_end is set on the
    solver itself, which the emitter's cache has to notice.
    """
    exact = getelec.metal_emitter(field=5.0, barrier="triangular",
                                  method="airy").calculate_current_density()
    emitter = getelec.metal_emitter(field=5.0, barrier="triangular", reference=True)
    on_node = emitter.calculate_current_density() / exact - 1
    emitter.solver.x_end = 20.0005
    mid_cell = emitter.calculate_current_density() / exact - 1
    assert 5e-3 < on_node < 2e-2, f"node on the surface: {on_node:+.2e}"
    assert abs(mid_cell) < 5e-5, f"surface mid-cell: {mid_cell:+.2e}"


def test_noumerov_reference_returns_zero_below_the_band(barrier):
    """
    At E <= 0 there is no travelling state in the metal and so no incident
    flux. The integration itself returns nan below zero and raises at zero; the
    solver answers 0 without calling it, which the valence band of a
    semiconductor needs, since its windows reach below the bottom of the band.
    """
    transmission = NoumerovReference().calculate_transmission(
        barrier, np.array([-1.0, 0.0, 7.5]))
    assert transmission[0] == 0.0 and transmission[1] == 0.0
    assert 0.0 < transmission[2] < 1.0


def test_noumerov_reference_warns_when_the_domain_ends_inside_the_barrier():
    """
    The domain is fixed and cannot follow the barrier. Where x_end is still
    inside it the transmission comes out as 0, which must not pass silently.
    At 5 V/nm the default domain is ample and nothing is said.
    """
    low_field = SchottkyPotential(7.5, 4.5, 0.5)
    with pytest.warns(UserWarning, match="lies inside the barrier"):
        transmission = NoumerovReference().calculate_transmission(
            low_field, np.array([1.0, 7.5]))
    assert transmission[0] == 0.0 and transmission[1] > 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NoumerovReference().calculate_transmission(
            SchottkyPotential(7.5, 4.5, 5.0), np.array([1.0, 7.5]))


def test_reference_flag_selects_the_reference_solver():
    """reference=True works like fast=True, and the two cannot be combined."""
    solver = getelec.metal_emitter(reference=True, x_end=25.0, seed="wkb").solver
    assert type(solver) is NoumerovReference
    assert (solver.x_start, solver.x_end, solver.h, solver.seed) == (-1.0, 25.0, 1e-3, "wkb")
    assert NoumerovReference().seed == "plane"
    assert type(Noumerov.reference()) is NoumerovReference
    assert type(NoumerovFast.reference()) is NoumerovReference
    with pytest.raises(ValueError, match="choose one"):
        getelec.metal_emitter(fast=True, reference=True)
    with pytest.raises(ValueError, match="seed must be"):
        NoumerovReference(seed="planar")


def test_emitter_cache_follows_the_seed():
    """Changing the seed on the solver directly must recompute, not reuse."""
    emitter = getelec.metal_emitter(field=5.0)
    wkb = emitter.calculate_current_density()
    emitter.solver.seed = "plane"
    assert emitter.calculate_current_density() != wkb


# --------------------------------------------------------------------------
# Transmission and supply extraction
# --------------------------------------------------------------------------

def test_transmission_coefficient_is_physical(emitter):
    energies, transmission = emitter.calculate_transmission_coefficient()
    assert energies.shape == transmission.shape
    assert np.all(transmission >= 0) and np.all(transmission <= 1)
    assert np.all(np.diff(transmission) > 0), "D must rise with energy"


def test_supply_function_is_positive_and_falls_above_the_fermi_level(emitter):
    energies, supply = emitter.calculate_supply_function()
    assert np.all(supply >= 0) and np.all(np.isfinite(supply))
    above = energies > emitter.potential.fermi_level + 0.5
    below = energies < emitter.potential.fermi_level - 0.5
    assert supply[above].max() < supply[below].max()


def test_supply_does_not_depend_on_field_but_transmission_does():
    """Physical separation: the barrier sets D, the occupancy sets N."""
    low = getelec.metal_emitter(field=3.0)
    high = getelec.metal_emitter(field=7.0)
    e_low, n_low = low.calculate_supply_function()
    e_high, n_high = high.calculate_supply_function()
    common = np.intersect1d(np.round(e_low, 9), np.round(e_high, 9))
    assert common.size > 50
    interp_low = np.interp(common, e_low, n_low)
    interp_high = np.interp(common, e_high, n_high)
    np.testing.assert_allclose(interp_low, interp_high, rtol=1e-9)
    assert (np.interp(common, e_high, high.calculate_transmission_coefficient()[1])
            > np.interp(common, e_low, low.calculate_transmission_coefficient()[1])).all()


def test_module_level_extraction_helpers():
    energies, transmission = getelec.transmission_coefficient(field=5.0)
    assert transmission.shape == energies.shape
    grid = np.array([5.0, 6.0, 7.0])
    got_e, got_d = getelec.transmission_coefficient(field=5.0, energies=grid)
    np.testing.assert_allclose(got_e, grid)
    assert np.all(np.diff(got_d) > 0)
    _, supply = getelec.supply_function(field=5.0, temperature=300.0)
    assert np.all(supply >= 0)


def test_semiconductor_extraction_returns_both_bands():
    em = getelec.semiconductor_emitter()
    e_cb, t_cb, e_vb, t_vb = em.calculate_transmission_coefficient()
    assert e_cb.shape == t_cb.shape and e_vb.shape == t_vb.shape
    e_cb2, n_cb, e_vb2, n_vb = em.calculate_supply_function()
    np.testing.assert_allclose(e_cb, e_cb2)
    assert np.all(np.isfinite(n_cb)) and np.all(np.isfinite(n_vb))


def test_semiconductor_transmission_is_the_bare_probability():
    """
    Both bands must report D(E) itself, a probability in [0, 1].

    It used to return the effective-mass-weighted combination that the current
    integral was once written with, which is not a probability: the valence
    band came out negative (down to -0.57 at the defaults), so the GUI could
    not plot D on a log axis and said the data contained negative values. The
    masses belong in the window limits of Eq. (8), which the distributions
    integrate D between -- not in D. J, P_N, TED and NED never used this.
    """
    for masses in ((0.98, 0.59), (1.64, 0.68), (0.3, 2.5)):
        em = getelec.semiconductor_emitter(electron_eff_mass=masses[0],
                                           hole_eff_mass=masses[1])
        e_cb, t_cb, e_vb, t_vb = em.calculate_transmission_coefficient()
        for band, energies, transmission in (("conduction", e_cb, t_cb),
                                             ("valence", e_vb, t_vb)):
            assert np.all(transmission >= 0.0), f"{band} D < 0 at m* = {masses}"
            assert np.all(transmission <= 1.0), f"{band} D > 1 at m* = {masses}"
            # The emitter solves its grids through the batched spline path,
            # a standalone call solves them directly, so these agree to the
            # spline's own tolerance -- 8e-5 here -- not to round-off.
            np.testing.assert_allclose(
                transmission,
                em.solver.calculate_transmission(em.potential, energies),
                rtol=2e-3)


# --------------------------------------------------------------------------
# Wide barriers and the log-transmission path
# --------------------------------------------------------------------------

def test_log_transmission_agrees_with_the_probability(barrier):
    energies = np.arange(3.0, 9.0, 0.5)
    solver = Noumerov()
    probability = solver.calculate_transmission(barrier, energies)
    log_probability = solver.calculate_log_transmission(barrier, energies)
    mask = probability > 0
    np.testing.assert_allclose(np.log(probability[mask]), log_probability[mask],
                               rtol=1e-12)


def test_log_transmission_reaches_past_the_float_floor():
    """
    The log path must stay finite where the probability itself underflows.

    Here ln T is about -824, i.e. T ~ 1e-358, which float64 cannot represent at
    all: calculate_transmission returns a hard zero while the log path is still
    giving digits. Deeper than about ln T = -1400 the wavefunction itself
    overflows and the solver returns -inf, which is at least an honest failure
    rather than a wrong finite number.
    """
    deep = SchottkyPotential(fermi_level=20.5, work_function=4.5, electric_field=1.0)
    energies = np.array([0.5, 1.5])
    log_probability = Noumerov().calculate_log_transmission(deep, energies)
    probability = Noumerov().calculate_transmission(deep, energies)
    assert np.all(np.isfinite(log_probability))
    assert log_probability.max() < -700
    assert np.all(probability == 0.0), "probability path should underflow here"


def test_auto_domain_covers_wide_barriers():
    """
    A 1.2 V/nm barrier reaches ~20 nm, well past the 3 nm default width.

    Integrating only part of a barrier gives a meaningless answer, not a slightly
    wrong one -- this was wrong by a factor of e**126 before auto-sizing.
    """
    barrier = SchottkyPotential(fermi_level=19.5, work_function=4.5,
                                electric_field=1.2)
    energies = np.array([0.5, 2.0, 4.0])
    auto = Noumerov().calculate_log_transmission(barrier, energies)
    wide = Noumerov(x_metal=-0.01, x_vac_plus=10.0, max_barrier_width=30.0,
                    h=2e-3, auto_domain=False).calculate_log_transmission(barrier, energies)
    np.testing.assert_allclose(auto, wide, atol=0.05)
    truncated = Noumerov(auto_domain=False).calculate_log_transmission(barrier, energies)
    assert np.abs(truncated - wide).max() > 50, "the truncated domain should be badly wrong"


def test_required_barrier_width_matches_the_analytic_turning_point():
    barrier = SchottkyPotential(fermi_level=7.5, work_function=4.5, electric_field=5.0)
    width = Noumerov().get_required_barrier_width([barrier], 0.01)
    assert abs(width - (12.0 - 0.01) / 5.0) < 0.05


def test_wkb_is_finite_far_above_the_vacuum_level():
    """Above the vacuum level the discriminant turns positive with negative roots."""
    transmission = WKB(7.5, 4.5, 5.0).calculate_transmission(
        None, np.linspace(0.05, 40.0, 300))
    assert np.all(np.isfinite(transmission))
    assert np.all(np.diff(transmission) >= -1e-12)
    assert transmission[-1] > 0.999


# --------------------------------------------------------------------------
# Transmission invariance, and the shipped Schottky model
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def accurate_solver():
    return Noumerov(x_metal=-0.01, x_vac_plus=3.0, h=5e-4, seed="wkb")


def test_transmission_depends_on_fermi_and_work_function_only_via_their_sum():
    """
    The exact invariance the learned models are built on.

    V(x) = E_F + phi - Fx - k_e/4x contains only the sum, and inside the metal
    V = 0, so nothing else can enter. This is why the models take W = E_F + phi
    rather than both, and why training one model per Fermi level fitted a
    variable the answer ignores.
    """
    solver = Noumerov()
    energies = np.array([3.0, 5.0, 7.0, 9.0])
    reference = solver.calculate_transmission(SchottkyPotential(7.5, 4.5, 5.0), energies)
    for fermi, work in ((8.5, 3.5), (6.0, 6.0), (10.0, 2.0)):
        assert fermi + work == 12.0
        got = solver.calculate_transmission(SchottkyPotential(fermi, work, 5.0), energies)
        np.testing.assert_allclose(got, reference, rtol=0)


@pytest.fixture(scope="module")
def schottky_ml():
    from getelec.transmission_solver import NeuralSolver
    return NeuralSolver()


def test_schottky_ml_pointwise_accuracy(schottky_ml, accurate_solver):
    """
    The shipped Schottky model on a deterministic sweep of its domain.

    A grid rather than random draws: the error concentrates at the low-field
    edge, where a random draw can miss it. The model is a worked example of
    the workflow, not a precision tool -- users retrain for their own
    conditions -- so the bounds are on the typical and 90th-percentile error.
    """
    domain = schottky_ml.get_model(SchottkyPotential()).domain
    lo_h, hi_h = domain["barrier_height"]
    errors = []
    for W in np.linspace(*domain["total_height"], 8):
        for F in np.linspace(*domain["field"], 8):
            barrier = SchottkyPotential(W - 4.5, 4.5, F)
            energies = W - np.linspace(lo_h, hi_h, 80)
            energies = energies[energies > 0.05]
            reference = accurate_solver.calculate_log_transmission(barrier, energies)
            finite = np.isfinite(reference)
            got = schottky_ml.calculate_log_transmission(barrier, energies)[finite]
            errors.append(np.abs(np.expm1(got - reference[finite])))
    errors = np.concatenate(errors)
    assert np.median(errors) < 0.005, f"median {np.median(errors):.3%}"
    assert np.percentile(errors, 90) < 0.015, f"p90 {np.percentile(errors, 90):.3%}"


def test_schottky_ml_current_density_for_field_and_thermionic_emission():
    """Both emission regimes, through the public API, against the exact solve."""
    for field, temperature in ((5.0, 300.0), (7.0, 300.0), (10.0, 300.0),  # field
                               (3.0, 1500.0), (4.0, 1000.0),               # thermal-field
                               (1.6, 2500.0), (2.0, 2000.0)):              # thermionic
        exact = getelec.current_density(field=field, temperature=temperature,
                                        fermi_level=7.5, work_function=4.5)
        learned = getelec.current_density(field=field, temperature=temperature,
                                          fermi_level=7.5, work_function=4.5,
                                          method="ml")
        assert abs(learned - exact) / exact < 0.02, (
            f"F={field} T={temperature}: {abs(learned - exact) / exact:.3%}")


def test_schottky_ml_is_faster_than_the_fast_noumerov(schottky_ml):
    """The point of the example: exact-class accuracy at lower cost."""
    import time

    barrier = SchottkyPotential(7.5, 4.5, 5.0)
    energies = np.linspace(4.0, 9.5, 3000)
    quick = Noumerov.fast()

    def elapsed(solver, repeat=10, rounds=3):
        # Best of several rounds: a single timing is at the mercy of whatever
        # else the machine is doing.
        solver.calculate_transmission(barrier, energies)
        best = float("inf")
        for _ in range(rounds):
            start = time.perf_counter()
            for _ in range(repeat):
                solver.calculate_transmission(barrier, energies)
            best = min(best, (time.perf_counter() - start) / repeat)
        return best

    assert elapsed(schottky_ml) < elapsed(quick)


def test_schottky_ml_falls_back_outside_its_domain(schottky_ml):
    """Out-of-domain barriers must be solved, not silently extrapolated."""
    outside = SchottkyPotential(fermi_level=30.0, work_function=4.5,
                                electric_field=5.0)   # W = 34.5, far outside
    energies = np.array([28.0, 30.0])
    np.testing.assert_allclose(
        schottky_ml.calculate_log_transmission(outside, energies),
        Noumerov.fast().calculate_log_transmission(outside, energies))


def test_neural_solver_cached_spline_equals_fitting_it_every_call():
    """
    The spline map is built once per grid and reused. It must give the same
    numbers as fitting the spline on each call, in either energy direction.
    """
    from scipy.interpolate import CubicSpline
    from getelec.transmission_solver import NeuralSolver

    solver = NeuralSolver()
    barrier = SchottkyPotential(7.5, 4.5, 5.0)
    model = solver.get_model(barrier)
    for energies in (np.linspace(4.0, 11.0, 834), np.linspace(11.0, 4.0, 834)):
        nodes = solver._nodes(energies)
        coarse = energies[nodes]
        log_d = solver._evaluate(barrier, coarse, model)
        order = np.argsort(coarse)
        direct = CubicSpline(coarse[order], log_d[order])(energies)
        direct[energies < coarse[order][0]] = log_d[order][0]
        np.testing.assert_allclose(solver.calculate_log_transmission(barrier, energies),
                                   np.minimum(direct, 0.0), rtol=0, atol=1e-10)
    cached = len(NeuralSolver._SPLINE_CACHE)
    solver.calculate_log_transmission(barrier, np.linspace(4.0, 11.0, 834))
    assert len(NeuralSolver._SPLINE_CACHE) == cached, "the map must be reused"


def test_single_precision_forward_pass_is_far_below_the_model_error():
    """float32 inference adds ~1e-6 to R; the model's own error is ~1e-3."""
    from getelec.transmission_solver import get_shipped_model

    model = get_shipped_model("SmallRadiiPotential")
    x = np.random.default_rng(1).normal(size=(500, model.weights[0].shape[0]))
    a = x
    for w, b in zip(model.weights[:-1], model.biases[:-1]):
        a = np.tanh(a @ w + b)
    double = (a @ model.weights[-1] + model.biases[-1]).ravel()
    assert np.abs(model.predict(x) - double).max() < 1e-4


# --------------------------------------------------------------------------
# Numerical WKB reference (for arbitrary barrier shapes)
# --------------------------------------------------------------------------

def test_numeric_wkb_reproduces_the_analytic_one_on_the_planar_barrier(barrier):
    """The generic reference must agree with the closed-form planar solver."""
    from getelec.transmission_solutions import calculate_log_kemble_numeric

    energies = np.linspace(0.5, 14.0, 300)
    numeric = calculate_log_kemble_numeric(barrier, energies)
    analytic = -np.logaddexp(0.0, 2.0 * WKB(7.5, 4.5, 5.0).get_gamow_exponent(energies))
    assert np.abs(numeric - analytic).max() < 0.01


def test_numeric_wkb_compresses_the_curved_barrier_residual():
    """
    The point of the reference: it must follow the barrier it references.

    Using the *planar* reference for a curved barrier gives a residual spanning
    more than a thousand e-folds, which is not learnable. The numerical one
    keeps it of order unity even where ln D reaches several hundred.
    """
    from getelec.potential_barrier import SmallRadiiPotential
    from getelec.transmission_solutions import calculate_log_kemble_numeric
    from getelec.transmission_solutions import get_log_kemble_schottky as log_kemble

    solver = Noumerov(x_metal=-0.01, x_vac_plus=2.0, h=2e-3, seed="wkb")
    # From 0.2 eV: across 10-50 nm the tip barrier is close to planar, and only
    # the lowest energies reach deep tunnelling (ln D = -64 here; -56 from 1 eV).
    energies = np.linspace(0.2, 11.0, 60)
    worst_numeric, worst_planar, deepest = 0.0, 0.0, 0.0
    for radius in (10.0, 20.0, 50.0):
        for gamma in (1.0, 200.0):
            # 10 nm is below the barrier's validity range and kept on purpose: the
            # most curved barrier is the hardest case for the numeric reference.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="radius = .* is below the range")
                curved = SmallRadiiPotential(7.5, 4.5, 5.0, radius=radius, gamma=gamma)
            log_d = solver.calculate_log_transmission(curved, energies)
            finite = np.isfinite(log_d)
            deepest = min(deepest, log_d[finite].min())
            worst_numeric = max(worst_numeric, np.abs(
                log_d[finite] - calculate_log_kemble_numeric(curved, energies)[finite]).max())
            worst_planar = max(worst_planar, np.abs(
                log_d[finite] - log_kemble(12.0, 5.0, energies)[finite]).max())
    assert deepest < -60, "expected genuinely deep tunnelling in this set"
    assert worst_numeric < 2.0, f"numeric reference left a residual of {worst_numeric:.1f}"
    assert worst_planar > 10 * worst_numeric


def test_numeric_wkb_is_vectorised_over_energy(barrier):
    """One call for many energies must equal the same energies one at a time."""
    from getelec.transmission_solutions import calculate_gamow_numeric

    energies = np.linspace(2.0, 12.0, 40)
    batched = calculate_gamow_numeric(barrier, energies)
    individual = np.array([calculate_gamow_numeric(barrier, np.array([e]))[0] for e in energies])
    np.testing.assert_allclose(batched, individual, rtol=1e-10)


# --------------------------------------------------------------------------
# NeuralSolver: learned transmission, curved-tip model
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_radii_ml():
    from getelec.transmission_solver import NeuralSolver
    return NeuralSolver()


def _curved_domain():
    from getelec.transmission_solver import get_shipped_model
    return get_shipped_model("SmallRadiiPotential").domain


def _curved(W, F, radius, gamma):
    from getelec.potential_barrier import SmallRadiiPotential
    # A few tests deliberately use a radius outside the barrier's validity range,
    # to check the model hands it to the exact solver; the warning is expected there.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="radius = ", category=UserWarning)
        return SmallRadiiPotential(W - 4.5, 4.5, F, radius=radius, gamma=gamma)


def test_shipped_sharp_tip_model_covers_exactly_the_valid_radii():
    """The network and the barrier's validity range must not drift apart again:
    the model shipped until 2026-09 was trained on 5-10 nm, below the range."""
    from getelec.potential_barrier import RADIUS_VALIDITY
    from getelec.transmission_solver import get_shipped_model
    from getelec.training import SMALL_RADII_DOMAIN

    model = get_shipped_model("SmallRadiiPotential")
    assert model.domain["radius"] == RADIUS_VALIDITY
    assert model.domain == {k: tuple(v) for k, v in SMALL_RADII_DOMAIN.items()}


def test_small_radii_potential_warns_outside_its_validity_range():
    """The barrier is valid for 20-1000 nm, and the default radius is inside it.

    Below 20 nm the current density depends on the tip's shape beyond its apex,
    which radius and gamma do not describe; above 1000 nm the tip is planar for
    the current. Each warning says which.
    """
    from getelec.potential_barrier import SmallRadiiPotential, RADIUS_VALIDITY

    assert RADIUS_VALIDITY == (20.0, 1000.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for radius in (20.0, 50.0, 1000.0):
            SmallRadiiPotential(7.5, 4.5, 5.0, radius=radius)
        SmallRadiiPotential(7.5, 4.5, 5.0)                    # the default radius
        getelec.metal_emitter(barrier="sharp_tip")            # the API default too
        assert not caught
    for radius in (5.0, 10.0, 19.9):
        with pytest.warns(UserWarning, match="shape of the emitter beyond its apex"):
            SmallRadiiPotential(7.5, 4.5, 5.0, radius=radius)
    with pytest.warns(UserWarning, match="effectively planar"):
        SmallRadiiPotential(7.5, 4.5, 5.0, radius=2000.0)


def test_sharp_tip_is_planar_for_the_current_at_the_top_of_its_range():
    """The upper bound's reason: at 1000 nm the curvature moves J by ~1% at 5 V/nm."""
    planar = getelec.current_density(field=5.0)
    tip = getelec.current_density(field=5.0, barrier="sharp_tip", radius=1000.0)
    assert 0.0 < 1.0 - tip / planar < 0.015


def test_small_radii_ml_current_density_within_three_percent(small_radii_ml, accurate_solver):
    """
    What the model is for: current density on curved tips.

    Compared against the semiclassical reference alone, which is what you would
    otherwise use at this cost.
    """
    from getelec.band_structure import SmartMetal
    from getelec.electron_supply import LogFermiDirac
    from getelec.electron_emitter import _SUPPLY_TO_A_PER_CM2
    from getelec.transmission_solutions import calculate_log_kemble_numeric

    def current(log_d, supply, energies):
        return (np.trapezoid(supply * np.exp(np.clip(log_d, -700, 0)), energies)
                * _SUPPLY_TO_A_PER_CM2)

    worst_ml, worst_wkb = 0.0, 0.0
    for W, F, radius, gamma, T in [(12, 5, 20, 100, 300), (12, 7, 50, 20, 300),
                                   (16, 4, 30, 150, 300), (12, 3, 100, 100, 1500),
                                   (12, 2, 400, 10, 2000), (20, 6, 1000, 80, 300)]:
        barrier = _curved(W, F, radius, gamma)
        energies = SmartMetal().generate_band_structure(
            fermi_level=W - 4.5, work_function=4.5,
            electric_field=F, temperature=T)
        supply = LogFermiDirac(W - 4.5, T).get_supply(energies)

        exact = current(accurate_solver.calculate_log_transmission(barrier, energies),
                        supply, energies)
        learned = current(small_radii_ml.calculate_log_transmission(barrier, energies),
                          supply, energies)
        semiclassical = current(calculate_log_kemble_numeric(barrier, energies), supply, energies)

        worst_ml = max(worst_ml, abs(learned - exact) / exact)
        worst_wkb = max(worst_wkb, abs(semiclassical - exact) / exact)

    assert worst_ml < 0.03, f"ML current density off by {worst_ml:.2%}"
    assert worst_ml < worst_wkb / 5, "the model should clearly beat its own reference"


def test_small_radii_ml_pointwise_accuracy_in_the_trained_band(small_radii_ml, accurate_solver):
    """Median accuracy inside the domain the model was trained on."""
    domain = _curved_domain()
    lo, hi = domain["barrier_height"]
    rng = np.random.default_rng(21)
    errors = []
    for _ in range(20):
        W = rng.uniform(*domain["total_height"])
        F = rng.uniform(*domain["field"])
        radius = np.exp(rng.uniform(*np.log(domain["radius"])))     # trained log-uniform
        gamma = np.exp(rng.uniform(*np.log(domain["gamma"])))
        barrier = _curved(W, F, radius, gamma)
        energies = W - np.linspace(lo, hi, 120)
        energies = energies[energies > 0.05]
        reference = accurate_solver.calculate_log_transmission(barrier, energies)
        finite = np.isfinite(reference)
        got = small_radii_ml.calculate_log_transmission(barrier, energies)[finite]
        errors.append(np.abs(np.expm1(got - reference[finite])))
    errors = np.concatenate(errors)
    assert np.median(errors) < 0.01
    assert np.percentile(errors, 90) < 0.03


def test_small_radii_ml_falls_back_outside_its_training_domain(small_radii_ml):
    """A radius the model never saw must not be silently extrapolated."""
    far = _curved(12.0, 5.0, radius=10.0, gamma=100.0)
    energies = np.linspace(4.0, 10.0, 40)
    got = small_radii_ml.calculate_log_transmission(far, energies)
    exact = Noumerov.fast().calculate_log_transmission(far, energies)
    assert np.abs(got - exact).max() < 0.1


def test_small_radii_ml_can_be_told_to_raise_instead():
    from getelec.transmission_solver import NeuralSolver
    strict = NeuralSolver(fallback="error")
    far = _curved(12.0, 5.0, radius=10.0, gamma=100.0)
    with pytest.raises(ValueError, match="outside the trained domain"):
        strict.calculate_transmission(far, np.array([6.0]))


def test_neural_models_load_without_pickle_or_scikit_learn():
    """
    The weights ship as plain arrays, so using a model needs only NumPy.

    Run in a fresh interpreter with scikit-learn and joblib made unimportable:
    the shipped model used to be a pickled scikit-learn estimator, which a
    plain `pip install getelec` could not load.
    """
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['sklearn'] = None; sys.modules['joblib'] = None\n"
        "import numpy as np\n"
        "from getelec.potential_barrier import SmallRadiiPotential, SchottkyPotential\n"
        "from getelec.transmission_solver import NeuralSolver\n"
        "solver = NeuralSolver(fallback='error')\n"
        "for b in (SmallRadiiPotential(7.5, 4.5, 5.0, radius=50.0, gamma=100.0),\n"
        "          SchottkyPotential(7.5, 4.5, 5.0)):\n"
        "    d = solver.calculate_transmission(b, np.linspace(5.5, 11.0, 50))\n"
        "    assert np.all((d > 0) & (d <= 1))\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_small_radii_ml_energy_subsampling_is_accurate_and_flat_in_cost(small_radii_ml):
    """
    Subsampling in energy must not change the answer, and must remove the
    linear cost growth that made the network slower than the tuned solver.
    """
    import time
    from getelec.transmission_solver import NeuralSolver

    barrier = _curved(12.0, 5.0, radius=50.0, gamma=100.0)
    every_point = NeuralSolver(energy_nodes=0)

    small = np.linspace(5.0, 12.0, 300)
    large = np.linspace(5.0, 12.0, 3000)

    deviation = np.abs(np.expm1(small_radii_ml.calculate_log_transmission(barrier, large)
                                - every_point.calculate_log_transmission(barrier, large)))
    assert deviation.max() < 0.01

    def elapsed(solver, grid, repeat=8, rounds=3):
        # Best of several rounds: a single timing is at the mercy of whatever
        # else the machine is doing, and this assertion used to fail only when
        # run alongside the rest of the suite.
        solver.calculate_transmission(barrier, grid)
        best = float("inf")
        for _ in range(rounds):
            start = time.perf_counter()
            for _ in range(repeat):
                solver.calculate_transmission(barrier, grid)
            best = min(best, (time.perf_counter() - start) / repeat)
        return best

    growth = elapsed(small_radii_ml, large) / elapsed(small_radii_ml, small)
    assert growth < 4.0, f"cost grew {growth:.1f}x from 300 to 3000 energies"
    assert elapsed(small_radii_ml, large) < elapsed(every_point, large) / 2


def test_neural_solver_matches_the_barrier_class_exactly():
    """
    A subclass of a barrier may change the potential, so it must not
    silently get its parent's model.
    """
    from getelec.transmission_solver import NeuralSolver

    class ModifiedSchottky(SchottkyPotential):
        pass

    barrier = ModifiedSchottky(7.5, 4.5, 5.0)
    assert NeuralSolver().get_model(barrier) is None
    with pytest.raises(ValueError, match="No trained model for ModifiedSchottky"):
        NeuralSolver(fallback="error").calculate_transmission(barrier, np.array([6.0]))

    energies = np.linspace(4.0, 9.0, 30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = NeuralSolver().calculate_log_transmission(barrier, energies)
    np.testing.assert_allclose(
        got, Noumerov.fast().calculate_log_transmission(barrier, energies))


def test_neural_model_round_trips_through_a_pickle_free_file(tmp_path):
    from getelec.transmission_solver import NeuralModel, get_shipped_model

    shipped = get_shipped_model("SmallRadiiPotential")
    path = tmp_path / "model.npz"
    shipped.save(path)
    with np.load(path, allow_pickle=False) as data:
        assert "weights_0" in data.files
    back = NeuralModel.load(path)
    assert ((back.barrier, back.features, back.reference, back.parameters)
            == (shipped.barrier, shipped.features, shipped.reference, shipped.parameters))
    assert back.domain == shipped.domain
    x = np.random.default_rng(0).normal(size=(5, shipped.weights[0].shape[0]))
    np.testing.assert_array_equal(back.predict(x), shipped.predict(x))


def test_model_with_unregistered_features_says_how_to_fix_it():
    from getelec.transmission_solver import NeuralModel, NeuralSolver, get_shipped_model

    shipped = get_shipped_model("SmallRadiiPotential")
    orphan = NeuralModel(shipped.weights, shipped.biases, features="not_registered",
                         reference=shipped.reference, barrier=shipped.barrier,
                         domain=shipped.domain, parameters=shipped.parameters)
    barrier = _curved(12.0, 5.0, radius=50.0, gamma=100.0)
    with pytest.raises(KeyError, match="register_features"):
        NeuralSolver(model=orphan).calculate_transmission(barrier, np.linspace(5, 11, 20))


def test_training_produces_a_model_the_solver_can_use(tmp_path):
    """End to end on a toy budget: generate, fit, package, save, load, predict."""
    pytest.importorskip("sklearn")
    from getelec import training
    from getelec.transmission_solver import NeuralSolver

    samples = training.generate_samples(
        training.make_schottky, training.SCHOTTKY_DOMAIN, n_barriers=20,
        heights_per_barrier=16, reference="kemble_schottky", progress=False)
    assert samples["barrier"] == "SchottkyPotential"
    # The point of the reference: the target stays of order unity even though
    # ln D itself spans hundreds of e-folds across this domain.
    assert np.abs(samples["residual"]).max() < 2.0

    estimator = training.fit(samples, "schottky", hidden_layers=(8,), max_iter=20)
    training.to_model(estimator, samples, "schottky").save(tmp_path / "toy.npz")
    solver = NeuralSolver(model=tmp_path / "toy.npz", fallback="error")
    transmission = solver.calculate_transmission(SchottkyPotential(7.5, 4.5, 5.0),
                                                 np.linspace(6.0, 11.0, 20))
    assert np.all(np.isfinite(transmission))
    assert np.all((transmission >= 0) & (transmission <= 1))


# --------------------------------------------------------------------------
# Analytical solutions
# --------------------------------------------------------------------------

def test_triangular_potential_is_the_field_tilted_vacuum_level():
    """V = E_F + phi - F x outside, zero inside, nothing clamped past the turning point."""
    from getelec.potential_barrier import TriangularPotential

    x = np.array([-1.0, 0.0, 1e-6, 1.0, 2.4, 3.0])
    expected = np.where(x > 0, 12.0 - 5.0 * x, 0.0)
    assert np.array_equal(TriangularPotential(7.5, 4.5, 5.0).get_potential(x), expected)
    # E_F and phi enter only through their sum.
    assert np.array_equal(TriangularPotential(8.5, 3.5, 5.0).get_potential(x), expected)
    # An integer grid must not truncate the potential.
    assert TriangularPotential(7.5, 4.5, 0.5).get_potential(np.array([1, 3])).tolist() == [11.5, 10.5]


def test_airy_solution_matches_noumerov_on_a_triangular_barrier():
    """
    The one case with a closed form, used to pin the numerical solver.

    Every other check in this suite compares a numerical result against another
    numerical result. The triangular barrier has an exact Airy solution, so this
    is the only test that validates the Noumerov integration against analysis
    rather than against itself -- and it holds across twenty orders of magnitude
    in transmission, and above the barrier top, where the Airy form used to
    lack its Wronskian term and came out 47% too high.
    """
    from getelec.potential_barrier import TriangularPotential
    from getelec.transmission_solutions import AiryTriangular

    barrier = TriangularPotential(9.5, 4.5, 5.0)
    energies = np.linspace(3.0, 20.0, 35)
    numerical = Noumerov(x_metal=-0.01, x_vac_plus=4.0, h=2e-4, seed="wkb",
                         auto_domain=False,
                         max_barrier_width=4.0).calculate_transmission(barrier, energies)
    analytic = AiryTriangular(9.5, 4.5, 5.0).calculate_transmission(barrier, energies)

    usable = numerical > 1e-13
    assert usable.sum() >= 25
    assert numerical[usable].max() / numerical[usable].min() > 1e10, (
        "expected a wide dynamic range here")
    assert (energies[usable] > 14.0).sum() >= 10, "expected energies above the barrier top"
    ratio = analytic[usable] / numerical[usable]
    # 1e-3 rather than the 1% this needed before the grid was aligned to the
    # surface: the jump at x = 0 is now resolved to second order.
    assert np.abs(ratio - 1.0).max() < 1e-3, f"ratios spanned {ratio.min()}..{ratio.max()}"


def test_airy_transmission_is_bounded_and_monotonic():
    from getelec.transmission_solutions import AiryTriangular

    solver = AiryTriangular(9.5, 4.5, 5.0)
    energies = np.linspace(1.0, 20.0, 200)
    transmission = solver.calculate_transmission(None, energies)
    assert np.all(transmission >= 0.0) and np.all(transmission <= 1.0)
    assert np.all(np.diff(transmission) > 0)
    # The surface is a step of E_F + phi = 14 eV, which still reflects 8% at
    # 20 eV: T stays below 1, as it must.
    assert transmission[0] < 1e-20 and 0.9 < transmission[-1] < 0.95


def test_airy_transmission_tends_to_the_step_barrier_far_above_the_top():
    """
    Far above the barrier the field hardly bends the wave over a wavelength,
    and what remains is reflection at the step: 4 k q / (k + q)^2.
    """
    from getelec.transmission_solutions import AiryTriangular

    energies = np.array([25.0, 40.0, 80.0])
    transmission = AiryTriangular(7.5, 4.5, 5.0).calculate_transmission(None, energies)
    k = np.sqrt(energies / const.HBAR2_2M)
    q = np.sqrt((energies - 12.0) / const.HBAR2_2M)
    np.testing.assert_allclose(transmission, 4 * k * q / (k + q) ** 2, rtol=1e-3)


def test_triangular_barrier_current_matches_the_exact_solution():
    """
    The triangular barrier jumps by E_F + phi at the surface, which Noumerov
    resolves only to first order in h -- by an amount set by where the surface
    falls between two grid points. The solver's grid puts the surface halfway
    between two nodes, which is the second-order placement: J lands within
    1e-5 of the exact Airy result instead of the 1.2% a node on the surface
    costs. It emits far less than the Schottky barrier, which the image charge
    lowers.
    """
    fields = np.array([2.0, 7.0, 10.0])
    exact = getelec.current_density(field=fields, barrier="triangular", method="airy")
    default = getelec.current_density(field=fields, barrier="triangular")
    quick = getelec.current_density(field=fields, barrier="triangular", fast=True)
    fine = getelec.current_density(field=fields, barrier="triangular", fast=True, h=1e-4)
    assert np.abs(default / exact - 1).max() < 5e-5
    assert np.abs(quick / exact - 1).max() < 1e-4
    assert np.abs(fine / exact - 1).max() < 1e-5
    assert np.all(default < 0.05 * getelec.current_density(field=fields))


def test_solutions_module_holds_the_closed_form_results():
    """Structure check: analytic solutions live apart from numerical solvers."""
    from getelec import transmission_solutions, transmission_solver

    for name in ("WKB", "AiryTriangular", "calculate_gamow_numeric", "calculate_log_kemble_numeric"):
        assert hasattr(transmission_solutions, name)
    for name in ("Noumerov", "NoumerovReference", "calculate_noumerov_reference",
                 "NeuralSolver", "NeuralModel"):
        assert hasattr(transmission_solver, name)


# --------------------------------------------------------------------------
# GUI backend (no Qt required)
# --------------------------------------------------------------------------

def test_every_gui_calculation_runs_for_both_materials():
    """
    The GUI offers a grid of calculation x material. All of it must work.

    The physics lives in gui_backend precisely so it can be checked here without
    a display; a broken combination would otherwise only show up when someone
    clicked it.
    """
    import gui_backend

    defaults = {name: value for name, (_, value) in gui_backend.PARAMETER_INFO.items()}
    for calculation, (swept, _, _, energy_axis) in gui_backend.CALCULATIONS.items():
        for material in gui_backend.MATERIALS:
            if calculation not in gui_backend.get_calculations(material):
                continue
            params = {name: defaults[name]
                      for name in gui_backend.get_parameters(calculation, material)}
            if material == "Semiconductor":
                params["fermi_level"] = 13.0
                params["top_valence"] = 12.5
            if energy_axis:
                bounds = (-2.0, 1.0, 0.1)
            elif swept == "field":
                bounds = (3.0, 6.0, 1.0)
            else:
                bounds = (300.0, 1200.0, 300.0)

            result = gui_backend.calculate(calculation, material, params, *bounds)
            if result["series"] is not None:
                assert len(result["series"]) == 2, "expected both bands"
                for name, x, y in result["series"]:
                    assert x.size == y.size and x.size > 10
                    assert np.all(np.isfinite(y))
            else:
                assert result["x"].size == result["y"].size > 1
                assert np.all(np.isfinite(result["y"]))


def test_gui_offers_the_window_transmission_only_for_semiconductors():
    """
    g(E) is D integrated over the effective-mass window; a metal has no window.

    It is absent from the metal menu rather than greyed out, and the backend
    refuses it outright, so a stale selection cannot reach the physics.
    """
    import gui_backend

    label = "g(E) (D over the mass window)"
    assert label in gui_backend.get_calculations("Semiconductor")
    assert label not in gui_backend.get_calculations("Metal")
    assert set(gui_backend.get_calculations("Semiconductor")) == set(gui_backend.CALCULATIONS)

    params = {name: gui_backend.PARAMETER_INFO[name][1]
              for name in gui_backend.get_parameters(label, "Metal")}
    with pytest.raises(ValueError, match="semiconductor calculation"):
        gui_backend.calculate(label, "Metal", params, -2.0, 1.0, 0.1)


def test_gui_backend_reports_bad_ranges_clearly():
    import gui_backend

    params = {name: gui_backend.PARAMETER_INFO[name][1]
              for name in gui_backend.get_parameters("I-F  (current vs field)", "Metal")}
    for bounds, message in (((3.0, 8.0, 0.0), "Step"),
                            ((8.0, 3.0, 0.5), "greater")):
        with pytest.raises(ValueError, match=message):
            gui_backend.calculate("I-F  (current vs field)", "Metal", params, *bounds)


def test_gui_backend_semiconductor_honours_the_energy_range():
    """
    The range boxes did nothing for a semiconductor.

    Whatever was typed, the curves came back on the full band grids: asking for
    -2..1 eV plotted -13..16 eV, and editing the boxes changed nothing. Unlike
    a metal, the grids are fixed by the band structure, so the range selects
    which part of the result to return rather than where it is evaluated.
    """
    import gui_backend

    params = dict(fermi_level=13.5, work_function=4.5, field=5.0,
                  temperature=300.0, band_gap=0.7, top_valence=12.5,
                  electron_eff_mass=0.98, hole_eff_mass=0.59)
    for calculation in ("TED  (total energy dist.)", "NED  (normal energy dist.)",
                        "D(E) (transmission)", "N(E) (supply function)"):
        result = gui_backend.calculate(calculation, "Semiconductor", params,
                                     -2.0, 1.0, 0.01)
        points = 0
        for name, x, y in result["series"]:
            assert x.size == y.size, f"{calculation}: {name} ragged"
            points += x.size
            if x.size:
                assert x.min() >= -2.0 and x.max() <= 1.0, f"{calculation}: {name}"
        assert points, f"{calculation}: both bands empty over -2..1 eV"


def test_gui_backend_parameter_sets_exclude_the_swept_quantity():
    """The swept quantity comes from the range boxes, not a parameter field."""
    import gui_backend

    assert "field" not in gui_backend.get_parameters("I-F  (current vs field)", "Metal")
    assert "field" in gui_backend.get_parameters("TED  (total energy dist.)", "Metal")
    metal = gui_backend.get_parameters("TED  (total energy dist.)", "Metal")
    semiconductor = gui_backend.get_parameters("TED  (total energy dist.)", "Semiconductor")
    assert set(metal) < set(semiconductor)
    assert "band_gap" in semiconductor


def test_compile_script_is_importable_and_finds_its_data():
    """The build script must be usable as a module, not only as a command."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("compile_script", str(ROOT / "compile.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for source, _ in module.DATA_FILES:
        assert (ROOT / source).exists(), f"compile.py bundles {source}, which is missing"
    assert any(source == "getelec/data" for source, _ in module.DATA_FILES)
    assert "numba" in module.HIDDEN_IMPORTS

    # The application is built outside dist/, which holds what is uploaded to
    # PyPI, and nothing it writes can be committed by `git add -A`.
    assert module.OUTPUT != ROOT / "dist"
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert f"{module.OUTPUT.name}/" in ignored and "*.spec" in ignored

    # The web engine and what only it uses are always left out, never a module
    # the rest of the application imports.
    assert "PyQt6.QtWebEngineWidgets" in module.WEB_ENGINE_MODULES
    assert not {"PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets"} & set(module.WEB_ENGINE_MODULES)


# --------------------------------------------------------------------------
# Branding, documentation and packaging
# --------------------------------------------------------------------------

def test_citation_is_the_same_everywhere():
    """
    CITATION.cff agrees with the package, and every citation text has the DOI.

    The software DOI is what a paper cites, and the same text is given in each
    place a reader may look: README (also the PyPI page), GUIDE, the notebook,
    the GUI's documentation tab and introduction page, the front page of the
    website, and the package docstring. Zenodo and GitHub read CITATION.cff, so
    its version, licence and authors have to be the package's.
    """
    import re

    yaml = pytest.importorskip("yaml")
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert citation["version"] == getelec.__version__
    assert citation["license"] == "MIT" and 'license = "MIT"' in pyproject
    declared = re.findall(r'\{ name = "([^"]+)" \}', pyproject.split("authors = [", 1)[1].split("]", 1)[0])
    cited = [f"{a['given-names']} {a['family-names']}" for a in citation["authors"]]
    assert cited == declared

    doi = citation["doi"]
    for path in ("README.md", "GUIDE.md", "gui.py", "docs/introduction.html",
                 "docs/index.html", "docs/build_intro_notebook.py", "getelec/__init__.py"):
        assert doi in (ROOT / path).read_text(encoding="utf-8"), f"{path} lacks {doi}"
    for reference in citation["references"]:
        for path in ("README.md", "docs/index.html"):
            assert reference["doi"] in (ROOT / path).read_text(encoding="utf-8"), \
                f"{path} lacks {reference['doi']}"


def test_readme_links_work_on_pypi():
    """
    README.md is also the PyPI page, where a relative link leads nowhere.

    Every link is absolute, and every one into this repository names a file
    that exists, so the page works on PyPI as it does on GitHub.
    """
    import re

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    assert targets
    relative = [target for target in targets if not re.match(r"https?://|mailto:", target)]
    assert not relative, f"relative links in README.md: {relative}"
    repository = "https://github.com/sbcarceles13/GETELEC/blob/main/"
    for target in targets:
        if target.startswith(repository):
            path = target[len(repository):].split("#")[0]
            assert (ROOT / path).is_file(), f"README.md links to {path}, which is missing"


def test_watermark_draws_on_axes_and_on_figures():
    """One helper, so every plot in the project is marked identically."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    on_axes = getelec.watermark(axes)
    assert on_axes.get_text() == getelec.WATERMARK_TEXT
    assert on_axes.get_ha() == "right" and on_axes.get_va() == "bottom"

    # A Figure target marks the whole figure once, rather than every panel.
    multi, _ = plt.subplots(1, 3)
    on_figure = getelec.watermark(multi)
    assert on_figure.get_text() == getelec.WATERMARK_TEXT
    assert on_figure.figure is multi

    custom = getelec.watermark(axes, text="Powered by GETELEC")
    assert custom.get_text() == "Powered by GETELEC"
    plt.close("all")


def test_every_example_watermarks_its_plots():
    """A figure that leaves the project should carry the attribution."""

    for script in sorted((ROOT / "examples").glob("*.py")):
        source = script.read_text(encoding="utf-8")
        if "plt.show()" not in source:
            continue
        assert "getelec.watermark" in source, f"{script.name} plots without a watermark"


def test_documentation_files_are_present():
    """
    The GUI's documentation tab loads these from disk.

    They were once lost from a working copy, and the failure showed up only as a
    blank browser pane inside the application -- so it is worth a test.
    """

    docs = ROOT / "docs"
    assert docs.is_dir(), "docs/ is missing; the GUI documentation tab needs it"
    for page in ("introduction.html", "usage.html", "getelec.html"):
        assert (docs / page).is_file(), f"docs/{page} is missing"


def test_documentation_is_ready_to_serve_as_a_website():
    """
    docs/ is published as it is, so what a search engine needs is kept here.

    Every page has a title and a description, which is what a search result
    shows, and sitemap.xml lists every page by its address. The API pages are
    overwritten by pdoc, which writes no description: docs/regenerate.py puts
    it back, and this fails if they were regenerated some other way.
    """
    import re

    docs = ROOT / "docs"
    site = "https://sbcarceles13.github.io/GETELEC/"
    assert (docs / ".nojekyll").is_file(), "without it GitHub Pages runs Jekyll over docs/"
    listed = re.findall(r"<loc>([^<]+)</loc>", (docs / "sitemap.xml").read_text(encoding="utf-8"))
    pages = sorted(path.relative_to(docs).as_posix() for path in docs.rglob("*.html"))
    assert sorted(listed) == sorted(site + page.replace("index.html", "") for page in pages)
    for page in pages:
        text = (docs / page).read_text(encoding="utf-8")
        head = text[:text.index("</head>")]
        assert re.search(r"<title>[^<]+</title>", head), f"docs/{page} has no title"
        assert head.count('<meta name="description" content="') == 1, \
            f"docs/{page} needs one description"


def test_dev_extra_installs_a_runnable_gui():
    """
    Installing the documented dev extra must give a working application.

    It previously omitted PyQt, so `python gui.py` failed immediately after
    following the install instructions. PyQt6-WebEngine is deliberately not
    in it: the documentation opens in the system browser.
    """

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dev_section = pyproject.split("dev = [", 1)[1].split("]", 1)[0]
    for requirement in ("PyQt6", "matplotlib", "pytest",
                        "ipykernel", "notebook", "pandas", "openpyxl", "xlrd"):
        assert requirement in dev_section, f"dev extra is missing {requirement}"
    assert "PyQt6-WebEngine" not in dev_section


def test_gui_imports_only_declared_packages():
    """
    Every third-party package the GUI imports is declared in pyproject.toml.

    gui.py imported pandas without declaring it, so `python gui.py` failed in a
    fresh environment while the tests, run where pandas happened to be
    installed, passed.
    """
    import ast
    import sys

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0] \
        + pyproject.split("dev = [", 1)[1].split("]", 1)[0]
    local = {"getelec", "gui", "gui_backend"}
    for script in ("gui.py", "gui_backend.py"):
        tree = ast.parse((ROOT / script).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top in sys.stdlib_module_names or top in local:
                    continue
                assert top in declared, f"{script} imports {top}, not in pyproject.toml"


def test_gui_reads_measured_data_files(tmp_path):
    """
    The GUI's fitting tab loads .txt, .csv and Excel files through
    gui_backend.read_two_columns. Whitespace-separated .txt used pandas'
    delim_whitespace, which pandas 3 removed, so every .txt file -- including
    examples/iv.txt -- failed to load, silently.
    """
    import pandas as pd

    import gui_backend

    voltage, current = gui_backend.read_two_columns(str(ROOT / "examples" / "iv.txt"))
    reference = np.loadtxt(ROOT / "examples" / "iv.txt")
    np.testing.assert_array_equal(voltage, reference[:, 0])
    np.testing.assert_array_equal(current, reference[:, 1])

    x, y = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])
    (tmp_path / "tabs.TXT").write_text("1\t10\n2   20\n3 30\n", encoding="utf-8")
    (tmp_path / "data.csv").write_text("1,10\n2,20\n3,30\n", encoding="utf-8")
    pd.DataFrame({0: x, 1: y}).to_excel(tmp_path / "data.xlsx", header=False, index=False)
    for name in ("tabs.TXT", "data.csv", "data.xlsx"):
        got_x, got_y = gui_backend.read_two_columns(str(tmp_path / name))
        np.testing.assert_array_equal(got_x, x)
        np.testing.assert_array_equal(got_y, y)

    (tmp_path / "header.csv").write_text("V,I\n1,10\n", encoding="utf-8")
    (tmp_path / "one.txt").write_text("1\n2\n", encoding="utf-8")
    (tmp_path / "data.dat").write_text("1 10\n", encoding="utf-8")
    for name, message in (("header.csv", "header"), ("one.txt", "two columns"),
                          ("data.dat", "Unsupported")):
        with pytest.raises(ValueError, match=message):
            gui_backend.read_two_columns(str(tmp_path / name))


def _fit_defaults(model):
    import gui_backend
    return {name: default for name, _, default, *_ in gui_backend.FIT_MODELS[model][2]}


def test_gui_fit_recovers_known_iv_parameters():
    """
    The GUI's I-V fit, on data generated from known parameters, returns them.

    Its predecessor fitted the raw emission area across ten decades from a
    start at the edge of its bounds, and stopped with a misfit seven times that
    of a two-parameter fit to the same curve. The area is now fitted as its
    logarithm, and only the ticked parameters are free.
    """
    import gui_backend

    truth = dict(_fit_defaults("I-V"), gamma=0.0042, area=30.0)
    voltage = np.linspace(700.0, 1200.0, 30)
    current = np.asarray(getelec.current_density(
        field=voltage * truth["gamma"], work_function=truth["work_function"],
        fermi_level=truth["fermi_level"], temperature=truth["temperature"],
        fast=True)) * truth["area"] * 1e-14 * 1e9

    result = gui_backend.fit("I-V", voltage, current, _fit_defaults("I-V"),
                             free=["gamma", "area"])
    assert result["values"]["gamma"] == pytest.approx(truth["gamma"], rel=1e-5)
    assert result["values"]["area"] == pytest.approx(truth["area"], rel=1e-4)
    assert result["values"]["work_function"] == truth["work_function"]   # fixed
    assert result["misfit"] < 1e-5
    assert set(result["errors"]) == {"gamma", "area"}
    np.testing.assert_allclose(result["fit"], current, rtol=1e-4)


def test_gui_fit_recovers_known_ted_parameters():
    """The TED fit returns the field, temperature and analyser offset it was made with."""
    import gui_backend

    energy = np.linspace(-1.5, 0.3, 120)
    truth = dict(_fit_defaults("TED"), field=5.5, temperature=420.0, offset=25.0)
    counts = gui_backend.fit("TED", energy, np.ones_like(energy), truth, free=[])["fit"]

    result = gui_backend.fit("TED", energy, 1000.0 * counts, _fit_defaults("TED"),
                             free=["field", "temperature", "offset"])
    assert result["values"]["field"] == pytest.approx(5.5, rel=1e-4)
    assert result["values"]["temperature"] == pytest.approx(420.0, rel=1e-3)
    assert result["values"]["offset"] == pytest.approx(25.0, abs=0.05)


def test_gui_fit_rejects_values_outside_their_range():
    import gui_backend

    values = dict(_fit_defaults("TED"), field=99.0)
    with pytest.raises(ValueError, match="outside the allowed range"):
        gui_backend.fit("TED", np.linspace(-1, 0, 10), np.ones(10), values, free=["field"])


def test_gui_ted_fit_does_not_depend_on_the_order_of_the_data():
    """
    A TED file listed from high to low energy fits like one listed upwards.

    The TED model builds its energy grid from the data and integrates the
    transmission along it; run backwards, the integral changed sign and the fit
    stopped at its first step with "Residuals are not finite".
    """
    import gui_backend

    energy = np.linspace(-1.5, 0.3, 120)
    truth = dict(_fit_defaults("TED"), field=5.5, temperature=420.0, offset=25.0)
    counts = 1000.0 * gui_backend.fit("TED", energy, np.ones_like(energy), truth,
                                      free=[])["fit"]
    free = ["field", "temperature", "offset"]
    upwards = gui_backend.fit("TED", energy, counts, _fit_defaults("TED"), free)
    downwards = gui_backend.fit("TED", energy[::-1], counts[::-1],
                                _fit_defaults("TED"), free)
    for name in free:
        assert downwards["values"][name] == upwards["values"][name]
    assert downwards["values"]["field"] == pytest.approx(5.5, rel=1e-4)


def test_gui_metal_distributions_are_not_extended_past_their_grid():
    """
    Outside the energies the emitter computed, the plotted TED and NED are zero.

    They were interpolated with the end values held, so asking for a wider
    range than the grid covers drew a flat line at each end: on a log axis the
    distribution appeared to level off at 4e-12 below the grid and 3e-20
    above it, at 2 V/nm.
    """
    import gui_backend

    params = dict(fermi_level=7.5, work_function=4.5, field=2.0, temperature=300.0)
    emitter = getelec.metal_emitter(**params)
    grid, _ = emitter.calculate_total_energy_distribution()
    for calculation in ("TED  (total energy dist.)", "NED  (normal energy dist.)"):
        result = gui_backend.calculate(calculation, "Metal", params, -4.0, 3.0, 0.01)
        energy = result["x"] + params["fermi_level"]
        outside = (energy < grid.min()) | (energy > grid.max())
        assert outside.sum() > 100, "the range should reach past the grid at both ends"
        assert np.all(result["y"][outside] == 0.0), calculation
        assert np.all(result["y"][~outside] > 0.0), calculation


def test_gui_self_test_passes_from_source(tmp_path):
    """
    The application's self-test, which compile.py runs on every build, passes.

    It drives the real window -- a decimal comma typed in a box and a change of
    material, which used to end the application -- finds the documentation,
    and checks the solvers, both networks, the data-file readers and a fit.
    Run in its own process, as a build runs it, off screen.
    """
    import os
    import subprocess
    import sys

    pytest.importorskip("PyQt6.QtWidgets")
    report = tmp_path / "self_test.txt"
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, str(ROOT / "gui.py"), "--self-test",
                             str(report)], env=environment, cwd=ROOT,
                            capture_output=True, text=True, timeout=600)
    text = report.read_text(encoding="utf-8") if report.exists() else result.stderr[-2000:]
    assert result.returncode == 0, text
    assert text.rstrip().endswith("ALL PASSED"), text


def test_documentation_tab_opens_the_pages_in_a_browser():
    """
    The Documentation tab opens each page in the system browser, and only on request.

    Nothing is opened when the application starts, a link opens the page it
    names, and a path outside the documentation is refused with a message
    rather than handed to the browser. The browser is replaced by a recorder.
    """
    import os
    import subprocess
    import sys
    import textwrap

    pytest.importorskip("PyQt6.QtWidgets")
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        import gui
        from PyQt6.QtWidgets import QApplication

        app = QApplication([])
        opened = []

        class Browser:
            @staticmethod
            def openUrl(url):
                opened.append(url.toLocalFile())
                return True

        gui.QDesktopServices = Browser
        tab = gui.DocumentationTab(None)
        assert not opened, "a browser was opened unasked"
        tab.load_page("usage.html")
        assert len(opened) == 1 and opened[0].endswith("docs/usage.html"), opened
        assert "Opened usage.html" in tab.page.toPlainText()
        tab.load_page("../README.md")
        assert len(opened) == 1, "a file outside the documentation was opened"
        assert "is missing from" in tab.page.toPlainText()
        assert "s [dot] barranco" in tab.page.toPlainText()
    """)
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, "-c", script], env=environment,
                            cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]


def test_guide_renders_into_paragraphs_with_code_left_alone():
    """
    The in-application guide joins a paragraph's lines, as Markdown does.

    Rendered one line at a time, every line of GUIDE.md became a paragraph of
    its own, bold text broken across lines kept its asterisks, and two
    asterisks inside code on one line (`m_e*` twice) paired into an emphasis
    crossing both code tags.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("regenerate", str(ROOT / "docs" / "regenerate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    html = module.render_markdown("That is **field\nemission**, at `m_e* = 1.2` and\n"
                                  "`m_e* = 1.64`.\n\n- one\n  item\n- two")
    assert html.count("<p>") == 1
    assert "<strong>field emission</strong>" in html
    assert "<code>m_e* = 1.2</code>" in html and "<em>" not in html
    assert "<li>one item</li><li>two</li>" in html
    assert module._link_target("GUIDE.md#units") == "usage.html#units"
    assert module._link_target("examples/iv.txt").startswith("https://github.com/")


def test_package_version_matches_pyproject():
    """
    The version is written in two places: pyproject.toml, which pip reads, and
    getelec.__version__, which the GUI and saved models report. They must agree.
    """

    import re

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert match, "no version in pyproject.toml"
    assert getelec.__version__ == match.group(1)


# --------------------------------------------------------------------------
# Generated documentation
# --------------------------------------------------------------------------

def test_bundled_documentation_covers_the_current_modules():
    """
    The API reference is generated from docstrings and goes stale silently.

    It once still described modules that had been merged away, which is only
    visible by opening the GUI and reading. Regenerate with docs/regenerate.py.
    """

    generated = ROOT / "docs" / "getelec"
    assert generated.is_dir()
    present = {path.stem for path in generated.glob("*.html")}
    for module in ("transmission_solver", "transmission_solutions",
                   "electron_emitter", "potential_barrier", "band_structure",
                   "electron_supply"):
        assert module in present, f"docs/getelec/{module}.html is missing"
    for gone in ("transmission_table", "wkb_numeric", "curved_barrier_ml",
                 "oscillations"):
        assert gone not in present, f"docs still describe the removed {gone}"


def test_usage_page_documents_the_new_features():

    usage = (ROOT / "docs" / "usage.html").read_text(encoding="utf-8")
    for topic in ("transmission_solutions", "AiryTriangular", "NeuralSolver",
                  "band gap", "graphical interface"):
        assert topic.lower() in usage.lower(), f"usage guide does not mention {topic}"
    assert "<aside>" in usage, "usage guide lost its navigation sidebar"


# --------------------------------------------------------------------------
# Supply normalisation and the zero-temperature limit
# --------------------------------------------------------------------------

def test_current_density_is_continuous_down_to_zero_temperature():
    """
    Cold field emission must not vanish at T = 0.

    The supply used to return k_B T ln(1 + exp(...)) with the k_B T left to the
    caller, and a separate T = 0 branch returning the already-integrated form.
    Multiplying that branch by T = 0 gave exactly zero current -- an eleven
    order of magnitude cliff across the 1e-6 K threshold, in the regime where
    tunnelling is strongest.
    """
    fields = dict(field=5.0, work_function=4.5, fermi_level=7.5)
    cold = getelec.current_density(temperature=0.0, **fields)
    warm = getelec.current_density(temperature=1.0, **fields)
    room = getelec.current_density(temperature=300.0, **fields)

    assert cold > 0
    # The T = 0 occupancy is a step, and where it falls between grid points
    # costs a couple of percent at the default resolution. Continuous, but not
    # to a part in a thousand.
    assert abs(warm - cold) / cold < 0.05, "T = 0 must join continuously onto T = 1 K"
    assert cold < room, "heating can only add current at fixed field"

    # Monotonic and smooth across the old threshold.
    temperatures = np.array([0.0, 1e-9, 1e-6, 1e-3, 1.0, 50.0, 300.0])
    currents = np.array([getelec.current_density(temperature=t, **fields)
                         for t in temperatures])
    assert np.all(np.diff(currents) >= -1e-9 * currents[:-1]


                  )


def test_supply_returns_the_same_physical_quantity_at_every_temperature():
    """The k_B T belongs to the supply, so the T -> 0 limit is well defined."""
    from getelec.electron_supply import LogFermiDirac

    energies = np.linspace(5.0, 9.0, 41)
    cold = LogFermiDirac(7.5, 0.0).get_supply(energies)
    nearly_cold = LogFermiDirac(7.5, 1e-3).get_supply(energies)

    # Below E_F the zero-temperature limit is exactly (E_F - E).
    below = energies < 7.4
    np.testing.assert_allclose(cold[below], 7.5 - energies[below], rtol=1e-12)
    np.testing.assert_allclose(nearly_cold[below], cold[below], atol=1e-6)
    assert np.all(cold[energies > 7.6] == 0.0)


def test_normal_energy_distribution_integrates_to_the_current_density():
    """
    The NED is the current-density integrand, so its integral is the current.

    It used to come out low by exactly the temperature -- a factor of 300 at
    room temperature -- because the distributions omitted a scaling the current
    density applied.
    """
    for temperature in (300.0, 1000.0):
        emitter = getelec.metal_emitter(field=5.0, temperature=temperature)
        energies, ned = emitter.calculate_normal_energy_distribution()
        integrated = np.trapezoid(ned, energies)
        assert abs(integrated - emitter.calculate_current_density()) \
            < 1e-3 * emitter.calculate_current_density()


# --------------------------------------------------------------------------
# The zero-temperature limit
# --------------------------------------------------------------------------

def test_current_density_is_continuous_through_absolute_zero():
    """
    T = 0 must be the cold-tunnelling limit, not a special case that returns 0.

    The supply function has a separate branch below 1e-6 K. It returns
    k_B T ln(1 + exp(-(E-E_F)/k_B T)), whose T -> 0 limit is (E_F - E) below the
    Fermi level and zero above -- the standard Fowler-Nordheim supply. If the
    k_B T were left to the caller instead, nothing could recover it at T = 0 and
    the current would collapse to zero across the branch.
    """
    fields = 5.0
    currents = [getelec.current_density(field=fields, temperature=T)
                for T in (0.0, 1e-9, 1e-4, 1.0, 300.0)]

    assert currents[0] > 0, "zero-temperature field emission must not vanish"
    # Continuous across the branch: the first four differ by far less than the
    # 300 K value differs from the cold limit.
    cold_spread = max(currents[:4]) / min(currents[:4])
    assert cold_spread < 1.05, f"discontinuity across the T=0 branch: {cold_spread}"
    assert currents[-1] > currents[0], "heating must increase emission"


def test_zero_temperature_supply_is_the_standard_triangular_form():
    """N(E) = E_F - E below the Fermi level, zero above."""
    supply = LogFermiDirac(fermi_level=7.5, temperature=0.0)
    energies = np.array([5.0, 7.0, 7.5, 8.0])
    np.testing.assert_allclose(supply.get_supply(energies), [2.5, 0.5, 0.0, 0.0])


def test_module_and_emitter_current_paths_agree():
    """
    Two implementations of the same integral must not drift apart.

    getelec.current_density() sweeps in a batch; the emitter computes one point.
    They previously used different prefactor conventions, which agreed for
    T > 0 and disagreed at T = 0.
    """
    for temperature in (0.0, 1.0, 300.0, 1500.0):
        batched = getelec.current_density(field=5.0, temperature=temperature)
        single = getelec.metal_emitter(
            field=5.0, temperature=temperature).calculate_current_density()
        assert abs(batched - single) / single < 1e-9


def test_smart_metal_grid_captures_essentially_all_the_current():
    """
    The two cutoffs must discard only what cannot contribute.

    Widening the grid by 8 eV below and 4 eV above must not move the current
    density: below the lower limit the barrier is wider than barrier_width, and
    above the upper limit the occupancy is below supply_threshold.
    """
    from getelec.electron_emitter import _SUPPLY_TO_A_PER_CM2

    solver = Noumerov()
    for field, temperature in ((7.0, 300.0), (5.0, 300.0), (3.0, 300.0),
                               (2.0, 2000.0), (1.5, 2500.0)):
        band = SmartMetal()
        energies = band.generate_band_structure(
            fermi_level=7.5, work_function=4.5,
            electric_field=field, temperature=temperature)
        barrier = SchottkyPotential(7.5, 4.5, field)
        supply = LogFermiDirac(7.5, temperature)
        step = band.energy_resolution

        def current(grid):
            return (np.sum(supply.get_supply(grid)
                           * solver.calculate_transmission(barrier, grid))
                    * step * _SUPPLY_TO_A_PER_CM2)

        wide = np.arange(max(step, energies[0] - 8.0), energies[-1] + 4.0 + step, step)
        assert abs(current(wide) - current(energies)) / current(wide) < 1e-4


def test_smart_metal_cuts_where_it_says_it_does():
    """Lower limit at the barrier-width criterion, upper at the supply one."""
    field = 3.0
    band = SmartMetal()
    energies = band.generate_band_structure(
        fermi_level=7.5, work_function=4.5, electric_field=field, temperature=300.0)

    # Barrier width at the lower limit should equal barrier_width.
    offset = energies[0] - 12.0
    discriminant = offset ** 2 - const.COULOMB_CONST * field
    width = np.sqrt(discriminant) / field
    assert abs(width - band.barrier_width) < 0.01
def test_barrier_top_scale_follows_the_three_quarter_power():
    """
    The curvature at the barrier top gives hbar*omega proportional to F**(3/4).

    x_max = sqrt(k_e/4F), so the curvature goes as F**(3/2) and its square root
    as F**(3/4). This sets the width of the above-barrier resonance, and is why
    it needs a 1e-4 eV energy grid to resolve at low field.
    """
    def scale(field):
        x_max = np.sqrt(const.COULOMB_CONST / (4.0 * field))
        curvature = const.COULOMB_CONST / (2.0 * x_max ** 3)
        return np.sqrt(2.0 * const.HBAR2_2M * curvature)

    fields = np.array([7.5e-4, 5e-3, 5e-2, 0.5])
    scales = np.array([scale(f) for f in fields])
    predicted = scales[0] * (fields / fields[0]) ** 0.75
    np.testing.assert_allclose(scales, predicted, rtol=1e-12)


def test_barrier_top_transmission_is_near_one_half():
    """Kemble gives exactly 1/2 at a parabolic barrier top."""
    for field in (5e-3, 5e-2, 0.5):
        v_max = 14.0 - np.sqrt(const.COULOMB_CONST * field)
        solver = Noumerov(x_metal=-1.0, x_vac_plus=10.0, max_barrier_width=60.0,
                          h=1e-3, seed="wkb", auto_domain=False)
        got = solver.calculate_transmission(
            SchottkyPotential(9.5, 4.5, field), np.array([v_max]))[0]
        assert 0.3 < got < 0.7, f"T(V_max) = {got} at F = {field}"
def test_trapezoid_integration_handles_edge_cases():
    from getelec.electron_emitter import get_energy_integral

    assert get_energy_integral(np.empty(0), np.empty(0)) == 0.0
    assert get_energy_integral(np.array([1.0]), np.array([5.0])) == 0.0
    # Exact for a linear integrand, and correct on a non-uniform grid.
    grid = np.array([0.0, 0.5, 2.0, 3.0])
    assert abs(get_energy_integral(grid, 2 * grid) - 9.0) < 1e-12


# --------------------------------------------------------------------------
# Barrier cache key, and the saturated-image barrier
# --------------------------------------------------------------------------

def test_barrier_cache_distinguishes_subclass_parameters():
    """
    A subclass that adds a parameter must not collide with its parent.

    The profile cache keyed on a fixed attribute list, so three barriers
    differing only in a new attribute returned identical transmissions from one
    solver instance -- silently, and only detectable because the numbers were
    too identical.
    """
    class Softened(SchottkyPotential):
        def __init__(self, *args, soft=0.05, **kwargs):
            super().__init__(*args, **kwargs)
            self.soft = soft

        def get_potential(self, x):
            x = np.asarray(x, float)
            return np.where(
                x > 0,
                self.fermi_level + self.work_function - self.electric_field * x
                - const.COULOMB_CONST / (4 * np.sqrt(x ** 2 + self.soft ** 2)),
                0.0)

    solver = Noumerov(x_metal=-1.0, x_vac_plus=10.0, max_barrier_width=60.0,
                      h=1e-3, seed="wkb", auto_domain=False)
    energies = np.array([13.78, 13.83, 14.0])
    results = [solver.calculate_transmission(Softened(9.5, 4.5, 0.05, soft=s), energies)
               for s in (0.02, 0.06, 0.15)]
    keys = {solver._barrier_key(Softened(9.5, 4.5, 0.05, soft=s))
            for s in (0.02, 0.06, 0.15)}
    assert len(keys) == 3
    assert not np.allclose(results[0], results[1])
    assert not np.allclose(results[1], results[2])


# --------------------------------------------------------------------------
# The full-wavefunction solver
# --------------------------------------------------------------------------

def test_both_solvers_give_the_same_transmission(barrier):
    """Keeping psi must not change the answer."""
    energies = np.linspace(4.0, 9.5, 40)
    np.testing.assert_allclose(
        Noumerov().calculate_transmission(barrier, energies),
        NoumerovFast().calculate_transmission(barrier, energies), rtol=0)


def test_transmission_can_be_rebuilt_from_the_wavefunction(barrier):
    """
    Extracting T by hand from psi must reproduce calculate_transmission.

    This is the check that the returned wavefunction is the one the solver
    actually used, rather than a separate and possibly inconsistent integration.
    """
    solver = Noumerov()
    energies = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    x, psi = solver.calculate_psi(barrier, energies)
    potential = barrier.get_potential(x)

    def wavevector(value):
        return np.sqrt(((energies - value) / const.HBAR2_2M).astype(complex))

    k_a, k_b = wavevector(potential[-1]), wavevector(potential[-2])
    x_a, x_b = x[-1], x[-2]
    m00, m01 = np.exp(1j * k_a * x_a), np.exp(-1j * k_a * x_a)
    m10, m11 = np.exp(1j * k_b * x_b), np.exp(-1j * k_b * x_b)
    amplitude = (m11 * psi[:, -1] - m01 * psi[:, -2]) / (m00 * m11 - m01 * m10)
    by_hand = 1.0 / (k_a.real * np.abs(amplitude) ** 2)

    np.testing.assert_allclose(by_hand,
                               solver.calculate_transmission(barrier, energies),
                               rtol=1e-12)


def test_probability_current_is_conserved(barrier):
    """
    A stationary scattering state carries a position-independent current.

    The Noumerov scheme does not enforce this, so its variation across the grid
    is an independent check on the integration.
    """
    # A nanometre of metal is not the default any more, so ask for one: the
    # point of the check is to watch the current over a long stretch of flat
    # potential, where nothing physical should change it.
    solver = Noumerov(x_metal=-1.0)
    energies = np.array([7.0, 8.0, 9.0])
    x, current = solver.calculate_probability_current(barrier, energies)
    inside = (x > -0.5) & (x < 0.0)
    for row in current:
        values = row[inside]
        spread = (values.max() - values.min()) / abs(values.mean())
        assert spread < 1e-6, f"current varies by {spread:.2e} across the metal"


def test_calculate_psi_can_return_a_subset_of_positions(barrier):
    solver = Noumerov()
    energies = np.array([7.0, 8.0])
    wanted = np.array([2.0, 1.0, 0.5, 0.0])
    x, psi = solver.calculate_psi(barrier, energies, x_points=wanted)
    assert psi.shape == (energies.size, wanted.size)
    assert np.abs(x - wanted).max() < 2e-3      # snapped to the nearest node


def test_fast_preset_never_returns_the_wavefunction_solver():
    """The preset exists for speed; keeping psi would defeat it."""
    assert type(Noumerov.fast()) is NoumerovFast
    assert type(NoumerovFast.fast()) is NoumerovFast


def test_reusing_an_emitter_is_free_when_nothing_changed():
    """
    The fingerprint cache makes a repeated identical query essentially free,
    which is the case for rebuilding versus updating.
    """
    import time

    emitter = getelec.metal_emitter(field=5.0)
    emitter.calculate_current_density()

    def time_once(fn):
        start = time.perf_counter()
        fn()
        return time.perf_counter() - start

    # Interleaved, and compared on medians: call times on a busy machine are
    # bimodal, so a single sample of each can land in different modes.
    calls = (emitter.calculate_current_density,
             lambda: getelec.metal_emitter(field=5.0).calculate_current_density())
    for call in calls:
        call()
    samples = [[time_once(call) for call in calls] for _ in range(9)]
    cached, rebuilt = (float(np.median([row[i] for row in samples])) for i in (0, 1))
    assert cached < rebuilt / 10, f"cached {cached:.2e} s against rebuilt {rebuilt:.2e} s"


# --------------------------------------------------------------------------
# Wavefunction access from the emitter
# --------------------------------------------------------------------------

def test_emitter_psi_is_consistent_with_its_own_transmission(emitter):
    """
    psi from the emitter must reproduce the transmission the emitter reports.

    Same barrier, same solver, same energies -- so extracting the flux ratio
    from the returned wavefunction has to give back what
    calculate_transmission_coefficient says.
    """
    energies_in = np.array([6.0, 7.0, 8.0])
    energies, x, psi = emitter.calculate_psi(energies=energies_in)
    np.testing.assert_allclose(energies, energies_in)
    assert psi.shape == (energies_in.size, x.size)

    potential = emitter.potential.get_potential(x)

    def wavevector(value):
        return np.sqrt(((energies - value) / const.HBAR2_2M).astype(complex))

    k_a, k_b = wavevector(potential[-1]), wavevector(potential[-2])
    m00, m01 = np.exp(1j * k_a * x[-1]), np.exp(-1j * k_a * x[-1])
    m10, m11 = np.exp(1j * k_b * x[-2]), np.exp(-1j * k_b * x[-2])
    amplitude = (m11 * psi[:, -1] - m01 * psi[:, -2]) / (m00 * m11 - m01 * m10)
    from_psi = 1.0 / (k_a.real * np.abs(amplitude) ** 2)

    reported = emitter.solver.calculate_transmission(emitter.potential, energies)
    np.testing.assert_allclose(from_psi, reported, rtol=1e-12)


def test_emitter_psi_defaults_to_its_own_energy_grid(emitter):
    energies, x, psi = emitter.calculate_psi(x_points=np.linspace(3.0, -0.5, 40))
    grid, _ = emitter.calculate_transmission_coefficient()
    np.testing.assert_allclose(energies, grid)
    assert psi.shape == (grid.size, 40)


def test_emitter_probability_current_is_conserved(emitter):
    energies, x, current = emitter.calculate_probability_current(
        energies=np.array([7.0, 8.0]))
    inside = (x > -0.5) & (x < 0.0)
    for row in current:
        values = row[inside]
        assert (values.max() - values.min()) / abs(values.mean()) < 1e-6


def test_emitter_psi_refuses_solvers_that_do_not_form_it():
    """
    A solver that never builds psi must say so rather than have one
    substituted, which would not correspond to the transmission it reports.
    """
    for kwargs in ({"method": "wkb"}, {"fast": True}):
        emitter = getelec.metal_emitter(field=5.0, **kwargs)
        with pytest.raises(TypeError, match="does not produce a wavefunction"):
            emitter.calculate_psi(energies=np.array([7.0]))


def test_semiconductor_psi_returns_both_bands_with_their_own_grids():
    emitter = getelec.semiconductor_emitter(field=5.0)
    e_cb, x_cb, psi_cb, e_vb, x_vb, psi_vb = emitter.calculate_psi(
        energies=([14.0, 15.0], [11.0, 12.0]))
    assert psi_cb.shape == (2, x_cb.size)
    assert psi_vb.shape == (2, x_vb.size)
    assert np.all(np.isfinite(psi_cb)) and np.all(np.isfinite(psi_vb))


def test_large_psi_request_warns_about_memory():
    """
    The warning fires on the product of energies and grid points, so ask for a
    grid fine enough to cross 100 MB now that the spatial domain is shorter.
    """
    from getelec.band_structure import SmartMetal
    from getelec.electron_emitter import MetalEmitter
    from getelec.electron_supply import LogFermiDirac

    emitter = MetalEmitter(SchottkyPotential(7.5, 4.5, 5.0), Noumerov(),
                           LogFermiDirac(7.5, 300.0),
                           SmartMetal(energy_resolution=0.002))
    with pytest.warns(ResourceWarning, match="MB"):
        emitter.calculate_psi()


def test_semiconductor_probability_current_returns_both_bands():
    """
    Inherited from the metal emitter this silently returned the conduction
    band alone, because that is the emitter's first energy grid. Every
    two-band accessor must return two bands.
    """
    emitter = getelec.semiconductor_emitter(field=5.0)
    result = emitter.calculate_probability_current(
        energies=([14.0, 15.0], [11.0, 12.0]))
    assert len(result) == 6, "expected both bands"

    e_cb, x_cb, current_cb, e_vb, x_vb, current_vb = result
    assert current_cb.shape == (2, x_cb.size)
    assert current_vb.shape == (2, x_vb.size)

    # And it must agree with calculate_psi on shapes and grids.
    _, px_cb, psi_cb, _, px_vb, psi_vb = emitter.calculate_psi(
        energies=([14.0, 15.0], [11.0, 12.0]))
    assert psi_cb.shape == current_cb.shape
    assert psi_vb.shape == current_vb.shape
    np.testing.assert_allclose(px_cb, x_cb)
    np.testing.assert_allclose(px_vb, x_vb)

def test_intro_notebook_cells_all_run(monkeypatch):
    """
    Every code cell in the tutorial must execute, in order, in one namespace.

    A tutorial whose cells do not run is worse than no tutorial, and the usual
    way they rot is an API change that nobody thinks to check against the docs.
    """
    import io
    from contextlib import redirect_stdout

    notebook_path = ROOT / "examples" / "intro_to_getelec.ipynb"
    if not notebook_path.exists():
        pytest.skip("tutorial notebook not present")
    nbformat = pytest.importorskip("nbformat")
    # Run from the repository root, not the notebook's folder: an editor can be
    # set to start the kernel there, and the notebook once failed that way
    # because it opened its data files by bare name. It must find them either way.
    monkeypatch.chdir(ROOT)

    import matplotlib
    matplotlib.use("Agg")

    notebook = nbformat.read(str(notebook_path), as_version=4)
    namespace = {}
    for number, cell in enumerate(
            (c for c in notebook.cells if c.cell_type == "code"), start=1):
        with redirect_stdout(io.StringIO()):
            try:
                exec(cell.source, namespace)
            except Exception as error:  # noqa: BLE001 - report which cell
                pytest.fail(f"notebook cell {number} failed: "
                            f"{type(error).__name__}: {error}\n{cell.source[:300]}")


def test_emitter_repr_shows_its_parameters():
    """Printing an emitter should say what it is set to, not where it lives."""
    text = repr(getelec.metal_emitter(work_function=4.5, fermi_level=7.5,
                                      field=5.0, temperature=300.0))
    assert "MetalEmitter" in text
    for expected in ("work_function=4.5", "fermi_level=7.5",
                     "electric_field=5", "temperature=300"):
        assert expected in text
    assert "0x" not in text, "still printing a memory address"


# --------------------------------------------------------------------------
# The image-term cutoff
# --------------------------------------------------------------------------

def test_image_cutoff_is_the_inner_zero_crossing():
    """
    The clamp must sit where the potential first crosses zero, not at a fixed
    distance. Below it the expression is negative only because the image term
    diverges; above it, negative values are the real barrier falling away.
    """
    from getelec.potential_barrier import get_image_cutoff

    for field in (1.0, 5.0, 13.0, 25.0):
        cutoff = get_image_cutoff(7.5, 4.5, field)
        barrier = SchottkyPotential(7.5, 4.5, field)
        # Just inside: clamped to zero. Just outside: positive and rising.
        assert barrier.get_potential(np.array([cutoff * 0.9]))[0] == 0.0
        assert barrier.get_potential(np.array([cutoff * 1.1]))[0] > 0.0


def test_potential_is_not_flattened_beyond_the_barrier_at_high_field():
    """
    Above about 12 V/nm the outer turning point moves inside 1 nm.

    A fixed 1 nm clamp then flattened the far side of the barrier to zero and
    let it drop discontinuously at exactly x = 1. The step reflected and made
    the transmission oscillate.
    """
    field = 20.0
    barrier = SchottkyPotential(7.5, 4.5, field)
    total = 12.0
    outer_turning_point = (total + np.sqrt(total ** 2
                                           - const.COULOMB_CONST * field)) / (2 * field)
    assert outer_turning_point < 1.0, "this test needs a sub-nm barrier"

    x = np.linspace(outer_turning_point + 0.02, 1.5, 500)
    potential = barrier.get_potential(x)
    # Past the turning point the potential must fall monotonically, with no
    # flat stretch and no step.
    assert np.all(potential < 0)
    assert np.all(np.diff(potential) < 0)
    assert np.abs(np.diff(potential)).max() < 0.1


def test_high_field_transmission_is_smooth():
    """
    The symptom the clamp produced: a wavy transmission at high field.

    log D against energy is smooth for a simple barrier, so a large third
    difference is a direct test for spurious ripple.
    """
    energies = np.arange(1.0, 14.0, 0.05)
    for field in (13.0, 15.0, 20.0):
        transmission = NoumerovFast(h=1e-3).calculate_transmission(
            SchottkyPotential(7.5, 4.5, field), energies)
        curvature = np.abs(np.diff(np.log(transmission), n=3))
        assert curvature.max() < 1e-3, (
            f"log D ripples at F = {field}: max third difference "
            f"{curvature.max():.2e}")


def test_high_field_matches_the_reference_implementation():
    """
    Cross-check against NoumerovReference, the plain, unoptimised integration
    on a fixed 20 nm domain with a plane-wave seed -- a different
    implementation of the same physics, sharing no code with the solver.
    """
    energies = np.array([2.0, 5.0, 8.0, 11.0])
    for field in (13.0, 20.0):
        barrier = SchottkyPotential(7.5, 4.5, field)
        reference = NoumerovReference().calculate_transmission(barrier, energies)
        got = NoumerovFast(h=1e-3).calculate_transmission(barrier, energies)
        assert np.max(np.abs(got - reference) / reference) < 0.01


# --------------------------------------------------------------------------
# Portability
# --------------------------------------------------------------------------

def test_no_text_file_is_read_without_an_explicit_encoding():
    """
    Reading text without ``encoding=`` uses the locale codec.

    On Linux and macOS that is UTF-8 and everything works. On Windows it is
    cp1252, and every file in this project containing an em-dash or a times
    sign then raises UnicodeDecodeError. The failure appears only on Windows,
    which is exactly the sort of thing that survives a long time unnoticed.
    """
    import re

    offenders = []
    for path in ROOT.rglob("*.py"):
        if any(part in {".venv", "build", "dist", "__pycache__"} for part in path.parts):
            continue
        source = path.read_text(encoding="utf-8")
        for number, line in enumerate(source.splitlines(), start=1):
            if re.search(r"\.(read_text|write_text)\(\s*\)", line):
                offenders.append(f"{path}:{number}")
            elif re.search(r"\.(read_text|write_text)\(", line) and "encoding" not in line:
                offenders.append(f"{path}:{number}")
            elif re.search(r"\bopen\([^)]*['\"][rwa]", line) and "encoding" not in line \
                    and "b" not in line.split("open(")[1][:30]:
                offenders.append(f"{path}:{number}")
    assert not offenders, ("text opened without an explicit encoding: "
                           + ", ".join(offenders))


def test_repository_root_is_importable_under_plain_pytest():
    """
    gui_backend and compile.py live at the repository root and the tests import
    them. `python -m pytest` puts the working directory on sys.path; plain
    `pytest` does not, so this used to pass one way and fail the other.
    `pythonpath = ["."]` in pyproject.toml makes both behave the same.
    """
    import importlib

    assert importlib.import_module("gui_backend") is not None


def test_runtime_dependencies():
    """
    pyproject.toml is the only dependency list.
    """

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    runtime = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    for package in ("numpy", "scipy", "numba"):
        assert package in runtime
    # Nothing imports joblib or scikit-learn at runtime any more: the network
    # weights are plain arrays. scikit-learn is installed separately, to train one.
    for package in ("joblib", "scikit-learn"):
        assert package not in runtime


# --------------------------------------------------------------------------
# TED and NED are built from different things
# --------------------------------------------------------------------------

def test_both_distributions_integrate_to_the_current_density():
    """
    The check that catches a swapped supply function.

    NED(E) = l(E) D(E) uses the log supply; TED(E) = f(E) x integral of D uses
    the bare occupancy. They are related by dl/dE = -f, so integrating one by
    parts gives the other and **both must integrate to the same current**.

    Using the log supply for the TED as well produces a curve that looks
    plausible and peaks in nearly the right place, but integrates to 22% of the
    current instead of 100%. Nothing short of this identity catches it.
    """
    emitter = getelec.metal_emitter(field=5.0, temperature=300.0)
    current = emitter.calculate_current_density()

    energies, ted = emitter.calculate_total_energy_distribution()
    grid, ned = emitter.calculate_normal_energy_distribution()

    assert abs(np.trapezoid(ned, grid) - current) / current < 1e-3
    assert abs(np.trapezoid(ted, energies) - current) / current < 1e-6


def test_supply_and_occupancy_are_different_functions():
    """
    l(E) is the occupancy integrated over transverse momentum; f(E) is not.

    They agree nowhere except in the limits, and dl/dE = -f.
    """
    supply = LogFermiDirac(fermi_level=7.5, temperature=300.0)
    energies = np.linspace(5.0, 10.0, 4001)

    log_term = supply.get_supply(energies)
    occupancy = supply.get_occupancy(energies)

    assert abs(occupancy[np.argmin(np.abs(energies - 7.5))] - 0.5) < 1e-9
    assert not np.allclose(log_term, occupancy)
    # dl/dE = -f, which is the relation that makes both distributions integrate
    # to the same current.
    derivative = np.gradient(log_term, energies)
    interior = slice(5, -5)
    np.testing.assert_allclose(derivative[interior], -occupancy[interior], atol=2e-3)


def test_occupancy_is_available_on_every_supply():
    for supply in (LogFermiDirac(7.5, 300.0), FermiDirac(7.5, 300.0)):
        values = supply.get_occupancy(np.array([5.0, 7.5, 10.0]))
        assert np.all((values >= 0) & (values <= 1))
        assert abs(values[1] - 0.5) < 1e-9
    # And at absolute zero it is a step.
    np.testing.assert_allclose(
        LogFermiDirac(7.5, 0.0).get_occupancy(np.array([7.0, 8.0])), [1.0, 0.0])


def test_a_metal_gives_the_same_results_with_either_supply_class():
    """
    FermiDirac and LogFermiDirac describe the same electrons, so every result agrees.

    Both are set by the Fermi level and the temperature alone; they differ only
    in what get_supply returns, f or l. The metal NED used to take whichever
    that was, and with FermiDirac it became f D, which lacks the integral over
    transverse energy: it integrated to about 1/d times the current density,
    4.5x at 5 V/nm. Nothing covered it, because the shortcuts always attach a
    LogFermiDirac.
    """
    from getelec.band_structure import SmartMetal
    from getelec.electron_emitter import MetalEmitter

    results = []
    for supply in (LogFermiDirac(7.5, 300.0), FermiDirac(7.5, 300.0)):
        emitter = MetalEmitter(SchottkyPotential(7.5, 4.5, 5.0), Noumerov(),
                               supply, SmartMetal())
        energies, ned = emitter.calculate_normal_energy_distribution()
        current = emitter.calculate_current_density()
        assert abs(np.trapezoid(ned, energies) / current - 1) < 1e-3
        results.append((current, emitter.calculate_nottingham_heat(), ned,
                        emitter.calculate_total_energy_distribution()[1],
                        emitter.calculate_supply_function()[1]))
    for log_value, bare_value in zip(*results):
        np.testing.assert_array_equal(log_value, bare_value)


def test_ted_and_ned_peak_in_different_places():
    """
    A structural check that they are not the same curve.

    The NED peaks below the Fermi level, where the supply is large and the
    barrier still thin enough. The TED peaks closer to it, because the
    cumulative transmission keeps rising with energy.
    """
    emitter = getelec.metal_emitter(field=5.0, temperature=300.0)
    energies, ted = emitter.calculate_total_energy_distribution()
    grid, ned = emitter.calculate_normal_energy_distribution()
    ted_peak = energies[np.argmax(ted)] - 7.5
    ned_peak = grid[np.argmax(ned)] - 7.5
    assert ted_peak > ned_peak + 0.05, (
        f"TED peaks at {ted_peak:+.3f}, NED at {ned_peak:+.3f} -- too close, "
        f"are they built from the same supply?")


# --------------------------------------------------------------------------
# The conduction band window
# --------------------------------------------------------------------------

def _semiconductor(electron_mass, step=0.002):
    from getelec.electron_emitter import SemiconductorEmitter
    from getelec.electron_supply import FermiDirac

    top, gap = 12.5, 0.7
    fermi = top + gap * 0.5
    band = Semiconductor(lower_energy_limit=10, top_valence=top, fermi_level=fermi,
                         band_gap=gap, upper_energy_limit=18, energy_resolution=step,
                         electron_eff_mass=electron_mass, hole_eff_mass=0.59)
    return SemiconductorEmitter(SchottkyPotential(fermi, 4.85, 1.0), Noumerov(),
                                FermiDirac(fermi, 300.0), band)


def test_conduction_window_follows_equation_eight():
    """
    The E_z window runs from E down to E - (m*/m)(E - E_C).

    Paper Eq. (8) with A.16/A.21: the transverse energy is bounded by
    E_r <= (m*/m) x kinetic energy, so the lower limit uses alpha_bar = 1 - m*/m
    measured from the band edge. Using alpha instead makes the window
    (m*/m)/(1 - m*/m) times too narrow -- 49x at m* = 0.98 -- and puts the lower
    limit above E once m* > m, which makes the band vanish.
    """
    for mass in (0.5, 0.98, 1.2, 1.64):
        band = Semiconductor(top_valence=12.5, band_gap=0.7, electron_eff_mass=mass)
        conduction, scaled, _, _ = band.generate_band_structure()
        bottom = band.top_valence + band.band_gap
        np.testing.assert_allclose(
            scaled, conduction - mass * (conduction - bottom), rtol=1e-12)


def _sc_emitter(electron_mass=0.98, hole_mass=0.59):
    return getelec.semiconductor_emitter(
        work_function=4.5, fermi_level=13.0, field=5.0, temperature=300.0,
        band_gap=0.7, top_valence=12.5,
        electron_eff_mass=electron_mass, hole_eff_mass=hole_mass)


@pytest.mark.parametrize("masses", [(0.5, 0.59), (0.98, 0.59), (1.64, 1.2)])
def test_window_transmission_is_the_ted_without_its_occupancy(masses):
    """
    g(E) is read out of the TED calculation, not recomputed beside it.

    The interface plots g so that changing an effective mass visibly changes
    something: D(E) holds no mass at all, and the masses enter only through the
    limits g is integrated between. That only helps if the plotted curve is the
    same g the current is built from, so the identity is exact, not approximate:
    ``TED = f(E) g(E)``, to the last bit.
    """
    from getelec.electron_emitter import _SUPPLY_TO_A_PER_CM2

    emitter = _sc_emitter(*masses)
    e_cb, g_cb, e_vb, g_vb = emitter.calculate_window_integrated_transmission()
    _, ted_cb, _, ted_vb = emitter.calculate_total_energy_distribution()

    for band, energies, window, ted in (("conduction", e_cb, g_cb, ted_cb),
                                        ("valence", e_vb, g_vb, ted_vb)):
        assert np.all(window >= 0.0), f"{band} g < 0 at m* = {masses}"
        assert np.all(np.isfinite(window))
        rebuilt = emitter.supply.get_occupancy(energies) * window * _SUPPLY_TO_A_PER_CM2
        np.testing.assert_array_equal(rebuilt, ted, err_msg=f"{band} at {masses}")


@pytest.mark.parametrize("band, mass", [("conduction", 0.98), ("conduction", 1.64),
                                        ("valence", 0.59)])
def test_window_transmission_matches_quadrature(band, mass):
    """The window integral of Eq. (8), against D integrated over it directly."""
    bottom, top = 13.2, 12.5
    emitter = (_sc_emitter(electron_mass=mass) if band == "conduction"
               else _sc_emitter(hole_mass=mass))
    e_cb, g_cb, e_vb, g_vb = emitter.calculate_window_integrated_transmission()
    energies, window = (e_cb, g_cb) if band == "conduction" else (e_vb, g_vb)

    def exact(energy):
        lower = (energy - mass * (energy - bottom) if band == "conduction"
                 else energy - mass * (top - energy))
        grid = np.linspace(lower, energy, 4001)
        return np.trapezoid(
            emitter.solver.calculate_transmission(emitter.potential, grid), grid)

    live = np.flatnonzero(window > window.max() * 1e-4)
    for i in live[np.linspace(0, live.size - 1, 5).astype(int)]:
        assert abs(window[i] / exact(energies[i]) - 1) < 1e-3, (band, mass, energies[i])


def test_window_transmission_moves_with_the_mass_where_d_does_not():
    """
    The reason g is offered at all.

    Changing an effective mass leaves D(E) bit-identical -- the barrier contains
    no mass, in 3.x as in the published 2.0 -- while the current moves, which
    reads in the interface as a control that does nothing. g responds, and in
    the same direction the current does: a heavier mass opens the window wider.
    """
    light, heavy = _sc_emitter(electron_mass=0.5), _sc_emitter(electron_mass=1.64)
    _, d_light, _, _ = light.calculate_transmission_coefficient()
    _, d_heavy, _, _ = heavy.calculate_transmission_coefficient()
    np.testing.assert_array_equal(d_light, d_heavy)

    _, g_light, _, _ = light.calculate_window_integrated_transmission()
    _, g_heavy, _, _ = heavy.calculate_window_integrated_transmission()
    assert np.all(g_heavy >= g_light - 1e-12), "a wider window cannot integrate to less"
    assert g_heavy.max() > 1.2 * g_light.max()


def test_conduction_band_survives_a_heavy_effective_mass():
    """m* > m must not make the conduction band disappear."""
    for mass in (1.0, 1.2, 1.64, 2.5):
        emitter = _semiconductor(mass)
        energies, ted, _, _ = emitter.calculate_total_energy_distribution()
        assert ted.max() > 0, f"conduction band vanished at m* = {mass}"
        assert emitter.calculate_current_density() > 0


def test_conduction_ted_and_ned_are_not_the_same_curve():
    """
    They are different functions of different variables.

    With the window 49x too narrow the transmission barely varies across it, so
    the TED collapsed onto the shape of the NED -- which is how the wrong window
    showed itself.
    """
    emitter = _semiconductor(0.98)
    energies, ted, _, _ = emitter.calculate_total_energy_distribution()
    normal, ned, _, _ = emitter.calculate_normal_energy_distribution()

    common = np.linspace(max(energies[0], normal[0]),
                         min(energies[-1], normal[-1]), 400)
    on_ted = np.interp(common, energies, ted)
    on_ned = np.interp(common, normal, ned)
    usable = (on_ted > on_ted.max() * 1e-6) & (on_ned > on_ned.max() * 1e-6)
    ratio = on_ted[usable] / on_ned[usable]
    assert ratio.max() / ratio.min() > 3.0, (
        "conduction TED and NED have the same shape; is the window too narrow?")


def test_both_bands_conserve_current_across_effective_masses():
    """
    The TED and NED integrals must agree, for any effective mass.

    The NED is returned on its own grid: normal energy extends below the band
    edge when m* > m, because the transverse energy can then exceed the total
    kinetic energy. Evaluating it on the total-energy grid loses ~17% at
    m* = 1.64.
    """
    for mass in (0.5, 0.98, 1.0, 1.2, 1.64, 2.5):
        emitter = _semiconductor(mass)
        e_cb, ted_cb, e_vb, ted_vb = emitter.calculate_total_energy_distribution()
        x_cb, ned_cb, x_vb, ned_vb = emitter.calculate_normal_energy_distribution()
        for label, (grid_t, dist_t, grid_n, dist_n) in (
                ("conduction", (e_cb, ted_cb, x_cb, ned_cb)),
                ("valence", (e_vb, ted_vb, x_vb, ned_vb))):
            total = np.trapezoid(dist_t, grid_t)
            normal = np.trapezoid(dist_n, grid_n)
            assert abs(normal - total) / total < 0.02, (
                f"{label} band at m* = {mass}: TED {total:.4e} vs NED {normal:.4e}")
            assert dist_t.min() >= 0 and dist_n.min() >= 0


# --------------------------------------------------------------------------
# Semiconductor distributions against brute-force quadrature
# --------------------------------------------------------------------------
# The TED and NED are built with shortcuts: one running integral of D for every
# TED window, an inverse of the lower-limit map and the log supply for the NED.
# These tests rebuild both from the kinematics alone -- the window limits
# written out, f integrated numerically -- sharing only the transmission D.

_SC_TOP, _SC_GAP, _SC_PHI = 12.5, 0.7, 4.85
_SC_FERMI = _SC_TOP + _SC_GAP / 2
_SC_EC = _SC_TOP + _SC_GAP


def _semiconductor_case(kind, mass, field=5.0, step=0.002):
    from getelec.band_structure import Semiconductor
    from getelec.electron_supply import FermiDirac

    band = Semiconductor(lower_energy_limit=10.0, top_valence=_SC_TOP,
                         fermi_level=_SC_FERMI, band_gap=_SC_GAP, upper_energy_limit=18.0,
                         energy_resolution=step,
                         electron_eff_mass=mass if kind == "cb" else 1.0,
                         hole_eff_mass=mass if kind == "vb" else 0.59)
    cb, scaled_cb, vb, scaled_vb = band.generate_band_structure()
    energies, lower = (cb, scaled_cb) if kind == "cb" else (vb, scaled_vb)
    return (SchottkyPotential(_SC_FERMI, _SC_PHI, field), FermiDirac(_SC_FERMI, 300.0),
            energies, lower)


class _Quadrature:
    """The same distributions from the kinematics, by explicit quadrature."""

    def __init__(self, solver, potential, supply, energies, kind, mass):
        from scipy.integrate import cumulative_trapezoid

        self.solver, self.potential, self.kind, self.mass = solver, potential, kind, mass
        self.e_min, self.e_max = energies.min(), energies.max()
        self.f = supply.get_occupancy
        self.e = np.linspace(self.e_min, self.e_max, 200001)
        self.cum = cumulative_trapezoid(self.f(self.e), self.e, initial=0.0)

    def lower(self, E):
        if self.kind == "cb":
            return _SC_EC + (1.0 - self.mass) * (E - _SC_EC)
        return E - self.mass * (_SC_TOP - E)

    def interval(self, z):
        """Total energies of the band whose window contains the normal energy z."""
        if self.kind == "vb":
            inverse = (z + self.mass * _SC_TOP) / (1.0 + self.mass)
            return np.maximum(z, self.e_min), np.minimum(inverse, self.e_max)
        slope = 1.0 - self.mass
        if abs(slope) < 1e-12:
            return np.where(z >= _SC_EC, z, np.inf), np.full_like(z, self.e_max)
        inverse = _SC_EC + (z - _SC_EC) / slope
        if slope > 0:
            return np.maximum(z, self.e_min), np.minimum(inverse, self.e_max)
        return np.maximum(np.maximum(z, inverse), self.e_min), np.full_like(z, self.e_max)

    def ned(self, z):
        a, b = self.interval(np.asarray(z, dtype=float))
        weight = np.where(a < b, np.interp(b, self.e, self.cum)
                          - np.interp(a, self.e, self.cum), 0.0)
        return self.solver.calculate_transmission(self.potential, z) * weight

    def ted(self, E):
        lo, hi = sorted((float(self.lower(E)), float(E)))
        x = np.linspace(lo, hi, 2001)
        return self.f(np.array([E]))[0] * np.trapezoid(
            self.solver.calculate_transmission(self.potential, x), x)


@pytest.mark.parametrize("kind, mass", [("cb", 0.5), ("cb", 0.98), ("cb", 1.02),
                                        ("cb", 1.64), ("vb", 0.3), ("vb", 1.2)])
def test_semiconductor_distributions_match_quadrature(kind, mass):
    """Both distributions, point by point, against the kinematics written out."""
    from getelec.electron_emitter import _band_window_distributions

    solver = Noumerov()
    potential, supply, energies, lower = _semiconductor_case(kind, mass)
    ted, _, normal, ned = _band_window_distributions(energies, lower, solver,
                                                     potential, supply)
    exact = _Quadrature(solver, potential, supply, energies, kind, mass)

    live = np.flatnonzero(ted > ted.max() * 1e-3)
    for i in live[np.linspace(0, live.size - 1, 4).astype(int)]:
        assert abs(ted[i] / exact.ted(energies[i]) - 1) < 2e-4, (kind, mass, energies[i])
    live = np.flatnonzero(ned > ned.max() * 1e-3)
    pick = live[np.linspace(0, live.size - 1, 8).astype(int)]
    np.testing.assert_allclose(ned[pick], exact.ned(normal[pick]), rtol=2e-4)


@pytest.mark.parametrize("mass", [0.95, 0.99, 1.0, 1.01, 1.05])
def test_ned_conserves_the_current_for_masses_near_the_free_mass(mass):
    """
    Near m* = m the conduction NED changes over |1 - m*/m| k_B T at the band
    edge -- 0.0003 eV at m* = 0.99, far below the grid step. Unresolved, its
    integral missed the current by up to 2.6%.
    """
    from getelec.electron_emitter import _band_window_distributions

    potential, supply, energies, lower = _semiconductor_case("cb", mass)
    ted, _, normal, ned = _band_window_distributions(energies, lower, Noumerov(),
                                                     potential, supply)
    assert abs(np.trapezoid(ned, normal) / np.trapezoid(ted, energies) - 1) < 2e-3


def test_valence_ned_is_zero_above_the_band_top_and_conserves_the_current():
    """No electron of the band lies above E_V; the NED used to take weight there."""
    from getelec.electron_emitter import _band_window_distributions

    potential, supply, energies, lower = _semiconductor_case("vb", 0.59, field=1.0)
    ted, _, normal, ned = _band_window_distributions(energies, lower, Noumerov(),
                                                     potential, supply)
    assert np.all(ned[normal > energies.max()] == 0.0)
    assert abs(np.trapezoid(ned, normal) / np.trapezoid(ted, energies) - 1) < 1e-3


def test_normal_energy_below_the_conduction_edge_matches_a_k_space_count():
    """
    The share of the current with E_z below E_C (m* > m), from the code's NED
    and from an independent count in k-space: E = E_C + (m/m*)(u + s),
    E_z = E - s, with u, s the normal and parallel kinetic energies in units of
    the free mass and the flux measure uniform in (u, s). Read at exactly E_C.
    """
    from scipy.integrate import cumulative_trapezoid
    from getelec.electron_emitter import _band_window_distributions

    mass = 1.64
    solver = Noumerov()
    potential, supply, energies, lower = _semiconductor_case("cb", mass)
    _, _, normal, ned = _band_window_distributions(energies, lower, solver, potential, supply)
    cumulative = cumulative_trapezoid(ned, normal, initial=0.0)
    code = np.interp(_SC_EC, normal, cumulative) / cumulative[-1]

    h = 0.002
    u = np.arange(h / 2, mass * 1.0, h)
    U, S = np.meshgrid(u, u, indexing="ij")
    E = _SC_EC + (U + S) / mass
    keep = E <= _SC_EC + 1.0
    E, Ez = E[keep], (E - S)[keep]
    grid = np.arange(_SC_EC - 3.0, 18.0, 0.002)
    log_d = np.log(np.maximum(solver.calculate_transmission(potential, grid), 1e-300))
    weight = supply.get_occupancy(E) * np.exp(np.interp(Ez, grid, log_d))
    k_space = weight[Ez < _SC_EC].sum() / weight.sum()
    assert 0.3 < code < 0.4, "a sizeable share should sit below the edge"
    assert abs(code - k_space) < 2e-3


def test_semiconductor_default_grid_is_fine_enough_for_the_band_edges():
    """
    A band grid ends at its band edge, where the TED has a finite slope, so the
    trapezoid error in the current is ~(h / k_B T)^2 / 12: 1.1% at the old
    default of 0.01 eV. At the default now, the current must agree with a grid
    four times finer to 0.2%, and the integral of the NED with the current.
    """
    from getelec.band_structure import Semiconductor
    from getelec.electron_emitter import SemiconductorEmitter
    from getelec.electron_supply import FermiDirac

    default = Semiconductor().energy_resolution
    assert default == 0.002
    assert getelec.semiconductor_emitter().band.energy_resolution == default

    def emitter(step):
        band = Semiconductor(lower_energy_limit=10.0, top_valence=_SC_TOP,
                             fermi_level=_SC_FERMI, band_gap=_SC_GAP,
                             upper_energy_limit=18.0, energy_resolution=step,
                             electron_eff_mass=1.64, hole_eff_mass=0.59)
        return SemiconductorEmitter(SchottkyPotential(_SC_FERMI, _SC_PHI, 5.0),
                                    Noumerov(), FermiDirac(_SC_FERMI, 300.0), band)

    coarse, fine = emitter(default), emitter(0.0005)
    current = coarse.calculate_current_density()
    assert abs(current / fine.calculate_current_density() - 1) < 2e-3
    x_cb, ned_cb, x_vb, ned_vb = coarse.calculate_normal_energy_distribution()
    ned_total = np.trapezoid(ned_cb, x_cb) + np.trapezoid(ned_vb, x_vb)
    assert abs(ned_total / current - 1) < 2e-3


def test_default_grid_covers_low_fields():
    """
    SmartMetal's grid starts where the barrier is 3 nm wide. Below ~1.4 V/nm at
    room temperature that lies above the Fermi level, so the grid came out empty
    and current_density(field=1.0) raised. The grid now never starts higher than
    1 eV below E_F, and the current agrees with a far wider grid.
    """
    from getelec.band_structure import CustomMetal, SmartMetal
    from getelec.electron_emitter import MetalEmitter
    from getelec.electron_supply import LogFermiDirac

    for field in (0.5, 1.0):
        grid = SmartMetal().generate_band_structure(
            fermi_level=9.2, work_function=4.5, electric_field=field, temperature=300.0)
        assert grid.size > 50, f"grid collapsed at {field} V/nm"

    # Compared at the same grid step as the default, so this measures the grid's
    # extent and not its resolution: J at 1 V/nm is 2.5100e-18 A/cm^2 at a step
    # of 0.01 eV and converges to 2.5006e-18 at 0.0025 eV.
    step = SmartMetal().energy_resolution
    current = getelec.current_density(field=1.0, fermi_level=9.2, work_function=4.5)
    wide = MetalEmitter(SchottkyPotential(9.2, 4.5, 1.0), Noumerov(),
                        LogFermiDirac(9.2, 300.0),
                        CustomMetal(np.arange(0.05, 10.04, step)))
    assert current > 0
    assert abs(current / wide.calculate_current_density() - 1) < 1e-3


# --------------------------------------------------------------------------
# SmartSemiconductor and CustomSemiconductor
# --------------------------------------------------------------------------

def _semiconductor_emitter(band, fermi, field, temperature):
    from getelec.electron_emitter import SemiconductorEmitter
    return SemiconductorEmitter(SchottkyPotential(fermi, 4.5, field), Noumerov(),
                                LogFermiDirac(fermi, temperature), band)


@pytest.mark.parametrize("fermi, field, temperature", [
    (13.0, 5.0, 300.0),     # cold field emission, valence band leads
    (13.0, 2.0, 2000.0),    # hot, both bands emit
    (12.6, 5.0, 200.0),     # conduction band above the supply cut: nothing to emit
])
def test_smart_semiconductor_trims_the_grids_without_losing_current(fermi, field, temperature):
    """
    SmartSemiconductor keeps a subset of the Semiconductor grid points, with
    the SmartMetal criteria per band, and the current and Nottingham heat do
    not move. Compared at one grid step, so this measures the trimming only.
    """
    from getelec.band_structure import SmartSemiconductor

    band = dict(fermi_level=fermi, top_valence=12.5, band_gap=1.12,
                electron_eff_mass=1.64, hole_eff_mass=0.68, energy_resolution=0.01)
    full = _semiconductor_emitter(Semiconductor(**band), fermi, field, temperature)
    smart = _semiconductor_emitter(SmartSemiconductor(**band), fermi, field, temperature)

    full_cb, full_vb = full._base_data()[:2]
    smart_cb, smart_vb = smart._base_data()[:2]
    assert np.all(np.isin(smart_cb, full_cb)) and np.all(np.isin(smart_vb, full_vb))
    assert smart_cb.size + smart_vb.size < (full_cb.size + full_vb.size) / 2.5
    if fermi == 12.6:
        # E_C lies above the supply cut here, so the conduction band holds no
        # electrons worth counting and its grid goes away entirely.
        threshold = SmartSemiconductor(**band).supply_threshold
        cut = fermi - const.KB * temperature * np.log(np.expm1(threshold))
        assert cut < band["top_valence"] + band["band_gap"], "the case an empty band needs"
        assert smart_cb.size == 0, "a conduction band with no electrons should be dropped"

    current = full.calculate_current_density()
    assert abs(smart.calculate_current_density() / current - 1) < 1e-4
    heat = full.calculate_nottingham_heat()
    assert abs(smart.calculate_nottingham_heat() / heat - 1) < 1e-4


def test_smart_semiconductor_keeps_a_valence_band_far_below_the_fermi_level():
    """
    SmartMetal never starts its grid more than 1 eV below E_F. Applied to a
    valence band whose top lies more than 1 eV below E_F, that floor alone
    would cut the whole band away at low field, where the barrier criterion
    starts above E_V. The floor is taken from min(E_F, E_V) instead.
    """
    from getelec.band_structure import SmartSemiconductor

    fermi, field = 13.6, 1.5
    band = dict(fermi_level=fermi, top_valence=12.5, band_gap=2.0,
                electron_eff_mass=1.0, hole_eff_mass=0.68, energy_resolution=0.01)
    smart = _semiconductor_emitter(SmartSemiconductor(**band), fermi, field, 300.0)
    full = _semiconductor_emitter(Semiconductor(**band), fermi, field, 300.0)

    width_criterion = fermi + 4.5 - np.sqrt((3.0 * field) ** 2 + const.COULOMB_CONST * field)
    # SmartMetal's rule would start the grid at min(width_criterion, E_F - 1),
    # which is above E_V = 12.5 here: no valence energies at all.
    assert min(width_criterion, fermi - 1.0) > 12.5, "the case the floor exists for"
    valence = smart._base_data()[1]
    assert valence.size > 50 and valence[0] <= 11.5

    e_vb, ted_vb = full.calculate_total_energy_distribution()[2:]
    s_vb, smart_ted_vb = smart.calculate_total_energy_distribution()[2:]
    valence_current = np.trapezoid(ted_vb, e_vb)
    assert valence_current > 0
    assert abs(np.trapezoid(smart_ted_vb, s_vb) / valence_current - 1) < 1e-3


def test_custom_semiconductor_takes_its_windows_from_equation_eight():
    """
    The user's grids come back unchanged, with the window limits Semiconductor
    would give the same energies, so the physics is the same whichever built
    the grid.
    """
    from getelec.band_structure import CustomSemiconductor

    parameters = dict(top_valence=12.5, band_gap=1.12, electron_eff_mass=0.98,
                      hole_eff_mass=0.59)
    grids = Semiconductor(lower_energy_limit=10.0, upper_energy_limit=18.0,
                          energy_resolution=0.01, **parameters)
    e_cb, lower_cb, e_vb, lower_vb = grids.generate_band_structure()
    inside = e_vb <= 12.5
    custom = CustomSemiconductor(e_cb, e_vb[inside], **parameters)
    c_cb, c_lower_cb, c_vb, c_lower_vb = custom.generate_band_structure()
    assert np.array_equal(c_cb, e_cb) and np.array_equal(c_vb, e_vb[inside])
    assert np.array_equal(c_lower_cb, lower_cb)
    assert np.array_equal(c_lower_vb, lower_vb[inside])

    current = _semiconductor_emitter(grids, 13.0, 5.0, 300.0).calculate_current_density()
    assert abs(_semiconductor_emitter(custom, 13.0, 5.0, 300.0).calculate_current_density()
               / current - 1) < 1e-3


def test_custom_semiconductor_rejects_energies_outside_their_band():
    from getelec.band_structure import CustomSemiconductor

    valence = np.linspace(11.0, 12.5, 16)
    conduction = np.linspace(13.62, 14.5, 12)
    for bad in [(np.linspace(13.0, 14.5, 16), valence),           # below E_C
                (conduction, np.linspace(11.0, 12.6, 17)),        # above E_V
                (conduction[::-1], valence)]:                     # not increasing
        with pytest.raises(ValueError):
            CustomSemiconductor(*bad, top_valence=12.5, band_gap=1.12).generate_band_structure()
    empty = CustomSemiconductor(np.empty(0), valence, top_valence=12.5, band_gap=1.12)
    assert empty.generate_band_structure()[0].size == 0


# --------------------------------------------------------------------------
# Tabulated density of states
# --------------------------------------------------------------------------

DOS_FILE = ROOT / "examples" / "dos.txt"
#: The Fermi level and work function the shipped table was computed for.
DOS_FERMI, DOS_WORK = 10.268, 4.67


def _dos_band():
    from getelec.band_structure import DensityOfStatesMetal
    return DensityOfStatesMetal.from_file(DOS_FILE, DOS_FERMI, DOS_WORK)


def _free_electron_table(fermi_level=7.5, work_function=4.5):
    """A table that is exactly the free electron density of states."""
    from getelec.band_structure import DensityOfStatesMetal
    grid = np.arange(0.001, 30.0 + 0.01, 0.01)
    return grid, DensityOfStatesMetal(grid, np.sqrt(grid), fermi_level, work_function)


def test_free_electron_table_reproduces_the_free_electron_metal_exactly():
    """
    The whole weighting has to vanish when the table says nothing new.

    A ``sqrt(E)`` table is the free electron density of states, so every weight
    is exactly 1.0, the emitter takes the unweighted path, and the results have
    to be bit-identical to those of Metal -- not merely close.
    """
    from getelec.electron_emitter import MetalEmitter

    grid, table = _free_electron_table()
    assert np.all(table.get_state_weights(grid) == 1.0)

    def build(band):
        return MetalEmitter(SchottkyPotential(7.5, 4.5, 5.0), Noumerov(),
                            LogFermiDirac(7.5, 300.0), band)

    plain, dosed = build(Metal(0.001, 30.0, 0.01)), build(table)
    assert plain.calculate_current_density() == dosed.calculate_current_density()
    assert plain.calculate_nottingham_heat() == dosed.calculate_nottingham_heat()
    np.testing.assert_array_equal(plain.calculate_normal_energy_distribution()[1],
                                  dosed.calculate_normal_energy_distribution()[1])


def test_tabulated_weights_are_normalised_at_the_fermi_level():
    band = _dos_band()
    assert abs(float(band.get_state_weights(np.array([DOS_FERMI]))[0]) - 1.0) < 1e-12


def test_tabulated_table_is_cropped_to_positive_energies():
    """Zero is the bottom of the conduction band, so nothing below it survives."""
    band = _dos_band()
    raw = np.loadtxt(DOS_FILE, usecols=(0,))
    assert band.dos_energies[0] > 0.0
    assert band.dos_energies.size == int((raw > 0).sum())
    np.testing.assert_allclose(band.dos_energies, raw[raw > 0])
    # The table sets the resolution as well as the range.
    assert abs(band.energy_resolution - 0.01) < 1e-12


def test_tabulated_dos_keeps_the_two_distributions_consistent():
    """
    The standing invariant: both distributions are the same double integral.

    The weighted normal supply has no closed form, so it is integrated on the
    grid; it has to be integrated downwards from the top, because the supply
    falls through more than fifty orders of magnitude while its integral stays
    of order one. Accumulating upwards and subtracting leaves a floor of a few
    machine epsilon, which a D of order one near the barrier top turns into a
    normal distribution integrating to 945 times the current density at
    1.5 V/nm. That field is in this list for exactly that reason.
    """
    from getelec.electron_emitter import get_energy_integral

    band = _dos_band()
    for field in (1.5, 3.0, 5.0, 7.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            emitter = getelec.metal_emitter(work_function=DOS_WORK,
                                            fermi_level=DOS_FERMI,
                                            field=field, band=band)
            energies, ted = emitter.calculate_total_energy_distribution()
            normal, ned = emitter.calculate_normal_energy_distribution()
            current = emitter.calculate_current_density()
        assert ted.min() >= 0.0 and ned.min() >= 0.0
        assert abs(get_energy_integral(energies, ted) / current - 1) < 1e-12
        assert abs(get_energy_integral(normal, ned) / current - 1) < 1e-9, field


def test_tabulated_dos_reaches_the_one_line_api():
    """The sweep and the emitter must build the same integrand."""
    band = _dos_band()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emitter = getelec.metal_emitter(work_function=DOS_WORK, fermi_level=DOS_FERMI,
                                        field=5.0, band=band)
        direct = emitter.calculate_current_density()
        swept = getelec.current_density(field=5.0, work_function=DOS_WORK,
                                        fermi_level=DOS_FERMI, band=band)
    assert abs(swept / direct - 1) < 1e-12


def test_tabulated_dos_changes_the_current_it_weights():
    """A structured density of states must actually move the answer."""
    band = _dos_band()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weighted = getelec.current_density(field=5.0, work_function=DOS_WORK,
                                           fermi_level=DOS_FERMI, band=band)
        free = getelec.current_density(field=5.0, work_function=DOS_WORK,
                                       fermi_level=DOS_FERMI)
    assert 1.05 < weighted / free < 1.5


def test_tabulated_weights_follow_the_fermi_level():
    """The normalisation is at E_F, so moving E_F must invalidate the cache."""
    band = _dos_band()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emitter = getelec.metal_emitter(work_function=DOS_WORK, fermi_level=DOS_FERMI,
                                        field=5.0, band=band)
        first = emitter.calculate_current_density()
        emitter.update_params(fermi_level=DOS_FERMI + 0.2)
        moved = emitter.calculate_current_density()
        emitter.update_params(fermi_level=DOS_FERMI)
        back = emitter.calculate_current_density()
    assert moved != first and back == first


def test_tabulated_grid_warns_when_it_stops_below_the_barrier():
    """
    A table ends where its calculation ran out of bands, not where physics does.

    Field emission never notices; thermionic emission is made almost entirely of
    the states the table is missing, so the short grid has to say so.
    """
    band = _dos_band()
    for field, temperature, expected in ((5.0, 300.0, False), (7.0, 300.0, False),
                                         (2.0, 1000.0, True), (1.5, 2000.0, True)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            band.generate_band_structure(electric_field=field, temperature=temperature)
        fired = any(issubclass(entry.category, RuntimeWarning) for entry in caught)
        assert fired == expected, (field, temperature)


def test_tabulated_dos_rejects_tables_it_cannot_use():
    from getelec.band_structure import DensityOfStatesMetal

    grid = np.arange(0.001, 30.0 + 0.01, 0.01)
    values = np.sqrt(grid)
    # A table referenced to the Fermi level, the mistake worth catching loudly.
    with pytest.raises(ValueError, match="positive"):
        DensityOfStatesMetal(np.linspace(-5.0, -1.0, 50), np.ones(50), -3.0, 4.5)
    with pytest.raises(ValueError, match="outside the tabulated range"):
        DensityOfStatesMetal(grid, values, 99.0, 4.5)
    with pytest.raises(ValueError, match="zero at the Fermi level"):
        DensityOfStatesMetal(grid, np.zeros_like(grid), 7.5, 4.5)
    with pytest.raises(ValueError, match="energies but"):
        DensityOfStatesMetal(grid, np.ones(3), 7.5, 4.5)
    with pytest.raises(ValueError, match="strictly increasing"):
        DensityOfStatesMetal(grid[::-1], values, 7.5, 4.5)


def test_tabulated_dos_clips_negative_values_but_says_so():
    """Smearing can make a DFT density of states slightly negative."""
    from getelec.band_structure import DensityOfStatesMetal

    grid = np.arange(0.001, 30.0 + 0.01, 0.01)
    values = np.sqrt(grid)
    values[10] = -1e-6
    with pytest.warns(RuntimeWarning, match="negative"):
        band = DensityOfStatesMetal(grid, values, 7.5, 4.5)
    assert band.dos_values.min() == 0.0


def test_tabulated_loader_skips_a_header_without_being_told_to():
    """Density of states files carry a header and rarely mark it as a comment."""
    band = _dos_band()
    assert band.dos_values.min() >= 0.0 and band.dos_values.max() > 0.0
    assert np.all(np.isfinite(band.dos_energies))


def test_metal_emitter_syncs_a_supplied_band_to_its_arguments():
    """
    A band structure built separately must not keep a stale Fermi level.

    DensityOfStatesMetal normalises its weights at E_F, so a band built for one
    Fermi level and then handed to metal_emitter with another would weight the
    emission at the wrong energy while the barrier used the right one.
    """
    band = _dos_band()
    assert band.fermi_level == DOS_FERMI
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getelec.metal_emitter(work_function=5.1, fermi_level=9.0, field=5.0, band=band)
    assert band.fermi_level == 9.0 and band.work_function == 5.1
    # and the weights follow, so R is one at the new Fermi level
    assert abs(float(band.get_state_weights(np.array([9.0]))[0]) - 1.0) < 1e-12


# --------------------------------------------------------------------------
# Customised: a barrier from a potential the user supplies
# --------------------------------------------------------------------------

def test_customised_reproduces_a_shipped_barrier_exactly():
    """
    Wrapping the triangular formula must give the shipped barrier back.

    Bit-identical on the profile and on the current density: the class only
    routes the user's potential to the solvers, so anything else means it is
    reshaping it on the way -- clamping, interpolating or mis-scaling.
    """
    from getelec.potential_barrier import Customised, TriangularPotential

    def tilted_vacuum_level(x, fermi_level, work_function, electric_field):
        return fermi_level + work_function - electric_field * x

    shipped = TriangularPotential(7.5, 4.5, 5.0)
    custom = Customised(tilted_vacuum_level, fermi_level=7.5,
                        work_function=4.5, electric_field=5.0)
    x = np.linspace(-0.5, 3.0, 400)
    assert np.array_equal(custom.get_potential(x), shipped.get_potential(x))

    reference = getelec.current_density(field=5.0, work_function=4.5,
                                        fermi_level=7.5, barrier="triangular")
    emitter = getelec.metal_emitter(field=5.0, work_function=4.5,
                                    fermi_level=7.5, barrier=custom)
    assert emitter.calculate_current_density() == reference


def test_customised_takes_a_tabulated_potential_and_warns_beyond_it():
    """
    A table is interpolated linearly, and asking outside it must say so.

    np.interp holds the end value, so a table that stops before the barrier
    does turns the far side into a plateau -- a barrier that never falls below
    the emitted energies, whose transmission means nothing. Silence there is
    the dangerous behaviour, not the wrong number.
    """
    from getelec.potential_barrier import Customised

    positions = np.array([0.0, 1.0, 2.0])
    barrier = Customised(np.array([0.0, 5.0, 2.0]), positions)
    np.testing.assert_allclose(barrier.get_potential(np.array([-1.0, 0.5, 1.5])),
                               [0.0, 2.5, 3.5])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        held = barrier.get_potential(np.array([3.0]))
    assert held[0] == 2.0, "np.interp holds the last tabulated value"
    assert any("outside the tabulated" in str(w.message) for w in caught)


def test_customised_parameters_reach_the_callable_and_the_caches():
    """
    update_params must move the custom barrier too, and recompute.

    The parameters are handed to the callable, so a field left behind on the
    barrier while the supply moved would be exactly the inconsistency
    update_params exists to prevent -- and both the solver's profile cache and
    the emitter's result cache have to notice.
    """
    from getelec.potential_barrier import Customised

    def tilted_vacuum_level(x, fermi_level, work_function, electric_field):
        return fermi_level + work_function - electric_field * x

    barrier = Customised(tilted_vacuum_level, fermi_level=7.5,
                         work_function=4.5, electric_field=3.0)
    emitter = getelec.metal_emitter(field=3.0, work_function=4.5,
                                    fermi_level=7.5, barrier=barrier)
    low_field = emitter.calculate_current_density()

    emitter.update_params(field=5.0)
    assert barrier.electric_field == 5.0
    assert emitter.calculate_current_density() > 100 * low_field
    assert emitter.calculate_current_density() == getelec.current_density(
        field=5.0, work_function=4.5, fermi_level=7.5, barrier="triangular")


def test_customised_rejects_what_it_cannot_use():
    """
    Parameters the callable will not take, a table with no positions and a
    non-finite profile are all errors, not values quietly dropped or NaNs
    handed to the solver.
    """
    from getelec.potential_barrier import Customised

    def screened(x, screening_length):
        return np.exp(-x / screening_length)

    with pytest.raises(ValueError, match="not parameters of"):
        Customised(screened, screening_length=0.5, decay_length=0.5)
    with pytest.raises(ValueError, match="needs the positions"):
        Customised(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        Customised(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="not finite"):
        Customised(lambda x: np.full_like(x, np.nan)).get_potential(np.array([1.0]))
    with pytest.raises(ValueError, match="pointwise"):
        Customised(lambda x: np.array([1.0, 2.0])).get_potential(np.array([1.0]))


def test_customised_barriers_do_not_share_a_cached_profile():
    """
    Two Customised barriers must never collide in the solver's profile cache.

    A callable enters the cache key by its repr, which is its memory address,
    and an address freed by one function can be reused by the next -- so the
    key carries a serial number as well. The failure this guards against is
    the one that has happened here before: a second barrier silently given the
    first one's potential.
    """
    from getelec.potential_barrier import Customised, get_barrier_parameters

    solver = Noumerov()
    energies = np.linspace(5.0, 9.0, 12)
    results = []
    keys = []
    for work_function in (4.0, 4.5, 5.0):
        barrier = Customised(
            lambda x, work_function=work_function: 7.5 + work_function - 5.0 * x)
        keys.append(get_barrier_parameters(barrier))
        results.append(solver.calculate_transmission(barrier, energies))
    assert len(set(keys)) == 3
    assert not np.allclose(results[0], results[1])
    assert not np.allclose(results[1], results[2])


def test_a_customised_barrier_follows_the_emitter_temperature():
    """
    A barrier that depends on the temperature must not disagree with the supply.

    Self-consistent barriers depend on T -- the density that screens the surface
    is the finite-temperature one -- so the emitter has to set it on the barrier
    as it does the Fermi level. The failure it prevents is silent: the supply
    function moves to 1800 K while the barrier stays at whatever it was built
    with.
    """
    from getelec.potential_barrier import Customised

    def softening_barrier(x, fermi_level, work_function, electric_field, temperature):
        # Shape is beside the point; it only has to move with T.
        height = fermi_level + work_function - 1e-4 * temperature
        return height - electric_field * x

    barrier = Customised(softening_barrier, fermi_level=7.5, work_function=4.5,
                         electric_field=5.0, temperature=300.0)
    emitter = getelec.metal_emitter(field=5.0, work_function=4.5, fermi_level=7.5,
                                    temperature=1000.0, barrier=barrier)
    assert barrier.temperature == 1000.0, "metal_emitter did not sync the temperature"
    cool = emitter.calculate_current_density()

    emitter.update_params(temp=1800.0)
    assert barrier.temperature == 1800.0
    assert emitter.supply.temperature == 1800.0
    # A lower barrier and a hotter supply both raise the current; what matters
    # here is that the barrier moved at all, which a stale temperature would not.
    assert emitter.calculate_current_density() > cool

    height_at_1800 = barrier.get_potential(np.array([0.0001]))[0]
    assert abs(height_at_1800 - (12.0 - 0.18)) < 1e-3


def test_metal_emitter_sets_the_field_on_a_customised_barrier():
    """
    The field asked for is the field computed at, whatever the barrier held.

    metal_emitter set the Fermi level, work function and temperature on a
    barrier passed in, but not the field: a Customised barrier built at 3 V/nm
    stayed there under metal_emitter(field=5.0), and one built without a field
    failed with a bare TypeError. A sweep leaves the barrier at its last field,
    so the next call silently computed at that one -- 58 times the current
    asked for, at 7 V/nm against 5.
    """
    from getelec.potential_barrier import Customised

    def tilted_vacuum_level(x, fermi_level, work_function, electric_field):
        return fermi_level + work_function - electric_field * x

    expected = getelec.current_density(field=5.0, barrier="triangular")
    built_at_three = Customised(tilted_vacuum_level, fermi_level=7.5,
                                work_function=4.5, electric_field=3.0)
    emitter = getelec.metal_emitter(field=5.0, barrier=built_at_three)
    assert built_at_three.electric_field == 5.0
    assert emitter.calculate_current_density() == expected

    bare = Customised(tilted_vacuum_level)
    assert getelec.metal_emitter(field=5.0, barrier=bare).calculate_current_density() == expected

    getelec.current_density(field=[3.0, 7.0], barrier=bare)
    assert getelec.current_density(field=5.0, barrier=bare) == expected
