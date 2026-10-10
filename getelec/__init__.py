"""
GETELEC -- General Tool for Electron Emission Calculations.

Thermal-field electron emission current density and Nottingham heat for metallic
and semiconducting emitters.

Quick start
-----------
The common case is one line, and any argument accepts an array:

    >>> import getelec                                        # doctest: +SKIP
    >>> getelec.current_density(field=5.0, work_function=4.5) # doctest: +SKIP
    >>> getelec.current_density(field=np.arange(3, 8, 0.5))   # doctest: +SKIP

A sweep is solved as a single batched call, not a Python loop, so it is
substantially faster than calling the scalar version repeatedly.

For distributions, or to reuse one configuration across many calls:

    >>> emitter = getelec.metal_emitter(work_function=4.5,    # doctest: +SKIP
    ...                                 fermi_level=7.5, temperature=300)
    >>> emitter.calculate_current_density()                   # doctest: +SKIP
    >>> energies, ted = emitter.calculate_total_energy_distribution()  # doctest: +SKIP

Full control
------------
The convenience layer just assembles four interchangeable components. Build them
yourself whenever you need something the shortcuts do not expose:

    potential_barrier    -- SchottkyPotential, SmallRadiiPotential, TriangularPotential,
                            Customised (a potential of your own)
    band_structure       -- Metal, SmartMetal, Semiconductor, SmartSemiconductor,
                            CustomMetal, CustomSemiconductor,
                            DensityOfStatesMetal (a tabulated DOS)
    electron_supply      -- FermiDirac, LogFermiDirac
    transmission_solver  -- Noumerov (exact), NoumerovReference (written out
                            plainly, for checking), NeuralSolver (learned)
    transmission_solutions -- WKB (fast, semiclassical), AiryTriangular

    >>> from getelec.electron_emitter import MetalEmitter     # doctest: +SKIP
    >>> MetalEmitter(potential, solver, supply, band)         # doctest: +SKIP

Units
-----
Inputs are eV, nm, fs, K, and V/nm for the field. Current density comes back in
A/cm^2, Nottingham heat P_N in W/cm^2, and distributions in A/(eV cm^2).

Citing
------
If you use GETELEC, please cite the software, S. Barranco Cárceles,
A. Kyritsakis and A. Ayari, GETELEC: General Tool for Electron Emission
Calculations, Zenodo, https://doi.org/10.5281/zenodo.23093209, and the papers
listed in README.md and CITATION.cff.
"""

from __future__ import annotations

import copy

import numpy as np

from getelec import constants
from getelec import potential_barrier
from getelec import band_structure
from getelec import electron_supply
from getelec import transmission_solver
from getelec import transmission_solutions
from getelec.transmission_solver import NeuralSolver
from getelec.transmission_solutions import WKB, AiryTriangular
from getelec import electron_emitter
from getelec.electron_emitter import (MetalEmitter, SemiconductorEmitter,
                                      get_energy_integral)

__version__ = "3.1.1"

__all__ = [
    "metal_emitter", "semiconductor_emitter",
    "current_density", "nottingham_heat",
    "MetalEmitter", "SemiconductorEmitter",
    "NeuralSolver",
    "WKB", "AiryTriangular", "transmission_solutions",
    "transmission_coefficient", "supply_function", "watermark",
    "constants", "potential_barrier", "band_structure",
    "electron_supply", "transmission_solver", "electron_emitter",
    "__version__",
]

_SOLVERS = {"noumerov": "Noumerov", "wkb": "WKB",
            "ml": "NeuralSolver", "airy": "AiryTriangular"}


def _build_solver(method, fermi_level, work_function, field, solver_kwargs):
    key = str(method).lower()
    if key not in _SOLVERS:
        raise ValueError(
            f"Unknown method {method!r}. Choose from {sorted(set(_SOLVERS))}."
        )
    name = _SOLVERS[key]
    if name == "NeuralSolver":
        return transmission_solver.NeuralSolver(**solver_kwargs)
    if name == "WKB":
        return transmission_solutions.WKB(
            fermi_level=fermi_level, work_function=work_function,
            electric_field=float(np.ravel(field)[0]), **solver_kwargs)
    if name == "AiryTriangular":
        return transmission_solutions.AiryTriangular(
            fermi_level=fermi_level, work_function=work_function,
            electric_field=float(np.ravel(field)[0]))
    cls = getattr(transmission_solver, name)
    if cls is transmission_solver.Noumerov:
        fast = solver_kwargs.pop("fast", False)
        reference = solver_kwargs.pop("reference", False)
        if fast and reference:
            raise ValueError("fast=True and reference=True select two different "
                             "Noumerov solvers; choose one.")
        if fast:
            return cls.fast(**solver_kwargs)
        if reference:
            return cls.reference(**solver_kwargs)
        return cls(**solver_kwargs)
    solver_kwargs.pop("fast", None)
    return cls(fermi_level=fermi_level, work_function=work_function,
               electric_field=float(np.ravel(field)[0]), **solver_kwargs)


def _build_barrier(barrier, fermi_level, work_function, field, radius, gamma):
    # A barrier object passed through as it is, so that one built by hand --
    # potential_barrier.Customised, above all -- reaches the emitter without
    # having to assemble the other three components as well. metal_emitter
    # then sets the Fermi level, work function, field and temperature on it,
    # as on every component; semiconductor_emitter deliberately does not.
    if isinstance(barrier, potential_barrier.Barrier):
        return barrier
    key = str(barrier).lower()
    if key in ("schottky", "planar"):
        return potential_barrier.SchottkyPotential(fermi_level, work_function, field)
    if key in ("sharp_tip", "small_radii", "curved"):
        return potential_barrier.SmallRadiiPotential(
            fermi_level, work_function, field, radius=radius, gamma=gamma
        )
    if key == "triangular":
        return potential_barrier.TriangularPotential(fermi_level, work_function, field)
    raise ValueError(
        f"Unknown barrier {barrier!r}. Choose 'schottky', 'sharp_tip' or "
        f"'triangular', or pass a potential_barrier.Barrier -- "
        f"potential_barrier.Customised wraps a potential of your own."
    )


def metal_emitter(work_function=4.5, fermi_level=7.5, temperature=300.0,
                  field=5.0, barrier="schottky", method="noumerov",
                  radius=20.0, gamma=100.0, energy_resolution=0.01,
                  band=None, **solver_kwargs) -> MetalEmitter:
    """
    Build a metal emitter with sensible defaults.

    Parameters
    ----------
    work_function : float
        Surface work function, eV.
    fermi_level : float
        Fermi level measured from the bottom of the conduction band, eV. Set
        here once; it is propagated to the barrier and the supply function
        together, which is the pairing most easily got wrong by hand.
    temperature : float
        Emitter temperature, K.
    field : float
        Local electric field at the surface, V/nm.
    barrier : {'schottky', 'sharp_tip', 'triangular'} or potential_barrier.Barrier
        Planar image-charge barrier, one corrected for tip curvature, or the
        triangular barrier with no image charge, for which ``method='airy'``
        is exact. A barrier object is used as it is given, most usefully
        :class:`~getelec.potential_barrier.Customised`, which wraps a potential
        of your own; ``fermi_level``, ``work_function``, ``field`` and
        ``temperature`` are then set on it, as on every other component.
    method : {'noumerov', 'ml', 'wkb', 'airy'}
        ``'noumerov'`` integrates the Schrodinger equation, with no
        approximation beyond discretisation. ``'ml'`` uses :class:`~getelec.transmission_solver.NeuralSolver`, a
        trained network with models shipped for both barriers. It pays off for
        barriers with several parameters; for the planar barrier it is a
        worked example, and :mod:`getelec.training` trains one for your own
        conditions. ``'wkb'`` is semiclassical: cheapest, but accurate only to
        a factor of order unity deep in the tunnelling regime.
    fast : bool, default False
        Use :meth:`Noumerov.fast`, which is within a small factor of the WKB cost
        while keeping current density and energy distributions converged. See its
        docstring for the exact accuracy contract.
    reference : bool, default False
        Use :class:`~getelec.transmission_solver.NoumerovReference`, the same
        integration written out plainly, one energy at a time, to check a
        result: about half a second for a metal current density. It takes
        ``x_start``, ``x_end``, ``h`` and ``seed``. Cannot be combined with
        ``fast``.
    radius, gamma : float
        Tip radius (nm) and field enhancement, used only by ``'sharp_tip'``.
        The barrier is valid for radii of 20-1000 nm.
    energy_resolution : float
        Energy grid spacing, eV. Ignored when ``band`` is given, since a band
        structure carries its own.
    band : band_structure.BandStructure, optional
        The band structure to use instead of the default
        :class:`~getelec.band_structure.SmartMetal`. Pass a
        :class:`~getelec.band_structure.DensityOfStatesMetal` to weight the
        emission by a tabulated density of states; ``fermi_level`` and
        ``work_function`` are then set on it as well, so its normalisation
        follows a sweep.
    **solver_kwargs
        Passed to the solver, e.g. ``h=5e-4`` to refine the spatial grid.

    Returns
    -------
    MetalEmitter

    Examples
    --------
    >>> em = metal_emitter(work_function=4.5, temperature=1000)   # doctest: +SKIP
    >>> em.calculate_current_density()                            # doctest: +SKIP
    >>> em.update_params(field=6.0)                               # doctest: +SKIP
    """
    barrier_obj = _build_barrier(barrier, fermi_level, work_function, field, radius, gamma)
    solver = _build_solver(method, fermi_level, work_function, field, solver_kwargs)
    supply = electron_supply.LogFermiDirac(fermi_level, temperature)
    if band is None:
        band = band_structure.SmartMetal(energy_resolution=energy_resolution)
    emitter = MetalEmitter(barrier_obj, solver, supply, band)
    # A band structure built separately carries its own Fermi level and work
    # function -- DensityOfStatesMetal normalises its state weights at the
    # former -- and they would otherwise stay at whatever they were when it was
    # built while the barrier used the arguments given here. Setting them
    # through update_params puts the same values on every component, which is
    # what it is for. A no-op for the band structures that hold neither.
    # The field as well, for a Customised barrier passed in: it would
    # otherwise keep the field it was built with -- or the last one of an
    # earlier sweep, which moves the barrier it is given -- and the current
    # would be computed at a field other than the one asked for.
    emitter.update_params(fermi_level=fermi_level, work_function=work_function,
                          field=field, temperature=temperature)
    return emitter


def semiconductor_emitter(work_function=4.5, fermi_level=13.0, temperature=300.0,
                          field=5.0, top_valence=12.5, band_gap=1.12,
                          electron_eff_mass=1.64, hole_eff_mass=0.68,
                          barrier="schottky", method="noumerov",
                          radius=20.0, gamma=100.0, energy_resolution=0.002,
                          **solver_kwargs) -> SemiconductorEmitter:
    """
    Build a semiconductor emitter with sensible defaults.

    Defaults describe silicon. Parameters shared with :func:`metal_emitter`
    behave the same way, with one exception: a barrier object passed as
    ``barrier`` is used exactly as it is, and none of the Fermi level, work
    function, field or temperature is set on it. Give a
    :class:`~getelec.potential_barrier.Customised` barrier those values
    yourself.

    Parameters
    ----------
    top_valence : float
        Energy of the valence band maximum, eV, measured from the same zero as
        ``fermi_level``.
    band_gap : float
        Band gap, eV.
    electron_eff_mass, hole_eff_mass : float
        Effective masses relative to the free electron mass.
    energy_resolution : float, default 0.002
        Energy grid spacing, eV. Five times finer than for a metal, because
        each band's grid ends at its band edge, where the TED has a finite
        slope: the trapezoid error in the current is then ~(h / k_B T)^2 / 12,
        about 1.1% at 0.01 eV and 300 K, and 0.05% at 0.002 eV. A metal's grid
        ends in tails where the TED is flat, which is why 0.01 suffices there.

    Returns
    -------
    SemiconductorEmitter
    """
    barrier_obj = _build_barrier(barrier, fermi_level, work_function, field, radius, gamma)
    solver = _build_solver(method, fermi_level, work_function, field, solver_kwargs)
    supply = electron_supply.LogFermiDirac(fermi_level, temperature)
    band = band_structure.Semiconductor(
        top_valence=top_valence, fermi_level=fermi_level, band_gap=band_gap,
        electron_eff_mass=electron_eff_mass, hole_eff_mass=hole_eff_mass,
        energy_resolution=energy_resolution,
    )
    return SemiconductorEmitter(barrier_obj, solver, supply, band)


def _broadcast(**kwargs):
    """Broadcast scalar/array parameters into a list of per-point parameter dicts."""
    names = list(kwargs)
    arrays = np.broadcast_arrays(*[np.asarray(kwargs[n], dtype=float) for n in names])
    shape = arrays[0].shape
    flat = [a.ravel() for a in arrays]
    points = [dict(zip(names, vals)) for vals in zip(*flat)]
    return points, shape


def _sweep_metal(emitter, points):
    """
    Evaluate a metal emitter at many parameter points in one batched solve.

    All energy grids and barriers are assembled first, then handed to the solver
    as a single call. That keeps every worker thread busy and pays the kernel
    launch cost once instead of once per point.
    """
    barriers, grids, weights = [], [], []
    for pt in points:
        emitter.update_params(**pt)
        energies = emitter.band.generate_band_structure(
            fermi_level=emitter.potential.fermi_level,
            work_function=emitter.potential.work_function,
            electric_field=emitter.potential.electric_field,
            temperature=emitter.supply.temperature,
        )
        grids.append(energies)
        # The band structure's state weights, one everywhere unless it holds a
        # tabulated density of states. Carried per point because the
        # normalisation follows the Fermi level, which a sweep may vary.
        weights.append(emitter.band.get_state_weights(energies))
        # Snapshot: the shared barrier object is mutated on the next iteration.
        barriers.append(copy.copy(emitter.potential))

    transmissions = emitter.solver.calculate_transmission_batch(barriers, grids)
    return grids, weights, transmissions


def current_density(field=5.0, work_function=4.5, fermi_level=7.5,
                    temperature=300.0, **kwargs):
    """
    Emitted current density in A/cm^2.

    Any of ``field``, ``work_function``, ``fermi_level`` and ``temperature`` may
    be arrays; they are broadcast against each other and the whole set is solved
    in one batched pass.

    Parameters
    ----------
    field : float or array_like
        Local surface field, V/nm.
    work_function : float or array_like
        Work function, eV.
    fermi_level : float or array_like
        Fermi level, eV.
    temperature : float or array_like
        Temperature, K.
    **kwargs
        Forwarded to :func:`metal_emitter` (``barrier``, ``method``, ``h``, ...).

    Returns
    -------
    float or np.ndarray
        Scalar if every input was scalar, otherwise the broadcast shape.

    Examples
    --------
    >>> current_density(field=5.0)                                # doctest: +SKIP
    >>> current_density(field=np.linspace(3, 8, 50))              # doctest: +SKIP
    >>> current_density(field=5.0, temperature=[300, 800, 1500])  # doctest: +SKIP
    """
    points, shape = _broadcast(field=field, work_function=work_function,
                               fermi_level=fermi_level, temperature=temperature)
    emitter = metal_emitter(work_function=points[0]["work_function"],
                            fermi_level=points[0]["fermi_level"],
                            temperature=points[0]["temperature"],
                            field=points[0]["field"], **kwargs)

    grids, weights, transmissions = _sweep_metal(emitter, points)
    # The integral of the TED, built by the same function as
    # MetalEmitter.calculate_current_density so the two paths cannot disagree.
    out = np.array([
        get_energy_integral(e, electron_emitter.get_metal_total_energy_distribution(
            e, t, emitter.supply, w))
        for e, w, t in zip(grids, weights, transmissions)
    ])
    return float(out[0]) if shape == () else out.reshape(shape)


def nottingham_heat(field=5.0, work_function=4.5, fermi_level=7.5,
                    temperature=300.0, **kwargs):
    """
    Nottingham heat P_N in W/cm^2: negative when the surface heats (electrons
    leave from below E_F and are replaced by hotter ones), positive when it
    cools.

    Broadcasts over its arguments exactly like :func:`current_density`.

    Returns
    -------
    float or np.ndarray
    """
    points, shape = _broadcast(field=field, work_function=work_function,
                               fermi_level=fermi_level, temperature=temperature)
    emitter = metal_emitter(work_function=points[0]["work_function"],
                            fermi_level=points[0]["fermi_level"],
                            temperature=points[0]["temperature"],
                            field=points[0]["field"], **kwargs)

    grids, weights, transmissions = _sweep_metal(emitter, points)
    out = np.empty(len(points))
    for i, (e, w, t, pt) in enumerate(zip(grids, weights, transmissions, points)):
        # f(E) times the cumulative transmission -- not the log supply, which
        # belongs to the normal energy distribution. See Supply.get_occupancy.
        ted = electron_emitter.get_metal_total_energy_distribution(
            e, t, emitter.supply, w)
        moment = get_energy_integral(e, (e - pt["fermi_level"]) * ted)
        out[i] = moment
    return float(out[0]) if shape == () else out.reshape(shape)


def get_flux_constant() -> float:
    """
    Conversion from the supply-weighted flux to A/cm^2.

    Applied to the supply function, which carries its own ``k_B T`` -- so this
    must not be multiplied by the temperature again. See
    ``electron_supply.LogFermiDirac.get_supply``.
    """
    return electron_emitter._SUPPLY_TO_A_PER_CM2


def transmission_coefficient(field=5.0, work_function=4.5, fermi_level=7.5,
                             temperature=300.0, energies=None, **kwargs):
    """
    Transmission coefficient D(E), the bare tunnelling probability.

    Parameters
    ----------
    field, work_function, fermi_level, temperature : float
        Barrier and emitter parameters. ``temperature`` only sets the default
        energy grid; D itself does not depend on it.
    energies : array_like, optional
        Energies at which to evaluate D, in eV. Defaults to the emitter's own
        grid, which is chosen to cover everything that carries current.
    **kwargs
        Forwarded to :func:`metal_emitter` (``method``, ``barrier``, ...).

    Returns
    -------
    energies, transmission : np.ndarray

    Examples
    --------
    >>> E, D = transmission_coefficient(field=5.0)               # doctest: +SKIP
    >>> E, D = transmission_coefficient(field=5.0, method="ml")  # doctest: +SKIP
    """
    emitter = metal_emitter(work_function=work_function, fermi_level=fermi_level,
                            temperature=temperature, field=field, **kwargs)
    if energies is None:
        return emitter.calculate_transmission_coefficient()
    energies = np.asarray(energies, dtype=float)
    return energies, emitter.solver.calculate_transmission(emitter.potential, energies)


def supply_function(field=5.0, work_function=4.5, fermi_level=7.5,
                    temperature=300.0, energies=None, **kwargs):
    """
    Electron supply function N(E).

    Electrons arriving at the barrier per unit energy. Independent of the field
    and of the barrier shape; it depends only on temperature and Fermi level.

    Returns
    -------
    energies, supply : np.ndarray
    """
    emitter = metal_emitter(work_function=work_function, fermi_level=fermi_level,
                            temperature=temperature, field=field, **kwargs)
    if energies is None:
        return emitter.calculate_supply_function()
    energies = np.asarray(energies, dtype=float)
    return energies, emitter.supply.get_log_supply(energies)


#: Standard attribution placed on figures. Kept in one place so the GUI, the
#: examples and anything you write all mark plots identically.
WATERMARK_TEXT = "Calculated with GETELEC"


def watermark(target=None, text=None, alpha=0.45, size=8):
    """
    Put the GETELEC attribution in the bottom-right corner of a plot.

    Parameters
    ----------
    target : matplotlib Axes or Figure, optional
        Where to draw. Defaults to the current axes. Passing a Figure places the
        mark once for the whole figure rather than once per panel.
    text : str, optional
        Overrides :data:`WATERMARK_TEXT`.
    alpha, size : float
        Opacity and font size.

    Returns
    -------
    matplotlib.text.Text

    Examples
    --------
    >>> import matplotlib.pyplot as plt          # doctest: +SKIP
    >>> fig, ax = plt.subplots()                 # doctest: +SKIP
    >>> ax.plot(fields, current)                 # doctest: +SKIP
    >>> getelec.watermark(fig)                   # doctest: +SKIP
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    label = WATERMARK_TEXT if text is None else text
    if target is None:
        target = plt.gca()

    if isinstance(target, Figure):
        return target.text(0.995, 0.005, label, ha="right", va="bottom",
                           fontsize=size, color="gray", alpha=alpha)
    return target.text(0.995, 0.01, label, transform=target.transAxes,
                       ha="right", va="bottom", fontsize=size,
                       color="gray", alpha=alpha)
