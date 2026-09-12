"""
Calculation backend for the GUI.

Every physics call the interface can make lives here, with no Qt anywhere in the
file. Two reasons that matters: the calculations can be tested without a display,
which is how the ones below are checked in the test suite, and the GUI itself
stays a thin layer that only moves numbers between widgets and plots.

The GUI runs :func:`calculate` on a worker thread, so nothing here may touch a
widget or assume it is on the main thread.
"""

from __future__ import annotations

import numpy as np

import getelec
# The supply carries its own k_B T, so it scales to A/(eV cm^2) with the same
# prefactor the TED uses. The flux prefactor used here before left the plotted
# N(E) a factor k_B = 8.6e-5 too small.
from getelec.electron_emitter import _SUPPLY_TO_A_PER_CM2

__all__ = ["CALCULATIONS", "MATERIALS", "SOLVERS", "get_calculations",
           "get_parameters", "calculate", "read_two_columns", "FIT_MODELS", "fit"]


def read_two_columns(path):
    """
    The first two columns of a measured-data file, as float arrays.

    ``.csv`` is comma-separated, ``.txt`` separated by any whitespace, and
    ``.xlsx`` / ``.xls`` are read from the first sheet. No header row.

    Raises
    ------
    ValueError
        For an unsupported extension, fewer than two columns, or entries that
        are not numbers (a header row, for instance).
    """
    import pandas as pd
    from pathlib import Path

    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        table = pd.read_csv(path, header=None)
    elif suffix == ".txt":
        table = pd.read_csv(path, header=None, sep=r"\s+")
    elif suffix in (".xlsx", ".xls"):
        table = pd.read_excel(path, header=None)
    else:
        raise ValueError(f"Unsupported file type {suffix!r}: use .csv, .txt, .xlsx or .xls")

    if table.shape[1] < 2:
        raise ValueError("The file must contain at least two columns")
    try:
        return (np.asarray(table.iloc[:, 0], dtype=float),
                np.asarray(table.iloc[:, 1], dtype=float))
    except ValueError:
        raise ValueError("The first two columns must contain only numbers "
                         "(remove any header row)") from None


#: Material models the interface offers. The fitting tab is metals-only, because
#: a semiconductor fit has more free parameters than an I-V curve can constrain.
MATERIALS = ("Metal", "Semiconductor")

#: Solver choices, mapped to the ``method`` argument of the top-level API.
SOLVERS = {
    "Noumerov (exact)": "noumerov",
    "Neural network (fast, ~0.1% in D)": "ml",
    "WKB (semiclassical)": "wkb",
}

# Each entry: label -> (x quantity, x unit, y label, whether x is an energy grid)
CALCULATIONS = {
    "I-F  (current vs field)":        ("field", "Field F (V/nm)",
                                       "Current density J (A/cm$^2$)", False),
    "I-T  (current vs temperature)":  ("temperature", "Temperature T (K)",
                                       "Current density J (A/cm$^2$)", False),
    "Nottingham vs field":            ("field", "Field F (V/nm)",
                                       "Nottingham heat P$_N$ (W/cm$^2$)", False),
    "TED  (total energy dist.)":      ("energy", "E - E$_F$ (eV)",
                                       "TED (A/(eV cm$^2$))", True),
    "NED  (normal energy dist.)":     ("energy", "E - E$_F$ (eV)",
                                       "NED (A/(eV cm$^2$))", True),
    "D(E) (transmission)":            ("energy", "E - E$_F$ (eV)",
                                       "Transmission D(E)", True),
    "N(E) (supply function)":         ("energy", "E - E$_F$ (eV)",
                                       "Supply N(E) (A/(eV cm$^2$))", True),
    "g(E) (D over the mass window)":  ("energy", "E - E$_F$ (eV)",
                                       r"g(E) = $\int$D dE$_z$ (eV)", True),
}

#: Calculations offered for semiconductors only. ``g(E)`` is the transmission
#: integrated over the effective-mass window of Eq. (8); a metal has no such
#: window -- its electrons carry the free mass, so D(E) is the whole story and
#: g would just be the same curve integrated over a trivial range.
SEMICONDUCTOR_ONLY = frozenset({"g(E) (D over the mass window)"})

# Parameters each calculation needs, beyond the ones swept on the x axis.
_METAL_PARAMS = ("fermi_level", "work_function", "field", "temperature")
_SEMI_EXTRA = ("band_gap", "top_valence", "electron_eff_mass", "hole_eff_mass")

#: Human labels and defaults, so the GUI does not hard-code them.
PARAMETER_INFO = {
    "fermi_level":       ("Fermi level (eV)", 7.5),
    "work_function":     ("Work function (eV)", 4.5),
    "field":             ("Electric field (V/nm)", 5.0),
    "temperature":       ("Temperature (K)", 300.0),
    "band_gap":          ("Band gap (eV)", 0.7),
    "top_valence":       ("Valence band top (eV)", 12.5),
    "electron_eff_mass": ("Electron eff. mass (m0)", 0.98),
    "hole_eff_mass":     ("Hole eff. mass (m0)", 0.59),
}


def get_calculations(material: str):
    """
    Calculation labels offered for a given material, in display order.

    Parameters
    ----------
    material : {'Metal', 'Semiconductor'}

    Returns
    -------
    tuple of str
    """
    return tuple(name for name in CALCULATIONS
                 if material == "Semiconductor" or name not in SEMICONDUCTOR_ONLY)


def get_parameters(calculation: str, material: str):
    """
    Names of the parameters a given calculation needs.

    The swept quantity is dropped, because the GUI takes it from the x-range
    boxes instead. Returned in display order.

    Parameters
    ----------
    calculation : str
        A key of :data:`CALCULATIONS`.
    material : {'Metal', 'Semiconductor'}

    Returns
    -------
    tuple of str
    """
    swept, _, _, _ = CALCULATIONS[calculation]
    names = list(_METAL_PARAMS)
    if material == "Semiconductor":
        names += list(_SEMI_EXTRA)
    if swept in names:
        names.remove(swept)
    return tuple(names)


def _build_emitter(material, params, method):
    common = dict(work_function=params["work_function"],
                  fermi_level=params["fermi_level"],
                  temperature=params["temperature"],
                  field=params["field"], method=method)
    if material == "Metal":
        return getelec.metal_emitter(**common)
    return getelec.semiconductor_emitter(
        band_gap=params["band_gap"], top_valence=params["top_valence"],
        electron_eff_mass=params["electron_eff_mass"],
        hole_eff_mass=params["hole_eff_mass"], **common)


def calculate(calculation, material, params, x_min, x_max, x_step,
            method="noumerov", progress=None):
    """
    Run one calculation and return everything the plot needs.

    Parameters
    ----------
    calculation : str
        A key of :data:`CALCULATIONS`.
    material : {'Metal', 'Semiconductor'}
    params : dict
        Values for the names returned by :func:`get_parameters`.
    x_min, x_max, x_step : float
        Range of the swept quantity. For energy-axis calculations this is
        measured relative to the Fermi level.
    method : str
        Solver, one of the values of :data:`SOLVERS`.
    progress : callable, optional
        Called as ``progress(fraction)`` with 0..1. Lets the GUI show progress
        and stay responsive; ignored if ``None``.

    Returns
    -------
    dict
        ``x``, ``y`` (or ``series`` for the two-band semiconductor results),
        ``x_label``, ``y_label``, ``title``, ``legend``.

    Raises
    ------
    ValueError
        On an empty or inverted range, or an unknown calculation.
    """
    if calculation not in CALCULATIONS:
        raise ValueError(f"Unknown calculation {calculation!r}.")
    if calculation in SEMICONDUCTOR_ONLY and material != "Semiconductor":
        raise ValueError(f"{calculation} is a semiconductor calculation: it is "
                         "the transmission integrated over the effective-mass "
                         "window, and a metal has no such window.")
    if x_step <= 0:
        raise ValueError("Step must be positive.")
    if x_max <= x_min:
        raise ValueError("Maximum must be greater than minimum.")

    swept, x_label, y_label, energy_axis = CALCULATIONS[calculation]
    x = np.arange(x_min, x_max + 0.5 * x_step, x_step)
    if x.size < 2:
        raise ValueError("Range is too narrow for the chosen step.")

    values = dict(params)
    fermi = values["fermi_level"]

    if progress:
        progress(0.05)

    # ---- swept-parameter calculations: one batched call, not a Python loop --
    if not energy_axis:
        sweep = {swept: x}
        extra = {k: v for k, v in values.items() if k != swept}
        if material == "Metal":
            if "Nottingham" in calculation:
                y = getelec.nottingham_heat(method=method, **sweep, **_metal_only(extra))
            else:
                y = getelec.current_density(method=method, **sweep, **_metal_only(extra))
            if progress:
                progress(1.0)
            return _result(x, y, x_label, y_label, calculation, _legend(values, swept))

        # The semiconductor path has no batched entry point, so it is a loop --
        # but the emitter is built once and only the swept parameter changes,
        # which keeps the cached band structure and supply alive between points.
        # The swept quantity is absent from `values` (the GUI supplies it as a
        # range), so seed it with the first point before constructing.
        seeded = dict(values)
        seeded.setdefault(swept, float(x[0]))
        emitter = _build_emitter(material, seeded, method)
        y = np.empty(x.size)
        want_heat = "Nottingham" in calculation
        for i, value in enumerate(x):
            emitter.update_params(**{swept: float(value)})
            y[i] = (emitter.calculate_nottingham_heat() if want_heat
                    else emitter.calculate_current_density())
            if progress and (i % 5 == 0):
                progress(0.05 + 0.9 * (i + 1) / x.size)
        if progress:
            progress(1.0)
        return _result(x, y, x_label, y_label, calculation, _legend(values, swept))

    # ---- energy-axis calculations ------------------------------------------
    emitter = _build_emitter(material, values, method)
    energies = fermi + x

    if material == "Metal":
        y = _metal_energy_curve(calculation, emitter, energies)
        if progress:
            progress(1.0)
        return _result(x, y, x_label, y_label, calculation, _legend(values, None))

    series = _semiconductor_energy_curves(calculation, emitter, fermi,
                                          window=(x_min, x_max))
    if progress:
        progress(1.0)
    out = _result(None, None, x_label, y_label, calculation, _legend(values, None))
    out["series"] = series
    return out


def _metal_only(values):
    """Drop semiconductor-only keys, which the metal API does not accept."""
    return {k: v for k, v in values.items() if k in _METAL_PARAMS}


def _metal_energy_curve(calculation, emitter, energies):
    if calculation.startswith("D(E)"):
        return emitter.solver.calculate_transmission(emitter.potential, energies)
    if calculation.startswith("N(E)"):
        return emitter.supply.get_log_supply(energies) * _SUPPLY_TO_A_PER_CM2
    # The distributions exist only on the emitter's grid, which ends where they
    # stop carrying current. Beyond it they are zero here, as in the current
    # integral: np.interp would otherwise repeat the last value across the rest
    # of the range, a flat line that on a log axis reads as the distribution
    # levelling off.
    if "TED" in calculation:
        grid, ted = emitter.calculate_total_energy_distribution()
        return np.interp(energies, grid, ted, left=0.0, right=0.0)
    grid, ned = emitter.calculate_normal_energy_distribution()
    return np.interp(energies, grid, ned, left=0.0, right=0.0)


def _semiconductor_energy_curves(calculation, emitter, fermi, window=None):
    """
    Conduction and valence band curves, each on its own energy grid.

    They are returned separately rather than summed: the bands have different
    grids and, more to the point, seeing which one carries the current is the
    reason to run a semiconductor calculation at all.

    ``window`` is the requested ``(min, max)`` in ``E - E_F``. Unlike a metal,
    the grids here are fixed by the band structure -- the band edges and the
    window limits of Eq. (8) -- because the distributions have to be computed
    on them. So the range selects which part of the result to return; it cannot
    move where it is evaluated, and the step does not apply.
    """
    def clipped(curves):
        if window is None:
            return curves
        low, high = window
        return [(name, x[(x >= low) & (x <= high)], y[(x >= low) & (x <= high)])
                for name, x, y in curves]

    if "TED" in calculation:
        e_cb, ted_cb, e_vb, ted_vb = emitter.calculate_total_energy_distribution()
        return clipped([("conduction band", e_cb - fermi, ted_cb),
                        ("valence band", e_vb - fermi, ted_vb)])
    if "NED" in calculation:
        e_cb, ned_cb, e_vb, ned_vb = emitter.calculate_normal_energy_distribution()
        return clipped([("conduction band", e_cb - fermi, ned_cb),
                        ("valence band", e_vb - fermi, ned_vb)])
    if calculation.startswith("D(E)"):
        e_cb, t_cb, e_vb, t_vb = emitter.calculate_transmission_coefficient()
        return clipped([("conduction band", e_cb - fermi, t_cb),
                        ("valence band", e_vb - fermi, t_vb)])
    if calculation.startswith("g(E)"):
        # D(E) does not move when the effective masses do -- the barrier holds
        # no mass. The masses set the limits this integral runs between, so
        # this is the curve that responds to them. Read out of the same
        # calculation the TED is built from, not recomputed alongside it.
        e_cb, g_cb, e_vb, g_vb = emitter.calculate_window_integrated_transmission()
        return clipped([("conduction band", e_cb - fermi, g_cb),
                        ("valence band", e_vb - fermi, g_vb)])
    e_cb, n_cb, e_vb, n_vb = emitter.calculate_supply_function()
    return clipped([("conduction band", e_cb - fermi, n_cb * _SUPPLY_TO_A_PER_CM2),
                    ("valence band", e_vb - fermi, n_vb * _SUPPLY_TO_A_PER_CM2)])


def _legend(values, swept):
    parts = []
    for name in ("fermi_level", "work_function", "field", "temperature"):
        if name == swept or name not in values:
            continue
        label = {"fermi_level": "E$_F$", "work_function": "$\\phi$",
                 "field": "F", "temperature": "T"}[name]
        parts.append(f"{label}={values[name]:g}")
    return ", ".join(parts)


def _result(x, y, x_label, y_label, title, legend):
    return {"x": x, "y": y, "series": None, "x_label": x_label,
            "y_label": y_label, "title": title, "legend": legend}


# --------------------------------------------------------------------------
# Fitting measured data
# --------------------------------------------------------------------------

#: Fit models the interface offers. For each: the x and y axis labels, then its
#: parameters in display order as (name, label, default, (lower, upper), typical
#: step for the optimiser, free by default).
#:
#: The defaults leave free only what that kind of data can determine, and fix
#: the rest at a typed value: one I-V curve cannot separate the work function
#: from the field factor, and a TED's shape cannot separate it from the field.
#: Freeing more is allowed; the fit then reports how correlated they are.
FIT_MODELS = {
    "I-V": ("Voltage (V)", "Current (nA)", (
        ("gamma", "γ, field per volt (1/nm)", 0.005, (1e-6, 1.0), 1e-3, True),
        ("area", "Emission area (nm²)", 1000.0, (1e-6, 1e14), 1.0, True),
        ("fermi_level", "Fermi level (eV)", 7.5, (0.5, 20.0), 0.5, False),
        ("work_function", "Work function (eV)", 4.5, (2.0, 8.0), 0.5, False),
        ("temperature", "Temperature (K)", 300.0, (1.0, 3000.0), 50.0, False),
    )),
    "I-T": ("Temperature T (K)", "Current (nA)", (
        ("field", "Field (V/nm)", 5.0, (0.2, 20.0), 0.5, True),
        ("area", "Emission area (nm²)", 1000.0, (1e-6, 1e14), 1.0, True),
        ("fermi_level", "Fermi level (eV)", 7.5, (0.5, 20.0), 0.5, False),
        ("work_function", "Work function (eV)", 4.5, (2.0, 8.0), 0.5, False),
    )),
    "TED": ("E - E$_F$ (eV)", "Normalized counts (a.u.)", (
        ("field", "Field (V/nm)", 5.0, (0.5, 20.0), 0.5, True),
        ("temperature", "Temperature (K)", 300.0, (1.0, 3000.0), 50.0, True),
        ("offset", "Energy offset (meV)", 0.0, (-500.0, 500.0), 10.0, True),
        ("work_function", "Work function (eV)", 4.5, (2.0, 8.0), 0.5, False),
        ("fermi_level", "Fermi level (eV)", 7.5, (0.5, 20.0), 0.5, False),
    )),
}

#: Symbols and units for the parameter summary.
_FIT_SYMBOLS = {
    "gamma": ("γ", "nm⁻¹"), "area": ("area", "nm²"), "fermi_level": ("E_F", "eV"),
    "work_function": ("φ", "eV"), "temperature": ("T", "K"),
    "field": ("F", "V/nm"), "offset": ("offset", "meV"),
}

#: Free parameters correlated more strongly than this are flagged: their
#: individual standard errors then say little.
CORRELATION_WARNING = 0.95


def _fit_function(model, x, start):
    """The model as a function of a dict of every parameter, on the grid ``x``."""
    if model in ("I-V", "I-T"):
        def evaluate(p):
            field = x * p["gamma"] if model == "I-V" else p["field"]
            temperature = p["temperature"] if model == "I-V" else x
            j = getelec.current_density(field=field, temperature=temperature,
                                        work_function=p["work_function"],
                                        fermi_level=p["fermi_level"], fast=True)
            current = np.asarray(j, dtype=float) * p["area"] * 1e-14 * 1e9  # nm^2 -> cm^2, A -> nA
            return np.clip(current, 1e-300, None)
        return evaluate

    from getelec.band_structure import CustomMetal
    from getelec.electron_emitter import MetalEmitter
    from getelec.electron_supply import FermiDirac
    from getelec.potential_barrier import SchottkyPotential
    from getelec.transmission_solver import Noumerov

    emitter = MetalEmitter(
        SchottkyPotential(start["fermi_level"], start["work_function"], start["field"]),
        Noumerov(), FermiDirac(start["fermi_level"], start["temperature"]),
        CustomMetal(x + start["fermi_level"]))

    def evaluate(p):
        emitter.update_params(field=p["field"], temp=p["temperature"],
                              wf=p["work_function"], fermi=p["fermi_level"])
        # The measured axis is E - E_F; the analyser offset shifts it.
        emitter.band = CustomMetal(x + p["fermi_level"] + p["offset"] * 1e-3)
        _, y = emitter.calculate_total_energy_distribution()
        return y / np.max(y)
    return evaluate


def fit(model, x, y, values, free):
    """
    Fit a model to measured data, with any subset of its parameters free.

    Parameters
    ----------
    model : str
        A key of :data:`FIT_MODELS`.
    x, y : array_like
        The measured data: voltage (V) and current (nA) for I-V, temperature
        (K) and current (nA) for I-T, E - E_F (eV) and counts for TED.
    values : dict
        A value for every parameter of the model: the starting guess if it is
        free, the value used if it is fixed.
    free : iterable of str
        Names of the parameters to fit. May be empty, which only evaluates the
        model at ``values``.

    Returns
    -------
    dict
        ``x``, ``data`` and ``fit`` to plot; ``values``, every parameter, fitted
        or fixed; ``errors``, standard errors of the free ones; ``correlated``,
        pairs of free parameters above :data:`CORRELATION_WARNING`; ``misfit``;
        ``summary``, lines of text for the plot; ``x_label``, ``y_label``.

    Raises
    ------
    ValueError
        If a value is outside its allowed range, or the data are unusable.
    """
    from scipy.optimize import curve_fit

    x_label, y_label, specs = FIT_MODELS[model]
    wanted = set(free)
    free = [name for name, *_ in specs if name in wanted]
    values = {name: float(values[name]) for name, *_ in specs}
    for name, label, _, (lower, upper), _, _ in specs:
        if not lower <= values[name] <= upper:
            raise ValueError(f"{label}: {values[name]:g} is outside the allowed "
                             f"range [{lower:g}, {upper:g}]")

    # In increasing x. The TED model builds its energy grid from x and
    # integrates the transmission along it, which a file listed from high to
    # low energy would run backwards -- the fit then failed at its first step.
    order = np.argsort(np.asarray(x, dtype=float), kind="stable")
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    logarithmic = model in ("I-V", "I-T")
    if logarithmic:
        keep = y > 0                       # fitted in log current
        x, data = x[keep], y[keep]
        target = np.log(data)
    else:
        if not np.max(y) > 0:
            raise ValueError("The counts must contain positive values")
        data = y / np.max(y)
        target = data
    if x.size < len(free) + 2:
        raise ValueError(f"{x.size} usable data points are not enough "
                         f"for {len(free)} free parameters")

    evaluate = _fit_function(model, x, values)
    bounds = {name: limits for name, _, _, limits, _, _ in specs}
    steps = {name: step for name, _, _, _, step, _ in specs}

    # The area spans decades, so it is fitted as log10(area).
    def internal(name, v):
        return np.log10(v) if name == "area" else v

    def external(name, v):
        return 10.0 ** v if name == "area" else v

    def with_free(vector):
        p = dict(values)
        p.update({name: external(name, v) for name, v in zip(free, vector)})
        return p

    def model_target(_, *vector):
        out = evaluate(with_free(vector))
        return np.log(out) if logarithmic else out

    errors, correlated = {}, []
    if free:
        popt, pcov = curve_fit(
            model_target, x, target,
            p0=[internal(n, values[n]) for n in free],
            bounds=([internal(n, bounds[n][0]) for n in free],
                    [internal(n, bounds[n][1]) for n in free]),
            x_scale=[steps[n] for n in free])
        values = with_free(popt)
        with np.errstate(invalid="ignore", divide="ignore"):
            sigma = np.sqrt(np.diag(pcov))
            correlation = pcov / np.outer(sigma, sigma)
        for k, name in enumerate(free):
            # To first order, the error of 10**u is 10**u ln(10) sigma_u.
            errors[name] = (values[name] * np.log(10.0) * sigma[k] if name == "area"
                            else float(sigma[k]))
        for i in range(len(free)):
            for j in range(i + 1, len(free)):
                r = correlation[i, j]
                if np.isfinite(r) and abs(r) > CORRELATION_WARNING:
                    correlated.append((free[i], free[j], float(r)))

    fitted = evaluate(values)
    misfit = float(np.std(np.log(data / fitted)) if logarithmic
                   else np.std(data - fitted))
    return {"x": x, "data": data, "fit": fitted, "values": values,
            "errors": errors, "free": free, "correlated": correlated,
            "misfit": misfit, "x_label": x_label, "y_label": y_label,
            "summary": _fit_summary(model, values, errors, free, correlated, misfit)}


def _fit_summary(model, values, errors, free, correlated, misfit):
    """Lines of text describing a fit, one parameter per line."""
    lines = []
    for name, *_ in FIT_MODELS[model][2]:
        symbol, unit = _FIT_SYMBOLS[name]
        value = values[name]
        text = f"{value:+.1f}" if name == "offset" else f"{value:.4g}"
        if name in free:
            error = errors.get(name, np.nan)
            text += f" ± {error:.2g}" if np.isfinite(error) else " ± n/a"
            lines.append(f"{symbol} = {text} {unit}")
        else:
            lines.append(f"{symbol} = {text} {unit}  (fixed)")
        if name == "gamma":
            lines.append(f"R ≈ 1/(5γ) = {1 / (5 * value):.1f} nm")
    lines.append(f"misfit = {misfit:.3g}" + (" in ln I" if model in ("I-V", "I-T") else ""))
    for a, b, r in correlated:
        lines.append(f"{_FIT_SYMBOLS[a][0]}–{_FIT_SYMBOLS[b][0]} correlation {r:+.3f}")
    return lines
