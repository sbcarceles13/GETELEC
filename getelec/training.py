"""
Train a neural transmission model for :class:`~getelec.transmission_solver.NeuralSolver`.

**When this is worth it.** A network's cost does not grow with the number of
barrier parameters, while a sweep of exact solves or a table of them does. So it
pays off for barriers with several parameters -- tip radius and field
enhancement, say, as in the shipped curved-tip model. For the plain Schottky
barrier it is several times faster than ``Noumerov.fast()``, but that
calculation was already cheap; GETELEC ships a Schottky model as a worked
example of the workflow.

The recipe is the same for any barrier, and each step is there for a reason:

1. **Ground truth from the exact solve.** Every sample is a Noumerov result.
2. **Learn the residual, not the transmission.** ``ln D`` spans hundreds of
   e-folds; the residual ``R = ln D_exact - ln D_kemble`` against a
   semiclassical reference for the *same* barrier stays of order unity. A 1%
   target in ``D`` is then 0.01 in ``R``, and none of the exponential has to be
   learned at all.
3. **Physics-derived features.** The scaled barrier field ``k_e F / h**2`` is 1
   exactly at the barrier top, wherever the other parameters put it, which
   hands the network the coordinate in which the function is smooth. See
   :func:`~getelec.transmission_solver.get_schottky_features`.
4. **Validate against the exact solve.** The worst-case error printed at the
   end is the number to trust, not the training loss.

Fitting uses scikit-learn (``pip install scikit-learn``). Using the trained model
needs only NumPy: :class:`~getelec.transmission_solver.NeuralModel` stores plain
arrays, with no pickle.

Reproduce the shipped models::

    python -m getelec.training                          # neural_schottky.npz
    python -m getelec.training --barrier small_radii    # neural_small_radii.npz

Train one for your own barrier::

    from getelec import training
    from getelec.transmission_solver import NeuralSolver

    def make_barrier(total_height, field, radius):
        return MyBarrier(total_height - 4.5, 4.5, field, radius=radius)

    domain = {"barrier_height": (-3, 7), "total_height": (9.5, 23.5),
              "field": (1.5, 15), "radius": (20, 1000)}
    samples = training.generate_samples(make_barrier, domain, n_barriers=4000,
                                        log_uniform=("radius",))
    estimator = training.fit(samples, features=my_features)
    model = training.to_model(estimator, samples, features=my_features)
    print(training.validate(model, make_barrier))
    model.save("my_barrier.npz")

    solver = NeuralSolver(model="my_barrier.npz")

``my_features(barrier_height, total_height, field, radius)`` returns an
``(n, k)`` array. Start from the Schottky features and add terms in your
parameters -- :func:`~getelec.transmission_solver.get_small_radii_features` shows how.
"""

from __future__ import annotations

import argparse
import time
import warnings
from datetime import date

import numpy as np

from getelec.potential_barrier import RADIUS_VALIDITY
from getelec.transmission_solver import (FEATURES, REFERENCES, NeuralModel,
                                         NeuralSolver, Noumerov,
                                         register_features)

__all__ = ["SCHOTTKY_DOMAIN", "SMALL_RADII_DOMAIN", "generate_samples", "fit",
           "to_model", "validate", "ground_truth_solver", "reference_solver",
           "make_schottky", "make_small_radii"]

#: The domain the shipped Schottky model covers: ``W = E_F + phi`` and field
#: span field emission from low-work-function coatings to refractory metals,
#: and barrier heights ``h = W - E`` from 3 eV above the barrier top down to
#: where D no longer moves any integral.
SCHOTTKY_DOMAIN = {
    "barrier_height": (-3.0, 7.0),    # h = W - E, eV
    "total_height": (9.5, 23.5),      # W = E_F + phi, eV
    "field": (1.5, 15.0),             # V/nm
}

#: The domain the shipped sharp-tip model covers: the Schottky domain, the
#: radii over which ``SmallRadiiPotential`` is valid, and field enhancements
#: from 1 to 200. Radius and gamma are sampled uniformly in their logarithm.
SMALL_RADII_DOMAIN = dict(SCHOTTKY_DOMAIN, radius=RADIUS_VALIDITY, gamma=(1.0, 200.0))
_SMALL_RADII_LOG_UNIFORM = ("radius", "gamma")
#: ``SmallRadiiPotential`` has a pole in x for gamma < 1, so the padding stops there.
_SMALL_RADII_LIMITS = {"gamma": (1.0, np.inf)}

_BASE = ("barrier_height", "total_height", "field")


def make_schottky(total_height, field):
    """A Schottky-Nordheim barrier with the given ``W = E_F + phi`` and field."""
    from getelec.potential_barrier import SchottkyPotential
    # phi is arbitrary: only W = E_F + phi enters the potential.
    return SchottkyPotential(fermi_level=total_height - 4.5, work_function=4.5,
                             electric_field=field)


def make_small_radii(total_height, field, radius, gamma):
    """
    A ``SmallRadiiPotential`` with the given ``W = E_F + phi``, field, radius and
    gamma.

    Training samples a little beyond the valid radii (see :func:`_padded`),
    where the class warns; the barrier is well defined there and the model is
    certified only inside the domain, so the warning is silenced here.
    """
    from getelec.potential_barrier import SmallRadiiPotential
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="radius = ", category=UserWarning)
        return SmallRadiiPotential(fermi_level=total_height - 4.5, work_function=4.5,
                                   electric_field=field, radius=radius, gamma=gamma)


def ground_truth_solver():
    """
    The exact solver used to generate training data.

    Accurate to ~1e-4 in ``ln D``, well below what the network resolves.
    ``auto_domain`` matters: at 1.5 V/nm the barrier reaches tens of nm, and a
    truncated barrier gives a meaningless answer rather than a slightly wrong
    one.
    """
    return Noumerov(x_metal=-0.01, x_vac_plus=2.0, h=2e-3, seed="wkb",
                    auto_domain=True)


def reference_solver():
    """A finer exact solver for validation, independent of the training data."""
    return Noumerov(x_metal=-0.01, x_vac_plus=3.0, h=5e-4, seed="wkb")


def _parameter_names(domain):
    missing = [name for name in _BASE if name not in domain]
    if missing:
        raise ValueError(f"domain is missing {missing}")
    return [name for name in domain if name not in _BASE]


def _draw(rng, domain, log_uniform):
    values = {}
    for name, (lo, hi) in domain.items():
        if name == "barrier_height":
            continue
        if name in log_uniform:
            values[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            values[name] = float(rng.uniform(lo, hi))
    return values


def _padded(domain, pad, log_uniform, limits=None):
    """
    The sampling ranges: each axis widened by ``pad`` of its span on both sides.

    A fitted function is least accurate at the edges of its data, so the model
    is trained a little beyond the domain it is then certified for -- the same
    reason the table this solver replaced was built past its advertised range.
    Log-uniform axes are widened in the logarithm, which keeps them positive;
    the field is kept positive too. ``limits`` clips an axis where the barrier
    stops being defined: ``SmallRadiiPotential`` has a pole in x for gamma < 1.
    """
    ranges = {}
    for name, (lo, hi) in domain.items():
        if name in log_uniform:
            ratio = (hi / lo) ** pad
            ranges[name] = (lo / ratio, hi * ratio)
        else:
            span = (hi - lo) * pad
            ranges[name] = (lo - span, hi + span)
    lo_f, hi_f = ranges["field"]
    ranges["field"] = (max(lo_f, 0.5 * domain["field"][0]), hi_f)
    for name, (lo, hi) in (limits or {}).items():
        a, b = ranges[name]
        ranges[name] = (max(a, lo), min(b, hi))
    return ranges


def generate_samples(make_barrier, domain, n_barriers, heights_per_barrier=64,
                     reference="kemble_numeric", log_uniform=(), pad=0.1, seed=0,
                     solver=None, progress=True, limits=None):
    """
    Exact residuals at random points of the domain.

    Parameters
    ----------
    make_barrier : callable
        ``make_barrier(total_height, field, **parameters)`` returning a barrier.
        Every barrier it makes must be of the same class.
    domain : dict
        ``name -> (lo, hi)``. Must contain ``barrier_height``, ``total_height``
        and ``field``; any other entry is a barrier parameter, passed to
        ``make_barrier`` by name and to the features in the order given here.
    n_barriers : int
        Random barriers to solve. Each is one batched Noumerov call.
    heights_per_barrier : int
        Random barrier heights sampled per barrier.
    reference : str
        Key into :data:`~getelec.transmission_solver.REFERENCES`.
        ``'kemble_numeric'`` works for any barrier with a single maximum;
        ``'kemble_schottky'`` is the closed form, for the planar barrier only.
    log_uniform : sequence of str
        Parameters to sample uniformly in their logarithm, for ranges spanning
        orders of magnitude.
    pad : float, default 0.1
        Sample this fraction of each span beyond the domain on both sides. The
        model is still certified only for ``domain``; see :func:`_padded`.
    seed : int
    solver : optional
        Exact solver. Defaults to :func:`ground_truth_solver`.
    progress : bool
    limits : dict, optional
        ``name -> (lo, hi)``: hard bounds the padding may not cross, for axes
        where the barrier stops being defined.

    Returns
    -------
    dict
        ``inputs`` with columns ``(barrier_height, total_height, field,
        *parameters)``, ``residual``, and what :func:`to_model` needs to
        describe the model: ``domain`` (unpadded), ``parameters``,
        ``reference``, ``barrier``.
    """
    parameters = _parameter_names(domain)
    if reference not in REFERENCES:
        raise ValueError(f"unknown reference {reference!r}; choose from {sorted(REFERENCES)}")
    solver = solver or ground_truth_solver()
    reference_fn = REFERENCES[reference]
    rng = np.random.default_rng(seed)
    sampling = _padded(domain, pad, log_uniform, limits)
    lo_h, hi_h = sampling["barrier_height"]

    rows, targets, barrier_class = [], [], None
    start = time.perf_counter()
    for i in range(n_barriers):
        values = _draw(rng, sampling, log_uniform)
        W, F = values["total_height"], values["field"]
        barrier = make_barrier(W, F, **{p: values[p] for p in parameters})
        name = type(barrier).__name__
        if barrier_class is None:
            barrier_class = name
        elif name != barrier_class:
            raise TypeError(f"make_barrier returned both {barrier_class} and {name}")

        heights = np.sort(rng.uniform(lo_h, hi_h, heights_per_barrier))
        energies = W - heights
        physical = energies > 0.05          # E <= 0 is not a state of the metal
        heights, energies = heights[physical], energies[physical]
        if energies.size == 0:
            continue
        residual = (solver.calculate_log_transmission(barrier, energies)
                    - reference_fn(barrier, energies))
        finite = np.isfinite(residual)
        n = int(finite.sum())
        if n == 0:
            continue
        rows.append(np.column_stack(
            [heights[finite], np.full(n, W), np.full(n, F)]
            + [np.full(n, values[p]) for p in parameters]))
        targets.append(residual[finite])
        if progress and (i + 1) % max(1, n_barriers // 10) == 0:
            print(f"  solved {i + 1}/{n_barriers} barriers "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)

    return {"inputs": np.concatenate(rows), "residual": np.concatenate(targets),
            "domain": dict(domain), "parameters": parameters,
            "reference": reference, "barrier": barrier_class}


def _feature_function(features):
    if callable(features):
        return features
    if features not in FEATURES:
        raise KeyError(f"unknown features {features!r}; registered: {sorted(FEATURES)}")
    return FEATURES[features]


def _feature_matrix(features, inputs):
    columns = [inputs[:, i] for i in range(inputs.shape[1])]
    return _feature_function(features)(*columns)


def fit(samples, features, hidden_layers=(72, 72, 72), max_iter=3000, seed=0,
        estimator=None):
    """
    Fit a network to the residuals.

    L-BFGS on the full batch with tanh units. The target is a smooth,
    noise-free function, so a quasi-Newton method converges far further than
    stochastic gradient descent, and tanh keeps the fitted function smooth
    where ReLU would leave facets.

    Parameters
    ----------
    samples : dict
        From :func:`generate_samples`.
    features : str or callable
        Name in :data:`~getelec.transmission_solver.FEATURES`, or the function.
    hidden_layers : tuple of int
    max_iter : int
        L-BFGS iterations. Stopping at the limit is normal, not a failure;
        judge the result by :func:`validate`.
    seed : int
    estimator : sklearn.neural_network.MLPRegressor, optional
        Continue training this one instead of starting afresh.

    Returns
    -------
    sklearn.neural_network.MLPRegressor
    """
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor
    except ImportError as error:
        raise ImportError(
            "Training needs scikit-learn: pip install scikit-learn. "
            "Using a trained model does not.") from error

    X = _feature_matrix(features, samples["inputs"])
    if estimator is None:
        estimator = MLPRegressor(hidden_layer_sizes=tuple(hidden_layers),
                                 activation="tanh", solver="lbfgs", alpha=1e-9,
                                 max_iter=max_iter, tol=1e-14, random_state=seed,
                                 warm_start=True)
    else:
        estimator.set_params(max_iter=max_iter, warm_start=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(X, samples["residual"])
    return estimator


def to_model(estimator, samples, features, metadata=None):
    """
    Package a fitted estimator as a :class:`NeuralModel`.

    Only the weights and biases are kept. A callable ``features`` is
    registered under its ``__name__``, which is what the model records;
    register it again under that name in any session that loads the model.
    """
    if callable(features):
        name = features.__name__
        register_features(name, features)
    else:
        name = str(features)
    return NeuralModel(estimator.coefs_, estimator.intercepts_, features=name,
                       reference=samples["reference"], barrier=samples["barrier"],
                       domain=samples["domain"], parameters=samples["parameters"],
                       metadata=metadata)


def validate(model, make_barrier, n_barriers=30, heights_per_barrier=120,
             log_uniform=(), seed=21, solver=None):
    """
    Pointwise error in D against a finer exact solve, on fresh random barriers.

    Returns
    -------
    dict
        ``median``, ``p90``, ``p99`` and ``worst`` relative error in ``D``,
        over energies inside the trained band.
    """
    solver = solver or reference_solver()
    learned = NeuralSolver(model=model, fallback="error", energy_nodes=0)
    rng = np.random.default_rng(seed)
    lo_h, hi_h = model.domain["barrier_height"]
    errors = []
    for _ in range(n_barriers):
        values = _draw(rng, model.domain, log_uniform)
        barrier = make_barrier(values["total_height"], values["field"],
                               **{p: values[p] for p in model.parameters})
        energies = values["total_height"] - np.linspace(lo_h, hi_h, heights_per_barrier)
        energies = energies[energies > 0.05]
        exact = solver.calculate_log_transmission(barrier, energies)
        finite = np.isfinite(exact)
        got = learned.calculate_log_transmission(barrier, energies[finite])
        errors.append(np.abs(np.expm1(got - exact[finite])))
    errors = np.concatenate(errors)
    return {"median": float(np.median(errors)),
            "p90": float(np.percentile(errors, 90)),
            "p99": float(np.percentile(errors, 99)),
            "worst": float(errors.max())}


#: What each shipped model is trained with: barrier, domain, features, reference,
#: log-uniform axes, hard limits, and the defaults of the command line.
_RECIPES = {
    "schottky": dict(make=make_schottky, domain=SCHOTTKY_DOMAIN, features="schottky",
                     reference="kemble_schottky", log_uniform=(), limits=None,
                     out="neural_schottky.npz", barriers=2000),
    "small_radii": dict(make=make_small_radii, domain=SMALL_RADII_DOMAIN,
                        features="small_radii", reference="kemble_numeric",
                        log_uniform=_SMALL_RADII_LOG_UNIFORM, limits=_SMALL_RADII_LIMITS,
                        out="neural_small_radii.npz", barriers=3000),
}


def main(argv=None):
    """Reproduce a shipped model: the Schottky one (default) or the sharp-tip one."""
    parser = argparse.ArgumentParser(
        prog="python -m getelec.training",
        description="Train a shipped transmission model.")
    parser.add_argument("--barrier", choices=sorted(_RECIPES), default="schottky",
                        help="which shipped model to reproduce")
    parser.add_argument("--out", default=None,
                        help="output file (default: the shipped file name)")
    parser.add_argument("--barriers", type=int, default=None,
                        help="random barriers to solve (default: 2000 Schottky, "
                             "3000 sharp tip)")
    parser.add_argument("--heights", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--hidden", default="64,64,64",
                        help="hidden layer sizes, comma separated")
    parser.add_argument("--pad", type=float, default=0.1,
                        help="fraction of each span sampled beyond the domain")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    recipe = _RECIPES[args.barrier]
    out = args.out or recipe["out"]
    n_barriers = args.barriers or recipe["barriers"]
    hidden = tuple(int(n) for n in args.hidden.split(","))

    print(f"1. Solving {n_barriers} {args.barrier} barriers exactly")
    samples = generate_samples(recipe["make"], recipe["domain"], n_barriers,
                               heights_per_barrier=args.heights,
                               reference=recipe["reference"], pad=args.pad,
                               seed=args.seed, log_uniform=recipe["log_uniform"],
                               limits=recipe["limits"])
    residual = samples["residual"]
    print(f"   {residual.size} samples, residual in "
          f"[{residual.min():.2f}, {residual.max():.2f}]")

    print(f"2. Fitting {hidden} for {args.iterations} L-BFGS iterations")
    start = time.perf_counter()
    estimator = fit(samples, recipe["features"], hidden_layers=hidden,
                    max_iter=args.iterations, seed=args.seed)
    elapsed = time.perf_counter() - start
    print(f"   {elapsed:.0f} s, final loss {estimator.loss_:.3e}")

    import getelec
    model = to_model(estimator, samples, recipe["features"], metadata={
        "trained": date.today().isoformat(),
        "getelec_version": getelec.__version__,
        "samples": int(residual.size), "barriers": n_barriers,
        "hidden_layers": list(hidden), "iterations": args.iterations,
        "pad": args.pad, "seed": args.seed,
        "log_uniform": list(recipe["log_uniform"]),
    })

    # Saved before validating, so a long fit is never lost to a failure after it.
    model.save(out)
    print(f"3. Saved {out}")

    print("4. Validating against a finer exact solve")
    report = validate(model, recipe["make"], log_uniform=recipe["log_uniform"])
    print("   relative error in D: " + ", ".join(
        f"{k} {v:.3%}" for k, v in report.items()))
    model.metadata["validation"] = report
    model.save(out)
    return model


if __name__ == "__main__":
    main()
