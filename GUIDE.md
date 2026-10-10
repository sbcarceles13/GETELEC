# GETELEC: how it works

A guide to the physics the code implements and how the modules fit together.
For installation see [INSTALL.md](INSTALL.md); for what changed in this
release see [CHANGES.md](CHANGES.md).

---

## Contents

- [The physical problem](#the-physical-problem)
- [Architecture](#architecture)
- [Module by module](#module-by-module)
  - [constants](#constants)
  - [potential_barrier](#potential_barrier)
  - [band_structure](#band_structure)
  - [electron_supply](#electron_supply)
  - [transmission_solver](#transmission_solver)
  - [transmission_solutions](#transmission_solutions)
  - [electron_emitter](#electron_emitter)
  - [_kernels](#_kernels)
- [Semiconductors: the two energy distributions](#semiconductors-the-two-energy-distributions)
  - [Seeing the mass dependence](#seeing-the-mass-dependence)
- [The graphical interface](#the-graphical-interface)
- [Choosing a solver](#choosing-a-solver)
- [Accuracy and how to check it](#accuracy-and-how-to-check-it)
- [Units](#units)
- [Citing GETELEC](#citing-getelec)

---

## The physical problem

An electron inside a metal sits in a potential well. Outside is vacuum. Apply a
strong electric field to the surface and the vacuum barrier is bent into a
triangle, thin enough that electrons tunnel through it. That is **field
emission**. Heat the metal instead and electrons go *over* the barrier: that is
**thermionic emission**. In between — strong field and high temperature at once
— is the **thermal-field** regime, which is where most real devices operate and
where neither limiting formula works.

GETELEC computes the emitted current density by integrating over electron
energy:

```
J = e ∫ N(E) D(E) dE
```

Two factors, carrying quite different physics:

**D(E), the transmission coefficient.** The probability that an electron of
energy `E` gets through the barrier. Set entirely by the barrier shape — work
function, applied field, tip curvature. It does not depend on temperature. It is
the expensive part: computing it means solving the Schrödinger equation, and it
spans tens or hundreds of orders of magnitude across the band.

**N(E), the supply function.** How many electrons arrive at the surface per unit
energy. Set entirely by the occupancy — Fermi level and temperature. It does not
depend on the barrier, and it is cheap: a Fermi–Dirac expression.

You can pull both out separately (see
section 4 of [`examples/intro_to_getelec.ipynb`](examples/intro_to_getelec.ipynb)),
and doing so is usually the fastest way to understand a result.

The same integral, weighted by `E − E_F`, gives the **Nottingham heat** `P_N`,
in W/cm² — the power per unit area the emitted electrons carry away, measured
from the Fermi level. Emit mostly from below `E_F` and
the tip heats, because each electron is replaced by a hotter one at `E_F`, and
`P_N` is negative; emit from above and it cools, and `P_N` is positive.

### The barrier

The planar barrier is the Schottky–Nordheim form:

```
V(x) = E_F + φ − F·x − k_e/(4x)
```

The three terms are the vacuum level, the applied field tilting it down, and the
image-charge attraction between the escaping electron and the surface it left.
The image term is what rounds off the triangle and lowers the barrier maximum to

```
V_max = E_F + φ − √(k_e F)
```

which is the Schottky effect. For a sharp tip the field is not uniform and
`SmallRadiiPotential` adds curvature corrections. Without the image term the
barrier is the triangle of Fowler and Nordheim, `TriangularPotential`, the one
barrier with an exact solution.

---

## Architecture

Four interchangeable pieces, assembled into an emitter:

```
   potential_barrier ──┐
   band_structure    ──┤
   electron_supply   ──┼──►  electron_emitter  ──►  J, Nottingham heat,
   transmission_solver ┘                             D(E), N(E), TED, NED
```

Each is an abstract base class with several implementations, so you can swap the
barrier model or the solver without touching anything else. The convenience
functions in `getelec/__init__.py` just assemble sensible defaults.

```python
from getelec.potential_barrier import SchottkyPotential
from getelec.band_structure import SmartMetal
from getelec.electron_supply import LogFermiDirac
from getelec.transmission_solver import Noumerov
from getelec.electron_emitter import MetalEmitter

emitter = MetalEmitter(
    SchottkyPotential(fermi_level=7.5, work_function=4.5, electric_field=5.0),
    Noumerov(),
    LogFermiDirac(fermi_level=7.5, temperature=300.0),
    SmartMetal(),
)
```

---

## Module by module

### `constants`

Physical constants in the code's internal unit system (eV, nm, fs). The two that
appear most often:

- `HBAR2_2M` — ħ²/2m, the kinetic prefactor.
- `COULOMB_CONST` — `k_e`, setting the strength of the image-charge term.

### `potential_barrier`

Returns `V(x)` on a spatial grid.

| Class | Use |
|---|---|
| `SchottkyPotential` | Planar image-charge barrier. The standard choice. |
| `SmallRadiiPotential` | Adds tip-curvature corrections, via `radius` and `gamma`. Valid for tip radii of 20–1000 nm; see its accuracy below. |
| `TriangularPotential` | No image charge: `V = E_F + φ − F·x`. Taller than the Schottky barrier, so it emits less; `AiryTriangular` gives its transmission exactly. |
| `Customised` | A potential of your own, as a function `V(x)` or a table. See [Your own barrier](#your-own-barrier-customised) below. |

The triangular barrier jumps by `E_F + φ` at the surface, which Noumerov
resolves only to first order in its step — by an amount set by where the surface
falls between grid points, so a node landing on it would cost 1.3% in current
density at the default `h = 1e-3` nm. The solver's grid puts the surface halfway
between two nodes, which cancels that term: the current density comes out within
1e-5 of the exact `method="airy"` result, and the transmission within 6e-5.

At fixed apex field a sharp tip emits *less* than a planar surface, not more:
away from the apex the field falls off as `F/γ`, so the barrier is wider. The
planar result is the `R → ∞` limit.

Across its valid range the potential is close to the planar Schottky–Nordheim
barrier: at `R = 20 nm` it lies within 0.3 eV of it across the barrier at the
Fermi level (under 7% of φ) at 3–8 V/nm, for `γ = 100` and `E_F + φ = 12 eV`.
The current depends exponentially on the barrier, so it is not close: at
`R = 20 nm` it is 0.22, 0.57 and 0.79 times the planar current density at 3, 5
and 8 V/nm, and at 50 nm 0.56, 0.80 and 0.91 times. The difference falls as
1/R: at 1000 nm, the top of the valid range, it is 2.8%, 1.1% and 0.6% at 3, 5
and 7 V/nm, and above that the planar barrier gives the same current density.

**Accuracy of the sharp-tip barrier.** On the axis, `SmallRadiiPotential` is
exact to second order in `x/R`: its curvature is the one Laplace's equation
requires at an apex of radius `R`. The third-order term is set by the shape of
the emitter away from its apex, which `radius` and `gamma` do not describe —
written as `c₃ F x³/R²`, it has `c₃ = 4/3` for an ellipsoidal tip of any aspect
ratio, about 1.05 for a hemisphere on a cylindrical post at `γ = 100`, and
`γ/(γ − 1)` in this barrier. The current density is sensitive to that term, most
of all near the outer turning point of the barrier. Measured against the exact
electrostatics of the two shapes (`γ = 100`, `φ = 4.5 eV`, 300 K), this barrier
under-predicts the current density by:

| `R` | tip shape | 3 V/nm | 5 V/nm | 7 V/nm |
|---|---|---|---|---|
| 20 nm | ellipsoidal | 3.3% | 0.7% | 0.26% |
| 20 nm | hemisphere on a post | 0.45% | 0.09% | 0.03% |
| 10 nm | ellipsoidal | 16% | 3.1% | 1.1% |
| 10 nm | hemisphere on a post | 2.3% | 0.43% | 0.15% |

The difference grows roughly as `φ^(7/2)/(F³R²)`: at `φ = 5.5 eV` and 20 nm the
ellipsoidal figures are 7.0% and 1.4% at 3 and 5 V/nm. This is why the class
warns below 20 nm. In the Schottky and thermionic regime the barrier top matters
most, and the difference is smaller: at 1500 K and 20 nm it is under 0.5% for
both shapes at 0.1–0.3 V/nm; at 0.03 V/nm it is 1.2% for an ellipsoidal tip and
0.17% for a post.

#### Your own barrier: `Customised`

`Customised` takes a potential worked out elsewhere — a formula of your own, a
self-consistent or DFT profile, the output of a field solver — and gives it to
the same solvers, emitters and distributions as the shipped barriers, without
subclassing anything. Pass it wherever a barrier name goes:

```python
import numpy as np
import getelec
from getelec import constants
from getelec.potential_barrier import Customised

def image_scaled(x, fermi_level, work_function, electric_field, image_strength):
    W = fermi_level + work_function
    V = W - electric_field * x - image_strength * constants.COULOMB_CONST / (4 * x)
    # Cut the image term off where it first crosses zero, as the shipped barriers do.
    V[x < (W - np.sqrt(W**2 - image_strength * constants.COULOMB_CONST
                       * electric_field)) / (2 * electric_field)] = 0.0
    return V

barrier = Customised(image_scaled, image_strength=0.5)
getelec.current_density(field=np.linspace(3, 7, 9), barrier=barrier)
```

`image_strength = 1` gives back the Schottky–Nordheim barrier and `0` the
triangular one, both to the last digit of the current density. Checking that a
barrier of your own reproduces a known one at some limit is the first thing to
do with it.

The conventions are those of the shipped barriers:

- **Units and zero.** `x` in nm, `V` in eV, measured from the bottom of the
  conduction band: 0 inside the metal and `E_F + φ` at the surface, before the
  field and the image charge pull it down. `V` is only evaluated at `x > 0` and
  held at 0 inside, unless `zero_inside=False`.
- **A function** is called with an array of positions and returns an array of
  the same shape.
- **Parameters.** A function's parameters named `fermi_level`,
  `work_function`, `electric_field` and `temperature` are filled in by the
  emitter: `metal_emitter`, `update_params` and the sweeps of
  `current_density` set them on the barrier as on every other component, so a
  field sweep moves your barrier too. They also size the solver's integration
  domain. Any other named parameter, like `image_strength`, is given to
  `Customised` and passed at every call; a name the function does not take is
  an error. `semiconductor_emitter` sets none of the four: give them to
  `Customised` yourself.
- **Divergences.** Nothing is clamped for you. An image term has to be cut off
  near the surface; `get_image_cutoff` is where the shipped barriers do it.
- **The function must depend only on its arguments.** The solver caches the
  profile for each set of parameters.

**A table** is `Customised(values, positions)`, positions in nm and strictly
increasing, interpolated linearly. It is one potential: it does not follow the
field. It has to reach as far as the solver's integration domain, which is
several nanometres and longer at low field: beyond its ends the end values are
held, and the emitter warns, naming the range it needed. Linear interpolation
converges slowly where the potential bends sharply, near the surface. The
Schottky–Nordheim barrier at 5 V/nm, tabulated every 0.1, 0.05 and 0.02 nm,
gives a current density 18%, 5% and 0.8% high. Halve the step until the current
stops moving.

**Solvers.** Only the solvers that integrate on a grid read the potential: the
default `Noumerov`, `fast=True` and `reference=True`. `method="wkb"` and
`method="airy"` compute from `E_F`, `φ` and `F` alone, so they return the
Schottky–Nordheim and triangular answers, not yours. `method="ml"` has no
trained model for it and falls back to the exact solver, with a warning.

Section 15 of the introduction notebook works through both forms.

### `band_structure`

Chooses which energies to sample. The integrand spans enormous ranges, so
picking the grid well matters more than it sounds.

| Class | Use |
|---|---|
| `SmartMetal` | Picks limits automatically from the field, work function and temperature. The default. |
| `Metal` | Fixed limits you supply. |
| `Semiconductor` | Separate conduction and valence band grids. |
| `SmartSemiconductor` | `Semiconductor`'s grids, trimmed band by band with `SmartMetal`'s two criteria. |
| `CustomMetal` | Your own grid. |
| `CustomSemiconductor` | Your own grid for each band; the windows of Eq. (8) follow from the band edges and effective masses. |
| `DensityOfStatesMetal` | A tabulated density of states, which sets the grid and weights the emission. |

The two automatic classes are built for the currents an emitter actually
produces. Their cuts hold the current density and the Nottingham heat within 1%
of a far wider grid wherever the current density reaches 10⁻¹² A/cm² — 1 pA
from 1 cm² — checked for the planar, sharp-tip and triangular barriers at work
functions of 2.5–6 eV, fields of 0.2–12 V/nm and 300–3000 K. Below 1 pA/cm² the
cuts start to show, and the smaller the current the more it costs to reach it:
the same 1% at arbitrarily small currents would need a supply threshold around
10⁻⁴⁰ and the energies to match. If those currents are what you need, pass a
smaller `supply_threshold` and a larger `barrier_width`, or set the limits
yourself with `Metal`, `CustomMetal` or `CustomSemiconductor`.

#### A tabulated density of states

Every class above treats the metal as a free electron gas, whose density of
states goes as `sqrt(E)` above the bottom of the conduction band.
`DensityOfStatesMetal` instead reads `g(E)` from a two-column table and weights
each total energy by how many states it holds relative to that reference,

    R(E) = [g(E) / sqrt(E)] / [g(E_F) / sqrt(E_F)] ,

normalised so `R(E_F) = 1`, so the current density stays comparable to the free
electron one and the table acts as a shape correction rather than an unknown
overall scale. That matters for a projected density of states, which has no
absolute normalisation of its own. Energies at or below zero are dropped — zero
is the bottom of the conduction band — and what is left sets both the range and
the resolution of the grid.

```python
from getelec.band_structure import DensityOfStatesMetal

band = DensityOfStatesMetal.from_file("dos.txt", fermi_level=10.268,
                                      work_function=4.67)
emitter = getelec.metal_emitter(work_function=4.67, fermi_level=10.268,
                                field=5.0, band=band)
```

The total energy distribution is multiplied by `R`, and the normal one is
rebuilt from the same double integral over total and normal energy, so both
still integrate to the current density. `examples/dos.txt` and section 14 of the
example notebook show it end to end.

Two limits are worth knowing. First, the density of states is a proxy for the
right quantity rather than the right quantity: in the supply integral the group
velocity cancels the Jacobian, so what belongs here is the number of
forward-propagating channels at energy E — the constant-energy surface projected
on the surface plane — while the density of states weights states by one over
that same velocity. Flat bands therefore have their contribution overestimated.
Second, a table ends where the calculation that produced it ran out of bands,
and its density of states falls to zero there for that reason and not a physical
one. Field emission is unaffected, drawing from within a fraction of an
electronvolt of `E_F`; thermionic emission draws from the top of the barrier,
which sits close to the end of a table, so the emitter warns when the grid clears
the barrier maximum by less than `8 k_B T`.

### `electron_supply`

Returns `N(E)`.

| Class | Use |
|---|---|
| `LogFermiDirac` | Supply integrated over transverse momentum, with `k_B T` factored out. The default; pairs with the `× temperature` in the current integral. |
| `FermiDirac` | Plain occupancy `1/(1 + exp((E−E_F)/k_B T))`. |

Both are evaluated with `scipy.special.expit` and `np.logaddexp`, which are
exact across the whole real line — the hand-written stable forms they replaced
lost up to 5.7% near the crossover.

### `transmission_solver`

The expensive part: `D(E)`.

There are two: **`NoumerovFast`** keeps only psi at the last two grid points,
which is all the transmission needs, and is what everything uses by default.
**`Noumerov`** adds `calculate_psi(barrier, energies)` and
`calculate_probability_current(...)`, keeping the whole wavefunction for when that is the
answer — a charge density, a current, the decaying tail inside the barrier. A
third, **`NoumerovReference`**, writes the same integration out plainly, to
check the other two against (see below).

**`Noumerov`** integrates the 1D Schrödinger equation directly, with no
approximation beyond discretisation. Noumerov's method has a
local truncation error of O(h⁶), which makes it one of the most accurate
methods for this equation, and it needs only a single three-term recurrence per
grid point, which makes it fast. It integrates right-to-left
from vacuum into the metal, seeds an outgoing wave, and projects the result onto
incoming and outgoing waves inside the metal; the flux ratio gives `D`.

Three settings are worth knowing:

- `seed` — `"wkb"` (default) or `"plane"`. The boundary condition, and the thing
  that actually limits accuracy. A plane wave is exact only where the potential
  is flat, which it is not, so it needs ~10 nm of vacuum to converge. The WKB
  form `ψ = k^(−1/2) exp(i∫k dx)` is accurate much closer in, which is what lets
  the domain shrink.
- `auto_domain` — `True` by default. Measures the outer classical turning point
  and sizes the grid to enclose the whole barrier. At 1.2 V/nm the barrier is
  4.5 nm wide at the bottom of the default energy grid and 10 nm at the bottom
  of the band, against the 3 nm a fixed domain would assume; integrating only
  part of a barrier does not give a slightly wrong answer, it gives a
  meaningless one.
- `Noumerov.fast()` — a preset trading pointwise accuracy in the far tail for
  speed. It takes a tenth of the integration steps, which on a grid of a few
  hundred energies makes the transmission 1.5 to 3 times faster, depending on
  the machine: fixed costs are most of such a call. The current density stays
  within 5e-5 of a converged reference and the distributions within 2e-3.

**`NoumerovReference`** is the same integration written out plainly, for
checking the solvers above and debugging a result. For one energy,
`calculate_noumerov_reference(barrier, energy)` works from full arrays and
returns the transmission together with the grid, the potential, the wavefunction
and the wavevector in the metal. The solver calls it once per energy. Nothing is
cached, batched or interpolated, and the seed is a plane wave unless
`seed="wkb"`. Select it with `reference=True` or `Noumerov.reference()`. A metal
current density takes about half a second, against a few milliseconds for
`Noumerov`. For the planar barrier at 3–7 V/nm the two agree to 3e-5 in J and
P_N, and to 5e-4 in D(E) with the plane-wave seed (3e-4 with WKB).

Its grid is fixed, `np.arange(x_end, x_start - h, -h)`, from 20 nm in the vacuum
to −1 nm in the metal by default, and two things follow from that:

- It has to enclose the barrier. Where `x_end` is still inside it, the seed is
  not a travelling wave and the transmission comes out as 0; the solver warns.
- The defaults put a node on the surface. A continuous barrier does not notice,
  but the triangular one jumps there, which costs about 1% in J.
  `x_end=20.0005` puts the surface halfway between two nodes and recovers the
  exact result to 2e-5.

`calculate_log_transmission()` returns `ln D` and should be preferred whenever
the answer may be tiny: the probability path underflows to a hard zero around
`D ~ 10⁻³⁰⁸`, while the log path keeps giving digits to about `ln D = −1400`.

**`WKB`** is the semiclassical approximation. The Gamow exponent

```
G = √(2m)/ħ ∫ √(V(x) − E) dx
```

is integrated between the turning points by Gauss–Chebyshev quadrature of the
second kind — which carries exactly the √ endpoint weight the integrand has, so
32 nodes give about 12 significant figures. Transmission follows from Kemble's
form `D = 1/(1 + exp(2G))`, which continues smoothly above the barrier.

WKB is roughly 6× cheaper than an exact solve, and wrong by a factor of order
unity: about 7% in current density, and up to 60% pointwise.

**`NeuralSolver`** is a trained network standing in for the exact solve,
`method="ml"`. See *Learned solver* below for when it is worth it and how to
train your own.

### `transmission_solutions`

The results you can write down, as opposed to integrate:

| Class | What it is |
|---|---|
| `WKB` | semiclassical Gamow exponent for the Schottky-Nordheim barrier, with the Kemble form. Cheap; accurate to a factor of order unity deep in the tunnelling regime. |
| `AiryTriangular` | the **exact** solution for a pure triangular barrier, in Airy functions. Not the barrier a real metal presents, but the one case with a closed form — so it is the natural check on a numerical solver. It agrees with Noumerov to 6e-5 across twenty orders of magnitude. |
| `calculate_gamow_numeric` | the same semiclassical physics for a barrier of *any* shape, with the turning points found numerically. Used as the reference the learned solvers correct. |

#### Generic semiclassical reference (in `transmission_solutions`)

A semiclassical reference for **arbitrary** barrier shapes. The analytic `WKB`
solver is specialised to the planar barrier, where the turning points are roots
of a quadratic; for a curved tip there is no closed form, so the turning points
are located numerically and the Gamow integral is taken by quadrature.

It exists to be a *reference function* for residual learning, and what matters
for that job is not accuracy — the learned correction fixes whatever it gets
wrong — but that it be cheap and smooth. It captures the exponential, which is
the whole point.

Vectorised over energy: the potential is scanned once, and because the barrier
is unimodal both turning points come from one interpolation per branch rather
than a search per energy.

#### Learned solver: `NeuralSolver` (in `transmission_solver`)

A trained neural network standing in for the exact solve: `method="ml"`.

**When it is worth it.** Not, in general, for the plain planar barrier. There
`Noumerov.fast()` is already cheap, and the network gains a small factor over it
at the price of a small, measured error. The case for a network is
**dimensional**: its cost does not grow with the number of barrier parameters,
while the cost of the alternatives does. A table at fixed per-axis resolution
multiplies in size with every parameter. The sharp-tip barrier adds tip radius
and field enhancement, making the problem five-dimensional, where a table would
need of order 10⁹ nodes — tens of gigabytes. The network is under 100 kB. Further
parameters — effective masses, a coating, a different image model — extend the
same argument.

Two models ship, in `getelec/data/`:

| barrier | inputs | file | role |
|---|---|---|---|
| `SchottkyPotential` | h, W, F | `neural_schottky.npz` | the running example of the workflow |
| `SmallRadiiPotential` | h, W, F, R, γ | `neural_small_radii.npz` | where a network pays off |

The solver picks the model by the barrier's exact class, so a subclass of a
barrier — which may change the potential — never silently gets its parent's
model. A barrier with no model, or outside a model's trained domain, is
handed to the exact solver (`fallback="exact"`, the default), to WKB
(`"wkb"`), or rejected (`"error"`) — never extrapolated.

**What makes it work is the target, not the architecture.** The model never sees
`D`. It learns

```
R = ln D_noumerov − ln D_kemble
```

against a semiclassical reference for the *same* barrier: the closed-form Kemble
expression for the planar barrier, and the numerically evaluated Gamow exponent
(`calculate_gamow_numeric`, above) for the sharp tip. Across the domain `ln D` runs to
several hundred while `R` stays of order unity. Using the planar reference for a
curved barrier instead — the obvious shortcut — leaves a residual spanning more
than a thousand e-folds, which is not learnable at any reasonable cost. **The
reference has to follow the barrier it is referencing.** This also makes the
error budget explicit: relative error in `D` is exactly absolute error in `R`,
so 1% means predicting `R` to 0.01, and none of the hundreds of e-folds has to be
learned at all.

Inputs are physics-derived, not raw parameters: the scaled barrier field
`f = k_e F / h²` equals 1 exactly at the barrier top wherever that top happens to
be, so it hands the network the coordinate in which the function is smooth.
Adding these features cut worst-case error by roughly threefold at fixed
architecture.

Measured against the exact Noumerov solution, over each model's domain:

| | Schottky model | sharp-tip model |
|---|---|---|
| transmission, median | 0.11% | 0.14% |
| transmission, 99th percentile | 0.8% | 0.8% |
| current density, worst of the test cases | 0.85% | 0.5% |
| cost against `Noumerov.fast()` | 4–8× faster | about 2× faster |

The pointwise tail sits where `D` is tens of e-folds below its peak, and for the
Schottky model at the low-field edge of the domain (up to 3.8% at
`W = 23.5 eV, F = 1.5 V/nm`). Neither moves integrals much, which is why the
current density comes out tighter than the pointwise numbers suggest.

**On speed.** Both the reference and the forward pass cost time proportional to
the number of energies, so the solver evaluates ~48 energies and splines `ln D`
onto the rest — the same trick `Noumerov.fast()` uses, changing the answer by
well under 1%. For a given grid that spline is a fixed linear map from the node
values, so it is built once, cached, and each call is a single matrix–vector
product; the forward pass runs in single precision. Together these change
`ln D` by under 1e-6 and make the cost nearly flat in grid size. Against
`Noumerov.fast()` the Schottky model is then 4–8× faster and the sharp-tip
model about 2× (1.8–3.3× measured), because its numerical reference is now
most of its cost and is fixed by the training. On grids where `Noumerov.fast()` has to
solve every energy the margin grows, up to ~25× at 3000 energies. A
whole `current_density(method="ml")` is about 3× faster than `fast=True`: the
rest of the emitter now dominates. For the sharp tip the point is less the
speed than that five dimensions are covered at all.

**Train your own.** The shipped models are examples of a workflow, trained on
one domain. For your own conditions — a different range, a different barrier —
retrain with `getelec.training`. Training needs scikit-learn
(`pip install scikit-learn`); using a model needs only NumPy.

```bash
python -m getelec.training                          # reproduces neural_schottky.npz
python -m getelec.training --barrier small_radii    # reproduces neural_small_radii.npz
```

For a new barrier, supply a function that builds it from `(W, F, *parameters)`,
the domain, and a feature function; the `getelec.training` docstring has the
recipe. Training samples 10% beyond the domain on every axis, because a fitted
function is least accurate at the edges of its data, and certifies only the
domain itself. Read the validation error it prints: that, not the training loss,
is the number to trust.

Models are saved as plain `.npz` arrays — weights, biases, and the names of the
features, reference and barrier class — and load with `allow_pickle=False`. No
scikit-learn at run time, and no coupling to the version that trained them.

**Validity of the sharp-tip model.** The shipped sharp-tip model covers the
radii over which `SmallRadiiPotential` is valid, 20–1000 nm (trained uniformly
in the logarithm of `R`, with `1/R` as its input), and `γ` = 1–200. Outside
that range the calculation is handed to the exact solver, and the barrier class
warns about the radius as well (see its accuracy under `potential_barrier`).

### `electron_emitter`

Assembles the pieces and produces physical outputs.

| Method | Returns |
|---|---|
| `calculate_current_density()` | J, A/cm² |
| `calculate_nottingham_heat()` | P_N, W/cm² |
| `calculate_transmission_coefficient()` | E, D(E) |
| `calculate_supply_function()` | E, N(E) |
| `calculate_psi()` | E, x, ψ(E,x) — needs a full `Noumerov` solver |
| `calculate_probability_current()` | E, x, j(E,x) |
| `calculate_total_energy_distribution()` | E, TED = f(E)·∫D dE_z |
| `calculate_normal_energy_distribution()` | E_z, NED = l(E_z)·D(E_z) for a metal |
| `update_params(**kwargs)` | sets a parameter everywhere it appears |

Semiconductor emitters return both bands from every accessor, each on its own
energy and spatial grid — six arrays from `calculate_psi` and
`calculate_probability_current`, four from the rest. For how the two
distributions are built in a band, and why the conduction NED can extend into
the gap, see [Semiconductors: the two energy distributions](#semiconductors-the-two-energy-distributions).

The expensive step — energy grid, supply, transmission — is computed once and
memoised against a fingerprint of every parameter it depends on. Asking for
Nottingham heat after the current density does not redo the tunnelling
calculation. A fingerprint rather than a dirty flag means the cache stays correct
even if you mutate a sub-object directly.

`update_params` sets a parameter on **every** component that has it, and raises
on names that match nothing. `fermi_level` lives on the barrier, the supply
function *and* the semiconductor band structure; setting it in one place only is
an easy way to get quietly inconsistent physics.

### `_kernels`

The Noumerov inner loop, kept separate so the physics modules stay readable.
Compiled with Numba where available, with an identical NumPy fallback otherwise.
Three structural choices carry the performance: only the last two grid points are
kept rather than the whole wavefunction; the grid loop runs outside and the
energy loop inside, so the innermost loop vectorises; and the coefficient
`T[i,j]` is recomputed as a fused multiply-add rather than read back from a large
array. Together these are worth about 30× over the straightforward version.
`run_noumerov_integration`, the kernel of `NoumerovReference`, has none of this:
the plain recurrence for one energy, keeping the whole wavefunction, serial and
compiled without `fastmath`.

---

## Semiconductors: the two energy distributions

The current from a band is a double integral over total energy `E` and normal
energy `E_z`, on the wedge `lower(E) ≤ E_z ≤ E`. The lower limit is Eq. (8) of
[Barranco Cárceles *et al.*, J. Appl. Phys. **138**, 155705 (2025)](https://doi.org/10.1063/5.0284808):

```
conduction:  lower(E) = E_C + (1 − m_e*/m)(E − E_C)
valence:     lower(E) = E − (m_h*/m)(E_V − E)
```

Doing the `E_z` integral first gives the total energy distribution; doing the
`E` integral first gives the normal one:

- **TED(E)** = `f(E) · ∫ D(E_z) dE_z` over `[lower(E), E]`.
- **NED(E_z)** = `D(E_z) · ∫ f(E) dE` over every total energy `E` whose window
  contains `E_z`, written as a difference of log supplies `l`.

`calculate_transmission_coefficient()` returns `D(E)` itself, the
tunnelling probability, which always lies in `[0, 1]` — in a semiconductor too,
where the effective masses enter through the window limits above, not through
the transmission.

A consequence worth knowing before it looks like a bug: changing an effective
mass leaves `D(E)` **completely unchanged** while the TED and the NED both move.
That is the model. `D` is the tunnelling probability of the vacuum barrier,
which the masses do not enter — the published 2.0 code computes it the same way,
with no mass anywhere in its barrier. Giving `D` a mass dependence of its own
means BenDaniel–Duke interface matching, which changes its prefactor; that is an
open item and is not implemented here.

### Seeing the mass dependence

Because `D(E)` does not move when the masses do, there is nothing to look at
that shows how the masses reach the transmission. The smallest object that does
is the inner integral of the TED on its own:

```
g(E) = ∫ D(E_z) dE_z   from lower(E) to E          [eV]
```

`calculate_window_integrated_transmission()` returns it for both bands, and the
interface plots it as **`g(E)` (D over the mass window)**, a calculation offered
only for semiconductors — a metal has no such window.

Read it as *the transmission, weighted by how much of the band can use it*. The
masses set the width of the window: a heavier mass opens it wider, `g` rises,
and the current rises with it. That is the direction the physics moves in, which
is why `g` is the quantity plotted rather than any of the alternatives.

Three things to keep straight:

- **`g` is not a probability.** It is a transmission integrated over an energy
  range, so it carries units of eV and is not bounded by 1. `D(E)` remains the
  probability, in `[0, 1]`.
- **It is not the per-electron transmission.** The mean transmission across the
  window, `g/(width)`, *falls* as the mass grows, because the window opens
  downwards towards smaller `D`. Both statements are true and they point in
  opposite directions; `g` is the one that combines with the occupancy to give
  the current, so `g` is the one plotted.
- **It is not `g′(E)`**, the integrand `D(E) − ᾱ D(lower(E))` left by
  integrating the current by parts. That function is negative throughout the
  valence band for *every* hole mass — at the band edge it is exactly
  `−(m_h*/m) D(E_V)`, since the window closes there — and above 1 in the
  conduction band. It is a derivative, not a transmission, and GETELEC does not
  return it anywhere.

`g` is read out of the same calculation the TED is built from, not computed
alongside it, so `TED(E) = f(E) · g(E)` holds exactly and the plotted curve
cannot drift from the current. The current itself is unaffected by any of this:
it is `∫TED dE` as before.

### Why the conduction NED can sit in the band gap

For `m_e* > m` the conduction-band NED extends **below** `E_C`, into the gap.
This is correct, not a bug.

`E_z` is the normal energy **in vacuum**, `E − ħ²k_∥²/2m`, because that is what
the transmission depends on. The parallel momentum `k_∥` is conserved across
the surface, but the mass changes. Inside the band the transverse kinetic
energy `ħ²k_∥²/2m_e*` is at most `E − E_C`. In vacuum the same `k_∥` carries up
to `(m_e*/m)(E − E_C)`. When `m_e* > m` that exceeds the whole kinetic energy
the electron has in the band, and `E_z` falls below `E_C`.

No electron is in the gap: every electron still has `E ≥ E_C`. `E_z` is not the
energy of a state. It measures how much energy goes into crossing the barrier,
and a heavy electron with a large `k_∥` sees a taller effective barrier.

What happens at the band edge depends on the mass:

| `m_e*` | TED at `E_C` | NED at `E_C` |
|---|---|---|
| `< m` | 0 | 0 — the lower limit never goes below `E_C` |
| `= m` | 0 | steps up to its full value, as the free-electron supply function does at a band bottom |
| `> m` | 0 | continuous through `E_C`, decaying into the gap |

The TED always vanishes at `E_C` because the window width `(m_e*/m)(E − E_C)`
closes. The NED at `E_C` counts every electron whose window contains `E_C`, and
for `m_e* ≥ m` that is all of them.

The tail carries real current. For `E_g = 0.7 eV` with `E_F` mid-gap,
`φ = 4.85 eV`, `F = 5 V/nm` and `T = 300 K`, it is 14% of the conduction current
at `m_e* = 1.2` and 33% at `m_e* = 1.64`. Cutting the NED off at `E_C` would
break `∫NED dE_z = J`. That is why `calculate_normal_energy_distribution()`
returns the NED on its own `E_z` grid, which reaches below the total-energy
grid.

The valence band has no counterpart: its window lies entirely below `E`, which
is below `E_V`.

### How the distributions are checked

Both are rebuilt in the tests by brute-force quadrature — the window limits
written out from the kinematics, `f` integrated numerically, only `D` shared —
and agree point by point to a few parts in 10⁵ for both bands across effective
masses from 0.3 to 2.5. `∫NED dE_z = ∫TED dE = J` to ~0.1%, including for
`m_e*` close to `m`, where the NED changes over `|1 − m*/m| k_B T` at the edge
and the grid is refined there automatically. The below-gap share of the NED is
also checked against an independent count in k-space.

The semiconductor grid step defaults to 0.002 eV, five times finer than a
metal's. A band grid ends at its band edge, where the TED has a finite slope,
so the trapezoid error in the current is `~(h / k_B T)² / 12` — 1.1% at 0.01 eV
and 300 K, 0.05% at 0.002 eV. A metal's grid ends in tails where the TED is
flat, which is why 0.01 suffices there.

---

## The graphical interface

```bash
python gui.py
```

Three tabs.

**Calculate** runs a single calculation and plots it. Pick a material (metal or
semiconductor), a calculation, and a solver; the parameter fields change to
match. Available calculations:

| | |
|---|---|
| `I-F`, `I-T` | current density vs field or temperature |
| `Nottingham vs field` | Nottingham heat P_N; negative when the tip heats |
| `TED`, `NED` | total and normal energy distributions |
| `D(E)` | transmission probability |
| `N(E)` | supply function |
| `g(E)` | D integrated over the effective-mass window — **semiconductors only** |

Semiconductor results are plotted as two curves, conduction and valence,
rather than summed: which band carries the current is usually the point.

`g(E)` appears in the calculation list only when the material is
*Semiconductor*, and it is there to answer a question the other curves cannot:
the effective-mass boxes change the current but leave `D(E)` untouched, and `g`
is what they act on. See [Seeing the mass dependence](#seeing-the-mass-dependence)
for what it is and what it is not.

For a semiconductor, the energy range **selects** which part of the result to
plot rather than where it is evaluated. The two band grids are fixed by the
band structure — the band edges, and the window limits of Eq. (8) — because the
distributions have to be computed on them; the range clips the curves to what
you asked for, and the step box does not apply.

Calculations run on a worker thread with a progress bar, so the window stays
responsive during a long sweep. A log y-axis cannot draw values at or below
zero — a distribution underflows to zero in its tails — so those points simply
do not appear. If no value is positive, as for the Nottingham heat, the axis
falls back to linear instead of showing an empty plot.

**Fit** fits experimental data. **Metals only**: a semiconductor emitter carries
the band gap, the valence band edge and two effective masses on top of the metal
parameters, and an I-V curve does not contain enough independent information to
determine them — the fit would converge to wherever it started.

It loads `.txt`, `.csv` or Excel files (two columns, no header) and fits I-V,
I-T or TED data. Each parameter has a **fit** box: ticked, it is fitted
from the value shown; unticked, it is held at that value. The defaults free only
what the data can determine — the field factor and emission area for an I-V
curve, the field, temperature and analyser offset for a TED — and hold the work
function, Fermi level and temperature at typed values. The plot lists every
parameter with its standard error, the misfit, and any pair of free parameters
correlated beyond ±0.95: such a pair is fixed by the data only in combination,
so their individual errors say little. The TED fit uses the full calculated
distribution, not Young's formula with a constant transverse energy d: the
apparent d changes with energy, so a d fitted over some range is off by more
than 10% (Barranco Cárceles *et al.*, IVNC 2026,
doi:10.1109/IVNC69421.2026.11660967).

**Documentation** gives the authors, version and contact, and opens these pages
in your web browser.

The physics behind every calculation lives in `gui_backend.py`, which imports no
Qt, so it can be tested without a display and scripted directly.

### Standalone executable

From a checkout with the `[dev]` install:

```bash
python compile.py              # the application, in app/dist/, then a test of it
python compile.py --onefile    # a single file instead (slower to start)
python compile.py --console    # keep a console window, to see tracebacks
```

Everything the build produces goes into `app/`, which git ignores. The folder
build is the one to ship: a single file unpacks itself on every launch, so it
starts more slowly and recompiles the solver each time. Once built, the
application runs its own self-test (`--self-test`): the solvers against pinned
numbers, both trained networks, the data-file readers, a fit, saving a figure
as SVG and PNG, the window and this documentation. A build that fails it is reported as failed, since a
missing file in a frozen application otherwise shows up only when a user
reaches the feature that needs it. Run `docs/regenerate.py` first if the guide
or a docstring has changed: the documentation is bundled as it stands.

PyInstaller cannot cross-compile: build on the platform you are targeting.
Built applications are published as downloads on the repository's
[GitHub Releases](https://github.com/sbcarceles13/GETELEC/releases) page, one per version and platform -- never committed to the
repository, where a binary of hundreds of MB would stay in the history for good.

---

## Choosing a solver

```python
getelec.current_density(field=5.0)                 # Noumerov, the default
getelec.current_density(field=5.0, fast=True)      # Noumerov.fast()
getelec.current_density(field=5.0, reference=True) # NoumerovReference
getelec.current_density(field=5.0, method="ml")    # trained network
```

| Situation | Use |
|---|---|
| A published number, or an unusual barrier | `"noumerov"` (the default) |
| Checking another solver's result, or debugging one | `reference=True` (slow) |
| Sweeps, fitting, large energy grids | `fast=True`, or `"ml"` |
| Sharp tips, R = 20–1000 nm, repeated | `"ml"` |
| Sharp tips below 20 nm | the current then depends on the tip's shape beyond its apex, which the barrier does not describe; see its accuracy under `potential_barrier` |
| Sharp tips above 1000 nm | effectively planar for the current: `SchottkyPotential` |
| A barrier with several parameters of your own | train a model with `getelec.training`, then `"ml"` |
| Quick exploration | `"wkb"` |
| A triangular barrier (`barrier="triangular"`), or an exact check on one | `"airy"` (`AiryTriangular`) |

---

## Accuracy and how to check it

Nothing here asks you to take the accuracy on trust.

**Grid convergence.** `h = 1e-3` is a default, not a guarantee:

```python
solver = Noumerov(h=1e-3)
solver.calculate_convergence_report(barrier, energies)
# {0.001: 4.8e-05, 0.0005: 1.3e-05, 0.00025: 0.0}
```

**Solver agreement.** Compare against the exact solve directly:

```python
exact = getelec.current_density(field=5.0, method="noumerov")
learned = getelec.current_density(field=5.0, method="ml")
print(abs(learned - exact) / exact)   # a few tenths of a percent
```

**The test suite.** `pytest` runs about 210 checks in a few minutes. The load-bearing
one reimplements the original dense-array Noumerov independently in plain NumPy
and diffs it against the optimised kernel, so an optimisation that changed the
physics would fail immediately. Others pin the shipped models' accuracy on a
deterministic sweep of their domain, check that current density is right in both
the field and thermionic regimes, and verify the `E_F + φ` invariance
bit-for-bit.

---

## Units

| Quantity | Unit |
|---|---|
| Energy, work function, Fermi level, band gap | eV |
| Distance, tip radius, grid spacing | nm |
| Electric field | V/nm |
| Temperature | K |
| Current density | A/cm² |
| Nottingham heat P_N | W/cm² |
| TED, NED | A/(eV·cm²) |

Effective masses are relative to the free electron mass. `gamma` is
dimensionless. The supply function is returned in the internal
per-nm²-per-fs-per-eV convention that the current integral uses, so scaling it
to A/(eV·cm²) requires the same prefactor the emitter applies.

---

## Citing GETELEC

If you use GETELEC, please cite the software,

- S. Barranco Cárceles, A. Kyritsakis and A. Ayari, *GETELEC: General Tool for
  Electron Emission Calculations*, Zenodo,
  <https://doi.org/10.5281/zenodo.23093209>

and the papers:

- A. Kyritsakis and F. Djurabekova, Comput. Mater. Sci. **128**, 15 (2017),
  <https://doi.org/10.1016/j.commatsci.2016.11.010>
- S. Barranco Cárceles, V. Zadin, A. Mavalankar, I. Underwood and A. Kyritsakis,
  J. Appl. Phys. **138**, 155705 (2025), <https://doi.org/10.1063/5.0284808>
- S. Barranco Cárceles, A. Kyritsakis and A. Ayari, arXiv:2610.07013 (2026),
  <https://doi.org/10.48550/arXiv.2610.07013>

The software DOI covers every version and resolves to the latest one;
`CITATION.cff` holds the same in machine-readable form.
