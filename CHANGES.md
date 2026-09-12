# Changes

## 3.1.0

Changes since 3.0.0, the version previously on the GitHub `main` branch.
Comparisons with 3.0.0 below were run against that code on the same machine.

Reference values, pinned by the test suite (metal, φ = 4.5 eV, E_F = 7.5 eV,
T = 300 K, F = 5 V/nm): J = 4.087404e+05 A/cm², P_N = −8.4180e+04 W/cm².

### Highlights

**Speed.** The Noumerov solver was rewritten: Numba kernels batched over
energies and barriers, keeping only the last two grid points of the
wavefunction. The integration domain is set from the barrier itself — 3 nm of
vacuum past the outer turning point and 0.01 nm into the metal, both fixed by a
convergence study rather than assumed — which is about a fifth of the grid the
first version integrated. A 20-point current-density sweep takes 0.06 s.
`Noumerov.fast()` is a preset for sweeps and fitting, about twice as cheap again
on a whole current density; against a converged reference its current density is
within 5e-5, its energy distributions within 2e-3 and its pointwise transmission
within 7e-4, where the defaults reach 2e-5, 2e-4 and 2e-4.

**Semiconductor energy distributions.** The total and normal energy
distributions are now built from the same double integral over total energy E
and normal energy E_z. 

**A learned solver.** `NeuralSolver` (`method="ml"`) is a small network trained
on the residual against a semiclassical reference. Models for the planar and
sharp-tip barriers ship inside the package as plain `.npz` files: about 0.1%
median error in D(E) and under 1% in current density. Outside its trained
domain it falls back to the exact solver. `getelec.training` trains a model for
another barrier (training also needs scikit-learn).

**More components.** `TriangularPotential`, the triangular barrier with no
image charge, whose transmission `AiryTriangular` (`method="airy"`) gives
exactly; `SmartSemiconductor` and `CustomSemiconductor`, the semiconductor
counterparts of `SmartMetal` and `CustomMetal`; and `DensityOfStatesMetal`,
which reads a tabulated density of states and weights the emission by how many
states each energy holds relative to a free electron gas, normalised at the
Fermi level. Both energy distributions are built from the same double integral,
so they still integrate to the current density. `examples/dos.txt` and section
14 of the example notebook show it end to end. `NoumerovReference`
(`reference=True`) is the Noumerov integration written out plainly, one energy
at a time, to check the fast solvers against; for the planar barrier it agrees
with `Noumerov` to 3e-5 in current density and Nottingham heat.

**Your own barrier.** `Customised` takes a potential of your own, as a function
`V(x)` or a table, and passes it to the same solvers and emitters as the shipped
barriers: `getelec.current_density(field=..., barrier=Customised(...))`. Its
parameters named `fermi_level`, `work_function`, `electric_field` and
`temperature` follow the emitter, so sweeps and `update_params` move it too.
[GUIDE.md](GUIDE.md#your-own-barrier-customised) gives the conventions, and
section 15 of the example notebook works through a function and a table.

**Documentation and packaging.** Installable with pip, with the model weights
included. A one-line API (`getelec.current_density`, `nottingham_heat`,
`transmission_coefficient`, `supply_function`). [GUIDE.md](GUIDE.md) describes
the physics and the code structure. A single tutorial,
`examples/intro_to_getelec.ipynb`, replaces the example scripts and is run cell
by cell by the test suite.

**The application.** The graphical interface ships as a Windows application
that needs no Python, attached to each release on
[GitHub Releases](https://github.com/sbcarceles13/GETELEC/releases) as
`GETELEC-windows.zip`; [INSTALL.md](INSTALL.md#the-application-windows) says
how to run it.

### Fixes that change results

- **The triangular barrier at the surface.** `TriangularPotential` jumps by
  `E_F + φ` at x = 0, and Noumerov resolves a jump only to first order in its
  step — by an amount set by where the surface falls between two grid points,
  which nothing controlled. The grid now places the surface halfway between two
  nodes, the placement at which that term cancels: against the exact Airy
  solution the current density is within 1e-5, where a node landing on the
  surface costs 1.3%. A continuous barrier moves by about 1e-5 either way, so
  the planar and sharp-tip results are unaffected beyond their own
  discretisation. The pinned reference values above moved by up to 2e-5 for
  this reason and the shorter domain together.
- **Thermionic emission on the default grid.** The default grid (`SmartMetal`)
  stops where the supply falls below `supply_threshold`, which at the old
  default of 1e-14 is 32.2 k_B T above the Fermi level. Thermionic electrons
  leave from the top of the barrier, so once the top lies more than about
  25 k_B T above the Fermi level the grid stopped below it and the current was
  underestimated: by 7% for φ = 4.5 eV at 0.5 V/nm and 1500 K, up to 100% at
  the smallest currents counted, and the largest current affected was a few
  mA/cm². The default threshold is now 5e-24 (53.7 k_B T), which holds the
  current density within 0.5% of a far wider grid wherever it reaches
  1e-12 A/cm². Field emission is unchanged, including the reference values
  above; the grid is 1.07× larger at 5 V/nm and 300 K, and up to 1.5× larger in
  the thermionic regime.
- **High fields.** The potential was set to zero wherever it was negative
  within 1 nm of the surface. Above ~12 V/nm the outer turning point lies
  inside 1 nm, so this flattened the far side of the barrier: the transmission
  was off by 22% at 13 V/nm and 360% at 20 V/nm against an independent shooting
  solver. The cutoff is now the zero crossing of the potential
  (`potential_barrier.get_image_cutoff`). Total current density changes by about 1%
  at these fields.

### Licence

From 3.1.0, GETELEC is distributed under the MIT licence
([LICENSE.md](LICENSE.md)). 3.0.0 keeps the licence it was released with.

### Updating code written for 3.0.0

- Install with `pip install getelec`, or `pip install -e ".[dev]"` from a clone.
  `joblib` is no longer needed.
- `transmission_solver.WBK` is now `transmission_solutions.WKB`, also available
  as `method="wkb"`.
- `transmission_solver.ML`, the decision-tree solver, is replaced by
  `NeuralSolver` (`method="ml"`). `machine_memory/` is gone.
- `band_structure.Test` is removed.
- `SmallRadiiPotential` defaults to a 20 nm radius instead of 5 nm, and warns
  outside 20–1000 nm, the range over which it is valid. Below 20 nm the current
  density depends on the shape of the tip beyond its apex, which the barrier's
  radius and `gamma` do not describe; the class docstring and the guide give the
  size of the difference. Above 1000 nm the tip is planar for the current, and
  `SchottkyPotential` gives the same result.
- `generate_band_structure()` no longer takes `above_barrier`.
- `Noumerov` keeps its arguments and adds `seed`, `energy_nodes`, `auto_domain`
  and `interpolation_tolerance`. The default boundary condition is now a WKB
  seed; `seed="plane"` selects the plane-wave one used in 3.0.0.
- `update_params()` raises `KeyError` for a parameter that no component has,
  instead of ignoring it.
- The supply object no longer selects the distribution. The total energy
  distribution is always built from the occupancy f and the normal one from the
  supply function l, whether `FermiDirac` or `LogFermiDirac` is attached, and
  both give identical results. Swapping `emitter.supply` between them, as the
  3.0.0 example did to get one distribution or the other, is no longer needed;
  it still gives the right answer.
- `tests.py` is replaced by `pytest`, and the example scripts by
  `examples/intro_to_getelec.ipynb`.
