# GETELEC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23093209.svg)](https://doi.org/10.5281/zenodo.23093209)
[![Download the app for Windows](https://img.shields.io/badge/download-app%20for%20Windows-2ea44f)](https://github.com/sbcarceles13/GETELEC/releases/latest/download/GETELEC-windows.zip)

**General Tool for Electron Emission Calculations** — thermal-field electron
emission current density and Nottingham heat for metallic and semiconducting
emitters.

**The GETELEC application for Windows needs no Python:
[download GETELEC-windows.zip](https://github.com/sbcarceles13/GETELEC/releases/latest/download/GETELEC-windows.zip)**,
extract it and double-click `GETELEC.exe`
([details](https://github.com/sbcarceles13/GETELEC/blob/main/INSTALL.md#the-application-windows)).
From Python:

```python
import getelec

getelec.current_density(field=5.0, work_function=4.5)     # A/cm^2
```

- **[INSTALL.md](https://github.com/sbcarceles13/GETELEC/blob/main/INSTALL.md)** — installation from scratch: Python, VS Code, venv.
- **[GUIDE.md](https://github.com/sbcarceles13/GETELEC/blob/main/GUIDE.md)** — the physics and how the code is organised.
- **[CHANGES.md](https://github.com/sbcarceles13/GETELEC/blob/main/CHANGES.md)** — what changed in this release, and how to update existing code.

If you use GETELEC, please cite the software,

- S. Barranco Cárceles, A. Kyritsakis and A. Ayari, *GETELEC: General Tool for
  Electron Emission Calculations*, Zenodo,
  <https://doi.org/10.5281/zenodo.23093209>

and the papers:

- A. Kyritsakis and F. Djurabekova, Comput. Mater. Sci. **128**, 15 (2017),
  <https://doi.org/10.1016/j.commatsci.2016.11.010>
- S. Barranco Cárceles, V. Zadin, A. Mavalankar, I. Underwood and A. Kyritsakis,
  J. Appl. Phys. **138**, 155705 (2025), <https://doi.org/10.1063/5.0284808>
- S. Barranco Cárceles, A. Kyritsakis and A. Ayari, 39th International Vacuum
  Nanoelectronics Conference (IVNC 2026),
  <https://doi.org/10.1109/IVNC69421.2026.11660987>

The software DOI covers every version and resolves to the latest one.
[CITATION.cff](https://github.com/sbcarceles13/GETELEC/blob/main/CITATION.cff)
holds the same in machine-readable form, which GitHub offers as "Cite this
repository".

Earlier versions, GETELEC 1.0 and 2.0, are archived at
[GETELEC_legacy](https://github.com/sbcarceles13/GETELEC_legacy).

---

## Install

The package alone, to use GETELEC from your own scripts and notebooks:

```bash
pip install getelec
```

With the GUI, the introduction notebook and the tests, from GitHub:

```bash
git clone https://github.com/sbcarceles13/GETELEC.git
cd GETELEC
python -m venv .venv

source .venv/bin/activate          # macOS / Linux
.\.venv\Scripts\Activate.ps1       # Windows PowerShell

pip install -e ".[dev]"      # includes PyQt6 for the GUI
pytest
```

Step-by-step instructions for both, including installing Python and VS Code from
scratch, are in **[INSTALL.md](https://github.com/sbcarceles13/GETELEC/blob/main/INSTALL.md)**.

## Quick start

Any argument accepts an array, and the whole set is solved in one pass:

```python
import numpy as np
import getelec

getelec.current_density(field=5.0)                             # A/cm^2
getelec.current_density(field=np.linspace(3, 8, 50))           # a sweep
getelec.nottingham_heat(field=5.0, temperature=[300, 1500])    # P_N, W/cm^2
```

For distributions, or to reuse one configuration:

```python
emitter = getelec.metal_emitter(work_function=4.5, fermi_level=7.5,
                                temperature=300.0, field=5.0)

j       = emitter.calculate_current_density()
heat    = emitter.calculate_nottingham_heat()
e, ted  = emitter.calculate_total_energy_distribution()
e, ned  = emitter.calculate_normal_energy_distribution()

emitter.update_params(field=6.0, temp=800)    # everything downstream updates
```

### Transmission and supply

The two factors behind the current come out separately — `D` is set by the
barrier, `N` by the occupancy:

```python
energies, D = getelec.transmission_coefficient(field=5.0)
energies, N = getelec.supply_function(field=5.0, temperature=300.0)
```

See section 4 of the [introduction notebook](https://github.com/sbcarceles13/GETELEC/blob/main/examples/intro_to_getelec.ipynb).

### Semiconductors

```python
emitter = getelec.semiconductor_emitter(work_function=4.5, fermi_level=13.0,
                                        band_gap=1.12, top_valence=12.5)
e_cb, ted_cb, e_vb, ted_vb = emitter.calculate_total_energy_distribution()
```

## Choosing a solver

```python
getelec.current_density(field=5.0)                 # Noumerov, the default
getelec.current_density(field=5.0, method="ml")    # trained network
```

| method | error in D | error in J | use for |
|---|---|---|---|
| `"noumerov"` | reference | reference | published numbers, unusual barriers |
| `fast=True` | ~0.2% far tail | ~3e-5 | sweeps and fitting |
| `reference=True` | ~5e-4 | ~3e-5 | checking a result (slow, one energy at a time) |
| `"ml"` | ~0.1% median | <1% | barriers with several parameters (sharp tips) |
| `"wkb"` | up to 63% | ~7% | quick exploration |

`"noumerov"` solves the Schrödinger equation with Noumerov's method, whose local
truncation error is O(h⁶): one of the most accurate methods for this equation,
and fast, with a single three-term recurrence per grid point.

`"ml"` is `NeuralSolver`, a trained network, with models shipped for the planar
and sharp-tip barriers. For a planar barrier it is a worked example rather than
a big win — several times faster than `fast=True`, at a small measured error. The
case for a network is barriers with several parameters, where the cost of the
alternatives multiplies and the network's does not. Outside a model's trained
domain it falls back to the exact solver rather than extrapolating. To train one
for your own barrier and conditions, see `getelec.training`
(training also needs `pip install scikit-learn`), [GUIDE.md](https://github.com/sbcarceles13/GETELEC/blob/main/GUIDE.md#transmission_solver) and section 9 of the
[introduction notebook](https://github.com/sbcarceles13/GETELEC/blob/main/examples/intro_to_getelec.ipynb).

## Full control

The shortcuts assemble four interchangeable components. Build them yourself for
anything the shortcuts do not expose:

An emitter is a supply, a barrier, a band structure and a solver — one module
each:

| Module | Options |
|---|---|
| `potential_barrier` | `SchottkyPotential`, `SmallRadiiPotential`, `TriangularPotential`, `Customised` (your own) |
| `band_structure` | `Metal`, `SmartMetal`, `Semiconductor`, `SmartSemiconductor`, `CustomMetal`, `CustomSemiconductor`, `DensityOfStatesMetal` |
| `electron_supply` | `FermiDirac`, `LogFermiDirac` |
| `transmission_solver` | `Noumerov`, `NoumerovFast`, `NoumerovReference`, `NeuralSolver` (train with `getelec.training`) |
| `transmission_solutions` | `WKB`, `AiryTriangular`, `calculate_gamow_numeric` |
| `electron_emitter` | `MetalEmitter`, `SemiconductorEmitter` |

`transmission_solver` holds the numerical solvers; `transmission_solutions`
holds the results you can write down — the semiclassical WKB form and the exact
Airy solution for a triangular barrier. A private helper, `_kernels`, holds the
Noumerov inner loop.

```python
from getelec.potential_barrier import SchottkyPotential
from getelec.band_structure import SmartMetal
from getelec.transmission_solver import Noumerov
from getelec.electron_supply import LogFermiDirac
from getelec.electron_emitter import MetalEmitter

emitter = MetalEmitter(
    SchottkyPotential(fermi_level=7.5, work_function=4.5, electric_field=5.0),
    Noumerov(h=5e-4),
    LogFermiDirac(fermi_level=7.5, temperature=300.0),
    SmartMetal(energy_resolution=0.01),
)
```

### Your own barrier

`Customised` wraps a potential of your own, a function `V(x)` (nm in, eV out)
or a table, and passes it wherever a barrier name goes:

```python
from getelec.potential_barrier import Customised

def triangle(x, fermi_level, work_function, electric_field):
    return fermi_level + work_function - electric_field * x

getelec.current_density(field=[4.0, 5.0, 6.0], barrier=Customised(triangle))
```

Parameters named `fermi_level`, `work_function`, `electric_field` and
`temperature` are filled in by the emitter, so a sweep moves the barrier too.
The rules (zero of energy, divergences, tables, which solvers read it) are in
[GUIDE.md](https://github.com/sbcarceles13/GETELEC/blob/main/GUIDE.md#your-own-barrier-customised).

## Checking accuracy

`h = 1e-3` is a default, not a guarantee. Verify it for your parameters:

```python
Noumerov(h=1e-3).calculate_convergence_report(barrier, energies)
# {0.001: 4.8e-05, 0.0005: 1.3e-05, 0.00025: 0.0}
```

## Examples

Everything is in one notebook, [`examples/intro_to_getelec.ipynb`](https://github.com/sbcarceles13/GETELEC/blob/main/examples/intro_to_getelec.ipynb): from the
one-line current density through distributions, solvers, semiconductors, sharp
tips and the neural solver, and the wavefunction, to fitting measured I–V and
energy-distribution data (`examples/iv.txt`, `examples/ted.txt`) and what such
a fit can and cannot determine. The test suite runs every cell.


## Units

Energies in eV, distances in nm, field in V/nm, temperature in K. Current
density in A/cm², Nottingham heat P_N in W/cm², distributions in A/(eV·cm²).

## GUI

```bash
python gui.py
```

Calculates I-F, I-T, Nottingham heat, TED, NED, transmission D(E), supply N(E)
for metals and semiconductors, with a choice of
solver. Calculations run on a worker thread, so the window stays responsive.
Fits I-V, I-T and TED data from `.txt`, `.csv` or Excel files, with a
choice of which parameters are free. Fitting is metals-only: a semiconductor
emitter has more free parameters than an I-V curve can constrain.

### Standalone executable

From a checkout with the `[dev]` install:

```bash
python compile.py              # the application, in app/dist/, then a test of it
python compile.py --onefile    # a single file instead (slower to start)
```

The build is tested before it is reported done: the finished application runs
its own self-test (solvers, trained networks, data files, a fit, saved figures,
the window and the documentation) and the build fails if any of it does. PyInstaller cannot
cross-compile, so build on the platform you are targeting.
Built applications are published as downloads on the repository's
[GitHub Releases](https://github.com/sbcarceles13/GETELEC/releases) page, one per version and platform -- never committed to the
repository, where a binary of hundreds of MB would stay in the history for good.

## What changed in 3.1.0

A faster Noumerov solver (a 20-point field sweep in 0.06 s),
semiconductor energy distributions built consistently from the emission
integral, a trained neural solver shipped with the package, more comprehensive
documentation. Code written for 3.0.0 may
need updating; the list is in [CHANGES.md](https://github.com/sbcarceles13/GETELEC/blob/main/CHANGES.md).

## Contributing

See [CONTRIBUTING.md](https://github.com/sbcarceles13/GETELEC/blob/main/CONTRIBUTING.md).

## Acknowledgements

The documentation was written with the help of Claude (Anthropic) and fully
verified by the authors.

## License

MIT. See [LICENSE.md](https://github.com/sbcarceles13/GETELEC/blob/main/LICENSE.md).

The licence does not require a citation; if GETELEC contributes to published
work, please cite it as given at the top of this page.

## Contact

- s [dot] barranco [dot] carceles [at] gmail [dot] com
- anthony [dot] ayari [at] univ-lyon1 [dot] fr
