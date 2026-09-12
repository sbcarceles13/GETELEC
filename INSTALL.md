# Installing GETELEC

## The application (Windows)

To calculate and fit through windows and buttons, no Python is needed:

1. Download **[GETELEC-windows.zip](https://github.com/sbcarceles13/GETELEC/releases/latest/download/GETELEC-windows.zip)**
   (about 155 MB). Your browser may ask whether to keep it: choose **Keep**.
2. Right-click the downloaded file → **Extract All…**, into a folder of your
   own such as Documents. Do not run it from inside the zip.
3. Open the extracted `GETELEC` folder and double-click `GETELEC.exe`.
4. The application is not signed with a certificate, so Windows may show
   "Windows protected your PC". Click **More info**, then **Run anyway**.
5. The first calculation after extracting takes several seconds, while the
   solver is compiled for this computer. After that it takes under a second,
   in later sessions too.

On macOS and Linux, run the same interface from a GitHub install (B below),
with `python gui.py`.

The rest of this page installs GETELEC for use from Python.

## Two ways to install

| | A. With pip | B. From GitHub |
|---|---|---|
| **What you get** | The `getelec` package, to use from your own scripts and notebooks | The package, plus the graphical interface, the introduction notebook, the test suite and the source code |
| **Choose it to** | use GETELEC in your calculations | use the GUI, run the tests, or change the code |
| **Needs Git** | No | Yes, or a ZIP download |

Both need **Python 3.10 or newer**, and both install into a virtual environment
(a private copy of Python for this one project, explained in
[step 4](#4-create-a-virtual-environment)).

## Quick install

If you already have Python and a terminal you are comfortable with:

**A. With pip**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install getelec
```

Add `pip install matplotlib` for plotting. Training your own `NeuralSolver`
models also needs `pip install scikit-learn`; using the models that ship with
GETELEC does not.

**B. From GitHub**

```bash
git clone https://github.com/sbcarceles13/GETELEC.git
cd GETELEC
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Otherwise, follow the walkthrough below.

---

## Full walkthrough

This assumes nothing is installed on your machine. Where the two ways differ, a
step is split into **A. With pip** and **B. From GitHub**: follow the one you
chose.

Total time: about 15 minutes.

## Contents

1. [Install Python](#1-install-python)
2. [Install VS Code](#2-install-vs-code)
3. [Get a folder for GETELEC](#3-get-a-folder-for-getelec)
4. [Create a virtual environment](#4-create-a-virtual-environment)
5. [Activate the environment](#5-activate-the-environment)
6. [Install GETELEC](#6-install-getelec)
7. [Check that it works](#7-check-that-it-works)
8. [Set VS Code to use the environment](#8-set-vs-code-to-use-the-environment)
9. [Run the introduction notebook](#9-run-the-introduction-notebook)
10. [Everyday use](#10-everyday-use)
11. [Troubleshooting](#troubleshooting)

---

## 1. Install Python

GETELEC needs **Python 3.10 or newer**.

First check whether you already have it. Open a terminal
(Windows: press `Win`, type `powershell`, press Enter. macOS: press `Cmd+Space`,
type `terminal`, press Enter. Linux: `Ctrl+Alt+T`) and run:

```bash
python --version
```

If that prints `Python 3.10.x` or higher, skip to step 2. If it prints 3.9 or
lower, or says the command was not found, install Python:

**Windows**
Download the installer from <https://www.python.org/downloads/>. During
installation, **tick "Add python.exe to PATH"** on the first screen. This is easy
to miss and is the single most common cause of "python is not recognised" later.

**macOS**
Download from <https://www.python.org/downloads/>, or if you use Homebrew:

```bash
brew install python@3.12
```

macOS ships with an old system Python. Do not use it or modify it; install your
own as above.

**Linux (Debian/Ubuntu)**

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

The `python3-venv` package is separate on Debian and Ubuntu and is required for
step 4.

> **A note on the command name.** On Windows the command is usually `python`. On
> macOS and Linux it is usually `python3`, and plain `python` may not exist at
> all. Throughout this guide, use whichever works on your system.

---

## 2. Install VS Code

VS Code is a free editor. It is not required — GETELEC is a normal Python
package and works from any terminal — but it makes running the examples and
reading the code much easier.

Download from <https://code.visualstudio.com/> and install it.

Then install the Python extension:

1. Open VS Code.
2. Click the Extensions icon in the left sidebar (four squares), or press
   `Ctrl+Shift+X` (`Cmd+Shift+X` on macOS).
3. Search for **Python** and install the one published by Microsoft. Do the
   same for **Jupyter**, which runs the introduction notebook.

If you install from GitHub, VS Code also offers to install both when you open
the GETELEC folder, because the repository lists them as recommended extensions.

---

## 3. Get a folder for GETELEC

### A. With pip

Make a folder for your GETELEC work and move into it. Any name and place will
do; this guide calls it `getelec-work`:

```bash
mkdir getelec-work
cd getelec-work
```

### B. From GitHub

**Install Git.** Git downloads the code and lets you get updates later. If you
would rather not install it, see the ZIP alternative below.

Check whether you have it:

```bash
git --version
```

If not:

- **Windows:** download from <https://git-scm.com/download/win> and accept the
  defaults.
- **macOS:** run `git --version` and macOS will offer to install the developer
  tools. Accept.
- **Linux:** `sudo apt install git`

**Download GETELEC.** Pick a folder to work in, then:

```bash
git clone https://github.com/sbcarceles13/GETELEC.git
cd GETELEC
```

**Without Git:** go to the GitHub page, click the green **Code** button, choose
**Download ZIP**, unzip it, and `cd` into the unzipped folder.

To confirm you are in the right place, list the files. You should see
`pyproject.toml`, `README.md`, and folders named `getelec` and `examples`:

```bash
# Windows PowerShell
dir
# macOS / Linux
ls
```

---

## 4. Create a virtual environment

A virtual environment is a private copy of Python for this one project. It keeps
GETELEC's dependencies from colliding with anything else on your machine, and it
means you never need administrator rights to install packages.

From inside the folder of step 3 (`getelec-work` or `GETELEC`):

```bash
python -m venv .venv
```

(Use `python3` if that is your command.) This creates a `.venv` folder. It is
disposable — if anything goes wrong you can delete it and run this again.

---

## 5. Activate the environment

Activation tells your terminal to use the project's Python instead of the
system one. **You need to do this every time you open a new terminal.**

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

If that fails with a message about execution policies, run this once and try
again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows Command Prompt (cmd.exe)**

```cmd
.\.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

You will know it worked because your prompt now starts with `(.venv)`:

```
(.venv) C:\Users\you\GETELEC>
```

To leave the environment later, type `deactivate`.

---

## 6. Install GETELEC

With the environment active:

### A. With pip

```bash
pip install getelec
```

This installs the solver and what it needs to run. Two more packages are worth
knowing about, each installed the same way when you need it:

```bash
pip install matplotlib      # plotting, and the introduction notebook
pip install scikit-learn    # training your own NeuralSolver models
```

Using the trained models that ship with GETELEC needs neither. The graphical
interface is not part of the pip package: for that, install from GitHub.

### B. From GitHub

```bash
pip install -e ".[dev]"
```

What this does:

- `-e` installs in *editable* mode, so edits to the source take effect
  immediately without reinstalling.
- `.` means "the package in the current folder".
- `[dev]` adds everything needed to run the notebook, the tests **and the GUI**
  (matplotlib, Jupyter's `ipykernel` and `notebook`, pytest, scikit-learn,
  PyQt6, pandas with its Excel readers, pyinstaller).

For just the solver from the checkout, without any of those, use
`pip install -e .` instead.

The GUI runs from a checkout with the `[dev]` install above (`python gui.py`),
or as the standalone application.

### Either way

The install pulls in NumPy, SciPy and Numba. Numba compiles the inner
loop of the Noumerov solver; if it is unavailable on your platform, GETELEC falls
back to a slower pure-NumPy implementation automatically and still gives
identical results.

---

## 7. Check that it works

Try a calculation:

```bash
python -c "import getelec; print(getelec.current_density(field=5.0))"
```

This should print a number close to `408742` (A/cm²). The first call is slow —
a few seconds — because Numba compiles the solver. Later calls are fast.

**From GitHub**, also run the test suite:

```bash
pytest
```

You should see about 210 tests pass, in a few minutes. If so, you are done.

---

## 8. Set VS Code to use the environment

1. Open your folder in VS Code: **File → Open Folder**, and select
   `getelec-work` (with pip) or `GETELEC` (from GitHub).
2. Look at the bottom-right status bar. If it shows a Python version with
   `.venv`, VS Code has already picked the environment and you can skip the
   rest of this step.
3. Otherwise press `Ctrl+Shift+P` (`Cmd+Shift+P` on macOS), type
   `Python: Select Interpreter`, press Enter, and choose the one whose path
   contains `.venv`.

Without this, VS Code uses the system Python and reports that `getelec` cannot
be imported. Any terminal you open inside VS Code from now on
(**Terminal → New Terminal**) activates the environment for you.

To run a file, open it and press the ▷ button top-right, or `F5` to run it under
the debugger.

---

## 9. Run the introduction notebook

`intro_to_getelec.ipynb` is a guided tour, from the first current density to
fitting measured data. Run its cells in order.

**Getting it.**

- **From GitHub** it is already in the `examples` folder, next to the three
  data files it reads, `iv.txt`, `ted.txt` and `dos.txt`.
- **With pip**, copy those four files into `getelec-work`: on
  <https://github.com/sbcarceles13/GETELEC>, click the green **Code** button,
  choose **Download ZIP**, unzip it, and take `intro_to_getelec.ipynb`,
  `iv.txt`, `ted.txt` and `dos.txt` from its `examples` folder. Then, with the
  environment active, install what the notebook uses:

  ```bash
  pip install matplotlib ipykernel      # add notebook to run it in a browser
  ```

**In VS Code**

1. Open `intro_to_getelec.ipynb`.
2. The first time, click **Select Kernel** (top right) → **Python
   Environments** → the one whose path contains `.venv`. VS Code remembers this
   for the notebook, so you only do it once.
   If `.venv` is not in that list, give VS Code its path by hand: **Select
   Kernel** → **Create Python Environment** → **Enter interpreter path** →
   your project folder (`getelec-work` or `GETELEC`) → `.venv` → `Scripts` →
   `python.exe` (macOS/Linux: `.venv/bin/python`). Scroll down the file list
   to find it; Windows may show it as just `python`, of type Application.
3. Click **Run All**, or run the cells one by one with `Shift+Enter`.

**In a browser**

From a terminal with the environment active (step 5):

```bash
jupyter notebook intro_to_getelec.ipynb             # with pip, from getelec-work
jupyter notebook examples/intro_to_getelec.ipynb    # from GitHub, from GETELEC
```

Started this way, Jupyter uses the environment it was started from, so there
is no kernel to choose.

Either way, the first code cell checks that it can import GETELEC and, if not,
prints which Python it is running on and what to change.

---

## 10. Everyday use

Each time you come back to the project:

```bash
cd path/to/your/folder         # getelec-work or GETELEC

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

python your_script.py
```

To update to a newer version:

**A. With pip**

```bash
pip install --upgrade getelec
```

**B. From GitHub**

```bash
git pull
pip install -e ".[dev]"
```

---

## Troubleshooting

**`python: command not found`, or `'python' is not recognized`**
On macOS and Linux try `python3`. On Windows this almost always means the "Add
python.exe to PATH" box was not ticked during installation. Re-run the Python
installer, choose **Modify**, and enable it — or reinstall and tick the box.

**`No module named venv`**
On Debian/Ubuntu: `sudo apt install python3-venv`.

**PowerShell refuses to run the activation script**
Windows blocks local scripts by default. Run this once:
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**`No matching distribution found for getelec`**
pip found no version of GETELEC for the Python it is running on. With the
environment active, check that `python --version` shows 3.10 or newer, and
update pip with `python -m pip install --upgrade pip`.

**`ModuleNotFoundError: No module named 'getelec'`**
Either the environment is not activated (look for `(.venv)` in your prompt), or
GETELEC was installed into a different environment. With pip, activate the
environment and run `pip install getelec` again. From GitHub, `pip install -e .`
must be run from the folder containing `pyproject.toml`.

**VS Code says `getelec` cannot be resolved, but the terminal works**
VS Code is pointing at a different interpreter. Redo step 8.

**The notebook says `getelec is not installed in the Python running this notebook`**
The notebook is running on a Python other than the project's `.venv`; the
message shows which one. In VS Code, click the kernel name at the top right and
choose the `.venv` environment (step 9). In a browser, close Jupyter, activate
the environment, and start it again from that terminal.

**VS Code does not list `.venv` as a kernel, or asks to install `ipykernel`**
The environment was set up without the notebook packages. With it active, run
`pip install ipykernel` (with pip) or `pip install -e ".[dev]"` again (from
GitHub), then reload VS Code (`Ctrl+Shift+P` → `Developer: Reload Window`).
If `.venv` is still not listed, give its path by hand, as in step 9.

**The notebook cannot find `iv.txt` or `dos.txt`**
The three data files have to sit in the same folder as the notebook. See step 9.

**The first calculation takes several seconds**
That is Numba compiling the solver. It happens once per Python process. The
compiled code is cached on disk, so later runs start faster too.

**A plot window never appears**
You are probably on a headless machine or over SSH. Either save the figure
instead of showing it, or install a GUI backend. To check quickly, run with
`MPLBACKEND=Agg` set — the script will complete without trying to draw.

**Numba fails to install**
GETELEC works without it, just more slowly. Install the other dependencies
first, `pip install numpy scipy`, then GETELEC without its dependencies:
`pip install getelec --no-deps` (with pip) or `pip install -e . --no-deps`
(from GitHub).

---

If none of this helps, open an issue at
<https://github.com/sbcarceles13/GETELEC/issues> and include your operating
system, the output of `python --version`, and the full error message.
Or get in touch with Salvador Barranco Cárceles at
s [dot] barranco [dot] carceles [at] gmail [dot] com, or Anthony Ayari at
anthony [dot] ayari [at] univ-lyon1 [dot] fr.
